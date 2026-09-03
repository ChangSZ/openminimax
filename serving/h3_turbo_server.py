#!/usr/bin/env python3
"""SGLang-protocol HTTP shim over the diffusers MiniMax-H3 Turbo pipeline.

The gateway worker's SGLangBackend already speaks an async /v1/videos protocol and
points at http://127.0.0.1:30010. SGLang can't apply the Turbo LoRA correctly
(gives noise); the diffusers path with `pipe.load_lora_weights` (QKV de-interleave)
does. So we hold ONE warm diffusers pipeline here and expose the exact endpoints the
worker expects, so nothing else in the stack changes:

  POST /v1/videos            body: {prompt, task, target:{aspect_ratio,duration_seconds,short_edge},
                                    num_inference_steps, conditions:[{type,role,uri,frame_index}]}
                             -> {"id": <job>, "status": "queued"}
  GET  /v1/videos/{id}       -> {"status": "queued|running|completed|failed",
                                 "file_paths": ["/root/outputs/<id>.mp4"], "error": ...}
  GET  /health               -> 200 {"status":"ok","ready":bool}

One generation at a time (H3 saturates the box). A background worker thread drains a
FIFO of submitted jobs. Turbo 8-step; keyframes (fl2va/i2va) are fetched from `uri`.
"""
import io, json, os, sys, time, threading, traceback, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from diffusers import ModularPipeline
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
from diffusers.hooks import apply_group_offloading
from diffusers.utils.export_utils import encode_video
from PIL import Image

# All paths are env-overridable; defaults are the layout on the g6e.12xlarge box.
MODEL = os.environ.get("H3_MODEL", "/mnt/models/MiniMax-H3-modular")   # HF *modular* layout
# fl2v (keyframe) Turbo LoRA, 8-step -> num_inference_steps 9. Targets `transformer/`.
LORA = os.environ.get("H3_LORA",
    "/mnt/models/lora/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors")
# ref2v (reference) Turbo LoRA, 4-step -> num_inference_steps 5. Targets `transformer_ref/`.
REF_LORA = os.environ.get("H3_REF_LORA",
    "/mnt/models/lora/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors")
OUTDIR = os.environ.get("H3_OUTDIR", "/mnt/models/outputs")   # worker reads file_paths[0] locally
STEPS = int(os.environ.get("H3_STEPS", "8"))         # 8-step turbo -> num_inference_steps = 9
REF_STEPS = int(os.environ.get("H3_REF_STEPS", "4"))  # 4-step turbo -> num_inference_steps = 5
# Which workflow(s) this process serves: "fl2va" (t2va+keyframe, current prod),
# "ref2va" (reference image→video), or "both" (loads both transformer partitions +
# both LoRAs in one pipeline — heavier host RAM). Default keeps prod behaviour.
WORKFLOW = os.environ.get("H3_WORKFLOW", "fl2va").lower()
# ref2va encodes each reference image into vision tokens at this short edge (H3
# default 2048). On a 46GB L40S, several refs at 2048 + a long-clip latent OOMs the
# DiT activations; 1024 quarters the reference-token count for a modest quality cost.
REF_IMAGE_SHORT_EDGE = int(os.environ.get("H3_REF_IMAGE_SHORT_EDGE", "1024"))
# Offloading the VAEs (vs resident on cuda:0) frees a few GB but makes long-clip VAE
# decode CRAWL (each VAE block streams from CPU per call — a 294f ref2va decode took
# >15min). The real memory hog is reference-image encoding resolution (above), so keep
# the VAE RESIDENT by default and rely on the smaller ref short-edge to fit. Opt-in
# only via H3_VAE_OFFLOAD=1 for an extreme case.
VAE_OFFLOAD = os.environ.get("H3_VAE_OFFLOAD", "0") == "1"
VIDEO_SHIFT, AUDIO_SHIFT = 6.0, 3.0
FPS = 24
PORT = int(os.environ.get("H3_PORT", "30010"))

os.makedirs(OUTDIR, exist_ok=True)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ---- job store -------------------------------------------------------------
_jobs = {}            # id -> {"status":..,"file_paths":[],"error":..,"req":..}
_queue = []           # FIFO of ids
_lock = threading.Lock()
_pipe = None
_ready = threading.Event()

# ---- pipeline (warm, loaded once) -----------------------------------------
def _offload_transformer(tr):
    """bf16 block-level streamed group offload for a 61.7GB transformer partition."""
    off = dict(onload_device=torch.device("cuda:0"),
               offload_device=torch.device("cpu"), use_stream=True)
    tr.requires_grad_(False); tr.eval()
    tr.enable_group_offload(offload_type="block_level",
                            num_blocks_per_group=1, **off)

def build_pipe():
    """Load ONE warm pipeline for the configured H3_WORKFLOW.

      * "fl2va"  — loads `transformer/` + the fl2v (keyframe) Turbo LoRA. Serves
        t2va + keyframe i2va/fl2va. (Current prod, pixel-verified.)
      * "ref2va" — loads `transformer_ref/` + the ref2v Turbo LoRA. Serves reference
        image→video (image not a frame). (Pixel-verified 2026-08-31.)
      * "both"   — loads BOTH partitions and BOTH LoRAs (named adapters) in one
        pipeline; the workflow is picked per call from the request. Heavier host RAM
        (~188GB resident) and the two-adapter combo is the least-battle-tested path.
    """
    global _pipe
    t0 = time.time()
    off = dict(onload_device=torch.device("cuda:0"),
               offload_device=torch.device("cpu"), use_stream=True)

    if WORKFLOW == "ref2va":
        log("loading pipeline (workflow=ref2va: transformer_ref + ref2v Turbo LoRA)")
        pipe = ModularPipeline.from_pretrained(MODEL, workflow="ref2va")
        pipe.load_components(dtype=torch.bfloat16,
                             pretrained_model_name_or_path=MODEL)
        pipe.load_lora_weights(REF_LORA)   # de-interleaves onto transformer_ref
        _offload_transformer(pipe.transformer_ref)
    elif WORKFLOW == "both":
        log("loading pipeline (workflow=both: transformer + transformer_ref, 2 LoRAs)")
        pipe = ModularPipeline.from_pretrained(MODEL)
        pipe.load_components(dtype=torch.bfloat16,            # no workflow= -> both partitions
                             pretrained_model_name_or_path=MODEL)
        # Named adapters so both LoRAs coexist; each targets its own partition.
        pipe.load_lora_weights(LORA, adapter_name="fl2v")
        pipe.load_lora_weights(REF_LORA, adapter_name="ref2v")
        _offload_transformer(pipe.transformer)
        _offload_transformer(pipe.transformer_ref)
    else:  # "fl2va" (default, current prod)
        log("loading pipeline (workflow=fl2va: transformer + fl2v Turbo LoRA)")
        pipe = ModularPipeline.from_pretrained(MODEL)
        pipe.load_components(workflow="fl2va", dtype=torch.bfloat16,
                             pretrained_model_name_or_path=MODEL)
        pipe.load_lora_weights(LORA)       # de-interleaves onto transformer
        _offload_transformer(pipe.transformer)

    pipe.text_encoder.requires_grad_(False)
    pipe.scheduler.set_shift(VIDEO_SHIFT)
    pipe.audio_scheduler.set_shift(AUDIO_SHIFT)
    # Shrink ref-image encoding resolution (ref2va memory hog on long clips).
    try:
        pipe.register_to_config(reference_image_short_edge=REF_IMAGE_SHORT_EDGE)
    except Exception as e:
        log(f"warn: could not set reference_image_short_edge: {e}")
    apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **off)
    vae_offloaded = False
    if VAE_OFFLOAD:
        # Stream VAE blocks on demand instead of pinning them resident — frees GBs
        # for DiT activations on long clips (the 294f ref2va OOM). Fall back to
        # resident if this VAE's module structure doesn't take group offload.
        try:
            apply_group_offloading(pipe.vae, offload_type="leaf_level", **off)
            apply_group_offloading(pipe.audio_vae, offload_type="leaf_level", **off)
            vae_offloaded = True
        except Exception as e:
            log(f"warn: VAE group offload failed ({e}); keeping VAE resident")
    if not vae_offloaded:
        pipe.vae.to("cuda:0"); pipe.audio_vae.to("cuda:0")
    _pipe = pipe
    _ready.set()
    log(f"config: ref_short_edge={REF_IMAGE_SHORT_EDGE} vae_offload={vae_offloaded}")
    log(f"pipeline warm in {time.time()-t0:.0f}s; ready on :{PORT} (workflow={WORKFLOW})")

def _fetch_image(uri):
    with urllib.request.urlopen(uri, timeout=30) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")

def _short_edge_dims(short_edge, ratio):
    # 768p canvas: H3 trained 1344x768 for 16:9. Map short edge + ratio -> multiples of 32.
    se = int(short_edge or 768)
    try:
        w, h = (int(x) for x in str(ratio or "16:9").split(":"))
    except Exception:
        w, h = 16, 9
    if w >= h:   # landscape: short edge is height
        height = se; width = round(se * w / h / 32) * 32
    else:
        width = se; height = round(se * h / w / 32) * 32
    return width, height

def _frames_for(duration_s):
    n = max(5, round(float(duration_s) * FPS))
    f = n + (5 - (n % 17)) % 17             # round up to 17*k+5
    # H3 accepts 120..360 frames (5..15s @ 24fps). If rounding up overshoots the
    # 360 ceiling (e.g. a ~15s shot -> 362), step DOWN one 17-frame chunk so we
    # stay valid instead of failing the whole shot.
    while f > 360:
        f -= 17
    return max(120, f)

def _build_conditioning(req):
    """Turn the request's `conditions` into pipeline kwargs + a resolved step count,
    branching on the task. Returns (kwargs, task, steps_default).

      * keyframe conditions (have `frame_index`) -> `image=`/`last_image=` (fl2va).
      * reference conditions (`role:"reference"`, no frame_index) ->
        `references=[MiniMaxH3ImageReference, ...]` in order (ref2va).
    """
    conds = req.get("conditions") or []
    task = (req.get("task") or "").lower()
    kwargs = {}
    # Keyframes are addressed by frame_index; references are ordered, no frame_index.
    keyframes = {c.get("frame_index"): c.get("uri") for c in conds
                 if c.get("type") == "image" and c.get("uri")
                 and c.get("frame_index") is not None}
    ref_conds = [c for c in conds
                 if c.get("type") == "image" and c.get("uri")
                 and c.get("frame_index") is None]

    if not task:  # infer if the worker didn't tag it
        task = "fl2va" if keyframes else ("ref2va" if ref_conds else "t2va")

    if task == "ref2va":
        refs = sorted(ref_conds, key=lambda c: c.get("index", 0))
        kwargs["references"] = [
            MiniMaxH3ImageReference.from_file(c["uri"]) for c in refs]
        steps_default = REF_STEPS + 1
    else:  # fl2va / t2va
        if 0 in keyframes:  kwargs["image"] = _fetch_image(keyframes[0])
        if -1 in keyframes: kwargs["last_image"] = _fetch_image(keyframes[-1])
        steps_default = STEPS + 1
    return kwargs, task, steps_default

def run_job(job_id):
    job = _jobs[job_id]; req = job["req"]
    job["status"] = "running"
    try:
        target = req.get("target") or {}
        width, height = _short_edge_dims(target.get("short_edge"),
                                         target.get("aspect_ratio"))
        num_frames = _frames_for(target.get("duration_seconds") or 6)
        kwargs, task, steps_default = _build_conditioning(req)
        # The shim owns the step count PER WORKFLOW (fl2v 8-step->9, ref2v 4-step->5),
        # because the two Turbo LoRAs are distilled to different NFE. The worker sends a
        # single generic SGLANG_STEPS (tuned for fl2v=9); honour an incoming hint ONLY
        # when it matches this workflow's own default, else use the workflow default so
        # a ref2va job never runs 9 steps against the 4-step LoRA (slow + mismatched).
        hint = req.get("num_inference_steps")
        steps = int(hint) if hint and int(hint) == steps_default else steps_default
        # In "both" mode, activate the LoRA matching this job's workflow.
        if WORKFLOW == "both":
            _pipe.set_adapters("ref2v" if task == "ref2va" else "fl2v")
        log(f"job {job_id}: task={task} {width}x{height} {num_frames}f steps={steps} "
            f"refs={len(kwargs.get('references', []))} "
            f"kf={[k for k in ('image','last_image') if k in kwargs]}")
        g0 = time.time()
        with torch.inference_mode():
            res = _pipe(prompt=req.get("prompt", ""), height=height, width=width,
                        num_frames=num_frames, num_inference_steps=steps,
                        generator=torch.Generator().manual_seed(42),
                        output_type="np", output=["videos", "audio", "sampling_rate"],
                        **kwargs)
        out = os.path.join(OUTDIR, f"{job_id}.mp4")
        audio = res.get("audio"); a0 = audio[0] if audio is not None else None
        sr = int(res["sampling_rate"]) if res.get("sampling_rate") is not None else None
        encode_video(res["videos"][0], fps=FPS, output_path=out,
                     audio=a0, audio_sample_rate=sr)
        job["file_paths"] = [out]; job["status"] = "completed"
        log(f"job {job_id}: completed in {time.time()-g0:.0f}s -> {out}")
    except Exception as exc:
        job["status"] = "failed"; job["error"] = f"{type(exc).__name__}: {exc}"
        log(f"job {job_id}: FAILED {job['error']}\n{traceback.format_exc()}")

def worker_loop():
    _ready.wait()
    while True:
        jid = None
        with _lock:
            if _queue: jid = _queue.pop(0)
        if jid is None:
            time.sleep(0.5); continue
        run_job(jid)

# ---- HTTP ------------------------------------------------------------------
class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok", "ready": _ready.is_set()})
        if self.path.startswith("/v1/videos/"):
            jid = self.path.rsplit("/", 1)[-1]
            job = _jobs.get(jid)
            if not job: return self._send(404, {"error": "no such job"})
            out = {"id": jid, "status": job["status"],
                   "file_paths": job.get("file_paths", [])}
            if job.get("error"): out["error"] = job["error"]
            return self._send(200, out)
        self._send(404, {"error": "not found"})
    def do_POST(self):
        if self.path.rstrip("/") != "/v1/videos":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"error": f"bad json: {e}"})
        jid = uuid.uuid4().hex[:16]
        _jobs[jid] = {"status": "queued", "file_paths": [], "error": None, "req": req}
        with _lock: _queue.append(jid)
        self._send(200, {"id": jid, "status": "queued"})

if __name__ == "__main__":
    threading.Thread(target=worker_loop, daemon=True).start()
    threading.Thread(target=build_pipe, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    log(f"serving on 127.0.0.1:{PORT} (pipeline loading in background)")
    srv.serve_forever()
