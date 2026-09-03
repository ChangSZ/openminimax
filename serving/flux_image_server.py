#!/usr/bin/env python3
"""HTTP shim over a diffusers FLUX.2 text-to-image pipeline — the IMAGE half of the
self-hosted MiniMax gateway (video is h3_turbo_server.py). See README §2.

Runs as its OWN process pinned to ONE otherwise-idle L40S (CUDA_VISIBLE_DEVICES=2),
so it is physically isolated from H3 on cuda:0 — no model swapping, they run in
parallel. The gateway image worker (app.backend.FluxImageBackend) speaks the exact
async protocol below and points at http://127.0.0.1:30020:

  POST /v1/images    body: {prompt, n(1-9), width/height OR aspect_ratio,
                            subject_reference?:[url]}
                     -> {"id": <job>, "status": "queued"}
  GET  /v1/images/{id} -> {"status": "queued|running|completed|failed",
                           "file_paths": ["/mnt/models/outputs/<id>_0.png", ...],
                           "error": ...}
  GET  /health       -> 200 {"status":"ok","ready":bool}

FLUX.2-dev fp8 + Flux2Turbo LoRA, ~8 steps. fp8 weights are RESIDENT on cuda:2 (no
CPU offload — the whole model fits one 48GB card), so an image is seconds. One
generation at a time (a background worker thread drains a FIFO), matching the H3 shim.

Loopback-only bind (127.0.0.1) — never a public port (docs/PLAN.md §7.1); the gateway
reaches it inside the box.

VERIFIED on the box (2026-09-02): FLUX.2 fp8 loads RESIDENT on cuda:2 (~31GB) and
produces pixel-clean 1024x768 images at ~37.5s each. Re-check with gen_flux_smoketest.py
after a rebuild. The model-loading specifics (exact class names / fp8 loader) were
finalized against the real weights during that smoke test.
"""
import io, json, os, time, threading, traceback, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Pin to the idle card BEFORE importing torch, so this process only ever sees cuda:2
# as its cuda:0 — it physically cannot touch H3's card. Override with H3_FLUX_DEVICE.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("H3_FLUX_DEVICE", "2"))

import torch
from PIL import Image

# All paths env-overridable; defaults are the layout that PASSED the smoke test.
# MODEL = the OFFICIAL diffusers dir (transformer/ text_encoder/ vae/ tokenizer/).
# The Comfy fp8 single-files are NOT used: the fp8mixed DiT loses its scale tensors
# through from_single_file (-> noise) and the fp8 text-enc is renamed/vision-stripped.
# Default lives on the EBS volume (/mnt/models) so it SURVIVES stop/start — do NOT put
# it on /opt/dlami/nvme (instance-store, wiped on every stop -> unit fails to load).
MODEL = os.environ.get("FLUX_MODEL", "/mnt/models/flux2-official")
LORA = os.environ.get("FLUX_LORA", "")                            # Turbo LoRA DEFERRED
OUTDIR = os.environ.get("FLUX_OUTDIR", "/mnt/models/outputs")     # worker reads locally
STEPS = int(os.environ.get("FLUX_STEPS", "12"))                   # ~20s warm @1024x768 to
# fit the synchronous /v1/image_generation call under the API Gateway 30s hard limit
# (STEPS=20 measured ~29s — too close). Turbo LoRA deferred; bump FLUX_STEPS for quality
# if you move image gen to an async path.
GUIDANCE = float(os.environ.get("FLUX_GUIDANCE", "3.5"))
# fp8-quantize the DiT at load (fits 48GB resident); "0" => bf16 + sequential offload.
QUANTIZE = os.environ.get("FLUX_QUANTIZE", "1") == "1"
# Long edge (px) when the caller sends a ratio instead of explicit width/height.
DEFAULT_LONG_EDGE = int(os.environ.get("FLUX_LONG_EDGE", "1024"))
PORT = int(os.environ.get("FLUX_PORT", "30020"))

os.makedirs(OUTDIR, exist_ok=True)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] flux {m}", flush=True)


# ---- job store -------------------------------------------------------------
_jobs = {}            # id -> {"status":..,"file_paths":[],"error":..,"req":..}
_queue = []           # FIFO of ids
_lock = threading.Lock()
_pipe = None
_ready = threading.Event()


# ---- pipeline (warm, loaded once) — the SMOKE-TEST-VERIFIED recipe ----------
def build_pipe():
    """Assemble the FLUX.2 pipeline from the OFFICIAL diffusers dir, the exact recipe
    that produced a clean 34GB-peak image in gen_flux_smoketest.py:

      * DiT: bf16 -> torchao fp8 (Float8DynamicActivationFloat8WeightConfig) at load,
        RESIDENT on the pinned card (~32GB). fp8 is what makes it fit 48GB.
      * VAE: official AutoencoderKLFlux2, resident.
      * text encoder: official Mistral3ForConditionalGeneration (bf16, ~48GB) — bigger
        than the card, so OFFLOAD ONLY THIS MODULE via accelerate.cpu_offload. We must
        NOT use whole-pipeline offload: it calls .to(dtype) on the Float8 DiT, which
        torchao doesn't implement (the H3 torchao-offload wall).
    """
    global _pipe
    t0 = time.time()
    dev = "cuda:0"     # after CUDA_VISIBLE_DEVICES, index 0 IS the pinned physical card
    torch.cuda.init()
    from diffusers import (Flux2Pipeline, Flux2Transformer2DModel,
                           AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler,
                           TorchAoConfig)
    from transformers import Mistral3ForConditionalGeneration, AutoProcessor

    tr_dir = os.path.join(MODEL, "transformer")
    vae_dir = os.path.join(MODEL, "vae")
    te_dir = os.path.join(MODEL, "text_encoder")
    tok_dir = os.path.join(MODEL, "tokenizer")

    if QUANTIZE:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        log("loading DiT bf16 -> fp8 (torchao) resident")
        qc = TorchAoConfig(Float8DynamicActivationFloat8WeightConfig())
        transformer = Flux2Transformer2DModel.from_pretrained(
            tr_dir, quantization_config=qc, torch_dtype=torch.bfloat16)
    else:
        log("loading DiT bf16 (no quant) — will sequential-offload to fit")
        transformer = Flux2Transformer2DModel.from_pretrained(
            tr_dir, torch_dtype=torch.bfloat16)
    vae = AutoencoderKLFlux2.from_pretrained(vae_dir, torch_dtype=torch.bfloat16)
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        te_dir, torch_dtype=torch.bfloat16)
    tokenizer = AutoProcessor.from_pretrained(tok_dir)

    pipe = Flux2Pipeline(scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae,
                         text_encoder=text_encoder, tokenizer=tokenizer,
                         transformer=transformer)
    if LORA and os.path.isfile(LORA):
        try:
            pipe.load_lora_weights(LORA)
            log(f"applied Turbo LoRA {os.path.basename(LORA)}")
        except Exception as e:
            log(f"warn: Turbo LoRA not applied ({e}); base steps")

    if QUANTIZE:
        from accelerate import cpu_offload
        pipe.transformer.to(dev)
        pipe.vae.to(dev)
        cpu_offload(pipe.text_encoder, execution_device=dev)   # only the 48GB encoder
    else:
        pipe.enable_sequential_cpu_offload(device=dev)

    _pipe = pipe
    # Pre-warm: the fp8 torchao path JIT-compiles its kernels on the FIRST inference,
    # which takes ~400s (with the GPU near-idle during CPU-side compile). If that happened
    # on a client request it would blow the sync /v1/image_generation budget. So we burn a
    # throwaway inference HERE, before signalling ready — /health stays {"ready":false}
    # until warm inference is actually fast (~20-30s). TimeoutStartSec=0 lets it take as
    # long as it needs. Set FLUX_PREWARM=0 to skip (e.g. for a bf16/no-quant deploy).
    if os.environ.get("FLUX_PREWARM", "1") == "1":
        try:
            w0 = time.time()
            log("pre-warming fp8 kernels (first inference compiles; ~400s, one-time)")
            with torch.inference_mode():
                _pipe(prompt="warmup", width=1024, height=768,
                      num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                      num_images_per_prompt=1,
                      generator=torch.Generator(device=dev).manual_seed(0))
            log(f"pre-warm inference done in {time.time()-w0:.0f}s — serving fast now")
        except Exception as e:
            log(f"warn: pre-warm failed ({e}); first real request will pay the compile")
    _ready.set()
    log(f"pipeline warm in {time.time()-t0:.0f}s; ready on :{PORT} (quant={QUANTIZE})")


def _dims(req):
    """(width, height) as multiples of 16. Explicit width/height win (image-01
    semantics); else derive from aspect_ratio at DEFAULT_LONG_EDGE."""
    w, h = int(req.get("width") or 0), int(req.get("height") or 0)
    if w and h:
        return (max(256, w // 16 * 16), max(256, h // 16 * 16))
    ratio = str(req.get("aspect_ratio") or "16:9")
    try:
        rw, rh = (int(x) for x in ratio.split(":"))
    except Exception:
        rw, rh = 16, 9
    le = DEFAULT_LONG_EDGE
    if rw >= rh:
        width, height = le, round(le * rh / rw / 16) * 16
    else:
        width, height = round(le * rw / rh / 16) * 16, le
    return (max(256, width), max(256, height))


def _fetch_image(uri):
    with urllib.request.urlopen(uri, timeout=30) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def run_job(job_id):
    job = _jobs[job_id]; req = job["req"]
    job["status"] = "running"
    try:
        width, height = _dims(req)
        n = max(1, min(9, int(req.get("n") or 1)))
        # image-01's single subject_reference (a face to keep consistent). FLUX.2
        # can take an image prompt; wire it if present, else pure text-to-image.
        refs = [u for u in (req.get("subject_reference") or []) if u][:1]
        image_ref = _fetch_image(refs[0]) if refs else None
        log(f"job {job_id}: {width}x{height} n={n} steps={STEPS} "
            f"ref={'yes' if image_ref else 'no'}")
        g0 = time.time()
        # Seed from the job id so repeat calls / batches differ (a fixed seed would
        # return the same image every time). job_id is a random hex from do_POST.
        seed = int(job_id[:8], 16)
        kwargs = dict(prompt=req.get("prompt", ""), width=width, height=height,
                      num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                      num_images_per_prompt=n,
                      generator=torch.Generator(device="cuda:0").manual_seed(seed))
        if image_ref is not None:
            kwargs["image"] = image_ref     # subject/reference conditioning
        with torch.inference_mode():
            result = _pipe(**kwargs)
        paths = []
        for i, img in enumerate(result.images):
            out = os.path.join(OUTDIR, f"{job_id}_{i}.png")
            img.save(out)
            paths.append(out)
        job["file_paths"] = paths
        job["status"] = "completed"
        log(f"job {job_id}: completed {len(paths)} img in {time.time()-g0:.1f}s")
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
            time.sleep(0.2); continue
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
        if self.path.startswith("/v1/images/"):
            jid = self.path.rsplit("/", 1)[-1]
            job = _jobs.get(jid)
            if not job:
                return self._send(404, {"error": "no such job"})
            out = {"id": jid, "status": job["status"],
                   "file_paths": job.get("file_paths", [])}
            if job.get("error"):
                out["error"] = job["error"]
            return self._send(200, out)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/images":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"error": f"bad json: {e}"})
        jid = uuid.uuid4().hex[:16]
        _jobs[jid] = {"status": "queued", "file_paths": [], "error": None, "req": req}
        with _lock:
            _queue.append(jid)
        self._send(200, {"id": jid, "status": "queued"})


if __name__ == "__main__":
    threading.Thread(target=worker_loop, daemon=True).start()
    threading.Thread(target=build_pipe, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    log(f"serving on 127.0.0.1:{PORT} (pipeline loading in background)")
    srv.serve_forever()
