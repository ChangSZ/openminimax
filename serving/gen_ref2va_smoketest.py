#!/usr/bin/env python3
"""ref2va smoke test: MiniMax-H3 REFERENCE-to-video (subject/style ref, NOT a keyframe).

Goal of this run (the go/no-go for the whole ref2va feature):
  (a) prove the reference image does NOT appear as frame 0 — it only steers the
      subject/appearance (unlike fl2va where `image=` is pinned as the first frame);
  (b) prove the ref2v Turbo LoRA (lightx2v `minimax_h3_ref2v_turbo_4step_v0.1`,
      trained against the transformer_ref partition) loads via `load_lora_weights`
      and produces CLEAN pixels — same QKV-de-interleave concern as the fl2v LoRA.

Recipe mirrors gen_turbo_smoketest.py (bf16 + block-level streamed group offload)
but selects the `ref2va` workflow (loads `transformer_ref/`) and passes the image as
a `MiniMaxH3ImageReference` via `references=`, NOT as `image=`.

After it writes the mp4: extract frame 0 AND a mid frame and LOOK at pixels.
  - frame 0 == the reference image  -> STILL keyframe behaviour (wrong, would mean
    ref2va didn't take / we loaded the wrong workflow)
  - frame 0 is a fresh generated frame, subject resembles the ref -> ref2va WORKS.
"""
import os, time, torch
from diffusers import ModularPipeline
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
from diffusers.hooks import apply_group_offloading

MODEL = os.environ.get("H3_MODEL", "/mnt/models/MiniMax-H3-modular")
# ref2v Turbo LoRA (4-step, v0.1) — targets the transformer_ref partition.
LORA = os.environ.get("H3_REF_LORA",
    "/mnt/models/lora/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors")
# A subject reference image. Override with any URL/path of a clear subject.
REF_IMAGE = os.environ.get("H3_REF_IMAGE",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg")
OUT = os.environ.get("H3_OUT", "/mnt/models/outputs/smoketest_ref2va_4step.mp4")
# 4-step Turbo -> num_inference_steps = 5 (off-by-one: grid points = steps + 1).
INFERENCE_STEPS = int(os.environ.get("H3_STEPS", "4"))
VIDEO_SHIFT, AUDIO_SHIFT = 6.0, 3.0
WIDTH, HEIGHT = 1344, 768
NUM_FRAMES = 124   # required for ref2va; ~5.2s @ 24fps (17k+5)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    log("building pipeline (workflow=ref2va -> loads transformer_ref/)")
    pipe = ModularPipeline.from_pretrained(MODEL, workflow="ref2va")
    log("load_components(ref2va) bf16 (weights land in host RAM)")
    pipe.load_components(dtype=torch.bfloat16,
                         pretrained_model_name_or_path=MODEL)

    # Same de-interleave loader that fixed the fl2v noise — the ref2v LoRA is also a
    # non-diffusers PEFT dict against the per-head fused QKV. If it de-interleaves
    # onto transformer_ref cleanly we get sharp pixels; if not, this is where noise
    # would come from (LOOK at the frames, don't trust status).
    log(f"load_lora_weights ref2v: {os.path.basename(LORA)}")
    pipe.load_lora_weights(LORA)
    # ref2va workflow's transformer component is `transformer_ref` (not `transformer`).
    pipe.transformer_ref.requires_grad_(False); pipe.transformer_ref.eval()
    pipe.text_encoder.requires_grad_(False)
    pipe.scheduler.set_shift(VIDEO_SHIFT)
    pipe.audio_scheduler.set_shift(AUDIO_SHIFT)
    log(f"shifts video={pipe.scheduler.shift} audio={pipe.audio_scheduler.shift}")

    off = dict(onload_device=torch.device("cuda:0"),
               offload_device=torch.device("cpu"), use_stream=True)
    pipe.transformer_ref.enable_group_offload(offload_type="block_level",
                                          num_blocks_per_group=1, **off)
    apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **off)
    pipe.vae.to("cuda:0")
    pipe.audio_vae.to("cuda:0")
    log(f"setup done {time.time()-t0:.0f}s; building reference from {REF_IMAGE}")

    subject = MiniMaxH3ImageReference.from_file(REF_IMAGE)

    prompt = ("integrated_multimodal_description: [Shot 1] Live-action, cinematic. The "
        "subject from the reference image walks slowly toward the camera across a sunlit "
        "plaza, turning to look at the lens, gentle breeze, shallow depth of field, "
        "smooth tracking shot.\n\noverall_soundscape: soft outdoor ambience, distant "
        "footsteps.\n\nnon_diegetic_music: a light, warm piano motif.")

    g0 = time.time()
    with torch.inference_mode():
        res = pipe(prompt=prompt, references=[subject],
                   height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
                   num_inference_steps=INFERENCE_STEPS + 1,
                   generator=torch.Generator().manual_seed(42),
                   output_type="np", output=["videos", "audio", "sampling_rate"])
    gen_s = time.time() - g0
    peak = max(torch.cuda.max_memory_allocated(i)/1e9
               for i in range(torch.cuda.device_count()))
    log(f"DENOISE DONE {gen_s:.1f}s; peak {peak:.1f}GB/card")

    from diffusers.utils.export_utils import encode_video
    audio = res.get("audio"); a0 = audio[0] if audio is not None else None
    sr = int(res["sampling_rate"]) if res.get("sampling_rate") is not None else None
    encode_video(res["videos"][0], fps=24, output_path=OUT, audio=a0, audio_sample_rate=sr)
    log(f"SAVED {OUT} (gen {gen_s:.1f}s total {time.time()-t0:.0f}s)")
    log("NEXT: extract frame 0 AND a mid frame, LOOK — frame 0 must NOT be the ref image.")

if __name__ == "__main__":
    main()
