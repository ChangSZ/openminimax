#!/usr/bin/env python3
"""Single-clip smoke test: MiniMax-H3 Turbo (8-step LoRA) 768p on 4x L40S.

This is the minimal reference that proved the working recipe before it was wrapped
into the HTTP shim (h3_turbo_server.py). Use it to sanity-check the box after a
resume: `python gen_turbo_smoketest.py` → one mp4 → extract a frame and LOOK at it.

Why this shape (see ../serving/README.md and README §3):
- 46GB cards fit no full H3 component (transformer 61.7GB + Qwen3-VL 62.1GB): FSDP2
  bf16 needs 8 cards; single-card auto-offload can't place a 61.7GB component; torchao
  int8 + any diffusers offload hits a storage-aliasing bug on this torch 2.13 build.
  So: bf16 + block-level streamed group offload (weights stream from 372GB host RAM).
- LoRA via `pipe.load_lora_weights()` (MiniMaxH3LoraLoaderMixin) which DE-INTERLEAVES
  the per-head fused QKV — the raw PEFT load_state_dict path gives NOISE.
"""
import os, time, torch
from diffusers import ModularPipeline
from diffusers.hooks import apply_group_offloading

MODEL = os.environ.get("H3_MODEL", "/mnt/models/MiniMax-H3-modular")
LORA = os.environ.get("H3_LORA", "/mnt/models/lora/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors")
OUT = os.environ.get("H3_OUT", "/mnt/models/outputs/smoketest_turbo_8step.mp4")
INFERENCE_STEPS = 8
VIDEO_SHIFT, AUDIO_SHIFT = 6.0, 3.0
WIDTH, HEIGHT = 1344, 768
NUM_FRAMES = 124

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def main():
    t0=time.time()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    log("building pipeline")
    pipe = ModularPipeline.from_pretrained(MODEL)
    log("load_components(workflow=t2va) bf16 (weights land in host RAM)")
    pipe.load_components(workflow="t2va", dtype=torch.bfloat16,
                         pretrained_model_name_or_path=MODEL)

    # Use the pipeline's OWN LoRA loader (MiniMaxH3LoraLoaderMixin) — it runs
    # `_convert_non_diffusers_minimax_h3_lora_to_diffusers`, which DE-INTERLEAVES the
    # per-head fused QKV that the published LoRA is trained against onto the modular
    # checkpoint's [q_all;k_all;v_all] layout. The raw PEFT load_state_dict path
    # (official inference_minimax_h3.py) SKIPS this remap -> LoRA lands on the wrong
    # layout -> noise (verified). Alpha is read from the file's __metadata__.
    log("load_lora_weights via MiniMaxH3LoraLoaderMixin (de-interleaves QKV)")
    pipe.load_lora_weights(LORA)
    pipe.transformer.requires_grad_(False); pipe.transformer.eval()
    pipe.text_encoder.requires_grad_(False)
    pipe.scheduler.set_shift(VIDEO_SHIFT)
    pipe.audio_scheduler.set_shift(AUDIO_SHIFT)
    log(f"shifts video={pipe.scheduler.shift} audio={pipe.audio_scheduler.shift}")

    # bf16 block-level streamed offload — no torchao, so no storage-aliasing bug.
    off = dict(onload_device=torch.device("cuda:0"),
               offload_device=torch.device("cpu"), use_stream=True)
    pipe.transformer.enable_group_offload(offload_type="block_level",
                                          num_blocks_per_group=1, **off)
    apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **off)
    pipe.vae.to("cuda:0")
    pipe.audio_vae.to("cuda:0")
    log(f"setup done {time.time()-t0:.0f}s; starting denoise")

    prompt = ("integrated_multimodal_description: [Shot 1] Live-action, cinematic, a "
        "low-angle medium-wide tracking shot follows a red panda ambling along a mossy "
        "log in a misty bamboo forest at golden hour, soft shafts of light between the "
        "stalks. The camera arcs gently right at slow speed as the red panda pauses, "
        "sniffs a bamboo leaf, and looks up toward the lens with bright curious eyes.\n\n"
        "overall_soundscape: Gentle forest ambience, soft rustling bamboo leaves, faint "
        "birdsong.\n\nnon_diegetic_music: A warm, slow acoustic guitar melody.")

    g0=time.time()
    with torch.inference_mode():
        res = pipe(prompt=prompt, height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
                   num_inference_steps=INFERENCE_STEPS+1,
                   generator=torch.Generator().manual_seed(42),
                   output_type="np", output=["videos","audio","sampling_rate"])
    gen_s=time.time()-g0
    peak=max(torch.cuda.max_memory_allocated(i)/1e9 for i in range(torch.cuda.device_count()))
    log(f"DENOISE DONE {gen_s:.1f}s; peak {peak:.1f}GB/card")

    from diffusers.utils.export_utils import encode_video
    audio=res.get("audio"); a0=audio[0] if audio is not None else None
    sr=int(res["sampling_rate"]) if res.get("sampling_rate") is not None else None
    encode_video(res["videos"][0], fps=24, output_path=OUT, audio=a0, audio_sample_rate=sr)
    log(f"SAVED {OUT} (gen {gen_s:.1f}s total {time.time()-t0:.0f}s)")

if __name__=="__main__":
    main()
