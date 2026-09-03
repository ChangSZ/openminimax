#!/usr/bin/env python3
"""FLUX.2 fp8 single-image smoke test — the go/no-go for the image path.

Measures the ONE unknown that decides the whole design (README §2):
does FLUX.2-dev fp8 FIT and run FAST on a single 48GB L40S? Prints peak VRAM and
wall-clock and saves a PNG to LOOK AT (the iron rule from the H3 work: trust pixels,
not status — see README §3).

Assembly (learned by probing the real weights):
  * DiT  : Flux2Transformer2DModel.from_single_file(flux2_dev_fp8mixed.safetensors)  [Comfy fp8]
  * VAE  : AutoencoderKLFlux2.from_single_file(flux2-vae.safetensors)                [Comfy fp8]
  * text : official Mistral3ForConditionalGeneration + PixtralProcessor from the gated
           black-forest-labs/FLUX.2-dev `text_encoder/` + `tokenizer/` (bf16).
           The Comfy fp8 text-encoder single-file is UNUSABLE: renamed keys
           (`model.layers.*` vs official `language_model.model.layers.*`, 0/585 overlap)
           AND vision-stripped (no vision_tower/multi_modal_projector), so it can't
           instantiate the Mistral3 VLM the flux2 pipeline needs.

VRAM math (why this is go/no-go): DiT fp8 ~33GB + VAE resident on the card; the text
encoder is OFFLOADED (bf16, runs once to embed the prompt, then streams off) so it
does NOT eat the card's VRAM. What must fit 48GB = DiT + VAE + activations (~40GB).

Pinned to an idle card via CUDA_VISIBLE_DEVICES so it never touches H3 on cuda:0:
  CUDA_VISIBLE_DEVICES=2 python serving/gen_flux_smoketest.py \
      --root /opt/dlami/nvme/flux2 --official /opt/dlami/nvme/flux2-official \
      --steps 8 --long-edge 1024

Standalone (no gateway import). Whatever assembly works here gets mirrored back into
serving/flux_image_server.py.
"""
import argparse, os, time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("H3_FLUX_DEVICE", "2"))
# Reduce fragmentation at the 48GB edge (same trick H3 uses at its 46GB edge).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

# Comfy-Org/flux2-dev layout (relative to --root) — only the Turbo LoRA is still used
# from here; DiT/VAE/text-enc now come from the official diffusers dir (--official),
# because the Comfy fp8mixed DiT loses its scale tensors through from_single_file.
_LORA = "split_files/loras/Flux2TurboComfyv2.safetensors"


def _load_text_encoder(official_dir, dtype):
    """The official FLUX.2 text encoder — a full `Mistral3ForConditionalGeneration`
    VLM — from the gated repo's `text_encoder/` + `tokenizer/` (downloaded to
    `official_dir`). The tokenizer is a Pixtral processor (model_index declares it),
    which the flux2 pipeline's apply_chat_template / pixel_values path needs.
    Returns (text_encoder, tokenizer, how)."""
    from transformers import Mistral3ForConditionalGeneration, AutoProcessor
    te = Mistral3ForConditionalGeneration.from_pretrained(
        os.path.join(official_dir, "text_encoder"), torch_dtype=dtype)
    tok = AutoProcessor.from_pretrained(os.path.join(official_dir, "tokenizer"))
    return te, tok, "official Mistral3ForConditionalGeneration + AutoProcessor"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("FLUX_ROOT", "/opt/dlami/nvme/flux2"))
    ap.add_argument("--official", default=os.environ.get(
        "FLUX_OFFICIAL", "/opt/dlami/nvme/flux2-official"),
        help="dir with the official text_encoder/ + tokenizer/ (bf16)")
    ap.add_argument("--prompt", default="a red panda on a mossy log, golden hour, sharp")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--long-edge", type=int, default=1024)
    ap.add_argument("--no-lora", action="store_true", help="skip the Turbo LoRA")
    ap.add_argument("--quant", default="float8dq",
                    help="torchao quant type for the DiT (e.g. float8dq, float8wo)")
    ap.add_argument("--no-quant", action="store_true",
                    help="load the DiT in bf16, no torchao (needs offload to fit)")
    ap.add_argument("--out", default="/mnt/models/outputs/flux_smoke.png")
    args = ap.parse_args()

    lora_p = os.path.join(args.root, _LORA)   # Turbo LoRA still from the Comfy pack
    for sub in ("transformer", "vae", "text_encoder"):
        if not os.path.isdir(os.path.join(args.official, sub)):
            raise SystemExit(f"missing official {sub}/ under {args.official}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # After CUDA_VISIBLE_DEVICES the pinned physical card is the ONLY visible device,
    # index 0. Use the integer index for the memory APIs (a "cuda:0" string can trip
    # "Invalid device argument" on some torch builds); pass the string to diffusers.
    dev_idx = 0
    dev = "cuda:0"
    dtype = torch.bfloat16
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available to this process (check CUDA_VISIBLE_DEVICES "
                         "and that no cu13 LD_LIBRARY_PATH override breaks CUDA init)")
    torch.cuda.init()                         # force lazy CUDA init before mem stats
    torch.cuda.reset_peak_memory_stats(dev_idx)
    t0 = time.time()

    from diffusers import (Flux2Pipeline, Flux2Transformer2DModel,
                           AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler,
                           TorchAoConfig)
    tr_dir = os.path.join(args.official, "transformer")
    vae_dir = os.path.join(args.official, "vae")
    # DiT: the Comfy fp8mixed single-file gives NOISE (diffusers from_single_file drops
    # its fp8 *_scale tensors). Load the OFFICIAL bf16 transformer and quantize at load
    # with torchao fp8 (diffusers-native) to fit the card. --no-quant falls back to
    # bf16 (needs offload to fit).
    if args.no_quant:
        print(f"[load] DiT bf16 (no quant) <- official {tr_dir}")
        transformer = Flux2Transformer2DModel.from_pretrained(tr_dir, torch_dtype=dtype)
    else:
        # This diffusers/torchao build wants an AOBaseConfig INSTANCE (not a string).
        # float8dq == dynamic-activation float8 weight (standard fp8 inference).
        from torchao.quantization import (Float8DynamicActivationFloat8WeightConfig,
                                          Float8WeightOnlyConfig)
        ao = (Float8WeightOnlyConfig() if args.quant == "float8wo"
              else Float8DynamicActivationFloat8WeightConfig())
        print(f"[load] DiT bf16->fp8 (torchao {type(ao).__name__}) <- official {tr_dir}")
        qc = TorchAoConfig(ao)
        transformer = Flux2Transformer2DModel.from_pretrained(
            tr_dir, quantization_config=qc, torch_dtype=dtype)
    print(f"[load] VAE     <- official {vae_dir}")
    vae = AutoencoderKLFlux2.from_pretrained(vae_dir, torch_dtype=dtype)
    print(f"[load] text-encoder <- official {args.official}")
    text_encoder, tokenizer, how = _load_text_encoder(args.official, dtype)
    print(f"[load]   text-encoder via {how}")

    scheduler = FlowMatchEulerDiscreteScheduler()
    pipe = Flux2Pipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder,
                         tokenizer=tokenizer, transformer=transformer)
    if not args.no_lora and os.path.isfile(lora_p):
        pipe.load_lora_weights(lora_p)
        print(f"[load] applied Turbo LoRA {os.path.basename(lora_p)}")

    # Placement is manual, NOT whole-pipeline offload, because the DiT is now a
    # torchao Float8Tensor and accelerate's offload calls `.to(dtype)` on it, which
    # torchao does not implement ("Float8Tensor dispatch ... aten.to.dtype" — same
    # wall the H3 int8 attempt hit). So:
    #   * fp8 DiT (~32GB) + VAE (0.3GB)  -> RESIDENT on the card (no offload needed;
    #     that was the whole point of quantizing to fit 48GB);
    #   * bf16 text encoder (~48GB, > card) -> offload ONLY this module to CPU and
    #     stream its submodules on demand (it runs once per image to embed the prompt).
    if args.no_quant:
        pipe.enable_sequential_cpu_offload(device=dev)   # bf16 DiT can't stay resident
    else:
        from accelerate import cpu_offload
        pipe.transformer.to(dev)
        pipe.vae.to(dev)
        cpu_offload(pipe.text_encoder, execution_device=dev)   # only the big encoder
    load_s = time.time() - t0
    print(f"[load] done in {load_s:.0f}s")

    le = args.long_edge
    width, height = le, round(le * 9 / 16 / 16) * 16   # 16:9 canvas
    print(f"[gen] {width}x{height} steps={args.steps}")
    g0 = time.time()
    with torch.inference_mode():
        img = pipe(prompt=args.prompt, width=width, height=height,
                   num_inference_steps=args.steps, guidance_scale=args.guidance,
                   generator=torch.Generator(device=dev).manual_seed(0)).images[0]
    gen_s = time.time() - g0
    peak_gb = torch.cuda.max_memory_allocated(dev_idx) / 1e9

    img.save(args.out)
    print(f"[done] image -> {args.out}")
    print(f"[perf] gen {gen_s:.1f}s | PEAK VRAM {peak_gb:.1f} GB / 48 GB "
          f"| {'FITS' if peak_gb < 47 else 'TIGHT/OOM-RISK'}")
    print("[verify] OPEN the png and LOOK (pixels, not status)")


if __name__ == "__main__":
    main()
