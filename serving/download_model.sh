#!/usr/bin/env bash
# Download MiniMax-H3 to the gp3 EBS volume. HF ships H3 in the *modular* layout, which is
# what the diffusers serving shim (serving/h3_turbo_server.py) loads — so the DEFAULT dest
# is the same path the shim's H3_MODEL defaults to: fresh clone -> this script -> serve
# works with no env. We pull ONLY the root modular layout (~196GB: transformer 62G +
# transformer_ref 62G + text_encoder 63G + vae/audio_vae ~10G), excluding the redundant
# self-contained FL2VA/ + Ref2VA/ per-variant copies (270GB the shim never loads). NOTE the
# HF download uses a transient ~40GB .cache/ during the pull, so budget ~240GB peak for H3.
#
# Video (H3 + Turbo LoRAs) is ungated — no HF token needed. Image (FLUX.2-dev) is GATED:
# set HF_TOKEN (and accept the license, see the FLUX block below) and it's pulled too;
# without a token the FLUX step is skipped and you get a video-only box.
#
# Caching weights on EBS is what makes auto-stop / Spot cheap: a restart re-attaches
# the volume instead of re-pulling ~196GB. Target dir must be on the >=500GB gp3
# volume (infra/), NOT the root disk — H3 (~196GB) + FLUX.2 (~110GB) + transient HF
# cache + output headroom.
#
# License note (docs/PLAN.md §4.3 and NOTICE.md): MiniMax H3 Community License —
# serving this to others is commercial use and has a regional application
# (US/EU/UK/KR). Clear that BEFORE exposing the gateway to users/clients.
set -euo pipefail

# hf CLI lives in the DLAMI venv, NOT on the plain PATH. Bare `pip`/`hf` fail under the
# systemd/SSM shell (that venv's bin dir isn't exported), so always call them explicitly.
# Override PIPBIN/HFBIN for a non-DLAMI host.
PIPBIN="${PIPBIN:-/opt/pytorch/bin/pip}"
HFBIN="${HFBIN:-/opt/pytorch/bin/hf}"
command -v "$PIPBIN" >/dev/null || PIPBIN=pip     # fall back to PATH if the venv isn't there
command -v "$HFBIN"  >/dev/null || HFBIN=hf

MODEL_ID="${MODEL_ID:-MiniMaxAI/MiniMax-H3}"
# Must match serving/h3_turbo_server.py's H3_MODEL default.
DEST="${MODEL_PATH:-/mnt/models/MiniMax-H3-modular}"

# --- Image model (FLUX.2-dev) — OPTIONAL and GATED ---------------------------------
# black-forest-labs/FLUX.2-dev is a GATED repo: you must (1) accept its license at
# https://huggingface.co/black-forest-labs/FLUX.2-dev and (2) supply an HF access token
# (https://huggingface.co/settings/tokens) via HF_TOKEN. Video (H3) is NOT gated and
# needs no token. Set DOWNLOAD_FLUX=1 (default: auto — on iff HF_TOKEN is present) to pull
# the official diffusers layout the FLUX shim (serving/flux_image_server.py) expects.
FLUX_ID="${FLUX_ID:-black-forest-labs/FLUX.2-dev}"
FLUX_DEST="${FLUX_MODEL:-/mnt/models/flux2-official}"
HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "${DOWNLOAD_FLUX:-}" ]]; then
  [[ -n "$HF_TOKEN" ]] && DOWNLOAD_FLUX=1 || DOWNLOAD_FLUX=0
fi

# Turbo LoRAs — pulled by default (the diffusers shim loads them via load_lora_weights).
# BOTH are small (a couple GB total) next to the ~196GB base, so we grab both so the
# shim serves every workflow out of the box:
#   * fl2v 8-step  -> transformer/     (t2va + keyframe i2va/fl2va;  num_inference_steps=9)
#   * ref2v 4-step -> transformer_ref/ (reference image->video, ref2va; steps=5)
# Both live in the same lightx2v repo. Set DOWNLOAD_TURBO_LORA=0 to skip all LoRAs;
# override LORA_FILES="a.safetensors b.safetensors" to pick a subset.
DOWNLOAD_TURBO_LORA="${DOWNLOAD_TURBO_LORA:-1}"
LORA_REPO="${LORA_REPO:-lightx2v/Minimax-h3-Turbo}"
LORA_FILES="${LORA_FILES:-minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors}"
LORA_DEST="${LORA_DEST:-/mnt/models/lora}"

mkdir -p "$DEST"
df -h "$(dirname "$DEST")" | awk 'NR==1||NR==2{print}'
echo "Downloading $MODEL_ID -> $DEST (~196GB; skips the redundant FL2VA/Ref2VA copies)"

# The repo ships the weights THREE ways: the root-level *modular* layout
# (modular_model_index.json -> root transformer/ transformer_ref/ text_encoder/ vae/ ...)
# that our shim loads, PLUS self-contained per-variant copies under FL2VA/ (135GB) and
# Ref2VA/ (135GB) for the classic (non-modular) loader. h3_turbo_server.py uses ONLY the
# modular layout (verified: modular_model_index.json references no FL2VA/Ref2VA path), so
# we exclude those two subdirs — cuts the pull from ~465GB to ~196GB. Re-include them
# (drop the two --exclude lines) only if you switch to the classic per-variant loader.
#
# hf CLI (huggingface_hub). H3 is NOT gated -> no token needed.
# NOTE: `hf download` takes ONE glob per --exclude, and any bare positional after the repo
# id is treated as a FILENAME to fetch — so the patterns must each have their own --exclude
# (the old `--exclude "*.pth" "original/*"` made `original/*` a filename -> 404 on hf>=1.x).
"$PIPBIN" install -q "huggingface_hub"
"$HFBIN" download "$MODEL_ID" \
  --local-dir "$DEST" \
  --exclude "*.pth" --exclude "original/*" \
  --exclude "FL2VA/*" --exclude "Ref2VA/*"    # keep root modular layout; skip dup packagings

if [[ "$DOWNLOAD_TURBO_LORA" == "1" ]]; then
  mkdir -p "$LORA_DEST"
  for f in $LORA_FILES; do
    if [[ -s "$LORA_DEST/$f" ]]; then
      echo "Turbo LoRA already present, skipping: $LORA_DEST/$f"
      continue
    fi
    echo "Downloading Turbo LoRA $LORA_REPO/$f -> $LORA_DEST (small)"
    "$HFBIN" download "$LORA_REPO" "$f" --local-dir "$LORA_DEST"
  done
fi

# --- FLUX.2-dev (image) — only when requested; requires the license + HF_TOKEN --------
if [[ "$DOWNLOAD_FLUX" == "1" ]]; then
  if [[ -z "$HF_TOKEN" ]]; then
    echo "!! DOWNLOAD_FLUX=1 but HF_TOKEN is empty. FLUX.2-dev is GATED — accept the license" >&2
    echo "!! at https://huggingface.co/$FLUX_ID and pass HF_TOKEN=<token>. Skipping image model." >&2
  else
    mkdir -p "$FLUX_DEST"
    echo "Downloading $FLUX_ID -> $FLUX_DEST (gated; ~110GB: bf16 DiT 7 shards + Mistral3 text_encoder 45G + vae)"
    # transformer/ vae/ text_encoder/ tokenizer/ — the dirs flux_image_server.py loads.
    # One glob per --exclude (see the H3 note above); --token authenticates the gated pull.
    "$HFBIN" download "$FLUX_ID" \
      --local-dir "$FLUX_DEST" \
      --token "$HF_TOKEN" \
      --exclude "*.pth" --exclude "original/*" --exclude "*.gguf"
  fi
fi

echo "Done. Weights at $DEST:"
du -sh "$DEST"
[[ "$DOWNLOAD_TURBO_LORA" == "1" ]] && du -sh "$LORA_DEST" 2>/dev/null || true
[[ "$DOWNLOAD_FLUX" == "1" && -d "$FLUX_DEST" ]] && du -sh "$FLUX_DEST" 2>/dev/null || true
