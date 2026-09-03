#!/usr/bin/env bash
# One-shot deploy for a fresh GPU box: turns a clean g6e.12xlarge (PyTorch DLAMI) into a
# request-ready MiniMax-H3 node — WITHOUT git or any network access to this repo. You
# ship this repo to the box (scp/rsync/tar over SSM), then run this ONCE from the repo
# root's serving/ dir. It is the single "clone -> copy -> deploy -> reboot-ready" step.
#
#   sudo RESULT_BUCKET=<your-result-bucket> bash serving/bootstrap.sh
#
# Idempotent and EBS-aware: weights/LoRAs on /mnt/models and deps in /opt/pytorch are
# detected and SKIPPED, so re-running after a stop/start is cheap (nothing re-downloads).
# After it completes, the shim + worker are systemd units enabled on boot — a plain
# `stop`/`start` of the instance brings the box back to serving with no manual steps.
#
# What it does, in order:
#   1. system packages — ffmpeg (REQUIRED: without ffprobe the pipeline generates a clip
#      then fails validation with a fake "ffprobe is required" error).
#   2. python deps into the DLAMI venv /opt/pytorch — diffusers(main) + peft. H3's
#      modular pipeline + the QKV-de-interleaving LoRA loader only exist on diffusers>=0.41.
#   3. cu13 lib symlink so the JIT fused-attention kernel can link -lcudart.
#   4. model + LoRA weights onto the EBS volume via download_model.sh (~196GB modular
#      checkpoint — the redundant FL2VA/Ref2VA copies are excluded — + two small Turbo
#      LoRAs; skips whatever is already there). Pass HF_TOKEN to also pull the GATED
#      FLUX.2-dev image model (accept its license first); omit for video-only.
#   5. deploy the repo (gateway/ + serving/) to /opt/openminimax/ — where the units expect it.
#   6. write /etc/openminimax.env with the account-specific values (RESULT_BUCKET, tables).
#   7. install + enable the systemd units (serve + worker) so they auto-start on boot.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PYBIN="${PYBIN:-/opt/pytorch/bin/python}"
PIPBIN="${PIPBIN:-/opt/pytorch/bin/pip}"
MODEL_PATH="${MODEL_PATH:-/mnt/models/MiniMax-H3-modular}"
OPT="${OPT:-/opt/openminimax}"

# --- control-plane values that the worker units need (account/region-specific) --------
# Defaults match the serverless-stack naming; RESULT_BUCKET has no safe default, so it
# must be supplied (env or arg) unless it's already in a prior /etc/openminimax.env.
AWS_REGION="${AWS_REGION:-us-west-2}"
KEYS_TABLE="${KEYS_TABLE:-openminimax-serverless-keys}"
TASKS_TABLE="${TASKS_TABLE:-openminimax-serverless-tasks}"
RESULT_BUCKET="${RESULT_BUCKET:-}"
if [[ -z "$RESULT_BUCKET" && -f /etc/openminimax.env ]]; then
  RESULT_BUCKET="$(sed -n 's/^RESULT_BUCKET=//p' /etc/openminimax.env | tail -1)"
fi
# OPTIONAL: image generation. FLUX.2-dev is a GATED HF repo — to also self-host images,
# accept its license at https://huggingface.co/black-forest-labs/FLUX.2-dev and pass an
# HF access token: `sudo HF_TOKEN=hf_... RESULT_BUCKET=<bucket> bash serving/bootstrap.sh`.
# Video (H3) is ungated and needs no token; without HF_TOKEN the box is video-only.
HF_TOKEN="${HF_TOKEN:-}"

echo "==> [1/7] system packages (ffmpeg)"
if command -v ffprobe >/dev/null 2>&1; then
  echo "    ffprobe present, skipping"
else
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq && sudo apt-get install -y ffmpeg
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y ffmpeg || sudo dnf install -y ffmpeg-free
  else
    echo "    !! no apt-get/dnf found — install ffmpeg manually" >&2
  fi
fi

echo "==> [2/7] python deps into /opt/pytorch (diffusers main + peft + torchao)"
if "$PYBIN" - <<'PY' 2>/dev/null
import sys
try:
    import diffusers, peft
    # need the modular H3 pipeline (diffusers >= 0.41 / main)
    import diffusers.modular_pipelines.minimax_h3  # noqa: F401
    # torchao provides the fp8 config the FLUX.2 image shim loads (flux_image_server.py);
    # harmless for video-only, required for image gen.
    import torchao  # noqa: F401
    # PyAV backs diffusers' encode_video (h3_turbo_server.py) — without it the video shim
    # generates frames then dies with "PyAV is required to use encode_video".
    import av  # noqa: F401
    print("ok")
except Exception:
    sys.exit(1)
PY
then
  echo "    diffusers(modular H3) + peft + torchao + av already installed, skipping"
else
  echo "    installing diffusers(main) + peft + torchao + av (this can take a few minutes)"
  "$PIPBIN" install -q "git+https://github.com/huggingface/diffusers.git" peft torchao av
fi

echo "==> [3/7] cu13 lib symlink for the JIT attention kernel"
CU13="/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13"
if [[ -d "$CU13/lib" && ! -e "$CU13/lib64" ]]; then
  ln -sfn "$CU13/lib" "$CU13/lib64" && echo "    linked $CU13/lib64 -> lib"
else
  echo "    symlink present or cu13 layout differs, skipping"
fi

echo "==> [4/7] model + LoRA weights (EBS; skips what's already downloaded)"
if [[ -f "$MODEL_PATH/model_index.json" || -f "$MODEL_PATH/modular_model_index.json" ]]; then
  echo "    checkpoint already at $MODEL_PATH — download_model.sh will fill any gaps"
fi
# Pass HF_TOKEN through so download_model.sh can also pull gated FLUX.2-dev when supplied
# (empty token -> video-only; the download script skips FLUX with a clear message).
MODEL_PATH="$MODEL_PATH" HF_TOKEN="$HF_TOKEN" bash "$HERE/download_model.sh"

echo "==> [5/7] deploy repo -> $OPT (gateway/ + serving/)"
sudo mkdir -p "$OPT"
# If the repo was shipped straight to $OPT (e.g. tarball unpacked there), $REPO == $OPT and
# there is nothing to copy — cp would error ("same file") and abort. Skip in that case.
if [[ "$REPO" -ef "$OPT" ]]; then
  echo "    repo already lives at $OPT (REPO==OPT) — nothing to copy"
else
  # -a preserves modes; --delete would risk wiping local-only files, so we mirror additively.
  sudo cp -a "$REPO/gateway" "$OPT/"
  sudo cp -a "$REPO/serving" "$OPT/"
  echo "    deployed $OPT/gateway and $OPT/serving"
fi

echo "==> [6/7] write /etc/openminimax.env (control-plane values for the units)"
if [[ -z "$RESULT_BUCKET" ]]; then
  echo "    !! RESULT_BUCKET is empty — pass RESULT_BUCKET=<bucket> to this script." >&2
  echo "    !! (workers won't start without it; re-run bootstrap once you have it.)" >&2
fi
sudo tee /etc/openminimax.env >/dev/null <<EOF
# Generated by serving/bootstrap.sh — control-plane values the worker units read.
# Edit + 'systemctl restart openminimax-worker' to change; NOT tracked in git.
AWS_REGION=${AWS_REGION}
AWS_DEFAULT_REGION=${AWS_REGION}
KEYS_TABLE=${KEYS_TABLE}
TASKS_TABLE=${TASKS_TABLE}
RESULT_BUCKET=${RESULT_BUCKET}
EOF
sudo chmod 600 /etc/openminimax.env
echo "    wrote /etc/openminimax.env (RESULT_BUCKET='${RESULT_BUCKET:-<UNSET>}')"

echo "==> [7/7] install + enable systemd units (auto-start on boot)"
sudo cp "$REPO/infra/openminimax-serve.service"  /etc/systemd/system/
sudo cp "$REPO/infra/openminimax-worker.service" /etc/systemd/system/
UNITS="openminimax-serve openminimax-worker"
# Image generation (FLUX.2) is OPTIONAL — only wire it if the FLUX weights are on EBS.
FLUX_MODEL_DIR="${FLUX_MODEL:-/mnt/models/flux2-official}"
if [[ -f "$FLUX_MODEL_DIR/transformer/config.json" ]]; then
  sudo cp "$REPO/infra/openminimax-flux.service"         /etc/systemd/system/
  sudo cp "$REPO/infra/openminimax-image-worker.service" /etc/systemd/system/
  UNITS="$UNITS openminimax-flux openminimax-image-worker"
  echo "    FLUX weights found at $FLUX_MODEL_DIR — image gen enabled too"
else
  echo "    no FLUX weights at $FLUX_MODEL_DIR — skipping image gen (video-only). See README §2."
fi
sudo systemctl daemon-reload
sudo systemctl enable $UNITS
echo "    enabled: $UNITS"

echo
echo "==> bootstrap done. Start serving now (first warm load ~2-18min while EBS hydrates):"
echo "    sudo systemctl start $UNITS"
echo "    curl -s http://127.0.0.1:30010/health   # video ready when {\"ready\":true}"
[[ "$UNITS" == *flux* ]] && echo "    curl -s http://127.0.0.1:30020/health   # image ready when {\"ready\":true}"
echo "    # After a reboot / stop-start, all units come up on their own — no manual steps."
