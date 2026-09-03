#!/bin/bash
# FLUX.2 shim launcher — points at the PERSISTENT EBS copy (/mnt/models), so weights
# survive stop/start (the ephemeral /opt/dlami/nvme copy gets wiped on every restart).
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/models/hf-cache
export FLUX_MODEL=/mnt/models/flux2-official
export FLUX_OUTDIR=/mnt/models/outputs
export FLUX_STEPS=28
export FLUX_QUANTIZE=1
mkdir -p /mnt/models/logs /mnt/models/outputs
nohup /opt/pytorch/bin/python /opt/openminimax/serving/flux_image_server.py \
  > /mnt/models/logs/flux_shim.log 2>&1 &
echo "flux shim started pid $! (model=$FLUX_MODEL steps=$FLUX_STEPS)"
