#!/bin/bash
# Launch the diffusers MiniMax-H3 Turbo HTTP shim (serving/h3_turbo_server.py) on the
# g6e.12xlarge box. Binds 127.0.0.1:30010 with the async /v1/videos protocol the gateway
# worker expects — a drop-in for the deprecated SGLang serve (which produced noise).
#
# Run:   nohup bash run_h3_server.sh > /mnt/models/logs/h3_server.log 2>&1 &
#        curl -s http://127.0.0.1:30010/health   # wait for {"ready":true} (~100s warm load)
#
# Prereqs on the box (persist on the EBS volume):
#   /mnt/models/MiniMax-H3-modular   HF *modular*-layout checkpoint (~196GB root layout)
#   /mnt/models/lora/...8step...      Turbo LoRA
#   diffusers 0.41.dev + peft in /opt/pytorch  (pip install "git+.../diffusers" peft)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/opt/pytorch/bin:$PATH

# cu13 libs: the JIT fused-attention kernel links -lcudart; libcudart is under cu13/lib
# but the linker looks in cu13/lib64. Point both LIBRARY_PATH and the lib64 symlink at it.
export CUDA_HOME=/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13
[ -e "$CUDA_HOME/lib64" ] || ln -sfn "$CUDA_HOME/lib" "$CUDA_HOME/lib64" 2>/dev/null || true
export LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation at the 46GB edge
export HF_HUB_OFFLINE=1                                   # everything is local; never hit the Hub
export TOKENIZERS_PARALLELISM=false

# Optional overrides (defaults live in h3_turbo_server.py):
#   H3_MODEL, H3_LORA, H3_OUTDIR, H3_STEPS (8), H3_PORT (30010)

exec /opt/pytorch/bin/python "$HERE/h3_turbo_server.py"
