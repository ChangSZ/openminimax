#!/usr/bin/env python3
"""Phase-0 go/no-go: measure H3 768P wall-clock + peak VRAM on this box.

The one number the docs don't give (docs/PLAN.md §1): official H3 step-times are all
anchored at 480P, so nobody has published 768P wall-clock, and on 4×L40S (PCIe, no
NVLink) it will differ from the H100 reference. This is what decides whether the
economics work. Run it on the GPU box AFTER serve_h3.sh is up.

It talks to SGLang's private `/v1/videos` directly (NOT the gateway) so it measures
the model, not our queueing. VRAM is read from `nvidia-smi` around the call.

    python measure_768p.py --duration 6 --prompt "a red fox trots through snow"

Prints one wall-clock per clip and peak VRAM, and appends a row to
serving/phase0_results.csv so repeated runs (different durations) accumulate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import subprocess
import time
import urllib.request

SGLANG_URL = os.environ.get("SGLANG_URL", "http://127.0.0.1:30010")
RESULTS = pathlib.Path(__file__).with_name("phase0_results.csv")


def peak_vram_mb() -> int:
    """Highest per-GPU memory.used across the 4 cards, in MiB. 0 if nvidia-smi absent."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True)
    except (OSError, subprocess.CalledProcessError):
        return 0
    return max((int(x) for x in out.split() if x.strip().isdigit()), default=0)


def generate(prompt: str, duration_s: int, ratio: str, resolution: str,
             steps: int) -> float:
    """One /v1/videos call. Returns wall-clock seconds. Mirrors backend._to_sglang
    (⚠️ PHASE-0-VERIFY: adjust both together once the live shape is confirmed).

    `steps` is `num_inference_steps` with SGLang's off-by-one: the 8-step Turbo LoRA
    (serve_h3.sh default) is measured with steps=9. Measure the config you'll ACTUALLY
    serve, so the go/no-go number reflects the Turbo path, not bare 50-step."""
    body = json.dumps({
        "prompt": prompt,
        "task": "t2va",
        "target": {"aspect_ratio": ratio, "duration_seconds": duration_s,
                   "short_edge": 768 if resolution.upper() == "768P" else 480},
        "num_inference_steps": steps,
        "num_outputs_per_prompt": 1,
    }).encode()
    req = urllib.request.Request(
        f"{SGLANG_URL.rstrip('/')}/v1/videos", data=body, method="POST",
        headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        resp.read()
    return time.monotonic() - start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a red fox trots across fresh snow at dawn")
    ap.add_argument("--duration", type=int, default=6)
    ap.add_argument("--ratio", default="16:9")
    ap.add_argument("--resolution", default="768P")
    ap.add_argument("--runs", type=int, default=1)
    # Default 9 == the 8-step Turbo LoRA (off-by-one). Use 5 for a 4-step LoRA, ~50
    # for bare base. Must match whatever serve_h3.sh loaded.
    ap.add_argument("--steps", type=int, default=int(os.environ.get("SGLANG_STEPS", "9")),
                    help="num_inference_steps (8-step Turbo -> 9; base -> ~50)")
    args = ap.parse_args()

    for i in range(args.runs):
        before = peak_vram_mb()
        wall = generate(args.prompt, args.duration, args.ratio, args.resolution,
                        args.steps)
        after = peak_vram_mb()
        peak = max(before, after)
        print(f"run {i+1}/{args.runs}: {args.resolution} {args.ratio} "
              f"{args.duration}s clip, {args.steps} steps -> {wall:.1f}s wall, "
              f"peak VRAM {peak} MiB")
        new = not RESULTS.exists()
        with RESULTS.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["resolution", "ratio", "duration_s", "steps",
                            "wall_s", "peak_vram_mb"])
            w.writerow([args.resolution, args.ratio, args.duration, args.steps,
                        round(wall, 1), peak])
    print(f"\nAppended to {RESULTS}. This wall-clock is the Phase-0 go/no-go number "
          f"(docs/PLAN.md §1) — for the Turbo config you'll actually serve.")


if __name__ == "__main__":
    main()
