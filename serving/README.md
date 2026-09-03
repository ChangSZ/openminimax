# serving — MiniMax-H3 (768P) Turbo on 4×L40S, via diffusers

Runs the open-weights **MiniMax-H3** video model on a `g6e.12xlarge` (4×L40S, 372GB
RAM) and exposes it on `127.0.0.1:30010` with the async `/v1/videos` submit+poll
protocol the gateway worker expects. The gateway ([`../gateway`](../gateway)) is the
only thing that talks to this; external clients never do.

> **Nothing here runs until the GPU instance is launched** (see [`../infra`](../infra)),
> which bills **~$10/hr**. Launch is an explicitly-approved step, not automatic.

## ⚠️ Read this first: what actually works (2026-08-30)

The original plan was "**SGLang** serves H3 **with the Turbo LoRA**". That path is
**dead** — SGLang applies the lightx2v Turbo LoRA wrong and produces pure noise (not
a param problem — the root cause is the LoRA loader, explained just below). The working path is:

- **diffusers modular pipeline** (`ModularPipeline` + `MiniMaxH3Blocks`), NOT SGLang.
- **LoRA loaded via `pipe.load_lora_weights(...)`** — this runs diffusers'
  `_convert_non_diffusers_minimax_h3_lora_to_diffusers`, which **de-interleaves the
  per-head fused QKV** the published LoRA was trained against onto the modular
  checkpoint's layout. The raw PEFT `load_state_dict` path (used by SGLang and by
  lightx2v's own `inference_minimax_h3.py`) matches key *names* but lands the deltas
  on the wrong QKV layout → noise. **This one-line loader choice is the whole fix.**
- **bf16 + block-level streamed group offload** (weights live in the 372GB host RAM,
  streamed onto cuda:0 per transformer block). NOT int8 (torchao 0.18 + this torch
  build hits a CPU↔GPU storage-aliasing bug on any diffusers offload). NOT FSDP2
  (that's for 8×H100; on 4×L40S the text-encoder all-gather OOMs a 46GB card).
- Pixel-verified clean for t2va, i2va, and fl2va (first/last keyframe) shots.

## The serving process

**`serving/h3_turbo_server.py`** (launched by `serving/run_h3_server.sh`; on the box at
`/opt/openminimax/serving/`) — a stdlib
`ThreadingHTTPServer` holding ONE warm diffusers Turbo pipeline, speaking the exact
protocol `../gateway/app/backend.py::SGLangBackend` already talks:

| endpoint | shape |
|---|---|
| `POST /v1/videos` | `{prompt, task, target:{aspect_ratio,duration_seconds,short_edge}, num_inference_steps, conditions:[{type:"image",role:"keyframe",uri,frame_index}]}` → `{id, status:"queued"}` |
| `GET /v1/videos/{id}` | `{status:"queued"|"running"|"completed"|"failed", file_paths:["/root/outputs/<id>.mp4"], error?}` |
| `GET /health` | `{status:"ok", ready:bool}` |

Single-flight FIFO worker thread (H3 saturates the box, one clip at a time). Keyframes
are fetched from each condition's `uri` (frame_index `0`→`image`, `-1`→`last_image`).
`short_edge`+`aspect_ratio` → dims (multiples of 32); `duration_seconds` → frame count
of the form `17k+5`, **clamped to ≤360** (H3's 5–15s ceiling; a naive round-up of a
15s shot overshoots to 362 and the pipeline rejects it).

Because the gateway worker's `SGLANG_URL` already points at `:30010`, this shim is a
drop-in for SGLang — **no gateway/worker code changes**. The worker's `SGLANG_STEPS`
must be **9** (8-step Turbo, SGLang off-by-one).

## Files in this dir

| file | status |
|---|---|
| **`h3_turbo_server.py`** | **production** — the diffusers Turbo HTTP shim (the process described above). Paths are env-overridable (`H3_MODEL`/`H3_LORA`/`H3_OUTDIR`/`H3_STEPS`/`H3_PORT`); defaults are the box layout. |
| **`run_h3_server.sh`** | **production** — launcher: sets cu13 lib paths + `PYTORCH_CUDA_ALLOC_CONF`/`HF_HUB_OFFLINE`, then runs the shim. `nohup bash run_h3_server.sh > … &`. |
| **`gen_turbo_smoketest.py`** | reference/smoke test — generates ONE clip with the same recipe (bf16 offload + `load_lora_weights`). Run after a resume, then extract a frame and LOOK at it. |
| `README.md` (this) | current |
| **`flux_image_server.py`** | **production (optional image)** — the diffusers FLUX.2 fp8 text-to-image shim on `:30020` (cuda:2). Paths env-overridable (`FLUX_MODEL`/`FLUX_OUTDIR`/`FLUX_STEPS`/`FLUX_PORT`). |
| **`bootstrap.sh`** | **production** — one-time first-boot setup (ffmpeg + diffusers/peft into `/opt/pytorch` + cu13 symlink + calls `download_model.sh`, deploys code to `/opt`, installs + enables the systemd units). Idempotent & EBS-aware: re-running after a stop/start skips everything already present. |
| **`download_model.sh`** | **production** — pulls the *modular*-layout checkpoint to `/mnt/models/MiniMax-H3-modular` (matches the shim's `H3_MODEL` default) + both Turbo LoRAs (fl2v 8-step, ref2v 4-step). Skips what's already on EBS. |
| `measure_768p.py` | still works (same `/v1/videos` protocol) to benchmark the shim. (`serve_h3.sh` was **deleted** — SGLang served noise for Turbo.) |

> Deploy: these live in the repo; on the box they're expected at `/opt/openminimax/serving/`
> (the systemd unit [`../infra/openminimax-serve.service`](../infra/openminimax-serve.service)
> runs `h3_turbo_server.py` from there and auto-starts on reboot — resolving the earlier
> "nohup process won't survive a restart" gap).

## Bring-up sequence (on the GPU box, after infra launches it)

**First boot — run `bootstrap.sh` ONCE** (via SSM). It installs ffmpeg, installs
diffusers(main)+peft into `/opt/pytorch`, makes the cu13 lib symlink, and downloads the
checkpoint + both Turbo LoRAs to the EBS volume. It's idempotent and EBS-aware, so a
later stop/start does NOT re-download — everything already on `/mnt/models` is skipped.

```bash
cd /opt/openminimax          # the repo, deployed here (both systemd units expect this path)
sudo bash serving/bootstrap.sh
```
This leaves, all on the persistent EBS volume:
- **Modular checkpoint** at `/mnt/models/MiniMax-H3-modular` (~196GB — the root-level
  layout `modular_model_index.json` loads: transformer 62G + transformer_ref 62G +
  text_encoder 63G + vae/audio_vae. `download_model.sh` excludes the repo's redundant
  self-contained `FL2VA/`+`Ref2VA/` copies, 270GB the modular loader never touches).
- **Turbo LoRAs** in `/mnt/models/lora/`: `...fl2v_turbo_8step_v1.0_768p...` (keyframe /
  t2va) and `...ref2v_turbo_4step_v0.1...` (reference image→video).
- **diffusers(main) + peft** in `/opt/pytorch`.

**Then start serving** (and on every subsequent boot, this is all you need):
```bash
# preferred: systemd unit (auto-starts on reboot)
sudo cp infra/openminimax-serve.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now openminimax-serve
# or, ad hoc: nohup bash serving/run_h3_server.sh > /mnt/models/logs/h3_server.log 2>&1 &

curl -s http://127.0.0.1:30010/health            # wait for {"ready":true} (~100s warm load)
```

The systemd `openminimax-worker` (`SGLANG_STEPS=9`) drains the DynamoDB queue against
`:30010` unchanged. Both units expect the repo deployed to `/opt/openminimax/`.

## Performance (measured, real Turbo, pixel-verified)

Cost scales super-linearly with clip length (bigger latent thrashes the 46GB ceiling,
per-step ~30s → ~130s).

### fl2va (8-step Turbo LoRA, keyframe / t2va)

| frames | video length | wall-clock | **per second of video** |
|---|---|---|---|
| 124 | 5.2s | ~207s | ~40s |
| 158 | 6.6s | ~300s | ~46s |
| 294 | 12.3s | ~1030s | ~84s |
| 345 | 14.4s | ~1100s | ~76s |

### ref2va (4-step Turbo LoRA, reference image→video) — measured 2026-08-31

1376×768, `ref_short_edge=1024`, VAE resident, 2–3 image references per shot. Six real
reference-to-video shots:

| frames | video length | refs | wall-clock | **per second of video** |
|---|---|---|---|---|
| 243 | 10.1s | 2 | 358s | 35.4s |
| 294 | 12.2s | 3 | 498s | 40.7s |
| 294 | 12.2s | 3 | 499s | 40.7s |
| 328 | 13.7s | 2 | 571s | 41.8s |
| 345 | 14.4s | 3 | 668s | 46.5s |
| 345 | 14.4s | 3 | 666s | 46.3s |

**Weighted average: ~42s of compute per second of video** (range 35–47s; grows with clip
length as the latent rides the 46GB ceiling). Interestingly ref2va (~42s/s) is roughly
**2× faster than fl2va at the same 12s length** (~84s/s) because ref2v is a 4-step LoRA
(`num_inference_steps=5`) vs fl2v's 8-step (steps=9) — the halved step count more than
offsets the extra reference-image encoding. Reference *count* (2 vs 3) barely moves it;
frame count dominates. Cold-start warm-load ~100–130s (one-time).

Peak GPU ~24–45GB/card (rides the edge on long clips; transient `expandable_segments`
remap warnings are non-fatal). **The slow part is block-streaming from host RAM.** The
speed lever if this ever needs to scale: resident weights on **80GB cards** (H100/H200)
or a working int8-resident path — either removes the per-step offload and is ~10–20×.
Only cuda:0 is used — the other three L40S sit idle (single-card offload, not multi-GPU).

## Security

The shim listens on **loopback only** (`127.0.0.1:30010`) — never a public port. The
security group in [`../infra`](../infra) allows inbound `:30010` only from the gateway's
SG; the GPU box has no public IP and no SSH (SSM only). See [`../docs/PLAN.md`](../docs/PLAN.md) §7.
