# openminimax — self-hosted, MiniMax-compatible video + image API

Self-host open-weights models on your own GPU and expose them behind a **MiniMax-compatible
API** (MiniMax v2 video protocol + a MiniMax v1 image endpoint, gated by self-issued/-verified
API keys). Point any MiniMax client at it with `MINIMAX_BASE_URL` and **change no code**.

- **Video** — open-source **MiniMax-H3** (text-to-video / image-to-video, 33B) + a step-distilled Turbo LoRA.
- **Image** — open-source **FLUX.2-dev** fp8 (text-to-image), on the same box's otherwise-idle GPU.
- **Deploy to your own account** — the full IaC is parameterized with `AWS_REGION` / `STACK` etc.; account-specific values like bucket names are auto-assembled by CloudFormation via `${AWS::AccountId}`; just replace `<your-account-id>` / `<your-region>` in the example commands with your own.
- **GPU** `g6e.12xlarge` (4×L40S, 192GB) — requires On-Demand G/VT quota (~768 vCPU); this instance type's capacity fluctuates by AZ, and the scripts in `infra/` will automatically search for capacity across AZs/regions.
- **Integration** MiniMax-compatible gateway + self-signed key verification → **any MiniMax client works unchanged**.
- **Status**: ✅ video (Turbo) and image (FLUX.2) both run end-to-end and are pixel-level verified. The video inference framework is **diffusers** (see the important note in §1 below).

> This README is the **overview from zero to production** (for whoever operates it). To dig deeper:
> how to call the API (issue keys, submit/poll) → [`docs/API.md`](docs/API.md);
> deploy it into a fresh AWS account (agent-friendly playbook) → [`AGENTS.md`](AGENTS.md);
> architecture & design decisions → [`docs/PLAN.md`](docs/PLAN.md).
>
> **Before first deploy, read**: [`.env.example`](.env.example) (list of all environment variables) · [`LICENSE`](LICENSE) (code is MIT)
> · [`NOTICE.md`](NOTICE.md) (⚠️ **MiniMax-H3 and FLUX.2 are under their own licenses with restrictions; you must pass compliance before serving externally**).

---

## 1. What is used (model / LoRA / framework)

> **⚠️ Important correction (2026-08-30): the inference framework is no longer SGLang.** Early on, `sglang serve` was used to load the
> Turbo LoRA, and single-machine tests produced clear clips (see the red fox in `serving/phase0_samples/`), but **it produced noise and was unstable when wired end-to-end through the gateway**;
> the root cause was the LoRA loading method (raw PEFT did not do QKV de-interleave; not an SGLang bug — see §3 below).
> Now it uses the **diffusers modular pipeline**, with the LoRA loaded via `pipe.load_lora_weights()` (which does the
> critical QKV de-interleave and stably produces clear clips), wrapped into an HTTP shim that replicates the `/v1/videos` protocol (:30010),
> with **zero changes to the gateway/worker**. Details in [`serving/README.md`](serving/README.md).

| Item | Choice | Notes |
|---|---|---|
| **Video model** | `MiniMaxAI/MiniMax-H3` (33B dense, image-text-to-video, BF16 open-source weights) | HF has **re-exported it to a modular layout** (root-level `modular_model_index.json`); we serve the **~196GB** root layout (DiT **61.7GB** + a second `transformer_ref` **62GB** for ref2v + Qwen3-VL conditioner **62.1GB** — none fits a single 46GB card). The repo also ships redundant self-contained `FL2VA/`+`Ref2VA/` copies (270GB) that `download_model.sh` excludes |
| **Video acceleration LoRA** | `lightx2v/Minimax-h3-Turbo` **8-step v1.0 768p** (Apache-2.0), a step-distilled Turbo LoRA | Cuts denoising from ~50 steps down to **8 steps**. **Must be loaded with `load_lora_weights`** (which de-interleaves and fuses QKV per head), otherwise noise |
| **Image model** | **`FLUX.2-dev` fp8** (black-forest-labs, official diffusers `Flux2Pipeline`) | The video model only fully occupies 1 card, leaving the other 3 L40S idle → **FLUX.2 fp8 resident on cuda:2** (~31GB), running **truly in parallel** with video and with zero switching. Verified 2026-09-02 to produce clear images (see §3) |
| **Inference framework** | **diffusers 0.41.dev** (`ModularPipeline` + `MiniMaxH3Blocks`) + **bf16 block-stream group offload**, wrapped into an HTTP shim exposing `POST /v1/videos` (:30010); images are a separate `Flux2Pipeline` shim (:30020) | Neither MiniMax v2 nor strictly OpenAI — the gateway does the protocol translation. SGLang has been deprecated for serving |
| **VRAM strategy** | Video bf16 + **block-level streamed offload** (weights reside in 372GB host RAM, streamed onto the card block by block); images fp8 **resident**, not offloaded | Video's 46GB single card cannot hold any full component, so it is not resident; the image fp8 at ~31GB fits and stays resident with zero latency. int8 video residency is blocked by a torchao/torch version bug; FSDP2 is a recipe for 8×H100 |

**Why g6e.12xlarge (4×L40S)**: 372GB of host RAM can hold all bf16 weights for streamed offload,
and the 4 cards have spare VRAM to run activations. `g6e.16xlarge` is **1×L40S** (a downgrade). **Cost**: block-streaming is slow
(see §3); to go much faster you need an **80GB card (H100/H200)** so weights can stay resident and offload is removed. See [`docs/PLAN.md`](docs/PLAN.md) §5.

---

## 2. What capabilities are supported (and what is not)

> **In one sentence: video uses the self-hosted MiniMax-H3 (+Turbo LoRA), images use the self-hosted FLUX.2-dev fp8.** Both run on
> different L40S cards of the same g6e.12xlarge, exposed externally through the same MiniMax-compatible gateway.

**Supported**:
- **Text-to-video / image-to-video** (MiniMax-H3; reference images are passed in order in `content[]`, and the prompt refers to them as "reference image N").
- **True 768P** (1344×768), 24fps, H.264+AAC, **clip length 4–15 seconds** (integer seconds, specified by the request `duration`).
- **Text-to-image** (FLUX.2-dev fp8; `POST /v1/image_generation`). fp8 resident on an idle card, in parallel with video; on 2026-09-02
  pixel-verified as clear on a real machine, ~37.5s/image (1024×768, base 28 steps; can be sped up further once the Turbo LoRA is wired in).
- **The calling convention is identical to MiniMax's own API** — video is MiniMax v2
  (`POST /v2/video_generation` + `GET /v2/query/video_generation/{id}`), image is MiniMax v1
  (`POST /v1/image_generation`), all with `Authorization: Bearer <key>`. Any MiniMax client
  works by just setting `MINIMAX_BASE_URL` to this gateway — **zero code changes**. See the
  official reference at <https://www.minimax.io/platform/document/video_generation> and our
  own [`docs/API.md`](docs/API.md) for the exact request/response shapes.

**Not supported (known and accepted trade-offs, see [`docs/PLAN.md`](docs/PLAN.md) §4)**:
- ❌ **2K video output**: the cloud-side `H3-Regenerate-2K` and `H3-Context-IR` are not open-sourced, so self-hosting cannot obtain them; only 768P.
- ⚠️ **Image generation uses FLUX.2, not MiniMax `image-01`** — the interface is compatible with MiniMax `/v1/image_generation`,
  but the underlying engine is the self-hosted FLUX.2 (the `image-01` weights are not open-sourced, so self-hosting cannot obtain them). Style/capabilities differ from the official `image-01`.

---

## 3. Performance (2026-08-30 real-machine benchmarks, diffusers Turbo, pixel-verified as clear)

> The 92s/177s in the Phase-0 report are numbers from the **SGLang path** (samples were clear, but that route is deprecated for serving).
> Below are the numbers for the **current production diffusers Turbo** (also clear). **Time grows super-linearly with clip length** (large latents are
> repeatedly remapped at the 46GB edge, with each step rising from ~30s to ~130s):

**fl2va (8-step Turbo, first-frame / text-to-video):**

| Frames | Video duration | Wall clock per clip | **Approx. compute per second of video** |
|---|---|---|---|
| 124f | 5.2s | ~207s | ~40s |
| 158f | 6.6s | ~300s | ~46s |
| 294f | 12.3s | ~1030s | ~84s |
| 345f | 14.4s | ~1100s | ~76s |

**ref2va (4-step Turbo, reference-image-to-video, benchmarked 2026-08-31 on 6 real reference-to-video shots):**

| Frames | Video duration | Reference images | Wall clock per clip | **Approx. compute per second of video** |
|---|---|---|---|---|
| 243f | 10.1s | 2 | 358s | 35.4s |
| 294f | 12.2s | 3 | 498s | 40.7s |
| 294f | 12.2s | 3 | 499s | 40.7s |
| 328f | 13.7s | 2 | 571s | 41.8s |
| 345f | 14.4s | 3 | 668s | 46.5s |
| 345f | 14.4s | 3 | 666s | 46.3s |

- In one sentence: **ref2va averages about 42 seconds of compute per second of video** (range 35–47s, growing with clip length); an fl2va 12s shot is about 84s per second.
- **ref2va is about twice as fast as fl2va of the same length** — ref2v uses 4-step (steps=5) vs fl2v's 8-step (steps=9), and halving the step count outweighs the extra overhead of reference-image encoding. The **number** of reference images (2 vs 3) has little effect; the **frame count** is the main factor.
- Peak VRAM ~24–45GB/card (long clips hug the 46GB limit, with occasional remap warnings that are not fatal). ref2va squeezes into 46GB via `ref_short_edge=1024` + a resident VAE (2048 would OOM).
- The root of the slowness is **block-stream offload** (weights stream from host RAM onto the card every step). **Only one card, cuda:0, is used**, while the other 3 L40S are idle (single-card offload, not multi-card parallelism).
- ⚠️ Pure inference wall clock (single request); the gateway queue is **serial**, so multiple shots stack up in the queue. Cold start ~100–130s (one-time).
- **Speed-up levers**: switch to an **80GB card** to keep weights resident and remove offload (~10–20×); or int8 residency (once the torchao version is aligned).

**Images (FLUX.2-dev fp8, benchmarked 2026-09-02):** 1024×768, 28 steps, **~37.5s/image**, peak VRAM ~31GB (fp8 resident on
cuda:2, **not offloaded**, so much faster than video). Cold start ~500s (EBS lazy hydration + fp8 quantization, one-time); afterwards it stays warm and resident,
with steady state ~37.5s per image. Runs **truly in parallel** with video (different cards), without contending for VRAM. Once the Turbo LoRA is wired in, it can drop further to ~8-12 steps.

### Root cause of the Turbo noise (a full day was spent on this pitfall, worth writing down)
lightx2v's Turbo LoRA was trained against the **original checkpoint's per-head interleaved fused QKV**; the HF re-exported
modular checkpoint uses a **de-interleaved** layout. A raw PEFT `load_state_dict` (which both SGLang and the official
`inference_minimax_h3.py` do) only matches key **names**, landing the LoRA delta on the wrong layout → noise.
Using diffusers' **`pipe.load_lora_weights()`** (whose internal `_convert_non_diffusers_minimax_h3_lora_to_diffusers`
does the de-interleave) resolves it. **It is not SGLang's fault; it was the wrong LoRA loader.**

---

## 4. Components (what is in this repo)

```
openminimax/
├── serving/     # diffusers Turbo video shim (h3_turbo_server.py) + FLUX.2 image shim (flux_image_server.py) + deprecated SGLang scripts
├── gateway/     # MiniMax v2 compatible endpoints + key issuance/verification/rate-limiting/metering;
│                #   app/ shared core (protocol/backend/publish/worker/keys/tasks,
│                #   keys+tasks each have both a SQLite and a DynamoDB version);
│                #   lambdas/ = serverless entry points (authorizer / api / autostop)
├── infra/       # CloudFormation: base (private VPC + locked-down SG + private buckets + least-privilege IAM)
│                #   + serverless (HTTP API + authorizer + DynamoDB + autostop);
│                #   deploy scripts + systemd units for the single-machine route
└── docs/        # API.md (how to call the API) · PLAN.md (architecture & design decisions)
```

**Two deployment routes, reusing the same tested core**:
- **Serverless (recommended)**: the only public-facing surface is **API Gateway (HTTP API + TLS)**, and every route first passes through
  a **Lambda authorizer** that verifies the self-signed `mmh3_` key; submit/poll are per-invocation-billed Lambdas; the GPU scales to zero via
  EventBridge autostop. No always-on ALB / t3.small.
- **Single-machine fallback**: the FastAPI version in `gateway/app/main.py` runs on one EC2 instance (SQLite storage), for local
  development or scenarios that do not use serverless.

Tests: `gateway/` has a full offline test suite (moto for DynamoDB, **no GPU/AWS cost**), covering the wire-level contract and authentication.

---

## 5. From zero to production (setup process)

> ⚠️ The base/serverless layers are nearly zero cost; **once the GPU starts up it bills at ~$10/hr**, and starting it is an explicit approval step.

**① Infrastructure (free, reversible)** — in [`infra/`](infra/):
```bash
export AWS_REGION=us-west-2
./deploy.sh base                              # private VPC/SG/IAM/KMS/empty buckets, no NAT, no compute, ~$0
./deploy_serverless.sh <artifact-bucket>      # API layer (HTTP API + authorizer + DynamoDB), no GPU yet
```

**② Start the GPU instance (after approval)** — requires the literal `APPROVE` to prevent accidental start:
```bash
./deploy.sh gpu APPROVE                        # g6e.12xlarge (4×L40S), ~$10/hr
./deploy_serverless.sh <artifact-bucket> <GpuInstanceId>   # hand the GPU id to autostop
```

**③ One-click deploy on the GPU instance (install deps + download model + install and enable-on-boot two systemd units)** — via SSM (no SSH).
Transfer this repo to the machine (scp/rsync/tar, **git is not needed on the machine**), then run `bootstrap.sh` once from the repo root:
```bash
# One command does it all: install ffmpeg + diffusers/peft (→/opt/pytorch) + cu13 symlinks
#  + download the model (~196GB root modular layout) and the two Turbo LoRAs to the EBS volume
#  + deploy gateway/ + serving/ to /opt/openminimax
#  + write /etc/openminimax.env (control-plane values such as RESULT_BUCKET)
#  + install and enable openminimax-serve + openminimax-worker (start on boot).
#  Idempotent, EBS-aware — later stop/start will not re-download (anything already in /mnt/models is skipped).
cd /opt/openminimax    # or wherever you transferred the repo
sudo RESULT_BUCKET=<your-result-bucket> bash serving/bootstrap.sh
# To ALSO self-host images: FLUX.2-dev is a GATED HF repo — accept its license at
# https://huggingface.co/black-forest-labs/FLUX.2-dev, then add HF_TOKEN=hf_... to the
# command above (video H3 is ungated; without a token the box is video-only).

# First start warms the pipeline (cold-start EBS lazy hydration, ~18min; fast afterwards when warm); wait until ready:
sudo systemctl start openminimax-serve openminimax-worker
curl -s http://127.0.0.1:30010/health          # wait for {"ready":true}
```
> **★ Key point: both units are already `enable`d to start on boot.** After that, on any `stop`/`start` or reboot, the shim and worker
> will start themselves — **send a request and it works, with zero manual steps** (no need to manually bring up the shim again). That first cold start is slow because the just-
> started gp3 volume must lazily hydrate ~196GB of weights from S3, a one-time thing; once warm, a warm load is ~100s.
> Weights and dependencies all land on the EBS volume and persist across reboots: `/mnt/models/MiniMax-H3-modular` (checkpoint),
> `/mnt/models/lora/...` (the two Turbo LoRAs), and diffusers+peft in `/opt/pytorch`. Rerunning `bootstrap.sh`
> automatically skips anything already downloaded. All paths can be overridden with environment variables (`H3_MODEL`/`H3_LORA`/`MODEL_PATH`, see
> [`.env.example`](.env.example)).
> ⚠️ The old SGLang `serve_h3.sh` **has been deleted** (it produced Turbo noise end-to-end through the gateway). Production serving is
> `serving/h3_turbo_server.py` (diffusers + `load_lora_weights`), managed by `infra/openminimax-serve.service`
> and started on boot; details in [`serving/README.md`](serving/README.md).

**④ Issue a key** (the worker was installed and auto-started by ③, pulling from the DynamoDB queue, producing clips uploaded to the private S3, and writing back presigned URLs):
```bash
# Issue a key (operator AWS credentials, under gateway/):
KEYS_TABLE=openminimax-serverless-keys AWS_DEFAULT_REGION=us-west-2 python -m app.admin_keys issue --label team-1
```
> The account-specific values (`RESULT_BUCKET`/table names/region) of the worker unit (`infra/openminimax-worker.service`) are read from
> the `/etc/openminimax.env` generated by ③; fixed values like `SGLANG_URL`/`SGLANG_STEPS=9` are hardcoded in the unit, so **the unit itself
> contains no placeholders and is ready to use as-is**.

**⑤ Point a client at it (Phase 3)**: set the client's `MINIMAX_BASE_URL=<ApiEndpoint>`
and hand it a key issued in ④. Any MiniMax-compatible client then works unchanged. See [`docs/API.md`](docs/API.md).

> ⚠️ **The worker's `SGLANG_STEPS` must be 9** (8-step Turbo, off-by-one). This pitfall was hit during wiring:
> a leftover `SGLANG_STEPS=50` in the unit (an early ship-base decision) would override the request's step count → running the 8-step LoRA at 50 steps, slow and mismatched.
> Also, the worker is **single-flight** (one generation at a time — H3 saturates the box), so submitting many jobs at once just makes them queue serially; submit at a measured pace.

---

## 6. Cost and compliance (two unavoidable things)

- **Cost**: base ~$0 · gateway tier +t3.small (~$15/mo) + NAT (~$32/mo) · GPU **~$10/hr**.
  The real money-saving lever is **autostop scaling to zero** (start/stop by queue state, never interrupting a generation), see
  [`docs/PLAN.md`](docs/PLAN.md) §2 and [`infra/README.md`](infra/README.md).
- **License**: **MiniMax H3 Community License** (custom, non-OSI). Serving externally counts as **commercial use**, with a
  **regional application (US/EU/UK/KR only)**. ⚠️ **You must pass this before opening it to users** (see [`docs/PLAN.md`](docs/PLAN.md) §4.3).
- **Security baseline** (mandatory, fixed in IaC): SGLang :30010 is **never exposed to the public internet**, only allowing inbound from the gateway SG; the GPU instance
  has no public IP and no SSH (uses SSM); the only public-facing surface is API Gateway; result buckets are private + presigned URLs; least-privilege IAM.
  Item by item in [`docs/PLAN.md`](docs/PLAN.md) §7.
