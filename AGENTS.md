# AGENTS.md — deploying openminimax into a fresh AWS account

This file is written for an AI agent (or a human) who has this repo and needs to stand
up the whole service — a self-hosted, MiniMax-v2-compatible **video + image** generation
API — in **its own AWS account**, from nothing. Follow the phases in order. Each phase
lists the exact command, what it costs, and how to verify it before moving on.

**What you are deploying (so you can explain it):**
- **Video** = self-hosted `MiniMaxAI/MiniMax-H3` (33B i2v/t2v) + lightx2v Turbo LoRA, on diffusers.
- **Image** = self-hosted `FLUX.2-dev` fp8 (text-to-image), on the same box's idle GPU.
- **Front door** = one HTTPS API Gateway that speaks the MiniMax v2 protocol
  (`POST /v2/video_generation`, `GET /v2/query/video_generation/{id}`) + a MiniMax
  `image-01`-shaped `POST /v1/image_generation`, gated by a Lambda authorizer that
  validates self-signed `mmh3_` keys. Clients set `MINIMAX_BASE_URL` to it and change no code.
- **Compute** = one `g6e.12xlarge` (4×L40S). Video shim on cuda:0, FLUX on cuda:2, in parallel.

---

## Ground rules for the agent (read first)

1. **NEVER hardcode account-specific values.** Account id, bucket names, and the API
   endpoint are all discovered from stack outputs at deploy time. Do not paste real IDs
   into committed files. This repo has been scrubbed clean — keep it that way.
2. **The GPU costs ~$10/hr.** `infra/deploy.sh gpu` refuses to run without the literal
   arg `APPROVE`. Only launch it when the user has approved spend. Stop it when idle
   (autostop does this automatically after `IDLE_STOP_MINUTES`).
3. **License gate.** MiniMax-H3 is a custom **community license with regional limits
   (US/EU/UK/KR only)** — see `NOTICE.md`. **FLUX.2-dev is a GATED HF repo** with its own
   license: image gen requires accepting it at
   <https://huggingface.co/black-forest-labs/FLUX.2-dev> and supplying an `HF_TOKEN`
   (Phase 3b). Video (H3) is ungated. If the user is exposing this as a service, confirm
   they've cleared BOTH licenses BEFORE opening it up.
4. **Region.** Everything defaults to `us-west-2`; override with `AWS_REGION`. The GPU can
   roam to any region with g6e capacity (the control plane stays fixed) — see `docs/PLAN.md`.
5. **Verify by pixels, not status.** A `completed` job with a valid mp4/png can still be
   noise. After any generation smoke test, extract a frame and LOOK at it. This is the
   single most important lesson in this repo's history (see README §3 for the root cause).

---

## Phase 0 — prerequisites

- AWS credentials with permission to deploy CloudFormation (VPC, IAM, Lambda, DynamoDB,
  S3, API Gateway, EC2). Confirm the caller: `aws sts get-caller-identity`.
- An **On-Demand G/VT vCPU quota** big enough for a `g6e.12xlarge` (48 vCPU) in the target
  region. If quota or capacity is missing, `deploy.sh gpu` will fail — see `docs/PLAN.md`
  for the capacity-hunt scripts (`infra/probe_capacity.sh`, `infra/deploy_gpu_global.sh`).
- The model weights are pulled from Hugging Face on the box (Phase 3). Ensure the box's
  egress can reach HF (the roaming GPU stack has a public IP for egress only).
- **Ask the user up front whether they want image gen (FLUX.2).** If yes, they must accept
  the FLUX.2-dev license and give you an **HF access token** BEFORE Phase 3 — you pass it as
  `HF_TOKEN` to `bootstrap.sh` (Phase 3b). Video-only needs no token. Never commit the token.

---

## Phase 1 — control plane (free / near-free, reversible)

```bash
cd infra
export AWS_REGION=us-west-2                       # or your region

./deploy.sh base                                  # VPC, SGs, IAM, KMS, empty private bucket. ~$0
./deploy_serverless.sh <your-artifact-bucket>     # HTTP API + authorizer + DynamoDB + autostop
```
- `<your-artifact-bucket>` is any S3 bucket you own for holding the Lambda zip (create one
  if needed: `aws s3 mb s3://<name> --region $AWS_REGION`).
- **Verify:** `deploy_serverless.sh` prints `ApiEndpoint`. Save it — it's your
  `MINIMAX_BASE_URL`. Hitting it with no key must return **401**:
  `curl -si <ApiEndpoint>/v2/video_generation -X POST | head -1`  → `HTTP/2 401`.

---

## Phase 2 — launch the GPU (approved spend, ~$10/hr)

```bash
cd infra
./deploy.sh gpu APPROVE                            # g6e.12xlarge (4×L40S). Sweeps AZs for capacity.
# capture the instance id from the printed outputs, then hand it to autostop:
./deploy_serverless.sh <your-artifact-bucket> <GpuInstanceId>
```
- The GPU box has **no SSH and no public inbound** (SSM-only, dial-out). Operate it with
  `aws ssm send-command` / Session Manager.
- **Verify:** wait for `aws ec2 wait instance-status-ok --instance-ids <id>` (2/2), then
  confirm it's an SSM managed node: `aws ssm describe-instance-information`.

---

## Phase 3 — one-shot box setup (deps + weights + boot-ready units)

Ship this repo to the box (the box does **NOT** need git — copy it over):
```bash
# from your machine, e.g. tar over SSM or scp via a bastion; end state: repo at /opt/openminimax on the box
```
Then run the single bootstrap, as root, from the repo dir on the box (via SSM):
```bash
cd /opt/openminimax
sudo RESULT_BUCKET=<ResultBucketName-from-base-stack> bash serving/bootstrap.sh
```
`bootstrap.sh` is idempotent and EBS-aware. It: installs ffmpeg + diffusers/peft into
`/opt/pytorch`, symlinks cu13 libs, **downloads the ~196GB H3 root modular checkpoint + Turbo
LoRA** to `/mnt/models` (skips what's present), deploys `gateway/`+`serving/` to
`/opt/openminimax`, writes `/etc/openminimax.env` (RESULT_BUCKET/tables/region), and
**installs + `enable`s the systemd units** so they auto-start on every boot:
- `openminimax-serve` — the H3 video shim on `:30010` (cuda:0).
- `openminimax-worker` — drains the video queue → shim → S3.
- (optional) `openminimax-flux` + `openminimax-image-worker` — **only if** FLUX.2 weights
  are present at `/mnt/models/flux2-official` (see Phase 3b). Bootstrap detects and wires
  them automatically; otherwise it deploys video-only.

Start them (first warm load is slow — EBS lazily hydrates the weights, ~2–18 min):
```bash
sudo systemctl start openminimax-serve openminimax-worker
curl -s http://127.0.0.1:30010/health          # wait for {"ready":true}
```
**★ Boot guarantee:** because the units are `enable`d, a later `stop`/`start` of the
instance brings the box back **request-ready with zero manual steps**. Do not rely on any
nohup process.

### Phase 3b (optional) — image generation (FLUX.2)

Image gen is optional. **`black-forest-labs/FLUX.2-dev` is a GATED Hugging Face repo**, so
before it can be downloaded you MUST, as the account's HF user:
1. Open <https://huggingface.co/black-forest-labs/FLUX.2-dev> and **accept the license**
   (one click; access is auto-granted).
2. Create an **access token** at <https://huggingface.co/settings/tokens> (read scope).

Then re-run bootstrap **with `HF_TOKEN` set** — it pulls the official diffusers FLUX.2
layout (`transformer/ text_encoder/ vae/ tokenizer/`) to `/mnt/models/flux2-official` on
the EBS volume and enables `openminimax-flux` + `openminimax-image-worker`:
```bash
sudo HF_TOKEN=hf_xxx RESULT_BUCKET=<ResultBucketName> bash serving/bootstrap.sh
```
(Video H3 is **ungated** — needs no token. Without `HF_TOKEN` the box stays video-only.)
Then start + verify:
```bash
curl -s http://127.0.0.1:30020/health          # {"ready":true}; warm load ~500s (fp8 quant, one-time)
```
Measured on 4×L40S: fp8 resident on cuda:2 (~31GB), ~37.5s per 1024×768 image, in
parallel with video on cuda:0. (Already have the FLUX dir on EBS from a prior run? Bootstrap
detects it and wires the units without re-downloading — no token needed then.)

---

## Phase 4 — issue a key and wire the client

```bash
cd gateway
KEYS_TABLE=openminimax-serverless-keys AWS_DEFAULT_REGION=$AWS_REGION \
  python -m app.admin_keys issue --label team-1
#   -> prints a plaintext mmh3_... key ONCE (only the hash is stored; you cannot recover it later).
```
Give that key to the client and set the client's `MINIMAX_BASE_URL=<ApiEndpoint>`. Done —
the client calls it exactly like the MiniMax API.

**End-to-end verify (do this before declaring success):**
1. `POST <ApiEndpoint>/v2/video_generation` with `Authorization: Bearer <mmh3_ key>` and a
   MiniMax v2 body → expect `{task_id}`.
2. Poll `GET <ApiEndpoint>/v2/query/video_generation/{task_id}` until `succeeded` with a
   presigned `content.url`.
3. Download the mp4, extract a frame, and **look at the pixels** — confirm it matches the prompt.

---

## Operating notes (hand these to whoever runs it)

- **Autostop** (`openminimax-serverless-autostop`, EventBridge every 1 min) stops the box
  after `IDLE_STOP_MINUTES` of an empty queue. It measures **queue state, not CPU**, so it
  never interrupts a running generation. Tune: `aws lambda update-function-configuration
  --function-name openminimax-serverless-autostop --environment Variables={IDLE_STOP_MINUTES=60}`.
- **There is NO auto-START.** After autostop stops the box, a new request enqueues fine but
  won't run until someone `start`s the instance again. Starting it:
  `aws ec2 start-instances --instance-ids <id>`; wait 2/2; the enabled units come up on
  their own. (Cold start warms the weights, several minutes.)
- **Worker step count MUST be `SGLANG_STEPS=9`** (8-step Turbo off-by-one). It's baked into
  the unit. Never set 50 (that's the deprecated base path — slow + mismatched).
- **The worker is single-flight** (one generation at a time — H3 saturates the box). Bulk
  submitting many jobs at once just makes them queue serially; submit at a measured pace.
- How to call the API (issue keys, submit/poll, error codes): `docs/API.md`. Full
  architecture + decision history: `docs/PLAN.md`. Serving internals: `serving/README.md`.
