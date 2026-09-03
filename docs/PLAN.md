# Self-Hosted MiniMax-H3 Video Service — Design Document

> **📌 Status note (2026-08-30): This is the original design/decision document from 2026-08-28, retained as a decision history.**
> The overall direction (self-host H3, g6e.12xlarge, MiniMax-compatible gateway + self-signed key, serverless entry point, security baseline)
> **all holds and has been shipped**. **One key path has been overturned**: §4.4 planned to use "SGLang serve + Turbo LoRA",
> but what it actually produced was noise — the genuinely working approach is **diffusers modular pipeline + `load_lora_weights`** (see the correction block in §4.4
> and [`../serving/README.md`](../serving/README.md)). The "SGLang inference service" in the §2 architecture diagram should now be read as
> "diffusers Turbo shim, likewise exposing `/v1/videos` (:30010), transparent to the gateway".
>
> ---
>
> Goal: self-host a GPU instance running the open-source **MiniMax-H3** text-to-video/image-to-video model, **exposing externally a calling convention
> identical to minimax.io** (MiniMax v2 video protocol + self-issued/self-verified API keys),
> providing a self-hosted, MiniMax-compatible generation API,
> so any MiniMax-compatible client can use your self-hosted service "just like calling the MiniMax website".
>
> Recorded: 2026-08-28 · target account `<your-account-id>` · region `us-west-2` (example)
> Decisions locked: ① we will self-host H3; ② GPU choice is `g6e.12xlarge` (4×L40S, 192GB);
> ③ integration = MiniMax-compatible gateway + self-signed key verification (zero code changes required in any MiniMax-compatible client);
> ④ **the image path (image-01) is deferred for now** (see §4.1);
> ⑤ **first land the code/infra as IaC (free, reversible); only spin up the GPU instance with one click after approval**;
> ⑥ **AWS security: SGLang exposes no public ports**, only allowing private-network access from the gateway (see §7).

---

## 0. One-Line Conclusion

Because any typical MiniMax client (e.g. an integration like `app/integrations/providers/minimax.py`) already supports pinning the MiniMax host to any address via the environment variable
`MINIMAX_BASE_URL`, and it already uses
`Authorization: Bearer <key>` to call `/v2/video_generation` + poll `/v2/query/...`,
all we need is to **build a gateway that replicates the MiniMax v2 protocol and does its own key verification**, then point the
client at it via a single environment variable — **not a single line of client business code needs to change**.

The exact request/response shapes are in [`API.md`](API.md) (replicate against it when building the gateway).

---

## 1. Verified Facts (must know before starting)

| Item | Conclusion | Source |
|---|---|---|
| Is H3 open source | ✅ Yes. `MiniMaxAI/MiniMax-H3`, 33B dense, image-text-to-video, BF16 weights already released | HuggingFace org page |
| Full version vs. self-hostable | ⚠️ **Only H3-Base (768P) can run locally**; `H3-Context-IR`, `H3-Regenerate-2K` (2K output) are **cloud API only**. Self-hosting cannot get 2K | HF model card / SGLang cookbook |
| Full deployment size | **108GB checkpoint**, of which the DiT alone is **61.7GB (does not fit on a single card)** | SGLang cookbook |
| Official reference hardware | H100×4 (80GB, TP2) / H200×4 (141GB) / B200×8 (192GB) resident | SGLang cookbook |
| g6e.12xlarge feasibility | 4×L40S = **192GB > 108GB**, can stay resident after sharding. But L40S uses PCIe (no NVLink), so multi-card communication is slower than the H100 reference → **it runs, slower than the reference** | spec estimate |
| Inference service protocol | SGLang exposes `POST /v1/videos` (port 30010), with fields `prompt/task(t2va/fl2va/ref2va)/target{short_edge,aspect_ratio,duration_seconds}/num_inference_steps` etc. **Not the MiniMax v2 protocol**, nor strictly OpenAI-compatible | SGLang cookbook |
| 768P per-clip time | Official docs **do not give** the 768P wall clock (all step-times are anchored to 480P). Consumer-card 480P example ~250s/clip → **must be measured firsthand**; this is the make-or-break number | SGLang cookbook |
| GPU quota | ✅ This account's On-Demand G/VT = **768 vCPU** (g6e.12xlarge needs only 48), **no request needed**. Spot G/VT = 64 vCPU (enough for one instance) | `service-quotas` (checked live 2026-08-28) |
| License | **MiniMax H3 Community License** (custom, non-OSI). Providing an external service = commercial use, **with a regional application form (US/EU/UK/KR only)**. ⚠️ Must clear compliance before going public | HF model card |

---

## 2. Architecture

```
Client
   │  (select the MiniMax provider, enter the [issued-by-you] key)
   ▼
Any MiniMax-compatible client  ──set MINIMAX_BASE_URL to point at your gateway──┐
   │  POST /v2/video_generation  (Bearer <your key>)      │
   │  GET  /v2/query/video_generation/{task_id}           │
   ▼                                                      │
┌─────────────────── Your VPC (us-west-2) ───────────────┘
│  Gateway entry: private, no public port exposed (see §7)
│  ① MiniMax-compatible gateway (FastAPI, CPU side, ~300 lines)
│     - verify Bearer key (self-issued; revocable/rate-limited/metered)
│     - POST /v2/video_generation → enqueue, return { task_id }
│     - GET  /v2/query/video_generation/{id}
│              → { task: { status, content: { url } } }
│     - result mp4 uploaded to S3, content.url returns an S3 presigned/CloudFront link
│         │ internally translated into the SGLang /v1/videos protocol
│         ▼
│  ② SGLang inference service (g6e.12xlarge, 4×L40S, /v1/videos, port 30010)
│     listens on private network only; SG allows 30010 from the gateway only; stop the whole machine when scaled to zero
│     sglang serve --model-path MiniMaxAI/MiniMax-H3 \
│       --num-gpus 4 --tp-size 2 --ulysses-degree 2 \
│       --host 0.0.0.0 --port 30010 --performance-mode speed
└─────────────────────────────────────────────────────────
```

### 2.1 Integration Form: Fully Serverless (adopted) + API Gateway + Lambda key verification

The external entry point uses **API Gateway (HTTP API, the only public face, TLS enforced)**, and **uses a Lambda
authorizer to verify clients' self-signed keys** — this satisfies the requirement of "only expose API Gateway externally, verify keys in
Lambda". Aside from the GPU instance that is started/stopped on demand, nothing is resident.

```
Client → Any MiniMax-compatible client (MINIMAX_BASE_URL = API Gateway address)
                         │  Authorization: Bearer <the mmh3_ key you issued>
                         ▼
   ┌────────────── API Gateway (HTTP API, only public face, TLS) ──────────────┐
   │  each route first passes the Lambda authorizer (verify mmh3_ key, result cached ~300s)     │
   │        │ isAuthorized=true + context{keyPrefix, rateLimit}         │
   │        ▼                                                           │
   │  submit/poll integration Lambda (reads auth context, no re-verification)               │
   │     POST /v2/video_generation → rate-limit+meter → enqueue to DynamoDB          │
   │     GET  /v2/query/.../{id}    → read DynamoDB → {task:{...}}         │
   └───────────────────────────┬───────────────────────────────────────┘
                     DynamoDB (keys table + tasks queue, the shared single source of truth)
                                │  (GPU machine accesses via the DynamoDB VPC gateway endpoint)
                                ▼
   GPU-machine worker (systemd, private network) pulls the queue → SGLang produces the clip → mp4 uploaded to private S3 →
     writes back to DynamoDB (content.url = presigned URL); EventBridge triggers every minute the
     autostop Lambda: start the GPU when there is work, stop when idle long enough (by queue state, never interrupting generation)
```

Why this is the most economical: client polling happens roughly every ~5s for a few minutes → the authorizer's **result cache** means
one key is verified only once every few minutes; submit/poll are pay-per-invocation Lambdas; **no resident t3.small,
no ALB**; the GPU scales to zero via autostop. The management plane (issuing keys) is an **IAM-authorized CLI**
(`app.admin_keys`), with **no public admin route**, one fewer attack surface.

> Single-machine fallback: the FastAPI version in `gateway/app/main.py` is still there and can run the same
> endpoints on a single EC2 instance (SQLite storage), for local development or scenarios where you don't want serverless. Both paths reuse the same
> already-tested `protocol/backend/publish/worker`, and key encryption/decryption also shares the module-level
> functions in `app.keys`, so self-signed keys are byte-identical across both paths.

### Why a MiniMax-compatible client needs zero code changes
The key contract of a typical MiniMax client integration (e.g. a `minimax.py` provider; details in [`API.md`](API.md)):
- host is pinned by `MINIMAX_BASE_URL` (once set there is no fallback);
- video submission: `POST /v2/video_generation`, reads `resp["task_id"]`;
- polling: `GET /v2/query/video_generation/{id}`, reads
  `task.status` ∈ `queued|running|succeeded|failed|cancelled`, and on success reads
  `task.content.url` (a direct time-limited URL, no secondary fetch);
- authentication: `Authorization: Bearer <caller's key>`, the key is **user-supplied** and passed in per call.

→ Your gateway only needs to **faithfully replicate the request/response shapes of these few endpoints** above, and the client is none the wiser.

---

## 3. Phased Rollout

### Phase 0 — Just get inference working (verify this card can produce clips, and how slow)
> ⚠️ This phase **starts billing at ~$10/hr the moment the machine is up**. Per decision ⑤, first write the startup scripts under §infra
> (including security groups + auto-stop) as IaC and review them, **and only execute when you say go**.
- Launch `g6e.12xlarge` using the **Deep Learning OSS Nvidia AMI**; attach EBS **gp3 ≥500GB**
  (the served H3 root modular checkpoint is ~196GB — not the 108GB the old SGLang cookbook
  quoted; download_model.sh excludes the repo's redundant FL2VA/Ref2VA copies, 270GB more —
  + optional FLUX.2 ~110GB + transient HF cache + output headroom. 800GB recommended. The
  original 108GB figure below is retained as decision history.).
- `pip install "sglang[all]"` (or the cookbook version) → `sglang serve ...` to bring up
  H3-Base (768P).
- `curl POST /v1/videos` to produce a sample clip, and **measure firsthand the per-clip wall clock + VRAM usage**.
  → This is the key number deciding whether to keep investing in Phase 1/2 (official docs don't give 768P).

### Phase 1 — MiniMax-compatible gateway (the key-verification layer you want)
- A small FastAPI service (~300 lines) implementing:
  - `POST /v2/video_generation`: verify Bearer → enqueue → return `{ "task_id": ... }`;
  - `GET /v2/query/video_generation/{id}`: return
    `{ "task": { "status": ..., "content": { "url": ... } } }`;
  - `POST /v1/image_generation`: **per decision ④, directly return an explicit error** (see §4.1),
    not fake success, not silently hang.
  - Bearer key table (issue / revoke / rate-limit / meter), stored in either SQLite or DynamoDB.
- Because H3 generation is slow, the gateway maintains an internal **asynchronous task queue**; after SGLang produces a clip it uploads the mp4 to S3,
  and `content.url` returns the S3/CloudFront link (same semantics as MiniMax's time-limited direct URL).

### Phase 2 — Cost control (the real lever for cost-effectiveness)
> **GPUs are billed by the hour; idling burns money.** Which card tier you pick matters less than "scaling to zero".
- **Auto start/stop**: MiniMax-compatible clients typically poll the video path asynchronously
  (a Step Functions poller can tolerate ~25 minutes), which comfortably absorbs a cold start.
  Keep a cheap resident machine (t3.small) for the gateway; only `ec2 start` the GPU machine when there is a job, and `ec2 stop`
  after N minutes idle.
- Optional **Spot**: g6e Spot saves another ~65% (Spot quota of 64 vCPU is enough for one instance); caching weights to
  EBS reduces the cost of re-pulling after an interruption.

### Phase 3 — Integrate a downstream client
- Add the environment variable `MINIMAX_BASE_URL=https://<your-gateway-domain>` to any downstream
  MiniMax-compatible client (e.g. set it on its backend and redeploy).
- Hand out your issued keys to callers. Done.

---

## 4. Decisions / Pitfalls to Know Up Front

### 4.1 The image path (image-01) — decision: **deferred for now**
`MINIMAX_BASE_URL` pins both the image and video paths at once. H3 does video only; images use
`image-01` (`POST /v1/image_generation`).

**This project's decision: defer for now.** The gateway only implements the two video endpoints; `/v1/image_generation`
returns an **explicit error response** (rather than silently hanging or faking success), so that any client's image path
"fails fast, with a readable reason". This means:
- pure video scenarios work;
- flows involving image generation will fail — this is a **known and accepted** trade-off.
- If we later want to fill the gap: two options — (a) the gateway **passes through** image requests to the real
  minimax.io using the operator's real minimax key; (b) the client's image provider switches to fal.ai. Neither is done this round.

### 4.2 Only 768P is possible
Cloud 2K (`H3-Regenerate-2K`) is not open-sourced, so self-hosting cannot get it. Clients that pin the `MINIMAX_H3`
spec's `resolution` to `768P` are already consistent with self-hosted capability; no change needed.

### 4.4 Using Turbo LoRA to make H3 runnable on 4×L40S (verified, key)
> **⚠️ Correction (2026-08-30): the landing approach chosen in this section, "SGLang serve stacked with Turbo LoRA", is wrong — it produces noise.**
> SGLang (and the official `inference_minimax_h3.py`) load LoRA with raw PEFT `load_state_dict`, matching only key
> names, without doing **QKV de-interleave** (the LoRA was trained on the original checkpoint's per-head interleaved fused QKV, while the modular
> checkpoint is a de-interleaved layout) → the delta lands in the wrong place → noise. **The correct answer: diffusers `pipe.load_lora_weights()`**,
> which de-interleaves internally. Also a 46GB card cannot hold it resident (the inference below that "VRAM is enough after TP4 sharding" was a false impression masked by the noise),
> and it actually runs **bf16 block-stream offload**. Full root cause summarized in [`../README.md`](../README.md) §3.
> The content below this section is retained as the reasoning record at the time.
>
> Decision ① "use H3" is unchanged; this section is the landing approach **that makes it feasible on the chosen instance type**.

**Problem**: SGLang cookbook measurements show that H3's most VRAM-frugal **resident** topology (768P, H100 TP2+U2) requires
**~62GB/card**, while g6e.12xlarge's L40S has only **48GB/card**; naive 50-step BF16 is also slow
(~13s/clip on H100; L40S without NVLink over PCIe is slower).

**Community consensus**: people running H3 generally stack two kinds of adapters — which happen to each solve one bottleneck:
1. **Step-distillation Turbo LoRA** (cuts latency). The most-downloaded is
   **`lightx2v/Minimax-h3-Turbo` (Apache-2.0)**, which reduces denoising from ~50 steps to **4/8 steps**,
   with ready-made 768P BF16 weights; LightX2V Studio runs its **8-step v1.0** online.
   There are also `larryvrh/MiniMax-H3-Turbo-Lora`, Alibaba's `alibaba-pai/MiniMax-H3-Acc-LoRAs` (8step), etc.
2. **INT8 / NVFP4 quantization/pruning** (compresses VRAM to fit 48GB). E.g.
   `Abiray/Minimax-H3-nvfp4-INT4-INT8`, `unsloth/MiniMax-H3-GGUF`.

**This project adopts**: H3-Base + **lightx2v 8-step Turbo LoRA** (more stable image quality than 4-step, still
~6× faster). Locked into:
- `serving/serve_h3.sh`: defaults to `USE_TURBO_LORA=1`, loads that LoRA with `--lora-path/--lora-weight-name/--lora-scale/--lora-merge-mode auto` (prefers the local file cached on EBS);
- `gateway/app/backend.py`: `/v1/videos` requests carry `num_inference_steps=9`
  (**SGLang has an off-by-one**: 8 evaluations = 9; 4-step = 5; base ≈ 50);
- `serving/measure_768p.py`: defaults to `--steps 9`, **measuring exactly the Turbo configuration that will go live**;
- `serving/download_model.sh`: pulls the Turbo LoRA weights along the way.

**Still requires Phase 0 firsthand measurement**: even with Turbo stacked, whether 4×L40S can stay **resident** (no offload) is still not verified
on real hardware — if VRAM is still insufficient, the next step is to add INT8 quantization (the second category above, candidate already selected). This is the
go/no-go that `measure_768p.py` must answer after the machine is up.

### 4.3 License compliance prerequisite
MiniMax H3 Community License; external service is commercial use, with a regional application form (US/EU/UK/KR only).
Must clear this before opening externally — outside of tech, but unavoidable.

---

## 5. GPU Selection Reference (us-west-2 on-demand, approximate; use the aws-billing skill for official quotes)

| Instance | GPU | Approx. $/hr | Notes |
|---|---|---|---|
| `g6e.xlarge` | 1× L40S 48GB | ~$2 | VRAM insufficient to hold 108GB resident, needs layerwise offload + NVMe, very slow |
| **`g6e.12xlarge`** ✅ | **4× L40S 192GB** | **~$10** | **Chosen**. Enough to stay resident, PCIe with no NVLink is slower than the H100 reference; **must configure auto-stop** |
| `p4d.24xlarge` | 8× A100 40GB | ~$32 | Fast, but wasteful for this workload |
| `p5.48xlarge` | 8× H100 | ~$55+ | Full-speed chase; overkill for this workload, and quota is hard to get |

---

## 6. Directory Layout (this repo)

```
openminimax/
├── serving/     # production: h3_turbo_server.py (diffusers Turbo shim) + run_h3_server.sh + smoke scripts
│                #   deprecated: serve_h3.sh/download_model.sh/measure_768p.py (SGLang era, produced noise)
├── gateway/
│   ├── app/         # reusable core: protocol / backend / publish / worker / keys / tasks
│   │                #   keys.py, tasks.py each contain both a SQLite and a DynamoDB backend
│   │                #   main.py = single-machine FastAPI fallback; worker_main.py = GPU-machine worker
│   │                #   admin_keys.py = IAM-authorized issuing CLI
│   ├── lambdas/     # serverless entry points: authorizer (verify key) / api (submit/poll) / autostop
│   └── tests/       # 56 tests, no GPU/AWS throughout (moto), including wire-level contract and auth
├── infra/
│   ├── template.yaml          # base: private VPC + locked-down security groups + private result bucket + least-privilege IAM + (optional) GPU/gateway machines
│   ├── serverless.yaml        # API layer: HTTP API + Lambda authorizer + DynamoDB + EventBridge autostop
│   ├── deploy.sh / deploy_serverless.sh
│   └── *.service              # systemd units (serve=diffusers shim / GPU worker / autostop / gateway)
└── docs/        # PLAN.md (this file), API.md
```

---

## 7. AWS Security Baseline (hard requirement: no directly exposed public ports)

> Explicit user requirement: **conform to AWS security norms, no exposing SGLang / gateway ports directly to the public.**
> The following constraints are locked into the IaC under `infra/`; check each one during review.

1. **SGLang (port 30010) must never face 0.0.0.0/0.** Its security group inbound only allows 30010 **from the gateway
   security group** (security-group-referencing, not CIDR), everything else denied. The GPU machine
   needs no public IP.
2. **External entry = API Gateway, with Lambda verifying keys (settled, see §2.1).** The only public face is
   **API Gateway (HTTP API, TLS enforced)**, where each route first passes through a **Lambda authorizer** to verify
   the self-signed mmh3_ key; only after verification does it reach the submit/poll integration Lambda. There are no other public ports: SGLang
   and the GPU machine are entirely private; submit/poll/authorizer/worker communicate via DynamoDB + the AWS backbone.
   The authorizer result is cached per Authorization header (~300s), minimizing the polling verification cost.
   **The management plane has no public route** — issuing keys is an IAM-authorized CLI (`app.admin_keys`).
   API Gateway-level throttling (burst/rate) + optional WAF as an extra gate.
3. **Least-privilege IAM.** The GPU machine's / gateway's respective instance profiles are given only: the S3 read needed to
   pull the model, the S3 write to the result bucket, and `ec2:StartInstances/StopInstances` (limited to this project's instances tagged
   with a specific tag, tightened with an `aws:ResourceTag` condition).
4. **Private S3 result bucket.** Block Public Access fully on; `content.url` uses a **presigned URL**
   (short-lived) or CloudFront + OAC, no bucket-level public read.
5. **Encryption at rest + in transit.** EBS/S3 default KMS encryption; gateway↔Lambda over TLS.
6. **Observable + revocable.** The gateway meters/rate-limits each key, and keys can be revoked instantly; enable
   VPC Flow Logs / access logs for after-the-fact auditing.

---

## 8. Current State and Next Steps

- The gateway/API layer is deployed and healthy (entry point is your own CloudFront domain). GPU quota is confirmed available.
- **All code is landed and tested** (all 56 tests in `gateway/` green, including the serverless path's authorizer/
  submit/poll/autostop, running DynamoDB via moto, with no GPU/AWS cost).
- **Both infra templates pass cfn-lint + AWS server-side validation**; GPU startup is still the explicit approval-gated step `deploy.sh gpu
  APPROVE`, and the base/serverless layers are near zero cost.
- The integration form and private-network gateway entry are **settled**: fully serverless + API Gateway + Lambda key verification (§2.1, §7.2).

### Still to be decided before starting the GPU machine
1. **When to go for Phase 0** (launch `g6e.12xlarge` to measure 768P wall clock/VRAM firsthand — the go/no-go number the official docs lack).
2. Whether to add **WAF / custom domain + ACM certificate** to API Gateway (optional enhancement, not required).
