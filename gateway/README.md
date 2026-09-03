# gateway — MiniMax-compatible gateway (with self-signed key auth)

FastAPI service that speaks the **MiniMax v2 video protocol** (and the MiniMax v1 image
endpoint) so any MiniMax client reaches the self-hosted MiniMax-H3 / FLUX.2 backends with
only `MINIMAX_BASE_URL=https://<this gateway>` set — **zero client code change**.

The exact request/response shapes this implements are documented in [`../docs/API.md`](../docs/API.md).

## Endpoints

| Method + path | Purpose |
|---|---|
| `POST /v2/video_generation` | submit → `{ "task_id": ... }` (fast; only enqueues) |
| `GET /v2/query/video_generation/{id}` | poll → `{ "task": { status, content.url } }` |
| `POST /v1/image_generation` | **not implemented** — returns a readable v1-shaped refusal (project decision, [PLAN §4.1](../docs/PLAN.md)) |
| `GET /healthz` | liveness, no auth |
| `POST/GET/DELETE /admin/keys[...]` | issue / list / revoke keys (separate `X-Admin-Token`) |

## How it fits together

```
request path                     background
────────────                     ──────────
POST /v2/video_generation        Worker (daemon thread / process)
  → verify Bearer key (keys.py)    → claim_next() queued task
  → parse v2 body (protocol.py)    → backend.generate()  ← SGLang (private) OR Fake
  → rate-limit + meter (keys.py)   → publish() mp4 → private S3 → presigned URL
  → enqueue (tasks.py)             → mark_succeeded/failed + meter
  → return {task_id}  (no wait)
GET /v2/query/.../{id} → poll_response(task)
```

- `keys.py`   — issue/verify/revoke/rate-limit/meter; raw key never stored (salted PBKDF2).
- `protocol.py` — MiniMax v2 ⇄ internal request; the contract shapes, pure/tested.
- `tasks.py`  — SQLite queue + task state; crash-safe (`requeue_stale_running`).
- `backend.py` — the ONLY GPU-facing code. `SGLangBackend` (real) vs `FakeBackend` (no GPU). Named for the original SGLang plan, but as of 2026-08-30 `SGLANG_URL` points at the **diffusers Turbo HTTP shim** (`../serving/h3_turbo_server.py`), which speaks the identical async `/v1/videos` protocol — so this class is unchanged. (SGLang couldn't apply the Turbo LoRA; see `../serving/README.md`.)
- `publish.py` — mp4 → private S3 presigned URL (`S3Publisher`) or `file://` (`LocalPublisher`).
- `worker.py` — drains the queue; a failure becomes the task's `failed` reason, never a crash.

## Run locally (no GPU, no AWS)

```bash
pip install -r requirements.txt pytest httpx
python -m pytest                       # 33 tests, all offline

# serve with fakes (instant mp4, file:// URLs)
USE_FAKE_BACKEND=1 ADMIN_TOKEN=dev \
  python -m uvicorn --factory app.main:get_app --host 127.0.0.1 --port 8099
```

## Run for real

Set (no code change):

| env | meaning |
|---|---|
| `SGLANG_URL` | private SGLang address, e.g. `http://10.0.x.x:30010` (**never public** — [PLAN §7](../docs/PLAN.md)) |
| `RESULT_BUCKET` | private S3 bucket for result mp4s (Block Public Access on, KMS) |
| `PRESIGN_TTL_S` | presigned-URL lifetime (default 3600) |
| `ADMIN_TOKEN` | shared secret for `/admin/*` (unset ⇒ admin surface closed) |
| `GATEWAY_DB` | SQLite path (default `gateway.db`) |

```bash
SGLANG_URL=http://10.0.1.20:30010 RESULT_BUCKET=my-private-bucket ADMIN_TOKEN=... \
  python -m uvicorn --factory app.main:get_app --host 0.0.0.0 --port 8080
```

> **Security:** this process must sit behind the private ingress described in
> [PLAN §7](../docs/PLAN.md) — do **not** expose it (or SGLang) as a raw public port.
> Every `/v2` `/v1` call is Bearer-gated; `/admin/*` needs the separate admin token.

## Issue a key for a client

```bash
curl -X POST $GW/admin/keys -H "X-Admin-Token: $ADMIN_TOKEN" \
  -H 'Content-Type: application/json' -d '{"label":"team-3","rate_limit_per_min":6}'
# → {"api_key":"mmh3_....."}   ← the ONLY time the secret is shown; hand it over
```
