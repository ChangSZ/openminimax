# API Reference

How to call an `openminimax` deployment: issue a key, authenticate, generate video
(async, MiniMax v2-compatible) and images (sync, MiniMax v1-compatible). Every shape
below is what the gateway actually accepts and returns.

> **The calling convention is identical to MiniMax's own API** — the endpoints, request
> bodies, and response shapes below mirror MiniMax's. Any existing MiniMax client works by
> just pointing `MINIMAX_BASE_URL` at this gateway; no code changes. For the upstream
> reference see MiniMax's docs: <https://www.minimax.io/platform/document/video_generation>
> (video) and <https://www.minimax.io/platform/document/image_generation> (image). The one
> difference is that the underlying image model here is FLUX.2, not MiniMax `image-01`.

- **Base URL** — the `ApiEndpoint` output of the serverless stack, e.g.
  `https://<id>.execute-api.<region>.amazonaws.com`. Set it as `MINIMAX_BASE_URL` in any
  MiniMax-compatible client and no other code changes are needed. A trailing slash is stripped.
- **Auth** — every request carries `Authorization: Bearer <key>`, where `<key>` is an
  `mmh3_...` key you issued (see [Keys](#keys)). A bad/missing key is **HTTP 401**.
- **Content-Type** — requests with a body use `application/json`.
- **Endpoints:**

  | Method | Path | Purpose | Style |
  |---|---|---|---|
  | `POST` | `/v2/video_generation` | submit a video job | async → `{task_id}` |
  | `GET`  | `/v2/query/video_generation/{task_id}` | poll a video job | returns `{task:{...}}` |
  | `POST` | `/v1/image_generation` | generate image(s) | sync → `{data:{image_urls}}` |

---

## Keys

Keys are self-signed `mmh3_` tokens. Issue/list/revoke them with the operator CLI
(`gateway/app/admin_keys.py`), which needs AWS credentials that can read/write the keys
table and the table name in `KEYS_TABLE`. Run from `gateway/`:

```bash
# mint a key — the plaintext is printed ONCE and is NOT recoverable (only its hash is stored)
KEYS_TABLE=openminimax-serverless-keys AWS_DEFAULT_REGION=us-west-2 \
  python -m app.admin_keys issue --label team-1 --rate 6
#   --label  free-text owner tag (optional)
#   --rate   submits allowed per minute (default 6)

# list keys (never shows secrets): prefix, state, rate, submit count, seconds billed, label
KEYS_TABLE=openminimax-serverless-keys AWS_DEFAULT_REGION=us-west-2 \
  python -m app.admin_keys list

# revoke a key by its prefix (the leading token shown by `list`)
KEYS_TABLE=openminimax-serverless-keys AWS_DEFAULT_REGION=us-west-2 \
  python -m app.admin_keys revoke <prefix>
```

Each key has a **per-minute submit rate limit**; exceeding it returns **HTTP 429** with a
`retry-after` header. Tasks are isolated per key — polling another key's `task_id` returns 404.

---

## Video generation (async, MiniMax v2)

Video is self-hosted MiniMax-H3. Submit returns immediately with a `task_id`; the clip is
produced by the GPU worker in the background, and you poll until it is ready. (Wall-clock
per clip: see the perf table in the top-level [`README.md`](../README.md) §3.)

### Submit

```
POST /v2/video_generation
Authorization: Bearer <key>
Content-Type: application/json
```
```json
{
  "model": "MiniMax-H3",
  "content": [
    { "type": "text", "text": "<your prompt>" },
    { "type": "image_url", "role": "reference_image",
      "image_url": { "url": "https://.../ref1.png" } }
  ],
  "resolution": "768P",
  "duration": 6,
  "ratio": "16:9"
}
```

Field notes (this is the MiniMax v2 shape — there is **no top-level `prompt`**):
- `content[]` — an **ordered** list. Exactly one `{type:"text", text}` entry is the prompt.
  Zero or more `{type:"image_url", role:"reference_image", image_url:{url}}` entries are
  reference images; the prompt refers to them in prose as "reference image N" (N counts
  from 1 in `content` order). With no reference images, `content` holds only the text entry.
  Each `url` must be fetchable by the GPU box (a 403/unreachable URL fails the job).
- `resolution` — `"768P"` (the only supported output; true 768P = 1344×768).
- `duration` — **integer seconds**, valid range `[4, 15]`.
- `ratio` — one of `"21:9" | "16:9" | "4:3" | "1:1" | "3:4" | "9:16"`. Note the field is
  `ratio`, not `aspect_ratio`. Values outside the enum are normalized to `16:9`.

**Response** (submit only enqueues; it does not wait):
```json
{ "task_id": "vt_..." }
```

### Poll

```
GET /v2/query/video_generation/{task_id}
Authorization: Bearer <key>
```
```json
{
  "task": {
    "status": "queued | running | succeeded | failed",
    "content": { "url": "https://.../result.mp4" },
    "error":   { "message": "<failure reason>" }
  }
}
```

- `queued` / `running` (or any unrecognized status) → keep polling.
- `succeeded` → `task.content.url` is a **short-lived, directly downloadable presigned mp4
  URL**, signed fresh at poll time. Download it promptly.
- `failed` → read `task.error.message`.

**HTTP status semantics while polling** (matches how MiniMax clients classify errors):
- **429 / 5xx** → transient; the caller should keep polling.
- **Other 4xx** (e.g. 404 unknown/expired `task_id`, 401 revoked key) → terminal failure
  for that task. The gateway never expresses a permanent failure as a 5xx.

### curl example

```bash
BASE=https://<id>.execute-api.<region>.amazonaws.com
KEY=mmh3_...

TASK=$(curl -s -X POST "$BASE/v2/video_generation" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"MiniMax-H3","content":[{"type":"text","text":"a red panda in a bamboo forest at golden hour"}],"resolution":"768P","duration":6,"ratio":"16:9"}' \
  | python -c 'import sys,json;print(json.load(sys.stdin)["task_id"])')

# poll until succeeded, then grab the url
curl -s "$BASE/v2/query/video_generation/$TASK" -H "Authorization: Bearer $KEY"
```

---

## Image generation (sync, MiniMax v1)

Image is self-hosted **FLUX.2-dev** (not MiniMax `image-01`, which is not open-source — so
style/behavior differ). The endpoint is MiniMax v1-compatible: **synchronous for the
caller** — the request blocks until the image(s) are ready (fp8-resident FLUX runs in
~37.5s per 1024×768 image), then returns the URLs. Requires the optional image stack to be
deployed (see [`README`](../README.md) §2); on a video-only deployment this route returns a readable refusal.

### Request

```
POST /v1/image_generation
Authorization: Bearer <key>
Content-Type: application/json
```
```json
{
  "model": "image-01",
  "prompt": "<your prompt>",
  "n": 1,
  "aspect_ratio": "16:9"
}
```
- `prompt` — **required** (a missing/empty prompt is an error).
- `n` — number of images, clamped to `[1, 9]` (default 1).
- `width` / `height` — pixel dimensions; when present they take priority over `aspect_ratio`.
- `aspect_ratio` — default `"16:9"`.
- `subject_reference` — optional; a list of `{type:"character", image_file:<url>}` (only the
  first is used), a bare url list, or a single url string.

### Response

Success (HTTP 200):
```json
{
  "data": { "image_urls": ["https://.../img0.png"] },
  "base_resp": { "status_code": 0, "status_msg": "success" }
}
```
- Read `data.image_urls` (an object-with-array, **not** `data[].url`). Each is a short-lived
  presigned PNG URL. `status_code: 0` is the only success code.

**Failure / refusal** (also **HTTP 200** — the v1 dialect signals errors in the body, not the
HTTP status): a **non-zero `base_resp.status_code`** with the reason in `status_msg`, and no
`image_urls`. Read the body, not the status line:
```json
{ "base_resp": { "status_code": 1500, "status_msg": "<reason>" } }
```
(A bad request, a failed generation, or an in-request timeout all surface this way.) The one
exception is **rate limiting, which is a real HTTP 429** — MiniMax clients retry their
transport on 429, so it is not folded into `base_resp`.

---

## Error reference

| Condition | HTTP | Body |
|---|---|---|
| Missing / invalid key | 401 | `{"type":"error","error":{"type":"unauthorized","message":"..."}}` |
| Rate limit exceeded (video & image) | 429 | `{"type":"error","error":{"type":"rate_limit_error",...}}` + `retry-after` header |
| Bad video request body | 400 | `{"type":"error","error":{"type":"invalid_request_error","message":"..."}}` |
| Unknown/expired video `task_id` | 404 | `{"type":"error","error":{"type":"not_found",...}}` |
| Bad image request / failed / timed out | 200 | `{"base_resp":{"status_code":1500,"status_msg":"..."}}` |
| Unknown route | 404 | `{"type":"error","error":{"type":"not_found",...}}` |
