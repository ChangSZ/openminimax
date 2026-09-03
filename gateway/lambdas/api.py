"""HTTP API integration Lambda — the MiniMax v2 endpoints, serverless edition.

Same three routes as the FastAPI gateway (docs/API.md), but as an HTTP
API (payload 2.0) proxy handler backed by DynamoDB instead of a long-lived process:

  POST /v2/video_generation            -> enqueue, return {"task_id": ...}
  GET  /v2/query/video_generation/{id} -> poll -> {"task": {...}}
  POST /v1/image_generation            -> enqueue (kind=image), short-poll to done,
                                          return {"data":{"image_urls":[...]}} (sync)

Auth already happened in the Lambda authorizer (lambdas/authorizer.py); this handler
does NOT re-verify. It reads the caller's key prefix + rate limit from the authorizer
context, so it can rate-limit/meter and enforce per-key task isolation without ever
touching the secret again.

The actual generation is done by the worker on the GPU box (app.worker) draining the
SAME DynamoDB task table — this handler only enqueues and reads state, so it always
returns fast (well inside the client's timeout and the API's 30s hard cap).

Reuses the already-tested `app.protocol` for every wire shape; the only new thing is
event (de)serialization. `stores` is injectable for tests."""

from __future__ import annotations

import json
import logging
import os
import time

from app.keys import DynamoDBKeyStore, RateLimitedError
from app.protocol import (BadRequest, error_body, image_error_body, image_response,
                          parse_image_request, parse_video_request, poll_response)
from app.publish import presign_s3_ref
from app.tasks import FAILED, IMAGE, SUCCEEDED, DynamoDBTaskStore

logger = logging.getLogger()
logger.setLevel(logging.INFO)

KEYS_TABLE = os.environ.get("KEYS_TABLE", "")
TASKS_TABLE = os.environ.get("TASKS_TABLE", "")
# The image route is synchronous FOR THE CLIENT: we enqueue an image task, then poll
# it to completion IN THIS REQUEST before returning `image_urls`. The client's image
# worker sets MINIMAX_IMAGE_TIMEOUT_S to minutes, so it will wait — but this Lambda has
# its own ceiling, so cap the wait a little under the function timeout. 8-step Turbo is
# seconds; this is the safety budget, not the expected wait.
IMAGE_WAIT_S = float(os.environ.get("IMAGE_WAIT_S", "120"))
IMAGE_POLL_INTERVAL_S = float(os.environ.get("IMAGE_POLL_INTERVAL_S", "1"))
_KEYS: DynamoDBKeyStore | None = None
_TASKS: DynamoDBTaskStore | None = None


def _stores() -> tuple[DynamoDBKeyStore, DynamoDBTaskStore]:
    global _KEYS, _TASKS
    if _KEYS is None or _TASKS is None:
        if not KEYS_TABLE or not TASKS_TABLE:
            raise RuntimeError("KEYS_TABLE and TASKS_TABLE env vars are required")
        _KEYS = DynamoDBKeyStore(KEYS_TABLE)
        _TASKS = DynamoDBTaskStore(TASKS_TABLE)
    return _KEYS, _TASKS


def _resp(status: int, body: dict, headers: dict | None = None) -> dict:
    """An HTTP API proxy response. `body` MUST be a JSON string (the #1 cause of a 502
    is returning a dict here) — see the API Gateway reference."""
    return {"statusCode": status,
            "headers": {"content-type": "application/json", **(headers or {})},
            "body": json.dumps(body)}


def _key_prefix(event: dict) -> str:
    """The caller's key prefix, put there by the authorizer. Absent only if the route
    were wired without the authorizer — which would be a deploy bug, so fail closed."""
    ctx = ((event.get("requestContext") or {}).get("authorizer") or {}).get("lambda") or {}
    return ctx.get("keyPrefix", "")


def handler(event: dict, context=None, *, stores=None) -> dict:
    keys, tasks = stores or _stores()
    method = (event.get("requestContext") or {}).get("http", {}).get("method", "")
    raw_path = event.get("rawPath", "")

    prefix = _key_prefix(event)
    if not prefix:
        return _resp(401, error_body("unauthorized", "unauthorized"))

    if method == "POST" and raw_path.endswith("/v2/video_generation"):
        return _submit_video(event, keys, tasks, prefix)
    if method == "GET" and "/v2/query/video_generation/" in raw_path:
        return _query_video(event, tasks, prefix)
    if method == "POST" and raw_path.endswith("/v1/image_generation"):
        return _submit_image(event, keys, tasks, prefix)
    return _resp(404, error_body("not found", "not_found"))


def _submit_video(event, keys, tasks, prefix) -> dict:
    try:
        body = json.loads(event.get("body") or "{}")
    except ValueError:
        return _resp(400, error_body("body must be JSON"))
    try:
        gen = parse_video_request(body)
    except BadRequest as exc:
        return _resp(400, error_body(str(exc)))

    try:
        keys.check_and_count_submit(prefix)
    except RateLimitedError as exc:
        # 429 is transient to the client (CONTRACT §2) — it retries.
        return _resp(429, error_body("rate limited", "rate_limit_error"),
                     {"retry-after": str(exc.retry_after_s)})
    except KeyError:
        return _resp(401, error_body("unauthorized", "unauthorized"))

    task_id = tasks.enqueue(key_prefix=prefix, request=gen.to_dict(),
                            duration_s=gen.duration_s)
    return _resp(200, {"task_id": task_id})


def _query_video(event, tasks, prefix) -> dict:
    task_id = (event.get("pathParameters") or {}).get("task_id") or \
        event["rawPath"].rsplit("/", 1)[-1]
    task = tasks.get(task_id)
    if task is None or task.key_prefix != prefix:
        # Non-429/5xx 4xx => the client treats it as this task's terminal verdict.
        return _resp(404, error_body("task not found", "not_found"))
    # Sign the S3 reference into a fresh, short-lived URL HERE (this Lambda's role
    # has s3:GetObject) — never at generation time. See publish.presign_s3_ref.
    return _resp(200, poll_response(task, resolve_url=presign_s3_ref))


def _submit_image(event, keys, tasks, prefix, *, now=time.monotonic,
                  sleep=time.sleep) -> dict:
    """`POST /v1/image_generation` — synchronous FOR THE CLIENT.

    Enqueue an image task (kind=image, drained by the image worker on cuda:2), then
    poll it to a terminal state IN THIS REQUEST and return the finished `image_urls`.
    The client (providers/minimax.py::submit_image) does one call and reads
    `data.image_urls`; it has no image task id to poll, so the wait happens here.

    v1 error dialect (verified against the client): a refusal is HTTP 200 + non-zero
    `base_resp.status_code` (the client reads the BODY, not the HTTP status). The one
    exception is rate limiting — the client's transport retries on HTTP 429 — so a
    rate-limited submit returns a real 429, same as the video route.

    `now`/`sleep` are injectable so tests drive the poll loop without real time."""
    try:
        body = json.loads(event.get("body") or "{}")
    except ValueError:
        return _resp(200, image_error_body("body must be JSON"))
    try:
        img = parse_image_request(body)
    except BadRequest as exc:
        return _resp(200, image_error_body(str(exc)))

    try:
        keys.check_and_count_submit(prefix)
    except RateLimitedError as exc:
        return _resp(429, error_body("rate limited", "rate_limit_error"),
                     {"retry-after": str(exc.retry_after_s)})
    except KeyError:
        return _resp(401, error_body("unauthorized", "unauthorized"))

    task_id = tasks.enqueue(key_prefix=prefix, request=img.to_dict(),
                            duration_s=0, kind=IMAGE)

    # Short-poll to completion, in-request. duration_s carries no meaning for images.
    deadline = now() + IMAGE_WAIT_S
    while True:
        task = tasks.get(task_id)
        if task is not None and task.status == SUCCEEDED:
            refs = [r for r in (task.url or "").split(" ") if r]
            if not refs:
                return _resp(200, image_error_body("no image returned"))
            return _resp(200, image_response(refs, resolve_url=presign_s3_ref))
        if task is not None and task.status == FAILED:
            return _resp(200, image_error_body(task.error or "generation failed"))
        if now() >= deadline:
            # The task keeps running on the box; the client just didn't get to wait
            # long enough. A non-zero base_resp is the readable verdict.
            return _resp(200, image_error_body(
                "image generation timed out waiting for the worker"))
        sleep(IMAGE_POLL_INTERVAL_S)
