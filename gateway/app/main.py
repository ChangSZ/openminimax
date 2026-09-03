"""MiniMax v2-compatible gateway (FastAPI).

Endpoints, matched to docs/API.md so a MiniMax-compatible client reaches this
service with `MINIMAX_BASE_URL=https://<this gateway>` and no code change:

  POST /v2/video_generation            submit -> {"task_id": ...}   (fast; enqueues)
  GET  /v2/query/video_generation/{id} poll   -> {"task": {...}}
  POST /v1/image_generation            NOT IMPLEMENTED — returns a clear, readable
                                       failure (project decision, docs/PLAN.md §4.1)

  GET  /healthz                        liveness (no auth)
  POST /admin/keys, GET /admin/keys, DELETE /admin/keys/{prefix}
                                       key issue/list/revoke (separate admin auth)

Security posture (docs/PLAN.md §7): every /v2 and /v1 call is gated by a Bearer key
WE issued (app.keys); admin routes require a separate `ADMIN_TOKEN`. This app listens
only where infra puts it — it must NOT be exposed as a public port; SGLang behind it
is private-only. Auth failures return 401 (the contract's auth code); rate-limit
returns 429 (which the client treats as transient and retries).

Wiring is via a small `Deps` container so tests build the app with fakes (FakeBackend
+ LocalPublisher + in-memory sqlite) and exercise the real endpoints end-to-end with
no GPU, no AWS, no network.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from app.backend import (Backend, FakeBackend, FakeImageBackend, FluxImageBackend,
                         ImageBackend, SGLangBackend)
from app.keys import KeyInfo, KeyStore, RateLimitedError, RevokedError
from app.protocol import (BadRequest, error_body, image_error_body, image_response,
                          parse_image_request, parse_video_request, poll_response)
from app.publish import LocalPublisher, Publisher, S3Publisher, presign_s3_ref
from app.tasks import FAILED, IMAGE, SUCCEEDED, VIDEO, TaskStore
from app.worker import Worker

logger = logging.getLogger("gateway")

# Image route is synchronous FOR THE CLIENT: enqueue, then short-poll to done in the
# request. 8-step Turbo is seconds; this is the safety ceiling (see lambdas/api.py).
IMAGE_WAIT_S = float(os.environ.get("IMAGE_WAIT_S", "120"))
IMAGE_POLL_INTERVAL_S = float(os.environ.get("IMAGE_POLL_INTERVAL_S", "1"))


@dataclass
class Deps:
    keys: KeyStore
    tasks: TaskStore
    backend: Backend
    publisher: Publisher
    worker: Worker
    admin_token: str
    image_backend: ImageBackend
    image_worker: Worker


def build_deps() -> Deps:
    """Assemble real dependencies from the environment.

    `USE_FAKE_BACKEND=1` swaps in Fake{,Image}Backend + LocalPublisher so the whole
    service runs with no GPU/S3 — used for Phase-0 smoke tests before the model is up,
    and by the test suite. Otherwise: SGLang (video) + FLUX (image) + a private S3
    bucket. The single-box FastAPI path runs BOTH workers in-process (video drains H3
    on cuda:0, image drains FLUX on cuda:2); the deployed serverless path runs them as
    two separate processes on the GPU box instead."""
    db_path = os.environ.get("GATEWAY_DB", "gateway.db")
    keys = KeyStore(db_path)
    tasks = TaskStore(db_path)
    if os.environ.get("USE_FAKE_BACKEND") == "1":
        backend: Backend = FakeBackend()
        image_backend: ImageBackend = FakeImageBackend()
        publisher: Publisher = LocalPublisher()
    else:
        backend = SGLangBackend()
        image_backend = FluxImageBackend()
        publisher = S3Publisher()
    worker = Worker(tasks=tasks, keys=keys, backend=backend, publisher=publisher,
                    kind=VIDEO)
    image_worker = Worker(tasks=tasks, keys=keys, backend=image_backend,
                          publisher=publisher, kind=IMAGE)
    admin_token = os.environ.get("ADMIN_TOKEN", "")
    return Deps(keys=keys, tasks=tasks, backend=backend, publisher=publisher,
                worker=worker, admin_token=admin_token,
                image_backend=image_backend, image_worker=image_worker)


def create_app(deps: Deps | None = None, *, start_worker: bool = True) -> FastAPI:
    deps = deps or build_deps()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if start_worker:
            deps.worker.start()
            deps.image_worker.start()
        yield
        deps.worker.stop()
        deps.image_worker.stop()

    app = FastAPI(title="openminimax-gateway", docs_url=None, redoc_url=None,
                  lifespan=lifespan)
    app.state.deps = deps

    # --- auth dependencies --------------------------------------------------

    def require_key(authorization: str = Header(default="")) -> KeyInfo:
        """Resolve `Authorization: Bearer <key>` to a valid key, else 401.

        Both 'unknown' and 'revoked' surface as 401 (the contract's auth code) but
        are logged distinctly. No detail leaks to the caller."""
        token = _bearer(authorization)
        try:
            return deps.keys.verify(token)
        except RevokedError:
            logger.info("auth: revoked key presented")
            raise HTTPException(status_code=401, detail="invalid api key")
        except KeyError:
            raise HTTPException(status_code=401, detail="invalid api key")

    def require_admin(x_admin_token: str = Header(default="")) -> None:
        """Admin routes use a separate shared token, never a caller's key. If no
        ADMIN_TOKEN is configured the admin surface is closed entirely (403), so a
        misconfigured deploy can't accidentally expose key issuance."""
        import hmac
        if not deps.admin_token or not hmac.compare_digest(
                x_admin_token, deps.admin_token):
            raise HTTPException(status_code=403, detail="forbidden")

    # --- MiniMax v2: video (the contract) -----------------------------------

    @app.post("/v2/video_generation")
    async def submit_video(request: Request, key: KeyInfo = Depends(require_key)):
        """Enqueue a generation and return its task_id FAST (CONTRACT §1)."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400,
                                content=error_body("body must be JSON"))
        try:
            gen = parse_video_request(body)
        except BadRequest as exc:
            return JSONResponse(status_code=400, content=error_body(str(exc)))

        try:
            deps.keys.check_and_count_submit(key.prefix)
        except RateLimitedError as exc:
            # 429 is transient to the client — it keeps polling/retries (CONTRACT §2).
            return JSONResponse(
                status_code=429, content=error_body("rate limited", "rate_limit_error"),
                headers={"Retry-After": str(exc.retry_after_s)})

        task_id = deps.tasks.enqueue(
            key_prefix=key.prefix, request=gen.to_dict(), duration_s=gen.duration_s)
        return {"task_id": task_id}

    @app.get("/v2/query/video_generation/{task_id}")
    def query_video(task_id: str, key: KeyInfo = Depends(require_key)):
        """Poll one task (CONTRACT §2). A task is only visible to the key that owns
        it — a wrong/other key gets 404-as-terminal semantics (unknown task)."""
        task = deps.tasks.get(task_id)
        if task is None or task.key_prefix != key.prefix:
            # Non-429/5xx 4xx => the client treats it as this task's terminal verdict,
            # which is correct: an unknown id will read the same way forever.
            return JSONResponse(status_code=404,
                                content=error_body("task not found", "not_found"))
        return poll_response(task, resolve_url=presign_s3_ref)

    # --- MiniMax v1: image (intentionally not implemented) ------------------

    @app.post("/v1/image_generation")
    async def image_generation(request: Request, key: KeyInfo = Depends(require_key)):
        """MiniMax v1 image-01: synchronous FOR THE CLIENT.

        Enqueue an image task (kind=image, drained by the in-process image worker on
        cuda:2), then short-poll it to a terminal state and return the finished
        `image_urls`. The client does one call and reads `data.image_urls`; the wait
        happens here. A refusal is HTTP 200 + non-zero base_resp (the v1 dialect the
        client reads); a rate-limit is a real 429 (its transport retries on 429)."""
        import asyncio
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=200,
                                content=image_error_body("body must be JSON"))
        try:
            img = parse_image_request(body)
        except BadRequest as exc:
            return JSONResponse(status_code=200,
                                content=image_error_body(str(exc)))

        try:
            deps.keys.check_and_count_submit(key.prefix)
        except RateLimitedError as exc:
            return JSONResponse(
                status_code=429, content=error_body("rate limited", "rate_limit_error"),
                headers={"Retry-After": str(exc.retry_after_s)})

        task_id = deps.tasks.enqueue(key_prefix=key.prefix, request=img.to_dict(),
                                     duration_s=0, kind=IMAGE)
        loop = asyncio.get_event_loop()
        deadline = loop.time() + IMAGE_WAIT_S
        while True:
            task = deps.tasks.get(task_id)
            if task is not None and task.status == SUCCEEDED:
                refs = [r for r in (task.url or "").split(" ") if r]
                if not refs:
                    return JSONResponse(status_code=200,
                                        content=image_error_body("no image returned"))
                return image_response(refs, resolve_url=presign_s3_ref)
            if task is not None and task.status == FAILED:
                return JSONResponse(
                    status_code=200,
                    content=image_error_body(task.error or "generation failed"))
            if loop.time() >= deadline:
                return JSONResponse(status_code=200, content=image_error_body(
                    "image generation timed out waiting for the worker"))
            await asyncio.sleep(IMAGE_POLL_INTERVAL_S)

    # --- health + admin -----------------------------------------------------

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.post("/admin/keys")
    def admin_issue(payload: dict | None = None, _: None = Depends(require_admin)):
        payload = payload or {}
        secret = deps.keys.issue(
            label=str(payload.get("label", "")),
            rate_limit_per_min=int(payload.get("rate_limit_per_min", 6)))
        # The ONLY time the secret is ever returned. Hand it to the caller.
        return {"api_key": secret}

    @app.get("/admin/keys")
    def admin_list(_: None = Depends(require_admin)):
        return {"keys": deps.keys.list_keys()}

    @app.delete("/admin/keys/{prefix}")
    def admin_revoke(prefix: str, _: None = Depends(require_admin)):
        return {"revoked": deps.keys.revoke(prefix)}

    return app


def _bearer(authorization: str) -> str:
    """Extract the token from an `Authorization: Bearer <token>` header (case-
    insensitive scheme), or return '' — which verify() rejects."""
    if not authorization:
        return ""
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return authorization.strip()   # tolerate a bare token


def get_app() -> FastAPI:
    """Build the app from environment deps. Use as `uvicorn --factory app.main:get_app`.

    A factory (not a module-level instance) so merely IMPORTING this module never
    constructs a real S3Publisher / SGLangBackend — tests build the app with fakes,
    and `build_deps()` (which reads RESULT_BUCKET etc.) runs only for a real serve."""
    return create_app()
