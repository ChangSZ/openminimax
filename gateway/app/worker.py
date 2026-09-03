"""The generation worker: drain the queue, run the backend, publish, update state.

The submit path only enqueues (it must return a task_id fast). This is where the
minutes-long work actually happens, off the request path. It is deliberately simple
and single-flight: H3 on one box serves ~one clip at a time, so the worker claims
one queued task, generates it, publishes the media, and moves on. Running it as a
background thread inside the gateway process is enough for a small deployment; the same
`run_once` is what a separate worker process/loop would call.

ONE KIND PER WORKER: a worker is constructed for `kind="video"` or `kind="image"`
and only ever claims/runs tasks of that kind (`tasks.claim_next(kind)`). Two workers
share one queue but feed different GPUs — video -> H3 on cuda:0, image -> FLUX on
cuda:2 — so they run in parallel and never steal each other's work. The video path
is byte-for-byte what it always was; the image path publishes the N generated PNGs
and stores their references space-joined in `task.url` (a space is URL-safe, so it
round-trips as a delimiter), which the image handler splits back into `image_urls`.

Crash safety: `requeue_stale_running` (called on start) returns orphaned `running`
rows to `queued`, so a worker restart doesn't strand a task. Because the backend is
not idempotent (a re-run bills a second generation), we only requeue rows stuck far
longer than any real generation — the caller passes that cutoff.

Nothing here imports AWS/SGLang/FLUX directly; it takes a backend and a Publisher, so
tests inject Fake{,Image}Backend + LocalPublisher and exercise the whole path.
"""

from __future__ import annotations

import logging
import threading
import time

from app.backend import Backend, ImageBackend
from app.keys import KeyStore
from app.protocol import GenRequest, ImageRequest
from app.publish import Publisher
from app.tasks import IMAGE, VIDEO, TaskStore

logger = logging.getLogger("gateway.worker")

# Space-joined references round-trip through `task.url` for a multi-image result
# (a space cannot appear in a URL, so it is a safe delimiter — same idea the client
# uses in its `urls:` handle).
_URL_SEP = " "


class Worker:
    def __init__(self, *, tasks: TaskStore, keys: KeyStore,
                 backend: Backend | ImageBackend, publisher: Publisher,
                 kind: str = VIDEO):
        self._tasks = tasks
        self._keys = keys
        self._backend = backend
        self._publisher = publisher
        self._kind = kind
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run_once(self) -> bool:
        """Claim and process one task OF THIS WORKER'S KIND. Returns True if it did
        work, False if none was queued. Never raises — a generation failure becomes
        the task's `failed` reason (which the client reads), because a raise here
        would kill the loop and strand every other task."""
        task = self._tasks.claim_next(self._kind)
        if task is None:
            return False
        logger.info("worker[%s]: generating task %s (owner=%s)",
                    self._kind, task.task_id, task.key_prefix)
        try:
            if task.kind == IMAGE:
                self._run_image(task)
            else:
                self._run_video(task)
            logger.info("worker[%s]: task %s succeeded", self._kind, task.task_id)
        except Exception as exc:  # noqa: BLE001 — a failure is the task's verdict
            logger.warning("worker[%s]: task %s failed: %s",
                           self._kind, task.task_id, exc)
            self._tasks.mark_failed(task.task_id, str(exc) or "generation failed",
                                    kind=task.kind)
        return True

    def _run_video(self, task) -> None:
        req = GenRequest.from_dict(task.request)
        mp4 = self._backend.generate(req)
        url = self._publisher.publish(
            key_prefix=task.key_prefix, task_id=task.task_id, mp4=mp4)
        self._tasks.mark_succeeded(task.task_id, url, kind=VIDEO)
        # Coarse metering: charge the requested clip length on success.
        self._keys.add_seconds_billed(task.key_prefix, task.duration_s)

    def _run_image(self, task) -> None:
        req = ImageRequest.from_dict(task.request)
        images = self._backend.generate(req)   # list[bytes], one PNG per image
        if not images:
            raise RuntimeError("no image returned")
        refs = [
            self._publisher.publish_media(
                key_prefix=task.key_prefix, object_name=f"{task.task_id}_{i}.png",
                data=png, content_type="image/png")
            for i, png in enumerate(images)
        ]
        # Store all refs space-joined; the image handler splits + presigns each.
        self._tasks.mark_succeeded(task.task_id, _URL_SEP.join(refs), kind=IMAGE)
        # Coarse metering: charge per image produced.
        self._keys.add_seconds_billed(task.key_prefix, len(images))

    # --- background loop ----------------------------------------------------

    def start(self, *, poll_interval_s: float = 1.0,
              stale_running_s: float = 3600.0) -> None:
        """Spawn the drain loop in a daemon thread. Requeues stale `running` tasks
        once at startup (crash recovery)."""
        requeued = self._tasks.requeue_stale_running(stale_running_s)
        if requeued:
            logger.info("worker: requeued %d stale running task(s)", requeued)

        def _loop() -> None:
            while not self._stop.is_set():
                try:
                    did_work = self.run_once()
                except Exception:  # pragma: no cover - run_once shouldn't raise
                    logger.exception("worker: unexpected error in loop")
                    did_work = False
                if not did_work:
                    self._stop.wait(poll_interval_s)

        self._thread = threading.Thread(
            target=_loop, name=f"gen-worker-{self._kind}", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout_s)
