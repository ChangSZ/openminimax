"""GPU-box worker entrypoint (serverless path).

In the serverless architecture the submit/poll API is Lambda + DynamoDB, but the
generation itself must run where the GPU is. This is what runs on the g6e.12xlarge:
it drains the SAME DynamoDB task table the API Lambda writes to, calls the private
SGLang server, uploads the mp4 to the private result bucket, and writes the result
back to DynamoDB (which the poll Lambda then serves).

It wires the DynamoDB-backed stores (not sqlite) into the already-tested `Worker`
loop — the Worker is storage-agnostic, so nothing about the drain/generate/publish/
mark logic changes between the single-box and serverless paths.

ONE PROCESS PER KIND: `WORKER_KIND` selects what this process drains — `video` (the
default, SGLang/H3 on cuda:0) or `image` (FLUX on cuda:2). The two run as two systemd
units on the same GPU box (openminimax-worker.service / openminimax-image-worker.service)
so they drain the shared queue in parallel, each feeding its own card.

Config is all environment:

  KEYS_TABLE     DynamoDB keys table  (for the seconds-billed meter)
  TASKS_TABLE    DynamoDB tasks table (the shared queue)
  WORKER_KIND    `video` (default) | `image`
  SGLANG_URL     private SGLang/H3 address (video), e.g. http://127.0.0.1:30010
  FLUX_URL       private FLUX shim address (image), e.g. http://127.0.0.1:30020
  RESULT_BUCKET  private S3 bucket for results
  USE_FAKE_BACKEND=1  -> Fake{,Image}Backend + LocalPublisher (smoke, no model/S3)
"""

from __future__ import annotations

import logging
import os

from app.backend import (Backend, FakeBackend, FakeImageBackend, FluxImageBackend,
                         ImageBackend, SGLangBackend)
from app.keys import DynamoDBKeyStore
from app.publish import LocalPublisher, Publisher, S3Publisher
from app.tasks import IMAGE, VIDEO, DynamoDBTaskStore
from app.worker import Worker

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s worker %(levelname)s %(message)s")
log = logging.getLogger("worker_main")


def build_worker() -> Worker:
    keys_table = os.environ.get("KEYS_TABLE", "")
    tasks_table = os.environ.get("TASKS_TABLE", "")
    if not keys_table or not tasks_table:
        raise RuntimeError("KEYS_TABLE and TASKS_TABLE are required")
    kind = os.environ.get("WORKER_KIND", VIDEO).lower()
    if kind not in (VIDEO, IMAGE):
        raise RuntimeError(f"WORKER_KIND must be '{VIDEO}' or '{IMAGE}', got {kind!r}")

    keys = DynamoDBKeyStore(keys_table)
    tasks = DynamoDBTaskStore(tasks_table)
    fake = os.environ.get("USE_FAKE_BACKEND") == "1"
    publisher: Publisher = LocalPublisher() if fake else S3Publisher()
    if kind == IMAGE:
        image_backend: ImageBackend = FakeImageBackend() if fake else FluxImageBackend()
        log.info("worker[image]: %s + %s",
                 type(image_backend).__name__, type(publisher).__name__)
        return Worker(tasks=tasks, keys=keys, backend=image_backend,
                      publisher=publisher, kind=IMAGE)
    backend: Backend = FakeBackend() if fake else SGLangBackend()
    log.info("worker[video]: %s + %s",
             type(backend).__name__, type(publisher).__name__)
    return Worker(tasks=tasks, keys=keys, backend=backend, publisher=publisher,
                  kind=VIDEO)


def main() -> None:
    worker = build_worker()
    poll = float(os.environ.get("WORKER_POLL_SECONDS", "2"))
    stale = float(os.environ.get("WORKER_STALE_RUNNING_S", "3600"))
    log.info("worker: draining queue (poll=%.1fs, stale_running=%.0fs)", poll, stale)
    # Reuse the Worker's own loop; block forever (systemd owns the lifecycle).
    worker.start(poll_interval_s=poll, stale_running_s=stale)
    try:
        worker._thread.join()   # the daemon loop
    except KeyboardInterrupt:
        worker.stop()


if __name__ == "__main__":
    main()
