"""Publish a finished mp4 and, at read time, hand back a short-lived download URL.

The client's poll response carries `task.content.url` — a URL it will GET to copy
the clip into its own bucket (docs/API.md §2). MiniMax hands back a
time-limited direct URL; we do the same, but with the signing split from the upload:

  * the WORKER uploads the mp4 to a PRIVATE S3 bucket (Block Public Access on —
    docs/PLAN.md §7.4) and stores only a DURABLE reference `s3://bucket/key` in
    DynamoDB. It never signs a URL — its GPU-box role is write-only, and a URL
    signed at generation time would be counting down its TTL before the client
    ever polls.
  * the POLL handler calls `presign_s3_ref()` on that reference every time it
    answers `succeeded`, so the client always gets a FRESH, short-lived URL signed
    by a role that actually has s3:GetObject. This is why `content.url` can't go
    stale between generation and download, and why a write-only worker is enough.

Two publishers:
  - `S3Publisher`    : real. `RESULT_BUCKET` env names a private bucket; the object
    key is namespaced by key-prefix and task id. Returns the `s3://…` reference.
  - `LocalPublisher` : no AWS — writes under a temp dir and returns a file:// URL, so
    the gateway runs end-to-end on a laptop / in CI with no S3.

boto3 is imported lazily so the module (and the tests that use LocalPublisher) never
require AWS credentials.
"""

from __future__ import annotations

import os
import pathlib
import tempfile
from typing import Protocol

RESULT_BUCKET = os.environ.get("RESULT_BUCKET", "")
PRESIGN_TTL_S = int(os.environ.get("PRESIGN_TTL_S", "3600"))


def presign_s3_ref(ref: str, ttl_s: int = PRESIGN_TTL_S) -> str:
    """Turn a stored reference into a downloadable URL, signing at CALL time.

    `s3://bucket/key` -> a fresh presigned GET URL (TTL from now). Anything else
    (a file:// URL from LocalPublisher, an already-http URL, or "") is returned
    unchanged, so this is a safe default resolver for every path and for tests.

    Signed here — in the poll handler's role — NOT at upload time, so the URL's
    clock starts when the client is about to use it, and the signer is a role with
    s3:GetObject (the worker's write-only role could not produce a usable URL)."""
    if not ref.startswith("s3://"):
        return ref
    bucket, _, key = ref[len("s3://"):].partition("/")
    import boto3  # lazy: only the poll handler needs AWS
    from botocore.config import Config
    # SSE-KMS objects REQUIRE SigV4 (S3 rejects the legacy SigV2 the global
    # s3.amazonaws.com endpoint would otherwise use: "require AWS Signature
    # Version 4"). Pin s3v4 + the regional virtual-host endpoint so the presigned
    # GET is actually usable.
    s3 = boto3.client("s3", config=Config(signature_version="s3v4",
                                          s3={"addressing_style": "virtual"}))
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=ttl_s)


class Publisher(Protocol):
    def publish_media(self, *, key_prefix: str, object_name: str, data: bytes,
                      content_type: str) -> str:
        """Store `data` under `results/<key_prefix>/<object_name>` and return a
        DURABLE reference (resolved to a URL at read time via `presign_s3_ref`), not
        a pre-signed URL. Media-agnostic: `object_name` carries the extension and
        `content_type` the MIME type, so the same path serves an mp4 or a png."""
        ...

    def publish(self, *, key_prefix: str, task_id: str, mp4: bytes) -> str:
        """Video convenience wrapper over `publish_media` (unchanged callers)."""
        ...


class S3Publisher:
    def __init__(self, bucket: str = RESULT_BUCKET, ttl_s: int = PRESIGN_TTL_S):
        if not bucket:
            raise ValueError("RESULT_BUCKET is required for S3Publisher")
        import boto3  # lazy: no AWS import unless actually used
        self._s3 = boto3.client("s3")
        self._bucket = bucket
        self._ttl = ttl_s

    def publish_media(self, *, key_prefix: str, object_name: str, data: bytes,
                      content_type: str) -> str:
        object_key = f"results/{key_prefix}/{object_name}"
        self._s3.put_object(
            Bucket=self._bucket, Key=object_key, Body=data,
            ContentType=content_type,
            # Defense in depth even though the bucket is already private+encrypted.
            ServerSideEncryption="aws:kms")
        # Durable reference only — the read handler signs a fresh URL on demand.
        return f"s3://{self._bucket}/{object_key}"

    def publish(self, *, key_prefix: str, task_id: str, mp4: bytes) -> str:
        return self.publish_media(
            key_prefix=key_prefix, object_name=f"{task_id}.mp4", data=mp4,
            content_type="video/mp4")


class LocalPublisher:
    """Filesystem fallback for dev/CI. Returns a file:// URL."""

    def __init__(self, root: str | None = None):
        self._root = pathlib.Path(root or tempfile.mkdtemp(prefix="mmh3-results-"))

    def publish_media(self, *, key_prefix: str, object_name: str, data: bytes,
                      content_type: str) -> str:
        out = self._root / key_prefix / object_name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return out.as_uri()

    def publish(self, *, key_prefix: str, task_id: str, mp4: bytes) -> str:
        return self.publish_media(
            key_prefix=key_prefix, object_name=f"{task_id}.mp4", data=mp4,
            content_type="video/mp4")
