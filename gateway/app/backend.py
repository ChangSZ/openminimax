"""The GPU-facing backend: turn a GenRequest into an mp4, then into a URL.

Everything else in the gateway is protocol/state and provably correct today. This
module is the ONE place that talks to the unknowns — SGLang's `/v1/videos` request
shape and the L40S wall-clock — so it is quarantined here behind a small interface:

    class Backend:
        def generate(self, req: GenRequest) -> bytes   # the mp4, or raise

Two implementations:
  - `SGLangBackend`  : real. Calls SGLang, whose exact field names are marked
    ⚠️PHASE-0-VERIFY because they are the one thing NOT yet confirmed against a
    running server (docs/PLAN.md §1: the 768P protocol is from the cookbook, not a
    live call). When Phase 0 brings a server up, confirm/adjust `_to_sglang` and
    nothing else in the codebase moves.
  - `FakeBackend`    : returns a tiny valid mp4 immediately, so the whole gateway +
    worker + poll loop is testable with no GPU and no spend.

The result-to-URL step (upload mp4 to a private S3 bucket, hand back a short-lived
presigned URL) lives in `publish()` and is backend-independent. The bucket is
private (docs/PLAN.md §7.4); the client only ever sees a time-limited URL, exactly
like MiniMax's own direct URLs.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Protocol

from app.protocol import GenRequest, ImageRequest

# SGLang binds to loopback/private only — NEVER a public port (docs/PLAN.md §7.1).
# The gateway reaches it inside the VPC; this default is the co-located dev case.
SGLANG_URL = os.environ.get("SGLANG_URL", "http://127.0.0.1:30010")
SGLANG_TIMEOUT_S = int(os.environ.get("SGLANG_TIMEOUT_S", "1800"))  # a clip is minutes

# The FLUX image shim (serving/flux_image_server.py) — a SEPARATE process on cuda:2,
# same loopback-only rule as SGLang (README §2). Async like /v1/videos:
# submit returns a job id, poll until completed. An 8-step Turbo image is seconds, so
# the total budget is far smaller than a clip's.
FLUX_URL = os.environ.get("FLUX_URL", "http://127.0.0.1:30020")
FLUX_TIMEOUT_S = int(os.environ.get("FLUX_TIMEOUT_S", "180"))       # an image is seconds
FLUX_POLL_INTERVAL_S = float(os.environ.get("FLUX_POLL_INTERVAL_S", "1"))

# Denoise evaluations per clip. The default matches serve_h3.sh's 8-step Turbo LoRA.
# NOTE the off-by-one baked into SGLang: `num_inference_steps` counts sigma grid
# points INCLUDING the terminal zero, so the loop runs one fewer model eval — an
# 8-eval adapter therefore uses 9 (4-step -> 5, base -> ~50). Set SGLANG_STEPS to
# match whatever LoRA (if any) serve_h3.sh loaded; they must agree.
SGLANG_STEPS = int(os.environ.get("SGLANG_STEPS", "9"))

# `/v1/videos` is async (PHASE-0-VERIFIED): submit returns a job id, then we poll
# GET /v1/videos/{id} this often until it is completed/failed.
SGLANG_POLL_INTERVAL_S = float(os.environ.get("SGLANG_POLL_INTERVAL_S", "5"))


class Backend(Protocol):
    def generate(self, req: GenRequest) -> bytes:
        """Produce an mp4 (bytes) for `req`, or raise on failure."""
        ...


class SGLangBackend:
    """Drives a private SGLang server running MiniMax-H3 (serving/)."""

    def __init__(self, url: str = SGLANG_URL, timeout_s: int = SGLANG_TIMEOUT_S,
                 poll_interval_s: float = SGLANG_POLL_INTERVAL_S):
        self._url = url.rstrip("/")
        self._timeout = timeout_s               # total budget for one clip to finish
        self._poll_interval = poll_interval_s   # gap between status polls
        self._poll_timeout = 30                 # per-HTTP-call timeout (submit/poll)

    def generate(self, req: GenRequest) -> bytes:
        """Submit a clip, poll to completion, return the mp4 bytes.

        PHASE-0-VERIFIED (2026-08-29, live L40S box): SGLang's `/v1/videos` is
        ASYNC. POST returns `{"id": ..., "status": "queued"}` immediately; the clip
        is produced in the background. We then poll `GET /v1/videos/{id}` until
        `status` is `completed`/`failed`, and read the finished mp4 from the
        `file_paths[0]` LOCAL path the server writes on the GPU box (the worker runs
        co-located with SGLang, so the file is on the same disk). `url`/base64 are
        also handled in case a future build returns the clip inline."""
        payload = self._to_sglang(req)
        data = json.dumps(payload).encode()
        http = urllib.request.Request(
            f"{self._url}/v1/videos", data=data, method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(http, timeout=self._poll_timeout) as resp:
            submit = json.loads(resp.read())
        job_id = submit.get("id")
        if not job_id:
            raise RuntimeError(f"SGLang submit returned no id: {submit!r}")
        return self._await_result(job_id)

    def _await_result(self, job_id: str) -> bytes:
        """Poll GET /v1/videos/{id} until terminal, then materialize the mp4."""
        import time
        deadline = time.monotonic() + self._timeout
        url = f"{self._url}/v1/videos/{job_id}"
        while True:
            with urllib.request.urlopen(url, timeout=self._poll_timeout) as resp:
                doc = json.loads(resp.read())
            status = (doc.get("status") or "").lower()
            if status in ("completed", "succeeded"):
                return self._materialize_mp4(doc)
            if status in ("failed", "cancelled", "error"):
                err = doc.get("error") or doc.get("message") or "SGLang job failed"
                raise RuntimeError(f"SGLang job {job_id} {status}: {err}")
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"SGLang job {job_id} not done after {self._timeout}s "
                    f"(last status={status!r})")
            time.sleep(self._poll_interval)

    @staticmethod
    def _to_sglang(req: GenRequest) -> dict:
        """GenRequest -> the diffusers Turbo shim's `/v1/videos` body (serving/).

        Three H3 workflows, selected by `req.task`:
          * `t2va`  — text only, no `conditions`.
          * `fl2va` — KEYFRAME image→video: the image IS a frame. `conditions` carry
            `role:"keyframe"` + `frame_index` 0 (opening) and/or -1 (closing); at most
            two, ordered.
          * `ref2va` — REFERENCE image→video: the image is NOT a frame, only a
            subject/style/scene reference. `conditions` carry `role:"reference"` and NO
            `frame_index` (H3 ref2va accepts up to 9 image references, ordered — the
            prompt refers to them as "reference image N").
        A condition's location is `uri` (NOT `url`) and `type` is `"image"`.
        `num_inference_steps` carries the Turbo step count; it differs per workflow
        (fl2v 8-step→9, ref2v 4-step→5, off-by-one), so the shim picks the matching
        LoRA/steps by task and we send the request's own step hint too."""
        target = {
            "aspect_ratio": req.ratio,
            "duration_seconds": req.duration_s,
            # 768P == short edge 768; the shim's vocabulary is the short edge in px.
            "short_edge": 768 if req.resolution.upper() == "768P" else 480,
        }
        task = req.task
        body: dict = {
            "prompt": req.prompt,
            "task": task,
            "target": target,
            "num_inference_steps": SGLANG_STEPS,
            "num_outputs_per_prompt": 1,
        }
        if task == "fl2va":
            # First keyframe = opening frame (index 0); a second = closing (-1).
            # H3 fl2va takes at most two ordered keyframes, so ignore any extras.
            frame_indices = [0, -1]
            body["conditions"] = [
                {"type": "image", "role": "keyframe", "uri": u,
                 "frame_index": frame_indices[i]}
                for i, u in enumerate(req.keyframe_urls[:2])
            ]
        elif task == "ref2va":
            # Subject/style references — ordered, NO frame_index (not frames).
            # H3 ref2va accepts up to 9 image references.
            body["conditions"] = [
                {"type": "image", "role": "reference", "uri": u, "index": i}
                for i, u in enumerate(req.reference_urls[:9])
            ]
        return body

    @staticmethod
    def _materialize_mp4(doc: dict) -> bytes:
        """Turn a completed job doc into mp4 bytes.

        PHASE-0-VERIFIED shape: `{"file_paths": ["/root/outputs/<id>.mp4"],
        "file_path": "outputs/<id>.mp4", ...}`. The worker is co-located with SGLang
        on the GPU box, so `file_paths[0]` is a readable local file. Fallbacks: a
        direct `url`, or inline base64 `video`, in case a future build returns those."""
        paths = doc.get("file_paths") or ([doc["file_path"]] if doc.get("file_path")
                                          else [])
        for p in paths:
            if p and os.path.isfile(p):
                with open(p, "rb") as f:
                    data = f.read()
                # The clip is now in memory (about to be published to S3); SGLang
                # keeps every finished mp4 under outputs/ forever, so delete it here
                # to stop the GPU box's disk from growing unbounded across jobs.
                for q in paths:
                    try:
                        if q and os.path.isfile(q):
                            os.remove(q)
                    except OSError:
                        pass  # best-effort cleanup; never fail a good generation
                return data
        if doc.get("url"):
            with urllib.request.urlopen(doc["url"], timeout=SGLANG_TIMEOUT_S) as r:
                return r.read()
        if doc.get("video"):
            import base64
            return base64.b64decode(doc["video"])
        raise RuntimeError(
            f"SGLang completed but no readable mp4 (file_paths={paths!r})")


class FakeBackend:
    """No GPU: returns a minimal valid mp4 so the pipeline is end-to-end testable.

    The bytes are a real (empty) ISO-BMFF `ftyp`+`moov` skeleton — enough that
    downstream code treating it as an mp4 (content-type, extension, an S3 put) is
    exercised for real. Generation is instant."""

    _MP4 = (b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
            b"\x00\x00\x00\x08free")

    def __init__(self, *, fail_with: str | None = None):
        self._fail_with = fail_with

    def generate(self, req: GenRequest) -> bytes:
        if self._fail_with:
            raise RuntimeError(self._fail_with)
        return self._MP4


# --- image backend (FLUX on cuda:2) -----------------------------------------


class ImageBackend(Protocol):
    def generate(self, req: ImageRequest) -> list[bytes]:
        """Produce `req.n` images (PNG bytes each) for `req`, or raise on failure."""
        ...


class FluxImageBackend:
    """Drives the private FLUX image shim (serving/flux_image_server.py) on cuda:2.

    Mirrors SGLangBackend's async submit-then-poll, so the image worker looks exactly
    like the video worker: POST /v1/images returns `{"id":..,"status":"queued"}`; poll
    GET /v1/images/{id} until `completed`/`failed`, then read the finished PNG(s) from
    the local `file_paths` the shim writes on the GPU box (worker is co-located). A
    request for `n` images yields `n` file paths. `url`/inline base64 handled as a
    fallback in case a future build returns images inline."""

    def __init__(self, url: str = FLUX_URL, timeout_s: int = FLUX_TIMEOUT_S,
                 poll_interval_s: float = FLUX_POLL_INTERVAL_S):
        self._url = url.rstrip("/")
        self._timeout = timeout_s
        self._poll_interval = poll_interval_s
        self._poll_timeout = 30

    def generate(self, req: ImageRequest) -> list[bytes]:
        payload = self._to_flux(req)
        data = json.dumps(payload).encode()
        http = urllib.request.Request(
            f"{self._url}/v1/images", data=data, method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(http, timeout=self._poll_timeout) as resp:
            submit = json.loads(resp.read())
        job_id = submit.get("id")
        if not job_id:
            raise RuntimeError(f"FLUX submit returned no id: {submit!r}")
        return self._await_result(job_id)

    def _await_result(self, job_id: str) -> list[bytes]:
        import time
        deadline = time.monotonic() + self._timeout
        url = f"{self._url}/v1/images/{job_id}"
        while True:
            with urllib.request.urlopen(url, timeout=self._poll_timeout) as resp:
                doc = json.loads(resp.read())
            status = (doc.get("status") or "").lower()
            if status in ("completed", "succeeded"):
                return self._materialize(doc)
            if status in ("failed", "cancelled", "error"):
                err = doc.get("error") or doc.get("message") or "FLUX job failed"
                raise RuntimeError(f"FLUX job {job_id} {status}: {err}")
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"FLUX job {job_id} not done after {self._timeout}s "
                    f"(last status={status!r})")
            time.sleep(self._poll_interval)

    @staticmethod
    def _to_flux(req: ImageRequest) -> dict:
        """ImageRequest -> FLUX shim `/v1/images` body.

        Sends explicit `width`/`height` when the caller resolved them (image-01 gives
        dimensions priority), else the `aspect_ratio` for the shim to size. A single
        subject reference (a face to keep consistent) rides along as `subject_reference`
        — the shim decides whether/how to condition on it."""
        body: dict = {"prompt": req.prompt, "n": max(1, req.n)}
        if req.width and req.height:
            body["width"], body["height"] = req.width, req.height
        else:
            body["aspect_ratio"] = req.ratio
        if req.subject_reference:
            body["subject_reference"] = req.subject_reference[:1]
        return body

    @staticmethod
    def _materialize(doc: dict) -> list[bytes]:
        """A completed job doc -> a list of PNG byte-strings (one per image)."""
        out: list[bytes] = []
        for p in (doc.get("file_paths") or []):
            if p and os.path.isfile(p):
                with open(p, "rb") as f:
                    out.append(f.read())
        if out:
            # Clean up the shim's on-disk copies now they're in memory (about to be
            # published) — mirrors the video path, keeps the GPU box's disk bounded.
            for p in (doc.get("file_paths") or []):
                try:
                    if p and os.path.isfile(p):
                        os.remove(p)
                except OSError:
                    pass  # best-effort; never fail a good generation
            return out
        # Fallbacks: inline urls / base64, in case a future shim returns images inline.
        for u in (doc.get("urls") or []):
            with urllib.request.urlopen(u, timeout=FLUX_TIMEOUT_S) as r:
                out.append(r.read())
        if out:
            return out
        for b64 in (doc.get("images") or []):
            import base64
            out.append(base64.b64decode(b64))
        if not out:
            raise RuntimeError("FLUX completed but returned no readable image")
        return out


class FakeImageBackend:
    """No GPU: returns `req.n` minimal valid PNGs so the image pipeline is end-to-end
    testable. Each is a real 1×1 PNG (valid signature + IHDR/IDAT/IEND), so downstream
    code treating it as a png (content-type, extension, an S3 put) runs for real."""

    # A minimal valid 1×1 transparent PNG.
    _PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
            b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")

    def __init__(self, *, fail_with: str | None = None):
        self._fail_with = fail_with

    def generate(self, req: ImageRequest) -> list[bytes]:
        if self._fail_with:
            raise RuntimeError(self._fail_with)
        return [self._PNG] * max(1, req.n)
