"""MiniMax v2 wire-protocol translation.

This is the exact shape a MiniMax-compatible client sends and reads — see
docs/API.md, which documents the same MiniMax v2/v1 wire shapes this file implements.
Everything here is pure (dicts in, dicts out), so it's the most heavily tested part
of the gateway and needs no GPU or network.

Two directions:
  - `parse_video_request`  : client's `/v2/video_generation` body -> our internal
                             GenRequest (prompt text, ordered reference-image URLs,
                             resolution, duration, ratio).
  - `poll_response`        : a Task -> the `{ "task": { ... } }` body the client
                             polls, with status mapped to the client's vocabulary.

The client's rules this encodes (all verified against its behaviour):
  * v2 has NO `prompt` field. The prompt is a `content[]` item {type:"text", text}.
  * reference images are further `content[]` items {type:"image_url",
    role:"reference_image", image_url:{url}}, ORDERED — the text refers to them as
    "reference image N" (N from 1). We keep that order. These map to H3 `ref2va`
    (subject/style reference — the image does NOT become a frame).
  * a first/last FRAME image (H3 `fl2va`, image IS a frame) is a different mode,
    selected by MiniMax's `first_frame_image`/`last_frame_image` top-level fields or a
    content item tagged role `first_frame`/`last_frame`. It is mutually exclusive with
    reference images. See `parse_video_request`.
  * `resolution` / `duration` / `ratio` are the knobs; `duration` is a whole-second
    int, `ratio` is (already-normalized-by-the-client) one of a fixed enum.
  * We are lenient on parse: a missing text item is an empty prompt (the client
    shouldn't send that, but we don't 500 on it); non-image content items are
    ignored rather than rejected, so a future client addition doesn't break us.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.tasks import FAILED, QUEUED, RUNNING, SUCCEEDED, Task

# H3 only serves 768P locally (2K is cloud-only, not open) — see docs/PLAN.md §4.2.
# The client already sends "768P"; we accept whatever it sends and let the backend
# adapter decide, but default to this.
DEFAULT_RESOLUTION = "768P"
DEFAULT_RATIO = "16:9"
DEFAULT_DURATION_S = 6


class BadRequest(Exception):
    """The submit body is unusable (e.g. wrong model, no content). Maps to a 4xx
    with an OpenAI-shaped error body the client can read."""


@dataclass
class GenRequest:
    """What the worker needs to drive one generation, protocol-independent.

    An image can play one of TWO roles, which are DIFFERENT H3 workflows and must not
    be mixed (see `task`):
      * `reference_urls` — subject/style/scene references (H3 `ref2va`). They steer the
        content but do NOT become a frame. This is what a MiniMax-compatible client
        sends (`role:"reference_image"`, prompts say "reference image N").
      * `keyframe_urls`  — first (and optionally last) FRAME the video is built from
        (H3 `fl2va`). `keyframe_urls[0]` is the opening frame, an optional
        `keyframe_urls[1]` is the closing frame."""
    prompt: str
    reference_urls: list[str] = field(default_factory=list)
    keyframe_urls: list[str] = field(default_factory=list)
    resolution: str = DEFAULT_RESOLUTION
    ratio: str = DEFAULT_RATIO
    duration_s: int = DEFAULT_DURATION_S

    @property
    def task(self) -> str:
        """H3's workflow: `fl2va` (keyframe image→video), `ref2va` (reference
        image→video, image not a frame), or `t2va` (text only). Keyframes win if
        somehow both are set, but parse rejects that combination up front."""
        if self.keyframe_urls:
            return "fl2va"
        if self.reference_urls:
            return "ref2va"
        return "t2va"

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt, "reference_urls": self.reference_urls,
            "keyframe_urls": self.keyframe_urls,
            "resolution": self.resolution, "ratio": self.ratio,
            "duration_s": self.duration_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GenRequest":
        return cls(
            prompt=d.get("prompt", ""),
            reference_urls=list(d.get("reference_urls") or []),
            keyframe_urls=list(d.get("keyframe_urls") or []),
            resolution=d.get("resolution") or DEFAULT_RESOLUTION,
            ratio=d.get("ratio") or DEFAULT_RATIO,
            duration_s=int(d.get("duration_s") or DEFAULT_DURATION_S))


# content[] image roles that mean "first/last FRAME" (H3 fl2va) rather than a
# subject reference (ref2va). MiniMax's official API spells the frame case with a
# top-level `first_frame_image`/`last_frame_image`; a client may also tag a content
# item with one of these roles. Everything else (incl. a MiniMax-compatible
# client's `reference_image`, or no role at all) is treated as a subject reference.
_KEYFRAME_FIRST_ROLES = {"first_frame", "first_frame_image", "keyframe", "first"}
_KEYFRAME_LAST_ROLES = {"last_frame", "last_frame_image", "last"}


def _image_url(item: dict) -> str | None:
    """Pull the URL out of a v2 `image_url` content item, tolerating shapes."""
    iu = item.get("image_url")
    url = (iu if isinstance(iu, dict) else {}).get("url")
    return url if isinstance(url, str) and url else None


def parse_video_request(body: dict) -> GenRequest:
    """Client `/v2/video_generation` body -> GenRequest. See module doc for the shape.

    Routes an image to the right H3 workflow (this is the fix for "the reference image
    showed up as the first frames"):
      * top-level `first_frame_image`/`last_frame_image`, or a content item tagged
        role `first_frame`/`last_frame` -> KEYFRAME (fl2va, image becomes a frame);
      * top-level `subject_reference`, or a content `image_url` with role
        `reference_image` / no role -> REFERENCE (ref2va, image is NOT a frame).
    Keyframe and reference are MUTUALLY EXCLUSIVE (MiniMax ships them as different
    models — I2V vs S2V; H3 as different transformer partitions): sending both is a
    hard error rather than a silently-wrong, minutes-long billed generation.

    Deliberately forgiving elsewhere: an empty content array is a hard error, but
    stray/unknown content items are skipped (forward-compatible)."""
    if not isinstance(body, dict):
        raise BadRequest("request body must be a JSON object")
    content = body.get("content")
    if not isinstance(content, list) or not content:
        raise BadRequest("`content` must be a non-empty array (v2 has no `prompt`)")

    prompt_parts: list[str] = []
    reference_urls: list[str] = []
    first_frame_url: str | None = None
    last_frame_url: str | None = None

    for item in content:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "text":
            text = item.get("text")
            if isinstance(text, str) and text:
                prompt_parts.append(text)
        elif itype == "image_url":
            url = _image_url(item)
            if not url:
                continue
            role = (item.get("role") or "").lower()
            if role in _KEYFRAME_FIRST_ROLES:
                first_frame_url = first_frame_url or url
            elif role in _KEYFRAME_LAST_ROLES:
                last_frame_url = last_frame_url or url
            else:  # reference_image, or no role → subject reference (ref2va)
                reference_urls.append(url)
        # any other item type: ignored on purpose (forward-compatible)

    # Top-level MiniMax-official fields take precedence / augment content roles.
    ff = body.get("first_frame_image")
    if isinstance(ff, str) and ff:
        first_frame_url = first_frame_url or ff
    lf = body.get("last_frame_image")
    if isinstance(lf, str) and lf:
        last_frame_url = last_frame_url or lf
    sr = body.get("subject_reference")
    for u in _subject_reference_urls(sr):
        reference_urls.append(u)

    keyframe_urls = [u for u in (first_frame_url, last_frame_url) if u]
    if keyframe_urls and reference_urls:
        raise BadRequest(
            "a first/last-frame image and a subject reference cannot be combined in "
            "one request — they are different generation modes (pick one)")

    duration = body.get("duration")
    try:
        duration_s = int(duration) if duration is not None else DEFAULT_DURATION_S
    except (TypeError, ValueError):
        duration_s = DEFAULT_DURATION_S

    return GenRequest(
        prompt=" ".join(prompt_parts),
        reference_urls=reference_urls,
        keyframe_urls=keyframe_urls,
        resolution=body.get("resolution") or DEFAULT_RESOLUTION,
        ratio=body.get("ratio") or DEFAULT_RATIO,
        duration_s=duration_s)


def _subject_reference_urls(sr) -> list[str]:
    """MiniMax's `subject_reference` is a list of {type, image_file:[urls]} (S2V).
    Accept that, a bare list of urls, or a single url string — skip anything else."""
    out: list[str] = []
    if isinstance(sr, str) and sr:
        return [sr]
    if not isinstance(sr, list):
        return out
    for entry in sr:
        if isinstance(entry, str) and entry:
            out.append(entry)
        elif isinstance(entry, dict):
            imgs = entry.get("image_file") or entry.get("image") or entry.get("url")
            if isinstance(imgs, str) and imgs:
                out.append(imgs)
            elif isinstance(imgs, list):
                out.extend(u for u in imgs if isinstance(u, str) and u)
    return out


# Our internal task status -> the client's poll vocabulary. We never emit the
# client's `cancelled`, but it maps it to failed regardless.
_STATUS_OUT = {QUEUED: "queued", RUNNING: "running",
               SUCCEEDED: "succeeded", FAILED: "failed"}


def poll_response(task: Task, *, resolve_url=None) -> dict:
    """A Task -> the `{ "task": { ... } }` body the client polls (CONTRACT §2).

    `task.url` holds a DURABLE reference (an `s3://bucket/key`, or a file:// URL on
    the local path). `resolve_url` turns it into the URL the client downloads — the
    poll handler passes `publish.presign_s3_ref` so the presigned URL is minted HERE,
    at read time, fresh on every poll (see publish.py for why signing is split from
    upload). Default is identity, so callers with already-usable URLs (LocalPublisher,
    tests) need pass nothing.

    Rules the client enforces, mirrored here:
      * `succeeded` MUST carry `content.url`, or the client treats it as failed. We
        only set SUCCEEDED after the worker stored a reference, so this holds; the
        guard is belt-and-suspenders.
      * `failed` carries `error.message` for the UI.
      * queued/running carry neither."""
    status = _STATUS_OUT.get(task.status, "running")   # unknown -> keep polling
    if status == "succeeded" and not task.url:
        # Shouldn't happen (we gate the SUCCEEDED transition on a reference), but
        # never hand the client a succeed it can't download.
        status, task_error = "failed", "no media returned"
    else:
        task_error = task.error

    inner: dict = {"status": status}
    if status == "succeeded":
        url = resolve_url(task.url) if resolve_url else task.url
        inner["content"] = {"url": url}
    elif status == "failed":
        inner["error"] = {"message": task_error or "generation failed"}
    return {"task": inner}


def error_body(message: str, etype: str = "invalid_request_error") -> dict:
    """The OpenAI-shaped error the client reads `error.message` out of (CONTRACT §1)."""
    return {"type": "error", "error": {"type": etype, "message": message}}


# --- MiniMax v1: image (image-01) -------------------------------------------
#
# A DIFFERENT API generation from v2 video, verified against a real MiniMax v1
# client and its tests:
#   request  = flat body {model, prompt, n (1-9), response_format:"url",
#              width/height OR aspect_ratio (mutually exclusive), subject_reference?}
#   response = {"data": {"image_urls": [...]}, "base_resp": {"status_code": 0}}
#              — note `data.image_urls` (object-with-array), NOT `data[].url`.
#   a REFUSAL is HTTP 200 + non-zero base_resp.status_code; an empty image_urls makes
#   the client raise "no image returned", so a no-output must be a non-zero base_resp.
# Synchronous FOR THE CLIENT (one call, no task id): our gateway satisfies that by
# enqueuing an image task and short-polling it to completion in-request
# (lambdas/api.py / app/main.py), because the GPU box is reached only via the queue.

_IMAGE_MAX_N = 9              # image-01's `n` range is 1-9
_IMAGE_OK = 0                # base_resp.status_code for success
IMAGE_ERROR_CODE = 1500      # our non-zero code for a failed/timed-out image request


@dataclass
class ImageRequest:
    """What the image backend needs to drive one text-to-image generation.

    Dimensions vs ratio mirror image-01: `width`/`height` (pixels) take priority when
    present, else `aspect_ratio`. `subject_reference` is at most one character face to
    keep consistent (image-01's only reference kind)."""
    prompt: str
    n: int = 1
    width: int = 0
    height: int = 0
    ratio: str = DEFAULT_RATIO
    subject_reference: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"prompt": self.prompt, "n": self.n, "width": self.width,
                "height": self.height, "ratio": self.ratio,
                "subject_reference": self.subject_reference}

    @classmethod
    def from_dict(cls, d: dict) -> "ImageRequest":
        return cls(
            prompt=d.get("prompt", ""),
            n=int(d.get("n") or 1),
            width=int(d.get("width") or 0),
            height=int(d.get("height") or 0),
            ratio=d.get("ratio") or DEFAULT_RATIO,
            subject_reference=list(d.get("subject_reference") or []))


def parse_image_request(body: dict) -> ImageRequest:
    """Client `POST /v1/image_generation` body -> ImageRequest (see the block above).

    Forgiving in the same spirit as the video parser: a missing prompt is a hard error
    (nothing to generate), but unknown fields are ignored. `n` is clamped to
    image-01's 1-9. `subject_reference` is a list of `{type:"character",
    image_file:<url>}`; we keep the urls in order (image-01 uses only the first)."""
    if not isinstance(body, dict):
        raise BadRequest("request body must be a JSON object")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise BadRequest("`prompt` is required")

    n = body.get("n")
    try:
        n = max(1, min(_IMAGE_MAX_N, int(n))) if n is not None else 1
    except (TypeError, ValueError):
        n = 1

    try:
        width = int(body.get("width") or 0)
        height = int(body.get("height") or 0)
    except (TypeError, ValueError):
        width = height = 0

    # `subject_reference` reuses the video parser's tolerant extractor (S2V shape,
    # bare list, or a single url string) — image-01 keeps only the first.
    refs = _subject_reference_urls(body.get("subject_reference"))

    return ImageRequest(
        prompt=prompt, n=n, width=width, height=height,
        ratio=body.get("aspect_ratio") or DEFAULT_RATIO,
        subject_reference=refs)


def image_response(refs: list[str], *, resolve_url=None) -> dict:
    """A list of DURABLE media refs -> the v1 body the client reads.

    Each ref is resolved to a downloadable URL via `resolve_url` (the handler passes
    `publish.presign_s3_ref`, same split-signing as the video poll), then placed under
    `data.image_urls` — the exact path the client reads (providers/minimax.py:394).
    The caller must not pass an empty list: the client raises on empty image_urls, so
    a no-output generation becomes `image_error_body` instead."""
    urls = [resolve_url(r) if resolve_url else r for r in refs]
    return {"data": {"image_urls": urls},
            "base_resp": {"status_code": _IMAGE_OK, "status_msg": "success"}}


def image_error_body(message: str, status_code: int = IMAGE_ERROR_CODE) -> dict:
    """A v1 refusal: HTTP 200 + non-zero base_resp, which the client surfaces as the
    failure reason (it reads `base_resp.status_msg`). Used for a bad request, a failed
    generation, or a timeout — anything where there are no image_urls to return."""
    return {"base_resp": {"status_code": status_code, "status_msg": message}}
