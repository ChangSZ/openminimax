"""protocol.py: the exact MiniMax v2 wire shapes (docs/API.md).

These pin every place the client's shape is non-obvious — the same traps a
MiniMax-compatible client's own provider tests pin, from the SERVER side."""

import pytest

from app.protocol import (
    BadRequest,
    GenRequest,
    error_body,
    parse_video_request,
    poll_response,
)
from app.tasks import Task


def test_prompt_is_a_content_item_not_a_prompt_field():
    """v2 has NO `prompt` field: the text is a content[] item (CONTRACT §1)."""
    gen = parse_video_request({
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": "a chase"}],
        "resolution": "768P", "duration": 8, "ratio": "9:16"})
    assert gen.prompt == "a chase"
    assert gen.duration_s == 8
    assert gen.ratio == "9:16"
    assert gen.resolution == "768P"
    assert gen.reference_urls == []
    assert gen.task == "t2va"


def test_reference_images_are_ordered_content_items():
    """Refs are further content[] items, in order; the text counts from image 1.
    role=reference_image → ref2va (subject reference, NOT a frame)."""
    gen = parse_video_request({
        "content": [
            {"type": "text", "text": "王山 (reference image 1) 走进 老宅 (reference image 2)"},
            {"type": "image_url", "role": "reference_image",
             "image_url": {"url": "https://s/wang.png"}},
            {"type": "image_url", "role": "reference_image",
             "image_url": {"url": "https://s/house.png"}},
        ]})
    assert gen.reference_urls == ["https://s/wang.png", "https://s/house.png"]
    assert gen.keyframe_urls == []
    assert gen.task == "ref2va"


def test_image_with_no_role_defaults_to_reference():
    """The fix: a role-less image_url is a REFERENCE (ref2va), not a
    first-frame keyframe. This is what made the ref image show up as frame 0."""
    gen = parse_video_request({
        "content": [
            {"type": "text", "text": "x"},
            {"type": "image_url", "image_url": {"url": "https://s/subj.png"}},
        ]})
    assert gen.reference_urls == ["https://s/subj.png"]
    assert gen.task == "ref2va"


def test_first_frame_role_routes_to_fl2va_keyframe():
    """role=first_frame / last_frame → fl2va; the image IS a frame."""
    gen = parse_video_request({
        "content": [
            {"type": "text", "text": "open on the shot"},
            {"type": "image_url", "role": "first_frame",
             "image_url": {"url": "https://s/first.png"}},
            {"type": "image_url", "role": "last_frame",
             "image_url": {"url": "https://s/last.png"}},
        ]})
    assert gen.keyframe_urls == ["https://s/first.png", "https://s/last.png"]
    assert gen.reference_urls == []
    assert gen.task == "fl2va"


def test_official_first_frame_image_field_routes_to_fl2va():
    """MiniMax's top-level `first_frame_image` → fl2va keyframe."""
    gen = parse_video_request({
        "content": [{"type": "text", "text": "x"}],
        "first_frame_image": "https://s/ff.png"})
    assert gen.keyframe_urls == ["https://s/ff.png"]
    assert gen.task == "fl2va"


def test_official_subject_reference_field_routes_to_ref2va():
    """MiniMax's `subject_reference` (list of {type,image_file}) → ref2va."""
    gen = parse_video_request({
        "content": [{"type": "text", "text": "x"}],
        "subject_reference": [{"type": "character",
                               "image_file": ["https://s/char.png"]}]})
    assert gen.reference_urls == ["https://s/char.png"]
    assert gen.task == "ref2va"


def test_keyframe_and_reference_together_is_rejected():
    """First-frame + subject reference are different modes (I2V vs S2V) — 400."""
    with pytest.raises(BadRequest):
        parse_video_request({
            "content": [
                {"type": "text", "text": "x"},
                {"type": "image_url", "role": "first_frame",
                 "image_url": {"url": "https://s/ff.png"}},
                {"type": "image_url", "role": "reference_image",
                 "image_url": {"url": "https://s/ref.png"}},
            ]})


def test_empty_or_missing_content_is_a_bad_request():
    """An empty content array would waste a billed generation on nothing."""
    for body in [{}, {"content": []}, {"content": "notalist"}, "notadict"]:
        with pytest.raises(BadRequest):
            parse_video_request(body)


def test_unknown_content_items_are_skipped_not_rejected():
    """Forward-compatible: a future item type shouldn't break an otherwise-valid req."""
    gen = parse_video_request({
        "content": [
            {"type": "text", "text": "hi"},
            {"type": "audio_url", "audio_url": {"url": "https://s/a.mp3"}},
            {"type": "image_url", "image_url": {"url": "https://s/i.png"}},
        ]})
    assert gen.prompt == "hi"
    assert gen.reference_urls == ["https://s/i.png"]


def test_duration_junk_falls_back_not_crashes():
    assert parse_video_request(
        {"content": [{"type": "text", "text": "x"}], "duration": "eight"}).duration_s == 6
    assert parse_video_request(
        {"content": [{"type": "text", "text": "x"}]}).duration_s == 6


# --- poll_response ----------------------------------------------------------

def _task(status, **kw):
    return Task(task_id="t", key_prefix="p", status=status, request={}, **kw)


def test_poll_queued_and_running_carry_no_media():
    assert poll_response(_task("queued")) == {"task": {"status": "queued"}}
    assert poll_response(_task("running")) == {"task": {"status": "running"}}


def test_poll_succeeded_carries_content_url():
    """CONTRACT §2: succeed MUST carry content.url or the client treats it failed."""
    out = poll_response(_task("succeeded", url="https://cdn/out.mp4"))
    assert out == {"task": {"status": "succeeded",
                            "content": {"url": "https://cdn/out.mp4"}}}


def test_poll_succeeded_without_url_degrades_to_failed():
    out = poll_response(_task("succeeded", url=""))
    assert out["task"]["status"] == "failed"
    assert out["task"]["error"]["message"]


def test_poll_failed_carries_the_reason():
    out = poll_response(_task("failed", error="content policy"))
    assert out == {"task": {"status": "failed",
                            "error": {"message": "content policy"}}}


def test_poll_resolve_url_signs_the_stored_ref_at_read_time():
    """The stored url is a durable ref; the poll handler signs it via resolve_url,
    so the URL is minted fresh on each read (publish.presign_s3_ref in prod)."""
    signed = []

    def fake_resolver(ref):
        signed.append(ref)
        return f"https://signed.example/{ref}?sig=abc"

    out = poll_response(_task("succeeded", url="s3://bkt/results/p/t.mp4"),
                        resolve_url=fake_resolver)
    assert signed == ["s3://bkt/results/p/t.mp4"]              # signed at poll time
    assert out["task"]["content"]["url"].startswith("https://signed.example/")


def test_presign_s3_ref_passes_through_non_s3():
    """file:// (LocalPublisher) and already-http refs are returned unchanged."""
    from app.publish import presign_s3_ref
    assert presign_s3_ref("file:///tmp/x.mp4") == "file:///tmp/x.mp4"
    assert presign_s3_ref("https://cdn/x.mp4") == "https://cdn/x.mp4"
    assert presign_s3_ref("") == ""


def test_error_body_is_openai_shaped():
    body = error_body("nope")
    assert body["error"]["message"] == "nope"
    assert body["type"] == "error"


def test_genrequest_roundtrips_through_dict():
    gen = GenRequest(prompt="p", reference_urls=["u"], keyframe_urls=["k"],
                     resolution="768P", ratio="16:9", duration_s=6)
    assert GenRequest.from_dict(gen.to_dict()) == gen


def test_genrequest_from_legacy_dict_without_keyframe_field():
    """Old queued tasks serialized before keyframe_urls existed still load."""
    gen = GenRequest.from_dict({"prompt": "p", "reference_urls": ["u"],
                                "resolution": "768P", "ratio": "16:9",
                                "duration_s": 6})
    assert gen.keyframe_urls == []
    assert gen.task == "ref2va"
