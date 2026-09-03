"""TaskStore queue semantics + Worker drain loop, with fakes (no GPU/S3)."""

from app.backend import FakeBackend
from app.keys import KeyStore
from app.protocol import GenRequest
from app.publish import LocalPublisher
from app.tasks import QUEUED, RUNNING, SUCCEEDED, TaskStore
from app.worker import Worker


def _gen():
    return GenRequest(prompt="hi", duration_s=6).to_dict()


def test_claim_next_is_fifo_and_single_flight():
    store = TaskStore()
    a = store.enqueue(key_prefix="p", request=_gen(), duration_s=6)
    b = store.enqueue(key_prefix="p", request=_gen(), duration_s=6)
    first = store.claim_next()
    assert first.task_id == a and first.status == RUNNING
    assert store.get(b).status == QUEUED          # untouched, still queued
    second = store.claim_next()
    assert second.task_id == b
    assert store.claim_next() is None             # queue drained


def test_requeue_stale_running_recovers_a_crash():
    clock = {"t": 1000.0}
    store = TaskStore(now=lambda: clock["t"])
    tid = store.enqueue(key_prefix="p", request=_gen(), duration_s=6)
    store.claim_next()                             # -> running at t=1000
    clock["t"] += 4000                             # long past any real clip
    assert store.requeue_stale_running(3600) == 1
    assert store.get(tid).status == QUEUED
    # A freshly-running task is NOT requeued.
    store.claim_next()
    assert store.requeue_stale_running(3600) == 0


def _worker(backend):
    tasks, keys = TaskStore(), KeyStore()
    w = Worker(tasks=tasks, keys=keys, backend=backend, publisher=LocalPublisher())
    return w, tasks, keys


def test_worker_generates_publishes_and_meters():
    w, tasks, keys = _worker(FakeBackend())
    key = keys.issue()
    prefix = key.split("_")[1]
    tid = tasks.enqueue(key_prefix=prefix, request=_gen(), duration_s=6)

    assert w.run_once() is True
    task = tasks.get(tid)
    assert task.status == SUCCEEDED
    assert task.url.startswith("file://") and task.url.endswith(".mp4")
    assert keys.list_keys()[0]["seconds_billed"] == 6
    assert w.run_once() is False                   # nothing left


def test_worker_records_a_backend_failure_as_the_task_reason():
    w, tasks, _ = _worker(FakeBackend(fail_with="cuda oom"))
    tid = tasks.enqueue(key_prefix="p", request=_gen(), duration_s=6)
    assert w.run_once() is True                    # never raises
    task = tasks.get(tid)
    assert task.status == "failed"
    assert "cuda oom" in task.error


# --- SGLangBackend._to_sglang: the per-workflow request shapes ----------------

def test_to_sglang_text_only_is_t2va_no_conditions():
    from app.backend import SGLangBackend
    body = SGLangBackend._to_sglang(GenRequest(prompt="a fox", duration_s=6))
    assert body["task"] == "t2va"
    assert "conditions" not in body
    assert body["target"]["short_edge"] == 768


def test_to_sglang_references_become_ref2va_conditions_no_frame_index():
    """Subject references -> ref2va conditions: role 'reference', ORDERED, and NO
    frame_index (the fix — they must NOT be pinned as frames like the old bug did)."""
    from app.backend import SGLangBackend
    body = SGLangBackend._to_sglang(GenRequest(
        prompt="王山 (reference image 1)",
        reference_urls=["https://s/a.png", "https://s/b.png", "https://s/c.png"],
        duration_s=12))
    assert body["task"] == "ref2va"
    assert body["conditions"] == [
        {"type": "image", "role": "reference", "uri": "https://s/a.png", "index": 0},
        {"type": "image", "role": "reference", "uri": "https://s/b.png", "index": 1},
        {"type": "image", "role": "reference", "uri": "https://s/c.png", "index": 2},
    ]
    assert all("frame_index" not in c for c in body["conditions"])


def test_to_sglang_keyframes_become_fl2va_conditions():
    """Keyframe images -> fl2va conditions: role 'keyframe' with frame_index 0/-1."""
    from app.backend import SGLangBackend
    body = SGLangBackend._to_sglang(GenRequest(
        prompt="wave", keyframe_urls=["https://s/first.png", "https://s/last.png"],
        duration_s=6))
    assert body["task"] == "fl2va"
    assert body["conditions"] == [
        {"type": "image", "role": "keyframe", "uri": "https://s/first.png", "frame_index": 0},
        {"type": "image", "role": "keyframe", "uri": "https://s/last.png", "frame_index": -1},
    ]
    allowed = {"type", "uri", "role", "frame_index", "start_time_seconds", "index"}
    for c in body["conditions"]:
        assert set(c) <= allowed
