"""End-to-end HTTP tests against the real FastAPI app with fakes injected.

Drives the exact requests a MiniMax-compatible client makes (docs/API.md),
with FakeBackend + LocalPublisher + in-memory sqlite — no GPU, no AWS, no network.
The worker is driven manually (start_worker=False) so the submit->poll transition is
deterministic."""

import pytest
from fastapi.testclient import TestClient

from app.backend import FakeBackend, FakeImageBackend
from app.keys import KeyStore
from app.main import Deps, create_app
from app.publish import LocalPublisher
from app.tasks import IMAGE, VIDEO, TaskStore
from app.worker import Worker

ADMIN = "admin-secret"


@pytest.fixture
def ctx():
    keys, tasks = KeyStore(), TaskStore()
    backend, publisher = FakeBackend(), LocalPublisher()
    image_backend = FakeImageBackend()
    worker = Worker(tasks=tasks, keys=keys, backend=backend, publisher=publisher,
                    kind=VIDEO)
    image_worker = Worker(tasks=tasks, keys=keys, backend=image_backend,
                          publisher=publisher, kind=IMAGE)
    deps = Deps(keys=keys, tasks=tasks, backend=backend, publisher=publisher,
                worker=worker, admin_token=ADMIN, image_backend=image_backend,
                image_worker=image_worker)
    app = create_app(deps, start_worker=False)   # we pump the worker(s) by hand
    return TestClient(app), deps


def _issue(client, **kw):
    r = client.post("/admin/keys", json=kw, headers={"X-Admin-Token": ADMIN})
    assert r.status_code == 200
    return r.json()["api_key"]


def _auth(key):
    return {"Authorization": f"Bearer {key}"}


# --- the video contract, end to end -----------------------------------------

def test_submit_returns_task_id_then_poll_reaches_succeeded(ctx):
    client, deps = ctx
    key = _issue(client)

    submit = client.post("/v2/video_generation", headers=_auth(key), json={
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": "a castle at dawn"}],
        "resolution": "768P", "duration": 6, "ratio": "16:9"})
    assert submit.status_code == 200
    task_id = submit.json()["task_id"]
    assert task_id

    # Before the worker runs: queued/running, no media (client keeps polling).
    poll = client.get(f"/v2/query/video_generation/{task_id}", headers=_auth(key))
    assert poll.json()["task"]["status"] in ("queued", "running")

    deps.worker.run_once()                        # generate + publish

    poll = client.get(f"/v2/query/video_generation/{task_id}", headers=_auth(key))
    body = poll.json()["task"]
    assert body["status"] == "succeeded"
    assert body["content"]["url"].endswith(".mp4")


def test_a_generation_failure_polls_as_failed_with_reason(ctx):
    client, deps = ctx
    deps.backend = FakeBackend(fail_with="cuda oom")   # swap in a failing backend
    deps.worker._backend = deps.backend
    key = _issue(client)
    task_id = client.post("/v2/video_generation", headers=_auth(key), json={
        "content": [{"type": "text", "text": "x"}]}).json()["task_id"]
    deps.worker.run_once()
    body = client.get(f"/v2/query/video_generation/{task_id}",
                      headers=_auth(key)).json()["task"]
    assert body["status"] == "failed"
    assert "cuda oom" in body["error"]["message"]


# --- auth + isolation -------------------------------------------------------

def test_missing_or_bad_key_is_401(ctx):
    client, _ = ctx
    body = {"content": [{"type": "text", "text": "x"}]}
    assert client.post("/v2/video_generation", json=body).status_code == 401
    assert client.post("/v2/video_generation", json=body,
                       headers=_auth("mmh3_bogus_key")).status_code == 401


def test_a_revoked_key_stops_working(ctx):
    client, deps = ctx
    key = _issue(client)
    prefix = key.split("_")[1]
    assert client.delete(f"/admin/keys/{prefix}",
                         headers={"X-Admin-Token": ADMIN}).status_code == 200
    r = client.post("/v2/video_generation", headers=_auth(key),
                    json={"content": [{"type": "text", "text": "x"}]})
    assert r.status_code == 401


def test_a_task_is_only_visible_to_its_owner(ctx):
    client, deps = ctx
    owner = _issue(client)
    other = _issue(client)
    task_id = client.post("/v2/video_generation", headers=_auth(owner),
                          json={"content": [{"type": "text", "text": "x"}]}
                          ).json()["task_id"]
    # Another valid key must not be able to read someone else's task.
    r = client.get(f"/v2/query/video_generation/{task_id}", headers=_auth(other))
    assert r.status_code == 404


def test_unknown_task_id_is_terminal_404(ctx):
    client, _ = ctx
    key = _issue(client)
    r = client.get("/v2/query/video_generation/vt_does_not_exist", headers=_auth(key))
    assert r.status_code == 404   # non-429/5xx 4xx => client treats as terminal


# --- rate limiting ----------------------------------------------------------

def test_rate_limit_returns_429_with_retry_after(ctx):
    client, _ = ctx
    key = _issue(client, rate_limit_per_min=2)
    body = {"content": [{"type": "text", "text": "x"}]}
    assert client.post("/v2/video_generation", headers=_auth(key), json=body).status_code == 200
    assert client.post("/v2/video_generation", headers=_auth(key), json=body).status_code == 200
    r = client.post("/v2/video_generation", headers=_auth(key), json=body)
    assert r.status_code == 429                    # transient to the client
    assert int(r.headers["Retry-After"]) >= 1


# --- image path (MiniMax v1 image-01, synchronous for the client) -----------


def _pump_image_worker(deps):
    """Drain the image queue in a background thread while the handler short-polls
    (the handler blocks until the task is terminal, so the worker must run
    concurrently). Returns a stop() to join it."""
    import threading
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            if not deps.image_worker.run_once():
                stop.wait(0.02)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return lambda: (stop.set(), t.join(2))


def test_image_generation_returns_image_urls_synchronously(ctx):
    """v1 image-01 is synchronous FOR THE CLIENT: one POST returns the finished
    `data.image_urls`. Internally we enqueue + short-poll while the image worker
    (FakeImageBackend here) generates on its own card."""
    client, deps = ctx
    key = _issue(client)
    stop = _pump_image_worker(deps)
    try:
        r = client.post("/v1/image_generation", headers=_auth(key),
                        json={"model": "image-01", "prompt": "a red panda", "n": 2})
    finally:
        stop()
    assert r.status_code == 200
    body = r.json()
    assert body["base_resp"]["status_code"] == 0
    urls = body["data"]["image_urls"]          # NOTE: data.image_urls, not data[].url
    assert len(urls) == 2 and all(u.endswith(".png") for u in urls)


def test_image_generation_failure_is_a_readable_v1_refusal(ctx):
    """A failed generation must be HTTP 200 + non-zero base_resp (the v1 dialect the
    client reads), never a 5xx or a success with no images."""
    client, deps = ctx
    from app.backend import FakeImageBackend
    deps.image_worker._backend = FakeImageBackend(fail_with="flux oom")
    key = _issue(client)
    stop = _pump_image_worker(deps)
    try:
        r = client.post("/v1/image_generation", headers=_auth(key),
                        json={"prompt": "x"})
    finally:
        stop()
    assert r.status_code == 200
    base = r.json()["base_resp"]
    assert base["status_code"] != 0 and "flux oom" in base["status_msg"]


def test_image_generation_requires_a_prompt(ctx):
    """A malformed v1 body is a readable refusal (200 + non-zero base_resp), not a
    400 — the client reads the body, not the HTTP status, on the image route."""
    client, _ = ctx
    key = _issue(client)
    r = client.post("/v1/image_generation", headers=_auth(key), json={})
    assert r.status_code == 200
    assert r.json()["base_resp"]["status_code"] != 0


def test_image_and_video_use_separate_queues(ctx):
    """kind routing: the video worker never claims an image task and vice-versa, so
    the two run in parallel instead of one blocking the other."""
    client, deps = ctx
    key = _issue(client)
    # Enqueue a video task; the IMAGE worker must NOT claim it.
    client.post("/v2/video_generation", headers=_auth(key),
                json={"content": [{"type": "text", "text": "x"}]})
    assert deps.image_worker.run_once() is False, "image worker claimed a video task"
    # The video worker still can.
    assert deps.worker.run_once() is True


# --- admin auth -------------------------------------------------------------

def test_admin_requires_the_admin_token(ctx):
    client, _ = ctx
    assert client.post("/admin/keys", json={}).status_code == 403
    assert client.get("/admin/keys").status_code == 403
    assert client.post("/admin/keys", json={},
                       headers={"X-Admin-Token": "wrong"}).status_code == 403


def test_healthz_needs_no_auth(ctx):
    client, _ = ctx
    assert client.get("/healthz").json() == {"ok": True}
