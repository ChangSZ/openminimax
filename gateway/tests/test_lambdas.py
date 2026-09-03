"""Lambda authorizer + HTTP API integration handler (the serverless / API Gateway
path). Drives realistic HTTP API payload-2.0 events against moto-backed DynamoDB —
no AWS, no network. This is the "API Gateway validates the key in a Lambda" flow:
authorizer.handler decides isAuthorized, then api.handler serves submit/poll using
the prefix the authorizer put in the request context."""

import json

import boto3
import pytest
from moto import mock_aws

from app.keys import DynamoDBKeyStore
from app.tasks import DynamoDBTaskStore
from lambdas import api, authorizer

KEYS_TABLE = "mmh3-keys"
TASKS_TABLE = "mmh3-tasks"


@pytest.fixture
def stores():
    with mock_aws():
        client = boto3.client("dynamodb", region_name="us-west-2")
        client.create_table(
            TableName=KEYS_TABLE,
            AttributeDefinitions=[{"AttributeName": "prefix", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "prefix", "KeyType": "HASH"}],
            BillingMode="PAY_PER_REQUEST")
        client.create_table(
            TableName=TASKS_TABLE,
            AttributeDefinitions=[
                {"AttributeName": "task_id", "AttributeType": "S"},
                {"AttributeName": "status", "AttributeType": "S"},
                {"AttributeName": "kind_status", "AttributeType": "S"},
                {"AttributeName": "created_at", "AttributeType": "N"}],
            KeySchema=[{"AttributeName": "task_id", "KeyType": "HASH"}],
            GlobalSecondaryIndexes=[
                {"IndexName": DynamoDBTaskStore.KIND_STATUS_INDEX,
                 "KeySchema": [
                     {"AttributeName": "kind_status", "KeyType": "HASH"},
                     {"AttributeName": "created_at", "KeyType": "RANGE"}],
                 "Projection": {"ProjectionType": "ALL"}},
                {"IndexName": DynamoDBTaskStore.STATUS_INDEX,
                 "KeySchema": [
                     {"AttributeName": "status", "KeyType": "HASH"},
                     {"AttributeName": "created_at", "KeyType": "RANGE"}],
                 "Projection": {"ProjectionType": "ALL"}}],
            BillingMode="PAY_PER_REQUEST")
        keys = DynamoDBKeyStore(KEYS_TABLE, client=client)
        tasks = DynamoDBTaskStore(TASKS_TABLE, client=client)
        yield keys, tasks


def _auth_event(bearer):
    return {"headers": {"authorization": f"Bearer {bearer}"}}


def _api_event(method, path, prefix, body=None, task_id=None):
    """A minimal HTTP API payload-2.0 event with the authorizer context populated."""
    return {
        "rawPath": path,
        "requestContext": {
            "http": {"method": method},
            "authorizer": {"lambda": {"keyPrefix": prefix, "rateLimitPerMin": "6"}}},
        "pathParameters": {"task_id": task_id} if task_id else {},
        "body": json.dumps(body) if body is not None else None,
    }


# --- authorizer -------------------------------------------------------------

def test_authorizer_allows_a_valid_key_and_passes_prefix(stores):
    keys, _ = stores
    key = keys.issue(label="team-1", rate_limit_per_min=6)
    out = authorizer.handler(_auth_event(key), store=keys)
    assert out["isAuthorized"] is True
    assert out["context"]["keyPrefix"] == key.split("_")[1]
    assert out["context"]["rateLimitPerMin"] == "6"


def test_authorizer_denies_bad_missing_and_revoked_keys(stores):
    keys, _ = stores
    key = keys.issue()
    assert authorizer.handler(_auth_event("mmh3_bad_nope"), store=keys)["isAuthorized"] is False
    assert authorizer.handler({"headers": {}}, store=keys)["isAuthorized"] is False
    keys.revoke(key.split("_")[1])
    assert authorizer.handler(_auth_event(key), store=keys)["isAuthorized"] is False


def test_authorizer_never_raises_to_a_500(stores):
    """A raised authorizer becomes a 500 that API Gateway may treat as deny anyway,
    but we fail closed explicitly rather than relying on that."""
    class Boom:
        def verify(self, _):
            raise RuntimeError("dynamo down")
    assert authorizer.handler(_auth_event("x" * 30), store=Boom())["isAuthorized"] is False


# --- api handler: the contract, serverless ----------------------------------

def test_submit_then_poll_via_lambda(stores):
    keys, tasks = stores
    key = keys.issue()
    prefix = key.split("_")[1]

    submit = api.handler(_api_event(
        "POST", "/v2/video_generation", prefix,
        body={"content": [{"type": "text", "text": "a castle"}],
              "resolution": "768P", "duration": 6, "ratio": "16:9"}), stores=stores)
    assert submit["statusCode"] == 200
    task_id = json.loads(submit["body"])["task_id"]

    # Poll: queued (no worker in this test), still a valid contract response.
    poll = api.handler(_api_event(
        "GET", f"/v2/query/video_generation/{task_id}", prefix, task_id=task_id),
        stores=stores)
    assert poll["statusCode"] == 200
    assert json.loads(poll["body"])["task"]["status"] == "queued"

    # Simulate the GPU-box worker finishing it, then poll succeeds with a URL.
    tasks.mark_succeeded(task_id, "https://cdn/out.mp4")
    poll = api.handler(_api_event(
        "GET", f"/v2/query/video_generation/{task_id}", prefix, task_id=task_id),
        stores=stores)
    body = json.loads(poll["body"])["task"]
    assert body["status"] == "succeeded"
    assert body["content"]["url"] == "https://cdn/out.mp4"


def test_body_must_be_json_and_content_required(stores):
    keys, _ = stores
    prefix = keys.issue().split("_")[1]
    bad = api.handler({"rawPath": "/v2/video_generation",
                       "requestContext": {"http": {"method": "POST"},
                                          "authorizer": {"lambda": {"keyPrefix": prefix}}},
                       "body": "not json{"}, stores=stores)
    assert bad["statusCode"] == 400
    empty = api.handler(_api_event("POST", "/v2/video_generation", prefix, body={}),
                        stores=stores)
    assert empty["statusCode"] == 400


def test_missing_authorizer_context_is_401(stores):
    """If a route were ever wired without the authorizer, fail closed."""
    out = api.handler({"rawPath": "/v2/video_generation",
                       "requestContext": {"http": {"method": "POST"}},
                       "body": "{}"}, stores=stores)
    assert out["statusCode"] == 401


def test_a_task_is_only_visible_to_its_owner(stores):
    keys, tasks = stores
    owner = keys.issue().split("_")[1]
    other = keys.issue().split("_")[1]
    tid = json.loads(api.handler(_api_event(
        "POST", "/v2/video_generation", owner,
        body={"content": [{"type": "text", "text": "x"}]}), stores=stores)["body"])["task_id"]
    out = api.handler(_api_event(
        "GET", f"/v2/query/video_generation/{tid}", other, task_id=tid), stores=stores)
    assert out["statusCode"] == 404


def test_rate_limit_returns_429_with_retry_after(stores):
    keys, _ = stores
    prefix = keys.issue(rate_limit_per_min=2).split("_")[1]
    ev = lambda: _api_event("POST", "/v2/video_generation", prefix,
                            body={"content": [{"type": "text", "text": "x"}]})
    assert api.handler(ev(), stores=stores)["statusCode"] == 200
    assert api.handler(ev(), stores=stores)["statusCode"] == 200
    limited = api.handler(ev(), stores=stores)
    assert limited["statusCode"] == 429
    assert int(limited["headers"]["retry-after"]) >= 1


def _pump_image_worker(stores):
    """Drain the image queue in a thread while the lambda handler short-polls (the
    handler blocks until terminal, so the worker must run concurrently). The worker
    uses FakeImageBackend + LocalPublisher on the SAME moto tables."""
    import threading

    from app.backend import FakeImageBackend
    from app.publish import LocalPublisher
    from app.tasks import IMAGE
    from app.worker import Worker

    keys, tasks = stores
    stop = threading.Event()
    worker = Worker(tasks=tasks, keys=keys, backend=FakeImageBackend(),
                    publisher=LocalPublisher(), kind=IMAGE)

    def loop():
        while not stop.is_set():
            if not worker.run_once():
                stop.wait(0.02)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return worker, (lambda: (stop.set(), t.join(2)))


def test_image_route_returns_image_urls(stores):
    """v1 image-01 is synchronous for the client: POST returns data.image_urls once
    the image worker has produced them."""
    keys, _ = stores
    prefix = keys.issue().split("_")[1]
    _, stop = _pump_image_worker(stores)
    try:
        out = api.handler(
            _api_event("POST", "/v1/image_generation", prefix,
                       body={"model": "image-01", "prompt": "a red panda", "n": 2}),
            stores=stores)
    finally:
        stop()
    assert out["statusCode"] == 200
    body = json.loads(out["body"])
    assert body["base_resp"]["status_code"] == 0
    urls = body["data"]["image_urls"]
    assert len(urls) == 2 and all(u.endswith(".png") for u in urls)


def test_image_route_bad_body_is_readable_v1_refusal(stores):
    """A malformed body (no prompt) is HTTP 200 + non-zero base_resp — the v1 dialect
    the client reads — not a 400 and not a fake success."""
    keys, _ = stores
    prefix = keys.issue().split("_")[1]
    out = api.handler(_api_event("POST", "/v1/image_generation", prefix, body={}),
                      stores=stores)
    assert out["statusCode"] == 200
    assert json.loads(out["body"])["base_resp"]["status_code"] != 0
