"""DynamoDBKeyStore / DynamoDBTaskStore behave identically to the SQLite stores.

Uses moto to run DynamoDB in-memory (no AWS). These pin that the serverless path
(Lambda authorizer + submit/poll Lambdas + GPU-box worker, all sharing one table)
gets the same auth, rate-limit, isolation and queue semantics the SQLite suite
already proves for the single-box path."""

import boto3
import pytest
from moto import mock_aws

from app.keys import DynamoDBKeyStore, RateLimitedError, RevokedError
from app.tasks import DynamoDBTaskStore

KEYS_TABLE = "mmh3-keys"
TASKS_TABLE = "mmh3-tasks"


@pytest.fixture
def ddb():
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
                {   # kind-filtered claim: query "<kind>#queued" oldest-first
                    "IndexName": DynamoDBTaskStore.KIND_STATUS_INDEX,
                    "KeySchema": [
                        {"AttributeName": "kind_status", "KeyType": "HASH"},
                        {"AttributeName": "created_at", "KeyType": "RANGE"}],
                    "Projection": {"ProjectionType": "ALL"}},
                {   # kind-agnostic requeue of stale RUNNING items
                    "IndexName": DynamoDBTaskStore.STATUS_INDEX,
                    "KeySchema": [
                        {"AttributeName": "status", "KeyType": "HASH"},
                        {"AttributeName": "created_at", "KeyType": "RANGE"}],
                    "Projection": {"ProjectionType": "ALL"}}],
            BillingMode="PAY_PER_REQUEST")
        yield client


# --- keys -------------------------------------------------------------------

def test_dynamo_key_issue_verify_revoke(ddb):
    store = DynamoDBKeyStore(KEYS_TABLE, client=ddb)
    key = store.issue(label="team-1")
    assert " " not in key and 20 <= len(key) <= 4000
    info = store.verify(key)
    assert info.label == "team-1"

    prefix = key.split("_")[1]
    assert store.revoke(prefix) is True
    with pytest.raises(RevokedError):
        store.verify(key)
    assert store.revoke("nope") is False


def test_dynamo_key_rejects_bad_and_unknown(ddb):
    store = DynamoDBKeyStore(KEYS_TABLE, client=ddb)
    key = store.issue()
    prefix = key.split("_")[1]
    with pytest.raises(KeyError):
        store.verify(f"mmh3_{prefix}_wrongwrongwrongwrong")
    with pytest.raises(KeyError):
        store.verify("mmh3_deadbeef_nope")     # unknown prefix
    with pytest.raises(KeyError):
        store.verify("garbage")                 # malformed


def test_dynamo_raw_secret_never_stored(ddb):
    store = DynamoDBKeyStore(KEYS_TABLE, client=ddb)
    key = store.issue()
    # The secret may itself contain '_'; split with maxsplit=2 like parse() does.
    secret = key.split("_", 2)[2]
    assert len(secret) >= 20                       # guard against a bad split
    dump = ddb.scan(TableName=KEYS_TABLE)["Items"]
    assert secret not in str(dump)


def test_dynamo_rate_limit_fixed_window(ddb):
    clock = {"t": 1000.0}
    store = DynamoDBKeyStore(KEYS_TABLE, client=ddb, now=lambda: clock["t"])
    key = store.issue(rate_limit_per_min=3)
    prefix = key.split("_")[1]
    for _ in range(3):
        store.check_and_count_submit(prefix)
    with pytest.raises(RateLimitedError) as exc:
        store.check_and_count_submit(prefix)
    assert 1 <= exc.value.retry_after_s <= 60
    clock["t"] += 61
    store.check_and_count_submit(prefix)        # window rolled


def test_dynamo_rate_limit_isolates_keys_and_meters(ddb):
    store = DynamoDBKeyStore(KEYS_TABLE, client=ddb)
    a = store.issue(rate_limit_per_min=1).split("_")[1]
    b = store.issue(rate_limit_per_min=1).split("_")[1]
    store.check_and_count_submit(a)
    with pytest.raises(RateLimitedError):
        store.check_and_count_submit(a)
    store.check_and_count_submit(b)             # b unaffected
    store.add_seconds_billed(a, 6)
    rows = {r["prefix"]: r for r in store.list_keys()}
    assert rows[a]["submits"] == 1 and rows[a]["seconds_billed"] == 6
    assert "secret_hash" not in rows[a]


# --- tasks ------------------------------------------------------------------

def _gen():
    return {"prompt": "hi", "duration_s": 6}


def test_dynamo_task_enqueue_get_and_fifo_claim(ddb):
    clock = {"t": 1000.0}
    def now():
        clock["t"] += 1        # strictly increasing created_at for FIFO ordering
        return clock["t"]
    store = DynamoDBTaskStore(TASKS_TABLE, client=ddb, now=now)
    a = store.enqueue(key_prefix="p", request=_gen(), duration_s=6)
    b = store.enqueue(key_prefix="p", request=_gen(), duration_s=6)

    assert store.get(a).status == "queued"
    first = store.claim_next()
    assert first.task_id == a and first.status == "running"
    second = store.claim_next()
    assert second.task_id == b
    assert store.claim_next() is None           # drained


def test_dynamo_task_mark_succeeded_and_failed(ddb):
    store = DynamoDBTaskStore(TASKS_TABLE, client=ddb)
    tid = store.enqueue(key_prefix="p", request=_gen(), duration_s=6)
    store.mark_succeeded(tid, "https://cdn/out.mp4")
    t = store.get(tid)
    assert t.status == "succeeded" and t.url == "https://cdn/out.mp4"

    tid2 = store.enqueue(key_prefix="p", request=_gen(), duration_s=6)
    store.mark_failed(tid2, "cuda oom")
    assert store.get(tid2).status == "failed"
    assert store.get(tid2).error == "cuda oom"


def test_dynamo_task_requeue_stale_running(ddb):
    clock = {"t": 1000.0}
    store = DynamoDBTaskStore(TASKS_TABLE, client=ddb, now=lambda: clock["t"])
    tid = store.enqueue(key_prefix="p", request=_gen(), duration_s=6)
    store.claim_next()                          # -> running at t=1000
    clock["t"] += 4000
    assert store.requeue_stale_running(3600) == 1
    assert store.get(tid).status == "queued"
    # A freshly-running task is not requeued.
    store.claim_next()
    assert store.requeue_stale_running(3600) == 0
