"""app.admin_keys CLI + app.worker_main wiring, on the DynamoDB path (moto)."""

import boto3
import pytest
from moto import mock_aws

from app import admin_keys, worker_main
from app.keys import DynamoDBKeyStore
from app.tasks import DynamoDBTaskStore

KEYS_TABLE = "mmh3-keys"
TASKS_TABLE = "mmh3-tasks"


@pytest.fixture
def aws(monkeypatch):
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
        monkeypatch.setenv("KEYS_TABLE", KEYS_TABLE)
        monkeypatch.setenv("TASKS_TABLE", TASKS_TABLE)
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        yield client


def test_admin_issue_list_revoke_roundtrip(aws, capsys):
    admin_keys.main(["issue", "--label", "team-3", "--rate", "4"])
    printed = capsys.readouterr().out.strip().splitlines()
    key = printed[0]
    assert key.startswith("mmh3_")

    admin_keys.main(["list"])
    listed = capsys.readouterr().out
    prefix = key.split("_")[1]
    assert prefix in listed and "team-3" in listed and "active" in listed

    admin_keys.main(["revoke", prefix])
    assert "revoked" in capsys.readouterr().out
    admin_keys.main(["list"])
    assert "REVOKED" in capsys.readouterr().out

    # And the issued key really verifies against the shared store (authorizer path).
    DynamoDBKeyStore(KEYS_TABLE, client=aws)  # table exists
    with pytest.raises(Exception):
        DynamoDBKeyStore(KEYS_TABLE, client=aws).verify(key)  # revoked -> raises


def test_worker_main_smoke_mode_drains_the_dynamo_queue(aws, monkeypatch):
    """USE_FAKE_BACKEND=1 wires FakeBackend + LocalPublisher onto the DynamoDB stores;
    a queued task drains to succeeded with a file:// URL — no GPU, no S3."""
    monkeypatch.setenv("USE_FAKE_BACKEND", "1")
    keys = DynamoDBKeyStore(KEYS_TABLE, client=aws)
    tasks = DynamoDBTaskStore(TASKS_TABLE, client=aws)
    prefix = keys.issue().split("_")[1]
    tid = tasks.enqueue(key_prefix=prefix, request={"prompt": "x", "duration_s": 6},
                        duration_s=6)

    worker = worker_main.build_worker()
    assert worker.run_once() is True
    task = tasks.get(tid)
    assert task.status == "succeeded"
    assert task.url.endswith(".mp4")
    assert keys.list_keys()[0]["seconds_billed"] == 6
    assert worker.run_once() is False   # queue drained
