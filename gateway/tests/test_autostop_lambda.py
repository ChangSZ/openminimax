"""EventBridge autostop Lambda: starts/stops the GPU on queue state alone (no CPU),
so it can never stop the box mid-generation. moto-backed DynamoDB + a fake EC2."""

import boto3
import pytest
from moto import mock_aws

from app.tasks import DynamoDBTaskStore
from lambdas import autostop

TASKS_TABLE = "mmh3-tasks"


@pytest.fixture
def env(monkeypatch):
    with mock_aws():
        ddb = boto3.client("dynamodb", region_name="us-west-2")
        ddb.create_table(
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
        monkeypatch.setattr(autostop, "TASKS_TABLE", TASKS_TABLE)
        monkeypatch.setattr(autostop, "GPU_INSTANCE_ID", "i-gpu")
        monkeypatch.setattr(autostop, "IDLE_STOP_MINUTES", 15.0)
        tasks = DynamoDBTaskStore(TASKS_TABLE, client=ddb, now=lambda: 1000.0)
        yield {"ddb": ddb, "tasks": tasks}


class _FakeEc2:
    def __init__(self, state):
        self.state, self.calls = state, []
    def describe_instances(self, InstanceIds):
        return {"Reservations": [{"Instances": [{"State": {"Name": self.state}}]}]}
    def start_instances(self, InstanceIds):
        self.calls.append(("start", tuple(InstanceIds)))
    def stop_instances(self, InstanceIds):
        self.calls.append(("stop", tuple(InstanceIds)))


def test_starts_gpu_when_work_is_queued(env):
    env["tasks"].enqueue(key_prefix="p", request={"prompt": "x"}, duration_s=6)
    ec2 = _FakeEc2("stopped")
    out = autostop.handler(ddb=env["ddb"], ec2=ec2, now=lambda: 2000.0)
    assert out["action"] == "start"
    assert ec2.calls == [("start", ("i-gpu",))]


def test_stops_gpu_after_idle_window(env):
    tid = env["tasks"].enqueue(key_prefix="p", request={"prompt": "x"}, duration_s=6)
    env["tasks"].claim_next()
    env["tasks"].mark_succeeded(tid, "u")        # newest updated_at ~ 1000
    ec2 = _FakeEc2("running")
    # now well past 1000 + 15min -> idle elapsed, nothing queued/running.
    out = autostop.handler(ddb=env["ddb"], ec2=ec2, now=lambda: 1000.0 + 16 * 60)
    assert out["action"] == "stop"
    assert ec2.calls == [("stop", ("i-gpu",))]


def test_never_stops_while_a_task_is_running(env):
    env["tasks"].enqueue(key_prefix="p", request={"prompt": "x"}, duration_s=6)
    env["tasks"].claim_next()                     # RUNNING => work in flight
    ec2 = _FakeEc2("running")
    out = autostop.handler(ddb=env["ddb"], ec2=ec2, now=lambda: 9e12)
    assert out["action"] == "none"
    assert ec2.calls == []


def test_no_action_when_idle_but_not_long_enough(env):
    tid = env["tasks"].enqueue(key_prefix="p", request={"prompt": "x"}, duration_s=6)
    env["tasks"].claim_next()
    env["tasks"].mark_succeeded(tid, "u")        # updated_at ~ 1000
    ec2 = _FakeEc2("running")
    out = autostop.handler(ddb=env["ddb"], ec2=ec2, now=lambda: 1000.0 + 5 * 60)
    assert out["action"] == "none"               # only 5 min idle < 15
    assert ec2.calls == []
