"""EventBridge-scheduled autostop Lambda — the serverless scale-to-zero controller.

Replaces the always-on t3.small + systemd controller (infra/autostop_controller.py):
an EventBridge rule invokes this every minute. Same rule as before, on DynamoDB:

  * work waiting (queued OR running tasks) and the GPU is stopped -> start it;
  * nothing queued AND nothing running for >= IDLE_STOP_MINUTES -> stop it.

Idle is measured on QUEUE STATE, never CPU, so it cannot stop the box mid-generation
(a running task counts as work). The "idle since" clock cannot live in Lambda memory
(each invocation may be a cold start), so it is derived from the tasks table itself:
the box is stoppable once the MOST RECENT task update is older than the idle window
and nothing is currently queued/running. That is durable across invocations without
extra state.

ec2:Start/StopInstances is scoped by the Project=openminimax tag in this function's
role (infra/), so it can only ever touch this project's GPU box.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TASKS_TABLE = os.environ.get("TASKS_TABLE", "")
GPU_INSTANCE_ID = os.environ.get("GPU_INSTANCE_ID", "")
IDLE_STOP_MINUTES = float(os.environ.get("IDLE_STOP_MINUTES", "15"))


def _active_and_last_update(ddb, table: str) -> tuple[int, float]:
    """(count of queued+running tasks, newest updated_at across the table).

    queued+running come from the status GSI; newest updated_at bounds the idle timer.
    A brand-new/empty table reports (0, 0.0) -> treated as idle."""
    from app.tasks import DynamoDBTaskStore, QUEUED, RUNNING
    active = 0
    for status in (QUEUED, RUNNING):
        resp = ddb.query(
            TableName=table, IndexName=DynamoDBTaskStore.STATUS_INDEX,
            Select="COUNT",
            KeyConditionExpression="#s = :v",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":v": {"S": status}})
        active += resp.get("Count", 0)
    # Newest update across ALL tasks — a scan projecting one numeric attr; the table
    # is small (small deployment) and this runs once a minute.
    newest = 0.0
    kwargs = {"TableName": table, "ProjectionExpression": "updated_at"}
    while True:
        resp = ddb.scan(**kwargs)
        for it in resp.get("Items", []):
            newest = max(newest, float(it.get("updated_at", {}).get("N", "0")))
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return active, newest


def _gpu_state(ec2, instance_id: str) -> str:
    resp = ec2.describe_instances(InstanceIds=[instance_id])
    for r in resp["Reservations"]:
        for inst in r["Instances"]:
            return inst["State"]["Name"]
    return "unknown"


def handler(event=None, context=None, *, ddb=None, ec2=None, now=None) -> dict:
    """One evaluation. Clients are injectable for tests; `now` defaults to time.time
    (import-local so the module loads without it being patchable-away)."""
    if not TASKS_TABLE or not GPU_INSTANCE_ID:
        raise RuntimeError("TASKS_TABLE and GPU_INSTANCE_ID env vars are required")
    if ddb is None or ec2 is None:
        import boto3
        ddb = ddb or boto3.client("dynamodb")
        ec2 = ec2 or boto3.client("ec2")
    if now is None:
        import time
        now = time.time

    active, newest = _active_and_last_update(ddb, TASKS_TABLE)
    state = _gpu_state(ec2, GPU_INSTANCE_ID)
    action = "none"

    if active > 0:
        if state == "stopped":
            logger.info("autostop: %d active task(s), GPU stopped -> start", active)
            ec2.start_instances(InstanceIds=[GPU_INSTANCE_ID])
            action = "start"
    elif state == "running":
        idle_min = (now() - newest) / 60 if newest else float("inf")
        if idle_min >= IDLE_STOP_MINUTES:
            logger.info("autostop: idle %.1f min >= %.1f -> stop GPU",
                        idle_min, IDLE_STOP_MINUTES)
            ec2.stop_instances(InstanceIds=[GPU_INSTANCE_ID])
            action = "stop"

    return {"active": active, "gpu_state": state, "action": action}
