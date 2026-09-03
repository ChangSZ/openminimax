#!/usr/bin/env python3
"""Scale-to-zero controller for the GPU box (docs/PLAN.md §2, the real cost lever).

The GPU bills ~$10/hr whether or not it's generating, so it must be OFF whenever the
queue is empty. This runs on the always-on t3.small gateway box and:

  * STARTS the GPU instance when there is queued work and it is stopped;
  * STOPS the GPU instance after it has been idle (no queued AND no running tasks)
    for `IDLE_STOP_MINUTES`.

Why idle is measured on QUEUE STATE, not CPU: a CPU-utilization alarm would happily
stop the box in the middle of a multi-minute generation (H3 is not pegged at 100%
throughout). We instead read the gateway's own task DB — a stop only happens when
there is genuinely nothing queued and nothing running, so an in-flight clip is never
killed. The video path is async: clients poll and tolerate ~25 min, which
comfortably covers the GPU cold start.

ec2:StartInstances/StopInstances are scoped by the `Project=openminimax` tag in the
instance role (infra/template.yaml), so this controller can only ever touch this
project's own box.

Runs as a simple loop (systemd unit: infra/openminimax-autostop.service). No external
deps beyond boto3 + the gateway's TaskStore, both already present on the box.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import time

import boto3

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s autostop %(levelname)s %(message)s")
log = logging.getLogger("autostop")

REGION = os.environ.get("AWS_REGION", "us-west-2")
GPU_INSTANCE_ID = os.environ.get("GPU_INSTANCE_ID", "")
GATEWAY_DB = os.environ.get("GATEWAY_DB", "gateway.db")
IDLE_STOP_MINUTES = float(os.environ.get("IDLE_STOP_MINUTES", "15"))
POLL_SECONDS = float(os.environ.get("AUTOSTOP_POLL_SECONDS", "30"))


def pending_work(db_path: str) -> int:
    """Count tasks that need the GPU up: queued OR running. A missing/locked DB
    returns 0 (treat as idle) rather than crashing the controller — a spurious
    'idle' just delays a start by one poll, which the async path tolerates."""
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
    except sqlite3.Error:
        return 0
    try:
        row = con.execute(
            "SELECT COUNT(*) FROM tasks WHERE status IN ('queued', 'running')"
        ).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0
    finally:
        con.close()


def instance_state(ec2, instance_id: str) -> str:
    resp = ec2.describe_instances(InstanceIds=[instance_id])
    for res in resp["Reservations"]:
        for inst in res["Instances"]:
            return inst["State"]["Name"]
    return "unknown"


def control_loop(ec2, instance_id: str, db_path: str) -> None:
    idle_since: float | None = None
    while True:
        work = pending_work(db_path)
        state = instance_state(ec2, instance_id)

        if work > 0:
            idle_since = None
            if state == "stopped":
                log.info("work=%d, GPU is %s -> starting", work, state)
                ec2.start_instances(InstanceIds=[instance_id])
            # running/pending: nothing to do, the gateway will reach it once up.
        else:
            # No queued and no running tasks. Start the idle clock; stop only after
            # it has been continuously idle for the threshold — never mid-generation
            # (running tasks count as work above).
            if state == "running":
                now = time.monotonic()
                idle_since = idle_since or now
                idle_min = (now - idle_since) / 60
                if idle_min >= IDLE_STOP_MINUTES:
                    log.info("idle %.1f min >= %.1f -> stopping GPU",
                             idle_min, IDLE_STOP_MINUTES)
                    ec2.stop_instances(InstanceIds=[instance_id])
                    idle_since = None
            else:
                idle_since = None

        time.sleep(POLL_SECONDS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-id", default=GPU_INSTANCE_ID)
    ap.add_argument("--db", default=GATEWAY_DB)
    ap.add_argument("--once", action="store_true",
                    help="evaluate one cycle and exit (for testing)")
    args = ap.parse_args()
    if not args.instance_id:
        raise SystemExit("GPU_INSTANCE_ID (or --instance-id) is required")

    ec2 = boto3.client("ec2", region_name=REGION)
    log.info("autostop: instance=%s db=%s idle_stop=%.0fmin poll=%.0fs",
             args.instance_id, args.db, IDLE_STOP_MINUTES, POLL_SECONDS)
    if args.once:
        work = pending_work(args.db)
        log.info("one-shot: pending_work=%d state=%s", work,
                 instance_state(ec2, args.instance_id))
        return
    control_loop(ec2, args.instance_id, args.db)


if __name__ == "__main__":
    main()
