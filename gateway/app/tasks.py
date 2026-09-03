"""Async task store for video generation.

The MiniMax-compatible client's contract is submit-then-poll (docs/API.md
§1-2): `POST /v2/video_generation` must return a `task_id` *fast* (within the
client's ~15s timeout) without waiting for the clip, and `GET /v2/query/...` reads
the task's status until a URL appears. H3 on 4×L40S takes minutes per clip, so the
submit CANNOT block — it enqueues and returns.

This module is the queue + state, persisted in SQLite so an in-flight task survives
a gateway restart (a worker process re-reads `running` rows and reconciles). The
actual generation runs in a worker (app.worker), which is the only thing that talks
to the GPU; this store is pure state and has no GPU/network dependency, so it's
fully testable now.

Status vocabulary is the CLIENT's, verbatim, because it is copied straight into the
poll response: ``queued | running | succeeded | failed | cancelled``. We only ever
emit queued/running/succeeded/failed (nothing cancels a task here), but the client
maps `cancelled` too, so it stays in the enum for faithfulness.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from dataclasses import dataclass

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"

_TERMINAL = (SUCCEEDED, FAILED)

# A task's media kind. Two workers share ONE queue but each drains only its own
# kind (video worker -> H3 on cuda:0; image worker -> FLUX on cuda:2), so a claim
# is filtered by kind (see `claim_next`). `video` is the default everywhere so a
# task/row written before this field existed reads as a video task, unchanged.
VIDEO = "video"
IMAGE = "image"


@dataclass
class Task:
    task_id: str
    key_prefix: str          # who owns it (for metering / isolation)
    status: str
    request: dict            # the translated generation request (prompt, refs, ...)
    url: str = ""            # set on SUCCEEDED — a downloadable (presigned) media URL
    error: str = ""          # set on FAILED — a human-readable reason
    duration_s: int = 0      # requested clip length, for metering (video only)
    created_at: float = 0.0
    updated_at: float = 0.0
    kind: str = VIDEO        # `video` | `image` — which worker/backend runs it


class TaskStore:
    def __init__(self, db_path: str = ":memory:", *, now=time.time):
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._now = now
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id     TEXT PRIMARY KEY,
                key_prefix  TEXT NOT NULL,
                status      TEXT NOT NULL,
                request     TEXT NOT NULL,
                url         TEXT NOT NULL DEFAULT '',
                error       TEXT NOT NULL DEFAULT '',
                duration_s  INTEGER NOT NULL DEFAULT 0,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL,
                kind        TEXT NOT NULL DEFAULT 'video'
            );
            CREATE INDEX IF NOT EXISTS tasks_status ON tasks(status);
            """
        )
        # `kind` was added after the first schema; a DB created before it has the
        # table but not the column. Add it idempotently so an existing dev/CI db
        # keeps working (a fresh db already has it from CREATE above).
        cols = {r["name"] for r in self._db.execute("PRAGMA table_info(tasks)")}
        if "kind" not in cols:
            self._db.execute(
                "ALTER TABLE tasks ADD COLUMN kind TEXT NOT NULL DEFAULT 'video'")
        self._db.commit()

    def enqueue(self, *, key_prefix: str, request: dict, duration_s: int,
                kind: str = VIDEO) -> str:
        """Create a queued task and return its id. The id is opaque to the client;
        it round-trips it back on every poll (URL-quoted), so any URL-safe token is
        fine. Prefixed for greppability in logs (`vt_`=video, `it_`=image)."""
        prefix = "it_" if kind == IMAGE else "vt_"
        task_id = prefix + secrets.token_urlsafe(16)
        now = self._now()
        self._db.execute(
            "INSERT INTO tasks (task_id, key_prefix, status, request, duration_s, "
            "created_at, updated_at, kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, key_prefix, QUEUED, json.dumps(request), int(duration_s),
             now, now, kind))
        self._db.commit()
        return task_id

    def get(self, task_id: str) -> Task | None:
        row = self._db.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return self._row_to_task(row) if row else None

    def claim_next(self, kind: str = VIDEO) -> Task | None:
        """Atomically take the oldest queued task OF THIS KIND and mark it running.

        The UPDATE...WHERE task_id IN (SELECT ... LIMIT 1) is a single statement, so
        two workers cannot claim the same row: sqlite serializes writes and the
        second finds nothing left queued for that id. Filtering by `kind` is what
        lets the video worker (cuda:0) and image worker (cuda:2) share one queue
        without stealing each other's tasks. Returns the claimed task, or None if no
        task of this kind is queued."""
        now = self._now()
        cur = self._db.execute(
            "UPDATE tasks SET status = ?, updated_at = ? "
            "WHERE task_id IN (SELECT task_id FROM tasks WHERE status = ? AND kind = ? "
            "ORDER BY created_at LIMIT 1) RETURNING task_id",
            (RUNNING, now, QUEUED, kind))
        row = cur.fetchone()
        self._db.commit()
        return self.get(row["task_id"]) if row else None

    def mark_succeeded(self, task_id: str, url: str, *, kind: str = VIDEO) -> None:
        # `kind` is accepted for signature-parity with DynamoDBTaskStore (which needs
        # it to maintain kind_status); sqlite keys off the row's own column, so it is
        # unused here.
        self._set(task_id, status=SUCCEEDED, url=url)

    def mark_failed(self, task_id: str, error: str, *, kind: str = VIDEO) -> None:
        self._set(task_id, status=FAILED, error=error)

    def _set(self, task_id: str, **fields) -> None:
        fields["updated_at"] = self._now()
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._db.execute(f"UPDATE tasks SET {cols} WHERE task_id = ?",
                         (*fields.values(), task_id))
        self._db.commit()

    def requeue_stale_running(self, older_than_s: float) -> int:
        """On worker startup, put orphaned `running` rows (a crash mid-generation)
        back to `queued`. Returns how many. Idempotent and safe to call on a loop."""
        cutoff = self._now() - older_than_s
        cur = self._db.execute(
            "UPDATE tasks SET status = ?, updated_at = ? "
            "WHERE status = ? AND updated_at < ?",
            (QUEUED, self._now(), RUNNING, cutoff))
        self._db.commit()
        return cur.rowcount

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> Task:
        keys = row.keys()
        return Task(
            task_id=row["task_id"], key_prefix=row["key_prefix"],
            status=row["status"], request=json.loads(row["request"]),
            url=row["url"], error=row["error"], duration_s=row["duration_s"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            kind=row["kind"] if "kind" in keys else VIDEO)


class DynamoDBTaskStore:
    """The SAME task queue, in DynamoDB — for when submit/poll run in Lambda and the
    worker runs on the GPU box: three processes sharing one queue that no local disk
    can hold.

    Interface-compatible with `TaskStore` (enqueue/get/claim_next/mark_succeeded/
    mark_failed/requeue_stale_running). The `request` dict is stored as a JSON string
    (one attribute) to keep the item shape flat and avoid DynamoDB's map-typing of an
    arbitrary payload.

    Table (created in infra/, PK = `task_id`) with two GSIs:
      * `kind_status-created_at-index` (PK = `kind_status` = "<kind>#<status>",
        SK = created_at) — `claim_next(kind)` queries the oldest `"<kind>#queued"`
        item, so the video worker and image worker drain ONE queue without stealing
        each other's tasks.
      * `status-created_at-index` (PK = status, SK = created_at) — kept for
        `requeue_stale_running`, which reclaims stale RUNNING items of ANY kind.
    Single-flight claiming is a conditional UpdateItem: the QUEUED -> RUNNING
    transition only applies if the item is still QUEUED, so of two workers racing for
    the same id only one wins.

    `kind_status` is a synthetic attribute maintained on every write (enqueue/_set/
    requeue) — an item written before this field existed simply won't appear in the
    kind GSI, which is fine because it also predates image tasks and is a video task
    the old index still serves. boto3 lazily imported; client injectable for tests.
    """

    STATUS_INDEX = "status-created_at-index"
    KIND_STATUS_INDEX = "kind_status-created_at-index"

    def __init__(self, table_name: str, *, now=time.time, client=None):
        if client is not None:
            self._ddb = client
        else:
            import boto3
            self._ddb = boto3.client("dynamodb")
        self._table = table_name
        self._now = now

    def enqueue(self, *, key_prefix: str, request: dict, duration_s: int,
                kind: str = VIDEO) -> str:
        task_id = ("it_" if kind == IMAGE else "vt_") + secrets.token_urlsafe(16)
        now = self._now()
        self._ddb.put_item(
            TableName=self._table,
            Item={
                "task_id": {"S": task_id},
                "key_prefix": {"S": key_prefix},
                "status": {"S": QUEUED},
                "kind": {"S": kind},
                "kind_status": {"S": f"{kind}#{QUEUED}"},   # kind GSI partition key
                "request": {"S": json.dumps(request)},
                "url": {"S": ""},
                "error": {"S": ""},
                "duration_s": {"N": str(int(duration_s))},
                "created_at": {"N": repr(now)},
                "updated_at": {"N": repr(now)},
            },
        )
        return task_id

    def get(self, task_id: str) -> Task | None:
        resp = self._ddb.get_item(TableName=self._table,
                                  Key={"task_id": {"S": task_id}})
        item = resp.get("Item")
        return self._item_to_task(item) if item else None

    def claim_next(self, kind: str = VIDEO) -> Task | None:
        """Take the oldest queued task OF THIS KIND and flip it to running, atomically.

        Query the kind GSI for the oldest `"<kind>#queued"` item, then conditionally
        update it to RUNNING only while it is still QUEUED — and also flip its
        `kind_status` to `"<kind>#running"` so it leaves the queued partition. If
        another worker claimed it between the query and the update, the condition
        fails and we try the next candidate; an empty query means no task of this
        kind is queued. Filtering by kind is what lets the video worker (cuda:0) and
        image worker (cuda:2) share one table without stealing each other's tasks."""
        resp = self._ddb.query(
            TableName=self._table,
            IndexName=self.KIND_STATUS_INDEX,
            KeyConditionExpression="kind_status = :ks",
            ExpressionAttributeValues={":ks": {"S": f"{kind}#{QUEUED}"}},
            ScanIndexForward=True,          # oldest created_at first
            Limit=10,
        )
        for cand in resp.get("Items", []):
            task_id = cand["task_id"]["S"]
            now = self._now()
            try:
                updated = self._ddb.update_item(
                    TableName=self._table,
                    Key={"task_id": {"S": task_id}},
                    UpdateExpression=("SET #s = :r, kind_status = :ksr, "
                                      "updated_at = :now"),
                    ConditionExpression="#s = :q",
                    ExpressionAttributeNames={"#s": "status"},
                    ExpressionAttributeValues={
                        ":r": {"S": RUNNING}, ":q": {"S": QUEUED},
                        ":ksr": {"S": f"{kind}#{RUNNING}"},
                        ":now": {"N": repr(now)}},
                    ReturnValues="ALL_NEW",
                )
                return self._item_to_task(updated["Attributes"])
            except self._ddb.exceptions.ConditionalCheckFailedException:
                continue                    # someone else got it; try the next
        return None

    def mark_succeeded(self, task_id: str, url: str, *, kind: str = VIDEO) -> None:
        self._set(task_id, SUCCEEDED, url=url, kind=kind)

    def mark_failed(self, task_id: str, error: str, *, kind: str = VIDEO) -> None:
        self._set(task_id, FAILED, error=error, kind=kind)

    def _set(self, task_id: str, status: str, *, url: str = "", error: str = "",
             kind: str = VIDEO) -> None:
        # Keep `kind_status` in step with `status` so a terminal item leaves the
        # queued/running partitions of the kind GSI. Caller passes the task's kind
        # (the worker has the Task in hand); default `video` for legacy callers.
        self._ddb.update_item(
            TableName=self._table,
            Key={"task_id": {"S": task_id}},
            UpdateExpression=("SET #s = :s, kind_status = :ks, #u = :u, "
                              "updated_at = :now, #e = :e"),
            ExpressionAttributeNames={"#s": "status", "#u": "url", "#e": "error"},
            ExpressionAttributeValues={
                ":s": {"S": status}, ":ks": {"S": f"{kind}#{status}"},
                ":u": {"S": url}, ":e": {"S": error},
                ":now": {"N": repr(self._now())}},
        )

    def requeue_stale_running(self, older_than_s: float) -> int:
        """Return orphaned `running` items (crash mid-generation) to `queued`.

        Scans the status GSI for RUNNING items older than the cutoff and flips each
        back conditionally (still RUNNING and still stale), so a task that legitimately
        just started is never yanked. RUNNING items are few (~one at a time on a
        single GPU), so this is cheap."""
        cutoff = self._now() - older_than_s
        resp = self._ddb.query(
            TableName=self._table,
            IndexName=self.STATUS_INDEX,
            KeyConditionExpression="#s = :r",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":r": {"S": RUNNING}},
        )
        requeued = 0
        for item in resp.get("Items", []):
            if float(item["updated_at"]["N"]) >= cutoff:
                continue
            task_id = item["task_id"]["S"]
            kind = item.get("kind", {}).get("S", VIDEO)   # legacy rows -> video
            try:
                self._ddb.update_item(
                    TableName=self._table,
                    Key={"task_id": {"S": task_id}},
                    UpdateExpression="SET #s = :q, kind_status = :ks, updated_at = :now",
                    ConditionExpression="#s = :r AND updated_at < :cutoff",
                    ExpressionAttributeNames={"#s": "status"},
                    ExpressionAttributeValues={
                        ":q": {"S": QUEUED}, ":r": {"S": RUNNING},
                        ":ks": {"S": f"{kind}#{QUEUED}"},
                        ":now": {"N": repr(self._now())},
                        ":cutoff": {"N": repr(cutoff)}},
                )
                requeued += 1
            except self._ddb.exceptions.ConditionalCheckFailedException:
                continue
        return requeued

    @staticmethod
    def _item_to_task(item: dict) -> Task:
        return Task(
            task_id=item["task_id"]["S"],
            key_prefix=item["key_prefix"]["S"],
            status=item["status"]["S"],
            request=json.loads(item.get("request", {}).get("S", "{}")),
            url=item.get("url", {}).get("S", ""),
            error=item.get("error", {}).get("S", ""),
            duration_s=int(item.get("duration_s", {}).get("N", "0")),
            created_at=float(item.get("created_at", {}).get("N", "0")),
            updated_at=float(item.get("updated_at", {}).get("N", "0")),
            kind=item.get("kind", {}).get("S", VIDEO))   # legacy rows -> video
