"""Self-issued Bearer key store: issue / verify / revoke / rate-limit / meter.

This is the gateway's security core. A caller is handed a key WE
minted (not a real MiniMax key), and every `Authorization: Bearer <key>` on the
MiniMax-compatible endpoints is checked here.

Design choices, and why:

- **We never store the raw key.** `issue()` returns the secret exactly once; the
  DB keeps only a salted SHA-256 hash. A leaked DB therefore cannot be replayed as
  the keys themselves. Lookup is by a short, non-secret `prefix` carried in the key
  string, so verification is one indexed row read plus a constant-time hash compare
  (no table scan, no timing oracle on the prefix).

- **Key shape fits the client's contract.** A MiniMax-compatible client validates a
  pasted key against roughly ``^\\S{20,4000}$`` (no whitespace, length 20-4000). Our
  format `mmh3_<prefix>_<secret>` is URL-safe base64 (no whitespace) and ~55 chars,
  well inside that window — see docs/API.md §0.

- **Rate limiting is per-key, fixed-window, in the same DB.** Small-scale traffic is
  bursty; a Redis/token-bucket is overkill. One row per key holds a window start
  and a counter, reset when the window rolls. The GPU is the real scarce resource,
  so the limit is on *submits*, checked by the caller at submit time.

- **Metering is a monotonic counter per key** (submits + a coarse "seconds billed"
  estimate the worker can add to). Enough to answer "who used how much" and to
  revoke an abuser; not a billing system.

Two backends, ONE interface (issue/verify/revoke/list_keys/check_and_count_submit/
add_seconds_billed):
  - `KeyStore`          — SQLite. Local dev / a single-box gateway / the test suite.
  - `DynamoDBKeyStore`  — the SAME store, in DynamoDB, so a separate process (the API
    Gateway Lambda authorizer, app.authorizer) can `verify()` a key without sharing
    the gateway's local disk. This is what makes "API Gateway validates the key in a
    Lambda" work: authorizer and gateway read one source of truth.

The key crypto (format, parse, hash) is module-level (`make_key`, `parse_key`,
`hash_secret`) so both backends and the authorizer produce/verify byte-identical
keys. Pure stdlib for the sqlite path; boto3 imported lazily only for the dynamo one.
Time is injected (`now`) so tests are deterministic.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from dataclasses import dataclass

_KEY_PREFIX = "mmh3"          # brands the key so a human can tell what it is
_PREFIX_BYTES = 6            # public lookup handle (indexed); not a secret
_SECRET_BYTES = 24          # the actual entropy the caller proves knowledge of
_HASH_ITERS = 100_000       # pbkdf2 rounds — the secret is high-entropy, so this is
                            # belt-and-suspenders, cheap at small scale


# --- key crypto: shared by every backend AND the Lambda authorizer -----------

def make_key() -> tuple[str, str, bytes, bytes]:
    """Mint a fresh key. Returns (full_key_string, prefix, secret_hash, salt).

    Prefix is HEX (no `_`/`-`) so the key parses unambiguously; the secret is
    urlsafe-base64 and may contain `_`/`-`, which `parse_key` handles via maxsplit.
    Only the hash+salt are ever persisted — the full string is shown to the caller
    exactly once."""
    prefix = secrets.token_hex(_PREFIX_BYTES)
    secret = secrets.token_urlsafe(_SECRET_BYTES)
    salt = secrets.token_bytes(16)
    return f"{_KEY_PREFIX}_{prefix}_{secret}", prefix, hash_secret(secret, salt), salt


def parse_key(bearer: str) -> tuple[str, str] | None:
    """`mmh3_<prefix>_<secret>` -> (prefix, secret). None if it isn't one of ours."""
    if not bearer:
        return None
    # maxsplit=2: the secret is the whole remainder, so a `_`/`-` inside the
    # urlsafe-base64 secret is preserved rather than truncating the key.
    parts = bearer.split("_", 2)
    if len(parts) != 3 or parts[0] != _KEY_PREFIX or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


def hash_secret(secret: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", secret.encode(), salt, _HASH_ITERS)


@dataclass(frozen=True)
class KeyInfo:
    """A resolved, currently-valid key. Never carries the secret."""
    prefix: str
    label: str
    rate_limit_per_min: int


class RevokedError(Exception):
    """Key exists but was revoked — distinct from 'never existed' so the caller can
    log the difference, though both surface to the client as a plain 401."""


class RateLimitedError(Exception):
    """Per-key submit budget for the current window is spent."""
    def __init__(self, retry_after_s: int):
        super().__init__("rate limited")
        self.retry_after_s = retry_after_s


class KeyStore:
    def __init__(self, db_path: str = ":memory:", *, now=time.time):
        # check_same_thread=False: uvicorn serves requests on a threadpool and this
        # store is a process-wide singleton. sqlite serializes writes itself; our
        # writes are tiny and rare (issue/revoke) vs. reads (every request).
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._now = now
        self._init_schema()

    def _init_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS keys (
                prefix              TEXT PRIMARY KEY,
                secret_hash         BLOB NOT NULL,
                salt                BLOB NOT NULL,
                label               TEXT NOT NULL DEFAULT '',
                rate_limit_per_min  INTEGER NOT NULL DEFAULT 6,
                revoked             INTEGER NOT NULL DEFAULT 0,
                created_at          REAL NOT NULL,
                -- fixed-window rate limiter state
                window_start        REAL NOT NULL DEFAULT 0,
                window_count        INTEGER NOT NULL DEFAULT 0,
                -- metering
                submits             INTEGER NOT NULL DEFAULT 0,
                seconds_billed      REAL NOT NULL DEFAULT 0
            );
            """
        )
        self._db.commit()

    # --- issuance -----------------------------------------------------------

    def issue(self, *, label: str = "", rate_limit_per_min: int = 6) -> str:
        """Mint a key. Returns the full secret string ONCE — it is never recoverable
        afterwards, only verifiable. Format: ``mmh3_<prefix>_<secret>``.

        Uses the module-level `make_key` so this store and the DynamoDB one (and the
        Lambda authorizer) produce byte-identical keys."""
        full, prefix, secret_hash, salt = make_key()
        self._db.execute(
            "INSERT INTO keys (prefix, secret_hash, salt, label, "
            "rate_limit_per_min, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (prefix, secret_hash, salt, label, int(rate_limit_per_min), self._now()),
        )
        self._db.commit()
        return full

    def revoke(self, prefix: str) -> bool:
        """Kill a key immediately. Returns False if no such key. Idempotent."""
        cur = self._db.execute(
            "UPDATE keys SET revoked = 1 WHERE prefix = ?", (prefix,))
        self._db.commit()
        return cur.rowcount > 0

    def list_keys(self) -> list[dict]:
        """Admin view: every key's prefix/label/limits/usage — never the secret."""
        rows = self._db.execute(
            "SELECT prefix, label, rate_limit_per_min, revoked, created_at, "
            "submits, seconds_billed FROM keys ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    # --- verification -------------------------------------------------------

    def verify(self, bearer: str) -> KeyInfo:
        """Resolve a Bearer value to a valid key, or raise.

        Raises ``KeyError`` for a malformed/unknown key and ``RevokedError`` for a
        known-but-revoked one; the endpoint maps both to HTTP 401 (the contract's
        auth-failure code), logging the distinction. Constant-time secret compare so
        a wrong secret cannot be probed byte-by-byte via timing."""
        parsed = parse_key(bearer)
        if parsed is None:
            raise KeyError("malformed key")
        prefix, secret = parsed
        row = self._db.execute(
            "SELECT * FROM keys WHERE prefix = ?", (prefix,)).fetchone()
        if row is None:
            raise KeyError("unknown key")
        expected = row["secret_hash"]
        if not hmac.compare_digest(expected, hash_secret(secret, row["salt"])):
            raise KeyError("bad secret")
        if row["revoked"]:
            raise RevokedError(prefix)
        return KeyInfo(prefix=prefix, label=row["label"],
                       rate_limit_per_min=row["rate_limit_per_min"])

    # --- rate limiting + metering ------------------------------------------

    def check_and_count_submit(self, prefix: str) -> None:
        """Charge one submit against the key's per-minute window, or raise
        ``RateLimitedError``. Also bumps the lifetime submit meter.

        Fixed 60s window: if the current window has rolled over, reset it; else if
        the count is at the limit, refuse with the seconds left in the window. The
        read-modify-write is a single connection so concurrent submits serialize on
        sqlite's write lock — correct for our scale without extra locking."""
        row = self._db.execute(
            "SELECT rate_limit_per_min, window_start, window_count "
            "FROM keys WHERE prefix = ?", (prefix,)).fetchone()
        if row is None:                       # verified moments ago; treat as gone
            raise KeyError("unknown key")
        now = self._now()
        limit = row["rate_limit_per_min"]
        start, count = row["window_start"], row["window_count"]
        if now - start >= 60:                 # window rolled (also the first ever)
            start, count = now, 0
        if limit > 0 and count >= limit:
            raise RateLimitedError(retry_after_s=max(1, int(60 - (now - start))))
        self._db.execute(
            "UPDATE keys SET window_start = ?, window_count = ?, submits = submits + 1 "
            "WHERE prefix = ?", (start, count + 1, prefix))
        self._db.commit()

    def add_seconds_billed(self, prefix: str, seconds: float) -> None:
        """Coarse usage meter the worker adds a finished clip's length to."""
        self._db.execute(
            "UPDATE keys SET seconds_billed = seconds_billed + ? WHERE prefix = ?",
            (float(seconds), prefix))
        self._db.commit()


class DynamoDBKeyStore:
    """The SAME key store, backed by DynamoDB instead of local SQLite.

    Why it exists: when the gateway sits behind API Gateway, the thing that checks a
    caller's key is a Lambda AUTHORIZER (app.authorizer) running in its own
    process — it cannot read the gateway box's SQLite file. Both the authorizer and
    whoever issues/revokes keys must read/write ONE shared source of truth, and a
    DynamoDB table is the cheap, serverless, always-consistent way to do that.

    Interface-compatible with `KeyStore` (issue/verify/revoke/list_keys/
    check_and_count_submit/add_seconds_billed) so every caller and test is agnostic to
    which backend it holds. Byte-identical keys because both go through `make_key` /
    `parse_key` / `hash_secret`.

    Table (created in infra/, PK = `prefix`):
        prefix (S, PK), secret_hash (B), salt (B), label (S), rate_limit_per_min (N),
        revoked (BOOL), created_at (N), window_start (N), window_count (N),
        submits (N), seconds_billed (N)

    Rate limiting is a per-item atomic UpdateItem (see check_and_count_submit); no
    lock, no scan — DynamoDB serializes the conditional update on the single item.
    boto3 is imported lazily so importing this module never requires AWS.
    """

    def __init__(self, table_name: str, *, now=time.time, client=None):
        if client is not None:
            self._ddb = client
        else:
            import boto3
            self._ddb = boto3.client("dynamodb")
        self._table = table_name
        self._now = now

    def issue(self, *, label: str = "", rate_limit_per_min: int = 6) -> str:
        full, prefix, secret_hash, salt = make_key()
        self._ddb.put_item(
            TableName=self._table,
            Item={
                "prefix": {"S": prefix},
                "secret_hash": {"B": secret_hash},
                "salt": {"B": salt},
                "label": {"S": label},
                "rate_limit_per_min": {"N": str(int(rate_limit_per_min))},
                "revoked": {"BOOL": False},
                "created_at": {"N": repr(self._now())},
                "window_start": {"N": "0"},
                "window_count": {"N": "0"},
                "submits": {"N": "0"},
                "seconds_billed": {"N": "0"},
            },
            # Never silently clobber an existing key on a prefix collision.
            ConditionExpression="attribute_not_exists(prefix)",
        )
        return full

    def revoke(self, prefix: str) -> bool:
        try:
            self._ddb.update_item(
                TableName=self._table,
                Key={"prefix": {"S": prefix}},
                UpdateExpression="SET revoked = :t",
                ExpressionAttributeValues={":t": {"BOOL": True}},
                ConditionExpression="attribute_exists(prefix)",
            )
            return True
        except self._ddb.exceptions.ConditionalCheckFailedException:
            return False

    def list_keys(self) -> list[dict]:
        """Admin view — never the secret. A Scan is fine: this table is small (one
        row per client) and the admin list is a rare, non-hot-path call."""
        out: list[dict] = []
        kwargs = {"TableName": self._table,
                  "ProjectionExpression": "prefix, label, rate_limit_per_min, "
                  "revoked, created_at, submits, seconds_billed"}
        while True:
            resp = self._ddb.scan(**kwargs)
            for it in resp.get("Items", []):
                out.append({
                    "prefix": it["prefix"]["S"],
                    "label": it.get("label", {}).get("S", ""),
                    "rate_limit_per_min": int(it["rate_limit_per_min"]["N"]),
                    "revoked": 1 if it.get("revoked", {}).get("BOOL") else 0,
                    "created_at": float(it["created_at"]["N"]),
                    "submits": int(it.get("submits", {}).get("N", "0")),
                    "seconds_billed": float(it.get("seconds_billed", {}).get("N", "0")),
                })
            if "LastEvaluatedKey" not in resp:
                return out
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    def verify(self, bearer: str) -> KeyInfo:
        """Same contract as KeyStore.verify — KeyError (malformed/unknown/bad secret),
        RevokedError (known but revoked). Constant-time hash compare."""
        parsed = parse_key(bearer)
        if parsed is None:
            raise KeyError("malformed key")
        prefix, secret = parsed
        resp = self._ddb.get_item(
            TableName=self._table, Key={"prefix": {"S": prefix}}, ConsistentRead=False)
        item = resp.get("Item")
        if not item:
            raise KeyError("unknown key")
        expected = item["secret_hash"]["B"]
        if not hmac.compare_digest(expected, hash_secret(secret, item["salt"]["B"])):
            raise KeyError("bad secret")
        if item.get("revoked", {}).get("BOOL"):
            raise RevokedError(prefix)
        return KeyInfo(prefix=prefix, label=item.get("label", {}).get("S", ""),
                       rate_limit_per_min=int(item["rate_limit_per_min"]["N"]))

    def check_and_count_submit(self, prefix: str) -> None:
        """Charge one submit against the per-minute window atomically, or raise
        RateLimitedError. Same fixed-window semantics as the SQLite store.

        Done as a single conditional UpdateItem so concurrent submits (many Lambdas)
        cannot over-count: the update only applies when EITHER the window has rolled
        (now - window_start >= 60) OR the count is still under the limit. A failed
        condition means the budget is spent this window. A separate branch resets the
        window; because both branches are conditional single-item writes, DynamoDB
        serializes them on the item."""
        now = self._now()
        # Fast path: window still open and under limit -> increment.
        try:
            self._ddb.update_item(
                TableName=self._table,
                Key={"prefix": {"S": prefix}},
                UpdateExpression="SET window_count = window_count + :one, "
                                 "submits = submits + :one",
                ConditionExpression="attribute_exists(prefix) AND "
                                    "window_start > :winfloor AND "
                                    "window_count < rate_limit_per_min",
                ExpressionAttributeValues={
                    ":one": {"N": "1"},
                    ":winfloor": {"N": repr(now - 60)},
                },
            )
            return
        except self._ddb.exceptions.ConditionalCheckFailedException:
            pass
        # Slow path: either the window rolled (>=60s old) or the item was fresh —
        # start a new window at count 1. Guard on the window still being old so we
        # don't stomp a window another writer just opened.
        try:
            self._ddb.update_item(
                TableName=self._table,
                Key={"prefix": {"S": prefix}},
                UpdateExpression="SET window_start = :now, window_count = :one, "
                                 "submits = submits + :one",
                ConditionExpression="attribute_exists(prefix) AND "
                                    "window_start <= :winfloor",
                ExpressionAttributeValues={
                    ":now": {"N": repr(now)},
                    ":one": {"N": "1"},
                    ":winfloor": {"N": repr(now - 60)},
                },
            )
            return
        except self._ddb.exceptions.ConditionalCheckFailedException:
            # Neither path applied: the key exists, the window is current, and the
            # count is at the limit -> genuinely rate limited. (If the key were gone,
            # verify() would already have rejected it upstream.)
            resp = self._ddb.get_item(TableName=self._table,
                                      Key={"prefix": {"S": prefix}})
            item = resp.get("Item")
            if not item:
                raise KeyError("unknown key")
            start = float(item.get("window_start", {}).get("N", "0"))
            raise RateLimitedError(retry_after_s=max(1, int(60 - (now - start))))

    def add_seconds_billed(self, prefix: str, seconds: float) -> None:
        self._ddb.update_item(
            TableName=self._table,
            Key={"prefix": {"S": prefix}},
            UpdateExpression="SET seconds_billed = seconds_billed + :s",
            ExpressionAttributeValues={":s": {"N": repr(float(seconds))}},
        )
