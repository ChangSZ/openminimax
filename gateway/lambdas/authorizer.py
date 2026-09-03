"""API Gateway (HTTP API) Lambda authorizer — validates our self-signed Bearer key.

This is the "API Gateway 要进行 lambda 验证 key" piece. Every request to the public
HTTP API hits this authorizer FIRST; only if it returns `isAuthorized: true` does the
request reach the submit/poll integration. So this is the single choke point where a
caller's key is checked, and it is the ONLY thing exposed to the internet-facing
API besides the routes themselves.

Format: HTTP API REQUEST authorizer, **payload format 2.0**, **simple responses**
(`enableSimpleResponses: true`). We return `{"isAuthorized": bool, "context": {...}}`.
The `context` is forwarded to the integration Lambda as
`event.requestContext.authorizer.lambda.*`, so submit/poll get the key's prefix and
rate limit WITHOUT re-verifying (and without ever seeing the secret again).

Caching (set on the authorizer in infra/): identity source is the `Authorization`
header, TTL ~300s. Clients poll every ~5s for minutes, so without
caching every poll would verify against DynamoDB; with it, a given key is verified
about once per 5 min. This is the main cost lever on the auth path. Because the cache
key is the whole Authorization header, a revoked key stays usable until its cache
entry expires (<=TTL) — acceptable for a small deployment; lower the TTL if you need faster
revocation.

A malformed/absent Authorization header never even reaches here (API Gateway 401s on
a missing identity source), but we still handle the empty case defensively.

Reuses `app.keys.DynamoDBKeyStore` so the authorizer and key issuance share one
source of truth and one hash. The store (and its boto3 client) is created at module
load and reused across warm invocations.
"""

from __future__ import annotations

import logging
import os

from app.keys import DynamoDBKeyStore, RevokedError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

KEYS_TABLE = os.environ.get("KEYS_TABLE", "")
_STORE: DynamoDBKeyStore | None = None


def _store() -> DynamoDBKeyStore:
    """Lazily build (and warm-cache) the key store. Not built at import so a unit test
    can inject its own via `handler(event, store=...)`."""
    global _STORE
    if _STORE is None:
        if not KEYS_TABLE:
            raise RuntimeError("KEYS_TABLE env var is required")
        _STORE = DynamoDBKeyStore(KEYS_TABLE)
    return _STORE


def _bearer(event: dict) -> str:
    """Pull the token out of the Authorization header (case-insensitive header name
    and scheme). HTTP API lowercases header keys in v2 events."""
    headers = event.get("headers") or {}
    raw = headers.get("authorization") or headers.get("Authorization") or ""
    parts = raw.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return raw.strip()


def handler(event: dict, context=None, *, store: DynamoDBKeyStore | None = None) -> dict:
    """Simple-response authorizer. Returns {isAuthorized, context}.

    `store` is injectable for tests; production uses the module-cached DynamoDB store.
    Any verification failure -> not authorized. A valid key -> authorized, with the
    prefix and rate limit passed through in `context` for the integration Lambda."""
    ks = store or _store()
    token = _bearer(event)
    try:
        info = ks.verify(token)
    except RevokedError:
        logger.info("authorizer: revoked key")
        return {"isAuthorized": False}
    except KeyError:
        logger.info("authorizer: invalid key")
        return {"isAuthorized": False}
    except Exception:                       # never leak a stack trace as a 500->allow
        logger.exception("authorizer: unexpected error verifying key")
        return {"isAuthorized": False}

    # Context values must be strings for API Gateway to forward them.
    return {
        "isAuthorized": True,
        "context": {
            "keyPrefix": info.prefix,
            "rateLimitPerMin": str(info.rate_limit_per_min),
            "label": info.label,
        },
    }
