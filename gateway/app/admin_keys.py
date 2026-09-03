"""Operator CLI: issue / list / revoke client keys in the DynamoDB key table.

In the serverless architecture there is deliberately NO public admin HTTP route —
key issuance is an operator action, gated by AWS IAM (you need
dynamodb:PutItem/UpdateItem on the keys table), not by an in-app admin token that
could leak. Run it with the operator's AWS credentials:

    KEYS_TABLE=openminimax-serverless-keys python -m app.admin_keys issue --label team-3 --rate 6
    KEYS_TABLE=openminimax-serverless-keys python -m app.admin_keys list
    KEYS_TABLE=openminimax-serverless-keys python -m app.admin_keys revoke <prefix>

`issue` prints the full key ONCE (it is never recoverable) — hand it to the
caller to paste into the client's MiniMax key field. Reuses
`app.keys.DynamoDBKeyStore`, so issued keys are byte-identical to what the authorizer
verifies."""

from __future__ import annotations

import argparse
import os

from app.keys import DynamoDBKeyStore


def _store() -> DynamoDBKeyStore:
    table = os.environ.get("KEYS_TABLE", "")
    if not table:
        raise SystemExit("KEYS_TABLE env var is required")
    return DynamoDBKeyStore(table)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="admin_keys",
                                 description="Manage openminimax client keys.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_issue = sub.add_parser("issue", help="mint a new key (printed once)")
    p_issue.add_argument("--label", default="", help="who it's for, e.g. team-3")
    p_issue.add_argument("--rate", type=int, default=6,
                         help="submits allowed per minute (default 6)")

    sub.add_parser("list", help="list keys (never shows secrets)")

    p_revoke = sub.add_parser("revoke", help="revoke a key by its prefix")
    p_revoke.add_argument("prefix")

    args = ap.parse_args(argv)
    store = _store()

    if args.cmd == "issue":
        key = store.issue(label=args.label, rate_limit_per_min=args.rate)
        print(key)
        print("^ give this to the caller now — it is not recoverable.",
              flush=True)
    elif args.cmd == "list":
        rows = store.list_keys()
        if not rows:
            print("(no keys)")
        for r in rows:
            state = "REVOKED" if r["revoked"] else "active"
            print(f'{r["prefix"]}  {state:7}  rate={r["rate_limit_per_min"]}/min  '
                  f'submits={r["submits"]}  billed={r["seconds_billed"]:.0f}s  '
                  f'{r["label"]}')
    elif args.cmd == "revoke":
        print("revoked" if store.revoke(args.prefix) else "no such key")


if __name__ == "__main__":
    main()
