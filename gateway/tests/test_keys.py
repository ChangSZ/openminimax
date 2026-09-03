"""KeyStore: issue/verify/revoke/rate-limit/meter — the gateway's security core."""

import pytest

from app.keys import KeyStore, RateLimitedError, RevokedError


def test_issued_key_verifies_and_matches_client_shape():
    store = KeyStore()
    key = store.issue(label="client-1")
    # Client validates ~^\S{20,4000}$ — no whitespace, plausible length (CONTRACT §0).
    assert " " not in key and "\t" not in key
    assert 20 <= len(key) <= 4000
    info = store.verify(key)
    assert info.label == "client-1"


def test_raw_key_is_never_stored():
    store = KeyStore()
    key = store.issue()
    secret = key.split("_")[-1]
    # The secret must not be recoverable from the DB in any column.
    for row in store._db.execute("SELECT * FROM keys").fetchall():
        assert secret not in str(tuple(row))


def test_a_wrong_secret_is_rejected():
    store = KeyStore()
    key = store.issue()
    prefix = key.split("_")[1]
    with pytest.raises(KeyError):
        store.verify(f"mmh3_{prefix}_wrongsecretwrongsecret")


def test_unknown_and_malformed_keys_raise_keyerror():
    store = KeyStore()
    for bad in ["", "garbage", "mmh3_only", "notours_a_b", "mmh3__b", "mmh3_a_"]:
        with pytest.raises(KeyError):
            store.verify(bad)


def test_revoke_is_distinguishable_and_idempotent():
    store = KeyStore()
    key = store.issue()
    prefix = key.split("_")[1]
    assert store.revoke(prefix) is True
    with pytest.raises(RevokedError):
        store.verify(key)
    assert store.revoke(prefix) is True   # idempotent
    assert store.revoke("nosuchprefix") is False


def test_rate_limit_is_per_key_and_fixed_window():
    clock = {"t": 1000.0}
    store = KeyStore(now=lambda: clock["t"])
    key = store.issue(rate_limit_per_min=3)
    prefix = key.split("_")[1]

    for _ in range(3):
        store.check_and_count_submit(prefix)      # 3 allowed
    with pytest.raises(RateLimitedError) as exc:
        store.check_and_count_submit(prefix)      # 4th refused
    assert 1 <= exc.value.retry_after_s <= 60

    clock["t"] += 61                              # window rolls
    store.check_and_count_submit(prefix)          # allowed again


def test_rate_limit_isolates_keys():
    store = KeyStore()
    a = store.issue(rate_limit_per_min=1).split("_")[1]
    b = store.issue(rate_limit_per_min=1).split("_")[1]
    store.check_and_count_submit(a)
    with pytest.raises(RateLimitedError):
        store.check_and_count_submit(a)
    store.check_and_count_submit(b)   # b unaffected by a's spent budget


def test_metering_counts_submits_and_seconds():
    store = KeyStore()
    key = store.issue()
    prefix = key.split("_")[1]
    store.check_and_count_submit(prefix)
    store.check_and_count_submit(prefix)
    store.add_seconds_billed(prefix, 6)
    store.add_seconds_billed(prefix, 4)
    row = store.list_keys()[0]
    assert row["submits"] == 2
    assert row["seconds_billed"] == 10
    assert "secret" not in row and "secret_hash" not in row   # admin view is safe
