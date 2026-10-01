"""Store-level tests: lifecycle, pragmas, durability, keying, errors.

Every test runs against a real on-disk database under ``tmp_path`` — the
durability contract (WAL, permissions, backup) cannot be verified on
``:memory:`` databases.
"""

from __future__ import annotations

import os
import sqlite3
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.storage.conftest import make_envelope, make_scope

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.storage import migrations
from verbatim.storage.repos import SourcesRepo, ensure_scope
from verbatim.storage.schema import SCHEMA_VERSION
from verbatim.storage.store import Store


# ----------------------------------------------------------------------
# create / open
# ----------------------------------------------------------------------


def test_create_open_roundtrip(store_path: str) -> None:
    store = Store.create(store_path)
    db_id = store.db_id()
    assert db_id
    assert store.schema_version == SCHEMA_VERSION
    assert store.projection_generation() == 1
    assert store.policy_epoch() == 0
    store.close()

    reopened = Store.open(store_path)
    assert reopened.db_id() == db_id
    assert reopened.schema_version == SCHEMA_VERSION
    reopened.close()


def test_create_initializes_meta_and_history(store: Store) -> None:
    with store.read() as conn:
        meta = {
            r[0]: r[1]
            for r in conn.execute("SELECT key, value_json FROM meta").fetchall()
        }
        hist = conn.execute(
            "SELECT version, migration_digest FROM migration_history"
        ).fetchall()
    assert meta["schema_version"] == str(SCHEMA_VERSION)
    assert meta["projection_generation"] == "1"
    assert meta["policy_epoch"] == "0"
    assert len(hist) == 1
    assert hist[0][0] == SCHEMA_VERSION
    assert len(hist[0][1]) == 64  # sha256 hex digest of the applied DDL


def test_create_refuses_existing_database(store_path: str) -> None:
    Store.create(store_path).close()
    with pytest.raises(VerbatimError) as exc:
        Store.create(store_path)
    assert exc.value.code == ErrorCode.CONFIG_INVALID


def test_open_missing_database_fails(tmp_path: Path) -> None:
    with pytest.raises(VerbatimError) as exc:
        Store.open(str(tmp_path / "nope.db"))
    assert exc.value.code == ErrorCode.CONFIG_INVALID


# ----------------------------------------------------------------------
# pragmas and filesystem contract
# ----------------------------------------------------------------------


def test_durability_pragmas(store: Store) -> None:
    writer = store._writer
    assert writer.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert writer.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
    assert writer.execute("PRAGMA busy_timeout").fetchone()[0] == 250
    assert writer.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with store.read() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1


def test_file_permissions(store: Store, store_path: str) -> None:
    imode = stat.S_IMODE
    assert imode(os.stat(store_path).st_mode) == 0o600
    assert imode(os.stat(store_path + ".key").st_mode) == 0o600
    # the data directory Store.create made for us is owner-only
    assert imode(os.stat(os.path.dirname(store_path)).st_mode) == 0o700


def test_readonly_open(store_path: str) -> None:
    store = Store.create(store_path)
    db_id = store.db_id()
    store.close()

    ro = Store.open(store_path, readonly=True)
    assert ro.db_id() == db_id
    assert ro.projection_generation() == 1
    assert ro.check_integrity()["foreign_key_violations"] == 0
    with pytest.raises(VerbatimError) as exc:
        with ro.tx():
            pass
    assert exc.value.code == ErrorCode.STORE_WRITE_FAILED
    ro.close()


# ----------------------------------------------------------------------
# hmac keying
# ----------------------------------------------------------------------


def test_hmac_deterministic_and_keyed(store: Store, tmp_path: Path) -> None:
    a = store.hmac(b"payload")
    assert a == store.hmac(b"payload")
    assert a != store.hmac(b"other")
    assert len(a) == 32
    # a different profile key yields different fingerprints for same bytes
    other = Store.create(str(tmp_path / "other" / "o.db"))
    try:
        assert other.hmac(b"payload") != a
    finally:
        other.close()


def test_missing_key_fails_without_regenerating(store_path: str) -> None:
    Store.create(store_path).close()
    key_path = store_path + ".key"
    os.remove(key_path)
    with pytest.raises(VerbatimError) as exc:
        Store.open(store_path)
    # V4-39.02 compat note: missing keys were CONFIG_INVALID; the v4
    # contract fails LOCKED — a locked-key state, never silent regeneration.
    assert exc.value.code == ErrorCode.LOCKED
    # never silently generate a different key — dedup HMACs would corrupt
    assert not os.path.exists(key_path)


def test_corrupt_key_rejected(store_path: str, tmp_path: Path) -> None:
    key_path = store_path + ".key"
    Store.create(store_path).close()
    with open(key_path, "wb") as fh:
        fh.write(b"short")
    with pytest.raises(VerbatimError) as exc:
        Store.open(store_path)
    # V4-39.02 compat note: malformed keys were CONFIG_INVALID → LOCKED.
    assert exc.value.code == ErrorCode.LOCKED


def test_existing_key_file_is_reused(tmp_path: Path) -> None:
    first = Store.create(str(tmp_path / "a" / "a.db"))
    digest = first.hmac(b"x")
    key_path = str(tmp_path / "a" / "a.db.key")
    first.close()
    second = Store.create(
        str(tmp_path / "b" / "b.db"), hmac_key_path=key_path
    )
    try:
        assert second.hmac(b"x") == digest
    finally:
        second.close()


# ----------------------------------------------------------------------
# contention and schema compatibility
# ----------------------------------------------------------------------


def test_store_busy_is_retryable(store: Store, store_path: str) -> None:
    # a second connection holds the write lock; the busy timeout expires
    other = sqlite3.connect(store_path, isolation_level=None)
    other.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(VerbatimError) as exc:
            with store.tx():
                pass
        assert exc.value.code == ErrorCode.STORE_BUSY
        assert exc.value.retryable is True
    finally:
        other.execute("ROLLBACK")
        other.close()
    # the store recovers once the lock is released
    with store.tx() as conn:
        ensure_scope(store, conn, make_scope())


def test_future_schema_rejected(store_path: str) -> None:
    store = Store.create(store_path)
    with store.tx() as conn:
        conn.execute("UPDATE meta SET value_json = '99' WHERE key = 'schema_version'")
    store.close()
    with pytest.raises(VerbatimError) as exc:
        Store.open(store_path)
    assert exc.value.code == ErrorCode.SCHEMA_UNSUPPORTED


def test_not_a_verbatim_db_fails(tmp_path: Path) -> None:
    path = str(tmp_path / "random.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE stuff(x)")
    conn.close()
    with pytest.raises(VerbatimError) as exc:
        Store.open(path)
    assert exc.value.code == ErrorCode.STORE_CORRUPT


# ----------------------------------------------------------------------
# logical clock, integrity, backup, migrations
# ----------------------------------------------------------------------


def test_next_event_us_nondecreasing_under_rollback(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = store.next_event_us()
    # simulate a wall-clock regression far into the past
    monkeypatch.setattr("verbatim.storage.store.now_us", lambda: 1)
    second = store.next_event_us()
    assert second > first


def test_check_integrity(store: Store) -> None:
    info = store.check_integrity()
    assert info["foreign_key_violations"] == 0
    assert info["quick_check"] == "ok"
    assert info["fts5"] is True
    assert info["counts"]["claims"] == 0
    assert info["schema_version"] == SCHEMA_VERSION


def test_backup_produces_consistent_copy(
    store: Store, tmp_path: Path, scope
) -> None:
    SourcesRepo(store).insert(make_envelope(scope, payload=b"evidence"))
    dest = str(tmp_path / "backup" / "backup.db")
    returned = store.backup(dest)
    assert returned == dest
    assert stat.S_IMODE(os.stat(dest).st_mode) == 0o600

    backup = Store.open(dest, hmac_key_path=store.key_path)
    try:
        assert backup.db_id() == store.db_id()
        info = backup.check_integrity()
        assert info["foreign_key_violations"] == 0
        assert info["counts"]["sources"] == 1
        assert info["counts"]["source_revisions"] == 1
    finally:
        backup.close()


def test_backup_refuses_same_path(store: Store, store_path: str) -> None:
    with pytest.raises(VerbatimError) as exc:
        store.backup(store_path)
    assert exc.value.code == ErrorCode.VALIDATION


def test_migrations_apply_is_noop_on_current(store: Store) -> None:
    assert migrations.apply(store) == SCHEMA_VERSION


def test_migrations_reject_future(store: Store) -> None:
    store._schema_version = 99
    with pytest.raises(VerbatimError) as exc:
        migrations.apply(store)
    assert exc.value.code == ErrorCode.SCHEMA_UNSUPPORTED


def test_close_is_idempotent_and_blocks_ops(store: Store) -> None:
    store.close()
    store.close()
    with pytest.raises(VerbatimError):
        with store.read():
            pass


def test_bump_generation_and_policy_epoch(store: Store) -> None:
    assert store.projection_generation() == 1
    with store.tx() as conn:
        assert store.bump_generation(conn) == 2
        store.set_policy_epoch(conn, 3)
    assert store.projection_generation() == 2
    assert store.policy_epoch() == 3


def test_nested_write_tx_fails_fast(store: Store) -> None:
    """A second tx() inside an open one must not deadlock the process."""
    with store.tx():
        with pytest.raises(VerbatimError) as exc:
            with store.tx():
                pass
        assert exc.value.code == ErrorCode.VALIDATION
