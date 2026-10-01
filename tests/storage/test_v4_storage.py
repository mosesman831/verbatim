"""V4 storage hardening regressions (SPEC_V4 §39–§41; findings F4-13/F4-14).

Covers:

- deadline-aware writer admission: the in-process write lock honors the
  request's remaining budget and SQLite busy_timeout is bounded by the
  smaller of the configured allowance and remaining time
  (V4-40.02/40.03), with measured wait/hold metrics surfaced through
  ``Store.diagnostics`` (V4-40.07);
- the durable key protocol: exclusive O_EXCL creation, write-all +
  length verification, 0o600 permissions, fsync of the key file and the
  parent directory (V4-39.01), and LOCKED — never silent regeneration —
  on missing or malformed keys (V4-39.02, C60);
- verifiable migrations: recorded ``migration_history`` digests are
  checked against immutable migration/creation definitions before any
  mutation (V4-41.03, C63), phases are recorded in ``schema_operations``
  with declared→running→applied transitions and the public version
  advances only after all phases succeed (V4-41.05/06), and migration
  ownership is rechecked after the exclusive cross-process lock so two
  processes cannot both migrate (V4-41.07, C64).
"""

from __future__ import annotations

import os
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.storage.test_migration_resilience import _fixture_db

from verbatim.core.time import now_us
from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.storage import migrations
from verbatim.storage.migrations import MIGRATIONS
from verbatim.storage.schema import SCHEMA_VERSION
from verbatim.storage.store import Store


# ----------------------------------------------------------------------
# F4-13: deadline-aware writer admission + bounded busy_timeout (V4-40.*)
# ----------------------------------------------------------------------


def test_write_tx_admission_bounded_by_budget(store: Store) -> None:
    """A held in-process write lock + a bounded request budget must fail
    with typed backpressure, not an unbounded wait (V4-40.02)."""
    store._write_lock.acquire()
    try:
        t0 = time.monotonic()
        with pytest.raises(VerbatimError) as exc:
            with store.tx(budget_ms=50):
                pass
        assert exc.value.code == ErrorCode.BACKPRESSURE
        assert exc.value.retryable is True
        # well under the budget horizon; proves the wait was bounded
        assert time.monotonic() - t0 < 0.5
    finally:
        store._write_lock.release()
    assert store.diagnostics()["write_tx"]["admission_timeouts"] == 1
    # the store recovers once the holder releases
    with store.tx():
        pass


def test_write_tx_expired_deadline_fails_immediately(store: Store) -> None:
    """A deadline already in the past must not touch the lock at all."""
    t0 = time.monotonic()
    with pytest.raises(VerbatimError) as exc:
        with store.tx(deadline_us=now_us() - 1):
            pass
    assert exc.value.code == ErrorCode.BACKPRESSURE
    assert exc.value.retryable is True
    assert time.monotonic() - t0 < 0.5


def test_write_tx_budget_ms_zero_is_backpressure(store: Store) -> None:
    with pytest.raises(VerbatimError) as exc:
        with store.tx(budget_ms=0):
            pass
    assert exc.value.code == ErrorCode.BACKPRESSURE


def test_busy_timeout_bounded_by_remaining_budget(
    store: Store, store_path: str
) -> None:
    """Cross-process contention: SQLite busy_timeout is clamped to
    min(configured 250ms, remaining budget) — V4-40.03."""
    other = sqlite3.connect(store_path, isolation_level=None)
    other.execute("BEGIN IMMEDIATE")
    try:
        t0 = time.monotonic()
        with pytest.raises(VerbatimError) as exc:
            with store.tx(budget_ms=40):
                pass
        assert exc.value.code == ErrorCode.STORE_BUSY
        assert exc.value.retryable is True
        # fails near the 40ms remaining budget — far below the configured
        # 250ms contention allowance
        assert time.monotonic() - t0 < 0.2
    finally:
        other.execute("ROLLBACK")
        other.close()
    # the configured allowance is restored for unbudgeted writers
    assert store._writer.execute("PRAGMA busy_timeout").fetchone()[0] == 250
    assert store.diagnostics()["write_tx"]["busy_errors"] >= 1
    with store.tx():
        pass


def test_write_metrics_surface_in_diagnostics(store: Store) -> None:
    """V4-40.07: wait/hold are measured and exposed via diagnostics."""
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES ('s-diag', 'prof', 'owner')"
        )
    with store.read():
        pass
    d = store.diagnostics()
    wt = d["write_tx"]
    assert wt["count"] >= 1
    assert wt["hold_us_total"] >= wt["hold_us_last"] >= 0
    assert wt["wait_us_max"] >= 0
    assert wt["hold_target_ms"] == 25
    assert wt["admission_timeouts"] == 0
    assert d["wal_bytes"] >= 0
    assert d["reader_count"] >= 1
    assert d["oldest_reader_age_us"] >= 0
    # PASSIVE checkpoint progress is reported on a writable store
    assert isinstance(d["wal_checkpoint"], dict)
    assert set(d["wal_checkpoint"]) == {
        "busy",
        "log_frames",
        "checkpointed_frames",
    }


# ----------------------------------------------------------------------
# F4-14: durable key protocol (V4-39.01/02, C60)
# ----------------------------------------------------------------------


def test_key_creation_is_durable_and_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The key file is written completely, fsynced, then the parent
    directory is fsynced before the key may back commits (V4-39.01)."""
    fsynced: list[str] = []
    real_fsync = os.fsync

    def _spy(fd: int) -> None:
        try:
            fsynced.append(os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:
            fsynced.append("?")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _spy)
    db = str(tmp_path / "k.db")
    store = Store.create(db)
    store.close()

    key_path = db + ".key"
    assert os.path.getsize(key_path) == 32
    assert stat.S_IMODE(os.stat(key_path).st_mode) == 0o600
    # at least the key file and its parent directory were fsynced
    assert len(fsynced) >= 2
    if any(t != "?" for t in fsynced):  # /proc fd resolution is Linux-only
        assert key_path in fsynced
        assert os.path.dirname(key_path) in fsynced


def test_concurrent_key_creation_converges(tmp_path: Path) -> None:
    """Losers of the O_EXCL race converge on the winner's durable key —
    never overwrite, never regenerate (V4-39.01/02)."""
    key_path = str(tmp_path / "race.key")
    results: list[bytes] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def _worker() -> None:
        try:
            barrier.wait(timeout=5)
            results.append(Store._load_or_create_key(key_path, create=True))
        except BaseException as exc:  # noqa: BLE001 - collect for assert
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert not errors
    assert len(results) == 8
    assert all(k == results[0] for k in results)
    assert len(results[0]) == 32


def test_partial_key_fails_locked_on_open(tmp_path: Path) -> None:
    """A torn/short key fails LOCKED and is never regenerated (C60)."""
    db = str(tmp_path / "p.db")
    Store.create(db).close()
    key_path = db + ".key"
    with open(key_path, "wb") as fh:
        fh.write(b"\x01" * 8)
    with pytest.raises(VerbatimError) as exc:
        Store.open(db)
    assert exc.value.code == ErrorCode.LOCKED
    assert os.path.getsize(key_path) == 8  # untouched


def test_missing_key_fails_locked(tmp_path: Path) -> None:
    db = str(tmp_path / "m.db")
    Store.create(db).close()
    os.remove(db + ".key")
    with pytest.raises(VerbatimError) as exc:
        Store.open(db)
    assert exc.value.code == ErrorCode.LOCKED
    assert not os.path.exists(db + ".key")


def test_create_never_overwrites_malformed_key(tmp_path: Path) -> None:
    """A malformed pre-existing key blocks even creation — no silent
    regeneration path exists (V4-39.02)."""
    db = str(tmp_path / "c.db")
    key_path = db + ".key"
    with open(key_path, "wb") as fh:
        fh.write(b"torn")
    with pytest.raises(VerbatimError) as exc:
        Store.create(db)
    assert exc.value.code == ErrorCode.LOCKED
    assert Path(key_path).read_bytes() == b"torn"
    assert not os.path.exists(db)


# ----------------------------------------------------------------------
# F4-14: verifiable phased migrations (V4-41.03/05/06/07, C63, C64)
# ----------------------------------------------------------------------


def test_corrupted_history_digest_blocks_apply(tmp_path: Path) -> None:
    """C63: a changed historical migration checksum blocks mutation."""
    db = str(tmp_path / "corrupt.db")
    conn = _fixture_db(db, version=3, v3_ddl=True)
    conn.commit()
    conn.close()
    raw = sqlite3.connect(db)
    raw.execute(
        "UPDATE migration_history SET migration_digest='tampered'"
        " WHERE version = 3"
    )
    raw.commit()
    raw.close()

    with pytest.raises(VerbatimError) as exc:
        Store.open(db)
    assert exc.value.code == ErrorCode.STORE_CORRUPT
    assert "checksum mismatch" in exc.value.message
    assert "version 3" in exc.value.message


def test_corrupted_history_digest_blocks_apply_on_open_store(
    tmp_path: Path,
) -> None:
    """Verification runs on every apply(), not just at open."""
    db = str(tmp_path / "live.db")
    conn = _fixture_db(db, version=3, v3_ddl=True)
    conn.commit()
    conn.close()
    store = Store.open(db)
    try:
        with store.tx() as conn:
            conn.execute(
                "UPDATE migration_history SET migration_digest='x' ||"
                " substr(migration_digest, 3) WHERE version = 4"
            )
        with pytest.raises(VerbatimError) as exc:
            migrations.apply(store)
        assert exc.value.code == ErrorCode.STORE_CORRUPT
    finally:
        store.close()


def test_unknown_history_version_blocks(tmp_path: Path) -> None:
    """A history row for a version this binary never produced is drift."""
    db = str(tmp_path / "unknown.db")
    conn = _fixture_db(db, version=3, v3_ddl=True)
    conn.execute(
        "INSERT INTO migration_history(version, applied_us, migration_digest)"
        " VALUES (9, 0, 'whatever')"
    )
    conn.commit()
    conn.close()
    with pytest.raises(VerbatimError) as exc:
        Store.open(db)
    assert exc.value.code == ErrorCode.STORE_CORRUPT
    assert "unknown schema version" in exc.value.message


def test_migration_phases_recorded_in_schema_operations(
    tmp_path: Path,
) -> None:
    """V4-41.05: each migration phase lands in schema_operations with the
    migration's immutable checksum; the durable state is 'applied' only
    after the phase's transaction commits."""
    db = str(tmp_path / "phases.db")
    conn = _fixture_db(db, version=3, v3_ddl=True)
    conn.commit()
    conn.close()

    store = Store.open(db)
    try:
        assert store.schema_version == SCHEMA_VERSION
        with store.read() as conn:
            rows = conn.execute(
                "SELECT operation_id, version_from, version_to, phase,"
                "       checksum, state, owner_lease"
                " FROM schema_operations"
            ).fetchall()
        assert rows, "no schema_operations recorded"
        ops = {r[0]: r for r in rows}
        mig = ops["schema-migration:v4-kernel-contracts"]
        assert mig[1] == 3 and mig[2] == 4
        assert mig[3] == "ddl"
        # v5 compat note: MIGRATIONS[-1] is now the v5 step; pin the v4
        # migration by name so this assertion keeps its meaning.
        v4_migration = next(
            m for m in MIGRATIONS if m.name == "v4-kernel-contracts"
        )
        assert mig[4] == v4_migration.digest()
        assert mig[5] == "applied"
        assert mig[6].startswith("pid:")
        # deferred resumable phases that did work are recorded too
        # (the v3 CHECK rebuilds run on this fixture)
        assert "schema-phase:v3-table-rebuild" in ops
        assert ops["schema-phase:v3-table-rebuild"][5] == "applied"
        # no phase may be left in a non-terminal state
        assert all(r[5] == "applied" for r in rows)
    finally:
        store.close()


def test_fresh_store_history_verifies_clean(store: Store) -> None:
    """A fresh store's creation digest is a legitimate history row —
    apply() stays a no-op (V4-41.03 must not brick healthy stores)."""
    assert migrations.apply(store) == SCHEMA_VERSION


def test_concurrent_open_does_not_double_migrate(tmp_path: Path) -> None:
    """C64/V4-40.12: two Store objects on one file serialize migration
    through BEGIN EXCLUSIVE + the post-lock version recheck."""
    db = str(tmp_path / "shared.db")
    conn = _fixture_db(db, version=3, v3_ddl=True)
    conn.commit()
    conn.close()

    results: list[Store] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def _open() -> None:
        try:
            barrier.wait(timeout=5)
            results.append(Store.open(db))
        except BaseException as exc:  # noqa: BLE001 - collect for assert
            errors.append(exc)

    threads = [threading.Thread(target=_open) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    try:
        assert not errors
        assert len(results) == 2
        assert all(s.schema_version == SCHEMA_VERSION for s in results)
        raw = sqlite3.connect(db)
        try:
            versions = [
                r[0]
                for r in raw.execute(
                    "SELECT version FROM migration_history ORDER BY version"
                )
            ]
            assert versions == sorted(set(versions))  # no duplicated version
            assert SCHEMA_VERSION in versions
            applied = raw.execute(
                "SELECT COUNT(*) FROM schema_operations"
                " WHERE operation_id='schema-migration:v4-kernel-contracts'"
                "   AND state='applied'"
            ).fetchone()[0]
            assert applied == 1
        finally:
            raw.close()
    finally:
        for s in results:
            s.close()


def test_concurrent_process_open_migrates_once(tmp_path: Path) -> None:
    """C64: two real processes opening the same unmigrated file cannot
    both migrate — BEGIN EXCLUSIVE serializes them and the loser adopts
    the winner's recorded version."""
    db = str(tmp_path / "proc.db")
    conn = _fixture_db(db, version=3, v3_ddl=True)
    conn.commit()
    conn.close()

    repo = str(Path(__file__).resolve().parents[2])
    script = (
        "import sys; sys.path.insert(0, {repo!r});"
        " from verbatim.storage.store import Store;"
        " s = Store.open({db!r});"
        " assert s.schema_version == {ver}, s.schema_version;"
        " s.close()"
    ).format(repo=repo, db=db, ver=SCHEMA_VERSION)
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        for _ in range(2)
    ]
    for p in procs:
        _, stderr = p.communicate(timeout=60)
        assert p.returncode == 0, stderr.decode()

    raw = sqlite3.connect(db)
    try:
        v4_rows = raw.execute(
            "SELECT COUNT(*) FROM migration_history WHERE version = ?",
            (SCHEMA_VERSION,),
        ).fetchone()[0]
        assert v4_rows == 1
    finally:
        raw.close()
