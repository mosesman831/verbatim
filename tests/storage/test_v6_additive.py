"""V6 additive-schema tests (SPEC_V6 V6-03.14/15, docs/v6_contracts.md §8).

Covers the additive-table pattern choice: ``source_exposure`` and
``policy_artifact_attestations`` ride NEITHER ``v5_statements()`` nor
``_creation_ddl_v5()`` — those byte strings are hashed into
``Migration(5).digest()`` and every recorded ``migration_history`` row
verifies against them (V4-41.03). Instead the tables arrive through the
resumable ``_ensure_v6_additive`` phase on every ``apply()`` plus lazy
``ensure_additive_tables`` inside writer transactions for
``Store.create``-fresh stores. Real on-disk databases throughout.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verbatim.core.types import json_dumps
from verbatim.core.time import now_us
from verbatim.storage import migrations
from verbatim.storage.schema_v5 import (
    ADDITIVE_TABLES,
    ensure_additive_tables,
    v5_additive_statements,
)
from verbatim.storage.store import Store

from tests.storage.test_v5_migration import _v4_fixture_db


def _table_names(conn: sqlite3.Connection) -> set:
    return {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def test_fresh_create_lacks_tables_until_open_or_write(tmp_path):
    """Store.create never runs apply(): a fresh store has no additive
    tables until either a writable open (ensure phase) or a writer's
    lazy ensure — documented V6 additive path."""
    db = str(tmp_path / "fresh.db")
    s = Store.create(db)
    with s.read() as conn:
        names = _table_names(conn)
        for t in ADDITIVE_TABLES:
            assert t not in names
    # Lazy path: ensure inside the caller's own write tx.
    with s.tx() as conn:
        ensure_additive_tables(conn)
    with s.read() as conn:
        names = _table_names(conn)
        for t in ADDITIVE_TABLES:
            assert t in names
    s.close()


def test_writable_open_ensures_tables_and_ledgers(tmp_path):
    """The deferred phase runs on Store.open→apply: tables appear and
    the schema_operations ledger records exactly one applied phase."""
    db = str(tmp_path / "open.db")
    Store.create(db).close()
    s = Store.open(db)
    try:
        with s.read() as conn:
            names = _table_names(conn)
            for t in ADDITIVE_TABLES:
                assert t in names
            ops = {
                r[0]: r[1]
                for r in conn.execute(
                    "SELECT operation_id, state FROM schema_operations"
                )
            }
            assert ops["schema-phase:v6-additive-ensure"] == "applied"
    finally:
        s.close()
    # Idempotent: a second apply does real-work=False → no new ledger row.
    s = Store.open(db)
    try:
        assert migrations._ensure_v6_additive(s) is False
        with s.read() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM schema_operations"
                " WHERE operation_id='schema-phase:v6-additive-ensure'"
            ).fetchone()[0]
            assert n == 1
    finally:
        s.close()


def test_migrated_v4_store_gains_additive_tables(tmp_path):
    """A pre-existing v4 store migrating 4→5 gets the additive tables in
    the same open — the ensure phase runs after the version commit."""
    db = str(tmp_path / "v4.db")
    conn = _v4_fixture_db(db)
    conn.commit()
    conn.close()
    s = Store.open(db)
    try:
        assert s.schema_version == 5
        with s.read() as conn:
            names = _table_names(conn)
            for t in ADDITIVE_TABLES:
                assert t in names
    finally:
        s.close()


def test_additive_tables_not_in_v5_statements():
    """The digests-verified v5 material stays byte-frozen: neither
    additive table's DDL may ride v5_statements() (it would change
    Migration(5).digest() and _creation_ddl_v5(), invalidating every
    recorded v5 history row — V4-41.03)."""
    stmts = "\n".join(migrations.MIGRATIONS[-1].statements)
    for t in ADDITIVE_TABLES:
        assert f"CREATE TABLE IF NOT EXISTS {t}" not in stmts
    additive = "\n".join(v5_additive_statements())
    for t in ADDITIVE_TABLES:
        assert f"CREATE TABLE IF NOT EXISTS {t}" in additive


def test_history_digests_still_verify(tmp_path):
    """Recorded history remains verifiable end-to-end: a fresh store's
    creation digest plus a migrated store's migration digest both pass
    acceptable_history_digests — proof the additive path left the
    immutable definitions untouched."""
    db = str(tmp_path / "d.db")
    Store.create(db).close()
    raw = sqlite3.connect(db)
    digest = raw.execute(
        "SELECT migration_digest FROM migration_history WHERE version=5"
    ).fetchone()[0]
    raw.close()
    assert digest in migrations.acceptable_history_digests(5)


def test_ensure_additive_rolls_back_with_caller_tx(tmp_path):
    """The lazy ensure is transactional: a rollback takes the CREATEs
    with it — the table can never exist half-committed beside a failed
    write."""
    s = Store.create(str(tmp_path / "rb.db"))
    try:
        with pytest.raises(RuntimeError):
            with s.tx() as conn:
                ensure_additive_tables(conn)
                raise RuntimeError("caller abort")
        with s.read() as conn:
            for t in ADDITIVE_TABLES:
                assert t not in _table_names(conn)
    finally:
        s.close()


def test_source_exposure_columns(tmp_path):
    """The contract columns + PK land exactly (contracts §8:
    receipt_id, ord, source_id, revision, score_family,
    delivered_at_us — plus namespace per the worker brief)."""
    s = Store.create(str(tmp_path / "cols.db"))
    try:
        with s.tx() as conn:
            ensure_additive_tables(conn)
        with s.read() as conn:
            cols = {
                r[1]: r for r in conn.execute(
                    "PRAGMA table_info(source_exposure)"
                )
            }
            assert set(cols) == {
                "receipt_id", "ord", "source_id", "revision",
                "score_family", "delivered_at_us", "namespace",
            }
            pk = sorted(
                r[1] for r in conn.execute(
                    "PRAGMA table_info(source_exposure)"
                ) if r[5] > 0
            )
            assert pk == ["ord", "receipt_id"]
    finally:
        s.close()


def test_attestation_columns_and_state_check(tmp_path):
    s = Store.create(str(tmp_path / "att.db"))
    try:
        with s.tx() as conn:
            ensure_additive_tables(conn)
            conn.execute(
                "INSERT INTO policy_artifact_attestations"
                " (artifact_id, attested_us, state, evidence_json)"
                " VALUES ('a1', ?, 'validated', ?)",
                (now_us(), json_dumps({"gate": "G8"})),
            )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO policy_artifact_attestations"
                    " (artifact_id, attested_us, state, evidence_json)"
                    " VALUES ('a1', ?, 'unvalidated', '{}')",
                    (now_us() + 1,),
                )
    finally:
        s.close()
