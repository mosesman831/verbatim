"""Migration resilience: crash-after-version-commit recovery, open-time
auto-migration, FTS backfill on migrated stores, and control-lane
preservation for in-flight jobs (SPEC §21, SPEC_V3 §39–§40, §57).
"""

from __future__ import annotations

import os
import sqlite3

from verbatim.core.time import now_us
from verbatim.core.types import json_dumps
from verbatim.storage import migrations
from verbatim.storage.schema import (
    DDL_FTS5,
    DDL_V1,
    DDL_V2,
    DDL_V2_ALTER,
    DDL_V2_JOBS_REBUILD,
    FTS_TRIGGERS,
    SCHEMA_VERSION,
)
from verbatim.storage.schema_v3 import DDL_V3, DDL_V3_ALTER
from verbatim.storage.store import Store


def _fixture_db(
    path: str,
    *,
    version: int,
    fts: bool = True,
    v3_ddl: bool = False,
    v2_jobs_rebuild: bool = True,
) -> sqlite3.Connection:
    """Hand-build a database file at a chosen recorded schema_version.

    ``v3_ddl`` applies the v3 new-table + ALTER set WITHOUT the
    CHECK-widening rebuilds — the durable state a crash leaves behind when
    it commits ``meta.schema_version`` before the post-migration rebuilds
    finish. ``v2_jobs_rebuild=False`` additionally leaves the v1 jobs
    CHECK, the state left when even the first rebuild never ran.
    """
    key_path = path + ".key"
    with open(key_path, "wb") as fh:
        fh.write(os.urandom(32))
    os.chmod(key_path, 0o600)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(DDL_V1)
    conn.executescript(DDL_V2)
    for stmt in DDL_V2_ALTER.split(";"):
        if stmt.strip():
            conn.execute(stmt)
    if v2_jobs_rebuild:
        for stmt in DDL_V2_JOBS_REBUILD.split(";"):
            if stmt.strip():
                conn.execute(stmt)
    if v3_ddl:
        conn.executescript(DDL_V3)
        for stmt in DDL_V3_ALTER.split(";"):
            if stmt.strip():
                conn.execute(stmt)
    if fts:
        conn.executescript(DDL_FTS5)
        conn.executescript(FTS_TRIGGERS)
    for key, value in (
        ("schema_version", version),
        ("db_id", "db-fixture"),
        ("created_us", now_us()),
        ("projection_generation", 1),
        ("policy_epoch", 0),
    ):
        conn.execute(
            "INSERT INTO meta(key, value_json) VALUES (?, ?)",
            (key, json_dumps(value)),
        )
    # V4-41.03: the recorded digest must equal an immutable definition —
    # a migrated-to-this-version row carries Migration.digest(); a
    # created-at version with no migration carries the creation digest.
    hist_digest = next(
        (m.digest() for m in migrations.MIGRATIONS if m.version == version),
        None,
    )
    if hist_digest is None:
        accepted = sorted(migrations.acceptable_history_digests(version))
        hist_digest = accepted[0] if accepted else "unknown-fixture-version"
    conn.execute(
        "INSERT INTO migration_history(version, applied_us, migration_digest)"
        " VALUES (?, ?, ?)",
        (version, now_us(), hist_digest),
    )
    conn.execute(
        "INSERT INTO scopes (scope_id, profile_id, visibility)"
        " VALUES ('s1', 'prof', 'owner')"
    )
    return conn


def _sql_of(conn: sqlite3.Connection, table: str) -> str:
    return conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()[0]


def test_open_recovers_crash_after_version_commit(tmp_path):
    """schema_version=3 recorded while jobs/procedures keep pre-v3 CHECKs:
    a writable open must resume the deferred rebuilds (no reachable path
    used to exist — every v3 job enqueue would have failed forever)."""
    db = str(tmp_path / "half.db")
    conn = _fixture_db(db, version=3, v3_ddl=True)
    conn.execute(
        "INSERT INTO procedures (procedure_id, scope_id, task_label, state)"
        " VALUES ('p1', 's1', 't', 'proposed')"
    )
    conn.execute(
        "INSERT INTO jobs (job_id, scope_id, kind, state, lane)"
        " VALUES ('j-purge', 's1', 'purge', 'queued', 'ordinary')"
    )
    conn.commit()
    conn.close()

    store = Store.open(db)
    try:
        assert store.schema_version == SCHEMA_VERSION
        with store.read() as conn:
            assert "procedure_compile" in _sql_of(conn, "jobs")
            assert "candidate" in _sql_of(conn, "procedures")
            # V3-57.05: proposed→candidate mapping ran during the rebuild.
            assert conn.execute(
                "SELECT state FROM procedures WHERE procedure_id='p1'"
            ).fetchone()[0] == "candidate"
            # In-flight control job inherited the 'ordinary' ALTER default;
            # the rebuild restores its reserved lane (jobs/queue.py).
            assert conn.execute(
                "SELECT lane FROM jobs WHERE job_id='j-purge'"
            ).fetchone()[0] == "control"
        with store.tx() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, scope_id, kind, state)"
                " VALUES ('j2', 's1', 'procedure_compile', 'queued')"
            )
            conn.execute(
                "INSERT INTO procedures (procedure_id, scope_id, task_label, state)"
                " VALUES ('p2', 's1', 't', 'candidate')"
            )
        # apply() is a no-op once recovered.
        assert migrations.apply(store) == SCHEMA_VERSION
    finally:
        store.close()


def test_open_recovers_v1_origin_half_migration(tmp_path):
    """v1 jobs CHECK (no 'replay') under a recorded v3 version: the v2
    rebuild must resume before the v3 rebuild, and lanes survive."""
    db = str(tmp_path / "half1.db")
    conn = _fixture_db(db, version=3, v3_ddl=True, v2_jobs_rebuild=False)
    conn.execute(
        "INSERT INTO jobs (job_id, scope_id, kind, state)"
        " VALUES ('j-purge', 's1', 'purge', 'queued')"
    )
    conn.execute(
        "INSERT INTO procedures (procedure_id, scope_id, task_label, state)"
        " VALUES ('p1', 's1', 't', 'proposed')"
    )
    conn.commit()
    conn.close()

    store = Store.open(db)
    try:
        assert store.schema_version == SCHEMA_VERSION
        with store.read() as conn:
            assert "procedure_compile" in _sql_of(conn, "jobs")
            assert "candidate" in _sql_of(conn, "procedures")
            assert conn.execute(
                "SELECT lane FROM jobs WHERE job_id='j-purge'"
            ).fetchone()[0] == "control"
            assert conn.execute(
                "SELECT state FROM procedures WHERE procedure_id='p1'"
            ).fetchone()[0] == "candidate"
        with store.tx() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, scope_id, kind, state)"
                " VALUES ('j2', 's1', 'procedure_compile', 'queued')"
            )
    finally:
        store.close()


def test_open_store_auto_migrates_v2_and_writes(tmp_path):
    """api.open_store on a v2-versioned DB upgrades and accepts writes —
    previously every shipped surface left v1/v2 stores write-frozen."""
    from verbatim.api import open_store
    from verbatim.config import config_from_mapping
    from verbatim.host import LocalHost

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db = str(data_dir / "demo.db")
    conn = _fixture_db(db, version=2)
    conn.commit()
    conn.close()

    eng = open_store(
        str(data_dir),
        config_from_mapping({"mode": "offline_rules"}),
        LocalHost(profile_id="demo", principal_id="me", conversation_id="c1"),
    )
    try:
        assert eng.store.schema_version == SCHEMA_VERSION
        with eng.store.tx() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, scope_id, kind, state)"
                " VALUES ('j1', 's1', 'procedure_compile', 'queued')"
            )
        with eng.store.read() as conn:
            names = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            assert "vault_entries" in names
            hist = [
                r[0]
                for r in conn.execute(
                    "SELECT version FROM migration_history ORDER BY applied_us"
                )
            ]
            assert 3 in hist
    finally:
        eng.close()


def test_migrated_store_gains_fts(tmp_path):
    """A v2 database without the FTS5 index gains it on migrate/open —
    previously migrated stores stayed fts_enabled=False forever."""
    db = str(tmp_path / "nofts.db")
    conn = _fixture_db(db, version=2, fts=False)
    conn.commit()
    conn.close()

    store = Store.open(db)
    try:
        assert store.schema_version == SCHEMA_VERSION
        assert store.fts_enabled is True
        with store.read() as conn:
            assert conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table'"
                " AND name='facts_fts_idx'"
            ).fetchone() is not None
            triggers = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                )
            }
            assert {"facts_fts_ai", "facts_fts_ad", "facts_fts_au"} <= triggers
    finally:
        store.close()


def test_open_rebuilds_global_sources_dedup_index(tmp_path):
    """A store carrying the pre-partition ``UNIQUE(origin, external_id)``
    index regains the scoped ``UNIQUE(scope_id, origin, external_id)``
    shape on writable open — the same host key then dedups per scope
    instead of colliding across the partition boundary."""
    db = str(tmp_path / "dedup.db")
    conn = _fixture_db(db, version=3, v3_ddl=True)
    conn.execute("DROP INDEX idx_sources_origin_ext")
    conn.execute(
        "CREATE UNIQUE INDEX idx_sources_origin_ext"
        " ON sources(origin, external_id)"
        " WHERE external_id IS NOT NULL"
    )
    conn.execute(
        "INSERT INTO scopes (scope_id, profile_id, visibility)"
        " VALUES ('s2', 'prof', 'owner')"
    )
    conn.commit()
    conn.close()

    store = Store.open(db)
    try:
        with store.read() as conn:
            sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='index'"
                " AND name='idx_sources_origin_ext'"
            ).fetchone()[0]
            assert "scope_id" in sql
        with store.tx() as conn:
            for sid in ("s1", "s2"):
                conn.execute(
                    "INSERT INTO sources (source_id, scope_id, origin,"
                    " external_id, source_kind, speaker_id, created_us)"
                    " VALUES (?, ?, 'ext', 'host-key-1', 'user_message',"
                    " 'u1', 1)",
                    (f"src-{sid}", sid),
                )
            # Same-scope retry still collides (idempotent dedup preserved).
            try:
                conn.execute(
                    "INSERT INTO sources (source_id, scope_id, origin,"
                    " external_id, source_kind, speaker_id, created_us)"
                    " VALUES ('src-s1b', 's1', 'ext', 'host-key-1',"
                    " 'user_message', 'u1', 1)"
                )
            except sqlite3.IntegrityError:
                pass
            else:
                raise AssertionError(
                    "scoped dedup lost: same-scope duplicate accepted"
                )
        # The rebuild is idempotent — a second apply is a no-op.
        assert migrations.apply(store) == SCHEMA_VERSION
    finally:
        store.close()


def test_open_migrates_v3_store_to_v4(tmp_path):
    """A v3 store gains the v4 kernel tables on writable open, the public
    version advances to SCHEMA_VERSION in the same commit as the ledger's
    applied phase, and migration_history records the v4 digest."""
    db = str(tmp_path / "v3.db")
    conn = _fixture_db(db, version=3, v3_ddl=True)
    conn.commit()
    conn.close()

    store = Store.open(db)
    try:
        assert store.schema_version == SCHEMA_VERSION
        with store.read() as conn:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            for t in (
                "objects",
                "object_revisions",
                "dependency_edges",
                "operation_receipts",
                "delivery_permits",
                "dispatch_permits",
                "readiness_obligations",
                "closure_runs",
                "closure_frontier",
                "producer_manifests",
                "view_support",
                "schema_operations",
            ):
                assert t in tables, f"missing v4 table {t}"
            # The ledger recorded the v4 migration's applied phase.
            op = conn.execute(
                "SELECT state, version_to FROM schema_operations"
                " WHERE operation_id='schema-migration:v4-kernel-contracts'"
            ).fetchone()
            assert op is not None and op[0] == "applied" and op[1] == 4
            hist = [
                r[0]
                for r in conn.execute(
                    "SELECT version FROM migration_history ORDER BY version"
                )
            ]
            assert 4 in hist
        # Writes through the v4 surface work on the migrated store.
        from verbatim.jobs.coordinator import Coordinator
        from verbatim.core.types_v4 import Effect, EffectKind, EffectPlan

        receipt = Coordinator(store).apply_plan(
            EffectPlan(
                operation_id="op-mig",
                scope_id="s1",
                producer_id="test",
                input_digests=("d",),
                effects=(
                    Effect(
                        EffectKind.INSERT_OBJECT,
                        "objects",
                        {
                            "object_id": "o-mig",
                            "kind": "claim",
                            "scope_id": "s1",
                            "current_revision": 1,
                            "created_event": 1,
                        },
                    ),
                ),
            )
        )
        assert receipt.effects_applied == 1
    finally:
        store.close()


def test_readonly_open_does_not_migrate(tmp_path):
    """Read-only opens stay read-only: no migration is attempted."""
    db = str(tmp_path / "ro.db")
    conn = _fixture_db(db, version=2)
    conn.commit()
    conn.close()

    store = Store.open(db, readonly=True)
    try:
        assert store.schema_version == 2
    finally:
        store.close()
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT value_json FROM meta WHERE key='schema_version'"
    ).fetchone()
    assert "2" in row[0]
    conn.close()
