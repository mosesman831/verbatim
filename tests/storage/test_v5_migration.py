"""V5 migration tests: 4→5 upgrade of a populated store (docs/v5_contracts.md
§3, SPEC_V5 §07, SPEC §21).

Covers: data preservation across the migration, the checksummed
``migration_history`` row, the ``schema_operations`` ledger entries for
the migration and its deferred resumable phases (jobs CHECK widening,
source FTS creation), crash-after-version-commit resume, checksum
tamper refusal, and the open-time version guards in both directions.
Real on-disk databases throughout (contracts §13.6).
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.storage.test_migration_resilience import _fixture_db

from verbatim.core.time import now_us
from verbatim.core.types import ErrorCode, VerbatimError, json_dumps
from verbatim.storage import migrations
from verbatim.storage.migrations import MIGRATIONS
from verbatim.storage.schema import (
    DDL_FTS5,
    DDL_V1,
    DDL_V2,
    DDL_V2_ALTER,
    DDL_V2_JOBS_REBUILD,
    FTS_TRIGGERS,
    SCHEMA_VERSION,
)
from verbatim.storage.schema_v3 import (
    DDL_V3,
    DDL_V3_ALTER,
    DDL_V3_JOBS_REBUILD,
    DDL_V3_PROCEDURES_REBUILD,
)
from verbatim.storage.schema_v4 import v4_statements
from verbatim.storage.schema_v5 import v5_statements
from verbatim.storage.store import Store, _creation_ddl

V5_MIGRATION = next(m for m in MIGRATIONS if m.version == 5)


def _v4_fixture_db(path: str) -> sqlite3.Connection:
    """Build a byte-legitimate v4-era store: the exact statement sequence
    v4-era ``Store.create`` ran, with the recorded history digest equal to
    the immutable v4 creation material — the state a real v4 store
    presents to v5 code."""
    key_path = path + ".key"
    with open(key_path, "wb") as fh:
        fh.write(os.urandom(32))
    os.chmod(key_path, 0o600)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(DDL_V1)
    conn.executescript(DDL_V2)
    for script in (
        DDL_V2_ALTER,
        DDL_V2_JOBS_REBUILD,
    ):
        for stmt in script.split(";"):
            if stmt.strip():
                conn.execute(stmt)
    conn.executescript(DDL_V3)
    for script in (
        DDL_V3_ALTER,
        DDL_V3_PROCEDURES_REBUILD,
        DDL_V3_JOBS_REBUILD,
    ):
        for stmt in script.split(";"):
            if stmt.strip():
                conn.execute(stmt)
    for stmt in v4_statements():
        conn.execute(stmt)
    conn.executescript(DDL_FTS5)
    conn.executescript(FTS_TRIGGERS)
    for key, value in (
        ("schema_version", 4),
        ("db_id", "db-v4"),
        ("created_us", now_us()),
        ("projection_generation", 1),
        ("policy_epoch", 0),
    ):
        conn.execute(
            "INSERT INTO meta(key, value_json) VALUES (?, ?)",
            (key, json_dumps(value)),
        )
    digest = hashlib.sha256(_creation_ddl().encode("utf-8")).hexdigest()
    conn.execute(
        "INSERT INTO migration_history(version, applied_us, migration_digest)"
        " VALUES (4, ?, ?)",
        (now_us(), digest),
    )
    conn.execute(
        "INSERT INTO scopes (scope_id, profile_id, visibility)"
        " VALUES ('s1', 'prof', 'owner')"
    )
    return conn


def _populate_v4(conn: sqlite3.Connection) -> None:
    """Evidence-plane rows the migration must preserve byte-for-byte."""
    conn.execute(
        "INSERT INTO sources (source_id, scope_id, origin, external_id,"
        " source_kind, speaker_id, created_us) VALUES"
        " ('src-1', 's1', 'ext', 'host-1', 'user_message', 'alice', 1)"
    )
    conn.execute(
        "INSERT INTO source_revisions (source_id, revision, payload,"
        " payload_hmac, event_us, captured_us, timezone, provenance)"
        " VALUES ('src-1', 1, ?, ?, 1, 1, 'UTC', 'direct_user')",
        (b"canonical bytes \x00\xff", b"\xaa" * 32),
    )
    conn.execute(
        "INSERT INTO claims (claim_id, scope_id, subject_id, predicate,"
        " created_event) VALUES ('c1', 's1', 'subj', 'pred', 1)"
    )
    conn.execute(
        "INSERT INTO claim_revisions (claim_id, revision, state,"
        " object_json, recorded_from) VALUES ('c1', 1, 'active', '{}', 1)"
    )
    conn.execute(
        "INSERT INTO jobs (job_id, scope_id, kind, state, lane) VALUES"
        " ('j-old', 's1', 'embed', 'queued', 'ordinary')"
    )
    conn.execute(
        "INSERT INTO objects (object_id, kind, scope_id,"
        " current_revision, created_event) VALUES"
        " ('o1', 'claim', 's1', 1, 1)"
    )


def _table_names(conn: sqlite3.Connection) -> set:
    return {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def test_v4_store_migrates_to_v5_preserving_data(tmp_path: Path) -> None:
    """A populated v4 store opens at version 5 with every pre-existing
    row intact — canonical source bytes byte-for-byte."""
    db = str(tmp_path / "v4.db")
    conn = _v4_fixture_db(db)
    _populate_v4(conn)
    conn.commit()
    conn.close()

    store = Store.open(db)
    try:
        assert store.schema_version == 5
        assert store.source_fts_enabled is True
        with store.read() as conn:
            names = _table_names(conn)
            for t in (
                "source_state",
                "source_lexical_projection",
                "source_fts_rows",
                "source_fts",
                "source_fts_idx",
                "source_vectors",
                "entity_postings",
                "duplicate_links",
                "enrichment",
                "update_candidates",
                "backfill_cursor",
            ):
                assert t in names, f"missing v5 table {t}"
            # evidence preserved
            payload = conn.execute(
                "SELECT payload FROM source_revisions"
                " WHERE source_id='src-1' AND revision=1"
            ).fetchone()[0]
            assert payload == b"canonical bytes \x00\xff"
            assert conn.execute(
                "SELECT state FROM claim_revisions WHERE claim_id='c1'"
            ).fetchone()[0] == "active"
            assert conn.execute(
                "SELECT kind FROM jobs WHERE job_id='j-old'"
            ).fetchone()[0] == "embed"
            assert conn.execute(
                "SELECT COUNT(*) FROM objects WHERE object_id='o1'"
            ).fetchone()[0] == 1
            assert conn.execute(
                "SELECT COUNT(*) FROM sources WHERE source_id='src-1'"
            ).fetchone()[0] == 1
    finally:
        store.close()


def test_v5_migration_records_checksum_and_ledger(tmp_path: Path) -> None:
    """The v5 step lands in migration_history with Migration.digest()
    and in schema_operations with the applied transition (V4-41.05)."""
    db = str(tmp_path / "v4hist.db")
    conn = _v4_fixture_db(db)
    conn.commit()
    conn.close()

    store = Store.open(db)
    try:
        with store.read() as conn:
            hist = conn.execute(
                "SELECT version, migration_digest FROM migration_history"
                " ORDER BY version"
            ).fetchall()
            assert [r[0] for r in hist] == [4, 5]
            assert hist[1][1] == V5_MIGRATION.digest()
            ops = {
                r[0]: r
                for r in conn.execute(
                    "SELECT operation_id, version_from, version_to,"
                    "       phase, checksum, state FROM schema_operations"
                )
            }
            mig = ops["schema-migration:v5-source-projections"]
            assert mig[1] == 4 and mig[2] == 5
            assert mig[3] == "ddl"
            assert mig[4] == V5_MIGRATION.digest()
            assert mig[5] == "applied"
            # deferred resumable phases that did real work are recorded
            jobs_phase = ops["schema-phase:v5-jobs-rebuild"]
            assert jobs_phase[5] == "applied"
            assert jobs_phase[4] == hashlib.sha256(
                migrations.DDL_V5_JOBS_REBUILD.encode("utf-8")
            ).hexdigest()
            fts_phase = ops["schema-phase:source-fts-ensure"]
            assert fts_phase[5] == "applied"
    finally:
        store.close()


def test_migrated_store_accepts_v5_job_kinds(tmp_path: Path) -> None:
    """The deferred jobs-CHECK rebuild ran: source pipeline kinds
    enqueue on the upgraded store."""
    db = str(tmp_path / "v4jobs.db")
    conn = _v4_fixture_db(db)
    conn.commit()
    conn.close()

    store = Store.open(db)
    try:
        with store.tx() as conn:
            for i, kind in enumerate(
                ("source_project", "source_embed", "source_backfill")
            ):
                conn.execute(
                    "INSERT INTO jobs (job_id, scope_id, kind, state)"
                    " VALUES (?, 's1', ?, 'queued')",
                    (f"jv5-{i}", kind),
                )
    finally:
        store.close()


def test_v5_migration_is_idempotent(tmp_path: Path) -> None:
    db = str(tmp_path / "v4idem.db")
    conn = _v4_fixture_db(db)
    conn.commit()
    conn.close()

    store = Store.open(db)
    try:
        assert migrations.apply(store) == 5
        with store.read() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM migration_history WHERE version=5"
            ).fetchone()[0]
            assert n == 1
    finally:
        store.close()


def test_crash_after_v5_version_commit_resumes(tmp_path: Path) -> None:
    """Durable state: version=5 + history committed with the v5 tables,
    but the deferred jobs rebuild and source FTS never ran (crash between
    the migration transaction and the fenced phases). Open must resume
    them by schema introspection."""
    db = str(tmp_path / "v5crash.db")
    conn = _v4_fixture_db(db)
    _populate_v4(conn)
    # The migration transaction's committed output:
    for stmt in v5_statements():
        conn.execute(stmt)
    conn.execute(
        "UPDATE meta SET value_json = ? WHERE key = 'schema_version'",
        (json_dumps(5),),
    )
    conn.execute(
        "INSERT INTO migration_history(version, applied_us,"
        " migration_digest) VALUES (5, ?, ?)",
        (now_us(), V5_MIGRATION.digest()),
    )
    conn.commit()
    conn.close()

    store = Store.open(db)
    try:
        assert store.schema_version == 5
        assert store.source_fts_enabled is True
        with store.read() as conn:
            jobs_sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table'"
                " AND name='jobs'"
            ).fetchone()[0]
            assert "source_project" in jobs_sql
            assert conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table'"
                " AND name='source_fts_idx'"
            ).fetchone() is not None
            # preserved through the rebuild
            assert conn.execute(
                "SELECT kind FROM jobs WHERE job_id='j-old'"
            ).fetchone()[0] == "embed"
            ops = {
                r[0]: r[1]
                for r in conn.execute(
                    "SELECT operation_id, state FROM schema_operations"
                )
            }
            assert ops["schema-phase:v5-jobs-rebuild"] == "applied"
            assert ops["schema-phase:source-fts-ensure"] == "applied"
        assert migrations.apply(store) == 5
    finally:
        store.close()


def test_tampered_v5_history_digest_blocks_open(tmp_path: Path) -> None:
    """V4-41.03 extended to v5: a changed v5 migration checksum is
    unverifiable history and blocks the next writable open."""
    db = str(tmp_path / "v4tamper.db")
    conn = _v4_fixture_db(db)
    conn.commit()
    conn.close()
    Store.open(db).close()

    raw = sqlite3.connect(db)
    raw.execute(
        "UPDATE migration_history SET migration_digest='tampered'"
        " WHERE version = 5"
    )
    raw.commit()
    raw.close()

    with pytest.raises(VerbatimError) as exc:
        Store.open(db)
    assert exc.value.code == ErrorCode.STORE_CORRUPT
    assert "checksum mismatch" in exc.value.message
    assert "version 5" in exc.value.message


def test_tampered_v5_creation_digest_blocks_open(tmp_path: Path) -> None:
    """The same verification covers fresh-created v5 stores."""
    db = str(tmp_path / "v5tamper.db")
    Store.create(db).close()
    raw = sqlite3.connect(db)
    raw.execute(
        "UPDATE migration_history SET migration_digest='tampered'"
        " WHERE version = 5"
    )
    raw.commit()
    raw.close()
    with pytest.raises(VerbatimError) as exc:
        Store.open(db)
    assert exc.value.code == ErrorCode.STORE_CORRUPT


def test_newer_schema_refused(tmp_path: Path) -> None:
    """A store newer than this binary supports fails explicitly —
    never a downgrade attempt (SPEC §21, V4-41.08)."""
    db = str(tmp_path / "v6.db")
    conn = _fixture_db(db, version=SCHEMA_VERSION + 1)
    conn.commit()
    conn.close()
    with pytest.raises(VerbatimError) as exc:
        Store.open(db)
    assert exc.value.code == ErrorCode.SCHEMA_UNSUPPORTED


def test_v4_code_refuses_v5_store(tmp_path: Path) -> None:
    """The symmetric guard: a v5 database opened under a binary whose
    supported version is 4 fails SCHEMA_UNSUPPORTED. Simulated honestly
    in a subprocess by binding the v4-era constant — the open path's
    guard is exactly ``recorded > SCHEMA_VERSION``, so pinning the
    constant reproduces the v4 binary's behavior byte-for-byte."""
    db = str(tmp_path / "v5.db")
    Store.create(db).close()

    repo = str(Path(__file__).resolve().parents[2])
    script = f"""
import sys
sys.path.insert(0, {repo!r})
import verbatim.storage.store as st
import verbatim.storage.migrations as mg
st.SCHEMA_VERSION = 4
mg.SCHEMA_VERSION = 4
from verbatim.core.types import ErrorCode, VerbatimError
try:
    st.Store.open({db!r})
except VerbatimError as e:
    sys.exit(0 if e.code == ErrorCode.SCHEMA_UNSUPPORTED else 3)
sys.exit(2)
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr.decode()


def test_readonly_open_does_not_migrate_v4(tmp_path: Path) -> None:
    """A read-only open of a v4 store reports version 4 and writes
    nothing — parity with the existing readonly contract."""
    db = str(tmp_path / "v4ro.db")
    conn = _v4_fixture_db(db)
    conn.commit()
    conn.close()

    store = Store.open(db, readonly=True)
    try:
        assert store.schema_version == 4
    finally:
        store.close()
    raw = sqlite3.connect(db)
    try:
        row = raw.execute(
            "SELECT value_json FROM meta WHERE key='schema_version'"
        ).fetchone()
        assert "4" in row[0]
        names = _table_names(raw)
        assert "source_state" not in names
    finally:
        raw.close()
