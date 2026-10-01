"""V5 schema tests (docs/v5_contracts.md §3, SPEC_V5 §07/§14.3/§30).

Covers fresh v5 store creation: the eight contract tables plus the
source-FTS carrier pair land with the exact contract columns and primary
keys, the capability-gated FTS5 shadow mirrors through triggers, the
frozen-domain CHECKs reject out-of-contract values, the jobs CHECK
accepts the v5 source-pipeline kinds, and ``repos_v5`` performs
allowlist CRUD on every new table. Real on-disk ``Store`` instances
throughout (contracts §13.6).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.storage import migrations, repos_v5
from verbatim.storage.schema import SCHEMA_VERSION
from verbatim.storage.store import Store, _creation_ddl_v5

# The eight contract tables of docs/v5_contracts.md §3 plus the FTS
# carrier pair the source lexical shadow requires (source_fts_idx is a
# virtual table, probed separately).
V5_TABLES = (
    "source_state",
    "source_lexical_projection",
    "source_fts_rows",
    "source_fts",
    "source_vectors",
    "entity_postings",
    "duplicate_links",
    "enrichment",
    "update_candidates",
    "backfill_cursor",
)

# Exact contract columns → PK columns, per §3.
V5_COLUMNS = {
    "source_state": (
        {
            "source_id", "namespace", "control_version", "mutation_head",
            "disposition", "superseded_by", "effective_at", "known_at",
            "valid_from", "valid_to", "updated_at", "producer",
        },
        {"source_id"},
    ),
    "source_lexical_projection": (
        {
            "source_id", "revision", "scope_id", "generation",
            "tokens", "doc_len", "digest",
        },
        {"source_id", "revision"},
    ),
    "source_vectors": (
        {
            "source_id", "revision", "namespace", "encoder",
            "generation", "vector", "digest",
        },
        {"source_id", "revision", "encoder"},
    ),
    "entity_postings": (
        {
            "namespace", "entity", "entity_kind", "source_id",
            "revision", "offsets", "generation",
        },
        {"namespace", "entity", "source_id", "revision"},
    ),
    "duplicate_links": (
        {
            "source_id", "revision", "group_id", "method",
            "score", "created_at",
        },
        {"source_id", "revision", "method"},
    ),
    "enrichment": (
        {
            "source_id", "revision", "producer", "type", "polarity",
            "time_precision", "time_status", "event_at", "anchor_at",
            "fields_json",
        },
        {"source_id", "revision", "producer"},
    ),
    "update_candidates": (
        {
            "candidate_id", "namespace", "new_source_id", "new_revision",
            "prior_source_id", "prior_revision", "relation", "score",
            "state", "created_at",
        },
        {"candidate_id"},
    ),
    "backfill_cursor": (
        {"job_key", "last_source_id", "generation", "done", "updated_at"},
        {"job_key"},
    ),
}


def _table_info(conn: sqlite3.Connection, table: str):
    cols = {
        r[1] for r in conn.execute(f"PRAGMA table_info({table})")
    }
    pk = {
        r[1]
        for r in conn.execute(f"PRAGMA table_info({table})")
        if r[5]
    }
    return cols, pk


def test_fresh_store_is_v5(store: Store) -> None:
    """Store.create lands at SCHEMA_VERSION=5 with every contract table,
    the FTS5 shadow, and the mirror triggers."""
    assert SCHEMA_VERSION == 5
    assert store.schema_version == 5
    with store.read() as conn:
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for t in V5_TABLES:
            assert t in names, f"missing v5 table {t}"
        triggers = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            )
        }
    if store.source_fts_enabled:
        assert "source_fts_idx" in names
        assert {"source_fts_ai", "source_fts_ad", "source_fts_au"} <= triggers


def test_v5_table_shapes_match_contract(store: Store) -> None:
    """Every §3 table carries exactly the frozen columns and PK."""
    with store.read() as conn:
        for table, (cols, pk) in V5_COLUMNS.items():
            actual_cols, actual_pk = _table_info(conn, table)
            assert actual_cols == cols, f"{table} columns drifted"
            assert actual_pk == pk, f"{table} primary key drifted"


def test_v5_tables_have_no_evidence_plane_foreign_keys(store: Store) -> None:
    """Derived projections must not pin purge order: no FK from any v5
    projection table into the evidence plane (contracts §3 — closure
    removes them, rebuilds regenerate them)."""
    with store.read() as conn:
        for table in V5_COLUMNS:
            fks = conn.execute(
                f"PRAGMA foreign_key_list({table})"
            ).fetchall()
            assert fks == [], f"{table} carries foreign keys"


def test_source_fts_mirrors_through_triggers(store: Store) -> None:
    """INSERT/DELETE on the content table keeps the FTS5 shadow exact —
    the reprojection discipline the schema docstring mandates."""
    if not store.source_fts_enabled:
        pytest.skip("sqlite build lacks FTS5")
    with store.tx() as conn:
        repos_v5.insert(conn, "source_fts_rows", {
            "row_id": 1,
            "source_id": "src-1",
            "revision": 1,
            "scope_id": "s1",
            "generation": 1,
        })
        repos_v5.insert(conn, "source_fts", {
            "fts_row_id": 1,
            "text": "alpha beta gamma",
        })
    with store.read() as conn:
        hits = conn.execute(
            "SELECT rowid FROM source_fts_idx"
            " WHERE source_fts_idx MATCH 'gamma'"
        ).fetchall()
        assert [r[0] for r in hits] == [1]
    with store.tx() as conn:
        repos_v5.delete(conn, "source_fts", {"fts_row_id": 1})
    with store.read() as conn:
        hits = conn.execute(
            "SELECT rowid FROM source_fts_idx"
            " WHERE source_fts_idx MATCH 'gamma'"
        ).fetchall()
        assert hits == []


def test_source_fts_rowid_carrier_uniqueness(store: Store) -> None:
    """One indexed row per (source, revision, generation) — the same
    invariant fts_rows carries for claims."""
    with store.tx() as conn:
        row = {
            "source_id": "src-1", "revision": 1,
            "scope_id": "s1", "generation": 1,
        }
        repos_v5.insert(conn, "source_fts_rows", dict(row))
        with pytest.raises(sqlite3.IntegrityError):
            repos_v5.insert(conn, "source_fts_rows", dict(row))


def test_jobs_accepts_v5_source_kinds(store: Store) -> None:
    """V5-08.16: the jobs CHECK was widened for the source pipeline."""
    with store.tx() as conn:
        for i, kind in enumerate(
            ("source_project", "source_embed", "source_backfill")
        ):
            conn.execute(
                "INSERT INTO jobs (job_id, scope_id, kind, state)"
                " VALUES (?, 's1', ?, 'queued')",
                (f"jv5-{i}", kind),
            )
    with store.read() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE kind LIKE 'source_%'"
        ).fetchone()[0]
        assert n == 3


def test_v5_check_domains_enforced(store: Store) -> None:
    """Frozen contract domains are CHECK-enforced: duplicate_links.method,
    update_candidates.relation and .state."""
    with store.tx() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO duplicate_links (source_id, revision,"
                " group_id, method, score, created_at) VALUES"
                " ('s', 1, 'g', 'fuzzy_magic', 0.5, '2026-01-01T00:00:00Z')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO update_candidates (candidate_id, namespace,"
                " new_source_id, new_revision, prior_source_id,"
                " prior_revision, relation, score, state, created_at)"
                " VALUES ('c', 'ns', 'n', 1, 'p', 1, 'maybe_related',"
                " 0.5, 'open', '2026-01-01T00:00:00Z')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO update_candidates (candidate_id, namespace,"
                " new_source_id, new_revision, prior_source_id,"
                " prior_revision, relation, score, state, created_at)"
                " VALUES ('c', 'ns', 'n', 1, 'p', 1, 'refines',"
                " 0.5, 'pending_forever', '2026-01-01T00:00:00Z')"
            )


def test_fresh_v5_creation_digest_verifies(store: Store) -> None:
    """The recorded creation digest is the immutable v5 creation
    material's hash — apply() verifies it and stays a no-op."""
    import hashlib

    expected = hashlib.sha256(_creation_ddl_v5().encode("utf-8")).hexdigest()
    with store.read() as conn:
        rows = conn.execute(
            "SELECT version, migration_digest FROM migration_history"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 5
    assert rows[0][1] == expected
    assert rows[0][1] in migrations.acceptable_history_digests(5)
    assert migrations.apply(store) == 5


def test_check_integrity_reports_v5(store: Store) -> None:
    out = store.check_integrity()
    assert out["schema_version"] == 5
    assert out["source_fts5"] is store.source_fts_enabled
    for t in V5_TABLES:
        assert out["counts"][t] == 0


# ----------------------------------------------------------------------
# repos_v5 allowlist CRUD
# ----------------------------------------------------------------------


def _seed_projection(conn: sqlite3.Connection) -> None:
    """One row per table, written through the allowlist path."""
    repos_v5.insert(conn, "source_state", {
        "source_id": "src-1",
        "namespace": "ns",
        "control_version": 1,
        "mutation_head": "rev-1",
        "disposition": "active",
        "superseded_by": None,
        "effective_at": None,
        "known_at": "2026-01-01T00:00:00Z",
        "valid_from": None,
        "valid_to": None,
        "updated_at": "2026-01-01T00:00:00Z",
        "producer": "test/v1",
    })
    repos_v5.insert(conn, "source_lexical_projection", {
        "source_id": "src-1",
        "revision": 1,
        "scope_id": "s1",
        "generation": 1,
        "tokens": "alpha beta",
        "doc_len": 2,
        "digest": "d1",
    })
    repos_v5.insert(conn, "source_vectors", {
        "source_id": "src-1",
        "revision": 1,
        "namespace": "ns",
        "encoder": "hashing:subword-ngram:v1",
        "generation": 1,
        "vector": b"\x00" * 16,
        "digest": "vd1",
    })
    repos_v5.insert(conn, "entity_postings", {
        "namespace": "ns",
        "entity": "Alice",
        "entity_kind": "person",
        "source_id": "src-1",
        "revision": 1,
        "offsets": "[0, 5]",
        "generation": 1,
    })
    repos_v5.insert(conn, "duplicate_links", {
        "source_id": "src-1",
        "revision": 1,
        "group_id": "src-0",
        "method": "exact_digest",
        "score": 1.0,
        "created_at": "2026-01-01T00:00:00Z",
    })
    repos_v5.insert(conn, "enrichment", {
        "source_id": "src-1",
        "revision": 1,
        "producer": "enrich/v1",
        "type": "fact",
        "polarity": "affirmative",
        "time_precision": "day",
        "time_status": "explicit",
        "event_at": "2026-01-01",
        "anchor_at": "2026-01-01",
        "fields_json": {"identifiers": [{"kind": "path", "value": "/a/b"}]},
    })
    repos_v5.insert(conn, "update_candidates", {
        "candidate_id": "cand-1",
        "namespace": "ns",
        "new_source_id": "src-1",
        "new_revision": 1,
        "prior_source_id": "src-0",
        "prior_revision": 3,
        "relation": "newer_value",
        "score": 0.9,
        "state": "open",
        "created_at": "2026-01-01T00:00:00Z",
    })
    repos_v5.insert(conn, "backfill_cursor", {
        "job_key": "source_backfill:ns",
        "last_source_id": "src-1",
        "generation": 1,
        "done": 0,
        "updated_at": "2026-01-01T00:00:00Z",
    })


def test_repos_v5_insert_get_roundtrip(store: Store) -> None:
    with store.tx() as conn:
        _seed_projection(conn)
    with store.read() as conn:
        state = repos_v5.get(
            conn, "source_state", {"source_id": "src-1"}
        )
        assert state["disposition"] == "active"
        assert state["control_version"] == 1

        lex = repos_v5.get(
            conn, "source_lexical_projection",
            {"source_id": "src-1", "revision": 1},
        )
        assert lex["tokens"] == "alpha beta"

        vec = repos_v5.get(
            conn, "source_vectors",
            {"source_id": "src-1", "revision": 1,
             "encoder": "hashing:subword-ngram:v1"},
        )
        assert vec["vector"] == b"\x00" * 16

        posts = repos_v5.query(
            conn, "entity_postings", {"namespace": "ns", "entity": "Alice"}
        )
        assert len(posts) == 1 and posts[0]["entity_kind"] == "person"

        link = repos_v5.get(
            conn, "duplicate_links",
            {"source_id": "src-1", "revision": 1, "method": "exact_digest"},
        )
        assert link["group_id"] == "src-0" and link["score"] == 1.0

        enr = repos_v5.get(
            conn, "enrichment",
            {"source_id": "src-1", "revision": 1, "producer": "enrich/v1"},
        )
        assert repos_v5.json_field(enr, "fields_json") == {
            "identifiers": [{"kind": "path", "value": "/a/b"}]
        }

        cand = repos_v5.get(
            conn, "update_candidates", {"candidate_id": "cand-1"}
        )
        assert cand["relation"] == "newer_value" and cand["state"] == "open"

        cur = repos_v5.get(
            conn, "backfill_cursor", {"job_key": "source_backfill:ns"}
        )
        assert cur["done"] == 0 and cur["last_source_id"] == "src-1"


def test_repos_v5_update_and_cas(store: Store) -> None:
    """control_version is the CAS target: a stale-version update matches
    zero rows instead of silently overwriting (§14.3)."""
    with store.tx() as conn:
        _seed_projection(conn)
    with store.tx() as conn:
        n = repos_v5.update(
            conn, "source_state",
            {"control_version": 2, "disposition": "superseded",
             "superseded_by": "src-2"},
            {"source_id": "src-1", "control_version": 1},
        )
        assert n == 1
    with store.tx() as conn:
        # The version-1 holder's retry loses — the row is now version 2.
        n = repos_v5.update(
            conn, "source_state",
            {"control_version": 3},
            {"source_id": "src-1", "control_version": 1},
        )
        assert n == 0
    with store.read() as conn:
        state = repos_v5.get(
            conn, "source_state", {"source_id": "src-1"}
        )
        assert state["control_version"] == 2
        assert state["disposition"] == "superseded"


def test_repos_v5_query_order_limit(store: Store) -> None:
    with store.tx() as conn:
        _seed_projection(conn)
        for i in (2, 3):
            repos_v5.insert(conn, "entity_postings", {
                "namespace": "ns",
                "entity": "Alice",
                "entity_kind": "person",
                "source_id": f"src-{i}",
                "revision": 1,
                "offsets": "[]",
                "generation": 1,
            })
    with store.read() as conn:
        rows = repos_v5.query(
            conn, "entity_postings",
            {"namespace": "ns", "entity": "Alice"},
            order="source_id DESC", limit=2,
        )
        assert [r["source_id"] for r in rows] == ["src-3", "src-2"]


def test_repos_v5_delete_roundtrip(store: Store) -> None:
    with store.tx() as conn:
        _seed_projection(conn)
    with store.tx() as conn:
        n = repos_v5.delete(
            conn, "duplicate_links",
            {"source_id": "src-1", "revision": 1, "method": "exact_digest"},
        )
        assert n == 1
        n = repos_v5.delete(
            conn, "duplicate_links", {"source_id": "src-1"}
        )
        assert n == 0  # already gone; predicate still required
    with store.read() as conn:
        assert repos_v5.get(
            conn, "duplicate_links", {"source_id": "src-1"}
        ) is None


def test_repos_v5_upsert_replaces_projection(store: Store) -> None:
    """Reprojection of (source_id, revision) replaces the row — the
    generation column records which index generation produced it."""
    with store.tx() as conn:
        _seed_projection(conn)
        repos_v5.upsert(conn, "source_lexical_projection", {
            "source_id": "src-1",
            "revision": 1,
            "scope_id": "s1",
            "generation": 2,
            "tokens": "alpha beta delta",
            "doc_len": 3,
            "digest": "d2",
        })
    with store.read() as conn:
        rows = repos_v5.query(
            conn, "source_lexical_projection", {"source_id": "src-1"}
        )
        assert len(rows) == 1
        assert rows[0]["generation"] == 2
        assert rows[0]["digest"] == "d2"


def test_repos_v5_rejects_unknown_table(store: Store) -> None:
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            repos_v5.insert(conn, "source_fts_idx", {"text": "x"})
        assert exc.value.code == ErrorCode.VALIDATION
        with pytest.raises(VerbatimError) as exc:
            repos_v5.query(conn, "claims", {})
        assert exc.value.code == ErrorCode.VALIDATION


def test_repos_v5_rejects_unknown_columns(store: Store) -> None:
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            repos_v5.insert(conn, "source_state", {
                "source_id": "x", "namespace": "ns",
                "control_version": 1, "mutation_head": "r",
                "disposition": "active", "known_at": "t",
                "updated_at": "t", "producer": "p",
                "raw_sql": "DROP TABLE sources",
            })
        assert exc.value.code == ErrorCode.VALIDATION
        with pytest.raises(VerbatimError) as exc:
            repos_v5.query(
                conn, "source_state", {"1=1; DROP TABLE sources": "x"}
            )
        assert exc.value.code == ErrorCode.VALIDATION


def test_repos_v5_delete_requires_predicate(store: Store) -> None:
    with store.tx() as conn:
        _seed_projection(conn)
        with pytest.raises(VerbatimError) as exc:
            repos_v5.delete(conn, "enrichment", {})
        assert exc.value.code == ErrorCode.VALIDATION
