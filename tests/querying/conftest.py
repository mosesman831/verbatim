"""Fixtures for the querying worker: real on-disk ``Store`` plus the
contracts-§3 v5 tables (source_state / enrichment / update_candidates).

``schema_v5.py`` has landed — ``Store.create`` already yields these
tables — so the DDL below is an idempotent safety net matching the
frozen contract exactly, harmless when the real schema is present and
sufficient when it is not.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional

import pytest

from verbatim.core.time import now_us, rfc3339
from verbatim.storage.store import Store

# contracts §3 DDL (verbatim, frozen) ---------------------------------

V5_CONTRACT_DDL = """
CREATE TABLE IF NOT EXISTS source_state (
    source_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    control_version INTEGER NOT NULL,
    mutation_head TEXT NOT NULL,
    disposition TEXT NOT NULL,
    superseded_by TEXT,
    effective_at TEXT,
    known_at TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    updated_at TEXT NOT NULL,
    producer TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS enrichment (
    source_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    producer TEXT NOT NULL,
    type TEXT,
    polarity TEXT,
    time_precision TEXT,
    time_status TEXT,
    event_at TEXT,
    anchor_at TEXT,
    fields_json TEXT NOT NULL,
    PRIMARY KEY (source_id, revision, producer)
);

CREATE TABLE IF NOT EXISTS update_candidates (
    candidate_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    new_source_id TEXT NOT NULL,
    new_revision INTEGER NOT NULL,
    prior_source_id TEXT NOT NULL,
    prior_revision INTEGER NOT NULL,
    relation TEXT NOT NULL,
    score REAL NOT NULL,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


@pytest.fixture
def v5store(tmp_path):
    """Real ``Store.create`` + the v5 contract tables applied inside a
    genuine write transaction (executescript would implicitly COMMIT, so
    statements run one at a time inside tx())."""
    store = Store.create(str(tmp_path / "querying.db"))
    with store.tx() as conn:
        for stmt in V5_CONTRACT_DDL.split(";"):
            if stmt.strip():
                conn.execute(stmt)
    yield store
    store.close()


def seed_source(
    conn: sqlite3.Connection,
    namespace: str,
    source_id: str,
    text: str,
    *,
    revision: int = 1,
    disposition: str = "active",
    control_version: int = 1,
    enrichment: Optional[dict] = None,
) -> None:
    """Seed a retained source revision + its live source_state binding.

    ``enrichment`` optionally writes the contracts-§3 enrichment row —
    keys: type, polarity, event_at, anchor_at, fields (identifiers /
    entities lists serialized into fields_json).
    """
    now = rfc3339(now_us())
    conn.execute(
        "INSERT OR IGNORE INTO scopes"
        " (scope_id, profile_id, principal_id, visibility)"
        " VALUES (?, 'p', 'alice', 'owner')",
        (namespace,),
    )
    conn.execute(
        "INSERT OR IGNORE INTO sources"
        " (source_id, origin, external_id, source_kind, scope_id,"
        "  speaker_id, created_us)"
        " VALUES (?, 'test', NULL, 'user_message', ?, NULL, 0)",
        (source_id, namespace),
    )
    payload = text.encode("utf-8")
    conn.execute(
        "INSERT OR REPLACE INTO source_revisions"
        " (source_id, revision, payload, payload_hmac, event_us,"
        "  captured_us, timezone, provenance, metadata_json)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (
            source_id,
            revision,
            payload,
            b"\x00" * 32,
            0,
            0,
            None,
            "direct_user",
            "{}",
        ),
    )
    conn.execute(
        "INSERT OR REPLACE INTO source_state"
        " (source_id, namespace, control_version, mutation_head,"
        "  disposition, superseded_by, effective_at, known_at,"
        "  valid_from, valid_to, updated_at, producer)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            source_id,
            namespace,
            control_version,
            str(revision),
            disposition,
            None,
            None,
            now,
            None,
            None,
            now,
            "test",
        ),
    )
    if enrichment is not None:
        conn.execute(
            "INSERT OR REPLACE INTO enrichment"
            " (source_id, revision, producer, type, polarity,"
            "  time_precision, time_status, event_at, anchor_at,"
            "  fields_json)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                source_id,
                revision,
                enrichment.get("producer", "enrich/v1"),
                enrichment.get("type"),
                enrichment.get("polarity"),
                enrichment.get("time_precision"),
                enrichment.get("time_status"),
                enrichment.get("event_at"),
                enrichment.get("anchor_at"),
                json.dumps(enrichment.get("fields", {})),
            ),
        )


def candidates(conn: sqlite3.Connection, namespace: str) -> list[dict]:
    cur = conn.execute(
        "SELECT candidate_id, namespace, new_source_id, new_revision,"
        " prior_source_id, prior_revision, relation, score, state,"
        " created_at FROM update_candidates WHERE namespace = ?"
        " ORDER BY score DESC, candidate_id",
        (namespace,),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]
