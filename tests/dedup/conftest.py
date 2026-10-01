"""Fixtures for dedup tests: real on-disk Store plus the frozen §3 v5
contract tables (source_state / source_lexical_projection / enrichment /
duplicate_links).

``schema_v5.py`` is owned by the storage worker and was absent at
authoring time; the DDL below is copied verbatim from
docs/v5_contracts.md §3 and uses CREATE TABLE IF NOT EXISTS so it stays
compatible when the real schema lands. Namespace membership is seeded
through ``source_state`` (the authoritative v5 map); the sources row's
``scope_id`` is set to the same namespace token so the pre-v5 scope
fallback path is exercised identically.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verbatim.core.time import rfc3339
from verbatim.core.types import json_dumps
from verbatim.dedup import links
from verbatim.storage.store import Store

# Contract §3 DDL (verbatim copy; IF NOT EXISTS added only where the
# contract text omitted it — the shipped DDL must converge).
V5_DDL = """
CREATE TABLE IF NOT EXISTS source_state(
  source_id TEXT PRIMARY KEY,
  namespace TEXT NOT NULL,
  control_version INTEGER NOT NULL,
  mutation_head TEXT NOT NULL,
  disposition TEXT NOT NULL,
  superseded_by TEXT, effective_at TEXT,
  known_at TEXT NOT NULL, valid_from TEXT, valid_to TEXT,
  updated_at TEXT NOT NULL, producer TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS source_lexical_projection(
  source_id TEXT NOT NULL, revision INTEGER NOT NULL,
  scope_id TEXT NOT NULL, generation INTEGER NOT NULL,
  tokens TEXT NOT NULL,
  doc_len INTEGER NOT NULL, digest TEXT NOT NULL,
  PRIMARY KEY (source_id, revision));

CREATE TABLE IF NOT EXISTS entity_postings(
  namespace TEXT NOT NULL, entity TEXT NOT NULL, entity_kind TEXT NOT NULL,
  source_id TEXT NOT NULL, revision INTEGER NOT NULL, offsets TEXT NOT NULL,
  generation INTEGER NOT NULL,
  PRIMARY KEY (namespace, entity, source_id, revision));

CREATE TABLE IF NOT EXISTS duplicate_links(
  source_id TEXT NOT NULL, revision INTEGER NOT NULL, group_id TEXT NOT NULL,
  method TEXT NOT NULL,
  score REAL NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY (source_id, revision, method));

CREATE TABLE IF NOT EXISTS enrichment(
  source_id TEXT NOT NULL, revision INTEGER NOT NULL,
  producer TEXT NOT NULL,
  type TEXT, polarity TEXT,
  time_precision TEXT, time_status TEXT, event_at TEXT, anchor_at TEXT,
  fields_json TEXT NOT NULL,
  PRIMARY KEY (source_id, revision, producer));

CREATE TABLE IF NOT EXISTS update_candidates(
  candidate_id TEXT PRIMARY KEY, namespace TEXT NOT NULL,
  new_source_id TEXT NOT NULL, new_revision INTEGER NOT NULL,
  prior_source_id TEXT NOT NULL, prior_revision INTEGER NOT NULL,
  relation TEXT NOT NULL,
  score REAL NOT NULL, state TEXT NOT NULL,
  created_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS backfill_cursor(
  job_key TEXT PRIMARY KEY, last_source_id TEXT NOT NULL,
  generation INTEGER NOT NULL, done INTEGER NOT NULL, updated_at TEXT NOT NULL);
"""


@pytest.fixture()
def store(tmp_path):
    # Real on-disk store: Store.create builds the schema_v5 contract
    # tables (source_state / source_lexical_projection / enrichment /
    # duplicate_links) — V5_DDL below is kept as documentation of the
    # frozen §3 shape these tests were written against.
    s = Store.create(str(tmp_path / "v.db"))
    yield s
    s.close()


def norm_tokens(text: str) -> str:
    """The normalized token stream a source_project job would persist."""
    return links._norm(text)


def norm_digest(text: str) -> str:
    """The digest a projection row would carry for normalized matching —
    the real ``enrichment.normalized_digest`` when the worker module is
    importable, else a deterministic local stand-in."""
    try:
        from verbatim.enrichment import normalized_digest
        return normalized_digest(text)
    except ImportError:  # pragma: no cover - parallel build window
        return hashlib.blake2b(
            links._norm(text).encode("utf-8"), digest_size=32
        ).hexdigest()


def signature_of(text: str) -> frozenset:
    return links._signature(links._norm(text).split())


def seed_source(conn, store: Store, *, source_id: str, text: str,
                namespace: str, revision: int = 1,
                origin: str = "test-harness", speaker: str = "alice",
                provenance: str = "direct_user",
                disposition: str = "active", created_us: int,
                project: bool = True, enrich: bool = True,
                state: bool = True) -> str:
    """Seed one source + revision + v5 projection/enrichment/state rows.

    ``created_us`` orders members ("earliest live"). ``state=False``
    simulates a source not yet registered in the control artifact.
    """
    payload = text.encode("utf-8")
    conn.execute(
        "INSERT OR IGNORE INTO scopes"
        " (scope_id, profile_id, principal_id, workspace_id,"
        "  conversation_id, visibility)"
        " VALUES (?, 'prof', 'alice', 'ws1', 'conv1', 'conversation')",
        (namespace,),
    )
    conn.execute(
        "INSERT INTO sources"
        " (source_id, origin, external_id, source_kind, scope_id,"
        "  speaker_id, created_us)"
        " VALUES (?, ?, NULL, 'user_message', ?, ?, ?)",
        (source_id, origin, namespace, speaker, created_us),
    )
    conn.execute(
        "INSERT INTO source_revisions"
        " (source_id, revision, payload, payload_hmac, event_us,"
        "  captured_us, timezone, provenance, metadata_json)"
        " VALUES (?, ?, ?, ?, ?, ?, 'UTC', ?, '{}')",
        (source_id, revision, payload, store.hmac(payload),
         created_us, created_us, provenance),
    )
    if state:
        now = rfc3339(created_us)
        conn.execute(
            "INSERT INTO source_state"
            " (source_id, namespace, control_version, mutation_head,"
            "  disposition, known_at, updated_at, producer)"
            " VALUES (?, ?, 1, ?, ?, ?, ?, 'source_state/v1')",
            (source_id, namespace, str(revision), disposition, now, now),
        )
    if project:
        tokens = norm_tokens(text)
        conn.execute(
            "INSERT INTO source_lexical_projection"
            " (source_id, revision, scope_id, generation, tokens,"
            "  doc_len, digest)"
            " VALUES (?, ?, ?, 1, ?, ?, ?)",
            (source_id, revision, namespace, tokens,
             len(tokens.split()), norm_digest(text)),
        )
    if enrich:
        f = links.fields_from_text(text)
        conn.execute(
            "INSERT INTO enrichment"
            " (source_id, revision, producer, type, polarity,"
            "  fields_json)"
            " VALUES (?, ?, 'enrich/v1', ?, ?, ?)",
            (source_id, revision, f.memory_type, f.polarity,
             json_dumps({
                 "identifiers": sorted(f.identifiers),
                 "number_tokens": sorted(f.number_tokens),
                 "time_expressions": sorted(f.time_expressions),
             })),
        )
    return source_id


def add_revision(conn, store: Store, *, source_id: str, text: str,
                 namespace: str, revision: int, created_us: int,
                 provenance: str = "direct_user") -> None:
    """Append a further revision to an existing source (revision chains
    are the source's own history — never duplicate-group members)."""
    payload = text.encode("utf-8")
    conn.execute(
        "INSERT INTO source_revisions"
        " (source_id, revision, payload, payload_hmac, event_us,"
        "  captured_us, timezone, provenance, metadata_json)"
        " VALUES (?, ?, ?, ?, ?, ?, 'UTC', ?, '{}')",
        (source_id, revision, payload, store.hmac(payload),
         created_us, created_us, provenance),
    )
    tokens = norm_tokens(text)
    conn.execute(
        "INSERT INTO source_lexical_projection"
        " (source_id, revision, scope_id, generation, tokens,"
        "  doc_len, digest)"
        " VALUES (?, ?, ?, 1, ?, ?, ?)",
        (source_id, revision, namespace, tokens,
         len(tokens.split()), norm_digest(text)),
    )
    f = links.fields_from_text(text)
    conn.execute(
        "INSERT INTO enrichment"
        " (source_id, revision, producer, type, polarity, fields_json)"
        " VALUES (?, ?, 'enrich/v1', ?, ?, ?)",
        (source_id, revision, f.memory_type, f.polarity,
         json_dumps({
             "identifiers": sorted(f.identifiers),
             "number_tokens": sorted(f.number_tokens),
             "time_expressions": sorted(f.time_expressions),
         })),
    )


def payload_hmac_hex(store: Store, text: str) -> str:
    return store.hmac(text.encode("utf-8")).hex()


def erase_source(conn, store: Store, source_id: str) -> None:
    """Simulate the closure erasure path (verbatim/purge.py semantics):
    revision payloads emptied, submitter fields scrubbed, source_state
    disposition -> erased."""
    conn.execute(
        "UPDATE source_revisions SET payload = X'', payload_hmac = ?,"
        " metadata_json = '{}' WHERE source_id = ?",
        (store.hmac(b""), source_id),
    )
    conn.execute(
        "UPDATE sources SET external_id = NULL, speaker_id = NULL"
        " WHERE source_id = ?",
        (source_id,),
    )
    conn.execute(
        "UPDATE source_state SET disposition = 'erased',"
        " control_version = control_version + 1 WHERE source_id = ?",
        (source_id,),
    )
