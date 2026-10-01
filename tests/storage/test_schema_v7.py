"""V7 additive-schema tests (SPEC_V7 §30, V7-06.06, docs/v7_contracts.md).

Covers the lazy-additive pattern (fresh ``Store.create`` stores gain the
tables on the first writer's in-tx ensure, never eagerly — V6 semantics
carried), the §30 table/column contract, the covering indexes behind the
§04.2/§08/§09 hot paths (V7-30.03 — asserted by EXPLAIN QUERY PLAN), the
three-tokenizer FTS5 shadow wiring, and the incremental corpus statistics
(V7-06.06). Real on-disk databases throughout.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verbatim.storage.schema_v7 import (
    DDL_V7,
    DDL_V7_FTS5,
    SCHEMA_V7_TAG,
    V7_FTS_TABLES,
    V7_INDEXES,
    V7_INTERNAL_TABLES,
    V7_TABLE_RENAMES,
    V7_TABLES,
    V7_TRIGGERS,
    _TOKENIZER_PROBE_CACHE,
    ensure_v7_additive,
    fts5_tokenizer_available,
    v7_fts_present,
    v7_statements,
    v7_tables_present,
)
from verbatim.storage.stats_v7 import (
    corpus_stats,
    decrement_stats,
    term_df,
    term_dfs,
    update_stats,
)
from verbatim.storage.store import Store


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _names(conn: sqlite3.Connection, kind: str) -> set:
    return {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type=?", (kind,)
        )
    }


def _cols(conn: sqlite3.Connection, table: str) -> set:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _plan(conn: sqlite3.Connection, sql: str, params: tuple) -> str:
    rows = conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
    return "\n".join("|".join(str(c) for c in r) for r in rows)


def _unit(unit_id: str, scope_id: str = "s1", generation: int = 1) -> dict:
    return {
        "unit_id": unit_id,
        "source_id": "src-1",
        "revision": 1,
        "scope_id": scope_id,
        "kind": "turn",
        "parent_unit_id": None,
        "session_id": "sess-1",
        "seq": 1,
        "speaker_canon": "alice",
        "perspective": "user_stated",
        "recorded_at_us": 1_700_000_000_000_000,
        "occurred_start_us": 1_683_456_000_000_000,
        "occurred_end_us": 1_683_456_000_000_000,
        "occurred_precision": "day",
        "occurred_source": "explicit",
        "byte_start": 0,
        "byte_end": 42,
        "generation": generation,
    }


def _insert_unit(conn, row: dict) -> None:
    cols = list(row)
    conn.execute(
        f"INSERT INTO units ({', '.join(cols)})"
        f" VALUES ({', '.join('?' for _ in cols)})",
        tuple(row[c] for c in cols),
    )


def _insert_fts(
    conn, unit_id, scope_id, generation, text, speaker, entities,
    session, when,
) -> int:
    cur = conn.execute(
        "INSERT INTO unit_fts_rows(unit_id, scope_id, generation)"
        " VALUES (?, ?, ?)",
        (unit_id, scope_id, generation),
    )
    conn.execute(
        "INSERT INTO unit_fts_content("
        " fts_row_id, text, speaker, entities, session, \"when\")"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (cur.lastrowid, text, speaker, entities, session, when),
    )
    return cur.lastrowid


@pytest.fixture()
def ensured(store):
    """Fresh store with the V7 additive schema ensured in one tx."""
    with store.tx() as conn:
        ensure_v7_additive(conn)
    return store


# ---------------------------------------------------------------------------
# presence / idempotence
# ---------------------------------------------------------------------------


def test_fresh_store_lacks_v7_tables_then_ensure(store):
    """Store.create never runs apply() and nothing V7 is eager: a fresh
    store reports absent, the in-tx ensure creates every §30 table plus
    wiring, and presence flips false → true."""
    with store.read() as conn:
        assert v7_tables_present(conn) is False
        names = _names(conn, "table")
        for t in V7_TABLES + V7_INTERNAL_TABLES:
            assert t not in names
    with store.tx() as conn:
        ensure_v7_additive(conn)
    with store.read() as conn:
        assert v7_tables_present(conn) is True
        names = _names(conn, "table")
        for t in V7_TABLES + V7_INTERNAL_TABLES:
            assert t in names


def test_ensure_idempotent(ensured):
    """A second ensure in its own tx is a pure no-op (IF NOT EXISTS all the
    way down — tables, indexes, virtual tables, triggers)."""
    s = ensured
    with s.tx() as conn:
        ensure_v7_additive(conn)
        ensure_v7_additive(conn)
    with s.read() as conn:
        assert v7_tables_present(conn) is True


def test_ensure_rolls_back_with_caller_tx(store):
    """The lazy ensure is transactional — a caller rollback takes the
    CREATEs with it (no half-committed schema)."""
    with pytest.raises(RuntimeError):
        with store.tx() as conn:
            ensure_v7_additive(conn)
            raise RuntimeError("caller abort")
    with store.read() as conn:
        assert v7_tables_present(conn) is False
        names = _names(conn, "table")
        for t in V7_TABLES + V7_INTERNAL_TABLES:
            assert t not in names


def test_ddl_is_statements_and_tag():
    assert SCHEMA_V7_TAG == "schema_v7/v1"
    stmts = v7_statements()
    assert len(stmts) > 0
    assert all(isinstance(s, str) and s for s in stmts)
    # The plain DDL never contains the FTS5 virtuals — capability-gated.
    for name in V7_FTS_TABLES:
        assert f"TABLE IF NOT EXISTS {name} " not in DDL_V7
        assert name in DDL_V7_FTS5


def test_collision_renames_documented():
    """The two §30 names that collide with live v1 tables are renamed —
    events → events_v7, entity_aliases → entity_aliases_v7."""
    assert V7_TABLE_RENAMES == {
        "events": "events_v7",
        "entity_aliases": "entity_aliases_v7",
    }
    assert "events_v7" in V7_TABLES and "entity_aliases_v7" in V7_TABLES
    assert "events" not in V7_TABLES and "entity_aliases" not in V7_TABLES


def test_existing_events_table_untouched(ensured):
    """The v1 audit-journal ``events`` keeps its own shape — the V7 event
    calendar never aliases onto it."""
    s = ensured
    with s.read() as conn:
        assert {
            "event_seq", "event_id", "scope_id", "kind", "actor_id",
            "recorded_us", "observed_wall_us", "policy_version",
            "payload_json",
        } <= _cols(conn, "events")
        assert {
            "event_id", "unit_id", "scope_id", "subject_canon",
            "predicate_lemma", "object_text", "polarity",
            "occurred_start_us", "occurred_end_us", "precision",
            "pins_json", "rule_id", "generation",
        } <= _cols(conn, "events_v7")
        # v1 entity_aliases (entity_id, normalized_alias, …) also intact.
        assert "entity_id" in _cols(conn, "entity_aliases")
        assert "normalized_alias" in _cols(conn, "entity_aliases")


# ---------------------------------------------------------------------------
# §30 columns (normative minimums) + generation on every table
# ---------------------------------------------------------------------------


def test_units_columns(ensured):
    with ensured.read() as conn:
        assert {
            "unit_id", "source_id", "revision", "scope_id", "kind",
            "parent_unit_id", "session_id", "seq", "speaker_canon",
            "perspective", "recorded_at_us", "occurred_start_us",
            "occurred_end_us", "occurred_precision", "occurred_source",
            "byte_start", "byte_end", "generation",
        } <= _cols(conn, "units")
        pk = [r[1] for r in conn.execute("PRAGMA table_info(units)")
              if r[5] > 0]
        # §30 key (unit_id) leads; generation appended for rebuild
        # coexistence under the fence (V7-30.02).
        assert pk == ["unit_id", "generation"]


def test_stat_tables_columns(ensured):
    with ensured.read() as conn:
        assert {
            "scope_id", "generation", "field", "stats_version",
            "n_units", "total_len",
        } <= _cols(conn, "lex_stats")
        assert {
            "scope_id", "generation", "field", "term", "stats_version",
            "df",
        } <= _cols(conn, "lex_df")


def test_vector_block_columns(ensured):
    with ensured.read() as conn:
        assert {
            "encoder_id", "scope_id", "generation", "block_no",
            "n_rows", "dims", "quant", "scale_blob", "data_blob",
            "rowmap_blob",
        } <= _cols(conn, "unit_vectors_block")


def test_entity_tables_columns(ensured):
    with ensured.read() as conn:
        assert {
            "scope_id", "canon", "display", "df_units", "kind",
            "first_seen_us", "last_seen_us", "generation",
        } <= _cols(conn, "entity_canon")
        assert {
            "scope_id", "canon", "unit_id", "surface", "byte_start",
            "byte_end", "role", "generation",
        } <= _cols(conn, "entity_mentions")
        assert {
            "scope_id", "canon", "alias_canon", "rule_id",
            "evidence_count", "method", "state", "generation",
        } <= _cols(conn, "entity_aliases_v7")


def test_graph_edge_columns(ensured):
    with ensured.read() as conn:
        assert {
            "scope_id", "src_unit", "type", "dst_unit", "weight",
            "evidence_ref", "generation",
        } <= _cols(conn, "graph_edges")


def test_derived_tables_columns(ensured):
    with ensured.read() as conn:
        assert {
            "scope_id", "state_key", "unit_id", "value_text",
            "value_norm", "valid_from_us", "valid_to_us", "status",
            "producer", "pins_json", "generation",
        } <= _cols(conn, "state_facts")
        assert {
            "scope_id", "subject_canon", "unit_id", "object_text",
            "polarity", "strength", "occurred_start_us", "pins_json",
            "generation",
        } <= _cols(conn, "preferences")
        assert {
            "rule_id", "scope_id", "unit_id", "text_pin",
            "trigger_entities", "trigger_topics", "valid_until_expr",
            "status", "generation",
        } <= _cols(conn, "standing_rules")
        assert {
            "obs_id", "scope_id", "slot", "text", "producer",
            "proof_count", "support_refs_json", "contradict_refs_json",
            "first_us", "last_us", "stale", "generation",
        } <= _cols(conn, "observations_v7")
        assert {
            "scope_id", "subject_canon", "slot", "value", "status",
            "support_refs_json", "updated_us", "generation",
        } <= _cols(conn, "profiles_v7")
        assert {
            "sq_id", "scope_id", "principal", "query", "filters_json",
            "pack_blob", "pack_digest", "built_generation", "dirty",
            "built_us", "generation",
        } <= _cols(conn, "standing_queries")
        assert {
            "fact_id", "scope_id", "unit_ids_json", "statement",
            "quotes_json", "subject_canon", "predicate", "object",
            "occurred_start_us", "occurred_end_us", "state_key",
            "model_id", "prompt_digest", "verified", "generation",
        } <= _cols(conn, "t2_facts")
        assert {
            "decision_id", "unit_id", "rules_version", "outcome",
            "rule_ids", "at_us", "generation",
        } <= _cols(conn, "screening_log")
        assert {"digest", "manifest_json"} <= _cols(conn, "run_manifests")


def test_all_indexes_exist(ensured):
    with ensured.read() as conn:
        names = _names(conn, "index")
        for idx in V7_INDEXES:
            assert idx in names


# ---------------------------------------------------------------------------
# covering indexes (V7-30.03): EXPLAIN QUERY PLAN assertions
# ---------------------------------------------------------------------------


def test_covering_index_plans(ensured):
    """Every hot-path pattern resolves through its covering index — no
    SCAN over units, postings, or edges (V7-17.04)."""
    s = ensured
    with s.tx() as conn:
        _insert_unit(conn, _unit("u1"))
        conn.execute(
            "INSERT INTO entity_mentions(scope_id, canon, unit_id,"
            " generation, surface, byte_start, byte_end, role)"
            " VALUES ('s1', 'alice', 'u1', 1, 'Alice', 0, 5, 'speaker')"
        )
        conn.execute(
            "INSERT INTO entity_aliases_v7(scope_id, canon, alias_canon,"
            " generation, rule_id, evidence_count, method, state)"
            " VALUES ('s1', 'alice chen', 'alice', 1, 'A1', 3, 'rule',"
            " 'active')"
        )
        conn.execute(
            "INSERT INTO entity_canon(scope_id, canon, generation,"
            " display, df_units) VALUES ('s1', 'alice', 1, 'Alice', 4)"
        )
        conn.execute(
            "INSERT INTO graph_edges(scope_id, src_unit, type, dst_unit,"
            " generation, weight) VALUES ('s1', 'u1', 'co_mention', 'u2',"
            " 1, 0.5)"
        )
        conn.execute(
            "INSERT INTO events_v7(event_id, unit_id, scope_id,"
            " subject_canon, predicate_lemma, object_text, polarity,"
            " occurred_start_us, occurred_end_us, precision, pins_json,"
            " rule_id, generation) VALUES ('e1', 'u1', 's1', 'alice',"
            " 'moved', 'Berlin', 'affirm', 10, 20, 'day', '{}', 'T01', 1)"
        )
        conn.execute(
            "INSERT INTO state_facts(scope_id, state_key, unit_id,"
            " generation, value_text, status) VALUES ('s1',"
            " 'user/home_city', 'u1', 1, 'Berlin', 'current')"
        )
        conn.execute(
            "INSERT INTO observations_v7(obs_id, scope_id, slot, text,"
            " stale, generation) VALUES ('o1', 's1', 'alice/food',"
            " 'Alice likes sushi', 0, 1)"
        )
        conn.execute(
            "INSERT INTO unit_fts_rows(unit_id, scope_id, generation)"
            " VALUES ('u1', 's1', 1)"
        )
    with s.read() as conn:
        # fence scan → unit ids (L-* lanes' eligibility universe); several
        # scope+generation covering indexes tie here — the planner may pick
        # any of them; what matters is COVERING, never SCAN (V7-17.04).
        plan = _plan(
            conn,
            "SELECT unit_id FROM units"
            " WHERE scope_id = ? AND generation = ?",
            ("s1", 1),
        )
        assert "COVERING INDEX" in plan and "SCAN" not in plan
        # the dedicated fence index wins when unit_id ordering is wanted
        plan = _plan(
            conn,
            "SELECT unit_id FROM units"
            " WHERE scope_id = ? AND generation = ? ORDER BY unit_id",
            ("s1", 1),
        )
        assert "COVERING INDEX idx_units_scope_gen" in plan
        # neighbor expansion inside a session (V7-12.05)
        plan = _plan(
            conn,
            "SELECT unit_id FROM units"
            " WHERE scope_id = ? AND generation = ? AND session_id = ?"
            "   AND seq BETWEEN ? AND ?",
            ("s1", 1, "sess-1", 0, 2),
        )
        assert "COVERING INDEX idx_units_session" in plan
        # temporal lane: occurred-window overlap (V7-09.06)
        plan = _plan(
            conn,
            "SELECT unit_id FROM units"
            " WHERE scope_id = ? AND generation = ?"
            "   AND occurred_start_us <= ? AND occurred_end_us >= ?",
            ("s1", 1, 100, 0),
        )
        assert "COVERING INDEX idx_units_occurred" in plan
        # recorded axis, secondary temporal retrieval (V7-09.06)
        plan = _plan(
            conn,
            "SELECT unit_id FROM units"
            " WHERE scope_id = ? AND generation = ?"
            "   AND recorded_at_us BETWEEN ? AND ?",
            ("s1", 1, 0, 2**62),
        )
        assert "COVERING INDEX idx_units_recorded" in plan
        # speaker-scoped questions (V7-08.05)
        plan = _plan(
            conn,
            "SELECT unit_id FROM units"
            " WHERE scope_id = ? AND generation = ? AND speaker_canon = ?",
            ("s1", 1, "alice"),
        )
        assert "COVERING INDEX idx_units_speaker" in plan
        # L-ent postings: canon → units (V7-08.01/05); the PK is covering.
        plan = _plan(
            conn,
            "SELECT unit_id FROM entity_mentions"
            " WHERE scope_id = ? AND canon = ? AND generation = ?",
            ("s1", "alice", 1),
        )
        assert "COVERING INDEX" in plan and "SCAN" not in plan
        # alias expansion reverse direction (alias → canon, V7-08.04)
        plan = _plan(
            conn,
            "SELECT canon FROM entity_aliases_v7"
            " WHERE scope_id = ? AND alias_canon = ? AND generation = ?"
            "   AND state = 'active'",
            ("s1", "alice", 1),
        )
        assert "COVERING INDEX idx_entity_aliases_v7_alias" in plan
        # canon vocabulary enumeration per generation (V7-08.02)
        plan = _plan(
            conn,
            "SELECT canon FROM entity_canon"
            " WHERE scope_id = ? AND generation = ?",
            ("s1", 1),
        )
        assert "COVERING INDEX idx_entity_canon_scope" in plan
        # bounded PPR expansion: outgoing typed edges + weight (V7-08.10)
        plan = _plan(
            conn,
            "SELECT dst_unit, weight FROM graph_edges"
            " WHERE scope_id = ? AND src_unit = ? AND generation = ?"
            "   AND type = ?",
            ("s1", "u1", 1, "co_mention"),
        )
        assert "COVERING INDEX idx_graph_edges_src" in plan
        # event-index-first for temporal intents (V7-09.09)
        plan = _plan(
            conn,
            "SELECT event_id, occurred_start_us, occurred_end_us"
            " FROM events_v7"
            " WHERE scope_id = ? AND subject_canon = ?"
            "   AND predicate_lemma = ? AND generation = ?",
            ("s1", "alice", "moved", 1),
        )
        assert "COVERING INDEX idx_events_v7_subject" in plan
        # pure window overlap on the event calendar
        plan = _plan(
            conn,
            "SELECT event_id FROM events_v7"
            " WHERE scope_id = ? AND generation = ?"
            "   AND occurred_start_us <= ? AND occurred_end_us >= ?",
            ("s1", 1, 100, 0),
        )
        assert "COVERING INDEX idx_events_v7_window" in plan
        # current_value: latest non-disputed value per state key (V7-16.04)
        plan = _plan(
            conn,
            "SELECT unit_id FROM state_facts"
            " WHERE scope_id = ? AND state_key = ? AND status = 'current'"
            "   AND generation = ?",
            ("s1", "user/home_city", 1),
        )
        assert "COVERING INDEX idx_state_facts_key" in plan
        # L-obs slot read (V7-14.06/09)
        plan = _plan(
            conn,
            "SELECT stale FROM observations_v7"
            " WHERE scope_id = ? AND slot = ? AND generation = ?",
            ("s1", "alice/food", 1),
        )
        assert "COVERING INDEX idx_obs_v7_slot" in plan
        # MATCH rowids → eligibility join on the carrier
        plan = _plan(
            conn,
            "SELECT row_id FROM unit_fts_rows"
            " WHERE scope_id = ? AND generation = ?",
            ("s1", 1),
        )
        assert "COVERING INDEX idx_unit_fts_rows_scope" in plan
        # preference + profile + standing-query scans ride their indexes
        plan = _plan(
            conn,
            "SELECT unit_id FROM preferences"
            " WHERE scope_id = ? AND subject_canon = ?",
            ("s1", "alice"),
        )
        assert "SCAN" not in plan
        plan = _plan(
            conn,
            "SELECT sq_id FROM standing_queries"
            " WHERE scope_id = ? AND dirty = 1",
            ("s1",),
        )
        assert "SCAN" not in plan
        plan = _plan(
            conn,
            "SELECT rule_id FROM standing_rules"
            " WHERE scope_id = ? AND status = 'active'",
            ("s1",),
        )
        assert "SCAN" not in plan


# ---------------------------------------------------------------------------
# FTS5 shadow wiring (capability-gated)
# ---------------------------------------------------------------------------


def test_fts_tokenizer_probe_reports_build(store):
    """The tokenizer probe answers honestly for this build — unicode61,
    porter, and trigram are all present on SQLite 3.45.1 (probe results are
    reported, not assumed). The live CREATE path runs on the writer; the
    query_only reader falls back to the static compile-options answer, and
    both agree."""
    with store.tx() as conn:
        assert fts5_tokenizer_available(conn, "unicode61") is True
        assert fts5_tokenizer_available(
            conn, "unicode61 remove_diacritics 2"
        ) is True
        assert fts5_tokenizer_available(conn, "porter unicode61") is True
        assert fts5_tokenizer_available(conn, "trigram") is True
        # A nonsense spec probes False rather than raising — the honest
        # degradation path (unavailable, never a failed open).
        assert fts5_tokenizer_available(conn, "no_such_tokenizer") is False
    _TOKENIZER_PROBE_CACHE.clear()  # re-probe through the reader path
    with store.read() as conn:
        assert fts5_tokenizer_available(conn, "unicode61") is True
        assert fts5_tokenizer_available(
            conn, "unicode61 remove_diacritics 2"
        ) is True
        assert fts5_tokenizer_available(conn, "porter unicode61") is True
        assert fts5_tokenizer_available(conn, "trigram") is True
        assert fts5_tokenizer_available(conn, "no_such_tokenizer") is False
    _TOKENIZER_PROBE_CACHE.clear()


def test_fts_indexes_created_and_mirrored(ensured):
    """All three §30 virtual tables exist with their trigger sets; a
    content-row insert mirrors into all three indexes and a delete removes
    from all three (external-content wiring, v5 shape)."""
    s = ensured
    with s.read() as conn:
        present = v7_fts_present(conn)
        assert present == {
            "unit_fts": True,
            "unit_fts_stem": True,
            "unit_fts_tri": True,
        }
        for trg in V7_TRIGGERS:
            assert trg in _names(conn, "trigger")
        # §30 FTS column contract on the virtual table.
        assert {"text", "speaker", "entities", "session", "when"} <= _cols(
            conn, "unit_fts"
        )
        assert "text" in _cols(conn, "unit_fts_stem")
        assert "text" in _cols(conn, "unit_fts_tri")
    with s.tx() as conn:
        _insert_fts(
            conn, "u1", "s1", 1,
            "I love running in the park café", "alice", "alice park",
            "sess-1", "2023-05-07",
        )
    with s.read() as conn:
        assert conn.execute(
            "SELECT rowid FROM unit_fts WHERE unit_fts MATCH 'love'"
        ).fetchall() == [(1,)]
        # fielded query + the quoted ``when`` column
        assert conn.execute(
            "SELECT rowid FROM unit_fts WHERE unit_fts MATCH 'speaker:alice'"
        ).fetchall() == [(1,)]
        assert conn.execute(
            'SELECT rowid FROM unit_fts WHERE unit_fts MATCH'
            ' \'"when":"2023-05-07"\''
        ).fetchall() == [(1,)]
        # diacritics fold (remove_diacritics 2)
        assert conn.execute(
            "SELECT rowid FROM unit_fts WHERE unit_fts MATCH 'cafe'"
        ).fetchall() == [(1,)]
        # stemmed shadow (porter)
        assert conn.execute(
            "SELECT rowid FROM unit_fts_stem"
            " WHERE unit_fts_stem MATCH 'run'"
        ).fetchall() == [(1,)]
        # trigram substring
        assert conn.execute(
            "SELECT rowid FROM unit_fts_tri"
            " WHERE unit_fts_tri MATCH 'unnin'"
        ).fetchall() == [(1,)]
    # delete content-first, then carrier — the v5 reprojection order; the
    # ad triggers strip every index's postings.
    with s.tx() as conn:
        conn.execute(
            "DELETE FROM unit_fts_content WHERE fts_row_id IN"
            " (SELECT row_id FROM unit_fts_rows WHERE unit_id = ?"
            "  AND generation = ?)",
            ("u1", 1),
        )
        conn.execute(
            "DELETE FROM unit_fts_rows WHERE unit_id = ? AND generation = ?",
            ("u1", 1),
        )
    with s.read() as conn:
        for tbl, term in (
            ("unit_fts", "love"),
            ("unit_fts_stem", "run"),
            ("unit_fts_tri", "unnin"),
        ):
            assert conn.execute(
                f"SELECT rowid FROM {tbl} WHERE {tbl} MATCH '{term}'"
            ).fetchall() == []


def test_fts_generation_fence(ensured):
    """One indexed row per (unit_id, generation): a rebuild writes the new
    generation beside the live one; readers filter by generation."""
    s = ensured
    with s.tx() as conn:
        _insert_fts(conn, "u1", "s1", 1, "alpha beta", "alice", "", "", "")
        _insert_fts(conn, "u1", "s1", 2, "gamma delta", "alice", "", "", "")
    with s.read() as conn:
        rows = conn.execute(
            "SELECT r.unit_id, r.generation FROM unit_fts f"
            " JOIN unit_fts_rows r ON r.row_id = f.rowid"
            " WHERE f.unit_fts MATCH 'alpha' AND r.scope_id = 's1'"
        ).fetchall()
        assert rows == [("u1", 1)]
        rows = conn.execute(
            "SELECT r.unit_id, r.generation FROM unit_fts f"
            " JOIN unit_fts_rows r ON r.row_id = f.rowid"
            " WHERE f.unit_fts MATCH 'gamma' AND r.scope_id = 's1'"
        ).fetchall()
        assert rows == [("u1", 2)]


def test_multi_mention_and_disputed_values_coexist(ensured):
    """PK discriminators keep granular rows: two spans of the same canon in
    one unit, and two values under one state key (the V7-16.05 disputed
    case), both insert without collapsing."""
    s = ensured
    with s.tx() as conn:
        conn.execute(
            "INSERT INTO entity_mentions(scope_id, canon, unit_id,"
            " generation, surface, byte_start, byte_end)"
            " VALUES ('s1', 'alice', 'u1', 1, 'Alice', 0, 5)"
        )
        conn.execute(
            "INSERT INTO entity_mentions(scope_id, canon, unit_id,"
            " generation, surface, byte_start, byte_end)"
            " VALUES ('s1', 'alice', 'u1', 1, 'Alice', 40, 45)"
        )
        conn.execute(
            "INSERT INTO state_facts(scope_id, state_key, unit_id,"
            " generation, value_text, value_norm, status)"
            " VALUES ('s1', 'user/home_city', 'u1', 1, 'Berlin',"
            " 'berlin', 'current')"
        )
        conn.execute(
            "INSERT INTO state_facts(scope_id, state_key, unit_id,"
            " generation, value_text, value_norm, status)"
            " VALUES ('s1', 'user/home_city', 'u1', 1, 'Munich',"
            " 'munich', 'disputed')"
        )
        conn.execute(
            "INSERT INTO preferences(scope_id, subject_canon, unit_id,"
            " generation, object_text, polarity)"
            " VALUES ('s1', 'alice', 'u1', 1, 'sushi', 'affirm')"
        )
        conn.execute(
            "INSERT INTO preferences(scope_id, subject_canon, unit_id,"
            " generation, object_text, polarity)"
            " VALUES ('s1', 'alice', 'u1', 1, 'olives', 'negate')"
        )
    with s.read() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM entity_mentions"
            " WHERE scope_id='s1' AND canon='alice' AND unit_id='u1'"
        ).fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM state_facts"
            " WHERE scope_id='s1' AND state_key='user/home_city'"
            " AND unit_id='u1'"
        ).fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM preferences"
            " WHERE scope_id='s1' AND subject_canon='alice'"
            " AND unit_id='u1'"
        ).fetchone()[0] == 2
    # and generation coexistence: same natural key, two generations
    with s.tx() as conn:
        conn.execute(
            "INSERT INTO state_facts(scope_id, state_key, unit_id,"
            " generation, value_text, value_norm, status)"
            " VALUES ('s1', 'user/home_city', 'u1', 2, 'Berlin',"
            " 'berlin', 'current')"
        )
    with s.read() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM state_facts"
            " WHERE scope_id='s1' AND state_key='user/home_city'"
            " AND unit_id='u1' AND value_norm='berlin'"
        ).fetchone()[0] == 2  # gen 1 + gen 2


# ---------------------------------------------------------------------------
# incremental corpus stats (V7-06.06)
# ---------------------------------------------------------------------------


def test_stats_update_and_lookup(ensured):
    """update_stats in the posting tx → corpus_stats/df reflect the add;
    df counts units (not occurrences) per field."""
    s = ensured
    rows = [
        {"unit_id": "u1", "fields": {
            "text": ["i", "love", "sushi", "sushi"],
            "speaker": ["alice"], "when": ["2023-05-07"],
        }},
        {"unit_id": "u2", "fields": {
            "text": "i love ramen",  # whitespace-joined string also accepted
            "entities": ["tokyo"],
        }},
    ]
    with s.tx() as conn:
        update_stats(conn, "s1", 1, rows)
    with s.read() as conn:
        stats = corpus_stats(conn, "s1", 1)
        # every declared field present; n_units = corpus N on every field
        assert stats["text"] == {"n": 2, "total_len": 7}
        assert stats["speaker"] == {"n": 2, "total_len": 1}
        assert stats["entities"] == {"n": 2, "total_len": 1}
        assert stats["session"] == {"n": 2, "total_len": 0}
        assert stats["when"] == {"n": 2, "total_len": 1}
        # df = units containing the term at least once in that field
        assert term_df(conn, "s1", 1, "text", "love") == 2
        assert term_df(conn, "s1", 1, "text", "sushi") == 1  # 2× in u1 → 1
        assert term_df(conn, "s1", 1, "text", "absent") == 0
        assert term_df(conn, "s1", 1, "entities", "tokyo") == 1
        assert term_df(conn, "s1", 1, "speaker", "alice") == 1
        assert term_dfs(conn, "s1", 1, ["love", "ramen", "nope"]) == {
            "love": 2, "ramen": 1, "nope": 0,
        }


def test_stats_second_add_increments_not_recomputes(ensured):
    """A second batch adjusts the same rows (upsert), never recomputes —
    the df/length sums accumulate exactly."""
    s = ensured
    with s.tx() as conn:
        update_stats(conn, "s1", 1, [
            {"unit_id": "u1", "fields": {"text": ["alpha", "beta"]}},
        ])
    with s.tx() as conn:
        update_stats(conn, "s1", 1, [
            {"unit_id": "u2", "fields": {"text": ["beta", "gamma"]}},
            {"unit_id": "u3", "fields": {"text": ["beta"]}},
        ])
    with s.read() as conn:
        assert corpus_stats(conn, "s1", 1)["text"] == {
            "n": 3, "total_len": 5,
        }
        assert term_df(conn, "s1", 1, "text", "beta") == 3
        assert term_df(conn, "s1", 1, "text", "alpha") == 1
        assert term_df(conn, "s1", 1, "text", "gamma") == 1


def test_stats_scope_and_generation_isolated(ensured):
    """Stats are keyed by (scope, generation, stats_version) — a sibling
    scope and a rebuild generation stay disjoint (D7-10)."""
    s = ensured
    with s.tx() as conn:
        update_stats(conn, "s1", 1, [
            {"unit_id": "u1", "fields": {"text": ["alpha"]}},
        ])
        update_stats(conn, "s2", 1, [
            {"unit_id": "u2", "fields": {"text": ["alpha", "alpha", "b"]}},
        ])
        update_stats(conn, "s1", 2, [
            {"unit_id": "u1", "fields": {"text": ["zeta"]}},
        ])
    with s.read() as conn:
        assert corpus_stats(conn, "s1", 1)["text"] == {
            "n": 1, "total_len": 1,
        }
        assert corpus_stats(conn, "s2", 1)["text"] == {
            "n": 1, "total_len": 3,
        }
        assert corpus_stats(conn, "s1", 2)["text"] == {
            "n": 1, "total_len": 1,
        }
        assert term_df(conn, "s1", 1, "text", "alpha") == 1
        assert term_df(conn, "s2", 1, "text", "alpha") == 1
        assert term_df(conn, "s2", 1, "text", "b") == 1
        assert term_df(conn, "s1", 2, "text", "alpha") == 0
        assert term_df(conn, "s1", 2, "text", "zeta") == 1


def test_stats_decrement_on_unit_removal(ensured):
    """decrement_stats subtracts the same per-unit contributions (closure,
    V7-06.11) and floors at zero instead of corrupting the aggregates."""
    s = ensured
    unit = {"unit_id": "u1", "fields": {"text": ["alpha", "beta"]}}
    with s.tx() as conn:
        update_stats(conn, "s1", 1, [
            unit,
            {"unit_id": "u2", "fields": {"text": ["beta", "gamma"]}},
        ])
    with s.tx() as conn:
        decrement_stats(conn, "s1", 1, [unit])
    with s.read() as conn:
        assert corpus_stats(conn, "s1", 1)["text"] == {
            "n": 1, "total_len": 2,
        }
        assert term_df(conn, "s1", 1, "text", "alpha") == 0
        assert term_df(conn, "s1", 1, "text", "beta") == 1
        assert term_df(conn, "s1", 1, "text", "gamma") == 1
    # Over-decrement floors at 0 — never a negative df/n_units.
    with s.tx() as conn:
        decrement_stats(conn, "s1", 1, [unit, unit])
    with s.read() as conn:
        stats = corpus_stats(conn, "s1", 1)["text"]
        assert stats["n"] == 0 and stats["total_len"] == 0
        assert term_df(conn, "s1", 1, "text", "beta") == 0


def test_stats_version_isolated(ensured):
    """A second stats_version writes its own rows beside v1 (the key is
    (scope, generation, stats_version), never a corpus fingerprint)."""
    s = ensured
    rows = [{"unit_id": "u1", "fields": {"text": ["alpha"]}}]
    with s.tx() as conn:
        update_stats(conn, "s1", 1, rows)
        update_stats(conn, "s1", 1, rows, stats_version="bm25f/v2-draft")
    with s.read() as conn:
        assert corpus_stats(conn, "s1", 1)["text"] == {
            "n": 1, "total_len": 1,
        }
        assert corpus_stats(
            conn, "s1", 1, stats_version="bm25f/v2-draft"
        )["text"] == {"n": 1, "total_len": 1}
        assert term_df(
            conn, "s1", 1, "text", "alpha",
            stats_version="bm25f/v2-draft",
        ) == 1
        assert term_df(conn, "s1", 1, "text", "alpha") == 1


def test_stats_same_tx_as_posting_write(ensured):
    """update_stats rolls back with the posting write — it runs on the
    caller's connection, never commits its own (V7-06.06)."""
    s = ensured
    with pytest.raises(RuntimeError):
        with s.tx() as conn:
            _insert_fts(conn, "u1", "s1", 1, "alpha", "a", "", "", "")
            update_stats(conn, "s1", 1, [
                {"unit_id": "u1", "fields": {"text": ["alpha"]}},
            ])
            raise RuntimeError("abort")
    with s.read() as conn:
        assert corpus_stats(conn, "s1", 1)["text"] == {
            "n": 0, "total_len": 0,
        }
        assert term_df(conn, "s1", 1, "text", "alpha") == 0
