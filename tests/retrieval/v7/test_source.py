"""Tests for the V7 source lane — ``verbatim/retrieval/v7/source.py``
(w-lane-source, wave B).

Mirror-DDL approach per the wave brief: the §30 ``units``/``unit_fts``
carrier pair and the v5 source-projection trio
(``source_lexical_projection`` / ``source_fts_rows`` / ``source_fts`` /
``source_fts_idx``) are created inline — ``schema_v7`` is a concurrent
worker's file.  Both ``unit_fts`` rowid conventions are exercised: the
real carrier pair (``unit_fts.rowid`` → ``unit_fts_rows.row_id`` →
``unit_id``) and the wave-A standalone convention
(``unit_fts.rowid == units.rowid``).

These tests unit-test the LANE: dual-mode matching, the synthetic
``source_id:revision`` unit_id for source-only hits, eligibility
closure, the generation/scope fences, quarantine/purge/lifecycle gates,
cap/deadline discipline, deterministic ordering, and honest
degradation — never through the full pipeline.
"""

from __future__ import annotations

import itertools
import re
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verbatim.core.types_v7 import (  # noqa: E402
    BudgetClass,
    IntentClass,
    IntentResult,
    LaneContextV7,
    LaneName,
    LaneSlice,
    LaneStatus,
    NormAnalysis,
    NormTerm,
    QueryViewV7,
    RetrievalPolicyV7,
)
from verbatim.enrichment.normalize import normalize_text  # noqa: E402
from verbatim.retrieval.v7 import source as srcl  # noqa: E402
from verbatim.retrieval.v7.lanes_base import LANE_REGISTRY  # noqa: E402


# ---------------------------------------------------------------------------
# mirror DDL — §30 unit index (carrier convention) + v5 source projection
# ---------------------------------------------------------------------------

_UNITS_DDL = """
CREATE TABLE units (
  unit_id TEXT NOT NULL, source_id TEXT NOT NULL, revision INTEGER NOT NULL,
  scope_id TEXT NOT NULL, kind TEXT, parent_unit_id TEXT, session_id TEXT,
  seq INTEGER, speaker_canon TEXT, perspective TEXT,
  recorded_at_us INTEGER, occurred_start_us INTEGER, occurred_end_us INTEGER,
  occurred_precision TEXT, occurred_source TEXT, byte_start INTEGER,
  byte_end INTEGER, generation INTEGER NOT NULL,
  PRIMARY KEY (unit_id, generation)
);
"""

_CARRIER_DDL = """
CREATE TABLE unit_fts_rows (
  row_id INTEGER PRIMARY KEY, unit_id TEXT NOT NULL, scope_id TEXT NOT NULL,
  generation INTEGER NOT NULL, UNIQUE (unit_id, generation)
);
CREATE TABLE unit_fts_content (
  fts_row_id INTEGER PRIMARY KEY REFERENCES unit_fts_rows(row_id),
  text TEXT, speaker TEXT, entities TEXT, session TEXT, "when" TEXT
);
"""

_UNIT_FTS_CARRIER_DDL = """
CREATE VIRTUAL TABLE unit_fts USING fts5(
  text, speaker, entities, session, "when",
  content='unit_fts_content', content_rowid='fts_row_id',
  tokenize='unicode61 remove_diacritics 2');
CREATE TRIGGER unit_fts_ai AFTER INSERT ON unit_fts_content BEGIN
  INSERT INTO unit_fts(rowid, text, speaker, entities, session, "when")
    VALUES (new.fts_row_id, new.text, new.speaker, new.entities,
            new.session, new."when");
END;
CREATE TRIGGER unit_fts_ad AFTER DELETE ON unit_fts_content BEGIN
  INSERT INTO unit_fts(unit_fts, rowid, text, speaker, entities, session,
                       "when")
    VALUES ('delete', old.fts_row_id, old.text, old.speaker, old.entities,
            old.session, old."when");
END;
CREATE TRIGGER unit_fts_au AFTER UPDATE ON unit_fts_content BEGIN
  INSERT INTO unit_fts(unit_fts, rowid, text, speaker, entities, session,
                       "when")
    VALUES ('delete', old.fts_row_id, old.text, old.speaker, old.entities,
            old.session, old."when");
  INSERT INTO unit_fts(rowid, text, speaker, entities, session, "when")
    VALUES (new.fts_row_id, new.text, new.speaker, new.entities,
            new.session, new."when");
END;
"""

_UNIT_FTS_STANDALONE_DDL = """
CREATE VIRTUAL TABLE unit_fts USING fts5(
  text, speaker, entities, session, "when",
  tokenize='unicode61 remove_diacritics 2');
"""

_SOURCE_PROJ_DDL = """
CREATE TABLE source_lexical_projection (
  source_id TEXT NOT NULL, revision INTEGER NOT NULL, scope_id TEXT NOT NULL,
  generation INTEGER NOT NULL, tokens TEXT NOT NULL, doc_len INTEGER NOT NULL,
  digest TEXT NOT NULL, PRIMARY KEY (source_id, revision)
);
"""

_SOURCE_CARRIER_DDL = """
CREATE TABLE source_fts_rows (
  row_id INTEGER PRIMARY KEY, source_id TEXT NOT NULL, revision INTEGER NOT NULL,
  scope_id TEXT NOT NULL, generation INTEGER NOT NULL,
  UNIQUE (source_id, revision, generation)
);
CREATE TABLE source_fts (
  fts_row_id INTEGER PRIMARY KEY REFERENCES source_fts_rows(row_id),
  text TEXT
);
"""

_SOURCE_IDX_DDL = """
CREATE VIRTUAL TABLE source_fts_idx USING fts5(
  text, content='source_fts', content_rowid='fts_row_id',
  tokenize='unicode61');
CREATE TRIGGER source_fts_ai AFTER INSERT ON source_fts BEGIN
  INSERT INTO source_fts_idx(rowid, text) VALUES (new.fts_row_id, new.text);
END;
CREATE TRIGGER source_fts_ad AFTER DELETE ON source_fts BEGIN
  INSERT INTO source_fts_idx(source_fts_idx, rowid, text)
    VALUES ('delete', old.fts_row_id, old.text);
END;
CREATE TRIGGER source_fts_au AFTER UPDATE ON source_fts BEGIN
  INSERT INTO source_fts_idx(source_fts_idx, rowid, text)
    VALUES ('delete', old.fts_row_id, old.text);
  INSERT INTO source_fts_idx(rowid, text) VALUES (new.fts_row_id, new.text);
END;
"""

_SOURCES_DDL = """
CREATE TABLE sources (
  source_id TEXT PRIMARY KEY, origin TEXT NOT NULL, external_id TEXT,
  source_kind TEXT NOT NULL, scope_id TEXT NOT NULL, speaker_id TEXT,
  created_us INTEGER NOT NULL
);
"""

_GOVERNANCE_DDL = """
CREATE TABLE source_state (
  source_id TEXT PRIMARY KEY, namespace TEXT NOT NULL,
  control_version INTEGER NOT NULL, mutation_head TEXT NOT NULL,
  disposition TEXT NOT NULL, superseded_by TEXT, effective_at TEXT,
  known_at TEXT NOT NULL, valid_from TEXT, valid_to TEXT,
  updated_at TEXT NOT NULL, producer TEXT NOT NULL
);
CREATE TABLE quarantine (
  object_kind TEXT NOT NULL, object_id TEXT NOT NULL, revision INTEGER NOT NULL,
  scope_id TEXT NOT NULL, reason_codes_json TEXT NOT NULL DEFAULT '[]',
  findings_json TEXT NOT NULL DEFAULT '[]', state TEXT NOT NULL DEFAULT 'pending',
  opened_event INTEGER NOT NULL DEFAULT 0, decided_event INTEGER,
  decided_by TEXT, decision_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (object_kind, object_id, revision)
);
CREATE TABLE source_envelopes (
  envelope_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
  revision INTEGER NOT NULL, scope_id TEXT NOT NULL
);
CREATE TABLE purges (
  purge_id TEXT PRIMARY KEY, selection_digest BLOB NOT NULL,
  scope_id TEXT NOT NULL, state TEXT NOT NULL, requested_us INTEGER NOT NULL,
  approved_us INTEGER, completed_us INTEGER
);
CREATE TABLE purge_targets (
  purge_id TEXT NOT NULL, object_kind TEXT NOT NULL, object_id TEXT NOT NULL,
  PRIMARY KEY (purge_id, object_kind, object_id)
);
"""


def _fts5_ok() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE t USING fts5(x, tokenize='unicode61')"
        )
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


pytestmark = pytest.mark.skipif(
    not _fts5_ok(), reason="FTS5/unicode61 not compiled in"
)


def _db(
    *,
    units: bool = True,
    unit_fts: bool = True,
    carrier: bool = True,
    proj: bool = True,
    src_carrier: bool = True,
    src_idx: bool = True,
    sources: bool = True,
    governance: bool = True,
) -> sqlite3.Connection:
    """Mirror store. ``carrier=True`` builds the real §30
    ``unit_fts_rows``+``unit_fts_content`` pair; ``carrier=False`` builds
    the wave-A standalone convention (``unit_fts.rowid==units.rowid``)."""
    conn = sqlite3.connect(":memory:")
    ddl = ""
    if units:
        ddl += _UNITS_DDL
    if carrier:
        ddl += _CARRIER_DDL
    if unit_fts:
        ddl += (
            _UNIT_FTS_CARRIER_DDL if carrier else _UNIT_FTS_STANDALONE_DDL
        )
    if proj:
        ddl += _SOURCE_PROJ_DDL
    if src_carrier:
        ddl += _SOURCE_CARRIER_DDL
    if src_idx:
        ddl += _SOURCE_IDX_DDL
    if sources:
        ddl += _SOURCES_DDL
    if governance:
        ddl += _GOVERNANCE_DDL
    conn.executescript(ddl)
    return conn


def seed_unit(conn, unit_id, source_id, revision, text, *, scope="s1",
              gen=1, kind="turn", speaker="alice", session="sess1",
              carrier=True, seq=0):
    """Insert one units row + its FTS shadow(s) at one generation."""
    conn.execute(
        "INSERT INTO units(unit_id, source_id, revision, scope_id, kind,"
        " parent_unit_id, session_id, seq, speaker_canon, perspective,"
        " recorded_at_us, occurred_start_us, occurred_end_us,"
        " occurred_precision, occurred_source, byte_start, byte_end,"
        " generation) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (unit_id, source_id, revision, scope, kind, None, session, seq,
         speaker, "user_stated", 1_700_000_000_000_000, 0, 0, "instant",
         "explicit", 0, len(text.encode()), gen),
    )
    if carrier:
        rid = conn.execute(
            "INSERT INTO unit_fts_rows(unit_id, scope_id, generation)"
            " VALUES(?,?,?)",
            (unit_id, scope, gen),
        ).lastrowid
        conn.execute(
            'INSERT INTO unit_fts_content(fts_row_id, text, speaker,'
            ' entities, session, "when") VALUES(?,?,?,?,?,?)',
            (rid, text, speaker, "", session, ""),
        )
    else:
        rid = conn.execute(
            "SELECT rowid FROM units WHERE unit_id=? AND generation=?",
            (unit_id, gen),
        ).fetchone()[0]
        conn.execute(
            'INSERT INTO unit_fts(rowid, text, speaker, entities, session,'
            ' "when") VALUES(?,?,?,?,?,?)',
            (rid, text, speaker, "", session, ""),
        )
    return int(rid)


def seed_source(conn, source_id, revision, text, *, scope="s1", gen=1,
                kind="user_message", speaker="alice", proj=True,
                src_carrier=True, sources=True):
    """Insert one source + its norm/v1 lexical projection + FTS carrier —
    the same transaction shape the v5 source-projection writer commits."""
    toks = normalize_text(text)
    if sources:
        conn.execute(
            "INSERT OR REPLACE INTO sources(source_id, origin,"
            " external_id, source_kind, scope_id, speaker_id, created_us)"
            " VALUES(?,?,?,?,?,?,?)",
            (source_id, "test", None, kind, scope, speaker,
             1_700_000_000_000_000),
        )
    if proj:
        conn.execute(
            "INSERT OR REPLACE INTO source_lexical_projection(source_id,"
            " revision, scope_id, generation, tokens, doc_len, digest)"
            " VALUES(?,?,?,?,?,?,?)",
            (source_id, revision, scope, gen, toks,
             len(toks.split()), "dg"),
        )
    if src_carrier:
        rid = conn.execute(
            "INSERT INTO source_fts_rows(source_id, revision, scope_id,"
            " generation) VALUES(?,?,?,?)",
            (source_id, revision, scope, gen),
        ).lastrowid
        conn.execute(
            "INSERT INTO source_fts(fts_row_id, text) VALUES(?,?)",
            (rid, toks),
        )
    return toks


def mkqv(query, *, channels=None):
    """QueryViewV7 literal — whitespace tokens folded into the unit_fts
    projection; ``channels`` maps a term to 'identifier'."""
    channels = channels or {}
    terms = []
    idents = []
    for m in re.finditer(r"[\w./:@#-]+", query):
        tok = m.group(0).strip("\"'")
        if not tok:
            continue
        term = srcl._fold(tok)
        start_b = len(query[: m.start(0)].encode())
        end_b = start_b + len(tok.encode())
        ch = channels.get(term, "text")
        nt = NormTerm(term=term, channel=ch, byte_start=start_b,
                      byte_end=end_b)
        terms.append(nt)
        if ch == "identifier":
            idents.append(nt)
    norm = NormAnalysis(analyzer_id="norm/v2", terms=tuple(terms),
                        identifiers=tuple(idents), text=srcl._fold(query))
    return QueryViewV7(
        query=query, norm=norm,
        intent=IntentResult(primary=IntentClass.LOOKUP,
                            classes=(IntentClass.LOOKUP,)),
    )


def mkctx(conn, *, eligible=None, scope="s1", gen=10):
    return LaneContextV7(
        store=conn, scope_id=scope, generation=gen, eligible=eligible,
        query_time_us=0, profile="test", budget=BudgetClass.MID,
        policy=RetrievalPolicyV7(
            policy_id="retrieval_policy/v7", profile="test",
            lanes=(LaneName.SOURCE,), lane_weights={}),
        manifest={})


def mkslice(cap=50, deadline_ms=60_000.0):
    return LaneSlice(deadline_ms=deadline_ms, cap=cap)


def run(conn, query, *, eligible=None, scope="s1", gen=10, cap=50,
        deadline_ms=60_000.0, channels=None):
    return srcl.lane_source(
        mkctx(conn, eligible=eligible, scope=scope, gen=gen),
        mkqv(query, channels=channels),
        mkslice(cap=cap, deadline_ms=deadline_ms),
    )


# ---------------------------------------------------------------------------
# registration + term extraction
# ---------------------------------------------------------------------------


def test_registered_in_lane_registry():
    assert LANE_REGISTRY[LaneName.SOURCE] is srcl.lane_source
    assert srcl.SourceLane().name == "source"


def test_unit_fts_term_match_emits_candidate():
    """A text-field term match on a projected unit emits a unit-backed
    candidate carrying the real unit_id and the required signal shape."""
    conn = _db()
    seed_unit(conn, "u1", "srcA", 1, "the harbor lights went out")
    seed_unit(conn, "u2", "srcA", 1, "unrelated content entirely")
    out = run(conn, "harbor")
    assert out.status == LaneStatus.OK
    assert len(out.candidates) == 1
    c = out.candidates[0]
    assert c.unit_id == "u1"
    assert c.source_id == "srcA"
    assert c.revision == 1
    assert c.rank == 1
    assert c.raw_score > 0
    assert c.signals["match"] == "unit_fts"
    assert c.signals["terms_hit"] == ["harbor"]
    assert c.signals["coverage"] == 1.0
    assert c.signals["unit_level"] is True
    assert c.signals["eligibility"] == "unit"
    assert out.examined > 0 and out.eligible > 0


def test_source_fts_catches_unprojected_source():
    """A source-only store (no V7 units at all) still retrieves through
    the governed lane — the coverage floor the lane exists for."""
    conn = _db(units=False, unit_fts=False, carrier=False)
    seed_source(conn, "srcA", 1, "the harbor lights went out")
    seed_source(conn, "srcB", 1, "something else entirely")
    out = run(conn, "harbor")
    assert out.status == LaneStatus.OK
    assert len(out.candidates) == 1
    c = out.candidates[0]
    assert c.unit_id == "srcA:1"
    assert c.source_id == "srcA"
    assert c.revision == 1
    assert c.signals["unit_level"] is False
    assert c.signals["source_granular"] is True
    assert c.signals["no_units"] is True
    assert c.signals["match"] in ("source_fts", "source_projection")
    assert out.stats["source_only_orphaned"] == 1
    assert out.stats["unit_mode"].startswith("unavailable")


def test_identifier_free_query_runs():
    """The lane is never intent-gated: a plain natural-language query on
    a source-only store still executes (no thin/run_source gate)."""
    conn = _db(units=False, unit_fts=False, carrier=False)
    seed_source(conn, "srcA", 1, "maria keeps the ledger in the drawer")
    out = run(conn, "where does maria keep the ledger")
    assert out.status == LaneStatus.OK
    assert any(c.source_id == "srcA" for c in out.candidates)
    assert out.stats["unit_mode"].startswith("unavailable")


def test_standalone_rowid_convention():
    """The wave-A mirror convention (``unit_fts.rowid == units.rowid``)
    is honored when the §30 carrier table is absent."""
    conn = _db(carrier=False)
    seed_unit(conn, "u1", "srcA", 1, "harbor lights", carrier=False)
    out = run(conn, "harbor")
    assert out.status == LaneStatus.OK
    assert [c.unit_id for c in out.candidates] == ["u1"]
    assert out.stats["unit_rowid_space"] == "units.rowid"


def test_source_mode_carrier_without_projection():
    """``source_lexical_projection`` absent but the FTS carrier present:
    ``source_fts.text`` supplies the token stream (mode='carrier')."""
    conn = _db(units=False, unit_fts=False, carrier=False, proj=False)
    seed_source(conn, "srcA", 1, "harbor lights", proj=False)
    out = run(conn, "harbor")
    assert out.status == LaneStatus.OK
    assert out.stats["source_mode"] == "carrier"
    assert [c.unit_id for c in out.candidates] == ["srcA:1"]


def test_projection_scan_without_fts_idx():
    """``source_fts_idx`` absent: the projection scan itself decides
    matching (nomination=None) — still governed, still emitted."""
    conn = _db(units=False, unit_fts=False, carrier=False, src_idx=False)
    seed_source(conn, "srcA", 1, "harbor lights")
    out = run(conn, "harbor")
    assert out.status == LaneStatus.OK
    assert out.stats["source_nomination"] == "no_fts_idx"
    c = out.candidates[0]
    assert c.signals["match"] == "source_projection"
    assert c.unit_id == "srcA:1"


def test_phrase_boost():
    """The consecutive query-term sequence in the document earns the
    declared phrase bonus on top of BM25."""
    conn = _db()
    seed_unit(conn, "u1", "srcA", 1, "harbor lights went out last night")
    seed_unit(conn, "u2", "srcA", 1, "lights over the distant harbor")
    out = run(conn, "harbor lights")
    assert out.status == LaneStatus.OK
    by_id = {c.unit_id: c for c in out.candidates}
    assert set(by_id) == {"u1", "u2"}
    assert by_id["u1"].signals["phrase"] == 1.0
    assert by_id["u2"].signals["phrase"] == 0.0
    assert by_id["u1"].raw_score > by_id["u2"].raw_score


# ---------------------------------------------------------------------------
# eligibility + fences
# ---------------------------------------------------------------------------


def test_eligibility_set_filters_units():
    """A unit-id set handle admits only its members."""
    conn = _db()
    seed_unit(conn, "u1", "srcA", 1, "harbor lights")
    seed_unit(conn, "u2", "srcA", 1, "harbor lights")
    out = run(conn, "harbor", eligible={"u1"})
    assert [c.unit_id for c in out.candidates] == ["u1"]
    assert out.stats["unit_eligible"] == 1


def test_eligibility_callable_filters():
    """A callable receives the unit row dict; denials are withheld."""
    conn = _db()
    seed_unit(conn, "u1", "srcA", 1, "harbor lights", speaker="alice")
    seed_unit(conn, "u2", "srcA", 1, "harbor lights", speaker="bob")
    out = run(
        conn, "harbor",
        eligible=lambda row: row.get("speaker_canon") != "bob",
    )
    assert [c.unit_id for c in out.candidates] == ["u1"]


def test_eligibility_set_blocks_source_level_hits():
    """A unit-id container cannot name a source-granular row — the
    synthetic id is never a member, so source-level hits fail closed
    (counted, never emitted)."""
    conn = _db(units=False, unit_fts=False, carrier=False)
    seed_source(conn, "srcA", 1, "harbor lights")
    out = run(conn, "harbor", eligible={"u1"})
    assert out.candidates == []
    assert out.stats["eligible_dropped"] == 1
    assert out.status == LaneStatus.OK


def test_eligibility_callable_on_source_rows():
    """A callable still gates source-level rows — it sees a synthesized
    row with kind='source' and the synthetic unit_id."""
    conn = _db(units=False, unit_fts=False, carrier=False)
    seed_source(conn, "srcA", 1, "harbor lights")
    seen = []
    out = run(
        conn, "harbor",
        eligible=lambda row: seen.append(dict(row)) or False,
    )
    assert out.candidates == []
    assert seen and seen[0]["unit_id"] == "srcA:1"
    assert seen[0]["kind"] == "source"


def test_unrecognized_eligibility_fails_closed():
    conn = _db()
    seed_unit(conn, "u1", "srcA", 1, "harbor lights")
    out = run(conn, "harbor", eligible=object())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "eligibility_shape_unknown"


def test_generation_fence_units():
    """``generation <= ctx.generation`` with latest-visible resolution:
    a unit projected at gen 5 is invisible at fence 3; one with rows at
    gen 2 and gen 4 resolves to the gen-2 projection at fence 3."""
    conn = _db()
    seed_unit(conn, "u_new", "srcA", 1, "harbor lights", gen=5)
    seed_unit(conn, "u_old", "srcB", 1, "harbor lights v1", gen=2)
    seed_unit(conn, "u_old", "srcB", 1, "harbor lights v2", gen=4)
    out = run(conn, "harbor", gen=3)
    assert [c.unit_id for c in out.candidates] == ["u_old"]
    # and at fence 6 the gen-4 rebuild row wins
    out2 = run(conn, "harbor v2", gen=6)
    assert "u_old" in {c.unit_id for c in out2.candidates}
    assert "u_new" in {c.unit_id for c in out2.candidates}


def test_generation_fence_source():
    """The source projection fences at ``generation <= ctx.generation``
    — a row stamped above the pinned snapshot is invisible."""
    conn = _db(units=False, unit_fts=False, carrier=False)
    seed_source(conn, "srcA", 1, "harbor lights", gen=5)
    out = run(conn, "harbor", gen=3)
    assert out.candidates == []
    out2 = run(conn, "harbor", gen=10)
    assert [c.unit_id for c in out2.candidates] == ["srcA:1"]


def test_scope_fence():
    """Foreign-scope rows never surface — units and sources alike."""
    conn = _db()
    seed_unit(conn, "u1", "srcA", 1, "harbor lights", scope="s2")
    seed_source(conn, "srcB", 1, "harbor lights", scope="s2")
    out = run(conn, "harbor", scope="s1")
    assert out.candidates == []


def test_sources_scope_veto():
    """A projection row claiming scope s1 whose ``sources`` row carries
    s2 is vetoed by the sources join (never trusted on its own word)."""
    conn = _db(units=False, unit_fts=False, carrier=False)
    seed_source(conn, "srcA", 1, "harbor lights", scope="s1")
    conn.execute("UPDATE sources SET scope_id='s2' WHERE source_id='srcA'")
    out = run(conn, "harbor", scope="s1")
    assert out.candidates == []
    assert out.stats["source_drops"]["scope"] == 1


# ---------------------------------------------------------------------------
# governance gates — quarantine / purge / lifecycle
# ---------------------------------------------------------------------------


def test_quarantine_hold_withholds_source():
    conn = _db(units=False, unit_fts=False, carrier=False)
    seed_source(conn, "srcA", 1, "harbor lights")
    seed_source(conn, "srcB", 1, "harbor lights")
    conn.execute(
        "INSERT INTO quarantine(object_kind, object_id, revision,"
        " scope_id, state) VALUES('source','srcA',1,'s1','pending')")
    out = run(conn, "harbor")
    assert [c.unit_id for c in out.candidates] == ["srcB:1"]
    assert out.stats["source_drops"]["held"] == 1


def test_envelope_hold_cascade():
    """A hold on the covering ``source_envelope`` withholds the source —
    the V3-14.10 cascade carried."""
    conn = _db(units=False, unit_fts=False, carrier=False)
    seed_source(conn, "srcA", 1, "harbor lights")
    conn.execute(
        "INSERT INTO source_envelopes(envelope_id, source_id, revision,"
        " scope_id) VALUES('env1','srcA',1,'s1')")
    conn.execute(
        "INSERT INTO quarantine(object_kind, object_id, revision,"
        " scope_id, state) VALUES('source_envelope','env1',1,'s1','pending')")
    out = run(conn, "harbor")
    assert out.candidates == []
    assert out.stats["source_drops"]["held"] == 1


def test_purge_suppression_source_revision():
    conn = _db(units=False, unit_fts=False, carrier=False)
    seed_source(conn, "srcA", 2, "harbor lights")
    seed_source(conn, "srcB", 1, "harbor lights")
    conn.execute(
        "INSERT INTO purges(purge_id, selection_digest, scope_id, state,"
        " requested_us) VALUES('p1',x'00','s1','suppressed',1)")
    conn.execute(
        "INSERT INTO purge_targets(purge_id, object_kind, object_id)"
        " VALUES('p1','source_revision','srcA:2')")
    out = run(conn, "harbor")
    assert [c.unit_id for c in out.candidates] == ["srcB:1"]
    assert out.stats["source_drops"]["purged"] == 1


def test_lifecycle_gate():
    """``source_state`` currency: ``erased`` never answers; ``active``
    inside its window does; a missing row stays admissible (V5-14.16)."""
    conn = _db(units=False, unit_fts=False, carrier=False)
    seed_source(conn, "srcA", 1, "harbor lights")
    seed_source(conn, "srcB", 1, "harbor lights")
    seed_source(conn, "srcC", 1, "harbor lights")
    conn.execute(
        "INSERT INTO source_state(source_id, namespace, control_version,"
        " mutation_head, disposition, known_at, updated_at, producer)"
        " VALUES('srcA','s1',1,'m','erased','2020-01-01T00:00:00Z',"
        " '2020-01-01T00:00:00Z','test')")
    conn.execute(
        "INSERT INTO source_state(source_id, namespace, control_version,"
        " mutation_head, disposition, known_at, valid_from, updated_at,"
        " producer) VALUES('srcB','s1',1,'m','active','2020-01-01T00:00:00Z',"
        " '2020-01-01T00:00:00Z','2020-01-01T00:00:00Z','test')")
    out = run(conn, "harbor")
    ids = sorted(c.unit_id for c in out.candidates)
    assert ids == ["srcB:1", "srcC:1"]
    assert out.stats["source_drops"]["lifecycle"] == 1


# ---------------------------------------------------------------------------
# degradation honesty
# ---------------------------------------------------------------------------


def test_missing_unit_fts_with_populated_units_partial():
    """``units`` populated but ``unit_fts`` absent is a real coverage
    gap — ``partial``/``unit_fts_missing`` while source-level hits still
    ship (never a silent success)."""
    conn = _db(unit_fts=False)
    seed_unit(conn, "u1", "srcA", 1, "harbor lights")
    seed_source(conn, "srcA", 1, "harbor lights")
    out = run(conn, "harbor")
    assert out.status == LaneStatus.PARTIAL
    assert out.reason == "unit_fts_missing"
    assert "populated_units" in out.stats["unit_mode"]
    # the source-level fallback still covers the gap
    assert [c.unit_id for c in out.candidates] == ["srcA:1"]


def test_both_paths_missing_unavailable():
    """No ``unit_fts`` and no source projection/carrier: the lane is
    honestly unavailable and the reason names the missing substrate."""
    conn = _db(units=False, unit_fts=False, carrier=False, proj=False,
               src_carrier=False, src_idx=False)
    out = run(conn, "harbor")
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason.startswith("no_indexes")
    assert "unit_fts" in out.reason
    assert "source" in out.reason


def test_no_read_snapshot_unavailable():
    ctx = mkctx(object())  # no conn, no execute — nothing to read
    out = srcl.lane_source(ctx, mkqv("harbor"), mkslice())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "no_read_snapshot"


def test_generation_unpinned_unavailable():
    conn = _db()
    seed_unit(conn, "u1", "srcA", 1, "harbor lights")
    out = srcl.lane_source(
        mkctx(conn, gen=None), mkqv("harbor"), mkslice()
    )
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "generation_unpinned"


def test_no_terms_skipped():
    conn = _db()
    qv = QueryViewV7(
        query="", norm=NormAnalysis(analyzer_id="norm/v2", terms=(),
                                    identifiers=(), text=""),
        intent=IntentResult(primary=IntentClass.LOOKUP,
                            classes=(IntentClass.LOOKUP,)))
    out = srcl.lane_source(mkctx(conn), qv, mkslice())
    assert out.status == LaneStatus.SKIPPED
    assert out.reason == "no_terms"


def test_deadline_partial():
    """A deadline that expires before the first scan page returns
    ``partial``/``deadline`` with honest counts — never a fabricated
    empty ``ok``."""
    conn = _db()
    seed_unit(conn, "u1", "srcA", 1, "harbor lights")
    out = run(conn, "harbor", deadline_ms=0.0)
    assert out.status == LaneStatus.PARTIAL
    assert out.reason == "deadline"
    assert out.stats["deadline"] is True


def test_deadline_mid_scan_partial(monkeypatch):
    """A deadline cut inside the unit-universe scan degrades honestly:
    partial + deadline, real examined counts, no unverifiable hits."""
    conn = _db()
    seed_unit(conn, "u1", "srcA", 1, "harbor lights")
    seed_source(conn, "srcA", 1, "harbor lights")
    # init at t=0, first expired() call False, everything after expired
    clock = itertools.chain([0.0, 0.0, 0.0, 0.0], itertools.repeat(1e9))
    monkeypatch.setattr(srcl, "_monotonic", lambda: next(clock))
    out = run(conn, "harbor", deadline_ms=50.0)
    assert out.status == LaneStatus.PARTIAL
    assert out.reason == "deadline"
    assert out.stats["deadline"] is True
    assert out.examined >= 0


def test_cap_truncation():
    conn = _db()
    seed_unit(conn, "u1", "srcA", 1, "harbor lights")
    seed_unit(conn, "u2", "srcA", 1, "harbor lights harbor")
    seed_unit(conn, "u3", "srcA", 1, "harbor lights harbor harbor")
    out = run(conn, "harbor", cap=1)
    assert len(out.candidates) == 1
    assert out.stats["cap_truncated"] == 2
    assert out.stats["pool"] == 3


def test_cap_zero_emits_nothing():
    conn = _db()
    seed_unit(conn, "u1", "srcA", 1, "harbor lights")
    out = run(conn, "harbor", cap=0)
    assert out.status == LaneStatus.OK
    assert out.candidates == []
    assert out.stats["mode"] == "capped"


# ---------------------------------------------------------------------------
# ordering + merge semantics
# ---------------------------------------------------------------------------


def test_deterministic_ordering():
    """Identical inputs → identical candidate order; order is
    (-raw_score, unit_id), never SQLite's incidental FTS row order."""
    conn = _db()
    seed_unit(conn, "u1", "srcA", 1, "harbor")
    seed_unit(conn, "u2", "srcA", 1, "harbor harbor")
    seed_unit(conn, "u3", "srcA", 1, "harbor lights harbor")
    seed_source(conn, "srcB", 1, "harbor lights harbor")
    a = run(conn, "harbor lights")
    b = run(conn, "harbor lights")
    ids_a = [c.unit_id for c in a.candidates]
    ids_b = [c.unit_id for c in b.candidates]
    assert ids_a == ids_b
    assert ids_a == sorted(
        ids_a,
        key=lambda k: (-{c.unit_id: c.raw_score
                         for c in a.candidates}[k], k))
    assert [c.rank for c in a.candidates] == list(
        range(1, len(a.candidates) + 1))


def test_unit_backed_hit_suppresses_source_duplicate():
    """A source already represented by an emitted unit-backed candidate
    is not double-represented at source granularity — counted in
    ``source_covered_by_units``."""
    conn = _db()
    seed_unit(conn, "u1", "srcA", 1, "harbor lights")
    seed_source(conn, "srcA", 1, "harbor lights")
    out = run(conn, "harbor")
    assert [c.unit_id for c in out.candidates] == ["u1"]
    assert out.stats["source_covered_by_units"] == 1


def test_source_hit_alongside_uncovered_sources():
    """A projected source whose units did NOT match still surfaces at
    source granularity next to unit-backed hits from other sources."""
    conn = _db()
    seed_unit(conn, "u1", "srcA", 1, "harbor lights")
    # srcB's units carry unrelated text; the source payload matches
    seed_source(conn, "srcB", 1, "harbor lights at dawn")
    seed_unit(conn, "uB1", "srcB", 1, "unrelated tokens")
    out = run(conn, "harbor")
    ids = {c.unit_id for c in out.candidates}
    assert "u1" in ids
    assert "srcB:1" in ids
    srcb = next(c for c in out.candidates if c.unit_id == "srcB:1")
    assert srcb.signals["unit_level"] is False
    assert "no_units" not in srcb.signals  # units exist; they just missed


def test_signal_shape_and_formula():
    conn = _db()
    seed_unit(conn, "u1", "srcA", 1, "harbor lights")
    out = run(conn, "harbor lights")
    c = out.candidates[0]
    assert c.signals["formula"] == srcl.FORMULA_ID
    assert isinstance(c.signals["terms_hit"], list)
    assert isinstance(c.signals["coverage"], float)
    assert 0.0 < c.signals["coverage"] <= 1.0
    assert c.signals["bm25"] > 0
    assert out.stats["lane_version"] == srcl.LANE_VERSION
    assert out.stats["formula_status"] == srcl.FORMULA_STATUS


def test_multi_revision_source_identity():
    """Two revisions of one source are distinct identities —
    ``{source_id}:{revision}`` never collapses them."""
    conn = _db(units=False, unit_fts=False, carrier=False)
    seed_source(conn, "srcA", 1, "harbor lights")
    seed_source(conn, "srcA", 2, "harbor lights again")
    out = run(conn, "harbor")
    assert sorted(c.unit_id for c in out.candidates) == ["srcA:1", "srcA:2"]


def test_fused_entry_compatible(tmp_path):
    """The synthetic source-level unit_id flows through fusion cleanly —
    keys on str(unit_id); a unit-backed + source-level pair fuse as
    separate entries without dereferencing ``units``."""
    from verbatim.retrieval.v7 import fusion

    conn = _db()
    seed_unit(conn, "u1", "srcA", 1, "harbor lights")
    seed_source(conn, "srcB", 1, "harbor lights")
    out = run(conn, "harbor")
    fused = fusion.rrf_fuse([out])
    keys = {(f.unit_id, f.source_id, f.revision) for f in fused}
    assert ("u1", "srcA", 1) in keys
    assert ("srcB:1", "srcB", 1) in keys
