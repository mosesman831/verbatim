"""V8 graph-lane verification tests — SPEC_V8 §05 / §24 / §25 scenarios
K21–K25.

These exercise ``verbatim/retrieval/v7/graph.py::lane_graph`` against a
real SQLite mirror of the §30 minimum schema (``units``/``graph_edges``/
``entity_*``/``source_revisions``) — no mocks.  Scenario coverage:

- K21: a 400-node traversal issues ≤ 8 ``units`` statements (trace
  callback counted) — the D8-01 N+1 is dead.
- K22: with the eligibility set unavailable/invalid the lane returns
  ``status="error"``, zero candidates, and issues zero traversal
  statements — it never traverses unfiltered.
- K23: a node with 50 outgoing edges contributes exactly ``K_fan``
  edges, in identical order across runs; the ``graph.K_fan`` arm is
  real.
- K24: the deadline check fires at 64-expanded-node granularity; a cut
  returns ``partial/deadline`` with ``nodes_expanded``/
  ``rounds_completed`` and a best-so-far ranking, and no statement
  starts after a failed check.
- K25: the scheduler-plumbed need gate (``graph.gate_decision``) skips
  with ``not_needed`` when disarmed by the decision and runs otherwise.
"""

from __future__ import annotations

import json
import re
import sqlite3

import pytest

from verbatim.core.types_v7 import (
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
from verbatim.retrieval.v7 import graph as graph_lane
from verbatim.retrieval.v7.graph import lane_graph

SCOPE = "scope-a"
HOUR = 3_600_000_000
DAY = 24 * HOUR
T0 = 1_700_000_000_000_000

MIRROR_DDL = """
-- MIRROR of SPEC_V7 §30 (minimum columns) — the same shape sibling
-- lane tests use; the lane only reads the declared columns.
CREATE TABLE units (
    unit_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    scope_id TEXT NOT NULL,
    kind TEXT,
    parent_unit_id TEXT,
    session_id TEXT,
    seq INTEGER,
    speaker_canon TEXT,
    perspective TEXT,
    recorded_at_us INTEGER,
    occurred_start_us INTEGER,
    occurred_end_us INTEGER,
    occurred_precision TEXT,
    occurred_source TEXT,
    byte_start INTEGER,
    byte_end INTEGER,
    generation INTEGER NOT NULL,
    PRIMARY KEY (unit_id, generation)
);
CREATE TABLE graph_edges (
    scope_id TEXT NOT NULL,
    src_unit TEXT NOT NULL,
    type TEXT NOT NULL,
    dst_unit TEXT NOT NULL,
    weight REAL NOT NULL,
    evidence_ref TEXT,
    generation INTEGER NOT NULL,
    PRIMARY KEY (scope_id, src_unit, type, dst_unit, generation)
);
CREATE INDEX idx_ge_src ON graph_edges(scope_id, src_unit, type);
CREATE INDEX idx_ge_dst ON graph_edges(scope_id, dst_unit, type);
CREATE TABLE entity_mentions (
    scope_id TEXT NOT NULL,
    canon TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    surface TEXT,
    byte_start INTEGER,
    byte_end INTEGER,
    role TEXT,
    generation INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (scope_id, canon, unit_id, byte_start, generation)
);
CREATE TABLE entity_canon (
    scope_id TEXT NOT NULL,
    canon TEXT NOT NULL,
    display TEXT,
    df_units INTEGER,
    kind TEXT,
    first_seen_us INTEGER,
    last_seen_us INTEGER,
    generation INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (scope_id, canon, generation)
);
CREATE TABLE source_revisions (
    source_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    payload BLOB NOT NULL,
    PRIMARY KEY (source_id, revision)
);
"""


def make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(MIRROR_DDL)
    return conn


def add_unit(
    conn: sqlite3.Connection,
    uid: str,
    text: str | None = None,
    *,
    scope: str = SCOPE,
    gen: int = 1,
    session: str | None = None,
    seq: int | None = None,
    recorded: int = T0,
) -> None:
    src = f"src-{uid}"
    payload = text.encode("utf-8") if text is not None else b""
    conn.execute(
        "INSERT OR REPLACE INTO source_revisions VALUES (?,?,?)",
        (src, 1, payload),
    )
    conn.execute(
        "INSERT OR REPLACE INTO units (unit_id, source_id, revision,"
        " scope_id, kind, parent_unit_id, session_id, seq, speaker_canon,"
        " perspective, recorded_at_us, occurred_start_us, occurred_end_us,"
        " occurred_precision, occurred_source, byte_start, byte_end,"
        " generation) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            uid, src, 1, scope, "turn", None, session, seq, None,
            "user_stated", recorded, None, None,
            "unknown", "unknown",
            0 if text is not None else None,
            len(payload) if text is not None else None,
            gen,
        ),
    )


def put_edge(conn, src, dst, type_, weight=1.0, *, scope=SCOPE, gen=1):
    conn.execute(
        "INSERT OR REPLACE INTO graph_edges"
        " (scope_id, src_unit, type, dst_unit, weight, evidence_ref,"
        " generation) VALUES (?,?,?,?,?,?,?)",
        (scope, src, type_, dst, weight, "{}", gen),
    )


def eligible_units(conn, scope: str = SCOPE, gen: int = 1) -> frozenset:
    """The request's eligible-``unit_id`` set — the test-side mirror of
    ``_Eligible.unit_ids`` (V8-05.02: set membership only)."""
    rows = conn.execute(
        "SELECT DISTINCT unit_id FROM units"
        " WHERE scope_id=? AND generation<=?",
        (scope, gen),
    ).fetchall()
    return frozenset(r[0] for r in rows)


def mk_ctx(
    conn,
    *,
    eligible=None,
    scope: str = SCOPE,
    generation: int = 1,
    manifest: dict | None = None,
) -> LaneContextV7:
    if eligible is None:
        eligible = eligible_units(conn, scope, generation)
    return LaneContextV7(
        store=conn,
        scope_id=scope,
        generation=generation,
        eligible=eligible,
        query_time_us=T0 + 30 * DAY,
        profile="test",
        budget=BudgetClass.MID,
        policy=RetrievalPolicyV7(
            policy_id="test/policy",
            profile="test",
            lanes=(LaneName.GRAPH,),
            lane_weights={},
        ),
        manifest=dict(manifest or {}),
    )


def mk_qv(query: str, canons: tuple[str, ...] = ()) -> QueryViewV7:
    terms = tuple(
        NormTerm(term=t, channel="text", byte_start=0, byte_end=len(t))
        for t in query.split()
    )
    return QueryViewV7(
        query=query,
        norm=NormAnalysis(analyzer_id="norm/v2", terms=terms,
                          identifiers=()),
        intent=IntentResult(
            primary=IntentClass.LOOKUP, classes=(IntentClass.LOOKUP,)
        ),
        entity_canons=canons,
        query_time_us=T0 + 30 * DAY,
    )


def _slice(ms=10_000.0, cap=50):
    return LaneSlice(deadline_ms=ms, cap=cap)


_UNITS_STMT = re.compile(
    r"\bFROM\s+units\b|\bINTO\s+units\b|\bUPDATE\s+units\b"
)
_EDGES_STMT = re.compile(r"\bFROM\s+graph_edges\b")


def _counting_trace(conn):
    """Attach a trace callback counting units/graph_edges statements;
    returns the counter dict."""
    counts = {"units": 0, "graph_edges": 0, "total": 0}

    def cb(sql):
        counts["total"] += 1
        if _UNITS_STMT.search(sql):
            counts["units"] += 1
        if _EDGES_STMT.search(sql):
            counts["graph_edges"] += 1

    conn.set_trace_callback(cb)
    return counts


def tree_fixture(conn, fan: int = 8, n: int = 500, prefix: str = "n"):
    """Deterministic ``fan``-ary tree: node i → children fan·i+1..fan·i+fan.
    BFS from the root reaches 8 nodes in round 1, 64 in round 2 and the
    rest of a 500-node graph in round 3 — past the 400 frontier cap while
    every out-degree stays within ``K_fan``."""
    for i in range(n):
        add_unit(conn, f"{prefix}{i}", "x")
    for i in range(n):
        for c in range(fan * i + 1, min(fan * i + fan + 1, n)):
            put_edge(conn, f"{prefix}{i}", f"{prefix}{c}",
                     "co_mention", 0.5)


# ---------------------------------------------------------------------------
# K21 — statement bound (V8-05.02; D8-01)
# ---------------------------------------------------------------------------


def test_k21_400_node_traversal_units_statements_le_8():
    """K21: traversing 400 nodes issues ≤ 8 ``units`` statements, counted
    by the trace callback — eligibility and metadata ride the batched
    row reads, never per-node SELECTs."""
    conn = make_conn()
    tree_fixture(conn)
    ctx = mk_ctx(conn)
    counts = _counting_trace(conn)
    out = lane_graph(ctx, mk_qv("q"), _slice(cap=10_000), seeds=["n0"])
    conn.set_trace_callback(None)
    assert out.stats["visited"] == graph_lane.FRONTIER_CAP_V1 == 400
    assert 0 < counts["units"] <= 8
    # sanity: the batched rows really fed candidate pins
    assert all(c.source_id for c in out.candidates)


# ---------------------------------------------------------------------------
# K22 — eligibility-set fail-closed (V8-05.02)
# ---------------------------------------------------------------------------


def test_k22_missing_eligibility_set_fails_closed():
    """K22: no usable eligible set → ``status=error``, zero candidates,
    and not one traversal statement (``units`` or ``graph_edges``)."""
    conn = make_conn()
    for u in ("A", "B"):
        add_unit(conn, u, "x")
    put_edge(conn, "A", "B", "co_mention", 0.9)

    for bad in (None, lambda row: True, object(), 42):
        ctx = mk_ctx(conn)
        ctx.eligible = bad  # replace the fixture's set post-construction
        counts = _counting_trace(conn)
        out = lane_graph(ctx, mk_qv("q"), _slice(), seeds=["A"])
        conn.set_trace_callback(None)
        assert out.status.value == "error", (bad, out.status)
        assert out.candidates == []
        assert out.reason == "eligibility_set_unavailable"
        assert out.stats["eligibility"] == "unavailable"
        assert counts["units"] == 0 and counts["graph_edges"] == 0


def test_k22_eligibility_set_forms():
    """K22 (positive half): the production set forms all resolve — an
    ``.unit_ids`` adapter, set/frozenset/dict/list/tuple containers, and
    a ``__contains__`` probe.  Membership is the whole authorization."""
    conn = make_conn()
    for u in ("A", "B", "C"):
        add_unit(conn, u, "x")
    put_edge(conn, "A", "B", "co_mention", 0.9)
    put_edge(conn, "B", "C", "co_mention", 0.9)

    class UnitIdsAdapter:
        @property
        def unit_ids(self):
            return frozenset({"A", "B"})

    for elig in (
        UnitIdsAdapter(),
        {"A", "B"},
        frozenset({"A", "B"}),
        {"A": True, "B": True},          # dict → keys are the set
        ["A", "B"],
        ("A", "B"),
    ):
        out = lane_graph(
            mk_ctx(conn, eligible=elig), mk_qv("q"), _slice(), seeds=["A"]
        )
        assert out.status is LaneStatus.OK, (type(elig), out.status)
        # C is reachable through B but absent from the set → never
        # traversed; the hold is set membership, not SQL (V7-08.11).
        assert {c.unit_id for c in out.candidates} == {"B"}, type(elig)

    # a probing fault denies membership (fail closed per node)
    class FaultyProbe:
        def __contains__(self, uid):
            raise RuntimeError("probe fault")

    out = lane_graph(
        mk_ctx(conn, eligible=FaultyProbe()),
        mk_qv("q"), _slice(), seeds=["A"],
    )
    assert out.status is LaneStatus.SKIPPED  # seed itself is denied
    assert out.reason == "no_eligible_seeds"
    assert out.candidates == []


# ---------------------------------------------------------------------------
# K23 — deterministic fan-out cap (V8-05.03)
# ---------------------------------------------------------------------------


def test_k23_fan_out_cap_deterministic():
    """K23: a node with 50 outgoing edges contributes exactly ``K_fan``
    edges — the top-K by (weight DESC, dst ASC) — in identical order
    across runs; ``graph.K_fan`` is a real arm."""
    conn = make_conn()
    add_unit(conn, "S", "x")
    for i in range(50):
        add_unit(conn, f"d{i:02d}", "x")
        # weight descending with i; dst ids ascending — the fan order is
        # exercised on both keys.
        put_edge(conn, "S", f"d{i:02d}", "co_mention", 1.0 - i * 0.01)
    ctx = mk_ctx(conn)
    out = lane_graph(ctx, mk_qv("q"), _slice(cap=100), seeds=["S"])
    # round 1: S contributes exactly K_fan edges (children's round-2
    # read-backs are their own K_fan-bounded reads).
    assert out.stats["edges_by_round"][1] == graph_lane.K_FAN_V8 == 8
    expect = {f"d{i:02d}" for i in range(8)}  # top-8 weights
    assert {c.unit_id for c in out.candidates} == expect
    out2 = lane_graph(ctx, mk_qv("q"), _slice(cap=100), seeds=["S"])
    assert [c.unit_id for c in out.candidates] == [
        c.unit_id for c in out2.candidates
    ]
    # the arm is real: K_fan=4 halves the contribution
    out4 = lane_graph(
        mk_ctx(conn, manifest={"graph.K_fan": 4}),
        mk_qv("q"), _slice(cap=100), seeds=["S"],
    )
    assert out4.stats["edges_by_round"][1] == 4
    assert {c.unit_id for c in out4.candidates} == {
        f"d{i:02d}" for i in range(4)
    }


# ---------------------------------------------------------------------------
# K24 — deadline granularity (V8-05.04)
# ---------------------------------------------------------------------------


def _wide_fixture(conn, n_seeds: int):
    """``n_seeds`` seeds each with ``K_fan`` distinct children — round-1
    admission candidate count is ``n_seeds * K_fan`` so the in-loop
    deadline check (every 64 expanded nodes) fires inside round 1."""
    seeds = [f"s{i}" for i in range(n_seeds)]
    for s in seeds:
        add_unit(conn, s, "x")
    for i, s in enumerate(seeds):
        for j in range(graph_lane.K_FAN_V8):
            child = f"c{i:02d}-{j}"
            add_unit(conn, child, "x")
            put_edge(conn, s, child, "co_mention", 0.9)
    return seeds


def test_k24_deadline_check_every_64_nodes(monkeypatch):
    """K24 / V8-05.04: the in-loop deadline check runs once per 64
    expanded nodes.  Expiring the cooperative clock at the first check
    admits exactly 63 round-1 nodes; at the second check, 127 — the
    cadence delta is the declared ``_DEADLINE_CHECK_EVERY = 64``.
    Every exit is ``partial/deadline`` with honest expansion stats and
    no statement starts after the failed check (the induced-subgraph
    read is skipped, so ``stats['edges'] == 0``)."""
    assert graph_lane._DEADLINE_CHECK_EVERY == 64

    # -- expiry at the first in-loop check (expanded == 63) ------------
    # cut() call order for this fixture: _Deadline construction, entry
    # check, seed-row fill, round top, pre-src read, pre-dst read,
    # candidate-row fill → the 8th _monotonic read is the first in-loop
    # check.  7 zero ticks then expiry.
    conn = make_conn()
    seeds = _wide_fixture(conn, 10)  # 80 round-1 candidates
    ctx = mk_ctx(conn)
    ticks = iter([0.0] * 7 + [1e9] * 200)
    monkeypatch.setattr(graph_lane, "_monotonic", lambda: next(ticks))
    counts = _counting_trace(conn)
    out = lane_graph(
        ctx, mk_qv("q"), _slice(ms=1.0, cap=500), seeds=seeds
    )
    conn.set_trace_callback(None)
    assert out.status is LaneStatus.PARTIAL
    assert out.reason == "deadline"
    assert out.stats["deadline"] is True
    assert out.stats["frontier_truncated"] is True
    assert out.stats["nodes_expanded"] == 63
    assert out.stats["graph"]["nodes_expanded"] == 63
    assert out.stats["rounds_completed"] == 0
    assert out.stats["visited"] == len(seeds) + 63
    # best-so-far ranking still delivered: the 63 admitted nodes emit.
    assert len(out.candidates) == 63
    assert [c.rank for c in out.candidates] == list(range(1, 64))
    # no statement started after the failed check — the W-matrix
    # induced-subgraph read (a third graph_edges statement) never ran.
    assert out.stats["edges"] == 0
    assert counts["graph_edges"] == 2   # src + dst round-1 reads only
    assert counts["units"] == 2         # seed fill + candidate fill

    # -- expiry at the second in-loop check (expanded == 127) ----------
    conn = make_conn()
    seeds = _wide_fixture(conn, 16)  # 128 round-1 candidates
    ticks = iter([0.0] * 8 + [1e9] * 200)  # first in-loop check passes
    monkeypatch.setattr(graph_lane, "_monotonic", lambda: next(ticks))
    out = lane_graph(
        mk_ctx(conn), mk_qv("q"), _slice(ms=1.0, cap=500), seeds=seeds
    )
    assert out.status is LaneStatus.PARTIAL
    assert out.stats["nodes_expanded"] == 127
    assert out.stats["visited"] == len(seeds) + 127
    # the cadence between checks is the declared 64-node granularity
    assert 127 - 63 == graph_lane._DEADLINE_CHECK_EVERY


# ---------------------------------------------------------------------------
# K25 — need gate (V8-05.05)
# ---------------------------------------------------------------------------


def test_k25_need_gate_skips_and_runs():
    """K25: the scheduler-plumbed decision is honored — ``needed=False``
    (a single-canon lookup whose core union already covers the limit)
    skips with ``not_needed`` and logs the inputs; a comparison-style
    decision runs the lane; the disarmed flag ignores the decision."""
    conn = make_conn()
    for u in ("A", "B"):
        add_unit(conn, u, "x")
    put_edge(conn, "A", "B", "co_mention", 0.9)

    ctx = mk_ctx(conn, manifest={
        "graph.gate_decision": {
            "needed": False,
            "inputs": {"intent": "lookup", "core_union": 40, "limit": 8},
        },
    })
    out = lane_graph(ctx, mk_qv("q"), _slice(), seeds=["A"])
    assert out.status is LaneStatus.SKIPPED
    assert out.reason == "not_needed"
    assert out.candidates == []
    gate = out.stats["gate"]
    assert gate["armed"] is True and gate["ran"] is False
    assert gate["decision"]["needed"] is False
    assert gate["decision"]["inputs"]["core_union"] == 40

    # needed=True (comparison-style), bare-bool shorthand, and malformed
    # decisions all run the lane — the gate is an optimization, never an
    # authorization boundary.
    for dec in ({"needed": True, "inputs": {"intent": "comparison"}},
                True, {"inputs": {}}, "junk"):
        ctx2 = mk_ctx(conn, manifest={"graph.gate_decision": dec})
        out2 = lane_graph(ctx2, mk_qv("q"), _slice(), seeds=["A"])
        assert out2.status is LaneStatus.OK, dec
        assert out2.stats["gate"]["ran"] is True
        assert {c.unit_id for c in out2.candidates} == {"B"}

    # the flag disarmed ignores the decision entirely
    ctx3 = mk_ctx(conn, manifest={
        "graph.gate": False,
        "graph.gate_decision": {"needed": False, "inputs": {}},
    })
    out3 = lane_graph(ctx3, mk_qv("q"), _slice(), seeds=["A"])
    assert out3.status is LaneStatus.OK
    assert out3.stats["gate"]["armed"] is False
    assert {c.unit_id for c in out3.candidates} == {"B"}
