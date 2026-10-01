"""Q6 ablation prep (V7-08.12, SPEC_V7_5 §05 row Q6): per-hop discovery
instrumentation + the ``graph_max_hops`` hop-limit knob on the graph lane.

Covers: ``stats["hop_hist"]`` first-discovery-depth counts (seeds = 0),
``stats["hop2_only"]`` (emitted candidates unreachable at max_hops=1),
cap truncation at the stated depth, cap=0's honest disabled status, and
bit-identical default output when the knob is absent.

Fixture note: ``units``/``graph_edges`` mirror the §30 minimum columns —
same INTEGRATION SWAP situation as ``test_graph.py`` (schema_v7 is owned
by a concurrent worker).
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v7 import (
    BudgetClass,
    IntentClass,
    IntentResult,
    LaneContextV7,
    LaneName,
    LaneSlice,
    LaneStatus,
    NormAnalysis,
    QueryViewV7,
    RetrievalPolicyV7,
)
from verbatim.retrieval.v7.graph import (
    GRAPH_MAX_HOPS_KEY,
    lane_graph,
)

SCOPE = "scope-a"
T0 = 1_700_000_000_000_000

MIRROR_DDL = """
-- MIRROR of SPEC_V7 §30 (minimum columns). INTEGRATION SWAP: schema_v7.
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
"""


def make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(MIRROR_DDL)
    return conn


def add_unit(conn: sqlite3.Connection, uid: str, *, gen: int = 1) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO units (unit_id, source_id, revision,"
        " scope_id, kind, generation) VALUES (?,?,?,?,?,?)",
        (uid, f"src-{uid}", 1, SCOPE, "turn", gen),
    )


def put_edge(
    conn, src: str, dst: str, type_: str = "co_mention",
    weight: float = 1.0, *, gen: int = 1,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO graph_edges"
        " (scope_id, src_unit, type, dst_unit, weight, evidence_ref,"
        " generation) VALUES (?,?,?,?,?,?,?)",
        (SCOPE, src, type_, dst, weight, json.dumps({}), gen),
    )


def mk_ctx(
    conn, *, manifest: dict | None = None, generation: int = 1
) -> LaneContextV7:
    eligible = frozenset(
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT unit_id FROM units"
            " WHERE scope_id=? AND generation<=?",
            (SCOPE, generation),
        )
    )
    return LaneContextV7(
        store=conn,
        scope_id=SCOPE,
        generation=generation,
        eligible=eligible,
        query_time_us=T0,
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


def mk_qv() -> QueryViewV7:
    return QueryViewV7(
        query="q",
        norm=NormAnalysis(analyzer_id="norm/v2", terms=(), identifiers=()),
        intent=IntentResult(
            primary=IntentClass.LOOKUP, classes=(IntentClass.LOOKUP,)
        ),
        entity_canons=(),
        query_time_us=T0,
    )


def _slice(ms: float = 10_000.0, cap: int = 50) -> LaneSlice:
    return LaneSlice(deadline_ms=ms, cap=cap)


def chain_fixture(conn) -> None:
    """S --A--> B --> C (a 3-deep chain) plus S --> D (a second hop-1 hit).

    First-discovery depths: A=1, D=1, B=2, C=3. All edges are the same
    type so depth is purely topological.
    """
    for u in ("S", "A", "B", "C", "D"):
        add_unit(conn, u)
    put_edge(conn, "S", "A")
    put_edge(conn, "S", "D")
    put_edge(conn, "A", "B")
    put_edge(conn, "B", "C")


def candidate_view(out):
    """The comparison surface for identical-output checks."""
    return [
        (c.unit_id, c.rank, c.raw_score, c.signals) for c in out.candidates
    ]


# ---------------------------------------------------------------------------
# hop_hist / hop2_only instrumentation
# ---------------------------------------------------------------------------


def test_hop_hist_first_discovery_depths():
    conn = make_conn()
    chain_fixture(conn)
    out = lane_graph(mk_ctx(conn), mk_qv(), _slice(), seeds=["S"])
    assert out.status is LaneStatus.OK
    # visited = S(0), A(1), D(1), B(2), C(3)
    assert out.stats["hop_hist"] == {0: 1, 1: 2, 2: 1, 3: 1}
    assert sum(out.stats["hop_hist"].values()) == out.stats["visited"]
    # per-candidate depth agrees with the histogram (signals["hops"]
    # already carries first-discovery depth)
    assert {
        c.unit_id: c.signals["hops"] for c in out.candidates
    } == {"A": 1, "D": 1, "B": 2, "C": 3}


def test_hop2_only_counts_genuine_deep_discoveries():
    conn = make_conn()
    chain_fixture(conn)
    out = lane_graph(mk_ctx(conn), mk_qv(), _slice(), seeds=["S"])
    # emitted candidates A(1), D(1), B(2), C(3) — only B and C needed ≥2
    # expansion rounds, so exactly these vanish under max_hops=1.
    assert out.stats["hop2_only"] == 2
    ids = {c.unit_id for c in out.candidates}
    assert ids == {"A", "B", "C", "D"}


def test_first_discovery_not_shortest_after_later_edge():
    """hop_hist records FIRST discovery: a hop-1 shortcut edge reclassifies
    C even though a longer path to it also exists."""
    conn = make_conn()
    chain_fixture(conn)
    put_edge(conn, "S", "C")  # C now discovered at depth 1
    out = lane_graph(mk_ctx(conn), mk_qv(), _slice(), seeds=["S"])
    assert out.stats["hop_hist"] == {0: 1, 1: 3, 2: 1}
    assert out.stats["hop2_only"] == 1  # only B still needs 2 hops
    c = next(c for c in out.candidates if c.unit_id == "C")
    assert c.signals["hops"] == 1


def test_hop2_only_respects_emitted_not_visited():
    """hop2_only is over *emitted* candidates (post slice.cap): a hop-2 unit
    cut by the pool cap never reaches packs and is not counted."""
    conn = make_conn()
    add_unit(conn, "S")
    add_unit(conn, "A")
    add_unit(conn, "B")
    # order visit so the hop-2 unit is the highest-scored non-seed:
    # A is hop-1, B is hop-2 via A. cap=1 keeps only the top candidate.
    put_edge(conn, "S", "A", weight=0.1)
    put_edge(conn, "A", "B", weight=10.0)
    out = lane_graph(mk_ctx(conn), mk_qv(), _slice(cap=1), seeds=["S"])
    assert len(out.candidates) == 1
    top = out.candidates[0].unit_id
    # whichever of A/B out-scored the other, hop2_only counts it only if
    # it was emitted and first discovered at depth >= 2
    expected = 1 if top == "B" else 0
    assert out.stats["hop2_only"] == expected
    assert out.stats["hop_hist"] == {0: 1, 1: 1, 2: 1}


# ---------------------------------------------------------------------------
# graph_max_hops knob
# ---------------------------------------------------------------------------


def test_cap_1_truncates_at_one_hop():
    conn = make_conn()
    chain_fixture(conn)
    out = lane_graph(
        mk_ctx(conn, manifest={GRAPH_MAX_HOPS_KEY: 1}),
        mk_qv(), _slice(), seeds=["S"],
    )
    assert out.status is LaneStatus.OK
    assert {c.unit_id for c in out.candidates} == {"A", "D"}
    assert out.stats["hop_cap"] == 1
    assert out.stats["rounds_run"] == 1
    assert out.stats["hop_hist"] == {0: 1, 1: 2}
    assert out.stats["hop2_only"] == 0


def test_cap_2_truncates_at_two_hops():
    conn = make_conn()
    chain_fixture(conn)
    out = lane_graph(
        mk_ctx(conn, manifest={GRAPH_MAX_HOPS_KEY: 2}),
        mk_qv(), _slice(), seeds=["S"],
    )
    assert {c.unit_id for c in out.candidates} == {"A", "B", "D"}
    assert out.stats["hop_cap"] == 2
    assert out.stats["rounds_run"] == 2
    assert out.stats["hop_hist"] == {0: 1, 1: 2, 2: 1}
    assert out.stats["hop2_only"] == 1


def test_cap_0_emits_nothing_with_honest_status():
    conn = make_conn()
    chain_fixture(conn)
    out = lane_graph(
        mk_ctx(conn, manifest={GRAPH_MAX_HOPS_KEY: 0}),
        mk_qv(), _slice(), seeds=["S"],
    )
    assert out.status is LaneStatus.SKIPPED
    assert out.reason == f"disabled:{GRAPH_MAX_HOPS_KEY}=0"
    assert out.candidates == []
    assert out.stats["hop_cap"] == 0
    # the seed phase still ran and reported honestly
    assert out.stats["seeds"] == 1
    assert out.stats["eligible_seeds"] == 1
    assert out.stats["hop_hist"] == {0: 1}
    assert out.stats["hop2_only"] == 0


def test_cap_above_declared_bound_clamps():
    conn = make_conn()
    chain_fixture(conn)
    out = lane_graph(
        mk_ctx(conn, manifest={GRAPH_MAX_HOPS_KEY: 9}),
        mk_qv(), _slice(), seeds=["S"],
    )
    assert out.stats["hop_cap"] == 3  # EXPANSION_ROUNDS_V1 is the bound
    assert {c.unit_id for c in out.candidates} == {"A", "B", "C", "D"}


@pytest.mark.parametrize("bad", ["2", -1, 1.5, True, float("inf")])
def test_cap_invalid_values_fail_loudly(bad):
    conn = make_conn()
    chain_fixture(conn)
    with pytest.raises(VerbatimError) as exc:
        lane_graph(
            mk_ctx(conn, manifest={GRAPH_MAX_HOPS_KEY: bad}),
            mk_qv(), _slice(), seeds=["S"],
        )
    assert exc.value.code is ErrorCode.VALIDATION


# ---------------------------------------------------------------------------
# Default-behavior equivalence
# ---------------------------------------------------------------------------


def test_absent_knob_matches_explicit_none_and_bound():
    conn = make_conn()
    chain_fixture(conn)
    base = lane_graph(mk_ctx(conn), mk_qv(), _slice(), seeds=["S"])
    explicit_none = lane_graph(
        mk_ctx(conn, manifest={GRAPH_MAX_HOPS_KEY: None}),
        mk_qv(), _slice(), seeds=["S"],
    )
    explicit_bound = lane_graph(
        mk_ctx(conn, manifest={GRAPH_MAX_HOPS_KEY: 3}),
        mk_qv(), _slice(), seeds=["S"],
    )
    assert base.stats["hop_cap"] is None
    # candidates and every stat except the self-describing hop_cap are
    # identical across absent / null / at-bound configurations
    assert candidate_view(base) == candidate_view(explicit_none)
    assert candidate_view(base) == candidate_view(explicit_bound)
    for key, val in base.stats.items():
        if key == "hop_cap":
            continue
        assert explicit_none.stats[key] == val
        assert explicit_bound.stats[key] == val


def test_existing_stats_keys_unchanged():
    """The instrumentation only *adds* keys — the pre-existing stats surface
    is untouched (V7-08.12 prep is additive, not a rewrite)."""
    conn = make_conn()
    chain_fixture(conn)
    out = lane_graph(mk_ctx(conn), mk_qv(), _slice(), seeds=["S"])
    for key in (
        "formula", "formula_status", "seed_source", "seeds",
        "eligible_seeds", "rounds_run", "visited", "frontier_truncated",
        "edges", "iterations", "seed_scores",
    ):
        assert key in out.stats
    assert out.stats["rounds_run"] == 3
    assert out.stats["visited"] == 5
