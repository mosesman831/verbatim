"""w-graph tests: edge derivation (V7-08.07/10) + bounded PPR lane
(V7-08.08/09/11, §32.6 graph_ppr/v1).

Mirror-DDL note: ``units``/``graph_edges``/``entity_*``/``source_revisions``
below mirror the §30 minimum columns because ``schema_v7`` is owned by a
concurrent worker — INTEGRATION SWAP: replace ``MIRROR_DDL`` with
``verbatim.storage.schema_v7.ensure_v7_additive(conn)`` once it lands.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3

import pytest

from verbatim.core.types import VerbatimError, safe_json_loads
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
from verbatim.jobs import graph_jobs
from verbatim.jobs.graph_jobs import build_edges, write_supplied_edges
from verbatim.retrieval.v7 import graph as graph_lane
from verbatim.retrieval.v7.graph import lane_graph

SCOPE = "scope-a"
HOUR = 3_600_000_000
DAY = 24 * HOUR
T0 = 1_700_000_000_000_000  # fixed anchor for occurred intervals

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
    occ: tuple[int, int] | None = None,
    canons: tuple[str, ...] = (),
    recorded: int = T0,
    speaker: str | None = None,
) -> dict:
    """Insert a units row + its payload; returns the dict form suitable for
    ``build_edges``' ``unit_rows``. Payloads carry a 4-byte header so byte
    pins exercise the ``byte_start`` offset path."""
    src = f"src-{uid}"
    payload = b""
    bs = be = None
    if text is not None:
        payload = b"Hdr|" + text.encode("utf-8")
        bs, be = 4, len(payload)
    conn.execute(
        "INSERT OR REPLACE INTO source_revisions VALUES (?,?,?)",
        (src, 1, payload),
    )
    os_, oe = occ if occ else (None, None)
    conn.execute(
        "INSERT OR REPLACE INTO units (unit_id, source_id, revision,"
        " scope_id, kind, parent_unit_id, session_id, seq, speaker_canon,"
        " perspective, recorded_at_us, occurred_start_us, occurred_end_us,"
        " occurred_precision, occurred_source, byte_start, byte_end,"
        " generation) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            uid, src, 1, scope, "turn", None, session, seq, speaker,
            "user_stated", recorded, os_, oe,
            "day" if occ else "unknown",
            "explicit" if occ else "unknown",
            bs, be, gen,
        ),
    )
    for c in canons:
        conn.execute(
            "INSERT OR REPLACE INTO entity_mentions"
            " (scope_id, canon, unit_id, byte_start, generation)"
            " VALUES (?,?,?,?,?)",
            (scope, c, uid, 0, gen),
        )
        df = conn.execute(
            "SELECT COUNT(DISTINCT unit_id) FROM entity_mentions"
            " WHERE scope_id=? AND canon=?",
            (scope, c),
        ).fetchone()[0]
        conn.execute(
            "INSERT OR REPLACE INTO entity_canon"
            " (scope_id, canon, generation, df_units) VALUES (?,?,?,?)",
            (scope, c, gen, df),
        )
    return {
        "unit_id": uid,
        "source_id": src,
        "revision": 1,
        "scope_id": scope,
        "kind": "turn",
        "session_id": session,
        "seq": seq,
        "speaker_canon": speaker,
        "recorded_at_us": recorded,
        "occurred_start_us": os_,
        "occurred_end_us": oe,
        "byte_start": bs,
        "byte_end": be,
        "generation": gen,
        "canons": canons,
    }


def edges_of(conn, scope=SCOPE, type_=None):
    q = "SELECT src_unit, type, dst_unit, weight, evidence_ref, generation" \
        " FROM graph_edges WHERE scope_id=?"
    params: list = [scope]
    if type_:
        q += " AND type=?"
        params.append(type_)
    q += " ORDER BY src_unit, type, dst_unit"
    return conn.execute(q, params).fetchall()


def eligible_units(conn, scope: str = SCOPE, gen: int = 1) -> frozenset:
    """The eligible-unit_id set over visible units — the test-side mirror
    of ``_Eligible.unit_ids`` (V8-05.02: the lane consumes set membership
    only; rows outside the generation fence are simply absent)."""
    try:
        rows = conn.execute(
            "SELECT DISTINCT unit_id FROM units"
            " WHERE scope_id=? AND generation<=?",
            (scope, gen),
        ).fetchall()
    except sqlite3.Error:
        return frozenset()
    return frozenset(r[0] for r in rows)


def mk_ctx(
    conn,
    *,
    eligible=None,
    scope: str = SCOPE,
    generation: int = 1,
    query_time_us: int = T0 + 30 * DAY,
    manifest: dict | None = None,
) -> LaneContextV7:
    if eligible is None:
        eligible = eligible_units(conn, scope, generation)
    return LaneContextV7(
        store=conn,
        scope_id=scope,
        generation=generation,
        eligible=eligible,
        query_time_us=query_time_us,
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
        norm=NormAnalysis(analyzer_id="norm/v2", terms=terms, identifiers=()),
        intent=IntentResult(
            primary=IntentClass.LOOKUP, classes=(IntentClass.LOOKUP,)
        ),
        entity_canons=canons,
        query_time_us=T0 + 30 * DAY,
    )


def put_edge(conn, src, dst, type_, weight=1.0, *, scope=SCOPE, gen=1, ev=None):
    conn.execute(
        "INSERT OR REPLACE INTO graph_edges"
        " (scope_id, src_unit, type, dst_unit, weight, evidence_ref, generation)"
        " VALUES (?,?,?,?,?,?,?)",
        (scope, src, type_, dst, weight, json.dumps(ev or {}), gen),
    )


def eligible_all(row) -> bool:
    return True


# ---------------------------------------------------------------------------
# build_edges — derived families
# ---------------------------------------------------------------------------


def test_co_mention_inverse_df_weight():
    conn = make_conn()
    rows = [
        add_unit(conn, f"u{i}", f"text {i}", canons=("promotion",))
        for i in range(3)
    ]
    n = build_edges(conn, SCOPE, 1, rows)
    edges = edges_of(conn, type_="co_mention")
    assert n == 3 and len(edges) == 3  # C(3,2) clique under the fan-out band
    for _s, _t, _d, w, ev, g in edges:
        assert w == pytest.approx(1.0 / 3)  # inverse canon df
        assert g == 1
        assert safe_json_loads(ev)["canon"] == "promotion"


def test_co_mention_requires_shared_canon():
    conn = make_conn()
    a = add_unit(conn, "a", "x", canons=("alpha",))
    b = add_unit(conn, "b", "y", canons=("beta",))
    assert build_edges(conn, SCOPE, 1, [a, b]) == 0


def test_session_edges_adjacency_and_band():
    conn = make_conn()
    rows = [
        add_unit(conn, f"t{i}", f"turn {i}", session="s1", seq=i)
        for i in range(1, 5)
    ]
    n = build_edges(conn, SCOPE, 1, rows)
    same = edges_of(conn, type_="same_session")
    adj = edges_of(conn, type_="adjacent_turn")
    assert len(same) == 6  # all pairs of 4 (within fan-out band)
    # t±1,t±2: (1,2)(1,3)(2,3)(2,4)(3,4)
    assert len(adj) == 5
    assert n == 11
    pairs = {(s, d) for s, _t, d, _w, _e, _g in adj}
    assert ("t1", "t4") not in pairs and ("t1", "t2") in pairs


def test_temporal_near_window():
    conn = make_conn()
    w = graph_jobs.TEMPORAL_NEAR_WINDOW_US_V1
    a = add_unit(conn, "a", occ=(T0, T0 + HOUR))
    b = add_unit(conn, "b", occ=(T0 + 2 * HOUR, T0 + 3 * HOUR))
    c = add_unit(conn, "c", occ=(T0 + 100 * DAY, T0 + 100 * DAY + HOUR))
    d = add_unit(conn, "d")  # unknown occurred — never participates
    build_edges(conn, SCOPE, 1, [a, b, c, d])
    edges = edges_of(conn, type_="temporal_near")
    assert {(s, d_) for s, _t, d_, *_ in edges} == {("a", "b")}
    ev = safe_json_loads(edges[0][4])
    assert ev["gap_us"] == HOUR  # b.start(2h) - a.end(1h)
    assert edges[0][3] == pytest.approx(1.0 - HOUR / w)


def test_causal_self_loop_both_clauses_pinned():
    conn = make_conn()
    text = "He was happy because he got the promotion."
    row = add_unit(conn, "u1", text)
    n = build_edges(conn, SCOPE, 1, [row])
    edges = edges_of(conn, type_="causal")
    assert n == 1 and len(edges) == 1
    src, _t, dst, w, ev, _g = edges[0]
    assert src == dst == "u1"
    ev = safe_json_loads(ev)
    assert ev["connective"] == "because"
    assert ev["self_loop"] is True
    payload = conn.execute(
        "SELECT payload FROM source_revisions WHERE source_id='src-u1'"
    ).fetchone()[0]
    for pin in (ev["effect"], ev["cause"]):
        frag = bytes(payload[pin["byte_start"] : pin["byte_end"]]).decode()
        assert frag == pin["text"]
    assert ev["effect"]["text"] == "He was happy"
    assert ev["cause"]["text"] == "he got the promotion"


def test_causal_cross_unit_cause_resolution():
    conn = make_conn()
    # The cause clause "promotion news" occurs literally in u2 — a lexical
    # lookup, not inference. Both units share canon "promotion" so u2 is in
    # the build context.
    other = add_unit(
        conn, "u2", "Promotion news spread quickly.", canons=("promotion",)
    )
    build_edges(conn, SCOPE, 1, [other])
    row = add_unit(
        conn,
        "u1",
        "He was happy because of the promotion news.",
        canons=("promotion",),
    )
    build_edges(conn, SCOPE, 1, [row])
    edges = edges_of(conn, type_="causal")
    assert len(edges) == 1
    src, _t, dst, _w, ev, _g = edges[0]
    assert (src, dst) == ("u2", "u1")  # cause-unit → effect-unit
    ev = safe_json_loads(ev)
    assert ev["self_loop"] is False
    assert ev["cause"]["unit_id"] == "u2"
    assert ev["effect"]["unit_id"] == "u1"
    p2 = conn.execute(
        "SELECT payload FROM source_revisions WHERE source_id='src-u2'"
    ).fetchone()[0]
    frag = bytes(p2[ev["cause"]["byte_start"] : ev["cause"]["byte_end"]])
    # the pin records the dst unit's own bytes verbatim (folded match,
    # exact-bytes pin)
    assert frag.decode() == ev["cause"]["text"] == "Promotion news"


def test_causal_never_from_co_occurrence():
    conn = make_conn()
    rows = [
        add_unit(conn, "a", "We talked about the promotion.", canons=("promotion",)),
        add_unit(conn, "b", "The promotion was announced.", canons=("promotion",)),
        add_unit(conn, "c", "So the meeting ended early."),  # empty left clause
        add_unit(conn, "d", "I think so, maybe."),  # 'so' mid-clause w/ left
    ]
    build_edges(conn, SCOPE, 1, rows)
    causal = edges_of(conn, type_="causal")
    # 'd' has "I think" left of "so" — explicit connective, both clauses
    # non-empty → that IS a legitimate T0 causal row (self-loop).
    assert {(s, d_) for s, _t, d_, *_ in causal} == {("d", "d")}
    # no causal edge between the merely-co-occurring a/b
    assert not any({s, d_} == {"a", "b"} for s, _t, d_, *_ in causal)


def test_causal_therefore_direction():
    conn = make_conn()
    row = add_unit(conn, "u1", "It rained. Therefore, we stayed home.")
    build_edges(conn, SCOPE, 1, [row])
    (src, _t, dst, _w, ev, _g), = edges_of(conn, type_="causal")
    ev = safe_json_loads(ev)
    assert ev["connective"] == "therefore"
    assert ev["cause"]["text"] == "It rained"
    assert ev["effect"]["text"] == "we stayed home"


def test_supplied_edges_and_validation():
    conn = make_conn()
    add_unit(conn, "a"), add_unit(conn, "b"), add_unit(conn, "c")
    n = write_supplied_edges(
        conn,
        SCOPE,
        1,
        [
            {"type": "supersedes", "src_unit": "a", "dst_unit": "b"},
            {"type": "refines", "src_unit": "b", "dst_unit": "c",
             "weight": 0.5, "evidence": {"note": "x"}},
            {"type": "semantic_knn", "src_unit": "a", "dst_unit": "c",
             "similarity": 0.83},
        ],
    )
    assert n == 3
    types = {t for _s, t, _d, *_ in edges_of(conn)}
    assert types == {"supersedes", "refines", "semantic_knn"}
    # derived types are computed, never supplied
    with pytest.raises(Exception):
        write_supplied_edges(
            conn, SCOPE, 1,
            [{"type": "co_mention", "src_unit": "a", "dst_unit": "b"}],
        )
    # causal_candidate requires both clause pins (V7-08.14)
    with pytest.raises(Exception):
        write_supplied_edges(
            conn, SCOPE, 1,
            [{"type": "causal_candidate", "src_unit": "a", "dst_unit": "b",
              "evidence": {"cause": {"byte_start": 0, "byte_end": 3}}}],
        )
    ok = write_supplied_edges(
        conn, SCOPE, 1,
        [{"type": "causal_candidate", "src_unit": "a", "dst_unit": "b",
          "evidence": {
              "cause": {"byte_start": 0, "byte_end": 5},
              "effect": {"byte_start": 6, "byte_end": 12},
          }}],
    )
    assert ok == 1
    row = edges_of(conn, type_="causal_candidate")[0]
    assert safe_json_loads(row[4])["eligible_input"] is False


def test_knn_list_cap_and_incremental():
    conn = make_conn()
    rec = add_unit(conn, "u0", "base")
    knn = [(f"n{i:02d}", 1.0 - i * 0.01) for i in range(12)]
    rec["knn"] = [{"dst_unit": d, "similarity": s} for d, s in knn]
    for d, _s in knn:
        add_unit(conn, d, "x")
    n = build_edges(conn, SCOPE, 1, [rec])
    knn_edges = edges_of(conn, type_="semantic_knn")
    assert len(knn_edges) == graph_jobs.KNN_MAX_V1 == 8
    # top-8 by similarity; semantic_knn is symmetric so each pair is stored
    # once in canonical (lower, higher) unit_id order
    partners = {
        d if s == "u0" else s for s, _t, d, *_ in knn_edges
    }
    assert partners == {d for d, _s in knn[:8]}
    assert all(
        safe_json_loads(e)["similarity"] == pytest.approx(sim)
        for (_s, _t, _d, _w, e, _g), (_d, sim) in zip(
            knn_edges, sorted(knn[:8], key=lambda t: t[0])
        )
    )


def test_incremental_rebuild_idempotent():
    conn = make_conn()
    a = add_unit(conn, "a", "x", canons=("k",))
    b = add_unit(conn, "b", "y", canons=("k",))
    assert build_edges(conn, SCOPE, 1, [a, b]) == 1
    assert edges_of(conn)[0][3] == pytest.approx(0.5)  # df=2
    c = add_unit(conn, "c", "z", canons=("k",))
    n = build_edges(conn, SCOPE, 1, [c])
    assert n == 2  # only new-incident pairs (a,c),(b,c)
    assert len(edges_of(conn)) == 3
    # rerun is idempotent
    n2 = build_edges(conn, SCOPE, 1, [c])
    assert n2 == 2 and len(edges_of(conn)) == 3


def test_build_edges_accepts_bare_ids():
    conn = make_conn()
    add_unit(conn, "a", "x", canons=("k",))
    add_unit(conn, "b", "y", canons=("k",))
    assert build_edges(conn, SCOPE, 1, ["a", "b"]) == 1


# ---------------------------------------------------------------------------
# lane_graph — bounded PPR
# ---------------------------------------------------------------------------


def _slice(ms=10_000.0, cap=50):
    return LaneSlice(deadline_ms=ms, cap=cap)


def test_ppr_multihop_chain_handcomputed():
    conn = make_conn()
    for u in ("A", "B", "C"):
        add_unit(conn, u, "x")
    put_edge(conn, "A", "B", "supersedes")
    put_edge(conn, "B", "C", "supersedes")
    out = lane_graph(mk_ctx(conn), mk_qv("q"), _slice(), seeds=["A"])
    assert out.status is LaneStatus.OK
    got = {c.unit_id: c.raw_score for c in out.candidates}
    assert set(got) == {"B", "C"}  # seed excluded
    # hand-computed r after 3 iterations, damping 0.5 (see module test plan):
    assert got["B"] == pytest.approx(0.375)
    assert got["C"] == pytest.approx(0.0625)
    b = next(c for c in out.candidates if c.unit_id == "B")
    assert b.rank == 1
    assert [h["unit"] for h in b.signals["path"]] == ["A", "B"]
    assert b.signals["components"]["update"] == pytest.approx(1.0)


def test_ppr_mass_conservation_and_bounds():
    conn = make_conn()
    for u in ("S", "X", "Y", "Z"):
        add_unit(conn, u, "x")
    put_edge(conn, "S", "X", "co_mention", 0.5)
    put_edge(conn, "X", "Y", "co_mention", 0.5)
    put_edge(conn, "S", "Z", "semantic_knn", 0.9)
    ctx = mk_ctx(conn)
    out = lane_graph(ctx, mk_qv("q"), _slice(), seeds=["S"])
    assert out.status is LaneStatus.OK
    assert all(0.0 < c.raw_score <= 1.0 for c in out.candidates)
    # Σr over the whole visited subgraph is conserved: W is row-stochastic
    # and p sums to 1, so seeds' residual mass + candidate mass = 1.
    visited_mass = sum(c.raw_score for c in out.candidates) + sum(
        out.stats["seed_scores"].values()
    )
    assert visited_mass == pytest.approx(1.0)
    ranks = [c.rank for c in out.candidates]
    assert ranks == list(range(1, len(ranks) + 1))


def test_held_unit_breaks_paths():
    """The security property (V7-08.11): a held unit is never traversed, so
    nothing reachable only through it can surface."""
    conn = make_conn()
    for u in ("A", "H", "C", "N"):
        add_unit(conn, u, "x")
    put_edge(conn, "A", "H", "co_mention", 0.9)
    put_edge(conn, "H", "C", "co_mention", 0.9)
    put_edge(conn, "H", "N", "co_mention", 0.9)
    ctx = mk_ctx(conn, eligible=eligible_units(conn) - {"H"})
    out = lane_graph(ctx, mk_qv("q"), _slice(), seeds=["A"])
    assert out.status is LaneStatus.OK
    ids = {c.unit_id for c in out.candidates}
    assert ids == set()  # C and N unreachable; H itself never visited
    assert out.stats["visited"] == 1
    # lifting the hold restores the path — eligibility, not the graph, gated it
    out2 = lane_graph(mk_ctx(conn), mk_qv("q"), _slice(), seeds=["A"])
    assert {c.unit_id for c in out2.candidates} == {"H", "C", "N"}


def test_held_seed_gets_no_mass():
    conn = make_conn()
    for u in ("A", "B", "C"):
        add_unit(conn, u, "x")
    put_edge(conn, "A", "C", "co_mention", 0.5)
    put_edge(conn, "B", "C", "co_mention", 0.5)
    ctx = mk_ctx(conn, eligible=eligible_units(conn) - {"B"})
    out = lane_graph(ctx, mk_qv("q"), _slice(), seeds=["A", "B"])
    assert out.stats["eligible_seeds"] == 1
    assert {c.unit_id for c in out.candidates} == {"C"}


def test_eligible_set_object_form():
    conn = make_conn()
    for u in ("A", "B"):
        add_unit(conn, u, "x")
    put_edge(conn, "A", "B", "co_mention", 0.5)
    ctx = mk_ctx(conn, eligible=frozenset({"A", "B"}))
    out = lane_graph(ctx, mk_qv("q"), _slice(), seeds=["A"])
    assert {c.unit_id for c in out.candidates} == {"B"}


def test_components_decompose():
    conn = make_conn()
    for u in ("S", "X", "M", "Y"):
        add_unit(conn, u, "x")
    put_edge(conn, "S", "X", "semantic_knn", 0.8)
    put_edge(conn, "X", "M", "causal", 1.0)
    put_edge(conn, "S", "Y", "co_mention", 0.4)
    out = lane_graph(mk_ctx(conn), mk_qv("q"), _slice(), seeds=["S"])
    by_id = {c.unit_id: c for c in out.candidates}
    assert set(by_id) == {"X", "M", "Y"}
    # X is reached by the knn edge AND returns causal mass from M — its
    # share decomposes over both families, knn dominant.
    xc = by_id["X"].signals["components"]
    assert xc["knn"] > 0.5 and xc["causal"] > 0.0
    assert xc["knn"] + xc["causal"] == pytest.approx(1.0)
    # M's only inbound family is causal; Y's is co_mention (entity).
    assert by_id["M"].signals["components"]["causal"] == pytest.approx(1.0)
    assert by_id["Y"].signals["components"]["entity"] == pytest.approx(1.0)
    # every component key present for ablatability (V7-08.09)
    assert set(by_id["X"].signals["components"]) == {
        "entity", "knn", "temporal", "causal", "session", "update",
    }


def tree_fixture(conn, fan: int = 8, n: int = 500, prefix: str = "n"):
    """Deterministic ``fan``-ary tree: node i → children fan·i+1..fan·i+fan.

    BFS from the root reaches 8 nodes in round 1, 64 in round 2 and the
    rest of a 500-node graph in round 3 — comfortably past the 400
    frontier cap while every node's out-degree stays within ``K_fan``.
    """
    for i in range(n):
        add_unit(conn, f"{prefix}{i}", "x")
    for i in range(n):
        for c in range(fan * i + 1, min(fan * i + fan + 1, n)):
            put_edge(conn, f"{prefix}{i}", f"{prefix}{c}", "co_mention", 0.5)


def test_frontier_cap_respected():
    conn = make_conn()
    tree_fixture(conn)
    out = lane_graph(
        mk_ctx(conn), mk_qv("q"), _slice(cap=10_000), seeds=["n0"]
    )
    assert out.stats["visited"] == graph_lane.FRONTIER_CAP_V1
    assert out.stats["frontier_truncated"] is True
    assert len(out.candidates) == graph_lane.FRONTIER_CAP_V1 - 1
    assert out.reason == "frontier_cap"


def test_expansion_rounds_bound():
    conn = make_conn()
    chain = ["u0", "u1", "u2", "u3", "u4"]
    for u in chain:
        add_unit(conn, u, "x")
    for a, b in zip(chain, chain[1:]):
        put_edge(conn, a, b, "co_mention", 0.5)
    out = lane_graph(mk_ctx(conn), mk_qv("q"), _slice(), seeds=["u0"])
    reached = {c.unit_id for c in out.candidates}
    assert reached == {"u1", "u2", "u3"}  # ≤3 rounds: u4 out of reach
    assert out.stats["rounds_run"] == 3


def test_determinism_bit_exact():
    conn = make_conn()
    for u in ("S", "a", "b", "c", "d"):
        add_unit(conn, u, "x")
    put_edge(conn, "S", "a", "co_mention", 0.5)
    put_edge(conn, "S", "b", "semantic_knn", 0.7)
    put_edge(conn, "a", "c", "temporal_near", 0.9)
    put_edge(conn, "b", "c", "causal", 1.0)
    put_edge(conn, "c", "d", "same_session", 1.0)
    ctx = mk_ctx(conn)
    r1 = lane_graph(ctx, mk_qv("q"), _slice(), seeds=["S"])
    r2 = lane_graph(ctx, mk_qv("q"), _slice(), seeds=["S"])
    s1 = [(c.unit_id, c.raw_score, c.rank) for c in r1.candidates]
    s2 = [(c.unit_id, c.raw_score, c.rank) for c in r2.candidates]
    assert s1 == s2 and r1.stats == r2.stats


def test_causal_candidate_never_traversed():
    conn = make_conn()
    for u in ("A", "X"):
        add_unit(conn, u, "x")
    put_edge(conn, "A", "X", "causal_candidate", 1.0)
    assert "causal_candidate" not in graph_lane.DECLARED_EDGE_TYPES
    out = lane_graph(mk_ctx(conn), mk_qv("q"), _slice(), seeds=["A"])
    assert out.candidates == []


def test_generation_fence_on_edges_and_units():
    conn = make_conn()
    add_unit(conn, "A", "x"), add_unit(conn, "B", "x")
    put_edge(conn, "A", "B", "co_mention", 0.5, gen=5)
    ctx = mk_ctx(conn, generation=3)
    out = lane_graph(ctx, mk_qv("q"), _slice(), seeds=["A"])
    assert out.candidates == []
    # a unit minted after the snapshot generation is also ineligible
    add_unit(conn, "G", "x", gen=9)
    put_edge(conn, "A", "G", "co_mention", 0.5, gen=1)
    out2 = lane_graph(mk_ctx(conn, generation=3), mk_qv("q"), _slice(), seeds=["A"])
    assert {c.unit_id for c in out2.candidates} == set()


def test_multigeneration_edge_resolution():
    """graph_edges PK ends in generation — during rebuild coexistence both
    generations match ``generation<=snapshot``; the lane must take only the
    latest visible row per logical edge."""
    conn = make_conn()
    for u in ("A", "B", "C"):
        add_unit(conn, u, "x")
    put_edge(conn, "A", "B", "co_mention", 0.5, gen=1)
    put_edge(conn, "A", "C", "co_mention", 0.5, gen=1)
    put_edge(conn, "A", "B", "co_mention", 0.9, gen=3)  # rebuild re-weighted
    ctx = mk_ctx(conn, generation=3)
    out = lane_graph(ctx, mk_qv("q"), _slice(), seeds=["A"])
    r = {c.unit_id: c.raw_score for c in out.candidates}
    assert r["B"] > r["C"]  # gen-3 weight won — no double-count of gen-1
    out1 = lane_graph(
        mk_ctx(conn, generation=1), mk_qv("q"), _slice(), seeds=["A"]
    )
    r1 = {c.unit_id: c.raw_score for c in out1.candidates}
    assert r1["B"] == pytest.approx(r1["C"])  # symmetric before the rebuild


def test_scope_isolation():
    conn = make_conn()
    add_unit(conn, "A", "x"), add_unit(conn, "B", "x", scope="other")
    put_edge(conn, "A", "B", "co_mention", 0.5, scope="other")
    out = lane_graph(mk_ctx(conn), mk_qv("q"), _slice(), seeds=["A"])
    assert out.candidates == []


def test_derived_seeds_from_entity_canons():
    conn = make_conn()
    add_unit(conn, "E1", "talked about promotion", canons=("promotion",))
    add_unit(conn, "E2", "follow-up", canons=())
    put_edge(conn, "E1", "E2", "same_session", 1.0)
    out = lane_graph(
        mk_ctx(conn), mk_qv("promotion stuff", canons=("promotion",)),
        _slice(), seeds=None,
    )
    assert out.stats["seed_source"] == "derived"
    assert out.stats["eligible_seeds"] == 1
    assert {c.unit_id for c in out.candidates} == {"E2"}


def test_no_seeds_skips():
    conn = make_conn()
    add_unit(conn, "A", "x")
    out = lane_graph(mk_ctx(conn), mk_qv("nothing"), _slice(), seeds=[])
    assert out.status is LaneStatus.SKIPPED
    assert out.reason == "no_eligible_seeds"


def test_missing_tables_unavailable():
    conn = sqlite3.connect(":memory:")
    out = lane_graph(mk_ctx(conn), mk_qv("q"), _slice(), seeds=["A"])
    assert out.status is LaneStatus.UNAVAILABLE
    assert out.reason


def test_deadline_at_entry():
    conn = make_conn()
    add_unit(conn, "A", "x")
    out = lane_graph(mk_ctx(conn), mk_qv("q"), _slice(ms=0.0), seeds=["A"])
    assert out.status is LaneStatus.DEADLINE
    assert out.reason == "deadline"


def test_deadline_mid_expansion_partial(monkeypatch):
    conn = make_conn()
    chain = ["u0", "u1", "u2", "u3"]
    for u in chain:
        add_unit(conn, u, "x")
    for a, b in zip(chain, chain[1:]):
        put_edge(conn, a, b, "co_mention", 0.5)
    # V8-05.04 deadline checks — one ``_monotonic`` read at _Deadline
    # construction, then one per gate: entry, seed-row fetch, round top,
    # pre-src-fetch, pre-dst-fetch, candidate-row fetch; the 8th read is
    # the round-2 top check — expire there so exactly u1 was admitted.
    ticks = iter([0.0] * 7 + [1e9] * 100)
    monkeypatch.setattr(graph_lane, "_monotonic", lambda: next(ticks))
    out = lane_graph(
        mk_ctx(conn), mk_qv("q"), _slice(ms=1.0), seeds=["u0"]
    )
    assert out.status is LaneStatus.PARTIAL
    assert out.reason == "deadline"
    assert out.stats["deadline"] is True
    # only round-1 neighbours were visited before the clock expired
    assert {c.unit_id for c in out.candidates} == {"u1"}


def test_cap_and_lane_metadata():
    conn = make_conn()
    for u in ("S", "a", "b", "c"):
        add_unit(conn, u, "x")
    for v in ("a", "b", "c"):
        put_edge(conn, "S", v, "co_mention", 0.5)
    out = lane_graph(mk_ctx(conn), mk_qv("q"), _slice(cap=2), seeds=["S"])
    assert len(out.candidates) == 2
    assert all(c.lane == "graph" for c in out.candidates)
    assert out.stats["formula"] == "graph_ppr/v1"
    assert out.stats["formula_status"] == "provisional/v7-r0"


def test_end_to_end_build_then_traverse():
    """Edges derived by build_edges feed the lane: session chain surfaces a
    2-hop unit for a seed-anchored query."""
    conn = make_conn()
    rows = [
        add_unit(conn, "t1", "I joined the team.", session="s", seq=1,
                 canons=("team",)),
        add_unit(conn, "t2", "Onboarding was smooth.", session="s", seq=2),
        add_unit(conn, "t3", "The mentor helped.", session="s", seq=3),
    ]
    assert build_edges(conn, SCOPE, 1, rows) > 0
    out = lane_graph(mk_ctx(conn), mk_qv("q"), _slice(), seeds=["t1"])
    ids = {c.unit_id for c in out.candidates}
    assert "t2" in ids and "t3" in ids


# ---------------------------------------------------------------------------
# V8 batched traversal (V8-05.02–05.09, §21.3, scenarios K21–K29)
# ---------------------------------------------------------------------------

_UNITS_STMT = re.compile(r"\bFROM\s+units\b|\bINTO\s+units\b|\bUPDATE\s+units\b")


def _counting_trace(conn):
    """Attach a trace callback counting units-shaped and graph_edges
    statements; returns the counter dict."""
    counts = {"units": 0, "graph_edges": 0, "total": 0}

    def cb(sql):
        counts["total"] += 1
        if _UNITS_STMT.search(sql):
            counts["units"] += 1
        if re.search(r"\bFROM\s+graph_edges\b", sql):
            counts["graph_edges"] += 1

    conn.set_trace_callback(cb)
    return counts


def test_k21_traversing_400_nodes_units_statements_le_8():
    """K21 / V8-05.02: the D8-01 N+1 is dead — a 400-node traversal issues
    ≤ 8 ``units`` statements, counted by the trace callback."""
    conn = make_conn()
    tree_fixture(conn)
    counts = _counting_trace(conn)
    out = lane_graph(
        mk_ctx(conn), mk_qv("q"), _slice(cap=10_000), seeds=["n0"]
    )
    conn.set_trace_callback(None)
    assert out.stats["visited"] == graph_lane.FRONTIER_CAP_V1
    assert 0 < counts["units"] <= 8
    # sanity: eligibility + metadata came from the batched rows — every
    # candidate carries its source pin.
    assert all(c.source_id for c in out.candidates)


def test_k22_missing_eligibility_set_fails_closed():
    """K22 / V8-05.02: no eligible set → status=error, zero candidates,
    and NOT A SINGLE traversal statement (units or graph_edges)."""
    conn = make_conn()
    for u in ("A", "B"):
        add_unit(conn, u, "x")
    put_edge(conn, "A", "B", "co_mention", 0.9)

    for bad in (None, lambda row: True, object(), 42):
        ctx = mk_ctx(conn)
        ctx.eligible = bad  # bypass the fixture's set construction
        counts = _counting_trace(conn)
        out = lane_graph(ctx, mk_qv("q"), _slice(), seeds=["A"])
        conn.set_trace_callback(None)
        assert str(out.status) == "error", (bad, out.status)
        assert out.status.value == "error"  # coverage.lane() renders it
        assert out.candidates == []
        assert out.reason == "eligibility_set_unavailable"
        assert counts["units"] == 0 and counts["graph_edges"] == 0


def test_k22_unit_ids_object_form_is_the_set():
    """``_Eligible``-shaped adapters resolve through ``.unit_ids`` — the
    production set path (V8-05.02)."""
    conn = make_conn()
    for u in ("A", "B", "C"):
        add_unit(conn, u, "x")
    put_edge(conn, "A", "B", "co_mention", 0.9)
    put_edge(conn, "B", "C", "co_mention", 0.9)

    class FakeEligible:
        @property
        def unit_ids(self):
            return frozenset({"A", "B"})

    out = lane_graph(
        mk_ctx(conn, eligible=FakeEligible()),
        mk_qv("q"), _slice(), seeds=["A"],
    )
    assert out.status is LaneStatus.OK
    # C is reachable through B's edge but absent from the set → never
    # traversed (V7-08.11 — the hold is set membership, not SQL).
    assert {c.unit_id for c in out.candidates} == {"B"}


def test_k23_k_fan_caps_per_node_in_sql():
    """K23 / V8-05.03: a node with 50 outgoing edges contributes exactly
    K_fan edges — the top-K by (weight DESC, dst ASC) — identically
    ordered across runs."""
    conn = make_conn()
    add_unit(conn, "S", "x")
    for i in range(50):
        add_unit(conn, f"d{i:02d}", "x")
        # weight descending with i; dst ids ascending — the fan order is
        # exercised on both keys.
        put_edge(conn, "S", f"d{i:02d}", "co_mention", 1.0 - i * 0.01)
    ctx = mk_ctx(conn)
    out = lane_graph(ctx, mk_qv("q"), _slice(cap=100), seeds=["S"])
    # S contributes exactly K_fan outgoing edges in round 1 (the reverse
    # reads by its children in round 2 are their own K_fan-bounded reads).
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
    assert {c.unit_id for c in out4.candidates} == {f"d{i:02d}" for i in range(4)}


def test_generation_fence_batched_rows():
    """Generation fencing on the BATCHED row read (invariant 3): the
    latest row at/below the pin is chosen in Python; a unit whose only
    row is past the fence is never visited."""
    conn = make_conn()
    add_unit(conn, "S", "x")
    add_unit(conn, "V", "x")  # will gain a gen-3 restatement below
    conn.execute(
        "INSERT OR REPLACE INTO units (unit_id, source_id, revision,"
        " scope_id, kind, parent_unit_id, session_id, seq, speaker_canon,"
        " perspective, recorded_at_us, occurred_start_us, occurred_end_us,"
        " occurred_precision, occurred_source, byte_start, byte_end,"
        " generation) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("V", "src-V-new", 2, SCOPE, "turn", None, None, None, None,
         "user_stated", T0, None, None, "unknown", "unknown",
         None, None, 3),
    )
    add_unit(conn, "G", "x", gen=9)  # minted past every test snapshot
    put_edge(conn, "S", "V", "co_mention", 0.9, gen=1)
    put_edge(conn, "S", "G", "co_mention", 0.9, gen=1)
    # snapshot gen 5: V's gen-3 row is latest-visible; G has no row ≤ 5.
    ctx = mk_ctx(conn, generation=5,
                 eligible=eligible_units(conn, gen=5) | {"G"})
    out = lane_graph(ctx, mk_qv("q"), _slice(), seeds=["S"])
    by_id = {c.unit_id: c for c in out.candidates}
    assert "G" not in by_id  # no visible row — never visited
    assert by_id["V"].source_id == "src-V-new"
    assert by_id["V"].revision == 2


def test_m_seed_arm_truncates_provided_seeds():
    conn = make_conn()
    ids = [f"s{i}" for i in range(10)]
    for u in ids + ["x"]:
        add_unit(conn, u, "x")
    for u in ids:
        put_edge(conn, u, "x", "co_mention", 0.5)
    out = lane_graph(
        mk_ctx(conn, manifest={"graph.M_seed": 5}),
        mk_qv("q"), _slice(), seeds=ids,
    )
    assert out.stats["M_seed"] == 5
    assert out.stats["eligible_seeds"] == 5
    assert out.stats["seeds"] == 5


def test_gate_decision_hook_not_needed():
    """V8-05.05 skeleton: the scheduler-plumbed decision is honored —
    ``needed=False`` skips with ``not_needed`` and the inputs are logged;
    absent/malformed decisions run the lane (the gate is an optimization,
    never an authorization boundary)."""
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

    # needed=True (and a bare-bool shorthand) runs the lane.
    for dec in ({"needed": True, "inputs": {}}, True, {"inputs": {}}, "junk"):
        ctx2 = mk_ctx(conn, manifest={"graph.gate_decision": dec})
        out2 = lane_graph(ctx2, mk_qv("q"), _slice(), seeds=["A"])
        assert out2.status is LaneStatus.OK, dec
        assert out2.stats["gate"]["ran"] is True

    # the flag disarmed ignores the decision entirely.
    ctx3 = mk_ctx(conn, manifest={
        "graph.gate": False,
        "graph.gate_decision": {"needed": False, "inputs": {}},
    })
    out3 = lane_graph(ctx3, mk_qv("q"), _slice(), seeds=["A"])
    assert out3.status is LaneStatus.OK
    assert out3.stats["gate"]["armed"] is False


def test_edge_types_arm():
    """V8-05.09: per-type traversal is an arm — disabling a type removes
    it from expansion (its rows stay written, untouched)."""
    conn = make_conn()
    for u in ("A", "B", "C"):
        add_unit(conn, u, "x")
    put_edge(conn, "A", "B", "same_session", 0.9)
    put_edge(conn, "A", "C", "co_mention", 0.9)
    ctx = mk_ctx(conn, manifest={"graph.edge_types": ["co_mention"]})
    out = lane_graph(ctx, mk_qv("q"), _slice(), seeds=["A"])
    assert {c.unit_id for c in out.candidates} == {"C"}
    assert out.stats["edge_types"] == ["co_mention"]
    # the edge row is still on disk — only traversal was gated (K29).
    assert edges_of(conn, type_="same_session")
    # naming the never-traversable type fails loudly (V7-08.14)
    with pytest.raises(VerbatimError):
        lane_graph(
            mk_ctx(conn, manifest={"graph.edge_types": ["causal_candidate"]}),
            mk_qv("q"), _slice(), seeds=["A"],
        )


def test_onehop_additive_arm():
    """V8-05.08: Hindsight's one-hop additive form — a single expansion
    round, score = tanh(0.5·shared) + max(link)."""
    conn = make_conn()
    for u in ("S1", "S2", "X", "Y", "Z"):
        add_unit(conn, u, "x")
    put_edge(conn, "S1", "X", "supersedes", 1.0)   # eff 0.6
    put_edge(conn, "S2", "X", "supersedes", 1.0)   # eff 0.6
    put_edge(conn, "S1", "Y", "co_mention", 0.5)   # eff 0.5
    put_edge(conn, "X", "Z", "co_mention", 0.9)    # Z is 2 hops away
    ctx = mk_ctx(conn, manifest={"graph.hop_policy": "onehop_additive"})
    out = lane_graph(ctx, mk_qv("q"), _slice(), seeds=["S1", "S2"])
    by_id = {c.unit_id: c for c in out.candidates}
    assert set(by_id) == {"X", "Y"}  # one round — Z out of reach
    assert out.stats["hop_policy"] == "onehop_additive"
    assert out.stats["formula"] == "graph_onehop_add/v1"
    assert out.stats["iterations"] == 0
    # X: shared = 0.6+0.6 = 1.2, link = 0.6 → tanh(0.6) + 0.6
    assert by_id["X"].raw_score == pytest.approx(math.tanh(0.6) + 0.6)
    assert by_id["X"].signals["shared"] == pytest.approx(1.2)
    assert by_id["X"].signals["max_link"] == pytest.approx(0.6)
    # Y: shared = link = 0.5
    assert by_id["Y"].raw_score == pytest.approx(math.tanh(0.25) + 0.5)
    assert all(c.signals["hops"] == 1 for c in out.candidates)
    assert [h["unit"] for h in by_id["X"].signals["path"]] == ["S1", "X"]


def test_coverage_graph_block_and_containment_tag():
    """V8-20.03 ``coverage.graph`` + V8-05.06/11.04 containment tags."""
    conn = make_conn()
    for u in ("A", "B"):
        add_unit(conn, u, "x")
    put_edge(conn, "A", "B", "co_mention", 0.9)
    out = lane_graph(mk_ctx(conn), mk_qv("q"), _slice(), seeds=["A"])
    block = out.stats["graph"]
    assert set(block) == {
        "nodes_expanded", "rounds_completed", "edges_read", "contained_N_g",
    }
    assert block["nodes_expanded"] == 1
    # round 1 expands A→B; round 2 processes B's frontier (its read-back
    # edge is already visited) and completes with an empty next frontier.
    assert block["rounds_completed"] == 2
    assert block["edges_read"] == 2
    assert block["contained_N_g"] == 20
    assert out.stats["nodes_expanded"] == 1
    assert out.stats["edges_read"] == 2
    cand = out.candidates[0]
    assert cand.signals["contain_N_g"] == 20
    # arm resolution: "off" disarms → null, honestly reported
    out2 = lane_graph(
        mk_ctx(conn, manifest={"graph.contain_N_g": "off"}),
        mk_qv("q"), _slice(), seeds=["A"],
    )
    assert out2.stats["contained_N_g"] is None
    assert out2.stats["graph"]["contained_N_g"] is None
    assert out2.candidates[0].signals["contain_N_g"] is None


def test_rarest_term_seed():
    """V8-05.07/V75-04.04: the lowest-df content term's eligible postings
    extend the fused seeds — drawn through the eligible set."""
    conn = make_conn()
    conn.executescript(
        """
        CREATE VIRTUAL TABLE unit_fts USING fts5(text);
        CREATE TABLE lex_df (
            scope_id TEXT NOT NULL, generation INTEGER NOT NULL,
            field TEXT NOT NULL, term TEXT NOT NULL,
            stats_version TEXT NOT NULL, df INTEGER NOT NULL,
            PRIMARY KEY (scope_id, generation, field, term, stats_version)
        );
        """
    )
    add_unit(conn, "S", "seed")
    tgt = add_unit(conn, "T", "the rare clue hides here")
    put_edge(conn, "T", "Z", "co_mention", 0.9)
    add_unit(conn, "Z", "downstream")
    # unit_fts.rowid == units.rowid (units_jobs write contract)
    rid = conn.execute(
        "SELECT rowid FROM units WHERE unit_id='T'"
    ).fetchone()[0]
    conn.execute("INSERT INTO unit_fts(rowid, text) VALUES (?,?)",
                 (rid, "the rare clue hides here"))
    conn.execute(
        "INSERT INTO lex_df VALUES (?,?,?,?,?,?)",
        (SCOPE, 1, "text", "common", "bm25f/v1", 99),
    )
    conn.execute(
        "INSERT INTO lex_df VALUES (?,?,?,?,?,?)",
        (SCOPE, 1, "text", "rare", "bm25f/v1", 1),
    )
    out = lane_graph(
        mk_ctx(conn), mk_qv("common rare"), _slice(), seeds=["S"]
    )
    assert out.stats["rarest_seed"]["term"] == "rare"
    assert out.stats["rarest_seed"]["seeds"] == 1
    # T became a seed (in the eligible set), so it is not a candidate —
    # but its neighbour Z is.
    ids = {c.unit_id for c in out.candidates}
    assert "T" not in ids and "Z" in ids
    assert out.stats["eligible_seeds"] == 2


def test_manifest_seeds_are_consumed():
    """V8-05.07: the pipeline's ``ctx.manifest['seeds']`` union feeds the
    lane when no explicit seeds are passed."""
    conn = make_conn()
    for u in ("L1", "E1", "X"):
        add_unit(conn, u, "x")
    put_edge(conn, "L1", "X", "co_mention", 0.9)
    put_edge(conn, "E1", "X", "co_mention", 0.9)
    out = lane_graph(
        mk_ctx(conn, manifest={"seeds": ["L1", "E1"]}),
        mk_qv("q"), _slice(), seeds=None,
    )
    assert out.stats["seed_source"] == "manifest"
    assert out.stats["eligible_seeds"] == 2
    assert {c.unit_id for c in out.candidates} == {"X"}
