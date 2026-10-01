"""V8 scheduler tests — the §21.2 two-phase pipeline (V8-14.03/14.04/14.05).

Covers acceptance scenarios K82 (expansion slices re-derated at each
lane's start; exhausted budget → ``skipped(deadline_exhausted)``) and
K83 (post-lane stages bounded by ``R_post`` with deterministic
degradation reported in ``coverage.post.degraded``), the need gates
(V8-05.05 + each lane's own rule), the §23 scheduler/policy params
(``scheduler.R_post``, ``scheduler.post_pool``, ``graph.gate``), the
V8-14.05 slim post pool, per-lane coverage fields (V8-20.03), coverage
grafting, and the V8-12.06 deadline-cut propagation into the verdict.

Engine modules are faked via ``sys.modules`` injection where a seam is
needed; the landed deadline/policy modules run for real so the tests
exercise the production gate + slice math.
"""

from __future__ import annotations

import sys
import types

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v7 import (
    BudgetClass,
    CandidateV7,
    CoverageV7,
    IntentClass,
    IntentResult,
    LaneContextV7,
    LaneName,
    LaneOutput,
    LaneSlice,
    LaneStatus,
    NormAnalysis,
    NormTerm,
    QueryViewV7,
    ResultStatus,
    RetrievalPolicyV7,
    ScoredCandidate,
)
from verbatim.retrieval.v7.lanes_base import LANE_REGISTRY, register_lane
from verbatim.retrieval.v7.pipeline import run_search
from verbatim.retrieval.v7.policy import GatedPolicyV7


# ---------------------------------------------------------------------------
# Helpers (mirroring test_pipeline.py — kept self-contained)
# ---------------------------------------------------------------------------


class FakeClock:
    """Deterministic injectable clock (seconds, like time.monotonic)."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, ms: float) -> None:
        self.t += ms / 1000.0


@pytest.fixture(autouse=True)
def clean_registry():
    saved = dict(LANE_REGISTRY)
    LANE_REGISTRY.clear()
    yield
    LANE_REGISTRY.clear()
    LANE_REGISTRY.update(saved)


def make_query(
    text: str = "what did alice say",
    facets=(),
    intent: IntentClass = IntentClass.LOOKUP,
    canons=(),
) -> QueryViewV7:
    terms = tuple(
        NormTerm(term=w, channel="text", byte_start=i * 4, byte_end=i * 4 + len(w))
        for i, w in enumerate(text.split())
    )
    return QueryViewV7(
        query=text,
        norm=NormAnalysis(analyzer_id="norm/v2", terms=terms, identifiers=()),
        intent=IntentResult(primary=intent, classes=(intent,)),
        facets=facets,
        entity_canons=tuple(canons),
        query_time_us=1_000_000,
    )


def make_ctx(
    lanes=(LaneName.LEX, LaneName.DENSE),
    *,
    budget=BudgetClass.MID,
    manifest=None,
    eligible=None,
    params=None,
) -> LaneContextV7:
    cls = GatedPolicyV7 if params is not None else RetrievalPolicyV7
    kw = {"params": params} if params is not None else {}
    policy = cls(
        policy_id="retrieval_policy/v7",
        profile="test",
        lanes=tuple(lanes),
        lane_weights={},
        **kw,
    )
    return LaneContextV7(
        store=None,
        scope_id="scope",
        generation=1,
        eligible=eligible,
        query_time_us=1_000_000,
        profile="test",
        budget=budget,
        policy=policy,
        manifest=manifest or {},
    )


def cand(unit_id: str, lane: str, rank: int, score: float = 1.0) -> CandidateV7:
    return CandidateV7(
        unit_id=unit_id,
        source_id="src-" + unit_id,
        revision=1,
        lane=lane,
        rank=rank,
        raw_score=score,
        signals={"s": score},
    )


def fake_lane(name: str, units, *, calls=None, advance=None, clock=None, stats=None):
    def fn(ctx, qv, slice):
        if calls is not None:
            calls.append({"query": qv.query, "slice": slice})
        if advance and clock is not None:
            clock.advance(advance)
        out = LaneOutput(
            lane=name,
            status=LaneStatus.OK,
            candidates=[cand(u, name, i + 1, 1.0 - i * 0.01) for i, u in enumerate(units)],
            examined=len(units),
            eligible=len(units),
        )
        if stats:
            out.stats.update(stats)
        return out

    return fn


def run(ctx, query, deadline_ms=1000.0, clock=None, **kw):
    # Frozen clock by default: post-phase deadline gates (R_post slices)
    # must not flake under suite load; tests that exercise deadline
    # semantics pass an explicit advancing clock.
    return run_search(ctx, query, deadline_ms, clock=clock or FakeClock(), **kw)


def _fake_module(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _scored_echo(fused):
    return [
        ScoredCandidate(
            unit_id=f.unit_id,
            source_id=f.source_id,
            revision=f.revision,
            score=f.rrf,
            score_family="ranking/test",
        )
        for f in fused
    ]


# ---------------------------------------------------------------------------
# K82 — S2b re-derated expansion slices (V8-14.03, §21.2)
# ---------------------------------------------------------------------------


def test_expansion_slices_rederated_at_lane_start():
    """K82: each expansion lane's slice is computed from the budget
    remaining *when it starts*, shared over the costs of the needed
    lanes not yet run.  Costs: time 15, typed 25, graph 40 (obs is
    gated out by the temporal intent — its cost leaves the divisor)."""
    clock = FakeClock()
    calls = {n: [] for n in ("lex", "time", "typed", "obs", "graph")}
    register_lane(
        LaneName.LEX, fake_lane("lex", ["u0"], calls=calls["lex"], advance=100.0, clock=clock)
    )
    register_lane(
        LaneName.TIME, fake_lane("time", ["t0"], calls=calls["time"], advance=50.0, clock=clock)
    )
    register_lane(LaneName.TYPED, fake_lane("typed", ["y0"], calls=calls["typed"]))
    register_lane(LaneName.OBS, fake_lane("obs", ["o0"], calls=calls["obs"]))
    register_lane(LaneName.GRAPH, fake_lane("graph", ["g0"], calls=calls["graph"]))

    ctx = make_ctx(
        lanes=(LaneName.LEX, LaneName.TIME, LaneName.TYPED, LaneName.OBS, LaneName.GRAPH)
    )
    result = run(
        ctx, make_query(intent=IntentClass.TEMPORAL_POINT),
        deadline_ms=500.0, clock=clock,
    )

    # lex ran under the core slice (budget − R_post = 440).
    assert calls["lex"][0]["slice"].deadline_ms == pytest.approx(440.0)
    # obs gated out by its own intent rule — never invoked, zero cost.
    assert calls["obs"] == []
    assert result.lanes["obs"].status == LaneStatus.SKIPPED
    assert result.lanes["obs"].reason == "not_needed"
    # Needed expansion lanes re-derated at their starts:
    #   time:  rem = 500−100−60 = 340 → 340·15/80 = 63.75
    #   typed: rem = 500−150−60 = 290 → 290·25/65 ≈ 111.538
    #   graph: rem = 290 (last lane) → the whole remainder.
    assert calls["time"][0]["slice"].deadline_ms == pytest.approx(63.75)
    assert calls["typed"][0]["slice"].deadline_ms == pytest.approx(290.0 * 25 / 65)
    assert calls["graph"][0]["slice"].deadline_ms == pytest.approx(290.0)
    for name in ("time", "typed", "graph"):
        assert result.lanes[name].status == LaneStatus.OK


def test_expansion_lane_coverage_fields():
    """V8-20.03: every lane entry carries status/reason/phase/gate/
    slice_ms/t_ms — gated-out lanes included."""
    register_lane(LaneName.LEX, fake_lane("lex", ["u0"]))
    register_lane(LaneName.TIME, fake_lane("time", ["t0"]))
    ctx = make_ctx(lanes=(LaneName.LEX, LaneName.TIME))
    result = run(ctx, make_query(), deadline_ms=500.0)

    lex = result.coverage.lanes["lex"]
    assert lex["phase"] == "core"
    assert lex["gate"] == {"needed": True, "inputs": {}}
    assert lex["slice_ms"] > 0
    assert "t_ms" in lex and "status" in lex and "produced" in lex

    tim = result.coverage.lanes["time"]
    assert tim["status"] == "skipped"
    assert tim["reason"] == "not_needed"
    assert tim["phase"] == "expansion"
    assert tim["slice_ms"] == 0.0
    assert tim["gate"]["needed"] is False
    # Gate inputs logged verbatim (V8-05.05's disclosure rule).
    assert tim["gate"]["inputs"] == {"window": False, "temporal_intent": False}


def test_expansion_deadline_exhausted_when_no_budget():
    """K82 tail: a needed expansion lane finding no remaining budget
    reports ``skipped(deadline_exhausted)`` and consumes nothing."""
    clock = FakeClock()
    calls = {"time": [], "typed": [], "graph": []}
    register_lane(LaneName.LEX, fake_lane("lex", ["u0"], advance=200.0, clock=clock))
    register_lane(LaneName.TIME, fake_lane("time", ["t0"], calls=calls["time"]))
    register_lane(LaneName.TYPED, fake_lane("typed", ["y0"], calls=calls["typed"]))
    register_lane(LaneName.GRAPH, fake_lane("graph", ["g0"], calls=calls["graph"]))

    ctx = make_ctx(lanes=(LaneName.LEX, LaneName.TIME, LaneName.TYPED, LaneName.GRAPH))
    result = run(
        ctx, make_query(intent=IntentClass.TEMPORAL_POINT),
        deadline_ms=200.0, clock=clock,
    )

    for name in ("time", "typed", "graph"):
        assert calls[name] == []  # never invoked — zero budget spent
        out = result.lanes[name]
        assert out.status == LaneStatus.SKIPPED
        assert out.reason == "deadline_exhausted"
        entry = result.coverage.lanes[name]
        assert entry["status"] == "skipped"
        assert entry["reason"] == "deadline_exhausted"
        assert entry["phase"] == "expansion"
        assert entry["gate"]["needed"] is True
        assert entry["slice_ms"] == 0.0
    # Exhausted lanes are declared deadline-cut for the verdict (V8-12.06).
    assert set(ctx.manifest["deadline_cut_lanes"]) >= {"time", "typed", "graph"}


def test_auxiliary_lane_runs_in_core_phase():
    """Auxiliary lanes outside the §21.2 sets (exact_id/source/scope)
    run with the S2a core, gated ``needed=True`` and disclosed as
    auxiliary in the gate inputs."""
    register_lane(LaneName.LEX, fake_lane("lex", ["u0"]))
    calls = []
    register_lane(LaneName.EXACT_ID, fake_lane("exact_id", ["x0"], calls=calls))

    ctx = make_ctx(lanes=(LaneName.LEX, LaneName.EXACT_ID))
    result = run(ctx, make_query(), deadline_ms=500.0)

    assert calls  # ran — not need-gated
    entry = result.coverage.lanes["exact_id"]
    assert entry["status"] == "ok"
    assert entry["phase"] == "core"
    assert entry["gate"]["needed"] is True
    assert entry["gate"]["inputs"] == {"auxiliary": True}


def test_graph_gate_flag_off_always_runs():
    """``graph.gate=false`` (§23): the need gate is disabled — the lane
    runs even when every input says not needed; inputs still disclosed."""
    big = [f"u{i}" for i in range(50)]  # core union ≥ 4×limit=40
    register_lane(LaneName.LEX, fake_lane("lex", big))
    calls = []
    register_lane(LaneName.GRAPH, fake_lane("graph", ["g0"], calls=calls))

    ctx = make_ctx(
        lanes=(LaneName.LEX, LaneName.GRAPH), params={"graph.gate": False}
    )
    result = run(ctx, make_query(), deadline_ms=500.0)

    assert calls  # ran despite union_below=False, no canons, no request
    entry = result.coverage.lanes["graph"]
    assert entry["status"] == "ok"
    assert entry["gate"]["needed"] is True
    assert entry["gate"]["inputs"]["gate_flag"] == "off"
    assert entry["gate"]["inputs"]["union_below"] is False


def test_graph_gate_on_skips_when_not_needed():
    """Gate on (default): a fat core union + no intent/canon/request
    signal leaves graph ``skipped(not_needed)`` — zero budget spent."""
    big = [f"u{i}" for i in range(50)]
    register_lane(LaneName.LEX, fake_lane("lex", big))
    calls = []
    register_lane(LaneName.GRAPH, fake_lane("graph", ["g0"], calls=calls))

    ctx = make_ctx(lanes=(LaneName.LEX, LaneName.GRAPH))
    result = run(ctx, make_query(), deadline_ms=500.0)

    assert calls == []
    entry = result.coverage.lanes["graph"]
    assert entry["status"] == "skipped"
    assert entry["reason"] == "not_needed"
    assert entry["gate"]["needed"] is False
    assert entry["gate"]["inputs"]["union_below"] is False


def test_graph_requested_lane_overrides_gate():
    """V8-05.05(c): a caller ``lanes=["graph", …]`` request is a need
    signal of its own — graph runs and the request input is logged."""
    big = [f"u{i}" for i in range(50)]
    register_lane(LaneName.LEX, fake_lane("lex", big))
    calls = []
    register_lane(LaneName.GRAPH, fake_lane("graph", ["g0"], calls=calls))

    ctx = make_ctx(
        lanes=(LaneName.LEX, LaneName.GRAPH), manifest={"lanes": ["graph"]}
    )
    result = run(ctx, make_query(), deadline_ms=500.0)

    assert calls
    inputs = result.coverage.lanes["graph"]["gate"]["inputs"]
    assert inputs["requested"] is True
    assert inputs["union_below"] is False  # need came from the request


# ---------------------------------------------------------------------------
# K83 — deadline-bounded post stages (V8-14.04/14.05)
# ---------------------------------------------------------------------------


def test_post_degrades_in_order_under_pressure():
    """K83: 25 ms of post window left → rerank drops to the cheap
    subset and the optional CE detail is skipped; verdict still runs."""
    clock = FakeClock()
    register_lane(LaneName.LEX, fake_lane("lex", ["u0"], advance=100.0, clock=clock))

    ctx = make_ctx(lanes=(LaneName.LEX,), budget=BudgetClass.MID)
    result = run(ctx, make_query(), deadline_ms=100.0, clock=clock)

    # Post window: min(R_post=60, deadline+25 − elapsed=25) → 25 ms.
    post = getattr(result.coverage, "post", None)
    assert post is not None
    assert post["R_post_ms"] == 60.0
    assert post["degraded"] == ["rerank_features_expensive", "rerank_ce"]
    assert result.coverage.rerank["status"] == "skipped"
    assert result.coverage.rerank["reason"] == "deadline"
    assert result.coverage.rerank["stages"]["features"]["mode"] == "cheap"
    # Verdict + pack still produced — optional detail dropped, never the stage.
    assert result.verdict in (ResultStatus.READY, ResultStatus.INSUFFICIENT)
    assert result.pack is not None


def test_post_hard_boundary_degrades_everything_but_pack():
    """K83 hard rule: with the post window fully exhausted the pipeline
    degrades every droppable stage and still returns verdict + pack."""
    clock = FakeClock()
    register_lane(LaneName.LEX, fake_lane("lex", ["u0"], advance=550.0, clock=clock))

    ctx = make_ctx(lanes=(LaneName.LEX,), budget=BudgetClass.MID)
    result = run(ctx, make_query(), deadline_ms=500.0, clock=clock)

    post = getattr(result.coverage, "post", None)
    assert post is not None
    # Drop order (V8-14.04): expensive rerank → CE detail → boosts →
    # verdict detail.  ``materialize`` is absent honestly: with no store
    # the fill loop has no work to cut (the hook is only polled while
    # items remain unfilled).
    assert post["degraded"] == [
        "rerank_features",
        "rerank_ce",
        "boosts",
        "verdict_detail",
    ]
    assert result.coverage.rerank["stages"]["features"]["status"] == "degraded"
    assert result.coverage.rerank["stages"]["verdict"]["status"] == "degraded"
    assert result.coverage.rerank["stages"]["pack"]["status"] == "ok"
    assert result.verdict == ResultStatus.READY  # structural fallback
    assert result.pack is not None


def test_post_pool_trims_rerank_but_not_fused():
    """V8-14.05: the fused pool handed to rerank is trimmed to
    ``max(4 × limit, 64)``; the untrimmed pool stays on the result for
    explain."""
    units = [f"u{i}" for i in range(100)]
    register_lane(LaneName.LEX, fake_lane("lex", units))

    seen = {}

    def score_candidates(query, fused, **kw):
        seen["pool"] = len(fused)
        return _scored_echo(fused)

    mod = _fake_module(
        "verbatim.retrieval.v7.rerank_features", score_candidates=score_candidates
    )
    sys.modules["verbatim.retrieval.v7.rerank_features"] = mod
    try:
        ctx = make_ctx(lanes=(LaneName.LEX,))
        result = run(ctx, make_query(), deadline_ms=500.0, clock=FakeClock())
        assert len(result.fused) == 100          # full pool preserved
        assert seen["pool"] == 64                # trimmed before scoring
        assert len(result.scored) == 64          # rerank output bound
        note = result.coverage.rerank["stages"]["features"]
        assert note["post_pool_trimmed"] == 36
    finally:
        del sys.modules["verbatim.retrieval.v7.rerank_features"]


def test_post_coverage_block_always_emitted():
    register_lane(LaneName.LEX, fake_lane("lex", ["u0"]))
    result = run(make_ctx(lanes=(LaneName.LEX,)), make_query(), deadline_ms=500.0)
    post = getattr(result.coverage, "post", None)
    assert post == {"R_post_ms": 60.0, "degraded": []}


# ---------------------------------------------------------------------------
# §23 params — resolution + malformed handling
# ---------------------------------------------------------------------------


def test_scheduler_params_resolve_off_policy():
    """Declared arms land in coverage and change scheduling:
    R_post=10 widens the lane budget; post_pool=8 trims scored to 8."""
    units = [f"u{i}" for i in range(10)]
    register_lane(LaneName.LEX, fake_lane("lex", units))

    mod = _fake_module(
        "verbatim.retrieval.v7.rerank_features",
        score_candidates=lambda q, fused, **kw: _scored_echo(fused),
    )
    sys.modules["verbatim.retrieval.v7.rerank_features"] = mod
    try:
        ctx = make_ctx(
            lanes=(LaneName.LEX,),
            params={"scheduler.R_post": 10.0, "scheduler.post_pool": 8},
        )
        result = run(ctx, make_query(), deadline_ms=500.0)
        assert result.coverage.budget["r_post_ms"] == 10.0
        assert getattr(result.coverage, "post")["R_post_ms"] == 10.0
        assert len(result.scored) == 8
    finally:
        del sys.modules["verbatim.retrieval.v7.rerank_features"]


def test_malformed_scheduler_param_raises():
    """A declared-but-malformed arm is a policy defect — VALIDATION
    propagates; it is never silently reverted to the prior."""
    register_lane(LaneName.LEX, fake_lane("lex", ["u0"]))
    ctx = make_ctx(lanes=(LaneName.LEX,), params={"scheduler.R_post": "fast"})
    with pytest.raises(VerbatimError) as ei:
        run(ctx, make_query(), deadline_ms=500.0)
    assert ei.value.code == ErrorCode.VALIDATION


# ---------------------------------------------------------------------------
# Coverage grafting + deadline-cut propagation (V8-20.03 / V8-12.06)
# ---------------------------------------------------------------------------


def test_lane_stats_coverage_blocks_graft_into_top_level():
    """A lane's ``stats['coverage_*']``/listed bare blocks lift verbatim
    into top-level coverage (dense + graph shown)."""
    register_lane(
        LaneName.LEX,
        fake_lane("lex", ["u0"], stats={"coverage_lexical": {"rescue": True}}),
    )
    register_lane(
        LaneName.DENSE,
        fake_lane("dense", ["d0"], stats={"coverage_dense": {"mode": "stub"}, "graph": {"edges": 3}}),
    )
    result = run(make_ctx(), make_query(), deadline_ms=500.0)
    assert result.coverage.lexical["rescue"] is True
    assert result.coverage.dense["mode"] == "stub"
    assert getattr(result.coverage, "graph") == {"edges": 3}
    # Extension attrs participate in the stage coverage digest — a
    # grafted block changes the digest exactly like a declared field.
    from verbatim.retrieval.v7.pipeline import _coverage_digest

    plain = CoverageV7()
    grafted = CoverageV7()
    setattr(grafted, "graph", {"edges": 3})
    assert _coverage_digest(grafted) != _coverage_digest(plain)


def test_deadline_cut_lanes_reach_verdict_manifest():
    """V8-12.06: lanes the request deadline cut are declared on
    ``ctx.manifest['deadline_cut_lanes']`` before classify_groups runs."""
    captured = {}

    def classify_groups(items, query, ctx):
        captured["cut"] = list(ctx.manifest.get("deadline_cut_lanes", ()))
        return []

    def result_verdict(groups, query, calibration, **kw):
        return ResultStatus.READY, None

    mod = _fake_module(
        "verbatim.querying.verdict_v2",
        classify_groups=classify_groups,
        result_verdict=result_verdict,
    )
    sys.modules["verbatim.querying.verdict_v2"] = mod
    try:
        clock = FakeClock()
        lex_calls, dense_calls = [], []
        register_lane(LaneName.LEX, fake_lane("lex", ["l0"], calls=lex_calls))
        register_lane(LaneName.DENSE, fake_lane("dense", ["d0"], calls=dense_calls))
        # deadline == R_post → every core slice is 0 → both lanes cut.
        result = run(make_ctx(), make_query(), deadline_ms=60.0, clock=clock)
        assert lex_calls == [] and dense_calls == []
        assert result.lanes["lex"].status == LaneStatus.DEADLINE
        assert captured["cut"] == ["dense", "lex"]
    finally:
        del sys.modules["verbatim.querying.verdict_v2"]


def test_coverage_sql_block_emitted():
    register_lane(
        LaneName.LEX,
        fake_lane("lex", ["u0"], stats={"sql_statements": 3.0}),
    )
    result = run(make_ctx(lanes=(LaneName.LEX,)), make_query(), deadline_ms=500.0)
    sql = getattr(result.coverage, "sql", None)
    assert sql is not None
    assert sql["statements"] == 3.0
