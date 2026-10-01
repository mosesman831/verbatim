"""V8.5 retrieval arms — SPEC_V8_5 V85-05.05/05.06/05.07.

- V85-05.05 (second half): facet fan-out lanes = the policy's enabled
  lanes ∩ {lex, ent} — a policy without ``ent`` fans facets out on lex
  only (the c4 arm shape); ``facets.dense_per_facet`` opts dense in
  only when dense itself is enabled.
- V85-05.06 ``temporal.as_of_scope``: ``"window"`` (default) — the
  caller's ``as_of`` anchor resolves the window and feeds the verdict
  but must not re-anchor the boost recency clock; ``"global"`` keeps
  the prior re-anchored behavior for paired ablation. Malformed
  declared values raise VALIDATION.
- V85-05.07 ``scheduler.two_phase``: default ON (the §21.2 two-phase
  schedule); ``false`` runs every enabled lane in one phase sliced
  once from ``budget − R_post`` — no need gates. ``coverage.budget``
  discloses the schedule and the over-deadline measurement (elapsed
  vs deadline + 25 ms hard slack — c3's ``over25``).
"""

from __future__ import annotations

import sys
import types

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v7 import (
    BudgetClass,
    CandidateV7,
    IntentClass,
    IntentResult,
    LaneContextV7,
    LaneName,
    LaneOutput,
    LaneStatus,
    NormAnalysis,
    NormTerm,
    QueryViewV7,
)
from verbatim.retrieval.v7.lanes_base import LANE_REGISTRY, register_lane
from verbatim.retrieval.v7.pipeline import run_search
from verbatim.retrieval.v7.policy import GatedPolicyV7, RetrievalPolicyV7
from verbatim.retrieval.v7.temporal import (
    AS_OF_SCOPES,
    resolve_as_of_scope,
)
from verbatim.retrieval.v7 import deadline as deadline_mod


# ---------------------------------------------------------------------------
# Helpers (same shape as test_scheduler_v8.py — kept self-contained)
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
        source_id=f"src-{unit_id}",
        revision=1,
        lane=lane,
        rank=rank,
        raw_score=score,
        signals={"s": score},
    )


def fake_lane(name: str, units, *, calls=None, advance=None, clock=None):
    def fn(ctx, qv, slice):
        if calls is not None:
            calls.append({"query": qv.query, "slice": slice})
        if advance and clock is not None:
            clock.advance(advance)
        return LaneOutput(
            lane=name,
            status=LaneStatus.OK,
            candidates=[cand(u, name, i + 1, 1.0 - i * 0.01) for i, u in enumerate(units)],
            examined=len(units),
            eligible=len(units),
        )

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


def _capture_boosts():
    """Install a fake boosts module; returns the captured-call list."""
    calls = []

    def apply_boosts(scored, query, now_us):
        calls.append(now_us)
        return list(scored)

    sys.modules["verbatim.retrieval.v7.boosts"] = _fake_module(
        "verbatim.retrieval.v7.boosts", apply_boosts=apply_boosts
    )
    return calls


def _drop_boosts():
    sys.modules.pop("verbatim.retrieval.v7.boosts", None)


# ---------------------------------------------------------------------------
# V85-05.05 — facet lanes = enabled lanes ∩ {lex, ent}
# ---------------------------------------------------------------------------


def test_facet_lanes_intersect_enabled_lanes():
    """A policy without ``ent`` (the c4 arm) fans facets out on lex
    only — dense and disabled lanes never see a facet pass."""
    facet_q = make_query(text="melanie")
    register_lane(LaneName.LEX, fake_lane("lex", ["l0"]))
    register_lane(LaneName.DENSE, fake_lane("dense", ["d0"]))

    ctx = make_ctx(lanes=(LaneName.LEX, LaneName.DENSE))
    result = run(ctx, make_query(facets=(facet_q,)))

    assert result.lanes["lex"].stats.get("facets"), "lex ran no facet pass"
    assert "facets" not in result.lanes["dense"].stats


def test_facet_lanes_ent_only_when_lex_disabled():
    register_lane(LaneName.ENT, fake_lane("ent", ["e0"]))
    register_lane(LaneName.DENSE, fake_lane("dense", ["d0"]))

    facet_q = make_query(text="melanie")
    ctx = make_ctx(lanes=(LaneName.ENT, LaneName.DENSE))
    result = run(ctx, make_query(facets=(facet_q,)))

    assert result.lanes["ent"].stats.get("facets")
    assert "facets" not in result.lanes["dense"].stats


def test_facet_dense_arm_needs_dense_enabled():
    """``facets.dense_per_facet`` opts dense into the fan-out — but
    only when the dense lane itself is enabled by the policy."""
    facet_q = make_query(text="melanie")
    register_lane(LaneName.LEX, fake_lane("lex", ["l0"]))
    register_lane(LaneName.DENSE, fake_lane("dense", ["d0"]))

    ctx = make_ctx(
        lanes=(LaneName.LEX, LaneName.DENSE),
        params={"facets.dense_per_facet": True},
    )
    result = run(ctx, make_query(facets=(facet_q,)))
    assert result.lanes["dense"].stats.get("facets"), "armed+enabled dense saw no facet pass"

    ctx_off = make_ctx(
        lanes=(LaneName.LEX,), params={"facets.dense_per_facet": True}
    )
    result_off = run(ctx_off, make_query(facets=(facet_q,)))
    # dense is not enabled — the arm cannot conjure a disabled lane.
    assert "dense" not in result_off.lanes
    assert result_off.lanes["lex"].stats.get("facets")


# ---------------------------------------------------------------------------
# V85-05.06 — temporal.as_of_scope
# ---------------------------------------------------------------------------


def test_as_of_scope_resolver_defaults_window():
    assert resolve_as_of_scope(RetrievalPolicyV7(
        policy_id="p", profile="t", lanes=(), lane_weights={}
    )) == "window"
    assert AS_OF_SCOPES == frozenset({"window", "global"})


def test_as_of_scope_resolver_declared_values():
    pol = GatedPolicyV7(
        policy_id="p", profile="t", lanes=(), lane_weights={},
        params={"temporal.as_of_scope": "global"},
    )
    assert resolve_as_of_scope(pol) == "global"
    for bad in ("sideways", 7, True, ["window"]):
        pol = GatedPolicyV7(
            policy_id="p", profile="t", lanes=(), lane_weights={},
            params={"temporal.as_of_scope": bad},
        )
        with pytest.raises(VerbatimError) as ei:
            resolve_as_of_scope(pol)
        assert ei.value.code == ErrorCode.VALIDATION


def test_boost_anchor_window_scope_uses_wall_time(monkeypatch):
    """``as_of`` in the past (ctx anchor = 1_000_000) must not become
    the recency clock — under ``window`` the boost anchor is wall now."""
    monkeypatch.setattr(
        "verbatim.retrieval.v7.pipeline._wall_now_us", lambda: 9_999_000
    )
    calls = _capture_boosts()
    try:
        register_lane(LaneName.LEX, fake_lane("lex", ["l0"]))
        run(make_ctx(lanes=(LaneName.LEX,)), make_query())
        assert calls == [9_999_000]
    finally:
        _drop_boosts()


def test_boost_anchor_global_scope_uses_as_of_anchor(monkeypatch):
    """``temporal.as_of_scope="global"`` — the prior behavior: the
    caller's anchor re-anchors the recency clock."""
    monkeypatch.setattr(
        "verbatim.retrieval.v7.pipeline._wall_now_us", lambda: 9_999_000
    )
    calls = _capture_boosts()
    try:
        register_lane(LaneName.LEX, fake_lane("lex", ["l0"]))
        ctx = make_ctx(
            lanes=(LaneName.LEX,),
            params={"temporal.as_of_scope": "global"},
        )
        run(ctx, make_query())
        assert calls == [1_000_000]  # ctx.query_time_us, not wall time
    finally:
        _drop_boosts()


def test_as_of_scope_disclosed_in_coverage():
    register_lane(LaneName.LEX, fake_lane("lex", ["l0"]))
    result = run(make_ctx(lanes=(LaneName.LEX,)), make_query())
    assert getattr(result.coverage, "temporal")["as_of_scope"] == "window"
    ctx = make_ctx(
        lanes=(LaneName.LEX,), params={"temporal.as_of_scope": "global"}
    )
    result = run(ctx, make_query())
    assert getattr(result.coverage, "temporal")["as_of_scope"] == "global"


def test_malformed_as_of_scope_raises():
    register_lane(LaneName.LEX, fake_lane("lex", ["l0"]))
    ctx = make_ctx(
        lanes=(LaneName.LEX,), params={"temporal.as_of_scope": "sideways"}
    )
    with pytest.raises(VerbatimError) as ei:
        run(ctx, make_query())
    assert ei.value.code == ErrorCode.VALIDATION


# ---------------------------------------------------------------------------
# V85-05.07 — scheduler.two_phase + over-deadline coverage
# ---------------------------------------------------------------------------


def test_two_phase_default_disclosed_and_gates_apply():
    """Default (declared prior): two-phase — expansion lanes are
    need-gated; the mode rides ``coverage.budget['two_phase']``."""
    big = [f"u{i}" for i in range(50)]  # core union ≥ 4×limit → no need
    register_lane(LaneName.LEX, fake_lane("lex", big))
    register_lane(LaneName.GRAPH, fake_lane("graph", ["g0"]))

    ctx = make_ctx(lanes=(LaneName.LEX, LaneName.GRAPH))
    result = run(ctx, make_query(), deadline_ms=500.0)

    assert result.coverage.budget["two_phase"] is True
    # graph gated out — LOOKUP intent, no canons, small union.
    entry = result.coverage.lanes["graph"]
    assert entry["phase"] == "expansion"
    assert entry["status"] == "skipped"
    assert entry["reason"] == "not_needed"
    assert entry["gate"]["needed"] is False


def test_single_phase_runs_every_lane_ungated():
    """``scheduler.two_phase=false``: all enabled lanes are sliced once
    from ``budget − R_post`` and run in policy order — no need gate, no
    re-deration; in-lane skips still apply."""
    clock = FakeClock()
    calls = {}
    for name, lane in (
        ("lex", LaneName.LEX),
        ("time", LaneName.TIME),
        ("graph", LaneName.GRAPH),
    ):
        calls[name] = []
        register_lane(
            lane,
            fake_lane(name, [f"{name}0"], calls=calls[name], advance=10.0, clock=clock),
        )

    ctx = make_ctx(
        lanes=(LaneName.LEX, LaneName.TIME, LaneName.GRAPH),
        params={"scheduler.two_phase": False},
    )
    result = run(
        ctx, make_query(intent=IntentClass.LOOKUP), deadline_ms=500.0, clock=clock
    )

    assert result.coverage.budget["two_phase"] is False
    # LOOKUP intent, no window, no canons — every need gate would have
    # said no; under single-phase the lanes ran anyway.
    for name in ("lex", "time", "graph"):
        assert calls[name], f"{name} never invoked under single-phase"
        assert result.lanes[name].status == LaneStatus.OK
        entry = result.coverage.lanes[name]
        assert entry["phase"] == "core"
        assert entry["gate"]["needed"] is True
        assert entry["gate"]["inputs"]["scheduler"] == "single_phase"
        assert entry["slice_ms"] > 0.0
    # all slices dealt from budget − R_post = 440
    for name in ("lex", "time", "graph"):
        sl = calls[name][0]["slice"]
        assert sl.deadline_ms > 0.0
        assert sl.deadline_ms <= 440.0


def test_single_phase_slices_match_allocate_output():
    """Single-phase slices are exactly ``deadline.allocate`` over the
    full enabled set — the V7-05.05 shape, honestly disclosed."""
    clock = FakeClock()
    calls = {"lex": [], "graph": []}
    register_lane(LaneName.LEX, fake_lane("lex", ["l0"], calls=calls["lex"]))
    register_lane(LaneName.GRAPH, fake_lane("graph", ["g0"], calls=calls["graph"]))

    ctx = make_ctx(
        lanes=(LaneName.LEX, LaneName.GRAPH),
        params={"scheduler.two_phase": False},
    )
    run(ctx, make_query(), deadline_ms=500.0, clock=clock)

    expect = deadline_mod.allocate(
        440.0, [LaneName.LEX, LaneName.GRAPH], ctx.policy
    )
    for name in ("lex", "graph"):
        got = calls[name][0]["slice"].deadline_ms
        assert got == pytest.approx(expect[LaneName(name)].deadline_ms)


def test_two_phase_flag_malformed_raises():
    register_lane(LaneName.LEX, fake_lane("lex", ["l0"]))
    ctx = make_ctx(
        lanes=(LaneName.LEX,), params={"scheduler.two_phase": "yes"}
    )
    with pytest.raises(VerbatimError) as ei:
        run(ctx, make_query())
    assert ei.value.code == ErrorCode.VALIDATION


def test_two_phase_resolver_unit():
    assert deadline_mod.two_phase_enabled(
        RetrievalPolicyV7(policy_id="p", profile="t", lanes=(), lane_weights={})
    ) is True
    pol = GatedPolicyV7(
        policy_id="p", profile="t", lanes=(), lane_weights={},
        params={"scheduler.two_phase": False},
    )
    assert deadline_mod.two_phase_enabled(pol) is False


def test_over_deadline_coverage_stat():
    """The c3 ``over25`` criterion rides ``coverage.budget`` — elapsed
    past ``deadline + 25 ms`` marks ``over_deadline`` with the overrun."""
    clock = FakeClock()
    register_lane(
        LaneName.LEX, fake_lane("lex", ["l0"], advance=600.0, clock=clock)
    )
    result = run(make_ctx(lanes=(LaneName.LEX,)), make_query(), deadline_ms=500.0, clock=clock)

    budget = result.coverage.budget
    assert budget["deadline_ms"] == 500.0
    assert budget["hard_wall_ms"] == 525.0
    # fake clock: 600 ms lane + 0 elsewhere → elapsed 600 > 525.
    assert budget["over_deadline"] is True
    assert budget["over_by_ms"] == pytest.approx(75.0)
    assert budget["elapsed_ms"] == pytest.approx(budget["over_by_ms"] + 525.0)


def test_under_deadline_coverage_stat_clean():
    register_lane(LaneName.LEX, fake_lane("lex", ["l0"]))
    result = run(make_ctx(lanes=(LaneName.LEX,)), make_query(), deadline_ms=500.0)
    budget = result.coverage.budget
    assert budget["over_deadline"] is False
    assert "over_by_ms" not in budget
    assert budget["elapsed_ms"] >= 0.0
    assert budget["hard_wall_ms"] == 525.0


def test_unbounded_deadline_never_over():
    register_lane(LaneName.LEX, fake_lane("lex", ["l0"]))
    result = run(
        make_ctx(lanes=(LaneName.LEX,)), make_query(), deadline_ms=float("inf")
    )
    assert result.coverage.budget["over_deadline"] is False
    assert result.coverage.budget["hard_wall_ms"] is None


def test_deadline_stats_present_under_single_phase():
    register_lane(LaneName.LEX, fake_lane("lex", ["l0"]))
    ctx = make_ctx(
        lanes=(LaneName.LEX,), params={"scheduler.two_phase": False}
    )
    result = run(ctx, make_query(), deadline_ms=500.0)
    budget = result.coverage.budget
    assert budget["two_phase"] is False
    assert budget["over_deadline"] is False
    assert "elapsed_ms" in budget
