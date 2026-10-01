"""Tests for the §04.2 read-path orchestrator (w-pipeline).

Covers: all enabled lanes run as peers (the D7-09 regression — a lane
filling its cap must not gate the others), the V8-14.03 two-phase
scheduler (core slices dealt once from ``budget − R_post``; expansion
lanes need-gated and re-derated), fail-closed vs degraded error
handling, deadline enforcement, cap truncation, deterministic fused
output, the V7-05.15 explain payload, and the §32.17 stage record.

Engine modules are faked via ``sys.modules`` injection (exercises the
real lazy-import path); the absent-module paths are pinned by
monkeypatching ``pipeline._lazy_import`` so the tests stay green whether
or not sibling wave-A modules have landed.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import replace as dc_replace

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v7 import (
    POOLS,
    BudgetClass,
    CandidateV7,
    FusedCandidate,
    IntentClass,
    IntentResult,
    LaneContextV7,
    LaneName,
    LaneOutput,
    LaneSlice,
    LaneStatus,
    LaneV7,
    MissingDescriptor,
    NormAnalysis,
    NormTerm,
    QueryViewV7,
    ResultStatus,
    RetrievalPolicyV7,
    ScoredCandidate,
)
from verbatim.retrieval.v7 import pipeline
from verbatim.retrieval.v7.lanes_base import (
    LANE_REGISTRY,
    register_lane,
    run_one,
)
from verbatim.retrieval.v7.pipeline import PipelineResult, run_search


# ---------------------------------------------------------------------------
# Helpers
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
        query_time_us=1_000_000,
    )


def make_ctx(
    lanes=(LaneName.LEX, LaneName.DENSE),
    *,
    budget=BudgetClass.MID,
    weights=None,
    manifest=None,
    eligible=None,
) -> LaneContextV7:
    policy = RetrievalPolicyV7(
        policy_id="retrieval_policy/v7",
        profile="test",
        lanes=tuple(lanes),
        lane_weights=weights or {},
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


def fake_lane(name: str, units, *, calls: list | None = None, advance=None, clock=None):
    """Build a fake lane returning ``units``; records calls/slices."""

    def fn(ctx, qv, slice):
        if calls is not None:
            calls.append({"query": qv.query, "slice": slice})
        if advance and clock is not None:
            clock.advance(advance)
        return LaneOutput(
            lane=name,
            status=LaneStatus.OK,
            candidates=[cand(u, name, i + 1, 1.0 - i * 0.1) for i, u in enumerate(units)],
            examined=len(units),
            eligible=len(units),
        )

    return fn


def run(ctx, query, deadline_ms=1000.0, clock=None, **kw):
    # Frozen clock by default: post-phase deadline gates (R_post slices)
    # must not flake under suite load; tests that exercise deadline
    # semantics pass an explicit advancing clock.
    return run_search(ctx, query, deadline_ms, clock=clock or FakeClock(), **kw)


# ---------------------------------------------------------------------------
# run_one — the lane wrapper
# ---------------------------------------------------------------------------


def test_run_one_unregistered_lane_unavailable():
    out = run_one(make_ctx(), make_query(), LaneName.OBS, LaneSlice(10.0, 5))
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "not_registered"


def test_run_one_zero_slice_deadline_without_call():
    calls = []
    register_lane(LaneName.LEX, fake_lane("lex", ["u0"], calls=calls))
    out = run_one(make_ctx(), make_query(), LaneName.LEX, LaneSlice(0.0, 5))
    assert out.status == LaneStatus.DEADLINE
    assert out.reason == "no_slice_budget"
    assert calls == []


def test_run_one_exception_degrades_unavailable():
    def boom(ctx, qv, slice):
        raise RuntimeError("kaboom")

    register_lane(LaneName.LEX, boom)
    out = run_one(make_ctx(), make_query(), LaneName.LEX, LaneSlice(10.0, 5))
    assert out.status == LaneStatus.UNAVAILABLE
    assert "kaboom" in out.reason


def test_run_one_fail_closed_verbatim_error_propagates():
    def boom(ctx, qv, slice):
        raise VerbatimError(ErrorCode.INTEGRITY, "hmac mismatch")

    register_lane(LaneName.LEX, boom)
    with pytest.raises(VerbatimError):
        run_one(make_ctx(), make_query(), LaneName.LEX, LaneSlice(10.0, 5))


def test_run_one_nonfailclosed_verbatim_error_degrades():
    def boom(ctx, qv, slice):
        raise VerbatimError(ErrorCode.VALIDATION, "bad lane input")

    register_lane(LaneName.LEX, boom)
    out = run_one(make_ctx(), make_query(), LaneName.LEX, LaneSlice(10.0, 5))
    assert out.status == LaneStatus.UNAVAILABLE


def test_run_one_truncates_at_cap():
    register_lane(LaneName.LEX, fake_lane("lex", [f"u{i}" for i in range(7)]))
    out = run_one(make_ctx(), make_query(), LaneName.LEX, LaneSlice(10.0, 3))
    assert out.status == LaneStatus.OK
    assert [c.unit_id for c in out.candidates] == ["u0", "u1", "u2"]
    assert out.stats["cap_truncated"] == 4


def test_run_one_overrun_downgrades_to_partial():
    clock = FakeClock()

    def slow(ctx, qv, slice):
        clock.advance(50.0)
        return LaneOutput(lane="lex", status=LaneStatus.OK, candidates=[cand("u0", "lex", 1)])

    register_lane(LaneName.LEX, slow)
    out = run_one(
        make_ctx(), make_query(), LaneName.LEX, LaneSlice(10.0, 5), clock=clock
    )
    assert out.status == LaneStatus.PARTIAL
    assert out.reason == "deadline_overrun"
    assert out.stats["overrun_ms"] > 0


def test_run_one_lanev7_instance_and_class():
    class MyLane(LaneV7):
        name = "ent"

        def run(self, ctx, query, slice):
            return LaneOutput(lane="ent", status=LaneStatus.OK, candidates=[cand("e0", "ent", 1)])

    register_lane(LaneName.ENT, MyLane())
    out = run_one(make_ctx(), make_query(), LaneName.ENT, LaneSlice(10.0, 5))
    assert out.status == LaneStatus.OK and out.candidates[0].unit_id == "e0"

    register_lane(LaneName.TIME, MyLane)  # class form
    out2 = run_one(make_ctx(), make_query(), LaneName.TIME, LaneSlice(10.0, 5))
    assert out2.status == LaneStatus.OK


# ---------------------------------------------------------------------------
# S2 — peers, slices, deadline (V7-05.01/05.05; D7-09/D7-14 dead)
# ---------------------------------------------------------------------------


def test_all_enabled_lanes_run_when_early_lane_fills_cap():
    """D7-09 regression: the typed lane returning a full pool must NOT
    gate lexical/dense out of the run or out of coverage.

    V8-14.03: ``typed`` is an S2b expansion lane — it needs a
    fact-shaped intent to pass its need gate (CURRENT_VALUE is in the
    lane's own typed-intent set)."""

    pool = POOLS[BudgetClass.LOW]  # lane_cap = 50
    typed_units = [f"t{i}" for i in range(pool.lane_cap)]  # fills the cap

    register_lane(LaneName.TYPED, fake_lane("typed", typed_units))
    register_lane(LaneName.LEX, fake_lane("lex", ["l0", "l1", "l2"]))
    register_lane(LaneName.DENSE, fake_lane("dense", ["d0", "d1"]))

    ctx = make_ctx(lanes=(LaneName.TYPED, LaneName.LEX, LaneName.DENSE), budget=BudgetClass.LOW)
    result = run(ctx, make_query(intent=IntentClass.CURRENT_VALUE))

    # Every enabled lane ran and reports ok in coverage.
    for name in ("typed", "lex", "dense"):
        assert result.lanes[name].status == LaneStatus.OK
        assert result.coverage.lanes[name]["status"] == "ok"

    # The full-pool lane did not starve the others: their candidates are
    # in the fused set.
    fused_ids = {f.unit_id for f in result.fused}
    assert {"l0", "l1", "l2", "d0", "d1"} <= fused_ids
    assert len(result.lanes["typed"].candidates) == pool.lane_cap


def test_per_lane_slices_dealt_at_s2_entry():
    """V7-05.05: slices come from deadline.allocate at S2 entry, keyed by
    lane — not derated sequentially."""

    mod = types.ModuleType("verbatim.retrieval.v7.deadline")
    calls = []

    def allocate(remaining_ms, lanes, policy):
        calls.append({"remaining": remaining_ms, "lanes": list(lanes)})
        return {
            LaneName.LEX: LaneSlice(deadline_ms=11.0, cap=7),
            LaneName.DENSE: LaneSlice(deadline_ms=22.0, cap=9),
        }

    mod.allocate = allocate
    sys.modules["verbatim.retrieval.v7.deadline"] = mod
    try:
        lex_calls, dense_calls = [], []
        register_lane(LaneName.LEX, fake_lane("lex", ["l0"], calls=lex_calls))
        register_lane(LaneName.DENSE, fake_lane("dense", ["d0"], calls=dense_calls))

        run(make_ctx(), make_query())

        assert len(calls) == 1  # exactly one allocation at S2 entry
        assert set(calls[0]["lanes"]) == {LaneName.LEX, LaneName.DENSE}
        assert lex_calls[0]["slice"].deadline_ms == 11.0
        assert lex_calls[0]["slice"].cap == 7
        assert dense_calls[0]["slice"].deadline_ms == 22.0
        assert dense_calls[0]["slice"].cap == 9
    finally:
        del sys.modules["verbatim.retrieval.v7.deadline"]


def test_slow_lane_does_not_shrink_peer_slice():
    """D7-14 dead: lane A overruns -> A goes partial; lane B still runs
    under the slice it was dealt at S2 entry (not derated)."""

    mod = types.ModuleType("verbatim.retrieval.v7.deadline")

    def allocate(remaining_ms, lanes, policy, **kw):
        return {
            LaneName.LEX: LaneSlice(deadline_ms=10.0, cap=50),
            LaneName.DENSE: LaneSlice(deadline_ms=40.0, cap=50),
        }

    mod.allocate = allocate
    sys.modules["verbatim.retrieval.v7.deadline"] = mod
    try:
        clock = FakeClock()
        register_lane(
            LaneName.LEX, fake_lane("lex", ["l0"], advance=60.0, clock=clock)
        )
        b_calls = []
        register_lane(LaneName.DENSE, fake_lane("dense", ["d0"], calls=b_calls))

        result = run(make_ctx(), make_query(), deadline_ms=100.0, clock=clock)

        assert result.lanes["lex"].status == LaneStatus.PARTIAL
        assert result.lanes["lex"].reason == "deadline_overrun"
        assert result.lanes["dense"].status == LaneStatus.OK
        # B's slice was dealt at entry — A's 60ms burn did not shrink it.
        assert b_calls[0]["slice"].deadline_ms == 40.0
    finally:
        del sys.modules["verbatim.retrieval.v7.deadline"]


def test_total_deadline_exhausted_skips_later_lanes():
    """V8-14.03: the core slice comes from ``deadline − R_post`` (60 ms
    prior) — a 100 ms request affords lex a slice, but its 100 ms burn
    leaves dense zero remaining budget → ``deadline_exhausted``."""
    clock = FakeClock()
    register_lane(LaneName.LEX, fake_lane("lex", ["l0"], advance=100.0, clock=clock))
    b_calls = []
    register_lane(LaneName.DENSE, fake_lane("dense", ["d0"], calls=b_calls))

    result = run(make_ctx(), make_query(), deadline_ms=100.0, clock=clock)

    assert result.lanes["lex"].status == LaneStatus.PARTIAL
    assert result.lanes["dense"].status == LaneStatus.DEADLINE
    assert result.lanes["dense"].reason == "deadline_exhausted"
    assert b_calls == []  # never invoked
    assert result.coverage.lanes["dense"]["status"] == "deadline"


def test_missing_lane_impl_reports_unavailable_and_continues(monkeypatch):
    register_lane(LaneName.LEX, fake_lane("lex", ["l0"]))
    # Every LaneName now has a module in _LANE_MODULES — drop ``ent``'s
    # entry for this test so it stays unregistered and exercises the
    # not_registered path (auto-load finds no module to import).
    from verbatim.retrieval.v7 import pipeline as _pl
    monkeypatch.delitem(_pl._LANE_MODULES, LaneName.ENT)
    ctx = make_ctx(lanes=(LaneName.LEX, LaneName.ENT))
    result = run(ctx, make_query())
    assert result.lanes["ent"].status == LaneStatus.UNAVAILABLE
    assert result.coverage.lanes["ent"]["reason"] == "not_registered"
    assert result.lanes["lex"].status == LaneStatus.OK
    assert {f.unit_id for f in result.fused} == {"l0"}


def test_lane_over_cap_truncated_in_pipeline():
    pool = POOLS[BudgetClass.MID]
    units = [f"u{i}" for i in range(pool.lane_cap + 5)]
    register_lane(LaneName.LEX, fake_lane("lex", units))
    result = run(make_ctx(lanes=(LaneName.LEX,)), make_query())
    assert len(result.lanes["lex"].candidates) == pool.lane_cap
    assert result.lanes["lex"].stats["cap_truncated"] == 5


def test_pipeline_fail_closed_error_propagates():
    def boom(ctx, qv, slice):
        raise VerbatimError(ErrorCode.STALE_EPOCH, "epoch moved")

    register_lane(LaneName.LEX, fake_lane("lex", ["l0"]))
    register_lane(LaneName.DENSE, boom)
    with pytest.raises(VerbatimError):
        run(make_ctx(), make_query())


# ---------------------------------------------------------------------------
# S3–S8 — fusion, rerank, CE, boosts, verdict, pack
# ---------------------------------------------------------------------------


def _fake_module(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def test_fused_output_deterministic(monkeypatch):
    _block(monkeypatch, "verbatim.retrieval.v7.fusion")  # exercise the fallback
    register_lane(LaneName.LEX, fake_lane("lex", ["a", "b", "c"]))
    register_lane(LaneName.DENSE, fake_lane("dense", ["b", "d"]))
    r1 = run(make_ctx(), make_query())
    r2 = run(make_ctx(), make_query())
    assert [(f.unit_id, f.rrf) for f in r1.fused] == [
        (f.unit_id, f.rrf) for f in r2.fused
    ]
    # b in both lanes outranks single-lane candidates under RRF
    assert r1.fused[0].unit_id == "b"
    assert r1.fused[0].lane_ranks == {"lex": 2, "dense": 1}


def test_fusion_module_used_when_present():
    seen = {}

    def rrf_fuse(outputs, weights, k):
        seen["outputs"] = [o.lane for o in outputs]
        seen["weights"] = weights
        seen["k"] = k
        return [
            FusedCandidate(
                unit_id="z", source_id="s", revision=1, rrf=9.0, lane_ranks={}
            )
        ]

    mod = _fake_module("verbatim.retrieval.v7.fusion", rrf_fuse=rrf_fuse)
    sys.modules["verbatim.retrieval.v7.fusion"] = mod
    try:
        weights = {IntentClass.LOOKUP: {LaneName.LEX: 2.5}}
        register_lane(LaneName.LEX, fake_lane("lex", ["l0"]))
        result = run(make_ctx(weights=weights), make_query())
        assert seen["outputs"] == ["lex"]
        assert seen["weights"] == {LaneName.LEX: 2.5}
        assert seen["k"] == 60
        assert [f.unit_id for f in result.fused] == ["z"]
        assert result.coverage.rerank["stages"]["fusion"]["status"] == "ok"
    finally:
        del sys.modules["verbatim.retrieval.v7.fusion"]


def test_rerank_features_cut_to_pool_when_present():
    """V8-14.05: the rerank pool is ``scheduler.post_pool`` =
    ``max(4 × limit, 64)`` — the slim post pool supersedes V7's
    ``pool.rerank_pool`` bound on the scored list."""
    big = [
        ScoredCandidate(
            unit_id=f"s{i}", source_id="s", revision=1,
            score=1.0 - i * 0.01, score_family="ranking/v7",
        )
        for i in range(500)
    ]
    seen = {}

    def score_candidates(query, fused, ctx):
        seen["pool"] = len(fused)
        return big

    mod = _fake_module(
        "verbatim.retrieval.v7.rerank_features", score_candidates=score_candidates
    )
    sys.modules["verbatim.retrieval.v7.rerank_features"] = mod
    try:
        register_lane(LaneName.LEX, fake_lane("lex", ["l0"]))
        result = run(make_ctx(budget=BudgetClass.LOW), make_query())
        # default limit 10 → post_pool = max(40, 64) = 64
        assert len(result.scored) == 64
        assert seen["pool"] == 1  # the fused pool itself was trimmed
        assert result.scored[0].score_family == "ranking/v7"
    finally:
        del sys.modules["verbatim.retrieval.v7.rerank_features"]


def _block(monkeypatch, *modnames):
    """Pin specific engine modules absent — robust whether or not the
    sibling wave-A modules have landed yet."""

    real = pipeline._lazy_import
    monkeypatch.setattr(
        pipeline,
        "_lazy_import",
        lambda name: None if name in modnames else real(name),
    )


def test_ce_hook_absent_reports_no_ce(monkeypatch):
    _block(monkeypatch, "verbatim.retrieval.v7.rerank_ce")
    register_lane(LaneName.LEX, fake_lane("lex", ["l0"]))
    result = run(make_ctx(budget=BudgetClass.MID), make_query())
    assert result.coverage.rerank["status"] == "unavailable"
    assert result.coverage.rerank["reason"] == "no_ce"
    assert result.scored  # pipeline continued


def test_ce_hook_rescores_head_when_present():
    seen = {}

    def rerank(head, query, ctx):
        seen["n"] = len(head)
        return [
            dc_replace(s, detail={**s.detail, "ce": True}) for s in head
        ]

    mod = _fake_module("verbatim.retrieval.v7.rerank_ce", rerank=rerank)
    sys.modules["verbatim.retrieval.v7.rerank_ce"] = mod
    try:
        register_lane(LaneName.LEX, fake_lane("lex", ["a", "b", "c"]))
        result = run(make_ctx(budget=BudgetClass.HIGH), make_query())
        assert result.coverage.rerank["status"] == "ok"
        assert seen["n"] == 3  # head = scored[:ce_pool=32] = all three
        assert all(s.detail.get("ce") for s in result.scored)
    finally:
        del sys.modules["verbatim.retrieval.v7.rerank_ce"]


def test_boosts_applied_when_present(monkeypatch):
    calls = []

    def apply_boosts(scored, query, now_us):
        calls.append({"n": len(scored), "now_us": now_us})
        return [
            ScoredCandidate(
                unit_id=s.unit_id, source_id=s.source_id, revision=s.revision,
                score=s.score * 2.0, score_family=s.score_family,
                detail={**s.detail, "boost": 2.0},
            )
            for s in scored
        ]

    # V85-05.06 — under the default ``temporal.as_of_scope="window"`` the
    # boost recency clock is wall time, not the caller's ``as_of`` anchor
    # (the anchor still resolves the window upstream and feeds the
    # verdict via ``ctx``). ``_wall_now_us`` is pinned for determinism.
    monkeypatch.setattr(
        "verbatim.retrieval.v7.pipeline._wall_now_us", lambda: 9_999_000
    )
    register_lane(LaneName.LEX, fake_lane("lex", ["l0"]))
    mod = _fake_module("verbatim.retrieval.v7.boosts", apply_boosts=apply_boosts)
    sys.modules["verbatim.retrieval.v7.boosts"] = mod
    try:
        result = run(make_ctx(), make_query())
        assert calls == [{"n": 1, "now_us": 9_999_000}]
        assert result.scored[0].detail["boost"] == 2.0
        assert result.coverage.rerank["stages"]["boosts"]["status"] == "ok"
    finally:
        del sys.modules["verbatim.retrieval.v7.boosts"]


def test_verdict_v2_used_when_present():
    def classify_groups(items, query, ctx):
        return []

    miss = MissingDescriptor(facets={"topic": ("nothing",)}, note="nope")

    def result_verdict(groups, query, calibration):
        return ResultStatus.INSUFFICIENT, miss

    mod = _fake_module(
        "verbatim.querying.verdict_v2",
        classify_groups=classify_groups,
        result_verdict=result_verdict,
    )
    sys.modules["verbatim.querying.verdict_v2"] = mod
    try:
        register_lane(LaneName.LEX, fake_lane("lex", ["l0"]))
        result = run(make_ctx(), make_query())
        assert result.verdict == ResultStatus.INSUFFICIENT
        assert result.missing is miss
        assert result.coverage.rerank["stages"]["verdict"]["status"] == "ok"
    finally:
        del sys.modules["verbatim.querying.verdict_v2"]


def test_verdict_structural_fallback_provisional(monkeypatch):
    _block(monkeypatch, "verbatim.querying.verdict_v2")
    register_lane(LaneName.LEX, fake_lane("lex", ["l0"]))
    result = run(make_ctx(), make_query())
    assert result.verdict == ResultStatus.READY
    note = result.coverage.rerank["stages"]["verdict"]
    assert note["status"] == "unavailable"
    assert note["provisional"] is True


def test_verdict_structural_fallback_empty_is_insufficient(monkeypatch):
    _block(monkeypatch, "verbatim.querying.verdict_v2")
    register_lane(LaneName.LEX, fake_lane("lex", []))
    result = run(make_ctx(), make_query())
    assert result.verdict == ResultStatus.INSUFFICIENT
    assert result.missing is not None
    assert "zero_candidates" in result.missing.facets["eligible"]


def test_pack_module_used_when_present():
    def assemble_pack(scored, query, ctx, max_tokens, limit, neighbor_n):
        assert max_tokens == 512
        assert limit == 3
        return {"mode": "packed", "items": scored[:2], "tokens": 42}

    mod = _fake_module("verbatim.retrieval.v7.pack", assemble_pack=assemble_pack)
    sys.modules["verbatim.retrieval.v7.pack"] = mod
    try:
        register_lane(LaneName.LEX, fake_lane("lex", ["a", "b", "c"]))
        ctx = make_ctx(manifest={"max_tokens": 512, "limit": 3})
        result = run(ctx, make_query())
        assert result.pack["mode"] == "packed"
        assert len(result.pack["items"]) == 2
        assert result.stage.fields["tokens"] == 42.0
        assert result.coverage.rerank["stages"]["pack"]["status"] == "ok"
    finally:
        del sys.modules["verbatim.retrieval.v7.pack"]


def test_pack_absent_raw_passthrough(monkeypatch):
    _block(monkeypatch, "verbatim.retrieval.v7.pack")
    register_lane(LaneName.LEX, fake_lane("lex", ["a", "b", "c"]))
    result = run(make_ctx(manifest={"limit": 2}), make_query())
    assert result.pack["mode"] == "raw_passthrough"
    assert [s.unit_id for s in result.pack["items"]] == ["a", "b"]
    assert result.coverage.rerank["stages"]["pack"]["reason"] == "module_absent"


def test_all_engine_modules_absent_end_to_end(monkeypatch):
    """Incremental integration: with zero engine modules landed the
    pipeline still returns an honest result."""
    monkeypatch.setattr(pipeline, "_lazy_import", lambda name: None)
    register_lane(LaneName.LEX, fake_lane("lex", ["a", "b"]))
    result = run(make_ctx(), make_query())
    assert isinstance(result, PipelineResult)
    assert result.lanes["lex"].status == LaneStatus.OK
    assert len(result.fused) == 2
    assert result.scored[0].score_family == "rrf/fusion-fallback"
    assert result.verdict == ResultStatus.READY
    assert result.coverage.budget["slices"] == "fallback_equal"
    stages = result.coverage.rerank["stages"]
    assert stages["fusion"]["reason"] == "module_absent"
    assert stages["features"]["reason"] == "module_absent"
    assert stages["boosts"]["reason"] == "module_absent"
    assert stages["verdict"]["provisional"] is True


def test_eligibility_recheck_filters_union():
    register_lane(LaneName.LEX, fake_lane("lex", ["u0", "u1", "u2"]))
    ctx = make_ctx(lanes=(LaneName.LEX,), eligible={"u0", "u2"})
    result = run(ctx, make_query())
    assert {f.unit_id for f in result.fused} == {"u0", "u2"}
    assert result.coverage.security["eligibility_recheck_dropped"] == 1


def test_seeds_propagate_to_manifest():
    register_lane(LaneName.LEX, fake_lane("lex", ["l0", "l1"]))
    register_lane(LaneName.ENT, fake_lane("ent", ["e0"]))
    ctx = make_ctx(lanes=(LaneName.LEX, LaneName.ENT, LaneName.GRAPH))
    run(ctx, make_query())
    # §32.6: top-10 of L-ent ∪ top-10 of L-lex, in policy order
    assert ctx.manifest["seeds"] == ["l0", "l1", "e0"]


def test_facets_run_through_lanes_and_merge():
    facet_q = make_query("facet sub-question")
    calls = []

    def lane(ctx, qv, slice):
        calls.append(qv.query)
        uid = "main0" if qv.query.startswith("what") else "facet0"
        return LaneOutput(
            lane="lex",
            status=LaneStatus.OK,
            candidates=[cand(uid, "lex", 1)],
            examined=1,
            eligible=1,
        )

    register_lane(LaneName.LEX, lane)
    ctx = make_ctx(lanes=(LaneName.LEX,), budget=BudgetClass.MID)
    result = run(ctx, make_query(facets=(facet_q,)))
    assert calls == ["what did alice say", "facet sub-question"]
    assert result.coverage.facets["ran"] == 1
    ids = {c.unit_id for c in result.lanes["lex"].candidates}
    assert ids == {"main0", "facet0"}
    tagged = [c for c in result.lanes["lex"].candidates if "facet" in c.signals]
    assert tagged and tagged[0].signals["facet"] == 0


# ---------------------------------------------------------------------------
# Explain + stage record (V7-05.15, §32.17)
# ---------------------------------------------------------------------------


def test_explain_payload_complete(monkeypatch):
    _block(monkeypatch, "verbatim.retrieval.v7.pack")  # passthrough decision
    register_lane(LaneName.LEX, fake_lane("lex", ["a", "b"]))
    register_lane(LaneName.DENSE, fake_lane("dense", ["b", "c"]))
    result = run(make_ctx(), make_query(), explain=True)
    ex = result.explain
    assert ex is not None
    assert ex["query"]["intent"] == "lookup"
    assert set(ex["lanes"].keys()) == {"lex", "dense"}
    assert ex["lanes"]["lex"]["candidates"][0]["unit_id"] == "a"
    assert ex["slices"]  # per-lane dealt slices
    items = {it["unit_id"]: it for it in ex["items"]}
    assert items["b"]["lane_ranks"] == {"lex": 2, "dense": 1}
    assert items["b"]["fused_rrf"] > 0
    assert items["b"]["pack"] == "passthrough"
    assert ex["verdict"]["status"] == "ready"
    # byte-stable: must serialize deterministically
    json.dumps(ex, sort_keys=True)


def test_stage_record_populated(monkeypatch):
    _block(monkeypatch, "verbatim.querying.verdict_v2")  # pin "ready"
    register_lane(LaneName.LEX, fake_lane("lex", ["a", "b"]))
    register_lane(LaneName.DENSE, fake_lane("dense", ["c"]))
    result = run(make_ctx(), make_query())
    st = result.stage
    assert st.kind == "search"
    for f in (
        "t_total", "t_union", "t_rrf", "t_rerank_feat", "t_rerank_ce",
        "t_boost", "t_verdict", "t_pack", "t_barrier", "t_analyze",
        "pool_R", "items", "status", "coverage_digest",
    ):
        assert f in st.fields, f
    assert st.fields["t_total"] >= 0
    assert st.t_lane.keys() == {"lex", "dense"}
    assert st.candidates == {"lex": 2, "dense": 1}
    assert st.fields["status"] == "ready"
    assert st.fields["pool_R"] == POOLS[BudgetClass.MID].rerank_pool
