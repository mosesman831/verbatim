"""v4.5 M5 bake-off harness + disposition table (SPEC_V4_5 §09 X1,
§10–§11, §16; D13, D21, D22, D24).

X1 is the exact-then-semantic comparison surface: every registry row is
named tested/unavailable/out_of_scope with a reason (V45-10.01), tested
rows carry the full V4-54.01 pin set, a win over a weakened comparator
is refused (V45-10.04, D13), native/controlled tracks never mix
(V45-10.02), and the eight-category cost accounting never presents a
partial total as complete (V45-10.05). The disposition table keeps every
I1–I6 and X1–X8 row — failed, deferred, and research-only mechanisms
stay visible and out of recommended routing (V45-09.09, V45-16.01, D21).
"""

from __future__ import annotations

import pytest

from eval.v45 import bakeoff
from eval.v45 import dispositions


def _row(cid, **kw):
    base = dict(
        comparator_id=cid,
        edition="oss",
        track=bakeoff.TRACK_CONTROLLED,
        status=bakeoff.STATUS_TESTED,
        commit="deadbeef",
        deployment="local_in_process",
        models={"encoder": "hashing:subword-ngram:v1",
                "reader": "reader-v1"},
        prompts="p1",
        extraction={"harvester": "v3"},
        budgets={"k": 5},
        readiness="drained",
        pricing_date="2026-09-01",
        metrics={"aggregate": {"recall_at_k": 0.5}},
        costs=bakeoff.CostBreakdown(ingest=1.0),
    )
    base.update(kw)
    return bakeoff.ComparatorRow(**base)


# ---------------------------------------------------------------------------
# registry + claim set (V45-10.01, D22)
# ---------------------------------------------------------------------------


def test_registry_names_every_comparator():
    rows = bakeoff.registry()
    ids = [r.comparator_id for r in rows]
    # The parent-registry claim set is fully enumerated — nothing is
    # silently dropped (V45-10.01, D22).
    assert len(ids) == len(set(ids)) == len(bakeoff._REGISTRY_DECLS)
    statuses = {r.status for r in rows}
    assert statuses <= set(bakeoff.STATUSES)
    # Hosted-only/platform editions with no runnable probe are
    # out_of_scope with an explicit reason — never attributed to an OSS
    # edition and never marked tested (V45-10.03).
    by_id = {r.comparator_id: r for r in rows}
    hosted_only = [
        cid for (cid, ed, _t, probe, _r) in bakeoff._REGISTRY_DECLS
        if probe is None and ed in ("hosted", "platform")
    ]
    assert hosted_only
    for cid in hosted_only:
        assert by_id[cid].status == bakeoff.STATUS_OUT_OF_SCOPE
        assert by_id[cid].reason
    # Every unavailable/out_of_scope row carries a reason.
    assert all(
        r.reason for r in rows
        if r.status != bakeoff.STATUS_TESTED
    )
    # OSS editions without an installed package are unavailable, named,
    # reasoned — and never marked tested.
    oss = [r for r in rows if r.comparator_id == "mem0_oss"]
    assert len(oss) == 1
    assert oss[0].status in (
        bakeoff.STATUS_UNAVAILABLE,
        bakeoff.STATUS_TESTED,
    )
    if oss[0].status == bakeoff.STATUS_UNAVAILABLE:
        assert "not installed" in oss[0].reason
    # The controlled local set is declared tested.
    local = {"verbatim_oss", "no_memory", "lexical_reference",
             "neural_reference", "verbatim_previous_stable"}
    assert local <= {r.comparator_id for r in rows
                     if r.status == bakeoff.STATUS_TESTED}


def test_claim_set_separates_statuses():
    rows = [
        _row("tested_a"),
        _row("gone", status=bakeoff.STATUS_UNAVAILABLE,
             reason="not installed", metrics={}),
        _row("hosted", status=bakeoff.STATUS_OUT_OF_SCOPE,
             reason="hosted", metrics={}),
    ]
    cs = bakeoff.claim_set(rows)
    assert cs["tested"] == ["tested_a"]
    assert cs["unavailable"] == ["gone"]
    assert cs["out_of_scope"] == ["hosted"]
    assert cs["defeated"] == []  # never inferred


# ---------------------------------------------------------------------------
# compare — the refusal matrix (V45-10.02/03/04, V45-11.03/04, D13)
# ---------------------------------------------------------------------------


def test_weakened_comparator_cannot_lose():
    """D13: a disabled-embedding comparator is named — it cannot count
    as a competitor loss."""
    a = _row("verbatim", metrics={"aggregate": {"recall_at_k": 0.9}})
    weak = _row("mem0_weakened",
                metrics={"aggregate": {"recall_at_k": 0.2}},
                weakened=("disabled_embeddings",))
    verdict = bakeoff.compare(a, weak)
    assert verdict["valid"] is False
    assert verdict["winner"] is None
    assert "weakened_comparator" in verdict["reason"]
    assert "disabled_embeddings" in verdict["reason"]


def test_weakened_self_arm_also_invalid():
    a = _row("verbatim",
             metrics={"aggregate": {"recall_at_k": 0.9}},
             weakened=("truncated_context",))
    b = _row("other", metrics={"aggregate": {"recall_at_k": 0.2}})
    verdict = bakeoff.compare(a, b)
    assert verdict["valid"] is False


def test_native_controlled_tracks_separate():
    """V45-10.02: a native-track row never compares with a controlled
    row — the tracks are separate rows in one claim set."""
    native = _row("mem0_native", track=bakeoff.TRACK_NATIVE)
    controlled = _row("verbatim", track=bakeoff.TRACK_CONTROLLED)
    verdict = bakeoff.compare(native, controlled)
    assert verdict["valid"] is False
    assert "track_mismatch" in verdict["reason"]


def test_unavailable_row_is_never_defeated():
    a = _row("verbatim", metrics={"aggregate": {"recall_at_k": 1.0}})
    gone = _row("zep_hosted", status=bakeoff.STATUS_UNAVAILABLE,
                reason="hosted", metrics={})
    verdict = bakeoff.compare(a, gone)
    assert verdict["valid"] is False
    assert "untested" in verdict["reason"]


def test_unpinned_row_refused():
    a = _row("verbatim")
    b = _row("loose", commit=None, pricing_date=None)
    verdict = bakeoff.compare(a, b)
    assert verdict["valid"] is False
    assert "unpinned" in verdict["reason"]
    assert "commit" in verdict["reason"]


def test_reader_mismatch_refused():
    a = _row("verbatim")
    b = _row("other",
             models={"encoder": "e", "reader": "reader-v2"})
    verdict = bakeoff.compare(a, b)
    assert verdict["valid"] is False
    assert "reader_mismatch" in verdict["reason"]


def test_synthesis_classes_never_mixed():
    a = _row("verbatim")
    b = _row("provider_native", synthesis="provider_native")
    verdict = bakeoff.compare(a, b)
    assert verdict["valid"] is False
    assert "synthesis_mismatch" in verdict["reason"]


def test_valid_comparison_reports_delta():
    a = _row("verbatim", metrics={"aggregate": {"recall_at_k": 0.9}})
    b = _row("other", metrics={"aggregate": {"recall_at_k": 0.6}})
    verdict = bakeoff.compare(a, b)
    assert verdict["valid"] is True
    assert verdict["winner"] == "verbatim"
    assert verdict["delta"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# cost accounting (V45-10.05)
# ---------------------------------------------------------------------------


def test_total_cost_categories():
    """The eight V4-54.07 categories exist; an unmeasured category is
    labeled, and a partial total is never presented complete."""
    assert set(bakeoff.COST_CATEGORIES) == {
        "ingest", "extraction", "embeddings", "consolidation",
        "query_inference", "answer_reading", "storage", "maintenance",
    }
    partial = bakeoff.CostBreakdown(ingest=10.0, storage=2048.0)
    total = partial.total()
    assert total["complete"] is False
    assert set(total["unmeasured"]) == set(bakeoff.COST_CATEGORIES) - {
        "ingest", "storage"
    }
    assert total["measured_total"] == pytest.approx(2058.0)
    full = bakeoff.CostBreakdown(
        **{c: 1.0 for c in bakeoff.COST_CATEGORIES}
    )
    assert full.total()["complete"] is True


def test_row_validation():
    good = _row("ok")
    assert bakeoff.validate_row(good) == []
    untested_with_metrics = _row(
        "bad", status=bakeoff.STATUS_UNAVAILABLE,
        metrics={"aggregate": {"recall_at_k": 1.0}},
    )
    problems = bakeoff.validate_row(untested_with_metrics)
    assert problems


# ---------------------------------------------------------------------------
# run() — the executed local set
# ---------------------------------------------------------------------------


def test_run_executes_local_arms_and_names_everyone():
    report = bakeoff.run()
    assert report["kind"] == "v45_bakeoff"
    assert report["gold_protected"] is True
    rows = {r["comparator_id"]: r for r in report["rows"]}
    # Every registry comparator is present exactly once.
    assert len(rows) == len(bakeoff._REGISTRY_DECLS)
    cs = report["claim_set"]
    assert set(cs["tested"]) | set(cs["unavailable"]) | set(
        cs["out_of_scope"]
    ) == set(rows)
    # The verbatim controlled row carries the full pin set.
    v = rows["verbatim_oss"]
    assert v["status"] == "tested"
    assert v["pinned"] is True
    pinning = v["pinning"]
    assert pinning["commit"]
    assert pinning["deployment"] == "local_in_process"
    assert pinning["models"]["reader"] == report["reader_model"]
    assert pinning["budgets"]["k"] >= 1
    assert pinning["readiness"]
    assert pinning["pricing_date"] == report["pricing_date"]
    # Measured metrics + honest cost surface.
    assert v["metrics"]["aggregate"]["tasks"] == 2
    assert v["costs"]["total"]["complete"] is False
    assert "maintenance" in v["costs"]["total"]["unmeasured"]
    # Hosted editions stay out of scope — never tested rows.
    assert rows["mem0_platform"]["status"] == "out_of_scope"
    # Unavailable comparators are named, never counted.
    assert cs["defeated"] == []


def test_run_comparisons_refuse_nothing_for_local_refs():
    report = bakeoff.run()
    verdicts = {
        c["b"]: c for c in report["comparisons"]
    }
    # The executed reference arms produce valid measured comparisons
    # against the verbatim controlled arm (same reader, same budgets).
    for ref in ("no_memory", "lexical_reference",
                "verbatim_previous_stable"):
        assert ref in verdicts
        assert verdicts[ref]["valid"] is True


def test_run_extra_row_validated():
    bad = _row("ghost", status=bakeoff.STATUS_UNAVAILABLE,
               metrics={"aggregate": {"recall_at_k": 1.0}})
    with pytest.raises(ValueError, match="cannot carry metrics"):
        bakeoff.run(extra_rows=[bad])


def test_run_unknown_arm_rejected():
    with pytest.raises(ValueError, match="unknown arms"):
        bakeoff.run(arms=["not_an_arm"])


def test_weakened_extra_row_blocks_win():
    """D13 end-to-end: inject a weakened competitor row — the baked
    comparison must come back invalid, never a win."""
    weak = _row(
        "holographic",
        metrics={"aggregate": {"recall_at_k": 0.1}},
        weakened=("disabled_embeddings",),
    )
    report = bakeoff.run(extra_rows=[weak])
    v = next(
        c for c in report["comparisons"] if c["b"] == "holographic"
    )
    assert v["valid"] is False
    assert "weakened_comparator" in v["reason"]


# ---------------------------------------------------------------------------
# disposition table (V45-16.01, D21, D24)
# ---------------------------------------------------------------------------


def test_dispositions_cover_every_mechanism():
    rows = dispositions.load()
    ids = {r["id"] for r in rows}
    assert ids == set(dispositions.MECHANISM_IDS)
    for r in rows:
        assert r["disposition"] in dispositions.DISPOSITIONS
        assert "measured" in r
        assert isinstance(r["evidence"], list)
        assert isinstance(r["recommended"], bool)


def test_dispositions_file_matches_generator():
    assert dispositions.check() == []


def test_failed_mechanism_never_recommended():
    """D21: a mechanism that lost its ablation stays in the published
    table and out of recommended routing — the flag is the gate."""
    rows = dispositions.load()
    by_id = {r["id"]: r for r in rows}
    # X8 is the §16-prescribed research row: visible, never recommended.
    assert by_id["X8"]["disposition"] == "research_only"
    assert by_id["X8"]["recommended"] is False
    assert dispositions.is_recommended("X8") is False
    # Deferred I-experiments are named but not recommended — presence of
    # a runner file is not acceptance evidence.
    for iid in ("I1", "I2", "I3", "I4", "I5", "I6"):
        assert iid in by_id
        assert by_id[iid]["recommended"] is False
        assert dispositions.is_recommended(iid) is False
    # Recommended set = adopted guard mechanisms only.
    rec = set(dispositions.recommended_mechanisms())
    assert "X8" not in rec
    assert rec == {
        r["id"] for r in rows if r["disposition"] == "adopt"
    }
    # A non-adopt row can never carry recommended=true — the validator
    # enforces the D21 rule structurally.
    bad = [
        {
            "id": "I1",
            "mechanism": "x",
            "disposition": "reject",
            "recommended": True,
            "evidence": [],
            "measured": "unmeasured",
        }
    ]
    problems = dispositions.validate_rows(bad)
    assert any("recommended" in p for p in problems)


def test_d24_no_market_beating_claim_without_evidence():
    """D24: I1–I6 carry no paired outcomes — nothing in the table may be
    read as market-beating evidence."""
    rows = dispositions.load()
    for r in rows:
        if r["id"].startswith("I"):
            assert r["disposition"] == "defer"
            assert r["measured"] == "unmeasured"
