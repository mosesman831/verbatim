"""Tests for the v3 evaluation harness (SPEC_V3 §53–§56).

Covers: suite declaration and registration, the ledger's evidence-kind
completion rule, denominator preservation, inconclusive-vs-pass honesty,
the seeded 616-requirement registry, the report generator's
not_estimable readiness rule, an end-to-end conformance case on a real
``Store.create``, and the F27 investigation's attributable recall
numbers.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from eval.v3 import (
    CONFORMANCE,
    PAIRED_USEFULNESS,
    RETRIEVAL_QUALITY,
    SECURITY_PRIVACY,
    SUITE_IDS,
    Case,
    CaseResult,
    Ledger,
    Outcome,
    Suite,
    load_ledger,
    run_suite,
    wilson_interval,
)
from eval.v3.f27 import PathRecorder, SAVED_BASELINE, _attribute
from eval.v3.harness import is_valid_report
from eval.v3.ledger import (
    EVIDENCE_KINDS,
    required_evidence_for_section,
)
from eval.v3.registry import (
    LEDGER_PATH,
    parse_requirements,
    seed_ledger,
)
from eval.v3.report_v3 import (
    competitor_claim_readiness,
    render_markdown,
)
from eval.v3.suites import (
    base_metrics,
    conformance_case,
    declare_suites,
    make_suite,
    paired_case,
    paired_delta_interval,
    retrieval_case,
    security_case,
    suite_verdict,
)

from verbatim.storage.store import Store


# ---------------------------------------------------------------------------
# suite declaration / case schema (§53)
# ---------------------------------------------------------------------------


def test_four_suites_declared():
    decl = declare_suites()
    assert set(decl) == set(SUITE_IDS) == {
        CONFORMANCE, RETRIEVAL_QUALITY, PAIRED_USEFULNESS,
        SECURITY_PRIVACY,
    }
    # only the paired suite declares control arms
    arms = decl[PAIRED_USEFULNESS]["control_arms"]
    assert "no_memory" in arms and "memory" in arms
    assert "oracle" in arms  # diagnostic ceiling, never a result
    assert decl[CONFORMANCE]["control_arms"] == []


def test_case_schema_fields():
    c = conformance_case(
        "c1", ["V3-56.01"], check="x", expected=True, tags=["denied"]
    )
    d = c.to_dict()
    for k in ("case_id", "requirement_ids", "input", "expected", "tags"):
        assert k in d
    assert d["requirement_ids"] == ["V3-56.01"]
    assert d["tags"] == ["denied"]
    assert Case.from_dict(d) == c


def test_case_constructors_shape():
    r = retrieval_case("q1", "windows 11", expect=["s1"], kind="point")
    assert r.input["query"] == "windows 11"
    assert r.expected["expect"] == ["s1"]
    p = paired_case("t1", {"task_id": "t"}, arms=["no_memory", "memory"])
    assert p.input["arms"] == ["no_memory", "memory"]
    s = security_case(
        "a1", ["V3-34.01"], probe="poison_write", stage="3",
        benign_twin=True,
    )
    assert "stage:3" in s.tags and "benign_twin" in s.tags
    assert s.input["probe"] == "poison_write"


def test_case_validation():
    with pytest.raises(ValueError):
        Case(case_id="x", suite="not_a_suite")
    with pytest.raises(ValueError):
        Case(case_id="", suite=CONFORMANCE)
    with pytest.raises(ValueError):
        Case(case_id="x", suite=CONFORMANCE, min_support=0)


def test_suite_rejects_foreign_case():
    foreign = retrieval_case("q", "t", expect=["s"])
    with pytest.raises(ValueError):
        make_suite(
            CONFORMANCE, name="bad", cases=[foreign],
            runner=lambda c, ctx: True,
        )


# ---------------------------------------------------------------------------
# interval math — stdlib only, honest about empty denominators
# ---------------------------------------------------------------------------


def test_wilson_interval_properties():
    lo, hi = wilson_interval(50, 100)
    assert 0.0 < lo < 0.5 < hi < 1.0
    # wider interval at smaller n
    lo_s, hi_s = wilson_interval(5, 10)
    assert hi_s - lo_s > hi - lo
    # perfect and zero rates stay in bounds
    lo_p, hi_p = wilson_interval(100, 100)
    assert hi_p <= 1.0 and lo_p > 0.9
    lo_z, hi_z = wilson_interval(0, 100)
    assert lo_z < 0.01 and hi_z < 0.1


def test_wilson_empty_denominator_is_undefined_not_perfect():
    # §54.11: no support -> (0,0); callers must treat as inconclusive,
    # never as 1.0 or 0.0 evidence
    assert wilson_interval(0, 0) == (0.0, 0.0)
    assert wilson_interval(-1, 10) == (0.0, 0.0)
    assert wilson_interval(11, 10) == (0.0, 0.0)


def test_paired_delta_interval():
    est, lo, hi = paired_delta_interval(50, wins_memory=10, wins_control=2)
    assert est == pytest.approx(0.16)
    assert lo < est < hi
    assert paired_delta_interval(0, 0, 0) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# ledger completion (§56.01)
# ---------------------------------------------------------------------------


def _ledger() -> Ledger:
    lg = Ledger()
    lg.add_requirement("V3-53.01", section=53)
    lg.add_requirement("V3-07.01", section=7)
    lg.add_requirement("V3-62.01", section=62)
    return lg


def test_required_evidence_by_section():
    assert required_evidence_for_section(53) == (
        "impl_test", "integration", "empirical"
    )
    assert required_evidence_for_section(7) == ("impl_test", "integration")
    assert required_evidence_for_section(62) == ("integration",)


def test_compute_completion_requires_all_kinds():
    lg = _ledger()
    # no evidence -> unverified
    assert lg.compute_completion("V3-53.01") == "unverified"
    # one kind -> partial, never pass (V3-56.01)
    lg.register("V3-53.01", "impl_test", "tests/x.py::t", "pass")
    assert lg.compute_completion("V3-53.01") == "partial"
    lg.register("V3-53.01", "integration", "run-1", "pass")
    assert lg.compute_completion("V3-53.01") == "partial"
    lg.register("V3-53.01", "empirical", "eval/out.json", "pass")
    assert lg.compute_completion("V3-53.01") == "pass"
    assert lg.is_complete("V3-53.01")


def test_partial_and_inconclusive_evidence_never_pass():
    lg = _ledger()
    lg.register("V3-07.01", "impl_test", "tests/x.py::t", "pass")
    lg.register("V3-07.01", "integration", "run", "inconclusive")
    assert lg.compute_completion("V3-07.01") == "partial"
    # declared-but-unvalidated does not count either
    lg2 = _ledger()
    lg2.register("V3-07.01", "impl_test", "tests/x.py::t", "pass")
    lg2.register("V3-07.01", "integration", "run", "declared")
    assert lg2.compute_completion("V3-07.01") == "partial"


def test_failed_required_kind_fails():
    lg = _ledger()
    lg.register("V3-07.01", "impl_test", "tests/x.py::t", "pass")
    lg.register("V3-07.01", "integration", "run", "fail")
    assert lg.compute_completion("V3-07.01") == "fail"


def test_register_is_idempotent_and_validated():
    lg = _ledger()
    lg.register("V3-07.01", "impl_test", "tests/x.py::t", "declared")
    lg.register("V3-07.01", "impl_test", "tests/x.py::t", "pass")
    rec = lg.requirements["V3-07.01"]
    assert len(rec.evidence) == 1 and rec.evidence[0].status == "pass"
    with pytest.raises(ValueError):
        lg.register("V3-07.01", "bogus_kind", "a", "pass")
    with pytest.raises(ValueError):
        lg.register("V3-07.01", "impl_test", "a", "bogus_status")
    with pytest.raises(KeyError):
        lg.register("V3-99.99", "impl_test", "a", "pass")


def test_coverage_report():
    lg = _ledger()
    lg.register("V3-07.01", "impl_test", "t", "pass")
    cov = lg.coverage_report(gates={"G1": "not_run"})
    assert cov["total_requirements"] == 3
    assert cov["verdicts"]["partial"] == 1
    assert cov["verdicts"]["unverified"] == 2
    assert "V3-07.01" in cov["incomplete_requirements"]
    assert cov["gates"]["G1"] == "not_run"
    assert cov["evidence_kinds"]["impl_test"]["registered"] == 1


def test_ledger_roundtrip(tmp_path):
    lg = _ledger()
    lg.register("V3-07.01", "impl_test", "t", "pass", run_id="r1")
    p = lg.write(str(tmp_path / "ledger.json"))
    lg2 = load_ledger(p)
    assert lg2.compute_completion("V3-07.01") == "partial"
    assert lg2.requirements["V3-07.01"].evidence[0].run_id == "r1"


# ---------------------------------------------------------------------------
# registry — all 616 V3-NN.MM requirements seeded pending (§56.10)
# ---------------------------------------------------------------------------


def test_registry_parses_all_requirements():
    reqs = parse_requirements()
    assert len(reqs) == 616
    ids = {r["req_id"] for r in reqs}
    for expected in ("V3-53.01", "V3-54.11", "V3-56.01", "V3-33.01"):
        assert expected in ids


def test_registry_seed_file_complete_and_pending():
    lg = load_ledger(LEDGER_PATH)
    assert len(lg.requirements) == 616
    for rec in lg.requirements.values():
        assert rec.status == "pending"
        assert rec.required_evidence  # frozen per §56.01
        for k in rec.required_evidence:
            assert k in EVIDENCE_KINDS


def test_seed_ledger_is_idempotent(tmp_path):
    lg = seed_ledger()
    assert len(lg.requirements) == 616
    # reseeding preserves already-registered evidence
    lg.register("V3-53.01", "impl_test", "t", "pass")
    lg2 = seed_ledger(ledger=lg)
    assert lg2.requirements["V3-53.01"].evidence[0].status == "pass"


# ---------------------------------------------------------------------------
# harness — denominators, inconclusive, errors (§53.05, §54.11)
# ---------------------------------------------------------------------------


def _trivial_runner(case, ctx):
    return CaseResult(case.case_id, Outcome.PASS)


def _conformance_suite(runner=None, **kw):
    cases = kw.pop(
        "cases",
        [conformance_case("c1", ["V3-56.01"], check="ok")],
    )
    return make_suite(
        CONFORMANCE, name="toy", cases=cases,
        runner=runner or _trivial_runner, **kw,
    )


def test_run_suite_produces_valid_report(tmp_path):
    suite = _conformance_suite()
    rep = run_suite(suite, lambda: Store.create(str(tmp_path / "s.db")))
    assert is_valid_report(rep)
    assert rep["verdict"] == "pass"
    counts = rep["metrics"]["counts"]
    assert counts["executed"] == 1 and counts["pass"] == 1
    assert rep["metrics"]["pass_rate"] == 1.0
    assert rep["metrics"]["ci95"][0] > 0 and rep["metrics"]["ci95"][1] == 1.0


def test_error_stays_in_denominator(tmp_path):
    def boom(case, ctx):
        raise RuntimeError("kaboom")

    suite = _conformance_suite(runner=boom)
    rep = run_suite(suite, lambda: Store.create(str(tmp_path / "e.db")))
    counts = rep["metrics"]["counts"]
    assert counts["executed"] == 1 and counts["error"] == 1
    assert rep["verdict"] == "fail"
    assert rep["metrics"]["pass_rate"] == 0.0


def test_pass_without_support_becomes_inconclusive(tmp_path):
    def thin(case, ctx):
        # claims pass with support 0 — must never stand (§54.11)
        return CaseResult(case.case_id, Outcome.PASS, support=0)

    suite = _conformance_suite(runner=thin)
    rep = run_suite(suite, lambda: Store.create(str(tmp_path / "i.db")))
    r = rep["results"][0]
    assert r["outcome"] == "inconclusive"
    assert rep["verdict"] == "inconclusive"
    assert rep["metrics"]["counts"]["inconclusive"] == 1
    assert rep["inconclusive"] == ["c1"]


def test_prepare_failure_fails_whole_suite(tmp_path):
    suite = _conformance_suite(prepare=lambda f: 1 / 0)
    rep = run_suite(suite, lambda: Store.create(str(tmp_path / "p.db")))
    assert rep["verdict"] == "fail"
    assert rep["prepare_error"]
    assert rep["metrics"]["counts"]["error"] == 1


def test_suite_verdict_rules():
    suite = _conformance_suite(
        cases=[
            conformance_case("a", ["V3-56.01"], check="x"),
            conformance_case(
                "b", ["V3-56.01"], check="y", required=False
            ),
        ]
    )
    ok = [
        CaseResult("a", Outcome.PASS),
        CaseResult("b", Outcome.FAIL),
    ]
    # non-required failure doesn't fail the suite
    assert suite_verdict(suite, ok) == "pass"
    bad = [CaseResult("a", Outcome.ERROR), CaseResult("b", Outcome.PASS)]
    assert suite_verdict(suite, bad) == "fail"
    gap = [CaseResult("a", Outcome.INCONCLUSIVE),
           CaseResult("b", Outcome.PASS)]
    assert suite_verdict(suite, gap) == "inconclusive"


def test_base_metrics_denominators():
    rs = [
        CaseResult("1", Outcome.PASS),
        CaseResult("2", Outcome.FAIL),
        CaseResult("3", Outcome.ERROR),
        CaseResult("4", Outcome.SKIPPED),
    ]
    m = base_metrics(rs)
    assert m["counts"]["executed"] == 3  # skipped stays out of the rate
    assert m["counts"]["skipped"] == 1
    assert m["pass_rate"] == pytest.approx(1 / 3)


def test_bool_and_outcome_runner_returns(tmp_path):
    suite = _conformance_suite(runner=lambda c, ctx: True)
    rep = run_suite(suite, lambda: Store.create(str(tmp_path / "b.db")))
    assert rep["results"][0]["outcome"] == "pass"


def test_end_to_end_conformance_on_real_store(tmp_path):
    """A real conformance case: assert observable store behavior, not
    source strings (§56.02)."""
    def check_fts_enabled(case, ctx):
        # ctx is the prepared store
        return CaseResult(
            case.case_id,
            Outcome.PASS if ctx.fts_enabled else Outcome.FAIL,
            metrics={"fts_enabled": ctx.fts_enabled},
        )

    suite = _conformance_suite(
        runner=check_fts_enabled,
        cases=[conformance_case(
            "fts5_available", ["V3-29.01"], check="fts_enabled"
        )],
    )
    rep = run_suite(suite, lambda: Store.create(str(tmp_path / "v3.db")))
    assert rep["verdict"] in ("pass", "fail")  # never inconclusive here
    assert rep["results"][0]["metrics"]["fts_enabled"] in (True, False)


# ---------------------------------------------------------------------------
# report generator — §54.13 readiness + §55.02 sections
# ---------------------------------------------------------------------------


def _paired_report(executed: bool) -> dict:
    arms = {}
    if executed:
        arms = {
            "no_memory": {"executed": True, "success": False},
            "memory": {"executed": True, "success": True},
        }
    return {
        "suite_id": PAIRED_USEFULNESS,
        "verdict": "pass" if executed else "inconclusive",
        "declared_cases": 1, "selected_cases": 1,
        "results": [{
            "case_id": "t1", "outcome": "pass" if executed else "inconclusive",
            "support": 1, "metrics": {"arms": arms},
        }],
        "metrics": {"counts": {"executed": 1, "pass": 1 if executed else 0}},
        "inconclusive": [] if executed else ["t1"],
        "gates": ["G7", "G8"],
    }


def test_competitor_readiness_not_estimable_without_paired_arms():
    assert competitor_claim_readiness(_paired_report(False)) == "not_estimable"
    # non-paired suites are never estimable
    assert competitor_claim_readiness(
        {"suite_id": RETRIEVAL_QUALITY, "results": []}
    ) == "not_estimable"
    # arms present but not executed -> still not estimable
    rep = _paired_report(False)
    rep["results"][0]["metrics"]["arms"] = {
        "no_memory": {"executed": False},
        "memory": {"executed": False},
    }
    assert competitor_claim_readiness(rep) == "not_estimable"


def test_competitor_readiness_estimable_only_when_executed():
    assert competitor_claim_readiness(_paired_report(True)) == "estimable"


def test_render_markdown_sections(tmp_path):
    suite = _conformance_suite()
    rep = run_suite(suite, lambda: Store.create(str(tmp_path / "r.db")))
    md = render_markdown([rep])
    assert "## Suites" in md
    assert "Inconclusive" in md
    assert "Competitor-claim readiness" in md
    assert "not_estimable" in md
    # denominators are visible
    assert "1/1" in md


# ---------------------------------------------------------------------------
# F27 — attributable recall numbers (V3-03.01, V3-53.15, B49)
# ---------------------------------------------------------------------------


def test_f27_saved_baseline_constants():
    assert SAVED_BASELINE["historical_pre_rebuild"]["hit_rate"] == 0.58
    assert SAVED_BASELINE["historical_post_rebuild"]["hit_rate"] == 0.89
    assert SAVED_BASELINE["historical_pre_rebuild"]["n"] == 200


def test_f27_path_recorder_last_call_wins():
    r = PathRecorder()
    assert r.path == "none" and r.candidate_ids == []
    r.record("fts_repo", ["c1", "c2"])
    r.record("scan", ["c3"])
    assert r.path == "scan" and r.candidate_ids == ["c3"]
    r.clear()
    assert r.path == "none"


def test_f27_attribution_categories():
    base_pre = {
        "hit": False, "path": "fts_repo",
        "expected_indexed_at_generation": False,
        "expected_candidate_rank": None,
    }
    base_post = {
        "hit": True, "path": "fts_repo",
        "expected_indexed_at_generation": True,
        "expected_candidate_rank": 1,
    }
    tags = _attribute(base_pre, base_post)
    assert "generation_stranded" in tags and "gained" in tags
    # rank shift: candidate both phases, different rank
    pre2 = dict(base_pre, expected_indexed_at_generation=True,
                expected_candidate_rank=4)
    post2 = dict(base_post, expected_candidate_rank=1)
    tags2 = _attribute(pre2, post2)
    assert "rank_shift" in tags2
    # packed out: was a candidate pre but never reached the result
    pre3 = dict(pre2)
    post3 = dict(post2, hit=False, expected_candidate_rank=1)
    tags3 = _attribute(pre3, post3)
    assert "packed_out_pre" in tags3 or "packed_out_post" in tags3
    # stable hit
    tags4 = _attribute(
        dict(base_pre, hit=True, expected_indexed_at_generation=True),
        base_post,
    )
    assert "stable_hit" in tags4


def test_f27_investigation_attributable(tmp_path):
    """End-to-end: small corpus through the public path, asserting the
    report carries attributable per-query numbers (denominators, path,
    generation, ranks) — not a verdict on the size of the delta."""
    from eval.v3.f27 import investigate_f27, summarize_attribution

    rep = investigate_f27(
        corpus_size=120, seed=42, work_dir=str(tmp_path / "f27"),
        query_kinds=("historical",),
    )
    try:
        assert rep["investigation"] == "F27"
        assert rep["recall"]["pre_rebuild"]["n"] == len(rep["per_query"])
        assert rep["recall"]["post_rebuild"]["n"] == len(rep["per_query"])
        assert rep["per_query"], "expected historical queries to be scored"
        for q in rep["per_query"]:
            pre = q["pre"]
            assert pre["path"] in ("fts_repo", "fts_sql", "scan", "none")
            assert isinstance(pre["projection_generation"], int)
            assert isinstance(pre["expected_indexed_at_generation"], bool)
            assert pre["expected_candidate_rank"] is None or (
                isinstance(pre["expected_candidate_rank"], int)
                and pre["expected_candidate_rank"] >= 1
            )
            assert isinstance(pre["hit"], bool)
            assert isinstance(q["attribution"], list) and q["attribution"]
        # attribution summary is well-formed and covers every query
        s = summarize_attribution(rep)
        assert s["queries"] == len(rep["per_query"])
        oc = rep["outcome_counts"]
        assert sum(oc.values()) == len(rep["per_query"])
        # the saved baseline target travels with the report
        assert rep["saved_baseline"]["historical_pre_rebuild"]["hits"] == 116
    finally:
        json.dumps(rep)  # report must stay JSON-serializable


def test_f27_json_serializable_structure():
    # cheap structural check without running a corpus
    from eval.v3.f27 import SAVED_BASELINE as sb
    json.dumps(sb)


def test_conformance_suite_records_outcomes():
    """The conformance runner maps collected pytest nodes into
    requirement-tagged records and keeps unavailable lanes declared —
    exercised on the smallest file spec for speed."""
    import eval.v3.suite_conformance as sc

    specs = tuple(s for s in sc._FILE_SPECS if s.surface == "mcp-stdio")
    assert specs, "mcp-stdio spec missing"
    orig = sc._FILE_SPECS
    sc._FILE_SPECS = specs
    try:
        run = sc.run_conformance_suite(None, "verbatim_v3", quiet=True)
    finally:
        sc._FILE_SPECS = orig
    assert run.suite == "conformance" and run.baseline == "verbatim_v3"
    assert run.records, "no conformance records collected"
    for rec in run.records:
        assert rec["node"].startswith("tests/test_v2_mcp.py::")
        assert rec["requirement_ids"] == ("V3-48.01",)
        assert rec["result"] in ("passed", "failed", "skipped")
    m = run.metrics
    assert m["cases_total"] == len(run.records)
    assert m["cases_total"] == (
        m["cases_passed"] + m["cases_failed"] + m["cases_skipped"]
    )
    assert {u["surface"] for u in m["unavailable_surfaces"]} == {
        "hermes-live", "adk-live",
    }
    assert set(run.unavailable_lanes) == {"hermes-live", "adk-live"}


def test_conformance_a_scenario_tags():
    """Every test_aNN node in the acceptance file resolves to a
    non-empty requirement tuple — the §56.01 tagging contract."""
    from eval.v3.suite_conformance import _a_reqs, _A_REQS

    for key, reqs in _A_REQS.items():
        assert reqs and all(r.startswith("V3-") for r in reqs), key
    assert _a_reqs("tests/x.py::test_a15b_extra") == _A_REQS["a15"]
    assert _a_reqs("tests/x.py::test_unrelated") == ()
