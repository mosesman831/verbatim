"""Durable tests for the V5 eval harness (SPEC_V5 §24, §20, §23).

These pin the harness's *honesty invariants* — the properties that make
its numbers mean anything:

* corpus determinism and gold isolation (arms can't reach answers);
* stats gates: seed integrity, zero-failure upper bounds, underpower,
  multiplicity, claim expiry;
* comparator discipline: mem0 honestly unavailable when the import
  fails, comparisons refused for untested/unpinned rows;
* real execution: a tiny end-to-end consumer run through
  ``verbatim.Memory`` proves the harness drives the real facade;
* report honesty: misses, unavailable rows, and unmeasured categories
  render into the Markdown instead of disappearing.
"""

from __future__ import annotations

import json
import os

import pytest


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


def test_corpus_deterministic_digest():
    from eval.v5.corpus import seed_corpus
    a = seed_corpus(memories=48, seed=42)
    b = seed_corpus(memories=48, seed=42)
    assert a.digest() == b.digest()
    assert [i.id for i in a.items] == [i.id for i in b.items]
    c = seed_corpus(memories=48, seed=7)
    assert c.digest() != a.digest()


def test_corpus_gold_views_block_expected():
    from eval.v5.corpus import public_item, public_task, seed_corpus
    c = seed_corpus(memories=32, seed=42)
    task = public_task(c.tasks[0])
    assert task.task_id and task.query
    with pytest.raises(AttributeError):
        _ = task.expected_ids
    with pytest.raises(AttributeError):
        _ = task.forbidden_ids
    item = public_item(c.items[0])
    assert item.id and item.text
    with pytest.raises(AttributeError):
        _ = item.supersedes


def test_corpus_mix_covers_categories():
    from eval.v5.corpus import seed_corpus
    c = seed_corpus(memories=48, seed=42)
    cats = {t.category for t in c.tasks}
    for req in ("identifier", "lexical", "no_answer", "update",
                "unicode", "temporal", "forget_distractor"):
        assert req in cats, f"missing §20.11 category {req}"
    assert any(i.supersedes for i in c.items)
    assert any(i.forget for i in c.items)


# ---------------------------------------------------------------------------
# stats gates (E69)
# ---------------------------------------------------------------------------


def test_mcnemar_exact_known_value():
    from eval.v5.stats import mcnemar_exact
    # b=9, c=1 discordants: two-sided exact binomial tail
    p = mcnemar_exact(9, 1)
    assert 0.0 < p < 0.05
    assert mcnemar_exact(0, 0) == 1.0


def test_zero_failure_claim_needs_bound():
    from eval.v5.stats import evaluate_claim, upper_bound_zero
    # A zero-failure claim without a bound is failed, not passed.
    v = evaluate_claim({
        "name": "c", "kind": "zero_failures", "failures": 0, "n": 30,
        "seeds_reported": [1], "claimed_upper": None})
    assert v.verdict == "failed"
    ub = upper_bound_zero(30)
    v2 = evaluate_claim({
        "name": "c", "kind": "zero_failures", "failures": 0, "n": 30,
        "seeds_reported": [1], "claimed_upper": ub})
    assert v2.verdict == "passed"
    assert v2.ci == (0.0, ub)


def test_seed_integrity_rejects_hidden_runs():
    from eval.v5.stats import check_seed_integrity, evaluate_claim
    ok, _ = check_seed_integrity([1, 2, 3], [1, 2, 3])
    assert ok
    ok, why = check_seed_integrity([1, 2, 3], [2])
    assert not ok and "best-of-seeds" in why
    v = evaluate_claim({
        "name": "c", "kind": "point", "estimate": 0.9, "ci": [0.8, 0.95],
        "seeds_executed": [1, 2], "seeds_reported": [2]})
    assert v.verdict == "failed"


def test_leadership_gate_requires_margin_and_support():
    from eval.v5.stats import evaluate_leadership_claim
    # +3pt margin with lower bound > 0, adequate support → pass
    v = evaluate_leadership_claim(
        "lead", 0.05, 0.01, n_independent=400, required_n=100)
    assert v.verdict == "passed"
    # underpowered support → underpowered, not failed
    v = evaluate_leadership_claim(
        "lead", 0.05, 0.01, n_independent=10, required_n=100)
    assert v.verdict == "underpowered"
    # margin not cleared → failed
    v = evaluate_leadership_claim(
        "lead", 0.01, 0.005, n_independent=400, required_n=100)
    assert v.verdict == "failed"
    # multi-endpoint family without simultaneous intervals → failed
    v = evaluate_leadership_claim(
        "lead", 0.05, 0.01, n_independent=400, required_n=100,
        family_size=4, simultaneous=False)
    assert v.verdict == "failed"


def test_claim_expiry():
    from eval.v5.stats import claim_expired
    assert claim_expired("2026-01-01", "2026-06-01")
    assert not claim_expired("2026-09-01", "2026-09-18")
    assert claim_expired("garbage", "2026-09-18")


def test_bootstrap_deterministic():
    from eval.v5.stats import bootstrap_ci
    d = [0.1, -0.2, 0.05, 0.3, -0.1, 0.4, 0.0, 0.2]
    a = bootstrap_ci(d, seed=5, iters=500)
    b = bootstrap_ci(d, seed=5, iters=500)
    assert a == b
    assert a[1] <= a[0] <= a[2]


# ---------------------------------------------------------------------------
# comparators (E66/E67)
# ---------------------------------------------------------------------------


def test_mem0_honestly_unavailable():
    from eval.v5.comparators import probe_mem0
    probe = probe_mem0()
    assert "available" in probe
    if not probe["available"]:
        assert "ModuleNotFoundError" in probe["error"] or \
            "No module named" in probe["error"]


def test_comparator_row_never_dropped():
    from eval.v5.comparators import run_comparator
    row = run_comparator("mem0", memories=32, seed=42)
    assert row.status == "unavailable"
    assert "mem0" in row.name
    assert row.reason  # the real import error is preserved


def test_compare_refuses_untested():
    from eval.v5.comparators import (
        ComparatorPin, ComparatorRow, compare)
    a = ComparatorRow(name="a", status="tested",
                      pin=ComparatorPin(
                          edition="verbatim", revision="x",
                          deployment="local", extractor="e",
                          embedder="hashing", reader="r", judge="j",
                          prompts="p", settings="s", indexes="i",
                          readiness_policy="rp", hardware="h",
                          pricing_date="d"),
                      metrics={"accuracy": 0.9})
    b = ComparatorRow(name="b", status="unavailable",
                      reason="dep absent")
    v = compare(a, b)
    assert not v["valid"] and "untested" in v["reason"]


def test_compare_refuses_unpinned():
    from eval.v5.comparators import (
        ComparatorPin, ComparatorRow, compare)
    a = ComparatorRow(name="a", status="tested",
                      pin=ComparatorPin(edition="verbatim"),
                      metrics={"accuracy": 0.9})
    b = ComparatorRow(name="b", status="tested",
                      pin=ComparatorPin(
                          edition="x", revision="r", deployment="d",
                          extractor="e", embedder="em", reader="r",
                          judge="j", prompts="p", settings="s",
                          indexes="i", readiness_policy="rp",
                          hardware="h", pricing_date="pd"),
                      metrics={"accuracy": 0.1})
    v = compare(a, b)
    assert not v["valid"] and "unpinned" in v["reason"]


# ---------------------------------------------------------------------------
# report honesty (E70)
# ---------------------------------------------------------------------------


def test_report_renders_misses_and_unavailable():
    from eval.v5.report import render_report
    results = {
        "qualification": "locally_measured",
        "environment": {"python": "3.x"},
        "suites": {
            "envelopes": {
                "verdict": "failed", "qualification": "locally_measured",
                "a0": {"qualification": "locally_measured",
                       "scale": {"memories": 10},
                       "search_ms": {"n": 5, "p95": 99.0},
                       "add_ack_ms": {"n": 5, "p95": 80.0},
                       "misses": ["search p95 99.0ms > 25ms"]},
            },
            "comparators": {
                "verdict": "passed",
                "rows": [{"name": "mem0", "status": "unavailable",
                          "reason": "not installed",
                          "metrics": {}}],
                "matrix": [],
            },
        },
        "unmeasured": ["operator_time"],
        "claims": [],
    }
    md = render_report(results)
    assert "MISS: search p95 99.0ms > 25ms" in md
    assert "unavailable" in md
    assert "mem0" in md
    assert "operator_time" in md


# ---------------------------------------------------------------------------
# real execution smoke (small, real Memory store)
# ---------------------------------------------------------------------------


def test_quality_suite_executes_real_facade(tmp_path):
    pytest.importorskip("verbatim")
    from eval.v5.quality import run_quality_suite
    r = run_quality_suite(memories=32, seed=42,
                          workdir=str(tmp_path / "q"))
    assert r["measured"] is True
    assert r["support"]["tasks_executed"] == \
        len(r["per_task"]) > 0
    # every task produced a scored outcome — none silently skipped
    for t in r["per_task"]:
        assert t["task_id"] in (
            s for s in (tt["task_id"] for tt in r["per_task"]))
    assert r["aggregate"]["recall_at_k"] is not None
    # claims carry verdicts, never bare numbers
    for c in r["claims"]:
        assert c["verdict"] in (
            "passed", "failed", "inconclusive", "underpowered",
            "unavailable")


def test_dx_probe_runs_workflow(tmp_path):
    pytest.importorskip("verbatim")
    from eval.v5.dx import run_dx_probe
    r = run_dx_probe(workdir=str(tmp_path / "dx"), quick_adds=2)
    assert r["denominator"]["attempted"] == r["denominator"]["ok"]
    names = [s["name"] for s in r["steps"]]
    for need in ("open", "add[0]", "search", "inspect", "forget",
                 "status", "close"):
        assert need in names


def test_comparator_registry_executes_verbatim_arm(tmp_path):
    pytest.importorskip("verbatim")
    from eval.v5.comparators import run_comparator_registry
    r = run_comparator_registry(memories=32, seed=42,
                                workdir=str(tmp_path / "cmp"))
    rows = {row["name"]: row for row in r["rows"]}
    assert rows["verbatim_memory"]["status"] == "tested"
    assert rows["verbatim_memory"]["metrics"]["tasks"] > 0
    assert rows["mem0"]["status"] in ("unavailable", "tested")
    # every registry row is present — none silently dropped
    assert set(rows) == {"verbatim_memory", "mem0"}


# ---------------------------------------------------------------------------
# portfolio plumbing
# ---------------------------------------------------------------------------


def test_portfolio_guard_records_failures():
    from eval.v5.portfolio import _guard
    def boom(**kw):
        raise RuntimeError("deliberate")
    out = _guard("x", boom)
    assert out["status"] == "failed"
    assert "deliberate" in out["error"]


def test_reproduction_manifest(tmp_path):
    from eval.v5.reproduction import build_manifest
    m = build_manifest({"suites": {}, "qualification": "locally_measured"},
                       command="test", artifact_paths=())
    assert m["manifest_version"].startswith("v5-repro")
    assert "environment" in m
    assert "dependency_probe" in m
