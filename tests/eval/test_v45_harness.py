"""Durable tests for the v4.5 measured-ablation harnesses
(SPEC_V4_5 §03/§05 anchors: D01/D02, D05/D06; V45-11.01 honesty).

Each harness runs a SMALL realized slice through the real
``recall_v3`` production path inside a disposable ``Store`` — the
durable properties are the report contract, the measured sample size,
the real token estimator output, and the honesty rules (numbers from
execution, no fabricated comparators, misses reported). Full
qualification numbers live in the generated report artifacts, not in
asserted constants.
"""

from __future__ import annotations

import json
import os

import pytest

from eval.v45 import corpus, i1_manifest, i3_sufficiency, i4_transfer
from eval.v45 import i2_repair_locality, i5_branch_apply, i6_refresh
from verbatim.core.types import ErrorCode


# ---------------------------------------------------------------------
# I1 — counterevidence-first (D01/D02)
# ---------------------------------------------------------------------

def test_i1_report_shape_and_sample_size(tmp_path):
    """The report carries the declared contract: experiment, spec rows,
    realized sample size, matched bound, per-topic arms, totals, and
    the D01/D02 verdicts — all fields present, none hard-coded."""
    report = i1_manifest.run_i1(topics=3, max_items=2,
                                workdir=str(tmp_path / "i1"))
    assert report["experiment"] == "i1_counterevidence_first"
    assert report["spec"]["acceptance"] == ["D01", "D02"]
    n = report["sample_size"]["topics"]
    assert n == 3
    assert report["sample_size"]["runs"] == 2 * n
    assert len(report["topics"]) == n
    for row in report["topics"]:
        assert set(row["arms"]) == {"topk", "manifest"}
        assert isinstance(
            row["arms"]["topk"]["false_current"], bool
        )
        assert isinstance(row["arms"]["manifest"]["tokens"], int)
    t = report["totals"]
    assert 0 <= t["topk_false_current"] <= n
    assert 0 <= t["manifest_false_current"] <= n
    assert t["identifier_hits"]["manifest"] >= 0
    assert "false_current_fell" in report["d02"]
    assert isinstance(report["met"], bool)


def test_i1_numbers_are_measured(tmp_path):
    """Every numeric claim derives from executed arms — the totals are
    the sums of per-topic outcomes, never constants."""
    report = i1_manifest.run_i1(topics=3, max_items=2,
                                workdir=str(tmp_path / "i1b"))
    rows = report["topics"]
    assert report["totals"]["topk_false_current"] == sum(
        1 for r in rows if r["arms"]["topk"]["false_current"]
    )
    assert report["totals"]["manifest_false_current"] == sum(
        1 for r in rows if r["arms"]["manifest"]["false_current"]
    )
    assert report["totals"]["tokens"]["topk"] == sum(
        r["arms"]["topk"]["tokens"] for r in rows
    )
    # the manifest doc is real on every manifest-arm run (D01 surface)
    for row in rows:
        assert row["manifest_producer"] == "deterministic_pack"
        assert isinstance(row["manifest_labels"], list)


def test_i1_false_current_definition_declared(tmp_path):
    """The measured predicate is declared in the report — a reader can
    audit what 'false-current' counted rather than trust the label."""
    report = i1_manifest.run_i1(topics=2, max_items=2,
                                workdir=str(tmp_path / "i1c"))
    assert "false_current" in report["definitions"]
    assert "contrary_disclosed" in report["definitions"]


def test_i1_reports_written(tmp_path):
    """JSON + MD land with the same measured numbers."""
    report = i1_manifest.run_i1(topics=2, max_items=2,
                                workdir=str(tmp_path / "i1d"))
    paths = i1_manifest.write_reports(report, str(tmp_path / "out"))
    assert os.path.exists(paths["json"]) and os.path.exists(paths["md"])
    loaded = json.load(open(paths["json"]))
    assert loaded["totals"] == report["totals"]


# ---------------------------------------------------------------------
# I3 — minimal-sufficient progressive (D05/D06)
# ---------------------------------------------------------------------

def test_i3_report_shape_and_real_tokens(tmp_path):
    """The token reduction is measured through ContextPack.tokens —
    flat and progressive totals are per-arm sums, the ratio derives
    from them, and per-pack estimates ride each topic row."""
    report = i3_sufficiency.run_i3(topics=3, depth=3,
                                   workdir=str(tmp_path / "i3"))
    assert report["experiment"] == "i3_minimal_sufficient_progressive"
    assert report["spec"]["acceptance"] == ["D05", "D06"]
    n = report["sample_size"]["topics"]
    assert n == 3 and len(report["topics"]) == n
    tok = report["tokens"]
    assert tok["flat_total"] == sum(
        r["flat"]["tokens"] for r in report["topics"]
    )
    assert tok["progressive_total"] == sum(
        r["progressive"]["tokens"] for r in report["topics"]
    )
    assert tok["flat_total"] > 0
    assert 0.0 <= tok["reduction_ratio"] <= 1.0
    # the estimator is the packer's own, declared
    assert "approx_chars_per_token" in report["token_estimator"]
    for r in report["topics"]:
        assert isinstance(r["flat"]["per_pack_tokens"], dict)
        assert isinstance(r["progressive"]["per_pack_tokens"], dict)


def test_i3_coverage_rule_and_identifier_coverage(tmp_path):
    """The declared coverage rule rides the report; identifier and
    condition coverage are measured on both arms (V45-05.02/05.03)."""
    report = i3_sufficiency.run_i3(topics=3, depth=3,
                                   workdir=str(tmp_path / "i3b"))
    assert report["coverage_rule"]
    cov = report["coverage"]
    n = report["sample_size"]["topics"]
    assert 0 <= cov["flat_covered"] <= n
    assert 0 <= cov["progressive_covered"] <= n
    assert cov["condition_text_topics"]["progressive"] <= n
    # sufficiency coverage report present per progressive arm
    for r in report["topics"]:
        scov = r["progressive"]["coverage"]
        assert scov is not None
        assert scov["pack_mode"] == "sufficiency"
        assert "required_identifiers" in scov
        assert "coverage_tier" in scov and "depth_tier" in scov


def test_i3_utility_and_non_inferiority_measured(tmp_path):
    """Utility is a per-topic fraction of declared markers; the
    one-point margin is applied to the measured means."""
    report = i3_sufficiency.run_i3(topics=3, depth=3,
                                   workdir=str(tmp_path / "i3c"))
    u = report["utility"]
    assert 0.0 <= u["flat_mean"] <= 1.0
    assert 0.0 <= u["progressive_mean"] <= 1.0
    assert u["margin"] == 0.01
    assert u["non_inferior"] == (
        u["progressive_mean"] >= u["flat_mean"] - u["margin"]
    )
    d = report["d05"]
    assert d["token_reduction_met"] == (
        report["tokens"]["reduction_ratio"] >= 0.20
    )
    assert isinstance(report["met"], bool)


def test_i3_expansion_and_d06_probe(tmp_path):
    """The expansion surface is real: roundtrips counted, and the
    revocation probe ran the real revoke→expand path (D06 anchor)."""
    report = i3_sufficiency.run_i3(topics=2, depth=3,
                                   workdir=str(tmp_path / "i3d"))
    exp = report["expansion"]
    assert exp["d06"]["anchor"] == (
        "tests/retrieval/test_disclosure_tiers.py"
    )
    if exp["d06"]["ref_obtained"]:
        assert exp["d06"]["revocation_denied"] is True
        assert exp["d06"]["denial_code"] is not None


def test_i3_reports_written(tmp_path):
    report = i3_sufficiency.run_i3(topics=2, depth=2,
                                   workdir=str(tmp_path / "i3e"))
    paths = i3_sufficiency.write_reports(report, str(tmp_path / "out"))
    assert os.path.exists(paths["json"]) and os.path.exists(paths["md"])
    loaded = json.load(open(paths["json"]))
    assert loaded["tokens"]["flat_total"] == report["tokens"]["flat_total"]
    md = open(paths["md"]).read()
    assert "minimal-sufficient" in md


# ---------------------------------------------------------------------
# corpus seeders — fixture sanity (the seeded store is real)
# ---------------------------------------------------------------------

def test_corpus_slices_seed_real_store(tmp_path):
    """Both slice seeders write real tables — conflict groups, edges,
    and hard-identifier claims exist for the arms to find."""
    store = corpus.make_store(str(tmp_path / "corp"))
    seeded = corpus.seed_stale_slice(store, topics=2)
    with store.read() as conn:
        n_conf = conn.execute(
            "SELECT COUNT(*) FROM conflict_groups"
        ).fetchone()[0]
        assert n_conf >= 2
    store.close()
    store2 = corpus.make_store(str(tmp_path / "corp2"), name="h.db")
    seeded2 = corpus.seed_history_slice(store2, topics=2, depth=2)
    with store2.read() as conn:
        n_edges = conn.execute(
            "SELECT COUNT(*) FROM edges WHERE edge_type='conflicts_with'"
        ).fetchone()[0]
        assert n_edges >= 2
        n_cond = conn.execute(
            "SELECT COUNT(*) FROM claim_revisions"
            " WHERE condition_json IS NOT NULL"
        ).fetchone()[0]
        assert n_cond >= 2
    store2.close()


# ---------------------------------------------------------------------
# I4 — qualified procedure transfer (D07/D08)
# ---------------------------------------------------------------------

def test_i4_report_shape_and_sample_size(tmp_path):
    """The report carries the declared contract: experiment, spec rows,
    realized sample size, declared host ground truth, per-scenario arm
    tables, and the D07/D08 verdicts — none hard-coded."""
    report = i4_transfer.run_i4(
        procedures=3, workdir=str(tmp_path / "i4")
    )
    assert report["experiment"] == "i4_qualified_transfer"
    assert report["spec"]["acceptance"] == ["D07", "D08"]
    n = report["sample_size"]["procedures_per_arm"]
    assert n == 3
    assert report["sample_size"]["tasks_per_arm"] == (
        n * len(i4_transfer.SCENARIOS)
    )
    for arm in ("positive_only", "failure_aware"):
        rows = report["arms"][arm]["rows"]
        assert len(rows) == report["sample_size"]["tasks_per_arm"]
        for row in rows:
            assert row["scenario"] in i4_transfer.SCENARIOS
            assert isinstance(row["transfer_success"], bool)
            assert isinstance(row["negative_transfer"], bool)
    assert "negative_transfer" in report["definitions"]
    assert "transfer_success" in report["definitions"]
    assert report["host_ground_truth"]["heldout"]["delivered"] == "failure"
    assert isinstance(report["met"], bool)


def test_i4_numbers_are_measured(tmp_path):
    """Every total derives from executed deliveries — per-scenario
    counts and grand totals are sums of the row-level verdicts from
    the real ``transfer_success`` calls."""
    report = i4_transfer.run_i4(
        procedures=3, workdir=str(tmp_path / "i4b")
    )
    for arm in ("positive_only", "failure_aware"):
        rows = report["arms"][arm]["rows"]
        totals = report["arms"][arm]["totals"]
        assert totals["deliveries"] == sum(
            1 for r in rows if r["deliverable"]
        )
        assert totals["negative_transfer"] == sum(
            1 for r in rows if r["negative_transfer"]
        )
        assert totals["transfer_success"] == sum(
            1 for r in rows if r["transfer_success"]
        )
        for sc in i4_transfer.SCENARIOS:
            bysc = report["arms"][arm]["by_scenario"][sc]
            assert bysc["negative_transfer"] == sum(
                1 for r in rows
                if r["scenario"] == sc and r["negative_transfer"]
            )


def test_i4_d07_structure(tmp_path):
    """D07 compares held-out negative transfer across arms and checks
    the success side is preserved in-env — the comparator arm must
    deliver where the qualified arm blocks."""
    report = i4_transfer.run_i4(
        procedures=3, workdir=str(tmp_path / "i4c")
    )
    d07 = report["d07"]
    n = report["sample_size"]["procedures_per_arm"]
    assert set(d07["heldout_negative_transfer"]) == {
        "positive_only", "failure_aware"
    }
    # The paired design: positive-only delivered into the held-out
    # environment and the reuse measurably failed there.
    assert (
        d07["heldout_negative_transfer"]["positive_only"]
        == report["arms"]["positive_only"]["by_scenario"]
        ["heldout"]["negative_transfer"]
    )
    # Failure-aware blocked every held-out delivery.
    assert d07["heldout_blocked"] == n
    assert d07["inenv_success_preserved"] is True
    assert isinstance(report["met"], bool)


def test_i4_d08_probes(tmp_path):
    """D08 probes: exposure-only and self-reported outcomes never count
    as transfer success, on every case on the qualified arm."""
    report = i4_transfer.run_i4(
        procedures=3, workdir=str(tmp_path / "i4d")
    )
    d08 = report["d08"]
    n = report["sample_size"]["procedures_per_arm"]
    assert d08["cases"] == n
    assert d08["exposed_only_not_counted"] == n
    assert d08["self_report_not_counted"] == n
    rows = report["arms"]["failure_aware"]["rows"]
    for r in rows:
        if r["scenario"] == "self_report":
            assert r["attested_outcome"] == "none"
            assert r["transfer_success"] is False


def test_i4_reports_written(tmp_path):
    report = i4_transfer.run_i4(
        procedures=2, workdir=str(tmp_path / "i4e")
    )
    paths = i4_transfer.write_reports(report, str(tmp_path / "out"))
    assert os.path.exists(paths["json"]) and os.path.exists(paths["md"])
    loaded = json.load(open(paths["json"]))
    assert loaded["totals"] == report["totals"]
    md = open(paths["md"]).read()
    assert "negative transfer" in md


# ---------------------------------------------------------------------
# I6 — utility-budgeted refresh (D11/D12)
# ---------------------------------------------------------------------

def test_i6_report_shape_and_sample_size(tmp_path):
    """The report carries the declared contract: experiment, spec rows,
    paired arm tables, the matched-freshness bound, and D11/D12 — all
    measured, none hard-coded."""
    report = i6_refresh.run_i6(
        periods=4, changes_per_period=2, workdir=str(tmp_path / "i6")
    )
    assert report["experiment"] == "i6_utility_budgeted_refresh"
    assert report["spec"]["acceptance"] == ["D11", "D12"]
    assert report["sample_size"]["periods"] == 4
    for arm in ("periodic", "budgeted"):
        assert len(report["arms"][arm]["periods"]) == 4
    assert "stale_answers" in report["definitions"]
    assert "spend" in report["definitions"]
    assert isinstance(report["met"], bool)


def test_i6_spend_is_measured(tmp_path):
    """Spend totals are the sums of per-period plan spend — and the
    periodic arm re-scans every claim every cycle, the real cost the
    budgeted arm avoids."""
    report = i6_refresh.run_i6(
        periods=4, changes_per_period=2, workdir=str(tmp_path / "i6b")
    )
    for arm in ("periodic", "budgeted"):
        rows = report["arms"][arm]["periods"]
        spend = report["arms"][arm]["spend"]
        assert spend["claims_processed"] == sum(
            r["claims_processed"] for r in rows
        )
        assert spend["jobs"] == sum(r["jobs"] for r in rows)
    n_claims = (
        report["sample_size"]["slots"] * i6_refresh.CLAIMS_PER_SLOT
    )
    # Periodic: every live claim re-scanned every cycle, twice over
    # when bounded reflection runs.
    periodic_rows = report["arms"]["periodic"]["periods"]
    assert all(r["claims_processed"] == 2 * n_claims
               for r in periodic_rows)
    # Budgeted: strictly less claim work — the D11 margin is real.
    bs = report["arms"]["budgeted"]["spend"]
    ps = report["arms"]["periodic"]["spend"]
    assert bs["claims_processed"] < ps["claims_processed"]


def test_i6_d12_probe_structure(tmp_path):
    """The owner probe ran under a saturated budget: the owner request
    scheduled and counted while ordinary candidates deferred, and the
    backlog was caught up afterwards (watermark-held rediscovery)."""
    report = i6_refresh.run_i6(
        periods=4, changes_per_period=2, workdir=str(tmp_path / "i6c")
    )
    d12 = report["d12"]
    assert d12["owner_scheduled"] >= 1
    assert d12["owner_deferred"] == 0
    assert d12["non_owner_deferred"] >= 1
    probe_rows = [
        r for r in report["arms"]["budgeted"]["periods"]
        if r["phase"] == "d12_probe"
    ]
    assert len(probe_rows) == 1
    assert probe_rows[0]["deferred"] >= 1
    # The catch-up cycle restored the freshness floor.
    assert (
        report["arms"]["budgeted"]["periods"][-1]
        ["stale_answers_after"] == 0
    )


def test_i6_freshness_floor_measured(tmp_path):
    """Matched freshness is a measured floor — zero stale answers at the
    end of the window on every arm, and steady-state cycles held it."""
    report = i6_refresh.run_i6(
        periods=4, changes_per_period=2, workdir=str(tmp_path / "i6d")
    )
    b = report["arms"]["budgeted"]
    assert b["freshness_floor_zero_stale_final"] is True
    assert b["steady_state_at_floor"] is True
    p = report["arms"]["periodic"]
    assert p["freshness_floor_zero_stale"] is True
    assert report["d11"]["matched_freshness"] is True
    assert report["d11"]["claims_savings"] >= 0.20


def test_i6_reports_written(tmp_path):
    report = i6_refresh.run_i6(
        periods=4, changes_per_period=2, workdir=str(tmp_path / "i6e")
    )
    paths = i6_refresh.write_reports(report, str(tmp_path / "out"))
    assert os.path.exists(paths["json"]) and os.path.exists(paths["md"])
    loaded = json.load(open(paths["json"]))
    assert loaded["totals"] == report["totals"]
    md = open(paths["md"]).read()
    assert "utility-budgeted" in md


# ---------------------------------------------------------------------
# I2 — repair-local recomputation (D03/D04)
# ---------------------------------------------------------------------

def test_i2_report_shape_and_sample_size(tmp_path):
    """The report carries the declared contract: experiment, spec rows,
    paired arm counters, the same-input bound, and D03/D04 — all
    measured, none hard-coded."""
    report = i2_repair_locality.run_i2(
        slots=8, workdir=str(tmp_path / "i2")
    )
    assert report["experiment"] == "i2_repair_locality"
    assert report["spec"]["acceptance"] == ["D03", "D04"]
    assert report["sample_size"]["slots"] == 8
    for arm in ("repair", "full_rebuild"):
        for key in (
            "objects_scanned", "objects_evaluated", "objects_recomputed",
        ):
            assert isinstance(report["arms"][arm][key], int)
    assert "objects_evaluated" in report["definitions"]
    assert isinstance(report["met"], bool)


def test_i2_locality_is_measured(tmp_path):
    """The repair arm's counters are strictly below the rebuild arm's —
    the same-input twin-store comparison is real work on both sides."""
    report = i2_repair_locality.run_i2(
        slots=8, workdir=str(tmp_path / "i2b")
    )
    rep, full = report["arms"]["repair"], report["arms"]["full_rebuild"]
    assert rep["objects_evaluated"] < full["objects_evaluated"]
    assert rep["objects_recomputed"] < full["objects_recomputed"]
    assert rep["objects_scanned"] < full["objects_scanned"]
    # The held/stale marks landed inside the correction's own commit.
    held = rep["held_marks_in_correction_tx"]
    assert held["views"] >= 1 and held["observations"] >= 1
    # Equal post-state: the corrected observation is live on both arms.
    assert report["d03"]["post_state_equal"] is True
    assert report["d03"]["met"] is True


def test_i2_d04_fences_measured(tmp_path):
    """Both incomplete-plan fences trip CONTEXT_INCOMPLETE with zero
    partial writes — the honest-refusal half of D04 is measured, not
    assumed."""
    report = i2_repair_locality.run_i2(
        slots=8, workdir=str(tmp_path / "i2c")
    )
    d04 = report["d04"]
    assert d04["incomplete_forecast"]["complete"] is False
    assert d04["incomplete_forecast"]["apply_refused"] is True
    assert (
        d04["incomplete_forecast"]["error_code"]
        == ErrorCode.CONTEXT_INCOMPLETE.value
    )
    assert d04["grown_closure"]["apply_refused"] is True
    assert (
        d04["grown_closure"]["error_code"]
        == ErrorCode.CONTEXT_INCOMPLETE.value
    )
    assert d04["partial_writes"] == 0
    assert report["d04_met"] is True


def test_i2_reports_written(tmp_path):
    report = i2_repair_locality.run_i2(
        slots=8, workdir=str(tmp_path / "i2d")
    )
    paths = i2_repair_locality.write_reports(report, str(tmp_path / "out"))
    assert os.path.exists(paths["json"]) and os.path.exists(paths["md"])
    loaded = json.load(open(paths["json"]))
    assert loaded["totals"] == report["totals"]
    md = open(paths["md"]).read()
    assert "repair-local" in md


# ---------------------------------------------------------------------
# I5 — reversible memory branches (D09/D10)
# ---------------------------------------------------------------------

def test_i5_report_shape_and_checks(tmp_path):
    """Every scenario ran and produced measured gate outcomes — the
    report declares the spec rows and per-scenario checks."""
    report = i5_branch_apply.run_i5(workdir=str(tmp_path / "i5"))
    assert report["experiment"] == "i5_branch_apply"
    assert report["spec"]["acceptance"] == ["D09", "D10"]
    assert len(report["scenarios"]) == 6
    for name in (
        "d09", "moved_parent", "d10_suppressed", "d10_purged",
        "held_rebase", "abandon_diff",
    ):
        assert isinstance(report["checks"][name], bool)


def test_i5_d09_isolation_and_review_gate(tmp_path):
    """The isolation/apply measurement: the proposed op is invisible to
    live recall until the reviewed apply lands; the review gate refuses
    first; replay is idempotent."""
    report = i5_branch_apply.run_i5(workdir=str(tmp_path / "i5b"))
    d09 = report["scenarios"]["d09_isolation_reviewed_apply"]
    assert d09["isolated_before_apply"] is True
    assert d09["review_gate_code"] == ErrorCode.INVALID_TRANSITION.value
    assert d09["applied"] is True
    assert d09["head_state_after"] == "archived"
    assert d09["recall_dropped_after_apply"] is True
    assert d09["idempotent_replay"] is True
    assert report["checks"]["d09"] is True


def test_i5_purge_propagation_measured(tmp_path):
    """D10: the real purge closure tombstones the branch and the apply
    fence refuses without restoring any bytes — measured from committed
    state, never fixture intent."""
    report = i5_branch_apply.run_i5(workdir=str(tmp_path / "i5c"))
    purged = report["scenarios"]["d10_purged_parent"]
    assert purged["apply_refused_typed"] is True
    assert purged["claim_revisions_states"] == ["active", "erased"]
    assert purged["no_resurrect_text"] is True
    assert purged["no_apply_receipt"] is True
    assert purged["branch_disposition"] == "erased"
    assert purged["tombstones_carry_no_text"] is True
    supp = report["scenarios"]["d10_suppressed_parent"]
    assert supp["apply_code"] == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED.value
    assert supp["branch_state_after"] == "live"
    assert report["checks"]["d10_purged"] is True
    assert report["checks"]["d10_suppressed"] is True


def test_i5_reports_written(tmp_path):
    report = i5_branch_apply.run_i5(workdir=str(tmp_path / "i5d"))
    paths = i5_branch_apply.write_reports(report, str(tmp_path / "out"))
    assert os.path.exists(paths["json"]) and os.path.exists(paths["md"])
    loaded = json.load(open(paths["json"]))
    assert loaded["checks"] == report["checks"]
    md = open(paths["md"]).read()
    assert "branches" in md
