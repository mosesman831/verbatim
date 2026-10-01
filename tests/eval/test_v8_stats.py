"""Durable tests for the V8 paired statistics (SPEC_V8 V8-15.05,
V8-00.06 — scenario K08).

Pins: question-level bootstrap reproducibility (same seed ⇒ identical
bounds), the ≥10,000-resample default, per-conversation delta reporting
with the cluster count, McNemar's exact binomial against a
hand-computed case, and the V8-00.06 keep-rule decision table.
"""

from __future__ import annotations

import pytest

from eval.v8 import stats as S


def _pairs(n=60, base_p=0.5, cand_p=0.7, seed=1):
    """Deterministic synthetic paired binary outcomes."""
    import random
    rng = random.Random(seed)
    return [
        (1.0 if rng.random() < base_p else 0.0,
         1.0 if rng.random() < cand_p else 0.0)
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# paired bootstrap — determinism + contract
# ---------------------------------------------------------------------------


def test_bootstrap_reproducible_same_seed():
    pairs = _pairs()
    r1 = S.paired_bootstrap(pairs, resamples=500, seed=42)
    r2 = S.paired_bootstrap(pairs, resamples=500, seed=42)
    assert r1 == r2
    assert r1["ci_lower"] == r2["ci_lower"]
    assert r1["ci_upper"] == r2["ci_upper"]


def test_bootstrap_default_resamples_is_spec_minimum():
    pairs = _pairs(n=20)
    r = S.paired_bootstrap(pairs, seed=0)
    assert r["resamples"] == 10_000  # V8-15.05 ≥ 10,000


def test_bootstrap_reports_delta_and_bounds():
    pairs = [(0.0, 1.0)] * 40 + [(1.0, 1.0)] * 60  # delta +0.4 constant
    r = S.paired_bootstrap(pairs, resamples=200, seed=7)
    assert r["n"] == 100
    assert r["delta"] == pytest.approx(0.4)
    # a strictly positive constant delta can never resample below 0
    assert r["ci_lower"] > 0.0
    assert r["ci_lower"] <= r["ci_upper"]


def test_bootstrap_alpha_configurable():
    pairs = _pairs(n=80)
    r95 = S.paired_bootstrap(pairs, resamples=500, seed=3, alpha=0.05)
    r50 = S.paired_bootstrap(pairs, resamples=500, seed=3, alpha=0.25)
    assert r95["alpha"] == 0.05 and r50["alpha"] == 0.25
    # wider alpha ⇒ tighter lower bound
    assert r50["ci_lower"] >= r95["ci_lower"]


def test_bootstrap_negative_delta_has_negative_lower():
    pairs = [(1.0, 0.0)] * 100  # candidate strictly worse
    r = S.paired_bootstrap(pairs, resamples=200, seed=0)
    assert r["delta"] == pytest.approx(-1.0)
    assert r["ci_lower"] <= 0.0


def test_bootstrap_cluster_reporting():
    pairs = _pairs(n=50)
    conv = ["conv_a"] * 20 + ["conv_b"] * 20 + ["conv_c"] * 10
    r = S.paired_bootstrap(
        pairs, resamples=200, seed=5, conversation_ids=conv)
    cl = r["clusters"]
    assert cl["count"] == 3
    per = cl["per_conversation"]
    assert set(per) == {"conv_a", "conv_b", "conv_c"}
    assert per["conv_a"]["n"] == 20 and per["conv_c"]["n"] == 10
    # per-conversation deltas are the plain paired means — hand-check
    exp_a = (
        sum(p[1] for p in pairs[:20]) - sum(p[0] for p in pairs[:20])
    ) / 20
    assert per["conv_a"]["delta"] == pytest.approx(exp_a)


def test_bootstrap_cluster_unit_resamples_conversations():
    pairs = _pairs(n=30)
    conv = ["c1"] * 10 + ["c2"] * 10 + ["c3"] * 10
    r = S.paired_bootstrap(
        pairs, resamples=300, seed=9, conversation_ids=conv,
        unit="cluster")
    assert r["unit"] == "cluster"
    assert r["clusters"]["count"] == 3
    # deterministic under the cluster path too
    r2 = S.paired_bootstrap(
        pairs, resamples=300, seed=9, conversation_ids=conv,
        unit="cluster")
    assert r2["ci_lower"] == r["ci_lower"]


def test_bootstrap_input_validation():
    with pytest.raises(ValueError):
        S.paired_bootstrap([], resamples=10)
    with pytest.raises(ValueError):
        S.paired_bootstrap(_pairs(5), resamples=0)
    with pytest.raises(ValueError):
        S.paired_bootstrap(_pairs(5), resamples=10, alpha=0.9)
    with pytest.raises(ValueError):
        S.paired_bootstrap(_pairs(5), resamples=10,
                           conversation_ids=["only-one"])
    with pytest.raises(ValueError):
        S.paired_bootstrap(_pairs(5), resamples=10, unit="question?")


def test_bootstrap_mapping_pairs_accepted():
    pairs = [{"base": 0.0, "candidate": 1.0}] * 10
    r = S.paired_bootstrap(pairs, resamples=50, seed=0)
    assert r["delta"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# McNemar exact — hand-computed cases
# ---------------------------------------------------------------------------


def test_mcnemar_exact_hand_computed():
    # b=2, c=8 → n=10, k=2: tail = C(10,0)+C(10,1)+C(10,2) over 2^10
    # = (1 + 10 + 45) / 1024; two-sided = 112/1024 = 0.109375
    assert S.mcnemar_exact(2, 8) == pytest.approx(0.109375)
    assert S.mcnemar_exact(8, 2) == pytest.approx(0.109375)  # symmetric


def test_mcnemar_edge_cases():
    assert S.mcnemar_exact(0, 0) == 1.0          # no discordants
    assert S.mcnemar_exact(0, 1) == 1.0          # n=1: p = 2*(1/2)
    assert S.mcnemar_exact(0, 5) == pytest.approx(2 * (1 / 32))
    # b=0,c=9 → 2 * C(9,0)/2^9 = 2/512
    assert S.mcnemar_exact(0, 9) == pytest.approx(2 / 512)


def test_mcnemar_from_pairs():
    pairs = [
        (1, 0), (1, 0),          # base_only ×2
        (0, 1),                  # candidate_only ×1
        (1, 1), (0, 0),          # concordant ×2
    ]
    r = S.mcnemar_from_pairs(pairs)
    assert r["n"] == 5
    assert r["base_only"] == 2
    assert r["candidate_only"] == 1
    assert r["concordant"] == 2
    assert r["discordant"] == 3
    # n=3, k=1 → 2*(C(3,0)+C(3,1))/8 = 2*4/8 = 1.0
    assert r["p"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# the V8-00.06 keep rule
# ---------------------------------------------------------------------------


def test_keep_rule_default_on():
    r = S.evaluate_keep_rule(
        delta_any10=0.02, ci_lower=0.005,
        category_deltas={"1": 0.01, "2": 0.03})
    assert r["decision"] == "default_on"
    assert r["rejecting_metric"] is None
    assert r["rule_a"]["holds"] is True


def test_keep_rule_flag_off_weak_gain():
    r = S.evaluate_keep_rule(delta_any10=0.004, ci_lower=0.001)
    assert r["decision"] == "flag_off"
    assert r["rejecting_metric"] == "answerable.any@10"


def test_keep_rule_flag_off_lower_bound_zero():
    r = S.evaluate_keep_rule(delta_any10=0.02, ci_lower=-0.001)
    assert r["decision"] == "flag_off"
    assert r["rejecting_metric"] == "answerable.any@10"
    assert any("lower bound" in s for s in r["reasons"])


def test_keep_rule_fails_closed_without_stats():
    r = S.evaluate_keep_rule(delta_any10=None, ci_lower=None)
    assert r["decision"] == "flag_off"
    assert r["rule_a"]["holds"] is None


def test_keep_rule_structural_exempt_from_a_not_b():
    ok = S.evaluate_keep_rule(
        delta_any10=None, ci_lower=None, structural=True,
        category_deltas={"1": 0.0, "2": -0.002})
    assert ok["decision"] == "structural"

    bad = S.evaluate_keep_rule(
        delta_any10=None, ci_lower=None, structural=True,
        category_deltas={"1": -0.03, "2": 0.0})
    assert bad["decision"] == "flag_off"
    assert bad["rejecting_metric"].startswith("categories.1.")


def test_keep_rule_fix_kind_path_b():
    ok = S.evaluate_keep_rule(
        delta_any10=-0.01, ci_lower=-0.02, fix_kind="latency",
        category_deltas={"1": -0.004, "2": 0.0})
    assert ok["decision"] == "default_on"  # recall floor held

    bad = S.evaluate_keep_rule(
        delta_any10=-0.01, ci_lower=-0.02, fix_kind="sql",
        category_deltas={"1": -0.02})
    assert bad["decision"] == "flag_off"
    assert "categories.1" in bad["rejecting_metric"]


def test_keep_rule_category_regression_surfaces_warning():
    # passes rule (a) but regresses a category — ships under the
    # letter of the rule yet the regression is named, not hidden
    r = S.evaluate_keep_rule(
        delta_any10=0.02, ci_lower=0.005,
        category_deltas={"1": 0.01, "3": -0.02})
    assert r["decision"] == "default_on"
    assert r["rule_b"]["holds"] is False
    assert any("3" in w for w in r["warnings"])
