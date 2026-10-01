"""Paired-comparison statistics for SPEC_V8 (V8-15.05, V8-00.06).

The LoCoMo dev partition has five conversations — too few clusters for
a cluster bootstrap to carry weight — so dev comparisons run a paired
**question-level** bootstrap (≥ 10,000 resamples) beside McNemar's
exact test on binary per-question outcomes, and print the
per-conversation deltas and cluster count next to them (V8-15.05,
scenario K08).  ``unit="cluster"`` resamples whole conversations for
splits with enough clusters (test partition, O13) — the unit is
declared in the result, never implied.

Everything here is pure stdlib and deterministic for a fixed seed —
the seed is part of the published record, matching the manifest
convention (``eval.v5.stats`` precedent; the exact-binomial McNemar
form mirrors its ``mcnemar_exact``).

Also home to :func:`evaluate_keep_rule`, the V8-00.06 decision a
ledger record asserts: ``default_on`` needs Δ answerable any@10
≥ +0.010 with a bootstrap lower bound > 0 (rule a), or a
latency/SQL/trust/correctness fix with recall change ≥ −0.005 on every
category (rule b); ``structural`` is exempt from (a), never from (b).
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: V8-15.05 — dev comparisons resample ≥ 10,000 times.
DEFAULT_RESAMPLES = 10_000
DEFAULT_ALPHA = 0.05

#: Resampling units: per-question (dev — few clusters) or per-cluster
#: (splits with enough independent conversations).
RESAMPLE_UNITS = ("question", "cluster")

#: V8-00.06 thresholds.
KEEP_MIN_GAIN = 0.010        # rule (a): Δ answerable any@10
KEEP_MAX_REGRESS = -0.005    # rule (b): recall change per category

#: Fix kinds eligible for keep-rule path (b) (V8-00.06).
FIX_KINDS = ("latency", "sql", "trust", "correctness")


# ---------------------------------------------------------------------------
# input normalization
# ---------------------------------------------------------------------------


def _pair_value(p: Any) -> Tuple[float, float]:
    """One (base, candidate) observation — a 2-sequence or a mapping
    with ``base``/``candidate`` keys."""
    if isinstance(p, Mapping):
        return float(p["base"]), float(p["candidate"])
    return float(p[0]), float(p[1])


def _binary_pair(p: Any) -> Tuple[int, int]:
    b, c = _pair_value(p)
    return (1 if b else 0), (1 if c else 0)


# ---------------------------------------------------------------------------
# paired bootstrap (V8-15.05)
# ---------------------------------------------------------------------------


def paired_bootstrap(
    pairs: Sequence[Any],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
    conversation_ids: Optional[Sequence[Any]] = None,
    unit: str = "question",
) -> Dict[str, Any]:
    """Paired question-level bootstrap of the mean per-question delta.

    ``pairs`` holds one ``(base, candidate)`` score per question
    (0/1 any@k values, prop@k fractions, reciprocal ranks, …).  The
    resampled statistic is ``mean(candidate − base)``; ``ci_lower`` is
    the one-sided percentile bound at ``alpha`` — the number the
    V8-00.06 keep rule compares to zero.

    ``conversation_ids`` (aligned with ``pairs``) attaches the
    per-conversation delta table and the cluster count V8-15.05 prints
    beside every dev comparison.  ``unit="cluster"`` additionally makes
    the resampling draw whole conversations — honest only when the
    split has enough clusters to resample.

    Deterministic: same pairs + seed + resamples ⇒ identical bounds.
    """
    vals = [_pair_value(p) for p in pairs]
    n = len(vals)
    if n == 0:
        raise ValueError("paired_bootstrap: pairs must be non-empty")
    if resamples < 1:
        raise ValueError("paired_bootstrap: resamples must be >= 1")
    if not (0.0 < alpha < 0.5):
        raise ValueError("paired_bootstrap: alpha must be in (0, 0.5)")
    if unit not in RESAMPLE_UNITS:
        raise ValueError(
            f"paired_bootstrap: unit must be one of {RESAMPLE_UNITS}")
    conv = None
    if conversation_ids is not None:
        conv = [str(c) for c in conversation_ids]
        if len(conv) != n:
            raise ValueError(
                "paired_bootstrap: conversation_ids length "
                f"{len(conv)} != pairs length {n}")

    deltas = [c - b for b, c in vals]
    delta = sum(deltas) / n
    base_mean = sum(b for b, _ in vals) / n
    cand_mean = sum(c for _, c in vals) / n

    rng = random.Random(int(seed))
    means: List[float] = []
    if unit == "cluster" and conv is not None:
        # Whole-conversation resampling — the declared cluster unit.
        by_conv: Dict[str, List[float]] = {}
        for cid, d in zip(conv, deltas):
            by_conv.setdefault(cid, []).append(d)
        labels = sorted(by_conv)
        clusters = [by_conv[c] for c in labels]
        k = len(clusters)
        for _ in range(resamples):
            picked = rng.choices(clusters, k=k)
            pool = [x for c in picked for x in c]
            means.append(sum(pool) / len(pool))
    else:
        # Question-level resampling (V8-15.05 dev rule).  Also the
        # fallback when unit="cluster" was requested without ids —
        # recorded honestly in ``unit`` below.
        unit = "question"
        for _ in range(resamples):
            means.append(sum(rng.choices(deltas, k=n)) / n)
    means.sort()
    lo = means[max(0, int(alpha * resamples) - 1)]
    hi = means[min(resamples - 1, int((1.0 - alpha) * resamples))]

    clusters_block = None
    if conv is not None:
        per: Dict[str, Dict[str, Any]] = {}
        for cid in sorted(set(conv)):
            idx = [i for i, c in enumerate(conv) if c == cid]
            cb = sum(vals[i][0] for i in idx) / len(idx)
            cc = sum(vals[i][1] for i in idx) / len(idx)
            per[cid] = {
                "n": len(idx),
                "base_mean": cb,
                "candidate_mean": cc,
                "delta": cc - cb,
            }
        clusters_block = {
            "count": len(per),
            "per_conversation": per,
        }

    return {
        "schema": "v8_paired_stats/v1",
        "stat": "mean_delta",
        "n": n,
        "base_mean": base_mean,
        "candidate_mean": cand_mean,
        "delta": delta,
        "ci_lower": lo,
        "ci_upper": hi,
        "alpha": float(alpha),
        "resamples": int(resamples),
        "seed": int(seed),
        "unit": unit,
        "clusters": clusters_block,
    }


# ---------------------------------------------------------------------------
# McNemar's exact test (V8-15.05)
# ---------------------------------------------------------------------------


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value over discordant pairs.

    Under the null the ``n = b + c`` discordants split binomially with
    p = 0.5; the two-sided exact p doubles the smaller tail (capped at
    1.0).  Same closed form as ``eval.v5.stats.mcnemar_exact`` —
    re-implemented so v8 carries no cross-generation import.
    """
    b, c = int(b), int(c)
    n = b + c
    if n <= 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar_from_pairs(pairs: Sequence[Any]) -> Dict[str, Any]:
    """McNemar over per-question binary outcomes (hit/miss at k).

    ``pairs`` are ``(base, candidate)`` binary values.  ``base_only``
    counts questions the base alone got right; ``candidate_only`` the
    reverse — the discordant cells the exact test consumes.
    """
    b_only = c_only = concordant = 0
    for p in pairs:
        b, c = _binary_pair(p)
        if b and not c:
            b_only += 1
        elif c and not b:
            c_only += 1
        else:
            concordant += 1
    n = b_only + c_only + concordant
    return {
        "schema": "v8_mcnemar/v1",
        "n": n,
        "discordant": b_only + c_only,
        "base_only": b_only,
        "candidate_only": c_only,
        "concordant": concordant,
        "p": mcnemar_exact(b_only, c_only),
    }


# ---------------------------------------------------------------------------
# the V8-00.06 keep rule
# ---------------------------------------------------------------------------


def evaluate_keep_rule(
    *,
    delta_any10: Optional[float],
    ci_lower: Optional[float],
    category_deltas: Optional[Mapping[str, float]] = None,
    structural: bool = False,
    fix_kind: Optional[str] = None,
    min_gain: float = KEEP_MIN_GAIN,
    max_regress: float = KEEP_MAX_REGRESS,
    metric: str = "any@10",
) -> Dict[str, Any]:
    """Evaluate one landed requirement against the V8-00.06 keep rule.

    Returns ``{"decision": ..., "rejecting_metric": ...|None, ...}``
    where ``decision`` is ``default_on | flag_off | structural`` — the
    vocabulary the ledger record carries (V8-15.04).  ``fix_kind`` is
    one of :data:`FIX_KINDS` (a latency/SQL/trust/correctness fix
    eligible for path b); ``structural`` marks an owner-declared
    structural requirement exempt from (a) but not (b).

    A retrieval change shipping on (a) while a category regresses past
    (b)'s floor still ships under the letter of the rule — the
    regression is surfaced in ``warnings`` so the decision record names
    it rather than hiding it.
    """
    cats = {str(k): float(v) for k, v in (category_deltas or {}).items()}
    violations = {
        c: d for c, d in cats.items() if d < max_regress
    }
    rule_b_holds = not violations
    worst = min(violations.items(), key=lambda kv: kv[1])[0] \
        if violations else None

    have_stats = delta_any10 is not None and ci_lower is not None
    rule_a_holds = bool(
        have_stats
        and float(delta_any10) >= min_gain
        and float(ci_lower) > 0.0
    )

    rule_a = {
        "delta": delta_any10,
        "ci_lower": ci_lower,
        "min_gain": min_gain,
        "holds": rule_a_holds if have_stats else None,
    }
    rule_b = {
        "max_regress": max_regress,
        "metric": metric,
        "violations": violations,
        "holds": rule_b_holds,
    }

    reasons: List[str] = []
    warnings: List[str] = []
    rejecting: Optional[str] = None

    if not rule_b_holds:
        reject_detail = (
            f"categories.{worst}.{metric} "
            f"({violations[worst]:+.4f} < {max_regress:+.3f})")

    if structural:
        kind = "structural"
        if rule_b_holds:
            decision = "structural"
        else:
            decision = "flag_off"
            rejecting = reject_detail
            reasons.append(
                "structural requirement regressed recall past the "
                f"−0.005 floor on {worst} (exempt from (a), not (b))")
    elif fix_kind in FIX_KINDS:
        kind = f"fix:{fix_kind}"
        if rule_b_holds:
            decision = "default_on"
            reasons.append(
                f"{fix_kind} fix inside the −0.005 recall floor on "
                "every category (V8-00.06 (b))")
        else:
            decision = "flag_off"
            rejecting = reject_detail
            reasons.append(
                f"{fix_kind} fix regressed recall on {worst} beyond "
                "the −0.005 floor (V8-00.06 (b))")
    else:
        kind = "retrieval"
        if rule_a_holds:
            decision = "default_on"
            reasons.append(
                f"Δ answerable any@10 {delta_any10:+.4f} ≥ "
                f"+{min_gain:.3f} with lower bound {ci_lower:+.4f} > 0")
            if not rule_b_holds:
                warnings.append(
                    f"category recall regression on {worst} "
                    f"({violations[worst]:+.4f}) — name it in the "
                    "decision record")
        else:
            decision = "flag_off"
            rejecting = "answerable.any@10"
            if not have_stats:
                reasons.append(
                    "no paired statistics — a keep claim without a "
                    "bootstrap lower bound fails closed")
            else:
                if float(delta_any10) < min_gain:
                    reasons.append(
                        f"Δ answerable any@10 {delta_any10:+.4f} "
                        f"< +{min_gain:.3f}")
                if float(ci_lower) <= 0.0:
                    reasons.append(
                        f"bootstrap lower bound {ci_lower:+.4f} ≤ 0")
            if not rule_b_holds:
                reasons.append(
                    f"also regressed on {reject_detail}")

    return {
        "schema": "v8_keep_rule/v1",
        "decision": decision,
        "rejecting_metric": rejecting,
        "kind": kind,
        "reasons": reasons,
        "warnings": warnings,
        "rule_a": rule_a,
        "rule_b": rule_b,
    }


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_RESAMPLES",
    "FIX_KINDS",
    "KEEP_MAX_REGRESS",
    "KEEP_MIN_GAIN",
    "RESAMPLE_UNITS",
    "evaluate_keep_rule",
    "mcnemar_exact",
    "mcnemar_from_pairs",
    "paired_bootstrap",
]
