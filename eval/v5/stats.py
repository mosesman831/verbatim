"""Paired-comparison statistics and §23 claim gates for the V5 harness
(E69 / V5-23.01–23.12).

Interval math is stdlib-only and reuses the v3 implementations
(``eval.v3.suites.wilson_interval`` / ``paired_delta_interval``) rather
than re-deriving them; the V5-specific layer on top is the *decision
discipline* the spec adds:

* **cluster awareness** (V5-23.02): paired tests resample at the
  declared cluster unit (history/task family), never per question —
  thousands of template variants are not thousands of independent
  observations;
* **power** (V5-23.03): a claim must carry an ``n`` justified by pilot
  discordance — ``required_independent_units`` computes the binomial
  minimum for a target detectable margin, and ``evaluate_claim``
  returns ``underpowered`` below it;
* **zero-failure honesty** (V5-23.08): zero observed failures report a
  denominator and an upper confidence bound — ``upper_bound_zero`` is
  the Wilson upper (rule-of-three scale), and a claim phrased as
  ``failures == 0`` (not ``upper bound ≤ x``) is rejected;
* **multiplicity** (V5-23.04): families wider than one endpoint require
  simultaneous intervals — ``bonferroni_alpha`` supplies the frozen
  correction and the gate refuses an uncorrected family claim;
* **seed picking** (V5-23.10): ``check_seed_integrity`` requires every
  executed seed to be reported — a claim resting on the best of several
  runs is ``failed``, not ``passed``;
* **expiry** (V5-23.12): ``claim_expired`` enforces the 90-day window.

``evaluate_claim`` is the single entry point the report generator calls;
it returns a :class:`ClaimVerdict` whose ``verdict`` is one of
``passed | failed | inconclusive | underpowered | unavailable`` — five
different outcomes, never a boolean masquerading as proof.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

#: Reuse the proven v3 interval implementations — one definition of
#: Wilson/paired-delta math across every eval generation.
from eval.v3.suites import (  # noqa: F401  (re-exported)
    paired_delta_interval,
    wilson_interval,
    wilson_lower_bound,
    wilson_upper_bound,
)

Z_95 = 1.959963984540054

#: V5-23.10 outcome vocabulary — five distinct outcomes.
VERDICTS = (
    "passed",
    "failed",
    "inconclusive",
    "underpowered",
    "unavailable",
)

#: Claim expiry (V5-23.12): claims expire 90 days after measurement or
#: on material comparator change, whichever is earlier.
CLAIM_TTL_DAYS = 90


# ---------------------------------------------------------------------------
# exact tests — pure math, no scipy
# ---------------------------------------------------------------------------


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value over discordant pairs.

    ``b`` = memory-only wins, ``c`` = control-only wins. Under the null
    the discordants split binomially with p=0.5; the two-sided exact
    p-value doubles the smaller tail (capped at 1.0).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(
        math.comb(n, i) for i in range(0, k + 1)
    ) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def paired_binary_test(wins_a: int, wins_b: int, n: int) -> dict:
    """Paired binary comparison (memory-vs-control task success).

    Returns the discordant-pair point estimate, its §54.10-style
    interval, and the exact McNemar p-value — the three numbers a paired
    claim must publish together.
    """
    est, lo, hi = paired_delta_interval(n, wins_a, wins_b)
    return {
        "n": n,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "estimate": est,
        "ci95": [lo, hi],
        "mcnemar_p": mcnemar_exact(wins_a, wins_b),
    }


def bootstrap_ci(deltas: Sequence[float], *, seed: int = 7,
                 iters: int = 2000,
                 alpha: float = 0.05) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for a mean paired numeric difference.

    ``deltas`` is the per-unit (per task, per cluster) signed difference
    a−b. Resampling is unit-level; callers pass cluster-aggregated
    deltas when units are correlated (V5-23.02). Deterministic for a
    fixed seed — the seed is part of the published manifest.
    """
    vals = list(deltas)
    n = len(vals)
    if n == 0:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        means.append(
            sum(vals[rng.randrange(n)] for _ in range(n)) / n
        )
    means.sort()
    lo = means[max(0, int((alpha / 2) * iters) - 1)]
    hi = means[min(iters - 1, int((1 - alpha / 2) * iters))]
    return (sum(vals) / n, lo, hi)


def cluster_bootstrap_ci(clusters: Sequence[Sequence[float]], *,
                         seed: int = 7, iters: int = 2000,
                         alpha: float = 0.05) -> Tuple[float, float, float]:
    """Cluster-aware paired CI (V5-23.02): resample *clusters* — whole
    histories/families — never individual questions inside them.

    Each cluster contributes its mean delta; resampling draws clusters
    with replacement and pools their members. Independent units are the
    cluster count, not the pooled question count.
    """
    clusters = [list(c) for c in clusters if c]
    k = len(clusters)
    flat = [x for c in clusters for x in c]
    if k == 0 or not flat:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        picked = [clusters[rng.randrange(k)] for _ in range(k)]
        pool = [x for c in picked for x in c]
        means.append(sum(pool) / len(pool))
    means.sort()
    lo = means[max(0, int((alpha / 2) * iters) - 1)]
    hi = means[min(iters - 1, int((1 - alpha / 2) * iters))]
    return (sum(flat) / len(flat), lo, hi)


# ---------------------------------------------------------------------------
# support / power
# ---------------------------------------------------------------------------


def upper_bound_zero(n: int, z: float = Z_95) -> float:
    """Upper 95% bound on a failure rate when zero failures were seen
    in ``n`` independent trials (Wilson upper — rule-of-three scale).
    ``n=0`` returns 1.0: nothing observed bounds nothing."""
    if n <= 0:
        return 1.0
    return wilson_interval(0, n, z)[1]


def required_independent_units(p: float = 0.5, margin: float = 0.05,
                               z: float = Z_95) -> int:
    """Minimum independent units for a ±``margin`` proportion estimate
    at the 95% level — the power justification V5-23.03 requires before
    an ``n`` can be called sufficient. Worst-case ``p=0.5`` default.
    """
    if not (0.0 < p < 1.0) or margin <= 0:
        raise ValueError("p in (0,1) and margin > 0 required")
    return math.ceil(z * z * p * (1 - p) / (margin * margin))


def bonferroni_alpha(family_size: int, alpha: float = 0.05) -> float:
    """Frozen family-wise correction (V5-23.04): the per-comparison
    alpha a ``family_size``-wide claim family must clear."""
    if family_size < 1:
        raise ValueError("family_size must be >= 1")
    return alpha / family_size


def simultaneous_z(family_size: int, alpha: float = 0.05) -> float:
    """Two-sided z for Bonferroni-simultaneous intervals."""
    a = bonferroni_alpha(family_size, alpha)
    return _normal_quantile(1 - a / 2)


def _normal_quantile(p: float) -> float:
    """Acklam's rational approximation — no scipy needed."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q
                 + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q
                  + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r
             + a[4]) * r + a[5]) * q / (
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
          + b[4]) * r + 1))


# ---------------------------------------------------------------------------
# claim gates (V5-23.05/06/08/10/12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimVerdict:
    """The §23 verdict on one preregistered claim — with reasons."""

    claim: str
    verdict: str                      # VERDICTS member
    reasons: Tuple[str, ...] = ()
    value: Optional[float] = None
    ci: Optional[Tuple[float, float]] = None

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "value": self.value,
            "ci": list(self.ci) if self.ci else None,
        }


def check_seed_integrity(seeds_executed: Sequence[int],
                         seeds_reported: Sequence[int]) -> Tuple[bool, str]:
    """V5-23.10: every executed seed must appear in the report.

    Reporting a subset is only legal when the subset was fixed before
    observation — the caller declares ``seeds_reported`` as the
    preregistered set; any executed-but-unreported seed (a silent
    best-pick) fails integrity.
    """
    executed = list(seeds_executed)
    reported = list(seeds_reported)
    hidden = sorted(set(executed) - set(reported))
    if hidden:
        return False, (
            f"executed seeds {hidden} absent from the report — "
            "best-of-seeds selection is prohibited (V5-23.10)"
        )
    if not reported:
        return False, "no reported seeds — nothing executed"
    return True, "all executed seeds reported"


def evaluate_leadership_claim(
    name: str,
    estimate: float,
    ci_low: float,
    *,
    margin: float = 0.03,
    n_independent: int,
    required_n: Optional[int] = None,
    family_size: int = 1,
    simultaneous: bool = False,
) -> ClaimVerdict:
    """V5-23.05 quality-leadership gate.

    A leadership claim requires ≥ ``margin`` (default 3-point) primary
    gain with the *lower* confidence bound above zero — plus declared
    support and, for multi-endpoint families, simultaneous intervals.
    ``underpowered`` is returned below the required independent-unit
    count; a family>1 claim without simultaneous intervals is
    ``failed`` (the correction is not optional).
    """
    reasons: list[str] = []
    if n_independent <= 0:
        return ClaimVerdict(name, "unavailable",
                            ("no executed units",), estimate, None)
    if required_n is not None and n_independent < required_n:
        reasons.append(
            f"underpowered: {n_independent} independent units below "
            f"required {required_n} (V5-23.03)"
        )
        return ClaimVerdict(name, "underpowered", tuple(reasons),
                            estimate, (ci_low, estimate))
    if family_size > 1 and not simultaneous:
        return ClaimVerdict(
            name, "failed",
            (f"family of {family_size} endpoints claimed without "
             "simultaneous intervals (V5-23.04)",),
            estimate, (ci_low, estimate))
    if estimate < margin:
        reasons.append(
            f"estimate {estimate:.4f} below {margin:.2f} margin"
        )
    if ci_low <= 0.0:
        reasons.append(f"lower bound {ci_low:.4f} not above zero")
    if reasons:
        return ClaimVerdict(name, "failed", tuple(reasons),
                            estimate, (ci_low, estimate))
    return ClaimVerdict(name, "passed", (), estimate,
                        (ci_low, estimate))


def evaluate_zero_failures(
    name: str, failures: int, n: int,
    *,
    claimed_upper: Optional[float] = None,
) -> ClaimVerdict:
    """V5-23.08: zero observed failures is an *upper bound* statement,
    never a zero-rate statement."""
    if n <= 0:
        return ClaimVerdict(name, "unavailable",
                            ("no denominator",), None, None)
    if failures != 0:
        return ClaimVerdict(name, "inconclusive",
                            ("failures observed — not a zero-failure "
                             "claim",), failures / n, None)
    ub = upper_bound_zero(n)
    if claimed_upper is None:
        return ClaimVerdict(
            name, "failed",
            ("zero failures claimed without an upper bound — report "
             f"denominator {n} and upper95 {ub:.4f}",),
            0.0, (0.0, ub))
    if claimed_upper > ub + 1e-9:
        return ClaimVerdict(
            name, "failed",
            (f"claimed upper bound {claimed_upper:.4f} looser than "
             f"measured upper95 {ub:.4f}",),
            0.0, (0.0, ub))
    return ClaimVerdict(
        name, "passed",
        (f"0/{n} failures; upper95 {ub:.4f} (denominator published)",),
        0.0, (0.0, ub))


def evaluate_claim(record: dict) -> ClaimVerdict:
    """Dispatch one preregistered claim record through the §23 gates.

    ``record`` keys: ``name``, ``kind`` (leadership|zero_failures|
    frontier|point), ``estimate``, ``ci``, ``n_independent``,
    ``required_n``, ``family_size``, ``simultaneous``, ``failures``,
    ``claimed_upper``, ``seeds_executed``, ``seeds_reported``,
    ``noninferior``, ``cost_reduction``. Unknown kinds and missing
    measurements return ``inconclusive``/``unavailable`` — the gates
    fail closed.
    """
    name = str(record.get("name", "unnamed"))
    kind = record.get("kind", "point")

    ok, why = check_seed_integrity(
        record.get("seeds_executed") or record.get("seeds_reported") or [],
        record.get("seeds_reported") or [],
    )
    if not ok:
        return ClaimVerdict(name, "failed", (why,))

    if record.get("expired"):
        return ClaimVerdict(
            name, "failed",
            ("claim past its 90-day expiry (V5-23.12)",))

    if kind == "zero_failures":
        return evaluate_zero_failures(
            name,
            int(record.get("failures", -1)),
            int(record.get("n", 0)),
            claimed_upper=record.get("claimed_upper"),
        )
    if kind == "leadership":
        ci = record.get("ci") or [None, None]
        return evaluate_leadership_claim(
            name,
            float(record.get("estimate") or 0.0),
            float(ci[0] if ci[0] is not None else 0.0),
            margin=float(record.get("margin", 0.03)),
            n_independent=int(record.get("n_independent", 0)),
            required_n=record.get("required_n"),
            family_size=int(record.get("family_size", 1)),
            simultaneous=bool(record.get("simultaneous", False)),
        )
    if kind == "frontier":
        # V5-23.06: non-inferior within 1pt AND ≥20% lower cost — both
        # lower bounds must hold simultaneously.
        ni = record.get("noninferior")
        cost = record.get("cost_reduction")
        if ni is None or cost is None:
            return ClaimVerdict(name, "unavailable",
                                ("noninferior/cost_reduction unmeasured",))
        reasons = []
        if not ni.get("holds"):
            reasons.append("quality non-inferiority (≤1pt) did not hold")
        if float(cost) < 0.20:
            reasons.append(
                f"cost reduction {float(cost):.3f} below 0.20"
            )
        if record.get("safety_regressed"):
            reasons.append(
                "safety slice regressed — savings from dropped "
                "conditions fail the claim (V5-32.03)"
            )
        if reasons:
            return ClaimVerdict(name, "failed", tuple(reasons))
        return ClaimVerdict(name, "passed",
                            ("non-inferior at ≥20% lower cost",))
    # point claims: must carry ci + support
    ci = record.get("ci")
    if ci is None or record.get("estimate") is None:
        return ClaimVerdict(name, "inconclusive",
                            ("point claim without interval/estimate",))
    return ClaimVerdict(
        name, "passed" if not record.get("failed") else "failed",
        tuple(record.get("reasons") or ()),
        float(record["estimate"]), (float(ci[0]), float(ci[1])))


def claim_expired(measured_date: str, today: str,
                  ttl_days: int = CLAIM_TTL_DAYS) -> bool:
    """V5-23.12: a claim expires ``ttl_days`` after measurement
    (ISO dates, ``YYYY-MM-DD``)."""
    import datetime as _dt
    try:
        m = _dt.date.fromisoformat(measured_date[:10])
        t = _dt.date.fromisoformat(today[:10])
    except ValueError:
        return True  # unparseable dates are not a live claim
    return (t - m).days > ttl_days


__all__ = [
    "CLAIM_TTL_DAYS",
    "VERDICTS",
    "Z_95",
    "ClaimVerdict",
    "bonferroni_alpha",
    "bootstrap_ci",
    "check_seed_integrity",
    "claim_expired",
    "cluster_bootstrap_ci",
    "evaluate_claim",
    "evaluate_leadership_claim",
    "evaluate_zero_failures",
    "mcnemar_exact",
    "paired_binary_test",
    "paired_delta_interval",
    "required_independent_units",
    "simultaneous_z",
    "upper_bound_zero",
    "wilson_interval",
    "wilson_lower_bound",
    "wilson_upper_bound",
]
