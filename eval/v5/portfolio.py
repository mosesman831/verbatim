"""The V5 eval portfolio — one entry point that runs every suite and
assembles the combined results record (§24 qualification surface).

Each suite runs in a guarded call: a suite that raises records
``status=failed`` with the exception — the portfolio never crashes out
of an honest half-empty report, and a failed suite is visible as failed,
not absent.

``quick`` scale is the CI-shaped run; ``full`` raises envelope/probe
sizes toward spec scale (still disclosed as locally measured unless the
§20.02 qualification floors are actually met).
"""

from __future__ import annotations

import traceback
from typing import Any, Callable, Dict, Optional

from .harness import environment


def _guard(name: str, fn: Callable[..., Any], **kw: Any) -> dict:
    try:
        out = fn(**kw)
        if hasattr(out, "to_dict"):
            out = out.to_dict()
        elif not isinstance(out, dict):
            out = {"value": out}
        out.setdefault("suite", name)
        out.setdefault("status", "executed")
        return out
    except Exception as exc:  # noqa: BLE001 — honest failure record
        return {
            "suite": name,
            "status": "failed",
            "verdict": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
        }


def run_portfolio(*, quick: bool = True, seed: int = 42,
                  suites: Optional[list] = None,
                  workdir: Optional[str] = None) -> Dict[str, Any]:
    """Run the V5 suite set and return the combined record."""
    from . import (
        comparators,
        consolidation,
        dx,
        envelopes,
        feedback,
        frontier,
        leakage,
        perf,
        quality,
        timers,
    )

    scale = {
        "quality_memories": 64,
        "a0_memories": 256, "a0_queries": 60, "a0_reps": 1,
        "a1_memories": 256, "a1_reads": 40, "a1_writes": 16,
        "frontier_memories": 96,
        "a3_scales": (192, 768), "a3_queries": 40,
        "timers_memories": 96, "timers_queries": 24,
        "comparators_memories": 48,
        "leakage_memories": 48,
        "consolidation_memories": 40,
        "feedback_memories": 40,
    } if quick else {
        "quality_memories": 256,
        "a0_memories": 1000, "a0_queries": 400, "a0_reps": 3,
        "a1_memories": 1500, "a1_reads": 300, "a1_writes": 120,
        "frontier_memories": 256,
        "a3_scales": (512, 2048), "a3_queries": 120,
        "timers_memories": 256, "timers_queries": 60,
        "comparators_memories": 128,
        "leakage_memories": 96,
        "consolidation_memories": 96,
        "feedback_memories": 96,
    }

    want = set(suites or [
        "quality", "envelopes", "frontier", "a3", "timers",
        "comparators", "leakage", "dx", "consolidation", "feedback",
    ])
    out: Dict[str, Any] = {
        "portfolio": "v5",
        "quick": quick,
        "seed": seed,
        "qualification": "locally_measured",
        "environment": environment(),
        "suites": {},
        "unmeasured": [
            "operator_time cost category (no human-time meter in run)",
            "hosted/priced comparator costs (no paid benchmarks run)",
            "deployment cost beyond local embedded store",
        ],
    }
    s = out["suites"]

    if "quality" in want:
        s["quality"] = _guard(
            "quality", quality.run_quality_suite,
            memories=scale["quality_memories"], seed=seed,
            workdir=workdir)
    if "envelopes" in want:
        a0 = _guard(
            "a0", envelopes.measure_a0,
            memories=scale["a0_memories"], seed=seed,
            queries=scale["a0_queries"],
            repetitions=scale["a0_reps"], workdir=workdir)
        a1 = _guard(
            "a1", envelopes.measure_a1,
            memories=scale["a1_memories"], seed=seed + 1,
            reader_queries=scale["a1_reads"],
            writer_ops=scale["a1_writes"], workdir=workdir)
        s["envelopes"] = {
            "suite": "envelopes",
            "qualification": "locally_measured",
            "a0": a0,
            "a1": a1,
            "spec_budgets": {"A0": dict(envelopes.A0),
                             "A1": dict(envelopes.A1)},
            "qualification_floor": dict(envelopes.QUALIFICATION),
            "verdict": (
                "failed" if (
                    a0.get("misses") or a1.get("misses") or
                    a0.get("status") == "failed" or
                    a1.get("status") == "failed")
                else "passed"),
        }
    if "frontier" in want:
        s["frontier"] = _guard(
            "frontier", frontier.run_frontier,
            memories=scale["frontier_memories"], seed=seed,
            workdir=workdir)
    if "a3" in want:
        s["a3_probe"] = _guard(
            "a3_probe", perf.run_a3_probe,
            scales=scale["a3_scales"], seed=seed,
            queries=scale["a3_queries"], workdir=workdir)
    if "timers" in want:
        s["timers"] = _guard(
            "timers", timers.run_timers,
            memories=scale["timers_memories"], seed=seed,
            queries=scale["timers_queries"], workdir=workdir)
    if "comparators" in want:
        s["comparators"] = _guard(
            "comparators", comparators.run_comparator_registry,
            memories=scale["comparators_memories"], seed=seed,
            workdir=workdir)
    if "leakage" in want:
        s["leakage"] = _guard(
            "leakage", leakage.run_leakage,
            memories=scale["leakage_memories"], seed=seed,
            workdir=workdir)
    if "dx" in want:
        s["dx"] = _guard("dx", dx.run_dx_probe, workdir=workdir)
    if "consolidation" in want:
        s["consolidation"] = _guard(
            "consolidation", consolidation.run_consolidation,
            memories=scale["consolidation_memories"], seed=seed,
            workdir=workdir)
    if "feedback" in want:
        s["feedback"] = _guard(
            "feedback", feedback.run_feedback,
            memories=scale["feedback_memories"], seed=seed,
            workdir=workdir)

    # Portfolio-level claim roll-up: every suite's claim verdicts.
    claims: list = []
    for name, suite in s.items():
        for c in (suite.get("claims") or []):
            claims.append(c)
        fc = suite.get("frontier_claim")
        if fc:
            claims.append({"claim": f"{name}:frontier",
                           "verdict": fc.get("verdict"),
                           "reasons": fc.get("reasons") or []})
    out["claims"] = claims

    # Honest roll-up: any suite failure or explicit loss marks the run.
    suite_states = [
        (s[n].get("verdict") or s[n].get("status"))
        for n in s
    ]
    out["verdict"] = (
        "failed" if any(v == "failed" for v in suite_states)
        else "passed" if all(v in ("passed", "executed")
                             for v in suite_states)
        else "inconclusive"
    )
    return out


__all__ = ["run_portfolio"]
