"""The v3 suite runner (SPEC_V3 §53.05, §53.10, §54.02, §54.11).

``run_suite`` executes a :class:`~eval.v3.suites.Suite` against a
``store_factory`` callable — the same ``() -> Store`` seam the v2 harness
and the test fixtures use — and produces the ``Report`` dict. The report
is the auditable artifact (§55.02): machine-checkable, with every
denominator, every outcome, and the uncertainty intervals; it never
converts a spec target or a shadow log into a result (§54.11, §54.13).

Denominator and verdict rules implemented here:

* A case that raises is ``error`` — it stays in the denominator and
  counts toward ``fail`` at suite level (§53.05).
* A runner that returns ``pass`` with ``support < case.min_support`` is
  downgraded to ``inconclusive`` — missing statistical support can never
  be a pass (§54.11). This applies to binary *case* support; suite-level
  empirical support is additionally enforced by
  ``Suite.min_support``.
* A case that is never executed is ``skipped`` — reported, never folded
  into the pass rate.
* The suite verdict follows :func:`eval.v3.suites.suite_verdict`:
  any required failure → ``fail``; any required gap or insufficient
  support → ``inconclusive``; else ``pass``.
* ``report["gates"]`` exposes which §54 gates this suite's outcomes could
  feed (a pointer, not a claim — the release manifest records actual
  gate runs).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Sequence

from .suites import (
    EXECUTED_OUTCOMES,
    Case,
    CaseResult,
    Outcome,
    Suite,
    base_metrics,
    suite_verdict,
)

#: Which §54 automation gates each suite feeds (pointers for §63's
#: release checklist — the manifest decides gate status, not the suite).
SUITE_GATES = {
    "conformance": ("G1", "G2"),
    "retrieval_quality": ("G3", "G5"),
    "paired_usefulness": ("G7", "G8"),
    "security_privacy": ("G9",),
}


def _coerce_result(case: Case, raw: Any) -> CaseResult:
    """Normalize a runner's return into a :class:`CaseResult`.

    ``bool`` and ``Outcome`` returns are upgraded so simple conformance
    runners can stay terse; everything else must already be a
    ``CaseResult``.
    """
    if isinstance(raw, CaseResult):
        return raw
    if isinstance(raw, Outcome):
        return CaseResult(case.case_id, raw)
    if isinstance(raw, bool):
        return CaseResult(
            case.case_id, Outcome.PASS if raw else Outcome.FAIL
        )
    raise TypeError(
        f"runner for {case.case_id} returned {type(raw).__name__}, "
        "expected CaseResult/Outcome/bool"
    )


def _apply_support_rule(case: Case, result: CaseResult) -> CaseResult:
    """§54.11: a `pass` without the declared minimum support becomes
    ``inconclusive`` — measured honesty, not a silent upgrade."""
    if result.outcome is Outcome.PASS and result.support < case.min_support:
        return CaseResult(
            result.case_id,
            Outcome.INCONCLUSIVE,
            support=result.support,
            detail=(
                f"insufficient support: {result.support} executed sample(s) "
                f"below min_support {case.min_support} — inconclusive, "
                "not pass (§54.11)"
            ),
            metrics=result.metrics,
            latency_ns=result.latency_ns,
        )
    return result


def run_case(case: Case, runner: Callable, ctx: Any) -> CaseResult:
    """Execute one case; exceptions become ``error`` results that stay in
    the denominator (§53.05)."""
    t0 = time.perf_counter_ns()
    try:
        raw = runner(case, ctx)
        result = _coerce_result(case, raw)
    except Exception as exc:  # noqa: BLE001 — any failure is recorded, never swallowed
        result = CaseResult(
            case.case_id,
            Outcome.ERROR,
            detail=f"{type(exc).__name__}: {exc}",
        )
    result = _apply_support_rule(case, result)
    if result.latency_ns == 0:
        result = CaseResult(
            result.case_id,
            result.outcome,
            support=result.support,
            detail=result.detail,
            metrics=result.metrics,
            latency_ns=time.perf_counter_ns() - t0,
        )
    return result


def run_suite(
    suite: Suite,
    store_factory: Callable[[], Any],
    *,
    run_id: str = "",
    cases: Optional[Sequence[str]] = None,
) -> dict:
    """Run ``suite`` against ``store_factory`` and return the Report dict.

    ``store_factory`` must be a callable producing a fresh store (or
    whatever ``Suite.prepare`` builds from it). The same callable is
    handed to every arm/phase so paired runs get identical starting
    snapshots (§53.14).

    ``cases`` optionally filters to a subset of case_ids — a filtered
    run is reported honestly as covering only those cases (the report
    lists ``declared`` vs ``selected``).

    The returned dict is JSON-serializable and shaped for
    ``report_v3.render_markdown`` and the release manifest's
    ``evidence`` pointers.
    """
    selected: list[Case] = list(suite.cases)
    if cases is not None:
        wanted = set(cases)
        selected = [c for c in selected if c.case_id in wanted]

    # Prepare context once (§53.10: suite runner calls the case runner
    # over the prepared context; a suite with no prepare hook receives
    # one fresh store directly).
    ctx = None
    prepare_error = ""
    t_prep = time.perf_counter_ns()
    try:
        if suite.prepare is not None:
            ctx = suite.prepare(store_factory)
        else:
            ctx = store_factory()
    except Exception as exc:  # noqa: BLE001
        prepare_error = f"{type(exc).__name__}: {exc}"
    prepare_ns = time.perf_counter_ns() - t_prep

    results: list[CaseResult] = []
    if prepare_error:
        # Suite could not start: every selected case is an executed
        # error — the denominator is preserved and the suite fails
        # honestly rather than reporting zero cases.
        for c in selected:
            results.append(
                CaseResult(
                    c.case_id,
                    Outcome.ERROR,
                    detail=f"suite prepare failed: {prepare_error}",
                )
            )
    else:
        for c in selected:
            results.append(run_case(c, suite.runner, ctx))

    metrics = base_metrics(results)
    if suite.aggregate is not None and not prepare_error:
        try:
            metrics["suite_metrics"] = suite.aggregate(results)
        except Exception as exc:  # noqa: BLE001
            metrics["suite_metrics"] = {
                "aggregate_error": f"{type(exc).__name__}: {exc}"
            }

    return {
        "schema": 1,
        "suite_id": suite.suite_id,
        "suite_name": suite.name,
        "run_id": run_id,
        "requirement_ids": list(suite.requirement_ids),
        "manifest_digest": suite.manifest_digest,
        "declared_cases": len(suite.cases),
        "selected_cases": len(selected),
        "results": [r.to_dict() for r in results],
        "metrics": metrics,
        "verdict": suite_verdict(suite, results)
        if not prepare_error
        else "fail",
        "prepare_ns": prepare_ns,
        "prepare_error": prepare_error,
        "gates": list(SUITE_GATES.get(suite.suite_id, ())),
        "inconclusive": [
            r.case_id for r in results
            if r.outcome in (Outcome.INCONCLUSIVE, Outcome.SKIPPED)
        ],
    }


def report_pass_rate(report: dict) -> Optional[float]:
    """Convenience accessor — ``None`` when the denominator was zero."""
    return report.get("metrics", {}).get("pass_rate")


def is_valid_report(report: dict) -> bool:
    """Structural sanity check used by tests and the report generator."""
    required = (
        "suite_id", "results", "metrics", "verdict",
        "declared_cases", "selected_cases",
    )
    return isinstance(report, dict) and all(k in report for k in required)
