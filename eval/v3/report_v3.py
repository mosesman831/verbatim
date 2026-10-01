"""Markdown report generator for v3 suite Reports (SPEC_V3 §53.01,
§54.02, §55.02, §63).

Renders the harness ``Report`` dict into the auditable §55.02 artifact:
suite identity, denominator, pass rate with its Wilson interval, the
inconclusive list, and competitor-claim readiness. The readiness line is
the §54.13 honesty rule made visible: it is ``not_estimable`` unless the
paired-usefulness suite actually executed its required arms — no
shadow-run, source-count, or spec-count evidence can substitute.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from .suites import (
    PAIRED_REQUIRED_ARMS,
    PAIRED_USEFULNESS,
)

#: Ordered display names for the four suites.
SUITE_TITLES = {
    "conformance": "Conformance",
    "retrieval_quality": "Retrieval quality",
    "paired_usefulness": "Paired usefulness",
    "security_privacy": "Security & privacy",
}

#: What each readiness value means — kept in the report so a reader
#: cannot mistake a pointer for a result.
READINESS_MEANINGS = {
    "estimable": "required paired arms executed; a comparison may be computed from measured results",
    "not_estimable": "paired arms did not execute; no competitive/comparative claim is supported",
}


def competitor_claim_readiness(report: dict) -> str:
    """§54.13 gate on comparative claims.

    ``estimable`` only when this is the paired-usefulness report AND at
    least one case result carries executed metrics for every required
    arm (``no_memory`` floor + ``memory`` under test). Any other suite,
    or a paired run whose arms never executed, is ``not_estimable`` —
    shadow logs and spec targets are not comparisons.
    """
    if report.get("suite_id") != PAIRED_USEFULNESS:
        return "not_estimable"
    for r in report.get("results", []):
        arms = (r.get("metrics") or {}).get("arms") or {}
        if all(a in arms for a in PAIRED_REQUIRED_ARMS):
            ran = all(
                isinstance(arms[a], dict)
                and arms[a].get("executed")
                for a in PAIRED_REQUIRED_ARMS
            )
            if ran:
                return "estimable"
    return "not_estimable"


def _fmt_rate(rate: Optional[float]) -> str:
    return "n/a (empty denominator)" if rate is None else f"{rate:.4f}"


def _fmt_ci(ci: Optional[Sequence[float]]) -> str:
    if not ci:
        return "n/a"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def render_markdown(
    reports: Sequence[dict],
    *,
    title: str = "Verbatim v3 evaluation report",
    spec_ref: str = "SPEC_V3.md R1",
    ledger_coverage: Optional[dict] = None,
) -> str:
    """Render one or more suite Reports into the §55.02 Markdown artifact.

    ``ledger_coverage`` optionally carries ``Ledger.coverage_report()``
    output so the requirement-evidence matrix appears beside the suite
    results — the §56.10 pairing the release checklist consumes.
    """
    lines: list[str] = [
        f"# {title}",
        "",
        f"Spec: {spec_ref}. All numbers are measured results from this "
        "run; specification targets are never reported as results "
        "(V3-54.11).",
        "",
        "## Suites",
        "",
        "| suite | verdict | denominator (executed/declared) | pass rate | 95% CI |",
        "|---|---|---|---|---|",
    ]
    inconclusive_all: list[tuple[str, str]] = []
    readiness = "not_estimable"
    for rep in reports:
        sid = rep.get("suite_id", "?")
        m = rep.get("metrics", {})
        counts = m.get("counts", {})
        denom = f"{counts.get('executed', 0)}/{rep.get('declared_cases', 0)}"
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                SUITE_TITLES.get(sid, sid),
                rep.get("verdict", "?"),
                denom,
                _fmt_rate(m.get("pass_rate")),
                _fmt_ci(m.get("ci95")),
            )
        )
        for cid in rep.get("inconclusive", []):
            inconclusive_all.append((sid, cid))
        if competitor_claim_readiness(rep) == "estimable":
            readiness = "estimable"

    lines += [
        "",
        "## Per-suite detail",
        "",
    ]
    for rep in reports:
        sid = rep.get("suite_id", "?")
        m = rep.get("metrics", {})
        counts = m.get("counts", {})
        lines += [
            f"### {SUITE_TITLES.get(sid, sid)} (`{sid}`)",
            "",
            f"- Verdict: **{rep.get('verdict', '?')}**",
            f"- Denominator: {counts.get('executed', 0)} executed of "
            f"{rep.get('declared_cases', 0)} declared cases "
            f"({rep.get('selected_cases', '?')} selected)",
            f"- Outcomes: pass={counts.get('pass', 0)}, "
            f"fail={counts.get('fail', 0)}, "
            f"inconclusive={counts.get('inconclusive', 0)}, "
            f"error={counts.get('error', 0)}, "
            f"skipped={counts.get('skipped', 0)}",
            f"- Pass rate: {_fmt_rate(m.get('pass_rate'))} "
            f"CI95 {_fmt_ci(m.get('ci95'))}",
            f"- Manifest digest: `{rep.get('manifest_digest') or 'not frozen'}`",
            f"- Feeds gates: {', '.join(rep.get('gates', [])) or '—'}",
        ]
        if rep.get("prepare_error"):
            lines.append(f"- Prepare error: `{rep['prepare_error']}`")
        suite_metrics = m.get("suite_metrics")
        if suite_metrics:
            lines.append("- Suite metrics:")
            for k, v in suite_metrics.items():
                lines.append(f"  - `{k}`: {v}")
        lines.append("")

    lines += ["## Inconclusive and skipped cases", ""]
    if inconclusive_all:
        for sid, cid in inconclusive_all:
            lines.append(f"- `{sid}` / `{cid}` — inconclusive or "
                         "skipped: missing support is reported, never "
                         "passed (V3-54.11).")
    else:
        lines.append("- none")
    lines += [
        "",
        "## Competitor-claim readiness",
        "",
        f"**{readiness}** — {READINESS_MEANINGS[readiness]}.",
        "",
        "Shadow logs, source counts, and specification targets do not "
        "satisfy this gate; only executed paired arms do (V3-54.13).",
        "",
    ]
    if ledger_coverage is not None:
        cov = ledger_coverage
        lines += [
            "## Requirement evidence coverage (§56.10)",
            "",
            f"- Requirements tracked: {cov.get('total_requirements', 0)}",
            f"- Complete (every required evidence kind passing): "
            f"{cov.get('complete', 0)}",
            f"- Verdicts: "
            + ", ".join(
                f"{k}={v}"
                for k, v in sorted(cov.get("verdicts", {}).items())
            ),
            "",
            "| evidence kind | required refs | registered | requirements passing |",
            "|---|---|---|---|",
        ]
        for k, v in cov.get("evidence_kinds", {}).items():
            lines.append(
                f"| {k} | {v.get('required_refs', 0)} | "
                f"{v.get('registered', 0)} | "
                f"{v.get('passing_requirements', 0)} |"
            )
        gates = cov.get("gates") or {}
        if gates:
            lines += [
                "",
                "| gate | status |",
                "|---|---|",
            ]
            for g, s in gates.items():
                lines.append(f"| {g} | {s} |")
        lines.append("")
    return "\n".join(lines)


def render_f27_markdown(report: dict) -> str:
    """Render the F27 investigation report — the B49 evidence section."""
    from .f27 import summarize_attribution

    recall = report.get("recall", {})
    pre = recall.get("pre_rebuild", {})
    post = recall.get("post_rebuild", {})
    saved = report.get("saved_baseline", {})
    attr = summarize_attribution(report)
    lines = [
        "## F27 — historical retrieval investigation",
        "",
        f"Saved baseline (reproduction target): "
        f"{saved.get('historical_pre_rebuild', {}).get('hits')}/"
        f"{saved.get('historical_pre_rebuild', {}).get('n')} = "
        f"{saved.get('historical_pre_rebuild', {}).get('hit_rate')} "
        "pre-rebuild; "
        f"{saved.get('historical_post_rebuild', {}).get('hits')}/"
        f"{saved.get('historical_post_rebuild', {}).get('n')} = "
        f"{saved.get('historical_post_rebuild', {}).get('hit_rate')} "
        "post-rebuild (eval/report.md).",
        "",
        "This run:",
        "",
        "| phase | hits | n | hit rate | errors | gen | fts rows @ gen |",
        "|---|---|---|---|---|---|---|",
    ]
    phases = report.get("phases", {})
    for name, agg in (("pre_rebuild", pre), ("post_rebuild", post)):
        ph = phases.get(name, {})
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                name,
                agg.get("hits", 0),
                agg.get("n", 0),
                _fmt_rate(agg.get("hit_rate")),
                agg.get("errors", 0),
                ph.get("projection_generation", "?"),
                ph.get("fts_rows_at_generation", "?"),
            )
        )
    lines += [
        "",
        f"Delta: {recall.get('delta_hits', 0)} hit(s). "
        f"Outcome counts: {report.get('outcome_counts', {})}.",
        "",
        "### Attribution by mechanism",
        "",
    ]
    for k, v in attr["mechanisms"].items():
        lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        f"Unattributed queries: {attr['unattributed']} — F27/B49 stays "
        "open until every delta is explained.",
        "",
    ]
    return "\n".join(lines)


def write_report(
    reports: Sequence[dict],
    path: str,
    **kwargs: Any,
) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_markdown(reports, **kwargs))
    return path
