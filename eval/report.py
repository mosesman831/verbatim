"""Render a HarnessResult as Markdown + JSON.

Report principles (SPEC_V2 measurement posture):

* Measurements are labeled as measurements.  Spec *targets* are labeled as
  targets and never blended into the measured numbers.
* Capability gaps and degradation are first-class sections, not footnotes.
* No superiority claims: nothing here compares Verbatim to any other
  system, because no comparable external evaluation was run.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .harness import HarnessResult


def render_markdown(res: HarnessResult) -> str:
    c = res.corpus
    ing = res.ingest
    proc = res.processing
    cov = res.coverage
    rec = res.recall
    fts = res.fts

    def _row(label: str, d: dict[str, Any] | None, *keys: str) -> str:
        if not d:
            return f"| {label} | — | — | — |"
        return (
            f"| {label} | {d.get('n', '—')} | {d.get('hits', '—')} | "
            f"{d.get('hit_rate', '—')} | {d.get('items_returned', '—')} |"
        )

    lines: list[str] = []
    lines.append("# Verbatim v2 — synthetic-corpus baseline report")
    lines.append("")
    lines.append(
        "This report is **measured** output from `eval/harness.py` driving the "
        "public `verbatim.api.Engine` on a fully synthetic, locally generated "
        "corpus (no external benchmark data). Numbers below are observations "
        "of this configuration, not specification targets."
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("| setting | value |")
    lines.append("|---|---|")
    for k, v in sorted(res.config.items()):
        lines.append(f"| {k} | `{v}` |")
    lines.append(f"| corpus_sha256 | `{c.get('sha256', '')[:16]}…` |")
    lines.append(f"| corpus_seed | `{c.get('seed')}` |")
    lines.append(f"| elapsed_s | {res.elapsed_s} |")
    lines.append("")

    lines.append("## Corpus")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| statements | {c.get('statements')} |")
    lines.append(f"| queries | {c.get('queries')} |")
    lines.append(f"| update pairs | {c.get('update_pairs')} |")
    lines.append(f"| contradiction pairs | {c.get('contradiction_pairs')} |")
    lines.append("")

    lines.append("## Ingestion and processing")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| envelopes accepted | {ing.get('accepted')} / {ing.get('envelopes')} |")
    lines.append(f"| duplicate receipts | {ing.get('duplicates')} |")
    lines.append(f"| claims created | {proc.get('claims_created')} |")
    lines.append(f"| structured claims (predicate set) | {proc.get('structured_claims')} |")
    lines.append(f"| pending after processing | {proc.get('pending')} |")
    lines.append(f"| active (after operator approvals) | {proc.get('active_final')} |")
    lines.append(f"| superseded | {proc.get('superseded_final')} |")
    lines.append(f"| admit failures | {proc.get('admit_failures')} |")
    lines.append(f"| supersede failures | {proc.get('supersede_failures')} |")
    js = proc.get("job_states") or {}
    lines.append(f"| job states | {json.dumps(js)} |")
    lines.append("")

    lines.append("## Extraction coverage")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(
        f"| statements producing ≥1 claim | {cov.get('statements_with_claims')} |"
    )
    lines.append(f"| extraction coverage | {cov.get('extraction_coverage')} |")
    lines.append(f"| claims per statement | {cov.get('claim_yield')} |")
    lines.append("")

    lines.append("## Recall (post-rebuild where applicable)")
    lines.append("")
    lines.append("| query kind | n | hits | hit_rate | items_returned | notes |")
    lines.append("|---|---|---|---|---|---|")
    for kind in ("point", "current", "historical", "no_answer"):
        d = rec.get(kind)
        line = _row(kind, d)
        note = ""
        if kind == "no_answer" and d:
            note = f"false_positive_rate={d.get('false_positive_rate')}"
        lines.append(f"{line} {note} |")
    lat = rec.get("latency_ms") or {}
    lines.append("")
    lines.append(
        f"**Latency (ms):** p50={lat.get('p50')} p95={lat.get('p95')} "
        f"p99={lat.get('p99')} max={lat.get('max')} (n={lat.get('n')})"
    )
    lines.append("")

    pre = rec.get("pre_rebuild") or {}
    if pre:
        lines.append("### Pre-rebuild recall (generation-regression check)")
        lines.append("")
        lines.append("| query kind | n | hits | hit_rate | items_returned |")
        lines.append("|---|---|---|---|---|")
        for kind in ("point", "current", "historical", "no_answer"):
            lines.append(_row(kind, pre.get(kind)))
        lines.append("")

    lines.append("## FTS projection state")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| projection generation | {fts.get('projection_generation')} |")
    lines.append(
        f"| claims indexed at current generation | "
        f"{fts.get('claims_indexed_at_current_generation')} |"
    )
    lines.append(f"| rebuild used | {fts.get('rebuild_used')} |")
    if fts.get("reindexed_claims") is not None:
        lines.append(f"| claims re-indexed by rebuild | {fts.get('reindexed_claims')} |")
    lines.append("")

    lines.append("## Store")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| db size (bytes) | {res.store.get('size_bytes')} |")
    lines.append("")

    if res.defects:
        lines.append("## Observed capability gaps / defects")
        lines.append("")
        for d in res.defects:
            lines.append(f"- {d}")
        lines.append("")
    if res.notes:
        lines.append("## Notes")
        lines.append("")
        for n in res.notes:
            lines.append(f"- {n}")
        lines.append("")

    lines.append("## Specification targets vs measurement")
    lines.append("")
    lines.append(
        "The spec defines *targets*; this report records *measurements*. "
        "The two are shown side by side — agreement is a coincidence of "
        "this run, not a guarantee."
    )
    lines.append("")
    lines.append("| quantity | spec target | measured |")
    lines.append("|---|---|---|")
    lines.append(f"| corpus statements | ~1000 | {c.get('statements')} |")
    lines.append(f"| update pairs | ~300 | {c.get('update_pairs')} |")
    lines.append(f"| no-answer probes | ~200 | {(rec.get('no_answer') or {}).get('n')} |")
    lines.append(
        "| unstructured evidence preserved | preserved (A01) | "
        f"extraction coverage {cov.get('extraction_coverage')} |"
    )
    lines.append(
        "| temporal update semantics | current→new / historical→old (A07) | "
        f"current hit_rate {(rec.get('current') or {}).get('hit_rate')}, "
        f"historical hit_rate {(rec.get('historical') or {}).get('hit_rate')} "
        "— measured against gold source links |"
    )
    lines.append(
        "| abstention on unanswerable probes | no fabricated evidence | "
        f"false_positive_rate "
        f"{(rec.get('no_answer') or {}).get('false_positive_rate')} |"
    )
    lines.append("")
    lines.append(
        "Scenario-level acceptance status lives in "
        "`tests/test_v2_acceptance.py` (A01–A25); each test asserts the spec "
        "behavior and marks genuinely missing capabilities with strict xfail."
    )
    lines.append("")

    lines.append("## Honesty footer")
    lines.append("")
    lines.append(
        "- These numbers describe the `offline_rules` configuration on a "
        "synthetic corpus only. They do not measure semantic-encoder "
        "configurations, real conversational data, or any other system."
    )
    lines.append(
        "- The operator-assisted admission path (`apply_transition`) was used "
        "for approvals because the default configuration requires review; "
        "unassisted auto-admission is not measured here."
    )
    lines.append(
        "- Failed or errored queries remain in every denominator; nothing is "
        "dropped for looking bad."
    )
    lines.append("")
    return "\n".join(lines)


def write_report(res: HarnessResult, out_dir: str) -> tuple[str, str]:
    """Write report.md + report.json into ``out_dir``; return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "report.md")
    json_path = os.path.join(out_dir, "report.json")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(res))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res.to_dict(), f, indent=2, sort_keys=True, default=str)
    return md_path, json_path
