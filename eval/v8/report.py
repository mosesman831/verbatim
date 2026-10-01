"""V8 report renderer (V8-15.04, V8-03.01, V8-15.05, V8-15.17).

Renders ``report_v8.md``-shaped markdown from the ledger and the run
manifests.  Three honesty rules are structural, not cosmetic:

* **``not_run`` is verbatim** (K19 / V8-26.02 carried): a gate, an
  input, or a metric cell whose status is ``not_run`` renders the
  string ``not_run`` — never a number, never an omission.
* **No unpinned numbers** (K03 / V7-22.24 carried): a scoreboard value
  without a manifest digest renders ``unpinned``; the report refuses
  to print the figure.
* **Missing rows are ``not_run``, not absent** (V8-03.01): every
  declared SB8 id renders a row; one with no artifact reads
  ``not_run``.

The renderer is deterministic — ``header`` fields (date, git rev,
dirty flag, machine, load, manifest digests per V8-15.17) are supplied
by the caller, never read off the clock inside.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from . import ledger as L

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

#: The declared SB8 registry (§03.2) — a row missing from the input
#: still renders, as ``not_run`` (V8-03.01).
SCOREBOARD_IDS = tuple(f"SB8-{i:02d}" for i in range(1, 17))

#: Row statuses (V8-03.01 / §22.4 vocabulary).
SCOREBOARD_STATUSES = (
    "passed",
    "missed",
    "not_run",
    "blocked_on_authorization",
    "invalid",
)

#: The absence floor — rendered verbatim wherever a metric would go.
NOT_RUN = "not_run"

#: A value with no manifest digest is refused this label (K03).
UNPINNED = "unpinned"


# ---------------------------------------------------------------------------
# cell helpers
# ---------------------------------------------------------------------------


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _row(cells: Sequence[Any]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def metric_cell(value: Any, *, status: Optional[str] = None,
                manifest_digest: Optional[str] = None) -> str:
    """One metric cell under the V8 honesty rules.

    ``status == "not_run"`` (or a value that *is* the string) renders
    ``not_run`` verbatim; a numeric value without a manifest digest is
    refused behind ``unpinned`` (K03); otherwise the formatted value.
    """
    if status == NOT_RUN or value == NOT_RUN:
        return NOT_RUN
    if value is None:
        return "—"
    if not manifest_digest:
        return UNPINNED
    return _f(value)


def _short(x: Any, n: int = 10) -> str:
    s = str(x or "")
    return s[:n] if s else "—"


# ---------------------------------------------------------------------------
# ledger table (V8-15.04 — "report_v8.md renders the ledger as a table")
# ---------------------------------------------------------------------------


def render_ledger_table(records: Sequence[Mapping[str, Any]]) -> str:
    """One row per ledger record: requirement, commits, digests, the
    Δany@10 axis, the keep decision (+ rejecting metric when flagged
    off), and the decision-record path."""
    lines = [
        "### Requirement ledger (V8-15.04)",
        "",
        _row(["requirement", "defects", "base→head", "manifests b/c",
              "Δany@10", "decision", "rejecting metric",
              "decision record"]),
        _row(["---"] * 8),
    ]
    if not records:
        lines.append(_row(["—"] * 8))
    for rec in records:
        d = L.delta_any10(rec)
        delta = "not_run" if d is None else f"{d:+.4f}"
        lines.append(_row([
            ",".join(rec.get("requirement_ids") or []) or "—",
            ",".join(rec.get("defect_ids") or []) or "—",
            f"{_short(rec.get('base_commit'), 8)}→"
            f"{_short(rec.get('head_commit'), 8)}",
            f"{_short(rec.get('base_manifest_digest'))}/"
            f"{_short(rec.get('candidate_manifest_digest'))}",
            delta,
            rec.get("decision") or "—",
            rec.get("rejecting_metric") or "—",
            rec.get("decision_record") or "—",
        ]))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# scoreboard (V8-03.01)
# ---------------------------------------------------------------------------


def render_scoreboard(
    rows: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> str:
    """The §03.2 table.  Every declared SB8 id renders; a missing row
    is ``not_run``; a digest-less value is ``unpinned``."""
    rows = rows or {}
    lines = [
        "### Scoreboard (§03.2)",
        "",
        _row(["row", "status", "value", "target", "reference",
              "protocol", "manifest"]),
        _row(["---"] * 7),
    ]
    unpinned: List[str] = []
    for sb in SCOREBOARD_IDS:
        r = rows.get(sb)
        if not isinstance(r, Mapping):
            lines.append(_row([sb, NOT_RUN, NOT_RUN, "—", "—", "—", "—"]))
            continue
        status = r.get("status")
        if status not in SCOREBOARD_STATUSES:
            status = NOT_RUN
        digest = r.get("manifest_digest")
        value = metric_cell(
            r.get("value"), status=status, manifest_digest=digest)
        if value == UNPINNED:
            unpinned.append(sb)
        lines.append(_row([
            sb, status, value,
            r.get("target") or "—",
            r.get("reference") or "—",
            r.get("protocol") or "—",
            _short(digest) if digest else "—",
        ]))
    lines.append("")
    if unpinned:
        lines.append(
            "> unpinned rows (value withheld — no manifest digest, "
            "K03): " + ", ".join(unpinned))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# gates (K19 / V8-17.01 — input statuses render verbatim)
# ---------------------------------------------------------------------------


def render_gates_table(
    gates: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> str:
    """The G8 gate table.  An input (or gate) at ``not_run`` renders
    ``not_run`` — the report never substitutes a metric for an
    unexecuted input."""
    lines = [
        "### Gates (§17)",
        "",
        _row(["gate", "name", "status", "inputs ready"]),
        _row(["---"] * 4),
    ]
    counts: Dict[str, int] = {}
    for gid, g in sorted((gates or {}).items()):
        status = g.get("status") or NOT_RUN
        counts[status] = counts.get(status, 0) + 1
        lines.append(_row([
            gid, g.get("name") or "—", status,
            f"{g.get('inputs_ready', 0)}/{g.get('inputs_total', 0)}",
        ]))
        for inp in g.get("inputs") or ():
            st = inp.get("status") or NOT_RUN
            # the input's own metric, if any, flows through the same
            # honesty cell — not_run inputs render not_run verbatim
            val = metric_cell(
                inp.get("value"), status=st,
                manifest_digest=inp.get("manifest_digest"))
            lines.append(_row([
                "", f"· {inp.get('artifact', '—')}", st, val]))
    lines.append("")
    if counts:
        lines.append("statuses: " + " ".join(
            f"{k}={v}" for k, v in sorted(counts.items())))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# paired statistics (V8-15.05)
# ---------------------------------------------------------------------------


def render_comparison(
    label: str,
    comp: Mapping[str, Any],
) -> str:
    """One paired-comparison block: delta + bootstrap bounds + McNemar
    + per-conversation deltas + cluster count (V8-15.05, K08)."""
    lines = [f"#### {label}", ""]
    if not isinstance(comp, Mapping) or comp.get("n") is None:
        lines.append(NOT_RUN)
        lines.append("")
        return "\n".join(lines)
    lines.append(
        f"paired bootstrap ({comp.get('unit', 'question')}-level, "
        f"{comp.get('resamples')} resamples, seed {comp.get('seed')}, "
        f"alpha {comp.get('alpha')}): n={comp.get('n')}")
    lines.append(
        f"- delta = {_f(comp.get('delta'), 4)} "
        f"(base {_f(comp.get('base_mean'), 4)} → "
        f"candidate {_f(comp.get('candidate_mean'), 4)})")
    lines.append(
        f"- bounds: ci_lower {_f(comp.get('ci_lower'), 4)} · "
        f"ci_upper {_f(comp.get('ci_upper'), 4)}")
    mc = comp.get("mcnemar")
    if isinstance(mc, Mapping):
        lines.append(
            f"- McNemar exact p = {_f(mc.get('p'), 4)} "
            f"(base-only {mc.get('base_only')}, "
            f"candidate-only {mc.get('candidate_only')}, "
            f"discordant {mc.get('discordant')})")
    clusters = comp.get("clusters")
    if isinstance(clusters, Mapping):
        lines.append(
            f"- clusters: {clusters.get('count')} conversation(s)")
        per = clusters.get("per_conversation") or {}
        for cid in sorted(per):
            c = per[cid]
            lines.append(
                f"  - {cid}: delta {_f(c.get('delta'), 4)} "
                f"(n={c.get('n')}, "
                f"{_f(c.get('base_mean'), 4)} → "
                f"{_f(c.get('candidate_mean'), 4)})")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the full report (V8-15.17)
# ---------------------------------------------------------------------------


def render_report(
    *,
    ledger_records: Sequence[Mapping[str, Any]] = (),
    scoreboard: Optional[Mapping[str, Mapping[str, Any]]] = None,
    gates: Optional[Mapping[str, Mapping[str, Any]]] = None,
    comparisons: Optional[Mapping[str, Mapping[str, Any]]] = None,
    header: Optional[Mapping[str, Any]] = None,
    blocked: Optional[Iterable[str]] = None,
) -> str:
    """Assemble ``report_v8.md``: header (V8-15.17), scoreboard, gates,
    ledger table, paired statistics, blocked/deferred items."""
    h = dict(header or {})
    lines = ["# Verbatim V8 report — report_v8.md", ""]
    head_bits = []
    if h.get("date"):
        head_bits.append(f"date {h['date']}")
    if h.get("git_rev"):
        dirty = h.get("dirty")
        tag = ("dirty" if dirty else "clean") if dirty is not None \
            else "dirty=?"
        head_bits.append(f"git {_short(h['git_rev'], 12)} ({tag})")
    if h.get("machine"):
        head_bits.append(f"machine {h['machine']}")
    if h.get("load") is not None:
        head_bits.append(f"load {h['load']}")
    mans = h.get("manifests") or []
    head_bits.append(
        "manifests " + (", ".join(_short(m, 12) for m in mans)
                        if mans else "none"))
    lines.append(" · ".join(head_bits))
    lines.append("")
    lines.append(render_scoreboard(scoreboard))
    if gates is not None:
        lines.append(render_gates_table(gates))
    lines.append(render_ledger_table(ledger_records))
    if comparisons:
        lines.append("### Paired statistics (V8-15.05)")
        lines.append("")
        for label in sorted(comparisons):
            lines.append(render_comparison(label, comparisons[label]))
    bl = list(blocked or [])
    if bl:
        lines.append("### Blocked / deferred")
        lines.append("")
        for b in bl:
            lines.append(f"- {b}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m eval.v8.report",
        description=__doc__.splitlines()[0])
    ap.add_argument("--ledger", default=L.DEFAULT_LEDGER_PATH,
                    help="ledger.jsonl to render")
    args = ap.parse_args(argv)
    recs = L.read_ledger(args.ledger)
    print(render_report(ledger_records=recs, scoreboard={}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
