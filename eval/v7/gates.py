"""V7 gate evaluator — skeleton (SPEC_V7 §26, V7-26.02).

Gates are evaluated from **executed artifacts and the ledger only**.
``GATE_INPUTS`` below is the gate → required-artifact map declared by
the §26 ``Requires`` cells; each input is a stable artifact id that a
later measurement wave fills by writing an artifact record into the
artifacts index (default ``eval/v7/artifacts/index.json`` — an index
is read, never probed: an artifact the index does not name is
``not_run``).

Aggregation (V7-26.02 — a gate with any ``not_run`` input is
``not_run``, never ``passed``):

    missed/failed input  -> gate ``missed``
    invalid input        -> gate ``invalid``   (e.g. same-reader breach)
    blocked input        -> gate ``blocked_on_authorization``
    missing/not_run input-> gate ``not_run``   (the absence floor)
    all inputs ``passed``-> gate ``passed``

``blocked_on_authorization`` outranks ``not_run`` because it names the
actionable reason (V7-00.04/V7-03.02) rather than mere absence; a gate
with both is reported blocked. ``passed`` requires every input passed.

Skeleton status: no artifact index exists yet, so every gate evaluates
``not_run`` — that is the correct honest output for wave A, and the
table CLI prints exactly that.

CLI: ``python -m eval.v7.gates [--root R] [--ledger PATH]
[--artifacts PATH]`` prints the gate table (always exit 0 — this is a
report, not an enforcement hook).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

#: Gate rollup statuses — the §22.4/SB row vocabulary carried to gates.
GATE_STATUSES = (
    "passed",
    "missed",
    "not_run",
    "blocked_on_authorization",
    "invalid",
)

#: Input-level statuses an artifact record may carry. ``failed`` folds
#: into ``missed`` at the gate; anything else is treated ``not_run``.
INPUT_STATUSES = GATE_STATUSES + ("failed",)

#: Default inputs, relative to repo root.
DEFAULT_LEDGER_PATH = os.path.join("eval", "v7", "ledger_v7.json")
DEFAULT_ARTIFACTS_PATH = os.path.join(
    "eval", "v7", "artifacts", "index.json")

# ---------------------------------------------------------------------------
# the gate -> required-artifact map, transcribed from §26 Requires cells
# ---------------------------------------------------------------------------

#: Each entry ``(artifact_id, what)`` — ``what`` quotes the Requires
#: cell clause it satisfies so the map is auditable against §26.
GATE_INPUTS: dict = {
    "G7-00": (
        ("v6.gate.G6-00", "V6 G6-00 holds"),
        ("mutation.v7_lanes",
         "mutation suite kills quote/eligibility/closure/generation "
         "skips on every new lane and index; new mutants for V7 lanes "
         "(≥ 12) killed"),
    ),
    "G7-01": (
        ("track_r.owned_twins",
         "Track R on owned twins: V7 ≥ BM25 reference on every "
         "category at any@10 and any@20"),
        ("track_r.permitted_public",
         "Track R on every permitted public set: same parity rule"),
    ),
    "G7-02": (
        ("sb.SB-01", "SB-01 pass"),
        ("sb.SB-02", "SB-02 pass"),
        ("sb.SB-03", "SB-03 pass"),
        ("ablation.leave_one_lane_out",
         "leave-one-lane-out table published"),
        ("track_r.zero_result_rate", "zero-result rate ≤ 1%"),
        ("formula_search.report",
         "the formula-search artifact (winning tags, ablation table, "
         "rejected alternatives with the metric that rejected each, "
         "search manifest digest) accepted under O9"),
    ),
    "G7-03": (
        ("fixture.temporal_v2", "V7-09.04 fixture ≥ 0.97"),
        ("suite.temporal_algebra", "algebra suite green"),
        ("track_r.locomo_cat2", "LoCoMo cat 2 Track R target met"),
        ("track_r.lme_temporal", "LME temporal Track R target met"),
    ),
    "G7-04": (
        ("envelope.B0", "B0 pass on the reference machine"),
        ("envelope.B1", "B1 pass on the reference machine"),
        ("envelope.B2", "B2 pass on the reference machine"),
        ("envelope.B6", "B6 pass on the reference machine"),
        ("budget.stage", "stage budgets met"),
        ("budget.sql", "SQL budget met"),
    ),
    "G7-05": (
        ("envelope.B3", "B3 (100K) pass"),
        ("envelope.B4", "B4 (1M) pass"),
        ("scale.stage_alpha", "α ≤ 1.0 per stage"),
        ("budget.rss_disk", "RSS and disk budgets met"),
    ),
    "G7-06": (
        ("sb.SB-17", "SB-17 pass under B2"),
        ("sb.SB-18", "SB-18 pass under B2"),
        ("scale.add_ack_alpha", "add-ack flat 1K→100K (α ≤ 0.2)"),
    ),
    "G7-07": (
        ("paired.v7-07.03", "V7-07.03 paired gains"),
        ("dense.quantization_loss", "quantization loss ≤ 1.0 pt"),
        ("license.flags", "license flags clean"),
    ),
    "G7-08": (
        ("paired.v7-10.09", "V7-10.09 deltas"),
        ("envelope.B1-CE", "B1-CE passes"),
    ),
    "G7-09": (
        ("sb.SB-19", "SB-19 pass"),
        ("sb.SB-20", "SB-20 pass"),
        ("trust.closure_verify",
         "closure verify covers every V7 artifact class"),
        ("trust.grounding_sweep",
         "zero delivered unpinned items across all runs"),
    ),
    "G7-10": (
        ("track_q.open",
         "Track Q-open rows executed under authorization with ≥ 2 "
         "readers, neutral prompts, CIs published"),
        ("authorization.track_q_open", "authorization record (V7-00.04)"),
        ("trust.anti_gaming_tripwire", "tripwire clean"),
    ),
    "G7-11": (
        ("track_q.frontier", "Track Q-frontier rows executed"),
        ("track_q.neutral", "Track Q-neutral rows executed"),
        ("check.same_reader", "same-reader rule holds (V7-22.14)"),
        ("trust.anti_gaming_tripwire", "anti-gaming tripwire clean"),
    ),
    "G7-12": (
        ("comparator.in_harness",
         "the vendor's OSS arm executed in-harness (V7-23.01) on the "
         "same rows"),
        ("stats.paired_lower_bound", "paired lower bound > 0"),
        ("matrix.hindsight_parity", "Hindsight parity matrix closed"),
        ("note.vendor_review",
         "vendor review note sent (V7-23.03)"),
    ),
    "G7-13": (
        ("dataset.beam", "BEAM licensed and executed"),
        ("actions.dolphinbench", "DolphinBench licensed and executed"),
        ("envelope.B5", "B5 envelope executed"),
    ),
    "G7-14": (
        ("ci.v7-21.03_matrix", "V7-21.03 CI matrix green"),
        ("release.distribution_name", "distribution name resolved"),
        ("ci.docs_executed", "docs executed in CI"),
        ("page.benchmarks",
         "benchmark page generated from artifacts only"),
    ),
}

#: Per-gate honesty notes — constraints the artifact map alone cannot
#: express (V7-26.03): retrieval-only artifacts can never satisfy the
#: Track Q gates, and G7-02 additionally gates on the O9 decision.
GATE_NOTES = {
    "G7-02": "formula=unselected until O9 accepts the search report",
    "G7-10": "no retrieval-only artifact may satisfy this gate",
    "G7-11": "no retrieval-only artifact may satisfy this gate",
}

#: Severity order for aggregation — earlier wins.
_SEVERITY = ("missed", "invalid", "blocked_on_authorization", "not_run")


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def load_artifacts(path: str) -> dict:
    """Read an artifacts index: ``{artifact_id: {"status": ...,
    "manifest_digest": ..., "path": ...}}``. A missing/empty file is
    honest absence — every input is ``not_run``."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(doc, dict):
        return {}
    arts = doc.get("artifacts", doc)
    if not isinstance(arts, dict):
        return {}
    out = {}
    for aid, body in arts.items():
        out[str(aid)] = body if isinstance(body, dict) else {}
    return out


def load_ledger(path: str) -> dict:
    """Read ``ledger_v7.json`` (gate names + bound requirements).
    Missing/unparseable -> empty dict; gate ids still evaluate."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _input_status(record: Optional[dict]) -> str:
    """Normalize one artifact record to an input status."""
    if record is None:
        return "not_run"
    st = record.get("status")
    if st == "failed":
        return "missed"
    if st in GATE_STATUSES:
        return st
    return "not_run"


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def evaluate_gates(ledger: dict, artifacts: dict) -> dict:
    """``{gate_id: {...}}`` for every declared gate (G7-00..G7-14).

    ``ledger`` is a parsed ``ledger_v7.json`` (supplies names, permits,
    requires text, bound requirements); ``artifacts`` is the artifact
    index. Inputs the index does not name are ``not_run``.
    """
    ledger_gates = (ledger.get("gates") or {}) if ledger else {}
    out = {}
    for gid in sorted(GATE_INPUTS):
        inputs = []
        worst = None
        for aid, what in GATE_INPUTS[gid]:
            rec = artifacts.get(aid)
            st = _input_status(rec)
            inputs.append({
                "artifact": aid,
                "what": what,
                "status": st,
                "manifest_digest": (
                    (rec or {}).get("manifest_digest")),
            })
            if st != "passed" and (
                worst is None
                or _SEVERITY.index(st) < _SEVERITY.index(worst)
            ):
                worst = st
        status = worst if worst is not None else "passed"
        lg = ledger_gates.get(gid) or {}
        out[gid] = {
            "name": lg.get("name") or "",
            "permits": lg.get("permits") or "",
            "requires": lg.get("requires") or "",
            "status": status,
            "inputs": inputs,
            "inputs_ready": sum(1 for i in inputs
                              if i["status"] == "passed"),
            "inputs_total": len(inputs),
            "bound_requirements": list(lg.get("bound_requirements") or []),
            "note": GATE_NOTES.get(gid, ""),
        }
    return out


def evaluate(root: str,
             ledger_path: Optional[str] = None,
             artifacts_path: Optional[str] = None) -> dict:
    """Evaluate against the on-disk ledger + artifacts index."""
    ledger = load_ledger(
        ledger_path or os.path.join(root, DEFAULT_LEDGER_PATH))
    artifacts = load_artifacts(
        artifacts_path or os.path.join(root, DEFAULT_ARTIFACTS_PATH))
    return evaluate_gates(ledger, artifacts)


# ---------------------------------------------------------------------------
# rendering / CLI
# ---------------------------------------------------------------------------


def render_table(results: dict) -> str:
    """The §26 gate table as text — one line per gate."""
    lines = [
        "| Gate | Name | Status | Inputs ready |",
        "| --- | --- | --- | --- |",
    ]
    for gid, g in sorted(results.items()):
        lines.append(
            f"| {gid} | {g['name']} | {g['status']} | "
            f"{g['inputs_ready']}/{g['inputs_total']} |"
        )
    counts: dict = {}
    for g in results.values():
        counts[g["status"]] = counts.get(g["status"], 0) + 1
    lines.append("")
    lines.append(
        "statuses: " + " ".join(
            f"{k}={v}" for k, v in sorted(counts.items())))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m eval.v7.gates",
        description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
        help="repo root (default: auto)")
    ap.add_argument("--ledger", default=None,
                    help=f"ledger path (default {DEFAULT_LEDGER_PATH})")
    ap.add_argument("--artifacts", default=None,
                    help=f"artifacts index (default "
                         f"{DEFAULT_ARTIFACTS_PATH})")
    ap.add_argument("--json", action="store_true",
                    help="emit the full result map as JSON")
    args = ap.parse_args(argv)
    results = evaluate(
        args.root, ledger_path=args.ledger,
        artifacts_path=args.artifacts)
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print(render_table(results), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
