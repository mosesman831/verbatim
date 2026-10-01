"""V6-03.03 ``auto_safe`` twin suite — the published bound on
false-auto-update.

Paired corpus run end-to-end through the ``Memory`` facade on
disposable stores, with the namespace policy set to ``auto_safe``:

* **TRUE-update pairs** — same entity/identifier anchor, a newer
  value/version/date or a strictly more specific restatement — SHOULD
  auto-apply: the prior's ``source_state`` must fence to
  ``superseded`` naming the new revision.
* **FALSE-update twins** — hedged/intent mentions, quoted/hearsay,
  hypothetical/future-plan text, missing or cross-entity anchors,
  ``contradicts``/``negates`` relations, and below-floor scores — MUST
  NOT auto-apply; they stay advisory ``possible_updates``.

The runner drives the real surfaces: ``Memory.add`` — where
``detect_update_candidates`` runs inside the commit and ``auto_safe``
application lands through the same coordinator effect as
``add(replaces=)`` — then the committed ``source_state`` rows for the
verdict ("applied" = the prior is superseded *by this new record*; a
residual post-hoc pass over still-open candidates audits the seam).
Each pair gets a fresh store so no pair's priors can leak into another
pair's candidate scan.

Report keys: ``auto_applied`` (true pairs that applied),
``should_apply`` (true-pair count), ``false_applied`` / ``false_total``
(false twins applied / total), ``upper95_bound`` (Wilson upper bound on
the false-apply rate — the honest form of "no false applies observed"),
and ``verdict`` ``"pass"|"fail"`` (pass = every true pair applied AND
zero false applies). ``bound_target``/``bound_met`` publish the
V6-03.03 <0.01 upper-95% gate honestly: ~300 zero-failure twins are
required before the bound can clear it — the corpus's measured outcome
is reported, never dressed up as proof.

CLI: ``python -m eval.v6.twins --workdir <dir>`` →
``twins_report.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import Any, Dict, List, Optional

from eval.v3.suites import wilson_upper_bound
from verbatim.memory.facade import Memory
from verbatim.querying.auto_update import (
    auto_safe_replace,
    set_update_policy,
)
from verbatim.querying.updates import NewRecord, list_open_candidates


# ---------------------------------------------------------------------
# twin corpus
# ---------------------------------------------------------------------
#
# ``kind`` — ``true`` pairs SHOULD auto-apply under auto_safe;
# ``false`` twins MUST stay advisory. ``guard`` names the auto_update
# guard expected to refuse (None when detection itself emits no
# candidate — the advisory path never reaches the auto gate).

TWIN_PAIRS: List[Dict[str, Any]] = [
    # --- TRUE updates: anchored, unhedged, safe relation, >= floor ---
    {
        "name": "version_bump_anchored",
        "kind": "true",
        "prior": "Ticket INC-431: the deploy command is deploy-v1.",
        "new": "Ticket INC-431: the deploy command is deploy-v2.",
        "expect_relation": "newer_value",
    },
    {
        "name": "toolchain_version",
        "kind": "true",
        "prior": "BUG-2210 build pipeline uses gcc 13.2.",
        "new": "BUG-2210 build pipeline uses gcc 13.3.",
        "expect_relation": "newer_value",
    },
    {
        "name": "anchored_refines",
        "kind": "true",
        "prior": "MAINT-9 decision is deploy on Tuesday.",
        "new": "MAINT-9 decision is deploy on Tuesday at 3pm.",
        "expect_relation": "refines",
    },
    {
        "name": "date_move_anchored",
        "kind": "true",
        "prior": "CHANGE-7 maintenance window starts 2024-05-01.",
        "new": "CHANGE-7 maintenance window starts 2024-06-03.",
        "expect_relation": "newer_value",
    },
    {
        "name": "timeout_move",
        "kind": "true",
        "prior": "SEC-12 session timeout is 30 minutes.",
        "new": "SEC-12 session timeout is 60 minutes.",
        "expect_relation": "newer_value",
    },
    # --- FALSE twins: every one MUST stay advisory -------------------
    {
        # normative intent, not an assertion — a REAL high-scoring
        # newer_value candidate with a shared anchor; refused by
        # unhedged_new (and same_type: it classifies procedure_hint).
        "name": "intent_modal",
        "kind": "false",
        "prior": "INC-431 deploy command is deploy-v1.",
        "new": "We should switch INC-431 deploy command to deploy-v2.",
        "expect_relation": "newer_value",
        "guard": "unhedged_new",
    },
    {
        # modal + complementizer — a request to verify, not an update.
        "name": "confirm_that",
        "kind": "false",
        "prior": "INC-431 deploy command is deploy-v1.",
        "new": "Please confirm that INC-431 deploy command is deploy-v2.",
        "expect_relation": None,
        "guard": "unhedged_new",
    },
    {
        # hedge adverb.
        "name": "hedged_maybe",
        "kind": "false",
        "prior": "INC-431 deploy command is deploy-v1.",
        "new": "Maybe the INC-431 deploy command is deploy-v2.",
        "expect_relation": None,
        "guard": "unhedged_new",
    },
    {
        # same subject, value moved, but NO shared identifier/entity
        # anchor — the canonical advisory case.
        "name": "no_anchor",
        "kind": "false",
        "prior": "The deploy command is deploy-v1.",
        "new": "The deploy command is deploy-v2.",
        "expect_relation": "newer_value",
        "guard": "shared_anchor",
    },
    {
        # identifier anchor exists but points at a DIFFERENT entity —
        # cross-entity is never safe-auto.
        "name": "cross_entity",
        "kind": "false",
        "prior": "INC-431 deploy command is deploy-v1.",
        "new": "INC-999 deploy command is deploy-v2.",
        "expect_relation": "newer_value",
        "guard": "shared_anchor",
    },
    {
        # a real candidate above floor with anchor — but a conflicting
        # value is exactly what an operator must see.
        "name": "contradicts_color",
        "kind": "false",
        "prior": "INC-431 favorite color is blue.",
        "new": "INC-431 favorite color is green.",
        "expect_relation": "contradicts",
        "guard": "relation",
    },
    {
        # explicit negation of a prior affirmative — never safe-auto.
        "name": "negates_usage",
        "kind": "false",
        "prior": "INC-431 service uses the staging database.",
        "new": "INC-431 service no longer uses the staging database.",
        "expect_relation": "negates",
        "guard": "relation",
    },
    {
        # quoted/hearsay — attributed, not asserted.
        "name": "quoted_runbook",
        "kind": "false",
        "prior": "INC-431 deploy command is deploy-v1.",
        "new": 'The runbook says "INC-431 deploy command is deploy-v2".',
        "expect_relation": None,
        "guard": "unhedged_new",
    },
    {
        "name": "hypothetical",
        "kind": "false",
        "prior": "INC-431 deploy command is deploy-v1.",
        "new": "What if the INC-431 deploy command were deploy-v3?",
        "expect_relation": None,
        "guard": "unhedged_new",
    },
    {
        "name": "future_plan",
        "kind": "false",
        "prior": "INC-431 deploy command is deploy-v1.",
        "new": "We plan to change INC-431 deploy command to deploy-v3.",
        "expect_relation": None,
        "guard": "unhedged_new",
    },
    {
        # safe relation class but below the pinned floor — the score
        # guard, not detection, keeps it advisory.
        "name": "below_floor_refines",
        "kind": "false",
        "prior": "INC-55 retro meeting is Tuesday.",
        "new": "INC-55 retro meeting is Tuesday at 3pm.",
        "expect_relation": "refines",
        "guard": "score_floor",
    },
    {
        # disjoint environment qualifiers — different worlds.
        "name": "environment_difference",
        "kind": "false",
        "prior": "On macOS the install command is brew install foo.",
        "new": "On linux the install command is apt install foo.",
        "expect_relation": None,
        "guard": None,
    },
    {
        # pronoun subject binds nothing.
        "name": "ambiguous_pronoun",
        "kind": "false",
        "prior": "INC-431 deploy command is deploy-v1.",
        "new": "It changed to deploy-v2.",
        "expect_relation": None,
        "guard": None,
    },
]

#: V6-03.03's published false-auto-update gate.
BOUND_TARGET = 0.01


def twin_corpus() -> List[Dict[str, Any]]:
    """The frozen pair list — sized so ≥4 true + ≥8 false twins run."""
    return list(TWIN_PAIRS)


# ---------------------------------------------------------------------
# one pair, one disposable store
# ---------------------------------------------------------------------


def _run_pair(stores_dir: str, pair: Dict[str, Any]) -> Dict[str, Any]:
    path = os.path.join(stores_dir, pair["name"] + ".db")
    mem = Memory(path=path, worker="external")
    try:
        ns = mem._namespace
        with mem._store.tx() as conn:
            set_update_policy(conn, ns, "auto_safe")

        r_prior = mem.add(pair["prior"])
        res = mem.add(pair["new"])

        applied: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []
        cand_rows: List[Dict[str, Any]] = []
        with mem._store.tx() as conn:
            cand_rows = list_open_candidates(
                conn, ns, source_id=res.memory_id
            )
            for cand in cand_rows:
                out = auto_safe_replace(
                    conn,
                    namespace=ns,
                    new_record=NewRecord(
                        source_id=res.memory_id,
                        revision=res.source_revision,
                        text=pair["new"],
                    ),
                    candidate=cand,
                    store=mem._store,
                )
                if out is not None:
                    applied.append(
                        {
                            "candidate_id": cand.get("candidate_id"),
                            "relation": cand.get("relation"),
                            "prior_source_id": cand.get("prior_source_id"),
                        }
                    )
                    decisions.append(out["decision"])

        with mem._store.read() as conn:
            row = conn.execute(
                "SELECT disposition, superseded_by, control_version"
                " FROM source_state WHERE source_id = ?",
                (r_prior.memory_id,),
            ).fetchone()
        prior_disposition = row[0] if row else None
        superseded_by = row[1] if row else None

        # Post-integration the facade applies auto_safe inside the add
        # commit, so "applied" is measured by the end state — the prior
        # is superseded BY this new record — not by a post-hoc call
        # seeing open candidates (there are none; they resolve in-tx).
        did_apply = bool(
            prior_disposition == "superseded"
            and superseded_by == f"{res.memory_id}:{res.source_revision}"
        )
        return {
            "name": pair["name"],
            "kind": pair["kind"],
            "should_apply": pair["kind"] == "true",
            "candidates": len(cand_rows),
            "relations": [c.get("relation") for c in cand_rows],
            "applied": did_apply,
            "applied_detail": applied,
            "prior_disposition": prior_disposition,
            "superseded_by": superseded_by,
            "ok": did_apply == (pair["kind"] == "true"),
        }
    finally:
        mem.close()


# ---------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------


def run_twins(workdir: Optional[str] = None) -> Dict[str, Any]:
    """Run the paired corpus under ``auto_safe`` and report the bound.

    ``workdir`` holds the disposable per-pair stores and the emitted
    ``twins_report.json``; ``None`` uses a temp directory.
    """
    root = workdir or tempfile.mkdtemp(prefix="v6-twins-")
    os.makedirs(root, exist_ok=True)
    stores_dir = os.path.join(root, "stores")
    os.makedirs(stores_dir, exist_ok=True)

    pairs = [_run_pair(stores_dir, pair) for pair in TWIN_PAIRS]

    should_apply = sum(1 for p in pairs if p["should_apply"])
    auto_applied = sum(
        1 for p in pairs if p["should_apply"] and p["applied"]
    )
    false_total = sum(1 for p in pairs if not p["should_apply"])
    false_applied = sum(
        1 for p in pairs if not p["should_apply"] and p["applied"]
    )
    bound = (
        wilson_upper_bound(false_applied, false_total)
        if false_total
        else 0.0
    )
    verdict = (
        "pass"
        if (auto_applied == should_apply and false_applied == 0)
        else "fail"
    )
    report = {
        "suite": "v6_auto_safe_twins/v1",
        "auto_applied": auto_applied,
        "should_apply": should_apply,
        "false_applied": false_applied,
        "false_total": false_total,
        "upper95_bound": round(bound, 4),
        "bound_target": BOUND_TARGET,
        "bound_met": bool(bound < BOUND_TARGET),
        "verdict": verdict,
        "pairs": pairs,
    }
    out_path = os.path.join(root, "twins_report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    report["report_path"] = out_path
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--workdir",
        default=None,
        help="directory for disposable stores + twins_report.json",
    )
    args = ap.parse_args(argv)
    report = run_twins(args.workdir)
    print(
        f"auto_safe twins: {report['auto_applied']}/"
        f"{report['should_apply']} true applied, "
        f"{report['false_applied']}/{report['false_total']} false applied, "
        f"upper95 {report['upper95_bound']:.4f} "
        f"(target <{BOUND_TARGET}) — verdict {report['verdict']}"
    )
    print(f"report: {report['report_path']}")
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
