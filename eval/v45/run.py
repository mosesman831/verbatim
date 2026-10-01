"""v4.5 ablation runner — one entry point for the measured experiments
(SPEC_V4_5; I1+D02, I2+D03/D04, I3+D05/D06, I4+D07/D08, I5+D09/D10,
I6+D11/D12).

``python -m eval.v45.run --experiment i1|i2|i3|i4|i5|i6|all --out <dir>``
executes each arm through real production paths inside disposable
``Store`` instances and writes JSON+MD reports. Nothing is simulated;
a comparator that cannot run would be declared ``unavailable`` with a
reason — never fabricated.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from . import (
    i1_manifest,
    i2_repair_locality,
    i3_sufficiency,
    i4_transfer,
    i5_branch_apply,
    i6_refresh,
)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="v4.5 measured-ablation runner (SPEC_V4_5)"
    )
    ap.add_argument(
        "--experiment", default="all",
        choices=["i1", "i2", "i3", "i4", "i5", "i6", "all"],
    )
    ap.add_argument("--out", default="eval/v45/out")
    ap.add_argument("--topics", type=int, default=None,
                    help="topics per slice (i1 default 8, i3 default 6)")
    ap.add_argument("--procedures", type=int, default=None,
                    help="compiled procedures per arm (i4 default 6)")
    ap.add_argument("--slots", type=int, default=None,
                    help="claim slots in the repair graph (i2 default 16)")
    ap.add_argument("--periods", type=int, default=None,
                    help="refresh cycles per arm (i6 default 6)")
    ap.add_argument("--workdir", default=None,
                    help="seed the disposable store here instead of a "
                         "tempdir (kept for inspection)")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    summary = {}
    if args.experiment in ("i1", "all"):
        rep = i1_manifest.run_i1(
            topics=args.topics or 8, workdir=args.workdir
        )
        paths = i1_manifest.write_reports(rep, args.out)
        summary["i1"] = {
            "met": rep["met"],
            "topk_false_current": rep["totals"]["topk_false_current"],
            "manifest_false_current":
                rep["totals"]["manifest_false_current"],
            "report": paths["json"],
        }
    if args.experiment in ("i2", "all"):
        rep = i2_repair_locality.run_i2(
            slots=args.slots or 16, workdir=args.workdir
        )
        paths = i2_repair_locality.write_reports(rep, args.out)
        summary["i2"] = {
            "met": rep["met"],
            "evaluated_reduction":
                rep["totals"]["reduction"]["objects_evaluated"],
            "recomputed_reduction":
                rep["totals"]["reduction"]["objects_recomputed"],
            "d04_fences": rep["d04_met"],
            "report": paths["json"],
        }
    if args.experiment in ("i3", "all"):
        rep = i3_sufficiency.run_i3(
            topics=args.topics or 6, workdir=args.workdir
        )
        paths = i3_sufficiency.write_reports(rep, args.out)
        summary["i3"] = {
            "met": rep["met"],
            "flat_tokens": rep["tokens"]["flat_total"],
            "progressive_tokens": rep["tokens"]["progressive_total"],
            "reduction_pct": rep["tokens"]["reduction_pct"],
            "non_inferior": rep["utility"]["non_inferior"],
            "report": paths["json"],
        }
    if args.experiment in ("i4", "all"):
        rep = i4_transfer.run_i4(
            procedures=args.procedures or 6, workdir=args.workdir
        )
        paths = i4_transfer.write_reports(rep, args.out)
        summary["i4"] = {
            "met": rep["met"],
            "heldout_negative_transfer":
                rep["d07"]["heldout_negative_transfer"],
            "inenv_success_preserved":
                rep["d07"]["inenv_success_preserved"],
            "exposed_only_not_counted":
                rep["d08"]["exposed_only_not_counted"],
            "self_report_not_counted":
                rep["d08"]["self_report_not_counted"],
            "report": paths["json"],
        }
    if args.experiment in ("i5", "all"):
        rep = i5_branch_apply.run_i5(workdir=args.workdir)
        paths = i5_branch_apply.write_reports(rep, args.out)
        summary["i5"] = {
            "met": rep["met"],
            "checks": rep["checks"],
            "report": paths["json"],
        }
    if args.experiment in ("i6", "all"):
        rep = i6_refresh.run_i6(
            periods=args.periods or 6, workdir=args.workdir
        )
        paths = i6_refresh.write_reports(rep, args.out)
        summary["i6"] = {
            "met": rep["met"],
            "matched_freshness": rep["d11"]["matched_freshness"],
            "claims_savings": rep["d11"]["claims_savings"],
            "bytes_savings": rep["d11"]["bytes_savings"],
            "owner_scheduled": rep["d12"]["owner_scheduled"],
            "owner_deferred": rep["d12"]["owner_deferred"],
            "report": paths["json"],
        }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
