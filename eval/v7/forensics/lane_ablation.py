"""Lane on/off paired ablation — SPEC_V8 V8-05.01 (graph), V8-11.01
(leave-one-lane-out substrate).

Runs the Track R question stream twice over ONE ingested store —
once with the lane in the policy tuple (``<lane>_on``) and once with it
removed by :class:`~eval.v7.forensics._common.PolicyPatch`
(``no_<lane>``).  Removing the lane from ``ctx.policy.lanes`` is the
§02.3-correct mechanism: the ``lanes_disabled``→``config.v3.retrieval``
surface does NOT gate V7 lanes and is deliberately not used.

Output (``forensics/lane_ablation-v1`` via the shared
``forensics/v8-a`` envelope)::

  {
    "specs": {"<lane>_on": {...applied, verify, overall, categories...},
              "no_<lane>": {...}},
    "questions": [{task_id, conv_id, category, category_id, answerable,
                   gold, gold_session,
                   "arms": {"<lane>_on": {delivered, surfaced,
                            gold_ranks, first_gold_rank, pool_gold_rank,
                            status, attribution, latency_ms, sql, …},
                            "no_<lane>": {...same…}}}],
    "paired": {"first", "second",
               "any_at_k": {k: {both_hit, only_first, only_second,
                                both_miss}},
               "metric_delta": {any@10: {first, second, delta}, …}}
  }

``paired.any_at_k`` is the McNemar substrate; per-question rows carry
delivered refs + gold ranks + latency for paired bootstrap.  Application
is verified against ``coverage.explain.policy.lanes`` — a run whose lane
did not actually leave the tuple reports ``applied: false``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ._common import (
    ArmSpec,
    PolicyPatch,
    arm_report_kwargs,
    paired_run,
    write_report,
)

SCHEMA = "forensics/lane_ablation-v1"

#: Lane names the V7 policy tuple can carry (``LANES_V1`` + declared
#: extension lanes).  Unknown names are passed through — ``ablation_lanes``
#: validates and the failure lands in the spec's ``applied: false``.
KNOWN_V7_LANES = (
    "lex", "fuzzy", "dense", "ent", "time", "graph", "typed", "obs",
    "exact_id", "source", "scope",
)


def run_lane_ablation(
    corpus: Any,
    lane: str,
    *,
    k_list: Sequence[int] = (10, 20),
    arm_kwargs: Optional[Mapping[str, Any]] = None,
    policy_overrides: Optional[Mapping[str, Any]] = None,
    policy_doc: Optional[Mapping[str, Any]] = None,
    census: bool = False,
    full_explain: bool = False,
    out: Optional[str] = None,
) -> Dict[str, Any]:
    """Paired lane-on vs lane-off measurement (V8-05.01 shape).

    ``policy_doc`` optionally carries a base ``load_policy`` table applied
    to BOTH arms (e.g. a non-default ``lanes`` set); the off arm removes
    ``lane`` from whatever the doc resolved.  ``census`` attaches a
    :class:`SqlCensus` so each question row reports the measured call's
    statement census (V8-14.02 pairing for e.g. the graph ≤ 8-units
    scenario)."""
    lane = str(lane)
    specs = [
        ArmSpec(
            label=f"{lane}_on",
            patch=PolicyPatch(policy_doc=policy_doc) if policy_doc else None,
            census=census,
            notes=[f"baseline — lane {lane!r} left in the policy tuple"],
        ),
        ArmSpec(
            label=f"no_{lane}",
            patch=PolicyPatch(
                lanes_disabled=[lane], policy_doc=policy_doc,
            ),
            census=census,
            notes=[f"lane {lane!r} removed from ctx.policy.lanes "
                   "(§02.3-correct ablation)"],
        ),
    ]
    report = paired_run(
        corpus,
        specs,
        k_list=k_list,
        arm_kwargs=arm_kwargs,
        policy_overrides=policy_overrides,
        full_explain=full_explain,
        need_census=census,
        tool="lane_ablation",
        extra_manifest={
            "requirement": "V8-05.01/V8-11.01",
            "ablated_lane": lane,
            "base_policy_doc": policy_doc,
            "mechanism": "ablation_lanes on the resolved policy tuple "
                         "(never config.v3.retrieval)",
        },
    )
    report["schema"] = SCHEMA
    write_report(report, out)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m eval.v7.forensics.lane_ablation",
        description="V8-05.01 paired lane on/off ablation (policy tuple)",
    )
    ap.add_argument("--dataset", required=True,
                    help="dataset-registry id (e.g. owned_locomo_like, locomo)")
    ap.add_argument("--split", default=None)
    ap.add_argument("--lane", required=True,
                    help="lane to ablate (lex, fuzzy, dense, ent, time, "
                         "graph, typed, obs, exact_id, source, scope)")
    ap.add_argument("--k", default="10,20")
    ap.add_argument("--timeout-ms", type=float, default=None)
    ap.add_argument("--pool-limit", type=int, default=None)
    ap.add_argument("--settle-timeout", type=float, default=None)
    ap.add_argument("--policy-overrides", default=None,
                    help="JSON object → VerbatimArm(policy_overrides=…)")
    ap.add_argument("--policy-doc", default=None,
                    help="JSON load_policy document applied to both arms")
    ap.add_argument("--census", action="store_true",
                    help="attach the V8-14.02 statement census per query")
    ap.add_argument("--full-explain", action="store_true",
                    help="retain full coverage.explain per question")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from ..corpora import load_corpus

    corpus = load_corpus(args.dataset, args.split)
    ks = tuple(int(x) for x in args.k.split(",") if x.strip())
    rep = run_lane_ablation(
        corpus,
        args.lane,
        k_list=ks,
        arm_kwargs=arm_report_kwargs(
            timeout_ms=args.timeout_ms,
            pool_limit=args.pool_limit,
            settle_timeout_s=args.settle_timeout,
        ),
        policy_overrides=(
            json.loads(args.policy_overrides)
            if args.policy_overrides else None
        ),
        policy_doc=json.loads(args.policy_doc) if args.policy_doc else None,
        census=args.census,
        full_explain=args.full_explain,
        out=args.out,
    )
    specs = rep.get("specs") or {}
    for label, spec in specs.items():
        ov = spec.get("overall") or {}
        lat = (ov.get("latency_ms") or {})
        print(
            f"{label:>16s} applied={spec.get('applied')} "
            f"any@10={ov.get('any@10')} mrr@10={ov.get('mrr@10')} "
            f"p50={lat.get('p50')} p95={lat.get('p95')}",
            file=sys.stderr,
        )
    return 0


__all__ = ["KNOWN_V7_LANES", "SCHEMA", "run_lane_ablation"]


if __name__ == "__main__":
    raise SystemExit(main())
