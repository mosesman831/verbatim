"""Rank forensic — SPEC_V8 V8-11.01.

Population: every answerable question where the flat-BM25 reference arm
ranks a gold ref at 1 and the verbatim arm does not (delivered rank > 1
or absent).  For each such question the tool records the *displacing
items'* evidence — lane ranks, per-lane signals, fused RRF, rerank
score, and feature contributions — decomposed from the real
``coverage.explain`` payload, plus a head-to-head decomposition of the
first gold item vs the top non-gold item.

It also publishes a leave-one-lane-out table for ``mrr@10`` and
``ndcg@10``: each named lane is removed from the policy tuple
(``ablation_lanes`` — the §02.3-correct channel) on the SAME ingested
store, so the table is a clean paired measurement, not a re-ingest.

Score decomposition (marked ``derived`` — recomputed from the recorded
policy and the item's own ``detail`` fields, never invented):

* ``lane_contrib[lane] = w_lane / (60 + lane_rank)`` — the lane's RRF
  share; ``w_lane`` comes from ``weights_for(intent, resolved_policy)``.
* ``rrf_residual = fused_rrf − Σ lane_contrib`` — gates/containment and
  constant-signal vetoes surface here honestly instead of being hidden.
* ``feature_contrib[f] = weights[f] · features[f]`` — the rerank
  feature's signed contribution to ``score``.
* ``signals`` — the per-lane raw signals exactly as explain printed them.

Rows for questions outside the population are not emitted; the file's
``n_population``/``n_answerable`` counts keep denominators honest.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .. import metrics as M
from ..arms import FlatBM25Arm, arm_task
from ._common import (
    ArmSpec,
    PolicyPatch,
    arm_report_kwargs,
    paired_run,
    task_views,
    write_report,
)

SCHEMA = "forensics/rank_forensic-v1"

#: RRF k — V7-10.01's declared constant (``retrieval/v7/fusion.py``).
RRF_K = 60.0

#: Default LOLO set: the eight S2 lanes of the r0 table.
DEFAULT_LOLO_LANES = (
    "lex", "fuzzy", "dense", "ent", "time", "graph", "typed", "obs",
)


def _lane_weights(intent: Any, policy_doc: Optional[Mapping[str, Any]],
                  profile: str = "default") -> Dict[str, float]:
    """Resolve the lane-weight row the pipeline applied — the same
    ``load_policy``/``weights_for`` path, run scorer-side."""
    try:
        from verbatim.retrieval.v7.policy import load_policy, weights_for
    except Exception:  # noqa: BLE001
        return {}
    try:
        pol = load_policy(profile, dict(policy_doc)) if policy_doc \
            else load_policy(profile)
        return {getattr(l, "value", str(l)): float(w)
                for l, w in weights_for(intent, pol).items()}
    except Exception:  # noqa: BLE001
        return {}


def decompose_item(
    item: Mapping[str, Any],
    *,
    ref: Optional[str],
    lane_weights: Mapping[str, float],
) -> Dict[str, Any]:
    """One ``explain.items`` entry → its score decomposition row."""
    detail = item.get("detail") or {}
    lane_ranks = detail.get("lane_ranks") or item.get("lane_ranks") or {}
    features = detail.get("features") or {}
    weights = detail.get("weights") or {}
    fused_rrf = item.get("fused_rrf")
    if fused_rrf is None:
        fused_rrf = detail.get("rrf")

    lane_contrib: Dict[str, float] = {}
    for lane, rank in lane_ranks.items():
        try:
            w = float(lane_weights.get(str(lane), 1.0))
            lane_contrib[str(lane)] = round(
                w / (RRF_K + float(rank)), 9
            )
        except Exception:  # noqa: BLE001 — bad rank: skip lane, not item
            continue
    contrib_sum = sum(lane_contrib.values())
    feature_contrib = {
        str(f): round(float(weights.get(f, 0.0)) * float(v), 9)
        for f, v in features.items()
        if _is_num(v)
    }
    return {
        "unit_id": item.get("unit_id"),
        "source_id": item.get("source_id"),
        "revision": item.get("revision"),
        "ref": ref,
        "mapped": ref is not None,
        "lane_ranks": dict(lane_ranks),
        "signals": item.get("signals") or detail.get("signals") or {},
        "fused_rrf": fused_rrf,
        "lane_contrib": lane_contrib,
        "lane_contrib_sum": round(contrib_sum, 9),
        "rrf_residual": (
            round(float(fused_rrf) - contrib_sum, 9)
            if fused_rrf is not None else None
        ),
        "score": item.get("score"),
        "score_family": item.get("score_family"),
        "feature_score": detail.get("feature_score"),
        "features": dict(features),
        "feature_weights": dict(weights),
        "feature_contrib": feature_contrib,
        "tie_epsilon": detail.get("tie_epsilon"),
        "group_label": item.get("group_label"),
        "pack": item.get("pack"),
    }


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _ref_of(item: Mapping[str, Any], source_ref: Mapping[str, str]) -> Optional[str]:
    sid = item.get("source_id")
    return source_ref.get(str(sid)) if sid is not None else None


def forensic_rows(
    views: Sequence[Any],
    baseline_rows: Mapping[str, Mapping[str, Any]],
    explains: Mapping[str, Any],
    bm25: FlatBM25Arm,
    source_ref: Mapping[str, str],
    *,
    k: int,
    lane_weights_fn: Any,
) -> List[Dict[str, Any]]:
    """The V8-11.01 population + decomposition."""
    out: List[Dict[str, Any]] = []
    for tv in views:
        if not tv.answerable or not tv.gold_item:
            continue
        gold = set(tv.gold_item)

        bm_out = bm25.query(arm_task(tv), k)
        bm_ranks = [
            i + 1 for i, r in enumerate(bm_out.refs) if str(r) in gold
        ]
        bm_rank = min(bm_ranks) if bm_ranks else None
        if bm_rank != 1:
            continue  # outside the forensic population

        row = (baseline_rows.get(tv.task_id) or {})
        v_rank = row.get("first_gold_rank")
        if v_rank == 1:
            continue  # Verbatim also ranked gold first — no displacement

        explain = explains.get(tv.task_id)
        items = (explain or {}).get("items") or []
        intent = ((explain or {}).get("query") or {}).get("intent")
        lw = lane_weights_fn(intent)

        decomp: List[Tuple[int, Dict[str, Any]]] = []
        for pos, it in enumerate(items):
            ref = _ref_of(it, source_ref)
            decomp.append(
                (pos, decompose_item(it, ref=ref, lane_weights=lw))
            )
        gold_idx = next(
            (pos for pos, d in decomp
             if d["ref"] is not None and str(d["ref"]) in gold),
            None,
        )
        gold_item = decomp[gold_idx][1] if gold_idx is not None else None
        nongold = next(
            (d for _p, d in decomp
             if not (d["ref"] is not None and str(d["ref"]) in gold)),
            None,
        )
        displacers = [
            d for _p, d in decomp[:gold_idx or 0]
            if not (d["ref"] is not None and str(d["ref"]) in gold)
        ] if gold_idx is not None else []

        out.append({
            "task_id": tv.task_id,
            "conv_id": tv.group_id,
            "category": tv.category,
            "bm25": {
                "gold_rank": bm_rank,
                "top_refs": list(bm_out.refs[:k]),
            },
            "verbatim": {
                "delivered_gold_rank": v_rank,
                "pool_gold_rank": row.get("pool_gold_rank"),
                "status": row.get("status"),
                "attribution": row.get("attribution"),
                "explain_rank_of_gold": (
                    gold_idx + 1 if gold_idx is not None else None
                ),
            },
            "gold_item": gold_item,
            "top_nongold": nongold,
            "displacers": displacers,
            "explain_status": (
                "ok" if explain else "unavailable"
            ),
        })
    return out


def run_rank_forensics(
    corpus: Any,
    *,
    k: int = 10,
    lolo_lanes: Sequence[str] = (),
    arm_kwargs: Optional[Mapping[str, Any]] = None,
    policy_overrides: Optional[Mapping[str, Any]] = None,
    policy_doc: Optional[Mapping[str, Any]] = None,
    out: Optional[str] = None,
) -> Dict[str, Any]:
    """V8-11.01 measurement: forensic rows + leave-one-lane-out table.

    ``lolo_lanes`` names the lanes ablated for the MRR@10/nDCG@10 table
    (empty → the table reports ``not_run`` rather than fabricating rows).
    The baseline arm's explains drive the decomposition; every spec runs
    on one shared ingested store."""
    views = task_views(corpus)
    specs = [
        ArmSpec(label="baseline", patch=None, notes=["unmodified policy"]),
    ]
    for lane in lolo_lanes:
        specs.append(ArmSpec(
            label=f"no_{lane}",
            patch=PolicyPatch(
                lanes_disabled=[lane],
                policy_doc=policy_doc,
            ),
            notes=[f"leave-one-lane-out: {lane!r}"],
        ))

    sink: Dict[str, Any] = {}
    report = paired_run(
        corpus,
        specs,
        k_list=(k, 10) if k != 10 else (k,),
        arm_kwargs=arm_kwargs,
        policy_overrides=policy_overrides,
        full_explain=True,
        tool="rank_forensics",
        extra_manifest={
            "requirement": "V8-11.01",
            "rrf_k": RRF_K,
            "lane_weights_source": "weights_for(explain.query.intent, "
                                   "resolved policy) — derived",
            "lolo_lanes": list(lolo_lanes),
            "base_policy_doc": policy_doc,
        },
        sink=sink,
    )
    report["schema"] = SCHEMA
    if report.get("status") == "not_run":
        write_report(report, out)
        return report

    # The baseline spec's explain payloads (captured per spec inside
    # paired_run's loop) drive the decomposition rows.
    base_spec = report["specs"].get("baseline") or {}
    base_rows = {
        q["task_id"]: q["arms"]["baseline"]
        for q in report["questions"]
        if "baseline" in q.get("arms", {})
    }
    explains = (sink.get("explains") or {}).get("baseline") or {}
    source_ref = sink.get("source_ref") or {}

    bm = FlatBM25Arm()
    bm.ingest(corpus)
    try:
        rows = forensic_rows(
            views,
            base_rows,
            explains,
            bm,
            source_ref,
            k=int(k),
            lane_weights_fn=lambda intent: _lane_weights(intent, policy_doc),
        )
    finally:
        bm.close()

    report["population"] = {
        "definition": "answerable questions with bm25 gold_rank == 1 "
                      "and verbatim delivered gold rank != 1",
        "n_answerable": sum(
            1 for v in views if v.answerable and v.gold_item
        ),
        "n_population": len(rows),
    }
    report["rows"] = rows

    # ---- leave-one-lane-out table (MRR@10 / nDCG@10) ---------------------
    base_ov = base_spec.get("overall") or {}
    lolo: Dict[str, Any] = {
        "baseline": {
            "mrr@10": base_ov.get("mrr@10"),
            "ndcg@10": base_ov.get("ndcg@10"),
        }
    }
    if lolo_lanes:
        for lane in lolo_lanes:
            ov = (report["specs"].get(f"no_{lane}") or {}).get("overall") or {}
            lolo[f"no_{lane}"] = {
                "mrr@10": ov.get("mrr@10"),
                "ndcg@10": ov.get("ndcg@10"),
                "delta_mrr@10": _delta(ov.get("mrr@10"), base_ov.get("mrr@10")),
                "delta_ndcg@10": _delta(
                    ov.get("ndcg@10"), base_ov.get("ndcg@10")
                ),
            }
        lolo["status"] = "executed"
    else:
        lolo["status"] = "not_run"
        lolo["reason"] = "lolo_lanes empty — no ablations requested"
    report["leave_one_lane_out"] = lolo

    write_report(report, out)
    return report


def _delta(v: Any, base: Any) -> Optional[float]:
    if v is None or base is None:
        return None
    return round(float(v) - float(base), 9)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m eval.v7.forensics.rank_forensics",
        description="V8-11.01 rank forensic + leave-one-lane-out table",
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--lolo", default="",
                    help="comma-separated lanes for the LOLO table "
                         "(default: none — the forensic rows still run)")
    ap.add_argument("--lolo-all", action="store_true",
                    help="ablate all eight S2 lanes")
    ap.add_argument("--timeout-ms", type=float, default=None)
    ap.add_argument("--pool-limit", type=int, default=None)
    ap.add_argument("--settle-timeout", type=float, default=None)
    ap.add_argument("--policy-overrides", default=None)
    ap.add_argument("--policy-doc", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from ..corpora import load_corpus

    corpus = load_corpus(args.dataset, args.split)
    if args.lolo_all:
        lanes: Sequence[str] = DEFAULT_LOLO_LANES
    else:
        lanes = tuple(x.strip() for x in args.lolo.split(",") if x.strip())
    rep = run_rank_forensics(
        corpus,
        k=args.k,
        lolo_lanes=lanes,
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
        out=args.out,
    )
    pop = rep.get("population") or {}
    lolo = rep.get("leave_one_lane_out") or {}
    print(
        f"population={pop.get('n_population')}/{pop.get('n_answerable')} "
        f"lolo={lolo.get('status')} "
        f"baseline mrr@10={(lolo.get('baseline') or {}).get('mrr@10')} "
        f"ndcg@10={(lolo.get('baseline') or {}).get('ndcg@10')}",
        file=sys.stderr,
    )
    return 0


__all__ = [
    "DEFAULT_LOLO_LANES",
    "RRF_K",
    "SCHEMA",
    "decompose_item",
    "forensic_rows",
    "run_rank_forensics",
]


if __name__ == "__main__":
    raise SystemExit(main())
