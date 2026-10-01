"""Speaker-match audit — SPEC_V8 V8-11.02 (cat-5 attribution).

Two paired measurements on one ingested store:

1. The arm: ``speaker_match`` boost weight at ``{0, current}`` — the
   current value is read live from
   ``rerank_features.FEATURE_WEIGHTS_V1["speaker_match"]`` (0.4 at
   ``rerank_features.py:91``) and the zero arm is injected through
   ``score_candidates(weights=…)``.  Metrics: answerable any@10 and
   cat-5 premise any@10, paired per question.

2. The hypothesis evidence (V8-11.02 (a)/(b)): for every cat-5 question
   where flat BM25 delivers the premise turn in the top 10 and Verbatim
   does not, the row records

   * ``premise_speaker`` — the gold turn's speaker from the corpus;
   * ``resolved_speaker`` — the query's resolved speaker canon,
     recovered from lane signals on items whose ``speaker_match``
     feature fired (``null`` when unobservable — never guessed);
   * ``speaker_match_source`` — the feature's resolution path
     (hint | query | ambiguous | none) from ``detail``;
   * ``displacers_speaker_match`` — how many displacing items carried
     ``speaker_match == 1.0`` with a positive applied weight;
   * ``both_hold`` — (a) premise speaker differs from resolved speaker
     AND (b) the feature fired on ≥ 1 displacer.

   ``summary.both_hold_fraction`` is the ≥ 50 % decision input — when
   the resolved speaker is unobservable the question counts in ``n``
   but not in ``n_both``/``n_either`` (no guessed values).

Cat-5 identification: ``premise_categories`` (category names) plus
``premise_category_ids`` (LoCoMo numeric ids) — defaults cover the
LoCoMo ``adversarial``/id-5 convention; dict corpora can pass their own.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .. import metrics as M
from ..arms import FlatBM25Arm, arm_task, corpus_items, item_ref
from ._common import (
    ArmSpec,
    PolicyPatch,
    arm_report_kwargs,
    paired_run,
    task_views,
    write_report,
)
from .rank_forensics import decompose_item

SCHEMA = "forensics/speaker_audit-v1"

#: LoCoMo cat-5 name + id (V7-22.07 table).
DEFAULT_PREMISE_CATEGORIES = ("adversarial",)
DEFAULT_PREMISE_IDS = (5,)


def _current_speaker_weight() -> float:
    try:
        from verbatim.retrieval.v7.rerank_features import FEATURE_WEIGHTS_V1

        return float(FEATURE_WEIGHTS_V1.get("speaker_match", 0.0))
    except Exception:  # noqa: BLE001 — honest None beats a guessed weight
        return float("nan")


def _signal_speaker(signals: Mapping[str, Any]) -> Optional[str]:
    """Best-effort ``speaker_canon`` recovery from per-lane signal dicts."""
    for lane, sig in (signals or {}).items():
        if isinstance(sig, Mapping) and sig.get("speaker_canon") is not None:
            return str(sig["speaker_canon"])
    return None


def _resolved_speaker(decomp_items: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """The query's resolved speaker canon = the ``speaker_canon`` signal
    shared by items whose ``speaker_match`` feature fired at 1.0."""
    canon: Optional[str] = None
    for d in decomp_items:
        feats = d.get("features") or {}
        if feats.get("speaker_match") == 1.0:
            s = _signal_speaker(d.get("signals") or {})
            if s is not None:
                return s
            if canon is not None:
                return canon
    return canon


def _premise_diag(
    tv: Any,
    explain: Optional[Mapping[str, Any]],
    item_speaker: Mapping[str, Optional[str]],
    source_ref: Mapping[str, str],
    lane_weights: Mapping[str, float],
) -> Dict[str, Any]:
    """V8-11.02 (a)/(b) evidence for one cat-5 question."""
    gold = set(tv.gold_item)
    premise_speakers = sorted(
        {item_speaker.get(g) for g in gold if item_speaker.get(g)}
    )
    items = (explain or {}).get("items") or []
    decomp = [
        decompose_item(
            it,
            ref=source_ref.get(str(it.get("source_id"))),
            lane_weights=lane_weights,
        )
        for it in items
    ]
    gold_idx = next(
        (i for i, d in enumerate(decomp)
         if d["ref"] is not None and str(d["ref"]) in gold),
        None,
    )
    displacers = [
        d for d in decomp[: gold_idx if gold_idx is not None else len(decomp)]
        if not (d["ref"] is not None and str(d["ref"]) in gold)
    ]
    fired = [
        d for d in displacers
        if (d.get("features") or {}).get("speaker_match") == 1.0
        and (d.get("feature_weights") or {}).get("speaker_match", 0.0) > 0
    ]
    resolved = _resolved_speaker(decomp)
    src = next(
        (
            ((it.get("detail") or {}).get("speaker_match_source"))
            for it in items
            if (it.get("detail") or {}).get("speaker_match_source") is not None
        ),
        None,
    )
    differs = (
        bool(premise_speakers) and resolved is not None
        and resolved not in premise_speakers
    )
    both = differs and bool(fired)
    return {
        "premise_speaker": premise_speakers or None,
        "resolved_speaker": resolved,
        "speaker_match_source": src,
        "premise_differs": differs if (premise_speakers and resolved) else None,
        "n_displacers": len(displacers),
        "displacers_speaker_match": len(fired),
        "displacer_refs": [d["ref"] or d["unit_id"] for d in displacers],
        "both_hold": both if (premise_speakers and resolved) else None,
        "explain_status": "ok" if explain else "unavailable",
    }


def run_speaker_audit(
    corpus: Any,
    *,
    k: int = 10,
    weights: Sequence[float] = (0.0,),
    premise_categories: Sequence[str] = DEFAULT_PREMISE_CATEGORIES,
    premise_category_ids: Sequence[int] = DEFAULT_PREMISE_IDS,
    arm_kwargs: Optional[Mapping[str, Any]] = None,
    policy_overrides: Optional[Mapping[str, Any]] = None,
    policy_doc: Optional[Mapping[str, Any]] = None,
    out: Optional[str] = None,
) -> Dict[str, Any]:
    """V8-11.02 measurement: weight arm {0, current} + premise evidence.

    ``weights`` are the *extra* arm values to measure alongside the
    current default (spec: ``{0, current}`` — pass ``(0.0,)``; a
    ``half`` arm is ``(0.5 * current,)`` from the caller)."""
    views = task_views(corpus)
    cur = _current_speaker_weight()
    cats = {str(c) for c in premise_categories}
    cat_ids = {int(i) for i in premise_category_ids}

    from ._common import category_id_of

    def is_premise(tv: Any) -> bool:
        if tv.category in cats:
            return True
        try:
            cid = category_id_of(tv.raw)
            return cid is not None and int(cid) in cat_ids
        except (TypeError, ValueError):
            return False

    specs: List[ArmSpec] = [
        ArmSpec(
            label="w_current",
            patch=None,
            notes=[f"speaker_match weight = current default ({cur})"],
        )
    ]
    for w in weights:
        specs.append(ArmSpec(
            label=f"w_{w:g}",
            patch=PolicyPatch(
                feature_weights={"speaker_match": float(w)},
                policy_doc=policy_doc,
            ),
            notes=[f"speaker_match weight overridden to {w:g} via "
                   "score_candidates(weights=…)"],
        ))

    sink: Dict[str, Any] = {}
    report = paired_run(
        corpus,
        specs,
        k_list=(k, 10) if k != 10 else (k,),
        arm_kwargs=arm_kwargs,
        policy_overrides=policy_overrides,
        full_explain=True,
        tool="speaker_audit",
        extra_manifest={
            "requirement": "V8-11.02",
            "speaker_match_weight_current": cur,
            "weights_measured": [cur, *[float(w) for w in weights]],
            "premise_categories": list(premise_categories),
            "premise_category_ids": list(premise_category_ids),
            "base_policy_doc": policy_doc,
        },
        sink=sink,
    )
    report["schema"] = SCHEMA
    if report.get("status") == "not_run":
        write_report(report, out)
        return report

    # ---- premise-category aggregates per arm -----------------------------
    qrows = report["questions"]
    for label, spec in report["specs"].items():
        vals = [
            M.evidence_any_at_k(
                q["gold"], (q["arms"][label] or {}).get("delivered") or (), 10
            )
            for q in qrows
            if q["answerable"] and q["gold"] and is_premise_t(q, cats, cat_ids)
        ]
        vals = [v for v in vals if v is not None]
        spec["premise_any@10"] = (
            sum(vals) / len(vals) if vals else None
        )
        spec["premise_n"] = len(vals)

    # ---- (a)/(b) evidence on the current arm ------------------------------
    bm = FlatBM25Arm()
    bm.ingest(corpus)
    item_speaker: Dict[str, Optional[str]] = {}
    for i, it in enumerate(corpus_items(corpus)):
        sp = None
        if isinstance(it, Mapping):
            sp = it.get("speaker")
        else:
            sp = getattr(it, "speaker", None)
        item_speaker[item_ref(it, i)] = str(sp) if sp is not None else None
    source_ref = sink.get("source_ref") or {}
    explains = (sink.get("explains") or {}).get("w_current") or {}
    try:
        evidence: List[Dict[str, Any]] = []
        from .rank_forensics import _lane_weights

        for tv in views:
            if not tv.answerable or not tv.gold_item or not is_premise(tv):
                continue
            gold = set(tv.gold_item)
            bm_out = bm.query(arm_task(tv), k)
            bm_hit = any(str(r) in gold for r in bm_out.refs[:k])
            row = next(
                (q for q in qrows if q["task_id"] == tv.task_id), {}
            )
            vrow = (row.get("arms") or {}).get("w_current") or {}
            v_hit = bool(vrow.get("first_gold_rank"))
            if not (bm_hit and not v_hit):
                continue
            intent = (
                ((explains.get(tv.task_id) or {}).get("query") or {})
                .get("intent")
            )
            evidence.append({
                "task_id": tv.task_id,
                "conv_id": tv.group_id,
                "category": tv.category,
                "bm25_premise_rank": next(
                    (i + 1 for i, r in enumerate(bm_out.refs)
                     if str(r) in gold),
                    None,
                ),
                "verbatim_gold_rank": vrow.get("first_gold_rank"),
                **_premise_diag(
                    tv,
                    explains.get(tv.task_id),
                    item_speaker,
                    source_ref,
                    _lane_weights(intent, policy_doc),
                ),
            })
    finally:
        bm.close()

    n = len(evidence)
    n_eval = sum(1 for e in evidence if e["both_hold"] is not None)
    n_both = sum(1 for e in evidence if e["both_hold"])
    report["premise_evidence"] = {
        "definition": "cat-5 questions where bm25 delivers the premise "
                      "turn top-10 and verbatim does not",
        "n": n,
        "n_evaluable": n_eval,
        "n_both_hold": n_both,
        "both_hold_fraction": (n_both / n_eval) if n_eval else None,
        "questions": evidence,
    }
    write_report(report, out)
    return report


def is_premise_t(qrow: Mapping[str, Any], cats: set, cat_ids: set) -> bool:
    """Paired-question-row form of the cat-5 predicate."""
    if qrow.get("category") in cats:
        return True
    cid = qrow.get("category_id")
    try:
        return cid is not None and int(cid) in cat_ids
    except (TypeError, ValueError):
        return False


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m eval.v7.forensics.speaker_audit",
        description="V8-11.02 speaker-match weight audit",
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--weights", default="0",
                    help="comma-separated extra weight values "
                         "(current default is always measured)")
    ap.add_argument("--premise-categories", default=",".join(
        DEFAULT_PREMISE_CATEGORIES))
    ap.add_argument("--premise-ids", default=",".join(
        str(i) for i in DEFAULT_PREMISE_IDS))
    ap.add_argument("--timeout-ms", type=float, default=None)
    ap.add_argument("--pool-limit", type=int, default=None)
    ap.add_argument("--settle-timeout", type=float, default=None)
    ap.add_argument("--policy-overrides", default=None)
    ap.add_argument("--policy-doc", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from ..corpora import load_corpus

    corpus = load_corpus(args.dataset, args.split)
    rep = run_speaker_audit(
        corpus,
        k=args.k,
        weights=tuple(
            float(x) for x in args.weights.split(",") if x.strip()
        ),
        premise_categories=tuple(
            x.strip() for x in args.premise_categories.split(",") if x.strip()
        ),
        premise_category_ids=tuple(
            int(x) for x in args.premise_ids.split(",") if x.strip()
        ),
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
    for label, spec in (rep.get("specs") or {}).items():
        ov = spec.get("overall") or {}
        print(
            f"{label:>12s} applied={spec.get('applied')} "
            f"any@10={ov.get('any@10')} "
            f"premise_any@10={spec.get('premise_any@10')} "
            f"n={spec.get('premise_n')}",
            file=sys.stderr,
        )
    pe = rep.get("premise_evidence") or {}
    print(
        f"premise evidence n={pe.get('n')} both_hold={pe.get('n_both_hold')}/"
        f"{pe.get('n_evaluable')} frac={pe.get('both_hold_fraction')}",
        file=sys.stderr,
    )
    return 0


__all__ = [
    "DEFAULT_PREMISE_CATEGORIES",
    "DEFAULT_PREMISE_IDS",
    "SCHEMA",
    "run_speaker_audit",
]


if __name__ == "__main__":
    raise SystemExit(main())
