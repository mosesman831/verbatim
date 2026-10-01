"""Paired keep-rule analysis over ``eval.v8.ablate`` reports (V8-23).

Reads one or more ``ablate_*.json`` reports (shared-store paired runs
whose first spec is the ``v8_all_on`` baseline) and, for every arm:

- per-question any@10 pairs -> paired bootstrap delta + ci_lower
  (cluster unit: conversation id — V8-15.05)
- McNemar exact over discordant any@10 pairs
- per-category any@10 deltas (the V8-00.06 (b) regression axis)
- mrr@10 / all@10 deltas and latency deltas for context
- ``evaluate_keep_rule`` verdict

Off-arms (``no_*`` / ``*_off`` / ``sm0`` / ``dense_N*`` etc.) are
reported in BOTH directions: the arm's delta vs baseline, and the
implied feature-on delta (negated) — the ledger records whichever
direction the decision lands.

Usage::

    python -m eval.v8.analyze_ablate eval/v8/results/ablate_v85_dev_a.json [...]
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from eval.v7 import metrics as M
from eval.v8 import stats as S


def _any10(gold: Any, row: Any) -> Optional[float]:
    delivered = (row or {}).get("delivered") or ()
    v = M.evidence_any_at_k(gold or (), delivered, 10)
    return None if v is None else float(v)


def _metric(spec: Dict[str, Any], key: str) -> Optional[float]:
    v = (spec.get("overall") or {}).get(key)
    return float(v) if isinstance(v, (int, float)) else None


def _cat_delta(specs: Dict[str, Any], base: str, cand: str,
               metric: str = "any@10") -> Dict[str, float]:
    bc = (specs.get(base) or {}).get("categories") or {}
    cc = (specs.get(cand) or {}).get("categories") or {}
    out: Dict[str, float] = {}
    for cat in sorted(set(bc) | set(cc)):
        b = (bc.get(cat) or {}).get(metric)
        c = (cc.get(cat) or {}).get(metric)
        if isinstance(b, (int, float)) and isinstance(c, (int, float)):
            out[cat] = float(c) - float(b)
    return out


def analyze_report(path: str) -> Dict[str, Any]:
    report = json.load(open(path))
    specs: Dict[str, Any] = report.get("specs") or {}
    questions: List[Dict[str, Any]] = report.get("questions") or []
    labels = list(specs)
    if not labels:
        return {"path": path, "error": "no specs", "arms": []}
    base = labels[0]

    # per-question any@10 pairs for every arm vs baseline
    pairs: Dict[str, List[Tuple[float, float]]] = {
        l: [] for l in labels if l != base}
    convs: Dict[str, List[str]] = {l: [] for l in pairs}
    for q in questions:
        if not q.get("answerable"):
            continue
        arms = q.get("arms") or {}
        rb = arms.get(base)
        if not rb or "delivered" not in rb:
            continue
        gb = _any10(q.get("gold"), rb)
        if gb is None:
            continue
        for l in pairs:
            rc = arms.get(l)
            if not rc or "delivered" not in rc:
                continue
            gc = _any10(q.get("gold"), rc)
            if gc is None:
                continue
            pairs[l].append((gb, gc))
            convs[l].append(str(q.get("conv_id") or ""))

    arms_out: List[Dict[str, Any]] = []
    for l in labels:
        if l == base:
            continue
        spec = specs.get(l) or {}
        row: Dict[str, Any] = {
            "label": l,
            "error": spec.get("error"),
            "applied": spec.get("applied"),
            "any10": _metric(spec, "any@10"),
            "all10": _metric(spec, "all@10"),
            "mrr10": _metric(spec, "mrr@10"),
            "lat_p50": ((spec.get("overall") or {}).get("latency_ms")
                        or {}).get("p50"),
            "lat_p95": ((spec.get("overall") or {}).get("latency_ms")
                        or {}).get("p95"),
        }
        prs = pairs.get(l) or []
        if prs:
            boot = S.paired_bootstrap(
                prs, conversation_ids=convs[l], unit="cluster")
            mc = S.mcnemar_from_pairs(prs)
            row.update({
                "n_pairs": len(prs),
                "delta_any10": boot.get("delta"),
                "ci_lower": boot.get("ci_lower"),
                "mcnemar_p": mc.get("p"),
                "disc_base_only": mc.get("base_only"),
                "disc_cand_only": mc.get("candidate_only"),
            })
            cat_d = _cat_delta(specs, base, l)
            row["cat_any10_deltas"] = cat_d
            mrr_d = None
            bm, cm = _metric(specs[base], "mrr@10"), _metric(spec, "mrr@10")
            if bm is not None and cm is not None:
                mrr_d = cm - bm
            row["delta_mrr10"] = mrr_d
            row["keep_rule"] = S.evaluate_keep_rule(
                delta_any10=boot.get("delta"),
                ci_lower=boot.get("ci_lower"),
                category_deltas=cat_d,
            )
        arms_out.append(row)

    b = specs.get(base) or {}
    return {
        "path": path,
        "status": report.get("status"),
        "baseline": {
            "label": base,
            "any10": _metric(b, "any@10"),
            "all10": _metric(b, "all@10"),
            "mrr10": _metric(b, "mrr@10"),
            "lat_p50": ((b.get("overall") or {}).get("latency_ms")
                        or {}).get("p50"),
            "lat_p95": ((b.get("overall") or {}).get("latency_ms")
                        or {}).get("p95"),
        },
        "manifest_digest": _manifest_digest(report),
        "arms": arms_out,
    }


def _manifest_digest(report: Dict[str, Any]) -> Optional[str]:
    import hashlib
    man = report.get("manifest")
    if not man:
        return None
    try:
        from verbatim.core.serialize import json_dumps
        blob = json_dumps(man)
    except Exception:
        blob = json.dumps(man, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    paths = list(argv if argv is not None else sys.argv[1:])
    if not paths:
        print(__doc__)
        return 2
    for p in paths:
        try:
            res = analyze_report(p)
        except Exception as exc:  # noqa: BLE001
            print(f"{p}: FAILED to analyze — {type(exc).__name__}: {exc}")
            continue
        b = res.get("baseline") or {}
        print(f"\n=== {p} ===")
        print(f"baseline {b.get('label')}: any@10={b.get('any10')} "
              f"all@10={b.get('all10')} mrr@10={b.get('mrr10')} "
              f"p50={b.get('lat_p50')}")
        print(f"manifest_digest={res.get('manifest_digest')}")
        for a in res.get("arms") or []:
            if a.get("error"):
                print(f"  {a['label']:<22} ERROR: {a['error']}")
                continue
            d = a.get("delta_any10")
            ci = a.get("ci_lower")
            kr = a.get("keep_rule") or {}
            cats = a.get("cat_any10_deltas") or {}
            worst = min(cats.items(), key=lambda kv: kv[1])[0] \
                if cats else None
            cat_txt = (f"worst_cat={worst}({cats[worst]:+.3f})"
                       if worst else "worst_cat=-")
            print(
                f"  {a['label']:<22} any@10={a.get('any10')} "
                f"Δ={d if d is None else f'{d:+.4f}'} "
                f"ci_lo={ci if ci is None else f'{ci:+.4f}'} "
                f"p={a.get('mcnemar_p')} "
                f"Δmrr={a.get('delta_mrr10')} {cat_txt}")
            print(f"      -> keep_rule decision: {kr.get('decision')} "
                  f"rejecting={kr.get('rejecting_metric')} "
                  f"applied={a.get('applied')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
