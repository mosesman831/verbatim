"""Entity-resolution fixture harness (SPEC_V7 V7-08.13, SPEC_V7_5 §05 Q4).

Runs the current resolver — ``verbatim.enrichment.entities_v2``'s
``extract_mentions`` + ``propose_aliases`` (``alias/v1``) — over the owned
fixture produced by ``make_er_fixture.py`` and reports the Q4 gate
metrics:

* **over-merge rate** — fraction of ``no_merge``/``abstain`` pairs where
  the resolver emitted an ACTIVE alias row linking the two referents'
  canons.  The gate is ≤ 0.01 on the same-name strata (≥ 100 pairs).
* **correct-merge recall** — fraction of ``merge`` pairs the resolver
  actively linked (same-canon folds count as merges — canon identity is
  the merge).
* **abstention rate** — fraction of all pairs resolved to a ``candidate``
  link (the documented abstain path).

Each pair is evaluated in an isolated scope: its two context snippets
(plus optional ``extra_units`` scaffolding) are the only units, its
``known_canons`` the only vocabulary.  This mirrors how
``units_jobs._write_aliases`` invokes the resolver (mentions + entity
vocabulary + raw texts + caller rows), one decision at a time.

Outcome vocabulary per pair:

* ``merge``       — an ACTIVE row links a canon in ``canons_a`` to one in
                    ``canons_b`` (either direction).
* ``abstain``     — no active link, but a CANDIDATE row links them.
* ``separate``    — no row links them at all.
* ``same_canon``  — ``canons_a`` and ``canons_b`` already coincide; the
                    canon layer has conflated (or unified) them, so the
                    alias layer has no decision to make.

This is measurement machinery only — it never edits the resolver.
"""

from __future__ import annotations

import json
import os
import sys
import types
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from verbatim.core.types_v7 import AliasState
from verbatim.enrichment.entities_v2 import (
    ALIAS_RULES_VERSION,
    EXTRACTOR_ID,
    canon,
    extract_mentions,
    propose_aliases,
)

FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixtures",
    "entity_resolution_owned.jsonl",
)

#: Strata whose pairs are "same-name (different-person)" for the V7-08.13
#: gate denominator — every ``sn_*`` stratum qualifies.
SAME_NAME_PREFIX = "sn_"


def load_fixture(path: str = FIXTURE_PATH) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _mentions_for(text: str, unit_id: str, speaker: Optional[str] = None):
    """extract_mentions over raw unit text — the jobs layer substitutes
    the raw text into ``norm.text``; a SimpleNamespace carries the same
    attribute contract here."""
    return extract_mentions(types.SimpleNamespace(text=text), unit_id, speaker)


def decide_pair(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Run the resolver over one fixture pair and classify the outcome."""
    mentions = []
    texts: List[str] = []
    pid = rec["pair_id"]
    units = [
        {"text": rec["context_a"], "speaker": rec.get("speaker_a")},
        {"text": rec["context_b"], "speaker": rec.get("speaker_b")},
    ]
    units.extend(rec.get("extra_units") or [])
    for i, u in enumerate(units):
        text = u["text"]
        texts.append(text)
        mentions.extend(_mentions_for(text, f"{pid}:u{i}", u.get("speaker")))

    rows = propose_aliases(
        mentions,
        rec.get("known_canons") or (),
        texts=texts,
        scope_id="er-eval",
        caller_aliases=[
            tuple(p) for p in (rec.get("caller_aliases") or [])
        ] or None,
    )

    set_a = {canon(c) for c in rec["canons_a"]}
    set_b = {canon(c) for c in rec["canons_b"]}
    links, other_rows = [], []
    for r in rows:
        link = (r.canon in set_a and r.alias_canon in set_b) or (
            r.canon in set_b and r.alias_canon in set_a
        )
        entry = {
            "canon": r.canon,
            "alias_canon": r.alias_canon,
            "rule_id": r.rule_id,
            "state": r.state.value if isinstance(r.state, AliasState) else str(r.state),
        }
        (links if link else other_rows).append(entry)

    if set_a & set_b:
        outcome = "same_canon"
    elif any(l["state"] == AliasState.ACTIVE.value for l in links):
        outcome = "merge"
    elif links:
        outcome = "abstain"
    else:
        outcome = "separate"

    return {
        "pair_id": pid,
        "stratum": rec["stratum"],
        "expected": rec["expected"],
        "outcome": outcome,
        "links": links,
        "other_rows": other_rows,
        "n_mentions": len(mentions),
        "mention_canons": sorted({m.canon for m in mentions}),
    }


def evaluate(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Full fixture report: gate metrics overall and per stratum."""
    details = [decide_pair(r) for r in records]

    def _is_same_name(d) -> bool:
        return d["stratum"].startswith(SAME_NAME_PREFIX)

    n_total = len(details)
    n_merge_exp = sum(1 for d in details if d["expected"] == "merge")
    n_nomerge_exp = sum(1 for d in details if d["expected"] == "no_merge")
    n_abstain_exp = sum(1 for d in details if d["expected"] == "abstain")

    merges_ok = sum(
        1 for d in details
        if d["expected"] == "merge" and d["outcome"] in ("merge", "same_canon")
    )
    merges_ok_canon = sum(
        1 for d in details
        if d["expected"] == "merge" and d["outcome"] == "same_canon"
    )
    abstained_miss = sum(
        1 for d in details
        if d["expected"] == "merge" and d["outcome"] == "abstain"
    )
    silent_miss = sum(
        1 for d in details
        if d["expected"] == "merge" and d["outcome"] == "separate"
    )

    # over-merge: an ACTIVE link on a pair that must not merge.
    neg = [d for d in details if d["expected"] in ("no_merge", "abstain")]
    over = [d for d in neg if d["outcome"] == "merge"]
    conflated = [d for d in neg if d["outcome"] == "same_canon"]
    neg_abstained = [d for d in neg if d["outcome"] == "abstain"]
    neg_separate = [d for d in neg if d["outcome"] == "separate"]

    sn_neg = [d for d in neg if _is_same_name(d)]
    sn_over = [d for d in over if _is_same_name(d)]
    sn_conflated = [d for d in conflated if _is_same_name(d)]

    abstain_outcomes = sum(1 for d in details if d["outcome"] == "abstain")

    per_stratum: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "expected": Counter(), "outcome": Counter(),
                 "over_merge": 0, "merged_ok": 0}
    )
    for d in details:
        st = per_stratum[d["stratum"]]
        st["n"] += 1
        st["expected"][d["expected"]] += 1
        st["outcome"][d["outcome"]] += 1
        if d["expected"] in ("no_merge", "abstain") and d["outcome"] == "merge":
            st["over_merge"] += 1
        if d["expected"] == "merge" and d["outcome"] in ("merge", "same_canon"):
            st["merged_ok"] += 1

    return {
        "resolver": {
            "extractor_id": EXTRACTOR_ID,
            "alias_rules": ALIAS_RULES_VERSION,
        },
        "n_pairs": n_total,
        "expected_counts": {
            "merge": n_merge_exp,
            "no_merge": n_nomerge_exp,
            "abstain": n_abstain_exp,
        },
        "outcome_counts": dict(Counter(d["outcome"] for d in details)),
        # Q4 gate numbers
        "over_merge_rate": (len(over) / len(neg)) if neg else None,
        "over_merge_n": len(over),
        "over_merge_denominator": len(neg),
        "over_merge_rate_no_merge_only": (
            sum(1 for d in details
                if d["expected"] == "no_merge" and d["outcome"] == "merge")
            / n_nomerge_exp if n_nomerge_exp else None
        ),
        "same_name_over_merge_rate": (
            len(sn_over) / len(sn_neg)) if sn_neg else None,
        "same_name_n": len(sn_neg),
        "same_name_over_merge_n": len(sn_over),
        # canon-level conflation on negative pairs is reported separately:
        # identical surfaces are already one canon before the alias layer
        # runs — arguably an over-merge upstream of the resolver.
        "canon_conflation_n": len(conflated),
        "same_name_canon_conflation_n": len(sn_conflated),
        "same_name_over_merge_rate_incl_conflation": (
            (len(sn_over) + len(sn_conflated)) / len(sn_neg)
            if sn_neg else None
        ),
        "merge_recall": (merges_ok / n_merge_exp) if n_merge_exp else None,
        "merge_recall_n": merges_ok,
        "merge_recall_via_same_canon_n": merges_ok_canon,
        "merge_denominator": n_merge_exp,
        "abstained_miss_n": abstained_miss,
        "silent_miss_n": silent_miss,
        "abstain_rate": (abstain_outcomes / n_total) if n_total else None,
        "negative_pairs": {
            "n": len(neg),
            "over_merged": len(over),
            "abstained": len(neg_abstained),
            "separated": len(neg_separate),
            "canon_conflated": len(conflated),
        },
        "per_stratum": {
            s: {
                "n": v["n"],
                "expected": dict(v["expected"]),
                "outcome": dict(v["outcome"]),
                "over_merge": v["over_merge"],
                "merged_ok": v["merged_ok"],
            }
            for s, v in sorted(per_stratum.items())
        },
        "over_merge_pairs": [
            {"pair_id": d["pair_id"], "stratum": d["stratum"],
             "links": d["links"]} for d in over
        ],
        "details": details,
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixture", default=FIXTURE_PATH)
    ap.add_argument("--json", action="store_true",
                    help="print the full JSON report (default: summary)")
    ap.add_argument("--out", default=None, help="write JSON report here")
    args = ap.parse_args(argv)

    recs = load_fixture(args.fixture)
    rep = evaluate(recs)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2, sort_keys=True)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 0

    print(f"fixture pairs:            {rep['n_pairs']}")
    print(f"resolver:                 {rep['resolver']['alias_rules']} "
          f"({rep['resolver']['extractor_id']})")
    print(f"expected:                 {rep['expected_counts']}")
    print(f"outcomes:                 {rep['outcome_counts']}")
    print(f"over-merge rate:          {rep['over_merge_rate']:.4f} "
          f"({rep['over_merge_n']}/{rep['over_merge_denominator']})")
    print(f"  same-name subset:       {rep['same_name_over_merge_rate']:.4f} "
          f"({rep['same_name_over_merge_n']}/{rep['same_name_n']})")
    print(f"  + canon conflation:     "
          f"{rep['same_name_over_merge_rate_incl_conflation']:.4f} "
          f"(incl {rep['same_name_canon_conflation_n']} same-canon pairs)")
    print(f"merge recall:             {rep['merge_recall']:.4f} "
          f"({rep['merge_recall_n']}/{rep['merge_denominator']}, "
          f"{rep['merge_recall_via_same_canon_n']} via same-canon)")
    print(f"  abstained misses:       {rep['abstained_miss_n']}")
    print(f"  silent misses:          {rep['silent_miss_n']}")
    print(f"abstain rate:             {rep['abstain_rate']:.4f}")
    print("\nper-stratum:")
    for s, v in rep["per_stratum"].items():
        print(f"  {s:32s} n={v['n']:3d} expected={v['expected']} "
              f"outcome={v['outcome']}")
    if rep["over_merge_pairs"]:
        print("\nover-merge pairs:")
        for p in rep["over_merge_pairs"]:
            print(f"  {p['pair_id']:16s} {p['stratum']:26s} "
                  f"{[(l['canon'], l['alias_canon'], l['rule_id']) for l in p['links']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
