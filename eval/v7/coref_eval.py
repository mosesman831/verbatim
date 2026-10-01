"""Coreference fixture harness (SPEC_V7 V7-13.20, SPEC_V7_5 §05 Q8).

Runs ``verbatim.enrichment.coref_sieve`` (``coref_sieve/v1``) over the
owned fixture produced by ``make_coref_fixture.py`` and reports the Q8
gate metrics:

* **precision** — of the mentions the arm resolved, the fraction whose
  predicted canon equals the expected antecedent.  Resolving when the
  correct behavior is abstention (traps, unsupported forms, out-of-window
  decoys) counts as a wrong resolution.  Gate: >= 0.95 (V7-13.20).
* **recall** — of the mentions whose expected label is a canon, the
  fraction resolved correctly.  Published, never gated.
* **abstention rate** — fraction of all mentions the arm declined.

Two arms are measured, mirroring the Q8 comparison row:

* ``sieve``         — the shipped ``coref_sieve/v1`` (lookback N=6).
* ``previous_turn`` — the R0 baseline retained for Q8:
                      ``last_resort_previous_turn=True`` restricts
                      candidates to the immediately previous turn and
                      excludes same-unit mentions.

This is measurement machinery only — it never edits the sieve and never
adjusts labels.  ``python -m eval.v7.coref_eval`` prints the report;
``--json`` emits the metrics dict; ``--failures`` lists every case where
prediction and label disagree.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from verbatim.enrichment import coref_sieve
from eval.v7.make_coref_fixture import (
    ABSTAIN,
    FIXTURE_PATH,
    LOOKBACK,
    iter_jsonl,
)

ARMS: Tuple[str, ...] = ("sieve", "previous_turn")


def load_cases(path: str = FIXTURE_PATH) -> List[Dict[str, Any]]:
    """Load case records (the ``fixture`` header line is skipped)."""
    return [r for r in iter_jsonl(path) if r.get("record") != "fixture"]


def predict(case: Dict[str, Any], arm: str = "sieve"
            ) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """One sieve call → (canon|None, status, detail)."""
    ctx = dict(case.get("context") or {})
    out = coref_sieve.explain(
        case["mention"], case["unit_index"], case["turns"],
        lookback=LOOKBACK, **ctx,
        last_resort_previous_turn=(arm == "previous_turn"))
    return out["canon"], out["status"], out["detail"]


def _blank() -> Dict[str, int]:
    return {"total": 0, "resolved": 0, "correct": 0, "wrong": 0,
            "abstained": 0, "correct_abstain": 0, "missed": 0,
            "expected_resolvable": 0, "expected_abstain": 0}


def _finish(c: Dict[str, int]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(c)
    out["precision"] = (c["correct"] / c["resolved"]) if c["resolved"] \
        else None
    out["recall"] = (c["correct"] / c["expected_resolvable"]) \
        if c["expected_resolvable"] else None
    out["abstain_rate"] = (c["abstained"] / c["total"]) if c["total"] \
        else None
    return out


def evaluate(cases: Sequence[Dict[str, Any]], arm: str = "sieve"
             ) -> Dict[str, Any]:
    """Score one arm over the cases; returns counters + rates."""
    assert arm in ARMS, arm
    tot = _blank()
    per_stratum: Dict[str, Dict[str, int]] = {}
    statuses: Counter = Counter()
    errors: List[Dict[str, Any]] = []
    for case in cases:
        pred, status, detail = predict(case, arm)
        statuses[status] += 1
        exp = case["expected"]
        resolvable = exp != ABSTAIN
        buckets = [tot] + [per_stratum.setdefault(t, _blank())
                           for t in case.get("strata") or ()]
        for bkt in buckets:
            bkt["total"] += 1
            bkt["expected_resolvable" if resolvable
                else "expected_abstain"] += 1
            if pred is None:
                bkt["abstained"] += 1
                if resolvable:
                    bkt["missed"] += 1
                else:
                    bkt["correct_abstain"] += 1
            else:
                bkt["resolved"] += 1
                if resolvable and pred == exp:
                    bkt["correct"] += 1
                else:
                    bkt["wrong"] += 1
        if pred is not None and not (resolvable and pred == exp):
            errors.append({
                "case_id": case["case_id"], "stratum": case["stratum"],
                "mention": case["mention"], "expected": exp,
                "predicted": pred, "status": status,
                "intended": case.get("intended_antecedent"),
            })
    return {
        "arm": arm,
        "sieve": coref_sieve.SIEVE_ID,
        "lookback": LOOKBACK,
        **_finish(tot),
        "per_stratum": {t: _finish(c) for t, c in sorted(per_stratum.items())},
        "status_histogram": dict(statuses.most_common()),
        "errors": errors,
    }


def run(path: str = FIXTURE_PATH,
        arms: Sequence[str] = ARMS) -> Dict[str, Any]:
    """Load the fixture and evaluate every arm; returns the report dict."""
    cases = load_cases(path)
    return {
        "fixture": os.path.abspath(path),
        "cases": len(cases),
        "arms": {arm: evaluate(cases, arm) for arm in arms},
    }


def _fmt_rate(x: Optional[float]) -> str:
    return "  n/a " if x is None else f"{x:6.3f}"


def format_report(result: Dict[str, Any],
                  show_errors: bool = True) -> str:
    lines: List[str] = []
    lines.append(f"coref fixture: {result['fixture']} "
                 f"({result['cases']} cases)")
    for arm, m in result["arms"].items():
        lines.append("")
        lines.append(f"ARM {arm} ({m['sieve']}, lookback={m['lookback']})")
        lines.append(
            f"  total {m['total']} | resolved {m['resolved']} "
            f"(correct {m['correct']}, wrong {m['wrong']}) "
            f"| abstained {m['abstained']} "
            f"(correct {m['correct_abstain']}, missed {m['missed']})")
        lines.append(
            f"  precision {_fmt_rate(m['precision'])}  "
            f"recall {_fmt_rate(m['recall'])}  "
            f"abstain {_fmt_rate(m['abstain_rate'])}   "
            f"[gate: precision >= 0.95, V7-13.20]")
        lines.append("  per-stratum:")
        lines.append("    stratum                 n   res  ok   bad  "
                     " abst  prec   recall")
        for tag, c in m["per_stratum"].items():
            lines.append(
                f"    {tag:24s} {c['total']:3d} {c['resolved']:4d} "
                f"{c['correct']:4d} {c['wrong']:4d} {c['abstained']:5d}  "
                f"{_fmt_rate(c['precision'])} {_fmt_rate(c['recall'])}")
        if show_errors and m["errors"]:
            lines.append(f"  disagreements ({len(m['errors'])}):")
            for e in m["errors"]:
                lines.append(
                    f"    {e['case_id']} [{e['stratum']}] "
                    f"{e['mention']!r} -> {e['predicted']!r} "
                    f"(expected {e['expected']!r}, status={e['status']})")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    fixture = FIXTURE_PATH
    if "--fixture" in argv:
        i = argv.index("--fixture")
        fixture = argv[i + 1]
    arms = ARMS
    if "--arm" in argv:
        i = argv.index("--arm")
        arms = (argv[i + 1],)
    result = run(fixture, arms)
    if "--json" in argv:
        print(json.dumps(result, sort_keys=True, indent=2))
    else:
        # disagreements are the point of the report — always shown unless
        # explicitly silenced.
        print(format_report(result, show_errors="--no-failures" not in argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
