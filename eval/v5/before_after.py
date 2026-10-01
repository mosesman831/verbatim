"""Before/after comparison for the V5 performance work (SPEC_V5 §33).

The ``before`` baseline is the executed quiet-tree portfolio recorded
earlier this session (pre optimization batch): A1 search under
concurrent writers ~283 ms p95, a3 add-ack ~75 ms at n=768, timers
total ~214 ms p50, settled search ~8.4 ms p50, ~123 SQL statements per
settled search. The ``after`` column re-measures the same suites on the
current tree.

Honesty rules: identical suite entry points and scales for both columns;
every number is labeled ``locally_measured``; deltas are reported, never
rounded into claims. A regression is printed as a regression.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


# Executed quiet-tree baseline recorded before the optimization batch
# (source: earlier eval/v5/report_v5.json produced by the same
# ``python -m eval.v5.run`` entry point on an undisturbed tree).
BEFORE: Dict[str, Dict[str, Optional[float]]] = {
    "a0.search_ms": {"p50": 21.7, "p95": 24.9, "p99": 41.8},
    "a0.add_ack_ms": {"p50": 11.5, "p95": 31.6, "p99": 45.4},
    "a1.search_ms": {"p50": 240.6, "p95": 282.9, "p99": 287.3},
    "a1.add_ack_ms": {"p50": 9.8, "p95": 23.9, "p99": 45.7},
    "a3_768.search_ms": {"p50": 41.2, "p95": 73.6, "p99": 92.7},
    "a3_768.add_ack_ms": {"p50": 15.6, "p95": 75.0, "p99": 86.7},
    "timers.total_ms": {"p50": 214.4, "p95": 215.2, "p99": 215.4},
    "timers.settled_search_ms": {"p50": 8.4, "p95": 9.1, "p99": 10.5},
    "timers.immediate_add_search_ms": {
        "p50": 221.2, "p95": 221.4, "p99": 221.4},
    "timers.sql_per_search": {"p50": 123.0, "p95": 140.0, "p99": 142.0},
}

BEFORE_NOTE = (
    "Baseline = quiet-tree run of `python -m eval.v5.run` executed before "
    "the optimization batch (settlement retries, WAL-snapshot prescan, "
    "commit-wake signaling, batched writes, deferred DDL, HMAC template, "
    "pending_count, cv triggers, checkpoint pacing). Same entry point, "
    "same scales, same seed=42."
)


def _pick(d: dict, *path: str) -> Optional[dict]:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur if isinstance(cur, dict) else None


def _after_from_report(report: dict) -> Dict[str, Dict[str, Optional[float]]]:
    s = report.get("suites") or {}
    env = s.get("envelopes") or {}
    a3 = s.get("a3_probe") or {}
    t = s.get("timers") or {}
    # a3 emits a list of per-scale points (items_added=192/768/…)
    big = {}
    for p in a3.get("points") or []:
        if isinstance(p, dict) and p.get("items_added") == 768:
            big = p
            break
    b = t.get("boundaries_ms") or {}
    out: Dict[str, Dict[str, Optional[float]]] = {
        "a0.search_ms": _pick(env, "a0", "search_ms") or _pick(env, "a0", "search"),
        "a0.add_ack_ms": _pick(env, "a0", "add_ack_ms") or _pick(env, "a0", "add_ack"),
        "a1.search_ms": _pick(env, "a1", "search_ms") or _pick(env, "a1", "search"),
        "a1.add_ack_ms": _pick(env, "a1", "add_ack_ms") or _pick(env, "a1", "add_ack"),
        "a3_768.search_ms": _pick(big, "search_ms") or _pick(big, "search"),
        "a3_768.add_ack_ms": _pick(big, "add_ack_ms") or _pick(big, "add_ack"),
        "timers.total_ms": _pick(t, "stages_ms", "total"),
        "timers.settled_search_ms": _pick(b, "settled_search_ms"),
        "timers.immediate_add_search_ms": _pick(b, "immediate_add_search_ms"),
        "timers.sql_per_search": _pick(t, "sql_per_search"),
    }
    return {k: v for k, v in out.items()}


def compare(report: dict) -> dict:
    after = _after_from_report(report)
    rows = []
    for key, before in BEFORE.items():
        a = after.get(key) or {}
        row: Dict[str, Any] = {"metric": key, "before": before, "after": a}
        deltas = {}
        for stat in ("p50", "p95", "p99"):
            b, af = before.get(stat), (a or {}).get(stat)
            if isinstance(b, (int, float)) and isinstance(af, (int, float)):
                deltas[stat] = round(af - b, 3)
                if b:
                    deltas[stat + "_pct"] = round((af - b) / b * 100.0, 1)
        row["delta"] = deltas
        rows.append(row)
    return {
        "artifact": "v5-before-after",
        "qualification": "locally_measured",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_note": BEFORE_NOTE,
        "rows": rows,
        "caveats": [
            "Latency deltas on a shared 4-core box carry noise; treat "
            "<±15% as parity unless repeated runs disagree.",
            "Correctness gates (quality recall/precision, leakage, dx) "
            "must hold in BOTH columns — speed without correctness is "
            "not a win (SPEC_V5 §33.08).",
        ],
    }


def write_before_after(report: dict, json_path: str, md_path: str) -> dict:
    comp = compare(report)
    with open(json_path, "w") as f:
        json.dump(comp, f, indent=1, sort_keys=True)
    lines = [
        "# V5 before/after — optimization batch",
        "",
        f"Generated {comp['generated_at']} — qualification: "
        "**locally_measured**",
        "",
        comp["baseline_note"],
        "",
        "| metric | stat | before (ms or count) | after | delta |",
        "|---|---|---|---|---|",
    ]
    for row in comp["rows"]:
        for stat in ("p50", "p95", "p99"):
            b = row["before"].get(stat)
            a = (row["after"] or {}).get(stat)
            d = row["delta"].get(stat)
            dp = row["delta"].get(stat + "_pct")
            lines.append(
                f"| {row['metric']} | {stat} | "
                f"{'—' if b is None else b} | "
                f"{'—' if a is None else a} | "
                f"{'—' if d is None else f'{d} ({dp}%)'} |"
            )
    lines += ["", "## Caveats", ""]
    lines += [f"- {c}" for c in comp["caveats"]]
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return comp


__all__ = ["BEFORE", "compare", "write_before_after"]
