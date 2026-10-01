#!/usr/bin/env python3
"""Merge per-unit AMB shard outputs into one locomo10 result file.

Each shard writes ``/tmp/amb-shards/<unit>/outputs/locomo/verbatim/rag/
locomo10.json`` with a ``{"summary": …, "results": [...]}`` payload.
This concatenates the results arrays (unit-scoped query ids are unique)
and recomputes the aggregate summary honestly — never averages averages.
"""
import glob
import json
import sys
from collections import defaultdict


def main() -> int:
    OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/amb-shards"
    DEST = f"{OUT}/merged_locomo10.json"
    results = []
    summaries = []
    for path in sorted(glob.glob(f"{OUT}/*/outputs/locomo/verbatim/rag/locomo10.json")):
        d = json.load(open(path))
        results.extend(d.get("results") or [])
        summaries.append(d.get("summary") or {})
        print(f"{path}: {len(d.get('results') or [])} results")
    if not results:
        print("no shard results found", file=sys.stderr)
        return 1
    n = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    lat = sorted(
        float(r.get("retrieve_time_ms") or 0) for r in results
    )
    by_cat: dict = defaultdict(lambda: [0, 0])
    for r in results:
        cat = (r.get("meta") or {}).get("category") or "?"
        by_cat[cat][0] += 1
        by_cat[cat][1] += 1 if r.get("correct") else 0
    merged = {
        "dataset": "locomo",
        "split": "locomo10",
        "memory_provider": "verbatim",
        "mode": "rag",
        "answer_llm": (summaries[0] or {}).get("answer_llm"),
        "judge_llm": (summaries[0] or {}).get("judge_llm"),
        "total_queries": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "query_errors": sum((s or {}).get("query_errors", 0) for s in summaries),
        "avg_retrieve_time_ms": sum(lat) / n if n else 0.0,
        "p50_retrieve_ms": lat[n // 2] if n else 0.0,
        "avg_context_tokens": sum(
            float(r.get("context_tokens") or 0) for r in results
        ) / n if n else 0.0,
        "by_category": {
            c: {"n": m, "acc": k / m} for c, (m, k) in sorted(by_cat.items())
        },
        "shards": len(summaries),
    }
    with open(DEST, "w") as fh:
        json.dump({"summary": merged, "results": results}, fh, indent=2)
    print(f"\nmerged {n} queries from {len(summaries)} shards -> {DEST}")
    print(f"accuracy {merged['accuracy']:.4f} ({correct}/{n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
