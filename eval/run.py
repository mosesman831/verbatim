"""CLI entry point for the evaluation harness.

Usage::

    python -m eval.run --corpus-size 1000 --out eval/
    python -m eval --corpus-size 200 --seed 7 --no-rebuild

Prints a short summary to stdout and writes ``report.md`` + ``report.json``
into the output directory.
"""

from __future__ import annotations

import argparse
import sys

from .corpus import (
    corpus_stats,
    generate_corpus,
    generate_realistic_chat_corpus,
)
from .harness import run_harness
from .report import render_markdown, write_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="eval",
        description="Verbatim v2 synthetic-corpus evaluation harness",
    )
    p.add_argument("--corpus-size", type=int, default=1000,
                   help="target statement count (default 1000)")
    p.add_argument("--seed", type=int, default=42,
                   help="corpus RNG seed (default 42)")
    p.add_argument("--realistic-chat", action="store_true",
                   help="run the fixed punctuation-light chat regression slice")
    p.add_argument("--out", default="eval",
                   help="output directory for report.md/report.json")
    p.add_argument("--no-approve", action="store_true",
                   help="do not drive operator-assisted admit transitions")
    p.add_argument("--no-rebuild", action="store_true",
                   help="do not rebuild the FTS projection before scoring")
    p.add_argument("--query-limit", type=int, default=None,
                   help="cap number of queries executed")
    p.add_argument("--top-k", type=int, default=5,
                   help="recall limit per query (default 5)")
    p.add_argument("--work-dir", default=None,
                   help="reuse a directory for the store (default: fresh tmpdir)")
    p.add_argument("--print-md", action="store_true",
                   help="also print the full markdown report to stdout")
    args = p.parse_args(argv)

    corpus = (
        generate_realistic_chat_corpus()
        if args.realistic_chat
        else generate_corpus(size=args.corpus_size, seed=args.seed)
    )
    stats = corpus_stats(corpus)
    print(
        f"corpus: {stats['statements']} statements, "
        f"{sum(stats['queries'].values())} queries "
        f"(sha256 {stats['sha256'][:16]}…, seed {stats['seed']})"
    )

    res = run_harness(
        corpus,
        work_dir=args.work_dir,
        approve=not args.no_approve,
        rebuild_fts=not args.no_rebuild,
        query_limit=args.query_limit,
        top_k=args.top_k,
    )
    md_path, json_path = write_report(res, args.out)

    rec = res.recall
    cov = res.coverage
    print(f"extraction coverage: {cov.get('extraction_coverage')}")
    for kind in ("point", "current", "historical", "no_answer"):
        d = rec.get(kind)
        if d:
            extra = ""
            if kind == "no_answer":
                extra = f" fp_rate={d.get('false_positive_rate')}"
            print(
                f"  {kind}: n={d.get('n')} hits={d.get('hits')} "
                f"hit_rate={d.get('hit_rate')}{extra}"
            )
    lat = rec.get("latency_ms") or {}
    print(
        f"latency ms: p50={lat.get('p50')} p95={lat.get('p95')} "
        f"p99={lat.get('p99')}"
    )
    if res.defects:
        print("defects observed:")
        for d in res.defects:
            print(f"  - {d}")
    print(f"wrote {md_path} and {json_path}")
    if args.print_md:
        print()
        print(render_markdown(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
