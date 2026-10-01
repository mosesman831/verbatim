"""CLI entry point for the concrete v3 eval harness.

    python -m eval.v3.run \
        --suite retrieval|grounding|security|tasks|conformance|all \
        --baseline verbatim_v2|verbatim_v3|naive_fts|vector_rag|no_memory|all \
        --out eval/v3/report_v3.md

Options:
    --corpus PATH   JSONL corpus manifest (default: bundled seed corpus)
    --k K           top-k evidence cutoff (default 5)
    --tasks IDS     comma-separated task ids to restrict the run
    --shadow        tasks suite only: shadow mode — arms are recorded,
                    never executed; produces no paired evidence
    --out PATH      markdown report destination (default
                    eval/v3/report_v3.md); use ``-`` for stdout only

Every run uses fresh disposable SQLite stores and the public ingest
path; nothing here touches live state, network, or credentials.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Sequence

from .baselines import BASELINE_NAMES, BASELINES, SuiteRun, probe_capabilities
from .corpus import Corpus, load_corpus, load_seed_corpus
from .report import render_report
from .suite_conformance import run_conformance_suite
from .suite_grounding import run_grounding_suite
from .suite_retrieval import run_retrieval_suite
from .suite_security import run_security_suite
from .suite_tasks import run_tasks_suite

_SUITES = ("retrieval", "grounding", "security", "tasks", "conformance", "all")

_SUITE_RUNNERS = {
    "retrieval": run_retrieval_suite,
    "grounding": run_grounding_suite,
    "security": run_security_suite,
    "tasks": run_tasks_suite,
    "conformance": run_conformance_suite,
}

#: Conformance measures the v3 implementation, not a baseline comparison —
#: it runs once per invocation under this arm label regardless of --baseline.
_CONFORMANCE_ARM = "verbatim_v3"

_DEFAULT_OUT = os.path.join(
    os.path.dirname(__file__), "report_v3.md"
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m eval.v3.run",
        description="Run the v3 evaluation harness against fresh stores.",
    )
    p.add_argument(
        "--suite", required=True, choices=_SUITES,
        help="suite to run (or 'all')",
    )
    p.add_argument(
        "--baseline", default="verbatim_v2",
        help=f"baseline arm: {', '.join(BASELINE_NAMES)} or 'all' "
        "(default verbatim_v2)",
    )
    p.add_argument(
        "--corpus", default=None,
        help="JSONL corpus manifest (default: bundled seed corpus)",
    )
    p.add_argument("--k", type=int, default=5, help="top-k cutoff (default 5)")
    p.add_argument(
        "--tasks", default=None,
        help="comma-separated task ids to restrict the run",
    )
    p.add_argument(
        "--shadow", action="store_true",
        help="tasks suite: shadow mode — record but never execute arms",
    )
    p.add_argument(
        "--out", default=_DEFAULT_OUT,
        help=f"markdown report path (default {_DEFAULT_OUT}; '-' = stdout)",
    )
    return p.parse_args(argv)


def _baselines(name: str) -> List[str]:
    if name == "all":
        return list(BASELINE_NAMES)
    if name not in BASELINES:
        raise SystemExit(
            f"unknown baseline {name!r}; choose from "
            f"{sorted(BASELINE_NAMES)} or 'all'"
        )
    return [name]


def _suite_names(name: str) -> List[str]:
    return list(_SUITE_RUNNERS) if name == "all" else [name]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    corpus: Corpus = (
        load_corpus(args.corpus) if args.corpus else load_seed_corpus()
    )
    task_ids = (
        [t.strip() for t in args.tasks.split(",") if t.strip()]
        if args.tasks else None
    )
    caps = probe_capabilities()

    runs: List[SuiteRun] = []
    t0 = time.perf_counter()
    for suite_name in _suite_names(args.suite):
        runner = _SUITE_RUNNERS[suite_name]
        bases = (
            [_CONFORMANCE_ARM] if suite_name == "conformance"
            else _baselines(args.baseline)
        )
        for base in bases:
            kwargs = {}
            if suite_name == "tasks":
                kwargs["mode"] = "shadow" if args.shadow else "paired"
            run = runner(
                corpus, base, k=args.k, task_ids=task_ids,
                capabilities=caps, **kwargs,
            )
            runs.append(run)
            print(
                f"[eval.v3] {suite_name}/{base}: {len(run.records)} records, "
                f"{run.errors} errors"
                + (
                    f", unavailable lanes: {run.unavailable_lanes}"
                    if run.unavailable_lanes else ""
                ),
                file=sys.stderr,
            )
    elapsed = time.perf_counter() - t0

    text = render_report(runs, corpus, elapsed_s=elapsed)
    if args.out == "-":
        print(text)
    else:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"[eval.v3] report written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
