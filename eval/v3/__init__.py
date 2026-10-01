"""Verbatim v3 evaluation harness and P0 measurement artifacts (SPEC_V3 §53–§56).

This subpackage extends the v2 harness additively — ``python -m eval`` and
``eval.harness`` are untouched. Everything here is measurement
infrastructure: it declares what the v3.0 evidence contract requires, runs
suites against a store factory, and reports measured results with
denominators and intervals. It never converts a specification target, a
shadow log, or a single marked test into a claimed result
(V3-54.11, V3-54.13, V3-56.01).

Layout:
    suites.py     — the four §53 suite definitions, case schema, runner
                    protocol, result records, and stdlib-only interval math.
    ledger.py     — requirement→evidence ledger; a requirement completes
                    only when EVERY required evidence kind passes (§56).
    harness.py    — suite runner: executes cases against a store factory
                    and produces the Report dict with denominators, CIs,
                    and first-class ``inconclusive`` results.
    registry.py   — seeds ``registry/ledger.json`` from SPEC_V3.md.
    registry/ledger.json — the frozen V3.0 requirement seed (616 ids).
    f27.py        — F27 historical-retrieval investigation harness: the
                    saved 0.58/0.89 pre/post-rebuild gap, reproduced and
                    attributed per query (V3-03 F27, V3-53.15, B49).
    report_v3.py  — Markdown report generator for v3 suite Reports.

Concrete P0 harness (fresh-store, capability-aware):

    corpus.py     — JSONL corpus schema + bundled ``corpus_seed.jsonl``
                    (~44 synthetic coding-task items incl. 8 poison
                    patterns, benign instructional twins, scope-isolation
                    cases, and 8 runnable paired-task fixtures).
    baselines.py  — ``prepare_case`` (fresh store via the public ingest
                    path) + no_memory / naive_fts / vector_rag /
                    verbatim_v2 / verbatim_v3 adapters + capability probe.
    metrics.py    — pure scoring over ScoredRecord (recall/precision/
                    abstention/grounding/security/paired-delta).
    suite_retrieval.py / suite_grounding.py / suite_security.py /
    suite_tasks.py — the four operational lanes.
    report.py     — measured-vs-target Markdown report.
    run.py        — ``python -m eval.v3.run --suite ... --baseline ...``
"""

from .suites import (
    CONFORMANCE,
    PAIRED_USEFULNESS,
    RETRIEVAL_QUALITY,
    SECURITY_PRIVACY,
    SUITE_IDS,
    Case,
    CaseResult,
    Outcome,
    Suite,
    wilson_interval,
)
from .harness import run_suite
from .ledger import Ledger, load_ledger

__all__ = [
    "CONFORMANCE",
    "PAIRED_USEFULNESS",
    "RETRIEVAL_QUALITY",
    "SECURITY_PRIVACY",
    "SUITE_IDS",
    "Case",
    "CaseResult",
    "Outcome",
    "Suite",
    "wilson_interval",
    "run_suite",
    "Ledger",
    "load_ledger",
]

__version__ = "0.1.0"
