"""Verbatim v2 evaluation harness and owned synthetic corpus.

This package is measurement infrastructure. It generates a reproducible,
fully synthetic corpus (no external datasets), drives the public
``verbatim.api.Engine`` pipeline end to end inside a disposable store, and
reports measured results — never specification targets dressed up as
measurements.

Layout:
    corpus.py   — deterministic corpus generator (pure stdlib, no verbatim
                  imports; the corpus itself must not depend on the engine).
    harness.py  — ingestion/processing/recall driver and metric scoring.
    report.py   — Markdown/JSON report rendering.
    run.py      — CLI entry point (``python -m eval.run`` / ``python -m eval``).
"""

__all__ = ["corpus", "harness", "report", "run"]

__version__ = "0.1.0"
