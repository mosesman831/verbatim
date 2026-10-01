"""V8 evaluation infrastructure (SPEC_V8 §15.1 harness integrity).

Public API:

* :mod:`eval.v8.ledger` — the paired-run keep-decision ledger
  (``ledger.jsonl``, V8-15.04 / V8-00.05 / V8-00.06): ``append_record``,
  ``read_ledger``, ``validate_record``, ``check_chain``.
* :mod:`eval.v8.stats` — paired question-level bootstrap (≥ 10,000
  resamples, seeded, cluster-reported), McNemar's exact test, and the
  V8-00.06 keep-rule evaluator (V8-15.05).
* :mod:`eval.v8.report` — the ``report_v8.md`` renderer: ledger table,
  SB8 scoreboard, gates — ``not_run`` rendered verbatim, numbers never
  shown without a manifest digest (V8-03.01, K03, K19).

Re-exports are lazy (PEP 562) so ``python -m eval.v8.<mod>`` never
double-imports a submodule.
"""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULES = ("ledger", "report", "stats")

#: name → submodule it is re-exported from.
_EXPORTS = {
    # ledger
    "DECISIONS": "ledger",
    "DEFAULT_LEDGER_PATH": "ledger",
    "RECORD_SCHEMA": "ledger",
    "LedgerError": "ledger",
    "append_record": "ledger",
    "category_recall_deltas": "ledger",
    "check_chain": "ledger",
    "delta_any10": "ledger",
    "latest": "ledger",
    "read_ledger": "ledger",
    "record_problems": "ledger",
    "validate_record": "ledger",
    # stats
    "DEFAULT_ALPHA": "stats",
    "DEFAULT_RESAMPLES": "stats",
    "FIX_KINDS": "stats",
    "evaluate_keep_rule": "stats",
    "mcnemar_exact": "stats",
    "mcnemar_from_pairs": "stats",
    "paired_bootstrap": "stats",
    # report
    "NOT_RUN": "report",
    "SCOREBOARD_IDS": "report",
    "SCOREBOARD_STATUSES": "report",
    "UNPINNED": "report",
    "metric_cell": "report",
    "render_comparison": "report",
    "render_gates_table": "report",
    "render_ledger_table": "report",
    "render_report": "report",
    "render_scoreboard": "report",
}

__all__ = sorted(list(_SUBMODULES) + list(_EXPORTS))


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    mod = _EXPORTS.get(name)
    if mod is not None:
        return getattr(importlib.import_module(f".{mod}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
