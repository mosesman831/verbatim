"""V8 forensic measurement tools — ``eval/v7/forensics``.

Reusable, manifest-recording measurements required by SPEC_V8 wave 0:

* :mod:`.sql_census` — V8-14.02 statement census via the SQLite trace
  callback (per-query ``{statements, by_shape}``).
* :mod:`.lane_ablation` — V8-05.01 paired lane on/off runs (lane removed
  from the policy tuple — the §02.3-correct mechanism).
* :mod:`.rank_forensics` — V8-11.01 gold-vs-displacer score
  decomposition + leave-one-lane-out MRR@10/nDCG@10 table.
* :mod:`.speaker_audit` — V8-11.02 ``speaker_match`` weight arm
  {0, current} + cat-5 premise (a)/(b) evidence.
* :mod:`.lane_miss` — V8-07.01 first-loss classification of every
  answerable ``lane_miss`` question (classes (a)–(g), scenario K37).

Shared machinery (``_common``): :class:`PolicyPatch` (the verified
override seam), :class:`ForensicVerbatimArm` (census/explain/policy-lane
capture on the real ``Memory`` path), and :func:`paired_run` (one
ingested store → paired per-question rows consumable by paired
statistics).  Every tool emits the ``forensics/v8-a`` envelope:
``manifest`` records exactly which arm/config produced each number, and
``not_run`` records what could not be measured — no fabricated metrics.
"""

from ._common import (
    ArmSpec,
    ForensicVerbatimArm,
    PolicyPatch,
    paired_run,
)
from .sql_census import SqlCensus, census_search, normalize_shape

__all__ = [
    "ArmSpec",
    "ForensicVerbatimArm",
    "PolicyPatch",
    "SqlCensus",
    "census_search",
    "classify_lane_miss",
    "normalize_shape",
    "paired_run",
    "run_lane_miss_forensic",
]

_LAZY = {
    "classify_lane_miss": ".lane_miss",
    "run_lane_miss_forensic": ".lane_miss",
}


def __getattr__(name: str):
    """PEP 562 lazy export for :mod:`.lane_miss` — importing it eagerly
    here makes ``python -m eval.v7.forensics.lane_miss`` emit a runpy
    double-import warning (the package pulls the module into sys.modules
    before runpy executes it)."""
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(mod, __name__), name)


def __dir__():
    return sorted(__all__)
