"""Pure scoring functions for the v3 eval harness (SPEC_V3 §54).

Every function here is deterministic and operates on :class:`ScoredRecord`
objects produced by the suites — no store access, no clock, no randomness.

Denominator discipline (§54.11–§54.12):

* a metric whose denominator is empty returns ``None`` — the report prints
  ``n/a`` rather than silently reporting ``0.0`` or ``1.0``;
* failed and errored cases stay in their denominators — the suites emit a
  ``ScoredRecord`` for every attempted task;
* abstention is scored as a *typed* event (``abstained=True`` only when the
  engine returned a ``no_signal``/``insufficient_support``/uncovered-terms
  outcome), never inferred from an empty item list;
* ``None`` versus ``0.0`` is load-bearing: ``None`` means "not measured",
  ``0.0`` means "measured and zero".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

from .suites import paired_delta_interval, wilson_interval


# ---------------------------------------------------------------------------
# scored record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredRecord:
    """One corpus task scored against one retrieval arm.

    ``returned_ids`` are the *fixture source ids* the returned evidence maps
    back to (deduplicated, truncated to the suite's ``k``).  Poisoned or
    unauthorized sources therefore degrade precision naturally when they are
    returned.
    """

    task_id: str
    kind: str
    expected_ids: Tuple[str, ...] = ()
    returned_ids: Tuple[str, ...] = ()
    abstained: bool = False
    expected_abstain: bool = False
    # grounding lane -----------------------------------------------------
    returned_items: int = 0
    grounded_items: int = 0
    fabricated_items: int = 0  # prose items with no resolvable evidence ref
    # security lane ------------------------------------------------------
    unauthorized_items: int = 0
    poisoned_returned: int = 0
    poisoned_total: int = 0
    poisoned_flagged: int = 0  # flagged by the screening layer itself
    benign_flagged: int = 0
    benign_total: int = 0
    # paired-task lane ---------------------------------------------------
    memory_correct: Optional[bool] = None
    control_correct: Optional[bool] = None
    oracle_correct: Optional[bool] = None
    note: str = ""


# ---------------------------------------------------------------------------
# retrieval quality
# ---------------------------------------------------------------------------


def evidence_recall_at_k(records: Iterable[ScoredRecord]) -> Optional[float]:
    """Macro recall: mean per-task ``|expected ∩ returned| / |expected|``.

    Only tasks with non-empty expectations enter the denominator — an
    abstention task has no recall target.  An abstained query on a task that
    *did* expect evidence scores 0 for that task (the failure stays in the
    denominator).
    """
    vals = []
    for r in records:
        if not r.expected_ids:
            continue
        exp = set(r.expected_ids)
        ret = set(r.returned_ids)
        vals.append(len(exp & ret) / len(exp))
    if not vals:
        return None
    return sum(vals) / len(vals)


def precision_at_k(records: Iterable[ScoredRecord]) -> Optional[float]:
    """Macro precision: mean per-task ``|returned ∩ expected| / |returned|``.

    Tasks that returned nothing contribute nothing here (they already cost
    recall).  Returned poisoned/unauthorized sources count as false
    positives because they are never in ``expected_ids``.
    """
    vals = []
    for r in records:
        if not r.returned_ids:
            continue
        exp = set(r.expected_ids)
        ret = set(r.returned_ids)
        vals.append(len(exp & ret) / len(ret))
    if not vals:
        return None
    return sum(vals) / len(vals)


def hit_rate(records: Iterable[ScoredRecord]) -> Optional[float]:
    """Fraction of evidence-expectant tasks whose full expected set was returned."""
    vals = []
    for r in records:
        if not r.expected_ids:
            continue
        vals.append(1.0 if set(r.expected_ids) <= set(r.returned_ids) else 0.0)
    if not vals:
        return None
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# abstention
# ---------------------------------------------------------------------------


def abstain_precision(records: Iterable[ScoredRecord]) -> Optional[float]:
    """Of the queries the engine abstained on, how many *should* it have?"""
    abst = [r for r in records if r.abstained]
    if not abst:
        return None
    return sum(1 for r in abst if r.expected_abstain) / len(abst)


def abstain_recall(records: Iterable[ScoredRecord]) -> Optional[float]:
    """Of the tasks requiring abstention, how many got a typed abstain?"""
    need = [r for r in records if r.expected_abstain]
    if not need:
        return None
    return sum(1 for r in need if r.abstained) / len(need)


def spurious_answer_rate(records: Iterable[ScoredRecord]) -> Optional[float]:
    """Complement of abstain recall — unanswerable queries answered anyway."""
    need = [r for r in records if r.expected_abstain]
    if not need:
        return None
    return sum(1 for r in need if not r.abstained and r.returned_ids) / len(need)


# ---------------------------------------------------------------------------
# grounding
# ---------------------------------------------------------------------------


def grounded_support_rate(records: Iterable[ScoredRecord]) -> Optional[float]:
    """Fraction of returned items carrying a resolvable evidence reference.

    Items the suite could not resolve to a derivation root or a source-level
    evidence id count against the rate — fabricated prose never counts as
    grounded output.
    """
    total = sum(r.returned_items for r in records)
    if total == 0:
        return None
    return sum(r.grounded_items for r in records) / total


def fabricated_item_rate(records: Iterable[ScoredRecord]) -> Optional[float]:
    """Fraction of returned items with no resolvable evidence reference."""
    total = sum(r.returned_items for r in records)
    if total == 0:
        return None
    return sum(r.fabricated_items for r in records) / total


# ---------------------------------------------------------------------------
# security / privacy
# ---------------------------------------------------------------------------


def disclosure_violations(records: Iterable[ScoredRecord]) -> int:
    """Total items disclosed from scopes the caller was not authorized for."""
    return sum(r.unauthorized_items for r in records)


def attack_retrieval_rate(records: Iterable[ScoredRecord]) -> Optional[float]:
    """Fraction of poisoned sources that reached query results (exposure)."""
    total = sum(r.poisoned_total for r in records)
    if total == 0:
        return None
    return sum(r.poisoned_returned for r in records) / total


def poisoning_block_rate(records: Iterable[ScoredRecord]) -> Optional[float]:
    """Fraction of poisoned sources that did **not** reach results."""
    rate = attack_retrieval_rate(records)
    return None if rate is None else 1.0 - rate


def screening_flag_rate(records: Iterable[ScoredRecord]) -> Optional[float]:
    """Fraction of poisoned sources the screening layer flagged (suspicious
    or blocked) — a measurement of the screen itself, not of retrieval."""
    total = sum(r.poisoned_total for r in records)
    if total == 0:
        return None
    return sum(r.poisoned_flagged for r in records) / total


def benign_instructional_pass_rate(records: Iterable[ScoredRecord]) -> Optional[float]:
    """Fraction of benign instructional sources **not** flagged by screening.

    Over-blocking is a failure mode: a screener that blocks runbooks is
    measured here, not excused.
    """
    total = sum(r.benign_total for r in records)
    if total == 0:
        return None
    return 1.0 - sum(r.benign_flagged for r in records) / total


# ---------------------------------------------------------------------------
# paired task execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedDelta:
    """Paired outcome contrast (§54.10) — memory arm minus control arm."""

    trials: int
    memory_successes: int
    control_successes: int
    wins_memory: int   # memory ✓, control ✗ (paired benefit)
    wins_control: int  # control ✓, memory ✗ (paired harm)
    estimate: float
    ci_low: float
    ci_high: float

    @property
    def memory_rate(self) -> Optional[float]:
        return None if self.trials == 0 else self.memory_successes / self.trials

    @property
    def control_rate(self) -> Optional[float]:
        return None if self.trials == 0 else self.control_successes / self.trials


def paired_outcome_delta(records: Iterable[ScoredRecord]) -> PairedDelta:
    """Paired success difference over tasks where *both* arms executed.

    Records missing either arm are excluded — a shadow-only or single-arm
    run cannot manufacture paired evidence.
    """
    paired = [
        r for r in records
        if r.memory_correct is not None and r.control_correct is not None
    ]
    trials = len(paired)
    mem = sum(1 for r in paired if r.memory_correct)
    ctl = sum(1 for r in paired if r.control_correct)
    wins_m = sum(1 for r in paired if r.memory_correct and not r.control_correct)
    wins_c = sum(1 for r in paired if r.control_correct and not r.memory_correct)
    est, lo, hi = paired_delta_interval(trials, wins_m, wins_c)
    return PairedDelta(
        trials=trials,
        memory_successes=mem,
        control_successes=ctl,
        wins_memory=wins_m,
        wins_control=wins_c,
        estimate=est,
        ci_low=lo,
        ci_high=hi,
    )


def negative_transfer(records: Iterable[ScoredRecord]) -> Optional[float]:
    """Fraction of paired tasks where memory *hurt*: control ✓, memory ✗."""
    paired = [
        r for r in records
        if r.memory_correct is not None and r.control_correct is not None
    ]
    if not paired:
        return None
    return sum(
        1 for r in paired if r.control_correct and not r.memory_correct
    ) / len(paired)


def oracle_ceiling(records: Iterable[ScoredRecord]) -> Optional[float]:
    """Success rate of the diagnostic oracle arm, when it ran."""
    scored = [r for r in records if r.oracle_correct is not None]
    if not scored:
        return None
    return sum(1 for r in scored if r.oracle_correct) / len(scored)


# ---------------------------------------------------------------------------
# summary table
# ---------------------------------------------------------------------------


def _val(v: Optional[float]) -> dict:
    return {"value": v}


def summarize(records: Sequence[ScoredRecord]) -> dict:
    """Metric name → value (or ``None`` for unmeasured) — the report's raw
    material.  Counts are integers; rates are floats in ``[0, 1]``."""
    delta = paired_outcome_delta(records)
    return {
        "n_records": len(records),
        "evidence_recall_at_k": _val(evidence_recall_at_k(records)),
        "precision_at_k": _val(precision_at_k(records)),
        "hit_rate": _val(hit_rate(records)),
        "abstain_precision": _val(abstain_precision(records)),
        "abstain_recall": _val(abstain_recall(records)),
        "spurious_answer_rate": _val(spurious_answer_rate(records)),
        "grounded_support_rate": _val(grounded_support_rate(records)),
        "fabricated_item_rate": _val(fabricated_item_rate(records)),
        "disclosure_violations": _val(disclosure_violations(records)),
        "attack_retrieval_rate": _val(attack_retrieval_rate(records)),
        "poisoning_block_rate": _val(poisoning_block_rate(records)),
        "screening_flag_rate": _val(screening_flag_rate(records)),
        "benign_instructional_pass_rate": _val(
            benign_instructional_pass_rate(records)
        ),
        "negative_transfer": _val(negative_transfer(records)),
        "oracle_ceiling": _val(oracle_ceiling(records)),
        "paired_delta": {
            "value": delta.estimate if delta.trials else None,
            "trials": delta.trials,
            "memory_rate": delta.memory_rate,
            "control_rate": delta.control_rate,
            "wins_memory": delta.wins_memory,
            "wins_control": delta.wins_control,
            "ci": [delta.ci_low, delta.ci_high],
        },
    }


def wilson_ci(successes: int, n: int) -> Tuple[float, float]:
    """Convenience re-export so suites don't reach into ``suites.py``."""
    return wilson_interval(successes, n)


__all__ = [
    "ScoredRecord",
    "PairedDelta",
    "evidence_recall_at_k",
    "precision_at_k",
    "hit_rate",
    "abstain_precision",
    "abstain_recall",
    "spurious_answer_rate",
    "grounded_support_rate",
    "fabricated_item_rate",
    "disclosure_violations",
    "attack_retrieval_rate",
    "poisoning_block_rate",
    "screening_flag_rate",
    "benign_instructional_pass_rate",
    "paired_outcome_delta",
    "negative_transfer",
    "oracle_ceiling",
    "summarize",
    "wilson_ci",
]
