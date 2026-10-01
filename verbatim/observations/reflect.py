"""Bounded reflection over consolidated memory (SPEC_V4 §24, §27).

``reflect`` performs a *read-only* pass over the observation plane:
it scans slot-aggregation inputs for **rival slots** — one
(subject, predicate, perspective, condition) asserted with ≥2 distinct
values by independent evidence families — and emits *labeled
hypothesis* observations naming the open conflict.

Contract anchors:

- V4-24.06 / §27 reflection boundary (C50): this module performs NO host
  work. There is no tool dispatch, no shell, no filesystem or connector
  access, no callback hook — the inputs are allowlisted read-only memory
  reads (claims/family rows through ``fetch_claim_inputs``), and the
  outputs are labeled derived observations. Nothing here can modify
  grants, trust scores, receipts, or canonical source bytes.
- V4-27.06 iteration bound: ``ReflectBudget.max_iterations`` defaults to
  3 — the spec's default three-retrieval-round ceiling — and the pass
  reports ``stopped='iteration_cap'`` + ``regions_remaining`` honestly
  rather than running on.
- V4-24.06 hypotheses are *derived labeled views*: each hypothesis lands
  in ``observations`` with ``producer='v4.reflect.v1'``, a ``hypothesis:``
  text marker, ``observation_evidence`` rows naming the exact claim
  revisions behind it, and ``derivations`` edges to all of them — a
  purge or correction to any input propagates through the real graph.
- V4-24.04: rival values are never merged — a hypothesis names the
  conflict and keeps every side's evidence; it does not pick a winner.
- V4-24.08: hypothesis ids are deterministic over (scope, slot), so an
  unchanged conflict re-derives the same observation and writes nothing.
- V4-24.09: the pass stops early and says why (``no_material_change`` /
  ``iteration_cap`` / ``output_cap`` / ``deadline``) — utility-floor and
  duty-cycle bounds are first-class outcomes, not silent truncation.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from ..core.types import ErrorCode, VerbatimError, require_id
from .aggregate import (
    ELIGIBLE_MODALITIES,
    ELIGIBLE_STATES,
    ClaimInput,
    _merged_freshness,
    fetch_claim_inputs,
    persist_observation,
    value_key_for,
)
from .consolidate import _retire_uncovered
from .freshness import next_seq

#: Producer identity for reflection outputs (V3-17.02). Every hypothesis
#: carries it on the ``observations.producer`` column and on its
#: derivations edges.
REFLECT_PRODUCER_KIND = "producer"
REFLECT_PRODUCER_ID = "v4.reflect.v1"

#: V4-27.06 default: at most three retrieval rounds per reflection.
_DEFAULT_MAX_ITERATIONS = 3
_DEFAULT_MAX_REGIONS = 256
_DEFAULT_MAX_OUTPUTS = 64


@dataclass(frozen=True)
class ReflectBudget:
    """Hard bounds for one reflection pass (V4-27.05/06).

    - ``max_iterations``: retrieval/processing rounds, default 3 —
      the spec's default ceiling; raising it is an explicit caller
      decision recorded on the report.
    - ``max_regions``: rival slots considered in total (input bound).
    - ``max_outputs``: hypothesis observations written per *round*;
      total writes never exceed ``max_iterations × max_outputs``.
    - ``deadline_s``: optional wall-clock seconds for the whole pass,
      checked against the injectable ``clock`` at each round boundary
      (§27 deadline accounting; default ``time.monotonic``).
    """

    max_iterations: int = _DEFAULT_MAX_ITERATIONS
    max_regions: int = _DEFAULT_MAX_REGIONS
    max_outputs: int = _DEFAULT_MAX_OUTPUTS
    deadline_s: Optional[float] = None


@dataclass
class ReflectReport:
    """Honest pass outcome — every bound that bound is named."""

    iterations: int = 0
    regions_seen: int = 0
    regions_remaining: int = 0
    hypotheses_written: int = 0
    edges_written: int = 0
    retired: list[str] = field(default_factory=list)
    stopped: Optional[str] = None
    observation_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "regions_seen": self.regions_seen,
            "regions_remaining": self.regions_remaining,
            "hypotheses_written": self.hypotheses_written,
            "edges_written": self.edges_written,
            "retired": list(self.retired),
            "stopped": self.stopped,
            "observation_ids": list(self.observation_ids),
        }


def hypothesis_id_for(
    scope_id: str,
    subject_id: str,
    predicate: str,
    perspective_id: Optional[str],
    condition_key: str,
) -> str:
    """Deterministic hypothesis observation id for one rival slot."""
    raw = "\x00".join(
        [
            "v4:reflect",
            scope_id,
            subject_id,
            predicate,
            perspective_id or "",
            condition_key or "",
        ]
    )
    return "obs_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _slot_key_of(ci: ClaimInput) -> tuple[str, str, str, str]:
    return (
        ci.subject_id,
        ci.predicate,
        ci.perspective_id or "",
        ci.condition_key,
    )


def _rival_slots(
    inputs: list[ClaimInput], *, min_rival_values: int
) -> list[tuple[tuple[str, str, str, str], list[ClaimInput]]]:
    """Slots asserting ≥``min_rival_values`` distinct values, ordered by
    family coverage desc then key asc — high-value regions first (§24
    trigger table: "changed or high-value regions only")."""
    slots: dict[tuple[str, str, str, str], list[ClaimInput]] = {}
    for ci in inputs:
        slots.setdefault(_slot_key_of(ci), []).append(ci)
    rivals: list[tuple[tuple[str, str, str, str], list[ClaimInput]]] = []
    for key, members in slots.items():
        values = {value_key_for(m.polarity, m.value_text) for m in members}
        if len(values) >= min_rival_values:
            rivals.append((key, members))
    rivals.sort(
        key=lambda kv: (
            -len(
                {
                    m.family_id if m.family_id else f"claim:{m.claim_id}"
                    for m in kv[1]
                }
            ),
            kv[0],
        )
    )
    return rivals


def _hypothesis_text(
    subject_id: str, predicate: str, members: list[ClaimInput]
) -> str:
    """Render the labeled hypothesis — structured slot values only,
    never source bytes (V3-17.04, V4-24.04)."""
    counts: dict[str, tuple[str, int]] = {}
    fams: set[str] = set()
    for m in members:
        vkey = value_key_for(m.polarity, m.value_text)
        fam = m.family_id if m.family_id else f"claim:{m.claim_id}"
        fams.add(fam)
        text, n = counts.get(vkey, (m.value_text, 0))
        counts[vkey] = (text, n + 1)
    sides = sorted(counts.values(), key=lambda t: (-t[1], t[0]))
    shown = " vs ".join(
        f"'{t}' ×{n}" for t, n in sides[:4]
    )
    if len(sides) > 4:
        shown += f" (+{len(sides) - 4} more)"
    return (
        f"hypothesis: {subject_id} {predicate} contested — {shown}; "
        f"{len(fams)} independent "
        f"{'family' if len(fams) == 1 else 'families'}; unresolved"
    )


def reflect(
    conn: sqlite3.Connection,
    scope_id: str,
    *,
    budget: ReflectBudget = ReflectBudget(),
    enabled: bool = True,
    min_rival_values: int = 2,
    states: Iterable[str] = ELIGIBLE_STATES,
    modalities: Iterable[str] = ELIGIBLE_MODALITIES,
    seq: Optional[int] = None,
    clock: Callable[[], float] = time.monotonic,
) -> ReflectReport:
    """One bounded reflection pass over ``scope_id``.

    Read-only over the memory plane; writes only labeled derived
    observations. Returns a :class:`ReflectReport` naming every bound
    that bound. ``enabled=False`` fails CAPABILITY_UNAVAILABLE — a
    disabled producer never writes silently (V3-23.09).
    """
    require_id(scope_id, "scope_id")
    if not enabled:
        raise VerbatimError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            "reflection is disabled for this scope",
        )
    for name, val in (
        ("max_iterations", budget.max_iterations),
        ("max_regions", budget.max_regions),
        ("max_outputs", budget.max_outputs),
    ):
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"budget.{name} must be a positive int"
            )
    if min_rival_values < 2:
        raise VerbatimError(
            ErrorCode.VALIDATION, "min_rival_values must be >= 2"
        )
    if seq is None:
        seq = next_seq(conn, scope_id)

    inputs = fetch_claim_inputs(
        conn, scope_id, states=states, modalities=modalities
    )
    rivals = _rival_slots(inputs, min_rival_values=min_rival_values)
    report = ReflectReport(regions_seen=len(rivals))
    if len(rivals) > budget.max_regions:
        rivals = rivals[: budget.max_regions]
        report.stopped = "region_cap"

    if not rivals:
        report.stopped = report.stopped or "no_material_change"
        report.regions_remaining = 0
        # Even an empty pass retires hypotheses whose slots resolved.
        report.retired = _retire_uncovered(
            conn, scope_id, REFLECT_PRODUCER_ID, set(), seq
        )
        return report

    start = clock()
    covered: set[str] = set()
    pending = list(rivals)
    for _ in range(budget.max_iterations):
        if not pending:
            break
        report.iterations += 1
        if budget.deadline_s is not None and clock() - start > budget.deadline_s:
            report.stopped = "deadline"
            break
        round_slice, pending = (
            pending[: budget.max_outputs],
            pending[budget.max_outputs :],
        )
        for key, members in round_slice:
            subject, predicate, persp, cond_key = key
            obs_id = hypothesis_id_for(
                scope_id, subject, predicate, persp or None, cond_key
            )
            fams = {
                m.family_id if m.family_id else f"claim:{m.claim_id}"
                for m in members
            }
            supports = sorted(
                {("claim", m.claim_id, m.revision) for m in members}
            )
            res = persist_observation(
                conn,
                scope_id=scope_id,
                observation_id=obs_id,
                text=_hypothesis_text(subject, predicate, members),
                proof_count=len(fams),
                perspective_id=persp or None,
                freshness=_merged_freshness(members),
                producer=REFLECT_PRODUCER_ID,
                supports=supports,
                contradicts=(),
                seq=seq,
            )
            covered.add(obs_id)
            if res.wrote:
                report.hypotheses_written += 1
                report.edges_written += res.edges_written
            report.observation_ids.append(obs_id)
    report.regions_remaining = len(pending)
    if pending and report.stopped is None:
        # Iterations exhausted with regions still queued (a deadline stop
        # is already named).
        report.stopped = "iteration_cap"
    # A bounded pass retires hypotheses whose slots resolved ONLY when it
    # saw every region — partial coverage can't judge the unseen.
    if not pending and report.stopped != "region_cap":
        report.retired = _retire_uncovered(
            conn, scope_id, REFLECT_PRODUCER_ID, covered, seq
        )
    report.stopped = report.stopped or (
        None if report.hypotheses_written or report.retired
        else "no_material_change"
    )
    return report


__all__ = [
    "REFLECT_PRODUCER_KIND",
    "REFLECT_PRODUCER_ID",
    "ReflectBudget",
    "ReflectReport",
    "hypothesis_id_for",
    "reflect",
]
