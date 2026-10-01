"""Consolidation passes over a scope (SPEC_V3 §23, SPEC_V4 §24).

``consolidate`` runs a full deterministic pass: aggregate eligible claim
revisions into slot candidates, write an observation per candidate that
meets the family-distinct proof threshold, link rival values as
``contradicts`` evidence, and record ``derivations`` edges to every input
claim revision. Observations whose support dropped below the threshold —
or whose slot vanished entirely (source fact deleted/superseded,
V3-23.07) — are invalidated prospectively via ``recorded_until`` +
``stale_since_seq``, never silently rewritten (V3-17.03, V3-23.04).

``consolidate_windowed`` is the V4-24.01 trigger-scoped form: instead of
an unconditional whole-scope scan, the caller supplies a
``ConsolidationWindow`` naming the changed region (``since_seq`` —
revision rows opened or closed after that event seq — and/or explicit
``claim_ids``), plus ``max_inputs``/``max_outputs``/``expected_generation``
bounds. Only the touched slots re-derive, and only observations whose
evidence intersects the touched claims are eligible for retirement — an
untouched slot's observations are left alone, so a bounded pass can never
retire a belief it did not recompute (V4-24.08/24.09: bounded work, no
churn, honest ``stopped`` reporting when caps bind).

Both forms are idempotent by construction: a pass over unchanged inputs
finds the same deterministic observation ids carrying the same evidence
sets and writes nothing — a new revision appears only when the support
set actually changed, and ``PersistResult.added``/``removed`` records
exactly which inputs moved (the durable "why", V4-24.07).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Tuple

from ..core.types import ErrorCode, VerbatimError, json_dumps, require_id, safe_json_loads
from ..storage import repos_v3
from .aggregate import (
    DEFAULT_MIN_PROOF,
    ELIGIBLE_MODALITIES,
    ELIGIBLE_STATES,
    SLOT_AGGREGATE_V1,
    ClaimInput,
    SlotCandidate,
    aggregate_candidates,
    fetch_claim_inputs,
    persist_observation,
)
from .freshness import next_seq


def _retire_uncovered(
    conn: sqlite3.Connection,
    scope_id: str,
    producer: str,
    covered: set[str],
    seq: int,
    *,
    touched_claim_ids: Optional[set[str]] = None,
) -> list[str]:
    """Invalidate producer observations no candidate re-derived (V3-23.07).

    A source fact's deletion or supersession removes its slot support;
    the dependent observation is closed at ``seq`` and marked stale so
    readers see an invalidated belief, never a phantom current one.

    When ``touched_claim_ids`` is given (windowed pass), only
    observations citing a touched claim may retire — a bounded pass never
    invalidates a belief outside its trigger region (V4-24.01).
    """
    retired: list[str] = []
    for row in repos_v3.query(
        conn, "observations", {"scope_id": scope_id, "producer": producer}
    ):
        if row["observation_id"] in covered or row["recorded_until"] is not None:
            continue
        if touched_claim_ids is not None:
            ev = repos_v3.query(
                conn,
                "observation_evidence",
                {
                    "observation_id": row["observation_id"],
                    "revision": int(row["revision"]),
                    "object_kind": "claim",
                },
            )
            if not any(e["object_id"] in touched_claim_ids for e in ev):
                continue
        repos_v3.update(
            conn,
            "observations",
            {"recorded_until": seq, "stale_since_seq": seq},
            {"observation_id": row["observation_id"]},
        )
        retired.append(row["observation_id"])
    return retired


def consolidate(
    conn: sqlite3.Connection,
    scope_id: str,
    *,
    producer: str = SLOT_AGGREGATE_V1,
    min_proof: int = DEFAULT_MIN_PROOF,
    enabled: bool = True,
    states: Iterable[str] = ELIGIBLE_STATES,
    modalities: Iterable[str] = ELIGIBLE_MODALITIES,
    seq: Optional[int] = None,
) -> dict[str, int]:
    """Full consolidation pass for one scope.

    Returns ``{"observations_written", "edges_written"}`` — counts of
    observation rows inserted or revision-bumped, and derivations edges
    appended. ``min_proof`` is the family-distinct support threshold
    (default 2, V3-23.01: consolidated from *multiple* facts *across
    distinct evidence families*). ``enabled=False`` maps the
    ``observations.enabled`` gate to a loud capability failure
    (V3-23.01/23.09) — a disabled producer never writes silently.
    """
    require_id(scope_id, "scope_id")
    require_id(producer, "producer")
    if not enabled:
        raise VerbatimError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            "observations are disabled for this scope",
        )
    if min_proof < 1:
        raise VerbatimError(ErrorCode.VALIDATION, "min_proof must be >= 1")
    if seq is None:
        seq = next_seq(conn, scope_id)

    written = 0
    edges = 0
    covered: set[str] = set()
    for cand in aggregate_candidates(
        conn, scope_id, producer=producer, states=states, modalities=modalities
    ):
        if cand.proof_count < min_proof:
            continue
        covered.add(cand.observation_id)
        res = persist_observation(
            conn,
            scope_id=scope_id,
            observation_id=cand.observation_id,
            text=cand.text,
            proof_count=cand.proof_count,
            perspective_id=cand.perspective_id,
            freshness=cand.freshness,
            producer=producer,
            supports=cand.supports,
            contradicts=cand.contradicts,
            seq=seq,
        )
        written += int(res.wrote)
        edges += res.edges_written

    _retire_uncovered(conn, scope_id, producer, covered, seq)
    return {"observations_written": written, "edges_written": edges}


# ---------------------------------------------------------------------------
# windowed / incremental pass (V4-24.01/24.08/24.09)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsolidationWindow:
    """The declared trigger region of a bounded consolidation pass.

    - ``since_seq``: claim revisions with ``recorded_from`` or
      ``recorded_until`` above this event seq touched their slot.
    - ``claim_ids``: explicit claims whose slots are touched.
    - ``max_inputs`` / ``max_outputs``: hard bounds on claims read and
      observations written this pass (V4-24.09 duty-cycle bound).
    - ``expected_generation``: caller-observed index/job generation —
      when the pass is launched through a durable job, the handler fences
      on it (stale triggers fail STALE_PROPOSAL, never write from an
      abandoned view).
    - ``retention``: a declared retention tag recorded in the report —
      the lifecycle service owns actual erasure; the tag makes the
      trigger auditable (§24 trigger table: "retention event").
    """

    since_seq: Optional[int] = None
    claim_ids: Tuple[str, ...] = ()
    max_inputs: Optional[int] = None
    max_outputs: Optional[int] = None
    expected_generation: Optional[int] = None
    retention: Optional[str] = None


@dataclass
class ConsolidationReport:
    """Honest outcome of a windowed pass (V4-24.09)."""

    observations_written: int = 0
    edges_written: int = 0
    retired: list[str] = field(default_factory=list)
    slots_touched: int = 0
    candidates_seen: int = 0
    truncated: bool = False
    stopped: Optional[str] = None
    refinements: list[dict[str, Any]] = field(default_factory=list)
    window: Optional[ConsolidationWindow] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "observations_written": self.observations_written,
            "edges_written": self.edges_written,
            "retired": list(self.retired),
            "slots_touched": self.slots_touched,
            "candidates_seen": self.candidates_seen,
            "truncated": self.truncated,
            "stopped": self.stopped,
            "refinements": list(self.refinements),
        }


def _slot_key(ci: ClaimInput) -> tuple[str, str, str, str]:
    return (
        ci.subject_id,
        ci.predicate,
        ci.perspective_id or "",
        ci.condition_key,
    )


def _revision_slot_key(
    subject: str, predicate: str, perspective: Optional[str], cond_json: Any
) -> tuple[str, str, str, str]:
    cond = safe_json_loads(cond_json) if cond_json else None
    cond_key = json_dumps(cond) if cond is not None else ""
    return (subject, predicate, perspective or "", cond_key)


def _touched_slots(
    conn: sqlite3.Connection,
    scope_id: str,
    window: ConsolidationWindow,
) -> tuple[set[tuple[str, str, str, str]], set[str]]:
    """(touched slot keys, triggering claim ids) for the window.

    A revision OPENED or CLOSED after ``since_seq`` touches the slot its
    own revision carried — a correction that moves a claim between
    conditions touches both the old revision's slot and the new one's
    (the close and the open are distinct revision rows). Explicit
    ``claim_ids`` touch their current slot.
    """
    slots: set[tuple[str, str, str, str]] = set()
    triggers: set[str] = set()
    params: list[Any] = [scope_id]
    conds: list[str] = []
    if window.since_seq is not None:
        conds.append("(r.recorded_from > ? OR r.recorded_until > ?)")
        params.extend([window.since_seq, window.since_seq])
    if window.claim_ids:
        ph = ",".join("?" for _ in window.claim_ids)
        conds.append(f"(c.claim_id IN ({ph}) AND r.recorded_until IS NULL)")
        params.extend(window.claim_ids)
    if not conds:
        return slots, triggers
    where = "c.scope_id = ? AND (" + " OR ".join(conds) + ")"
    cur = conn.execute(
        "SELECT c.claim_id, c.subject_id, c.predicate,"
        "       r.perspective_id, r.condition_json"
        " FROM claims c JOIN claim_revisions r ON r.claim_id = c.claim_id"
        f" WHERE {where}"
        "   AND c.predicate IS NOT NULL AND c.subject_id IS NOT NULL",
        tuple(params),
    )
    for cid, subj, pred, persp, cond in cur.fetchall():
        slots.add(_revision_slot_key(subj, pred, persp, cond))
        triggers.add(cid)
    return slots, triggers


def consolidate_windowed(
    conn: sqlite3.Connection,
    scope_id: str,
    *,
    window: ConsolidationWindow,
    producer: str = SLOT_AGGREGATE_V1,
    min_proof: int = DEFAULT_MIN_PROOF,
    enabled: bool = True,
    states: Iterable[str] = ELIGIBLE_STATES,
    modalities: Iterable[str] = ELIGIBLE_MODALITIES,
    seq: Optional[int] = None,
) -> ConsolidationReport:
    """Trigger-scoped, bounded consolidation pass (V4-24.01).

    Only the slots the window touches are re-derived; only observations
    citing a touched claim may retire. ``max_inputs`` bounds the claims
    scanned, ``max_outputs`` the observations written — when either binds,
    the report says ``truncated``/``stopped`` honestly and *retirement is
    deferred* (a partial pass cannot know which uncovered observations
    are truly stale).
    """
    require_id(scope_id, "scope_id")
    require_id(producer, "producer")
    if not enabled:
        raise VerbatimError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            "observations are disabled for this scope",
        )
    if min_proof < 1:
        raise VerbatimError(ErrorCode.VALIDATION, "min_proof must be >= 1")
    for name, val in (
        ("max_inputs", window.max_inputs),
        ("max_outputs", window.max_outputs),
    ):
        if val is not None and (
            isinstance(val, bool) or not isinstance(val, int) or val < 1
        ):
            raise VerbatimError(
                ErrorCode.VALIDATION, f"{name} must be a positive int"
            )
    if seq is None:
        seq = next_seq(conn, scope_id)

    report = ConsolidationReport(window=window)
    touched, triggers = _touched_slots(conn, scope_id, window)
    report.slots_touched = len(touched)
    if not touched:
        report.stopped = "no_material_change"
        return report

    # Touched claim ids = the triggering revisions' claims plus every
    # live input sitting in a touched slot (a claim's slot membership —
    # not its revision — decides which observation it can support).
    all_inputs = _fetch_inputs(
        conn, scope_id, states=states, modalities=modalities
    )
    if window.max_inputs is not None and len(all_inputs) > window.max_inputs:
        # Bound the *scan*, not the slot: keep whole touched slots intact
        # (a half-read slot would miscount families) — drop untouched
        # claims first, then truncate deterministically.
        in_touched = [ci for ci in all_inputs if _slot_key(ci) in touched]
        all_inputs = in_touched[: window.max_inputs]
        report.truncated = True
        if len(in_touched) > window.max_inputs:
            report.stopped = "max_inputs"
    claim_ids_in_touched = {
        ci.claim_id for ci in all_inputs if _slot_key(ci) in touched
    }
    c_touched = triggers | claim_ids_in_touched

    # Aggregate ONLY the touched slots, but with their full live
    # membership so family counts stay exact (V4-24.05 compaction keeps
    # the whole slot's evidence).
    candidates = aggregate_candidates(
        conn,
        scope_id,
        producer=producer,
        inputs=[ci for ci in all_inputs if _slot_key(ci) in touched],
    )
    report.candidates_seen = len(candidates)

    covered: set[str] = set()
    wrote_count = 0
    for cand in candidates:
        if cand.proof_count < min_proof:
            continue
        if (
            window.max_outputs is not None
            and wrote_count >= window.max_outputs
        ):
            report.truncated = True
            report.stopped = "max_outputs"
            break
        covered.add(cand.observation_id)
        res = persist_observation(
            conn,
            scope_id=scope_id,
            observation_id=cand.observation_id,
            text=cand.text,
            proof_count=cand.proof_count,
            perspective_id=cand.perspective_id,
            freshness=cand.freshness,
            producer=producer,
            supports=cand.supports,
            contradicts=cand.contradicts,
            seq=seq,
        )
        if res.wrote:
            wrote_count += 1
            report.observations_written += 1
            report.edges_written += res.edges_written
            report.refinements.append(
                {
                    "observation_id": cand.observation_id,
                    "revision": res.revision,
                    "previous_revision": res.previous_revision,
                    "reactivated": res.reactivated,
                    "added": [list(a) for a in res.added],
                    "removed": [list(r) for r in res.removed],
                }
            )

    if report.truncated:
        # A partial pass retires nothing it did not recompute.
        report.stopped = report.stopped or "truncated"
    else:
        report.retired = _retire_uncovered(
            conn, scope_id, producer, covered, seq,
            touched_claim_ids=c_touched,
        )
    report.stopped = report.stopped or (
        None
        if report.observations_written or report.retired
        else "no_material_change"
    )
    return report


def _fetch_inputs(
    conn: sqlite3.Connection,
    scope_id: str,
    *,
    states: Iterable[str],
    modalities: Iterable[str],
) -> list[ClaimInput]:
    return fetch_claim_inputs(
        conn, scope_id, states=states, modalities=modalities
    )
