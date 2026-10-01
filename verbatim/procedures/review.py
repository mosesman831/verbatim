"""The procedure promotion ladder (SPEC_V3 §21–§22.04).

``candidate → reviewed → active`` — v3.0 requires *explicit review* for
activation (V3-22.04). Rejection retires the candidate. A verified
applicability or safety violation suspends an active procedure to
``deprecated`` (V3-22.07). Retired and deprecated procedures stay
retrievable as history and counterexamples — retirement never erases
evidence (V3-22.08).

Automatic promotion is configuration (``v3.procedure_promotion=auto``)
but even then it still requires the G5 paired-evidence bar —
:func:`requires_paired_evidence` reports those requirements verbatim so no
code path can pretend a flag alone is promotion evidence. In v3.0 the
only implemented path is the manual ladder here; there is no silent
auto-approve.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

from ..core.types import ErrorCode, VerbatimError, json_dumps, require_id
from ..core.types_v3 import ProcedureStateV3
from ..storage.repos import _json_parse, _next_event_seq, _row

#: The v3 promotion ladder (§21 state row, §22.04).
_LADDER: dict[str, frozenset[str]] = {
    "candidate": frozenset({"reviewed", "retired"}),
    "reviewed": frozenset({"active", "retired"}),
    "active": frozenset({"deprecated", "retired"}),
    "deprecated": frozenset({"retired"}),
    "retired": frozenset(),
}

#: What ``v3.procedure_promotion=auto`` still requires before any automatic
#: activation may be proposed (V3-22.04). This is a reporting contract —
#: nothing in v3.0 auto-promotes.
_PAIRED_EVIDENCE_REQUIREMENTS: dict[str, Any] = {
    "explicit_review_required_v3_0": True,
    "auto_promotion_implemented": False,
    "paired_executions_required": True,
    "paired_arm_description": (
        "outcomes measured on paired executions (with/without the "
        "procedure) across environment classes, not adoption/failure "
        "correlation (V3-22.07, §53)"
    ),
    "promotion_precision_protocol": (
        "a separately frozen promotion-precision/coverage protocol and "
        "artifact under §42"
    ),
    "min_checker_receipt": True,
    "no_unresolved_findings": True,
}


def _procedure_row(
    conn: sqlite3.Connection, procedure_id: str
) -> dict[str, Any]:
    require_id(procedure_id, "procedure_id")
    row = _row(
        conn.execute(
            "SELECT * FROM procedures WHERE procedure_id = ?",
            (procedure_id,),
        )
    )
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "procedure not found"
        )
    return row


def _transition(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    new_state: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Guarded state write: the expected current state is the fence."""
    pid = row["procedure_id"]
    cur = conn.execute(
        "UPDATE procedures SET state = ?, provenance_json = ?,"
        " row_version = row_version + 1"
        " WHERE procedure_id = ? AND state = ?",
        (new_state, json_dumps(provenance), pid, row["state"]),
    )
    if cur.rowcount == 0:
        raise VerbatimError(
            ErrorCode.STALE_PROPOSAL,
            f"procedure {pid} state changed under transition",
        )
    return {"procedure_id": pid, "state": new_state,
            "revision": int(row["revision"])}


def requires_paired_evidence() -> dict[str, Any]:
    """Evidence requirements for any automatic promotion (V3-22.04).

    Callers wiring ``v3.procedure_promotion=auto`` must satisfy this whole
    contract first — paired task executions across environment classes plus
    a frozen promotion-precision artifact. v3.0 implements manual review
    only.
    """
    return dict(_PAIRED_EVIDENCE_REQUIREMENTS)


def review(
    conn: sqlite3.Connection,
    procedure_id: str,
    decision: str,
    reviewer_id: str,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    """Apply a reviewer decision to a candidate (V3-22.04).

    ``approve`` moves ``candidate → reviewed``; ``reject`` retires the
    candidate. Every decision is recorded in the procedure's provenance
    with reviewer identity and notes — the review evidence
    :func:`activate` requires.
    """
    require_id(reviewer_id, "reviewer_id")
    if decision not in ("approve", "reject"):
        raise VerbatimError(
            ErrorCode.VALIDATION, "decision must be 'approve' or 'reject'"
        )
    row = _procedure_row(conn, procedure_id)
    state = row["state"]
    allowed_from = ("candidate",) if decision == "approve" else (
        "candidate", "reviewed",
    )
    if state not in allowed_from:
        raise VerbatimError(
            ErrorCode.INVALID_TRANSITION,
            f"cannot {decision} a procedure in state {state!r}",
        )
    new_state = (
        ProcedureStateV3.REVIEWED.value
        if decision == "approve"
        else ProcedureStateV3.RETIRED.value
    )
    if new_state not in _LADDER.get(state, frozenset()):
        raise VerbatimError(
            ErrorCode.INVALID_TRANSITION,
            f"procedure state {state} -> {new_state} is not allowed",
        )
    provenance = _json_parse(row.get("provenance_json")) or {}
    reviews = list(provenance.get("reviews") or [])
    reviews.append(
        {
            "reviewer_id": reviewer_id,
            "decision": decision,
            "notes": notes or "",
            "seq": _next_event_seq(conn),
        }
    )
    provenance["reviews"] = reviews
    return _transition(conn, row, new_state, provenance)


def activate(
    conn: sqlite3.Connection,
    procedure_id: str,
    activator_id: str,
) -> dict[str, Any]:
    """``reviewed → active`` — only with recorded review evidence.

    v3.0 activation is explicit (V3-22.04): the procedure must already be
    ``reviewed`` *and* carry an ``approve`` review entry in its provenance.
    Anything else is refused, not bypassed.
    """
    require_id(activator_id, "activator_id")
    row = _procedure_row(conn, procedure_id)
    if row["state"] != ProcedureStateV3.REVIEWED.value:
        raise VerbatimError(
            ErrorCode.INVALID_TRANSITION,
            f"activate requires state 'reviewed', got {row['state']!r}",
        )
    provenance = _json_parse(row.get("provenance_json")) or {}
    approvals = [
        r for r in provenance.get("reviews") or []
        if isinstance(r, dict) and r.get("decision") == "approve"
    ]
    if not approvals:
        # A 'reviewed' row without review evidence cannot activate — the
        # ladder is evidence-gated, not state-gated (V3-22.04).
        raise VerbatimError(
            ErrorCode.INVALID_TRANSITION,
            "activate requires approve-review evidence",
        )
    activations = list(provenance.get("activations") or [])
    activations.append(
        {"activator_id": activator_id, "seq": _next_event_seq(conn)}
    )
    provenance["activations"] = activations
    return _transition(
        conn, row, ProcedureStateV3.ACTIVE.value, provenance
    )


def suspend(
    conn: sqlite3.Connection,
    procedure_id: str,
    reason: str,
) -> dict[str, Any]:
    """``active → deprecated`` on a verified applicability/safety violation
    (V3-22.07). The suspension reason is recorded; ordinary unpaired
    failures open investigation elsewhere, not this path."""
    if not isinstance(reason, str) or not reason:
        raise VerbatimError(
            ErrorCode.VALIDATION, "suspend requires a non-empty reason"
        )
    row = _procedure_row(conn, procedure_id)
    if row["state"] != ProcedureStateV3.ACTIVE.value:
        raise VerbatimError(
            ErrorCode.INVALID_TRANSITION,
            f"suspend requires state 'active', got {row['state']!r}",
        )
    provenance = _json_parse(row.get("provenance_json")) or {}
    suspensions = list(provenance.get("suspensions") or [])
    suspensions.append({"reason": reason, "seq": _next_event_seq(conn)})
    provenance["suspensions"] = suspensions
    return _transition(
        conn, row, ProcedureStateV3.DEPRECATED.value, provenance
    )


def review_procedure(
    conn: sqlite3.Connection,
    procedure_id: str,
    decision: str,
    reviewer_id: str,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    """Package-level alias for :func:`review`."""
    return review(conn, procedure_id, decision, reviewer_id, notes)
