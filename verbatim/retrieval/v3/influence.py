"""Influence handles, delivery ledger, feedback, blast radius (SPEC_V3 §32,
V3-32.01–32.08).

Every delivered pack item carries an :class:`InfluenceHandle` minted at
pack delivery time (V3-32.01): bound to the delivery receipt, the caller,
the authorization epoch in force, the pack kind, and the exact
(object_kind, object_id, revision) delivered. Delivery rows are written
transactionally inside the recall's write phase (V3-32.02).

Feedback (``used`` | ``cited`` | ``used_for_action`` | ``ignored`` |
``harmful``) and action receipts attach to the handle later
(V3-32.03/32.04); a missing feedback kind is ``NULL`` — recorded
*exposure*, never a claim that the item went uninfluenced or that the
model reasoned from it (V3-32.05).

``blast_radius(object)`` joins influence exposure with the propagation
ledger (§10.04) to answer "who has seen this object" for revocation and
harm assessment (V3-32.06). It reports exposure facts only — causal
attribution of model behavior is explicitly out of scope (V3-32.07).
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ...core.types import ErrorCode, VerbatimError, require_id, safe_json_loads
from ...core.types_v3 import (
    InfluenceFeedback,
    InfluenceHandle,
    PackKind,
)
from ...core.time import now_us

# ---------------------------------------------------------------------------
# handle minting (V3-32.01)
# ---------------------------------------------------------------------------


def mint_handle(
    receipt_id: str,
    caller_id: str,
    epoch: int,
    pack: PackKind,
    object_kind: str,
    object_id: str,
    revision: int,
    seq: int = 0,
) -> InfluenceHandle:
    """Deterministic handle id over the full delivery binding.

    Content-derived so a replayed recall mints identical handles — the
    delivery insert is then idempotent by primary key (V3-32.02).
    """
    digest = hashlib.sha256(
        (
            f"{receipt_id}|{caller_id}|{epoch}|{pack.value}|"
            f"{object_kind}|{object_id}|{revision}|{seq}"
        ).encode("utf-8")
    ).hexdigest()[:40]
    return InfluenceHandle(
        handle_id=f"ih_{digest}",
        receipt_id=receipt_id,
        caller_id=caller_id,
        epoch=epoch,
        pack=pack,
        object_kind=object_kind,
        object_id=object_id,
        revision=revision,
    )


# ---------------------------------------------------------------------------
# delivery + feedback writes (transaction-scoped, V3-32.02)
# ---------------------------------------------------------------------------


def record_delivery(
    conn: sqlite3.Connection,
    handle: InfluenceHandle,
    scope_id: str,
) -> str:
    """Insert the exposure row for one handle (idempotent by PK)."""
    record_deliveries(conn, [handle], scope_id)
    return handle.handle_id


def record_deliveries(
    conn: sqlite3.Connection,
    handles: Iterable[InfluenceHandle],
    scope_id: str,
) -> None:
    """Batched ``record_delivery`` — the same INSERT OR IGNORE rows under
    one ``executemany`` round-trip; per-row ``created_us`` is preserved
    (stamped per handle, matching the sequential loop's values up to the
    usual microsecond skew of any multi-statement write)."""
    rows = [
        (
            handle.handle_id,
            handle.receipt_id,
            scope_id,
            handle.caller_id,
            handle.epoch,
            handle.pack.value if isinstance(handle.pack, PackKind) else handle.pack,
            handle.object_kind,
            handle.object_id,
            handle.revision,
            now_us(),
        )
        for handle in handles
    ]
    if not rows:
        return
    conn.executemany(
        "INSERT OR IGNORE INTO influence"
        " (handle_id, receipt_id, scope_id, caller_id, epoch, pack,"
        "  object_kind, object_id, revision, created_us)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def record_feedback(
    conn: sqlite3.Connection,
    handle_id: str,
    feedback: InfluenceFeedback,
    action_receipt: Optional[str] = None,
) -> None:
    """Attach host feedback / an action receipt to a delivered handle.

    The handle must exist — feedback for an undelivered item is a
    ``VALIDATION`` error, not an invented exposure (V3-32.04).
    """
    require_id(handle_id, "handle_id")
    fb = feedback.value if isinstance(
        feedback, InfluenceFeedback
    ) else InfluenceFeedback(feedback).value
    cur = conn.execute(
        "UPDATE influence SET feedback_kind = ?, action_receipt = ?"
        " WHERE handle_id = ?",
        (fb, action_receipt, handle_id),
    )
    if cur.rowcount == 0:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown influence handle {handle_id!r}"
        )


def redact_handle(conn: sqlite3.Connection, handle_id: str) -> None:
    """Tombstone a handle (privacy purge) — exposure stays countable."""
    conn.execute(
        "UPDATE influence SET redacted = 1 WHERE handle_id = ?",
        (handle_id,),
    )


# ---------------------------------------------------------------------------
# blast radius (V3-32.06)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Exposure:
    """One exposure fact: who saw what, through which channel."""

    channel: str          # influence|propagation
    recipient_id: str
    pack: Optional[str]
    feedback_kind: Optional[str]
    epoch: int
    created_us: int
    redacted: bool = False


@dataclass
class BlastRadius:
    """Exposure report for one object — facts only, no causal inference."""

    object_kind: str
    object_id: str
    revision: Optional[int]
    exposures: list = field(default_factory=list)
    receipts: list = field(default_factory=list)

    @property
    def recipients(self) -> set:
        return {e.recipient_id for e in self.exposures}


def blast_radius(
    conn: sqlite3.Connection,
    object_kind: str,
    object_id: str,
    revision: Optional[int] = None,
) -> BlastRadius:
    """Join influence + propagation ledgers for one object.

    Rows are exposure records — missing feedback stays ``NULL``; the
    report never marks an exposure "unused" and never claims the model
    reasoned from it (V3-32.05/32.07).
    """
    require_id(object_id, "object_id")
    out = BlastRadius(object_kind, object_id, revision)
    params: list = [object_kind, object_id]
    rev_pred = ""
    if revision is not None:
        rev_pred = " AND revision = ?"
        params.append(revision)

    try:
        rows = conn.execute(
            "SELECT receipt_id, caller_id, pack, feedback_kind, epoch,"
            " created_us, redacted FROM influence"
            f" WHERE object_kind = ? AND object_id = ?{rev_pred}"
            " ORDER BY created_us, handle_id",
            params,
        ).fetchall()
    except sqlite3.Error:
        rows = []
    for receipt_id, caller_id, pack, fb, epoch, created, redacted in rows:
        out.exposures.append(Exposure(
            channel="influence",
            recipient_id=caller_id,
            pack=pack,
            feedback_kind=fb,  # NULL = delivered, never "unused"
            epoch=epoch,
            created_us=created,
            redacted=bool(redacted),
        ))
        if receipt_id not in out.receipts:
            out.receipts.append(receipt_id)

    # propagation ledger (§10.04): disclosures across boundaries
    try:
        params2: list = [object_kind, object_id]
        rev_pred2 = ""
        if revision is not None:
            rev_pred2 = " AND revision = ?"
            params2.append(revision)
        prows = conn.execute(
            "SELECT recipient_id, purpose, epoch, created_us, capsule_id"
            " FROM propagations"
            f" WHERE object_kind = ? AND object_id = ?{rev_pred2}"
            " ORDER BY created_us, propagation_id",
            params2,
        ).fetchall()
    except sqlite3.Error:
        prows = []
    for recipient_id, purpose, epoch, created, capsule_id in prows:
        out.exposures.append(Exposure(
            channel="propagation",
            recipient_id=recipient_id,
            pack=purpose,
            feedback_kind=None,
            epoch=epoch,
            created_us=created,
        ))
    return out


def feedback_summary(conn: sqlite3.Connection,
                     receipt_id: str) -> dict:
    """Per-receipt feedback tallies for routing_stats credit (§26.12)."""
    rows = conn.execute(
        "SELECT feedback_kind, COUNT(*) FROM influence"
        " WHERE receipt_id = ? GROUP BY feedback_kind",
        (receipt_id,),
    ).fetchall()
    return {(k or "delivered"): n for k, n in rows}
