"""Capture receipts for the v3 evidence plane (SPEC_V3 §12.05, §13.13, §47.03).

A ``CaptureReceipt`` is the durable-acceptance contract returned by
``ingest_envelope``: it separates *durable acceptance* from *interpretation
readiness* (V3-12.05) and binds the committed identities — source, revision,
envelope, scope, the committed event sequence, and the dedup sequence key
(V3-12.04, V3-47.03).

Receipts are value objects, not a table: ``receipt_id`` derives
deterministically from ``(source_id, revision, envelope_kind)`` so the same
logical capture always mints the same receipt, and ``verify_receipt``
re-reads the persisted ``source_envelopes``/``events`` rows to confirm every
bound field — a receipt that cannot be re-derived from durable state is an
integrity failure, never a silent pass.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from ..core.types import ErrorCode, VerbatimError, require_id
from ..storage import repos_v3

# Event kind recorded for each accepted envelope (mirrors the v2
# "source_accepted" journal convention, SPEC §20).
ENVELOPE_EVENT_KIND = "envelope_captured"


def _receipt_id(source_id: str, revision: int, envelope_kind: str) -> str:
    """Deterministic receipt identity over the unique envelope triple
    ``(source_id, revision, envelope_kind)`` — the same key
    ``idx_envelopes_source`` enforces UNIQUE on."""
    digest = hashlib.sha256(
        f"{source_id}\x00{revision}\x00{envelope_kind}".encode("utf-8")
    ).hexdigest()
    return f"cr_{digest[:32]}"


def receipt_id_for(source_id: str, revision: int, envelope_kind: str) -> str:
    """The deterministic receipt identity ``mint_receipt`` stamps.

    Exposed for readers that need the receipt *id* alone: deriving it
    directly skips the event/span anchoring queries
    ``receipt_for_envelope`` runs to populate the full
    ``CaptureReceipt`` (``event_seq``, ``span_ids``, ``accepted_bytes``).
    """
    return _receipt_id(source_id, revision, envelope_kind)


@dataclass(frozen=True)
class CaptureReceipt:
    """What ``ingest_envelope`` durably accepted (§12.05, §13.13).

    - ``event_seq``: the committed ``envelope_captured`` event sequence —
      receipt anchor for mutation responses (§47.03).
    - ``dedup_key``: the effective sequence key (host ``external_id`` or the
      synthesized ``v3seq:`` key) that makes retries idempotent (§12.04).
    - ``readiness``: ``accepted`` means durable; interpretation readiness is
      a separate later stage and is never implied here (§12.05).
    """

    receipt_id: str
    envelope_id: str
    source_id: str
    revision: int
    scope_id: str
    envelope_kind: str
    event_seq: int
    dedup_key: str
    accepted_bytes: int
    span_ids: tuple[str, ...] = ()
    trust_class: str = "unknown"
    receipt_us: int = 0
    readiness: str = "accepted"

    def __post_init__(self) -> None:
        for name in ("receipt_id", "envelope_id", "source_id", "scope_id"):
            require_id(getattr(self, name), name)
        object.__setattr__(self, "span_ids", tuple(self.span_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "envelope_id": self.envelope_id,
            "source_id": self.source_id,
            "revision": self.revision,
            "scope_id": self.scope_id,
            "envelope_kind": self.envelope_kind,
            "event_seq": self.event_seq,
            "dedup_key": self.dedup_key,
            "accepted_bytes": self.accepted_bytes,
            "span_ids": list(self.span_ids),
            "trust_class": self.trust_class,
            "receipt_us": self.receipt_us,
            "readiness": self.readiness,
        }


def _event_seq_for(conn: sqlite3.Connection, scope_id: str, envelope_id: str) -> int:
    """The committed ``envelope_captured`` sequence for one envelope.

    The events journal is the durable anchor; payload lookup is parameterized
    and ``envelope_id`` is data, never SQL structure.
    """
    row = conn.execute(
        "SELECT event_seq FROM events"
        " WHERE scope_id = ? AND kind = ?"
        "   AND json_extract(payload_json, '$.envelope_id') = ?"
        " ORDER BY event_seq LIMIT 1",
        (scope_id, ENVELOPE_EVENT_KIND, envelope_id),
    ).fetchone()
    return int(row[0]) if row is not None else 0


def _accepted_bytes(conn: sqlite3.Connection, source_id: str, revision: int) -> int:
    row = conn.execute(
        "SELECT length(payload) FROM source_revisions"
        " WHERE source_id = ? AND revision = ?",
        (source_id, revision),
    ).fetchone()
    return int(row[0]) if row is not None and row[0] is not None else 0


def mint_receipt(
    conn: sqlite3.Connection,
    envelope_row: dict[str, Any],
    *,
    event_seq: int,
    dedup_key: str,
    span_ids: Iterable[str] = (),
    accepted_bytes: Optional[int] = None,
    readiness: str = "accepted",
) -> CaptureReceipt:
    """Mint the receipt for a persisted ``source_envelopes`` row.

    Idempotent by construction (V3-12.04): the receipt id is a pure function
    of the UNIQUE ``(source_id, revision, envelope_kind)`` triple, so a
    retried ingest over the same stored row mints an identical receipt.
    """
    if not isinstance(envelope_row, dict):
        raise VerbatimError(ErrorCode.VALIDATION, "envelope_row must be a row dict")
    missing = {
        "envelope_id", "source_id", "revision", "scope_id", "envelope_kind"
    } - set(envelope_row)
    if missing:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"envelope_row missing columns: {sorted(missing)}",
        )
    source_id = envelope_row["source_id"]
    revision = int(envelope_row["revision"])
    kind = envelope_row["envelope_kind"]
    if accepted_bytes is None:
        accepted_bytes = _accepted_bytes(conn, source_id, revision)
    return CaptureReceipt(
        receipt_id=_receipt_id(source_id, revision, kind),
        envelope_id=envelope_row["envelope_id"],
        source_id=source_id,
        revision=revision,
        scope_id=envelope_row["scope_id"],
        envelope_kind=kind,
        event_seq=int(event_seq),
        dedup_key=dedup_key,
        accepted_bytes=int(accepted_bytes),
        span_ids=tuple(span_ids),
        trust_class=envelope_row.get("trust_class") or "unknown",
        receipt_us=int(envelope_row.get("receipt_us") or 0),
        readiness=readiness,
    )


def _span_ids_for(conn: sqlite3.Connection, source_id: str, revision: int) -> tuple[str, ...]:
    rows = conn.execute(
        "SELECT span_id FROM spans WHERE source_id = ? AND revision = ?"
        " ORDER BY start_byte, end_byte",
        (source_id, revision),
    ).fetchall()
    return tuple(str(r[0]) for r in rows)


def receipt_for_envelope(
    conn: sqlite3.Connection,
    envelope_row: dict[str, Any],
    *,
    dedup_key: str,
    span_ids: Optional[Iterable[str]] = None,
) -> CaptureReceipt:
    """Rebuild the receipt for an already-persisted envelope row — the
    dedup-hit path of ingest and the replay manifest's receipt list.
    The committed event sequence and span handles are recovered from
    durable state so a replay mints a byte-identical receipt."""
    seq = _event_seq_for(conn, envelope_row["scope_id"], envelope_row["envelope_id"])
    if span_ids is None:
        span_ids = _span_ids_for(
            conn, envelope_row["source_id"], int(envelope_row["revision"])
        )
    return mint_receipt(
        conn,
        envelope_row,
        event_seq=seq,
        dedup_key=dedup_key,
        span_ids=span_ids,
    )


def verify_receipt(conn: sqlite3.Connection, receipt: CaptureReceipt) -> CaptureReceipt:
    """Re-verify a minted receipt against durable state (§13.13, §47.03).

    Reads the bound ``source_envelopes`` row back and recomputes the receipt
    id. Absent rows are ``NOT_FOUND_OR_UNAUTHORIZED`` (existence is never
    distinguished from authorization publicly); a field or identity mismatch
    is ``INTEGRITY`` — a receipt that does not re-derive is not honored.
    """
    if not isinstance(receipt, CaptureReceipt):
        raise VerbatimError(ErrorCode.VALIDATION, "verify_receipt needs a CaptureReceipt")
    row = repos_v3.get(conn, "source_envelopes", {"envelope_id": receipt.envelope_id})
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "receipt target not found"
        )
    mismatches = [
        name
        for name, got, want in (
            ("source_id", receipt.source_id, row["source_id"]),
            ("revision", receipt.revision, int(row["revision"])),
            ("scope_id", receipt.scope_id, row["scope_id"]),
            ("envelope_kind", receipt.envelope_kind, row["envelope_kind"]),
        )
        if got != want
    ]
    if mismatches:
        raise VerbatimError(
            ErrorCode.INTEGRITY,
            f"receipt fields diverge from durable row: {', '.join(mismatches)}",
        )
    if receipt.receipt_id != _receipt_id(row["source_id"], int(row["revision"]), row["envelope_kind"]):
        raise VerbatimError(
            ErrorCode.INTEGRITY, "receipt id does not re-derive from durable identity"
        )
    if receipt.event_seq:
        ev = conn.execute(
            "SELECT 1 FROM events WHERE event_seq = ? AND scope_id = ? AND kind = ?"
            "   AND json_extract(payload_json, '$.envelope_id') = ?",
            (receipt.event_seq, receipt.scope_id, ENVELOPE_EVENT_KIND, receipt.envelope_id),
        ).fetchone()
        if ev is None:
            raise VerbatimError(
                ErrorCode.INTEGRITY, "receipt event anchor missing or mismatched"
            )
    return receipt
