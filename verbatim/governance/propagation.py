"""Propagation ledger (SPEC_V3 §10.04–§10.05).

Every cross-boundary disclosure — share, handoff, publication, hydration —
is recorded with recipient, verbs, purpose, and the authorization epoch so
blast radius is computable per source revision. Ledger rows are facts of
delivery: revocation marks ``revoked_seq`` (the epoch at which copies were
fenced), it never pretends delivered bytes were recalled (§10.05).
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable, Optional

from ..core.time import now_us
from ..core.types import ErrorCode, VerbatimError, new_id, require_id
from ..core.types_v3 import Propagation, Verb
from ..storage import repos_v3

_TABLE = "propagations"


def record_propagation(
    conn: sqlite3.Connection, propagation: "Propagation | dict"
) -> str:
    """Append one disclosure record; returns propagation_id.

    Accepts the frozen ``Propagation`` contract or an equivalent mapping;
    verbs serialize through the enum so stored values are always §09 verbs.
    """
    if isinstance(propagation, dict):
        p = propagation
        pid = p.get("propagation_id") or f"prop:{new_id()}"
        scope_id = p["scope_id"]
        object_kind = p["object_kind"]
        object_id = p["object_id"]
        revision = int(p["revision"])
        recipient_id = p["recipient_id"]
        verbs = frozenset(p["verbs"])
        purpose = p["purpose"]
        epoch = int(p["epoch"])
        created = p.get("created_us") or now_us()
        capsule_id = p.get("capsule_id")
        revoked_seq = p.get("revoked_seq")
        acknowledged = bool(p.get("acknowledged", False))
    else:
        pid = propagation.propagation_id or f"prop:{new_id()}"
        scope_id = propagation.scope_id
        object_kind = propagation.object_kind
        object_id = propagation.object_id
        revision = int(propagation.revision)
        recipient_id = propagation.recipient_id
        verbs = propagation.verbs
        purpose = propagation.purpose
        epoch = int(propagation.epoch)
        created = propagation.created_us or now_us()
        capsule_id = propagation.capsule_id
        revoked_seq = propagation.revoked_seq
        acknowledged = propagation.acknowledged
    require_id(pid, "propagation_id")
    require_id(scope_id, "scope_id")
    require_id(object_id, "object_id")
    require_id(recipient_id, "recipient_id")
    if not isinstance(purpose, str) or not purpose:
        raise VerbatimError(ErrorCode.VALIDATION, "purpose required")
    try:
        verb_values = sorted(
            (v if isinstance(v, Verb) else Verb(v)).value for v in verbs
        )
    except ValueError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown verb in propagation: {exc}"
        ) from exc
    repos_v3.insert(
        conn,
        _TABLE,
        {
            "propagation_id": pid,
            "scope_id": scope_id,
            "object_kind": object_kind,
            "object_id": object_id,
            "revision": revision,
            "recipient_id": recipient_id,
            "verbs_json": verb_values,
            "purpose": purpose,
            "epoch": epoch,
            "created_us": created,
            "capsule_id": capsule_id,
            "revoked_seq": revoked_seq,
            "acknowledged": 1 if acknowledged else 0,
        },
    )
    return pid


def propagations_for(
    conn: sqlite3.Connection,
    object_kind: str,
    object_id: str,
    *,
    include_revoked: bool = False,
) -> list[dict]:
    """Ledger rows for one object, oldest first; revoked rows hidden unless
    asked for."""
    rows = repos_v3.query(
        conn,
        _TABLE,
        {"object_kind": object_kind, "object_id": object_id},
        order="created_us",
    )
    if include_revoked:
        return rows
    return [r for r in rows if r.get("revoked_seq") is None]


def blast_radius(
    conn: sqlite3.Connection, object_kind: str, object_id: str
) -> list[dict]:
    """Per-recipient disclosure view of one object (§10.04, §45.04).

    Every recorded propagation becomes one entry — recipient, verbs,
    purpose, the epoch it was recorded at, and whether a fleet revocation
    has since fenced it. Already-delivered context is listed, never
    hidden (§10.05).
    """
    rows = repos_v3.query(
        conn,
        _TABLE,
        {"object_kind": object_kind, "object_id": object_id},
        order="created_us",
    )
    return [
        {
            "propagation_id": r["propagation_id"],
            "scope_id": r["scope_id"],
            "recipient_id": r["recipient_id"],
            "verbs": repos_v3.json_field(r, "verbs_json") or [],
            "purpose": r["purpose"],
            "epoch": r["epoch"],
            "capsule_id": r.get("capsule_id"),
            "revoked": r.get("revoked_seq") is not None,
            "revoked_seq": r.get("revoked_seq"),
            "acknowledged": bool(r.get("acknowledged")),
        }
        for r in rows
    ]


def recipients_of(
    conn: sqlite3.Connection, object_kind: str, object_id: str
) -> frozenset:
    """Distinct recipients that ever received the object (live + fenced)."""
    return frozenset(
        r["recipient_id"] for r in blast_radius(conn, object_kind, object_id)
    )


def mark_propagations_revoked(
    conn: sqlite3.Connection,
    *,
    seq: int,
    scope_id: Optional[str] = None,
    recipient_id: Optional[str] = None,
    object_kind: Optional[str] = None,
    object_id: Optional[str] = None,
    capsule_id: Optional[str] = None,
) -> int:
    """Fleet-revocation marking (§10.05): set ``revoked_seq`` on live rows.

    Filters are conjunctive; at least one is required so a caller can never
    blanket-mark the whole ledger. Returns the number of rows fenced.
    """
    where: dict[str, Any] = {"revoked_seq": None}
    if scope_id is not None:
        where["scope_id"] = scope_id
    if recipient_id is not None:
        where["recipient_id"] = recipient_id
    if object_kind is not None:
        where["object_kind"] = object_kind
    if object_id is not None:
        where["object_id"] = object_id
    if capsule_id is not None:
        where["capsule_id"] = capsule_id
    if len(where) == 1:
        raise VerbatimError(
            ErrorCode.VALIDATION, "revocation marking needs a filter"
        )
    return repos_v3.update(conn, _TABLE, {"revoked_seq": int(seq)}, where)


def acknowledge_propagation(
    conn: sqlite3.Connection, propagation_id: str
) -> bool:
    """Recipient acknowledged a revocation notice (§10.05).

    Only a fenced row can be acknowledged — the notice says a delivered
    copy was revoked, so acknowledging an unrevoked row would be a lie.
    """
    require_id(propagation_id, "propagation_id")
    row = repos_v3.get(conn, _TABLE, {"propagation_id": propagation_id})
    if row is None or row.get("revoked_seq") is None:
        return False
    return bool(
        repos_v3.update(
            conn,
            _TABLE,
            {"acknowledged": 1},
            {"propagation_id": propagation_id},
        )
    )
