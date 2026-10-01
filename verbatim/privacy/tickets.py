"""Action tickets: engine-issued, gateway-checked authority tokens
(SPEC_V3 §09.07, §10, §35.04).

An action ticket is a short-lived, one-use authorization for a *specific*
action digest, recipient, purpose, and object set. The engine issues; a
registered execution gateway validates immediately before running the
action — ``act`` is never implied by ``read`` or ``quote``, and without a
registered gateway ``action_use_enforcement`` is ``advisory_only`` (§09.07).

Semantics:

- ``issue`` writes ``action_tickets`` + ``ticket_objects`` inside the
  caller's transaction. Lifetime defaults to 30 s and is capped at 60 s —
  the ``ActionTicket`` type enforces the same bound.
- ``verify`` checks the presenting gateway, the nonce's one-use state,
  expiry, the exact action digest, and the pinned epoch against the
  scope's current ``authz_revision`` — then consumes the ticket
  atomically in the same transaction. Replay via the nonce is impossible:
  the first consume sets ``consumed_us`` and any second presentation fails
  ``VALIDATION``.
- Mismatched gateway, digest, or recipient surfaces as
  ``NOT_FOUND_OR_UNAUTHORIZED`` — the check never distinguishes "no
  ticket" from "not your ticket" (§09.09). A superseded epoch is
  ``STALE_EPOCH`` (retryable after rebind, §46).
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from typing import Any, Iterable, Optional

from ..core.time import now_us
from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    require_id,
)
from ..core.types_v3 import ActionTicket
from ..storage import repos_v3

DEFAULT_TTL_US = ActionTicket.DEFAULT_LIFETIME_S * 1_000_000
MAX_TTL_US = ActionTicket.MAX_LIFETIME_S * 1_000_000
_NONCE_BYTES = 16


def _current_epoch(conn: sqlite3.Connection, scope_id: str) -> int:
    row = conn.execute(
        "SELECT authz_revision FROM scopes WHERE scope_id = ?", (scope_id,)
    ).fetchone()
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "scope unavailable"
        )
    return int(row[0])


def _ticket_id(
    *,
    scope_id: str,
    recipient_id: str,
    action_digest: bytes,
    purpose: str,
    epoch: int,
    nonce: str,
    issued_us: int,
    expires_us: int,
    gateway_id: str,
) -> str:
    """Deterministic id bound to every issued field plus the nonce — the
    digest makes the ticket id unforgeable-by-construction while the row
    remains the authority."""
    canonical = json_dumps(
        {
            "scope_id": scope_id,
            "recipient_id": recipient_id,
            "action_digest": action_digest.hex(),
            "purpose": purpose,
            "epoch": int(epoch),
            "nonce": nonce,
            "issued_us": int(issued_us),
            "expires_us": int(expires_us),
            "gateway_id": gateway_id,
        }
    )
    return "tk-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:40]


def issue(
    conn: sqlite3.Connection,
    scope_id: str,
    recipient_id: str,
    action_digest: bytes,
    purpose: str,
    epoch: Optional[int],
    objects: Iterable[tuple[str, int]],
    ttl_us: Optional[int],
    gateway_id: str,
    *,
    now: Optional[int] = None,
) -> str:
    """Issue a one-use ticket bound to ``(recipient, action_digest,
    purpose, epoch, objects, gateway)``; returns the ``ticket_id``.

    ``epoch=None`` pins the scope's current ``authz_revision``; an explicit
    epoch must equal it or the request is already stale. ``objects`` are
    the exact ``(object_id, revision)`` pairs the action may consume.
    """
    require_id(scope_id, "scope_id")
    require_id(recipient_id, "recipient_id")
    require_id(gateway_id, "gateway_id")
    if not purpose:
        raise VerbatimError(ErrorCode.VALIDATION, "purpose is mandatory")
    if not isinstance(action_digest, (bytes, bytearray)) or not action_digest:
        raise VerbatimError(
            ErrorCode.VALIDATION, "action_digest must be non-empty bytes"
        )
    ts = now if now is not None else now_us()
    current = _current_epoch(conn, scope_id)
    if epoch is not None and int(epoch) != current:
        raise VerbatimError(
            ErrorCode.STALE_EPOCH,
            "pinned epoch superseded — rebind authorization",
            retryable=True,
        )
    ttl = DEFAULT_TTL_US if ttl_us is None else int(ttl_us)
    if ttl <= 0 or ttl > MAX_TTL_US:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "ticket lifetime must be within (0, 60s]",
        )
    issued_us = int(ts)
    expires_us = issued_us + ttl
    nonce = secrets.token_hex(_NONCE_BYTES)
    tid = _ticket_id(
        scope_id=scope_id,
        recipient_id=recipient_id,
        action_digest=bytes(action_digest),
        purpose=purpose,
        epoch=current,
        nonce=nonce,
        issued_us=issued_us,
        expires_us=expires_us,
        gateway_id=gateway_id,
    )
    repos_v3.insert(
        conn,
        "action_tickets",
        {
            "ticket_id": tid,
            "scope_id": scope_id,
            "recipient_id": recipient_id,
            "action_digest": bytes(action_digest),
            "purpose": purpose,
            "epoch": current,
            "nonce": nonce,
            "issued_us": issued_us,
            "expires_us": expires_us,
            "gateway_id": gateway_id,
            "consumed_us": None,
        },
    )
    for object_id, revision in objects or ():
        require_id(object_id, "object_id")
        repos_v3.insert(
            conn,
            "ticket_objects",
            {
                "ticket_id": tid,
                "object_id": object_id,
                "revision": int(revision),
            },
        )
    return tid


def verify(
    conn: sqlite3.Connection,
    ticket_id: str,
    gateway_id: str,
    action_digest: bytes,
    now: Optional[int] = None,
) -> dict[str, Any]:
    """Validate *and consume* a ticket at the execution gateway (§09.07).

    Success returns the ticket snapshot plus its bound object refs. The
    consume is an atomic ``UPDATE … WHERE consumed_us IS NULL`` inside the
    caller's transaction — replay is impossible even under a race.
    """
    ts = now if now is not None else now_us()
    row = repos_v3.get(conn, "action_tickets", {"ticket_id": ticket_id})
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "ticket unavailable"
        )
    if row["gateway_id"] != gateway_id:
        # A ticket is bound to one registered gateway; any other presenter
        # gets the same indistinguishable denial as a missing ticket.
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "ticket unavailable"
        )
    if row.get("consumed_us") is not None:
        raise VerbatimError(
            ErrorCode.VALIDATION, "ticket already consumed"
        )
    if int(row["expires_us"]) <= ts:
        raise VerbatimError(ErrorCode.VALIDATION, "ticket expired")
    if bytes(row["action_digest"]) != bytes(action_digest):
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "ticket unavailable"
        )
    current = _current_epoch(conn, row["scope_id"])
    if int(row["epoch"]) != current:
        raise VerbatimError(
            ErrorCode.STALE_EPOCH,
            "ticket epoch superseded — rebind authorization",
            retryable=True,
        )
    claimed = repos_v3.update(
        conn,
        "action_tickets",
        {"consumed_us": int(ts)},
        {"ticket_id": ticket_id, "consumed_us": None},
    )
    if claimed != 1:
        raise VerbatimError(
            ErrorCode.VALIDATION, "ticket already consumed"
        )
    objects = repos_v3.query(
        conn, "ticket_objects", {"ticket_id": ticket_id}
    )
    out = dict(row)
    out["consumed_us"] = int(ts)
    out["objects"] = [
        {"object_id": o["object_id"], "revision": o["revision"]}
        for o in objects
    ]
    return out
