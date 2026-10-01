"""Scope authorization epochs (SPEC_V3 §09.04, §46).

Every scope carries ``authz_revision``: bound callers pin an epoch,
revocation increments it, and a pinned caller whose epoch no longer matches
fences with ``STALE_EPOCH`` (retryable after rebind — §46). The scope row is
the v1 ``scopes`` table (``authz_revision`` added by the v2 ALTER), so these
helpers use explicit parameterized SQL rather than ``repos_v3`` — that
allowlist covers only the §39 v3 tables.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from ..core.types import ErrorCode, VerbatimError, require_id


def current_epoch(conn: sqlite3.Connection, scope_id: str) -> int:
    """The scope's current ``authz_revision``; 0 when the row is absent."""
    require_id(scope_id, "scope_id")
    row = conn.execute(
        "SELECT authz_revision FROM scopes WHERE scope_id = ?",
        (scope_id,),
    ).fetchone()
    return int(row[0]) if row else 0


def bump_epoch(conn: sqlite3.Connection, scope_id: str) -> int:
    """Advance the scope's authorization epoch; returns the new value.

    A revocation or grant change calls this inside the same write
    transaction so callers either see the old epoch or the new state —
    never a half-applied revocation (§09.04).
    """
    require_id(scope_id, "scope_id")
    cur = conn.execute(
        "UPDATE scopes SET authz_revision = authz_revision + 1"
        " WHERE scope_id = ?",
        (scope_id,),
    )
    if cur.rowcount == 0:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown scope {scope_id!r}"
        )
    return current_epoch(conn, scope_id)


def bump_epoch_if_present(conn: sqlite3.Connection, scope_id: str) -> Optional[int]:
    """Best-effort epoch bump for auxiliary records (capture auths) whose
    scope list may name partitions without a ``scopes`` row yet."""
    require_id(scope_id, "scope_id")
    cur = conn.execute(
        "UPDATE scopes SET authz_revision = authz_revision + 1"
        " WHERE scope_id = ?",
        (scope_id,),
    )
    if cur.rowcount == 0:
        return None
    return current_epoch(conn, scope_id)


def check_epoch(
    conn: sqlite3.Connection, pinned: Optional[int], scope_id: str
) -> None:
    """Fence a pinned caller against superseded authorization state.

    ``None`` is an unversioned bind — the caller evaluates against current
    state. A pinned epoch must equal the scope's current ``authz_revision``:
    a mismatch means a revocation landed inside the caller's pinned window,
    so the call fails ``STALE_EPOCH`` and may retry after rebinding (§09.04,
    §46). Equality — not ``<`` — is required: a pin ahead of current is
    equally invalid and fails closed.
    """
    if pinned is None:
        return
    if isinstance(pinned, bool) or not isinstance(pinned, int) or pinned < 0:
        raise VerbatimError(
            ErrorCode.VALIDATION, "pinned epoch must be a non-negative int"
        )
    if pinned != current_epoch(conn, scope_id):
        raise VerbatimError(
            ErrorCode.STALE_EPOCH,
            "pinned authorization epoch superseded",
            retryable=True,
        )
