"""Grant/delegation revocation (SPEC_V3 §09.04–§09.05, §10.05).

Revocation sets ``revoked_us`` and bumps the scope's ``authz_revision`` in
the same transaction — pinned callers fence with ``STALE_EPOCH`` on their
next call, and revoked verbs leave the effective context before any
dereference (§09.05). A child never outlives its parent's revocation:
killing a grant or a delegation edge cascades to every child whose
remaining delegation chain no longer reaches a live root.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from ..core.time import now_us
from ..core.types import ErrorCode, VerbatimError, require_id
from ..storage import repos_v3
from . import epochs
from .grants import DENIAL_MESSAGE, _chain_live, get_grant

_GRANTS = "grants_v3"
_DELEGATIONS = "delegations"


def _deny() -> None:
    raise VerbatimError(ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, DENIAL_MESSAGE)


def _live(row: Optional[dict], now: int) -> bool:
    return (
        row is not None
        and row.get("revoked_us") is None
        and (row.get("expires_us") is None or row["expires_us"] > now)
    )


def _has_live_parent_path(
    conn: sqlite3.Connection, child_row: dict, now: int
) -> bool:
    """Any remaining unrevoked, unexpired delegation edge to a live root."""
    pinned = epochs.current_epoch(conn, child_row["scope_id"])
    return _chain_live(conn, child_row, now, pinned)


def _revoke_grant_inner(
    conn: sqlite3.Connection,
    grant_row: dict,
    now: int,
    bumped: dict[str, int],
    seen: set,
) -> dict:
    """Mark one grant dead, bump its scope epoch once, cascade to children
    whose chain is now broken. Returns running totals."""
    gid = grant_row["grant_id"]
    if gid in seen:  # pathological delegation cycle — stop descending
        return {"grants": 0, "delegations": 0}
    seen.add(gid)
    repos_v3.update(conn, _GRANTS, {"revoked_us": now}, {"grant_id": gid})
    sid = grant_row["scope_id"]
    if sid not in bumped:
        bumped[sid] = epochs.bump_epoch(conn, sid)
    totals = {"grants": 1, "delegations": 0}
    # Outgoing edges die with the parent; children re-check their chains.
    links = repos_v3.query(
        conn, _DELEGATIONS, {"parent_grant_id": gid, "revoked_us": None}
    )
    for link in links:
        repos_v3.update(
            conn,
            _DELEGATIONS,
            {"revoked_us": now},
            {"delegation_id": link["delegation_id"]},
        )
        totals["delegations"] += 1
        child = repos_v3.get(
            conn, _GRANTS, {"grant_id": link["child_grant_id"]}
        )
        if _live(child, now) and not _has_live_parent_path(conn, child, now):
            sub = _revoke_grant_inner(conn, child, now, bumped, seen)
            totals["grants"] += sub["grants"]
            totals["delegations"] += sub["delegations"]
    return totals


def revoke_grant(
    conn: sqlite3.Connection,
    grant_id: str,
    *,
    revoked_us: Optional[int] = None,
) -> dict:
    """Revoke a grant and cascade to children that lose their last live
    delegation path.

    Returns a receipt dict (idempotent — re-revoking reports
    ``revoked=False`` with no extra epoch bump). A missing grant raises the
    same NOT_FOUND_OR_UNAUTHORIZED as any denied access (§09.09).
    """
    require_id(grant_id, "grant_id")
    row = get_grant(conn, grant_id)
    if row is None:
        _deny()
    if row.get("revoked_us") is not None:
        return {
            "grant_id": grant_id,
            "scope_id": row["scope_id"],
            "principal_id": row["principal_id"],
            "revoked": False,
            "already_revoked": True,
            "epoch": epochs.current_epoch(conn, row["scope_id"]),
            "grants_revoked": 0,
            "delegations_revoked": 0,
        }
    now = revoked_us if revoked_us is not None else now_us()
    bumped: dict[str, int] = {}
    totals = _revoke_grant_inner(conn, row, now, bumped, set())
    return {
        "grant_id": grant_id,
        "scope_id": row["scope_id"],
        "principal_id": row["principal_id"],
        "revoked": True,
        "already_revoked": False,
        "epoch": bumped[row["scope_id"]],
        "epochs": bumped,
        "grants_revoked": totals["grants"],
        "delegations_revoked": totals["delegations"],
    }


def revoke_delegation(
    conn: sqlite3.Connection,
    delegation_id: str,
    *,
    revoked_us: Optional[int] = None,
) -> dict:
    """Revoke one delegation edge; its child grant dies with it when no
    other live parent path remains (a child never outlives the revocation
    of its last delegation chain)."""
    require_id(delegation_id, "delegation_id")
    link = repos_v3.get(
        conn, _DELEGATIONS, {"delegation_id": delegation_id}
    )
    if link is None:
        _deny()
    child = repos_v3.get(conn, _GRANTS, {"grant_id": link["child_grant_id"]})
    scope_id = child["scope_id"] if child is not None else None
    if link.get("revoked_us") is not None:
        return {
            "delegation_id": delegation_id,
            "scope_id": scope_id,
            "delegate_id": link["delegate_id"],
            "revoked": False,
            "already_revoked": True,
            "epoch": (
                epochs.current_epoch(conn, scope_id)
                if scope_id is not None
                else 0
            ),
            "grants_revoked": 0,
        }
    now = revoked_us if revoked_us is not None else now_us()
    repos_v3.update(
        conn,
        _DELEGATIONS,
        {"revoked_us": now},
        {"delegation_id": delegation_id},
    )
    bumped: dict[str, int] = {}
    grants_revoked = 0
    if scope_id is not None:
        bumped[scope_id] = epochs.bump_epoch(conn, scope_id)
        if _live(child, now) and not _has_live_parent_path(conn, child, now):
            totals = _revoke_grant_inner(conn, child, now, bumped, set())
            grants_revoked = totals["grants"]
    return {
        "delegation_id": delegation_id,
        "scope_id": scope_id,
        "delegate_id": link["delegate_id"],
        "revoked": True,
        "already_revoked": False,
        "epoch": bumped.get(scope_id, 0) if scope_id else 0,
        "epochs": bumped,
        "grants_revoked": grants_revoked,
    }
