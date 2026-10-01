"""Principal registry (SPEC_V3 §08).

Every call binds an authenticated principal; ``external_party`` rows are
subjects of facts and are never callers (§08.01). Registration is an
explicit operator record — display names and email-like strings never
establish identity (§08.09). Retirement marks the principal and revokes its
live grants so delegated authority cannot outlive the identity (§08.10).
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from ..core.time import now_us
from ..core.types import ErrorCode, VerbatimError, new_id, require_id
from ..core.types_v3 import PrincipalKind
from ..storage import repos_v3
from . import epochs

_TABLE = "principals"


def _kind(kind: "PrincipalKind | str") -> PrincipalKind:
    try:
        return kind if isinstance(kind, PrincipalKind) else PrincipalKind(kind)
    except ValueError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown principal kind {kind!r}"
        ) from exc


def register_principal(
    conn: sqlite3.Connection,
    *,
    kind: "PrincipalKind | str",
    principal_id: Optional[str] = None,
    display_name: Optional[str] = None,
    host_binding: Optional[str] = None,
    created_us: Optional[int] = None,
) -> str:
    """Register a principal; returns its id.

    Idempotent for an identical re-registration (same id, same kind). A
    conflicting kind for an existing id is a caller error — identities are
    never silently reinterpreted.
    """
    k = _kind(kind)
    pid = principal_id or f"principal:{new_id()}"
    require_id(pid, "principal_id")
    existing = repos_v3.get(conn, _TABLE, {"principal_id": pid})
    if existing is not None:
        if existing["kind"] != k.value:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"principal {pid!r} already registered as {existing['kind']!r}",
            )
        return pid
    repos_v3.insert(
        conn,
        _TABLE,
        {
            "principal_id": pid,
            "kind": k.value,
            "display_name": display_name,
            "host_binding": host_binding,
            "created_us": created_us if created_us is not None else now_us(),
            "retired": 0,
        },
    )
    return pid


def get_principal(
    conn: sqlite3.Connection, principal_id: str
) -> Optional[dict]:
    """Row snapshot or None; callers check ``retired`` themselves."""
    require_id(principal_id, "principal_id")
    return repos_v3.get(conn, _TABLE, {"principal_id": principal_id})


def is_retired(conn: sqlite3.Connection, principal_id: str) -> bool:
    """True only when a registered principal is marked retired."""
    row = repos_v3.get(conn, _TABLE, {"principal_id": principal_id})
    return row is not None and bool(row.get("retired"))


def list_principals(
    conn: sqlite3.Connection,
    *,
    kind: "PrincipalKind | str | None" = None,
    include_retired: bool = False,
) -> list[dict]:
    """Registry rows, optionally filtered by kind; retired excluded unless
    asked for (governance review needs them — §08.10)."""
    where = {"kind": _kind(kind).value} if kind is not None else None
    rows = repos_v3.query(conn, _TABLE, where, order="principal_id")
    if include_retired:
        return rows
    return [r for r in rows if not r.get("retired")]


def retire_principal(
    conn: sqlite3.Connection,
    principal_id: str,
    *,
    retired_us: Optional[int] = None,
    revoke_grants: bool = True,
) -> dict:
    """Retire a principal and fail closed on its remaining authority.

    Marks ``retired`` and — by default — revokes every live grant the
    principal holds, bumping each touched scope's ``authz_revision`` so
    pinned callers fence on their next call. Returns a small receipt dict;
    a second call is an idempotent no-op. This is the enforcement half of
    §08.10 (the governance-review workflow itself is a separate surface).
    """
    require_id(principal_id, "principal_id")
    row = repos_v3.get(conn, _TABLE, {"principal_id": principal_id})
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "not found or unauthorized"
        )
    if row.get("retired"):
        return {
            "principal_id": principal_id,
            "retired": False,
            "already_retired": True,
            "grants_revoked": 0,
        }
    now = retired_us if retired_us is not None else now_us()
    repos_v3.update(
        conn, _TABLE, {"retired": 1}, {"principal_id": principal_id}
    )
    revoked = 0
    bumped: dict[str, int] = {}
    if revoke_grants:
        live = repos_v3.query(
            conn,
            "grants_v3",
            {"principal_id": principal_id, "revoked_us": None},
        )
        for grant in live:
            repos_v3.update(
                conn,
                "grants_v3",
                {"revoked_us": now},
                {"grant_id": grant["grant_id"]},
            )
            revoked += 1
            sid = grant["scope_id"]
            if sid not in bumped:
                new = epochs.bump_epoch_if_present(conn, sid)
                if new is not None:
                    bumped[sid] = new
    return {
        "principal_id": principal_id,
        "retired": True,
        "already_retired": False,
        "grants_revoked": revoked,
        "scope_epochs": bumped,
    }
