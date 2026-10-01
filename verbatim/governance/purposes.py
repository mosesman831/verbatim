"""Purpose registry (SPEC_V3 §11.06, §09.08).

Purpose strings come from a registry — free-text purposes are rejected so
purpose limitation is enforceable. Grants bind registry entries; a caller's
declaration is checked against them, never invented on the fly.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from ..core.time import now_us
from ..core.types import ErrorCode, VerbatimError, require_id
from ..storage import repos_v3

# Built-in purposes seeded into every governed store. ``evaluate`` covers
# offline harness work; the rest map to §09 verbs and §11 flows.
BUILTIN_PURPOSES: tuple[str, ...] = (
    "recall",
    "derive",
    "share",
    "hydrate",
    "review",
    "ingest",
    "admin",
    "evaluate",
)

_DESCRIPTIONS = {
    "recall": "read/quote evidence and derived objects for a caller",
    "derive": "create learning-plane objects from evidence",
    "share": "re-disclose to another principal within attenuation limits",
    "hydrate": "redeem exact vault values under matching consent",
    "review": "approve/reject transitions, quarantine, promotions",
    "ingest": "write new evidence into a scope",
    "admin": "grants, retention, deletion policy, profile changes",
    "evaluate": "offline measurement and conformance runs",
}


def seed_purposes(
    conn: sqlite3.Connection, *, registered_us: Optional[int] = None
) -> int:
    """Insert the built-in purposes; returns how many were newly added.

    Idempotent — re-seeding is a no-op and never resets ``retired``.
    """
    now = registered_us if registered_us is not None else now_us()
    added = 0
    for purpose in BUILTIN_PURPOSES:
        if repos_v3.get(conn, "purposes", {"purpose": purpose}) is None:
            repos_v3.insert(
                conn,
                "purposes",
                {
                    "purpose": purpose,
                    "description": _DESCRIPTIONS.get(purpose, ""),
                    "registered_us": now,
                    "retired": 0,
                },
            )
            added += 1
    return added


def register_purpose(
    conn: sqlite3.Connection,
    purpose: str,
    description: str = "",
    *,
    registered_us: Optional[int] = None,
) -> None:
    """Register an additional purpose; idempotent for an identical row."""
    require_id(purpose, "purpose")
    existing = repos_v3.get(conn, "purposes", {"purpose": purpose})
    if existing is not None:
        return
    repos_v3.insert(
        conn,
        "purposes",
        {
            "purpose": purpose,
            "description": description,
            "registered_us": (
                registered_us if registered_us is not None else now_us()
            ),
            "retired": 0,
        },
    )


def retire_purpose(conn: sqlite3.Connection, purpose: str) -> bool:
    """Retire a registry entry; existing grants keep their stored string but
    new grants can no longer bind it. Returns True when a live row retired."""
    require_id(purpose, "purpose")
    row = repos_v3.get(conn, "purposes", {"purpose": purpose})
    if row is None or row.get("retired"):
        return False
    repos_v3.update(
        conn, "purposes", {"retired": 1}, {"purpose": purpose}
    )
    return True


def is_registered(conn: sqlite3.Connection, purpose: str) -> bool:
    """True when ``purpose`` is a live registry entry."""
    row = repos_v3.get(conn, "purposes", {"purpose": purpose})
    return row is not None and not row.get("retired")


def require_purpose(conn: sqlite3.Connection, purpose: str) -> None:
    """Enforce registry membership (§11.06); VALIDATION when unregistered."""
    if not isinstance(purpose, str) or not purpose:
        raise VerbatimError(ErrorCode.VALIDATION, "purpose must be a string")
    if not is_registered(conn, purpose):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"unregistered purpose {purpose!r} — purposes come from the registry",
        )


def list_purposes(
    conn: sqlite3.Connection, *, include_retired: bool = False
) -> list[dict]:
    """Registry rows; dict snapshots per the repos contract."""
    rows = repos_v3.query(conn, "purposes", order="purpose")
    if include_retired:
        return rows
    return [r for r in rows if not r.get("retired")]
