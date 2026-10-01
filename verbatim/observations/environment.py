"""Environment state: host-observed volatile facts (SPEC_V3 §24, V3-24.04).

``environment_state`` rows record host-observed values (versions, paths,
configuration, endpoints) bound to optional state anchors. When a
``volatile`` key's value *changes*, every freshness-anchored object
referencing that anchor is marked stale (``freshness.mark_anchored_stale``)
— the stale marker routes them to ``verify``, it never deletes or
contradicts them (V3-18.11, V3-24.05).

Rewriting an identical value refreshes ``observed_us`` without marking
anything: only a change moves the anchor.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

from ..core.time import now_us
from ..core.types import ErrorCode, VerbatimError, require_id
from ..storage import repos_v3
from . import freshness as _freshness


def set_state(
    conn: sqlite3.Connection,
    scope_id: str,
    key: str,
    value: str,
    *,
    anchor_id: Optional[str] = None,
    volatile: bool = True,
    observed_us: Optional[int] = None,
    seq: Optional[int] = None,
) -> dict[str, Any]:
    """Upsert one environment value; returns the row plus change metadata.

    On a *changed* volatile value with an ``anchor_id``, dependent objects
    (freshness rows whose anchor list contains the anchor) are marked
    stale at ``seq``. ``{"changed", "marked", "seq"}`` report what the
    update did so callers/tests can observe idempotence: an unchanged
    rewrite marks nothing.
    """
    require_id(scope_id, "scope_id")
    require_id(key, "key")
    if not isinstance(value, str) or not value:
        raise VerbatimError(ErrorCode.VALIDATION, "value must be non-empty text")
    if anchor_id is not None:
        require_id(anchor_id, "anchor_id")
    if seq is None:
        seq = _freshness.next_seq(conn, scope_id)
    at = now_us() if observed_us is None else observed_us

    existing = repos_v3.get(
        conn, "environment_state", {"scope_id": scope_id, "key": key}
    )
    changed = (
        existing is None
        or existing["value"] != value
        or (existing["anchor_id"] or None) != anchor_id
    )
    payload = {
        "scope_id": scope_id,
        "key": key,
        "value": value,
        "anchor_id": anchor_id,
        "observed_us": at,
        "volatile": 1 if volatile else 0,
    }
    if existing is None:
        repos_v3.insert(conn, "environment_state", payload)
    else:
        repos_v3.update(
            conn,
            "environment_state",
            {
                "value": value,
                "anchor_id": anchor_id,
                "observed_us": at,
                "volatile": 1 if volatile else 0,
            },
            {"scope_id": scope_id, "key": key},
        )

    marked = 0
    if changed and volatile and anchor_id is not None:
        marked = _freshness.mark_anchored_stale(
            conn, scope_id, anchor_id, seq
        )
    return {**payload, "changed": changed, "marked": marked, "seq": seq}


def get_state(
    conn: sqlite3.Connection, scope_id: str, key: str
) -> Optional[dict[str, Any]]:
    """One environment value row, or ``None``."""
    require_id(scope_id, "scope_id")
    require_id(key, "key")
    return repos_v3.get(
        conn, "environment_state", {"scope_id": scope_id, "key": key}
    )


def list_state(conn: sqlite3.Connection, scope_id: str) -> list[dict[str, Any]]:
    """All environment rows in a scope, key-ordered (§24.06 bounded view)."""
    require_id(scope_id, "scope_id")
    return repos_v3.query(
        conn, "environment_state", {"scope_id": scope_id}, order="key"
    )


def clear_state(
    conn: sqlite3.Connection,
    scope_id: str,
    key: str,
    *,
    seq: Optional[int] = None,
) -> bool:
    """Delete one environment row; returns True when a row existed.

    Removing a volatile anchored value is itself a state change — anchored
    objects are marked stale exactly as on a value move.
    """
    require_id(scope_id, "scope_id")
    require_id(key, "key")
    row = repos_v3.get(
        conn, "environment_state", {"scope_id": scope_id, "key": key}
    )
    if row is None:
        return False
    repos_v3.delete(
        conn, "environment_state", {"scope_id": scope_id, "key": key}
    )
    if row["volatile"] and row["anchor_id"]:
        _freshness.mark_anchored_stale(
            conn, scope_id, row["anchor_id"], seq
        )
    return True
