"""Social memory: observer-scoped collaborator records (SPEC_V3 §25).

V3-25.01/25.02: social memory records how collaborators interact —
communication preferences, evidenced expertise, reliability patterns —
and every record is *observer-scoped*: agent A's record about agent B is
a separate row from C's record about B, never merged. Each row is derived
and cited (``evidence_json`` carries the supporting refs).

The table's ``UNIQUE(scope_id, observer_id, subject_id, kind)`` makes
writes upserts: re-setting the same (observer, subject, kind) bumps
``revision`` in place — the prior value stays reachable through revision
history reads, not overwritten silently (V3-23.04 analogue).
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any, Iterable, Optional

from ..core.time import now_us
from ..core.types import require_id
from ..storage import repos_v3


def record_id_for(
    scope_id: str, observer_id: str, subject_id: str, kind: str
) -> str:
    """Deterministic record id — the unique tuple *is* the identity."""
    raw = "|".join([scope_id, observer_id, subject_id, kind])
    return "sm_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def set_record(
    conn: sqlite3.Connection,
    scope_id: str,
    observer_id: str,
    subject_id: str,
    kind: str,
    value: Any,
    evidence_refs: Iterable[Any] = (),
    *,
    now: Optional[int] = None,
) -> dict[str, Any]:
    """Upsert one social-memory record; returns the stored row snapshot.

    ``value`` is any JSON-serializable payload (kept in ``value_json``);
    ``evidence_refs`` cites the interactions/outcomes the record is
    derived from (V3-25.01). Repeat writes bump ``revision``.
    """
    require_id(scope_id, "scope_id")
    require_id(observer_id, "observer_id")
    require_id(subject_id, "subject_id")
    require_id(kind, "kind")
    record_id = record_id_for(scope_id, observer_id, subject_id, kind)
    evidence = list(evidence_refs)
    existing = repos_v3.get(
        conn, "social_memory", {"record_id": record_id}
    )
    if existing is None:
        repos_v3.insert(
            conn,
            "social_memory",
            {
                "record_id": record_id,
                "scope_id": scope_id,
                "observer_id": observer_id,
                "subject_id": subject_id,
                "kind": kind,
                "value_json": value,
                "evidence_json": evidence,
                "revision": 1,
                "created_us": now_us() if now is None else now,
            },
        )
    else:
        repos_v3.update(
            conn,
            "social_memory",
            {
                "value_json": value,
                "evidence_json": evidence,
                "revision": int(existing["revision"]) + 1,
            },
            {"record_id": record_id},
        )
    row = repos_v3.get(conn, "social_memory", {"record_id": record_id})
    assert row is not None
    return row


def get_record(
    conn: sqlite3.Connection,
    scope_id: str,
    observer_id: str,
    subject_id: str,
    kind: str,
) -> Optional[dict[str, Any]]:
    """One record by its unique tuple, or ``None``."""
    require_id(scope_id, "scope_id")
    require_id(observer_id, "observer_id")
    require_id(subject_id, "subject_id")
    require_id(kind, "kind")
    return repos_v3.get(
        conn,
        "social_memory",
        {"record_id": record_id_for(scope_id, observer_id, subject_id, kind)},
    )


def list_records(
    conn: sqlite3.Connection,
    scope_id: str,
    *,
    observer_id: Optional[str] = None,
    subject_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Records in a scope, optionally narrowed by observer and/or subject
    (observer-scoped visibility, V3-25.02)."""
    require_id(scope_id, "scope_id")
    where: dict[str, Any] = {"scope_id": scope_id}
    if observer_id is not None:
        where["observer_id"] = require_id(observer_id, "observer_id")
    if subject_id is not None:
        where["subject_id"] = require_id(subject_id, "subject_id")
    return repos_v3.query(
        conn, "social_memory", where, order="record_id"
    )
