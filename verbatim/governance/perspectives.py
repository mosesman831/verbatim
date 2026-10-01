"""Perspective rows (SPEC_V3 §08.03–§08.07).

Each evidence item records asserter, subject(s), observer, and audience as
explicit roles — ``None`` means *not recorded*, never widened silently.
The ``perspectives`` row carries scalar roles plus the audience set; the
subjects live in ``perspective_subjects`` (one row per subject) so a
perspective is resolvable back into the frozen ``Perspective`` contract.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable, Optional

from ..core.types import new_id, require_id
from ..core.types_v3 import Perspective
from ..storage import repos_v3

_TABLE = "perspectives"
_SUBJECTS = "perspective_subjects"


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    """Order-preserving dedupe; every entry re-validated as an identifier."""
    seen: dict[str, None] = {}
    for v in values:
        seen[require_id(v, "subject/audience")] = None
    return tuple(seen)


def create_perspective(
    conn: sqlite3.Connection,
    *,
    scope_id: str,
    asserter: Optional[str] = None,
    observer: Optional[str] = None,
    subjects: Iterable[str] = (),
    audience: Iterable[str] = (),
    perspective_id: Optional[str] = None,
    created_event: int = 0,
) -> str:
    """Persist one perspective row plus its subject rows; returns its id.

    Validation rides the frozen ``Perspective`` contract so role ids are
    checked exactly once, here.
    """
    p = Perspective(
        asserter=asserter,
        observer=observer,
        subjects=_dedupe(subjects),
        audience=_dedupe(audience),
    )
    require_id(scope_id, "scope_id")
    pid = perspective_id or f"persp:{new_id()}"
    require_id(pid, "perspective_id")
    repos_v3.insert(
        conn,
        _TABLE,
        {
            "perspective_id": pid,
            "scope_id": scope_id,
            "asserter": p.asserter,
            "observer": p.observer,
            "audience_json": sorted(p.audience),
            "created_event": int(created_event),
        },
    )
    for subject in p.subjects:
        repos_v3.insert(
            conn,
            _SUBJECTS,
            {"perspective_id": pid, "subject_id": subject},
        )
    return pid


def subjects_of(
    conn: sqlite3.Connection, perspective_id: str
) -> tuple[str, ...]:
    """Ordered subject ids recorded on the perspective."""
    require_id(perspective_id, "perspective_id")
    rows = repos_v3.query(
        conn, _SUBJECTS, {"perspective_id": perspective_id}, order="subject_id"
    )
    return tuple(r["subject_id"] for r in rows)


def audience_of(
    conn: sqlite3.Connection, perspective_id: str
) -> tuple[str, ...]:
    """Audience set stored on the row; empty means none recorded."""
    row = repos_v3.get(conn, _TABLE, {"perspective_id": perspective_id})
    if row is None:
        return ()
    audience = repos_v3.json_field(row, "audience_json") or []
    return tuple(audience)


def resolve_perspective(
    conn: sqlite3.Connection, perspective_id: str
) -> Optional[Perspective]:
    """Rebuild the frozen Perspective contract from its rows."""
    require_id(perspective_id, "perspective_id")
    row = repos_v3.get(conn, _TABLE, {"perspective_id": perspective_id})
    if row is None:
        return None
    return Perspective(
        asserter=row.get("asserter"),
        observer=row.get("observer"),
        subjects=subjects_of(conn, perspective_id),
        audience=tuple(repos_v3.json_field(row, "audience_json") or ()),
    )


def find_perspective(
    conn: sqlite3.Connection,
    *,
    scope_id: str,
    asserter: Optional[str] = None,
    observer: Optional[str] = None,
    subjects: Iterable[str] = (),
    audience: Iterable[str] = (),
) -> Optional[str]:
    """Exact-match lookup for an identical role set (dedupe helper).

    Returns the existing perspective_id or None — callers reuse the same row
    instead of persisting duplicate role records (§08.04 keeps perspectives
    distinct records, not duplicated ones).
    """
    want_subjects = frozenset(_dedupe(subjects))
    want_audience = sorted(_dedupe(audience))
    candidates = repos_v3.query(
        conn,
        _TABLE,
        {"scope_id": scope_id, "asserter": asserter, "observer": observer},
    )
    for row in candidates:
        if sorted(repos_v3.json_field(row, "audience_json") or []) != want_audience:
            continue
        if frozenset(subjects_of(conn, row["perspective_id"])) != want_subjects:
            continue
        return row["perspective_id"]
    return None


def get_or_create_perspective(
    conn: sqlite3.Connection,
    *,
    scope_id: str,
    asserter: Optional[str] = None,
    observer: Optional[str] = None,
    subjects: Iterable[str] = (),
    audience: Iterable[str] = (),
    created_event: int = 0,
) -> str:
    """Resolve-or-insert: identical roles share one persisted row."""
    found = find_perspective(
        conn,
        scope_id=scope_id,
        asserter=asserter,
        observer=observer,
        subjects=subjects,
        audience=audience,
    )
    if found is not None:
        return found
    return create_perspective(
        conn,
        scope_id=scope_id,
        asserter=asserter,
        observer=observer,
        subjects=subjects,
        audience=audience,
        created_event=created_event,
    )


def perspectives_for_scope(
    conn: sqlite3.Connection, scope_id: str
) -> list[dict]:
    """All perspective rows in a scope (dict snapshots)."""
    require_id(scope_id, "scope_id")
    return repos_v3.query(conn, _TABLE, {"scope_id": scope_id})
