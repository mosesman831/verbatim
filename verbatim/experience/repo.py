"""Experience-layer row helpers: SQL not covered by ``storage/repos_v2.py``.

These functions stay inside the experience package so the shared v2
repositories remain untouched. Mutations take the caller's transaction
``conn`` exactly like the repos do; nothing here opens its own transaction
(SPEC_V2 §06: repositories participate in caller-owned transactions).
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional, Sequence

from ..core.types import ErrorCode, VerbatimError, require_id
from ..storage.repos import _next_event_seq, _require_int, _row, _rows

# Statuses that mean "still open" for a prospective record. Terminal
# statuses (completed/cancelled) are never auto-overdue (V2-22.05/06).
OPEN_PLAN_STATUSES = ("planned", "in_progress")
TERMINAL_PLAN_STATUSES = ("completed", "cancelled")
PLAN_STATUSES = frozenset(
    {"planned", "in_progress", "completed", "cancelled", "overdue", "unknown"}
)


def episode_row(conn: sqlite3.Connection, episode_id: str) -> Optional[dict[str, Any]]:
    """Fetch the episode row inside the caller's transaction."""
    require_id(episode_id, "episode_id")
    return _row(
        conn.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        )
    )


def close_episode_row(conn: sqlite3.Connection, episode_id: str) -> int:
    """Close the episode's recorded interval; returns the closing event seq.

    Idempotent: an already-closed episode returns its existing
    ``recorded_until``. A missing episode raises NOT_FOUND_OR_FORBIDDEN so
    callers cannot silently "close" something they cannot see.
    """
    row = episode_row(conn, episode_id)
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_FORBIDDEN, "episode not found"
        )
    if row["recorded_until"] is not None:
        return int(row["recorded_until"])
    seq = _next_event_seq(conn)
    conn.execute(
        "UPDATE episodes SET recorded_until = ?, row_version = row_version + 1"
        " WHERE episode_id = ? AND recorded_until IS NULL",
        (seq, episode_id),
    )
    return seq


def procedure_row(conn: sqlite3.Connection, procedure_id: str) -> Optional[dict[str, Any]]:
    """Fetch the procedure row inside the caller's transaction."""
    require_id(procedure_id, "procedure_id")
    return _row(
        conn.execute(
            "SELECT * FROM procedures WHERE procedure_id = ?",
            (procedure_id,),
        )
    )


def prospective_row(conn: sqlite3.Connection, record_id: str) -> Optional[dict[str, Any]]:
    """Fetch the prospective record inside the caller's transaction."""
    require_id(record_id, "record_id")
    return _row(
        conn.execute(
            "SELECT * FROM prospective_records WHERE record_id = ?",
            (record_id,),
        )
    )


def mark_overdue(conn: sqlite3.Connection, scope_id: str, now_us: int) -> int:
    """Mark still-open records whose due time passed; returns rows changed.

    Passing a due date only ever yields ``overdue`` — never success,
    deletion, or contradiction (V2-22.06).
    """
    require_id(scope_id, "scope_id")
    _require_int(now_us, "now_us")
    marks = ",".join("?" for _ in OPEN_PLAN_STATUSES)
    cur = conn.execute(
        "UPDATE prospective_records SET status = 'overdue'"
        " WHERE scope_id = ? AND due_us IS NOT NULL AND due_us < ?"
        f" AND status IN ({marks}) AND recorded_until IS NULL",
        (scope_id, now_us, *OPEN_PLAN_STATUSES),
    )
    return cur.rowcount


def due_rows(
    conn: sqlite3.Connection,
    scope_id: str,
    before_us: int,
    statuses: Sequence[str],
) -> list[dict[str, Any]]:
    """Prospective rows due at or before ``before_us`` in ``statuses``."""
    require_id(scope_id, "scope_id")
    _require_int(before_us, "before_us")
    marks = ",".join("?" for _ in statuses)
    return _rows(
        conn.execute(
            "SELECT * FROM prospective_records"
            " WHERE scope_id = ? AND due_us IS NOT NULL AND due_us <= ?"
            f" AND status IN ({marks}) AND recorded_until IS NULL"
            " ORDER BY due_us, record_id",
            (scope_id, before_us, *statuses),
        )
    )


def outcomes_for(
    conn: sqlite3.Connection, procedure_id: str
) -> list[dict[str, Any]]:
    """Outcome receipts for one procedure, deterministically ordered.

    ``recorded_us`` is the receipt's logical clock; ``rowid`` breaks ties
    so equal timestamps still replay in insertion order.
    """
    require_id(procedure_id, "procedure_id")
    return _rows(
        conn.execute(
            "SELECT * FROM outcome_receipts WHERE procedure_id = ?"
            " ORDER BY recorded_us, rowid",
            (procedure_id,),
        )
    )


def plans_for_scope_rows(
    conn: sqlite3.Connection,
    scope_id: str,
    statuses: Optional[Sequence[str]],
) -> list[dict[str, Any]]:
    """Current prospective records for a scope, optionally status-filtered."""
    require_id(scope_id, "scope_id")
    if statuses:
        marks = ",".join("?" for _ in statuses)
        return _rows(
            conn.execute(
                "SELECT * FROM prospective_records"
                f" WHERE scope_id = ? AND status IN ({marks})"
                " AND recorded_until IS NULL ORDER BY due_us, record_id",
                (scope_id, *statuses),
            )
        )
    return _rows(
        conn.execute(
            "SELECT * FROM prospective_records"
            " WHERE scope_id = ? AND recorded_until IS NULL"
            " ORDER BY due_us, record_id",
            (scope_id,),
        )
    )
