"""Prospective memory: plans, commitments, deadlines (SPEC_V2 §22).

A prospective record is an *intention* — a plan, commitment, deadline, or
recurring intent — never a completed fact. Passing a due date marks the
record ``overdue``; it never marks it successful, false, or deletable
(V2-22.06). Completion and cancellation are explicit authorized updates
(V2-22.05). Recurrence uses deterministic calendar semantics only —
anything the engine cannot interpret stays unresolved (V2-22.07).
"""

from __future__ import annotations

import calendar
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from ..core.types import ErrorCode, VerbatimError, require_id
from ..storage.repos import _json_parse, _require_int, _require_str
from ..storage.repos_v2 import ProspectiveRepo
from ..storage.store import Store
from . import repo as _repo

_DAY_US = 86_400 * 1_000_000

# Deterministic recurrence frequencies; anything else stays unresolved.
_FREQS = ("daily", "weekly", "monthly")
_MAX_MONTH_STEPS = 2400  # bounded iteration: 200 years of monthly steps

# Status transitions. Terminal states absorb nothing further — a
# completed/cancelled plan can only be read, never quietly reopened by
# the service layer (reopening needs new evidence → a new record).
_STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    "planned": frozenset({"in_progress", "completed", "cancelled", "overdue"}),
    "in_progress": frozenset({"planned", "completed", "cancelled", "overdue"}),
    "overdue": frozenset({"planned", "in_progress", "completed", "cancelled"}),
    "unknown": frozenset({"planned", "in_progress", "completed", "cancelled", "overdue"}),
    "completed": frozenset(),
    "cancelled": frozenset(),
}


def _validate_recurrence(recurrence: Any) -> None:
    """Only deterministic schedule forms are persisted (V2-22.07)."""
    if recurrence is None:
        return
    if not isinstance(recurrence, dict):
        raise VerbatimError(ErrorCode.VALIDATION, "recurrence must be a dict")
    freq = recurrence.get("freq")
    if freq not in _FREQS:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unsupported recurrence freq {freq!r}"
        )
    interval = recurrence.get("interval", 1)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise VerbatimError(
            ErrorCode.VALIDATION, "recurrence interval must be an int >= 1"
        )
    anchor = recurrence.get("anchor_us")
    if anchor is not None:
        _require_int(anchor, "anchor_us")


def plan(
    store: Store,
    conn: sqlite3.Connection,
    scope_id: str,
    owner_id: str,
    intention_text: str,
    *,
    due_us: Optional[int] = None,
    recurrence: Any = None,
    claim_id: Optional[str] = None,
    episode_id: Optional[str] = None,
) -> str:
    """Record a plan/commitment inside the caller's tx; returns record_id.

    ``claim_id``/``episode_id`` link the intention to its source evidence
    (V2-22.04); the record keeps status ``planned`` — it does not assert
    that anything happened.
    """
    _require_str(intention_text, "intention_text")
    if due_us is not None:
        _require_int(due_us, "due_us")
    _validate_recurrence(recurrence)
    return ProspectiveRepo(store).create(
        conn,
        scope_id,
        owner_id,
        intention_text,
        due_us=due_us,
        recurrence=recurrence,
        claim_id=claim_id,
        episode_id=episode_id,
    )


def due(
    store: Store, scope_id: str, now_us: int, *, limit: int = 256
) -> list[dict[str, Any]]:
    """Due/overdue records; marks newly-overdue rows inside one tx.

    Bounded query (V2-22.08): returns records with ``due_us <= now_us``
    whose status is still open or already overdue, ordered by due time.
    Records that just passed their due time are set to ``overdue`` in the
    same transaction — elapsed time alone never completes them.
    """
    require_id(scope_id, "scope_id")
    _require_int(now_us, "now_us")
    _require_int(limit, "limit", minimum=1)
    with store.tx() as conn:
        _repo.mark_overdue(conn, scope_id, now_us)
        rows = _repo.due_rows(
            conn,
            scope_id,
            now_us,
            _repo.OPEN_PLAN_STATUSES + ("overdue",),
        )
    out = rows[:limit]
    for r in out:
        r["kind"] = "plan"
        r["is_fact"] = False
    return out


def _transition(
    store: Store,
    conn: sqlite3.Connection,
    record_id: str,
    target: str,
) -> None:
    row = _repo.prospective_row(conn, record_id)
    if row is None or row["recorded_until"] is not None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_FORBIDDEN, "prospective record not found"
        )
    current = row["status"]
    if current == target:
        return
    if target not in _STATUS_TRANSITIONS.get(current, frozenset()):
        raise VerbatimError(
            ErrorCode.INVALID_TRANSITION,
            f"plan status {current} -> {target} is not allowed",
        )
    ProspectiveRepo(store).set_status(conn, record_id, target)


def complete(store: Store, conn: sqlite3.Connection, record_id: str) -> None:
    """Explicit completion — the authorized update, not elapsed time."""
    _transition(store, conn, record_id, "completed")


def cancel(store: Store, conn: sqlite3.Connection, record_id: str) -> None:
    """Explicit cancellation; the historical record remains queryable."""
    _transition(store, conn, record_id, "cancelled")


def start(store: Store, conn: sqlite3.Connection, record_id: str) -> None:
    """Mark a planned/overdue record in progress."""
    _transition(store, conn, record_id, "in_progress")


def reschedule(
    store: Store,
    conn: sqlite3.Connection,
    record_id: str,
    new_due_us: int,
) -> None:
    """Authorized reschedule: back to ``planned`` with a new due time."""
    _require_int(new_due_us, "new_due_us")
    row = _repo.prospective_row(conn, record_id)
    if row is None or row["recorded_until"] is not None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_FORBIDDEN, "prospective record not found"
        )
    if row["status"] in _repo.TERMINAL_PLAN_STATUSES:
        raise VerbatimError(
            ErrorCode.INVALID_TRANSITION,
            f"plan status {row['status']} cannot be rescheduled",
        )
    conn.execute(
        "UPDATE prospective_records SET due_us = ?, status = 'planned'"
        " WHERE record_id = ? AND recorded_until IS NULL",
        (new_due_us, record_id),
    )


def _add_months(dt: datetime, months: int) -> datetime:
    total = dt.year * 12 + (dt.month - 1) + months
    year, month0 = divmod(total, 12)
    month = month0 + 1
    last_day = calendar.monthrange(year, month)[1]
    return dt.replace(year=year, month=month, day=min(dt.day, last_day))


def recurrence_next(recurrence: Any, after_us: int) -> Optional[int]:
    """Next scheduled instant strictly after ``after_us``; None if unresolved.

    ``recurrence`` shape: ``{"freq": "daily"|"weekly"|"monthly",
    "interval": n (default 1), "anchor_us": optional first occurrence}``.
    Daily/weekly are fixed microsecond steps; monthly is calendar-month
    arithmetic with day clamping (e.g. Jan 31 → Feb 28). An unrecognized
    or malformed schedule returns None — it stays unresolved rather than
    guessing (V2-22.07).
    """
    _require_int(after_us, "after_us")
    if not isinstance(recurrence, dict):
        return None
    freq = recurrence.get("freq")
    if freq not in _FREQS:
        return None
    interval = recurrence.get("interval", 1)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        return None
    anchor = recurrence.get("anchor_us")
    base = anchor if isinstance(anchor, int) and not isinstance(anchor, bool) else None

    if base is not None and base > after_us:
        return base
    if freq in ("daily", "weekly"):
        step = interval * _DAY_US * (1 if freq == "daily" else 7)
        origin = base if base is not None else after_us
        n = (after_us - origin) // step + 1
        return origin + n * step
    # monthly: calendar stepping from the anchor (or from after_us)
    origin = base if base is not None else after_us
    cur = datetime.fromtimestamp(origin / 1_000_000, tz=timezone.utc)
    for _ in range(_MAX_MONTH_STEPS):
        cur = _add_months(cur, interval)
        us = int(cur.timestamp() * 1_000_000)
        if us > after_us:
            return us
    return None


def plan_view(store: Store, record_id: str) -> dict[str, Any]:
    """Inspectable record — labeled a plan, never a completed fact (§22)."""
    row = ProspectiveRepo(store).get(record_id)
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_FORBIDDEN, "prospective record not found"
        )
    out = dict(row)
    out["kind"] = "plan"
    out["is_fact"] = False  # intention recorded; occurrence not asserted
    out["recurrence"] = _json_parse(row["recurrence_json"])
    return out


def plans_for_scope(
    store: Store,
    scope_id: str,
    *,
    statuses: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    """Current plans in a scope — MemoryKind.PLAN candidate source."""
    require_id(scope_id, "scope_id")
    if statuses is not None:
        bad = [s for s in statuses if s not in _repo.PLAN_STATUSES]
        if bad:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"unknown plan statuses {bad!r}"
            )
    with store.read() as conn:
        rows = _repo.plans_for_scope_rows(conn, scope_id, statuses)
    for r in rows:
        r["kind"] = "plan"
        r["is_fact"] = False
        r["recurrence"] = _json_parse(r["recurrence_json"])
    return rows
