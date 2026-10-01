"""V85-03.01 — effective turn position (read-time neighbor order).

Wherever V8 needs a turn's neighbors (context propagation/injection,
pack windows, delivery expansion), neighbors are defined by **effective
position**: a session's ``turn`` members ordered by

    (seq, occurred_start_us, recorded_at_us, byte_start, unit_id)

and indexed ``0 … n−1``; the neighbors of the member at position ``p``
are the members at positions ``[p − W, p + W] \\ {p}``.  This module is
the single authority for that ordering.

Why position instead of raw ``seq`` deltas: pre-V8.5 stores (and the
one-source-per-turn capture paths — Track R, the AMB provider) wrote
every turn unit with ``seq = 0`` inside its own source slice, so
``1 <= |Δseq| <= W`` never fired.  Ordering by the tuple above still
leads with ``seq`` — stores that carry real ordinals (V85-03.02's
write path) order identically — while degenerate ``seq`` stores fall
through to the time and byte pins.  No migration is needed: the read
side recomputes positions from live rows every query.

Contract:

- ``member_row`` normalizes the inventory row shapes the pipeline's
  neighbor fetch produces (mappings, or tuples of
  ``(unit_id, session_id, seq[, kind[, source_id[, revision[,
  occurred_start_us[, recorded_at_us[, byte_start]]]]]])``).  Trailing
  fields are optional so legacy 6-tuples keep parsing; absent ordering
  fields sort *last* (a turn with no time pin may not pretend an early
  position) and ``unit_id`` is the total-order floor — the comparator
  is total and deterministic.
- ``SessionIndex`` indexes an already-fetched inventory: deduped by
  ``unit_id``, one ordered member list per session, O(1) position and
  ±W window lookups.  ``eligible`` filters rows out of the index
  entirely — an ineligible member is never injected, never a
  propagation source, and holds no position (V8-06.02/§21.1: ``N(u)``
  ranges over *eligible* members only).
- ``fetch_session_members`` / ``session_ids_for`` are the one batched
  statement per chunk (``_IN_CHUNK`` headroom, the repo convention —
  V8-06.07's ≤2-statement budget is preserved: one candidate→sessions
  read plus one sessions→members read).
- Generation fencing matches the lanes: per ``unit_id`` the row at the
  latest ``generation <= pinned`` is the visible one.  ``kind='turn'``
  and the caller's ``scope_id`` fence the member set (V8-06.05 —
  sentence windows are never context members).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from typing import Any, Iterable, Optional

#: SQLite variable-limit headroom (repo convention — same bound the
#: pipeline's inventory chunking and the lexical lane use).
_IN_CHUNK = 400

_KIND_TURN = "turn"

#: Tuple layout ``fetch_session_members`` emits — the legacy six fields
#: first (positional consumers keep working), then the three ordering
#: fields V85-03.01 needs.
MEMBER_TUPLE_LEN = 9


def _ord_num(value: Any) -> tuple:
    """Ordering-tuple element: present numeric → ``(0, v)``; absent or
    non-numeric → ``(1, 0)`` so unscheduled/untimed members sort last,
    never first."""
    if value is None or isinstance(value, bool):
        return (1, 0)
    if isinstance(value, (int, float)):
        return (0, value)
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, 0)


def _ord_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def member_row(row: Any) -> Optional[dict]:
    """Normalize one session-member row to the canonical dict, or
    ``None`` when the row lacks a usable turn view.

    A row is usable when it carries ``unit_id`` + ``session_id`` and is
    a turn (``kind`` absent or ``'turn'``); ``seq`` may be ``None`` —
    the ordering tuple's later fields still position it (that tolerance
    is what repairs all-``seq=0`` stores).  Non-turn rows are dropped:
    sentence windows are children of turns and never context members
    (V8-06.05).
    """
    if isinstance(row, Mapping):
        uid = row.get("unit_id", row.get("uid"))
        session = row.get("session_id")
        seq = row.get("seq")
        kind = row.get("kind", _KIND_TURN)
        source_id = row.get("source_id", "")
        revision = row.get("revision", 0)
        occurred = row.get("occurred_start_us")
        recorded = row.get("recorded_at_us")
        byte_start = row.get("byte_start")
    elif isinstance(row, (list, tuple)) and len(row) >= 3:
        uid, session, seq = row[0], row[1], row[2]
        kind = row[3] if len(row) > 3 else _KIND_TURN
        source_id = row[4] if len(row) > 4 else ""
        revision = row[5] if len(row) > 5 else 0
        occurred = row[6] if len(row) > 6 else None
        recorded = row[7] if len(row) > 7 else None
        byte_start = row[8] if len(row) > 8 else None
    else:
        return None
    if kind is not None and str(kind) != _KIND_TURN:
        return None
    if uid is None or session is None:
        return None
    try:
        rev_i = int(revision)
    except (TypeError, ValueError):
        rev_i = 0
    out = {
        "unit_id": str(uid),
        "session_id": str(session),
        "seq": _ord_int(seq),
        "source_id": "" if source_id is None else str(source_id),
        "revision": rev_i,
        "occurred_start_us": _ord_int(occurred),
        "recorded_at_us": _ord_int(recorded),
        "byte_start": _ord_int(byte_start),
    }
    out["key"] = member_sort_key(out)
    return out


def member_sort_key(member: Any) -> tuple:
    """The V85-03.01 ordering tuple — ``(seq, occurred_start_us,
    recorded_at_us, byte_start, unit_id)``; each numeric field sorts
    absent-last so unlabeled members take the tail deterministically."""
    get = member.get if isinstance(member, Mapping) else lambda k, d=None: getattr(member, k, d)
    return (
        _ord_num(get("seq")),
        _ord_num(get("occurred_start_us")),
        _ord_num(get("recorded_at_us")),
        _ord_num(get("byte_start")),
        str(get("unit_id") or ""),
    )


def eligible_member(eligible: Any, unit_id: str) -> bool:
    """Fail-closed membership probe shared by the context stage and the
    index builder.  ``None`` means the caller guarantees the inventory
    is already eligibility-filtered; a callable/set gets a bare
    ``unit_id``; a verdict-side exception denies (never a silent pass —
    eligibility before ranking)."""
    if eligible is None:
        return True
    if callable(eligible):
        try:
            return bool(eligible(unit_id))
        except Exception:
            return False
    try:
        return unit_id in eligible
    except TypeError:
        return True


class SessionIndex:
    """Position index over one query's session-member inventory.

    ``sessions`` maps ``session_id`` → the ordered member dicts (each
    carries its ``pos``); ``pos_of`` maps ``unit_id`` →
    ``(session_id, position)`` for O(1) resolution.  Construction is a
    pure function of the rows: no clock, no store.
    """

    __slots__ = ("sessions", "pos_of", "ineligible_rows", "skipped_rows")

    def __init__(self, sessions: dict, pos_of: dict) -> None:
        self.sessions = sessions
        self.pos_of = pos_of
        #: rows filtered out by ``eligible`` before indexing (honest stat).
        self.ineligible_rows = 0
        #: rows that failed to normalize (missing unit_id/session_id or
        #: non-turn kind) — surfaced so silent inventory gaps are visible.
        self.skipped_rows = 0

    @classmethod
    def build(
        cls,
        rows: Iterable,
        *,
        eligible: Any = None,
    ) -> "SessionIndex":
        """Index a flat member inventory (``member_row``-parseable).

        Rows failing ``eligible`` are excluded *before* indexing — an
        ineligible turn holds no position (its neighbors close ranks
        around the gap rather than straddling it), matching §21.1's
        "eligible N(u) members".  Duplicate ``unit_id``s (adjacency
        flattening, double-listed rows) keep the first occurrence.
        """
        sessions: dict = {}
        seen: set = set()
        ineligible = 0
        skipped = 0
        for row in rows or ():
            rv = member_row(row)
            if rv is None:
                skipped += 1
                continue
            uid = rv["unit_id"]
            if uid in seen:
                continue
            if not eligible_member(eligible, uid):
                ineligible += 1
                continue
            seen.add(uid)
            sessions.setdefault(rv["session_id"], []).append(rv)
        pos_of: dict = {}
        for sid, members in sessions.items():
            members.sort(key=lambda m: m["key"])
            for pos, m in enumerate(members):
                m["pos"] = pos
                pos_of[m["unit_id"]] = (sid, pos)
        idx = cls(sessions, pos_of)
        idx.ineligible_rows = ineligible  # honest stats, read by callers
        idx.skipped_rows = skipped
        return idx

    def position(self, unit_id: Any) -> Optional[tuple]:
        """``(session_id, position)`` for a member, else ``None``."""
        return self.pos_of.get(str(unit_id)) if unit_id is not None else None

    def member(self, unit_id: Any) -> Optional[dict]:
        ent = self.position(unit_id)
        if ent is None:
            return None
        return self.sessions[ent[0]][ent[1]]

    def neighbors(self, unit_id: Any, W: int) -> list:
        """Members within ±``W`` positions of ``unit_id`` — ordered by
        position (session order), the unit itself excluded.  Unknown or
        unpositioned ids yield ``[]`` (honest empty, never invented)."""
        ent = self.position(unit_id)
        if ent is None or W <= 0:
            return []
        sid, pos = ent
        members = self.sessions[sid]
        lo = max(0, pos - W)
        hi = min(len(members), pos + W + 1)
        return [m for m in members[lo:hi] if m["unit_id"] != str(unit_id)]


# ---------------------------------------------------------------------------
# store reads — one batched statement per chunk (V8-06.07)
# ---------------------------------------------------------------------------

_MEMBER_COLS = (
    "u.unit_id", "u.session_id", "u.seq", "u.kind",
    "u.source_id", "u.revision",
    "u.occurred_start_us", "u.recorded_at_us", "u.byte_start",
)

#: Latest in-fence generation per unit — the same fencing shape the
#: lanes and the pipeline inventory use (SPEC_V8 §21.1 note).
_FENCE = (
    "u.generation = ("
    "   SELECT MAX(g.generation) FROM units g"
    "   WHERE g.unit_id = u.unit_id AND g.scope_id = u.scope_id"
    "     AND g.generation <= ?)"
)


def fetch_session_members(
    conn: sqlite3.Connection,
    scope_id: Any,
    session_ids: Iterable,
    generation: int,
) -> list:
    """Every visible ``kind='turn'`` member of the named sessions —
    one ``IN``-chunked statement per 400-id block, scope- and
    generation-fenced.  Rows are the 9-tuple layout
    ``MEMBER_TUPLE_LEN`` documents; callers index them with
    :meth:`SessionIndex.build`.  Propagates ``sqlite3.Error`` — a
    broken read must not masquerade as an empty session (the caller
    decides the honest-empty policy)."""
    sessions = sorted({str(s) for s in (session_ids or ()) if s is not None})
    if not sessions:
        return []
    cols = ", ".join(_MEMBER_COLS)
    rows: list = []
    for i in range(0, len(sessions), _IN_CHUNK):
        part = sessions[i : i + _IN_CHUNK]
        ph = ",".join("?" * len(part))
        rows.extend(
            conn.execute(
                f"SELECT {cols} FROM units u"
                f" WHERE u.scope_id = ? AND u.kind = 'turn'"
                f" AND u.session_id IN ({ph})"
                f" AND {_FENCE}",
                [scope_id, *part, generation],
            ).fetchall()
        )
    return rows


def session_ids_for(
    conn: sqlite3.Connection,
    scope_id: Any,
    unit_ids: Iterable,
    generation: int,
) -> list:
    """The sessions a candidate set touches — one ``IN``-chunked read
    of the candidates' own fenced rows (the first half of the V8-06.07
    inventory budget)."""
    ids = sorted({str(u) for u in (unit_ids or ()) if u})
    if not ids:
        return []
    out: set = set()
    for i in range(0, len(ids), _IN_CHUNK):
        part = ids[i : i + _IN_CHUNK]
        ph = ",".join("?" * len(part))
        for row in conn.execute(
            "SELECT DISTINCT u.session_id FROM units u"
            f" WHERE u.scope_id = ? AND u.unit_id IN ({ph})"
            f" AND {_FENCE}",
            [scope_id, *part, generation],
        ).fetchall():
            if row[0] is not None:
                out.add(row[0])
    return sorted(out)


def context_inventory(
    conn: sqlite3.Connection,
    scope_id: Any,
    unit_ids: Iterable,
    generation: int,
) -> list:
    """The full V8-06.07 neighbor inventory for a nominated candidate
    set: sessions touched by the candidates, then every fenced
    ``kind='turn'`` member of those sessions — two batched reads."""
    sessions = session_ids_for(conn, scope_id, unit_ids, generation)
    if not sessions:
        return []
    return fetch_session_members(conn, scope_id, sessions, generation)


def neighbors_for_unit(
    conn: sqlite3.Connection,
    scope_id: Any,
    unit_id: Any,
    *,
    W: int = 1,
    generation: int = 0,
    session_id: Any = None,
    eligible: Any = None,
) -> list:
    """A hit's ±``W`` position neighbors — the V85-02.04/delivery-window
    entry point.  ``session_id`` may be supplied to skip the
    unit→session resolution read.  Returns ordered member dicts; the
    unit itself is never in its own neighbor list."""
    if session_id is None:
        sessions = session_ids_for(conn, scope_id, [unit_id], generation)
        if not sessions:
            return []
        session_id = sessions[0]
    members = fetch_session_members(conn, scope_id, [session_id], generation)
    index = SessionIndex.build(members, eligible=eligible)
    return index.neighbors(unit_id, W)


__all__ = [
    "MEMBER_TUPLE_LEN",
    "SessionIndex",
    "context_inventory",
    "eligible_member",
    "fetch_session_members",
    "member_row",
    "member_sort_key",
    "neighbors_for_unit",
    "session_ids_for",
]
