"""Episodes: typed task/conversation/event groupings (SPEC_V2 §21).

An episode groups memory objects (claims, spans, procedures, other
episodes) under an authenticated host task/session id or an explicit
operator label. Membership is bitemporal — every member row carries a
recorded interval, so ``members(as_of_seq=N)`` reconstructs exactly what
was believed to belong at event seq N.

Everything here is a thin service layer over ``EpisodesRepo`` plus the
package-local helpers in ``repo.py`` (closing an episode's recorded
interval lives here so ``repos_v2.py`` stays untouched).
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

from ..core.types import ErrorCode, VerbatimError, require_id
from ..storage.repos_v2 import EpisodesRepo
from ..storage.store import Store
from . import repo as _repo


def open_episode(
    store: Store,
    conn: sqlite3.Connection,
    scope_id: str,
    *,
    kind: str = "task",
    host_task_id: Optional[str] = None,
    host_session_id: Optional[str] = None,
    parent_episode_id: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    """Open an episode inside the caller's transaction; returns episode_id.

    ``kind`` is a free-form but required grouping tag (task, meeting,
    conversation_segment, event, …). Host ids come from the authenticated
    host context — they are recorded, never synthesized (V2-21.05).
    """
    return EpisodesRepo(store).create(
        conn,
        scope_id,
        kind=kind,
        host_task_id=host_task_id,
        host_session_id=host_session_id,
        parent_episode_id=parent_episode_id,
        label=label,
    )


def _require_open_episode(conn: sqlite3.Connection, episode_id: str) -> dict[str, Any]:
    row = _repo.episode_row(conn, episode_id)
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_FORBIDDEN, "episode not found"
        )
    if row["recorded_until"] is not None:
        raise VerbatimError(
            ErrorCode.INVALID_TRANSITION, "episode is closed"
        )
    return row


def attach(
    store: Store,
    conn: sqlite3.Connection,
    episode_id: str,
    object_kind: str,
    object_id: str,
    ord: int = 0,
) -> None:
    """Add a member to an open episode.

    Membership is append/retire — re-attaching an existing (kind, id)
    refreshes ``ord`` and the recorded interval via INSERT OR REPLACE.
    Attaching to a closed episode is an INVALID_TRANSITION, not a silent
    reopen.
    """
    _require_open_episode(conn, episode_id)
    EpisodesRepo(store).add_member(
        conn, episode_id, object_kind, object_id, ord=ord
    )


def detach(
    store: Store,
    conn: sqlite3.Connection,
    episode_id: str,
    object_kind: str,
    object_id: str,
) -> None:
    """Retire a member's recorded interval (temporal remove)."""
    _require_open_episode(conn, episode_id)
    EpisodesRepo(store).remove_member(conn, episode_id, object_kind, object_id)


def close_episode(store: Store, conn: sqlite3.Connection, episode_id: str) -> int:
    """Close the episode row's recorded interval; returns the closing seq.

    Closing does not delete members: the historical record — including who
    belonged and when — stays queryable through ``as_of_seq``.
    """
    return _repo.close_episode_row(conn, episode_id)


def members(
    store: Store, episode_id: str, *, as_of_seq: Optional[int] = None
) -> list[dict[str, Any]]:
    """Current membership, or the membership believed at ``as_of_seq``."""
    return EpisodesRepo(store).members(episode_id, as_of_seq=as_of_seq)


def episode_summary(
    store: Store, episode_id: str, *, as_of_seq: Optional[int] = None
) -> dict[str, Any]:
    """Inspectable episode view: state plus members grouped by kind.

    Returns quoted/recorded data only — episodes are navigation structure,
    never synthesized facts (V2-21.06).
    """
    row = EpisodesRepo(store).get(episode_id)
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_FORBIDDEN, "episode not found"
        )
    member_rows = EpisodesRepo(store).members(episode_id, as_of_seq=as_of_seq)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for m in member_rows:
        grouped.setdefault(m["object_kind"], []).append(
            {
                "object_id": m["object_id"],
                "ord": m["ord"],
                "recorded_from": m["recorded_from"],
            }
        )
    return {
        "kind": "episode",
        "episode_id": row["episode_id"],
        "scope_id": row["scope_id"],
        "revision": row["revision"],
        "row_version": row["row_version"],
        "episode_kind": row["kind"],
        "label": row["label"],
        "host_task_id": row["host_task_id"],
        "host_session_id": row["host_session_id"],
        "parent_episode_id": row["parent_episode_id"],
        "recorded_from": row["recorded_from"],
        "recorded_until": row["recorded_until"],
        "state": "closed" if row["recorded_until"] is not None else "open",
        "members": grouped,
        "member_count": len(member_rows),
    }


def episodes_for_scope(
    store: Store, scope_id: str, *, kind: Optional[str] = None
) -> list[dict[str, Any]]:
    """Current (non-retired) episodes in a scope — MemoryKind.EPISODE
    candidate source for retrieval."""
    require_id(scope_id, "scope_id")
    return EpisodesRepo(store).for_scope(scope_id, kind=kind)
