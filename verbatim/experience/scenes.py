"""Task-family scenes over v3 episodes (SPEC_V4 §22, V4-22.05/07/09).

A *scene* groups sibling task episodes that share one declared task
family — the level between episode and overview in §22's hierarchy:

    trajectory step → episode → scene → overview → detailed expansion

Rather than a parallel table, a scene is stored as an ``episodes`` row
with ``kind='scene'`` and its membership lives in the existing
``episode_members`` table (``object_kind='episode'``). That keeps the
whole lifecycle honest for free:

- ``episodes.revision`` is the scene's *membership revision* — every
  membership change bumps it, and one immutable ``derivations`` edge per
  member is recorded at each revision, so the membership history of a
  scene is reconstructable revision-by-revision (V4-22.05, V3-17.02).
- ``episode_members.recorded_from``/``recorded_until`` carries the
  member's own interval — removal is prospective, never a rewrite.
- ``episodes`` is already a dated, resolvable kind: purge closure,
  suppression, and impact traversal cover scenes with no new tables.
- ``episodes.scope_id`` bounds the scene: membership is only ever to
  episodes in the SAME scope — similarity grouping can never broaden an
  audience across scopes (V4-22.05), and conflicting episodes are kept
  as members (never merged or erased).

Grouping is declared, not inferred by similarity: the *family key* comes
from the member trajectory's ``metadata.task_family`` when the host
declared one, else the trajectory ``task_id``, else the episode's
``label``/``host_task_id``. An episode with no declared family stays
ungrouped — a scene never claims members it cannot justify.

Construction is incremental, bounded, and resumable (V4-22.09):
``build_scenes`` scans only *unassigned* live episodes past a cursor
plus a bounded scene-refresh slice per call; all writes are idempotent
on deterministic scene ids, so redelivery converges.

``episodes``/``episode_members`` are v1/v2 tables — not in the
``repos_v3`` allowlist — so, exactly like ``episodes_v3``/``repo.py``,
this module addresses them with explicit SQL on the caller's ``conn``.
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any, Iterable, Optional

from .. import derivations
from ..core.types import ErrorCode, VerbatimError, require_id
from ..storage import repos_v3
from ..storage.repos import _next_event_seq, _row, _rows
from ..storage.repos import has_table as _has_table
from . import repo as _repo

#: Producer identity recorded on every derivations edge this module
#: writes (V3-17.02, V4-22.05 — producer kind + revisioned identity).
SCENE_PRODUCER_KIND = "producer"
SCENE_PRODUCER_ID = "v4.scene_build.v1"

#: ``episodes.kind`` value marking a scene row.
SCENE_KIND = "scene"

#: The boundary rule scenes record (V4-22.02 — the boundary names a
#: rule, never an opaque inference).
SCENE_BOUNDARY_RULE = "task_family"

_SCENE_ID_PREFIX = "sc_"
_FAMILY_KEY_MAX = 200
_DEFAULT_LIMIT = 256

#: Purge row states that tombstone an object (mirrors
#: ``retrieval.candidates._SUPPRESSING_STATES``).
_SUPPRESSING_STATES = ("suppressed", "purging", "completed")


# ---------------------------------------------------------------------------
# identity + family keys
# ---------------------------------------------------------------------------


def scene_id_for(scope_id: str, family_key: str) -> str:
    """Deterministic scene identity for (scope, family key) — V4-22.09."""
    require_id(scope_id, "scope_id")
    require_id(family_key, "family_key")
    digest = hashlib.sha256(
        f"v4:scene\x00{scope_id}\x00{family_key}".encode("utf-8")
    ).hexdigest()
    return f"{_SCENE_ID_PREFIX}{digest[:32]}"


def normalize_family_key(value: Any) -> Optional[str]:
    """Canonical family key text, or ``None`` when nothing is declared.

    Whitespace-normalized, case-folded, length-capped — the same family
    must hash identically regardless of incidental formatting.
    """
    if not isinstance(value, str):
        return None
    key = " ".join(value.split()).lower()
    if not key:
        return None
    return key[:_FAMILY_KEY_MAX]


def trajectory_of_episode(
    conn: sqlite3.Connection, episode_id: str
) -> Optional[str]:
    """The trajectory this episode was built from, if any (V4-22.07 —
    exact episode→trajectory lookup via the recorded derivation edge)."""
    require_id(episode_id, "episode_id")
    row = conn.execute(
        "SELECT parent_id FROM derivations"
        " WHERE child_kind = 'episode' AND child_id = ?"
        "   AND parent_kind = 'trajectory' ORDER BY seq LIMIT 1",
        (episode_id,),
    ).fetchone()
    return str(row[0]) if row is not None else None


def family_key_for(
    conn: sqlite3.Connection, episode: dict[str, Any]
) -> Optional[str]:
    """Resolve the declared task family for one task episode.

    Priority: the source trajectory's ``metadata.task_family`` (a host
    that declares families explicitly wins), then the trajectory's
    ``task_id``, then the episode's ``label``, then ``host_task_id``.
    ``None`` means *no declared family* — the episode stays ungrouped.
    Scenes themselves have no family (``None``), which also makes
    scene-in-scene membership impossible by construction.
    """
    if episode.get("kind") == SCENE_KIND:
        return None
    traj_id = trajectory_of_episode(conn, episode["episode_id"])
    if traj_id is not None:
        traj = repos_v3.get(
            conn, "trajectories", {"trajectory_id": traj_id}
        )
        if traj is not None:
            meta = repos_v3.json_field(traj, "metadata_json", {}) or {}
            key = normalize_family_key(meta.get("task_family"))
            if key is not None:
                return key
            key = normalize_family_key(traj.get("task_id"))
            if key is not None:
                return key
    for field in ("label", "host_task_id"):
        key = normalize_family_key(episode.get(field))
        if key is not None:
            return key
    return None


# ---------------------------------------------------------------------------
# suppression awareness (fail closed — mirrors observations.aggregate)
# ---------------------------------------------------------------------------


def _purge_suppressed(
    conn: sqlite3.Connection, episode_ids: Iterable[str]
) -> set[str]:
    """Episode ids carrying a suppressing purge tombstone."""
    ids = list(dict.fromkeys(episode_ids))
    if not ids or not _has_table(conn, "purge_targets"):
        return set()
    ph = ",".join("?" for _ in ids)
    sph = ",".join("?" for _ in _SUPPRESSING_STATES)
    rows = conn.execute(
        "SELECT DISTINCT pt.object_id FROM purge_targets pt"
        " JOIN purges p ON p.purge_id = pt.purge_id"
        " WHERE pt.object_kind = 'episode'"
        f" AND p.state IN ({sph}) AND pt.object_id IN ({ph})",
        (*_SUPPRESSING_STATES, *ids),
    ).fetchall()
    return {str(r[0]) for r in rows}


def _member_rows(conn: sqlite3.Connection, scene_id: str) -> list[dict[str, Any]]:
    """All member rows of one scene (open and closed intervals)."""
    return _rows(
        conn.execute(
            "SELECT * FROM episode_members"
            " WHERE episode_id = ? AND object_kind = 'episode'"
            " ORDER BY ord, object_id",
            (scene_id,),
        )
    )


def _member_row(
    conn: sqlite3.Connection, scene_id: str, episode_id: str
) -> Optional[dict[str, Any]]:
    return _row(
        conn.execute(
            "SELECT * FROM episode_members"
            " WHERE episode_id = ? AND object_kind = 'episode'"
            "   AND object_id = ?",
            (scene_id, episode_id),
        )
    )


def _current_members(
    conn: sqlite3.Connection, scene_id: str
) -> list[dict[str, Any]]:
    return [m for m in _member_rows(conn, scene_id) if m["recorded_until"] is None]


def _bump_revision(
    conn: sqlite3.Connection, scene: dict[str, Any], seq: int
) -> int:
    """Advance the scene's membership revision and re-record the member
    derivation edges at the new revision (V4-22.05).

    Edges are immutable and revision-specific: each revision's parent set
    IS the membership history — no member row is ever rewritten to hide
    that an episode once belonged.
    """
    scene_id = scene["episode_id"]
    new_rev = int(scene["revision"]) + 1
    conn.execute(
        "UPDATE episodes SET revision = ?, row_version = row_version + 1"
        " WHERE episode_id = ?",
        (new_rev, scene_id),
    )
    members = _current_members(conn, scene_id)
    suppressed = _purge_suppressed(
        conn, [m["object_id"] for m in members]
    )
    for i, member in enumerate(members):
        mid = member["object_id"]
        mrow = _repo.episode_row(conn, mid)
        if mrow is None or mid in suppressed:
            continue
        derivations.record_edge(
            conn,
            child=("episode", scene_id, new_rev),
            parent=("episode", mid, int(mrow["revision"])),
            producer_kind=SCENE_PRODUCER_KIND,
            producer_id=SCENE_PRODUCER_ID,
            scope_id=scene["scope_id"],
            seq=seq + i,
        )
    return new_rev


def _ensure_scene(
    conn: sqlite3.Connection, scope_id: str, family_key: str, seq: int
) -> tuple[Optional[str], str]:
    """Fetch or create the scene row for a family.

    Returns ``(scene_id|None, status)`` — ``None`` + ``'suppressed'``
    when a purge tombstone still holds the row (a suppressed scene is
    never silently resurrected; recomputation waits for governance).
    A producer-retired empty scene reopens on new membership.
    """
    scene_id = scene_id_for(scope_id, family_key)
    scene = _repo.episode_row(conn, scene_id)
    if scene is not None:
        if scene_id in _purge_suppressed(conn, [scene_id]):
            return None, "suppressed"
        if scene["recorded_until"] is not None:
            conn.execute(
                "UPDATE episodes SET recorded_until = NULL,"
                " row_version = row_version + 1 WHERE episode_id = ?",
                (scene_id,),
            )
        return scene_id, "existing"
    conn.execute(
        "INSERT INTO episodes"
        " (episode_id, scope_id, revision, kind, label, recorded_from,"
        "  boundary_rule, outcome)"
        " VALUES (?, ?, 1, 'scene', ?, ?, ?, 'unknown')",
        (
            scene_id,
            scope_id,
            family_key,
            seq,
            SCENE_BOUNDARY_RULE,
        ),
    )
    return scene_id, "created"


def assign_episode(
    conn: sqlite3.Connection,
    episode_id: str,
    *,
    family_key: Optional[str] = None,
    seq: Optional[int] = None,
) -> dict[str, Any]:
    """Add one task episode to its declared-family scene (idempotent).

    ``family_key`` overrides resolution only when explicitly supplied —
    an operator-declared membership is evidence-linked the same way.
    Returns a small report: ``assigned``, ``scene_id``, ``revision``,
    and ``reason`` when nothing was written. Scenes cannot be members
    of scenes, and a closed/purged episode or a purge-suppressed scene
    never gains membership.
    """
    require_id(episode_id, "episode_id")
    ep = _repo.episode_row(conn, episode_id)
    if ep is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "episode not found"
        )
    if ep["kind"] == SCENE_KIND:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "a scene cannot be a member of a scene (V4-22.05)",
        )
    if ep["recorded_until"] is not None or episode_id in _purge_suppressed(
        conn, [episode_id]
    ):
        return {"assigned": False, "reason": "episode_closed_or_purged"}

    scope_id = ep["scope_id"]
    if seq is None:
        seq = _next_event_seq(conn)
    if family_key is not None:
        fam = normalize_family_key(family_key)
        if fam is None:
            raise VerbatimError(
                ErrorCode.VALIDATION, "family_key must be non-empty"
            )
    else:
        fam = family_key_for(conn, ep)
    if fam is None:
        return {"assigned": False, "reason": "no_declared_family"}

    scene_id, status = _ensure_scene(conn, scope_id, fam, seq)
    if scene_id is None:
        return {
            "assigned": False,
            "reason": "scene_suppressed",
            "family_key": fam,
        }

    member = _member_row(conn, scene_id, episode_id)
    if member is not None and member["recorded_until"] is None:
        scene = _repo.episode_row(conn, scene_id)
        return {
            "assigned": True,
            "changed": False,
            "scene_id": scene_id,
            "scene_status": status,
            "revision": int(scene["revision"]),
            "family_key": fam,
        }

    ord_ = int(ep["recorded_from"])
    if member is None:
        conn.execute(
            "INSERT INTO episode_members"
            " (episode_id, object_kind, object_id, ord, recorded_from)"
            " VALUES (?, 'episode', ?, ?, ?)",
            (scene_id, episode_id, ord_, seq),
        )
    else:
        # Re-membership: reopen the row; the revisioned derivations
        # edges preserve the intervening non-membership (V4-22.05).
        conn.execute(
            "UPDATE episode_members SET recorded_until = NULL, ord = ?,"
            " recorded_from = ?"
            " WHERE episode_id = ? AND object_kind = 'episode'"
            "   AND object_id = ?",
            (ord_, seq, scene_id, episode_id),
        )
    scene = _repo.episode_row(conn, scene_id)
    rev = _bump_revision(conn, scene, seq)
    return {
        "assigned": True,
        "changed": True,
        "scene_id": scene_id,
        "scene_status": status,
        "revision": rev,
        "family_key": fam,
    }


def remove_member(
    conn: sqlite3.Connection,
    scene_id: str,
    episode_id: str,
    *,
    seq: Optional[int] = None,
) -> dict[str, Any]:
    """Close one membership prospectively (V4-22.05 revisable boundary).

    The member row keeps its ``recorded_from`` and gains
    ``recorded_until`` — the membership interval stays on record — and
    the scene revision bumps so the derivations graph records the new
    parent set. Removing a member that is not present is a no-op.
    """
    require_id(scene_id, "scene_id")
    require_id(episode_id, "episode_id")
    scene = _repo.episode_row(conn, scene_id)
    if scene is None or scene["kind"] != SCENE_KIND:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "scene not found"
        )
    if seq is None:
        seq = _next_event_seq(conn)
    member = _member_row(conn, scene_id, episode_id)
    if member is None or member["recorded_until"] is not None:
        return {
            "removed": False,
            "scene_id": scene_id,
            "revision": int(scene["revision"]),
        }
    conn.execute(
        "UPDATE episode_members SET recorded_until = ?"
        " WHERE episode_id = ? AND object_kind = 'episode'"
        "   AND object_id = ?",
        (seq, scene_id, episode_id),
    )
    rev = _bump_revision(conn, scene, seq)
    return {"removed": True, "scene_id": scene_id, "revision": rev}


def refresh_scene(
    conn: sqlite3.Connection,
    scene_id: str,
    *,
    seq: Optional[int] = None,
) -> dict[str, Any]:
    """Reconcile one scene's membership against member episode state.

    Members whose episode row is closed or purge-suppressed are dropped
    (prospectively — ``recorded_until`` on the member row). A scene left
    empty is retired (``episodes.recorded_until``) rather than deleted:
    its prior revisions and derivations remain the membership history.
    A purge-tombstoned scene is never written — report ``suppressed``.
    """
    require_id(scene_id, "scene_id")
    scene = _repo.episode_row(conn, scene_id)
    if scene is None or scene["kind"] != SCENE_KIND:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "scene not found"
        )
    if scene_id in _purge_suppressed(conn, [scene_id]):
        return {"refreshed": False, "reason": "suppressed",
                "scene_id": scene_id}
    if seq is None:
        seq = _next_event_seq(conn)

    members = _current_members(conn, scene_id)
    member_eps = {
        m["object_id"]: _repo.episode_row(conn, m["object_id"])
        for m in members
    }
    suppressed = _purge_suppressed(conn, list(member_eps))
    dropped: list[str] = []
    for m in members:
        mid = m["object_id"]
        ep = member_eps.get(mid)
        if ep is None or ep["recorded_until"] is not None or mid in suppressed:
            conn.execute(
                "UPDATE episode_members SET recorded_until = ?"
                " WHERE episode_id = ? AND object_kind = 'episode'"
                "   AND object_id = ?",
                (seq, scene_id, mid),
            )
            dropped.append(mid)
    changed = bool(dropped)
    rev = int(scene["revision"])
    if changed:
        rev = _bump_revision(conn, scene, seq)
        if not _current_members(conn, scene_id):
            conn.execute(
                "UPDATE episodes SET recorded_until = ?"
                " WHERE episode_id = ?",
                (seq, scene_id),
            )
    return {
        "refreshed": True,
        "scene_id": scene_id,
        "dropped": dropped,
        "changed": changed,
        "revision": rev,
    }


# ---------------------------------------------------------------------------
# bounded incremental pass (V4-22.09)
# ---------------------------------------------------------------------------


def _unassigned_episodes(
    conn: sqlite3.Connection,
    scope_id: str,
    *,
    cursor: Optional[tuple[int, str]],
    limit: int,
) -> list[dict[str, Any]]:
    """Live non-scene episodes with no current scene membership, ordered
    by (recorded_from, episode_id) past ``cursor`` — bounded by ``limit``.
    """
    params: list[Any] = [scope_id]
    where = (
        "e.scope_id = ? AND e.kind != 'scene' AND e.recorded_until IS NULL"
        " AND NOT EXISTS (SELECT 1 FROM episode_members m"
        "  WHERE m.object_kind = 'episode' AND m.object_id = e.episode_id"
        "    AND m.recorded_until IS NULL)"
    )
    if cursor is not None:
        where += " AND (e.recorded_from > ? OR (e.recorded_from = ?"
        where += " AND e.episode_id > ?))"
        params.extend([cursor[0], cursor[0], cursor[1]])
    cur = conn.execute(
        "SELECT e.* FROM episodes e"
        f" WHERE {where} ORDER BY e.recorded_from, e.episode_id"
        " LIMIT ?",
        (*params, limit + 1),
    )
    return _rows(cur)


def build_scenes(
    conn: sqlite3.Connection,
    scope_id: str,
    *,
    limit: int = _DEFAULT_LIMIT,
    cursor: Optional[tuple[int, str]] = None,
    scene_cursor: Optional[tuple[int, str]] = None,
    seq: Optional[int] = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """One bounded incremental scene-construction pass (V4-22.09).

    Two phases, both bounded by ``limit``:
    1. assign live ungrouped episodes (past ``cursor``) to their
       declared-family scene;
    2. refresh existing scenes (past ``scene_cursor``) so closed/purged
       members drop out prospectively.

    Returns a report with ``cursor``/``scene_cursor`` for resumption and
    ``truncated`` when work remains; feeding the cursors back resumes
    each phase where it stopped. A second pass over unchanged state
    writes nothing — assignment and refresh are both idempotent.
    """
    require_id(scope_id, "scope_id")
    if not enabled:
        raise VerbatimError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            "scene construction is disabled for this scope",
        )
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise VerbatimError(ErrorCode.VALIDATION, "limit must be >= 1")
    if seq is None:
        seq = _next_event_seq(conn)

    report: dict[str, Any] = {
        "assigned": 0,
        "scenes_created": 0,
        "revisions_written": 0,
        "members_dropped": 0,
        "scenes_retired": 0,
        "ungrouped": 0,
        "skipped": [],
        "truncated": False,
        "cursor": None,
        "scene_cursor": None,
    }

    batch = _unassigned_episodes(
        conn, scope_id, cursor=cursor, limit=limit
    )
    truncated = len(batch) > limit
    for ep in batch[:limit]:
        res = assign_episode(conn, ep["episode_id"], seq=seq)
        report["cursor"] = (int(ep["recorded_from"]), ep["episode_id"])
        if res["assigned"]:
            report["assigned"] += 1
            if res.get("changed"):
                report["revisions_written"] += 1
            if res.get("scene_status") == "created":
                report["scenes_created"] += 1
        else:
            report["ungrouped"] += 1
            if res.get("reason") == "scene_suppressed":
                report["skipped"].append(res.get("family_key"))
    report["truncated"] = truncated

    # Phase 2 — refresh existing scenes in the same scope, bounded and
    # resumable on the same (recorded_from, episode_id) keyset.
    sparams: list[Any] = [scope_id]
    swhere = "scope_id = ? AND kind = 'scene'"
    if scene_cursor is not None:
        swhere += (" AND (recorded_from > ? OR (recorded_from = ?"
                   " AND episode_id > ?))")
        sparams.extend([scene_cursor[0], scene_cursor[0], scene_cursor[1]])
    scene_rows = _rows(
        conn.execute(
            f"SELECT * FROM episodes WHERE {swhere}"
            " ORDER BY recorded_from, episode_id LIMIT ?",
            (*sparams, limit + 1),
        )
    )
    for srow in scene_rows[:limit]:
        res = refresh_scene(conn, srow["episode_id"], seq=seq)
        report["scene_cursor"] = (int(srow["recorded_from"]),
                                  srow["episode_id"])
        if res.get("changed"):
            report["revisions_written"] += 1
            report["members_dropped"] += len(res["dropped"])
            after = _repo.episode_row(conn, srow["episode_id"])
            if after is not None and after["recorded_until"] is not None:
                report["scenes_retired"] += 1
    if len(scene_rows) > limit:
        report["truncated"] = True
    return report


# ---------------------------------------------------------------------------
# read helpers (V4-22.07 — exact + family + temporal-sequence lookup)
# ---------------------------------------------------------------------------


def scene_row(conn: sqlite3.Connection, scene_id: str) -> Optional[dict[str, Any]]:
    """The scene row, validated as a scene."""
    require_id(scene_id, "scene_id")
    row = _repo.episode_row(conn, scene_id)
    if row is None or row["kind"] != SCENE_KIND:
        return None
    return row


def scenes_for_scope(
    conn: sqlite3.Connection,
    scope_id: str,
    *,
    include_closed: bool = False,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Scene rows in one scope (live unless ``include_closed``)."""
    require_id(scope_id, "scope_id")
    if include_closed:
        return _rows(
            conn.execute(
                "SELECT * FROM episodes WHERE scope_id = ?"
                " AND kind = 'scene'"
                " ORDER BY recorded_from, episode_id LIMIT ?",
                (scope_id, limit),
            )
        )
    return _rows(
        conn.execute(
            "SELECT * FROM episodes WHERE scope_id = ? AND kind = 'scene'"
            " AND recorded_until IS NULL"
            " ORDER BY recorded_from, episode_id LIMIT ?",
            (scope_id, limit),
        )
    )


def scene_members(
    conn: sqlite3.Connection,
    scene_id: str,
    *,
    include_closed: bool = False,
) -> list[dict[str, Any]]:
    """Member episodes of one scene in temporal order (member ``ord``).

    Each entry carries the member's ``episodes`` row fields plus
    ``member_recorded_from``/``member_recorded_until`` — the membership
    interval — so callers see both object state and membership state.
    """
    if scene_row(conn, scene_id) is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "scene not found"
        )
    rows = _member_rows(conn, scene_id)
    out: list[dict[str, Any]] = []
    for m in rows:
        if not include_closed and m["recorded_until"] is not None:
            continue
        ep = _repo.episode_row(conn, m["object_id"])
        entry = dict(ep) if ep is not None else {
            "episode_id": m["object_id"], "missing": True
        }
        entry["member_ord"] = int(m["ord"])
        entry["member_recorded_from"] = int(m["recorded_from"])
        entry["member_recorded_until"] = m["recorded_until"]
        out.append(entry)
    return out


def scene_for_episode(
    conn: sqlite3.Connection, episode_id: str
) -> Optional[dict[str, Any]]:
    """The scene currently containing ``episode_id``, if any."""
    require_id(episode_id, "episode_id")
    row = conn.execute(
        "SELECT episode_id FROM episode_members"
        " WHERE object_kind = 'episode' AND object_id = ?"
        "   AND recorded_until IS NULL ORDER BY episode_id LIMIT 1",
        (episode_id,),
    ).fetchone()
    if row is None:
        return None
    return scene_row(conn, str(row[0]))


def episodes_for_family(
    conn: sqlite3.Connection,
    scope_id: str,
    family_key: str,
) -> list[dict[str, Any]]:
    """Exact task-family search: members of the family's scene."""
    require_id(scope_id, "scope_id")
    fam = normalize_family_key(family_key)
    if fam is None:
        raise VerbatimError(
            ErrorCode.VALIDATION, "family_key must be non-empty"
        )
    scene_id = scene_id_for(scope_id, fam)
    if scene_row(conn, scene_id) is None:
        return []
    return scene_members(conn, scene_id)


def episode_sequence(
    conn: sqlite3.Connection,
    scope_id: str,
    *,
    family_key: Optional[str] = None,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Temporal sequence of episodes (V4-22.07).

    Without ``family_key``: all live non-scene episodes in
    ``recorded_from`` order. With one: the family scene's members in
    their membership order — the ordered episode sequence for that task
    family.
    """
    require_id(scope_id, "scope_id")
    if family_key is not None:
        return episodes_for_family(conn, scope_id, family_key)[:limit]
    return _rows(
        conn.execute(
            "SELECT * FROM episodes WHERE scope_id = ? AND kind != 'scene'"
            " AND recorded_until IS NULL"
            " ORDER BY recorded_from, episode_id LIMIT ?",
            (scope_id, limit),
        )
    )


__all__ = [
    "SCENE_PRODUCER_KIND",
    "SCENE_PRODUCER_ID",
    "SCENE_KIND",
    "SCENE_BOUNDARY_RULE",
    "scene_id_for",
    "normalize_family_key",
    "family_key_for",
    "trajectory_of_episode",
    "assign_episode",
    "remove_member",
    "refresh_scene",
    "build_scenes",
    "scene_row",
    "scenes_for_scope",
    "scene_members",
    "scene_for_episode",
    "episodes_for_family",
    "episode_sequence",
]
