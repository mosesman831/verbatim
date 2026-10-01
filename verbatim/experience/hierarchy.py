"""Hierarchical derived views over episodes and scenes (SPEC_V4 §22).

V4-22.06 requires hierarchical summaries that keep *pointers to detailed
evidence*, preserve contradictory branches, temporal bounds, and missing
coverage — and V4-22.07 wants progressive scene→episode→span expansion
where each level still passes its own eligibility checks (V4-22.08: a
parent's score can never substitute for a child's relevance/eligibility).

Implementation: an *overview* is an ``observations`` row produced by
``v4.hierarchy.v1`` — always a derived object (``observation`` kind in
the derived set), carrying:

- a fixed-template text rendered from structured columns only — never a
  quotation of source/envelope payload bytes (V3-17.04, V4-24.04);
- ``observation_evidence`` ``supports`` rows pointing at the exact
  member revisions it summarizes (bounded pointer list, honest count of
  what was omitted);
- ``derivations`` edges to every cited input — purge and impact
  traversal see the overview exactly like any other derivative, so a
  purged episode/scene suppresses or erases its overview by ancestry
  (V4-38.06).

Construction is incremental, bounded, and resumable (V4-22.09): each
overview id is deterministic over (scope, target kind, target id), so a
pass re-derives only what it is asked to and ``persist_observation``
dedups unchanged inputs — no churn, no duplicate derivatives
(V4-24.08).

``expand`` performs progressive disclosure: scene → member episodes →
member envelopes → span locators. It returns *locators and eligibility
flags only* — never source bytes; authorization stays with the
governance/retrieval layers, and each child reports its own
eligibility so a scene's presence never launders a withheld member
(V4-22.08).
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any, Optional

from .. import derivations
from ..core.types import ErrorCode, VerbatimError, require_id
from ..storage import repos_v3
from ..storage.repos import _next_event_seq, _rows
from ..storage.repos import has_table as _has_table
from ..observations.aggregate import persist_observation
from . import repo as _repo
from . import scenes as _scenes

#: Producer identity for overview observations (V3-17.02).
HIERARCHY_PRODUCER_KIND = "producer"
HIERARCHY_PRODUCER_ID = "v4.hierarchy.v1"

#: Bound on evidence pointers one overview carries; the omitted count is
#: reported honestly in the overview report (V4-22.06 keeps *pointers*,
#: not copies).
_MAX_POINTERS = 64
_DEFAULT_LIMIT = 128

_SUPPRESSING_STATES = _scenes._SUPPRESSING_STATES
_EXCLUDING_QUARANTINE = frozenset({"pending", "suppressed"})


# ---------------------------------------------------------------------------
# overview identity + rendering
# ---------------------------------------------------------------------------


def overview_id_for(scope_id: str, object_kind: str, object_id: str) -> str:
    """Deterministic overview observation id for one parent object."""
    require_id(scope_id, "scope_id")
    require_id(object_id, "object_id")
    digest = hashlib.sha256(
        f"v4:overview\x00{scope_id}\x00{object_kind}\x00{object_id}".encode(
            "utf-8"
        )
    ).hexdigest()
    return f"obs_{digest[:32]}"


def _episode_overview_text(ep: dict[str, Any], n_members: int, n_transitions: int) -> str:
    label = ep.get("label") or ep.get("host_task_id") or ep["episode_id"]
    until = ep["recorded_until"] if ep["recorded_until"] is not None else "open"
    env = ep.get("environment_digest") or "none"
    return (
        f"episode {label}: outcome={ep.get('outcome') or 'unknown'}; "
        f"{n_members} member envelopes; {n_transitions} transitions; "
        f"env={env}; recorded [{ep['recorded_from']}..{until}]"
    )


def _scene_overview_text(
    scene: dict[str, Any], members: list[dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    """Scene summary text + coverage detail.

    Preserves (V4-22.06):
    - contradictory branches — distinct member outcomes are reported as
      a histogram, never merged into one verdict;
    - temporal bounds — min/max member ``recorded_from``/``recorded_until``;
    - missing coverage — members with outcome ``unknown`` or no
      environment digest are counted and named by count.
    """
    outcomes: dict[str, int] = {}
    missing_env = 0
    starts: list[int] = []
    ends: list[int] = []
    open_ended = False
    for m in members:
        outcomes[m.get("outcome") or "unknown"] = (
            outcomes.get(m.get("outcome") or "unknown", 0) + 1
        )
        if not m.get("environment_digest"):
            missing_env += 1
        if m.get("member_recorded_from") is not None:
            starts.append(int(m["member_recorded_from"]))
        if m.get("recorded_until") is None:
            open_ended = True
        elif m.get("recorded_until") is not None:
            ends.append(int(m["recorded_until"]))
    hist = ", ".join(f"{k}×{v}" for k, v in sorted(outcomes.items()))
    contested = len([k for k in outcomes if k != "unknown"]) > 1
    low = min(starts) if starts else 0
    high = "open" if open_ended else str(max(ends) if ends else 0)
    unknown = outcomes.get("unknown", 0)
    text = (
        f"scene {scene.get('label') or scene['episode_id']}: "
        f"{len(members)} episodes; outcomes {hist or 'none'}; "
        f"conflict={'yes' if contested else 'no'}; "
        f"uncovered={unknown}; missing_env={missing_env}; "
        f"span [{low}..{high}]"
    )
    detail = {
        "outcomes": outcomes,
        "contested": contested,
        "uncovered": unknown,
        "missing_env": missing_env,
        "bounds": [low, high],
    }
    return text, detail


# ---------------------------------------------------------------------------
# overview construction
# ---------------------------------------------------------------------------


def _member_envelopes(
    conn: sqlite3.Connection, episode_id: str
) -> list[dict[str, Any]]:
    rows = _rows(
        conn.execute(
            "SELECT * FROM episode_members"
            " WHERE episode_id = ? AND object_kind = 'envelope'"
            "   AND recorded_until IS NULL ORDER BY ord",
            (episode_id,),
        )
    )
    return rows


def _transition_count(conn: sqlite3.Connection, episode_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM transitions WHERE episode_id = ?",
        (episode_id,),
    ).fetchone()
    return int(row[0]) if row else 0


def build_overview(
    conn: sqlite3.Connection,
    scope_id: str,
    object_kind: str,
    object_id: str,
    *,
    seq: Optional[int] = None,
) -> dict[str, Any]:
    """Build/refresh the derived overview for one episode or scene.

    ``object_kind`` is ``'episode'``; a scene is an episodes row with
    ``kind='scene'`` (validated). The overview lands in ``observations``
    with ``supports`` pointers to the parent's members and a
    ``derivations`` parent edge to the parent object itself. Idempotent
    — identical member state writes nothing (V4-24.08). A purge-
    tombstoned parent is never summarized (fail closed): the report
    says ``held``.
    """
    require_id(scope_id, "scope_id")
    if object_kind != "episode":
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"overview supports kind 'episode' (incl. scenes), got {object_kind!r}",
        )
    require_id(object_id, "object_id")
    target = _repo.episode_row(conn, object_id)
    if target is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "episode not found"
        )
    if target["scope_id"] != scope_id:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "episode not in scope"
        )
    if seq is None:
        seq = _next_event_seq(conn)
    obs_id = overview_id_for(scope_id, object_kind, object_id)

    if object_id in _scenes._purge_suppressed(conn, [object_id]):
        return {
            "observation_id": obs_id,
            "written": False,
            "reason": "parent_purged",
        }

    # Evidence pointers = the parent's MEMBERS at their own stable
    # revisions (envelopes for a task episode, member episodes for a
    # scene). The parent itself is linked through a derivations edge
    # (recorded below), not an evidence row: ``observation_evidence``
    # citations are revision-pinned, and a scene's revision moves on
    # every membership change — pinning it as evidence would leave a
    # stale-revision pointer whenever membership evolved. The derivations
    # edge keeps the parent linkage precise (obs@rev → parent@rev-as-read).
    supports: list[tuple[str, str, int]] = []
    omitted = 0
    if target["kind"] == _scenes.SCENE_KIND:
        members = _scenes.scene_members(conn, object_id)
        for m in members:
            if len(supports) >= _MAX_POINTERS:
                omitted += 1
                continue
            supports.append(("episode", m["episode_id"], int(m["revision"])))
        text, detail = _scene_overview_text(target, members)
        proof = len(members)
        freshness = "revalidate_after"
    else:
        members_env = _member_envelopes(conn, object_id)
        for m in members_env:
            if len(supports) >= _MAX_POINTERS:
                omitted += 1
                continue
            env = repos_v3.get(
                conn, "source_envelopes", {"envelope_id": m["object_id"]}
            )
            rev = int(env["revision"]) if env else 1
            supports.append(("envelope", m["object_id"], rev))
        text = _episode_overview_text(
            target, len(members_env), _transition_count(conn, object_id)
        )
        detail = {}
        proof = len(members_env)
        freshness = "unknown"

    res = persist_observation(
        conn,
        scope_id=scope_id,
        observation_id=obs_id,
        text=text,
        proof_count=proof,
        perspective_id=None,
        freshness=freshness,
        producer=HIERARCHY_PRODUCER_ID,
        supports=supports,
        contradicts=(),
        seq=seq,
    )
    if res.wrote:
        # Parent linkage: obs@rev → the object it summarizes, at the
        # revision actually read — the derivations graph is the
        # provenance, and purge traversal follows it revision-precisely.
        derivations.record_edge(
            conn,
            child=("observation", obs_id, res.revision),
            parent=("episode", object_id, int(target["revision"])),
            producer_kind=HIERARCHY_PRODUCER_ID,
            producer_id=f"obs:{obs_id}",
            scope_id=scope_id,
            seq=seq,
        )
        res_edges = res.edges_written + 1
    else:
        res_edges = res.edges_written
    return {
        "observation_id": obs_id,
        "written": res.wrote,
        "revision": res.revision,
        "previous_revision": res.previous_revision,
        "edges_written": res_edges,
        "pointers": len(supports),
        "pointers_omitted": omitted,
        "detail": detail,
    }


def build_overviews(
    conn: sqlite3.Connection,
    scope_id: str,
    *,
    limit: int = _DEFAULT_LIMIT,
    seq: Optional[int] = None,
) -> dict[str, Any]:
    """One bounded pass refreshing overviews for every live episode and
    scene in ``scope_id`` (V4-22.09 incremental/bounded).

    Deterministic order (recorded_from, episode_id); each overview is
    idempotent, so unchanged parents cost only their (capped) member
    reads and write nothing.
    """
    require_id(scope_id, "scope_id")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise VerbatimError(ErrorCode.VALIDATION, "limit must be >= 1")
    if seq is None:
        seq = _next_event_seq(conn)
    rows = _rows(
        conn.execute(
            "SELECT * FROM episodes WHERE scope_id = ?"
            " ORDER BY recorded_from, episode_id LIMIT ?",
            (scope_id, limit + 1),
        )
    )
    report = {"overviews_written": 0, "parents": 0, "truncated": False,
              "held": []}
    for ep in rows[:limit]:
        if ep["recorded_until"] is not None:
            continue
        res = build_overview(
            conn, scope_id, "episode", ep["episode_id"], seq=seq
        )
        report["parents"] += 1
        if res["written"]:
            report["overviews_written"] += 1
        if res.get("reason") == "parent_purged":
            report["held"].append(ep["episode_id"])
    if len(rows) > limit:
        report["truncated"] = True
    return report


def overview_for(
    conn: sqlite3.Connection, scope_id: str, object_kind: str, object_id: str
) -> Optional[dict[str, Any]]:
    """The current overview observation row for a parent, if any."""
    obs_id = overview_id_for(scope_id, object_kind, object_id)
    return repos_v3.get(conn, "observations", {"observation_id": obs_id})


# ---------------------------------------------------------------------------
# progressive expansion (V4-22.07/.08)
# ---------------------------------------------------------------------------


def _eligibility(
    conn: sqlite3.Connection, kind: str, oid: str, rev: Optional[int]
) -> tuple[bool, Optional[str]]:
    """Independent per-child eligibility (V4-22.08).

    Returns ``(eligible, reason)``: purge tombstones, quarantine holds,
    and closed rows each withhold the child on their own — a parent's
    presence is never evidence of a child's availability. Lookup errors
    fail closed (``eligible=False, reason='lookup_failed'``).
    """
    try:
        suppressed_kind = "source_envelope" if kind == "envelope" else kind
        if _has_table(conn, "purge_targets"):
            row = conn.execute(
                "SELECT 1 FROM purge_targets pt JOIN purges p"
                " ON p.purge_id = pt.purge_id"
                " WHERE pt.object_kind = ? AND pt.object_id = ?"
                f"   AND p.state IN ({','.join('?' for _ in _SUPPRESSING_STATES)})"
                " LIMIT 1",
                (suppressed_kind, oid, *_SUPPRESSING_STATES),
            ).fetchone()
            if row is not None:
                return False, "purged"
        if _has_table(conn, "quarantine") and rev is not None:
            qk = "source_envelope" if kind == "envelope" else kind
            row = conn.execute(
                "SELECT state FROM quarantine"
                " WHERE object_kind = ? AND object_id = ? AND revision = ?",
                (qk, oid, rev),
            ).fetchone()
            if row is not None and row[0] in _EXCLUDING_QUARANTINE:
                return False, "held"
        return True, None
    except sqlite3.Error:
        return False, "lookup_failed"


def _expand_episode(
    conn: sqlite3.Connection, ep: dict[str, Any], *, limit: int
) -> dict[str, Any]:
    """One episode node: member envelopes + their span locators."""
    members = _member_envelopes(conn, ep["episode_id"])[:limit]
    envs: list[dict[str, Any]] = []
    for m in members:
        env = repos_v3.get(
            conn, "source_envelopes", {"envelope_id": m["object_id"]}
        )
        node: dict[str, Any] = {
            "envelope_id": m["object_id"],
            "ord": int(m["ord"]),
            "found": env is not None,
        }
        if env is None:
            node["eligible"] = False
            node["reason"] = "missing"
        else:
            rev = int(env["revision"])
            ok, why = _eligibility(
                conn, "envelope", env["envelope_id"], rev
            )
            node.update(
                {
                    "kind": env["envelope_kind"],
                    "revision": rev,
                    "source_id": env["source_id"],
                    "source_revision": rev,
                    "eligible": ok,
                    "reason": why,
                }
            )
            spans = conn.execute(
                "SELECT span_id FROM spans"
                " WHERE source_id = ? AND revision = ? ORDER BY start_byte",
                (env["source_id"], rev),
            ).fetchall()
            node["spans"] = [
                {
                    "span_id": s[0],
                    "eligible": _eligibility(conn, "span", s[0], rev)[0],
                }
                for s in spans
            ]
        envs.append(node)
    ok, why = _eligibility(
        conn, "episode", ep["episode_id"], int(ep["revision"])
    )
    if ep.get("recorded_until") is not None:
        ok, why = False, "closed"
    return {
        "episode_id": ep["episode_id"],
        "kind": ep["kind"],
        "label": ep.get("label"),
        "outcome": ep.get("outcome"),
        "revision": int(ep["revision"]),
        "eligible": ok,
        "reason": why,
        "envelopes": envs,
    }


def expand(
    conn: sqlite3.Connection,
    scope_id: str,
    object_id: str,
    *,
    limit: int = _DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Progressive scene→episode→envelope→span expansion (V4-22.07).

    Returns a plain-dict tree of *locators* — episode rows, envelope
    ids, (source_id, revision), span ids — each with an independent
    ``eligible`` flag and ``reason`` (V4-22.08: child eligibility never
    inherits from the parent). No payload bytes are ever returned; the
    caller's own authorization path decides whether to dereference.
    """
    require_id(scope_id, "scope_id")
    require_id(object_id, "object_id")
    target = _repo.episode_row(conn, object_id)
    if target is None or target["scope_id"] != scope_id:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "episode not found"
        )
    if target["kind"] != _scenes.SCENE_KIND:
        return {"object": _expand_episode(conn, target, limit=limit)}
    members = _scenes.scene_members(conn, object_id)[:limit]
    ok, why = _eligibility(
        conn, "episode", object_id, int(target["revision"])
    )
    if target.get("recorded_until") is not None:
        ok, why = False, "closed"
    return {
        "object": {
            "episode_id": object_id,
            "kind": "scene",
            "label": target.get("label"),
            "revision": int(target["revision"]),
            "eligible": ok,
            "reason": why,
        },
        "members": [
            _expand_episode(conn, m, limit=limit) for m in members
        ],
    }


__all__ = [
    "HIERARCHY_PRODUCER_KIND",
    "HIERARCHY_PRODUCER_ID",
    "overview_id_for",
    "build_overview",
    "build_overviews",
    "overview_for",
    "expand",
]
