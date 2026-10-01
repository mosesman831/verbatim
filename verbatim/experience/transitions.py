"""V3 transition derivation from trajectory steps (SPEC_V3 §20.02–§20.04).

For every ``trajectory_step`` of an episode's trajectory this writes one
deterministic ``transitions`` row — ``(pre_state anchors, action step,
post_state anchors, checker_ref, environment_digest, edge)`` — plus its
``transition_anchors`` links and ``derivations`` edges. No model invents
transitions (V3-20.02).

Anchor windows (§20.03 — anchors reconstruct order, never causation):

- ``pre``: anchors attached to trajectory steps with ord strictly below
  the action step's ord — the observed prior state.
- ``post``: anchors attached to the action step itself or the next step
  in ord order — the observed state at/after the action.

Edges (V3-20.04):

- ``verified_by`` when a ``test_result``/``verification`` envelope with
  an *identified* checker receipt is linked to the step through
  ``step_observations``; ``checker_ref`` is that envelope's id.
- ``observed_after`` otherwise — order, never ``caused``. Causal claims
  need a separately labeled ``causal_hypothesis`` method, which this
  producer never emits.

Idempotency (V3-20.08): ``transition_id`` is a pure function of
``(episode_id, step_id)`` and ``ord`` equals the step's ord, so replaying
a trajectory yields identical transition identities. Rebuilds dedup on
the primary key; ``transition_anchors``/``derivations`` dedup on their
own keys. When checker evidence arrives after the first build, a rebuild
refines the existing row's ``edge``/``checker_ref`` in place — the row
count stays stable at one transition per step.
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any, Optional

from ..core.types import ErrorCode, VerbatimError, require_id
from ..core.types_v3 import TransitionEdge
from ..storage import repos_v3
from ..storage.repos import _next_event_seq
from . import anchors as _anchors
from . import episodes_v3 as _ep
from . import repo as _repo

TRANSITION_PRODUCER_KIND = "producer"
TRANSITION_PRODUCER_ID = "v3.transition_build.v1"


def transition_id_for(episode_id: str, step_id: str) -> str:
    """Deterministic transition identity for (episode, step) — V3-20.08."""
    require_id(episode_id, "episode_id")
    require_id(step_id, "step_id")
    digest = hashlib.sha256(
        f"v3:transition\x00{episode_id}\x00{step_id}".encode("utf-8")
    ).hexdigest()
    return f"tr_{digest[:32]}"


def _trajectory_for_episode(
    conn: sqlite3.Connection, episode_id: str
) -> Optional[str]:
    """The trajectory this episode was built from, via derivations.

    Producer-scoped: only the episode builder's own edge identifies the
    source trajectory — a foreign producer's edge never redirects the
    derivation into unrelated evidence.
    """
    row = conn.execute(
        "SELECT parent_id FROM derivations"
        " WHERE child_kind = 'episode' AND child_id = ?"
        "   AND parent_kind = 'trajectory' AND producer_id = ?"
        " ORDER BY child_revision DESC, parent_id LIMIT 1",
        (episode_id, _ep.EPISODE_PRODUCER_ID),
    ).fetchone()
    return str(row[0]) if row is not None else None


def _checker_envelopes(
    conn: sqlite3.Connection, step_id: str
) -> list[dict[str, Any]]:
    """Observation envelopes on this step carrying an identified checker
    receipt, in recorded observation order (V3-12.03, §20.04)."""
    obs = repos_v3.query(
        conn, "step_observations", {"step_id": step_id}, order="ord"
    )
    out: list[dict[str, Any]] = []
    for o in obs:
        env = repos_v3.get(
            conn, "source_envelopes", {"envelope_id": o["envelope_id"]}
        )
        if env is None:
            continue
        if env["envelope_kind"] not in _ep.OUTCOME_ENVELOPE_KINDS:
            continue
        metadata = repos_v3.json_field(env, "metadata_json", {}) or {}
        if _ep.checker_receipt(metadata) is None:
            continue
        out.append(env)
    return out


def _link_anchors(
    conn: sqlite3.Connection,
    transition_id: str,
    role: str,
    anchors: list[dict[str, Any]],
) -> None:
    for i, anchor in enumerate(anchors):
        conn.execute(
            "INSERT OR IGNORE INTO transition_anchors"
            " (transition_id, role, anchor_id, ord) VALUES (?, ?, ?, ?)",
            (transition_id, role, anchor["anchor_id"], i),
        )


def _derivation(
    conn: sqlite3.Connection,
    *,
    transition_id: str,
    parent_kind: str,
    parent_id: str,
    parent_revision: int,
    seq: int,
    scope_id: str,
) -> None:
    """One immutable derivations edge (V3-17.02). Exact replays are
    no-ops; an identical edge under a different producer is INTEGRITY —
    matching the ``derivations.record_edge`` contract without importing
    the in-flight module."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO derivations"
        " (child_kind, child_id, child_revision, parent_kind, parent_id,"
        "  parent_revision, producer_kind, producer_id, seq, scope_id)"
        " VALUES ('transition', ?, 1, ?, ?, ?, ?, ?, ?, ?)",
        (
            transition_id,
            parent_kind,
            parent_id,
            parent_revision,
            TRANSITION_PRODUCER_KIND,
            TRANSITION_PRODUCER_ID,
            seq,
            scope_id,
        ),
    )
    if cur.rowcount == 0:
        existing = conn.execute(
            "SELECT producer_kind, producer_id FROM derivations"
            " WHERE child_kind = 'transition' AND child_id = ?"
            "   AND child_revision = 1 AND parent_kind = ?"
            "   AND parent_id = ? AND parent_revision = ?",
            (transition_id, parent_kind, parent_id, parent_revision),
        ).fetchone()
        if existing is not None and (
            existing[0] != TRANSITION_PRODUCER_KIND
            or existing[1] != TRANSITION_PRODUCER_ID
        ):
            raise VerbatimError(
                ErrorCode.INTEGRITY,
                "derivation edge exists under a different producer — "
                "edges are immutable (V3-17.02)",
            )


def _envelope_revision(conn: sqlite3.Connection, envelope_id: str) -> int:
    row = repos_v3.get(conn, "source_envelopes", {"envelope_id": envelope_id})
    return int(row["revision"]) if row is not None else 1


def transitions_for_episode(
    conn: sqlite3.Connection, episode_id: str
) -> list[dict[str, Any]]:
    """Transition rows for one episode, ord-ordered."""
    require_id(episode_id, "episode_id")
    return repos_v3.query(
        conn, "transitions", {"episode_id": episode_id}, order="ord"
    )


def anchors_for_transition(
    conn: sqlite3.Connection, transition_id: str
) -> dict[str, list[dict[str, Any]]]:
    """``{'pre': [...], 'post': [...]}`` anchor rows for a transition."""
    require_id(transition_id, "transition_id")
    out: dict[str, list[dict[str, Any]]] = {"pre": [], "post": []}
    links = repos_v3.query(
        conn, "transition_anchors", {"transition_id": transition_id}, order="ord"
    )
    for link in links:
        anchor = repos_v3.get(
            conn, "state_anchors", {"anchor_id": link["anchor_id"]}
        )
        if anchor is not None:
            out[link["role"]].append(anchor)
    return out


def build_transitions(conn: sqlite3.Connection, episode_id: str) -> int:
    """Derive transitions for every step of the episode's trajectory.

    Returns the number of transition rows for the episode (one per
    trajectory step). Raises ``NOT_FOUND_OR_UNAUTHORIZED`` when the
    episode is absent and ``VALIDATION`` when it carries no trajectory
    derivation — transitions are only ever derived from recorded
    trajectories.
    """
    require_id(episode_id, "episode_id")
    episode = _repo.episode_row(conn, episode_id)
    if episode is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "episode not found"
        )
    trajectory_id = _trajectory_for_episode(conn, episode_id)
    if trajectory_id is None:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "episode has no trajectory derivation — transitions derive "
            "from recorded trajectories only (§20.02)",
        )
    traj = repos_v3.get(conn, "trajectories", {"trajectory_id": trajectory_id})
    if traj is None:
        raise VerbatimError(
            ErrorCode.INTEGRITY,
            "episode derivation references a missing trajectory",
        )
    steps = _ep.trajectory_steps(conn, trajectory_id)
    scope_id = episode["scope_id"]
    created_event = _next_event_seq(conn)

    for i, step in enumerate(steps):
        step_id = step["step_id"]
        ord_ = int(step["ord"])
        next_ord = int(steps[i + 1]["ord"]) if i + 1 < len(steps) else None

        checkers = _checker_envelopes(conn, step_id)
        edge = (
            TransitionEdge.VERIFIED_BY if checkers else TransitionEdge.OBSERVED_AFTER
        )
        checker_ref = checkers[0]["envelope_id"] if checkers else None
        env_digest = step.get("environment_digest") or traj.get(
            "environment_digest"
        )
        tid = transition_id_for(episode_id, step_id)

        existing = repos_v3.get(conn, "transitions", {"transition_id": tid})
        if existing is None:
            repos_v3.insert(
                conn,
                "transitions",
                {
                    "transition_id": tid,
                    "episode_id": episode_id,
                    "scope_id": scope_id,
                    "ord": ord_,
                    "action_step_id": step_id,
                    "checker_ref": checker_ref,
                    "environment_digest": env_digest,
                    "edge": edge.value,
                    "created_event": created_event,
                },
            )
        else:
            if (
                int(existing["ord"]) != ord_
                or existing["action_step_id"] != step_id
            ):
                raise VerbatimError(
                    ErrorCode.INTEGRITY,
                    f"transition {tid} replays against different step data",
                )
            updates: dict[str, Any] = {}
            if existing["edge"] != edge.value:
                updates["edge"] = edge.value
            if existing.get("checker_ref") != checker_ref:
                updates["checker_ref"] = checker_ref
            if existing.get("environment_digest") != env_digest:
                updates["environment_digest"] = env_digest
            if updates:
                repos_v3.update(
                    conn, "transitions", updates, {"transition_id": tid}
                )

        # Anchor windows: pre = earlier steps; post = this step or the
        # next step in ord order (§20.03 — order, never causation).
        pre = _anchors.anchors_near(conn, step_id, "before")
        post_ords = {ord_} | ({next_ord} if next_ord is not None else set())
        post = _anchors.anchors_for_ords(conn, trajectory_id, post_ords)
        _link_anchors(conn, tid, "pre", pre)
        _link_anchors(conn, tid, "post", post)

        # Derivation edges record every input this row is built from
        # (V3-17.02): the step, its action envelope, checker envelopes,
        # and both anchor windows.
        seq = 0
        _derivation(
            conn,
            transition_id=tid,
            parent_kind="trajectory_step",
            parent_id=step_id,
            parent_revision=1,
            seq=seq,
            scope_id=scope_id,
        )
        seq += 1
        action_env = step.get("action_envelope_id")
        if action_env:
            _derivation(
                conn,
                transition_id=tid,
                parent_kind="envelope",
                parent_id=action_env,
                parent_revision=_envelope_revision(conn, action_env),
                seq=seq,
                scope_id=scope_id,
            )
            seq += 1
        for env in checkers:
            _derivation(
                conn,
                transition_id=tid,
                parent_kind="envelope",
                parent_id=env["envelope_id"],
                parent_revision=int(env["revision"]),
                seq=seq,
                scope_id=scope_id,
            )
            seq += 1
        for anchor in pre + post:
            _derivation(
                conn,
                transition_id=tid,
                parent_kind="state_anchor",
                parent_id=anchor["anchor_id"],
                parent_revision=1,
                seq=seq,
                scope_id=scope_id,
            )
            seq += 1

    row = conn.execute(
        "SELECT COUNT(*) FROM transitions WHERE episode_id = ?", (episode_id,)
    ).fetchone()
    return int(row[0])
