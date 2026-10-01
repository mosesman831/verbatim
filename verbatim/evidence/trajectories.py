"""Trajectory recording for the v3 evidence plane (SPEC_V3 §12.02, §20.03).

A trajectory is the ordered set of steps inside one task boundary; each step
references its action envelope, observation envelopes, state-delta refs, and
the environment fingerprint current at that step (V3-12.02). Steps are
dense and monotonic — ``ord`` is the insertion order, enforced by
``UNIQUE(trajectory_id, ord)`` plus a dense-sequence check so a host cannot
silently skip or reorder steps.

``created_event``/``completed_event`` bind the trajectory to the append-only
events journal (the same provenance discipline as v2, SPEC §20): creation
and completion are journaled transitions, never bare row mutations.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

from ..core.types import ErrorCode, VerbatimError, require_id
from ..core.types_v3 import StateAnchor, TrajectoryRecord, TrajectoryStep
from ..storage import repos_v3
from ..storage.repos import EventsRepo
from .envelopes import ensure_scope_row

_TRAJECTORY_POLICY = "trajectory_v3_v1"
_CREATED_KIND = "trajectory_created"
_COMPLETED_KIND = "trajectory_completed"


def _next_event_seq(conn: sqlite3.Connection) -> int:
    """Estimated next event_seq — materializes as the real sequence when the
    caller's tx appends/commits (same convention as repos._next_event_seq)."""
    row = conn.execute(
        "SELECT COALESCE(MAX(event_seq), 0) + 1 FROM events"
    ).fetchone()
    return int(row[0])


def _trajectory_row(conn: sqlite3.Connection, trajectory_id: str) -> dict[str, Any]:
    row = repos_v3.get(conn, "trajectories", {"trajectory_id": trajectory_id})
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
            "trajectory not found or unauthorized",
        )
    return row


def get_trajectory(conn: sqlite3.Connection, trajectory_id: str) -> Optional[dict[str, Any]]:
    """Plain dict snapshot; None when absent (repos convention)."""
    return repos_v3.get(conn, "trajectories", {"trajectory_id": trajectory_id})


def record_trajectory(
    conn: sqlite3.Connection,
    record: TrajectoryRecord,
    *,
    store: Any = None,
) -> str:
    """Open a trajectory; returns ``trajectory_id``.

    Idempotent: re-recording an existing id returns it without a second
    journal entry (boundary rules may fire twice; the record is written
    once). The ``trajectory_created`` journal event supplies
    ``created_event``.
    """
    if not isinstance(record, TrajectoryRecord):
        raise VerbatimError(ErrorCode.VALIDATION, "record_trajectory needs a TrajectoryRecord")
    require_id(record.trajectory_id, "trajectory_id")
    require_id(record.scope_id, "scope_id")
    if record.created_event is not None and record.created_event < 0:
        raise VerbatimError(ErrorCode.VALIDATION, "created_event must be >= 0")
    existing = repos_v3.get(
        conn, "trajectories", {"trajectory_id": record.trajectory_id}
    )
    if existing is not None:
        return record.trajectory_id
    ensure_scope_row(conn, record.scope_id)
    event_seq = EventsRepo(store or _ClockShim()).append(
        conn,
        record.scope_id,
        _CREATED_KIND,
        record.host_id or "engine",
        {
            "trajectory_id": record.trajectory_id,
            "task_id": record.task_id,
            "boundary_rule": record.boundary_rule,
        },
        _TRAJECTORY_POLICY,
    )
    repos_v3.insert(
        conn,
        "trajectories",
        {
            "trajectory_id": record.trajectory_id,
            "scope_id": record.scope_id,
            "host_id": record.host_id,
            "session_id": record.session_id,
            "task_id": record.task_id,
            "boundary_rule": record.boundary_rule,
            "created_event": event_seq,
            "completed_event": record.completed_event,
            "environment_digest": record.environment_digest,
            "metadata_json": dict(record.metadata),
        },
    )
    return record.trajectory_id


class _ClockShim:
    """Minimal store contract for EventsRepo.append — ``next_event_us`` only.
    Trajectory helpers take ``conn`` like every other repo; when the caller
    does not pass its Store, this shim supplies the nondecreasing logical
    clock append() uses. The caller's Store remains the writer."""

    _last = 0

    def next_event_us(self) -> int:
        from ..core.time import now_us

        now = now_us()
        nxt = now if now > _ClockShim._last else _ClockShim._last + 1
        _ClockShim._last = nxt
        return nxt


def add_step(
    conn: sqlite3.Connection,
    step: TrajectoryStep,
) -> str:
    """Append one ordered step; returns ``step_id``.

    ``ord`` is dense/monotonic insertion order: the next step must carry
    exactly ``existing_count`` as its ord — a gap or repeat is a typed
    VALIDATION error on top of the schema's UNIQUE(trajectory_id, ord).
    Action/observation envelope references must resolve to persisted
    envelopes in the step's scope — a trajectory cannot cite evidence that
    was never captured.
    """
    if not isinstance(step, TrajectoryStep):
        raise VerbatimError(ErrorCode.VALIDATION, "add_step needs a TrajectoryStep")
    require_id(step.step_id, "step_id")
    traj = _trajectory_row(conn, step.trajectory_id)
    scope_id = traj["scope_id"]

    row = conn.execute(
        "SELECT COALESCE(MAX(ord) + 1, 0) FROM trajectory_steps"
        " WHERE trajectory_id = ?",
        (step.trajectory_id,),
    ).fetchone()
    expected = int(row[0])
    if step.ord != expected:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"step ord {step.ord} is not the next dense ord {expected} "
            f"for trajectory {step.trajectory_id}",
        )

    if step.action_envelope_id is not None:
        _require_envelope(conn, step.action_envelope_id, scope_id, "action")
    obs_ids = tuple(step.observation_envelope_ids)
    for env_id in obs_ids:
        _require_envelope(conn, env_id, scope_id, "observation")

    env_digest = step.environment.digest() if step.environment is not None else None
    repos_v3.insert(
        conn,
        "trajectory_steps",
        {
            "step_id": step.step_id,
            "trajectory_id": step.trajectory_id,
            "scope_id": scope_id,
            "ord": step.ord,
            "action_envelope_id": step.action_envelope_id,
            "environment_digest": env_digest,
            "metadata_json": {
                "state_delta_refs": list(step.state_delta_refs),
            },
        },
    )
    for i, env_id in enumerate(obs_ids):
        repos_v3.insert(
            conn,
            "step_observations",
            {"step_id": step.step_id, "envelope_id": env_id, "ord": i},
        )
    return step.step_id


def _require_envelope(
    conn: sqlite3.Connection, envelope_id: str, scope_id: str, role: str
) -> dict[str, Any]:
    row = repos_v3.get(conn, "source_envelopes", {"envelope_id": envelope_id})
    if row is None or row["scope_id"] != scope_id:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"{role} envelope does not resolve inside the step's scope",
        )
    return row


def anchor(
    conn: sqlite3.Connection,
    anchor_: StateAnchor,
    *,
    scope_id: Optional[str] = None,
    step_id: Optional[str] = None,
    created_event: Optional[int] = None,
) -> str:
    """Record a state anchor (§20.03); returns ``anchor_id``.

    Anchors reconstruct order — they never establish causation. ``scope_id``
    is required unless resolvable from ``step_id``.
    """
    if not isinstance(anchor_, StateAnchor):
        raise VerbatimError(ErrorCode.VALIDATION, "anchor needs a StateAnchor")
    require_id(anchor_.anchor_id, "anchor_id")
    resolved_scope = scope_id
    if step_id is not None:
        step = repos_v3.get(conn, "trajectory_steps", {"step_id": step_id})
        if step is None:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
                "step not found or unauthorized",
            )
        if resolved_scope is None:
            resolved_scope = step["scope_id"]
        elif resolved_scope != step["scope_id"]:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "anchor scope does not match the step's scope",
            )
    if resolved_scope is None:
        raise VerbatimError(
            ErrorCode.VALIDATION, "anchor requires scope_id or a resolvable step_id"
        )
    if not anchor_.kind or not anchor_.ref or not anchor_.digest:
        raise VerbatimError(
            ErrorCode.VALIDATION, "anchor kind, ref and digest are required"
        )
    ensure_scope_row(conn, resolved_scope)
    repos_v3.insert(
        conn,
        "state_anchors",
        {
            "anchor_id": anchor_.anchor_id,
            "scope_id": resolved_scope,
            "kind": anchor_.kind,
            "ref": anchor_.ref,
            "digest": anchor_.digest,
            "step_id": step_id,
            "created_event": (
                int(created_event) if created_event is not None else _next_event_seq(conn)
            ),
        },
    )
    return anchor_.anchor_id


def complete(
    conn: sqlite3.Connection,
    trajectory_id: str,
    environment_digest: Optional[str] = None,
    *,
    store: Any = None,
) -> dict[str, Any]:
    """Close a trajectory; returns the updated row.

    Idempotent: an already-completed trajectory keeps its original
    ``completed_event`` — completion is a one-way transition, not a
    rewritable field.
    """
    require_id(trajectory_id, "trajectory_id")
    traj = _trajectory_row(conn, trajectory_id)
    if traj["completed_event"] is not None:
        return traj
    if environment_digest is not None and not isinstance(environment_digest, str):
        raise VerbatimError(ErrorCode.VALIDATION, "environment_digest must be a string")
    event_seq = EventsRepo(store or _ClockShim()).append(
        conn,
        traj["scope_id"],
        _COMPLETED_KIND,
        traj["host_id"] or "engine",
        {"trajectory_id": trajectory_id},
        _TRAJECTORY_POLICY,
    )
    set_: dict[str, Any] = {"completed_event": event_seq}
    if environment_digest is not None:
        set_["environment_digest"] = environment_digest
    repos_v3.update(
        conn, "trajectories", set_, {"trajectory_id": trajectory_id}
    )
    return _trajectory_row(conn, trajectory_id)


def steps(conn: sqlite3.Connection, trajectory_id: str) -> list[dict[str, Any]]:
    """Ordered step snapshots with their observation envelope ids."""
    _trajectory_row(conn, trajectory_id)
    rows = repos_v3.query(
        conn,
        "trajectory_steps",
        {"trajectory_id": trajectory_id},
        order="ord",
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        obs = repos_v3.query(
            conn, "step_observations", {"step_id": row["step_id"]}, order="ord"
        )
        row = dict(row)
        row["observation_envelope_ids"] = [o["envelope_id"] for o in obs]
        out.append(row)
    return out
