"""Seed helpers for v3 experience tests (SPEC_V3 §12–§13, §20).

The evidence-plane tables are the contract: tests insert
``source_envelopes``/``trajectories``/``trajectory_steps``/
``step_observations``/``state_anchors`` rows directly through
``repos_v3`` + parameterized SQL — no evidence-module dependency.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import sqlite3
from typing import Any, Iterable, Optional

from verbatim.storage import repos_v3


def _content_digest(conn: sqlite3.Connection, payload: bytes) -> bytes:
    """Compute the store's real content digest for a seeded payload.

    Production reads re-verify ``payload_hmac``/``excerpt_hmac`` against
    ``store.hmac`` (``hmac.new(store._hmac_key, data, sha256)``). These
    helpers receive only the writer connection, but ``Store.create``
    persists the profile key at ``<db path>.key`` and the connection's
    own ``database_list`` yields that path — so the seeded digest is the
    same value the store itself would compute.
    """
    db_file = conn.execute("PRAGMA database_list").fetchone()[2]
    with open(db_file + ".key", "rb") as fh:
        key = fh.read()
    return hmac_mod.new(key, payload, hashlib.sha256).digest()


def seed_scope(conn: sqlite3.Connection, scope_id: str) -> None:
    conn.execute(
        "INSERT INTO scopes (scope_id, profile_id, visibility)"
        " VALUES (?, 'prof', 'owner')",
        (scope_id,),
    )


def checker_metadata(
    outcome: Optional[str] = None,
    *,
    checker_id: str = "pytest",
    exit_code: Optional[int] = None,
    completed: bool = True,
    invocation_id: str = "inv-1",
    host_attested: bool = True,
) -> dict[str, Any]:
    """Envelope metadata carrying an identified checker receipt
    (types_v3.CheckerReceipt fields) plus an optional declared outcome."""
    metadata: dict[str, Any] = {
        "checker": {
            "checker_id": checker_id,
            "checker_version": "1.0",
            "repo_revision": "rev-1",
            "tree_digest": "td-1",
            "invocation_id": invocation_id,
            "selected_tests": ["tests/test_x.py"],
            "completed": completed,
            "exit_code": exit_code,
            "result_json": {},
            "host_attested": host_attested,
        }
    }
    if outcome is not None:
        metadata["outcome"] = outcome
    return metadata


def seed_envelope(
    conn: sqlite3.Connection,
    scope_id: str,
    envelope_id: str,
    kind: str,
    *,
    metadata: Optional[dict[str, Any]] = None,
    task_id: str = "",
    step_id: str = "",
    session_id: str = "",
    event_us: int = 1,
    revision: int = 1,
    actor: str = "agent-1",
    trust_class: str = "host_observed",
) -> None:
    """Insert sources + source_revisions + source_envelopes for one id."""
    src_id = f"src-{envelope_id}"
    conn.execute(
        "INSERT INTO sources"
        " (source_id, origin, source_kind, scope_id, created_us)"
        " VALUES (?, 'test', 'tool_output', ?, 1)",
        (src_id, scope_id),
    )
    payload = f"payload-{envelope_id}".encode("utf-8")
    conn.execute(
        "INSERT INTO source_revisions"
        " (source_id, revision, payload, payload_hmac, event_us,"
        "  captured_us, provenance)"
        " VALUES (?, ?, ?, ?, ?, ?, 'approved_tool')",
        (
            src_id,
            revision,
            payload,
            _content_digest(conn, payload),
            event_us,
            event_us,
        ),
    )
    repos_v3.insert(
        conn,
        "source_envelopes",
        {
            "envelope_id": envelope_id,
            "source_id": src_id,
            "revision": revision,
            "scope_id": scope_id,
            "envelope_kind": kind,
            "actor_principal": actor,
            "event_us": event_us,
            "receipt_us": event_us,
            "trust_class": trust_class,
            "task_id": task_id,
            "step_id": step_id,
            "session_id": session_id,
            "metadata_json": metadata or {},
        },
    )


def seed_trajectory(
    conn: sqlite3.Connection,
    scope_id: str,
    trajectory_id: str,
    *,
    steps: Iterable[dict[str, Any]],
    completed: bool = True,
    boundary_rule: str = "task_id",
    task_id: str = "task-1",
    session_id: str = "sess-1",
    host_id: str = "host-1",
    environment_digest: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Insert one trajectory plus its ordered steps/observations.

    ``steps`` entries: ``{"step_id", "ord", "action", "observations",
    "environment_digest"}`` — ``action`` is the action envelope id,
    ``observations`` a list of envelope ids linked through
    ``step_observations``.
    """
    repos_v3.insert(
        conn,
        "trajectories",
        {
            "trajectory_id": trajectory_id,
            "scope_id": scope_id,
            "host_id": host_id,
            "session_id": session_id,
            "task_id": task_id,
            "boundary_rule": boundary_rule,
            "created_event": 1,
            "completed_event": 2 if completed else None,
            "environment_digest": environment_digest,
            "metadata_json": metadata or {},
        },
    )
    for step in steps:
        repos_v3.insert(
            conn,
            "trajectory_steps",
            {
                "step_id": step["step_id"],
                "trajectory_id": trajectory_id,
                "scope_id": scope_id,
                "ord": step["ord"],
                "action_envelope_id": step.get("action"),
                "environment_digest": step.get("environment_digest"),
            },
        )
        for i, obs_id in enumerate(step.get("observations", ())):
            repos_v3.insert(
                conn,
                "step_observations",
                {"step_id": step["step_id"], "envelope_id": obs_id, "ord": i},
            )


def seed_three_step_task(
    conn: sqlite3.Connection,
    scope_id: str,
    trajectory_id: str = "traj-1",
    *,
    completed: bool = True,
    checker: bool = True,
    outcome: Optional[str] = "success",
    environment_digest: Optional[str] = "env-digest-1",
    task_id: str = "task-1",
    prefix: str = "",
) -> dict[str, str]:
    """The canonical fixture trajectory: tool_call → file_diff →
    run-check tool_call with a test_result observation.

    ``prefix`` namespaces the envelope/step ids so several trajectories
    can coexist in one scope. Returns the envelope ids keyed by role.
    """
    ids = {
        "tool_call": f"{prefix}env-tc",
        "file_diff": f"{prefix}env-diff",
        "run_check": f"{prefix}env-run",
        "test_result": f"{prefix}env-test",
    }
    steps = [f"{prefix}step-1", f"{prefix}step-2", f"{prefix}step-3"]
    seed_envelope(
        conn, scope_id, ids["tool_call"], "tool_call",
        task_id=task_id, step_id=steps[0], event_us=10,
    )
    seed_envelope(
        conn, scope_id, ids["file_diff"], "file_diff",
        task_id=task_id, step_id=steps[1], event_us=20,
    )
    seed_envelope(
        conn, scope_id, ids["run_check"], "tool_call",
        task_id=task_id, step_id=steps[2], event_us=30,
    )
    test_md = checker_metadata(outcome, exit_code=0) if checker else {}
    seed_envelope(
        conn, scope_id, ids["test_result"], "test_result",
        metadata=test_md, task_id=task_id, step_id=steps[2], event_us=40,
    )
    seed_trajectory(
        conn,
        scope_id,
        trajectory_id,
        steps=[
            {"step_id": steps[0], "ord": 0, "action": ids["tool_call"]},
            {"step_id": steps[1], "ord": 1, "action": ids["file_diff"]},
            {
                "step_id": steps[2],
                "ord": 2,
                "action": ids["run_check"],
                "observations": [ids["test_result"]],
            },
        ],
        completed=completed,
        task_id=task_id,
        environment_digest=environment_digest,
    )
    return ids
