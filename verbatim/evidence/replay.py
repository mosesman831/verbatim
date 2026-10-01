"""Replay manifest construction (SPEC_V3 §43).

A replay manifest pins exactly what a sandbox replay needs from the
evidence plane: the trajectory's ordered steps, their action/observation
envelope references, state anchors, capture receipts, the environment
fingerprint, and the event range — plus the sandbox target the replay must
run against.

V3-43.01 is enforced structurally: every manifest carries a ``sandbox_ref``
and the run row records it. The default ref is a fresh ``sandbox://``
identity — replay NEVER targets the live store, and a caller-supplied
``sandbox_ref`` that names the live store path is a typed VALIDATION error.
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any, Iterable, Optional

from ..core.time import now_us
from ..core.types import ErrorCode, VerbatimError, json_dumps, require_id
from ..core.types_v3 import ReplayManifest, ReplayStage
from ..storage import repos_v3
from .receipts import receipt_for_envelope

_SANDBOX_NOTE = (
    "replay runs against a sandbox store, never production (V3-43.01)"
)
_DEFAULT_STAGES = tuple(s.value for s in ReplayStage)


def _manifest_id(material: dict[str, Any]) -> str:
    digest = hashlib.sha256(json_dumps(material).encode("utf-8")).hexdigest()
    return f"rm_{digest[:32]}"


def _suppression_seq(conn: sqlite3.Connection) -> int:
    """Suppression-ledger position the manifest pins (§43.01): the current
    committed event sequence — deletions/suppressions recorded at or below
    it are the replay's visible state."""
    row = conn.execute("SELECT COALESCE(MAX(event_seq), 0) FROM events").fetchone()
    return int(row[0])


def build_replay_manifest(
    conn: sqlite3.Connection,
    trajectory_id: str,
    *,
    sandbox_ref: Optional[str] = None,
    policy_revision: str = "",
    controller_revision: str = "",
    stages: Iterable[str] = (),
    store: Any = None,
) -> ReplayManifest:
    """Build (and durably record) the replay manifest for one trajectory.

    Returns the ``ReplayManifest``; the run row lands in ``replay_runs``
    inside the caller's transaction, carrying the manifest JSON — steps,
    anchors, receipts, environment digest — and the sandbox target.

    Idempotent: rebuilding an identical trajectory mints the identical
    manifest id, and the existing run row stands.
    """
    require_id(trajectory_id, "trajectory_id")
    traj = repos_v3.get(conn, "trajectories", {"trajectory_id": trajectory_id})
    if traj is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
            "trajectory not found or unauthorized",
        )
    if sandbox_ref is not None:
        if not isinstance(sandbox_ref, str) or not sandbox_ref:
            raise VerbatimError(ErrorCode.VALIDATION, "sandbox_ref must be a non-empty string")
        live_path = getattr(store, "path", None) if store is not None else None
        if live_path is not None and sandbox_ref == live_path:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "replay sandbox_ref names the live store — replay targets a "
                "sandbox store only (V3-43.01)",
            )

    # --- ordered steps + observations ---------------------------------------
    step_rows = repos_v3.query(
        conn, "trajectory_steps", {"trajectory_id": trajectory_id}, order="ord"
    )
    steps: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    envelope_ids: list[str] = []
    for row in step_rows:
        obs = repos_v3.query(
            conn, "step_observations", {"step_id": row["step_id"]}, order="ord"
        )
        obs_ids = [o["envelope_id"] for o in obs]
        if row["action_envelope_id"]:
            envelope_ids.append(row["action_envelope_id"])
        envelope_ids.extend(obs_ids)
        steps.append(
            {
                "step_id": row["step_id"],
                "ord": row["ord"],
                "action_envelope_id": row["action_envelope_id"],
                "observation_envelope_ids": obs_ids,
                "environment_digest": row["environment_digest"],
                "state_delta_refs": repos_v3.json_field(
                    row, "metadata_json", {}
                ).get("state_delta_refs", []),
            }
        )
        for a in repos_v3.query(conn, "state_anchors", {"step_id": row["step_id"]}):
            anchors.append(
                {
                    "anchor_id": a["anchor_id"],
                    "kind": a["kind"],
                    "ref": a["ref"],
                    "digest": a["digest"],
                    "step_id": a["step_id"],
                }
            )

    # --- receipts for every referenced envelope ------------------------------
    receipts: list[dict[str, Any]] = []
    artifact_ids: list[str] = []
    seen: set[str] = set()
    for env_id in envelope_ids:
        if env_id in seen:
            continue
        seen.add(env_id)
        env_row = repos_v3.get(conn, "source_envelopes", {"envelope_id": env_id})
        if env_row is None:
            raise VerbatimError(
                ErrorCode.INTEGRITY,
                f"trajectory references missing envelope {env_id!r}",
            )
        src = conn.execute(
            "SELECT external_id FROM sources WHERE source_id = ?",
            (env_row["source_id"],),
        ).fetchone()
        dedup_key = str(src[0]) if src is not None and src[0] is not None else ""
        receipts.append(
            receipt_for_envelope(conn, env_row, dedup_key=dedup_key).to_dict()
        )
        if env_row.get("artifact_ref"):
            artifact_ids.append(env_row["artifact_ref"])

    # --- manifest --------------------------------------------------------------
    created = int(traj["created_event"] or 0)
    completed = traj["completed_event"]
    source_range = (created, int(completed) if completed is not None else created)
    stage_values = tuple(stages) if tuple(stages) else _DEFAULT_STAGES
    material = {
        "trajectory_id": trajectory_id,
        "steps": [s["step_id"] for s in steps],
        "anchors": [a["anchor_id"] for a in anchors],
        "receipts": [r["receipt_id"] for r in receipts],
        "scope_id": traj["scope_id"],
    }
    manifest_id = _manifest_id(material)
    if sandbox_ref is None:
        sandbox_ref = f"sandbox://replay/{manifest_id}"

    detail = {
        "manifest_id": manifest_id,
        "trajectory_id": trajectory_id,
        "source_range": list(source_range),
        "policy_revision": policy_revision,
        "artifact_ids": artifact_ids,
        "simulated_clock_us": None,
        "scope_ids": [traj["scope_id"]],
        "taint_state": "recorded",
        "suppression_seq": _suppression_seq(conn),
        "controller_revision": controller_revision,
        "stages": list(stage_values),
        "environment_digest": traj["environment_digest"],
        "sandbox_ref": sandbox_ref,
        "steps": steps,
        "anchors": anchors,
        "receipts": receipts,
        "note": _SANDBOX_NOTE,
    }

    existing = repos_v3.get(conn, "replay_runs", {"run_id": manifest_id})
    if existing is None:
        repos_v3.insert(
            conn,
            "replay_runs",
            {
                "run_id": manifest_id,
                "manifest_json": detail,
                "report_json": {
                    "status": "manifest_recorded",
                    "step_count": len(steps),
                    "anchor_count": len(anchors),
                    "receipt_count": len(receipts),
                },
                "sandbox_ref": sandbox_ref,
                "created_us": now_us(),
            },
        )

    return ReplayManifest(
        manifest_id=manifest_id,
        source_range=source_range,
        policy_revision=policy_revision,
        artifact_ids=tuple(artifact_ids),
        simulated_clock_us=None,
        scope_ids=(traj["scope_id"],),
        taint_state="recorded",
        suppression_seq=detail["suppression_seq"],
        controller_revision=controller_revision,
        stages=tuple(stage_values),
    )
