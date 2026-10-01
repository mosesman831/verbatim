"""Procedure reuse accounting (SPEC_V3 §21 reuse_stats row, §22.06–§22.07).

Every exposure a procedure receives — was it shown, judged applicable,
adopted, did it succeed, fail, or end unknown — lands as a
``procedure_exposures`` row and rolls into the procedure's
``reuse_stats_json`` denominators. There is deliberately **no single
self-reinforcing usefulness score** (V3-22.06): counts stay separate per
outcome and per environment class so downstream measurement (negative
transfer, drift) works on honest denominators.

Paired-arm bookkeeping (V3-22.07): callers running controlled reuse
experiments pass ``experiment_id`` + ``arm``; those exposures are counted
under ``reuse_stats.paired[experiment][arm][outcome]`` so negative
transfer is estimated from paired executions — never inferred from mere
adoption/failure correlation.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    new_id,
    require_id,
)
from ..storage.repos import _json_parse, _row, _rows

#: Exposure outcomes — the ``procedure_exposures.outcome`` CHECK domain.
EXPOSURE_OUTCOMES = frozenset({
    "exposed", "applicable", "adopted", "success", "failure", "unknown",
})

_ZERO = {
    "exposed": 0, "applicable": 0, "adopted": 0,
    "success": 0, "failure": 0, "unknown": 0, "total": 0,
}


def _stats(conn: sqlite3.Connection, row: dict[str, Any]) -> dict[str, Any]:
    stats = _json_parse(row.get("reuse_stats_json")) or {}
    for k, v in _ZERO.items():
        stats.setdefault(k, v)
    stats.setdefault("by_environment", {})
    stats.setdefault("paired", {})
    return stats


def record_exposure(
    conn: sqlite3.Connection,
    procedure_id: str,
    task_id: Optional[str] = None,
    outcome: str = "unknown",
    *,
    session_id: Optional[str] = None,
    environment_digest: Optional[str] = None,
    experiment_id: Optional[str] = None,
    arm: Optional[str] = None,
    recorded_us: int = 0,
) -> str:
    """Append one exposure and update reuse denominators atomically.

    Returns the ``exposure_id``. ``experiment_id`` + ``arm`` mark a paired
    measurement arm (V3-22.07); both must be given together.
    """
    require_id(procedure_id, "procedure_id")
    if outcome not in EXPOSURE_OUTCOMES:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"exposure outcome must be one of {sorted(EXPOSURE_OUTCOMES)}",
        )
    if (experiment_id is None) != (arm is None):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "paired accounting needs experiment_id and arm together",
        )
    row = _row(
        conn.execute(
            "SELECT * FROM procedures WHERE procedure_id = ?",
            (procedure_id,),
        )
    )
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "procedure not found"
        )
    exposure_id = new_id()
    conn.execute(
        "INSERT INTO procedure_exposures"
        " (exposure_id, procedure_id, revision, scope_id, task_id,"
        "  session_id, outcome, environment_digest, recorded_us)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (
            exposure_id, procedure_id, int(row["revision"]),
            row["scope_id"], task_id, session_id, outcome,
            environment_digest, int(recorded_us),
        ),
    )
    stats = _stats(conn, row)
    stats[outcome] = int(stats.get(outcome, 0)) + 1
    stats["total"] = int(stats.get("total", 0)) + 1
    env_key = environment_digest or "unrecorded"
    env_stats = dict(stats["by_environment"].get(env_key) or {})
    env_stats[outcome] = int(env_stats.get(outcome, 0)) + 1
    env_stats["total"] = int(env_stats.get("total", 0)) + 1
    stats["by_environment"][env_key] = env_stats
    if experiment_id is not None:
        paired = stats["paired"].setdefault(experiment_id, {})
        arm_stats = dict(paired.get(arm) or {})
        arm_stats[outcome] = int(arm_stats.get(outcome, 0)) + 1
        arm_stats["total"] = int(arm_stats.get("total", 0)) + 1
        paired[arm] = arm_stats
    cur = conn.execute(
        "UPDATE procedures SET reuse_stats_json = ?,"
        " row_version = row_version + 1 WHERE procedure_id = ?",
        (json_dumps(stats), procedure_id),
    )
    if cur.rowcount == 0:
        raise VerbatimError(
            ErrorCode.STALE_PROPOSAL, "procedure row changed under exposure"
        )
    return exposure_id


def exposures_for(
    conn: sqlite3.Connection, procedure_id: str
) -> list[dict[str, Any]]:
    """All recorded exposures for a procedure, insertion-ordered."""
    require_id(procedure_id, "procedure_id")
    return _rows(
        conn.execute(
            "SELECT * FROM procedure_exposures WHERE procedure_id = ?"
            " ORDER BY rowid",
            (procedure_id,),
        )
    )


def reuse_stats(
    conn: sqlite3.Connection, procedure_id: str
) -> dict[str, Any]:
    """Current denominators for one procedure (V3-22.06)."""
    row = _row(
        conn.execute(
            "SELECT reuse_stats_json FROM procedures WHERE procedure_id = ?",
            (require_id(procedure_id, "procedure_id"),),
        )
    )
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "procedure not found"
        )
    return _stats(conn, row)
