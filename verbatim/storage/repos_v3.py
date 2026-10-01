"""V3 repositories: row-level access for the SPEC_V3 §39 tables.

Same contract as ``repos`` (SPEC §19-20, §41): mutable rows never escape as
objects — reads return plain ``dict`` snapshots; mutating helpers take the
caller's transaction ``conn``; every statement is parameterized and every
column name comes from the fixed ``_COLUMNS`` allowlist below — caller text
never becomes SQL structure.

Domain logic (grant evaluation, vault sealing, compilation) lives in the
feature modules; this file is the mechanical CRUD they share so no two
modules write SQL for the same table twice.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable, Optional

from ..core.types import ErrorCode, VerbatimError, json_dumps, safe_json_loads


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _row(cur: sqlite3.Cursor) -> Optional[dict[str, Any]]:
    rows = _rows(cur)
    return rows[0] if rows else None


# Fixed column allowlists per v3 table (§41). JSON-typed columns are marked
# in _JSON_COLUMNS so callers pass native values, not pre-serialized text —
# serialization happens in exactly one place.
_COLUMNS: dict[str, frozenset] = {
    "principals": frozenset({
        "principal_id", "kind", "display_name", "host_binding",
        "created_us", "retired",
    }),
    "perspectives": frozenset({
        "perspective_id", "scope_id", "asserter", "observer",
        "audience_json", "created_event",
    }),
    "perspective_subjects": frozenset({"perspective_id", "subject_id"}),
    "purposes": frozenset({"purpose", "description", "registered_us", "retired"}),
    "grants_v3": frozenset({
        "grant_id", "scope_id", "principal_id", "verbs_json",
        "purposes_json", "caveats_json", "delegation_depth", "issuer_id",
        "issued_us", "expires_us", "revoked_us", "epoch",
    }),
    "delegations": frozenset({
        "delegation_id", "parent_grant_id", "child_grant_id",
        "delegator_id", "delegate_id", "created_us", "expires_us",
        "revoked_us",
    }),
    "capture_authorizations": frozenset({
        "authorization_id", "issuer_id", "principal_id",
        "allowed_kinds_json", "scope_ids_json", "retention_policy",
        "policy_revision", "issued_us", "expires_us", "revoked_us",
    }),
    "source_envelopes": frozenset({
        "envelope_id", "source_id", "revision", "scope_id",
        "envelope_kind", "actor_principal", "perspective_id", "event_us",
        "receipt_us", "media_type", "trust_class", "capture_proof",
        "redaction_status", "adapter_version", "host_id", "session_id",
        "task_id", "step_id", "artifact_ref", "metadata_json",
    }),
    "trajectories": frozenset({
        "trajectory_id", "scope_id", "host_id", "session_id", "task_id",
        "boundary_rule", "created_event", "completed_event",
        "environment_digest", "metadata_json",
    }),
    "trajectory_steps": frozenset({
        "step_id", "trajectory_id", "scope_id", "ord",
        "action_envelope_id", "environment_digest", "metadata_json",
    }),
    "step_observations": frozenset({"step_id", "envelope_id", "ord"}),
    "state_anchors": frozenset({
        "anchor_id", "scope_id", "kind", "ref", "digest", "step_id",
        "created_event",
    }),
    "transitions": frozenset({
        "transition_id", "episode_id", "scope_id", "ord",
        "action_step_id", "checker_ref", "environment_digest", "edge",
        "created_event",
    }),
    "transition_anchors": frozenset({
        "transition_id", "role", "anchor_id", "ord",
    }),
    "procedure_signatures": frozenset({
        "procedure_id", "revision", "signature_digest", "ordered_ops_json",
        "intent_key", "scope_id",
    }),
    "procedure_exposures": frozenset({
        "exposure_id", "procedure_id", "revision", "scope_id", "task_id",
        "session_id", "outcome", "environment_digest", "recorded_us",
    }),
    "observations": frozenset({
        "observation_id", "scope_id", "revision", "text", "proof_count",
        "perspective_id", "freshness", "stale_since_seq", "producer",
        "recorded_from", "recorded_until",
    }),
    "observation_evidence": frozenset({
        "observation_id", "revision", "role", "object_kind", "object_id",
        "object_revision",
    }),
    "derivations": frozenset({
        "child_kind", "child_id", "child_revision", "parent_kind",
        "parent_id", "parent_revision", "producer_kind", "producer_id",
        "seq", "scope_id",
    }),
    "security_labels": frozenset({
        "label_id", "scope_id", "source_trust", "content_form",
        "attack_risk", "review_state", "findings_json", "method",
        "rules_revision", "created_event",
    }),
    "quarantine": frozenset({
        "object_kind", "object_id", "revision", "scope_id",
        "reason_codes_json", "findings_json", "state", "opened_event",
        "decided_event", "decided_by", "decision_json",
    }),
    "vault_entries": frozenset({
        "entry_id", "scope_id", "revision", "sensitivity", "placeholder",
        "algorithm", "key_version", "wrap_key_version", "nonce",
        "ciphertext", "wrapped_key", "aad_digest", "detection",
        "created_event", "erased_event",
    }),
    "vault_refs": frozenset({
        "placeholder", "entry_id", "scope_id", "view_id",
        "start_byte", "end_byte",
    }),
    "action_tickets": frozenset({
        "ticket_id", "scope_id", "recipient_id", "action_digest",
        "purpose", "epoch", "nonce", "issued_us", "expires_us",
        "gateway_id", "consumed_us",
    }),
    "ticket_objects": frozenset({"ticket_id", "object_id", "revision"}),
    "value_handles": frozenset({
        "handle_id", "vault_entry_id", "scope_id", "recipient_id",
        "action_digest", "consent_id", "expires_us", "consumed_us",
    }),
    "redaction_spans": frozenset({
        "scope_id", "view_id", "orig_start", "orig_end",
        "accepted_start", "accepted_end", "sensitivity", "entry_id",
    }),
    "propagations": frozenset({
        "propagation_id", "scope_id", "object_kind", "object_id",
        "revision", "recipient_id", "verbs_json", "purpose", "epoch",
        "created_us", "capsule_id", "revoked_seq", "acknowledged",
    }),
    "routing_decisions": frozenset({
        "decision_id", "scope_id", "state_key", "routes_json",
        "lane_set_json", "budgets_json", "policy_revision",
        "result_sizes_json", "outcome_credit", "created_us",
    }),
    "routing_stats": frozenset({
        "scope_id", "state_key", "action_key", "exposures",
        "credit_json", "policy_revision",
    }),
    "influence": frozenset({
        "handle_id", "receipt_id", "scope_id", "caller_id", "epoch",
        "pack", "object_kind", "object_id", "revision", "feedback_kind",
        "action_receipt", "created_us", "redacted",
    }),
    "freshness": frozenset({
        "scope_id", "object_kind", "object_id", "revision", "class",
        "revalidate_after_us", "anchor_refs_json",
    }),
    "environment_state": frozenset({
        "scope_id", "key", "value", "anchor_id", "observed_us", "volatile",
    }),
    "working_sets": frozenset({
        "set_id", "scope_id", "session_id", "created_us", "expires_us",
    }),
    "working_set_items": frozenset({
        "item_id", "set_id", "kind", "object_ref", "text", "ord",
    }),
    "social_memory": frozenset({
        "record_id", "scope_id", "observer_id", "subject_id", "kind",
        "value_json", "evidence_json", "revision", "created_us",
    }),
    "learning_snapshots": frozenset({
        "snapshot_id", "seq", "digest", "created_us", "validation",
    }),
    "index_generations": frozenset({
        "generation_id", "kind", "scope_partition", "manifest_json",
        "snapshot_seq", "status", "created_us",
    }),
    "replay_runs": frozenset({
        "run_id", "manifest_json", "report_json", "sandbox_ref",
        "created_us",
    }),
}

_JSON_SUFFIX = "_json"


def _check_table(table: str) -> frozenset:
    cols = _COLUMNS.get(table)
    if cols is None:
        raise VerbatimError(ErrorCode.VALIDATION, f"unknown v3 table {table!r}")
    return cols


def insert(conn: sqlite3.Connection, table: str, row: dict[str, Any]) -> None:
    """INSERT one row; keys outside the table allowlist are rejected."""
    cols = _check_table(table)
    bad = set(row) - cols
    if bad:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"{table}: unknown columns {sorted(bad)}"
        )
    names = sorted(row)
    stmt = (
        f"INSERT INTO {table} ({', '.join(names)}) "
        f"VALUES ({', '.join('?' for _ in names)})"
    )
    values = [
        json_dumps(row[n]) if n.endswith(_JSON_SUFFIX) and not isinstance(row[n], str)
        else row[n]
        for n in names
    ]
    conn.execute(stmt, values)


def get(conn: sqlite3.Connection, table: str,
        where: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Exact-match single row lookup."""
    rows = query(conn, table, where, limit=1)
    return rows[0] if rows else None


def query(conn: sqlite3.Connection, table: str,
          where: Optional[dict[str, Any]] = None, *,
          order: Optional[str] = None, limit: Optional[int] = None,
          offset: int = 0) -> list[dict[str, Any]]:
    """Exact-match query; ``order`` must be a column name (+ optional DESC)."""
    cols = _check_table(table)
    clauses: list[str] = []
    params: list[Any] = []
    for key, value in (where or {}).items():
        if key not in cols:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"{table}: unknown column {key!r}"
            )
        if value is None:
            clauses.append(f"{key} IS NULL")
        else:
            clauses.append(f"{key} = ?")
            params.append(value)
    sql = f"SELECT * FROM {table}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    if order is not None:
        desc = order.endswith(" DESC")
        col = order[:-5] if desc else order
        if col not in cols:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"{table}: unknown order column {order!r}"
            )
        sql += f" ORDER BY {col}{' DESC' if desc else ''}"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
        if offset:
            sql += " OFFSET ?"
            params.append(offset)
    return _rows(conn.execute(sql, params))


def query_in(conn: sqlite3.Connection, table: str, column: str,
             values: Iterable[Any], *, order_rowid: bool = False,
             chunk: int = 200) -> list[dict[str, Any]]:
    """Rows where ``column`` equals any of ``values`` — one IN() query per
    ``chunk`` of distinct values so statement variable counts stay bounded.

    ``order_rowid`` appends ``ORDER BY rowid`` so a "first match" pick by
    the caller reproduces the scan order an unordered ``get``/``LIMIT 1``
    observes; only valid on ordinary rowid tables.
    """
    cols = _check_table(table)
    if column not in cols:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"{table}: unknown column {column!r}"
        )
    vals = list(dict.fromkeys(values))
    out: list[dict[str, Any]] = []
    for i in range(0, len(vals), chunk):
        part = vals[i:i + chunk]
        ph = ",".join("?" for _ in part)
        sql = f"SELECT * FROM {table} WHERE {column} IN ({ph})"
        if order_rowid:
            sql += " ORDER BY rowid"
        out.extend(_rows(conn.execute(sql, part)))
    return out


def update(conn: sqlite3.Connection, table: str,
           set_: dict[str, Any], where: dict[str, Any]) -> int:
    """UPDATE matched rows; returns the affected count. All column names —
    SET and WHERE alike — come from the fixed allowlist."""
    cols = _check_table(table)
    if not set_:
        raise VerbatimError(ErrorCode.VALIDATION, f"{table}: empty SET")
    for key in set_:
        if key not in cols:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"{table}: unknown column {key!r}"
            )
    set_sql = ", ".join(f"{k} = ?" for k in sorted(set_))
    params = [
        json_dumps(set_[k]) if k.endswith(_JSON_SUFFIX) and not isinstance(set_[k], str)
        else set_[k]
        for k in sorted(set_)
    ]
    clauses: list[str] = []
    for key, value in where.items():
        if key not in cols:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"{table}: unknown column {key!r}"
            )
        if value is None:
            clauses.append(f"{key} IS NULL")
        else:
            clauses.append(f"{key} = ?")
            params.append(value)
    sql = f"UPDATE {table} SET {set_sql}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    return conn.execute(sql, params).rowcount


def delete(conn: sqlite3.Connection, table: str,
           where: dict[str, Any]) -> int:
    """DELETE matched rows. Only privacy/lifecycle code should call this —
    learning-plane objects retire, they are not deleted (V3-17.02)."""
    cols = _check_table(table)
    if not where:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"{table}: unconditional DELETE refused"
        )
    clauses: list[str] = []
    params: list[Any] = []
    for key, value in where.items():
        if key not in cols:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"{table}: unknown column {key!r}"
            )
        if value is None:
            clauses.append(f"{key} IS NULL")
        else:
            clauses.append(f"{key} = ?")
            params.append(value)
    sql = f"DELETE FROM {table} WHERE " + " AND ".join(clauses)
    return conn.execute(sql, params).rowcount


def json_field(row: dict[str, Any], key: str, default: Any = None) -> Any:
    """Decode a ``*_json`` column into its native value."""
    raw = row.get(key)
    if raw is None:
        return default
    val = safe_json_loads(raw)
    return val if val is not None else default
