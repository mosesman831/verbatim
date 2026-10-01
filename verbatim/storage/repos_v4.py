"""V4 repositories: row-level access for the SPEC_V4 §41 tables.

Same contract as ``repos_v3``: reads return plain ``dict`` snapshots,
mutating helpers take the caller's transaction ``conn``, every statement is
parameterized, and every column name comes from the fixed ``_COLUMNS``
allowlist — caller text never becomes SQL structure.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

from ..core.types import ErrorCode, VerbatimError, json_dumps, safe_json_loads

_COLUMNS: dict[str, frozenset] = {
    "objects": frozenset({
        "object_id", "kind", "scope_id", "current_revision",
        "disposition", "created_event",
    }),
    "object_revisions": frozenset({
        "kind", "object_id", "revision", "digest", "recorded_from",
        "recorded_until", "producer_ref", "metadata_json",
    }),
    "dependency_edges": frozenset({
        "child_kind", "child_id", "child_revision", "parent_kind",
        "parent_id", "parent_revision", "role", "producer_id",
        "operation_id", "seq",
    }),
    "operation_receipts": frozenset({
        "operation_id", "scope_id", "input_digest", "result_ref",
        "effects_applied", "jobs_json", "applied_seq", "created_us",
    }),
    "delivery_permits": frozenset({
        "permit_id", "caller_id", "purpose", "epoch_vector_json",
        "payload_digest", "dependency_versions_json", "state",
        "issued_us", "expires_us", "receipt_id",
    }),
    "dispatch_permits": frozenset({
        "permit_id", "recipient", "purpose", "payload_digest",
        "scope_ids_json", "consent_refs_json", "reservation_id",
        "max_spend", "issued_us", "expires_us", "state",
    }),
    "readiness_obligations": frozenset({
        "obligation_id", "receipt_id", "scope_id", "capability",
        "depends_on_json", "state", "error", "created_us", "updated_us",
    }),
    "closure_runs": frozenset({
        "run_id", "scope_id", "roots_json", "erasure_epoch", "phase",
        "boundary_json", "verification_json", "error", "created_us",
        "updated_us",
    }),
    "closure_frontier": frozenset({
        "run_id", "cursor", "object_kind", "object_id", "revision",
        "action", "state", "detail_json",
    }),
    "producer_manifests": frozenset({
        "producer_id", "kind", "artifact_digest", "rubric_digest",
        "config_digest", "schema_version", "license_ref", "health",
        "registered_us",
    }),
    "view_support": frozenset({
        "view_kind", "view_id", "view_revision", "proposition_locator",
        "evidence_kind", "evidence_id", "evidence_revision", "verdict",
    }),
    "schema_operations": frozenset({
        "operation_id", "version_from", "version_to", "phase", "checksum",
        "cursor", "state", "owner_lease", "created_us", "updated_us",
    }),
}

_JSON_SUFFIX = "_json"


def _check_table(table: str) -> frozenset:
    cols = _COLUMNS.get(table)
    if cols is None:
        raise VerbatimError(ErrorCode.VALIDATION, f"unknown v4 table {table!r}")
    return cols


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _row(cur: sqlite3.Cursor) -> Optional[dict[str, Any]]:
    rows = _rows(cur)
    return rows[0] if rows else None


def insert(conn: sqlite3.Connection, table: str, row: dict[str, Any]) -> None:
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
    rows = query(conn, table, where, limit=1)
    return rows[0] if rows else None


def query(conn: sqlite3.Connection, table: str,
          where: Optional[dict[str, Any]] = None, *,
          order: Optional[str] = None, limit: Optional[int] = None,
          offset: int = 0) -> list[dict[str, Any]]:
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


def update(conn: sqlite3.Connection, table: str,
           set_: dict[str, Any], where: dict[str, Any]) -> int:
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
    cols = _check_table(table)
    if not where:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unconditional DELETE refused"
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
    raw = row.get(key)
    if raw is None:
        return default
    val = safe_json_loads(raw)
    return val if val is not None else default
