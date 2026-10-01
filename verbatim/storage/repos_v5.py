"""V5 repositories: row-level access for the v5 source-projection tables
(docs/v5_contracts.md §3).

Same contract as ``repos_v4``: reads return plain ``dict`` snapshots,
mutating helpers take the caller's transaction ``conn``, every statement is
parameterized, and every column name comes from the fixed ``_COLUMNS``
allowlist — caller text never becomes SQL structure.

The allowlist covers every real table the v5 schema adds, including the
FTS carrier pair (``source_fts_rows``/``source_fts``) so the projection
jobs write them through the same audited path. ``source_fts_idx`` is a
trigger-maintained FTS5 virtual table — it is intentionally NOT in the
allowlist: its only legal writes are the mirroring triggers (plus the
``'delete'`` command, which does not fit this CRUD shape), and reads go
through MATCH queries in the retrieval lane.

These are derived projections of retained source bytes: deletion closure
and rebuilds write here; nothing in this module grants itself meaning —
producers pin revisions, generations, and digests in the row values.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

from ..core.types import ErrorCode, VerbatimError, json_dumps, safe_json_loads

_COLUMNS: dict[str, frozenset] = {
    # source_state/v1 control artifact (§14.3) — CAS via control_version.
    "source_state": frozenset({
        "source_id", "namespace", "control_version", "mutation_head",
        "disposition", "superseded_by", "effective_at", "known_at",
        "valid_from", "valid_to", "updated_at", "producer",
    }),
    "source_lexical_projection": frozenset({
        "source_id", "revision", "scope_id", "generation",
        "tokens", "doc_len", "digest",
    }),
    # FTS shadow carrier pair (source_fts_idx is trigger-maintained and
    # deliberately excluded — see module docstring).
    "source_fts_rows": frozenset({
        "row_id", "source_id", "revision", "scope_id", "generation",
    }),
    "source_fts": frozenset({"fts_row_id", "text"}),
    "source_vectors": frozenset({
        "source_id", "revision", "namespace", "encoder",
        "generation", "vector", "digest",
    }),
    "entity_postings": frozenset({
        "namespace", "entity", "entity_kind", "source_id",
        "revision", "offsets", "generation",
    }),
    "duplicate_links": frozenset({
        "source_id", "revision", "group_id", "method",
        "score", "created_at",
    }),
    "enrichment": frozenset({
        "source_id", "revision", "producer", "type", "polarity",
        "time_precision", "time_status", "event_at", "anchor_at",
        "fields_json",
    }),
    "update_candidates": frozenset({
        "candidate_id", "namespace", "new_source_id", "new_revision",
        "prior_source_id", "prior_revision", "relation", "score",
        "state", "created_at",
    }),
    "backfill_cursor": frozenset({
        "job_key", "last_source_id", "generation", "done", "updated_at",
    }),
}

_JSON_SUFFIX = "_json"


def _check_table(table: str) -> frozenset:
    cols = _COLUMNS.get(table)
    if cols is None:
        raise VerbatimError(ErrorCode.VALIDATION, f"unknown v5 table {table!r}")
    return cols


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _row(cur: sqlite3.Cursor) -> Optional[dict[str, Any]]:
    rows = _rows(cur)
    return rows[0] if rows else None


def insert(conn: sqlite3.Connection, table: str, row: dict[str, Any]) -> None:
    """INSERT one row; every key must be in the table's column allowlist."""
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


def upsert(conn: sqlite3.Connection, table: str, row: dict[str, Any]) -> None:
    """INSERT OR REPLACE — legal here because every v5 table's PRIMARY KEY
    is declared on the row's own columns; the FTS carrier pair is exempt
    anyway (reprojection is DELETE+INSERT, never REPLACE of a rowid that
    the shadow index points at)."""
    cols = _check_table(table)
    bad = set(row) - cols
    if bad:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"{table}: unknown columns {sorted(bad)}"
        )
    names = sorted(row)
    stmt = (
        f"INSERT OR REPLACE INTO {table} ({', '.join(names)}) "
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
    """SELECT with equality predicates from the allowlist. ``order`` is a
    column name optionally suffixed by `` DESC``; ``where`` values of None
    compile to ``IS NULL``."""
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
    """DELETE matched rows. The v5 tables are derived projections — the
    callers are deletion closure and rebuild paths (contracts §3)."""
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
