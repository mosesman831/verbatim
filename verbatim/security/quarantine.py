"""Quarantine workflow (SPEC_V3 §14.10, §34.02–§34.03).

A quarantine row isolates one object revision — keyed by
``(object_kind, object_id, revision)`` — until an authorized reviewer
decides. States:

* ``pending``    — open, awaiting review. Excluded from retrieval.
* ``suppressed`` — reviewer-confirmed permanent hold. Excluded.
* ``released``   — scoped release decision; the object is visible again.
* ``purged``     — content destroyed by privacy; row is a tombstone.

The retrieval contract (V3-14.10): quarantined content is excluded from
ordinary retrieval, consolidation, procedures, sharing, and exports.
``is_quarantined`` / ``should_exclude`` implement the invisibility rule —
``pending`` and ``suppressed`` both hide the object; only an explicit
``release`` restores it. Review shows reason codes, trust, and findings;
release, suppress, and purge are separate audited actions (V3-34.03).
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional, Union

from ..core.types import ErrorCode, VerbatimError, json_dumps, require_id
from ..storage import repos_v3

#: States that hide the object from ordinary retrieval (V3-14.10).
EXCLUDING_STATES = frozenset({"pending", "suppressed"})

#: Decisions a reviewer may record (§34.03: separate audited actions).
DECISIONS = frozenset({"release", "suppress", "purge"})

# object_ref accepted shapes: a (kind, id, revision) or
# (kind, id, revision, scope_id) tuple, or a dict carrying those keys.
ObjectRef = Union[tuple, dict[str, Any]]


def _next_event_seq(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(event_seq), 0) + 1 FROM events"
    ).fetchone()
    return int(row[0])


def _parse_ref(
    object_ref: ObjectRef, scope_id: Optional[str]
) -> tuple[str, str, int, str]:
    """Normalize the object reference; ``scope_id`` kwarg wins over a
    ref-carried value; the column is NOT NULL so it must resolve."""
    if isinstance(object_ref, dict):
        kind = object_ref.get("object_kind")
        oid = object_ref.get("object_id")
        rev = object_ref.get("revision")
        sid = object_ref.get("scope_id")
    elif isinstance(object_ref, (tuple, list)) and len(object_ref) in (3, 4):
        kind, oid, rev = object_ref[0], object_ref[1], object_ref[2]
        sid = object_ref[3] if len(object_ref) == 4 else None
    else:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "object_ref must be (kind, id, revision[, scope_id]) or a dict",
        )
    require_id(kind, "object_kind")
    require_id(oid, "object_id")
    if isinstance(rev, bool) or not isinstance(rev, int) or rev < 0:
        raise VerbatimError(
            ErrorCode.VALIDATION, "object_ref revision must be a non-negative int"
        )
    resolved_scope = scope_id if scope_id is not None else sid
    if resolved_scope is not None:
        require_id(resolved_scope, "scope_id")
    return kind, oid, int(rev), resolved_scope


def get_quarantine(
    conn: sqlite3.Connection, object_kind: str, object_id: str, revision: int
) -> Optional[dict[str, Any]]:
    """Quarantine row for one object revision; ``None`` when absent."""
    row = repos_v3.get(
        conn,
        "quarantine",
        {"object_kind": object_kind, "object_id": object_id, "revision": revision},
    )
    if row is None:
        return None
    row["reason_codes"] = repos_v3.json_field(row, "reason_codes_json", [])
    row["findings"] = repos_v3.json_field(row, "findings_json", [])
    row["decision"] = repos_v3.json_field(row, "decision_json", {})
    return row


def open_quarantine(
    conn: sqlite3.Connection,
    object_ref: ObjectRef,
    reason_codes: list[str],
    findings: Optional[list[dict[str, Any]]],
    opened_event: Optional[int] = None,
    *,
    scope_id: Optional[str] = None,
) -> bool:
    """Open a quarantine hold; returns True when a row was created.

    Idempotent on the primary key — a second screen of the same object
    revision records nothing twice (job dedup + idempotent effects,
    V3-40.01). ``reason_codes`` names *why* the hold exists (e.g.
    ``"attack_risk:blocked"``, ``"rule:boundary_redirection.ignore_prior"``)
    — §34.03 requires review to show reason codes, not a bare flag.
    ``opened_event`` defaults to the estimated next event_seq so the hold
    lands inside the caller's event order.
    """
    kind, oid, rev, sid = _parse_ref(object_ref, scope_id)
    if sid is None:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "scope_id is required to open a quarantine row",
        )
    codes = list(reason_codes or [])
    if not codes or not all(isinstance(c, str) and c for c in codes):
        raise VerbatimError(
            ErrorCode.VALIDATION, "reason_codes must be non-empty strings"
        )
    existing = repos_v3.get(
        conn,
        "quarantine",
        {"object_kind": kind, "object_id": oid, "revision": rev},
    )
    if existing is not None:
        return False
    repos_v3.insert(
        conn,
        "quarantine",
        {
            "object_kind": kind,
            "object_id": oid,
            "revision": rev,
            "scope_id": sid,
            "reason_codes_json": codes,
            "findings_json": list(findings or []),
            "state": "pending",
            "opened_event": (
                _next_event_seq(conn) if opened_event is None else int(opened_event)
            ),
            "decision_json": {},
        },
    )
    return True


def _row_state(conn: sqlite3.Connection, kind: str, oid: str, rev: int) -> Optional[str]:
    row = repos_v3.get(
        conn,
        "quarantine",
        {"object_kind": kind, "object_id": oid, "revision": rev},
    )
    return row["state"] if row else None


def is_quarantined(
    conn: sqlite3.Connection, object_kind: str, object_id: str, revision: int
) -> bool:
    """True while the object is hidden — ``pending`` or ``suppressed``
    (V3-14.10). ``released`` and ``purged`` do not quarantine for retrieval
    purposes (a purged object is gone through deletion closure, §36)."""
    return _row_state(conn, object_kind, object_id, revision) in EXCLUDING_STATES


def should_exclude(
    conn: sqlite3.Connection, object_kind: str, object_id: str, revision: int
) -> bool:
    """Retrieval invisibility contract: True → the caller must withhold the
    item from ordinary context (V3-14.10, V3-31.02)."""
    return is_quarantined(conn, object_kind, object_id, revision)


# Chunk bound for the OR-chained ref set — 3 bound params per ref keeps
# each statement far below SQLITE_MAX_VARIABLE_NUMBER.
_REF_CHUNK = 150


def excluded_refs(
    conn: sqlite3.Connection, refs: Any
) -> "set[tuple[str, str, int]]":
    """Batch form of ``should_exclude``: the subset of ``refs`` currently
    held (``pending``/``suppressed``).

    ``refs`` is an iterable of ``(object_kind, object_id, revision)``
    triples. One chunked table pass replaces a row probe per candidate —
    the verdict per ref is identical to ``should_exclude`` (V3-14.10,
    V3-31.02). The cheap liveness probe short-circuits an empty hold set
    without touching the ref build.
    """
    triples = {
        (kind, oid, int(rev)) for kind, oid, rev in refs
    }
    if not triples:
        return set()
    live = conn.execute(
        "SELECT 1 FROM quarantine"
        " WHERE state IN ('pending','suppressed') LIMIT 1"
    ).fetchone()
    if live is None:
        return set()
    out: set = set()
    ordered = sorted(triples)
    for i in range(0, len(ordered), _REF_CHUNK):
        part = ordered[i:i + _REF_CHUNK]
        where = " OR ".join(
            "(object_kind = ? AND object_id = ? AND revision = ?)"
            for _ in part
        )
        out.update(
            (r[0], r[1], int(r[2]))
            for r in conn.execute(
                "SELECT object_kind, object_id, revision FROM quarantine"
                " WHERE state IN ('pending','suppressed')"
                " AND (" + where + ")",
                [v for triple in part for v in triple],
            ).fetchall()
        )
    return out


def _decide(
    conn: sqlite3.Connection,
    object_ref: ObjectRef,
    decided_by: str,
    decision: Optional[dict[str, Any]],
    state: str,
    action: str,
    scope_id: Optional[str],
) -> dict[str, Any]:
    kind, oid, rev, _sid = _parse_ref(object_ref, scope_id)
    require_id(decided_by, "decided_by")
    if decision is not None and not isinstance(decision, dict):
        raise VerbatimError(ErrorCode.VALIDATION, "decision must be a mapping")
    payload = dict(decision or {})
    payload["action"] = action
    try:
        json_dumps(payload)
    except (TypeError, ValueError) as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"decision is not JSON-serializable: {exc}"
        ) from exc
    row = repos_v3.get(
        conn,
        "quarantine",
        {"object_kind": kind, "object_id": oid, "revision": rev},
    )
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
            "no quarantine row for object revision",
        )
    repos_v3.update(
        conn,
        "quarantine",
        {
            "state": state,
            "decided_event": _next_event_seq(conn),
            "decided_by": decided_by,
            "decision_json": payload,
        },
        {"object_kind": kind, "object_id": oid, "revision": rev},
    )
    out = repos_v3.get(
        conn,
        "quarantine",
        {"object_kind": kind, "object_id": oid, "revision": rev},
    )
    return out if out is not None else {}


def release(
    conn: sqlite3.Connection,
    object_ref: ObjectRef,
    decided_by: str,
    decision: Optional[dict[str, Any]] = None,
    *,
    scope_id: Optional[str] = None,
) -> dict[str, Any]:
    """Scoped release decision (§14.05): the hold lifts; the label's origin
    and findings are untouched — review approves a use, it never relabels
    the source ``principal_direct`` (V3-14.04)."""
    return _decide(conn, object_ref, decided_by, decision, "released", "release", scope_id)


def suppress(
    conn: sqlite3.Connection,
    object_ref: ObjectRef,
    decided_by: str,
    decision: Optional[dict[str, Any]] = None,
    *,
    scope_id: Optional[str] = None,
) -> dict[str, Any]:
    """Permanent hold: the finding stands; the object stays invisible to
    ordinary retrieval (§34.03)."""
    return _decide(conn, object_ref, decided_by, decision, "suppressed", "suppress", scope_id)


def mark_purged(
    conn: sqlite3.Connection,
    object_ref: ObjectRef,
    decided_by: str,
    decision: Optional[dict[str, Any]] = None,
    *,
    scope_id: Optional[str] = None,
) -> dict[str, Any]:
    """Tombstone after privacy purge — content destruction is privacy's
    job; this records the quarantine side of the same audited action."""
    return _decide(conn, object_ref, decided_by, decision, "purged", "purge", scope_id)


def pending_items(
    conn: sqlite3.Connection,
    scope_id: Optional[str] = None,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Open holds for the review surface (§34.03 shows reason codes, trust,
    findings — the raw rows; authz filtering is governance's job)."""
    where = {"state": "pending"}
    if scope_id is not None:
        require_id(scope_id, "scope_id")
        where["scope_id"] = scope_id
    rows = repos_v3.query(conn, "quarantine", where, order="opened_event", limit=limit)
    for row in rows:
        row["reason_codes"] = repos_v3.json_field(row, "reason_codes_json", [])
        row["findings"] = repos_v3.json_field(row, "findings_json", [])
    return rows


__all__ = [
    "DECISIONS",
    "EXCLUDING_STATES",
    "get_quarantine",
    "is_quarantined",
    "mark_purged",
    "open_quarantine",
    "pending_items",
    "release",
    "should_exclude",
    "suppress",
]
