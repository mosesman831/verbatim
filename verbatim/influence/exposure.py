"""Source-lane exposure sink (SPEC_V6 V6-03.14, docs/v6_contracts.md §8).

``record_source_deliveries`` writes one ``source_exposure`` row per
delivered item — the source-side twin of the claim-lane ``influence``
rows. Rows are delivery facts only: ``source_id``, ``revision``, the
``score_family`` that ranked it, the ``namespace`` it was delivered
into, and ``delivered_at_us``. Like ``influence``, a missing feedback
kind is recorded *exposure* — never a claim that the item was used
(V3-32.05 semantics carry over).

Write discipline (contract §7.4): the consumer path calls
``emit_deliveries`` AFTER the delivery read tx closes; the hook opens
its own short ``store.tx()`` — the same pattern as
``controller.log_decision`` inside ``recall_v3``'s write phase. This
module never writes inside a read transaction and never fabricates a
receipt: callers without a receipt context pass ``receipt_id=None`` and
the row stores ``''`` — the absence stays honest rather than inventing
an identifier.

Idempotence: ``(receipt_id, ord)`` is the primary key; a replayed
delivery batch (retried search write, crash-recovered emit) is an
insert-ignore no-op, and ``record_source_deliveries`` returns the count
of rows actually written.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable, Mapping, Optional

from ..core.time import now_us
from ..core.types import ErrorCode, VerbatimError, require_id
from ..storage.repos import has_table
from ..storage.schema_v5 import ensure_additive_tables

_TABLE = "source_exposure"

#: Score families a source-lane delivery may declare. ``score_family`` is
#: NOT CHECK-constrained (producers evolve), but the column is required —
#: an empty family would hide *how* the item was ranked, so callers must
#: name something (``"ranking/v1"`` is the consumer-path default).
_DEFAULT_SCORE_FAMILY = "ranking/v1"


def _normalize_deliveries(
    deliveries: Iterable[Mapping[str, Any]],
) -> list[tuple[str, int, str]]:
    """Validate + project the delivery dicts into row tuples.

    Each delivery names ``source_id``, ``revision``, and (optionally)
    ``score_family``. Dicts and duck-typed objects exposing
    ``source_id``/``revision``/``score_family`` attributes (e.g. the
    source lane's ``SourceHit`` candidates, or ``memory_id``-carrying
    hits) are both accepted; callers whose delivery objects don't expose
    those attributes map them to dicts first.
    """
    rows: list[tuple[str, int, str]] = []
    for i, d in enumerate(deliveries or ()):  # ord = delivery order
        if isinstance(d, Mapping):
            source_id = d.get("source_id")
            revision = d.get("revision")
            score_family = d.get("score_family")
        else:
            # Duck-typed delivery records (Hit): memory_id is the source
            # id on source-lane hits.
            source_id = getattr(d, "source_id", None) or getattr(
                d, "memory_id", None
            )
            revision = getattr(d, "revision", None)
            score_family = getattr(d, "score_family", None)
        if not isinstance(source_id, str) or not source_id:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"deliveries[{i}].source_id must be a non-empty string",
            )
        require_id(source_id, "source_id")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"deliveries[{i}].revision must be a non-negative int",
            )
        if score_family is None:
            score_family = _DEFAULT_SCORE_FAMILY
        if not isinstance(score_family, str) or not score_family:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"deliveries[{i}].score_family must be a non-empty string",
            )
        rows.append((source_id, revision, score_family))
    return rows


def record_source_deliveries(
    conn: sqlite3.Connection,
    receipt_id: Optional[str] = None,
    deliveries: Optional[Iterable[Mapping[str, Any]]] = None,
    *,
    namespace: Optional[str] = None,
) -> int:
    """Append one ``source_exposure`` row per delivered item.

    ``conn`` is the caller's WRITE transaction — the function commits
    nothing (same contract as ``controller.log_decision`` /
    ``influence.record_deliveries``), so the exposure journal lands
    atomically with whatever else the post-delivery write carries.

    Both call shapes are honored: the contract form
    ``record_source_deliveries(conn, receipt_id, deliveries)``
    (docs/v6_contracts.md §8) and the keyword form
    ``record_source_deliveries(conn, receipt_id=..., namespace=...,
    deliveries=...)`` — ``deliveries`` is positional-second so a caller
    without a namespace context never has to fabricate one.

    ``receipt_id=None`` is legal: the consumer path may emit before a
    receipt exists; the rows store ``''`` rather than a fabricated id.
    ``ord`` is the item's position in ``deliveries``. Returns the number
    of rows actually inserted (idempotent replays return 0 or the
    shortfall).

    The additive table is ensured inside this transaction — a
    ``Store.create``-fresh store gains it atomically with its first
    delivery batch (``ensure_additive_tables`` never commits early).
    """
    rows = _normalize_deliveries(deliveries)
    if not rows:
        return 0
    rid = receipt_id if isinstance(receipt_id, str) else (receipt_id or "")
    if not isinstance(rid, str):
        raise VerbatimError(ErrorCode.VALIDATION, "receipt_id must be a string")
    ns = namespace or ""
    if not isinstance(ns, str):
        raise VerbatimError(ErrorCode.VALIDATION, "namespace must be a string")
    ensure_additive_tables(conn)
    stamped = now_us()
    cur = conn.executemany(
        "INSERT OR IGNORE INTO source_exposure"
        " (receipt_id, ord, source_id, revision, score_family,"
        "  delivered_at_us, namespace)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (rid, ord_, sid, rev, fam, stamped, ns)
            for ord_, (sid, rev, fam) in enumerate(rows)
        ],
    )
    return int(cur.rowcount if cur.rowcount is not None else 0)


def emit_deliveries(
    store: Any,
    receipt_id: Optional[str] = None,
    namespace: Optional[str] = None,
    deliveries: Iterable[Mapping[str, Any]] = (),
) -> int:
    """Post-delivery emission hook for the consumer path (contract §7.4).

    ``facade.search`` calls this AFTER the delivery read tx closes; it
    opens its own short ``store.tx()`` — never runs inside the read tx —
    and returns the rows written. ``receipt_id=None`` (no receipt
    context, e.g. a search-minted token that never persisted) stores
    ``''`` honestly.

    Store-side failures (busy, readonly, closed) propagate as the
    store's own typed errors — the hook never swallows them and never
    reports a write it could not commit; whether a failed exposure write
    should degrade the search is the caller's policy choice.
    """
    rows = _normalize_deliveries(deliveries)
    if not rows:
        return 0
    with store.tx() as conn:
        return record_source_deliveries(
            conn,
            receipt_id=receipt_id,
            namespace=namespace,
            deliveries=[
                {"source_id": s, "revision": r, "score_family": f}
                for s, r, f in rows
            ],
        )


def exposure_count(
    conn: sqlite3.Connection,
    receipt_id: Optional[str] = None,
    namespace: Optional[str] = None,
) -> int:
    """Count recorded source-lane exposures (read snapshot).

    ``receipt_id``/``namespace`` filter independently; neither required.
    A store that predates the additive ensure — or one that simply never
    emitted — reports 0 (the table's absence means zero deliveries, not
    an error).
    """
    if not has_table(conn, _TABLE):
        return 0
    where: list[str] = []
    params: list[Any] = []
    if receipt_id is not None:
        where.append("receipt_id = ?")
        params.append(receipt_id)
    if namespace is not None:
        where.append("namespace = ?")
        params.append(namespace)
    sql = "SELECT COUNT(*) FROM source_exposure"
    if where:
        sql += " WHERE " + " AND ".join(where)
    row = conn.execute(sql, params).fetchone()
    return int(row[0]) if row else 0


def recent_exposures(
    conn: sqlite3.Connection,
    namespace: str,
    limit: int = 32,
) -> list[dict]:
    """Newest-first ``source_exposure`` rows for one namespace — the
    probe surface ``eval/v5/feedback.py`` and audits use. Plain dicts,
    newest ``delivered_at_us`` first; empty when the table is absent or
    the namespace has no deliveries.
    """
    if not isinstance(namespace, str) or not namespace:
        raise VerbatimError(
            ErrorCode.VALIDATION, "namespace must be a non-empty string"
        )
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise VerbatimError(
            ErrorCode.VALIDATION, "limit must be a positive int"
        )
    if not has_table(conn, _TABLE):
        return []
    rows = conn.execute(
        "SELECT receipt_id, ord, source_id, revision, score_family,"
        " delivered_at_us, namespace FROM source_exposure"
        " WHERE namespace = ?"
        " ORDER BY delivered_at_us DESC, receipt_id DESC, ord DESC"
        " LIMIT ?",
        (namespace, int(limit)),
    ).fetchall()
    return [
        {
            "receipt_id": r[0],
            "ord": r[1],
            "source_id": r[2],
            "revision": r[3],
            "score_family": r[4],
            "delivered_at_us": r[5],
            "namespace": r[6],
        }
        for r in rows
    ]
