"""Learning snapshots for rollback / poisoning recovery (SPEC_V3 §34.02, §39).

A ``learning_snapshots`` row pins ``(seq, digest)``: ``seq`` is the store-wide
derivation-sequence watermark at snapshot time, and ``digest`` is a
deterministic SHA-256 over the scope's derivation edges plus the derived
objects they name — tamper evidence for the snapshot, never a claim of
guaranteed truth (V3-34.02: snapshots record sequence and validation
status, not truth).

``rollback_plan`` is the honest inverse: every derived object *created after*
the watermark, which is what a poisoning rollback must reconsider. Rollback
restores selected derived revisions only after applying current grants,
retention, erasure, and risk dispositions — that filtering is the caller's
job; this module only enumerates the candidate set. Source revisions
implicated by poisoned derivations stay quarantined from producers until
review; recomputing from unchanged poisoned evidence cannot silently
recreate removed effects (V3-34.02).
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any, Union

from ..core.time import wall_us
from ..core.types import (
    ErrorCode,
    Scope,
    VerbatimError,
    json_dumps,
    new_id,
    require_id,
)
from ..storage import repos_v3


def _scope_id(scope: Union[Scope, str]) -> str:
    if isinstance(scope, Scope):
        from ..core.identity import scope_key

        return scope_key(scope)
    return require_id(scope, "scope_id")


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def take_snapshot(
    conn: sqlite3.Connection,
    scope: Union[Scope, str],
    *,
    validation: str = "unvalidated",
) -> str:
    """Pin the scope's current derivation state; returns ``snapshot_id``.

    ``seq`` is the store-wide max ``derivations.seq`` watermark (a global
    creation sequence — scope-local watermarks would let foreign writes
    slip under the rollback line). ``digest`` covers this scope's edges
    and the derived-object refs they name, canonicalized so the same graph
    always produces the same digest.
    """
    sid = _scope_id(scope)
    if validation not in ("unvalidated", "validated", "rejected"):
        raise VerbatimError(
            ErrorCode.VALIDATION, f"invalid validation {validation!r}"
        )
    seq = int(
        conn.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM derivations"
        ).fetchone()[0]
    )
    edges = repos_v3.query(
        conn, "derivations", {"scope_id": sid}, order="seq"
    )
    canonical_edges = [
        [
            r["child_kind"], r["child_id"], int(r["child_revision"]),
            r["parent_kind"], r["parent_id"], int(r["parent_revision"]),
            r["producer_kind"], r["producer_id"], int(r["seq"]),
        ]
        for r in edges
    ]
    derived_refs = sorted(
        {
            (r["child_kind"], r["child_id"], int(r["child_revision"]))
            for r in edges
        }
    )
    digest = hashlib.sha256(
        json_dumps(
            {
                "scope_id": sid,
                "seq": seq,
                "edges": canonical_edges,
                "derived": [list(d) for d in derived_refs],
            }
        ).encode("utf-8")
    ).digest()
    snapshot_id = new_id()
    repos_v3.insert(
        conn,
        "learning_snapshots",
        {
            "snapshot_id": snapshot_id,
            "seq": seq,
            "digest": digest,
            "created_us": wall_us(),
            "validation": validation,
        },
    )
    return snapshot_id


def rollback_plan(
    conn: sqlite3.Connection, snapshot_id: str
) -> dict[str, Any]:
    """Derived objects created after the snapshot's watermark.

    Returns ``{"snapshot_id", "seq", "validation", "candidates": [...]}``
    where each candidate is ``{kind, object_id, revision, created_seq,
    scope_id}`` sorted by creation sequence — the derived rows a §34.02
    rollback must re-evaluate. Objects *at or before* the watermark are
    never listed: they predate the snapshot. The candidate set is the
    honest input to rollback; applying current grants, retention, erasure,
    and risk dispositions to it is the caller's responsibility — and audit
    history/security epochs never roll backward.
    """
    require_id(snapshot_id, "snapshot_id")
    row = repos_v3.get(
        conn, "learning_snapshots", {"snapshot_id": snapshot_id}
    )
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "snapshot not found"
        )
    seq = int(row["seq"])
    candidates = _rows(
        conn.execute(
            "SELECT child_kind, child_id, child_revision, MIN(seq) AS created_seq,"
            " scope_id"
            " FROM derivations WHERE seq > ?"
            " GROUP BY child_kind, child_id, child_revision"
            " ORDER BY created_seq, child_kind, child_id",
            (seq,),
        )
    )
    return {
        "snapshot_id": snapshot_id,
        "seq": seq,
        "validation": row["validation"],
        "candidates": [
            {
                "kind": r["child_kind"],
                "object_id": r["child_id"],
                "revision": int(r["child_revision"]),
                "created_seq": int(r["created_seq"]),
                "scope_id": r["scope_id"],
            }
            for r in candidates
        ],
    }


__all__ = ["rollback_plan", "take_snapshot"]
