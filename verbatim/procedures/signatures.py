"""Procedural signatures (SPEC_V3 §21.04, §22): structure-aware dedup keys.

A procedure's signature is ``sha256(intent_key + ordered op classes)`` where
``intent_key`` is the canonical JSON of the host-declared goal class, the
check kind, and the binding kinds — matching §22's signature row. The
signature is what makes compilation idempotent: recompiling the same
episode lands on the same ``(scope_id, signature_digest)`` pair, so the
compiler bumps the existing procedure's revision instead of minting a
duplicate row.

``procedure_signatures`` rows are ``(procedure_id, revision)``-keyed so the
signature history of a procedure is preserved alongside the current row
(V3-21.04, V3-17.02).
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any, Optional

from ..core.types import ErrorCode, VerbatimError, json_dumps, require_id
from ..storage import repos_v3
from ..storage.repos import _json_parse, _row

#: Compiler manifest pinned on every candidate (V3-22.13): the operation
#: mapping revision, binding rules, checker allowlist, limits, and template
#: digest all live under this identity. ``:1`` is the rules revision.
COMPILER_MANIFEST = "coding_rules_v1:1"

#: Producer identity recorded on derivation edges.
PRODUCER_KIND = "compiler"
PRODUCER_ID = COMPILER_MANIFEST

#: Bounded-producer limits (§22 candidate-record row).
MAX_OPERATIONS = 128
MAX_BINDINGS = 32
#: Contrastive refinement compares at most this many same-signature
#: episodes per pass (§22 contrast row).
MAX_CONTRAST_EPISODES = 20


def intent_key(goal_class: str, check_kind: str,
               binding_kinds: list[str]) -> str:
    """Canonical intent key: goal class + check kind + binding types."""
    return json_dumps({
        "goal_class": goal_class or "unknown",
        "check_kind": check_kind or "unknown",
        "binding_kinds": sorted(set(binding_kinds)),
    })


def compute_signature(intent_key_value: str,
                      ordered_op_classes: list[str]) -> str:
    """``sha256(intent_key + ordered operation classes)`` — the dedup key."""
    return hashlib.sha256(
        json_dumps({
            "intent": intent_key_value,
            "ops": [str(c) for c in ordered_op_classes],
        }).encode("utf-8")
    ).hexdigest()


def signature_from_row(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    """Recompute ``(intent_key, digest, ordered_ops)`` from a procedures row."""
    intent = _json_parse(row.get("intent_signature_json")) or {}
    ops = _json_parse(row.get("operations_json")) or []
    ordered = [str(o.get("op_class")) for o in ops if isinstance(o, dict)]
    bindings = _json_parse(row.get("bindings_json")) or []
    kinds = [b.get("kind") for b in bindings if isinstance(b, dict)]
    key = intent_key(
        str(intent.get("goal_class") or "unknown"),
        str(intent.get("check_kind") or "unknown"),
        [str(k) for k in kinds if k],
    )
    return key, compute_signature(key, ordered), ordered


def find_by_signature(
    conn: sqlite3.Connection,
    scope_id: str,
    signature_digest: str,
) -> Optional[dict[str, Any]]:
    """Latest ``procedure_signatures`` row for (scope, digest), or None."""
    require_id(scope_id, "scope_id")
    require_id(signature_digest, "signature_digest")
    rows = repos_v3.query(
        conn, "procedure_signatures",
        {"scope_id": scope_id, "signature_digest": signature_digest},
        order="revision DESC", limit=1,
    )
    return rows[0] if rows else None


def upsert_signature(
    conn: sqlite3.Connection,
    procedure_id: str,
    revision: int,
    scope_id: str,
    intent_key_value: str,
    ordered_ops: list[str],
    signature_digest: str,
) -> None:
    """Idempotent write of one ``procedure_signatures`` row."""
    existing = repos_v3.get(
        conn, "procedure_signatures",
        {"procedure_id": procedure_id, "revision": revision},
    )
    row = {
        "procedure_id": procedure_id,
        "revision": int(revision),
        "signature_digest": signature_digest,
        "ordered_ops_json": list(ordered_ops),
        "intent_key": intent_key_value,
        "scope_id": scope_id,
    }
    if existing is None:
        repos_v3.insert(conn, "procedure_signatures", row)
        return
    if existing["signature_digest"] != signature_digest:
        raise VerbatimError(
            ErrorCode.INTEGRITY,
            "signature rows are immutable: digest changed for "
            f"{procedure_id} rev {revision}",
        )


def index_procedure(conn: sqlite3.Connection, procedure_id: str) -> int:
    """(Re)build the signature row for one procedure's current revision.

    Returns 1 when a row was written/verified, 0 when the procedure is
    absent. The ``signature_index`` job target: deterministic and
    idempotent under redelivery (V3-20.08).
    """
    require_id(procedure_id, "procedure_id")
    rec = _row(
        conn.execute(
            "SELECT * FROM procedures WHERE procedure_id = ?", (procedure_id,)
        )
    )
    if rec is None:
        return 0
    key, digest, ordered = signature_from_row(rec)
    upsert_signature(
        conn, procedure_id, int(rec["revision"]), rec["scope_id"],
        key, ordered, digest,
    )
    return 1


def index_scope(conn: sqlite3.Connection, scope_id: str) -> int:
    """Index every procedure in a scope; returns rows written/verified."""
    require_id(scope_id, "scope_id")
    cur = conn.execute(
        "SELECT procedure_id FROM procedures WHERE scope_id = ?"
        " ORDER BY procedure_id",
        (scope_id,),
    )
    n = 0
    for (pid,) in cur.fetchall():
        n += index_procedure(conn, pid)
    return n
