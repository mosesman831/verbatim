"""Purge orchestration: preview → suppression → physical erasure (SPEC_V2 §41).

Two distinct paths share one tombstone registry (``purges`` /
``purge_targets``):

* **Suppression** (reversible soft forget, V2-41.03): tombstones activate
  immediately — retrieval, history replay, exports, and egress all observe
  them — but no bytes are deleted and no erasure ledger rows are written.
  ``lift_suppression`` restores availability exactly.
* **Purge** (irreversible, V2-41.03): suppression commits first, then the
  same transaction scrubs content bytes, derived interpretations, and
  projection rows, and writes opaque erasure-ledger digests so a restored
  backup can never resurrect the objects (V2-41.14, V2-40.14).

Erasure granularity follows the safe default (V2-41.05): purging a span
empties its whole parent source revision, because deleting subspan bytes
would leave the same sensitive text elsewhere in that revision. Claims
whose evidence is purged are collateral dependents — the preview lists
them and execution erases them through the lifecycle machine.

All mutators take the caller's transaction ``conn`` (or open their own
``store.tx()`` when ``conn`` is None) — effects, tombstones, and receipts
commit atomically; no helper nests a second write transaction.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable, Optional, Union

from .core.lifecycle import PURGE_ACTOR, LifecycleMachine, read_claim_head
from .core.types import (
    ErrorCode,
    Lifecycle,
    Scope,
    TransitionCommand,
    VerbatimError,
    json_dumps,
    require_id,
    safe_json_loads,
)
from .storage.repos import (
    EventsRepo,
    PurgesRepo,
    ensure_scope,
    has_table as _has_table,
)
from .storage.repos_v2 import ErasureRepo

#: Object kinds purge/suppression understands (unknown kinds → VALIDATION).
#: ``source_revision`` is the revision-scoped form of ``source``: object_id
#: is ``"<source_id>:<revision>"`` so ``purge_targets`` (which has no
#: revision column) still names the exact erased bytes (V2-41.04).
OBJECT_KINDS = (
    "source",
    "source_revision",
    "span",
    "claim",
    "artifact",
    "episode",
    "procedure",
)

#: Purge row states in which targets are tombstoned (mirrors PurgesRepo).
_SUPPRESSING_STATES = ("suppressed", "purging", "completed")

_PURGE_POLICY_VERSION = "policy-1"

#: Bound on the dependency-closure expansion so a pathological graph cannot
#: turn a preview into an unbounded scan (SPEC §41 fixed allowlists/bounds).
_MAX_CLOSURE = 50_000


# ----------------------------------------------------------------------
# small conn-bound helpers (parameterized; fixed allowlists only)
# ----------------------------------------------------------------------


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _row(cur: sqlite3.Cursor) -> Optional[dict[str, Any]]:
    rs = _rows(cur)
    return rs[0] if rs else None


def _scope_id(scope: Union[Scope, str]) -> str:
    if isinstance(scope, Scope):
        from .core.identity import scope_key

        return scope_key(scope)
    return require_id(scope, "scope_id")


def _normalize_targets(targets: Iterable[Any]) -> list[tuple[str, str, Optional[int]]]:
    """Targets → deduped ``(object_kind, object_id, revision|None)`` triples.

    Accepts ``(kind, id)`` pairs, ``(kind, id, revision)`` triples, or dicts
    with ``object_kind``/``object_id``/optional ``revision`` keys.
    """
    out: list[tuple[str, str, Optional[int]]] = []
    seen: set[tuple[str, str]] = set()
    for item in targets or ():
        kind: Any
        oid: Any
        rev: Any = None
        if isinstance(item, dict):
            kind = item.get("object_kind") or item.get("kind")
            oid = item.get("object_id") or item.get("id")
            rev = item.get("revision")
        else:
            try:
                parts = tuple(item)
            except TypeError:
                parts = ()
            if len(parts) == 2:
                kind, oid = parts
            elif len(parts) == 3:
                kind, oid, rev = parts
            else:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "targets must be (object_kind, object_id[, revision])",
                )
        if kind not in OBJECT_KINDS:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"unknown purge object_kind {kind!r}"
            )
        require_id(str(oid), "object_id")
        if rev is not None:
            if isinstance(rev, bool) or not isinstance(rev, int) or rev < 1:
                raise VerbatimError(
                    ErrorCode.VALIDATION, "target revision must be an int >= 1"
                )
            if kind not in ("source", "source_revision", "span", "claim"):
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    f"revision-scoped targeting is not supported for {kind!r}",
                )
        if kind == "source" and rev is not None:
            # Revision-scoped source erasure gets its own object kind so the
            # tombstone/ledger names the exact bytes.
            kind, oid, rev = "source_revision", f"{oid}:{rev}", None
        elif kind == "source_revision" and rev is not None and ":" not in str(oid):
            oid, rev = f"{oid}:{rev}", None
        key = (str(kind), str(oid))
        if key not in seen:
            seen.add(key)
            out.append((str(kind), str(oid), rev))
    if not out:
        raise VerbatimError(ErrorCode.VALIDATION, "purge needs at least one target")
    return out


def _object_scope(
    conn: sqlite3.Connection, kind: str, object_id: str
) -> Optional[str]:
    """Owning scope_id for a target, or None when it does not exist."""
    if kind == "claim":
        r = conn.execute(
            "SELECT scope_id FROM claims WHERE claim_id = ?", (object_id,)
        ).fetchone()
        return r[0] if r else None
    if kind == "source":
        r = conn.execute(
            "SELECT scope_id FROM sources WHERE source_id = ?", (object_id,)
        ).fetchone()
        return r[0] if r else None
    if kind == "source_revision":
        source_id, _, rev_text = object_id.rpartition(":")
        if not source_id or not rev_text.isdigit():
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "source_revision object_id must be '<source_id>:<revision>'",
            )
        r = conn.execute(
            "SELECT so.scope_id FROM sources so"
            " JOIN source_revisions sr ON sr.source_id = so.source_id"
            " WHERE so.source_id = ? AND sr.revision = ?",
            (source_id, int(rev_text)),
        ).fetchone()
        return r[0] if r else None
    if kind == "span":
        r = conn.execute(
            "SELECT so.scope_id FROM spans sp"
            " JOIN sources so ON so.source_id = sp.source_id"
            " WHERE sp.span_id = ?",
            (object_id,),
        ).fetchone()
        return r[0] if r else None
    if kind == "artifact" and _has_table(conn, "artifacts"):
        r = conn.execute(
            "SELECT scope_id FROM artifacts WHERE artifact_id = ?", (object_id,)
        ).fetchone()
        return r[0] if r else None
    if kind == "episode" and _has_table(conn, "episodes"):
        r = conn.execute(
            "SELECT scope_id FROM episodes WHERE episode_id = ?", (object_id,)
        ).fetchone()
        return r[0] if r else None
    if kind == "procedure" and _has_table(conn, "procedures"):
        r = conn.execute(
            "SELECT scope_id FROM procedures WHERE procedure_id = ?", (object_id,)
        ).fetchone()
        return r[0] if r else None
    return None


def _expand_closure(
    conn: sqlite3.Connection, targets: list[tuple[str, str, Optional[int]]]
) -> list[tuple[str, str]]:
    """Reverse-dependency closure over requested targets (V2-41.04/07).

    source → every span on every revision → every claim citing those spans.
    span → claims citing it. Claims/artifacts/episodes/procedures are leaves:
    erasing an interpretation does not erase its underlying evidence bytes.
    """
    seen: set[tuple[str, str]] = {(k, i) for k, i, _r in targets}
    ordered: list[tuple[str, str]] = [(k, i) for k, i, _r in targets]

    def add(kind: str, oid: str) -> None:
        if len(seen) >= _MAX_CLOSURE:
            raise VerbatimError(
                ErrorCode.VALIDATION, "purge dependency closure exceeds bound"
            )
        if (kind, oid) not in seen:
            seen.add((kind, oid))
            ordered.append((kind, oid))

    def claims_citing(span_id: str) -> None:
        for (cid,) in conn.execute(
            "SELECT DISTINCT claim_id FROM claim_evidence WHERE span_id = ?",
            (span_id,),
        ).fetchall():
            add("claim", cid)

    for kind, oid, _rev in targets:
        if kind == "source":
            span_rows = conn.execute(
                "SELECT span_id FROM spans WHERE source_id = ?", (oid,)
            ).fetchall()
            for (sid_,) in span_rows:
                add("span", sid_)
                claims_citing(sid_)
        elif kind == "source_revision":
            source_id, _, rev_text = oid.rpartition(":")
            span_rows = conn.execute(
                "SELECT span_id FROM spans WHERE source_id = ? AND revision = ?",
                (source_id, int(rev_text)),
            ).fetchall()
            for (sid_,) in span_rows:
                add("span", sid_)
                claims_citing(sid_)
        elif kind == "span":
            claims_citing(oid)
    return ordered


def _selection_digest(store: Any, scope_id: str, targets: list[tuple[str, str]]) -> bytes:
    canonical = json_dumps(
        {"scope_id": scope_id, "targets": sorted([k, i] for k, i in targets)}
    )
    return store.hmac(canonical.encode("utf-8"))


def _purge_row(conn: sqlite3.Connection, purge_id: str) -> Optional[dict[str, Any]]:
    return _row(
        conn.execute(
            "SELECT purge_id, hex(selection_digest) AS selection_digest_hex,"
            " scope_id, state, requested_us, approved_us, completed_us"
            " FROM purges WHERE purge_id = ?",
            (purge_id,),
        )
    )


def _purge_targets(conn: sqlite3.Connection, purge_id: str) -> list[tuple[str, str]]:
    return [
        (r["object_kind"], r["object_id"])
        for r in _rows(
            conn.execute(
                "SELECT object_kind, object_id FROM purge_targets"
                " WHERE purge_id = ? ORDER BY object_kind, object_id",
                (purge_id,),
            )
        )
    ]


# ----------------------------------------------------------------------
# planning
# ----------------------------------------------------------------------


def plan_purge(
    store: Any,
    scope: Union[Scope, str],
    targets: Iterable[Any],
    actor: str,
    *,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    """Create a ``previewed`` purge: exact selection + dependency closure.

    The preview identifies every object the erasure will touch — requested
    targets plus collateral dependents (spans of a source revision, claims
    citing purged spans) — and binds the selection into ``selection_digest``
    so approval cannot be silently redirected (V2-41.04). Nothing is
    suppressed or deleted at this stage.
    """
    sid = _scope_id(scope)
    require_id(actor, "actor")
    norm = _normalize_targets(targets)
    if conn is None:
        with store.tx() as owned:
            return _plan(store, owned, scope, sid, norm, actor)
    return _plan(store, conn, scope, sid, norm, actor)


def _plan(
    store: Any,
    conn: sqlite3.Connection,
    scope: Union[Scope, str],
    sid: str,
    norm: list[tuple[str, str, Optional[int]]],
    actor: str,
) -> dict[str, Any]:
    if isinstance(scope, Scope):
        ensure_scope(store, conn, scope)
    for kind, oid, rev in norm:
        owner = _object_scope(conn, kind, oid)
        if owner is None or owner != sid:
            # Missing and foreign objects share one response (V2-09.14).
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                f"{kind} {oid!r} not found in scope",
            )
        if rev is not None:
            _require_revision_exists(conn, kind, oid, rev)
    expanded = _expand_closure(conn, norm)
    digest = _selection_digest(store, sid, expanded)
    purge_id = PurgesRepo(store).create_preview(conn, sid, expanded, digest)
    requested = {(k, i) for k, i, _r in norm}
    collateral = [t for t in expanded if t not in requested]
    EventsRepo(store).append(
        conn,
        sid,
        "purge_previewed",
        actor,
        {
            "purge_id": purge_id,
            "requested": len(requested),
            "collateral": len(collateral),
        },
        _PURGE_POLICY_VERSION,
    )
    return {
        "purge_id": purge_id,
        "scope_id": sid,
        "state": "previewed",
        "selection_digest": digest.hex(),
        "requested": sorted(requested),
        "targets": expanded,
        "collateral": collateral,
    }


def _require_revision_exists(
    conn: sqlite3.Connection, kind: str, oid: str, rev: int
) -> None:
    if kind == "source":
        ok = conn.execute(
            "SELECT 1 FROM source_revisions WHERE source_id = ? AND revision = ?",
            (oid, rev),
        ).fetchone()
    elif kind == "claim":
        ok = conn.execute(
            "SELECT 1 FROM claim_revisions WHERE claim_id = ? AND revision = ?",
            (oid, rev),
        ).fetchone()
    else:
        ok = conn.execute(
            "SELECT 1 FROM spans WHERE span_id = ? AND revision = ?",
            (oid, rev),
        ).fetchone()
    if ok is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_FORBIDDEN,
            f"{kind} {oid!r} revision {rev} not found",
        )


# ----------------------------------------------------------------------
# suppression — reversible soft forget (V2-41.03)
# ----------------------------------------------------------------------


def suppress(
    store: Any,
    scope: Union[Scope, str],
    targets: Iterable[Any],
    actor: str,
    *,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    """Immediately tombstone objects without deleting any bytes.

    The ``suppressed`` purge row plus ``recorded_until``/``availability``
    tombstones take effect atomically: recall, inspection, exports, and
    egress all observe the suppression set in the same transaction
    (V2-41.06). No payload is touched and no erasure ledger row is written —
    ``lift_suppression`` restores the objects exactly.
    """
    sid = _scope_id(scope)
    require_id(actor, "actor")
    norm = _normalize_targets(targets)
    if conn is None:
        with store.tx() as owned:
            return _suppress(store, owned, scope, sid, norm, actor)
    return _suppress(store, conn, scope, sid, norm, actor)


def _suppress(
    store: Any,
    conn: sqlite3.Connection,
    scope: Union[Scope, str],
    sid: str,
    norm: list[tuple[str, str, Optional[int]]],
    actor: str,
) -> dict[str, Any]:
    if isinstance(scope, Scope):
        ensure_scope(store, conn, scope)
    for kind, oid, rev in norm:
        owner = _object_scope(conn, kind, oid)
        if owner is None or owner != sid:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                f"{kind} {oid!r} not found in scope",
            )
        if rev is not None:
            _require_revision_exists(conn, kind, oid, rev)
    expanded = _expand_closure(conn, norm)
    purges = PurgesRepo(store)
    digest = _selection_digest(store, sid, expanded)
    purge_id = purges.create_preview(conn, sid, expanded, digest)
    purges.confirm_suppress(conn, purge_id)
    seq = _apply_tombstones(conn, expanded)
    EventsRepo(store).append(
        conn,
        sid,
        "objects_suppressed",
        actor,
        {
            "purge_id": purge_id,
            "suppress_seq": seq,
            "targets": [list(t) for t in expanded],
        },
        _PURGE_POLICY_VERSION,
    )
    return {
        "purge_id": purge_id,
        "scope_id": sid,
        "state": "suppressed",
        "suppress_seq": seq,
        "targets": expanded,
        "reversible": True,
    }


def _apply_tombstones(
    conn: sqlite3.Connection, targets: list[tuple[str, str]]
) -> int:
    """Close recorded-time/availability on suppressible objects; return seq.

    ``S`` is the event sequence the suppression event will land on — the
    same value written into ``recorded_until`` so lift can restore exactly
    the rows this suppression closed (a later unrelated transition would
    have overwritten ``recorded_until`` and correctly restores nothing).
    """
    seq_row = conn.execute(
        "SELECT COALESCE(MAX(event_seq), 0) + 1 FROM events"
    ).fetchone()
    seq = int(seq_row[0])
    claim_ids = [i for k, i in targets if k == "claim"]
    for cid in claim_ids:
        head = read_claim_head(conn, cid)
        if head is not None and head.recorded_until is None:
            conn.execute(
                "UPDATE claim_revisions SET recorded_until = ?"
                " WHERE claim_id = ? AND revision = ?",
                (seq, cid, head.revision),
            )
    for table, col in (
        ("episodes", "episode_id"),
        ("procedures", "procedure_id"),
        ("context_groups", "group_id"),
    ):
        if not _has_table(conn, table):
            continue
        kind = "episode" if table == "episodes" else (
            "procedure" if table == "procedures" else "context_group"
        )
        ids = [i for k, i in targets if k == kind]
        for oid in ids:
            conn.execute(
                f"UPDATE {table} SET recorded_until = ?"
                f" WHERE {col} = ? AND recorded_until IS NULL",
                (seq, oid),
            )
    if _has_table(conn, "artifacts"):
        for aid in (i for k, i in targets if k == "artifact"):
            conn.execute(
                "UPDATE artifacts SET availability = 'suppressed'"
                " WHERE artifact_id = ? AND availability = 'available'",
                (aid,),
            )
    return seq


def lift_suppression(
    store: Any,
    purge_id: str,
    actor: str,
    *,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    """Reverse a suppression: tombstones off, purge record removed.

    Only legal while the purge never reached physical erasure — ``previewed``
    (cancels the plan) or ``suppressed``. ``purging``/``completed`` purges are
    irreversible by definition (V2-41.03) and fail with INVALID_TRANSITION.
    """
    require_id(purge_id, "purge_id")
    require_id(actor, "actor")
    if conn is None:
        with store.tx() as owned:
            return _lift(store, owned, purge_id, actor)
    return _lift(store, conn, purge_id, actor)


def _lift(
    store: Any, conn: sqlite3.Connection, purge_id: str, actor: str
) -> dict[str, Any]:
    row = _purge_row(conn, purge_id)
    if row is None:
        raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "purge not found")
    state = row["state"]
    if state not in ("previewed", "suppressed"):
        raise VerbatimError(
            ErrorCode.INVALID_TRANSITION,
            f"purge in state {state!r} is irreversible",
        )
    sid = row["scope_id"]
    restored = 0
    if state == "suppressed":
        seq = _suppress_seq(conn, sid, purge_id)
        targets = _purge_targets(conn, purge_id)
        restored = _restore_tombstones(conn, targets, seq)
    conn.execute("DELETE FROM purge_targets WHERE purge_id = ?", (purge_id,))
    conn.execute("DELETE FROM purges WHERE purge_id = ?", (purge_id,))
    EventsRepo(store).append(
        conn,
        sid,
        "suppression_lifted",
        actor,
        {"purge_id": purge_id, "restored": restored},
        _PURGE_POLICY_VERSION,
    )
    return {
        "purge_id": purge_id,
        "scope_id": sid,
        "lifted": True,
        "restored": restored,
    }


def _suppress_seq(
    conn: sqlite3.Connection, scope_id: str, purge_id: str
) -> Optional[int]:
    """Recover the recorded_until watermark this suppression wrote."""
    rows = conn.execute(
        "SELECT payload_json FROM events WHERE scope_id = ?"
        " AND kind = 'objects_suppressed' ORDER BY event_seq",
        (scope_id,),
    ).fetchall()
    for (text,) in rows:
        payload = safe_json_loads(text)
        if isinstance(payload, dict) and payload.get("purge_id") == purge_id:
            seq = payload.get("suppress_seq")
            return int(seq) if isinstance(seq, int) else None
    return None


def _restore_tombstones(
    conn: sqlite3.Connection,
    targets: list[tuple[str, str]],
    seq: Optional[int],
) -> int:
    """Undo exactly the tombstones this suppression wrote (seq-matched)."""
    if seq is None:
        return 0
    restored = 0
    for kind, oid in targets:
        if kind == "claim":
            cur = conn.execute(
                "UPDATE claim_revisions SET recorded_until = NULL"
                " WHERE claim_id = ? AND recorded_until = ?",
                (oid, seq),
            )
            restored += cur.rowcount
        elif kind in ("episode", "procedure", "context_group"):
            table = {
                "episode": "episodes",
                "procedure": "procedures",
                "context_group": "context_groups",
            }[kind]
            col = {
                "episode": "episode_id",
                "procedure": "procedure_id",
                "context_group": "group_id",
            }[kind]
            if _has_table(conn, table):
                cur = conn.execute(
                    f"UPDATE {table} SET recorded_until = NULL"
                    f" WHERE {col} = ? AND recorded_until = ?",
                    (oid, seq),
                )
                restored += cur.rowcount
        elif kind == "artifact" and _has_table(conn, "artifacts"):
            cur = conn.execute(
                "UPDATE artifacts SET availability = 'available'"
                " WHERE artifact_id = ? AND availability = 'suppressed'",
                (oid,),
            )
            restored += cur.rowcount
    return restored


# ----------------------------------------------------------------------
# physical purge — irreversible (V2-41.03, §41.06–14)
# ----------------------------------------------------------------------


def execute_purge(
    store: Any,
    purge_id: str,
    *,
    actor: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    """Run physical erasure for a previewed/suppressed purge.

    One transaction: suppression commits (previewed → suppressed), then
    claims are erased through the lifecycle machine, source revision bytes
    are emptied, derived rows and projections are scrubbed, pending jobs
    referencing the objects are cancelled (V2-41.08), opaque erasure
    digests are recorded for restore fencing, the projection generation and
    erasure epoch advance, and the purge completes. A rollback leaves the
    preview untouched — never a half-erased object.
    """
    require_id(purge_id, "purge_id")
    if conn is None:
        with store.tx() as owned:
            return _execute(store, owned, purge_id, actor)
    return _execute(store, conn, purge_id, actor)


def _execute(
    store: Any, conn: sqlite3.Connection, purge_id: str, actor: Optional[str]
) -> dict[str, Any]:
    purges = PurgesRepo(store)
    row = _purge_row(conn, purge_id)
    if row is None:
        raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "purge not found")
    if row["state"] == "previewed":
        purges.confirm_suppress(conn, purge_id)  # tombstones commit first
    elif row["state"] != "suppressed":
        raise VerbatimError(
            ErrorCode.STALE_PROPOSAL,
            f"purge in state {row['state']!r} cannot execute",
        )
    sid = row["scope_id"]
    targets = _purge_targets(conn, purge_id)
    conn.execute("UPDATE purges SET state = 'purging' WHERE purge_id = ?", (purge_id,))

    epoch = _bump_erasure_epoch(store, conn)
    machine = LifecycleMachine(store)
    erasure = ErasureRepo(store) if _has_table(conn, "erasure_ledger") else None
    scrubbed = {"payloads": 0, "claims": 0, "ledger": 0, "projections": 0, "jobs": 0}

    emptied_revisions: set[tuple[str, int]] = set()
    ledger_done: set[tuple[str, str]] = set()

    # Imported objects carry their origin id in the operation receipt; purge
    # fences BOTH so a different bundle carrying the origin id cannot
    # resurrect the erased content (V2-41.15).
    origin_of: dict[tuple[str, str], str] = {}
    if erasure is not None and _has_table(conn, "operations"):
        for (receipt_text,) in conn.execute(
            "SELECT receipt_json FROM operations WHERE scope_id = ?"
            " AND effect_kind = 'import_object'",
            (sid,),
        ).fetchall():
            rec = safe_json_loads(receipt_text)
            if (
                isinstance(rec, dict)
                and isinstance(rec.get("new_id"), str)
                and isinstance(rec.get("origin_id"), str)
                and isinstance(rec.get("kind"), str)
            ):
                origin_of[(rec["kind"], rec["new_id"])] = rec["origin_id"]

    def _ledger(kind_: str, oid_: str) -> None:
        if erasure is None or (kind_, oid_) in ledger_done:
            return
        ledger_done.add((kind_, oid_))
        erasure.record(
            conn, sid, kind_, oid_, purge_id=purge_id, erasure_epoch=epoch
        )
        scrubbed["ledger"] += 1
        origin_id = origin_of.get((kind_, oid_))
        if origin_id is not None and (kind_, origin_id) not in ledger_done:
            ledger_done.add((kind_, origin_id))
            erasure.record(
                conn, sid, kind_, origin_id,
                purge_id=purge_id, erasure_epoch=epoch,
            )
            scrubbed["ledger"] += 1

    derived_stats: dict[str, int] = {}
    for kind, oid in targets:
        if kind == "claim":
            _erase_claim(conn, store, machine, oid, purge_id)
            scrubbed["claims"] += 1
        elif kind == "span":
            emptied_revisions.update(_erase_span(conn, store, oid, derived_stats))
        elif kind == "source":
            emptied_revisions.update(_erase_source(conn, store, oid, derived_stats))
        elif kind == "source_revision":
            source_id, _, rev_text = oid.rpartition(":")
            emptied_revisions.update(
                _erase_source_revision(
                    conn, store, source_id, int(rev_text), derived_stats
                )
            )
        elif kind == "artifact":
            _erase_artifact(conn, oid)
        elif kind in ("episode", "procedure"):
            _erase_dated(conn, kind, oid)
        _ledger(kind, oid)
    scrubbed["payloads"] = len(emptied_revisions)
    for src, rev in sorted(emptied_revisions):
        _ledger("source_revision", f"{src}:{rev}")
    scrubbed["jobs"] = _cancel_jobs(conn, sid, targets)
    seq = _next_seq(conn)
    derived_counts, unhandled = _scrub_derived_content(
        conn, store, sid, purge_id, targets, emptied_revisions, seq,
        derived_stats,
    )
    generation = _bump_generation(store, conn)
    purges.complete(conn, purge_id)
    seq = EventsRepo(store).append(
        conn,
        sid,
        "purged",
        actor or PURGE_ACTOR,
        {
            "purge_id": purge_id,
            "targets": len(targets),
            "payloads": scrubbed["payloads"],
            "derived": derived_counts,
            "unhandled": unhandled,
            "erasure_epoch": epoch,
        },
        _PURGE_POLICY_VERSION,
    )
    return {
        "purge_id": purge_id,
        "scope_id": sid,
        "state": "completed",
        "erased": targets,
        "erasure_epoch": epoch,
        "projection_generation": generation,
        "event_seq": seq,
        "derived": derived_counts,
        "unhandled": unhandled,
        **scrubbed,
    }


def _erase_claim(
    conn: sqlite3.Connection, store: Any, machine: LifecycleMachine, claim_id: str, purge_id: str
) -> None:
    """Drive the claim to ``erased`` and redact derived interpretation text.

    The lifecycle machine appends the erased revision (NULL object) through
    the purge-only actor; the extra scrub clears content-bearing fields on
    *every* revision — append-only audit yields to authorized erasure
    (V2-41.10).
    """
    head = read_claim_head(conn, claim_id)
    if head is None:
        return
    if head.state != Lifecycle.ERASED:
        machine.apply(
            TransitionCommand(
                claim_id=claim_id,
                expected_revision=head.revision,
                effect="erase",
                actor_id=PURGE_ACTOR,
                reason=f"purge {purge_id}",
            ),
            conn,
        )
    # Security labels on any revision carry ``findings_json`` excerpts of
    # the screened content — the label goes with the claim that cites it
    # (labels bind to objects only through this column, a v3 addition).
    rev_cols = {
        r[1] for r in conn.execute("PRAGMA table_info(claim_revisions)")
    }
    if "security_label_id" in rev_cols and _has_table(conn, "security_labels"):
        label_ids = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT security_label_id FROM claim_revisions"
                " WHERE claim_id = ? AND security_label_id IS NOT NULL",
                (claim_id,),
            ).fetchall()
        ]
        for lid in label_ids:
            conn.execute(
                "DELETE FROM security_labels WHERE label_id = ?", (lid,)
            )
        conn.execute(
            "UPDATE claim_revisions SET security_label_id = NULL"
            " WHERE claim_id = ?",
            (claim_id,),
        )
    conn.execute(
        "UPDATE claim_revisions SET object_json = NULL, condition_json = NULL,"
        " interpretation_json = NULL WHERE claim_id = ?",
        (claim_id,),
    )
    conn.execute("DELETE FROM valid_intervals WHERE claim_id = ?", (claim_id,))
    # Dependent rows carrying the claim's interpretations/links (V2-41.07,
    # V2-38.03): feedback notes, entity links, conflict membership, decision
    # inputs, and FTS projection rows are all subject to the same erasure.
    conn.execute("DELETE FROM feedback WHERE claim_id = ?", (claim_id,))
    conn.execute("DELETE FROM claim_entities WHERE claim_id = ?", (claim_id,))
    if _has_table(conn, "conflict_members"):
        conn.execute(
            "DELETE FROM conflict_members WHERE claim_id = ?", (claim_id,)
        )
    conn.execute(
        "DELETE FROM decision_inputs WHERE object_kind = 'claim' AND object_id = ?",
        (claim_id,),
    )
    seq = _next_seq(conn)
    conn.execute(
        "UPDATE edges SET retired_event = ? WHERE retired_event IS NULL"
        " AND ((source_kind = 'claim' AND source_id = ?)"
        "  OR (target_kind = 'claim' AND target_id = ?))",
        (seq, claim_id, claim_id),
    )
    _scrub_projection(conn, "claim", claim_id)
    _scrub_member_refs(conn, "claim", claim_id)


def _erase_span(
    conn: sqlite3.Connection,
    store: Any,
    span_id: str,
    stats: Optional[dict[str, int]] = None,
) -> set[tuple[str, int]]:
    """Mark a span unavailable: empty its whole parent revision's bytes.

    Whole-revision granularity is the safe default — subspan deletion would
    leave the same sensitive bytes elsewhere in the revision (V2-41.05).
    The span row stays as an opaque skeleton so claim_evidence lineage keeps
    resolving; text can never be reconstructed again.
    """
    row = conn.execute(
        "SELECT source_id, revision FROM spans WHERE span_id = ?", (span_id,)
    ).fetchone()
    if row is None:
        return set()
    emptied = _empty_revision(conn, store, row[0], row[1], stats)
    for table, col, extra in (
        ("embeddings", "span_id", ""),
        ("embedding_inputs", "span_id", ""),
        ("entity_aliases", "source_span_id", ""),
        ("context_members", "span_id", ""),
        ("decision_inputs", "object_id", " AND object_kind = 'span'"),
    ):
        if _has_table(conn, table):
            conn.execute(
                f"DELETE FROM {table} WHERE {col} = ?{extra}", (span_id,)
            )
    if _has_table(conn, "procedure_steps"):
        conn.execute(
            "UPDATE procedure_steps SET span_id = NULL WHERE span_id = ?",
            (span_id,),
        )
    _scrub_member_refs(conn, "span", span_id)
    return emptied


def _erase_source_revision(
    conn: sqlite3.Connection,
    store: Any,
    source_id: str,
    revision: int,
    stats: Optional[dict[str, int]] = None,
) -> set[tuple[str, int]]:
    """Empty exactly one revision's bytes; revision-scoped erasure."""
    emptied = _empty_revision(conn, store, source_id, revision, stats)
    if _has_table(conn, "context_groups"):
        groups = conn.execute(
            "SELECT group_id FROM context_groups"
            " WHERE source_id = ? AND revision = ?",
            (source_id, revision),
        ).fetchall()
        for (gid,) in groups:
            conn.execute("DELETE FROM context_members WHERE group_id = ?", (gid,))
            conn.execute("DELETE FROM context_groups WHERE group_id = ?", (gid,))
    _scrub_member_refs(conn, "source_revision", f"{source_id}:{revision}")
    return emptied


def _erase_source(
    conn: sqlite3.Connection,
    store: Any,
    source_id: str,
    stats: Optional[dict[str, int]] = None,
) -> set[tuple[str, int]]:
    """Suppress every revision payload of a source; keep identity skeleton.

    The ``sources`` row's own submitter-controlled fields go too —
    ``external_id`` and ``speaker_id`` name the captured thing, so a
    whole-source purge scrubs them. ``origin``/``source_kind`` stay: they
    are the structural skeleton the purge registry and ledger reference.
    """
    emptied: set[tuple[str, int]] = set()
    revs = conn.execute(
        "SELECT revision FROM source_revisions WHERE source_id = ?", (source_id,)
    ).fetchall()
    for (rev,) in revs:
        emptied |= _empty_revision(conn, store, source_id, rev, stats)
    conn.execute(
        "UPDATE sources SET external_id = NULL, speaker_id = NULL"
        " WHERE source_id = ?",
        (source_id,),
    )
    if stats is not None:
        stats["source_rows"] = stats.get("source_rows", 0) + 1
    # Context groups harvested from this source carry derived ordering
    # metadata about it; the groups and their members go too (V2-41.07).
    if _has_table(conn, "context_groups"):
        groups = conn.execute(
            "SELECT group_id FROM context_groups WHERE source_id = ?", (source_id,)
        ).fetchall()
        for (gid,) in groups:
            conn.execute("DELETE FROM context_members WHERE group_id = ?", (gid,))
            conn.execute("DELETE FROM context_groups WHERE group_id = ?", (gid,))
    _scrub_member_refs(conn, "source", source_id)
    return emptied


def _empty_revision(
    conn: sqlite3.Connection,
    store: Any,
    source_id: str,
    revision: int,
    stats: Optional[dict[str, int]] = None,
) -> set[tuple[str, int]]:
    """Empty one revision's bytes (NOT NULL column → zero-length blob).

    Erasure covers the revision's *metadata* as well as its payload:
    ``metadata_json`` is submitter-controlled content and the covering
    ``source_envelopes`` rows carry caller-supplied capture fields
    (``metadata_json``, ``capture_proof``, ``artifact_ref``,
    ``actor_principal``, perspective/session/task correlation) — all of it
    is content of the purged capture, so all of it goes. The envelope row
    itself stays as an audit skeleton (kind, trust, timestamps), the same
    shape the span skeleton keeps for lineage.
    """
    empty_hmac = store.hmac(b"")
    conn.execute(
        "UPDATE source_revisions SET payload = X'', payload_hmac = ?,"
        " metadata_json = '{}'"
        " WHERE source_id = ? AND revision = ?",
        (empty_hmac, source_id, revision),
    )
    if stats is not None:
        stats["source_metadata"] = stats.get("source_metadata", 0) + 1
    if _has_table(conn, "source_views"):
        conn.execute(
            "UPDATE source_views SET derived_bytes = NULL,"
            " integrity_digest = NULL"
            " WHERE source_id = ? AND revision = ?",
            (source_id, revision),
        )
    if _has_table(conn, "source_envelopes"):
        if _has_table(conn, "security_labels"):
            # The label id rides in envelope metadata; harvest it before the
            # wipe so the findings excerpts the label carries go too.
            for (meta,) in conn.execute(
                "SELECT metadata_json FROM source_envelopes"
                " WHERE source_id = ? AND revision = ?",
                (source_id, revision),
            ).fetchall():
                doc = safe_json_loads(meta)
                lid = doc.get("security_label_id") if isinstance(doc, dict) else None
                if isinstance(lid, str) and lid:
                    conn.execute(
                        "DELETE FROM security_labels WHERE label_id = ?", (lid,)
                    )
        cur = conn.execute(
            "UPDATE source_envelopes SET metadata_json = '{}',"
            " capture_proof = NULL, artifact_ref = NULL,"
            " actor_principal = NULL, perspective_id = NULL,"
            " adapter_version = '', host_id = '', session_id = '',"
            " task_id = '', step_id = ''"
            " WHERE source_id = ? AND revision = ?",
            (source_id, revision),
        )
        if stats is not None:
            stats["source_envelopes"] = (
                stats.get("source_envelopes", 0) + cur.rowcount
            )
    # V7 derived plane (SPEC_V7 §30 / V7-30.02): units projected from this
    # revision — and every artifact standing on them — die with its bytes,
    # at EVERY projection generation. The sweep rides this same tx, so the
    # unit plane can never outlive the evidence it projects (the D7-class
    # gap the schema audit flagged). Lazy import keeps privacy.closure_v7
    # → purge acyclic; no-op on pre-V7 stores.
    from .privacy import closure_v7 as _closure_v7

    v7 = _closure_v7.delete_source_v7(
        conn, source_id, revision=revision
    )
    if stats is not None:
        for table, n in v7["deleted"].items():
            stats[f"v7_{table}"] = stats.get(f"v7_{table}", 0) + n
        for table, n in v7["audit_retained"].items():
            stats[f"v7_audit_{table}"] = (
                stats.get(f"v7_audit_{table}", 0) + n
            )
    return {(source_id, revision)}


def _erase_artifact(conn: sqlite3.Connection, artifact_id: str) -> None:
    if _has_table(conn, "artifacts"):
        conn.execute(
            "UPDATE artifacts SET availability = 'purged'"
            " WHERE artifact_id = ?",
            (artifact_id,),
        )
    if _has_table(conn, "artifact_links"):
        conn.execute(
            "DELETE FROM artifact_links WHERE artifact_id = ?", (artifact_id,)
        )
    _scrub_member_refs(conn, "artifact", artifact_id)


def _table_columns(conn: sqlite3.Connection, table: str) -> set:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


#: Content-bearing columns a purged procedure gives up (v2 + v3 sets;
#: applied only to columns present in this schema). ``task_label`` is NOT
#: NULL so the row keeps the tombstone marker — the same "skeleton stays"
#: shape sources keep.
_PROCEDURE_CONTENT_COLS = (
    "environment_json",
    "condition_json",
    "intent_signature_json",
    "operations_json",
    "bindings_json",
    "hazards_json",
    "verification_json",
    "expected_outcome",
    "failure_modes_json",
    "applicability_json",
    "preconditions_json",
    "provenance_json",
    "reuse_stats_json",
    "compiler_manifest",
    "evidence_family_id",
)


def _scrub_procedure_content(
    conn: sqlite3.Connection, procedure_id: str
) -> None:
    """Empty a purged procedure's content fields; skeleton + tombstone stay."""
    cols = _table_columns(conn, "procedures")
    if "security_label_id" in cols and _has_table(conn, "security_labels"):
        for (lid,) in conn.execute(
            "SELECT DISTINCT security_label_id FROM procedures"
            " WHERE procedure_id = ? AND security_label_id IS NOT NULL",
            (procedure_id,),
        ).fetchall():
            conn.execute(
                "DELETE FROM security_labels WHERE label_id = ?", (lid,)
            )
        conn.execute(
            "UPDATE procedures SET security_label_id = NULL"
            " WHERE procedure_id = ?",
            (procedure_id,),
        )
    sets = ["task_label = '[purged]'"]
    sets += [f"{c} = NULL" for c in _PROCEDURE_CONTENT_COLS if c in cols]
    conn.execute(
        f"UPDATE procedures SET {', '.join(sets)} WHERE procedure_id = ?",
        (procedure_id,),
    )
    # Receipts on the procedure carry checker detail — they are the
    # procedure's own dependents and go with it.
    if _has_table(conn, "outcome_receipts"):
        conn.execute(
            "DELETE FROM outcome_receipts WHERE procedure_id = ?",
            (procedure_id,),
        )


def _erase_dated(conn: sqlite3.Connection, kind: str, oid: str) -> None:
    """Episodes/procedures: close recorded time, scrub content and member rows."""
    table = "episodes" if kind == "episode" else "procedures"
    col = "episode_id" if kind == "episode" else "procedure_id"
    if not _has_table(conn, table):
        return
    conn.execute(
        f"UPDATE {table} SET recorded_until = COALESCE(recorded_until, ?)"
        f" WHERE {col} = ?",
        (_next_seq(conn), oid),
    )
    if kind == "episode":
        # host_task_id/host_session_id/label are the episode's content —
        # a purged episode keeps only its identity skeleton + tombstone.
        ecols = _table_columns(conn, "episodes")
        escrub = [
            f"{c} = NULL"
            for c in ("label", "host_task_id", "host_session_id")
            if c in ecols
        ]
        if escrub:
            conn.execute(
                f"UPDATE episodes SET {', '.join(escrub)}"
                " WHERE episode_id = ?",
                (oid,),
            )
        if _has_table(conn, "episode_members"):
            conn.execute(
                "DELETE FROM episode_members WHERE episode_id = ?", (oid,)
            )
        # Transitions/anchors are the episode's own dependent rows (FK
        # mapped) — they go with it.
        if _has_table(conn, "transitions"):
            for (tid,) in conn.execute(
                "SELECT transition_id FROM transitions WHERE episode_id = ?",
                (oid,),
            ).fetchall():
                if _has_table(conn, "transition_anchors"):
                    conn.execute(
                        "DELETE FROM transition_anchors WHERE transition_id = ?",
                        (tid,),
                    )
            conn.execute(
                "DELETE FROM transitions WHERE episode_id = ?", (oid,)
            )
    if kind == "procedure":
        conn.execute("DELETE FROM procedure_steps WHERE procedure_id = ?", (oid,))
        for dep in ("procedure_signatures", "procedure_exposures"):
            if _has_table(conn, dep):
                conn.execute(
                    f"DELETE FROM {dep} WHERE procedure_id = ?", (oid,)
                )
        _scrub_procedure_content(conn, oid)
    _scrub_member_refs(conn, kind, oid)


def _scrub_projection(conn: sqlite3.Connection, kind: str, oid: str) -> None:
    """Drop derived index rows for the erased object (V2-41.04 vectors/FTS)."""
    if kind == "claim":
        conn.execute(
            "DELETE FROM facts_fts WHERE fts_row_id IN"
            " (SELECT row_id FROM fts_rows WHERE claim_id = ?)",
            (oid,),
        )
        conn.execute("DELETE FROM fts_rows WHERE claim_id = ?", (oid,))
    if _has_table(conn, "dependency_refs"):
        conn.execute(
            "DELETE FROM dependency_refs WHERE (derived_kind = ? AND derived_id = ?)"
            " OR (input_kind = ? AND input_id = ?)",
            (kind, oid, kind, oid),
        )


def _scrub_member_refs(conn: sqlite3.Connection, kind: str, oid: str) -> None:
    """Remove membership rows that would resurrect the object elsewhere."""
    for table in ("family_members", "episode_members", "capsule_members"):
        if _has_table(conn, table):
            conn.execute(
                f"DELETE FROM {table} WHERE object_kind = ? AND object_id = ?",
                (kind, oid),
            )


def _cancel_jobs(
    conn: sqlite3.Connection, scope_id: str, targets: list[tuple[str, str]]
) -> int:
    """Cancel pending/leased jobs referencing purged objects (V2-41.08).

    Cancelling a leased row flips it out of ``leased`` so the worker's
    generation fence fails at commit — an in-flight job cannot recreate
    erased derivatives. Erasure-lane jobs (``purge``/``purge_derived``/
    ``purge_vault``) are exempt — they *are* the erasure; without the
    exemption a closure run cancels the very job driving it (F4-07).
    """
    cancelled = 0
    for _kind, oid in targets:
        cur = conn.execute(
            "UPDATE jobs SET state = 'cancelled'"
            " WHERE state IN ('queued','retry_wait','leased')"
            "   AND kind NOT IN ('purge','purge_derived','purge_vault')"
            "   AND scope_id = ?"
            "   AND input_refs_json LIKE '%' || ? || '%'",
            (scope_id, oid),
        )
        cancelled += cur.rowcount
    return cancelled


# ----------------------------------------------------------------------
# derived-content closure (V3-17.03, V3-36.02)
#
# After the evidence plane is emptied, content derived *from* purged
# objects must not remain recallable. Two mapped channels handle it:
#
# 1. The ``derivations`` graph — ``privacy.closure.ClosureEngine`` walks
#    descendants of the purged roots; derived objects are deleted (all
#    parents purged), suppressed (mixed ancestry), or flagged for
#    revalidation, per the same action semantics the v3 closure executor
#    applies. Edges and member-table references to purged objects are
#    stripped, and the closure check runs inline: a surviving orphan rolls
#    the whole purge back — never a half-erased derived plane.
# 2. Reference-mapped columns the graph does not cover (``working_set_
#    items.object_ref``, ``prospective_records.claim_id/episode_id/
#    evidence_ref``, quarantine holds on the objects, ``social_memory``
#    evidence, open ``reviews`` effects, ``outcome_receipts`` detail,
#    ``environment_state`` values, vault refs on emptied views).
#
# Anything reached but not safely mapped — propagated copies, foreign
# scopes, unknown kinds, audit-log payloads (``events``, ``decisions``,
# capsules), resolved reviews — is reported in ``unhandled`` verbatim.
# Coverage is only ever claimed for the mapped set.
# ----------------------------------------------------------------------


def _derivation_seeds(
    conn: sqlite3.Connection,
    targets: list[tuple[str, str]],
    emptied_revisions: set[tuple[str, int]],
) -> list[tuple[str, str, Optional[int]]]:
    """Purge targets as derivation-graph refs (rev=None = any revision).

    A ``source`` seeds both ``source`` and ``source_revision`` parent
    kinds — producers may record either — and every emptied revision's
    covering ``source_envelopes`` rows seed ``envelope`` so envelope-fed
    derivations (episodes, transitions) are reached.
    """
    seeds: list[tuple[str, str, Optional[int]]] = []
    seen: set[tuple[str, str, Optional[int]]] = set()

    def add(kind: str, oid: str, rev: Optional[int] = None) -> None:
        ref = (kind, oid, rev)
        if ref not in seen:
            seen.add(ref)
            seeds.append(ref)

    for kind, oid in targets:
        if kind == "source":
            add("source", oid)
            add("source_revision", oid)
        elif kind == "source_revision":
            src, _, rt = oid.rpartition(":")
            if src and rt.isdigit():
                add("source_revision", src, int(rt))
        elif kind in ("span", "claim", "artifact", "episode", "procedure"):
            add(kind, oid)
    if _has_table(conn, "source_envelopes"):
        for src, rev in emptied_revisions:
            for (eid,) in conn.execute(
                "SELECT envelope_id FROM source_envelopes"
                " WHERE source_id = ? AND revision = ?",
                (src, rev),
            ).fetchall():
                add("envelope", eid)
    return seeds


def _scrub_derived_content(
    conn: sqlite3.Connection,
    store: Any,
    sid: str,
    purge_id: str,
    targets: list[tuple[str, str]],
    emptied_revisions: set[tuple[str, int]],
    seq: int,
    stats: dict[str, int],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Delete/suppress derived content mapped to the purged objects.

    Returns ``(counts, unhandled)``: per-area scrub counts for the purge
    receipt, and the honest list of reached-but-unhandled remnants
    (outside-boundary refs, audit payloads, unresolvable mappings).
    """
    counts: dict[str, int] = dict(stats)
    unhandled: list[dict[str, Any]] = []

    purged_pairs = {(k, i) for k, i in targets}
    for src, rev in emptied_revisions:
        purged_pairs.add(("source", src))
        purged_pairs.add(("source_revision", f"{src}:{rev}"))
    # Covering envelopes of emptied revisions are purged objects too —
    # their ids drive quarantine-hold closing and reference scrubbing
    # before the skeleton rows themselves are deleted below.
    if _has_table(conn, "source_envelopes"):
        for src, rev in emptied_revisions:
            for (eid,) in conn.execute(
                "SELECT envelope_id FROM source_envelopes"
                " WHERE source_id = ? AND revision = ?",
                (src, rev),
            ).fetchall():
                purged_pairs.add(("envelope", eid))
                purged_pairs.add(("source_envelope", eid))
    purged_ids = [i for _k, i in sorted(purged_pairs)]

    # Observations whose evidence rows are about to be stripped: collect
    # them now so the orphan check below only tombstones rows that
    # actually lost evidence in this purge.
    obs_at_risk: set[str] = set()
    if _has_table(conn, "observation_evidence"):
        for k, i in purged_pairs:
            for (oid,) in conn.execute(
                "SELECT DISTINCT observation_id FROM observation_evidence"
                " WHERE object_kind = ? AND object_id = ?",
                (k, i),
            ).fetchall():
                obs_at_risk.add(oid)

    # -- derivation-graph closure via the one resumable engine (F4-07).
    #    Absent seeds are legal roots — the engine still enumerates their
    #    dangling derivation edges. privacy.closure imports this module at
    #    top level, so the engine import stays inside the call.
    seeds = _derivation_seeds(conn, targets, emptied_revisions)
    if seeds and (
        _has_table(conn, "derivations") or _has_table(conn, "dependency_edges")
    ):
        counts_closure, closure_unhandled = _apply_closure(
            conn,
            store=store,
            sid=sid,
            purge_id=purge_id,
            seeds=seeds,
            seq=seq,
        )
        for k, v in counts_closure.items():
            counts[k] = counts.get(k, 0) + v
        unhandled.extend(closure_unhandled)
    elif seeds:
        # No graph — derived coverage cannot be established; say so.
        unhandled.append(
            {
                "reason": "no_derivations_table",
                "detail": "derivation graph absent — derived-plane coverage"
                " limited to reference-mapped columns",
            }
        )

    # Envelope skeletons: seeded envelopes' content was emptied above; the
    # rows themselves go (v3 closure semantics — the erasure ledger carries
    # the tombstone). Only revisions whose bytes were emptied are affected.
    if _has_table(conn, "source_envelopes"):
        for src, rev in emptied_revisions:
            if _has_table(conn, "step_observations"):
                conn.execute(
                    "DELETE FROM step_observations WHERE envelope_id IN"
                    " (SELECT envelope_id FROM source_envelopes"
                    "  WHERE source_id = ? AND revision = ?)",
                    (src, rev),
                )
            cur = conn.execute(
                "DELETE FROM source_envelopes WHERE source_id = ? AND revision = ?",
                (src, rev),
            )
            counts["source_envelopes_deleted"] = (
                counts.get("source_envelopes_deleted", 0) + cur.rowcount
            )
        # Unconstrained refs onto deleted envelopes are cleared, not left
        # dangling into erased material.
        env_ids = [i for k, i in purged_pairs if k == "source_envelope"]
        if env_ids and _has_table(conn, "trajectory_steps"):
            ph = ",".join("?" for _ in env_ids)
            conn.execute(
                f"UPDATE trajectory_steps SET action_envelope_id = NULL"
                f" WHERE action_envelope_id IN ({ph})",
                env_ids,
            )

    # V7 plane (SPEC_V7 §30): per-revision sweeps already ran inside
    # _empty_revision above; this sweeps whole-source for each purged
    # source — catching units whose revision row is already gone and
    # re-asserting idempotence after the closure drain (the drain can
    # mint nothing V7, but a concurrent same-tx projection could).
    if _has_table(conn, "units"):
        from .privacy import closure_v7 as _closure_v7

        for kind, oid in sorted(purged_pairs):
            if kind != "source":
                continue
            v7 = _closure_v7.delete_source_v7(conn, oid)
            for table, n in v7["deleted"].items():
                counts[f"v7_{table}"] = counts.get(f"v7_{table}", 0) + n
            for table, n in v7["audit_retained"].items():
                counts[f"v7_audit_{table}"] = (
                    counts.get(f"v7_audit_{table}", 0) + n
                )

    # -- reference-mapped scrubbing -------------------------------------
    counts.update(_scrub_prospective_refs(conn, purged_pairs))
    counts["working_items"] = _scrub_working_refs(conn, purged_pairs)
    counts["social_memory"] = _scrub_social_refs(conn, sid, purged_ids)
    counts["receipts"] = _scrub_receipt_refs(conn, sid, purged_ids)
    counts["environment_state"] = _scrub_environment_refs(conn, sid, purged_ids)
    counts["episodes_refs"] = _scrub_episode_refs(conn, purged_ids)
    counts["edges"] = _retire_edges(conn, purged_pairs, seq)
    counts["freshness"] = _scrub_freshness_refs(conn, purged_pairs)
    counts["quarantine"] = _close_quarantine_holds(conn, purged_pairs, seq)
    counts["ticket_objects"] = _scrub_ticket_refs(conn, purged_ids)
    counts["vault"] = _scrub_vault_refs(conn, emptied_revisions, seq)
    counts["redaction_spans"] = _scrub_redaction_refs(conn, emptied_revisions)
    counts["observations_suppressed"] = _suppress_orphan_observations(
        conn, obs_at_risk, seq
    )

    # -- audit surfaces that cannot be rewritten: report, never claim ----
    unhandled.extend(_audit_remnants(conn, sid, purged_ids, seq, counts))
    return counts, unhandled


def _apply_closure(
    conn: sqlite3.Connection,
    store: Any,
    sid: str,
    purge_id: str,
    seeds: list[tuple[str, str, Optional[int]]],
    seq: int,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Plan + apply the derivation-graph closure inside this transaction,
    through the shared resumable engine (F4-07, V4-38).

    ``privacy.closure.ClosureEngine`` runs ``begin`` (suppression registry
    + frontier) then ``drain`` — bounded frontier steps executed inside
    this same caller transaction, so the purge keeps its atomic
    all-or-nothing semantics: an incomplete run raises and the whole purge
    rolls back, leaving suppression committed from the earlier phase.
    ``privacy.closure`` imports ``purge`` at module level — the lazy import
    here keeps the module graph acyclic.
    """
    from .privacy.closure import ClosureEngine  # lazy: closure → purge
    from .core.types_v4 import ClosurePhase

    engine = ClosureEngine(store)
    run = engine.begin(
        seeds, sid, conn=conn, purge_id=purge_id, run_id=f"purge:{purge_id}"
    )
    run = engine.drain(run.run_id, conn=conn)
    if run.phase is not ClosurePhase.COMPLETED:
        raise VerbatimError(
            ErrorCode.INTEGRITY,
            f"derived-content closure incomplete: run {run.run_id} "
            f"{run.phase.value}: {run.error}",
        )
    receipt = engine.receipt(run.run_id, conn=conn)
    counts: dict[str, int] = dict(receipt["surfaces"])
    unhandled: list[dict[str, Any]] = [
        {
            "reason": o.get("reason", "outside_boundary"),
            "ref": [str(v) for v in o.get("ref", ())],
            "detail": o.get("detail", ""),
        }
        for o in receipt["outside_boundary"]
    ]
    unhandled.extend(
        {
            "reason": "unaffected_derived",
            "ref": [str(v) for v in u.get("ref", ())],
            "detail": "reached by the closure; needed no action",
        }
        for u in receipt["unaffected"]
    )
    unhandled.extend(
        {
            "reason": "propagated_copy",
            "ref": [str(v) for v in e.get("ref", ())],
            "detail": e.get("detail", ""),
        }
        for e in receipt["external_copies"]
    )
    return counts, unhandled


def _any_ref_clause(col: str, ids: list[str]) -> tuple[str, list[str]]:
    """SQL fragment matching ``col`` against any literal id occurrence."""
    clause = " OR ".join(f"{col} LIKE '%' || ? || '%'" for _ in ids)
    return f"({clause})", list(ids)


def _scrub_prospective_refs(
    conn: sqlite3.Connection,
    purged_pairs: set[tuple[str, str]],
) -> dict[str, int]:
    """Prospective records whose bound claim/episode is purged lose their
    intention text — the text stands on the erased object."""
    if not _has_table(conn, "prospective_records"):
        return {}
    cols = _table_columns(conn, "prospective_records")
    claim_ids = [i for k, i in purged_pairs if k == "claim"]
    episode_ids = [i for k, i in purged_pairs if k == "episode"]
    where: list[str] = []
    params: list[Any] = []
    if claim_ids:
        where.append("claim_id IN (%s)" % ",".join("?" for _ in claim_ids))
        params.extend(claim_ids)
    if episode_ids:
        where.append("episode_id IN (%s)" % ",".join("?" for _ in episode_ids))
        params.extend(episode_ids)
    if "evidence_ref" in cols:
        all_ids = [i for _k, i in purged_pairs]
        if all_ids:
            clause, extra = _any_ref_clause("evidence_ref", all_ids)
            where.append(clause)
            params.extend(extra)
    if not where:
        return {}
    scrub = ["intention_text = '[purged]'", "claim_id = NULL",
             "episode_id = NULL", "recurrence_json = NULL"]
    if "evidence_ref" in cols:
        scrub.append("evidence_ref = NULL")
    if "event_id" in cols:
        scrub.append("event_id = NULL")
    if "subgoal_of" in cols:
        scrub.append("subgoal_of = NULL")
    cur = conn.execute(
        f"UPDATE prospective_records SET {', '.join(scrub)}"
        f" WHERE {' OR '.join('(' + w + ')' for w in where)}",
        params,
    )
    return {"prospective_records": cur.rowcount}


def _scrub_working_refs(
    conn: sqlite3.Connection, purged_pairs: set[tuple[str, str]]
) -> int:
    """Working items pointing at a purged object lose ref + note text.

    ``object_ref`` is stored as ``"kind:id"`` — exact-match each purged
    pair so lookalike ids never take an innocent item down with them.
    """
    if not _has_table(conn, "working_set_items"):
        return 0
    refs = [f"{k}:{i}" for k, i in sorted(purged_pairs)]
    if not refs:
        return 0
    ph = ",".join("?" for _ in refs)
    cur = conn.execute(
        f"UPDATE working_set_items SET object_ref = NULL, text = NULL"
        f" WHERE object_ref IN ({ph})",
        refs,
    )
    return cur.rowcount


def _scrub_social_refs(
    conn: sqlite3.Connection, sid: str, purged_ids: list[str]
) -> int:
    """Social records citing purged objects lose value + evidence."""
    if not _has_table(conn, "social_memory") or not purged_ids:
        return 0
    clause, params = _any_ref_clause("evidence_json", purged_ids)
    cur = conn.execute(
        f"UPDATE social_memory SET value_json = '{{}}', evidence_json = '[]'"
        f" WHERE scope_id = ? AND {clause}",
        [sid, *params],
    )
    return cur.rowcount


def _scrub_receipt_refs(
    conn: sqlite3.Connection, sid: str, purged_ids: list[str]
) -> int:
    """Outcome receipts quoting purged objects lose checker detail."""
    if not _has_table(conn, "outcome_receipts") or not purged_ids:
        return 0
    clause, params = _any_ref_clause("detail_json", purged_ids)
    cur = conn.execute(
        f"UPDATE outcome_receipts SET detail_json = '{{}}',"
        f" checked_artifact = NULL WHERE {clause}",
        params,
    )
    return cur.rowcount


def _scrub_environment_refs(
    conn: sqlite3.Connection, sid: str, purged_ids: list[str]
) -> int:
    """Environment values embedding a purged object ref are scrubbed."""
    if not _has_table(conn, "environment_state") or not purged_ids:
        return 0
    clause, params = _any_ref_clause("value", purged_ids)
    cur = conn.execute(
        f"UPDATE environment_state SET value = '[purged]'"
        f" WHERE scope_id = ? AND {clause}",
        [sid, *params],
    )
    return cur.rowcount


def _scrub_episode_refs(conn: sqlite3.Connection, purged_ids: list[str]) -> int:
    """Episode host correlation fields naming a purged object are cleared."""
    if not _has_table(conn, "episodes") or not purged_ids:
        return 0
    total = 0
    for col in ("host_task_id", "host_session_id", "parent_episode_id"):
        clause, params = _any_ref_clause(col, purged_ids)
        cur = conn.execute(
            f"UPDATE episodes SET {col} = NULL WHERE {clause}", params
        )
        total += cur.rowcount
    return total


def _retire_edges(
    conn: sqlite3.Connection, purged_pairs: set[tuple[str, str]], seq: int
) -> int:
    """Retire live edges that name a purged object on either end."""
    if not _has_table(conn, "edges"):
        return 0
    total = 0
    for k, i in sorted(purged_pairs):
        cur = conn.execute(
            "UPDATE edges SET retired_event = ? WHERE retired_event IS NULL"
            " AND ((source_kind = ? AND source_id = ?)"
            "  OR (target_kind = ? AND target_id = ?))",
            (seq, k, i, k, i),
        )
        total += cur.rowcount
    return total


def _scrub_freshness_refs(
    conn: sqlite3.Connection, purged_pairs: set[tuple[str, str]]
) -> int:
    """Freshness markers for erased objects are meaningless — drop them."""
    if not _has_table(conn, "freshness"):
        return 0
    total = 0
    for k, i in sorted(purged_pairs):
        cur = conn.execute(
            "DELETE FROM freshness WHERE object_kind = ? AND object_id = ?",
            (k, i),
        )
        total += cur.rowcount
    return total


def _close_quarantine_holds(
    conn: sqlite3.Connection, purged_pairs: set[tuple[str, str]], seq: int
) -> int:
    """Open holds on purged objects close as 'purged'; findings excerpts go."""
    if not _has_table(conn, "quarantine"):
        return 0
    # Quarantine kinds observed on the write path; a purged object closes
    # under every alias the producers use.
    aliases = {
        "source_envelope": ("envelope", "source_envelope"),
        "envelope": ("envelope", "source_envelope"),
    }
    total = 0
    for k, i in sorted(purged_pairs):
        for qk in aliases.get(k, (k,)):
            cur = conn.execute(
                "UPDATE quarantine SET state = 'purged', findings_json = '[]',"
                " decided_event = ?"
                " WHERE object_kind = ? AND object_id = ?"
                " AND state != 'purged'",
                (seq, qk, i),
            )
            total += cur.rowcount
    return total


def _scrub_ticket_refs(conn: sqlite3.Connection, purged_ids: list[str]) -> int:
    """Action tickets naming a purged object can no longer hydrate it."""
    if not _has_table(conn, "ticket_objects") or not purged_ids:
        return 0
    ph = ",".join("?" for _ in purged_ids)
    cur = conn.execute(
        f"DELETE FROM ticket_objects WHERE object_id IN ({ph})", purged_ids
    )
    return cur.rowcount


def _scrub_vault_refs(
    conn: sqlite3.Connection,
    emptied_revisions: set[tuple[str, int]],
    seq: int,
) -> int:
    """Vault rows bound to emptied views are erased with the view bytes."""
    if not (_has_table(conn, "vault_refs") and _has_table(conn, "vault_entries")):
        return 0
    if not (_has_table(conn, "source_views") and emptied_revisions):
        return 0
    total = 0
    for src, rev in emptied_revisions:
        view_ids = [
            r[0]
            for r in conn.execute(
                "SELECT view_id FROM source_views"
                " WHERE source_id = ? AND revision = ?",
                (src, rev),
            ).fetchall()
        ]
        if not view_ids:
            continue
        ph = ",".join("?" for _ in view_ids)
        entry_ids = [
            r[0]
            for r in conn.execute(
                f"SELECT DISTINCT entry_id FROM vault_refs WHERE view_id IN ({ph})",
                view_ids,
            ).fetchall()
        ]
        conn.execute(
            f"DELETE FROM vault_refs WHERE view_id IN ({ph})", view_ids
        )
        for eid in entry_ids:
            conn.execute(
                "UPDATE vault_entries SET erased_event = ?"
                " WHERE entry_id = ? AND erased_event IS NULL",
                (seq, eid),
            )
        total += len(entry_ids)
    return total


def _scrub_redaction_refs(
    conn: sqlite3.Connection, emptied_revisions: set[tuple, int]
) -> int:
    """Redaction bookkeeping for emptied views goes with the view."""
    if not (
        _has_table(conn, "redaction_spans")
        and _has_table(conn, "source_views")
        and emptied_revisions
    ):
        return 0
    total = 0
    for src, rev in emptied_revisions:
        view_ids = [
            r[0]
            for r in conn.execute(
                "SELECT view_id FROM source_views"
                " WHERE source_id = ? AND revision = ?",
                (src, rev),
            ).fetchall()
        ]
        if view_ids:
            ph = ",".join("?" for _ in view_ids)
            cur = conn.execute(
                f"DELETE FROM redaction_spans WHERE view_id IN ({ph})", view_ids
            )
            total += cur.rowcount
    return total


def _suppress_orphan_observations(
    conn: sqlite3.Connection, obs_at_risk: set[str], seq: int
) -> int:
    """Tombstone observations whose evidence was stripped by this purge.

    ``obs_at_risk`` was collected before reference stripping; an at-risk
    observation left with zero evidence rows has nothing to stand on —
    close its recorded time. At-risk observations retaining some evidence
    are marked stale so producers re-derive them (never a silent partial
    remnant). Observations already deleted by the closure are unaffected.
    """
    if not (
        obs_at_risk
        and _has_table(conn, "observations")
        and _has_table(conn, "observation_evidence")
    ):
        return 0
    ids = sorted(obs_at_risk)
    ph = ",".join("?" for _ in ids)
    suppressed = conn.execute(
        f"UPDATE observations SET"
        f" stale_since_seq = COALESCE(stale_since_seq, ?),"
        f" recorded_until = COALESCE(recorded_until, ?)"
        f" WHERE observation_id IN ({ph}) AND recorded_until IS NULL"
        f" AND NOT EXISTS (SELECT 1 FROM observation_evidence e"
        f"  WHERE e.observation_id = observations.observation_id)",
        (seq, seq, *ids),
    ).rowcount
    conn.execute(
        f"UPDATE observations SET"
        f" stale_since_seq = COALESCE(stale_since_seq, ?)"
        f" WHERE observation_id IN ({ph})",
        (seq, *ids),
    )
    return suppressed


def _audit_remnants(
    conn: sqlite3.Connection,
    sid: str,
    purged_ids: list[str],
    seq: int,
    counts: dict[str, int],
) -> list[dict[str, Any]]:
    """Reference-scan audit surfaces; scrub what is mapped, report the rest.

    ``events``/``decisions`` payloads and capsule snapshots are the
    append-only audit spine — rewriting them would falsify the record, so
    hits are counted into ``counts`` for visibility and disclosed as
    ``unhandled`` (they carry object ids/keys, and may embed content a
    caller placed in metadata). Open reviews that name a purged object
    can no longer apply: they go ``stale`` with their effect scrubbed.
    """
    unhandled: list[dict[str, Any]] = []
    if not purged_ids:
        return unhandled

    if _has_table(conn, "reviews"):
        clause, params = _any_ref_clause("proposed_effect_json", purged_ids)
        cur = conn.execute(
            f"UPDATE reviews SET proposed_effect_json = '{{}}', state = 'stale',"
            f" resolved_event = ? WHERE scope_id = ? AND state = 'open'"
            f" AND {clause}",
            [seq, sid, *params],
        )
        counts["reviews"] = cur.rowcount
        n = conn.execute(
            f"SELECT COUNT(*) FROM reviews WHERE scope_id = ?"
            f" AND state != 'open' AND {clause}",
            [sid, *params],
        ).fetchone()[0]
        if n:
            unhandled.append(
                {
                    "reason": "audit_immutable",
                    "table": "reviews",
                    "detail": f"{n} resolved review(s) reference purged objects;"
                    " proposed_effect_json retained as audit",
                }
            )

    for table, col, label in (
        ("events", "payload_json", "event log entries"),
        ("decisions", "result_json", "decision records"),
        ("handoff_capsules", "snapshot_json", "capsule snapshots"),
    ):
        if not _has_table(conn, table):
            continue
        clause, params = _any_ref_clause(col, purged_ids)
        n = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE scope_id = ? AND {clause}",
            [sid, *params],
        ).fetchone()[0]
        if n:
            unhandled.append(
                {
                    "reason": "audit_immutable",
                    "table": table,
                    "detail": f"{n} {label} reference purged object ids;"
                    " append-only audit payloads are reported, not rewritten",
                }
            )
    return unhandled


def _next_seq(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute("SELECT COALESCE(MAX(event_seq), 0) + 1 FROM events").fetchone()[0]
    )


def _bump_generation(store: Any, conn: sqlite3.Connection) -> int:
    bump = getattr(store, "bump_generation", None)
    if callable(bump):
        return int(bump(conn))
    return 0


def _bump_erasure_epoch(store: Any, conn: sqlite3.Connection) -> int:
    """Advance the store's erasure epoch; fences later commits (V2-41.08)."""
    get = getattr(store, "_meta_get", None)
    set_ = getattr(store, "_meta_set", None)
    if not (callable(get) and callable(set_)):
        return 0
    current = get(conn, "erasure_epoch") or 0
    nxt = int(current) + 1
    set_(conn, "erasure_epoch", nxt)
    return nxt


# ----------------------------------------------------------------------
# restore fencing (V2-40.14, V2-41.15)
# ----------------------------------------------------------------------


def check_restore_fence(
    store: Any,
    conn: sqlite3.Connection,
    scope_id: str,
    object_kind: str,
    object_id: str,
) -> bool:
    """True when the object was erased — restore/import must skip it.

    The erasure ledger stores only opaque profile-keyed digests, never the
    raw id, so this check cannot confirm low-entropy identities by
    enumeration. A missing ledger (schema v1 store) fails open — there is
    no recorded erasure to enforce.
    """
    require_id(scope_id, "scope_id")
    if object_kind not in OBJECT_KINDS:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown object_kind {object_kind!r}"
        )
    require_id(object_id, "object_id")
    if not _has_table(conn, "erasure_ledger"):
        return False
    return ErasureRepo(store).is_erased(conn, scope_id, object_kind, object_id)


__all__ = [
    "OBJECT_KINDS",
    "check_restore_fence",
    "execute_purge",
    "lift_suppression",
    "plan_purge",
    "suppress",
]
