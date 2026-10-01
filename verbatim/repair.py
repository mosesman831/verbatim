"""Repair-local memory recomputation (SPEC_V4_5 §04, V45-04.*).

A localized source correction must invalidate dependents immediately and
recompute only the affected dependency closure — never a full rebuild of
the scope (D03). An impact plan that misses a real dependent fails the
job rather than publishing a partial view (D04, V45-04.03).

This module is a composition layer, not a second engine: it walks the
same ``derivations`` + ``dependency_edges`` graphs the kernel and the
deletion-closure engine walk (plus every registered closure side
walker), plans against the caller's snapshot, and dispatches to the real
producers — ``synthesis.Synthesizer`` for derived views,
``observations.consolidate.consolidate_windowed`` for observations,
``experience.scenes.refresh_scene`` for scenes, and
``profiles.ProfileService.refresh`` for profile entries. Anything it
cannot recompute is named honestly in the plan; applying an incomplete
plan raises ``CONTEXT_INCOMPLETE`` before any write commits.

Two entry points matter operationally:

- ``invalidate_dependents`` — the *immediate* step (V45-04.02). Runs
  ``kernel.invalidate`` inside the caller's transaction, then marks
  affected views ``held`` and affected observations stale in the same
  commit so no stale readable view survives the correction.
- ``apply_repair`` — the *recompute* step. Re-walks the graph inside the
  write transaction, fences the plan against it, marks held, recomputes
  the touched consolidation window and scenes, then recomposes each
  affected view through the real ``compose`` path and refreshes profile
  entries through the real service.

``full_rebuild`` is the comparison baseline: the same producers run over
every recomputable object in the scope, so measured recomputation counts
are directly comparable (V45-04.04).
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any, Iterable, Mapping, Optional

from .core.time import now_us as _now_us
from .core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    new_id,
    safe_json_loads,
)
from .core.types_v4 import LifecycleState
from .experience.scenes import SCENE_KIND, refresh_scene
from .governance import CallerV3
from .governance import epochs as _epochs
from .observations.consolidate import (
    ConsolidationWindow,
    consolidate_windowed,
)
from .core.lifecycle import read_claim_head
from .synthesis.types import VIEW_OBJECT_KIND
from .storage import repos_v4


#: Producer identities the plan records per affected member.
P_CONSOLIDATION = "consolidation"
P_SYNTHESIS = "synthesis"
P_SCENES = "scenes"
P_PROFILES = "profiles"
P_BRANCHES = "branches"           # held-only: rebased by operators, never recomputed

#: ``objects`` kind for persisted impact plans (V45-13.02 — durable,
#: digest-bound, content-minimized).
REPAIR_PLAN_KIND = "repair_plan"
PLAN_DOC = "repair_plan/v1"
BRANCH_KIND = "branch"
PRODUCER_ID = "producer:verbatim.repair.v1"

#: Traversal bound for the impact walk — same order as derivations._MAX_NODES.
_MAX_NODES = 4096
_MAX_DEPTH = 64

#: Kinds that legitimately appear as dependents but are lifecycle/evidence
#: objects — named in the plan for invalidation, never "recomputed".
_AFFECTED_ONLY_KINDS = frozenset({
    "claim",
    "span",
    "source",
    "source_revision",
    "envelope",
    "artifact",
    "transition",
    "procedure",
    "plan",
    "working",
    "social",
    "environment",
    "state_anchor",
    "trajectory",
    "trajectory_step",
    "vault_entry",
    "entity",
    "repair_plan",
})

#: Objects-registry kinds whose rows carry disposition — set directly by
#: ``mark_held``; everything else has a producer-owned marker.
_OBJECTS_HELD_KINDS = frozenset({BRANCH_KIND})


# ----------------------------------------------------------------------
# reference normalization + impact walk
# ----------------------------------------------------------------------


def _norm_ref(item: Any) -> tuple[str, str, Optional[int]]:
    """Lax ref normalization — closure-style tombstone intents, not reads.

    Accepts ``(kind, id[, revision])`` tuples or dicts with
    ``kind``/``object_kind``, ``id``/``object_id``, ``revision``. Unlike
    ``derivations.normalize_ref`` this does not restrict kinds to the v3
    OBJECT_KINDS allowlist: the impact set legitimately contains
    ``derived_view``, ``branch``, and ``repair_plan`` refs that the v3
    registry predates.
    """
    kind: Any = None
    oid: Any = None
    rev: Any = None
    if isinstance(item, Mapping):
        kind = item.get("kind", item.get("object_kind"))
        oid = item.get("id", item.get("object_id"))
        rev = item.get("revision", item.get("object_revision"))
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
                "object refs must be (kind, id[, revision])",
            )
    if not isinstance(kind, str) or not kind:
        raise VerbatimError(ErrorCode.VALIDATION, "object ref kind required")
    if not isinstance(oid, str) or not oid:
        raise VerbatimError(ErrorCode.VALIDATION, "object id required")
    if rev is not None:
        if isinstance(rev, bool) or not isinstance(rev, int) or rev < 1:
            raise VerbatimError(
                ErrorCode.VALIDATION, "object revision must be an int >= 1"
            )
    return (kind, oid, rev)


def _children(
    conn: sqlite3.Connection,
    ref: tuple[str, str, Optional[int]],
) -> list[tuple[str, str, int]]:
    """Children of ``ref`` across every registered edge source.

    ``derivations`` + ``dependency_edges`` are queried with the kernel's
    revision semantics (a ``None`` parent revision matches any revision);
    every closure side walker contributes its own children so reference
    tables that live outside the two edge tables still propagate impact.
    """
    from .privacy.closure import _SIDE_WALKERS  # shared registry

    out: set[tuple[str, str, int]] = set()
    kind, oid, rev = ref
    for table in ("derivations", "dependency_edges"):
        sql = (
            f"SELECT child_kind, child_id, child_revision FROM {table}"
            " WHERE parent_kind = ? AND parent_id = ?"
        )
        params: list[Any] = [kind, oid]
        if rev is not None:
            sql += " AND parent_revision = ?"
            params.append(rev)
        for r in conn.execute(sql, params).fetchall():
            out.add((r[0], r[1], int(r[2])))
    for walker in _SIDE_WALKERS.values():
        try:
            for child in walker["children"](conn, ref) or ():
                ck, cid, crev = child
                out.add(
                    (ck, cid, int(crev) if crev is not None else 0)
                )
        except Exception:
            continue  # a broken walker degrades, never blocks (closure idiom)
    return sorted(out)


def impact_closure(
    conn: sqlite3.Connection,
    seeds: Iterable[Any],
    *,
    max_depth: int = _MAX_DEPTH,
    max_nodes: int = _MAX_NODES,
) -> dict[tuple[str, str, int], int]:
    """Complete transitive dependent set of ``seeds`` — the impact set.

    Returns ``{ref: min_depth}`` over every object reachable through
    ``derivations``, ``dependency_edges``, and registered closure side
    walkers. Bounded BFS (cycle-safe via the depth bound); exceeding the
    node bound raises ``VALIDATION`` rather than silently truncating —
    a truncated closure is exactly the hidden-dependency failure D04
    forbids.
    """
    if isinstance(max_depth, bool) or not (1 <= int(max_depth) <= _MAX_DEPTH):
        raise VerbatimError(
            ErrorCode.VALIDATION, f"max_depth must be 1..{_MAX_DEPTH}"
        )
    if isinstance(max_nodes, bool) or not (1 <= int(max_nodes)):
        raise VerbatimError(
            ErrorCode.VALIDATION, "max_nodes must be a positive int"
        )
    seed_list = [_norm_ref(s) for s in (seeds or ())]
    out: dict[tuple[str, str, int], int] = {}
    seen: set[tuple[str, str, Optional[int]]] = set(seed_list)
    frontier: list[tuple[tuple[str, str, Optional[int]], int]] = [
        (s, 0) for s in seed_list
    ]
    while frontier:
        ref, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        for child in _children(conn, ref):
            if child in seen:
                continue
            seen.add(child)
            d = depth + 1
            if child not in out or d < out[child]:
                out[child] = d
            frontier.append((child, d))
            if len(out) > max_nodes:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "impact traversal exceeds node bound — refusing to plan "
                    "against a truncated dependency closure",
                )
    return out


# ----------------------------------------------------------------------
# classification helpers
# ----------------------------------------------------------------------


def _is_scene(conn: sqlite3.Connection, episode_id: str) -> bool:
    row = conn.execute(
        "SELECT kind FROM episodes WHERE episode_id = ?", (episode_id,)
    ).fetchone()
    return row is not None and row[0] == SCENE_KIND


def _classify(
    conn: sqlite3.Connection, ref: tuple[str, str, int]
) -> tuple[str, Optional[str]]:
    """``(bucket, producer)`` for one impacted ref.

    Buckets: ``target`` (a registered producer can recompute it),
    ``held_only`` (marked held; rebased/applied by an operator, never
    auto-recomputed), ``affected_only`` (lifecycle/evidence object named
    for invalidation), ``unsupported`` (an impacted dependent no producer
    can honestly recompute — fails the plan).
    """
    kind, oid, _rev = ref
    if kind == "observation":
        return ("target", P_CONSOLIDATION)
    if kind == VIEW_OBJECT_KIND:
        return ("target", P_SYNTHESIS)
    if kind == "profile":
        return ("target", P_PROFILES)
    if kind == "episode":
        return ("target", P_SCENES) if _is_scene(conn, oid) else (
            "affected_only", None
        )
    if kind == BRANCH_KIND:
        return ("held_only", P_BRANCHES)
    if kind in _AFFECTED_ONLY_KINDS:
        return ("affected_only", None)
    return ("unsupported", None)


# ----------------------------------------------------------------------
# impact plans (V45-04.01/04.03)
# ----------------------------------------------------------------------


def plan_repair(
    conn: sqlite3.Connection,
    scope_id: str,
    changed_refs: Iterable[Any],
    *,
    declared: Optional[Iterable[Any]] = None,
    since_seq: Optional[int] = None,
    claim_ids: Optional[Iterable[str]] = None,
    now_us: Optional[int] = None,
) -> dict[str, Any]:
    """Build the dependency impact plan for a localized correction.

    Runs entirely against the caller's snapshot (read or write conn —
    caller chooses). Returns the plan dict; persist it durably through
    ``persist_plan`` before work starts (V45-04.01).

    ``declared`` — an optional caller/ producer-supplied forecast of the
    affected set. Any real closure member missing from ``declared`` is a
    hidden dependency: the plan records it under ``missing`` and sets
    ``complete = False`` so ``apply_repair`` refuses it (D04, V45-04.03).

    ``claim_ids`` — explicit claims whose consolidation slots are touched
    (usually the corrected claims); the changed/affected claim refs are
    always unioned in automatically.
    """
    if not isinstance(scope_id, str) or not scope_id:
        raise VerbatimError(ErrorCode.VALIDATION, "scope_id required")
    seeds = [_norm_ref(s) for s in (changed_refs or ())]
    if not seeds:
        raise VerbatimError(
            ErrorCode.VALIDATION, "a repair needs >= 1 changed object"
        )
    affected = impact_closure(conn, seeds)
    declared_refs: Optional[set[tuple[str, str, Optional[int]]]] = None
    if declared is not None:
        declared_refs = {_norm_ref(d) for d in declared}
        missing = [
            list(r) for r in sorted(affected) if r not in declared_refs
        ]
    else:
        missing = []

    targets: list[dict[str, Any]] = []
    held_only: list[dict[str, Any]] = []
    affected_only: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    touched_claims: set[str] = set(claim_ids or ())
    for ref, depth in sorted(affected.items()):
        kind, oid, rev = ref
        bucket, producer = _classify(conn, ref)
        entry = {
            "ref": [kind, oid, rev],
            "depth": depth,
            "producer": producer,
        }
        if bucket == "target":
            targets.append(entry)
        elif bucket == "held_only":
            held_only.append(entry)
        elif bucket == "affected_only":
            affected_only.append(entry)
        else:
            unsupported.append(entry)
        if kind == "claim":
            touched_claims.add(oid)
    for kind, oid, _rev in seeds:
        if kind == "claim":
            touched_claims.add(oid)

    # Content-minimized follow-up naming (V45-04.01): the jobs a repair
    # must run are the recompute dispatches themselves; index/cache
    # impact is named by class, never enumerated by bytes.
    follow_ups = [
        {
            "job": "recompute",
            "producer": t["producer"],
            "ref": t["ref"],
        }
        for t in targets
    ] + [
        {"job": "mark_held", "producer": P_BRANCHES, "ref": h["ref"]}
        for h in held_only
    ]
    changed_kinds = {s[0] for s in seeds}
    affected_kinds = {r[0] for r in affected} | changed_kinds
    indexes = sorted(
        {"fts_rows", "facts_fts"} & (
            {"fts_rows", "facts_fts"}
            if affected_kinds & {"claim", "span", "source", "source_revision"}
            else set()
        )
    )
    caches = [f"retrieval.cache:{scope_id}"] if affected else []

    return {
        "doc": PLAN_DOC,
        "plan_id": None,
        "scope_id": scope_id,
        "changed": [list(s) for s in seeds],
        "affected": [list(r) for r in sorted(affected)],
        "depths": {
            f"{k}|{i}|{r}": d for (k, i, r), d in affected.items()
        },
        "targets": targets,
        "held_only": held_only,
        "affected_only": affected_only,
        "unsupported": unsupported,
        "missing": missing,
        "indexes": indexes,
        "caches": caches,
        "follow_ups": follow_ups,
        "claim_ids": sorted(touched_claims),
        "since_seq": since_seq,
        "base_epoch": _epochs.current_epoch(conn, scope_id),
        "complete": not unsupported and not missing,
        "created_us": int(now_us) if now_us is not None else _now_us(),
    }


def _doc_digest(store: Any, doc: Mapping[str, Any]) -> str:
    """Keyed digest binding the persisted plan doc (tamper-evident)."""
    return "hmac-sha256:" + store.hmac(json_dumps(doc).encode("utf-8")).hex()


def persist_plan(
    store: Any,
    conn: sqlite3.Connection,
    plan: dict[str, Any],
) -> str:
    """Persist ``plan`` as a digest-bound ``repair_plan`` object.

    Registers the repair producer manifest on first write (V45-13.01)
    and records ``dependency_edges`` from the plan to each changed
    revision so the plan joins the same ancestry graph every other
    derived object uses. Returns the plan id. Caller owns the tx.
    """
    if plan.get("doc") != PLAN_DOC:
        raise VerbatimError(ErrorCode.VALIDATION, "not a repair plan doc")
    register_producer(store, conn)
    plan_id = f"rplan:{new_id()}"
    doc = dict(plan)
    doc["plan_id"] = plan_id
    doc["revision"] = 1
    doc["state"] = "planned"
    repos_v4.insert(
        conn,
        "objects",
        {
            "object_id": plan_id,
            "kind": REPAIR_PLAN_KIND,
            "scope_id": plan["scope_id"],
            "current_revision": 1,
            "disposition": LifecycleState.ACTIVE.value,
            "created_event": store.next_event_us(),
        },
    )
    repos_v4.insert(
        conn,
        "object_revisions",
        {
            "kind": REPAIR_PLAN_KIND,
            "object_id": plan_id,
            "revision": 1,
            "digest": _doc_digest(store, doc),
            "recorded_from": store.next_event_us(),
            "producer_ref": PRODUCER_ID,
            "metadata_json": doc,
        },
    )
    op_id = f"op:{new_id()}"
    for seq, (kind, oid, rev) in enumerate(plan["changed"]):
        repos_v4.insert(
            conn,
            "dependency_edges",
            {
                "child_kind": REPAIR_PLAN_KIND,
                "child_id": plan_id,
                "child_revision": 1,
                "parent_kind": kind,
                "parent_id": oid,
                "parent_revision": int(rev) if rev is not None else 1,
                "role": "trigger",
                "producer_id": PRODUCER_ID,
                "operation_id": op_id,
                "seq": seq,
            },
        )
    return plan_id


def register_producer(store: Any, conn: sqlite3.Connection) -> str:
    """Idempotent producer-manifest registration (V45-13.01)."""
    row = repos_v4.get(
        conn, "producer_manifests", {"producer_id": PRODUCER_ID}
    )
    if row is None:
        descriptor = {
            "producer_id": PRODUCER_ID,
            "kind": "repair",
            "plan_doc": PLAN_DOC,
        }

        def _req(parts: Mapping[str, Any]) -> str:
            return "sha256:" + hashlib.sha256(
                json_dumps(parts).encode("utf-8")
            ).hexdigest()

        repos_v4.insert(
            conn,
            "producer_manifests",
            {
                "producer_id": PRODUCER_ID,
                "kind": "repair",
                "artifact_digest": _req(
                    {"producer_code": "verbatim.repair", **descriptor}
                ),
                "rubric_digest": _req({"rubric": "repair.v1"}),
                "config_digest": _req(descriptor),
                "schema_version": 4,
                "license_ref": None,
                "health": "available",
                "registered_us": _now_us(),
            },
        )
    return PRODUCER_ID


def load_plan(
    conn: sqlite3.Connection, store: Any, plan_id: str
) -> Optional[dict[str, Any]]:
    """Latest persisted plan doc with integrity verification.

    ``None`` for an absent plan; ``STORE_CORRUPT`` on a digest mismatch —
    a tampered plan is never applied.
    """
    obj = repos_v4.get(
        conn, "objects", {"kind": REPAIR_PLAN_KIND, "object_id": plan_id}
    )
    if obj is None:
        return None
    rev = repos_v4.get(
        conn,
        "object_revisions",
        {
            "kind": REPAIR_PLAN_KIND,
            "object_id": plan_id,
            "revision": int(obj["current_revision"]),
        },
    )
    if rev is None:
        return None
    raw = rev.get("metadata_json") or "{}"
    doc = safe_json_loads(raw) if isinstance(raw, str) else dict(raw)
    stored = rev.get("digest") or ""
    if stored and stored != _doc_digest(store, doc):
        raise VerbatimError(
            ErrorCode.STORE_CORRUPT,
            f"repair plan {plan_id} doc fails integrity check",
        )
    return doc


# ----------------------------------------------------------------------
# immediate invalidation (V45-04.02)
# ----------------------------------------------------------------------


def mark_held(
    conn: sqlite3.Connection,
    refs: Iterable[Any],
    *,
    synthesizer: Any = None,
    seq: Optional[int] = None,
) -> dict[str, int]:
    """Mark affected objects unavailable inside the caller's transaction.

    - derived views → ``held`` through ``synthesizer.note_invalidation``
      when provided (its doc semantics), else the ``objects.disposition``
      flip directly;
    - observations → ``stale_since_seq`` (the v3 staleness marker — reads
      treat it as invalidated pending recompute);
    - branches → ``objects.disposition = 'held'`` (the branch's own
      ``note_invalidation`` additionally bumps its doc when called through
      ``BranchService``).

    Returns counts per kind. Caller owns the transaction — the marks
    commit atomically with whatever produced them.
    """
    if seq is None:
        seq = int(
            conn.execute(
                "SELECT COALESCE(MAX(event_seq), 0) + 1 FROM events"
            ).fetchone()[0]
        )
    refs = [_norm_ref(r) for r in (refs or ())]
    marked = {"views": 0, "observations": 0, "branches": 0}
    view_refs = [r for r in refs if r[0] == VIEW_OBJECT_KIND]
    if view_refs and synthesizer is not None:
        marked["views"] += synthesizer.note_invalidation(
            conn, {"affected": [tuple(r) for r in view_refs]}
        )
    for kind, oid, _rev in refs:
        if kind == "observation":
            cur = conn.execute(
                "UPDATE observations SET"
                " stale_since_seq = COALESCE(stale_since_seq, ?)"
                " WHERE observation_id = ? AND stale_since_seq IS NULL",
                (seq, oid),
            )
            marked["observations"] += int(cur.rowcount > 0)
        elif kind in _OBJECTS_HELD_KINDS or (
            kind == VIEW_OBJECT_KIND and synthesizer is None
        ):
            obj = repos_v4.get(
                conn, "objects", {"kind": kind, "object_id": oid}
            )
            if obj is not None and obj[
                "disposition"
            ] == LifecycleState.ACTIVE.value:
                repos_v4.update(
                    conn,
                    "objects",
                    {"disposition": LifecycleState.HELD.value},
                    {"kind": kind, "object_id": oid},
                )
                marked[
                    "views" if kind == VIEW_OBJECT_KIND else "branches"
                ] += 1
    return marked


def invalidate_dependents(
    conn: sqlite3.Connection,
    kernel: Any,
    scope_id: str,
    changed_refs: Iterable[Any],
    *,
    event_kind: str = "correction",
    synthesizer: Any = None,
    now_us: Optional[int] = None,
) -> dict[str, Any]:
    """The immediate step of a localized repair (V45-04.02, D03).

    One transaction does everything: ``kernel.invalidate`` bumps the
    scope's authorization epoch and walks ``dependency_edges`` transitively,
    then affected views flip to ``held`` and affected observations take a
    staleness marker — all inside the caller's commit, so a stale readable
    view never survives the correction. Returns the kernel report plus the
    held counts and the full affected set (for ``plan_repair`` symmetry).
    """
    if not isinstance(scope_id, str) or not scope_id:
        raise VerbatimError(ErrorCode.VALIDATION, "scope_id required")
    seeds = [_norm_ref(s) for s in (changed_refs or ())]
    if not seeds:
        raise VerbatimError(
            ErrorCode.VALIDATION, "invalidation needs >= 1 changed object"
        )
    report = kernel.invalidate(
        conn,
        {
            "kind": event_kind,
            "scope_ids": [scope_id],
            "object_refs": [list(s) for s in seeds],
        },
        now_us=now_us,
    )
    affected = impact_closure(conn, seeds)
    marked = mark_held(
        conn, affected, synthesizer=synthesizer
    )
    return {
        "report": report,
        "affected": [list(r) for r in sorted(affected)],
        "marked_held": marked,
    }


# ----------------------------------------------------------------------
# producer dispatch helpers
# ----------------------------------------------------------------------


def _current_revision(
    conn: sqlite3.Connection, kind: str, oid: str
) -> Optional[int]:
    """Current revision of a view input, by kind — ``None`` when the
    object no longer resolves (purged/absent)."""
    if kind == "claim":
        head = read_claim_head(conn, oid)
        if head is None:
            return None
        # An erased or suppression-marked head does not resolve for
        # recompute — rebinding to it would mint a view over dead bytes.
        if head.state.value == "erased" or head.recorded_until is not None:
            return None
        return int(head.revision)
    if kind == "span":
        row = conn.execute(
            "SELECT revision FROM spans WHERE span_id = ?", (oid,)
        ).fetchone()
        return int(row[0]) if row else None
    if kind == "source":
        row = conn.execute(
            "SELECT MAX(revision) FROM source_revisions WHERE source_id = ?",
            (oid,),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None
    if kind == "source_revision":
        row = conn.execute(
            "SELECT MAX(revision) FROM source_revisions WHERE source_id = ?",
            (oid,),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None
    if kind == "observation":
        row = conn.execute(
            "SELECT revision FROM observations WHERE observation_id = ?",
            (oid,),
        ).fetchone()
        return int(row[0]) if row else None
    if kind == "episode":
        row = conn.execute(
            "SELECT revision FROM episodes WHERE episode_id = ?", (oid,)
        ).fetchone()
        return int(row[0]) if row else None
    row = repos_v4.get(conn, "objects", {"kind": kind, "object_id": oid})
    return int(row["current_revision"]) if row else None


def _latest_view_doc(
    conn: sqlite3.Connection, store: Any, view_id: str
) -> Optional[dict[str, Any]]:
    obj = repos_v4.get(
        conn, "objects", {"kind": VIEW_OBJECT_KIND, "object_id": view_id}
    )
    if obj is None:
        return None
    rev = repos_v4.get(
        conn,
        "object_revisions",
        {
            "kind": VIEW_OBJECT_KIND,
            "object_id": view_id,
            "revision": int(obj["current_revision"]),
        },
    )
    if rev is None:
        return None
    raw = rev.get("metadata_json") or "{}"
    doc = safe_json_loads(raw) if isinstance(raw, str) else dict(raw)
    return {"doc": doc, "disposition": obj["disposition"]}


def _recompute_view(
    store: Any,
    synthesizer: Any,
    view_id: str,
    *,
    caller: CallerV3,
    purpose: Optional[str],
    producer_grant: Any = None,
    request: Optional[Mapping[str, Any]] = None,
    now_us: Optional[int] = None,
) -> dict[str, Any]:
    """Recompose one persisted view through the real ``compose`` path.

    Default recompute rebinds the doc's pinned inputs at their CURRENT
    revisions — the honest repair semantics: same evidence set, fresh
    revisions. When a bound input is purged the view genuinely cannot be
    recomputed; it stays ``held`` and the outcome is reported, never
    silently dropped. A ``request`` override (``inputs``/``query``/
    ``sections``) re-derives the logical request instead — when that
    request resolves to a different ``view_id`` the old object is marked
    ``superseded`` and the new one is recorded as its replacement.
    """
    with store.read() as conn:
        found = _latest_view_doc(conn, store, view_id)
    if found is None:
        return {"view_id": view_id, "status": "gone"}
    doc = found["doc"]
    request = request or {}
    try:
        if "inputs" in request or "query" in request or "topic" in request:
            inputs = request.get("inputs")
            query = request.get("query") or request.get("topic")
        else:
            # Rebind pinned inputs at current revisions; unresolvable
            # inputs leave the view held — a partial rebind would mint a
            # view that silently dropped evidence.
            inputs = []
            with store.read() as conn:
                for inp in doc.get("inputs") or ():
                    cur = _current_revision(
                        conn, inp["kind"], inp["id"]
                    )
                    if cur is None:
                        return {
                            "view_id": view_id,
                            "status": "unrecomputable",
                            "reason": "input_gone",
                            "input": [inp["kind"], inp["id"], inp["rev"]],
                        }
                    inputs.append(
                        {"kind": inp["kind"], "id": inp["id"], "revision": cur}
                    )
            query = None
        view = synthesizer.compose(
            doc["scope_id"],
            query,
            caller=caller,
            purpose=purpose,
            view_kind=doc["view_kind"],
            inputs=inputs,
            sections=request.get("sections") or doc.get("sections"),
            producer_grant=producer_grant,
            persist=True,
            now_us=now_us,
        )
    except VerbatimError as exc:
        return {
            "view_id": view_id,
            "status": "failed",
            "error_code": exc.code.value,
            "error": exc.message,
        }
    if view.view_id == view_id:
        return {
            "view_id": view_id,
            "status": "repaired",
            "revision": view.revision,
        }
    # Successor request → the old object is superseded, not deleted: its
    # prior revisions remain the recomputation history.
    with store.tx() as conn:
        obj = repos_v4.get(
            conn, "objects", {"kind": VIEW_OBJECT_KIND, "object_id": view_id}
        )
        if obj is not None and obj["disposition"] in (
            LifecycleState.HELD.value,
            LifecycleState.ACTIVE.value,
        ):
            repos_v4.update(
                conn,
                "objects",
                {"disposition": LifecycleState.SUPERSEDED.value},
                {"kind": VIEW_OBJECT_KIND, "object_id": view_id},
            )
    return {
        "view_id": view_id,
        "status": "recomputed_as",
        "successor": view.view_id,
        "revision": view.revision,
    }


# ----------------------------------------------------------------------
# apply + full-rebuild baseline
# ----------------------------------------------------------------------


def _plan_object(plan_or_id: Any) -> dict[str, Any]:
    if isinstance(plan_or_id, Mapping):
        return dict(plan_or_id)
    raise VerbatimError(
        ErrorCode.VALIDATION, "plan must be a persisted plan doc"
    )


def apply_repair(
    store: Any,
    plan_or_id: Any,
    *,
    caller: Optional[CallerV3] = None,
    purpose: Optional[str] = "admin",
    kernel: Any = None,
    synthesizer: Any = None,
    profile_service: Any = None,
    recompose_requests: Optional[Mapping[str, Mapping[str, Any]]] = None,
    allow_unsupported: bool = False,
    states: Any = None,
    modalities: Any = None,
    min_proof: Optional[int] = None,
    producer_grant: Any = None,
    now_us: Optional[int] = None,
) -> dict[str, Any]:
    """Apply a repair plan — fenced, atomic core + producer dispatches.

    Phase 1 (one ``store.tx()``): the dependency closure is re-walked and
    the plan fenced against it — any dependent the plan does not name is
    a hidden dependency and the whole job fails ``CONTEXT_INCOMPLETE``
    with nothing committed (D04). Inside the same commit the plan's held
    marks are made durable, the touched consolidation window re-derives
    observation slots, and affected scenes reconcile membership.

    Phase 2 (per-object transactions): each affected view recomposes
    through ``Synthesizer.compose`` and profile entries refresh through
    ``ProfileService.refresh`` — real producer paths with their own
    authorization and freshness fences. Failures are reported per object
    and leave the object ``held``; they never publish a partial view.

    ``allow_unsupported`` is the explicit escape hatch: unsupported
    dependents are reported rather than failing the job — the plan
    still records them, so nothing is hidden.
    """
    if isinstance(plan_or_id, str):
        with store.read() as conn:
            plan = load_plan(conn, store, plan_or_id)
        if plan is None:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "repair plan not found"
            )
    else:
        plan = _plan_object(plan_or_id)
    if plan.get("doc") != PLAN_DOC:
        raise VerbatimError(ErrorCode.VALIDATION, "not a repair plan doc")
    scope_id = plan["scope_id"]
    changed = [_norm_ref(s) for s in plan["changed"]]
    plan_affected = {tuple(r) for r in plan["affected"]}
    targets = plan.get("targets") or []
    unsupported = plan.get("unsupported") or []
    missing = plan.get("missing") or []
    if missing:
        raise VerbatimError(
            ErrorCode.CONTEXT_INCOMPLETE,
            "impact plan forecast missed real dependents "
            f"{missing[:8]} — replan before repair",
        )
    if unsupported and not allow_unsupported:
        raise VerbatimError(
            ErrorCode.CONTEXT_INCOMPLETE,
            "impact plan names dependents no producer can recompute: "
            f"{[u['ref'] for u in unsupported][:8]}",
        )
    producers_needed = {t["producer"] for t in targets}
    if P_SYNTHESIS in producers_needed and (
        synthesizer is None or caller is None
    ):
        raise VerbatimError(
            ErrorCode.CONTEXT_INCOMPLETE,
            "plan names derived views but no synthesizer/caller was provided",
        )
    if P_PROFILES in producers_needed and (
        profile_service is None or caller is None
    ):
        raise VerbatimError(
            ErrorCode.CONTEXT_INCOMPLETE,
            "plan names profile entries but no profile service/caller",
        )

    cons_kwargs: dict[str, Any] = {}
    if states is not None:
        cons_kwargs["states"] = states
    if modalities is not None:
        cons_kwargs["modalities"] = modalities
    if min_proof is not None:
        cons_kwargs["min_proof"] = min_proof

    report: dict[str, Any] = {
        "plan_id": plan.get("plan_id"),
        "scope_id": scope_id,
        "objects_scanned": 0,
        "objects_evaluated": 0,
        "objects_recomputed": 0,
        "marked_held": {},
        "consolidation": None,
        "scenes": [],
        "views": [],
        "profiles": None,
        "skipped": [],
        "unsupported": [u["ref"] for u in unsupported],
        "failures": [],
        "complete": True,
    }

    # ---- phase 1: fence + core recomputes in one transaction -----------
    with store.tx() as conn:
        actual = impact_closure(conn, changed)
        # ``repair_plan`` refs are forecast artifacts, not dependents:
        # ``persist_plan`` binds the plan object to its changed refs
        # through the same ancestry graph, so the plan itself (and any
        # sibling plan written against the same correction) appears in
        # the apply-time walk — a plan cannot forecast its own
        # existence. Every real dependent kind still trips the fence.
        hidden = [
            list(r)
            for r in sorted(set(actual) - plan_affected - set(changed))
            if r[0] != REPAIR_PLAN_KIND
        ]
        if hidden:
            raise VerbatimError(
                ErrorCode.CONTEXT_INCOMPLETE,
                "dependency closure grew past the plan — hidden dependents "
                f"found at apply time {hidden[:8]}; nothing committed",
            )
        # The changed revisions must still resolve — a correction planned
        # against a parent that was since purged or erased is a stale
        # plan (same resolvability semantics as input rebinding).
        for kind, oid, rev in changed:
            if _current_revision(conn, kind, oid) is None:
                raise VerbatimError(
                    ErrorCode.STALE_PROPOSAL,
                    f"changed {kind} {oid} no longer resolves",
                )
        marked = mark_held(conn, actual, synthesizer=synthesizer)
        report["marked_held"] = marked

        window = ConsolidationWindow(
            since_seq=plan.get("since_seq"),
            claim_ids=tuple(plan.get("claim_ids") or ()),
        )
        cons = consolidate_windowed(
            conn, scope_id, window=window, **cons_kwargs
        )
        report["consolidation"] = cons.as_dict()
        # ``objects_evaluated`` is the honest D03 work metric: every
        # slot-value candidate a producer actually re-derived — an
        # idempotent pass still did the work even when the stored head
        # was already correct. ``objects_recomputed`` counts durable
        # writes + retirements only.
        report["objects_evaluated"] += cons.candidates_seen
        report["objects_recomputed"] += (
            cons.observations_written + len(cons.retired or [])
        )
        report["objects_scanned"] += cons.slots_touched

        # Scenes dedupe to the object: one ``refresh_scene`` reconciles
        # every impacted revision of the same scene.
        seen_scenes: set[str] = set()
        for t in targets:
            if t["producer"] != P_SCENES:
                continue
            scene_id = t["ref"][1]
            if scene_id in seen_scenes:
                continue
            seen_scenes.add(scene_id)
            res = refresh_scene(conn, scene_id)
            report["scenes"].append({"scene_id": scene_id, **res})
            report["objects_evaluated"] += 1
            if res.get("refreshed"):
                report["objects_recomputed"] += 1
            report["objects_scanned"] += 1

        if plan.get("plan_id"):
            obj = repos_v4.get(
                conn,
                "objects",
                {"kind": REPAIR_PLAN_KIND, "object_id": plan["plan_id"]},
            )
            if obj is not None:
                rev_no = int(obj["current_revision"]) + 1
                doc = dict(plan)
                doc["state"] = "applied"
                doc["revision"] = rev_no
                doc["applied_us"] = (
                    int(now_us) if now_us is not None else _now_us()
                )
                repos_v4.update(
                    conn,
                    "objects",
                    {"current_revision": rev_no},
                    {"kind": REPAIR_PLAN_KIND, "object_id": plan["plan_id"]},
                )
                repos_v4.insert(
                    conn,
                    "object_revisions",
                    {
                        "kind": REPAIR_PLAN_KIND,
                        "object_id": plan["plan_id"],
                        "revision": rev_no,
                        "digest": _doc_digest(store, doc),
                        "recorded_from": store.next_event_us(),
                        "producer_ref": PRODUCER_ID,
                        "metadata_json": doc,
                    },
                )

    # ---- phase 2: producer dispatches in their own transactions --------
    seen_views: set[str] = set()
    for t in targets:
        kind, oid, _rev = t["ref"]
        if t["producer"] == P_SYNTHESIS:
            if oid in seen_views:
                continue  # one recompute covers every impacted revision
            seen_views.add(oid)
            res = _recompute_view(
                store,
                synthesizer,
                oid,
                caller=caller,
                purpose=purpose,
                producer_grant=producer_grant,
                request=(recompose_requests or {}).get(oid),
                now_us=now_us,
            )
            report["views"].append(res)
            report["objects_scanned"] += 1
            report["objects_evaluated"] += 1
            if res["status"] in ("repaired", "recomputed_as"):
                report["objects_recomputed"] += 1
            else:
                report["failures"].append(res)
        elif t["producer"] == P_PROFILES:
            # Entry-level refresh is internal to the service — the real
            # producer path is the scope refresh, run once below.
            continue
        elif t["producer"] in (P_CONSOLIDATION, P_SCENES):
            continue
        else:
            report["skipped"].append(t)

    if P_PROFILES in producers_needed:
        prof = profile_service.refresh(caller, scope_id, purpose=purpose)
        report["profiles"] = prof
        report["objects_evaluated"] += int(prof.get("scanned") or 0)
        report["objects_recomputed"] += (
            len(prof.get("withheld") or ())
            + len(prof.get("tombstoned") or ())
            + len(prof.get("revived") or ())
            + len(prof.get("expired") or ())
        )
        report["objects_scanned"] += int(prof.get("scanned") or 0)

    return report


def full_rebuild(
    store: Any,
    scope_id: str,
    *,
    caller: Optional[CallerV3] = None,
    purpose: Optional[str] = "admin",
    synthesizer: Any = None,
    profile_service: Any = None,
    producer_grant: Any = None,
    states: Any = None,
    modalities: Any = None,
    min_proof: Optional[int] = None,
    now_us: Optional[int] = None,
) -> dict[str, Any]:
    """The full-scope rebuild baseline — same producers, whole scope.

    Recomputes every recomputable object in the scope (all consolidation
    slots, every scene, every persisted view, profile refresh) so the
    recomputation count is directly comparable with ``apply_repair``
    (V45-04.04's ≥50% locality requirement).
    """
    if not isinstance(scope_id, str) or not scope_id:
        raise VerbatimError(ErrorCode.VALIDATION, "scope_id required")
    cons_kwargs: dict[str, Any] = {}
    if states is not None:
        cons_kwargs["states"] = states
    if modalities is not None:
        cons_kwargs["modalities"] = modalities
    if min_proof is not None:
        cons_kwargs["min_proof"] = min_proof

    report: dict[str, Any] = {
        "scope_id": scope_id,
        "objects_scanned": 0,
        "objects_evaluated": 0,
        "objects_recomputed": 0,
        "consolidation": None,
        "scenes": [],
        "views": [],
        "profiles": None,
        "failures": [],
    }
    with store.tx() as conn:
        # The whole-surface pass uses the SAME windowed producer with an
        # unbounded ``since_seq=0`` window — every slot touches, so the
        # evaluated/recomputed counters are instrument-identical to the
        # repair arm's and the counts compare honestly (V45-04.04).
        cons = consolidate_windowed(
            conn,
            scope_id,
            window=ConsolidationWindow(since_seq=0),
            **cons_kwargs,
        )
        report["consolidation"] = cons.as_dict()
        report["objects_evaluated"] += cons.candidates_seen
        report["objects_recomputed"] += (
            cons.observations_written + len(cons.retired or [])
        )
        report["objects_scanned"] += cons.slots_touched
        scenes = [
            r[0]
            for r in conn.execute(
                "SELECT episode_id FROM episodes"
                " WHERE scope_id = ? AND kind = ? AND recorded_until IS NULL",
                (scope_id, SCENE_KIND),
            ).fetchall()
        ]
        for scene_id in scenes:
            res = refresh_scene(conn, scene_id)
            report["scenes"].append({"scene_id": scene_id, **res})
            report["objects_evaluated"] += 1
            if res.get("refreshed"):
                report["objects_recomputed"] += 1
            report["objects_scanned"] += 1
        # Scanned surface = every object the rebuild touches by class.
        report["objects_scanned"] += int(
            conn.execute(
                "SELECT COUNT(*) FROM claims WHERE scope_id = ?", (scope_id,)
            ).fetchone()[0]
        )
        report["objects_scanned"] += int(
            conn.execute(
                "SELECT COUNT(*) FROM observations WHERE scope_id = ?",
                (scope_id,),
            ).fetchone()[0]
        )
        view_ids = [
            r["object_id"]
            for r in repos_v4.query(
                conn,
                "objects",
                {"kind": VIEW_OBJECT_KIND, "scope_id": scope_id},
            )
        ]
    for view_id in view_ids:
        if synthesizer is None or caller is None:
            report["failures"].append(
                {"view_id": view_id, "status": "failed",
                 "error_code": "context_incomplete",
                 "error": "no synthesizer/caller provided"}
            )
            continue
        res = _recompute_view(
            store,
            synthesizer,
            view_id,
            caller=caller,
            purpose=purpose,
            producer_grant=producer_grant,
            now_us=now_us,
        )
        report["views"].append(res)
        report["objects_scanned"] += 1
        report["objects_evaluated"] += 1
        if res["status"] in ("repaired", "recomputed_as"):
            report["objects_recomputed"] += 1
        else:
            report["failures"].append(res)

    if profile_service is not None and caller is not None:
        prof = profile_service.refresh(caller, scope_id, purpose=purpose)
        report["profiles"] = prof
        report["objects_evaluated"] += int(prof.get("scanned") or 0)
        report["objects_recomputed"] += (
            len(prof.get("withheld") or ())
            + len(prof.get("tombstoned") or ())
            + len(prof.get("revived") or ())
            + len(prof.get("expired") or ())
        )
        report["objects_scanned"] += int(prof.get("scanned") or 0)

    return report


__all__ = [
    "BRANCH_KIND",
    "PLAN_DOC",
    "PRODUCER_ID",
    "P_BRANCHES",
    "P_CONSOLIDATION",
    "P_PROFILES",
    "P_SCENES",
    "P_SYNTHESIS",
    "REPAIR_PLAN_KIND",
    "apply_repair",
    "full_rebuild",
    "impact_closure",
    "invalidate_dependents",
    "load_plan",
    "mark_held",
    "persist_plan",
    "plan_repair",
    "register_producer",
]
