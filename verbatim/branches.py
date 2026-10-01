"""Reversible memory branches (SPEC_V4_5 §07, V45-07.*).

A branch is a named, generation-scoped overlay of proposed corrections.
Branch state is durable in the parent ``objects``/``object_revisions``
registry (kind ``branch``, doc format ``branch/v1``, digest-bound like
derived views) — never a second store, never a live-table mutation. Ops
store *references* to parent claims plus proposed replacement values —
never canonical byte copies — so a branch physically cannot resurrect
bytes that were purged after its creation (D10).

Lifecycle::

    create → live ──submit──▶ live+pending_review ──apply──▶ applied
                 │                                            ▲
                 ├──held──(kernel invalidation)──rebase───────┘
                 └──abandon──▶ abandoned          purge──▶ tombstoned

Apply semantics (V45-07.03): the caller must hold ``review`` on the
scope (the operator gate), every op's review fence
(``expected_versions``) is re-verified inside the apply transaction,
and effects run through the real lifecycle machine — the same
``LifecycleMachine.apply`` + ``ReviewsRepo.resolve`` path
``_do_review_apply`` uses — inside one ``store.tx()`` with an
``operation_receipts`` row for idempotent replay. A parent that moved
fails ``STALE_PROPOSAL``; a parent that was purged fails
``NOT_FOUND_OR_UNAUTHORIZED``; nothing commits partially.

Invalidation forecasts (V45-07.04): ``submit`` records the exact
dependent set the apply is predicted to invalidate; ``apply`` re-walks
the graph and refuses when reality no longer matches the forecast — a
branch that cannot predict its blast radius cannot land.

Live erasure/revocation propagation (V45-07.02): purging or correcting
a pinned parent reaches the branch through the same ``dependency_edges``
the closure engine walks — purge suppresses or tombstones the branch
through ``privacy.deletion``'s registered kind, and
``note_invalidation`` flips open branches to ``held`` inside the
invalidating transaction.
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any, Iterable, Mapping, Optional

from .core.lifecycle import (
    LifecycleMachine,
    TransitionCommand,
    read_claim_head,
)
from .core.time import now_us as _now_us
from .core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    new_id,
    safe_json_loads,
)
from .core.types_v3 import Verb
from .core.types_v4 import LifecycleState
from .derivations import record_edge
from .governance import CallerV3, authorize
from .governance import epochs as _epochs
from .repair import impact_closure
from .storage import repos_v4
from .storage.repos import EventsRepo, ReviewsRepo


BRANCH_KIND = "branch"
BRANCH_DOC = "branch/v1"
PRODUCER_ID = "producer:verbatim.branches.v1"
_POLICY_VERSION = "branches.v1"

#: Branch lifecycle states (``objects.disposition`` mirrors the coarse
#: availability: live→active, held→held, applied/abandoned→archived,
#: tombstoned→erased).
STATE_LIVE = "live"
STATE_HELD = "held"
STATE_APPLIED = "applied"
STATE_ABANDONED = "abandoned"
STATE_TOMBSTONED = "tombstoned"

_DISPOSITION = {
    STATE_LIVE: LifecycleState.ACTIVE.value,
    STATE_HELD: LifecycleState.HELD.value,
    STATE_APPLIED: LifecycleState.ARCHIVED.value,
    STATE_ABANDONED: LifecycleState.ARCHIVED.value,
    STATE_TOMBSTONED: LifecycleState.ERASED.value,
}

#: Ops map to claim-lifecycle effects — the TRANSITIONS vocabulary minus
#: ``erase`` (erasure belongs to the purge path, never to a what-if).
ALLOWED_EFFECTS = frozenset({
    "admit", "dispute", "supersede", "resolve", "archive", "restore",
    "reject", "reverse_supersede",
})

_MAX_OPS = 64
#: Default retention on a branch snapshot (V45-13.02): 7 days.
_DEFAULT_RETENTION_US = 7 * 24 * 60 * 60 * 1_000_000


def _digest(store: Any, doc: Mapping[str, Any]) -> str:
    return "hmac-sha256:" + store.hmac(
        json_dumps(doc).encode("utf-8")
    ).hex()


def _req_digest(parts: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        json_dumps(parts).encode("utf-8")
    ).hexdigest()


class BranchService:
    """Branch lifecycle over one ``Store`` — composed from the parent
    objects registry, review queue, lifecycle machine, and impact walk.
    """

    def __init__(self, store: Any, *, producer_id: str = PRODUCER_ID) -> None:
        self._store = store
        self._producer_id = producer_id

    # ------------------------------------------------------------------
    # registration + doc IO
    # ------------------------------------------------------------------

    def register_producer(self, conn: sqlite3.Connection) -> str:
        """Idempotent producer-manifest registration (V45-13.01)."""
        row = repos_v4.get(
            conn, "producer_manifests", {"producer_id": self._producer_id}
        )
        if row is None:
            descriptor = {
                "producer_id": self._producer_id,
                "kind": "branch",
                "doc": BRANCH_DOC,
                "op_effects": sorted(ALLOWED_EFFECTS),
            }
            repos_v4.insert(
                conn,
                "producer_manifests",
                {
                    "producer_id": self._producer_id,
                    "kind": "branch",
                    "artifact_digest": _req_digest(
                        {"producer_code": "verbatim.branches", **descriptor}
                    ),
                    "rubric_digest": _req_digest({"rubric": "branches.v1"}),
                    "config_digest": _req_digest(descriptor),
                    "schema_version": 4,
                    "license_ref": None,
                    "health": "available",
                    "registered_us": _now_us(),
                },
            )
        return self._producer_id

    def _write_doc(
        self,
        conn: sqlite3.Connection,
        doc: dict[str, Any],
        *,
        disposition: Optional[str] = None,
    ) -> int:
        """Persist ``doc`` as the branch's next object revision."""
        branch_id = doc["branch_id"]
        obj = repos_v4.get(
            conn, "objects", {"kind": BRANCH_KIND, "object_id": branch_id}
        )
        if obj is None:
            revision = 1
            repos_v4.insert(
                conn,
                "objects",
                {
                    "object_id": branch_id,
                    "kind": BRANCH_KIND,
                    "scope_id": doc["scope_id"],
                    "current_revision": revision,
                    "disposition": (
                        disposition
                        or _DISPOSITION[doc["state"]]
                    ),
                    "created_event": self._store.next_event_us(),
                },
            )
        else:
            revision = int(obj["current_revision"]) + 1
            repos_v4.update(
                conn,
                "objects",
                {
                    "current_revision": revision,
                    "disposition": (
                        disposition or _DISPOSITION[doc["state"]]
                    ),
                },
                {"kind": BRANCH_KIND, "object_id": branch_id},
            )
        doc["revision"] = revision
        repos_v4.insert(
            conn,
            "object_revisions",
            {
                "kind": BRANCH_KIND,
                "object_id": branch_id,
                "revision": revision,
                "digest": _digest(self._store, doc),
                "recorded_from": self._store.next_event_us(),
                "producer_ref": self._producer_id,
                "metadata_json": doc,
            },
        )
        return revision

    def _doc(
        self,
        conn: sqlite3.Connection,
        branch_id: str,
    ) -> Optional[tuple[dict[str, Any], dict[str, Any]]]:
        """``(objects row, latest doc)`` with integrity verification."""
        obj = repos_v4.get(
            conn, "objects", {"kind": BRANCH_KIND, "object_id": branch_id}
        )
        if obj is None:
            return None
        rev = repos_v4.get(
            conn,
            "object_revisions",
            {
                "kind": BRANCH_KIND,
                "object_id": branch_id,
                "revision": int(obj["current_revision"]),
            },
        )
        if rev is None:
            return None
        raw = rev.get("metadata_json") or "{}"
        doc = safe_json_loads(raw) if isinstance(raw, str) else dict(raw)
        stored = rev.get("digest") or ""
        if stored and stored != _digest(self._store, doc):
            raise VerbatimError(
                ErrorCode.STORE_CORRUPT,
                f"branch {branch_id} doc fails integrity check",
            )
        return obj, doc

    def _get(self, branch_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._store.read() as conn:
            found = self._doc(conn, branch_id)
        if found is None:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "branch not found"
            )
        return found

    # ------------------------------------------------------------------
    # op normalization + parent re-verification
    # ------------------------------------------------------------------

    def _norm_ops(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        ops: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Validate + normalize proposed ops against the live snapshot.

        Every op's parent is pinned at its CURRENT head — the identical
        base snapshot the apply will later re-verify (V45-07.01). A op
        that names a missing, foreign, or suppressed parent fails loudly
        at proposal time.
        """
        out: list[dict[str, Any]] = []
        for i, op in enumerate(ops or ()):
            if not isinstance(op, Mapping):
                raise VerbatimError(
                    ErrorCode.VALIDATION, "branch ops must be mappings"
                )
            effect = op.get("effect")
            if effect not in ALLOWED_EFFECTS:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    f"branch op effect must be one of "
                    f"{sorted(ALLOWED_EFFECTS)}",
                )
            claim_id = op.get("claim_id")
            if not isinstance(claim_id, str) or not claim_id:
                raise VerbatimError(
                    ErrorCode.VALIDATION, "branch op needs a claim_id"
                )
            head = read_claim_head(conn, claim_id)
            if head is None or head.scope_id != scope_id:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
                    "branch parent claim not found",
                )
            if self._suppressed(conn, "claim", [claim_id]):
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
                    "branch parent claim is under suppression",
                )
            base_rev = op.get("base_revision")
            if base_rev is None:
                base_rev = int(head.revision)
            if int(base_rev) != int(head.revision):
                raise VerbatimError(
                    ErrorCode.STALE_PROPOSAL,
                    f"claim {claim_id} head is revision {head.revision}, "
                    f"op pinned {base_rev} — rebase first",
                )
            succ_id = op.get("successor_claim_id")
            succ_rev = None
            if succ_id is not None:
                succ = read_claim_head(conn, succ_id)
                if (
                    succ is None
                    or succ.scope_id != scope_id
                    or succ.claim_id == claim_id
                ):
                    raise VerbatimError(
                        ErrorCode.VALIDATION,
                        "successor must be a different live claim in scope",
                    )
                succ_rev = int(succ.revision)
            out.append(
                {
                    "op_id": op.get("op_id") or f"op:{new_id()}",
                    "effect": effect,
                    "claim_id": claim_id,
                    "base_revision": int(base_rev),
                    "successor_claim_id": succ_id,
                    "successor_expected_revision": succ_rev,
                    "new_object": op.get("new_object"),
                    "note": op.get("note") or "",
                    "status": "proposed",
                }
            )
        if len(out) > _MAX_OPS:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"branch ops exceed {_MAX_OPS}"
            )
        return out

    @staticmethod
    def _suppressed(
        conn: sqlite3.Connection, kind: str, ids: Iterable[str]
    ) -> set[str]:
        """Suppressed ids among ``ids`` — checked on the tx conn so
        uncommitted same-tx suppression is honored."""
        ids = list(dict.fromkeys(ids))
        if not ids:
            return set()
        ph = ",".join("?" for _ in ids)
        rows = conn.execute(
            "SELECT DISTINCT pt.object_id FROM purge_targets pt"
            " JOIN purges p ON p.purge_id = pt.purge_id"
            f" WHERE pt.object_kind = ? AND pt.object_id IN ({ph})"
            " AND p.state IN ('suppressed', 'purging', 'completed')",
            (kind, *ids),
        ).fetchall()
        return {str(r[0]) for r in rows}

    def _verify_parents(
        self, conn: sqlite3.Connection, doc: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """Re-verify every op's parents at apply/diff time.

        Returns per-op health: ``ok``, ``moved`` (head advanced),
        ``gone`` (purged/deleted), ``suppressed``. This is the re-check
        that makes a purged-parent apply impossible (D10) — the branch
        stores references, and a reference to erased bytes resolves to
        nothing restorable.
        """
        out: list[dict[str, Any]] = []
        suppressed = self._suppressed(
            conn, "claim",
            [o["claim_id"] for o in (doc.get("ops") or ())],
        )
        for op in doc.get("ops") or ():
            if op.get("status") == "dead":
                out.append({"op_id": op["op_id"], "health": "dead"})
                continue
            cid = op["claim_id"]
            head = read_claim_head(conn, cid)
            if head is None:
                out.append({"op_id": op["op_id"], "health": "gone",
                            "claim_id": cid})
                continue
            if cid in suppressed or head.state.value == "erased":
                out.append({"op_id": op["op_id"], "health": "suppressed",
                            "claim_id": cid})
                continue
            if int(head.revision) != int(op["base_revision"]):
                out.append({
                    "op_id": op["op_id"], "health": "moved",
                    "claim_id": cid,
                    "expected": op["base_revision"],
                    "current": int(head.revision),
                })
                continue
            succ = op.get("successor_claim_id")
            if succ is not None:
                shead = read_claim_head(conn, succ)
                if shead is None or succ in suppressed:
                    out.append({"op_id": op["op_id"], "health": "gone",
                                "claim_id": succ})
                    continue
                if int(shead.revision) != int(
                    op.get("successor_expected_revision") or -1
                ):
                    out.append({
                        "op_id": op["op_id"], "health": "moved",
                        "claim_id": succ,
                        "expected": op.get("successor_expected_revision"),
                        "current": int(shead.revision),
                    })
                    continue
            out.append({"op_id": op["op_id"], "health": "ok",
                        "claim_id": cid})
        return out

    def _parent_refs(
        self, doc: Mapping[str, Any]
    ) -> list[tuple[str, str, int]]:
        refs: list[tuple[str, str, int]] = []
        for op in doc.get("ops") or ():
            if op.get("status") == "dead":
                continue
            refs.append(("claim", op["claim_id"], int(op["base_revision"])))
            succ = op.get("successor_claim_id")
            if succ is not None:
                refs.append(
                    ("claim", succ, int(op["successor_expected_revision"]))
                )
        return refs

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def create(
        self,
        caller: CallerV3,
        scope_id: str,
        *,
        name: Optional[str] = None,
        ops: Iterable[Mapping[str, Any]] = (),
        purpose: Optional[str] = "admin",
        retention_us: Optional[int] = None,
        now_us: Optional[int] = None,
    ) -> dict[str, Any]:
        """Create a live branch — an isolated overlay, never a write to
        live claim state (V45-07.01/D09).

        The base snapshot pins the scope's authorization epoch, the
        projection generation, and every op-parent's head revision.
        ``dependency_edges`` rows bind the branch to those parents so
        kernel invalidation and deletion closure reach it exactly like
        any other dependent.
        """
        if not isinstance(caller, CallerV3):
            raise VerbatimError(
                ErrorCode.VALIDATION, "caller must be a bound CallerV3"
            )
        now = int(now_us) if now_us is not None else _now_us()
        branch_id = f"branch:{new_id()}"
        with self._store.tx() as conn:
            authorize(conn, caller, scope_id, Verb.REVIEW.value, purpose)
            authorize(conn, caller, scope_id, Verb.READ.value, purpose)
            self.register_producer(conn)
            norm = self._norm_ops(conn, scope_id, ops)
            doc = {
                "doc": BRANCH_DOC,
                "branch_id": branch_id,
                "name": name or branch_id,
                "scope_id": scope_id,
                "state": STATE_LIVE,
                "base": {
                    "epoch": _epochs.current_epoch(conn, scope_id),
                    "objects": [
                        list(r) for r in self._parent_refs({"ops": norm})
                    ],
                },
                "ops": norm,
                "forecast": None,
                "review_id": None,
                "created_by": caller.principal_id,
                "created_us": now,
                "retention_until_us": (
                    now + int(retention_us)
                    if retention_us is not None
                    else now + _DEFAULT_RETENTION_US
                ),
                "applied_us": None,
            }
            self._write_doc(conn, doc)
            op_id = f"op:{new_id()}"
            for seq, (kind, oid, rev) in enumerate(
                self._parent_refs({"ops": norm})
            ):
                repos_v4.insert(
                    conn,
                    "dependency_edges",
                    {
                        "child_kind": BRANCH_KIND,
                        "child_id": branch_id,
                        "child_revision": 1,
                        "parent_kind": kind,
                        "parent_id": oid,
                        "parent_revision": int(rev),
                        "role": "base",
                        "producer_id": self._producer_id,
                        "operation_id": op_id,
                        "seq": seq,
                    },
                )
                # Provenance edge too (V45-12.03): derivations is the v3
                # closure substrate — plan_closure/affected_by_purge reach
                # the branch here exactly like ClosureEngine reaches it
                # through dependency_edges.
                record_edge(
                    conn,
                    (BRANCH_KIND, branch_id, 1),
                    (kind, oid, int(rev)),
                    BRANCH_KIND,
                    self._producer_id,
                    scope_id,
                    seq,
                )
            EventsRepo(self._store).append(
                conn,
                scope_id,
                "branch_created",
                caller.principal_id,
                {"branch_id": branch_id, "ops": len(norm)},
                _POLICY_VERSION,
            )
        return {
            "branch_id": branch_id,
            "name": doc["name"],
            "state": STATE_LIVE,
            "ops": len(norm),
        }

    def get(
        self,
        caller: CallerV3,
        branch_id: str,
        *,
        purpose: Optional[str] = "admin",
    ) -> dict[str, Any]:
        """Latest branch doc (integrity-verified) plus object metadata.

        The doc carries proposed replacement text — scope content — so
        reads authorize like any other scope read (branch state never
        reaches normal recall surfaces; this IS the authorized metadata
        surface for it).
        """
        obj, doc = self._get(branch_id)
        with self._store.read() as conn:
            authorize(
                conn, caller, doc["scope_id"], Verb.READ.value, purpose
            )
        out = dict(doc)
        out["disposition"] = obj["disposition"]
        return out

    def list(
        self,
        caller: CallerV3,
        scope_id: Optional[str] = None,
        *,
        state: Optional[str] = None,
        purpose: Optional[str] = "admin",
    ) -> list[dict[str, Any]]:
        """Branch summaries — metadata only, no op payloads."""
        with self._store.read() as conn:
            where: dict[str, Any] = {"kind": BRANCH_KIND}
            if scope_id is not None:
                where["scope_id"] = scope_id
            rows = repos_v4.query(conn, "objects", where)
            out = []
            scope_ok: dict[str, bool] = {}
            for row in rows:
                sid = row["scope_id"]
                if sid not in scope_ok:
                    try:
                        authorize(
                            conn, caller, sid, Verb.READ.value, purpose
                        )
                        scope_ok[sid] = True
                    except VerbatimError:
                        scope_ok[sid] = False  # unseen scope → unseen branch
                if not scope_ok[sid]:
                    continue
                found = self._doc(conn, row["object_id"])
                if found is None:
                    continue
                _obj, doc = found
                if state is not None and doc["state"] != state:
                    continue
                out.append(
                    {
                        "branch_id": doc["branch_id"],
                        "name": doc["name"],
                        "scope_id": doc["scope_id"],
                        "state": doc["state"],
                        "ops": len(doc["ops"]),
                        "revision": doc["revision"],
                        "created_us": doc["created_us"],
                        "retention_until_us": doc["retention_until_us"],
                    }
                )
        return sorted(out, key=lambda b: b["created_us"])

    def add_op(
        self,
        caller: CallerV3,
        branch_id: str,
        ops: Iterable[Mapping[str, Any]],
        *,
        purpose: Optional[str] = "admin",
    ) -> dict[str, Any]:
        """Append proposed ops to a live branch (new doc revision)."""
        with self._store.tx() as conn:
            found = self._doc(conn, branch_id)
            if found is None:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "branch not found"
                )
            _obj, doc = found
            authorize(
                conn, caller, doc["scope_id"], Verb.REVIEW.value, purpose
            )
            if doc["state"] != STATE_LIVE or doc.get("review_id"):
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    "ops can only be added to a live, un-submitted branch",
                )
            norm = self._norm_ops(conn, doc["scope_id"], ops)
            if len(doc["ops"]) + len(norm) > _MAX_OPS:
                raise VerbatimError(
                    ErrorCode.VALIDATION, f"branch ops exceed {_MAX_OPS}"
                )
            doc["ops"].extend(norm)
            doc["base"]["objects"] = [
                list(r) for r in self._parent_refs(doc)
            ]
            revision = self._write_doc(conn, doc)
            op_id = f"op:{new_id()}"
            for seq, (kind, oid, rev) in enumerate(
                [("claim", o["claim_id"], o["base_revision"]) for o in norm]
                + [
                    ("claim", o["successor_claim_id"],
                     o["successor_expected_revision"])
                    for o in norm if o.get("successor_claim_id")
                ]
            ):
                repos_v4.insert(
                    conn,
                    "dependency_edges",
                    {
                        "child_kind": BRANCH_KIND,
                        "child_id": branch_id,
                        "child_revision": revision,
                        "parent_kind": kind,
                        "parent_id": oid,
                        "parent_revision": int(rev),
                        "role": "base",
                        "producer_id": self._producer_id,
                        "operation_id": op_id,
                        "seq": seq,
                    },
                )
                record_edge(
                    conn,
                    (BRANCH_KIND, branch_id, revision),
                    (kind, oid, int(rev)),
                    BRANCH_KIND,
                    self._producer_id,
                    doc["scope_id"],
                    seq,
                )
        return {"branch_id": branch_id, "ops": len(doc["ops"]),
                "revision": revision}

    def submit(
        self,
        caller: CallerV3,
        branch_id: str,
        *,
        purpose: Optional[str] = "review",
    ) -> dict[str, Any]:
        """Submit the branch for operator review.

        Records the invalidation forecast (V45-07.04) and opens ONE
        review row carrying ``expected_versions`` for every pinned
        parent — the same fenced-proposal machinery ``propose_edit``
        uses. The branch stays isolated; the review is the operator's
        apply gate.
        """
        with self._store.tx() as conn:
            found = self._doc(conn, branch_id)
            if found is None:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "branch not found"
                )
            _obj, doc = found
            authorize(
                conn, caller, doc["scope_id"], Verb.REVIEW.value, purpose
            )
            if (
                doc["state"] != STATE_LIVE
                or _obj["disposition"] == LifecycleState.HELD.value
            ):
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    f"branch is {doc['state']}"
                    + (
                        " (held by live invalidation — rebase first)"
                        if _obj["disposition"]
                        == LifecycleState.HELD.value
                        else ""
                    )
                    + "; only live branches submit",
                )
            if doc.get("review_id"):
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    "branch already submitted for review",
                )
            if not doc["ops"]:
                raise VerbatimError(
                    ErrorCode.VALIDATION, "a branch with no ops cannot submit"
                )
            if doc["retention_until_us"] < _now_us():
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    "branch retention elapsed — snapshot is not applicable",
                )
            health = self._verify_parents(conn, doc)
            bad = [h for h in health if h["health"] != "ok"]
            if bad:
                raise VerbatimError(
                    ErrorCode.STALE_PROPOSAL,
                    "branch parents moved or vanished — rebase first: "
                    f"{bad[:8]}",
                )
            parents = self._parent_refs(doc)
            forecast = sorted(
                impact_closure(conn, parents), key=lambda r: (r[0], r[1], r[2])
            )
            expected = {
                op["claim_id"]: int(op["base_revision"])
                for op in doc["ops"]
            }
            for op in doc["ops"]:
                succ = op.get("successor_claim_id")
                if succ is not None:
                    expected[succ] = int(op["successor_expected_revision"])
            review_id = ReviewsRepo(self._store).create(
                conn,
                doc["scope_id"],
                {
                    "effect": "apply_branch",
                    "branch_id": branch_id,
                    "ops": [o["op_id"] for o in doc["ops"]],
                    "reason": f"branch {doc['name']} apply",
                },
                expected,
            )
            doc["forecast"] = [list(r) for r in forecast]
            doc["review_id"] = review_id
            for op in doc["ops"]:
                op["status"] = "reviewed"
            revision = self._write_doc(conn, doc)
            EventsRepo(self._store).append(
                conn,
                doc["scope_id"],
                "branch_submitted",
                caller.principal_id,
                {"branch_id": branch_id, "review_id": review_id,
                 "forecast": len(forecast)},
                _POLICY_VERSION,
            )
        return {
            "branch_id": branch_id,
            "review_id": review_id,
            "forecast": doc["forecast"],
            "revision": revision,
        }

    def apply(
        self,
        caller: CallerV3,
        branch_id: str,
        *,
        purpose: Optional[str] = "review",
        kernel: Any = None,
        synthesizer: Any = None,
        actor_id: Optional[str] = None,
        now_us: Optional[int] = None,
    ) -> dict[str, Any]:
        """Reviewed, fenced, atomic apply through the real lifecycle.

        Gates, in order, all inside one ``store.tx()``:
        operator ``review`` authorization → branch live+submitted →
        review row still open → forecast re-walk equals the recorded
        forecast (V45-07.04) → every parent re-verified at its pinned
        revision and not suppressed/purged (D10) → per-op
        ``LifecycleMachine.apply`` → review resolved → branch doc marked
        ``applied`` → ``operation_receipts`` idempotency row.

        Any gate failure aborts the transaction — a failed apply commits
        nothing, so no partial view of the branch ever exists.
        """
        if not isinstance(caller, CallerV3):
            raise VerbatimError(
                ErrorCode.VALIDATION, "caller must be a bound CallerV3"
            )
        now = int(now_us) if now_us is not None else _now_us()
        actor = actor_id or caller.principal_id
        machine = LifecycleMachine(self._store)
        with self._store.tx() as conn:
            found = self._doc(conn, branch_id)
            if found is None:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "branch not found"
                )
            _obj, doc = found
            authorize(
                conn, caller, doc["scope_id"], Verb.REVIEW.value, purpose
            )
            # Idempotency first: a stored receipt replays honestly even
            # after the state advanced (the receipt and the applied
            # mark commit in the same transaction, so they cannot
            # diverge). The same key with different input is a caller
            # conflict (V4-09.05).
            op_id = f"branch-apply:{branch_id}"
            input_digest = _req_digest(
                {"branch": branch_id, "ops": [
                    (o["op_id"], o["claim_id"], o["base_revision"])
                    for o in (doc.get("ops") or [])
                ]}
            )
            prior = repos_v4.get(
                conn, "operation_receipts", {"operation_id": op_id}
            )
            if prior is not None:
                if prior["input_digest"] != input_digest:
                    raise VerbatimError(
                        ErrorCode.OPERATION_CONFLICT,
                        "branch-apply receipt exists with different input",
                    )
                return {
                    "branch_id": branch_id,
                    "state": doc["state"],
                    "replayed": True,
                    "receipt": prior,
                }
            if (
                doc["state"] == STATE_HELD
                or _obj["disposition"] == LifecycleState.HELD.value
            ):
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    "branch is held by live invalidation — rebase first",
                )
            if doc["state"] != STATE_LIVE:
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    f"branch is {doc['state']}; cannot apply",
                )
            review_id = doc.get("review_id")
            if not review_id:
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    "operator review required before apply (V45-07.03)",
                )
            rrow = conn.execute(
                "SELECT state, expected_versions_json FROM reviews"
                " WHERE review_id = ?",
                (review_id,),
            ).fetchone()
            if rrow is None or rrow[0] != "open":
                raise VerbatimError(
                    ErrorCode.STALE_PROPOSAL,
                    "branch review is not open",
                )
            if doc["retention_until_us"] < now:
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    "branch retention elapsed — snapshot is not applicable",
                )

            # V45-07.04: the apply-time closure must equal the recorded
            # forecast — a branch that cannot predict affected objects
            # cannot be applied.
            parents = self._parent_refs(doc)
            actual = sorted(
                impact_closure(conn, parents), key=lambda r: (r[0], r[1], r[2])
            )
            forecast_pairs = {
                (r[0], r[1]) for r in (doc.get("forecast") or [])
            }
            actual_pairs = {(r[0], r[1]) for r in actual}
            if actual_pairs != forecast_pairs:
                raise VerbatimError(
                    ErrorCode.CONTEXT_INCOMPLETE,
                    "invalidation forecast no longer matches the apply "
                    "plan — rebase the branch first (V45-07.04): "
                    f"forecast-only {sorted(forecast_pairs - actual_pairs)[:6]}"
                    f" actual-only {sorted(actual_pairs - forecast_pairs)[:6]}",
                )

            # D10: re-verify every parent inside the apply transaction —
            # purged or moved parents abort the whole apply.
            health = self._verify_parents(conn, doc)
            failed = [h for h in health if h["health"] != "ok"]
            if failed:
                code = (
                    ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
                    if any(
                        h["health"] in ("gone", "suppressed", "dead")
                        for h in failed
                    )
                    else ErrorCode.STALE_PROPOSAL
                )
                raise VerbatimError(
                    code,
                    "branch parents purged or moved — the branch cannot "
                    f"restore what no longer resolves: {failed[:8]}",
                )

            applied: list[dict[str, Any]] = []
            last_seq = 0
            for op in doc["ops"]:
                cmd = TransitionCommand(
                    claim_id=op["claim_id"],
                    expected_revision=int(op["base_revision"]),
                    effect=op["effect"],
                    actor_id=actor,
                    reason=(
                        op.get("note")
                        or f"branch {doc['name']} apply"
                    ),
                    successor_claim_id=op.get("successor_claim_id"),
                )
                seq = machine.apply(cmd, conn)
                last_seq = seq
                op["status"] = "applied"
                applied.append(
                    {
                        "op_id": op["op_id"],
                        "claim_id": op["claim_id"],
                        "effect": op["effect"],
                        "event_seq": seq,
                    }
                )
            ev_seq = EventsRepo(self._store).append(
                conn,
                doc["scope_id"],
                "branch_applied",
                actor,
                {"branch_id": branch_id, "ops": len(applied)},
                _POLICY_VERSION,
            )
            ReviewsRepo(self._store).resolve(
                conn, review_id, "approved", max(last_seq, ev_seq)
            )
            doc["state"] = STATE_APPLIED
            doc["applied_us"] = now
            doc["applied_by"] = actor
            doc["applied_ops"] = applied
            revision = self._write_doc(conn, doc)
            repos_v4.insert(
                conn,
                "operation_receipts",
                {
                    "operation_id": op_id,
                    "scope_id": doc["scope_id"],
                    "input_digest": input_digest,
                    "result_ref": branch_id,
                    "effects_applied": len(applied),
                    "jobs_json": [],
                    "applied_seq": max(last_seq, ev_seq),
                    "created_us": now,
                },
            )
            # Optional eager invalidation of the forecast set through the
            # real kernel — same transaction, so dependents flip to held
            # atomically with the apply (V45-04.02 semantics).
            marked = None
            if kernel is not None:
                report = kernel.invalidate(
                    conn,
                    {
                        "kind": "correction",
                        "scope_ids": [doc["scope_id"]],
                        "object_refs": [
                            list(r) for r in self._parent_refs(doc)
                        ],
                    },
                    now_us=now,
                )
                marked = {"affected": len(report.affected)}
                if synthesizer is not None:
                    marked["views_held"] = synthesizer.note_invalidation(
                        conn, report
                    )
        return {
            "branch_id": branch_id,
            "state": STATE_APPLIED,
            "applied": applied,
            "revision": revision,
            "invalidated": marked,
            "replayed": False,
        }

    def abandon(
        self,
        caller: CallerV3,
        branch_id: str,
        *,
        purpose: Optional[str] = "admin",
        reason: Optional[str] = None,
    ) -> dict[str, Any]:
        """Operator-driven retirement — never applied, never deleted."""
        with self._store.tx() as conn:
            found = self._doc(conn, branch_id)
            if found is None:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "branch not found"
                )
            _obj, doc = found
            authorize(
                conn, caller, doc["scope_id"], Verb.REVIEW.value, purpose
            )
            if doc["state"] not in (STATE_LIVE, STATE_HELD):
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    f"branch is {doc['state']}; cannot abandon",
                )
            doc["state"] = STATE_ABANDONED
            doc["abandoned_by"] = caller.principal_id
            doc["abandon_reason"] = reason
            revision = self._write_doc(conn, doc)
            EventsRepo(self._store).append(
                conn,
                doc["scope_id"],
                "branch_abandoned",
                caller.principal_id,
                {"branch_id": branch_id, "reason": reason},
                _POLICY_VERSION,
            )
        return {"branch_id": branch_id, "state": STATE_ABANDONED,
                "revision": revision}

    def rebase(
        self,
        caller: CallerV3,
        branch_id: str,
        *,
        purpose: Optional[str] = "admin",
    ) -> dict[str, Any]:
        """Re-pin a live/held branch to current heads.

        Ops whose parents moved re-pin their base revision; ops whose
        parents are gone or suppressed become ``dead`` — permanently
        inapplicable, recorded honestly rather than silently dropped.
        The forecast and pending review are cleared: a rebased branch
        must be re-submitted before it can apply.
        """
        with self._store.tx() as conn:
            found = self._doc(conn, branch_id)
            if found is None:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "branch not found"
                )
            _obj, doc = found
            authorize(
                conn, caller, doc["scope_id"], Verb.REVIEW.value, purpose
            )
            if doc["state"] not in (STATE_LIVE, STATE_HELD):
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    f"branch is {doc['state']}; cannot rebase",
                )
            suppressed = self._suppressed(
                conn, "claim", [o["claim_id"] for o in doc["ops"]]
            )
            dead_ops: list[dict[str, Any]] = []
            moved = 0
            live_ops: list[dict[str, Any]] = []
            for op in doc["ops"]:
                if op.get("status") == "dead":
                    dead_ops.append(op)
                    continue
                cid = op["claim_id"]
                head = read_claim_head(conn, cid)
                dead = (
                    head is None
                    or cid in suppressed
                    or head.state.value == "erased"
                )
                if not dead:
                    succ = op.get("successor_claim_id")
                    if succ is not None:
                        shead = read_claim_head(conn, succ)
                        if shead is None or succ in suppressed:
                            dead = True
                        else:
                            op["successor_expected_revision"] = int(
                                shead.revision
                            )
                if dead:
                    op["status"] = "dead"
                    dead_ops.append(op)
                    continue
                if int(head.revision) != int(op["base_revision"]):
                    op["base_revision"] = int(head.revision)
                    moved += 1
                op["status"] = "proposed"
                live_ops.append(op)
            # Dead ops are removed from the applicable set but retained in
            # the doc's audit record — a purged parent can never produce an
            # applicable op again (D10), and the removal is visible.
            doc["dead_ops"] = (doc.get("dead_ops") or []) + dead_ops
            doc["ops"] = live_ops
            doc["base"]["objects"] = [
                list(r) for r in self._parent_refs(doc)
            ]
            doc["base"]["epoch"] = _epochs.current_epoch(
                conn, doc["scope_id"]
            )
            doc["forecast"] = None
            # A pending review was opened against the pre-rebase pins —
            # void it honestly so the operator queue never carries a
            # proposal whose fenced versions no longer describe the branch.
            if doc.get("review_id"):
                rrow = conn.execute(
                    "SELECT state FROM reviews WHERE review_id = ?",
                    (doc["review_id"],),
                ).fetchone()
                if rrow is not None and rrow[0] == "open":
                    ReviewsRepo(self._store).resolve(
                        conn, doc["review_id"], "rejected",
                        int(
                            conn.execute(
                                "SELECT COALESCE(MAX(event_seq), 0)"
                                " FROM events"
                            ).fetchone()[0]
                        ),
                    )
            doc["review_id"] = None
            doc["state"] = STATE_LIVE
            revision = self._write_doc(
                conn, doc, disposition=LifecycleState.ACTIVE.value
            )
            EventsRepo(self._store).append(
                conn,
                doc["scope_id"],
                "branch_rebased",
                caller.principal_id,
                {"branch_id": branch_id, "moved": moved,
                 "dead": len(dead_ops)},
                _POLICY_VERSION,
            )
        return {
            "branch_id": branch_id,
            "state": STATE_LIVE,
            "moved": moved,
            "dead": len(dead_ops),
            "revision": revision,
        }

    def diff(
        self,
        caller: CallerV3,
        branch_id: str,
        *,
        purpose: Optional[str] = "admin",
    ) -> dict[str, Any]:
        """Branch-vs-live health per op — the operator's review surface."""
        _obj, doc = self._get(branch_id)
        with self._store.read() as conn:
            authorize(
                conn, caller, doc["scope_id"], Verb.READ.value, purpose
            )
            health = self._verify_parents(conn, doc)
        return {
            "branch_id": branch_id,
            "state": doc["state"],
            "ops": [
                {**op, "health": h["health"]}
                for op, h in zip(doc.get("ops") or (), health)
            ],
        }

    # ------------------------------------------------------------------
    # invalidation + closure integration (V45-07.02, V45-12.03)
    # ------------------------------------------------------------------

    def note_invalidation(
        self, conn: sqlite3.Connection, report: Any
    ) -> int:
        """Flip live branches named by an invalidation report to ``held``.

        Same contract as ``Synthesizer.note_invalidation``: call inside
        the same transaction as ``kernel.invalidate`` so branch
        suppression commits atomically with the epoch bump. A held
        branch must ``rebase`` before it can submit or apply.
        """
        affected = getattr(report, "affected", None)
        if affected is None and isinstance(report, Mapping):
            affected = report.get("affected")
        marked = 0
        for item in affected or ():
            try:
                kind, oid, _rev = item
            except (TypeError, ValueError):
                continue
            if kind != BRANCH_KIND:
                continue
            found = self._doc(conn, oid)
            if found is None:
                continue
            _obj, doc = found
            if doc["state"] != STATE_LIVE:
                continue
            doc["state"] = STATE_HELD
            doc["held_us"] = _now_us()
            self._write_doc(conn, doc)
            marked += 1
        return marked


__all__ = [
    "ALLOWED_EFFECTS",
    "BRANCH_DOC",
    "BRANCH_KIND",
    "BranchService",
    "PRODUCER_ID",
    "STATE_ABANDONED",
    "STATE_APPLIED",
    "STATE_HELD",
    "STATE_LIVE",
    "STATE_TOMBSTONED",
]
