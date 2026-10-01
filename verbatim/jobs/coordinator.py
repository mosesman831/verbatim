"""Fenced effect-plan coordinator (SPEC_V4 §09, §42).

One path applies authoritative domain effects. Policy and model components
compute an ``EffectPlan`` *outside* any transaction; ``apply_plan`` re-verifies
the lease fencing token, authorization epochs, expected revisions, and input
digests inside the same transaction that writes the effects (V4-09.02).

Guarantees:

- Atomicity (V4-09.03): domain revisions, provenance edges, invalidation
  events, receipts, outbox obligations, and job completion commit together.
- Fencing (V4-09.04, V4-42.01): a stale or cancelled worker — lease lost,
  revoked grant, advanced epoch — commits nothing; the plan is discarded.
- Idempotency (V4-09.05): replaying an operation with identical input returns
  its prior receipt; reusing the key with different input fails
  OPERATION_CONFLICT.
- Deadline honesty (V4-09.09): no transaction stays open waiting on a model,
  socket, or user decision — plans arrive fully computed.
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any, Optional

from ..core.time import now_us
from ..core.types import ErrorCode, VerbatimError, json_dumps, new_id, safe_json_loads
from ..governance import epochs as _epochs
from ..core.types_v4 import (
    Effect,
    EffectKind,
    EffectPlan,
    JobRequest,
    OperationReceipt,
)
from ..storage import repos_v4

# Tables the coordinator may write through effects. Effect.table is checked
# against this map — a plan cannot reach an unregistered table (V4-19.09).
_EFFECT_TABLES: dict[EffectKind, frozenset[str]] = {
    EffectKind.INSERT_OBJECT: frozenset({"objects", "object_revisions"}),
    EffectKind.UPDATE_LIFECYCLE: frozenset({"objects"}),
    EffectKind.INSERT_EDGE: frozenset({"dependency_edges", "derivations"}),
    EffectKind.INVALIDATE: frozenset({"objects", "object_revisions"}),
    EffectKind.RECORD_RECEIPT: frozenset({"operation_receipts"}),
    EffectKind.PUBLISH_INDEX: frozenset({"index_generations"}),
    EffectKind.WRITE_PAYLOAD: frozenset({"sources", "spans"}),
    EffectKind.ENQUEUE: frozenset(),  # handled via follow_ups, not a table
}


def _input_digest(plan: EffectPlan) -> str:
    h = hashlib.sha256()
    for d in sorted(plan.input_digests):
        h.update(d.encode("utf-8"))
    return h.hexdigest()


class Coordinator:
    """Applies EffectPlans inside fenced transactions on ``store``."""

    def __init__(self, store: Any) -> None:
        self.store = store

    # -- public ------------------------------------------------------------

    def apply_plan(
        self,
        plan: EffectPlan,
        *,
        job_id: Optional[str] = None,
        expected_lease: Optional[int] = None,
        verify_epochs: bool = True,
    ) -> OperationReceipt:
        """Verify, then atomically apply, ``plan``.

        ``job_id`` + ``expected_lease`` fence a worker-driven plan: inside the
        transaction the job must still be live and its generation must equal
        the token the worker was leased under — otherwise the worker is stale
        and commits nothing (CANCELLED / LEASE_LOST).
        """
        with self.store.tx() as conn:
            receipt = self._check_idempotent(conn, plan)
            if receipt is not None:
                return receipt
            if job_id is not None:
                self._check_lease(conn, job_id, expected_lease)
            if verify_epochs:
                self._check_epochs(conn, plan)
            self._check_expected_revisions(conn, plan)
            applied = self._apply_effects(conn, plan)
            jobs = self._enqueue_followups(conn, plan)
            receipt = self._record_receipt(conn, plan, applied, jobs)
            return receipt

    # -- guards (all inside the apply transaction) -------------------------

    def _check_idempotent(
        self, conn: sqlite3.Connection, plan: EffectPlan
    ) -> Optional[OperationReceipt]:
        row = repos_v4.get(
            conn, "operation_receipts", {"operation_id": plan.operation_id}
        )
        if row is None:
            return None
        digest = _input_digest(plan)
        if row["input_digest"] != digest:
            # Same operation key, different input — a caller bug, not a retry
            # (V4-09.05).
            raise VerbatimError(
                ErrorCode.OPERATION_CONFLICT,
                f"operation {plan.operation_id} already applied with different input",
            )
        return OperationReceipt(
            operation_id=row["operation_id"],
            scope_id=row["scope_id"],
            input_digest=row["input_digest"],
            applied_seq=row["applied_seq"],
            result_ref=row["result_ref"],
            effects_applied=row["effects_applied"],
            jobs_enqueued=tuple(safe_json_loads(row.get("jobs_json")) or ()),
            created_us=row["created_us"],
        )

    def _check_lease(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        expected_lease: Optional[int],
    ) -> None:
        """The job's fencing token must still be the live lease."""
        row = conn.execute(
            "SELECT generation, state FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise VerbatimError(
                ErrorCode.LEASE_LOST, f"job {job_id} no longer exists"
            )
        generation, state = row[0], row[1]
        if state in ("cancelled", "failed", "succeeded"):
            raise VerbatimError(
                ErrorCode.CANCELLED,
                f"job already terminal ({state}); stale worker cannot commit",
            )
        if expected_lease is not None and generation != expected_lease:
            raise VerbatimError(
                ErrorCode.LEASE_LOST,
                "lease fencing token stale — a replacement worker owns the job",
            )

    def _check_epochs(self, conn: sqlite3.Connection, plan: EffectPlan) -> None:
        """Every pinned scope epoch must still be current (V4-11.07).

        The pinned value is the scope's authorization epoch — the same
        ``authz_revision`` counter ``EligibilityLease.epoch_vector`` pins
        at ``resolve_access`` and ``governance.revoke_grant`` bumps. A
        revocation between plan computation and application moves the
        epoch, so the stale plan fences here and publishes nothing.
        """
        for scope_id, epoch in plan.epoch_vector.items():
            if scope_id == "policy_epoch":
                continue  # global epoch checked below
            # An absent scope row reads as epoch 0 — a caller that pinned
            # a nonzero value observed a state this store never reached:
            # stale.
            current = _epochs.current_epoch(conn, scope_id)
            if current != epoch:
                raise VerbatimError(
                    ErrorCode.STALE_EPOCH,
                    f"scope {scope_id} epoch is {current}, pinned {epoch}",
                )
        # Global policy epoch participates when pinned under the store key.
        if "policy_epoch" in plan.epoch_vector:
            current = self.store.policy_epoch()
            if current != plan.epoch_vector["policy_epoch"]:
                raise VerbatimError(
                    ErrorCode.STALE_EPOCH,
                    "policy epoch advanced since plan computation",
                )

    def _check_expected_revisions(
        self, conn: sqlite3.Connection, plan: EffectPlan
    ) -> None:
        for object_id, expected in plan.expected_revisions.items():
            row = conn.execute(
                "SELECT current_revision FROM objects WHERE object_id = ?",
                (object_id,),
            ).fetchone()
            if row is None:
                raise VerbatimError(
                    ErrorCode.STALE_DEPENDENCY,
                    f"object {object_id} expected by plan does not exist",
                )
            if row[0] != expected:
                raise VerbatimError(
                    ErrorCode.STALE_DEPENDENCY,
                    f"object {object_id} at revision {row[0]}, expected {expected}",
                )

    # -- application -------------------------------------------------------

    def _apply_effects(
        self, conn: sqlite3.Connection, plan: EffectPlan
    ) -> int:
        applied = 0
        for effect in plan.effects:
            allowed = _EFFECT_TABLES.get(effect.kind)
            if allowed is None:
                raise VerbatimError(
                    ErrorCode.VALIDATION, f"unhandled effect kind {effect.kind}"
                )
            if effect.table and effect.table not in allowed:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    f"effect {effect.kind.value} cannot write {effect.table}",
                )
            self._apply_effect(conn, plan, effect)
            applied += 1
        return applied

    def _apply_effect(
        self, conn: sqlite3.Connection, plan: EffectPlan, effect: Effect
    ) -> None:
        payload = dict(effect.payload)
        if effect.kind is EffectKind.INSERT_OBJECT:
            if "object_id" in payload and "revision" in payload:
                repos_v4.insert(
                    conn,
                    "object_revisions",
                    {
                        "kind": effect.table or payload.pop("kind", "object"),
                        "object_id": payload["object_id"],
                        "revision": payload["revision"],
                        "digest": payload.get("digest", ""),
                        "recorded_from": payload.get("recorded_from", 0),
                        "producer_ref": plan.producer_id,
                        "metadata_json": payload.get("metadata_json", {}),
                    },
                )
                return
            repos_v4.insert(conn, "objects", payload)
            return
        if effect.kind is EffectKind.INSERT_EDGE:
            payload.setdefault("operation_id", plan.operation_id)
            repos_v4.insert(conn, "dependency_edges", payload)
            return
        if effect.kind is EffectKind.UPDATE_LIFECYCLE:
            where = payload.pop("_where", None) or {
                "object_id": payload.get("object_id"),
            }
            if "kind" not in where and "kind" in payload:
                where["kind"] = payload.pop("kind")
            repos_v4.update(conn, "objects", payload, where)
            return
        if effect.kind is EffectKind.INVALIDATE:
            self.store.bump_generation(conn)
            return
        if effect.kind is EffectKind.RECORD_RECEIPT:
            repos_v4.insert(conn, "operation_receipts", payload)
            return
        if effect.kind is EffectKind.PUBLISH_INDEX:
            from ..storage import repos_v3

            repos_v3.insert(conn, "index_generations", payload)
            return
        if effect.kind is EffectKind.WRITE_PAYLOAD:
            # Payload writes go through the repos allowlist of the declared
            # table — callers build a fully-formed row dict.
            from ..storage import repos

            repos.insert(conn, effect.table, payload)
            return
        raise VerbatimError(
            ErrorCode.VALIDATION, f"effect kind {effect.kind} has no handler"
        )

    def _enqueue_followups(
        self, conn: sqlite3.Connection, plan: EffectPlan
    ) -> tuple[str, ...]:
        if not plan.follow_ups:
            return ()
        enqueued: list[str] = []
        for req in plan.follow_ups:
            jid = self._enqueue_in_tx(conn, plan, req)
            enqueued.append(jid)
        return tuple(enqueued)

    def _enqueue_in_tx(
        self,
        conn: sqlite3.Connection,
        plan: EffectPlan,
        req: JobRequest,
    ) -> str:
        """Insert the job row inside the caller's transaction (V4-09.03).

        The queue's own enqueue opens its own tx; inside apply we write the
        row directly so the obligation commits atomically with the effects.
        """
        import hashlib as _hashlib

        from .queue import _LANE_DEFAULTS, LANES

        lane = req.lane if req.lane in LANES else _LANE_DEFAULTS.get(req.kind, "ordinary")
        job_id = new_id()
        dedup = req.dedup_key or f"{plan.operation_id}:{req.kind}"
        dedup_bytes = _hashlib.sha256(dedup.encode("utf-8")).digest()
        conn.execute(
            "INSERT INTO jobs (job_id, scope_id, kind, state, dedup_key,"
            " input_refs_json, policy_epoch, attempts, not_before_us,"
            " deadline_us, generation, operation_key, lane)"
            " VALUES (?,?,?,?,?,?,?,0,?,?,?,?,?)",
            (
                job_id,
                plan.scope_id,
                req.kind,
                "queued",
                dedup_bytes,
                json_dumps(req.input_refs),
                plan.epoch_vector.get("policy_epoch", self.store.policy_epoch()),
                req.not_before_us,
                plan.deadline_us or None,
                plan.lease_token,
                plan.operation_id,
                lane,
            ),
        )
        return job_id

    def _record_receipt(
        self,
        conn: sqlite3.Connection,
        plan: EffectPlan,
        applied: int,
        jobs: tuple[str, ...],
    ) -> OperationReceipt:
        seq = self.store.next_event_us()
        now = now_us()
        digest = _input_digest(plan)
        repos_v4.insert(
            conn,
            "operation_receipts",
            {
                "operation_id": plan.operation_id,
                "scope_id": plan.scope_id,
                "input_digest": digest,
                "result_ref": None,
                "effects_applied": applied,
                "jobs_json": list(jobs),
                "applied_seq": seq,
                "created_us": now,
            },
        )
        return OperationReceipt(
            operation_id=plan.operation_id,
            scope_id=plan.scope_id,
            input_digest=digest,
            applied_seq=seq,
            result_ref=None,
            effects_applied=applied,
            jobs_enqueued=jobs,
            created_us=now,
        )
