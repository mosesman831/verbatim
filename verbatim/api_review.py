"""Engine sibling — transitions, effect proposals, and review effects
(V3-06.01).

Owns the review/lifecycle topic: transition proposals, proposal
application with the versioned-fence + operations-ledger contract
(V2-19/20/39), dispute conflict edges, v2-table revision effects, and
same-transaction active-head indexing. Methods and module helpers are
verbatim moves from the former ``api.py`` god file.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from .core.identity import can_write, scope_key
from .core.types import (
    CallerContext,
    EffectProposal,
    ErrorCode,
    GrantKind,
    Lifecycle,
    Scope,
    TransitionCommand,
    VerbatimError,
    json_dumps,
    safe_json_loads,
)

#: Authority required per effect (SPEC_V2 §09.10, §20): destructive effects
#: carry their own grants — restore needs SUPPRESS, erase needs PURGE.
_EFFECT_GRANTS: dict[str, GrantKind] = {
    "admit": GrantKind.RESOLVE,
    "reject": GrantKind.RESOLVE,
    "dispute": GrantKind.RESOLVE,
    "resolve": GrantKind.RESOLVE,
    "correct": GrantKind.RESOLVE,
    "supersede": GrantKind.RESOLVE,
    "archive": GrantKind.RESOLVE,
    "reconsider": GrantKind.RESOLVE,
    "reverse_supersede": GrantKind.RESOLVE,
    "restore": GrantKind.SUPPRESS,
    "erase": GrantKind.PURGE,
}

#: Effects the v1 lifecycle transition table can apply directly. ``correct``
#: and ``reconsider`` are v2 table additions handled by the engine until the
#: core table learns them (they reuse the identical revision/audit writes).
_MACHINE_EFFECTS = frozenset(
    {
        "admit",
        "reject",
        "dispute",
        "resolve",
        "supersede",
        "archive",
        "restore",
        "reverse_supersede",
        "erase",
    }
)


def _proposal_digest(proposal: "EffectProposal") -> bytes:
    """SHA-256 over the canonical proposal JSON — the idempotence input
    digest stored next to the operation key (SPEC_V2 §39.04)."""
    return hashlib.sha256(json_dumps(proposal.to_json()).encode("utf-8")).digest()


def _proposal_interval(proposal: "EffectProposal") -> Optional["TimeInterval"]:
    """The cut interval a supersede carries — ``interval_effects[0]`` when
    constructed in-process, else ``params["interval"]`` for JSON-decoded
    proposals (``EffectProposal.from_json`` has no interval field)."""
    from .core.types import TimeInterval

    if proposal.interval_effects:
        iv = proposal.interval_effects[0]
        if not isinstance(iv, TimeInterval):
            raise VerbatimError(
                ErrorCode.VALIDATION, "interval_effects must hold TimeInterval"
            )
        return iv
    raw = proposal.params.get("interval")
    if raw is None:
        return None
    if isinstance(raw, TimeInterval):
        return raw
    if not isinstance(raw, dict):
        raise VerbatimError(
            ErrorCode.VALIDATION, "params.interval must be an object"
        )
    return TimeInterval(
        from_us=raw.get("from_us"),
        until_us=raw.get("until_us"),
        precision=raw.get("precision", "unknown"),
        timezone=raw.get("timezone"),
        basis=raw.get("basis", "unknown"),
        start_kind=raw.get("start_kind", "exact"),
        end_kind=raw.get("end_kind", "exact"),
        from_us_hi=raw.get("from_us_hi"),
        until_us_hi=raw.get("until_us_hi"),
    )


class ReviewMixin:
    """Transition/proposal/review-effects topic (composed by the facade)."""

    def propose_transition(
        self, cmd: TransitionCommand, scope: Scope, *, caller: Optional[CallerContext] = None
    ) -> str:
        """Create a reviewable transition proposal; returns review_id."""
        self._require_open()
        c = self._resolve_caller(caller, scope)
        self._require_grant(c, GrantKind.PROPOSE)
        if not can_write(c.scope(scope.visibility), scope):
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, "scope not writable by caller"
            )
        from .core.lifecycle import read_claim_head
        from .core.policy import propose_supersede
        from .storage.repos import EventsRepo, ReviewsRepo

        if cmd.effect == "supersede":
            if not cmd.successor_claim_id:
                raise VerbatimError(
                    ErrorCode.VALIDATION, "supersede proposal requires a successor"
                )
            return propose_supersede(
                self.store, cmd.claim_id, cmd.successor_claim_id,
                cmd.actor_id, cmd.reason or "", cmd.interval,
                ctx=self._ingester.policy,
            )

        with self.store.tx() as conn:
            head = read_claim_head(conn, cmd.claim_id)
            if head is None:
                raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "claim not found")
            if head.scope_id != scope_key(scope):
                raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "claim outside scope")
            reviews = ReviewsRepo(self.store)
            EventsRepo(self.store).append(
                conn, head.scope_id, "review_proposed", cmd.actor_id,
                {"effect": cmd.effect, "claim_id": cmd.claim_id,
                 "reason": cmd.reason},
                self._ingester.policy.policy_version,
            )
            return reviews.create(
                conn, head.scope_id,
                {
                    "effect": cmd.effect,
                    "claim_id": cmd.claim_id,
                    "interval": None,
                    "reason": cmd.reason,
                },
                {cmd.claim_id: head.revision},
            )

    def apply_transition(
        self, cmd: TransitionCommand, scope: Scope, *, caller: Optional[CallerContext] = None
    ) -> int:
        """Apply an authorized transition; returns the new event sequence."""
        self._require_open()
        c = self._resolve_caller(caller, scope)
        self._require_grant(c, GrantKind.RESOLVE)
        self._require_write(c, scope)
        from .core.lifecycle import LifecycleMachine, read_claim_head

        machine = LifecycleMachine(self.store)
        activated = False
        with self.store.tx() as conn:
            seq = machine.apply(cmd, conn)
            # No generation bump here (V2-19.10): FTS rows are filtered by
            # ``projection_generation = current``, so bumping would strand
            # every other indexed claim behind the new generation — the
            # transitioned claim is re-indexed at the SAME generation instead.
            # Reindex and purge own structural bumps. The index write rides
            # inside the effect's own transaction (V2-39.01): a crash between
            # transition and indexing can never leave an active claim
            # lexically invisible.
            head = read_claim_head(conn, cmd.claim_id)
            if head is not None and head.state == Lifecycle.ACTIVE:
                self._index_active_in_tx(
                    conn, cmd.claim_id, head.revision, head.scope_id
                )
                activated = cmd.effect == "admit"
        if activated:
            from .core.policy import relate

            relate(self.store, cmd.claim_id, ctx=self._ingester.policy)
        return seq

    def apply_proposal(
        self,
        proposal: "EffectProposal | dict[str, Any]",
        *,
        caller: Optional[CallerContext] = None,
    ) -> dict[str, Any]:
        """Apply one versioned ``EffectProposal`` atomically; returns an
        OperationReceipt-shaped dict.

        Every target's expected revision is re-read inside the write
        transaction — a drifted version fails the whole batch with
        ``STALE_PROPOSAL`` and no partial effects (V2-19.08). The operation
        ledger row commits in the same transaction, so retrying the same
        ``operation_id`` replays the stored receipt (V2-39, V2-19.14).

        Grant checks are per effect (V2-09.10): resolution effects need
        RESOLVE, ``restore`` needs SUPPRESS, ``erase`` needs PURGE. All
        targets must live in one scope partition — cross-scope batches are
        rejected rather than silently splitting authority.
        """
        self._require_open()
        if not isinstance(proposal, EffectProposal):
            proposal = EffectProposal.from_json(proposal)
        grant = _EFFECT_GRANTS[proposal.effect]
        if caller is not None:
            # Authority precedes target dereference (V2-43.02).
            self._require_grant(caller, grant)
        input_digest = _proposal_digest(proposal)
        from .core.lifecycle import (
            LifecycleMachine,
            PURGE_ACTOR,
            can_transition,
            read_claim_head,
        )
        from .storage.repos import EventsRepo, ReviewsRepo
        from .storage.repos_v2 import OperationsRepo

        ops = OperationsRepo(self.store)
        machine = LifecycleMachine(
            self.store,
            policy_version=self._ingester.policy.policy_version,
        )
        active_admissions: list[str] = []
        with self.store.tx() as conn:
            heads = []
            for claim_id, _expected in proposal.targets:
                head = read_claim_head(conn, claim_id)
                if head is None:
                    raise VerbatimError(
                        ErrorCode.NOT_FOUND_OR_FORBIDDEN, "unknown claim"
                    )
                heads.append(head)
            scope_ids = {h.scope_id for h in heads}
            if len(scope_ids) != 1:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "proposal targets must live in one scope partition",
                )
            scope_id = heads[0].scope_id
            owner = self._scope_of(conn, scope_id)
            if owner is None:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN, "unknown claim"
                )
            c = self._resolve_caller(caller, owner, conn)
            self._require_grant(c, grant)
            if proposal.effect == "erase" and not c.is_operator:
                # Erasure is operator-only on top of the PURGE grant
                # (V2-09.10): a transport-issued purge grant alone never
                # drives a claim to erased.
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                    "erase requires operator authority",
                )
            self._require_write(c, owner)
            prior = ops.check(conn, scope_id, proposal.operation_id, input_digest)
            if prior is not None:
                # Committed receipt: replay it verbatim, never re-apply.
                receipt = safe_json_loads(prior["receipt_json"])
                if not isinstance(receipt, dict):
                    receipt = {}
                receipt["replayed"] = True
                return receipt

            # A successor pinned by expected_versions is validated before any
            # mutation, mirroring the per-target revision checks.
            if proposal.successor_claim_id is not None:
                succ = read_claim_head(conn, proposal.successor_claim_id)
                if succ is None:
                    raise VerbatimError(
                        ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                        "successor claim not found",
                    )
                pinned = proposal.params.get("successor_expected_revision")
                if pinned is not None and succ.revision != int(pinned):
                    raise VerbatimError(
                        ErrorCode.STALE_PROPOSAL,
                        f"successor expected revision {pinned}, "
                        f"current {succ.revision}",
                    )
            if proposal.effect == "dispute":
                self._ensure_conflict_edge(conn, proposal, heads[0])

            results: list[dict[str, Any]] = []
            last_seq = 0
            for (claim_id, expected_rev), head in zip(proposal.targets, heads):
                if proposal.effect == "erase":
                    # The lifecycle machine restricts ``erase`` to the purge
                    # workflow actor; the PURGE grant + operator check above
                    # are the authority. The requesting actor is preserved on
                    # this audit event and in the operation receipt.
                    EventsRepo(self.store).append(
                        conn, scope_id, "erase_requested", proposal.actor_id,
                        {"claim_id": claim_id,
                         "operation_id": proposal.operation_id,
                         "reason": proposal.reason},
                        self._ingester.policy.policy_version,
                    )
                    cmd = TransitionCommand(
                        claim_id=claim_id,
                        expected_revision=expected_rev,
                        effect="erase",
                        actor_id=PURGE_ACTOR,
                        reason=proposal.reason,
                    )
                    seq = machine.apply(cmd, conn)
                elif can_transition(head.state, proposal.effect):
                    cmd = TransitionCommand(
                        claim_id=claim_id,
                        expected_revision=expected_rev,
                        effect=proposal.effect,
                        actor_id=proposal.actor_id,
                        reason=proposal.reason,
                        successor_claim_id=proposal.successor_claim_id,
                        interval=_proposal_interval(proposal),
                    )
                    seq = machine.apply(cmd, conn)
                else:
                    # V2-table edges the v1 machine does not list (correct,
                    # reconsider, disputed→supersede, rejected→archive).
                    seq = self._apply_revision_effect(
                        conn, proposal, claim_id, expected_rev, head
                    )
                new_head = read_claim_head(conn, claim_id)
                new_state = new_head.state if new_head is not None else None
                if new_state == Lifecycle.ACTIVE:
                    self._index_active_in_tx(
                        conn, claim_id, new_head.revision, scope_id
                    )
                    if proposal.effect == "admit":
                        active_admissions.append(claim_id)
                results.append(
                    {
                        "claim_id": claim_id,
                        "expected_revision": expected_rev,
                        "revision": new_head.revision if new_head else None,
                        "state": new_state.value if new_state else None,
                        "event_seq": seq,
                    }
                )
                last_seq = seq

            review_id = proposal.params.get("review_id")
            if review_id is not None:
                # Approve = effect + review resolution in ONE transaction
                # (V2-20.04); a stale/re-resolved review fails here. The
                # review must live in the same partition — a foreign
                # review id is indistinguishable from a missing one.
                rrow = conn.execute(
                    "SELECT scope_id, state FROM reviews WHERE review_id = ?",
                    (str(review_id),),
                ).fetchone()
                if rrow is None or rrow[0] != scope_id:
                    raise VerbatimError(
                        ErrorCode.NOT_FOUND_OR_FORBIDDEN, "review not found"
                    )
                ReviewsRepo(self.store).resolve(
                    conn, str(review_id), "approved", last_seq
                )

            receipt = {
                "operation_id": proposal.operation_id,
                "effect": proposal.effect,
                "actor_id": proposal.actor_id,
                "scope_id": scope_id,
                "targets": results,
                "committed_event": last_seq,
                "review_id": str(review_id) if review_id is not None else None,
                "replayed": False,
            }
            ops.record(
                conn, scope_id, proposal.operation_id,
                input_digest=input_digest, effect_kind=proposal.effect,
                receipt=receipt, committed_event=last_seq,
            )
        if active_admissions:
            from .core.policy import relate

            for claim_id in active_admissions:
                relate(self.store, claim_id, ctx=self._ingester.policy)
        return receipt

    def _scope_of(self, conn: Any, scope_id: str) -> Optional[Scope]:
        from .core.types import Visibility

        row = conn.execute(
            "SELECT profile_id, principal_id, workspace_id, conversation_id,"
            " visibility FROM scopes WHERE scope_id = ?",
            (scope_id,),
        ).fetchone()
        if row is None:
            return None
        return Scope(
            profile_id=row[0],
            principal_id=row[1],
            workspace_id=row[2],
            conversation_id=row[3],
            visibility=Visibility(row[4]),
        )

    def _ensure_conflict_edge(
        self, conn: Any, proposal: EffectProposal, head: Any
    ) -> None:
        """A dispute needs a live conflicts_with edge; an operator proposal
        may name the counter-claim explicitly in ``params``."""
        other_id = proposal.params.get("conflict_with_claim_id")
        if not other_id:
            return  # the machine enforces edge presence itself
        from .core.lifecycle import read_claim_head
        from .storage.repos import EdgesRepo

        other = read_claim_head(conn, str(other_id))
        if other is None or other.scope_id != head.scope_id:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, "conflict claim not found"
            )
        EdgesRepo(self.store).add(
            conn, head.scope_id, "claim", head.claim_id,
            "claim", str(other_id), "conflicts_with",
        )

    def _apply_revision_effect(
        self,
        conn: Any,
        proposal: EffectProposal,
        claim_id: str,
        expected_revision: int,
        head: Any,
    ) -> int:
        """V2-table edges the v1 machine does not list (SPEC_V2 §20):

        ``correct``    active/disputed/superseded → rejected (+ ``corrects`` edge)
        ``reconsider`` rejected → pending (re-review; never auto-activation)
        ``supersede``  disputed → superseded (v1 table only admits active→…)
        ``archive``    rejected → archived (v1 table lacks this edge)

        The write phase mirrors ``LifecycleMachine.apply`` exactly —
        transition event, ``recorded_until`` close, typed edges, revision
        append, v2 revision stamp — inside a savepoint so a failed check
        mid-effect cannot leave partial rows.
        """
        from .core.lifecycle import (
            _stamp_revision_v2,
            _supersession_path_exists,
            _truncate_intervals,
            read_claim_head,
            read_evidence,
            read_intervals,
        )
        from .storage.repos import ClaimsRepo, EdgesRepo, EventsRepo

        effect = proposal.effect
        if head.revision != expected_revision:
            raise VerbatimError(
                ErrorCode.STALE_PROPOSAL,
                f"expected revision {expected_revision}, current {head.revision}",
            )
        intervals = read_intervals(conn, claim_id, head.revision)
        evidence = read_evidence(conn, claim_id, head.revision)
        new_intervals = list(intervals)
        edges: list[tuple[str, str, str]] = []  # (source_id, target_id, type)

        if effect == "correct" and head.state in (
            Lifecycle.ACTIVE, Lifecycle.DISPUTED, Lifecycle.SUPERSEDED,
        ):
            to_state = Lifecycle.REJECTED
            if proposal.successor_claim_id:
                # The explicit replacement: a live same-scope claim, linked
                # by a ``corrects`` edge (V2-20.17).
                succ = read_claim_head(conn, proposal.successor_claim_id)
                if succ is None:
                    raise VerbatimError(
                        ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                        "successor claim not found",
                    )
                if succ.claim_id == claim_id or succ.scope_id != head.scope_id:
                    raise VerbatimError(
                        ErrorCode.INVALID_TRANSITION,
                        "correct successor must be a different claim in the same scope",
                    )
                if succ.state in (
                    Lifecycle.ERASED, Lifecycle.REJECTED, Lifecycle.SUPERSEDED,
                ):
                    raise VerbatimError(
                        ErrorCode.INVALID_TRANSITION,
                        f"successor in state {succ.state.value} cannot replace",
                    )
                edges.append((succ.claim_id, claim_id, "corrects"))
        elif effect == "reconsider" and head.state == Lifecycle.REJECTED:
            to_state = Lifecycle.PENDING
        elif effect == "supersede" and head.state == Lifecycle.DISPUTED:
            to_state = Lifecycle.SUPERSEDED
            if not proposal.successor_claim_id:
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    "supersede requires an identified successor claim",
                )
            succ = read_claim_head(conn, proposal.successor_claim_id)
            if succ is None:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN, "successor claim not found"
                )
            if succ.claim_id == claim_id or succ.scope_id != head.scope_id:
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    "successor must be a different claim in the same scope",
                )
            if succ.state in (
                Lifecycle.ERASED, Lifecycle.REJECTED, Lifecycle.SUPERSEDED,
            ):
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    f"successor in state {succ.state.value} cannot supersede",
                )
            if _supersession_path_exists(conn, claim_id, succ.claim_id):
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    "supersession would create a cycle",
                )
            new_intervals = _truncate_intervals(
                intervals, _proposal_interval(proposal)
            )
            edges.append((succ.claim_id, claim_id, "supersedes"))
        elif effect == "archive" and head.state == Lifecycle.REJECTED:
            to_state = Lifecycle.ARCHIVED
        else:
            raise VerbatimError(
                ErrorCode.INVALID_TRANSITION,
                f"effect {effect!r} not permitted from {head.state.value}",
            )

        claims = ClaimsRepo(self.store)
        conn.execute("SAVEPOINT verbatim_proposal_effect")
        try:
            seq = EventsRepo(self.store).append(
                conn, head.scope_id, "claim_transition", proposal.actor_id,
                {
                    "claim_id": claim_id,
                    "effect": effect,
                    "from_state": head.state.value,
                    "to_state": to_state.value,
                    "reason": proposal.reason,
                    "successor_claim_id": proposal.successor_claim_id,
                },
                self._ingester.policy.policy_version,
            )
            claims.set_recorded_until(claim_id, head.revision, seq, conn)
            edge_repo = EdgesRepo(self.store)
            for src, dst, et in edges:
                edge_repo.add(
                    conn, head.scope_id, "claim", src, "claim", dst, et
                )
            new_rev = claims.add_revision(
                claim_id, to_state.value, head.object_json, head.polarity,
                head.modality, head.condition_json, head.interpretation_json,
                new_intervals, evidence, seq, conn,
            )
            _stamp_revision_v2(conn, claim_id, new_rev, head, new_intervals)
            conn.execute("RELEASE verbatim_proposal_effect")
        except BaseException:
            conn.execute("ROLLBACK TO verbatim_proposal_effect")
            conn.execute("RELEASE verbatim_proposal_effect")
            raise
        return seq

    def _index_active_in_tx(
        self, conn: Any, claim_id: str, revision: int, scope_id: str
    ) -> None:
        """Index a newly-ACTIVE head revision under the CURRENT projection
        generation — inside the effect's own transaction (V2-39.01).

        Mirrors ``Ingester.index_active_claim``'s query on the caller's
        ``conn``. The generation is not bumped: ordinary updates must not
        strand unrelated rows behind a fresh global index generation
        (V2-19.10, fixes the F02 recall gap for this path).
        """
        from .storage.repos import FtsRepo

        if self.store.fts_enabled:
            generation = self.store.projection_generation()
            FtsRepo(self.store).index_claim_primary(
                conn, claim_id, revision, scope_id, generation
            )
        # The embed obligation rides the same commit so a claim activated
        # through apply_transition reaches the semantic lane exactly like
        # the review_apply/index_active_claim paths — independent of FTS
        # (``_enqueue_embed`` no-ops when no encoder is configured).
        if "claim_evidence" in self._ingester._tables:
            span = conn.execute(
                "SELECT ce.span_id FROM claim_evidence ce"
                " WHERE ce.claim_id = ? AND ce.revision = ?"
                "   AND ce.evidence_role = 'primary'"
                " ORDER BY ce.span_id LIMIT 1",
                (claim_id, revision),
            ).fetchone()
            if span is not None:
                self._ingester._enqueue_embed(conn, scope_id, [span[0]])


__all__ = [
    "ReviewMixin",
    "_EFFECT_GRANTS",
    "_MACHINE_EFFECTS",
    "_proposal_digest",
    "_proposal_interval",
]
