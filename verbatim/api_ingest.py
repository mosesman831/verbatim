"""Engine sibling — evidence ingest and grounded remember (V3-06.01).

Owns the write-path topics: envelope ingest with the operations-ledger
idempotence contract (V2-39), pending-job draining, and ``remember`` —
revalidated-evidence admission through the policy pipeline. Methods are
verbatim moves from the former ``api.py`` god file; the facade composes
this mixin, so cross-topic private helpers resolve on ``self`` as before.
"""

from __future__ import annotations

from typing import Any, Optional

from .core.identity import scope_key
from .core.types import (
    CallerContext,
    ClaimProposal,
    ErrorCode,
    GrantKind,
    IngestReceipt,
    JobKind,
    Lifecycle,
    Scope,
    SourceEnvelope,
    VerbatimError,
    json_dumps,
    require_id,
    safe_json_loads,
)

#: Quarantine states that hide an object revision (mirrors
#: ``security.EXCLUDING_STATES``; re-declared so the fallback path never
#: imports a parallel module — same convention as retrieval/v3 union).
_HELD_STATES = frozenset({"pending", "suppressed"})


def _held(conn: Any, object_kind: str, object_id: str, revision: int) -> bool:
    """Quarantine exclusion for one object revision (V3-14.10).

    ``security.should_exclude`` when the parallel module is provisioned,
    the quarantine-table state otherwise — the same defensive pair
    ``retrieval/v3/union.py::_should_exclude`` uses. Fails closed on
    unreadable state.
    """
    try:
        from . import security as _security  # type: ignore
    except Exception:
        _security = None
    if _security is not None:
        try:
            return bool(
                _security.should_exclude(conn, object_kind, object_id, revision)
            )
        except Exception:
            pass  # fall back to the local table check below
    try:
        row = conn.execute(
            "SELECT state FROM quarantine"
            " WHERE object_kind = ? AND object_id = ? AND revision = ?",
            (object_kind, object_id, revision),
        ).fetchone()
    except Exception:
        return True  # no readable quarantine state — fail closed
    return row is not None and row[0] in _HELD_STATES


class IngestMixin:
    """Evidence ingest + remember topic (composed by the api.py facade)."""

    def ingest(
        self, envelope: SourceEnvelope, *, caller: Optional[CallerContext] = None
    ) -> IngestReceipt:
        """Persist an accepted source revision and queue interpretation.

        Idempotent under a stable operation key (SPEC_V2 §39): the key is the
        envelope's ``metadata["operation_id"]`` when supplied, else a
        profile-keyed digest of (origin, external_id, source_id, revision,
        payload). The operations ledger row commits in the SAME transaction
        as the source revision + harvest job, so a retried call replays the
        stored receipt instead of duplicating work (v1 gap F05).
        """
        self._require_open()
        c = self._resolve_caller(caller, envelope.scope)
        self._require_grant(c, GrantKind.INGEST)
        self._require_write(c, envelope.scope)
        from .storage.repos import EventsRepo
        from .storage.repos_v2 import OperationsRepo

        # The privacy gate runs before any persistence — identical to
        # ``Ingester.ingest``; the body below mirrors that method exactly so
        # the operations receipt can join its transaction (the Ingester owns
        # its own tx and cannot accept ours).
        self._ingester._gate(envelope)
        scope_id = scope_key(envelope.scope)
        operation_key, input_digest = self._ingest_operation(envelope)
        ops = OperationsRepo(self.store)
        generation = self.store.projection_generation()
        with self.store.tx() as conn:
            prior = ops.check(conn, scope_id, operation_key, input_digest)
            if prior is not None:
                return self._ingest_receipt_from(prior, generation)
            sid, created = self._ingester.sources.insert(envelope, conn=conn)
            if not created:
                receipt = IngestReceipt(
                    accepted=(), rejected=(), job_ids=(),
                    projection_generation=generation, duplicate=True,
                )
                ops.record(
                    conn, scope_id, operation_key,
                    input_digest=input_digest, effect_kind="ingest",
                    receipt=self._ingest_receipt_json(receipt),
                )
                return receipt
            EventsRepo(self.store).append(
                conn, scope_id, "source_accepted", "engine",
                {"source_id": sid, "kind": envelope.source_kind.value},
                self._ingester.policy.policy_version,
            )
            dedup = self.store.hmac(f"harvest:{sid}:{envelope.revision}".encode())
            # The harvest job carries a stable operation key so a redelivered
            # execution replays its receipt instead of harvesting the same
            # source twice — mirrors Ingester.ingest (V2-39.02/39.10).
            job_id = self._ingester.jobs.enqueue(
                conn, scope_id, JobKind.HARVEST,
                {"source_id": sid, "revision": envelope.revision},
                dedup_key=dedup,
                operation_key=(
                    f"harvest:{sid}:{envelope.revision}"
                    if self._ingester.jobs.supports_durability
                    else None
                ),
            )
            # V4-14.01: processing obligations commit atomically with the
            # source + harvest job — mirrors Ingester.ingest exactly.
            self._ingester._record_capture_obligations(
                conn, sid, envelope.revision, scope_id
            )
            receipt = IngestReceipt(
                accepted=(sid,), rejected=(), job_ids=(job_id,),
                projection_generation=generation,
            )
            ops.record(
                conn, scope_id, operation_key,
                input_digest=input_digest, effect_kind="ingest",
                receipt=self._ingest_receipt_json(receipt),
            )
            return receipt

    def run_pending(self, limit: int = 64, *, caller: Optional[CallerContext] = None) -> int:
        """Synchronously drain durable jobs (CLI/test path)."""
        self._require_open()
        c = self._resolve_caller(caller, self.host.default_scope())
        self._require_grant(c, GrantKind.OPERATOR)
        return self._ingester.run_pending(scope=self.host.default_scope(), limit=limit)

    def remember(
        self,
        source_id: str,
        start_byte: int,
        end_byte: int,
        scope: Scope,
        predicate_suggestion: Optional[str] = None,
        *,
        caller: Optional[CallerContext] = None,
        revision: int = 1,
        operation_id: Optional[str] = None,
    ) -> str:
        """Grounded remember: revalidates exact evidence against an accepted
        source revision, then admits through the normal policy pipeline as
        agent-suggested (SPEC §35). Returns the claim_id.

        Idempotent (SPEC_V2 §39): the operation key defaults to a
        profile-keyed digest of the full argument set, and the span id is
        deterministic — a retried remember replays the recorded claim
        instead of persisting a second span+claim (v1 gap F05).
        """
        self._require_open()
        c = self._resolve_caller(caller, scope)
        self._require_grant(c, GrantKind.PROPOSE)
        self._require_write(c, scope)
        from .core.claims import propose
        from .core.types import SpanRef
        from .storage.repos_v2 import OperationsRepo

        require_id(source_id, "source_id")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise VerbatimError(ErrorCode.VALIDATION, "revision must be >= 1")
        scope_id = scope_key(scope)
        material = json_dumps(
            {
                "source_id": source_id,
                "revision": revision,
                "start_byte": start_byte,
                "end_byte": end_byte,
                "predicate": predicate_suggestion or "",
                "scope_id": scope_id,
            }
        )
        input_digest = self.store.hmac(material.encode("utf-8"))
        if operation_id is not None:
            operation_key = require_id(operation_id, "operation_id")
        else:
            operation_key = "remember:" + input_digest.hex()[:48]
        # Deterministic span identity is the atomic fence: an identical
        # remember (same material → same digest → same span id) cannot insert
        # twice, so a crash between the span insert and admission commits
        # still converges on one claim.
        span_id = "sp" + input_digest.hex()[:30]

        ops = OperationsRepo(self.store)
        with self.store.read() as conn:
            prior = ops.check(conn, scope_id, operation_key, input_digest)
            if prior is not None:
                receipt = safe_json_loads(prior["receipt_json"])
                if isinstance(receipt, dict) and receipt.get("claim_id"):
                    return str(receipt["claim_id"])
            healed = self._claim_for_span(conn, span_id)
            if healed is not None:
                self._record_operation(
                    scope_id, operation_key, input_digest, "remember",
                    {"claim_id": healed},
                )
                return healed

        payload = self._ingester.sources.payload(source_id, revision)
        if payload is None:
            raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "unknown source")
        if self._source_revision_held(source_id, revision):
            # Held and unknown are indistinguishable (SPEC §9): a purge
            # tombstone or quarantine hold covering these bytes makes the
            # source unavailable for new evidence — suppressed or
            # quarantined content must never mint claims (V3-14.10).
            raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "unknown source")
        data = bytes(payload)
        if not (0 <= start_byte < end_byte <= len(data)):
            raise VerbatimError(ErrorCode.VALIDATION, "invalid byte range")
        try:
            text = data[start_byte:end_byte].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VerbatimError(ErrorCode.VALIDATION, "range splits a UTF-8 character") from exc
        from .ingest import envelope_for

        envelope = envelope_for(self.store, source_id, revision)
        if envelope is None:
            raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "unknown source")
        span = SpanRef(
            span_id=span_id, source_id=source_id, revision=revision,
            start_byte=start_byte, end_byte=end_byte,
        )
        proposal = propose(text, span, envelope, envelope.event_us)
        proposal = ClaimProposal(
            evidence=(span,),
            predicate=predicate_suggestion or proposal.predicate,
            object_json=proposal.object_json,
            subject_entity_id=proposal.subject_entity_id,
            polarity=proposal.polarity,
            modality=proposal.modality,
            condition=proposal.condition,
            valid=proposal.valid,
            method="agent_remember",
            agent_suggested=True,
        )
        try:
            with self.store.tx() as conn:
                self._ingester.spans.insert(
                    span_id, source_id, revision, start_byte, end_byte,
                    "agent-remember-1", conn=conn,
                )
        except VerbatimError as exc:
            # Unique-key conflict: an identical remember already committed the
            # span (concurrent caller, or a crashed earlier attempt). If the
            # claim landed too, replay it; if the span outlived a crash
            # before admission, fall through and admit on top of it — the
            # deterministic span id keeps the retry converged.
            if exc.code != ErrorCode.VALIDATION or "constraint" not in exc.message:
                raise
            healed = self._claim_for_span_read(span_id)
            if healed is not None:
                self._record_operation(
                    scope_id, operation_key, input_digest, "remember",
                    {"claim_id": healed},
                )
                return healed
        # admit() manages its own transaction (SPEC §21); nesting would raise.
        from .core.policy import admit
        outcome = admit(self.store, proposal, envelope, ctx=self._ingester.policy)
        if not outcome.claim_id:
            raise VerbatimError(ErrorCode.VALIDATION, f"not admitted: {outcome.reason}")
        if outcome.state == Lifecycle.ACTIVE:
            self._ingester.index_claim(outcome.claim_id, scope, text)
        self._record_operation(
            scope_id, operation_key, input_digest, "remember",
            {"claim_id": outcome.claim_id},
        )
        return outcome.claim_id

    def _source_revision_held(self, source_id: str, revision: int) -> bool:
        """Active-hold gate for the source revision ``remember`` quotes.

        The same cascade retrieval applies (V3-14.10, SPEC §40): a purge
        tombstone on the source or its ``<sid>:<rev>`` revision record, or
        a quarantine hold on the source revision or any covering
        ``source_envelopes`` row, makes the bytes unavailable for new
        evidence — during the suppress→execute window, and indefinitely
        under quarantine. Fails closed: a check that cannot run reports
        held rather than leaking held bytes into a new claim.
        """
        from .storage.repos import PurgesRepo, has_table

        try:
            purges = PurgesRepo(self.store)
            if purges.suppressed_ids("source", [source_id]):
                return True
            if purges.suppressed_ids(
                "source_revision", [f"{source_id}:{revision}"]
            ):
                return True
        except Exception:
            return True  # tombstone lookup broken — fail closed
        try:
            with self.store.read() as conn:
                if not has_table(conn, "quarantine"):
                    return False  # schema predates quarantine — no holds exist
                if _held(conn, "source", source_id, revision):
                    return True
                if has_table(conn, "source_envelopes"):
                    rows = conn.execute(
                        "SELECT envelope_id FROM source_envelopes"
                        " WHERE source_id = ? AND revision = ?",
                        (source_id, revision),
                    ).fetchall()
                    if any(
                        _held(conn, "source_envelope", r[0], revision)
                        for r in rows
                    ):
                        return True
        except VerbatimError:
            raise
        except Exception:
            return True  # hold check broken — fail closed
        return False

    def _claim_for_span(self, conn: Any, span_id: str) -> Optional[str]:
        """Claim citing this deterministic span, if one was already admitted."""
        row = conn.execute(
            "SELECT claim_id FROM claim_evidence WHERE span_id = ?"
            " ORDER BY revision DESC LIMIT 1",
            (span_id,),
        ).fetchone()
        return str(row[0]) if row is not None else None

    def _claim_for_span_read(self, span_id: str) -> Optional[str]:
        with self.store.read() as conn:
            return self._claim_for_span(conn, span_id)

    def _ingest_operation(self, envelope: SourceEnvelope) -> tuple[str, bytes]:
        """Stable (operation_key, input_digest) for an ingest envelope.

        An explicit ``metadata["operation_id"]`` wins so transports can pin
        their own retry identity; the digest still guards key reuse against
        changed payloads (V2-39.07).
        """
        material = json_dumps(
            {
                "origin": envelope.origin,
                "external_id": envelope.external_id,
                "source_id": envelope.source_id,
                "revision": envelope.revision,
                "payload_hmac": self.store.hmac(bytes(envelope.payload)).hex(),
            }
        )
        input_digest = self.store.hmac(material.encode("utf-8"))
        explicit = None
        if isinstance(envelope.metadata, dict):
            explicit = envelope.metadata.get("operation_id")
        if explicit is not None:
            return require_id(str(explicit), "operation_id"), input_digest
        return "ingest:" + input_digest.hex()[:48], input_digest

    def _ingest_receipt_json(self, receipt: IngestReceipt) -> dict[str, Any]:
        return {
            "accepted": list(receipt.accepted),
            "rejected": [list(r) for r in receipt.rejected],
            "job_ids": list(receipt.job_ids),
            "projection_generation": receipt.projection_generation,
            "duplicate": receipt.duplicate,
        }

    def _ingest_receipt_from(self, op_row: dict[str, Any], generation: int) -> IngestReceipt:
        """Rebuild the stored ingest receipt for a replayed request (V2-39)."""
        data = safe_json_loads(op_row["receipt_json"])
        if not isinstance(data, dict):
            data = {}
        return IngestReceipt(
            accepted=tuple(str(s) for s in data.get("accepted") or ()),
            rejected=tuple(
                (str(r[0]), str(r[1])) for r in data.get("rejected") or ()
            ),
            job_ids=tuple(str(j) for j in data.get("job_ids") or ()),
            projection_generation=generation,
            duplicate=True,
        )

    def _record_operation(
        self,
        scope_id: str,
        operation_key: str,
        input_digest: bytes,
        effect_kind: str,
        receipt: dict[str, Any],
    ) -> None:
        """Persist an operation receipt in its own short transaction.

        Used for effects whose domain commits already landed (remember's
        admit path owns its tx); the check-inside guards the ledger against
        the last narrow race.
        """
        from .storage.repos_v2 import OperationsRepo

        ops = OperationsRepo(self.store)
        with self.store.tx() as conn:
            existing = ops.check(conn, scope_id, operation_key, input_digest)
            if existing is None:
                ops.record(
                    conn, scope_id, operation_key,
                    input_digest=input_digest, effect_kind=effect_kind,
                    receipt=receipt,
                )


__all__ = ["IngestMixin"]
