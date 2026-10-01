"""Host-neutral V3 facade — ``VerbatimV3`` (SPEC_V3 §47, §48.01).

This module is the v3 entry point for embedded hosts and transport
adapters. It composes the frozen worker contracts — governance
(§08–§11), evidence ingest (§12–§13), security labels/quarantine
(§14, §34), purge closure (§36), and the job lanes (§40) — behind one
caller's ``Store``. Nothing here mints authority: a caller principal is
authenticated by the host at construction time, never by request fields
(V3-48.01/V3-48.05).

Requirement coverage:

- V3-11.11 / V3-12.10: ``issue_capture_authorization`` writes the
  ``capture_authorizations`` consent record and ``capture_submitted``
  binds it as ``capture_proof`` — tool permission is never retention
  consent, and a missing consent fails ``CONSENT_REQUIRED``, not a
  silent downgrade.
- V3-13.11: agent-submitted text is recorded verbatim but attributed
  ``agent_generated`` — a ``declared_type`` naming a principal-authored
  kind is recorded as ``agent_note`` because an agent cannot mint human
  authorship; the declaration is preserved in envelope metadata.
- V3-10.05 / V3-09.09: denials are ``NOT_FOUND_OR_UNAUTHORIZED``;
  existence and missing authority are publicly indistinguishable.
- V3-14.01/14.10: capture runs deterministic rules_v1 screening and opens
  quarantine holds on ``suspicious``/``blocked`` verdicts inside the same
  transaction.
- V3-26.07/27/30: ``recall`` authorizes ``read``, prefers the W9
  ``verbatim.retrieval.v3.recall.recall_v3`` pipeline through a lazy
  import, and answers bare evidence from the lexical fallback when the
  derived-object lanes have nothing to return yet.
- V3-36.02: ``delete_source`` authorizes ``admin``, tombstones the
  source's closure in the same transaction, and enqueues the durable
  ``purge``/``purge_derived``/``purge_vault`` jobs — logical deletion
  now, physical erasure when the privacy-control lane drains. This is
  NOT a claim of cryptographic erasure (§36.05).
- V3-47.01..03: inputs are validated before identifier dereference;
  mutation responses carry receipt anchors.
- V3-62.04: ``capabilities`` reports honest lane/vault/encoder
  availability with explicit degradation notes.

Scope ownership (fresh-store onboarding, §48.01): a scope row without
``owner_principal_id`` is unowned. The first launch-bound principal to
issue a capture authorization on it bootstraps as owner — the embedded
profile's launch binding IS the sovereign identity — receiving a root
grant for the facade's ``bound_verbs``. Issuing consent for a scope that
already names a different owner requires ``admin`` on that scope.
"""

from __future__ import annotations

import os
import re
import sqlite3
from typing import Any, Iterable, Optional

from .. import governance
from .. import purge as _purge
from .. import security
from ..config import VerbatimConfig
from ..core.time import now_us
from ..core.types import (
    ErrorCode,
    JobKind,
    VerbatimError,
    new_id,
    require_id,
)
from ..core.types_v3 import (
    BudgetTier,
    ContextPack,
    EnvelopeKind,
    FreshnessClass,
    InfluenceHandle,
    OutcomeClass,
    PackItem,
    PackKind,
    Perspective,
    RecallRequestV3,
    RecallResultV3,
    SecurityLabel,
    SourceEnvelopeV3,
    TaskContext,
    TrajectoryRecord,
    TrajectoryStep,
    TrustClass,
    Verb,
)
from ..core.types_v4 import CapabilityReport, CapabilityRung
from ..evidence import ingest_envelope
from ..evidence.envelopes import ensure_scope_row
from ..evidence.receipts import receipt_for_envelope, receipt_id_for
from ..jobs.queue import JobQueue
from ..storage import repos_v3
from ..storage.repos import SourcesRepo
from ..storage.store import Store

__all__ = ["VerbatimV3", "OWNER_VERBS", "AGENT_SUBMITTED_KINDS"]

#: Verbs granted to a bootstrapping scope owner through this facade. The
#: embedded owner is sovereign on its own partition; transports narrow
#: this through ``bound_verbs`` (MCP drops admin/review — V3-47.09).
OWNER_VERBS: frozenset = frozenset(v.value for v in Verb)

#: Envelope kinds an agent may submit through ``capture_submitted``
#: (§12.10 agent-submitted set). Anything else declared is recorded as
#: ``agent_note`` — attribution can only ever go *more* honest, never
#: mint a principal-authored kind (§13.11).
AGENT_SUBMITTED_KINDS: frozenset = frozenset(
    {
        EnvelopeKind.AGENT_NOTE,
        EnvelopeKind.LESSON,
        EnvelopeKind.TOOL_CALL,
        EnvelopeKind.TOOL_RESULT,
    }
)

_FACADE_POLICY = "api_v3.facade.v1"
_CAPTURE_POLICY_REVISION = "api_v3.capture_policy.v1"
_DEFAULT_RETENTION_POLICY = "explicit_agent_note"
_MAX_CONTENT_BYTES = 65536
_EVIDENCE_SCAN_LIMIT = 512
_SUPPRESSING_PURGE_STATES = ("suppressed", "purging", "completed")
_TERM_RE = re.compile(r"[\w']+", re.UNICODE)

#: Declared request modes that opt into the raw archive/evidence lane
#: (V4-08.07): the lane runs only when the caller explicitly declares
#: one of these (or calls ``browse_evidence``) — it is never an implicit
#: fallback after denied, empty, or budget-exhausted recall.
_EVIDENCE_LANE_MODES = frozenset({"archive", "evidence", "browse"})

#: Fields a host checker resolver's receipt may carry. ``scope_id``,
#: ``task_id``, ``artifact_digest``, and ``invocation_id`` are binding
#: fields (V4-23.01): a non-empty bound value that disagrees with the
#: submission context rejects the receipt — a receipt for another
#: task/scope/artifact cannot certify this one (C20).
_CHECKER_RECEIPT_FIELDS = (
    "checker_id",
    "checker_version",
    "invocation_id",
    "completed",
    "exit_code",
    "outcome",
    "scope_id",
    "task_id",
    "artifact_digest",
    "environment_digest",
    "repo_revision",
    "tree_digest",
    "nonce",
    "issued_us",
    "selected_tests",
    "result_json",
    "issuer",
)
_RECEIPT_BINDINGS = ("invocation_id", "scope_id", "task_id", "artifact_digest")
_ATTESTATION_DOMAIN = "v3.checker_attestation.v1"


def _err(code: ErrorCode, msg: str) -> VerbatimError:
    return VerbatimError(code, msg)


def _deny(msg: str = "not found or unauthorized") -> "Any":
    raise _err(ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, msg)


class VerbatimV3:
    """The host-neutral V3 API surface over one open ``Store``.

    ``bound_verbs`` is the launch-declared verb set a bootstrapping owner
    receives (§48.01); it never widens later — existing grants decide.
    ``principal_kind`` is only stamped when the facade first registers a
    principal row; pre-provisioned identity rows win.
    """

    def __init__(
        self,
        store: Store,
        config: Optional[VerbatimConfig] = None,
        *,
        bound_verbs: Optional[Iterable[str]] = None,
        host_id: str = "",
        adapter_version: str = "api_v3/1.0",
        principal_kind: str = "agent",
        checker_resolver: Optional[Any] = None,
    ) -> None:
        if store is None:
            raise _err(ErrorCode.VALIDATION, "store is required")
        self._store = store
        self._config = config if config is not None else VerbatimConfig()
        verbs = (
            frozenset(str(v) for v in bound_verbs)
            if bound_verbs is not None
            else OWNER_VERBS
        )
        try:
            for v in verbs:
                Verb(v)
        except ValueError as exc:
            raise _err(
                ErrorCode.VALIDATION, f"unknown verb in bound_verbs: {exc}"
            ) from exc
        if not verbs:
            raise _err(ErrorCode.VALIDATION, "bound_verbs cannot be empty")
        self._bound_verbs = verbs
        self._host_id = host_id
        self._adapter_version = adapter_version
        self._principal_kind = principal_kind
        # The host-registered checker resolver (V4-23.02): a callable
        # ``(invocation_id) -> receipt`` or an object exposing
        # ``resolve_checker(invocation_id)``. It is the ONLY source of
        # host-observed execution receipts — caller/model-supplied checker
        # fields are persisted as agent reports, never as verification.
        if checker_resolver is not None and not (
            callable(checker_resolver)
            or callable(getattr(checker_resolver, "resolve_checker", None))
        ):
            raise _err(
                ErrorCode.VALIDATION,
                "checker_resolver must be callable or expose resolve_checker()",
            )
        self._checker_resolver = checker_resolver

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        path: str,
        config: Optional[VerbatimConfig] = None,
        **kwargs: Any,
    ) -> "VerbatimV3":
        """Open (or create) a store at ``path`` and bind the facade."""
        if os.path.exists(path) and os.path.getsize(path) > 0:
            store = Store.open(path)
        else:
            store = Store.create(path)
        return cls(store, config, **kwargs)

    @property
    def store(self) -> Store:
        return self._store

    @property
    def config(self) -> VerbatimConfig:
        return self._config

    @property
    def bound_verbs(self) -> frozenset:
        """The launch-declared verb set (§48.01), read-only. Transports
        compare it against their safe set — e.g. the MCP stdio surface
        refuses a facade bound wider than ``MCP_V3_BOUND_VERBS``."""
        return self._bound_verbs

    def close(self) -> None:
        self._store.close()

    # ------------------------------------------------------------------
    # internal: caller + scope authority
    # ------------------------------------------------------------------

    def _caller(self, principal_id: str, session_id: str = "") -> governance.CallerV3:
        """The bound caller record; identity is asserted by the host, not
        by request parameters (§08.02, V3-48.01)."""
        require_id(principal_id, "principal_id")
        return governance.CallerV3(
            principal_id=principal_id,
            session_id=session_id,
            host_id=self._host_id,
        )

    @staticmethod
    def _scope_row(conn, scope_id: str):
        return conn.execute(
            "SELECT scope_id, owner_principal_id, principal_id"
            " FROM scopes WHERE scope_id = ?",
            (scope_id,),
        ).fetchone()

    def _scope_owner(self, conn, scope_id: str) -> Optional[str]:
        row = self._scope_row(conn, scope_id)
        if row is None:
            return None
        owner = row[1]
        return owner if isinstance(owner, str) and owner else None

    def _ensure_owner_grant(self, conn, scope_id: str, principal_id: str) -> None:
        """Create the owner's root grant when no live grant covers it.

        Ownership recorded on the scope without any grant happens when a
        host provisioned ``owner_principal_id`` out-of-band; the launch-
        bound verb set backfills exactly once (attenuation-only thereafter
        — revoked grants are never resurrected).
        """
        rows = repos_v3.query(
            conn,
            "grants_v3",
            {
                "scope_id": scope_id,
                "principal_id": principal_id,
                "revoked_us": None,
            },
        )
        now = now_us()
        for row in rows:
            exp = row.get("expires_us")
            if exp is None or int(exp) > now:
                return
        governance.create_grant(
            conn,
            scope_id=scope_id,
            principal_id=principal_id,
            verbs=sorted(self._bound_verbs),
            issuer_id=principal_id,
            delegation_depth=1,
        )

    def _bootstrap_owner(self, conn, scope_id: str, principal_id: str) -> None:
        """Claim an unowned scope for the launch-bound principal (§48.01)."""
        ensure_scope_row(conn, scope_id, principal_id=principal_id)
        conn.execute(
            "UPDATE scopes SET owner_principal_id = ?"
            " WHERE scope_id = ? AND owner_principal_id IS NULL",
            (principal_id, scope_id),
        )
        governance.register_principal(
            conn, kind=self._principal_kind, principal_id=principal_id
        )
        self._ensure_owner_grant(conn, scope_id, principal_id)

    def _require_issuer_authority(
        self, conn, scope_id: str, issuer_id: str
    ) -> None:
        """Who may issue capture consent on this scope.

        Unowned scope → the issuer bootstraps as owner. Owner → itself
        (the grant is backfilled if the row was provisioned bare). Anyone
        else → ``admin`` on the scope: retention consent for another
        principal is an administrative act (§09 verb table).
        """
        owner = self._scope_owner(conn, scope_id)
        if owner is None:
            self._bootstrap_owner(conn, scope_id, issuer_id)
            return
        if issuer_id == owner:
            self._ensure_owner_grant(conn, scope_id, issuer_id)
            return
        governance.authorize(
            conn,
            self._caller(issuer_id),
            scope_id,
            Verb.ADMIN.value,
        )

    # ------------------------------------------------------------------
    # §11.11 capture authorization (retention consent, not tool permission)
    # ------------------------------------------------------------------

    def issue_capture_authorization(
        self,
        principal_id: str,
        scope_id: str,
        *,
        granted_by: str,
        purpose: Optional[str] = None,
        ttl_s: Optional[float] = None,
    ) -> str:
        """Issue retention consent for agent-submitted capture; returns
        the ``authorization_id`` (§11.11).

        ``granted_by`` is the issuer asserting authority — the scope
        owner, or any ``admin``-holding principal when consenting for
        another principal. The authorization covers the agent-submitted
        envelope kinds on ``scope_id``; ``purpose`` narrows the recorded
        retention policy name and ``ttl_s`` bounds its lifetime.
        """
        require_id(principal_id, "principal_id")
        require_id(scope_id, "scope_id")
        require_id(granted_by, "granted_by")
        if ttl_s is not None and (
            isinstance(ttl_s, bool)
            or not isinstance(ttl_s, (int, float))
            or ttl_s <= 0
        ):
            raise _err(ErrorCode.VALIDATION, "ttl_s must be a positive number")
        if purpose is not None and not isinstance(purpose, str):
            raise _err(ErrorCode.VALIDATION, "purpose must be a string")
        with self._store.tx() as conn:
            governance.seed_purposes(conn)
            self._require_issuer_authority(conn, scope_id, granted_by)
            expires = (
                now_us() + int(ttl_s * 1_000_000)
                if ttl_s is not None
                else None
            )
            return governance.issue_capture_authorization(
                conn,
                principal_id=principal_id,
                issuer_id=granted_by,
                allowed_kinds=sorted(
                    k.value for k in AGENT_SUBMITTED_KINDS
                ),
                retention_policy=purpose or _DEFAULT_RETENTION_POLICY,
                policy_revision=_CAPTURE_POLICY_REVISION,
                scope_ids=(scope_id,),
                expires_us=expires,
            )

    # ------------------------------------------------------------------
    # §13 agent-submitted capture
    # ------------------------------------------------------------------

    @staticmethod
    def _submitted_kind(declared_type: Any) -> EnvelopeKind:
        """Resolve the honest envelope kind for an agent submission.

        Agent-submittable kinds pass through; every other declaration is
        recorded as ``agent_note`` — an agent cannot mint human or host
        authorship, so the kind only ever moves toward *more* honest
        attribution (§13.11). The original declaration stays in metadata.
        """
        try:
            kind = (
                declared_type
                if isinstance(declared_type, EnvelopeKind)
                else EnvelopeKind(str(declared_type))
            )
        except ValueError:
            return EnvelopeKind.AGENT_NOTE
        return kind if kind in AGENT_SUBMITTED_KINDS else EnvelopeKind.AGENT_NOTE

    def _live_capture_authorization(
        self, conn, principal_id: str, kind: EnvelopeKind, scope_id: str
    ) -> str:
        """The authorization_id of the live consent row covering
        (principal, kind, scope) — the value bound as ``capture_proof``.
        The caller already passed ``require_capture_authorization``, so a
        matching row exists; a mismatch here is an integrity error."""
        ts = now_us()
        rows = repos_v3.query(
            conn,
            "capture_authorizations",
            {"principal_id": principal_id, "revoked_us": None},
        )
        for row in rows:
            exp = row.get("expires_us")
            if exp is not None and int(exp) <= ts:
                continue
            kinds = repos_v3.json_field(row, "allowed_kinds_json") or ()
            if kind.value not in kinds:
                continue
            scopes = repos_v3.json_field(row, "scope_ids_json") or []
            if scopes and scope_id not in scopes:
                continue
            return row["authorization_id"]
        raise _err(
            ErrorCode.INTEGRITY,
            "capture consent verified but no live row resolves",
        )

    def capture_submitted(
        self,
        principal_id: str,
        scope_id: str,
        content: Any,
        *,
        declared_type: str = "agent_note",
        title: Optional[str] = None,
        purpose: Optional[str] = None,
        session_id: str = "",
        external_id: Optional[str] = None,
        event_us: Optional[int] = None,
    ) -> str:
        """Explicit agent-submitted capture; returns ``source_id``.

        Requires an ``ingest`` grant AND live capture consent (§11.11):
        an approved tool call is not retention consent. The text is
        persisted verbatim with ``agent_generated`` trust — never human
        testimony — and screened before the transaction commits.
        """
        require_id(principal_id, "principal_id")
        require_id(scope_id, "scope_id")
        payload = (
            bytes(content)
            if isinstance(content, (bytes, bytearray, memoryview))
            else str(content).encode("utf-8")
        )
        if not payload:
            raise _err(ErrorCode.VALIDATION, "content must not be empty")
        if len(payload) > _MAX_CONTENT_BYTES:
            raise _err(
                ErrorCode.VALIDATION,
                f"content exceeds {_MAX_CONTENT_BYTES} bytes",
            )
        kind = self._submitted_kind(declared_type)
        declared = (
            declared_type.value
            if isinstance(declared_type, EnvelopeKind)
            else str(declared_type)
        )
        with self._store.tx() as conn:
            governance.seed_purposes(conn)
            caller = self._caller(principal_id, session_id)
            governance.authorize(
                conn, caller, scope_id, Verb.INGEST.value, purpose=purpose
            )
            governance.require_capture_authorization(
                conn, principal_id, kind, scope_id
            )
            proof = self._live_capture_authorization(
                conn, principal_id, kind, scope_id
            )
            envelope = SourceEnvelopeV3(
                kind=kind,
                scope_id=scope_id,
                actor_principal=principal_id,
                perspective=Perspective(
                    asserter=principal_id, observer=principal_id
                ),
                event_us=int(event_us) if event_us is not None else now_us(),
                receipt_us=0,
                content=payload,
                media_type="text/plain",
                trust_class=TrustClass.AGENT_GENERATED,
                capture_proof=proof,
                adapter_version=self._adapter_version,
                host_id=self._host_id,
                session_id=session_id,
                external_id=external_id,
                metadata={
                    "declared_type": declared,
                    "title": title or "",
                    "purpose": purpose or "",
                    "submitted_via": "api_v3.capture_submitted",
                },
            )
            # ingest_envelope performs the write-channel screen itself
            # (§34.01): one verdict, one security label linked onto the
            # envelope row, one ("source_envelope", …) quarantine hold on
            # suspicious/blocked content — all inside this transaction.
            # The facade deliberately does not screen again: a second
            # pass produced a duplicate label and a parallel ("source",…)
            # hold that quarantine_review could not see or resolve.
            receipt = ingest_envelope(conn, self._store, envelope)
            # V4-14.01: the receipt's obligation DAG commits in the same
            # transaction as the envelope + harvest job — capture
            # acceptance and its readiness rows are never torn apart.
            self._record_readiness(conn, receipt)
            return receipt.source_id

    # ------------------------------------------------------------------
    # §14 durable readiness
    # ------------------------------------------------------------------

    def _readiness_engine(self) -> "ReadinessEngine":
        """The durable readiness service over ``readiness_obligations`` —
        ``CAPABILITY_UNAVAILABLE`` on a pre-v4 store, never a silent no-op."""
        eng = getattr(self, "_readiness_cached", None)
        if eng is None:
            from ..readiness import ReadinessEngine

            eng = ReadinessEngine(self._store)
            self._readiness_cached = eng
        if not eng.available:
            raise _err(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "readiness_obligations table absent — store schema predates v4",
            )
        return eng

    def _record_readiness(self, conn: Any, receipt: Any) -> None:
        """Materialize the receipt's obligation DAG inside the capture tx.

        Records for every receipt id the source revision answers under —
        the ``cr_*`` envelope receipt plus the source-level ``rc_ingest``
        id — so ``wait_ready`` resolves whichever identity the caller
        holds (V4-14.02). On a pre-v4 store this is a no-op, not a
        capture-blocking failure.
        """
        try:
            from ..readiness import ReadinessEngine

            ReadinessEngine(self._store).ensure_for_source(
                conn, receipt.source_id, int(receipt.revision)
            )
        except VerbatimError:
            raise
        except Exception:
            # A store without the v4 table must not fail capture — the
            # lazy convergence path rebuilds obligations at first drain.
            return

    @staticmethod
    def _resolve_receipt_id(
        receipt_id: Optional[str],
        source_id: Optional[str],
        revision: int,
    ) -> str:
        """Callers may hold the ``cr_*`` envelope receipt id or just the
        ``source_id`` ``capture_submitted`` returned — the source-level
        ``rc_ingest`` receipt covers every capture path."""
        from ..readiness import ingest_receipt_id

        if receipt_id is not None:
            require_id(receipt_id, "receipt_id")
            return receipt_id
        if source_id is None:
            raise _err(
                ErrorCode.VALIDATION,
                "wait_ready needs receipt_id or source_id",
            )
        require_id(source_id, "source_id")
        return ingest_receipt_id(source_id, int(revision))

    def wait_ready(
        self,
        receipt_id: Optional[str] = None,
        *,
        source_id: Optional[str] = None,
        revision: int = 1,
        capabilities: Optional[Iterable[Any]] = None,
        timeout_s: Optional[float] = 30.0,
        principal_id: str,
        scope_id: str,
        purpose: Optional[str] = None,
        session_id: str = "",
    ) -> dict[str, Any]:
        """Wait on one capture receipt's durable obligation DAG (V4-14.03).

        Authorizes ``read`` on the receipt's scope before any snapshot
        leaves the store; a receipt outside the caller's authority is
        indistinguishable from unknown. Deadline expiry returns the
        honest pending snapshot — never an error, never invented
        readiness.
        """
        rid = self._resolve_receipt_id(receipt_id, source_id, revision)
        require_id(scope_id, "scope_id")
        require_id(principal_id, "principal_id")
        caller = self._caller(principal_id, session_id)
        with self._store.read() as conn:
            governance.authorize(
                conn, caller, scope_id, Verb.READ.value,
                purpose=purpose or "readiness",
            )
        deadline = (
            now_us() + int(timeout_s * 1_000_000)
            if timeout_s is not None
            else None
        )
        return self._readiness_engine().wait_ready(
            rid,
            capabilities,
            deadline_us=deadline,
            scope_id=scope_id,
        )

    def receipt_state(
        self,
        receipt_id: Optional[str] = None,
        *,
        source_id: Optional[str] = None,
        revision: int = 1,
        principal_id: str,
        scope_id: str,
        purpose: Optional[str] = None,
        session_id: str = "",
    ) -> dict[str, Any]:
        """One receipt's per-capability durable snapshot (V4-14.02)."""
        rid = self._resolve_receipt_id(receipt_id, source_id, revision)
        require_id(scope_id, "scope_id")
        require_id(principal_id, "principal_id")
        caller = self._caller(principal_id, session_id)
        with self._store.read() as conn:
            governance.authorize(
                conn, caller, scope_id, Verb.READ.value,
                purpose=purpose or "readiness",
            )
        return self._readiness_engine().receipt_state(
            rid, scope_id=scope_id
        )

    # ------------------------------------------------------------------
    # §27–§30 governed recall
    # ------------------------------------------------------------------

    def _recall_request(
        self,
        scope_id: str,
        query: str,
        principal_id: str,
        purpose: str,
        budget: Optional[Any],
    ) -> RecallRequestV3:
        opts: dict[str, Any] = {}
        if budget is not None:
            src = (
                budget
                if isinstance(budget, dict)
                else {
                    k: getattr(budget, k)
                    for k in dir(budget)
                    if not k.startswith("_")
                }
            )
            for key in (
                "max_items",
                "max_bytes",
                "target_tokens",
                "deadline_ms",
                "budget_tier",
                "modes",
                "memory_kinds",
                "entity_ids",
                "task_id",
                "session_id",
                "detail_tier",
                "manifest",
                "pack_mode",
            ):
                if src.get(key) is not None:
                    opts[key] = src[key]
        task = None
        if opts.get("task_id") or opts.get("session_id"):
            task = TaskContext(task_id=opts.get("task_id"))
        request = RecallRequestV3(
            query=query,
            scope_id=scope_id,
            caller_id=principal_id,
            purpose=purpose,
            modes=tuple(opts.get("modes") or ()),
            memory_kinds=tuple(opts.get("memory_kinds") or ()),
            task=task,
            entity_ids=tuple(opts.get("entity_ids") or ()),
            max_items=int(opts.get("max_items") or 8),
            max_bytes=int(opts.get("max_bytes") or 6000),
            target_tokens=int(opts.get("target_tokens") or 1536),
            deadline_ms=int(opts.get("deadline_ms") or 200),
            budget_tier=BudgetTier(opts.get("budget_tier") or "mid"),
            manifest=opts.get("manifest"),
            pack_mode=str(opts.get("pack_mode") or "standard"),
        )
        if opts.get("detail_tier") is not None:
            # Progressive disclosure (V4-32.05): the tier rides the
            # request object as an optional attribute — the frozen public
            # dataclass predates the field, so the channel is an explicit
            # attribute the pipeline resolves (never silently defaulted).
            object.__setattr__(
                request, "detail_tier", str(opts["detail_tier"])
            )
        return request

    @staticmethod
    def _byte_cap(request: RecallRequestV3) -> int:
        """The serialized-byte ceiling a result must honor: the request's
        ``max_bytes`` AND its token budget (≈4 bytes/token) — whichever
        binds tighter (§27.01)."""
        return max(0, min(request.max_bytes, request.target_tokens * 4))

    @staticmethod
    def _evidence_lane_declared(request: RecallRequestV3) -> bool:
        """True only when the caller explicitly opted into the raw
        archive/evidence lane via declared modes (V4-08.07)."""
        return bool(set(request.modes) & _EVIDENCE_LANE_MODES)

    def recall(
        self,
        scope_id: str,
        query: str,
        *,
        principal_id: str,
        purpose: Optional[str] = None,
        budget: Optional[Any] = None,
        session_id: str = "",
    ) -> RecallResultV3:
        """Governed recall; returns ``RecallResultV3``.

        Authorization is asserted before any lane runs (``read`` verb on
        the scope). The W9 pipeline is preferred through a lazy import —
        the facade stays importable before it lands.

        Raw evidence is reachable ONLY through the explicit
        archive/evidence lane: declare a mode in ``budget["modes"]``
        (``archive``/``evidence``/``browse``) or call
        ``browse_evidence``. Without a declaration an empty, held, or
        budget-exhausted pipeline is an honest abstain — never a silent
        raw-source bypass (V4-08.07, C13).
        """
        require_id(scope_id, "scope_id")
        require_id(principal_id, "principal_id")
        if not isinstance(query, str) or not query.strip():
            raise _err(ErrorCode.VALIDATION, "recall requires a query")
        declared_purpose = purpose or "recall"
        caller = self._caller(principal_id, session_id)
        with self._store.read() as conn:
            governance.authorize(
                conn, caller, scope_id, Verb.READ.value, purpose=declared_purpose
            )

        request = self._recall_request(
            scope_id, query, principal_id, declared_purpose, budget
        )
        v3_result = self._try_recall_v3(request)
        if not self._evidence_lane_declared(request):
            # No archive/evidence declaration → the derived pipeline's own
            # answer (possibly an abstention) is the whole answer.
            if v3_result is not None:
                return v3_result
            return RecallResultV3(
                packs=(),
                omitted=0,
                warnings=("retrieval_v3_unavailable",),
                capabilities={
                    "degraded": ["retrieval_v3"],
                    "lanes": {},
                },
                abstained=True,
            )
        # Explicit archive/evidence declaration: the governed lane shares
        # the request's item/byte/token budget with whatever the derived
        # pipeline already delivered.
        used_items = (
            sum(len(p.items) for p in v3_result.packs) if v3_result else 0
        )
        used_bytes = (
            sum(int(p.serialized_bytes or 0) for p in v3_result.packs)
            if v3_result
            else 0
        )
        ev_pack, ev_omitted, ev_warnings = self._evidence_lane(
            request,
            caller,
            item_cap=max(0, request.max_items - used_items),
            byte_cap=max(0, self._byte_cap(request) - used_bytes),
        )
        if v3_result is None:
            return RecallResultV3(
                packs=(ev_pack,) if ev_pack is not None else (),
                omitted=ev_omitted,
                warnings=tuple(dict.fromkeys(ev_warnings)),
                capabilities={
                    "degraded": ["retrieval_v3"],
                    "lanes": {"archive_evidence": "ok"},
                },
                abstained=ev_pack is None,
            )
        packs = v3_result.packs + ((ev_pack,) if ev_pack is not None else ())
        lanes = dict(v3_result.capabilities.get("lanes") or {})
        lanes["archive_evidence"] = "ok" if ev_pack is not None else "empty"
        capabilities = dict(v3_result.capabilities)
        capabilities["lanes"] = lanes
        return RecallResultV3(
            packs=packs,
            omitted=v3_result.omitted + ev_omitted,
            warnings=tuple(
                dict.fromkeys(tuple(v3_result.warnings) + tuple(ev_warnings))
            ),
            capabilities=capabilities,
            decision_id=v3_result.decision_id,
            projection_generation=v3_result.projection_generation,
            abstained=not any(len(p.items) for p in packs),
        )

    def browse_evidence(
        self,
        scope_id: str,
        query: str = "",
        *,
        principal_id: str,
        purpose: Optional[str] = None,
        budget: Optional[Any] = None,
        session_id: str = "",
    ) -> RecallResultV3:
        """Explicit archive/evidence browsing (V4-08.07); returns
        ``RecallResultV3`` with an ``evidence_bundle`` pack.

        This is the ONLY way raw source revisions ship: the caller asks
        for it by name. The lane authorizes ``read`` on the scope,
        withholds purge-suppressed and quarantine-held evidence (source,
        covering ``source_envelope``, cascade — V3-14.10), ships exact
        bytes only under the ``quote`` verb (metadata-only items
        otherwise), and serializes inside the request's
        ``max_items``/``max_bytes``/``target_tokens`` bounds. An empty or
        fully-withheld scan is an honest abstention carrying
        ``held_evidence_withheld``/``no_evidence_matched`` warnings —
        never a silent payload.
        """
        require_id(scope_id, "scope_id")
        require_id(principal_id, "principal_id")
        if query is not None and not isinstance(query, str):
            raise _err(ErrorCode.VALIDATION, "query must be a string")
        declared_purpose = purpose or "recall"
        caller = self._caller(principal_id, session_id)
        with self._store.read() as conn:
            governance.authorize(
                conn, caller, scope_id, Verb.READ.value, purpose=declared_purpose
            )
        request = self._recall_request(
            scope_id, (query or "").strip() or "*", principal_id,
            declared_purpose, budget,
        )
        ev_pack, ev_omitted, ev_warnings = self._evidence_lane(
            request,
            caller,
            item_cap=request.max_items,
            byte_cap=self._byte_cap(request),
        )
        return RecallResultV3(
            packs=(ev_pack,) if ev_pack is not None else (),
            omitted=ev_omitted,
            warnings=tuple(dict.fromkeys(ev_warnings)),
            capabilities={
                "degraded": [] if ev_pack is not None else ["archive_evidence"],
                "lanes": {
                    "archive_evidence": "ok" if ev_pack is not None else "empty"
                },
            },
            abstained=ev_pack is None,
        )

    def expand(
        self,
        expand_ref: str,
        *,
        principal_id: str,
        detail_tier: Optional[str] = None,
        session_id: str = "",
    ) -> RecallResultV3:
        """Expand one delivered item to a deeper disclosure tier
        (V4-32.09); returns a one-pack ``RecallResultV3``.

        The ``expand`` ref carried on a delivered item is bound to the
        original caller, scope, purpose, object revision, and expiry.
        Expansion re-runs CURRENT authorization — a revoked grant,
        quarantined/purged/superseded object, moved epoch, expired or
        forged token, or a caller other than the original recipient all
        deny with the indistinguishable ``NOT_FOUND_OR_UNAUTHORIZED``.
        """
        require_id(principal_id, "principal_id")
        if not isinstance(expand_ref, str) or not expand_ref:
            raise _err(ErrorCode.VALIDATION, "expand requires a ref")
        try:
            from ..retrieval.v3.recall import expand_item  # type: ignore
        except Exception:
            raise _err(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "expansion requires the v3 retrieval pipeline",
            )
        return expand_item(
            self._store,
            expand_ref,
            caller_id=principal_id,
            detail_tier=detail_tier or "l2",
            cfg=self._config,
        )

    def _try_recall_v3(self, request: RecallRequestV3) -> Optional[RecallResultV3]:
        """Lazy route into W9's pipeline; None when unavailable."""
        try:
            from ..retrieval.v3.recall import recall_v3  # type: ignore
        except Exception:
            return None
        try:
            result = recall_v3(self._store, request, cfg=self._config)
        except ImportError:
            return None  # optional lane dependency absent inside the module
        return result if isinstance(result, RecallResultV3) else None

    @staticmethod
    def _truncate_utf8(text: str, limit: int) -> str:
        """Bound ``text`` to ``limit`` UTF-8 bytes without splitting a
        codepoint."""
        raw = text.encode("utf-8")
        if len(raw) <= limit:
            return text
        return raw[:limit].decode("utf-8", "ignore")

    def _evidence_lane(
        self,
        request: RecallRequestV3,
        caller: governance.CallerV3,
        *,
        item_cap: int,
        byte_cap: int,
    ) -> tuple:
        """The governed archive/evidence lane (V4-08.07).

        Runs only on explicit declaration — ``recall`` never routes here
        implicitly. The scan applies the same authority (``read``/
        ``quote``), purge suppression, quarantine cascade (source +
        covering ``source_envelope`` holds — V3-14.10), and serialized
        byte/item budgets the derived pipeline enforces; it is the raw
        lane INSIDE the governed surface, not around it.

        Returns ``(pack_or_none, omitted, warnings)``; ``None`` when
        nothing eligible matched or the budget was already exhausted —
        an honest empty, never a bypass.
        """
        terms = [
            t for t in _TERM_RE.findall(request.query.lower()) if len(t) > 1
        ]
        warnings: list[str] = [
            "archive_evidence_lane",
            "evidence_pending_derivation",
        ]
        with self._store.read() as conn:
            try:
                verbs = governance.effective_verbs(
                    conn, caller, request.scope_id
                )
            except VerbatimError:
                verbs = frozenset()
            can_quote = Verb.QUOTE.value in verbs or Verb.QUOTE in verbs
            if not can_quote:
                warnings.append("quote_not_granted")
            suppressed = self._suppressed_sources(conn, request.scope_id)
            suppressed_revs = self._suppressed_revisions(
                conn, request.scope_id
            )
            # The projection's normalized ``tokens`` ride along on the
            # same query — ``normalize_text`` preserves every ASCII word
            # char contiguously, so a ``_TERM_RE`` term that is absent
            # from ``tokens`` can never satisfy the post-fetch
            # ``t in low`` check. Rows that fail the probe are skipped
            # before the verified-payload read (the same skip-the-read
            # discipline the suppression/held filters apply); rows with
            # no projection row pass through to the read unchanged.
            from ..storage.repos import has_table

            proj = has_table(conn, "source_lexical_projection")
            # Terms that fold (``'``/unicode/punctuation) are excluded
            # from the probe: ``t in low`` does not imply ``t in
            # tokens`` for them. Normalization-stable terms are pushed
            # into SQL as LIKE prefilters — LIKE's ASCII case-folding
            # can only admit a SUPERSET of the case-sensitive ``t in
            # row_tokens`` probe the loop below re-applies verbatim, so
            # no row the probe would keep is filtered out here.
            stable_terms = terms
            if proj and terms:
                from ..enrichment.normalize import normalize_text

                stable_terms = [
                    t for t in terms if normalize_text(t) == t
                ]
            like_pred = ""
            like_params: list = []
            if proj and stable_terms:
                like_pred = (
                    " AND (p.tokens IS NULL OR ("
                    + " AND ".join(
                        "p.tokens LIKE ? ESCAPE '\\'"
                        for _ in stable_terms
                    )
                    + "))"
                )
                like_params = [
                    "%"
                    + str(t)
                    .replace("\\", "\\\\")
                    .replace("%", "\\%")
                    .replace("_", "\\_")
                    + "%"
                    for t in stable_terms
                ]
            rows = conn.execute(
                "SELECT s.source_id, s.origin, s.external_id,"
                " sr.revision, sr.event_us, sr.captured_us"
                + (", p.tokens" if proj else ", NULL")
                + " FROM sources s"
                " JOIN source_revisions sr ON sr.source_id = s.source_id"
                + (
                    " LEFT JOIN source_lexical_projection p"
                    "  ON p.source_id = s.source_id"
                    "  AND p.revision = sr.revision"
                    "  AND p.scope_id = s.scope_id"
                    if proj
                    else ""
                )
                + " WHERE s.scope_id = ?"
                + like_pred
                + " ORDER BY sr.captured_us DESC, s.source_id, sr.revision"
                " LIMIT ?",
                (request.scope_id, *like_params, _EVIDENCE_SCAN_LIMIT),
            ).fetchall()
            epoch = governance.current_epoch(conn, request.scope_id)
            # Quarantine cascade (V3-14.10) — mirrors retrieval/v3/union.py:
            # a revision is withheld when a hold sits on the source itself
            # OR on any source_envelope covering it. The write-channel
            # screen (evidence.envelopes._maybe_screen) and the SDK hold
            # envelopes, not sources — checking only ("source", …) shipped
            # held payloads verbatim (audit F1).
            held = self._held_objects(conn, rows)
            items: list[PackItem] = []
            omitted = 0
            withheld = 0
            truncated = False
            used = 0
            # Suppression/quarantine decide read eligibility first —
            # only rows that would reach the per-row payload() call are
            # fetched, so a withheld revision's bytes are never read and
            # its corruption can never fail the lane (unchanged).
            # Projection-token prefilter: a *normalization-stable* term
            # (``normalize_text(t) == t`` — pure ASCII alnum) that is
            # absent from the stored tokens cannot appear contiguously
            # in the payload either, so the post-fetch ``t in low``
            # check would reject this row — its bytes are never read.
            # Terms that fold (``'``/unicode/punctuation) are excluded
            # from the probe: ``t in low`` does not imply ``t in
            # tokens`` for them. NULL tokens (unprojected revisions,
            # pre-v5 stores) still take the read path. The identical
            # probe is re-applied here on the SQL-prefiltered rows —
            # LIKE admits a superset only, never the final word.
            eligible: list[tuple] = []
            for row in rows:
                source_id, revision = row[0], row[3]
                if source_id in suppressed:
                    continue
                if f"{source_id}:{int(revision)}" in suppressed_revs:
                    continue
                if held.get((source_id, int(revision))):
                    withheld += 1
                    continue
                row_tokens = row[6]
                if (
                    stable_terms
                    and row_tokens is not None
                    and not all(t in row_tokens for t in stable_terms)
                ):
                    continue
                eligible.append(row)
            # Verified byte reads batched under this snapshot — the same
            # persisted-hmac re-check as SourcesRepo.payload per row
            # (V4-07.02/V4-08.03); a tampered revision still fails the
            # whole lane STORE_CORRUPT rather than serving forged
            # evidence, and absent/purged bytes are simply skipped.
            verified, corrupt = SourcesRepo(self._store).payload_many(
                [(r[0], int(r[3])) for r in eligible], conn=conn
            )
            if corrupt:
                bad_sid, bad_rev = corrupt[0]
                raise VerbatimError(
                    ErrorCode.STORE_CORRUPT,
                    f"source {bad_sid}@{bad_rev} payload fails integrity check",
                )
            # Per-row eligibility and budget accounting first (unchanged
            # order/caps); envelope/label metadata for the surviving rows
            # is fetched in two batched passes by _evidence_items instead
            # of ~4 queries per item.
            pending: list[tuple] = []
            for row in eligible:
                source_id, revision = row[0], row[3]
                payload = verified.get((source_id, int(revision)))
                if payload is None or len(payload) == 0:
                    continue
                text = bytes(payload).decode("utf-8", "replace")
                low = text.lower()
                if terms and not all(t in low for t in terms):
                    continue
                if len(pending) >= item_cap:
                    omitted += 1
                    continue
                remaining = byte_cap - used
                if remaining <= 0:
                    omitted += 1
                    continue
                served = len(text.encode("utf-8")) if can_quote else 0
                if can_quote and served > remaining:
                    text = self._truncate_utf8(text, remaining)
                    truncated = True
                    served = len(text.encode("utf-8"))
                pending.append((row, text))
                # item.text is "" when quote is not granted — the byte
                # budget counts only served payload bytes (unchanged).
                used += served
            if pending:
                items = self._evidence_items(
                    conn, caller, epoch, pending, can_quote
                )
            if withheld:
                warnings.append("held_evidence_withheld")
            if truncated:
                warnings.append("items_truncated_to_budget")
            if not items:
                warnings.append("no_evidence_matched")
            pack = None
            if items:
                pack = ContextPack(
                    kind=PackKind.EVIDENCE_BUNDLE,
                    items=tuple(items),
                    tokens=max(1, used // 4),
                    serialized_bytes=used,
                    warnings=tuple(dict.fromkeys(warnings)),
                )
            return pack, omitted, warnings

    def _evidence_items(
        self, conn, caller, epoch, pending, can_quote
    ) -> list[PackItem]:
        """Batched ``PackItem`` assembly for the surviving ``(row, text)``
        pairs — two metadata passes for the whole pack instead of ~4
        queries per item (envelope get + event/span anchors inside
        ``receipt_for_envelope`` + label get).

        Semantics preserved:

        - The envelope pick replicates ``repos_v3.get``'s unordered
          ``LIMIT 1``: ``query_in(..., order_rowid=True)`` returns the
          lowest-rowid envelope carrying ``(source_id, revision)``, which
          is also the row the dead-code fallback (all envelopes for the
          source, first revision match) would have found.
        - ``receipt_id`` is recomputed as ``receipt_id_for(source_id,
          revision, envelope_kind)`` — the identical pure digest
          ``mint_receipt`` stamps on the persisted row. The anchoring
          queries ``receipt_for_envelope`` runs only fill receipt fields
          this lane never reads, and the ``VerbatimError`` fallback is
          reproduced by applying ``CaptureReceipt``'s ``require_id``
          checks (``envelope_id``/``source_id``/``scope_id``) up front.
        - ``security.label_for``'s ``require_id`` validation still
          propagates ``VALIDATION`` for malformed label ids.
        """
        sids = list(dict.fromkeys(r[0] for r, _t in pending))
        env_pick: dict[tuple, dict] = {}
        for e in repos_v3.query_in(
            conn, "source_envelopes", "source_id", sids, order_rowid=True
        ):
            env_pick.setdefault((e["source_id"], int(e["revision"])), e)
        # Metadata + label ids per picked envelope; validating every
        # referenced id up front preserves label_for's failure on
        # malformed metadata (the lane aborts the same way).
        meta_by_pair: dict[tuple, dict] = {}
        label_ids: list[str] = []
        for row, _t in pending:
            key = (row[0], int(row[3]))
            env_row = env_pick.get(key)
            if env_row is None or key in meta_by_pair:
                continue
            meta = repos_v3.json_field(env_row, "metadata_json", {}) or {}
            meta_by_pair[key] = meta
            lid = meta.get("security_label_id")
            if isinstance(lid, str) and lid and lid not in label_ids:
                require_id(lid, "label_id")
                label_ids.append(lid)
        label_rows: dict[str, dict] = {}
        for lrow in repos_v3.query_in(
            conn, "security_labels", "label_id", label_ids
        ):
            lrow["findings"] = repos_v3.json_field(
                lrow, "findings_json", []
            )
            label_rows[lrow["label_id"]] = lrow
        items: list[PackItem] = []
        for row, text in pending:
            source_id, external_id, revision = (
                row[0], row[2], int(row[3]),
            )
            env_row = env_pick.get((source_id, revision))
            receipt_id = f"rc_evidence:{source_id}:{revision}"
            trust = TrustClass.UNKNOWN.value
            perspective = Perspective()
            if env_row is not None:
                try:
                    require_id(env_row["envelope_id"], "envelope_id")
                    require_id(env_row["source_id"], "source_id")
                    require_id(env_row["scope_id"], "scope_id")
                except VerbatimError:
                    pass
                else:
                    receipt_id = receipt_id_for(
                        source_id, revision, env_row["envelope_kind"]
                    )
                trust = env_row.get("trust_class") or TrustClass.UNKNOWN.value
                if env_row.get("actor_principal"):
                    perspective = Perspective(
                        asserter=env_row["actor_principal"],
                        observer=env_row["actor_principal"],
                    )
            label = SecurityLabel(source_trust=TrustClass(trust))
            meta = meta_by_pair.get((source_id, revision))
            if meta is not None:
                lid = meta.get("security_label_id")
                if isinstance(lid, str) and lid:
                    lrow = label_rows.get(lid)
                    if lrow is not None:
                        label = SecurityLabel(
                            source_trust=TrustClass(lrow["source_trust"]),
                            content_form=lrow["content_form"],
                            attack_risk=lrow["attack_risk"],
                            review_state=lrow["review_state"],
                            findings=tuple(lrow.get("findings") or ()),
                            method=lrow["method"],
                            rules_revision=lrow["rules_revision"],
                        )
            handle = InfluenceHandle(
                handle_id=new_id(),
                receipt_id=receipt_id,
                caller_id=caller.principal_id,
                epoch=epoch,
                pack=PackKind.EVIDENCE_BUNDLE,
                object_kind="source",
                object_id=source_id,
                revision=revision,
            )
            items.append(
                PackItem(
                    handle=handle,
                    text=text if can_quote else "",
                    lifecycle="accepted",
                    freshness=FreshnessClass.UNKNOWN,
                    security=label,
                    perspective=perspective,
                    derived=False,
                    proof_count=0,
                    # Raw evidence pending derivation — the archive lane
                    # labels every item verify-on-delivery; it is not a
                    # settled answer.
                    verify_recommended=True,
                )
            )
        return items

    #: IN() fan-out bound while resolving held envelope/span ids into
    #: ``(source_id, revision)`` pairs.
    _HELD_IN_CHUNK = 400

    def _held_objects(self, conn, rows) -> dict:
        """``(source_id, revision) -> True`` for revisions any pending or
        suppressed quarantine hold covers — the source hold itself or a
        hold on any covering ``source_envelopes``/``spans`` row (the
        cascade the union lane applies to claims, V3-14.10).

        The cascade lookups run *inverted*: only the held envelope/span
        ids can cover a candidate revision, so two primary-key IN()
        probes bounded by the hold count replace the previous fan-out
        over every candidate source — identical ``out`` set, and zero
        cascade queries when no envelope/span hold exists (the common
        case). The excluding-state row set is read directly each call
        (journal writes bump ``PRAGMA data_version`` every search, so a
        version-keyed memo never survives one query anyway).
        """
        from ..security.quarantine import EXCLUDING_STATES

        src_revs = {(r[0], int(r[3])) for r in rows}
        if not src_revs:
            return {}
        states = sorted(EXCLUDING_STATES)
        phq = ",".join("?" for _ in states)
        held_rows = conn.execute(
            "SELECT object_kind, object_id, revision FROM quarantine"
            f" WHERE state IN ({phq})",
            states,
        ).fetchall()
        if not held_rows:
            return {}
        src_held: set = set()
        env_held: set = set()
        span_held: set = set()
        for kind, oid, rev in held_rows:
            rev_i = int(rev)
            if kind == "source":
                src_held.add((oid, rev_i))
            elif kind == "source_envelope":
                env_held.add((oid, rev_i))
            elif kind == "span":
                span_held.add((oid, rev_i))
        out: dict[tuple, bool] = {}
        for sid, rev in src_revs:
            if (sid, rev) in src_held:
                out[(sid, rev)] = True
        # A span-level hold withholds the containing revision's bytes too:
        # the lane serves whole revisions and cannot splice out a held
        # byte range, so fail closed (V4-36.03 — same resolution export
        # applies in ``export._source_revision_held``). Envelope/span
        # holds resolve through the object's own primary key — the same
        # join the old candidate-side fan-out computed, traversed from
        # the (usually empty) hold side.
        if env_held:
            eids = sorted({e for e, _r in env_held})
            for i in range(0, len(eids), self._HELD_IN_CHUNK):
                chunk = eids[i : i + self._HELD_IN_CHUNK]
                ph = ",".join("?" for _ in chunk)
                for eid, sid, rev in conn.execute(
                    "SELECT envelope_id, source_id, revision"
                    " FROM source_envelopes"
                    f" WHERE envelope_id IN ({ph})",
                    chunk,
                ).fetchall():
                    rev_i = int(rev)
                    if (eid, rev_i) in env_held and (sid, rev_i) in src_revs:
                        out[(sid, rev_i)] = True
        if span_held:
            spids = sorted({s for s, _r in span_held})
            for i in range(0, len(spids), self._HELD_IN_CHUNK):
                chunk = spids[i : i + self._HELD_IN_CHUNK]
                ph = ",".join("?" for _ in chunk)
                for sp_id, sid, rev in conn.execute(
                    "SELECT span_id, source_id, revision FROM spans"
                    f" WHERE span_id IN ({ph})",
                    chunk,
                ).fetchall():
                    rev_i = int(rev)
                    if (sp_id, rev_i) in span_held and (sid, rev_i) in src_revs:
                        out[(sid, rev_i)] = True
        return out

    @staticmethod
    def _suppressed_sources(conn, scope_id: str) -> set:
        """Source ids under an active purge suppression (§36.03)."""
        ph = ",".join("?" for _ in _SUPPRESSING_PURGE_STATES)
        rows = conn.execute(
            "SELECT DISTINCT pt.object_id FROM purge_targets pt"
            " JOIN purges p ON p.purge_id = pt.purge_id"
            " WHERE pt.object_kind = 'source'"
            f" AND p.state IN ({ph}) AND p.scope_id = ?",
            [*_SUPPRESSING_PURGE_STATES, scope_id],
        ).fetchall()
        return {r[0] for r in rows}

    @staticmethod
    def _suppressed_revisions(conn, scope_id: str) -> set:
        """``source_revision`` tombstone keys (``"<source_id>:<rev>"``)
        under an active purge suppression — the revision-scoped half of
        the predicate the kernel's ``_suppressed_refs`` applies
        (V4-07.07)."""
        ph = ",".join("?" for _ in _SUPPRESSING_PURGE_STATES)
        rows = conn.execute(
            "SELECT DISTINCT pt.object_id FROM purge_targets pt"
            " JOIN purges p ON p.purge_id = pt.purge_id"
            " WHERE pt.object_kind = 'source_revision'"
            f" AND p.state IN ({ph}) AND p.scope_id = ?",
            [*_SUPPRESSING_PURGE_STATES, scope_id],
        ).fetchall()
        return {r[0] for r in rows}

    # ------------------------------------------------------------------
    # §47 inspect / explain — evidence-plane metadata, never payload
    # ------------------------------------------------------------------

    def inspect_evidence(self, source_id: str, *, principal_id: str) -> dict:
        """Lineage report for one source: revisions, envelopes, security
        labels, quarantine holds, spans, receipts, vault placeholders.

        Authorization precedes identifier dereference semantics: the
        source's scope is resolved, then ``read`` is authorized on it —
        a missing row and a denied row are the same public outcome
        (§10.05). Payload bytes and vault plaintext never leave this
        boundary.
        """
        require_id(source_id, "source_id")
        require_id(principal_id, "principal_id")
        with self._store.read() as conn:
            src = conn.execute(
                "SELECT scope_id, origin, external_id, source_kind,"
                " speaker_id, created_us FROM sources WHERE source_id = ?",
                (source_id,),
            ).fetchone()
            if src is None:
                _deny("source not found or unauthorized")
            scope_id = src[0]
            governance.authorize(
                conn,
                self._caller(principal_id),
                scope_id,
                Verb.READ.value,
            )
            revisions = [
                {
                    "revision": int(r[0]),
                    "accepted_bytes": int(r[1] or 0),
                    "event_us": int(r[2] or 0),
                    "captured_us": int(r[3] or 0),
                    "provenance": r[4],
                }
                for r in conn.execute(
                    "SELECT revision, length(payload), event_us, captured_us,"
                    " provenance FROM source_revisions"
                    " WHERE source_id = ? ORDER BY revision",
                    (source_id,),
                ).fetchall()
            ]
            env_rows = repos_v3.query(
                conn, "source_envelopes", {"source_id": source_id}
            )
            envelopes: list[dict[str, Any]] = []
            label_ids: list[str] = []
            for e in env_rows:
                meta = repos_v3.json_field(e, "metadata_json", {}) or {}
                lid = meta.get("security_label_id")
                if isinstance(lid, str) and lid:
                    label_ids.append(lid)
                envelopes.append(
                    {
                        "envelope_id": e["envelope_id"],
                        "revision": int(e["revision"]),
                        "envelope_kind": e["envelope_kind"],
                        "actor_principal": e["actor_principal"],
                        "trust_class": e["trust_class"],
                        "capture_proof": e["capture_proof"],
                        "event_us": e["event_us"],
                        "receipt_us": e["receipt_us"],
                        "media_type": e["media_type"],
                        "host_id": e["host_id"],
                        "session_id": e["session_id"],
                        "task_id": e["task_id"],
                        "adapter_version": e["adapter_version"],
                        "artifact_ref": e["artifact_ref"],
                        "metadata": meta,
                    }
                )
            labels = []
            for lid in dict.fromkeys(label_ids):
                row = security.label_for(conn, lid)
                if row is not None:
                    labels.append(row)
            spans = [
                {
                    "span_id": r[0],
                    "revision": int(r[1]),
                    "start_byte": int(r[2]),
                    "end_byte": int(r[3]),
                    "harvester_version": r[4],
                }
                for r in conn.execute(
                    "SELECT span_id, revision, start_byte, end_byte,"
                    " harvester_version FROM spans"
                    " WHERE source_id = ? ORDER BY revision, start_byte",
                    (source_id,),
                ).fetchall()
            ]
            # Every hold on the evidence chain — the source revision, the
            # covering source_envelope rows (the write-channel screen's
            # hold kind), and this source's spans. Listing only
            # ("source", …) hid SDK/envelope-path holds from review (F3).
            quarantines = []
            seen_holds: set = set()
            hold_refs = [
                ("source", source_id, rev["revision"]) for rev in revisions
            ] + [
                ("source_envelope", e["envelope_id"], int(e["revision"]))
                for e in env_rows
            ] + [
                ("span", s["span_id"], s["revision"]) for s in spans
            ]
            for kind, oid, rev in hold_refs:
                key = (kind, oid, rev)
                if key in seen_holds:
                    continue
                q = security.get_quarantine(conn, kind, oid, rev)
                if q is not None:
                    seen_holds.add(key)
                    quarantines.append(q)
            receipts = []
            for e in env_rows:
                try:
                    receipts.append(
                        receipt_for_envelope(
                            conn, e, dedup_key=src[2] or ""
                        ).to_dict()
                    )
                except VerbatimError:
                    continue
            view_ids = [source_id] + [s["span_id"] for s in spans]
            vault_refs = [
                {"placeholder": r[0], "view_id": r[1],
                 "start_byte": int(r[2]), "end_byte": int(r[3])}
                for r in conn.execute(
                    "SELECT placeholder, view_id, start_byte, end_byte"
                    " FROM vault_refs WHERE scope_id = ?"
                    f" AND view_id IN ({','.join('?' for _ in view_ids)})",
                    [scope_id, *view_ids],
                ).fetchall()
            ]
            suppressed = source_id in self._suppressed_sources(conn, scope_id)
            return {
                "source_id": source_id,
                "scope_id": scope_id,
                "origin": src[1],
                "external_id": src[2],
                "source_kind": src[3],
                "speaker_id": src[4],
                "created_us": src[5],
                "suppressed": suppressed,
                "revisions": revisions,
                "envelopes": envelopes,
                "security_labels": labels,
                "quarantine": quarantines,
                "spans": spans,
                "receipts": receipts,
                "vault_refs": vault_refs,
            }

    # ------------------------------------------------------------------
    # §34.03 quarantine review
    # ------------------------------------------------------------------

    @staticmethod
    def _source_chain_refs(conn, source_id: str, revision: int) -> list:
        """Every quarantine ref covering one source revision: each
        ``source_envelopes`` row for it plus the source row itself.
        Envelope refs lead — the write-channel screen's hold kind is the
        label's primary target; the ``("source", …)`` ref covers holds
        opened by SCREEN jobs or earlier facade versions."""
        refs = [
            ("source_envelope", e["envelope_id"], int(revision))
            for e in repos_v3.query(
                conn,
                "source_envelopes",
                {"source_id": source_id, "revision": int(revision)},
                order="envelope_id",
            )
        ]
        refs.append(("source", source_id, int(revision)))
        return refs

    def _chain_refs(self, conn, ref) -> list:
        """Expand one resolved ref into the whole source-revision chain;
        non-source kinds decide alone."""
        kind, oid, rev = ref
        if kind == "source_envelope":
            env = repos_v3.get(
                conn, "source_envelopes", {"envelope_id": oid}
            )
            if env is None:
                return [ref]
            return self._source_chain_refs(
                conn, env["source_id"], int(env["revision"])
            )
        if kind == "source":
            return self._source_chain_refs(conn, oid, int(rev))
        return [ref]

    def _refs_for_label(
        self, conn, label_id: str, scope_id: str
    ) -> Optional[list]:
        """Resolve the quarantine refs covering the evidence chain a
        label was attached to.

        Primary link: envelope metadata's ``security_label_id`` recorded
        by ``ingest_envelope`` at capture. Fallback: a single pending
        hold in the label's scope (labels without an envelope link —
        e.g. an adapter's own screen label) expands the same way.
        Ambiguous or absent targets resolve to None — the caller denies
        identically to unauthorized.
        """
        rows = conn.execute(
            "SELECT DISTINCT source_id, revision FROM source_envelopes"
            " WHERE scope_id = ?"
            " AND json_extract(metadata_json, '$.security_label_id') = ?"
            " ORDER BY source_id",
            (scope_id, label_id),
        ).fetchall()
        if rows:
            refs: list = []
            seen: set = set()
            for sid, rev in rows:
                for ref in self._source_chain_refs(conn, sid, int(rev)):
                    if ref not in seen:
                        seen.add(ref)
                        refs.append(ref)
            return refs
        pending = security.pending_items(conn, scope_id)
        if len(pending) == 1:
            q = pending[0]
            return self._chain_refs(
                conn, (q["object_kind"], q["object_id"], int(q["revision"]))
            )
        return None

    def quarantine_review(
        self,
        security_label_id: str,
        *,
        principal_id: str,
        decision: str,
        reviewer_note: Optional[str] = None,
    ) -> dict:
        """Apply a quarantine decision under the ``review`` verb.

        The label's scope authorizes the call; the held object is then
        resolved and released/suppressed/purged through the security
        layer — origin and findings are never rewritten (§14.01), so a
        release never relabels agent content as human testimony.
        """
        require_id(security_label_id, "security_label_id")
        require_id(principal_id, "principal_id")
        from ..security.quarantine import DECISIONS

        if decision not in DECISIONS:
            raise _err(
                ErrorCode.VALIDATION,
                f"decision must be one of {sorted(DECISIONS)}",
            )
        with self._store.tx() as conn:
            label = repos_v3.get(
                conn, "security_labels", {"label_id": security_label_id}
            )
            if label is None:
                _deny("security label not found or unauthorized")
            scope_id = label["scope_id"]
            governance.authorize(
                conn,
                self._caller(principal_id),
                scope_id,
                Verb.REVIEW.value,
            )
            refs = self._refs_for_label(conn, security_label_id, scope_id)
            if not refs:
                _deny("no quarantine hold resolves for this label")
            payload: dict[str, Any] = {}
            if reviewer_note:
                payload["rationale"] = str(reviewer_note)[:2000]
            # One decision covers the WHOLE evidence chain in this
            # transaction — a release that leaves any covering hold
            # pending would report success while the item stays
            # invisible (F2). Purged tombstones are never re-decided:
            # recorded content destruction stands (§36).
            decided: list = []
            for ref in refs:
                qrow = security.get_quarantine(conn, ref[0], ref[1], ref[2])
                if qrow is None or qrow.get("state") == "purged":
                    continue
                if decision == "release":
                    row = security.release(conn, ref, principal_id, payload)
                elif decision == "suppress":
                    row = security.suppress(conn, ref, principal_id, payload)
                else:
                    row = security.mark_purged(
                        conn, ref, principal_id, payload
                    )
                decided.append((ref, row))
            if not decided:
                _deny("no quarantine hold resolves for this label")
            label_state = (
                "released" if decision == "release" else "quarantined"
            )
            security.update_review_state(
                conn, security_label_id, label_state
            )
            primary = decided[0]
            return {
                "security_label_id": security_label_id,
                "object_kind": primary[0][0],
                "object_id": primary[0][1],
                "revision": primary[0][2],
                "decision": decision,
                "state": primary[1].get("state"),
                "decided_by": principal_id,
                "decided_refs": [
                    {
                        "object_kind": r[0],
                        "object_id": r[1],
                        "revision": r[2],
                        "state": rrow.get("state"),
                    }
                    for r, rrow in decided
                ],
            }

    def quarantine_pending(
        self,
        scope_id: str,
        *,
        principal_id: str,
        limit: int = 100,
    ) -> list:
        """Open quarantine holds awaiting review on ``scope_id`` (§34.03).

        The review queue is an operator surface — authorized under the
        ``review`` verb, the same authority ``quarantine_review``
        requires. Every pending hold is listed with its reason codes and
        findings, whatever object kind it covers (``source``,
        ``source_envelope``, ``span``, …) — model-visible transports
        bound without ``review`` deny identically to a missing scope.
        """
        require_id(scope_id, "scope_id")
        require_id(principal_id, "principal_id")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise _err(ErrorCode.VALIDATION, "limit must be a positive int")
        with self._store.read() as conn:
            governance.authorize(
                conn,
                self._caller(principal_id),
                scope_id,
                Verb.REVIEW.value,
            )
            return security.pending_items(conn, scope_id, limit=limit)

    # ------------------------------------------------------------------
    # §62.04 / V4-50 capability reporting — provider-observed, never
    # inferred from config flags or importable modules
    # ------------------------------------------------------------------

    @staticmethod
    def _module_present(dotted: str) -> bool:
        try:
            __import__(dotted)
            return True
        except Exception:
            return False

    @staticmethod
    def _cap(
        name: str,
        rung: CapabilityRung,
        reason: Optional[str] = None,
        **details: Any,
    ) -> dict:
        """Serialize a provider-owned ``CapabilityReport`` (V4-50.01):
        the rung is the highest *observed* state — never config-implied.
        ``available`` means the component's own probe says it can serve
        (``healthy`` or better), not that a flag is set."""
        report = CapabilityReport(
            name=name,
            rung=rung,
            degraded_reason=reason,
            details=details,
            observed_us=now_us(),
        )
        healthy = report.rung in (
            CapabilityRung.HEALTHY,
            CapabilityRung.MEASURED,
            CapabilityRung.RECOMMENDED,
        )
        return {
            "name": report.name,
            "rung": report.rung.value,
            "state": report.rung.value,
            "available": healthy,
            "degraded_reason": report.degraded_reason,
            "details": report.details,
            "observed_us": report.observed_us,
        }

    def _cache_stats(self) -> dict:
        """The store's final-pack cache counters (V4-33.07) — an empty
        honest report when the cache module or store attribute is absent.
        Never fabricates hit-rate numbers."""
        try:
            from ..retrieval import cache as _rcache

            return _rcache.stats_for(self._store, self._config)
        except Exception:
            return {
                "enabled": False,
                "state": "unavailable",
                "degraded_reason": "verbatim.retrieval.cache not importable",
            }

    def _schema_probe(self) -> dict:
        """One read-connection observation of the v3 tables the lanes
        serve from — presence of the artifacts, not just importable code
        (V4-50.02)."""
        from ..storage.repos import has_table

        tables = (
            "claims",
            "claim_revisions",
            "spans",
            "sources",
            "source_revisions",
            "source_envelopes",
            "security_labels",
            "quarantine",
            "grants_v3",
            "trajectories",
            "embeddings",
            "index_generations",
            "claim_edges",
            "edges",
            "derivations",
            "claim_entities",
            "episode_members",
            "transitions",
            "conflict_members",
        )
        out: dict[str, Any] = {
            "tables": {},
            "embedding_rows": 0,
            "index_generations": 0,
        }
        try:
            with self._store.read() as conn:
                for t in tables:
                    out["tables"][t] = has_table(conn, t)
                if out["tables"]["embeddings"]:
                    out["embedding_rows"] = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM embeddings"
                        ).fetchone()[0]
                    )
                if out["tables"]["index_generations"]:
                    out["index_generations"] = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM index_generations"
                        ).fetchone()[0]
                    )
        except Exception:
            out["tables"] = {t: False for t in tables}
        return out

    def _encoder_observation(self) -> dict:
        """The bound encoder provider's own report (F4-15).

        Only a provider attached to this store can claim usability —
        ``store.encoder.available()`` is the component's probe; a bare
        ``encode_query`` callable without a probe caps at ``configured``;
        a config backend with no provider bound at all is ``configured``
        at best (V4-50.02: configured flags and importable numpy never
        imply a usable encoder).
        """
        backend = self._config.embedding.backend
        details: dict[str, Any] = {
            "backend": backend,
            "provider": None,
        }
        enc = getattr(self._store, "encoder", None)
        if enc is not None:
            eid = getattr(enc, "encoder_id", None)
            if callable(eid):
                try:
                    eid = eid()
                except Exception:
                    eid = None
            details["provider"] = eid or type(enc).__name__
            probe = getattr(enc, "available", None)
            if not callable(probe):
                return {
                    "rung": CapabilityRung.CONFIGURED,
                    "reason": "bound encoder exposes no availability probe",
                    "details": details,
                }
            try:
                ok = bool(probe())
            except Exception:
                ok = False
            if ok:
                return {
                    "rung": CapabilityRung.HEALTHY,
                    "reason": None,
                    "details": details,
                }
            return {
                "rung": CapabilityRung.UNAVAILABLE,
                "reason": (
                    f"bound encoder {details['provider']!r} reports "
                    "unavailable"
                ),
                "details": details,
            }
        if callable(getattr(self._store, "encode_query", None)):
            details["provider"] = "store.encode_query"
            return {
                "rung": CapabilityRung.CONFIGURED,
                "reason": "encode provider bound without availability probe",
                "details": details,
            }
        if backend == "none":
            return {
                "rung": CapabilityRung.IMPLEMENTED,
                "reason": "embedding backend disabled (backend='none')",
                "details": details,
            }
        return {
            "rung": CapabilityRung.CONFIGURED,
            "reason": (
                f"embedding backend {backend!r} configured but no encoder "
                "provider is bound to this store — configuration is not "
                "availability (V4-50.02)"
            ),
            "details": details,
        }

    def _vault_key_state(self) -> dict:
        v = self._config.v3.vault
        crypto = self._module_present("cryptography.hazmat.primitives.ciphers.aead")
        provisioned: Optional[bool] = None
        if v.enabled and crypto:
            if v.key_source == "env":
                provisioned = any(
                    k.startswith("VERBATIM_VAULT_KEY_") for k in os.environ
                )
            elif v.key_source == "file":
                provisioned = bool(v.key_file) and os.path.isfile(v.key_file)
            else:
                provisioned = None  # external provider binds at runtime
        return {
            "enabled": bool(v.enabled),
            "key_source": v.key_source,
            "crypto_backend": crypto,
            "key_provisioned": provisioned,
            "plaintext_hydration": bool(v.allow_plaintext_hydration),
        }

    def capabilities(self) -> dict:
        """The capability matrix the host may rely on (§62.04, V4-50).

        Every lane reports a ``CapabilityReport``-shaped observation —
        ``rung`` on the V4-50.01 ladder, ``degraded_reason``, and
        provider details — produced from runtime probes (bound encoder
        ``available()``, schema/table presence, key provisioning), never
        inferred from config flags or importable modules alone.
        ``degradation_notes`` carries the degradation the caller should
        actually plan around.
        """
        v3 = self._config.v3
        rcfg = v3.retrieval
        numpy = self._module_present("numpy")
        retrieval_v3 = self._module_present("verbatim.retrieval.v3.recall")
        vault = self._vault_key_state()
        schema = self._schema_probe()
        tables = schema["tables"]

        notes: list[str] = []
        lanes: dict[str, dict] = {}

        # In-process lanes: health = the store actually has the tables
        # they serve from (schema probe), not merely that code imports.
        fts = bool(getattr(self._store, "fts_enabled", False))
        lanes["lexical"] = self._cap(
            "lexical",
            CapabilityRung.HEALTHY
            if fts and tables.get("claims")
            else (
                CapabilityRung.IMPLEMENTED
                if not fts
                else CapabilityRung.UNAVAILABLE
            ),
            None
            if fts and tables.get("claims")
            else (
                "sqlite build lacks FTS5; bounded substring fallback only"
                if not fts
                else "claims tables absent in this store"
            ),
            backend="sqlite-fts5",
            fts5=fts,
        )
        for name, table in (
            ("temporal", "claims"),
            ("structured", "claims"),
            ("exact_id", "claims"),
            ("browse", "sources"),
        ):
            lanes[name] = self._cap(
                name,
                CapabilityRung.HEALTHY
                if tables.get(table)
                else CapabilityRung.UNAVAILABLE,
                None
                if tables.get(table)
                else f"{table} table absent in this store",
                serves=table,
            )
        lanes["archive_evidence"] = self._cap(
            "archive_evidence",
            CapabilityRung.HEALTHY
            if tables.get("source_revisions")
            else CapabilityRung.UNAVAILABLE,
            None
            if tables.get("source_revisions")
            else "source_revisions table absent in this store",
            lane="explicit archive/evidence browsing (V4-08.07)",
            requires_verbs=["read", "quote"],
            declared_modes=sorted(_EVIDENCE_LANE_MODES),
        )

        # Dense/semantic lane — provider-owned observation only.
        enc = self._encoder_observation()
        dense_configured = bool(rcfg.dense) or enc["details"]["backend"] != "none"
        details = dict(enc["details"])
        details["numpy_importable"] = numpy
        details["embedding_rows"] = schema["embedding_rows"]
        details["dense_flag"] = bool(rcfg.dense)
        reason = enc["reason"]
        rung = enc["rung"]
        if rung == CapabilityRung.HEALTHY and not dense_configured:
            # A bound working encoder still reports healthy — the lane
            # serves — but note the flag mismatch rather than hiding it.
            reason = "encoder bound and healthy; v3.retrieval.dense flag off"
        lanes["dense"] = self._cap(
            "dense", rung, reason, **details
        )

        # Artifact-gated lanes: configured flags never imply usable
        # index artifacts or scorers (V4-50.02).
        index_rows = bool(
            tables.get("index_generations")
            and schema["index_generations"] > 0
        )
        for name, flag, artifact in (
            ("sparse", rcfg.sparse, "sparse index artifacts + scorer"),
            (
                "late_interaction",
                rcfg.late_interaction,
                "late-interaction index artifacts",
            ),
        ):
            if not flag:
                lanes[name] = self._cap(
                    name,
                    CapabilityRung.IMPLEMENTED,
                    f"{name} lane not configured",
                    configured=False,
                )
            elif index_rows:
                lanes[name] = self._cap(
                    name,
                    CapabilityRung.INSTALLED,
                    f"{name} enabled and index generations exist; scorer "
                    "provisioning is not observed by this surface",
                    configured=True,
                )
            else:
                lanes[name] = self._cap(
                    name,
                    CapabilityRung.UNAVAILABLE,
                    f"{name} configured but {artifact} are not provisioned",
                    configured=True,
                )
        # Graph lane (V4-30.*): bounded typed-edge expansion over the
        # real edge stores — edges, derivations, entity co-membership,
        # episode membership, transitions, open conflict groups.
        # Composition is STRUCTURAL (provenance-preserving typed-edge
        # weights), never HRR vector binding — this build has no pinned
        # atom encoding/numeric backend, so holographic composition is
        # reported honestly as structural-only (V4-04.02 reference =
        # exact entity intersection).
        graph_tables = (
            "edges", "derivations", "claim_entities",
        )
        graph_ready = all(tables.get(t) for t in graph_tables)
        graph_details = {
            "composition": "structural",
            "binding": "typed_edge",
            "hrr": "unavailable",
            "hrr_reason": "no pinned atom encoding / numeric backend in "
                          "this build; exact entity intersection is the "
                          "reference composition (V4-04.02)",
            "bounds": {"hops": 1, "max_nodes": 100, "max_edges": 200,
                       "fanout": 32},
            "serves": ["edges", "derivations", "claim_entities",
                       "episode_members", "transitions",
                       "conflict_members"],
        }
        for name, flag, ready in (
            ("graph", rcfg.graph, graph_ready),
            ("causal", rcfg.causal, bool(tables.get("transitions"))),
        ):
            extra = graph_details if name == "graph" else {}
            if not flag:
                lanes[name] = self._cap(
                    name,
                    CapabilityRung.IMPLEMENTED,
                    f"{name} lane not configured",
                    configured=False,
                    **extra,
                )
            elif ready:
                lanes[name] = self._cap(
                    name, CapabilityRung.HEALTHY, None, configured=True,
                    **extra,
                )
            else:
                lanes[name] = self._cap(
                    name,
                    CapabilityRung.UNAVAILABLE,
                    f"{name} configured but its edge tables are absent",
                    configured=True,
                    **extra,
                )
        # Profiles lane (V4-21.*): provider-owned runtime probe — the
        # module reports its own rung from table presence, and the
        # deferred durable compile job is stated, never implied.
        try:
            from ..profiles import probe_capability as _profile_probe

            probe = _profile_probe(self._store)
            lanes["profiles"] = self._cap(
                "profiles",
                CapabilityRung(probe["state"]),
                probe["degraded_reason"],
                **probe["details"],
            )
        except Exception:
            lanes["profiles"] = self._cap(
                "profiles",
                CapabilityRung.UNAVAILABLE,
                "verbatim.profiles probe failed or module not importable",
            )

        # Honest tokenizer bound (V4-17.07): unicode61 strips diacritics
        # but cannot segment CJK-class scripts; those queries take the
        # bounded folded-substring path — substring recall, not
        # semantic coverage, and no trigram index exists in this build.
        lanes["lexical"]["details"]["tokenizer"] = "unicode61"
        lanes["lexical"]["details"]["unsegmented_scripts"] = (
            "bounded folded substring scan (no trigram index)"
        )

        if not dense_configured:
            notes.append("semantic lane not configured (v3.retrieval.dense=false)")
        elif lanes["dense"]["rung"] != CapabilityRung.HEALTHY.value:
            notes.append(
                "dense lane degraded: " + str(lanes["dense"]["degraded_reason"])
            )
        if not retrieval_v3:
            notes.append(
                "v3 retrieval pipeline unavailable; recall abstains unless "
                "the caller explicitly declares the archive/evidence lane"
            )
        # Vault rung: enabled+backend+key provisioned → authorized (the
        # highest claimable rung without a live hydration round-trip);
        # enabled but missing pieces → unavailable; off → implemented.
        if not v3.vault.enabled:
            vault_rung = CapabilityRung.IMPLEMENTED
            vault_reason = "vault disabled; sensitivity classes store as metadata only"
            notes.append(vault_reason)
        elif vault["crypto_backend"] is False:
            vault_rung = CapabilityRung.UNAVAILABLE
            vault_reason = "vault enabled but cryptography backend is missing"
            notes.append(vault_reason)
        elif vault["key_provisioned"] is False:
            vault_rung = CapabilityRung.UNAVAILABLE
            vault_reason = "vault enabled but no wrap key is provisioned"
            notes.append(vault_reason)
        elif vault["key_provisioned"] is None:
            vault_rung = CapabilityRung.CONFIGURED
            vault_reason = (
                "vault key_source 'external' binds at runtime — not observed"
            )
        else:
            vault_rung = CapabilityRung.AUTHORIZED
            vault_reason = None
        vault_report = self._cap("vault", vault_rung, vault_reason, **vault)

        # Progressive disclosure (V4-32.05/32.09): always available on
        # the v3 pipeline — the tiers need no artifacts, only the
        # pipeline itself.
        lanes["progressive_disclosure"] = self._cap(
            "progressive_disclosure",
            CapabilityRung.HEALTHY
            if retrieval_v3
            else CapabilityRung.UNAVAILABLE,
            None
            if retrieval_v3
            else "verbatim.retrieval.v3.recall not importable",
            tiers=["l0", "l1", "l2"],
            expansion="caller/scope/purpose/revision/expiry-bound "
                      "expand refs on every delivered item; expansion "
                      "re-runs current authorization",
        )

        # Final-pack cache (V4-33): report its REAL counters — hits sit
        # beside misses/invalidations, never a fabricated hit rate. Off
        # by default (configured=False is the honest state, not an
        # outage) — the invalidation model is conservative but operators
        # opt in explicitly. Reported as a status surface, NOT a query
        # lane — it changes no routing decision.
        cache_stats = self._cache_stats()
        cache_enabled = bool(cache_stats.get("enabled"))
        recall_cache = self._cap(
            "recall_cache",
            (
                CapabilityRung.HEALTHY
                if cache_enabled
                else CapabilityRung.IMPLEMENTED
            ) if retrieval_v3 else CapabilityRung.UNAVAILABLE,
            (
                None
                if cache_enabled
                else "retrieval.cache.enabled is off (default-off "
                     "safety gate)"
            ) if retrieval_v3 else (
                "verbatim.retrieval.v3.recall not importable"
            ),
            configured=cache_enabled,
            **{
                k: v for k, v in cache_stats.items()
                if k not in ("enabled", "state", "degraded_reason")
            },
        )
        if not cache_enabled and retrieval_v3:
            notes.append(
                "recall cache off (retrieval.cache.enabled=false) — "
                "every recall recomputes"
            )

        return {
            "wire_version": 3,
            "schema_version": getattr(self._store, "schema_version", None),
            "profile": v3.profile,
            "capture_depth": v3.capture_depth,
            "lanes": lanes,
            "recall_cache": recall_cache,
            "retrieval_v3": self._cap(
                "retrieval_v3",
                CapabilityRung.HEALTHY
                if retrieval_v3 and tables.get("claims")
                else (
                    CapabilityRung.INSTALLED
                    if retrieval_v3
                    else CapabilityRung.UNAVAILABLE
                ),
                None
                if retrieval_v3 and tables.get("claims")
                else (
                    "retrieval schema tables absent"
                    if retrieval_v3
                    else "verbatim.retrieval.v3.recall not importable"
                ),
                controller=rcfg.controller,
            ),
            "vault": vault_report,
            "capture": self._cap(
                "capture",
                CapabilityRung.HEALTHY
                if tables.get("source_envelopes")
                else CapabilityRung.UNAVAILABLE,
                None
                if tables.get("source_envelopes")
                else "evidence tables absent in this store",
                agent_submitted_kinds=sorted(
                    k.value for k in AGENT_SUBMITTED_KINDS
                ),
                requires_capture_authorization=True,
            ),
            "degradation_notes": notes,
        }

    # ------------------------------------------------------------------
    # §12.02/§12.03 trajectories + checker-attested outcomes
    # ------------------------------------------------------------------

    def submit_trajectory(
        self,
        scope_id: str,
        *,
        principal_id: str,
        trajectory_id: Optional[str] = None,
        task_id: str = "",
        session_id: str = "",
        steps: Iterable[Any] = (),
        environment_digest: Optional[str] = None,
        boundary_rule: str = "task_id",
        complete_trajectory: bool = True,
        purpose: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Record a task-bounded trajectory under the ``derive`` verb.

        ``steps`` are dicts carrying optional ``action_envelope_id``,
        ``observation_envelope_ids``, ``state_delta_refs`` and an
        ``environment`` mapping — envelope references must already exist
        in the scope (evidence cannot be cited before it is captured).
        """
        require_id(scope_id, "scope_id")
        require_id(principal_id, "principal_id")
        tid = trajectory_id or f"traj:{new_id()}"
        step_list = list(steps or ())
        if len(step_list) > 512:
            raise _err(ErrorCode.VALIDATION, "trajectory steps bounded at 512")
        with self._store.tx() as conn:
            governance.seed_purposes(conn)
            caller = self._caller(principal_id, session_id)
            governance.authorize(
                conn, caller, scope_id, Verb.DERIVE.value, purpose=purpose
            )
            from ..core.types_v3 import EnvironmentFingerprint

            record = TrajectoryRecord(
                trajectory_id=tid,
                scope_id=scope_id,
                host_id=self._host_id,
                session_id=session_id,
                task_id=task_id,
                boundary_rule=boundary_rule,
                environment_digest=environment_digest,
                metadata=dict(metadata or {}),
            )
            from ..evidence import add_step, complete, record_trajectory

            record_trajectory(conn, record, store=self._store)
            written: list[str] = []
            for i, spec in enumerate(step_list):
                if not isinstance(spec, dict):
                    raise _err(
                        ErrorCode.VALIDATION, "trajectory steps must be mappings"
                    )
                env = spec.get("environment")
                fingerprint = None
                if isinstance(env, dict):
                    fingerprint = EnvironmentFingerprint(
                        repo_id=env.get("repo_id"),
                        repo_revision=env.get("repo_revision"),
                        runtime_versions=tuple(
                            tuple(p) for p in env.get("runtime_versions") or ()
                        ),
                        tool_schema_versions=tuple(
                            tuple(p)
                            for p in env.get("tool_schema_versions") or ()
                        ),
                        platform=env.get("platform"),
                    )
                step = TrajectoryStep(
                    step_id=str(spec.get("step_id") or f"{tid}:step:{i}"),
                    trajectory_id=tid,
                    ord=i,
                    action_envelope_id=spec.get("action_envelope_id"),
                    observation_envelope_ids=tuple(
                        spec.get("observation_envelope_ids") or ()
                    ),
                    state_delta_refs=tuple(spec.get("state_delta_refs") or ()),
                    environment=fingerprint,
                )
                written.append(add_step(conn, step))
            completed = None
            if complete_trajectory:
                completed = complete(
                    conn, tid, environment_digest, store=self._store
                )
            return {
                "trajectory_id": tid,
                "scope_id": scope_id,
                "steps": len(written),
                "completed_event": (
                    completed.get("completed_event") if completed else None
                ),
            }

    # ------------------------------------------------------------------
    # host checker resolution (V4-23.01/23.02 — C19/C20)
    # ------------------------------------------------------------------

    def _resolve_checker(self, invocation_id: Optional[str]) -> Optional[dict]:
        """Resolve an execution receipt through the registered host
        checker resolver — the ONLY path to ``host_attested`` (V4-23.02).

        ``invocation_id`` is the host-side execution identifier; the
        resolver answers with the receipt it observed, never with fields
        the caller supplied. Returns ``None`` when no resolver is bound
        or the invocation is unknown — callers then record an agent
        report. A resolver returning a malformed receipt is a host
        contract violation (``VALIDATION``), not a silent downgrade.
        """
        if not invocation_id or not isinstance(invocation_id, str):
            return None
        resolver = self._checker_resolver
        if resolver is None:
            return None
        fn = getattr(resolver, "resolve_checker", None)
        if not callable(fn):
            fn = resolver if callable(resolver) else None
        if fn is None:
            return None
        raw = fn(invocation_id)
        if raw is None:
            return None
        if isinstance(raw, dict):
            resolved = dict(raw)
        else:
            resolved = {
                f: getattr(raw, f, None) for f in _CHECKER_RECEIPT_FIELDS
            }
        checker_id = resolved.get("checker_id")
        if not isinstance(checker_id, str) or not checker_id:
            raise _err(
                ErrorCode.VALIDATION,
                "checker resolver returned a receipt without checker_id",
            )
        resolved["checker_id"] = checker_id
        resolved["invocation_id"] = resolved.get("invocation_id") or (
            invocation_id
        )
        return resolved

    @staticmethod
    def _check_receipt_binding(
        resolved: dict,
        *,
        invocation_id: str,
        scope_id: str,
        task_id: str,
        artifact_digest: Optional[str],
    ) -> None:
        """Reject a resolved receipt bound to a different invocation,
        scope, task, or artifact (V4-23.01, C20): a receipt certifies
        only the execution it names."""
        expected = {
            "invocation_id": invocation_id,
            "scope_id": scope_id,
            "task_id": task_id,
            "artifact_digest": artifact_digest,
        }
        conflicts = [
            field
            for field in _RECEIPT_BINDINGS
            if resolved.get(field) not in (None, "")
            and str(resolved[field]) != str(expected[field] or "")
        ]
        if conflicts:
            raise _err(
                ErrorCode.VALIDATION,
                "checker receipt binds a different "
                + "/".join(conflicts)
                + " — a receipt cannot certify another task, scope, or "
                "artifact (V4-23.01)",
            )

    @staticmethod
    def _receipt_outcome(resolved: dict) -> OutcomeClass:
        """The outcome the host-observed receipt itself reports — the
        caller's declared outcome never overrides the receipt."""
        declared = resolved.get("outcome")
        if isinstance(declared, str):
            try:
                return OutcomeClass(declared)
            except ValueError:
                pass
        if resolved.get("completed") is False:
            return OutcomeClass.UNKNOWN
        code = resolved.get("exit_code")
        if isinstance(code, int) and not isinstance(code, bool):
            return OutcomeClass.SUCCESS if code == 0 else OutcomeClass.FAILURE
        return OutcomeClass.UNKNOWN

    def _checker_attestation(self, resolved: dict, outcome: OutcomeClass) -> str:
        """Store-keyed digest over the resolved binding — the write-time
        attestation anchor distinguishing resolver-attested receipts from
        caller-shaped metadata (V4-23.01/02)."""
        canonical = "|".join(
            str(resolved.get(f) or "")
            for f in _CHECKER_RECEIPT_FIELDS
        )
        canonical += f"|{_ATTESTATION_DOMAIN}|{outcome.value}"
        return self._store.hmac(canonical.encode("utf-8")).hex()

    def submit_outcome(
        self,
        scope_id: str,
        *,
        principal_id: str,
        outcome: Any,
        checker_id: str = "",
        invocation_id: Optional[str] = None,
        artifact_digest: Optional[str] = None,
        task_id: str = "",
        session_id: str = "",
        trajectory_id: Optional[str] = None,
        purpose: Optional[str] = None,
        evidence_refs: Iterable[str] = (),
        recorded_us: Optional[int] = None,
    ) -> dict:
        """Ingest an outcome envelope (§12.03, V4-23.01/23.02).

        Two honest shapes, decided by the registered host checker
        resolver — never by caller fields:

        * ``invocation_id`` resolves through the bound resolver → a
          ``host_observed`` verification envelope carrying the receipt
          (binding-checked against this scope/task/artifact — C20), a
          store-keyed ``attestation`` digest, and the outcome the
          receipt itself reports.
        * Otherwise → an ``agent_generated`` report: the caller-named
          ``checker_id`` and declared ``outcome`` persist as a
          self-reported claim (``host_attested: False``,
          ``agent_report: True``) that can never upgrade an episode
          outcome (C19).

        The ``derive`` verb authorizes the write; ``trajectory_id``
        optionally links the envelope onto the trajectory's metadata.
        """
        require_id(scope_id, "scope_id")
        require_id(principal_id, "principal_id")
        if not checker_id and not invocation_id:
            raise _err(
                ErrorCode.VALIDATION,
                "checker_id or invocation_id is required",
            )
        try:
            declared_cls = OutcomeClass(outcome)
        except ValueError as exc:
            raise _err(
                ErrorCode.VALIDATION, f"unknown outcome {outcome!r}"
            ) from exc
        resolved = self._resolve_checker(invocation_id)
        if resolved is not None:
            self._check_receipt_binding(
                resolved,
                invocation_id=invocation_id or "",
                scope_id=scope_id,
                task_id=task_id,
                artifact_digest=artifact_digest,
            )
            outcome_cls = self._receipt_outcome(resolved)
            attested_by = (
                getattr(self._checker_resolver, "resolver_id", None)
                or self._host_id
                or "host-checker-resolver"
            )
            checker_payload = {
                "checker_id": resolved["checker_id"],
                "checker_version": resolved.get("checker_version") or "",
                "invocation_id": resolved["invocation_id"],
                "completed": bool(resolved.get("completed", True)),
                "exit_code": resolved.get("exit_code"),
                "result_json": dict(resolved.get("result_json") or {}),
                "selected_tests": list(resolved.get("selected_tests") or ()),
                "host_attested": True,
                "attested_by": attested_by,
                "attestation": self._checker_attestation(
                    resolved, outcome_cls
                ),
                "scope_id": scope_id,
                "task_id": task_id,
                "artifact_digest": artifact_digest
                or resolved.get("artifact_digest"),
                "environment_digest": resolved.get("environment_digest"),
                "nonce": resolved.get("nonce"),
                "issued_us": resolved.get("issued_us"),
                "resolved_us": now_us(),
            }
            trust = TrustClass.HOST_OBSERVED
            asserter = resolved["checker_id"]
            verification = "host_attested"
        else:
            outcome_cls = declared_cls
            checker_payload = {
                "checker_id": checker_id,
                "host_attested": False,
                "agent_report": True,
                "reported_invocation_id": invocation_id or "",
            }
            trust = TrustClass.AGENT_GENERATED
            asserter = principal_id
            verification = "agent_report"
        body = {
            "outcome": outcome_cls.value,
            "declared_outcome": declared_cls.value,
            "checker": checker_payload,
            "task_id": task_id,
            "evidence_refs": list(evidence_refs or ()),
            "reported_by": principal_id,
        }
        import json as _json

        with self._store.tx() as conn:
            governance.seed_purposes(conn)
            caller = self._caller(principal_id, session_id)
            governance.authorize(
                conn, caller, scope_id, Verb.DERIVE.value, purpose=purpose
            )
            if trajectory_id is not None:
                require_id(trajectory_id, "trajectory_id")
                from ..evidence import get_trajectory

                if get_trajectory(conn, trajectory_id) is None:
                    _deny("trajectory not found or unauthorized")
            envelope = SourceEnvelopeV3(
                kind=EnvelopeKind.VERIFICATION,
                scope_id=scope_id,
                actor_principal=principal_id,
                perspective=Perspective(
                    asserter=asserter, observer=principal_id
                ),
                event_us=int(recorded_us) if recorded_us is not None else now_us(),
                receipt_us=0,
                content=_json.dumps(body, sort_keys=True).encode("utf-8"),
                media_type="application/json",
                trust_class=trust,
                adapter_version=self._adapter_version,
                host_id=self._host_id,
                session_id=session_id,
                task_id=task_id,
                metadata={
                    "outcome": outcome_cls.value,
                    "declared_outcome": declared_cls.value,
                    "verification": verification,
                    "checker": checker_payload,
                    "task_id": task_id,
                    "trajectory_id": trajectory_id or "",
                    "evidence_refs": list(evidence_refs or ()),
                    "submitted_via": "api_v3.submit_outcome",
                },
            )
            receipt = ingest_envelope(conn, self._store, envelope)
            # V4-14.01: obligations commit atomically with the outcome
            # envelope (verification kinds are structural-harvest
            # eligible — their DAG carries real pipeline stages).
            self._record_readiness(conn, receipt)
            if trajectory_id is not None:
                row = repos_v3.get(
                    conn, "trajectories", {"trajectory_id": trajectory_id}
                )
                meta = repos_v3.json_field(row, "metadata_json", {}) or {}
                meta["outcome"] = outcome_cls.value
                meta["outcome_envelope_id"] = receipt.envelope_id
                meta["outcome_verification"] = verification
                repos_v3.update(
                    conn,
                    "trajectories",
                    {"metadata_json": meta},
                    {"trajectory_id": trajectory_id},
                )
            return {
                "source_id": receipt.source_id,
                "envelope_id": receipt.envelope_id,
                "receipt_id": receipt.receipt_id,
                "outcome": outcome_cls.value,
                "declared_outcome": declared_cls.value,
                "attested": resolved is not None,
                "agent_report": resolved is None,
                "checker_id": checker_payload["checker_id"],
                "trajectory_id": trajectory_id,
            }

    # ------------------------------------------------------------------
    # §36 deletion: logical suppression now, physical closure via jobs
    # ------------------------------------------------------------------

    def delete_source(self, source_id: str, *, principal_id: str) -> dict:
        """Delete one source under the ``admin`` verb (§09 verb table:
        retention/deletion policy is admin).

        In one transaction: the reverse-dependency closure is tombstoned
        (immediate logical deletion), the v2 ``purge`` job is queued for
        byte erasure, and the v3 ``purge_derived`` / ``purge_vault`` jobs
        are queued on the privacy-control lane. The returned dict states
        the closure honestly: suppression is in effect now; physical and
        vault erasure complete asynchronously — this is NOT cryptographic
        erasure (§36.05 needs backup-aware key management).
        """
        require_id(source_id, "source_id")
        require_id(principal_id, "principal_id")
        with self._store.tx() as conn:
            row = conn.execute(
                "SELECT scope_id FROM sources WHERE source_id = ?",
                (source_id,),
            ).fetchone()
            if row is None:
                _deny("source not found or unauthorized")
            scope_id = row[0]
            governance.authorize(
                conn,
                self._caller(principal_id),
                scope_id,
                Verb.ADMIN.value,
            )
            result = _purge.suppress(
                self._store,
                scope_id,
                [("source", source_id)],
                principal_id,
                conn=conn,
            )
            purge_id = result["purge_id"]
            queue = JobQueue(self._store)
            span_view_ids = [
                r[0]
                for r in conn.execute(
                    "SELECT span_id FROM spans WHERE source_id = ?",
                    (source_id,),
                ).fetchall()
            ]
            entry_ids = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT entry_id FROM vault_refs"
                    " WHERE scope_id = ?"
                    f" AND view_id IN ({','.join('?' for _ in ([source_id] + span_view_ids))})",
                    [scope_id, source_id, *span_view_ids],
                ).fetchall()
            ]
            jobs: dict[str, Any] = {}
            jobs["purge"] = queue.enqueue(
                conn,
                scope_id,
                JobKind.PURGE,
                {"purge_id": purge_id},
            )
            jobs["purge_derived"] = queue.enqueue(
                conn,
                scope_id,
                JobKind.PURGE_DERIVED,
                {
                    "parents": [
                        {"kind": "source", "id": source_id, "revision": None}
                    ],
                    "purge_id": purge_id,
                },
            )
            jobs["purge_vault"] = (
                queue.enqueue(
                    conn,
                    scope_id,
                    JobKind.PURGE_VAULT,
                    {"entry_ids": entry_ids, "purge_id": purge_id},
                )
                if entry_ids
                else None
            )
            return {
                "source_id": source_id,
                "scope_id": scope_id,
                "status": "suppressed",
                "purge_id": purge_id,
                "suppress_seq": result.get("suppress_seq"),
                "closure": {
                    "logical": "suppressed_now",
                    "targets": [list(t) for t in result.get("targets", ())],
                    "physical": "queued_purge_job",
                    "derived": "queued_purge_derived_job",
                    "vault": (
                        "queued_purge_vault_job"
                        if entry_ids
                        else "no_vault_entries"
                    ),
                },
                "jobs": jobs,
                "vault_entries": len(entry_ids),
                "notes": (
                    "Logical deletion (suppression) is in effect now; byte "
                    "and vault erasure complete when the privacy-control "
                    "lane drains the queued jobs. Cryptographic erasure "
                    "additionally depends on key management outside this "
                    "call and is not claimed here."
                ),
            }
