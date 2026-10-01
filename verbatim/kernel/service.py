"""Evidence-access and delivery kernel (SPEC_V4 §08, V4-08.01–V4-08.10).

The kernel is the single gate between stored evidence and any consumer:
``resolve_access`` evaluates authorization + ancestry + lifecycle and
returns an ``EligibilityLease``; ``read_verified`` re-authenticates bytes
against persisted digests before releasing them; ``seal_delivery`` is the
disclosure linearization point; ``open_dispatch`` hands a sealed delivery
to the transport broker; ``invalidate`` records epoch bumps and
affected-object obligations for quarantine/correction/revocation/erasure.

Denial discipline (V4-10.06 — absent == unauthorized):

* ``resolve_access`` NEVER raises for denials. It returns
  ``EligibilityLease(denied=True)`` carrying only the caller's own inputs
  (caller, verb, purpose, requested scopes) — no store-derived detail such
  as which scope failed, which object is held, or whether the object
  exists at all. Distinguishable caller-visible classes still raise:
  ``VALIDATION`` (malformed input, unregistered purpose — caller bugs) and
  ``STALE_EPOCH`` (the caller's pinned epoch was superseded — retryable
  rebinding signal, not an existence/authorization leak).
* Every other surface raises ``VerbatimError(NOT_FOUND_OR_UNAUTHORIZED)``
  on denial — one public code, one message, no counts or reasons.
* ``read_verified`` is atomic at group level (V4-08.05): a denied or
  unavailable locator invalidates the whole read; no partial subsets.

One read path (V4-08.09): this module is the only verified-read surface;
there is no fallback SQL and no unverified serialization path. Failures
propagate as typed errors — callers must not retry around them.

Object identity: a locator/ref ``object_id`` resolves through the v4
``objects`` registry first (which also gates on ``disposition``), then the
legacy evidence tables (``sources``, ``spans``, ``claims``,
``source_envelopes``). An id matching more than one kind is ambiguous and
denies. Byte reads are supported for source revisions (canonical payload,
``view_id=None``), named ``source_views`` (``view_id``), and spans (the
locator must exactly equal the span's stored range — the persisted
``excerpt_hmac`` authenticates the slice binding). Registered objects of
kinds with no byte surface resolve for access evaluation but deny reads.

Lifecycle gates: ``objects.disposition`` must be ``active``; a claim's
revision state must be ``active`` or ``disputed`` (still visible, flagged);
quarantine ``pending``/``suppressed`` holds and purge suppression
(``suppressed``/``purging``/``completed``) withhold — including the
ancestor cascade (span → its source revision + covering envelopes, claim →
its evidence spans and their sources, registered object → dependency-edge
parents). An emptied (purged) payload reads as absent. Retention gates
express through the same suppression/erasure rows, never a second path.

Verb gate (V4-08.02): exact original bytes (canonical payload or a
``primary`` view) require ``quote``. ``read`` releases authorized metadata
plus non-primary approved derived views; a ``read`` lease on canonical
bytes yields a ``metadata_only`` slice (metadata review — V4-08.06 — is
the lease itself). Other verbs never release bytes through this path.
"""

from __future__ import annotations

import hashlib
import hmac
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from ..core import time as _time
from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    new_id,
    require_id,
    safe_json_loads,
)
from ..core.types_v3 import Verb
from ..core.types_v4 import (
    DeliveryPermit,
    DispatchPermit,
    EligibilityLease,
    EvidenceLocator,
    LifecycleState,
    PurposeConstraint,
    VerifiedSlice,
)
from ..governance import CallerV3
from ..governance import epochs as _epochs
from ..governance import grants as _grants
from ..governance import purposes as _purposes
from ..security import quarantine as _quarantine
from ..storage import repos_v3, repos_v4

DENIAL_MESSAGE = _grants.DENIAL_MESSAGE

#: Purge row states that withhold an object (mirrors PurgesRepo._SUPPRESSED).
_SUPPRESSING_STATES = ("suppressed", "purging", "completed")

#: Claim revision states still visible to evidence reads (disputed stays
#: flagged-visible; pending/rejected/archived/erased/superseded withhold).
_CLAIM_VISIBLE_STATES = ("active", "disputed")

#: Object kinds with a verified byte surface through ``read_verified``.
_BYTE_KINDS = ("source", "span")

#: Bound on ancestry/dependency walks so a pathological graph cannot turn
#: access resolution into an unbounded scan (SPEC fixed-allowlist rule).
_MAX_ANCESTORS = 10_000
_MAX_CLAIM_SPANS = 512

#: Domain separation for kernel-computed slice digests (V4-13.04).
_SLICE_DOMAIN = b"verbatim.kernel.slice.v1\x00"
_ALGORITHM = "hmac-sha256"


def _pack_digest(pack: bytes) -> str:
    """Canonical egress digest — ``sha256:<64 hex>`` over the exact pack
    bytes. This is the broker's interchange form
    (``privacy.broker.payload_digest``): the dispatch permit binds the same
    digest the encoder computes over the serialized request, so a mutated
    body is detectable rather than silently re-derived."""
    return "sha256:" + hashlib.sha256(bytes(pack)).hexdigest()

#: Invalidation event kinds ``invalidate`` understands.
INVALIDATION_KINDS = frozenset(
    {"quarantine", "correction", "revocation", "erasure", "lifecycle"}
)


def _deny() -> "None":
    """The single public denial — identical for absent and forbidden."""
    raise VerbatimError(ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, DENIAL_MESSAGE)


def _verb(value: "Verb | str") -> Verb:
    try:
        return value if isinstance(value, Verb) else Verb(value)
    except ValueError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown verb {value!r}"
        ) from exc


def _row(cur: sqlite3.Cursor) -> Optional[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return dict(zip(cols, rows[0])) if rows else None


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


@dataclass(frozen=True)
class _Binding:
    """A resolved object: identity, owning scope, and lifecycle carrier."""

    kind: str                     # source | span | claim | source_envelope | registry kind
    object_id: str                # canonical id (span→span_id, source→source_id)
    scope_id: str
    registered: bool = False      # backed by a v4 ``objects`` row
    disposition: Optional[str] = None
    current_revision: Optional[int] = None
    source_id: Optional[str] = None   # span/envelope → covering source
    span_start: Optional[int] = None
    span_end: Optional[int] = None


@dataclass(frozen=True)
class DerivedInputs:
    """``derive_inputs`` output: a verified input bundle plus the inherited
    restrictions a derived object must carry (V4-10.07).

    ``allowed_purposes`` is the intersection of the producer's effective
    per-scope purpose sets; ``effective_audience`` is the requested audience
    intersected with principals holding a live grant in EVERY contributing
    scope. A derivative must not widen either — absent an independently
    authorized declassification operation.
    """

    producer_id: str
    inputs: tuple[VerifiedSlice, ...]
    scope_ids: tuple[str, ...]
    epoch_vector: dict[str, int]
    allowed_purposes: PurposeConstraint
    effective_audience: tuple[str, ...]
    parent_refs: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class InvalidationReport:
    """``invalidate`` output: epoch bumps and affected-object obligations.

    ``in_flight_permits`` enumerates sealed delivery permits touched by the
    event — committed disclosures are honestly irreversible (V4-09.07) and
    are reported, not retro-denied. ``expired_permits`` are sealed permits
    whose max age lapsed; they are transitioned to ``expired`` durably.
    """

    event_kind: str
    epochs: dict[str, Optional[int]]
    affected: tuple[tuple[str, str, Optional[int]], ...]
    obligations: tuple[dict[str, Any], ...]
    in_flight_permits: tuple[str, ...]
    expired_permits: tuple[str, ...]
    evaluated_us: int


@dataclass(frozen=True)
class ReviewedMetadata:
    """``review_metadata`` output: scope-local metadata released under a
    lease — counts, titles, snippets, identifiers, and error details get
    the same disclosure review as primary content (V4-08.06)."""

    fields: dict[str, Any]
    lease_id: str
    scope_ids: tuple[str, ...]
    reviewed_us: int


class Kernel:
    """The §08 evidence kernel over one ``Store``.

    All methods take the caller's ``conn``: read surfaces work on a
    ``store.read()`` snapshot; ``seal_delivery``, ``mark_delivered``, and
    ``invalidate`` mutate and must run inside ``store.tx()``.
    """

    def __init__(self, store: Any, *, broker: Any = None) -> None:
        self._store = store
        # Optional provisioned TransportBroker (verbatim.privacy.broker) —
        # injected by the host so the kernel never hard-depends on, or
        # invents, egress configuration (V4-12.04/09 stay fail-closed).
        self._broker = broker

    # ------------------------------------------------------------------
    # resolve_access (V4-08.01, V4-10.04/05/06)
    # ------------------------------------------------------------------

    def resolve_access(
        self,
        conn: sqlite3.Connection,
        caller: CallerV3,
        verb: "Verb | str",
        purpose: Optional[str],
        scope_ids: Iterable[str],
        object_refs: Iterable[Any] = (),
        *,
        now_us: Optional[int] = None,
        max_age_us: Optional[int] = None,
        operation_id: Optional[str] = None,
    ) -> EligibilityLease:
        """Evaluate access and mint a short-lived eligibility lease.

        Returns ``EligibilityLease(denied=True)`` — never raises — for any
        authorization, existence, lifecycle, quarantine, suppression, or
        retention failure (indistinguishable denial). Raises only
        ``VALIDATION`` (caller bugs) and ``STALE_EPOCH`` (the caller's own
        pinned epoch was superseded — a retryable fence, not a leak).
        """
        now = int(now_us) if now_us is not None else _time.now_us()
        v = _verb(verb)
        if not isinstance(caller, CallerV3):
            raise VerbatimError(
                ErrorCode.VALIDATION, "caller must be a bound CallerV3"
            )
        if purpose is not None:
            if not isinstance(purpose, str) or not purpose:
                raise VerbatimError(
                    ErrorCode.VALIDATION, "purpose must be a non-empty string"
                )
            # V4-11.08: purposes resolve through the registry; retired or
            # unknown purposes cannot authorize new operations.
            _purposes.require_purpose(conn, purpose)
        scopes = tuple(sorted({self._scope_id(s) for s in scope_ids}))
        if not scopes:
            # V4-10.04: omission of a scope filter never means global access.
            raise VerbatimError(
                ErrorCode.VALIDATION, "resolve_access requires scope_ids"
            )
        refs = self._normalize_refs(object_refs)
        max_age = self._max_age(max_age_us)
        op_id = operation_id or f"op:{new_id()}"
        require_id(op_id, "operation_id")
        require_id(caller.principal_id, "principal_id")

        def denied() -> EligibilityLease:
            return EligibilityLease(
                lease_id=f"lease:{new_id()}",
                caller_id=caller.principal_id,
                operation_id=op_id,
                verb=v,
                purpose=purpose or "",
                scope_ids=scopes,
                object_refs=(),
                epoch_vector={},
                issued_us=now,
                expires_us=now + max_age,
                denied=True,
            )

        try:
            for sid in scopes:
                # The frozen grant evaluator: (scope, verb, purpose) at the
                # caller's pinned epoch. STALE_EPOCH propagates by design.
                _grants.authorize(conn, caller, sid, v.value, purpose=purpose)
            resolved: list[tuple[str, int]] = []
            for kind, oid, rev in refs:
                binding = self._resolve(conn, oid, kind)
                if binding is None or binding.scope_id not in scopes:
                    _deny()
                self._check_object(conn, binding, rev)
                resolved.append((binding.object_id, rev))
        except VerbatimError as exc:
            if exc.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED:
                return denied()
            raise

        epoch_vector = {sid: _epochs.current_epoch(conn, sid) for sid in scopes}
        return EligibilityLease(
            lease_id=f"lease:{new_id()}",
            caller_id=caller.principal_id,
            operation_id=op_id,
            verb=v,
            purpose=purpose or "",
            scope_ids=scopes,
            object_refs=tuple(resolved),
            epoch_vector=epoch_vector,
            issued_us=now,
            expires_us=now + max_age,
        )

    # ------------------------------------------------------------------
    # read_verified (V4-08.02/03/05, V4-13.05)
    # ------------------------------------------------------------------

    def read_verified(
        self,
        conn: sqlite3.Connection,
        lease: EligibilityLease,
        locators: Iterable[EvidenceLocator],
        *,
        now_us: Optional[int] = None,
    ) -> list[VerifiedSlice]:
        """Re-authenticated byte reads under a live lease.

        One failure fails the whole read (V4-08.05) — corrupt bytes raise
        ``STORE_CORRUPT``, expired leases ``PERMIT_EXPIRED``, epoch drift
        ``STALE_EPOCH``, and any denied/absent/withheld locator
        ``NOT_FOUND_OR_UNAUTHORIZED``. No partial results escape.
        """
        now = int(now_us) if now_us is not None else _time.now_us()
        self._require_lease(conn, lease, now)
        out: list[VerifiedSlice] = []
        for loc in locators:
            if not isinstance(loc, EvidenceLocator):
                raise VerbatimError(
                    ErrorCode.VALIDATION, "locators must be EvidenceLocator"
                )
            out.append(self._read_one(conn, lease, loc))
        return out

    # ------------------------------------------------------------------
    # derive_inputs (V4-10.07, V4-11.05)
    # ------------------------------------------------------------------

    def derive_inputs(
        self,
        conn: sqlite3.Connection,
        producer_grant: Any,
        input_refs: Iterable[Any],
        output_audience: Iterable[str] = (),
        output_purpose: Optional[str] = None,
        *,
        now_us: Optional[int] = None,
    ) -> DerivedInputs:
        """Verify a producer's inputs and compute inherited restrictions.

        ``producer_grant`` (grant_id or row) must be a live grant carrying
        ``derive``. ``derive`` never implies ``read``/``quote`` (V4-11.05):
        the producer principal needs an effective ``quote`` grant on every
        contributing input scope for ``output_purpose``. Denied or
        unavailable inputs invalidate the bundle (V4-08.05).
        """
        now = int(now_us) if now_us is not None else _time.now_us()
        grant = self._producer_grant(conn, producer_grant)
        if grant is None:
            _deny()
        if output_purpose is not None:
            if not isinstance(output_purpose, str) or not output_purpose:
                raise VerbatimError(
                    ErrorCode.VALIDATION, "output_purpose must be a string"
                )
            _purposes.require_purpose(conn, output_purpose)
        verbs = frozenset(repos_v3.json_field(grant, "verbs_json") or ())
        gpurposes = frozenset(repos_v3.json_field(grant, "purposes_json") or ())
        if Verb.DERIVE.value not in verbs:
            _deny()
        if gpurposes and output_purpose not in gpurposes:
            _deny()
        producer_id = grant["principal_id"]
        producer = CallerV3(principal_id=producer_id)

        refs = self._normalize_input_refs(input_refs)
        contributing: list[str] = []
        locators: list[EvidenceLocator] = []
        canonical_refs: list[tuple[str, int]] = []
        for kind, oid, rev, start, end, view_id in refs:
            binding = self._resolve(conn, oid, kind)
            if binding is None:
                _deny()
            _grants.authorize(
                conn, producer, binding.scope_id, Verb.QUOTE.value,
                purpose=output_purpose,
            )
            self._check_object(conn, binding, rev)
            if binding.scope_id not in contributing:
                contributing.append(binding.scope_id)
            canonical_refs.append((binding.object_id, rev))
            if start is None or end is None:
                start, end = self._full_range(conn, binding, rev)
            locators.append(
                EvidenceLocator(
                    object_id=binding.object_id,
                    revision=rev,
                    start_byte=start,
                    end_byte=end,
                    view_id=view_id,
                )
            )
        scopes = tuple(sorted(set(contributing)))
        epoch_vector = {sid: _epochs.current_epoch(conn, sid) for sid in scopes}
        lease = EligibilityLease(
            lease_id=f"lease:{new_id()}",
            caller_id=producer_id,
            operation_id=f"op:{new_id()}",
            verb=Verb.QUOTE,
            purpose=output_purpose or "",
            scope_ids=scopes,
            object_refs=tuple(canonical_refs),
            epoch_vector=epoch_vector,
            issued_us=now,
            expires_us=now + EligibilityLease.DEFAULT_MAX_AGE_US,
        )
        inputs = tuple(self._read_one(conn, lease, loc) for loc in locators)
        return DerivedInputs(
            producer_id=producer_id,
            inputs=inputs,
            scope_ids=scopes,
            epoch_vector=epoch_vector,
            allowed_purposes=self._purpose_intersection(
                conn, producer_id, scopes, output_purpose, now
            ),
            effective_audience=self._audience_intersection(
                conn, output_audience, scopes, now
            ),
            parent_refs=tuple(canonical_refs),
        )

    # ------------------------------------------------------------------
    # seal_delivery (V4-09.06/07/08) — the disclosure linearization point
    # ------------------------------------------------------------------

    def seal_delivery(
        self,
        conn: sqlite3.Connection,
        lease: EligibilityLease,
        serialized_pack: bytes,
        dependency_versions: Optional[Mapping[str, int]] = None,
        *,
        now_us: Optional[int] = None,
        max_age_us: Optional[int] = None,
    ) -> DeliveryPermit:
        """Linearize a disclosure: recheck inside one tx, then persist.

        Re-evaluates the lease (denial/expiry/epoch drift), the caller's
        CURRENT authorization on every contributing scope (a revocation
        since ``resolve_access`` blocks the permit — V4-09.07), object-level
        constraints on the lease's refs, and every declared dependency
        version. Only then does the ``delivery_permits`` row commit; after
        commit the disclosure is durable and honestly irreversible.

        Must be called inside ``store.tx()``. Permit max age defaults to one
        second (V4-09.08); an unhanded permit expires and needs
        reauthorization.
        """
        now = int(now_us) if now_us is not None else _time.now_us()
        self._require_lease(conn, lease, now)
        if not isinstance(serialized_pack, (bytes, bytearray)):
            raise VerbatimError(
                ErrorCode.VALIDATION, "serialized_pack must be bytes"
            )
        max_age = self._max_age(max_age_us)
        # Re-authorize against current state — the linearization recheck.
        caller = CallerV3(principal_id=lease.caller_id)
        for sid in lease.scope_ids:
            _grants.authorize(
                conn, caller, sid, lease.verb.value,
                purpose=lease.purpose or None,
            )
        for oid, rev in lease.object_refs:
            binding = self._resolve(conn, oid)
            if binding is None or binding.scope_id not in lease.scope_ids:
                _deny()
            self._check_object(conn, binding, rev)
        deps = self._check_dependencies(conn, lease, dependency_versions)
        digest = _pack_digest(bytes(serialized_pack))
        epoch_vector = {
            sid: _epochs.current_epoch(conn, sid) for sid in lease.scope_ids
        }
        permit_id = f"dpermit:{new_id()}"
        repos_v4.insert(
            conn,
            "delivery_permits",
            {
                "permit_id": permit_id,
                "caller_id": lease.caller_id,
                "purpose": lease.purpose,
                "epoch_vector_json": epoch_vector,
                "payload_digest": digest,
                "dependency_versions_json": deps,
                "state": "sealed",
                "issued_us": now,
                "expires_us": now + max_age,
                "receipt_id": None,
            },
        )
        return DeliveryPermit(
            permit_id=permit_id,
            caller_id=lease.caller_id,
            purpose=lease.purpose,
            epoch_vector=epoch_vector,
            payload_digest=digest,
            dependency_versions=deps,
            state="sealed",
            issued_us=now,
            expires_us=now + max_age,
        )

    def verify_delivery(
        self,
        conn: sqlite3.Connection,
        permit: "DeliveryPermit | str",
        *,
        now_us: Optional[int] = None,
    ) -> DeliveryPermit:
        """Load + gate a persisted delivery permit for handoff.

        Unknown, non-sealed (delivered/revoked/expired-state), or
        time-expired permits are unusable: absent vs dead is deliberately
        indistinguishable, while an expired-in-time permit raises
        ``PERMIT_EXPIRED`` so the caller reauthorizes (V4-09.08).
        """
        now = int(now_us) if now_us is not None else _time.now_us()
        permit_id = permit.permit_id if isinstance(permit, DeliveryPermit) else permit
        require_id(permit_id, "permit_id")
        row = repos_v4.get(
            conn, "delivery_permits", {"permit_id": permit_id}
        )
        if row is None or row["state"] != "sealed":
            _deny()
        if int(row["expires_us"]) <= now:
            raise VerbatimError(
                ErrorCode.PERMIT_EXPIRED,
                "delivery permit expired — reauthorize",
                retryable=True,
            )
        return self._permit_from_row(row)

    def mark_delivered(
        self,
        conn: sqlite3.Connection,
        permit_id: str,
        *,
        now_us: Optional[int] = None,
        receipt_id: Optional[str] = None,
    ) -> DeliveryPermit:
        """Transition a sealed permit to ``delivered`` inside the caller's
        tx — the honest record that the disclosure handoff happened."""
        now = int(now_us) if now_us is not None else _time.now_us()
        permit = self.verify_delivery(conn, permit_id, now_us=now)
        repos_v4.update(
            conn,
            "delivery_permits",
            {"state": "delivered", "receipt_id": receipt_id},
            {"permit_id": permit.permit_id},
        )
        row = repos_v4.get(
            conn, "delivery_permits", {"permit_id": permit.permit_id}
        )
        return self._permit_from_row(row)

    # ------------------------------------------------------------------
    # open_dispatch (V4-12.02) — delegated to the transport broker
    # ------------------------------------------------------------------

    def open_dispatch(
        self,
        conn: sqlite3.Connection,
        permit: "DeliveryPermit | str",
        *,
        recipient: str,
        max_spend: float,
        payload_digest: Optional[str] = None,
        consent_refs: Iterable[str] = (),
        broker: Any = None,
        deadline_us: Optional[int] = None,
        est_tokens: int = 0,
        job_id: Optional[str] = None,
        input_refs: Any = None,
        now_us: Optional[int] = None,
    ) -> DispatchPermit:
        """Hand a sealed delivery to the transport egress broker.

        Kernel-side duties (inside the caller's tx): the delivery permit
        must be persisted, sealed, and unexpired, and a caller-supplied
        ``payload_digest`` must match the permit's binding (V4-12.02).
        Consent recheck, budget reservation, and endpoint allowlisting are
        delegated to a provisioned
        :class:`verbatim.privacy.broker.TransportBroker` — passed as
        ``broker`` or wired at ``Kernel(store, broker=...)``. The kernel
        imports the module lazily and never constructs a broker from thin
        air: no provisioned broker → ``CAPABILITY_UNAVAILABLE``, never a
        false permit. On success the delivery is recorded ``delivered`` in
        the same transaction.
        """
        now = int(now_us) if now_us is not None else _time.now_us()
        p = self.verify_delivery(conn, permit, now_us=now)
        require_id(recipient, "recipient")
        if payload_digest is not None and payload_digest != p.payload_digest:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "payload_digest does not match the delivery permit binding",
            )
        brk = broker if broker is not None else self._broker
        if brk is None:
            try:
                # Lazy import only — the egress plane is another worker's
                # module; its absence must degrade honestly.
                from ..privacy import broker as _mod
            except ImportError:
                _mod = None
            if getattr(_mod, "TransportBroker", None) is None:
                raise VerbatimError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "transport broker module is not provisioned",
                )
            raise VerbatimError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "no provisioned TransportBroker — pass broker= or wire "
                "Kernel(store, broker=...)",
            )
        dispatch = brk.open_dispatch(
            conn,
            caller=p.caller_id,
            recipient=recipient,
            purpose=p.purpose,
            payload_digest=p.payload_digest,
            scope_ids=tuple(p.epoch_vector.keys()),
            consent_refs=tuple(consent_refs),
            max_spend=max_spend,
            deadline_us=deadline_us if deadline_us is not None else p.expires_us,
            est_tokens=est_tokens,
            job_id=job_id,
            input_refs=input_refs,
        )
        if not isinstance(dispatch, DispatchPermit):
            raise VerbatimError(
                ErrorCode.STORE_WRITE_FAILED,
                "transport broker returned no dispatch permit",
            )
        # The one-use dispatch permit minted ⇒ the delivery handoff is
        # recorded delivered inside the same transaction.
        self.mark_delivered(conn, p.permit_id, now_us=now)
        return dispatch

    # ------------------------------------------------------------------
    # review_metadata (V4-08.06)
    # ------------------------------------------------------------------

    def review_metadata(
        self,
        conn: sqlite3.Connection,
        lease: EligibilityLease,
        fields: Mapping[str, Any],
        *,
        scope_id: Optional[str] = None,
        now_us: Optional[int] = None,
    ) -> ReviewedMetadata:
        """Disclosure review for scope-local metadata (V4-08.06).

        Counts, titles, snippets, identifiers, and error details only flow
        inside a verified structure under a live lease covering their
        scope. The lease check IS the review — a denied/expired/drifted
        lease withholds metadata exactly like primary content.
        """
        now = int(now_us) if now_us is not None else _time.now_us()
        self._require_lease(conn, lease, now)
        if scope_id is not None and scope_id not in lease.scope_ids:
            _deny()
        if not isinstance(fields, Mapping):
            raise VerbatimError(
                ErrorCode.VALIDATION, "metadata fields must be a mapping"
            )
        try:
            json_dumps(dict(fields))
        except (TypeError, ValueError) as exc:
            raise VerbatimError(
                ErrorCode.VALIDATION, "metadata fields must be JSON-serializable"
            ) from exc
        return ReviewedMetadata(
            fields=dict(fields),
            lease_id=lease.lease_id,
            scope_ids=lease.scope_ids,
            reviewed_us=now,
        )

    # ------------------------------------------------------------------
    # invalidate (V4-09.07, V4-10.08)
    # ------------------------------------------------------------------

    def invalidate(
        self,
        conn: sqlite3.Connection,
        event: Any,
        *,
        now_us: Optional[int] = None,
    ) -> InvalidationReport:
        """Apply an invalidation event: epoch bumps + object obligations.

        ``event`` is an ``InvalidationEvent``-shaped mapping
        (``{"kind": ..., "scope_ids": [...], "object_refs": [(kind,id,rev)]}``)
        for ``quarantine``/``correction``/``revocation``/``erasure``/
        ``lifecycle`` events. Each named scope's authorization epoch is
        bumped (absent scope rows record ``None`` honestly); the affected
        set expands over ``dependency_edges`` dependents (bounded);
        still-usable sealed permits touching the event's scopes are
        enumerated as honestly irreversible in-flight disclosures, and
        lapsed sealed permits are durably marked ``expired``.
        """
        now = int(now_us) if now_us is not None else _time.now_us()
        kind, scope_ids, refs = self._normalize_event(event)
        epochs_out: dict[str, Optional[int]] = {}
        for sid in scope_ids:
            epochs_out[sid] = _epochs.bump_epoch_if_present(conn, sid)

        affected: list[tuple[str, str, Optional[int]]] = []
        seen: set[tuple[str, str, Optional[int]]] = set()
        queue: list[tuple[str, str, Optional[int]]] = []
        for k, oid, rev in refs:
            key = (k, oid, rev)
            if key not in seen:
                seen.add(key)
                affected.append(key)
                queue.append(key)
        # Dependents: objects whose parent edge names an affected object
        # inherit the invalidation (correction/quarantine cascades down).
        steps = 0
        while queue and steps < _MAX_ANCESTORS:
            pk, pid, prev = queue.pop(0)
            steps += 1
            if prev is None:
                rows = repos_v4.query(
                    conn, "dependency_edges",
                    {"parent_kind": pk, "parent_id": pid},
                )
            else:
                rows = repos_v4.query(
                    conn, "dependency_edges",
                    {
                        "parent_kind": pk,
                        "parent_id": pid,
                        "parent_revision": prev,
                    },
                )
            for r in rows:
                key = (r["child_kind"], r["child_id"], int(r["child_revision"]))
                if key not in seen:
                    seen.add(key)
                    affected.append(key)
                    queue.append(key)

        obligations: list[dict[str, Any]] = []
        for k, oid, rev in affected:
            binding = self._resolve(conn, oid, k)
            obligations.append(
                {
                    "obligation": "reevaluate",
                    "event": kind,
                    "object_kind": k,
                    "object_id": oid,
                    "revision": rev,
                    "scope_id": binding.scope_id if binding else None,
                }
            )

        in_flight: list[str] = []
        expired: list[str] = []
        scope_set = set(scope_ids)
        for row in repos_v4.query(conn, "delivery_permits", {"state": "sealed"}):
            vector = safe_json_loads(row.get("epoch_vector_json") or "{}")
            touched = scope_set & set(vector.keys() if isinstance(vector, dict) else ())
            if int(row["expires_us"]) <= now:
                repos_v4.update(
                    conn, "delivery_permits", {"state": "expired"},
                    {"permit_id": row["permit_id"]},
                )
                expired.append(row["permit_id"])
            elif touched:
                # Committed before this event — enumerated, never
                # retro-denied (V4-09.07).
                in_flight.append(row["permit_id"])
        return InvalidationReport(
            event_kind=kind,
            epochs=epochs_out,
            affected=tuple(affected),
            obligations=tuple(obligations),
            in_flight_permits=tuple(in_flight),
            expired_permits=tuple(expired),
            evaluated_us=now,
        )

    # ------------------------------------------------------------------
    # lease gate
    # ------------------------------------------------------------------

    def _require_lease(
        self, conn: sqlite3.Connection, lease: EligibilityLease, now: int
    ) -> None:
        """Deny dead leases: denied → indistinguishable; expired →
        PERMIT_EXPIRED; epoch drift → STALE_EPOCH (V4-08.04)."""
        if not isinstance(lease, EligibilityLease):
            raise VerbatimError(
                ErrorCode.VALIDATION, "expected an EligibilityLease"
            )
        if lease.denied:
            _deny()
        if lease.expired(now):
            raise VerbatimError(
                ErrorCode.PERMIT_EXPIRED,
                "eligibility lease expired — re-resolve access",
                retryable=True,
            )
        for sid, pinned in lease.epoch_vector.items():
            if _epochs.current_epoch(conn, sid) != int(pinned):
                raise VerbatimError(
                    ErrorCode.STALE_EPOCH,
                    "authorization epoch superseded",
                    retryable=True,
                )

    # ------------------------------------------------------------------
    # object resolution + constraint evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _scope_id(value: Any) -> str:
        if not isinstance(value, str) or not value:
            raise VerbatimError(
                ErrorCode.VALIDATION, "scope ids must be non-empty strings"
            )
        return require_id(value, "scope_id")

    def _normalize_refs(self, object_refs: Iterable[Any]) -> list[tuple[Optional[str], str, int]]:
        """(kind, id, rev) | (id, rev) | dict | EvidenceLocator → triples."""
        out: list[tuple[Optional[str], str, int]] = []
        for ref in object_refs or ():
            kind: Optional[str] = None
            oid: Any = None
            rev: Any = None
            if isinstance(ref, EvidenceLocator):
                oid, rev = ref.object_id, ref.revision
            elif isinstance(ref, Mapping):
                kind = ref.get("object_kind") or ref.get("kind")
                oid = ref.get("object_id") or ref.get("id")
                rev = ref.get("revision")
            else:
                try:
                    parts = tuple(ref)
                except TypeError:
                    parts = ()
                if len(parts) == 3:
                    kind, oid, rev = parts
                elif len(parts) == 2:
                    oid, rev = parts
            if oid is None or rev is None:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "object refs must be (object_id, revision) or "
                    "(object_kind, object_id, revision)",
                )
            if isinstance(rev, bool) or not isinstance(rev, int) or rev < 0:
                raise VerbatimError(
                    ErrorCode.VALIDATION, "object revision must be a non-negative int"
                )
            if kind is not None:
                require_id(kind, "object_kind")
            out.append((kind, require_id(oid, "object_id"), int(rev)))
        return out

    def _resolve(
        self,
        conn: sqlite3.Connection,
        object_id: str,
        kind_hint: Optional[str] = None,
    ) -> Optional[_Binding]:
        """Resolve an object id to (kind, scope, lifecycle carrier).

        Ambiguous or absent ids return None → the caller denies. Id
        collisions across kinds are ambiguous by construction.
        """
        matches: list[_Binding] = []
        kinds = (kind_hint,) if kind_hint else None

        # v4 registry first — it carries the authoritative disposition.
        where: dict[str, Any] = {"object_id": object_id}
        if kind_hint is not None:
            where["kind"] = kind_hint
        for r in repos_v4.query(conn, "objects", where):
            matches.append(
                _Binding(
                    kind=r["kind"],
                    object_id=object_id,
                    scope_id=r["scope_id"],
                    registered=True,
                    disposition=r["disposition"],
                    current_revision=int(r["current_revision"]),
                )
            )

        def legacy() -> None:
            if kinds is None or "source" in kinds:
                r = _row(
                    conn.execute(
                        "SELECT scope_id FROM sources WHERE source_id = ?",
                        (object_id,),
                    )
                )
                if r is not None:
                    matches.append(
                        _Binding("source", object_id, r["scope_id"])
                    )
            if kinds is None or "span" in kinds:
                r = _row(
                    conn.execute(
                        "SELECT s.source_id AS source_id, s.start_byte AS st,"
                        " s.end_byte AS en, src.scope_id AS scope_id"
                        " FROM spans s JOIN sources src"
                        "   ON src.source_id = s.source_id"
                        " WHERE s.span_id = ?",
                        (object_id,),
                    )
                )
                if r is not None:
                    matches.append(
                        _Binding(
                            "span", object_id, r["scope_id"],
                            source_id=r["source_id"],
                            span_start=int(r["st"]), span_end=int(r["en"]),
                        )
                    )
            if kinds is None or "claim" in kinds:
                r = _row(
                    conn.execute(
                        "SELECT scope_id FROM claims WHERE claim_id = ?",
                        (object_id,),
                    )
                )
                if r is not None:
                    matches.append(_Binding("claim", object_id, r["scope_id"]))
            if kinds is None or "source_envelope" in kinds:
                r = _row(
                    conn.execute(
                        "SELECT scope_id, source_id FROM source_envelopes"
                        " WHERE envelope_id = ?",
                        (object_id,),
                    )
                )
                if r is not None:
                    matches.append(
                        _Binding(
                            "source_envelope", object_id, r["scope_id"],
                            source_id=r["source_id"],
                        )
                    )

        legacy()
        distinct = {m.kind for m in matches}
        if not matches or len(distinct) != 1:
            # Absent or ambiguous (same id under multiple kinds) — both deny.
            return None
        if len(matches) > 1:
            # Same object visible through registry + legacy table: merge,
            # preferring the registry's lifecycle fields.
            reg = next((m for m in matches if m.registered), None)
            leg = next((m for m in matches if not m.registered), None)
            if reg is not None and leg is not None:
                return _Binding(
                    kind=reg.kind,
                    object_id=reg.object_id,
                    scope_id=reg.scope_id,
                    registered=True,
                    disposition=reg.disposition,
                    current_revision=reg.current_revision,
                    source_id=leg.source_id,
                    span_start=leg.span_start,
                    span_end=leg.span_end,
                )
            # Two same-kind legacy hits can't happen (PK per table), but
            # never guess — deny on any residual ambiguity.
            return None
        return matches[0]

    def _check_object(
        self, conn: sqlite3.Connection, binding: _Binding, revision: int
    ) -> None:
        """Lifecycle + quarantine + suppression gates (V4-08.01)."""
        if binding.registered:
            if binding.disposition != LifecycleState.ACTIVE.value:
                _deny()
            if (
                binding.current_revision is not None
                and revision > binding.current_revision
            ):
                _deny()
        if binding.kind == "claim":
            row = _row(
                conn.execute(
                    "SELECT state FROM claim_revisions"
                    " WHERE claim_id = ? AND revision = ?",
                    (binding.object_id, revision),
                )
            )
            if row is None or row["state"] not in _CLAIM_VISIBLE_STATES:
                _deny()
        if binding.kind == "source":
            row = _row(
                conn.execute(
                    "SELECT 1 FROM source_revisions"
                    " WHERE source_id = ? AND revision = ?",
                    (binding.object_id, revision),
                )
            )
            if row is None:
                _deny()
        if binding.kind == "span" and revision != self._span_revision(
            conn, binding.object_id
        ):
            _deny()
        for k, oid, rev in self._held_refs(conn, binding, revision):
            if _quarantine.is_quarantined(conn, k, oid, rev):
                _deny()
        if self._suppressed_refs(conn, binding, revision):
            _deny()

    def _span_revision(self, conn: sqlite3.Connection, span_id: str) -> Optional[int]:
        row = conn.execute(
            "SELECT revision FROM spans WHERE span_id = ?", (span_id,)
        ).fetchone()
        return int(row[0]) if row else None

    def _held_refs(
        self, conn: sqlite3.Connection, binding: _Binding, revision: int
    ) -> set[tuple[str, str, int]]:
        """Quarantine refs: the object plus its ancestors (V3-14.10 cascade)."""
        refs = {(binding.kind, binding.object_id, revision)}
        if binding.kind == "span" and binding.source_id:
            refs.add(("source", binding.source_id, revision))
            refs |= self._envelope_refs(conn, binding.source_id, revision)
        elif binding.kind == "source":
            refs |= self._envelope_refs(conn, binding.object_id, revision)
        elif binding.kind == "source_envelope" and binding.source_id:
            refs.add(("source", binding.source_id, revision))
        elif binding.kind == "claim":
            spans = conn.execute(
                "SELECT span_id FROM claim_evidence"
                " WHERE claim_id = ? AND revision = ? LIMIT ?",
                (binding.object_id, revision, _MAX_CLAIM_SPANS),
            ).fetchall()
            for (span_id,) in spans:
                srev = self._span_revision(conn, span_id)
                if srev is None:
                    continue
                refs.add(("span", span_id, srev))
                src = conn.execute(
                    "SELECT source_id FROM spans WHERE span_id = ?",
                    (span_id,),
                ).fetchone()
                if src:
                    refs.add(("source", src[0], srev))
                    refs |= self._envelope_refs(conn, src[0], srev)
        if binding.registered:
            refs |= self._parent_refs(conn, binding, revision)
        return refs

    def _envelope_refs(
        self, conn: sqlite3.Connection, source_id: str, revision: int
    ) -> set[tuple[str, str, int]]:
        rows = conn.execute(
            "SELECT envelope_id FROM source_envelopes"
            " WHERE source_id = ? AND revision = ?",
            (source_id, revision),
        ).fetchall()
        return {("source_envelope", r[0], revision) for r in rows}

    def _parent_refs(
        self, conn: sqlite3.Connection, binding: _Binding, revision: int
    ) -> set[tuple[str, str, int]]:
        """Dependency-edge parents — contributing ancestry (V4-08.01)."""
        out: set[tuple[str, str, int]] = set()
        queue = [(binding.kind, binding.object_id, revision)]
        seen = set(queue)
        steps = 0
        while queue and steps < _MAX_ANCESTORS:
            k, oid, rev = queue.pop(0)
            steps += 1
            for r in repos_v4.query(
                conn, "dependency_edges",
                {"child_kind": k, "child_id": oid, "child_revision": rev},
            ):
                key = (r["parent_kind"], r["parent_id"], int(r["parent_revision"]))
                if key not in seen:
                    seen.add(key)
                    out.add(key)
                    queue.append(key)
        return out

    def _suppressed_refs(
        self, conn: sqlite3.Connection, binding: _Binding, revision: int
    ) -> bool:
        """True when the object or an ancestor sits in a suppressing purge."""
        pairs = {(binding.kind, binding.object_id)}
        if binding.kind == "source":
            pairs.add(("source_revision", f"{binding.object_id}:{revision}"))
        elif binding.kind == "span" and binding.source_id:
            pairs.add(("source", binding.source_id))
            pairs.add(
                ("source_revision", f"{binding.source_id}:{revision}")
            )
        elif binding.kind == "source_envelope" and binding.source_id:
            pairs.add(("source", binding.source_id))
            pairs.add(
                ("source_revision", f"{binding.source_id}:{revision}")
            )
        states = ", ".join("?" for _ in _SUPPRESSING_STATES)
        for kind, oid in pairs:
            row = conn.execute(
                "SELECT 1 FROM purge_targets pt JOIN purges p"
                "  ON p.purge_id = pt.purge_id"
                f" WHERE pt.object_kind = ? AND pt.object_id = ?"
                f"   AND p.state IN ({states}) LIMIT 1",
                (kind, oid, *_SUPPRESSING_STATES),
            ).fetchone()
            if row is not None:
                return True
        return False

    # ------------------------------------------------------------------
    # verified byte read (single path — V4-08.09)
    # ------------------------------------------------------------------

    def _read_one(
        self,
        conn: sqlite3.Connection,
        lease: EligibilityLease,
        loc: EvidenceLocator,
    ) -> VerifiedSlice:
        binding = self._resolve(conn, loc.object_id)
        if binding is None or binding.scope_id not in lease.scope_ids:
            _deny()
        if lease.object_refs and (
            (binding.object_id, loc.revision) not in lease.object_refs
        ):
            _deny()
        self._check_object(conn, binding, loc.revision)

        # Surface selection: canonical payload, a named view, or a span's
        # own authenticated excerpt. Other kinds have no byte surface.
        # Stored digests authenticate the FULL fetched object (payload_hmac
        # covers the whole revision payload; integrity_digest the whole
        # view); the span's excerpt_hmac is the one slice-level binding.
        canonical = True
        verified_stored = False
        data: bytes
        provenance: dict[str, Any] = {"kind": binding.kind}
        if binding.kind == "span":
            if (
                loc.view_id is not None
                or binding.span_start is None
                or (loc.start_byte, loc.end_byte)
                != (binding.span_start, binding.span_end)
            ):
                _deny()
            payload, payload_hmac, excerpt_hmac = self._span_bytes(
                conn, binding
            )
            if not payload:
                _deny()  # purged/emptied parent — indistinguishable absent
            if payload_hmac is not None:
                self._verify_digest(
                    payload_hmac, payload,
                    f"source {binding.source_id}@{loc.revision} payload",
                )
            data = payload[binding.span_start : binding.span_end]
            if excerpt_hmac is not None:
                self._verify_digest(
                    excerpt_hmac, data,
                    f"span {binding.object_id} excerpt",
                )
            verified_stored = (
                payload_hmac is not None and excerpt_hmac is not None
            )
            provenance["source_id"] = binding.source_id
            provenance["byte_length"] = len(data)
        elif binding.kind == "source":
            if loc.view_id is None:
                payload, stored_digest = self._source_bytes(conn, binding, loc.revision)
                if not payload:
                    _deny()
                data = self._slice(payload, loc)
                verified_stored = stored_digest is not None
                provenance.update(
                    self._source_meta(conn, binding.object_id, loc.revision)
                )
            else:
                view = self._view_row(conn, binding, loc.revision, loc.view_id)
                canonical = view["view_kind"] == "primary"
                if canonical:
                    payload, stored_digest = self._source_bytes(
                        conn, binding, loc.revision
                    )
                    if not payload:
                        _deny()
                    data = self._slice(payload, loc)
                    verified_stored = stored_digest is not None
                else:
                    derived = view["derived_bytes"]
                    if derived is None:
                        _deny()
                    blob = bytes(derived)
                    if view["integrity_digest"] is not None:
                        self._verify_digest(
                            bytes(view["integrity_digest"]), blob,
                            f"view {loc.view_id} on "
                            f"{binding.object_id}@{loc.revision}",
                        )
                        verified_stored = True
                    data = self._slice(blob, loc)
                    provenance["byte_length"] = len(blob)
                provenance["view_id"] = loc.view_id
                provenance["view_kind"] = view["view_kind"]
                provenance["media_type"] = view["media_type"]
        else:
            _deny()

        if lease.verb is not Verb.QUOTE:
            # V4-08.02: exact original bytes require quote. read releases
            # authorized metadata + approved derived views only.
            if canonical or lease.verb is not Verb.READ:
                return VerifiedSlice(
                    locator=loc,
                    data=b"",
                    digest="",
                    algorithm="none",
                    provenance=provenance,
                    verification="metadata_only",
                )
        verification = "verified" if verified_stored else "legacy_unverified"
        return VerifiedSlice(
            locator=loc,
            data=data,
            digest=self._slice_digest(binding.kind, loc, data),
            algorithm=_ALGORITHM,
            provenance=provenance,
            verification=verification,
        )

    def _source_bytes(
        self, conn: sqlite3.Connection, binding: _Binding, revision: int
    ) -> tuple[bytes, Optional[bytes]]:
        row = conn.execute(
            "SELECT payload, payload_hmac FROM source_revisions"
            " WHERE source_id = ? AND revision = ?",
            (binding.object_id, revision),
        ).fetchone()
        if row is None or row[0] is None:
            _deny()
        payload = bytes(row[0])
        stored = bytes(row[1]) if row[1] is not None else None
        if stored is not None:
            self._verify_digest(
                stored, payload,
                f"source {binding.object_id}@{revision} payload",
            )
        return payload, stored

    def _span_bytes(
        self, conn: sqlite3.Connection, binding: _Binding
    ) -> tuple[bytes, Optional[bytes], Optional[bytes]]:
        row = conn.execute(
            "SELECT r.payload, r.payload_hmac, s.excerpt_hmac"
            " FROM spans s JOIN source_revisions r"
            "   ON r.source_id = s.source_id AND r.revision = s.revision"
            " WHERE s.span_id = ?",
            (binding.object_id,),
        ).fetchone()
        if row is None or row[0] is None:
            _deny()
        return (
            bytes(row[0]),
            bytes(row[1]) if row[1] is not None else None,
            bytes(row[2]) if row[2] is not None else None,
        )

    def _view_row(
        self, conn: sqlite3.Connection, binding: _Binding, revision: int, view_id: str
    ) -> dict[str, Any]:
        row = _row(
            conn.execute(
                "SELECT view_kind, media_type, derived_bytes, integrity_digest"
                " FROM source_views"
                " WHERE source_id = ? AND revision = ? AND view_id = ?",
                (binding.object_id, revision, view_id),
            )
        )
        if row is None:
            _deny()
        return row

    def _source_meta(
        self, conn: sqlite3.Connection, source_id: str, revision: int
    ) -> dict[str, Any]:
        row = _row(
            conn.execute(
                "SELECT provenance, metadata_json, length(payload) AS n"
                " FROM source_revisions"
                " WHERE source_id = ? AND revision = ?",
                (source_id, revision),
            )
        )
        if row is None:
            return {}
        return {
            "provenance": row["provenance"],
            "byte_length": int(row["n"]),
            "media_type": "text/plain",
        }

    @staticmethod
    def _slice(payload: bytes, loc: EvidenceLocator) -> bytes:
        # Payload length is private metadata — an out-of-bounds locator
        # denies indistinguishably rather than sizing the object.
        if not (0 <= loc.start_byte <= loc.end_byte <= len(payload)):
            _deny()
        return payload[loc.start_byte : loc.end_byte]

    def _verify_digest(self, stored: bytes, data: bytes, what: str) -> None:
        if not hmac.compare_digest(self._store.hmac(data), stored):
            raise VerbatimError(
                ErrorCode.STORE_CORRUPT, f"{what} fails integrity check"
            )

    def _slice_digest(
        self, kind: str, loc: EvidenceLocator, data: bytes
    ) -> str:
        """Domain-separated digest binding store, object type, identity,
        revision, locator, and algorithm (V4-13.04)."""
        bind = json_dumps(
            {
                "kind": kind,
                "object_id": loc.object_id,
                "revision": loc.revision,
                "start": loc.start_byte,
                "end": loc.end_byte,
                "view_id": loc.view_id,
                "algorithm": _ALGORITHM,
            }
        ).encode("utf-8")
        return self._store.hmac(_SLICE_DOMAIN + bind + b"\x00" + data).hex()

    # ------------------------------------------------------------------
    # derive_inputs helpers
    # ------------------------------------------------------------------

    def _producer_grant(
        self, conn: sqlite3.Connection, producer_grant: Any
    ) -> Optional[dict]:
        row: Optional[dict] = None
        if isinstance(producer_grant, str):
            row = _grants.get_grant(conn, producer_grant)
        elif isinstance(producer_grant, Mapping):
            row = dict(producer_grant)
        elif hasattr(producer_grant, "grant_id"):
            row = _grants.get_grant(conn, producer_grant.grant_id)
        if row is None:
            return None
        now = _time.now_us()
        if row.get("revoked_us") is not None:
            return None
        exp = row.get("expires_us")
        if exp is not None and exp <= now:
            return None
        return row

    def _normalize_input_refs(self, refs: Iterable[Any]) -> list[tuple]:
        """(kind,id,rev)|(id,rev)|dict|EvidenceLocator →
        (kind, oid, rev, start|None, end|None, view_id|None)."""
        out: list[tuple] = []
        for ref in refs or ():
            kind = oid = rev = start = end = view_id = None
            if isinstance(ref, EvidenceLocator):
                oid, rev = ref.object_id, ref.revision
                start, end, view_id = ref.start_byte, ref.end_byte, ref.view_id
            elif isinstance(ref, Mapping):
                kind = ref.get("object_kind") or ref.get("kind")
                oid = ref.get("object_id") or ref.get("id")
                rev = ref.get("revision")
                start, end = ref.get("start_byte"), ref.get("end_byte")
                view_id = ref.get("view_id")
            else:
                try:
                    parts = tuple(ref)
                except TypeError:
                    parts = ()
                if len(parts) == 3:
                    kind, oid, rev = parts
                elif len(parts) == 2:
                    oid, rev = parts
            if oid is None or rev is None:
                raise VerbatimError(
                    ErrorCode.VALIDATION, "input refs need object id + revision"
                )
            if isinstance(rev, bool) or not isinstance(rev, int) or rev < 0:
                raise VerbatimError(
                    ErrorCode.VALIDATION, "input revision must be a non-negative int"
                )
            if kind is not None:
                require_id(kind, "object_kind")
            out.append((kind, require_id(oid, "object_id"), int(rev), start, end, view_id))
        return out

    def _full_range(
        self, conn: sqlite3.Connection, binding: _Binding, revision: int
    ) -> tuple[int, int]:
        """A range-less input ref means the object's full verified extent."""
        if binding.kind == "span":
            assert binding.span_start is not None
            return binding.span_start, binding.span_end
        if binding.kind == "source":
            row = conn.execute(
                "SELECT length(payload) FROM source_revisions"
                " WHERE source_id = ? AND revision = ?",
                (binding.object_id, revision),
            ).fetchone()
            if row is None:
                _deny()
            return 0, int(row[0])
        _deny()

    def _purpose_intersection(
        self,
        conn: sqlite3.Connection,
        principal_id: str,
        scope_ids: Sequence[str],
        purpose: Optional[str],
        now: int,
    ) -> PurposeConstraint:
        """⋂ over contributing scopes of the producer's effective purpose
        set for input access — V4-10.07 inherited restriction."""
        if not scope_ids:
            return PurposeConstraint.none()
        result: Optional[frozenset] = None
        for sid in scope_ids:
            pinned = _epochs.current_epoch(conn, sid)
            scope_purposes: set[str] = set()
            unrestricted = False
            for row in _grants.grants_for(conn, sid, principal_id):
                if not _grants._row_live(row, now):
                    continue
                # Effective input access = a live quote grant chain.
                if not _grants._grant_applies(
                    conn, row, Verb.QUOTE, purpose, now, pinned
                ):
                    continue
                purps = frozenset(repos_v3.json_field(row, "purposes_json") or ())
                if not purps:
                    unrestricted = True  # unbound grant ⇒ any purpose
                else:
                    scope_purposes |= set(purps)
            contrib = None if unrestricted else frozenset(scope_purposes)
            if contrib is None:
                continue  # ANY is the identity element of ∩
            result = contrib if result is None else (result & contrib)
        if result is None:
            return PurposeConstraint.any()
        return PurposeConstraint.set(result)

    def _audience_intersection(
        self,
        conn: sqlite3.Connection,
        audience: Iterable[str],
        scope_ids: Sequence[str],
        now: int,
    ) -> tuple[str, ...]:
        """Requested audience members holding a live grant in EVERY
        contributing scope (C48: a derivative is unavailable to a
        recipient denied either required parent)."""
        out: list[str] = []
        for member in dict.fromkeys(audience or ()):
            require_id(member, "audience member")
            ok = True
            for sid in scope_ids:
                rows = _grants.grants_for(conn, sid, member)
                if not any(_grants._row_live(r, now) for r in rows):
                    ok = False
                    break
            if ok:
                out.append(member)
        return tuple(out)

    # ------------------------------------------------------------------
    # seal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _max_age(requested: Optional[int]) -> int:
        # V4-09.08: one second default; callers may shorten, never widen.
        default = EligibilityLease.DEFAULT_MAX_AGE_US
        if requested is None:
            return default
        if (
            isinstance(requested, bool)
            or not isinstance(requested, int)
            or not (0 < requested <= default)
        ):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"max_age_us must be in (0, {default}]",
            )
        return requested

    def _check_dependencies(
        self,
        conn: sqlite3.Connection,
        lease: EligibilityLease,
        dependency_versions: Optional[Mapping[str, int]],
    ) -> dict[str, int]:
        """Recheck every dependency's current revision inside the seal tx.
        A denied/absent/stale dependency invalidates the group (V4-08.05,
        V4-09.06)."""
        deps: dict[str, int] = {}
        for oid, expected in (dependency_versions or {}).items():
            require_id(oid, "dependency object_id")
            if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
                raise VerbatimError(
                    ErrorCode.VALIDATION, "dependency revisions are non-negative ints"
                )
            binding = self._resolve(conn, oid)
            if binding is None or binding.scope_id not in lease.scope_ids:
                _deny()
            current = self._current_revision(conn, binding)
            if current is None:
                _deny()
            if current != expected:
                raise VerbatimError(
                    ErrorCode.STALE_DEPENDENCY,
                    "dependency revision superseded — rebuild the pack",
                    retryable=True,
                )
            # Constraints at the (now-matched) pinned revision.
            self._check_object(conn, binding, expected)
            deps[oid] = expected
        return deps

    def _current_revision(
        self, conn: sqlite3.Connection, binding: _Binding
    ) -> Optional[int]:
        if binding.registered and binding.current_revision is not None:
            return binding.current_revision
        table, col = {
            "source": ("source_revisions", "source_id"),
            "claim": ("claim_revisions", "claim_id"),
            "span": ("spans", "span_id"),
            "source_envelope": ("source_envelopes", "envelope_id"),
        }.get(binding.kind, (None, None))
        if table is None:
            return None
        row = conn.execute(
            f"SELECT MAX(revision) FROM {table} WHERE {col} = ?",
            (binding.object_id,),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None

    @staticmethod
    def _permit_from_row(row: dict[str, Any]) -> DeliveryPermit:
        vector = safe_json_loads(row.get("epoch_vector_json") or "{}")
        deps = safe_json_loads(row.get("dependency_versions_json") or "{}")
        return DeliveryPermit(
            permit_id=row["permit_id"],
            caller_id=row["caller_id"],
            purpose=row["purpose"],
            epoch_vector={k: int(v) for k, v in (vector or {}).items()},
            payload_digest=row["payload_digest"],
            dependency_versions={k: int(v) for k, v in (deps or {}).items()},
            state=row["state"],
            issued_us=int(row["issued_us"]),
            expires_us=int(row["expires_us"]),
            receipt_id=row.get("receipt_id"),
        )

    # ------------------------------------------------------------------
    # invalidate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_event(
        event: Any,
    ) -> tuple[str, tuple[str, ...], list[tuple[str, str, Optional[int]]]]:
        if isinstance(event, Mapping):
            kind = event.get("kind") or event.get("event_kind")
            scopes = event.get("scope_ids") or ()
            refs = event.get("object_refs") or ()
        else:
            kind = getattr(event, "kind", None) or getattr(event, "event_kind", None)
            scopes = getattr(event, "scope_ids", ())
            refs = getattr(event, "object_refs", ())
        if kind not in INVALIDATION_KINDS:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"invalidation kind must be one of {sorted(INVALIDATION_KINDS)}",
            )
        scope_ids = tuple(
            dict.fromkeys(
                Kernel._scope_id(s) for s in scopes
            )
        )
        if not scope_ids:
            raise VerbatimError(
                ErrorCode.VALIDATION, "invalidation events name >= 1 scope"
            )
        out: list[tuple[str, str, Optional[int]]] = []
        for ref in refs or ():
            if isinstance(ref, Mapping):
                k = ref.get("object_kind") or ref.get("kind")
                oid = ref.get("object_id") or ref.get("id")
                rev = ref.get("revision")
            else:
                parts = tuple(ref)
                if len(parts) == 3:
                    k, oid, rev = parts
                elif len(parts) == 2:
                    k, oid = parts
                    rev = None
                else:
                    raise VerbatimError(
                        ErrorCode.VALIDATION,
                        "object refs are (kind, id[, revision])",
                    )
            require_id(k, "object_kind")
            require_id(oid, "object_id")
            if rev is not None and (
                isinstance(rev, bool) or not isinstance(rev, int) or rev < 0
            ):
                raise VerbatimError(
                    ErrorCode.VALIDATION, "revision must be a non-negative int"
                )
            out.append((k, oid, rev))
        return str(kind), scope_ids, out


#: Backwards-friendly alias — both names name the §08 kernel.
EvidenceKernel = Kernel

__all__ = [
    "DENIAL_MESSAGE",
    "DerivedInputs",
    "EvidenceKernel",
    "INVALIDATION_KINDS",
    "InvalidationReport",
    "Kernel",
    "ReviewedMetadata",
]
