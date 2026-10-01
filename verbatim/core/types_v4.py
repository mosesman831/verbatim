"""V4 core types: evidence kernel, effect plans, permits, closure, readiness.

Frozen contract for Verbatim v4 (SPEC_V4). Mirrors ``core/types_v3.py``
conventions — frozen dataclasses, str-valued enums, ``VerbatimError`` /
``ErrorCode``. V4 types import shared vocabulary from ``core/types`` and may
reference ``core/types_v3`` where a contract is inherited unchanged.

Design rules baked in here (SPEC_V4 §08–§14, §38, §41–§43):

- Purpose constraints are *tagged* ANY / SET / NONE (V4-11.01): an empty SET
  normalizes to NONE, never to ANY. The ambiguous bare-frozenset encoding of
  v3 grants is migrated to explicit tags at schema v4.
- An ``EffectPlan`` is computed outside transactions; ``apply_plan`` verifies
  lease + authorization + expected versions inside one fenced transaction
  (V4-09.02). No domain effect commits outside the coordinator.
- Permits are capability-bound values, not byte strings with comments
  (V4-08.04). In-process leases bind caller, operation, epoch vector, and
  lifetime; serialized signed forms exist only for untrusted boundaries.
- Disclosure linearizes at ``seal_delivery`` commit (V4-09.06): revocation
  before permit commit blocks it; after commit the disclosure is recorded as
  irreversible, not denied retroactively.
- Closure runs persist frontier + cursor so deletion is resumable and never
  reports terminal success at a bound (V4-38.03/04).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .types import (
    ErrorCode,
    VerbatimError,
    new_id,
    require_id,
)
from .types_v3 import Verb

# ---------------------------------------------------------------------------
# Purpose constraints (V4-11.01) — tagged ANY / SET / NONE
# ---------------------------------------------------------------------------


class PurposeTag(str, enum.Enum):
    """The three explicit purpose-constraint forms (V4-11.01)."""

    ANY = "any"
    SET = "set"
    NONE = "none"


@dataclass(frozen=True)
class PurposeConstraint:
    """Tagged purpose restriction for grants, delegations, and requests.

    ``tag=ANY`` permits any declared purpose; ``SET`` permits only listed
    purposes; ``NONE`` permits none. An empty SET normalizes to NONE —
    an empty restriction set can never silently mean ANY (F4-03).
    """

    tag: PurposeTag
    values: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.tag, PurposeTag):
            object.__setattr__(self, "tag", PurposeTag(self.tag))
        object.__setattr__(self, "values", frozenset(self.values))
        for v in self.values:
            if not isinstance(v, str) or not v:
                raise VerbatimError(ErrorCode.VALIDATION, "purpose names are non-empty strings")
        if self.tag is PurposeTag.SET and not self.values:
            # V4-11.01: empty SET normalizes to NONE.
            object.__setattr__(self, "tag", PurposeTag.NONE)
        if self.tag is not PurposeTag.SET and self.values:
            raise VerbatimError(
                ErrorCode.VALIDATION, "purpose values only valid on SET tag"
            )

    @classmethod
    def any(cls) -> "PurposeConstraint":
        return cls(PurposeTag.ANY)

    @classmethod
    def none(cls) -> "PurposeConstraint":
        return cls(PurposeTag.NONE)

    @classmethod
    def set(cls, values: Iterable[str]) -> "PurposeConstraint":
        return cls(PurposeTag.SET, frozenset(values))

    def permits(self, purpose: Optional[str]) -> bool:
        """True when a request declaring ``purpose`` satisfies this constraint."""
        if self.tag is PurposeTag.ANY:
            return True
        if self.tag is PurposeTag.NONE:
            return False
        return purpose is not None and purpose in self.values

    def is_subset_of(self, parent: "PurposeConstraint") -> bool:
        """Attenuation check (V4-11.02): child authority ⊆ parent authority.

        NONE ⊆ everything. ANY ⊆ only ANY. SET ⊆ ANY or a superset SET.
        """
        if self.tag is PurposeTag.NONE:
            return True
        if parent.tag is PurposeTag.ANY:
            return True
        if self.tag is PurposeTag.ANY:
            return False
        # self is SET, parent is SET or NONE
        if parent.tag is PurposeTag.NONE:
            return False
        return self.values <= parent.values

    def to_json(self) -> dict[str, Any]:
        return {"tag": self.tag.value, "values": sorted(self.values)}

    @classmethod
    def from_json(cls, raw: Any) -> "PurposeConstraint":
        """Parse persisted tagged form. Legacy bare lists migrate to SET."""
        if raw is None:
            return cls.any()
        if isinstance(raw, (list, tuple)):
            # Legacy ambiguous encoding: bare list = SET (V4-62.05 migration
            # maps the old ambiguous empty-set separately at schema v4).
            return cls.set(raw)
        if isinstance(raw, dict):
            tag = raw.get("tag")
            if tag == "any":
                return cls.any()
            if tag == "none":
                return cls.none()
            if tag == "set":
                return cls.set(raw.get("values") or ())
        raise VerbatimError(ErrorCode.VALIDATION, f"invalid purpose constraint: {raw!r}")


# ---------------------------------------------------------------------------
# Atomic effect plans (V4-09)
# ---------------------------------------------------------------------------


class EffectKind(str, enum.Enum):
    """Domain effects a plan may declare. Each kind maps to exactly one
    coordinator apply handler — no handler may self-transact (V4-42)."""

    INSERT_OBJECT = "insert_object"          # objects + object_revisions row
    UPDATE_LIFECYCLE = "update_lifecycle"    # disposition/state transition
    INSERT_EDGE = "insert_edge"              # dependency_edges row
    INVALIDATE = "invalidate"                # epoch bump + cache/view suppression
    RECORD_RECEIPT = "record_receipt"        # operation/delivery/checker receipt
    PUBLISH_INDEX = "publish_index"          # projection generation publication
    WRITE_PAYLOAD = "write_payload"          # canonical evidence bytes
    ENQUEUE = "enqueue"                      # follow-up job obligation


@dataclass(frozen=True)
class Effect:
    """One intended domain effect inside an ``EffectPlan``."""

    kind: EffectKind
    table: str                    # registered target table
    payload: dict[str, Any]       # row fields / transition args
    expected_revision: Optional[int] = None  # optimistic concurrency guard

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EffectKind):
            object.__setattr__(self, "kind", EffectKind(self.kind))


@dataclass(frozen=True)
class JobRequest:
    """A follow-up job obligation inside a plan (enqueued atomically)."""

    kind: str                     # JobKind value
    lane: str                     # one of jobs.queue.LANES
    input_refs: dict[str, Any]
    dedup_key: Optional[str] = None
    not_before_us: int = 0


@dataclass(frozen=True)
class EffectPlan:
    """A complete, fenced unit of authoritative work (V4-09.01).

    Computed by policy/model components OUTSIDE any transaction; applied by
    the coordinator inside one transaction that rechecks the lease,
    authorization, expected revisions, and epoch vector.
    """

    operation_id: str             # idempotency key — replay returns prior receipt
    scope_id: str
    producer_id: str              # producer manifest identity
    input_digests: tuple[str, ...]
    expected_revisions: dict[str, int] = field(default_factory=dict)
    epoch_vector: dict[str, int] = field(default_factory=dict)
    effects: tuple[Effect, ...] = ()
    follow_ups: tuple[JobRequest, ...] = ()
    lease_token: int = 0          # fencing token from the job lease (0 = foreground)
    created_us: int = 0
    deadline_us: int = 0

    def __post_init__(self) -> None:
        require_id(self.operation_id, "operation_id")
        require_id(self.scope_id, "scope_id")
        object.__setattr__(self, "effects", tuple(self.effects))
        object.__setattr__(self, "follow_ups", tuple(self.follow_ups))
        object.__setattr__(
            self, "input_digests", tuple(self.input_digests)
        )


@dataclass(frozen=True)
class OperationReceipt:
    """The durable, replayable result of ``apply_plan`` (V4-09.05)."""

    operation_id: str
    scope_id: str
    input_digest: str             # digest over sorted input_digests
    applied_seq: int              # event sequence at commit
    result_ref: Optional[str]     # primary produced object, if any
    effects_applied: int
    jobs_enqueued: tuple[str, ...] = ()
    created_us: int = 0


# ---------------------------------------------------------------------------
# Evidence-access and delivery kernel (V4-08)
# ---------------------------------------------------------------------------


class LifecycleState(str, enum.Enum):
    """Object lifecycle for access resolution (V4-08.01)."""

    ACTIVE = "active"
    HELD = "held"                 # quarantined / suppressed pending review
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    ERASED = "erased"


@dataclass(frozen=True)
class EvidenceLocator:
    """A validated byte-range or artifact reference into a source/view
    revision (V4-13.03). Byte ranges are UTF-8-validated; malformed ranges
    are rejected at construction, never replace-decoded into minted offsets."""

    object_id: str
    revision: int
    start_byte: int
    end_byte: int
    view_id: Optional[str] = None  # None = canonical source revision

    def __post_init__(self) -> None:
        require_id(self.object_id, "object_id")
        if self.revision < 0:
            raise VerbatimError(ErrorCode.VALIDATION, "revision < 0")
        if not (0 <= self.start_byte <= self.end_byte):
            raise VerbatimError(
                ErrorCode.VALIDATION, "locator requires 0 <= start <= end"
            )


@dataclass(frozen=True)
class EligibilityLease:
    """``resolve_access`` output (V4-08): the caller's authorization to
    *evaluate* objects for delivery. In-process only — binds caller,
    operation, scope set, purpose, verb, and the epoch vector the decision
    was made under. Expires quickly; re-resolution is cheap."""

    lease_id: str
    caller_id: str
    operation_id: str
    verb: Verb
    purpose: str
    scope_ids: tuple[str, ...]
    object_refs: tuple[tuple[str, int], ...] = ()  # empty = scope-wide eval
    epoch_vector: dict[str, int] = field(default_factory=dict)
    issued_us: int = 0
    expires_us: int = 0
    denied: bool = False          # indistinguishable denial carries no detail

    DEFAULT_MAX_AGE_US = 1_000_000  # V4-09.08: one second default

    def __post_init__(self) -> None:
        if not isinstance(self.verb, Verb):
            object.__setattr__(self, "verb", Verb(self.verb))
        if self.expires_us <= self.issued_us:
            raise VerbatimError(ErrorCode.VALIDATION, "lease expiry <= issue")

    def expired(self, now_us: int) -> bool:
        return now_us >= self.expires_us


@dataclass(frozen=True)
class VerifiedSlice:
    """``read_verified`` output (V4-08.03): exact bytes whose digest was
    re-authenticated against store, object, revision, locator, algorithm."""

    locator: EvidenceLocator
    data: bytes
    digest: str
    algorithm: str                # e.g. "hmac-sha256"
    provenance: dict[str, Any]    # captured envelope metadata (no payload leaks)
    verification: str = "verified"  # verified | legacy_unverified


@dataclass(frozen=True)
class DeliveryPermit:
    """``seal_delivery`` output (V4-09.06): the linearization point of
    disclosure. Committed only after rechecking contributing scopes and
    dependency versions; afterwards the disclosure is durable and honestly
    irreversible."""

    permit_id: str
    caller_id: str
    purpose: str
    epoch_vector: dict[str, int]
    payload_digest: str           # digest of the final serialized pack
    dependency_versions: dict[str, int]
    state: str                    # sealed | delivered | expired | revoked
    issued_us: int
    expires_us: int               # default issued + 1s (V4-09.08)
    receipt_id: Optional[str] = None

    def expired(self, now_us: int) -> bool:
        return now_us >= self.expires_us


@dataclass(frozen=True)
class DispatchPermit:
    """``open_dispatch`` output (V4-12.02): one-use egress authorization bound
    to recipient, purpose, exact payload digest, contributing scopes, consent
    versions, and spend reservation. Expires if not handed to its transport."""

    permit_id: str
    recipient: str                # allowlisted endpoint identity
    purpose: str
    payload_digest: str
    scope_ids: tuple[str, ...]
    consent_refs: tuple[str, ...]
    reservation_id: str           # budget reservation token
    max_spend: float
    issued_us: int
    expires_us: int
    state: str = "open"           # open | dispatched | expired | denied | reconciled


# ---------------------------------------------------------------------------
# Deletion closure (V4-38) — resumable, truthful
# ---------------------------------------------------------------------------


class ClosurePhase(str, enum.Enum):
    REQUESTED = "requested"        # validated, not yet suppressed
    SUPPRESSED = "suppressed"      # tombstoned; deliveries exclude target
    CLEANING = "cleaning"          # durable bounded closure in progress
    VERIFYING = "verifying"        # physical/reference checks running
    COMPLETED = "completed"        # enumerated local boundary verified
    FAILED = "failed_cleanup"      # suppression retained; obligations pending


@dataclass(frozen=True)
class ClosureRun:
    """One erasure/suppression operation's durable lifecycle (V4-38.03)."""

    run_id: str
    scope_id: str
    roots: tuple[tuple[str, str], ...]   # (kind, object_id) suppression roots
    erasure_epoch: int
    phase: ClosurePhase
    boundary: dict[str, Any]             # enumerated surfaces + counts
    verification: dict[str, Any] = field(default_factory=dict)
    created_us: int = 0
    updated_us: int = 0
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, ClosurePhase):
            object.__setattr__(self, "phase", ClosurePhase(self.phase))


@dataclass(frozen=True)
class ClosureFrontierItem:
    """One unit of closure work (V4-38.03): resumable across crashes.
    ``pending`` at a bound is a continuation, never terminal success."""

    run_id: str
    cursor: int                    # monotonically increasing per run
    object_kind: str
    object_id: str
    revision: Optional[int]
    action: str                    # erase_row | suppress_view | drop_index | ...
    state: str = "pending"         # pending | done | failed
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Readiness obligations (V4-14) — per-receipt capability DAG
# ---------------------------------------------------------------------------


class ReadinessState(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEFERRED = "deferred"          # capability not provisioned; honest backlog
    CANCELLED = "cancelled"


class CapabilityName(str, enum.Enum):
    """Named readiness capabilities (V4-14.02)."""

    ACCEPTED = "accepted"
    SCREENED = "screened"
    LEXICAL_READY = "lexical_ready"
    SEMANTIC_READY = "semantic_ready"
    DERIVED_READY = "derived_ready"
    FAILED = "failed"


@dataclass(frozen=True)
class ReadinessObligation:
    """One input's obligation to reach a named capability (V4-14.02).
    ``wait_ready`` follows the receipt's own DAG — unrelated later events
    never satisfy it (V4-14.03, C26/C88)."""

    obligation_id: str
    receipt_id: str               # capture receipt / operation id
    scope_id: str
    capability: CapabilityName
    depends_on: tuple[str, ...] = ()   # obligation ids
    state: ReadinessState = ReadinessState.PENDING
    error: Optional[str] = None
    created_us: int = 0
    updated_us: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.capability, CapabilityName):
            object.__setattr__(self, "capability", CapabilityName(self.capability))
        if not isinstance(self.state, ReadinessState):
            object.__setattr__(self, "state", ReadinessState(self.state))


# ---------------------------------------------------------------------------
# Schema operations (V4-41) — verifiable phased migrations
# ---------------------------------------------------------------------------


class SchemaOpState(str, enum.Enum):
    DECLARED = "declared"
    RUNNING = "running"
    RESUMABLE = "resumable"        # checkpoint persisted; safe to continue
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class SchemaOperation:
    """One migration phase (V4-41.04/06): checksummed, cursored, owned."""

    operation_id: str
    version_from: int
    version_to: int
    phase: str
    checksum: str                  # digest of immutable migration definition
    cursor: Optional[str] = None
    state: SchemaOpState = SchemaOpState.DECLARED
    owner_lease: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, SchemaOpState):
            object.__setattr__(self, "state", SchemaOpState(self.state))


# ---------------------------------------------------------------------------
# Store resolution (V4-07.10, F4-18) — one resolver, explicit conflicts
# ---------------------------------------------------------------------------


class StoreResolutionKind(str, enum.Enum):
    LEGACY_PROFILE = "legacy_profile"    # {profile_id}.db convention
    V3_DEFAULT = "v3_default"            # v3.db convention
    EXPLICIT = "explicit"                # caller-supplied path
    NONE = "none"                        # neither present


@dataclass(frozen=True)
class StoreResolution:
    """The single store-resolver output (V4-05.10). ``conflicts`` non-empty
    means an operator decision is required — never a silent merge or an
    empty replacement (C82)."""

    kind: StoreResolutionKind
    path: Optional[str]
    candidates: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    def needs_operator_decision(self) -> bool:
        return bool(self.conflicts) or (
            len([c for c in self.candidates if c]) > 1 and self.path is None
        )


# ---------------------------------------------------------------------------
# Manifest / capability reporting (V4-50)
# ---------------------------------------------------------------------------


class CapabilityRung(str, enum.Enum):
    """The capability-state ladder (V4-50.01): a component reports its
    highest *observed* rung — never config-implied health (F4-15)."""

    IMPLEMENTED = "implemented"
    INSTALLED = "installed"
    CONFIGURED = "configured"
    AUTHORIZED = "authorized"
    HEALTHY = "healthy"
    MEASURED = "measured"
    RECOMMENDED = "recommended"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class CapabilityReport:
    """A runtime provider's own capability observation (V4-50). The facade
    aggregates these; it never guesses from config or importable modules."""

    name: str
    rung: CapabilityRung
    degraded_reason: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)
    observed_us: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.rung, CapabilityRung):
            object.__setattr__(self, "rung", CapabilityRung(self.rung))


__all__ = [
    "CapabilityName",
    "CapabilityReport",
    "CapabilityRung",
    "ClosureFrontierItem",
    "ClosurePhase",
    "ClosureRun",
    "DeliveryPermit",
    "DispatchPermit",
    "Effect",
    "EffectKind",
    "EffectPlan",
    "EligibilityLease",
    "EvidenceLocator",
    "JobRequest",
    "LifecycleState",
    "OperationReceipt",
    "PurposeConstraint",
    "PurposeTag",
    "ReadinessObligation",
    "ReadinessState",
    "SchemaOpState",
    "SchemaOperation",
    "StoreResolution",
    "StoreResolutionKind",
    "VerifiedSlice",
]
