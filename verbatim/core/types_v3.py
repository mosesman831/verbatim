"""V3 core types: governance, evidence-plane, learning-plane, and wire contracts.

This module is the frozen contract for Verbatim v3 (SPEC_V3). It mirrors
``core/types.py`` conventions — frozen dataclasses, str-valued enums,
``VerbatimError``/``ErrorCode`` — and deliberately does NOT import from it so
that v3 contracts can evolve without rewriting v2 callers. Shared vocabulary
(errors, identifiers, JSON helpers) is re-exported for convenience.

Design rules baked in here (SPEC_V3 §04, §08–§14, §17, §21, §26–§32, §35):
- Perspective roles (asserter/subject/observer/audience) are explicit and
  never inferred; ``None`` means *not recorded*, never "everyone".
- Security metadata is multidimensional: origin, content form, attack risk,
  and review state are independent fields, not one ordered taint scale.
- Every derived object carries producer identity and derivation edges; a
  derived object is never a quotation.
- ``act`` is a gateway-enforced ticket contract, not a promise that a model's
  reasoning is controlled (§09.07).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

from .types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    new_id,
    require_id,
    safe_json_loads,
)

# Error taxonomy additions (SPEC_V3 §46) live in the shared ErrorCode enum
# in core/types.py; this module references them directly.


class PrincipalKind(str, enum.Enum):
    """Authenticated identity classes (§08). ``external_party`` principals
    are subjects of facts and are never callers."""

    HUMAN = "human"
    AGENT = "agent"
    SERVICE = "service"
    WORKSPACE = "workspace"
    ORGANIZATION = "organization"
    EXTERNAL_PARTY = "external_party"


class Verb(str, enum.Enum):
    """Permission verbs (§09). ``act`` is ticket issuance only; enforcement
    exists at registered execution gateways (§09.07)."""

    READ = "read"
    QUOTE = "quote"
    DERIVE = "derive"
    ACT = "act"
    SHARE = "share"
    HYDRATE = "hydrate"
    REVIEW = "review"
    INGEST = "ingest"
    ADMIN = "admin"


class TrustClass(str, enum.Enum):
    """Provenance class of accepted evidence (§14). Origin never changes
    after review; promotion rules read it together with screening state."""

    PRINCIPAL_DIRECT = "principal_direct"
    PRINCIPAL_REPORTED = "principal_reported"
    HOST_OBSERVED = "host_observed"
    EXTERNAL_CONTENT = "external_content"
    AGENT_GENERATED = "agent_generated"
    IMPORTED = "imported"
    UNKNOWN = "unknown"


class ContentForm(str, enum.Enum):
    """Content shape, independent of origin (§14.01). Instructional form is
    not an attack signal by itself."""

    DESCRIPTIVE = "descriptive"
    INSTRUCTIONAL = "instructional"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class AttackRisk(str, enum.Enum):
    """Screening outcome dimension (§14.01). ``no_findings`` means no listed
    attack signal was found — not certified safety."""

    UNASSESSED = "unassessed"
    NO_FINDINGS = "no_findings"
    SUSPICIOUS = "suspicious"
    BLOCKED = "blocked"


class SecurityReviewState(str, enum.Enum):
    """Review disposition of security findings (§14.01). Distinct from the
    v2 review-workflow states."""

    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    RELEASED = "released"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"


class FreshnessClass(str, enum.Enum):
    """Delivery-time freshness semantics (§24)."""

    STABLE = "stable"
    REVALIDATE_AFTER = "revalidate_after"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class SensitivityClass(str, enum.Enum):
    """Vault routing classes (§35). S4 is zero-retention: detected or
    declared S4 values are redacted before persistence."""

    S1 = "s1"  # low sensitivity: preferences, habits
    S2 = "s2"  # identifiable PII: vault + searchable placeholder
    S3 = "s3"  # highly sensitive: vault, excluded from consolidation
    S4 = "s4"  # critical secrets: zero retention


class DetectionBasis(str, enum.Enum):
    """How a sensitivity classification was established (§35.01)."""

    DETECTED = "detected"
    OWNER_DECLARED = "owner_declared"
    NO_DETECTION = "no_detection"


class EnvelopeKind(str, enum.Enum):
    """Universal source-envelope kinds (§12)."""

    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FILE_DIFF = "file_diff"
    FILE_SNAPSHOT_REF = "file_snapshot_ref"
    TEST_RESULT = "test_result"
    VERIFICATION = "verification"
    BROWSER_STATE = "browser_state"
    SCREENSHOT_REF = "screenshot_ref"
    PLAN = "plan"
    SUBGOAL = "subgoal"
    DECISION = "decision"
    ERROR = "error"
    RECOVERY = "recovery"
    HANDOFF = "handoff"
    DELEGATION = "delegation"
    DOCUMENT = "document"
    IMPORT = "import"
    CONNECTOR_ITEM = "connector_item"
    AGENT_NOTE = "agent_note"
    LESSON = "lesson"
    SYSTEM_EVENT = "system_event"


class MemoryKindV3(str, enum.Enum):
    """Learning-plane memory kinds (§17). ``fact`` is the claim object."""

    FACT = "fact"
    EPISODE = "episode"
    TRANSITION = "transition"
    PROCEDURE = "procedure"
    OBSERVATION = "observation"
    PROFILE = "profile"
    PLAN = "plan"
    WORKING = "working"
    SOCIAL = "social"
    ENVIRONMENT = "environment"


class Route(str, enum.Enum):
    """Controller routes (§26). ``investigate`` is never implicit."""

    NONE = "none"
    WORKING = "working"
    EXACT = "exact"
    CURRENT = "current"
    HISTORICAL = "historical"
    PROCEDURE = "procedure"
    FAILURE = "failure"
    GRAPH = "graph"
    VERIFY = "verify"
    INVESTIGATE = "investigate"


class BudgetTier(str, enum.Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class ActionIntent(str, enum.Enum):
    """What the host is about to do with recalled context (§27.01)."""

    NONE = "none"
    READ = "read"
    WRITE = "write"
    IRREVERSIBLE = "irreversible"


class OutcomeClass(str, enum.Enum):
    """Checker-attested task outcome (§12.03). An agent's statement of
    completion is never an outcome."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class ProcedureStateV3(str, enum.Enum):
    """Promotion ladder (§21–§22). v3.0 activation requires explicit review."""

    CANDIDATE = "candidate"
    REVIEWED = "reviewed"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class ProcedureRisk(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TransitionEdge(str, enum.Enum):
    """Core transition edges (§20.04). ``observed_after`` records order, not
    causation; causal hypotheses are a separately labeled method."""

    PRECEDES = "precedes"
    OBSERVED_AFTER = "observed_after"
    VERIFIED_BY = "verified_by"
    CAUSAL_HYPOTHESIS = "causal_hypothesis"


class OperationClass(str, enum.Enum):
    """Bounded coding-domain operation taxonomy (§22 coding_rules_v1).
    ``opaque`` marks evidence the compiler cannot abstract."""

    INSPECT_FILE = "inspect_file"
    SEARCH_REPO = "search_repo"
    APPLY_PATCH = "apply_patch"
    RUN_CHECK = "run_check"
    OPAQUE = "opaque"


class InfluenceKind(str, enum.Enum):
    """Influence record kinds (§32.02). Missing feedback is ``unknown``,
    never proof of no influence."""

    DELIVERED = "delivered"
    HOST_REPORTED = "host_reported"
    GATEWAY_OBSERVED = "gateway_observed"
    ESTIMATED = "estimated"


class InfluenceFeedback(str, enum.Enum):
    """Host-reported per-handle feedback (§32.01)."""

    USED = "used"
    CITED = "cited"
    USED_FOR_ACTION = "used_for_action"
    IGNORED = "ignored"
    HARMFUL = "harmful"


class HydrationTarget(str, enum.Enum):
    """Allowed exact-value delivery targets (§35.04)."""

    OPERATOR = "operator"
    LOCAL_MODEL = "local_model"
    TASK_TOOL_BROKER = "task_tool_broker"
    CLOUD_MODEL = "cloud_model"


class EgressClass(str, enum.Enum):
    """Enrichment egress classes (§11.02). No class permits vault values."""

    NONE = "none"
    SANITIZED = "sanitized"
    EVIDENCE_EXCERPT = "evidence_excerpt"
    FULL = "full"


class JobLane(str, enum.Enum):
    """Queue lanes (§40). Privacy/control work is always available."""

    ORDINARY = "ordinary"
    BACKGROUND = "background"
    PRIVACY_CONTROL = "privacy_control"
    MAINTENANCE = "maintenance"


class PlanStatusV3(str, enum.Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    OVERDUE = "overdue"
    UNKNOWN = "unknown"


class ApplicabilityVerdict(str, enum.Enum):
    """Three-valued applicability (§22.15): unknown is never a wildcard."""

    APPLIES = "applies"
    DOES_NOT_APPLY = "does_not_apply"
    UNKNOWN = "unknown"


class CompilationStatus(str, enum.Enum):
    """Episode compilation outcome (§22). Unsupported episodes keep their
    exact evidence without claiming a usable procedure."""

    CANDIDATE = "candidate"
    UNSUPPORTED = "unsupported"
    INCOMPLETE = "incomplete"


class ReplayStage(str, enum.Enum):
    HARVEST = "harvest"
    ADMISSION = "admission"
    RELATION = "relation"
    SCREENING = "screening"
    ROUTING = "routing"
    FUSION = "fusion"
    ABSTENTION = "abstention"
    PACKING = "packing"
    CONSOLIDATION = "consolidation"


class DeploymentProfile(str, enum.Enum):
    """Profiles (§05). v3.0 ships embedded + local_semantic."""

    EMBEDDED = "embedded"
    LOCAL_SEMANTIC = "local_semantic"
    TEAM_SERVICE = "team_service"
    SCALED_SERVICE = "scaled_service"
    SPLIT_PRIVACY = "split_privacy"
    PORTABLE_WORKSPACE = "portable_workspace"


class CapabilityStatus(str, enum.Enum):
    """The seven reported capability states (§05.01)."""

    IMPLEMENTED = "implemented"
    INSTALLED = "installed"
    CONFIGURED = "configured"
    AUTHORIZED = "authorized"
    HEALTHY = "healthy"
    MEASURED = "measured"
    RECOMMENDED = "recommended"


class QueryClass(str, enum.Enum):
    """Query-plan classes (§27.06)."""

    EXACT_LOOKUP = "exact_lookup"
    CURRENT_STATE = "current_state"
    PAST_STATE = "past_state"
    TIMELINE = "timeline"
    RELATIONSHIP = "relationship"
    CAUSE = "cause"
    PROCEDURE = "procedure"
    FAILURE = "failure"
    ARCHIVE = "archive"
    EXPLORATORY = "exploratory"


class GroupSupport(str, enum.Enum):
    """Abstention verdict for a complete evidence group (§29.05)."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class PackKind(str, enum.Enum):
    """Context-assembly packs (§30)."""

    EVIDENCE_BUNDLE = "evidence_bundle"
    PROCEDURE_PACK = "procedure_pack"
    WORKING_STATE = "working_state"
    VERIFY_PACK = "verify_pack"
    CONFLICT_PACK = "conflict_pack"
    HANDOFF_CAPSULE = "handoff_capsule"


# ---------------------------------------------------------------------------
# Governance contracts (§08–§11)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Principal:
    """An authenticated identity record (§08). ``external_party`` rows are
    subjects, never callers."""

    principal_id: str
    kind: PrincipalKind
    created_us: int
    display_name: Optional[str] = None
    host_binding: Optional[str] = None
    retired: bool = False

    def __post_init__(self) -> None:
        require_id(self.principal_id, "principal_id")
        if not isinstance(self.kind, PrincipalKind):
            object.__setattr__(self, "kind", PrincipalKind(self.kind))


@dataclass(frozen=True)
class Perspective:
    """The four explicit roles on evidence and derived objects (§04.09).

    ``asserter``: who made the statement. ``subjects``: whom/what it is
    about. ``observer``: who recorded it. ``audience``: whom it is visible
    to. ``None`` means *not recorded* — never widened silently.
    """

    asserter: Optional[str] = None
    subjects: tuple[str, ...] = ()
    observer: Optional[str] = None
    audience: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("asserter", "observer"):
            v = getattr(self, name)
            if v is not None:
                require_id(v, name)
        object.__setattr__(
            self, "subjects", tuple(require_id(s, "subject") for s in self.subjects)
        )
        object.__setattr__(
            self, "audience", tuple(require_id(a, "audience") for a in self.audience)
        )


@dataclass(frozen=True)
class Grant:
    """A scope-bound permission record (§09.03). Attenuation-only: a
    delegation may narrow verbs/purposes/expiry, never widen (§09.03)."""

    grant_id: str
    scope_id: str
    principal_id: str
    verbs: frozenset[Verb]
    purposes: frozenset[str]
    issuer_id: str
    issued_us: int
    expires_us: Optional[int] = None
    delegation_depth: int = 0
    caveats: tuple[str, ...] = ()
    revoked_us: Optional[int] = None
    epoch: int = 0

    def __post_init__(self) -> None:
        for name in ("grant_id", "scope_id", "principal_id", "issuer_id"):
            require_id(getattr(self, name), name)
        object.__setattr__(
            self,
            "verbs",
            frozenset(v if isinstance(v, Verb) else Verb(v) for v in self.verbs),
        )
        object.__setattr__(self, "purposes", frozenset(self.purposes))
        if not self.verbs:
            raise VerbatimError(ErrorCode.VALIDATION, "grant requires >= 1 verb")
        if self.delegation_depth < 0:
            raise VerbatimError(ErrorCode.VALIDATION, "delegation_depth < 0")
        if self.expires_us is not None and self.expires_us <= self.issued_us:
            raise VerbatimError(ErrorCode.VALIDATION, "grant expiry precedes issue")

    def attenuates(self, parent: "Grant") -> bool:
        """True when this grant is a strict narrowing of ``parent``."""
        return (
            self.verbs <= parent.verbs
            and self.purposes <= parent.purposes
            and (
                self.expires_us is None
                or parent.expires_us is None
                or self.expires_us <= parent.expires_us
            )
        )


@dataclass(frozen=True)
class Delegation:
    """A recorded attenuation edge between two grants (§08.02)."""

    delegation_id: str
    parent_grant_id: str
    child_grant_id: str
    delegator_id: str
    delegate_id: str
    created_us: int
    expires_us: Optional[int] = None
    revoked_us: Optional[int] = None


@dataclass(frozen=True)
class CaptureAuthorization:
    """Host/operator-issued retention consent for a source class (§11.11).

    Tool permission is not retention consent: this record is what makes
    durable capture of a gated envelope kind lawful. ``None`` fields mean
    not constrained, not universal.
    """

    authorization_id: str
    issuer_id: str
    principal_id: str
    allowed_kinds: frozenset[EnvelopeKind]
    scope_ids: frozenset[str]
    retention_policy: str
    policy_revision: str
    issued_us: int
    expires_us: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_kinds",
            frozenset(
                k if isinstance(k, EnvelopeKind) else EnvelopeKind(k)
                for k in self.allowed_kinds
            ),
        )
        object.__setattr__(self, "scope_ids", frozenset(self.scope_ids))
        if not self.retention_policy:
            raise VerbatimError(ErrorCode.VALIDATION, "retention_policy required")


@dataclass(frozen=True)
class ConsentRecord:
    """Disclosure consent (§11.01) — processor, purpose, classes, expiry.
    Memory access grants are not consent to disclose to a processor."""

    consent_id: str
    scope_id: str
    processor: str
    purpose: str
    data_classes: frozenset[str]
    sanitization: EgressClass
    retention_promise: Optional[str]
    granted_us: int
    expires_us: Optional[int] = None
    revoked_us: Optional[int] = None
    budget_microusd: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.sanitization, EgressClass):
            object.__setattr__(
                self, "sanitization", EgressClass(self.sanitization)
            )


# ---------------------------------------------------------------------------
# Security metadata (§14) — multidimensional, never one ordered scale
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecurityLabel:
    """Independent security dimensions on evidence and derived objects.

    ``findings`` carries the specific attempted boundary violations named by
    screening (e.g. ``memory_directed_instruction``, ``role_claim``,
    ``tool_invocation``, ``compositional_payload``) with span references.
    """

    source_trust: TrustClass = TrustClass.UNKNOWN
    content_form: ContentForm = ContentForm.UNKNOWN
    attack_risk: AttackRisk = AttackRisk.UNASSESSED
    review_state: SecurityReviewState = SecurityReviewState.NOT_REQUIRED
    findings: tuple[dict[str, Any], ...] = ()
    method: str = "rules"
    rules_revision: str = ""

    def __post_init__(self) -> None:
        for name, cls in (
            ("source_trust", TrustClass),
            ("content_form", ContentForm),
            ("attack_risk", AttackRisk),
            ("review_state", SecurityReviewState),
        ):
            v = getattr(self, name)
            if not isinstance(v, cls):
                object.__setattr__(self, name, cls(v))

    def merge(self, other: "SecurityLabel") -> "SecurityLabel":
        """Derived-metadata merge (§14.06): union origins' findings, keep
        the most restrictive unresolved disposition. Origin is never
        upgraded."""
        _RISK_ORDER = {
            AttackRisk.UNASSESSED: 0,
            AttackRisk.NO_FINDINGS: 1,
            AttackRisk.SUSPICIOUS: 2,
            AttackRisk.BLOCKED: 3,
        }
        _REVIEW_ORDER = {
            SecurityReviewState.NOT_REQUIRED: 0,
            SecurityReviewState.RELEASED: 0,
            SecurityReviewState.PENDING: 1,
            SecurityReviewState.QUARANTINED: 2,
            SecurityReviewState.REJECTED: 3,
        }
        return SecurityLabel(
            source_trust=(
                self.source_trust
                if self.source_trust != TrustClass.UNKNOWN
                else other.source_trust
            ),
            content_form=(
                self.content_form
                if self.content_form != ContentForm.UNKNOWN
                else other.content_form
            ),
            attack_risk=max(
                (self.attack_risk, other.attack_risk),
                key=lambda r: _RISK_ORDER[r],
            ),
            review_state=max(
                (self.review_state, other.review_state),
                key=lambda s: _REVIEW_ORDER[s],
            ),
            findings=tuple(self.findings) + tuple(other.findings),
            method=self.method,
            rules_revision=self.rules_revision,
        )


# ---------------------------------------------------------------------------
# Evidence plane (§12–§13): envelopes, trajectories, anchors, outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceEnvelopeV3:
    """A typed capture event before persistence (§12.01).

    ``content`` is exact bytes for in-line envelopes; large artifacts arrive
    as content-addressed ``artifact_ref`` instead. ``capture_proof`` binds a
    CaptureAuthorization id for gated kinds (§12.10)."""

    kind: EnvelopeKind
    scope_id: str
    actor_principal: str
    perspective: Perspective
    event_us: int
    receipt_us: int
    content: Optional[bytes] = None
    artifact_ref: Optional[str] = None
    media_type: str = "text/plain"
    trust_class: TrustClass = TrustClass.UNKNOWN
    capture_proof: Optional[str] = None
    redaction_status: str = "none"  # none|applied|failed_closed
    adapter_version: str = ""
    host_id: str = ""
    session_id: str = ""
    task_id: str = ""
    step_id: str = ""
    external_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EnvelopeKind):
            object.__setattr__(self, "kind", EnvelopeKind(self.kind))
        if not isinstance(self.trust_class, TrustClass):
            object.__setattr__(self, "trust_class", TrustClass(self.trust_class))
        require_id(self.scope_id, "scope_id")
        require_id(self.actor_principal, "actor_principal")
        if (self.content is None) == (self.artifact_ref is None):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "envelope needs exactly one of content or artifact_ref",
            )
        if self.content is not None and len(self.content) == 0:
            raise VerbatimError(ErrorCode.VALIDATION, "empty envelope content")


@dataclass(frozen=True)
class EnvironmentFingerprint:
    """Observed environment identity at a step (§20–§22). Missing fields are
    ``None``: absent information is unknown, never a wildcard match."""

    repo_id: Optional[str] = None
    repo_revision: Optional[str] = None
    runtime_versions: tuple[tuple[str, str], ...] = ()
    tool_schema_versions: tuple[tuple[str, str], ...] = ()
    platform: Optional[str] = None

    def digest(self) -> str:
        import hashlib

        return hashlib.sha256(
            json_dumps(
                {
                    "repo_id": self.repo_id,
                    "repo_revision": self.repo_revision,
                    "runtime_versions": list(self.runtime_versions),
                    "tool_schema_versions": list(self.tool_schema_versions),
                    "platform": self.platform,
                }
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class StateAnchor:
    """An exact reference to observed environment state (§20.03): file
    revision, artifact digest, tool version. Anchors reconstruct order;
    they never establish causation."""

    anchor_id: str
    kind: str  # file_revision|artifact_digest|tool_version|dom_digest|screen_digest
    ref: str
    digest: str


@dataclass(frozen=True)
class CheckerReceipt:
    """What a checker actually established (§22.16). A green check proves
    only that its declared checks passed."""

    checker_id: str
    checker_version: str
    repo_revision: Optional[str]
    tree_digest: Optional[str]
    invocation_id: str
    selected_tests: tuple[str, ...]
    completed: bool
    exit_code: Optional[int]
    result_json: dict[str, Any] = field(default_factory=dict)
    host_attested: bool = False


@dataclass(frozen=True)
class OutcomeEnvelope:
    """A task outcome from an identified checker (§12.03)."""

    outcome: OutcomeClass
    checker: Optional[CheckerReceipt]
    evidence_refs: tuple[str, ...] = ()
    recorded_us: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, OutcomeClass):
            object.__setattr__(self, "outcome", OutcomeClass(self.outcome))


@dataclass(frozen=True)
class TrajectoryRecord:
    """A task-bounded envelope sequence (§12.02): the evidence-plane input
    to episode/transition derivation. ``boundary_rule`` declares how the
    task boundary was detected — never inferred silently."""

    trajectory_id: str
    scope_id: str
    host_id: str = ""
    session_id: str = ""
    task_id: str = ""
    boundary_rule: str = "task_id"
    environment_digest: Optional[str] = None
    created_event: int = 0
    completed_event: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryStep:
    """One ordered step inside a task trajectory (§12.02)."""

    step_id: str
    trajectory_id: str
    ord: int
    action_envelope_id: Optional[str] = None
    observation_envelope_ids: tuple[str, ...] = ()
    state_delta_refs: tuple[str, ...] = ()
    environment: Optional[EnvironmentFingerprint] = None


@dataclass(frozen=True)
class Transition:
    """A deterministic (pre_state, action, post_state) record (§20.02)."""

    transition_id: str
    episode_id: str
    scope_id: str
    ord: int
    pre_anchor_ids: tuple[str, ...]
    action_step_id: str
    post_anchor_ids: tuple[str, ...]
    checker_ref: Optional[str]
    environment_digest: Optional[str]
    edge: TransitionEdge = TransitionEdge.OBSERVED_AFTER

    def __post_init__(self) -> None:
        if not isinstance(self.edge, TransitionEdge):
            object.__setattr__(self, "edge", TransitionEdge(self.edge))


# ---------------------------------------------------------------------------
# Learning plane: procedures (§21–§22)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperationTemplate:
    """An abstracted procedure operation (§21.02): a class plus parameter
    template with binding slots, never a stored executable command."""

    op_class: OperationClass
    tool: Optional[str]
    param_template: dict[str, Any] = field(default_factory=dict)
    ord: int = 0
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.op_class, OperationClass):
            object.__setattr__(self, "op_class", OperationClass(self.op_class))


@dataclass(frozen=True)
class Binding:
    """A value the host must re-acquire at reuse time (§21)."""

    name: str
    kind: str  # repo_path|test_target|revision|other
    observed_value: Optional[str] = None
    required: bool = True


@dataclass(frozen=True)
class FailureMode:
    """A recorded failure with its evidence (§21.05)."""

    signature: str
    description: str
    evidence_refs: tuple[str, ...]
    recovery: Optional[str] = None
    occurrences: int = 1


@dataclass(frozen=True)
class TypedCondition:
    """An evaluable applicability condition (§22.15): true/false/unknown."""

    key: str
    op: str  # eq|neq|in|present
    value: Optional[str] = None
    provenance: str = "observed"  # observed|hypothesis|reviewer
    validated: bool = False


@dataclass(frozen=True)
class ProcedureRecord:
    """The structured procedure object (§21). Advice, never executable
    code, never authority (§21.01)."""

    procedure_id: str
    scope_id: str
    revision: int
    intent_signature: dict[str, Any]
    operations: tuple[OperationTemplate, ...]
    bindings: tuple[Binding, ...]
    environment: Optional[EnvironmentFingerprint]
    hazards: tuple[str, ...]
    verification: tuple[str, ...]
    expected_outcome: Optional[str]
    failure_modes: tuple[FailureMode, ...]
    applicability: tuple[TypedCondition, ...]
    preconditions: tuple[TypedCondition, ...]
    provenance: dict[str, Any]
    state: ProcedureStateV3
    risk_class: ProcedureRisk
    reuse_stats: dict[str, int] = field(default_factory=dict)
    evidence_family_id: Optional[str] = None
    security: Optional[SecurityLabel] = None
    freshness: FreshnessClass = FreshnessClass.UNKNOWN
    compiler_manifest: Optional[str] = None

    def __post_init__(self) -> None:
        for name, cls in (
            ("state", ProcedureStateV3),
            ("risk_class", ProcedureRisk),
            ("freshness", FreshnessClass),
        ):
            v = getattr(self, name)
            if not isinstance(v, cls):
                object.__setattr__(self, name, cls(v))


# ---------------------------------------------------------------------------
# Query/context contracts (§27, §30)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskContext:
    """What the host is doing right now (§27.01). All optional; absent
    information reduces confidence, never widens authorization."""

    task_id: Optional[str] = None
    goal_class: Optional[str] = None
    action_intent: ActionIntent = ActionIntent.NONE
    tools_pending: tuple[str, ...] = ()
    environment: Optional[EnvironmentFingerprint] = None
    session_phase: Optional[str] = None  # start|continuation|stuck|end
    stuck: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.action_intent, ActionIntent):
            object.__setattr__(
                self, "action_intent", ActionIntent(self.action_intent)
            )


@dataclass(frozen=True)
class RecallRequestV3:
    """The v3 recall contract (§27.01). Hard bounds fail before any
    dereference with a fixed transport error (§27.03)."""

    query: str
    scope_id: str
    caller_id: str
    purpose: str
    modes: tuple[str, ...] = ()
    memory_kinds: tuple[MemoryKindV3, ...] = ()
    perspective: Optional[Perspective] = None
    task: Optional[TaskContext] = None
    valid_at_us: Optional[int] = None
    valid_until_us: Optional[int] = None
    known_at_seq: Optional[int] = None
    freshness_required: bool = False
    entity_ids: tuple[str, ...] = ()
    max_items: int = 8
    max_bytes: int = 6000
    target_tokens: int = 1536
    deadline_ms: int = 200
    budget_tier: BudgetTier = BudgetTier.MID
    min_ready_seq: Optional[int] = None
    wire_version: int = 3
    # V45-03/I1 (opt-in): ``"counterevidence_first"`` emits the evidence
    # manifest (supporting/contrary refs, omissions+reasons, insufficiency
    # labels) via ``capabilities["manifest"]`` and admits contrary groups
    # before depth under the shared budget. ``None``/``"none"``/
    # ``"standard"`` = current behavior. V45-05/I3: ``pack_mode``
    # ``"sufficiency"`` admits the minimal identifier/condition-covering
    # group set before depth. The value registry lives in
    # ``verbatim.retrieval.manifest``; validation here mirrors it so a bad
    # token fails at construction, and the pipeline re-validates for
    # dynamically-attached attributes (fail closed both ways).
    manifest: Optional[str] = None
    pack_mode: str = "standard"

    def __post_init__(self) -> None:
        if not (1 <= len(self.query.encode("utf-8", errors="replace")) <= 32768):
            raise VerbatimError(ErrorCode.VALIDATION, "query must be 1..32768 bytes")
        if len(self.query) > 8192:
            raise VerbatimError(ErrorCode.VALIDATION, "query exceeds 8192 chars")
        if len(self.entity_ids) > 8:
            raise VerbatimError(ErrorCode.VALIDATION, "entity_ids bounded at 8")
        if not (1 <= self.max_items <= 32):
            raise VerbatimError(ErrorCode.VALIDATION, "max_items must be 1..32")
        if not (512 <= self.max_bytes <= 24000):
            raise VerbatimError(ErrorCode.VALIDATION, "max_bytes must be 512..24000")
        if not (128 <= self.target_tokens <= 6144):
            raise VerbatimError(ErrorCode.VALIDATION, "target_tokens 128..6144")
        if not (20 <= self.deadline_ms <= 2000):
            raise VerbatimError(ErrorCode.VALIDATION, "deadline_ms must be 20..2000")
        if not isinstance(self.budget_tier, BudgetTier):
            object.__setattr__(self, "budget_tier", BudgetTier(self.budget_tier))
        require_id(self.scope_id, "scope_id")
        require_id(self.caller_id, "caller_id")
        if not self.purpose:
            raise VerbatimError(ErrorCode.VALIDATION, "purpose is mandatory")
        if self.manifest not in (
            None, "", "none", "standard", "counterevidence_first"
        ):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"unknown manifest mode {self.manifest!r}",
            )
        if self.pack_mode not in ("standard", "sufficiency"):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"unknown pack_mode {self.pack_mode!r}",
            )
        object.__setattr__(
            self,
            "memory_kinds",
            tuple(
                k if isinstance(k, MemoryKindV3) else MemoryKindV3(k)
                for k in self.memory_kinds
            ),
        )


@dataclass(frozen=True)
class ContextRequestV3:
    """Task-aware assembly request (§27.02): typed packs under one budget."""

    query: str
    scope_id: str
    caller_id: str
    purpose: str
    task: Optional[TaskContext] = None
    packs: tuple[PackKind, ...] = ()
    max_bytes: int = 6000
    target_tokens: int = 1536
    deadline_ms: int = 200
    budget_tier: BudgetTier = BudgetTier.MID
    wire_version: int = 3
    # V45-03/V45-05 opt-ins — same contract as RecallRequestV3.
    manifest: Optional[str] = None
    pack_mode: str = "standard"

    def __post_init__(self) -> None:
        if not isinstance(self.budget_tier, BudgetTier):
            object.__setattr__(self, "budget_tier", BudgetTier(self.budget_tier))
        object.__setattr__(
            self,
            "packs",
            tuple(p if isinstance(p, PackKind) else PackKind(p) for p in self.packs),
        )
        require_id(self.scope_id, "scope_id")
        require_id(self.caller_id, "caller_id")
        if not self.purpose:
            raise VerbatimError(ErrorCode.VALIDATION, "purpose is mandatory")
        if self.manifest not in (
            None, "", "none", "standard", "counterevidence_first"
        ):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"unknown manifest mode {self.manifest!r}",
            )
        if self.pack_mode not in ("standard", "sufficiency"):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"unknown pack_mode {self.pack_mode!r}",
            )


@dataclass(frozen=True)
class InfluenceHandle:
    """Per-delivered-item handle for feedback and blast radius (§32.01)."""

    handle_id: str
    receipt_id: str
    caller_id: str
    epoch: int
    pack: PackKind
    object_kind: str
    object_id: str
    revision: int


@dataclass(frozen=True)
class PackItem:
    """One serialized item inside a pack: influence handle, lifecycle,
    freshness, security metadata, perspective, derived/exact marker (§30.04)."""

    handle: InfluenceHandle
    text: str
    lifecycle: str
    freshness: FreshnessClass
    security: SecurityLabel
    perspective: Perspective
    derived: bool
    proof_count: int = 0
    verify_recommended: bool = False


@dataclass(frozen=True)
class ContextPack:
    """A typed pack inside one assembled context budget (§30)."""

    kind: PackKind
    items: tuple[PackItem, ...]
    tokens: int
    serialized_bytes: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecallResultV3:
    """The v3 recall response (§27/§30): typed packs plus the routing
    decision that produced them. ``decision_id`` links to the recorded
    routing_decisions row for replay/influence (§26.06, §32)."""

    packs: tuple[ContextPack, ...]
    omitted: int
    warnings: tuple[str, ...]
    capabilities: dict[str, Any]
    decision_id: Optional[str] = None
    projection_generation: int = 0
    abstained: bool = False


# ---------------------------------------------------------------------------
# Action-use tickets and value handles (§09.07, §35)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionTicket:
    """A short-lived, one-use memory-use ticket (§09.07). Validated by a
    registered gateway immediately before execution; advisory-only without
    one."""

    ticket_id: str
    recipient_id: str
    action_digest: bytes
    object_refs: tuple[tuple[str, int], ...]  # (object_id, revision)
    purpose: str
    epoch: int
    nonce: str
    issued_us: int
    expires_us: int
    gateway_id: str
    consumed_us: Optional[int] = None

    DEFAULT_LIFETIME_S = 30
    MAX_LIFETIME_S = 60

    def __post_init__(self) -> None:
        if self.expires_us <= self.issued_us:
            raise VerbatimError(ErrorCode.VALIDATION, "ticket expiry <= issue")
        if (self.expires_us - self.issued_us) > self.MAX_LIFETIME_S * 1_000_000:
            raise VerbatimError(
                ErrorCode.VALIDATION, "ticket lifetime exceeds 60s maximum"
            )


@dataclass(frozen=True)
class ValueHandle:
    """An opaque, one-use handle a broker redeems for an exact value at the
    approved tool/action boundary (§35.04). Model-visible responses carry
    handles, not plaintext."""

    handle_id: str
    vault_entry_id: str
    recipient_id: str
    action_digest: bytes
    consent_id: str
    expires_us: int
    consumed_us: Optional[int] = None


@dataclass(frozen=True)
class VaultEntryMeta:
    """Non-secret vault record (§35.02): ciphertext + wrapped key + AEAD
    metadata. No plaintext, no unkeyed digest of the value."""

    entry_id: str
    scope_id: str
    revision: int
    sensitivity: SensitivityClass
    placeholder: str
    algorithm: str  # e.g. "AES-256-GCM"
    key_version: int
    wrap_key_version: int
    nonce: bytes
    ciphertext: bytes
    wrapped_key: bytes
    aad_digest: bytes
    detection: DetectionBasis = DetectionBasis.DETECTED

    def __post_init__(self) -> None:
        for name, cls in (
            ("sensitivity", SensitivityClass),
            ("detection", DetectionBasis),
        ):
            v = getattr(self, name)
            if not isinstance(v, cls):
                object.__setattr__(self, name, cls(v))


# ---------------------------------------------------------------------------
# Replay and routing records (§26.06, §43)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingDecision:
    """A recorded controller decision (§26.06): replayable, never
    authorization-widening."""

    decision_id: str
    state_key: str
    routes: tuple[Route, ...]
    lane_set: tuple[str, ...]
    budgets: dict[str, int]
    policy_revision: str
    result_sizes: dict[str, int] = field(default_factory=dict)
    outcome_credit: Optional[str] = None
    created_us: int = 0


@dataclass(frozen=True)
class ReplayManifest:
    """What a replay run pins (§43.01). Digests alone cannot reconstruct
    inputs; replay needs consent-retained inputs and recorded lane results."""

    manifest_id: str
    source_range: tuple[int, int]
    policy_revision: str
    artifact_ids: tuple[str, ...]
    simulated_clock_us: Optional[int]
    scope_ids: tuple[str, ...]
    taint_state: str
    suppression_seq: int
    controller_revision: str
    stages: tuple[ReplayStage, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stages",
            tuple(
                s if isinstance(s, ReplayStage) else ReplayStage(s)
                for s in self.stages
            ),
        )


# ---------------------------------------------------------------------------
# Episode / observation / misc learning-plane records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeRecord:
    """A task/session grouping produced by a declared boundary rule
    (§20.01). Failed and abandoned episodes are retained as negative
    experience."""

    episode_id: str
    scope_id: str
    revision: int
    boundary_rule: str
    task_id: Optional[str]
    kind: str = "task"  # task|session_segment|operator_selection
    label: Optional[str] = None
    outcome: OutcomeClass = OutcomeClass.UNKNOWN
    environment: Optional[EnvironmentFingerprint] = None
    recorded_from: int = 0
    recorded_until: Optional[int] = None


@dataclass(frozen=True)
class ObservationRecord:
    """A consolidated derived belief (§23): supporting + contradicting
    evidence links, proof count over distinct families, freshness."""

    observation_id: str
    scope_id: str
    revision: int
    text: str
    proof_count: int
    supporting: tuple[str, ...]
    contradicting: tuple[str, ...]
    perspective: Perspective
    freshness: FreshnessClass
    stale_since_seq: Optional[int] = None
    producer: str = "slot_aggregate_v1"


@dataclass(frozen=True)
class WorkingSetItem:
    """A bounded session/task view entry (§24.03). Expiry suppresses the
    view only — never the underlying evidence."""

    item_id: str
    scope_id: str
    session_id: str
    kind: str  # goal|unresolved_ref|decision|recent_evidence
    object_ref: Optional[str]
    text: Optional[str]
    expires_us: int


@dataclass(frozen=True)
class Propagation:
    """A recorded cross-boundary disclosure (§10.04): share, handoff,
    publication, hydration. Blast radius is computed from this ledger."""

    propagation_id: str
    scope_id: str
    object_kind: str
    object_id: str
    revision: int
    recipient_id: str
    verbs: frozenset[Verb]
    purpose: str
    epoch: int
    created_us: int
    capsule_id: Optional[str] = None
    revoked_seq: Optional[int] = None
    acknowledged: bool = False


@dataclass(frozen=True)
class LearningSnapshot:
    """A rollback target over derived objects (§34.02): sequence + digest,
    never a claim of guaranteed truth."""

    snapshot_id: str
    seq: int
    digest: bytes
    created_us: int
    validation: str = "unvalidated"
