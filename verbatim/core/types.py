"""Core frozen types, identifiers, and the stable error taxonomy.

Public request/result types are immutable snapshots; mutable database rows
never escape repository boundaries (SPEC §8). Unknown times, identities,
scores, and conditions are represented explicitly — never as zero or empty
truthy defaults.
"""

from __future__ import annotations

import enum
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


class VerbatimError(Exception):
    """Typed error with a stable machine code (SPEC §43)."""

    def __init__(
        self,
        code: "ErrorCode",
        message: str,
        *,
        retryable: bool = False,
        detail_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.detail_id = detail_id

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.detail_id is not None:
            out["detail_id"] = self.detail_id
        return out


class ErrorCode(str, enum.Enum):
    CONFIG_INVALID = "CONFIG_INVALID"
    NOT_FOUND_OR_FORBIDDEN = "NOT_FOUND_OR_FORBIDDEN"
    CAPTURE_DISABLED = "CAPTURE_DISABLED"
    RETENTION_DENIED = "RETENTION_DENIED"
    EGRESS_DISABLED = "EGRESS_DISABLED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    REMOTE_AUTH = "REMOTE_AUTH"
    REMOTE_BILLING = "REMOTE_BILLING"
    REMOTE_BUSY = "REMOTE_BUSY"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    DECISION_INVALID = "DECISION_INVALID"
    MODEL_DRIFT = "MODEL_DRIFT"
    ENCODER_UNAVAILABLE = "ENCODER_UNAVAILABLE"
    VECTOR_INVALID = "VECTOR_INVALID"
    STORE_BUSY = "STORE_BUSY"
    STORE_WRITE_FAILED = "STORE_WRITE_FAILED"
    STORE_CORRUPT = "STORE_CORRUPT"
    SCHEMA_UNSUPPORTED = "SCHEMA_UNSUPPORTED"
    STALE_PROPOSAL = "STALE_PROPOSAL"
    LEASE_LOST = "LEASE_LOST"
    PROCESSING_PENDING = "PROCESSING_PENDING"
    EVIDENCE_UNAVAILABLE = "EVIDENCE_UNAVAILABLE"
    CONTEXT_INCOMPLETE = "CONTEXT_INCOMPLETE"
    ENVIRONMENT_MISMATCH = "ENVIRONMENT_MISMATCH"
    APPLICABILITY_UNKNOWN = "APPLICABILITY_UNKNOWN"
    EVIDENCE_TOO_LARGE = "EVIDENCE_TOO_LARGE"
    BUDGET_TOO_SMALL = "BUDGET_TOO_SMALL"
    BACKPRESSURE = "BACKPRESSURE"
    CHECKPOINT_FAILED = "CHECKPOINT_FAILED"
    INVALID_TRANSITION = "INVALID_TRANSITION"
    VALIDATION = "VALIDATION"
    # --- v3 additions (SPEC_V3 §46) ---
    NOT_FOUND_OR_UNAUTHORIZED = "NOT_FOUND_OR_UNAUTHORIZED"
    CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    CONSENT_REQUIRED = "CONSENT_REQUIRED"
    QUARANTINED = "QUARANTINED"
    COMPILATION_UNSUPPORTED = "COMPILATION_UNSUPPORTED"
    INVESTIGATION_UNSUPPORTED = "INVESTIGATION_UNSUPPORTED"
    STALE_EPOCH = "STALE_EPOCH"
    INTEGRITY = "INTEGRITY"
    LOCKED = "LOCKED"
    RETRYABLE_OPERATION = "RETRYABLE_OPERATION"
    # --- v4 additions (SPEC_V4 §52) ---
    EGRESS_DENIED = "EGRESS_DENIED"
    OPERATION_CONFLICT = "OPERATION_CONFLICT"
    CLOSURE_PENDING = "CLOSURE_PENDING"
    ERASURE_UNPROVEN = "ERASURE_UNPROVEN"
    STALE_DEPENDENCY = "STALE_DEPENDENCY"
    CANCELLED = "CANCELLED"
    PERMIT_EXPIRED = "PERMIT_EXPIRED"
    STORE_CONFLICT = "STORE_CONFLICT"


class Mode(str, enum.Enum):
    OFFLINE_RULES = "offline_rules"
    OFFLINE_SEMANTIC = "offline_semantic"
    LOCAL_SERVICE = "local_service"
    REMOTE_ASSISTED = "remote_assisted"
    JEV_ASSISTED = "jev_assisted"  # migration alias for restricted remote_assisted


class AdmissionProfile(str, enum.Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    STRICT_AUDIT = "strict_audit"


class Visibility(str, enum.Enum):
    CONVERSATION = "conversation"
    WORKSPACE = "workspace"
    OWNER = "owner"


class Provenance(str, enum.Enum):
    DIRECT_USER = "direct_user"
    APPROVED_TOOL = "approved_tool"
    ASSISTANT_GENERATED = "assistant_generated"
    LEGACY_IMPORT = "legacy_import"
    OPERATOR = "operator"
    UNKNOWN = "unknown"


class SourceKind(str, enum.Enum):
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_OUTPUT = "tool_output"
    IMPORT = "import"
    OPERATOR_RECORD = "operator_record"


class Lifecycle(str, enum.Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    ARCHIVED = "archived"
    ERASED = "erased"


class PairLabel(str, enum.Enum):
    EQUIVALENT = "equivalent"
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    DIFFERENT_SCOPE = "different_scope"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class ChangeSignal(str, enum.Enum):
    STATES_CHANGE = "states_change"
    STATES_CORRECTION = "states_correction"
    NEITHER = "neither"


class Precision(str, enum.Enum):
    INSTANT = "instant"
    DAY = "day"
    MONTH = "month"
    YEAR = "year"
    UNKNOWN = "unknown"


class Polarity(str, enum.Enum):
    AFFIRMATIVE = "affirmative"
    NEGATED = "negated"


class Modality(str, enum.Enum):
    ASSERTED = "asserted"
    HYPOTHETICAL = "hypothetical"
    HABITUAL = "habitual"
    UNCERTAIN = "uncertain"


class EdgeType(str, enum.Enum):
    SUPPORTS = "supports"
    CONFLICTS_WITH = "conflicts_with"
    SUPERSEDES = "supersedes"
    CORRECTS = "corrects"
    CONTEXT_OF = "context_of"
    DERIVED_FROM = "derived_from"


class JobKind(str, enum.Enum):
    HARVEST = "harvest"
    ADMIT = "admit"
    COMPARE = "compare"
    EMBED = "embed"
    REINDEX = "reindex"
    REVIEW_APPLY = "review_apply"
    PURGE = "purge"
    EPISODE_INDEX = "episode_index"
    PROCEDURE_VALIDATE = "procedure_validate"
    REPLAY = "replay"
    # --- v3 additions (SPEC_V3 §40) ---
    SCREEN = "screen"
    SPARSE_INDEX = "sparse_index"
    LATE_INDEX = "late_index"
    SIGNATURE_INDEX = "signature_index"
    EPISODE_BUILD = "episode_build"
    TRANSITION_BUILD = "transition_build"
    PROCEDURE_COMPILE = "procedure_compile"
    PROCEDURE_REFINE = "procedure_refine"
    CONSOLIDATE = "consolidate"
    PURGE_DERIVED = "purge_derived"
    PURGE_VAULT = "purge_vault"
    QUARANTINE_REVIEW = "quarantine_review"
    REVOCATION_NOTIFY = "revocation_notify"
    VAULT_ROTATE = "vault_rotate"
    PROJECTION_SYNC = "projection_sync"
    CONNECTOR_PULL = "connector_pull"
    # --- v5 additions (SPEC_V5 §07–§09) ---
    SOURCE_PROJECT = "source_project"
    SOURCE_EMBED = "source_embed"
    SOURCE_BACKFILL = "source_backfill"


class JobState(str, enum.Enum):
    QUEUED = "queued"
    LEASED = "leased"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskKind(str, enum.Enum):
    DURABILITY = "durability"
    PAIR_RELATION = "pair_relation"
    CHANGE_SIGNAL = "change_signal"
    CONDITION_MATCH = "condition_match"
    RELEVANCE = "relevance"
    IMPORTANCE = "importance"
    SPAN_SELECTION = "span_selection"
    STRUCTURE_SELECT = "structure_select"


class RuleOutcome(str, enum.Enum):
    RULE_MATCH = "rule_match"
    RULE_NO_MATCH = "rule_no_match"
    ABSTAIN = "abstain"


class RecallMode(str, enum.Enum):
    CURRENT = "current"
    HISTORICAL = "historical"
    TIMELINE = "timeline"
    EXPANDED = "expanded"
    ARCHIVE = "archive"


class MemoryKind(str, enum.Enum):
    CLAIM = "claim"
    EPISODE = "episode"
    PROCEDURE = "procedure"
    PLAN = "plan"
    WORKING_SET = "working_set"


class FeedbackKind(str, enum.Enum):
    HELPFUL = "helpful"
    IRRELEVANT = "irrelevant"
    POSSIBLY_WRONG = "possibly_wrong"
    FACTUAL_CORRECTION = "factual_correction"
    APPLICABILITY_CORRECTION = "applicability_correction"
    PRIVACY_COMPLAINT = "privacy_complaint"
    TASK_OUTCOME = "task_outcome"


class InterpretationStatus(str, enum.Enum):
    UNSTRUCTURED = "unstructured"
    PARTIAL = "partial"
    STRUCTURED = "structured"


class GrantKind(str, enum.Enum):
    READ_EVIDENCE = "read_evidence"
    PROPOSE = "propose"
    INGEST = "ingest"
    RESOLVE = "resolve"
    SHARE = "share"
    EXPORT = "export"
    SUPPRESS = "suppress"
    PURGE = "purge"
    OPERATOR = "operator"


class EndpointKind(str, enum.Enum):
    """Valid-time endpoint semantics (SPEC_V2 §16): unknown is not infinity."""
    EXACT = "exact"
    UNCERTAIN_RANGE = "uncertain_range"
    UNBOUNDED = "unbounded"
    UNKNOWN = "unknown"


class PlanStatus(str, enum.Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    OVERDUE = "overdue"
    UNKNOWN = "unknown"


class ProcedureState(str, enum.Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    REVIEW = "review"
    RETIRED = "retired"


class CapabilityState(str, enum.Enum):
    IMPLEMENTED = "implemented"
    INSTALLED = "installed"
    CONFIGURED = "configured"
    AUTHORIZED = "authorized"
    HEALTHY = "healthy"
    VALIDATED = "validated"


class ReviewState(str, enum.Enum):
    OPEN = "open"
    APPROVED = "approved"
    REJECTED = "rejected"
    STALE = "stale"


class PurgeState(str, enum.Enum):
    PREVIEWED = "previewed"
    SUPPRESSED = "suppressed"
    PURGING = "purging"
    COMPLETED = "completed"


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def new_id() -> str:
    """UUID4 identifier for sources, claims, decisions, jobs, events."""
    return uuid.uuid4().hex


def require_id(value: str, field_name: str = "id") -> str:
    """Validate an identifier token; rejects text that could confuse logs or SQL."""
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise VerbatimError(ErrorCode.VALIDATION, f"invalid {field_name}: {value!r}")
    return value


def safe_json_loads(text: str, *, max_bytes: int = 1 << 20, max_depth: int = 32) -> Any:
    """Strict JSON: bounded size, rejects NaN/Infinity and duplicate keys."""
    # Size gate: the check is defined on the UTF-8 encoding of ``text``
    # (``errors="replace"`` can only shrink lone surrogates to 1-byte
    # '?', never expand). Every codepoint encodes to ≤4 bytes, so
    # ``len(text) * 4 <= max_bytes`` proves the payload fits without
    # materializing the encoded copy — the common case for the small
    # fields/offsets strings parsed per candidate row.
    if len(text) * 4 > max_bytes and (
        len(text.encode("utf-8", errors="replace")) > max_bytes
    ):
        raise VerbatimError(ErrorCode.VALIDATION, "json payload too large")

    def _no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in pairs:
            if k in out:
                raise VerbatimError(ErrorCode.VALIDATION, f"duplicate json key {k!r}")
            out[k] = v
        return out

    def _const(name: str) -> Any:
        raise VerbatimError(ErrorCode.VALIDATION, f"non-finite json constant {name}")

    try:
        value = json.loads(text, object_pairs_hook=_no_dupes, parse_constant=_const)
    except VerbatimError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise VerbatimError(ErrorCode.VALIDATION, f"invalid json: {exc}") from exc

    # Depth gate, iterative — identical verdict to the recursive walk:
    # any container nested deeper than ``max_depth`` levels fails.
    stack = [(value, 0)]
    push = stack.append
    pop = stack.pop
    while stack:
        v, d = pop()
        if d > max_depth:
            raise VerbatimError(ErrorCode.VALIDATION, "json nesting too deep")
        if isinstance(v, dict):
            for x in v.values():
                push((x, d + 1))
        elif isinstance(v, list):
            for x in v:
                push((x, d + 1))
    return value


def json_dumps(value: Any) -> str:
    """Canonical JSON for persistence: sorted keys, no NaN, compact."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Scope:
    """Authorization partition: profile + principal + workspace + conversation."""

    profile_id: str
    principal_id: Optional[str] = None
    workspace_id: Optional[str] = None
    conversation_id: Optional[str] = None
    visibility: Visibility = Visibility.CONVERSATION

    def __post_init__(self) -> None:
        require_id(self.profile_id, "profile_id")
        for name in ("principal_id", "workspace_id", "conversation_id"):
            v = getattr(self, name)
            if v is not None:
                require_id(v, name)
        if not isinstance(self.visibility, Visibility):
            object.__setattr__(self, "visibility", Visibility(self.visibility))


@dataclass(frozen=True)
class SourceEnvelope:
    """An accepted input event before persistence (SPEC §8, §10)."""

    origin: str
    source_kind: SourceKind
    scope: Scope
    speaker_id: Optional[str]
    payload: bytes
    event_us: int
    captured_us: int
    timezone: Optional[str] = None
    provenance: Provenance = Provenance.UNKNOWN
    external_id: Optional[str] = None
    source_id: Optional[str] = None
    revision: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.payload, (bytes, bytearray)) or len(self.payload) == 0:
            raise VerbatimError(ErrorCode.VALIDATION, "payload must be non-empty bytes")
        if self.revision < 1:
            raise VerbatimError(ErrorCode.VALIDATION, "revision must be >= 1")
        if not isinstance(self.source_kind, SourceKind):
            object.__setattr__(self, "source_kind", SourceKind(self.source_kind))
        if not isinstance(self.provenance, Provenance):
            object.__setattr__(self, "provenance", Provenance(self.provenance))


@dataclass(frozen=True)
class IngestReceipt:
    accepted: tuple[str, ...]
    rejected: tuple[tuple[str, str], ...]
    job_ids: tuple[str, ...]
    projection_generation: int
    duplicate: bool = False


@dataclass(frozen=True)
class SpanRef:
    span_id: str
    source_id: str
    revision: int
    start_byte: int
    end_byte: int


@dataclass(frozen=True)
class TimeInterval:
    """Half-open valid-time interval [from_us, until_us); NULL = unknown.

    V2 endpoint semantics (SPEC_V2 §16): ``start_kind``/``end_kind`` mark each
    bound as exact, an uncertain range, explicitly unbounded, or unknown.
    For an uncertain start the true bound lies in [from_us, from_us_hi];
    ``*_hi`` fields are None unless the endpoint is UNCERTAIN_RANGE.
    """

    from_us: Optional[int] = None
    until_us: Optional[int] = None
    precision: Precision = Precision.UNKNOWN
    timezone: Optional[str] = None
    basis: str = "unknown"
    start_kind: EndpointKind = EndpointKind.EXACT
    end_kind: EndpointKind = EndpointKind.EXACT
    from_us_hi: Optional[int] = None
    until_us_hi: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.precision, Precision):
            object.__setattr__(self, "precision", Precision(self.precision))
        if not isinstance(self.start_kind, EndpointKind):
            object.__setattr__(self, "start_kind", EndpointKind(self.start_kind))
        if not isinstance(self.end_kind, EndpointKind):
            object.__setattr__(self, "end_kind", EndpointKind(self.end_kind))
        if self.from_us is not None and self.until_us is not None:
            if self.until_us <= self.from_us:
                raise VerbatimError(ErrorCode.VALIDATION, "empty or inverted interval")
        if self.from_us_hi is not None and self.from_us is not None:
            if self.from_us_hi < self.from_us:
                raise VerbatimError(ErrorCode.VALIDATION, "uncertain start hi < lo")
        if self.until_us_hi is not None and self.until_us is not None:
            if self.until_us_hi < self.until_us:
                raise VerbatimError(ErrorCode.VALIDATION, "uncertain end hi < lo")

    def applicability_at(self, t_us: int) -> Optional[bool]:
        """Three-valued applicability of timestamp T (SPEC_V2 §16.18).

        True  — certainly applicable: s_hi <= T < e_lo.
        False — provably outside the bounds.
        None  — possible but not certain (unknown lane).

        For an exact endpoint, lo == hi. For UNCERTAIN_RANGE, lo/hi bracket
        the true bound. UNBOUNDED means no constraint; UNKNOWN means the
        bound cannot prove membership — but a known other endpoint can
        still prove non-membership.
        """
        # Resolve effective bounds. UNBOUNDED/UNKNOWN endpoints contribute
        # no limit on their side. Legacy shorthand: an EXACT start with no
        # stored bound is an unknown start; an EXACT end with no stored
        # bound is an unbounded (open-ended) end — v1's asserted_current_at.
        if self.start_kind in (EndpointKind.UNBOUNDED, EndpointKind.UNKNOWN):
            s_lo = s_hi = None
        else:
            s_lo = self.from_us
            s_hi = self.from_us_hi if self.from_us_hi is not None else self.from_us
        if self.end_kind in (EndpointKind.UNBOUNDED, EndpointKind.UNKNOWN):
            e_lo = e_hi = None
        else:
            e_lo = self.until_us
            e_hi = self.until_us_hi if self.until_us_hi is not None else self.until_us

        start_unknown = self.start_kind == EndpointKind.UNKNOWN or (
            self.start_kind == EndpointKind.EXACT and s_lo is None
        )
        end_unknown = self.end_kind == EndpointKind.UNKNOWN
        unbounded_end = self.end_kind == EndpointKind.UNBOUNDED or (
            self.end_kind == EndpointKind.EXACT and e_lo is None
        )

        # Provable violations: T above the highest possible end, or below
        # the lowest possible start.
        if e_hi is not None and t_us >= e_hi:
            return False
        if s_lo is not None and t_us < s_lo:
            return False

        # Certainty requires provable lower AND upper containment.
        lower_ok = (s_hi is not None and t_us >= s_hi) and not start_unknown
        upper_ok = unbounded_end or (e_lo is not None and t_us < e_lo)
        if end_unknown:
            upper_ok = False
        if lower_ok and upper_ok:
            return True
        return None


@dataclass(frozen=True)
class Condition:
    """Bounded expression tree: all/any/not over typed equality predicates.

    ``op`` is one of "all", "any", "not", "eq". ``key``/``value`` are set only
    on "eq" leaves; ``children`` on composite nodes. Max depth 3, max 8 leaves
    (SPEC §12).
    """

    op: str
    key: Optional[str] = None
    value: Optional[str] = None
    children: tuple["Condition", ...] = ()

    MAX_DEPTH = 3
    MAX_LEAVES = 8

    def validate(self, depth: int = 1) -> int:
        if depth > self.MAX_DEPTH:
            raise VerbatimError(ErrorCode.VALIDATION, "condition depth exceeds 3")
        if self.op == "eq":
            if self.key is None or self.value is None:
                raise VerbatimError(ErrorCode.VALIDATION, "eq condition needs key/value")
            return 1
        if self.op not in ("all", "any", "not"):
            raise VerbatimError(ErrorCode.VALIDATION, f"unknown condition op {self.op!r}")
        if self.op == "not" and len(self.children) != 1:
            raise VerbatimError(ErrorCode.VALIDATION, "not takes exactly one child")
        if not self.children:
            raise VerbatimError(ErrorCode.VALIDATION, "composite condition needs children")
        leaves = sum(c.validate(depth + 1) for c in self.children)
        if leaves > self.MAX_LEAVES:
            raise VerbatimError(ErrorCode.VALIDATION, "condition leaf count exceeds 8")
        return leaves

    def to_json(self) -> dict[str, Any]:
        if self.op == "eq":
            return {"op": "eq", "key": self.key, "value": self.value}
        return {"op": self.op, "children": [c.to_json() for c in self.children]}

    @staticmethod
    def from_json(obj: Any) -> "Condition":
        if not isinstance(obj, dict) or "op" not in obj:
            raise VerbatimError(ErrorCode.VALIDATION, "invalid condition json")
        if obj["op"] == "eq":
            c = Condition("eq", key=obj.get("key"), value=obj.get("value"))
        else:
            c = Condition(
                obj["op"],
                children=tuple(Condition.from_json(x) for x in obj.get("children", [])),
            )
        c.validate()
        return c

    def evaluate(self, context: dict[str, Any]) -> Optional[bool]:
        """Three-valued evaluation against a supplied context (SPEC_V2 §17).

        ``context`` maps keys to values; a missing key yields UNKNOWN for
        that leaf. ``not unknown`` stays unknown; an empty context never
        satisfies a non-empty condition. Returns True/False/None.
        """
        self.validate()
        return self._eval(context)

    def _eval(self, context: dict[str, Any]) -> Optional[bool]:
        if self.op == "eq":
            if self.key not in context:
                return None
            return context[self.key] == self.value
        values = [c._eval(context) for c in self.children]
        if self.op == "not":
            v = values[0]
            return None if v is None else not v
        if self.op == "all":
            if any(v is False for v in values):
                return False
            if any(v is None for v in values):
                return None
            return True
        # "any"
        if any(v is True for v in values):
            return True
        if any(v is None for v in values):
            return None
        return False

    def required_keys(self) -> tuple[str, ...]:
        """Context keys this expression reads (for clarification hints)."""
        if self.op == "eq":
            return (self.key,) if self.key else ()
        out: list[str] = []
        for c in self.children:
            out.extend(c.required_keys())
        return tuple(sorted(set(out)))


@dataclass(frozen=True)
class ClaimProposal:
    """A typed interpretation candidate; not automatically a true fact."""

    evidence: tuple[SpanRef, ...]
    predicate: Optional[str] = None
    object_json: Optional[dict[str, Any]] = None
    subject_entity_id: Optional[str] = None
    polarity: Polarity = Polarity.AFFIRMATIVE
    modality: Modality = Modality.ASSERTED
    condition: Optional[Condition] = None
    valid: TimeInterval = field(default_factory=TimeInterval)
    method: str = "rules"
    confidence_kind: str = "rule"
    agent_suggested: bool = False


@dataclass(frozen=True)
class DecisionRequest:
    """Immutable snapshot handed to a decision backend (SPEC §23)."""

    task: TaskKind
    state: dict[str, Any]
    allowed_labels: tuple[str, ...]
    deadline_s: float
    policy_epoch: int
    rubric_version: str = "rubric-1"
    purpose: str = "candidate_curation"


@dataclass(frozen=True)
class DecisionResult:
    """Advisory typed judgment; deterministic code applies effects."""

    task: TaskKind
    backend: str
    model_revision: Optional[str]
    rubric_version: str
    outcome: dict[str, Any]
    abstained: bool = False
    reason: Optional[str] = None
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RecallRequest:
    """RecallRequestV2 (SPEC_V2 §25): bounded, typed, immutable query.

    ``scope`` narrows the bound caller's audience — it may only shrink the
    caller's authorized visibility, never widen it. ``context`` carries
    explicit typed context for condition evaluation (e.g.
    {"environment": "work"}); free-form keys are allowed but bounded.
    """

    query: str
    scope: Scope
    mode: RecallMode = RecallMode.CURRENT
    limit: int = 8
    valid_at_us: Optional[int] = None
    valid_until_us: Optional[int] = None
    known_at_seq: Optional[int] = None
    entity_ids: tuple[str, ...] = ()
    max_bytes: int = 6000
    include_sources: tuple[str, ...] = ("lexical", "semantic", "structured", "graph")
    memory_kinds: tuple[MemoryKind, ...] = (MemoryKind.CLAIM,)
    context: dict[str, Any] = field(default_factory=dict)
    deadline_ms: int = 200
    target_tokens: int = 1536
    min_ready_seq: Optional[int] = None
    wire_version: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RecallMode):
            object.__setattr__(self, "mode", RecallMode(self.mode))
        if not (1 <= self.limit <= 32):
            raise VerbatimError(ErrorCode.VALIDATION, "limit must be 1..32")
        if len(self.query) > 8192:
            raise VerbatimError(ErrorCode.VALIDATION, "query exceeds 8192 characters")
        if len(self.query.encode("utf-8", errors="replace")) > 32768:
            raise VerbatimError(ErrorCode.VALIDATION, "query exceeds 32 KiB")
        if len(self.entity_ids) > 8:
            raise VerbatimError(ErrorCode.VALIDATION, "entity_ids bounded at 8")
        if not (512 <= self.max_bytes <= 24000):
            raise VerbatimError(ErrorCode.VALIDATION, "max_bytes must be 512..24000")
        if not (1 <= self.deadline_ms <= 10_000):
            raise VerbatimError(ErrorCode.VALIDATION, "deadline_ms must be 1..10000")
        if self.valid_until_us is not None and self.valid_at_us is not None:
            if self.valid_until_us <= self.valid_at_us:
                raise VerbatimError(ErrorCode.VALIDATION, "empty valid range")
        mk = tuple(
            k if isinstance(k, MemoryKind) else MemoryKind(k)
            for k in self.memory_kinds
        )
        object.__setattr__(self, "memory_kinds", mk)
        if len(self.context) > 16:
            raise VerbatimError(ErrorCode.VALIDATION, "context bounded at 16 keys")


@dataclass(frozen=True)
class EvidenceItem:
    claim_id: str
    claim_revision: int
    text: str
    span: SpanRef
    speaker_id: Optional[str]
    lifecycle: Lifecycle
    valid_label: str
    recorded_seq: int
    reasons: tuple[str, ...]
    provenance: Provenance = Provenance.UNKNOWN
    disputed: bool = False
    historical: bool = False


@dataclass(frozen=True)
class RecallResult:
    items: tuple[EvidenceItem, ...]
    omitted: int = 0
    warnings: tuple[str, ...] = ()
    capabilities: dict[str, Any] = field(default_factory=dict)
    projection_generation: int = 0
    # Atomic evidence groups behind ``items`` (SPEC_V2 §30). Each group is a
    # primary claim plus its required context and unresolved contrary
    # evidence; ``items`` remains the flattened view for v1 consumers.
    groups: tuple["EvidenceGroup", ...] = ()


@dataclass(frozen=True)
class TransitionCommand:
    """Authorized lifecycle mutation request (SPEC §15, §35)."""

    claim_id: str
    expected_revision: int
    effect: str
    actor_id: str
    reason: str
    successor_claim_id: Optional[str] = None
    interval: Optional[TimeInterval] = None


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: Optional[dict[str, Any]] = None
    warnings: tuple[str, ...] = ()
    error: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": self.ok}
        if self.data is not None:
            out["data"] = self.data
        if self.warnings:
            out["warnings"] = list(self.warnings)
        if self.error is not None:
            out["error"] = self.error
        return out


def tool_error(exc: VerbatimError) -> ToolResult:
    return ToolResult(ok=False, error=exc.to_dict())


# ---------------------------------------------------------------------------
# V2 contract types (SPEC_V2 §09, §20, §25, §30, §37)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallerContext:
    """Bound authenticated caller identity (SPEC_V2 §09).

    Established by the host or a trusted transport; untrusted request data
    may narrow but never widen it. ``grants`` holds GrantKind values the
    principal was explicitly issued; ``audience`` is the set of principals
    a response may be disclosed to (empty = owner-private to principal).
    """

    profile_id: str
    principal_id: str
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    workspace_id: Optional[str] = None
    conversation_id: Optional[str] = None
    audience: tuple[str, ...] = ()
    grants: frozenset = frozenset()
    authz_epoch: int = 0
    is_operator: bool = False

    def __post_init__(self) -> None:
        require_id(self.profile_id, "profile_id")
        require_id(self.principal_id, "principal_id")
        for name in ("agent_id", "session_id", "workspace_id", "conversation_id"):
            v = getattr(self, name)
            if v is not None:
                require_id(v, name)
        object.__setattr__(
            self,
            "grants",
            frozenset(
                g if isinstance(g, GrantKind) else GrantKind(g) for g in self.grants
            ),
        )
        if self.authz_epoch < 0:
            raise VerbatimError(ErrorCode.VALIDATION, "authz_epoch must be >= 0")

    def has(self, grant: GrantKind) -> bool:
        return grant in self.grants or GrantKind.OPERATOR in self.grants or self.is_operator

    def scope(self, visibility: "Visibility" = None) -> Scope:
        """The caller's home scope; narrowing happens per-request."""
        return Scope(
            profile_id=self.profile_id,
            principal_id=self.principal_id,
            workspace_id=self.workspace_id,
            conversation_id=self.conversation_id,
            visibility=visibility or Visibility.CONVERSATION,
        )


@dataclass(frozen=True)
class EffectProposal:
    """One versioned proposal schema for every effect (SPEC_V2 §20).

    API, CLI, workbench, jobs, and adapters all carry this exact shape;
    ``targets`` maps object IDs to their expected current revisions.
    """

    version: int
    operation_id: str
    effect: str  # admit/reject/dispute/resolve/correct/supersede/archive/
    #            restore/reconsider/reverse_supersede/erase
    targets: tuple[tuple[str, int], ...]
    actor_id: str
    reason: str
    successor_claim_id: Optional[str] = None
    interval_effects: tuple[TimeInterval, ...] = ()
    reason_codes: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    policy_epoch: int = 0
    authz_epoch: int = 0

    EFFECTS = (
        "admit", "reject", "dispute", "resolve", "correct", "supersede",
        "archive", "restore", "reconsider", "reverse_supersede", "erase",
    )

    def __post_init__(self) -> None:
        if self.version < 1:
            raise VerbatimError(ErrorCode.VALIDATION, "proposal version must be >= 1")
        if self.effect not in self.EFFECTS:
            raise VerbatimError(ErrorCode.VALIDATION, f"unknown effect {self.effect!r}")
        if not self.targets:
            raise VerbatimError(ErrorCode.VALIDATION, "proposal needs >= 1 target")
        require_id(self.operation_id, "operation_id")
        if not self.actor_id or not self.reason:
            raise VerbatimError(ErrorCode.VALIDATION, "actor and reason required")

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "operation_id": self.operation_id,
            "effect": self.effect,
            "targets": [list(t) for t in self.targets],
            "actor_id": self.actor_id,
            "reason": self.reason,
            "successor_claim_id": self.successor_claim_id,
            "reason_codes": list(self.reason_codes),
            "params": self.params,
            "policy_epoch": self.policy_epoch,
            "authz_epoch": self.authz_epoch,
        }

    @staticmethod
    def from_json(obj: Any) -> "EffectProposal":
        if not isinstance(obj, dict):
            raise VerbatimError(ErrorCode.VALIDATION, "invalid proposal json")
        return EffectProposal(
            version=int(obj.get("version", 1)),
            operation_id=str(obj.get("operation_id", "")),
            effect=str(obj.get("effect", "")),
            targets=tuple((str(t[0]), int(t[1])) for t in obj.get("targets", ())),
            actor_id=str(obj.get("actor_id", "")),
            reason=str(obj.get("reason", "")),
            successor_claim_id=obj.get("successor_claim_id"),
            reason_codes=tuple(obj.get("reason_codes", ())),
            params=dict(obj.get("params", {})),
            policy_epoch=int(obj.get("policy_epoch", 0)),
            authz_epoch=int(obj.get("authz_epoch", 0)),
        )


@dataclass(frozen=True)
class ContextGroup:
    """A harvested evidence group: primary spans plus required context."""

    group_id: str
    scope_id: str
    source_id: str
    revision: int
    parser_version: str
    operation_key: str
    completeness: str = "complete"  # complete|partial|deferred
    recorded_from: int = 0
    recorded_until: Optional[int] = None


@dataclass(frozen=True)
class EvidenceGroup:
    """An atomic retrieval unit: primary claim + essential context +
    unresolved contrary evidence (SPEC_V2 §30)."""

    primary_claim_id: str
    items: tuple[EvidenceItem, ...]
    complete: bool
    reasons: tuple[str, ...] = ()
    group_score: float = 0.0
    serialized_bytes: int = 0


@dataclass(frozen=True)
class OperationReceipt:
    """Durable idempotence record (SPEC_V2 §38, §39)."""

    operation_key: str
    scope_id: str
    effect_kind: str
    input_digest: bytes
    committed_event: int
    result_json: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeRef:
    episode_id: str
    scope_id: str
    revision: int
    host_task_id: Optional[str] = None
    parent_episode_id: Optional[str] = None


@dataclass(frozen=True)
class CapabilityReport:
    """Per-capability truthfulness (SPEC_V2 §05, §50)."""

    name: str
    state: CapabilityState
    degraded_reason: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)
