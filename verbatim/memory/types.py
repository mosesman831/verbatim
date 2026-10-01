"""Frozen V5 public contract types (SPEC_V5 §06, §30).

Everything here is a typed, versioned, deterministic-to-serialize record.
Workers: treat these shapes as the contract — extend, never reshape.
"""
from __future__ import annotations

import dataclasses
import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

CONTRACT = "v5-contracts/1"


# ---------------------------------------------------------------- enums


class Acceptance(str, enum.Enum):
    ACCEPTED = "accepted"
    HELD = "held"
    PROTECTED = "protected"
    #: Per-item capture failure inside ``Memory.bulk_add`` — the item's
    #: own record rolled back (typed reason on ``AddResult.error``); never
    #: reported for a committed write.
    FAILED = "failed"


class SearchStatus(str, enum.Enum):
    READY = "ready"
    PARTIAL = "partial"
    PENDING = "pending"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"


class ChangeKind(str, enum.Enum):
    SUPERSEDE = "supersede"
    CORRECT = "correct"


class Consistency(str, enum.Enum):
    SESSION = "session"
    EVENTUAL = "eventual"


class WorkerMode(str, enum.Enum):
    MANAGED = "managed"
    EXTERNAL = "external"


class MemoryType(str, enum.Enum):
    """§30.1 interpretation label — never authority."""

    FACT = "fact"
    PREFERENCE = "preference"
    DECISION = "decision"
    PLAN = "plan"
    STATE = "state"
    EVENT = "event"
    RELATIONSHIP = "relationship"
    PROCEDURE_HINT = "procedure_hint"
    ABSENCE = "absence"
    UNTYPED = "untyped"


class Polarity(str, enum.Enum):
    AFFIRMATIVE = "affirmative"
    NEGATED = "negated"
    HEDGED = "hedged"
    HYPOTHETICAL = "hypothetical"
    QUOTED = "quoted"


class TimePrecision(str, enum.Enum):
    EXACT = "exact"
    DAY = "day"
    MONTH = "month"
    YEAR = "year"
    RELATIVE = "relative"
    UNKNOWN = "unknown"


class TimeStatus(str, enum.Enum):
    ONGOING = "ongoing"
    COMPLETED = "completed"
    PLANNED = "planned"
    UNKNOWN = "unknown"


class MatchClass(str, enum.Enum):
    EXACT = "exact"
    NORMALIZED = "normalized"
    PARAPHRASE = "paraphrase"


class Lifecycle(str, enum.Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CORRECTED = "corrected"
    RETRACTED = "retracted"
    EXPIRED = "expired"


class SupportStatus(str, enum.Enum):
    SUPPORTED = "supported"
    DISPUTED = "disputed"
    INSUFFICIENT = "insufficient"
    UNASSESSED = "unassessed"


class QueryClass(str, enum.Enum):
    """Deterministic query_analysis/v1 classes (§31.1)."""

    IDENTIFIER = "identifier"
    ENTITY = "entity"
    TEMPORAL = "temporal"
    PREFERENCE = "preference"
    PROCEDURAL = "procedural"
    FACTUAL = "factual"
    NO_ANSWER_LIKELY = "no_answer_likely"


QUERY_ANALYSIS_VERSION = "query_analysis/v1"
RANKING_VERSION = "ranking/v1"
ENRICHMENT_VERSION = "enrich/v1"
SOURCE_STATE_KIND = "source_state/v1"

#: Readiness capabilities added by V5 (§08) — registered by readiness layer.
CAP_SOURCE_LEXICAL = "source_lexical_ready"
CAP_SOURCE_VECTOR = "source_vector_ready"


# ---------------------------------------------------------------- refs


@dataclass(frozen=True)
class MemoryRef:
    """Version-bound reference to a source-backed memory (§06.11).

    Serialized form: ``mref1.<store_tag>.<namespace>.<source_id>.<rev>.<ctl>``
    """

    store_tag: str
    namespace: str
    source_id: str
    expected_revision: int
    control_version: int

    PREFIX = "mref1"

    def to_string(self) -> str:
        parts = (
            self.PREFIX,
            _enc(self.store_tag),
            _enc(self.namespace),
            _enc(self.source_id),
            str(self.expected_revision),
            str(self.control_version),
        )
        return ".".join(parts)

    @classmethod
    def parse(cls, text: str) -> "MemoryRef":
        parts = text.split(".")
        if len(parts) != 6 or parts[0] != cls.PREFIX:
            raise ValueError(f"not a MemoryRef: {text!r}")
        return cls(
            store_tag=_dec(parts[1]),
            namespace=_dec(parts[2]),
            source_id=_dec(parts[3]),
            expected_revision=int(parts[4]),
            control_version=int(parts[5]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref": self.to_string(),
            "namespace": self.namespace,
            "source_id": self.source_id,
            "expected_revision": self.expected_revision,
            "control_version": self.control_version,
        }


def _enc(value: str) -> str:
    return value.replace("%", "%25").replace(".", "%2E")


def _dec(value: str) -> str:
    return value.replace("%2E", ".").replace("%25", "%")


# ---------------------------------------------------------------- results


def _ser(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {k: _ser(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_ser(v) for v in value]
    if isinstance(value, dict):
        return {k: _ser(v) for k, v in value.items()}
    return value


@dataclass
class Result:
    schema: str = CONTRACT

    def to_dict(self) -> Dict[str, Any]:
        return _ser(self)


@dataclass
class UpdateCandidate(Result):
    """Advisory possible_update (§30.5) — nothing mutates until replaces=."""

    ref: str = ""  # serialized MemoryRef of the prior record
    relation: str = ""  # contradicts | newer_value | negates | refines
    reason: str = ""
    score: float = 0.0


@dataclass
class AddResult(Result):
    memory_id: str = ""
    ref: str = ""
    source_revision: int = 0
    receipt_id: str = ""
    acceptance: str = Acceptance.ACCEPTED.value
    replayed: bool = False
    readiness: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    possible_updates: List[UpdateCandidate] = field(default_factory=list)
    inference: str = "not_requested"  # not_requested | queued | deferred | unavailable
    #: Per-item failure detail for ``bulk_add`` results — populated only
    #: when ``acceptance == "failed"`` (the item's record rolled back);
    #: ``None`` on every successful/replayed capture.
    error: Optional[str] = None


@dataclass
class Hit(Result):
    memory_id: str = ""
    ref: str = ""  # MemoryRef for source-backed hits
    object_ref: str = ""  # exact claim/view object ref
    kind: str = "source"  # source | claim | view | card
    quote: str = ""
    score: float = 0.0
    score_family: str = "ranking/v1"
    lifecycle: str = Lifecycle.ACTIVE.value
    support_status: str = SupportStatus.UNASSESSED.value
    role: str = "supporting"  # supporting | contrary | context
    type: str = MemoryType.UNTYPED.value
    valid_time: Optional[str] = None
    recorded_time: Optional[str] = None
    collapsed_duplicates: int = 0
    corroboration: int = 1
    warnings: List[str] = field(default_factory=list)
    score_detail: Dict[str, float] = field(default_factory=dict)


@dataclass
class SearchResult(Result):
    status: str = SearchStatus.PENDING.value
    items: List[Hit] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    readiness: Dict[str, Any] = field(default_factory=dict)
    coverage: Dict[str, Any] = field(default_factory=dict)
    causal_token: str = ""
    #: V8-12.03 — the verdict report's advisory answerability
    #: (``supported | partial | weak_only | unverified_premise |
    #: contradicted_premise | no_evidence``).  ``None`` when the pipeline
    #: carried no verdict report — never fabricated from ``status``.
    #: Declared (not dynamic) so ``to_dict``/``asdict``/the MCP
    #: ``_to_jsonable`` wire shape all carry it.
    answerability: Optional[str] = None


@dataclass
class Inspection(Result):
    ref: str = ""
    found: bool = False
    detail: str = "evidence"
    provenance: Dict[str, Any] = field(default_factory=dict)
    revisions: List[Dict[str, Any]] = field(default_factory=list)
    lifecycle: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    enrichment: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ForgetResult(Result):
    mode: str = "operation"  # preview | operation
    mutated: bool = True
    confirmation_token: str = ""
    selection: List[str] = field(default_factory=list)
    suppression_state: str = ""
    closure_state: str = ""
    receipt_id: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class Readiness(Result):
    receipt_id: str = ""
    state: str = "pending"  # ready | partial | pending | blocked | unavailable
    capabilities: Dict[str, str] = field(default_factory=dict)
    causal_satisfied: bool = False
    waited_ms: float = 0.0


@dataclass
class MemoryStatus(Result):
    profile: str = "local_memory"
    store_tag: str = ""
    namespace: str = ""
    caller: str = ""
    worker: Dict[str, Any] = field(default_factory=dict)
    encoder: str = "hashing:subword-ngram:v1"
    cache: Dict[str, Any] = field(default_factory=dict)
    readiness_counts: Dict[str, int] = field(default_factory=dict)
    capabilities: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class CloseReport(Result):
    closed: bool = False
    drained: bool = False
    worker_stopped: bool = False
    pending_obligations: int = 0
    incomplete: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def monotonic_ms() -> float:
    return time.monotonic() * 1000.0
