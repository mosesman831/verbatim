"""V7 contract types (SPEC_V7 R1).

Frozen coordination surface for the V7 wave program (V7-27.01). Every module
under ``verbatim/retrieval/v7/``, the V7 enrichment path, and ``eval/v7/``
imports from here. Ownership: main session only — workers never edit this
file (``docs/v7_contracts.md``).

Design rules carried from V3–V6 and restated here so implementers cannot
miss them:

- Eligibility is evaluated before rank on every lane (V7-05.08).
- Degraded stages report honestly; degradation never widens authorization
  and never turns a result ``ready`` (V7-04.03).
- Every deliverable-as-text output carries byte-verified pins into retained
  source bytes or is marked ``unsupported_extraction`` (V7-13.06).
- No stage issues a network call on default profiles (V7-04.02).
- All §32 formula constants are ``provisional/v7-r0`` until the formula
  search selects (V7-32.01); artifacts record the tag that produced them.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

FORMULA_STATUS_PROVISIONAL = "provisional/v7-r0"


class UnitKind(str, enum.Enum):
    """Retrievable evidence unit granularities (V7-13.03, §30 `units.kind`)."""

    TURN = "turn"
    SENTENCE_WINDOW = "sentence_window"
    SESSION = "session"
    EPISODE = "episode"


class OccurredPrecision(str, enum.Enum):
    INSTANT = "instant"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    SEASON = "season"
    YEAR = "year"
    DECADE = "decade"
    UNKNOWN = "unknown"


class OccurredSource(str, enum.Enum):
    EXPLICIT = "explicit"
    RESOLVED_RELATIVE = "resolved_relative"
    SESSION_DEFAULT = "session_default"
    UNKNOWN = "unknown"


class Perspective(str, enum.Enum):
    """Unit perspective classification (V7-13.07)."""

    USER_STATED = "user_stated"
    AGENT_STATED = "agent_stated"
    AGENT_ACTION = "agent_action"
    THIRD_PARTY = "third_party"
    DOCUMENT = "document"
    SYSTEM = "system"


class IntentClass(str, enum.Enum):
    """Query intent classes (V7-05.12, `intent/v2` §32.14)."""

    LOOKUP = "lookup"
    IDENTIFIER = "identifier"
    TEMPORAL_POINT = "temporal_point"
    TEMPORAL_RANGE = "temporal_range"
    TEMPORAL_ORDER = "temporal_order"
    DURATION = "duration"
    COUNT_AGGREGATE = "count_aggregate"
    CURRENT_VALUE = "current_value"
    HISTORY_OF = "history_of"
    PREFERENCE = "preference"
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    OPEN_DOMAIN = "open_domain"
    WHY_CAUSAL = "why/causal"
    ABSTAIN_LIKELY = "abstain_likely"


class LaneName(str, enum.Enum):
    """S2 lane identifiers (§04.2). Values are stable coverage keys."""

    LEX = "lex"
    FUZZY = "fuzzy"
    DENSE = "dense"
    ENT = "ent"
    TIME = "time"
    GRAPH = "graph"
    TYPED = "typed"
    OBS = "obs"
    EXACT_ID = "exact_id"
    SOURCE = "source"


class LaneStatus(str, enum.Enum):
    """Per-stage honesty status (V7-04.03)."""

    OK = "ok"
    SKIPPED = "skipped"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    DEADLINE = "deadline"


class BudgetClass(str, enum.Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class SupportLabel(str, enum.Enum):
    """Per-group support verdict labels (V7-11.01)."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    WEAK = "weak"


class ResultStatus(str, enum.Enum):
    READY = "ready"
    INSUFFICIENT = "insufficient"


class QuantMode(str, enum.Enum):
    F32 = "f32"
    INT8 = "int8"
    BIT = "bit"


class EdgeType(str, enum.Enum):
    """Typed weighted graph edges (V7-08.07)."""

    CO_MENTION = "co_mention"
    SAME_SESSION = "same_session"
    ADJACENT_TURN = "adjacent_turn"
    TEMPORAL_NEAR = "temporal_near"
    SEMANTIC_KNN = "semantic_knn"
    SUPERSEDES = "supersedes"
    CONTRADICTS = "contradicts"
    REFINES = "refines"
    CAUSAL = "causal"
    CAUSAL_CANDIDATE = "causal_candidate"  # model-proposed; never eligibility (V7-08.14)


class AliasState(str, enum.Enum):
    ACTIVE = "active"
    CANDIDATE = "candidate"
    REJECTED = "rejected"


class AliasMethod(str, enum.Enum):
    RULE = "rule"
    CALLER = "caller"
    REVIEW = "review"


class MentionRole(str, enum.Enum):
    SUBJECT = "subject"
    OBJECT = "object"
    SPEAKER = "speaker"
    MENTION = "mention"


class StateFactStatus(str, enum.Enum):
    CURRENT = "current"
    HISTORICAL = "historical"
    DISPUTED = "disputed"


class LifecycleLabel(str, enum.Enum):
    CURRENT = "current"
    HISTORICAL = "historical"
    SUPERSEDED = "superseded"
    DISPUTED = "disputed"


class ProducerTier(str, enum.Enum):
    """Write-path tier that produced a derived artifact (§04.1)."""

    T0 = "t0"  # deterministic, default on every profile
    T1 = "t1"  # optional local small models
    T2 = "t2"  # optional grounded LLM extraction


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntervalUs:
    """An occurred/recorded interval in microseconds with precision metadata
    (V7-09.01). ``rule_id`` names the `temporal/v2` rule (T01–T30) that
    produced it; ``anchor_us`` records the anchor it resolved against."""

    start_us: Optional[int]
    end_us: Optional[int]
    precision: OccurredPrecision = OccurredPrecision.UNKNOWN
    source: OccurredSource = OccurredSource.UNKNOWN
    rule_id: Optional[str] = None
    anchor_us: Optional[int] = None

    @property
    def known(self) -> bool:
        return self.start_us is not None and self.end_us is not None


@dataclass(frozen=True)
class ResolvedTime:
    """Output of `temporal/v2` resolution for one expression span (§32.9)."""

    text: str
    byte_start: int
    byte_end: int
    interval: IntervalUs
    rule_id: str
    ambiguous_locale: bool = False


# ---------------------------------------------------------------------------
# Analyzer / query view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormTerm:
    """One analyzed term. ``channel`` is ``text`` | ``stem`` | ``identifier``;
    offsets always refer to the original source bytes (§32.1)."""

    term: str
    channel: str
    byte_start: int
    byte_end: int


@dataclass(frozen=True)
class NormAnalysis:
    """`norm/v2` output (V7-05.10, §32.1). Immutable and deterministic."""

    analyzer_id: str  # "norm/v2"
    terms: tuple[NormTerm, ...]
    identifiers: tuple[NormTerm, ...]
    text: str = ""  # the matching-projection string (folded), for debugging


@dataclass(frozen=True)
class IntentResult:
    """`intent/v2` output (V7-05.12). ``classes`` includes the primary first."""

    primary: IntentClass
    classes: tuple[IntentClass, ...]
    window: Optional[IntervalUs] = None
    rule_trace: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryViewV7:
    """The S1 analysis product every lane consumes (§04.2 S1)."""

    query: str
    norm: NormAnalysis
    intent: IntentResult
    entity_canons: tuple[str, ...] = ()
    speaker_canon: Optional[str] = None
    facets: tuple["QueryViewV7", ...] = ()  # decomposition sub-queries (V7-05.13)
    query_time_us: Optional[int] = None


# ---------------------------------------------------------------------------
# Lane contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateV7:
    """One lane candidate before fusion (V7-05.04). ``rank`` is 1-based within
    its lane; ``signals`` are raw, un-normalized lane-local measurements."""

    unit_id: str
    source_id: str
    revision: int
    lane: str
    rank: int
    raw_score: float
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class LaneOutput:
    """What a lane returns to fusion. ``status``/`reason` are coverage-honest
    (V7-04.03); ``candidates`` is ranked, capped by the pool profile."""

    lane: str
    status: LaneStatus
    candidates: list[CandidateV7] = field(default_factory=list)
    reason: Optional[str] = None
    examined: int = 0
    eligible: int = 0
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LaneSlice:
    """Per-lane deadline + pool allocation computed at S2 entry (V7-05.05)."""

    deadline_ms: float
    cap: int


class LaneV7:
    """Lane protocol (V7-05.01–09). Implementations live in
    ``verbatim/retrieval/v7/``. Rules:

    - Evaluate eligibility inside candidate production; never oversample-then-
      filter past the eligible bound (V3/V4 discipline carried).
    - Consume the caller's read snapshot; never open a new transaction.
    - Honor ``slice.deadline_ms``; a cut scan returns ``PARTIAL`` with
      ``reason="deadline"`` and honest ``examined``/``eligible`` counts.
    - ``unavailable``/``skipped`` always carry a ``reason``.
    """

    name: str

    def run(self, ctx: "LaneContextV7", query: QueryViewV7, slice: LaneSlice) -> LaneOutput:
        raise NotImplementedError


@dataclass
class LaneContextV7:
    """Everything a lane may touch: the caller's pinned read snapshot plus
    request-scoped authorization and eligibility handles. Concrete wiring is
    assembled by the pipeline; lanes never widen ``eligible``."""

    store: Any  # Store (read snapshot already pinned by caller)
    scope_id: str
    generation: int
    eligible: Any  # callable(unit_row) -> bool, or eligible-set object
    query_time_us: int
    profile: str
    budget: BudgetClass
    policy: "RetrievalPolicyV7"
    manifest: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fusion / rerank / boosts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusedCandidate:
    """Post-S3 candidate: RRF score plus lane-rank provenance (V7-10.05)."""

    unit_id: str
    source_id: str
    revision: int
    rrf: float
    lane_ranks: dict[str, int]  # lane name -> 1-based rank
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureVector:
    """Named S4 features for one candidate (§32.4). Missing features are
    absent from ``values``, never invented."""

    unit_id: str
    values: dict[str, float]


@dataclass(frozen=True)
class ScoredCandidate:
    """Post-S4/S5/S6 candidate. ``score`` is comparable within one result
    only; ``score_family`` declares the family (V7-10.12)."""

    unit_id: str
    source_id: str
    revision: int
    score: float
    score_family: str  # "ranking/v7"
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolProfile:
    """Pool sizes per budget class (§32.3)."""

    lane_cap: int
    rerank_pool: int
    ce_pool: int
    neighbor_window: int
    max_facets: int


POOLS: dict[BudgetClass, PoolProfile] = {
    BudgetClass.LOW: PoolProfile(50, 40, 0, 0, 1),
    BudgetClass.MID: PoolProfile(200, 100, 16, 1, 2),
    BudgetClass.HIGH: PoolProfile(800, 300, 32, 2, 3),
}


#: V75-04.02 lexical nomination budget — the r0 snapshot value carried as
#: the default ``RetrievalPolicyV7.nominate_terms_max``.  A Q1
#: formula-search constant (SPEC_V7_5 §05); 32 reproduces the pre-V7.5
#: ``_MAX_TERMS`` envelope the positional truncation enforced.
NOMINATE_TERMS_MAX_R0 = 32


@dataclass(frozen=True)
class RetrievalPolicyV7:
    """`retrieval_policy/v7` — declared, versioned, printed in coverage
    (V7-05.02). ``lane_weights[intent][lane]`` default 1.0; lanes absent from
    ``lanes`` never run.  ``nominate_terms_max`` is the V75-04.02 lexical
    nomination budget (identifiers first, then content terms by ascending
    eligible df); ``nominate_df_theta`` arms the optional ``df > θ·N_E``
    nomination exclusion (``None`` = off — the r0 default)."""

    policy_id: str  # e.g. "retrieval_policy/v7" (provisional until O9)
    profile: str
    lanes: tuple[LaneName, ...]
    lane_weights: dict[IntentClass, dict[LaneName, float]]
    formula_status: str = FORMULA_STATUS_PROVISIONAL
    nominate_terms_max: int = NOMINATE_TERMS_MAX_R0
    nominate_df_theta: Optional[float] = None


# ---------------------------------------------------------------------------
# Verdict / packs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupVerdict:
    """Support verdict for one delivered group (V7-11.01)."""

    group_key: str
    label: SupportLabel
    trigger: Optional[str] = None  # (a) identifier (b) empty (c) calibrated (d) negative-evidence
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MissingDescriptor:
    """Why a result abstained (V7-11.08): which facets found no support."""

    facets: dict[str, tuple[str, ...]]  # facet kind -> unsupported values
    note: str = ""


@dataclass(frozen=True)
class ComputedItem:
    """A computed answer in a pack (V7-12.09, V7-31.03). Every ``inputs`` ref
    must appear in delivered items or context."""

    kind: str  # timeline|count|current_value|order|duration
    text: str
    inputs: tuple[str, ...]
    formula: str


@dataclass
class PackItemV7:
    """One delivered pack item (V7-12.07). Missing fields stay None."""

    ref: str
    unit_id: str
    quote: bytes
    speaker: Optional[str] = None
    recorded_at: Optional[str] = None
    occurred: Optional[IntervalUs] = None
    session: Optional[dict[str, Any]] = None
    lifecycle: LifecycleLabel = LifecycleLabel.CURRENT
    support: SupportLabel = SupportLabel.SUPPORTED
    perspective: Optional[Perspective] = None
    derived: bool = False
    proof_count: int = 0
    context: list["PackItemV7"] = field(default_factory=list)
    pins: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Entities / events / state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityMention:
    canon: str
    surface: str
    unit_id: str
    byte_start: int
    byte_end: int
    role: MentionRole = MentionRole.MENTION


@dataclass(frozen=True)
class AliasRow:
    """One `entity_aliases` row (§30). ``rule_id`` ∈ A1–A6 (§32.7)."""

    scope_id: str
    canon: str
    alias_canon: str
    rule_id: str
    evidence_count: int
    method: AliasMethod
    state: AliasState
    generation: int


@dataclass(frozen=True)
class EventTuple:
    """One `event/v1` record (V7-09.08). Every element is span-pinned;
    ``subject_canon=None`` means the sieve abstained (subject=unknown)."""

    unit_id: str
    subject_canon: Optional[str]
    predicate_lemma: str
    object_text: str
    polarity: str  # "affirm" | "negate" | ... (V5 polarity vocabulary)
    occurred: IntervalUs
    pins: dict[str, Any]
    rule_id: str


@dataclass(frozen=True)
class StateFact:
    scope_id: str
    state_key: str
    unit_id: str
    value_text: str
    value_norm: str
    valid_from_us: Optional[int]
    valid_to_us: Optional[int]
    status: StateFactStatus
    producer: str
    pins: dict[str, Any]


@dataclass(frozen=True)
class PreferenceFact:
    scope_id: str
    subject_canon: str
    unit_id: str
    object_text: str
    polarity: str
    strength: str  # constraint|favorite|love_hate|like_dislike|habitual (§32.12)
    occurred: IntervalUs
    pins: dict[str, Any]


# ---------------------------------------------------------------------------
# Stage profile (§32.17)
# ---------------------------------------------------------------------------

SEARCH_STAGE_FIELDS: tuple[str, ...] = (
    "t_total", "t_barrier", "t_analyze", "t_union", "t_rrf",
    "t_rerank_feat", "t_rerank_ce", "t_boost", "t_verdict", "t_pack",
    "t_post", "sql_statements", "snapshots", "bytes_read", "pool_R",
    "items", "tokens", "status", "coverage_digest",
)

ADD_STAGE_FIELDS: tuple[str, ...] = (
    "t_ack", "t_screen", "t_tx", "t_enqueue", "t_visible", "t_t0",
)


@dataclass
class StageRecord:
    """One search/add stage-profile record (§32.17). ``t_lane`` keys are lane
    names; ``candidates`` maps lane -> produced count."""

    kind: str  # "search" | "add"
    fields: dict[str, float] = field(default_factory=dict)
    t_lane: dict[str, float] = field(default_factory=dict)
    candidates: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Coverage (V7-31.01)
# ---------------------------------------------------------------------------


@dataclass
class CoverageV7:
    """The `coverage` block on every SearchResult. All sub-blocks are plain
    JSON-able dicts; absent capability is a named block, never omitted."""

    policy: dict[str, Any] = field(default_factory=dict)
    lanes: dict[str, dict[str, Any]] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    temporal: dict[str, Any] = field(default_factory=dict)
    entities: dict[str, Any] = field(default_factory=dict)
    facets: dict[str, Any] = field(default_factory=dict)
    rerank: dict[str, Any] = field(default_factory=dict)
    dense: dict[str, Any] = field(default_factory=dict)
    security: dict[str, Any] = field(default_factory=dict)
    migration: dict[str, Any] = field(default_factory=dict)

    def lane(self, name: str, status: LaneStatus, reason: Optional[str] = None,
             **extra: Any) -> None:
        entry: dict[str, Any] = {"status": status.value}
        if reason:
            entry["reason"] = reason
        entry.update(extra)
        self.lanes[name] = entry


__all__ = [name for name in dir() if not name.startswith("_")]
