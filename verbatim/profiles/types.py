"""Profile module types (SPEC_V4 §21).

These are module-local contracts, not kernel-level frozen types — the
profile plane consumes the frozen ``CallerV3``/``Verb``/``Perspective``
contracts and never redefines them.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional


class EntryKind(str, enum.Enum):
    """How a profile entry came to exist (V4-21.02 — the labels are
    distinguishable, never conflated)."""

    EXPLICIT = "explicit"        # user-declared, scoped, dated
    OBSERVED = "observed"        # recurring observed pattern, confidence-qualified
    INFERRED = "inferred"        # generated hypothesis — opt-in, reversible
    TASK_LOCAL = "task_local"    # session/task-bound; expires with task policy
    SENSITIVE = "sensitive"      # restricted attribute; separate consent+retention


class EntryState(str, enum.Enum):
    """Lifecycle of a profile entry head revision."""

    ACTIVE = "active"            # currently believed, servable
    CONFLICTED = "conflicted"    # live member of an open contradiction (servable)
    SUPERSEDED = "superseded"    # replaced by a newer revision/entry
    WITHHELD = "withheld"        # support invalidated — purged/held/quarantined
    TOMBSTONED = "tombstoned"    # owner-deleted or fully unsupported; terminal
    EXPIRED = "expired"          # task-local past its expiry bound


#: States an ordinary read may serve.
SERVABLE_STATES = frozenset({EntryState.ACTIVE.value, EntryState.CONFLICTED.value})


class Multiplicity(str, enum.Enum):
    SINGLE = "single"    # one value per subject — distinct values conflict
    SET = "set"          # legitimately multi-valued — members coexist


class Sensitivity(str, enum.Enum):
    NORMAL = "normal"
    SENSITIVE = "sensitive"


class ConflictPolicy(str, enum.Enum):
    """V4-21.01 conflict handling — declared per topic, never implicit."""

    KEEP_BOTH = "keep_both"        # retain every alternative, mark the group
    LATEST_WINS = "latest_wins"    # newest effective_us supersedes the rest
    EXPLICIT_ONLY = "explicit_only"  # only explicit declarations may write


class InferenceMode(str, enum.Enum):
    OFF = "off"   # default — sensitive inference never runs (V4-21.04)
    ON = "on"     # permitted purpose + owner policy both required


#: Relations a support row may carry into the derivation graph.
SUPPORT_RELATIONS = frozenset({"supports", "contradicts"})

#: The derivation-graph child kind. ``profile`` is the declared learning
#: kind (verbatim/derivations.py LEARNING_KINDS); entry identity lives in
#: child_id. No second kind is invented — undeclared kinds are rejected
#: by the graph contract (V3-17.01).
PROFILE_KIND = "profile"

#: Producer identities recorded in derivations/dependency_edges and the
#: producer_manifests registry.
PRODUCER_COMPILE = "profile-compiler:rules:v1"
PRODUCER_UPSERT = "profile-upsert:v1"
PRODUCER_REFRESH = "profile-refresh:v1"

#: Bounded scan/fan-out limits — every cap is reported, never silent.
MAX_COMPILE_CLAIMS = 2048
MAX_SUPPORT_REFS = 64
MAX_ENTRIES_PER_READ = 512


@dataclass(frozen=True)
class TopicSpec:
    """A configured profile topic (V4-21.01).

    ``match`` is the deterministic derivation surface: ``keywords``
    matched against claim predicate + object text, ``predicates`` as an
    exact predicate allowlist, ``subjects`` restricting which claim
    subjects feed the topic.
    """

    topic_key: str
    value_kind: str = "freeform"
    multiplicity: str = Multiplicity.SINGLE.value
    sensitivity: str = Sensitivity.NORMAL.value
    preferred_evidence: tuple[str, ...] = ()
    required_purposes: tuple[str, ...] = ()
    inference: str = InferenceMode.OFF.value
    inference_purposes: tuple[str, ...] = ()
    conflict_policy: str = ConflictPolicy.KEEP_BOTH.value
    expiry_s: Optional[int] = None
    max_entries: int = 8
    min_support: int = 1
    match: dict[str, Any] = field(default_factory=dict)
    state: str = "active"


@dataclass(frozen=True)
class PerspectivePack:
    """A declared retrieval perspective (V4-21.03).

    The pack *narrows and ranks* a caller-supplied candidate set under the
    caller's own grants — attenuation only, never widening. It carries no
    authority of its own: ``verbs`` are re-verified per application, and a
    pack without an authorized ``read`` evaluates to nothing.
    """

    perspective_id: str
    scope_id: str
    caller_id: str
    subjects: tuple[str, ...] = ()
    audience: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    verbs: tuple[str, ...] = ("read", "quote")
    purpose: str = "recall"
    persisted: bool = False
    issued_us: int = 0
    expires_us: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "perspective_id": self.perspective_id,
            "scope_id": self.scope_id,
            "caller_id": self.caller_id,
            "subjects": list(self.subjects),
            "audience": list(self.audience),
            "topics": list(self.topics),
            "verbs": list(self.verbs),
            "purpose": self.purpose,
            "persisted": self.persisted,
            "issued_us": self.issued_us,
            "expires_us": self.expires_us,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PerspectivePack":
        return cls(
            perspective_id=str(d["perspective_id"]),
            scope_id=str(d["scope_id"]),
            caller_id=str(d["caller_id"]),
            subjects=tuple(d.get("subjects") or ()),
            audience=tuple(d.get("audience") or ()),
            topics=tuple(d.get("topics") or ()),
            verbs=tuple(d.get("verbs") or ("read", "quote")),
            purpose=str(d.get("purpose") or "recall"),
            persisted=bool(d.get("persisted", False)),
            issued_us=int(d.get("issued_us") or 0),
            expires_us=int(d.get("expires_us") or 0),
        )
