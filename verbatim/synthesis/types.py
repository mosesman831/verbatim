"""Synthesis contracts (SPEC_V4 §20, V4-20.01–V4-20.10).

These are the module-local frozen types for ``verbatim.synthesis`` — the
same convention ``kernel/service.py`` uses for ``DerivedInputs`` /
``InvalidationReport``: contracts that only this module produces live
beside their producer rather than in ``core/types_v4.py``.

Design invariants baked into the types:

- A ``DerivedView`` NEVER impersonates canonical evidence (V4-20.01/02):
  every ``ViewStatement`` carries ``support`` bindings naming the exact
  evidence refs (``kind:id@rev`` plus byte range where applicable) that
  ground it, and the view is honestly labeled
  ``synthesis_mode="grounded_composition"``.
- Confidence is a three-valued honest tag: ``supported`` (every bound ref
  verified at the level the statement form requires), ``partial`` (some
  bound refs downgraded/withheld), and ``unsupported`` — which is never
  persisted on a statement because unsupported propositions are omitted
  from the view entirely (V4-20.05, C47).
- Persisted views contain NO canonical bytes: quote statements persist
  only locators + slice digests; their text is re-materialized at delivery
  through the kernel under the *delivering* caller's ``quote`` verb.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.types import ErrorCode, VerbatimError, require_id
from ..core.types_v4 import PurposeConstraint

#: The only synthesis mode this build implements — deterministic grounded
#: composition over kernel-verified slices. No neural producer exists in
#: this environment; requesting one is a typed CAPABILITY_UNAVAILABLE.
SYNTHESIS_MODE = "grounded_composition"

#: Registered ``objects``/``object_revisions`` kind for persisted views.
VIEW_OBJECT_KIND = "derived_view"

#: ``view_support.view_kind`` value for rows this module writes.
VIEW_SUPPORT_KIND = "derived_view"


class ViewKind(str, enum.Enum):
    """View kinds from the §20 table this build can honestly produce.

    ``GROUNDED_SUMMARY``, ``HYPOTHESIS``, and ``USER_EDITED`` require a
    neural/interactive producer that does not exist here — they are
    declared so callers get a typed ``CAPABILITY_UNAVAILABLE`` instead of
    a silently degraded view.
    """

    EXTRACTIVE_DIGEST = "extractive_digest"      # exact selected spans
    TYPED_SUMMARY = "typed_summary"              # deterministic field rendering
    GROUNDED_SUMMARY = "grounded_summary"        # needs a neural producer
    HYPOTHESIS = "hypothesis"                    # needs a neural producer
    USER_EDITED = "user_edited"                  # needs owner review flow


#: View kinds this deterministic composer can actually emit.
SUPPORTED_VIEW_KINDS = frozenset(
    {ViewKind.EXTRACTIVE_DIGEST.value, ViewKind.TYPED_SUMMARY.value}
)


class StatementForm(str, enum.Enum):
    """How a statement's content was produced."""

    QUOTE = "quote"    # verbatim byte slice of canonical evidence
    FIELD = "field"    # deterministic rendering of object metadata


class Confidence(str, enum.Enum):
    """Grounding verdict for one statement (V4-20.03/05)."""

    SUPPORTED = "supported"    # every bound ref verified at the needed level
    PARTIAL = "partial"        # ≥1 bound ref downgraded or withheld
    # UNSUPPORTED is deliberately absent: unsupported propositions are
    # omitted from the view, never shipped with a label (V4-20.05).


#: Per-ref verification verdict persisted in ``view_support.verdict``.
class SupportVerdict(str, enum.Enum):
    VERIFIED = "verified"              # slice bytes digest-authenticated
    ATTESTED = "attested"              # object-level check passed (no bytes)
    METADATA_ONLY = "metadata_only"    # object checked; bytes not read
    LEGACY_UNVERIFIED = "legacy_unverified"  # bytes readable, no stored digest
    WITHHELD = "withheld"              # bound at compose but denied/unavailable


@dataclass(frozen=True)
class SupportBinding:
    """One evidence ref grounding a statement (V4-20.03).

    ``start_byte``/``end_byte``/``view_id`` locate the exact slice when the
    support is byte-level; ``verdict`` records how strongly it was
    authenticated at composition time.
    """

    evidence_kind: str
    evidence_id: str
    evidence_revision: int
    verdict: str = SupportVerdict.ATTESTED.value
    start_byte: Optional[int] = None
    end_byte: Optional[int] = None
    view_id: Optional[str] = None

    def __post_init__(self) -> None:
        require_id(self.evidence_kind, "evidence_kind")
        require_id(self.evidence_id, "evidence_id")
        if (
            isinstance(self.evidence_revision, bool)
            or not isinstance(self.evidence_revision, int)
            or self.evidence_revision < 0
        ):
            raise VerbatimError(
                ErrorCode.VALIDATION, "evidence_revision must be an int >= 0"
            )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.evidence_kind,
            "id": self.evidence_id,
            "rev": self.evidence_revision,
            "verdict": self.verdict,
        }
        if self.start_byte is not None:
            out["start_byte"] = self.start_byte
            out["end_byte"] = self.end_byte
        if self.view_id is not None:
            out["view_id"] = self.view_id
        return out

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "SupportBinding":
        return cls(
            evidence_kind=raw["kind"],
            evidence_id=raw["id"],
            evidence_revision=int(raw["rev"]),
            verdict=raw.get("verdict", SupportVerdict.ATTESTED.value),
            start_byte=raw.get("start_byte"),
            end_byte=raw.get("end_byte"),
            view_id=raw.get("view_id"),
        )


@dataclass(frozen=True)
class ViewStatement:
    """One delivered proposition inside a derived view.

    ``text`` is populated only for statements the receiving caller may see
    verbatim: field statements render metadata (always allowed) while quote
    statements carry canonical bytes and ship text only under the caller's
    ``quote`` verb (V4-08.02). ``locator`` names the exact byte range a
    quote statement re-reads at delivery.
    """

    statement_id: str
    form: str                         # StatementForm value
    confidence: str                   # Confidence value
    support: tuple[SupportBinding, ...]
    text: Optional[str] = None        # None = metadata-only delivery
    section: str = "evidence"
    seq: int = 0
    locator: Optional[dict[str, Any]] = None   # quote re-read coordinates
    evidence_digest: Optional[str] = None    # slice digest bound at mint
    screen: str = "no_findings"       # independent output-screen verdict

    def to_json(self) -> dict[str, Any]:
        """Persisted form — NEVER carries canonical bytes (quote ``text``
        is always persisted as None; field renderings are metadata)."""
        return {
            "statement_id": self.statement_id,
            "form": self.form,
            "confidence": self.confidence,
            "section": self.section,
            "seq": self.seq,
            "text": self.text if self.form == StatementForm.FIELD.value else None,
            "locator": self.locator,
            "evidence_digest": self.evidence_digest,
            "support": [s.to_json() for s in self.support],
            "screen": self.screen,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "ViewStatement":
        return cls(
            statement_id=raw["statement_id"],
            form=raw["form"],
            confidence=raw["confidence"],
            support=tuple(
                SupportBinding.from_json(s) for s in raw.get("support") or ()
            ),
            text=raw.get("text"),
            section=raw.get("section", "evidence"),
            seq=int(raw.get("seq", 0)),
            locator=raw.get("locator"),
            evidence_digest=raw.get("evidence_digest"),
            screen=raw.get("screen", "no_findings"),
        )


@dataclass(frozen=True)
class DerivedView:
    """``compose``/``deliver`` output: a labeled, support-bound derivative.

    ``inputs_digests`` pins the exact verified inputs (V4-20.02);
    ``allowed_purposes``/``effective_audience`` carry the inherited
    restrictions computed by ``kernel.derive_inputs`` (V4-10.07/20.06);
    ``omitted`` honestly counts propositions that never shipped.
    """

    view_id: str
    revision: int
    scope_id: str
    view_kind: str
    producer_id: str
    statements: tuple[ViewStatement, ...]
    inputs_digests: tuple[str, ...]
    output_digest: str
    epoch_vector: dict[str, int] = field(default_factory=dict)
    allowed_purposes: PurposeConstraint = field(
        default_factory=PurposeConstraint.any
    )
    effective_audience: tuple[str, ...] = ()
    contributing_scopes: tuple[str, ...] = ()
    omitted: dict[str, int] = field(default_factory=dict)
    created_us: int = 0
    persisted: bool = True
    stale: bool = False
    synthesis_mode: str = SYNTHESIS_MODE
    request_digest: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "revision": self.revision,
            "scope_id": self.scope_id,
            "view_kind": self.view_kind,
            "synthesis_mode": self.synthesis_mode,
            "producer_id": self.producer_id,
            "request_digest": self.request_digest,
            "contributing_scopes": list(self.contributing_scopes),
            "epoch_vector": dict(self.epoch_vector),
            "allowed_purposes": self.allowed_purposes.to_json(),
            "effective_audience": list(self.effective_audience),
            "inputs_digests": list(self.inputs_digests),
            "output_digest": self.output_digest,
            "omitted": dict(self.omitted),
            "statements": [s.to_json() for s in self.statements],
            "created_us": self.created_us,
            "stale": self.stale,
        }


__all__ = [
    "Confidence",
    "DerivedView",
    "StatementForm",
    "SupportBinding",
    "SupportVerdict",
    "SYNTHESIS_MODE",
    "SUPPORTED_VIEW_KINDS",
    "VIEW_OBJECT_KIND",
    "VIEW_SUPPORT_KIND",
    "ViewKind",
    "ViewStatement",
]
