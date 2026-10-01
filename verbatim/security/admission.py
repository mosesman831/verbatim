"""Admission ladder (SPEC_V3 §14): source_trust → review/promotion defaults.

The §14 trust table maps provenance classes to *promotion defaults* —
retained, interpreted, admitted, promoted, and shared are distinct
milestones, not one truth ladder (V3-14.03). This module computes two
policy inputs from a screening verdict:

* ``default_review_state`` — the label's ``review_state`` dimension when a
  screen commits. Blocked content lands ``quarantined``; suspicious or
  unresolved findings land ``pending``; a clean screen on trusted provenance
  needs no security review.
* ``auto_promotion_allowed`` — §14.04: unknown provenance, suspicious
  findings, blocked content, external/agent/imported origins, or an
  unscreened item can never promote automatically. Only
  ``principal_direct`` / ``principal_reported`` / ``host_observed`` items
  with a clean screen are *eligible* — eligibility is not admission.

``agent_generated`` is derive-only by default: its promotion requires an
observed outcome or review (§14 table), never the screen result alone.
"""

from __future__ import annotations

from typing import Any, Optional

from ..core.types import ErrorCode, VerbatimError
from ..core.types_v3 import (
    AttackRisk,
    ContentForm,
    SecurityReviewState,
    TrustClass,
)

#: §14 trust-table promotion defaults, verbatim.
PROMOTION_DEFAULTS: dict[str, str] = {
    TrustClass.PRINCIPAL_DIRECT.value: "eligible",
    TrustClass.PRINCIPAL_REPORTED.value: "eligible_attributed",
    TrustClass.HOST_OBSERVED.value: "experience",
    TrustClass.EXTERNAL_CONTENT.value: "evidence_only",
    TrustClass.AGENT_GENERATED.value: "derive_only",
    TrustClass.IMPORTED.value: "capped",
    TrustClass.UNKNOWN.value: "never",
}

#: Origins eligible for *automatic* promotion when the screen is clean —
#: every other class requires review or a verified outcome (§14.04).
_AUTO_PROMOTABLE = frozenset(
    {
        TrustClass.PRINCIPAL_DIRECT.value,
        TrustClass.PRINCIPAL_REPORTED.value,
        TrustClass.HOST_OBSERVED.value,
    }
)


def _trust(value: Any) -> TrustClass:
    try:
        return value if isinstance(value, TrustClass) else TrustClass(value)
    except ValueError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown source_trust {value!r}"
        ) from exc


def _risk(value: Any) -> AttackRisk:
    try:
        return value if isinstance(value, AttackRisk) else AttackRisk(value)
    except ValueError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown attack_risk {value!r}"
        ) from exc


def _review(value: Any) -> SecurityReviewState:
    try:
        return (
            value
            if isinstance(value, SecurityReviewState)
            else SecurityReviewState(value)
        )
    except ValueError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown review_state {value!r}"
        ) from exc


def promotion_default(source_trust: Any) -> str:
    """The §14 table's promotion default for a provenance class."""
    return PROMOTION_DEFAULTS[_trust(source_trust).value]


def default_review_state(
    source_trust: Any,
    *,
    attack_risk: Any = AttackRisk.UNASSESSED,
    content_form: Any = ContentForm.UNKNOWN,
    findings: Optional[list[dict[str, Any]]] = None,
) -> str:
    """Review-state default when a screen result is committed (§14.01/§14.04).

    * ``blocked`` content → ``quarantined`` (stage-1 control, §34).
    * ``suspicious`` or any recorded findings → ``pending``.
    * ``unknown`` provenance → ``pending`` (quarantine-eligible; never
      promoted automatically).
    * ``external_content`` still ``unassessed`` → ``pending`` — promotion
      requires screening first.
    * otherwise → ``not_required``.
    """
    trust = _trust(source_trust)
    risk = _risk(attack_risk)
    if content_form is not None:
        ContentForm(content_form)  # validate only
    open_findings = bool(findings)
    if risk == AttackRisk.BLOCKED:
        return SecurityReviewState.QUARANTINED.value
    if risk == AttackRisk.SUSPICIOUS or open_findings:
        return SecurityReviewState.PENDING.value
    if trust == TrustClass.UNKNOWN:
        return SecurityReviewState.PENDING.value
    if trust == TrustClass.EXTERNAL_CONTENT and risk == AttackRisk.UNASSESSED:
        return SecurityReviewState.PENDING.value
    return SecurityReviewState.NOT_REQUIRED.value


def auto_promotion_allowed(
    source_trust: Any,
    *,
    attack_risk: Any = AttackRisk.NO_FINDINGS,
    review_state: Any = SecurityReviewState.NOT_REQUIRED,
) -> bool:
    """§14.04: True only for clean-screened principal/host origins.

    ``unassessed`` blocks promotion — screening is a precondition, and a
    skipped screen is not a clean screen. ``agent_generated`` (derive-only),
    ``external_content`` (needs review or verified outcome), ``imported``
    (capped), and ``unknown`` never promote automatically.
    """
    trust = _trust(source_trust)
    risk = _risk(attack_risk)
    state = _review(review_state)
    if trust not in _AUTO_PROMOTABLE:
        return False
    if risk != AttackRisk.NO_FINDINGS:
        return False
    return state in (
        SecurityReviewState.NOT_REQUIRED,
        SecurityReviewState.RELEASED,
    )


__all__ = [
    "PROMOTION_DEFAULTS",
    "auto_promotion_allowed",
    "default_review_state",
    "promotion_default",
]
