"""Security labels: multidimensional metadata on evidence and derived
objects (SPEC_V3 §14.01, §34.01).

A label row records four INDEPENDENT dimensions — ``source_trust``,
``content_form``, ``attack_risk``, ``review_state`` — plus the findings a
screen produced, the screening method, and the rules revision. They are
not one ordered taint scale: ``no_findings`` means no listed attack signal
matched, never certified safety, and origin never changes after review
(V3-14.01).

Object rows (sources, claims, procedures, …) carry ``security_label_id``
where their schema allows; ``attach_label`` returns the new ``label_id``
for the caller to store — the labels table deliberately has no back-pointer
so any object kind can reference it.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

from ..core.types import ErrorCode, VerbatimError, new_id, require_id
from ..core.types_v3 import (
    AttackRisk,
    ContentForm,
    SecurityLabel,
    SecurityReviewState,
    TrustClass,
)
from ..storage import repos_v3


def _next_event_seq(conn: sqlite3.Connection) -> int:
    """Estimated next event_seq for ``created_event`` (repos convention).

    Callers append the transition event inside the same transaction, so the
    estimate materializes as the real sequence number at commit.
    """
    row = conn.execute(
        "SELECT COALESCE(MAX(event_seq), 0) + 1 FROM events"
    ).fetchone()
    return int(row[0])


def _enum_value(value: Any, enum_cls: Any, field: str) -> str:
    try:
        return enum_cls(value).value
    except ValueError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"invalid {field}: {value!r}"
        ) from exc


def attach_label(
    conn: sqlite3.Connection,
    scope_id: str,
    *,
    source_trust: Any,
    content_form: Any = "unknown",
    attack_risk: Any = "unassessed",
    review_state: Any = "not_required",
    findings: Optional[list[dict[str, Any]]] = None,
    method: str = "rules",
    rules_revision: str = "",
) -> str:
    """Persist one security label; returns ``label_id`` (§14.01).

    All four dimensions are required-positioned independently — callers
    pass the enum member or its string value; unknown strings are rejected
    (VALIDATION) rather than stored, so a label can never carry a state
    outside the frozen vocabulary. ``findings`` is the screener's
    ``[{rule_id, span, excerpt}]`` list, persisted verbatim as JSON.
    """
    require_id(scope_id, "scope_id")
    label = SecurityLabel(
        source_trust=_enum_value(source_trust, TrustClass, "source_trust"),
        content_form=_enum_value(content_form, ContentForm, "content_form"),
        attack_risk=_enum_value(attack_risk, AttackRisk, "attack_risk"),
        review_state=_enum_value(
            review_state, SecurityReviewState, "review_state"
        ),
        findings=tuple(findings or ()),
        method=method,
        rules_revision=rules_revision,
    )
    if not isinstance(label.method, str) or not label.method:
        raise VerbatimError(ErrorCode.VALIDATION, "method must be non-empty")
    if not isinstance(label.rules_revision, str):
        raise VerbatimError(
            ErrorCode.VALIDATION, "rules_revision must be a string"
        )
    label_id = new_id()
    repos_v3.insert(
        conn,
        "security_labels",
        {
            "label_id": label_id,
            "scope_id": scope_id,
            "source_trust": label.source_trust.value,
            "content_form": label.content_form.value,
            "attack_risk": label.attack_risk.value,
            "review_state": label.review_state.value,
            "findings_json": list(label.findings),
            "method": label.method,
            "rules_revision": label.rules_revision,
            "created_event": _next_event_seq(conn),
        },
    )
    return label_id


def label_for(
    conn: sqlite3.Connection, label_id: str
) -> Optional[dict[str, Any]]:
    """Read one label row as a dict snapshot; ``None`` when absent.

    The returned dict carries the persisted columns plus a decoded
    ``findings`` list (``findings_json`` is kept too — raw and decoded are
    both useful to auditors).
    """
    require_id(label_id, "label_id")
    row = repos_v3.get(conn, "security_labels", {"label_id": label_id})
    if row is None:
        return None
    row["findings"] = repos_v3.json_field(row, "findings_json", [])
    return row


def update_review_state(
    conn: sqlite3.Connection, label_id: str, review_state: Any
) -> int:
    """Advance a label's ``review_state`` (review never rewrites origin,
    findings, or risk — V3-14.01: origin never changes after review)."""
    require_id(label_id, "label_id")
    state = _enum_value(review_state, SecurityReviewState, "review_state")
    return repos_v3.update(
        conn,
        "security_labels",
        {"review_state": state},
        {"label_id": label_id},
    )


__all__ = ["attach_label", "label_for", "update_review_state"]
