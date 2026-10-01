"""Security layer: instruction/evidence classification, screening labels,
quarantine, and the admission ladder (SPEC_V3 §14, §31, §34).

Frozen contract surface (docs/v3_contracts.md): ``attach_label`` /
``is_quarantined`` / ``label_for`` — plus the rules_v1 screener
(``screen_content`` → :class:`SecurityVerdict`), the quarantine workflow,
and job handlers registered by ``ingest._V3_KIND_HANDLERS``.

Importable standalone: nothing here touches governance, vault, or evidence
modules; authorization decisions are governance's job — this package only
records and enforces the metadata.
"""

from .admission import (
    PROMOTION_DEFAULTS,
    auto_promotion_allowed,
    default_review_state,
    promotion_default,
)
from .labels import attach_label, label_for, update_review_state
from .quarantine import (
    get_quarantine,
    is_quarantined,
    mark_purged,
    open_quarantine,
    pending_items,
    release,
    should_exclude,
    suppress,
)
from .screening import RULES_REVISION, SecurityVerdict, screen_content

__all__ = [
    "PROMOTION_DEFAULTS",
    "RULES_REVISION",
    "SecurityVerdict",
    "attach_label",
    "auto_promotion_allowed",
    "default_review_state",
    "get_quarantine",
    "is_quarantined",
    "label_for",
    "mark_purged",
    "open_quarantine",
    "pending_items",
    "promotion_default",
    "release",
    "screen_content",
    "should_exclude",
    "suppress",
    "update_review_state",
]
