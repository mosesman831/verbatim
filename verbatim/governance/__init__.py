"""Governance layer (SPEC_V3 §08–§11): principals, perspectives, purpose
registry, effective-verb grants, consent surfaces, epochs, propagation
ledger, and revocation.

Frozen worker contract (docs/v3_contracts.md): ``CallerV3``, ``authorize``,
``effective_verbs``, and ``record_propagation`` are the shared signatures
other v3 modules code against. Everything here takes the caller's
transaction ``conn`` — dict-snapshot reads via ``repos_v3``, parameterized
SQL only on the non-``repos_v3`` tables (``scopes``, ``consents``,
``handoff_capsules``). This package deliberately imports nothing from
security/vault/retrieval/procedures — those modules may import governance,
never the reverse.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional

from ..core.types_v3 import Propagation
from . import consent as _consent
from . import epochs as _epochs
from . import grants as _grants
from . import propagation as _propagation
from .consent import (
    capture_authorized,
    consent_for_hydration,
    get_capture_authorization,
    issue_capture_authorization,
    require_capture_authorization,
    require_hydration_consent,
    revoke_capture_authorization,
)
from .epochs import (
    bump_epoch,
    bump_epoch_if_present,
    check_epoch,
    current_epoch,
)
from .grants import (
    create_grant,
    delegate_grant,
    get_grant,
    grant_object,
    grant_purpose_constraint,
    grants_for,
    legacy_purpose_grants,
)
from .handlers import handle_revocation_notify
from .perspectives import (
    audience_of,
    create_perspective,
    find_perspective,
    get_or_create_perspective,
    perspectives_for_scope,
    resolve_perspective,
    subjects_of,
)
from .principals import (
    get_principal,
    is_retired,
    list_principals,
    register_principal,
    retire_principal,
)
from .propagation import (
    acknowledge_propagation,
    blast_radius,
    mark_propagations_revoked,
    propagations_for,
    recipients_of,
)
from .purposes import (
    BUILTIN_PURPOSES,
    is_registered,
    list_purposes,
    register_purpose,
    require_purpose,
    retire_purpose,
    seed_purposes,
)
from .revocation import revoke_delegation, revoke_grant


@dataclass(frozen=True)
class CallerV3:
    """A bound caller (§08.01): the principal is authenticated by the host
    outside request parameters — claimed actor names never establish it.
    ``epoch`` pins the scope's ``authz_revision``; ``None`` binds current.
    """

    principal_id: str
    session_id: str = ""
    host_id: str = ""
    epoch: Optional[int] = None  # pinned authz epoch; None = current


def authorize(
    conn: sqlite3.Connection,
    caller: CallerV3,
    scope_id: str,
    verb: str,
    purpose: Optional[str] = None,
    object_ref: Optional[tuple[str, str, int]] = None,
) -> None:
    """Raise NOT_FOUND_OR_UNAUTHORIZED when the caller lacks an effective
    grant for (scope, verb[, purpose]) at the caller's epoch. Never reveals
    existence vs authorization (§10.05). Returns None on success."""
    return _grants.authorize(
        conn, caller, scope_id, verb, purpose=purpose, object_ref=object_ref
    )


def effective_verbs(
    conn: sqlite3.Connection, caller: CallerV3, scope_id: str
) -> frozenset:
    """Union of verbs over the caller's currently effective grants."""
    return _grants.effective_verbs(conn, caller, scope_id)


def record_propagation(
    conn: sqlite3.Connection, propagation: Propagation
) -> str:
    """Append a disclosure record to the propagation ledger (§10.04)."""
    return _propagation.record_propagation(conn, propagation)


__all__ = [
    # frozen contract
    "CallerV3",
    "authorize",
    "effective_verbs",
    "record_propagation",
    # principals
    "register_principal",
    "get_principal",
    "is_retired",
    "list_principals",
    "retire_principal",
    # perspectives
    "create_perspective",
    "resolve_perspective",
    "find_perspective",
    "get_or_create_perspective",
    "perspectives_for_scope",
    "subjects_of",
    "audience_of",
    # purposes
    "BUILTIN_PURPOSES",
    "seed_purposes",
    "register_purpose",
    "retire_purpose",
    "is_registered",
    "require_purpose",
    "list_purposes",
    # grants
    "create_grant",
    "delegate_grant",
    "get_grant",
    "grants_for",
    "grant_object",
    "grant_purpose_constraint",
    "legacy_purpose_grants",
    # consent
    "issue_capture_authorization",
    "get_capture_authorization",
    "capture_authorized",
    "require_capture_authorization",
    "revoke_capture_authorization",
    "consent_for_hydration",
    "require_hydration_consent",
    # epochs
    "current_epoch",
    "bump_epoch",
    "bump_epoch_if_present",
    "check_epoch",
    # propagation
    "propagations_for",
    "blast_radius",
    "recipients_of",
    "mark_propagations_revoked",
    "acknowledge_propagation",
    # revocation
    "revoke_grant",
    "revoke_delegation",
    # handlers
    "handle_revocation_notify",
]
