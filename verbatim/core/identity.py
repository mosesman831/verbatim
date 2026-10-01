"""Scope resolution and access predicates (SPEC §9).

A scope is the authorization partition. Predicates here are pure functions —
repositories apply them before candidate generation, never after.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from .types import Scope, Visibility


def scope_key(scope: Scope) -> str:
    """Deterministic partition key for a Scope — the `scopes.scope_id` value.

    Engine-generated, stable across processes (sha256, not builtin hash),
    and carries no raw user text. One scopes row per distinct partition tuple.
    """
    parts = "|".join(
        [
            scope.profile_id,
            scope.principal_id or "",
            scope.workspace_id or "",
            scope.conversation_id or "",
            scope.visibility.value,
        ]
    )
    return "s" + hashlib.sha256(parts.encode()).hexdigest()[:31]


def scope_from_row(row) -> Scope:
    """Rebuild a Scope from a scopes-table row (dict-like or sqlite.Row)."""
    return Scope(
        profile_id=row["profile_id"],
        principal_id=row["principal_id"],
        workspace_id=row["workspace_id"],
        conversation_id=row["conversation_id"],
        visibility=Visibility(row["visibility"]),
    )


def can_read(reader: Scope, owner: Scope) -> bool:
    """Whether a principal operating in ``reader`` may see evidence in ``owner``.

    - conversation visibility: same profile + same conversation
    - workspace visibility: same profile + same workspace + same principal
      or conversation membership (simplified: same principal in v1)
    - owner visibility: same profile + same principal
    A missing principal never widens access (SPEC §9).
    """
    if reader.profile_id != owner.profile_id:
        return False
    v = owner.visibility
    if v == Visibility.CONVERSATION:
        return (
            owner.conversation_id is not None
            and reader.conversation_id == owner.conversation_id
        )
    if v == Visibility.WORKSPACE:
        return (
            owner.workspace_id is not None
            and reader.workspace_id == owner.workspace_id
            and reader.principal_id is not None
            and reader.principal_id == owner.principal_id
        )
    # owner visibility
    return (
        reader.principal_id is not None
        and reader.principal_id == owner.principal_id
    )


def can_write(reader: Scope, owner: Scope) -> bool:
    """Writes are allowed only within the caller's own scope partition."""
    return (
        reader.profile_id == owner.profile_id
        and reader.principal_id == owner.principal_id
        and reader.workspace_id == owner.workspace_id
        and reader.conversation_id == owner.conversation_id
    )


def may_promote(from_vis: Visibility, to_vis: Visibility) -> bool:
    """Promotion to wider visibility requires explicit authority (SPEC §9).

    Ordering: conversation < workspace < owner is *narrowing*; going the other
    direction is promotion and must be operator-approved (callers enforce).
    """
    order = (Visibility.CONVERSATION, Visibility.WORKSPACE, Visibility.OWNER)
    return order.index(to_vis) <= order.index(from_vis)


def conversation_scope(
    profile_id: str,
    conversation_id: str,
    principal_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> Scope:
    """Default capture scope: missing identity stays conversation-private."""
    return Scope(
        profile_id=profile_id,
        principal_id=principal_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        visibility=Visibility.CONVERSATION,
    )
