"""Engine sibling — bounded evidence recall, inspection, and feedback
(V3-06.01). Methods are verbatim moves from the former ``api.py`` god
file; the facade composes this mixin."""

from __future__ import annotations

from typing import Any, Optional

from .core.types import (
    CallerContext,
    GrantKind,
    RecallRequest,
    RecallResult,
    Scope,
)


class RecallMixin:
    """Read-path topic: recall, claim inspection, usefulness feedback."""

    def recall(
        self, request: RecallRequest, *, caller: Optional[CallerContext] = None
    ) -> RecallResult:
        """Bounded evidence recall under the request's authorized scope."""
        self._require_open()
        c = self._resolve_caller(caller, request.scope)
        self._require_grant(c, GrantKind.READ_EVIDENCE)
        self._authorize_read(c, request.scope)
        from .retrieval import search

        return search(self.store, request)

    def inspect(
        self, claim_id: str, scope: Scope, *, caller: Optional[CallerContext] = None
    ) -> dict[str, Any]:
        """Evidence + interpretation lineage for one claim (authorized only)."""
        self._require_open()
        c = self._resolve_caller(caller, scope)
        self._require_grant(c, GrantKind.READ_EVIDENCE)
        self._authorize_read(c, scope)
        from .retrieval.inspect import inspect_claim

        return inspect_claim(self.store, claim_id, scope)

    def feedback(
        self,
        claim_id: str,
        kind: Any,
        scope: Scope,
        actor_id: str = "agent",
        *,
        caller: Optional[CallerContext] = None,
    ) -> None:
        """Record usefulness feedback — never a truth judgment or deletion (SPEC §35)."""
        self._require_open()
        c = self._resolve_caller(caller, scope)
        self._require_grant(c, GrantKind.PROPOSE)
        self._authorize_read(c, scope)
        from .storage.repos import FeedbackRepo

        with self.store.tx() as conn:
            FeedbackRepo(self.store).add(conn, claim_id, actor_id, kind.value)


__all__ = ["RecallMixin"]
