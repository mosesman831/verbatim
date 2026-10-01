"""Engine sibling — purge/suppression, portability, and handoff
(V3-06.01; SPEC_V2 §41–§42).

Owns the privacy/portability topic: purge preview + erasure, reversible
suppression, scoped export/import bundles, and live-reference handoff
capsules. Methods are verbatim moves from the former ``api.py`` god
file; the facade composes this mixin.
"""

from __future__ import annotations

from typing import Any, Optional

from .core.types import (
    CallerContext,
    ErrorCode,
    GrantKind,
    Scope,
    VerbatimError,
)


class PrivacyMixin:
    """Purge/suppress/export/share topic (composed by the facade)."""

    def plan_purge(
        self,
        targets: Any,
        *,
        scope: Optional[Scope] = None,
        caller: Optional[CallerContext] = None,
        actor: Optional[str] = None,
    ) -> dict[str, Any]:
        """Preview an erasure: exact selection + dependency closure."""
        from .purge import plan_purge as _plan

        self._require_open()
        sc = scope or self.host.default_scope()
        c = self._resolve_caller(caller, sc)
        self._require_grant(c, GrantKind.PURGE)
        if not c.is_operator:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, "purge requires operator"
            )
        return _plan(self.store, sc, targets, actor or c.principal_id or "operator")

    def execute_purge(
        self,
        purge_id: str,
        *,
        scope: Optional[Scope] = None,
        caller: Optional[CallerContext] = None,
    ) -> dict[str, Any]:
        """Physical erasure for a previewed/suppressed purge (V2-41.03)."""
        from .purge import execute_purge as _exec

        self._require_open()
        sc = scope or self.host.default_scope()
        c = self._resolve_caller(caller, sc)
        self._require_grant(c, GrantKind.PURGE)
        if not c.is_operator:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, "purge requires operator"
            )
        return _exec(self.store, purge_id, actor=c.principal_id)

    def suppress(
        self,
        targets: Any,
        *,
        scope: Optional[Scope] = None,
        caller: Optional[CallerContext] = None,
        actor: Optional[str] = None,
    ) -> dict[str, Any]:
        """Reversible tombstoning — no byte deletion (V2-41.06)."""
        from .purge import suppress as _sup

        self._require_open()
        sc = scope or self.host.default_scope()
        c = self._resolve_caller(caller, sc)
        self._require_grant(c, GrantKind.SUPPRESS)
        return _sup(self.store, sc, targets, actor or c.principal_id or "operator")

    def lift_suppression(
        self,
        purge_id: str,
        *,
        scope: Optional[Scope] = None,
        caller: Optional[CallerContext] = None,
    ) -> dict[str, Any]:
        """Restore exactly the objects a suppression tombstoned."""
        from .purge import lift_suppression as _lift

        self._require_open()
        sc = scope or self.host.default_scope()
        c = self._resolve_caller(caller, sc)
        self._require_grant(c, GrantKind.SUPPRESS)
        return _lift(self.store, purge_id, c.principal_id or "operator")

    def export_scope(
        self,
        out_path: Optional[str] = None,
        *,
        portable: bool = True,
        scope: Optional[Scope] = None,
        caller: Optional[CallerContext] = None,
    ) -> dict[str, Any]:
        """Versioned evidence bundle; suppressed/erased excluded fail-closed."""
        from .export import export_scope as _export

        self._require_open()
        sc = scope or self.host.default_scope()
        c = self._resolve_caller(caller, sc)
        self._require_grant(c, GrantKind.EXPORT)
        return _export(self.store, sc, out_path, portable=portable, caller=c)

    def import_bundle(
        self,
        bundle: Any,
        *,
        scope: Optional[Scope] = None,
        actor: Optional[str] = None,
        caller: Optional[CallerContext] = None,
    ) -> dict[str, Any]:
        """Import a bundle with remapped ownership + erasure fencing."""
        from .export import import_bundle as _import

        self._require_open()
        sc = scope or self.host.default_scope()
        c = self._resolve_caller(caller, sc)
        self._require_grant(c, GrantKind.INGEST)
        return _import(self.store, None, bundle, sc,
                       actor or c.principal_id or "operator")

    def share(
        self,
        recipient_id: str,
        object_refs: Any,
        *,
        permission: str = "read_evidence",
        expires_us: Optional[int] = None,
        snapshot: Any = None,
        caller: Optional[CallerContext] = None,
    ) -> dict[str, Any]:
        """Create a live-reference handoff capsule (SHARE grant, V2-09.10)."""
        from .sharing import create_handoff

        self._require_open()
        c = self._resolve_caller(caller, self.host.default_scope())
        return create_handoff(
            self.store, c, recipient_id, object_refs,
            permission=permission, expires_us=expires_us, snapshot=snapshot,
        )

    def consume_handoff(
        self,
        capsule_id: str,
        *,
        mark_consumed: bool = True,
        caller: Optional[CallerContext] = None,
    ) -> dict[str, Any]:
        """Consume a capsule — recipient/issuer/operator only (V2-10.07)."""
        from .sharing import consume_handoff

        self._require_open()
        c = self._resolve_caller(caller, self.host.default_scope())
        return consume_handoff(self.store, c, capsule_id,
                               mark_consumed=mark_consumed)

    def revoke_handoff(
        self,
        capsule_id: str,
        *,
        caller: Optional[CallerContext] = None,
    ) -> dict[str, Any]:
        """Revoke a capsule — issuer/operator; terminal state."""
        from .sharing import revoke_handoff

        self._require_open()
        c = self._resolve_caller(caller, self.host.default_scope())
        return revoke_handoff(self.store, c, capsule_id)


__all__ = ["PrivacyMixin"]
