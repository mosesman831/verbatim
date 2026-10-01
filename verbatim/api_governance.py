"""Engine sibling — caller binding, authorization fencing, and grant
administration (V3-06.01).

Owns the governance topic (SPEC_V2 §09): bound-caller resolution,
authorization-epoch fencing, revoked-grant narrowing, operator grant/
revoke, and the read/write authorization helpers every topic sibling
calls through ``self`` on the composed facade.
"""

from __future__ import annotations

from typing import Any, Optional

from .core.identity import can_read, can_write, scope_key
from .core.types import (
    CallerContext,
    ErrorCode,
    GrantKind,
    Scope,
    VerbatimError,
    Visibility,
)


class GovernanceMixin:
    """Caller-binding + authorization topic (composed by the facade)."""

    # -- caller binding (SPEC_V2 §09) ----------------------------------------
    #
    # A bound CallerContext may only narrow authority — when a transport
    # supplies one, every operation checks it. When none is supplied the
    # trusted in-process boundary applies (V2-09.03): library/CLI callers act
    # with operator authority scoped to the partition they addressed, which
    # matches V2-09.17 (same-user shell already holds OS authority).

    def _resolve_caller(
        self,
        caller: Optional[CallerContext],
        scope: Scope,
        conn: Optional["sqlite3.Connection"] = None,
    ) -> CallerContext:
        if caller is not None:
            if caller.profile_id != scope.profile_id:
                # A caller may never cross profiles (V2-10.05).
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN, "caller outside profile"
                )
            return self._fence_bound_caller(caller, scope, conn)
        return CallerContext(
            profile_id=scope.profile_id,
            principal_id=scope.principal_id or self.host.default_scope().principal_id or "operator",
            workspace_id=scope.workspace_id,
            conversation_id=scope.conversation_id,
            grants=frozenset(GrantKind),
            is_operator=True,
        )

    def _fence_bound_caller(
        self,
        caller: CallerContext,
        scope: Scope,
        conn: Optional["sqlite3.Connection"] = None,
    ) -> CallerContext:
        """Authorization-epoch + revocation fencing for bound callers (V2-09.15).

        ``authz_epoch == 0`` is an unversioned bind — the caller asserts
        its grants against current scope state, and revoked rows still
        narrow the effective context. A pinned ``authz_epoch > 0`` must
        equal the scope's current ``authz_revision``: revocation bumps the
        revision, so a caller bound during a revoked window stays fenced
        even after a re-grant. On first contact the caller's bound grants
        are recorded into ``scope_grants`` (INSERT OR IGNORE — recording
        can never resurrect a revoked grant).

        When ``conn`` is supplied (an in-flight write transaction, e.g.
        ``apply_proposal``) all work happens on it; otherwise a read
        snapshot performs the checks and a short write tx records grants.
        """
        if conn is not None:
            return self._fence_on_conn(conn, caller, scope)

        from .storage.repos_v2 import GrantsRepo

        sid = scope_key(scope)
        grants = GrantsRepo(self.store)
        principal = caller.principal_id or "unknown"
        with self.store.read() as rconn:
            revision = grants.authz_revision(rconn, sid)
            rows = rconn.execute(
                "SELECT permission, revoked_event FROM scope_grants"
                " WHERE scope_id = ? AND principal_id = ?",
                (sid, principal),
            ).fetchall()
        # authz_epoch == 0 means "unversioned bind" — the caller asserts
        # its grant set against current state (revoked rows still narrow).
        # A pinned epoch (>0) must match exactly: a caller bound during a
        # since-revoked window stays fenced even after a re-grant.
        if caller.authz_epoch > 0 and caller.authz_epoch != revision:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                "stale authorization epoch",
            )
        existing = {r[0] for r in rows}
        missing = sorted(
            g.value for g in caller.grants if g.value not in existing
        )
        if missing:
            # Re-run the whole fence inside the write tx so a revocation
            # landing between the read and here still fences this call.
            with self.store.tx() as wconn:
                return self._fence_on_conn(wconn, caller, scope)
        return self._narrow_revoked(caller, rows)

    def _fence_on_conn(
        self,
        conn: "sqlite3.Connection",
        caller: CallerContext,
        scope: Scope,
    ) -> CallerContext:
        from .storage.repos import EventsRepo, ensure_scope
        from .storage.repos_v2 import GrantsRepo

        sid = scope_key(scope)
        grants = GrantsRepo(self.store)
        principal = caller.principal_id or "unknown"
        revision = grants.authz_revision(conn, sid)
        if caller.authz_epoch > 0 and caller.authz_epoch != revision:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                "stale authorization epoch",
            )
        rows = conn.execute(
            "SELECT permission, revoked_event FROM scope_grants"
            " WHERE scope_id = ? AND principal_id = ?",
            (sid, principal),
        ).fetchall()
        existing = {r[0] for r in rows}
        missing = sorted(
            g.value for g in caller.grants if g.value not in existing
        )
        if missing:
            ensure_scope(self.store, conn, scope)
            seq = EventsRepo(self.store).append(
                conn, sid, "grants_bound",
                caller.principal_id or "caller",
                {"permissions": missing}, "host-bind-v1",
            )
            grants.record_bound(conn, sid, principal, missing, "host-bind", seq)
            rows = conn.execute(
                "SELECT permission, revoked_event FROM scope_grants"
                " WHERE scope_id = ? AND principal_id = ?",
                (sid, principal),
            ).fetchall()
        return self._narrow_revoked(caller, rows)

    @staticmethod
    def _narrow_revoked(
        caller: CallerContext, rows: list
    ) -> CallerContext:
        import dataclasses

        revoked = {r[0] for r in rows if r[1] is not None}
        if not revoked:
            return caller
        # A principal with any revoked permission cannot retain the
        # operator wildcard — has() would re-derive the revoked grant.
        narrowed = frozenset(
            g for g in caller.grants
            if g.value not in revoked and g is not GrantKind.OPERATOR
        )
        return dataclasses.replace(
            caller, grants=narrowed, is_operator=False
        )

    def grant_permission(
        self,
        principal_id: str,
        permission: "GrantKind | str",
        *,
        scope: Optional[Scope] = None,
        caller: Optional[CallerContext] = None,
    ) -> dict[str, Any]:
        """Explicitly grant a permission in a scope (operator only).

        The only path that clears a prior revocation. Bumps the scope's
        authorization epoch so stale-bound callers re-verify.
        """
        from .storage.repos import EventsRepo, ensure_scope
        from .storage.repos_v2 import GrantsRepo

        self._require_open()
        sc = scope or self.host.default_scope()
        c = self._resolve_caller(caller, sc)
        if not c.is_operator:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                "grant administration requires operator",
            )
        perm = permission.value if isinstance(permission, GrantKind) else str(permission)
        GrantKind(perm)  # validate against the fixed taxonomy
        sid = scope_key(sc)
        with self.store.tx() as conn:
            ensure_scope(self.store, conn, sc)
            seq = EventsRepo(self.store).append(
                conn, sid, "grant", c.principal_id or "operator",
                {"principal_id": principal_id, "permission": perm},
                "policy-1",
            )
            GrantsRepo(self.store).grant(
                conn, sid, principal_id, perm,
                c.principal_id or "operator", seq,
            )
            revision = GrantsRepo(self.store).bump_authz(conn, sid)
        return {"granted": perm, "principal_id": principal_id,
                "authz_revision": revision}

    def revoke_permission(
        self,
        principal_id: str,
        permission: "GrantKind | str",
        *,
        scope: Optional[Scope] = None,
        caller: Optional[CallerContext] = None,
    ) -> dict[str, Any]:
        """Revoke a permission — bumps authz epoch, fencing stale callers."""
        from .storage.repos import EventsRepo, ensure_scope
        from .storage.repos_v2 import GrantsRepo

        self._require_open()
        sc = scope or self.host.default_scope()
        c = self._resolve_caller(caller, sc)
        if not c.is_operator:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                "grant administration requires operator",
            )
        perm = permission.value if isinstance(permission, GrantKind) else str(permission)
        GrantKind(perm)
        sid = scope_key(sc)
        grants = GrantsRepo(self.store)
        with self.store.tx() as conn:
            ensure_scope(self.store, conn, sc)
            seq = EventsRepo(self.store).append(
                conn, sid, "grant_revoked", c.principal_id or "operator",
                {"principal_id": principal_id, "permission": perm},
                "policy-1",
            )
            revoked = grants.revoke(conn, sid, principal_id, perm, seq)
            revision = grants.bump_authz(conn, sid)
        return {"revoked": perm, "principal_id": principal_id,
                "was_active": revoked, "authz_revision": revision}

    def _require_grant(self, caller: CallerContext, grant: GrantKind) -> None:
        if not caller.has(grant):
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, f"caller missing grant {grant.value}"
            )

    def _require_write(self, caller: CallerContext, scope: Scope) -> None:
        """Write authority: operator, or a caller inside its own partition.

        ``can_write`` demands an exact partition match — right for a bound
        caller (V2-09.12: narrow, never widen). An operator-addressed
        partition is writable wherever it sits (V2-09.17 trusted boundary),
        which also covers partitions whose principal is unset.
        """
        if caller.is_operator:
            return
        if not can_write(caller.scope(scope.visibility), scope):
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, "scope not writable by caller"
            )

    def _authorize_read(self, caller: CallerContext, target: Scope) -> None:
        """can_read for the caller plus every declared audience member.

        Owner-private evidence must not enter a shared audience merely
        because its owner is present (V2-09.06); missing audience metadata
        limits reads to the caller's own safe partition (V2-09.07).
        """
        home = caller.scope()
        if not can_read(home, target):
            raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "scope not readable")
        audience = caller.audience or (caller.principal_id,)
        others = [m for m in audience if m != target.principal_id]
        if not others:
            return
        if target.visibility == Visibility.OWNER:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, "owner-private evidence cannot enter a shared audience"
            )
        if target.visibility == Visibility.CONVERSATION:
            # Conversation evidence may reach the room's participants only
            # when the caller is bound to that very conversation.
            if target.conversation_id != caller.conversation_id:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN, "conversation evidence outside caller's room"
                )
        if target.visibility == Visibility.WORKSPACE and not caller.has(GrantKind.SHARE):
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, "workspace evidence to a shared audience requires share grant"
            )


__all__ = ["GovernanceMixin"]
