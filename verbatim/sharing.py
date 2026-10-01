"""Explicit sharing: scoped handoff capsules (SPEC_V2 §10).

A handoff capsule grants a named recipient time-boxed access to selected
evidence references inside one scope. Capsules reference live evidence —
they do not copy plaintext (V2-10.07) — so consumption reauthorizes
everything at read time: capsule status, expiry, caller identity, and the
current availability of every member. A member that has since been
suppressed or purged is dropped from the response rather than resurrected
from the reference.

* ``create_handoff`` requires ``GrantKind.SHARE`` on the caller and only
  admits objects that live in the caller's home scope and are currently
  disclosable (not suppressed, not erased).
* ``consume_handoff`` admits only the named recipient (or the issuer, for
  verification), requires an open unexpired capsule, and re-checks each
  member; revocation and expiry fail closed.
* ``revoke_handoff`` lets the issuer (or an operator) close a capsule;
  once revoked it can never be consumed again.

Sharing an interpretation implicitly carries its necessary supporting
context (V2-10.03): callers share claim *and* span/source members
explicitly, and consumption reports member availability rather than
silently presenting dangling assertions.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional, Sequence

from .core.time import wall_us
from .core.types import (
    CallerContext,
    ErrorCode,
    VerbatimError,
    require_id,
)
from .purge import OBJECT_KINDS, _object_scope
from .storage.repos import (
    EventsRepo,
    PurgesRepo,
    ensure_scope,
    has_table as _has_table,
)
from .storage.repos_v2 import ErasureRepo, HandoffRepo

#: Member kinds a capsule may reference (same vocabulary as purge targets).
_MEMBER_KINDS = OBJECT_KINDS

_POLICY_VERSION = "policy-1"


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _normalize_ref(ref: Any) -> tuple[str, str, Optional[int]]:
    """``(kind, id[, revision])`` — same normalization as purge targets."""
    if not isinstance(ref, (tuple, list)) or len(ref) not in (2, 3):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "object refs must be (kind, object_id) or (kind, object_id, revision)",
        )
    kind = str(ref[0])
    oid = str(ref[1])
    rev: Optional[int] = ref[2] if len(ref) == 3 else None
    if kind not in _MEMBER_KINDS:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown object kind {kind!r} for handoff"
        )
    if kind == "source" and rev is not None:
        # Revision-scoped source members are stored as source_revision rows.
        return "source_revision", f"{oid}:{rev}", None
    if kind == "source_revision":
        source_id, _, rev_text = oid.rpartition(":")
        if not source_id or not rev_text.isdigit():
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "source_revision object_id must be '<source_id>:<revision>'",
            )
        return kind, oid, None
    require_id(oid, "object_id")
    if rev is not None and (not isinstance(rev, int) or rev < 1):
        raise VerbatimError(ErrorCode.VALIDATION, "revision must be int >= 1")
    return kind, oid, rev


def _member_rows(
    store: Any, conn: sqlite3.Connection, capsule_id: str
) -> list[dict[str, Any]]:
    return _rows(
        conn.execute(
            "SELECT object_kind, object_id, revision FROM capsule_members"
            " WHERE capsule_id = ? ORDER BY object_kind, object_id",
            (capsule_id,),
        )
    )


#: Quarantine states that hide an object revision (mirrors
#: ``security.EXCLUDING_STATES``; re-declared so the fallback path never
#: imports a parallel module — same convention as retrieval/v3 union).
_HELD_STATES = frozenset({"pending", "suppressed"})


def _held(conn: sqlite3.Connection, object_kind: str, object_id: str,
          revision: int) -> bool:
    """Quarantine exclusion for one object revision (V3-14.10).

    ``security.should_exclude`` when the parallel module is provisioned,
    the quarantine-table state otherwise — the same defensive pair
    ``retrieval/v3/union.py::_should_exclude`` uses. Fails closed on
    unreadable state; callers gate on ``_has_table(conn, "quarantine")``
    so schemas that predate the table are unaffected.
    """
    try:
        from . import security as _security  # type: ignore
    except Exception:
        _security = None
    if _security is not None:
        try:
            return bool(
                _security.should_exclude(conn, object_kind, object_id, revision)
            )
        except Exception:
            pass  # fall back to the local table check below
    try:
        row = conn.execute(
            "SELECT state FROM quarantine"
            " WHERE object_kind = ? AND object_id = ? AND revision = ?",
            (object_kind, object_id, revision),
        ).fetchone()
    except sqlite3.Error:
        return True  # no readable quarantine state — fail closed
    return row is not None and row[0] in _HELD_STATES


def _source_revision_held(
    conn: sqlite3.Connection, source_id: str, revision: int
) -> bool:
    """Quarantine hold on a source revision or a covering envelope.

    A hold on ``("source", sid, rev)`` or on any ``source_envelopes`` row
    for ``(sid, rev)`` withholds the revision's bytes (V3-14.10) — the
    same resolution retrieval/v3's evidence cascade performs.
    """
    if not _has_table(conn, "quarantine"):
        return False  # schema predates quarantine — no holds exist
    if _held(conn, "source", source_id, revision):
        return True
    if not _has_table(conn, "source_envelopes"):
        return False
    rows = conn.execute(
        "SELECT envelope_id FROM source_envelopes"
        " WHERE source_id = ? AND revision = ?",
        (source_id, revision),
    ).fetchall()
    return any(
        _held(conn, "source_envelope", r[0], revision) for r in rows
    )


def _member_held(
    conn: sqlite3.Connection, kind: str, oid: str, revision: Optional[int]
) -> bool:
    """Quarantine cascade for a capsule member (V3-14.10).

    A hold on the member's own object ref — or anywhere in its resolvable
    evidence chain (span → source revision → covering source envelope) —
    makes it unavailable for disclosure. Fail closed on unreadable
    quarantine state via ``_held``.
    """
    if not _has_table(conn, "quarantine"):
        return False  # schema predates quarantine — no holds exist
    if kind == "source_revision":
        source_id, _, rev_text = oid.rpartition(":")
        # rpartition shape was validated by _normalize_ref.
        return _source_revision_held(conn, source_id, int(rev_text))
    if kind == "source":
        revs = [
            r[0]
            for r in conn.execute(
                "SELECT revision FROM source_revisions WHERE source_id = ?",
                (oid,),
            ).fetchall()
        ]
        return any(
            _source_revision_held(conn, oid, r)
            for r in (revs or [revision or 1])
        )
    if kind == "span":
        row = conn.execute(
            "SELECT source_id, revision FROM spans WHERE span_id = ?",
            (oid,),
        ).fetchone()
        span_rev = row[1] if row is not None else (revision or 1)
        if _held(conn, "span", oid, span_rev):
            return True
        return row is not None and _source_revision_held(conn, row[0], row[1])
    if kind == "claim":
        revs = [
            r[0]
            for r in conn.execute(
                "SELECT revision FROM claim_revisions WHERE claim_id = ?",
                (oid,),
            ).fetchall()
        ]
        if any(
            _held(conn, "claim", oid, r) for r in (revs or [revision or 1])
        ):
            return True
        # Evidence chain: a hold on a cited span or its source revision
        # (or covering envelope) taints the interpretation too.
        for span_id, src, srev in conn.execute(
            "SELECT ce.span_id, s.source_id, s.revision FROM claim_evidence ce"
            " JOIN spans s ON s.span_id = ce.span_id WHERE ce.claim_id = ?",
            (oid,),
        ).fetchall():
            if _held(conn, "span", span_id, srev):
                return True
            if _source_revision_held(conn, src, srev):
                return True
        return False
    # artifact / episode / procedure / other member kinds: direct hold only.
    return _held(conn, kind, oid, revision or 1)


def _member_available(
    store: Any,
    conn: sqlite3.Connection,
    scope_id: str,
    kind: str,
    oid: str,
    revision: Optional[int],
    suppressed: dict[str, set[str]],
    erasure: Optional[ErasureRepo],
) -> bool:
    """A member is disclosable only while it exists in the capsule's scope,
    is not tombstoned by a suppressing purge, is not erasure-fenced, and
    carries no quarantine hold on itself or its evidence chain."""
    owner = _object_scope(conn, kind, oid)
    if owner != scope_id:
        return False
    if oid in suppressed.get(kind, ()):
        return False
    if erasure is not None and erasure.is_erased(conn, scope_id, kind, oid):
        return False
    if _member_held(conn, kind, oid, revision):
        return False
    return True


def _suppressed_map(
    store: Any,
    conn: sqlite3.Connection,
    members: Sequence[tuple[str, str]],
) -> dict[str, set[str]]:
    purges = PurgesRepo(store)
    out: dict[str, set[str]] = {}
    by_kind: dict[str, list[str]] = {}
    for kind, oid in members:
        by_kind.setdefault(kind, []).append(oid)
    for kind, ids in by_kind.items():
        try:
            out[kind] = purges.suppressed_ids(kind, ids)
        except Exception:
            out[kind] = set(ids)  # fail closed on the disclosure side
    return out


# ----------------------------------------------------------------------
# create
# ----------------------------------------------------------------------


def create_handoff(
    store: Any,
    caller: CallerContext,
    recipient_id: str,
    object_refs: Sequence[Any],
    *,
    permission: str = "read_evidence",
    expires_us: Optional[int] = None,
    snapshot: Any = None,
) -> dict[str, Any]:
    """Create a live-reference handoff capsule in the caller's home scope.

    Requires ``GrantKind.SHARE`` (V2-09.10); every ref must resolve to an
    object inside the caller's scope and be currently disclosable — the
    same not-found/forbidden answer covers missing, foreign, suppressed,
    and erased objects (V2-09.14).
    """
    if not isinstance(caller, CallerContext):
        raise VerbatimError(ErrorCode.VALIDATION, "caller context required")
    if not caller.has(_SHARE):
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_FORBIDDEN, "caller missing share grant"
        )
    require_id(recipient_id, "recipient_id")
    if permission not in HandoffRepo._PERMISSIONS:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"invalid permission {permission!r}"
        )
    if expires_us is not None:
        if not isinstance(expires_us, int) or expires_us <= 0:
            raise VerbatimError(
                ErrorCode.VALIDATION, "expires_us must be a positive int"
            )
        if expires_us <= wall_us():
            raise VerbatimError(
                ErrorCode.VALIDATION, "expires_us is already in the past"
            )
    if not object_refs:
        raise VerbatimError(ErrorCode.VALIDATION, "object_refs must be non-empty")
    norm = [_normalize_ref(r) for r in object_refs]

    with store.tx() as conn:
        sid = ensure_scope(store, conn, caller.scope())
        erasure = ErasureRepo(store) if _has_table(conn, "erasure_ledger") else None
        suppressed = _suppressed_map(
            store, conn, [(k, o) for k, o, _r in norm]
        )
        for kind, oid, _rev in norm:
            owner = _object_scope(conn, kind, oid)
            if owner is None or owner != sid:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                    f"{kind} {oid!r} not found in caller scope",
                )
            if oid in suppressed.get(kind, ()):
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                    f"{kind} {oid!r} not found in caller scope",
                )
            if erasure is not None and erasure.is_erased(conn, sid, kind, oid):
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                    f"{kind} {oid!r} not found in caller scope",
                )
            if _member_held(conn, kind, oid, _rev):
                # Quarantine hold on the member or its evidence chain —
                # held content is never shareable (V3-14.10).
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                    f"{kind} {oid!r} not found in caller scope",
                )
        repo = HandoffRepo(store)
        issuer = caller.agent_id or caller.principal_id
        capsule_id = repo.create(
            conn,
            sid,
            recipient_id,
            issuer,
            permission=permission,
            expires_us=expires_us,
            portable=False,
            snapshot=snapshot,
        )
        for kind, oid, rev in norm:
            repo.add_member(conn, capsule_id, kind, oid, revision=rev or 1)
        EventsRepo(store).append(
            conn,
            sid,
            "handoff_created",
            issuer,
            {
                "capsule_id": capsule_id,
                "recipient_id": recipient_id,
                "permission": permission,
                "member_count": len(norm),
                "expires_us": expires_us,
            },
            _POLICY_VERSION,
        )
    return {
        "capsule_id": capsule_id,
        "scope_id": sid,
        "recipient_id": recipient_id,
        "issuer_id": issuer,
        "permission": permission,
        "expires_us": expires_us,
        "status": "open",
        "member_count": len(norm),
    }


# ----------------------------------------------------------------------
# consume
# ----------------------------------------------------------------------


def consume_handoff(
    store: Any,
    caller: CallerContext,
    capsule_id: str,
    *,
    mark_consumed: bool = True,
) -> dict[str, Any]:
    """Consume a capsule: reauthorize status, expiry, identity, members.

    Only the named recipient or the issuer may consume. A revoked or
    expired capsule fails closed (V2-10.07); expiry is latched into the
    capsule row the first time it is observed. Members that have since
    vanished, been suppressed, or been purged are reported under
    ``dropped`` — the response is the live view, not the recorded one.
    """
    if not isinstance(caller, CallerContext):
        raise VerbatimError(ErrorCode.VALIDATION, "caller context required")
    require_id(capsule_id, "capsule_id")
    # Phase 1: committed expiry latch. This runs in its own transaction so a
    # subsequent deny cannot roll the observation back — once expiry is seen
    # the capsule is 'expired' for good.
    with store.tx() as conn:
        cap0 = _capsule_row(conn, capsule_id)
        if (
            cap0 is not None
            and cap0["status"] == "open"
            and cap0["expires_us"] is not None
            and cap0["expires_us"] <= wall_us()
        ):
            HandoffRepo(store).set_status(conn, capsule_id, "expired")
            cap0["status"] = "expired"
    if cap0 is None:
        raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "capsule not found")
    if cap0["status"] != "open":
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_FORBIDDEN,
            f"capsule not open (status {cap0['status']!r})",
        )
    with store.tx() as conn:
        capsule = _capsule_row(conn, capsule_id)
        if capsule is None or capsule["status"] != "open":
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, "capsule not open"
            )
        if capsule["expires_us"] is not None and capsule["expires_us"] <= wall_us():
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, "capsule expired"
            )
        scope_row = conn.execute(
            "SELECT profile_id FROM scopes WHERE scope_id = ?",
            (capsule["scope_id"],),
        ).fetchone()
        if scope_row is not None and scope_row[0] != caller.profile_id:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, "capsule not found"
            )
        who = caller.agent_id or caller.principal_id
        role = None
        if caller.principal_id == capsule["recipient_id"] or (
            caller.agent_id is not None and caller.agent_id == capsule["recipient_id"]
        ):
            role = "recipient"
        elif caller.principal_id == capsule["issuer_id"] or (
            caller.agent_id is not None and caller.agent_id == capsule["issuer_id"]
        ):
            role = "issuer"
        if role is None and not caller.has(_OPERATOR):
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, "caller is not a capsule party"
            )
        if role is None:
            role = "operator"
        members = _member_rows(store, conn, capsule_id)
        erasure = ErasureRepo(store) if _has_table(conn, "erasure_ledger") else None
        suppressed = _suppressed_map(
            store, conn, [(m["object_kind"], m["object_id"]) for m in members]
        )
        live: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        for m in members:
            kind, oid = m["object_kind"], m["object_id"]
            if _member_available(
                store, conn, capsule["scope_id"], kind, oid,
                m["revision"], suppressed, erasure,
            ):
                live.append(
                    {"object_kind": kind, "object_id": oid, "revision": m["revision"]}
                )
            else:
                dropped.append(
                    {"object_kind": kind, "object_id": oid, "reason": "unavailable"}
                )
        warnings = [
            f"{len(dropped)} member(s) unavailable (suppressed, purged, or removed)"
        ] if dropped else []
        consumed_event = None
        if mark_consumed:
            consumed_event = conn.execute(
                "SELECT COALESCE(MAX(event_seq), 0) + 1 FROM events"
            ).fetchone()[0]
            HandoffRepo(store).set_status(
                conn, capsule_id, "consumed", consumed_event=consumed_event
            )
        EventsRepo(store).append(
            conn,
            capsule["scope_id"],
            "handoff_consumed",
            who,
            {
                "capsule_id": capsule_id,
                "role": role,
                "live_members": len(live),
                "dropped_members": len(dropped),
            },
            _POLICY_VERSION,
        )
    return {
        "capsule_id": capsule_id,
        "scope_id": capsule["scope_id"],
        "recipient_id": capsule["recipient_id"],
        "issuer_id": capsule["issuer_id"],
        "permission": capsule["permission"],
        "expires_us": capsule["expires_us"],
        "status": "consumed" if mark_consumed else "open",
        "role": role,
        "members": live,
        "dropped": dropped,
        "warnings": warnings,
    }


def _capsule_row(conn: sqlite3.Connection, capsule_id: str) -> Optional[dict[str, Any]]:
    rows = _rows(
        conn.execute(
            "SELECT * FROM handoff_capsules WHERE capsule_id = ?",
            (capsule_id,),
        )
    )
    return rows[0] if rows else None


# ----------------------------------------------------------------------
# revoke
# ----------------------------------------------------------------------


def revoke_handoff(
    store: Any,
    caller: CallerContext,
    capsule_id: str,
) -> dict[str, Any]:
    """Revoke a capsule. Issuer (or operator) only; revoked is terminal."""
    if not isinstance(caller, CallerContext):
        raise VerbatimError(ErrorCode.VALIDATION, "caller context required")
    require_id(capsule_id, "capsule_id")
    with store.tx() as conn:
        capsule = _capsule_row(conn, capsule_id)
        if capsule is None:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, "capsule not found"
            )
        who = caller.agent_id or caller.principal_id
        is_issuer = caller.principal_id == capsule["issuer_id"] or (
            caller.agent_id is not None and caller.agent_id == capsule["issuer_id"]
        )
        if not is_issuer and not caller.has(_OPERATOR):
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                "only the issuer may revoke a capsule",
            )
        if capsule["status"] != "revoked":
            HandoffRepo(store).set_status(conn, capsule_id, "revoked")
            EventsRepo(store).append(
                conn,
                capsule["scope_id"],
                "handoff_revoked",
                who,
                {"capsule_id": capsule_id},
                _POLICY_VERSION,
            )
    return {
        "capsule_id": capsule_id,
        "status": "revoked",
        "revoked_by": who,
    }


# GrantKind values referenced through module aliases so the file reads
# top-down without a second import block.
from .core.types import GrantKind as _GrantKind  # noqa: E402

_SHARE = _GrantKind.SHARE
_OPERATOR = _GrantKind.OPERATOR


__all__ = [
    "create_handoff",
    "consume_handoff",
    "revoke_handoff",
]
