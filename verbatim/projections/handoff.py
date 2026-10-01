"""Handoff capsules for file projections (SPEC_V4 V4-47.07/08).

A capsule is a *bounded read-only authorization handoff*: it binds a
recipient, a purpose, an expiry, an exact set of object revisions, and
an allowed-operation list — persisted in the existing
``handoff_capsules``/``capsule_members`` tables (v3 columns
``verbs_json``/``purpose``/``delegation_chain_json``/``source_epoch``
written when present).

Authority discipline (V4-47.07):

* The issuer must hold — at creation time, through ``governance.authorize``
  — every verb the capsule delegates, on the capsule's scope. A capsule
  is a *subset* of issuer authority, never an expansion.
* Member objects are resolved + liveness-checked through
  ``Kernel.resolve_access`` under each delegated verb — a held,
  suppressed, or out-of-scope member cannot ride a capsule.
* Consumption re-authorizes the *recipient* under the store's current
  grants — the capsule narrows to ∩(capsule verbs, recipient grants);
  it never confers a verb the recipient lacks locally.

Offline discipline (V4-47.08):

* ``present_offline`` is pure evaluation of a capsule snapshot: expiry
  vs. caller-supplied clock, ``revocation_state: "unknown"`` reported
  honestly (an offline consumer cannot observe the issuing store), and
  a *short cached-authorization lease* — the offline grant dies at
  ``min(expires_us, issued_us + lease_cap_us)`` regardless of the
  capsule's own expiry.
* ``act``/``hydrate``/``promote`` (and any non read/quote verb) are
  structurally refused at creation and rejected again at offline
  presentation — offline capsules are read-only by construction.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable, Optional

from ..core.time import now_us as _now_us
from ..core.types import ErrorCode, VerbatimError, json_dumps, require_id, safe_json_loads
from ..core.types_v3 import Verb
from ..governance import authorize
from ..kernel import Kernel
from ..storage.repos import _next_event_seq, has_table as _has_table
from ..storage.repos_v2 import HandoffRepo
from .builder import ProjectionAuthority

#: Verbs a capsule may ever delegate — read-only by construction.
CAPSULE_VERBS = frozenset({Verb.READ.value, Verb.QUOTE.value})

#: Verbs that must never ride a capsule (V4-47.08): mutation, hydration,
#: and promotion/derivation lanes are online-only operations.
PROHIBITED_VERBS = frozenset(
    {
        Verb.ACT.value,
        Verb.HYDRATE.value,
        Verb.DERIVE.value,
        Verb.SHARE.value,
        Verb.INGEST.value,
        Verb.ADMIN.value,
        "promote",
    }
)

#: Default cached-authorization lease for offline presentation — an
#: hour, not the capsule's full expiry. Callers may tighten, never loosen.
DEFAULT_OFFLINE_LEASE_US = 3_600_000_000
_MAX_OFFLINE_LEASE_US = 86_400_000_000  # absolute ceiling: 24h


def _fail(msg: str) -> VerbatimError:
    return VerbatimError(ErrorCode.VALIDATION, msg)


def _has_v3_capsule_cols(conn: sqlite3.Connection) -> bool:
    try:
        cols = {
            r[1] for r in conn.execute("PRAGMA table_info(handoff_capsules)")
        }
    except sqlite3.Error:
        return False
    return "verbs_json" in cols


def create_capsule(
    store: Any,
    *,
    issuer: ProjectionAuthority,
    scope_id: str,
    recipient_id: str,
    members: Iterable[tuple[str, str, int]],
    verbs: Iterable[str] = ("read",),
    expires_us: Optional[int] = None,
    portable: bool = False,
    now_us: Optional[int] = None,
) -> dict[str, Any]:
    """Mint a bound capsule row + member refs in one transaction.

    ``members`` are ``(object_kind, object_id, revision)`` triples — the
    exact revisions the handoff covers (V4-47.07). Every member must be
    authorized for the issuer under every delegated verb *and* live in
    the capsule's scope; a denied member denies the capsule.
    """
    if not isinstance(issuer, ProjectionAuthority):
        raise _fail("issuer must be a ProjectionAuthority")
    now = int(now_us) if now_us is not None else _now_us()
    scope_id = require_id(scope_id, "scope_id")
    recipient_id = require_id(recipient_id, "recipient_id")
    if expires_us is None or int(expires_us) <= now:
        raise _fail("capsule requires a future expires_us (V4-47.07)")
    verb_set = frozenset(str(v) for v in verbs)
    if not verb_set or not verb_set <= CAPSULE_VERBS:
        bad = sorted(verb_set - CAPSULE_VERBS)
        raise _fail(
            f"capsule verbs must be a non-empty subset of read/quote; "
            f"rejected: {bad}"
        )
    if verb_set & PROHIBITED_VERBS:
        raise _fail("capsule carries a prohibited verb")
    member_list = sorted(
        {
            (str(k), require_id(o, "object_id"), int(r))
            for k, o, r in (members or ())
        }
    )
    if not member_list:
        raise _fail("capsule requires at least one member object")

    caller = issuer.caller()
    issuer_id = caller.principal_id
    kernel = Kernel(store)
    capsule_id: Optional[str] = None
    with store.tx() as conn:
        if not _has_table(conn, "handoff_capsules"):
            raise VerbatimError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "handoff_capsules table absent — store predates v2 schema",
            )
        # Issuer must hold each delegated verb on this scope *now* —
        # authority can be narrowed, never expanded (V4-47.07).
        for v in sorted(verb_set):
            authorize(conn, caller, scope_id, v, purpose=issuer.purpose)
        # Members must be in-scope + live under each delegated verb.
        refs = [(k, o, r) for k, o, r in member_list]
        for v in sorted(verb_set):
            lease = kernel.resolve_access(
                conn,
                caller,
                v,
                issuer.purpose,
                [scope_id],
                object_refs=refs,
                now_us=now,
            )
            if lease.denied:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
                    "capsule member not authorized or not in scope",
                )
        epoch = None
        try:
            from ..governance import epochs as _epochs

            epoch = _epochs.current_epoch(conn, scope_id)
        except Exception:
            epoch = None  # epoch ledger absent on pre-v3 stores

        cid = None
        snapshot = {
            "issued_us": now,
            "expires_us": int(expires_us),
            "verbs": sorted(verb_set),
            "purpose": issuer.purpose,
            "members": [
                {"kind": k, "object_id": o, "revision": r}
                for k, o, r in member_list
            ],
            "revocation_state": "unknown-offline" if portable else "tracked",
            "offline_lease_us": (
                DEFAULT_OFFLINE_LEASE_US if portable else None
            ),
        }
        if _has_v3_capsule_cols(conn):
            from ..core.types import new_id as _new_id

            cid = _new_id()
            conn.execute(
                "INSERT INTO handoff_capsules"
                "(capsule_id, scope_id, recipient_id, issuer_id, permission,"
                " expires_us, portable, status, snapshot_json, created_event,"
                " verbs_json, purpose, delegation_chain_json, source_epoch)"
                " VALUES (?, ?, ?, ?, 'read_evidence', ?, ?, 'open', ?,"
                " ?, ?, ?, ?, ?)",
                (
                    cid,
                    scope_id,
                    recipient_id,
                    issuer_id,
                    int(expires_us),
                    1 if portable else 0,
                    json_dumps(snapshot),
                    _next_event_seq(conn),
                    json_dumps(sorted(verb_set)),
                    issuer.purpose,
                    json_dumps([issuer_id]),
                    epoch,
                ),
            )
        else:
            repo = HandoffRepo(store)
            cid = repo.create(
                conn,
                scope_id,
                recipient_id,
                issuer_id,
                permission="read_evidence",
                expires_us=int(expires_us),
                portable=portable,
                snapshot=snapshot,
            )
        repo = HandoffRepo(store)
        for k, o, r in member_list:
            repo.add_member(conn, cid, k, o, revision=r)
        capsule_id = cid
    return {
        "capsule_id": capsule_id,
        "scope_id": scope_id,
        "recipient_id": recipient_id,
        "issuer_id": issuer_id,
        "verbs": sorted(verb_set),
        "purpose": issuer.purpose,
        "expires_us": int(expires_us),
        "members": [
            {"kind": k, "object_id": o, "revision": r}
            for k, o, r in member_list
        ],
        "portable": bool(portable),
        "source_epoch": epoch,
    }


def open_capsule(
    store: Any,
    capsule_id: str,
    *,
    recipient: ProjectionAuthority,
    now_us: Optional[int] = None,
) -> dict[str, Any]:
    """Consume a capsule *online*: re-authorize the recipient under the
    store's live grants and mint a real kernel lease for the members.

    Returns ``{lease, granted_verbs, denied_verbs, members}`` — the
    lease covers only verbs the recipient actually holds (∩ rule); a
    fully denied capsule returns ``granted_verbs: []``, never bytes.
    """
    if not isinstance(recipient, ProjectionAuthority):
        raise _fail("recipient must be a ProjectionAuthority")
    now = int(now_us) if now_us is not None else _now_us()
    require_id(capsule_id, "capsule_id")
    kernel = Kernel(store)
    repo = HandoffRepo(store)
    cap = repo.get(capsule_id)
    if cap is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "capsule not found"
        )
    if cap["recipient_id"] != recipient.principal_id:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
            "capsule binds a different recipient",
        )
    if cap["status"] != "open":
        raise VerbatimError(
            ErrorCode.INVALID_TRANSITION,
            f"capsule is {cap['status']}, not open",
        )
    expires = cap.get("expires_us")
    if expires is not None and int(expires) <= now:
        with store.tx() as conn:
            repo.set_status(conn, capsule_id, "expired")
        raise VerbatimError(
            ErrorCode.PERMIT_EXPIRED, "capsule expired", retryable=False
        )

    members = repo.members(capsule_id)
    verbs = safe_json_loads(cap.get("verbs_json") or "[]")
    if not isinstance(verbs, list) or not verbs:
        # v2-schema capsules carry verbs in the snapshot envelope.
        snap = safe_json_loads(cap.get("snapshot_json") or "{}")
        verbs = snap.get("verbs") if isinstance(snap, dict) else None
    if not isinstance(verbs, list) or not verbs:
        verbs = [Verb.READ.value]
    verbs = sorted({str(v) for v in verbs} & CAPSULE_VERBS)
    purpose = cap.get("purpose") or recipient.purpose

    caller = recipient.caller()
    granted: list[str] = []
    leases: dict[str, str] = {}
    with store.read() as conn:
        for v in verbs:
            try:
                authorize(conn, caller, cap["scope_id"], v, purpose=purpose)
            except VerbatimError:
                continue  # recipient lacks it — capsule narrows, never widens
            refs = [
                (m["object_kind"], m["object_id"], int(m["revision"]))
                for m in members
            ]
            lease = kernel.resolve_access(
                conn,
                caller,
                v,
                purpose,
                [cap["scope_id"]],
                object_refs=refs,
                now_us=now,
            )
            if lease.denied:
                continue
            granted.append(v)
            leases[v] = lease.lease_id
    if granted:
        with store.tx() as conn:
            repo.set_status(conn, capsule_id, "consumed")
    return {
        "capsule_id": capsule_id,
        "scope_id": cap["scope_id"],
        "granted_verbs": granted,
        "denied_verbs": [v for v in verbs if v not in granted],
        "leases": leases,
        "members": members,
        "expires_us": expires,
        "purpose": purpose,
    }


def present_offline(
    capsule_snapshot: dict[str, Any],
    *,
    now_us: int,
    lease_cap_us: int = DEFAULT_OFFLINE_LEASE_US,
) -> dict[str, Any]:
    """Evaluate a shipped capsule snapshot *without* the issuing store.

    Returns the honest offline capability view: the delegated verbs as
    *cached* authorizations valid only until the short lease horizon,
    ``revocation_state`` reported as unknown (an offline party cannot
    observe the issuing store), and the prohibited lane listed
    explicitly. Never mints a store lease — offline presentation is a
    read-only description, not an authorization.
    """
    if not isinstance(capsule_snapshot, dict):
        raise _fail("capsule snapshot must be a dict")
    now = int(now_us)
    if (
        isinstance(lease_cap_us, bool)
        or not isinstance(lease_cap_us, int)
        or lease_cap_us <= 0
        or lease_cap_us > _MAX_OFFLINE_LEASE_US
    ):
        raise _fail("offline lease cap must be in (0, 24h]")
    expires = capsule_snapshot.get("expires_us")
    issued = capsule_snapshot.get("issued_us")
    verbs = [
        v
        for v in (capsule_snapshot.get("verbs") or [])
        if v in CAPSULE_VERBS
    ]
    expired = expires is None or int(expires) <= now
    lease_until = (
        min(int(expires), int(issued) + lease_cap_us)
        if isinstance(expires, int) and isinstance(issued, int)
        else now
    )
    return {
        "valid": not expired and lease_until > now and bool(verbs),
        "expired": expired,
        "cached_authorization_until": lease_until,
        "verbs": verbs,
        "prohibited": sorted(PROHIBITED_VERBS),
        "revocation_state": "unknown-offline",
        "members": capsule_snapshot.get("members") or [],
        "purpose": capsule_snapshot.get("purpose"),
        "note": (
            "offline presentation: issuing store unreachable — current "
            "revocation unknown; act/hydrate/promotion prohibited"
        ),
    }


__all__ = [
    "CAPSULE_VERBS",
    "DEFAULT_OFFLINE_LEASE_US",
    "PROHIBITED_VERBS",
    "create_capsule",
    "open_capsule",
    "present_offline",
]
