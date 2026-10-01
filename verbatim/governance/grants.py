"""Effective-verb authorization over ``grants_v3`` (SPEC_V3 §09, SPEC_V4 §11).

A request is permitted only when an unexpired, unrevoked grant covers the
(scope, verb[, purpose]) tuple at the caller's pinned epoch — verb
containment over the stored set, never string equality against a
principal's claim (§09.01, §09.03, §09.08). Denial and absence share one
public error class: ``NOT_FOUND_OR_UNAUTHORIZED`` (§09.09, §10.05), and
authorization is evaluated before any private identifier is dereferenced —
``object_ref`` is accepted for call-site compatibility and deliberately
never resolved here.

Purpose constraints are *tagged* (V4-11.01, F4-03): every grant row carries
``purpose_tag`` ∈ ``any`` | ``set`` | ``none`` alongside ``purposes_json``.
``ANY`` permits every declared purpose, ``SET`` only the listed ones,
``NONE`` none — and an empty SET normalizes to NONE, so an empty
``purposes_json`` can never silently evaluate as unrestricted again.
Pre-v4 rows migrated by schema v4 carry ``purpose_tag='legacy_empty'``
(or NULL when a row bypassed tagging entirely); they keep their historical
empty-means-ANY evaluation so existing authority does not silently change
underfoot, and they are surfaced by ``legacy_purpose_grants`` for
owner-reviewed remediation (V4-62.05).

Delegation is attenuation-only (§09.03, V4-11.02): the child grant must
prove a subset of the parent's *effective* authority across every
dimension — verbs (set ⊆), scope (inherited verbatim; grants have no
narrower resource selector), purposes (``PurposeConstraint.is_subset_of``),
expiry (≤ parent's, inherited when unspecified), and delegation depth
(parent − 1, so a depth-0 grant cannot delegate). Caveats are conjunctive
restrictions (V4-11.03): the child must *retain or strengthen* them —
``parent_caveats ⊆ child_caveats`` — never drop one through ordinary
subset logic. Delegating with an empty purpose set is legal but yields a
NONE-purpose child with no recall authority (C09) — never ANY.

Compat note (V4-05.04): ``create_grant``'s ``purposes=None`` default still
issues an ANY-tagged grant — the historical v3 root-grant semantic — but
the row is now *explicitly* tagged ``any`` rather than ambiguous. Callers
should pass a ``PurposeConstraint`` (or purpose iterable) explicitly; a
passed empty iterable is an explicit empty SET and normalizes to NONE.
"""

from __future__ import annotations

import math
import sqlite3
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Iterable, Optional

from ..core.time import now_us
from ..core.types import ErrorCode, VerbatimError, new_id, require_id
from ..core.types_v3 import Grant, Verb
from ..core.types_v4 import PurposeConstraint, PurposeTag
from ..storage import repos_v3
from . import epochs, purposes as purposes_registry

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from . import CallerV3

_GRANTS = "grants_v3"
_DELEGATIONS = "delegations"
# Hard cap on delegation-chain walks at evaluation: a cycle or pathological
# hand-written chain is dead authority, not a denial of service (§09.13
# depth-overflow tests). Independent of the creation bound below — it
# guards rows no API produced.
_MAX_CHAIN = 32
# Creation-side bound on the delegation-depth budget (V4-11.09): a grant
# minted through ``create_grant`` may carry at most this much further
# delegation depth, so real chains can never exceed it (each delegation
# decrements by one). Documented bound; rows above it are rejected.
MAX_DELEGATION_DEPTH = 8
# Schema-v4 migration tag for pre-v4 rows whose bare empty purposes_json
# historically meant ANY (schema_v4.DDL_V4_ALTER; V4-62.05).
LEGACY_EMPTY_TAG = "legacy_empty"

DENIAL_MESSAGE = "not found or unauthorized"


def _deny() -> "None":
    """The single public denial — identical for absent and forbidden."""
    raise VerbatimError(ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, DENIAL_MESSAGE)


def _verb(value: "Verb | str") -> Verb:
    try:
        return value if isinstance(value, Verb) else Verb(value)
    except ValueError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown verb {value!r}"
        ) from exc


def _verb_set(verbs: Iterable["Verb | str"]) -> "frozenset[Verb]":
    try:
        out = frozenset(_verb(v) for v in verbs)
    except VerbatimError:
        raise
    if not out:
        raise VerbatimError(ErrorCode.VALIDATION, "grant requires >= 1 verb")
    return out


def _json_set(row: dict, key: str) -> frozenset:
    return frozenset(repos_v3.json_field(row, key) or ())


def _finite_int(value, name: str) -> int:
    """V4-11.09: reject non-integer and non-finite numeric fields."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerbatimError(
            ErrorCode.VALIDATION, f"{name} must be an integer, got {value!r}"
        )
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"{name} must be a finite integer, got {value!r}",
            )
        value = int(value)
    return value


def _opt_finite_int(value, name: str) -> Optional[int]:
    if value is None:
        return None
    return _finite_int(value, name)


def _row_live(row: dict, now: int) -> bool:
    """Unrevoked and unexpired as of ``now`` (wall-clock expiry — §09.03).

    A malformed stored expiry (non-numeric or non-finite) fails closed:
    the row is dead, never accidentally live (V4-11.09).
    """
    if row.get("revoked_us") is not None:
        return False
    exp = row.get("expires_us")
    if exp is None:
        return True
    if isinstance(exp, bool) or not isinstance(exp, (int, float)):
        return False
    if isinstance(exp, float) and not math.isfinite(exp):
        return False
    return exp > now


def _row_epoch(row: dict) -> Optional[int]:
    """Persisted grant epoch; a malformed value reads as no epoch (the
    caller treats ``None`` as not-applicable — fail closed, V4-11.09)."""
    try:
        return int(row.get("epoch") or 0)
    except (TypeError, ValueError):
        return None


def _principal_retired(conn: sqlite3.Connection, principal_id: str) -> bool:
    row = repos_v3.get(conn, "principals", {"principal_id": principal_id})
    return row is not None and bool(row.get("retired"))


# ---------------------------------------------------------------------------
# Tagged purpose constraints (V4-11.01)
# ---------------------------------------------------------------------------


def _coerce_purposes(
    value: "PurposeConstraint | Iterable[str] | None",
    *,
    default: PurposeConstraint,
) -> PurposeConstraint:
    """Normalize a ``purposes`` argument to a ``PurposeConstraint``.

    ``None`` selects the caller-supplied ``default`` (create: explicit ANY
    for v3 compatibility; delegate: inherit the parent's constraint). A
    ``PurposeConstraint`` passes through. Any other iterable becomes
    ``SET(values)`` — an empty iterable is an explicit empty set and
    normalizes to NONE inside the dataclass, never ANY (F4-03). Bare
    strings, bytes, and raw dicts are ambiguous forms and are rejected
    (V4-11.09) — say ``PurposeConstraint.any()`` / ``.set(...)`` instead.
    """
    if value is None:
        return default
    if isinstance(value, PurposeConstraint):
        return value
    if isinstance(value, (str, bytes, dict)):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "ambiguous purpose constraint — pass a PurposeConstraint or an "
            "iterable of purpose names",
        )
    try:
        return PurposeConstraint.set(value)
    except TypeError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "purposes must be a PurposeConstraint or an iterable of names",
        ) from exc


def _row_constraint(row: dict) -> Optional[PurposeConstraint]:
    """The tagged purpose constraint a stored grant row evaluates under.

    ``purpose_tag`` decides (V4-11.01). ``legacy_empty``/NULL rows — the
    pre-v4 ambiguous encoding — keep their historical evaluation exactly:
    empty ``purposes_json`` meant ANY, a non-empty set meant SET. They are
    reportable via ``legacy_purpose_grants`` (V4-62.05). A malformed or
    self-contradictory persisted form (unknown tag, values on a non-SET
    tag, non-string members, undecodable JSON) returns ``None`` and fails
    closed.
    """
    tag = row.get("purpose_tag")
    try:
        values = _json_set(row, "purposes_json")
        if tag is None or tag == LEGACY_EMPTY_TAG:
            # Historical semantics: membership check on the stored set,
            # unrestricted when the set was empty.
            return (
                PurposeConstraint.set(values)
                if values
                else PurposeConstraint.any()
            )
        return PurposeConstraint(PurposeTag(str(tag)), values)
    except (VerbatimError, ValueError, TypeError):
        return None


def grant_purpose_constraint(row: dict) -> PurposeConstraint:
    """Public view of a stored grant row's tagged purpose constraint.

    Raises VALIDATION when the persisted form is malformed — callers
    inspecting a row get the same ambiguity rejection the evaluator
    applies (V4-11.09), never a guessed constraint.
    """
    constraint = _row_constraint(row)
    if constraint is None:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"grant {row.get('grant_id')!r} carries a malformed "
            "purpose constraint",
        )
    return constraint


def legacy_purpose_grants(
    conn: sqlite3.Connection, *, include_revoked: bool = False
) -> list[dict]:
    """V4-62.05 remediation surface: grants still on the ambiguous pre-v4
    purpose encoding.

    Lists ``grants_v3`` rows whose ``purpose_tag`` is ``'legacy_empty'``
    (stamped by the 3→4 migration) or NULL (written without ever being
    tagged). These retain their historical evaluation — an empty stored
    set still means ANY — so each row is reportable for owner review:
    re-issue the grant with an explicit ``PurposeConstraint`` and revoke
    the legacy row. Returns dict snapshots; ``live`` flags rows that
    still authorize today.
    """
    now = now_us()
    out: list[dict] = []
    for row in repos_v3.query(conn, _GRANTS, order="issued_us"):
        tag = row.get("purpose_tag")
        if tag is not None and tag != LEGACY_EMPTY_TAG:
            continue
        if not include_revoked and row.get("revoked_us") is not None:
            continue
        try:
            purposes = sorted(str(p) for p in _json_set(row, "purposes_json"))
            malformed = False
        except VerbatimError:
            purposes = []
            malformed = True
        constraint = _row_constraint(row)
        out.append(
            {
                "grant_id": row["grant_id"],
                "scope_id": row["scope_id"],
                "principal_id": row["principal_id"],
                "issuer_id": row["issuer_id"],
                "issued_us": row["issued_us"],
                "revoked_us": row.get("revoked_us"),
                "purpose_tag": tag if tag is not None else LEGACY_EMPTY_TAG,
                "purposes": purposes,
                "malformed_purposes_json": malformed,
                # what the evaluator actually does with this row today
                "effective_purposes": (
                    "denied" if constraint is None
                    else constraint.tag.value
                ),
                "live": _row_live(row, now),
                "remediation": (
                    "ambiguous pre-v4 purpose encoding (empty set "
                    "evaluates as ANY); re-issue with an explicit "
                    "PurposeConstraint and revoke this grant"
                ),
            }
        )
    return out


def get_grant(conn: sqlite3.Connection, grant_id: str) -> Optional[dict]:
    """Row snapshot or None."""
    require_id(grant_id, "grant_id")
    return repos_v3.get(conn, _GRANTS, {"grant_id": grant_id})


def grants_for(
    conn: sqlite3.Connection,
    scope_id: str,
    principal_id: str,
    *,
    include_revoked: bool = False,
) -> list[dict]:
    """Grant rows for (scope, principal); revoked rows hidden by default."""
    require_id(scope_id, "scope_id")
    require_id(principal_id, "principal_id")
    where: dict = {"scope_id": scope_id, "principal_id": principal_id}
    if not include_revoked:
        where["revoked_us"] = None
    return repos_v3.query(conn, _GRANTS, where, order="issued_us")


def grant_object(row: dict) -> Grant:
    """Rebuild the frozen Grant contract from a stored row."""
    return Grant(
        grant_id=row["grant_id"],
        scope_id=row["scope_id"],
        principal_id=row["principal_id"],
        verbs=frozenset(_json_set(row, "verbs_json")),
        purposes=frozenset(_json_set(row, "purposes_json")),
        issuer_id=row["issuer_id"],
        issued_us=int(row["issued_us"]),
        expires_us=row.get("expires_us"),
        delegation_depth=int(row.get("delegation_depth") or 0),
        caveats=tuple(repos_v3.json_field(row, "caveats_json") or ()),
        revoked_us=row.get("revoked_us"),
        epoch=int(row.get("epoch") or 0),
    )


def _persist_grant(conn: sqlite3.Connection, row: dict, tag: str) -> None:
    """Write one grant row including its v4 ``purpose_tag``.

    ``purpose_tag`` is the schema-v4 ALTER column and deliberately sits
    outside the ``repos_v3`` v3-column allowlist — it is written with one
    parameterized statement (same pattern ``epochs.py`` uses for the v1
    ``scopes`` table), inside the caller's transaction, so a row is never
    observable untagged.
    """
    repos_v3.insert(conn, _GRANTS, row)
    conn.execute(
        "UPDATE grants_v3 SET purpose_tag = ? WHERE grant_id = ?",
        (tag, row["grant_id"]),
    )


def create_grant(
    conn: sqlite3.Connection,
    *,
    scope_id: str,
    principal_id: str,
    verbs: Iterable["Verb | str"],
    issuer_id: str,
    purposes: "PurposeConstraint | Iterable[str] | None" = None,
    caveats: Iterable[str] = (),
    expires_us: Optional[int] = None,
    delegation_depth: int = 0,
    grant_id: Optional[str] = None,
    issued_us: Optional[int] = None,
    epoch: Optional[int] = None,
    strict: bool = True,
) -> str:
    """Issue a scope-bound grant; returns grant_id.

    ``purposes`` is a tagged constraint surface (V4-11.01): pass a
    ``PurposeConstraint`` or an iterable of purpose names. An iterable's
    empty form is an explicit empty SET → the grant persists tag ``none``
    and authorizes no purpose. ``purposes=None`` (the v3 default) issues
    an explicitly ``any``-tagged grant — unrestricted by purpose, now
    unambiguous in storage; new code should say
    ``PurposeConstraint.any()`` explicitly.

    ``strict`` is the ``governance_strict`` surface (§11.06): every
    declared purpose must already be a live registry entry. The grant's
    ``epoch`` stamps the scope's current ``authz_revision`` so a pinned
    caller never sees authority that postdates its bound view.
    """
    require_id(scope_id, "scope_id")
    require_id(principal_id, "principal_id")
    require_id(issuer_id, "issuer_id")
    vset = _verb_set(verbs)
    constraint = _coerce_purposes(purposes, default=PurposeConstraint.any())
    if strict:
        for p in sorted(constraint.values):
            purposes_registry.require_purpose(conn, p)
    cav = tuple(caveats)
    for c in cav:
        if not isinstance(c, str) or not c:
            raise VerbatimError(
                ErrorCode.VALIDATION, "caveats are non-empty strings"
            )
    depth = _finite_int(delegation_depth, "delegation_depth")
    if not 0 <= depth <= MAX_DELEGATION_DEPTH:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"delegation_depth must be in [0, {MAX_DELEGATION_DEPTH}]",
        )
    issued = _opt_finite_int(issued_us, "issued_us")
    issued = issued if issued is not None else now_us()
    expires = _opt_finite_int(expires_us, "expires_us")
    ep = _opt_finite_int(epoch, "epoch")
    ep = ep if ep is not None else epochs.current_epoch(conn, scope_id)
    if ep < 0:
        raise VerbatimError(ErrorCode.VALIDATION, "epoch must be >= 0")
    gid = grant_id or f"grant:{new_id()}"
    # The frozen contract validates ids, expiry ordering, and depth.
    Grant(
        grant_id=gid,
        scope_id=scope_id,
        principal_id=principal_id,
        verbs=vset,
        purposes=constraint.values,
        issuer_id=issuer_id,
        issued_us=issued,
        expires_us=expires,
        delegation_depth=depth,
        caveats=cav,
        epoch=ep,
    )
    _persist_grant(
        conn,
        {
            "grant_id": gid,
            "scope_id": scope_id,
            "principal_id": principal_id,
            "verbs_json": sorted(v.value for v in vset),
            "purposes_json": sorted(constraint.values),
            "caveats_json": list(cav),
            "delegation_depth": depth,
            "issuer_id": issuer_id,
            "issued_us": issued,
            "expires_us": expires,
            "revoked_us": None,
            "epoch": ep,
        },
        constraint.tag.value,
    )
    return gid


def delegate_grant(
    conn: sqlite3.Connection,
    *,
    parent_grant_id: str,
    delegate_id: str,
    verbs: Optional[Iterable["Verb | str"]] = None,
    purposes: "PurposeConstraint | Iterable[str] | None" = None,
    caveats: Optional[Iterable[str]] = None,
    expires_us: Optional[int] = None,
    delegator_id: Optional[str] = None,
    delegation_id: Optional[str] = None,
    child_grant_id: Optional[str] = None,
    created_us: Optional[int] = None,
    strict: bool = True,
) -> "tuple[str, str]":
    """Delegate an attenuated child grant; returns (delegation_id, grant_id).

    The parent must be live, carry remaining delegation depth, and hold
    *effective* authority — a delegated parent whose own chain no longer
    reaches a live root (revoked edge, cycle, excessive depth) is dead
    authority and denies identically to an absent grant. The child may
    only narrow (V4-11.02): verbs ⊆ parent's, purpose constraint
    ``is_subset_of`` the parent's (so an explicit empty iterable yields a
    NONE-purpose child — C09 — and an ANY child requires an ANY parent),
    expiry ≤ the parent's, depth = parent − 1, scope inherited verbatim.
    Caveats are conjunctive (V4-11.03): the child must retain every
    parent caveat and may only add more — dropping one is an authority
    expansion, rejected. Subset violations are caller bugs (VALIDATION);
    a missing, dead, or depth-exhausted parent denies identically to any
    absent authority (NOT_FOUND_OR_UNAUTHORIZED).
    """
    require_id(parent_grant_id, "parent_grant_id")
    require_id(delegate_id, "delegate_id")
    parent = repos_v3.get(conn, _GRANTS, {"grant_id": parent_grant_id})
    now = now_us()
    if parent is None or not _row_live(parent, now):
        _deny()
    try:
        depth = int(parent.get("delegation_depth") or 0)
    except (TypeError, ValueError):
        _deny()  # malformed depth — dead authority
    if depth <= 0:
        # No remaining delegation budget — the parent itself denies.
        _deny()
    if not _is_root_grant(conn, parent) and not _chain_live(
        conn, parent, now, epochs.current_epoch(conn, parent["scope_id"])
    ):
        # Dead intermediate: a delegated parent whose chain broke (or
        # cycles back on itself) holds no effective authority to pass on.
        _deny()
    try:
        pverbs = frozenset(
            _verb(v) for v in _json_set(parent, "verbs_json")
        )
    except VerbatimError:
        _deny()  # malformed stored verb set — dead authority
    parent_constraint = _row_constraint(parent)
    if parent_constraint is None:
        # Malformed persisted constraint — dead authority, deny closed.
        _deny()
    try:
        pcaveats = frozenset(
            repos_v3.json_field(parent, "caveats_json") or ()
        )
    except VerbatimError:
        _deny()  # malformed stored caveats — dead authority
    pexp = parent.get("expires_us")

    child_verbs = (
        _verb_set(verbs) if verbs is not None else pverbs
    )
    if not child_verbs <= pverbs:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "delegation may only attenuate verbs (child ⊄ parent)",
        )
    child_constraint = _coerce_purposes(purposes, default=parent_constraint)
    if not child_constraint.is_subset_of(parent_constraint):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "delegation may only attenuate purposes (child ⊄ parent)",
        )
    child_caveats = (
        tuple(caveats) if caveats is not None else tuple(pcaveats)
    )
    for c in child_caveats:
        if not isinstance(c, str) or not c:
            raise VerbatimError(
                ErrorCode.VALIDATION, "caveats are non-empty strings"
            )
    if not pcaveats <= frozenset(child_caveats):
        # V4-11.03: caveats are conjunctive — the child must keep every
        # parent restriction (adding more strengthens, and is allowed).
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "delegation must retain parent caveats (child dropped "
            "a restriction)",
        )
    expires = _opt_finite_int(expires_us, "expires_us")
    if pexp is not None:
        if expires is None:
            expires = pexp  # inherit the bound — never widen
        elif expires > pexp:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "delegated expiry exceeds the parent grant's",
            )

    # V4-11.09 cycle/malformed rejection: the child must be a *new* grant.
    # Reusing an existing grant id — in particular the parent itself or
    # any ancestor on its chain — would close a delegation cycle and kill
    # the whole chain at evaluation; a fresh minted id can never collide.
    if child_grant_id is not None:
        require_id(child_grant_id, "child_grant_id")
        if (
            repos_v3.get(conn, _GRANTS, {"grant_id": child_grant_id})
            is not None
        ):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "child_grant_id already exists — delegation edges "
                "require a fresh child grant",
            )
    did = delegation_id or f"dlg:{new_id()}"
    require_id(did, "delegation_id")
    if (
        repos_v3.get(conn, _DELEGATIONS, {"delegation_id": did})
        is not None
    ):
        raise VerbatimError(
            ErrorCode.VALIDATION, "delegation_id already exists"
        )
    created = _opt_finite_int(created_us, "created_us")

    gid = create_grant(
        conn,
        scope_id=parent["scope_id"],
        principal_id=delegate_id,
        verbs=child_verbs,
        purposes=child_constraint,
        caveats=child_caveats,
        expires_us=expires,
        issuer_id=delegator_id or parent["issuer_id"],
        delegation_depth=depth - 1,
        grant_id=child_grant_id,
        strict=strict,
    )
    repos_v3.insert(
        conn,
        _DELEGATIONS,
        {
            "delegation_id": did,
            "parent_grant_id": parent_grant_id,
            "child_grant_id": gid,
            "delegator_id": delegator_id or parent["principal_id"],
            "delegate_id": delegate_id,
            "created_us": created if created is not None else now,
            "expires_us": expires,
            "revoked_us": None,
        },
    )
    return did, gid


def _has_delegation_edges(conn: sqlite3.Connection, grant_id: str) -> bool:
    """Any delegation row names this grant as its child."""
    return (
        repos_v3.get(conn, _DELEGATIONS, {"child_grant_id": grant_id})
        is not None
    )


def _is_root_grant(conn: sqlite3.Connection, row: dict) -> bool:
    """A grant needing no delegation chain: no delegation edge into it.

    ``delegation_depth`` is a *budget for further delegation* (§09.03), not
    a chain marker — a root grant legitimately carries depth > 0 so it may
    delegate. What makes a grant delegated is an edge naming it as child.
    """
    return not _has_delegation_edges(conn, row["grant_id"])


def _chain_live(
    conn: sqlite3.Connection,
    grant_row: dict,
    now: int,
    pinned_epoch: int,
    _depth: int = 0,
    _seen: "frozenset[str] | None" = None,
) -> bool:
    """True when a live delegation path reaches a live root grant.

    Every edge must be unrevoked and unexpired, every ancestor grant live
    and issued at/below the pinned epoch. The walk ends at a true root —
    a live grant with no delegation edges into it — so a delegated grant
    whose last edge died is dead authority even when its own row is
    otherwise live. ``_seen`` carries the current path's grant ids: a
    delegation cycle never reaches a root and never terminates the walk
    early for a *sibling* path — diamonds stay evaluable, cycles die
    (V4-11.09).
    """
    if _depth > _MAX_CHAIN:
        return False
    seen = _seen if _seen is not None else frozenset()
    gid = grant_row["grant_id"]
    if gid in seen:
        return False  # this path loops — dead
    seen = seen | {gid}
    links = repos_v3.query(
        conn,
        _DELEGATIONS,
        {"child_grant_id": gid, "revoked_us": None},
    )
    for link in links:
        lexp = link.get("expires_us")
        if lexp is not None:
            if (
                isinstance(lexp, bool)
                or not isinstance(lexp, (int, float))
                or (isinstance(lexp, float) and not math.isfinite(lexp))
                or lexp <= now
            ):
                continue  # expired or malformed edge — dead
        parent = repos_v3.get(
            conn, _GRANTS, {"grant_id": link["parent_grant_id"]}
        )
        if parent is None or not _row_live(parent, now):
            continue
        pepoch = _row_epoch(parent)
        if pepoch is None or pepoch > pinned_epoch:
            continue
        if _is_root_grant(conn, parent):
            return True
        if _chain_live(conn, parent, now, pinned_epoch, _depth + 1, seen):
            return True
    return False


def _grant_applies(
    conn: sqlite3.Connection,
    row: dict,
    verb: Verb,
    purpose: Optional[str],
    now: int,
    pinned_epoch: int,
) -> bool:
    """One row's effective-permit check (§09.01 permit evaluation)."""
    row_epoch = _row_epoch(row)
    if row_epoch is None or row_epoch > pinned_epoch:
        return False  # malformed epoch, or issued after the pinned view
    try:
        verbs = _json_set(row, "verbs_json")
    except VerbatimError:
        return False  # malformed stored verb set — dead row, not a crash
    if verb.value not in verbs:
        return False
    constraint = _row_constraint(row)
    if constraint is None or not constraint.permits(purpose):
        # Tagged purpose limitation (V4-11.01/§09.08): SET answers only
        # its listed purposes, NONE none at all, and a malformed
        # persisted constraint fails closed rather than reading as ANY.
        return False
    if not _is_root_grant(conn, row) and not _chain_live(
        conn, row, now, pinned_epoch
    ):
        return False
    return True


# ---------------------------------------------------------------------------
# snapshot-bounded verdict memo (read-path hot loop)
# ---------------------------------------------------------------------------
#
# ``authorize``/``effective_verbs`` verdicts are pure functions of the read
# snapshot's grant/principal/epoch rows plus the wall clock: rows only
# change through commits, and a verdict can only flip *passively* when a
# declared ``expires_us`` bound passes — dead grants never revive, so a
# denial is stable for the whole snapshot while an allow lives until its
# covering grants could expire. A memo keyed by ``(id(conn),
# data_version)`` is therefore exact on ``PRAGMA query_only`` reader
# connections: the cookie freezes inside a read transaction and bumps on
# every other connection's commit, and a query_only conn can never write,
# so identical keys always see identical table content. Writer
# connections are excluded — their own uncommitted writes would change
# the visible rows without moving the cookie. The pinned strong conn
# reference keeps a live ``id`` from being recycled while its bucket
# exists; ``not_after`` bounds every entry by the earliest instant the
# verdict could change without a write (``None`` = snapshot-stable).

_SNAP_MEMO_LOCK = threading.Lock()
_snap_buckets: "OrderedDict[tuple, OrderedDict]" = OrderedDict()
_snap_conns: "OrderedDict[int, sqlite3.Connection]" = OrderedDict()
_SNAP_CONNS_MAX = 16
_SNAP_BUCKETS_MAX = 64
_SNAP_BUCKET_MAX = 256
_MISS = object()


def _snapshot_key(conn: sqlite3.Connection) -> Optional[tuple]:
    """``(id(conn), data_version)`` for query_only reader conns."""
    try:
        row = conn.execute("PRAGMA query_only").fetchone()
        if not row or not row[0]:
            return None
        row = conn.execute("PRAGMA data_version").fetchone()
        if not row:
            return None
        return (id(conn), row[0])
    except sqlite3.Error:
        return None


def _snap_get(skey: tuple, mkey: tuple) -> Any:
    """Memoized verdict valid at ``now`` or ``_MISS``."""
    with _SNAP_MEMO_LOCK:
        bucket = _snap_buckets.get(skey)
        if bucket is None:
            return _MISS
        ent = bucket.get(mkey)
        if ent is None:
            return _MISS
        value, not_after = ent
        if not_after is not None and now_us() >= not_after:
            # Passive flip boundary crossed — recompute rather than
            # replay a verdict that may have expired.
            try:
                del bucket[mkey]
            except KeyError:
                pass
            return _MISS
        bucket.move_to_end(mkey)
        return value


def _snap_put(skey: tuple, conn: sqlite3.Connection, mkey: tuple,
              value: Any, not_after: Optional[float]) -> None:
    with _SNAP_MEMO_LOCK:
        _snap_conns[id(conn)] = conn
        _snap_conns.move_to_end(id(conn))
        while len(_snap_conns) > _SNAP_CONNS_MAX:
            old_id, _old_conn = _snap_conns.popitem(last=False)
            # The conn reference kept its id unique; dropping it frees
            # the id for reuse, so every bucket keyed to it must go too.
            for k in [k for k in _snap_buckets if k[0] == old_id]:
                _snap_buckets.pop(k, None)
        bucket = _snap_buckets.get(skey)
        if bucket is None:
            bucket = OrderedDict()
            _snap_buckets[skey] = bucket
        bucket[mkey] = (value, not_after)
        bucket.move_to_end(mkey)
        while len(bucket) > _SNAP_BUCKET_MAX:
            bucket.popitem(last=False)
        while len(_snap_buckets) > _SNAP_BUCKETS_MAX:
            _snap_buckets.popitem(last=False)


def _finite_expiry(row: dict) -> Optional[float]:
    """The row's usable ``expires_us`` or ``None`` (malformed rows are
    dead under ``_row_live`` — they never bound a live verdict)."""
    exp = row.get("expires_us")
    if exp is None:
        return None
    if isinstance(exp, bool) or not isinstance(exp, (int, float)):
        return None
    if isinstance(exp, float) and not math.isfinite(exp):
        return None
    return float(exp)


def authorize(
    conn: sqlite3.Connection,
    caller: "CallerV3",
    scope_id: str,
    verb: str,
    purpose: Optional[str] = None,
    object_ref: Optional["tuple[str, str, int]"] = None,
) -> None:
    """Permit only when an effective grant covers (scope, verb[, purpose]).

    Raises ``NOT_FOUND_OR_UNAUTHORIZED`` on denial — absent and forbidden
    are publicly indistinguishable (§09.09, §10.05). ``STALE_EPOCH`` fires
    first when the caller's pinned epoch was superseded by a revocation.
    ``object_ref`` is deliberately not dereferenced: authorization precedes
    any private-identifier lookup, so existence never leaks through the
    decision (§09.09). Returns None on success.
    """
    require_id(caller.principal_id, "principal_id")
    require_id(scope_id, "scope_id")
    v = _verb(verb)
    if purpose is not None and not isinstance(purpose, str):
        raise VerbatimError(ErrorCode.VALIDATION, "purpose must be a string")
    if object_ref is not None:
        try:
            ok = len(object_ref) == 3
        except TypeError:
            ok = False
        if not ok:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "object_ref must be (object_kind, object_id, revision)",
            )
    # Epoch fencing runs on every call — a stale pin raises identically
    # whether or not the verdict below is memoized.
    epochs.check_epoch(conn, caller.epoch, scope_id)
    skey = _snapshot_key(conn)
    mkey: Optional[tuple] = None
    if skey is not None:
        try:
            mkey = (
                "authorize", caller.principal_id, caller.epoch,
                scope_id, v.value, purpose,
            )
            hash(mkey)
        except Exception:
            mkey = None
    if mkey is not None:
        hit = _snap_get(skey, mkey)
        if hit is not _MISS:
            if hit == "allow":
                return None
            _deny()  # memoized deny — snapshot-stable (see header)
    now = now_us()
    pinned = (
        caller.epoch
        if caller.epoch is not None
        else epochs.current_epoch(conn, scope_id)
    )
    if _principal_retired(conn, caller.principal_id):
        if mkey is not None:
            _snap_put(skey, conn, mkey, "deny", None)
        _deny()
    rows = repos_v3.query(
        conn,
        _GRANTS,
        {
            "scope_id": scope_id,
            "principal_id": caller.principal_id,
            "revoked_us": None,
        },
    )
    if mkey is None:
        # Memoless path (writer conns, opaque handles): the historical
        # first-match early exit, unchanged.
        for row in rows:
            if not _row_live(row, now):
                continue
            if _grant_applies(conn, row, v, purpose, now, pinned):
                return None
        _deny()
    # Memo path: every covering row is evaluated so the earliest passive
    # flip instant can be recorded — the verdict is served only while no
    # covering grant's expiry has passed. Coverage through a delegation
    # chain carries edge expiries this pass cannot bound, so a non-root
    # cover is served live but never memoized.
    # ``bound`` is the LAST passive death among covering root grants —
    # the verdict stays allow while any one cover survives, so it can
    # only flip at the latest covering expiry. A never-expiring root
    # cover (``eternal``) means no passive flip exists at all: the
    # verdict is then stable for the whole snapshot (``bound=None``).
    bound: Optional[float] = None
    allowed = False
    eternal = False
    memoizable = True
    for row in rows:
        if not _row_live(row, now):
            continue
        if not _grant_applies(conn, row, v, purpose, now, pinned):
            continue
        allowed = True
        if _is_root_grant(conn, row):
            exp = _finite_expiry(row)
            if exp is None:
                eternal = True
            elif bound is None or exp > bound:
                bound = exp
        else:
            memoizable = False
    if not allowed:
        _snap_put(skey, conn, mkey, "deny", None)
        _deny()
    if memoizable:
        _snap_put(skey, conn, mkey, "allow", None if eternal else bound)
    return None


def effective_verbs(
    conn: sqlite3.Connection, caller: "CallerV3", scope_id: str
) -> frozenset:
    """Union of verbs over the caller's currently effective grants.

    Diagnostic only (V4-11.04): the union is purpose-blind — a NONE-purpose
    grant still contributes its verbs here while ``authorize`` keeps
    denying it. Same epoch fencing as ``authorize`` — a stale pin raises
    ``STALE_EPOCH`` rather than answering from a superseded view. A retired
    principal has no effective verbs (empty set, not an error).
    """
    require_id(caller.principal_id, "principal_id")
    require_id(scope_id, "scope_id")
    epochs.check_epoch(conn, caller.epoch, scope_id)
    skey = _snapshot_key(conn)
    mkey: Optional[tuple] = None
    if skey is not None:
        try:
            mkey = ("effective_verbs", caller.principal_id,
                    caller.epoch, scope_id)
            hash(mkey)
        except Exception:
            mkey = None
    if mkey is not None:
        hit = _snap_get(skey, mkey)
        if hit is not _MISS:
            return hit
    now = now_us()
    pinned = (
        caller.epoch
        if caller.epoch is not None
        else epochs.current_epoch(conn, scope_id)
    )
    if _principal_retired(conn, caller.principal_id):
        out_empty = frozenset()
        if mkey is not None:
            _snap_put(skey, conn, mkey, out_empty, None)
        return out_empty
    out: set[str] = set()
    rows = repos_v3.query(
        conn,
        _GRANTS,
        {
            "scope_id": scope_id,
            "principal_id": caller.principal_id,
            "revoked_us": None,
        },
    )
    bound: Optional[float] = None
    memoizable = True
    for row in rows:
        if not _row_live(row, now):
            continue
        row_epoch = _row_epoch(row)
        if row_epoch is None or row_epoch > pinned:
            continue
        is_root = _is_root_grant(conn, row)
        if not is_root and not _chain_live(
            conn, row, now, pinned
        ):
            continue
        # This row contributes to the union; its expiry is a passive
        # flip bound (the union can only shrink without a write), and a
        # delegation chain's own edge expiries cannot be bounded here —
        # chained contributors keep the union unmemoized.
        exp = _finite_expiry(row)
        if exp is not None and (bound is None or exp < bound):
            bound = exp
        if not is_root:
            memoizable = False
        try:
            out |= set(_json_set(row, "verbs_json"))
        except VerbatimError:
            continue  # malformed stored verb set — contributes nothing
    result = frozenset(out)
    if mkey is not None and memoizable:
        _snap_put(skey, conn, mkey, result, bound)
    return result
    return frozenset(out)
