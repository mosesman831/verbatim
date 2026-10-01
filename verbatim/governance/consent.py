"""Consent surfaces (SPEC_V3 §11).

Two independent records live here (§11.11: tool permission, retention
consent, derivation permission, and disclosure consent are separate
decisions):

- ``capture_authorizations`` — host/operator-issued retention consent for a
  source class: is (principal, envelope_kind, scope, retention_policy)
  authorized for durable capture? An approved tool call is NOT this record.
- ``consents`` (v1 table + v3 columns) — disclosure consent for hydration:
  a request's purpose must match a live, unexpired consent row (§11.07);
  absence/expiry/withdrawal raises ``CONSENT_REQUIRED`` (§46).
"""

from __future__ import annotations

import sqlite3
from typing import Iterable, Optional

from ..core.time import now_us
from ..core.types import ErrorCode, VerbatimError, new_id, require_id
from ..core.types_v3 import EnvelopeKind
from ..storage import repos_v3
from . import epochs

_AUTHS = "capture_authorizations"


def _kind(value: "EnvelopeKind | str") -> EnvelopeKind:
    try:
        return value if isinstance(value, EnvelopeKind) else EnvelopeKind(value)
    except ValueError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown envelope kind {value!r}"
        ) from exc


def issue_capture_authorization(
    conn: sqlite3.Connection,
    *,
    principal_id: str,
    issuer_id: str,
    allowed_kinds: Iterable["EnvelopeKind | str"],
    retention_policy: str,
    policy_revision: str,
    scope_ids: Iterable[str] = (),
    issued_us: Optional[int] = None,
    expires_us: Optional[int] = None,
    authorization_id: Optional[str] = None,
) -> str:
    """Record a capture authorization; returns authorization_id.

    ``scope_ids`` empty means unconstrained across scopes — an explicit
    operator choice, never an inferred default.
    """
    require_id(principal_id, "principal_id")
    require_id(issuer_id, "issuer_id")
    kinds = frozenset(_kind(k) for k in allowed_kinds)
    if not kinds:
        raise VerbatimError(
            ErrorCode.VALIDATION, "authorization requires >= 1 envelope kind"
        )
    if not retention_policy:
        raise VerbatimError(ErrorCode.VALIDATION, "retention_policy required")
    if not policy_revision:
        raise VerbatimError(ErrorCode.VALIDATION, "policy_revision required")
    scopes = sorted(require_id(s, "scope_id") for s in scope_ids)
    aid = authorization_id or f"cauth:{new_id()}"
    require_id(aid, "authorization_id")
    issued = issued_us if issued_us is not None else now_us()
    if expires_us is not None and expires_us <= issued:
        raise VerbatimError(
            ErrorCode.VALIDATION, "authorization expiry precedes issue"
        )
    repos_v3.insert(
        conn,
        _AUTHS,
        {
            "authorization_id": aid,
            "issuer_id": issuer_id,
            "principal_id": principal_id,
            "allowed_kinds_json": sorted(k.value for k in kinds),
            "scope_ids_json": scopes,
            "retention_policy": retention_policy,
            "policy_revision": policy_revision,
            "issued_us": issued,
            "expires_us": expires_us,
            "revoked_us": None,
        },
    )
    return aid


def get_capture_authorization(
    conn: sqlite3.Connection, authorization_id: str
) -> Optional[dict]:
    require_id(authorization_id, "authorization_id")
    return repos_v3.get(conn, _AUTHS, {"authorization_id": authorization_id})


def capture_authorized(
    conn: sqlite3.Connection,
    principal_id: str,
    envelope_kind: "EnvelopeKind | str",
    scope_id: str,
    *,
    retention_policy: Optional[str] = None,
    now: Optional[int] = None,
) -> bool:
    """Whether durable capture of ``envelope_kind`` is consented for the
    principal in ``scope_id``.

    A live, unexpired, unrevoked authorization must name the kind, cover the
    scope (empty ``scope_ids`` = unconstrained), and — when the caller
    asserts one — carry the same retention policy (§11.11). Kind not in
    ``allowed_kinds_json``, expiry, revocation, and scope misses all deny
    identically: False.
    """
    require_id(principal_id, "principal_id")
    require_id(scope_id, "scope_id")
    kind = _kind(envelope_kind)
    ts = now if now is not None else now_us()
    rows = repos_v3.query(
        conn, _AUTHS, {"principal_id": principal_id, "revoked_us": None}
    )
    for row in rows:
        exp = row.get("expires_us")
        if exp is not None and exp <= ts:
            continue
        if kind.value not in (
            repos_v3.json_field(row, "allowed_kinds_json") or ()
        ):
            continue
        scopes = repos_v3.json_field(row, "scope_ids_json") or []
        if scopes and scope_id not in scopes:
            continue
        if (
            retention_policy is not None
            and row.get("retention_policy") != retention_policy
        ):
            continue
        return True
    return False


def require_capture_authorization(
    conn: sqlite3.Connection,
    principal_id: str,
    envelope_kind: "EnvelopeKind | str",
    scope_id: str,
    *,
    retention_policy: Optional[str] = None,
    now: Optional[int] = None,
) -> None:
    """Typed denial for gated capture (§12.10): missing consent raises
    ``CONSENT_REQUIRED`` — never a silent downgrade to another kind."""
    if not capture_authorized(
        conn,
        principal_id,
        envelope_kind,
        scope_id,
        retention_policy=retention_policy,
        now=now,
    ):
        raise VerbatimError(
            ErrorCode.CONSENT_REQUIRED,
            "capture lacks a matching authorization record",
        )


def revoke_capture_authorization(
    conn: sqlite3.Connection,
    authorization_id: str,
    *,
    revoked_us: Optional[int] = None,
) -> bool:
    """Withdraw a capture authorization; bumps covered scope epochs so
    pinned callers fence (§09.04). Idempotent — returns False when the row
    is absent or already revoked."""
    require_id(authorization_id, "authorization_id")
    row = repos_v3.get(conn, _AUTHS, {"authorization_id": authorization_id})
    if row is None or row.get("revoked_us") is not None:
        return False
    repos_v3.update(
        conn,
        _AUTHS,
        {"revoked_us": revoked_us if revoked_us is not None else now_us()},
        {"authorization_id": authorization_id},
    )
    for sid in repos_v3.json_field(row, "scope_ids_json") or ():
        epochs.bump_epoch_if_present(conn, sid)
    return True


# ---------------------------------------------------------------------------
# Disclosure consent (consents v1 table + §11 v3 columns) — explicit SQL:
# ``consents`` predates repos_v3 and is not in its allowlist.
# ---------------------------------------------------------------------------


def consent_for_hydration(
    conn: sqlite3.Connection,
    scope_id: str,
    purpose: str,
    *,
    processor: Optional[str] = None,
    now: Optional[int] = None,
) -> Optional[dict]:
    """The live disclosure-consent row matching (scope, purpose[, processor]).

    Hydration requires purpose-matched, unexpired, unrevoked consent
    (§11.07); each match is a disclosure decision, so this returns the
    newest live row or None — never a guess.
    """
    require_id(scope_id, "scope_id")
    if not purpose:
        raise VerbatimError(ErrorCode.VALIDATION, "purpose required")
    ts = now if now is not None else now_us()
    sql = (
        "SELECT consent_id, scope_id, processor, purpose, granted_us,"
        " revoked_us, expires_us, data_classes_json, sanitization,"
        " retention_promise, budget_microusd, policy_digest"
        " FROM consents"
        " WHERE scope_id = ? AND purpose = ? AND revoked_us IS NULL"
    )
    params: list = [scope_id, purpose]
    if processor is not None:
        sql += " AND processor = ?"
        params.append(processor)
    sql += " ORDER BY granted_us DESC"
    cols = [
        "consent_id", "scope_id", "processor", "purpose", "granted_us",
        "revoked_us", "expires_us", "data_classes_json", "sanitization",
        "retention_promise", "budget_microusd", "policy_digest",
    ]
    for raw in conn.execute(sql, params).fetchall():
        row = dict(zip(cols, raw))
        exp = row.get("expires_us")
        if exp is not None and exp <= ts:
            continue
        return row
    return None


def require_hydration_consent(
    conn: sqlite3.Connection,
    scope_id: str,
    purpose: str,
    *,
    processor: Optional[str] = None,
    now: Optional[int] = None,
) -> dict:
    """Purpose-matched consent or ``CONSENT_REQUIRED`` (§11.07, §46).

    Absent, expired, and withdrawn consent are the same public failure —
    a hydration request never learns which was missing.
    """
    row = consent_for_hydration(
        conn, scope_id, purpose, processor=processor, now=now
    )
    if row is None:
        raise VerbatimError(
            ErrorCode.CONSENT_REQUIRED,
            "hydration lacks a matching consent record",
        )
    return row
