"""Vault hydration: consent-gated exact-value delivery (SPEC_V3 §35.04,
§11.07, §11.12, §10.04).

The §35.04 flow in one place:

1. The entry must exist and not be erased — absent and erased are
   indistinguishable (``NOT_FOUND_OR_UNAUTHORIZED``, §09.09/§36.03).
2. A live consent row must cover the request: same scope, same declared
   purpose, unrevoked, unexpired, and covering the entry's data class
   (§11.01, §11.07). Anything short is ``CONSENT_REQUIRED``.
3. ``downstream_processor`` must be declared and must equal
   ``consent.processor`` — unknown downstream processing denies hydration
   rather than guessing locality (§11.12). The only relaxation is the
   explicit ``v3.vault.allow_plaintext_hydration`` escape.
4. The caller's pinned epoch must equal the scope's ``authz_revision`` —
   a superseded pin raises ``STALE_EPOCH`` (§09.04).
5. Delivery: plaintext bytes only to a local-trusted processor
   (``operator``/``local_model``) or when ``allow_plaintext_hydration``;
   every other declared processor receives an opaque one-use
   :class:`ValueHandle` bound to recipient + action digest + consent +
   expiry — model-visible answers carry handles, not plaintext (§35.04).

Every successful hydration and every handle redemption is recorded in the
propagation ledger with epoch and purpose so blast radius is computable
(§10.04).
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..config import VerbatimConfig
from ..core.time import now_us
from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    new_id,
    require_id,
    safe_json_loads,
)
from ..storage import repos_v3
from . import vault as _vault

# Processors the engine can hand plaintext to without an opaque handle:
# an authenticated operator or an approved local model (§35.04). Everything
# else — task-tool brokers, named cloud models, unrecognized processors —
# gets a one-use handle at most.
LOCAL_TRUSTED_PROCESSORS = frozenset({"operator", "local_model"})


@dataclass(frozen=True)
class HydrationResult:
    """What ``hydrate`` produced: either plaintext bytes for a
    local-trusted processor or an opaque handle for a brokered one."""

    kind: str  # "plaintext" | "handle"
    entry_id: str
    propagation_id: str
    value: Optional[bytes] = None
    handle_id: Optional[str] = None
    expires_us: Optional[int] = None
    action_digest: Optional[bytes] = None


def _consent_row(conn: sqlite3.Connection, consent_id: str) -> Optional[dict[str, Any]]:
    cur = conn.execute(
        "SELECT * FROM consents WHERE consent_id = ?", (consent_id,)
    )
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    return dict(zip(cols, row)) if row is not None else None


def _consent_covers(
    conn: sqlite3.Connection,
    entry: dict[str, Any],
    consent_id: str,
    purpose: str,
    now: int,
) -> dict[str, Any]:
    """Return the consent row or raise CONSENT_REQUIRED (§11.07)."""
    row = _consent_row(conn, consent_id)
    if row is None:
        raise VerbatimError(
            ErrorCode.CONSENT_REQUIRED, "no consent record for hydration"
        )
    if row["scope_id"] != entry["scope_id"]:
        raise VerbatimError(
            ErrorCode.CONSENT_REQUIRED, "consent does not cover this scope"
        )
    if row.get("revoked_us") is not None:
        raise VerbatimError(
            ErrorCode.CONSENT_REQUIRED, "consent has been revoked"
        )
    expires = row.get("expires_us")
    if expires is not None and int(expires) <= now:
        raise VerbatimError(
            ErrorCode.CONSENT_REQUIRED, "consent has expired"
        )
    if row["purpose"] != purpose:
        raise VerbatimError(
            ErrorCode.CONSENT_REQUIRED,
            "consent purpose does not match the declared purpose",
        )
    classes = safe_json_loads(row.get("data_classes_json") or "[]")
    if not isinstance(classes, list):
        classes = []
    if entry["sensitivity"] not in classes and "*" not in classes:
        raise VerbatimError(
            ErrorCode.CONSENT_REQUIRED,
            "consent does not cover this data class",
        )
    return row


def _purpose_registered(conn: sqlite3.Connection, purpose: str) -> None:
    """§11.06: purposes come from a registry — free text is rejected. A
    registry is enforced only once populated (empty table = unconfigured)."""
    total = conn.execute("SELECT COUNT(*) FROM purposes").fetchone()[0]
    if total == 0:
        return
    row = conn.execute(
        "SELECT retired FROM purposes WHERE purpose = ?", (purpose,)
    ).fetchone()
    if row is None or row[0]:
        raise VerbatimError(
            ErrorCode.CONSENT_REQUIRED,
            "purpose is not in the registered purpose registry",
        )


def _current_epoch(conn: sqlite3.Connection, scope_id: str) -> int:
    row = conn.execute(
        "SELECT authz_revision FROM scopes WHERE scope_id = ?", (scope_id,)
    ).fetchone()
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "scope unavailable"
        )
    return int(row[0])


def _check_epoch(conn: sqlite3.Connection, scope_id: str,
                 epoch: Optional[int]) -> int:
    current = _current_epoch(conn, scope_id)
    if epoch is not None and int(epoch) != current:
        raise VerbatimError(
            ErrorCode.STALE_EPOCH,
            "pinned epoch superseded — rebind authorization",
            retryable=True,
        )
    return current


def _record_propagation(
    conn: sqlite3.Connection,
    *,
    entry: dict[str, Any],
    recipient_id: str,
    purpose: str,
    epoch: int,
    now: int,
) -> str:
    pid = new_id()
    repos_v3.insert(
        conn,
        "propagations",
        {
            "propagation_id": pid,
            "scope_id": entry["scope_id"],
            "object_kind": "vault_entry",
            "object_id": entry["entry_id"],
            "revision": int(entry["revision"]),
            "recipient_id": recipient_id,
            "verbs_json": ["hydrate"],
            "purpose": purpose,
            "epoch": int(epoch),
            "created_us": int(now),
            "capsule_id": None,
            "revoked_seq": None,
            "acknowledged": 0,
        },
    )
    return pid


def issue_handle(
    conn: sqlite3.Connection,
    entry: dict[str, Any],
    *,
    recipient_id: str,
    action_digest: bytes,
    consent_id: str,
    ttl_us: int,
    now: Optional[int] = None,
) -> str:
    """Mint an opaque one-use ``value_handles`` row (§35.04).

    The handle binds recipient, the exact approved action digest, the
    consent that authorized it, and expiry — it carries no value bytes and
    resolves exactly once through :func:`resolve_handle`.
    """
    require_id(recipient_id, "recipient_id")
    if not isinstance(action_digest, (bytes, bytearray)) or not action_digest:
        raise VerbatimError(
            ErrorCode.VALIDATION, "action_digest must be non-empty bytes"
        )
    ts = now if now is not None else now_us()
    handle_id = new_id()
    repos_v3.insert(
        conn,
        "value_handles",
        {
            "handle_id": handle_id,
            "vault_entry_id": entry["entry_id"],
            "scope_id": entry["scope_id"],
            "recipient_id": recipient_id,
            "action_digest": bytes(action_digest),
            "consent_id": consent_id,
            "expires_us": int(ts) + int(ttl_us),
            "consumed_us": None,
        },
    )
    return handle_id


def hydrate(
    conn: sqlite3.Connection,
    caller_principal: str,
    entry_id: str,
    *,
    purpose: str,
    consent_id: str,
    downstream_processor: str,
    epoch: Optional[int],
    cfg: VerbatimConfig,
    key_provider: Optional[Callable[..., Any]] = None,
    action_digest: Optional[bytes] = None,
    now: Optional[int] = None,
) -> HydrationResult:
    """The §35.04 hydration flow — plaintext for local-trusted targets,
    an opaque one-use handle for everything else."""
    require_id(caller_principal, "caller_principal")
    if not purpose:
        raise VerbatimError(ErrorCode.VALIDATION, "purpose is mandatory")
    if not consent_id:
        raise VerbatimError(ErrorCode.CONSENT_REQUIRED, "consent_id required")
    ts = now if now is not None else now_us()

    entry = _vault.get_entry(conn, entry_id)
    if entry is None or entry.get("erased_event") is not None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "vault entry unavailable"
        )
    _purpose_registered(conn, purpose)
    consent = _consent_covers(conn, entry, consent_id, purpose, ts)

    if not downstream_processor:
        # Unknown downstream processing denies hydration (§11.12).
        raise VerbatimError(
            ErrorCode.CONSENT_REQUIRED,
            "downstream processor must be declared for hydration",
        )
    if (
        not cfg.v3.vault.allow_plaintext_hydration
        and downstream_processor != consent["processor"]
    ):
        raise VerbatimError(
            ErrorCode.CONSENT_REQUIRED,
            "downstream processor does not match the consent processor",
        )
    current_epoch = _check_epoch(conn, entry["scope_id"], epoch)

    plaintext_ok = (
        cfg.v3.vault.allow_plaintext_hydration
        or downstream_processor in LOCAL_TRUSTED_PROCESSORS
    )
    if plaintext_ok:
        value = _vault._decrypt_row(entry, cfg, key_provider)
        pid = _record_propagation(
            conn,
            entry=entry,
            recipient_id=caller_principal,
            purpose=purpose,
            epoch=current_epoch,
            now=ts,
        )
        return HydrationResult(
            kind="plaintext",
            entry_id=entry_id,
            propagation_id=pid,
            value=value,
        )

    digest = action_digest or hashlib.sha256(
        json_dumps(
            {
                "op": "hydrate",
                "entry_id": entry_id,
                "purpose": purpose,
                "recipient": caller_principal,
                "processor": downstream_processor,
                "consent_id": consent_id,
            }
        ).encode("utf-8")
    ).digest()
    handle_id = issue_handle(
        conn,
        entry,
        recipient_id=caller_principal,
        action_digest=digest,
        consent_id=consent_id,
        ttl_us=int(cfg.v3.vault.handle_ttl_s) * 1_000_000,
        now=ts,
    )
    pid = _record_propagation(
        conn,
        entry=entry,
        recipient_id=caller_principal,
        purpose=purpose,
        epoch=current_epoch,
        now=ts,
    )
    return HydrationResult(
        kind="handle",
        entry_id=entry_id,
        propagation_id=pid,
        handle_id=handle_id,
        expires_us=ts + int(cfg.v3.vault.handle_ttl_s) * 1_000_000,
        action_digest=digest,
    )


def resolve_handle(
    conn: sqlite3.Connection,
    handle_id: str,
    *,
    recipient_id: str,
    action_digest: bytes,
    cfg: VerbatimConfig,
    key_provider: Optional[Callable[..., Any]] = None,
    purpose: str = "",
    now: Optional[int] = None,
) -> bytes:
    """Redeem a one-use handle for the exact value (§35.04).

    The broker resolves only for the bound recipient and the exact approved
    action digest; the handle is consumed atomically in the caller's
    transaction — a replay (or a race) fails ``VALIDATION``. Expired
    handles and erased entries fail closed.
    """
    ts = now if now is not None else now_us()
    row = repos_v3.get(conn, "value_handles", {"handle_id": handle_id})
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "value handle unavailable"
        )
    if row.get("consumed_us") is not None:
        raise VerbatimError(
            ErrorCode.VALIDATION, "value handle already consumed"
        )
    if int(row["expires_us"]) <= ts:
        raise VerbatimError(ErrorCode.VALIDATION, "value handle expired")
    if row["recipient_id"] != recipient_id:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "value handle unavailable"
        )
    if bytes(row["action_digest"]) != bytes(action_digest):
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "value handle unavailable"
        )
    claimed = repos_v3.update(
        conn,
        "value_handles",
        {"consumed_us": int(ts)},
        {"handle_id": handle_id, "consumed_us": None},
    )
    if claimed != 1:
        raise VerbatimError(
            ErrorCode.VALIDATION, "value handle already consumed"
        )
    value = _vault.open(
        conn, row["vault_entry_id"], cfg=cfg, key_provider=key_provider
    )
    entry = _vault.get_entry(conn, row["vault_entry_id"])
    if entry is not None:
        _record_propagation(
            conn,
            entry=entry,
            recipient_id=recipient_id,
            purpose=purpose or "hydrate",
            epoch=_current_epoch(conn, row["scope_id"]),
            now=ts,
        )
    return value
