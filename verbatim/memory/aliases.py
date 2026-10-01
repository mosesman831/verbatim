"""Namespace-alias registry for the V5 consumer facade (SPEC_V5 §05.12–§05.13).

A *namespace* is a governed scope row (``scope_id`` — ``ns_…``).  An
*alias* is a durable label→namespace binding keyed by the trusted owner
identity + the resolved storage profile + alias kind + label:

    alias identity = HMAC(store_key, owner | profile | kind | label)

HMAC-keying means the alias keys stored in ``meta`` reveal neither the
labels nor which labels exist (indistinguishability, §10.05), and a
label can never collide with another owner's partition.  An alias label
is **never** an identity: it does not authenticate, does not select a
store path, and grants nothing by itself — authority flows only through
the bound caller's grants on the resolved namespace.

Storage choice: the existing ``meta`` store key/value table
(``Store._meta_get`` / ``Store._meta_set``).  No dedicated alias schema
exists in this checkout (schema files are frozen to other workers), and
``meta`` is the documented in-schema KV facility — records are
``meta`` keys ``memory.alias.<hmac-hex>`` holding a JSON record.  All
mutations happen inside the caller's ``store.tx()`` so an alias record
and the namespace scope/grant it points at commit atomically.

Kinds (V5-05.12):
  * ``user``  — durable personal memory.
  * ``agent`` — durable agent-private notes; the *record* preserves the
    agent label, but caller authority still comes from the principal's
    grants — cross-agent access needs an explicit grant, never the alias.
  * ``run``   — working memory with a declared TTL; records carry
    ``expires_us`` and are excluded from durable search by default.
    Promotion out of a run namespace is a separate consented path (X7).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .. import governance
from ..core.types import ErrorCode, VerbatimError, new_id
from ..core.time import now_us
from ..evidence.envelopes import ensure_scope_row

_KIND_USER = "user"
_KIND_AGENT = "agent"
_KIND_RUN = "run"
KINDS = (_KIND_USER, _KIND_AGENT, _KIND_RUN)

#: ``meta`` key prefix for alias records.
_KEY_PREFIX = "memory.alias."

#: Default TTL for run aliases when the caller does not declare one
#: (24h working memory — always declared, never silently durable).
RUN_DEFAULT_TTL_S = 24 * 3600

#: Labels are opaque strings — never ids, never paths.  Bounded so a
#: hostile label cannot blow up the meta table; NUL/control chars and
#: leading/trailing whitespace rejected to keep records printable.
_MAX_LABEL_BYTES = 256
_LABEL_RE = re.compile(r"^[^\x00-\x1f\x7f]+$")


def validate_label(label: Any, *, kind: str) -> str:
    """Validate an alias label: string, bounded, control-char free."""
    if not isinstance(label, str):
        raise VerbatimError(ErrorCode.VALIDATION, f"{kind} label must be a string")
    text = label.strip()
    if not text:
        raise VerbatimError(ErrorCode.VALIDATION, f"{kind} label must not be empty")
    if len(text.encode("utf-8", "strict")) > _MAX_LABEL_BYTES:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"{kind} label exceeds {_MAX_LABEL_BYTES} bytes",
        )
    if _LABEL_RE.match(text) is None:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"{kind} label contains control characters"
        )
    if kind not in KINDS:
        raise VerbatimError(ErrorCode.VALIDATION, f"unknown alias kind {kind!r}")
    return text


def validate_kind(kind: Any) -> str:
    if not isinstance(kind, str) or kind not in KINDS:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"alias kind must be one of {KINDS}",
        )
    return kind


def _record_key(store: Any, owner: str, profile_id: str, kind: str, label: str) -> str:
    """Opaque meta key — the label never appears on disk."""
    digest = store.hmac(
        f"memory-alias|{owner}|{profile_id}|{kind}|{label}".encode("utf-8")
    ).hex()
    return f"{_KEY_PREFIX}{digest}"


def _new_namespace_id() -> str:
    """Mint an opaque namespace id — the label is never encoded."""
    return f"ns_{new_id()}"


def resolve(
    conn,
    store: Any,
    *,
    owner: str,
    profile_id: str,
    kind: str,
    label: str,
    now: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """Look up an alias record; returns ``None`` when absent or expired.

    An expired ``run`` alias resolves to ``None`` — callers treat that as
    "no such alias" (opaque), while the record itself stays on disk as an
    audit artifact.
    """
    label = validate_label(label, kind=kind)
    rec = store._meta_get(conn, _record_key(store, owner, profile_id, kind, label))
    if not isinstance(rec, dict):
        return None
    if rec.get("kind") != kind or rec.get("owner") != owner:
        return None
    expires = rec.get("expires_us")
    if expires is not None and (now if now is not None else now_us()) >= int(expires):
        return None
    return rec


def provision(
    conn,
    store: Any,
    *,
    owner: str,
    profile_id: str,
    kind: str,
    label: str,
    verbs: tuple[str, ...],
    purposes: Optional[tuple[str, ...]] = None,
    ttl_s: Optional[int] = None,
    now: Optional[int] = None,
) -> dict[str, Any]:
    """Authorized namespace provisioning (V5-05.13).

    Creates the namespace scope row, the owner's grant over it, and the
    durable alias record — all inside the caller's transaction so a
    namespace can never exist without its authority binding.  This is an
    *explicit operator act*: the caller (bootstrap for the first alias;
    the trusted-owner binding for later ones) decides that a new
    partition may exist — mere construction of a facade object never
    reaches here for a foreign owner.  ``verbs``/``purposes`` are the
    grant surface issued once at provisioning; attenuation afterwards is
    governance's job, never re-minted here.
    """
    kind = validate_kind(kind)
    label = validate_label(label, kind=kind)
    ts = now if now is not None else now_us()
    if kind == _KIND_RUN:
        if ttl_s is not None and (not isinstance(ttl_s, (int, float)) or ttl_s <= 0):
            raise VerbatimError(
                ErrorCode.VALIDATION, "run alias ttl_s must be positive seconds"
            )
        expires_us = ts + int((ttl_s or RUN_DEFAULT_TTL_S) * 1_000_000)
    else:
        if ttl_s is not None:
            raise VerbatimError(
                ErrorCode.VALIDATION, "ttl_s applies only to run aliases"
            )
        expires_us = None

    namespace = _new_namespace_id()
    # v3 scopes are opaque partition tokens; the row exists for FK +
    # purge-closure partitioning.  ``owner_principal_id`` binds the
    # namespace to its trusted owner (the ``… IS NULL`` guard makes an
    # established owner unrewritable — V5-05.04).
    ensure_scope_row(
        conn, namespace, principal_id=owner, profile_id=profile_id
    )
    conn.execute(
        "UPDATE scopes SET owner_principal_id = ?"
        " WHERE scope_id = ? AND owner_principal_id IS NULL",
        (owner, namespace),
    )
    # Authority for the new partition is issued exactly once, here, with
    # the record — a namespace without a live grant is a revoked one and
    # is never silently re-minted on reopen (V5-05.09).
    governance.create_grant(
        conn,
        scope_id=namespace,
        principal_id=owner,
        verbs=sorted(verbs),
        issuer_id=owner,
        purposes=purposes,
        delegation_depth=0,
    )
    rec: dict[str, Any] = {
        "v": 1,
        "kind": kind,
        "owner": owner,
        "profile_id": profile_id,
        "label_hint": label[:32],  # bounded debugging hint, not the key
        "namespace": namespace,
        "created_us": ts,
        "expires_us": expires_us,
        "ephemeral": kind == _KIND_RUN,
        # Run namespaces are working memory: excluded from durable search
        # unless a caller explicitly opts in and the query path honors it.
        "durable_search": kind != _KIND_RUN,
    }
    store._meta_set(conn, _record_key(store, owner, profile_id, kind, label), rec)
    return rec


def list_for_owner(conn, store: Any, owner: str) -> list[dict[str, Any]]:
    """All alias records owned by ``owner`` (operator/status surface)."""
    out: list[dict[str, Any]] = []
    for (key,) in conn.execute(
        "SELECT key FROM meta WHERE key LIKE ?", (_KEY_PREFIX + "%",)
    ):
        rec = store._meta_get(conn, key)
        if isinstance(rec, dict) and rec.get("owner") == owner:
            out.append(rec)
    return out
