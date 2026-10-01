"""Sensitive-value vault: AES-256-GCM seal/open (SPEC_V3 §35.02, §35.05, §37.03).

Construction (reviewed crypto only — ``cryptography.hazmat AESGCM``; no
custom crypto and no plaintext fallback):

- Each entry gets a fresh random 256-bit *data key*; the value is encrypted
  under it as AES-256-GCM with a random 96-bit nonce.
- The data key is wrapped under the scope's externally supplied wrap key by
  a *separate* AESGCM instance with its own random nonce
  (``wrapped_key`` = ``wrap_nonce ‖ wrap_ciphertext``). The wrapped payload
  is ``data_key ‖ u32be(seal_wrap_version)`` — the wrap-key version in
  force at seal time travels inside the authenticated wrap so the value
  AAD can bind it permanently while rotation rewrites only the outer wrap
  (§35.12, §37.03).
- The value AAD binds the authenticated metadata record
  ``(entry_id, scope_id, revision, sensitivity, placeholder, key_version,
  wrap_key_version-at-seal)`` — tampering with any stored metadata column
  fails either the ``aad_digest`` comparison or the GCM tag, and both map
  to ``INTEGRITY``. The wrap AAD binds ``(entry_id, scope_id,
  wrap_key_version-current, key_version)`` so wrapped blobs cannot be
  transplanted across entries or misattributed to another wrap version.
- ``vault_entries`` rows carry algorithm tag ``aes-256-gcm:v1``, key
  versions, nonce, ciphertext, wrapped key, and the AAD digest — never key
  material, plaintext, or an unkeyed digest of the value (§35.02, §35.07).

Rotation (§35.12): re-wrapping rewrites ``wrapped_key`` and
``wrap_key_version`` only — the value ciphertext, nonce, data key, and
``aad_digest`` are untouched because the data key persists. ``open`` after
rotation uses the *new* wrap key; once the provider retires the old
version, opening pre-rotation wrapped blobs is impossible — the engine
reports that state honestly (CAPABILITY_UNAVAILABLE), never silent.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import struct
from typing import Any, Callable, Optional

from ..config import VerbatimConfig
from ..core.time import now_us
from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    new_id,
    require_id,
)
from ..core.types_v3 import DetectionBasis, SensitivityClass
from ..storage import repos_v3
from . import keys as _keys

ALGORITHM = "aes-256-gcm:v1"
_DATA_KEY_BYTES = 32
_NONCE_BYTES = 12          # 96-bit GCM nonce
_WRAP_NONCE_BYTES = 12
_SEAL_VER_STRUCT = struct.Struct(">I")


def _canonical(fields: dict[str, Any]) -> bytes:
    return json_dumps(fields).encode("utf-8")


def _value_aad(
    *,
    entry_id: str,
    scope_id: str,
    revision: int,
    sensitivity: str,
    placeholder: str,
    key_version: int,
    wrap_key_version: int,
) -> bytes:
    """AAD for the value ciphertext — seal-time metadata record.

    ``wrap_key_version`` here is the version that originally wrapped the
    data key (carried inside the wrapped payload), so rotation — which
    changes only the row's ``wrap_key_version`` column — leaves this AAD,
    the ciphertext, and the stored ``aad_digest`` intact.
    """
    return _canonical(
        {
            "alg": ALGORITHM,
            "op": "value",
            "entry_id": entry_id,
            "scope_id": scope_id,
            "revision": int(revision),
            "sensitivity": sensitivity,
            "placeholder": placeholder,
            "key_version": int(key_version),
            "wrap_key_version": int(wrap_key_version),
        }
    )


def _wrap_aad(
    *,
    entry_id: str,
    scope_id: str,
    wrap_key_version: int,
    key_version: int,
) -> bytes:
    """AAD for the wrap operation — binds the wrapped blob to this entry
    and to the wrap-key version actually used."""
    return _canonical(
        {
            "alg": ALGORITHM,
            "op": "wrap",
            "entry_id": entry_id,
            "scope_id": scope_id,
            "wrap_key_version": int(wrap_key_version),
            "key_version": int(key_version),
        }
    )


def _aesgcm() -> Any:
    """The reviewed AEAD primitive — absent provider is a capability failure,
    never a reason to fall back to plaintext or homegrown crypto (§35.02)."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise VerbatimError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            "cryptography provider unavailable; vault cannot operate",
        ) from exc
    return AESGCM


def _invalid_tag(exc: Exception) -> VerbatimError:
    return VerbatimError(
        ErrorCode.INTEGRITY, "vault entry failed authentication"
    )


def _entry_row(conn: sqlite3.Connection, entry_id: str) -> dict[str, Any]:
    row = repos_v3.get(conn, "vault_entries", {"entry_id": entry_id})
    if row is None or row.get("erased_event") is not None:
        # Absent and erased are indistinguishable (§09.09, §36.03).
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "vault entry unavailable"
        )
    return row


def _wrap_data_key(
    wrap_key: bytes,
    *,
    data_key: bytes,
    seal_version: int,
    entry_id: str,
    scope_id: str,
    wrap_key_version: int,
    key_version: int,
) -> bytes:
    AESGCM = _aesgcm()
    nonce = os.urandom(_WRAP_NONCE_BYTES)
    aad = _wrap_aad(
        entry_id=entry_id,
        scope_id=scope_id,
        wrap_key_version=wrap_key_version,
        key_version=key_version,
    )
    payload = data_key + _SEAL_VER_STRUCT.pack(seal_version)
    ct = AESGCM(wrap_key).encrypt(nonce, payload, aad)
    return nonce + ct


def _unwrap_data_key(
    wrap_key: bytes,
    blob: bytes,
    *,
    entry_id: str,
    scope_id: str,
    wrap_key_version: int,
    key_version: int,
) -> tuple[bytes, int]:
    """Return ``(data_key, seal_wrap_version)`` from a wrapped blob."""
    AESGCM = _aesgcm()
    if len(blob) <= _WRAP_NONCE_BYTES:
        raise VerbatimError(
            ErrorCode.INTEGRITY, "vault wrapped_key blob is truncated"
        )
    nonce, ct = blob[:_WRAP_NONCE_BYTES], blob[_WRAP_NONCE_BYTES:]
    aad = _wrap_aad(
        entry_id=entry_id,
        scope_id=scope_id,
        wrap_key_version=wrap_key_version,
        key_version=key_version,
    )
    try:
        payload = AESGCM(wrap_key).decrypt(nonce, ct, aad)
    except Exception as exc:  # AESGCM.InvalidTag and friends
        raise _invalid_tag(exc) from exc
    expect = _DATA_KEY_BYTES + _SEAL_VER_STRUCT.size
    if len(payload) != expect:
        raise VerbatimError(
            ErrorCode.INTEGRITY, "vault wrapped payload is malformed"
        )
    data_key = payload[:_DATA_KEY_BYTES]
    (seal_version,) = _SEAL_VER_STRUCT.unpack(payload[_DATA_KEY_BYTES:])
    return data_key, seal_version


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def seal(
    conn: sqlite3.Connection,
    scope_id: str,
    value: bytes,
    *,
    sensitivity: str,
    placeholder: str,
    cfg: VerbatimConfig,
    revision: int = 1,
    detection: str = DetectionBasis.DETECTED.value,
    key_provider: Optional[Callable[..., Any]] = None,
    entry_id: Optional[str] = None,
    created_event: int = 0,
) -> str:
    """Encrypt ``value`` into a new ``vault_entries`` row; returns entry_id.

    Fresh random data key per call — two seals of identical plaintext yield
    different nonces, ciphertexts, and wrapped keys (§35.02). The scope
    wrap key is resolved from the configured provider at its *current*
    version and is never persisted.
    """
    if not cfg.v3.vault.enabled:
        raise VerbatimError(
            ErrorCode.CAPABILITY_UNAVAILABLE, "vault is disabled"
        )
    require_id(scope_id, "scope_id")
    if not isinstance(value, (bytes, bytearray)) or len(value) == 0:
        raise VerbatimError(ErrorCode.VALIDATION, "vault value must be bytes")
    sens = SensitivityClass(sensitivity).value
    det = DetectionBasis(detection).value
    if not isinstance(placeholder, str) or not placeholder:
        raise VerbatimError(ErrorCode.VALIDATION, "placeholder required")
    eid = entry_id or new_id()
    require_id(eid, "entry_id")
    key_version = 1  # per-entry data key, first generation (§37.03)

    wrap_key, wrap_version = _keys.load_wrap_key(
        scope_id, cfg, provider=key_provider
    )
    AESGCM = _aesgcm()
    data_key = AESGCM.generate_key(bit_length=_DATA_KEY_BYTES * 8)
    aad = _value_aad(
        entry_id=eid,
        scope_id=scope_id,
        revision=revision,
        sensitivity=sens,
        placeholder=placeholder,
        key_version=key_version,
        wrap_key_version=wrap_version,
    )
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext = AESGCM(data_key).encrypt(nonce, bytes(value), aad)
    wrapped = _wrap_data_key(
        wrap_key,
        data_key=data_key,
        seal_version=wrap_version,
        entry_id=eid,
        scope_id=scope_id,
        wrap_key_version=wrap_version,
        key_version=key_version,
    )
    repos_v3.insert(
        conn,
        "vault_entries",
        {
            "entry_id": eid,
            "scope_id": scope_id,
            "revision": int(revision),
            "sensitivity": sens,
            "placeholder": placeholder,
            "algorithm": ALGORITHM,
            "key_version": key_version,
            "wrap_key_version": wrap_version,
            "nonce": nonce,
            "ciphertext": ciphertext,
            "wrapped_key": wrapped,
            "aad_digest": hashlib.sha256(aad).digest(),
            "detection": det,
            "created_event": int(created_event),
            "erased_event": None,
        },
    )
    return eid


def _decrypt_row(row: dict[str, Any], cfg: VerbatimConfig,
                 key_provider: Optional[Callable[..., Any]]) -> bytes:
    if row.get("algorithm") != ALGORITHM:
        raise VerbatimError(
            ErrorCode.INTEGRITY,
            f"vault entry uses unsupported algorithm {row.get('algorithm')!r}",
        )
    wrap_version = int(row["wrap_key_version"])
    wrap_key, _ = _keys.load_wrap_key(
        row["scope_id"], cfg, version=wrap_version, provider=key_provider
    )
    data_key, seal_version = _unwrap_data_key(
        wrap_key,
        bytes(row["wrapped_key"]),
        entry_id=row["entry_id"],
        scope_id=row["scope_id"],
        wrap_key_version=wrap_version,
        key_version=int(row["key_version"]),
    )
    aad = _value_aad(
        entry_id=row["entry_id"],
        scope_id=row["scope_id"],
        revision=int(row["revision"]),
        sensitivity=row["sensitivity"],
        placeholder=row["placeholder"],
        key_version=int(row["key_version"]),
        wrap_key_version=seal_version,
    )
    if hashlib.sha256(aad).digest() != bytes(row["aad_digest"]):
        raise VerbatimError(
            ErrorCode.INTEGRITY, "vault entry metadata failed integrity"
        )
    AESGCM = _aesgcm()
    try:
        return AESGCM(data_key).decrypt(
            bytes(row["nonce"]), bytes(row["ciphertext"]), aad
        )
    except Exception as exc:
        raise _invalid_tag(exc) from exc


def open(
    conn: sqlite3.Connection,
    entry_id: str,
    *,
    cfg: VerbatimConfig,
    key_provider: Optional[Callable[..., Any]] = None,
) -> bytes:
    """Unwrap the data key and decrypt one live entry → plaintext bytes.

    Absent or erased entries raise NOT_FOUND_OR_UNAUTHORIZED; tampered
    ciphertext/metadata raise INTEGRITY (§35.02, §46).
    """
    if not cfg.v3.vault.enabled:
        raise VerbatimError(
            ErrorCode.CAPABILITY_UNAVAILABLE, "vault is disabled"
        )
    row = _entry_row(conn, entry_id)
    return _decrypt_row(row, cfg, key_provider)


def get_entry(conn: sqlite3.Connection, entry_id: str) -> Optional[dict[str, Any]]:
    """Non-secret metadata snapshot; ``None`` when absent (not erased-aware)."""
    return repos_v3.get(conn, "vault_entries", {"entry_id": entry_id})


def rewrap(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    *,
    cfg: VerbatimConfig,
    key_provider: Optional[Callable[..., Any]] = None,
) -> bool:
    """Re-wrap one entry's data key under the current scope wrap version.

    Returns False when the entry is already at the current version. The
    value ciphertext, nonce, data key, ``key_version``, and ``aad_digest``
    are untouched — the seal-time version travels inside the wrapped
    payload (§35.12, §37.03).
    """
    wrap_version = int(row["wrap_key_version"])
    old_key, _ = _keys.load_wrap_key(
        row["scope_id"], cfg, version=wrap_version, provider=key_provider
    )
    new_key, new_version = _keys.load_wrap_key(
        row["scope_id"], cfg, provider=key_provider
    )
    if new_version == wrap_version:
        return False
    data_key, seal_version = _unwrap_data_key(
        old_key,
        bytes(row["wrapped_key"]),
        entry_id=row["entry_id"],
        scope_id=row["scope_id"],
        wrap_key_version=wrap_version,
        key_version=int(row["key_version"]),
    )
    wrapped = _wrap_data_key(
        new_key,
        data_key=data_key,
        seal_version=seal_version,
        entry_id=row["entry_id"],
        scope_id=row["scope_id"],
        wrap_key_version=new_version,
        key_version=int(row["key_version"]),
    )
    repos_v3.update(
        conn,
        "vault_entries",
        {"wrapped_key": wrapped, "wrap_key_version": new_version},
        {"entry_id": row["entry_id"]},
    )
    return True


class Vault:
    """Store-bound facade over the vault primitives.

    Owns transactions so callers stay at intent level; ``key_provider`` is
    the ``external`` source hook (§35.11) and ``clock`` keeps tests
    deterministic.
    """

    def __init__(
        self,
        store: Any,
        cfg: VerbatimConfig,
        *,
        key_provider: Optional[Callable[..., Any]] = None,
        clock: Optional[Callable[[], int]] = None,
    ) -> None:
        self._store = store
        self._cfg = cfg
        self._provider = key_provider
        self._now = clock or now_us

    @property
    def cfg(self) -> VerbatimConfig:
        return self._cfg

    def seal(
        self,
        scope_id: str,
        value: bytes,
        *,
        sensitivity: str,
        placeholder: str,
        revision: int = 1,
        detection: str = DetectionBasis.DETECTED.value,
        created_event: int = 0,
    ) -> str:
        with self._store.tx() as conn:
            return seal(
                conn,
                scope_id,
                value,
                sensitivity=sensitivity,
                placeholder=placeholder,
                cfg=self._cfg,
                revision=revision,
                detection=detection,
                key_provider=self._provider,
                created_event=created_event,
            )

    def open(self, entry_id: str) -> bytes:
        with self._store.read() as conn:
            return open(conn, entry_id, cfg=self._cfg,
                        key_provider=self._provider)

    def entry(self, entry_id: str) -> Optional[dict[str, Any]]:
        with self._store.read() as conn:
            return get_entry(conn, entry_id)

    def rotate_scope(self, scope_id: str) -> dict[str, Any]:
        """Re-wrap every live entry in the scope to the current wrap version.

        The rotation receipt lists the wrap versions retired by this pass —
        backups written before rotation may still depend on them (§35.12),
        which the receipt states plainly rather than claiming destruction.
        """
        with self._store.tx() as conn:
            rows = repos_v3.query(
                conn, "vault_entries",
                {"scope_id": scope_id, "erased_event": None},
            )
            retired: set[int] = set()
            rewapped = 0
            for row in rows:
                old_version = int(row["wrap_key_version"])
                if rewrap(conn, row, cfg=self._cfg,
                          key_provider=self._provider):
                    retired.add(old_version)
                    rewapped += 1
            return {
                "scope_id": scope_id,
                "rewrapped": rewapped,
                "entries": len(rows),
                "retired_wrap_versions": sorted(retired),
                "backup_dependency": (
                    "backups taken before this rotation may still require "
                    "the retired wrap key versions to decrypt"
                    if retired
                    else "no wrap version retired"
                ),
            }

    # ------------------------------------------------------------------
    # hydration / handles / tickets / redaction / erasure — thin
    # transaction-owning wrappers over the sibling modules (§35.04, §09.07)
    # ------------------------------------------------------------------

    def hydrate(self, caller_principal: str, entry_id: str, *,
                purpose: str, consent_id: str,
                downstream_processor: str,
                epoch: Optional[int] = None,
                action_digest: Optional[bytes] = None) -> Any:
        from .hydration import hydrate as _hydrate

        with self._store.tx() as conn:
            return _hydrate(
                conn, caller_principal, entry_id,
                purpose=purpose, consent_id=consent_id,
                downstream_processor=downstream_processor, epoch=epoch,
                cfg=self._cfg, key_provider=self._provider,
                action_digest=action_digest, now=self._now(),
            )

    def resolve_handle(self, handle_id: str, *, recipient_id: str,
                       action_digest: bytes) -> bytes:
        from .hydration import resolve_handle as _resolve

        with self._store.tx() as conn:
            return _resolve(
                conn, handle_id, recipient_id=recipient_id,
                action_digest=action_digest, cfg=self._cfg,
                key_provider=self._provider, now=self._now(),
            )

    def issue_ticket(self, scope_id: str, recipient_id: str,
                     action_digest: bytes, purpose: str,
                     epoch: Optional[int],
                     objects: Any, ttl_us: Optional[int],
                     gateway_id: str) -> str:
        from .tickets import issue as _issue

        with self._store.tx() as conn:
            return _issue(
                conn, scope_id, recipient_id, action_digest, purpose,
                epoch, objects, ttl_us, gateway_id, now=self._now(),
            )

    def verify_ticket(self, ticket_id: str, gateway_id: str,
                      action_digest: bytes) -> dict[str, Any]:
        from .tickets import verify as _verify

        with self._store.tx() as conn:
            return _verify(conn, ticket_id, gateway_id, action_digest,
                           now=self._now())

    def redact_view(self, scope_id: str, view_id: str, content: bytes, *,
                    detector: Any = None,
                    declared: Optional[list] = None,
                    revision: int = 1,
                    created_event: int = 0) -> Any:
        from .redaction import redact_view as _redact

        with self._store.tx() as conn:
            return _redact(
                conn, scope_id, view_id, content, cfg=self._cfg,
                detector=detector, declared=declared,
                key_provider=self._provider, revision=revision,
                created_event=created_event,
            )

    def erase(self, entry_id: str, *, erased_event: int = 0) -> bool:
        from .erasure import erase_entry

        with self._store.tx() as conn:
            return erase_entry(
                conn, entry_id, erased_event=erased_event,
                hmac_fn=self._store.hmac,
            )

    def report(self, entry_id: str, *,
               verified_key_destruction: bool = False) -> dict[str, Any]:
        from .erasure import report as _report

        with self._store.read() as conn:
            return _report(
                conn, entry_id,
                verified_key_destruction=verified_key_destruction,
            )
