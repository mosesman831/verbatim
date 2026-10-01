"""Vault scope wrapping-key provider (SPEC_V3 §35.02, §35.11, §37.02).

Key material is *never* stored in the database, never written to logs, and
never appears in error messages — failure messages name the scope and the
source only. Three sources are supported (``v3.vault.key_source``):

- ``env``: ``VERBATIM_VAULT_KEY_<SCOPE_SLUG>`` holds the *current* key as
  base64 or hex (optionally prefixed ``vN:`` to declare its version).
  ``VERBATIM_VAULT_KEY_<SCOPE_SLUG>_V<version>`` holds a specific past
  version so entries wrapped before a rotation remain openable while the
  old version is still provisioned (§35.12).
- ``file``: a JSON object mapping ``scope_id`` → key. Values may be a bare
  base64 string (version 1), ``{"key": "<b64>", "version": n}``, or
  ``{"versions": {"1": "<b64>", "2": "<b64>"}}``. The file must be
  owner-only (``mode & 0o077 == 0``) and must not be a symlink.
- ``external``: a caller-supplied provider callable
  ``f(scope_id, version) -> bytes | (bytes, int) | dict`` — the engine
  never generates or persists keys as an import side effect (§35.11).

Every failure to obtain a key raises :class:`KeyNotProvisioned`, which
carries ``CAPABILITY_UNAVAILABLE`` — vault retention/hydration fails
closed (§35.11) and missing keys are a provisioning state, not corruption.
"""

from __future__ import annotations

import base64
import binascii
import os
import re
import stat
from typing import Any, Callable, Optional, Union

from ..config import VerbatimConfig
from ..core.types import ErrorCode, VerbatimError, safe_json_loads

KEY_BYTES = 32  # AES-256 wrapping keys are 256-bit

_ENV_PREFIX = "VERBATIM_VAULT_KEY_"
_VERSION_RE = re.compile(r"^v(?P<v>\d+):(?P<k>.+)$", re.DOTALL)


class KeyNotProvisioned(VerbatimError):
    """The requested scope wrap key is not provisioned (§35.11).

    Always ``CAPABILITY_UNAVAILABLE``: retention/hydration fails closed and
    the error text never carries key bytes, file contents, or env values.
    """

    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.CAPABILITY_UNAVAILABLE, message)


def scope_slug(scope_id: str) -> str:
    """Env-var-safe rendering of a scope id: uppercase, ``[A-Z0-9_]`` only."""
    return re.sub(r"[^A-Z0-9]", "_", scope_id.upper())


def _decode_key_text(text: str, what: str) -> bytes:
    """Decode a base64/hex key string; wrong length is a provisioning bug.

    ``what`` is a safe descriptor (never the value itself) used in errors.
    """
    s = text.strip()
    if s.startswith("0x") or s.startswith("0X"):
        s = s[2:]
    raw: Optional[bytes] = None
    if re.fullmatch(r"[0-9a-fA-F]{64}", s):
        try:
            raw = bytes.fromhex(s)
        except ValueError:
            raw = None
    if raw is None:
        try:
            raw = base64.b64decode(s, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise KeyNotProvisioned(f"{what}: key material is not decodable") from exc
    if len(raw) != KEY_BYTES:
        raise KeyNotProvisioned(
            f"{what}: key material must be exactly {KEY_BYTES} bytes"
        )
    return raw


def _normalize_versioned(value: Any, what: str) -> dict[int, bytes]:
    """Normalize a per-scope key entry into ``{version: key_bytes}``.

    Accepts: ``"<b64>"`` (v1), ``"vN:<b64>"``, ``{"key": ..., "version": n}``,
    ``{"versions": {"n": "<b64>"}}``, or a nested ``{"1": "<b64>"}`` map.
    """
    out: dict[int, bytes] = {}
    if isinstance(value, str):
        m = _VERSION_RE.match(value.strip())
        if m:
            out[int(m.group("v"))] = _decode_key_text(m.group("k"), what)
        else:
            out[1] = _decode_key_text(value, what)
        return out
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        if len(raw) != KEY_BYTES:
            raise KeyNotProvisioned(
                f"{what}: key material must be exactly {KEY_BYTES} bytes"
            )
        out[1] = raw
        return out
    if isinstance(value, dict):
        if "key" in value:
            version = int(value.get("version", 1))
            out[version] = _decode_key_text(str(value["key"]), what)
            return out
        versions = value.get("versions")
        if isinstance(versions, dict):
            for k, v in versions.items():
                out[int(k)] = _decode_key_text(str(v), what)
            return out
        # Flat nested map {"1": "<b64>", ...}
        if value and all(str(k).isdigit() for k in value):
            for k, v in value.items():
                out[int(k)] = _decode_key_text(str(v), what)
            return out
    raise KeyNotProvisioned(f"{what}: unrecognized key entry shape")


def _select(keys: dict[int, bytes], scope_id: str,
            version: Optional[int]) -> tuple[bytes, int]:
    if not keys:
        raise KeyNotProvisioned(
            f"no vault wrap key provisioned for scope {scope_id!r}"
        )
    if version is None:
        v = max(keys)
        return keys[v], v
    try:
        return keys[int(version)], int(version)
    except KeyError as exc:
        raise KeyNotProvisioned(
            f"vault wrap key version {version} for scope {scope_id!r} is not "
            "provisioned (retired or never issued)"
        ) from exc


def _load_env(scope_id: str, version: Optional[int]) -> tuple[bytes, int]:
    slug = scope_slug(scope_id)
    keys: dict[int, bytes] = {}
    if version is not None:
        specific = os.environ.get(f"{_ENV_PREFIX}{slug}_V{int(version)}")
        if specific is not None:
            keys.update(
                _normalize_versioned({str(int(version)): specific},
                                     "env vault key")
            )
    base = os.environ.get(f"{_ENV_PREFIX}{slug}")
    if base is not None:
        keys.update(_normalize_versioned(base, "env vault key"))
    return _select(keys, scope_id, version)


def _key_file_path(cfg: VerbatimConfig) -> str:
    path = cfg.v3.vault.key_file
    if not path:
        raise KeyNotProvisioned(
            "v3.vault.key_source=file requires v3.vault.key_file"
        )
    abspath = os.path.abspath(path)
    if os.path.islink(abspath):
        raise KeyNotProvisioned("vault key file is a symlink — refused")
    return abspath


def _load_file(scope_id: str, cfg: VerbatimConfig,
               version: Optional[int]) -> tuple[bytes, int]:
    path = _key_file_path(cfg)
    try:
        st = os.stat(path)
    except OSError as exc:
        raise KeyNotProvisioned("vault key file is not readable") from exc
    if not stat.S_ISREG(st.st_mode):
        raise KeyNotProvisioned("vault key file is not a regular file")
    # Owner-only permission check (§37.02): group/other access means the
    # key file was provisioned insecurely — fail closed, never warn-and-use.
    if st.st_mode & 0o077:
        raise KeyNotProvisioned(
            "vault key file permissions are too open (require owner-only)"
        )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw_text = fh.read()
    except OSError as exc:
        raise KeyNotProvisioned("vault key file is not readable") from exc
    try:
        mapping = safe_json_loads(raw_text)
    except VerbatimError as exc:
        raise KeyNotProvisioned("vault key file is not valid JSON") from exc
    if not isinstance(mapping, dict):
        raise KeyNotProvisioned("vault key file must be a JSON object")
    entry = mapping.get(scope_id)
    if entry is None:
        raise KeyNotProvisioned(
            f"no vault wrap key provisioned for scope {scope_id!r}"
        )
    keys = _normalize_versioned(entry, "vault key file entry")
    return _select(keys, scope_id, version)


def _load_external(scope_id: str, provider: Callable[..., Any],
                   version: Optional[int]) -> tuple[bytes, int]:
    try:
        result = provider(scope_id, version)
    except VerbatimError:
        raise
    except Exception as exc:
        # Provider exceptions never propagate raw — they may contain secret
        # context; collapse to a safe provisioning failure (§46.01).
        raise KeyNotProvisioned(
            "external vault key provider failed to supply the scope key"
        ) from exc
    if result is None:
        raise KeyNotProvisioned(
            f"external provider returned no key for scope {scope_id!r}"
        )
    if isinstance(result, (bytes, bytearray)):
        raw = bytes(result)
        if len(raw) != KEY_BYTES:
            raise KeyNotProvisioned(
                f"external provider key must be exactly {KEY_BYTES} bytes"
            )
        # An unversioned external answer is only acceptable when no
        # specific version was requested.
        if version is not None:
            raise KeyNotProvisioned(
                "external provider must return (key, version) for a "
                "versioned request"
            )
        return raw, 1
    if isinstance(result, tuple) and len(result) == 2:
        raw, ver = result
        raw = bytes(raw)
        if len(raw) != KEY_BYTES:
            raise KeyNotProvisioned(
                f"external provider key must be exactly {KEY_BYTES} bytes"
            )
        return raw, int(ver)
    if isinstance(result, dict):
        keys = _normalize_versioned(result, "external provider entry")
        return _select(keys, scope_id, version)
    raise KeyNotProvisioned("external provider returned an unsupported shape")


def load_wrap_key(
    scope_id: str,
    cfg: VerbatimConfig,
    *,
    version: Optional[int] = None,
    provider: Optional[Callable[..., Any]] = None,
) -> tuple[bytes, int]:
    """Resolve the scope wrap key for ``scope_id``.

    ``version=None`` requests the *current* (newest provisioned) version —
    used when sealing/wrapping. A specific version is requested when opening
    an entry whose ``wrap_key_version`` was recorded at seal time, so
    entries survive rotation while the old version remains provisioned
    (§35.12). ``provider`` supplies the ``external`` source; it is ignored
    for ``env``/``file`` so a misconfigured build cannot silently consult a
    caller hook.
    """
    source = cfg.v3.vault.key_source
    if source == "env":
        return _load_env(scope_id, version)
    if source == "file":
        return _load_file(scope_id, cfg, version)
    if source == "external":
        if provider is None:
            raise KeyNotProvisioned(
                "v3.vault.key_source=external requires a key provider callable"
            )
        return _load_external(scope_id, provider, version)
    raise KeyNotProvisioned(f"unsupported vault key_source {source!r}")


WrapKeyProvider = Callable[[str, Optional[int]], Union[bytes, tuple[bytes, int], dict]]
