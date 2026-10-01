"""Bearer-token authentication for the HTTP surfaces (SPEC_V4 §45, §49.09).

The token is an *authentication* credential only — it proves which
principal is calling. It carries no authority of its own: the verbs a
credential names are the ceiling the operator provisioned, and every
request is still evaluated against persisted ``grants_v3`` rows through
``governance.authorize`` (single-authority rule — this module owns
authentication, never authorization decisions).

Provisioning is explicit (V4-45.03): tokens come from an operator-written
JSON file or an operator-set environment variable; nothing here generates,
prints, or guesses a token, and a request without a matching token gets
the same 401 shape whether the token is missing or wrong.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ..core.types import ErrorCode, VerbatimError, require_id
from ..core.types_v3 import Verb

#: Environment variable carrying the token file path (operator provisioned).
TOKEN_FILE_ENV = "VERBATIM_SERVICE_TOKEN_FILE"
#: Environment variable carrying an inline JSON token list — same schema as
#: the file. Intended for small deployments/tests; the file form is preferred
#: because environment variables leak through process listings.
TOKENS_ENV = "VERBATIM_SERVICE_TOKENS"

_MAX_TOKEN_BYTES = 4096
_MAX_TOKENS = 1024


def _verb_set(values: Iterable[str]) -> frozenset:
    out: set[str] = set()
    for v in values:
        if not isinstance(v, str):
            raise VerbatimError(ErrorCode.VALIDATION, "verbs must be strings")
        try:
            out.add(Verb(v).value)
        except ValueError:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"unknown grant verb {v!r}"
            ) from None
    if not out:
        raise VerbatimError(
            ErrorCode.VALIDATION, "credential requires at least one verb"
        )
    return frozenset(out)


@dataclass(frozen=True)
class TokenCredential:
    """One provisioned bearer token.

    ``token_digest`` is SHA-256 of the presented secret — the raw token is
    never retained past construction. ``verbs`` is the maximum grant class
    this credential may exercise; effective authority is whatever
    ``grants_v3`` currently permits within that ceiling (revocation narrows,
    the token never widens).
    """

    token_digest: bytes
    principal_id: str
    verbs: frozenset
    label: str = ""
    # Optional scope narrowing the operator pinned at provisioning time;
    # defaults to the served engine's home scope.
    scope: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def mint(cls, token: str, principal_id: str, verbs, **kw) -> "TokenCredential":
        """Build a credential from the raw secret; the secret is digested
        immediately and not stored on the credential."""
        if not isinstance(token, str) or not token.strip():
            raise VerbatimError(ErrorCode.VALIDATION, "token must be a non-empty string")
        if len(token.encode("utf-8")) > _MAX_TOKEN_BYTES:
            raise VerbatimError(ErrorCode.VALIDATION, "token exceeds size bound")
        require_id(principal_id, "principal_id")
        return cls(
            token_digest=hashlib.sha256(token.encode("utf-8")).digest(),
            principal_id=principal_id,
            verbs=_verb_set(verbs),
            **kw,
        )


def _credential_from(raw: Any) -> TokenCredential:
    if not isinstance(raw, dict):
        raise VerbatimError(ErrorCode.VALIDATION, "token entry must be an object")
    allowed = {"token", "principal_id", "verbs", "label", "scope"}
    unknown = set(raw) - allowed
    if unknown:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown token fields {sorted(unknown)}"
        )
    scope = raw.get("scope") or {}
    if not isinstance(scope, dict):
        raise VerbatimError(ErrorCode.VALIDATION, "token scope must be an object")
    unknown_scope = set(scope) - {"workspace_id", "conversation_id", "visibility"}
    if unknown_scope:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"unknown token scope fields {sorted(unknown_scope)}",
        )
    return TokenCredential.mint(
        raw.get("token"),
        raw.get("principal_id"),
        raw.get("verbs") or (),
        label=str(raw.get("label") or "")[:64],
        scope={
            k: str(v)
            for k, v in scope.items()
            if isinstance(v, str) and v
        },
    )


def parse_credentials(doc: Any) -> tuple[TokenCredential, ...]:
    """Validate a token document: ``{"tokens": [...]}`` or a bare list."""
    entries = doc.get("tokens") if isinstance(doc, dict) else doc
    if not isinstance(entries, list):
        raise VerbatimError(
            ErrorCode.VALIDATION, "token document must be a list or {'tokens': [...]}"
        )
    if not entries:
        raise VerbatimError(ErrorCode.VALIDATION, "no tokens provisioned")
    if len(entries) > _MAX_TOKENS:
        raise VerbatimError(ErrorCode.VALIDATION, "token list exceeds bound")
    creds = tuple(_credential_from(e) for e in entries)
    # One token secret must name one principal — a duplicate secret with a
    # different principal would make authentication ambiguous.
    seen: dict[bytes, str] = {}
    for c in creds:
        prior = seen.setdefault(c.token_digest, c.principal_id)
        if prior != c.principal_id:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "the same token secret cannot name two principals",
            )
    return creds


def load_token_file(path: str) -> tuple[TokenCredential, ...]:
    """Read the operator-provisioned token file (JSON)."""
    require_id(path, "path")
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise VerbatimError(
            ErrorCode.CONFIG_INVALID, f"cannot read token file {path!r}: {exc}"
        ) from exc
    if len(raw) > (1 << 20):
        raise VerbatimError(ErrorCode.VALIDATION, "token file exceeds 1 MiB")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise VerbatimError(
            ErrorCode.CONFIG_INVALID, f"token file {path!r} is not valid JSON"
        ) from exc
    return parse_credentials(doc)


def tokens_from_env(
    env: Optional[dict[str, str]] = None,
) -> tuple[TokenCredential, ...]:
    """Provision from ``VERBATIM_SERVICE_TOKEN_FILE`` / ``VERBATIM_SERVICE_TOKENS``.

    Returns an empty tuple when neither is set — callers decide whether
    zero credentials is a refusal (it is, for a network surface).
    """
    environ = os.environ if env is None else env
    file_path = environ.get(TOKEN_FILE_ENV)
    if file_path:
        return load_token_file(file_path)
    raw = environ.get(TOKENS_ENV)
    if raw:
        try:
            doc = json.loads(raw)
        except ValueError as exc:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID, f"{TOKENS_ENV} is not valid JSON"
            ) from exc
        return parse_credentials(doc)
    return ()


def scope_mismatch(cred: TokenCredential, bound_principal: str) -> Optional[str]:
    """Why a credential fails to bind to a pinned app identity, or None.

    Single-partition surfaces (the /v2/memory app) serve exactly one
    bound identity: a credential naming another principal is "scoped
    elsewhere", and a credential carrying an operator scope pin
    (workspace/conversation/visibility — the /v1 narrowing model) can
    never be honored there either. Returns a short reason token for
    logging; callers choose their own refusal shape (403 at request
    time, ``CONFIG_INVALID`` at bind time).
    """
    if cred.principal_id != bound_principal:
        return "principal_id"
    if cred.scope:
        return "scope"
    return None


class TokenAuthenticator:
    """Constant-time bearer-token check over provisioned credentials."""

    def __init__(self, credentials: Iterable[TokenCredential]) -> None:
        self._credentials = tuple(credentials)
        if not self._credentials:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "HTTP transport requires at least one provisioned token",
            )

    def authenticate(self, authorization_header: Optional[str]) -> Optional[TokenCredential]:
        """Resolve the bearer secret to its credential, or ``None``.

        The presented secret is hashed once and compared with
        ``hmac.compare_digest`` against every provisioned digest —
        fixed-length comparisons only, and the loop does not exit early,
        so neither token length nor prefix leaks through timing. A bad
        secret and a missing header are indistinguishable to the caller
        (both refuse); the response code is the transport's business.
        """
        if not authorization_header or not isinstance(authorization_header, str):
            return None
        parts = authorization_header.split(None, 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        secret = parts[1].strip()
        if not secret or len(secret.encode("utf-8", "replace")) > _MAX_TOKEN_BYTES:
            return None
        digest = hashlib.sha256(secret.encode("utf-8")).digest()
        match: Optional[TokenCredential] = None
        for cred in self._credentials:
            if hmac.compare_digest(digest, cred.token_digest):
                match = cred
        return match


__all__ = [
    "TOKEN_FILE_ENV",
    "TOKENS_ENV",
    "TokenAuthenticator",
    "TokenCredential",
    "load_token_file",
    "parse_credentials",
    "scope_mismatch",
    "tokens_from_env",
]
