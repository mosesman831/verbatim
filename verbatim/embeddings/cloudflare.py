"""Cloudflare Workers AI encoder — remote embeddings behind remote_assisted.

``remote_assisted`` mode permits explicitly authorized remote processors
only (SPEC_V2 §4, §36). This module is pure transport + validation:

- The host is fixed to ``api.cloudflare.com`` over TLS — a configured
  account id can never redirect the endpoint, and ``http`` or off-host
  URLs are rejected at construction.
- The API token arrives through the host's scoped ``secret_getter``
  (``CLOUDFLARE_API_TOKEN``) — never through config, env fallback in this
  module, or stored state. A missing token is a degraded capability.
- Authorization for egress is structural (SPEC_V4 §12, F4-04): ``encode``
  refuses to dispatch unless the caller passes a ``DispatchPermit`` minted
  by the TransportBroker for THIS endpoint and THIS exact request body —
  ``_authorize_transport`` consumes the permit (one-use) immediately
  before the HTTP request, inside the broker's own recheck transaction.
  An encoder built without ``broker=`` can never dispatch.
- ``available()`` therefore only reports credential presence, never
  consent — and never performs network I/O.
- Timeouts are explicit; response bodies are bounded and untrusted —
  every vector is codec-validated before it can persist (SPEC §28).

Wire shape (Workers AI REST):
``POST /client/v4/accounts/{account}/ai/run`` with
``{"model": "@cf/baai/bge-m3", "input": {"text": [...]}}`` returning
``{"result": {"data": [[f32, ...], ...]}, "success": true}``.
"""

from __future__ import annotations

import json
import socket
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from ..config import EmbeddingConfig
from ..core.types import ErrorCode, VerbatimError, require_id
from .codec import Float32Codec
from .encoder import RemoteEncoder, encoder_identity

_HOST = "api.cloudflare.com"
_PATH = "/client/v4/accounts/{account}/ai/run"
_CONNECT_TIMEOUT_S = 5.0
_TOTAL_TIMEOUT_S = 30.0
_MAX_BODY_BYTES = 32 << 20
_MAX_TEXTS_PER_CALL = 256
SECRET_NAME = "CLOUDFLARE_API_TOKEN"

# Engine-side request contract revision; bump when the payload shape or
# normalization changes so stored vectors identify their generation.
PREPROCESSING_VERSION = "verbatim-cloudflare-embed-v1"


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: bytes


class HttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        body: Optional[bytes],
        headers: dict[str, str],
        timeout_s: float,
    ) -> HttpResult: ...


class _UrllibTransport:
    """Real transport: TLS-verified, no redirect following.

    ``HTTPSHandler`` with the default verified context; redirects are not
    followed automatically because the transport issues ``Request`` objects
    through an ``OpenerDirector`` containing only ``HTTPSHandler`` — a 30x
    surfaces as an HTTPError status instead of silently re-posting
    credentials to a different origin.
    """

    def __init__(self) -> None:
        self._opener = urllib.request.OpenerDirector()
        self._opener.add_handler(
            urllib.request.HTTPSHandler(context=ssl.create_default_context())
        )

    def request(
        self,
        method: str,
        url: str,
        *,
        body: Optional[bytes],
        headers: dict[str, str],
        timeout_s: float,
    ) -> HttpResult:
        deadline = time.monotonic() + min(timeout_s, _TOTAL_TIMEOUT_S)
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with self._opener.open(req, timeout=_CONNECT_TIMEOUT_S) as resp:
                status = resp.status
                chunks: list[bytes] = []
                total = 0
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise VerbatimError(
                            ErrorCode.ENCODER_UNAVAILABLE,
                            "cloudflare encoder response exceeded total timeout",
                            retryable=True,
                        )
                    chunk = resp.read(min(1 << 20, _MAX_BODY_BYTES - total + 1))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > _MAX_BODY_BYTES:
                        raise VerbatimError(
                            ErrorCode.VECTOR_INVALID,
                            "cloudflare encoder response exceeds body bound",
                        )
                return HttpResult(status=status, body=b"".join(chunks))
        except urllib.error.HTTPError as exc:
            # Definitive non-2xx: surface status so callers can map billing/
            # auth/availability distinctly.
            body_bytes = b""
            try:
                body_bytes = exc.read(_MAX_BODY_BYTES)
            except Exception:
                pass
            return HttpResult(status=exc.code, body=body_bytes)
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
            raise VerbatimError(
                ErrorCode.ENCODER_UNAVAILABLE,
                f"cloudflare encoder unreachable: {exc}",
                retryable=True,
            ) from exc


class CloudflareEncoder(RemoteEncoder):
    """Encoder backed by Cloudflare Workers AI (e.g. ``@cf/baai/bge-m3``).

    Construction validates only configuration — the token is resolved
    lazily at encode time so a missing credential is a per-call degraded
    signal, not a construction failure.

    ``broker`` is the TransportBroker this encoder dispatches through.
    Without one, ``encode`` raises ``EGRESS_DENIED`` before any I/O —
    the permit requirement is enforced INSIDE the encoder so an unwired
    call site cannot accidentally dispatch (F4-04).
    """

    def __init__(
        self,
        cfg: EmbeddingConfig,
        *,
        account_id: str,
        secret_getter: Callable[[str], Optional[str]],
        http: Optional[HttpTransport] = None,
        broker: Optional[Any] = None,
    ) -> None:
        require_id(account_id, "account_id")
        self._model = cfg.model
        self._artifact_revision = cfg.artifact_revision
        self._account_id = account_id
        self._secret = secret_getter
        self._http: HttpTransport = http if http is not None else _UrllibTransport()
        self._broker = broker
        self._dimensions: Optional[int] = None

    # -- identity ---------------------------------------------------------

    @property
    def encoder_id(self) -> str:
        return encoder_identity("cloudflare", self._model, self._artifact_revision)

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            raise VerbatimError(
                ErrorCode.ENCODER_UNAVAILABLE,
                "encoder dimensions undiscovered — encode() has not succeeded yet",
            )
        return self._dimensions

    @property
    def normalization(self) -> str:
        return "l2"

    def manifest(self) -> dict[str, Any]:
        return {
            "artifact_revision": self._artifact_revision or "unpinned",
            "dimensions": self._dimensions or 0,
            "normalization": self.normalization,
            "license_id": None,
            "preprocessing_version": PREPROCESSING_VERSION,
            "manifest_json": {
                "backend": "cloudflare",
                "model": self._model,
                "endpoint_host": _HOST,
                "api": "POST /client/v4/accounts/{account}/ai/run",
            },
        }

    # -- contract ---------------------------------------------------------

    def available(self) -> bool:
        """Local-only: credential presence. Never performs network I/O.

        Consent/budget/egress state is deliberately NOT probed here — that
        is the caller's EgressGate decision per dispatch.
        """
        try:
            return bool(self._secret(SECRET_NAME))
        except Exception:
            return False

    def _token(self) -> str:
        token = self._secret(SECRET_NAME)
        if not token:
            raise VerbatimError(
                ErrorCode.ENCODER_UNAVAILABLE,
                f"{SECRET_NAME} unavailable in host secrets",
                retryable=False,
            )
        return token

    # -- transport permit seam (SPEC_V4 §12) -------------------------------

    @property
    def endpoint_id(self) -> str:
        """Allowlist identity for this transport — consent rows and permit
        recipients key on it."""
        return f"cloudflare:{_HOST}"

    def endpoint_descriptor(self) -> Any:
        """The EndpointDescriptor this encoder requires in the broker's
        allowlist: TLS-verified, no redirects, no credential forwarding,
        account-bound (V4-12.07)."""
        from ..privacy.broker import EndpointDescriptor

        return EndpointDescriptor(
            endpoint_id=self.endpoint_id,
            origin=f"https://{_HOST}",
            require_tls=True,
            allow_redirects=False,
            forward_credentials=False,
            max_request_bytes=_MAX_BODY_BYTES,
            account=self._account_id,
        )

    def request_payload(self, texts: list[str]) -> bytes:
        """The exact body ``encode`` POSTs — permit digests cover this."""
        return json.dumps(
            {"model": self._model, "input": {"text": list(texts)}}
        ).encode("utf-8")

    def encode(self, texts: list[str], *, permit: Any = None) -> list[bytes]:
        if not texts:
            return []
        if len(texts) > _MAX_TEXTS_PER_CALL:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"cloudflare batch {len(texts)} exceeds {_MAX_TEXTS_PER_CALL}",
            )
        url = f"https://{_HOST}" + _PATH.format(account=self._account_id)
        payload = self.request_payload(texts)
        # V4-12.03: the one-use permit is consumed immediately before
        # ownership transfers to transport — consent/suppression/reservation
        # are rechecked inside broker.dispatch's transaction. No permit,
        # no socket — and no credential material is even read.
        self._authorize_transport(permit, payload)
        token = self._token()
        try:
            result = self._http.request(
                "POST",
                url,
                body=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                timeout_s=_TOTAL_TIMEOUT_S,
            )
        except VerbatimError:
            raise
        except Exception as exc:
            raise VerbatimError(
                ErrorCode.ENCODER_UNAVAILABLE, f"cloudflare encode failed: {exc}",
                retryable=True,
            ) from exc
        if result.status in (401, 403):
            raise VerbatimError(
                ErrorCode.REMOTE_AUTH,
                f"cloudflare encoder auth failed (HTTP {result.status})",
                retryable=False,
            )
        if result.status == 402:
            raise VerbatimError(
                ErrorCode.REMOTE_BILLING,
                "cloudflare encoder billing required (HTTP 402)",
                retryable=False,
            )
        if result.status != 200:
            raise VerbatimError(
                ErrorCode.ENCODER_UNAVAILABLE,
                f"cloudflare encoder returned HTTP {result.status}",
                retryable=result.status >= 500,
            )
        try:
            body = json.loads(result.body)
        except (json.JSONDecodeError, ValueError) as exc:
            raise VerbatimError(
                ErrorCode.VECTOR_INVALID, "cloudflare encoder returned invalid JSON"
            ) from exc
        if not isinstance(body, dict) or body.get("success") is not True:
            raise VerbatimError(
                ErrorCode.VECTOR_INVALID, "cloudflare encoder reported failure"
            )
        data = body.get("result", {}).get("data") if isinstance(body.get("result"), dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise VerbatimError(
                ErrorCode.VECTOR_INVALID,
                "cloudflare encoder returned a mismatched embeddings list",
            )
        blobs: list[bytes] = []
        for vec in data:
            if not isinstance(vec, Sequence) or isinstance(vec, (str, bytes)):
                raise VerbatimError(
                    ErrorCode.VECTOR_INVALID, "embedding is not a number list"
                )
            n = len(vec)
            if self._dimensions is None:
                self._dimensions = n
            elif n != self._dimensions:
                raise VerbatimError(
                    ErrorCode.VECTOR_INVALID,
                    f"declared dims {self._dimensions} != actual vector length {n}",
                )
            Float32Codec.validate(vec, n)
            blobs.append(Float32Codec.pack(vec))
        return blobs


__all__ = [
    "CloudflareEncoder",
    "HttpResult",
    "HttpTransport",
    "PREPROCESSING_VERSION",
    "SECRET_NAME",
]
