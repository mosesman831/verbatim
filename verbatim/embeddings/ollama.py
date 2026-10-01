"""Ollama loopback encoder — the only network-capable encoder in v1.

Local-service mode permits explicitly allowlisted loopback inference and
nothing else (SPEC §4, §41). Every defense in this module exists because a
configured URL is *attacker-influenceable configuration*, not proof of
locality:

- The endpoint is verified at construction: ``http`` scheme only, host in
  ``{127.0.0.1, localhost, ::1}``, port in ``{11434}`` — a remote host, a
  weird port (e.g. a corporate proxy on :8080), userinfo, paths, or query
  strings all fail closed with ``CONFIG_INVALID``.
- The transport builds an ``OpenerDirector`` containing ONLY
  ``HTTPHandler`` — no ``ProxyHandler`` (environment proxy vars are never
  honored) and no redirect handler (a loopback URL must never 30x out to
  the internet). This is the SPEC §41 loopback verification contract.
- Timeouts are explicit: 5 s to connect, 30 s total including the body
  read — a hung service cannot stall the memory engine indefinitely.
- ``available()`` can never raise; a dead daemon is a degraded capability,
  not an engine failure (SPEC §43).

Response vectors are untrusted input: every returned embedding is packed
through :class:`Float32Codec` and must still pass codec validation at the
persistence call site.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional, Protocol
from urllib.parse import urlsplit

from ..config import EmbeddingConfig
from ..core.types import ErrorCode, VerbatimError
from .codec import Float32Codec
from .encoder import RemoteEncoder, encoder_identity

_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}
_ALLOWED_PORTS = {11434}
_DEFAULT_PORT = 11434
_CONNECT_TIMEOUT_S = 5.0
_TOTAL_TIMEOUT_S = 30.0
_MAX_BODY_BYTES = 32 << 20  # 32 MiB — embeddings responses are far smaller.

# Engine-side request/preprocessing contract revision. Bump when the request
# shape or text normalization changes so stored rows identify their
# generation (SPEC §28).
PREPROCESSING_VERSION = "verbatim-ollama-embed-v1"


@dataclass(frozen=True)
class HttpResult:
    """Minimal transport result so tests can inject a fake transport."""

    status: int
    body: bytes


class HttpTransport(Protocol):
    """Sync request contract; injected via ``OllamaEncoder(http=...)``."""

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
    """Real loopback transport: no proxies, no redirects, bounded reads.

    ``OpenerDirector`` starts EMPTY — unlike ``build_opener()``, which
    installs ``ProxyHandler`` (honoring ``http_proxy`` env vars) and
    ``HTTPRedirectHandler`` (following 30x anywhere, including off-host).
    Adding only ``HTTPHandler`` means the socket can literally only speak
    plain HTTP to the URL it was given.
    """

    def __init__(self) -> None:
        self._opener = urllib.request.OpenerDirector()
        self._opener.add_handler(urllib.request.HTTPHandler())

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
                            "loopback encoder response exceeded total timeout",
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
                            "loopback encoder response exceeds body bound",
                        )
                return HttpResult(status=status, body=b"".join(chunks))
        except urllib.error.HTTPError as exc:
            # Non-2xx is still a definitive response; surface the status.
            return HttpResult(status=exc.code, body=b"")
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
            raise VerbatimError(
                ErrorCode.ENCODER_UNAVAILABLE,
                f"loopback encoder unreachable: {exc}",
                retryable=True,
            ) from exc


def _verify_loopback(endpoint: str) -> str:
    """Validate and normalize the configured endpoint to ``http://host:port``.

    Anything that could route off-loopback or through an unintended
    intermediary fails with ``CONFIG_INVALID`` (SPEC §41). A missing port
    normalizes to the allowlisted default; an explicit non-allowlisted port
    is rejected — ``localhost:8080`` could be an outbound proxy.
    """
    try:
        parts = urlsplit(endpoint)
    except ValueError as exc:
        raise VerbatimError(ErrorCode.CONFIG_INVALID, f"embedding endpoint unparsable: {exc}") from exc
    if parts.scheme != "http":
        raise VerbatimError(
            ErrorCode.CONFIG_INVALID,
            "embedding endpoint must be plain http to a verified loopback host",
        )
    host = (parts.hostname or "").lower()
    if host not in _ALLOWED_HOSTS:
        raise VerbatimError(
            ErrorCode.CONFIG_INVALID,
            f"embedding endpoint host {host!r} is not loopback "
            "(allowed: 127.0.0.1, localhost, ::1)",
        )
    try:
        port = parts.port
    except ValueError as exc:
        raise VerbatimError(ErrorCode.CONFIG_INVALID, "embedding endpoint port invalid") from exc
    if port is None:
        port = _DEFAULT_PORT
    if port not in _ALLOWED_PORTS:
        raise VerbatimError(
            ErrorCode.CONFIG_INVALID,
            f"embedding endpoint port {port} not in allowlist {_ALLOWED_PORTS}",
        )
    if parts.username or parts.password:
        raise VerbatimError(
            ErrorCode.CONFIG_INVALID, "embedding endpoint must not carry credentials"
        )
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise VerbatimError(
            ErrorCode.CONFIG_INVALID,
            "embedding endpoint must be an origin only (no path/query/fragment)",
        )
    display_host = f"[{host}]" if ":" in host else host
    return f"http://{display_host}:{port}"


class OllamaEncoder(RemoteEncoder):
    """Encoder backed by a verified loopback Ollama daemon.

    ``dimensions`` is discovered on the first successful ``encode`` and
    pinned thereafter; a later response whose vector length differs from the
    pinned value is a declared-vs-actual mismatch → ``VECTOR_INVALID``
    (model replacement mid-process must produce a new encoder generation,
    not silently mixed vectors).

    ``broker`` is the TransportBroker this encoder dispatches through
    (SPEC_V4 §12: loopback HTTP is still egress — content leaves the
    process). Without one, ``encode`` raises ``EGRESS_DENIED`` before any
    I/O; an unwired call site cannot dispatch (F4-04). ``available()``
    remains an ungated capability probe: its ``GET /api/tags`` carries an
    empty body — no content crosses — and the method must never raise.
    """

    def __init__(
        self,
        cfg: EmbeddingConfig,
        http: Optional[HttpTransport] = None,
        broker: Optional[Any] = None,
    ) -> None:
        self._model = cfg.model
        self._artifact_revision = cfg.artifact_revision
        self._base = _verify_loopback(cfg.endpoint)
        self._http: HttpTransport = http if http is not None else _UrllibTransport()
        self._broker = broker
        self._dimensions: Optional[int] = None
        self._available_cache: Optional[bool] = None

    # -- identity ---------------------------------------------------------

    @property
    def encoder_id(self) -> str:
        return encoder_identity("ollama", self._model, self._artifact_revision)

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
        # Consumers must L2-normalize before cosine; recorded honestly rather
        # than assumed (SPEC §28).
        return "l2"

    def manifest(self) -> dict[str, Any]:
        return {
            "artifact_revision": self._artifact_revision or "unpinned",
            "dimensions": self._dimensions or 0,
            "normalization": self.normalization,
            # License unverified until a reviewed artifact manifest lands —
            # explicit unknown, not a guess (SPEC §8, §28).
            "license_id": None,
            "preprocessing_version": PREPROCESSING_VERSION,
            "manifest_json": {
                "backend": "ollama",
                "model": self._model,
                "endpoint_host": urlsplit(self._base).hostname,
                "api": "POST /api/embeddings",
            },
        }

    # -- transport permit seam (SPEC_V4 §12) -------------------------------

    @property
    def endpoint_id(self) -> str:
        """Allowlist identity for this transport — consent rows and permit
        recipients key on it."""
        parts = urlsplit(self._base)
        return f"ollama:{parts.hostname}:{parts.port}"

    def endpoint_descriptor(self) -> Any:
        """The EndpointDescriptor this encoder requires in the broker's
        allowlist: verified loopback, plain http, no redirects, no
        credential forwarding (V4-12.07)."""
        from ..privacy.broker import EndpointDescriptor

        return EndpointDescriptor(
            endpoint_id=self.endpoint_id,
            origin=self._base,
            require_tls=False,  # verified-loopback http only
            allow_redirects=False,
            forward_credentials=False,
            max_request_bytes=_MAX_BODY_BYTES,
        )

    def request_payload(self, texts: list[str]) -> bytes:
        """The exact body ``encode`` POSTs — permit digests cover this."""
        return json.dumps({"model": self._model, "input": texts}).encode("utf-8")

    # -- transport --------------------------------------------------------

    def _post_embeddings(self, texts: list[str], payload: bytes) -> HttpResult:
        return self._http.request(
            "POST",
            f"{self._base}/api/embeddings",
            body=payload,
            headers={"Content-Type": "application/json"},
            timeout_s=_TOTAL_TIMEOUT_S,
        )

    def _get_tags(self) -> HttpResult:
        return self._http.request(
            "GET",
            f"{self._base}/api/tags",
            body=None,
            headers={},
            timeout_s=_CONNECT_TIMEOUT_S,
        )

    # -- contract ---------------------------------------------------------

    def available(self) -> bool:
        """One-shot probe of ``/api/tags``; caches the first verdict.

        NEVER raises — availability is a capability signal for status
        reporting, not a gate that may fail the engine (SPEC §43).
        """
        if self._available_cache is not None:
            return self._available_cache
        try:
            result = self._get_tags()
            self._available_cache = 200 <= result.status < 300
        except Exception:
            self._available_cache = False
        return self._available_cache

    def encode(self, texts: list[str], *, permit: Any = None) -> list[bytes]:
        """POST ``/api/embeddings`` and pack each returned vector.

        ``permit`` is a TransportBroker-issued ``DispatchPermit`` bound to
        this endpoint and the exact request body — consumed one-use via
        ``_authorize_transport`` immediately before socket work. Without a
        valid permit the method raises ``EGRESS_DENIED`` and no request is
        ever issued.

        The response is untrusted: count must match the request, each vector
        must match the pinned dimension, and every vector is codec-validated
        here (defense in depth — the persistence call site validates again
        on the exact bytes stored).
        """
        if not texts:
            return []
        payload = self.request_payload(texts)
        # V4-12.03: consent/suppression/reservation recheck + the one-use
        # flip happen inside broker.dispatch — before any socket opens.
        self._authorize_transport(permit, payload)
        try:
            result = self._post_embeddings(texts, payload)
        except VerbatimError:
            raise
        except Exception as exc:
            raise VerbatimError(
                ErrorCode.ENCODER_UNAVAILABLE, f"loopback encode failed: {exc}",
                retryable=True,
            ) from exc
        if result.status != 200:
            raise VerbatimError(
                ErrorCode.ENCODER_UNAVAILABLE,
                f"loopback encoder returned HTTP {result.status}",
                retryable=result.status >= 500,
            )
        try:
            payload = json.loads(result.body)
        except (json.JSONDecodeError, ValueError) as exc:
            raise VerbatimError(
                ErrorCode.VECTOR_INVALID, "loopback encoder returned invalid JSON"
            ) from exc
        vectors = payload.get("embeddings") if isinstance(payload, dict) else None
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise VerbatimError(
                ErrorCode.VECTOR_INVALID,
                "loopback encoder returned a mismatched embeddings list",
            )
        blobs: list[bytes] = []
        for vec in vectors:
            if not isinstance(vec, Sequence) or isinstance(vec, (str, bytes)):
                raise VerbatimError(ErrorCode.VECTOR_INVALID, "embedding is not a number list")
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
    "HttpResult",
    "HttpTransport",
    "OllamaEncoder",
    "PREPROCESSING_VERSION",
]
