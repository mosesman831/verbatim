"""Shared honest HTTP core for the v4 loopback surfaces (SPEC_V4 §45, §49).

One stdlib ``http.server`` transport shared by the public service API
(``service.api``) and the operator workbench (``workbench.app``). It owns
only transport concerns — socket lifecycle, request bounds, JSON framing,
typed-error → status mapping, security headers — and delegates every
request to an ``Application``. There is no business logic here and no
authorization policy: authentication is the app's, authorization is the
engine's.

Transport rules honored:

- V4-45.07: bounded bodies, strict JSON, typed errors, serialized output
  caps.
- V4-45.08: loopback by default; importing this module binds nothing and
  a non-loopback bind requires an explicit ``allow_remote`` plus TLS is
  honestly reported as unimplemented rather than silently downgraded.
- V4-52/§09.09: missing and forbidden are one public shape — every
  ``NOT_FOUND_OR_*``/held denial returns the identical status and body.
- V4-49.08: JSON responses and the workbench's HTML both carry
  ``nosniff``/``no-store``/``frame-denied`` headers; HTML pages get a
  scriptless CSP.
"""

from __future__ import annotations

import json
import re
import socket
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterable, Optional

from ..core.types import ErrorCode, VerbatimError

#: Loopback bind addresses accepted without an explicit override.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

#: Error → HTTP status. Denial codes all collapse to the same 404 shape:
#: the response body is built from a fixed template, never from the
#: exception's message, so a forbidden object and a missing one are
#: byte-identical on the wire (§09.09, §10.05, §52).
_DENIAL_CODES = frozenset(
    {
        ErrorCode.NOT_FOUND_OR_FORBIDDEN,
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
        ErrorCode.QUARANTINED,
    }
)

_STATUS_FOR = {
    ErrorCode.VALIDATION: 400,
    ErrorCode.CONFIG_INVALID: 400,
    ErrorCode.DECISION_INVALID: 400,
    ErrorCode.CAPTURE_DISABLED: 403,
    ErrorCode.CONSENT_REQUIRED: 403,
    ErrorCode.EGRESS_DENIED: 403,
    ErrorCode.EGRESS_DISABLED: 403,
    ErrorCode.RETENTION_DENIED: 403,
    ErrorCode.PERMIT_EXPIRED: 403,
    ErrorCode.DEADLINE_EXCEEDED: 408,
    ErrorCode.STALE_PROPOSAL: 409,
    ErrorCode.STALE_EPOCH: 409,
    ErrorCode.STALE_DEPENDENCY: 409,
    ErrorCode.INVALID_TRANSITION: 409,
    ErrorCode.OPERATION_CONFLICT: 409,
    ErrorCode.LEASE_LOST: 409,
    ErrorCode.CANCELLED: 409,
    ErrorCode.CLOSURE_PENDING: 409,
    ErrorCode.ERASURE_UNPROVEN: 409,
    ErrorCode.PROCESSING_PENDING: 409,
    ErrorCode.ENVIRONMENT_MISMATCH: 409,
    ErrorCode.APPLICABILITY_UNKNOWN: 409,
    ErrorCode.BACKPRESSURE: 429,
    ErrorCode.BUDGET_EXCEEDED: 429,
    ErrorCode.BUDGET_EXHAUSTED: 429,
    ErrorCode.STORE_BUSY: 503,
    ErrorCode.CAPABILITY_UNAVAILABLE: 503,
    ErrorCode.ENCODER_UNAVAILABLE: 503,
    ErrorCode.EVIDENCE_UNAVAILABLE: 503,
    ErrorCode.COMPILATION_UNSUPPORTED: 503,
    ErrorCode.INVESTIGATION_UNSUPPORTED: 503,
    ErrorCode.REMOTE_BUSY: 503,
    ErrorCode.REMOTE_AUTH: 502,
    ErrorCode.REMOTE_BILLING: 502,
    ErrorCode.MODEL_DRIFT: 503,
    ErrorCode.CONTEXT_INCOMPLETE: 409,
    ErrorCode.LOCKED: 409,
    ErrorCode.RETRYABLE_OPERATION: 503,
    ErrorCode.STORE_CORRUPT: 500,
    ErrorCode.INTEGRITY: 500,
    ErrorCode.STORE_WRITE_FAILED: 500,
    ErrorCode.STORE_CONFLICT: 409,
    ErrorCode.CHECKPOINT_FAILED: 500,
    ErrorCode.SCHEMA_UNSUPPORTED: 500,
}

_DENIAL_BODY = {
    "code": ErrorCode.NOT_FOUND_OR_UNAUTHORIZED.value,
    "message": "not found or unauthorized",
    "retryable": False,
}


@dataclass(frozen=True)
class HttpConfig:
    """Transport knobs for one loopback HTTP surface.

    ``allow_remote`` is the explicit non-loopback opt-in (V4-45.08). TLS
    is NOT implemented in this reference transport — a non-loopback bind
    is cleartext and must be refused unless the caller explicitly accepts
    it; ``serve`` reports that honestly via ``tls="unimplemented"`` in the
    startup banner rather than claiming HTTPS.
    """

    host: str = "127.0.0.1"
    port: int = 0
    allow_remote: bool = False
    max_body_bytes: int = 1 << 20
    max_response_bytes: int = 4 << 20
    max_header_bytes: int = 16 << 10
    request_timeout_s: float = 30.0
    name: str = "verbatim-service"

    def validate(self) -> "HttpConfig":
        host = (self.host or "").strip() or "127.0.0.1"
        object.__setattr__(self, "host", host)
        if host not in _LOOPBACK_HOSTS and not self.allow_remote:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                f"HTTP bind host {host!r} is not loopback; pass "
                "allow_remote=True to accept a cleartext non-loopback bind "
                "(TLS is unimplemented in this transport)",
            )
        if not 0 <= int(self.port) <= 65535:
            raise VerbatimError(ErrorCode.CONFIG_INVALID, "port out of range")
        if self.max_body_bytes < 1024 or self.max_body_bytes > (64 << 20):
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID, "max_body_bytes out of range"
            )
        return self


@dataclass(frozen=True)
class Request:
    """One decoded HTTP request handed to an Application."""

    method: str
    path: str
    query: dict[str, str]
    headers: dict[str, str]
    body: Any            # decoded JSON (or None when no body)
    raw_body: bytes
    correlation_id: str
    client: str

    def header(self, name: str) -> Optional[str]:
        return self.headers.get(name.lower())


@dataclass(frozen=True)
class Response:
    """One serialized HTTP response from an Application."""

    status: int = 200
    body: Any = None                       # dict/list → JSON; str → text
    content_type: str = "application/json"
    headers: dict[str, str] = field(default_factory=dict)


class HttpError(VerbatimError):
    """A transport-level refusal carrying an explicit status."""

    def __init__(self, status: int, message: str, *, code: ErrorCode = ErrorCode.VALIDATION):
        super().__init__(code, message)
        self.status = status


def error_body(code: str, message: str, retryable: bool, correlation_id: str) -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
            "retryable": bool(retryable),
            "correlation_id": correlation_id,
        }
    }


def error_response(exc: BaseException, correlation_id: str) -> Response:
    """Map a failure to its wire shape (§52).

    Denials collapse to the fixed 404 body — the exception's message is
    deliberately dropped so existence/authority stay indistinguishable.
    Unexpected exceptions surface as a bare 500 with the correlation id
    only; internals never reach the wire.
    """
    if isinstance(exc, HttpError):
        if exc.status == 404:
            body = error_body(
                _DENIAL_BODY["code"], _DENIAL_BODY["message"], False, correlation_id
            )
        else:
            body = error_body(exc.code.value, exc.message, exc.retryable, correlation_id)
        return Response(status=exc.status, body=body)
    if isinstance(exc, VerbatimError):
        if exc.code in _DENIAL_CODES:
            return Response(
                status=404,
                body=error_body(
                    _DENIAL_BODY["code"], _DENIAL_BODY["message"], False, correlation_id
                ),
            )
        status = _STATUS_FOR.get(exc.code, 500)
        # A safe detail is the typed code + message; the message never
        # carries object internals for denial codes (handled above).
        return Response(
            status=status,
            body=error_body(exc.code.value, exc.message, exc.retryable, correlation_id),
        )
    return Response(
        status=500,
        body=error_body("INTERNAL", "internal error", False, correlation_id),
    )


def denial() -> VerbatimError:
    return VerbatimError(
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "not found or unauthorized"
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

_SEGMENT = re.compile(r"^[A-Za-z0-9._~:-]+$")


class Router:
    """Method + path-template routing with ``{name}`` segments."""

    def __init__(self) -> None:
        self._routes: list[tuple[str, tuple[str, ...], Callable]] = []

    def add(self, method: str, template: str, handler: Callable) -> None:
        parts = tuple(p for p in template.split("/") if p != "")
        for p in parts:
            if p.startswith("{"):
                if not p.endswith("}"):
                    raise VerbatimError(
                        ErrorCode.CONFIG_INVALID, f"bad route segment {p!r}"
                    )
            elif not _SEGMENT.match(p):
                raise VerbatimError(
                    ErrorCode.CONFIG_INVALID, f"bad route segment {p!r}"
                )
        self._routes.append((method.upper(), parts, handler))

    def match(self, method: str, path: str) -> tuple[Callable, dict[str, str]]:
        """Return ``(handler, params)`` or raise the shared 404 denial.

        Unknown paths and wrong methods share the same denial shape — a
        route probe cannot distinguish "no such route" from "no such
        object" or "not for you".
        """
        parts = tuple(p for p in path.split("/") if p != "")
        for m, template, handler in self._routes:
            if m != method.upper() or len(template) != len(parts):
                continue
            params: dict[str, str] = {}
            ok = True
            for tseg, pseg in zip(template, parts):
                if tseg.startswith("{"):
                    params[tseg[1:-1]] = pseg
                elif tseg != pseg:
                    ok = False
                    break
            if ok:
                return handler, params
        raise denial()


class Application:
    """Interface both surfaces implement: authenticate + dispatch."""

    realm = "verbatim"

    def authenticate(self, request: Request) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def dispatch(self, request: Request, auth: Any) -> Response:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Request handler + server plumbing
# ---------------------------------------------------------------------------


def _parse_query(path: str) -> tuple[str, dict[str, str]]:
    from urllib.parse import unquote_plus, urlsplit

    split = urlsplit(path)
    query: dict[str, str] = {}
    if split.query:
        for pair in split.query.split("&"):
            if not pair:
                continue
            key, _, val = pair.partition("=")
            key = unquote_plus(key)
            # First occurrence wins; duplicates are caller error noise, not
            # a merge — keep it deterministic and small.
            if key not in query:
                query[key] = unquote_plus(val)
    return split.path or "/", query


class _Handler(BaseHTTPRequestHandler):
    """Translates wire bytes → Application calls → wire bytes.

    Fail-closed at every layer: bounded header/body reads, strict JSON,
    no default HTML error pages, no request logging to stdout (access
    detail goes to the server's quiet logger hook if provided).
    """

    server_version = "verbatim-http/4"
    protocol_version = "HTTP/1.1"
    timeout = 30

    def log_message(self, fmt: str, *args: Any) -> None:  # silence default stderr log
        hook = getattr(self.server, "log_hook", None)
        if hook is not None:
            try:
                hook("access", fmt % args)
            except Exception:
                pass

    # BaseHTTPRequestHandler routes every verb through do_* methods.
    def do_GET(self) -> None:  # noqa: N802
        self._handle()

    def do_POST(self) -> None:  # noqa: N802
        self._handle()

    def do_PUT(self) -> None:  # noqa: N802
        self._handle()

    def do_DELETE(self) -> None:  # noqa: N802
        self._handle()

    def do_PATCH(self) -> None:  # noqa: N802
        self._handle()

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle()

    def _handle(self) -> None:
        cfg: HttpConfig = self.server.http_config  # type: ignore[attr-defined]
        correlation_id = uuid.uuid4().hex[:16]
        self.connection.settimeout(cfg.request_timeout_s)
        try:
            request = self._decode_request(cfg, correlation_id)
        except HttpError as exc:
            self._write(error_response(exc, correlation_id))
            return
        except (socket.timeout, TimeoutError):
            self._write(error_response(
                HttpError(408, "request timeout", code=ErrorCode.DEADLINE_EXCEEDED),
                correlation_id,
            ))
            return
        except Exception:
            self._write(error_response(
                VerbatimError(ErrorCode.VALIDATION, "malformed request"),
                correlation_id,
            ))
            return
        try:
            app: Application = self.server.application  # type: ignore[attr-defined]
            auth = app.authenticate(request)
            if auth is None:
                # Unauthenticated: one fixed shape + the Bearer challenge.
                response = Response(
                    status=401,
                    body=error_body(
                        "UNAUTHENTICATED",
                        "authentication required",
                        False,
                        correlation_id,
                    ),
                    headers={
                        "WWW-Authenticate": f'Bearer realm="{app.realm}"'
                    },
                )
            else:
                response = app.dispatch(request, auth)
        except Exception as exc:
            response = error_response(exc, correlation_id)
        self._write(response)

    def _decode_request(self, cfg: HttpConfig, correlation_id: str) -> Request:
        path, query = _parse_query(self.path or "/")
        headers = {k.lower(): v for k, v in self.headers.items()}
        raw = b""
        body: Any = None
        if self.command in ("POST", "PUT", "PATCH", "DELETE"):
            length_raw = headers.get("content-length")
            if length_raw is None:
                raise HttpError(411, "content-length required")
            try:
                length = int(length_raw)
            except ValueError:
                raise HttpError(400, "invalid content-length") from None
            if length < 0 or length > cfg.max_body_bytes:
                raise HttpError(413, "request body exceeds bound")
            raw = self.rfile.read(length) if length else b""
            if raw:
                ctype = (headers.get("content-type") or "").split(";")[0].strip().lower()
                if ctype == "application/json":
                    try:
                        body = json.loads(raw.decode("utf-8"))
                    except (ValueError, UnicodeDecodeError):
                        raise HttpError(400, "body is not valid UTF-8 JSON") from None
                elif ctype == "application/x-www-form-urlencoded":
                    # Operator-UI form posts — a flat string mapping, first
                    # occurrence wins (parse_qsl, bounded by max_body_bytes).
                    from urllib.parse import parse_qsl

                    try:
                        body = dict(
                            parse_qsl(
                                raw.decode("utf-8"),
                                keep_blank_values=True,
                                max_num_fields=256,
                            )
                        )
                    except (ValueError, UnicodeDecodeError):
                        raise HttpError(400, "body is not valid form data") from None
                else:
                    raise HttpError(
                        415,
                        "content-type must be application/json or "
                        "application/x-www-form-urlencoded",
                    )
        elif "content-length" in headers:
            try:
                if int(headers["content-length"] or "0") > 0:
                    raise HttpError(400, "GET/HEAD requests carry no body")
            except ValueError:
                raise HttpError(400, "invalid content-length") from None
        return Request(
            method=self.command,
            path=path,
            query=query,
            headers=headers,
            body=body,
            raw_body=raw,
            correlation_id=correlation_id,
            client=self.client_address[0] if self.client_address else "",
        )

    def _write(self, resp: Response) -> None:
        try:
            if isinstance(resp.body, (bytes, bytearray)):
                payload = bytes(resp.body)
                ctype = resp.content_type
            elif resp.body is None:
                payload = b""
                ctype = resp.content_type
            elif isinstance(resp.body, str):
                payload = resp.body.encode("utf-8")
                ctype = resp.content_type or "text/html; charset=utf-8"
            else:
                payload = json.dumps(
                    resp.body, ensure_ascii=False, default=str
                ).encode("utf-8")
                ctype = "application/json; charset=utf-8"
            cfg: HttpConfig = self.server.http_config  # type: ignore[attr-defined]
            if len(payload) > cfg.max_response_bytes:
                # Serialized output budget (V4-45.07): swap the oversized
                # body for a small typed error — never truncate content.
                payload = json.dumps(
                    error_body(
                        ErrorCode.BUDGET_EXCEEDED.value,
                        "response exceeds output budget",
                        False,
                        "",
                    )
                ).encode("utf-8")
                ctype = "application/json; charset=utf-8"
                resp = Response(status=500, body=payload, content_type=ctype)
            self.send_response(resp.status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            if ctype.startswith("text/html"):
                # No scripts ever; inline style only (V4-49.08).
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; style-src 'unsafe-inline'; "
                    "base-uri 'none'; form-action 'self'",
                )
            for k, v in resp.headers.items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD" and payload:
                self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass  # client vanished mid-reply — nothing more to say


class HttpServer(ThreadingHTTPServer):
    """Threaded loopback server bound to one Application."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        cfg: HttpConfig,
        application: Application,
        *,
        log_hook: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.http_config = cfg.validate()
        self.application = application
        self.log_hook = log_hook
        super().__init__((self.http_config.host, self.http_config.port), _Handler)

    @property
    def bound_address(self) -> tuple[str, int]:
        return self.server_address[0], self.server_address[1]


def create_server(
    application: Application, cfg: Optional[HttpConfig] = None
) -> HttpServer:
    """Bind a loopback HTTP server for ``application``. No serve loop yet."""
    return HttpServer(cfg or HttpConfig(), application)


class RunningServer:
    """A bound server + its serve thread; the handle tests and CLI share."""

    def __init__(self, httpd: HttpServer) -> None:
        self.httpd = httpd
        self._thread: Optional[threading.Thread] = None

    @property
    def host(self) -> str:
        return self.httpd.bound_address[0]

    @property
    def port(self) -> int:
        return self.httpd.bound_address[1]

    @property
    def url(self) -> str:
        host = self.host
        if ":" in host:  # IPv6 loopback
            host = f"[{host}]"
        return f"http://{host}:{self.port}"

    def start(self) -> "RunningServer":
        if self._thread is None:
            self._thread = threading.Thread(
                target=self.httpd.serve_forever,
                name=f"{self.httpd.http_config.name}-serve",
                daemon=True,
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "RunningServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


__all__ = [
    "Application",
    "HttpConfig",
    "HttpError",
    "HttpServer",
    "Request",
    "Response",
    "Router",
    "RunningServer",
    "create_server",
    "denial",
    "error_body",
    "error_response",
]
