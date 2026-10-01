"""V5 consumer HTTP surface (docs/v6_contracts.md §4, SPEC_V6 §05).

The Mem0-shaped front door on the existing loopback transport: one
``Memory`` facade per app, constructed at server start and bound to the
service identity the operator pinned — ``user_id`` resolves to the
credential-carried principal, never to anything in a request payload.

Routes (POST unless GET)::

    /v2/memory/add        {text, infer?, metadata?, replaces?}
    /v2/memory/search     {query, limit?, consistency?, timeout_ms?,
                           ready_timeout_ms?, after?}
    /v2/memory/inspect    {ref, detail?}
    /v2/memory/forget     {ref, confirm_token?, preview?}
    /v2/memory/status
    /v2/memory/readiness/{receipt_id}
    /v2/memory/capabilities

Binding rules (single-authority, §4):

* ``create_memory_app`` binds ONE ``Memory`` to one identity. Every
  provisioned credential must be *usable* by that binding or the launch
  is a misconfiguration — a credential naming another principal, or
  carrying an operator scope pin (workspace/conversation/visibility)
  this surface cannot honor, is "scoped elsewhere".
* At request time the bearer credential is re-checked against the bound
  identity; a mismatch is a flat 403, and the credential's verb ceiling
  (``auth.py`` — the token's maximum grant class) is enforced per route.
* Payloads can never mint identity: strict field allowlists reject
  ``user_id``/``namespace``/``principal_id``/``scope`` keys outright.

Wire rules: bodies parse through ``core.serialize.safe_json_loads``
(strict — bounded, no NaN, no duplicate keys); responses serialize
through ``json_dumps`` with enum ``.value`` mapping and byte fields
hex-encoded, never raw. Failures are flat ``{"error", "code",
"retryable"}`` dicts — 400 invalid, 403 scope-binding, 404
indistinguishable denial, 409 conflict, 503 unavailable — and an
unexpected failure is a bare ``500 {"error": "internal"}``, never a
traceback.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any, Callable, Iterable, Optional

from ..core.serialize import json_dumps, safe_json_loads
from ..core.types import ErrorCode, VerbatimError
from ..core.types_v3 import Verb
from ..memory.facade import (
    Memory,
    _MAX_CONTENT_BYTES,
    _MAX_LIMIT,
    _MAX_METADATA_BYTES,
)
from ..memory.types import CONTRACT
from .auth import (
    TokenAuthenticator,
    TokenCredential,
    parse_credentials,
    scope_mismatch,
    tokens_from_env,
)
from .httpd import (
    Application,
    HttpConfig,
    HttpServer,
    Request,
    Response,
    Router,
    _DENIAL_CODES,
    _STATUS_FOR,
)

#: Canonical JSON content type for every response this surface writes.
_JSON = "application/json; charset=utf-8"

#: Request-field bounds (transport mirrors; the facade re-validates).
_MAX_ID = 256
_MAX_STR = 8192
_MAX_REF = 4096
_MAX_TIMEOUT_MS = 120_000.0

#: The route → verb ceiling each credential must carry (``auth.py``
#: model: ``verbs`` is the token's maximum grant class — the bound
#: Memory's owner grant is the authority, the token narrows it).
_ROUTE_VERBS = {
    "add": Verb.INGEST.value,
    "search": Verb.READ.value,
    "inspect": Verb.READ.value,
    "forget": Verb.ADMIN.value,
    "status": Verb.READ.value,
    "readiness": Verb.READ.value,
    "capabilities": Verb.READ.value,
}


# ---------------------------------------------------------------------------
# errors + serialization
# ---------------------------------------------------------------------------


def _bad(msg: str) -> VerbatimError:
    return VerbatimError(ErrorCode.VALIDATION, msg)


def _flat(code: str, message: str, retryable: bool) -> dict:
    return {"error": message, "code": code, "retryable": bool(retryable)}


def _flat_response(status: int, code: str, message: str, retryable: bool = False) -> Response:
    return Response(
        status=status,
        body=json_dumps(_flat(code, message, retryable)),
        content_type=_JSON,
    )


def _forbidden(message: str = "credential is scoped elsewhere") -> Response:
    """Scope-binding refusal (§4) — a 403, never a route's own verdict."""
    return _flat_response(403, "FORBIDDEN", message)


def error_response_flat(exc: BaseException) -> Response:
    """``VerbatimError`` → flat ``{error, code, retryable}`` + status by
    class; anything else → a bare 500 with no internals (§4)."""
    if isinstance(exc, VerbatimError):
        if exc.code in _DENIAL_CODES:
            # Missing/forbidden/held collapse to one shape — the
            # exception's message is deliberately dropped (§52).
            return _flat_response(
                404,
                ErrorCode.NOT_FOUND_OR_UNAUTHORIZED.value,
                "not found or unauthorized",
            )
        status = _STATUS_FOR.get(exc.code, 500)
        return _flat_response(status, exc.code.value, exc.message, exc.retryable)
    return _flat_response(500, "INTERNAL", "internal")


def _jsonable(value: Any) -> Any:
    """Result → JSON-safe tree: dataclass → dict, enum → ``.value``,
    bytes → hex (never raw), non-finite floats → null."""
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return _jsonable(to_dict())
        return {
            f.name: _jsonable(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _serialize(result: Any) -> Response:
    return Response(
        status=200,
        body=json_dumps(_jsonable(result)),
        content_type=_JSON,
    )


# ---------------------------------------------------------------------------
# request validation (same idiom as service.api)
# ---------------------------------------------------------------------------


def _body(request: Request) -> dict:
    """Strict body decode — ``safe_json_loads`` over the raw bytes, so
    duplicate keys / NaN / depth bombs are typed rejections."""
    raw = request.raw_body or b""
    if not raw:
        raise _bad("request body must be a JSON object")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _bad("request body is not UTF-8") from exc
    data = safe_json_loads(text)
    if not isinstance(data, dict):
        raise _bad("request body must be a JSON object")
    return data


def _fields(body: dict, allowed: Iterable[str]) -> dict:
    unknown = set(body) - set(allowed)
    if unknown:
        raise _bad(f"unknown request fields {sorted(unknown)}")
    return body


def _str(body: dict, key: str, *, required: bool = False, max_len: int = _MAX_STR) -> Optional[str]:
    v = body.get(key)
    if v is None:
        if required:
            raise _bad(f"{key} is required")
        return None
    if not isinstance(v, str) or not v:
        raise _bad(f"{key} must be a non-empty string")
    if len(v) > max_len:
        raise _bad(f"{key} exceeds {max_len} characters")
    return v


def _int(body: dict, key: str, *, required: bool = False,
         lo: Optional[int] = None, hi: Optional[int] = None) -> Optional[int]:
    v = body.get(key)
    if v is None:
        if required:
            raise _bad(f"{key} is required")
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        raise _bad(f"{key} must be an integer")
    if lo is not None and v < lo:
        raise _bad(f"{key} must be >= {lo}")
    if hi is not None and v > hi:
        raise _bad(f"{key} must be <= {hi}")
    return v


def _num(body: dict, key: str, *, lo: float = 0.0, hi: float = _MAX_TIMEOUT_MS) -> Optional[float]:
    v = body.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise _bad(f"{key} must be a number")
    if not math.isfinite(float(v)) or float(v) < lo or float(v) > hi:
        raise _bad(f"{key} must be in [{lo}, {hi}]")
    return float(v)


def _bool(body: dict, key: str) -> Optional[bool]:
    v = body.get(key)
    if v is None:
        return None
    if not isinstance(v, bool):
        raise _bad(f"{key} must be boolean")
    return v


def _query_num(request: Request, key: str, *, default: float,
               lo: float = 0.0, hi: float = _MAX_TIMEOUT_MS) -> float:
    raw = request.query.get(key)
    if raw is None:
        return default
    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise _bad(f"{key} must be a number") from None
    if not math.isfinite(v) or v < lo or v > hi:
        raise _bad(f"{key} must be in [{lo}, {hi}]")
    return v


# ---------------------------------------------------------------------------
# credential binding (§4)
# ---------------------------------------------------------------------------


def resolve_credentials(tokens: Any) -> tuple[TokenCredential, ...]:
    """``tokens`` may be TokenCredential iterable, a token document
    (``{"tokens": [...]}`` / bare list), or None → environment."""
    if tokens is None:
        creds = tokens_from_env()
    elif isinstance(tokens, TokenCredential):
        creds = (tokens,)
    elif isinstance(tokens, dict):
        creds = parse_credentials(tokens)
    elif isinstance(tokens, (list, tuple)):
        items = tuple(tokens)
        if all(isinstance(t, TokenCredential) for t in items):
            creds = items
        else:
            creds = parse_credentials(list(items))
    else:
        creds = tuple(tokens)
    if not creds:
        raise VerbatimError(
            ErrorCode.CONFIG_INVALID,
            "no bearer tokens provisioned for the memory surface",
        )
    for c in creds:
        if not isinstance(c, TokenCredential):
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "tokens must be TokenCredential entries or a token document",
            )
    return creds


def resolve_bound(
    creds: Iterable[TokenCredential],
    user_id: Optional[str],
) -> str:
    """The principal this app serves — ``user_id`` when pinned, else the
    single principal shared by every credential.

    At least one credential must actually bind (matching principal, no
    foreign scope pin); a file that serves nobody is a CONFIG_INVALID
    launch-time refusal, never a listener that can only 403.
    """
    bound = user_id
    if bound is None:
        principals = {c.principal_id for c in creds}
        if len(principals) != 1:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "credentials name multiple principals — pass user_id to "
                "pin which one this memory app serves",
            )
        bound = principals.pop()
    if not isinstance(bound, str) or not bound:
        raise VerbatimError(
            ErrorCode.CONFIG_INVALID, "user_id must be a non-empty string"
        )
    usable = [c for c in creds if scope_mismatch(c, bound) is None]
    if not usable:
        raise VerbatimError(
            ErrorCode.CONFIG_INVALID,
            "no provisioned credential binds to this app's scope — "
            "every token is scoped elsewhere",
        )
    return bound


# ---------------------------------------------------------------------------
# the application
# ---------------------------------------------------------------------------


class MemoryApp(Application):
    """``/v2/memory/*`` bound to one ``Memory`` + one credential set.

    Authentication is the shared bearer model; authorization is the
    bound facade's (every call executes as the Memory's owner inside its
    namespace) — this layer contributes only the credential/scope
    binding check and the verb ceiling, never a parallel policy.
    """

    realm = "verbatim-memory"

    def __init__(
        self,
        memory: Memory,
        credentials: Iterable[TokenCredential],
        *,
        bound: str,
        http_cfg: Optional[HttpConfig] = None,
    ) -> None:
        self._memory = memory
        self._bound = bound
        self._auth = TokenAuthenticator(credentials)
        self._http_cfg = http_cfg or HttpConfig()
        self._router = Router()
        self._verbs: dict[Callable, str] = {}
        self._routes()

    @property
    def memory(self) -> Memory:
        """The bound facade — callers close it on shutdown."""
        return self._memory

    # -- auth --------------------------------------------------------------

    def authenticate(self, request: Request) -> Optional[TokenCredential]:
        return self._auth.authenticate(request.header("authorization"))

    def _bind_check(self, cred: TokenCredential) -> Optional[Response]:
        """The credential must bind to this app's pinned identity —
        a foreign principal or an unhonorable scope pin is a 403."""
        why = scope_mismatch(cred, self._bound)
        if why is not None:
            return _forbidden(
                "credential is scoped elsewhere — this app serves one "
                "bound identity"
            )
        return None

    def dispatch(self, request: Request, cred: TokenCredential) -> Response:
        try:
            refusal = self._bind_check(cred)
            if refusal is not None:
                return refusal
            handler, params = self._router.match(request.method, request.path)
            verb = self._verbs.get(handler)
            if verb is not None and verb not in cred.verbs:
                return _forbidden(
                    f"credential's verb ceiling does not include {verb!r}"
                )
            result = handler(self, request, cred, params)
            if isinstance(result, Response):
                return result
            return _serialize(result)
        except VerbatimError as exc:
            return error_response_flat(exc)
        except Exception:
            # Bare 500, no internals, no traceback — same refusal every
            # unexpected failure gets (§4).
            return _flat_response(500, "INTERNAL", "internal")

    # -- routes --------------------------------------------------------------

    def _routes(self) -> None:
        def reg(method: str, path: str, fn: Callable, name: str) -> None:
            self._router.add(method, path, fn)
            self._verbs[fn] = _ROUTE_VERBS[name]

        reg("POST", "/v2/memory/add", MemoryApp._add, "add")
        reg("POST", "/v2/memory/search", MemoryApp._search, "search")
        reg("POST", "/v2/memory/inspect", MemoryApp._inspect, "inspect")
        reg("POST", "/v2/memory/forget", MemoryApp._forget, "forget")
        reg("GET", "/v2/memory/status", MemoryApp._status, "status")
        reg("GET", "/v2/memory/readiness/{receipt_id}", MemoryApp._readiness, "readiness")
        reg("GET", "/v2/memory/capabilities", MemoryApp._capabilities, "capabilities")

    # -- endpoint handlers (validation + delegation only) ----------------

    def _add(self, request: Request, cred: TokenCredential, params: dict) -> Any:
        body = _fields(_body(request), ("text", "infer", "metadata", "replaces"))
        text = _str(body, "text", required=True, max_len=_MAX_CONTENT_BYTES)
        infer = _bool(body, "infer")
        metadata = body.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise _bad("metadata must be an object")
        replaces = _str(body, "replaces", max_len=_MAX_REF)
        return self._memory.add(
            text,
            infer=True if infer is None else infer,
            metadata=metadata,
            replaces=replaces,
        )

    def _search(self, request: Request, cred: TokenCredential, params: dict) -> Any:
        body = _fields(
            _body(request),
            ("query", "limit", "consistency", "timeout_ms",
             "ready_timeout_ms", "after"),
        )
        query = _str(body, "query", required=True)
        limit = _int(body, "limit", lo=1, hi=4096)
        consistency = _str(body, "consistency", max_len=32)
        if consistency is not None and consistency not in ("session", "eventual"):
            raise _bad("consistency must be 'session' or 'eventual'")
        timeout_ms = _num(body, "timeout_ms")
        ready_timeout_ms = _num(body, "ready_timeout_ms")
        after = body.get("after")
        if after is not None and not (
            (isinstance(after, str) and after) or isinstance(after, dict)
        ):
            raise _bad("after must be a causal token, receipt id, or receipt reference")
        return self._memory.search(
            query,
            limit=limit if limit is not None else 8,
            consistency=consistency or "session",
            timeout_ms=timeout_ms if timeout_ms is not None else 500,
            ready_timeout_ms=ready_timeout_ms,
            after=after,
        )

    def _inspect(self, request: Request, cred: TokenCredential, params: dict) -> Any:
        body = _fields(_body(request), ("ref", "detail"))
        ref = body.get("ref")
        if isinstance(ref, str):
            if not ref:
                raise _bad("ref is required")
            if len(ref) > _MAX_REF:
                raise _bad(f"ref exceeds {_MAX_REF} characters")
        elif not isinstance(ref, dict):
            raise _bad("ref must be a ref string or ref mapping")
        detail = _str(body, "detail", max_len=32)
        if detail is not None and detail not in ("evidence", "metadata", "enrichment"):
            raise _bad("detail must be 'evidence', 'metadata', or 'enrichment'")
        return self._memory.inspect(ref, detail=detail or "evidence")

    def _forget(self, request: Request, cred: TokenCredential, params: dict) -> Any:
        body = _fields(_body(request), ("ref", "confirm_token", "preview"))
        token = _str(body, "confirm_token", max_len=_MAX_REF)
        preview = _bool(body, "preview")
        ref = body.get("ref")
        if token is not None:
            # A minted confirmation token stands alone — the facade pins
            # the entire selection; ref/preview alongside it is a caller
            # error, never a merged request.
            if ref is not None or preview is not None:
                raise _bad("confirm_token stands alone — do not pass ref/preview")
            return self._memory.forget(confirmation=token)
        if not isinstance(ref, str) or not ref:
            raise _bad("ref is required")
        if len(ref) > _MAX_REF:
            raise _bad(f"ref exceeds {_MAX_REF} characters")
        if preview is True:
            # Preview mode runs the facade's query-forget machinery:
            # ``ref`` carries the selection text and the result is a
            # mutation-free preview + confirmation token (V5-15.04).
            return self._memory.forget(query=ref)
        return self._memory.forget(ref)

    def _status(self, request: Request, cred: TokenCredential, params: dict) -> Any:
        return self._memory.status()

    def _readiness(self, request: Request, cred: TokenCredential, params: dict) -> Any:
        receipt_id = params["receipt_id"]
        if len(receipt_id) > _MAX_ID:
            raise _bad("receipt_id too long")
        timeout_ms = _query_num(request, "timeout_ms", default=0.0)
        return self._memory.wait_ready(receipt_id, timeout_ms=timeout_ms)

    def _capabilities(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        status = self._memory.status()
        return {
            "contract": CONTRACT,
            "profile": status.profile,
            "worker": status.worker,
            "encoder": status.encoder,
            "capabilities": status.capabilities,
            "limits": {
                "max_text_bytes": _MAX_CONTENT_BYTES,
                "max_metadata_bytes": _MAX_METADATA_BYTES,
                "max_limit": _MAX_LIMIT,
                "max_body_bytes": self._http_cfg.max_body_bytes,
            },
            "transport": {
                "scheme": "http",
                "tls": "unimplemented",  # loopback-only reference transport
                "bind": "loopback",
                "authn": "bearer-token",
            },
            "routes": [
                "POST /v2/memory/add",
                "POST /v2/memory/search",
                "POST /v2/memory/inspect",
                "POST /v2/memory/forget",
                "GET /v2/memory/status",
                "GET /v2/memory/readiness/{receipt_id}",
                "GET /v2/memory/capabilities",
            ],
        }


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def create_memory_app(
    path: Optional[str],
    *,
    user_id: Optional[str],
    worker: str = "managed",
    tokens: Any = None,
    config: Any = None,
    http_config: Optional[HttpConfig] = None,
) -> HttpServer:
    """Bind one ``Memory`` + the bearer credential set to a loopback
    ``HttpServer`` (§4). Construction resolves the served identity,
    validates that at least one credential binds to it (a file whose
    tokens are all scoped elsewhere is ``CONFIG_INVALID`` at bind time),
    and opens the store — the returned server is bound but not yet
    serving; run ``serve_forever`` or wrap it in ``RunningServer``.
    """
    creds = resolve_credentials(tokens)
    bound = resolve_bound(creds, user_id)
    memory = Memory(path, user_id=bound, worker=worker, config=config)
    cfg = http_config or HttpConfig(name="verbatim-memory")
    app = MemoryApp(memory, creds, bound=bound, http_cfg=cfg)
    return HttpServer(cfg, app)


__all__ = [
    "MemoryApp",
    "create_memory_app",
    "error_response_flat",
    "resolve_bound",
    "resolve_credentials",
]
