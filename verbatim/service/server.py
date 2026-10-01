"""Service lifecycle: credential provisioning + the opt-in serve entry.

``serve(...)`` is the ONLY path that binds a socket (V4-45.08) — importing
``verbatim.service`` creates no listener, opens no file, and provisions
no grants.

Provisioning is the operator's explicit trusted-setup act (V4-45.03):
each credential's principal is registered and issued exactly the verbs
the token file declares, on the credential's bound scope, *only when no
live grant already covers the principal* — a revoked or narrower grant
is never silently widened by re-launching the service (attenuation-only,
mirroring the facade's ``_ensure_owner_grant`` semantics).
"""

from __future__ import annotations

import threading
from typing import Any, Iterable, Optional

from ..core.identity import scope_key
from ..core.types import ErrorCode, VerbatimError
from ..storage.repos import ensure_scope, has_table
from .api import ServiceApp
from .auth import TokenCredential, tokens_from_env
from .httpd import HttpConfig, HttpServer, RunningServer

#: Live servers for observability/tests — proves "no bind on import".
_LIVE: list[RunningServer] = []
_LIVE_LOCK = threading.Lock()


def live_servers() -> tuple[RunningServer, ...]:
    with _LIVE_LOCK:
        return tuple(s for s in _LIVE if s._thread is not None)


def _credential_scope(engine: Any, cred: TokenCredential):
    from ..core.types import Scope, Visibility

    base = engine.host.default_scope()
    pin = cred.scope or {}
    return Scope(
        profile_id=base.profile_id,
        principal_id=cred.principal_id,
        workspace_id=pin.get("workspace_id") or base.workspace_id,
        conversation_id=pin.get("conversation_id") or base.conversation_id,
        visibility=Visibility(pin["visibility"]) if pin.get("visibility") else base.visibility,
    )


def provision_credentials(
    engine: Any,
    credentials: Iterable[TokenCredential],
) -> list[dict[str, Any]]:
    """Issue each credential's declared verbs on its bound scope.

    Returns the provisioning report (one row per credential). Fails
    closed: a store without the grants table refuses provisioning rather
    than pretending authority exists (CAPABILITY_UNAVAILABLE).
    """
    from .. import governance
    from ..core.types_v3 import PrincipalKind
    from ..storage import repos_v3

    report = []
    with engine.store.tx() as conn:
        if not has_table(conn, "grants_v3"):
            raise VerbatimError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "grants_v3 is absent — this store cannot authorize HTTP callers",
            )
        for cred in credentials:
            scope = _credential_scope(engine, cred)
            sid = scope_key(scope)
            ensure_scope(engine.store, conn, scope)
            governance.register_principal(
                conn, kind=PrincipalKind.SERVICE, principal_id=cred.principal_id
            )
            rows = repos_v3.query(
                conn,
                "grants_v3",
                {
                    "scope_id": sid,
                    "principal_id": cred.principal_id,
                    "revoked_us": None,
                },
            )
            from ..core.time import now_us

            now = now_us()
            live = [
                r for r in rows
                if r.get("expires_us") is None or int(r["expires_us"]) > now
            ]
            if live:
                # Attenuation-only: existing authority stands; report what
                # the store actually permits, never auto-widen.
                verbs = set()
                for row in live:
                    verbs |= set(repos_v3.json_field(row, "verbs_json") or ())
                report.append({
                    "principal_id": cred.principal_id,
                    "scope_id": sid,
                    "provisioned": False,
                    "effective_verbs": sorted(verbs & set(cred.verbs)),
                })
                continue
            governance.create_grant(
                conn,
                scope_id=sid,
                principal_id=cred.principal_id,
                verbs=sorted(cred.verbs),
                issuer_id=cred.principal_id,
                delegation_depth=0,
                strict=False,
            )
            report.append({
                "principal_id": cred.principal_id,
                "scope_id": sid,
                "provisioned": True,
                "effective_verbs": sorted(cred.verbs),
            })
    return report


def create_server(
    engine: Any,
    credentials: Iterable[TokenCredential],
    cfg: Optional[HttpConfig] = None,
    *,
    enable_v5_memory: bool = False,
    memory_path: Optional[str] = None,
    memory_user: Optional[str] = None,
    memory_worker: str = "managed",
    memory_config: Any = None,
) -> RunningServer:
    """Provision credentials, bind the loopback socket, return a handle.

    The returned ``RunningServer`` is NOT yet serving — ``start()`` or the
    context manager runs the accept loop. ``enable_v5_memory`` mounts the
    consumer surface (``/v2/memory/*``, docs/v6_contracts.md §4) on the
    same socket via ``api.create_api``; the mounted ``Memory`` is then
    reachable as ``server.httpd.application.memory`` and is the caller's
    to close on shutdown.
    """
    creds = tuple(credentials)
    provision_credentials(engine, creds)
    if enable_v5_memory:
        from .api import create_api

        app = create_api(
            engine,
            creds,
            enable_v5_memory=True,
            memory_path=memory_path,
            memory_user=memory_user,
            memory_worker=memory_worker,
            memory_config=memory_config,
        )
    else:
        app = ServiceApp(engine, creds)
    httpd = HttpServer(cfg or HttpConfig(name="verbatim-service"), app)
    return RunningServer(httpd)


def create_memory_server(
    path: Optional[str],
    *,
    user_id: Optional[str],
    worker: str = "managed",
    tokens: Any = None,
    config: Any = None,
    cfg: Optional[HttpConfig] = None,
) -> RunningServer:
    """Standalone V5 consumer surface (v6 contracts §4): one ``Memory``
    bound to ``user_id`` behind bearer auth on a loopback socket.

    The returned ``RunningServer`` is NOT yet serving — ``start()`` runs
    the accept loop. The bound facade is
    ``server.httpd.application.memory``; close it on shutdown.
    """
    from .memory_api import create_memory_app

    httpd = create_memory_app(
        path, user_id=user_id, worker=worker, tokens=tokens, config=config,
        http_config=cfg,
    )
    return RunningServer(httpd)


def serve(
    target: Any,
    cfg: Optional[HttpConfig] = None,
    *,
    credentials: Optional[Iterable[TokenCredential]] = None,
    token_file: Optional[str] = None,
) -> int:
    """Blocking serve: provision → bind → serve_forever (Ctrl-C to stop).

    ``target`` is an ``Engine`` (or a bare ``Store``, which is wrapped in
    a local-host Engine). Credentials come from ``credentials``,
    ``token_file``, or the ``VERBATIM_SERVICE_*`` environment — in that
    order; all three absent is a CONFIG_INVALID refusal, never an
    unauthenticated listener.
    """
    engine = _as_engine(target)
    creds = _resolve_credentials(credentials, token_file)
    server = create_server(engine, creds, cfg)
    with _LIVE_LOCK:
        _LIVE.append(server)
    try:
        print(f"verbatim service listening on {server.url} "
              f"(loopback, http — TLS unimplemented)")
        server.httpd.serve_forever()
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        server.httpd.server_close()


def _resolve_credentials(
    credentials: Optional[Iterable[TokenCredential]],
    token_file: Optional[str],
) -> tuple[TokenCredential, ...]:
    from .auth import load_token_file

    if credentials is not None:
        creds = tuple(credentials)
    elif token_file:
        creds = load_token_file(token_file)
    else:
        creds = tokens_from_env()
    if not creds:
        raise VerbatimError(
            ErrorCode.CONFIG_INVALID,
            "no bearer tokens provisioned — pass credentials, --token-file, "
            "or set VERBATIM_SERVICE_TOKEN_FILE / VERBATIM_SERVICE_TOKENS",
        )
    return creds


def _as_engine(target: Any) -> Any:
    """Accept an Engine or a bare Store (wrapped in a local host)."""
    from ..api import Engine
    from ..config import VerbatimConfig
    from ..host import LocalHost
    from ..storage.store import Store

    if isinstance(target, Engine):
        return target
    if isinstance(target, Store):
        host = LocalHost()
        return Engine(target, VerbatimConfig(), host)
    raise VerbatimError(
        ErrorCode.VALIDATION,
        f"serve() needs an Engine or Store, got {type(target).__name__}",
    )


__all__ = [
    "create_memory_server",
    "create_server",
    "live_servers",
    "provision_credentials",
    "serve",
]
