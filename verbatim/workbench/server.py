"""Workbench lifecycle — same provisioning and loopback transport as the
service API (V4-45.08, §49.09). Importing binds nothing."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from ..service.auth import TokenCredential, tokens_from_env
from ..service.httpd import HttpConfig, HttpServer, RunningServer
from ..service.server import (
    _LIVE,
    _LIVE_LOCK,
    _as_engine,
    _resolve_credentials,
    provision_credentials,
)
from .app import WorkbenchApp


def create_server(
    engine: Any,
    credentials: Iterable[TokenCredential],
    cfg: Optional[HttpConfig] = None,
) -> RunningServer:
    """Provision + bind; ``start()`` runs the accept loop."""
    creds = tuple(credentials)
    provision_credentials(engine, creds)
    app = WorkbenchApp(engine, creds)
    httpd = HttpServer(cfg or HttpConfig(name="verbatim-workbench"), app)
    return RunningServer(httpd)


def serve(
    target: Any,
    cfg: Optional[HttpConfig] = None,
    *,
    credentials: Optional[Iterable[TokenCredential]] = None,
    token_file: Optional[str] = None,
) -> int:
    """Blocking workbench serve (Ctrl-C to stop). Same credential sources
    as ``service.serve``; a tokenless workbench refuses to start."""
    engine = _as_engine(target)
    creds = _resolve_credentials(credentials, token_file)
    server = create_server(engine, creds, cfg)
    with _LIVE_LOCK:
        _LIVE.append(server)
    try:
        print(
            f"verbatim workbench listening on {server.url} "
            "(loopback, http — TLS unimplemented)"
        )
        server.httpd.serve_forever()
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        server.httpd.server_close()


__all__ = ["create_server", "serve"]
