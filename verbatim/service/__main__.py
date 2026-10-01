"""``verbatim-service`` console entry — ``python -m verbatim.service``.

Serves the V5 consumer memory API (``/v2/memory/*``, docs/v6_contracts.md
§4) when ``verbatim.service.memory_api`` is present in this build; when it
is not, the launcher falls back to the v4 ``ServiceApp`` (``/v1/*``) over
the same loopback ``HttpServer`` + bearer-token transport. Importing this
module binds nothing — only ``main()`` opens a socket (V4-45.08).

Authentication is always explicit: ``--token`` (repeatable, each bound to
the ``--user`` principal at the full local verb ceiling), ``--token-file``
(operator-provisioned JSON, each entry naming its own principal), or the
``VERBATIM_SERVICE_TOKEN_FILE`` / ``VERBATIM_SERVICE_TOKENS`` environment.
No path here mints an unauthenticated listener.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
from typing import Any, Optional

from ..core.types import ErrorCode, VerbatimError

_DEFAULT_BIND = "127.0.0.1"
_DEFAULT_PORT = 8390


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="verbatim-service",
        description="Serve the Verbatim memory API over loopback HTTP.",
    )
    p.add_argument(
        "--path",
        default=None,
        help="store path (default: the profile store under the data dir)",
    )
    p.add_argument(
        "--user",
        default="local-owner",
        help="principal id --token credentials bind to (default: local-owner)",
    )
    p.add_argument("--bind", default=_DEFAULT_BIND, help="bind host (default: 127.0.0.1)")
    p.add_argument(
        "--port", type=int, default=_DEFAULT_PORT, help="bind port (default: 8390)"
    )
    p.add_argument(
        "--token",
        action="append",
        default=[],
        metavar="SECRET",
        help="bearer token secret; repeatable (bound to --user)",
    )
    p.add_argument(
        "--token-file",
        default=None,
        help="operator-provisioned JSON token file",
    )
    p.add_argument(
        "--worker",
        default="managed",
        choices=("managed", "external"),
        help="memory worker mode for the /v2/memory app (default: managed)",
    )
    p.add_argument(
        "--allow-remote",
        action="store_true",
        help="accept a non-loopback cleartext bind (TLS is unimplemented)",
    )
    return p.parse_args(argv)


def _resolve_credentials(args: argparse.Namespace):
    """Explicit --token / --token-file first, then the provisioned env."""
    from .auth import load_token_file, tokens_from_env
    from ..core.types_v3 import Verb

    if args.token:
        from .auth import TokenCredential

        # An operator-launched local service binds each secret to the --user
        # principal at the full verb ceiling; persisted grants still fence
        # effective authority underneath (single-authority rule).
        return tuple(
            TokenCredential.mint(
                token, principal_id=args.user, verbs=[v.value for v in Verb]
            )
            for token in args.token
        )
    if args.token_file:
        return load_token_file(args.token_file)
    return tokens_from_env()


def _load_memory_app_factory():
    """Locate ``create_memory_app`` without requiring it at import time."""
    if importlib.util.find_spec("verbatim.service.memory_api") is None:
        return None
    from .memory_api import create_memory_app

    return create_memory_app


def _build_memory_httpd(factory: Any, args: argparse.Namespace, creds: Any):
    """Call ``create_memory_app`` honoring the frozen §4 signature.

    The contract pins ``(path, *, user_id, worker="managed", tokens=None)
    -> HttpServer``; bind/port are applied through whichever optional
    parameter the landed factory exposes (``cfg``/``host``+``port``).
    """
    from .httpd import HttpConfig

    cfg = HttpConfig(
        host=args.bind,
        port=args.port,
        allow_remote=args.allow_remote,
        name="verbatim-memory",
    )
    params = set(inspect.signature(factory).parameters)
    kwargs: dict[str, Any] = {"user_id": args.user}
    if "worker" in params:
        kwargs["worker"] = args.worker
    if "tokens" in params:
        kwargs["tokens"] = creds if creds else None
    if "token_file" in params and args.token_file:
        kwargs["token_file"] = args.token_file
    for key in ("cfg", "config", "http_config"):
        if key in params:
            kwargs[key] = cfg
            break
    else:
        if "host" in params or "bind" in params:
            kwargs["host" if "host" in params else "bind"] = cfg.host
        if "port" in params:
            kwargs["port"] = cfg.port
        if "allow_remote" in params:
            kwargs["allow_remote"] = args.allow_remote
    return factory(args.path, **kwargs)


def _serve_httpd(httpd: Any, banner: str) -> int:
    """Blocking serve loop: bound URL to stdout, SIGINT closes cleanly."""
    host, port = httpd.bound_address
    if ":" in host:  # IPv6 loopback
        host = f"[{host}]"
    print(
        f"{banner} on http://{host}:{port} (loopback, http — TLS unimplemented)",
        flush=True,
    )
    try:
        httpd.serve_forever()
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        httpd.server_close()


def _serve_v4_fallback(args: argparse.Namespace, creds: Any) -> int:
    """No memory_api in this build — serve the v4 ``/v1/*`` ServiceApp."""
    import os

    from ..api import open_store
    from ..config import VerbatimConfig
    from ..host import LocalHost
    from .httpd import HttpConfig
    from .server import create_server

    if not creds:
        raise VerbatimError(
            ErrorCode.CONFIG_INVALID,
            "no bearer tokens provisioned — pass --token, --token-file, or set "
            "VERBATIM_SERVICE_TOKEN_FILE / VERBATIM_SERVICE_TOKENS",
        )
    cfg = VerbatimConfig()
    host = LocalHost(
        profile_id="local",
        principal_id=args.user,
        conversation_id="service",
        allow_env_secrets=False,
    )
    data_dir = (
        os.path.dirname(os.path.abspath(args.path)) if args.path else os.getcwd()
    )
    engine = open_store(
        data_dir, cfg, host, create=True, store_path=args.path or None
    )
    http_cfg = HttpConfig(
        host=args.bind,
        port=args.port,
        allow_remote=args.allow_remote,
        name="verbatim-service",
    )
    server = create_server(engine, creds, http_cfg)
    return _serve_httpd(server.httpd, "verbatim service (v4 /v1 API) listening")


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    creds = _resolve_credentials(args)

    factory = _load_memory_app_factory()
    if factory is not None:
        httpd = _build_memory_httpd(factory, args, creds)
        return _serve_httpd(httpd, "verbatim memory service (/v2/memory) listening")

    print(
        "verbatim.service.memory_api is not present in this build — "
        "falling back to the v4 /v1 service API",
        file=sys.stderr,
    )
    return _serve_v4_fallback(args, creds)


if __name__ == "__main__":
    raise SystemExit(main())
