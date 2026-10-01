"""Optional HTTP transport for the v4 engine (SPEC_V4 §44/§45).

Pure-stdlib, loopback-only by default, bearer-token authenticated. This
package is *opt-in*: importing it binds nothing, opens nothing, and
provisions nothing — ``serve()``/``create_server()`` are the only paths
that create a socket (V4-45.08).

Layout: ``httpd`` is the shared transport core (also used by
``verbatim.workbench``); ``auth`` provisions and checks credentials;
``api`` maps the public contract onto the Engine; ``server`` owns the
lifecycle.
"""

from .auth import (
    TOKEN_FILE_ENV,
    TOKENS_ENV,
    TokenAuthenticator,
    TokenCredential,
    load_token_file,
    parse_credentials,
    tokens_from_env,
)
from .httpd import (
    Application,
    HttpConfig,
    HttpError,
    HttpServer,
    Request,
    Response,
    Router,
    RunningServer,
)
from .api import GRANT_CLASSES, ServiceApp
from .server import (
    create_memory_server,
    create_server,
    live_servers,
    provision_credentials,
    serve,
)

__all__ = [
    "Application",
    "GRANT_CLASSES",
    "HttpConfig",
    "HttpError",
    "HttpServer",
    "Request",
    "Response",
    "Router",
    "RunningServer",
    "ServiceApp",
    "TOKEN_FILE_ENV",
    "create_memory_server",
    "TOKENS_ENV",
    "TokenAuthenticator",
    "TokenCredential",
    "create_server",
    "live_servers",
    "load_token_file",
    "parse_credentials",
    "provision_credentials",
    "serve",
    "tokens_from_env",
]
