"""Operator workbench (SPEC_V4 §49) — authenticated loopback UI.

Shares the stdlib HTTP core and token provisioning of
``verbatim.service``; every page authorizes the operator grant class
through ``governance.authorize`` and reads/mutates only through the
Engine's verified paths. Opt-in: importing binds nothing.
"""

from .app import WorkbenchApp
from .server import create_server, serve

__all__ = ["WorkbenchApp", "create_server", "serve"]
