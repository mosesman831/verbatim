"""Verbatim V3 API surface (SPEC_V3 §47–§48).

``facade.VerbatimV3`` is the host-neutral entry point: explicit
agent-submitted capture, governed recall, evidence inspection,
quarantine review, capabilities, outcomes/trajectories, and deletion
closure — all over one open ``Store``. ``mcp.McpV3Server`` exposes the
same surface over dependency-free JSON-RPC stdio with launch-bound
identity (§48.01).
"""

from .facade import AGENT_SUBMITTED_KINDS, OWNER_VERBS, VerbatimV3
from .mcp import (
    MCP_V3_BOUND_VERBS,
    McpV3Server,
    McpV5Server,
    serve_consumer_stdio,
    serve_stdio,
    tools,
)

__all__ = [
    "VerbatimV3",
    "OWNER_VERBS",
    "AGENT_SUBMITTED_KINDS",
    "McpV3Server",
    "McpV5Server",
    "MCP_V3_BOUND_VERBS",
    "serve_stdio",
    "serve_consumer_stdio",
    "tools",
]
