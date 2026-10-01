"""Stdio MCP adapter — dependency-free JSON-RPC over stdin/stdout (SPEC_V2 §44).

Host-neutral like the rest of the engine: this file translates JSON-RPC
frames into ``Engine`` calls under ONE startup-bound ``CallerContext``.
Transport identity is asserted once at startup — request JSON can narrow
the addressed scope but can never mint identity, grants, or a different
principal (V2-09, MCP authorization guidance).

Framing: accepts both ``Content-Length``-framed JSON (LSP style) and
newline-delimited JSON-RPC. Responses are newline-delimited — the MCP
stdio standard. Importing this module performs no I/O and starts nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, BinaryIO, Optional, TextIO, Union

from .core.types import (
    CallerContext,
    ErrorCode,
    FeedbackKind,
    GrantKind,
    RecallMode,
    RecallRequest,
    Scope,
    VerbatimError,
    Visibility,
)

#: Grants issued to an MCP agent session: evidence read + grounded writes.
#: Resolution, suppression, purge, export, sharing, and operator authority
#: are never available over this transport (V2-09.10).
MCP_GRANTS = frozenset(
    {GrantKind.READ_EVIDENCE, GrantKind.PROPOSE, GrantKind.INGEST}
)

_PROTOCOL_VERSION = "2024-11-05"
_SERVER_INFO = {"name": "verbatim", "version": "0.1.0"}

# JSON-RPC error codes (application range -32099..-32000 for server errors).
_E_INVALID = -32600
_E_METHOD = -32601
_E_PARAMS = -32602
_E_PARSE = -32700
_E_VERBATIM = -32000  # VerbatimError envelope; code carried in data


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "verbatim_recall",
            "description": (
                "Search verbatim evidence memory: exact quotations with "
                "speaker, time and lifecycle qualifiers."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "maxLength": 8192},
                    "mode": {
                        "type": "string",
                        "enum": [m.value for m in RecallMode],
                        "default": "current",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 32},
                    "conversation_id": {"type": "string"},
                    "workspace_id": {"type": "string"},
                    "visibility": {
                        "type": "string",
                        "enum": [v.value for v in Visibility],
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "verbatim_remember",
            "description": (
                "Grounded remember: persist a verbatim quotation by byte "
                "range of an accepted source. Never stores generated text."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 1},
                    "predicate": {"type": "string"},
                    "revision": {"type": "integer", "minimum": 1},
                },
                "required": ["source_id", "start", "end"],
            },
        },
        {
            "name": "verbatim_evidence",
            "description": (
                "Inspect one claim's evidence lineage: spans, revisions, "
                "validity intervals, and relations."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"claim_id": {"type": "string"}},
                "required": ["claim_id"],
            },
        },
        {
            "name": "verbatim_feedback",
            "description": (
                "Record usefulness feedback on a claim. Never a truth "
                "judgment and never a deletion request."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [k.value for k in FeedbackKind],
                    },
                },
                "required": ["claim_id", "kind"],
            },
        },
    ]


class McpServer:
    """JSON-RPC dispatcher bound to one engine + one caller identity.

    The caller is fixed at construction — ``initialize`` params are
    informational only and cannot rebind identity (V2-09: untrusted request
    data may narrow but never widen authority).
    """

    def __init__(self, engine: Any, caller: CallerContext) -> None:
        self._engine = engine
        self._caller = caller
        self.initialized = False

    # -- scope --------------------------------------------------------------

    def _request_scope(self, args: dict[str, Any]) -> Scope:
        """The caller's home partition optionally narrowed by scope fields.

        A supplied conversation/workspace that is not the caller's own is
        left for the engine to deny — this method never fabricates access.
        """
        base = self._caller.scope()
        vis = args.get("visibility")
        return Scope(
            profile_id=base.profile_id,
            principal_id=base.principal_id,
            workspace_id=args.get("workspace_id") or base.workspace_id,
            conversation_id=args.get("conversation_id") or base.conversation_id,
            visibility=Visibility(vis) if vis else base.visibility,
        )

    # -- tool handlers --------------------------------------------------------

    def _tool_recall(self, args: dict[str, Any]) -> dict[str, Any]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise VerbatimError(ErrorCode.VALIDATION, "recall requires a query")
        req = RecallRequest(
            query=query,
            scope=self._request_scope(args),
            mode=RecallMode(args.get("mode") or "current"),
            limit=min(int(args.get("limit") or 8), 32),
            max_bytes=24000,
        )
        res = self._engine.recall(req, caller=self._caller)
        items = [
            {
                "claim_id": i.claim_id,
                "text": i.text,
                "speaker": i.speaker_id,
                "lifecycle": i.lifecycle.value,
                "valid": i.valid_label,
                "historical": i.historical,
                "disputed": i.disputed,
                "reasons": list(i.reasons),
                "source": {
                    "id": i.span.source_id,
                    "revision": i.span.revision,
                    "start": i.span.start_byte,
                    "end": i.span.end_byte,
                },
            }
            for i in res.items
        ]
        return {
            "items": items,
            "text": "\n".join(i["text"] for i in items),
            "omitted": res.omitted,
            "warnings": list(res.warnings),
            "metadata": {
                "projection_generation": res.projection_generation,
                "count": len(items),
            },
        }

    def _tool_remember(self, args: dict[str, Any]) -> dict[str, Any]:
        source_id = args.get("source_id")
        start = args.get("start", args.get("start_byte"))
        end = args.get("end", args.get("end_byte"))
        if not source_id or start is None or end is None:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "remember requires source_id, start, end",
            )
        claim_id = self._engine.remember(
            str(source_id),
            int(start),
            int(end),
            self._request_scope(args),
            predicate_suggestion=args.get("predicate"),
            caller=self._caller,
            revision=int(args.get("revision") or 1),
        )
        return {"claim_id": claim_id}

    def _tool_evidence(self, args: dict[str, Any]) -> dict[str, Any]:
        claim_id = args.get("claim_id")
        if not claim_id:
            raise VerbatimError(ErrorCode.VALIDATION, "evidence requires claim_id")
        return self._engine.inspect(
            str(claim_id), self._request_scope(args), caller=self._caller
        )

    def _tool_feedback(self, args: dict[str, Any]) -> dict[str, Any]:
        claim_id = args.get("claim_id")
        kind = args.get("kind")
        if not claim_id or not kind:
            raise VerbatimError(
                ErrorCode.VALIDATION, "feedback requires claim_id and kind"
            )
        try:
            fk = FeedbackKind(str(kind))
        except ValueError:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"unknown feedback kind {kind!r}"
            ) from None
        self._engine.feedback(
            str(claim_id), fk, self._request_scope(args), caller=self._caller
        )
        return {"recorded": True}

    _TOOL_HANDLERS = {
        "verbatim_recall": _tool_recall,
        "verbatim_remember": _tool_remember,
        "verbatim_evidence": _tool_evidence,
        "verbatim_feedback": _tool_feedback,
    }

    # -- JSON-RPC plumbing ----------------------------------------------------

    @staticmethod
    def _result(msg_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _error(
        msg_id: Any, code: int, message: str, data: Any = None
    ) -> dict[str, Any]:
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": msg_id, "error": err}

    def _verbatim_error(self, msg_id: Any, exc: VerbatimError) -> dict[str, Any]:
        """Privacy-safe error envelope: the error class only — no stack
        traces, no scope internals. Unknown and forbidden stay merged in
        ``not_found_or_forbidden`` (V2-43)."""
        return self._error(
            msg_id,
            _E_VERBATIM,
            exc.code.value,
            data={"verbatim_error": exc.code.value, "message": exc.message},
        )

    def handle(self, msg: Any) -> Optional[Any]:
        """Dispatch one decoded JSON message; None for notifications."""
        if isinstance(msg, list):  # JSON-RPC batch
            if not msg:
                return self._error(None, _E_INVALID, "invalid_request")
            out = [self.handle(m) for m in msg]
            return [r for r in out if r is not None] or None
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return self._error(None, _E_INVALID, "invalid_request")
        msg_id = msg.get("id")
        method = msg.get("method")
        if not isinstance(method, str):
            return self._error(msg_id, _E_INVALID, "invalid_request")
        params = msg.get("params") or {}
        notification = "id" not in msg

        if method == "initialize":
            self.initialized = True
            pv = params.get("protocolVersion") or _PROTOCOL_VERSION
            return self._result(
                msg_id,
                {
                    "protocolVersion": pv,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": dict(_SERVER_INFO),
                },
            )
        if method == "notifications/initialized" or method.startswith("notifications/"):
            return None
        if notification:
            # Unknown notifications are ignored, never answered (JSON-RPC).
            return None
        if method == "ping":
            return self._result(msg_id, {})
        if method == "tools/list":
            return self._result(msg_id, {"tools": _tools()})
        if method == "tools/call":
            return self._tools_call(msg_id, params)
        return self._error(msg_id, _E_METHOD, "method_not_found")

    def _tools_call(self, msg_id: Any, params: Any) -> dict[str, Any]:
        if not isinstance(params, dict):
            return self._error(msg_id, _E_PARAMS, "invalid_params")
        name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            return self._error(msg_id, _E_PARAMS, "invalid_params")
        handler = self._TOOL_HANDLERS.get(str(name))
        if handler is None:
            return self._error(msg_id, _E_PARAMS, f"unknown tool {name!r}")
        try:
            data = handler(self, args)
        except VerbatimError as exc:
            return self._verbatim_error(msg_id, exc)
        return self._result(
            msg_id,
            {
                "content": [
                    {"type": "text", "text": json.dumps(data, ensure_ascii=False, default=str)}
                ],
                "structuredContent": data,
                "isError": False,
            },
        )


# ----------------------------------------------------------------------
# stdio framing
# ----------------------------------------------------------------------

_Stream = Union[TextIO, BinaryIO]


def _readline(stream: _Stream) -> str:
    raw = stream.readline()
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    return raw or ""


def _read(stream: _Stream, n: int) -> str:
    raw = stream.read(n)
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    return raw or ""


def read_message(stream: _Stream) -> Optional[Any]:
    """One JSON-RPC message: Content-Length framed or newline-delimited.

    Returns ``None`` at EOF. Raises ``VerbatimError(VALIDATION)`` on a
    malformed frame so the serve loop can answer -32700 and continue.
    """
    while True:
        line = _readline(stream)
        if line == "":
            return None
        if line.strip():
            break
    if line.lower().startswith("content-length:"):
        try:
            length = int(line.split(":", 1)[1].strip())
        except ValueError:
            raise VerbatimError(ErrorCode.VALIDATION, "bad Content-Length") from None
        if length <= 0 or length > (8 << 20):
            raise VerbatimError(ErrorCode.VALIDATION, "frame length out of bounds")
        # Consume remaining headers through the blank line.
        while True:
            h = _readline(stream)
            if h == "" or not h.strip():
                break
        body = _read(stream, length)
        if len(body.encode("utf-8")) < length and len(body) < length:
            raise VerbatimError(ErrorCode.VALIDATION, "truncated frame")
        try:
            return json.loads(body)
        except (ValueError, UnicodeDecodeError):
            raise VerbatimError(ErrorCode.VALIDATION, "invalid JSON") from None
    try:
        return json.loads(line)
    except ValueError:
        raise VerbatimError(ErrorCode.VALIDATION, "invalid JSON") from None


def write_message(stream: _Stream, message: Any) -> None:
    """Newline-delimited JSON — the MCP stdio framing."""
    text = json.dumps(message, ensure_ascii=False, default=str) + "\n"
    try:
        stream.write(text)  # type: ignore[arg-type]
    except TypeError:
        stream.write(text.encode("utf-8"))  # type: ignore[union-attr]
    try:
        stream.flush()
    except Exception:
        pass


def serve_stdio(
    engine: Any,
    caller: CallerContext,
    *,
    instream: Optional[_Stream] = None,
    outstream: Optional[_Stream] = None,
) -> int:
    """Run the read/dispatch/write loop until EOF. Returns a process code."""
    instream = instream if instream is not None else sys.stdin
    outstream = outstream if outstream is not None else sys.stdout
    server = McpServer(engine, caller)
    while True:
        try:
            msg = read_message(instream)
        except VerbatimError:
            write_message(outstream, McpServer._error(None, _E_PARSE, "parse_error"))
            continue
        except Exception:
            return 0  # unreadable transport — exit quietly, never a traceback
        if msg is None:
            return 0
        try:
            resp = server.handle(msg)
        except Exception:
            # A bug in the adapter must never become a stack trace on the
            # wire or a leaked detail — answer a generic server error.
            resp = McpServer._error(None, _E_VERBATIM, "internal_error")
        if resp is not None:
            write_message(outstream, resp)


def main(argv: Optional[list[str]] = None) -> int:
    """Standalone entry: ``python -m verbatim.mcp --profile X --principal Y``."""
    p = argparse.ArgumentParser(prog="verbatim-mcp")
    p.add_argument("--profile", default="local")
    p.add_argument("--principal", default="local-owner")
    p.add_argument("--conversation")
    p.add_argument("--workspace")
    p.add_argument("--data-dir")
    p.add_argument("--config")
    args = p.parse_args(argv)

    from .api import open_store
    from .cli import _cfg
    from .host import LocalHost

    cfg = _cfg(args)
    host = LocalHost(
        profile_id=args.profile,
        principal_id=args.principal,
        conversation_id=args.conversation or "mcp",
        workspace_id=args.workspace,
        allow_env_secrets=False,
    )
    data_dir = args.data_dir or os.path.join(os.getcwd(), cfg.data_dir)
    engine = open_store(data_dir, cfg, host, create=True)
    try:
        caller = CallerContext(
            profile_id=host.profile_id(),
            principal_id=host.default_scope().principal_id or args.principal,
            agent_id="mcp",
            session_id=args.conversation or "mcp-stdio",
            workspace_id=args.workspace,
            conversation_id=args.conversation or "mcp",
            grants=MCP_GRANTS,
        )
        return serve_stdio(engine, caller)
    finally:
        engine.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
