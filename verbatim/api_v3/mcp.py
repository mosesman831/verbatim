"""V3 MCP surface — JSON-RPC stdio adapter over ``VerbatimV3`` (§48).

Reuses the v2 transport mechanics (``read_message``/``write_message``
framing, initialize/tools.list/tools.call dispatch) but binds a
``VerbatimV3`` facade and one ``CallerV3`` identity fixed at launch
(V3-48.01). Request arguments can never mint principals, grants,
profiles, purposes, or endpoints (V3-48.05): ``principal_id`` fields are
rejected anywhere they could rebind identity, and ``purpose`` arguments
must name a purpose bound at launch.

Fresh-store onboarding (the review-mandated fix): a store with no owner
and no prior Python ingestion still completes
``v3_authorize_capture`` → ``v3_capture`` → ``v3_recall`` — the launch-
bound principal bootstraps as scope owner, self-issues retention
consent, captures an ``agent_generated`` note, and recalls it.

Tool set (model-visible by registration, never by authority — V3-48.04):
``v3_capture``, ``v3_recall``, ``v3_inspect``, ``v3_authorize_capture``,
``v3_capabilities``, ``v3_outcome``, ``v3_delete``,
``v3_quarantine_review``. Admin/review verbs are still enforced
server-side: under the default model-visible verb set those calls deny
``not_found_or_unauthorized`` (V3-47.09) — they answer only when the
launch bound an operator verb set. The verb set itself is checked at
server construction: a facade bound wider than ``MCP_V3_BOUND_VERBS``
is refused unless ``operator_launch=True`` declares the surface an
operator one.

Two profiles share this module (SPEC_V5 §19.2):

* ``tools(profile="consumer")`` — the DEFAULT — advertises exactly
  ``v5_capture`` and ``v5_recall`` (V5-19.06). ``McpV5Server`` binds a
  ``memory.Memory`` facade and dispatches only those two names: hidden
  operator calls are denied by *absence* from the consumer schema and
  dispatch map, not merely rejected after reaching authority
  (V5-19.07). Arguments can never mint consent, trust classes,
  principals, or speakers (V5-19.08).
* ``tools(profile="operator")`` is the explicit advanced surface — the
  legacy ``v3_*`` toolset above, served by ``McpV3Server`` under its
  existing bound-verb contract (V5-19.09).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Optional, TextIO, Union

from ..core.types import ErrorCode, VerbatimError
from ..governance import CallerV3
from ..mcp import read_message, write_message

__all__ = [
    "MCP_V3_BOUND_VERBS",
    "McpV3Server",
    "McpV5Server",
    "serve_stdio",
    "serve_consumer_stdio",
    "tools",
    "main",
]

#: Verbs a model-visible MCP launch binds on a bootstrapped scope —
#: admin and review stay on operator surfaces (V3-47.09). An operator
#: launch passes a wider ``bound_verbs`` into the facade.
MCP_V3_BOUND_VERBS: frozenset = frozenset(
    {"read", "quote", "derive", "ingest", "share"}
)

_PROTOCOL_VERSION = "2024-11-05"
_SERVER_INFO = {"name": "verbatim-v3", "version": "0.1.0"}

_E_INVALID = -32600
_E_METHOD = -32601
_E_PARAMS = -32602
_E_PARSE = -32700
_E_VERBATIM = -32000

_MAX_ARG_STRING = 65536
_MAX_OUTCOME_REFS = 32


def _bounded_str(value: Any, field: str, maximum: int = _MAX_ARG_STRING) -> str:
    if not isinstance(value, str):
        raise VerbatimError(ErrorCode.VALIDATION, f"{field} must be a string")
    if len(value) > maximum:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"{field} exceeds {maximum} chars"
        )
    return value


def tools(profile: str = "consumer") -> list[dict[str, Any]]:
    """The model-visible tool registry for one MCP profile.

    ``profile="consumer"`` (the default — V5-19.06) advertises exactly
    ``v5_capture`` + ``v5_recall``: every other verb is denied by
    absence from this schema (V5-19.07). ``profile="operator"`` is the
    explicit advanced surface — the legacy v3 toolset (V5-19.09), whose
    names stay stable within a session (V3-48.04).
    """
    if profile == "consumer":
        return _consumer_tools()
    if profile == "operator":
        return _operator_tools()
    raise VerbatimError(
        ErrorCode.VALIDATION, f"unknown MCP tool profile {profile!r}"
    )


def _consumer_tools() -> list[dict[str, Any]]:
    """The V5 consumer surface — capture and recall only (§19.2)."""
    return [
        {
            "name": "v5_capture",
            "description": (
                "Store a memory: bounded text captured under the "
                "launch-bound identity and namespace with honest "
                "attribution (V5-19.06/19.08). Consent, ownership, and "
                "authority come from the launch — never from arguments."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "maxLength": 65536},
                    "metadata": {"type": "object"},
                    "infer": {"type": "boolean", "default": True},
                    "idempotency_key": {
                        "type": "string",
                        "maxLength": 128,
                    },
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
        {
            "name": "v5_recall",
            "description": (
                "Recall memories: budgeted governed search over the "
                "launch-bound namespace, serializing the same "
                "result/readiness contract as Memory.search — pending "
                "and conflict labels preserved (V5-19.09)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "maxLength": 8192},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 64,
                    },
                    "detail": {
                        "type": "string",
                        "enum": ["summary", "evidence"],
                        "default": "summary",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    ]


def _operator_tools() -> list[dict[str, Any]]:
    """The v3 operator tool registry (names stable within a session —
    V3-48.04)."""
    return [
        {
            "name": "v3_authorize_capture",
            "description": (
                "Issue retention consent for agent-submitted capture on a "
                "scope (§11.11). The bound caller is the issuer; tool "
                "permission is not retention consent."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scope_id": {"type": "string"},
                    "ttl_s": {"type": "number", "exclusiveMinimum": 0},
                    "purpose": {"type": "string"},
                },
                "required": ["scope_id"],
            },
        },
        {
            "name": "v3_capture",
            "description": (
                "Explicit agent-submitted capture: bounded text stored "
                "agent-generated, never human testimony (§13). Requires "
                "prior capture authorization."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scope_id": {"type": "string"},
                    "content": {"type": "string", "maxLength": 65536},
                    "declared_type": {"type": "string", "default": "agent_note"},
                    "title": {"type": "string", "maxLength": 512},
                    "external_id": {"type": "string", "maxLength": 512},
                    "purpose": {"type": "string"},
                },
                "required": ["scope_id", "content"],
            },
        },
        {
            "name": "v3_recall",
            "description": (
                "Governed recall: typed packs with influence handles over "
                "authorized evidence (§27/§30)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scope_id": {"type": "string"},
                    "query": {"type": "string", "maxLength": 8192},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 32},
                    "max_bytes": {
                        "type": "integer", "minimum": 512, "maximum": 24000
                    },
                    "purpose": {"type": "string"},
                },
                "required": ["scope_id", "query"],
            },
        },
        {
            "name": "v3_inspect",
            "description": (
                "Evidence-plane lineage for one source: envelopes, "
                "revisions, security labels, quarantine, receipts. Never "
                "payload bytes or vault plaintext."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"source_id": {"type": "string"}},
                "required": ["source_id"],
            },
        },
        {
            "name": "v3_capabilities",
            "description": (
                "Honest capability matrix: lanes, vault, retrieval "
                "pipeline, degradation notes (§62.04)."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "v3_outcome",
            "description": (
                "Submit a checker-attested task outcome (§12.03). An "
                "identified checker_id is required — an agent's bare "
                "'done' is never an outcome."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scope_id": {"type": "string"},
                    "outcome": {
                        "type": "string",
                        "enum": ["success", "failure", "partial", "unknown"],
                    },
                    "checker_id": {"type": "string"},
                    "task_id": {"type": "string"},
                    "trajectory_id": {"type": "string"},
                },
                "required": ["scope_id", "outcome", "checker_id"],
            },
        },
        {
            "name": "v3_delete",
            "description": (
                "Delete a source: immediate suppression plus queued "
                "purge/purge_derived/purge_vault closure (§36). Requires "
                "the admin verb — denied under model-visible bindings "
                "(V3-47.09)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"source_id": {"type": "string"}},
                "required": ["source_id"],
            },
        },
        {
            "name": "v3_quarantine_review",
            "description": (
                "Decide a quarantine hold: release|suppress|purge under "
                "the review verb (§34.03). Origin and findings are never "
                "rewritten."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "security_label_id": {"type": "string"},
                    "decision": {
                        "type": "string",
                        "enum": ["release", "suppress", "purge"],
                    },
                    "reviewer_note": {"type": "string", "maxLength": 2000},
                },
                "required": ["security_label_id", "decision"],
            },
        },
    ]


def _to_jsonable(value: Any) -> Any:
    """Serialize typed v3 snapshots into JSON-safe structures (§47.01)."""
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, frozenset)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return value.hex()
    return value


def _recall_payload(result: Any) -> dict:
    """The wire shape of a ``RecallResultV3`` — packs, handles, provenance.

    Item provenance surfaces through ``security.source_trust`` (e.g.
    ``agent_generated`` for submitted captures) plus the handle's
    ``object_kind``/``object_id``/``revision``.
    """
    data = _to_jsonable(result)
    packs = data.get("packs", [])
    data["text"] = "\n".join(
        item.get("text", "")
        for pack in packs
        for item in pack.get("items", [])
        if item.get("text")
    )
    data["items"] = [
        item for pack in packs for item in pack.get("items", [])
    ]
    return data


class McpV3Server:
    """JSON-RPC dispatcher over one ``VerbatimV3`` + one bound caller.

    The caller is fixed at construction — ``initialize`` params are
    informational only and cannot rebind identity (V3-48.01/48.05).

    Transport-safe verbs (V3-47.09): this surface is model-visible, so a
    facade whose ``bound_verbs`` exceed ``MCP_V3_BOUND_VERBS`` is refused
    at construction — a bare ``VerbatimV3(store)`` defaults to
    ``OWNER_VERBS`` and would silently bind admin/review onto the
    transport. An operator surface passes ``operator_launch=True`` to
    acknowledge the wider binding explicitly.
    """

    def __init__(
        self,
        facade: Any,
        caller: CallerV3,
        *,
        bound_purposes: Optional[frozenset] = None,
        operator_launch: bool = False,
    ) -> None:
        verbs = getattr(facade, "bound_verbs", None)
        if verbs is not None and not operator_launch:
            extra = sorted(set(verbs) - set(MCP_V3_BOUND_VERBS))
            if extra:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "MCP stdio is model-visible: facade bound verbs "
                    f"{extra} exceed the transport-safe set "
                    f"{sorted(MCP_V3_BOUND_VERBS)} — bind "
                    "MCP_V3_BOUND_VERBS at the facade, or pass "
                    "operator_launch=True for an operator surface "
                    "(V3-47.09)",
                )
        self._facade = facade
        self._caller = caller
        # ``None`` means no launch narrowing — purpose arguments are then
        # checked against the live purposes registry (non-retired rows),
        # not against an empty set that would reject every purpose.
        self._bound_purposes = (
            None if bound_purposes is None else frozenset(bound_purposes)
        )
        self.operator_launch = bool(operator_launch)
        self.initialized = False

    #: Which ``tools()`` profile this server lists and dispatches, and
    #: the identity it reports at ``initialize``. The v3 surface is the
    #: operator/advanced profile (V5-19.09); the consumer subclass
    #: narrows it to ``"consumer"``.
    _tool_profile = "operator"
    _SERVER_INFO = _SERVER_INFO

    # -- argument hygiene ----------------------------------------------------

    _REBINDING_KEYS = frozenset(
        {
            "principal_id",
            "actor",
            "actor_id",
            "caller",
            "caller_id",
            "granted_by",
            "issuer_id",
            "issuer",
            "grants",
            "verbs",
            "profile",
            "profile_id",
            "endpoint",
            "session_id",
        }
    )

    def _check_args(self, args: dict[str, Any], allowed: frozenset) -> None:
        """Reject identity-minting keys and unbound purposes (V3-48.05)."""
        bad = set(args) & (self._REBINDING_KEYS - allowed)
        if bad:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"arguments cannot rebind identity or authority: {sorted(bad)}",
            )
        purpose = args.get("purpose")
        if purpose is not None:
            _bounded_str(purpose, "purpose", 256)
            if not self._purpose_allowed(purpose):
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "purpose not bound at launch",
                )

    def _schema_arg_names(self, tool_name: str) -> Optional[frozenset]:
        """Consumer profile: an argument absent from the advertised
        schema is denied — never silently ignored (V5-19.07/19.08). The
        operator profile keeps the legacy hygiene only (rebinding keys);
        its schemas predate argument allowlisting (``evidence_refs`` on
        ``v3_outcome`` is dispatched but undeclared)."""
        if self._tool_profile != "consumer":
            return None
        for spec in _consumer_tools():
            if spec["name"] == tool_name:
                return frozenset(
                    spec.get("inputSchema", {}).get("properties", {})
                )
        return frozenset()

    def _purpose_allowed(self, purpose: str) -> bool:
        """An explicit bound set narrows the registry; without one the
        live (non-retired) purposes registry is the universe — arbitrary
        strings still never pass (V3-48.05)."""
        if self._bound_purposes is not None:
            return purpose in self._bound_purposes
        store = getattr(self._facade, "store", None)
        if store is None:
            return False
        try:
            with store.read() as conn:
                row = conn.execute(
                    "SELECT 1 FROM purposes WHERE purpose = ? AND retired = 0",
                    (purpose,),
                ).fetchone()
        except Exception:
            return False
        return row is not None

    def _scope_arg(self, args: dict[str, Any]) -> str:
        scope_id = args.get("scope_id")
        if not isinstance(scope_id, str) or not scope_id:
            raise VerbatimError(ErrorCode.VALIDATION, "scope_id is required")
        return _bounded_str(scope_id, "scope_id", 512)

    # -- tool handlers --------------------------------------------------------

    def _tool_authorize_capture(self, args: dict[str, Any]) -> dict:
        scope_id = self._scope_arg(args)
        ttl = args.get("ttl_s")
        aid = self._facade.issue_capture_authorization(
            self._caller.principal_id,
            scope_id,
            granted_by=self._caller.principal_id,
            purpose=args.get("purpose"),
            ttl_s=ttl,
        )
        return {"authorization_id": aid, "scope_id": scope_id}

    def _tool_capture(self, args: dict[str, Any]) -> dict:
        scope_id = self._scope_arg(args)
        content = args.get("content")
        if content is None:
            raise VerbatimError(ErrorCode.VALIDATION, "content is required")
        source_id = self._facade.capture_submitted(
            self._caller.principal_id,
            scope_id,
            _bounded_str(content, "content"),
            declared_type=_bounded_str(
                args.get("declared_type") or "agent_note", "declared_type", 64
            ),
            title=(
                _bounded_str(args["title"], "title", 512)
                if args.get("title") is not None
                else None
            ),
            purpose=args.get("purpose"),
            session_id=self._caller.session_id,
            external_id=(
                _bounded_str(args["external_id"], "external_id", 512)
                if args.get("external_id") is not None
                else None
            ),
        )
        return {"source_id": source_id, "scope_id": scope_id}

    @staticmethod
    def _bounded_int(value: Any, field: str, lo: int, hi: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise VerbatimError(
                ErrorCode.VALIDATION, f"{field} must be an integer"
            )
        return max(lo, min(value, hi))

    def _tool_recall(self, args: dict[str, Any]) -> dict:
        scope_id = self._scope_arg(args)
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise VerbatimError(ErrorCode.VALIDATION, "recall requires a query")
        _bounded_str(query, "query", 8192)
        budget: dict[str, Any] = {}
        if args.get("limit") is not None:
            budget["max_items"] = self._bounded_int(
                args["limit"], "limit", 1, 32
            )
        if args.get("max_bytes") is not None:
            budget["max_bytes"] = self._bounded_int(
                args["max_bytes"], "max_bytes", 512, 24000
            )
        result = self._facade.recall(
            scope_id,
            query,
            principal_id=self._caller.principal_id,
            purpose=args.get("purpose"),
            budget=budget or None,
            session_id=self._caller.session_id,
        )
        return _recall_payload(result)

    def _tool_inspect(self, args: dict[str, Any]) -> dict:
        source_id = args.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise VerbatimError(ErrorCode.VALIDATION, "source_id is required")
        return self._facade.inspect_evidence(
            _bounded_str(source_id, "source_id", 512),
            principal_id=self._caller.principal_id,
        )

    def _tool_capabilities(self, args: dict[str, Any]) -> dict:
        return self._facade.capabilities()

    def _tool_outcome(self, args: dict[str, Any]) -> dict:
        scope_id = self._scope_arg(args)
        outcome = args.get("outcome")
        checker_id = args.get("checker_id")
        if not isinstance(outcome, str) or not outcome:
            raise VerbatimError(ErrorCode.VALIDATION, "outcome is required")
        if not isinstance(checker_id, str) or not checker_id:
            raise VerbatimError(
                ErrorCode.VALIDATION, "identified checker_id is required"
            )
        refs = args.get("evidence_refs") or ()
        if not isinstance(refs, (list, tuple)) or len(refs) > _MAX_OUTCOME_REFS:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"evidence_refs bounded at {_MAX_OUTCOME_REFS}",
            )
        return self._facade.submit_outcome(
            scope_id,
            principal_id=self._caller.principal_id,
            outcome=outcome,
            checker_id=_bounded_str(checker_id, "checker_id", 512),
            task_id=(
                _bounded_str(args["task_id"], "task_id", 512)
                if args.get("task_id") is not None
                else ""
            ),
            session_id=self._caller.session_id,
            trajectory_id=(
                _bounded_str(args["trajectory_id"], "trajectory_id", 512)
                if args.get("trajectory_id") is not None
                else None
            ),
            purpose=args.get("purpose"),
            evidence_refs=[str(r)[:512] for r in refs],
        )

    def _tool_delete(self, args: dict[str, Any]) -> dict:
        source_id = args.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise VerbatimError(ErrorCode.VALIDATION, "source_id is required")
        return self._facade.delete_source(
            _bounded_str(source_id, "source_id", 512),
            principal_id=self._caller.principal_id,
        )

    def _tool_quarantine_review(self, args: dict[str, Any]) -> dict:
        label_id = args.get("security_label_id")
        decision = args.get("decision")
        if not isinstance(label_id, str) or not label_id:
            raise VerbatimError(
                ErrorCode.VALIDATION, "security_label_id is required"
            )
        if not isinstance(decision, str) or not decision:
            raise VerbatimError(ErrorCode.VALIDATION, "decision is required")
        return self._facade.quarantine_review(
            _bounded_str(label_id, "security_label_id", 512),
            principal_id=self._caller.principal_id,
            decision=decision,
            reviewer_note=(
                _bounded_str(args["reviewer_note"], "reviewer_note", 2000)
                if args.get("reviewer_note") is not None
                else None
            ),
        )

    _TOOL_HANDLERS = {
        "v3_authorize_capture": _tool_authorize_capture,
        "v3_capture": _tool_capture,
        "v3_recall": _tool_recall,
        "v3_inspect": _tool_inspect,
        "v3_capabilities": _tool_capabilities,
        "v3_outcome": _tool_outcome,
        "v3_delete": _tool_delete,
        "v3_quarantine_review": _tool_quarantine_review,
    }

    # -- JSON-RPC plumbing (mirrors verbatim.mcp) ------------------------------

    @staticmethod
    def _result(msg_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _error(msg_id: Any, code: int, message: str, data: Any = None) -> dict:
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": msg_id, "error": err}

    def _verbatim_error(self, msg_id: Any, exc: VerbatimError) -> dict:
        """Privacy-safe error envelope: the error class only (§46)."""
        return self._error(
            msg_id,
            _E_VERBATIM,
            exc.code.value,
            data={"verbatim_error": exc.code.value, "message": exc.message},
        )

    def handle(self, msg: Any) -> Optional[Any]:
        """Dispatch one decoded JSON message; None for notifications."""
        if isinstance(msg, list):
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
                    "serverInfo": dict(self._SERVER_INFO),
                },
            )
        if method.startswith("notifications/"):
            return None
        if notification:
            return None
        if method == "ping":
            return self._result(msg_id, {})
        if method == "tools/list":
            return self._result(
                msg_id, {"tools": tools(profile=self._tool_profile)}
            )
        if method == "tools/call":
            return self._tools_call(msg_id, params)
        return self._error(msg_id, _E_METHOD, "method_not_found")

    def _tools_call(self, msg_id: Any, params: Any) -> dict:
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
            self._check_args(args, frozenset())
            allowed = self._schema_arg_names(str(name))
            if allowed is not None:
                extra = sorted(set(args) - allowed)
                if extra:
                    raise VerbatimError(
                        ErrorCode.VALIDATION,
                        f"{name}: arguments absent from the consumer "
                        f"schema are denied: {extra}",
                    )
            data = handler(self, args)
        except VerbatimError as exc:
            return self._verbatim_error(msg_id, exc)
        return self._result(
            msg_id,
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(data, ensure_ascii=False, default=str),
                    }
                ],
                "structuredContent": data,
                "isError": False,
            },
        )


# ----------------------------------------------------------------------
# consumer surface (SPEC_V5 §19.2)
# ----------------------------------------------------------------------


class McpV5Server(McpV3Server):
    """The consumer MCP profile: exactly ``v5_capture`` + ``v5_recall``.

    Binds a ``memory.Memory``-shaped facade instead of ``VerbatimV3`` —
    caller identity, consent, and namespace are whatever the facade was
    launched with, and tool arguments can never mint or rebind them
    (V5-19.08): the shared rebinding-key check still applies, and any
    argument absent from the advertised schema is denied outright.

    Hidden operator calls are denied by *absence* (V5-19.07): the
    dispatch map holds only the two consumer names, so
    ``v3_*``/inspect/delete/review/authorize calls answer
    ``unknown tool`` rather than reaching authority. No admin verb is
    reachable through this surface at all, which is why the
    construction-time bound-verb check does not apply — the transport
    bound is the two-name registry itself.
    """

    _tool_profile = "consumer"
    _SERVER_INFO = {"name": "verbatim-v5", "version": "0.1.0"}

    def __init__(
        self,
        memory: Any,
        *,
        bound_purposes: Optional[frozenset] = None,
    ) -> None:
        add = getattr(memory, "add", None)
        search = getattr(memory, "search", None)
        if not (callable(add) and callable(search)):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "consumer MCP binds a Memory-shaped facade "
                "(add/search) — pass verbatim.Memory or an equivalent",
            )
        # The facade's session-bound caller is the only identity; this
        # server stores no caller of its own. ``_facade`` exists solely
        # for the shared purpose check — a ``Memory`` exposes no
        # ``store``/``bound_verbs``, so purpose arguments deny and the
        # operator verb-boundary check is inapplicable by construction.
        self._facade = memory
        self._memory = memory
        self._caller = None
        self._bound_purposes = (
            None if bound_purposes is None else frozenset(bound_purposes)
        )
        self.operator_launch = False
        self.initialized = False

    # -- tool handlers --------------------------------------------------------

    def _tool_v5_capture(self, args: dict[str, Any]) -> dict:
        text = args.get("text")
        if not isinstance(text, str) or not text:
            raise VerbatimError(ErrorCode.VALIDATION, "text is required")
        _bounded_str(text, "text")
        meta = args.get("metadata")
        if meta is not None and not isinstance(meta, dict):
            raise VerbatimError(
                ErrorCode.VALIDATION, "metadata must be an object"
            )
        infer = args.get("infer", True)
        if not isinstance(infer, bool):
            raise VerbatimError(
                ErrorCode.VALIDATION, "infer must be a boolean"
            )
        idem = args.get("idempotency_key")
        result = self._memory.add(
            text,
            metadata=meta,
            infer=infer,
            idempotency_key=(
                _bounded_str(idem, "idempotency_key", 128)
                if idem is not None
                else None
            ),
        )
        return _to_jsonable(result)

    def _tool_v5_recall(self, args: dict[str, Any]) -> dict:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise VerbatimError(
                ErrorCode.VALIDATION, "v5_recall requires a query"
            )
        _bounded_str(query, "query", 8192)
        limit = (
            self._bounded_int(args["limit"], "limit", 1, 64)
            if args.get("limit") is not None
            else 8
        )
        detail = args.get("detail") or "summary"
        if detail not in ("summary", "evidence"):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "detail must be 'summary' or 'evidence'",
            )
        result = self._memory.search(query, limit=limit)
        data = _to_jsonable(result)
        if detail == "summary":
            data = _summary_recall_payload(data)
        return data

    _TOOL_HANDLERS = {
        "v5_capture": _tool_v5_capture,
        "v5_recall": _tool_v5_recall,
    }


def _summary_recall_payload(data: dict) -> dict:
    """``detail="summary"`` wire shape: per-hit scoring internals and
    coverage counters stay on the evidence detail — the recall contract
    (status, refs, quotes, lifecycle, warnings) is never slimmed."""
    slim = dict(data)
    slim.pop("coverage", None)
    items = []
    for hit in data.get("items") or []:
        if isinstance(hit, dict):
            h = dict(hit)
            h.pop("score_detail", None)
            items.append(h)
        else:
            items.append(hit)
    slim["items"] = items
    return slim


# ----------------------------------------------------------------------
# stdio loop
# ----------------------------------------------------------------------

_Stream = Union[TextIO, "Any"]


def _serve_loop(server: McpV3Server, instream: Any, outstream: Any) -> int:
    """Read/dispatch/write until EOF — shared by both profiles."""
    while True:
        try:
            msg = read_message(instream)
        except VerbatimError:
            write_message(
                outstream, McpV3Server._error(None, _E_PARSE, "parse_error")
            )
            continue
        except Exception:
            return 0
        if msg is None:
            return 0
        try:
            resp = server.handle(msg)
        except Exception:
            resp = McpV3Server._error(None, _E_VERBATIM, "internal_error")
        if resp is not None:
            write_message(outstream, resp)


def serve_stdio(
    facade: Any,
    caller: CallerV3,
    *,
    instream: Optional[Any] = None,
    outstream: Optional[Any] = None,
    bound_purposes: Optional[frozenset] = None,
    operator_launch: bool = False,
) -> int:
    """Run the read/dispatch/write loop until EOF (§48.01).

    Model-visible launches bind ``MCP_V3_BOUND_VERBS`` on the facade;
    ``operator_launch=True`` is the explicit operator-surface
    acknowledgment that lets a wider ``bound_verbs`` through the
    construction-time transport check (V3-47.09)."""
    instream = instream if instream is not None else sys.stdin
    outstream = outstream if outstream is not None else sys.stdout
    server = McpV3Server(
        facade,
        caller,
        bound_purposes=bound_purposes,
        operator_launch=operator_launch,
    )
    return _serve_loop(server, instream, outstream)


def serve_consumer_stdio(
    memory: Any,
    *,
    instream: Optional[Any] = None,
    outstream: Optional[Any] = None,
    bound_purposes: Optional[frozenset] = None,
) -> int:
    """Serve the V5 consumer profile (``v5_capture``/``v5_recall`` only)
    over stdio until EOF. ``memory`` is a launched ``Memory`` facade —
    identity and consent are whatever it was constructed with."""
    instream = instream if instream is not None else sys.stdin
    outstream = outstream if outstream is not None else sys.stdout
    server = McpV5Server(memory, bound_purposes=bound_purposes)
    return _serve_loop(server, instream, outstream)


def main(argv: Optional[list[str]] = None) -> int:
    """Console entry (``verbatim-mcp``): serve the V5 consumer profile.

    Launches a ``Memory`` facade — identity is the launch-bound host
    principal, ``--user`` is only an alias label — then runs the
    ``v5_capture``/``v5_recall`` stdio loop until EOF (SPEC_V5 §19.2).
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="verbatim-mcp",
        description="Serve the Verbatim consumer memory MCP tools over stdio.",
    )
    p.add_argument(
        "--path",
        default=None,
        help="store path (default: the profile store under the data dir)",
    )
    p.add_argument(
        "--user",
        dest="user_id",
        default=None,
        help="alias label for the launch-bound principal (never authority)",
    )
    p.add_argument(
        "--worker",
        default="managed",
        choices=("managed", "external"),
        help="memory worker mode (default: managed)",
    )
    args = p.parse_args(argv)

    from ..memory.facade import Memory

    memory = Memory(args.path, user_id=args.user_id, worker=args.worker)
    try:
        return serve_consumer_stdio(memory)
    finally:
        memory.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
