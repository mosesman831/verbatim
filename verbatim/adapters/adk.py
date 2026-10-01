"""Google ADK depth-2 adapter (SPEC_V3 §13.03, §15, §49) — MemoryService-
style operations translated onto the capture SDK.

The adapter speaks the four memory-service operations an ADK agent host
drives:

- ``add_session_to_memory(session)`` — translate every ADK session event
  into its §12 envelope (user turns → ``user_message``/``principal_direct``,
  model turns → ``assistant_message``/``agent_generated``, function calls →
  ``tool_call``, function responses → ``tool_result``/``host_observed``),
  then close the session trajectory (§13.03: the host attests authorship —
  the model's own text is never attributed to the user, §13.11).
- ``add_events_to_memory(events, ...)`` — incremental capture of raw event
  dicts/objects under an open session (no close; ADK may stream events).
- ``add_memory(note, ...)`` — host/agent-submitted note → the SDK's
  ``submit_source`` path with ``agent_note`` provenance (never human
  testimony).
- ``search_memory(app_name, user_id, query)`` — recall is NOT a capture
  concern: the adapter delegates to an injectable ``searcher``; absent one
  it raises ``CAPABILITY_UNAVAILABLE`` (v3 retrieval is a separate engine
  surface — adapters never reimplement it).

Live-host boundary (§15 depth 3): ADK objects (``google.adk.sessions.Session``,
``types.Content``/``Part``) are accessed duck-typed — this module imports
no ADK package, so the adapter is importable without the host installed.
The injectable ``transport`` is the *only* piece a live host must supply:
it resolves a session id to an event sequence (``fetch_session``) for
push-style invocations. Everything else works against in-memory dicts.

Authorization: ``setup_scope``/``capture_auth`` are the host/operator
provisioning surface (§11.11) — they issue records through SDK/governance
APIs and never decide access; ``provision=False`` defers to pre-seeded
authority exactly like the Hermes adapter.

Job draining (§40): session-close boundaries drain inline —
``add_session_to_memory`` and ``end_adk_session`` call
``client.drain_pending`` after ``end_session`` so queued harvest/admit,
screens, ``episode_build``, and privacy-control jobs execute without a
separate worker. A host running its own worker may instead call
:meth:`NativeAdapter.drain` directly and ignore the ``drained`` count in
the returned summary. The drain is synchronous and explicit; no threads
live in the adapter.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Protocol

from ..core.identity import scope_key
from ..core.time import now_us
from ..core.types import ErrorCode, Scope, VerbatimError, Visibility, require_id
from ..core.types_v3 import EnvelopeKind, TrustClass
from ..governance import create_grant, grants_for
from ..sdk.capture import CaptureClient
from ..sdk.envelope import EnvelopeBuilder
from ..storage import repos_v3
from .base import NativeAdapter

_ISSUER = "adk-adapter"
_ADAPTER_VERSION = "adk-v3/1.0"
_FALLBACK_PRINCIPAL = "adk-principal"
_ATTESTED_KINDS = (
    EnvelopeKind.USER_MESSAGE,
    EnvelopeKind.ASSISTANT_MESSAGE,
    EnvelopeKind.TOOL_CALL,
    EnvelopeKind.TOOL_RESULT,
    EnvelopeKind.TEST_RESULT,
    EnvelopeKind.VERIFICATION,
    EnvelopeKind.SYSTEM_EVENT,
    EnvelopeKind.AGENT_NOTE,
)


class AdkTransport(Protocol):
    """Injectable live-host surface (§15 depth 3).

    A real deployment binds the ADK ``SessionService``; tests bind a fake
    returning plain dicts. ``fetch_session`` returns the session's event
    sequence (dicts or ADK ``Event`` objects) for push-style calls that
    arrive with only ids.
    """

    def fetch_session(
        self, app_name: str, user_id: str, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Session mapping ``{"id", "app_name", "user_id", "events": [...]}``
        or ``None`` when the host has no such session."""


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attr-or-key access — ADK objects and plain dicts share one path."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def adk_scope_id(app_name: str, user_id: str, session_id: str = "") -> str:
    """The v3 scope partition for an ADK (app, user[, session]) triple —
    the same canonical ``scope_key`` digest v2 partitions use."""
    return scope_key(
        Scope(
            profile_id=require_id(app_name or "adk-app", "app_name"),
            principal_id=require_id(user_id or _FALLBACK_PRINCIPAL, "user_id"),
            workspace_id=None,
            conversation_id=session_id or "nosession",
            visibility=Visibility.CONVERSATION,
        )
    )


def _part_text(part: Any) -> Optional[str]:
    text = _get(part, "text")
    return text if isinstance(text, str) and text else None


def _part_kind(part: Any) -> str:
    """Classify one content part: text | function_call | function_response."""
    if _get(part, "function_call") is not None or _get(part, "functionCall") is not None:
        return "function_call"
    if (
        _get(part, "function_response") is not None
        or _get(part, "functionResponse") is not None
    ):
        return "function_response"
    return "text"


def _event_parts(event: Any) -> List[Any]:
    content = _get(event, "content")
    if content is None:
        return []
    parts = _get(content, "parts")
    if parts is None:
        text = _get(content, "text") or _get(content, "data")
        return [{"text": text}] if text else []
    return list(parts)


def _event_text(event: Any) -> str:
    """Concatenated text of an ADK event's content parts."""
    return "".join(
        t for t in (_part_text(p) for p in _event_parts(event)) if t
    )


def _call_of(part: Any) -> Dict[str, Any]:
    call = _get(part, "function_call") or _get(part, "functionCall") or {}
    if not isinstance(call, dict):
        call = {"name": _get(call, "name"), "args": _get(call, "args")}
    return call


def _response_of(part: Any) -> Dict[str, Any]:
    resp = _get(part, "function_response") or _get(part, "functionResponse") or {}
    if not isinstance(resp, dict):
        resp = {"name": _get(resp, "name"), "response": _get(resp, "response")}
    return resp


def adk_event_to_envelope(
    event: Any,
    *,
    scope_id: str,
    session_id: str,
    host_id: str = "adk",
    agent_principal: str = "adk:agent",
    user_principal: str = _FALLBACK_PRINCIPAL,
    adapter_version: str = _ADAPTER_VERSION,
) -> List[EnvelopeBuilder]:
    """Translate one ADK event → zero or more envelope builders (§13.03).

    Provenance rules: ``author == "user"`` attests ``principal_direct``
    testimony; anything else is agent-authored (``agent_generated`` —
    never upgraded); function-call parts are the agent's actions;
    function-response parts are host-observed results. Events with no
    capturable content produce no envelopes — total translation, never a
    fabricated payload.
    """
    builders: List[EnvelopeBuilder] = []
    author = str(_get(event, "author") or "agent")
    event_id = str(_get(event, "id") or "") or None
    ts = _get(event, "timestamp")
    event_us = int(ts * 1_000_000) if isinstance(ts, (int, float)) and ts else 0
    is_user = author == "user"
    actor = user_principal if is_user else agent_principal

    for idx, part in enumerate(_event_parts(event)):
        pkind = _part_kind(part)
        ext = f"adk:{session_id}:{event_id or 'evt'}:{idx}"
        b = (
            EnvelopeBuilder()
            .scope(scope_id)
            .host(host_id)
            .session(session_id)
            .adapter_version(adapter_version)
            .external_id(ext)
        )
        if event_us:
            b.event_us(event_us)
        if pkind == "function_call":
            call = _call_of(part)
            b.kind(EnvelopeKind.TOOL_CALL).actor(agent_principal).trust(
                TrustClass.AGENT_GENERATED
            ).content(
                f"{call.get('name') or 'tool'}({call.get('args')})"
            ).metadata(tool_name=str(call.get("name") or ""))
        elif pkind == "function_response":
            resp = _response_of(part)
            b.kind(EnvelopeKind.TOOL_RESULT).actor(agent_principal).trust(
                TrustClass.HOST_OBSERVED
            ).content(str(resp.get("response"))).metadata(
                tool_name=str(resp.get("name") or "")
            )
        else:
            text = _part_text(part)
            if text is None:
                continue
            b.content(text)
            if is_user:
                b.kind(EnvelopeKind.USER_MESSAGE).actor(actor).trust(
                    TrustClass.PRINCIPAL_DIRECT
                ).perspective({"asserter": actor, "observer": host_id})
            else:
                b.kind(EnvelopeKind.ASSISTANT_MESSAGE).actor(actor).trust(
                    TrustClass.AGENT_GENERATED
                ).perspective({"asserter": actor, "observer": host_id})
        builders.append(b)

    # Branching / state-delta markers capture as host-observed events.
    actions = _get(event, "actions")
    if actions is not None and _get(actions, "state_delta"):
        b = (
            EnvelopeBuilder()
            .kind(EnvelopeKind.SYSTEM_EVENT)
            .scope(scope_id)
            .actor(agent_principal)
            .content(str(_get(actions, "state_delta")))
            .trust(TrustClass.HOST_OBSERVED)
            .host(host_id)
            .session(session_id)
            .adapter_version(adapter_version)
            .external_id(f"adk:{session_id}:{event_id or 'evt'}:delta")
        )
        if event_us:
            b.event_us(event_us)
        builders.append(b)
    return builders


class AdkMemoryAdapter(NativeAdapter):
    """ADK ``BaseMemoryService``-shaped adapter over the capture SDK.

    ``client``: the bound :class:`CaptureClient` (injectable for tests).
    ``transport``: optional live-host session resolver — the ONLY piece a
    real ADK deployment must implement (``fetch_session``); absent it,
    ``add_session_to_memory`` accepts session objects/dicts directly.
    ``searcher``: optional ``(app_name, user_id, query) -> result`` recall
    delegate for ``search_memory`` (capture adapters never reimplement
    retrieval).
    """

    def __init__(
        self,
        client: Optional[CaptureClient] = None,
        *,
        transport: Optional[AdkTransport] = None,
        searcher: Optional[Any] = None,
        issuer: str = _ISSUER,
        provision: bool = True,
        user_id: Optional[str] = None,
    ) -> None:
        super().__init__(client)
        self._transport = transport
        self._searcher = searcher
        self._issuer = issuer
        self._provision = provision
        self._user_id = user_id
        self._sessions: dict[str, Dict[str, Any]] = {}
        self._auths: dict[str, str] = {}

    # ------------------------------------------------------------------
    # NativeAdapter contract
    # ------------------------------------------------------------------

    def identity(self) -> str:
        return "adk"

    def translate_event(self, raw: Any) -> Dict[str, Any]:
        """ADK event/object or canonical dict → canonical event dict."""
        if isinstance(raw, dict) and (raw.get("op") or raw.get("type")):
            event = dict(raw)
            event["op"] = event.get("op") or event.get("type")
            return event
        # An ADK event becomes a capture op; the envelope list is produced
        # at dispatch time by ``capture_adk_event``.
        return {"op": "adk_event", "event": raw}

    def emit(self, raw: Any) -> Dict[str, Any]:
        event = self.translate_event(raw)
        if event.get("op") == "adk_event":
            captured = self.capture_adk_event(
                event["event"],
                session_id=str(_get(event["event"], "session_id") or ""),
                app_name=str(_get(event["event"], "app_name") or "adk-app"),
                user_id=str(_get(event["event"], "user_id") or self._principal()),
            )
            return {"op": "adk_event", "ok": True, "result": captured}
        from .base import dispatch

        return dispatch(self.client, event)

    def capture_auth(self, scope_id: str, principal_id: Optional[str] = None) -> str:
        """Live §11.11 authorization for (scope, principal) — the ADK
        ``user_id`` is the session principal, so provisioning keys on
        both."""
        scope_id = require_id(scope_id, "scope_id")
        principal = principal_id or self._principal()
        key = f"{scope_id}{principal}"
        existing = self._auths.get(key)
        if existing is not None:
            return existing
        aid = self.client.authorize(
            scope_id,
            self._issuer,
            kinds=_ATTESTED_KINDS,
            principal_id=principal,
            retention_policy="task",
        )
        self._auths[key] = aid
        return aid

    def capability_matrix(self) -> Dict[str, Any]:
        matrix = super().capability_matrix()
        matrix.update(
            {
                "capture_kinds": sorted(k.value for k in _ATTESTED_KINDS),
                "hooks": [
                    "add_session_to_memory",
                    "add_events_to_memory",
                    "add_memory",
                    "search_memory",
                ],
                "outcome_delivery": True,
                "compaction_archive": False,
                "identity_binding": "adk (app_name, user_id, session_id)",
                "job_drain": "session close",
            }
        )
        return matrix

    # ------------------------------------------------------------------
    # host setup surface (§11.11 issuance — never an access decision)
    # ------------------------------------------------------------------

    def setup_scope(
        self, scope_id: str, principal_id: Optional[str] = None
    ) -> Dict[str, str]:
        """Provision the ingest grant + capture authorization for a
        (scope, principal) — idempotent; skip entirely with
        ``provision=False`` when the deployment pre-seeds authority."""
        scope_id = require_id(scope_id, "scope_id")
        principal = principal_id or self._principal()
        grant_id = None
        with self.client.store.tx() as conn:
            live = grants_for(conn, scope_id, principal)
            has_ingest = any(
                "ingest" in set(repos_v3.json_field(g, "verbs_json") or ())
                for g in live
            )
            if not has_ingest:
                grant_id = create_grant(
                    conn,
                    scope_id=scope_id,
                    principal_id=principal,
                    verbs=["ingest"],
                    issuer_id=self._issuer,
                )
        return {
            "scope_id": scope_id,
            "principal_id": principal,
            "grant_id": grant_id,
            "authorization_id": self.capture_auth(scope_id, principal),
        }

    # ------------------------------------------------------------------
    # MemoryService-style operations
    # ------------------------------------------------------------------

    def add_session_to_memory(self, session: Any) -> Dict[str, Any]:
        """Capture a finished ADK session: every event → envelopes → one
        ordered trajectory, then ``end_session`` (the session's boundary
        is declared — ADK hands us the whole object, §20.01).

        ``session`` may be a live ADK ``Session``, a plain dict
        (``{"id", "app_name", "user_id", "events": [...]}``), or a session
        *id* resolved through the injected transport.

        The session boundary is also the drain seam (§40): after the
        trajectory closes, queued work (harvest→admit, screens,
        ``episode_build``, privacy-control jobs) drains inline so a
        pure-adapter deployment never strands durable obligations.
        """
        if isinstance(session, str):
            session = self._resolve(session)
        app_name, user_id, session_id, events = self._session_parts(session)
        scope = adk_scope_id(app_name, user_id, session_id)
        sdk_session = self._open_sdk_session(scope, session_id, app_name, user_id)
        captured: List[str] = []
        for event in events:
            captured.extend(
                self._capture_one(sdk_session, scope, session_id, event, user_id)
            )
        summary = self.client.end_session(sdk_session, status="complete")
        try:
            summary["drained"] = self.client.drain_pending(limit=64)
        except VerbatimError as exc:
            summary["drain_error"] = exc.code.value
        summary["captured_sources"] = len(set(captured))
        summary["scope_id"] = scope
        self._sessions.pop(session_id, None)
        return summary

    def add_events_to_memory(
        self,
        events: Iterable[Any],
        *,
        app_name: str = "adk-app",
        user_id: Optional[str] = None,
        session_id: str = "",
    ) -> Dict[str, Any]:
        """Incremental capture of streamed events under an OPEN session —
        no trajectory close (the host decides the boundary via
        ``end_adk_session`` or ``add_session_to_memory``)."""
        user = user_id or self._principal()
        scope = adk_scope_id(app_name, user, session_id)
        sdk_session = self._open_sdk_session(scope, session_id, app_name, user)
        captured: List[str] = []
        for event in events:
            captured.extend(
                self._capture_one(sdk_session, scope, session_id, event, user)
            )
        return {
            "session_id": sdk_session,
            "scope_id": scope,
            "captured_sources": len(set(captured)),
            "closed": False,
        }

    def add_memory(
        self,
        note: str,
        *,
        app_name: str = "adk-app",
        user_id: Optional[str] = None,
        session_id: str = "",
        scope_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> str:
        """Host/agent-submitted note → ``submit_source`` with ``agent_note``
        provenance — recorded exactly, attributed honestly (§13.11)."""
        user = user_id or self._principal()
        scope = scope_id or adk_scope_id(app_name, user, session_id)
        sdk_session = self._open_sdk_session(scope, session_id, app_name, user)
        return self.client.submit_source(
            sdk_session, scope, note, declared_type="agent_note", title=title
        )

    def search_memory(
        self, *, app_name: str, user_id: str, query: str, **kwargs: Any
    ) -> Any:
        """Recall delegates to the injected ``searcher`` — adapters never
        reimplement retrieval (§49.04); without one the capability is
        reported unavailable, never a fake result."""
        if self._searcher is None:
            raise VerbatimError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "search_memory requires a bound searcher — the v3 retrieval "
                "surface is separate from this capture adapter",
            )
        return self._searcher(app_name, user_id, query, **kwargs)

    def end_adk_session(self, session_id: str, status: str = "complete") -> Dict[str, Any]:
        """Close a session opened by ``add_events_to_memory`` — the
        declared boundary also drains queued work (§40), same as
        ``add_session_to_memory``."""
        entry = self._sessions.pop(session_id, None)
        if entry is None:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "unknown adk capture session"
            )
        summary = self.client.end_session(entry["sdk_session"], status=status)
        try:
            summary["drained"] = self.client.drain_pending(limit=64)
        except VerbatimError as exc:
            summary["drain_error"] = exc.code.value
        return summary

    # ------------------------------------------------------------------
    # translation internals
    # ------------------------------------------------------------------

    def capture_adk_event(
        self,
        event: Any,
        *,
        session_id: str,
        app_name: str = "adk-app",
        user_id: Optional[str] = None,
    ) -> List[str]:
        """One ADK event → captured source ids (open session)."""
        user = user_id or self._principal()
        scope = adk_scope_id(app_name, user, session_id)
        sdk_session = self._open_sdk_session(scope, session_id, app_name, user)
        return self._capture_one(sdk_session, scope, session_id, event, user)

    def _capture_one(
        self,
        sdk_session: str,
        scope: str,
        session_id: str,
        event: Any,
        user_id: str,
    ) -> List[str]:
        builders = adk_event_to_envelope(
            event,
            scope_id=scope,
            session_id=session_id or "adk",
            agent_principal=self._agent_principal(),
            user_principal=user_id,
        )
        out: List[str] = []
        last_sid: Optional[str] = None
        for b in builders:
            sid = self.client.capture_envelope(sdk_session, b.build()).source_id
            out.append(sid)
            last_sid = sid
        if last_sid is not None:
            self.client.record_step(sdk_session, last_sid, {})
            entry = self._sessions.get(session_id)
            if entry is not None:
                entry["last_source_id"] = last_sid
        return out

    def _session_parts(self, session: Any) -> tuple:
        """Normalize dict/ADK Session → (app_name, user_id, session_id, events)."""
        if session is None:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "session not found"
            )
        app_name = str(_get(session, "app_name") or _get(session, "appName") or "adk-app")
        user_id = str(_get(session, "user_id") or _get(session, "userId") or self._principal())
        session_id = str(_get(session, "id") or _get(session, "session_id") or "adk-session")
        events = list(_get(session, "events") or ())
        return app_name, user_id, session_id, events

    def _resolve(self, session_id: str) -> Any:
        """Push-style lookup through the injected transport (the only
        live-host dependency this adapter has)."""
        if self._transport is None:
            raise VerbatimError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "session-id resolution requires an injected AdkTransport",
            )
        found = self._transport.fetch_session("", "", session_id)
        if found is None:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "session not found"
            )
        return found

    def _open_sdk_session(
        self, scope: str, session_id: str, app_name: str, user_id: str
    ) -> str:
        entry = self._sessions.get(session_id)
        if entry is not None and entry["scope_id"] == scope:
            return entry["sdk_session"]
        if self._provision:
            self.setup_scope(scope, principal_id=user_id)
        sdk_session = self.client.begin_session(
            user_id,
            "adk",
            metadata={
                "scope_id": scope,
                "boundary_rule": "session",
                "app_name": app_name,
                "adk_session": session_id,
            },
        )
        self._sessions[session_id] = {
            "sdk_session": sdk_session,
            "scope_id": scope,
            "last_source_id": None,
        }
        return sdk_session

    def _principal(self) -> str:
        return self._user_id or _FALLBACK_PRINCIPAL

    def _agent_principal(self) -> str:
        return "adk:agent"


__all__ = [
    "AdkMemoryAdapter",
    "AdkTransport",
    "adk_event_to_envelope",
    "adk_scope_id",
]
