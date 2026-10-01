"""Native-adapter contract — the §13 depth-2 integration path (SPEC_V3
§13.01–§13.10, §49, §06.10).

A native adapter *translates only*: it maps host lifecycle/tool/memory
events into capture-SDK calls and carries receipts back. Admission,
authorization, screening, and storage logic never live in an adapter
(V3-06.10, V3-49.04) — an adapter can obtain nothing the engine would deny
a direct caller (V3-13.01).

The contract in four methods:

- :meth:`identity` — the stable host id stamped on envelopes
  (``host_id``/``adapter_version``, V3-12.01).
- :meth:`translate_event` — one host event → one canonical event dict
  (the shape :func:`dispatch` consumes). Translation is total or it
  fails typed — partial events are never silently dropped.
- :meth:`capture_auth` — return a live §11.11 capture-authorization id
  for a scope. Only host/operator surfaces issue authorizations; the
  adapter's issuer identity is the host, never the model.
- :meth:`emit` — translate + dispatch one host event through the bound
  :class:`~verbatim.sdk.CaptureClient`; returns the SDK's result.

Adapters publish :meth:`capability_matrix` (V3-13.10) so operators know
which capture kinds, hooks, and guarantees the host supports.

Job draining (§40): capture calls only *enqueue* durable work — nothing on
the SDK/adapter path drains implicitly. The adapter exposes the bound
client's drain as a capability (:meth:`NativeAdapter.drain`, plus the
canonical ``"drain"`` event op) and the concrete adapters drive it on
their declared session-end seam (``job_drain`` in the matrix); a host that
runs its own worker may call :meth:`NativeAdapter.drain` /
:meth:`~verbatim.sdk.CaptureClient.drain_pending` directly instead. The
drain is a synchronous explicit call — no threads or loops live in the
adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional

from ..core.types import ErrorCode, VerbatimError, require_id
from ..sdk.capture import CaptureClient

#: Canonical event ops understood by :func:`dispatch` — the language-neutral
#: vocabulary the §13.02 event schema and GenericEventAdapter share.
EVENT_OPS = frozenset(
    {
        "begin_session",
        "resume_session",
        "authorize",
        "submit_source",
        "capture",
        "step",
        "outcome",
        "end_session",
        "drain",
    }
)


def dispatch(client: CaptureClient, event: dict[str, Any]) -> dict[str, Any]:
    """Execute one canonical event dict against the SDK; returns a result
    snapshot ``{"op", "ok", "result"}`` — typed errors propagate as
    ``VerbatimError`` (hosts translate them, adapters never swallow).
    """
    if not isinstance(event, dict):
        raise VerbatimError(ErrorCode.VALIDATION, "event must be a mapping")
    op = event.get("op")
    if op not in EVENT_OPS:
        raise VerbatimError(ErrorCode.VALIDATION, f"unknown event op {op!r}")

    if op == "begin_session":
        result = client.begin_session(
            require_id(str(event.get("principal_id") or ""), "principal_id"),
            str(event.get("host_id") or ""),
            metadata=event.get("metadata"),
        )
    elif op == "resume_session":
        result = client.resume_session(
            require_id(str(event.get("session_id") or ""), "session_id"),
            principal_id=event.get("principal_id"),
        )
    elif op == "authorize":
        result = client.authorize(
            require_id(str(event.get("scope_id") or ""), "scope_id"),
            require_id(str(event.get("granted_by") or ""), "granted_by"),
            purpose=event.get("purpose"),
            ttl_s=event.get("ttl_s"),
            kinds=event.get("kinds"),
            principal_id=event.get("principal_id"),
        )
    elif op == "submit_source":
        content = event.get("content", event.get("text"))
        result = client.submit_source(
            require_id(str(event.get("session_id") or ""), "session_id"),
            require_id(str(event.get("scope_id") or ""), "scope_id"),
            content,
            declared_type=require_id(
                str(event.get("declared_type") or event.get("kind") or ""),
                "declared_type",
            ),
            title=event.get("title"),
            media_type=event.get("media_type"),
            external_id=event.get("external_id"),
            event_us=event.get("event_us"),
        )
    elif op == "capture":
        result = client.capture_envelope(
            require_id(str(event.get("session_id") or ""), "session_id"),
            event.get("envelope") or {},
        ).to_dict()
    elif op == "step":
        result = client.record_step(
            require_id(str(event.get("session_id") or ""), "session_id"),
            require_id(str(event.get("source_id") or ""), "source_id"),
            event.get("step") or {},
        )
    elif op == "outcome":
        result = client.record_outcome(
            require_id(str(event.get("session_id") or ""), "session_id"),
            require_id(str(event.get("source_id") or ""), "source_id"),
            event.get("outcome") or {},
            receipts=event.get("receipts"),
        )
    elif op == "drain":
        result = client.drain_pending(
            event.get("scope_id"),
            limit=int(event.get("limit") or 64),
            kinds=event.get("kinds"),
            lane=event.get("lane"),
        )
    else:  # end_session
        result = client.end_session(
            require_id(str(event.get("session_id") or ""), "session_id"),
            status=str(event.get("status") or "complete"),
        )
    return {"op": op, "ok": True, "result": result}


class NativeAdapter(ABC):
    """The §13 depth-2 adapter contract: translate, never decide.

    ``client`` binds the adapter to a :class:`CaptureClient`; it may be
    attached later via :meth:`bind` so adapters can be constructed before
    the store is open (host lifecycle ordering).
    """

    def __init__(self, client: Optional[CaptureClient] = None) -> None:
        self._client = client

    # -- contract ------------------------------------------------------------

    @abstractmethod
    def identity(self) -> str:
        """Stable host identifier (``host_id`` on emitted envelopes)."""

    @abstractmethod
    def translate_event(self, raw: Any) -> dict[str, Any]:
        """Map one host event to a canonical event dict (see :func:`dispatch`).

        Returns ``{"op": None, "skip": reason}``-shaped dicts are NOT
        permitted: drop decisions belong to the host's declared capture
        gates, which the adapter checks *before* calling — anything passed
        to translate_event must become a canonical event or raise.
        """

    @abstractmethod
    def capture_auth(self, scope_id: str) -> str:
        """A live §11.11 capture-authorization id for ``scope_id``.

        Host/operator issuance only: the adapter either resolves an
        existing covering authorization or issues one under its host
        issuer identity — it never mints model-visible attestation.
        """

    # -- shared machinery ------------------------------------------------------

    @property
    def client(self) -> CaptureClient:
        if self._client is None:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID, "adapter has no bound CaptureClient"
            )
        return self._client

    def bind(self, client: CaptureClient) -> "NativeAdapter":
        self._client = client
        return self

    def emit(self, raw: Any) -> dict[str, Any]:
        """translate → dispatch one host event; returns the SDK result."""
        return dispatch(self.client, self.translate_event(raw))

    def emit_many(self, raws: Iterable[Any]) -> list[dict[str, Any]]:
        """Ordered emit for a host event batch/stream."""
        return [self.emit(raw) for raw in raws]

    def drain(
        self,
        scope_id: Optional[str] = None,
        *,
        limit: int = 64,
        kinds: Optional[Iterable[Any]] = None,
        lane: Optional[str] = None,
    ) -> int:
        """Drain due durable jobs through the bound client (§40).

        This is the adapter's worker capability — hosts call it on an idle
        or session boundary (or run their own loop). Nothing drains
        implicitly; without it, queued harvest/screen/admit/episode/purge
        work waits forever, privacy-control jobs included. Returns the
        drained count.
        """
        return self.client.drain_pending(
            scope_id, limit=limit, kinds=kinds, lane=lane
        )

    def capability_matrix(self) -> dict[str, Any]:
        """Published capability map (V3-13.10): capture kinds, hooks,
        outcome delivery, compaction archive, downstream processors,
        gateway enforcement, and identity binding."""
        return {
            "adapter": self.identity(),
            "depth": 2,
            "capture_kinds": [],
            "hooks": [],
            "outcome_delivery": False,
            "compaction_archive": False,
            "downstream_processors": ["episode_build"],
            "gateway_enforcement": "engine",
            "identity_binding": "host",
            "job_drain": "explicit",
        }


class GenericEventAdapter(NativeAdapter):
    """Reference adapter: canonical dict events → SDK calls.

    This is both the conformance reference for the §13.02 event schema and
    the fallback for hosts that already speak the canonical vocabulary —
    ``translate_event`` is the identity map (with light alias handling),
    so an event stream of dicts drives a real capture flow end to end:

    .. code-block:: python

        adapter = GenericEventAdapter(client, issuer="ops-1")
        adapter.emit({"op": "begin_session", "principal_id": "agent-1"})
        adapter.emit({"op": "submit_source", "session_id": sid,
                      "scope_id": scope, "content": "note",
                      "declared_type": "agent_note"})
        adapter.emit({"op": "end_session", "session_id": sid})

    ``issuer`` is the host/operator identity used when ``capture_auth``
    must provision an authorization (setup path); event-supplied
    ``granted_by`` values always win.
    """

    _ALIASES = {
        "session_begin": "begin_session",
        "begin": "begin_session",
        "session_end": "end_session",
        "end": "end_session",
        "source": "submit_source",
        "record_step": "step",
        "record_outcome": "outcome",
        "capture_envelope": "capture",
    }

    def __init__(
        self,
        client: Optional[CaptureClient] = None,
        *,
        issuer: str = "generic-adapter",
        auth_kinds: Optional[Iterable[str]] = None,
        auth_ttl_s: Optional[float] = None,
    ) -> None:
        super().__init__(client)
        self._issuer = issuer
        self._auth_kinds = auth_kinds
        self._auth_ttl_s = auth_ttl_s
        self._auths: dict[str, str] = {}

    def identity(self) -> str:
        return "generic"

    def translate_event(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise VerbatimError(
                ErrorCode.VALIDATION, "generic adapter events must be mappings"
            )
        event = dict(raw)
        op = event.get("op") or event.get("type")
        if isinstance(op, str):
            event["op"] = self._ALIASES.get(op, op)
        return event

    def capture_auth(self, scope_id: str) -> str:
        """Resolve or provision the adapter's authorization for a scope.

        Provisioning is idempotent per (scope): a live authorization is
        reused; otherwise one is issued under the adapter's issuer id.
        """
        scope_id = require_id(scope_id, "scope_id")
        existing = self._auths.get(scope_id)
        if existing is not None:
            return existing
        aid = self.client.authorize(
            scope_id,
            self._issuer,
            kinds=self._auth_kinds,
            ttl_s=self._auth_ttl_s,
        )
        self._auths[scope_id] = aid
        return aid

    def capability_matrix(self) -> dict[str, Any]:
        matrix = super().capability_matrix()
        matrix.update(
            {
                "capture_kinds": "all (event-declared)",
                "hooks": ["event_stream"],
                "outcome_delivery": True,
                "identity_binding": "event principal",
            }
        )
        return matrix


__all__ = [
    "EVENT_OPS",
    "GenericEventAdapter",
    "NativeAdapter",
    "dispatch",
]
