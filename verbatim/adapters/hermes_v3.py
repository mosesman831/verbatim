"""Hermes depth-2 adapter (SPEC_V3 §13.03, §49) — provider hooks → SDK.

This is the v3 sibling of ``verbatim/provider.py``: same Hermes hook
surface, same host/scope binding, but the capture path delegates to
:class:`~verbatim.sdk.CaptureClient` instead of the v2 ``Engine``. The
adapter *translates only* (V3-49.04): hook events become
``capture_envelope``/``record_step``/``record_outcome`` calls; admission,
authorization, screening, and storage all stay inside the SDK.

Mapping (§12 kind table, §13.03 attestation):

- ``sync_turn`` → ``user_message`` (``principal_direct`` — the host
  attests actual authorship; a bot-flagged author degrades to
  ``unknown``, never human testimony) + ``assistant_message``
  (``agent_generated``) captured as one trajectory step: the assistant
  reply is the action, the user turn its observation.
- ``on_tool_result`` → ``tool_result`` (``host_observed``) + step.
- ``on_task_end`` → checker-attested outcome on the session trajectory;
  a checker-less ``outcome`` dict persists as evidence but can never
  upgrade the episode (§12.03).
- ``on_session_end`` → ``end_session`` (``complete``) — the trajectory
  closes and ``episode_build`` enqueues in the SDK — then the session-end
  idle seam drains queued work (mirrors ``provider.on_session_end``'s
  ``run_pending``): harvest/screen/admit, episode_build, and
  privacy-control jobs all execute inline so a pure-hook deployment never
  strands durable obligations (§40). A drain failure is never silent
  (F4-21): ``summary["drain_error"]`` + ``adapter.drain_error`` expose it,
  a durable ``adapter_hook_failed`` event is journaled, and the host log
  is notified — while obligations stay queued and the host stays usable.
- ``on_session_switch`` → close the open capture session, rebind the
  host scope, and lazily open the next session on first capture.

Store selection goes through ``storage.resolver.resolve_store_path``
(V4-07.10, F4-18): the adapter adopts the profile's resolved store —
``{profile_id}.db`` canonical, a discovered ``v3.db`` adopted — and never
mints its own filename or silently picks between a conflicted pair
(``STORE_CONFLICT`` until the operator decides via ``store_path`` /
``prefer_store`` kwargs).

Host-side gates mirror provider.py: writes run only in the ``primary``
agent context, and ``capture.enabled`` / ``user_messages`` /
``assistant_context`` / ``tool_outputs`` toggles are honored *before*
translation (§12 table). Hook failures never break a turn — policy
denials decline silently per SPEC §43; real failures are recorded on
``last_hook_error``/``drain_error``, journaled, and logged
(V4-14.11); the canonical :meth:`emit` event surface still propagates
typed errors.

Authorization: ``capture_auth``/``setup_scope`` are the host/operator
provisioning surface (§11.11). They *issue* records by delegating to the
SDK/governance issuance APIs — they never *decide* access; that check
stays in ``grants.authorize``/``require_capture_authorization`` on every
SDK call. ``initialize(..., provision=False)`` disables self-
provisioning for deployments that pre-seed grants and authorizations.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

from ..config import VerbatimConfig
from ..core.identity import scope_key
from ..core.time import now_us
from ..core.types import ErrorCode, Scope, VerbatimError, require_id
from ..core.types_v3 import EnvelopeKind, TrustClass
from ..governance import create_grant, grants_for
from ..provider import _HermesHost, _load_hermes_config
from ..sdk.capture import CaptureClient
from ..sdk.envelope import EnvelopeBuilder
from ..storage import repos_v3
from ..storage.store import Store
from .base import NativeAdapter

_FALLBACK_PRINCIPAL = "hermes-principal"
_ISSUER = "hermes-adapter"
_ADAPTER_VERSION = "hermes-v3/1.0"
#: Kinds Hermes can legitimately attest (§13.03); the authorization the
#: adapter provisions covers exactly these — never ``agent_note`` (that
#: kind is the agent's own submit path, not a host observation).
_ATTESTED_KINDS = (
    EnvelopeKind.USER_MESSAGE,
    EnvelopeKind.ASSISTANT_MESSAGE,
    EnvelopeKind.TOOL_RESULT,
    EnvelopeKind.FILE_DIFF,
    EnvelopeKind.TEST_RESULT,
    EnvelopeKind.VERIFICATION,
    EnvelopeKind.SYSTEM_EVENT,
)

#: Policy denials a capture hook declines silently per SPEC §43 — any
#: other failure is recorded on ``last_hook_error``/``drain_error``,
#: journaled, and logged (V4-14.11); it is never a silent success.
_DECLINE_CODES = frozenset(
    {
        ErrorCode.CAPTURE_DISABLED,
        ErrorCode.CONSENT_REQUIRED,
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
        ErrorCode.NOT_FOUND_OR_FORBIDDEN,
        ErrorCode.RETENTION_DENIED,
        ErrorCode.QUARANTINED,
    }
)


def _err_text(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    if code is not None:
        return f"{getattr(code, 'value', code)}: {exc}"
    return f"{type(exc).__name__}: {exc}"


class HermesV3Adapter(NativeAdapter):
    """Thin Hermes edge adapter: hooks → ``CaptureClient`` (§13.03).

    ``client`` may be injected (tests, embedding hosts, the registered
    provider — which binds it to the engine's store); otherwise
    :meth:`initialize` creates one over the profile's resolved store
    (``storage.resolver``, V4-07.10) — the same profile-store policy every
    surface shares, never an adapter-private filename.
    """

    def __init__(self, client: Optional[CaptureClient] = None) -> None:
        super().__init__(client)
        self._owns_client = client is None
        self._host: Optional[_HermesHost] = None
        self._cfg: Optional[VerbatimConfig] = None
        self._session_id = ""
        self._agent_context = "primary"
        self._writes_enabled = False
        self._provision = True
        self._scope_id: Optional[str] = None
        self._sdk_session: Optional[str] = None
        self._auths: dict[str, str] = {}
        self._last_source_id: Optional[str] = None
        # F4-21/V4-14.11 failure surfaces — visible, never a silent pass.
        self._drain_error: Optional[Dict[str, Any]] = None
        self._last_hook_error: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # NativeAdapter contract
    # ------------------------------------------------------------------

    def identity(self) -> str:
        return "hermes"

    def translate_event(self, raw: Any) -> Dict[str, Any]:
        """Hermes hook-shaped or canonical dict → canonical event dict.

        Hook events (``{"hook": "sync_turn", ...}``) translate to the
        canonical op vocabulary; canonical dicts pass through with alias
        handling — the same total-or-typed-error contract as
        :func:`~verbatim.adapters.base.dispatch`.
        """
        if not isinstance(raw, dict):
            raise VerbatimError(
                ErrorCode.VALIDATION, "hermes adapter events must be mappings"
            )
        if "hook" not in raw:
            event = dict(raw)
            op = event.get("op") or event.get("type")
            if isinstance(op, str):
                event["op"] = op
            return event
        hook = raw["hook"]
        if hook == "sync_turn":
            return {
                "op": "_hook",
                "hook": "sync_turn",
                "user_content": raw.get("user_content"),
                "assistant_content": raw.get("assistant_content"),
                "session_id": raw.get("session_id", ""),
                "turn_author": raw.get("turn_author"),
            }
        if hook == "tool_result":
            return {
                "op": "_hook",
                "hook": "tool_result",
                "tool_name": raw.get("tool_name"),
                "content": raw.get("content"),
                "session_id": raw.get("session_id", ""),
            }
        if hook == "task_end":
            return {
                "op": "_hook",
                "hook": "task_end",
                "outcome": raw.get("outcome"),
                "checker": raw.get("checker"),
            }
        if hook in ("session_end", "session_switch"):
            return {"op": "_hook", "hook": hook, **{k: v for k, v in raw.items() if k != "hook"}}
        raise VerbatimError(ErrorCode.VALIDATION, f"unknown hermes hook {hook!r}")

    def emit(self, raw: Any) -> Dict[str, Any]:
        """Hook events run through the hook methods (which capture +
        step + outcome); canonical ops dispatch to the SDK."""
        event = self.translate_event(raw)
        if event.get("op") == "_hook":
            result = self._run_hook(event)
            return {"op": event["hook"], "ok": True, "result": result}
        from .base import dispatch

        return dispatch(self.client, event)

    def capture_auth(self, scope_id: str) -> str:
        """Live §11.11 authorization for the attested kinds — reused when
        present, issued under the adapter issuer id otherwise."""
        scope_id = require_id(scope_id, "scope_id")
        existing = self._auths.get(scope_id)
        if existing is not None:
            return existing
        aid = self.client.authorize(
            scope_id,
            _ISSUER,
            kinds=_ATTESTED_KINDS,
            principal_id=self._principal(),
            retention_policy="task",
        )
        self._auths[scope_id] = aid
        return aid

    def capability_matrix(self) -> Dict[str, Any]:
        matrix = super().capability_matrix()
        matrix.update(
            {
                "capture_kinds": sorted(k.value for k in _ATTESTED_KINDS),
                "hooks": [
                    "initialize",
                    "sync_turn",
                    "on_tool_result",
                    "on_task_end",
                    "on_session_end",
                    "on_session_switch",
                    "on_memory_write",
                    "shutdown",
                ],
                "outcome_delivery": True,
                "compaction_archive": False,
                "identity_binding": "hermes session principal",
                "job_drain": "on_session_end",
            }
        )
        return matrix

    # ------------------------------------------------------------------
    # lifecycle (mirrors VerbatimMemoryProvider)
    # ------------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Bind to a Hermes profile/session; opens the capture session.

        kwargs mirror provider.py: ``hermes_home``, ``user_id`` /
        ``agent_identity``, ``agent_workspace``, ``agent_context``, plus
        ``provision=False`` to disable adapter-side grant/auth issuance.
        ``store_path`` / ``prefer_store`` are the operator's
        store-resolution decisions when the profile directory holds both
        store conventions (``storage.resolver``, V4-07.10).
        """
        hermes_home = kwargs.get("hermes_home") or os.path.expanduser("~/.hermes")
        if self._host is not None and os.path.realpath(hermes_home) != self._host._home:
            self.shutdown()
        self._session_id = session_id
        self._agent_context = kwargs.get("agent_context") or "primary"
        self._cfg = _load_hermes_config(hermes_home).validate()
        self._provision = bool(kwargs.get("provision", True))
        self._host = _HermesHost(
            hermes_home=hermes_home,
            session_id=session_id,
            principal_id=kwargs.get("user_id") or kwargs.get("agent_identity"),
            workspace_id=kwargs.get("agent_workspace"),
        )
        self._writes_enabled = self._agent_context == "primary"
        self._drain_error = None
        self._last_hook_error = None
        if self._client is None:
            from ..storage.resolver import require_store_path, resolve_store_path

            resolution = resolve_store_path(
                os.path.join(self._host._home, "verbatim"),
                profile_id=self._host.profile_id(),
                explicit_path=kwargs.get("store_path"),
                create=True,
                prefer=kwargs.get("prefer_store"),
            )
            path = require_store_path(resolution)
            self._client = CaptureClient(
                Store.open(path) if os.path.exists(path) else Store.create(path),
                config=self._cfg,
                host_id="hermes",
                adapter_version=_ADAPTER_VERSION,
            )
            self._owns_client = True
        self._scope_id = scope_key(self._scope(session_id))
        if self._writes_enabled:
            # The scopes row is the partition registry, not an authority
            # record: register the decomposed tuple even when the operator
            # pre-provisions grants (provision=False) so captured evidence
            # stays legible to the shared read model.
            self._register_scope()
            # Provisioning and the capture session are write-side setup —
            # a read-only context (subagent/cron/flush) must not mint
            # durable grant/authorization/session records (SPEC §33).
            if self._provision:
                self.setup_scope(self._scope_id)
            self._sdk_session = self.client.begin_session(
                self._principal(),
                "hermes",
                metadata={
                    "scope_id": self._scope_id,
                    "boundary_rule": "session",
                    "hermes_session": session_id,
                },
            )
        self._last_source_id = None

    def shutdown(self) -> None:
        if self._client is not None and self._owns_client:
            try:
                self._client.close()
            finally:
                self._client = None
        self._sdk_session = None
        self._last_source_id = None

    # ------------------------------------------------------------------
    # host setup surface (§11.11 issuance — never an access decision)
    # ------------------------------------------------------------------

    def setup_scope(self, scope_id: Optional[str] = None) -> Dict[str, str]:
        """Provision the durable records v3 capture needs (idempotent):

        1. an ``ingest`` grant for the session principal (the v2 provider
           asserted ``_CAPTURE_GRANTS`` per call; v3 makes the same host
           authority durable) — skipped when a live grant already exists;
        2. a §11.11 capture authorization via :meth:`capture_auth`.

        Deployments that pre-provision both records (operator setup) may
        pass ``provision=False`` at initialize — every SDK call then runs
        against pre-seeded authority only.
        """
        sid = scope_id or self._scope_id
        if sid is None:
            raise VerbatimError(ErrorCode.VALIDATION, "no bound scope")
        if sid == self._scope_id:
            self._register_scope()
        principal = self._principal()
        with self.client.store.tx() as conn:
            live = grants_for(conn, sid, principal)
            has_ingest = any(
                "ingest" in set(repos_v3.json_field(g, "verbs_json") or ())
                for g in live
            )
            grant_id = None
            if not has_ingest:
                grant_id = create_grant(
                    conn,
                    scope_id=sid,
                    principal_id=principal,
                    verbs=["ingest"],
                    issuer_id=_ISSUER,
                )
        return {
            "scope_id": sid,
            "principal_id": principal,
            "grant_id": grant_id,
            "authorization_id": self.capture_auth(sid),
        }

    # ------------------------------------------------------------------
    # capture hooks (translate → SDK; errors decline silently per §43)
    # ------------------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        turn_author: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Host-attested turn capture (§13.03): user turn is
        ``principal_direct`` testimony (bot authors degrade to
        ``unknown``); assistant text is ``agent_generated`` — never
        human-attributed (§13.11)."""
        if not self._capturing():
            return
        try:
            user_sid = assist_sid = None
            user_env = assist_env = None
            scope = self._scope(session_id)
            if user_content and user_content.strip() and self._flag("user_messages"):
                speaker = self._speaker(scope, turn_author)
                user_env = (
                    EnvelopeBuilder()
                    .kind(EnvelopeKind.USER_MESSAGE)
                    .scope(self._scope_id)
                    .actor(speaker)
                    .perspective({"asserter": speaker, "observer": "hermes"})
                    .content(user_content)
                    .trust(
                        TrustClass.UNKNOWN
                        if (turn_author or {}).get("is_bot")
                        else TrustClass.PRINCIPAL_DIRECT
                    )
                    .external_id(f"hermes:{session_id or self._session_id}:turn:user:{now_us()}")
                    .build()
                )
                user_sid = self.client.capture_envelope(
                    self._ensure_session(), user_env
                ).source_id
            if (
                assistant_content
                and assistant_content.strip()
                and self._flag("assistant_context")
            ):
                agent = self._agent_principal()
                assist_env = (
                    EnvelopeBuilder()
                    .kind(EnvelopeKind.ASSISTANT_MESSAGE)
                    .scope(self._scope_id)
                    .actor(agent)
                    .perspective({"asserter": agent, "observer": "hermes"})
                    .content(assistant_content)
                    .trust(TrustClass.AGENT_GENERATED)
                    .external_id(f"hermes:{session_id or self._session_id}:turn:assistant:{now_us()}")
                    .build()
                )
                assist_sid = self.client.capture_envelope(
                    self._ensure_session(), assist_env
                ).source_id
            anchor = assist_sid or user_sid
            if anchor is not None:
                self.client.record_step(
                    self._sdk_session,
                    anchor,
                    {
                        "observation_source_ids": [
                            s for s in (user_sid,) if s and s != anchor
                        ]
                    },
                )
                self._last_source_id = anchor
        except VerbatimError as exc:
            if exc.code in _DECLINE_CODES:
                return  # gated capture declines silently (SPEC §43)
            self._note_hook_failure("sync_turn", exc)
        except Exception as exc:
            self._note_hook_failure("sync_turn", exc)

    def on_tool_result(
        self,
        tool_name: str,
        content: str,
        *,
        session_id: str = "",
        step_id: str = "",
        external_id: Optional[str] = None,
    ) -> None:
        """A host-observed tool result → ``tool_result`` envelope + step."""
        if not self._capturing() or not self._flag("tool_outputs"):
            return
        if not content or not str(content).strip():
            return
        try:
            agent = self._agent_principal()
            env = (
                EnvelopeBuilder()
                .kind(EnvelopeKind.TOOL_RESULT)
                .scope(self._scope_id)
                .actor(agent)
                .perspective({"asserter": "hermes", "observer": "hermes"})
                .content(str(content))
                .trust(TrustClass.HOST_OBSERVED)
                .step(step_id)
                .external_id(
                    external_id
                    or f"hermes:{session_id or self._session_id}:tool:{tool_name}:{now_us()}"
                )
                .metadata(tool_name=str(tool_name))
                .build()
            )
            sid = self.client.capture_envelope(self._ensure_session(), env).source_id
            self.client.record_step(self._sdk_session, sid, {})
            self._last_source_id = sid
        except VerbatimError as exc:
            if exc.code in _DECLINE_CODES:
                return
            self._note_hook_failure("tool_result", exc)
        except Exception as exc:
            self._note_hook_failure("tool_result", exc)

    def on_task_end(
        self,
        outcome: Optional[Dict[str, Any]] = None,
        *,
        checker: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Checker-attested task outcome (§12.03). ``checker`` maps to the
        frozen ``CheckerReceipt`` fields (checker_id, checker_version,
        invocation_id, selected_tests, completed, exit_code, ...); absent
        a checker_id the record persists as evidence only."""
        if not self._capturing() or self._last_source_id is None:
            return
        try:
            self.client.record_outcome(
                self._sdk_session,
                self._last_source_id,
                outcome or {},
                receipts=[checker] if checker else None,
            )
        except VerbatimError as exc:
            if exc.code in _DECLINE_CODES:
                return
            self._note_hook_failure("task_end", exc)
        except Exception as exc:
            self._note_hook_failure("task_end", exc)

    def end_capture_session(self, status: str = "complete") -> Dict[str, Any]:
        """Close the open capture session only — the trajectory completes
        and ``episode_build`` enqueues inside the SDK transaction (§20.08).
        No drain runs here; hosts that own their drain seam (the
        registered provider drains through the Engine's provisioned
        ingester) call this and drain themselves.

        A close failure is non-fatal but never silent (V4-14.11): the
        session handle is kept so a later seam can retry, and the error
        is recorded/journaled/logged — the summary reports
        ``{"closed": False, "end_session_error": ...}``.
        """
        if self._client is None or self._sdk_session is None:
            return {}
        try:
            summary = self.client.end_session(self._sdk_session, status=status)
        except Exception as exc:
            self._note_hook_failure(
                "end_session", exc, session_id=self._sdk_session
            )
            return {"closed": False, "end_session_error": _err_text(exc)}
        self._sdk_session = None
        self._last_source_id = None
        return summary

    def on_session_end(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        *,
        drain: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Close the capture session, then drain queued work on this
        session-end idle seam (§40).

        The drain mirrors ``provider.on_session_end``'s ``run_pending`` —
        without it a pure-hook deployment never executes harvest/admit,
        screens, episode builds, or privacy-control purges. ``drain``
        lets the caller substitute its own ingester (the registered
        provider drains through the Engine's provisioned pipeline —
        V4-07.09); the default is the client's own ``drain_pending``.

        F4-21/V4-05.13/V4-14.07: a drain or session-close failure is
        non-fatal for the host but NEVER a silent ``except: pass`` —
        ``summary["drain_error"]``/``["end_session_error"]`` carry it,
        ``self.drain_error`` stays set until a seam succeeds, a durable
        ``adapter_hook_failed`` event is journaled, and the host log is
        notified. Accepted obligations remain queued — ``summary["pending"]``
        reports the durable count so a caller cannot mistake the hook's
        return for completion (C83)."""
        if self._client is None:
            return {}
        summary = self.end_capture_session()
        drain_fn = drain if drain is not None else self.client.drain_pending
        try:
            summary["drained"] = drain_fn(limit=64)
        except Exception as exc:
            summary["drain_error"] = _err_text(exc)
            self._note_hook_failure("drain", exc, drain=True)
        else:
            self._drain_error = None
        summary["pending"] = self._pending_jobs()
        return summary

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        """Close the current capture session (if open), rebind, and arm
        the next scope — the next capture lazily opens its session.

        A close failure is recorded/logged (never silent, V4-14.11) but
        the handle is dropped regardless — it is bound to the OLD scope
        and would only poison captures under the new one; the durable
        trajectory row stays open for operator follow-up."""
        if self._sdk_session is not None:
            try:
                self.client.end_session(self._sdk_session, status="complete")
            except Exception as exc:
                self._note_hook_failure(
                    "session_switch_close", exc, session_id=self._sdk_session
                )
            self._sdk_session = None
        self._session_id = new_session_id
        self._last_source_id = None
        if self._host is not None:
            self._host.rebind(new_session_id)
            self._scope_id = scope_key(self._scope(new_session_id))

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirroring is off by default (SPEC §33); built-in memory text
        may be generated — nothing captures here unless the operator
        opted in, and then only as a host-observed system event."""
        if not (self._cfg and getattr(self._cfg.capture, "mirror_builtin", False)):
            return
        if not self._capturing() or not content:
            return
        try:
            env = (
                EnvelopeBuilder()
                .kind(EnvelopeKind.SYSTEM_EVENT)
                .scope(self._scope_id)
                .actor(self._agent_principal())
                .content(str(content))
                .trust(TrustClass.HOST_OBSERVED)
                .external_id(f"hermes:{self._session_id}:memwrite:{action}:{now_us()}")
                .metadata(action=str(action), target=str(target))
                .build()
            )
            self.client.capture_envelope(self._ensure_session(), env)
        except VerbatimError as exc:
            if exc.code in _DECLINE_CODES:
                return
            self._note_hook_failure("memory_write", exc)
        except Exception as exc:
            self._note_hook_failure("memory_write", exc)

    # ------------------------------------------------------------------
    # failure surfaces (F4-21 / V4-14.11)
    # ------------------------------------------------------------------

    @property
    def drain_error(self) -> Optional[Dict[str, Any]]:
        """The last session-end drain failure — ``None`` only when the
        most recent seam drained. Non-empty means accepted obligations
        remain queued; session-end processing did not complete."""
        return self._drain_error

    @property
    def last_hook_error(self) -> Optional[Dict[str, Any]]:
        """The last non-denial capture-hook failure recorded by this
        adapter (code/details preserved where typed)."""
        return self._last_hook_error

    def _note_hook_failure(
        self, phase: str, exc: Exception, *, drain: bool = False, **extra: Any
    ) -> None:
        """Record a hook failure: status surface + durable event + host
        log. Never raises — a hook must not break the host, but a failure
        must never masquerade as success."""
        rec: Dict[str, Any] = {"phase": phase, "at_us": now_us()}
        code = getattr(exc, "code", None)
        if code is not None:
            rec["code"] = getattr(code, "value", code)
        rec["error"] = _err_text(exc)
        rec.update({k: v for k, v in extra.items() if v is not None})
        self._last_hook_error = rec
        if drain:
            self._drain_error = rec
        try:
            if self._client is not None and self._scope_id:
                from ..storage.repos import EventsRepo

                with self.client.store.tx() as conn:
                    EventsRepo(self.client.store).append(
                        conn,
                        self._scope_id,
                        "adapter_hook_failed",
                        _ISSUER,
                        dict(rec),
                        "hermes-adapter-v1",
                    )
        except Exception:
            pass  # the store itself may be the failure — the log still reports
        if self._host is not None:
            self._host.log("error", f"verbatim_{phase}_failed", **rec)

    def _pending_jobs(self) -> Optional[int]:
        """Durable obligations still owed (queued/retry/leased) — None
        when the store itself is unreachable."""
        try:
            with self.client.store.read() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM jobs"
                    " WHERE state IN ('queued','retry_wait','leased')"
                ).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return None

    # ------------------------------------------------------------------
    # internals — translation only
    # ------------------------------------------------------------------

    def _run_hook(self, event: Dict[str, Any]) -> Any:
        hook = event["hook"]
        if hook == "sync_turn":
            self.sync_turn(
                event.get("user_content") or "",
                event.get("assistant_content") or "",
                session_id=event.get("session_id") or "",
                turn_author=event.get("turn_author"),
            )
            return {"captured": True}
        if hook == "tool_result":
            self.on_tool_result(
                event.get("tool_name") or "",
                event.get("content") or "",
                session_id=event.get("session_id") or "",
            )
            return {"captured": True}
        if hook == "task_end":
            self.on_task_end(event.get("outcome"), checker=event.get("checker"))
            return {"captured": True}
        if hook == "session_end":
            return self.on_session_end()
        if hook == "session_switch":
            self.on_session_switch(event.get("new_session_id") or "")
            return {"switched": True}
        raise VerbatimError(ErrorCode.VALIDATION, f"unknown hook {hook!r}")

    def _capturing(self) -> bool:
        return (
            self._client is not None
            and self._host is not None
            and self._writes_enabled
            and bool(getattr(self._cfg.capture, "enabled", True))
            and self._scope_id is not None
        )

    def _flag(self, name: str) -> bool:
        """The operator's capture toggle for a kind family (§12 table)."""
        return bool(getattr(self._cfg.capture, name, True))

    def _scope(self, session_id: str = "") -> Scope:
        """Mirror of provider's ``_session_scope``: per-session
        conversation partition over the host profile."""
        base = self._host.default_scope()
        return Scope(
            profile_id=base.profile_id,
            principal_id=base.principal_id or _FALLBACK_PRINCIPAL,
            workspace_id=base.workspace_id,
            conversation_id=session_id or self._session_id or "nosession",
            visibility=base.visibility,
        )

    def _principal(self) -> str:
        base = self._host.default_scope() if self._host is not None else None
        return (base.principal_id if base else None) or _FALLBACK_PRINCIPAL

    def _agent_principal(self) -> str:
        """The agent's own principal id — assistant/tool envelopes are
        attributed to it, never to the user (§13.11)."""
        return f"hermes:{self._agent_context}"

    def _speaker(self, scope: Scope, turn_author: Optional[Dict[str, Any]]) -> str:
        if turn_author and turn_author.get("is_bot"):
            return f"bot:{turn_author.get('id') or 'unknown'}"
        return (turn_author or {}).get("id") or scope.principal_id or _FALLBACK_PRINCIPAL

    def _register_scope(self) -> None:
        """Persist the bound session scope's decomposed row in ``scopes``.

        v3 partition ids are opaque: ``ingest_envelope`` would otherwise
        create the row as ``owner``/NULL and the shared v2 read model
        (``can_read`` over the ``scopes`` registry) could never authorize
        the partition — captured evidence would stay dark to
        ``Engine.recall`` on a consolidated store (F4-17/C81). The
        decomposed tuple is the honest metadata the token was derived
        from; ``INSERT OR IGNORE`` keeps whichever row lands first, so a
        host-pre-provisioned row is never rewritten.
        """
        if (
            self._client is None
            or not self._writes_enabled
            or self._scope_id is None
            or getattr(self.client, "store", None) is None
        ):
            return
        from ..storage.repos import ensure_scope

        with self.client.store.tx() as conn:
            ensure_scope(
                self.client.store, conn, self._scope(self._session_id)
            )

    def _ensure_session(self) -> str:
        """Lazily (re)open the SDK session for the bound scope — the next
        capture after initialize/session_switch arms a fresh session."""
        if self._sdk_session is None:
            self._register_scope()
            if self._provision:
                self.setup_scope(self._scope_id)
            self._sdk_session = self.client.begin_session(
                self._principal(),
                "hermes",
                metadata={
                    "scope_id": self._scope_id,
                    "boundary_rule": "session",
                    "hermes_session": self._session_id,
                },
            )
        return self._sdk_session


__all__ = ["HermesV3Adapter"]
