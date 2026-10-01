"""Capture SDK — the §13 depth-1 integration path (SPEC_V3 §13, §11.11, §14).

``CaptureClient`` is what an agent host embeds: it takes a ``Store`` (or a
store path + config) and turns host events into governed v3 evidence —
sessions, capture authorizations, envelopes, trajectory steps, checker
outcomes, and the ``episode_build`` pipeline trigger. It is local-first:
no network, no MCP server, no host runtime — every write is one
``store.tx()`` transaction and every call re-checks both the ``ingest``
verb grant and the §11.11 capture authorization (§13.01: an adapter can
obtain nothing the engine would deny a direct caller).

Authorization semantics (§46):

- Missing/revoked ``ingest`` grant → ``NOT_FOUND_OR_UNAUTHORIZED`` — absent
  and forbidden are publicly indistinguishable (§09.09).
- Missing/expired/revoked capture authorization → ``CONSENT_REQUIRED`` —
  the typed §12.10 denial, never a silent kind downgrade.
- Unknown session/source ids → ``NOT_FOUND_OR_UNAUTHORIZED``.

Two capture paths are deliberately distinct:

- :meth:`submit_source` — *agent-submitted* provenance (§13.11): text is
  recorded exactly but attributed to the agent; ``user_message`` and
  ``principal_*`` trust claims are rejected, never relabeled.
- :meth:`capture_envelope` — *host-attested* provenance (§13.03): the host
  asserts actual per-event authorship (user turns, tool observations).
  Native adapters (depth 2) use this path.

Deployment obligation (§40): captures only *enqueue* durable jobs —
harvest→admit→claim, ``episode_build``, screens, and the privacy-control
lane's purges. Nothing on the SDK path drains implicitly, so the host MUST
schedule :meth:`CaptureClient.drain_pending` (on an idle/session boundary,
or via its own worker loop) or every obligation waits forever — including
suppression and purge jobs. The drain is an explicit synchronous host
call; the library never spawns threads or loops of its own.
"""

from __future__ import annotations

import dataclasses
import hashlib
import sqlite3
from typing import Any, Iterable, Optional, Union

from ..config import VerbatimConfig
from ..core.time import now_us
from ..core.types import (
    ErrorCode,
    JobKind,
    Scope,
    VerbatimError,
    json_dumps,
    new_id,
    require_id,
)
from ..core.types_v3 import (
    CheckerReceipt,
    EnvelopeKind,
    OutcomeClass,
    OutcomeEnvelope,
    SourceEnvelopeV3,
    TrajectoryRecord,
    TrajectoryStep,
    TrustClass,
)
from ..evidence import ingest_envelope
from ..evidence.receipts import CaptureReceipt
from ..evidence.trajectories import (
    add_step,
    complete,
    get_trajectory,
    record_trajectory,
    steps as trajectory_steps,
)
from ..evidence.detect import is_textual
from ..governance import (
    CallerV3,
    authorize as _authorize_verb,
    issue_capture_authorization,
    require_capture_authorization,
)
from ..jobs.queue import JobQueue
from ..security import (
    attach_label,
    default_review_state,
    open_quarantine,
    screen_content,
)
from ..storage import repos_v3
from ..storage.store import Store
from .envelope import (
    SDK_VERSION,
    EnvelopeBuilder,
    checker_from,
    environment_from,
    outcome_descriptor,
    outcome_from,
    step_from,
)

_SDK_POLICY = "verbatim_sdk_capture_v1"
_OUTCOME_KINDS = {EnvelopeKind.TEST_RESULT, EnvelopeKind.VERIFICATION}
_TERMINAL_STATUS = frozenset({"complete", "abandoned", "failed"})
#: Envelope kinds whose provenance is the external world, not the agent —
#: the only non-agent trust an agent-submitted source may honestly carry.
_EXTERNAL_SUBMITTED = {
    EnvelopeKind.DOCUMENT: TrustClass.EXTERNAL_CONTENT,
    EnvelopeKind.CONNECTOR_ITEM: TrustClass.EXTERNAL_CONTENT,
    EnvelopeKind.IMPORT: TrustClass.IMPORTED,
}
#: Every kind a session capture may legitimately write; `authorize()`
#: defaults to this set and callers may narrow it (§11.11).
SESSION_KINDS = frozenset(EnvelopeKind)


def _kind(value: "EnvelopeKind | str") -> EnvelopeKind:
    try:
        return value if isinstance(value, EnvelopeKind) else EnvelopeKind(value)
    except ValueError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown envelope kind {value!r}"
        ) from exc


def _deny(msg: str = "not found or unauthorized") -> None:
    raise VerbatimError(ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, msg)


class _Session:
    """SDK-side session state; durable truth lives in the trajectory row."""

    __slots__ = (
        "session_id", "principal_id", "host_id", "metadata", "trajectory_id",
        "scope_id", "next_ord", "closed", "status", "captured_kinds",
    )

    def __init__(self, session_id: str, principal_id: str, host_id: str,
                 metadata: dict[str, Any]) -> None:
        self.session_id = session_id
        self.principal_id = principal_id
        self.host_id = host_id
        self.metadata = metadata
        self.trajectory_id: Optional[str] = None
        self.scope_id: Optional[str] = None
        self.next_ord = 0
        self.closed = False
        self.status = "open"
        self.captured_kinds: set[str] = set()


def _trajectory_id(session_id: str) -> str:
    """Deterministic trajectory identity for a session — a client restart
    resumes the same trajectory instead of forking evidence (§13.07)."""
    digest = hashlib.sha256(
        f"v3:sdk-session\x00{session_id}".encode("utf-8")
    ).hexdigest()
    return f"traj_{digest[:32]}"


def _submitted_trust(kind: EnvelopeKind) -> TrustClass:
    """Trust cap for agent-submitted content (§13.11, §14.09): the agent
    relays or authors it — never human testimony, never host-observed."""
    return _EXTERNAL_SUBMITTED.get(kind, TrustClass.AGENT_GENERATED)


class CaptureClient:
    """The embeddable capture SDK (§13 depth 1).

    ``store`` may be an open :class:`Store` or a filesystem path — a path is
    opened when the database exists and created otherwise (the client then
    owns it and :meth:`close` closes it). ``config`` tunes queue bounds;
    ``scope_id``/``principal_id`` set client-level defaults sessions
    inherit.
    """

    def __init__(
        self,
        store: "Store | str",
        *,
        config: Optional[VerbatimConfig] = None,
        scope_id: Optional[str] = None,
        principal_id: Optional[str] = None,
        host_id: str = "sdk",
        adapter_version: str = SDK_VERSION,
    ) -> None:
        import os

        self._owns_store = False
        if isinstance(store, Store):
            self.store = store
        elif isinstance(store, (str, os.PathLike)):
            path = os.fspath(store)
            self.store = (
                Store.open(path) if os.path.exists(path) else Store.create(path)
            )
            self._owns_store = True
        else:
            raise VerbatimError(
                ErrorCode.VALIDATION, "store must be a Store or a path"
            )
        self._cfg = config or VerbatimConfig()
        self._scope_id = scope_id
        self._principal_id = principal_id
        self._host_id = host_id
        self._adapter_version = adapter_version
        self._sessions: dict[str, _Session] = {}
        self._auth_ids: set[str] = set()
        self._queue: Optional[JobQueue] = None
        self._ingester: Optional[Any] = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._owns_store:
            self.store.close()

    def _jobs(self) -> JobQueue:
        if self._queue is None:
            self._queue = JobQueue(
                self.store,
                max_pending=self._cfg.jobs.max_pending,
                policy_epoch=self.store.policy_epoch(),
            )
        return self._queue

    def _worker(self) -> Any:
        """The lazily-constructed :class:`~verbatim.ingest.Ingester` that
        drains this client's store — same store + config the captures use,
        no judge/encoder provisions (embedding jobs degrade to recorded
        ``encoder_unavailable`` notes, never silent no-ops)."""
        if self._ingester is None:
            from ..ingest import Ingester  # local: sdk → ingest edge

            self._ingester = Ingester(self.store, self._cfg)
        return self._ingester

    def drain_pending(
        self,
        scope: "Scope | str | None" = None,
        *,
        limit: int = 64,
        kinds: Optional[Iterable["JobKind | str"]] = None,
        lane: Optional[str] = None,
        owner: Optional[str] = None,
    ) -> int:
        """Synchronously drain due durable jobs; returns the drained count.

        The host calls this on an idle/session boundary (or from its own
        worker) — §40 four-lane dequeue order applies: ``control`` and
        ``privacy_control`` lanes first, then ``maintenance``,
        ``background``, then ``ordinary``. ``scope`` may be an opaque v3
        scope id or a :class:`Scope`; ``None`` drains across scopes
        (trusted standalone-worker semantics — per-job scope is still
        re-verified by each handler at commit). ``kinds`` narrows the kind
        set, ``lane`` restricts to one lane for dedicated workers,
        ``limit`` bounds the drained count per call.

        Nothing drains implicitly: without this call (or a real worker),
        queued harvest/screen/admit/episode/purge obligations wait forever.
        """
        return self._worker().run_pending(
            scope,
            limit=limit,
            owner=owner or f"sdk:{self._host_id}",
            kinds=kinds,
            lane=lane,
        )

    # ------------------------------------------------------------------
    # sessions
    # ------------------------------------------------------------------

    def begin_session(
        self,
        principal_id: str,
        host_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Open a capture session; returns ``session_id``.

        ``metadata`` may carry ``scope_id`` (binds the session scope and
        opens its trajectory eagerly), ``task_id``, ``boundary_rule`` (how
        the task boundary is detected — declared, never inferred, §20.01),
        and ``environment`` (fingerprint mapping). Without a scope the
        session binds on its first scoped call.
        """
        principal = require_id(principal_id, "principal_id")
        host = str(host_id or self._host_id)
        meta = dict(metadata or {})
        session_id = f"sess_{new_id()}"
        session = _Session(session_id, principal, host, meta)
        self._sessions[session_id] = session
        scope = meta.get("scope_id") or self._scope_id
        if scope is not None:
            with self.store.tx() as conn:
                self._require_ingest_verb(conn, session, scope)
                self._bind_scope(session, scope)
                self._ensure_trajectory(conn, session, scope)
        return session_id

    def resume_session(
        self,
        session_id: str,
        *,
        principal_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Rebuild session state from the durable trajectory after a client
        restart (§13.07 restart conformance)."""
        require_id(session_id, "session_id")
        tid = _trajectory_id(session_id)
        with self.store.read() as conn:
            traj = get_trajectory(conn, tid)
            if traj is None:
                _deny("session not found or unauthorized")
            meta = repos_v3.json_field(traj, "metadata_json", {}) or {}
            session = _Session(
                session_id,
                principal_id or meta.get("principal_id") or traj["host_id"],
                traj["host_id"] or self._host_id,
                meta,
            )
            session.trajectory_id = tid
            session.scope_id = traj["scope_id"]
            done = trajectory_steps(conn, tid)
            session.next_ord = len(done)
            session.closed = traj["completed_event"] is not None
            session.status = meta.get("session_status") or (
                "complete" if session.closed else "open"
            )
            for row in done:
                for eid in [row.get("action_envelope_id")] + list(
                    row.get("observation_envelope_ids") or ()
                ):
                    if not eid:
                        continue
                    env = repos_v3.get(
                        conn, "source_envelopes", {"envelope_id": eid}
                    )
                    if env is not None:
                        session.captured_kinds.add(env["envelope_kind"])
            self._sessions[session_id] = session
        return self.session_state(session_id)

    def session_state(self, session_id: str) -> dict[str, Any]:
        """A plain snapshot of SDK session state (no store access)."""
        st = self._sessions.get(session_id)
        if st is None:
            _deny("session not found or unauthorized")
        return {
            "session_id": st.session_id,
            "principal_id": st.principal_id,
            "host_id": st.host_id,
            "trajectory_id": st.trajectory_id,
            "scope_id": st.scope_id,
            "next_ord": st.next_ord,
            "closed": st.closed,
            "status": st.status,
            "captured_kinds": sorted(st.captured_kinds),
        }

    def _open_session(self, session_id: str) -> _Session:
        require_id(session_id, "session_id")
        st = self._sessions.get(session_id)
        if st is None:
            _deny("session not found or unauthorized")
        if st.closed:
            raise VerbatimError(
                ErrorCode.VALIDATION, "session already ended"
            )
        return st

    # ------------------------------------------------------------------
    # capture authorization (§11.11)
    # ------------------------------------------------------------------

    def authorize(
        self,
        scope_id: str,
        granted_by: str,
        purpose: Optional[str] = None,
        ttl_s: Optional[float] = None,
        *,
        kinds: Optional[Iterable["EnvelopeKind | str"]] = None,
        principal_id: Optional[str] = None,
        retention_policy: Optional[str] = None,
    ) -> str:
        """Issue a capture authorization covering ``scope_id`` (§11.11).

        Delegates to ``governance.issue_capture_authorization``. Only
        host/operator surfaces may issue — ``granted_by`` is that issuer
        identity, not a model claim. ``kinds`` narrows the covered envelope
        kinds (default: the full session-capture set); ``ttl_s`` bounds the
        consent's lifetime; ``purpose``/``retention_policy`` records the
        declared retention policy.
        """
        scope = require_id(scope_id, "scope_id")
        issuer = require_id(granted_by, "granted_by")
        principal = principal_id or self._principal_id
        if principal is None:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "authorize needs a principal_id (argument or client default)",
            )
        kind_set = (
            frozenset(_kind(k) for k in kinds)
            if kinds is not None
            else SESSION_KINDS
        )
        expires = (
            now_us() + int(ttl_s * 1_000_000) if ttl_s is not None else None
        )
        with self.store.tx() as conn:
            aid = issue_capture_authorization(
                conn,
                principal_id=principal,
                issuer_id=issuer,
                allowed_kinds=sorted(kind_set, key=lambda k: k.value),
                retention_policy=retention_policy or purpose or "task",
                policy_revision=_SDK_POLICY,
                scope_ids=[scope],
                expires_us=expires,
            )
        self._auth_ids.add(aid)
        return aid

    # ------------------------------------------------------------------
    # capture paths
    # ------------------------------------------------------------------

    def submit_source(
        self,
        session_id: str,
        scope_id: str,
        content: "bytes | str",
        *,
        declared_type: "EnvelopeKind | str",
        title: Optional[str] = None,
        media_type: Optional[str] = None,
        external_id: Optional[str] = None,
        event_us: Optional[int] = None,
    ) -> str:
        """Persist an *agent-submitted* source; returns ``source_id``.

        §13.11: the payload is recorded exactly as submitted but attributed
        to the session's agent principal — ``user_message`` is rejected
        (agent text is never human testimony) and trust caps at
        ``agent_generated`` (or ``external_content``/``imported`` for
        document-class kinds). Every call re-checks the ``ingest`` verb and
        the §11.11 capture authorization, then runs security admission.
        """
        session = self._open_session(session_id)
        kind = _kind(declared_type)
        if kind == EnvelopeKind.USER_MESSAGE:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "agent-submitted content is never human testimony — "
                "use declared_type 'agent_note' (§13.11)",
            )
        builder = (
            EnvelopeBuilder()
            .kind(kind)
            .scope(scope_id)
            .actor(session.principal_id)
            .content(content, media_type=media_type)
            .trust(_submitted_trust(kind))
            .host(session.host_id)
            .session(session_id)
            .task(str(session.metadata.get("task_id") or ""))
            .adapter_version(self._adapter_version)
        )
        if title is not None:
            builder.meta("title", str(title))
        if external_id is not None:
            builder.external_id(external_id)
        builder.event_us(event_us if event_us is not None else now_us())
        receipt = self._ingest_authorized(session, builder.build())
        return receipt.source_id

    def capture_envelope(
        self,
        session_id: str,
        envelope: "SourceEnvelopeV3 | EnvelopeBuilder | dict",
    ) -> CaptureReceipt:
        """Persist a *host-attested* envelope; returns the CaptureReceipt.

        The host asserts actual authorship and provenance (§13.03) — user
        turns carry ``user_message``/``principal_direct``, host-observed
        tool events carry ``host_observed``. Session scope fields are
        stamped from the session when absent; an explicit ``capture_proof``
        wins, else the SDK binds the session's covering authorization.
        """
        session = self._open_session(session_id)
        env = self._normalize_envelope(envelope)
        if not env.session_id:
            env = dataclasses.replace(env, session_id=session_id)
        if not env.host_id:
            env = dataclasses.replace(env, host_id=session.host_id)
        if not env.adapter_version:
            env = dataclasses.replace(env, adapter_version=self._adapter_version)
        if env.event_us == 0:
            env = dataclasses.replace(env, event_us=now_us())
        return self._ingest_authorized(session, env)

    # ------------------------------------------------------------------
    # trajectories / outcomes (§12.02–§12.03)
    # ------------------------------------------------------------------

    def record_step(
        self,
        session_id: str,
        source_id: str,
        step: "TrajectoryStep | dict",
    ) -> str:
        """Append one ordered trajectory step; returns ``step_id``.

        ``source_id`` anchors the step: its latest envelope becomes the
        action envelope when the step names none, and its scope binds the
        session's trajectory. Dict steps may set ``ord`` explicitly
        (dense/monotonic — a gap is a typed error); omitted ords append at
        the next position. ``observation_envelope_ids`` and
        ``observation_source_ids`` are both accepted.
        """
        session = self._open_session(session_id)
        # A typed TrajectoryStep always carries an explicit ord (the caller
        # manages order); dict steps append at the next position unless
        # they set "ord" themselves — the dense-ord check rejects gaps.
        explicit_ord: Optional[int] = (
            step.get("ord") if isinstance(step, dict) else step.ord
        )
        with self.store.tx() as conn:
            env_row = self._envelope_for_source(conn, source_id)
            scope = env_row["scope_id"]
            self._bind_scope(session, scope)
            self._require_ingest_verb(conn, session, scope)
            self._ensure_trajectory(conn, session, scope)

            normalized = step_from(step)
            if normalized.trajectory_id and normalized.trajectory_id != session.trajectory_id:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "step.trajectory_id does not match the session trajectory",
                )
            action_env = normalized.action_envelope_id or env_row["envelope_id"]
            obs_ids = list(normalized.observation_envelope_ids)
            if isinstance(step, dict):
                for sid in step.get("observation_source_ids") or ():
                    obs_ids.append(
                        self._envelope_for_source(conn, sid)["envelope_id"]
                    )
            obs_ids = list(dict.fromkeys(obs_ids))
            self._require_capture_for_refs(
                conn, session, scope, [action_env, *obs_ids]
            )
            ord_ = (
                int(explicit_ord)
                if explicit_ord is not None
                else session.next_ord
            )
            record = TrajectoryStep(
                step_id=normalized.step_id,
                trajectory_id=session.trajectory_id,
                ord=ord_,
                action_envelope_id=action_env,
                observation_envelope_ids=tuple(obs_ids),
                state_delta_refs=tuple(normalized.state_delta_refs),
                environment=normalized.environment,
            )
            step_id = add_step(conn, record)
            session.next_ord = ord_ + 1
            return step_id

    def record_outcome(
        self,
        session_id: str,
        source_id: str,
        outcome: "OutcomeEnvelope | dict",
        receipts: Optional[Iterable["CheckerReceipt | dict"]] = None,
    ) -> str:
        """Persist a checker-attested outcome; returns the outcome
        ``envelope_id`` (§12.03, §13.05).

        The outcome lands as a ``test_result`` (or ``verification``)
        envelope whose metadata carries the identified checker receipt(s),
        then attaches to the session trajectory as an observation-only
        step — the episode builder resolves outcomes only from such
        envelopes, and an agent's bare "it worked" moves nothing (an
        outcome dict without ``checker.checker_id`` persists as evidence
        but can never upgrade the episode's ``unknown`` outcome).
        """
        session = self._open_session(session_id)
        normalized = outcome_from(outcome)
        extra = [c for c in (checker_from(r) for r in (receipts or ())) if c]
        declared_kind = (
            _kind(outcome.get("kind"))
            if isinstance(outcome, dict) and outcome.get("kind") is not None
            else EnvelopeKind.TEST_RESULT
        )
        if declared_kind not in _OUTCOME_KINDS:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "outcome kind must be test_result or verification",
            )
        with self.store.tx() as conn:
            env_row = self._envelope_for_source(conn, source_id)
            scope = env_row["scope_id"]
            self._bind_scope(session, scope)
            self._require_ingest_verb(conn, session, scope)
            self._ensure_trajectory(conn, session, scope)
            self._require_capture(conn, session, declared_kind, scope)

            descriptor = outcome_descriptor(normalized, extra_checkers=extra)
            descriptor["verifies_source_id"] = source_id
            raw_content = outcome.get("content") if isinstance(outcome, dict) else None
            content = (
                raw_content.encode("utf-8")
                if isinstance(raw_content, str)
                else bytes(raw_content)
                if raw_content is not None
                else json_dumps(descriptor).encode("utf-8")
            )
            auth_id = self._resolve_auth_id(
                conn, session, declared_kind, scope
            )
            env = (
                EnvelopeBuilder()
                .kind(declared_kind)
                .scope(scope)
                .actor(session.principal_id)
                .content(content, media_type="application/json")
                .trust(TrustClass.HOST_OBSERVED)
                .host(session.host_id)
                .session(session_id)
                .task(str(session.metadata.get("task_id") or ""))
                .capture_proof(auth_id)
                .adapter_version(self._adapter_version)
                .event_us(normalized.recorded_us or now_us())
                .metadata(**descriptor)
                .build()
            )
            # In-transaction ingest: outcome envelope + observation step
            # commit together — never an outcome without its step (§47.02).
            receipt = self._ingest_in_tx(conn, session, env)
            # Attach as an observation-only step: the episode builder reads
            # outcome envelopes through trajectory membership (§12.03).
            obs = TrajectoryStep(
                step_id=f"step:{new_id()}",
                trajectory_id=session.trajectory_id,
                ord=session.next_ord,
                action_envelope_id=None,
                observation_envelope_ids=(receipt.envelope_id,),
            )
            add_step(conn, obs)
            session.next_ord += 1
            session.captured_kinds.add(declared_kind.value)
            return receipt.envelope_id

    # ------------------------------------------------------------------
    # session end → episode pipeline (§12.02, §20.08, §40)
    # ------------------------------------------------------------------

    def end_session(self, session_id: str, status: str = "complete") -> dict[str, Any]:
        """Close the session's trajectory and enqueue ``episode_build``.

        Completion is a journaled one-way transition; the
        ``episode_build`` job (background lane) commits in the SAME
        transaction so a session can never end closed-but-unqueued
        (§20.08, §40). ``status`` is the host's lifecycle verdict —
        ``complete``/``abandoned``/``failed``; episode *outcome* is never
        set from it (only checker evidence moves an outcome, §12.03).
        """
        require_id(session_id, "session_id")
        session = self._sessions.get(session_id)
        if session is None:
            _deny("session not found or unauthorized")
        if status not in _TERMINAL_STATUS:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"status must be one of {sorted(_TERMINAL_STATUS)}",
            )
        if session.closed:
            # Idempotent close: a retried end reports the same terminal
            # state instead of double-committing (the trajectory's
            # completed_event and the deduped job already landed).
            return self.session_state(session_id)
        summary = self.session_state(session_id)
        summary["status"] = status
        if session.trajectory_id is None:
            # Nothing was ever captured — no trajectory, no episode work.
            session.closed = True
            session.status = status
            summary["closed"] = True
            return summary

        tid = session.trajectory_id
        scope = session.scope_id
        with self.store.tx() as conn:
            self._require_ingest_verb(conn, session, scope)
            # Revocation/expiry mid-session denies the close exactly like
            # any other capture event (§13.07, §48.12).
            for kind_value in sorted(session.captured_kinds):
                require_capture_authorization(
                    conn, session.principal_id, kind_value, scope
                )
            traj = complete(conn, tid, store=self.store)
            repos_v3.update(
                conn,
                "trajectories",
                {
                    "metadata_json": {
                        **(repos_v3.json_field(traj, "metadata_json", {}) or {}),
                        "session_status": status,
                    }
                },
                {"trajectory_id": tid},
            )
            dedup = self.store.hmac(f"episode_build:{tid}".encode("utf-8"))
            job_id = self._jobs().enqueue(
                conn,
                scope,
                JobKind.EPISODE_BUILD,
                {"trajectory_id": tid},
                dedup_key=dedup,
                operation_key=(
                    f"episode_build:{tid}"
                    if self._jobs().supports_durability
                    else None
                ),
            )
            session.closed = True
            session.status = status
        summary.update(
            {
                "closed": True,
                "status": status,
                "completed_event": traj["completed_event"],
                "episode_build_job_id": job_id,
            }
        )
        return summary

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _normalize_envelope(
        self, envelope: "SourceEnvelopeV3 | EnvelopeBuilder | dict"
    ) -> SourceEnvelopeV3:
        if isinstance(envelope, SourceEnvelopeV3):
            return envelope
        if isinstance(envelope, EnvelopeBuilder):
            return envelope.build()
        if isinstance(envelope, dict):
            return EnvelopeBuilder.from_dict(envelope).build()
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "envelope must be a SourceEnvelopeV3, EnvelopeBuilder, or dict",
        )

    def _caller(self, session: _Session) -> CallerV3:
        return CallerV3(
            principal_id=session.principal_id,
            session_id=session.session_id,
            host_id=session.host_id,
        )

    def _require_ingest_verb(
        self, conn: sqlite3.Connection, session: _Session, scope_id: str
    ) -> None:
        """§13.01: the SDK caller needs the same ``ingest`` grant a direct
        caller would — denial is indistinguishable from absence."""
        _authorize_verb(
            conn, self._caller(session), scope_id, "ingest"
        )

    def _require_capture(
        self,
        conn: sqlite3.Connection,
        session: _Session,
        kind: EnvelopeKind,
        scope_id: str,
    ) -> None:
        """§11.11/§12.10 typed consent check — ``CONSENT_REQUIRED``."""
        require_capture_authorization(
            conn, session.principal_id, kind, scope_id
        )

    def _resolve_auth_id(
        self,
        conn: sqlite3.Connection,
        session: _Session,
        kind: EnvelopeKind,
        scope_id: str,
    ) -> Optional[str]:
        """The live authorization covering (principal, kind, scope) — the
        ``capture_proof`` binding (§12.10)."""
        now = now_us()
        rows = repos_v3.query(
            conn,
            "capture_authorizations",
            {"principal_id": session.principal_id, "revoked_us": None},
        )
        for row in rows:
            exp = row.get("expires_us")
            if exp is not None and int(exp) <= now:
                continue
            allowed = repos_v3.json_field(row, "allowed_kinds_json", []) or []
            if kind.value not in allowed:
                continue
            scopes = repos_v3.json_field(row, "scope_ids_json", []) or []
            if scopes and scope_id not in scopes:
                continue
            return row["authorization_id"]
        return None

    def _require_capture_for_refs(
        self,
        conn: sqlite3.Connection,
        session: _Session,
        scope_id: str,
        envelope_ids: Iterable[str],
    ) -> None:
        """Consent is checked per captured kind on every event (§48.12's
        discipline applies to SDK writes too)."""
        for eid in dict.fromkeys(envelope_ids):
            row = repos_v3.get(conn, "source_envelopes", {"envelope_id": eid})
            if row is None:
                # add_step reports the dangling reference precisely; the
                # scope check stays honest here.
                continue
            if row["scope_id"] != scope_id:
                _deny("referenced envelope not found or unauthorized")
            self._require_capture(
                conn, session, EnvelopeKind(row["envelope_kind"]), scope_id
            )

    def _bind_scope(self, session: _Session, scope_id: str) -> None:
        """A session binds exactly one scope — cross-scope evidence never
        mixes inside one trajectory."""
        if session.scope_id is None:
            session.scope_id = scope_id
        elif session.scope_id != scope_id:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "session is bound to a different scope",
            )

    def _ensure_trajectory(
        self, conn: sqlite3.Connection, session: _Session, scope_id: str
    ) -> None:
        """Lazily open the session trajectory inside the caller's tx."""
        if session.trajectory_id is not None:
            return
        tid = _trajectory_id(session.session_id)
        env_digest = None
        if session.metadata.get("environment") is not None:
            fp = environment_from(session.metadata["environment"])
            env_digest = fp.digest() if fp is not None else None
        record = TrajectoryRecord(
            trajectory_id=tid,
            scope_id=scope_id,
            host_id=session.host_id,
            session_id=session.session_id,
            task_id=str(session.metadata.get("task_id") or ""),
            boundary_rule=str(
                session.metadata.get("boundary_rule") or "session"
            ),
            environment_digest=env_digest,
            metadata={
                **session.metadata,
                "session_id": session.session_id,
                "principal_id": session.principal_id,
                "sdk": self._adapter_version,
            },
        )
        record_trajectory(conn, record, store=self.store)
        session.trajectory_id = tid

    def _envelope_for_source(
        self, conn: sqlite3.Connection, source_id: str
    ) -> dict[str, Any]:
        """Latest ``source_envelopes`` row for a source id."""
        require_id(source_id, "source_id")
        rows = repos_v3.query(
            conn,
            "source_envelopes",
            {"source_id": source_id},
            order="revision DESC",
            limit=1,
        )
        if not rows:
            _deny("source not found or unauthorized")
        return rows[0]

    def _screen(
        self,
        conn: sqlite3.Connection,
        envelope: SourceEnvelopeV3,
        envelope_id: str,
        revision: int,
        trust: TrustClass,
    ) -> Optional[str]:
        """Security admission (§14.01–§14.02, §34): rules_v1 verdict →
        security label → quarantine hold for blocked content.

        The label lands in the same transaction as the envelope — content
        is retained as evidence but a ``blocked`` verdict immediately opens
        quarantine (excluded from retrieval pending review). This is the
        SDK's own screening pass; ``ingest_envelope`` may additionally
        persist its default trust label.
        """
        text: Optional[str] = None
        if envelope.content is not None and is_textual(envelope.media_type):
            try:
                text = bytes(envelope.content).decode("utf-8")
            except UnicodeDecodeError:
                text = None
        if text is None:
            label_id = attach_label(
                conn,
                envelope.scope_id,
                source_trust=trust.value,
                content_form="unknown",
                attack_risk="unassessed",
                review_state=default_review_state(
                    trust.value, attack_risk="unassessed"
                ),
                method="rules",
                rules_revision="",
            )
            return label_id
        verdict = screen_content(text)
        review_state = default_review_state(
            trust.value,
            attack_risk=verdict.attack_risk.value,
            content_form=verdict.content_form.value,
            findings=list(verdict.findings),
        )
        label_id = attach_label(
            conn,
            envelope.scope_id,
            source_trust=trust.value,
            content_form=verdict.content_form.value,
            attack_risk=verdict.attack_risk.value,
            review_state=review_state,
            findings=list(verdict.findings),
            method=verdict.method,
            rules_revision=verdict.rules_revision,
        )
        if verdict.attack_risk.value == "blocked":
            reasons = ["attack_risk:blocked"] + sorted(
                {str(f.get("rule_id")) for f in verdict.findings if f.get("rule_id")}
            )
            open_quarantine(
                conn,
                ("source_envelope", envelope_id, revision),
                reasons or ["attack_risk:blocked"],
                list(verdict.findings),
                scope_id=envelope.scope_id,
            )
        return label_id

    def _ingest_authorized(
        self,
        session: _Session,
        envelope: SourceEnvelopeV3,
    ) -> CaptureReceipt:
        """The shared gated write: verb → consent → ingest → screening →
        trajectory bind, atomically (§13.01, §11.11, §14)."""
        with self.store.tx() as conn:
            return self._ingest_in_tx(conn, session, envelope)

    def _ingest_in_tx(
        self,
        conn: sqlite3.Connection,
        session: _Session,
        envelope: SourceEnvelopeV3,
    ) -> CaptureReceipt:
        """``_ingest_authorized``'s body inside the caller's transaction —
        composite writes (outcome + step) stay single-tx (§47.02)."""
        scope = envelope.scope_id
        self._bind_scope(session, scope)
        self._require_ingest_verb(conn, session, scope)
        self._require_capture(conn, session, envelope.kind, scope)
        auth_id = self._resolve_auth_id(conn, session, envelope.kind, scope)
        if envelope.capture_proof is None and auth_id is not None:
            envelope = dataclasses.replace(envelope, capture_proof=auth_id)
        receipt = ingest_envelope(conn, self.store, envelope)
        env_row = repos_v3.get(
            conn, "source_envelopes", {"envelope_id": receipt.envelope_id}
        )
        trust = TrustClass(
            (env_row or {}).get("trust_class") or envelope.trust_class.value
        )
        self._screen(
            conn, envelope, receipt.envelope_id, receipt.revision, trust
        )
        self._ensure_trajectory(conn, session, scope)
        session.captured_kinds.add(envelope.kind.value)
        return receipt


__all__ = ["CaptureClient", "SESSION_KINDS"]
