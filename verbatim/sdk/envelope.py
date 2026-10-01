"""Envelope construction for the capture SDK (SPEC_V3 §12.01–§12.03, §13).

``EnvelopeBuilder`` is the fluent path to a validated
:class:`~verbatim.core.types_v3.SourceEnvelopeV3`: every mutator is typed,
``build()`` runs the frozen dataclass validation plus the evidence-plane's
own ``_validate`` (same checks ``ingest_envelope`` applies), and
``to_dict()`` yields the language-neutral JSON shape the §13.02 event
schema publishes.

The ``step_from`` / ``outcome_from`` normalizers accept plain dicts (what a
JSONL event stream carries) and return the frozen
``TrajectoryStep``/``OutcomeEnvelope`` contracts — SDK callers never need
to import ``types_v3`` to record a step or an outcome, but the same
validation runs either way.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from ..core.types import ErrorCode, VerbatimError, json_dumps, new_id, require_id
from ..core.types_v3 import (
    CheckerReceipt,
    EnvelopeKind,
    EnvironmentFingerprint,
    OutcomeClass,
    OutcomeEnvelope,
    Perspective,
    SourceEnvelopeV3,
    TrajectoryStep,
    TrustClass,
)
from ..evidence import envelopes as _envelopes

#: Adapter/SDK identity stamped on envelopes that do not declare one
#: (V3-12.01: adapter version is envelope metadata).
SDK_VERSION = "verbatim-sdk/3.0"


def _kind(value: "EnvelopeKind | str") -> EnvelopeKind:
    try:
        return value if isinstance(value, EnvelopeKind) else EnvelopeKind(value)
    except ValueError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown envelope kind {value!r}"
        ) from exc


def _trust(value: "TrustClass | str") -> TrustClass:
    try:
        return value if isinstance(value, TrustClass) else TrustClass(value)
    except ValueError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown trust class {value!r}"
        ) from exc


def _bytes(content: "bytes | str | bytearray | memoryview") -> bytes:
    if isinstance(content, str):
        return content.encode("utf-8")
    if isinstance(content, (bytes, bytearray, memoryview)):
        return bytes(content)
    raise VerbatimError(
        ErrorCode.VALIDATION, "content must be str or bytes-like"
    )


def environment_from(data: "EnvironmentFingerprint | dict | None") -> Optional[EnvironmentFingerprint]:
    """Normalize an environment fingerprint (§20–§22)."""
    if data is None or isinstance(data, EnvironmentFingerprint):
        return data
    if not isinstance(data, dict):
        raise VerbatimError(
            ErrorCode.VALIDATION, "environment must be a mapping"
        )
    rv = data.get("runtime_versions") or ()
    tv = data.get("tool_schema_versions") or ()
    return EnvironmentFingerprint(
        repo_id=data.get("repo_id"),
        repo_revision=data.get("repo_revision"),
        runtime_versions=tuple(tuple(pair) for pair in rv),
        tool_schema_versions=tuple(tuple(pair) for pair in tv),
        platform=data.get("platform"),
    )


def perspective_from(data: "Perspective | dict | None") -> Perspective:
    """Normalize the explicit perspective roles (§04.09)."""
    if data is None:
        return Perspective()
    if isinstance(data, Perspective):
        return data
    if not isinstance(data, dict):
        raise VerbatimError(
            ErrorCode.VALIDATION, "perspective must be a mapping"
        )
    return Perspective(
        asserter=data.get("asserter"),
        subjects=tuple(data.get("subjects") or ()),
        observer=data.get("observer"),
        audience=tuple(data.get("audience") or ()),
    )


def checker_from(data: "CheckerReceipt | dict | None") -> Optional[CheckerReceipt]:
    """Normalize a checker receipt (§22.16)."""
    if data is None or isinstance(data, CheckerReceipt):
        return data
    if not isinstance(data, dict):
        raise VerbatimError(
            ErrorCode.VALIDATION, "checker receipt must be a mapping"
        )
    return CheckerReceipt(
        checker_id=require_id(str(data.get("checker_id") or ""), "checker_id"),
        checker_version=str(data.get("checker_version") or ""),
        repo_revision=data.get("repo_revision"),
        tree_digest=data.get("tree_digest"),
        invocation_id=str(data.get("invocation_id") or ""),
        selected_tests=tuple(data.get("selected_tests") or ()),
        completed=bool(data.get("completed", True)),
        exit_code=data.get("exit_code"),
        result_json=dict(data.get("result_json") or {}),
        host_attested=bool(data.get("host_attested", False)),
    )


def checker_dict(checker: Optional[CheckerReceipt]) -> Optional[dict[str, Any]]:
    """Serialize a CheckerReceipt for envelope metadata (the shape
    ``experience.episodes_v3.checker_receipt`` reads back)."""
    if checker is None:
        return None
    return {
        "checker_id": checker.checker_id,
        "checker_version": checker.checker_version,
        "repo_revision": checker.repo_revision,
        "tree_digest": checker.tree_digest,
        "invocation_id": checker.invocation_id,
        "selected_tests": list(checker.selected_tests),
        "completed": checker.completed,
        "exit_code": checker.exit_code,
        "result_json": dict(checker.result_json),
        "host_attested": checker.host_attested,
    }


def step_from(data: "TrajectoryStep | dict") -> TrajectoryStep:
    """Normalize a trajectory step (§12.02).

    Dict keys: ``step_id``, ``trajectory_id``, ``ord``,
    ``action_envelope_id``, ``observation_envelope_ids``,
    ``state_delta_refs``, ``environment`` (mapping → fingerprint).
    """
    if isinstance(data, TrajectoryStep):
        return data
    if not isinstance(data, dict):
        raise VerbatimError(
            ErrorCode.VALIDATION, "step must be a TrajectoryStep or mapping"
        )
    return TrajectoryStep(
        step_id=str(data.get("step_id") or f"step:{new_id()}"),
        trajectory_id=str(data.get("trajectory_id") or ""),
        ord=int(data.get("ord") or 0),
        action_envelope_id=data.get("action_envelope_id"),
        observation_envelope_ids=tuple(
            data.get("observation_envelope_ids") or ()
        ),
        state_delta_refs=tuple(data.get("state_delta_refs") or ()),
        environment=environment_from(data.get("environment")),
    )


def outcome_from(data: "OutcomeEnvelope | dict") -> OutcomeEnvelope:
    """Normalize an outcome record (§12.03).

    Dict keys: ``outcome`` (success|failure|partial|unknown), ``checker``
    (CheckerReceipt or mapping), ``evidence_refs``, ``recorded_us``.
    """
    if isinstance(data, OutcomeEnvelope):
        return data
    if not isinstance(data, dict):
        raise VerbatimError(
            ErrorCode.VALIDATION, "outcome must be an OutcomeEnvelope or mapping"
        )
    raw = data.get("outcome", OutcomeClass.UNKNOWN)
    try:
        outcome = raw if isinstance(raw, OutcomeClass) else OutcomeClass(raw)
    except ValueError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown outcome class {raw!r}"
        ) from exc
    return OutcomeEnvelope(
        outcome=outcome,
        checker=checker_from(data.get("checker")),
        evidence_refs=tuple(data.get("evidence_refs") or ()),
        recorded_us=int(data.get("recorded_us") or 0),
    )


class EnvelopeBuilder:
    """Fluent builder for :class:`SourceEnvelopeV3` (§12.01).

    Usage::

        env = (EnvelopeBuilder()
               .kind("tool_result").scope(sid).actor("agent-1")
               .content(payload).host("hermes").session(sess)
               .trust("host_observed").capture_proof(auth_id)
               .build())

    ``build()`` applies both the frozen dataclass contract and the
    evidence-plane ``_validate`` (id fields, redaction states, event/receipt
    clocks, JSON-serializable metadata) — the same gate ``ingest_envelope``
    enforces, so a builder-produced envelope never fails ingest on shape.
    """

    def __init__(self) -> None:
        self._kind: Optional[EnvelopeKind] = None
        self._scope_id: Optional[str] = None
        self._actor: Optional[str] = None
        self._perspective: Perspective = Perspective()
        self._event_us: int = 0
        self._receipt_us: int = 0
        self._content: Optional[bytes] = None
        self._artifact_ref: Optional[str] = None
        self._media_type: str = "text/plain"
        self._trust: TrustClass = TrustClass.UNKNOWN
        self._capture_proof: Optional[str] = None
        self._redaction_status: str = "none"
        self._adapter_version: str = SDK_VERSION
        self._host_id: str = ""
        self._session_id: str = ""
        self._task_id: str = ""
        self._step_id: str = ""
        self._external_id: Optional[str] = None
        self._metadata: dict[str, Any] = {}

    # -- identity ------------------------------------------------------------
    def kind(self, kind: "EnvelopeKind | str") -> "EnvelopeBuilder":
        self._kind = _kind(kind)
        return self

    def scope(self, scope_id: str) -> "EnvelopeBuilder":
        self._scope_id = require_id(scope_id, "scope_id")
        return self

    def actor(self, principal_id: str) -> "EnvelopeBuilder":
        self._actor = require_id(principal_id, "actor_principal")
        return self

    def perspective(self, value: "Perspective | dict") -> "EnvelopeBuilder":
        self._perspective = perspective_from(value)
        return self

    # -- clocks (event time is host-supplied; receipt time is the engine's) --
    def event_us(self, ts: int) -> "EnvelopeBuilder":
        if isinstance(ts, bool) or not isinstance(ts, int) or ts < 0:
            raise VerbatimError(ErrorCode.VALIDATION, "event_us must be an int >= 0")
        self._event_us = ts
        return self

    def receipt_us(self, ts: int) -> "EnvelopeBuilder":
        if isinstance(ts, bool) or not isinstance(ts, int) or ts < 0:
            raise VerbatimError(ErrorCode.VALIDATION, "receipt_us must be an int >= 0")
        self._receipt_us = ts
        return self

    # -- payload (exactly one of content / artifact_ref, §12.06) -------------
    def content(self, data: "bytes | str", media_type: Optional[str] = None) -> "EnvelopeBuilder":
        self._content = _bytes(data)
        if media_type is not None:
            self._media_type = media_type
        if self._artifact_ref is not None:
            self._artifact_ref = None
        return self

    def artifact(self, ref: str, media_type: Optional[str] = None) -> "EnvelopeBuilder":
        self._artifact_ref = require_id(ref, "artifact_ref")
        if media_type is not None:
            self._media_type = media_type
        self._content = None
        return self

    def media_type(self, value: str) -> "EnvelopeBuilder":
        if not isinstance(value, str) or not value:
            raise VerbatimError(ErrorCode.VALIDATION, "media_type must be a string")
        self._media_type = value
        return self

    # -- provenance / consent -------------------------------------------------
    def trust(self, trust_class: "TrustClass | str") -> "EnvelopeBuilder":
        self._trust = _trust(trust_class)
        return self

    def capture_proof(self, authorization_id: Optional[str]) -> "EnvelopeBuilder":
        self._capture_proof = (
            require_id(authorization_id, "capture_proof")
            if authorization_id is not None
            else None
        )
        return self

    def redaction(self, status: str) -> "EnvelopeBuilder":
        if status not in ("none", "applied", "failed_closed"):
            raise VerbatimError(
                ErrorCode.VALIDATION, f"invalid redaction_status {status!r}"
            )
        self._redaction_status = status
        return self

    # -- host identifiers ------------------------------------------------------
    def host(self, host_id: str) -> "EnvelopeBuilder":
        self._host_id = str(host_id or "")
        return self

    def session(self, session_id: str) -> "EnvelopeBuilder":
        self._session_id = str(session_id or "")
        return self

    def task(self, task_id: str) -> "EnvelopeBuilder":
        self._task_id = str(task_id or "")
        return self

    def step(self, step_id: str) -> "EnvelopeBuilder":
        self._step_id = str(step_id or "")
        return self

    def external_id(self, value: Optional[str]) -> "EnvelopeBuilder":
        self._external_id = value
        return self

    def adapter_version(self, value: str) -> "EnvelopeBuilder":
        self._adapter_version = str(value or "")
        return self

    # -- metadata ---------------------------------------------------------------
    def meta(self, key: str, value: Any) -> "EnvelopeBuilder":
        if not isinstance(key, str) or not key:
            raise VerbatimError(ErrorCode.VALIDATION, "metadata key must be a string")
        self._metadata[key] = value
        return self

    def metadata(self, **entries: Any) -> "EnvelopeBuilder":
        for key, value in entries.items():
            self.meta(key, value)
        return self

    # -- terminal ---------------------------------------------------------------
    def build(self) -> SourceEnvelopeV3:
        """Construct and validate the frozen envelope contract."""
        if self._kind is None:
            raise VerbatimError(ErrorCode.VALIDATION, "envelope kind is required")
        if self._scope_id is None:
            raise VerbatimError(ErrorCode.VALIDATION, "scope_id is required")
        if self._actor is None:
            raise VerbatimError(ErrorCode.VALIDATION, "actor_principal is required")
        env = SourceEnvelopeV3(
            kind=self._kind,
            scope_id=self._scope_id,
            actor_principal=self._actor,
            perspective=self._perspective,
            event_us=self._event_us,
            receipt_us=self._receipt_us,
            content=self._content,
            artifact_ref=self._artifact_ref,
            media_type=self._media_type,
            trust_class=self._trust,
            capture_proof=self._capture_proof,
            redaction_status=self._redaction_status,
            adapter_version=self._adapter_version,
            host_id=self._host_id,
            session_id=self._session_id,
            task_id=self._task_id,
            step_id=self._step_id,
            external_id=self._external_id,
            metadata=dict(self._metadata),
        )
        # Reuse the evidence plane's own ingest-time validation so a builder
        # envelope is never a shape ingest would reject.
        _envelopes._validate(env)
        return env

    def to_dict(self) -> dict[str, Any]:
        """The language-neutral event form of the built envelope (§13.02)."""
        env = self.build()
        return {
            "kind": env.kind.value,
            "scope_id": env.scope_id,
            "actor_principal": env.actor_principal,
            "perspective": {
                "asserter": env.perspective.asserter,
                "subjects": list(env.perspective.subjects),
                "observer": env.perspective.observer,
                "audience": list(env.perspective.audience),
            },
            "event_us": env.event_us,
            "receipt_us": env.receipt_us,
            "content_hex": None if env.content is None else env.content.hex(),
            "artifact_ref": env.artifact_ref,
            "media_type": env.media_type,
            "trust_class": env.trust_class.value,
            "capture_proof": env.capture_proof,
            "redaction_status": env.redaction_status,
            "adapter_version": env.adapter_version,
            "host_id": env.host_id,
            "session_id": env.session_id,
            "task_id": env.task_id,
            "step_id": env.step_id,
            "external_id": env.external_id,
            "metadata": dict(env.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvelopeBuilder":
        """Hydrate a builder from the ``to_dict`` event shape (§13.02)."""
        if not isinstance(data, dict):
            raise VerbatimError(ErrorCode.VALIDATION, "envelope event must be a mapping")
        b = cls()
        if data.get("kind") is not None:
            b.kind(data["kind"])
        if data.get("scope_id") is not None:
            b.scope(data["scope_id"])
        if data.get("actor_principal") is not None:
            b.actor(data["actor_principal"])
        if data.get("perspective") is not None:
            b.perspective(data["perspective"])
        b.event_us(int(data.get("event_us") or 0))
        b.receipt_us(int(data.get("receipt_us") or 0))
        if data.get("content") is not None:
            b.content(data["content"])
        elif data.get("content_hex") is not None:
            b.content(bytes.fromhex(data["content_hex"]))
        if data.get("artifact_ref") is not None:
            b.artifact(data["artifact_ref"])
        if data.get("media_type") is not None:
            b.media_type(data["media_type"])
        if data.get("trust_class") is not None:
            b.trust(data["trust_class"])
        if data.get("capture_proof") is not None:
            b.capture_proof(data["capture_proof"])
        if data.get("redaction_status") is not None:
            b.redaction(data["redaction_status"])
        for name in ("host_id", "session_id", "task_id", "step_id"):
            if data.get(name) is not None:
                getattr(b, {"host_id": "host", "session_id": "session",
                            "task_id": "task", "step_id": "step"}[name])(data[name])
        if data.get("external_id") is not None:
            b.external_id(data["external_id"])
        if data.get("adapter_version") is not None:
            b.adapter_version(data["adapter_version"])
        for key, value in dict(data.get("metadata") or {}).items():
            b.meta(key, value)
        return b


def outcome_descriptor(
    outcome: OutcomeEnvelope,
    *,
    extra_checkers: Iterable[CheckerReceipt] = (),
) -> dict[str, Any]:
    """Metadata/payload descriptor for a persisted outcome envelope.

    The shape mirrors what ``experience.episodes_v3`` reads: an identified
    ``checker_receipt`` plus the declared ``outcome`` (§12.03 — only an
    identified checker moves an episode outcome off ``unknown``).
    """
    checkers = [c for c in ([outcome.checker, *extra_checkers]) if c is not None]
    return {
        "outcome": outcome.outcome.value,
        "checker_receipt": checker_dict(checkers[0]) if checkers else None,
        "checker_receipts": [
            d for d in (checker_dict(c) for c in checkers) if d is not None
        ],
        "evidence_refs": list(outcome.evidence_refs),
        "recorded_us": outcome.recorded_us,
    }


__all__ = [
    "SDK_VERSION",
    "EnvelopeBuilder",
    "checker_from",
    "checker_dict",
    "environment_from",
    "outcome_descriptor",
    "outcome_from",
    "perspective_from",
    "step_from",
]
