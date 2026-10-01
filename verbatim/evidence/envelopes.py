"""V3 envelope ingest — the evidence-plane write path (SPEC_V3 §11.11, §12,
§13.11–§13.13, §14, §46).

One ``SourceEnvelopeV3`` commits as: ``scopes`` ensure → ``sources`` +
``source_revisions`` (v2-compatible rows, coarse storage class) → one
whole-payload ``spans`` row → ``source_envelopes`` (full v3 metadata) →
optional screening label → ``envelope_captured`` event → ``CaptureReceipt``.
Everything runs inside the caller's transaction ``conn`` — a failure leaves
no partial rows (V3-21, V3-46).

Design decisions fixed here (v3 worker contracts, "envelope-ingest"):

- The ``sources.source_kind`` carries a COARSE v2 storage class so v2 read
  paths stay valid (V3-62); ``source_envelopes.envelope_kind`` is the
  authoritative v3 kind.
- ``sources.scope_id`` is the envelope's own ``scope_id`` — the v3 scope is
  the authorization partition and the source row must live in it for purge
  closure to stay total (V3-39.02).
- ``event_us`` is the host-supplied source clock; ``receipt_us`` /
  ``captured_us`` is the engine's receipt time — they stay distinct
  (V3-12.11).
- Idempotence follows V3-12.04: the dedup identity is the host
  ``external_id`` when present, else a synthesized ``v3seq:`` key over the
  host identifiers + content, so retries mint one logical source.
- Agent-authored kinds can never be attributed to the human principal
  (V3-13.11): ``actor_principal`` records the actual actor and
  ``principal_*`` trust claims on agent text are rejected, not relabeled.
- Capture permission (V3-11.11, V3-12.10): a passed ``CaptureAuthorization``
  is checked against kind/scope/expiry/revocation; a ``capture_proof`` must
  resolve to an effective ``capture_authorizations`` row. Agent-submitted
  kinds (``agent_note``, ``lesson``, ``tool_call``, ``tool_result``) require
  one of the two — ``CONSENT_REQUIRED`` when absent, never a silent
  downgrade of kind.
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any, Optional

from ..core.time import now_us
from ..core.types import (
    ErrorCode,
    JobKind,
    Provenance,
    SourceEnvelope,
    SourceKind,
    VerbatimError,
    json_dumps,
    new_id,
    require_id,
)
from ..core.types_v3 import (
    CaptureAuthorization,
    EnvelopeKind,
    SourceEnvelopeV3,
    TrustClass,
)
from ..storage import repos_v3
from ..storage.repos import EventsRepo, SourcesRepo, SpansRepo
from . import detect
from .receipts import ENVELOPE_EVENT_KIND, CaptureReceipt, mint_receipt, receipt_for_envelope

# EnvelopeV3 is the contract name for SourceEnvelopeV3 (docs/v3_contracts.md).
EnvelopeV3 = SourceEnvelopeV3

_ENVELOPE_POLICY = "capture_v3_envelope_v1"
_SCHEMA_KIND_VALUES = frozenset(k.value for k in EnvelopeKind)

# --- v3 kind → coarse v2 storage class (authoritative mapping) --------------
_V2_KIND: dict[EnvelopeKind, SourceKind] = {
    EnvelopeKind.USER_MESSAGE: SourceKind.USER_MESSAGE,
    EnvelopeKind.ASSISTANT_MESSAGE: SourceKind.ASSISTANT_MESSAGE,
    EnvelopeKind.PLAN: SourceKind.ASSISTANT_MESSAGE,
    EnvelopeKind.SUBGOAL: SourceKind.ASSISTANT_MESSAGE,
    EnvelopeKind.DECISION: SourceKind.ASSISTANT_MESSAGE,
    EnvelopeKind.AGENT_NOTE: SourceKind.ASSISTANT_MESSAGE,
    EnvelopeKind.LESSON: SourceKind.ASSISTANT_MESSAGE,
    EnvelopeKind.TOOL_CALL: SourceKind.TOOL_OUTPUT,
    EnvelopeKind.TOOL_RESULT: SourceKind.TOOL_OUTPUT,
    EnvelopeKind.FILE_DIFF: SourceKind.TOOL_OUTPUT,
    EnvelopeKind.FILE_SNAPSHOT_REF: SourceKind.TOOL_OUTPUT,
    EnvelopeKind.TEST_RESULT: SourceKind.TOOL_OUTPUT,
    EnvelopeKind.VERIFICATION: SourceKind.TOOL_OUTPUT,
    EnvelopeKind.BROWSER_STATE: SourceKind.TOOL_OUTPUT,
    EnvelopeKind.SCREENSHOT_REF: SourceKind.TOOL_OUTPUT,
    EnvelopeKind.ERROR: SourceKind.TOOL_OUTPUT,
    EnvelopeKind.RECOVERY: SourceKind.TOOL_OUTPUT,
    EnvelopeKind.DOCUMENT: SourceKind.IMPORT,
    EnvelopeKind.IMPORT: SourceKind.IMPORT,
    EnvelopeKind.CONNECTOR_ITEM: SourceKind.IMPORT,
    EnvelopeKind.HANDOFF: SourceKind.IMPORT,
    EnvelopeKind.DELEGATION: SourceKind.IMPORT,
    EnvelopeKind.SYSTEM_EVENT: SourceKind.OPERATOR_RECORD,
}

# Kinds an agent may submit directly — durable capture of these REQUIRES a
# capture_proof or explicit authorization (V3-11.11, V3-12.10, §13.12).
_AGENT_SUBMITTED = frozenset({
    EnvelopeKind.AGENT_NOTE,
    EnvelopeKind.LESSON,
    EnvelopeKind.TOOL_CALL,
    EnvelopeKind.TOOL_RESULT,
})

# Kinds whose payload text is authored by an agent — never attributable to
# the human principal regardless of caller attestation (V3-13.11).
_AGENT_AUTHORED = frozenset({
    EnvelopeKind.ASSISTANT_MESSAGE,
    EnvelopeKind.PLAN,
    EnvelopeKind.SUBGOAL,
    EnvelopeKind.DECISION,
    EnvelopeKind.AGENT_NOTE,
    EnvelopeKind.LESSON,
    EnvelopeKind.HANDOFF,
    EnvelopeKind.DELEGATION,
})

# Host-observed structured kinds — harvested into `host_observed` state-fact
# claims rather than durable personal facts (V3-15.13). The harvest worker
# routes them through harvest_v3's structure-aware segmenter and marks the
# resulting claims `volatile` (§15 table).
_STRUCTURAL_HARVEST = frozenset({
    EnvelopeKind.TOOL_CALL,
    EnvelopeKind.TOOL_RESULT,
    EnvelopeKind.FILE_DIFF,
    EnvelopeKind.TEST_RESULT,
    EnvelopeKind.VERIFICATION,
    EnvelopeKind.ERROR,
    EnvelopeKind.RECOVERY,
})

# Default trust class per kind when the caller attests none (§14 table).
_DEFAULT_TRUST: dict[EnvelopeKind, TrustClass] = {
    EnvelopeKind.USER_MESSAGE: TrustClass.PRINCIPAL_DIRECT,
    EnvelopeKind.DOCUMENT: TrustClass.EXTERNAL_CONTENT,
    EnvelopeKind.CONNECTOR_ITEM: TrustClass.EXTERNAL_CONTENT,
    EnvelopeKind.IMPORT: TrustClass.IMPORTED,
}

# v2 provenance for the source_revisions row, keyed by the coarse v2 kind.
_V2_PROVENANCE: dict[SourceKind, Provenance] = {
    SourceKind.USER_MESSAGE: Provenance.DIRECT_USER,
    SourceKind.ASSISTANT_MESSAGE: Provenance.ASSISTANT_GENERATED,
    SourceKind.TOOL_OUTPUT: Provenance.APPROVED_TOOL,
    SourceKind.IMPORT: Provenance.LEGACY_IMPORT,
    SourceKind.OPERATOR_RECORD: Provenance.OPERATOR,
}

_REDACTION_STATES = frozenset({"none", "applied", "failed_closed"})


def _default_trust(kind: EnvelopeKind) -> TrustClass:
    if kind in _DEFAULT_TRUST:
        return _DEFAULT_TRUST[kind]
    if kind in _AGENT_AUTHORED:
        return TrustClass.AGENT_GENERATED
    return TrustClass.HOST_OBSERVED


def _effective_trust(envelope: SourceEnvelopeV3) -> TrustClass:
    """Resolve the persisted trust class (§14).

    Caller-attested values are honored EXCEPT ``principal_*`` on agent-
    authored kinds — model-submitted text is never human testimony
    (V3-13.11); attempting that attribution is a typed error, not a
    relabeling.
    """
    attested = envelope.trust_class
    if envelope.kind in _AGENT_AUTHORED and attested in (
        TrustClass.PRINCIPAL_DIRECT,
        TrustClass.PRINCIPAL_REPORTED,
    ):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"{envelope.kind.value} cannot claim {attested.value} trust — "
            "agent-authored content is never attributed to the principal",
        )
    if attested != TrustClass.UNKNOWN:
        return attested
    return _default_trust(envelope.kind)


def _consent(msg: str) -> VerbatimError:
    return VerbatimError(ErrorCode.CONSENT_REQUIRED, msg)


def _check_authorization(
    conn: sqlite3.Connection,
    envelope: SourceEnvelopeV3,
    authorization: CaptureAuthorization,
) -> str:
    """Validate an explicit CaptureAuthorization (§11.11); returns the
    policy revision for the event record."""
    if (
        envelope.capture_proof
        and envelope.capture_proof != authorization.authorization_id
    ):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "capture_proof binds a different authorization_id",
        )
    if envelope.kind not in authorization.allowed_kinds:
        raise _consent(
            f"authorization does not cover envelope kind {envelope.kind.value!r}"
        )
    if authorization.scope_ids and envelope.scope_id not in authorization.scope_ids:
        raise _consent("authorization does not cover this scope")
    reference = envelope.receipt_us or now_us()
    if authorization.expires_us is not None and authorization.expires_us <= reference:
        raise _consent("capture authorization expired")
    row = repos_v3.get(
        conn,
        "capture_authorizations",
        {"authorization_id": authorization.authorization_id},
    )
    if row is not None and row["revoked_us"] is not None:
        raise _consent("capture authorization revoked")
    return authorization.policy_revision or _ENVELOPE_POLICY


def _check_capture_proof(conn: sqlite3.Connection, envelope: SourceEnvelopeV3) -> str:
    """Resolve ``envelope.capture_proof`` to an effective
    ``capture_authorizations`` row and check kind/scope/expiry/revocation
    (§11.11, §12.10); returns the recorded policy revision."""
    row = repos_v3.get(
        conn,
        "capture_authorizations",
        {"authorization_id": envelope.capture_proof},
    )
    if row is None:
        raise _consent("capture_proof does not resolve to a capture authorization")
    if row["revoked_us"] is not None:
        raise _consent("capture authorization revoked")
    reference = envelope.receipt_us or now_us()
    expires = row["expires_us"]
    if expires is not None and int(expires) <= reference:
        raise _consent("capture authorization expired")
    allowed = repos_v3.json_field(row, "allowed_kinds_json", []) or []
    if envelope.kind.value not in allowed:
        raise _consent(
            f"authorization does not cover envelope kind {envelope.kind.value!r}"
        )
    scopes = repos_v3.json_field(row, "scope_ids_json", []) or []
    if scopes and envelope.scope_id not in scopes:
        raise _consent("authorization does not cover this scope")
    return row["policy_revision"] or _ENVELOPE_POLICY


def _check_capture_permission(
    conn: sqlite3.Connection,
    envelope: SourceEnvelopeV3,
    authorization: Optional[CaptureAuthorization],
) -> str:
    """The §11.11/§12.10 gate; returns the policy revision to journal."""
    if authorization is not None:
        return _check_authorization(conn, envelope, authorization)
    if envelope.capture_proof:
        return _check_capture_proof(conn, envelope)
    if envelope.kind in _AGENT_SUBMITTED:
        raise _consent(
            f"envelope kind {envelope.kind.value!r} requires a capture_proof "
            "or an explicit capture authorization — tool permission is not "
            "retention consent (§11.11)"
        )
    # Host-produced kinds: capture gating (capture.enabled, tool_observations,
    # trajectory toggles) is the adapter's contract (§13.03); the engine-level
    # write path records what the trusted host attested.
    return _ENVELOPE_POLICY


def ensure_scope_row(
    conn: sqlite3.Connection,
    scope_id: str,
    *,
    principal_id: Optional[str] = None,
    profile_id: str = "v3",
) -> str:
    """Upsert a minimal ``scopes`` row for a v3 partition id.

    v3 scope ids are opaque partition tokens (unlike the v2 ``scope_key``
    digest of a tuple); the row exists so ``sources.scope_id`` and friends
    keep their FK and purge closure stays partitioned. A pre-provisioned
    row (host setup) is never rewritten — ``INSERT OR IGNORE``.
    """
    require_id(scope_id, "scope_id")
    conn.execute(
        "INSERT OR IGNORE INTO scopes"
        " (scope_id, profile_id, principal_id, visibility)"
        " VALUES (?, ?, ?, 'owner')",
        (scope_id, profile_id, principal_id),
    )
    return scope_id


def _payload_bytes(envelope: SourceEnvelopeV3) -> bytes:
    """Accepted bytes: inline content, or the artifact reference descriptor
    for content-addressed payloads (V3-12.06 — the stored bytes are the
    reference manifest, never a silent truncation).

    Inline content must be UTF-8-decodable: the whole-payload span contract
    cites exact UTF-8 byte ranges. Binary payloads travel as
    ``artifact_ref`` instead of inline bytes."""
    if envelope.content is not None:
        payload = bytes(envelope.content)
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "inline envelope content must be UTF-8; binary payloads "
                "use artifact_ref",
            ) from exc
        return payload
    return json_dumps(
        {"artifact_ref": envelope.artifact_ref, "media_type": envelope.media_type}
    ).encode("utf-8")


def _dedup_external_id(envelope: SourceEnvelopeV3, payload: bytes) -> str:
    """Effective sequence key (V3-12.04): the host external_id when given,
    else a synthesized key over host identifiers + content so retries and
    final-session capture produce one logical source."""
    if envelope.external_id:
        return envelope.external_id
    digest = hashlib.sha256(
        b"\x00".join(
            [
                envelope.scope_id.encode("utf-8"),
                envelope.kind.value.encode("utf-8"),
                envelope.host_id.encode("utf-8"),
                envelope.session_id.encode("utf-8"),
                envelope.task_id.encode("utf-8"),
                envelope.step_id.encode("utf-8"),
                str(envelope.event_us).encode("utf-8"),
                payload,
            ]
        )
    ).hexdigest()
    return f"v3seq:{digest[:32]}"


def _envelope_id_for(source_id: str, revision: int, kind: EnvelopeKind) -> str:
    digest = hashlib.sha256(
        f"{source_id}\x00{revision}\x00{kind.value}".encode("utf-8")
    ).hexdigest()
    return f"env_{digest[:32]}"


def _persist_perspective(conn: sqlite3.Connection, envelope: SourceEnvelopeV3) -> Optional[str]:
    """Persist the explicit perspective (§04.09): deterministic id over the
    role tuple; ``None`` when no role was recorded — never widened."""
    p = envelope.perspective
    if p is None or (
        p.asserter is None and p.observer is None and not p.subjects and not p.audience
    ):
        return None
    material = json_dumps(
        {
            "scope_id": envelope.scope_id,
            "asserter": p.asserter,
            "observer": p.observer,
            "subjects": sorted(p.subjects),
            "audience": sorted(p.audience),
        }
    )
    pid = "ps_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
    exists = conn.execute(
        "SELECT 1 FROM perspectives WHERE perspective_id = ?", (pid,)
    ).fetchone()
    if exists is None:
        repos_v3.insert(
            conn,
            "perspectives",
            {
                "perspective_id": pid,
                "scope_id": envelope.scope_id,
                "asserter": p.asserter,
                "observer": p.observer,
                "audience_json": list(p.audience),
            },
        )
        for subject in sorted(set(p.subjects)):
            repos_v3.insert(
                conn,
                "perspective_subjects",
                {"perspective_id": pid, "subject_id": subject},
            )
    return pid


def _maybe_screen(
    conn: sqlite3.Connection,
    envelope: SourceEnvelopeV3,
    trust: TrustClass,
    envelope_id: Optional[str] = None,
    revision: int = 1,
) -> Optional[str]:
    """Write-channel screening (§34.01): when the security module is
    provisioned, screen the textual payload inside the same tx — attach
    the label and open a quarantine hold on ``suspicious``/``blocked``
    findings (mirroring ``handle_screen`` semantics). Non-textual payloads
    attach an ``unassessed`` label. Absent the module, the envelope's
    declared trust class stands on its own — default trust labeling,
    never a fabricated screening result."""
    try:
        from ..security import (  # type: ignore
            attach_label,
            default_review_state,
            open_quarantine,
            screen_content,
        )
    except ImportError:
        return None
    text: Optional[str] = None
    if envelope.content is not None:
        try:
            text = bytes(envelope.content).decode("utf-8")
        except (TypeError, UnicodeDecodeError):
            text = None
    if text is None:
        return attach_label(
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
    verdict = screen_content(
        text, source_trust=trust.value, context_kind=envelope.kind.value
    )
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
    if verdict.attack_risk.value in ("suspicious", "blocked") and envelope_id:
        reasons = [f"attack_risk:{verdict.attack_risk.value}"] + sorted(
            {
                str(f.get("rule_id"))
                for f in verdict.findings
                if f.get("rule_id")
            }
        )
        open_quarantine(
            conn,
            ("source_envelope", envelope_id, revision),
            reasons or [f"attack_risk:{verdict.attack_risk.value}"],
            list(verdict.findings),
            scope_id=envelope.scope_id,
        )
    return label_id


def _link_label(
    conn: sqlite3.Connection, envelope_id: str, label_id: Optional[str]
) -> None:
    """Record the screening label on the envelope row's metadata so
    ``inspect_evidence``/``quarantine_review`` resolve it without a table
    scan (§47.02). The label row deliberately has no back-pointer, so the
    envelope carries the link — written here, once, on the canonical write
    path rather than left to each caller."""
    if not label_id:
        return
    env_row = repos_v3.get(
        conn, "source_envelopes", {"envelope_id": envelope_id}
    )
    if env_row is None:
        return
    meta = repos_v3.json_field(env_row, "metadata_json", {}) or {}
    meta["security_label_id"] = label_id
    repos_v3.update(
        conn,
        "source_envelopes",
        {"metadata_json": meta},
        {"envelope_id": envelope_id},
    )


def _enqueue_harvest(
    conn: sqlite3.Connection,
    store: Any,
    envelope: SourceEnvelopeV3,
    source_id: str,
    revision: int,
) -> None:
    """Queue the harvest → admit → claim obligation for harvest-eligible
    kinds (§15), inside the caller's tx so the obligation is atomic with
    the capture.

    Only kinds whose coarse v2 storage class is harvestable under the
    default admission policy reach claim derivation, and agent-authored
    kinds are excluded outright — agent text never launders into
    user-style claims through the storage-class mapping (V3-13.11). The
    dedup/operation keys mirror the v2 ingest path so redelivery replays
    the committed receipt instead of double-harvesting (V2-39.02/39.10).
    """
    if envelope.kind in _AGENT_AUTHORED:
        return
    if envelope.kind not in _STRUCTURAL_HARVEST:
        try:
            from ..core.harvest import _DEFAULT_ALLOW_KINDS  # type: ignore
        except ImportError:
            return
        if _V2_KIND[envelope.kind] not in _DEFAULT_ALLOW_KINDS:
            return
    from ..jobs.queue import JobQueue  # local import: evidence → jobs edge

    jobs = JobQueue(store)
    dedup = store.hmac(f"harvest:{source_id}:{revision}".encode())
    jobs.enqueue(
        conn,
        envelope.scope_id,
        JobKind.HARVEST,
        {"source_id": source_id, "revision": revision},
        dedup_key=dedup,
        operation_key=(
            f"harvest:{source_id}:{revision}"
            if jobs.supports_durability
            else None
        ),
    )


def _record_event(
    conn: sqlite3.Connection,
    store: Any,
    envelope: SourceEnvelopeV3,
    payload: dict[str, Any],
    policy_revision: str,
) -> int:
    """Append the ``envelope_captured`` journal row; returns its committed
    sequence (the receipt's event anchor, §47.03)."""
    return EventsRepo(store).append(
        conn,
        envelope.scope_id,
        ENVELOPE_EVENT_KIND,
        envelope.actor_principal,
        payload,
        policy_revision or _ENVELOPE_POLICY,
    )


def _insert_envelope_row(
    conn: sqlite3.Connection,
    envelope: SourceEnvelopeV3,
    *,
    envelope_id: str,
    source_id: str,
    revision: int,
    perspective_id: Optional[str],
    trust: TrustClass,
    receipt_us: int,
) -> dict[str, Any]:
    row = {
        "envelope_id": envelope_id,
        "source_id": source_id,
        "revision": revision,
        "scope_id": envelope.scope_id,
        "envelope_kind": envelope.kind.value,
        "actor_principal": envelope.actor_principal,
        "perspective_id": perspective_id,
        "event_us": envelope.event_us,
        "receipt_us": receipt_us,
        "media_type": envelope.media_type,
        "trust_class": trust.value,
        "capture_proof": envelope.capture_proof,
        "redaction_status": envelope.redaction_status,
        "adapter_version": envelope.adapter_version,
        "host_id": envelope.host_id,
        "session_id": envelope.session_id,
        "task_id": envelope.task_id,
        "step_id": envelope.step_id,
        "artifact_ref": envelope.artifact_ref,
        "metadata_json": dict(envelope.metadata),
    }
    repos_v3.insert(conn, "source_envelopes", row)
    return row


def _validate(envelope: SourceEnvelopeV3) -> None:
    if not isinstance(envelope, SourceEnvelopeV3):
        raise VerbatimError(ErrorCode.VALIDATION, "ingest_envelope needs a SourceEnvelopeV3")
    require_id(envelope.scope_id, "scope_id")
    require_id(envelope.actor_principal, "actor_principal")
    if envelope.kind.value not in _SCHEMA_KIND_VALUES:
        raise VerbatimError(ErrorCode.VALIDATION, f"unknown envelope kind {envelope.kind!r}")
    if envelope.redaction_status not in _REDACTION_STATES:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"invalid redaction_status {envelope.redaction_status!r}",
        )
    if isinstance(envelope.event_us, bool) or not isinstance(envelope.event_us, int) or envelope.event_us < 0:
        raise VerbatimError(ErrorCode.VALIDATION, "event_us must be an int >= 0")
    if isinstance(envelope.receipt_us, bool) or not isinstance(envelope.receipt_us, int) or envelope.receipt_us < 0:
        raise VerbatimError(ErrorCode.VALIDATION, "receipt_us must be an int >= 0")
    if envelope.redaction_status == "failed_closed":
        # V3-12.07: a kind that could not be safely scrubbed fails closed —
        # its bytes are never persisted, not stored and marked.
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "envelope scrubbing failed closed — refusing durable plaintext",
        )
    if not isinstance(envelope.metadata, dict):
        raise VerbatimError(ErrorCode.VALIDATION, "envelope metadata must be a mapping")
    try:
        json_dumps(envelope.metadata)
    except (TypeError, ValueError) as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"envelope metadata is not json-serializable: {exc}"
        ) from exc


def ingest_envelope(
    conn: sqlite3.Connection,
    store: Any,
    envelope: SourceEnvelopeV3,
    *,
    authorization: Optional[CaptureAuthorization] = None,
) -> CaptureReceipt:
    """Persist one V3 envelope atomically; returns the durable receipt.

    Contract (docs/v3_contracts.md "Envelope-ingest interface"):

    ``sources`` + ``source_revisions`` + ``spans`` + ``source_envelopes`` +
    screening + receipt, atomically inside the caller's ``conn``. Agent
    attribution never crosses — agent-authored text is never recorded as
    human testimony (V3-13.11).
    """
    _validate(envelope)
    policy_revision = _check_capture_permission(conn, envelope, authorization)
    trust = _effective_trust(envelope)
    payload = _payload_bytes(envelope)
    # Receipt time is the engine's clock, distinct from the host's event_us
    # (V3-12.11); a store without the logical clock falls back to wall time.
    _engine_us = getattr(store, "next_event_us", None) or now_us
    receipt_us = envelope.receipt_us or _engine_us()

    # --- dedup (V3-12.04): host sequence key or synthesized key -------------
    origin = envelope.host_id or "v3"
    dedup_key = _dedup_external_id(envelope, payload)
    payload_hmac = store.hmac(payload)

    profile_hint = envelope.metadata.get("profile_id")
    ensure_scope_row(
        conn,
        envelope.scope_id,
        principal_id=envelope.actor_principal,
        profile_id=profile_hint if isinstance(profile_hint, str) and profile_hint else "v3",
    )

    source_id: Optional[str] = None
    # Partition-local dedup (V3-12.04): the same host key in a different
    # scope is an independent source — a cross-scope graft would leak the
    # second capture's bytes under the first scope's partition and make
    # a purge of either scope damage the other's evidence.
    found = conn.execute(
        "SELECT source_id FROM sources"
        " WHERE scope_id = ? AND origin = ? AND external_id = ?",
        (envelope.scope_id, origin, dedup_key),
    ).fetchone()
    if found is not None:
        source_id = found[0]

    revision = 1
    if source_id is not None:
        existing = conn.execute(
            "SELECT payload_hmac FROM source_revisions"
            " WHERE source_id = ? AND revision = ?",
            (source_id, revision),
        ).fetchone()
        if existing is not None:
            if bytes(existing[0]) != payload_hmac:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "conflicting payload for dedup key (origin, external_id, revision)",
                )
            # Idempotent replay (V3-12.04): same logical envelope — return the
            # already-committed receipt, no double-write. A *new* envelope
            # kind over the same source still records its own row
            # (UNIQUE(source_id, revision, envelope_kind)).
            env_row = repos_v3.get(
                conn,
                "source_envelopes",
                {"source_id": source_id, "revision": revision, "envelope_kind": envelope.kind.value},
            )
            if env_row is None:
                env_row = _insert_envelope_row(
                    conn,
                    envelope,
                    envelope_id=_envelope_id_for(source_id, revision, envelope.kind),
                    source_id=source_id,
                    revision=revision,
                    perspective_id=_persist_perspective(conn, envelope),
                    trust=trust,
                    receipt_us=receipt_us,
                )
                label_id = _maybe_screen(
                    conn, envelope, trust,
                    envelope_id=env_row["envelope_id"], revision=revision,
                )
                _link_label(conn, env_row["envelope_id"], label_id)
                _record_event(
                    conn, store, envelope,
                    {
                        "envelope_id": env_row["envelope_id"],
                        "source_id": source_id,
                        "revision": revision,
                        "envelope_kind": envelope.kind.value,
                        "dedup_key": dedup_key,
                        "kind_added_over_existing_source": True,
                    },
                    policy_revision,
                )
            return receipt_for_envelope(conn, env_row, dedup_key=dedup_key)
        # A new revision on a known source is an edit (v2 semantics); v3
        # envelopes are revision-1 captures — a different revision-1 payload
        # under the same key already raised above, so this path is a fresh
        # revision-bearing retry and falls through to insert.
        row = conn.execute(
            "SELECT COALESCE(MAX(revision), 0) FROM source_revisions"
            " WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        revision = int(row[0]) + 1
        SourcesRepo._insert_revision(
            conn, source_id,
            _v2_envelope(envelope, origin, dedup_key, revision, payload, receipt_us),
            payload, payload_hmac,
        )
    else:
        source_id = new_id()
        conn.execute(
            "INSERT INTO sources"
            " (source_id, origin, external_id, source_kind, scope_id,"
            "  speaker_id, created_us)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                source_id,
                origin,
                dedup_key,
                _V2_KIND[envelope.kind].value,
                envelope.scope_id,
                envelope.actor_principal,
                receipt_us,
            ),
        )
        SourcesRepo._insert_revision(
            conn, source_id,
            _v2_envelope(envelope, origin, dedup_key, revision, payload, receipt_us),
            payload, payload_hmac,
        )

    # --- whole-payload span (§12.08; v2 span conventions via SpansRepo) -----
    start, end = detect.whole_payload_span(payload)
    span_id = detect.span_id_for(source_id, revision, start, end)
    SpansRepo(store).insert(
        span_id, source_id, revision, start, end, detect.PARSER_VERSION, conn=conn
    )
    conn.execute(
        "UPDATE spans SET operation_key = ? WHERE span_id = ?",
        (f"capture:{source_id}:{revision}:{envelope.kind.value}", span_id),
    )

    # --- v3 envelope row -----------------------------------------------------
    perspective_id = _persist_perspective(conn, envelope)
    env_row = _insert_envelope_row(
        conn,
        envelope,
        envelope_id=_envelope_id_for(source_id, revision, envelope.kind),
        source_id=source_id,
        revision=revision,
        perspective_id=perspective_id,
        trust=trust,
        receipt_us=receipt_us,
    )

    # --- screening hook (§34.01 write-channel screen) ------------------------
    label_id = _maybe_screen(
        conn, envelope, trust,
        envelope_id=env_row["envelope_id"], revision=revision,
    )
    # The screening label is linked onto the envelope row here — every
    # ingest caller (facade, SDK, adapters) gets a discoverable label and
    # one quarantine ref per envelope, not a second screen of their own.
    _link_label(conn, env_row["envelope_id"], label_id)

    # --- harvest obligation (§15): claim derivation for eligible kinds ------
    _enqueue_harvest(conn, store, envelope, source_id, revision)

    # --- journal + receipt ----------------------------------------------------
    event_seq = _record_event(
        conn,
        store,
        envelope,
        {
            "envelope_id": env_row["envelope_id"],
            "source_id": source_id,
            "revision": revision,
            "envelope_kind": envelope.kind.value,
            "dedup_key": dedup_key,
            "span_ids": [span_id],
            "security_label_id": label_id,
        },
        policy_revision,
    )
    return mint_receipt(
        conn,
        env_row,
        event_seq=event_seq,
        dedup_key=dedup_key,
        span_ids=(span_id,),
        accepted_bytes=len(payload),
    )


def _v2_envelope(
    envelope: SourceEnvelopeV3,
    origin: str,
    external_id: str,
    revision: int,
    payload: bytes,
    captured_us: int,
) -> SourceEnvelope:
    """The v2 envelope shape ``SourcesRepo._insert_revision`` persists —
    event clock vs receipt clock kept distinct (V3-12.11). Scope is
    placeholder-only: the sources row is written directly with the
    envelope's own scope_id."""
    from ..core.types import Scope, Visibility

    v2_kind = _V2_KIND[envelope.kind]
    return SourceEnvelope(
        origin=origin,
        source_kind=v2_kind,
        scope=Scope(profile_id="v3", visibility=Visibility.OWNER),
        speaker_id=envelope.actor_principal,
        payload=payload,
        event_us=envelope.event_us,
        captured_us=captured_us,
        timezone=None,
        provenance=_V2_PROVENANCE[v2_kind],
        external_id=external_id,
        source_id=None,
        revision=revision,
        metadata=dict(envelope.metadata),
    )
