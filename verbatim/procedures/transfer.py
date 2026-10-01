"""Environment- and counterexample-qualified procedure transfer
(SPEC_V4_5 §06, V45-06.*; SPEC_V4 §23; SPEC_V3 §22).

The procedure *reuse surface*: delivering a compiled procedure to a host
for a new task. Positive-only reuse — shipping a procedure because it
once succeeded — is how memory harms agents in held-out environments.
``deliver_procedure`` qualifies every delivery with

1. the **environment-match verdict** from
   ``applicability.check_applicability`` — a mismatch *blocks* delivery
   outright (the host never receives the reuse card), an ``unknown``
   verdict loudly qualifies it rather than passing as a wildcard
   (V3-22.15), and
2. the procedure's **known counterexamples** — recorded failure modes
   with their evidence references, parsed verbatim from
   ``failure_modes_json`` — attached to the delivery so the host sees
   exactly where the procedure has already broken.

``mode="positive_only"`` is the comparator arm for paired evaluation
(V45-06.04, D07): it ships the same card with no qualification at all, so a
measured negative-transfer delta attributes to the qualification, not to
different procedure content.

``transfer_success`` is the honest reuse metric (V45-06.02, D08):
*delivered for the task* — a ``procedure_exposures`` row binding the
procedure to the task — **and** a *host-attested positive outcome* for
that task. Self-reports (``agent_report`` envelopes), exposure counts
alone, and hypothetical trials never count; attestation resolution is
delegated to ``episodes_v3.envelope_outcome`` so the checker-receipt
rules stay identical to episode outcome resolution (V4-23.01/02).

Everything here is advisory ordering/qualification on the read side:
no function in this module rewrites provenance, grants, checker
identity, or evidence digests.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    require_id,
    safe_json_loads,
)
from ..core.types_v3 import ApplicabilityVerdict
from ..storage.repos import _json_parse, _row, _rows
from .applicability import check_applicability
from .exposures import record_exposure

#: Policy id stamped on delivery reports and job/dedup keys so measured
#: runs name the exact qualification rules that produced them.
TRANSFER_POLICY_ID = "v45.transfer.v1"

#: Delivery modes. ``failure_aware`` is the product path; ``positive_only``
#: is the honest comparator arm — identical card, no qualification.
DELIVERY_MODES = ("failure_aware", "positive_only")

#: Dispositions reported on a delivery decision.
DISPOSITION_DELIVERED = "delivered"
DISPOSITION_QUALIFIED = "qualified"
DISPOSITION_BLOCKED = "blocked"

#: Environment-match labels carried on the delivery surface.
ENV_MATCH = "match"
ENV_MISMATCH = "mismatch"
ENV_UNKNOWN = "unknown"
ENV_UNQUALIFIED = "unqualified"


@dataclass(frozen=True)
class Counterexample:
    """One recorded failure mode — a counterexample to blind reuse.

    ``evidence_refs`` are the exact ``{kind,id,revision}`` references the
    compiler extracted from the error envelope (V4-23.03); ``environments``
    names the environments the failure was recorded under when known —
    the empty tuple means *environment unrecorded*, never "all".
    """

    signature: str
    description: str
    occurrences: int = 1
    evidence_refs: Tuple[dict[str, Any], ...] = ()
    recovery_ref: Optional[dict[str, Any]] = None
    environments: Tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "description": self.description,
            "occurrences": self.occurrences,
            "evidence_refs": [dict(r) for r in self.evidence_refs],
            "recovery_ref": (
                dict(self.recovery_ref) if self.recovery_ref else None
            ),
            "environments": list(self.environments),
        }


@dataclass(frozen=True)
class TransferDelivery:
    """The qualification verdict on one procedure delivery.

    ``deliverable`` is the gate: when False the host receives no reuse
    card — only the verdict and the counterexamples that explain it.
    ``disposition`` is ``delivered`` (clean), ``qualified`` (delivered
    but loudly qualified by an unknown verdict or attached
    counterexamples), or ``blocked`` (environment mismatch or a non-live
    procedure).
    """

    procedure_id: str
    revision: int
    mode: str
    deliverable: bool
    disposition: str
    environment_match: str
    applicability: Optional[dict[str, Any]]
    counterexamples: Tuple[Counterexample, ...]
    failure_evidence: str  # "observed" | "no_failure_evidence" | "unqualified"
    card: Optional[dict[str, Any]]
    warnings: Tuple[str, ...]
    exposure_id: Optional[str]
    policy_id: str = TRANSFER_POLICY_ID

    def as_dict(self) -> dict[str, Any]:
        return {
            "procedure_id": self.procedure_id,
            "revision": self.revision,
            "mode": self.mode,
            "deliverable": self.deliverable,
            "disposition": self.disposition,
            "environment_match": self.environment_match,
            "applicability": (
                dict(self.applicability) if self.applicability else None
            ),
            "counterexamples": [c.as_dict() for c in self.counterexamples],
            "failure_evidence": self.failure_evidence,
            "card": dict(self.card) if self.card else None,
            "warnings": list(self.warnings),
            "exposure_id": self.exposure_id,
            "policy_id": self.policy_id,
        }


def _procedure_row(
    conn: sqlite3.Connection, procedure_id: str
) -> dict[str, Any]:
    row = _row(
        conn.execute(
            "SELECT * FROM procedures WHERE procedure_id = ?",
            (require_id(procedure_id, "procedure_id"),),
        )
    )
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "procedure not found"
        )
    return row


def _normalize_evidence_ref(raw: Any) -> Optional[dict[str, Any]]:
    """Normalize one failure-mode evidence reference.

    The compiler stores ``{kind,id,revision}`` dicts (V4-23.03); bare
    strings degrade to ``{id: ...}`` with kind/revision omitted rather
    than fabricated.
    """
    if isinstance(raw, dict):
        out = {}
        if raw.get("kind") is not None:
            out["kind"] = str(raw["kind"])
        if raw.get("id") is not None:
            out["id"] = str(raw["id"])
        elif raw.get("object_id") is not None:
            out["id"] = str(raw["object_id"])
        if raw.get("revision") is not None:
            out["revision"] = int(raw["revision"])
        return out or None
    if isinstance(raw, str) and raw:
        return {"id": raw}
    return None


def _failure_modes(row: dict[str, Any]) -> list[dict[str, Any]]:
    """The compiler's failure-mode list.

    ``failure_modes_json`` is ``{"modes": [...], "status": ...}`` on
    compiler-written rows; a bare list is tolerated for hand-seeded
    fixtures. Either way only dict entries count.
    """
    raw = _json_parse(row.get("failure_modes_json"))
    if isinstance(raw, dict):
        raw = raw.get("modes") or []
    if isinstance(raw, list):
        return [m for m in raw if isinstance(m, dict)]
    return []


def _failure_status(row: dict[str, Any]) -> str:
    raw = _json_parse(row.get("failure_modes_json"))
    if isinstance(raw, dict) and isinstance(raw.get("status"), str):
        return raw["status"]
    return "observed" if _failure_modes(row) else "no_failure_evidence"


def counterexamples(
    conn: sqlite3.Connection, procedure_id: str
) -> Tuple[Counterexample, ...]:
    """The procedure's recorded counterexamples, evidence refs intact.

    Reads ``failure_modes_json`` verbatim — occurrences, descriptions,
    recovery refs, and the evidence references the compiler attached to
    each failure envelope. No failure evidence is reported as
    ``no_failure_evidence`` by callers — the absence of recorded
    counterexamples is never presumed safety (V45-06.03).
    """
    row = _procedure_row(conn, procedure_id)
    out: list[Counterexample] = []
    for mode in _failure_modes(row):
        refs = tuple(
            r
            for r in (
                _normalize_evidence_ref(e)
                for e in (mode.get("evidence_refs") or ())
            )
            if r is not None
        )
        envs = mode.get("environments") or ()
        if isinstance(envs, str):
            envs = (envs,)
        out.append(
            Counterexample(
                signature=str(mode.get("signature") or "unknown"),
                description=str(mode.get("description") or ""),
                occurrences=int(mode.get("occurrences") or 1),
                evidence_refs=refs,
                recovery_ref=_normalize_evidence_ref(
                    mode.get("recovery_ref")
                ),
                environments=tuple(str(e) for e in envs),
            )
        )
    return tuple(out)


def _reuse_card(conn: sqlite3.Connection, row: dict[str, Any]) -> dict[str, Any]:
    """The delivery card: the procedure's advisory reuse surface.

    Assembled from the persisted row + quoted steps — advisory data only,
    never authority to execute (V2-24.04/08). Provenance stays attached
    verbatim; qualification never rewrites it.
    """
    from ..experience.procedures import environment_fingerprint

    env = _json_parse(row.get("environment_json")) or {}
    steps = [
        {
            "step_no": s["step_no"],
            "description": s["description"],
            "span_id": s["span_id"],
            "precondition": _json_parse(s["precondition_json"]),
            "hazard": s["hazard"],
            "verification": s["verification"],
        }
        for s in _rows(
            conn.execute(
                "SELECT * FROM procedure_steps"
                " WHERE procedure_id = ? AND revision = ?"
                " ORDER BY step_no",
                (row["procedure_id"], int(row["revision"])),
            )
        )
    ]
    return {
        "kind": "procedure",
        "advisory": True,
        "procedure_id": row["procedure_id"],
        "revision": int(row["revision"]),
        "task_label": row["task_label"],
        "state": row["state"],
        "environment": env,
        "environment_fingerprint": environment_fingerprint(env),
        "applicability": _json_parse(row.get("applicability_json")) or [],
        "preconditions": _json_parse(row.get("preconditions_json")) or [],
        "bindings": _json_parse(row.get("bindings_json")) or [],
        "verification": _json_parse(row.get("verification_json")) or [],
        "expected_outcome": row.get("expected_outcome"),
        "hazards": _json_parse(row.get("hazards_json")) or [],
        "risk_class": row.get("risk_class") or "medium",
        "provenance": _json_parse(row.get("provenance_json")) or {},
        "steps": steps,
    }


def _environment_digest(environment: Any) -> Optional[str]:
    """Digest of the *requested* environment for per-env denominators."""
    if environment is None:
        return None
    from ..experience.procedures import environment_fingerprint
    from .applicability import _env_map_of

    return environment_fingerprint(_env_map_of(environment))


def _is_live(row: dict[str, Any]) -> bool:
    return (
        row.get("recorded_until") is None
        and row.get("state") not in ("retired", "deprecated")
    )


def deliver_procedure(
    conn: sqlite3.Connection,
    procedure_id: str,
    *,
    task_id: Optional[str] = None,
    environment: Any = None,
    bindings: Optional[dict[str, Any]] = None,
    mode: str = "failure_aware",
    record: bool = True,
    recorded_us: int = 0,
    session_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    arm: Optional[str] = None,
) -> TransferDelivery:
    """Qualify and (when permitted) deliver a procedure for reuse.

    ``mode="failure_aware"`` — the product path:

    - a non-live procedure (``retired``/``deprecated``/closed
      ``recorded_until``) blocks outright;
    - an environment/condition *mismatch* blocks — the host receives the
      verdict and counterexamples, never the card (V45-06.01);
    - an *unknown* verdict — missing environment, missing bindings,
      unevaluable conditions — delivers but loudly qualifies, never a
      wildcard pass (V3-22.15);
    - recorded counterexamples always ride along on the delivery, so
      even an applicable procedure arrives with its known failure modes
      and their evidence references (V45-06.03).

    ``mode="positive_only"`` — the paired comparator arm: the identical
    card with no environment check and no counterexample attachment
    (``environment_match`` reported ``unqualified``). Liveness still
    gates — shipping a withdrawn procedure is a bug in either arm, not
    a positive-only semantic.

    ``record=True`` appends a ``procedure_exposures`` row for the task
    when the card actually ships — ``applicable`` when the verdict was
    ``applies``, ``exposed`` otherwise. A blocked delivery records no
    exposure: the host never saw the procedure, and the denominators
    must stay honest (V3-22.06). The blocked decision itself is the
    audit trail.
    """
    if mode not in DELIVERY_MODES:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"delivery mode must be one of {DELIVERY_MODES}",
        )
    row = _procedure_row(conn, procedure_id)
    pid = row["procedure_id"]
    revision = int(row["revision"])

    if not _is_live(row):
        # Non-live rows block in both modes — a withdrawn procedure is
        # never a valid delivery (V45-06.01).
        ces = counterexamples(conn, pid)
        return TransferDelivery(
            procedure_id=pid,
            revision=revision,
            mode=mode,
            deliverable=False,
            disposition=DISPOSITION_BLOCKED,
            environment_match=ENV_UNQUALIFIED,
            applicability=None,
            counterexamples=ces,
            failure_evidence=_failure_status(row),
            card=None,
            warnings=(
                f"procedure_not_live:{row['state']}",
            ),
            exposure_id=None,
        )

    if mode == "positive_only":
        exposure_id = None
        if record:
            exposure_id = record_exposure(
                conn,
                pid,
                task_id=task_id,
                outcome="exposed",
                session_id=session_id,
                environment_digest=_environment_digest(environment),
                experiment_id=experiment_id,
                arm=arm,
                recorded_us=recorded_us,
            )
        return TransferDelivery(
            procedure_id=pid,
            revision=revision,
            mode=mode,
            deliverable=True,
            disposition=DISPOSITION_DELIVERED,
            environment_match=ENV_UNQUALIFIED,
            applicability=None,
            counterexamples=(),
            failure_evidence="unqualified",
            card=_reuse_card(conn, row),
            warnings=(),
            exposure_id=exposure_id,
        )

    # failure_aware
    result = check_applicability(
        conn, pid, environment=environment, bindings=bindings
    )
    ces = counterexamples(conn, pid)
    warnings: list[str] = []

    if result.verdict is ApplicabilityVerdict.DOES_NOT_APPLY:
        return TransferDelivery(
            procedure_id=pid,
            revision=revision,
            mode=mode,
            deliverable=False,
            disposition=DISPOSITION_BLOCKED,
            environment_match=ENV_MISMATCH,
            applicability=result.as_dict(),
            counterexamples=ces,
            failure_evidence=_failure_status(row),
            card=None,
            warnings=tuple(
                [f"environment_mismatch:{r}" for r in result.reasons]
                + [f"counterexample:{c.signature}" for c in ces]
            ),
            exposure_id=None,
        )

    env_match = (
        ENV_MATCH
        if result.verdict is ApplicabilityVerdict.APPLIES
        else ENV_UNKNOWN
    )
    if result.verdict is ApplicabilityVerdict.UNKNOWN:
        warnings.extend(
            f"applicability_unknown:{r}" for r in result.reasons
        )
    if ces:
        warnings.extend(f"counterexample:{c.signature}" for c in ces)
    disposition = (
        DISPOSITION_DELIVERED
        if not warnings
        else DISPOSITION_QUALIFIED
    )
    exposure_id = None
    if record:
        exposure_id = record_exposure(
            conn,
            pid,
            task_id=task_id,
            outcome=(
                "applicable"
                if result.verdict is ApplicabilityVerdict.APPLIES
                else "exposed"
            ),
            session_id=session_id,
            environment_digest=_environment_digest(environment),
            experiment_id=experiment_id,
            arm=arm,
            recorded_us=recorded_us,
        )
    return TransferDelivery(
        procedure_id=pid,
        revision=revision,
        mode=mode,
        deliverable=True,
        disposition=disposition,
        environment_match=env_match,
        applicability=result.as_dict(),
        counterexamples=ces,
        failure_evidence=_failure_status(row),
        card=_reuse_card(conn, row),
        warnings=tuple(warnings),
        exposure_id=exposure_id,
    )


def transfer_success(
    conn: sqlite3.Connection,
    procedure_id: str,
    task_id: str,
    *,
    scope_id: Optional[str] = None,
) -> dict[str, Any]:
    """Honest transfer success for one (procedure, task) pair (D08).

    True only when *both* hold:

    - **delivered**: a ``procedure_exposures`` row binds the procedure
      to ``task_id`` (the reuse surface actually reached the host), and
    - **attested positive**: the task's most recent outcome envelope —
      ``test_result``/``verification`` in the procedure's scope —
      resolves to ``success`` through ``episodes_v3.envelope_outcome``,
      i.e. a host-executed checker receipt with ``host_attested`` and a
      concrete ``invocation_id``, bound to this task/scope when the
      receipt pins them (V4-23.01/02). Agent reports and envelopes
      without checker receipts are *not outcome evidence* — they can
      never move the verdict off ``unknown``/``none``.

    Exposure counts, adoption, and self-declared outcomes never satisfy
    the second clause.
    """
    require_id(procedure_id, "procedure_id")
    require_id(task_id, "task_id")
    row = _procedure_row(conn, procedure_id)
    sid = scope_id or row["scope_id"]

    exposures = _rows(
        conn.execute(
            "SELECT exposure_id, outcome FROM procedure_exposures"
            " WHERE procedure_id = ? AND task_id = ?",
            (procedure_id, task_id),
        )
    )
    delivered = len(exposures)

    from ..experience.episodes_v3 import (
        OUTCOME_ENVELOPE_KINDS,
        envelope_outcome,
    )

    envelopes = _rows(
        conn.execute(
            "SELECT metadata_json FROM source_envelopes"
            " WHERE scope_id = ? AND task_id = ?"
            " AND envelope_kind IN ('test_result','verification')"
            " ORDER BY rowid",
            (sid, task_id),
        )
    )
    attested: Optional[str] = None
    attested_envelopes = 0
    for env in envelopes:
        meta = safe_json_loads(env["metadata_json"]) or {}
        resolved = envelope_outcome(meta, task_id=task_id, scope_id=sid)
        if resolved is not None:
            attested = resolved.value
            attested_envelopes += 1

    reasons: list[str] = []
    if not delivered:
        reasons.append("no_delivery_record")
    if not envelopes:
        reasons.append("no_outcome_envelopes")
    elif attested is None:
        reasons.append("no_attested_outcome")
    elif attested != "success":
        reasons.append(f"attested_{attested}")

    success = bool(delivered) and attested == "success"
    return {
        "procedure_id": procedure_id,
        "task_id": task_id,
        "scope_id": sid,
        "delivered": delivered,
        "delivered_procedure": bool(delivered),
        "outcome_envelopes": len(envelopes),
        "attested_envelopes": attested_envelopes,
        "non_attested_envelopes": len(envelopes) - attested_envelopes,
        "attested_outcome": attested if attested is not None else "none",
        "transfer_success": success,
        "reasons": reasons,
        "policy_id": TRANSFER_POLICY_ID,
    }
