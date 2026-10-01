"""Policy-artifact registration, validation, and activation binding
(SPEC_V6 V6-03.15/16, contracts §"w6-feedback").

The ``policy_artifacts`` table (v2 schema, ``PolicyArtifactsRepo``)
carries ``validation_state ∈ {unvalidated, validated, revoked}`` but was
never wired: nothing could register, nothing could validate, and
``learned_active`` stayed fail-closed at config validation. This module
is the graduation path the spec requires:

* ``register_artifact`` — strict ``learned_policy_v1`` validation via
  ``learned.PolicyArtifact`` (the same parser the controller consumes —
  a payload the controller cannot load never enters the registry).
  Registration always lands ``unvalidated`` or ``revoked``; declaring
  ``validated`` at register time is refused because the state is earned
  through evidence, not asserted.
* ``mark_validated`` — the ``unvalidated → validated`` transition, gated
  on executed paired-evaluation evidence ``{"paired_run": str,
  "gate": str, "verdict": "pass"}``. Validation without the evidence
  dict refuses — that is the gate the kill switch exists for
  (V6-03.15). The attestation is persisted to
  ``policy_artifact_attestations`` inside the same transaction, so the
  state flip and its evidence commit or roll back together.
* ``bind_for_activation`` — resolves the configured
  ``v3.retrieval.controller_policy_artifact`` against THIS store's
  registry: the artifact must exist, be ``validated``, and load as a
  ``learned_policy_v1`` artifact. Every refusal names the artifact id —
  a learned controller never silently falls back to deterministic when
  asked to activate (the fallback belongs to shadow/replay arms).
* ``mark_revoked`` — withdraws an artifact (attested). Revocation is
  terminal: a revoked artifact can never validate.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from typing import Any, Optional

from ..core.time import wall_us
from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    require_id,
    safe_json_loads,
)
from ..retrieval.v3 import learned as _learned
from ..storage.repos import has_table
from ..storage.schema_v5 import ensure_additive_tables

_ARTIFACT_KIND = _learned.ARTIFACT_KIND  # "learned_policy_v1"
_ATTEST_TABLE = "policy_artifact_attestations"

#: Paired-evaluation evidence required for validation (V6-03.15): an
#: executed run name, the gate it satisfied, and a "pass" verdict.
_EVIDENCE_REQUIRED = ("paired_run", "gate", "verdict")


# ---------------------------------------------------------------------------
# artifact parsing — strict (never the swallow-everything load_artifact)
# ---------------------------------------------------------------------------


def _coerce_artifact(artifact_json: Any) -> dict:
    """Parse ``artifact_json`` into a validated artifact dict.

    Accepts a ``PolicyArtifact``, a dict, a JSON string, or a filesystem
    path to one — the same input surface as ``learned.load_artifact`` —
    but validates through ``PolicyArtifact.from_dict`` STRICTLY: a
    payload the controller could not load at activation time is refused
    here with a typed VALIDATION error instead of being registered.
    """
    if isinstance(artifact_json, _learned.PolicyArtifact):
        return artifact_json.to_dict()
    raw = artifact_json
    if isinstance(raw, str) and os.path.exists(raw):
        try:
            with open(raw, "r", encoding="utf-8") as fh:
                raw = fh.read()
        except OSError as exc:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"policy artifact path unreadable: {exc}",
            ) from exc
    if isinstance(raw, str):
        try:
            raw = safe_json_loads(raw)
        except VerbatimError as exc:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"policy artifact is not valid JSON: {exc}",
            ) from exc
    if not isinstance(raw, dict):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "policy artifact must be a mapping, JSON string, or path",
        )
    try:
        art = _learned.PolicyArtifact.from_dict(raw)
    except (ValueError, TypeError) as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"invalid {_ARTIFACT_KIND} artifact: {exc}"
        ) from exc
    return art.to_dict()


def _artifact_id(declared: dict) -> str:
    """Content-derived artifact id — deterministic, so re-registering
    the identical artifact is an idempotent no-op returning the same id."""
    digest = hashlib.sha256(json_dumps(declared).encode("utf-8")).hexdigest()
    return f"pa_{digest[:24]}"


def _row_for(conn: sqlite3.Connection, artifact_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT artifact_id, kind, digest, declared_json, license_id,"
        " validation_state, created_us FROM policy_artifacts"
        " WHERE artifact_id = ?",
        (artifact_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "artifact_id": row[0],
        "kind": row[1],
        "digest": row[2],
        "declared_json": row[3],
        "license_id": row[4],
        "validation_state": row[5],
        "created_us": row[6],
    }


def _attest(
    conn: sqlite3.Connection,
    artifact_id: str,
    state: str,
    evidence: dict,
) -> None:
    """Append the transition's evidence row — same tx as the state flip."""
    ensure_additive_tables(conn)
    conn.execute(
        "INSERT INTO policy_artifact_attestations"
        " (artifact_id, attested_us, state, evidence_json)"
        " VALUES (?, ?, ?, ?)",
        (artifact_id, wall_us(), state, json_dumps(evidence)),
    )


# ---------------------------------------------------------------------------
# register → validate → revoke
# ---------------------------------------------------------------------------


def register_artifact(
    conn: sqlite3.Connection,
    artifact_json: Any,
    *,
    validation_state: str = "unvalidated",
) -> str:
    """Validate and register a ``learned_policy_v1`` artifact.

    ``conn`` is the caller's write transaction (same contract as every
    repo mutator — the registration commits atomically with the caller's
    other writes). Returns the content-derived ``artifact_id``;
    re-registering the identical artifact is an idempotent no-op that
    returns the same id and leaves its validation state untouched.

    ``validation_state`` admits ``"unvalidated"`` (the normal path) or
    ``"revoked"`` (importing a tombstoned artifact). ``"validated"`` is
    REFUSED at registration — the validated state exists only through
    ``mark_validated``'s evidence gate; letting a registration declare
    itself validated would make the gate theater (V6-03.15).
    """
    if validation_state == "validated":
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "policy artifacts cannot register as 'validated' — "
            "validation requires mark_validated() with paired-run "
            "evidence {paired_run, gate, verdict}",
        )
    if validation_state not in ("unvalidated", "revoked"):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"invalid validation_state {validation_state!r}",
        )
    declared = _coerce_artifact(artifact_json)
    aid = _artifact_id(declared)
    digest = hashlib.sha256(
        json_dumps(declared).encode("utf-8")
    ).digest()
    existing = _row_for(conn, aid)
    if existing is not None:
        return aid  # identical content → identical id: idempotent
    conn.execute(
        "INSERT INTO policy_artifacts"
        "(artifact_id, kind, digest, declared_json, license_id,"
        " validation_state, created_us)"
        " VALUES (?, ?, ?, ?, NULL, ?, ?)",
        (
            aid,
            _ARTIFACT_KIND,
            digest,
            json_dumps(declared),
            validation_state,
            wall_us(),
        ),
    )
    if validation_state == "revoked":
        _attest(conn, aid, "revoked", {"registered_revoked": True})
    return aid


def _require_evidence(evidence: Any) -> dict:
    """The paired-gate evidence contract (V6-03.15): a dict carrying
    ``paired_run`` + ``gate`` (non-empty strings) and ``verdict ==
    "pass"``. Anything less refuses — validation without an executed
    paired evaluation is exactly the failure mode the gate exists for."""
    if not isinstance(evidence, dict):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "validation evidence must be a dict "
            "{paired_run, gate, verdict}",
        )
    for key in _EVIDENCE_REQUIRED:
        if key not in evidence:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"validation evidence missing {key!r} — a paired run "
                "must execute before an artifact validates",
            )
    for key in ("paired_run", "gate"):
        if not isinstance(evidence[key], str) or not evidence[key]:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"validation evidence {key!r} must be a non-empty string",
            )
    if evidence["verdict"] != "pass":
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"validation evidence verdict must be 'pass', got "
            f"{evidence['verdict']!r} — a failed or unrun gate cannot "
            "validate an artifact",
        )
    return evidence


def mark_validated(
    conn: sqlite3.Connection,
    artifact_id: str,
    *,
    evidence: Any,
) -> bool:
    """Transition ``unvalidated → validated``, gated on paired evidence.

    Returns ``True`` when the transition ran; ``False`` when the
    artifact was already validated (idempotent). Unknown artifacts and
    revoked artifacts refuse — revocation is terminal. The evidence is
    attested into ``policy_artifact_attestations`` in the same
    transaction, so the state flip is never separated from its proof.
    """
    require_id(artifact_id, "artifact_id")
    ev = _require_evidence(evidence)
    row = _row_for(conn, artifact_id)
    if row is None:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"unknown policy artifact {artifact_id!r}",
        )
    state = row["validation_state"]
    if state == "validated":
        return False
    if state == "revoked":
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"policy artifact {artifact_id!r} is revoked — revocation is "
            "terminal; register a new artifact instead",
        )
    conn.execute(
        "UPDATE policy_artifacts SET validation_state = 'validated'"
        " WHERE artifact_id = ?",
        (artifact_id,),
    )
    _attest(conn, artifact_id, "validated", ev)
    return True


def mark_revoked(
    conn: sqlite3.Connection,
    artifact_id: str,
    *,
    reason: Optional[str] = None,
) -> bool:
    """Transition ``unvalidated|validated → revoked`` (attested).

    Returns ``True`` on transition, ``False`` when already revoked.
    A revoked artifact refuses both activation and validation — the
    withdrawal is terminal and auditable.
    """
    require_id(artifact_id, "artifact_id")
    row = _row_for(conn, artifact_id)
    if row is None:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"unknown policy artifact {artifact_id!r}",
        )
    if row["validation_state"] == "revoked":
        return False
    conn.execute(
        "UPDATE policy_artifacts SET validation_state = 'revoked'"
        " WHERE artifact_id = ?",
        (artifact_id,),
    )
    _attest(
        conn, artifact_id, "revoked",
        {"reason": reason or "operator_revoked"},
    )
    return True


def attestations(conn: sqlite3.Connection, artifact_id: str) -> list[dict]:
    """Audit trail for one artifact — every attested state transition,
    oldest first. Empty when the additive table is absent or no
    transitions were ever attested.
    """
    require_id(artifact_id, "artifact_id")
    if not has_table(conn, _ATTEST_TABLE):
        return []
    rows = conn.execute(
        "SELECT artifact_id, attested_us, state, evidence_json"
        " FROM policy_artifact_attestations WHERE artifact_id = ?"
        " ORDER BY attested_us",
        (artifact_id,),
    ).fetchall()
    return [
        {
            "artifact_id": r[0],
            "attested_us": r[1],
            "state": r[2],
            "evidence": safe_json_loads(r[3]),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# activation binding (V6-03.15/16) — the store-side half of the config gate
# ---------------------------------------------------------------------------


def _configured_binding(cfg: Any) -> tuple[Optional[str], Optional[str]]:
    """``(controller, controller_policy_artifact)`` as the config
    declares them — ``(None, None)`` when cfg carries no v3.retrieval
    section (callers may pass the section, the v3 root, or the whole
    config; the lookup is defensive either way)."""
    if cfg is None:
        return None, None
    v3 = getattr(cfg, "v3", cfg)
    retrieval = getattr(v3, "retrieval", v3)
    controller = getattr(retrieval, "controller", None)
    bound = getattr(retrieval, "controller_policy_artifact", None)
    return controller, bound


def bind_for_activation(
    cfg: Any,
    artifact_id: Optional[str],
    store: Any,
    *,
    conn: Optional[sqlite3.Connection] = None,
) -> "_learned.PolicyArtifact":
    """Resolve the bound policy artifact for ``learned_active``.

    The config half already ran at validation time: ``learned_active``
    was accepted only with ``controller_policy_artifact`` declared. This
    is the store half — enforced at the point the controller is
    constructed (``recall_v3``'s controller-selection block, inside the
    read snapshot):

    1. ``artifact_id`` must be bound — either passed explicitly or read
       off ``cfg.v3.retrieval.controller_policy_artifact``; a passed id
       that disagrees with the configured binding refuses (activating a
       different artifact than configured is not a fallback).
    2. The artifact must exist in THIS store's ``policy_artifacts`` with
       ``validation_state == 'validated'`` — ``unvalidated`` and
       ``revoked`` name themselves in the refusal.
    3. The registered payload must parse as ``learned_policy_v1`` —
       corrupt registry content refuses rather than degrading.

    Returns the loaded :class:`PolicyArtifact` (read-only — activation
    applies it through ``learned.choose``). Every refusal is
    ``CONFIG_INVALID`` naming the artifact id; there is no silent
    deterministic fallback on this path.
    """
    controller, configured = _configured_binding(cfg)
    aid = artifact_id or configured
    if not aid:
        raise VerbatimError(
            ErrorCode.CONFIG_INVALID,
            "learned_active requires a validated policy artifact bound "
            "via v3.retrieval.controller_policy_artifact=<artifact_id> — "
            "no artifact is bound",
        )
    if not isinstance(aid, str) or not aid:
        raise VerbatimError(
            ErrorCode.CONFIG_INVALID,
            "learned_active requires a validated policy artifact bound "
            "via v3.retrieval.controller_policy_artifact=<artifact_id> — "
            f"the binding is not an artifact id ({aid!r})",
        )
    if configured and artifact_id and configured != artifact_id:
        raise VerbatimError(
            ErrorCode.CONFIG_INVALID,
            f"activation requested artifact {artifact_id!r} but config "
            f"binds {configured!r} — the bound artifact is the only one "
            "learned_active may use",
        )
    require_id(aid, "artifact_id")
    if controller is not None and controller != "learned_active":
        raise VerbatimError(
            ErrorCode.CONFIG_INVALID,
            f"controller is {controller!r}, not 'learned_active' — "
            f"policy artifact {aid!r} cannot activate",
        )

    def _probe(c: sqlite3.Connection) -> "_learned.PolicyArtifact":
        row = _row_for(c, aid)
        if row is None:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                f"learned_active bound artifact {aid!r} is not "
                "registered in this store's policy_artifacts — register "
                "and validate it before activating",
            )
        state = row["validation_state"]
        if state != "validated":
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                f"learned_active bound artifact {aid!r} is {state!r}, "
                "not 'validated' — a paired-run evidence gate "
                "(mark_validated) must pass before activation",
            )
        try:
            declared = safe_json_loads(row["declared_json"])
            return _learned.PolicyArtifact.from_dict(declared)
        except (VerbatimError, ValueError, TypeError) as exc:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                f"learned_active bound artifact {aid!r} payload does not "
                f"load as {_ARTIFACT_KIND}: {exc}",
            ) from exc

    if conn is not None:
        return _probe(conn)
    with store.read() as rconn:
        return _probe(rconn)
