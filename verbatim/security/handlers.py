"""Job handlers for the security pipeline (SPEC_V3 §40, contracts doc).

Registered in ``ingest._V3_KIND_HANDLERS``:

* ``screen`` → :func:`handle_screen` — resolves the object text, runs the
  deterministic rules_v1 screener, persists a ``security_labels`` row, and
  opens a ``quarantine`` hold when the verdict is ``suspicious`` or
  ``blocked`` (§14.02, §34 stage 1). Effects commit inside ONE
  generation-fenced transaction through ``ingester._commit_effects``, so a
  redelivered job replays its operation receipt instead of double-labeling
  (V3-40.01).
* ``quarantine_review`` → :func:`handle_quarantine_review` — applies a
  reviewer decision (``release`` | ``suppress`` | ``purge``) recorded in
  ``quarantine.decision_json`` with ``decided_by`` and the decided event.
  Enqueueing is restricted to ``review``-verb callers by governance; the
  handler itself validates ``decided_by`` non-empty so a job can never
  record an anonymous decision.

Job ``input_refs`` for ``screen``::

    {"object_kind": "span"|"claim"|..., "object_id": str,
     "revision": int, "text_ref": str (optional),
     "text": str (optional literal — the test path),
     "source_trust": str (default "unknown"),
     "context_kind": str (optional)}

``text_ref`` resolves through the spans repo when present; a missing text
source fails EVIDENCE_UNAVAILABLE (loud, never a silent no-op).
"""

from __future__ import annotations

from typing import Any, Optional

from ..core.types import ErrorCode, VerbatimError
from ..core.types_v3 import AttackRisk
from . import admission, labels, quarantine, screening


def _require_ref(refs: dict[str, Any], key: str) -> str:
    value = refs.get(key)
    if not isinstance(value, str) or not value:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"job input_refs.{key} must be a non-empty string",
        )
    return value


def _require_int_ref(refs: dict[str, Any], key: str) -> int:
    value = refs.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerbatimError(
            ErrorCode.VALIDATION, f"job input_refs.{key} must be an integer"
        )
    return value


def _resolve_text(refs: dict[str, Any], ingester: Any) -> str:
    """Screen target text: literal ``text`` ref first (test path), else the
    ``text_ref`` span id resolved through the spans repo. A ``source``
    object resolves its current revision payload from ``source_revisions``
    so sources can be screened without duplicating payload bytes into
    ``input_refs`` (§34.01)."""
    text = refs.get("text")
    if text is not None:
        if not isinstance(text, str):
            raise VerbatimError(
                ErrorCode.VALIDATION, "input_refs.text must be a string"
            )
        return text
    if refs.get("object_kind") == "source":
        object_id = refs.get("object_id")
        revision = refs.get("revision")
        if isinstance(object_id, str) and object_id:
            # Verified read through the storage authority: payload_hmac
            # is re-checked, so the screener never adjudicates forged
            # bytes — a tampered revision fails STORE_CORRUPT instead of
            # producing a label over corrupted content (V4-07.02).
            payload = ingester.sources.payload(
                object_id, int(revision or 1)
            )
            if payload is None:
                raise VerbatimError(
                    ErrorCode.EVIDENCE_UNAVAILABLE,
                    f"screen source {object_id!r} rev {revision} unavailable",
                )
            return bytes(payload).decode("utf-8", errors="replace")
    text_ref = refs.get("text_ref")
    if not isinstance(text_ref, str) or not text_ref:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "input_refs needs 'text' or 'text_ref' to resolve screen content",
        )
    resolved = ingester.spans.text(text_ref)
    if resolved is None:
        raise VerbatimError(
            ErrorCode.EVIDENCE_UNAVAILABLE,
            f"screen text_ref {text_ref!r} unavailable",
        )
    return resolved


def handle_screen(job: dict[str, Any], owner: str, ingester: Any) -> None:
    """Screen one object revision; attach a label; quarantine on findings.

    Idempotent: the caller supplies ``operation_key``/``dedup_key`` at
    enqueue; ``_commit_effects`` replays a committed receipt on redelivery,
    and ``open_quarantine`` is primary-key idempotent on top of that.
    """
    refs = job["input_refs"]
    object_kind = _require_ref(refs, "object_kind")
    object_id = _require_ref(refs, "object_id")
    revision = _require_int_ref(refs, "revision")
    source_trust = refs.get("source_trust", "unknown")
    context_kind = refs.get("context_kind")

    text = _resolve_text(refs, ingester)
    verdict = screening.screen_content(
        text, source_trust=source_trust, context_kind=context_kind
    )
    review_state = admission.default_review_state(
        source_trust,
        attack_risk=verdict.attack_risk,
        content_form=verdict.content_form,
        findings=list(verdict.findings),
    )
    findings = [dict(f) for f in verdict.findings]
    reason_codes = [f"attack_risk:{verdict.attack_risk.value}"] + [
        f"rule:{f['rule_id']}" for f in verdict.findings
    ]

    def _apply(conn: Any) -> dict[str, Any]:
        label_id = labels.attach_label(
            conn,
            job["scope_id"],
            source_trust=source_trust,
            content_form=verdict.content_form,
            attack_risk=verdict.attack_risk,
            review_state=review_state,
            findings=findings,
            method=verdict.method,
            rules_revision=verdict.rules_revision,
        )
        opened = False
        if verdict.attack_risk in (AttackRisk.SUSPICIOUS, AttackRisk.BLOCKED):
            opened = quarantine.open_quarantine(
                conn,
                (object_kind, object_id, revision),
                reason_codes,
                findings,
                scope_id=job["scope_id"],
            )
        return {
            "label_id": label_id,
            "content_form": verdict.content_form.value,
            "attack_risk": verdict.attack_risk.value,
            "review_state": review_state,
            "findings": len(findings),
            "quarantined": opened,
        }

    with ingester.store.tx() as conn:
        ingester._commit_effects(conn, job, owner, "screen", _apply)


def handle_quarantine_review(
    job: dict[str, Any], owner: str, ingester: Any
) -> None:
    """Apply a quarantine decision to an open hold (§34.03).

    ``input_refs``::

        {"object_kind": str, "object_id": str, "revision": int,
         "decision": "release"|"suppress"|"purge",
         "decided_by": str (required — anonymous decisions rejected),
         "rationale": str (optional), "label_id": str (optional),
         "expected_versions": {...} (optional, recorded verbatim)}

    When ``label_id`` is supplied the label's ``review_state`` advances in
    the same transaction — release → ``released``, suppress/purge →
    ``quarantined``. Origin and findings are never rewritten (V3-14.01).
    """
    refs = job["input_refs"]
    object_kind = _require_ref(refs, "object_kind")
    object_id = _require_ref(refs, "object_id")
    revision = _require_int_ref(refs, "revision")
    decided_by = _require_ref(refs, "decided_by")
    decision = _require_ref(refs, "decision")
    if decision not in quarantine.DECISIONS:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"decision must be one of {sorted(quarantine.DECISIONS)}",
        )
    label_id = refs.get("label_id")
    if label_id is not None and (
        not isinstance(label_id, str) or not label_id
    ):
        raise VerbatimError(
            ErrorCode.VALIDATION, "input_refs.label_id must be a non-empty string"
        )
    payload = {
        k: refs[k]
        for k in ("rationale", "expected_versions", "notes")
        if k in refs
    }

    def _apply(conn: Any) -> dict[str, Any]:
        ref = (object_kind, object_id, revision)
        if decision == "release":
            row = quarantine.release(conn, ref, decided_by, payload)
            label_state = "released"
        elif decision == "suppress":
            row = quarantine.suppress(conn, ref, decided_by, payload)
            label_state = "quarantined"
        else:  # purge — privacy performs the deletion; we record the tombstone
            row = quarantine.mark_purged(conn, ref, decided_by, payload)
            label_state = "quarantined"
        if label_id is not None:
            labels.update_review_state(conn, label_id, label_state)
        return {
            "object_kind": object_kind,
            "object_id": object_id,
            "revision": revision,
            "state": row.get("state"),
            "decided_by": decided_by,
        }

    with ingester.store.tx() as conn:
        ingester._commit_effects(conn, job, owner, "quarantine_review", _apply)


__all__ = ["handle_quarantine_review", "handle_screen"]
