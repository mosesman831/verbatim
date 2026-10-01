"""Working sets: bounded session/task views (SPEC_V3 §24, V3-24.03).

A working set is a session-scoped, TTL-bound view over recent evidence,
unresolved references, and temporary goals. Expiry suppresses the *view*
only — ``get_set`` returns ``None`` past ``expires_us`` — never the
underlying evidence (V3-24.03). Each set is bounded at
``WORKING_SET_ITEM_CAP`` items (default 64, V3-17.10 storage bound).

``promote`` (SPEC_V4_5 §09 X7, V45-09.07, D19) is the ONLY way session
working memory becomes durable: it requires an explicit persisted
``capture_authorizations`` row — retention consent, which an approved
tool call is not (V3-11.11) — covering the principal, scope, envelope
kind, and retention policy; then each item writes through the real
``ingest_envelope`` evidence path (sources + revisions + envelope +
screening + event), preserving item-level provenance and the session id.
Absent, expired, revoked, mismatched, or unscoped consent denies
identically: ``CONSENT_REQUIRED``, nothing written.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

from ..core.time import now_us
from ..core.types import (
    ErrorCode,
    VerbatimError,
    new_id,
    require_id,
)
from ..storage import repos_v3

#: Bound on items per working set (V3-17.10).
WORKING_SET_ITEM_CAP = 64

#: Recognized item kinds (§17 WorkingSetItem). Not enforced — hosts may
#: name their own item kinds; these are the documented set.
ITEM_KINDS = ("goal", "unresolved_ref", "decision", "recent_evidence")


def _require_set_live(
    conn: sqlite3.Connection, set_id: str, now: int
) -> dict[str, Any]:
    row = repos_v3.get(conn, "working_sets", {"set_id": set_id})
    if row is None or int(row["expires_us"]) <= now:
        raise VerbatimError(
            ErrorCode.VALIDATION, "working set not found or expired"
        )
    return row


def create_set(
    conn: sqlite3.Connection,
    scope_id: str,
    session_id: str,
    ttl_us: int,
    *,
    now: Optional[int] = None,
) -> str:
    """Create a bounded working set; returns ``set_id``.

    ``ttl_us`` is the view lifetime in microseconds — expiry suppresses the
    view, never the evidence (V3-24.03).
    """
    require_id(scope_id, "scope_id")
    require_id(session_id, "session_id")
    if not isinstance(ttl_us, int) or ttl_us <= 0:
        raise VerbatimError(ErrorCode.VALIDATION, "ttl_us must be positive")
    at = now_us() if now is None else now
    set_id = new_id()
    repos_v3.insert(
        conn,
        "working_sets",
        {
            "set_id": set_id,
            "scope_id": scope_id,
            "session_id": session_id,
            "created_us": at,
            "expires_us": at + ttl_us,
        },
    )
    return set_id


def item_count(conn: sqlite3.Connection, set_id: str) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM working_set_items WHERE set_id = ?",
            (set_id,),
        ).fetchone()[0]
    )


def _screen_text(
    conn: sqlite3.Connection, scope_id: str, item_id: str, text: str
) -> None:
    """Write-channel screening for caller-supplied item text (§34.01,
    V3-14.10).

    Same producer convention as envelope capture and procedure compile:
    the deterministic rules_v1 verdict is persisted as a security label,
    and a ``suspicious``/``blocked`` verdict opens a quarantine hold on
    ``("working_item", item_id, 1)`` — the object kind the recall union's
    admission check withholds — so poisoned session text lands held for
    review instead of shipping through the working_item recall lane.
    Degrades openly when the security module is unprovisioned: no screen
    is claimed, and the item stands on its own — never a fabricated
    verdict.
    """
    try:
        from ..core.types_v3 import AttackRisk  # type: ignore
        from ..security import (  # type: ignore
            attach_label,
            default_review_state,
            open_quarantine,
            screen_content,
        )
    except ImportError:
        return
    verdict = screen_content(
        text, source_trust="unknown", context_kind="working_set_item"
    )
    findings = [dict(f) for f in verdict.findings]
    review_state = default_review_state(
        "unknown",
        attack_risk=verdict.attack_risk,
        content_form=verdict.content_form,
        findings=findings,
    )
    attach_label(
        conn,
        scope_id,
        source_trust="unknown",
        content_form=verdict.content_form,
        attack_risk=verdict.attack_risk,
        review_state=review_state,
        findings=findings,
        method=verdict.method,
        rules_revision=verdict.rules_revision,
    )
    if verdict.attack_risk in (AttackRisk.SUSPICIOUS, AttackRisk.BLOCKED):
        open_quarantine(
            conn,
            ("working_item", item_id, 1),
            [f"attack_risk:{verdict.attack_risk.value}"]
            + [f"rule:{f['rule_id']}" for f in findings],
            findings,
            scope_id=scope_id,
        )


def _item_held(conn: sqlite3.Connection, item_id: str) -> bool:
    """Whether a quarantine hold hides this item (V3-14.10 invisibility —
    the same ``("working_item", item_id, 1)`` key the recall union checks).

    Fail closed (F4-20, V4-05.12, C84): a hold check that cannot be
    completed — ``security.should_exclude`` raising, or the local
    quarantine-table read failing — raises ``EVIDENCE_UNAVAILABLE``
    instead of reporting not-held. No exception on the enforcement path
    ever releases held working memory; ``get_set`` surfaces the typed
    failure rather than shipping items whose hold state is unverified.
    The only degrade-open case is the security module being genuinely
    unprovisioned (import failure) — then the local read is the
    authoritative check, and it too fails closed.
    """
    try:
        from .. import security as _security  # type: ignore
    except Exception:
        _security = None
    if _security is not None:
        try:
            return bool(_security.should_exclude(conn, "working_item", item_id, 1))
        except VerbatimError:
            raise
        except Exception as exc:
            raise VerbatimError(
                ErrorCode.EVIDENCE_UNAVAILABLE,
                f"quarantine hold check failed for working_item:{item_id}",
            ) from exc
    try:
        row = conn.execute(
            "SELECT state FROM quarantine"
            " WHERE object_kind = 'working_item' AND object_id = ?"
            " AND revision = 1",
            (item_id,),
        ).fetchone()
    except sqlite3.Error as exc:
        raise VerbatimError(
            ErrorCode.EVIDENCE_UNAVAILABLE,
            f"quarantine hold check failed for working_item:{item_id}",
        ) from exc
    return row is not None and row[0] in ("pending", "suppressed")


def add_item(
    conn: sqlite3.Connection,
    set_id: str,
    kind: str,
    *,
    object_ref: Optional[str] = None,
    text: Optional[str] = None,
    ord: Optional[int] = None,
    cap: int = WORKING_SET_ITEM_CAP,
    now: Optional[int] = None,
) -> str:
    """Append one item to a live set; returns ``item_id``.

    Exactly one of ``object_ref``/``text`` is required — an item either
    points at an object or carries its own note text. The set is bounded:
    adding past ``cap`` raises ``BUDGET_EXCEEDED`` (V3-17.10), it never
    evicts silently.
    """
    require_id(set_id, "set_id")
    require_id(kind, "kind")
    if (object_ref is None) == (text is None):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "working-set item needs exactly one of object_ref/text",
        )
    if object_ref is not None:
        require_id(object_ref, "object_ref")
    if cap < 1:
        raise VerbatimError(ErrorCode.VALIDATION, "cap must be >= 1")
    set_row = _require_set_live(conn, set_id, now_us() if now is None else now)
    count = item_count(conn, set_id)
    if count >= cap:
        raise VerbatimError(
            ErrorCode.BUDGET_EXCEEDED,
            f"working set {set_id} is full at {cap} items",
        )
    item_id = new_id()
    repos_v3.insert(
        conn,
        "working_set_items",
        {
            "item_id": item_id,
            "set_id": set_id,
            "kind": kind,
            "object_ref": object_ref,
            "text": text,
            "ord": count if ord is None else ord,
        },
    )
    if text is not None:
        # Raw caller text ships through the working_item recall lane —
        # screen it before it can (§34.01, V3-14.10): the verdict lands as
        # a security label, and suspicious/blocked text is held in
        # quarantine pending review rather than delivered to recall.
        _screen_text(conn, set_row["scope_id"], item_id, text)
    return item_id


def get_set(
    conn: sqlite3.Connection, set_id: str, *, now: Optional[int] = None
) -> Optional[dict[str, Any]]:
    """Live set snapshot with ordered items; ``None`` when missing or
    expired (V3-24.03 — expiry suppresses the view only).

    Raises ``EVIDENCE_UNAVAILABLE`` when an item's quarantine-hold check
    cannot be completed (F4-20, V4-05.12): the read reports typed
    unavailability instead of shipping items whose hold state is
    unverified — no exception-based release of held working memory."""
    require_id(set_id, "set_id")
    row = repos_v3.get(conn, "working_sets", {"set_id": set_id})
    if row is None:
        return None
    at = now_us() if now is None else now
    if int(row["expires_us"]) <= at:
        return None
    items = repos_v3.query(
        conn,
        "working_set_items",
        {"set_id": set_id},
        order="ord",
    )
    # Quarantine holds withhold items from this delivery view exactly as
    # they withhold them from the recall lane (V3-14.10) — the rows stay
    # for review and cap accounting, they are just never shipped.
    items = [i for i in items if not _item_held(conn, i["item_id"])]
    return {**row, "items": items}


def list_sets(
    conn: sqlite3.Connection,
    scope_id: str,
    session_id: Optional[str] = None,
    *,
    now: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Non-expired sets in a scope (optionally one session's)."""
    require_id(scope_id, "scope_id")
    where: dict[str, Any] = {"scope_id": scope_id}
    if session_id is not None:
        where["session_id"] = require_id(session_id, "session_id")
    at = now_us() if now is None else now
    return [
        row
        for row in repos_v3.query(conn, "working_sets", where, order="created_us")
        if int(row["expires_us"]) > at
    ]


# ---------------------------------------------------------------------------
# X7 — session-to-durable promotion (SPEC_V4_5 §09 X7, V45-09.07, D19)
# ---------------------------------------------------------------------------

#: Envelope kind promoted items take: a host-recorded event — never a
#: user/agent impersonation. It carries the honest default trust class
#: (HOST_OBSERVED) and flows through the standard screened ingest path.
PROMOTE_ENVELOPE_KIND = "system_event"


def _require_retention_consent(
    conn: sqlite3.Connection,
    *,
    authorization: Any,
    principal_id: str,
    scope_id: str,
    envelope_kind: str,
    retention_policy: Optional[str],
    now: int,
) -> dict[str, Any]:
    """The X7 gate: an explicit, persisted, live retention-consent row.

    Every mismatch — absent row, different principal, kind/scope/policy
    not covered, expiry, revocation, or a caller object that does not
    match its persisted record — denies identically with
    ``CONSENT_REQUIRED`` (V45-09.07: tool permission is never retention
    consent, so no implicit fallback exists).
    """
    from ..core.types_v3 import CaptureAuthorization  # local: v3 types
    from ..governance import consent as _consent  # local: governance edge

    deny = VerbatimError(
        ErrorCode.CONSENT_REQUIRED,
        "session working memory lacks a matching retention-consent "
        "record (V45-09.07 — tool permission is not retention consent)",
    )
    if not isinstance(authorization, CaptureAuthorization):
        raise deny
    row = _consent.get_capture_authorization(
        conn, authorization.authorization_id
    )
    if row is None:
        raise deny  # never persisted — an in-memory claim is not consent
    if row.get("revoked_us") is not None:
        raise deny
    exp = row.get("expires_us")
    if exp is not None and int(exp) <= now:
        raise deny
    # The persisted row is authoritative: the caller's object must match
    # it field for field — a minted object naming a real id but widening
    # kinds/scopes is not that consent.
    if row.get("principal_id") != authorization.principal_id:
        raise deny
    if row.get("principal_id") != principal_id:
        raise deny  # consent is issued FOR this principal, not the caller's say-so
    allowed = set(repos_v3.json_field(row, "allowed_kinds_json") or ())
    if set(k.value for k in authorization.allowed_kinds) != allowed:
        raise deny
    scopes = set(repos_v3.json_field(row, "scope_ids_json") or ())
    if set(authorization.scope_ids) != scopes:
        raise deny
    if row.get("retention_policy") != authorization.retention_policy:
        raise deny
    if (row.get("expires_us") or None) != authorization.expires_us:
        raise deny
    # Coverage of THIS promotion: kind, scope, and an asserted policy.
    if envelope_kind not in allowed:
        raise deny
    if scopes and scope_id not in scopes:
        raise deny
    if (
        retention_policy is not None
        and row.get("retention_policy") != retention_policy
    ):
        raise deny  # the caller asserted a policy the consent did not grant
    return row


def promote(
    conn: sqlite3.Connection,
    store: Any,
    scope_id: str,
    session_id: str,
    *,
    principal_id: str,
    authorization: Any,
    set_id: Optional[str] = None,
    retention_policy: Optional[str] = None,
    now: Optional[int] = None,
) -> dict[str, Any]:
    """Promote live session working items to durable evidence — only
    under explicit retention consent (V45-09.07, D19).

    ``authorization`` must be a ``CaptureAuthorization`` whose persisted
    row is live (unexpired, unrevoked), issued for ``principal_id``, and
    covering (scope, ``system_event``, retention policy). The check runs
    BEFORE any write; denial is always ``CONSENT_REQUIRED`` and nothing
    is persisted. Each text item then commits through the real
    ``ingest_envelope`` path — sources + revision + span + envelope +
    screening + journal event — with ``capture_proof`` bound to the
    authorization id and provenance naming the working set/item, so the
    durable record never forgets it was promoted session working memory.

    ``object_ref`` items already reference durable objects — they are
    counted in ``skipped_object_refs`` rather than re-captured.
    Quarantine-held items stay held pending review — counted in
    ``withheld_quarantine``, never laundered into durable memory. Sets
    must be live and belong to (``scope_id``, ``session_id``); expired or
    foreign sets deny rather than leak. Retries are idempotent: each
    item's dedup key (``working:{item_id}``) replays its committed
    receipt instead of double-capturing.
    """
    require_id(scope_id, "scope_id")
    require_id(session_id, "session_id")
    require_id(principal_id, "principal_id")
    at = now_us() if now is None else now

    if set_id is not None:
        require_id(set_id, "set_id")
        row = repos_v3.get(conn, "working_sets", {"set_id": set_id})
        if (
            row is None
            or row["scope_id"] != scope_id
            or row["session_id"] != session_id
        ):
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
                "no such working set in this scope/session",
            )
        if int(row["expires_us"]) <= at:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "expired working sets cannot be promoted",
            )
        sets = [row]
    else:
        sets = list_sets(conn, scope_id, session_id, now=at)

    # Consent gate FIRST — before any durable write (V45-09.07, D19).
    auth_row = _require_retention_consent(
        conn,
        authorization=authorization,
        principal_id=principal_id,
        scope_id=scope_id,
        envelope_kind=PROMOTE_ENVELOPE_KIND,
        retention_policy=retention_policy,
        now=at,
    )

    from ..core.types_v3 import (  # local: observations → v3 types edge
        EnvelopeKind,
        Perspective,
        SourceEnvelopeV3,
    )
    from ..evidence.envelopes import (  # local: observations → evidence
        ingest_envelope,
    )

    promoted: list[dict[str, Any]] = []
    skipped_refs: list[str] = []
    withheld: list[str] = []
    for srow in sets:
        items = repos_v3.query(
            conn, "working_set_items", {"set_id": srow["set_id"]},
            order="ord",
        )
        for item in items:
            if _item_held(conn, item["item_id"]):
                withheld.append(item["item_id"])
                continue
            if item.get("object_ref") is not None:
                skipped_refs.append(item["item_id"])
                continue
            env = SourceEnvelopeV3(
                kind=EnvelopeKind.SYSTEM_EVENT,
                scope_id=scope_id,
                actor_principal=principal_id,
                perspective=Perspective(
                    observer=principal_id,
                    audience=(principal_id,),
                ),
                event_us=at,
                receipt_us=0,
                content=str(item["text"]).encode("utf-8"),
                media_type="text/plain",
                capture_proof=authorization.authorization_id,
                host_id="working-promotion",
                session_id=session_id,
                external_id=f"working:{item['item_id']}",
                metadata={
                    "promoted_from": "working_set",
                    "working_set_id": srow["set_id"],
                    "working_item_id": item["item_id"],
                    "working_item_kind": item["kind"],
                    "working_item_ord": item["ord"],
                    "session_id": session_id,
                    "authorization_id": authorization.authorization_id,
                    "retention_policy": auth_row["retention_policy"],
                },
            )
            receipt = ingest_envelope(
                conn, store, env, authorization=authorization
            )
            promoted.append(
                {
                    "item_id": item["item_id"],
                    "receipt_id": receipt.receipt_id,
                    "envelope_id": receipt.envelope_id,
                    "source_id": receipt.source_id,
                    "event_seq": receipt.event_seq,
                    "dedup_key": receipt.dedup_key,
                }
            )
    return {
        "scope_id": scope_id,
        "session_id": session_id,
        "set_ids": [s["set_id"] for s in sets],
        "principal_id": principal_id,
        "authorization_id": authorization.authorization_id,
        "retention_policy": auth_row["retention_policy"],
        "envelope_kind": PROMOTE_ENVELOPE_KIND,
        "promoted": promoted,
        "skipped_object_refs": skipped_refs,
        "withheld_quarantine": withheld,
    }
