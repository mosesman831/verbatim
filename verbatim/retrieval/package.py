"""Evidence packaging, conflict closure, and context budgets (SPEC §31,
SPEC_V2 §30).

Packaging turns ranked claims into bounded, attributable EvidenceGroups:
exact span bytes are cut from source payloads (never paraphrased), every
group is atomic — a primary claim plus its required context members plus
the unresolved contrary claims of a touched open conflict group — and the
byte/item/span budgets drop whole groups, never truncating a quotation, a
condition, a speaker label, or a conflict warning to squeeze one more
result in (V2-30.09).

``items`` on the result remains the flattened compatibility view; ``groups``
carries the atomic units themselves.

Integrity (F4-06, V4-08.03, V4-13.05): every payload/span byte read goes
through ``_verify_slice`` before quoting — the same consumption-time
digest re-check ``SourcesRepo.payload`` and ``SpansRepo.text`` perform.
A digest mismatch fails closed as ``STORE_CORRUPT`` (never served); a
NULL legacy digest is served but labeled ``LEGACY_UNVERIFIED`` and can
never become verified through a surviving sibling digest (C03/C04).
"""

from __future__ import annotations

import hmac
import sqlite3
from typing import Any, Optional

from ..core.time import contains, precision_label
from ..core.types import (
    Condition,
    ErrorCode,
    EvidenceGroup,
    EvidenceItem,
    Lifecycle,
    Precision,
    Provenance,
    RecallResult,
    SpanRef,
    TimeInterval,
    VerbatimError,
    json_dumps,
    safe_json_loads,
)
from .candidates import (
    Deadline,
    GROUP_MEMBER_CAP,
    SPAN_CAP,
    _MODE_STATES,
    _allowed_scopes,
    _fts5_available,
    _has_table,
    _latest_known_revisions,
    _suppressed,
)

# Reason codes surfaced on EvidenceItem.reasons (SPEC §31, V2-30.14).
_SRC_CODES = {
    "lexical": "LEXICAL_MATCH",
    "semantic": "SEMANTIC_MATCH",
    "structured": "ENTITY_MATCH",
    "graph": "GRAPH_NEIGHBOR",
}

# Applicability verdict → item reason code (three-valued, V2-17).
_APP_CODES = {
    "applies": "APPLIES",
    "unknown": "APPLICABILITY_UNKNOWN",
    "does_not_apply": "DOES_NOT_APPLY",
}

# Lifecycle used to label non-claim memory-kind units (episodes,
# procedures, prospective records) on their synthetic items.
_KIND_LIFECYCLE = {
    "active": Lifecycle.ACTIVE,
    "proposed": Lifecycle.PENDING,
    "review": Lifecycle.PENDING,
    "planned": Lifecycle.ACTIVE,
    "in_progress": Lifecycle.ACTIVE,
    "overdue": Lifecycle.ACTIVE,
    "unknown": Lifecycle.ACTIVE,
    "retired": Lifecycle.ARCHIVED,
    "completed": Lifecycle.ARCHIVED,
    "cancelled": Lifecycle.ARCHIVED,
}


def _item_dict(item: EvidenceItem) -> dict:
    return {
        "claim_id": item.claim_id,
        "claim_revision": item.claim_revision,
        "text": item.text,
        "span": {
            "span_id": item.span.span_id,
            "source_id": item.span.source_id,
            "revision": item.span.revision,
            "start_byte": item.span.start_byte,
            "end_byte": item.span.end_byte,
        },
        "speaker_id": item.speaker_id,
        "lifecycle": item.lifecycle.value,
        "valid_label": item.valid_label,
        "recorded_seq": item.recorded_seq,
        "reasons": list(item.reasons),
        "provenance": item.provenance.value,
        "disputed": item.disputed,
        "historical": item.historical,
    }


def serialize_item(item: EvidenceItem) -> bytes:
    """Canonical wire form of one item; also the budget-accounting unit.

    Byte accounting covers text + metadata + delimiters (SPEC §31): the
    trailing record separator is part of the serialized size.
    """
    return json_dumps(_item_dict(item)).encode("utf-8") + b"\x1e\n"


def serialize_group(group: EvidenceGroup) -> bytes:
    """Canonical wire form of one atomic evidence group (V2-30).

    Group serialization wraps the member items' canonical forms — a quote
    is never re-encoded or paraphrased on its way into a group.
    """
    obj = {
        "primary_claim_id": group.primary_claim_id,
        "complete": group.complete,
        "reasons": list(group.reasons),
        "group_score": group.group_score,
        "items": [_item_dict(i) for i in group.items],
    }
    return json_dumps(obj).encode("utf-8") + b"\x1e\n"


def _ph(n: int) -> str:
    return ",".join("?" * n)


def _pair_where(pairs: list) -> tuple:
    where = " OR ".join("(claim_id=? AND revision=?)" for _ in pairs)
    return where, [v for pair in pairs for v in pair]


def _revision_meta(conn: sqlite3.Connection, pairs: list) -> dict:
    """(claim_id, revision) → (state, recorded_from) for ranked claims."""
    if not pairs:
        return {}
    where, flat = _pair_where(pairs)
    rows = conn.execute(
        "SELECT claim_id, revision, state, recorded_from"
        f" FROM claim_revisions WHERE {where}",
        flat,
    ).fetchall()
    return {(cid, rev): (state, rf) for cid, rev, state, rf in rows}


def _conflict_closure(
    conn: sqlite3.Connection,
    store: Any,
    ranked_ids: list,
    allowed_set: set,
    plan: Any,
    request: Any,
) -> tuple:
    """Open conflict groups touched by ranked claims (V2-26.08).

    Returns ``(claim→group, group→members, group→complete, member_meta)``.
    Closure pulls *every* member of a touched open group — not only the
    members that happened to rank — because shipping a lone side of an
    unresolved dispute would misrepresent it as settled truth (V2-26.09).

    A member that is suppressed counts as gone, not as missing; a member
    that is unauthorized, unresolvable at the known cutoff, or whose
    evidence is suppressed makes the whole group incomplete — such groups
    are omitted wholesale with a scoped explanation.
    """
    cid2gid: dict = {}
    groups: dict = {}
    member_meta: dict = {}
    if not ranked_ids or not (
        _has_table(conn, "conflict_members")
        and _has_table(conn, "conflict_groups")
    ):
        return cid2gid, groups, {}, member_meta

    rows = conn.execute(
        "SELECT cm.claim_id, cm.group_id FROM conflict_members cm"
        " JOIN conflict_groups g ON g.group_id = cm.group_id"
        f" WHERE cm.claim_id IN ({_ph(len(ranked_ids))}) AND g.status='open'"
        " ORDER BY cm.group_id, cm.claim_id",
        ranked_ids,
    ).fetchall()
    for cid, gid in rows:
        cid2gid.setdefault(cid, gid)
    touched = sorted(set(cid2gid.values()))
    if not touched:
        return cid2gid, groups, {}, member_meta

    member_rows = conn.execute(
        "SELECT group_id, claim_id FROM conflict_members"
        f" WHERE group_id IN ({_ph(len(touched))}) ORDER BY claim_id",
        touched,
    ).fetchall()
    all_members: dict = {}
    member_ids: list = []
    for gid, cid in member_rows:
        all_members.setdefault(gid, []).append(cid)
        if cid not in member_ids:
            member_ids.append(cid)

    scope_of = dict(
        conn.execute(
            f"SELECT claim_id, scope_id FROM claims"
            f" WHERE claim_id IN ({_ph(len(member_ids))})",
            member_ids,
        ).fetchall()
    )
    resolved = _latest_known_revisions(conn, member_ids, plan.known_at_seq)
    suppressed_claims = _suppressed(store, conn, "claim", member_ids)
    allowed_states = _MODE_STATES[request.mode]

    # Span-level suppression makes a member unrepresentable — the group is
    # then incomplete rather than silently half-quoted.
    resolvable = {
        cid: meta for cid, meta in resolved.items()
        if meta[1] in allowed_states and cid not in suppressed_claims
    }
    suppressed_spans: set = set()
    if resolvable:
        pairs = [(cid, resolvable[cid][0]) for cid in resolvable]
        where, flat = _pair_where(pairs)
        span_rows = conn.execute(
            f"SELECT claim_id, span_id FROM claim_evidence WHERE {where}",
            flat,
        ).fetchall()
        spans_of: dict = {}
        span_ids: list = []
        for cid, sid in span_rows:
            spans_of.setdefault(cid, []).append(sid)
            span_ids.append(sid)
        suppressed_spans = _suppressed(store, conn, "span", span_ids)
        for cid, sids in spans_of.items():
            if set(sids) & suppressed_spans:
                resolvable.pop(cid, None)

    complete: dict = {}
    for gid in touched:
        members = all_members[gid]
        oversized = len(members) > GROUP_MEMBER_CAP
        ok = not oversized
        for cid in members:
            if cid in suppressed_claims:
                continue  # a suppressed member is gone, not missing
            if scope_of.get(cid) not in allowed_set:
                ok = False  # partly unauthorized group (V2-26.09)
                continue
            if cid not in resolvable:
                ok = False  # unresolved/unrepresentable at the cutoff
                continue
            member_meta[cid] = resolvable[cid]
        complete[gid] = ok
        groups[gid] = members[:GROUP_MEMBER_CAP]
    return cid2gid, groups, complete, member_meta


def _units(ranked_ids: list, cid2gid: dict, groups: dict,
           complete: dict) -> list:
    """Partition ranked claims into ordered atomic units.

    A unit is either an open conflict group's full closure or a singleton.
    Units keep the position of their best-ranked member, so a dispute
    travels with its highest-ranked side.
    """
    units: list = []
    index: dict = {}
    for cid in ranked_ids:
        gid = cid2gid.get(cid)
        key = ("group", gid) if gid is not None else ("claim", cid)
        if key in index:
            units[index[key]]["claims"].append(cid)
        else:
            index[key] = len(units)
            units.append({"claims": [cid], "gid": gid})
    # Closure members that never ranked join their unit, deterministically.
    ranked_set = set(ranked_ids)
    for unit in units:
        gid = unit["gid"]
        if gid is None:
            continue
        extra = [c for c in groups.get(gid, ()) if c not in ranked_set]
        unit["claims"].extend(extra)
        unit["complete"] = complete.get(gid, True)
    for unit in units:
        unit.setdefault("complete", True)
    return units


def _item_reasons(hit: Any) -> tuple:
    """Reason codes from candidate source names plus eligibility flags."""
    if hit is None:
        return ()
    codes = set(hit.source_ranks.keys())
    for source in hit.source_ranks:
        codes.add(_SRC_CODES.get(source, f"{source.upper()}_MATCH"))
    if "structured" in hit.source_ranks:
        codes.add("EXACT_SLOT")
    if "phrase_hit" in hit.reasons:
        codes.add("PHRASE_MATCH")
    if "time_compatible" in hit.reasons:
        codes.add("TIME_COMPATIBLE")
    if "time_unknown" in hit.reasons:
        codes.add("TIME_UNKNOWN")
    if "time_inapplicable" in hit.reasons:
        codes.add("TIME_INAPPLICABLE")
    applicability = getattr(hit, "applicability", None)
    # Only conditioned claims carry an explicit verdict label; an
    # unconditional claim needs no applicability code.
    if applicability in _APP_CODES and getattr(hit, "cond_keys", ()):
        codes.add(_APP_CODES[applicability])
    return tuple(sorted(codes))


def _intervals(conn: sqlite3.Connection, claim_id: str, revision: int) -> list:
    rows = conn.execute(
        "SELECT from_us, until_us, precision, timezone, basis"
        " FROM valid_intervals WHERE claim_id=? AND revision=?"
        " ORDER BY interval_no",
        (claim_id, revision),
    ).fetchall()
    return [
        TimeInterval(f, u, Precision(p), tz, b) for f, u, p, tz, b in rows
    ]


def _valid_label(intervals: list, valid_at_us: Optional[int]) -> str:
    """Label the applicable interval when known, else the first, else
    'unknown' — never a silently guessed date (SPEC §14, §31)."""
    if not intervals:
        return "unknown"
    if valid_at_us is not None:
        for iv in intervals:
            if contains(iv, valid_at_us) is True:
                return precision_label(iv)
    return precision_label(intervals[0])


def _span_payloads(conn: sqlite3.Connection, span_ids: list) -> dict:
    """span_id → (source_id, revision, start, end, payload, provenance,
    speaker_id, excerpt_hmac, payload_hmac) for exact byte reconstruction.

    The digest columns ride along so the consumer verifies the slice with
    ``_verify_slice`` before quoting — the row bytes are never trusted on
    the strength of the SQL read alone (F4-06).
    """
    if not span_ids:
        return {}
    rows = conn.execute(
        "SELECT sp.span_id, sp.source_id, sp.revision, sp.start_byte,"
        " sp.end_byte, sr.payload, sr.provenance, so.speaker_id,"
        " sp.excerpt_hmac, sr.payload_hmac"
        " FROM spans sp"
        " JOIN sources so ON so.source_id = sp.source_id"
        " LEFT JOIN source_revisions sr"
        "   ON sr.source_id = sp.source_id AND sr.revision = sp.revision"
        f" WHERE sp.span_id IN ({_ph(len(span_ids))})",
        span_ids,
    ).fetchall()
    return {r[0]: r[1:] for r in rows}


# Item-level reason codes for evidence whose bytes could not be
# digest-verified (V4-13.05, V4-13.10). ``verified`` slices carry no extra
# code — verification is the default expectation, not a label.
_VER_REASONS = {
    "legacy_unverified": "LEGACY_UNVERIFIED",
    "unverified": "UNVERIFIED",
}


def _verify_slice(store: Any, row: tuple, span_id: Optional[str] = None) -> str:
    """Digest-verify a payload row + span slice before its bytes are used.

    Re-runs the same consumption-time checks the repositories perform:
    ``SourcesRepo.payload`` re-checks ``payload_hmac`` over the revision
    bytes and ``SpansRepo.text`` re-checks ``excerpt_hmac`` over the exact
    ``payload[start:end]`` excerpt. A present-but-mismatched digest is
    corruption — ``STORE_CORRUPT``, never content (fail closed, C03).

    Returns ``"verified"`` only when both bindings were present and
    matched; a NULL digest marks a legacy row — ``"legacy_unverified"`` —
    which a surviving sibling digest can never promote to verified
    (V4-13.05, C04). A missing or purged (emptied) payload reports
    ``"unverified"``; ``_quote`` still treats it as unavailable, matching
    the repo convention that a scrubbed excerpt is absent, not corrupt.
    """
    source_id, rev, start, end, payload, _prov, _speaker, *digests = row
    excerpt_hmac = digests[0] if digests else None
    payload_hmac = digests[1] if len(digests) > 1 else None
    if payload is None:
        return "unverified"
    payload_bytes = bytes(payload)
    if payload_hmac is not None and not hmac.compare_digest(
        store.hmac(payload_bytes), bytes(payload_hmac)
    ):
        raise VerbatimError(
            ErrorCode.STORE_CORRUPT,
            f"source {source_id}@{rev} payload fails integrity check",
        )
    if not payload_bytes:
        # Purged revision: the excerpt bytes no longer exist, so there is
        # nothing left to authenticate — absent, not corrupt (SpansRepo.text
        # makes the same distinction before its digest check). The payload
        # digest above still ran first: a purge row keeps a resealed
        # hmac(b""), so a tampered purge marker is corruption, not absence.
        return "unverified"
    if excerpt_hmac is not None and not hmac.compare_digest(
        store.hmac(payload_bytes[start:end]), bytes(excerpt_hmac)
    ):
        raise VerbatimError(
            ErrorCode.STORE_CORRUPT,
            f"span {span_id or '?'} excerpt fails integrity check",
        )
    if excerpt_hmac is None or payload_hmac is None:
        return "legacy_unverified"
    return "verified"


def _quote(row: tuple, store: Any = None) -> tuple:
    """(text, span_ref, speaker, provenance, ok) from a payload row.

    Byte-exact reconstruction only: a missing payload, stale offsets, or
    undecodable bytes never produce a partial quotation (SPEC §31).

    The row may carry ``(excerpt_hmac, payload_hmac)`` as trailing
    columns; every packaging path runs ``_verify_slice`` before quoting
    (F4-06). Passing ``store`` re-checks the digests here too, so a
    consumer that cannot pre-verify still fails closed on corruption.
    """
    source_id, rev, start, end, payload, provenance, speaker_id = row[:7]
    if store is not None:
        _verify_slice(store, row)
    if payload is None or len(payload) == 0:
        return None, None, None, None, False
    if not (0 <= start < end <= len(payload)):
        return None, None, None, None, False
    try:
        text = bytes(payload[start:end]).decode("utf-8")
    except UnicodeDecodeError:
        return None, None, None, None, False
    try:
        prov = Provenance(provenance) if provenance else Provenance.UNKNOWN
    except ValueError:
        prov = Provenance.UNKNOWN
    return text, (source_id, rev, start, end), speaker_id, prov, True


def _claim_applicability(
    conn: sqlite3.Connection, claim_id: str, revision: int, context: dict
) -> tuple:
    """(verdict, required_keys) for one resolved revision (V2-17)."""
    row = conn.execute(
        "SELECT condition_json FROM claim_revisions"
        " WHERE claim_id=? AND revision=?",
        (claim_id, revision),
    ).fetchone()
    if not row or not row[0]:
        return "applies", ()
    try:
        cond = Condition.from_json(safe_json_loads(row[0]))
        value = cond.evaluate(context)
    except Exception:
        return "unknown", ()
    keys = cond.required_keys()
    return (
        "applies" if value is True
        else "does_not_apply" if value is False
        else "unknown"
    ), keys


def _claim_items(conn: sqlite3.Connection, claim_id: str, revision: int,
                 state: str, recorded_from: int, hit: Any, plan: Any,
                 disputed: bool, extra_reasons: tuple = (),
                 store: Any = None) -> tuple:
    """EvidenceItems for one resolved claim revision.

    Returns (items, evidence_unavailable). Spans whose payload was purged,
    shrank, or fails to decode are skipped — the claim contributes an item
    only from evidence that still reconstructs byte-exactly.

    When ``store`` is given, every span row is digest-verified through
    ``_verify_slice`` before quoting (F4-06): a mismatch fails closed as
    ``STORE_CORRUPT``; a NULL-digest legacy row is served but labeled
    ``LEGACY_UNVERIFIED``. Without a verifier the bytes are served but
    labeled ``UNVERIFIED`` — neither state is ever reported as verified
    (V4-13.05).
    """
    rows = conn.execute(
        "SELECT ce.span_id, ce.evidence_role, sp.source_id, sp.revision,"
        " sp.start_byte, sp.end_byte, sr.payload, sr.provenance,"
        " so.speaker_id, sp.excerpt_hmac, sr.payload_hmac"
        " FROM claim_evidence ce"
        " JOIN spans sp ON sp.span_id = ce.span_id"
        " JOIN sources so ON so.source_id = sp.source_id"
        " LEFT JOIN source_revisions sr"
        "   ON sr.source_id = sp.source_id AND sr.revision = sp.revision"
        " WHERE ce.claim_id=? AND ce.revision=?"
        " ORDER BY CASE ce.evidence_role WHEN 'primary' THEN 0 ELSE 1 END,"
        "          ce.span_id",
        (claim_id, revision),
    ).fetchall()

    lifecycle = Lifecycle(state)
    reasons = _item_reasons(hit) + tuple(extra_reasons)
    label = _valid_label(_intervals(conn, claim_id, revision), plan.valid_at_us)
    items: list = []
    unavailable = False
    span_ids: list = []
    for (span_id, _role, source_id, span_rev, start, end,
         payload, provenance, speaker_id, excerpt_hmac,
         payload_hmac) in rows:
        span_ids.append(span_id)
        row = (source_id, span_rev, start, end, payload, provenance,
               speaker_id, excerpt_hmac, payload_hmac)
        verification = (
            _verify_slice(store, row, span_id=span_id)
            if store is not None
            else "unverified"
        )
        text, span_ref, speaker, prov, ok = _quote(row)
        if not ok:
            unavailable = True
            continue
        item_reasons = reasons
        if verification != "verified":
            item_reasons = reasons + (_VER_REASONS[verification],)
        items.append(
            EvidenceItem(
                claim_id=claim_id,
                claim_revision=revision,
                text=text,
                span=SpanRef(span_id, *span_ref),
                speaker_id=speaker,
                lifecycle=lifecycle,
                valid_label=label,
                recorded_seq=recorded_from,
                reasons=item_reasons,
                provenance=prov,
                disputed=disputed or lifecycle == Lifecycle.DISPUTED,
                historical=lifecycle in (Lifecycle.SUPERSEDED, Lifecycle.ARCHIVED),
            )
        )
    if not items:
        unavailable = True
    return items, unavailable, span_ids


def _context_items(
    conn: sqlite3.Connection,
    store: Any,
    seed_span_ids: list,
    allowed_set: set,
    plan: Any,
    request: Any,
    primary: dict,
    already_emitted: set,
) -> tuple:
    """Required/optional context-group members for a unit (V2-30.01).

    Returns ``(required_items, optional_items, incomplete, saw_partial)``.
    A required member that cannot be quoted byte-exactly — purged payload,
    suppressed span, unreadable scope — makes the whole group incomplete:
    context budgets may never strip a condition/negation/attribution while
    keeping its claim as unconditional evidence (V2-17.14).
    """
    required: list = []
    optional: list = []
    incomplete = False
    saw_partial = False
    seed_spans = set(seed_span_ids)
    if not seed_span_ids or not (
        _has_table(conn, "context_members")
        and _has_table(conn, "context_groups")
    ):
        return required, optional, incomplete, saw_partial

    group_rows = conn.execute(
        "SELECT DISTINCT cm.group_id FROM context_members cm"
        f" WHERE cm.span_id IN ({_ph(len(seed_span_ids))})",
        seed_span_ids,
    ).fetchall()
    group_ids = [r[0] for r in group_rows]
    if not group_ids:
        return required, optional, incomplete, saw_partial

    known = plan.known_at_seq
    meta_rows = conn.execute(
        "SELECT group_id, scope_id, completeness, recorded_from,"
        " recorded_until FROM context_groups"
        f" WHERE group_id IN ({_ph(len(group_ids))})",
        group_ids,
    ).fetchall()
    suppressed_groups = _suppressed(store, conn, "context_group", group_ids)
    visible: dict = {}
    for gid, scope_id, completeness, rec_from, rec_until in meta_rows:
        if gid in suppressed_groups:
            continue
        if scope_id not in allowed_set:
            # A context group outside the reader's scope can never be
            # quoted; if it holds required members the unit is incomplete.
            visible[gid] = (False, completeness, rec_from)
            continue
        if known is None:
            known_ok = rec_until is None
        else:
            known_ok = rec_from <= known and (
                rec_until is None or rec_until > known
            )
        visible[gid] = (known_ok, completeness, rec_from)

    member_rows = conn.execute(
        "SELECT group_id, span_id, role, required, ord FROM context_members"
        f" WHERE group_id IN ({_ph(len(group_ids))}) ORDER BY ord, span_id",
        group_ids,
    ).fetchall()
    members: dict = {}
    member_span_ids: list = []
    for gid, span_id, role, req, _ord in member_rows:
        members.setdefault(gid, []).append((span_id, role, req))
        if span_id not in member_span_ids:
            member_span_ids.append(span_id)
    payloads = _span_payloads(conn, member_span_ids)
    suppressed_spans = _suppressed(store, conn, "span", member_span_ids)

    def _make_item(span_id: str, role: str) -> tuple:
        row = payloads.get(span_id)
        if row is None or span_id in suppressed_spans:
            return None
        # F4-06: verify the slice's persisted digests before quoting —
        # a tampered payload/span fails closed (STORE_CORRUPT); a legacy
        # NULL-digest row is labeled, never silently verified (V4-13.05).
        verification = _verify_slice(store, row, span_id=span_id)
        text, span_ref, speaker, prov, ok = _quote(row)
        if not ok:
            return None
        reasons = ("REQUIRED_CONTEXT", f"CONTEXT_{role.upper()}")
        if verification != "verified":
            reasons = reasons + (_VER_REASONS[verification],)
        return EvidenceItem(
            claim_id=primary["claim_id"],
            claim_revision=primary["revision"],
            text=text,
            span=SpanRef(span_id, *span_ref),
            speaker_id=speaker,
            lifecycle=primary["lifecycle"],
            valid_label=primary["label"],
            recorded_seq=primary["recorded_from"],
            reasons=reasons,
            provenance=prov,
            disputed=primary["disputed"],
            historical=primary["historical"],
        )

    for gid, member_list in members.items():
        shown, completeness, _rec = visible.get(gid, (False, "complete", 0))
        if completeness in ("partial", "deferred"):
            saw_partial = True
        for span_id, role, req in member_list:
            if role == "primary" and span_id not in seed_spans:
                continue
            if span_id in already_emitted:
                continue  # the claim's own evidence is already quoted
            if span_id in suppressed_spans or payloads.get(span_id) is None:
                if req:
                    incomplete = True  # required member unrepresentable
                continue
            if not shown:
                if req:
                    incomplete = True  # group not visible at cutoff/scope
                continue
            item = _make_item(span_id, role)
            if item is None:
                if req:
                    incomplete = True
                continue
            already_emitted.add(span_id)
            if req:
                required.append(item)
            else:
                optional.append(item)
    return required, optional, incomplete, saw_partial


def _kind_group_items(
    conn: sqlite3.Connection,
    store: Any,
    unit: dict,
    plan: Any,
    request: Any,
    allowed_set: set,
) -> tuple:
    """EvidenceItems for an episode/procedure/plan unit (V2-21..24).

    The unit's stored text (label, intention, step descriptions) is itself
    the quotation — it is emitted byte-identically, never summarized.
    Member claims of an episode are resolved through the same known-at and
    state rules as ordinary claims.
    """
    kind = unit["kind"]
    reasons = [f"MEMORY_KIND_{kind.upper()}", "KIND_MATCH"]
    applicability = unit.get("applicability")
    if applicability in _APP_CODES and unit.get("cond_keys"):
        reasons.append(_APP_CODES[applicability])
    lifecycle = _KIND_LIFECYCLE.get(unit.get("state"), Lifecycle.ACTIVE)
    items: list = []
    complete = True
    text = unit.get("text") or ""
    items.append(
        EvidenceItem(
            claim_id=unit["id"],
            claim_revision=unit.get("revision", 1),
            text=text,
            span=SpanRef(unit["id"], "", unit.get("revision", 1), 0,
                         len(text.encode("utf-8"))),
            speaker_id=unit.get("speaker_id"),
            lifecycle=lifecycle,
            valid_label="unknown",
            recorded_seq=unit.get("recorded_from", 0),
            reasons=tuple(sorted(set(reasons))),
            provenance=Provenance.UNKNOWN,
            disputed=False,
            historical=lifecycle in (Lifecycle.SUPERSEDED, Lifecycle.ARCHIVED),
        )
    )
    if kind == "procedure":
        step_spans = {
            s[1] for s in unit.get("steps", []) if len(s) > 1 and s[1]
        }
        payloads = _span_payloads(conn, sorted(step_spans)) if step_spans else {}
        for step in unit.get("steps", []):
            step_no, span_id, desc = step[0], step[1], step[2]
            body = payloads.get(span_id) if span_id else None
            verification = "verified"
            if body is not None:
                # F4-06: digest-verify the step span before quoting —
                # corrupt bytes fail closed, never a silent description
                # fallback over tampered evidence.
                verification = _verify_slice(store, body, span_id=span_id)
                text, span_ref, speaker, prov, ok = _quote(body)
                if not ok:
                    text, span_ref, prov = None, None, None
            else:
                text, span_ref, prov = None, None, None
            step_reasons = set(reasons) | {"PROCEDURE_STEP"}
            if text is None:
                # Fall back to the stored step description — itself exact
                # persisted text, labeled with a synthetic span ref.
                desc_text = desc or ""
                span_ref = ("", unit.get("revision", 1), 0,
                            len(desc_text.encode("utf-8")))
                text, prov = desc_text, Provenance.UNKNOWN
            elif verification != "verified":
                step_reasons.add(_VER_REASONS[verification])
            items.append(
                EvidenceItem(
                    claim_id=unit["id"],
                    claim_revision=unit.get("revision", 1),
                    text=text,
                    span=SpanRef(span_id or unit["id"], *span_ref),
                    speaker_id=None,
                    lifecycle=lifecycle,
                    valid_label="unknown",
                    recorded_seq=unit.get("recorded_from", 0),
                    reasons=tuple(sorted(step_reasons)),
                    provenance=prov,
                    disputed=False,
                    historical=False,
                )
            )
    elif kind == "episode":
        member_ids = list(unit.get("member_claim_ids", ()))
        if member_ids:
            resolved = _latest_known_revisions(
                conn, member_ids, plan.known_at_seq
            )
            suppressed = _suppressed(store, conn, "claim", member_ids)
            scope_of = dict(
                conn.execute(
                    "SELECT claim_id, scope_id FROM claims"
                    f" WHERE claim_id IN ({_ph(len(member_ids))})",
                    member_ids,
                ).fetchall()
            )
            allowed_states = _MODE_STATES[request.mode]
            shown = 0
            for cid in member_ids:
                meta = resolved.get(cid)
                if (
                    meta is None
                    or cid in suppressed
                    or scope_of.get(cid) not in allowed_set
                    or meta[1] not in allowed_states
                ):
                    complete = False  # a member existed but cannot be shown
                    continue
                rev, state, recorded_from = meta
                member_items, unavailable, _sids = _claim_items(
                    conn, cid, rev, state, recorded_from, None, plan,
                    disputed=False,
                    extra_reasons=("EPISODE_MEMBER",),
                    store=store,
                )
                if member_items:
                    items.extend(member_items)
                    shown += 1
                if shown >= GROUP_MEMBER_CAP:
                    complete = False
                    break
    return items, complete


def _group_sort_key(entry: dict) -> tuple:
    """Greedy packer order (V2-30.04): relevance tier first, then fused
    group score per serialized byte, then a stable id."""
    size = max(1, entry["cost"])
    return (
        entry["tier"],
        -(entry["score"] / size),
        -entry["score"],
        entry["primary"],
    )


def package(conn: sqlite3.Connection, store: Any, ranked: list,
            request: Any, plan: Any, generation: int,
            deadline: Optional[Deadline] = None) -> RecallResult:
    """Turn a fused ranking into a budgeted RecallResult of atomic groups.

    Budget rules (§31, V2-30): groups are atomic — all their members' items
    fit or the whole group is omitted; groups whose required context or
    conflict members cannot be represented are omitted with a scoped
    warning; omissions and purged evidence stay visible through warnings
    and the ``omitted`` count. ``limit`` and ``max_bytes`` apply to the
    flattened item stream exactly as in v1.
    """
    hits = getattr(ranked, "hits", None)
    if hits is None:
        # An empty CandidateMap is falsy: it still carries kind units and
        # gather diagnostics, so only a genuinely absent map is replaced.
        hits = {}
    warnings: list = []
    emitted_items: list = []
    emitted_groups: list = []
    omitted = 0
    used_bytes = 0
    span_total = 0
    evidence_unavailable = False
    too_large = False
    incomplete_dropped = False
    context_dropped = False
    context_omitted = False
    deadline_exceeded = bool(getattr(hits, "deadline_exceeded", False))

    context = getattr(request, "context", None) or {}
    ranked_ids = [cid for cid, _score in ranked]
    score_of = dict(ranked)
    allowed = _allowed_scopes(conn, request.scope)
    allowed_set = set(allowed)

    pairs = [
        (cid, hits[cid].claim_revision) for cid in ranked_ids if cid in hits
    ]
    meta = _revision_meta(conn, pairs)
    cid2gid, groups, complete_map, member_meta = _conflict_closure(
        conn, store, ranked_ids, allowed_set, plan, request
    )
    units = _units(ranked_ids, cid2gid, groups, complete_map)

    # -- build every candidate group's items before packing -------------
    built: list = []
    stopped = False
    for seq, unit in enumerate(units):
        if stopped:
            omitted += len(unit["claims"])
            continue
        if deadline is not None and deadline.expired():
            stopped = True
            deadline_exceeded = True
            omitted += len(unit["claims"])
            continue
        unit_items: list = []
        unit_spans: set = set()
        failed = 0
        disputed = len(unit["claims"]) > 1
        for cid in unit["claims"]:
            hit = hits.get(cid)
            if hit is not None:
                revision = hit.claim_revision
                state_rf = meta.get((cid, revision))
                extras: tuple = (
                    ("UNRESOLVED_CONFLICT",) if disputed else ()
                )
            elif cid in member_meta:
                revision, state, recorded_from = member_meta[cid]
                state_rf = (state, recorded_from)
                verdict, keys = _claim_applicability(
                    conn, cid, revision, context
                )
                extras = ("UNRESOLVED_CONFLICT", "CONFLICT_MEMBER")
                if keys and verdict in _APP_CODES:
                    extras = extras + (_APP_CODES[verdict],)
            else:
                failed += 1
                continue
            if state_rf is None:
                failed += 1
                continue
            state, recorded_from = state_rf
            items, unavailable, span_ids = _claim_items(
                conn, cid, revision, state, recorded_from, hit, plan,
                disputed=disputed, extra_reasons=extras, store=store,
            )
            evidence_unavailable = evidence_unavailable or unavailable
            if not items:
                failed += 1
            unit_items.extend(items)
            unit_spans.update(span_ids)

        # Required context members ride inside the same atomic group.
        first_cid = unit["claims"][0] if unit["claims"] else ""
        first_hit = hits.get(first_cid)
        primary_meta = None
        if first_hit is not None:
            primary_meta = meta.get((first_cid, first_hit.claim_revision))
        elif first_cid in member_meta:
            rev, st, rf = member_meta[first_cid]
            primary_meta = (st, rf)
        group_complete = unit["complete"] is not False
        if primary_meta is not None and unit_items:
            p_state, p_rec = primary_meta
            primary = {
                "claim_id": first_cid,
                "revision": unit_items[0].claim_revision,
                "lifecycle": unit_items[0].lifecycle,
                "label": unit_items[0].valid_label,
                "recorded_from": p_rec,
                "disputed": unit_items[0].disputed,
                "historical": unit_items[0].historical,
            }
            req_items, opt_items, incomplete, partial = _context_items(
                conn, store, sorted(unit_spans), allowed_set, plan, request,
                primary, {i.span.span_id for i in unit_items},
            )
            context_omitted = context_omitted or partial
            group_complete = group_complete and not partial
            # Optional members join the group only when the group's required
            # content already fits with room to spare; a required member
            # that failed makes the whole group undeliverable.
            unit_items.extend(req_items)
            room = request.max_bytes - used_bytes - sum(
                len(serialize_item(i)) for i in unit_items
            )
            keep_opt = []
            opt_dropped = False
            for item in opt_items:
                size = len(serialize_item(item))
                if len(unit_items) + len(keep_opt) < request.limit \
                        and size <= room:
                    keep_opt.append(item)
                    room -= size
                else:
                    opt_dropped = True
            if opt_dropped:
                context_omitted = True
                group_complete = False
            unit_items.extend(keep_opt)
        else:
            incomplete = unit["complete"] is False

        group_reasons: list = []
        if disputed:
            group_reasons.append("UNRESOLVED_CONFLICT")
        if any("REQUIRED_CONTEXT" in i.reasons for i in unit_items):
            group_reasons.append("REQUIRED_CONTEXT")
        best = max(
            (score_of.get(cid, 0.0) for cid in unit["claims"]), default=0.0
        )
        tier = 0
        for cid in unit["claims"]:
            h = hits.get(cid)
            if h is not None and (
                "structured" in h.source_ranks or "phrase_hit" in h.reasons
            ):
                tier = 0
                break
            tier = 1
        # Packing cost is the serialized *item stream* (the v1 contract:
        # max_bytes bounds emitted evidence bytes). ``serialized_bytes`` on
        # the emitted group still reports its full envelope honestly.
        cost = sum(len(serialize_item(i)) for i in unit_items)
        built.append({
            "seq": seq,
            "primary": first_cid,
            "items": unit_items,
            "claims": len(unit["claims"]),
            "failed": failed,
            "cost": cost,
            "tier": tier,
            "score": best,
            "complete": group_complete,
            "undeliverable": incomplete or unit["complete"] is False,
            "undeliverable_kind": (
                "conflict" if unit["complete"] is False
                else "context" if incomplete
                else None
            ),
            "reasons": tuple(sorted(set(group_reasons))),
            "kind": "claims",
        })

    # -- non-claim memory-kind units -------------------------------------
    kind_seq = len(built)
    for unit in getattr(hits, "kind_units", []) or []:
        if deadline is not None and deadline.expired():
            stopped = True
            deadline_exceeded = True
            omitted += 1
            continue
        items, complete = _kind_group_items(
            conn, store, unit, plan, request, allowed_set
        )
        cost = sum(len(serialize_item(i)) for i in items)
        built.append({
            "seq": kind_seq,
            "primary": unit["id"],
            "items": items,
            "claims": 1,
            "failed": 0,
            "cost": cost,
            "tier": 1,
            "score": 0.0,
            "complete": complete,
            "undeliverable": not items,
            "reasons": (f"MEMORY_KIND_{unit['kind'].upper()}",),
            "kind": unit["kind"],
        })
        kind_seq += 1

    # -- greedy atomic packing -------------------------------------------
    for entry in sorted(built, key=_group_sort_key):
        n_items = len(entry["items"])
        fits = (
            n_items > 0
            and not entry["undeliverable"]
            and len(emitted_items) + n_items <= request.limit
            and used_bytes + entry["cost"] <= request.max_bytes
            and span_total + n_items <= SPAN_CAP
        )
        if fits:
            emitted_items.extend(entry["items"])
            used_bytes += entry["cost"]
            span_total += n_items
            emitted_groups.append(
                EvidenceGroup(
                    primary_claim_id=entry["primary"],
                    items=tuple(entry["items"]),
                    complete=entry["complete"],
                    reasons=entry["reasons"],
                    group_score=entry["score"],
                    serialized_bytes=0,
                )
            )
        else:
            omitted += max(n_items, entry["claims"])
            if entry["undeliverable"]:
                if entry.get("undeliverable_kind") == "conflict":
                    incomplete_dropped = True
                else:
                    context_dropped = True
            elif n_items:
                too_large = True

    # Report each emitted group's real serialized envelope size (the field
    # is informational: the packing budget bounds the item stream, and a
    # self-referential byte count can never be exact).
    emitted_groups = [
        EvidenceGroup(
            primary_claim_id=g.primary_claim_id,
            items=g.items,
            complete=g.complete,
            reasons=g.reasons,
            group_score=g.group_score,
            serialized_bytes=len(serialize_group(g)),
        )
        for g in emitted_groups
    ]

    if evidence_unavailable:
        warnings.append("evidence_unavailable")
    if any("LEGACY_UNVERIFIED" in item.reasons for item in emitted_items):
        # A NULL-digest legacy row was served — reported, never silently
        # presented as verified evidence (V4-13.05, C04).
        warnings.append("legacy_unverified")
    if any("UNVERIFIED" in item.reasons for item in emitted_items):
        warnings.append("unverified")
    if any(item.disputed for item in emitted_items):
        warnings.append("unresolved_conflict")
    if any("TIME_UNKNOWN" in item.reasons for item in emitted_items):
        warnings.append("time_unknown")
    if any("APPLICABILITY_UNKNOWN" in item.reasons for item in emitted_items):
        warnings.append("APPLICABILITY_UNKNOWN")
    if incomplete_dropped:
        warnings.append("conflict_group_incomplete")
    if context_dropped:
        warnings.append("context_incomplete")
    if context_omitted:
        warnings.append("context_omitted")
    if too_large:
        warnings.append("EVIDENCE_TOO_LARGE")
    if too_large and not emitted_items:
        # The caller's hard bound could not hold even the smallest group —
        # say so instead of silently returning an ordinary empty result.
        warnings.append("BUDGET_TOO_SMALL")
    if deadline_exceeded or stopped:
        warnings.append("DEADLINE_EXCEEDED")

    notes = dict(getattr(hits, "capability_notes", {}) or {})
    missing_keys = sorted({
        key
        for _cid, hit in hits.items()
        if getattr(hit, "applicability", None) == "unknown"
        for key in getattr(hit, "cond_keys", ())
    })
    if missing_keys:
        # Clarification hints (V2-17.15): the context keys whose absence
        # kept conditioned evidence in the unknown lane.
        notes["missing_context_keys"] = missing_keys
    capabilities = {
        "semantic": bool(getattr(hits, "semantic_active", False)),
        "rerank": False,
        "degraded": list(getattr(hits, "degraded", []) or []),
        "notes": notes,
        "reports": _capability_reports(conn, store, hits),
    }
    overflow = getattr(hits, "overflow", 0)
    omitted += overflow
    if overflow:
        warnings.append("candidate_overflow")

    return RecallResult(
        items=tuple(emitted_items),
        groups=tuple(emitted_groups),
        omitted=omitted,
        warnings=tuple(warnings),
        capabilities=capabilities,
        projection_generation=generation,
    )


def _capability_reports(conn: sqlite3.Connection, store: Any,
                        hits: Any) -> list:
    """Per-capability truthfulness (V2-05): every source that ran reports
    its real state; degraded/absent optional lanes say so explicitly.

    States: ``healthy`` — ran at full capability; ``degraded`` — ran on a
    reduced-quality path, or covered a healthy-but-empty index;
    ``unavailable`` — the capability itself failed or is absent;
    ``skipped`` — the lane was not exercised by this request.
    """
    degraded = set(getattr(hits, "degraded", []) or [])
    warnings = set(getattr(hits, "warnings", []) or [])
    semantic_active = bool(getattr(hits, "semantic_active", False))

    # Lexical: the bounded substring fallback taken on an FTS5-absent
    # store returns real rows without marking ``degraded`` — detect the
    # same absence the lane consulted so the report cannot claim
    # "healthy" over reduced-quality matching (Store.status reports the
    # same gap).
    fts_missing = (
        getattr(store, "fts_enabled", True) is False
        or not _fts5_available(conn)
    )
    if "lexical" in degraded or "lexical_unavailable" in warnings:
        lex_state, lex_reason = "unavailable", "lexical lane failed"
    elif fts_missing:
        lex_state = "degraded"
        lex_reason = "fts5 index unavailable; bounded substring fallback"
    else:
        lex_state, lex_reason = "healthy", None

    # Semantic: an explicit ``semantic_unavailable`` warning means the
    # capability broke (no encoder, ambiguous identity, failed encode);
    # degraded WITHOUT the warning means the lane ran over an empty
    # embeddings table — coverage is empty, the capability is intact.
    if semantic_active:
        sem_state, sem_reason = "healthy", None
    elif "semantic" in degraded:
        if "semantic_unavailable" in warnings:
            sem_state = "unavailable"
            sem_reason = "no embedding backend or encoder configured"
        else:
            sem_state = "degraded"
            sem_reason = "no embeddings indexed"
    else:
        sem_state = "skipped"
        sem_reason = (
            "deadline stopped the pipeline before the semantic lane ran"
            if getattr(hits, "deadline_exceeded", False)
            else "semantic lane not requested"
        )

    reports = [
        {"name": "lexical", "state": lex_state, "reason": lex_reason},
        {"name": "semantic", "state": sem_state, "reason": sem_reason},
        {"name": "rerank", "state": "unavailable",
         "reason": "no reranker configured"},
    ]
    # Remaining degraded lanes: a ``*_unavailable`` warning means the
    # lane broke; degraded with no warning means it ran and legitimately
    # found nothing — empty coverage, not a broken capability.
    for name in sorted(degraded - {"lexical", "semantic"}):
        if f"{name}_unavailable" in warnings:
            reports.append({"name": name, "state": "unavailable",
                            "reason": f"{name} lane unavailable"})
        else:
            reports.append({"name": name, "state": "degraded",
                            "reason": f"{name} lane returned no candidates"})
    return reports
