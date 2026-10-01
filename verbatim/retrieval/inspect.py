"""Claim inspection: exact evidence + interpretation lineage (SPEC §31, §35,
SPEC_V2 §26, §30).

Returns safe structured metadata — no hidden prompts, secrets, or unrelated
context. Unknown IDs and access-denied are indistinguishable (SPEC §9):
authorization is checked before any detail is read, and suppressed claims
report the same ``NOT_FOUND_OR_FORBIDDEN`` as ones that never existed.
"""

from __future__ import annotations

from typing import Any

from ..core.identity import can_read
from ..core.time import rfc3339
from ..core.types import (
    Condition,
    ErrorCode,
    Scope,
    VerbatimError,
    Visibility,
    safe_json_loads,
)
from ..storage.repos import has_table as _has_table


def inspect_claim(store: Any, claim_id: str, scope: Scope) -> dict[str, Any]:
    with store.read() as conn:
        claim = conn.execute(
            "SELECT claim_id, scope_id, subject_id, predicate, created_event, row_version "
            "FROM claims WHERE claim_id = ?",
            (claim_id,),
        ).fetchone()
        if claim is None:
            raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "unknown claim")
        (
            _cid,
            claim_scope_id,
            subject_id,
            predicate,
            _created_event,
            row_version,
        ) = claim
        owner_row = conn.execute(
            "SELECT profile_id, principal_id, workspace_id, conversation_id, visibility "
            "FROM scopes WHERE scope_id = ?",
            (claim_scope_id,),
        ).fetchone()
        if owner_row is None:
            raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "unknown claim")
        owner = Scope(
            profile_id=owner_row[0],
            principal_id=owner_row[1],
            workspace_id=owner_row[2],
            conversation_id=owner_row[3],
            visibility=Visibility(owner_row[4]),
        )
        if not can_read(scope, owner):
            raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "unknown claim")

        revisions = conn.execute(
            "SELECT revision, state, polarity, modality, condition_json, "
            "interpretation_json, recorded_from, recorded_until "
            "FROM claim_revisions WHERE claim_id = ? ORDER BY revision",
            (claim_id,),
        ).fetchall()
        intervals = conn.execute(
            "SELECT revision, from_us, until_us, precision, timezone, basis "
            "FROM valid_intervals WHERE claim_id = ? ORDER BY revision, interval_no",
            (claim_id,),
        ).fetchall()
        evidence = conn.execute(
            "SELECT ce.revision, ce.span_id, ce.evidence_role, s.source_id, s.revision, "
            "s.start_byte, s.end_byte FROM claim_evidence ce "
            "JOIN spans s ON s.span_id = ce.span_id WHERE ce.claim_id = ? ORDER BY ce.revision",
            (claim_id,),
        ).fetchall()
        edges = conn.execute(
            "SELECT edge_type, source_id, target_id, decision_id FROM edges "
            "WHERE (source_id = ? OR target_id = ?) AND retired_event IS NULL",
            (claim_id, claim_id),
        ).fetchall()

        # ---- v2 relations (all guarded: absent on v1 stores) ----------
        context_groups = _context_groups(conn, evidence)
        conflict_groups = _conflict_groups(conn, claim_id)
        episodes = _episodes(conn, claim_id)

        # ---- held-evidence cascade (V3-14.10) --------------------------
        # A span's excerpt is withheld when the span itself is
        # tombstoned, or a quarantine hold covers the span, its source
        # revision, or any source envelope for that revision — the report
        # keeps the evidence row but its text renders the same
        # "[unavailable]" it already uses for purged excerpts.
        held_spans: set = set(
            _suppressed(store, "span", [e[1] for e in evidence])
        )
        if _has_table(conn, "quarantine") and evidence:
            try:
                from ..security import should_exclude as _should_exclude
            except Exception:
                _should_exclude = None

            def _held(kind: str, oid: str, rev: int) -> bool:
                if _should_exclude is not None:
                    try:
                        return bool(_should_exclude(conn, kind, oid, rev))
                    except Exception:
                        pass  # fall back to the local table check
                try:
                    row = conn.execute(
                        "SELECT state FROM quarantine"
                        " WHERE object_kind = ? AND object_id = ? AND revision = ?",
                        (kind, oid, rev),
                    ).fetchone()
                except Exception:
                    return True  # no readable quarantine state — fail closed
                return row is not None and row[0] in ("pending", "suppressed")

            env_of: dict = {}
            if _has_table(conn, "source_envelopes"):
                for src, srev in {(e[3], e[4]) for e in evidence}:
                    env_of[(src, srev)] = [
                        r[0]
                        for r in conn.execute(
                            "SELECT envelope_id FROM source_envelopes"
                            " WHERE source_id = ? AND revision = ?",
                            (src, srev),
                        ).fetchall()
                    ]
            for _r, span_id, _role, source_id, src_rev, _sb, _eb in evidence:
                if span_id in held_spans:
                    continue
                if (
                    _held("span", span_id, src_rev)
                    or _held("source", source_id, src_rev)
                    or any(
                        _held("source_envelope", eid, src_rev)
                        for eid in env_of.get((source_id, src_rev), ())
                    )
                ):
                    held_spans.add(span_id)

    suppressed = _suppressed(store, "claim", [claim_id])
    if claim_id in suppressed:
        raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "unknown claim")

    items = []
    for rev, span_id, role, source_id, src_rev, start_b, end_b in evidence:
        text = None if span_id in held_spans else _span_text(store, span_id)
        items.append({
            "span_id": span_id,
            "role": role,
            "source_id": source_id,
            "source_revision": src_rev,
            "start_byte": start_b,
            "end_byte": end_b,
            "text": text if text is not None else "[unavailable]",
        })

    revision_reports = []
    for r in revisions:
        applicability = _applicability(r[4])
        revision_reports.append(
            {
                "revision": r[0],
                "state": r[1],
                "polarity": r[2],
                "modality": r[3],
                "condition": r[4],
                "recorded_from": r[6],
                "recorded_until": r[7],
                # Three-valued under an empty context (V2-17): a claim with
                # no condition applies; a conditioned claim reports the
                # context keys it needs rather than silently satisfying.
                "applicability": applicability[0],
                "condition_keys": applicability[1],
            }
        )

    return {
        "claim_id": claim_id,
        "subject_id": subject_id,
        "predicate": predicate,
        "row_version": row_version,
        "revisions": revision_reports,
        "valid_intervals": [
            {
                "revision": i[0],
                "from": rfc3339(i[1]) if i[1] is not None else None,
                "until": rfc3339(i[2]) if i[2] is not None else None,
                "precision": i[3],
                "basis": i[5],
            }
            for i in intervals
        ],
        "evidence": items,
        "edges": [
            {"type": e[0], "source": e[1], "target": e[2], "decision": e[3]}
            for e in edges
        ],
        "context_groups": context_groups,
        "conflict_groups": conflict_groups,
        "episodes": episodes,
    }


def _applicability(condition_json: Any) -> tuple:
    """(verdict, required_keys) for a stored condition under no context."""
    if not condition_json:
        return "applies", []
    try:
        cond = Condition.from_json(safe_json_loads(condition_json))
        value = cond.evaluate({})
    except Exception:
        return "unknown", []
    keys = list(cond.required_keys())
    return (
        "applies" if value is True
        else "does_not_apply" if value is False
        else "unknown"
    ), keys


def _context_groups(conn: Any, evidence: list) -> list:
    """Context-group memberships covering this claim's evidence spans."""
    if not (
        _has_table(conn, "context_members")
        and _has_table(conn, "context_groups")
    ):
        return []
    span_ids = sorted({e[1] for e in evidence})
    if not span_ids:
        return []
    ph = ",".join("?" * len(span_ids))
    rows = conn.execute(
        "SELECT DISTINCT cm.group_id FROM context_members cm"
        f" WHERE cm.span_id IN ({ph})",
        span_ids,
    ).fetchall()
    group_ids = [r[0] for r in rows]
    if not group_ids:
        return []
    ph2 = ",".join("?" * len(group_ids))
    groups = {
        r[0]: r[1:]
        for r in conn.execute(
            "SELECT group_id, source_id, revision, completeness,"
            " recorded_from, recorded_until FROM context_groups"
            f" WHERE group_id IN ({ph2})",
            group_ids,
        ).fetchall()
    }
    members = conn.execute(
        "SELECT group_id, span_id, role, required, ord FROM context_members"
        f" WHERE group_id IN ({ph2}) ORDER BY group_id, ord, span_id",
        group_ids,
    ).fetchall()
    out: dict = {}
    for gid, span_id, role, required, _ord in members:
        entry = out.setdefault(gid, [])
        entry.append(
            {
                "span_id": span_id,
                "role": role,
                "required": bool(required),
            }
        )
    return [
        {
            "group_id": gid,
            "source_id": meta[0],
            "revision": meta[1],
            "completeness": meta[2],
            "recorded_from": meta[3],
            "recorded_until": meta[4],
            "members": out.get(gid, []),
        }
        for gid, meta in sorted(groups.items())
    ]


def _conflict_groups(conn: Any, claim_id: str) -> list:
    """Conflict groups this claim belongs to, with all sibling members."""
    if not (
        _has_table(conn, "conflict_members")
        and _has_table(conn, "conflict_groups")
    ):
        return []
    rows = conn.execute(
        "SELECT g.group_id, g.status FROM conflict_groups g"
        " JOIN conflict_members cm ON cm.group_id = g.group_id"
        " WHERE cm.claim_id = ? ORDER BY g.group_id",
        (claim_id,),
    ).fetchall()
    if not rows:
        return []
    out = []
    for gid, status in rows:
        siblings = conn.execute(
            "SELECT claim_id FROM conflict_members WHERE group_id = ?"
            " ORDER BY claim_id",
            (gid,),
        ).fetchall()
        out.append(
            {
                "group_id": gid,
                "status": status,
                "members": [s[0] for s in siblings],
            }
        )
    return out


def _episodes(conn: Any, claim_id: str) -> list:
    """Episode memberships where this claim is a member object."""
    if not (
        _has_table(conn, "episode_members") and _has_table(conn, "episodes")
    ):
        return []
    rows = conn.execute(
        "SELECT em.episode_id, em.ord, em.recorded_from, em.recorded_until,"
        " e.kind, e.label, e.revision FROM episode_members em"
        " JOIN episodes e ON e.episode_id = em.episode_id"
        " WHERE em.object_kind = 'claim' AND em.object_id = ?"
        " ORDER BY em.episode_id",
        (claim_id,),
    ).fetchall()
    return [
        {
            "episode_id": r[0],
            "ord": r[1],
            "recorded_from": r[2],
            "recorded_until": r[3],
            "kind": r[4],
            "label": r[5],
            "episode_revision": r[6],
        }
        for r in rows
    ]


#: Purge states that suppress retrieval — mirrors the constant the
#: candidate path uses; a purge that passed preview removes the object
#: from every read surface.
_SUPPRESSING_STATES = ("suppressed", "purging", "completed")


def _suppressed(store: Any, kind: str, ids: list[str]) -> set[str]:
    """Suppressed object ids of one kind — fail CLOSED (SPEC §15, §40).

    The result gates whether a claim's full evidence detail may be
    shown, so an error can never default to "not suppressed". The repo
    is preferred; an absent module/class falls back to the same
    parameterized SQL the candidate path uses; any remaining failure
    raises ``NOT_FOUND_OR_FORBIDDEN`` so an unverifiable claim stays
    indistinguishable from one that never existed.
    """
    wanted = list(dict.fromkeys(ids))
    if not wanted:
        return set()
    try:
        from ..storage.repos import PurgesRepo
    except (ImportError, AttributeError):
        PurgesRepo = None  # module/class absent — equivalent SQL below
    if PurgesRepo is not None:
        try:
            return set(PurgesRepo(store).suppressed_ids(kind, wanted))
        except Exception:
            pass  # fall through to the equivalent SQL lookup
    ph = ",".join("?" * len(wanted))
    states = ",".join("?" * len(_SUPPRESSING_STATES))
    try:
        with store.read() as conn:
            rows = conn.execute(
                "SELECT DISTINCT pt.object_id FROM purge_targets pt"
                " JOIN purges p ON p.purge_id = pt.purge_id"
                " WHERE pt.object_kind = ?"
                f" AND p.state IN ({states})"
                f" AND pt.object_id IN ({ph})",
                [kind, *_SUPPRESSING_STATES, *wanted],
            ).fetchall()
    except Exception as exc:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_FORBIDDEN, "unknown claim"
        ) from exc
    return {r[0] for r in rows}


def _span_text(store: Any, span_id: str) -> str | None:
    """Decoded span text; None when the span/revision is absent.

    ``SpansRepo.text`` already re-verifies the excerpt digest — a typed
    error (``STORE_CORRUPT``, ``EVIDENCE_UNAVAILABLE``) must reach the
    inspect surface, not render as silent absence (F4-06).
    """
    from ..storage.repos import SpansRepo

    return SpansRepo(store).text(span_id)
