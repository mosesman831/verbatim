"""Suppression/quarantine cascade for projection builds (SPEC_V4 §36,
§43.09, §47.02).

A projection build reads through the SAME withholding rules the export
bundle and the retrieval union use — a regenerated projection can never
resurrect suppressed or quarantined content. The cascade is deliberately
identical to ``verbatim.export``'s:

* purge tombstones (``PurgesRepo.suppressed_ids``) on the object itself,
  its spans, its sources, and the exact ``source_revision`` keys
  (``"<source_id>:<revision>"``);
* quarantine holds (``security.quarantine``) on the object revision, its
  evidence spans, the spans' source revisions, and covering
  ``source_envelopes`` rows;
* an emptied (purged) payload reads as absent.

Everything fails closed: a lookup error withholds the object rather than
leaking it into a file a recipient will carry away.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable, Optional

from ..security import quarantine as _quarantine
from ..storage.repos import PurgesRepo, has_table as _has_table


def suppressed(
    purges: PurgesRepo, kind: str, ids: Iterable[str]
) -> set[str]:
    """Fail-closed tombstone set — mirrors ``export._suppressed``."""
    try:
        return purges.suppressed_ids(kind, ids)
    except Exception:
        return set(ids)


def held(conn: sqlite3.Connection, kind: str, oid: str, rev: int) -> bool:
    """Quarantine hold check that fails closed on unreadable state."""
    try:
        return bool(_quarantine.should_exclude(conn, kind, oid, rev))
    except Exception:
        return True


def source_revision_withheld(
    conn: sqlite3.Connection,
    source_id: str,
    revision: int,
    *,
    suppressed_rev_keys: Optional[set[str]] = None,
    suppressed_sources: Optional[set[str]] = None,
    held_refs: Optional[set[tuple[str, str, int]]] = None,
) -> bool:
    """True when a source revision's bytes must not ship (V3-14.10).

    Withheld when the source or exact revision is purge-tombstoned, the
    revision row itself is quarantined, any span carved from it is held
    (a projection cannot splice out a held byte range — the file ships
    whole objects), or a covering ``source_envelopes`` row is held.

    The ``*_refs``/``suppressed_*`` parameters let callers pass
    pre-collected batch results; omitted sets are resolved on ``conn``.
    """
    if suppressed_sources and source_id in suppressed_sources:
        return True
    if suppressed_rev_keys and f"{source_id}:{revision}" in suppressed_rev_keys:
        return True
    if held_refs is not None:
        if ("source", source_id, revision) in held_refs:
            return True
    elif held(conn, "source", source_id, revision):
        return True
    span_rows = conn.execute(
        "SELECT span_id FROM spans WHERE source_id = ? AND revision = ?",
        (source_id, revision),
    ).fetchall()
    for (span_id,) in span_rows:
        if held_refs is not None:
            if ("span", span_id, revision) in held_refs:
                return True
        elif held(conn, "span", span_id, revision):
            return True
    if not _has_table(conn, "source_envelopes"):
        return False
    env_rows = conn.execute(
        "SELECT envelope_id FROM source_envelopes"
        " WHERE source_id = ? AND revision = ?",
        (source_id, revision),
    ).fetchall()
    for (eid,) in env_rows:
        if held_refs is not None:
            if ("source_envelope", eid, revision) in held_refs:
                return True
        elif held(conn, "source_envelope", eid, revision):
            return True
    return False


def collect_withholding(
    conn: sqlite3.Connection,
    store: Any,
    scope_id: str,
) -> dict[str, Any]:
    """Batch withholding snapshot for one scope.

    Assembles the claim/evidence/source rows plus the suppression and
    quarantine verdicts, all inside the caller's snapshot ``conn`` (the
    repo helpers join it when it is already the reader transaction).
    """
    purges = PurgesRepo(store)
    claims = conn.execute(
        "SELECT c.claim_id, cr.revision, cr.state, cr.object_json,"
        " cr.recorded_from, cr.recorded_until, c.subject_id, c.predicate"
        " FROM claims c"
        " JOIN claim_revisions cr ON cr.claim_id = c.claim_id"
        "   AND cr.recorded_until IS NULL"
        " WHERE c.scope_id = ? ORDER BY c.claim_id, cr.revision",
        (scope_id,),
    ).fetchall()
    claim_ids = [r[0] for r in claims]
    all_revs = conn.execute(
        "SELECT cr.claim_id, cr.revision FROM claim_revisions cr"
        " JOIN claims c ON c.claim_id = cr.claim_id WHERE c.scope_id = ?",
        (scope_id,),
    ).fetchall()

    ev_rows = conn.execute(
        "SELECT ce.claim_id, ce.revision, ce.span_id, ce.evidence_role,"
        " sp.source_id, sp.revision, sp.start_byte, sp.end_byte"
        " FROM claim_evidence ce"
        " JOIN spans sp ON sp.span_id = ce.span_id"
        " WHERE ce.claim_id IN (SELECT claim_id FROM claims WHERE scope_id = ?)"
        " ORDER BY ce.claim_id, ce.span_id",
        (scope_id,),
    ).fetchall()
    span_ids = sorted({r[2] for r in ev_rows})
    src_revs: set[tuple[str, int]] = {(r[4], int(r[5])) for r in ev_rows}

    sources = conn.execute(
        "SELECT source_id, origin, external_id, source_kind, speaker_id,"
        " created_us FROM sources WHERE scope_id = ? ORDER BY source_id",
        (scope_id,),
    ).fetchall()
    src_ids = [r[0] for r in sources]
    rev_meta = conn.execute(
        "SELECT sr.source_id, sr.revision, sr.event_us, sr.provenance,"
        " (sr.payload IS NOT NULL AND length(sr.payload) > 0),"
        " COALESCE(length(sr.payload), 0)"
        " FROM source_revisions sr"
        " JOIN sources s ON s.source_id = sr.source_id"
        " WHERE s.scope_id = ? ORDER BY sr.source_id, sr.revision",
        (scope_id,),
    ).fetchall()
    for r in rev_meta:
        src_revs.add((r[0], int(r[1])))

    env_rows: list[tuple[str, str, int]] = []
    if _has_table(conn, "source_envelopes"):
        env_rows = [
            (r[0], r[1], int(r[2]))
            for r in conn.execute(
                "SELECT envelope_id, source_id, revision"
                " FROM source_envelopes WHERE scope_id = ?",
                (scope_id,),
            ).fetchall()
        ]

    supp_claims = suppressed(purges, "claim", claim_ids)
    supp_sources = suppressed(purges, "source", src_ids)
    supp_spans = suppressed(purges, "span", span_ids)
    supp_rev_keys = suppressed(
        purges, "source_revision", [f"{s}:{r}" for s, r in sorted(src_revs)]
    )

    held_refs: set[tuple[str, str, int]] = set()
    if _has_table(conn, "quarantine"):
        triples: set[tuple[str, str, int]] = set()
        triples.update(("claim", cid, int(rev)) for cid, rev in all_revs)
        triples.update(("span", sp, rev) for (sp, rev) in
                       {(r[2], int(r[5])) for r in ev_rows})
        triples.update(("source", s, r) for s, r in src_revs)
        triples.update(("source_envelope", e, r) for e, _s, r in env_rows)
        try:
            held_refs = _quarantine.excluded_refs(conn, triples)
        except Exception:
            # Fail closed: an unreadable quarantine table withholds
            # everything rather than leaking held content into a file.
            held_refs = triples

    return {
        "claims": claims,
        "all_revs": all_revs,
        "evidence": ev_rows,
        "sources": sources,
        "rev_meta": rev_meta,
        "envelopes": env_rows,
        "supp_claims": supp_claims,
        "supp_sources": supp_sources,
        "supp_spans": supp_spans,
        "supp_rev_keys": supp_rev_keys,
        "held_refs": held_refs,
    }


def claim_withheld(claim_id: str, wh: dict[str, Any]) -> bool:
    """True when the claim must be omitted (hold cascade, V3-14.10).

    Withholds when the claim is tombstoned, any of its revisions is
    quarantined, or any cited span / its source revision / a covering
    envelope is suppressed or held — the claim's text carries the same
    content the hold covers.
    """
    if claim_id in wh["supp_claims"]:
        return True
    held = wh["held_refs"]
    if any(k == "claim" and oid == claim_id for k, oid, _r in held):
        return True
    # Held envelope ids → the (source, revision) they cover.
    held_env_pairs = {
        (src, int(rev))
        for eid, src, rev in wh["envelopes"]
        if ("source_envelope", eid, int(rev)) in held
    }
    for _cid, _rev, span_id, _role, src, srev, _st, _en in wh["evidence"]:
        if _cid != claim_id:
            continue
        if span_id in wh["supp_spans"]:
            return True
        if ("span", span_id, int(srev)) in held:
            return True
        if src in wh["supp_sources"]:
            return True
        if f"{src}:{int(srev)}" in wh["supp_rev_keys"]:
            return True
        if ("source", src, int(srev)) in held:
            return True
        if (src, int(srev)) in held_env_pairs:
            return True
    return False


__all__ = [
    "claim_withheld",
    "collect_withholding",
    "held",
    "source_revision_withheld",
    "suppressed",
]
