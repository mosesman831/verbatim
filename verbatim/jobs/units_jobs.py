"""V7 unit-projection write path (SPEC_V7 §06/§08/§12/§13, V7-06.05).

``source_project`` owns the byte-exact source/revision rows; this module
writes the V7 artifacts derived from them inside the same
generation-fenced transaction.  One source row — the persisted
``sources`` row plus its ``source_revisions`` metadata and the verified
payload text — becomes, deterministically:

- ``units``                 — V7-06.05 (``projections/units_v7.py``)
- ``unit_fts_rows`` + ``unit_fts_content``
                            — five-field content row whose FTS5 triggers
                              mirror into ``unit_fts`` / ``unit_fts_stem``
                              / ``unit_fts_tri`` (V7-06.02)
- ``entity_mentions`` + ``entity_canon``
                            — capitals/identifier extraction + the scope
                              vocabulary with recomputed ``df_units``
                              (V7-08.03)
- ``entity_aliases_v7``     — conservative alias proposals (A1–A3 may be
                              active; A4/A5/A6 stay candidate; review
                              decisions are never overwritten)
- ``events_v7``             — typed temporal facts with byte pins
                              (V7-08.04/§12); V8-09.07 adds the
                              ``subject_source`` provenance column and
                              backfills first-person-pronoun subjects
                              from the unit's ``speaker_canon``
- ``unit_time_mentions``    — write-time relative-time mentions resolved
                              on the unit's own occurred anchor
                              (V8-09.04/§21.6, ``enrichment/reltime``)
- ``unit_doclen``           — maintained BM25F field lengths per unit
                              (V8-07.04/§19)
- ``state_facts`` / ``preferences``
                            — optional, via ``enrichment/prefs_state``
                              when that module ships (honest ``skipped``
                              stat otherwise — V7-10.01)
- ``lex_stats`` / ``lex_df`` — corpus statistics maintained in the same
                              transaction (V7-06.06, ``stats_v7``)

Byte grounding (V7-06.01): every unit's ``byte_start``/``byte_end`` are
UTF-8 byte offsets into the persisted payload.  The FTS ``text`` field is
*exactly* ``payload[byte_start:byte_end].decode("utf-8")`` — the same
slice a pack will later quote.  A unit whose pins cannot be resolved is
still written (metadata is real) but receives no FTS row and is counted
under ``stats["unpinned"]`` — nothing is fabricated.

FTS rowid contract: the schema's carrier is ``unit_fts_rows`` (MATCH
rowid → ``unit_fts_rows.row_id`` → ``unit_id``); the lexical and temporal
lanes additionally assume ``unit_fts.rowid == units.rowid``.  The writer
satisfies both by inserting ``unit_fts_rows.row_id = units.rowid``
explicitly — the two conventions coincide by construction on this write
path.  ``test_lexical.py::seed_unit`` documents the same convention.

Replay/closure: ``project_units_v7`` is idempotent for a fixed
``(source_id, revision, generation)`` — it deletes that slice first
(``_delete_slice``) and rewrites it.  ``delete_units_v7`` exposes the
same slice deletion for purge/closure paths.  Slice deletion removes
only rows it owns: ``entity_canon`` frequencies are *recomputed* (other
sources' mentions are untouched), alias rows survive when their canon
does, and ``method='review'``/``state='rejected'`` aliases are never
resurrected or deleted.  Audit journals (``screening_log``,
``run_manifests``) and producer-owned artifacts (``unit_vectors_block``,
``standing_queries`` packs) are deliberately left to their owners —
``standing_queries`` for the affected scope are instead marked
``dirty=1`` so the consolidation worker rebuilds packs against the new
corpus.

Fail-closed policy lives in ``project_source_v7``: ``VerbatimError``
with code INTEGRITY / STALE_EPOCH / QUARANTINED / STORE_CORRUPT
propagates (the job fails and retries are governed by the queue); any
other exception rolls back to the ``v7_units`` savepoint, is recorded as
``stats["v7_projection_error"] = repr(e)[:200]``, and the existing V5
commit proceeds.  The V7 slice is therefore either fully present or
fully absent — never half-written.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.time import rfc3339
from ..core.types import JobKind, VerbatimError, json_dumps, safe_json_loads
from ..core.types_v7 import (
    AliasMethod,
    AliasState,
    IntervalUs,
    MentionRole,
    OccurredPrecision,
    OccurredSource,
)
from ..storage.repos import has_table
from ..storage.schema_v7 import ensure_v7_additive, v7_fts_present, v7_tables_present
from ..storage.stats_v7 import corpus_stats, update_stats

UNITS_JOBS_VERSION = "units_jobs/v1"
_STATS_VERSION = "bm25f/v1"

# VerbatimError codes that must never be swallowed into stats — they are
# fail-closed fences (integrity evidence, epoch fencing, quarantine) and
# the job must fail rather than commit a V5 slice beside broken V7 state.
_FAIL_CLOSED = frozenset({"INTEGRITY", "STALE_EPOCH", "QUARANTINED", "STORE_CORRUPT"})

_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_WDAYS = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)
_WHEN_DAY_CAP = 14  # max distinct day tokens emitted into the `when` field
_MAX_EVENT_SPAN_BYTES = 512
_IN_CHUNK = 400  # SQLite variable-limit headroom (repo convention)
_SAVEPOINT = "v7_units"

#: V8-09.07 owned list — a lone subject token from this set that the
#: extractor left unresolved backfills to the unit's ``speaker_canon``.
_FIRST_PERSON_SUBJECTS = frozenset({"i", "me", "my", "we", "our"})

#: ``unit_doclen`` fields populated today (§19 field vocabulary is
#: ``text|entities|when|speaker|session|ctx`` — ``ctx`` is a later arm,
#: skipped while the field does not exist).
_DOCLEN_FIELDS = ("text", "entities", "when", "speaker", "session")


# ---------------------------------------------------------------------------
# V85-03.02 — session-continuing turn ordinals
# ---------------------------------------------------------------------------


def _assign_session_seqs(
    conn: sqlite3.Connection,
    units: List[dict],
    *,
    scope_id: str,
    generation: int,
) -> int:
    """Re-sequence turn ``seq`` values so they continue each session's
    stored ordinals instead of restarting per source.

    ``derive_units`` emits per-source session ordinals (0..n-1): a
    session spanning several sources — one ``Memory.add`` per turn, the
    Track R / AMB-provider shape — would mint ``seq = 0`` for every
    turn, and the context stage's neighbor window could never fire.
    Computed here, inside the projection transaction (the single writer
    makes the count race-free): ``base`` = ``turn`` rows already stored
    for ``(scope_id, session_id)`` at this generation — the slice being
    rewritten was already deleted by ``_delete_slice``, so replaying the
    same source lands the same ordinals — then each batch turn gets
    ``base +`` its in-source ordinal, preserving message order.

    ``unit_id`` is content-addressed over ``seq`` (and
    ``parent_unit_id``), so a re-sequenced turn is re-minted and its
    ``sentence_window`` children re-point at the new parent id.
    Returns the number of turns whose seq changed (honest stat).  The
    read side never depends on ``seq`` being correct — V85-03.01's
    effective position is the authority — so this stays a faithful
    write-time ordinal, not a load-bearing one.
    """
    from ..projections.units_v7 import _unit_id

    by_sess: Dict[str, List[dict]] = {}
    for u in units:
        if u.get("kind") == "turn" and u.get("session_id") is not None:
            by_sess.setdefault(str(u["session_id"]), []).append(u)
    if not by_sess:
        return 0

    def _ord(u: dict) -> tuple:
        def _n(v: Any) -> Any:
            return v if isinstance(v, int) and not isinstance(v, bool) else 1 << 60
        return (
            _n(u.get("seq")),
            _n(u.get("occurred_start_us")),
            _n(u.get("recorded_at_us")),
            _n(u.get("byte_start")),
            str(u.get("unit_id") or ""),
        )

    changed = 0
    remap: Dict[str, str] = {}
    for sess in sorted(by_sess):
        members = sorted(by_sess[sess], key=_ord)
        base = int(
            conn.execute(
                "SELECT COUNT(*) FROM units"
                " WHERE scope_id=? AND session_id=? AND kind='turn'"
                " AND generation=?",
                (scope_id, sess, generation),
            ).fetchone()[0]
        )
        for j, u in enumerate(members):
            new_seq = base + j
            if u.get("seq") == new_seq:
                continue
            old = u["unit_id"]
            u["seq"] = new_seq
            u["unit_id"] = _unit_id(u)
            remap[old] = u["unit_id"]
            changed += 1
    if remap:
        for u in units:
            parent = u.get("parent_unit_id")
            if parent in remap:
                u["parent_unit_id"] = remap[parent]
                u["unit_id"] = _unit_id(u)
    return changed


# ---------------------------------------------------------------------------
# public API — projection
# ---------------------------------------------------------------------------


def project_units_v7(
    conn: sqlite3.Connection,
    *,
    source_row: dict,
    revision_row: dict,
    add_args: Optional[dict],
    generation: int,
    scope_id: str,
) -> dict:
    """Derive and write every V7 artifact for one (source, revision).

    ``scope_id`` is the namespace authority the caller resolved for the
    projection plane (``source_state.namespace`` with a ``sources.scope_id``
    fallback — identical to the partition the sibling V5 writes use).  It is
    stamped on every emitted row so the whole V7 artifact set shares one
    scope partition.

    ``revision_row`` must carry ``payload`` (bytes or str — the persisted,
    HMAC-verified bytes the caller already decoded) plus the
    ``source_revisions`` metadata columns; ``add_args`` is the §31
    conversational-arguments dict (parsed ``metadata_json`` merged with
    any envelope-carried fields).  Returns a stats dict; raises nothing
    intentionally — the caller's savepoint wraps failure handling.
    """
    from ..projections.units_v7 import derive_units

    ensure_v7_additive(conn)

    stats: Dict[str, Any] = {
        "producer": UNITS_JOBS_VERSION,
        "generation": generation,
        "units": 0,
        "turns": 0,
        "sentence_windows": 0,
        "sessions": 0,
        "episodes": 0,
        "fts_rows": 0,
        "unpinned": 0,
        "mentions": 0,
        "canons": 0,
        "aliases_proposed": 0,
        "aliases_active": 0,
        "events": 0,
        "events_dropped_unpinned": 0,
        "events_speaker_backfill": 0,
        "time_mentions": "skipped",
        "doclen": "skipped",
        "state_facts": "skipped",
        "state_superseded": "skipped",
        "state_disputed": "skipped",
        "preferences": "skipped",
        "prefs": "skipped",
        "lex_stats": "skipped",
        "errors": [],
    }

    source_id = source_row.get("source_id")
    revision = int(revision_row.get("revision") or 1)
    if not source_id:
        stats["errors"].append("source_row without source_id")
        return stats

    payload = _payload_bytes(revision_row)
    # Empty payload carries no evidence — an honest zero, not a degenerate
    # [0,0) unit (V7-06.01: a unit pins a byte range; there are no bytes).
    if not payload:
        return stats

    # Namespace authority wins over whatever sources.scope_id says — the
    # same convention the sibling V5 writers use (they key on namespace).
    src = dict(source_row)
    src["scope_id"] = scope_id

    # Derive first: a derivation failure (VALIDATION — over-cap sources,
    # malformed pins) must not destroy an already-projected slice.
    units = derive_units(src, revision_row, add_args or {}, generation=generation)
    stats["units"] = len(units)
    if not units:
        return stats

    # Replay safety: wipe this exact (source, revision, generation) slice —
    # rows, FTS content/carrier, mentions, events, prefs/state, and the
    # lex_stats contributions — before rewriting.  Reprojection at a *new*
    # generation leaves the old generation's rows untouched (the fence
    # resolves visibility, V7-06.06/§30).
    _removed_ids, old_canons = _delete_slice(
        conn, source_id, revision, generation, stats
    )

    # V85-03.02 — continue session ordinals across sources, inside this
    # transaction (single writer ⇒ the COUNT base is race-free; replay of
    # the same slice deletes first, so the assignment is idempotent).
    stats["turn_seq_reassigned"] = _assign_session_seqs(
        conn, units, scope_id=scope_id, generation=generation
    )

    stats["turns"] = sum(1 for u in units if u["kind"] == "turn")
    stats["sentence_windows"] = sum(1 for u in units if u["kind"] == "sentence_window")
    stats["sessions"] = sum(1 for u in units if u["kind"] == "session")
    stats["episodes"] = sum(1 for u in units if u["kind"] == "episode")

    # Optional enrichment modules — lazy so a partially-deployed tree
    # still projects honestly (V7-10.01 fail-closed is about gates, not
    # about refusing to write when an enrichment lane is absent).
    sieve_fn = _load_sieve()
    state_fn, pref_fn, pref_producer = _load_prefs_state()
    compat_fn = _load_state_compat()

    addressee_canon = _addressee_canon(add_args or {})

    # V8 additive artifacts — §19 DDL lands via schema_v8; until it does
    # the projection runs and reports ``skipped`` rather than failing.
    has_utm = has_table(conn, "unit_time_mentions")
    has_doclen = has_table(conn, "unit_doclen")
    has_subj_src = _has_col(conn, "events_v7", "subject_source")
    if has_utm:
        stats["time_mentions"] = 0
    if has_doclen:
        stats["doclen"] = 0

    # ---- pass 1: units rows + FTS + per-unit mentions ---------------------
    unit_texts: Dict[str, str] = {}
    turn_ctx: Dict[str, dict] = {}
    stat_rows: List[dict] = []
    canon_first_seen: Dict[str, int] = {}  # canon -> earliest recorded_at_us
    canon_last_seen: Dict[str, int] = {}
    canon_display: Dict[str, str] = {}
    affected_canons: set = set()

    from ..text.norm_v2 import analyze
    from ..enrichment.entities_v2 import extract_mentions

    for u in units:
        cur = conn.execute(
            "INSERT INTO units(unit_id, source_id, revision, scope_id, kind,"
            " parent_unit_id, session_id, seq, speaker_canon, perspective,"
            " recorded_at_us, occurred_start_us, occurred_end_us,"
            " occurred_precision, occurred_source, byte_start, byte_end,"
            " generation)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                u["unit_id"], u["source_id"], u["revision"], scope_id,
                u["kind"], u.get("parent_unit_id"), u.get("session_id"),
                u.get("seq"), u.get("speaker_canon"), u.get("perspective"),
                u.get("recorded_at_us"), u.get("occurred_start_us"),
                u.get("occurred_end_us"), u.get("occurred_precision"),
                u.get("occurred_source"), u.get("byte_start"),
                u.get("byte_end"), generation,
            ),
        )
        unit_rowid = int(cur.lastrowid)

        text = _unit_text(u, payload)
        if text is None:
            stats["unpinned"] += 1
            continue
        unit_texts[u["unit_id"]] = text

        try:
            norm = analyze(text)
        except Exception as exc:  # analyzer hard-failure — index nothing for this unit
            stats["errors"].append(f"analyze:{u['unit_id']}:{repr(exc)[:120]}")
            stats["unpinned"] += 1
            continue
        # entities_v2 reads ``norm.text`` as the surface-bearing source
        # text — substitute the raw unit text (analyze() folds it).
        src_norm = replace(norm, text=text)

        mentions = extract_mentions(src_norm, u["unit_id"], u.get("speaker_canon"))
        ent_field = " ".join(sorted({m.canon for m in mentions if m.canon}))
        speaker_field = u.get("speaker_canon") or ""
        session_field = u.get("session_id") or ""
        when_field = _when_tokens(u)

        # FTS carrier + content.  row_id is bound to units.rowid so the
        # lexical/temporal lanes' `unit_fts.rowid == units.rowid`
        # convention holds while the schema's `unit_fts_rows` join also
        # resolves (see module docstring).
        conn.execute(
            "INSERT INTO unit_fts_rows(row_id, unit_id, scope_id, generation)"
            " VALUES (?,?,?,?)",
            (unit_rowid, u["unit_id"], scope_id, generation),
        )
        conn.execute(
            'INSERT INTO unit_fts_content(fts_row_id, text, speaker,'
            ' entities, session, "when") VALUES (?,?,?,?,?,?)',
            (unit_rowid, text, speaker_field, ent_field, session_field, when_field),
        )
        stats["fts_rows"] += 1

        # Corpus stats: the analyzer's folded text-channel terms for
        # ``text`` (the closest honest mirror of what unicode61 indexes);
        # the stored field string for the already-token-shaped fields.
        stat_fields = {
            "text": [t.term for t in norm.terms if t.channel == "text"],
            "speaker": speaker_field,
            "entities": ent_field,
            "session": session_field,
            "when": when_field,
        }
        stat_rows.append({"unit_id": u["unit_id"], "fields": stat_fields})

        # V8-07.04 — maintained BM25F field lengths for the indexed unit.
        if has_doclen:
            _write_doclen(
                conn, scope_id, generation, u["unit_id"], stat_fields, stats
            )

        if u["kind"] != "turn":
            continue

        # V8-09.04/§21.6 — resolve relative-time mentions on the unit's
        # own occurred anchor; precision-first, no rows without occurred.
        if has_utm:
            _write_time_mentions(
                conn, scope_id, generation, u, text, stats
            )

        # Mentions → postings + vocabulary bookkeeping for turn units.
        seen_pair = set()
        recorded = u.get("recorded_at_us")
        for m in mentions:
            canon = m.canon
            if not canon:
                continue
            key = (canon, int(m.byte_start or 0))
            if key in seen_pair:
                continue
            seen_pair.add(key)
            role = m.role.value if isinstance(m.role, MentionRole) else str(m.role or "mention")
            conn.execute(
                "INSERT INTO entity_mentions(scope_id, canon, unit_id,"
                " byte_start, byte_end, role, surface, generation)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (
                    scope_id, canon, u["unit_id"], int(m.byte_start or 0),
                    int(m.byte_end or 0), role, m.surface, generation,
                ),
            )
            stats["mentions"] += 1
            affected_canons.add(canon)
            if canon not in canon_display and m.surface:
                canon_display[canon] = m.surface
            if isinstance(recorded, int):
                canon_first_seen[canon] = min(canon_first_seen.get(canon, recorded), recorded)
                canon_last_seen[canon] = max(canon_last_seen.get(canon, recorded), recorded)

        turn_ctx[u["unit_id"]] = {
            "unit": u,
            "text": text,
            "norm": src_norm,
            "mentions": mentions,
        }

    # ---- entity_canon upserts (create-first, then recompute df) ----------
    # Union the canons this slice touched before deletion with the new
    # mention set so a re-run recomputes every affected row exactly once.
    _upsert_canons(
        conn, scope_id, generation, affected_canons,
        canon_display, canon_first_seen, canon_last_seen, stats,
    )
    _recompute_df(conn, scope_id, generation, affected_canons | old_canons)

    # ---- pass 2: per-turn events + optional prefs/state ------------------
    session_ctx = _session_context(units, unit_texts, turn_ctx)
    ev_drop = {"dropped_unpinned": 0, "dropped_malformed": 0, "dropped_timeout": 0}
    for uid, ctx in turn_ctx.items():
        u = ctx["unit"]
        occurred = _occurred_iv(u)
        ev_stats: Dict[str, int] = {}
        try:
            events = _extract_events(
                ctx["norm"], uid, u.get("speaker_canon"), occurred,
                ctx["text"], session_ctx, sieve_fn, addressee_canon, ev_stats,
            )
        except Exception as exc:
            stats["errors"].append(f"events:{uid}:{repr(exc)[:120]}")
            events = []
        for k in ev_drop:
            ev_drop[k] += int(ev_stats.get(k, 0))
        for ev in events:
            _write_event(
                conn, scope_id, generation, uid, ev, stats,
                speaker_canon=u.get("speaker_canon"),
                unit_text=ctx["text"],
                has_subject_source=has_subj_src,
            )
        if state_fn is not None or pref_fn is not None:
            _write_prefs_state(
                conn, scope_id, generation, u, ctx["norm"], occurred,
                state_fn, pref_fn, pref_producer, compat_fn, stats,
            )
    stats["events_dropped_unpinned"] = ev_drop["dropped_unpinned"]
    stats["events_dropped_malformed"] = ev_drop["dropped_malformed"]
    stats["events_dropped_timeout"] = ev_drop["dropped_timeout"]

    # ---- alias proposals over the scope vocabulary ------------------------
    _write_aliases(conn, scope_id, generation, turn_ctx, stats)

    # ---- corpus statistics (same transaction — V7-06.06) ------------------
    if stat_rows:
        try:
            update_stats(conn, scope_id, generation, stat_rows)
            stats["lex_stats"] = corpus_stats(conn, scope_id, generation)["text"]["n"]
        except Exception as exc:
            stats["errors"].append(f"lex_stats:{repr(exc)[:120]}")
            stats["lex_stats"] = "skipped"

    # ---- typed graph edges over the new slice (same tx) -------------------
    # ``build_edges`` replaces derived-type edges incident to these units,
    # so reprojection is idempotent; a failure is recorded, never fatal.
    try:
        from .graph_jobs import build_edges
        stats["graph_edges"] = build_edges(
            conn, scope_id, generation, units
        )
    except Exception as exc:
        stats["errors"].append(f"graph_edges:{repr(exc)[:120]}")
        stats["graph_edges"] = "skipped"

    # ---- observation consolidation over the new slice (same tx) ----------
    # ``consolidate_scope_v7`` is cursor-incremental: each call drains the
    # unconsolidated units this projection just wrote, so the observations
    # plane stays caught up with ingest instead of starving empty.
    try:
        from ..observations.consolidate_v7 import consolidate_scope_v7
        cst = consolidate_scope_v7(
            conn, scope_id=scope_id, generation=generation
        )
        stats["observations_v7"] = {
            "status": cst.get("status"),
            "processed": cst.get("processed"),
            "written": cst.get("observations_written"),
            "updated": cst.get("observations_updated"),
        }
    except Exception as exc:
        stats["errors"].append(f"observations_v7:{repr(exc)[:120]}")
        stats["observations_v7"] = "skipped"

    return stats


def project_source_v7(
    conn: sqlite3.Connection,
    *,
    source_id: str,
    revision: int,
    text: str,
    revision_meta: Optional[dict],
    namespace: str,
    generation: int,
) -> dict:
    """Job-facing wrapper: resolve the persisted source row + envelope args
    on ``conn`` and run :func:`project_units_v7` under a savepoint.

    Fail-closed ``VerbatimError`` codes propagate; everything else rolls
    back to ``v7_units`` and lands on the stats channel as
    ``v7_projection_error`` so the V5 commit is never held hostage by an
    additive-plane fault.
    """
    stats: Dict[str, Any] = {"producer": UNITS_JOBS_VERSION, "generation": generation,
                             "units": 0}

    row = conn.execute(
        "SELECT source_id, origin, external_id, source_kind, scope_id,"
        " speaker_id, created_us FROM sources WHERE source_id=?",
        (source_id,),
    ).fetchone()
    if row is None:
        stats["v7_projection_error"] = f"source row absent: {source_id}"
        return stats
    source_row = {
        "source_id": row[0], "origin": row[1], "external_id": row[2],
        "source_kind": row[3], "scope_id": row[4], "speaker_id": row[5],
        "created_us": row[6],
    }

    revision_row = dict(revision_meta or {})
    revision_row["source_id"] = source_id
    revision_row["revision"] = int(revision or 1)
    # The persisted bytes are the byte-pin ground truth (a purged revision
    # stores X'' — zero bytes, zero units, honestly).  Fall back to the
    # caller's verified text when the column is absent.
    prow = conn.execute(
        "SELECT payload FROM source_revisions WHERE source_id=? AND revision=?",
        (source_id, int(revision or 1)),
    ).fetchone()
    revision_row["payload"] = (
        prow[0] if prow and prow[0] is not None else (text or "")
    )

    add_args = _collect_add_args(conn, source_id, revision, revision_row)

    conn.execute(f"SAVEPOINT {_SAVEPOINT}")
    try:
        out = project_units_v7(
            conn,
            source_row=source_row,
            revision_row=revision_row,
            add_args=add_args,
            generation=generation,
            scope_id=namespace,
        )
    except VerbatimError as exc:
        _rollback_savepoint(conn)
        code = getattr(exc, "code", None)
        code_str = getattr(code, "value", code)
        if code_str in _FAIL_CLOSED:
            raise
        out = stats
        out["v7_projection_error"] = repr(exc)[:200]
    except Exception as exc:  # additive-plane fault — record, don't propagate
        _rollback_savepoint(conn)
        out = stats
        out["v7_projection_error"] = repr(exc)[:200]
    else:
        conn.execute(f"RELEASE {_SAVEPOINT}")
    return out


def delete_units_v7(
    conn: sqlite3.Connection,
    source_id: str,
    revision: int,
    generation: int,
) -> dict:
    """Remove every projection-owned artifact for (source, revision, generation).

    Shared with the replay path via :func:`_delete_slice`.  Returns per-table
    removal counts.  Audit/eval journals (``screening_log``, ``run_manifests``)
    and producer-owned blobs (``unit_vectors_block``) are not unit-attributed
    here — closure marks ``standing_queries`` dirty instead of fabricating
    pack semantics.  Safe no-op when V7 tables are absent or the slice is
    empty.
    """
    removed: Dict[str, Any] = {"producer": UNITS_JOBS_VERSION, "generation": generation}
    if not v7_tables_present(conn) or not has_table(conn, "units"):
        removed["units_removed"] = 0
        return removed
    _delete_slice(conn, source_id, int(revision), int(generation), removed)
    return removed


# ---------------------------------------------------------------------------
# V8-19.05 — owed-derivation frontier for coverage.derivations
# ---------------------------------------------------------------------------


def _pending_source_jobs(conn: sqlite3.Connection, scope_id: str) -> int:
    """Non-terminal ``source_project`` jobs owed to this scope — the same
    job-match pattern the dense lane's ``_coverage_lag`` uses (job
    ``scope_id`` or the ``namespace``/``scope_id`` carried in
    ``input_refs_json``); ``0`` when the jobs table is absent."""
    if not has_table(conn, "jobs"):
        return 0
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE kind = ?"
            " AND state IN ('queued','leased','retry_wait')"
            " AND (scope_id = ?"
            "      OR json_extract(input_refs_json, '$.namespace') = ?"
            "      OR json_extract(input_refs_json, '$.scope_id') = ?)",
            (
                JobKind.SOURCE_PROJECT.value,
                scope_id, scope_id, scope_id,
            ),
        ).fetchone()[0]
    )


def derivation_coverage_v8(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
) -> Dict[str, Optional[str]]:
    """The V8-19.05 owed-derivation frontier for ``coverage.derivations``.

    Returns the §20.03 key set — ``{units_v7, time_mentions, doclen,
    events_subject, claims_anchor, dense_compaction}`` — each ``"ok"``,
    ``"partial(n)"``, or ``None`` when the artifact cannot be evaluated
    on this store (absent §19 DDL or absent claims plane; the lane
    depending on it labels itself ``partial`` and keeps running).

    A ``source_project`` job owed to the scope is derivation debt for
    every units-plane artifact — the projection writes them in one tx —
    so the pending count joins each per-artifact debt figure. Per-key
    debt signals, all generation-fenced latest-row-per-unit reads:

    - ``units_v7`` — pending jobs plus committed non-empty
      ``source_revisions`` in the scope with no visible ``units`` slice
      at/below the pin (a stranded or failed projection, honest lower
      bound; an entirely absent ``units`` table makes every such
      revision owed, and payloads purged to X'' are excluded — there
      is nothing to project).
    - ``time_mentions`` — pending jobs plus units whose latest fenced
      mention rows carry a stale ``resolver_version`` (a version bump
      owes a rebuild, V8-09.04).
    - ``doclen`` — pending jobs plus FTS-carried units whose latest
      fenced carrier row has no ``unit_doclen`` rows at that generation
      (doclen is written for every indexed unit — absence is exact
      debt, not a guess).
    - ``events_subject`` — pending jobs plus latest-fenced ``events_v7``
      rows with ``subject_source IS NULL`` (pre-V8 rows owed
      re-derivation under the extractor bump, V8-09.07).
    - ``claims_anchor`` — latest-revision ``valid_intervals`` rows whose
      ``uncertainty_json`` carries no ``anchor`` (V8-09.05 debt; claim
      revisions fence on event-sequence, not projection generation, so
      the latest revision is the honest read).
    - ``dense_compaction`` — always ``None`` here: the compaction epoch
      lives in ``meta`` under the dense worker's ownership and is not
      computed by this module (V8-23.02 — never emit an uncomputed
      status).
    """
    gen = int(generation)
    pending = _pending_source_jobs(conn, scope_id)
    out: Dict[str, Optional[str]] = {}

    # ---- units_v7: owed jobs + stranded revisions -------------------------
    stranded = 0
    if has_table(conn, "source_revisions") and has_table(conn, "sources"):
        if has_table(conn, "units"):
            stranded = int(
                conn.execute(
                    "SELECT COUNT(*) FROM source_revisions sr"
                    " JOIN sources s ON s.source_id = sr.source_id"
                    " WHERE s.scope_id = ? AND length(sr.payload) > 0"
                    "  AND NOT EXISTS ("
                    "    SELECT 1 FROM units u"
                    "     WHERE u.source_id = sr.source_id"
                    "       AND u.revision = sr.revision"
                    "       AND u.generation <= ?)",
                    (scope_id, gen),
                ).fetchone()[0]
            )
        else:
            # the whole units plane is absent — every committed
            # non-empty revision in the scope is owed a projection.
            stranded = int(
                conn.execute(
                    "SELECT COUNT(*) FROM source_revisions sr"
                    " JOIN sources s ON s.source_id = sr.source_id"
                    " WHERE s.scope_id = ? AND length(sr.payload) > 0",
                    (scope_id,),
                ).fetchone()[0]
            )
    n = pending + stranded
    out["units_v7"] = "ok" if n == 0 else f"partial({n})"

    # ---- time_mentions: stale resolver versions at the fence --------------
    if has_table(conn, "unit_time_mentions"):
        from ..enrichment import reltime

        stale = int(
            conn.execute(
                "SELECT COUNT(DISTINCT m.unit_id) FROM unit_time_mentions m"
                " WHERE m.scope_id = ? AND m.resolver_version <> ?"
                "  AND m.generation = ("
                "    SELECT MAX(m2.generation) FROM unit_time_mentions m2"
                "     WHERE m2.unit_id = m.unit_id AND m2.generation <= ?)",
                (scope_id, reltime.RESOLVER_VERSION, gen),
            ).fetchone()[0]
        )
        n = pending + stale
        out["time_mentions"] = "ok" if n == 0 else f"partial({n})"
    else:
        out["time_mentions"] = None

    # ---- doclen: indexed units missing their length rows ------------------
    if has_table(conn, "unit_doclen") and has_table(conn, "unit_fts_rows"):
        missing = int(
            conn.execute(
                "SELECT COUNT(*) FROM unit_fts_rows r WHERE r.scope_id = ?"
                "  AND r.generation = ("
                "    SELECT MAX(r2.generation) FROM unit_fts_rows r2"
                "     WHERE r2.unit_id = r.unit_id AND r2.generation <= ?)"
                "  AND NOT EXISTS ("
                "    SELECT 1 FROM unit_doclen d WHERE d.unit_id = r.unit_id"
                "     AND d.generation = r.generation)",
                (scope_id, gen),
            ).fetchone()[0]
        )
        n = pending + missing
        out["doclen"] = "ok" if n == 0 else f"partial({n})"
    else:
        out["doclen"] = None

    # ---- events_subject: pre-V8 rows at the fence -------------------------
    if _has_col(conn, "events_v7", "subject_source"):
        stale = int(
            conn.execute(
                "SELECT COUNT(*) FROM events_v7 e WHERE e.scope_id = ?"
                "  AND e.subject_source IS NULL"
                "  AND e.generation = ("
                "    SELECT MAX(e2.generation) FROM events_v7 e2"
                "     WHERE e2.unit_id = e.unit_id AND e2.generation <= ?)",
                (scope_id, gen),
            ).fetchone()[0]
        )
        n = pending + stale
        out["events_subject"] = "ok" if n == 0 else f"partial({n})"
    else:
        out["events_subject"] = None

    # ---- claims_anchor: unanchored latest-revision intervals --------------
    if has_table(conn, "valid_intervals") and has_table(conn, "claims") \
            and has_table(conn, "claim_revisions"):
        unanchored = int(
            conn.execute(
                "SELECT COUNT(*) FROM valid_intervals vi"
                " JOIN claims c ON c.claim_id = vi.claim_id"
                " WHERE c.scope_id = ?"
                "  AND vi.revision = ("
                "    SELECT MAX(revision) FROM claim_revisions r"
                "     WHERE r.claim_id = vi.claim_id)"
                "  AND json_extract(vi.uncertainty_json, '$.anchor')"
                "      IS NULL",
                (scope_id,),
            ).fetchone()[0]
        )
        out["claims_anchor"] = (
            "ok" if unanchored == 0 else f"partial({unanchored})"
        )
    else:
        out["claims_anchor"] = None

    # ---- dense_compaction: owned by the dense worker (V8-08.01) -----------
    out["dense_compaction"] = None
    return out


# ---------------------------------------------------------------------------
# slice deletion (replay + closure share this)
# ---------------------------------------------------------------------------


_SLICE_UNITS = (
    "SELECT unit_id FROM units WHERE source_id=? AND revision=? AND generation=?"
)


def _delete_slice(
    conn: sqlite3.Connection,
    source_id: str,
    revision: int,
    generation: int,
    stats: Dict[str, Any],
) -> Tuple[List[str], set]:
    """Delete the (source, revision, generation) slice from every V7 table
    this projection owns; returns ``(deleted_unit_ids, canons_touched)``.

    Dependent-row deletes run while ``units`` still lists the slice so the
    IN-clause is a subquery — no parameter-list materialization, no
    variable-limit ceiling.  ``stats`` accumulates ``<table>_removed``
    counters for the job channel.
    """
    rows = conn.execute(
        "SELECT rowid, unit_id, scope_id FROM units"
        " WHERE source_id=? AND revision=? AND generation=?",
        (source_id, revision, generation),
    ).fetchall()
    if not rows:
        return [], set()
    unit_ids = [r[1] for r in rows]
    scope_id = rows[0][2]
    args = (source_id, revision, generation)

    # Canons whose df must be recomputed after mention removal.
    old_canons = {
        r[0]
        for r in conn.execute(
            f"SELECT DISTINCT canon FROM entity_mentions"
            f" WHERE generation=? AND unit_id IN ({_SLICE_UNITS})",
            (generation, *args),
        ).fetchall()
    } if has_table(conn, "entity_mentions") else set()

    # Reverse the lex_stats contribution of the FTS rows being removed —
    # re-analyzing the stored field strings yields exactly the terms that
    # were added at write time (deterministic mirror).
    if v7_fts_present(conn):
        old_content = conn.execute(
            f"SELECT r.unit_id, c.text, c.speaker, c.entities, c.session,"
            f" c.\"when\" FROM unit_fts_rows r"
            f" JOIN unit_fts_content c ON c.fts_row_id = r.row_id"
            f" WHERE r.unit_id IN ({_SLICE_UNITS})",
            args,
        ).fetchall()
        dec_rows = [
            {"unit_id": row[0],
             "fields": _fields_for_stats(row[1], row[2], row[3], row[4], row[5])}
            for row in old_content
        ]
        if dec_rows:
            try:
                _decrement_stats(conn, scope_id, generation, dec_rows)
            except Exception as exc:
                stats.setdefault("errors", []).append(
                    f"lex_stats_decrement:{repr(exc)[:120]}"
                )

        # Content-first deletion: the AFTER-DELETE trigger on
        # unit_fts_content removes the index rows; deleting the carrier
        # removes the external-content mapping.
        conn.execute(
            f"DELETE FROM unit_fts_content WHERE fts_row_id IN"
            f" (SELECT row_id FROM unit_fts_rows WHERE unit_id IN ({_SLICE_UNITS}))",
            args,
        )
        conn.execute(
            f"DELETE FROM unit_fts_rows WHERE unit_id IN ({_SLICE_UNITS})",
            args,
        )
        stats["fts_removed"] = stats.get("fts_removed", 0) + len(old_content)

    for table in ("entity_mentions", "events_v7", "state_facts",
                  "preferences", "standing_rules",
                  "unit_time_mentions", "unit_doclen"):
        if not has_table(conn, table):
            continue
        cur = conn.execute(
            f"DELETE FROM {table} WHERE generation=?"
            f" AND unit_id IN ({_SLICE_UNITS})",
            (generation, *args),
        )
        stats[f"{table}_removed"] = (
            stats.get(f"{table}_removed", 0) + max(cur.rowcount, 0)
        )

    if has_table(conn, "graph_edges"):
        # Edges are per-generation; remove any edge touching a removed unit
        # at that generation (unit_ids embed the generation, so the fence
        # column alone would suffice — both predicates are kept explicit).
        conn.execute(
            f"DELETE FROM graph_edges WHERE generation=?"
            f" AND (src_unit IN ({_SLICE_UNITS}) OR dst_unit IN ({_SLICE_UNITS}))",
            (generation, *args, *args),
        )

    # Unit-pinning artifacts from other producers retire with their
    # support (V7-13.13 semantics at the row level).
    for table, col in (("t2_facts", "unit_ids_json"),
                       ("observations_v7", "support_refs_json"),
                       ("profiles_v7", "support_refs_json")):
        if not has_table(conn, table):
            continue
        for chunk in _chunks(unit_ids):
            conn.execute(
                f"DELETE FROM {table} WHERE EXISTS ("
                f"  SELECT 1 FROM json_each({table}.{col}) je"
                f"  WHERE je.value IN ({_ph(chunk)})"
                f"     OR json_extract(je.value, '$.unit_id') IN ({_ph(chunk)})"
                f")",
                (*chunk, *chunk),
            )

    conn.execute(
        "DELETE FROM units WHERE source_id=? AND revision=? AND generation=?",
        args,
    )
    stats["units_removed"] = stats.get("units_removed", 0) + len(unit_ids)

    # Replay correctness (V8-13.05/K78): the consolidation pass is
    # cursor-incremental — a unit already in ``consolidation_seen_v7`` is
    # never reconsidered, and a rewritten unit may reuse a rowid at or
    # below the high-water mark.  The slice's unit-pinning artifacts
    # (observations_v7/profiles_v7) were retired above, so the seen
    # entries for the removed units must be cleared too: otherwise a
    # byte-identical reprojection leaves the observation plane
    # permanently stripped.  On the purge/delete path this is a no-op —
    # the units rows are gone, so the scan can never select them.
    if has_table(conn, "consolidation_seen_v7") and unit_ids:
        for chunk in _chunks(unit_ids):
            conn.execute(
                f"DELETE FROM consolidation_seen_v7 WHERE scope_id=?"
                f" AND generation=? AND unit_id IN ({_ph(chunk)})",
                (scope_id, generation, *chunk),
            )

    # Vocabulary maintenance: recompute df for canons that lost mentions;
    # drop canons with no remaining mentions at this generation; drop
    # rule-derived aliases whose canon no longer exists at this
    # generation.  Review/caller state is preserved.
    if has_table(conn, "entity_canon") and old_canons:
        _recompute_df(conn, scope_id, generation, old_canons)
        for chunk in _chunks(sorted(old_canons)):
            conn.execute(
                f"DELETE FROM entity_canon WHERE scope_id=? AND generation=?"
                f" AND canon IN ({_ph(chunk)}) AND df_units <= 0",
                (scope_id, generation, *chunk),
            )
    if has_table(conn, "entity_aliases_v7") and old_canons:
        for chunk in _chunks(sorted(old_canons)):
            conn.execute(
                f"DELETE FROM entity_aliases_v7 WHERE scope_id=?"
                f" AND generation=? AND method='rule'"
                f" AND canon IN ({_ph(chunk)})"
                f" AND canon NOT IN (SELECT canon FROM entity_canon"
                f"                  WHERE scope_id=? AND generation=?)",
                (scope_id, generation, *chunk, scope_id, generation),
            )

    # Consolidation artifacts embed unit refs in opaque blobs — flag the
    # scope's standing queries dirty rather than deleting row semantics we
    # don't own.
    if has_table(conn, "standing_queries"):
        conn.execute(
            "UPDATE standing_queries SET dirty=1 WHERE scope_id=?",
            (scope_id,),
        )

    return unit_ids, old_canons


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _ph(seq: Sequence) -> str:
    return ",".join("?" for _ in seq)


def _has_col(conn: sqlite3.Connection, table: str, col: str) -> bool:
    """Whether ``table`` carries ``col`` — additive-column gate for the
    §19 ALTERs that land via schema_v8 (absent → degrade, never crash)."""
    if not has_table(conn, table):
        return False
    return any(
        r[1] == col for r in conn.execute(f"PRAGMA table_info({table})")
    )


def _chunks(seq: Sequence, n: int = _IN_CHUNK) -> Iterable[Sequence]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _payload_bytes(revision_row: dict) -> bytes:
    payload = revision_row.get("payload")
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode("utf-8")
    return b""


def _unit_text(unit: dict, payload: bytes) -> Optional[str]:
    """Resolve the unit's byte pins against the persisted payload.

    Returns the verified slice, an explicit producer-carried ``text``
    fallback, or ``None`` when the pins cannot be grounded (the unit row
    is still written — metadata is real — but nothing unverifiable is
    indexed).
    """
    bs, be = unit.get("byte_start"), unit.get("byte_end")
    if (
        isinstance(bs, int) and isinstance(be, int)
        and not isinstance(bs, bool) and not isinstance(be, bool)
        and 0 <= bs <= be <= len(payload)
    ):
        try:
            return payload[bs:be].decode("utf-8")
        except UnicodeDecodeError:
            return None
    t = unit.get("text")
    return t if isinstance(t, str) and t != "" else None


def _when_tokens(unit: dict) -> str:
    """Normalized date tokens for the ``when`` field (V7-06.02): ISO day
    for each covered day (bounded), plus month, year, month-name and
    weekday tokens; precision-aware trimming for month/year/decade
    intervals; recorded_at_us is the honest fallback when the unit has no
    occurred interval."""
    start = unit.get("occurred_start_us")
    end = unit.get("occurred_end_us")
    precision = (unit.get("occurred_precision") or "").lower()
    if not isinstance(start, int) or not isinstance(end, int) or end < start:
        rec = unit.get("recorded_at_us")
        if not isinstance(rec, int):
            return ""
        start = end = rec
        precision = "day"

    toks: List[str] = []

    def _add(tok: str) -> None:
        if tok and tok not in toks:
            toks.append(tok)

    iso_s = rfc3339(start)[:10]
    iso_e = rfc3339(end)[:10]
    y_s, m_s, d_s = int(iso_s[:4]), int(iso_s[5:7]), int(iso_s[8:10])
    y_e, m_e, d_e = int(iso_e[:4]), int(iso_e[5:7]), int(iso_e[8:10])

    if precision in ("month",):
        for y, m in {(y_s, m_s), (y_e, m_e)}:
            _add(f"{y:04d}-{m:02d}")
            _add(f"{y:04d}")
            _add(_MONTHS[m - 1])
        return " ".join(toks)
    if precision in ("year",):
        for y in {y_s, y_e}:
            _add(f"{y:04d}")
        return " ".join(toks)
    if precision in ("decade",):
        _add(f"{y_s:04d}")
        _add(f"{(y_s // 10) * 10}s")
        return " ".join(toks)

    # day-ish precisions (instant/day/week/season/unknown): emit each
    # covered day up to the cap; beyond it keep first+last day so the
    # field stays honest and bounded.
    days = _covered_days(start, end)
    emit = days if len(days) <= _WHEN_DAY_CAP else [days[0], days[-1]]
    for iso in emit:
        y, m, d = int(iso[:4]), int(iso[5:7]), int(iso[8:10])
        _add(iso)
        _add(f"{y:04d}-{m:02d}")
        _add(f"{y:04d}")
        _add(_MONTHS[m - 1])
        _add(_WDAYS[_weekday(y, m, d)])
    return " ".join(toks)


def _covered_days(start_us: int, end_us: int) -> List[str]:
    """UTC ISO dates touched by [start, end] inclusive of both bounds."""
    import datetime as _dt

    out: List[str] = []
    cur = _dt.datetime.fromtimestamp(start_us / 1_000_000, tz=_dt.timezone.utc).date()
    last = _dt.datetime.fromtimestamp(end_us / 1_000_000, tz=_dt.timezone.utc).date()
    if last < cur:
        cur, last = last, cur
    while cur <= last and len(out) < _WHEN_DAY_CAP + 2:
        out.append(cur.isoformat())
        cur += _dt.timedelta(days=1)
    if cur <= last:
        out.append(last.isoformat())
    return out


def _weekday(y: int, m: int, d: int) -> int:
    import datetime as _dt

    return _dt.date(y, m, d).weekday()


def _occurred_iv(unit: dict) -> IntervalUs:
    def _prec(v) -> OccurredPrecision:
        try:
            return OccurredPrecision(str(v))
        except Exception:
            return OccurredPrecision.UNKNOWN

    def _src(v) -> OccurredSource:
        try:
            return OccurredSource(str(v))
        except Exception:
            return OccurredSource.UNKNOWN

    return IntervalUs(
        start_us=unit.get("occurred_start_us"),
        end_us=unit.get("occurred_end_us"),
        precision=_prec(unit.get("occurred_precision") or "unknown"),
        source=_src(unit.get("occurred_source") or "unknown"),
    )


def _load_sieve() -> Optional[Callable]:
    try:
        from ..enrichment import coref_sieve
        return coref_sieve.resolve_antecedent
    except Exception:
        return None


def _load_prefs_state() -> Tuple[Optional[Callable], Optional[Callable], str]:
    """``enrichment/prefs_state`` is a later wave deliverable — import
    lazily and report ``skipped`` honestly when absent."""
    try:
        from ..enrichment import prefs_state
    except Exception:
        return None, None, "prefs_state/v1"
    state_fn = getattr(prefs_state, "extract_state_facts", None)
    pref_fn = getattr(prefs_state, "extract_preferences", None)
    producer = getattr(prefs_state, "PRODUCER", "prefs_state/v1")
    return state_fn, pref_fn, producer


def _load_state_compat() -> Optional[Callable]:
    """``prefs_state.state_compatible`` — the family-mode coexistence
    test the slot-key supersession pass applies (V75-03.01/V7-16.03).
    Absent → the pass stays ``skipped``, never a guessed comparator."""
    try:
        from ..enrichment import prefs_state
    except Exception:
        return None
    return getattr(prefs_state, "state_compatible", None)


def _addressee_canon(add_args: dict) -> Optional[str]:
    """For 1:1 conversations the conversational partner's canon, when the
    add args distinguish the roles (V7-12.01 / SPEC_V7 §31 fields)."""
    try:
        from ..enrichment.entities_v2 import canon as ent_canon
    except Exception:
        return None
    speaker_c = ent_canon(add_args.get("speaker") or add_args.get("speaker_id") or "") or None
    user_c = ent_canon(add_args.get("user_id") or "") or None
    asst_c = ent_canon(add_args.get("assistant_id") or "") or None
    if speaker_c and user_c and speaker_c == user_c and asst_c:
        return asst_c
    if speaker_c and asst_c and speaker_c == asst_c and user_c:
        return user_c
    return None


def _session_context(
    units: Sequence[dict],
    unit_texts: Dict[str, str],
    turn_ctx: Dict[str, dict],
) -> Dict[str, dict]:
    """session_key -> ordered turn contexts for the coreference sieve.

    ``session_id`` groups real sessions; all sessionless turns share one
    ``_sessionless`` context so pronouns across adjacent unlabeled turns
    still resolve (deterministic, seq-ordered)."""
    groups: Dict[str, List[dict]] = {}
    for u in units:
        if u["kind"] != "turn":
            continue
        key = u.get("session_id") or "_sessionless"
        groups.setdefault(key, []).append(u)
    ctx: Dict[str, dict] = {}
    for key, members in groups.items():
        members.sort(key=lambda u: (u.get("seq") if u.get("seq") is not None else 1 << 60,
                                    u["unit_id"]))
        turns = []
        for u in members:
            tc = turn_ctx.get(u["unit_id"])
            mentions = tc["mentions"] if tc else []
            turns.append({
                "speaker": u.get("speaker_canon"),
                "speaker_canon": u.get("speaker_canon"),
                "text": unit_texts.get(u["unit_id"], ""),
                "canon_mentions": [
                    m.canon for m in mentions
                    if m.canon and m.role != MentionRole.SPEAKER
                ],
            })
        for idx, u in enumerate(members):
            ctx[u["unit_id"]] = {"turns": turns, "unit_index": idx}
    return ctx


def _extract_events(norm, unit_id, speaker_canon, occurred, text,
                    session_ctx, sieve_fn, addressee_canon, ev_stats):
    from ..enrichment.events import extract_events

    sieve = None
    if sieve_fn is not None:
        sctx = session_ctx.get(unit_id)
        if sctx is not None:
            idx, turns = sctx["unit_index"], sctx["turns"]
            sieve = lambda surface: sieve_fn(surface, idx, turns)  # noqa: E731
    return extract_events(
        norm, unit_id, speaker_canon, occurred,
        sieve=sieve, raw_text=text, addressee_canon=addressee_canon,
        stats=ev_stats,
    )


def _subject_surface_token(ev, unit_text) -> Optional[str]:
    """The extractor's subject token sliced from the unit's pinned bytes —
    casefolded; ``None`` when the span is absent or cannot be grounded."""
    pins = dict(getattr(ev, "pins", None) or {})
    span = pins.get("subject")
    if not span or unit_text is None:
        return None
    try:
        b = unit_text.encode("utf-8")
        s, e = int(span[0]), int(span[1])
        if not (0 <= s < e <= len(b)):
            return None
        return b[s:e].decode("utf-8").strip().casefold()
    except Exception:
        return None


def _write_event(conn, scope_id, generation, unit_id, ev, stats,
                 speaker_canon=None, unit_text=None,
                 has_subject_source=False) -> None:
    subject = getattr(ev, "subject_canon", None)
    # V8-09.07 — a lone first-person-pronoun subject token (owned list)
    # resolves to the source unit's ``speaker_canon``: when the extractor
    # would emit NULL the writer backfills it here; either way the row is
    # marked ``speaker_backfill`` (K56 — "I started painting" by melanie
    # is speaker-derived, not an extracted name). Any other subject keeps
    # the extractor's result and is marked ``extracted``; pre-V8 rows
    # carry NULL via the additive column's default.
    tok = _subject_surface_token(ev, unit_text)
    first_person = tok in _FIRST_PERSON_SUBJECTS
    if subject is None and first_person and speaker_canon:
        subject = speaker_canon
    subject_source = (
        "speaker_backfill"
        if first_person and subject is not None and subject == speaker_canon
        else "extracted"
    )
    if subject_source == "speaker_backfill":
        stats["events_speaker_backfill"] = (
            stats.get("events_speaker_backfill", 0) + 1
        )
    occurred = getattr(ev, "occurred", None)
    occ_s = getattr(occurred, "start_us", None) if occurred else None
    occ_e = getattr(occurred, "end_us", None) if occurred else None
    prec = getattr(occurred, "precision", None)
    prec = getattr(prec, "value", prec)
    pins = dict(getattr(ev, "pins", None) or {})
    span = pins.get("span") or [0, 0]
    try:
        if int(span[1]) - int(span[0]) > _MAX_EVENT_SPAN_BYTES:
            stats["events_dropped_malformed"] = stats.get("events_dropped_malformed", 0) + 1
            return
    except Exception:
        pass
    pins_json = json_dumps(pins)
    eid = "ev7:" + hashlib.sha256(json_dumps({
        "v": "event/v1", "unit_id": unit_id,
        "subject_canon": subject,
        "predicate_lemma": getattr(ev, "predicate_lemma", ""),
        "object_text": getattr(ev, "object_text", ""),
        "polarity": getattr(ev, "polarity", ""),
        "occ_s": occ_s, "occ_e": occ_e,
        "rule_id": getattr(ev, "rule_id", ""),
        "span": span,
    }).encode("utf-8")).hexdigest()[:24]
    if has_subject_source:
        cur = conn.execute(
            "INSERT INTO events_v7(event_id, unit_id, scope_id, subject_canon,"
            " predicate_lemma, object_text, polarity, occurred_start_us,"
            " occurred_end_us, precision, pins_json, rule_id, generation,"
            " subject_source)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(event_id) DO NOTHING",
            (
                eid, unit_id, scope_id, subject,
                getattr(ev, "predicate_lemma", ""),
                getattr(ev, "object_text", ""),
                getattr(ev, "polarity", ""), occ_s, occ_e, prec, pins_json,
                getattr(ev, "rule_id", ""), generation, subject_source,
            ),
        )
    else:
        cur = conn.execute(
            "INSERT INTO events_v7(event_id, unit_id, scope_id, subject_canon,"
            " predicate_lemma, object_text, polarity, occurred_start_us,"
            " occurred_end_us, precision, pins_json, rule_id, generation)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(event_id) DO NOTHING",
            (
                eid, unit_id, scope_id, subject,
                getattr(ev, "predicate_lemma", ""),
                getattr(ev, "object_text", ""),
                getattr(ev, "polarity", ""), occ_s, occ_e, prec, pins_json,
                getattr(ev, "rule_id", ""), generation,
            ),
        )
    stats["events"] += max(cur.rowcount, 0)


def _state_fact_anchor(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    unit_id: str,
    valid_from_us: Optional[int],
) -> Optional[int]:
    """Ordering instant for a stored state fact: its occurred start
    (``valid_from_us``), else the owning unit's ``recorded_at_us`` — the
    same fallback V75-03.01 declares for the successor. ``None`` means
    unordered (V7-16.05)."""
    if valid_from_us is not None:
        return valid_from_us
    row = conn.execute(
        "SELECT recorded_at_us FROM units WHERE scope_id=? AND unit_id=?"
        " AND generation<=? ORDER BY generation DESC LIMIT 1",
        (scope_id, unit_id, generation),
    ).fetchone()
    return row[0] if row else None


def _supersede_state_fact(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    state_key: str,
    new_value: str,
    new_anchor_us: Optional[int],
    compat_fn: Callable,
    stats: Dict[str, Any],
) -> Tuple[str, Optional[int]]:
    """V75-03.01 / V7-16.03: supersession pass for one incoming fact.

    Compares the new value against every live row of its
    ``(scope_id, state_key)`` slot — the latest fenced row per natural
    key (``unit_id, value_norm``; generation is the version fence, so
    the pass never demotes a row its own fence could not read):

    - compatible values (accumulate families, equal replacements under
      ``compat_fn``) coexist — no lifecycle change;
    - a strictly newer incompatible value closes a live ``current``
      predecessor at the successor anchor (``historical`` +
      ``valid_to_us`` = successor occurred-start, or its
      ``recorded_at_us`` when occurred is unknown);
    - a strictly older incompatible value lands ``historical`` itself,
      bounded by the nearest incompatible successor;
    - no clear temporal order (either anchor unknown, or equal) leaves
      both sides ``disputed`` (V7-16.05) — an already-closed
      ``historical`` row never disputes, while an open ``disputed`` one
      pulls the unordered newcomer into the dispute.

    Runs inside the caller's transaction.  Returns
    ``(status, valid_to_us)`` for the incoming row and bumps
    ``stats["state_superseded"]`` / ``stats["state_disputed"]`` per row
    it labels (predecessors demoted and the incoming row alike).
    """
    rows = conn.execute(
        "SELECT unit_id, value_norm, value_text, valid_from_us, status,"
        " MAX(generation) FROM state_facts"
        " WHERE scope_id=? AND state_key=? AND generation<=?"
        " GROUP BY unit_id, value_norm ORDER BY unit_id, value_norm",
        (scope_id, state_key, generation),
    ).fetchall()
    valid_to: Optional[int] = None
    disputed = False
    for p_uid, p_vnorm, p_vtext, p_vfrom, p_status, p_gen in rows:
        if compat_fn(state_key, p_vnorm or p_vtext or "", new_value):
            continue
        p_anchor = _state_fact_anchor(
            conn, scope_id, generation, p_uid, p_vfrom)
        if new_anchor_us is None or p_anchor is None \
                or new_anchor_us == p_anchor:
            if p_status == "current":
                conn.execute(
                    "UPDATE state_facts SET status='disputed'"
                    " WHERE scope_id=? AND state_key=? AND unit_id=?"
                    " AND value_norm IS ? AND generation=?",
                    (scope_id, state_key, p_uid, p_vnorm, p_gen),
                )
                stats["state_disputed"] += 1
                disputed = True
            elif p_status == "disputed":
                disputed = True
            continue
        if new_anchor_us < p_anchor:
            # the incoming claim is the prior value — it lands bounded
            # by the nearest incompatible successor.
            valid_to = (
                p_anchor if valid_to is None else min(valid_to, p_anchor)
            )
            continue
        if p_status == "current":
            conn.execute(
                "UPDATE state_facts SET status='historical',"
                " valid_to_us=? WHERE scope_id=? AND state_key=?"
                " AND unit_id=? AND value_norm IS ? AND generation=?",
                (new_anchor_us, scope_id, state_key, p_uid,
                 p_vnorm, p_gen),
            )
            stats["state_superseded"] += 1
    if disputed:
        return "disputed", None
    if valid_to is not None:
        return "historical", valid_to
    return "current", None


def _write_prefs_state(conn, scope_id, generation, unit, norm, occurred,
                       state_fn, pref_fn, producer, compat_fn, stats) -> None:
    uid = unit["unit_id"]
    speaker = unit.get("speaker_canon") or ""
    raw_text = getattr(norm, "text", None)
    if state_fn is not None:
        try:
            facts = state_fn(
                norm, uid, speaker, occurred,
                scope_id=scope_id, raw_text=raw_text,
            ) or []
        except Exception as exc:
            stats["errors"].append(f"state:{uid}:{repr(exc)[:120]}")
            facts = []
        if stats["state_facts"] == "skipped":
            stats["state_facts"] = 0
        if compat_fn is not None:
            if stats["state_superseded"] == "skipped":
                stats["state_superseded"] = 0
            if stats["state_disputed"] == "skipped":
                stats["state_disputed"] = 0
        for f in facts:
            status = getattr(getattr(f, "status", None), "value", None) or "current"
            if status not in ("current", "historical", "disputed"):
                status = "current"
            valid_to = getattr(f, "valid_to_us", None)
            if status == "current" and compat_fn is not None:
                anchor = getattr(f, "valid_from_us", None)
                if anchor is None:
                    anchor = unit.get("recorded_at_us")
                status, new_valid_to = _supersede_state_fact(
                    conn, scope_id, generation,
                    getattr(f, "state_key", "") or "",
                    getattr(f, "value_norm", "") or getattr(f, "value_text", "") or "",
                    anchor, compat_fn, stats,
                )
                if new_valid_to is not None:
                    valid_to = new_valid_to
                if status == "disputed":
                    stats["state_disputed"] += 1
                elif status == "historical":
                    stats["state_superseded"] += 1
            conn.execute(
                "INSERT INTO state_facts(scope_id, state_key, unit_id,"
                " generation, value_text, value_norm, valid_from_us,"
                " valid_to_us, status, producer, pins_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    scope_id, getattr(f, "state_key", ""), uid, generation,
                    getattr(f, "value_text", "") or "",
                    getattr(f, "value_norm", "") or "",
                    getattr(f, "valid_from_us", None),
                    valid_to,
                    status,
                    getattr(f, "producer", None) or producer,
                    json_dumps(dict(getattr(f, "pins", None) or {})),
                ),
            )
            stats["state_facts"] += 1
    if pref_fn is not None:
        try:
            prefs = pref_fn(
                norm, uid, speaker,
                scope_id=scope_id, occurred=occurred, raw_text=raw_text,
            ) or []
        except Exception as exc:
            stats["errors"].append(f"prefs:{uid}:{repr(exc)[:120]}")
            prefs = []
        if stats["preferences"] == "skipped":
            stats["preferences"] = 0
            stats["prefs"] = 0
        for p in prefs:
            occ = getattr(p, "occurred", None)
            conn.execute(
                "INSERT INTO preferences(scope_id, subject_canon, unit_id,"
                " generation, object_text, polarity, strength,"
                " occurred_start_us, pins_json)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    scope_id, getattr(p, "subject_canon", None) or speaker,
                    uid, generation, getattr(p, "object_text", "") or "",
                    getattr(p, "polarity", "") or "",
                    getattr(p, "strength", "") or "",
                    getattr(occ, "start_us", None) if occ else None,
                    json_dumps(dict(getattr(p, "pins", None) or {})),
                ),
            )
            stats["preferences"] += 1
            stats["prefs"] = stats["preferences"]


def _upsert_canons(conn, scope_id, generation, canons, display, first_seen,
                   last_seen, stats) -> None:
    for canon in sorted(canons):
        row = conn.execute(
            "SELECT display, first_seen_us, last_seen_us FROM entity_canon"
            " WHERE scope_id=? AND canon=? AND generation=?",
            (scope_id, canon, generation),
        ).fetchone()
        new_display = display.get(canon)
        fs, ls = first_seen.get(canon), last_seen.get(canon)
        if row is None:
            conn.execute(
                "INSERT INTO entity_canon(scope_id, canon, generation,"
                " display, df_units, kind, first_seen_us, last_seen_us)"
                " VALUES (?,?,?,?,0,NULL,?,?)",
                (scope_id, canon, generation, new_display, fs, ls),
            )
            stats["canons"] += 1
        else:
            d = row[0] or new_display
            f = min(x for x in (row[1], fs) if x is not None) if (row[1] is not None or fs is not None) else None
            l = max(x for x in (row[2], ls) if x is not None) if (row[2] is not None or ls is not None) else None
            conn.execute(
                "UPDATE entity_canon SET display=?, first_seen_us=?,"
                " last_seen_us=? WHERE scope_id=? AND canon=? AND generation=?",
                (d, f, l, scope_id, canon, generation),
            )


def _recompute_df(conn, scope_id, generation, canons) -> None:
    """df_units = DISTINCT units mentioning the canon at this generation —
    recomputed rather than incremented so replay and mixed-source scopes
    stay exact (V7-08.03)."""
    for chunk in _chunks(sorted(canons)):
        conn.execute(
            f"UPDATE entity_canon SET df_units = ("
            f"  SELECT COUNT(DISTINCT unit_id) FROM entity_mentions"
            f"  WHERE scope_id=? AND canon=entity_canon.canon"
            f"    AND generation=entity_canon.generation)"
            f" WHERE scope_id=? AND generation=? AND canon IN ({_ph(chunk)})",
            (scope_id, scope_id, generation, *chunk),
        )


def _fields_for_stats(text, speaker, entities, session, when) -> dict:
    """Rebuild the stats fields for a stored content row — analyzed terms
    for ``text`` (mirroring write-time exactly), stored strings elsewhere."""
    try:
        from ..text.norm_v2 import analyze
        terms = [t.term for t in analyze(text or "").terms if t.channel == "text"]
    except Exception:
        terms = (text or "").split()
    return {"text": terms, "speaker": speaker, "entities": entities,
            "session": session, "when": when}


_STATS_FIELDS = ("text", "speaker", "entities", "session", "when")


def _stat_terms(value: object) -> List[str]:
    """Same term fold as ``stats_v7._field_terms``: whitespace-joined
    string or an iterable of terms; anything else is length-0."""
    if value is None:
        return []
    if isinstance(value, str):
        return value.split()
    try:
        return [str(t) for t in value]  # type: ignore[union-attr]
    except TypeError:
        return []


def _decrement_stats(conn, scope_id: str, generation: int,
                     unit_rows: Iterable[Mapping]) -> None:
    """Subtract a batch's contribution from ``lex_stats``/``lex_df``.

    Mirrors ``stats_v7._apply(sign=-1)`` exactly — per-field ``n_units``/
    ``total_len`` deltas plus distinct-term df deltas, floored at 0 — but
    subtracts directly.  ``stats_v7.decrement_stats`` is not used: its
    upsert pre-clamps the *delta* with ``MAX(0, ?)``, so ``excluded``
    values are never negative and existing rows never decrement (a real
    bug in that module — reported).  The identical ``_aggregate`` fold is
    replicated here so add/remove stay symmetric.
    """
    field_len = {f: 0 for f in _STATS_FIELDS}
    df_delta: Dict[Tuple[str, str], int] = {}
    n = 0
    for row in unit_rows:
        fields = row.get("fields", {})
        if not isinstance(fields, Mapping):
            fields = {}
        n += 1
        for field in _STATS_FIELDS:
            terms = _stat_terms(fields.get(field))
            field_len[field] += len(terms)
            for term in set(terms):
                key = (field, term)
                df_delta[key] = df_delta.get(key, 0) + 1
    if n == 0:
        return
    for field in _STATS_FIELDS:
        conn.execute(
            "UPDATE lex_stats SET n_units=MAX(0, n_units-?),"
            " total_len=MAX(0, total_len-?)"
            " WHERE scope_id=? AND generation=? AND field=?"
            "   AND stats_version=?",
            (n, field_len[field], scope_id, int(generation), field,
             _STATS_VERSION),
        )
    for (field, term), ddf in df_delta.items():
        conn.execute(
            "UPDATE lex_df SET df=MAX(0, df-?)"
            " WHERE scope_id=? AND generation=? AND field=? AND term=?"
            "   AND stats_version=?",
            (ddf, scope_id, int(generation), field, term, _STATS_VERSION),
        )


def _write_doclen(conn, scope_id, generation, unit_id, fields,
                  stats) -> None:
    """V8-07.04 — one ``unit_doclen`` row per populated BM25F field.

    ``len`` is the same fold ``lex_stats`` maintains (``_stat_terms`` ≡
    ``stats_v7._field_terms``): the analyzed term list for ``text``, the
    stored field string's whitespace split for the token-shaped fields —
    so doclen and avglen share one length definition by construction."""
    for field in _DOCLEN_FIELDS:
        n = len(_stat_terms(fields.get(field)))
        conn.execute(
            "INSERT INTO unit_doclen(unit_id, generation, scope_id,"
            " field, len) VALUES (?,?,?,?,?)",
            (unit_id, generation, scope_id, field, n),
        )
        stats["doclen"] += 1


def _write_time_mentions(conn, scope_id, generation, unit, text,
                         stats) -> None:
    """V8-09.04/§21.6 — ``unit_time_mentions`` rows for a turn unit.

    Anchored on the unit's own occurred instant; precision-first —
    ambiguous/unresolvable/open expressions emit no row, and a unit
    without occurred bounds gets no rows (never a clock guess)."""
    from ..enrichment import reltime

    anchor = reltime.unit_anchor_us(unit)
    if anchor is None:
        return
    rows = reltime.resolve_mentions(text, anchor)
    for m in rows:
        conn.execute(
            "INSERT INTO unit_time_mentions(unit_id, generation, scope_id,"
            " ord, start_us, end_us, precision, span_start, span_end,"
            " anchor_us, resolver_version)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                unit["unit_id"], generation, scope_id, m["ord"],
                m["start_us"], m["end_us"], m["precision"],
                m["span_start"], m["span_end"], anchor,
                reltime.RESOLVER_VERSION,
            ),
        )
        stats["time_mentions"] += 1


def _write_aliases(conn, scope_id, generation, turn_ctx, stats) -> None:
    """Conservative alias proposals: A1/A2/A3 may be active; A4–A6 stay
    candidate; review/rejected rows are never demoted."""
    try:
        from ..enrichment.entities_v2 import propose_aliases
    except Exception:
        return
    mentions = [m for ctx in turn_ctx.values() for m in ctx["mentions"]]
    if not mentions:
        return
    known = [
        r[0]
        for r in conn.execute(
            "SELECT canon FROM entity_canon WHERE scope_id=? AND generation<=?",
            (scope_id, generation),
        ).fetchall()
    ]
    texts = [ctx["text"] for ctx in turn_ctx.values()]
    try:
        rows = propose_aliases(
            mentions, known, now_generation=generation,
            texts=texts, scope_id=scope_id,
        )
    except Exception as exc:
        stats["errors"].append(f"aliases:{repr(exc)[:120]}")
        return
    for row in rows:
        state = row.state.value if isinstance(row.state, AliasState) else str(row.state)
        method = row.method.value if isinstance(row.method, AliasMethod) else str(row.method)
        if method not in ("rule", "caller", "review"):
            method = "rule"
        # conservative: only A1/A2/A3 may auto-activate
        if state == "active" and row.rule_id not in ("A1", "A2", "A3"):
            state = "candidate"
        if state not in ("active", "candidate", "rejected"):
            state = "candidate"
        conn.execute(
            "INSERT INTO entity_aliases_v7(scope_id, canon, alias_canon,"
            " generation, rule_id, evidence_count, method, state)"
            " VALUES (?,?,?,?,?,?,?,?)"
            " ON CONFLICT(scope_id, canon, alias_canon, generation)"
            " DO UPDATE SET"
            "  evidence_count=MAX(entity_aliases_v7.evidence_count,"
            "                    excluded.evidence_count),"
            "  rule_id=excluded.rule_id,"
            "  method=CASE WHEN entity_aliases_v7.method='review' THEN 'review'"
            "              ELSE excluded.method END,"
            "  state=CASE WHEN entity_aliases_v7.state='rejected' THEN 'rejected'"
            "             WHEN entity_aliases_v7.method='review'"
            "                  THEN entity_aliases_v7.state"
            "             ELSE excluded.state END",
            (scope_id, row.canon, row.alias_canon, generation, row.rule_id,
             int(row.evidence_count or 1), method, state),
        )
        stats["aliases_proposed"] += 1
        if state == "active":
            stats["aliases_active"] += 1


def _collect_add_args(conn, source_id, revision, revision_row) -> dict:
    """§31 add_args: revision ``metadata_json`` merged with any
    envelope-carried fields.  ``envelope_kind`` fills only when the
    metadata doesn't already declare role/kind (``derive_units`` prefers
    explicit role/kind anyway); ``session_id`` is filled only when absent
    so a caller-declared session is never overridden by the capture
    session."""
    add_args: Dict[str, Any] = {}
    meta = safe_json_loads(revision_row.get("metadata_json") or "{}")
    if isinstance(meta, dict):
        add_args.update(meta)
    if has_table(conn, "source_envelopes"):
        for env in conn.execute(
            "SELECT envelope_kind, session_id, metadata_json FROM source_envelopes"
            " WHERE source_id=? AND revision=?",
            (source_id, revision),
        ).fetchall():
            em = safe_json_loads(env[2] or "{}")
            if isinstance(em, dict):
                for k, v in em.items():
                    add_args.setdefault(k, v)
            if env[0]:
                add_args.setdefault("envelope_kind", env[0])
            if env[1] and not add_args.get("session_id"):
                add_args["session_id"] = env[1]
    return add_args


def _rollback_savepoint(conn) -> None:
    try:
        conn.execute(f"ROLLBACK TO {_SAVEPOINT}")
        conn.execute(f"RELEASE {_SAVEPOINT}")
    except Exception:
        pass


__all__ = [
    "UNITS_JOBS_VERSION",
    "project_units_v7",
    "project_source_v7",
    "delete_units_v7",
    "derivation_coverage_v8",
]
