"""V6 typed retrieval lane — the grounded-fact index (SPEC_V6 §02,
V6-02.01/02.02/02.03; docs/v6_contracts.md §7.2).

The V6 hot path ranks *grounded typed memories and identifier hits*, not
a full source-transcript scan.  A typed memory is a T1 ``enrichment``
record — ``type``/``polarity``/temporal metadata plus the structured
``fields_json`` mentions (identifiers, entities, temporal expression) —
joined to its ``source_lexical_projection`` row; an identifier hit is an
``entity_postings`` mention.  Both carry byte offsets into the pinned
source revision, so every emitted candidate re-verifies its *span pins*
against the retained, HMAC-checked payload — the same ``byte_span``
contract ``enrichment.grounding._pin_ok`` enforces for the inspection
surface (V5-30.17/30.18).  An unpinnable record is
``unsupported_extraction`` — never delivered as a fact.

Two sub-lanes, merged per ``(source_id, revision)``:

- ``postings`` — exact ``entity_postings`` hits on the query's
  identifiers/entities (byte-exact ``=``; identifier hits FIRST per
  V6-02.01 — an exact identifier mention is its own candidate class);
- ``records`` — the eligible ``enrichment`` x projection join, scored on
  lexical term coverage, identifier/entity overlap, temporal match, and
  type affinity.

Eligibility mirrors the source lane exactly — this lane is a candidate
accelerator, never an authorization path (V6-02.03): generation fence
(``generation <= snapshot``), namespace partition, ``source_state``
lifecycle admissibility, quarantine holds (source + covering
source-envelope cascade), and purge suppression (``source`` and
``source_revision`` targets) all apply before a candidate is admitted.
Fusion ranks the admitted list; nothing here admits.

Coverage honesty: ``stats.details["uncovered"]`` counts signal-bearing
candidates the typed index can never deliver as grounded facts (no pins,
or pins that failed re-verification).  ``stats.details["unverified"]``
discloses pool keys whose verification was skipped because ``limit``
grounded hits were already found — an out-ranked tail, not a coverage
gap.  The caller treats a nonzero ``uncovered`` — or a partial/cut
scan, or a miss — as *thin coverage* and falls back to the existing
source lane; a fully-grounded eligible universe answers from this index
alone.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, Iterable, Optional, Sequence

from ...core.types import safe_json_loads
from ...enrichment.normalize import NORMALIZATION_VERSION, normalize_text
from ...storage.repos import SourcesRepo, has_table as _has_table
from .. import candidates as _cand
from .source_lane import (
    SourceLaneStats,
    _admissible_source_ids,
    _resolve_deadline,
    _snapshot_generation,
)

try:  # grounding/v1 — the pin-verification primitive this lane shares
    from ...enrichment.grounding import _pin_ok
except Exception:  # pragma: no cover - partial checkout
    _pin_ok = None


# Lane bounds — same scale as the source lane caps (§10 two-lane route).
TYPED_LANE_LIMIT = 40
TYPED_POSTING_LIMIT = 40

# Bounded-scan guard mirroring the source lane: bounds work, never rank.
_SCAN_CAP = 50_000

# Pin re-verification pool bound: scored candidates beyond it are
# reported ``truncated`` (more eligible material behind the bound),
# never silently dropped.
_VERIFY_CAP = 512

# Fusion signal names this lane emits (consumed by fusion_v1).
SIGNAL_LEXICAL = "lexical"
SIGNAL_IDENTIFIER = "identifier_hit"
SIGNAL_ENTITY_OVERLAP = "entity_overlap"
SIGNAL_TEMPORAL = "temporal_match"
SIGNAL_TYPE_AFFINITY = "type_affinity"

#: Query class → enrichment ``type`` values that interpret the intent
#: (interpretation metadata only — never admission, V5-30.02).  Classes
#: absent from the map carry no type prior at all.
_TYPE_AFFINITY: dict = {
    "preference": {"preference"},
    "procedural": {"procedure_hint", "plan", "decision"},
    "temporal": {"event", "state", "plan"},
    "entity": {"relationship", "state"},
}

#: Provisional per-candidate ordering for the bounded verification pool —
#: the frozen ``ranking/v1`` default weights applied to the raw signals
#: so the pool's survivors approximate the fused head.  Ranking itself
#: stays with fusion; this ordering only decides which candidates spend
#: verification budget first.
_PROVISIONAL_WEIGHTS: dict = {
    SIGNAL_IDENTIFIER: 2.0,
    SIGNAL_LEXICAL: 1.0,
    SIGNAL_ENTITY_OVERLAP: 0.5,
    SIGNAL_TEMPORAL: 0.4,
    SIGNAL_TYPE_AFFINITY: 0.3,
}

_TERM_RE = re.compile(r"[a-z0-9_]+")


@dataclass
class TypedHit:
    """One admitted typed candidate — identity, raw signals, verified pins.

    ``pins`` is the re-verified ``byte_span`` pin list (``kind`` /
    ``source_id`` / ``revision`` / ``start`` / ``end`` / ``value``) —
    every delivered typed line carries the grounding evidence that makes
    it a fact (V6-02.02).  ``mem_type``/``polarity``/``valid_time`` carry
    the enrichment interpretation labels for delivery decoration; they
    are never authority.
    """

    source_id: str
    revision: int
    signals: dict = field(default_factory=dict)
    lanes: dict = field(default_factory=dict)
    pins: list = field(default_factory=list)
    mem_type: Optional[str] = None
    polarity: Optional[str] = None
    valid_time: Optional[str] = None

    @property
    def key(self) -> tuple:
        return (self.source_id, self.revision)


# ---------------------------------------------------------------------------
# analysis normalization — accepts the QueryAnalysis object, a mapping, or
# None (re-derived from the query through the same seams the facade uses)
# ---------------------------------------------------------------------------


@dataclass
class _QueryParts:
    terms: list = field(default_factory=list)      # raw query tokens
    folded: list = field(default_factory=list)     # norm/v1-folded tokens
    idents: list = field(default_factory=list)     # exact identifier values
    ents: list = field(default_factory=list)       # exact entity values
    qclass: Optional[str] = None
    markers: list = field(default_factory=list)    # temporal markers


def _ident_value(entry: Any) -> Optional[str]:
    """One ``(kind, value)``/mapping/bare-string identifier → its value."""
    if isinstance(entry, Mapping):
        v = entry.get("value")
        return str(v) if v else None
    if isinstance(entry, (tuple, list)) and len(entry) >= 2:
        return str(entry[1]) if entry[1] else None
    if isinstance(entry, str) and entry:
        return entry
    return None


def _fold(terms: Iterable[str]) -> list:
    """Fold caller terms into the ``norm/v1`` projection token space —
    the identical contract the source lane's ``_fold_query_terms``
    applies: query-side only, stored bytes never rewritten."""
    out: list = []
    for term in terms:
        folded = normalize_text(str(term or ""), version=NORMALIZATION_VERSION)
        out.extend(t for t in folded.split() if t)
    return sorted(set(out))


def _query_parts(query: Any, analysis: Any) -> _QueryParts:
    """Normalize the caller's query analysis into lane-local parts."""
    parts = _QueryParts()
    terms: list = []
    markers: list = []
    if analysis is not None:
        get = (
            (lambda k, d=None: analysis.get(k, d))
            if isinstance(analysis, Mapping)
            else (lambda k, d=None: getattr(analysis, k, d))
        )
        terms = [str(t) for t in (get("terms") or ()) if t]
        for entry in get("identifiers") or ():
            v = _ident_value(entry)
            if v:
                parts.idents.append(v)
        for e in get("entities") or ():
            if e:
                parts.ents.append(str(e))
        qclass = get("primary") or get("qclass")
        parts.qclass = str(qclass) if qclass else None
        markers = [str(m) for m in (get("temporal_markers") or ()) if m]
    if not terms and isinstance(query, str) and query:
        # Analysis absent/unshaped — derive with the same fallback the
        # facade uses when query_analysis/v1 is unprovisioned.
        terms = [t for t in _TERM_RE.findall(query.lower()) if len(t) > 1]
    parts.terms = terms
    parts.idents = sorted(set(parts.idents))
    parts.ents = sorted(set(parts.ents))
    parts.folded = _fold(terms)
    parts.markers = _fold(markers)
    return parts


# ---------------------------------------------------------------------------
# eligibility gates — the same reads the facade/source lane apply
# ---------------------------------------------------------------------------


def _held_source_ids(conn: sqlite3.Connection, ids: Iterable[str]) -> set:
    """Source ids under a live quarantine hold — mirrors
    ``Memory._held_source_ids_in``: direct ``source`` holds plus the
    covering ``source_envelope`` cascade (V3-14.10).  Raises propagate;
    the caller's policy is fail-closed."""
    ids = sorted({str(i) for i in ids if i})
    if not ids:
        return set()
    if not _has_table(conn, "quarantine"):
        return set()
    if conn.execute(
        "SELECT 1 FROM quarantine"
        " WHERE state IN ('pending','suppressed') LIMIT 1"
    ).fetchone() is None:
        return set()
    ph = ",".join("?" for _ in ids)
    held = {
        str(r[0])
        for r in conn.execute(
            "SELECT object_id FROM quarantine"
            f" WHERE object_kind = 'source' AND object_id IN ({ph})"
            " AND state IN ('pending','suppressed')",
            ids,
        )
    }
    held |= {
        str(r[0])
        for r in conn.execute(
            "SELECT se.source_id FROM quarantine q"
            " JOIN source_envelopes se"
            "  ON se.envelope_id = q.object_id"
            f" WHERE q.object_kind = 'source_envelope'"
            f"  AND se.source_id IN ({ph})"
            "  AND q.state IN ('pending','suppressed')",
            ids,
        )
    }
    return held


def _suppressed_pairs(
    store: Any, conn: sqlite3.Connection, pairs: Iterable[tuple]
) -> set:
    """``(source_id, revision)`` keys under a suppressing purge —
    ``source`` and ``source_revision`` target forms, mirroring
    ``Ingester._source_suppressed``'s states and object-id conventions."""
    wanted = sorted({(str(s), int(r)) for s, r in pairs})
    if not wanted:
        return set()
    out: set = set()
    suppressed_sids = _cand._suppressed(
        store, conn, "source", [s for s, _ in wanted]
    )
    out.update(k for k in wanted if k[0] in suppressed_sids)
    suppressed_revs = _cand._suppressed(
        store,
        conn,
        "source_revision",
        [f"{s}:{r}" for s, r in wanted],
    )
    out.update(k for k in wanted if f"{k[0]}:{k[1]}" in suppressed_revs)
    return out


# ---------------------------------------------------------------------------
# pins + record parsing
# ---------------------------------------------------------------------------


def _byte_pin(source_id: str, revision: int, start: Any, end: Any,
              value: Any) -> dict:
    """One ``byte_span`` pin in the grounding contract's wire shape."""
    return {
        "kind": "byte_span",
        "source_id": source_id,
        "revision": revision,
        "start": start,
        "end": end,
        "value": value,
    }


# ``fields_json`` is immutable per (source, revision) — the producer
# wrote it once at enrichment time — so the parsed mapping is safely
# reusable by content.  A bounded content-keyed cache keeps a warm
# query from re-parsing every row it already saw this process.
_FIELDS_CACHE: dict = {}
_FIELDS_CACHE_CAP = 8192


def _parse_fields(raw: Any) -> dict:
    """Cached ``fields_json`` decode — keyed by the raw text itself, so
    revision changes simply re-key (stale entries can never alias)."""
    if not raw or not isinstance(raw, str):
        return {}
    cached = _FIELDS_CACHE.get(raw)
    if cached is not None:
        return cached
    try:
        parsed = safe_json_loads(raw)
    except Exception:
        parsed = None
    out = parsed if isinstance(parsed, Mapping) else {}
    if len(_FIELDS_CACHE) >= _FIELDS_CACHE_CAP:
        _FIELDS_CACHE.clear()
    _FIELDS_CACHE[raw] = out
    return out


# Derived per-fields signal sets — same immutability argument as the
# parse cache: one build per unique ``fields_json`` ever, shared across
# every query the record answers.
_FIELDSETS_CACHE: dict = {}


def _field_sets(raw: Any) -> tuple:
    """``(identifier_values, entity_values, temporal_expr_tokens)`` for
    one serialized ``fields_json`` — the scoring loop's per-candidate
    inputs, computed once per record rather than once per query."""
    if not raw or not isinstance(raw, str):
        return frozenset(), frozenset(), None
    cached = _FIELDSETS_CACHE.get(raw)
    if cached is not None:
        return cached
    fields = _parse_fields(raw)
    idents: set = set()
    ents: set = set()
    for collection, sink in (("identifiers", idents), ("entities", ents)):
        for m in fields.get(collection) or ():
            if isinstance(m, Mapping) and isinstance(m.get("value"), str):
                sink.add(m["value"])
    expr_tokens: Optional[frozenset] = None
    temporal = fields.get("temporal")
    if isinstance(temporal, Mapping):
        expression = temporal.get("expression")
        if isinstance(expression, str) and expression:
            toks = frozenset(_fold([expression]))
            if toks:
                expr_tokens = toks
    out = (frozenset(idents), frozenset(ents), expr_tokens)
    if len(_FIELDSETS_CACHE) >= _FIELDS_CACHE_CAP:
        _FIELDSETS_CACHE.clear()
    _FIELDSETS_CACHE[raw] = out
    return out


def _record_pins(source_id: str, revision: int, fields: Mapping) -> list:
    """``byte_span`` pins for one enrichment record's serialized mentions.

    Malformed entries produce malformed pins *on purpose* — they fail
    ``_pin_ok`` and withhold the whole record (a corrupt extraction is
    ``unsupported_extraction``, never a fact), rather than silently
    shaping the record into something the producer never wrote.
    """
    pins: list = []
    if not isinstance(fields, Mapping):
        return pins
    seen: set = set()

    def _add(start: Any, end: Any, value: Any) -> None:
        pin = _byte_pin(source_id, revision, start, end, value)
        mark = (pin["value"], pin["start"], pin["end"])
        if mark not in seen:
            seen.add(mark)
            pins.append(pin)

    for collection in ("identifiers", "entities"):
        for mention in fields.get(collection) or ():
            if not isinstance(mention, Mapping):
                continue
            _add(mention.get("start"), mention.get("end"), mention.get("value"))
    temporal = fields.get("temporal")
    if isinstance(temporal, Mapping):
        expression = temporal.get("expression")
        if isinstance(expression, str) and expression:
            _add(temporal.get("start"), temporal.get("end"), expression)
    return pins


_OFFSETS_CACHE: dict = {}
_OFFSETS_CACHE_MAX = 8192


def _offsets_list(offsets: Any) -> Any:
    """Parse the ``entity_postings`` offset list — content-keyed cache.

    Rows are immutable per write, so the parsed list is stable for the
    raw column text; the same postings rows are re-read every search.
    Returned lists are only iterated (never mutated) by callers.
    """
    if not isinstance(offsets, str):
        return offsets
    cached = _OFFSETS_CACHE.get(offsets)
    if cached is not None:
        return cached
    try:
        offs = safe_json_loads(offsets)
    except Exception:
        offs = []
    if len(_OFFSETS_CACHE) >= _OFFSETS_CACHE_MAX:
        _OFFSETS_CACHE.clear()
    _OFFSETS_CACHE[offsets] = offs
    return offs


def _posting_pins(
    source_id: str, revision: int, entity: Any, offsets: Any
) -> list:
    """``byte_span`` pins from one ``entity_postings`` offset list."""
    pins: list = []
    if not isinstance(entity, str) or not entity:
        return pins
    offs = _offsets_list(offsets)
    for off in offs or ():
        if isinstance(off, (list, tuple)) and len(off) == 2:
            pins.append(_byte_pin(source_id, revision, off[0], off[1], entity))
    return pins


def _mention_values(fields: Mapping, collection: str) -> set:
    """Distinct serialized mention values of one kind."""
    out: set = set()
    if not isinstance(fields, Mapping):
        return out
    for m in fields.get(collection) or ():
        if isinstance(m, Mapping) and isinstance(m.get("value"), str):
            out.add(m["value"])
    return out


def _temporal_match(
    temporal: Mapping, folded_terms: Sequence[str], folded_markers: Sequence[str]
) -> float:
    """Temporal-expression coverage vs. the query's folded terms+markers."""
    if not isinstance(temporal, Mapping):
        return 0.0
    expression = temporal.get("expression")
    if not isinstance(expression, str) or not expression:
        return 0.0
    expr_tokens = set(_fold([expression]))
    if not expr_tokens:
        return 0.0
    asked = set(folded_terms) | set(folded_markers)
    if not asked:
        return 0.0
    return len(expr_tokens & asked) / len(expr_tokens)


def _provisional(signals: Mapping) -> float:
    """Deterministic verification-pool order (not the ranking)."""
    return sum(
        _PROVISIONAL_WEIGHTS.get(name, 0.0) * float(value)
        for name, value in signals.items()
    )


# ---------------------------------------------------------------------------
# the lane
# ---------------------------------------------------------------------------


def typed_candidates(
    conn: sqlite3.Connection,
    *,
    namespace: Optional[str],
    query: Any,
    analysis: Any = None,
    generation: Any = None,
    limit: int = TYPED_LANE_LIMIT,
    store: Any = None,
    deadline: Any = None,
    deadline_ms: Optional[float] = None,
) -> tuple:
    """Grounded typed-memory + identifier-hit candidates (V6-02.01/02).

    Returns ``(hits, stats)`` — ``hits`` are :class:`TypedHit` objects
    carrying raw fusion signals plus the re-verified span pins; ``stats``
    is a :class:`SourceLaneStats` with honest coverage counters,
    ``details["uncovered"]`` being the thin-coverage signal the facade's
    source-lane fallback reads.

    ``analysis`` accepts the ``query_analysis/v1`` object, a mapping with
    ``terms``/``identifiers``/``entities``/``primary``/
    ``temporal_markers``, or ``None`` (terms are re-derived from
    ``query``).  ``generation`` accepts an int, a snapshot mapping/object,
    or ``None`` — resolved against this connection's committed
    ``projection_generation`` exactly like the source lane.  ``store``
    supplies the verified payload read + HMAC pins require; without it
    the lane cannot ground anything and reports ``unavailable``.
    """
    dl = _resolve_deadline(deadline, deadline_ms)
    stats = SourceLaneStats("typed")

    if not _has_table(conn, "source_lexical_projection"):
        stats.status = "unavailable"
        stats.reason = "no_source_lexical_projection"
        return [], stats
    have_enrich = _has_table(conn, "enrichment")
    have_postings = _has_table(conn, "entity_postings")
    if not (have_enrich or have_postings):
        stats.status = "unavailable"
        stats.reason = "no_typed_index"
        return [], stats
    if namespace is None:
        stats.status = "unavailable"
        stats.reason = "unscoped_request"
        return [], stats
    if store is None or _pin_ok is None:
        # Grounding requires verified canonical bytes + the pin checker —
        # without them nothing the lane emits could be a grounded fact.
        stats.status = "unavailable"
        stats.reason = "no_verified_reads"
        return [], stats

    gen = _snapshot_generation(conn, generation)
    if gen is None:
        stats.warnings.append("generation_unfenced")
    gen_pred = " AND generation <= ?" if gen is not None else ""
    gen_params: list = [] if gen is None else [gen]
    p_gen_pred = " AND p.generation <= ?" if gen is not None else ""

    q = _query_parts(query, analysis)
    q_ident_set = set(q.idents)
    q_ent_set = set(q.ents)
    q_asked = set(q.folded) | set(q.markers)

    def _cut() -> bool:
        """Deadline check with honest partial marking."""
        if dl.expired():
            stats.status = "partial"
            stats.reason = stats.reason or "deadline"
            stats.deadline_exceeded = True
            if "scan_incomplete:deadline" not in stats.warnings:
                stats.warnings.append("scan_incomplete:deadline")
            return True
        return False

    # ---- postings sub-lane: exact identifier/entity hits FIRST --------
    # (V6-02.01 — the identifier hit is its own candidate class; byte-
    # exact match, never folded — V5-30.17.)
    postings: dict = {}  # (sid, rev) -> {"identifier": set, "entity": set, "pins": list}
    post_stats = SourceLaneStats("postings")
    wanted_values = sorted(set(q.idents) | set(q.ents))
    if have_postings and wanted_values:
        value_kind = {v: "entity" for v in q.ents}
        value_kind.update({v: "identifier" for v in q.idents})
        stop = False
        for chunk in _cand._chunks(wanted_values, _cand._IN_CHUNK):
            if stop:
                break
            cur = conn.execute(
                "SELECT entity, entity_kind, source_id, revision, offsets"
                " FROM entity_postings"
                f" WHERE entity IN ({','.join('?' * len(chunk))})"
                " AND namespace = ?"
                f"{gen_pred} ORDER BY source_id, revision",
                [*chunk, namespace, *gen_params],
            )
            while True:
                if _cut():
                    post_stats.status = "partial"
                    post_stats.reason = "deadline"
                    post_stats.deadline_exceeded = True
                    stop = True
                    break
                batch = cur.fetchmany(512)
                if not batch:
                    break
                for entity, ekind, sid, rev, offsets in batch:
                    post_stats.candidates_examined += 1
                    entry = postings.setdefault(
                        (str(sid), int(rev)),
                        {"identifier": set(), "entity": set(), "pins": []},
                    )
                    entry[value_kind[entity]].add(entity)
                    entry["pins"].extend(
                        _posting_pins(str(sid), int(rev), entity, offsets)
                    )
        post_stats.scored = len(postings)
        post_stats.returned = len(postings)
    elif have_postings:
        post_stats.reason = "empty_identifiers"
    else:
        post_stats.status = "unavailable"
        post_stats.reason = "no_entity_postings"

    # ---- records sub-lane: token prefilter → restricted enrich fetch --
    # The evidence gate requires ≥1 of {identifier, entity, lexical}.
    # Identifier/entity signals are reachable through ``entity_postings``
    # (built from the same serialized mentions), and the lexical arm
    # needs only ``p.tokens`` — so the expensive ``fields_json`` parse +
    # pin construction runs only for records that can actually signal.
    records: dict = {}  # (sid, rev) -> record dict
    rec_stats = SourceLaneStats("records")

    token_hits: dict = {}  # (sid, rev) -> matched folded-term count
    folded_set = set(q.folded or ())
    # The lexical arm is an exact-token intersection — pushed into SQL
    # as space-delimited ``LIKE`` filters so only rows that CAN match
    # reach Python; with no folded terms there is no lexical arm and
    # the scan is skipped entirely (postings still cover ident/entity).
    if folded_set:
        like = " OR ".join(
            "' ' || p.tokens || ' ' LIKE ? ESCAPE '\\'" for _ in folded_set
        )
        like_params = [
            "% " + str(t).replace("\\", "\\\\").replace(
                "%", "\\%").replace("_", "\\_") + " %"
            for t in sorted(folded_set)
        ]
        cur = conn.execute(
            "SELECT p.source_id, p.revision, p.tokens"
            " FROM source_lexical_projection p"
            f" WHERE p.scope_id = ?{p_gen_pred} AND ({like})"
            " ORDER BY p.source_id, p.revision",
            [namespace, *gen_params, *like_params],
        )
        stop = False
        while True:
            if _cut():
                rec_stats.status = "partial"
                rec_stats.reason = "deadline"
                rec_stats.deadline_exceeded = True
                stop = True
                break
            batch = cur.fetchmany(512)
            if not batch:
                break
            for sid, rev, toks in batch:
                rec_stats.candidates_examined += 1
                if rec_stats.candidates_examined > _SCAN_CAP:
                    rec_stats.truncated = True
                    rec_stats.warnings.append("eligible_scan_capped")
                    stop = True
                    break
                if toks:
                    matched = len(folded_set & set(str(toks).split()))
                    if matched:
                        token_hits[(str(sid), int(rev))] = matched
            if stop:
                break
    else:
        stop = False

    # Enrichment rows only for records that can signal — posting-hit
    # pairs plus token matches.  Deterministic producer pick: the pinned
    # producer's row first (the old join's first-row-wins order).
    cand_keys = sorted(set(token_hits) | set(postings))
    if have_enrich and cand_keys and not stop:
        for chunk in _cand._chunks(cand_keys, _cand._PAIR_CHUNK):
            if _cut():
                rec_stats.status = "partial"
                rec_stats.reason = "deadline"
                rec_stats.deadline_exceeded = True
                break
            where = " OR ".join(
                "(e.source_id = ? AND e.revision = ?)" for _ in chunk
            )
            flat = [v for pair in chunk for v in pair]
            try:
                rows = conn.execute(
                    "SELECT e.source_id, e.revision, e.type, e.polarity,"
                    " e.event_at, e.anchor_at, e.fields_json, e.producer"
                    " FROM enrichment e"
                    f" WHERE ({where})"
                    " ORDER BY e.source_id, e.revision,"
                    " (e.producer = 'enrich/v1') DESC, e.producer",
                    flat,
                ).fetchall()
            except sqlite3.Error:
                rows = []
            for row in rows:
                key = (str(row[0]), int(row[1]))
                if key in records:
                    continue  # deterministic producer pick — first wins
                records[key] = {
                    "type": row[2],
                    "polarity": row[3],
                    "event_at": row[4],
                    "anchor_at": row[5],
                    "fields": _parse_fields(row[6]),
                    "fields_raw": row[6],
                    "enriched": True,
                }
    # A signalling projection row without an enrichment record is still
    # a candidate — it matched but cannot ground (honestly uncovered).
    for key in cand_keys:
        if key not in records:
            records[key] = {
                "type": None,
                "polarity": None,
                "event_at": None,
                "anchor_at": None,
                "fields": {},
                "fields_raw": None,
                "enriched": False,
            }

    # ---- eligibility gates (mirrors the source lane's admission) ------
    all_sids = {k[0] for k in cand_keys}
    adm_res = _admissible_source_ids(conn, all_sids)
    admissible = adm_res[0] if adm_res is not None else None
    try:
        held = _held_source_ids(conn, all_sids)
    except Exception:
        # Fail closed: an unreadable hold table withholds every id.
        held = set(all_sids)
        stats.warnings.append("hold_check_failed")
    all_pairs = set(records) | set(postings)
    try:
        suppressed = _suppressed_pairs(store, conn, all_pairs)
    except Exception:
        suppressed = set(all_pairs)
        stats.warnings.append("suppression_check_failed")

    def _admitted(key: tuple) -> bool:
        sid = key[0]
        if sid in held or key in suppressed:
            return False
        if admissible is not None and admissible.get(sid, True) is False:
            return False
        return True

    eligible_records = [k for k in records if _admitted(k)]
    stats.eligible = len(eligible_records)
    rec_stats.eligible = len(eligible_records)

    # ---- score records against the query ------------------------------
    # Pins are deferred to the verification pass: signals decide the
    # provisional order, grounding decides delivery — a candidate that
    # never reaches verification needs no pin material at all.
    candidates: dict = {}  # key -> {"signals": dict, "pins": list, "rec": dict|None}
    for key in eligible_records:
        rec = records[key]
        signals: dict = {}
        if key in token_hits:
            signals[SIGNAL_LEXICAL] = float(token_hits[key])
        ident_vals, ent_vals, expr_tokens = _field_sets(
            rec["fields_raw"]
        )
        ident_hits = q_ident_set & ident_vals
        if ident_hits:
            signals[SIGNAL_IDENTIFIER] = float(len(ident_hits))
        ent_hits = q_ent_set & ent_vals
        if ent_hits:
            signals[SIGNAL_ENTITY_OVERLAP] = float(len(ent_hits))
        if expr_tokens and q_asked:
            tm = len(expr_tokens & q_asked) / len(expr_tokens)
            if tm:
                signals[SIGNAL_TEMPORAL] = tm
        mtype = rec["type"]
        if mtype and mtype in _TYPE_AFFINITY.get(str(q.qclass or ""), ()):
            signals[SIGNAL_TYPE_AFFINITY] = 1.0
        # The evidence gate: a typed candidate needs at least one real
        # match signal — the support verdict could never keep a
        # signal-free record, so it never enters the admitted pool.
        if not (
            signals.get(SIGNAL_IDENTIFIER)
            or signals.get(SIGNAL_ENTITY_OVERLAP)
            or signals.get(SIGNAL_LEXICAL)
        ):
            continue
        entry = candidates.setdefault(
            key, {"signals": {}, "pins": [], "rec": rec}
        )
        entry["signals"].update(signals)
        rec_stats.scored += 1

    # ---- merge posting hits (identifier hits FIRST) -------------------
    for key, post in postings.items():
        if not _admitted(key):
            continue
        entry = candidates.setdefault(
            key, {"signals": {}, "pins": [], "rec": None}
        )
        if post["identifier"]:
            entry["signals"][SIGNAL_IDENTIFIER] = max(
                entry["signals"].get(SIGNAL_IDENTIFIER, 0.0),
                float(len(post["identifier"])),
            )
        if post["entity"]:
            entry["signals"][SIGNAL_ENTITY_OVERLAP] = max(
                entry["signals"].get(SIGNAL_ENTITY_OVERLAP, 0.0),
                float(len(post["entity"])),
            )
        entry["pins"].extend(post["pins"])

    stats.scored = len(candidates)
    stats.candidates_examined = (
        rec_stats.candidates_examined + post_stats.candidates_examined
    )

    # ---- bounded pin re-verification over canonical bytes -------------
    ordered = sorted(
        candidates,
        key=lambda k: (
            -_provisional(candidates[k]["signals"]),
            k[0],
            -k[1],
        ),
    )
    overflow = max(0, len(ordered) - _VERIFY_CAP)
    if overflow:
        stats.truncated = True
        stats.details["overflow"] = overflow
    pool = ordered[:_VERIFY_CAP]

    # Verification is lazy in provisional order: only enough pool keys to
    # fill ``limit`` grounded hits pay the HMAC-checked payload read —
    # the unverified tail is disclosed, never silently dropped (the
    # delivered top-limit set is identical either way).
    want = max(int(limit), 1)
    vchunk = max(64, want)
    unpinnable = 0
    unpinned = 0
    examined = 0
    hits: list = []
    idx = 0
    while idx < len(pool) and len(hits) < want and not _cut():
        chunk = pool[idx:idx + vchunk]
        idx += len(chunk)

        # Record pins for this chunk — fields were parsed at scoring;
        # dedupe against the pins the key already carries.
        for key in chunk:
            entry = candidates[key]
            rec = entry["rec"]
            if rec is None:
                continue
            have = {
                (p.get("value"), p.get("start"), p.get("end"))
                for p in entry["pins"]
            }
            for pin in _record_pins(key[0], key[1], rec["fields"]):
                mark = (pin["value"], pin["start"], pin["end"])
                if mark in have:
                    continue
                have.add(mark)
                entry["pins"].append(pin)

        # Union posting pins for chunk keys — the postings sub-lane only
        # fetched rows matching the query; a grounded record pins every
        # serialized mention, so fetch the remaining postings.
        if have_postings:
            covered = {
                (p.get("value"), p.get("start"), p.get("end"))
                for k in chunk
                for p in candidates[k]["pins"]
            }
            for pair_chunk in _cand._chunks(chunk, _cand._PAIR_CHUNK):
                if _cut():
                    break
                where = " OR ".join(
                    "(source_id = ? AND revision = ?)" for _ in pair_chunk
                )
                flat = [v for pair in pair_chunk for v in pair]
                try:
                    rows = conn.execute(
                        "SELECT source_id, revision, entity, offsets"
                        " FROM entity_postings"
                        f" WHERE ({where}) AND namespace = ?{gen_pred}",
                        [*flat, namespace, *gen_params],
                    ).fetchall()
                except sqlite3.Error:
                    rows = []
                for sid, rev, entity, offsets in rows:
                    for pin in _posting_pins(
                        str(sid), int(rev), entity, offsets
                    ):
                        mark = (
                            pin.get("value"),
                            pin.get("start"),
                            pin.get("end"),
                        )
                        if mark in covered:
                            continue
                        covered.add(mark)
                        candidates[(str(sid), int(rev))]["pins"].append(
                            pin
                        )

        # Verified canonical bytes for this chunk — the same
        # HMAC-checked read the delivery path uses (purged revisions
        # verify to ``b''`` and therefore fail every pin).
        try:
            verified, _corrupt = SourcesRepo(store).payload_many(
                chunk, conn=conn
            )
            payloads = dict(verified)
        except Exception:
            # An unreadable byte store cannot ground anything — fail
            # closed, no candidate survives verification.
            payloads = {}
            stats.warnings.append("payload_read_failed")

        for key in chunk:
            if len(hits) >= want:
                break
            examined += 1
            entry = candidates[key]
            sid, rev = key
            pins = entry["pins"]
            body = payloads.get(key)
            if pins and _pin_ok is not None and body:
                grounded = all(
                    _pin_ok(
                        p,
                        lambda r, b=body: b if int(r) == int(rev) else None,
                        store.hmac,
                    )
                    for p in pins
                )
            elif not pins:
                # No serialized mentions to pin — outside the grounded
                # typed universe (unpinnable, never a fact).
                unpinnable += 1
                continue
            else:
                grounded = False
            if not grounded:
                # Unpinnable under re-verification —
                # unsupported_extraction, never a delivered fact
                # (V6-02.02).
                unpinned += 1
                continue
            hit = TypedHit(sid, rev)
            hit.signals.update(entry["signals"])
            hit.pins = [dict(p) for p in pins]
            rec = entry["rec"]
            if rec is not None:
                hit.mem_type = str(rec["type"]) if rec["type"] else None
                hit.polarity = (
                    str(rec["polarity"]) if rec["polarity"] else None
                )
                hit.valid_time = rec["event_at"] or rec["anchor_at"]
            hit.lanes["typed"] = len(hits) + 1
            hits.append(hit)

    stats.returned = len(hits)
    stats.details["unverified"] = len(pool) - examined
    uncovered = unpinnable + unpinned
    stats.details["unpinnable"] = unpinnable
    stats.details["unpinned"] = unpinned
    stats.details["uncovered"] = uncovered
    stats.details["generation"] = gen
    stats.details["records"] = rec_stats.to_dict()
    stats.details["postings"] = post_stats.to_dict()
    for sub in (rec_stats, post_stats):
        for w in sub.warnings:
            if w not in stats.warnings:
                stats.warnings.append(w)
        if sub.truncated:
            stats.truncated = True
    return hits, stats


__all__ = [
    "TYPED_LANE_LIMIT",
    "TYPED_POSTING_LIMIT",
    "TypedHit",
    "typed_candidates",
]
