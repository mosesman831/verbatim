"""V5 source-projection retrieval lane (SPEC_V5 §10–§12, §30, §31).

The source lane surfaces *source records* — the retained bytes a memory
aliases (V5-06.11) — from the derived v5 projection tables:

- ``source_lexical_projection`` — space-joined normalized tokens +
  ``doc_len`` per ``(source_id, revision)``;
- ``source_vectors`` — pinned-encoder float32le vectors, contiguous per
  namespace/generation (V5-33.04);
- ``entity_postings`` — exact identifier/entity mentions with byte
  offsets, case/punctuation/version preserved (V5-30.14, V5-30.17);
- ``source_state`` — ``source_state/v1`` lifecycle control artifact
  (§14.3) supplying timeline labels.

Contract (docs/v5_contracts.md §6): ``source_candidates`` runs exact
eligible-corpus BM25 (F4-11 statistics discipline — N/df/avgdl measured
over the request-eligible set E, never over the candidate set or the
global table, V5-11.03/11.04), hashing-similarity candidates, and exact
identifier/entity posting hits. Fusion ranks the admitted list; nothing
here admits (V5-31.05).

Namespace note (contract ambiguity): ``source_lexical_projection`` is
keyed by ``scope_id`` while the other v5 tables key on ``namespace``.
The ``local_memory`` profile realizes a namespace as one scope partition
token, so the namespace-only path restricts the projection on
``scope_id = <namespace>``. Callers that compute the authoritative
request-eligible universe pass ``eligible_ids`` and bypass the mapping
entirely — that is the preferred path under V5-11.

Eligibility: ``eligible_ids`` is the caller-computed eligible universe E
(source ids, ``(source_id, revision)`` pairs, or objects exposing those
attributes). When it is provided it restricts every stage; when ``None``
the ``namespace`` partition is the restriction. Either way the pinned
``snapshot`` generation fences derived rows so a concurrent rebuild
cannot mix generations into one result (V5-11.10).

Honest coverage: every function returns ``(hits, stats)`` where stats
carries ``eligible``, ``candidates_examined``, ``scored``, ``returned``,
``truncated`` (a bound was hit with more eligible rows behind it), and
``partial``/``deadline_exceeded`` when the cooperative deadline stopped
a scan mid-corpus — a cut scan never reports complete coverage
(V4-27.09 carried forward).
"""

from __future__ import annotations

import hashlib
import heapq
import math
import sqlite3
import struct
import threading
from array import array
from collections import Counter, OrderedDict
from operator import mul as _mul
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

from ...core.types import VerbatimError, safe_json_loads
from ...embeddings.codec import Float32Codec
from ...embeddings import vectors as _vec
from ...enrichment.normalize import NORMALIZATION_VERSION, normalize_text
from ...storage.repos import has_table as _has_table
from .. import candidates as _cand

try:  # storage worker's v5 repos land in parallel — optional seam
    from ...storage import repos_v5 as _repos_v5  # noqa: F401  # type: ignore
except Exception:  # pragma: no cover - absent until schema_v5 lands
    _repos_v5 = None

# Lane bounds (§10 two-lane route; same scale as the v3 lane caps).
SOURCE_LANE_LIMIT = 40
SOURCE_LEXICAL_LIMIT = 40
SOURCE_VECTOR_LIMIT = 40
SOURCE_POSTING_LIMIT = 40
SOURCE_TIMELINE_LIMIT = 200

# Bounded-scan guard: a projection scan never reads past this many rows
# without reporting it (rowid-ordered bound like _cand._POSTING_CAP —
# it bounds work, never selects rank).
_SCAN_CAP = 50_000

# BM25 constants are the versioned ranking contract — reuse the F4-11
# values verbatim so the source lane scores on the same scale as the
# claim lane (V5-11: constants belong to the ranking contract).
_BM25_K1 = _cand._BM25_K1
_BM25_B = _cand._BM25_B
_IN_CHUNK = _cand._IN_CHUNK

# Fusion signal names this lane emits (consumed by fusion_v1).
SIGNAL_LEXICAL = "lexical"
SIGNAL_SIMILARITY = "similarity"
SIGNAL_IDENTIFIER = "identifier_hit"
SIGNAL_ENTITY_OVERLAP = "entity_overlap"

ObjectKey = tuple  # (source_id, revision)


# ---------------------------------------------------------------------------
# result types
# ---------------------------------------------------------------------------


@dataclass
class SourceHit:
    """One admitted source candidate with its raw per-signal values.

    ``signals`` maps fusion signal names (``lexical``, ``similarity``,
    ``identifier_hit``, ``entity_overlap``) to the raw value the lane
    measured — normalization is fusion's job (``ranking/v1``). ``lanes``
    records the 1-based rank each sub-lane assigned, for provenance.
    """

    source_id: str
    revision: int
    signals: dict = field(default_factory=dict)
    lanes: dict = field(default_factory=dict)

    @property
    def key(self) -> ObjectKey:
        return (self.source_id, self.revision)


@dataclass
class SourceLaneStats:
    """Coverage + honesty counters for one lane call.

    ``status``: ``ok`` | ``partial`` | ``unavailable`` | ``skipped``.
    ``truncated`` marks a bound (``limit``/scan cap) reached with more
    eligible material behind it; ``partial`` marks a deadline cut.
    ``eligible`` is the size of the eligible universe actually measured —
    the BM25 statistics base — never a global table count (V5-11.04).
    """

    lane: str
    status: str = "ok"
    reason: Optional[str] = None
    eligible: int = 0
    candidates_examined: int = 0
    scored: int = 0
    returned: int = 0
    truncated: bool = False
    deadline_exceeded: bool = False
    warnings: list = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "lane": self.lane,
            "status": self.status,
            "reason": self.reason,
            "eligible": self.eligible,
            "candidates_examined": self.candidates_examined,
            "scored": self.scored,
            "returned": self.returned,
            "truncated": self.truncated,
            "deadline_exceeded": self.deadline_exceeded,
            "warnings": list(self.warnings),
            "details": dict(self.details),
        }


def _ph(n: int) -> str:
    return ",".join("?" * n)


def _gen_pred(alias: Optional[str] = None) -> str:
    """Generation fence predicate for v5 source-projection tables.

    The fence is ``generation <= <snapshot>``, never ``=``: rows are
    *incrementally* maintained (V5-33.03 — a write updates its rows
    without rebuilding the namespace) and stamped with the committed
    ``projection_generation`` at write time. Lifecycle transitions,
    erasure, purge, and closure paths bump the meta counter inside their
    own commits *without* rewriting unaffected rows (the same hazard
    V2-19.10 documents for claim FTS), so equality would strand every
    row minted before the most recent bump — an empty lane, not a
    complete view. ``<=`` still fences what the fence exists for: a row
    minted under a generation the snapshot has not reached stays
    invisible (writers cannot produce one — the STALE_DEPENDENCY pin
    guard rejects it), and an atomic rebuild that deletes+rewrites rows
    at a newer generation is still observed whole-or-absent by snapshot
    isolation.
    """
    col = f"{alias}.generation" if alias else "generation"
    return f" AND {col} <= ?"


#: Dispositions that can never be the current answer (§14.3, V5-14.09 —
#: ``recorded`` is pre-admission registration; ``corrected``/``retracted``/
#: ``archived``/``erased`` carry no current head). ``superseded`` is
#: admitted only while its declared ``effective_at`` boundary is still in
#: the future (V5-14.12 — the predecessor answers until the boundary,
#: evaluated at read time); ``active`` answers inside its valid window.
_NEVER_CURRENT = frozenset({
    "recorded", "corrected", "retracted", "archived", "erased",
})


def _current_window_ok(
    disposition: str,
    effective_at: Any,
    valid_from: Any,
    valid_to: Any,
    now_ts: float,
) -> bool:
    """Read-time currency test for one ``source_state`` row (V5-14.12).

    ``active`` answers inside ``[valid_from, valid_to)``; ``superseded``
    answers only until its declared ``effective_at`` boundary; every
    other known disposition never answers. Unparseable bounds are treated
    as absent for ``superseded`` (boundary unknown → successor already
    owns the answer) and open for ``active`` (a window we cannot read is
    no evidence the record left currency). Unknown disposition values
    are admissible — a lifecycle state this build does not recognize is
    not an assertion of non-currency.
    """
    if disposition == "active":
        start = _rfc3339_ts(valid_from)
        if start is not None and now_ts < start:
            return False
        end = _rfc3339_ts(valid_to)
        if end is not None and now_ts >= end:
            return False
        return True
    if disposition == "superseded":
        boundary = _rfc3339_ts(effective_at)
        return boundary is not None and now_ts < boundary
    if disposition in _NEVER_CURRENT:
        return False
    return True


#: Bounded per-snapshot memo of the whole ``source_state`` table, keyed
#: by ``(id(conn), PRAGMA data_version)`` on ``PRAGMA query_only``
#: reader connections — the same snapshot discipline the governance
#: verdict memo uses. The table is small (one row per source) and the
#: cookie freezes inside a read transaction, so a keyed snapshot is
#: byte-identical to re-querying: rows only change through commits,
#: which bump ``data_version`` and mint a fresh bucket. Writer
#: connections are excluded — their own uncommitted writes would be
#: invisible to a cookie-keyed snapshot. The strong conn reference
#: keeps a live ``id`` from being recycled while its entry exists.
_SS_SNAPS_MAX = 16
_SS_LOCK = threading.Lock()
_ss_snaps: "OrderedDict[tuple, Any]" = OrderedDict()
_ss_conns: "OrderedDict[int, sqlite3.Connection]" = OrderedDict()
_SS_ABSENT = object()  # table missing inside this snapshot
_SS_LIVE = object()    # non-reader conn → caller must compute live


def _source_state_snapshot(conn: sqlite3.Connection) -> Any:
    """``{source_id: (disposition, effective_at, valid_from, valid_to)}``.

    The full ``source_state`` row map for the caller's read snapshot —
    memoized on ``(id(conn), data_version)`` so the per-search
    admissibility scan becomes dict lookups after the first read.
    ``_SS_ABSENT`` when the table does not exist in this snapshot
    (unprovisioned stores), ``_SS_LIVE`` when the connection is not a
    query_only reader (uncommitted writes could hide behind the
    cookie — the caller must run the live path).
    """
    try:
        row = conn.execute("PRAGMA query_only").fetchone()
        if not row or not row[0]:
            return _SS_LIVE
        dv = conn.execute("PRAGMA data_version").fetchone()
        if not dv:
            return _SS_LIVE
        skey = (id(conn), dv[0])
    except sqlite3.Error:
        return _SS_LIVE
    with _SS_LOCK:
        hit = _ss_snaps.get(skey)
        if hit is not None:
            _ss_snaps.move_to_end(skey)
            return hit
    if not _has_table(conn, "source_state"):
        snap: Any = _SS_ABSENT
    else:
        snap = {}
        for r in conn.execute(
            "SELECT source_id, disposition, effective_at, valid_from,"
            " valid_to FROM source_state"
        ):
            snap[str(r[0])] = (str(r[1] or ""), r[2], r[3], r[4])
    with _SS_LOCK:
        _ss_conns[id(conn)] = conn
        _ss_conns.move_to_end(id(conn))
        while len(_ss_conns) > _SS_SNAPS_MAX:
            old_id, _ = _ss_conns.popitem(last=False)
            # Dropping the conn frees its id for reuse — every snapshot
            # keyed to it must go too.
            for k in [k for k in _ss_snaps if k[0] == old_id]:
                _ss_snaps.pop(k, None)
        _ss_snaps[skey] = snap
        _ss_snaps.move_to_end(skey)
        while len(_ss_snaps) > _SS_SNAPS_MAX:
            _ss_snaps.popitem(last=False)
    return snap


def _admissible_source_ids(
    conn: sqlite3.Connection, ids: Iterable[str],
) -> Optional[tuple]:
    """``({source_id: deliverable}, not_after_ts)`` (§14.3).

    ``None`` when the ``source_state`` table is absent (unprovisioned
    stores keep the pre-v5 behavior — no lifecycle evidence exists to
    apply). A candidate source with no row stays admissible: its
    projection row was only writable past the publication gate, and the
    absence of a control record is unresolved state, not an assertion of
    suppression (V5-14.16).

    ``not_after_ts`` is the earliest wall-clock moment at which any
    examined window verdict could flip WITHOUT a write — the smallest
    future ``valid_from``/``valid_to``/``effective_at`` bound among the
    rows read. ``None`` means no passive expiry exists; callers that
    memoize the result must not serve it past this instant.
    """
    snap = _source_state_snapshot(conn)
    if snap is _SS_ABSENT:
        return None
    if snap is _SS_LIVE and not _has_table(conn, "source_state"):
        return None
    wanted = sorted({str(i) for i in ids if i})
    if not wanted:
        return {}, None
    now_ts = datetime.now(timezone.utc).timestamp()
    out: dict = {}
    boundary: Optional[float] = None

    if snap is _SS_LIVE:
        for chunk in _cand._chunks(wanted, _IN_CHUNK):
            for row in conn.execute(
                "SELECT source_id, disposition, effective_at, valid_from,"
                " valid_to FROM source_state"
                f" WHERE source_id IN ({_ph(len(chunk))})",
                list(chunk),
            ):
                disp = str(row[1] or "")
                verdict, fut = _window_verdict(
                    disp, row[2], row[3], row[4], now_ts
                )
                out[str(row[0])] = verdict
                if (
                    fut is not None
                    and (boundary is None or fut < boundary)
                ):
                    boundary = fut
        return out, boundary

    # Snapshot-memoized path — identical verdicts, dict lookups only.
    # Per-row verdicts stay memoized on the row's cell tuple: the same
    # ``_current_window_ok`` result plus the earliest declared flip
    # instant (active windows flip at valid_from/valid_to; superseded at
    # effective_at), served only while that instant is still in the
    # future. Sources with no row stay admissible (absent from ``out``).
    for sid in wanted:
        row = snap.get(sid)
        if row is None:
            continue
        verdict, fut = _window_verdict(
            row[0], row[1], row[2], row[3], now_ts
        )
        out[sid] = verdict
        if fut is not None and (boundary is None or fut < boundary):
            boundary = fut
    return out, boundary


def _resolve_deadline(deadline: Any = None,
                      deadline_ms: Optional[float] = None
                      ) -> _cand.Deadline:
    """Normalize the deadline knob into the shared cooperative Deadline.

    Accepts an existing ``_cand.Deadline``, a millisecond budget
    (``deadline`` number or ``deadline_ms``), or ``None`` for unbounded
    internal calls.
    """
    if isinstance(deadline, _cand.Deadline):
        return deadline
    if isinstance(deadline, (int, float)) and not isinstance(deadline, bool):
        return _cand.Deadline(float(deadline))
    if deadline_ms is not None:
        return _cand.Deadline(float(deadline_ms))
    return _cand.Deadline(None)


def _snapshot_generation(conn: sqlite3.Connection, snapshot: Any) -> Optional[int]:
    """Pin the projection generation this read snapshot fences at.

    ``snapshot`` may be an int generation, a mapping/object carrying
    ``generation``/``projection_generation``, or ``None`` — in which case
    the committed ``projection_generation`` meta key *inside this
    connection's* snapshot is used (the same fence ``recall.py`` reads),
    so a rebuild committed after the snapshot opened stays invisible.
    Returns ``None`` only when nothing resolvable exists — callers then
    scan unfenced and report ``generation_unfenced``.
    """
    gen: Any = None
    if isinstance(snapshot, bool):
        gen = None
    elif isinstance(snapshot, int):
        gen = snapshot
    elif isinstance(snapshot, Mapping):
        gen = snapshot.get("generation", snapshot.get("projection_generation"))
    elif snapshot is not None:
        for attr in ("generation", "projection_generation"):
            v = getattr(snapshot, attr, None)
            if isinstance(v, int) and not isinstance(v, bool):
                gen = v
                break
    if gen is None and snapshot is None:
        try:
            row = conn.execute(
                "SELECT value_json FROM meta WHERE key = 'projection_generation'"
            ).fetchone()
            gen = safe_json_loads(row[0]) if row else None
        except sqlite3.Error:
            gen = None
    if isinstance(gen, bool) or gen is None:
        return None
    try:
        return int(gen)
    except (TypeError, ValueError):
        return None


def _normalize_eligible(eligible_ids: Optional[Iterable]) -> Optional[tuple]:
    """Normalize the caller's eligible universe E.

    Returns ``None`` when unrestricted-by-list (namespace narrowing
    applies instead), else ``(ids, bare, pairs)``: ``ids`` is every
    eligible source_id (the SQL restriction), ``bare`` the ids admitted
    at any revision, ``pairs`` the ``(source_id, revision)`` tuples that
    pin one revision. A pair entry admits exactly that revision; a bare
    entry admits every revision of that source.
    """
    if eligible_ids is None:
        return None
    bare: set = set()
    pairs: set = set()
    for entry in eligible_ids:
        if isinstance(entry, Mapping):
            sid = entry.get("source_id")
            rev = entry.get("revision")
            if sid is None:
                continue
            if rev is None:
                bare.add(str(sid))
            else:
                pairs.add((str(sid), int(rev)))
        elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
            try:
                pairs.add((str(entry[0]), int(entry[1])))
            except (TypeError, ValueError):
                bare.add(str(entry[0]))
        elif isinstance(entry, (str, bytes)):
            bare.add(str(entry))
        else:
            sid = getattr(entry, "source_id", None)
            if sid is None:
                continue
            rev = getattr(entry, "revision", None)
            if rev is None:
                bare.add(str(sid))
            else:
                pairs.add((str(sid), int(rev)))
    ids = bare | {sid for sid, _rev in pairs}
    return ids, bare, pairs


def _eligible_member(elig: tuple, source_id: str, revision: int) -> bool:
    """Pair-precise membership test applied after the SQL id filter."""
    _ids, bare, pairs = elig
    return source_id in bare or (source_id, revision) in pairs


# ---------------------------------------------------------------------------
# lexical candidates — exact eligible-corpus BM25 (V5-11)
# ---------------------------------------------------------------------------


def _projection_row_stream(
    conn: sqlite3.Connection,
    *,
    namespace: Optional[str],
    elig: Optional[tuple],
    generation: Optional[int],
    deadline: _cand.Deadline,
    stats: SourceLaneStats,
) -> Iterator[tuple]:
    """Yield ``(source_id, revision, doc_len, digest, tokens)`` over E,
    ordered.

    Deterministic ``(source_id, revision)`` order. Streams in bounded
    batches so a deadline can stop mid-corpus — a cut stream flips
    ``stats`` to ``partial`` and the caller's statistics are then marked
    incomplete rather than silently relabeled exact (V5-11.04/11.07).
    """
    gen_pred = _gen_pred() if generation is not None else ""
    gen_params: list = [] if generation is None else [generation]

    def _emit(cur) -> Iterator[tuple]:
        while True:
            if deadline.expired():
                stats.status = "partial"
                stats.reason = stats.reason or "deadline"
                stats.deadline_exceeded = True
                if "scan_incomplete:deadline" not in stats.warnings:
                    stats.warnings.append("scan_incomplete:deadline")
                return
            batch = cur.fetchmany(512)
            if not batch:
                return
            for row in batch:
                stats.candidates_examined += 1
                if stats.candidates_examined > _SCAN_CAP:
                    stats.truncated = True
                    stats.warnings.append("eligible_scan_capped")
                    return
                yield row

    if elig is not None:
        ids = sorted(elig[0])
        # ``eligible_ids`` is the request-eligible universe E; a supplied
        # namespace additionally constrains the physical partition
        # (V5-10.03) — the intersection can only narrow, never widen.
        ns_pred = " AND scope_id = ?" if namespace is not None else ""
        ns_params: list = [] if namespace is None else [namespace]
        for chunk in _cand._chunks(ids, _IN_CHUNK):
            sql = (
                "SELECT source_id, revision, doc_len, digest, tokens"
                " FROM source_lexical_projection"
                f" WHERE source_id IN ({_ph(len(chunk))})"
                f"{ns_pred}{gen_pred}"
                " ORDER BY source_id, revision"
            )
            cur = conn.execute(sql, [*chunk, *ns_params, *gen_params])
            for row in _emit(cur):
                yield row
    else:
        sql = (
            "SELECT source_id, revision, doc_len, digest, tokens"
            " FROM source_lexical_projection"
            f" WHERE scope_id = ?{gen_pred}"
            " ORDER BY source_id, revision"
        )
        cur = conn.execute(sql, [namespace, *gen_params])
        for row in _emit(cur):
            yield row


def _fold_query_terms(query_terms: Sequence[str]) -> list:
    """Fold caller-supplied query terms into the projection's token space.

    ``source_lexical_projection.tokens`` are ``norm/v1`` output (NFKC →
    NFKD with combining marks dropped → casefold → punctuation/symbol/
    separator folded to spaces — V5-30.06). Query analysis emits raw
    regex tokens, so an accented/composed form ("café" NFC) would miss a
    stored decomposed twin ("café" NFD → "cafe") byte-exactly. The lane
    meets the pinned space itself — one normalization identity on both
    sides of the match, applied to the query terms only; stored bytes
    are never rewritten. ``entity_postings`` matching stays byte-exact
    (V5-30.17) and is deliberately not folded here.
    """
    out: list = []
    for term in query_terms:
        folded = normalize_text(term or "", version=NORMALIZATION_VERSION)
        out.extend(t for t in folded.split() if t)
    return sorted(set(out))


def _source_fts_ready(conn: sqlite3.Connection) -> bool:
    """``source_fts_idx`` + its insert trigger both exist.

    The trigger probe matters as much as the virtual table: an index
    whose mirroring trigger was never applied would silently under-
    nominate, which the token verification below cannot detect. Both
    objects are created/dropped together by the v5 migration.
    """
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master"
            " WHERE name IN ('source_fts_idx', 'source_fts_ai')"
        ).fetchone()
    except sqlite3.Error:
        return False
    return bool(row and row[0] == 2)


def _fts_term_safe(term: str) -> bool:
    """``term`` can be FTS-nominated without missing a real match.

    unicode61 emits maximal runs of letter/number codepoints as index
    tokens. ``tokens`` is the normalized token stream — whitespace-
    separated — so an all-alphanumeric term hits exactly the documents
    whose ``tokens.split()`` contains it. A term carrying any mark/
    punctuation/separator codepoint could be split differently by the
    index than by ``split()``; those terms stay on the token-scan path
    (over-nomination is safe — real ``tokens`` still verify — but an
    unsafe term could under-nominate, which is not).
    """
    return bool(term) and term.isalnum()


def _lexical_stats_fts(
    conn: sqlite3.Connection,
    *,
    terms: list,
    namespace: str,
    generation: Optional[int],
    deadline: _cand.Deadline,
    stats: SourceLaneStats,
    store: Any = None,
) -> Optional[tuple]:
    """``(n_docs, total_len, df, cand)`` — the eligible-corpus statistics
    the token-stream scan computes, sourced through the FTS5 shadow for
    the namespace-scoped call shape (the hot path).

    ``COUNT(*)``/``SUM(doc_len)`` aggregates give N and avgdl over E in
    one indexed pass instead of streaming every row's ``tokens``;
    per-term ``MATCH`` postings through ``source_fts_idx`` nominate the
    candidate set; the persisted ``tokens`` of nominated members still
    decide every df/tf — FTS is a nomination filter, never the match
    authority, so results are byte-identical whenever nomination cannot
    under-count (the caller gates on ``_fts_term_safe`` for every term).

    Returns ``None`` — with ``stats`` untouched — whenever the fast path
    cannot prove equivalence and the caller must run the token scan:

    * the shadow is incomplete: the trigger-maintained carrier must hold
      exactly the projection's ``(source_id, revision)`` set under the
      same scope/generation predicate (``_write_fts_pair`` writes them
      in one transaction). Equal counts plus zero carrier rows without
      a matching projection row is pair-set equality — a missing or
      extra carrier anywhere fails over;
    * the corpus exceeds ``_SCAN_CAP``: the stream's examined-prefix
      truncation semantics would be approximated, not reproduced.
    """
    gen_pred = _gen_pred() if generation is not None else ""
    gen_params: list = [] if generation is None else [generation]
    r_gen_pred = _gen_pred("r") if generation is not None else ""
    r_gen_params: list = [] if generation is None else [generation]

    if deadline.expired():
        # Same verdict the stream produces when the budget is already
        # gone: nothing examined, partial, honest.
        stats.status = "partial"
        stats.reason = stats.reason or "deadline"
        stats.deadline_exceeded = True
        if "scan_incomplete:deadline" not in stats.warnings:
            stats.warnings.append("scan_incomplete:deadline")
        return 0, 0, Counter(), {}

    # Whole-result memo — everything below is a pure function of the
    # committed data under this predicate. Versioned by the lexical
    # subset fingerprint only: the result reads the projection carrier
    # and its FTS shadow, so writes to source_state / vector / postings
    # tables cannot change it and must not invalidate it.
    version = _snapshot_version(conn, store, tables=_LEX_FP_TABLES)
    res_memo = _lexres_memo(store) if version is not None else None
    res_key = (tuple(terms), namespace, generation)
    if res_memo is not None:
        ent = res_memo.get(res_key)
        if ent is not None and ent[0] == version:
            res_memo.move_to_end(res_key)
            r_n, r_tl, r_df, r_cand = ent[1]
            stats.candidates_examined += int(r_n)
            return int(r_n), int(r_tl), Counter(r_df), dict(r_cand)

    # ---- corpus aggregates + shadow-completeness gate -----------------
    n_docs, total_len = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(doc_len), 0)"
        " FROM source_lexical_projection"
        f" WHERE scope_id = ?{gen_pred}",
        [namespace, *gen_params],
    ).fetchone()
    if int(n_docs) > _SCAN_CAP:
        return None
    carrier_count = conn.execute(
        "SELECT COUNT(*) FROM source_fts_rows r"
        f" WHERE r.scope_id = ?{r_gen_pred}",
        [namespace, *r_gen_params],
    ).fetchone()[0]
    if int(n_docs) != int(carrier_count):
        return None
    orphans = conn.execute(
        "SELECT COUNT(*) FROM source_fts_rows r"
        f" WHERE r.scope_id = ?{r_gen_pred}"
        " AND NOT EXISTS ("
        "   SELECT 1 FROM source_lexical_projection p"
        "   WHERE p.source_id = r.source_id"
        "     AND p.revision = r.revision)",
        [namespace, *r_gen_params],
    ).fetchone()[0]
    if orphans:
        return None
    # The namespace-scoped scan counts every fetched row — there is no
    # post-fetch member filter on this call shape — so examined == n_docs.
    stats.candidates_examined += int(n_docs)

    # ---- per-term postings through the FTS shadow --------------------
    # Two phases — the same shape retrieval.candidates._fts_match uses:
    # the bare MATCH emits rowids in index order, then chunked INTEGER-
    # PRIMARY-KEY probes on the carrier apply the scope/generation
    # fence. A flat JOIN lets the planner drive from the carrier and
    # re-evaluate MATCH per row — O(corpus × probe).
    #
    # The nominated set is a pure function of the committed index under
    # this predicate, so it is memoized against the corpus fingerprint —
    # see ``_snapshot_version``. A miss simply re-runs the postings probe.
    nom_memo = _nomination_memo(store) if version is not None else None
    nominated: set = set()
    for t in terms:
        if deadline.expired():
            stats.status = "partial"
            stats.reason = stats.reason or "deadline"
            stats.deadline_exceeded = True
            if "scan_incomplete:deadline" not in stats.warnings:
                stats.warnings.append("scan_incomplete:deadline")
            break
        mkey = (str(t), namespace, generation)
        ent = nom_memo.get(mkey) if nom_memo is not None else None
        if ent is not None and ent[0] == version:
            nom_memo.move_to_end(mkey)
            nominated.update(ent[1])
            continue
        rowids = [
            r[0] for r in conn.execute(
                "SELECT rowid FROM source_fts_idx"
                " WHERE source_fts_idx MATCH ?",
                ('"' + str(t).replace('"', '""') + '"',),
            ).fetchall()
        ]
        tset: set = set()
        for chunk in _cand._chunks(rowids, _IN_CHUNK):
            tset.update(
                (str(sid), int(rev))
                for sid, rev in conn.execute(
                    "SELECT r.source_id, r.revision"
                    " FROM source_fts_rows r"
                    f" WHERE r.row_id IN ({_ph(len(chunk))})"
                    f" AND r.scope_id = ?{r_gen_pred}",
                    [*chunk, namespace, *r_gen_params],
                ).fetchall()
            )
        nominated.update(tset)
        if nom_memo is not None:
            nom_memo[mkey] = (version, frozenset(tset))
            while len(nom_memo) > _NOM_MEMO_MAX:
                nom_memo.popitem(last=False)

    # ---- persisted tokens for nominated members only -----------------
    # df and tf are computed from the stored bytes exactly like the scan
    # path — the index only decides which rows get fetched. Token bytes
    # themselves are content-addressed by the row's ``digest``: a warm
    # memo entry replays the identical ``split()``+``Counter`` without
    # re-reading the payload column (fetched lazily for misses only).
    df: Counter = Counter()
    cand: dict = {}
    keys = sorted(nominated)
    memo = _lexical_memo(store)

    def _account(key: tuple, dl_val: int, ctr: Counter) -> None:
        # Counter membership is set-membership (nonzero counts only) —
        # the same ``t in tokset`` predicate the token scan applies, so
        # a stale-FTS nomination whose persisted tokens lack the term is
        # still filtered authoritatively (never indexed, never scored).
        matched = [t for t in terms if t in ctr]
        for t in matched:
            df[t] += 1
        if matched:
            cand[key] = (dl_val, ctr)

    for i in range(0, len(keys), _IN_CHUNK):
        if deadline.expired():
            stats.status = "partial"
            stats.reason = stats.reason or "deadline"
            stats.deadline_exceeded = True
            if "scan_incomplete:deadline" not in stats.warnings:
                stats.warnings.append("scan_incomplete:deadline")
            break
        part = keys[i:i + _IN_CHUNK]
        marks = ",".join("(?,?)" for _ in part)
        params = [v for pair in part for v in pair]
        pending: dict = {}  # key -> (doc_len, digest) needing tokens
        for sid, rev, doc_len, digest in conn.execute(
            "SELECT source_id, revision, doc_len, digest"
            " FROM source_lexical_projection"
            f" WHERE (source_id, revision) IN ({marks})"
            f" AND scope_id = ?{gen_pred}",
            [*params, namespace, *gen_params],
        ).fetchall():
            key = (str(sid), int(rev))
            dkey = str(digest) if digest else None
            ctr = memo.get(dkey) if (memo is not None and dkey) else None
            if ctr is not None:
                memo.move_to_end(dkey)
                _account(key, int(doc_len or 0), ctr)
            else:
                pending[key] = (int(doc_len or 0), digest)
        if not pending:
            continue
        pkeys = sorted(pending)
        pmarks = ",".join("(?,?)" for _ in pkeys)
        pparams = [v for pair in pkeys for v in pair]
        for sid, rev, tokens in conn.execute(
            "SELECT source_id, revision, tokens"
            " FROM source_lexical_projection"
            f" WHERE (source_id, revision) IN ({pmarks})"
            f" AND scope_id = ?{gen_pred}",
            [*pparams, namespace, *gen_params],
        ).fetchall():
            key = (str(sid), int(rev))
            info = pending.pop(key, None)
            if info is None:
                continue
            ctr = Counter((tokens or "").split())
            _lex_memo_put(memo, info[1], ctr)
            _account(key, info[0], ctr)
    if res_memo is not None and stats.status != "partial":
        # Store only complete results — a deadline-cut verdict encodes
        # the caller's budget, not just the data. Copies so later callers
        # can never mutate the memoized containers.
        res_memo[res_key] = (
            version,
            (int(n_docs), int(total_len or 0), Counter(df), dict(cand)),
        )
        while len(res_memo) > _RES_MEMO_MAX:
            res_memo.popitem(last=False)
    return int(n_docs), int(total_len or 0), df, cand


#: Whole-scan top-k memo for ``lexical_candidates`` — same contract as
#: the vector lane's: ``(terms, scan identity, content fingerprint) ->
#: frozen result``, replays the byte-identical BM25 outcome the fresh
#: scan would compute. The fingerprint folds ``(rowid, digest)`` over
#: the projection rows the scan's WHERE admits — ``rowid`` catches the
#: INSERT OR REPLACE rewrites the projection table receives,
#: ``digest`` content-addresses the persisted normalized tokens.
#: ``eligible_ids`` scans stay uncached (caller state); deadline-cut
#: partials are never memoized.
_LTOPK_MEMO_ATTR = "_lexical_topk_memo_v1"
_LTOPK_MEMO_MAX = 256


def _ltopk_memo(store: Any) -> Optional[OrderedDict]:
    if store is None:
        return None
    memo = getattr(store, _LTOPK_MEMO_ATTR, None)
    if memo is None:
        try:
            memo = OrderedDict()
            setattr(store, _LTOPK_MEMO_ATTR, memo)
        except Exception:
            return None
    return memo


def _lexical_scan_fp(
    conn: sqlite3.Connection,
    *,
    namespace: Optional[str],
    generation: Optional[int],
    elig: Optional[tuple] = None,
) -> Optional[str]:
    """Content fingerprint over exactly the projection rows the scan's
    WHERE admits — PK-ordered ``rowid|digest`` fold."""
    sql = (
        "SELECT group_concat(q, '') FROM ("
        " SELECT quote(rowid)||'|'||quote(digest) AS q"
        " FROM source_lexical_projection"
    )
    params: list = []
    conj = " WHERE "
    if elig is not None:
        ids = sorted(elig[0])
        sql += f"{conj}source_id IN ({_ph(len(ids))})"
        params.extend(ids)
        conj = " AND "
    if namespace is not None:
        sql += f"{conj}scope_id = ?"
        params.append(namespace)
        conj = " AND "
    if generation is not None:
        sql += _gen_pred()
        params.append(generation)
    sql += " ORDER BY source_id, revision)"
    try:
        row = conn.execute(sql, params).fetchone()
    except sqlite3.Error:
        return None
    return row[0] if row else None


def lexical_candidates(
    conn: sqlite3.Connection,
    *,
    query_terms: Sequence[str],
    eligible_ids: Optional[Iterable] = None,
    namespace: Optional[str] = None,
    generation: Optional[int] = None,
    limit: int = SOURCE_LEXICAL_LIMIT,
    deadline: Any = None,
    deadline_ms: Optional[float] = None,
    store: Any = None,
) -> tuple:
    """Exact BM25 over ``source_lexical_projection`` — stats over E only.

    The full eligible set is materialized once per call (V5-11.05's
    preferred sequence): N, per-term df, and avgdl are measured over
    exactly the rows the request may surface — never over the global
    table and never over the candidate subset (V5-11.03/11.04, F4-11).

    ``query_terms`` are the tokens emitted by query analysis; the lane
    folds them through ``norm/v1`` so they meet the projection's
    normalized token space byte-exactly regardless of caller-side
    casing/Unicode form. Returns ``([SourceHit], SourceLaneStats)``
    ordered by score desc, then ``(source_id, revision)`` — the lane's
    deterministic order.
    """
    dl = _resolve_deadline(deadline, deadline_ms)
    stats = SourceLaneStats("lexical")
    if not _has_table(conn, "source_lexical_projection"):
        stats.status = "unavailable"
        stats.reason = "no_source_lexical_projection"
        return [], stats
    terms = _fold_query_terms(query_terms)
    if not terms:
        stats.reason = "empty_query_terms"
        return [], stats

    elig = _normalize_eligible(eligible_ids)
    if elig is not None and not elig[0]:
        stats.reason = "empty_eligible_set"
        return [], stats
    if elig is None and namespace is None:
        # Neither an eligibility list nor a namespace partition: refuse a
        # store-wide corpus — N/df/avgdl would be global statistics
        # (V5-11.03/10.03).
        stats.status = "unavailable"
        stats.reason = "unscoped_request"
        return [], stats

    # Whole-scan top-k memo: identical terms over a
    # fingerprint-identical corpus replay the frozen BM25 outcome (the
    # lane is a pure function of those inputs; elig scans stay
    # uncached — the eligibility set is caller state).
    tk_memo = _ltopk_memo(store) if elig is None else None
    tk_key = None
    if tk_memo is not None:
        tk_fp = _lexical_scan_fp(
            conn, namespace=namespace, generation=generation)
        if tk_fp is not None:
            tk_key = (tuple(terms), namespace, generation,
                      int(limit), tk_fp)
            ent = tk_memo.get(tk_key)
            if ent is not None:
                tk_memo.move_to_end(tk_key)
                return _replay_lane_result(
                    ent, signal=SIGNAL_LEXICAL, lane="lexical")

    # ---- one materialization of the eligible corpus E -----------------
    n_docs = 0
    total_len = 0
    df: Counter = Counter()
    cand: dict = {}  # (source_id, revision) -> (doc_len, Counter)
    fast_stats = None
    if (
        elig is None
        and namespace is not None
        and _source_fts_ready(conn)
        and all(_fts_term_safe(t) for t in terms)
    ):
        # FTS-shadow fast path: the index nominates candidate docs; the
        # persisted ``tokens`` bytes still decide every match — stats are
        # byte-identical to the full token scan (see _lexical_stats_fts).
        # ``None`` = the shadow is provably incomplete under this
        # predicate; fall through to the scan untouched. Explicit
        # ``eligible_ids`` callers keep the pair-precise stream — that
        # shape is already selective and never widened by the index.
        fast_stats = _lexical_stats_fts(
            conn, terms=terms, namespace=namespace,
            generation=generation, deadline=dl, stats=stats,
            store=store,
        )
    if fast_stats is not None:
        n_docs, total_len, df, cand = fast_stats
    else:
        lex_memo = _lexical_memo(store)
        for sid, rev, doc_len, digest, tokens in _projection_row_stream(
            conn, namespace=namespace, elig=elig,
            generation=generation, deadline=dl, stats=stats,
        ):
            if elig is not None and not _eligible_member(elig, sid, int(rev)):
                continue
            n_docs += 1
            dl_val = int(doc_len or 0)
            total_len += dl_val
            dkey = str(digest) if digest else None
            counts = (
                lex_memo.get(dkey)
                if (lex_memo is not None and dkey)
                else None
            )
            if counts is None:
                counts = Counter((tokens or "").split())
                _lex_memo_put(lex_memo, digest, counts)
            elif lex_memo is not None:
                lex_memo.move_to_end(dkey)
            # Counter membership is set-membership — the same ``t in
            # tokset`` predicate; byte-identical df/tf.
            matched = [t for t in terms if t in counts]
            for t in matched:
                df[t] += 1
            if matched:
                cand[(sid, int(rev))] = (dl_val, counts)

    stats.eligible = n_docs
    if stats.status == "partial":
        # Deadline cut inside the corpus stream: N/df/avgdl cover only the
        # examined prefix — an honest partial result, never relabeled.
        stats.details["stats_complete"] = False
    else:
        stats.details["stats_complete"] = True
    if not cand:
        return [], stats

    # ---- BM25 over the eligible candidate set, stats from E -----------
    avgdl = (total_len / n_docs) if n_docs else 0.0
    idf = {
        t: math.log(1.0 + (n_docs - df[t] + 0.5) / (df[t] + 0.5))
        for t in terms
        if df[t]
    }
    scored: list = []
    for key in sorted(cand):
        dl_val, counts = cand[key]
        len_adj = (
            1.0 - _BM25_B + _BM25_B * (dl_val / avgdl)
            if avgdl > 0 else 1.0
        )
        denom_base = _BM25_K1 * len_adj
        score = math.fsum(
            idf[t] * (tf * (_BM25_K1 + 1.0)) / (tf + denom_base)
            for t in terms
            for tf in (counts.get(t, 0),)
            if tf and t in idf
        )
        scored.append((key, score))
    # Lane order: score desc, source_id asc, revision desc — the same
    # declared tie-breaks fusion applies (ranking/v1).
    scored.sort(key=lambda kv: (-kv[1], kv[0][0], -kv[0][1]))

    stats.scored = len(scored)
    stats.truncated = stats.truncated or len(scored) > limit
    hits: list = []
    for rank, ((sid, rev), score) in enumerate(scored[:limit], start=1):
        hit = SourceHit(sid, rev)
        hit.signals[SIGNAL_LEXICAL] = score
        hit.lanes["lexical"] = rank
        hits.append(hit)
    stats.returned = len(hits)
    stats.details.update(n_docs=n_docs, avgdl=avgdl, terms=len(terms))
    if (
        tk_memo is not None
        and tk_key is not None
        and stats.status != "partial"
        and not stats.deadline_exceeded
    ):
        # Completed scans only — a deadline-cut partial is per-call
        # state and must never replay as the complete answer.
        tk_memo[tk_key] = _freeze_lane_result(
            hits, stats, signal=SIGNAL_LEXICAL, lane="lexical")
        while len(tk_memo) > _LTOPK_MEMO_MAX:
            tk_memo.popitem(last=False)
    return hits, stats


# ---------------------------------------------------------------------------
# vector candidates — blocked exact cosine over source_vectors (V5-12.01)
# ---------------------------------------------------------------------------


def _query_blob(query_vector: Any) -> Optional[bytes]:
    """Normalize a query vector to its packed float32le blob."""
    if query_vector is None:
        return None
    if isinstance(query_vector, (bytes, bytearray, memoryview)):
        return bytes(query_vector)
    if isinstance(query_vector, Mapping):
        blob = query_vector.get("vector", query_vector.get("blob"))
        return _query_blob(blob)
    try:
        return Float32Codec.pack(list(query_vector))
    except (VerbatimError, TypeError, ValueError):
        return None


def _pinned_encoder(conn: sqlite3.Connection, encoder: Optional[str],
                    namespace: Optional[str]) -> Optional[str]:
    """Resolve the pinned encoder identity for the scan (V5-12.01).

    An explicit ``encoder`` always wins; otherwise the table must hold
    exactly one encoder identity inside the namespace — mixing embedding
    spaces is meaningless cosine, so ambiguity resolves to ``None`` and
    the lane reports unavailable rather than picking a space.
    """
    if isinstance(encoder, str) and encoder:
        return encoder
    if namespace is None:
        rows = conn.execute(
            "SELECT DISTINCT encoder FROM source_vectors"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT DISTINCT encoder FROM source_vectors WHERE namespace = ?",
            (namespace,),
        ).fetchall()
    ids = sorted({r[0] for r in rows})
    return ids[0] if len(ids) == 1 else None


def _cosine(query: tuple, q_norm: float, vec: tuple) -> Optional[float]:
    """Pure-Python float64 cosine — the vectors.py correctness reference.

    Kept numpy-free on purpose: identical inputs must produce identical
    scores on every host, so this lane always runs the reference math
    rather than a platform-dependent BLAS path (V5-31.03 determinism).
    """
    v_norm_sq = math.fsum(x * x for x in vec)
    if v_norm_sq <= 0.0 or not math.isfinite(v_norm_sq):
        return None
    dot = math.fsum(a * b for a, b in zip(query, vec))
    return dot / (q_norm * math.sqrt(v_norm_sq))


#: Per-store memo of decoded vector rows — ``(encoder_keyed_digest) ->
#: (array('d') vec, v_norm_sq)``. ``source_vectors`` rows are
#: content-immutable and the row's own ``digest`` column is the producer's
#: content address (``_vector_digest`` over encoder+blob), so the memo
#: only ever elides a byte-identical recompute — the same unpack, the
#: same finite check, the same float64 norm — while a rewritten row
#: mints a new key and re-decodes. Determinism is untouched: the stored
#: values are exactly what the reference path computes.
_VEC_MEMO_ATTR = "_v5_source_vec_memo_v1"
_VEC_MEMO_MAX = 4096


def _vector_memo(store: Any) -> Optional[OrderedDict]:
    if store is None:
        return None
    memo = getattr(store, _VEC_MEMO_ATTR, None)
    if memo is None:
        try:
            memo = OrderedDict()
            setattr(store, _VEC_MEMO_ATTR, memo)
        except Exception:
            return None
    return memo


#: Per-store memo of token multisets — ``digest -> Counter(tokens)``.
#: ``source_lexical_projection.digest`` is the row's persisted content
#: address, so a hit replays a byte-identical ``split()``+``Counter``;
#: a rewritten projection mints a new digest and re-derives. Counter
#: membership is the same predicate the scan applies to ``set(toks)``
#: (nonzero-count keys only), so df/tf/candidates are byte-identical.
_LEX_MEMO_ATTR = "_v5_source_lex_memo_v1"
_LEX_MEMO_MAX = 8192


def _lexical_memo(store: Any) -> Optional[OrderedDict]:
    if store is None:
        return None
    memo = getattr(store, _LEX_MEMO_ATTR, None)
    if memo is None:
        try:
            memo = OrderedDict()
            setattr(store, _LEX_MEMO_ATTR, memo)
        except Exception:
            return None
    return memo


def _lex_memo_put(
    memo: Optional[OrderedDict], digest: Any, counter: Counter
) -> None:
    if memo is None or not digest:
        return
    key = str(digest)
    memo[key] = counter
    while len(memo) > _LEX_MEMO_MAX:
        memo.popitem(last=False)


#: Per-store memo of FTS nomination sets — ``(term, scope, gen) ->
#: (corpus_fingerprint, frozenset[(source_id, revision)])``. The
#: fingerprint (``_snapshot_version``) covers the carrier, content, and
#: projection tables the nomination derives from — any write to them
#: invalidates; journal-table writes do not. Advisory only: a miss
#: re-runs MATCH+probe and persisted tokens remain the match authority.
_NOM_MEMO_ATTR = "_v5_source_nom_memo_v1"
_NOM_MEMO_MAX = 512


def _nomination_memo(store: Any) -> Optional[OrderedDict]:
    if store is None:
        return None
    memo = getattr(store, _NOM_MEMO_ATTR, None)
    if memo is None:
        try:
            memo = OrderedDict()
            setattr(store, _NOM_MEMO_ATTR, memo)
        except Exception:
            return None
    return memo


#: Tables the source lane reads — every dependency of the merged
#: candidate result. ``(table, pk_col, identity_cols, agg_sql)``:
#: COUNT+MAX(pk) catches every append/delete; the max-row identity
#: catches delete+reinsert reusing the max rowid (the reinserted row
#: carries different content). Journal tables (routing_decisions,
#: influence, readiness_obligations, jobs, …) are deliberately absent:
#: a per-search journal commit must not invalidate content memos.
#:
#: ``source_state`` is the exception — it is written by GUARDED
#: in-place CAS UPDATE (``sourcestate`` transitions; ``controls`` erase
#: fallback), which count/max cannot see on non-max rows. Every
#: artifact-path write bumps ``control_version`` AND appends an
#: immutable ``object_revisions`` doc, so the custom aggregate adds
#: ``SUM(control_version)`` + the ``source_state`` doc count + a packed
#: window-field length sum covering the columns the lane's currency
#: test reads.
_FP_TABLES: tuple = (
    ("source_lexical_projection", "rowid",
     "source_id, revision, scope_id, generation, digest", None),
    ("source_fts_rows", "row_id",
     "source_id, revision, scope_id, generation", None),
    ("source_fts", "fts_row_id", "text", None),
    ("source_vectors", "rowid",
     "source_id, revision, encoder, namespace, generation, digest", None),
    ("entity_postings", "rowid",
     "namespace, entity, entity_kind, source_id, revision, generation",
     None),
    ("source_state", "rowid",
     "source_id, namespace, control_version, mutation_head, disposition,"
     " superseded_by, effective_at, known_at, valid_from, valid_to,"
     " updated_at, producer",
     "SELECT COUNT(*), COALESCE(MAX(rowid),0),"
     " COALESCE(SUM(control_version),0),"
     " COALESCE(SUM(LENGTH(source_id||'|'||disposition||'|'"
     "   ||COALESCE(superseded_by,'')||'|'||COALESCE(effective_at,'')"
     "   ||'|'||COALESCE(valid_from,'')||'|'||COALESCE(valid_to,''))),0)"
     " FROM source_state"),
)

#: Append-only doc ledger every artifact-path ``source_state`` write
#: mints — probed separately so an absent ``object_revisions`` table
#: (pre-v4 store) degrades to a marker rather than blinding the whole
#: ``source_state`` signal.
_FP_STATE_DOCS_SQL = (
    "SELECT COUNT(*) FROM object_revisions WHERE kind='source_state'"
)

#: The tables ``_lexical_stats_fts``'s memoized results actually read —
#: the projection carrier + its FTS shadow pair. A subset fingerprint
#: keeps whole-result and nomination memos warm across writes to the
#: OTHER lane tables (source_state mutations, vector/entity rows) that
#: cannot alter ``(n_docs, total_len, df, cand)``.
_LEX_FP_TABLES: tuple = _FP_TABLES[:3]


def _compute_fp(
    conn: sqlite3.Connection, tables: tuple = _FP_TABLES
) -> Optional[tuple]:
    parts: list = []
    try:
        for table, pk, cols, agg_sql in tables:
            try:
                if agg_sql is not None:
                    agg = conn.execute(agg_sql).fetchone()
                else:
                    agg = conn.execute(
                        f'SELECT COUNT(*), COALESCE(MAX({pk}), 0)'
                        f' FROM "{table}"'
                    ).fetchone()
                cnt, mx = int(agg[0]), int(agg[1] or 0)
            except sqlite3.Error:
                # An absent/unprobeable table is itself stable snapshot
                # state — record a marker; table creation moves
                # data_version and re-fingerprints anyway.
                parts.append(("absent",))
                continue
            try:
                ident = (
                    conn.execute(
                        f'SELECT {cols} FROM "{table}" WHERE {pk} = ?',
                        (mx,),
                    ).fetchone()
                    if mx else None
                )
            except sqlite3.Error:
                ident = ("unprobeable",)
            extra: tuple = tuple(agg[2:])
            if table == "source_state":
                try:
                    docs = conn.execute(_FP_STATE_DOCS_SQL).fetchone()
                    extra += (int(docs[0]),)
                except sqlite3.Error:
                    extra += ("no_ledger",)
            parts.append((cnt, mx, extra, ident))
    except sqlite3.Error:
        return None
    return tuple(parts)


_FP_CACHE_ATTR = "_v5_source_fp_cache_v1"
_FP_CACHE_MAX = 16


def _snapshot_version(
    conn: sqlite3.Connection, store: Any, tables: tuple = _FP_TABLES
) -> Optional[tuple]:
    """Fingerprint of the source-lane corpus tables as this snapshot
    sees them, or ``None`` when unversionable.

    Per table: ``(COUNT(*), MAX(rowid), extra-aggregates,
    identity-of-the-max-row)`` — see ``_FP_TABLES``. The fingerprint is
    itself memoized per ``(connection, PRAGMA data_version,
    store._write_epoch, tables)``: data_version is frozen inside an open
    snapshot and bumps on every commit by any other connection, while
    ``_write_epoch`` covers commits made through this Store object on
    the committing connection itself (which never observes its own
    data_version move). Journal-table commits recompute the fingerprint
    — cheap — but leave it unchanged, so content memos stay warm across
    the recall journal write.

    ``tables`` scopes the fingerprint to a subset — a memo whose result
    only reads the lexical shadow stays valid across source_state /
    vector writes that a full-lane fingerprint would treat as churn.
    """
    try:
        row = conn.execute("PRAGMA data_version").fetchone()
        dver = int(row[0]) if row else None
    except (sqlite3.Error, TypeError, ValueError):
        return None
    if dver is None or store is None:
        return _compute_fp(conn, tables)
    cache = getattr(store, _FP_CACHE_ATTR, None)
    if cache is None:
        try:
            cache = OrderedDict()
            setattr(store, _FP_CACHE_ATTR, cache)
        except Exception:
            return _compute_fp(conn, tables)
    # ``_write_epoch`` covers commits issued through this Store object on
    # ANY of its connections (data_version does not move for the
    # committing connection itself); the conn object in the key scopes
    # the entry to one connection's view; ``tables`` scopes it to the
    # fingerprinted table set.
    key = (conn, dver, getattr(store, "_write_epoch", 0), tables)
    ent = cache.get(key)
    if ent is not None:
        cache.move_to_end(key)
        return ent
    fp = _compute_fp(conn, tables)
    if fp is not None:
        cache[key] = fp
        while len(cache) > _FP_CACHE_MAX:
            cache.popitem(last=False)
    return fp


#: Per-store memo of whole lexical statistics results —
#: ``(terms, scope, gen) -> (version, (n_docs, total_len, df, cand))``.
#: Only COMPLETE results are stored (a deadline-cut partial is a
#: function of the budget, not just the data). Containers are copied on
#: read so callers can never mutate a memoized structure.
_RES_MEMO_ATTR = "_v5_source_lexres_memo_v1"
_RES_MEMO_MAX = 256


def _lexres_memo(store: Any) -> Optional[OrderedDict]:
    if store is None:
        return None
    memo = getattr(store, _RES_MEMO_ATTR, None)
    if memo is None:
        try:
            memo = OrderedDict()
            setattr(store, _RES_MEMO_ATTR, memo)
        except Exception:
            return None
    return memo


#: Per-store memo of whole ``source_candidates`` results —
#: ``key -> (content_fingerprint,
#:           [(source_id, revision, signals, lanes), ...], stats_dict)``.
#: Stored only for complete results (``status != "partial"`` — a
#: deadline-cut result is a function of the budget, not just the data).
#:
#: The fingerprint covers ONLY the content tables the sub-lanes read —
#: ``source_state`` is deliberately excluded: the merged hit list is a
#: pure function of the five projection/index tables, while lifecycle
#: admission is re-evaluated against LIVE state on every memo serve
#: (``_admissible_source_ids`` — the same check the fresh path runs,
#: including its passive-expiry windows). A new add commits
#: ``source_state``/``source_revisions``/``sources`` rows only, so the
#: memo now stays valid through the whole pending-projection window;
#: the arriving source is invisible to the lanes until its own
#: projection rows commit — which is exactly what the fingerprint then
#: catches.
_SC_MEMO_ATTR = "_v5_source_cands_memo_v1"
_SC_MEMO_MAX = 128
#: Merged-hit ceiling for a memoizable result — an oversized merge is
#: left out rather than stored truncated (a tail beyond the stored
#: window could be promoted by a lifecycle flip, so a partial store
#: would not be answer-equivalent).
_SC_MEMO_HITCAP = 2048
#: Content-table fingerprint scope — ``_FP_TABLES`` minus the lifecycle
#: table, which the serve path re-checks live instead.
_SC_FP_TABLES: tuple = _FP_TABLES[:5]


def _sc_memo(store: Any) -> Optional[OrderedDict]:
    if store is None:
        return None
    memo = getattr(store, _SC_MEMO_ATTR, None)
    if memo is None:
        try:
            memo = OrderedDict()
            setattr(store, _SC_MEMO_ATTR, memo)
        except Exception:
            return None
    return memo


#: Per-store memo of *finished* similarity scores —
#: ``(query-vector digest, row content key) -> cosine`` — a pure
#: function of the two content identities, so a hit replays the
#: identical correctly-rounded value without the sparse product at all.
#: Covers repeat queries across corpus churn: the index rescan still
#: runs every call (eligibility + malformed verdicts are never memoized
#: here), but scored rows skip the dot entirely.
_VSCORE_MEMO_ATTR = "_v5_source_vscore_memo_v1"
_VSCORE_MEMO_MAX = 16384


def _vscore_memo(store: Any) -> Optional[OrderedDict]:
    if store is None:
        return None
    memo = getattr(store, _VSCORE_MEMO_ATTR, None)
    if memo is None:
        try:
            memo = OrderedDict()
            setattr(store, _VSCORE_MEMO_ATTR, memo)
        except Exception:
            return None
    return memo


def _decode_vector(
    memo: Optional[OrderedDict],
    key: bytes,
    vec_blob: Any,
    dims: int,
) -> Optional[tuple]:
    """``(v_norm, vmap)`` for one row — memoized on the row's own
    content digest; ``None`` is the malformed verdict (never cached —
    the decode path re-runs and re-counts it every call, exactly like
    the uncached lane).

    ``vmap`` maps each nonzero dim to its value in ascending-dim order.
    Scoring multiplies ``query[j]`` by the stored value over the
    intersection of nonzero dims only: the dropped products are
    ``q_j * ±0.0 = ±0.0`` addends, and ``math.fsum``'s compensated sum
    is untouched by zero addends — so the sparse-order dot is the
    *identical* correctly-rounded value the dense reference produces
    (same operand multiset, same ascending-dim order) at the hashing
    encoder's ~25% density, ~4× fewer interpreter steps."""
    entry = memo.get(key) if memo is not None else None
    if entry is not None:
        memo.move_to_end(key)
        return entry
    vec = Float32Codec.unpack(bytes(vec_blob), dims)
    if not all(map(math.isfinite, vec)):
        return None
    v_norm_sq = math.fsum(x * x for x in vec)
    if v_norm_sq <= 0.0 or not math.isfinite(v_norm_sq):
        return None
    # ``vmap`` iterates in ascending-dim order (range() ascends and dict
    # preserves insertion order) — the dot's intersection pass reads
    # products in the same order the dense reference emits them.
    vmap = {j: vec[j] for j in range(dims) if vec[j] != 0.0}
    entry = (math.sqrt(v_norm_sq), vmap)
    if memo is not None:
        memo[key] = entry
        while len(memo) > _VEC_MEMO_MAX:
            memo.popitem(last=False)
    return entry


#: Whole-scan top-k memo for ``vector_candidates`` — ``(qkey, scan
#: identity, content fingerprint) -> (frozen hits, frozen stats)``. A
#: hit replays the identical ordered hits + coverage counters a fresh
#: scan would compute: the fingerprint folds every deciding column of
#: every row the scan could read (``rowid`` catches INSERT OR REPLACE
#: rewrites, which always mint a new rowid even when the PK is reused;
#: ``digest`` content-addresses the vector bytes; the aggregate count
#: is implicit in the fold), scoped to the scan's own WHERE so rows the
#: predicate excludes can never invalidate. Only ``elig=None`` scans
#: memoize — an explicit eligibility set is caller state outside the
#: table fingerprint. Deadline-cut scans are never memoized: a
#: ``partial`` result must never replay as a complete one.
_VTOPK_MEMO_ATTR = "_vector_topk_memo_v1"
_VTOPK_MEMO_MAX = 256


def _vtopk_memo(store: Any) -> Optional[OrderedDict]:
    if store is None:
        return None
    memo = getattr(store, _VTOPK_MEMO_ATTR, None)
    if memo is None:
        try:
            memo = OrderedDict()
            setattr(store, _VTOPK_MEMO_ATTR, memo)
        except Exception:
            return None
    return memo


def _vector_scan_fp(
    conn: sqlite3.Connection,
    *,
    namespace: Optional[str],
    encoder: str,
    generation: Optional[int],
) -> Optional[str]:
    """Content fingerprint over exactly the rows a scan's WHERE admits —
    PK-ordered ``rowid|digest`` fold. Equality is byte-for-byte identity
    of the scored universe: an unseen row, a removed row, and a
    replaced blob all move the fold."""
    sql = (
        "SELECT group_concat(q, '') FROM ("
        " SELECT quote(rowid)||'|'||quote(digest) AS q"
        " FROM source_vectors WHERE encoder = ?"
    )
    params: list = [encoder]
    if namespace is not None:
        sql += " AND namespace = ?"
        params.append(namespace)
    if generation is not None:
        sql += _gen_pred()
        params.append(generation)
    sql += " ORDER BY source_id, revision, encoder)"
    try:
        row = conn.execute(sql, params).fetchone()
    except sqlite3.Error:
        return None
    return row[0] if row else None


def _freeze_lane_result(
    hits: list, stats: "SourceLaneStats", *, signal: str, lane: str
) -> tuple:
    """Immutable snapshot of a completed lane scan — hits as
    ``(source_id, revision, score, rank)`` tuples plus the stats'
    ``to_dict`` payload (deep-copied so downstream mutation of the
    returned objects can never touch the memo)."""
    frozen_hits = tuple(
        (h.source_id, int(h.revision),
         float(h.signals.get(signal) or 0.0),
         int(h.lanes.get(lane) or 0))
        for h in hits
    )
    frozen_stats = stats.to_dict()
    frozen_stats["warnings"] = tuple(frozen_stats["warnings"])
    frozen_stats["details"] = dict(
        (k, (dict(v) if isinstance(v, dict) else v))
        for k, v in frozen_stats["details"].items()
    )
    return (frozen_hits, frozen_stats)


def _replay_lane_result(
    frozen: tuple, *, signal: str, lane: str
) -> tuple:
    """Rebuild the ``(hits, stats)`` pair a completed scan returned —
    fresh ``SourceHit``/``SourceLaneStats`` objects carrying the frozen
    values, so callers may mutate freely (identical to a fresh call)."""
    frozen_hits, frozen_stats = frozen
    stats = SourceLaneStats(lane)
    stats.status = frozen_stats["status"]
    stats.reason = frozen_stats["reason"]
    stats.eligible = frozen_stats["eligible"]
    stats.candidates_examined = frozen_stats["candidates_examined"]
    stats.scored = frozen_stats["scored"]
    stats.returned = frozen_stats["returned"]
    stats.truncated = frozen_stats["truncated"]
    stats.deadline_exceeded = frozen_stats["deadline_exceeded"]
    stats.warnings = list(frozen_stats["warnings"])
    stats.details = dict(
        (k, (dict(v) if isinstance(v, dict) else v))
        for k, v in frozen_stats["details"].items()
    )
    hits: list = []
    for sid, rev, score, rank in frozen_hits:
        hit = SourceHit(sid, rev)
        hit.signals[signal] = score
        hit.lanes[lane] = rank
        hits.append(hit)
    return hits, stats


def _vector_where(
    *,
    namespace: Optional[str],
    encoder: str,
    generation: Optional[int],
    elig: Optional[tuple],
) -> list:
    """``[(where_sql, params)]`` — one group per streaming pass.

    The WHERE tail is identical for the row scan and the eligible-count
    probe, so the coverage denominator provably shares the scan's
    predicate (V5-11.03: the same eligibility bounds both).
    """
    groups: list = []
    if elig is not None:
        ids = sorted(elig[0])
        for chunk in _cand._chunks(ids, _IN_CHUNK):
            sql = f"encoder = ? AND source_id IN ({_ph(len(chunk))})"
            params: list = [encoder, *chunk]
            if namespace is not None:
                sql += " AND namespace = ?"
                params.append(namespace)
            if generation is not None:
                sql += _gen_pred()
                params.append(generation)
            groups.append((sql, params))
    else:
        sql = "encoder = ?"
        params = [encoder]
        if namespace is not None:
            sql += " AND namespace = ?"
            params.append(namespace)
        if generation is not None:
            sql += _gen_pred()
            params.append(generation)
        groups.append((sql, params))
    return groups


def vector_candidates(
    conn: sqlite3.Connection,
    *,
    query_vector: Any,
    eligible_ids: Optional[Iterable] = None,
    namespace: Optional[str] = None,
    encoder: Optional[str] = None,
    generation: Optional[int] = None,
    limit: int = SOURCE_VECTOR_LIMIT,
    deadline: Any = None,
    deadline_ms: Optional[float] = None,
    store: Any = None,
) -> tuple:
    """Blocked exact cosine scan over ``source_vectors`` (V5-12.01/33.04).

    Streams bounded ``SCAN_BATCH`` rounds of eligible vectors, decodes
    each blob contiguously (one ``Float32Codec.unpack`` per row), and
    keeps a ``limit``-bounded top-k heap — memory stays O(batch + k) for
    any eligible-set size. Exclusions are counted by cause, mirroring
    ``vectors.ScanCoverage``: ``malformed`` (corrupt blob, zero/non-finite
    norm) and ``generation`` (dimension/space mismatch). A deadline cut
    returns the best of the examined prefix flagged ``partial``.
    """
    dl = _resolve_deadline(deadline, deadline_ms)
    stats = SourceLaneStats("vector")
    if not _has_table(conn, "source_vectors"):
        stats.status = "unavailable"
        stats.reason = "no_source_vectors"
        return [], stats
    blob = _query_blob(query_vector)
    if not blob or len(blob) % 4:
        stats.status = "unavailable"
        stats.reason = "malformed_query_vector"
        return [], stats
    dims = len(blob) // 4
    if dims < 1 or dims > _vec.MAX_DIMENSIONS:
        stats.status = "unavailable"
        stats.reason = "malformed_query_vector"
        return [], stats
    query = Float32Codec.unpack(blob, dims)
    q_norm_sq = math.fsum(x * x for x in query)
    if q_norm_sq <= 0.0 or not math.isfinite(q_norm_sq):
        stats.status = "unavailable"
        stats.reason = "malformed_query_vector"
        return [], stats
    q_norm = math.sqrt(q_norm_sq)
    # Query nonzero dims once — the per-row dot iterates the smaller of
    # (query nz, row nz) probing the other's map; dict insertion order
    # is ascending dims so both branches emit products in the dense
    # reference's addend order.
    qmap = {j: query[j] for j in range(dims) if query[j] != 0.0}
    qnz_items = tuple(qmap.items())

    encoder_id = _pinned_encoder(conn, encoder, namespace)
    if encoder_id is None:
        stats.status = "unavailable"
        stats.reason = "encoder_unresolved"
        return [], stats

    elig = _normalize_eligible(eligible_ids)
    if elig is not None and not elig[0]:
        stats.reason = "empty_eligible_set"
        return [], stats
    if elig is None and namespace is None:
        # No eligibility list and no namespace partition — a store-wide
        # scan could mix foreign vectors into one ranking (V5-10.03);
        # refuse rather than widen.
        stats.status = "unavailable"
        stats.reason = "unscoped_request"
        return [], stats

    # ``eligible`` counts rows surviving pair-precise membership — the
    # same predicate the scan scores under (V5-11.03); ``examined`` is
    # everything fetched. A deadline cut leaves eligible as the examined
    # subset's count and flags ``partial`` — never a silent truncation.
    where_groups = _vector_where(
        namespace=namespace, encoder=encoder_id,
        generation=generation, elig=elig,
    )

    # Decoded-vector memo (content-addressed by the row's own digest) —
    # a hit skips the unpack + finite check + norm without changing any
    # score, exclusion count, or ordering (determinism preserved: the
    # memoized values are byte-identical recomputations).
    memo = _vector_memo(store)
    # Finished-score memo — ``(query-vector digest, row key) -> value``;
    # a hit skips the dot entirely and replays the identical float.
    vsmemo = _vscore_memo(store)
    # The score is fully determined by (q_norm, qnz_items): key on those
    # — content-exact for any sequence type (tuple/list/array/numpy),
    # and a query differing only in zero entries hashes identically,
    # correctly, since its scores are identical too.
    qkey = hashlib.blake2b(
        struct.pack("<d", q_norm)
        + struct.pack(
            "<%di" % len(qnz_items), *(d for d, _v in qnz_items)
        )
        + struct.pack(
            "<%dd" % len(qnz_items), *(_v for _d, _v in qnz_items)
        ),
        digest_size=16,
    ).digest()

    # Whole-scan top-k memo: a fingerprint-identical table + identical
    # query vector replays the frozen result — the scan is a pure
    # function of exactly those inputs (elig scans stay uncached —
    # the eligibility set is caller state outside the fingerprint).
    tk_memo = _vtopk_memo(store) if elig is None else None
    tk_key = None
    if tk_memo is not None:
        tk_fp = _vector_scan_fp(
            conn, namespace=namespace, encoder=encoder_id,
            generation=generation,
        )
        if tk_fp is not None:
            tk_key = (qkey, namespace, encoder_id, generation,
                      int(limit), tk_fp)
            ent = tk_memo.get(tk_key)
            if ent is not None:
                tk_memo.move_to_end(tk_key)
                return _replay_lane_result(
                    ent, signal=SIGNAL_SIMILARITY, lane="vector")

    heap: list = []  # (score, _Desc-like sid order, revision, source_id)
    malformed = space = 0
    done = False
    blob_len = dims * 4

    def _push_value(value: float, sid: Any, rev: int) -> None:
        stats.scored += 1
        # Top-k heap ordered worst-first: smallest score, then largest
        # source_id, then smallest revision pops out. The float compare
        # rejects strictly-losing values without building the ordering
        # entry — ``entry > heap[0]`` can only hold when the score is
        # not smaller than the current worst (equal scores fall through
        # to the full tuple compare, unchanged).
        if len(heap) >= limit and value < heap[0][0]:
            return
        entry = (value, _DescStr(sid), rev, sid, rev)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

    def _score_and_push(
        v_norm: float, vmap: dict, sid: Any, rev: int, skey: tuple
    ) -> None:
        """Score one decoded row — sparse exact dot over the nonzero-dim
        intersection: iterate the smaller side probing the other's map.
        Every dropped addend is q_j * ±0.0 = ±0.0, and fsum is invariant
        under zero addends, so this is the SAME compensated-sum reference
        value (V5-31.03 kept, never BLAS) at a fraction of the
        interpreter work. Both branches emit products in ascending-dim
        order — the dense reference's addend order."""
        if len(qnz_items) <= len(vmap):
            vget = vmap.get
            products = [
                q_d * v
                for d, q_d in qnz_items
                if (v := vget(d)) is not None
            ]
        else:
            qget = qmap.get
            products = [
                q_d * v
                for d, v in vmap.items()
                if (q_d := qget(d)) is not None
            ]
        # fsum over ≤1 addend is the addend itself (or 0.0) in every case
        # — including inf/nan — so the skip is bit-identical; longer lists
        # take the same compensated sum the dense reference applies to the
        # same ascending-dim addend sequence.
        if len(products) <= 1:
            dot = products[0] if products else 0.0
        else:
            dot = math.fsum(products)
        value = dot / (q_norm * v_norm)
        if vsmemo is not None:
            vsmemo[skey] = value
            while len(vsmemo) > _VSCORE_MEMO_MAX:
                vsmemo.popitem(last=False)
        _push_value(value, sid, rev)

    for where, params in where_groups:
        if done:
            break
        # Index pass: identity + digest + the cheap ``length``/``typeof``
        # probe reproducing the inline malformed verdict (non-blob type
        # or wrong byte length) WITHOUT dragging ~1.5KB of vector bytes
        # through the cursor — ``length()`` on a blob reads the value
        # header, not the payload. Vector bytes are fetched in a second
        # pass ONLY for rows whose content key misses the decode memo:
        # a warm query moves ~60B/row instead of ~1.5KB/row, and a row
        # can never enter the memo without passing the full decode-time
        # validation once (malformed verdicts are never cached).
        cur = conn.execute(
            "SELECT rowid, source_id, revision, digest,"
            " length(vector), typeof(vector) FROM source_vectors"
            f" WHERE {where} ORDER BY source_id, revision",
            params,
        )
        while True:
            if dl.expired():
                stats.status = "partial"
                stats.reason = "deadline"
                stats.deadline_exceeded = True
                if "scan_incomplete:deadline" not in stats.warnings:
                    stats.warnings.append("scan_incomplete:deadline")
                done = True
                break
            batch = cur.fetchmany(_vec.SCAN_BATCH)
            if not batch:
                break
            # Every fetched row is examined — the increment is
            # unconditional, so counting by batch is the same total.
            stats.candidates_examined += len(batch)
            pending: list = []  # (rowid, sid, rev, key-or-None) misses
            for rowid, sid, rev, vec_digest, vlen, vtype in batch:
                if elig is not None and not _eligible_member(
                    elig, sid, int(rev)
                ):
                    continue
                stats.eligible += 1
                if vtype != "blob" or vlen != blob_len:
                    malformed += 1
                    continue
                key = str(vec_digest) if vec_digest else None
                skey = (qkey, key) if key is not None else None
                cached = (
                    vsmemo.get(skey)
                    if (skey is not None and vsmemo is not None)
                    else None
                )
                if cached is not None:
                    vsmemo.move_to_end(skey)
                    _push_value(cached, sid, int(rev))
                    continue
                decoded = (
                    memo.get(key)
                    if (key is not None and memo is not None)
                    else None
                )
                if decoded is not None:
                    memo.move_to_end(key)
                    v_norm, vmap = decoded
                    _score_and_push(v_norm, vmap, sid, int(rev), skey)
                else:
                    pending.append((int(rowid), sid, int(rev), key))
            if not pending:
                continue
            blobs: dict = {}
            rowids = [p[0] for p in pending]
            for i in range(0, len(rowids), _IN_CHUNK):
                part = rowids[i : i + _IN_CHUNK]
                for r_rowid, r_blob in conn.execute(
                    "SELECT rowid, vector FROM source_vectors"
                    f" WHERE rowid IN ({_ph(len(part))})",
                    part,
                ):
                    blobs[int(r_rowid)] = r_blob
            for rowid, sid, rev, key in pending:
                vec_blob = blobs.get(rowid)
                if (not isinstance(vec_blob, (bytes, bytearray, memoryview))
                        or len(vec_blob) != blob_len):
                    malformed += 1
                    continue
                if key is None:
                    key = hashlib.blake2b(
                        bytes(vec_blob), digest_size=16
                    ).hexdigest()
                skey = (qkey, key)
                cached = (
                    vsmemo.get(skey) if vsmemo is not None else None
                )
                if cached is not None:
                    vsmemo.move_to_end(skey)
                    _push_value(cached, sid, rev)
                    continue
                decoded = _decode_vector(memo, key, vec_blob, dims)
                if decoded is None:
                    malformed += 1
                    continue
                v_norm, vmap = decoded
                _score_and_push(v_norm, vmap, sid, rev, skey)

    stats.details["excluded_malformed"] = malformed
    stats.details["excluded_generation"] = space
    stats.details["encoder"] = encoder_id
    if malformed or space:
        stats.warnings.append("vector_rows_excluded")

    ordered = sorted(heap, key=lambda e: (-e[0], e[3], -e[4]))
    stats.truncated = stats.truncated or stats.scored > len(ordered)
    hits: list = []
    for rank, (score, _d, _r, sid, rev) in enumerate(ordered, start=1):
        hit = SourceHit(sid, rev)
        hit.signals[SIGNAL_SIMILARITY] = score
        hit.lanes["vector"] = rank
        hits.append(hit)
    stats.returned = len(hits)
    if (
        tk_memo is not None
        and tk_key is not None
        and not done
        and not stats.deadline_exceeded
    ):
        # Completed scans only — a deadline-cut partial is per-call
        # state and must never replay as the complete answer.
        tk_memo[tk_key] = _freeze_lane_result(
            hits, stats, signal=SIGNAL_SIMILARITY, lane="vector")
        while len(tk_memo) > _VTOPK_MEMO_MAX:
            tk_memo.popitem(last=False)
    return hits, stats


class _DescStr(str):
    """Inverted string ordering for deterministic top-k heap ties.

    Min-heap pops the smallest entry; wrapping ``source_id`` here makes a
    *larger* id compare smaller, so equal scores evict the
    lexicographically-last id first — matching the declared output order
    ``(score desc, source_id asc, revision desc)``.
    """

    def __lt__(self, other: str) -> bool:
        return str.__gt__(self, other)

    def __le__(self, other: str) -> bool:
        return str.__ge__(self, other)


# ---------------------------------------------------------------------------
# identifier/entity candidates — exact entity_postings hits (V5-30.14/17)
# ---------------------------------------------------------------------------


def identifier_candidates(
    conn: sqlite3.Connection,
    *,
    identifiers: Sequence[str] = (),
    entities: Sequence[str] = (),
    eligible_ids: Optional[Iterable] = None,
    namespace: Optional[str] = None,
    generation: Optional[int] = None,
    limit: int = SOURCE_POSTING_LIMIT,
    deadline: Any = None,
    deadline_ms: Optional[float] = None,
) -> tuple:
    """Exact ``entity_postings`` hits — byte-exact match, never folded.

    Identifier postings preserve case, punctuation, and version segments
    exactly (V5-30.17): matching is SQL ``=`` on the stored entity value
    with no normalization on either side. Values listed in
    ``identifiers`` produce the ``identifier_hit`` signal; values in
    ``entities`` produce ``entity_overlap`` (distinct matched-entity
    count). Returns ``([SourceHit], SourceLaneStats)`` in deterministic
    ``(source_id asc, revision desc)`` order.
    """
    dl = _resolve_deadline(deadline, deadline_ms)
    stats = SourceLaneStats("identifier")
    if not _has_table(conn, "entity_postings"):
        stats.status = "unavailable"
        stats.reason = "no_entity_postings"
        return [], stats

    id_values = sorted({v for v in identifiers if v})
    ent_values = sorted({v for v in entities if v})
    if not id_values and not ent_values:
        stats.reason = "empty_identifiers"
        return [], stats

    elig = _normalize_eligible(eligible_ids)
    if elig is not None and not elig[0]:
        stats.reason = "empty_eligible_set"
        return [], stats
    if elig is None and namespace is None:
        stats.status = "unavailable"
        stats.reason = "unscoped_request"
        return [], stats

    gen_pred = _gen_pred() if generation is not None else ""
    gen_params: list = [] if generation is None else [generation]
    ns_pred = " AND namespace = ?" if namespace is not None else ""
    ns_params: list = [] if namespace is None else [namespace]

    # value -> which signal class asked for it ("identifier" wins ties so
    # a value submitted as both still counts as the exact hit).
    wanted: dict = {}
    for v in ent_values:
        wanted[v] = "entity"
    for v in id_values:
        wanted[v] = "identifier"

    matched: dict = {}   # (sid, rev) -> {"identifier": set, "entity": set}
    overflow = 0
    stop = False
    for chunk in _cand._chunks(sorted(wanted), _IN_CHUNK):
        if stop:
            break
        sql = (
            "SELECT entity, source_id, revision FROM entity_postings"
            f" WHERE entity IN ({_ph(len(chunk))}){ns_pred}{gen_pred}"
            " ORDER BY source_id, revision"
        )
        cur = conn.execute(sql, [*chunk, *ns_params, *gen_params])
        while True:
            if dl.expired():
                stats.status = "partial"
                stats.reason = "deadline"
                stats.deadline_exceeded = True
                if "scan_incomplete:deadline" not in stats.warnings:
                    stats.warnings.append("scan_incomplete:deadline")
                stop = True
                break
            batch = cur.fetchmany(512)
            if not batch:
                break
            for entity, sid, rev in batch:
                stats.candidates_examined += 1
                if elig is not None and not _eligible_member(
                    elig, sid, int(rev)
                ):
                    continue
                stats.eligible += 1
                entry = matched.setdefault(
                    (sid, int(rev)), {"identifier": set(), "entity": set()}
                )
                entry[wanted[entity]].add(entity)
    ordered = sorted(matched, key=lambda k: (k[0], -k[1]))
    stats.scored = len(ordered)
    if len(ordered) > limit:
        overflow = len(ordered) - limit
        stats.truncated = True
    hits: list = []
    for rank, key in enumerate(ordered[:limit], start=1):
        sid, rev = key
        hit = SourceHit(sid, rev)
        if matched[key]["identifier"]:
            hit.signals[SIGNAL_IDENTIFIER] = float(
                len(matched[key]["identifier"])
            )
        if matched[key]["entity"]:
            hit.signals[SIGNAL_ENTITY_OVERLAP] = float(
                len(matched[key]["entity"])
            )
        hit.lanes["identifier"] = rank
        hits.append(hit)
    stats.returned = len(hits)
    stats.details["overflow"] = overflow
    return hits, stats


# ---------------------------------------------------------------------------
# entity timeline — §30.16 read over entity_postings + source_state
# ---------------------------------------------------------------------------


#: Bounded parse memo for ``_rfc3339_ts`` — the function is pure (one
#: input string always maps to one epoch value or ``None``), so the
#: parsed result is content-keyed and can never go stale. Lifecycle
#: scans re-parse the same ``source_state`` cells on every search; the
#: memo turns those repeats into dict hits. ``None`` is a cached value
#: too — the sentinel distinguishes "not yet parsed" from "unparseable".
_TS_MEMO_MAX = 8192
_ts_memo: "OrderedDict[str, Optional[float]]" = OrderedDict()
_TS_MISS = object()


def _rfc3339_ts(value: Any) -> Optional[float]:
    """RFC3339/ISO-8601 → epoch seconds; ``None`` when unparseable."""
    if not isinstance(value, str) or not value.strip():
        return None
    key = value.strip()
    hit = _ts_memo.get(key, _TS_MISS)
    if hit is not _TS_MISS:
        try:
            _ts_memo.move_to_end(key)
        except KeyError:
            pass  # raced with a concurrent eviction — serve the hit
        return hit  # type: ignore[return-value]
    try:
        dt = datetime.fromisoformat(key)
    except ValueError:
        ts: Optional[float] = None
    else:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        try:
            ts = dt.timestamp()
        except (OverflowError, OSError, ValueError):
            ts = None
    _ts_memo[key] = ts
    if len(_ts_memo) > _TS_MEMO_MAX:
        try:
            _ts_memo.popitem(last=False)
        except KeyError:
            pass  # concurrent eviction already shrank the map
    return ts


#: Bounded verdict memo for ``_current_window_ok``-style lifecycle
#: checks — the per-row verdict is a pure function of the row's
#: ``(disposition, effective_at, valid_from, valid_to)`` cells plus the
#: evaluation instant. A verdict can only flip passively when a
#: declared bound is crossed, so each entry stores the earliest future
#: flip instant (``None`` = no passive flip exists) computed exactly as
#: ``_admissible_source_ids`` accounts it; serving is correct while
#: ``now < boundary``, precisely the contract ``not_after_ts`` already
#: exports to callers of ``_admissible_source_ids``.
_WIN_MEMO_MAX = 8192
_window_memo: "OrderedDict[tuple, tuple]" = OrderedDict()


def _window_verdict(
    disp: str, eff: Any, vf: Any, vt: Any, now_ts: float
) -> tuple:
    """``(admissible, not_after_ts)`` for one ``source_state`` row.

    Identical to evaluating ``_current_window_ok`` plus the future-bound
    accounting of ``_admissible_source_ids`` inline — memoized on the
    row's cell tuple so repeated searches over the same rows are dict
    hits. An entry is served only while ``now_ts < not_after_ts``: past
    the earliest declared bound the verdict is recomputed (the bound
    may have flipped it), never replayed.
    """
    key = (disp, eff, vf, vt)
    ent = _window_memo.get(key)
    if ent is not None:
        verdict, boundary = ent
        if boundary is None or now_ts < boundary:
            try:
                _window_memo.move_to_end(key)
            except KeyError:
                pass  # raced with a concurrent eviction — serve it
            return verdict, boundary
    verdict = _current_window_ok(disp, eff, vf, vt, now_ts)
    boundary: Optional[float] = None
    if disp == "active":
        for cell in (vf, vt):
            ts = _rfc3339_ts(cell)
            if ts is not None and ts > now_ts and (
                boundary is None or ts < boundary
            ):
                boundary = ts
    elif disp == "superseded":
        ts = _rfc3339_ts(eff)
        if ts is not None and ts > now_ts:
            boundary = ts
    _window_memo[key] = (verdict, boundary)
    if len(_window_memo) > _WIN_MEMO_MAX:
        try:
            _window_memo.popitem(last=False)
        except KeyError:
            pass  # concurrent eviction already shrank the map
    return verdict, boundary


def entity_timeline(
    conn: sqlite3.Connection,
    namespace: str,
    entity: str,
    *,
    generation: Optional[int] = None,
    limit: int = SOURCE_TIMELINE_LIMIT,
    deadline: Any = None,
    deadline_ms: Optional[float] = None,
) -> tuple:
    """All authorized records mentioning ``entity``, time-ordered (§30.16).

    Reads ``entity_postings`` for the namespace (exact entity value), then
    decorates each mention with its ``source_state`` lifecycle label —
    ``disposition`` verbatim plus ``current`` (``disposition == "active"``;
    absent control rows report ``"unknown"``, never inferred approval,
    V5-14.16) — and event/recorded time (``enrichment.event_at`` when a
    T1 row exists, else ``source_revisions.event_us``/``captured_us``).

    Order: ascending event/recorded time, then ``(source_id, revision)``.
    ``limit`` bounds the returned entries; ``truncated`` reports eligible
    mentions beyond it. Returns ``([entry_dict], SourceLaneStats)``.
    """
    dl = _resolve_deadline(deadline, deadline_ms)
    stats = SourceLaneStats("entity_timeline")
    if not _has_table(conn, "entity_postings"):
        stats.status = "unavailable"
        stats.reason = "no_entity_postings"
        return [], stats
    if not entity:
        stats.reason = "empty_entity"
        return [], stats

    gen_pred = _gen_pred() if generation is not None else ""
    gen_params: list = [] if generation is None else [generation]
    rows = conn.execute(
        "SELECT source_id, revision, entity_kind, offsets"
        " FROM entity_postings WHERE namespace = ? AND entity = ?"
        f"{gen_pred} ORDER BY source_id, revision",
        [namespace, entity, *gen_params],
    ).fetchall()
    stats.candidates_examined = len(rows)
    stats.eligible = len(rows)
    if not rows:
        return [], stats

    sids = sorted({r[0] for r in rows})
    # -- source_state lifecycle labels (control artifact; §14.3) --------
    state_by_src: dict = {}
    if _has_table(conn, "source_state"):
        for chunk in _cand._chunks(sids, _IN_CHUNK):
            if dl.expired():
                stats.status = "partial"
                stats.reason = "deadline"
                stats.deadline_exceeded = True
                break
            for r in conn.execute(
                "SELECT source_id, disposition, mutation_head,"
                " superseded_by, effective_at, known_at, valid_from,"
                " valid_to, control_version FROM source_state"
                f" WHERE source_id IN ({_ph(len(chunk))})"
                " AND namespace = ?",
                [*chunk, namespace],
            ).fetchall():
                state_by_src[r[0]] = r

    # -- event/recorded time: enrichment first, source_revisions after --
    # When several producers enriched the same revision the pinned T1
    # producer (``enrich/v1``) wins; otherwise the alphabetically-first
    # producer is used so the pick stays deterministic.
    event_by_key: dict = {}
    if _has_table(conn, "enrichment"):
        pairs = sorted({(r[0], int(r[1])) for r in rows})
        for chunk in _cand._chunks(pairs, _cand._PAIR_CHUNK):
            if dl.expired():
                stats.status = "partial"
                stats.reason = "deadline"
                stats.deadline_exceeded = True
                break
            where = " OR ".join(
                "(source_id = ? AND revision = ?)" for _ in chunk
            )
            flat = [v for pair in chunk for v in pair]
            for r in conn.execute(
                "SELECT source_id, revision, event_at, anchor_at,"
                " time_status, time_precision, producer FROM enrichment"
                f" WHERE ({where})"
                " ORDER BY (producer = 'enrich/v1') DESC, producer",
                flat,
            ).fetchall():
                event_by_key.setdefault((r[0], int(r[1])), r)
    times_by_src: dict = {}
    if _has_table(conn, "source_revisions"):
        pairs = sorted({(r[0], int(r[1])) for r in rows})
        for chunk in _cand._chunks(pairs, _cand._PAIR_CHUNK):
            if dl.expired():
                stats.status = "partial"
                stats.reason = "deadline"
                stats.deadline_exceeded = True
                break
            where = " OR ".join(
                "(source_id = ? AND revision = ?)" for _ in chunk
            )
            flat = [v for pair in chunk for v in pair]
            for r in conn.execute(
                "SELECT source_id, revision, event_us, captured_us"
                f" FROM source_revisions WHERE ({where})",
                flat,
            ).fetchall():
                times_by_src[(r[0], int(r[1]))] = (r[2], r[3])

    entries: list = []
    for sid, rev, entity_kind, offsets in rows:
        st = state_by_src.get(sid)
        enr = event_by_key.get((sid, rev))
        times = times_by_src.get((sid, rev))
        event_at = enr[2] if enr else None
        event_ts = _rfc3339_ts(event_at)
        recorded_ts: Optional[float] = None
        recorded_at: Optional[str] = None
        if times is not None and times[0] is not None:
            rev_event_ts = float(times[0]) / 1e6
        else:
            rev_event_ts = None
        if times is not None and times[1] is not None:
            recorded_ts = float(times[1]) / 1e6
        if recorded_ts is None and st is not None:
            recorded_ts = _rfc3339_ts(st[5])
            recorded_at = st[5]
        sort_ts = (
            event_ts if event_ts is not None
            else rev_event_ts if rev_event_ts is not None
            else recorded_ts
        )
        disposition = st[1] if st is not None else "unknown"
        try:
            head_rev: Optional[int] = (
                int(st[2]) if st is not None and st[2] is not None else None
            )
        except (TypeError, ValueError):
            head_rev = None
        try:
            parsed_offsets = safe_json_loads(offsets) if offsets else []
        except VerbatimError:
            parsed_offsets = offsets  # producer payload kept raw
        entries.append({
            "source_id": sid,
            "revision": rev,
            "entity": entity,
            "entity_kind": entity_kind,
            "offsets": parsed_offsets,
            "lifecycle": disposition,
            "current": disposition == "active",
            "mutation_head": head_rev,
            # headship is claimed only on a parseable head match — an
            # unresolved/none head marks no revision current-head.
            "is_head": head_rev is not None and rev == head_rev,
            "control_version": int(st[8]) if st is not None else None,
            "superseded_by": st[3] if st is not None else None,
            "effective_at": st[4] if st is not None else None,
            "valid_from": st[6] if st is not None else None,
            "valid_to": st[7] if st is not None else None,
            "event_at": event_at,
            "recorded_at": recorded_at,
            "time_status": enr[4] if enr else None,
            "time_precision": enr[5] if enr else None,
            "_sort_ts": sort_ts,
        })

    # Chronological order; timeless entries last; deterministic ties.
    entries.sort(
        key=lambda e: (
            e["_sort_ts"] is None,
            e["_sort_ts"] if e["_sort_ts"] is not None else 0.0,
            e["source_id"],
            e["revision"],
        )
    )
    stats.scored = len(entries)
    if len(entries) > limit:
        stats.truncated = True
        stats.details["overflow"] = len(entries) - limit
    out = entries[:limit]
    for e in out:
        del e["_sort_ts"]
    stats.returned = len(out)
    return out, stats


# ---------------------------------------------------------------------------
# orchestrator — the §6/§10 source candidate lane
# ---------------------------------------------------------------------------


def source_candidates(
    store: Any,
    conn: sqlite3.Connection,
    *,
    query_terms: Sequence[str] = (),
    identifiers: Sequence[str] = (),
    entities: Sequence[str] = (),
    eligible_ids: Optional[Iterable] = None,
    namespace: Optional[str] = None,
    snapshot: Any = None,
    limit: int = SOURCE_LANE_LIMIT,
    query_vector: Any = None,
    deadline: Any = None,
    deadline_ms: Optional[float] = None,
) -> tuple:
    """The source lane: lexical BM25 + hashing similarity + exact postings.

    Composes the three sub-lanes over the same eligible universe E and
    snapshot generation, merges per-source signals into ``SourceHit``
    records, and returns ``(hits, stats)`` ordered by each candidate's
    best sub-lane rank (deterministic: ties on ``(source_id, revision)
    desc``). Fusion (``ranking/v1``) is the ranking stage — this lane
    only produces candidates with raw signal values (V5-31.05).

    ``query_vector`` may be packed float32le bytes or a float sequence;
    when absent the lane encodes ``" ".join(query_terms)`` through the
    store's configured encoder (hashing by default). No encoder → the
    similarity channel reports ``unavailable`` without failing the lane.
    """
    dl = _resolve_deadline(deadline, deadline_ms)
    stats = SourceLaneStats("source")
    generation = _snapshot_generation(conn, snapshot)
    if generation is None:
        stats.warnings.append("generation_unfenced")

    # ---- whole-lane memo (namespace-scoped call shape only) ----------
    # Every input the merged result depends on is either in the key
    # (terms/ident/ents/query vector/limit/generation/encoder identity)
    # or fingerprinted (``_snapshot_version`` over the five content
    # tables — the lifecycle table is deliberately out of scope because
    # admissibility is re-checked live on serve, below). ``eligible_ids``
    # callers keep the live path — the set is not cheaply keyable.
    # Partial/deadline-cut results are never stored.
    sc_memo = None
    sc_key = None
    sc_version = None
    encode_fn = None
    encoder_id: Optional[str] = None
    qblob = _query_blob(query_vector)
    # Encoder resolution is cheap (identity only — the encode itself
    # happens lazily below so a memo hit skips it) and applies to every
    # call shape, not just the memoizable one.
    if qblob is None and query_terms and store is not None:
        resolved = _cand._query_encoder(store)
        if resolved is not None:
            encode_fn, encoder_id = resolved
    if eligible_ids is None and store is not None:
        sc_version = _snapshot_version(conn, store, tables=_SC_FP_TABLES)
        if sc_version is not None:
            sc_memo = _sc_memo(store)
            if sc_memo is not None:
                qv_key = (
                    hashlib.blake2b(qblob, digest_size=16).hexdigest()
                    if qblob is not None else None
                )
                sc_key = (
                    # order is significant: " ".join(query_terms) feeds
                    # the encoder — a sorted key would collide two
                    # different queries.
                    tuple(str(t) for t in query_terms),
                    tuple(str(i) for i in identifiers),
                    tuple(str(e) for e in entities),
                    qv_key,
                    namespace,
                    generation,
                    int(limit),
                    encoder_id,
                )
                ent = sc_memo.get(sc_key)
                if ent is not None and ent[0] == sc_version:
                    sc_memo.move_to_end(sc_key)
                    raw: list = []
                    for sid, rev, sig, lan in ent[1]:
                        h = SourceHit(sid, rev)
                        h.signals.update(sig)
                        h.lanes.update(lan)
                        raw.append(h)
                    # Live lifecycle re-check — identical to the fresh
                    # path's ``_admissible_source_ids`` over the merged
                    # set (the memo fingerprint intentionally excludes
                    # source_state, so this is what keeps the answer
                    # current across adds/holds/transitions).
                    adm_res = _admissible_source_ids(
                        conn, (h.source_id for h in raw)
                    )
                    admissible = (
                        adm_res[0] if adm_res is not None else None
                    )
                    filtered: list = []
                    lifecycle_excluded = 0
                    if admissible is not None:
                        for h in raw:
                            if admissible.get(h.source_id, True) is False:
                                lifecycle_excluded += 1
                            else:
                                filtered.append(h)
                    else:
                        filtered = raw
                    out_h = filtered[: int(limit)]
                    st_d = ent[2]
                    mstats = SourceLaneStats("source")
                    mstats.status = st_d["status"]
                    mstats.reason = st_d["reason"]
                    mstats.eligible = st_d["eligible"]
                    mstats.candidates_examined = st_d["candidates_examined"]
                    mstats.deadline_exceeded = st_d["deadline_exceeded"]
                    mstats.warnings = list(st_d["warnings"])
                    mstats.details = {
                        k: (
                            dict(v) if isinstance(v, dict)
                            else list(v) if isinstance(v, list)
                            else v
                        )
                        for k, v in st_d["details"].items()
                    }
                    mstats.scored = len(filtered)
                    mstats.truncated = len(filtered) > int(limit)
                    if mstats.truncated:
                        mstats.details["overflow"] = (
                            len(filtered) - int(limit)
                        )
                    else:
                        mstats.details.pop("overflow", None)
                    if lifecycle_excluded:
                        mstats.details["lifecycle_excluded"] = (
                            lifecycle_excluded
                        )
                    else:
                        mstats.details.pop("lifecycle_excluded", None)
                    mstats.returned = len(out_h)
                    return out_h, mstats

    merged: dict = {}
    sub_stats: dict = {}

    def _merge(hits: list, lane: str) -> None:
        for hit in hits:
            entry = merged.get(hit.key)
            if entry is None:
                entry = SourceHit(hit.source_id, hit.revision)
                merged[hit.key] = entry
            entry.signals.update(hit.signals)
            entry.lanes.update(hit.lanes)

    if query_terms:
        hits, sub_stats["lexical"] = lexical_candidates(
            conn, query_terms=query_terms, eligible_ids=eligible_ids,
            namespace=namespace, generation=generation,
            limit=max(limit, SOURCE_LEXICAL_LIMIT), deadline=dl,
            store=store,
        )
        _merge(hits, "lexical")

    if identifiers or entities:
        hits, sub_stats["identifier"] = identifier_candidates(
            conn, identifiers=identifiers, entities=entities,
            eligible_ids=eligible_ids, namespace=namespace,
            generation=generation, limit=max(limit, SOURCE_POSTING_LIMIT),
            deadline=dl,
        )
        _merge(hits, "identifier")

    # similarity channel — pinned-encoder exact scan when a vector exists
    if qblob is None and encode_fn is not None:
        try:
            raw = encode_fn(
                " ".join(query_terms),
                broker=getattr(store, "transport_broker", None),
                scope_ids=(),
                caller="",
            )
            qblob = _query_blob(raw)
        except Exception:
            qblob = None
    if qblob is not None:
        hits, sub_stats["vector"] = vector_candidates(
            conn, query_vector=qblob, eligible_ids=eligible_ids,
            namespace=namespace, encoder=encoder_id,
            generation=generation, limit=max(limit, SOURCE_VECTOR_LIMIT),
            deadline=dl, store=store,
        )
        _merge(hits, "vector")
    else:
        sub_stats["vector"] = SourceLaneStats(
            "vector", status="unavailable",
            reason="no_query_vector",
        )
        stats.warnings.append("vector_unavailable")

    # Deterministic merge order over the whole merged set — computed
    # BEFORE lifecycle admission so the memoized ordering stays a pure
    # function of the content tables. A stable sort restricted to the
    # admissible subset is exactly the fresh order's surviving prefix.
    ordered_all = sorted(
        merged.values(),
        key=lambda h: (min(h.lanes.values()), h.source_id, -h.revision),
    )

    # Lifecycle admission (V5-07.13 query-time eligibility): a merged
    # candidate whose ``source_state`` record is non-current is withheld —
    # the lane surfaces answers, not history (the labelled history read is
    # ``entity_timeline``; raw evidence stays reachable through the
    # governed archive lane). ``None`` map = unprovisioned, skip.
    adm_res = _admissible_source_ids(
        conn, (h.source_id for h in ordered_all)
    )
    admissible = adm_res[0] if adm_res is not None else None
    stats.candidates_examined = len(merged)
    lifecycle_excluded = 0
    if admissible is not None:
        ordered = []
        for h in ordered_all:
            if admissible.get(h.source_id, True) is False:
                lifecycle_excluded += 1
            else:
                ordered.append(h)
    else:
        ordered = ordered_all
    if lifecycle_excluded:
        stats.details["lifecycle_excluded"] = lifecycle_excluded

    # The eligible universe of the merged lane is the lexical corpus size
    # when the BM25 sub-lane ran (its N is the E measurement); the other
    # sub-lanes report their own eligible counts in ``details``.
    lex_stats = sub_stats.get("lexical")
    stats.eligible = lex_stats.eligible if lex_stats is not None else 0
    stats.scored = len(ordered)
    if len(ordered) > limit:
        stats.truncated = True
        stats.details["overflow"] = len(ordered) - limit
    out = ordered[:limit]
    stats.returned = len(out)
    unavailable: list = []
    for name, s in sub_stats.items():
        stats.details[name] = s.to_dict()
        for w in s.warnings:
            if w not in stats.warnings:
                stats.warnings.append(w)
        if s.status == "unavailable":
            unavailable.append(name)
        if s.deadline_exceeded:
            stats.deadline_exceeded = True
            stats.status = "partial"
            stats.reason = stats.reason or "deadline"
    if unavailable:
        # Degradation is declared, never hidden behind a clean status.
        stats.details["degraded_channels"] = unavailable
        if (
            stats.status != "partial"
            and sub_stats
            and len(unavailable) == len(sub_stats)
        ):
            # Every channel the request could use was unavailable: the
            # lane itself is unavailable, not merely empty (V5 honest
            # states — absence of capability ≠ absence of evidence).
            stats.status = "unavailable"
            stats.reason = "channels_unavailable:" + ",".join(unavailable)
        elif stats.status == "ok":
            stats.warnings.append("degraded:" + ",".join(unavailable))
    elif not sub_stats:
        stats.reason = "no_query_signal"

    # Store a memo entry only for complete results — a partial answer
    # is a function of this call's budget, not a reusable data fact.
    # The stored hit list is the FULL pre-lifecycle ordering (capped):
    # serving re-checks admissibility live, so a lifecycle flip promotes
    # exactly the tail the fresh path would surface — a post-limit
    # store could not.
    if (
        sc_memo is not None
        and sc_key is not None
        and stats.status != "partial"
        and len(ordered_all) <= _SC_MEMO_HITCAP
    ):
        sc_memo[sc_key] = (
            sc_version,
            [
                (h.source_id, h.revision, dict(h.signals), dict(h.lanes))
                for h in ordered_all
            ],
            stats.to_dict(),
        )
        while len(sc_memo) > _SC_MEMO_MAX:
            sc_memo.popitem(last=False)
    return out, stats


__all__ = [
    "SOURCE_LANE_LIMIT",
    "SOURCE_LEXICAL_LIMIT",
    "SOURCE_POSTING_LIMIT",
    "SOURCE_TIMELINE_LIMIT",
    "SOURCE_VECTOR_LIMIT",
    "SourceHit",
    "SourceLaneStats",
    "entity_timeline",
    "identifier_candidates",
    "lexical_candidates",
    "source_candidates",
    "vector_candidates",
]
