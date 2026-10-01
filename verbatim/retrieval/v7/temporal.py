"""Temporal lane for the V7 read path (V7-09.06–09).

L-time in the §04.2 pipeline: retrieve units whose ``occurred`` interval
overlaps the resolved query window (and, secondarily, whose ``recorded``
time falls inside it), select by *relevance within the window* — never by
recency — and spread the selection across time buckets so a "what happened
in 2023" query is representative (V7-09.06, Hindsight's rule adopted). For
the answer-bearing temporal intents (``temporal_point``, ``temporal_order``,
``duration``, ``count_aggregate``) the deterministic ``events`` index is
consulted FIRST and event-backed units lead the ranking (V7-09.09); the
plain window scan then supplies corroborating units.

V8 amendments (SPEC_V8 §09):

- ``unit_time_mentions`` overlap unions with ``occurred_*`` overlap in
  the window scan (V8-09.04), and a window-matched unit's mention
  interval — not its session-time ``occurred`` — drives ``t_prox`` and
  bucketing (V8-09.08, event date beats session date);
- the no-window fallback is restricted to units carrying a nominated
  query content term or resolved entity/speaker canon and ordered by
  event time — ``occurred_start_us`` newest-first for recency intents,
  oldest-first for first/earliest, ``recorded_at_us`` as the NULL and
  tie fallback (V8-09.06); the unrestricted ingest-order sweep is gone;
- event matching is a weighted subject/predicate disjunction gated on
  temporal compatibility, weights armed by ``temporal.events_weights``
  (§23, prior 0.5/0.5) — a strict conjunction no longer applies
  (V8-09.07).

Honesty rules honored here (V7-04.03, LaneV7 protocol):

- eligibility is evaluated while candidates are produced — a held or
  otherwise ineligible unit in the window is never emitted, and the cap
  bounds *admitted* candidates, not rows examined;
- a deadline-cut or bound-cut scan reports ``partial`` with an honest
  ``reason`` and real ``examined``/``eligible`` counts;
- a missing ``events`` index on an events-first query degrades to
  ``partial`` (window path still runs) or ``unavailable`` (no window);
  a missing ``unit_time_mentions`` on a windowed query degrades to
  ``partial`` likewise — ``ok`` is only ever earned by real paths;
- no usable window and a non-temporal intent reports
  ``skipped`` with ``reason="no_window"`` — the lane never fabricates a
  temporal contribution.

Signals emitted per candidate (rerank inputs, §32.4): ``t_prox`` —
``1 - |t_unit - window_centre| / half_width`` clamped to [0, 1] (absent
when the centre cannot be computed, never invented); ``overlap`` type
(``mention`` for the V8 union arm); ``mention_interval`` —
``(start_us, end_us, precision)`` of the scoring mention (V8-09.08);
``event_match``/``event_pred``/``subject_match`` flags for the event
index; ``axis`` naming the matched clock (``occurred`` | ``recorded`` |
``mention``); ``bucket`` for the spread selection.

formula_status: ``provisional/v7-r0`` — the composite weights below are
declared constants awaiting the §32.0 formula search, not tuned values.

The lane is stdlib + sqlite3 only. ``schema_v7``, ``temporal_v2`` and
``enrichment/events.py`` are concurrent wave-A modules; this file codes
against the §30 column contract and the frozen ``types_v7`` API only. The
window itself is resolved upstream by ``temporal_v2.resolve_query_window``
(S1) and arrives as ``qv.intent.window`` — the lane never parses time text.
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v7 import (
    CandidateV7,
    IntentClass,
    LaneContextV7,
    LaneOutput,
    LaneSlice,
    LaneStatus,
    LaneV7,
    QueryViewV7,
)

LANE_NAME = "time"  # LaneName.TIME — stable coverage key
LANE_VERSION = "temporal_lane/v8-r1"  # V8-09.* wave; constants declared

DAY_US = 86_400_000_000
MONTH_US = 30 * DAY_US  # "> 1 month" bucket-spread threshold (V7-09.06)
MIN_SPREAD_BUCKETS = 4  # V7-09.06: ≥ 4 buckets for windows > 1 month
MAX_SPREAD_BUCKETS = 12

SCAN_ROW_LIMIT = 4096  # bounded unit scans; hitting it marks partial/scan_bound
EVENT_ROW_LIMIT = 4096
FTS_TERM_LIMIT = 24
FTS_ROW_LIMIT = 1000
IN_CHUNK = 200
DEADLINE_CHECK_ROWS = 64

# V8-09.04/§19 — write-time relative-time mentions, unioned into the
# window scan and preferred over ``units.occurred_*`` for scoring
# (V8-09.08: the event happened then, not when the session ran).
MENTIONS_TABLE = "unit_time_mentions"

#: §23 arm ``temporal.events_weights`` — (subject, predicate) match
#: weights for the V8-09.07 weighted disjunction; prior 0.5 / 0.5.
EVENTS_WEIGHTS_KEY = "temporal.events_weights"
_EVENTS_WEIGHTS_DEFAULT = (0.5, 0.5)
_EV_OBJ_W = 0.5  # object-text overlap headroom (not armed — fixed prior)

#: No-window fallback ordering cues (V8-09.06): recency-seeking intents
#: scan newest event time first; "first/earliest" asks oldest first.
_EARLIEST_TERMS = frozenset(
    {
        "first", "earliest", "oldest", "originally", "initial",
        "initially", "began", "started",
    }
)
_RECENCY_TERMS = frozenset(
    {
        "latest", "recent", "recently", "newest", "current",
        "currently", "now", "lately", "today", "tonight", "last",
        "previous",
    }
)

# Relevance composite weights — provisional/v7-r0, declared not tuned.
W_FTS = 1.5
W_ENT = 1.0
W_EVENT = 2.0
W_TPROX = 0.25

_EVENTS_FIRST = frozenset(
    {
        IntentClass.TEMPORAL_POINT,
        IntentClass.TEMPORAL_ORDER,
        IntentClass.DURATION,
        IntentClass.COUNT_AGGREGATE,
    }
)
_TEMPORAL_INTENTS = _EVENTS_FIRST | frozenset(
    {
        IntentClass.TEMPORAL_RANGE,
        IntentClass.HISTORY_OF,
        IntentClass.CURRENT_VALUE,
    }
)

_TIER_EVENT = 0  # event-index backed units lead temporal intents (V7-09.09)
_TIER_OCCURRED = 1  # occurred-interval overlap with the window
_TIER_RECORDED = 2  # secondary: recorded axis inside the window

_UNIT_COLS = (
    "u.rowid, u.unit_id, u.source_id, u.revision, u.kind, u.session_id,"
    " u.speaker_canon, u.recorded_at_us, u.occurred_start_us,"
    " u.occurred_end_us, u.occurred_precision, u.occurred_source"
)

# V7-30.02 rebuild coexistence: ``units`` is keyed (unit_id, generation)
# and the fence is ``generation <= pinned``. The newest row per unit_id
# is authoritative, so window/recorded predicates must bind that row —
# join ``lm`` against alias ``u`` (params: scope_id, generation, placed
# before the outer WHERE's own scope param).
_LATEST_UNITS = (
    "JOIN (SELECT unit_id, MAX(generation) AS mg FROM units"
    "      WHERE scope_id = ? AND generation <= ? GROUP BY unit_id) lm"
    "   ON lm.unit_id = u.unit_id AND lm.mg = u.generation"
)

_TERM_RE = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------------------------
# small pure helpers (exported-adjacent for tests; deterministic)
# ---------------------------------------------------------------------------


def _fold(text: str) -> str:
    """``norm/v2``-equivalent matching fold (NFKC + casefold + diacritic
    strip). Kept local so the lane never imports a concurrent worker's
    module; idempotent over already-folded input."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TERM_RE.findall(_fold(text or "")))


def _mid_us(start: Optional[int], end: Optional[int]) -> Optional[int]:
    """Anchor instant of an interval: midpoint when both bounds are known,
    else the single known bound, else None."""
    if start is not None and end is not None:
        return (start + end) // 2
    return start if start is not None else end


def _t_prox(t_us: Optional[int], ws: Optional[int], we: Optional[int]) -> Optional[float]:
    """V7-09.07 temporal proximity: 1 - |t - centre|/half_width, clamped.

    Absent (None) when the window has no computable centre/half-width or the
    unit carries no anchor instant — the feature is then simply missing,
    matching the FeatureVector rule that absent signals are never invented.
    """
    if t_us is None or ws is None or we is None:
        return None
    centre = (ws + we) / 2.0
    half = (we - ws) / 2.0
    if half <= 0:  # instant window
        return 1.0 if t_us == centre else 0.0
    return max(0.0, min(1.0, 1.0 - abs(t_us - centre) / half))


def _overlap_type(
    s: Optional[int], e: Optional[int], ws: Optional[int], we: Optional[int]
) -> str:
    """Classify how the unit interval sits against the window."""
    inf = float("inf")
    se = -inf if s is None else s
    ee = inf if e is None else e
    wse = -inf if ws is None else ws
    wee = inf if we is None else we
    if se >= wse and ee <= wee:
        return "point" if (s is not None and s == e) else "contained"
    if se <= wse and ee >= wee:
        return "contains"
    return "partial"


def _overlap_clauses(alias: str, ws: Optional[int], we: Optional[int]) -> tuple[str, list]:
    """SQL fragment for interval overlap: ``unit.end >= window.start`` and
    ``unit.start <= window.end`` with NULL bounds treated as open sides.
    Rows with BOTH bounds NULL carry no temporal information and are
    excluded separately by the caller's presence predicate."""
    a = f"{alias}." if alias else ""
    parts: list[str] = []
    params: list = []
    if we is not None:
        parts.append(f"({a}occurred_start_us IS NULL OR {a}occurred_start_us <= ?)")
        params.append(we)
    if ws is not None:
        parts.append(f"({a}occurred_end_us IS NULL OR {a}occurred_end_us >= ?)")
        params.append(ws)
    return (" AND ".join(parts) if parts else "1=1"), params


def _ph(n: int) -> str:
    return ",".join("?" * n)


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ? AND type IN ('table','view')",
        (name,),
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# context resolution
# ---------------------------------------------------------------------------


def _resolve_conn(ctx: LaneContextV7) -> Optional[sqlite3.Connection]:
    """Find the caller's pinned read snapshot. Lanes never open a new
    transaction (LaneV7 protocol), so a bare Store without a pinned conn
    resolves to None -> ``unavailable`` rather than a fresh snapshot."""
    conn = getattr(ctx, "conn", None)
    if isinstance(conn, sqlite3.Connection):
        return conn
    store = getattr(ctx, "store", None)
    if isinstance(store, sqlite3.Connection):
        return store
    conn = getattr(store, "conn", None)
    if isinstance(conn, sqlite3.Connection):
        return conn
    return None


def _eligibility(elig: Any) -> Optional[Callable[[dict], bool]]:
    """Normalize ``ctx.eligible`` into ``row -> bool``. Accepted forms per
    the frozen contract: ``callable(unit_row)`` (falling back to a unit_id
    argument on signature mismatch), objects with ``is_eligible(row)``, and
    set-like containers of unit_ids. Anything else — including a missing
    handle — returns None so the lane fails closed."""
    if elig is None:
        return None
    if callable(elig):

        def _call(row: dict) -> bool:
            try:
                return bool(elig(row))
            except TypeError:
                return bool(elig(row["unit_id"]))

        return _call
    if hasattr(elig, "is_eligible"):
        return lambda row: bool(elig.is_eligible(row))
    if hasattr(elig, "__contains__"):
        return lambda row: row["unit_id"] in elig
    return None


def _query_terms(qv: QueryViewV7) -> tuple[str, ...]:
    """Folded query terms (text + stem channels), deduped, ordered."""
    seen: list[str] = []
    for t in qv.norm.terms or ():
        if t.channel not in ("text", "stem"):
            continue
        term = _fold(t.term)
        if term and term not in seen:
            seen.append(term)
    return tuple(seen)


def _subject_canons(qv: QueryViewV7) -> frozenset:
    """Canons eligible to match ``events.subject_canon``: the query's
    entities plus the query speaker ("when did *I* move")."""
    cans = {_fold(c) for c in (qv.entity_canons or ()) if c}
    if qv.speaker_canon:
        cans.add(_fold(qv.speaker_canon))
    cans.discard("")
    return frozenset(cans)


# ---------------------------------------------------------------------------
# candidate accumulator
# ---------------------------------------------------------------------------


@dataclass
class _Cand:
    unit_id: str
    source_id: str = ""
    revision: int = 0
    rowid: int = 0
    tier: int = _TIER_OCCURRED
    t_us: Optional[int] = None  # anchor instant for t_prox / bucketing
    t_axis: str = "occurred"
    overlap: str = ""
    occurred_precision: str = "unknown"
    occurred_source: str = "unknown"
    # V8-09.08 — (start_us, end_us, precision) of the unit_time_mentions
    # interval driving temporal scoring; None when the unit carries none.
    mention: Optional[tuple] = None
    ev_score: float = 0.0
    event_ids: list = field(default_factory=list)
    subj_match: bool = False
    pred_match: bool = False
    polarity: Optional[str] = None
    fts_raw: Optional[float] = None
    ent_frac: float = 0.0
    t_prox: Optional[float] = None
    bucket: int = 0
    score: float = 0.0
    # V8-09.06 — set on fallback candidates: signed event time so the
    # lane's emitted order IS the temporal ordering (recency → negative
    # so newest sorts first; earliest → positive).
    order_time: Optional[int] = None

    def order_key(self) -> tuple:
        if self.order_time is not None:
            return (self.tier, self.order_time, self.unit_id)
        return (self.tier, -self.score, self.unit_id)


def _row_dict(row: tuple) -> dict:
    """Eligibility view of a unit row: the §30 minimum columns."""
    return {
        "unit_id": row[1],
        "source_id": row[2],
        "revision": row[3],
        "kind": row[4],
        "session_id": row[5],
        "speaker_canon": row[6],
        "recorded_at_us": row[7],
        "occurred_start_us": row[8],
        "occurred_end_us": row[9],
        "occurred_precision": row[10],
        "occurred_source": row[11],
    }


def _cand_from_row(row: tuple, tier: int) -> _Cand:
    return _Cand(
        unit_id=row[1],
        source_id=row[2] or "",
        revision=int(row[3] or 0),
        rowid=int(row[0]),
        tier=tier,
        occurred_precision=row[10] or "unknown",
        occurred_source=row[11] or "unknown",
    )


# ---------------------------------------------------------------------------
# scoring enrichment
# ---------------------------------------------------------------------------


def _fts_scores(
    conn: sqlite3.Connection, rowids: list[int], terms: tuple[str, ...]
) -> Optional[dict[int, float]]:
    """``unit_fts`` BM25 over the pooled rowids (lexical relevance inside
    the window). Returns None when the index or the match query is
    unavailable — the caller records the degradation in stats and the lane
    continues on entity/event signals only."""
    if not terms or not rowids or not _has_table(conn, "unit_fts"):
        return None
    match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms[:FTS_TERM_LIMIT])
    if not match:
        return None
    try:
        rows = conn.execute(
            "SELECT rowid, bm25(unit_fts) FROM unit_fts"
            " WHERE unit_fts MATCH ? ORDER BY bm25(unit_fts), rowid LIMIT ?",
            (match, FTS_ROW_LIMIT),
        ).fetchall()
    except sqlite3.Error:
        return None
    pooled = set(rowids)
    # FTS5 bm25: smaller (more negative) is better -> negate so higher wins.
    return {int(r): -float(b) for r, b in rows if int(r) in pooled}


def _entity_fracs(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    unit_ids: list[str],
    canons: frozenset,
) -> dict[str, float]:
    """Fraction of the query's entity canons each unit mentions
    (``entity_mentions`` postings — canonical, capitalization-free).

    V7-30.02: the postings are versioned — ``generation <= pinned``
    fences the snapshot (DISTINCT canon already absorbs repeat mentions
    of one canon across generations)."""
    if not canons or not unit_ids or not _has_table(conn, "entity_mentions"):
        return {}
    out: dict[str, float] = {}
    canon_list = sorted(canons)
    try:
        for i in range(0, len(unit_ids), IN_CHUNK):
            chunk = unit_ids[i : i + IN_CHUNK]
            rows = conn.execute(
                "SELECT unit_id, COUNT(DISTINCT canon) FROM entity_mentions"
                f" WHERE scope_id = ? AND generation <= ?"
                f" AND canon IN ({_ph(len(canon_list))})"
                f" AND unit_id IN ({_ph(len(chunk))}) GROUP BY unit_id",
                [scope_id, generation, *canon_list, *chunk],
            ).fetchall()
            for uid, cnt in rows:
                out[uid] = min(1.0, cnt / len(canon_list))
    except sqlite3.Error:
        return {}
    return out


# ---------------------------------------------------------------------------
# V8 helpers — events_weights arm, fallback relevance/order, mentions
# ---------------------------------------------------------------------------

_MISSING = object()


def _events_weights(ctx: LaneContextV7) -> tuple[float, float]:
    """``temporal.events_weights`` (V8-09.07, §23 arm) → ``(w_subj, w_pred)``.

    Resolution follows the wave convention: ``ctx.policy.params`` via
    ``policy_param`` first, ``ctx.manifest`` as the harness fallback, then
    the 0.5/0.5 prior. Accepted shapes: a two-number list/tuple or a
    ``{"subject": w, "predicate": w}`` mapping — anything else raises
    ``VALIDATION`` naming the arm."""
    raw: Any = _MISSING
    pol = getattr(ctx, "policy", None)
    try:
        from .policy import policy_param

        raw = policy_param(pol, EVENTS_WEIGHTS_KEY, _MISSING)
    except Exception:  # noqa: BLE001 — policy module absent → fallback
        params = getattr(pol, "params", None)
        if isinstance(params, dict):
            raw = params.get(EVENTS_WEIGHTS_KEY, _MISSING)
    if raw is _MISSING:
        manifest = getattr(ctx, "manifest", None) or {}
        raw = manifest.get(EVENTS_WEIGHTS_KEY, _MISSING)
    if raw is _MISSING or raw is None:
        return _EVENTS_WEIGHTS_DEFAULT
    bad = VerbatimError(
        ErrorCode.VALIDATION,
        f"{EVENTS_WEIGHTS_KEY} must be a [subject, predicate] pair of"
        " non-negative numbers or a {'subject','predicate'} mapping",
    )
    if isinstance(raw, dict):
        pair = (raw.get("subject"), raw.get("predicate"))
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        pair = (raw[0], raw[1])
    else:
        raise bad
    out: list[float] = []
    for w in pair:
        if (
            isinstance(w, bool)
            or not isinstance(w, (int, float))
            or not math.isfinite(float(w))
            or float(w) < 0.0
        ):
            raise bad
        out.append(float(w))
    return out[0], out[1]


#: §23/V85-05.06 arm ``temporal.as_of_scope`` — which rankings the
#: caller's ``as_of`` anchor may re-anchor. ``"window"`` (the V8.5
#: default): the anchor resolves relative-expression windows and feeds
#: the verdict only; recency and fallback ordering stay on wall time.
#: ``"global"``: the prior behavior — the anchor also drives the boost
#: recency clock (the global re-anchor measured −0.029 temporal any@10
#: in the b3 arm; kept selectable for paired ablation only).
AS_OF_SCOPE_KEY = "temporal.as_of_scope"
AS_OF_SCOPE_DEFAULT = "window"
AS_OF_SCOPES = frozenset({"window", "global"})


def resolve_as_of_scope(policy: Any) -> str:
    """Effective ``temporal.as_of_scope`` (V85-05.06).

    Absent → ``"window"`` (the §23 declaration-register prior). A
    declared value must be one of :data:`AS_OF_SCOPES`; a malformed arm
    is a policy defect and fails validation rather than silently
    reverting (V7-05.02's declared-table rule).

    The lane itself needs no branching: its window arrives pre-anchored
    on ``qv.intent.window`` and its fallback ordering is event-time
    based — the anchor never enters. The consumer of this arm is the
    pipeline's boost anchor: under ``"window"`` recency keeps wall time.
    """
    raw: Any = _MISSING
    try:
        from .policy import policy_param

        raw = policy_param(policy, AS_OF_SCOPE_KEY, _MISSING)
    except Exception:  # noqa: BLE001 — policy module absent → params map
        params = getattr(policy, "params", None)
        if isinstance(params, dict):
            raw = params.get(AS_OF_SCOPE_KEY, _MISSING)
    if raw is _MISSING or raw is None:
        return AS_OF_SCOPE_DEFAULT
    if not isinstance(raw, str) or raw not in AS_OF_SCOPES:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"{AS_OF_SCOPE_KEY} must be one of {sorted(AS_OF_SCOPES)},"
            f" got {raw!r}",
        )
    return raw


def _fallback_desc(term_set: frozenset, classes: frozenset) -> bool:
    """V8-09.06 scan direction: ``True`` (DESC — newest event time first)
    for recency-seeking intents/terms and as the temporal default;
    ``False`` (ASC) only when a first/earliest cue is nominated."""
    if term_set & _EARLIEST_TERMS:
        return False
    if term_set & _RECENCY_TERMS or IntentClass.CURRENT_VALUE in classes:
        return True
    return True


def _relevance_legs(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    terms: tuple[str, ...],
    canons: frozenset,
    speaker: Optional[str],
) -> tuple[list[str], list]:
    """V8-09.06 relevance restriction for the no-window fallback: SQL OR
    legs admitting only units that carry a nominated query content term
    (``unit_fts``) or a resolved entity/speaker canon
    (``entity_mentions`` / the unit's own ``speaker_canon``). Legs are
    added only for indices that actually exist; ``([], [])`` means the
    restriction cannot be applied and the caller must not scan."""
    legs: list[str] = []
    params: list = []
    if terms and _has_table(conn, "unit_fts"):
        match = " OR ".join(
            '"' + t.replace('"', '""') + '"' for t in terms[:FTS_TERM_LIMIT]
        )
        if match:
            legs.append(
                "u.rowid IN (SELECT rowid FROM unit_fts"
                " WHERE unit_fts MATCH ?)"
            )
            params.append(match)
    canon_list = sorted(canons)
    if canon_list and _has_table(conn, "entity_mentions"):
        legs.append(
            "u.unit_id IN (SELECT unit_id FROM entity_mentions"
            " WHERE scope_id = ? AND generation <= ?"
            f" AND canon IN ({_ph(len(canon_list))}))"
        )
        params.extend([scope_id, generation, *canon_list])
    if speaker:
        legs.append("u.speaker_canon = ?")
        params.append(speaker)
    return legs, params


_MENTION_LATEST = (
    "JOIN (SELECT unit_id, MAX(generation) AS mg FROM unit_time_mentions"
    "      WHERE scope_id = ? AND generation <= ? GROUP BY unit_id) lm"
    "   ON lm.unit_id = m.unit_id AND lm.mg = m.generation"
)


def _mention_overlap_clause(ws: Optional[int], we: Optional[int]) -> tuple[str, list]:
    """Interval overlap on the (non-NULL) mention bounds — same shape as
    ``_overlap_clauses`` but both bounds are NOT NULL by DDL."""
    parts: list[str] = []
    params: list = []
    if we is not None:
        parts.append("m.start_us <= ?")
        params.append(we)
    if ws is not None:
        parts.append("m.end_us >= ?")
        params.append(ws)
    return (" AND ".join(parts) if parts else "1=1"), params


def _mentions_for(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    unit_ids: list[str],
) -> dict[str, list]:
    """Latest-generation ``unit_time_mentions`` rows per pooled unit →
    ``{unit_id: [(ord, start_us, end_us, precision), ...]}`` sorted by
    ``ord``. Generation-fenced and scope-bound (V7-30.02 carried)."""
    out: dict[str, list] = {}
    if not unit_ids:
        return out
    try:
        for i in range(0, len(unit_ids), IN_CHUNK):
            chunk = unit_ids[i : i + IN_CHUNK]
            rows = conn.execute(
                "SELECT m.unit_id, m.ord, m.start_us, m.end_us, m.precision"
                " FROM unit_time_mentions m " + _MENTION_LATEST
                + f" WHERE m.scope_id = ? AND m.unit_id IN ({_ph(len(chunk))})"
                " ORDER BY m.unit_id, m.ord",
                [scope_id, generation, scope_id, *chunk],
            ).fetchall()
            for uid, o, s, e, prec in rows:
                out.setdefault(uid, []).append(
                    (int(o), int(s), int(e), str(prec))
                )
    except sqlite3.Error:
        return {}
    return out


def _scoring_mention(
    mentions: list, ws: Optional[int], we: Optional[int]
) -> tuple:
    """Pick the mention interval that drives temporal scoring (V8-09.08):
    the lowest-``ord`` mention overlapping the window; absent any overlap,
    the one nearest the window centre. Returns ``(start, end, precision)``."""
    centre = (ws + we) / 2.0 if (ws is not None and we is not None) else None

    def _key(m: tuple) -> tuple:
        _ord, s, e, _prec = m
        overlaps = (we is None or s <= we) and (ws is None or e >= ws)
        dist = (
            abs(_mid_us(s, e) - centre) if centre is not None else 0.0
        )
        return (0 if overlaps else 1, dist, _ord)

    _ord, s, e, prec = min(mentions, key=_key)
    return (s, e, prec)


# ---------------------------------------------------------------------------
# the lane
# ---------------------------------------------------------------------------


def lane_temporal(ctx: LaneContextV7, qv: QueryViewV7, slice: LaneSlice) -> LaneOutput:
    """L-time (V7-09.06–09): window-overlap retrieval with bucket spread,
    event-index first for answer-bearing temporal intents."""
    out = LaneOutput(lane=LANE_NAME, status=LaneStatus.OK)
    stats = out.stats
    stats["lane_version"] = LANE_VERSION
    stats["formula_status"] = "provisional/v7-r0"

    t0 = time.monotonic()
    deadline_s = max(0.0, float(slice.deadline_ms or 0.0)) / 1000.0

    def expired() -> bool:
        return (time.monotonic() - t0) > deadline_s

    conn = _resolve_conn(ctx)
    if conn is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "no_read_snapshot"
        return out
    if ctx.generation is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "generation_unpinned"
        return out
    if not _has_table(conn, "units"):
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "units_table_missing"
        return out
    eligible_fn = _eligibility(getattr(ctx, "eligible", None))
    if eligible_fn is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "eligibility_handle_missing"
        return out

    intent = qv.intent
    window = intent.window if intent is not None else None
    ws = window.start_us if window is not None else None
    we = window.end_us if window is not None else None
    has_window = ws is not None or we is not None
    stats["window"] = (
        {
            "start_us": ws,
            "end_us": we,
            "precision": getattr(window.precision, "value", window.precision),
            "source": getattr(window.source, "value", window.source),
            "rule_id": window.rule_id,
        }
        if has_window
        else None
    )

    classes = (
        (set(intent.classes or ()) | {intent.primary})
        if intent is not None
        else set()
    )
    classes.discard(None)
    events_first = bool(classes & _EVENTS_FIRST)
    temporal = bool(classes & _TEMPORAL_INTENTS)
    stats["events_first"] = events_first

    if not has_window and not temporal:
        out.status = LaneStatus.SKIPPED
        out.reason = "no_window"
        return out

    cap = max(0, int(slice.cap or 0))
    if cap == 0:
        stats["mode"] = "capped"
        return out
    if expired():
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
        stats["mode"] = "none"
        return out

    terms = _query_terms(qv)
    term_set = frozenset(terms)
    canons = _subject_canons(qv)
    # V8-09.07 — the §23 arm resolves once per call; a malformed value is
    # a loud VALIDATION naming the arm, never a silent prior.
    w_subj, w_pred = _events_weights(ctx)
    ev_max = w_subj + w_pred + _EV_OBJ_W
    stats["events_weights"] = [w_subj, w_pred]
    pool: dict[str, _Cand] = {}
    truncated = False
    deadline_hit = False
    missing_events = False
    missing_mentions = False
    modes: list[str] = []

    # -- phase 1: event index first (V7-09.09) ------------------------------
    if events_first:
        if not _has_table(conn, "events_v7"):
            missing_events = True
            stats["events_index"] = "missing"
        elif not canons and not term_set:
            stats["events_index"] = "no_match_keys"
        else:
            stats["events_index"] = "ok"
            modes.append("events")
            sql = [
                "SELECT event_id, unit_id, subject_canon, predicate_lemma,"
                " object_text, polarity, occurred_start_us, occurred_end_us,"
                " precision FROM events_v7 WHERE scope_id = ? AND generation <= ?"
            ]
            params: list = [ctx.scope_id, ctx.generation]
            if has_window:
                # both-NULL occurred carries no temporal information; it
                # cannot truthfully satisfy a window constraint.
                sql.append(
                    "AND (occurred_start_us IS NOT NULL"
                    " OR occurred_end_us IS NOT NULL)"
                )
                cl, pp = _overlap_clauses("", ws, we)
                sql.append(f"AND ({cl})")
                params.extend(pp)
            subj_list = sorted(canons)
            pred_list = sorted(term_set)
            # V8-09.07 — weighted disjunction, not the V7 conjunction: a
            # subject OR predicate match nominates the event row (the
            # window gate above keeps temporal compatibility strict);
            # the arm weights score each side in ``ev`` below.
            if subj_list and pred_list:
                sql.append(
                    f"AND (subject_canon IN ({_ph(len(subj_list))})"
                    f" OR predicate_lemma IN ({_ph(len(pred_list))}))"
                )
                params.extend(subj_list + pred_list)
            elif subj_list:
                sql.append(f"AND subject_canon IN ({_ph(len(subj_list))})")
                params.extend(subj_list)
            else:
                sql.append(f"AND predicate_lemma IN ({_ph(len(pred_list))})")
                params.extend(pred_list)
            sql.append("ORDER BY event_id")
            best: dict[str, dict] = {}
            n_ev = 0
            cur = conn.execute(" ".join(sql), params)
            for row in cur:
                n_ev += 1
                out.examined += 1
                (
                    event_id,
                    unit_id,
                    subj,
                    pred,
                    obj,
                    polarity,
                    es,
                    ee,
                    _prec,
                ) = row
                sm = subj is not None and _fold(subj) in canons
                pm = _fold(pred) in term_set
                obj_overlap = (
                    len(set(_tokens(obj)) & term_set) / len(term_set) if term_set else 0.0
                )
                ev = (
                    w_subj * sm
                    + w_pred * pm
                    + _EV_OBJ_W * min(1.0, obj_overlap)
                )
                prev = best.get(unit_id)
                if prev is None:
                    best[unit_id] = {
                        "ev": ev,
                        "ids": [event_id],
                        "subj": sm,
                        "pred": pm,
                        "polarity": polarity,
                        "t": _mid_us(es, ee),
                        "overlap": _overlap_type(es, ee, ws, we)
                        if has_window
                        else "event",
                    }
                else:
                    prev["ids"].append(event_id)
                    prev["subj"] = prev["subj"] or sm
                    prev["pred"] = prev["pred"] or pm
                    if ev > prev["ev"]:
                        prev["ev"] = ev
                        prev["polarity"] = polarity
                        prev["t"] = _mid_us(es, ee)
                        prev["overlap"] = (
                            _overlap_type(es, ee, ws, we) if has_window else "event"
                        )
                if n_ev % DEADLINE_CHECK_ROWS == 0 and expired():
                    deadline_hit = True
                    break
                if n_ev >= EVENT_ROW_LIMIT:
                    truncated = True
                    break
            stats["events_matched"] = len(best)
            # resolve event-backed units through the same fence + eligibility
            ev_units = [u for u in sorted(best) if u not in pool]
            for i in range(0, len(ev_units), IN_CHUNK):
                chunk = ev_units[i : i + IN_CHUNK]
                # generation <= pinned, newest row per unit_id first
                urows = conn.execute(
                    f"SELECT {_UNIT_COLS} FROM units u"
                    " WHERE u.scope_id = ? AND u.generation <= ?"
                    f" AND u.unit_id IN ({_ph(len(chunk))})"
                    " ORDER BY u.unit_id, u.generation DESC",
                    [ctx.scope_id, ctx.generation, *chunk],
                ).fetchall()
                seen_units: set = set()
                for ur in urows:
                    out.examined += 1
                    if ur[1] in seen_units:
                        continue  # superseded projection of this unit
                    seen_units.add(ur[1])
                    rd = _row_dict(ur)
                    if not eligible_fn(rd):
                        continue
                    info = best[ur[1]]
                    cand = _cand_from_row(ur, _TIER_EVENT)
                    cand.ev_score = info["ev"]
                    cand.event_ids = sorted(info["ids"])
                    cand.subj_match = info["subj"]
                    cand.pred_match = info["pred"]
                    cand.polarity = info["polarity"]
                    cand.t_us = info["t"]
                    cand.overlap = info["overlap"]
                    pool[cand.unit_id] = cand
                if expired():
                    deadline_hit = True
                    break

    # -- phase 2: occurred-overlap window scan (V7-09.06) --------------------
    if has_window and not deadline_hit:
        modes.append("window")
        cl, pp = _overlap_clauses("u", ws, we)
        sql = (
            f"SELECT {_UNIT_COLS} FROM units u {_LATEST_UNITS}"
            " WHERE u.scope_id = ?"
            # both bounds NULL = no temporal information; such rows can only
            # enter through the recorded axis or the event index.
            " AND (u.occurred_start_us IS NOT NULL"
            "      OR u.occurred_end_us IS NOT NULL)"
            f" AND ({cl})"
            " ORDER BY (u.occurred_start_us IS NULL),"
            "          u.occurred_start_us, u.unit_id"
        )
        n = 0
        cur = conn.execute(
            sql, [ctx.scope_id, ctx.generation, ctx.scope_id, *pp]
        )
        for row in cur:
            n += 1
            out.examined += 1
            rd = _row_dict(row)
            if eligible_fn(rd):
                cand = pool.get(row[1])
                if cand is None:
                    cand = _cand_from_row(row, _TIER_OCCURRED)
                    pool[cand.unit_id] = cand
                cand.t_us = _mid_us(row[8], row[9])
                cand.t_axis = "occurred"
                cand.overlap = cand.overlap or _overlap_type(row[8], row[9], ws, we)
            if n % DEADLINE_CHECK_ROWS == 0 and expired():
                deadline_hit = True
                break
            if n >= SCAN_ROW_LIMIT:
                truncated = True
                break

    # -- phase 2b: write-time mention union (V8-09.04) ----------------------
    # ``unit_time_mentions`` overlap supplements ``units.occurred_*``: a
    # unit whose text mentions a time inside the window matches even when
    # its own occurred interval does not (labeled ``overlap="mention"``).
    # The read is table-guarded, scope-bound, and generation-fenced to
    # the latest generation per unit — absent table degrades honestly.
    if has_window and not deadline_hit:
        if not _has_table(conn, MENTIONS_TABLE):
            missing_mentions = True
            stats["mentions"] = "absent"
        else:
            stats["mentions"] = "ok"
            modes.append("mentions")
            cl, pp = _mention_overlap_clause(ws, we)
            mrows = conn.execute(
                "SELECT m.unit_id, m.ord, m.start_us, m.end_us, m.precision"
                " FROM unit_time_mentions m " + _MENTION_LATEST
                + " WHERE m.scope_id = ? AND (" + cl + ")"
                " ORDER BY m.unit_id, m.ord LIMIT ?",
                [ctx.scope_id, ctx.generation, ctx.scope_id, *pp,
                 SCAN_ROW_LIMIT + 1],
            ).fetchall()
            if len(mrows) > SCAN_ROW_LIMIT:
                truncated = True
                mrows = mrows[:SCAN_ROW_LIMIT]
            mention_hit: dict[str, tuple] = {}
            for uid, _ord, s, e, prec in mrows:
                out.examined += 1
                if uid not in mention_hit:
                    mention_hit[uid] = (int(s), int(e), str(prec))
            stats["mentions_matched"] = len(mention_hit)
            new_units = sorted(u for u in mention_hit if u not in pool)
            for i in range(0, len(new_units), IN_CHUNK):
                chunk = new_units[i : i + IN_CHUNK]
                urows = conn.execute(
                    f"SELECT {_UNIT_COLS} FROM units u {_LATEST_UNITS}"
                    " WHERE u.scope_id = ?"
                    f" AND u.unit_id IN ({_ph(len(chunk))})"
                    " ORDER BY u.unit_id",
                    [ctx.scope_id, ctx.generation, ctx.scope_id, *chunk],
                ).fetchall()
                for ur in urows:
                    out.examined += 1
                    rd = _row_dict(ur)
                    if not eligible_fn(rd):
                        continue
                    cand = _cand_from_row(ur, _TIER_OCCURRED)
                    cand.mention = mention_hit[ur[1]]
                    cand.t_us = _mid_us(cand.mention[0], cand.mention[1])
                    cand.t_axis = "mention"
                    cand.overlap = "mention"
                    pool[cand.unit_id] = cand
                if expired():
                    deadline_hit = True
                    break

    # -- phase 3: secondary recorded-axis match ------------------------------
    # V8-09.06: ordered by event time (occurred, recorded as NULL/tie
    # fallback) — newest first for recency-seeking intents.
    if has_window and not deadline_hit:
        modes.append("recorded")
        parts = ["u.recorded_at_us IS NOT NULL"]
        rp: list = []
        if ws is not None:
            parts.append("u.recorded_at_us >= ?")
            rp.append(ws)
        if we is not None:
            parts.append("u.recorded_at_us <= ?")
            rp.append(we)
        rec_dir = "DESC" if _fallback_desc(term_set, classes) else "ASC"
        sql = (
            f"SELECT {_UNIT_COLS} FROM units u {_LATEST_UNITS}"
            " WHERE u.scope_id = ?"
            f" AND {' AND '.join(parts)}"
            " ORDER BY COALESCE(u.occurred_start_us, u.recorded_at_us)"
            f" {rec_dir}, u.recorded_at_us {rec_dir}, u.unit_id"
        )
        n = 0
        cur = conn.execute(
            sql, [ctx.scope_id, ctx.generation, ctx.scope_id, *rp]
        )
        for row in cur:
            n += 1
            out.examined += 1
            if row[1] in pool:
                continue  # occurred/event match already claims this unit
            rd = _row_dict(row)
            if eligible_fn(rd):
                cand = _cand_from_row(row, _TIER_RECORDED)
                cand.t_us = row[7]
                cand.t_axis = "recorded"
                cand.overlap = "recorded_only"
                pool[cand.unit_id] = cand
            if n % DEADLINE_CHECK_ROWS == 0 and expired():
                deadline_hit = True
                break
            if n >= SCAN_ROW_LIMIT:
                truncated = True
                break

    # -- phase 4: relevant-only fallback ordered by event time (V8-09.06) --
    # Temporal intent, no window, nothing yet: never a global ingest-order
    # sweep — only units carrying a nominated query content term or a
    # resolved entity/speaker canon, ordered by ``occurred_start_us``
    # (newest first for recency intents, oldest for first/earliest) with
    # ``recorded_at_us`` as the NULL and tie fallback, ``unit_id`` last.
    if not has_window and temporal and not pool and not deadline_hit:
        modes.append("relevant_fallback")
        direction = "DESC" if _fallback_desc(term_set, classes) else "ASC"
        stats["fallback_order"] = "desc" if direction == "DESC" else "asc"
        legs, lparams = _relevance_legs(
            conn,
            ctx.scope_id,
            ctx.generation,
            terms,
            canons,
            _fold(qv.speaker_canon) if qv.speaker_canon else None,
        )
        if not legs:
            # No evaluable relevance leg (no terms/canons, or the indices
            # are absent) — the scan must not run unrestricted.
            stats["fallback"] = "no_match_keys"
        else:
            stats["fallback"] = "ok"
            sql = (
                f"SELECT {_UNIT_COLS} FROM units u {_LATEST_UNITS}"
                " WHERE u.scope_id = ?"
                " AND COALESCE(u.occurred_start_us, u.recorded_at_us)"
                "     IS NOT NULL"
                f" AND ({' OR '.join(legs)})"
                " ORDER BY COALESCE(u.occurred_start_us, u.recorded_at_us)"
                f" {direction}, u.recorded_at_us {direction}, u.unit_id"
            )
            n = 0
            cur = conn.execute(
                sql,
                [ctx.scope_id, ctx.generation, ctx.scope_id, *lparams],
            )
            for row in cur:
                n += 1
                out.examined += 1
                rd = _row_dict(row)
                if eligible_fn(rd):
                    cand = _cand_from_row(row, _TIER_RECORDED)
                    cand.t_us = (
                        row[8] if row[8] is not None else row[7]
                    )
                    cand.t_axis = (
                        "occurred" if row[8] is not None else "recorded"
                    )
                    cand.overlap = "fallback"
                    # event-time emission order: recency → newest first
                    cand.order_time = (
                        -cand.t_us if direction == "DESC" else cand.t_us
                    )
                    pool[cand.unit_id] = cand
                if n % DEADLINE_CHECK_ROWS == 0 and expired():
                    deadline_hit = True
                    break
                if n >= SCAN_ROW_LIMIT:
                    truncated = True
                    break

    stats["mode"] = "+".join(modes) if modes else "none"
    out.eligible = len(pool)

    # -- enrichment: lexical (fts) + entity coverage --------------------------
    unit_ids = [c.unit_id for c in pool.values()]
    rowids = [c.rowid for c in pool.values()]
    fts = _fts_scores(conn, rowids, terms)
    stats["lexical"] = (
        "fts" if fts is not None else ("unavailable" if terms else "no_terms")
    )
    if fts:
        for cand in pool.values():
            if cand.rowid in fts:
                cand.fts_raw = fts[cand.rowid]
    ent = _entity_fracs(conn, ctx.scope_id, ctx.generation, unit_ids, canons)
    for cand in pool.values():
        if cand.unit_id in ent:
            cand.ent_frac = ent[cand.unit_id]

    # -- V8-09.08: event date beats session date --------------------------
    # A window-matched unit's ``unit_time_mentions`` interval is the
    # scoring interval (the event happened then); the unit's own
    # occurred/recorded instant is used only when it carries no mention.
    # Event-tier candidates keep their event's own occurred instant.
    if has_window and not missing_mentions and pool:
        mmap = _mentions_for(conn, ctx.scope_id, ctx.generation, unit_ids)
        stats["mentions_scored"] = len(mmap)
        for cand in pool.values():
            if cand.tier == _TIER_EVENT:
                continue
            ms = mmap.get(cand.unit_id)
            if not ms:
                continue
            cand.mention = _scoring_mention(ms, ws, we)
            cand.t_us = _mid_us(cand.mention[0], cand.mention[1])
            cand.t_axis = "mention"

    # -- per-candidate signals + relevance composite --------------------------
    fts_max = max((c.fts_raw for c in pool.values() if c.fts_raw), default=0.0)
    for cand in pool.values():
        cand.t_prox = _t_prox(cand.t_us, ws, we)
        fts_norm = (cand.fts_raw / fts_max) if (cand.fts_raw and fts_max > 0) else 0.0
        cand.score = (
            W_FTS * fts_norm
            + W_ENT * cand.ent_frac
            + W_EVENT * (cand.ev_score / ev_max if ev_max > 0 else 0.0)
            + W_TPROX * (cand.t_prox if cand.t_prox is not None else 0.0)
        )

    # -- bucket-spread selection (V7-09.06) -----------------------------------
    span_lo = span_hi = None
    if has_window:
        span_lo, span_hi = ws, we
    elif pool:
        ts = [c.t_us for c in pool.values() if c.t_us is not None]
        if ts:
            span_lo, span_hi = min(ts), max(ts)
    width = (span_hi - span_lo) if (span_lo is not None and span_hi is not None) else None
    # Bucket spread is a *window-representativeness* mechanism (V7-09.06):
    # it must never scatter the V8-09.06 fallback's event-time ordering.
    spread = bool(
        has_window and width and width > MONTH_US and len(pool) > cap
    )
    n_buckets = 1
    if spread:
        n_buckets = min(MAX_SPREAD_BUCKETS, max(MIN_SPREAD_BUCKETS, cap))
        n_buckets = min(n_buckets, len(pool))
    stats["buckets"] = n_buckets
    stats["bucket_spread"] = spread

    def _bucket(c: _Cand) -> int:
        if not spread or c.t_us is None or width <= 0:
            return 0
        idx = int((c.t_us - span_lo) * n_buckets / width)
        return max(0, min(n_buckets - 1, idx))

    if spread:
        per_bucket: dict[int, list[_Cand]] = {}
        for cand in pool.values():
            cand.bucket = _bucket(cand)
            per_bucket.setdefault(cand.bucket, []).append(cand)
        for members in per_bucket.values():
            members.sort(key=lambda c: c.order_key())
        k = -(-cap // n_buckets)  # ceil
        selected: list[_Cand] = []
        for b in sorted(per_bucket):
            selected.extend(per_bucket[b][:k])
        if len(selected) < cap:  # redistribute unused quota globally
            chosen = {c.unit_id for c in selected}
            rest = sorted(
                (c for c in pool.values() if c.unit_id not in chosen),
                key=lambda c: c.order_key(),
            )
            selected.extend(rest[: cap - len(selected)])
    else:
        selected = sorted(pool.values(), key=lambda c: c.order_key())[:cap]

    final = sorted(selected, key=lambda c: c.order_key())[:cap]
    stats["overflow"] = max(0, len(pool) - len(final))
    stats["scan_truncated"] = truncated
    if expired():
        deadline_hit = True  # enrichment/finalization overran the slice

    for rank, cand in enumerate(final, start=1):
        signals: dict[str, Any] = {
            "tier": cand.tier,
            "axis": cand.t_axis,
            "overlap": cand.overlap,
            "occurred_precision": cand.occurred_precision,
            "bucket": cand.bucket,
            "ent_match": round(cand.ent_frac, 6),
            "event_match": bool(cand.event_ids),
        }
        if cand.t_prox is not None:
            signals["t_prox"] = round(cand.t_prox, 6)
        if cand.mention is not None:
            # V8-09.04/09.08 — the exact (start_us, end_us, precision)
            # mention interval that scored/matched this unit.
            signals["mention_interval"] = tuple(cand.mention)
        if cand.event_ids:
            signals["event_ids"] = list(cand.event_ids)
            signals["event_pred"] = int(cand.pred_match)
            signals["subject_match"] = int(cand.subj_match)
            if cand.polarity is not None:
                signals["event_polarity"] = cand.polarity
        if cand.fts_raw is not None:
            signals["fts"] = round(cand.fts_raw, 6)
        out.candidates.append(
            CandidateV7(
                unit_id=cand.unit_id,
                source_id=cand.source_id,
                revision=cand.revision,
                lane=LANE_NAME,
                rank=rank,
                raw_score=cand.score,
                signals=signals,
            )
        )

    if deadline_hit:
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
    elif truncated:
        out.status = LaneStatus.PARTIAL
        out.reason = "scan_bound"
    elif missing_events or missing_mentions:
        # An absent optional table is an honest degradation: a surviving
        # path (window overlap or the relevant fallback) still produced
        # candidates → ``partial``; nothing ran → ``unavailable`` — never
        # an "ok" over a dead index, and never ``unavailable`` while
        # holding evidence the fallback legitimately retrieved.
        if has_window or out.candidates:
            out.status = LaneStatus.PARTIAL
            out.reason = (
                "events_table_missing"
                if missing_events
                else "mentions_table_missing"
            )
        else:
            out.status = LaneStatus.UNAVAILABLE
            out.reason = (
                "events_table_missing"
                if missing_events
                else "mentions_table_missing"
            )
            out.candidates.clear()
            out.eligible = 0
    return out


class TemporalLane(LaneV7):
    """LaneV7 protocol wrapper for registry wiring (V7-05.01)."""

    name = LANE_NAME

    def run(self, ctx: LaneContextV7, query: QueryViewV7, slice: LaneSlice) -> LaneOutput:
        return lane_temporal(ctx, query, slice)


__all__ = [
    "AS_OF_SCOPES",
    "AS_OF_SCOPE_DEFAULT",
    "AS_OF_SCOPE_KEY",
    "LANE_NAME",
    "LANE_VERSION",
    "MONTH_US",
    "MIN_SPREAD_BUCKETS",
    "MAX_SPREAD_BUCKETS",
    "TemporalLane",
    "lane_temporal",
    "resolve_as_of_scope",
]
