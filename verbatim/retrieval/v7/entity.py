"""V7 entity lane — exact-match canonical-entity postings (V7-08, §32.7).

L-ent (§04.2): the canonical-entity postings lane.  ``qv.entity_canons``
— the S1 entity pass output — is re-folded through ``entities_v2.canon``
(idempotent; a caller that hands over raw surfaces like ``"Caroline's"``
lands on the same postings, closing D7-01/D7-02 on the query side), then
expanded through **active** ``entity_aliases_v7`` rows only — ``candidate``
and ``rejected`` rows never widen retrieval, so the A6 same-name conflict
abstention is honored by construction (V7-08.03/04/13).  Expansion state
is resolved at the caller's generation: the *latest* row per
``(canon, alias_canon)`` pair at/below the fence decides — an ``active``
row superseded by a ``candidate`` rewrite does not expand.

Postings: one covering scan of ``entity_mentions`` —
``scope_id=? AND canon IN (expanded) AND generation<=?`` — joined to
``units`` for ``source_id``/``revision`` and the eligibility row in the
same query.  ``units`` PK is ``(unit_id, generation)``, so the join is
``u.generation <= ctx.generation`` and each unit resolves to its latest
visible projection (V7-30.02 rebuild-coexistence rule, same convention as
the graph lane).  A mention whose unit has no visible row at/below the
fence is an orphan — counted, never emitted.

Scoring (``provisional/v7-r0``, V7-08.06): each matched canon contributes
``entities_v2.entity_weight`` — BM25-form IDF of ``entity_canon.df_units``
against the corpus unit count with the dominant-canon cap (a canon in
> 30% of units contributes at most 0.1 of a rare canon's weight — D7-07).
The denominator is the maintained ``lex_stats.n_units`` aggregate when
present, else an exact ``COUNT(DISTINCT unit_id)`` over fenced units.
A canon whose recorded df is zero or absent carries **no measured rarity
signal**: it is floored to ``df = N`` (the formula's own minimum — an
unmeasured canon is treated as ubiquitous rather than fabricated rare), so
it still matches and emits, just weights low.  A unit covering ≥ 2
distinct *query* canons (pre-expansion — the all-canons-present signal
Hindsight's entity matching exploits) adds ``ln(1 + n_extra)`` on top.

Conjunctive emission (V8-10.03, scenario K61): when the query resolves
≥ 2 distinct canons, units covering EVERY query canon — coverage runs
through the same expanded set, so a canon and its alias still count as
one entity, the multi-canon boost's query-canon accounting — are marked
``signals["joint"] = True`` and ranked above every partial-coverage
match.  The joint set is derived from the same fenced scan and the same
eligible pool, so eligibility and the generation fence apply to
conjunctive results identically by construction — no second postings
round-trip is needed (the spec's ``GROUP BY … HAVING`` sketch is the
semantics, not a required second query).

Per-canon quota (V8-10.04, §23 ``ent.per_canon_quota``, prior 50,
scenario K62): each resolved query canon contributes at most ``quota``
non-joint emissions — a unit charges every query canon it covers and
emits only while all of them have room, so a canon with df ≈ 10³ cannot
flood the lane's candidate cap and starve the rarer canons.  Joint
units are exempt (they are the point of the conjunctive phase).  The
arm resolves off the policy object (a params-style mapping keyed by the
dotted arm name, or a ``per_canon_quota`` attribute), then the
request-scoped ``ctx.manifest`` channel; absent everywhere → 50, ``0``
disables single-canon emission (joints still emit), and a mistyped
value raises VALIDATION — the ``graph_max_hops`` loud-fail convention.

Honesty rules honored (V7-04.03, LaneV7 protocol):

- eligibility is evaluated per unit *before* candidate emission; a held or
  otherwise ineligible unit is counted in ``stats["ineligible"]`` and
  never emitted;
- ``qv.entity_canons`` empty → ``skipped`` / ``"no_query_entities"``;
- missing ``entity_mentions``/``units`` (pre-migration store) →
  ``unavailable`` / ``"no_entity_tables"`` — probed via ``sqlite_master``,
  never caught-and-fabricated;
- a deadline-cut postings scan reports ``partial`` / ``"deadline"`` with
  honest ``examined``/``eligible`` counts; the declared row bound reports
  ``partial`` / ``"scan_bound"``;
- ``slice.cap`` truncation records ``stats["cap_truncated"]``.

Signals per candidate (rerank inputs): ``entity_idf`` (summed canon
weights before boost), ``matched_canons`` (expanded canons that matched),
``query_canons_covered``, ``role`` (strongest matched mention role),
``pinned`` (any matched mention carries a real byte span — speaker-role
metadata mentions pin at ``(0,0)`` and never set it), ``alias_matched``,
and ``multi_canon_boost`` when applied.  Conjunctive-phase candidates
add ``joint = True`` (V8-10.03); non-joint candidates carry no key.

The lane is stdlib + sqlite3 only; it consumes the caller's pinned read
snapshot and never opens a transaction.
"""

from __future__ import annotations

import math
import sqlite3
import time
from collections.abc import Mapping
from typing import Any, Iterable, Optional

from ...core.types import ErrorCode, VerbatimError
from ...core.types_v7 import (
    FORMULA_STATUS_PROVISIONAL,
    AliasMethod,
    AliasRow,
    AliasState,
    CandidateV7,
    LaneContextV7,
    LaneName,
    LaneOutput,
    LaneSlice,
    LaneStatus,
    LaneV7,
    QueryViewV7,
)
from ...enrichment.entities_v2 import (
    DEFAULT_EXPANSION_LIMIT,
    canon,
    entity_weight,
    expand_query,
)
from ...storage.repos import has_table

LANE_NAME = LaneName.ENT.value  # "ent" — stable coverage key
LANE_VERSION = "entity_lane/v7-r0"
FORMULA_STATUS = FORMULA_STATUS_PROVISIONAL

MENTIONS_TABLE = "entity_mentions"
CANON_TABLE = "entity_canon"
ALIAS_TABLE = "entity_aliases_v7"
UNITS_TABLE = "units"
STATS_TABLE = "lex_stats"

_PAGE = 1024                  # fetchmany page; the deadline is checked per page
_POSTINGS_ROW_LIMIT = 100_000  # declared scan bound → partial/scan_bound
_IN_CHUNK = 400               # < SQLITE_MAX_VARIABLE_NUMBER
_MAX_SPAN_SIGNALS = 8         # bounded span echo in signals

#: V8-10.04 arm — the §23 ``ent.per_canon_quota`` name (dotted form is
#: the cross-worker contract key) and its §23 prior.  Resolution order:
#: a params-style mapping on the policy object (the load_policy arm
#: plumbing), a direct ``per_canon_quota`` policy attribute, then
#: ``ctx.manifest`` (the ``graph_max_hops`` request channel); absent
#: everywhere → :data:`PER_CANON_QUOTA_DEFAULT`.
PER_CANON_QUOTA_ARM = "ent.per_canon_quota"
PER_CANON_QUOTA_DEFAULT = 50

#: Role precedence for the ``role`` signal — the strongest matched mention
#: role is reported.  Provisional ordering: event-attributed roles
#: (subject/object) outrank the identity metadata role (speaker), which
#: outranks a bare text mention.
_ROLE_RANK = {"subject": 3, "object": 2, "speaker": 1, "mention": 0}


def _monotonic() -> float:
    """Indirection so tests drive the cooperative clock deterministically."""
    return time.monotonic()


class _Deadline:
    """Relative per-lane slice budget (LaneSlice.deadline_ms)."""

    __slots__ = ("_end",)

    def __init__(self, deadline_ms: Optional[float]) -> None:
        if deadline_ms is None:
            self._end = None
            return
        try:
            ms = float(deadline_ms)
        except (TypeError, ValueError):
            ms = 0.0
        self._end = (
            None
            if math.isinf(ms)
            else _monotonic() + max(ms, 0.0) / 1000.0
        )

    def expired(self) -> bool:
        return self._end is not None and _monotonic() >= self._end


def _chunks(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _conn(ctx: LaneContextV7) -> Optional[sqlite3.Connection]:
    """Resolve the caller-pinned read connection.  Order: an explicit
    ``ctx.conn`` (pipeline-pinned snapshot), then ``ctx.store`` itself when
    it is already a connection, then ``ctx.store.conn``, then the Store's
    thread-local reader.  The lane never opens a transaction of its own
    (LaneV7 protocol, V7-05.06)."""
    conn = getattr(ctx, "conn", None)
    if isinstance(conn, sqlite3.Connection):
        return conn
    store = getattr(ctx, "store", None)
    if isinstance(store, sqlite3.Connection):
        return store
    conn = getattr(store, "conn", None)
    if isinstance(conn, sqlite3.Connection):
        return conn
    reader = getattr(store, "_reader", None)
    if callable(reader):
        try:
            conn = reader()
        except Exception:
            return None
        if isinstance(conn, sqlite3.Connection):
            return conn
    return None


def _eligibility(ctx: LaneContextV7):
    """Normalize ``ctx.eligible`` into ``row -> bool``.

    Contract forms (types_v7): ``callable(unit_row)`` (falling back to a
    bare unit_id argument on signature mismatch), objects exposing
    ``eligible(row)``/``is_eligible(row)``/``contains(unit_id)``, and
    containers of unit_ids.  ``None`` means the caller declared no
    restriction (allow-all).  Anything unrecognized — and any predicate
    that raises — fails CLOSED: eligibility-before-rank is the security
    property (V7-05.08)."""
    elig = getattr(ctx, "eligible", None)
    if elig is None:
        return lambda row: True
    if callable(elig):

        def _call(row: Mapping[str, Any]) -> bool:
            try:
                return bool(elig(row))
            except TypeError:
                return bool(elig(row.get("unit_id")))

        fn = _call
    elif callable(getattr(elig, "is_eligible", None)):
        fn = lambda row: bool(elig.is_eligible(row))
    elif callable(getattr(elig, "eligible", None)):
        fn = lambda row: bool(elig.eligible(row))
    elif getattr(elig, "unit_ids", None) is not None:
        ids = elig.unit_ids
        fn = lambda row: row.get("unit_id") in ids
    elif callable(getattr(elig, "contains", None)):
        contains = elig.contains
        fn = lambda row: bool(contains(row.get("unit_id")))
    elif hasattr(elig, "__contains__"):
        fn = lambda row: row.get("unit_id") in elig
    else:
        return lambda row: False

    def guarded(row: Mapping[str, Any]) -> bool:
        try:
            return bool(fn(row))
        except Exception:
            return False

    return guarded


# ---------------------------------------------------------------------------
# alias expansion (V7-08.04, §32.7 alias/v1)
# ---------------------------------------------------------------------------


def _load_alias_rows(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    qcanons: list,
    stats: dict,
) -> list:
    """``AliasRow`` list for expansion — latest visible state per
    ``(canon, alias_canon)`` pair at/below the fence.

    Rows are fetched in *all* states (not pre-filtered to ``active``): the
    PK ends in ``generation``, so rebuild coexistence can surface several
    states for one pair, and only the newest row at/below the snapshot is
    authoritative — a pair rewritten ``candidate`` at a later generation
    must not keep expanding through its stale ``active`` predecessor.
    Only pairs whose latest visible state is ``active`` are returned;
    ``candidate``/``rejected`` never widen retrieval (A6 abstention).
    """
    if not has_table(conn, ALIAS_TABLE):
        stats["aliases"] = "table_absent"
        return []
    latest: dict[tuple, tuple] = {}
    for chunk in _chunks(list(qcanons), _IN_CHUNK):
        ph = ",".join("?" * len(chunk))
        try:
            rows = conn.execute(
                f"SELECT canon, alias_canon, rule_id, evidence_count,"
                f" method, state, generation FROM {ALIAS_TABLE}"
                f" WHERE scope_id = ? AND generation <= ?"
                f" AND (canon IN ({ph}) OR alias_canon IN ({ph}))"
                f" ORDER BY generation",
                [scope_id, generation, *chunk, *chunk],
            ).fetchall()
        except sqlite3.Error:
            stats["aliases"] = "query_error"
            return []
        for cn, al, rule, ev, method, state, gen in rows:
            key = (str(cn), str(al))
            cur = latest.get(key)
            if cur is None or int(gen) >= cur[6]:
                latest[key] = (cn, al, rule, ev, method, state, int(gen))
    out: list[AliasRow] = []
    for cn, al, rule, ev, method, state, gen in sorted(latest.values()):
        if state != AliasState.ACTIVE.value:
            continue
        try:
            m = AliasMethod(method) if method else AliasMethod.RULE
        except ValueError:
            m = AliasMethod.RULE
        out.append(
            AliasRow(
                scope_id=str(scope_id),
                canon=str(cn),
                alias_canon=str(al),
                rule_id=str(rule or ""),
                evidence_count=int(ev or 0),
                method=m,
                state=AliasState.ACTIVE,
                generation=int(gen),
            )
        )
    stats["aliases"] = "ok"
    stats["alias_rows_active"] = len(out)
    return out


def _expand(qcanons: list, alias_rows: list, stats: dict) -> tuple:
    """``(expanded_of, all_canons, canon_to_q)`` — per-query-canon bounded
    expansion via the pinned ``entities_v2.expand_query`` (≤ 8 expansions
    per canon, V7-08.04), the deduped probe set, and the reverse map
    expanded-canon → covered query canons (the multi-canon boost counts
    *query* canons, so an alias and its canonical matching together never
    double as two entities)."""
    expanded_of: dict[str, list] = {}
    canon_to_q: dict[str, set] = {}
    for qc in qcanons:
        ex = expand_query([qc], alias_rows, limit=DEFAULT_EXPANSION_LIMIT)
        # expand_query returns [qc, *sorted aliases ≤ limit] — first is self.
        expanded_of[qc] = [c for c in ex if c != qc]
        for c in ex:
            canon_to_q.setdefault(c, set()).add(qc)
    all_canons = sorted(canon_to_q)
    stats["query_canons"] = len(qcanons)
    stats["expanded_canons"] = len(all_canons)
    stats["alias_links_used"] = sum(len(v) for v in expanded_of.values())
    return expanded_of, all_canons, canon_to_q


# ---------------------------------------------------------------------------
# corpus statistics (V7-08.06)
# ---------------------------------------------------------------------------


def _df_map(
    conn: sqlite3.Connection, scope_id: str, generation: int, canons: list
) -> tuple:
    """``(canon -> recorded df_units, source)`` — latest ``entity_canon``
    row per canon at/below the fence (PK ends in generation; the ascending
    scan lets the newest row win).  Absent table → ``{}`` and ``"absent"``:
    every canon then prices at the floor, never a fabricated rarity."""
    out: dict[str, int] = {}
    if not has_table(conn, CANON_TABLE):
        return out, "absent"
    for chunk in _chunks(list(canons), _IN_CHUNK):
        ph = ",".join("?" * len(chunk))
        try:
            rows = conn.execute(
                f"SELECT canon, df_units FROM {CANON_TABLE}"
                f" WHERE scope_id = ? AND generation <= ?"
                f" AND canon IN ({ph}) ORDER BY generation",
                [scope_id, generation, *chunk],
            ).fetchall()
        except sqlite3.Error:
            return out, "query_error"
        for c, df in rows:
            out[str(c)] = int(df or 0)
    return out, "ok"


def _corpus_units(
    conn: sqlite3.Connection, scope_id: str, generation: int
) -> tuple:
    """``(n_units, source)`` — the IDF denominator: maintained
    ``lex_stats.n_units`` when present (max across field/stats_version rows
    at/below the fence — n_units is the corpus count, identical across
    fields in a consistent build), else an exact fenced recount."""
    if has_table(conn, STATS_TABLE):
        try:
            row = conn.execute(
                f"SELECT MAX(n_units) FROM {STATS_TABLE}"
                " WHERE scope_id = ? AND generation <= ?",
                (scope_id, generation),
            ).fetchone()
        except sqlite3.Error:
            row = None
        if row is not None and row[0] is not None:
            return int(row[0]), "lex_stats"
    try:
        row = conn.execute(
            f"SELECT COUNT(DISTINCT unit_id) FROM {UNITS_TABLE}"
            " WHERE scope_id = ? AND generation <= ?",
            (scope_id, generation),
        ).fetchone()
    except sqlite3.Error:
        row = None
    return int(row[0] or 0) if row else 0, "units_fallback"


# ---------------------------------------------------------------------------
# query-speaker resolution (V75-03.05) — shared with the S4 reranker
# ---------------------------------------------------------------------------


def lane_conn(ctx: LaneContextV7) -> Optional[sqlite3.Connection]:
    """Public handle on the lane's pinned-read-connection resolution —
    same lookup order as :func:`_conn`. Sibling v7 modules (the S4
    feature reranker's providers) reuse it instead of re-deriving conn
    resolution; never opens a transaction."""
    return _conn(ctx)


def resolve_query_speaker_canons(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    canons: Iterable,
) -> list:
    """Which of ``canons`` are speaker canons present in scope (V75-03.05).

    Each candidate is re-folded through ``entities_v2.canon`` — the same
    idempotent prep the lane applies to ``qv.entity_canons`` — then probed
    against ``units.speaker_canon`` at/below the caller's generation
    fence, the same column the ``speaker_match`` feature compares a
    unit's speaker against. Output is sorted and deduplicated.

    Honest negatives: absent ``units`` table (pre-migration store) or any
    SQL error yields ``[]`` — a partial probe could read as a single
    resolution, and a wrong single speaker is worse than none.
    """
    folded: list[str] = []
    for c in canons or ():
        cc = canon(str(c))
        if cc and cc not in folded:
            folded.append(cc)
    if not folded or not has_table(conn, UNITS_TABLE):
        return []
    out: list[str] = []
    for chunk in _chunks(folded, _IN_CHUNK):
        ph = ",".join("?" * len(chunk))
        try:
            rows = conn.execute(
                f"SELECT DISTINCT speaker_canon FROM {UNITS_TABLE}"
                " WHERE scope_id = ? AND generation <= ?"
                f" AND speaker_canon IN ({ph})",
                [scope_id, generation, *chunk],
            ).fetchall()
        except sqlite3.Error:
            return []
        out.extend(str(r[0]) for r in rows if r[0])
    return sorted(set(out))


# ---------------------------------------------------------------------------
# V8-10.04 arm resolution — ent.per_canon_quota (§23, prior 50)
# ---------------------------------------------------------------------------


def _per_canon_quota(ctx: LaneContextV7) -> int:
    """Effective per-canon cap on non-joint emissions (V8-10.04).

    §23 query-time arms travel on the policy object the pipeline threads
    (the ``load_policy``/``policy_overrides`` plumbing): the resolver
    accepts a params-style mapping attribute (``params``/``arms``/
    ``overrides``) keyed by the dotted arm name or its leaf, or a direct
    ``per_canon_quota`` attribute.  The request-scoped ``ctx.manifest``
    channel (the ``graph_max_hops`` convention) is the fallback carrier.
    Absent everywhere → the §23 prior :data:`PER_CANON_QUOTA_DEFAULT`.
    ``0`` disables single-canon emission entirely — joint units still
    emit.  Bools, strings, non-integral or negative values raise
    ``VerbatimError(VALIDATION)``: a mistyped arm must fail loudly,
    never silently reconfigure the lane.
    """
    leaf = PER_CANON_QUOTA_ARM.split(".", 1)[1]
    raw: Any = None
    seen = False
    holders: list = []
    pol = getattr(ctx, "policy", None)
    if pol is not None:
        holders.append(pol)
    manifest = getattr(ctx, "manifest", None)
    if isinstance(manifest, Mapping):
        holders.append(manifest)
    for holder in holders:
        if isinstance(holder, Mapping):
            if PER_CANON_QUOTA_ARM in holder:
                raw, seen = holder[PER_CANON_QUOTA_ARM], True
            elif leaf in holder:
                raw, seen = holder[leaf], True
            if seen:
                break
            continue
        for attr in ("params", "arms", "overrides"):
            m = getattr(holder, attr, None)
            if isinstance(m, Mapping):
                if PER_CANON_QUOTA_ARM in m:
                    raw, seen = m[PER_CANON_QUOTA_ARM], True
                    break
                if leaf in m:
                    raw, seen = m[leaf], True
                    break
        if seen:
            break
        val = getattr(holder, leaf, None)
        if val is not None:
            raw, seen = val, True
            break
    if not seen or raw is None:
        return PER_CANON_QUOTA_DEFAULT

    def _bad() -> VerbatimError:
        return VerbatimError(
            ErrorCode.VALIDATION,
            f"{PER_CANON_QUOTA_ARM} must be null or a non-negative"
            f" integer, got {raw!r}",
        )

    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise _bad()
    if isinstance(raw, float) and (
        math.isnan(raw) or not raw.is_integer()
    ):
        raise _bad()
    value = int(raw)
    if value < 0:
        raise _bad()
    return value


# ---------------------------------------------------------------------------
# the lane
# ---------------------------------------------------------------------------


def lane_entity(
    ctx: LaneContextV7, query: QueryViewV7, slice: LaneSlice
) -> LaneOutput:
    """L-ent: canonical-entity postings probe with bounded alias
    expansion, eligibility gated inside candidate production."""
    out = LaneOutput(lane=LANE_NAME, status=LaneStatus.OK)
    stats = out.stats
    stats["lane_version"] = LANE_VERSION
    stats["formula_status"] = FORMULA_STATUS

    deadline = _Deadline(getattr(slice, "deadline_ms", None))

    conn = _conn(ctx)
    if conn is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "no_read_snapshot"
        return out
    generation = getattr(ctx, "generation", None)
    if generation is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "generation_unpinned"
        return out
    if not has_table(conn, MENTIONS_TABLE) or not has_table(conn, UNITS_TABLE):
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "no_entity_tables"
        return out

    # Query canons — re-folded through canon() so a raw surface arriving in
    # entity_canons (e.g. "Caroline's") lands on the same postings (the
    # fold is idempotent on already-canonical keys; D7-01/D7-02).
    qcanons: list[str] = []
    for c in getattr(query, "entity_canons", ()) or ():
        cc = canon(c)
        if cc and cc not in qcanons:
            qcanons.append(cc)
    if not qcanons:
        out.status = LaneStatus.SKIPPED
        out.reason = "no_query_entities"
        return out

    if deadline.expired():
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
        stats["deadline"] = True
        return out

    eligible_fn = _eligibility(ctx)
    quota = _per_canon_quota(ctx)  # V8-10.04 — VALIDATION on a mistyped arm
    stats["per_canon_quota"] = quota

    # -- alias expansion -----------------------------------------------------
    alias_rows = _load_alias_rows(conn, ctx.scope_id, generation, qcanons, stats)
    expanded_of, all_canons, canon_to_q = _expand(qcanons, alias_rows, stats)
    stats["expansion"] = {qc: list(v) for qc, v in expanded_of.items() if v}

    # -- corpus statistics ---------------------------------------------------
    df_map, stats["df_source"] = _df_map(
        conn, ctx.scope_id, generation, all_canons
    )
    n_units, stats["n_units_source"] = _corpus_units(
        conn, ctx.scope_id, generation
    )
    stats["n_units"] = n_units
    stats["df"] = {c: df_map.get(c, 0) for c in all_canons}
    # df = 0 / absent → no measured rarity: floor at df = N (the IDF
    # minimum), so an unmeasured canon still matches but weights low.
    df_eff = {c: (df_map.get(c, 0) or n_units) for c in all_canons}
    stats["df_floor_canons"] = sum(1 for c in all_canons if not df_map.get(c))

    # -- postings scan: one covering query, units joined for the §30 row ----
    ph = ",".join("?" * len(all_canons))
    sql = (
        "SELECT m.unit_id, m.canon, m.role, m.byte_start, m.byte_end,"
        " m.generation,"
        " u.source_id, u.revision, u.scope_id, u.kind, u.parent_unit_id,"
        " u.session_id, u.seq, u.speaker_canon, u.perspective,"
        " u.recorded_at_us, u.occurred_start_us, u.occurred_end_us,"
        " u.occurred_precision, u.occurred_source, u.generation"
        f" FROM {MENTIONS_TABLE} m"
        f" LEFT JOIN {UNITS_TABLE} u"
        "   ON u.unit_id = m.unit_id AND u.generation <= ?"
        f" WHERE m.scope_id = ? AND m.generation <= ? AND m.canon IN ({ph})"
        " ORDER BY m.canon, m.unit_id, m.byte_start, m.generation,"
        "          u.generation"
    )
    params = [generation, ctx.scope_id, generation, *all_canons]

    # unit_id -> accumulator; ``row``/``ugen`` hold the unit's latest
    # visible projection (rebuild coexistence — V7-30.02).
    acc: dict[str, dict] = {}
    deadline_hit = False
    truncated = False
    try:
        cur = conn.execute(sql, params)
    except sqlite3.Error:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "postings_query_failed"
        return out
    while True:
        if deadline.expired():
            deadline_hit = True
            break
        batch = cur.fetchmany(_PAGE)
        if not batch:
            break
        for row in batch:
            out.examined += 1
            if out.examined > _POSTINGS_ROW_LIMIT:
                truncated = True
                break
            uid = row[0]
            if uid is None:
                continue  # NOT NULL column — defensive
            entry = acc.get(uid)
            if entry is None:
                entry = acc[uid] = {
                    "row": None,
                    "ugen": -1,
                    "canons": set(),
                    "roles": set(),
                    "pinned": False,
                    "spans": [],
                }
            entry["canons"].add(str(row[1]))
            if row[2]:
                entry["roles"].add(str(row[2]))
            bs, be = row[3], row[4]
            if bs is not None and be is not None and be > bs:
                entry["pinned"] = True
                if len(entry["spans"]) < _MAX_SPAN_SIGNALS:
                    entry["spans"].append(
                        {
                            "canon": str(row[1]),
                            "byte_start": int(bs),
                            "byte_end": int(be),
                        }
                    )
            if row[6] is None:
                continue  # no visible units row ≤ fence on this row
            ugen = int(row[20])
            if ugen > entry["ugen"]:
                entry["ugen"] = ugen
                entry["row"] = {
                    "unit_id": uid,
                    "source_id": row[6],
                    "revision": row[7],
                    "scope_id": row[8],
                    "kind": row[9],
                    "parent_unit_id": row[10],
                    "session_id": row[11],
                    "seq": row[12],
                    "speaker_canon": row[13],
                    "perspective": row[14],
                    "recorded_at_us": row[15],
                    "occurred_start_us": row[16],
                    "occurred_end_us": row[17],
                    "occurred_precision": row[18],
                    "occurred_source": row[19],
                    "generation": ugen,
                }
        if truncated:
            break
    cur.close()

    # -- eligibility gate (per unit, on its latest visible row) --------------
    pool: dict[str, dict] = {}
    orphaned = 0
    ineligible = 0
    for uid in sorted(acc):
        entry = acc[uid]
        urow = entry["row"]
        if urow is None:
            # LEFT JOIN produced only NULL rows — no units projection is
            # visible at/below the fence (purged or not yet built).
            orphaned += 1
            continue
        if urow.get("scope_id") != ctx.scope_id:
            # scope-keyed postings pointing at a foreign unit — the row is
            # treated as ineligible rather than trusted.
            ineligible += 1
            continue
        if eligible_fn(urow):
            pool[uid] = entry
        else:
            ineligible += 1
    stats["orphaned_mentions"] = orphaned
    stats["ineligible"] = ineligible
    out.eligible = len(pool)

    # -- scoring + ranking -----------------------------------------------------
    # scores[uid] = (raw, base_idf, boost, covered_qcanons, matched_canons)
    scores: dict[str, tuple] = {}
    for uid, entry in pool.items():
        matched = sorted(entry["canons"])
        base = sum(entity_weight(c, df_eff, n_units) for c in matched)
        covered = {q for c in matched for q in canon_to_q.get(c, ())}
        # all-canons-present signal: ≥ 2 distinct *query* canons covered.
        boost = math.log(len(covered)) if len(covered) >= 2 else 0.0
        scores[uid] = (base + boost, base, boost, covered, matched)

    # V8-10.03 — conjunctive phase: with ≥ 2 resolved query canons, units
    # covering ALL of them rank above every partial-coverage match and
    # carry signals["joint"]=True.  The joint set comes out of the same
    # fenced scan and the same eligible pool, so eligibility and the
    # generation fence apply to it identically by construction.
    n_q = len(qcanons)
    joint_of = {u: (n_q >= 2 and len(scores[u][3]) == n_q) for u in pool}
    order = sorted(
        pool, key=lambda u: (0 if joint_of[u] else 1, -scores[u][0], u)
    )
    stats["joint_pool"] = sum(1 for v in joint_of.values() if v)

    # V8-10.04 — per-canon quota: a non-joint unit charges EVERY query
    # canon it covers and emits only while all of them are below the
    # quota, so each canon contributes at most ``quota`` singles and a
    # prolific entity cannot flood the lane cap.  Joint units are exempt.
    counts = {qc: 0 for qc in qcanons}
    per_canon = {qc: {"emitted": 0, "dropped": 0} for qc in qcanons}
    kept: list[str] = []
    quota_dropped = 0
    for uid in order:
        if joint_of[uid]:
            kept.append(uid)
            continue
        covered = scores[uid][3]
        blocked = [q for q in sorted(covered) if counts[q] >= quota]
        if blocked:
            quota_dropped += 1
            for q in blocked:  # charged to the saturated canon(s)
                per_canon[q]["dropped"] += 1
            continue
        for q in covered:
            counts[q] += 1
            per_canon[q]["emitted"] += 1
        kept.append(uid)
    stats["quota_dropped"] = quota_dropped
    stats["per_canon"] = {qc: per_canon[qc] for qc in sorted(per_canon)}

    cap = getattr(slice, "cap", None)
    cap = len(kept) if cap is None else max(0, int(cap))
    final = kept[:cap]
    dropped = len(kept) - len(final)
    if dropped:
        stats["cap_truncated"] = dropped
    stats["pool"] = len(order)
    stats["joint"] = sum(1 for u in final if joint_of[u])

    for rank, uid in enumerate(final, start=1):
        entry = pool[uid]
        raw, base, boost, covered, matched = scores[uid]
        urow = entry["row"]
        roles = entry["roles"]
        role = (
            max(roles, key=lambda r: (_ROLE_RANK.get(r, -1), r))
            if roles
            else None
        )
        signals: dict[str, Any] = {
            "entity_idf": round(base, 6),
            "matched_canons": matched,
            "query_canons_covered": len(covered),
            "role": role,
            "roles": sorted(roles),
            "pinned": bool(entry["pinned"]),
            "alias_matched": any(c not in qcanons for c in matched),
            "spans": list(entry["spans"]),
        }
        if boost:
            signals["multi_canon_boost"] = round(boost, 6)
        if joint_of[uid]:
            # V8-10.03 — unit mentions ALL resolved query canons; the
            # cross-worker explain/fusion contract key.
            signals["joint"] = True
        out.candidates.append(
            CandidateV7(
                unit_id=uid,
                source_id=str(urow.get("source_id") or ""),
                revision=int(urow.get("revision") or 0),
                lane=LANE_NAME,
                rank=rank,
                raw_score=raw,
                signals=signals,
            )
        )

    if deadline.expired():
        deadline_hit = True
    if deadline_hit:
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
        stats["deadline"] = True
    elif truncated:
        out.status = LaneStatus.PARTIAL
        out.reason = "scan_bound"
    return out


class EntityLane(LaneV7):
    """LaneV7 protocol wrapper for registry wiring (V7-05.01)."""

    name = LANE_NAME

    def run(
        self, ctx: LaneContextV7, query: QueryViewV7, slice: LaneSlice
    ) -> LaneOutput:
        return lane_entity(ctx, query, slice)


# Lane modules self-register at import (lanes_base contract); the
# pipeline's lane-module table is owned by the main-session integration.
from .lanes_base import register_lane  # noqa: E402

register_lane(LaneName.ENT, lane_entity)


__all__ = [
    "ALIAS_TABLE",
    "CANON_TABLE",
    "EntityLane",
    "FORMULA_STATUS",
    "LANE_NAME",
    "LANE_VERSION",
    "MENTIONS_TABLE",
    "PER_CANON_QUOTA_ARM",
    "PER_CANON_QUOTA_DEFAULT",
    "lane_conn",
    "lane_entity",
    "resolve_query_speaker_canons",
]
