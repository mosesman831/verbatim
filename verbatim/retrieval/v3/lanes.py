"""V3 retrieval lanes (SPEC_V3 §28, V3-28.01–28.13; SPEC_V4 F4-10/F4-13).

Every lane is a small pure-ish function over a :class:`LaneContext` that
returns one-based ranks keyed by ``(object_kind, object_id)`` — v3 retrieves
more than claims (procedures, episodes, observations, working items), so the
lane contract is object-generic rather than claim-only.

Core rules honored here:

- Eligibility runs before per-lane rank truncation (V3-28.01, V4-28.02):
  scope, quarantine, suppression, freshness, and known-at predicates are
  enforced inside the SQL of each lane where possible and re-verified per
  candidate through ``union.admit_keys`` BEFORE any candidate counts toward
  the lane's result bound — never "oversample then filter". Candidate
  production pages: a fetch window that fills with ineligible rows grows
  (geometrically) until the *eligible* bound is reached, the corpus is
  exhausted, or the request deadline stops the scan — a run of held
  matches can never hide a later eligible hit (F4-10, scenario C31).
- Deadline-honest coverage (V4-27.09): a lane whose scan is cut by the
  deadline mid-corpus reports ``status="partial"`` with
  ``reason="deadline"``; lanes never silently truncate.
- Lexical statistics are partitioned by authorization-equivalent scope
  (V3-28.04): the FTS5 projection carries ``scope_id`` per row and every
  lane query intersects the authorized set — idf/coverage can never see
  cross-scope rows.
- Historical strictness (V3-28.11): when ``known_at_seq`` is set, rows with
  ``recorded_from > known_at_seq`` are excluded at the SQL level so future
  rows cannot influence earlier-known-time ordering.
- Bounded honest degradation (V3-28.12): missing optional inference
  (numpy/encoder, sparse/late artifacts) marks the lane ``unavailable`` or
  ``degraded`` — never a failure. ``VerbatimError`` carrying authorization
  or integrity codes is re-raised (fail closed).
- Truthful concurrency reporting (V4-27.07, F4-13): ``run_lanes`` executes
  lanes sequentially on the caller's single read snapshot — seed-dependent
  lanes (graph/causal/episode_hierarchy) consume the accumulated union
  seeds in ``LANE_ORDER``, and parallel workers would need independent
  ``store.read()`` snapshots that can diverge from the pinned projection
  generation — so the report honestly states ``concurrency="sequential"``
  on every emitted result and on the returned :class:`LaneRunReport`.
"""

from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ...core.types import ErrorCode, VerbatimError, safe_json_loads
from ...core.types_v3 import QueryClass, RecallRequestV3
from ...core.time import now_us
from ...embeddings.codec import Float32Codec
from ...embeddings import vectors as _vec
from ...storage.repos import has_table as _has_table  # noqa: F401
from .. import candidates as _cand
from ..query import QueryPlan

# Per-lane initial caps (SPEC_V3 §28 table).
LANE_CAPS: dict[str, int] = {
    "exact_id": 40,
    "lexical": 40,
    "sparse": 40,
    "dense": 40,
    "late": 24,
    "structured": 20,
    "temporal": 20,
    "episode_hierarchy": 20,
    "graph": 20,
    "causal": 20,
    "procedural_signature": 20,
    "freshness_env": 20,
    "browse": 20,
    "working": 20,
}

# Lanes a query may ask the controller to enable; the controller's lane set
# decides which actually run (V3-26.10: skipped lanes stay visible in
# diagnostics).
LANE_ORDER: tuple[str, ...] = (
    "exact_id",
    "lexical",
    "structured",
    "temporal",
    "procedural_signature",
    "episode_hierarchy",
    "working",
    "freshness_env",
    "dense",
    "sparse",
    "late",
    "graph",
    "causal",
    "browse",
)

# Identifier-shaped tokens the exact lane treats as hard slots (§28 exact
# identifier lane; §29.05 hard identifier constraints).
_IDENTIFIER_RE = re.compile(
    r"^(?:[\w.\-]+::[\w.\-:]+"          # test ids / namespaced ids
    r"|[~./]*[\w.\-]+/[\w./\-]+"        # file paths
    r"|[\w\-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|rb|c|cc|cpp|h|hpp|cs|sh|"
    r"json|ya?ml|toml|xml|sql|md|txt|ini|cfg|css|html)"  # file names
    r"|v?\d+\.\d+(?:\.\d+)*(?:[-+][\w.]+)?"              # versions
    r"|[A-Za-z_][\w]*:[0-9a-f]{6,40}"                    # id:digest forms
    r")$"
)


@dataclass
class LaneResult:
    """One lane's output: object-keyed ranks plus honest status.

    ``status``: ``ok`` | ``degraded`` | ``partial`` | ``unavailable`` |
    ``skipped``. ``attempted`` counts rows inspected before the cap;
    ``overflow`` counts eligible rows the cap pushed out (V3-28.03).
    ``concurrency`` records how the lane actually executed when produced
    through :func:`run_lanes` (``"sequential"`` today — V4-27.07 requires
    reporting serial execution as serial); ``None`` when a lane ran
    standalone outside the runner.
    """

    lane: str
    hits: dict = field(default_factory=dict)  # (object_kind, object_id) -> rank
    status: str = "ok"
    reason: Optional[str] = None
    warnings: tuple = ()
    attempted: int = 0
    overflow: int = 0
    concurrency: Optional[str] = None
    # Bounded lane diagnostics surfaced into capabilities() — e.g. the
    # graph lane's visited/examined-edge accounting and provenance labels.
    details: dict = field(default_factory=dict)


class LaneRunReport(list):
    """``run_lanes`` output: the ordered LaneResults plus run diagnostics.

    ``concurrency`` is the truthfully-reported execution mode of the run
    (V4-27.07): ``"sequential"`` — lanes share the caller's read snapshot
    and seed-dependent lanes consume accumulated seeds in ``LANE_ORDER``.
    A plain ``list`` subclass so existing consumers iterating results are
    unaffected.
    """

    def __init__(self, iterable=(), *, concurrency: str = "sequential"):
        super().__init__(iterable)
        self.concurrency = concurrency


@dataclass
class LaneContext:
    """Everything a lane needs; assembled once per recall (read snapshot)."""

    conn: sqlite3.Connection
    store: Any
    request: RecallRequestV3
    plan: QueryPlan
    query_class: QueryClass
    scope_ids: tuple               # authorized scope ids (non-empty)
    generation: int
    deadline: _cand.Deadline
    lane_cap: int = 40
    seeds: dict = field(default_factory=dict)  # accumulated union hits
    cfg: Any = None                # RetrievalV3Config | None
    # Graph-expansion bounds (V4-30.09). Defaults are the spec floor — one
    # hop, ≤100 visited nodes, ≤200 examined edges. Wider bounds are never
    # assumed: recall_v3 fills them from the controller's declared tier
    # budgets, so an expanded allowance is always an explicit policy value.
    graph_hops: int = 1
    graph_fanout: int = 32  # novel neighbours enqueued per node
    graph_max_nodes: int = 100
    graph_max_edges: int = 200
    # Per-snapshot eligibility/schema memo (union.SnapshotCache). Lanes of
    # one recall share the parent's instance so admission verdicts resolve
    # once per snapshot — never carried into a later recall.
    snap: Any = None

    @property
    def known_at_seq(self) -> Optional[int]:
        return self.request.known_at_seq


def _ph(n: int) -> str:
    return ",".join("?" * n)


def _snap(ctx: LaneContext):
    """The per-snapshot cache shared across this recall's lanes — lazily
    created and bound to ``ctx.conn`` by union (no module cycle: the
    import stays local)."""
    snap = getattr(ctx, "snap", None)
    if snap is None or getattr(snap, "conn", None) is not ctx.conn:
        from . import union as _union  # local import: no module cycle
        snap = _union.snapshot_for(ctx)
    return snap


def _recorded_pred(alias: str, known: Optional[int]) -> tuple[str, list]:
    """Known-time predicate: no future row may influence ranking (V3-28.11)."""
    p = f"{alias}." if alias else ""
    if known is None:
        return f"{p}recorded_until IS NULL", []
    return (
        f"{p}recorded_from <= ? AND ({p}recorded_until IS NULL"
        f" OR {p}recorded_until > ?)",
        [known, known],
    )


def _rank_eligible(result: LaneResult, eligible: list, cap: int) -> None:
    """Assign one-based ranks to an already-admitted ordered key list.

    Eligible keys beyond ``cap`` count as overflow (V3-28.03) — the cap is
    a bound on *admitted* results, never on rows fetched.
    """
    for i, key in enumerate(eligible):
        if i >= cap:
            result.overflow += 1
            continue
        result.hits.setdefault(key, i + 1)


def _deadline_partial(result: LaneResult) -> None:
    """Mark a lane's coverage incomplete because the deadline stopped the
    scan mid-corpus (V4-27.09 — never claim healthy completed coverage)."""
    result.status = "partial"
    result.reason = result.reason or "deadline"
    if "scan_incomplete:deadline" not in result.warnings:
        result.warnings += ("scan_incomplete:deadline",)


def _paged(ctx: LaneContext, result: LaneResult, fetch_page,
           cap: int) -> list:
    """Eligibility-first paged admission (V4-28.02, F4-10).

    ``fetch_page(scale)`` returns ``(attempted_delta, exhausted,
    ordered_keys)`` for a geometrically growing window — the lane's
    candidate keys in its own rank order over ``base * scale`` underlying
    rows; ``exhausted`` is True only when the window provably covered the
    whole matchable corpus (every bounded query returned fewer rows than
    its bound). Keys are admitted per page through ``union.admit_keys`` —
    ineligible rows consume fetch budget but never rank capacity — and
    paging continues until ``cap`` eligible keys are found, the corpus is
    exhausted, or the request deadline stops the scan. A deadline stop
    reports ``partial`` coverage rather than silently truncating.

    Admission verdicts are per-candidate (``admit_keys`` batches identical
    checks), so per-page admission is equivalent to admitting the union of
    all pages; ordering follows each page's own produced order.
    """
    from . import union as _union  # local import: no module cycle

    seen: set = set()
    admitted: set = set()
    eligible: list = []
    scale = 1
    while True:
        delta, exhausted, ordered = fetch_page(scale)
        result.attempted += max(delta, 0)
        new_keys = [k for k in ordered if k not in seen]
        if new_keys:
            seen.update(new_keys)
            admitted.update(_union.admit_keys(ctx, new_keys))
        # Dedupe: a lane's produced order can repeat a key (a claim in two
        # temporal buckets, a member shared by episodes) — each distinct
        # eligible key consumes rank capacity exactly once (V3-28.03).
        page_eligible = dict.fromkeys(k for k in ordered if k in admitted)
        # Merge, don't rebuild: a deadline-truncated fetch can return a
        # SHRINKER window than the page before — previously admitted keys
        # the truncated page lost stay eligible, ordered behind the fresh
        # page. Complete windows always reproduce every prior key, so this
        # reduces to the page's own ordering in the normal case.
        merged = list(page_eligible)
        merged.extend(k for k in eligible if k not in page_eligible)
        eligible = merged
        if len(eligible) >= cap or exhausted:
            return eligible
        if scale > 1 and delta == 0:
            # The window grew but the lane produced nothing new — the
            # fetch path cannot expose further rows; treat as the end of
            # the reachable corpus rather than looping forever.
            return eligible
        if ctx.deadline.expired():
            _deadline_partial(result)
            return eligible
        scale *= 2


def _admit_stream(ctx: LaneContext, result: LaneResult, keys: list,
                  cap: int, chunk: int = 64) -> list:
    """Chunked admission over a complete pre-ranked key stream.

    For lanes whose ordering is already fully computed in memory (the
    dense lane's exact cosine order), the "fetch window" is the whole
    ranked stream — eligibility still gates the bound: chunks are admitted
    in order until ``cap`` eligible keys, stream exhaustion, or deadline.
    """
    from . import union as _union  # local import: no module cycle

    eligible: list = []
    emitted: set = set()
    for i in range(0, len(keys), max(chunk, 1)):
        batch = keys[i:i + chunk]
        result.attempted += len(batch)
        admitted = _union.admit_keys(ctx, batch)
        for k in batch:
            if k in admitted and k not in emitted:
                emitted.add(k)
                eligible.append(k)
        if len(eligible) >= cap:
            return eligible
        if ctx.deadline.expired():
            _deadline_partial(result)
            return eligible
    return eligible


def _bounded_fts_scan_clipped(ctx: LaneContext) -> bool:
    """Whether the no-FTS5 substring scan could have missed matches.

    ``_cand._fts_fallback_scan`` scans at most ``_cand._POSTING_CAP`` rows
    of the current projection generation (its bounded slice). When that
    path is the only one available and the authorized generation holds
    more rows than the scan bound, coverage cannot honestly be called
    complete (V4-28.09). Detectable only when the FTS5 index is genuinely
    absent — a corrupt-index OperationalError falls back silently inside
    ``_fts_search`` and is indistinguishable from here.
    """
    if _cand._fts5_available(ctx.conn):
        return False
    try:
        row = ctx.conn.execute(
            "SELECT COUNT(*) FROM fts_rows"
            " WHERE projection_generation = ?"
            f" AND scope_id IN ({_ph(len(ctx.scope_ids))})",
            [ctx.generation, *ctx.scope_ids],
        ).fetchone()
    except sqlite3.Error:
        return True  # cannot prove coverage — do not claim it
    return bool(row and row[0] > _cand._POSTING_CAP)


def _identifier_tokens(plan: QueryPlan) -> list:
    """Identifier-shaped tokens preserved verbatim by query analysis."""
    out: list = []
    seen: set = set()
    for token in (*plan.phrases, *plan.terms):
        t = token.strip().strip("\"'(),;")
        if len(t) < 2 or not _IDENTIFIER_RE.match(t):
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# exact identifier lane — cap 40, default on (§28 row 1)
# ---------------------------------------------------------------------------


def lane_exact_id(ctx: LaneContext) -> LaneResult:
    """Identifier/slot lookup: entity ids, paths, test names, object ids.

    All matches resolve to objects inside the authorized scope set — an
    identifier can narrow, never widen (V3-28.01). Each probe's bound
    scales with the paging window so ineligible identifier matches cannot
    starve later eligible ones (V4-28.02).
    """
    result = LaneResult("exact_id", attempted=0)
    conn = ctx.conn
    scopes = list(ctx.scope_ids)
    entity_ids = sorted(set(ctx.request.entity_ids) | set(ctx.plan.entity_ids))
    tokens = _identifier_tokens(ctx.plan)
    snap = _snap(ctx)
    has_claim_entities = snap.has_table("claim_entities")
    has_entities = snap.has_table("entities")
    has_fts = snap.has_table("facts_fts")
    state = {"scanned": 0}

    def fetch(scale: int):
        """One collection round at ``scale`` × the original probe bounds."""
        scanned = 0
        bound_hit = False  # any probe returned a full page → more may exist
        keys: list = []

        # 1. Explicit entity ids: entities -> claim_entities -> claims.
        ent_claim_limit = ctx.lane_cap * 2 * scale
        if entity_ids and has_claim_entities:
            rows = conn.execute(
                "SELECT DISTINCT ce.claim_id FROM claim_entities ce"
                " JOIN claims c ON c.claim_id = ce.claim_id"
                f" WHERE ce.entity_id IN ({_ph(len(entity_ids))})"
                f" AND c.scope_id IN ({_ph(len(scopes))})"
                " ORDER BY ce.claim_id LIMIT ?",
                [*entity_ids, *scopes, ent_claim_limit],
            ).fetchall()
            scanned += len(rows)
            bound_hit = bound_hit or len(rows) >= ent_claim_limit
            keys.extend(("claim", r[0]) for r in rows)

        # 2. Identifier tokens: object ids + entity labels/aliases + FTS.
        probe = 8 * scale
        for token in tokens:
            # literal object-id match across the object tables — a point
            # lookup, unbounded by the paging window.
            row = conn.execute(
                "SELECT 1 FROM claims WHERE claim_id = ?"
                f" AND scope_id IN ({_ph(len(scopes))}) LIMIT 1",
                [token, *scopes],
            ).fetchone()
            if row is not None:
                keys.append(("claim", token))
            if has_entities:
                ent_rows = conn.execute(
                    "SELECT e.entity_id FROM entities e"
                    f" WHERE e.scope_id IN ({_ph(len(scopes))})"
                    " AND lower(e.label) = lower(?)"
                    " UNION SELECT ea.entity_id FROM entity_aliases ea"
                    " JOIN entities e ON e.entity_id = ea.entity_id"
                    f" WHERE e.scope_id IN ({_ph(len(scopes))})"
                    " AND lower(ea.normalized_alias) = lower(?)"
                    " LIMIT ?",
                    [*scopes, token, *scopes, token.casefold(), probe],
                ).fetchall()
                scanned += len(ent_rows)
                bound_hit = bound_hit or len(ent_rows) >= probe
                for (eid,) in ent_rows:
                    crow = conn.execute(
                        "SELECT DISTINCT ce.claim_id FROM claim_entities ce"
                        " JOIN claims c ON c.claim_id = ce.claim_id"
                        " WHERE ce.entity_id = ?"
                        f" AND c.scope_id IN ({_ph(len(scopes))})"
                        " ORDER BY ce.claim_id LIMIT ?",
                        [eid, *scopes, probe],
                    ).fetchall()
                    scanned += len(crow)
                    bound_hit = bound_hit or len(crow) >= probe
                    keys.extend(("claim", r[0]) for r in crow)
            # exact substring over the sanctioned FTS projection (paths)
            if has_fts:
                fts_rows = conn.execute(
                    "SELECT fr.claim_id FROM fts_rows fr"
                    " JOIN facts_fts ft ON ft.fts_row_id = fr.row_id"
                    " WHERE fr.projection_generation = ?"
                    f" AND fr.scope_id IN ({_ph(len(scopes))})"
                    " AND instr(ft.text, ?) > 0"
                    " ORDER BY fr.row_id LIMIT ?",
                    [ctx.generation, *scopes, token, probe],
                ).fetchall()
                scanned += len(fts_rows)
                bound_hit = bound_hit or len(fts_rows) >= probe
                keys.extend(("claim", r[0]) for r in fts_rows)

        delta = max(0, scanned - state["scanned"])
        state["scanned"] = scanned
        # dedupe preserving first-seen order
        ordered = list(dict.fromkeys(keys))
        return delta, not bound_hit, ordered

    eligible = _paged(ctx, result, fetch, ctx.lane_cap)
    _rank_eligible(result, eligible, ctx.lane_cap)
    return result


# ---------------------------------------------------------------------------
# lexical lane — wraps the v2 FTS5/BM25 path (§28 row 2)
# ---------------------------------------------------------------------------


def lane_lexical(ctx: LaneContext) -> LaneResult:
    """FTS5/BM25 partitioned by authorized scope (V3-28.04, V4-28.02).

    Reuses the v2 ``_fts_search`` machinery (FTS5 MATCH with
    generation+scope predicates inside the query, or the bounded substring
    fallback for unsegmented scripts / index-free builds).

    F4-10 / scenario C31: the fetch window pages — it starts at the old
    ``lane_cap * 4`` oversample and doubles until ``lane_cap`` ELIGIBLE
    claims are admitted, the match set is exhausted, or the deadline stops
    the scan. Ineligible hits (quarantined, wrong-state, suppressed)
    consume fetch budget but never rank capacity, so >160 held matches can
    no longer hide the next eligible claim.
    """
    result = LaneResult("lexical")
    if not ctx.plan.match_query:
        result.status = "skipped"
        result.reason = "no_lexical_signal"
        return result
    base = min(ctx.lane_cap * 4, 400)
    state = {"rows": 0, "stats_cut": False}

    def fetch(scale: int):
        limit = base * scale
        rows = _cand._fts_search(
            ctx.store, ctx.conn, list(ctx.scope_ids), ctx.plan,
            ctx.generation, limit, deadline=ctx.deadline,
        )
        diag = getattr(rows, "diag", None) or {}
        if diag.get("stats_complete") is False:
            # The request deadline cut the corpus-statistics/scoring pass —
            # the returned page can be short of the true eligible set even
            # though the fetch window covered it (V4-27.09 honesty).
            state["stats_cut"] = True
        delta = max(0, len(rows) - state["rows"])
        state["rows"] = max(state["rows"], len(rows))
        ordered = list(dict.fromkeys(("claim", r[0]) for r in rows))
        # Only an unclipped short page proves the eligible corpus is
        # exhausted; a deadline-truncated scan must keep the loop honest.
        exhausted = len(rows) < limit and not state["stats_cut"]
        return delta, exhausted, ordered

    eligible = _paged(ctx, result, fetch, ctx.lane_cap)
    _rank_eligible(result, eligible, ctx.lane_cap)
    if result.status == "ok" and state["stats_cut"]:
        _deadline_partial(result)
    if result.status == "ok" and _bounded_fts_scan_clipped(ctx):
        # The substring scan served this query from a bounded slice of a
        # larger generation — coverage is honestly incomplete (V4-28.09).
        result.status = "partial"
        result.reason = "bounded_scan"
        result.warnings += (f"fts_scan_bound:{_cand._POSTING_CAP}",)
    return result


# ---------------------------------------------------------------------------
# structured lane — entities + predicates (§28 row 6)
# ---------------------------------------------------------------------------


def lane_structured(ctx: LaneContext) -> LaneResult:
    """Predicate/entity joins over ``claims``/``claim_entities``.

    Same queries as the v2 structured helpers — all-entities INTERSECT
    semantics plus predicate matches — with the window scaled by the paging
    loop so admission, not the fetch bound, decides the result (V4-28.02).
    The claims join keeps ``scope_id`` inside the authorized set
    (V3-28.01). Rank merge matches ``CandidateMap`` semantics: first source
    wins a shared claim's rank, then (rank, claim_id) orders the lane.
    """
    result = LaneResult("structured")
    if not (ctx.plan.entity_ids or ctx.plan.predicates):
        result.status = "skipped"
        result.reason = "no_structured_constraints"
        return result
    conn = ctx.conn
    scopes = list(ctx.scope_ids)
    entity_ids = sorted(set(ctx.plan.entity_ids))
    predicates = sorted(set(ctx.plan.predicates))
    has_claim_entities = _snap(ctx).has_table("claim_entities")
    state = {"rows": 0}

    def fetch(scale: int):
        limit = 20 * scale  # STRUCTURED_LIMIT window, grown per page
        scanned = 0
        bound_hit = False
        ranks: dict = {}
        if entity_ids and has_claim_entities:
            rows = conn.execute(
                "SELECT ce.claim_id FROM claim_entities ce"
                " JOIN claims c ON c.claim_id = ce.claim_id"
                f" WHERE ce.entity_id IN ({_ph(len(entity_ids))})"
                f" AND c.scope_id IN ({_ph(len(scopes))})"
                " GROUP BY ce.claim_id"
                " HAVING COUNT(DISTINCT ce.entity_id) = ?"
                " ORDER BY ce.claim_id LIMIT ?",
                [*entity_ids, *scopes, len(entity_ids), limit],
            ).fetchall()
            scanned += len(rows)
            bound_hit = bound_hit or len(rows) >= limit
            for rank, (cid,) in enumerate(rows, start=1):
                ranks.setdefault(cid, rank)
        if predicates:
            rows = conn.execute(
                "SELECT claim_id FROM claims"
                f" WHERE predicate IN ({_ph(len(predicates))})"
                f" AND scope_id IN ({_ph(len(scopes))})"
                " ORDER BY claim_id LIMIT ?",
                [*predicates, *scopes, limit],
            ).fetchall()
            scanned += len(rows)
            bound_hit = bound_hit or len(rows) >= limit
            for rank, (cid,) in enumerate(rows, start=1):
                ranks.setdefault(cid, rank)
        delta = max(0, scanned - state["rows"])
        state["rows"] = scanned
        ordered = [
            ("claim", cid)
            for cid, _rank in sorted(
                ranks.items(), key=lambda kv: (kv[1], kv[0])
            )
        ]
        return delta, not bound_hit, ordered

    eligible = _paged(ctx, result, fetch, ctx.lane_cap)
    _rank_eligible(result, eligible, ctx.lane_cap)
    return result


# ---------------------------------------------------------------------------
# temporal lane — valid-time windows with bucket spreading (§28 row 7)
# ---------------------------------------------------------------------------


def lane_temporal(ctx: LaneContext) -> LaneResult:
    """Valid-interval membership at the requested point/range (V3-28.10).

    Scores by interval tightness (shortest covering interval first) and
    spreads picks across time buckets so long windows are not dominated by
    the densest period. Historical queries exclude future rows at the SQL
    level (V3-28.11).
    """
    result = LaneResult("temporal")
    conn = ctx.conn
    point = ctx.plan.valid_at_us
    until = ctx.plan.valid_until_us
    known = ctx.known_at_seq
    scopes = list(ctx.scope_ids)

    if point is None and until is None:
        if ctx.query_class in (QueryClass.TIMELINE, QueryClass.PAST_STATE):
            # Bucket spread across recorded history for timeline browsing.
            # The scan window pages (V4-28.02): ineligible revisions never
            # hide later eligible picks.
            pred, pparams = _recorded_pred("cr", known)
            state = {"rows": 0}

            def fetch(scale: int):
                limit = 400 * scale
                rows = conn.execute(
                    "SELECT c.claim_id, cr.recorded_from FROM claims c"
                    " JOIN claim_revisions cr"
                    "   ON cr.claim_id = c.claim_id"
                    f" WHERE c.scope_id IN ({_ph(len(scopes))}) AND {pred}"
                    " ORDER BY cr.recorded_from DESC, c.claim_id LIMIT ?",
                    [*scopes, *pparams, limit],
                ).fetchall()
                delta = max(0, len(rows) - state["rows"])
                state["rows"] = len(rows)
                # Spread: interleave buckets of ~equal size so one dense
                # period cannot starve older evidence (V3-28.10).
                buckets: dict[int, list] = {}
                for cid, rf in rows:
                    buckets.setdefault(rf or 0, []).append(cid)
                ordered: list = []
                keys_sorted = sorted(buckets.keys(), reverse=True)
                idx = 0
                while len(ordered) < len(rows):
                    progressed = False
                    for b in keys_sorted:
                        bucket = buckets[b]
                        if idx < len(bucket):
                            ordered.append(("claim", bucket[idx]))
                            progressed = True
                    if not progressed:
                        break
                    idx += 1
                return delta, len(rows) < limit, ordered

            eligible = _paged(ctx, result, fetch, ctx.lane_cap)
            _rank_eligible(result, eligible, ctx.lane_cap)
            return result
        result.status = "skipped"
        result.reason = "no_time_filter"
        return result

    lo = point if point is not None else 0
    hi = until if until is not None else point
    pred, pparams = _recorded_pred("cr", known)
    state = {"rows": 0}

    def fetch_windowed(scale: int):
        limit = 400 * scale
        rows = conn.execute(
            "SELECT c.claim_id, vi.from_us, vi.until_us, cr.recorded_from"
            " FROM claims c"
            " JOIN claim_revisions cr ON cr.claim_id = c.claim_id"
            " JOIN valid_intervals vi"
            "   ON vi.claim_id = c.claim_id AND vi.revision = cr.revision"
            f" WHERE c.scope_id IN ({_ph(len(scopes))}) AND {pred}"
            " AND (vi.until_us IS NULL OR vi.until_us > ?)"
            " AND (vi.from_us IS NULL OR vi.from_us <= ?)"
            " ORDER BY c.claim_id, vi.interval_no LIMIT ?",
            [*scopes, *pparams, lo, hi if hi is not None else lo, limit],
        ).fetchall()
        delta = max(0, len(rows) - state["rows"])
        state["rows"] = len(rows)
        # Tightest covering interval first; dedupe per claim keeping best.
        scored: dict = {}
        for cid, f_us, u_us, rf in rows:
            width = (
                (u_us - f_us)
                if (f_us is not None and u_us is not None)
                else None
            )
            key = (0 if width is not None else 1, width or 0, cid)
            if cid not in scored or key < scored[cid]:
                scored[cid] = key
        ordered_ids = [
            cid for cid, _ in sorted(scored.items(), key=lambda kv: kv[1])
        ]
        return (
            delta,
            len(rows) < limit,
            [("claim", cid) for cid in ordered_ids],
        )

    eligible = _paged(ctx, result, fetch_windowed, ctx.lane_cap)
    _rank_eligible(result, eligible, ctx.lane_cap)
    return result


# ---------------------------------------------------------------------------
# procedural signature lane (§28 row 11)
# ---------------------------------------------------------------------------


_OP_HINTS: dict[str, str] = {
    "test": "run_check",
    "check": "run_check",
    "verify": "run_check",
    "build": "run_check",
    "patch": "apply_patch",
    "edit": "apply_patch",
    "fix": "apply_patch",
    "search": "search_repo",
    "find": "search_repo",
    "read": "inspect_file",
    "inspect": "inspect_file",
    "open": "inspect_file",
}


def lane_procedural_signature(ctx: LaneContext) -> LaneResult:
    """Ordered operation classes + intent before text similarity (V3-28.09).

    Reads ``procedure_signatures`` (written by the signature_index job);
    environment compatibility is a *hard* filter — a mismatched platform
    disqualifies the procedure outright.
    """
    result = LaneResult("procedural_signature")
    conn = ctx.conn
    if not _snap(ctx).has_table("procedure_signatures"):
        result.status = "unavailable"
        result.reason = "no_signature_index"
        return result
    scopes = list(ctx.scope_ids)

    task_env = getattr(getattr(ctx.request, "task", None), "environment", None)
    want_platform = getattr(task_env, "platform", None)

    hinted_ops = {
        op for t in ctx.plan.terms for op in (_OP_HINTS.get(t.lower()),) if op
    }
    state = {"rows": 0}

    def fetch(scale: int):
        # The signature scan window pages (V4-28.02): procedures dropped by
        # the hard filters or by admission never hide later eligible ones.
        limit = 200 * scale
        rows = conn.execute(
            "SELECT ps.procedure_id, ps.revision, ps.ordered_ops_json,"
            " ps.intent_key, p.task_label, p.state, p.environment_json"
            " FROM procedure_signatures ps"
            " JOIN procedures p ON p.procedure_id = ps.procedure_id"
            f" WHERE ps.scope_id IN ({_ph(len(scopes))})"
            " ORDER BY ps.procedure_id LIMIT ?",
            [*scopes, limit],
        ).fetchall()
        delta = max(0, len(rows) - state["rows"])
        state["rows"] = len(rows)
        scored: list = []
        for pid, rev, ops_json, intent_key, label, pstate, env_json in rows:
            if pstate in ("retired", "deprecated"):
                continue
            env = safe_json_loads(env_json) or {}
            if want_platform and env.get("platform") and (
                env["platform"] != want_platform
            ):
                continue  # environment incompatibility is a hard filter
            ops = safe_json_loads(ops_json) or []
            op_hits = len(set(ops) & hinted_ops)
            intent_hit = bool(
                intent_key and intent_key.lower() in ctx.request.query.lower()
            )
            text_hit = _cand._text_hit(label, ctx.plan)
            score = (
                op_hits * 2 + (2 if intent_hit else 0) + (1 if text_hit else 0)
            )
            if score > 0:
                scored.append((-score, pid, rev))
        scored.sort()
        return (
            delta,
            len(rows) < limit,
            [("procedure", pid) for _s, pid, _r in scored],
        )

    eligible = _paged(ctx, result, fetch, ctx.lane_cap)
    _rank_eligible(result, eligible, ctx.lane_cap)
    return result


# ---------------------------------------------------------------------------
# episode/hierarchy lane (§28 row 8)
# ---------------------------------------------------------------------------


def lane_episode_hierarchy(ctx: LaneContext) -> LaneResult:
    """Episodes whose label matches or whose member claims already seeded.

    The episode scan pages (V4-28.02): a window of episodes that admit
    nothing (held members, wrong lifecycle) cannot hide later episodes.
    """
    result = LaneResult("episode_hierarchy")
    conn = ctx.conn
    if not (_snap(ctx).has_table("episodes") and _snap(ctx).has_table("episode_members")):
        result.status = "unavailable"
        result.reason = "no_episode_tables"
        return result
    scopes = list(ctx.scope_ids)
    known = ctx.known_at_seq
    pred, pparams = _recorded_pred("e", known)
    mpred, mparams = _recorded_pred("em", known)

    seed_claims = {
        oid for (kind, oid) in ctx.seeds if kind == "claim"
    }
    state = {"rows": 0}

    def fetch(scale: int):
        limit = 200 * scale
        rows = conn.execute(
            "SELECT e.episode_id, e.label FROM episodes e"
            f" WHERE e.scope_id IN ({_ph(len(scopes))}) AND {pred}"
            " ORDER BY e.episode_id LIMIT ?",
            [*scopes, *pparams, limit],
        ).fetchall()
        delta = max(0, len(rows) - state["rows"])
        state["rows"] = len(rows)

        ep_ids = [r[0] for r in rows]
        member_rows: dict = {}
        if ep_ids:
            for ep_id, okind, oid in conn.execute(
                "SELECT em.episode_id, em.object_kind, em.object_id"
                " FROM episode_members em"
                f" WHERE em.episode_id IN ({_ph(len(ep_ids))}) AND {mpred}"
                " ORDER BY em.ord, em.object_id",
                [*ep_ids, *mparams],
            ).fetchall():
                member_rows.setdefault(ep_id, []).append((okind, oid))

        scored: list = []
        for ep_id, label in rows:
            members = member_rows.get(ep_id, [])
            member_seed_hits = sum(
                1 for okind, oid in members
                if okind == "claim" and oid in seed_claims
            )
            label_hit = _cand._text_hit(label, ctx.plan)
            if member_seed_hits or label_hit:
                scored.append(
                    (-(member_seed_hits * 2 + (1 if label_hit else 0)), ep_id,
                     [("claim", oid) for okind, oid in members
                      if okind == "claim"][:8])
                )
        scored.sort()
        ordered: list = []
        for _s, ep_id, member_claims in scored:
            ordered.append(("episode", ep_id))
            ordered.extend(member_claims)
        return delta, len(rows) < limit, ordered

    eligible = _paged(ctx, result, fetch, ctx.lane_cap)
    _rank_eligible(result, eligible, ctx.lane_cap)
    return result


# ---------------------------------------------------------------------------
# working lane — session working set + open plans (§26 working route)
# ---------------------------------------------------------------------------


def lane_working(ctx: LaneContext) -> LaneResult:
    """Session working-set items and open prospective records (§24.03).

    Each source pages independently until the eligible bound (V4-28.02):
    held/expired working items cannot hide a later eligible record.
    """
    result = LaneResult("working")
    conn = ctx.conn
    task = getattr(ctx.request, "task", None)
    session_id = getattr(task, "task_id", None) or ""
    now = now_us()
    eligible: list = []
    cap = ctx.lane_cap

    if (
        session_id
        and _snap(ctx).has_table("working_sets")
        and _snap(ctx).has_table("working_set_items")
    ):
        ws_state = {"rows": 0}

        def fetch_working(scale: int):
            limit = ctx.lane_cap * 4 * scale
            rows = conn.execute(
                "SELECT wsi.item_id FROM working_set_items wsi"
                " JOIN working_sets ws ON ws.set_id = wsi.set_id"
                f" WHERE ws.scope_id IN ({_ph(len(ctx.scope_ids))})"
                " AND ws.session_id = ? AND ws.expires_us > ?"
                " ORDER BY wsi.ord, wsi.item_id LIMIT ?",
                [*ctx.scope_ids, session_id, now, limit],
            ).fetchall()
            delta = max(0, len(rows) - ws_state["rows"])
            ws_state["rows"] = len(rows)
            return delta, len(rows) < limit, [
                ("working_item", r[0]) for r in rows
            ]

        eligible.extend(_paged(ctx, result, fetch_working, cap))

    if (
        len(eligible) < cap
        and result.status != "partial"
        and _snap(ctx).has_table("prospective_records")
    ):
        pr_state = {"rows": 0}

        def fetch_prospective(scale: int):
            limit = ctx.lane_cap * 4 * scale
            rows = conn.execute(
                "SELECT record_id FROM prospective_records"
                f" WHERE scope_id IN ({_ph(len(ctx.scope_ids))})"
                " AND status IN ('planned','in_progress','overdue','unknown')"
                " AND recorded_until IS NULL"
                " ORDER BY due_us IS NULL, due_us, record_id LIMIT ?",
                [*ctx.scope_ids, limit],
            ).fetchall()
            delta = max(0, len(rows) - pr_state["rows"])
            pr_state["rows"] = len(rows)
            return delta, len(rows) < limit, [
                ("prospective", r[0]) for r in rows
            ]

        eligible.extend(
            _paged(ctx, result, fetch_prospective, cap - len(eligible))
        )

    _rank_eligible(result, eligible, cap)
    # an empty eligible set is a legitimately empty working set — "ok"
    # unless the scan already reported partial coverage.
    return result


# ---------------------------------------------------------------------------
# freshness / environment lane (§28 row 12)
# ---------------------------------------------------------------------------


def lane_freshness_env(ctx: LaneContext) -> LaneResult:
    """Volatile / revalidation-due objects (§24.04–24.06, §28 row 12).

    Surfaces objects whose freshness row marks them ``volatile`` or whose
    ``revalidate_after`` deadline passed — the verify route's raw material.
    """
    result = LaneResult("freshness_env")
    conn = ctx.conn
    now = now_us()
    eligible: list = []
    cap = ctx.lane_cap
    if _snap(ctx).has_table("freshness"):
        fr_state = {"rows": 0}

        def fetch_freshness(scale: int):
            limit = ctx.lane_cap * 4 * scale
            rows = conn.execute(
                "SELECT object_kind, object_id, revision, class,"
                " revalidate_after_us, anchor_refs_json FROM freshness"
                f" WHERE scope_id IN ({_ph(len(ctx.scope_ids))})"
                " ORDER BY object_kind, object_id LIMIT ?",
                [*ctx.scope_ids, limit],
            ).fetchall()
            delta = max(0, len(rows) - fr_state["rows"])
            fr_state["rows"] = len(rows)
            keys: list = []
            for okind, oid, _rev, cls, ra, anchors_json in rows:
                stale = any(
                    isinstance(a, dict)
                    and a.get("stale_since_seq") is not None
                    for a in (safe_json_loads(anchors_json) or [])
                )
                if (
                    cls == "volatile"
                    or stale
                    or (
                        cls == "revalidate_after"
                        and ra is not None
                        and ra < now
                    )
                ):
                    keys.append((okind, oid))
            return delta, len(rows) < limit, keys

        eligible.extend(_paged(ctx, result, fetch_freshness, cap))
    if (
        len(eligible) < cap
        and result.status != "partial"
        and _snap(ctx).has_table("environment_state")
    ):
        es_state = {"rows": 0}

        def fetch_env(scale: int):
            limit = ctx.lane_cap * 4 * scale
            rows = conn.execute(
                "SELECT key FROM environment_state"
                f" WHERE scope_id IN ({_ph(len(ctx.scope_ids))})"
                " AND volatile = 1"
                " ORDER BY key LIMIT ?",
                [*ctx.scope_ids, limit],
            ).fetchall()
            delta = max(0, len(rows) - es_state["rows"])
            es_state["rows"] = len(rows)
            return delta, len(rows) < limit, [
                ("environment_state", r[0]) for r in rows
            ]

        eligible.extend(
            _paged(ctx, result, fetch_env, cap - len(eligible))
        )
    _rank_eligible(result, eligible, cap)
    return result


# ---------------------------------------------------------------------------
# dense lane — reuse the v2 cosine path (§28 row 4, optional)
# ---------------------------------------------------------------------------


def _semantic_claim_order(ctx: LaneContext) -> tuple:
    """Full cosine ordering over the authorized span→claim join.

    Mirrors ``candidates._semantic`` exactly — same encoder resolution,
    same malformed-vector guards, same authorized span mapping — but
    returns the COMPLETE ranked claim order so admission can page past
    ineligible hits instead of stopping at the fixed v2 oversample
    (V4-28.02). Returns ``(ordered_claim_ids, warnings,
    deadline_exceeded)``; ``ordered_claim_ids`` is ``None`` when the
    encoder path is unavailable (the lane then reports ``unavailable``
    exactly as the wrapped version did).
    """
    conn = ctx.conn
    warnings: list = []
    has_rows = conn.execute("SELECT 1 FROM embeddings LIMIT 1").fetchone()
    if not has_rows:
        return None, warnings, False
    resolved = _cand._query_encoder(ctx.store)
    if resolved is None:
        return None, ["semantic_unavailable"], False
    encoder, encoder_id = resolved

    if not isinstance(encoder_id, str) or not encoder_id:
        encoder_id = getattr(ctx.store, "encoder_id", None)
    if not isinstance(encoder_id, str) or not encoder_id:
        ids = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT encoder_id FROM embeddings"
            ).fetchall()
        ]
        if len(ids) != 1:
            return None, ["semantic_unavailable"], False
        encoder_id = ids[0]

    try:
        raw = encoder(
            ctx.request.query,
            broker=getattr(ctx.store, "transport_broker", None),
            scope_ids=tuple(ctx.scope_ids),
            caller=getattr(
                getattr(ctx.request, "scope", None), "principal_id", ""
            )
            or "",
        )
        qblob = (
            bytes(raw) if isinstance(raw, (bytes, bytearray))
            else Float32Codec.pack(list(raw))
        )
    except Exception:
        return None, ["semantic_unavailable"], False
    if not qblob or len(qblob) % 4:
        return None, ["semantic_unavailable"], False
    qvec = Float32Codec.unpack(qblob, len(qblob) // 4)
    if not all(map(math.isfinite, qvec)) or not any(qvec):
        # malformed or directionless query vector — unavailable, not silent
        return None, ["semantic_unavailable"], False

    # Authorized span→claim map (scoped before ranking, §6/§29); the
    # bounded matrix + exact cosine come from the embeddings module.
    scopes = list(ctx.scope_ids)
    pairs = conn.execute(
        "SELECT DISTINCT ce.span_id, ce.claim_id"
        " FROM claim_evidence ce"
        " JOIN claims c ON c.claim_id = ce.claim_id"
        f" WHERE c.scope_id IN ({_ph(len(scopes))})",
        scopes,
    ).fetchall()
    span_claims: dict[str, list[str]] = {}
    for span_id, claim_id in pairs:
        span_claims.setdefault(span_id, []).append(claim_id)

    matrix = _vec.load_matrix(conn, list(span_claims), encoder_id)
    ranked = _vec.cosine(qblob, matrix, max(len(matrix.vectors), 1))
    deadline_hit = ctx.deadline.expired()

    seen: set[str] = set()
    ordered: list = []
    for span_id, _score in ranked:
        for claim_id in sorted(span_claims.get(span_id, ())):
            if claim_id not in seen:
                seen.add(claim_id)
                ordered.append(claim_id)
    return ordered, warnings, deadline_hit


def lane_dense(ctx: LaneContext) -> LaneResult:
    """Cosine over authorized eligible vectors; absent without an encoder.

    Uses the same embeddings reference path as ``candidates._semantic``
    (pure-Python cosine, numpy optional acceleration) but admits the full
    ranked stream in chunks — ineligible dense hits no longer hide eligible
    ones ranked past the old fixed oversample (V4-28.02). Only a missing
    embeddings table/rows/encoder degrades the lane (V3-28.12).
    """
    result = LaneResult("dense")
    if ctx.cfg is not None and not getattr(ctx.cfg, "dense", True):
        result.status = "skipped"
        result.reason = "disabled_by_config"
        return result
    if not _snap(ctx).has_table("embeddings"):
        result.status = "unavailable"
        result.reason = "no_embedding_index"
        return result
    ordered_ids, warnings, deadline_hit = _semantic_claim_order(ctx)
    if ordered_ids is None:
        result.status = "unavailable"
        result.reason = "encoder_or_rows_unavailable"
        result.warnings = tuple(warnings)
        return result
    eligible = _admit_stream(
        ctx, result, [("claim", cid) for cid in ordered_ids], ctx.lane_cap
    )
    if deadline_hit and result.status == "ok":
        result.status = "partial"
        result.reason = "deadline"
    _rank_eligible(result, eligible, ctx.lane_cap)
    return result


# ---------------------------------------------------------------------------
# declared-but-absent research lanes (§28.06, §28.07, §28.08)
# ---------------------------------------------------------------------------


def _index_manifest_ready(ctx: LaneContext, kind: str) -> bool:
    """Whether a published index artifact exists for ``kind`` (§28.06)."""
    if not _snap(ctx).has_table("index_generations"):
        return False
    row = ctx.conn.execute(
        "SELECT 1 FROM index_generations WHERE kind = ?"
        " AND status IN ('ready','published') LIMIT 1",
        (kind,),
    ).fetchone()
    return row is not None


def lane_sparse(ctx: LaneContext) -> LaneResult:
    """Learned sparse lane: research-to-optional; unavailable without the
    provisioned artifact manifest (V3-28.06)."""
    result = LaneResult("sparse")
    if ctx.cfg is not None and not getattr(ctx.cfg, "sparse", True):
        result.status = "skipped"
        result.reason = "disabled_by_config"
        return result
    if not _index_manifest_ready(ctx, "sparse"):
        result.status = "unavailable"
        result.reason = "sparse_artifacts_not_provisioned"
        return result
    # Manifest present but no scoring implementation in this build —
    # honest unavailability, never a fabricated score (V3-28.12).
    result.status = "unavailable"
    result.reason = "sparse_scoring_not_implemented"
    return result


def lane_late(ctx: LaneResult) -> LaneResult:
    """Late-interaction lane: rerank pool only, artifact-gated (V3-28.06)."""
    result = LaneResult("late")
    if ctx.cfg is not None and not getattr(ctx.cfg, "late_interaction", True):
        result.status = "skipped"
        result.reason = "disabled_by_config"
        return result
    if not _index_manifest_ready(ctx, "late"):
        result.status = "unavailable"
        result.reason = "late_artifacts_not_provisioned"
        return result
    result.status = "unavailable"
    result.reason = "late_scoring_not_implemented"
    return result


# ---------------------------------------------------------------------------
# associative graph expansion (SPEC_V4 §30, V3-28.07)
# ---------------------------------------------------------------------------
#
# The graph lane is an *expansion* lane: it seeds from the eligible hits
# earlier lanes already produced, walks the typed-edge graph inside the same
# authorized scope partition, and emits the expanded candidate set through
# the same admission gate every other lane uses. Traversal can NEVER widen
# authorization — every edge is scope-filtered at the SQL level, and every
# node is admitted through ``union.admit_keys`` before it may emit a hit AND
# before it may act as an intermediate hop, so a quarantined or out-of-scope
# node blocks expansion through itself (V4-30.02).
#
# Composition is structural (V4-30.03/30.08): each typed edge contributes a
# bounded provenance-preserving weight, paths compose multiplicatively, and
# the walk records which edge type produced each hit. This is the honest
# "Holographic-style" reference — typed-edge associative composition over
# five distinguished relation classes — NOT HRR vector binding (no pinned
# atom encoding or numeric backend exists in this build; exact entity
# intersection via :func:`compose_entities` is the V4-04.02 reference).

# Distinguished relation classes (V4-30.03) — labels are never conflated.
REL_OBSERVED_SEQUENCE = "observed_sequence"
REL_CAUSAL_ASSERTION = "explicit_causal_assertion"
REL_CAUSAL_HYPOTHESIS = "inferred_causal_hypothesis"
REL_CONTRADICTION = "contradiction"
REL_TOPIC = "topic_association"

# Edge label -> (relation class, structural binding weight). The weights
# are provenance-preserving constants: a hit reached through a strong typed
# edge outranks one reached through weak co-membership, and the label that
# produced each hit is recorded in LaneResult.details["provenance"].
_GRAPH_RELATIONS: dict[str, tuple] = {
    "supersedes": (REL_OBSERVED_SEQUENCE, 1.0),
    "corrects": (REL_OBSERVED_SEQUENCE, 1.0),
    "precedes": (REL_OBSERVED_SEQUENCE, 0.7),
    "observed_after": (REL_OBSERVED_SEQUENCE, 0.6),
    "derived_from": (REL_OBSERVED_SEQUENCE, 0.7),
    "supports": (REL_CAUSAL_ASSERTION, 0.8),
    "verified_by": (REL_CAUSAL_ASSERTION, 0.9),
    "conflicts_with": (REL_CONTRADICTION, 0.9),
    "conflict_member": (REL_CONTRADICTION, 0.85),
    "context_of": (REL_TOPIC, 0.6),
    "entity": (REL_TOPIC, 0.4),       # co-membership in one entity's claims
    "episode_member": (REL_TOPIC, 0.5),
    "member_of": (REL_TOPIC, 0.5),
    "causal_hypothesis": (REL_CAUSAL_HYPOTHESIS, 0.5),
}

# Spec defaults and hard ceilings (V4-30.09): one hop, ≤100 visited nodes,
# ≤200 examined edges; widened bounds are always explicit caller policy.
GRAPH_DEFAULT_HOPS = 1
GRAPH_DEFAULT_NODES = 100
GRAPH_DEFAULT_EDGES = 200
GRAPH_DEFAULT_FANOUT = 32
GRAPH_HOP_CEILING = 4
GRAPH_NODE_CEILING = 1000
GRAPH_EDGE_CEILING = 2000
GRAPH_FANOUT_CEILING = 256
_GRAPH_CHUNK = 300

# Object kinds a graph walk may land on / expand through — the retrievable
# item plane. Evidence-plane objects (spans, artifacts, envelopes, sources)
# are never landed: they are not lane-emittable candidates, and admitting
# them as intermediates would leak evidence-plane identity into expansion.
_GRAPH_NODE_KINDS = frozenset({
    "claim", "procedure", "episode", "observation", "prospective",
    "transition", "environment_state", "working_item",
})


def _graph_bound(value, default: int, ceiling: int) -> int:
    """Explicit bounds stay bounded: parse a caller/ctx bound, clamp to the
    hard ceiling, never silently widen (V4-30.09)."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, min(v, ceiling))


def _graph_edges_for(ctx: LaneContext, frontier: set, remaining: int,
                     snap) -> list:
    """Fetch typed edges touching ``frontier``, scope-filtered, ≤ remaining.

    Returns ``(src_key, dst_key, label, via, source_rank)`` rows in a
    deterministic order. ``via`` carries provenance (entity id, episode id,
    conflict group id, edge id). ``source_rank`` stabilizes the merge order
    across the six edge sources.
    """
    conn = ctx.conn
    scopes = list(ctx.scope_ids)
    sph = _ph(len(scopes))
    by_kind: dict[str, list] = {}
    for kind, oid in frontier:
        by_kind.setdefault(kind, []).append(oid)
    rows: list = []

    def budget() -> int:
        return max(0, remaining - len(rows))

    def frontier_pred(alias_kind: str, alias_id: str) -> tuple[str, list]:
        """``(kind='?' AND id IN (...)) OR ...`` over frontier kinds."""
        parts, params = [], []
        for kind in sorted(by_kind):
            ids = sorted(by_kind[kind])
            for i in range(0, len(ids), _GRAPH_CHUNK):
                chunk = ids[i:i + _GRAPH_CHUNK]
                parts.append(
                    f"({alias_kind} = ? AND {alias_id} IN ({_ph(len(chunk))}))"
                )
                params.extend([kind, *chunk])
        return " OR ".join(parts) if parts else "0", params

    # 1) typed edges table — both directions (traversal is undirected;
    #    direction is recorded, not inferred).
    if budget() > 0 and snap.has_table("edges"):
        pred, params = frontier_pred("source_kind", "source_id")
        pred2, params2 = frontier_pred("target_kind", "target_id")
        for r in conn.execute(
            "SELECT edge_id, edge_type, source_kind, source_id,"
            "       target_kind, target_id FROM edges"
            " WHERE retired_event IS NULL"
            f" AND scope_id IN ({sph})"
            f" AND ({pred} OR {pred2})"
            " ORDER BY created_event, edge_id LIMIT ?",
            [*scopes, *params, *params2, budget()],
        ).fetchall():
            edge_id, etype, sk, sid, tk, tid = r
            src, dst = (sk, sid), (tk, tid)
            if src in frontier:
                rows.append((src, dst, etype, edge_id, 0))
            if dst in frontier and dst != src:
                rows.append((dst, src, etype, edge_id, 0))

    # 2) derivations — immutable provenance edges (§06.06). Parent/child
    #    revisions are collapsed to id-level association; the label stays
    #    ``derived_from`` with direction carried in ``via``.
    if budget() > 0 and snap.has_table("derivations"):
        pred, params = frontier_pred("parent_kind", "parent_id")
        pred2, params2 = frontier_pred("child_kind", "child_id")
        seen_dr: set = set()
        for r in conn.execute(
            "SELECT child_kind, child_id, parent_kind, parent_id, seq"
            " FROM derivations"
            f" WHERE scope_id IN ({sph}) AND ({pred} OR {pred2})"
            " ORDER BY seq LIMIT ?",
            [*scopes, *params, *params2, budget()],
        ).fetchall():
            ck, cid, pk, pid, seq = r
            child, parent = (ck, cid), (pk, pid)
            if parent in frontier and (parent, child, "up") not in seen_dr:
                seen_dr.add((parent, child, "up"))
                rows.append((parent, child, "derived_from",
                             f"up:{seq}", 1))
            if child in frontier and (child, parent, "down") not in seen_dr:
                seen_dr.add((child, parent, "down"))
                rows.append((child, parent, "derived_from",
                             f"down:{seq}", 1))

    # 3) entity co-membership — claims sharing an authorized entity. The
    #    entity node itself is validated (in-scope, not held) before its
    #    mediated edge counts (V4-30.02).
    claim_ids = sorted(by_kind.get("claim") or ())
    if budget() > 0 and claim_ids and snap.has_table("claim_entities"):
        ent_budget = budget()
        ent_rows: list = []
        for i in range(0, len(claim_ids), _GRAPH_CHUNK):
            if len(ent_rows) >= ent_budget:
                break
            chunk = claim_ids[i:i + _GRAPH_CHUNK]
            cph = _ph(len(chunk))
            ent_rows.extend(conn.execute(
                "SELECT ce1.claim_id, ce2.claim_id, ce1.entity_id,"
                "       e.row_version"
                " FROM claim_entities ce1"
                " JOIN entities e ON e.entity_id = ce1.entity_id"
                f"  AND e.scope_id IN ({sph})"
                " JOIN claim_entities ce2"
                "   ON ce2.entity_id = ce1.entity_id"
                "  AND ce2.claim_id <> ce1.claim_id"
                " JOIN claims c2 ON c2.claim_id = ce2.claim_id"
                f"  AND c2.scope_id IN ({sph})"
                f" WHERE ce1.claim_id IN ({cph})"
                " ORDER BY ce1.entity_id, ce2.claim_id LIMIT ?",
                [*scopes, *scopes, *chunk, ent_budget - len(ent_rows)],
            ).fetchall())
        # Held entity nodes mediate nothing.
        held = set()
        try:
            held = set(snap.held_refs(
                ("entity", str(r[2]), int(r[3])) for r in ent_rows))
        except Exception:
            held = set()
        for src_id, dst_id, eid, erv in ent_rows:
            if ("entity", str(eid), int(erv)) in held:
                continue
            rows.append((("claim", src_id), ("claim", dst_id),
                         "entity", str(eid), 2))

    # 4) episode membership — episode ↔ member, live memberships only.
    ep_ids = sorted(by_kind.get("episode") or ())
    if budget() > 0 and snap.has_table("episode_members") and (
            ep_ids or any(k in _GRAPH_NODE_KINDS for k in by_kind)):
        conds, mparams = [], []
        if ep_ids:
            conds.append(f"em.episode_id IN ({_ph(len(ep_ids))})")
            mparams.extend(ep_ids)
        pred_ob, params_ob = frontier_pred("em.object_kind", "em.object_id")
        if pred_ob != "0":
            conds.append(f"({pred_ob})")
            mparams.extend(params_ob)
        for r in conn.execute(
            "SELECT em.episode_id, em.object_kind, em.object_id"
            " FROM episode_members em"
            " JOIN episodes ep ON ep.episode_id = em.episode_id"
            f"  AND ep.scope_id IN ({sph})"
            " WHERE em.recorded_until IS NULL"
            f"  AND ({' OR '.join(conds)})"
            " ORDER BY em.episode_id, em.ord, em.object_kind, em.object_id"
            " LIMIT ?",
            [*scopes, *mparams, budget()],
        ).fetchall():
            epid, okind, oid = r
            ep_key, obj_key = ("episode", epid), (okind, oid)
            if ep_key in frontier:
                rows.append((ep_key, obj_key, "episode_member", epid, 3))
            if obj_key in frontier and okind in _GRAPH_NODE_KINDS:
                rows.append((obj_key, ep_key, "member_of", epid, 3))

    # 5) transitions — episode ↔ transition typed by transitions.edge.
    if budget() > 0 and snap.has_table("transitions"):
        ep_ids = sorted(by_kind.get("episode") or ())
        tr_ids = sorted(by_kind.get("transition") or ())
        if ep_ids or tr_ids:
            conds, tparams = [], []
            if ep_ids:
                conds.append(f"episode_id IN ({_ph(len(ep_ids))})")
                tparams.extend(ep_ids)
            if tr_ids:
                conds.append(f"transition_id IN ({_ph(len(tr_ids))})")
                tparams.extend(tr_ids)
            for r in conn.execute(
                "SELECT episode_id, transition_id, edge FROM transitions"
                f" WHERE scope_id IN ({sph}) AND ({' OR '.join(conds)})"
                " ORDER BY episode_id, ord LIMIT ?",
                [*scopes, *tparams, budget()],
            ).fetchall():
                epid, tid, tedge = r
                ep_key, tr_key = ("episode", epid), ("transition", tid)
                label = str(tedge) if str(tedge) in _GRAPH_RELATIONS \
                    else "observed_after"
                if ep_key in frontier:
                    rows.append((ep_key, tr_key, label, tid, 4))
                if tr_key in frontier:
                    rows.append((tr_key, ep_key, label, epid, 4))

    # 6) open conflict groups — disputed claims co-members expand as
    #    contradiction evidence.
    if budget() > 0 and claim_ids and snap.has_table("conflict_members"):
        cg_budget = budget()
        cg_rows: list = []
        for i in range(0, len(claim_ids), _GRAPH_CHUNK):
            if len(cg_rows) >= cg_budget:
                break
            chunk = claim_ids[i:i + _GRAPH_CHUNK]
            cph = _ph(len(chunk))
            cg_rows.extend(conn.execute(
                "SELECT cm1.claim_id, cm2.claim_id, cm1.group_id"
                " FROM conflict_members cm1"
                " JOIN conflict_groups g ON g.group_id = cm1.group_id"
                "   AND g.status = 'open'"
                f"  AND g.scope_id IN ({sph})"
                " JOIN conflict_members cm2 ON cm2.group_id = cm1.group_id"
                "  AND cm2.claim_id <> cm1.claim_id"
                " JOIN claims c2 ON c2.claim_id = cm2.claim_id"
                f"  AND c2.scope_id IN ({sph})"
                f" WHERE cm1.claim_id IN ({cph})"
                " ORDER BY cm1.group_id, cm2.claim_id LIMIT ?",
                [*scopes, *scopes, *chunk, cg_budget - len(cg_rows)],
            ).fetchall())
        for src_id, dst_id, gid in cg_rows:
            rows.append((("claim", src_id), ("claim", dst_id),
                         "conflict_member", str(gid), 5))

    rows.sort(key=lambda r: (r[4], r[0][0], r[0][1], r[2], r[1][0], r[1][1]))
    return rows


def graph_expand(
    ctx: LaneContext,
    seeds,
    *,
    hops: Optional[int] = None,
    max_nodes: Optional[int] = None,
    max_edges: Optional[int] = None,
    fanout: Optional[int] = None,
    target=None,
) -> tuple[list, dict, dict]:
    """Bounded multi-source BFS over the typed-edge graph (V4-30.01/09).

    ``seeds`` are ``(kind, oid)`` keys already produced by flat lanes.
    Returns ``(ordered_keys, provenance, stats)``:

    - ``ordered_keys``: expanded keys (seeds excluded), ordered by
      ``(depth asc, path weight desc, kind, id)`` — deterministic.
    - ``provenance``: ``key -> {"via", "relation", "class", "depth",
      "weight", "from"}`` — the strongest edge that reached the node.
    - ``stats``: ``{"visited", "examined_edges", "hops_used",
      "bounded_by", "relations"}`` — honest bound accounting.

    Every visited node is admitted through ``union.admit_keys`` before it
    may emit or expand (V4-30.02): an ineligible intermediate blocks the
    path through itself. ``target`` (optional) stops the walk early once
    reached — the ``why_related`` probe.
    """
    from . import union as _union  # local import: no module cycle

    snap = _snap(ctx)
    hops = _graph_bound(hops if hops is not None else ctx.graph_hops,
                        GRAPH_DEFAULT_HOPS, GRAPH_HOP_CEILING)
    max_nodes = _graph_bound(max_nodes if max_nodes is not None
                             else ctx.graph_max_nodes,
                             GRAPH_DEFAULT_NODES, GRAPH_NODE_CEILING)
    max_edges = _graph_bound(max_edges if max_edges is not None
                             else ctx.graph_max_edges,
                             GRAPH_DEFAULT_EDGES, GRAPH_EDGE_CEILING)
    fanout = _graph_bound(fanout if fanout is not None else ctx.graph_fanout,
                          GRAPH_DEFAULT_FANOUT, GRAPH_FANOUT_CEILING)

    seed_keys = [
        (k, o) for (k, o) in (seeds or ()) if k in _GRAPH_NODE_KINDS
    ]
    stats = {"visited": 0, "examined_edges": 0, "hops_used": 0,
             "bounded_by": None, "relations": {}}
    prov: dict = {}
    if not seed_keys or hops <= 0 or max_nodes <= 0 or max_edges <= 0:
        return [], prov, stats

    # Seeds are already-admitted lane hits; re-verifying through the same
    # gate is cheap (memoized) and keeps the traversal honest when it runs
    # standalone (why_related/related probes).
    admitted = set(_union.admit_keys(ctx, seed_keys))
    visited = {k for k in seed_keys if k in admitted}
    frontier = set(visited)
    stats["visited"] = len(visited)
    found: dict = {}  # key -> (depth, weight)

    for depth in range(1, hops + 1):
        if not frontier:
            break
        if ctx.deadline is not None and ctx.deadline.expired():
            stats["bounded_by"] = "deadline"
            break
        remaining = max_edges - stats["examined_edges"]
        if remaining <= 0:
            stats["bounded_by"] = "edges"
            break
        edge_rows = _graph_edges_for(ctx, frontier, remaining, snap)
        stats["examined_edges"] += len(edge_rows)
        if len(edge_rows) >= remaining:
            stats["bounded_by"] = "edges"
        stats["hops_used"] = depth

        fanout_used: dict = {}
        novel: list = []
        for src, dst, label, via, _rank in edge_rows:
            if dst[0] not in _GRAPH_NODE_KINDS or dst in visited:
                continue
            if len(visited) >= max_nodes:
                stats["bounded_by"] = stats["bounded_by"] or "nodes"
                break
            used = fanout_used.get(src, 0)
            if used >= fanout:
                stats["bounded_by"] = stats["bounded_by"] or "fanout"
                continue
            fanout_used[src] = used + 1
            visited.add(dst)
            novel.append((src, dst, label, via))
        if stats["bounded_by"] == "nodes":
            stats["visited"] = len(visited)

        # Admit BEFORE expanding: an ineligible node never lands and never
        # acts as an intermediate hop (V4-30.02).
        novel_keys = [dst for (_s, dst, _l, _v) in novel]
        admitted_round = set(_union.admit_keys(ctx, novel_keys))
        next_frontier = set()
        for src, dst, label, via in novel:
            if dst not in admitted_round:
                continue  # quarantined/out-of-scope node: blocked
            next_frontier.add(dst)
            cls, w = _GRAPH_RELATIONS.get(label, (REL_TOPIC, 0.3))
            depth_w = round(w * (0.9 ** (depth - 1)), 6)
            cur = found.get(dst)
            if cur is None or (depth, -depth_w) < (cur[0], -cur[1]):
                found[dst] = (depth, depth_w)
                prov[dst] = {
                    "via": label, "relation": label, "class": cls,
                    "depth": depth, "weight": depth_w,
                    "from_kind": src[0], "from_id": src[1],
                    "through": via,
                }
                stats["relations"][cls] = stats["relations"].get(cls, 0) + 1
        stats["visited"] = len(visited)
        frontier = next_frontier
        if target is not None and target in found:
            break

    ordered = sorted(
        found, key=lambda k: (found[k][0], -found[k][1], k[0], k[1])
    )
    return ordered, prov, stats


def lane_graph(ctx: LaneContext) -> LaneResult:
    """Bounded propagation from authorized flat-lane seeds (V3-28.07,
    V4-30.01/02/09).

    Off until ablation by default — ``cfg.v3.retrieval.graph`` enables the
    bounded expansion walk. Defaults: one hop, ≤100 visited nodes, ≤200
    examined edges, ≤32 fanout/node — the controller's declared tier hop
    budget supplies ``graph_hops``; wider bounds must be declared
    explicitly, never assumed. Expansion is structural typed-edge
    composition with provenance labels (``details["composition"] ==
    "structural"``); it is not HRR vector binding.
    """
    result = LaneResult("graph")
    cfg = ctx.cfg
    if not (cfg is not None and getattr(cfg, "graph", False)):
        result.status = "skipped"
        result.reason = "off_until_ablation"
        return result
    hops = _graph_bound(ctx.graph_hops, GRAPH_DEFAULT_HOPS,
                        GRAPH_HOP_CEILING)
    if hops <= 0:
        result.status = "skipped"
        result.reason = "no_hop_budget"
        return result
    seeds = sorted(ctx.seeds)
    if not seeds:
        result.status = "skipped"
        result.reason = "no_authorized_seeds"
        return result

    ordered, prov, stats = graph_expand(ctx, seeds)
    result.attempted = stats["examined_edges"]
    result.details = {
        "composition": "structural",
        "binding": "typed_edge",
        "bounds": {
            "hops": hops,
            "max_nodes": _graph_bound(ctx.graph_max_nodes,
                                      GRAPH_DEFAULT_NODES,
                                      GRAPH_NODE_CEILING),
            "max_edges": _graph_bound(ctx.graph_max_edges,
                                      GRAPH_DEFAULT_EDGES,
                                      GRAPH_EDGE_CEILING),
            "fanout": _graph_bound(ctx.graph_fanout,
                                   GRAPH_DEFAULT_FANOUT,
                                   GRAPH_FANOUT_CEILING),
        },
        "visited_nodes": stats["visited"],
        "examined_edges": stats["examined_edges"],
        "hops_used": stats["hops_used"],
        "bounded_by": stats["bounded_by"],
        "relations": dict(stats["relations"]),
        "provenance": {
            f"{k}:{o}": prov[(k, o)]
            for (k, o) in ordered[: ctx.lane_cap]
        },
    }
    if stats["bounded_by"]:
        result.status = "partial"
        result.reason = f"bounded_by:{stats['bounded_by']}"
        result.warnings += (f"graph_bound:{stats['bounded_by']}",)
    if not ordered:
        if result.status == "ok":
            result.status = "skipped"
            result.reason = "no_expansion"
        return result
    _rank_eligible(result, ordered, ctx.lane_cap)
    return result


# -- §30 reference operations (V4-30.01 deterministic SQL implementations) --
#
# These are the lane's own primitives exposed for callers/tests: each runs
# on a LaneContext (same snapshot, same authorized scopes, same admission
# gate). They compose typed edges structurally — none fabricates a vector
# binding.


def probe_entity(ctx: LaneContext, entity_id: str, *, limit: int = 40
                 ) -> list:
    """``probe(entity)``: authorized claims bound to one entity id."""
    scopes = list(ctx.scope_ids)
    if not scopes or not entity_id:
        return []
    rows = ctx.conn.execute(
        "SELECT ce.claim_id FROM claim_entities ce"
        " JOIN entities e ON e.entity_id = ce.entity_id"
        f"  AND e.scope_id IN ({_ph(len(scopes))})"
        " JOIN claims c ON c.claim_id = ce.claim_id"
        f"  AND c.scope_id IN ({_ph(len(scopes))})"
        " WHERE ce.entity_id = ? ORDER BY ce.claim_id LIMIT ?",
        [*scopes, *scopes, entity_id, max(1, int(limit))],
    ).fetchall()
    from . import union as _union
    admitted = _union.admit_keys(ctx, [("claim", r[0]) for r in rows])
    return [r[0] for r in rows if ("claim", r[0]) in admitted]


def compose_entities(ctx: LaneContext, entity_ids, *, limit: int = 40
                     ) -> list:
    """``compose(entity_set)``: exact intersection — claims linked to ALL
    given entities (the V4-04.02 reference composition)."""
    ids = list(dict.fromkeys(entity_ids or ()))
    scopes = list(ctx.scope_ids)
    if not ids or not scopes:
        return []
    joins = " ".join(
        f"JOIN claim_entities ce{i} ON ce{i}.claim_id = c.claim_id"
        f" AND ce{i}.entity_id = ?" for i in range(len(ids))
    )
    rows = ctx.conn.execute(
        f"SELECT c.claim_id FROM claims c {joins}"
        f" WHERE c.scope_id IN ({_ph(len(scopes))})"
        " ORDER BY c.claim_id LIMIT ?",
        [*ids, *scopes, max(1, int(limit))],
    ).fetchall()
    from . import union as _union
    admitted = _union.admit_keys(ctx, [("claim", r[0]) for r in rows])
    return [r[0] for r in rows if ("claim", r[0]) in admitted]


def compose_report(ctx: LaneContext, entity_ids, *, limit: int = 40,
                   hops: Optional[int] = None) -> dict:
    """X8 honest-composition report (SPEC_V4_5 §09 X8, V45-09.08, D14;
    V4-30.01, V4-04.02).

    Every associative candidate is measured AGAINST the exact
    intersection — never instead of it. ``compose(entity_set)`` is the
    reference: claims linked to ALL given entities. The associative set
    unions each entity's authorized claims (typed-edge co-membership —
    the same plane a HRR candidate accelerator would target) plus, when
    ``hops`` is given, a bounded ``graph_expand`` walk seeded from those
    claims.

    The report separates:

    - ``verified``: candidates the exact intersection confirms — the
      only set that may be treated as composed fact;
    - ``associative_only``: candidates exact intersection does NOT
      confirm — candidate accelerators, never minted facts
      (V45-09.08: decoding/association cannot create truth);
    - ``exact``: the reference intersection itself.

    No claim outside ``verified`` is ever presented as semantic truth;
    the report keeps both denominators explicit for ablation accounting.
    """
    ids = list(dict.fromkeys(entity_ids or ()))
    exact = compose_entities(ctx, ids, limit=limit)
    assoc: set = set()
    for eid in ids:
        assoc.update(probe_entity(ctx, eid, limit=limit))
    seeded = sorted(assoc)
    if hops:
        seeds = [("claim", c) for c in seeded]
        ordered, _prov, _stats = graph_expand(ctx, seeds, hops=hops)
        assoc.update(cid for (k, cid) in ordered if k == "claim")
    exact_set = set(exact)
    assoc_only = sorted(c for c in assoc if c not in exact_set)
    return {
        "reference": "exact_intersection",
        "entities": ids,
        "exact": list(exact),
        "verified": sorted(c for c in assoc if c in exact_set),
        "associative_only": assoc_only,
        "n_exact": len(exact),
        "n_associative": len(assoc),
        "n_associative_only": len(assoc_only),
        "binding": "structural_typed_edge",
        "composition": "structural",
        "note": (
            "associative candidates are reported against the exact "
            "intersection (V4-04.02), never instead of it — a candidate "
            "accelerator cannot mint facts (V45-09.08); HRR remains "
            "research until it beats this reference plus hybrid "
            "retrieval (V4-30.10)"
        ),
    }


def related(ctx: LaneContext, key, *, hops: Optional[int] = None) -> list:
    """``related(object)``: bounded typed expansion from one seed —
    ``[(key, provenance)]`` in deterministic rank order."""
    ordered, prov, _stats = graph_expand(ctx, [tuple(key)], hops=hops)
    return [(k, prov[k]) for k in ordered]


def contradictions(ctx: LaneContext, key, *, limit: int = 40) -> list:
    """``contradictions(object)``: conflicting claims — live
    ``conflicts_with`` edges plus open conflict-group co-members."""
    kind, oid = key
    scopes = list(ctx.scope_ids)
    if kind != "claim" or not scopes:
        return []
    snap = _snap(ctx)
    cands: list = []
    if snap.has_table("edges"):
        rows = ctx.conn.execute(
            "SELECT source_id, target_id FROM edges"
            " WHERE edge_type = 'conflicts_with' AND retired_event IS NULL"
            f"  AND scope_id IN ({_ph(len(scopes))})"
            "   AND ((source_kind = 'claim' AND source_id = ?)"
            "    OR (target_kind = 'claim' AND target_id = ?))",
            [*scopes, oid, oid],
        ).fetchall()
        cands.extend(r[0] if r[1] == oid else r[1] for r in rows)
    if snap.has_table("conflict_members"):
        rows = ctx.conn.execute(
            "SELECT cm2.claim_id FROM conflict_members cm1"
            " JOIN conflict_groups g ON g.group_id = cm1.group_id"
            "   AND g.status = 'open'"
            f"  AND g.scope_id IN ({_ph(len(scopes))})"
            " JOIN conflict_members cm2 ON cm2.group_id = cm1.group_id"
            "   AND cm2.claim_id <> cm1.claim_id"
            " WHERE cm1.claim_id = ? ORDER BY cm2.claim_id LIMIT ?",
            [*scopes, oid, max(1, int(limit))],
        ).fetchall()
        cands.extend(r[0] for r in rows)
    cands = sorted(set(cands))[: max(1, int(limit))]
    from . import union as _union
    admitted = _union.admit_keys(ctx, [("claim", c) for c in cands])
    return [c for c in cands if ("claim", c) in admitted]


def why_related(ctx: LaneContext, a, b,
                *, hops: Optional[int] = None) -> Optional[list]:
    """``why_related(a, b)``: one authorized path from ``a`` to ``b`` as
    ``[(key, edge_label, relation_class)]`` hops, or ``None`` when no
    admitted path exists within the bounds. ``None`` is honest — it never
    fabricates relatedness."""
    a, b = tuple(a), tuple(b)
    if a == b:
        return [(a, "self", REL_TOPIC)]
    _ordered, prov, _stats = graph_expand(ctx, [a], hops=hops, target=b)
    if b not in prov:
        return None
    # Walk the provenance chain backwards from b to a.
    path = [b]
    while path[-1] != a:
        p = prov.get(path[-1])
        if p is None:
            return None
        parent = (p["from_kind"], p["from_id"])
        if parent in path:
            return None  # defensive: never loop
        path.append(parent)
    path.reverse()
    out = [(a, "seed", REL_TOPIC)]
    for k in path[1:]:
        p = prov[k]
        out.append((k, p["via"], p["class"]))
    return out


def lane_causal(ctx: LaneContext) -> LaneResult:
    """Cited transition chains within an episode (V3-28.08).

    Off until ablation: returns cited ``transitions`` rows inside
    authorized episodes when ``cfg.v3.retrieval.causal`` is set; the lane
    never synthesizes causes.
    """
    result = LaneResult("causal")
    cfg = ctx.cfg
    if not (cfg is not None and getattr(cfg, "causal", False)):
        result.status = "skipped"
        result.reason = "off_until_ablation"
        return result
    if not _snap(ctx).has_table("transitions"):
        result.status = "unavailable"
        result.reason = "no_transition_table"
        return result
    seed_eps = sorted(
        {oid for (kind, oid) in ctx.seeds if kind == "episode"}
    )
    if not seed_eps:
        result.status = "skipped"
        result.reason = "no_episode_seeds"
        return result
    state = {"rows": 0}

    def fetch(scale: int):
        limit = ctx.lane_cap * 4 * scale
        rows = ctx.conn.execute(
            "SELECT transition_id FROM transitions"
            f" WHERE episode_id IN ({_ph(len(seed_eps))})"
            f" AND scope_id IN ({_ph(len(ctx.scope_ids))})"
            " ORDER BY ord, transition_id LIMIT ?",
            [*seed_eps, *ctx.scope_ids, limit],
        ).fetchall()
        delta = max(0, len(rows) - state["rows"])
        state["rows"] = len(rows)
        return delta, len(rows) < limit, [
            ("transition", r[0]) for r in rows
        ]

    eligible = _paged(ctx, result, fetch, ctx.lane_cap)
    _rank_eligible(result, eligible, ctx.lane_cap)
    return result


def lane_browse(ctx: LaneContext) -> LaneResult:
    """Bounded authorized listing for archive/timeline queries (§26).

    The listing pages (V4-28.02): ineligible rows (held, wrong lifecycle
    for the query class) consume fetch budget, never rank capacity.
    """
    result = LaneResult("browse")
    state = {"rows": 0}

    def fetch(scale: int):
        limit = ctx.lane_cap * 4 * scale
        rows = ctx.conn.execute(
            "SELECT claim_id FROM claims"
            f" WHERE scope_id IN ({_ph(len(ctx.scope_ids))})"
            " ORDER BY created_event DESC, claim_id LIMIT ?",
            [*ctx.scope_ids, limit],
        ).fetchall()
        delta = max(0, len(rows) - state["rows"])
        state["rows"] = len(rows)
        return delta, len(rows) < limit, [("claim", r[0]) for r in rows]

    eligible = _paged(ctx, result, fetch, ctx.lane_cap)
    _rank_eligible(result, eligible, ctx.lane_cap)
    return result


# ---------------------------------------------------------------------------
# registry + runner
# ---------------------------------------------------------------------------

LaneFn = Callable[[LaneContext], LaneResult]

LANES: dict[str, LaneFn] = {
    "exact_id": lane_exact_id,
    "lexical": lane_lexical,
    "structured": lane_structured,
    "temporal": lane_temporal,
    "procedural_signature": lane_procedural_signature,
    "episode_hierarchy": lane_episode_hierarchy,
    "working": lane_working,
    "freshness_env": lane_freshness_env,
    "dense": lane_dense,
    "sparse": lane_sparse,
    "late": lane_late,
    "graph": lane_graph,
    "causal": lane_causal,
    "browse": lane_browse,
}

# Errors that must abort the whole recall (fail closed, V3-28.12):
# authorization denial, integrity failures, epoch fencing.
_FAIL_CLOSED = frozenset({
    ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
    ErrorCode.INTEGRITY,
    ErrorCode.STALE_EPOCH,
    ErrorCode.LOCKED,
})


def run_lanes(ctx: LaneContext, lane_names: list) -> LaneRunReport:
    """Run the selected lanes in registry order; collect LaneResults.

    Sequential execution on the caller's one read snapshot — reported
    truthfully as ``concurrency="sequential"`` on the returned
    :class:`LaneRunReport` and stamped on every LaneResult (V4-27.07,
    F4-13). Parallel workers would each need an independent
    ``store.read()`` snapshot, which can diverge from the pinned
    ``ctx.generation`` mid-run, and seed-dependent lanes
    (graph/causal/episode_hierarchy) deliberately consume the seeds
    accumulated by earlier lanes in ``LANE_ORDER`` — serial execution on
    one snapshot is the semantics the pipeline is built on.

    A lane raising an unexpected error reports ``degraded`` (local
    degradation); lanes carrying authorization/integrity error codes
    propagate — retrieval must fail closed, never silently absorb them
    (V3-28.12). Skipped lanes (not selected, or cut by the deadline)
    remain visible in diagnostics (V3-26.10, V3-28.03).
    """
    results = LaneRunReport(concurrency="sequential")
    wanted = set(lane_names)
    # One per-snapshot eligibility/schema memo shared by every lane —
    # admission verdicts and schema probes resolve once per recall, never
    # carried into a later snapshot (the cache lives on this ctx).
    snap = _snap(ctx)
    for name in LANE_ORDER:
        if name not in wanted:
            continue
        fn = LANES[name]
        cap = min(ctx.lane_cap, LANE_CAPS.get(name, ctx.lane_cap))
        if ctx.deadline.expired():
            results.append(LaneResult(
                name, status="skipped", reason="deadline",
                concurrency="sequential",
            ))
            continue
        lane_ctx = LaneContext(
            conn=ctx.conn,
            store=ctx.store,
            request=ctx.request,
            plan=ctx.plan,
            query_class=ctx.query_class,
            scope_ids=ctx.scope_ids,
            generation=ctx.generation,
            deadline=ctx.deadline,
            lane_cap=cap,
            seeds=dict(ctx.seeds),
            cfg=ctx.cfg,
            graph_hops=ctx.graph_hops,
            graph_fanout=ctx.graph_fanout,
            graph_max_nodes=ctx.graph_max_nodes,
            graph_max_edges=ctx.graph_max_edges,
            snap=snap,
        )
        try:
            res = fn(lane_ctx)
        except VerbatimError as exc:
            if exc.code in _FAIL_CLOSED:
                raise
            res = LaneResult(
                name, status="degraded", reason=f"error:{exc.code.value}",
            )
        except sqlite3.Error:
            res = LaneResult(name, status="degraded", reason="sqlite")
        except Exception:
            res = LaneResult(name, status="degraded", reason="internal")
        res.concurrency = "sequential"
        results.append(res)
        ctx.seeds.update(res.hits)
    # Selected-but-missing lanes (empty results) still appear — callers see
    # attempted/skipped/degraded honestly per V3-28.03.
    return results
