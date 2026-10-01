"""V7 scope lane — speaker/entity scoped candidate seed (V75-04.01).

**What it is.**  When the query resolves to exactly one *subject* canon —
a resolved entity canon that is also a speaker canon present in this
scope — the lane emits the eligible units **spoken by** that canon
(``units.speaker_canon``) or **canonically mentioning** it
(``entity_mentions.canon``).  That is V75-04.01's scoped retrieval path:
"a candidate seed … whose output is fused with the unscoped lanes."

**Never a gate.**  The lane is additive only: it removes nothing and
filters no other lane's output.  The real-data split behind the
requirement (V75-01.2: 94.6% of single-speaker questions have all gold
evidence spoken by the named speaker, but 2.9% have none) is exactly why
scope must be a seed, not a mask — a hard speaker mask would zero ~3% of
LoCoMo answerable questions outright.  Fusion unions this lane's ranked
list with the unscoped lanes; a unit outside the scope stays retrievable
through them (J10).

**Abstention.**  The lane MUST run unscoped — emit nothing — when the
resolver is ambiguous:

- ``scope = "none"`` — zero subject canons resolved (no speaker canon of
  the query is present in the scope), or the query names no entity at
  all.  Status ``skipped`` / ``no_subject_canon``.
- ``scope = "ambiguous"`` — two or more subject canons resolve (e.g.
  "What did Caroline say to Melanie?" names both speakers).  Status
  ``skipped`` / ``ambiguous_subject_canons``.
- ``scope = "resolved"`` — exactly one; ``stats["scope_canon"]`` carries
  the canon id.  Status ``ok`` (``partial`` on a cut scan).

Coverage therefore reports ``scope ∈ {resolved, ambiguous, none}`` and
the canon, per V75-04.01, on every call — the stat is initialized to
``"none"`` before any early exit.

**Subject resolution.**  The lane uses the *same* resolver the
V75-03.05 ``speaker_match`` feature uses —
``entity.resolve_query_speaker_canons``: ``qv.entity_canons`` re-folded
through ``entities_v2.canon`` (idempotent — a raw surface like
``"Caroline's"`` lands on the same key, D7-01/02) then probed against
``units.speaker_canon`` at/below the generation fence, sorted and
deduplicated.  A canon counts as a subject only when it is a *speaker
canon present in the scope*; a resolved topic canon ("the promotion")
is not a subject — it neither scopes nor adds ambiguity.  A
caller-supplied ``qv.speaker_canon`` hint wins verbatim — the
``speaker_match`` feature's own precedence — even when the canon has
no units in scope (the caller declared the scope; an empty pool then
reports honestly).  Because the resolved population is produced by the
shared resolver, the Q1 scope-form arms (feature-only / scoped lane /
hard mask) all measure the same ``{resolved, ambiguous, none}`` split.
``stats["scope_source"]`` reports ``hint`` or ``query`` provenance.

**Scoring.**  Candidates are scored by the shared §32.2 BM25F machinery
(``lexical._collect_postings`` / ``lexical._score_candidates``) with
``N``, ``df``, and ``avglen`` computed over the **full request-eligible
set** — never the scoped subset — so raw scores stay calibrated against
L-lex in the fused pool (V75-04.01's "scored honestly"; V7-05.04's raw
signals).  In-scope units with no query-term match are still emitted at
their measured score (0.0, ``matched_terms={}``): the lane is a
candidate *seed*, not only a re-ranker — an in-scope paraphrase mismatch
that lexical missed must be able to surface.

**Eligibility before rank (V7-05.08).**  Scope membership is resolved
through the generation-fenced unit universe and intersected with the
caller's eligible set *before* scoring or emission; ineligible scope
members are counted in ``stats["ineligible"]`` and never emitted.  The
eligibility handle itself is never inspected for scoping and never
widened.

**Generation fence (V7-30.02, f2dd666 convention).**  All unit reads go
through ``lexical._universe`` — ``scope_id`` + ``generation <= pinned``,
latest row per ``unit_id`` — which is also where ``speaker_canon``
membership comes from (``need_full=True`` so the full unit row is in
hand for callable eligibility and for the ``speaker_canon`` unit-fact
signal fusion merges for ``speaker_match``).  ``entity_mentions``
postings are read with the same fence; mentions of units with no visible
row at/below the pin are orphaned — counted, never emitted.

**LaneName shim.**  ``types_v7.LaneName`` is contract-frozen
(docs/v7_contracts.md) and has no ``SCOPE`` member, so this module mints
a ``LaneName`` *instance* carrying ``"scope"`` (:data:`LANE_ENUM`).
LaneName is a ``str`` enum: the instance hashes and compares as
``"scope"`` and passes every ``LaneName(x)`` coercion site
(``register_lane``, ``run_one``, ``pipeline._policy_lanes``,
``deadline.allocate``, ``ablation_lanes``) unchanged.  If the frozen
enum ever grows a real member, ``_lane_enum`` returns it instead.
Caveat, disclosed: the minted instance is not a real member — it is
absent from ``list(LaneName)``/``LaneName.__members__`` and does not
pickle (nothing on the read path does either).

**Enablement — the Q1 arm knob.**  Default OFF: the lane is absent from
``policy.LANES_V1``, so it runs only when a declared policy table names
``"scope"`` in ``lanes`` (``load_policy(profile, {"lanes": […,
"scope"]})`` resolves it through ``policy._extension_lane``) or a
``RetrievalPolicyV7`` is built with ``LANE_ENUM`` directly.  It never
auto-enables, and it reports ``unavailable/not_registered`` through the
standard machinery if its module fails to import.

All constants are ``provisional/v7-r0``; the lane is arm machinery for
the Q1 scope-form measurement, not a selected formula.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

from ...core.types_v7 import (
    FORMULA_STATUS_PROVISIONAL,
    CandidateV7,
    LaneContextV7,
    LaneName,
    LaneOutput,
    LaneSlice,
    LaneStatus,
    LaneV7,
    POOLS,
    QueryViewV7,
)
from ...enrichment.entities_v2 import canon as _canon
from ...storage.repos import has_table
from . import entity as _ent
from . import lexical as _lex

LANE_NAME = "scope"  # stable coverage key — LaneOutput.lane / stats
LANE_VALUE = LANE_NAME
LANE_VERSION = "scope_lane/v75-r0"
FORMULA_STATUS = FORMULA_STATUS_PROVISIONAL

UNITS_TABLE = _lex.UNITS_TABLE  # "units"
MENTIONS_TABLE = "entity_mentions"

_MENTION_PAGE = 1024          # fetchmany page; deadline checked per page
_MENTION_ROW_LIMIT = 100_000  # declared scan bound → partial/scan_bound


def _lane_enum() -> LaneName:
    """The scope lane's ``LaneName`` identity (module docstring §"LaneName
    shim").

    Prefers a real enum member if the frozen contract ever grows one;
    otherwise mints a ``LaneName`` *instance* carrying ``"scope"`` — a
    str-enum instance whose value/hash compare equal to ``"scope"`` and
    which ``LaneName(x)`` returns unchanged at every coercion site.
    """
    try:
        return LaneName(LANE_VALUE)
    except ValueError:
        member = str.__new__(LaneName, LANE_VALUE)
        member._name_ = "SCOPE"      # noqa: SLF001 — frozen-enum shim
        member._value_ = LANE_VALUE  # noqa: SLF001 — frozen-enum shim
        return member


#: The lane's LaneName-carrying instance.  ``policy._extension_lane`` and
#: direct ``RetrievalPolicyV7`` construction use this object; it is also
#: the ``LANE_REGISTRY`` key.
LANE_ENUM = _lane_enum()


# ---------------------------------------------------------------------------
# subject resolution
# ---------------------------------------------------------------------------


def _resolve_subject(
    conn: sqlite3.Connection, scope_id: str, generation: int,
    qv: QueryViewV7, stats: dict,
) -> tuple:
    """``(state, subject)`` — the single resolved subject canon, or the
    abstention state (``"none"`` | ``"ambiguous"``).

    Shares ``entity.resolve_query_speaker_canons`` — the V75-03.05
    speaker_match resolver — so the scoped-lane and feature arms resolve
    an identical population: query canons re-folded through
    ``entities_v2.canon``, probed against ``units.speaker_canon`` at/below
    the generation fence, sorted + deduplicated.  A caller-supplied
    ``speaker_canon`` hint wins verbatim (speaker_match's precedence),
    even when the canon is not a scope speaker — the caller declared the
    scope; membership then honestly reports an empty pool.
    """
    hint = getattr(qv, "speaker_canon", None)
    hint_canon = _canon(str(hint)) if hint else ""
    if hint_canon:
        stats["speaker_hint"] = hint_canon
        stats["scope_source"] = "hint"
        return "resolved", hint_canon
    subjects = _ent.resolve_query_speaker_canons(
        conn, scope_id, generation, getattr(qv, "entity_canons", ())
    )
    stats["subject_canons"] = list(subjects)
    stats["scope_source"] = "query"
    if not subjects:
        return "none", None
    if len(subjects) > 1:
        return "ambiguous", None
    return "resolved", subjects[0]


def _scan_mentions(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    subject: str,
    deadline: "_lex._Deadline",
    stats: dict,
) -> tuple:
    """``(uids, roles_of, pinned, scanned, cut)`` — fenced mention postings
    for the subject canon (``scope_id`` + ``generation <= pinned``, the
    V7-30.02 fence; latest-row resolution happens later when uids map
    through the fenced universe).

    ``cut`` is ``"deadline"`` | ``"scan_bound"`` | ``"mention_query_error"``
    | ``None`` — a cut scan reports honestly and emits from the rows it
    did read (the lane is additive: an unscanned scope member is simply
    absent, never a false member).
    """
    uids: set = set()
    roles_of: dict[str, set] = {}
    pinned: set = set()
    scanned = 0
    cut: Optional[str] = None
    sql = (
        "SELECT unit_id, role, byte_start, byte_end"
        f" FROM {MENTIONS_TABLE}"
        " WHERE scope_id = ? AND canon = ? AND generation <= ?"
        " ORDER BY unit_id, generation"
    )
    try:
        cur = conn.execute(sql, (scope_id, subject, generation))
    except sqlite3.Error:
        stats["mentions"] = "query_error"
        return uids, roles_of, pinned, scanned, "mention_query_error"
    while True:
        if deadline.expired():
            cut = "deadline"
            break
        batch = cur.fetchmany(_MENTION_PAGE)
        if not batch:
            break
        for uid, role, bs, be in batch:
            scanned += 1
            if scanned > _MENTION_ROW_LIMIT:
                cut = "scan_bound"
                break
            uid = str(uid)
            uids.add(uid)
            if role:
                roles_of.setdefault(uid, set()).add(str(role))
            if bs is not None and be is not None and int(be) > int(bs):
                pinned.add(uid)
        if cut is not None:
            break
    cur.close()
    return uids, roles_of, pinned, scanned, cut


# ---------------------------------------------------------------------------
# the lane
# ---------------------------------------------------------------------------


def lane_scope(
    ctx: LaneContextV7, query: QueryViewV7, slice: LaneSlice
) -> LaneOutput:
    """L-scope: scoped candidate seed for the single resolved subject
    canon (V75-04.01).  Additive only — fused with the unscoped lanes,
    never an eligibility gate."""
    out = LaneOutput(lane=LANE_NAME, status=LaneStatus.OK)
    stats = out.stats
    stats["lane_version"] = LANE_VERSION
    stats["formula_status"] = FORMULA_STATUS
    stats["formula"] = _lex.FORMULA_ID
    stats["scope"] = "none"  # coverage reports it on every exit path

    deadline = _lex._Deadline(getattr(slice, "deadline_ms", 0.0))
    if deadline.expired():
        out.status = LaneStatus.DEADLINE
        out.reason = "deadline"
        return out

    conn = _lex._conn(ctx)
    if conn is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "no_read_snapshot"
        return out
    generation = getattr(ctx, "generation", None)
    if generation is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "generation_unpinned"
        return out
    if not has_table(conn, UNITS_TABLE):
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "no_units_table"
        return out
    have_mentions = has_table(conn, MENTIONS_TABLE)
    stats["mentions"] = "ok" if have_mentions else "table_absent"

    norm = getattr(query, "norm", None)
    if norm is None:
        out.status = LaneStatus.SKIPPED
        out.reason = "no_analysis"
        return out

    scope_id = str(getattr(ctx, "scope_id", ""))

    # -- subject resolution (shared resolver; cheap abstention exits) --------
    state, subject = _resolve_subject(conn, scope_id, generation, query, stats)
    stats["scope"] = state
    if state == "none":
        out.status = LaneStatus.SKIPPED
        out.reason = "no_subject_canon"
        return out
    if state == "ambiguous":
        # Two or more subject canons resolved — the lane MUST run
        # unscoped (V75-04.01); never a coin flip.
        out.status = LaneStatus.SKIPPED
        out.reason = "ambiguous_subject_canons"
        return out
    stats["scope_canon"] = subject

    # -- scoring index -------------------------------------------------------
    # BM25F needs the fielded index; without it the lane cannot produce
    # honest scores (membership alone is not a score) → unavailable.
    if not has_table(conn, _lex.FTS_TABLE):
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "no_unit_fts"
        return out
    columns = [
        c for c in _lex._fts_columns(conn, _lex.FTS_TABLE) if c in _lex.FIELD_W
    ]
    if not columns:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "no_field_columns"
        return out
    stem_ok = has_table(conn, _lex.STEM_TABLE)
    if not stem_ok:
        stats["stem_channel"] = "unavailable"

    # -- fenced universe ----------------------------------------------------
    # need_full=True always: ``speaker_canon`` is the membership predicate
    # and a per-candidate unit-fact signal, and a callable eligibility
    # handle needs the full unit row — the same rows serve all three.
    uni, uni_trunc = _lex._universe(
        conn, scope_id, generation, deadline, need_full=True, share=ctx
    )
    stats["n_universe"] = len(uni.by_rowid)
    if uni_trunc:
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
        stats["universe_scan"] = "truncated"
        return out
    elig_rowids, elig_via = _lex._resolve_eligibility(ctx, uni)
    stats["eligible_via"] = elig_via
    if elig_via == "unrecognized":
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "eligibility_shape_unknown"
        return out
    out.examined = len(uni.by_rowid)

    # -- scope membership (fenced) -------------------------------------------
    # speaker clause: units whose *latest visible row* carries
    # speaker_canon == subject (the universe scan already resolved
    # latest-row-per-key at/below the pin).
    speaker_uids = {
        str(u["unit_id"])
        for u in uni.by_rowid.values()
        if str(u.get("speaker_canon") or "") == subject
    }
    # mention clause: canonical entity_mentions postings for the subject.
    mention_uids: set = set()
    roles_of: dict[str, set] = {}
    pinned: set = set()
    cut: Optional[str] = None
    if have_mentions:
        mention_uids, roles_of, pinned, scanned, cut = _scan_mentions(
            conn, scope_id, generation, subject, deadline, stats
        )
        out.examined += scanned
        stats["mention_rows"] = scanned
    scoped_uids = speaker_uids | mention_uids
    stats["speaker_units"] = len(speaker_uids)
    stats["mention_units"] = len(mention_uids)
    stats["scoped_units"] = len(scoped_uids)

    # unit_id → universe rowid: membership resolves through the fenced
    # universe (latest row per key) — orphaned mentions and foreign-scope
    # rows have no rowid here and are counted, never emitted.
    uid_to_rid = {
        str(u["unit_id"]): rid for rid, u in uni.by_rowid.items()
    }
    stats["orphaned_mentions"] = sum(
        1 for u in mention_uids if u not in uid_to_rid
    )
    scoped_rids = {
        uid_to_rid[u] for u in scoped_uids if u in uid_to_rid
    }
    # Eligibility BEFORE rank (V7-05.08): intersect the caller's eligible
    # set before any scoring/emission.
    pool_rids = scoped_rids & elig_rowids
    stats["ineligible"] = len(scoped_rids) - len(pool_rids)
    out.eligible = len(pool_rids)

    # -- corpus stats on the FULL eligible set (calibration) -----------------
    corp = _lex._eligible_corpus_stats(
        conn, uni, elig_rowids, columns, scope_id, generation, deadline,
        stats, share=ctx)
    if corp is None:
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
        stats["stats_phase"] = "deadline_cut"
        return out
    n_eligible, avglen = corp
    stats["n_eligible"] = n_eligible

    # -- postings + BM25F scoring over the scope pool ------------------------
    qterms, ident_terms = _lex._query_terms(query)
    # Postings are collected for every query term (df/idf evidence is
    # measured over the full eligible set — calibration); the declared
    # V75-04.02 nomination knobs ride ctx.policy exactly as the lex lane
    # reads them, so a paired arm sees identical nomination diagnostics.
    nom_budget, nom_theta = _lex._nomination_knobs(ctx)
    posts_nom = _lex._collect_postings(
        conn, qterms, ident_terms, uni, elig_rowids, stem_ok, deadline,
        stats, nominate_budget=nom_budget, df_theta=nom_theta,
        n_eligible=n_eligible)
    if posts_nom is None:
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
        return out
    posts, _term_nominated = posts_nom
    out.examined += int(stats.get("posting_rows", 0))

    # Scope's nomination is *membership*, not term match: every eligible
    # scope member is scored — BM25F provably yields 0.0 for a doc with no
    # shared term, and the seed value is the scope itself (the lane is a
    # candidate seed, not only a re-ranker).  ``stats["nominated"]`` (set
    # by the scorer) is therefore the scope pool size.
    scores = _lex._score_candidates(
        conn, qterms, ident_terms, posts, pool_rids, uni, columns, avglen,
        n_eligible, deadline, stats, stem_cache={}, share=ctx)
    if scores is None:
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
        out.examined += int(stats.get("scored_docs", 0))
        return out
    stats["scored"] = len(scores)
    zero_seeded = 0
    for rid in pool_rids:
        if rid not in scores:
            # No field row for this unit — tf is provably 0 (the index is
            # the tf authority), so 0.0 is the measured score, not a
            # fabricated one; the scope membership is the signal.
            scores[rid] = {"score": 0.0, "matched": {}}
            zero_seeded += 1
    stats["seeded_zero_score"] = zero_seeded

    cap = int(getattr(slice, "cap", 0) or 0)
    if cap <= 0:
        pool = POOLS.get(getattr(ctx, "budget", None))
        cap = pool.lane_cap if pool is not None else 200
    ordered = sorted(
        scores.items(),
        key=lambda kv: (-kv[1]["score"], str(uni.by_rowid[kv[0]]["unit_id"]),
                        kv[0]),
    )
    admitted = ordered[:cap]
    stats["cap"] = cap
    stats["overflow"] = max(0, len(ordered) - cap)

    cand_rowids = {rid for rid, _ in admitted}
    phrase = _lex._phrase_flags(
        conn, qterms, _lex._quoted_spans(query), cand_rowids, stats)

    rank = 0
    for rid, acc in admitted:
        unit = uni.by_rowid[rid]
        uid = str(unit["unit_id"])
        rank += 1
        via: list[str] = []
        if uid in speaker_uids:
            via.append("speaker")
        if uid in mention_uids:
            via.append("mention")
        signals: dict[str, Any] = {
            "bm25f": float(acc["score"]),
            "matched_terms": acc["matched"],
            "phrase": float(phrase.get(rid, 0.0)),
            "formula": _lex.FORMULA_ID,
            "scope": "resolved",
            "scope_canon": subject,
            "scope_via": via,
            # unit fact — fusion merges it flat, feeding the
            # ``speaker_match`` rerank feature (the Q1 feature-only arm
            # reads the same fact).
            "speaker_canon": unit.get("speaker_canon"),
            "pinned": uid in pinned,
            "roles": sorted(roles_of.get(uid, ())),
        }
        out.candidates.append(
            CandidateV7(
                unit_id=uid,
                source_id=str(unit["source_id"]),
                revision=int(unit["revision"]),
                lane=LANE_NAME,
                rank=rank,
                raw_score=float(acc["score"]),
                signals=signals,
            )
        )

    if deadline.expired():
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
        stats["deadline"] = True
    elif cut is not None:
        out.status = LaneStatus.PARTIAL
        out.reason = cut
    return out


class ScopeLane(LaneV7):
    """LaneV7 protocol wrapper for registry wiring (V7-05.01)."""

    name = LANE_NAME

    def run(
        self, ctx: LaneContextV7, query: QueryViewV7, slice: LaneSlice
    ) -> LaneOutput:
        return lane_scope(ctx, query, slice)


# Lane modules self-register at import (lanes_base contract).  The key is
# the module-minted LaneName instance — ``register_lane``'s ``LaneName(x)``
# coercion returns it unchanged.
from .lanes_base import register_lane  # noqa: E402

register_lane(LANE_ENUM, lane_scope)


__all__ = [
    "FORMULA_STATUS",
    "LANE_ENUM",
    "LANE_NAME",
    "LANE_VALUE",
    "LANE_VERSION",
    "MENTIONS_TABLE",
    "ScopeLane",
    "UNITS_TABLE",
    "lane_scope",
]
