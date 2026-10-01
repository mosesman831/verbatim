"""J10/J11 / V75-04.01 — the speaker/entity scoped candidate-seed lane.

The ``scope`` lane is an *additive* candidate seed: when the query
resolves to exactly one subject canon (a resolved entity canon that is
also a speaker canon present in the scope — the same resolver the
V75-03.05 ``speaker_match`` feature uses), it emits the eligible units
spoken by that canon (``units.speaker_canon``) or canonically mentioning
it (``entity_mentions.canon``), BM25F-scored on full eligible-corpus
statistics, and fusion unions them with the unscoped lanes.  It never
filters another lane's candidates, and it abstains honestly —
``scope="none"`` / ``"ambiguous"`` — rather than guess.

Covered here:

- J10 — one resolved subject: scoped candidates appear and fuse with
  the unscoped lanes; a unit by *another* speaker stays retrievable
  through lexical (the lane is additive only — V7-05.08 eligibility is
  enforced before rank, never used to remove other lanes' output).
- J11 — two subject canons → ``scope="ambiguous"``, zero candidates.
- ``scope="none"`` — no subject canon (no entities, a non-speaker
  entity, or a canon with no scope presence).
- Flag plumbing — default OFF (absent from ``LANES_V1``), enabled via
  the declared ``lanes`` table (``load_policy``'s extension-lane
  resolution), ablatable via ``lanes_disabled``.
- Generation fence (V7-30.02) — speakers/mentions above the pin are
  invisible; the latest visible unit row decides membership.
- Eligibility-before-rank — ineligible scope members never emit.

The store side is the real §30 schema (``ensure_v7_additive``); the FTS
rowid convention is ``unit_fts.rowid == units.rowid`` via the
``unit_fts_rows`` carrier, exactly as the index writer maintains it.
Query views come from the real S1 builder (``build_query_view``) so the
extraction → canon → subject-resolution path runs end to end.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace as dc_replace

import pytest

from verbatim.core.types_v7 import (
    BudgetClass,
    LaneContextV7,
    LaneName,
    LaneSlice,
    LaneStatus,
)
from verbatim.querying.query_view import build_query_view
from verbatim.retrieval.v7 import pipeline as pp
from verbatim.retrieval.v7 import policy as pol
from verbatim.retrieval.v7 import scope as scope_mod
from verbatim.retrieval.v7.lanes_base import LANE_REGISTRY, run_one
from verbatim.storage.schema_v7 import ensure_v7_additive

SCOPE = "scope-a"
GEN = 1
T0 = 1_700_000_000_000_000

#: Scope vocabulary — speakers plus a non-speaker entity and a canon
#: with no scope presence at all.
KNOWN = ("caroline", "jon", "melanie", "promotion", "zelda")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def mk_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_v7_additive(conn)
    return conn


def add_unit(
    conn,
    uid,
    *,
    speaker=None,
    text="",
    scope=SCOPE,
    gen=GEN,
    rev=1,
    session="sess",
    entities="",
    index=True,
):
    """One unit + its FTS row carrier + field bytes.  ``row_id`` is the
    units row's rowid — the external-content convention the real index
    writer uses (``unit_fts.rowid == units.rowid``).  ``index=False``
    writes the unit row only — a unit the index has not ingested yet."""
    cur = conn.execute(
        "INSERT INTO units (unit_id, source_id, revision, scope_id, kind,"
        " speaker_canon, perspective, recorded_at_us, occurred_precision,"
        " occurred_source, byte_start, byte_end, session_id, generation)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            uid, f"src-{uid}", rev, scope, "turn", speaker, "user_stated",
            T0, "unknown", "unknown", 0, len(text.encode()), session, gen,
        ),
    )
    rid = int(cur.lastrowid)
    if index:
        conn.execute(
            "INSERT INTO unit_fts_rows (row_id, unit_id, scope_id,"
            " generation) VALUES (?,?,?,?)",
            (rid, uid, scope, gen),
        )
        conn.execute(
            'INSERT INTO unit_fts_content (fts_row_id, text, speaker,'
            ' entities, session, "when") VALUES (?,?,?,?,?,NULL)',
            (rid, text, speaker or "", entities, session or ""),
        )
    return rid


def add_mention(
    conn, uid, canon, *, role="mention", bs=0, be=4, gen=GEN, scope=SCOPE,
    surface=None,
):
    conn.execute(
        "INSERT INTO entity_mentions (scope_id, canon, unit_id,"
        " generation, surface, byte_start, byte_end, role)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (scope, canon, uid, gen, surface or canon, bs, be, role),
    )


def qv(text, **kw):
    return build_query_view(text, now_us=T0, known_canons=KNOWN, **kw)


def mk_ctx(
    conn,
    *,
    eligible=None,
    gen=GEN,
    scope=SCOPE,
    lanes=("lex", "scope"),
    policy_json_extra=None,
):
    doc = {"lanes": list(lanes)}
    if policy_json_extra:
        doc.update(policy_json_extra)
    policy = pol.load_policy("test", doc)
    return LaneContextV7(
        store=conn,
        scope_id=scope,
        generation=gen,
        eligible=eligible if eligible is not None else (lambda row: True),
        query_time_us=T0,
        profile="test",
        budget=BudgetClass.MID,
        policy=policy,
        manifest={},
    )


def slice_(ms=60_000.0, cap=200):
    return LaneSlice(deadline_ms=ms, cap=cap)


def seed_locomo_pair(conn):
    """Caroline's two turns, Melanie's two — one of Melanie's *mentions*
    Caroline (in-scope via the canon, spoken by the other speaker)."""
    add_unit(conn, "u_c1", speaker="caroline", text="i live in berlin now")
    add_unit(conn, "u_c2", speaker="caroline", text="work was busy today")
    add_unit(conn, "u_m1", speaker="melanie", text="i live in paris now")
    add_unit(conn, "u_m2", speaker="melanie", text="caroline told me the plan")
    add_mention(conn, "u_m2", "caroline", role="mention", bs=0, be=8)


def lane_ids(out):
    return {c.unit_id for c in out.candidates}


# ---------------------------------------------------------------------------
# J10 — one resolved subject canon
# ---------------------------------------------------------------------------


def test_j10_scoped_candidates_fuse_with_unscoped_lanes():
    """"where does caroline live" resolves the single subject canon
    caroline: her turns + the Melanie turn mentioning her are seeded;
    Melanie's other turn stays retrievable through lexical — additive,
    never a mask."""
    conn = mk_conn()
    seed_locomo_pair(conn)
    view = qv("where does caroline live")
    assert "caroline" in view.entity_canons

    res = pp.run_search(mk_ctx(conn), view, 60_000.0)

    sout = res.lanes["scope"]
    assert sout.status == LaneStatus.OK
    assert sout.stats["scope"] == "resolved"
    assert sout.stats["scope_canon"] == "caroline"
    assert sout.stats["scope_source"] == "query"

    # speaker clause + mention clause; the other speaker's turn is out.
    # V8-06.02: the context stage may *inject* a session neighbor into
    # the lane as a ``ctx_injected`` artifact — that is not a scope-lane
    # nomination, so the boundary assertion is on nominated candidates.
    nominated = {
        c.unit_id for c in sout.candidates
        if not (c.signals or {}).get("ctx_injected")
    }
    assert {"u_c1", "u_c2", "u_m2"} <= nominated
    assert "u_m1" not in nominated
    by_id = {c.unit_id: c for c in sout.candidates}
    assert by_id["u_c1"].signals["scope_via"] == ["speaker"]
    assert by_id["u_m2"].signals["scope_via"] == ["mention"]
    # unit-fact signal fusion merges for the speaker_match feature arm.
    assert by_id["u_c1"].signals["speaker_canon"] == "caroline"
    assert by_id["u_m2"].signals["speaker_canon"] == "melanie"
    assert by_id["u_m2"].signals["pinned"] is True

    # Lexical still produced its own list — nothing was masked.
    lex_ids = lane_ids(res.lanes["lex"])
    assert "u_m1" in lex_ids  # other speaker, term "live" — unscoped hit

    # Fusion carries both provenances: shared hits hold both lane ranks.
    # The out-of-scope turn's real lane is lexical — u_m1 was never
    # *nominated* by scope (asserted above); where a scope rank appears
    # it is a V8-06.02 context-injection artifact (``ctx_injected``),
    # carrying the declared provenance honestly rather than pretending
    # the lane surfaced it.
    fused = {f.unit_id: f for f in res.fused}
    assert {"lex", "scope"} <= set(fused["u_c1"].lane_ranks)
    m1_scope = next(
        (c for c in sout.candidates if c.unit_id == "u_m1"), None)
    if m1_scope is not None:
        assert m1_scope.signals.get("ctx_injected") is True
    assert "lex" in fused["u_m1"].lane_ranks
    assert "scope" in fused["u_m2"].lane_ranks
    # scope merged the unit fact through to the fused signals.
    assert fused["u_m2"].signals["speaker_canon"] == "melanie"

    # Coverage: flag on → the lane ran and reports its state.
    assert res.coverage.lanes["scope"]["status"] == "ok"
    assert res.coverage.lanes["scope"]["produced"] == len(sout.candidates)
    assert "scope" in res.coverage.policy["lanes"]


def test_j10_lane_never_removes_lexical_candidates():
    """Additive-only: the lexical lane's output is byte-identical with
    and without the scope lane enabled."""
    conn = mk_conn()
    seed_locomo_pair(conn)
    # A scope member lexical cannot reach: it mentions caroline but
    # shares no query term in any indexed field — seeded at 0.0.
    add_unit(conn, "u_m3", speaker="melanie", text="noted, thanks")
    add_mention(conn, "u_m3", "caroline", bs=0, be=8)
    view = qv("where does caroline live")

    # Context off: this test measures the scope lane's union
    # contribution to the fused pool.  All fixtures share one session,
    # so a live V8-06 stage would inject the same neighbors into both
    # arms and mask the delta the assertion exists to measure.
    ctx_off = {"params": {"context.mode": "off"}}
    solo = pp.run_search(
        mk_ctx(conn, lanes=("lex",), policy_json_extra=ctx_off),
        view, 60_000.0)
    duo = pp.run_search(
        mk_ctx(conn, lanes=("lex", "scope"), policy_json_extra=ctx_off),
        view, 60_000.0)

    def shape(out):
        return [
            (c.unit_id, c.source_id, c.revision, c.rank, c.raw_score)
            for c in out.candidates
        ]

    assert shape(solo.lanes["lex"]) == shape(duo.lanes["lex"])
    # And the fused pool only grew (scope seeds are unioned, never
    # subtracted).
    solo_fused = {f.unit_id for f in solo.fused}
    duo_fused = {f.unit_id for f in duo.fused}
    assert solo_fused <= duo_fused
    assert duo_fused - solo_fused  # the lane added something


# ---------------------------------------------------------------------------
# J11 — ambiguous / unresolved subjects abstain
# ---------------------------------------------------------------------------


def test_j11_two_subject_canons_abstain():
    """"what did caroline and jon discuss" names two speaker canons →
    the lane MUST run unscoped on that input: skipped status, zero
    candidates, ``scope="ambiguous"`` — never a coin flip."""
    conn = mk_conn()
    seed_locomo_pair(conn)
    add_unit(conn, "u_j1", speaker="jon", text="the launch plan thread")
    view = qv("what did caroline and jon discuss")
    assert {"caroline", "jon"} <= set(view.entity_canons)

    out = run_one(mk_ctx(conn), view, scope_mod.LANE_ENUM, slice_())
    assert out.status == LaneStatus.SKIPPED
    assert out.reason == "ambiguous_subject_canons"
    assert out.stats["scope"] == "ambiguous"
    assert out.stats["subject_canons"] == ["caroline", "jon"]
    assert out.candidates == []


def test_j11_through_pipeline_reports_abstention():
    """Pipeline composition on the same ambiguous query: the main call's
    ``scope="ambiguous"`` verdict is preserved in stats/coverage.  The
    V7-05.13 facet pass decomposes the comparison into single-subject
    sub-queries ("caroline", "jon discuss") — each resolves exactly one
    subject canon and may legitimately scope, so any scoped candidate
    carries an honest ``facet`` tag; none comes from the ambiguous main
    call, and the unscoped lanes answer regardless."""
    conn = mk_conn()
    seed_locomo_pair(conn)
    add_unit(conn, "u_j1", speaker="jon", text="the launch plan thread")
    view = qv("what did caroline and jon discuss")
    assert len(view.facets or ()) >= 2  # comparison decomposition present

    res = pp.run_search(mk_ctx(conn), view, 60_000.0)

    sout = res.lanes["scope"]
    assert sout.stats["scope"] == "ambiguous"
    assert res.coverage.lanes["scope"]["status"] == "skipped"
    # Facet-sourced candidates are tagged with their facet index — a
    # scoped contribution never pretends to answer the ambiguous query.
    for c in sout.candidates:
        assert "facet" in c.signals
    assert res.lanes["lex"].candidates  # unscoped lanes unaffected
    assert res.fused


def test_scope_none_when_no_subject_canon():
    """No resolved speaker canon in scope → ``scope="none"``."""
    conn = mk_conn()
    seed_locomo_pair(conn)

    for text in (
        "where does the team meet",     # names nothing
        "what is the promotion status",  # entity canon, not a speaker
        "did zelda call",                # known canon, no scope presence
    ):
        view = qv(text)
        out = run_one(mk_ctx(conn), view, scope_mod.LANE_ENUM, slice_())
        assert out.status == LaneStatus.SKIPPED, text
        assert out.stats["scope"] == "none", text
        assert out.reason == "no_subject_canon", text
        assert out.candidates == [], text


def test_hint_resolves_and_wins_verbatim():
    """A caller ``speaker_hint`` is the declared subject — it resolves
    even when the query names no entity, and wins verbatim over a
    query-named speaker (speaker_match's precedence; a canon with no
    scope units yields an honest empty pool)."""
    conn = mk_conn()
    seed_locomo_pair(conn)

    view = qv("where does the team meet", speaker_hint="Caroline")
    out = run_one(mk_ctx(conn), view, scope_mod.LANE_ENUM, slice_())
    assert out.status == LaneStatus.OK
    assert out.stats["scope"] == "resolved"
    assert out.stats["scope_canon"] == "caroline"
    assert out.stats["scope_source"] == "hint"
    assert lane_ids(out) >= {"u_c1", "u_c2"}

    # Hint wins verbatim even when it names no scope speaker — the
    # caller declared the scope; the pool is honestly empty.
    view = qv("where does caroline live", speaker_hint="Zelda")
    out = run_one(mk_ctx(conn), view, scope_mod.LANE_ENUM, slice_())
    assert out.status == LaneStatus.OK
    assert out.stats["scope"] == "resolved"
    assert out.stats["scope_canon"] == "zelda"
    assert out.candidates == []


# ---------------------------------------------------------------------------
# flag plumbing — default OFF, declared-table enablement, ablation
# ---------------------------------------------------------------------------


def test_flag_off_lane_never_runs():
    """Default OFF: ``scope`` is not in LANES_V1, so a default policy
    never runs it — absent from lane outputs and lane coverage."""
    conn = mk_conn()
    seed_locomo_pair(conn)
    assert "scope" not in [n.value for n in pol.LANES_V1]

    policy = pol.load_policy("test")  # default table
    assert "scope" not in [n.value for n in policy.lanes]

    ctx = LaneContextV7(
        store=conn, scope_id=SCOPE, generation=GEN,
        eligible=lambda row: True, query_time_us=T0, profile="test",
        budget=BudgetClass.MID, policy=policy, manifest={},
    )
    res = pp.run_search(ctx, qv("where does caroline live"), 60_000.0)
    assert "scope" not in res.lanes
    assert "scope" not in res.coverage.lanes
    assert "scope" not in res.coverage.policy["lanes"]


def test_flag_on_via_declared_lanes_table():
    """``lanes: [..., "scope"]`` in the declared policy enables the lane;
    policy resolution mints its LaneName instance and materializes its
    flat default weight."""
    policy = pol.load_policy("test", {"lanes": ["lex", "scope"]})
    member = next(n for n in policy.lanes if n.value == "scope")
    assert member is scope_mod.LANE_ENUM
    for row in policy.lane_weights.values():
        assert row[member] == pol.DEFAULT_LANE_WEIGHT
    # LANE_ENUM is a LaneName instance — every LaneName(x) coercion
    # site returns it unchanged.
    assert LaneName(member) is member
    assert member in LANE_REGISTRY

    # Ablation switch honors the extension name too.
    abl = pol.ablation_lanes(policy, ["scope"])
    assert [n.value for n in abl.lanes] == ["lex"]


def test_lane_reports_stats_even_when_skipped():
    """Coverage always carries the scope state — the stat is set before
    any early exit."""
    conn = mk_conn()
    seed_locomo_pair(conn)
    out = run_one(
        mk_ctx(conn), qv("where does the team meet"),
        scope_mod.LANE_ENUM, slice_(),
    )
    assert out.stats["scope"] == "none"


# ---------------------------------------------------------------------------
# generation fence + eligibility-before-rank
# ---------------------------------------------------------------------------


def test_generation_fence_hides_newer_rows():
    """A subject canon present only above the pin does not resolve; a
    mention above the pin never contributes membership; membership
    follows the *latest visible* unit row."""
    conn = mk_conn()
    # caroline speaks only at gen 5 — invisible at the GEN=1 pin.
    add_unit(conn, "u_c1", speaker="caroline", text="hello", gen=5)
    # melanie's gen-1 unit mentions caroline only at gen 5.
    add_unit(conn, "u_m1", speaker="melanie", text="a note", gen=1)
    add_mention(conn, "u_m1", "caroline", gen=5)
    view = qv("where does caroline live")

    out = run_one(
        mk_ctx(conn, gen=1), view, scope_mod.LANE_ENUM, slice_())
    assert out.stats["scope"] == "none"
    assert out.candidates == []

    # At the gen-5 pin the same store resolves and the fenced mention
    # now contributes membership.
    out = run_one(
        mk_ctx(conn, gen=5), view, scope_mod.LANE_ENUM, slice_())
    assert out.stats["scope"] == "resolved"
    assert lane_ids(out) >= {"u_c1", "u_m1"}


def test_membership_uses_latest_visible_unit_row():
    """The same unit re-projected at a newer generation with a different
    speaker: membership follows the latest row at/below the pin."""
    conn = mk_conn()
    add_unit(conn, "u1", speaker="caroline", text="the old version", gen=1)
    add_unit(conn, "u1", speaker="jon", text="the corrected version", gen=9)
    add_unit(conn, "u_c2", speaker="caroline", text="still caroline", gen=1)
    view = qv("where does caroline live")

    out = run_one(
        mk_ctx(conn, gen=1), view, scope_mod.LANE_ENUM, slice_())
    assert lane_ids(out) >= {"u1", "u_c2"}

    out = run_one(
        mk_ctx(conn, gen=9), view, scope_mod.LANE_ENUM, slice_())
    scoped = lane_ids(out)
    assert "u_c2" in scoped
    assert "u1" not in scoped  # latest visible row says jon


def test_eligibility_before_rank_gates_emission():
    """Ineligible scope members are counted and never emitted; the lane
    never widens the caller's set."""
    conn = mk_conn()
    seed_locomo_pair(conn)
    ctx = mk_ctx(conn, eligible={"u_c1"})  # unit-id set form
    out = run_one(ctx, qv("where does caroline live"),
                  scope_mod.LANE_ENUM, slice_())
    assert out.status == LaneStatus.OK
    assert lane_ids(out) == {"u_c1"}
    assert out.stats["scope"] == "resolved"
    assert out.stats["ineligible"] >= 2  # u_c2 + u_m2 in scope, withheld

    # Pipeline-level: the S3 re-check also holds under the set form.
    res = pp.run_search(ctx, qv("where does caroline live"), 60_000.0)
    assert {f.unit_id for f in res.fused} <= {"u_c1"}


def test_eligibility_callable_sees_full_rows():
    """A callable eligible handle receives the full unit row dict."""
    conn = mk_conn()
    seed_locomo_pair(conn)
    seen = []

    def elig(row):
        seen.append(row.get("speaker_canon"))
        return row.get("speaker_canon") == "caroline"

    out = run_one(
        mk_ctx(conn, eligible=elig), qv("where does caroline live"),
        scope_mod.LANE_ENUM, slice_())
    assert out.status == LaneStatus.OK
    assert lane_ids(out) == {"u_c1", "u_c2"}  # u_m2 (melanie) withheld
    assert seen  # the callable actually ran


def test_unrecognized_eligibility_fails_honestly():
    conn = mk_conn()
    seed_locomo_pair(conn)
    ctx = mk_ctx(conn, eligible=object())
    out = run_one(ctx, qv("where does caroline live"),
                  scope_mod.LANE_ENUM, slice_())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "eligibility_shape_unknown"
    assert out.candidates == []


# ---------------------------------------------------------------------------
# scoring honesty + honest degradation
# ---------------------------------------------------------------------------


def test_scope_seeds_zero_score_members():
    """An in-scope unit sharing no query term still emits at its measured
    0.0 — the lane is a candidate seed, not only a re-ranker.  Both zero
    shapes are honest: an indexed member is measured at 0.0 by the
    scorer; an unindexed member has provably no term frequency, so the
    0.0 fill is the measured-equivalent score (the index is the tf
    authority)."""
    conn = mk_conn()
    add_unit(conn, "u_c1", speaker="caroline", text="i live in berlin")
    # Indexed mention member with no term overlap → measured 0.0.
    add_unit(conn, "u_m3", speaker="melanie", text="noted, thanks")
    add_mention(conn, "u_m3", "caroline", bs=0, be=8)
    # Speaker unit with no index row at all → 0.0 fill path.
    add_unit(conn, "u_c9", speaker="caroline", text="", index=False)

    out = run_one(
        mk_ctx(conn), qv("where does caroline live"),
        scope_mod.LANE_ENUM, slice_())
    by_id = {c.unit_id: c for c in out.candidates}
    assert "u_m3" in by_id
    assert by_id["u_m3"].raw_score == 0.0
    assert by_id["u_m3"].signals["bm25f"] == 0.0
    assert by_id["u_m3"].signals["matched_terms"] == {}
    assert by_id["u_m3"].signals["scope_via"] == ["mention"]
    assert by_id["u_c9"].raw_score == 0.0
    assert by_id["u_c9"].signals["scope_via"] == ["speaker"]
    assert out.stats["seeded_zero_score"] == 1  # the unindexed member


def test_scores_calibrated_on_full_eligible_corpus():
    """df/idf denominators come from the whole eligible set, not the
    scoped subset — raw scores stay comparable to the lexical lane's."""
    conn = mk_conn()
    seed_locomo_pair(conn)
    ctx = mk_ctx(conn)
    view = qv("where does caroline live")
    lex_out = run_one(ctx, view, LaneName.LEX, slice_())
    sc_out = run_one(ctx, view, scope_mod.LANE_ENUM, slice_())

    assert sc_out.stats["n_eligible"] == lex_out.stats["n_eligible"]
    assert sc_out.stats["df"] == lex_out.stats["df"]
    # A shared unit carries the identical bm25f raw score on both lanes.
    lex_scores = {c.unit_id: c.raw_score for c in lex_out.candidates}
    sc_scores = {c.unit_id: c.raw_score for c in sc_out.candidates}
    shared = set(lex_scores) & set(sc_scores)
    assert shared
    for uid in shared:
        assert sc_scores[uid] == pytest.approx(lex_scores[uid])


def test_unavailable_without_fts_index():
    """units exist but the fielded index does not — a resolved scope
    cannot score honestly → unavailable, not a silent empty list."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE units (unit_id TEXT, source_id TEXT, revision INT,"
        " scope_id TEXT, kind TEXT, speaker_canon TEXT, generation INT,"
        " PRIMARY KEY (unit_id, generation))"
    )
    conn.execute(
        "INSERT INTO units VALUES ('u_c1','s',1,'scope-a','turn',"
        "'caroline',1)"
    )
    out = run_one(
        mk_ctx(conn), qv("where does caroline live"),
        scope_mod.LANE_ENUM, slice_())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "no_unit_fts"
    assert out.stats["scope"] == "resolved"  # resolution still reported


def test_unavailable_on_bare_store():
    conn = sqlite3.connect(":memory:")
    out = run_one(
        mk_ctx(conn), qv("where does caroline live"),
        scope_mod.LANE_ENUM, slice_())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "no_units_table"


def test_no_analysis_skips():
    conn = mk_conn()
    seed_locomo_pair(conn)
    view = dc_replace(qv("where does caroline live"), norm=None)
    out = run_one(
        mk_ctx(conn), view, scope_mod.LANE_ENUM, slice_())
    assert out.status == LaneStatus.SKIPPED
    assert out.reason == "no_analysis"


def test_deterministic_repeat_run():
    conn = mk_conn()
    seed_locomo_pair(conn)
    ctx = mk_ctx(conn)
    view = qv("where does caroline live")
    a = run_one(ctx, view, scope_mod.LANE_ENUM, slice_())
    b = run_one(ctx, view, scope_mod.LANE_ENUM, slice_())
    assert [(c.unit_id, c.rank, c.raw_score) for c in a.candidates] == [
        (c.unit_id, c.rank, c.raw_score) for c in b.candidates
    ]
