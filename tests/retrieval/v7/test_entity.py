"""Lane tests for ``verbatim/retrieval/v7/entity.py`` (w-lane-ent, wave B).

The real §30 schema is used (``ensure_v7_additive`` — landed); only the
``units``/``entity_mentions``/``entity_canon``/``entity_aliases_v7``/
``lex_stats`` tables the lane touches are seeded.  ``QueryViewV7`` is
hand-built with ``entity_canons`` already set (S1 output); the real
query-side extractor is never called here.  These tests unit-test the
LANE.
"""

from __future__ import annotations

import math
import sqlite3

import pytest

from verbatim.core.types import VerbatimError
from verbatim.core.types_v7 import (
    BudgetClass,
    IntentClass,
    IntentResult,
    LaneContextV7,
    LaneName,
    LaneSlice,
    LaneStatus,
    NormAnalysis,
    NormTerm,
    QueryViewV7,
    RetrievalPolicyV7,
)
from verbatim.enrichment.entities_v2 import canon, entity_weight, idf_weight
from verbatim.retrieval.v7 import entity as ent
from verbatim.retrieval.v7.entity import lane_entity
from verbatim.retrieval.v7.lanes_base import LANE_REGISTRY
from verbatim.storage.schema_v7 import ensure_v7_additive

SCOPE = "scope-a"
GEN = 1
T0 = 1_700_000_000_000_000


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def mk_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_v7_additive(conn)
    return conn


def add_unit(
    conn,
    uid,
    *,
    scope=SCOPE,
    gen=GEN,
    source=None,
    rev=1,
    speaker=None,
    session=None,
    seq=None,
):
    conn.execute(
        "INSERT OR REPLACE INTO units (unit_id, source_id, revision,"
        " scope_id, kind, parent_unit_id, session_id, seq, speaker_canon,"
        " perspective, recorded_at_us, occurred_start_us, occurred_end_us,"
        " occurred_precision, occurred_source, byte_start, byte_end,"
        " generation) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            uid,
            source or f"src-{uid}",
            rev,
            scope,
            "turn",
            None,
            session,
            seq,
            speaker,
            "user_stated",
            T0,
            None,
            None,
            "unknown",
            "unknown",
            None,
            None,
            gen,
        ),
    )


def add_mention(
    conn,
    uid,
    c,
    *,
    surface=None,
    bs=0,
    be=4,
    role="mention",
    scope=SCOPE,
    gen=GEN,
):
    conn.execute(
        "INSERT INTO entity_mentions (scope_id, canon, unit_id,"
        " generation, surface, byte_start, byte_end, role)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (scope, c, uid, gen, surface if surface is not None else c, bs, be, role),
    )


def set_df(conn, c, df, *, scope=SCOPE, gen=GEN, display=None):
    conn.execute(
        "INSERT OR REPLACE INTO entity_canon"
        " (scope_id, canon, generation, display, df_units)"
        " VALUES (?,?,?,?,?)",
        (scope, c, gen, display or c, df),
    )


def add_alias(
    conn,
    c,
    alias,
    *,
    state="active",
    rule="A5",
    method="caller",
    ev=1,
    gen=GEN,
    scope=SCOPE,
):
    conn.execute(
        "INSERT OR REPLACE INTO entity_aliases_v7"
        " (scope_id, canon, alias_canon, generation, rule_id,"
        "  evidence_count, method, state)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (scope, c, alias, gen, rule, ev, method, state),
    )


def set_lex_stats(conn, n, *, scope=SCOPE, gen=GEN, field="text"):
    conn.execute(
        "INSERT OR REPLACE INTO lex_stats"
        " (scope_id, generation, field, stats_version, n_units, total_len)"
        " VALUES (?,?,?,?,?,?)",
        (scope, gen, field, "bm25f/v1", n, n * 10),
    )


def mk_ctx(conn, *, eligible=None, gen=GEN, scope=SCOPE, store=None):
    if eligible is None:
        eligible = lambda row: True  # noqa: E731 — allow-all fixture
    return LaneContextV7(
        store=store if store is not None else conn,
        scope_id=scope,
        generation=gen,
        eligible=eligible,
        query_time_us=T0,
        profile="test",
        budget=BudgetClass.MID,
        policy=RetrievalPolicyV7(
            policy_id="retrieval_policy/v7",
            profile="test",
            lanes=(LaneName.ENT,),
            lane_weights={},
        ),
        manifest={},
    )


def mk_qv(canons=(), query="q"):
    terms = tuple(
        NormTerm(term=t, channel="text", byte_start=0, byte_end=len(t))
        for t in query.split()
    )
    return QueryViewV7(
        query=query,
        norm=NormAnalysis(analyzer_id="norm/v2", terms=terms, identifiers=()),
        intent=IntentResult(
            primary=IntentClass.LOOKUP, classes=(IntentClass.LOOKUP,)
        ),
        entity_canons=tuple(canons),
        query_time_us=T0,
    )


def sl(cap=50, ms=10_000.0):
    return LaneSlice(deadline_ms=ms, cap=cap)


def by_id(out):
    return {c.unit_id: c for c in out.candidates}


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_exact_canon_match_emits_and_ranks():
    conn = mk_conn()
    for i in range(8):  # corpus of 10 units total
        add_unit(conn, f"f{i}")
    add_unit(conn, "u_car")
    add_unit(conn, "u_pro")
    add_mention(conn, "u_car", "caroline")
    for i in range(8):
        add_mention(conn, f"f{i}", "promotion")
    add_mention(conn, "u_pro", "promotion")
    set_df(conn, "caroline", 1)
    set_df(conn, "promotion", 9)

    out = lane_entity(mk_ctx(conn), mk_qv(("caroline", "promotion")), sl())
    assert out.status is LaneStatus.OK
    ids = [c.unit_id for c in out.candidates]
    assert ids[0] == "u_car"  # rare canon beats the dominant one (D7-07)
    assert by_id(out)["u_car"].raw_score > by_id(out)["u_pro"].raw_score
    assert all(c.lane == "ent" for c in out.candidates)
    assert [c.rank for c in out.candidates] == list(
        range(1, len(out.candidates) + 1)
    )


def test_single_canon_rank_and_fields():
    conn = mk_conn()
    add_unit(conn, "u1", source="s1", rev=3)
    add_mention(conn, "u1", "caroline", surface="Caroline", bs=4, be=12)
    set_df(conn, "caroline", 1)
    out = lane_entity(mk_ctx(conn), mk_qv(("caroline",)), sl())
    (c,) = out.candidates
    assert c.unit_id == "u1" and c.source_id == "s1" and c.revision == 3
    assert c.signals["matched_canons"] == ["caroline"]
    assert c.signals["pinned"] is True
    assert c.signals["alias_matched"] is False
    # N = 1 unit corpus → df/N = 1 > 0.30 → the dominant-canon cap
    # (V7-08.06) binds: 0.1 × idf(1, N).
    assert c.signals["entity_idf"] == pytest.approx(
        entity_weight("caroline", {"caroline": 1}, 1), abs=1e-5
    )


def test_possessive_folded_canon_matches():
    """D7-01/D7-02: ``Caroline's`` in the query and a mention written with
    surface ``Caroline's`` both land on canon ``caroline``."""
    assert canon("Caroline's") == "caroline"
    conn = mk_conn()
    add_unit(conn, "u1")
    add_mention(conn, "u1", canon("Caroline's"), surface="Caroline's")
    set_df(conn, "caroline", 1)
    out = lane_entity(mk_ctx(conn), mk_qv(("Caroline's",)), sl())
    assert {c.unit_id for c in out.candidates} == {"u1"}
    # a lowercase query surface also folds onto the canon
    out2 = lane_entity(mk_ctx(conn), mk_qv(("CAROLINE",)), sl())
    assert {c.unit_id for c in out2.candidates} == {"u1"}


def test_alias_expansion_surfaces_aliased_mentions():
    conn = mk_conn()
    add_unit(conn, "u_mel")
    add_unit(conn, "u_melanie")
    add_mention(conn, "u_mel", "mel")
    add_mention(conn, "u_melanie", "melanie")
    set_df(conn, "melanie", 1)
    set_df(conn, "mel", 1)
    add_alias(conn, "melanie", "mel", state="active", rule="A3", method="rule")
    out = lane_entity(mk_ctx(conn), mk_qv(("melanie",)), sl())
    ids = by_id(out)
    assert set(ids) == {"u_mel", "u_melanie"}
    mel = ids["u_mel"]
    assert mel.signals["matched_canons"] == ["mel"]
    assert mel.signals["alias_matched"] is True
    assert out.stats["alias_links_used"] == 1


def test_alias_expansion_reverse_direction():
    """A query naming the *alias* reaches the canonical's postings —
    merges are links in both directions (V7-08.04)."""
    conn = mk_conn()
    add_unit(conn, "u1")
    add_mention(conn, "u1", "melanie")
    add_alias(conn, "melanie", "mel", state="active")
    out = lane_entity(mk_ctx(conn), mk_qv(("mel",)), sl())
    assert {c.unit_id for c in out.candidates} == {"u1"}


def test_candidate_and_rejected_aliases_never_expand():
    for state in ("candidate", "rejected"):
        conn = mk_conn()
        add_unit(conn, "u_mel")
        add_mention(conn, "u_mel", "mel")
        add_alias(conn, "melanie", "mel", state=state)
        out = lane_entity(mk_ctx(conn), mk_qv(("melanie",)), sl())
        assert out.status is LaneStatus.OK
        assert out.candidates == []  # unreviewed merges cannot widen


def test_alias_latest_generation_state_wins():
    """The newest row at/below the fence is authoritative: an active row
    superseded by a candidate rewrite (A6 conflict review) stops
    expanding; a candidate promoted to active starts."""
    conn = mk_conn()
    add_unit(conn, "u1")
    add_mention(conn, "u1", "mel")
    add_alias(conn, "melanie", "mel", state="active", gen=1)
    add_alias(conn, "melanie", "mel", state="candidate", rule="A6",
              method="rule", gen=3)
    out3 = lane_entity(mk_ctx(conn, gen=3), mk_qv(("melanie",)), sl())
    assert out3.candidates == []
    out1 = lane_entity(mk_ctx(conn, gen=1), mk_qv(("melanie",)), sl())
    assert {c.unit_id for c in out1.candidates} == {"u1"}

    conn2 = mk_conn()
    add_unit(conn2, "u1")
    add_mention(conn2, "u1", "mel")
    add_alias(conn2, "melanie", "mel", state="candidate", gen=1)
    add_alias(conn2, "melanie", "mel", state="active", gen=3)
    out = lane_entity(mk_ctx(conn2, gen=3), mk_qv(("melanie",)), sl())
    assert {c.unit_id for c in out.candidates} == {"u1"}


def test_alias_row_beyond_fence_does_not_expand():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_mention(conn, "u1", "mel")
    add_alias(conn, "melanie", "mel", state="active", gen=9)
    out = lane_entity(mk_ctx(conn, gen=3), mk_qv(("melanie",)), sl())
    assert out.candidates == []


def test_multi_canon_boost_orders():
    conn = mk_conn()
    # 8 filler units keep df/N ≤ 0.30 so the dominant cap stays out of
    # the arithmetic under test.
    for i in range(8):
        add_unit(conn, f"f{i}")
    add_unit(conn, "u_ab"), add_unit(conn, "u_a"), add_unit(conn, "u_b")
    add_mention(conn, "u_ab", "alpha"), add_mention(conn, "u_ab", "beta")
    add_mention(conn, "u_a", "alpha")
    add_mention(conn, "u_b", "beta")
    set_df(conn, "alpha", 2), set_df(conn, "beta", 2)
    out = lane_entity(mk_ctx(conn), mk_qv(("alpha", "beta")), sl())
    ids = [c.unit_id for c in out.candidates]
    assert ids == ["u_ab", "u_a", "u_b"]  # boost, then unit_id tie-break
    ab = by_id(out)["u_ab"]
    w = entity_weight("alpha", {"alpha": 2}, 11)
    assert ab.signals["query_canons_covered"] == 2
    assert ab.signals["multi_canon_boost"] == pytest.approx(math.log(2), abs=1e-5)
    assert ab.raw_score == pytest.approx(2 * w + math.log(2))
    assert by_id(out)["u_a"].raw_score == pytest.approx(w)


def test_alias_and_canonical_together_do_not_boost():
    """Matching a canon AND its own alias covers one query canon — the
    all-canons-present boost counts *query* canons, never double-counts
    one entity."""
    conn = mk_conn()
    add_unit(conn, "u1")
    add_mention(conn, "u1", "melanie")
    add_mention(conn, "u1", "mel", bs=20, be=23)
    add_alias(conn, "melanie", "mel", state="active")
    set_df(conn, "melanie", 1), set_df(conn, "mel", 1)
    out = lane_entity(mk_ctx(conn), mk_qv(("melanie",)), sl())
    (c,) = out.candidates
    assert sorted(c.signals["matched_canons"]) == ["mel", "melanie"]
    assert c.signals["query_canons_covered"] == 1
    assert "multi_canon_boost" not in c.signals


def test_eligibility_gate_drops_held_units():
    conn = mk_conn()
    for u in ("u1", "u2", "u3"):
        add_unit(conn, u)
        add_mention(conn, u, "caroline")
    set_df(conn, "caroline", 3)
    out = lane_entity(
        mk_ctx(conn, eligible={"u1", "u3"}), mk_qv(("caroline",)), sl()
    )
    assert {c.unit_id for c in out.candidates} == {"u1", "u3"}
    assert out.stats["ineligible"] == 1
    assert out.eligible == 2

    # callable form — same gate
    out2 = lane_entity(
        mk_ctx(conn, eligible=lambda row: row["unit_id"] != "u2"),
        mk_qv(("caroline",)),
        sl(),
    )
    assert {c.unit_id for c in out2.candidates} == {"u1", "u3"}


def test_eligibility_callable_exception_fails_closed():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_mention(conn, "u1", "x")

    def boom(row):
        raise RuntimeError("eligibility oracle down")

    out = lane_entity(mk_ctx(conn, eligible=boom), mk_qv(("x",)), sl())
    assert out.candidates == []
    assert out.stats["ineligible"] == 1


def test_generation_fence_on_mentions():
    conn = mk_conn()
    add_unit(conn, "u_old")
    add_unit(conn, "u_new")
    add_mention(conn, "u_old", "caroline", gen=1)
    add_mention(conn, "u_new", "caroline", gen=5)
    out = lane_entity(mk_ctx(conn, gen=3), mk_qv(("caroline",)), sl())
    assert {c.unit_id for c in out.candidates} == {"u_old"}
    out5 = lane_entity(mk_ctx(conn, gen=5), mk_qv(("caroline",)), sl())
    assert {c.unit_id for c in out5.candidates} == {"u_old", "u_new"}


def test_latest_unit_generation_supplies_row():
    """Rebuild coexistence: mention at g1 + unit projections at g1/g3 —
    the latest visible row supplies source_id/revision/eligibility."""
    conn = mk_conn()
    add_unit(conn, "u1", gen=1, source="src-old", rev=1)
    add_unit(conn, "u1", gen=3, source="src-new", rev=2)
    add_mention(conn, "u1", "caroline", gen=1)
    out = lane_entity(mk_ctx(conn, gen=3), mk_qv(("caroline",)), sl())
    (c,) = out.candidates
    assert c.source_id == "src-new" and c.revision == 2
    # eligibility consults the latest projection: a callable that only
    # admits revision >= 2 keeps the unit — the g1 row never gated it.
    out2 = lane_entity(
        mk_ctx(conn, gen=3, eligible=lambda row: row["revision"] >= 2),
        mk_qv(("caroline",)),
        sl(),
    )
    assert {c.unit_id for c in out2.candidates} == {"u1"}


def test_scope_isolation():
    conn = mk_conn()
    add_unit(conn, "u1", scope="other")
    add_mention(conn, "u1", "caroline", scope="other")
    out = lane_entity(mk_ctx(conn), mk_qv(("caroline",)), sl())
    assert out.candidates == []
    # a scope-a mention pointing at a foreign unit is dropped by the
    # scope guard, never emitted
    add_mention(conn, "u1", "caroline", scope=SCOPE)
    out2 = lane_entity(mk_ctx(conn), mk_qv(("caroline",)), sl())
    assert out2.candidates == []
    assert out2.stats["ineligible"] == 1


def test_orphan_mention_never_emitted():
    conn = mk_conn()
    add_mention(conn, "ghost", "caroline")  # no units row at all
    out = lane_entity(mk_ctx(conn), mk_qv(("caroline",)), sl())
    assert out.candidates == []
    assert out.stats["orphaned_mentions"] == 1


# ---------------------------------------------------------------------------
# scoring internals
# ---------------------------------------------------------------------------


def test_df_zero_canon_still_emits_low():
    conn = mk_conn()
    for i in range(8):  # keep rare's df/N ≤ 0.30 so it is not capped
        add_unit(conn, f"f{i}")
    add_unit(conn, "u_rare"), add_unit(conn, "u_ghost")
    add_mention(conn, "u_rare", "rare")
    add_mention(conn, "u_ghost", "ghost")
    set_df(conn, "rare", 1)
    set_df(conn, "ghost", 0)  # recorded df = 0 — no measured rarity
    out = lane_entity(mk_ctx(conn), mk_qv(("rare", "ghost")), sl())
    ids = by_id(out)
    assert set(ids) == {"u_rare", "u_ghost"}  # zero df still matches
    assert ids["u_ghost"].raw_score < ids["u_rare"].raw_score
    assert out.stats["df_floor_canons"] == 1
    # floored at df = N → dominant-canon cap binds: 0.1 * idf(1, N)
    n = out.stats["n_units"]
    assert ids["u_ghost"].signals["entity_idf"] == pytest.approx(
        min(idf_weight(n, n), 0.1 * idf_weight(1, n)), abs=1e-5
    )


def test_absent_canon_row_also_floors():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_mention(conn, "u1", "untracked")  # no entity_canon row at all
    out = lane_entity(mk_ctx(conn), mk_qv(("untracked",)), sl())
    assert len(out.candidates) == 1
    assert out.stats["df_floor_canons"] == 1
    assert out.candidates[0].signals["entity_idf"] < idf_weight(1, 1)


def test_lex_stats_denominator_and_units_fallback():
    conn = mk_conn()
    add_unit(conn, "u1"), add_unit(conn, "u2")
    add_mention(conn, "u1", "caroline")
    set_df(conn, "caroline", 1)
    out = lane_entity(mk_ctx(conn), mk_qv(("caroline",)), sl())
    assert out.stats["n_units_source"] == "units_fallback"
    assert out.stats["n_units"] == 2
    assert out.candidates[0].signals["entity_idf"] == pytest.approx(
        entity_weight("caroline", {"caroline": 1}, 2), abs=1e-5
    )

    set_lex_stats(conn, 100)
    out2 = lane_entity(mk_ctx(conn), mk_qv(("caroline",)), sl())
    assert out2.stats["n_units_source"] == "lex_stats"
    assert out2.stats["n_units"] == 100
    assert out2.candidates[0].signals["entity_idf"] == pytest.approx(
        entity_weight("caroline", {"caroline": 1}, 100), abs=1e-5
    )


def test_role_and_pinned_signals():
    conn = mk_conn()
    add_unit(conn, "u_sp")
    add_unit(conn, "u_sub")
    # speaker metadata mention: the (0,0) span convention — never a pin.
    add_mention(conn, "u_sp", "caroline", bs=0, be=0, role="speaker")
    add_mention(conn, "u_sub", "caroline", bs=10, be=18, role="subject")
    set_df(conn, "caroline", 2)
    out = lane_entity(mk_ctx(conn), mk_qv(("caroline",)), sl())
    ids = by_id(out)
    assert ids["u_sp"].signals["role"] == "speaker"
    assert ids["u_sp"].signals["pinned"] is False
    assert ids["u_sub"].signals["role"] == "subject"
    assert ids["u_sub"].signals["pinned"] is True
    assert ids["u_sub"].signals["spans"] == [
        {"canon": "caroline", "byte_start": 10, "byte_end": 18}
    ]


def test_role_precedence_strongest_wins():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_mention(conn, "u1", "caroline", bs=0, be=0, role="speaker")
    add_mention(conn, "u1", "caroline", bs=4, be=12, role="subject")
    out = lane_entity(mk_ctx(conn), mk_qv(("caroline",)), sl())
    c = out.candidates[0]
    assert c.signals["role"] == "subject"
    assert c.signals["roles"] == ["speaker", "subject"]


def test_multiple_spans_one_candidate():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_mention(conn, "u1", "caroline", bs=0, be=8)
    add_mention(conn, "u1", "caroline", bs=40, be=48)
    out = lane_entity(mk_ctx(conn), mk_qv(("caroline",)), sl())
    (c,) = out.candidates
    assert len(c.signals["spans"]) == 2
    assert c.signals["matched_canons"] == ["caroline"]


def test_deterministic_tiebreak_and_bit_exact():
    conn = mk_conn()
    for u in ("u_b", "u_a", "u_c"):
        add_unit(conn, u)
        add_mention(conn, u, "caroline")
    set_df(conn, "caroline", 3)
    ctx = mk_ctx(conn)
    o1 = lane_entity(ctx, mk_qv(("caroline",)), sl())
    o2 = lane_entity(ctx, mk_qv(("caroline",)), sl())
    assert [c.unit_id for c in o1.candidates] == ["u_a", "u_b", "u_c"]
    assert [
        (c.unit_id, c.raw_score, c.rank) for c in o1.candidates
    ] == [(c.unit_id, c.raw_score, c.rank) for c in o2.candidates]


def test_cap_truncation_recorded():
    conn = mk_conn()
    for u in ("u1", "u2", "u3"):
        add_unit(conn, u)
        add_mention(conn, u, "caroline")
    out = lane_entity(mk_ctx(conn), mk_qv(("caroline",)), sl(cap=1))
    assert len(out.candidates) == 1
    assert out.stats["cap_truncated"] == 2
    assert out.stats["pool"] == 3
    assert out.eligible == 3


def test_zero_cap_emits_nothing():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_mention(conn, "u1", "caroline")
    out = lane_entity(mk_ctx(conn), mk_qv(("caroline",)), sl(cap=0))
    assert out.candidates == []
    assert out.status is LaneStatus.OK
    assert out.stats["cap_truncated"] == 1


def test_expansion_limit_bounded():
    conn = mk_conn()
    for i in range(12):
        add_alias(conn, "mel", f"m{i:02d}", state="active")
        add_unit(conn, f"u{i}")
        add_mention(conn, f"u{i}", f"m{i:02d}")
    out = lane_entity(mk_ctx(conn), mk_qv(("mel",)), sl())
    # ≤ 8 expansions per mention (DEFAULT_EXPANSION_LIMIT)
    assert out.stats["alias_links_used"] <= ent.DEFAULT_EXPANSION_LIMIT
    assert out.stats["expanded_canons"] <= 1 + ent.DEFAULT_EXPANSION_LIMIT
    assert len(out.candidates) <= ent.DEFAULT_EXPANSION_LIMIT


# ---------------------------------------------------------------------------
# honest statuses
# ---------------------------------------------------------------------------


def test_empty_canons_skipped():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_mention(conn, "u1", "caroline")
    out = lane_entity(mk_ctx(conn), mk_qv(()), sl())
    assert out.status is LaneStatus.SKIPPED
    assert out.reason == "no_query_entities"
    # canons that fold to nothing skip identically
    out2 = lane_entity(mk_ctx(conn), mk_qv(("", "  ")), sl())
    assert out2.status is LaneStatus.SKIPPED


def test_missing_tables_unavailable():
    conn = sqlite3.connect(":memory:")
    out = lane_entity(mk_ctx(conn), mk_qv(("x",)), sl())
    assert out.status is LaneStatus.UNAVAILABLE
    assert out.reason == "no_entity_tables"

    # units without entity_mentions still unavailable
    conn2 = sqlite3.connect(":memory:")
    conn2.execute("CREATE TABLE units (unit_id TEXT)")
    out2 = lane_entity(mk_ctx(conn2), mk_qv(("x",)), sl())
    assert out2.status is LaneStatus.UNAVAILABLE
    assert out2.reason == "no_entity_tables"


def test_no_read_snapshot_unavailable():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_mention(conn, "u1", "caroline")
    ctx = mk_ctx(conn, store=object())  # no .conn anywhere
    out = lane_entity(ctx, mk_qv(("caroline",)), sl())
    assert out.status is LaneStatus.UNAVAILABLE
    assert out.reason == "no_read_snapshot"


def test_generation_unpinned_unavailable():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_mention(conn, "u1", "caroline")
    out = lane_entity(
        mk_ctx(conn, gen=None), mk_qv(("caroline",)), sl()
    )
    assert out.status is LaneStatus.UNAVAILABLE
    assert out.reason == "generation_unpinned"


def test_missing_alias_table_is_identity_expansion():
    conn = mk_conn()
    conn.execute("DROP TABLE entity_aliases_v7")
    add_unit(conn, "u1"), add_unit(conn, "u2")
    add_mention(conn, "u1", "melanie")
    add_mention(conn, "u2", "mel")
    out = lane_entity(mk_ctx(conn), mk_qv(("melanie",)), sl())
    assert out.stats["aliases"] == "table_absent"
    assert {c.unit_id for c in out.candidates} == {"u1"}


def test_missing_canon_table_floors_everything():
    conn = mk_conn()
    conn.execute("DROP TABLE entity_canon")
    add_unit(conn, "u1")
    add_mention(conn, "u1", "caroline")
    out = lane_entity(mk_ctx(conn), mk_qv(("caroline",)), sl())
    assert out.stats["df_source"] == "absent"
    assert len(out.candidates) == 1
    assert out.stats["df_floor_canons"] == 1


def test_deadline_cut_scan_is_partial(monkeypatch):
    conn = mk_conn()
    monkeypatch.setattr(ent, "_PAGE", 4)  # small pages so the cut bites
    for i in range(8):
        add_unit(conn, f"u{i}")
        add_mention(conn, f"u{i}", "caroline")
    set_df(conn, "caroline", 8)
    ticks = iter([0.0, 0.0, 0.0] + [1e9] * 100)
    monkeypatch.setattr(ent, "_monotonic", lambda: next(ticks))
    out = lane_entity(mk_ctx(conn), mk_qv(("caroline",)), sl(ms=1.0))
    assert out.status is LaneStatus.PARTIAL
    assert out.reason == "deadline"
    assert out.stats["deadline"] is True
    assert out.examined == 4  # one page scanned, then the clock expired
    assert out.eligible <= 4


def test_examined_and_eligible_counts_honest():
    conn = mk_conn()
    for u in ("u1", "u2", "u3"):
        add_unit(conn, u)
        add_mention(conn, u, "caroline")
    add_mention(conn, "u1", "caroline", bs=30, be=38)  # second span
    out = lane_entity(mk_ctx(conn), mk_qv(("caroline",)), sl())
    assert out.examined == 4  # posting rows scanned
    assert out.eligible == 3  # distinct eligible units


def test_lane_registered():
    assert LANE_REGISTRY.get(LaneName.ENT) is lane_entity


def test_expansion_map_reported_in_stats():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_mention(conn, "u1", "mel")
    add_alias(conn, "melanie", "mel", state="active")
    out = lane_entity(mk_ctx(conn), mk_qv(("melanie",)), sl())
    assert out.stats["expansion"] == {"melanie": ["mel"]}
    assert out.stats["formula_status"] == "provisional/v7-r0"
    assert out.stats["lane_version"] == ent.LANE_VERSION


# ---------------------------------------------------------------------------
# V8-10.03 conjunctive emission + V8-10.04 per-canon quota (§25.7: K61, K62)
# ---------------------------------------------------------------------------


def test_k61_joint_units_emit_first_and_marked():
    """Two resolved canons → the unit mentioning BOTH emits first with
    ``signals["joint"]=True`` — structural ordering, not score order."""
    conn = mk_conn()
    for i in range(7):  # 10-unit corpus; bob lands dominant (>30% df/N)
        add_unit(conn, f"f{i}")
    add_unit(conn, "u_joint")
    add_unit(conn, "u_a")
    add_unit(conn, "u_b")
    add_mention(conn, "u_joint", "alice")
    add_mention(conn, "u_joint", "bob", bs=20, be=23)
    add_mention(conn, "u_a", "alice")
    add_mention(conn, "u_a", "ally", bs=20, be=24)  # alias of alice
    add_mention(conn, "u_b", "bob")
    for i in range(6):
        add_mention(conn, f"f{i}", "bob")
    set_df(conn, "alice", 2)
    set_df(conn, "ally", 1)
    set_df(conn, "bob", 8)
    add_alias(conn, "alice", "ally", state="active")
    out = lane_entity(mk_ctx(conn), mk_qv(("alice", "bob")), sl())
    ids = by_id(out)
    assert out.candidates[0].unit_id == "u_joint"
    assert ids["u_joint"].signals["joint"] is True
    assert "joint" not in ids["u_a"].signals
    assert "joint" not in ids["u_b"].signals
    # u_a outscores u_joint on raw idf (two rare matched canons beat
    # rare + capped-dominant) — the joint still ranks first.
    assert ids["u_a"].raw_score > ids["u_joint"].raw_score
    # matching a canon AND its alias covers ONE resolved entity — u_a is
    # not joint.
    assert ids["u_a"].signals["query_canons_covered"] == 1
    assert out.stats["joint"] == 1
    assert out.stats["joint_pool"] == 1


def test_k61_partial_coverage_is_not_joint():
    """With 3 resolved canons, covering 2 is a boosted single — only
    all-canons-present marks joint."""
    conn = mk_conn()
    for u, cs in (
        ("u_abc", ("a", "b", "c")),
        ("u_ab", ("a", "b")),
        ("u_a", ("a",)),
    ):
        add_unit(conn, u)
        for j, c in enumerate(cs):
            add_mention(conn, u, c, bs=10 * j, be=10 * j + 4)
    set_df(conn, "a", 3)
    set_df(conn, "b", 2)
    set_df(conn, "c", 1)
    out = lane_entity(mk_ctx(conn), mk_qv(("a", "b", "c")), sl())
    ids = by_id(out)
    assert out.candidates[0].unit_id == "u_abc"
    assert ids["u_abc"].signals["joint"] is True
    assert ids["u_ab"].signals["query_canons_covered"] == 2
    assert "joint" not in ids["u_ab"].signals
    assert out.stats["joint"] == 1 and out.stats["joint_pool"] == 1


def test_k61_single_canon_query_never_marks_joint():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_mention(conn, "u1", "melanie")
    add_mention(conn, "u1", "mel", bs=20, be=23)
    add_alias(conn, "melanie", "mel", state="active")
    out = lane_entity(mk_ctx(conn), mk_qv(("melanie",)), sl())
    (c,) = out.candidates
    assert "joint" not in c.signals
    assert out.stats["joint"] == 0
    assert out.stats["joint_pool"] == 0


def test_k61_ineligible_joint_never_emitted():
    """Fail-closed: the conjunctive phase rides the same eligible pool —
    a gated joint unit is counted, never emitted."""
    conn = mk_conn()
    for u in ("j1", "j2", "a1"):
        add_unit(conn, u)
    for u in ("j1", "j2"):
        add_mention(conn, u, "a")
        add_mention(conn, u, "b", bs=20, be=24)
    add_mention(conn, "a1", "a")
    out = lane_entity(
        mk_ctx(conn, eligible={"j2", "a1"}), mk_qv(("a", "b")), sl()
    )
    ids = [c.unit_id for c in out.candidates]
    assert ids == ["j2", "a1"]
    assert out.stats["ineligible"] == 1
    assert out.stats["joint"] == 1 and out.stats["joint_pool"] == 1


def test_k61_future_generation_mention_invisible():
    """The joint verdict is computed on fenced mentions: a canon whose
    only mention is beyond the fence does not complete the conjunction."""
    conn = mk_conn()
    add_unit(conn, "u_x")
    add_unit(conn, "u_y")
    add_mention(conn, "u_x", "a", gen=1)
    add_mention(conn, "u_x", "b", gen=9, bs=20, be=24)
    add_mention(conn, "u_y", "a", gen=1)
    add_mention(conn, "u_y", "b", gen=1, bs=20, be=24)
    out3 = lane_entity(mk_ctx(conn, gen=3), mk_qv(("a", "b")), sl())
    ids3 = by_id(out3)
    assert out3.candidates[0].unit_id == "u_y"
    assert ids3["u_y"].signals["joint"] is True
    assert "joint" not in ids3["u_x"].signals  # b mention beyond fence
    out9 = lane_entity(mk_ctx(conn, gen=9), mk_qv(("a", "b")), sl())
    assert all(c.signals.get("joint") for c in out9.candidates)


def test_k62_prolific_canon_capped_at_quota():
    """Default quota (§23 prior 50): a df-61 canon contributes at most 50
    singles; the rare canon and the joint unit are untouched."""
    conn = mk_conn()
    add_unit(conn, "u_j")
    add_mention(conn, "u_j", "prolific")
    add_mention(conn, "u_j", "rare", bs=10, be=14)
    for i in range(60):
        add_unit(conn, f"p{i:02d}")
        add_mention(conn, f"p{i:02d}", "prolific")
    for i in range(3):
        add_unit(conn, f"r{i}")
        add_mention(conn, f"r{i}", "rare")
    set_df(conn, "prolific", 61)
    set_df(conn, "rare", 4)
    out = lane_entity(
        mk_ctx(conn), mk_qv(("prolific", "rare")), sl(cap=200)
    )
    ids = [c.unit_id for c in out.candidates]
    assert ids[0] == "u_j"
    assert by_id(out)["u_j"].signals["joint"] is True
    prolific = [u for u in ids if u.startswith("p")]
    rare = [u for u in ids if u.startswith("r")]
    assert len(prolific) == 50
    assert set(prolific) == {f"p{i:02d}" for i in range(50)}
    assert len(rare) == 3
    assert len(ids) == 54
    assert out.stats["per_canon_quota"] == 50
    assert out.stats["quota_dropped"] == 10
    assert out.stats["per_canon"]["prolific"] == {
        "emitted": 50,
        "dropped": 10,
    }
    assert out.stats["per_canon"]["rare"] == {"emitted": 3, "dropped": 0}
    assert out.stats["joint"] == 1


def test_k62_joint_units_exempt_from_quota():
    conn = mk_conn()
    for i in range(3):
        add_unit(conn, f"j{i}")
        add_mention(conn, f"j{i}", "a")
        add_mention(conn, f"j{i}", "b", bs=20, be=24)
    for i in range(4):
        add_unit(conn, f"a{i}")
        add_mention(conn, f"a{i}", "a")
        add_unit(conn, f"b{i}")
        add_mention(conn, f"b{i}", "b")
    set_df(conn, "a", 7)
    set_df(conn, "b", 7)
    ctx = mk_ctx(conn)
    ctx.manifest[ent.PER_CANON_QUOTA_ARM] = 1
    out = lane_entity(ctx, mk_qv(("a", "b")), sl())
    ids = [c.unit_id for c in out.candidates]
    assert ids[:3] == ["j0", "j1", "j2"]  # all joints, quota-exempt
    assert sum(1 for u in ids if u.startswith("a")) == 1
    assert sum(1 for u in ids if u.startswith("b")) == 1
    assert len(ids) == 5
    assert out.stats["joint"] == 3
    assert out.stats["quota_dropped"] == 6
    assert out.stats["per_canon"]["a"]["dropped"] == 3
    assert out.stats["per_canon"]["b"]["dropped"] == 3


def test_k62_multi_covering_single_charges_each_canon():
    """A unit covering 2 of 3 canons is a non-joint single: it charges
    EVERY canon it covers, so each canon's contribution stays ≤ quota."""
    conn = mk_conn()
    for u in ("u_ab", "u_a", "u_b", "u_c"):
        add_unit(conn, u)
    add_mention(conn, "u_ab", "a")
    add_mention(conn, "u_ab", "b", bs=10, be=14)
    add_mention(conn, "u_a", "a")
    add_mention(conn, "u_b", "b")
    add_mention(conn, "u_c", "c")
    set_df(conn, "a", 2)
    set_df(conn, "b", 2)
    set_df(conn, "c", 1)
    ctx = mk_ctx(conn)
    ctx.manifest[ent.PER_CANON_QUOTA_ARM] = 1
    out = lane_entity(ctx, mk_qv(("a", "b", "c")), sl())
    ids = [c.unit_id for c in out.candidates]
    assert set(ids) == {"u_ab", "u_c"}
    assert "joint" not in by_id(out)["u_ab"].signals
    assert out.stats["per_canon"]["a"] == {"emitted": 1, "dropped": 1}
    assert out.stats["per_canon"]["b"] == {"emitted": 1, "dropped": 1}
    assert out.stats["per_canon"]["c"] == {"emitted": 1, "dropped": 0}
    assert out.stats["quota_dropped"] == 2


def test_k62_quota_zero_disables_singles_keeps_joints():
    conn = mk_conn()
    add_unit(conn, "j0")
    add_mention(conn, "j0", "a")
    add_mention(conn, "j0", "b", bs=9, be=12)
    for u in ("a1", "a2"):
        add_unit(conn, u)
        add_mention(conn, u, "a")
    ctx = mk_ctx(conn)
    ctx.manifest[ent.PER_CANON_QUOTA_ARM] = 0
    out = lane_entity(ctx, mk_qv(("a", "b")), sl())
    assert [c.unit_id for c in out.candidates] == ["j0"]
    assert out.stats["quota_dropped"] == 2
    assert out.stats["per_canon"]["a"] == {"emitted": 0, "dropped": 2}


def test_k62_quota_resolves_via_policy_params():
    """The §23 arm channel: a params mapping on the policy object wins
    over the manifest fallback channel."""
    conn = mk_conn()
    add_unit(conn, "j0")
    add_mention(conn, "j0", "a")
    add_mention(conn, "j0", "b", bs=9, be=12)
    for u in ("a1", "a2", "a3"):
        add_unit(conn, u)
        add_mention(conn, u, "a")
    ctx = mk_ctx(conn)
    object.__setattr__(
        ctx.policy, "params", {ent.PER_CANON_QUOTA_ARM: 1}
    )
    ctx.manifest[ent.PER_CANON_QUOTA_ARM] = 3  # policy channel wins
    out = lane_entity(ctx, mk_qv(("a", "b")), sl())
    assert out.stats["per_canon_quota"] == 1
    assert sum(
        1 for c in out.candidates if not c.signals.get("joint")
    ) == 1


def test_k62_mistyped_quota_fails_loudly():
    """A mistyped arm raises VALIDATION — never silently reconfigures
    (the ``graph_max_hops`` convention)."""
    conn = mk_conn()
    add_unit(conn, "u1")
    add_mention(conn, "u1", "a")
    for bad in ("lots", -1, 1.5, True):
        ctx = mk_ctx(conn)
        ctx.manifest[ent.PER_CANON_QUOTA_ARM] = bad
        with pytest.raises(VerbatimError):
            lane_entity(ctx, mk_qv(("a",)), sl())
    # integral floats (JSON manifests) are accepted
    ctx = mk_ctx(conn)
    ctx.manifest[ent.PER_CANON_QUOTA_ARM] = 1.0
    out = lane_entity(ctx, mk_qv(("a",)), sl())
    assert out.stats["per_canon_quota"] == 1
    assert len(out.candidates) == 1
