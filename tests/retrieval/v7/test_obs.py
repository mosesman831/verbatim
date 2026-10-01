"""Lane tests for ``verbatim/retrieval/v7/obs.py`` (w-lane-obs, wave B).

Mirror-DDL approach per the wave brief: §30 ``units`` /
``observations_v7`` / ``profiles_v7`` / ``standing_queries`` are created
inline — ``schema_v7`` is a concurrent worker's file. ``QueryViewV7`` is
hand-built; these tests unit-test the LANE (matching, proof-count
scoring, stale/dirty honesty flags, eligibility closure, generation
fence, intent gate, cap/deadline discipline).
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

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
from verbatim.retrieval.v7 import obs as obslane
from verbatim.retrieval.v7.lanes_base import LANE_REGISTRY

# ---------------------------------------------------------------------------
# §30 mirror DDL (normative minimum column lists)
# ---------------------------------------------------------------------------

MIRROR_DDL = """
CREATE TABLE units (
  unit_id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL,
  revision INTEGER NOT NULL,
  scope_id TEXT NOT NULL,
  kind TEXT,
  parent_unit_id TEXT,
  session_id TEXT,
  seq INTEGER,
  speaker_canon TEXT,
  perspective TEXT,
  recorded_at_us INTEGER,
  occurred_start_us INTEGER,
  occurred_end_us INTEGER,
  occurred_precision TEXT,
  occurred_source TEXT,
  byte_start INTEGER,
  byte_end INTEGER,
  generation INTEGER NOT NULL
);
CREATE TABLE observations_v7 (
  obs_id TEXT PRIMARY KEY,
  scope_id TEXT NOT NULL,
  slot TEXT,
  text TEXT,
  producer TEXT,
  proof_count INTEGER NOT NULL DEFAULT 0,
  support_refs_json TEXT,
  contradict_refs_json TEXT,
  first_us INTEGER,
  last_us INTEGER,
  stale INTEGER NOT NULL DEFAULT 0,
  generation INTEGER NOT NULL
);
CREATE TABLE profiles_v7 (
  scope_id TEXT NOT NULL,
  subject_canon TEXT NOT NULL,
  slot TEXT NOT NULL,
  generation INTEGER NOT NULL,
  value TEXT,
  status TEXT,
  support_refs_json TEXT,
  updated_us INTEGER,
  PRIMARY KEY (scope_id, subject_canon, slot, generation)
);
CREATE TABLE standing_queries (
  sq_id TEXT PRIMARY KEY,
  scope_id TEXT NOT NULL,
  principal TEXT,
  query TEXT,
  filters_json TEXT,
  pack_blob BLOB,
  pack_digest TEXT,
  built_generation INTEGER,
  dirty INTEGER NOT NULL DEFAULT 0,
  built_us INTEGER,
  generation INTEGER NOT NULL
);
"""

SCOPE = "s1"
GEN = 1


def us(y: int, m: int = 1, d: int = 1) -> int:
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1_000_000)


def mk_conn(*, obs=True, profiles=True, sq=True, units=True) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    if units:
        conn.execute(
            "CREATE TABLE units ("
            " unit_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,"
            " revision INTEGER NOT NULL, scope_id TEXT NOT NULL, kind TEXT,"
            " parent_unit_id TEXT, session_id TEXT, seq INTEGER,"
            " speaker_canon TEXT, perspective TEXT, recorded_at_us INTEGER,"
            " occurred_start_us INTEGER, occurred_end_us INTEGER,"
            " occurred_precision TEXT, occurred_source TEXT,"
            " byte_start INTEGER, byte_end INTEGER, generation INTEGER NOT NULL)"
        )
    if obs:
        conn.execute(
            "CREATE TABLE observations_v7 ("
            " obs_id TEXT PRIMARY KEY, scope_id TEXT NOT NULL, slot TEXT,"
            " text TEXT, producer TEXT, proof_count INTEGER NOT NULL DEFAULT 0,"
            " support_refs_json TEXT, contradict_refs_json TEXT,"
            " first_us INTEGER, last_us INTEGER,"
            " stale INTEGER NOT NULL DEFAULT 0, generation INTEGER NOT NULL)"
        )
    if profiles:
        conn.execute(
            "CREATE TABLE profiles_v7 ("
            " scope_id TEXT NOT NULL, subject_canon TEXT NOT NULL,"
            " slot TEXT NOT NULL, generation INTEGER NOT NULL, value TEXT,"
            " status TEXT, support_refs_json TEXT, updated_us INTEGER,"
            " PRIMARY KEY (scope_id, subject_canon, slot, generation))"
        )
    if sq:
        conn.execute(
            "CREATE TABLE standing_queries ("
            " sq_id TEXT PRIMARY KEY, scope_id TEXT NOT NULL, principal TEXT,"
            " query TEXT, filters_json TEXT, pack_blob BLOB,"
            " pack_digest TEXT, built_generation INTEGER,"
            " dirty INTEGER NOT NULL DEFAULT 0, built_us INTEGER,"
            " generation INTEGER NOT NULL)"
        )
    return conn


def add_unit(conn, unit_id, *, scope=SCOPE, gen=GEN, source="src", rev=1):
    conn.execute(
        "INSERT INTO units(unit_id, source_id, revision, scope_id, kind,"
        " recorded_at_us, generation) VALUES (?,?,?,?,'turn',?,?)",
        (unit_id, source, rev, scope, us(2024, 1, 1), gen),
    )


def refs(*unit_ids):
    return json.dumps([{"unit_id": u, "quote": "q"} for u in unit_ids])


def add_obs(
    conn,
    obs_id,
    *,
    slot="",
    text="",
    producer="t0",
    proof=0,
    support=(),
    contradict=(),
    stale=0,
    scope=SCOPE,
    gen=GEN,
):
    conn.execute(
        "INSERT INTO observations_v7(obs_id, scope_id, slot, text, producer,"
        " proof_count, support_refs_json, contradict_refs_json, first_us,"
        " last_us, stale, generation) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            obs_id,
            scope,
            slot,
            text,
            producer,
            proof,
            refs(*support) if support else None,
            refs(*contradict) if contradict else None,
            us(2023, 1, 1),
            us(2024, 1, 1),
            stale,
            gen,
        ),
    )


def add_profile(
    conn,
    subject,
    slot,
    *,
    value="",
    status="current",
    support=(),
    scope=SCOPE,
    gen=GEN,
):
    conn.execute(
        "INSERT INTO profiles_v7(scope_id, subject_canon, slot, generation,"
        " value, status, support_refs_json, updated_us)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (scope, subject, slot, gen, value, status,
         refs(*support) if support else None, us(2024, 1, 1)),
    )


def add_sq(
    conn,
    sq_id,
    *,
    query="",
    pack=None,
    dirty=0,
    scope=SCOPE,
    gen=GEN,
):
    blob = None
    if pack is not None:
        blob = json.dumps({"items": [{"unit_id": u} for u in pack]}).encode()
    conn.execute(
        "INSERT INTO standing_queries(sq_id, scope_id, principal, query,"
        " filters_json, pack_blob, pack_digest, built_generation, dirty,"
        " built_us, generation) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (sq_id, scope, "p1", query, "{}", blob, "dig", gen, dirty,
         us(2024, 1, 1), gen),
    )


def mk_qv(
    primary=IntentClass.LOOKUP,
    *,
    classes=None,
    terms=(),
    canons=(),
    speaker=None,
    q="test query",
):
    norm = NormAnalysis(
        analyzer_id="norm/v2",
        terms=tuple(NormTerm(t, "text", 0, len(t)) for t in terms),
        identifiers=(),
        text=q,  # folded projection; the lane re-folds anyway
    )
    intent = IntentResult(primary=primary, classes=classes or (primary,))
    return QueryViewV7(
        query=q,
        norm=norm,
        intent=intent,
        entity_canons=tuple(canons),
        speaker_canon=speaker,
        query_time_us=us(2024, 1, 1),
    )


def mk_ctx(conn, *, eligible=None, gen=GEN, scope=SCOPE):
    if eligible is None:
        eligible = lambda row: True  # noqa: E731 - allow-all fixture
    policy = RetrievalPolicyV7(
        policy_id="retrieval_policy/v7",
        profile="test",
        lanes=(LaneName.OBS,),
        lane_weights={},
    )
    return LaneContextV7(
        store=conn,
        scope_id=scope,
        generation=gen,
        eligible=eligible,
        query_time_us=us(2024, 1, 1),
        profile="test",
        budget=BudgetClass.MID,
        policy=policy,
    )


def sl(cap=50, ms=10_000):
    return LaneSlice(deadline_ms=ms, cap=cap)


def by_id(out):
    return {c.unit_id: c for c in out.candidates}


# ---------------------------------------------------------------------------
# observation matching
# ---------------------------------------------------------------------------


def test_slot_term_match_surfaces_obs():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_unit(conn, "u2")
    add_obs(conn, "o_home", slot="home_city", text="Alice lives in Portland",
            proof=3, support=("u1",))
    add_obs(conn, "o_pet", slot="pet_name", text="her cat is Mochi",
            proof=2, support=("u2",))
    qv = mk_qv(terms=("city", "where", "alice", "live"))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.OK
    ids = by_id(out)
    assert "u1" in ids  # 'city' occurs inside folded slot 'home_city'
    assert "u2" not in ids
    assert ids["u1"].signals["obs_id"] == "o_home"
    assert ids["u1"].signals["slot"] == "home_city"
    assert ids["u1"].signals["proof_count"] == 3


def test_slot_normalized_equality_and_casefold():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_obs(conn, "o1", slot="Favorite Color", support=("u1",))
    qv = mk_qv(terms=("favorite", "color"))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert "u1" in by_id(out)


def test_canon_in_text_matches_without_term_overlap():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_obs(conn, "o1", slot="residence",
            text="alice moved to Portland", support=("u1",))
    qv = mk_qv(terms=("where",), canons=("alice",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert "u1" in by_id(out)


def test_canon_in_support_refs_matches():
    conn = mk_conn()
    add_unit(conn, "u1")
    conn.execute(
        "INSERT INTO observations_v7(obs_id, scope_id, slot, text, producer,"
        " proof_count, support_refs_json, generation)"
        " VALUES ('o1', ?, 'slotx', 'no canon here', 't0', 1, ?, ?)",
        (SCOPE, json.dumps([{"unit_id": "u1", "subject_canon": "alice"}]), GEN),
    )
    qv = mk_qv(terms=("nomatch",), canons=("alice",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert "u1" in by_id(out)


def test_speaker_canon_matches_obs_text():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_obs(conn, "o1", slot="diet", text="the user is vegetarian",
            support=("u1",))
    qv = mk_qv(terms=("what", "eat"), speaker="the user")
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert "u1" in by_id(out)


def test_no_match_keys_emits_nothing():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_obs(conn, "o1", slot="home_city", support=("u1",))
    qv = mk_qv(terms=(), canons=())
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.OK
    assert out.candidates == [] and out.eligible == 0


# ---------------------------------------------------------------------------
# proof-count scoring + stale ranking
# ---------------------------------------------------------------------------


def test_proof_count_scales_score_and_orders():
    conn = mk_conn()
    add_unit(conn, "u_low")
    add_unit(conn, "u_high")
    add_obs(conn, "o_low", slot="home", proof=1, support=("u_low",))
    add_obs(conn, "o_high", slot="home", proof=20, support=("u_high",))
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    ids = by_id(out)
    assert ids["u_high"].raw_score == pytest.approx(1.0 + 0.2 * math.log1p(20))
    assert ids["u_low"].raw_score == pytest.approx(1.0 + 0.2 * math.log1p(1))
    assert out.candidates[0].unit_id == "u_high"
    assert ids["u_high"].rank < ids["u_low"].rank


def test_stale_flagged_and_ranked_below_fresh():
    conn = mk_conn()
    add_unit(conn, "u_fresh")
    add_unit(conn, "u_stale")
    # stale obs has MORE proof — freshness still wins the ordering
    add_obs(conn, "o_stale", slot="home", proof=50, stale=1, support=("u_stale",))
    add_obs(conn, "o_fresh", slot="home", proof=1, support=("u_fresh",))
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    ids = by_id(out)
    assert ids["u_stale"].signals["stale"] == 1
    assert ids["u_fresh"].signals["stale"] == 0
    assert ids["u_fresh"].rank < ids["u_stale"].rank
    assert out.candidates[-1].unit_id == "u_stale"


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------


def test_profile_subject_match_emits_slot_value_status():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_profile(conn, "alice", "home_city", value="Portland",
                status="current", support=("u1",))
    qv = mk_qv(terms=("where",), canons=("alice",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    ids = by_id(out)
    assert "u1" in ids
    sig = ids["u1"].signals
    assert sig["profile_slot"] == "home_city"
    assert sig["value"] == "Portland"
    assert sig["status"] == "current"


def test_profile_other_subject_not_emitted():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_profile(conn, "bob", "home_city", value="Seattle", support=("u1",))
    qv = mk_qv(terms=("where",), canons=("alice",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert "u1" not in by_id(out)


def test_profile_casefolded_subject_match():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_profile(conn, "Alice", "pet", value="Mochi", support=("u1",))
    qv = mk_qv(terms=("pet",), canons=("ALICE",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert "u1" in by_id(out)


# ---------------------------------------------------------------------------
# standing queries
# ---------------------------------------------------------------------------


def test_standing_query_exact_match_seeds_candidates():
    conn = mk_conn()
    add_unit(conn, "u_a")
    add_unit(conn, "u_b")
    add_sq(conn, "sq1", query="what is alice's city", pack=("u_a", "u_b"))
    add_sq(conn, "sq2", query="a different question", pack=("u_a",))
    qv = mk_qv(q="what is alice's city", terms=("what", "alice", "city"))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    ids = by_id(out)
    assert {"u_a", "u_b"} <= set(ids)
    assert ids["u_a"].signals["standing"] == "sq1"
    assert ids["u_b"].signals["standing"] == "sq1"
    assert out.stats["standing_matched"] == 1


def test_standing_query_normalized_casefold_match():
    conn = mk_conn()
    add_unit(conn, "u_a")
    add_sq(conn, "sq1", query="What Is Alice's City", pack=("u_a",))
    qv = mk_qv(q="what is alice's city")
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert "u_a" in by_id(out)


def test_standing_query_dirty_still_emits_flagged():
    conn = mk_conn()
    add_unit(conn, "u_a")
    add_sq(conn, "sq1", query="q", pack=("u_a",), dirty=1)
    qv = mk_qv(q="q")
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    ids = by_id(out)
    assert "u_a" in ids
    assert ids["u_a"].signals["standing_dirty"] == 1


def test_standing_query_no_match_seeds_nothing():
    conn = mk_conn()
    add_unit(conn, "u_a")
    add_sq(conn, "sq1", query="unrelated question", pack=("u_a",))
    qv = mk_qv(q="totally different")
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert "u_a" not in by_id(out)
    assert out.stats["standing_matched"] == 0


def test_standing_unparseable_blob_seeds_nothing():
    conn = mk_conn()
    add_unit(conn, "u_a")
    add_sq(conn, "sq1", query="q", pack=None)
    conn.execute("UPDATE standing_queries SET pack_blob = ? WHERE sq_id='sq1'",
                 (b"\x00\xff not json",))
    qv = mk_qv(q="q")
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.OK
    assert "u_a" not in by_id(out)


# ---------------------------------------------------------------------------
# eligibility / closure (V7-14.08)
# ---------------------------------------------------------------------------


def test_obs_with_all_supports_held_emits_nothing():
    conn = mk_conn()
    add_unit(conn, "u_held")
    add_unit(conn, "u_ok")
    add_obs(conn, "o_held", slot="home", support=("u_held",))
    add_obs(conn, "o_ok", slot="home", support=("u_ok",))
    ctx = mk_ctx(conn, eligible=lambda r: r["unit_id"] != "u_held")
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(ctx, qv, sl())
    ids = by_id(out)
    assert "u_ok" in ids and "u_held" not in ids
    assert out.stats["ineligible"] == 1


def test_obs_falls_through_to_first_eligible_support():
    conn = mk_conn()
    add_unit(conn, "u_held")
    add_unit(conn, "u_ok")
    add_obs(conn, "o1", slot="home", support=("u_held", "u_ok"))
    ctx = mk_ctx(conn, eligible=lambda r: r["unit_id"] != "u_held")
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(ctx, qv, sl())
    ids = by_id(out)
    assert "u_ok" in ids  # backed by its second (eligible) pin
    assert out.stats["ineligible"] == 0


def test_support_missing_from_units_is_ineligible():
    conn = mk_conn()
    add_obs(conn, "o1", slot="home", support=("u_ghost",))
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert out.candidates == []
    assert out.stats["ineligible"] == 1


def test_obs_with_no_support_refs_counted_no_support():
    conn = mk_conn()
    add_obs(conn, "o1", slot="home", support=())
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert out.candidates == []
    assert out.stats["no_support"] == 1


def test_profile_first_unit_held_emits_nothing():
    conn = mk_conn()
    add_unit(conn, "u_held")
    add_profile(conn, "alice", "home_city", value="x", support=("u_held",))
    ctx = mk_ctx(conn, eligible=lambda r: r["unit_id"] != "u_held")
    qv = mk_qv(terms=("home",), canons=("alice",))
    out = obslane.lane_obs(ctx, qv, sl())
    assert "u_held" not in by_id(out)
    assert out.stats["ineligible"] == 1


def test_eligibility_as_set_object():
    conn = mk_conn()
    add_unit(conn, "u_a")
    add_unit(conn, "u_b")
    add_obs(conn, "o_a", slot="home", support=("u_a",))
    add_obs(conn, "o_b", slot="home", support=("u_b",))
    ctx = mk_ctx(conn, eligible={"u_a"})
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(ctx, qv, sl())
    assert set(by_id(out)) == {"u_a"}


# ---------------------------------------------------------------------------
# fences
# ---------------------------------------------------------------------------


def test_generation_fence():
    """V7-30.02: ``generation <= pinned`` — gen-1 rows stay visible at
    the gen-2 pin (distinct obs/unit ids, not superseded); rows above
    the pin stay fenced out."""
    conn = mk_conn()
    add_unit(conn, "u_old", gen=1)
    add_unit(conn, "u_new", gen=2)
    add_obs(conn, "o_old", slot="home", gen=1, support=("u_old",))
    add_obs(conn, "o_new", slot="home", gen=2, support=("u_new",))
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(mk_ctx(conn, gen=1), qv, sl())
    assert set(by_id(out)) == {"u_old"}  # gen-2 rows are above the pin
    out2 = obslane.lane_obs(mk_ctx(conn, gen=2), qv, sl())
    assert set(by_id(out2)) == {"u_old", "u_new"}


def test_profiles_latest_generation_wins():
    """V7-30.02: ``profiles_v7`` is versioned per
    (subject_canon, slot, generation) — after a bump re-projects the
    slot, the newest row at/below the pin is authoritative and the
    stale-generation row must not emit a second candidate."""
    conn = mk_conn()
    add_unit(conn, "u_v1", gen=1)
    add_unit(conn, "u_v2", gen=2)
    add_profile(conn, "alice", "home_city", value="portland", gen=1,
                support=("u_v1",))
    add_profile(conn, "alice", "home_city", value="seattle", gen=2,
                support=("u_v2",))
    qv = mk_qv(terms=("home",), canons=("alice",))
    out = obslane.lane_obs(mk_ctx(conn, gen=2), qv, sl())
    assert set(by_id(out)) == {"u_v2"}  # gen-2 slot projection wins
    out1 = obslane.lane_obs(mk_ctx(conn, gen=1), qv, sl())
    assert set(by_id(out1)) == {"u_v1"}  # gen-2 row above the pin


def test_scope_fence():
    conn = mk_conn()
    add_unit(conn, "u_mine", scope=SCOPE)
    add_unit(conn, "u_theirs", scope="s2")
    add_obs(conn, "o_mine", slot="home", scope=SCOPE, support=("u_mine",))
    add_obs(conn, "o_theirs", slot="home", scope="s2", support=("u_theirs",))
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert set(by_id(out)) == {"u_mine"}


def test_generation_unpinned_unavailable():
    conn = mk_conn()
    ctx = mk_ctx(conn)
    ctx.generation = None
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(ctx, qv, sl())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "generation_unpinned"


# ---------------------------------------------------------------------------
# intent gate
# ---------------------------------------------------------------------------


def test_identifier_only_intent_skips():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_obs(conn, "o1", slot="home", support=("u1",))
    qv = mk_qv(IntentClass.IDENTIFIER, terms=("home",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.SKIPPED
    assert out.reason == "intent_identifier"
    assert out.candidates == [] and out.examined == 0


def test_temporal_intents_skip():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_obs(conn, "o1", slot="home", support=("u1",))
    for primary in (
        IntentClass.TEMPORAL_POINT,
        IntentClass.TEMPORAL_RANGE,
        IntentClass.TEMPORAL_ORDER,
        IntentClass.DURATION,
        IntentClass.COUNT_AGGREGATE,
        IntentClass.HISTORY_OF,
        IntentClass.CURRENT_VALUE,
    ):
        qv = mk_qv(primary, terms=("home",))
        out = obslane.lane_obs(mk_ctx(conn), qv, sl())
        assert out.status == LaneStatus.SKIPPED, primary
        assert out.reason == "intent_temporal"


def test_preference_and_lookup_intents_run():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_obs(conn, "o1", slot="favorite_color", support=("u1",))
    for primary in (IntentClass.PREFERENCE, IntentClass.LOOKUP,
                    IntentClass.OPEN_DOMAIN):
        qv = mk_qv(primary, terms=("favorite",))
        out = obslane.lane_obs(mk_ctx(conn), qv, sl())
        assert out.status == LaneStatus.OK, primary
        assert "u1" in by_id(out)


# ---------------------------------------------------------------------------
# availability / honesty
# ---------------------------------------------------------------------------


def test_missing_artifact_tables_unavailable():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE units (unit_id TEXT PRIMARY KEY, source_id TEXT,"
        " revision INTEGER, scope_id TEXT, kind TEXT, parent_unit_id TEXT,"
        " session_id TEXT, seq INTEGER, speaker_canon TEXT, perspective TEXT,"
        " recorded_at_us INTEGER, occurred_start_us INTEGER,"
        " occurred_end_us INTEGER, occurred_precision TEXT,"
        " occurred_source TEXT, byte_start INTEGER, byte_end INTEGER,"
        " generation INTEGER NOT NULL)"
    )
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "no_obs_tables"


def test_partial_table_set_degrades_named_phase_only():
    conn = mk_conn(obs=False)  # profiles + standing exist
    add_unit(conn, "u1")
    add_profile(conn, "alice", "home_city", value="x", support=("u1",))
    qv = mk_qv(terms=("home",), canons=("alice",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.OK
    assert out.stats["observations_table"] == "missing"
    assert out.stats["profiles_table"] == "ok"
    assert "u1" in by_id(out)


def test_missing_units_table_unavailable():
    conn = mk_conn(units=False)
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "units_table_missing"


def test_no_read_snapshot_unavailable():
    ctx = mk_ctx(mk_conn())
    ctx.store = object()  # no pinned connection anywhere
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(ctx, qv, sl())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "no_read_snapshot"


def test_store_wrapper_with_conn_attribute():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_obs(conn, "o1", slot="home", support=("u1",))
    ctx = mk_ctx(conn)
    ctx.store = SimpleNamespace(conn=conn)
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(ctx, qv, sl())
    assert out.status == LaneStatus.OK and "u1" in by_id(out)


def test_eligibility_handle_missing_unavailable():
    conn = mk_conn()
    ctx = mk_ctx(conn, eligible=object())  # not callable / not a container
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(ctx, qv, sl())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "eligibility_handle_missing"


def test_empty_tables_ok_zero():
    conn = mk_conn()
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.OK
    assert out.candidates == [] and out.examined == 0 and out.eligible == 0


def test_deadline_partial():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_obs(conn, "o1", slot="home", support=("u1",))
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl(ms=0))
    assert out.status == LaneStatus.PARTIAL
    assert out.reason == "deadline"
    assert out.examined == 0


def test_cap_zero_ok_empty():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_obs(conn, "o1", slot="home", support=("u1",))
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl(cap=0))
    assert out.status == LaneStatus.OK
    assert out.candidates == []


def test_cap_truncation():
    conn = mk_conn()
    for i in range(5):
        add_unit(conn, f"u{i}")
        add_obs(conn, f"o{i}", slot="home", proof=i, support=(f"u{i}",))
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl(cap=2))
    assert len(out.candidates) == 2
    assert out.stats["overflow"] == 3
    # highest proof counts win the cap
    assert {c.unit_id for c in out.candidates} == {"u4", "u3"}
    assert [c.rank for c in out.candidates] == [1, 2]


# ---------------------------------------------------------------------------
# determinism / merging / registration
# ---------------------------------------------------------------------------


def test_deterministic_order_and_tiebreak():
    conn = mk_conn()
    for i in range(4):
        add_unit(conn, f"u{i}")
        add_obs(conn, f"o{i}", slot="home", proof=2, support=(f"u{i}",))
    qv = mk_qv(terms=("home",))
    ctx = mk_ctx(conn)
    out1 = obslane.lane_obs(ctx, qv, sl())
    out2 = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert [c.unit_id for c in out1.candidates] == [
        c.unit_id for c in out2.candidates
    ]
    # equal scores -> unit_id ascending
    assert [c.unit_id for c in out1.candidates] == sorted(
        c.unit_id for c in out1.candidates
    )
    assert [c.rank for c in out1.candidates] == [1, 2, 3, 4]


def test_merged_contributions_carry_all_signals():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_obs(conn, "o1", slot="home_city", proof=4, support=("u1",))
    add_profile(conn, "alice", "home_city", value="Portland", support=("u1",))
    add_sq(conn, "sq1", query="where does alice live", pack=("u1",))
    qv = mk_qv(q="where does alice live", terms=("home", "city", "live"),
               canons=("alice",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    ids = by_id(out)
    assert set(ids) == {"u1"}
    sig = ids["u1"].signals
    assert sig["obs_id"] == "o1"
    assert sig["profile_slot"] == "home_city"
    assert sig["standing"] == "sq1"
    assert sig["proof_count"] == 4


def test_contradict_refs_counted_in_signals():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_unit(conn, "u2")
    add_unit(conn, "u3")
    add_obs(conn, "o1", slot="home", support=("u1", "u2"),
            contradict=("u3",))
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    sig = by_id(out)["u1"].signals
    assert sig["supports"] == 2
    assert sig["contradicts"] == 1


def test_lane_registers_at_import():
    assert LANE_REGISTRY.get(LaneName.OBS) is obslane.lane_obs


def test_lane_name_and_version_stats():
    conn = mk_conn()
    qv = mk_qv(terms=("home",))
    out = obslane.lane_obs(mk_ctx(conn), qv, sl())
    assert out.lane == "obs"
    assert out.stats["lane_version"] == obslane.LANE_VERSION
    assert out.stats["formula_status"] == "provisional/v7-r0"
    assert "mode" in out.stats
