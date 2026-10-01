"""Lane tests for ``verbatim/retrieval/v7/typed.py`` (w-lane-typed, wave B).

Mirror-DDL approach per the wave brief: §30 ``units`` / ``state_facts`` /
``preferences`` / ``events_v7`` / ``t2_facts`` (and a plain FTS5
``unit_fts`` + ``unit_fts_rows`` carrier where the FTS path is exercised)
are created inline. ``QueryViewV7`` is hand-built; the lane under test is
the only production code exercised.
"""

from __future__ import annotations

import json
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
from verbatim.retrieval.v7 import lanes_base
from verbatim.retrieval.v7 import typed as tlane

# ---------------------------------------------------------------------------
# §30 mirror DDL (normative minimum column lists)
# ---------------------------------------------------------------------------

UNITS_DDL = """
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
"""

TYPED_DDL = """
CREATE TABLE state_facts (
  scope_id TEXT NOT NULL,
  state_key TEXT NOT NULL,
  unit_id TEXT NOT NULL,
  generation INTEGER NOT NULL,
  value_text TEXT,
  value_norm TEXT,
  valid_from_us INTEGER,
  valid_to_us INTEGER,
  status TEXT,
  producer TEXT,
  pins_json TEXT
);
CREATE TABLE preferences (
  scope_id TEXT NOT NULL,
  subject_canon TEXT NOT NULL,
  unit_id TEXT NOT NULL,
  generation INTEGER NOT NULL,
  object_text TEXT,
  polarity TEXT,
  strength TEXT,
  occurred_start_us INTEGER,
  pins_json TEXT
);
CREATE TABLE events_v7 (
  event_id TEXT PRIMARY KEY,
  unit_id TEXT NOT NULL,
  scope_id TEXT NOT NULL,
  subject_canon TEXT,
  predicate_lemma TEXT,
  object_text TEXT,
  polarity TEXT,
  occurred_start_us INTEGER,
  occurred_end_us INTEGER,
  precision TEXT,
  pins_json TEXT,
  rule_id TEXT,
  generation INTEGER NOT NULL
);
CREATE TABLE t2_facts (
  fact_id TEXT PRIMARY KEY,
  scope_id TEXT NOT NULL,
  unit_ids_json TEXT,
  statement TEXT,
  quotes_json TEXT,
  subject_canon TEXT,
  predicate TEXT,
  "object" TEXT,
  occurred_start_us INTEGER,
  occurred_end_us INTEGER,
  state_key TEXT,
  model_id TEXT,
  prompt_digest TEXT,
  verified INTEGER NOT NULL DEFAULT 0,
  generation INTEGER NOT NULL
);
"""

FTS_DDL = """
CREATE TABLE unit_fts_rows (
  row_id INTEGER PRIMARY KEY,
  unit_id TEXT NOT NULL,
  scope_id TEXT NOT NULL,
  generation INTEGER NOT NULL
);
CREATE VIRTUAL TABLE unit_fts USING fts5(text);
"""

SCOPE = "s1"
GEN = 1


def us(y: int, m: int = 1, d: int = 1, h: int = 0, mi: int = 0) -> int:
    return int(datetime(y, m, d, h, mi, tzinfo=timezone.utc).timestamp() * 1_000_000)


def mk_conn(fts: bool = False, typed: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(UNITS_DDL)
    if typed:
        conn.executescript(TYPED_DDL)
    if fts:
        conn.executescript(FTS_DDL)
    return conn


def add_unit(
    conn,
    unit_id,
    *,
    scope=SCOPE,
    gen=GEN,
    source="src",
    rev=1,
    speaker=None,
    occurred=(None, None),
    recorded=None,
):
    conn.execute(
        "INSERT INTO units(unit_id, source_id, revision, scope_id, kind,"
        " speaker_canon, recorded_at_us, occurred_start_us,"
        " occurred_end_us, generation) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            unit_id,
            source,
            rev,
            scope,
            "turn",
            speaker,
            recorded,
            occurred[0],
            occurred[1],
            gen,
        ),
    )


def add_state(
    conn,
    unit_id,
    state_key,
    value,
    *,
    status="current",
    vfrom=None,
    vto=None,
    norm=None,
    producer="state_keys/v1",
    pins=None,
    scope=SCOPE,
    gen=GEN,
):
    conn.execute(
        "INSERT INTO state_facts(scope_id, state_key, unit_id, generation,"
        " value_text, value_norm, valid_from_us, valid_to_us, status,"
        " producer, pins_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            scope,
            state_key,
            unit_id,
            gen,
            value,
            norm if norm is not None else (value or "").casefold(),
            vfrom,
            vto,
            status,
            producer,
            json.dumps(pins or {}),
        ),
    )


def add_pref(
    conn,
    unit_id,
    subject,
    obj,
    *,
    polarity="positive",
    strength="like_dislike",
    occ=None,
    pins=None,
    scope=SCOPE,
    gen=GEN,
):
    conn.execute(
        "INSERT INTO preferences(scope_id, subject_canon, unit_id,"
        " generation, object_text, polarity, strength, occurred_start_us,"
        " pins_json) VALUES (?,?,?,?,?,?,?,?,?)",
        (scope, subject, unit_id, gen, obj, polarity, strength, occ,
         json.dumps(pins or {})),
    )


def add_event(
    conn,
    event_id,
    unit_id,
    *,
    subject=None,
    predicate="move",
    obj="",
    polarity="affirm",
    occurred=(None, None),
    pins=None,
    scope=SCOPE,
    gen=GEN,
):
    conn.execute(
        "INSERT INTO events_v7(event_id, unit_id, scope_id, subject_canon,"
        " predicate_lemma, object_text, polarity, occurred_start_us,"
        " occurred_end_us, precision, pins_json, rule_id, generation)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            event_id,
            unit_id,
            scope,
            subject,
            predicate,
            obj,
            polarity,
            occurred[0],
            occurred[1],
            "day",
            json.dumps(pins or {}),
            "event/v1:test",
            gen,
        ),
    )


def add_t2(
    conn,
    fact_id,
    unit_ids,
    statement,
    *,
    subject=None,
    predicate=None,
    obj=None,
    state_key=None,
    verified=1,
    model="t2/test-model",
    quotes=None,
    occurred=(None, None),
    scope=SCOPE,
    gen=GEN,
):
    conn.execute(
        'INSERT INTO t2_facts(fact_id, scope_id, unit_ids_json, statement,'
        ' quotes_json, subject_canon, predicate, "object",'
        ' occurred_start_us, occurred_end_us, state_key, model_id,'
        ' prompt_digest, verified, generation)'
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            fact_id,
            scope,
            json.dumps(list(unit_ids)),
            statement,
            json.dumps(quotes or []),
            subject,
            predicate,
            obj,
            occurred[0],
            occurred[1],
            state_key,
            model,
            "pd:1",
            int(verified),
            gen,
        ),
    )


def add_fts(conn, row_id, unit_id, text, scope=SCOPE, gen=GEN):
    conn.execute("INSERT INTO unit_fts(rowid, text) VALUES (?,?)", (row_id, text))
    conn.execute(
        "INSERT INTO unit_fts_rows(row_id, unit_id, scope_id, generation)"
        " VALUES (?,?,?,?)",
        (row_id, unit_id, scope, gen),
    )


def mk_qv(
    primary=IntentClass.CURRENT_VALUE,
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
    )
    intent = IntentResult(
        primary=primary, classes=classes if classes is not None else (primary,)
    )
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
        lanes=(LaneName.TYPED,),
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


def seed_city_pair(conn):
    """alice/home_city: current portland + historical seattle + disputed."""
    add_unit(conn, "u_cur")
    add_unit(conn, "u_hist")
    add_unit(conn, "u_disp")
    add_state(conn, "u_cur", "alice/home_city", "Portland",
              status="current", vfrom=us(2023, 6, 1))
    add_state(conn, "u_hist", "alice/home_city", "Seattle",
              status="historical", vfrom=us(2019, 1, 1), vto=us(2023, 6, 1))
    add_state(conn, "u_disp", "alice/home_city", "Tacoma",
              status="disputed", vfrom=us(2020, 1, 1))


# ---------------------------------------------------------------------------
# registration + intent gate
# ---------------------------------------------------------------------------


def test_registers_at_import():
    assert lanes_base.LANE_REGISTRY.get(LaneName.TYPED) is tlane.lane_typed
    assert lanes_base.LANE_REGISTRY[LaneName.TYPED].__name__ == "lane_typed"


def test_intent_gate_skips_identifier():
    conn = mk_conn()
    seed_city_pair(conn)
    qv = mk_qv(IntentClass.IDENTIFIER, terms=("x",), canons=("alice",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.SKIPPED
    assert out.reason == "intent_not_typed"
    assert out.candidates == [] and out.examined == 0


def test_intent_gate_skips_lookup_and_why():
    conn = mk_conn()
    seed_city_pair(conn)
    for ic in (IntentClass.LOOKUP, IntentClass.WHY_CAUSAL,
               IntentClass.TEMPORAL_ORDER, IntentClass.DURATION):
        qv = mk_qv(ic, terms=("home_city",), canons=("alice",))
        out = tlane.lane_typed(mk_ctx(conn), qv, sl())
        assert out.status == LaneStatus.SKIPPED, ic
        assert out.reason == "intent_not_typed"


def test_intent_gate_union_of_classes():
    conn = mk_conn()
    seed_city_pair(conn)
    # primary is identifier but a secondary class is fact-shaped -> runs
    qv = mk_qv(
        IntentClass.IDENTIFIER,
        classes=(IntentClass.IDENTIFIER, IntentClass.CURRENT_VALUE),
        terms=("home_city",),
        canons=("alice",),
    )
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.OK
    assert "u_cur" in by_id(out)


def test_every_typed_intent_runs():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_state(conn, "u1", "alice/home_city", "Portland")
    for ic in (
        IntentClass.CURRENT_VALUE,
        IntentClass.HISTORY_OF,
        IntentClass.PREFERENCE,
        IntentClass.COMPARISON,
        IntentClass.OPEN_DOMAIN,
        IntentClass.TEMPORAL_POINT,
        IntentClass.TEMPORAL_RANGE,
    ):
        qv = mk_qv(ic, terms=("home_city",), canons=("alice",))
        out = tlane.lane_typed(mk_ctx(conn), qv, sl())
        assert out.status == LaneStatus.OK, ic
        assert "u1" in by_id(out), ic


# ---------------------------------------------------------------------------
# subject matching + term fallback
# ---------------------------------------------------------------------------


def test_subject_canon_exact_match_preferences_events():
    conn = mk_conn()
    add_unit(conn, "u_pa")
    add_unit(conn, "u_pb")
    add_unit(conn, "u_ea")
    add_pref(conn, "u_pa", "alice", "sushi", strength="love_hate")
    add_pref(conn, "u_pb", "bob", "olives")
    add_event(conn, "e1", "u_ea", subject="alice", predicate="move")

    qv = mk_qv(IntentClass.PREFERENCE, canons=("alice",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    ids = by_id(out)
    assert "u_pa" in ids and "u_ea" in ids
    assert "u_pb" not in ids
    # exact canon equality — 'alicia' does not match 'alice'
    qv2 = mk_qv(IntentClass.PREFERENCE, canons=("alicia",))
    out2 = tlane.lane_typed(mk_ctx(conn), qv2, sl())
    assert by_id(out2) == {}


def test_state_key_subject_prefix_match():
    conn = mk_conn()
    add_unit(conn, "u_a")
    add_unit(conn, "u_b")
    add_state(conn, "u_a", "alice/home_city", "Portland")
    add_state(conn, "u_b", "bob/home_city", "Denver")
    # 'alice2' must NOT prefix-match 'alice/…' (boundary at the slash)
    add_unit(conn, "u_a2")
    add_state(conn, "u_a2", "alice2/home_city", "Nowhere")

    qv = mk_qv(IntentClass.CURRENT_VALUE, canons=("alice",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    ids = by_id(out)
    assert "u_a" in ids and "u_b" not in ids and "u_a2" not in ids
    assert ids["u_a"].signals["subject_match"] == 1


def test_speaker_canon_joins_subject_match():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_pref(conn, "u1", "user", "sushi")
    qv = mk_qv(IntentClass.PREFERENCE, speaker="user")
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    assert "u1" in by_id(out)


def test_open_domain_term_fallback_on_state_key_family():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_unit(conn, "u2")
    add_state(conn, "u1", "bob/home_city", "Denver")
    add_state(conn, "u2", "bob/favorite_color", "teal")
    qv = mk_qv(IntentClass.OPEN_DOMAIN, terms=("home_city",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    ids = by_id(out)
    assert "u1" in ids and "u2" not in ids
    assert "home_city" in ids["u1"].signals["matched_terms"]


def test_term_match_on_value_norm():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_unit(conn, "u2")
    add_state(conn, "u1", "bob/home_city", "Portland", norm="portland")
    add_state(conn, "u2", "bob/home_country", "USA", norm="usa")
    qv = mk_qv(IntentClass.OPEN_DOMAIN, terms=("portland",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    assert set(by_id(out)) == {"u1"}


def test_term_match_on_event_predicate():
    conn = mk_conn()
    add_unit(conn, "u_mv")
    add_unit(conn, "u_rd")
    add_event(conn, "e1", "u_mv", subject="bob", predicate="move")
    add_event(conn, "e2", "u_rd", subject="bob", predicate="read")
    qv = mk_qv(IntentClass.TEMPORAL_POINT, terms=("move",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    assert set(by_id(out)) == {"u_mv"}
    assert by_id(out)["u_mv"].signals["predicate"] == "move"


def test_fts_path_open_domain_unit_text():
    """Term candidacy via FTS postings ∩ typed keys (V7-06.03): the fact's
    typed columns carry no matching term, but its unit's text does."""
    conn = mk_conn(fts=True)
    add_unit(conn, "u_run")
    add_unit(conn, "u_other")
    add_state(conn, "u_run", "user/sport", "running", norm="running")
    add_state(conn, "u_other", "user/diet", "vegan", norm="vegan")
    add_fts(conn, 100, "u_run", "long trail runs every weekend")
    add_fts(conn, 101, "u_other", "unrelated chatter")

    qv = mk_qv(IntentClass.OPEN_DOMAIN, terms=("trail",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    ids = by_id(out)
    assert set(ids) == {"u_run"}
    assert ids["u_run"].signals["fts"] == 1
    assert out.stats["fts"] == "ok"


def test_no_match_keys_is_honest_ok():
    conn = mk_conn()
    seed_city_pair(conn)
    qv = mk_qv(IntentClass.CURRENT_VALUE)  # no canons, no terms
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.OK
    assert out.candidates == []
    assert out.stats["match_keys"] == "none"


# ---------------------------------------------------------------------------
# intent-conditioned ordering (brief item 3)
# ---------------------------------------------------------------------------


def test_current_value_ranks_current_above_disputed_and_historical():
    conn = mk_conn()
    seed_city_pair(conn)
    qv = mk_qv(IntentClass.CURRENT_VALUE, canons=("alice",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    order = [c.unit_id for c in out.candidates]
    assert order[0] == "u_cur"
    assert order.index("u_cur") < order.index("u_disp") < order.index("u_hist")
    ids = by_id(out)
    assert ids["u_cur"].signals["status"] == "current"
    assert ids["u_cur"].signals["lifecycle"] == "current"
    assert ids["u_hist"].signals["lifecycle"] == "historical"
    assert ids["u_disp"].signals["lifecycle"] == "disputed"


def test_history_ranks_by_valid_from_desc_all_statuses():
    conn = mk_conn()
    seed_city_pair(conn)
    # historical (2019) < disputed (2020) < current (2023) anchors
    qv = mk_qv(IntentClass.HISTORY_OF, canons=("alice",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    order = [c.unit_id for c in out.candidates]
    assert order == ["u_cur", "u_disp", "u_hist"]


def test_history_includes_historical_rows_not_just_current():
    conn = mk_conn()
    add_unit(conn, "u_old")
    add_unit(conn, "u_new")
    add_state(conn, "u_old", "user/job", "Chef", status="historical",
              vfrom=us(2018, 1, 1))
    add_state(conn, "u_new", "user/job", "Engineer", status="current",
              vfrom=us(2022, 1, 1))
    qv = mk_qv(IntentClass.HISTORY_OF, terms=("job",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    assert [c.unit_id for c in out.candidates] == ["u_new", "u_old"]


def test_preference_intent_hits_preferences_first():
    conn = mk_conn()
    add_unit(conn, "u_pref")
    add_unit(conn, "u_state")
    add_pref(conn, "u_pref", "alice", "sushi", polarity="positive",
             strength="love_hate")
    add_state(conn, "u_state", "alice/favorite_food", "sushi",
              status="current")
    qv = mk_qv(IntentClass.PREFERENCE, canons=("alice",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    assert out.candidates[0].unit_id == "u_pref"
    assert out.candidates[0].signals["fact_kind"] == "preference"
    assert out.candidates[0].signals["polarity"] == "positive"
    assert out.candidates[0].signals["strength"] == "love_hate"


def test_subject_match_bonus_orders():
    conn = mk_conn()
    add_unit(conn, "u_subj")
    add_unit(conn, "u_term")
    # same kind/prior; u_subj matches canon, u_term only a term
    add_pref(conn, "u_subj", "alice", "x")
    add_pref(conn, "u_term", "bob", "sushi")
    qv = mk_qv(IntentClass.PREFERENCE, canons=("alice",), terms=("sushi",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    ids = by_id(out)
    assert ids["u_subj"].raw_score > ids["u_term"].raw_score


# ---------------------------------------------------------------------------
# t2 facts: verified ordering, closure, unit selection
# ---------------------------------------------------------------------------


def test_t2_verified_ranks_above_unverified():
    conn = mk_conn()
    add_unit(conn, "u_v")
    add_unit(conn, "u_u")
    add_t2(conn, "f_v", ["u_v"], "Alice lives in Portland",
           subject="alice", predicate="live_in", obj="Portland", verified=1)
    add_t2(conn, "f_u", ["u_u"], "Alice lives in Seattle",
           subject="alice", predicate="live_in", obj="Seattle", verified=0)
    qv = mk_qv(IntentClass.CURRENT_VALUE, canons=("alice",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    assert out.candidates[0].unit_id == "u_v"
    ids = by_id(out)
    assert ids["u_v"].signals["verified"] == 1
    assert ids["u_v"].signals["t2_fact"] == "f_v"
    # unverified still emits — honest signal for the verdict, never hidden
    assert ids["u_u"].signals["verified"] == 0
    assert ids["u_u"].signals["fact_kind"] == "t2"


def test_t2_candidate_uses_first_unit_id():
    conn = mk_conn()
    add_unit(conn, "u_a")
    add_unit(conn, "u_b")
    add_t2(conn, "f1", ["u_a", "u_b"], "Alice runs", subject="alice")
    qv = mk_qv(IntentClass.OPEN_DOMAIN, canons=("alice",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    assert out.candidates[0].unit_id == "u_a"
    assert out.candidates[0].signals["t2_fact"] == "f1"
    assert out.candidates[0].signals["derived"] == 1


def test_t2_statement_term_match():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_unit(conn, "u2")
    add_t2(conn, "f1", ["u1"], "Alice adopted a border collie",
           subject="alice", verified=0)
    add_t2(conn, "f2", ["u2"], "Bob likes tea", subject="bob")
    qv = mk_qv(IntentClass.OPEN_DOMAIN, terms=("collie",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    assert set(by_id(out)) == {"u1"}
    assert "collie" in by_id(out)["u1"].signals["matched_terms"]


def test_t2_closure_withheld_when_support_held():
    """V7-13.13: a t2 fact retires when ANY pinned support is ineligible."""
    conn = mk_conn()
    add_unit(conn, "u_a")
    add_unit(conn, "u_held")
    add_t2(conn, "f1", ["u_a", "u_held"], "Alice runs", subject="alice")
    ctx = mk_ctx(conn, eligible=lambda row: row["unit_id"] != "u_held")
    qv = mk_qv(IntentClass.OPEN_DOMAIN, canons=("alice",))
    out = tlane.lane_typed(ctx, qv, sl())
    assert by_id(out) == {}
    assert out.stats["closure_withheld"] == 1


def test_t2_orphaned_when_support_unit_missing():
    conn = mk_conn()
    add_unit(conn, "u_a")
    add_t2(conn, "f1", ["u_a", "u_gone"], "Alice runs", subject="alice")
    qv = mk_qv(IntentClass.OPEN_DOMAIN, canons=("alice",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    assert by_id(out) == {}
    assert out.stats["orphaned_facts"] == 1


def test_t2_empty_unit_ids_withheld():
    conn = mk_conn()
    add_t2(conn, "f1", [], "orphan statement", subject="alice")
    qv = mk_qv(IntentClass.OPEN_DOMAIN, canons=("alice",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    assert by_id(out) == {}
    assert out.stats["closure_withheld"] == 1


# ---------------------------------------------------------------------------
# signals payload for pack rendering (V7-12.10)
# ---------------------------------------------------------------------------


def test_signals_carry_structured_payload():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_state(conn, "u1", "alice/home_city", "Portland", status="current",
              vfrom=us(2023, 6, 1),
              pins={"spans": [{"start": 4, "end": 12}]})
    qv = mk_qv(IntentClass.CURRENT_VALUE, canons=("alice",))
    c = by_id(tlane.lane_typed(mk_ctx(conn), qv, sl()))["u1"]
    s = c.signals
    assert s["fact_kind"] == "state"
    assert s["state_key"] == "alice/home_city"
    assert s["value_text"] == "Portland"
    assert s["state_value"] == "Portland"
    assert s["status"] == "current"
    assert s["valid_from_us"] == us(2023, 6, 1)
    assert s["pins"] == {"spans": [{"start": 4, "end": 12}]}
    assert s["subject_canon"] == "alice"
    # candidate carries unit join fields
    assert c.source_id == "src" and c.revision == 1


def test_pins_parse_failure_is_empty_dict_not_crash():
    conn = mk_conn()
    add_unit(conn, "u1")
    conn.execute(
        "INSERT INTO state_facts(scope_id, state_key, unit_id, generation,"
        " value_text, value_norm, status, pins_json)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (SCOPE, "alice/home_city", "u1", GEN, "Portland", "portland",
         "current", "not-json{"),
    )
    qv = mk_qv(IntentClass.CURRENT_VALUE, canons=("alice",))
    c = by_id(tlane.lane_typed(mk_ctx(conn), qv, sl()))["u1"]
    assert c.signals["pins"] == {}


def test_unit_aggregates_multiple_facts():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_state(conn, "u1", "alice/home_city", "Portland")
    add_pref(conn, "u1", "alice", "sushi")
    add_event(conn, "e1", "u1", subject="alice", predicate="move")
    qv = mk_qv(IntentClass.OPEN_DOMAIN, canons=("alice",))
    c = by_id(tlane.lane_typed(mk_ctx(conn), qv, sl()))["u1"]
    assert c.signals["n_facts"] == 3
    assert c.signals["state_keys"] == ["alice/home_city"]
    assert c.signals["event_ids"] == ["e1"]


# ---------------------------------------------------------------------------
# eligibility / fences / honesty
# ---------------------------------------------------------------------------


def test_eligibility_drops_held_unit():
    conn = mk_conn()
    add_unit(conn, "u_ok")
    add_unit(conn, "u_held")
    add_state(conn, "u_ok", "alice/home_city", "Portland")
    add_state(conn, "u_held", "alice/home_city", "Seattle")
    ctx = mk_ctx(conn, eligible=lambda row: row["unit_id"] != "u_held")
    qv = mk_qv(IntentClass.CURRENT_VALUE, canons=("alice",))
    out = tlane.lane_typed(ctx, qv, sl())
    ids = by_id(out)
    assert "u_ok" in ids and "u_held" not in ids
    assert out.eligible == 1
    assert out.stats["dropped_ineligible"] == 1


def test_eligibility_as_set_object():
    conn = mk_conn()
    add_unit(conn, "u_a")
    add_unit(conn, "u_b")
    add_pref(conn, "u_a", "alice", "sushi")
    add_pref(conn, "u_b", "alice", "ramen")
    ctx = mk_ctx(conn, eligible={"u_a"})
    qv = mk_qv(IntentClass.PREFERENCE, canons=("alice",))
    out = tlane.lane_typed(ctx, qv, sl())
    assert set(by_id(out)) == {"u_a"}


def test_generation_fence():
    """V7-30.02: ``generation <= pinned`` — gen-1 state rows stay visible
    at the gen-2 pin (distinct unit_ids, not superseded); rows above the
    pin stay fenced out."""
    conn = mk_conn()
    add_unit(conn, "u_g1", gen=1)
    add_unit(conn, "u_g2", gen=2)
    add_state(conn, "u_g1", "alice/home_city", "Portland", gen=1)
    add_state(conn, "u_g2", "alice/home_city", "Portland", gen=2)
    qv = mk_qv(IntentClass.CURRENT_VALUE, canons=("alice",))
    out1 = tlane.lane_typed(mk_ctx(conn, gen=1), qv, sl())
    assert set(by_id(out1)) == {"u_g1"}  # u_g2's fact is above the pin
    out2 = tlane.lane_typed(mk_ctx(conn, gen=2), qv, sl())
    assert set(by_id(out2)) == {"u_g1", "u_g2"}


def test_scope_fence():
    conn = mk_conn()
    add_unit(conn, "u_s1", scope="s1")
    add_unit(conn, "u_s2", scope="s2")
    add_state(conn, "u_s1", "alice/home_city", "Portland", scope="s1")
    add_state(conn, "u_s2", "alice/home_city", "Portland", scope="s2")
    qv = mk_qv(IntentClass.CURRENT_VALUE, canons=("alice",))
    out = tlane.lane_typed(mk_ctx(conn, scope="s1"), qv, sl())
    assert set(by_id(out)) == {"u_s1"}


def test_missing_typed_tables_unavailable():
    conn = sqlite3.connect(":memory:")
    conn.executescript(UNITS_DDL)
    qv = mk_qv(IntentClass.CURRENT_VALUE, canons=("alice",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "no_typed_tables"


def test_partial_table_presence_is_honest_partial():
    conn = sqlite3.connect(":memory:")
    conn.executescript(UNITS_DDL)
    conn.execute(
        "CREATE TABLE preferences (scope_id TEXT, subject_canon TEXT,"
        " unit_id TEXT, generation INTEGER, object_text TEXT,"
        " polarity TEXT, strength TEXT, occurred_start_us INTEGER,"
        " pins_json TEXT)"
    )
    add_unit(conn, "u1")
    add_pref(conn, "u1", "alice", "sushi")
    qv = mk_qv(IntentClass.PREFERENCE, canons=("alice",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.PARTIAL
    assert out.reason == "typed_table_missing"
    assert "u1" in by_id(out)
    assert set(out.stats["missing_tables"]) == {
        "state_facts", "events_v7", "t2_facts"
    }


def test_empty_but_present_tables_ok_zero():
    conn = mk_conn()  # all four tables, zero rows
    add_unit(conn, "u1")
    qv = mk_qv(IntentClass.CURRENT_VALUE, canons=("alice",), terms=("job",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.OK
    assert out.candidates == []
    assert out.examined == 0


def test_orphaned_fact_unit_not_in_units():
    conn = mk_conn()
    # fact points at a unit that does not exist in-fence
    add_state(conn, "u_ghost", "alice/home_city", "Portland")
    qv = mk_qv(IntentClass.CURRENT_VALUE, canons=("alice",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    assert by_id(out) == {}
    assert out.stats["orphaned_facts"] == 1
    assert out.eligible == 0


def test_units_table_missing_unavailable():
    conn = sqlite3.connect(":memory:")
    conn.executescript(TYPED_DDL)
    qv = mk_qv(IntentClass.CURRENT_VALUE, canons=("alice",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "units_table_missing"


def test_unavailable_without_read_snapshot():
    ctx = mk_ctx(mk_conn())
    ctx.store = object()
    qv = mk_qv(IntentClass.CURRENT_VALUE, canons=("alice",))
    out = tlane.lane_typed(ctx, qv, sl())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "no_read_snapshot"


def test_generation_unpinned():
    conn = mk_conn()
    ctx = mk_ctx(conn)
    ctx.generation = None
    qv = mk_qv(IntentClass.CURRENT_VALUE, canons=("alice",))
    out = tlane.lane_typed(ctx, qv, sl())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "generation_unpinned"


def test_eligibility_handle_missing():
    conn = mk_conn()
    ctx = mk_ctx(conn, eligible=None)
    ctx.eligible = None
    qv = mk_qv(IntentClass.CURRENT_VALUE, canons=("alice",))
    out = tlane.lane_typed(ctx, qv, sl())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "eligibility_handle_missing"


def test_store_wrapper_with_conn_attribute():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_state(conn, "u1", "alice/home_city", "Portland")
    ctx = mk_ctx(conn)
    ctx.store = SimpleNamespace(conn=conn)
    qv = mk_qv(IntentClass.CURRENT_VALUE, canons=("alice",))
    out = tlane.lane_typed(ctx, qv, sl())
    assert "u1" in by_id(out)


def test_deadline_partial():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_state(conn, "u1", "alice/home_city", "Portland")
    qv = mk_qv(IntentClass.CURRENT_VALUE, canons=("alice",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl(ms=0))
    assert out.status == LaneStatus.PARTIAL
    assert out.reason == "deadline"
    assert out.examined == 0


def test_cap_truncates_and_reports():
    conn = mk_conn()
    for i in range(5):
        add_unit(conn, f"u{i}")
        add_pref(conn, f"u{i}", "alice", f"thing{i}")
    qv = mk_qv(IntentClass.PREFERENCE, canons=("alice",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl(cap=2))
    assert out.status == LaneStatus.OK
    assert len(out.candidates) == 2
    assert out.stats["overflow"] == 3
    assert out.eligible == 5


def test_examined_and_eligible_counts_honest():
    conn = mk_conn()
    add_unit(conn, "u_a")
    add_unit(conn, "u_b")
    add_pref(conn, "u_a", "alice", "sushi")
    add_pref(conn, "u_b", "bob", "ramen")
    qv = mk_qv(IntentClass.PREFERENCE, canons=("alice",))
    out = tlane.lane_typed(mk_ctx(conn), qv, sl())
    # subject-prefiltered fetch examines only alice's row; the unit join
    # examines the one resolved unit row.
    assert out.examined == 2
    assert out.eligible == 1
    assert out.stats["facts_matched"] == 1
    assert out.stats["facts_admitted"] == 1


def test_deterministic_repeat_run():
    conn = mk_conn()
    for i in range(8):
        add_unit(conn, f"u{i:02d}")
        add_pref(conn, f"u{i:02d}", "alice", f"thing{i}")
        add_state(conn, f"u{i:02d}", f"alice/key_{i}", f"v{i}")
    qv = mk_qv(IntentClass.PREFERENCE, canons=("alice",))
    ctx = mk_ctx(conn)
    first = [
        (c.unit_id, c.rank, c.raw_score, sorted(c.signals.items()))
        for c in tlane.lane_typed(ctx, qv, sl(cap=6)).candidates
    ]
    second = [
        (c.unit_id, c.rank, c.raw_score, sorted(c.signals.items()))
        for c in tlane.lane_typed(ctx, qv, sl(cap=6)).candidates
    ]
    assert first == second
    assert [c.rank for c in tlane.lane_typed(ctx, qv, sl(cap=6)).candidates] == [
        1, 2, 3, 4, 5, 6,
    ]


def test_lanev7_wrapper_class():
    conn = mk_conn()
    add_unit(conn, "u1")
    add_state(conn, "u1", "alice/home_city", "Portland")
    qv = mk_qv(IntentClass.CURRENT_VALUE, canons=("alice",))
    lane = tlane.TypedLane()
    out = lane.run(mk_ctx(conn), qv, sl())
    assert out.lane == "typed" and "u1" in by_id(out)


def test_run_one_integration_registers():
    """Through the orchestrator seam: the import-time registration makes
    ``run_one`` dispatch the real lane, not ``not_registered``."""
    conn = mk_conn()
    add_unit(conn, "u1")
    add_state(conn, "u1", "alice/home_city", "Portland")
    qv = mk_qv(IntentClass.CURRENT_VALUE, canons=("alice",))
    out = lanes_base.run_one(mk_ctx(conn), qv, LaneName.TYPED, sl())
    assert out.status == LaneStatus.OK
    assert "u1" in by_id(out)
