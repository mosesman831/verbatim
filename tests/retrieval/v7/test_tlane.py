"""Lane tests for ``verbatim/retrieval/v7/temporal.py`` (w-tlane, wave A).

Mirror-DDL approach per the wave brief: §30 ``units`` / ``events`` /
``entity_mentions`` (and a real FTS5 ``unit_fts`` where lexical relevance is
exercised) are created inline — ``schema_v7`` is a concurrent worker's file.
``QueryViewV7`` is hand-built with ``intent.window`` already resolved; the
real resolver (``temporal_v2.resolve_query_window``, w-temporal) is never
called here. These tests unit-test the LANE.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v7 import (
    BudgetClass,
    IntentClass,
    IntentResult,
    IntervalUs,
    LaneContextV7,
    LaneName,
    LaneSlice,
    LaneStatus,
    NormAnalysis,
    NormTerm,
    OccurredPrecision,
    QueryViewV7,
    RetrievalPolicyV7,
)
from verbatim.retrieval.v7 import temporal as tlane

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
CREATE TABLE entity_mentions (
  scope_id TEXT NOT NULL,
  canon TEXT NOT NULL,
  unit_id TEXT NOT NULL,
  generation INTEGER NOT NULL,
  surface TEXT,
  byte_start INTEGER,
  byte_end INTEGER,
  role TEXT
);
CREATE TABLE unit_time_mentions (
  unit_id TEXT NOT NULL,
  generation INTEGER NOT NULL,
  scope_id TEXT NOT NULL,
  ord INTEGER NOT NULL,
  start_us INTEGER NOT NULL,
  end_us INTEGER NOT NULL,
  precision TEXT NOT NULL,
  span_start INTEGER NOT NULL,
  span_end INTEGER NOT NULL,
  anchor_us INTEGER NOT NULL,
  resolver_version TEXT NOT NULL,
  PRIMARY KEY (unit_id, generation, ord)
);
"""

FTS_DDL = """
CREATE VIRTUAL TABLE unit_fts USING fts5(
  text, speaker, entities, session, "when",
  content='units', content_rowid='rowid'
);
"""

SCOPE = "s1"
GEN = 1


def us(y: int, m: int = 1, d: int = 1, h: int = 0, mi: int = 0) -> int:
    """Deterministic µs timestamp."""
    return int(datetime(y, m, d, h, mi, tzinfo=timezone.utc).timestamp() * 1_000_000)


def mk_conn(fts: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(MIRROR_DDL)
    if fts:
        conn.executescript(FTS_DDL)
    return conn


def add_unit(
    conn,
    unit_id,
    *,
    occurred=(None, None),
    precision="day",
    occurred_source="explicit",
    recorded=None,
    scope=SCOPE,
    gen=GEN,
    source="src",
    rev=1,
    rowid=None,
    speaker=None,
):
    cols = (
        "unit_id, source_id, revision, scope_id, kind, recorded_at_us,"
        " occurred_start_us, occurred_end_us, occurred_precision,"
        " occurred_source, generation"
    )
    vals = [
        unit_id,
        source,
        rev,
        scope,
        "turn",
        recorded,
        occurred[0],
        occurred[1],
        precision,
        occurred_source,
        gen,
    ]
    if rowid is not None:
        cols = "rowid, " + cols
        vals = [rowid, *vals]
    conn.execute(
        f"INSERT INTO units({cols}) VALUES ({','.join('?' * len(vals))})", vals
    )


def add_fts(conn, rowid, text, speaker="", entities="", session="", when=""):
    conn.execute(
        'INSERT INTO unit_fts(rowid, text, speaker, entities, session, "when")'
        " VALUES (?,?,?,?,?,?)",
        (rowid, text, speaker, entities, session, when),
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
    precision="day",
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
            precision,
            "{}",
            "event/v1:test",
            gen,
        ),
    )


def add_mention(conn, unit_id, canon, scope=SCOPE, gen=GEN):
    conn.execute(
        "INSERT INTO entity_mentions(scope_id, canon, unit_id, generation,"
        " surface, byte_start, byte_end, role) VALUES (?,?,?,?,?,0,1,'mention')",
        (scope, canon, unit_id, gen, canon),
    )


def add_time_mention(
    conn,
    unit_id,
    *,
    start,
    end,
    precision="day",
    ord_=0,
    anchor=None,
    scope=SCOPE,
    gen=GEN,
):
    """One ``unit_time_mentions`` row (V8-09.04) — the write path's shape."""
    conn.execute(
        "INSERT INTO unit_time_mentions(unit_id, generation, scope_id, ord,"
        " start_us, end_us, precision, span_start, span_end, anchor_us,"
        " resolver_version)"
        " VALUES (?,?,?,?,?,?,?,0,10,?,'resolver/v8:test')",
        (unit_id, gen, scope, ord_, start, end, precision, anchor or start),
    )


def mk_qv(
    primary=IntentClass.TEMPORAL_RANGE,
    *,
    classes=None,
    window=None,
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
        primary=primary, classes=classes or (primary,), window=window
    )
    return QueryViewV7(
        query=q,
        norm=norm,
        intent=intent,
        entity_canons=tuple(canons),
        speaker_canon=speaker,
        query_time_us=us(2024, 1, 1),
    )


def mk_ctx(conn, *, eligible=None, gen=GEN, scope=SCOPE, manifest=None):
    if eligible is None:
        eligible = lambda row: True  # noqa: E731 - allow-all fixture
    policy = RetrievalPolicyV7(
        policy_id="retrieval_policy/v7",
        profile="test",
        lanes=(LaneName.TIME,),
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
        manifest=dict(manifest or {}),
    )


def sl(cap=50, ms=10_000):
    return LaneSlice(deadline_ms=ms, cap=cap)


def by_id(out):
    return {c.unit_id: c for c in out.candidates}


# ---------------------------------------------------------------------------
# window overlap correctness
# ---------------------------------------------------------------------------


def test_window_overlap_partial_containment_and_instant():
    conn = mk_conn()
    ws, we = us(2023, 3, 1), us(2023, 4, 1)  # March 2023
    add_unit(conn, "u_partial", occurred=(us(2023, 2, 20), us(2023, 3, 10)))
    add_unit(conn, "u_instant", occurred=(us(2023, 3, 15), us(2023, 3, 15)),
             precision="instant")
    add_unit(conn, "u_contains", occurred=(us(2023, 2, 1), us(2023, 5, 1)))
    add_unit(conn, "u_open_start", occurred=(None, us(2023, 3, 5)))
    add_unit(conn, "u_outside", occurred=(us(2023, 4, 5), us(2023, 4, 20)))
    # occurred fully unknown: never an occurred-axis match (recorded outside
    # the window too, so it has no honest path into the candidate set).
    add_unit(conn, "u_unknown", occurred=(None, None), precision="unknown",
             occurred_source="unknown", recorded=us(2022, 6, 1))
    # unknown precision WITH real bounds: still an honest overlap, flagged.
    add_unit(conn, "u_unkprec", occurred=(us(2023, 3, 3), us(2023, 3, 4)),
             precision="unknown", occurred_source="unknown")

    qv = mk_qv(IntentClass.TEMPORAL_RANGE, window=IntervalUs(ws, we))
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())

    assert out.status == LaneStatus.OK
    ids = by_id(out)
    assert set(ids) == {
        "u_partial",
        "u_instant",
        "u_contains",
        "u_open_start",
        "u_unkprec",
    }
    assert ids["u_partial"].signals["overlap"] == "partial"
    # an instant inside a wider window counts (V7-09.06)
    assert ids["u_instant"].signals["overlap"] == "point"
    assert ids["u_contains"].signals["overlap"] == "contains"
    assert ids["u_open_start"].signals["overlap"] == "partial"
    assert ids["u_unkprec"].signals["occurred_precision"] == "unknown"
    assert "u_outside" not in ids and "u_unknown" not in ids
    assert out.eligible == 5 and out.examined == 5


def test_half_open_window_overlap():
    conn = mk_conn()
    ws = us(2023, 6, 1)  # "since June 2023" — open-ended window
    add_unit(conn, "u_before", occurred=(us(2023, 5, 1), us(2023, 5, 10)))
    add_unit(conn, "u_after", occurred=(us(2023, 7, 1), us(2023, 7, 10)))
    add_unit(conn, "u_straddle", occurred=(us(2023, 5, 20), us(2023, 6, 5)))

    qv = mk_qv(IntentClass.TEMPORAL_RANGE, window=IntervalUs(ws, None))
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())
    ids = by_id(out)
    assert "u_after" in ids and "u_straddle" in ids
    assert "u_before" not in ids
    # no centre on a half-open window -> t_prox absent, never invented
    assert all("t_prox" not in c.signals for c in out.candidates)


def test_recorded_axis_secondary():
    conn = mk_conn()
    ws, we = us(2023, 3, 1), us(2023, 4, 1)
    add_unit(conn, "u_occ", occurred=(us(2023, 3, 10), us(2023, 3, 11)),
             recorded=us(2022, 1, 1))
    # occurred outside the window but recorded inside it -> secondary match
    add_unit(conn, "u_rec", occurred=(us(2020, 1, 1), us(2020, 1, 2)),
             recorded=us(2023, 3, 15))
    add_unit(conn, "u_neither", occurred=(us(2020, 5, 1), us(2020, 5, 2)),
             recorded=us(2020, 5, 3))

    qv = mk_qv(IntentClass.TEMPORAL_RANGE, window=IntervalUs(ws, we))
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())
    ids = by_id(out)
    assert "u_neither" not in ids
    assert ids["u_rec"].signals["axis"] == "recorded"
    assert ids["u_rec"].signals["overlap"] == "recorded_only"
    assert ids["u_rec"].signals["tier"] > ids["u_occ"].signals["tier"]
    # secondary axis ranks after the occurred-overlap unit
    assert ids["u_occ"].rank < ids["u_rec"].rank


# ---------------------------------------------------------------------------
# bucket spread (V7-09.06)
# ---------------------------------------------------------------------------


def test_bucket_spread_representative_over_year():
    conn = mk_conn()
    ws, we = us(2023, 1, 1), us(2024, 1, 1)
    # a dense January cluster plus three lone units later in the year
    for i in range(8):
        add_unit(conn, f"u_jan_{i}",
                 occurred=(us(2023, 1, 2 + i), us(2023, 1, 2 + i)))
    for i, m in enumerate((4, 7, 10)):
        add_unit(conn, f"u_m{m}", occurred=(us(2023, m, 15), us(2023, m, 15)))

    qv = mk_qv(IntentClass.TEMPORAL_RANGE, window=IntervalUs(ws, we))
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl(cap=4))

    assert out.status == LaneStatus.OK
    assert len(out.candidates) == 4
    assert out.stats["bucket_spread"] is True
    assert out.stats["buckets"] >= tlane.MIN_SPREAD_BUCKETS
    # the selection is representative: the three lone months all survive the
    # cap instead of four January picks
    ids = by_id(out)
    assert {"u_m4", "u_m7", "u_m10"} <= set(ids)
    jan = [u for u in ids if u.startswith("u_jan_")]
    assert len(jan) == 1
    buckets = {c.signals["bucket"] for c in out.candidates}
    assert len(buckets) == 4


def test_no_spread_for_submonth_window():
    conn = mk_conn()
    ws, we = us(2023, 3, 1), us(2023, 3, 15)
    for i in range(6):
        add_unit(conn, f"u{i}", occurred=(us(2023, 3, 2 + i), us(2023, 3, 2 + i)))
    qv = mk_qv(IntentClass.TEMPORAL_RANGE, window=IntervalUs(ws, we))
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl(cap=3))
    assert out.stats["bucket_spread"] is False
    assert out.stats["buckets"] == 1
    assert len(out.candidates) == 3


# ---------------------------------------------------------------------------
# events index first (V7-09.09)
# ---------------------------------------------------------------------------


def test_events_first_ordering_for_temporal_point():
    conn = mk_conn()
    ws, we = us(2023, 1, 1), us(2024, 1, 1)
    # event-backed unit (alice moved in June)
    add_unit(conn, "u_ev", occurred=(us(2023, 6, 3), us(2023, 6, 3)))
    add_event(conn, "e1", "u_ev", subject="alice", predicate="move",
              obj="to Portland", occurred=(us(2023, 6, 3), us(2023, 6, 3)))
    # a plain in-window unit that ALSO mentions alice (entity-relevant but
    # no matching event)
    add_unit(conn, "u_plain", occurred=(us(2023, 6, 10), us(2023, 6, 10)))
    add_mention(conn, "u_plain", "alice")

    qv = mk_qv(
        IntentClass.TEMPORAL_POINT,
        window=IntervalUs(ws, we),
        terms=("when", "did", "alice", "move"),
        canons=("alice",),
    )
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())

    assert out.status == LaneStatus.OK
    assert out.candidates[0].unit_id == "u_ev"
    ev = by_id(out)["u_ev"]
    assert ev.signals["event_match"] is True
    assert ev.signals["event_pred"] == 1
    assert ev.signals["subject_match"] == 1
    assert ev.signals["event_ids"] == ["e1"]
    assert out.stats["events_matched"] == 1


def test_events_first_without_window():
    conn = mk_conn()
    # temporal_order needs no window: the events themselves carry intervals
    add_unit(conn, "u_a", occurred=(us(2022, 3, 1), us(2022, 3, 1)))
    add_unit(conn, "u_b", occurred=(us(2022, 5, 1), us(2022, 5, 1)))
    add_event(conn, "e_a", "u_a", subject="alice", predicate="move",
              occurred=(us(2022, 3, 1), us(2022, 3, 1)))
    add_event(conn, "e_b", "u_b", subject="alice", predicate="marry",
              occurred=(us(2022, 5, 1), us(2022, 5, 1)))
    add_unit(conn, "u_noise", occurred=(us(2022, 4, 1), us(2022, 4, 1)))

    qv = mk_qv(
        IntentClass.TEMPORAL_ORDER,
        terms=("which", "came", "first", "alice", "move", "marry"),
        canons=("alice",),
    )
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())
    ids = by_id(out)
    assert {"u_a", "u_b"} <= set(ids)
    assert "u_noise" not in ids
    assert all(c.signals["event_match"] for c in out.candidates)
    assert out.stats["mode"] == "events"


def test_events_respect_window_when_present():
    conn = mk_conn()
    # alice moved twice; the window only covers the 2023 one
    add_unit(conn, "u_old", occurred=(us(2019, 1, 1), us(2019, 1, 1)))
    add_event(conn, "e_old", "u_old", subject="alice", predicate="move",
              occurred=(us(2019, 1, 1), us(2019, 1, 1)))
    add_unit(conn, "u_new", occurred=(us(2023, 6, 3), us(2023, 6, 3)))
    add_event(conn, "e_new", "u_new", subject="alice", predicate="move",
              occurred=(us(2023, 6, 3), us(2023, 6, 3)))

    qv = mk_qv(
        IntentClass.TEMPORAL_POINT,
        window=IntervalUs(us(2023, 1, 1), us(2024, 1, 1)),
        terms=("alice", "move"),
        canons=("alice",),
    )
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())
    ids = by_id(out)
    assert "u_new" in ids and "u_old" not in ids


# ---------------------------------------------------------------------------
# t_prox signal math (V7-09.07 / §32.5)
# ---------------------------------------------------------------------------


def test_t_prox_signal_math():
    conn = mk_conn()
    ws, we = us(2023, 1, 1), us(2023, 1, 11)  # 10-day window
    centre = (ws + we) // 2
    half = (we - ws) / 2.0
    add_unit(conn, "u_c", occurred=(centre, centre))
    quarter = int(ws + 0.75 * (we - ws))
    add_unit(conn, "u_q", occurred=(quarter, quarter))
    add_unit(conn, "u_edge", occurred=(ws, ws))
    add_unit(conn, "u_span", occurred=(ws, we))  # midpoint == centre

    qv = mk_qv(IntentClass.TEMPORAL_RANGE, window=IntervalUs(ws, we))
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())
    ids = by_id(out)
    assert ids["u_c"].signals["t_prox"] == pytest.approx(1.0)
    assert ids["u_span"].signals["t_prox"] == pytest.approx(1.0)
    assert ids["u_q"].signals["t_prox"] == pytest.approx(0.5, abs=0.01)
    assert ids["u_edge"].signals["t_prox"] == pytest.approx(0.0)
    # clamped to [0, 1]
    for c in out.candidates:
        assert 0.0 <= c.signals["t_prox"] <= 1.0
    assert half > 0  # sanity on the fixture itself


# ---------------------------------------------------------------------------
# honesty: skip / deadline / unavailable / eligibility
# ---------------------------------------------------------------------------


def test_skipped_without_window_for_nontemporal_intent():
    conn = mk_conn()
    add_unit(conn, "u1", occurred=(us(2023, 1, 1), us(2023, 1, 1)))
    qv = mk_qv(IntentClass.LOOKUP, terms=("hello",))
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.SKIPPED
    assert out.reason == "no_window"
    assert out.candidates == [] and out.examined == 0 and out.eligible == 0


def test_window_runs_even_for_nontemporal_intent():
    conn = mk_conn()
    ws, we = us(2023, 3, 1), us(2023, 4, 1)
    add_unit(conn, "u1", occurred=(us(2023, 3, 10), us(2023, 3, 11)))
    qv = mk_qv(IntentClass.LOOKUP, window=IntervalUs(ws, we))
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.OK
    assert "u1" in by_id(out)


def test_deadline_partial():
    conn = mk_conn()
    ws, we = us(2023, 1, 1), us(2024, 1, 1)
    add_unit(conn, "u1", occurred=(us(2023, 6, 1), us(2023, 6, 1)))
    qv = mk_qv(IntentClass.TEMPORAL_RANGE, window=IntervalUs(ws, we))
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl(ms=0))
    assert out.status == LaneStatus.PARTIAL
    assert out.reason == "deadline"
    assert out.examined == 0  # honest: nothing was scanned


def test_unavailable_without_read_snapshot():
    ctx = mk_ctx(mk_conn())
    ctx.store = object()  # no pinned connection anywhere
    qv = mk_qv(IntentClass.TEMPORAL_RANGE,
               window=IntervalUs(us(2023, 1, 1), us(2024, 1, 1)))
    out = tlane.lane_temporal(ctx, qv, sl())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "no_read_snapshot"


def test_eligibility_inside_production_held_unit_never_returned():
    conn = mk_conn()
    ws, we = us(2023, 1, 1), us(2024, 1, 1)
    add_unit(conn, "u_ok", occurred=(us(2023, 2, 1), us(2023, 2, 1)))
    add_unit(conn, "u_held", occurred=(us(2023, 3, 1), us(2023, 3, 1)))
    held = {"u_held"}
    ctx = mk_ctx(conn, eligible=lambda row: row["unit_id"] not in held)
    qv = mk_qv(IntentClass.TEMPORAL_RANGE, window=IntervalUs(ws, we))
    out = tlane.lane_temporal(ctx, qv, sl())
    ids = by_id(out)
    assert "u_ok" in ids and "u_held" not in ids
    assert out.examined == 2 and out.eligible == 1


def test_eligibility_as_set_object():
    conn = mk_conn()
    ws, we = us(2023, 1, 1), us(2024, 1, 1)
    add_unit(conn, "u_a", occurred=(us(2023, 2, 1), us(2023, 2, 1)))
    add_unit(conn, "u_b", occurred=(us(2023, 3, 1), us(2023, 3, 1)))
    ctx = mk_ctx(conn, eligible={"u_a"})  # set-like eligible handle
    qv = mk_qv(IntentClass.TEMPORAL_RANGE, window=IntervalUs(ws, we))
    out = tlane.lane_temporal(ctx, qv, sl())
    assert set(by_id(out)) == {"u_a"}


def test_store_wrapper_with_conn_attribute():
    conn = mk_conn()
    ws, we = us(2023, 1, 1), us(2024, 1, 1)
    add_unit(conn, "u1", occurred=(us(2023, 6, 1), us(2023, 6, 1)))
    ctx = mk_ctx(conn)
    ctx.store = SimpleNamespace(conn=conn)  # snapshot-wrapper store
    qv = mk_qv(IntentClass.TEMPORAL_RANGE, window=IntervalUs(ws, we))
    out = tlane.lane_temporal(ctx, qv, sl())
    assert out.status == LaneStatus.OK and "u1" in by_id(out)


# ---------------------------------------------------------------------------
# fences / degradation
# ---------------------------------------------------------------------------


def test_generation_fence():
    """V7-30.02: ``generation <= pinned`` — gen-1 rows stay visible at the
    gen-2 pin (distinct unit_ids, not superseded); rows above the pin
    stay fenced out."""
    conn = mk_conn()
    ws, we = us(2023, 1, 1), us(2024, 1, 1)
    add_unit(conn, "u_gen1", occurred=(us(2023, 2, 1), us(2023, 2, 1)), gen=1)
    add_unit(conn, "u_gen2", occurred=(us(2023, 3, 1), us(2023, 3, 1)), gen=2)
    qv = mk_qv(IntentClass.TEMPORAL_RANGE, window=IntervalUs(ws, we))
    out = tlane.lane_temporal(mk_ctx(conn, gen=1), qv, sl())
    assert set(by_id(out)) == {"u_gen1"}  # u_gen2 is above the pin
    out2 = tlane.lane_temporal(mk_ctx(conn, gen=2), qv, sl())
    assert set(by_id(out2)) == {"u_gen1", "u_gen2"}


def test_events_table_missing_degrades_honestly():
    conn = sqlite3.connect(":memory:")
    # units + entity_mentions only — no events table
    conn.executescript(
        "CREATE TABLE units (unit_id TEXT PRIMARY KEY, source_id TEXT,"
        " revision INTEGER, scope_id TEXT, kind TEXT, parent_unit_id TEXT,"
        " session_id TEXT, seq INTEGER, speaker_canon TEXT, perspective TEXT,"
        " recorded_at_us INTEGER, occurred_start_us INTEGER,"
        " occurred_end_us INTEGER, occurred_precision TEXT,"
        " occurred_source TEXT, byte_start INTEGER, byte_end INTEGER,"
        " generation INTEGER);"
        "CREATE TABLE entity_mentions (scope_id TEXT, canon TEXT,"
        " unit_id TEXT, generation INTEGER, surface TEXT,"
        " byte_start INTEGER, byte_end INTEGER, role TEXT);"
    )
    ws, we = us(2023, 1, 1), us(2024, 1, 1)
    add_unit(conn, "u1", occurred=(us(2023, 6, 1), us(2023, 6, 1)))
    qv = mk_qv(IntentClass.TEMPORAL_POINT, window=IntervalUs(ws, we),
               terms=("move",), canons=("alice",))
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())
    # window path still ran; the missing event index is an honest partial
    assert out.status == LaneStatus.PARTIAL
    assert out.reason == "events_table_missing"
    assert "u1" in by_id(out)

    # events-first intent with NO window and no events table -> unavailable
    qv2 = mk_qv(IntentClass.TEMPORAL_ORDER, terms=("move",), canons=("alice",))
    out2 = tlane.lane_temporal(mk_ctx(conn), qv2, sl())
    assert out2.status == LaneStatus.UNAVAILABLE
    assert out2.reason == "events_table_missing"


def test_recorded_fallback_for_history_of():
    # V8-09.06 — the no-window fallback is relevance-restricted and
    # event-time ordered; units without a nominated term/canon are never
    # admitted, and NULL-occurred units order on the recorded fallback.
    conn = mk_conn(fts=True)
    add_unit(conn, "u_old", rowid=1, occurred=(None, None),
             precision="unknown", occurred_source="unknown",
             recorded=us(2023, 1, 10))
    add_unit(conn, "u_new", rowid=2, occurred=(None, None),
             precision="unknown", occurred_source="unknown",
             recorded=us(2023, 11, 10))
    add_unit(conn, "u_irrelevant", rowid=3, occurred=(None, None),
             precision="unknown", occurred_source="unknown",
             recorded=us(2023, 6, 10))
    add_fts(conn, 1, "the relationship started slowly")
    add_fts(conn, 2, "our relationship today")
    add_fts(conn, 3, "unrelated chatter")
    qv = mk_qv(IntentClass.HISTORY_OF, terms=("relationship",))
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.OK
    assert out.stats["mode"] == "relevant_fallback"
    ids = by_id(out)
    assert set(ids) == {"u_old", "u_new"}
    assert "u_irrelevant" not in ids
    assert all(c.signals["axis"] == "recorded" for c in out.candidates)
    assert all(c.signals["overlap"] == "fallback" for c in out.candidates)


# ---------------------------------------------------------------------------
# relevance within the window (never recency) + determinism
# ---------------------------------------------------------------------------


def test_lexical_relevance_selects_within_window_not_recency():
    conn = mk_conn(fts=True)
    ws, we = us(2023, 1, 1), us(2024, 1, 1)
    # older unit matches the query term; newer unit does not
    add_unit(conn, "u_old", rowid=1,
             occurred=(us(2023, 1, 10), us(2023, 1, 10)))
    add_unit(conn, "u_new", rowid=2,
             occurred=(us(2023, 11, 10), us(2023, 11, 10)))
    add_fts(conn, 1, "I bought a bicycle for the trail")
    add_fts(conn, 2, "unrelated chatter about lunch")

    qv = mk_qv(IntentClass.TEMPORAL_RANGE, window=IntervalUs(ws, we),
               terms=("bicycle",))
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())
    assert out.stats["lexical"] == "fts"
    # relevance beats recency: the older matching unit outranks
    assert out.candidates[0].unit_id == "u_old"
    assert by_id(out)["u_old"].raw_score > by_id(out)["u_new"].raw_score


def test_entity_mention_relevance_without_fts():
    conn = mk_conn()  # no unit_fts mirror
    ws, we = us(2023, 1, 1), us(2024, 1, 1)
    add_unit(conn, "u_alice", occurred=(us(2023, 1, 10), us(2023, 1, 10)))
    add_unit(conn, "u_other", occurred=(us(2023, 6, 10), us(2023, 6, 10)))
    add_mention(conn, "u_alice", "alice")

    qv = mk_qv(IntentClass.TEMPORAL_RANGE, window=IntervalUs(ws, we),
               terms=("alice", "trip"), canons=("alice",))
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())
    assert out.stats["lexical"] == "unavailable"  # honest degradation
    ids = by_id(out)
    assert ids["u_alice"].signals["ent_match"] == pytest.approx(1.0)
    assert ids["u_other"].signals["ent_match"] == pytest.approx(0.0)
    assert out.candidates[0].unit_id == "u_alice"


def test_deterministic_repeat_run():
    conn = mk_conn()
    ws, we = us(2023, 1, 1), us(2024, 1, 1)
    for i in range(10):
        add_unit(conn, f"u{i:02d}",
                 occurred=(us(2023, i + 1, 15), us(2023, i + 1, 15)))
    qv = mk_qv(IntentClass.TEMPORAL_RANGE, window=IntervalUs(ws, we))
    ctx = mk_ctx(conn)
    first = [(c.unit_id, c.rank, c.raw_score, sorted(c.signals.items()))
             for c in tlane.lane_temporal(ctx, qv, sl(cap=6)).candidates]
    second = [(c.unit_id, c.rank, c.raw_score, sorted(c.signals.items()))
              for c in tlane.lane_temporal(ctx, qv, sl(cap=6)).candidates]
    assert first == second
    assert [c.rank for c in tlane.lane_temporal(ctx, qv, sl(cap=6)).candidates] == [
        1, 2, 3, 4, 5, 6
    ]


# ---------------------------------------------------------------------------
# V8-09.04/09.08 — unit_time_mentions union + mention-driven scoring
# ---------------------------------------------------------------------------


def test_mention_union_matches_when_occurred_misses():
    """V8-09.04 — a unit whose *text* mentions a time inside the window
    is admitted even when its own occurred interval does not overlap."""
    conn = mk_conn()
    ws, we = us(2023, 3, 1), us(2023, 4, 1)  # March 2023
    # session-time occurred is 2020 — outside the window — but the unit
    # text resolved "last March" → a March-2023 mention.
    ms, me = us(2023, 3, 10), us(2023, 3, 11)
    add_unit(conn, "u_mention", occurred=(us(2020, 6, 1), us(2020, 6, 2)))
    add_time_mention(conn, "u_mention", start=ms, end=me, precision="day")
    add_unit(conn, "u_occ", occurred=(us(2023, 3, 15), us(2023, 3, 15)))
    add_unit(conn, "u_none", occurred=(us(2020, 1, 1), us(2020, 1, 2)))

    qv = mk_qv(IntentClass.TEMPORAL_RANGE, window=IntervalUs(ws, we))
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())

    assert out.status == LaneStatus.OK
    assert out.stats["mentions"] == "ok"
    assert "mentions" in out.stats["mode"]
    ids = by_id(out)
    assert "u_mention" in ids and "u_occ" in ids and "u_none" not in ids
    sig = ids["u_mention"].signals
    assert sig["overlap"] == "mention"
    assert sig["axis"] == "mention"
    # the exact mention interval is emitted (V8-09.08 explain contract)
    assert sig["mention_interval"] == (ms, me, "day")


def test_mention_interval_drives_scoring_not_session_date():
    """V8-09.08 — temporal proximity scores the mention interval (event
    date), not the unit's own occurred (session date)."""
    conn = mk_conn()
    ws, we = us(2023, 1, 1), us(2023, 1, 11)  # 10-day window
    centre = (ws + we) // 2
    edge = ws
    # u_a: occurred AT the window centre but the event it describes
    # (mention) sits at the window edge.
    add_unit(conn, "u_a", occurred=(centre, centre))
    add_time_mention(conn, "u_a", start=edge, end=edge, precision="day")
    # u_b: occurred at the edge, mention at the centre.
    add_unit(conn, "u_b", occurred=(edge, edge))
    add_time_mention(conn, "u_b", start=centre, end=centre, precision="day")

    qv = mk_qv(IntentClass.TEMPORAL_RANGE, window=IntervalUs(ws, we))
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())
    ids = by_id(out)
    # scoring used the mention interval: u_b (mention at centre) beats
    # u_a (mention at edge) even though their occurred instants reverse.
    assert ids["u_b"].signals["t_prox"] == pytest.approx(1.0)
    assert ids["u_a"].signals["t_prox"] == pytest.approx(0.0)
    assert ids["u_b"].raw_score > ids["u_a"].raw_score
    assert ids["u_b"].signals["mention_interval"] == (centre, centre, "day")


def test_mentions_table_absent_degrades_partial():
    """Absent ``unit_time_mentions`` on a windowed query → honest
    ``partial`` (the window path still ran); never silent ``ok``."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE units (unit_id TEXT PRIMARY KEY, source_id TEXT,"
        " revision INTEGER, scope_id TEXT, kind TEXT, parent_unit_id TEXT,"
        " session_id TEXT, seq INTEGER, speaker_canon TEXT, perspective TEXT,"
        " recorded_at_us INTEGER, occurred_start_us INTEGER,"
        " occurred_end_us INTEGER, occurred_precision TEXT,"
        " occurred_source TEXT, byte_start INTEGER, byte_end INTEGER,"
        " generation INTEGER);"
    )
    ws, we = us(2023, 3, 1), us(2023, 4, 1)
    add_unit(conn, "u1", occurred=(us(2023, 3, 10), us(2023, 3, 10)))
    qv = mk_qv(IntentClass.TEMPORAL_RANGE, window=IntervalUs(ws, we))
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.PARTIAL
    assert out.reason == "mentions_table_missing"
    assert out.stats["mentions"] == "absent"
    assert "u1" in by_id(out)  # occurred path still delivered


def test_mentions_generation_fence():
    """Mention rows above the pinned generation are fenced out
    (V7-30.02 carried) — only the latest in-fence generation reads."""
    conn = mk_conn()
    ws, we = us(2023, 3, 1), us(2023, 4, 1)
    add_unit(conn, "u1", occurred=(us(2020, 1, 1), us(2020, 1, 2)), gen=1)
    # in-window mention exists only at gen 2 — invisible at pin 1
    add_time_mention(conn, "u1", start=us(2023, 3, 5), end=us(2023, 3, 6),
                     gen=2)
    qv = mk_qv(IntentClass.TEMPORAL_RANGE, window=IntervalUs(ws, we))
    assert "u1" not in by_id(tlane.lane_temporal(mk_ctx(conn, gen=1), qv, sl()))
    assert "u1" in by_id(tlane.lane_temporal(mk_ctx(conn, gen=2), qv, sl()))


# ---------------------------------------------------------------------------
# V8-09.07 — weighted subject/predicate disjunction (temporal.events_weights)
# ---------------------------------------------------------------------------


def _seed_events_disjunction(conn):
    add_unit(conn, "u_subj", occurred=(us(2023, 6, 3), us(2023, 6, 3)))
    add_event(conn, "e_subj", "u_subj", subject="alice", predicate="zzz",
              occurred=(us(2023, 6, 3), us(2023, 6, 3)))
    add_unit(conn, "u_pred", occurred=(us(2023, 6, 4), us(2023, 6, 4)))
    add_event(conn, "e_pred", "u_pred", subject="bob", predicate="move",
              occurred=(us(2023, 6, 4), us(2023, 6, 4)))
    add_unit(conn, "u_both", occurred=(us(2023, 6, 5), us(2023, 6, 5)))
    add_event(conn, "e_both", "u_both", subject="alice", predicate="move",
              occurred=(us(2023, 6, 5), us(2023, 6, 5)))


def test_events_weighted_disjunction_admits_either_side():
    """V8-09.07 — subject-only and predicate-only event matches now
    nominate candidates; the strict conjunction is gone."""
    conn = mk_conn()
    _seed_events_disjunction(conn)
    qv = mk_qv(
        IntentClass.TEMPORAL_POINT,
        terms=("when", "did", "alice", "move"),
        canons=("alice",),
    )
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())
    ids = by_id(out)
    assert {"u_subj", "u_pred", "u_both"} <= set(ids)
    assert ids["u_subj"].signals["subject_match"] == 1
    assert ids["u_subj"].signals["event_pred"] == 0
    assert ids["u_pred"].signals["subject_match"] == 0
    assert ids["u_pred"].signals["event_pred"] == 1
    # both-match scores higher than either single-side match
    assert ids["u_both"].raw_score > ids["u_subj"].raw_score
    assert ids["u_both"].raw_score > ids["u_pred"].raw_score
    assert out.stats["events_weights"] == [0.5, 0.5]


def test_events_weights_arm_reweights():
    """The §23 arm changes the composite: predicate-only beats
    subject-only when predicate weight dominates."""
    conn = mk_conn()
    _seed_events_disjunction(conn)
    qv = mk_qv(
        IntentClass.TEMPORAL_POINT,
        terms=("when", "did", "alice", "move"),
        canons=("alice",),
    )
    ctx = mk_ctx(conn, manifest={tlane.EVENTS_WEIGHTS_KEY: [0.1, 0.9]})
    out = tlane.lane_temporal(ctx, qv, sl())
    ids = by_id(out)
    assert ids["u_pred"].raw_score > ids["u_subj"].raw_score
    assert out.stats["events_weights"] == [0.1, 0.9]


def test_events_weights_invalid_shape_raises_validation():
    """A malformed arm value is a loud VALIDATION naming the arm —
    never a silent default."""
    conn = mk_conn()
    _seed_events_disjunction(conn)
    qv = mk_qv(
        IntentClass.TEMPORAL_POINT,
        terms=("move",), canons=("alice",),
    )
    ctx = mk_ctx(conn, manifest={tlane.EVENTS_WEIGHTS_KEY: "heavy"})
    with pytest.raises(VerbatimError) as ei:
        tlane.lane_temporal(ctx, qv, sl())
    assert ei.value.code == ErrorCode.VALIDATION
    assert tlane.EVENTS_WEIGHTS_KEY in str(ei.value)


# ---------------------------------------------------------------------------
# V8-09.06 — relevant-only no-window fallback, event-time ordered
# ---------------------------------------------------------------------------


def _seed_fallback(conn):
    """Relevant units (mention 'paris') at scattered event times, plus an
    irrelevant unit newer than all of them."""
    add_unit(conn, "u_2019", rowid=1,
             occurred=(us(2019, 3, 1), us(2019, 3, 1)),
             recorded=us(2023, 1, 1))
    add_unit(conn, "u_2023", rowid=2,
             occurred=(us(2023, 6, 1), us(2023, 6, 1)),
             recorded=us(2023, 6, 2))
    add_unit(conn, "u_2021", rowid=3,
             occurred=(us(2021, 1, 1), us(2021, 1, 1)),
             recorded=us(2021, 1, 2))
    add_unit(conn, "u_noise", rowid=4,
             occurred=(us(2024, 1, 1), us(2024, 1, 1)),
             recorded=us(2024, 1, 2))
    add_fts(conn, 1, "we moved to paris")
    add_fts(conn, 2, "paris again, finally")
    add_fts(conn, 3, "paris was cold")
    add_fts(conn, 4, "totally unrelated")


def test_fallback_orders_by_event_time_desc_for_recency():
    conn = mk_conn(fts=True)
    _seed_fallback(conn)
    qv = mk_qv(IntentClass.HISTORY_OF, terms=("latest", "paris"))
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.OK
    assert out.stats["mode"] == "relevant_fallback"
    assert out.stats["fallback_order"] == "desc"
    ids = [c.unit_id for c in out.candidates]
    # event-time order — newest first — and the irrelevant unit is gone
    assert ids == ["u_2023", "u_2021", "u_2019"]
    assert by_id(out)["u_2023"].signals["axis"] == "occurred"


def test_fallback_orders_asc_for_earliest_cue():
    conn = mk_conn(fts=True)
    _seed_fallback(conn)
    qv = mk_qv(IntentClass.TEMPORAL_ORDER,
               terms=("first", "paris", "visit"))
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())
    assert out.stats["fallback_order"] == "asc"
    assert [c.unit_id for c in out.candidates] == ["u_2019", "u_2021", "u_2023"]


def test_fallback_canon_leg_without_fts():
    """Relevance via resolved entity canon when the FTS index is absent."""
    conn = mk_conn()  # no unit_fts
    add_unit(conn, "u_rel", occurred=(us(2022, 5, 1), us(2022, 5, 1)),
             recorded=us(2022, 5, 2))
    add_mention(conn, "u_rel", "caroline")
    add_unit(conn, "u_noise", occurred=(us(2023, 1, 1), us(2023, 1, 1)),
             recorded=us(2023, 1, 2))
    qv = mk_qv(IntentClass.HISTORY_OF, terms=("when",), canons=("caroline",))
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())
    assert set(by_id(out)) == {"u_rel"}


def test_fallback_no_match_keys_scans_nothing():
    """No nominated terms/canons → the restriction cannot be applied, so
    the lane scans nothing rather than sweeping unrestricted."""
    conn = mk_conn(fts=True)
    _seed_fallback(conn)
    qv = mk_qv(IntentClass.HISTORY_OF, terms=())
    out = tlane.lane_temporal(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.OK
    assert out.stats["fallback"] == "no_match_keys"
    assert out.examined == 0 and out.candidates == []
