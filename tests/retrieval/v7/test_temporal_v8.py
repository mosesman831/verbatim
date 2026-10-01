"""V8 temporal verification (SPEC_V8 §09, §24 Temporal row, §25 K49–K58).

The §24 verification row this module owns:

  "``as_of`` validation; anchor precedence; year guard; owned mention
  fixtures; claim anchor; fallback ordering; event subject backfill"

Scenario map:

- K49: ``Memory.search(as_of=…)`` — naive datetime raises ``VALIDATION``
  naming ``as_of``; RFC 3339 accepted and ``coverage.temporal.anchor``
  reports ``{source: "caller", us}``; no ``as_of`` reports ``"wall"``.
- K50: the caller anchor beats the wall clock — "last week" under
  ``as_of=2023-05-20`` resolves a window inside May 2023 and a different
  anchor resolves a different window (the anchor is per-call).
- K51: V8-09.03 year guard — "Cyberpunk 2077" forms no window, "in 2019"
  does, "in May" anchored September 2023 resolves May 2023.
- K52: ``reltime.resolve_mentions`` + ``units_jobs._write_time_mentions``
  anchor "yesterday" on the unit's own ``occurred`` (2023-05-08 →
  [2023-05-07, 2023-05-08)); the lane's window scan unions the mention
  and labels it ``overlap="mention"``.
- K53: ambiguous "later" emits no mention row; a unit with no occurred
  bound gets no rows (``unit_anchor_us`` → ``None``, never a clock
  guess).
- K54: claim valid intervals anchored on the source unit's occurred with
  ``uncertainty_json.anchor`` recorded — the storage half is confirmed
  unlanded at HEAD (``repos.py`` writes ``uncertainty_json = NULL``
  unconditionally; ``propose`` has no occurred anchor parameter), so the
  scenario test is a strict xfail; the ``claims_anchor`` derivation
  honestly reports the debt as ``partial(n)``.
- K55: the no-window fallback orders by event time (ascending for
  "first/earliest", descending for recency), is restricted to nominated
  terms/canons, and truncation reports ``partial/scan_bound``.
- K56: ``_write_event`` backfills a first-person subject from the
  source unit's ``speaker_canon`` and marks ``subject_source =
  'speaker_backfill'``; the lane's weighted disjunction then matches
  "when did melanie …" queries.
- K57: a window-matched unit whose mention interval lies outside the
  window is scored by the mention (``axis="mention"``,
  ``mention_interval`` signal carries the interval).
- K58: the S2b need gate — ``need_gate(TIME, non-temporal query)``
  reports ``not needed`` and the lane itself reports ``skipped``
  when run anyway.

Lane tests reuse the sibling mirror-schema fixture module
``tests/retrieval/v7/test_tlane.py`` (real SQLite + real lane code);
write-path tests run against the real ``DDL_V1`` + ``schema_v7`` DDL.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from verbatim.core.types import (
    ErrorCode,
    Lifecycle,
    Modality,
    Polarity,
    Precision,
    TimeInterval,
    VerbatimError,
)
from verbatim.core.types_v7 import (
    IntentClass,
    IntervalUs,
    LaneName,
    LaneStatus,
    OccurredPrecision,
)
from verbatim.enrichment import reltime
from verbatim.jobs import units_jobs
from verbatim.retrieval.v7 import deadline as dl
from verbatim.retrieval.v7 import temporal as tlane
from verbatim.storage.schema import DDL_V1
from verbatim.storage.schema_v7 import ensure_v7_additive
from verbatim import Memory

from tests.retrieval.v7 import test_tlane as tl


def _us(y: int, m: int = 1, d: int = 1, h: int = 0, mi: int = 0) -> int:
    return int(
        datetime(y, m, d, h, mi, tzinfo=timezone.utc).timestamp() * 1_000_000
    )


# ---------------------------------------------------------------------------
# K49–K51 — caller as_of anchor + year guard (real facade)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mem(tmp_path):
    m = Memory(path=str(tmp_path / "t8.db"), worker="external")
    yield m
    try:
        m.close()
    except Exception:
        pass


def test_k49_as_of_validation_and_coverage(mem):
    """``as_of`` admission: naive datetime / offset-less strings raise
    VALIDATION naming ``as_of``; RFC 3339 is accepted and coverage
    reports ``anchor.source == "caller"`` with the resolved µs."""
    for bad in (
        datetime(2023, 5, 20, 12),       # naive datetime
        "2023-05-20",                    # offset-less string
        "garbage",
        True,
        10**30,                          # out of representable range
    ):
        with pytest.raises(VerbatimError) as ei:
            mem.search("anything", retrieval="v7", as_of=bad)
        assert ei.value.code == ErrorCode.VALIDATION
        assert "as_of" in str(ei.value)

    res = mem.search("anything", retrieval="v7", as_of="2023-05-20T12:00:00Z")
    assert res.coverage["temporal"]["anchor"] == {
        "source": "caller",
        "us": _us(2023, 5, 20, 12),
    }
    res2 = mem.search("anything", retrieval="v7")
    assert res2.coverage["temporal"]["anchor"]["source"] == "wall"
    assert isinstance(res2.coverage["temporal"]["anchor"]["us"], int)


def test_k50_caller_anchor_drives_window(mem):
    """"What did Caroline do last week?" anchored 2023-05-20 resolves a
    window inside May 2023 — the wall clock is never consulted; a
    different anchor resolves a different window."""
    with mem._store.read() as conn:
        qv = mem._v7_query_view(
            "what did caroline do last week", _us(2023, 5, 20), conn
        )
        win = qv.intent.window
        assert win is not None
        assert _us(2023, 5, 1) <= win.start_us
        assert win.end_us <= _us(2023, 6, 1)
        # A different caller anchor moves the window — the wall clock
        # is not what resolved it.
        qv2 = mem._v7_query_view(
            "what did caroline do last week", _us(2024, 1, 10), conn
        )
        win2 = qv2.intent.window
        assert win2 is not None and win2.start_us != win.start_us
        assert _us(2023, 12, 15) <= win2.start_us <= _us(2024, 1, 10)


def test_k51_year_guard(mem):
    """V8-09.03 — bare ``YYYY`` windows survive only with a cue or in
    range: "Cyberpunk 2077" → no window; "in 2019" → 2019; "in May"
    anchored Sep 2023 → May 2023 (most-recent-past rule)."""
    with mem._store.read() as conn:
        qv = mem._v7_query_view(
            "should i play cyberpunk 2077", _us(2023, 9, 15), conn
        )
        win = qv.intent.window
        assert win is None or not (
            win.start_us == _us(2077, 1, 1) and win.end_us == _us(2078, 1, 1)
        ), "cyberpunk 2077 must not become a year window"

        qv2 = mem._v7_query_view("what happened in 2019", _us(2023, 9, 15), conn)
        win2 = qv2.intent.window
        assert win2 is not None
        assert win2.start_us <= _us(2019, 6, 1) <= win2.end_us

        qv3 = mem._v7_query_view("what did i do in may", _us(2023, 9, 15), conn)
        win3 = qv3.intent.window
        assert win3 is not None
        assert win3.start_us <= _us(2023, 5, 1) <= win3.end_us
        assert win3.end_us <= _us(2023, 6, 3)


# ---------------------------------------------------------------------------
# K52/K53 — write-time mentions (reltime + units_jobs on the real schema)
# ---------------------------------------------------------------------------


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:", isolation_level=None)
    c.executescript(DDL_V1)
    ensure_v7_additive(c)
    yield c
    c.close()


def _unit_row(conn, unit_id, *, occurred, scope=tl.SCOPE, gen=tl.GEN,
              speaker=None, text=None):
    """Insert a real ``units`` row; returns the dict shape the writer
    reads (``occurred_start_us``/``occurred_end_us``)."""
    tl.add_unit(
        conn, unit_id, occurred=occurred, scope=scope, gen=gen,
        speaker=speaker,
    )
    row = conn.execute(
        "SELECT * FROM units WHERE unit_id=?", (unit_id,)
    ).fetchone()
    cols = [d[0] for d in conn.execute("SELECT * FROM units LIMIT 0").description]
    return dict(zip(cols, row))


def test_k52_write_time_mention_and_lane_union(conn):
    """Occurred-anchored "two days ago" (mention disjoint from occurred)
    persists a ``unit_time_mentions`` row; the window scan matches the
    unit through the mention union labeled ``overlap="mention"``."""
    text = "two days ago I adopted a puppy"
    anchor = _us(2023, 5, 8)
    unit = _unit_row(conn, "u-pup", occurred=(_us(2023, 5, 8), _us(2023, 5, 9)))

    stats = {"time_mentions": 0}
    units_jobs._write_time_mentions(
        conn, tl.SCOPE, tl.GEN, unit, text, stats
    )
    rows = conn.execute(
        "SELECT start_us, end_us, precision, span_start, span_end,"
        " anchor_us, resolver_version FROM unit_time_mentions"
    ).fetchall()
    assert stats["time_mentions"] == 1 and len(rows) == 1
    assert rows[0][:2] == (_us(2023, 5, 6), _us(2023, 5, 7))
    assert rows[0][2] == "day"
    assert rows[0][5] == anchor
    assert rows[0][6] == reltime.RESOLVER_VERSION

    # The unit's occurred [5-8, 5-9) does NOT overlap the mention window
    # [5-6, 5-7) — the match can only arrive through the mention union.
    window = IntervalUs(
        start_us=_us(2023, 5, 6), end_us=_us(2023, 5, 7),
        precision=OccurredPrecision.DAY,
    )
    qv = tl.mk_qv(
        primary=IntentClass.TEMPORAL_RANGE, window=window,
        terms=("puppy",), q="when did i adopt a puppy",
    )
    out = tlane.lane_temporal(tl.mk_ctx(conn), qv, tl.sl())
    assert out.status == LaneStatus.OK
    assert out.stats["mentions"] == "ok"
    got = tl.by_id(out)
    assert "u-pup" in got
    assert got["u-pup"].signals["overlap"] == "mention"
    assert got["u-pup"].signals["axis"] == "mention"
    assert got["u-pup"].signals["mention_interval"] == (
        _us(2023, 5, 6), _us(2023, 5, 7), "day",
    )


def test_k53_ambiguous_and_unanchored_emit_no_rows(conn):
    """Precision-first: "see you later" resolves to nothing storable;
    a unit without occurred bounds gets no rows at all."""
    stats = {"time_mentions": 0}
    # Ambiguous: ``resolve`` returns unknown-bounds candidates; the
    # writer emits zero rows.
    unit = _unit_row(conn, "u-later", occurred=(_us(2023, 5, 8), _us(2023, 5, 9)))
    units_jobs._write_time_mentions(conn, tl.SCOPE, tl.GEN, unit,
                                    "see you later", stats)
    assert stats["time_mentions"] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM unit_time_mentions"
    ).fetchone()[0] == 0

    # No occurred bounds → ``unit_anchor_us`` is None → no rows (never
    # a wall-clock substitute).
    bare = _unit_row(conn, "u-none", occurred=(None, None))
    assert reltime.unit_anchor_us(bare) is None
    units_jobs._write_time_mentions(conn, tl.SCOPE, tl.GEN, bare,
                                  "yesterday I adopted a puppy", stats)
    assert stats["time_mentions"] == 0


# ---------------------------------------------------------------------------
# K54 — claim anchor (V8-09.05): storage half unlanded → strict xfail
# ---------------------------------------------------------------------------


def _seed_claim(tmp_path):
    """A real claim + revision with a relative-date valid interval,
    written through the real ``Store`` + ``ClaimsRepo`` path — the
    shape ``claims_anchor`` reads."""
    from verbatim.core.types import Scope, Visibility
    from verbatim.storage.repos import ClaimsRepo, ensure_scope
    from verbatim.storage.store import Store

    store = Store.create(str(tmp_path / "k54.db"))
    scope = Scope(profile_id="p", principal_id="alice",
                  visibility=Visibility.CONVERSATION)
    with store.tx() as conn:
        sid = ensure_scope(store, conn, scope)
        repo = ClaimsRepo(store)
        claim_id = repo.create(sid, None, "adopted", conn)
        seq = conn.execute(
            "SELECT COALESCE(MAX(event_seq), 0) + 1 FROM events"
        ).fetchone()[0]
        repo.add_revision(
            claim_id,
            Lifecycle.ACTIVE,
            {"text": "I adopted a puppy last month"},
            Polarity.AFFIRMATIVE,
            Modality.ASSERTED,
            None,
            None,
            [TimeInterval(from_us=_us(2023, 4, 1), precision=Precision.MONTH)],
            [],
            seq,
            conn,
        )
    return store, sid, claim_id


def test_k54_claims_anchor_derivation_reports_debt(tmp_path):
    """The ``claims_anchor`` derivation honestly reports unanchored
    latest-revision intervals as ``partial(n)`` — real machinery even
    while the writer half is unlanded."""
    store, sid, _claim = _seed_claim(tmp_path)
    with store.read() as conn:
        out = units_jobs.derivation_coverage_v8(conn, sid, 1)
        assert out["claims_anchor"] == "partial(1)"
    store.close()


@pytest.mark.xfail(
    strict=True,
    reason="V8-09.05 unlanded at HEAD: propose() takes only "
           "envelope.event_us (no occurred anchor) and "
           "ClaimsRepo.add_revision writes uncertainty_json=NULL "
           "unconditionally (repos.py) — the anchor is never recorded.")
def test_k54_claim_anchor_recorded(tmp_path):
    """K54 — a claim from an occurred-bearing source anchors relative
    intervals at occurred and records ``uncertainty_json.anchor =
    "occurred"``; an unanchored source records ``"ingest"``."""
    store, _sid, claim_id = _seed_claim(tmp_path)
    with store.read() as conn:
        row = conn.execute(
            "SELECT uncertainty_json FROM valid_intervals WHERE claim_id=?",
            (claim_id,),
        ).fetchone()
    store.close()
    import json
    assert json.loads(row[0])["anchor"] in ("occurred", "ingest")


# ---------------------------------------------------------------------------
# K55 — relevant-only fallback ordering + scan bound
# ---------------------------------------------------------------------------


def _fallback_units(conn, *, fts=True):
    """Three occurred-dated, term-bearing units for the fallback scan."""
    rows = [
        ("u-old", _us(2020, 3, 1), "alice adopted a cat years ago"),
        ("u-mid", _us(2022, 6, 1), "alice adopted a dog last year"),
        ("u-new", _us(2023, 8, 1), "alice adopted a puppy recently"),
    ]
    for i, (uid, occ, text) in enumerate(rows, start=1):
        tl.add_unit(
            conn, uid, occurred=(occ, occ + 86_400_000_000), rowid=i,
        )
        if fts:
            tl.add_fts(conn, i, text)
        tl.add_mention(conn, uid, "alice")
    return rows


def test_k55_fallback_earliest_ascending():
    conn = tl.mk_conn(fts=True)
    _fallback_units(conn)
    qv = tl.mk_qv(
        primary=IntentClass.TEMPORAL_RANGE,  # temporal, no window
        terms=("first", "adopt", "alice"),
        canons=("alice",),
        q="when did alice first adopt a pet",
    )
    out = tlane.lane_temporal(tl.mk_ctx(conn), qv, tl.sl())
    assert out.status == LaneStatus.OK
    assert out.stats["mode"] == "relevant_fallback"
    assert out.stats["fallback"] == "ok"
    assert out.stats["fallback_order"] == "asc"
    # Ascending event time — the fallback IS the temporal ordering.
    assert [c.unit_id for c in out.candidates] == ["u-old", "u-mid", "u-new"]
    assert all(c.signals["overlap"] == "fallback" for c in out.candidates)


def test_k55_fallback_recency_descending():
    conn = tl.mk_conn(fts=True)
    _fallback_units(conn)
    qv = tl.mk_qv(
        primary=IntentClass.TEMPORAL_RANGE,
        terms=("latest", "adopt", "alice"),
        canons=("alice",),
        q="when did alice latest adopt a pet",
    )
    out = tlane.lane_temporal(tl.mk_ctx(conn), qv, tl.sl())
    assert out.status == LaneStatus.OK
    assert out.stats["fallback_order"] == "desc"
    assert [c.unit_id for c in out.candidates] == ["u-new", "u-mid", "u-old"]


def test_k55_fallback_restriction_legs():
    """The fallback is never an ingest-order sweep: with no terms,
    canons, or speaker the scan cannot run (``no_match_keys``); and a
    unit matching no leg is excluded."""
    conn = tl.mk_conn(fts=True)
    _fallback_units(conn)
    tl.add_unit(conn, "u-irrelevant",
                occurred=(_us(2021, 1, 1), _us(2021, 1, 2)), rowid=9)
    tl.add_fts(conn, 9, "bob fixed his bicycle")
    qv = tl.mk_qv(
        primary=IntentClass.TEMPORAL_RANGE,
        terms=("latest", "adopt", "alice"),
        canons=("alice",),
        q="when did alice latest adopt",
    )
    out = tlane.lane_temporal(tl.mk_ctx(conn), qv, tl.sl())
    ids = {c.unit_id for c in out.candidates}
    assert "u-irrelevant" not in ids

    qv2 = tl.mk_qv(primary=IntentClass.TEMPORAL_RANGE, terms=(),
                   canons=(), q="when")
    out2 = tlane.lane_temporal(tl.mk_ctx(conn), qv2, tl.sl())
    assert out2.stats["fallback"] == "no_match_keys"
    assert out2.candidates == []


def test_k55_fallback_scan_bound_reports_partial(monkeypatch):
    conn = tl.mk_conn(fts=True)
    _fallback_units(conn)
    monkeypatch.setattr(tlane, "SCAN_ROW_LIMIT", 2)
    qv = tl.mk_qv(
        primary=IntentClass.TEMPORAL_RANGE,
        terms=("latest", "adopt", "alice"),
        canons=("alice",),
        q="when did alice latest adopt",
    )
    out = tlane.lane_temporal(tl.mk_ctx(conn), qv, tl.sl())
    assert out.status == LaneStatus.PARTIAL
    assert out.reason == "scan_bound"
    assert out.stats["scan_truncated"] is True


# ---------------------------------------------------------------------------
# K56 — first-person subject backfill (V8-09.07)
# ---------------------------------------------------------------------------


def test_k56_speaker_backfill_write_and_match(conn):
    """"I started painting" by melanie: the extractor's subject token is
    a first-person pronoun so ``_write_event`` backfills
    ``subject_canon='melanie'`` and marks ``subject_source=
    'speaker_backfill'``; the lane's weighted disjunction then matches
    a "when did melanie …" query."""
    from verbatim.enrichment.events import extract_events
    from verbatim.text.norm_v2 import analyze

    text = "I started painting"
    unit = _unit_row(
        conn, "u-paint",
        occurred=(_us(2023, 5, 8), _us(2023, 5, 9)), speaker="melanie",
    )
    norm = analyze(text)
    occurred = IntervalUs(
        start_us=unit["occurred_start_us"],
        end_us=unit["occurred_end_us"],
        precision=OccurredPrecision.DAY,
    )
    # Real extractor, no speaker supplied — first-person subject emits
    # subject_canon=None (the writer's backfill case).
    evs = extract_events(norm, "u-paint", None, occurred, raw_text=text)
    assert evs, "extractor produced no events"
    ev = evs[0]
    assert ev.subject_canon is None
    assert ev.pins["subject"] == (0, 1)  # the "I" span

    stats = {"events": 0, "events_speaker_backfill": 0}
    units_jobs._write_event(
        conn, tl.SCOPE, tl.GEN, "u-paint", ev, stats,
        speaker_canon="melanie", unit_text=text, has_subject_source=True,
    )
    row = conn.execute(
        "SELECT subject_canon, subject_source, predicate_lemma"
        " FROM events_v7"
    ).fetchone()
    assert row is not None
    assert row[0] == "melanie" and row[1] == "speaker_backfill"
    assert stats["events_speaker_backfill"] >= 1

    # Query side: "when did melanie start painting" — a subject-canon
    # match through the weighted disjunction (no conjunction required).
    window = IntervalUs(
        start_us=_us(2023, 5, 1), end_us=_us(2023, 6, 1),
        precision=OccurredPrecision.MONTH,
    )
    qv = tl.mk_qv(
        primary=IntentClass.TEMPORAL_POINT,
        window=window,
        terms=("start", "paint"),
        canons=("melanie",),
        q="when did melanie start painting",
    )
    out = tlane.lane_temporal(tl.mk_ctx(conn), qv, tl.sl())
    assert out.status == LaneStatus.OK
    got = tl.by_id(out)
    assert "u-paint" in got
    assert got["u-paint"].signals["event_match"] is True
    assert got["u-paint"].signals["subject_match"] == 1


def test_k56_non_pronoun_subject_not_backfilled(conn):
    """Only first-person pronouns backfill: an unresolved non-pronoun
    subject stays NULL and is marked ``extracted``."""
    from verbatim.core.types import new_id
    import types as _types

    text = "the manager started painting"
    _unit_row(conn, "u-mgr", occurred=(_us(2023, 5, 8), _us(2023, 5, 9)),
              speaker="melanie")
    ev = _types.SimpleNamespace(
        subject_canon=None,
        predicate_lemma="start",
        object_text="",
        polarity="affirm",
        occurred=IntervalUs(_us(2023, 5, 8), _us(2023, 5, 9),
                            precision=OccurredPrecision.DAY),
        pins={"subject": (0, 12), "span": (0, 27)},  # "the manager"
        rule_id="event/v1:test",
    )
    stats = {"events": 0, "events_speaker_backfill": 0}
    units_jobs._write_event(
        conn, tl.SCOPE, tl.GEN, "u-mgr", ev, stats,
        speaker_canon="melanie", unit_text=text, has_subject_source=True,
    )
    row = conn.execute(
        "SELECT subject_canon, subject_source FROM events_v7"
    ).fetchone()
    assert row[0] is None and row[1] == "extracted"
    assert stats["events_speaker_backfill"] == 0


# ---------------------------------------------------------------------------
# K57 — event date beats session date (V8-09.08)
# ---------------------------------------------------------------------------


def test_k57_mention_interval_scores_over_session_occurred():
    conn = tl.mk_conn()
    # Window: May 2023. Unit occurred inside it; its only mention is in
    # April (outside) — the mention drives t_prox/axis anyway.
    ws, we = _us(2023, 5, 1), _us(2023, 6, 1)
    tl.add_unit(conn, "u-m", occurred=(_us(2023, 5, 10), _us(2023, 5, 11)))
    tl.add_time_mention(
        conn, "u-m", start=_us(2023, 4, 19), end=_us(2023, 4, 20),
        precision="day",
    )
    tl.add_unit(conn, "u-plain",
                occurred=(_us(2023, 5, 10), _us(2023, 5, 11)))
    window = IntervalUs(start_us=ws, end_us=we,
                        precision=OccurredPrecision.MONTH)
    qv = tl.mk_qv(primary=IntentClass.TEMPORAL_RANGE, window=window,
                  terms=("anything",), q="what happened in may")
    out = tlane.lane_temporal(tl.mk_ctx(conn), qv, tl.sl())
    assert out.status == LaneStatus.OK
    got = tl.by_id(out)
    assert "u-m" in got and "u-plain" in got

    sig = got["u-m"].signals
    assert sig["axis"] == "mention"
    assert sig["mention_interval"] == (_us(2023, 4, 19), _us(2023, 4, 20), "day")
    # The mention midpoint is ~3 weeks before the window centre →
    # t_prox clamps to 0; the control unit scores on its occurred.
    assert sig["t_prox"] == 0.0
    assert got["u-plain"].signals.get("t_prox", 0) > 0.5
    assert "mention_interval" not in got["u-plain"].signals


# ---------------------------------------------------------------------------
# K58 — S2b need gate
# ---------------------------------------------------------------------------


def test_k58_need_gate_skips_nontemporal_query():
    """A non-temporal, windowless query gates the time lane out —
    ``need_gate`` reports ``(False, inputs)`` and the scheduler maps it
    to ``skipped(not_needed)``; the lane's own gate agrees."""
    qv = tl.mk_qv(primary=IntentClass.LOOKUP, terms=("sardines",),
                  q="what food do i love")
    needed, inputs = dl.need_gate(
        LaneName.TIME, qv, core_union=0, union_floor=32,
    )
    assert needed is False
    assert inputs == {"window": False, "temporal_intent": False}

    # The lane's own gate is the same contract: run anyway → skipped.
    out = tlane.lane_temporal(tl.mk_ctx(tl.mk_conn()), qv, tl.sl())
    assert out.status == LaneStatus.SKIPPED
    assert out.reason == "no_window"


def test_k58_need_gate_admits_temporal():
    """Window or temporal intent flips the gate on — gate inputs are
    logged for explain (V8-05.05's coverage contract)."""
    window = IntervalUs(start_us=_us(2023, 5, 1), end_us=_us(2023, 6, 1),
                        precision=OccurredPrecision.MONTH)
    qv_win = tl.mk_qv(primary=IntentClass.LOOKUP, window=window)
    needed, inputs = dl.need_gate(
        LaneName.TIME, qv_win, core_union=0, union_floor=32,
    )
    assert needed is True and inputs["window"] is True

    qv_int = tl.mk_qv(primary=IntentClass.TEMPORAL_RANGE)
    needed2, inputs2 = dl.need_gate(
        LaneName.TIME, qv_int, core_union=0, union_floor=32,
    )
    assert needed2 is True and inputs2["temporal_intent"] is True
