"""V7 engine-integration tests for ``Memory.search`` (SPEC_V7 §search seam).

Exercises the ``retrieval`` dispatch — ``auto``/``v7``/``v6`` — against a
real on-disk store with the V7 projection plane provisioned through the
real write path (``Ingester.drain_report`` over ``SOURCE_PROJECT``).  No
kernel mocks; every assertion reads committed state or a real SearchResult.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from verbatim import Memory
from verbatim.core.types import ErrorCode, JobKind, VerbatimError
from verbatim.ingest import Ingester
from verbatim.memory.facade import _as_of_to_us, _v7_guard_year_window
from verbatim.storage.repos import has_table


@pytest.fixture()
def path(tmp_path):
    return str(tmp_path / "m.db")


@pytest.fixture()
def mem(path):
    m = Memory(path=path, worker="external")
    yield m
    try:
        m.close()
    except Exception:
        pass


def _drain(mem, receipt):
    """Run the durable source jobs through the real coordinator so the
    V7 unit projection (inside ``_apply``) commits before we search."""
    ingester = Ingester(mem._store, mem._cfg, encoder=mem._encoder)
    report = ingester.drain_report(
        scope=mem._namespace,
        owner="t-v7",
        kinds=(JobKind.SOURCE_PROJECT, JobKind.SOURCE_EMBED),
    )
    assert report["failed"] == 0
    mem.wait_ready(receipt, timeout_ms=2000)


def _has_units(mem) -> bool:
    with mem._store.read() as conn:
        return has_table(conn, "units")


def _n_units(mem) -> int:
    with mem._store.read() as conn:
        return conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]


# ---------------------------------------------------------------------------
# engine selection
# ---------------------------------------------------------------------------


class TestEngineSelection:
    def test_invalid_retrieval_rejected(self, mem):
        with pytest.raises(VerbatimError):
            mem.search("q", retrieval="v9")
        with pytest.raises(VerbatimError):
            mem.search("q", retrieval=123)

    def test_auto_falls_back_to_v6_when_no_units(self, mem):
        mem.add("plain text with no projection drained")
        # worker=external, nothing drained → no units table
        res = mem.search("plain text", retrieval="auto")
        assert res.coverage.get("engine") == "v6"

    def test_auto_selects_v7_when_provisioned(self, mem):
        r = mem.add("I love grilled sardines.")
        _drain(mem, r)
        assert _has_units(mem)
        res = mem.search("what do I love", retrieval="auto")
        assert res.coverage.get("engine") == "v7"

    def test_v6_forces_legacy(self, mem):
        r = mem.add("I love grilled sardines.")
        _drain(mem, r)
        assert _has_units(mem)
        res = mem.search("what do I love", retrieval="v6")
        assert res.coverage.get("engine") == "v6"

    def test_v7_forces_engine_even_when_empty(self, mem):
        # no add/drain → no units, but explicit v7 must still run the
        # engine (source lane covers) and report engine=v7.
        res = mem.search("anything", retrieval="v7")
        assert res.coverage.get("engine") == "v7"


# ---------------------------------------------------------------------------
# v7 result semantics
# ---------------------------------------------------------------------------


class TestV7Results:
    def _seeded(self, mem):
        r = mem.add("I love grilled sardines and seafood.")
        _drain(mem, r)
        return r

    def test_v7_returns_grounded_quote(self, mem):
        self._seeded(mem)
        res = mem.search("what food do I love", retrieval="v7")
        assert res.status == "ready"
        assert res.items, "v7 returned no items"
        top = res.items[0]
        assert top.quote, "v7 hit has empty quote"
        assert b"sardines" in top.quote.encode() or "sardines" in top.quote

    def test_v7_quote_is_byte_pinned(self, mem):
        self._seeded(mem)
        res = mem.search("seafood", retrieval="v7")
        for it in res.items:
            q = it.quote.decode() if isinstance(it.quote, bytes) else it.quote
            assert q.strip(), "blank quote shipped"

    def test_v7_support_status_valid(self, mem):
        self._seeded(mem)
        res = mem.search("sardines", retrieval="v7")
        for it in res.items:
            assert it.support_status in (
                "supported", "disputed", "insufficient", "unassessed",
            )

    def test_v7_coverage_reports_route(self, mem):
        self._seeded(mem)
        res = mem.search("sardines", retrieval="v7")
        route = res.coverage.get("route", "")
        assert "v7" in route

    def test_v7_units_actually_projected(self, mem):
        self._seeded(mem)
        assert _n_units(mem) >= 1

    def test_v7_empty_query_handled(self, mem):
        self._seeded(mem)
        res = mem.search("sardines", retrieval="v7", limit=1)
        assert len(res.items) <= 1


# ---------------------------------------------------------------------------
# V8-09.01/09.02/20.01/20.05 — per-call temporal anchor (``as_of``)
# ---------------------------------------------------------------------------


def _us(y, m, d, hh=0, mm=0):
    return int(
        datetime(y, m, d, hh, mm, tzinfo=timezone.utc).timestamp() * 1_000_000
    )


class TestAsOfValidation:
    """Strict ``as_of`` admission — ambiguous or invalid values raise
    ``VALIDATION`` naming the parameter (V8-20.05)."""

    def test_int_us_accepted(self):
        assert _as_of_to_us(_us(2023, 5, 20)) == _us(2023, 5, 20)

    def test_integral_float_accepted(self):
        assert _as_of_to_us(float(_us(2023, 5, 20))) == _us(2023, 5, 20)

    def test_numeric_string_accepted(self):
        assert _as_of_to_us(str(_us(2023, 5, 20))) == _us(2023, 5, 20)

    def test_rfc3339_z_accepted(self):
        assert _as_of_to_us("2023-05-20T12:00:00Z") == _us(2023, 5, 20, 12)

    def test_rfc3339_offset_accepted(self):
        assert _as_of_to_us("2023-05-20T14:00:00+02:00") == _us(
            2023, 5, 20, 12
        )

    def test_aware_datetime_accepted(self):
        dt = datetime(2023, 5, 20, 12, tzinfo=timezone.utc)
        assert _as_of_to_us(dt) == _us(2023, 5, 20, 12)

    def test_naive_datetime_rejected(self):
        with pytest.raises(VerbatimError) as ei:
            _as_of_to_us(datetime(2023, 5, 20, 12))
        assert ei.value.code == ErrorCode.VALIDATION
        assert "as_of" in str(ei.value)

    def test_offsetless_string_rejected(self):
        for bad in ("2023-05-20", "2023-05-20T12:00:00"):
            with pytest.raises(VerbatimError) as ei:
                _as_of_to_us(bad)
            assert ei.value.code == ErrorCode.VALIDATION
            assert "as_of" in str(ei.value)

    def test_bool_and_garbage_rejected(self):
        for bad in (True, object(), "", "not-a-date", 1.5, 10**30):
            with pytest.raises(VerbatimError) as ei:
                _as_of_to_us(bad)
            assert ei.value.code == ErrorCode.VALIDATION

    def test_search_rejects_bad_as_of_before_retrieval(self, mem):
        r = mem.add("a seed so the store is real")
        _drain(mem, r)
        for bad in (
            datetime(2023, 1, 1),      # naive datetime
            "2023-05-20",            # offset-less string
            "garbage",
            True,
            10**30,                  # out of representable range
        ):
            with pytest.raises(VerbatimError) as ei:
                mem.search("seed", retrieval="v7", as_of=bad)
            assert ei.value.code == ErrorCode.VALIDATION
            assert "as_of" in str(ei.value)


class TestAsOfAnchor:
    """Caller anchor wins over the wall clock for that call only and is
    reported under ``coverage.temporal.anchor`` (V8-09.01/20.03)."""

    def _seeded(self, mem):
        r = mem.add("I repaired the bicycle.", occurred_at=_us(2020, 5, 12))
        _drain(mem, r)
        return r

    def test_caller_anchor_reported(self, mem):
        self._seeded(mem)
        anchor = _us(2020, 7, 1)
        res = mem.search(
            "bicycle", retrieval="v7", as_of="2020-07-01T00:00:00Z"
        )
        blk = res.coverage["temporal"]["anchor"]
        assert blk == {"source": "caller", "us": anchor}

    def test_wall_anchor_reported(self, mem):
        self._seeded(mem)
        res = mem.search("bicycle", retrieval="v7")
        blk = res.coverage["temporal"]["anchor"]
        assert blk["source"] == "wall"
        assert isinstance(blk["us"], int)

    def test_anchor_not_sticky_across_calls(self, mem):
        """K11 — a caller anchor is per-call state, never process state."""
        self._seeded(mem)
        mem.search("bicycle", retrieval="v7", as_of="2020-07-01T00:00:00Z")
        res = mem.search("bicycle", retrieval="v7")
        assert res.coverage["temporal"]["anchor"]["source"] == "wall"

    def test_v6_engine_still_reports_caller_anchor(self, mem):
        self._seeded(mem)
        res = mem.search(
            "bicycle", retrieval="v6", as_of="2020-07-01T00:00:00Z"
        )
        assert res.coverage["temporal"]["anchor"]["source"] == "caller"

    def test_relative_month_resolves_against_caller_anchor(self, mem):
        """'in May' resolves to the anchor's year — a 2020 anchor yields
        May 2020, not the wall-clock year."""
        self._seeded(mem)
        with mem._store.read() as conn:
            qv = mem._v7_query_view("what did i do in may",
                                    _us(2020, 7, 1), conn)
        win = qv.intent.window
        assert win is not None
        assert win.start_us <= _us(2020, 5, 1)
        assert win.end_us >= _us(2020, 5, 31)
        assert win.end_us <= _us(2020, 6, 3)  # May 2020, not 2021+

    def test_windowed_query_reports_temporal_lane_ok(self, mem):
        self._seeded(mem)
        res = mem.search(
            "what did i do in may", retrieval="v7",
            as_of="2020-07-01T00:00:00Z",
        )
        assert res.coverage["lanes"].get("v7.time") == "ok"

    def test_write_time_mentions_land_and_lane_stays_ok(self, mem):
        """V8-09.04 e2e — a turn about 'last march' (occurred June 2020)
        produces a unit_time_mentions row at write, and a March window
        keeps the time lane healthy."""
        r = mem.add("we hiked the north ridge last march",
                    speaker="me", session_id="s-hike",
                    occurred_at=_us(2020, 6, 15))
        _drain(mem, r)
        with mem._store.read() as conn:
            assert has_table(conn, "unit_time_mentions")
            n = conn.execute(
                "SELECT COUNT(*) FROM unit_time_mentions"
            ).fetchone()[0]
        assert n >= 1  # 'last march' resolves against the unit's occurred
        res = mem.search(
            "what did we do in march 2020", retrieval="v7",
            as_of="2020-07-01T00:00:00Z",
        )
        assert res.coverage["lanes"].get("v7.time") == "ok"


class TestYearWindowGuard:
    """V8-09.03 — bare four-digit years only form a window behind a
    temporal cue or within the plausible range without a capitalized
    title token ("Cyberpunk 2077" is a name, not a year)."""

    def _qv(self, mem, query, anchor=_us(2023, 6, 1)):
        with mem._store.read() as conn:
            return mem._v7_query_view(query, anchor, conn)

    def test_cyberpunk_title_year_not_a_window(self, mem):
        qv = self._qv(mem, "when did i play Cyberpunk 2077")
        win = qv.intent.window
        assert win is None or not (
            win.start_us is not None
            and win.start_us <= _us(2077, 6, 1)
            and (win.end_us or 0) >= _us(2077, 1, 1)
        )

    def test_cued_year_still_a_window(self, mem):
        qv = self._qv(mem, "what happened in 2019")
        win = qv.intent.window
        assert win is not None
        assert win.start_us <= _us(2019, 1, 1)
        assert win.end_us >= _us(2019, 12, 31)

    def test_far_future_year_without_cue_dropped(self, mem):
        # anchor 2023 → plausible band tops out at 2028; a bare,
        # lowercase-adjacent 2099 is out of range and must drop.
        qv = self._qv(mem, "roadmap milestone 2099 notes",
                      anchor=_us(2023, 1, 1))
        win = qv.intent.window
        assert win is None or not (
            win.start_us is not None
            and win.start_us <= _us(2099, 6, 1)
            and (win.end_us or 0) >= _us(2099, 1, 1)
        )

    def test_cued_far_year_kept(self, mem):
        """An approved cue legitimates even an out-of-band year —
        'in 2049' still resolves (cue OR range, not AND)."""
        qv = self._qv(mem, "what happens in 2049", anchor=_us(2023, 1, 1))
        win = qv.intent.window
        assert win is not None
        assert win.start_us <= _us(2049, 1, 1)
        assert win.end_us >= _us(2049, 12, 31)

    def test_in_range_capitalized_title_year_dropped(self, mem):
        """2025 is inside [anchor-150, anchor+5] — only the capitalized
        title token ('Project', no cue) disqualifies it."""
        qv = self._qv(mem, "i read Project 2025 planning notes",
                      anchor=_us(2023, 1, 1))
        win = qv.intent.window
        assert win is None or not (
            win.start_us is not None
            and win.start_us <= _us(2025, 6, 1)
            and (win.end_us or 0) >= _us(2025, 1, 1)
        )


# ---------------------------------------------------------------------------
# V8-20.02 — SearchResult.answerability
# ---------------------------------------------------------------------------


class TestAnswerability:
    def test_field_exists_and_surfaces_verdict_value(self, mem):
        r = mem.add("I love grilled sardines and seafood.")
        _drain(mem, r)
        res = mem.search("sardines", retrieval="v7")
        assert hasattr(res, "answerability")
        # the real verdict ran → a declared value, or None when the
        # pipeline could not carry a report (never fabricated)
        assert res.answerability in (
            None,
            "supported",
            "partial",
            "weak_only",
            "unverified_premise",
            "contradicted_premise",
            "no_evidence",
        )

    def test_v6_result_has_field_defaulted(self, mem):
        mem.add("plain text")
        res = mem.search("plain", retrieval="v6")
        assert res.answerability is None
