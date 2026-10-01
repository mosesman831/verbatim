"""``enrichment/reltime`` tests (SPEC_V8 V8-09.04, §21.6).

The write-time relative-time resolver: relative expressions resolve on
the unit's own occurred anchor, precision-first — ambiguous, explicit,
or unbounded hits emit no row. Pure functions only; no store needed.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verbatim.enrichment import reltime  # noqa: E402


def _us(y, m, d, hh=0, mm=0):
    return int(datetime(y, m, d, hh, mm, tzinfo=timezone.utc).timestamp() * 1e6)


ANCHOR_1219 = _us(2023, 12, 19, 10, 4)          # a Tuesday
D20, D21 = _us(2023, 12, 20), _us(2023, 12, 21)


class TestUnitAnchor:
    def test_start_preferred(self):
        u = {"occurred_start_us": 11, "occurred_end_us": 22}
        assert reltime.unit_anchor_us(u) == 11

    def test_end_fallback(self):
        assert reltime.unit_anchor_us({"occurred_end_us": 22}) == 22

    def test_no_occurred(self):
        assert reltime.unit_anchor_us({}) is None
        assert reltime.unit_anchor_us(
            {"occurred_start_us": None, "occurred_end_us": None}
        ) is None


class TestResolveMentions:
    def test_tomorrow_day_interval(self):
        """K52-class: 'see you tomorrow' on a unit occurred 2023-12-19
        resolves [2023-12-20, 2023-12-21) at day precision."""
        rows = reltime.resolve_mentions("see you tomorrow", ANCHOR_1219)
        assert len(rows) == 1
        m = rows[0]
        assert (m["start_us"], m["end_us"]) == (D20, D21)
        assert m["precision"] == "day"
        assert m["ord"] == 0
        text = "see you tomorrow"
        assert text.encode()[m["span_start"]:m["span_end"]] == b"tomorrow"

    def test_yesterday(self):
        rows = reltime.resolve_mentions(
            "yesterday I adopted a puppy", _us(2023, 5, 8)
        )
        assert len(rows) == 1
        assert (rows[0]["start_us"], rows[0]["end_us"]) == (
            _us(2023, 5, 7), _us(2023, 5, 8))
        assert rows[0]["precision"] == "day"

    def test_last_summer_season(self):
        rows = reltime.resolve_mentions("last summer", ANCHOR_1219)
        assert len(rows) == 1
        m = rows[0]
        assert m["precision"] == "season"
        # northern meteorological summer 2023: Jun 1 – Sep 1
        assert (m["start_us"], m["end_us"]) == (
            _us(2023, 6, 1), _us(2023, 9, 1))

    def test_spec_list_forms(self):
        for text, prec in (
            ("last week", "week"),
            ("two days ago", "day"),
            ("next month", "month"),
            ("this weekend", "day"),
            ("on Friday", "day"),
        ):
            rows = reltime.resolve_mentions(text, ANCHOR_1219)
            assert len(rows) == 1, text
            assert rows[0]["precision"] == prec, text
            assert rows[0]["end_us"] > rows[0]["start_us"]

    def test_ambiguous_emits_nothing(self):
        """K53: 'later' is recognized but unresolvable → no row."""
        assert reltime.resolve_mentions("later", ANCHOR_1219) == []
        assert reltime.resolve_mentions(
            "sometime soon maybe", ANCHOR_1219) == []

    def test_explicit_only_excluded(self):
        """The grammar is the relative-expression one (§21.6): absolute
        expressions never needed the anchor and are not mentions —
        including the bare-year misanchor class."""
        assert reltime.resolve_mentions("on May 8, 2023", ANCHOR_1219) == []
        assert reltime.resolve_mentions("Cyberpunk 2077", ANCHOR_1219) == []
        assert reltime.resolve_mentions("in 2019", ANCHOR_1219) == []

    def test_mixed_relative_and_explicit(self):
        rows = reltime.resolve_mentions(
            "I moved on May 8, 2023 and saw her yesterday", ANCHOR_1219)
        assert len(rows) == 1
        m = rows[0]
        text = "I moved on May 8, 2023 and saw her yesterday"
        assert text.encode()[m["span_start"]:m["span_end"]] == b"yesterday"

    def test_multiple_mentions_ord(self):
        rows = reltime.resolve_mentions(
            "yesterday and the day before tomorrow", ANCHOR_1219)
        assert len(rows) == 2
        assert [m["ord"] for m in rows] == [0, 1]
        assert rows[0]["span_start"] < rows[1]["span_start"]

    def test_deterministic(self):
        a = reltime.resolve_mentions(
            "yesterday and last week and next month", ANCHOR_1219)
        b = reltime.resolve_mentions(
            "yesterday and last week and next month", ANCHOR_1219)
        assert a == b

    def test_empty_and_none_text(self):
        assert reltime.resolve_mentions("", ANCHOR_1219) == []
        assert reltime.resolve_mentions(None, ANCHOR_1219) == []

    def test_unsupported_locale_precision_first(self):
        # an undeclared locale marks every hit ambiguous → no rows
        assert reltime.resolve_mentions(
            "tomorrow", ANCHOR_1219, locale="fr-FR") == []
