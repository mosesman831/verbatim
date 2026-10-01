"""parse_temporal — deterministic temporal enrichment (V5-30.10–13).

Anchor for most tests is 2025-06-15T12:00:00Z — a Sunday.
"""

from __future__ import annotations

import pytest

from verbatim.enrichment import parse_temporal
from verbatim.enrichment.temporal import PARSER_VERSION

A = "2025-06-15T12:00:00Z"  # Sunday


class TestIsoDates:
    def test_iso_date_day(self):
        r = parse_temporal("deployed 2025-03-14", A)
        assert r.precision == "day"
        assert r.event_at == "2025-03-14"
        assert r.event_end == "2025-03-15"  # half-open interval
        assert r.expression == "2025-03-14"

    def test_iso_datetime_exact(self):
        r = parse_temporal("met on 2025-03-14T10:30:00Z", A)
        assert r.precision == "exact"
        assert r.event_at == "2025-03-14T10:30:00Z"
        assert r.event_end == r.event_at  # instant = zero-width

    def test_iso_datetime_offset_normalizes(self):
        r = parse_temporal("at 2025-03-14T10:30:00+05:00", A)
        assert r.event_at == "2025-03-14T05:30:00Z"

    def test_iso_naive_time_assumed_utc(self):
        # published convention: naive ISO times read as UTC
        r = parse_temporal("meeting 2025-03-14 10:30", A)
        assert r.precision == "exact"
        assert r.event_at == "2025-03-14T10:30:00Z"

    def test_iso_month(self):
        r = parse_temporal("text 2025-03 plain", A)
        assert r.precision == "month"
        assert (r.event_at, r.event_end) == ("2025-03-01", "2025-04-01")

    def test_invalid_date_rejected(self):
        r = parse_temporal("2025-13-40 bad date", A)
        assert r.precision == "unknown" and r.event_at == ""


class TestMonthYear:
    def test_month_year(self):
        r = parse_temporal("March 2025 release", A)
        assert r.precision == "month"
        assert (r.event_at, r.event_end) == ("2025-03-01", "2025-04-01")

    def test_month_day_year_both_orders(self):
        for t in ("moved March 3, 2025", "moved 3 March 2025"):
            r = parse_temporal(t, A)
            assert r.precision == "day"
            assert r.event_at == "2025-03-03"

    def test_month_day_no_year_nearest(self):
        # anchor June 15 — nearest March 3 is 2025-03-03 (past)
        r = parse_temporal("met March 3", A)
        assert r.event_at == "2025-03-03"
        assert r.status == "completed"  # "met" is a completed marker

    def test_month_day_leap_nearest(self):
        # 2025 has no Feb 29 — nearest valid is 2024
        r = parse_temporal("on February 29", A)
        assert r.event_at == "2024-02-29"

    def test_bare_month_needs_preposition(self):
        # "May" is also a modal/name — no preposition, no guess
        r = parse_temporal("May I help", A)
        assert r.precision == "unknown"
        r = parse_temporal("release in March", A)
        assert r.precision == "month"
        assert r.event_at == "2025-03-01"

    def test_in_year(self):
        r = parse_temporal("in 1999", A)
        assert r.precision == "year"
        assert (r.event_at, r.event_end) == ("1999-01-01", "2000-01-01")


class TestWeekdays:
    def test_last_weekday(self):
        r = parse_temporal("met last Monday", A)
        assert r.event_at == "2025-06-09"  # previous Monday
        assert r.status == "completed"

    def test_next_weekday(self):
        r = parse_temporal("deploy next Friday", A)
        assert r.event_at == "2025-06-20"
        assert r.status == "planned"

    def test_this_weekday(self):
        r = parse_temporal("meeting this Friday", A)
        assert r.event_at == "2025-06-13"  # Friday of the anchor's week

    def test_bare_weekday_past_marker(self):
        # "met" marks completed → most recent Monday, not next
        r = parse_temporal("met on Monday", A)
        assert r.event_at == "2025-06-09"
        assert r.status == "completed"

    def test_bare_weekday_nearest_default(self):
        # no aspect markers → unique nearest occurrence (±3 days)
        r = parse_temporal("on Monday", A)
        assert r.event_at == "2025-06-16"   # +1 day beats −6
        r = parse_temporal("on Thursday", A)
        assert r.event_at == "2025-06-12"   # −3 beats +4


class TestRelative:
    def test_yesterday_today_tomorrow(self):
        assert parse_temporal("shipped yesterday", A).event_at == \
            "2025-06-14"
        assert parse_temporal("today we met", A).event_at == "2025-06-15"
        r = parse_temporal("deploy tomorrow", A)
        assert r.event_at == "2025-06-16" and r.status == "planned"

    def test_last_this_next_week(self):
        r = parse_temporal("the outage last week", A)
        assert r.precision == "relative" and r.status == "completed"
        assert (r.event_at, r.event_end) == ("2025-06-02", "2025-06-09")
        r = parse_temporal("busy this week", A)
        assert (r.event_at, r.event_end) == ("2025-06-09", "2025-06-16")
        assert r.status == "ongoing"
        r = parse_temporal("review next week", A)
        assert r.status == "planned"
        assert (r.event_at, r.event_end) == ("2025-06-16", "2025-06-23")

    def test_month_year_relatives(self):
        r = parse_temporal("closed last month", A)
        assert r.precision == "month" and r.status == "completed"
        assert (r.event_at, r.event_end) == ("2025-05-01", "2025-06-01")
        r = parse_temporal("audit next year", A)
        assert r.precision == "year" and r.status == "planned"
        assert r.event_at == "2026-01-01"

    def test_n_units_ago(self):
        r = parse_temporal("3 days ago", A)
        assert r.event_at == "2025-06-12" and r.status == "completed"
        r = parse_temporal("a year ago", A)
        assert r.event_at == "2024-06-15"
        r = parse_temporal("3 months ago", A)
        assert r.event_at == "2025-03-15"  # calendar-month arithmetic

    def test_subday_units_relative(self):
        r = parse_temporal("2 hours ago", A)
        assert r.precision == "relative" and r.status == "completed"
        assert r.event_at == "2025-06-15T10:00:00Z"
        r = parse_temporal("in 30 minutes", A)
        assert r.event_at == "2025-06-15T12:30:00Z"
        assert r.status == "planned"

    def test_in_n_units(self):
        r = parse_temporal("we will deploy in 3 days", A)
        assert r.event_at == "2025-06-18" and r.status == "planned"


class TestSince:
    def test_since_month_ongoing_interval(self):
        # records BOTH the resolved interval and the anchor (V5-30.11)
        r = parse_temporal("since March", A)
        assert r.precision == "month" and r.status == "ongoing"
        assert r.event_at == "2025-03-01"     # most recent March
        assert r.event_end == A               # anchor closes the interval

    def test_since_iso_date(self):
        r = parse_temporal("running since 2025-03-01", A)
        assert r.status == "ongoing"
        assert r.event_at == "2025-03-01" and r.event_end == A

    def test_since_weekday(self):
        r = parse_temporal("since last Monday", A)
        assert r.status == "ongoing"
        assert r.event_at == "2025-06-09" and r.event_end == A


class TestAmbiguity:
    """Ambiguity resolves to unknown — never a guessed date (V5-30.10)."""

    def test_slash_date_ambiguous(self):
        # 03/04/2025 is valid as both MDY and DMY → unknown
        r = parse_temporal("03/04/2025 ambiguous", A)
        assert r.precision == "unknown"
        assert r.event_at == ""

    def test_slash_date_single_valid_reading(self):
        # only DD/MM can produce a valid date → resolved
        r = parse_temporal("25/12/2024 ok", A)
        assert r.event_at == "2024-12-25"

    def test_bounded_month_not_invented(self):
        r = parse_temporal("until March", A)
        assert r.precision == "unknown" and r.event_at == ""

    def test_no_expression(self):
        r = parse_temporal("no date here", A)
        assert r.precision == "unknown" and r.status == "unknown"
        assert r.expression == "" and r.start == -1

    def test_conflicting_markers_unknown(self):
        # "was" (completed) + "planned" (planned) in one sentence —
        # conflicting aspect ⇒ unknown, not a coin flip
        r = parse_temporal("was planned for March 3", A)
        assert r.status == "unknown"

    def test_future_completed_unknown(self):
        # a wholly-future interval can never be completed
        r = parse_temporal("was completed on 2999-01-01", A)
        assert r.status == "unknown"


class TestContract:
    def test_anchor_recorded(self):
        r = parse_temporal("met yesterday", A)
        assert r.anchor_at == "2025-06-15T12:00:00Z"

    def test_anchor_offset_canonicalized(self):
        r = parse_temporal("met on Monday", "2025-06-15T12:00:00+05:00")
        assert r.anchor_at == "2025-06-15T07:00:00Z"

    def test_anchor_change_recomputes(self):
        # changing the anchor changes the resolution — the enrichment is
        # a derived view, invalidated+recomputed, never edited (V5-30.11)
        r1 = parse_temporal("met on Monday", "2025-06-15T12:00:00Z")
        r2 = parse_temporal("met on Monday", "2025-06-18T12:00:00Z")
        assert r1.event_at == "2025-06-09"
        assert r2.event_at == "2025-06-16"
        assert r1.anchor_at != r2.anchor_at

    def test_anchor_validation(self):
        for bad in ("", "not-a-date", "2025-06-15", "2025-06-15T12:00"):
            with pytest.raises(ValueError):
                parse_temporal("x", bad)

    def test_status_from_sentence_markers(self):
        # "deployed" is a completed aspect marker (V5-30.12)
        assert parse_temporal("deployed 2025-03-14", A).status == \
            "completed"
        # no markers → unknown
        assert parse_temporal("release March 2025", A).status == \
            "unknown"

    def test_byte_offsets(self):
        text = "café met on 2025-03-14"   # é = 2 UTF-8 bytes
        r = parse_temporal(text, A)
        assert text.encode("utf-8")[r.start:r.end] == b"2025-03-14"

    def test_best_precision_wins(self):
        # day precision beats the earlier relative-precision match
        r = parse_temporal("met last week and shipped 2025-03-14", A)
        assert r.expression == "2025-03-14"
        assert r.precision == "day"

    def test_earliest_on_precision_tie(self):
        r = parse_temporal("met Monday and deployed Tuesday", A)
        assert r.expression == "Monday"

    def test_parser_version_and_producer(self):
        r = parse_temporal("met yesterday", A)
        assert r.parser == PARSER_VERSION == "temporal/v1"
        assert r.producer == "enrich/v1"

    def test_determinism(self):
        text = "met last Monday since March at café ☕"
        r1 = parse_temporal(text, A)
        r2 = parse_temporal(text, A)
        assert r1 == r2
