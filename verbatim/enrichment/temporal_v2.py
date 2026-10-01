"""Deterministic temporal resolution — ``temporal/v2``.

SPEC_V7 §32.9 rules T01–T30 (V7-09.03–05). ``resolve`` scans a text and
returns every non-overlapping temporal expression as a ``ResolvedTime``
carrying a half-open ``IntervalUs`` ``[start_us, end_us)`` in UTC
microseconds; ``resolve_query_window`` folds a query's time expressions
into one covering ``IntervalUs`` for the temporal lane.

Published contract notes (V7-09.04 — honest bounds):

- **Locale.** English month/weekday names and the Gregorian calendar
  only. ``locale`` pins the slash-date order: ``en-US`` (the default)
  reads ``M/D/YYYY``; the day-first English locales in
  ``DAY_FIRST_LOCALES`` read ``D/M/YYYY``. When both readings of a slash
  date are valid, the pinned order wins and ``ambiguous_locale=True`` is
  recorded (§32.9). A locale outside ``SUPPORTED_LOCALES`` is *not*
  declared supported: the English rule set still runs, slash dates fall
  back to M/D order, and every emitted ``ResolvedTime`` carries
  ``ambiguous_locale=True`` so downstream coverage sees the guess.
- **Hemisphere.** ``"north"`` (default) or ``"south"`` selects the
  meteorological season calendar (T18); any other value behaves as
  ``"north"``.
- **Anchor.** ``anchor_us`` is epoch microseconds; all arithmetic is on
  the anchor's UTC calendar. No wall clock is read inside ``resolve`` —
  identical inputs give identical outputs (V7-09.04 determinism).
- **Intervals.** Half-open ``[start_us, end_us)``; ``start == end``
  marks an instant (T01 datetimes). Open bounds use ``None`` (``until
  X`` → ``[None, X.start)``) or the anchor itself (``since X`` →
  ``[X.start, anchor_us]``).
- **Kept unknowns.** Durations (T26), age expressions needing a birth
  fact (T27), and recognized-but-unresolvable phrases (T30 — including
  date-shaped strings that fail validation) are emitted with
  ``start_us = end_us = None`` and ``precision=unknown``. Never dropped.

Rule map (§32.9): T01 ISO dates/datetimes; T02 named/slash dates; T03
``Month YYYY``; T04 ``YYYY`` and ``1990s`` decades; T05 bare ``Month``;
T06–T08 today/yesterday/tomorrow family; T09–T12 ``N units ago|later``
/ ``in N units`` (day / week ±3d / calendar month / calendar year);
T13–T16 last|this|next + ISO week / weekend / month / year; T17 weekday
names; T18 meteorological seasons; T19 ``a few … ago``; T20 ``a couple
of … ago``; T21 recently/lately (30-day window, month precision, low
confidence); T22 ``[N] day|week|month|year(s) before|after <expr>``;
T23 early/mid/late; T24 ordinal weeks of a month; T25 since/until/
before/after/through + ``from X to Y`` / ``between X and Y``; T26
``for N units`` durations (marked, not resolved); T27 age expressions
(unknown — no birth-fact input exists); T28 holidays (fixed dates, US
Thanksgiving = 4th Thursday of November, Easter = Gregorian computus);
T29 ``the other day|night``; T30 kept-unknown vague phrases.

Resolution conventions where the spec leaves room (documented per
V7-09.04's determinism requirement):

- Yearless month / month-day / holiday forms resolve to the most recent
  occurrence on-or-before the anchor — the "nearest past" reading of
  T05/T28 and V7-09.03 — except inside open-bound slots where direction
  is forced by the bound word (``until X`` → the next occurrence,
  ``since X`` → the latest one ≤ A).
- Weekdays: ``last``/bare/``on``/``this past`` pick the most recent such
  weekday strictly before A; ``next``/``this coming`` the first strictly
  after; ``this`` the occurrence inside A's ISO week (A itself when they
  coincide — the spec's "(or on A for 'this')").
- Seasons: ``last`` picks the latest season fully ended before A (the
  previous one when A is inside it), ``next`` the first starting after
  A, ``this`` the containing else nearest occurrence (ties past-ward),
  bare the containing else most recent past one.
- ``this weekend`` is the Saturday–Sunday of A's current weekend
  (containing it when A is Sat/Sun); ``last``/``next`` shift by a week.
- T19/T20 generalize over units at the same relative spans (a few ≈
  2–5 units, a couple ≈ spec's 10–21 days for weeks, ±1 otherwise).
- T23 thirds are day-granular inside a month (1–10 / 11–20 / 21–end)
  and month-granular inside a year (Jan–Apr / May–Aug / Sep–Dec).
- T24 weeks-of-month are 7-day chunks from the 1st; ``last`` is the
  final 7 days ending at month end.
- ``the last|past|next|coming N units`` produce the N-unit window
  ending at end-of-anchor-day (or starting at the anchor date).
- Sub-day units (hours/minutes) are not resolved — v7 windows are
  day-granular; such phrases simply don't match.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Tuple, TYPE_CHECKING

from verbatim.core.types_v7 import (
    IntervalUs,
    OccurredPrecision,
    OccurredSource,
    ResolvedTime,
)

from .normalize import utf8_offsets

if TYPE_CHECKING:  # typing only — never imported at runtime (contract)
    from verbatim.core.types_v7 import NormAnalysis


RESOLVER_ID = "temporal/v2"
RULE_SET_STATUS = "provisional/v7-r0"  # all §32 constants (V7-32.01)
PARSER_LOCALE = "en"
PARSER_CALENDAR = "gregorian"

#: Locales `temporal/v2` declares supported. English rules only; the set
#: exists so callers can distinguish "resolved under a declared locale"
#: from "ran English rules anyway" (ambiguous_locale=True on every hit).
DAY_FIRST_LOCALES = frozenset({
    "en-gb", "en-au", "en-nz", "en-ie", "en-za", "en-in",
    "en-hk", "en-sg", "en-ca",
})
SUPPORTED_LOCALES = frozenset({"en-us"} | DAY_FIRST_LOCALES)

_HEMISPHERES = frozenset({"north", "south"})


def locale_supported(locale: str) -> bool:
    """Whether ``locale`` is a declared-supported English locale."""
    return str(locale or "").strip().lower() in SUPPORTED_LOCALES


# ---------------------------------------------------------------------------
# us <-> datetime (integer arithmetic only — no float round-trip)
# ---------------------------------------------------------------------------

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_DAY_US = 86_400_000_000


def _to_dt(us: int) -> datetime:
    return _EPOCH + timedelta(microseconds=int(us))


def _to_us(dt: datetime) -> int:
    d = dt - _EPOCH
    return d.days * _DAY_US + d.seconds * 1_000_000 + d.microseconds


def _dus(d: date) -> int:
    """µs of the UTC midnight starting ``d``."""
    return _to_us(datetime(d.year, d.month, d.day, tzinfo=timezone.utc))


def _valid(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _add_months(d: date, months: int) -> date:
    total = d.year * 12 + (d.month - 1) + months
    y, m = divmod(total, 12)
    m += 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def _add_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # Feb 29 -> Feb 28
        return d.replace(year=d.year + years, day=28)


def _month_bounds(y: int, m: int) -> Tuple[date, date]:
    start = date(y, m, 1)
    return start, _add_months(start, 1)


_PREC_RANK = {
    OccurredPrecision.INSTANT: 0,
    OccurredPrecision.DAY: 1,
    OccurredPrecision.WEEK: 2,
    OccurredPrecision.MONTH: 3,
    OccurredPrecision.SEASON: 4,
    OccurredPrecision.YEAR: 5,
    OccurredPrecision.DECADE: 6,
    OccurredPrecision.UNKNOWN: 7,
}

_P = OccurredPrecision
_S = OccurredSource


# ---------------------------------------------------------------------------
# lexicon (carried from temporal/v1 — proven tables)
# ---------------------------------------------------------------------------

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3,
    "mar": 3, "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
    "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9,
    "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11,
    "nov": 11, "december": 12, "dec": 12,
}
_MONTH_RE = (
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|"
    r"Oct|Nov|Dec)"
)
_WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thur": 3,
    "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}
_WD_RE = (
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
    r"Mon|Tues|Tue|Wed|Thurs|Thur|Thu|Fri|Sat|Sun)"
)
_NUMWORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}
_NUM_RE = (
    r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve)"
)


def _num(tok: str) -> int:
    tok = tok.lower()
    if tok.isdigit():
        return int(tok)
    return _NUMWORDS[tok]


#: Temporal lead-ins that license a lowercase ambiguous token ("may",
#: "march", short abbrevs, "fall", "spring", "sat"…). One filler word
#: (article/possessive) is allowed between the lead-in and the token so
#: "in the spring" licenses "spring"; "the" alone does not — "the sun"
#: must not license "sun" as Sunday nor "the march" "march" as a month.
_PREP_TAIL_RE = re.compile(
    r"(?:in|of|during|since|until|till|til|from|before|after|through|"
    r"last|this|next|for|over|around|about|back|early|mid|late|"
    r"per|every|each)\s+(?:(?:the|a|an|my|our|your|his|her|their)\s+)?$",
    re.IGNORECASE,
)


def _guarded_ok(text: str, cs: int, surface: str) -> bool:
    """Gate for lowercase surface forms that double as common words.

    Full month/weekday names are safe lowercase except ``may`` and
    ``march`` (modal/verb); abbreviations (<= 4 chars) and the seasons
    ``fall``/``spring`` need capitalization or a temporal lead-in.
    """
    if not surface.islower():
        return True
    low = surface.lower()
    if low in _MONTHS and len(low) > 4 and low not in ("may", "march"):
        return True
    if low in _WEEKDAYS and len(low) > 4:
        return True
    if low in ("summer", "winter", "autumn"):
        return True
    # ambiguous lowercase forms need a temporal lead-in
    return bool(_PREP_TAIL_RE.search(text[:cs]))


# ---------------------------------------------------------------------------
# holidays (T28) — fixed dates, US Thanksgiving, Gregorian computus
# ---------------------------------------------------------------------------

def _easter(year: int) -> date:
    """Gregorian Easter Sunday (Anonymous computus), stdlib only."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _thanksgiving(year: int) -> date:
    """US Thanksgiving = fourth Thursday of November."""
    first_thu = 1 + (3 - date(year, 11, 1).weekday()) % 7
    return date(year, 11, first_thu + 21)


_HOLIDAYS = {
    "christmas": (12, 25), "christmas day": (12, 25),
    "christmas eve": (12, 24), "xmas": (12, 25), "boxing day": (12, 26),
    "new year": (1, 1), "new year's": (1, 1), "new years": (1, 1),
    "new year's day": (1, 1), "new years day": (1, 1),
    "new year's eve": (12, 31), "new years eve": (12, 31),
    "halloween": (10, 31),
    "valentine's day": (2, 14), "valentines day": (2, 14),
    "st valentine's day": (2, 14), "st. valentine's day": (2, 14),
    "st patrick's day": (3, 17), "st. patrick's day": (3, 17),
    "saint patrick's day": (3, 17),
    "independence day": (7, 4), "fourth of july": (7, 4),
    "4th of july": (7, 4),
    "thanksgiving": "thanksgiving", "thanksgiving day": "thanksgiving",
    "easter": "easter", "easter sunday": "easter", "easter day": "easter",
}


def _holiday_date(key: str, year: int) -> Optional[date]:
    v = _HOLIDAYS[key]
    if v == "easter":
        return _easter(year)
    if v == "thanksgiving":
        return _thanksgiving(year)
    m, d = v
    return _valid(year, m, d)


def _holiday_pattern() -> str:
    parts = []
    for name in sorted(_HOLIDAYS, key=len, reverse=True):
        words = [re.escape(w).replace("'", r"[’']")
                 for w in name.split()]
        parts.append(r"\s+".join(words))
    return "(?:" + "|".join(parts) + ")"


_HOLIDAY_RE = _holiday_pattern()


def _holiday_key(surface: str) -> str:
    s = surface.lower().replace("’", "'")
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# seasons (T18) — meteorological, hemisphere-selectable
# ---------------------------------------------------------------------------

_SEASON_START_N = {"spring": 3, "summer": 6, "autumn": 9, "winter": 12}
_SEASON_ALIASES = {"fall": "autumn"}


def _season_bounds(season: str, year: int, hemi: str) -> Tuple[date, date]:
    """[start, end) of a meteorological season labelled by start year."""
    sm = _SEASON_START_N[season]
    if hemi == "south":
        sm = (sm + 5) % 12 + 1
    start = date(year, sm, 1)
    em, ey = sm + 3, year
    while em > 12:
        em -= 12
        ey += 1
    return start, date(ey, em, 1)


def _season_pick(season: str, direction: str, adate: date,
                 hemi: str, year: Optional[int]) -> Optional[Tuple[date, date]]:
    if year is not None:
        return _season_bounds(season, year, hemi)
    # A 3-month season ending just before A may have started up to ~15
    # months earlier — cross-year seasons (Dec-start: north winter,
    # south summer) need start year A-2 in the window or the most
    # recent *ended* occurrence is invisible while one is in progress.
    occs = [_season_bounds(season, y, hemi)
            for y in range(adate.year - 2, adate.year + 3)]
    containing = [o for o in occs if o[0] <= adate < o[1]]
    ended = [o for o in occs if o[1] <= adate]
    future = [o for o in occs if o[0] > adate]
    if direction == "last":
        return max(ended) if ended else None
    if direction == "next":
        return min(future) if future else None
    if direction == "this":
        if containing:
            return containing[0]
        cand = None  # nearest occurrence, ties past-ward
        for o in ended + future:
            dist = min(abs((o[0] - adate).days), abs((o[1] - adate).days))
            key = (dist, 0 if o[1] <= adate else 1)
            if cand is None or key < cand[0]:
                cand = (key, o)
        return cand[1] if cand else None
    # bare / in / during — containing else most recent past
    if containing:
        return containing[0]
    if ended:
        return max(ended)
    return min(future) if future else None


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------

class _Cand:
    """One resolved (or kept-unknown) expression, char offsets."""

    __slots__ = ("cs", "ce", "s", "e", "prec", "src", "rule", "amb", "prio")

    def __init__(self, cs, ce, s, e, prec, src, rule, amb=False, prio=20):
        self.cs, self.ce = cs, ce
        self.s, self.e = s, e  # µs ints or None (open bounds)
        self.prec, self.src, self.rule = prec, src, rule
        self.amb, self.prio = amb, prio


def _month_le(adate: date, m: int) -> Tuple[int, int]:
    """Most recent month-name occurrence with start on-or-before A."""
    return (adate.year, m) if m <= adate.month else (adate.year - 1, m)


def _month_lt(adate: date, m: int) -> Tuple[int, int]:
    """Most recent month-name occurrence with start strictly before A."""
    if date(adate.year, m, 1) < adate:
        return (adate.year, m)
    return (adate.year - 1, m)


def _month_ge(adate: date, m: int) -> Tuple[int, int]:
    """First month-name occurrence with start on-or-after A."""
    return (adate.year, m) if date(adate.year, m, 1) >= adate \
        else (adate.year + 1, m)


def _md_le(adate: date, m: int, d: int) -> Optional[date]:
    cand = _valid(adate.year, m, d)
    if cand is not None and cand <= adate:
        return cand
    return _valid(adate.year - 1, m, d)


def _md_lt(adate: date, m: int, d: int) -> Optional[date]:
    cand = _valid(adate.year, m, d)
    if cand is not None and cand < adate:
        return cand
    return _valid(adate.year - 1, m, d)


def _md_ge(adate: date, m: int, d: int) -> Optional[date]:
    cand = _valid(adate.year, m, d)
    if cand is not None and cand >= adate:
        return cand
    return _valid(adate.year + 1, m, d)


def _weekend_sat(adate: date) -> date:
    """Saturday of the anchor's current weekend (containing-or-upcoming)."""
    return adate + timedelta(days=5 - adate.weekday())


# ---------------------------------------------------------------------------
# embedded-expression parser (T22 "the week before X", T25 bounds)
# ---------------------------------------------------------------------------

class _Emb:
    """A parsed embedded expression. ``roll`` marks inferrable forms the
    ``from X to Y`` second endpoint may roll forward when it lands before
    the first endpoint ("year" +1y, "month" +1mo, "week" +7d)."""

    __slots__ = ("s", "e", "prec", "end", "roll")

    def __init__(self, s, e, prec, end, roll=None):
        self.s, self.e, self.prec, self.end, self.roll = \
            s, e, prec, end, roll


def _bump(emb: _Emb) -> _Emb:
    s = e = None
    if emb.roll == "year":
        s = _add_years(emb.s, 1)
        e = _add_years(emb.e, 1)
    elif emb.roll == "month":
        s = _add_months(emb.s, 1)
        e = _add_months(emb.e, 1)
    elif emb.roll == "week":
        s = emb.s + timedelta(weeks=1)
        e = emb.e + timedelta(weeks=1)
    if s is None:
        return emb
    return _Emb(s, e, emb.prec, emb.end, emb.roll)


_EMB_ISO = re.compile(r"\d{4}-\d{2}-\d{2}(?![-\d])")
_EMB_MDY = re.compile(
    rf"(?P<mn>{_MONTH_RE})\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?"
    rf"(?:\s*,?\s*(?P<y>(?:19|20)\d{{2}}))?(?!\d)",
    re.IGNORECASE)
_EMB_DMY = re.compile(
    rf"(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<mn>{_MONTH_RE})"
    rf"(?:\s*,?\s*(?P<y>(?:19|20)\d{{2}}))?",
    re.IGNORECASE)
_EMB_MY = re.compile(
    rf"(?P<mn>{_MONTH_RE})\s+(?:of\s+)?(?P<y>(?:19|20)\d{{2}})\b",
    re.IGNORECASE)
_EMB_REL = re.compile(
    r"(?P<w>yesterday|today|tomorrow|tonight|last\s+night|now)\b",
    re.IGNORECASE)
_EMB_LX = re.compile(
    rf"(?P<dir>last|this|next)\s+"
    rf"(?P<u>week|weekend|month|year|{_WD_RE})\b",
    re.IGNORECASE)
_EMB_HOL = re.compile(
    rf"(?:(?P<dir>last|this|next)\s+)?(?:the\s+)?(?P<h>{_HOLIDAY_RE})"
    rf"(?:\s+(?P<y>(?:19|20)\d{{2}}))?\b",
    re.IGNORECASE)
_EMB_MON = re.compile(rf"(?P<mn>{_MONTH_RE})\b", re.IGNORECASE)
_EMB_Y = re.compile(r"(?P<y>(?:19|20)\d{2})\b")
_EMB_WD = re.compile(rf"(?P<wd>{_WD_RE})\b", re.IGNORECASE)
_EMB_BAREDAY = re.compile(r"(?P<d>\d{1,2})(?:st|nd|rd|th)?(?!\d)")
_EMB_AGO = re.compile(
    rf"(?P<n>{_NUM_RE})\s+(?P<u>days?|weeks?|months?|years?)\s+ago\b",
    re.IGNORECASE)


def _hol_pick(key: str, dirw: str, adate: date, prefer: str) -> Optional[date]:
    """Holiday occurrence under a direction/preference regime."""
    if dirw == "next":  # first occurrence strictly after A
        d = _holiday_date(key, adate.year)
        if d is not None and d <= adate:
            d = _holiday_date(key, adate.year + 1)
        return d
    if dirw == "this":  # containing else next
        d = _holiday_date(key, adate.year)
        if d is not None and d < adate:
            d = _holiday_date(key, adate.year + 1)
        return d
    if dirw == "last":  # most recent strictly before A
        d = _holiday_date(key, adate.year)
        if d is not None and d >= adate:
            d = _holiday_date(key, adate.year - 1)
        return d
    if prefer == "future":
        d = _holiday_date(key, adate.year)
        if d is not None and d < adate:
            d = _holiday_date(key, adate.year + 1)
        return d
    if prefer == "past":
        d = _holiday_date(key, adate.year)
        if d is not None and d >= adate:
            d = _holiday_date(key, adate.year - 1)
        return d
    # bare, prefer past_le — most recent on-or-before A
    d = _holiday_date(key, adate.year)
    if d is not None and d > adate:
        d = _holiday_date(key, adate.year - 1)
    return d


def _lx_block(dirw: str, unit: str, adate: date) -> Optional[_Emb]:
    """last/this/next + week|weekend|month|year|weekday (embedded)."""
    dirw = dirw.lower()
    unit = unit.lower()
    if unit == "week":
        mon = adate - timedelta(days=adate.weekday())
        s = mon + timedelta(weeks={"last": -1, "this": 0, "next": 1}[dirw])
        return _Emb(s, s + timedelta(days=7), _P.WEEK, 0)
    if unit == "weekend":
        s = _weekend_sat(adate) + timedelta(
            weeks={"last": -1, "this": 0, "next": 1}[dirw])
        return _Emb(s, s + timedelta(days=2), _P.DAY, 0)
    if unit == "month":
        s = _add_months(date(adate.year, adate.month, 1),
                        {"last": -1, "this": 0, "next": 1}[dirw])
        return _Emb(s, _add_months(s, 1), _P.MONTH, 0)
    if unit == "year":
        y = adate.year + {"last": -1, "this": 0, "next": 1}[dirw]
        return _Emb(date(y, 1, 1), date(y + 1, 1, 1), _P.YEAR, 0)
    if unit in _WEEKDAYS:
        twd, awd = _WEEKDAYS[unit], adate.weekday()
        if dirw == "this":
            delta = twd - awd
        elif dirw == "next":
            delta = (twd - awd) % 7 or 7
        else:
            delta = -((awd - twd) % 7 or 7)
        d = adate + timedelta(days=delta)
        return _Emb(d, d + timedelta(days=1), _P.DAY, 0)
    return None


def _embedded(text: str, pos: int, adate: date,
              prefer: str = "past_le") -> Optional[_Emb]:
    """Longest parseable date-ish expression starting at ``pos``.

    ``prefer`` steers yearless/relative forms: ``past`` = strictly before
    A, ``past_le`` = on-or-before A (default — "nearest past"), ``future``
    = on-or-after A (for until/before/after/through bounds).
    """
    best: Optional[_Emb] = None

    def keep(e: Optional[_Emb]):
        nonlocal best
        if e is not None and (best is None or e.end > best.end):
            best = e

    m = _EMB_ISO.match(text, pos)
    if m:
        try:
            d = date.fromisoformat(m.group(0))
        except ValueError:
            d = None
        if d:
            keep(_Emb(d, d + timedelta(days=1), _P.DAY, m.end()))

    m = _EMB_MDY.match(text, pos)
    if m:
        mn, dy = _MONTHS[m["mn"].lower()], int(m["d"])
        yr = int(m["y"]) if m["y"] else None
        if yr:
            d = _valid(yr, mn, dy)
            roll = None
        elif prefer == "future":
            d, roll = _md_ge(adate, mn, dy), "year"
        elif prefer == "past":
            d, roll = _md_lt(adate, mn, dy), "year"
        else:
            d, roll = _md_le(adate, mn, dy), "year"
        if d:
            keep(_Emb(d, d + timedelta(days=1), _P.DAY, m.end(), roll))

    m = _EMB_DMY.match(text, pos)
    if m:
        mn, dy = _MONTHS[m["mn"].lower()], int(m["d"])
        yr = int(m["y"]) if m["y"] else None
        if dy <= 31:
            if yr:
                d = _valid(yr, mn, dy)
                roll = None
            elif prefer == "future":
                d, roll = _md_ge(adate, mn, dy), "year"
            elif prefer == "past":
                d, roll = _md_lt(adate, mn, dy), "year"
            else:
                d, roll = _md_le(adate, mn, dy), "year"
            if d:
                keep(_Emb(d, d + timedelta(days=1), _P.DAY,
                          m.end(), roll))

    m = _EMB_MY.match(text, pos)
    if m:
        s, e = _month_bounds(int(m["y"]), _MONTHS[m["mn"].lower()])
        keep(_Emb(s, e, _P.MONTH, m.end()))

    m = _EMB_REL.match(text, pos)
    if m:
        w = re.sub(r"\s+", " ", m["w"].lower())
        if w in ("yesterday", "last night"):
            d = adate - timedelta(days=1)
        elif w == "tomorrow":
            d = adate + timedelta(days=1)
        else:  # today / tonight / now
            d = adate
        keep(_Emb(d, d + timedelta(days=1), _P.DAY, m.end()))

    m = _EMB_LX.match(text, pos)
    if m:
        e = _lx_block(m["dir"], m["u"], adate)
        if e:
            e.end = m.end()
            keep(e)

    m = _EMB_HOL.match(text, pos)
    if m:
        key = _holiday_key(m["h"])
        if key in _HOLIDAYS:
            yr = int(m["y"]) if m["y"] else None
            if yr:
                d = _holiday_date(key, yr)
                roll = None
            else:
                d = _hol_pick(key, (m["dir"] or "").lower(), adate,
                              prefer)
                roll = "year"
            if d:
                keep(_Emb(d, d + timedelta(days=1), _P.DAY,
                          m.end(), roll))

    m = _EMB_MON.match(text, pos)
    if m and _guarded_ok(text, pos, m["mn"]):
        mo = _MONTHS[m["mn"].lower()]
        if prefer == "future":
            y, mo = _month_ge(adate, mo)
        elif prefer == "past":
            y, mo = _month_lt(adate, mo)
        else:
            y, mo = _month_le(adate, mo)
        s, e = _month_bounds(y, mo)
        keep(_Emb(s, e, _P.MONTH, m.end(), "year"))

    m = _EMB_Y.match(text, pos)
    if m:
        y = int(m["y"])
        keep(_Emb(date(y, 1, 1), date(y + 1, 1, 1), _P.YEAR, m.end()))

    m = _EMB_WD.match(text, pos)
    if m:
        twd, awd = _WEEKDAYS[m["wd"].lower()], adate.weekday()
        if prefer == "future":
            delta = (twd - awd) % 7
        elif prefer == "past":
            delta = -((awd - twd) % 7 or 7)
        else:
            delta = -((awd - twd) % 7)
        d = adate + timedelta(days=delta)
        keep(_Emb(d, d + timedelta(days=1), _P.DAY, m.end(), "week"))

    m = _EMB_AGO.match(text, pos)
    if m:
        n, unit = _num(m["n"]), m["u"].lower().rstrip("s")
        c = _shift_cand(0, 0, adate, -n, unit, "")
        if c is not None and c.s is not None:
            s = _to_dt(c.s).date()
            e = _to_dt(c.e).date()
            keep(_Emb(s, e, c.prec, m.end()))

    return best


def _bare_day(text: str, pos: int, e1: _Emb) -> Optional[_Emb]:
    """``from March 5 to 10`` — bare day-of-month second endpoint."""
    if e1.prec != _P.DAY:
        return None
    m = _EMB_BAREDAY.match(text, pos)
    if not m:
        return None
    d = _valid(e1.s.year, e1.s.month, int(m["d"]))
    if d is None:
        return None
    return _Emb(d, d + timedelta(days=1), _P.DAY, m.end(), "month")


# ---------------------------------------------------------------------------
# patterns
# ---------------------------------------------------------------------------

_ISO_DT_RE = re.compile(
    r"\b(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})"
    r"(?:[Tt ](?P<h>\d{2}):(?P<mi>\d{2})"
    r"(?::(?P<s>\d{2})(?:\.(?P<us>\d{1,6}))?)?"
    r"\s*(?P<tz>[Zz]|[+-]\d{2}:?\d{2})?)?\b"
)
_SLASH_RE = re.compile(
    r"\b(?P<a>\d{1,2})(?P<sep>[/.\-])(?P<b>\d{1,2})(?P=sep)(?P<y>\d{2,4})\b"
)
_MDY_RE = re.compile(
    rf"\b(?P<mn>{_MONTH_RE})\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?"
    rf"(?:\s*,?\s*(?P<y>(?:19|20)\d{{2}}))?(?!\d)\b",
    re.IGNORECASE,
)
_DMY_RE = re.compile(
    rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?"
    rf"(?P<mn>{_MONTH_RE})(?:\s*,?\s*(?P<y>(?:19|20)\d{{2}}))?\b",
    re.IGNORECASE,
)
_MY_RE = re.compile(
    rf"\b(?P<mn>{_MONTH_RE})\s+(?:of\s+)?(?P<y>(?:19|20)\d{{2}})\b",
    re.IGNORECASE,
)
_BARE_MONTH_RE = re.compile(rf"\b(?P<mn>{_MONTH_RE})\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?P<y>(?:19|20)\d{2})\b")
_DECADE_RE = re.compile(r"\b(?P<y>(?:19|20)\d)0s\b")
_DAYWORD_RE = re.compile(
    r"\b(?P<w>today|tonight|"
    r"this\s+morning|this\s+afternoon|this\s+evening|"
    r"yesterday(?:\s+morning|\s+afternoon|\s+evening)?|last\s+night|"
    r"tomorrow(?:\s+morning|\s+afternoon|\s+evening|\s+night)?|"
    r"right\s+now|at\s+the\s+moment|as\s+of\s+now|currently|now)\b",
    re.IGNORECASE,
)
_BLOCK_REL_RE = re.compile(
    r"\b(?P<dir>last|this|next)\s+(?P<u>week|weekend|month|year)\b",
    re.IGNORECASE,
)
_NREL_RE = re.compile(
    rf"\b(?P<n>{_NUM_RE})\s+"
    rf"(?P<u>days?|weeks?|months?|years?)\s+"
    rf"(?P<dir>ago|earlier|back|later|prior)\b",
    re.IGNORECASE,
)
_IN_N_RE = re.compile(
    rf"\bin\s+(?P<n>{_NUM_RE})\s+"
    rf"(?P<u>days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
_SPAN_N_RE = re.compile(
    rf"\b(?:the\s+)?(?P<dir>past|last|next|coming)\s+"
    rf"(?P<n>{_NUM_RE})\s+(?P<u>days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
_WEEKDAY_RE = re.compile(
    rf"\b(?:(?P<dir>last|next|this|on|this\s+past|this\s+coming)\s+)?"
    rf"(?P<wd>{_WD_RE})\b",
    re.IGNORECASE,
)
_SEASON_RE = re.compile(
    r"\b(?:(?P<dir>last|this|next)\s+)?"
    r"(?P<s>spring|summer|autumn|fall|winter)"
    r"(?:\s+(?P<y>(?:19|20)\d{2}))?\b",
    re.IGNORECASE,
)
_FEW_RE = re.compile(
    rf"\b(?P<q>a\s+few|a\s+couple(?:\s+of)?|several)\s+"
    rf"(?P<u>days?|weeks?|months?|years?)\s+(?:ago|back)\b",
    re.IGNORECASE,
)
_FEW_IN_RE = re.compile(
    rf"\bin\s+(?P<q>a\s+few|a\s+couple(?:\s+of)?)\s+"
    rf"(?P<u>days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
_RECENT_RE = re.compile(
    r"\b(?P<w>recently|lately|of\s+late|as\s+of\s+late)\b",
    re.IGNORECASE,
)
_OTHER_RE = re.compile(r"\bthe\s+other\s+(?:day|night)\b", re.IGNORECASE)
_T22_RE = re.compile(
    rf"\b(?:(?P<q>the|a)\s+|(?P<n>{_NUM_RE})\s+)?"
    rf"(?P<u>days?|weeks?|months?|years?)\s+"
    rf"(?P<dir>before|after|prior\s+to)\s+",
    re.IGNORECASE,
)
_T23_RE = re.compile(
    rf"\b(?P<part>early|mid(?:dle)?|late)[\s-]+"
    rf"(?:(?:in|of)\s+)?"
    rf"(?:(?P<mn>{_MONTH_RE})(?:\s*,?\s*(?P<y>(?:19|20)\d{{2}}))?|"
    rf"(?P<y2>(?:19|20)\d{{2}}))\b",
    re.IGNORECASE,
)
_T24_RE = re.compile(
    rf"\b(?:the\s+)?(?P<ord>first|second|third|fourth|fifth|last|final|"
    rf"1st|2nd|3rd|4th|5th)\s+week\s+(?:of|in)\s+"
    rf"(?P<mn>{_MONTH_RE})(?:\s*,?\s*(?P<y>(?:19|20)\d{{2}}))?\b",
    re.IGNORECASE,
)
_T25_FROM_RE = re.compile(r"\bfrom\s+", re.IGNORECASE)
_T25_BETWEEN_RE = re.compile(r"\bbetween\s+", re.IGNORECASE)
_T25_BOUND_RE = re.compile(
    r"\b(?P<b>since|until|till|til|before|after|through)\s+",
    re.IGNORECASE,
)
_T26_RE = re.compile(
    rf"\bfor\s+(?!the\s+(?:past|last)\b)"
    rf"(?:(?:about|around|roughly|approximately|nearly|almost|over|"
    rf"more\s+than|under|close\s+to)\s+)?"
    rf"(?:{_NUM_RE}|a\s+few|a\s+couple(?:\s+of)?|several)\s+"
    rf"(?:minutes?|hours?|days?|weeks?|fortnights?|months?|years?|"
    rf"decades?)\b"
    rf"|\bfor\s+(?:a\s+)?(?:while|ages|years|months|weeks|days|hours|"
    rf"minutes|decades)\b",
    re.IGNORECASE,
)
_AGE_RES = [
    re.compile(
        rf"\bwhen\s+(?:i|we)\s+(?:was|were)\s+"
        rf"(?:about\s+|around\s+|only\s+|just\s+)?"
        # ``a kid``/``a teenager`` must outrank the bare ``a`` inside
        # _NUM_RE or the match truncates at the article; bare plurals
        # cover "when we were kids".
        rf"(?:\d{{1,3}}|(?:a|an)\s+(?:kid|child|baby|teenager|teens?|"
        rf"students?)|kids|children|babies|teens|teenagers|students|"
        rf"{_NUM_RE}|younger|young|little|small|born|growing\s+up)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bat\s+(?:the\s+)?age\s+(?:of\s+)?"
        rf"(?:\d{{1,3}}|{_NUM_RE})\b",
        re.IGNORECASE,
    ),
    re.compile(r"\baged?\s+\d{1,3}\b", re.IGNORECASE),
    re.compile(
        r"\bas\s+a\s+(?:kid|child|baby|teenager|teen|student)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bin\s+my\s+(?:teens|twenties|thirties|forties|fifties|"
        r"sixties|seventies|eighties|nineties|childhood|youth|"
        r"early\s+years)\b",
        re.IGNORECASE,
    ),
]
_HOL_RE = re.compile(
    rf"\b(?:(?P<dir>last|this|next)\s+)?(?:the\s+)?(?P<h>{_HOLIDAY_RE})"
    rf"(?:\s+(?P<y>(?:19|20)\d{{2}}))?\b",
    re.IGNORECASE,
)

_VAGUE = [
    "once upon a time", "in the near future", "in the distant past",
    "in the distant future", "one of these days", "some day", "someday",
    "one day", "sooner or later", "before too long", "before long",
    "a long time ago", "a little while ago", "not long ago",
    "a while ago", "a while back", "some time ago", "ages ago",
    "long ago", "way back", "back in the day", "back then",
    "in the past", "in the future", "down the line", "down the road",
    "in due course", "in due time", "from now on", "any day now",
    "at some point", "in a bit", "in a moment", "in a minute",
    "right away", "any minute now", "later on", "the other week",
    "the other month", "the other year", "in the meantime",
    "at the time", "up to now", "by then", "so far", "as yet",
    "in time", "on time", "going forward", "in retrospect",
    "these days", "nowadays", "meanwhile", "afterwards", "afterward",
    "beforehand", "shortly", "eventually", "sometime", "some time",
    "soon", "later", "earlier", "once",
]
_VAGUE_RE = re.compile(
    r"\b(?:" + "|".join(
        r"\s+".join(re.escape(w) for w in v.split())
        for v in sorted(_VAGUE, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
#: Sub-day units are unresolvable at day granularity — recognized and
#: kept unknown (T30) rather than silently dropped.
_SUBDAY_RE = re.compile(
    rf"\b(?:{_NUM_RE}|a\s+few|a\s+couple(?:\s+of)?|several)\s+"
    rf"(?:seconds?|minutes?|hours?)\s+(?:ago|later|back|earlier)\b"
    rf"|\bin\s+(?:{_NUM_RE}|a\s+few|a\s+couple(?:\s+of)?)\s+"
    rf"(?:seconds?|minutes?|hours?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# rule bodies — each appends _Cands to ``out``
# ---------------------------------------------------------------------------

def _r_t01(text, anchor_us, out):
    """T01 — ISO dates and datetimes (naive times ⇒ UTC)."""
    for m in _ISO_DT_RE.finditer(text):
        y, mo, d = int(m["y"]), int(m["mo"]), int(m["d"])
        day = _valid(y, mo, d)
        if day is None:
            out.append(_Cand(m.start(), m.end(), None, None,
                             _P.UNKNOWN, _S.UNKNOWN, "T30", prio=40))
            continue
        if m["h"] is not None:
            tz = m["tz"]
            if tz and tz.upper() == "Z":
                off = "+00:00"
            elif tz and ":" not in tz:
                off = tz[:3] + ":" + tz[3:]
            else:
                off = tz or "+00:00"
            iso = (f"{y:04d}-{mo:02d}-{d:02d}T{m['h']}:{m['mi']}"
                   f":{m['s'] or '00'}")
            if m["us"]:
                iso += "." + m["us"].ljust(6, "0")
            try:
                dt = datetime.fromisoformat(iso + off)
            except ValueError:
                out.append(_Cand(m.start(), m.end(), None, None,
                                 _P.UNKNOWN, _S.UNKNOWN, "T30", prio=40))
                continue
            us = _to_us(dt.astimezone(timezone.utc))
            out.append(_Cand(m.start(), m.end(), us, us,
                             _P.INSTANT, _S.EXPLICIT, "T01", prio=10))
        else:
            out.append(_Cand(m.start(), m.end(), _dus(day),
                             _dus(day + timedelta(days=1)),
                             _P.DAY, _S.EXPLICIT, "T01", prio=10))


def _r_t02_named(text, adate, out):
    """T02 — ``Month D[, YYYY]`` / ``D Month [YYYY]`` (nearest past year)."""
    for rex in (_MDY_RE, _DMY_RE):
        for m in rex.finditer(text):
            mn, dy = _MONTHS[m["mn"].lower()], int(m["d"])
            d = None
            if dy <= 31:
                if m["y"]:
                    d = _valid(int(m["y"]), mn, dy)
                    src = _S.EXPLICIT
                else:
                    d = _md_le(adate, mn, dy)
                    src = _S.RESOLVED_RELATIVE
            if d is None:
                out.append(_Cand(m.start(), m.end(), None, None,
                                 _P.UNKNOWN, _S.UNKNOWN, "T30", prio=40))
                continue
            out.append(_Cand(m.start(), m.end(), _dus(d),
                             _dus(d + timedelta(days=1)),
                             _P.DAY, src, "T02", prio=15))


def _r_t02_slash(text, day_first, out):
    """T02 — ``M/D/YYYY`` (en-US) / ``D/M/YYYY`` (day-first locales)."""
    for m in _SLASH_RE.finditer(text):
        a, b, y = int(m["a"]), int(m["b"]), int(m["y"])
        if y < 100:
            y += 2000 if y <= 69 else 1900
        mdy, dmy = _valid(y, a, b), _valid(y, b, a)
        if mdy and dmy:
            d = dmy if day_first else mdy
            out.append(_Cand(m.start(), m.end(), _dus(d),
                             _dus(d + timedelta(days=1)),
                             _P.DAY, _S.EXPLICIT, "T02",
                             amb=True, prio=15))
        elif mdy or dmy:
            d = mdy or dmy
            out.append(_Cand(m.start(), m.end(), _dus(d),
                             _dus(d + timedelta(days=1)),
                             _P.DAY, _S.EXPLICIT, "T02", prio=15))
        else:
            out.append(_Cand(m.start(), m.end(), None, None,
                             _P.UNKNOWN, _S.UNKNOWN, "T30", prio=40))


def _r_t03_t04_t05(text, adate, out):
    """T03 ``Month YYYY`` · T04 ``YYYY``/decades · T05 bare ``Month``."""
    for m in _MY_RE.finditer(text):
        s, e = _month_bounds(int(m["y"]), _MONTHS[m["mn"].lower()])
        out.append(_Cand(m.start(), m.end(), _dus(s), _dus(e),
                         _P.MONTH, _S.EXPLICIT, "T03", prio=15))
    for m in _DECADE_RE.finditer(text):
        y = int(m["y"]) * 10
        out.append(_Cand(m.start(), m.end(), _dus(date(y, 1, 1)),
                         _dus(date(y + 10, 1, 1)),
                         _P.DECADE, _S.EXPLICIT, "T04", prio=15))
    for m in _YEAR_RE.finditer(text):
        y = int(m["y"])
        out.append(_Cand(m.start(), m.end(), _dus(date(y, 1, 1)),
                         _dus(date(y + 1, 1, 1)),
                         _P.YEAR, _S.EXPLICIT, "T04", prio=18))
    for m in _BARE_MONTH_RE.finditer(text):
        if not _guarded_ok(text, m.start(), m["mn"]):
            continue
        y, mo = _month_le(adate, _MONTHS[m["mn"].lower()])
        s, e = _month_bounds(y, mo)
        out.append(_Cand(m.start(), m.end(), _dus(s), _dus(e),
                         _P.MONTH, _S.RESOLVED_RELATIVE, "T05", prio=20))


def _r_daywords(text, adate, anchor_us, out):
    """T06/T07/T08 — today/yesterday/tomorrow family (+ instants)."""
    for m in _DAYWORD_RE.finditer(text):
        w = re.sub(r"\s+", " ", m["w"].lower())
        if w in ("right now", "at the moment", "as of now",
                 "currently", "now"):
            out.append(_Cand(m.start(), m.end(), anchor_us, anchor_us,
                             _P.INSTANT, _S.RESOLVED_RELATIVE, "T06",
                             prio=20))
            continue
        if w.startswith("yesterday") or w == "last night":
            d = adate - timedelta(days=1)
        elif w.startswith("tomorrow"):
            d = adate + timedelta(days=1)
        else:
            d = adate
        rule = "T06" if d == adate else ("T07" if d < adate else "T08")
        out.append(_Cand(m.start(), m.end(), _dus(d),
                         _dus(d + timedelta(days=1)),
                         _P.DAY, _S.RESOLVED_RELATIVE, rule, prio=20))


def _r_blocks(text, adate, out):
    """T13/T14/T15/T16 — last|this|next + week|weekend|month|year."""
    for m in _BLOCK_REL_RE.finditer(text):
        e = _lx_block(m["dir"], m["u"], adate)
        if e is None:
            continue
        rule = {"week": "T13", "weekend": "T14",
                "month": "T15", "year": "T16"}[m["u"].lower()]
        out.append(_Cand(m.start(), m.end(), _dus(e.s), _dus(e.e),
                         e.prec, _S.RESOLVED_RELATIVE, rule, prio=20))


_UNIT_RULE = {"day": "T09", "week": "T10", "month": "T11", "year": "T12"}


def _shift_cand(cs, ce, adate, delta, unit, rule):
    """Anchor-relative shift: day exact, week ±3d, month/year calendar
    arithmetic landing on the whole containing period."""
    if unit == "day":
        s = adate + timedelta(days=delta)
        return _Cand(cs, ce, _dus(s), _dus(s + timedelta(days=1)),
                     _P.DAY, _S.RESOLVED_RELATIVE, rule, prio=20)
    if unit == "week":
        c = adate + timedelta(weeks=delta)
        return _Cand(cs, ce, _dus(c - timedelta(days=3)),
                     _dus(c + timedelta(days=4)),
                     _P.WEEK, _S.RESOLVED_RELATIVE, rule, prio=20)
    if unit == "month":
        t = _add_months(adate, delta)
        s, e = _month_bounds(t.year, t.month)
        return _Cand(cs, ce, _dus(s), _dus(e),
                     _P.MONTH, _S.RESOLVED_RELATIVE, rule, prio=20)
    t = _add_years(adate, delta)
    return _Cand(cs, ce, _dus(date(t.year, 1, 1)),
                 _dus(date(t.year + 1, 1, 1)),
                 _P.YEAR, _S.RESOLVED_RELATIVE, rule, prio=20)


def _r_n_units(text, adate, out):
    """T09–T12 — ``N units ago|later`` / ``in N units`` / span forms."""
    for m in _NREL_RE.finditer(text):
        n, unit = _num(m["n"]), m["u"].lower().rstrip("s")
        delta = n if m["dir"].lower() == "later" else -n
        out.append(_shift_cand(m.start(), m.end(), adate, delta, unit,
                               _UNIT_RULE[unit]))
    for m in _IN_N_RE.finditer(text):
        n, unit = _num(m["n"]), m["u"].lower().rstrip("s")
        out.append(_shift_cand(m.start(), m.end(), adate, n, unit,
                               _UNIT_RULE[unit]))
    # "the past|last|next|coming N units" — N-unit window ending at
    # end-of-anchor-day or starting at the anchor date.
    for m in _SPAN_N_RE.finditer(text):
        n, unit = _num(m["n"]), m["u"].lower().rstrip("s")
        dirw = m["dir"].lower()
        if dirw in ("past", "last"):
            e = adate + timedelta(days=1)
            if unit == "day":
                s = e - timedelta(days=n)
            elif unit == "week":
                s = e - timedelta(weeks=n)
            elif unit == "month":
                s = _add_months(e, -n)
            else:
                s = _add_years(e, -n)
        else:
            s = adate
            if unit == "day":
                e = s + timedelta(days=n)
            elif unit == "week":
                e = s + timedelta(weeks=n)
            elif unit == "month":
                e = _add_months(s, n)
            else:
                e = _add_years(s, n)
        out.append(_Cand(m.start(), m.end(), _dus(s), _dus(e),
                         _P.DAY, _S.RESOLVED_RELATIVE, _UNIT_RULE[unit],
                         prio=22))


def _r_weekdays(text, adate, out):
    """T17 — weekday names (see module docstring for direction rules)."""
    for m in _WEEKDAY_RE.finditer(text):
        dirw = re.sub(r"\s+", " ", (m["dir"] or "").lower())
        if not dirw and not _guarded_ok(text, m.start("wd"), m["wd"]):
            continue
        twd, awd = _WEEKDAYS[m["wd"].lower()], adate.weekday()
        if dirw in ("next", "this coming"):
            delta = (twd - awd) % 7 or 7
        elif dirw == "this":
            delta = twd - awd
        else:  # last / this past / on / bare → most recent strictly < A
            delta = -((awd - twd) % 7 or 7)
        d = adate + timedelta(days=delta)
        out.append(_Cand(m.start(), m.end(), _dus(d),
                         _dus(d + timedelta(days=1)),
                         _P.DAY, _S.RESOLVED_RELATIVE, "T17", prio=20))


def _r_seasons(text, adate, hemi, out):
    """T18 — meteorological seasons, hemisphere-selectable."""
    for m in _SEASON_RE.finditer(text):
        dirw = (m["dir"] or "").lower()
        if not dirw and not _guarded_ok(text, m.start("s"), m["s"]):
            continue
        name = _SEASON_ALIASES.get(m["s"].lower(), m["s"].lower())
        year = int(m["y"]) if m["y"] else None
        bounds = _season_pick(name, dirw or "bare", adate, hemi, year)
        if bounds is None:
            out.append(_Cand(m.start(), m.end(), None, None,
                             _P.UNKNOWN, _S.UNKNOWN, "T30", prio=40))
            continue
        s, e = bounds
        src = _S.EXPLICIT if year is not None else _S.RESOLVED_RELATIVE
        out.append(_Cand(m.start(), m.end(), _dus(s), _dus(e),
                         _P.SEASON, src, "T18", prio=20))


def _few_range(adate, couple, unit):
    """Relative span for 'a few'/'a couple of'/'several' + unit ago."""
    if unit == "day":
        if couple:
            return adate - timedelta(days=3), adate
        return adate - timedelta(days=5), adate - timedelta(days=1)
    if unit == "week":
        if couple:
            return adate - timedelta(days=21), adate - timedelta(days=9)
        return adate - timedelta(days=35), adate - timedelta(days=13)
    if unit == "month":
        lo, hi = (3, 1) if couple else (5, 2)
        base = date(adate.year, adate.month, 1)
        return _add_months(base, -lo), _add_months(base, -(hi - 1))
    lo, hi = (3, 1) if couple else (5, 2)
    return date(adate.year - lo, 1, 1), date(adate.year - hi + 1, 1, 1)


def _few_in_range(adate, couple, unit):
    """Symmetric forward span for 'in a few/couple of …'."""
    if unit == "day":
        if couple:
            return adate + timedelta(days=2), adate + timedelta(days=4)
        return adate + timedelta(days=2), adate + timedelta(days=6)
    if unit == "week":
        if couple:
            return adate + timedelta(days=10), adate + timedelta(days=22)
        return adate + timedelta(days=14), adate + timedelta(days=36)
    if unit == "month":
        lo, hi = (1, 3) if couple else (2, 5)
        base = date(adate.year, adate.month, 1)
        return _add_months(base, lo), _add_months(base, hi + 1)
    lo, hi = (1, 3) if couple else (2, 5)
    return date(adate.year + lo, 1, 1), date(adate.year + hi + 1, 1, 1)


def _r_fuzzy(text, adate, out):
    """T19/T20/T21/T29 — fuzzy past ranges (kept coarse on purpose)."""
    for m in _FEW_RE.finditer(text):
        q = re.sub(r"\s+", " ", m["q"].lower())
        unit = m["u"].lower().rstrip("s")
        couple = q.startswith("a couple")
        s, e = _few_range(adate, couple, unit)
        prec = {"day": _P.WEEK, "week": _P.WEEK,
                "month": _P.MONTH, "year": _P.YEAR}[unit]
        out.append(_Cand(m.start(), m.end(), _dus(s), _dus(e), prec,
                         _S.RESOLVED_RELATIVE,
                         "T20" if couple else "T19", prio=25))
    for m in _FEW_IN_RE.finditer(text):
        q = re.sub(r"\s+", " ", m["q"].lower())
        unit = m["u"].lower().rstrip("s")
        couple = q.startswith("a couple")
        s, e = _few_in_range(adate, couple, unit)
        prec = {"day": _P.WEEK, "week": _P.WEEK,
                "month": _P.MONTH, "year": _P.YEAR}[unit]
        out.append(_Cand(m.start(), m.end(), _dus(s), _dus(e), prec,
                         _S.RESOLVED_RELATIVE,
                         "T20" if couple else "T19", prio=25))
    for m in _RECENT_RE.finditer(text):
        s = adate - timedelta(days=30)
        e = adate + timedelta(days=1)
        out.append(_Cand(m.start(), m.end(), _dus(s), _dus(e),
                         _P.MONTH, _S.RESOLVED_RELATIVE, "T21", prio=25))
    for m in _OTHER_RE.finditer(text):
        s = adate - timedelta(days=7)
        out.append(_Cand(m.start(), m.end(), _dus(s), _dus(adate),
                         _P.WEEK, _S.RESOLVED_RELATIVE, "T29", prio=25))


def _r_t22(text, adate, out):
    """T22 — ``[N] day|week|month|year(s) before|after|prior to <expr>``."""
    for m in _T22_RE.finditer(text):
        unit = m["u"].lower().rstrip("s")
        if m["n"]:
            n = _num(m["n"])
        elif not m["u"].lower().endswith("s"):
            n = 1  # "the day after X" / bare "day after X"
        else:
            continue  # plural without a count ("weeks before X") — skip
        emb = _embedded(text, m.end(), adate, prefer="past_le")
        if emb is None:
            continue
        dirw = "before" if m["dir"].lower().startswith(("before", "prior")) \
            else "after"
        if dirw == "before":
            if unit == "day":
                s, e, prec = emb.s - timedelta(days=n), \
                    emb.s - timedelta(days=n - 1), _P.DAY
            elif unit == "week":
                s, e, prec = emb.s - timedelta(weeks=n), \
                    emb.s - timedelta(weeks=n - 1), _P.WEEK
            elif unit == "month":
                t = _add_months(emb.s, -n)
                s, e = _month_bounds(t.year, t.month)
                prec = _P.MONTH
            else:
                t = emb.s.year - n
                s, e, prec = date(t, 1, 1), date(t + 1, 1, 1), _P.YEAR
        else:
            if unit == "day":
                s, e, prec = emb.e + timedelta(days=n - 1), \
                    emb.e + timedelta(days=n), _P.DAY
            elif unit == "week":
                s, e, prec = emb.e + timedelta(weeks=n - 1), \
                    emb.e + timedelta(weeks=n), _P.WEEK
            elif unit == "month":
                t = _add_months(emb.e - timedelta(days=1), n)
                s, e = _month_bounds(t.year, t.month)
                prec = _P.MONTH
            else:
                t = (emb.e - timedelta(days=1)).year + n
                s, e, prec = date(t, 1, 1), date(t + 1, 1, 1), _P.YEAR
        out.append(_Cand(m.start(), emb.end, _dus(s), _dus(e),
                         prec, _S.RESOLVED_RELATIVE, "T22", prio=12))


def _r_t23(text, adate, out):
    """T23 — early/mid/late <Month [YYYY]|YYYY> → period thirds."""
    for m in _T23_RE.finditer(text):
        part = m["part"].lower()
        part = "mid" if part.startswith("mid") else part
        idx = {"early": 0, "mid": 1, "late": 2}[part]
        if m["y2"]:
            y = int(m["y2"])
            thirds = [(date(y, 1, 1), date(y, 5, 1)),
                      (date(y, 5, 1), date(y, 9, 1)),
                      (date(y, 9, 1), date(y + 1, 1, 1))]
            s, e = thirds[idx]
            out.append(_Cand(m.start(), m.end(), _dus(s), _dus(e),
                             _P.MONTH, _S.EXPLICIT, "T23", prio=12))
            continue
        mo = _MONTHS[m["mn"].lower()]
        if m["y"]:
            y, src = int(m["y"]), _S.EXPLICIT
        else:
            y, mo = _month_le(adate, mo)
            src = _S.RESOLVED_RELATIVE
        ms, me = _month_bounds(y, mo)
        mid = ms + timedelta(days=10)
        late = ms + timedelta(days=20)
        s, e = [(ms, mid), (mid, late), (late, me)][idx]
        out.append(_Cand(m.start(), m.end(), _dus(s), _dus(e),
                         _P.DAY, src, "T23", prio=12))


def _r_t24(text, adate, out):
    """T24 — ``first|second|…|last week of <Month [YYYY]>``."""
    ords = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2,
            "3rd": 2, "fourth": 3, "4th": 3, "fifth": 4, "5th": 4}
    for m in _T24_RE.finditer(text):
        mo = _MONTHS[m["mn"].lower()]
        if m["y"]:
            y, src = int(m["y"]), _S.EXPLICIT
        else:
            y, mo = _month_le(adate, mo)
            src = _S.RESOLVED_RELATIVE
        ms, me = _month_bounds(y, mo)
        ordw = m["ord"].lower()
        if ordw in ("last", "final"):
            s, e = me - timedelta(days=7), me
        else:
            s = ms + timedelta(weeks=ords[ordw])
            e = min(s + timedelta(days=7), me)
            if s >= me:  # e.g. "fifth week of February" (non-leap)
                out.append(_Cand(m.start(), m.end(), None, None,
                                 _P.UNKNOWN, _S.UNKNOWN, "T30", prio=40))
                continue
        out.append(_Cand(m.start(), m.end(), _dus(s), _dus(e),
                         _P.WEEK, src, "T24", prio=12))


def _r_t25(text, adate, anchor_us, out):
    """T25 — since/until/before/after/through + from X to Y / between."""
    for m in _T25_FROM_RE.finditer(text):
        e1 = _embedded(text, m.end(), adate, prefer="past_le")
        if e1 is None:
            continue
        sep = re.match(r"\s+(?:to|until|till|til|through|thru)\s+"
                       r"|\s*[-–—]\s*", text[e1.end:])
        if not sep:
            continue
        pos = e1.end + sep.end()
        e2 = (_bare_day(text, pos, e1)
              or _embedded(text, pos, adate, prefer="past_le"))
        if e2 is None:
            continue
        tries = 0
        while e2.roll and e2.s < e1.s and tries < 4:
            e2 = _bump(e2)
            tries += 1
        if e2.e <= e1.s:
            continue  # degenerate/reversed range — never emit
        prec = min((e1.prec, e2.prec), key=lambda p: _PREC_RANK[p])
        out.append(_Cand(m.start(), e2.end, _dus(e1.s), _dus(e2.e),
                         prec, _S.RESOLVED_RELATIVE, "T25", prio=10))
    for m in _T25_BETWEEN_RE.finditer(text):
        e1 = _embedded(text, m.end(), adate, prefer="past_le")
        if e1 is None:
            continue
        sep = re.match(r"\s+and\s+", text[e1.end:])
        if not sep:
            continue
        pos = e1.end + sep.end()
        e2 = (_bare_day(text, pos, e1)
              or _embedded(text, pos, adate, prefer="past_le"))
        if e2 is None:
            continue
        tries = 0
        while e2.roll and e2.s < e1.s and tries < 4:
            e2 = _bump(e2)
            tries += 1
        if e2.e <= e1.s:
            continue
        prec = min((e1.prec, e2.prec), key=lambda p: _PREC_RANK[p])
        out.append(_Cand(m.start(), e2.end, _dus(e1.s), _dus(e2.e),
                         prec, _S.RESOLVED_RELATIVE, "T25", prio=10))
    for m in _T25_BOUND_RE.finditer(text):
        b = m["b"].lower()
        # Nearest-past reading for every bound word ("worked until
        # Friday" = the Friday that just passed). Documented convention.
        emb = _embedded(text, m.end(), adate, prefer="past_le")
        if emb is None:
            continue
        if b == "since":
            s, e = _dus(emb.s), anchor_us
        elif b in ("until", "till", "til", "before"):
            s, e = None, _dus(emb.s)
        elif b == "through":
            s, e = None, _dus(emb.e)
        else:  # after
            s, e = _dus(emb.e), None
        out.append(_Cand(m.start(), emb.end, s, e,
                         emb.prec, _S.RESOLVED_RELATIVE, "T25", prio=10))


def _r_t26(text, out):
    """T26 — ``for N units`` durations: marked, never resolved."""
    for m in _T26_RE.finditer(text):
        out.append(_Cand(m.start(), m.end(), None, None,
                         _P.UNKNOWN, _S.UNKNOWN, "T26", prio=30))


def _r_t27(text, out):
    """T27 — age expressions need a birth fact; none exists ⇒ unknown."""
    for rex in _AGE_RES:
        for m in rex.finditer(text):
            out.append(_Cand(m.start(), m.end(), None, None,
                             _P.UNKNOWN, _S.UNKNOWN, "T27", prio=30))


def _r_t28(text, adate, out):
    """T28 — holidays: fixed, Thanksgiving (4th Thu Nov), computus Easter."""
    for m in _HOL_RE.finditer(text):
        key = _holiday_key(m["h"])
        if key not in _HOLIDAYS:
            continue
        year = int(m["y"]) if m["y"] else None
        if year is not None:
            d = _holiday_date(key, year)
            src = _S.EXPLICIT
        else:
            d = _hol_pick(key, (m["dir"] or "").lower(), adate, "past_le")
            src = _S.RESOLVED_RELATIVE
        if d is None:
            out.append(_Cand(m.start(), m.end(), None, None,
                             _P.UNKNOWN, _S.UNKNOWN, "T30", prio=40))
            continue
        out.append(_Cand(m.start(), m.end(), _dus(d),
                         _dus(d + timedelta(days=1)),
                         # prio 14: holiday surfaces beat the generic
                         # T02 ``D Month`` parse on identical spans
                         # ("4th of July" is a holiday, not just DMY).
                         _P.DAY, src, "T28", prio=14))


def _r_t30(text, out):
    """T30 — recognized temporal hedges that cannot be bounded, and
    sub-day units the day-granular model cannot express."""
    for m in _VAGUE_RE.finditer(text):
        out.append(_Cand(m.start(), m.end(), None, None,
                         _P.UNKNOWN, _S.UNKNOWN, "T30", prio=40))
    for m in _SUBDAY_RE.finditer(text):
        out.append(_Cand(m.start(), m.end(), None, None,
                         _P.UNKNOWN, _S.UNKNOWN, "T30", prio=40))


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------

def _candidates(text: str, adate: date, anchor_us: int,
                day_first: bool, hemi: str) -> List[_Cand]:
    out: List[_Cand] = []
    _r_t01(text, anchor_us, out)
    _r_t02_named(text, adate, out)
    _r_t02_slash(text, day_first, out)
    _r_t03_t04_t05(text, adate, out)
    _r_daywords(text, adate, anchor_us, out)
    _r_blocks(text, adate, out)
    _r_n_units(text, adate, out)
    _r_weekdays(text, adate, out)
    _r_seasons(text, adate, hemi, out)
    _r_fuzzy(text, adate, out)
    _r_t22(text, adate, out)
    _r_t23(text, adate, out)
    _r_t24(text, adate, out)
    _r_t25(text, adate, anchor_us, out)
    _r_t26(text, out)
    _r_t27(text, out)
    _r_t28(text, adate, out)
    _r_t30(text, out)
    return out


def _resolve_overlaps(cands: List[_Cand]) -> List[_Cand]:
    """Overlapping candidates keep the longest span; ties go to the
    stronger rule class (lower prio) then the earlier start. Identical
    (span, rule) duplicates collapse."""
    seen = set()
    uniq = []
    for c in cands:
        k = (c.cs, c.ce, c.rule)
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    order = sorted(uniq, key=lambda c: (-(c.ce - c.cs), c.prio, c.cs))
    kept: List[_Cand] = []
    for c in order:
        if any(c.cs < k.ce and k.cs < c.ce for k in kept):
            continue
        kept.append(c)
    kept.sort(key=lambda c: c.cs)
    return kept


def resolve(text: str, anchor_us: int, *, locale: str = "en-US",
            hemisphere: str = "north") -> List[ResolvedTime]:
    """Resolve every temporal expression in ``text`` against ``anchor_us``.

    Returns ``ResolvedTime`` objects sorted by byte offset; each carries
    its rule id (T01–T30), the anchor it resolved against, UTF-8 byte
    offsets slicing back to the expression, and an ``IntervalUs``.
    Unresolvable phrases are kept with ``precision=unknown`` — never
    dropped (V7-09.03). Deterministic: no wall clock is consulted.
    """
    t = str(text if text is not None else "")
    if not t:
        return []
    anchor_us = int(anchor_us)
    adate = _to_dt(anchor_us).date()
    loc = str(locale or "").strip().lower()
    supported = loc in SUPPORTED_LOCALES
    day_first = loc in DAY_FIRST_LOCALES
    hemi = str(hemisphere or "").strip().lower()
    if hemi not in _HEMISPHERES:
        hemi = "north"

    cands = _resolve_overlaps(
        _candidates(t, adate, anchor_us, day_first, hemi))

    offsets = utf8_offsets(t)
    out: List[ResolvedTime] = []
    for c in cands:
        amb = c.amb or not supported
        out.append(ResolvedTime(
            text=t[c.cs:c.ce],
            byte_start=offsets[c.cs],
            byte_end=offsets[c.ce],
            interval=IntervalUs(
                start_us=c.s, end_us=c.e,
                precision=c.prec, source=c.src,
                rule_id=c.rule, anchor_us=anchor_us),
            rule_id=c.rule,
            ambiguous_locale=amb,
        ))
    return out


def resolve_query_window(norm: "NormAnalysis",
                         query_time_us: int) -> Optional[IntervalUs]:
    """Fold a query's temporal expressions into one covering window.

    ``norm`` is a ``norm/v2`` analysis (imported for typing only). The
    resolver runs on ``norm.text`` — the folded matching projection —
    falling back to the term surfaces when it is empty. Known intervals
    union into a single window (precision = coarsest component, source
    = explicit only when every component is explicit). Expressions that
    produced no bounds (T26 durations, T27 ages, T30 vague phrases) are
    ignored; if nothing bounded exists the window is ``None``.
    """
    text = getattr(norm, "text", "") or ""
    if not text:
        terms = getattr(norm, "terms", ()) or ()
        text = " ".join(getattr(t, "term", "") for t in terms)
    rts = resolve(text, int(query_time_us))
    usable = [r for r in rts
              if r.interval.start_us is not None
              or r.interval.end_us is not None]
    if not usable:
        return None
    starts = [r.interval.start_us for r in usable
              if r.interval.start_us is not None]
    ends = [r.interval.end_us for r in usable
            if r.interval.end_us is not None]
    prec = max((r.interval.precision for r in usable),
               key=lambda p: _PREC_RANK[p])
    src = (_S.EXPLICIT
           if all(r.interval.source == _S.EXPLICIT for r in usable)
           else _S.RESOLVED_RELATIVE)
    rules = "+".join(sorted({r.rule_id for r in usable}))
    return IntervalUs(
        start_us=min(starts) if starts else None,
        end_us=max(ends) if ends else None,
        precision=prec, source=src,
        rule_id=rules, anchor_us=int(query_time_us))


__all__ = [
    "RESOLVER_ID", "RULE_SET_STATUS", "PARSER_LOCALE", "PARSER_CALENDAR",
    "SUPPORTED_LOCALES", "DAY_FIRST_LOCALES", "locale_supported",
    "resolve", "resolve_query_window",
]
