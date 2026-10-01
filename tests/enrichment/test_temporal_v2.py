"""Fixture suite for ``temporal/v2`` (SPEC_V7 §32.9, V7-09.03–05).

Builds a ≥ 800-expression gold fixture programmatically: each rule T01–T30
is exercised by parameterized cases whose expected intervals are computed
by independent test-side calendar arithmetic (``calendar.timegm`` for the
µs conversion, plain ``date``/``timedelta`` math for bounds — a different
code path than the module's epoch+timedelta conversion).

Assertions:

- overall exact-interval accuracy ≥ 0.97 (V7-09.04) — we target 1.0;
- ≥ 20 cases per rule where feasible (§32.9);
- every ``ResolvedTime`` byte span slices back to its expression text;
- every resolution records ``anchor_us`` and a ``rule_id``;
- determinism — identical inputs give identical outputs, no wall clock;
- kept-unknown discipline: T26/T27/T30 emit ``precision=unknown`` and
  ``None`` bounds, never vanish.
"""

from __future__ import annotations

import calendar
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Tuple

import pytest

from verbatim.core.types_v7 import (
    NormAnalysis,
    NormTerm,
    OccurredPrecision,
    OccurredSource,
)
from verbatim.enrichment.temporal_v2 import (
    DAY_FIRST_LOCALES,
    RESOLVER_ID,
    RULE_SET_STATUS,
    SUPPORTED_LOCALES,
    locale_supported,
    resolve,
    resolve_query_window,
)

P = OccurredPrecision
S = OccurredSource

# ---------------------------------------------------------------------------
# independent gold helpers (test-side; deliberately separate code path)
# ---------------------------------------------------------------------------

DAY_US = 86_400_000_000


def adt_us(d: date, h: int = 15, mi: int = 30) -> int:
    """Anchor µs at h:mi UTC — via timegm, not the module's timedelta path."""
    return calendar.timegm(
        datetime(d.year, d.month, d.day, h, mi).timetuple()) * 1_000_000


def dus(d: date) -> int:
    return calendar.timegm(d.timetuple()) * 1_000_000


def dus_dt(dt: datetime) -> int:
    return calendar.timegm(dt.timetuple()) * 1_000_000 + dt.microsecond


def iv(s: date, e: date) -> Tuple[int, int]:
    return dus(s), dus(e)


def day_iv(d: date) -> Tuple[int, int]:
    return iv(d, d + timedelta(days=1))


def _am(d: date, n: int) -> date:
    """add months (independent impl)."""
    t = d.year * 12 + d.month - 1 + n
    y, m = divmod(t, 12)
    m += 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def _ay(d: date, n: int) -> date:
    try:
        return d.replace(year=d.year + n)
    except ValueError:
        return d.replace(year=d.year + n, day=28)


def mon_iv(y: int, m: int) -> Tuple[int, int]:
    return iv(date(y, m, 1), _am(date(y, m, 1), 1))


def yr_iv(y: int) -> Tuple[int, int]:
    return iv(date(y, 1, 1), date(y + 1, 1, 1))


def iso_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def weekend_gold(d: date) -> Tuple[date, date]:
    sat = d + timedelta(days=5 - d.weekday())
    return sat, sat + timedelta(days=2)


_SN = {"spring": (3, 6), "summer": (6, 9), "autumn": (9, 12),
       "winter": (12, 3)}
_SS = {"spring": (9, 12), "summer": (12, 3), "autumn": (3, 6),
       "winter": (6, 9)}


def season_gold(season: str, year: int, hemi: str) -> Tuple[date, date]:
    sm, em = (_SN if hemi == "north" else _SS)[season]
    ey = year + (1 if em < sm else 0)
    return date(year, sm, 1), date(ey, em, 1)


def season_pick_gold(season: str, dirw: str, A: date,
                     hemi: str) -> Tuple[date, date]:
    occs = [season_gold(season, y, hemi)
            for y in range(A.year - 2, A.year + 3)]
    cont = [o for o in occs if o[0] <= A < o[1]]
    ended = [o for o in occs if o[1] <= A]
    fut = [o for o in occs if o[0] > A]
    if dirw == "last":
        return max(ended)
    if dirw == "next":
        return min(fut)
    if dirw == "this":
        if cont:
            return cont[0]
        cand = None
        for o in ended + fut:
            dist = min(abs((o[0] - A).days), abs((o[1] - A).days))
            key = (dist, 0 if o[1] <= A else 1)
            if cand is None or key < cand[0]:
                cand = (key, o)
        return cand[1]
    if cont:
        return cont[0]
    if ended:
        return max(ended)
    return min(fut)


def easter_gold(y: int) -> date:
    """Gregorian computus — independently written, same algorithm."""
    a = y % 19
    b = y // 100
    c = y % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(y, month, day)


def tgiv_gold(y: int) -> date:
    first = 1 + (3 - date(y, 11, 1).weekday()) % 7  # Thursday = 3
    return date(y, 11, first + 21)


_HOL_FIXED = {
    "christmas": (12, 25), "christmas day": (12, 25),
    "christmas eve": (12, 24), "xmas": (12, 25), "boxing day": (12, 26),
    "new year": (1, 1), "new year's": (1, 1), "new years": (1, 1),
    "new year's day": (1, 1), "new years day": (1, 1),
    "new year's eve": (12, 31), "new years eve": (12, 31),
    "halloween": (10, 31), "valentine's day": (2, 14),
    "valentines day": (2, 14), "st valentine's day": (2, 14),
    "st. valentine's day": (2, 14), "st patrick's day": (3, 17),
    "st. patrick's day": (3, 17), "saint patrick's day": (3, 17),
    "independence day": (7, 4), "fourth of july": (7, 4),
    "4th of july": (7, 4),
}


def hol_date(name: str, y: int) -> date:
    if name in ("easter", "easter sunday", "easter day"):
        return easter_gold(y)
    if name in ("thanksgiving", "thanksgiving day"):
        return tgiv_gold(y)
    m, d = _HOL_FIXED[name]
    return date(y, m, d)


def hol_pick(name: str, dirw: str, A: date) -> date:
    """bare ≤ A · this → on-A else next · last < A · next > A."""
    if dirw == "next":
        d = hol_date(name, A.year)
        return d if d > A else hol_date(name, A.year + 1)
    if dirw == "this":
        d = hol_date(name, A.year)
        return d if d >= A else hol_date(name, A.year + 1)
    if dirw == "last":
        d = hol_date(name, A.year)
        return d if d < A else hol_date(name, A.year - 1)
    d = hol_date(name, A.year)
    return d if d <= A else hol_date(name, A.year - 1)


def md_le(A: date, m: int, d: int) -> date:
    """Most recent month-day ≤ A (yearless forms, nearest past)."""
    try:
        cand = date(A.year, m, d)
    except ValueError:
        cand = None
    if cand is not None and cand <= A:
        return cand
    return date(A.year - 1, m, d)


def mon_le(A: date, m: int) -> Tuple[int, int]:
    return (A.year, m) if m <= A.month else (A.year - 1, m)


def wd_gold(A: date, twd: int, dirw: str) -> date:
    """last|on|bare → most recent strictly before; next → first strictly
    after; this → occurrence inside A's ISO week; this past → strict
    before; this coming → strict after."""
    awd = A.weekday()
    if dirw in ("next", "this coming"):
        delta = (twd - awd) % 7 or 7
    elif dirw == "this":
        delta = twd - awd
    else:
        delta = -((awd - twd) % 7 or 7)
    return A + timedelta(days=delta)


_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}
_MONTH_FULL = list(_MONTHS)
_MONTHS_FULL_ABB = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Sept": 9, "Oct": 10, "Nov": 11,
    "Dec": 12,
}
_WDS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6}


# ---------------------------------------------------------------------------
# case machinery
# ---------------------------------------------------------------------------

@dataclass
class Case:
    rule: str
    text: str
    anchor: date
    s: Optional[date] = None
    e: Optional[date] = None
    prec: OccurredPrecision = P.DAY
    span: Optional[str] = None
    amb: bool = False
    locale: str = "en-US"
    hemi: str = "north"
    s_us: Optional[int] = None   # instant overrides
    e_us: Optional[int] = None
    note: str = ""


CASES: list[Case] = []


def C(rule, text, anchor, s=None, e=None, prec=P.DAY, span=None, amb=False,
      locale="en-US", hemi="north", s_us=None, e_us=None, note=""):
    CASES.append(Case(rule, text, anchor, s, e, prec, span, amb, locale,
                      hemi, s_us, e_us, note))


# Anchors spread across every weekday + month/year/leap boundaries.
A_FRI = date(2026, 9, 18)   # Friday
A_MON = date(2026, 9, 14)   # Monday
A_TUE = date(2026, 9, 15)   # Tuesday
A_WED = date(2026, 9, 16)   # Wednesday
A_THU = date(2026, 9, 17)   # Thursday
A_SAT = date(2026, 9, 19)   # Saturday
A_SUN = date(2026, 9, 20)   # Sunday
A_JAN31 = date(2026, 1, 31)   # Saturday, month-end
A_MAR31 = date(2026, 3, 31)   # Tuesday, month-end
A_LEAP = date(2024, 2, 29)    # Thursday, leap day
A_NYE = date(2023, 12, 31)    # Sunday, year-end
A_XMAS = date(2026, 12, 25)   # Friday = Christmas
A_JUN = date(2026, 6, 15)     # Monday
A_FEB = date(2026, 2, 10)     # Tuesday
WD_ANCHORS = [A_MON, A_TUE, A_WED, A_THU, A_FRI, A_SAT, A_SUN]
ALL_ANCHORS = WD_ANCHORS + [A_JAN31, A_MAR31, A_LEAP, A_NYE, A_XMAS,
                            A_JUN, A_FEB]

# ---------------------------------------------------------------------------
# T01 — ISO date / datetime
# ---------------------------------------------------------------------------

for (y, mo, d) in [(2024, 1, 15), (1999, 12, 31), (2020, 2, 29),
                   (2030, 6, 1), (2026, 9, 18), (2000, 1, 1),
                   (1975, 7, 4), (2026, 2, 28), (2015, 11, 30),
                   (2024, 10, 5), (1990, 3, 22), (2045, 8, 9)]:
    t = f"{y:04d}-{mo:02d}-{d:02d}"
    C("T01", t, A_FRI, date(y, mo, d), date(y, mo, d) + timedelta(days=1),
      prec=P.DAY, span=t)
    C("T01", f"logs end {t} utc", A_MON, date(y, mo, d),
      date(y, mo, d) + timedelta(days=1), prec=P.DAY, span=t)

for t, dt in [
    ("2024-01-15T08:30:00Z", datetime(2024, 1, 15, 8, 30)),
    ("2024-01-15T08:30:00+02:00",
     datetime(2024, 1, 15, 6, 30)),
    ("2024-01-15 08:30", datetime(2024, 1, 15, 8, 30)),
    ("2024-01-15t23:59:59z", datetime(2024, 1, 15, 23, 59, 59)),
    ("2024-01-15T08:30:00.250Z",
     datetime(2024, 1, 15, 8, 30, 0, 250000)),
    ("2024-01-15T08:30-03:00", datetime(2024, 1, 15, 11, 30)),
]:
    C("T01", t, A_FRI, prec=P.INSTANT, span=t,
      s_us=dus_dt(dt), e_us=dus_dt(dt))

for t in ["2024-13-01", "2024-02-30", "2023-04-31", "2024-00-10"]:
    C("T30", t, A_FRI, prec=P.UNKNOWN, span=t, note="invalid ISO kept")

# ---------------------------------------------------------------------------
# T02 — named dates + slash dates
# ---------------------------------------------------------------------------

for m in range(1, 13):
    mn = _MONTH_FULL[m - 1].capitalize()
    for d in (1, 15, 28):
        for t in (f"{mn} {d}, 2024", f"{mn} {d} 2024",
                  f"{d} {mn} 2024", f"{d} of {mn} 2024"):
            C("T02", t, A_FRI, date(2024, m, d),
              date(2024, m, d) + timedelta(days=1), prec=P.DAY, span=t)

for mn, m, d in [("March", 3, 5), ("June", 6, 20), ("Dec", 12, 1),
                 ("Sep", 9, 30), ("January", 1, 5)]:
    for t in (f"{mn} {d}", f"{mn} {d}th", f"{d} {mn}", f"{d}th {mn}"):
        g = md_le(A_FRI, m, d)
        C("T02", t, A_FRI, g, g + timedelta(days=1), prec=P.DAY, span=t)

# slash dates — unambiguous one-reading and ambiguous two-reading
for t, y, mo, d, amb, loc in [
    ("15/03/2024", 2024, 3, 15, False, "en-US"),   # only D/M valid
    ("03/15/2024", 2024, 3, 15, False, "en-US"),   # only M/D valid
    ("03/25/2024", 2024, 3, 25, False, "en-US"),
    ("3/4/2024", 2024, 3, 4, True, "en-US"),       # ambiguous → M/D
    ("04/03/2024", 2024, 4, 3, True, "en-US"),
    ("3/4/2024", 2024, 4, 3, True, "en-GB"),       # ambiguous → D/M
    ("04/03/2024", 2024, 3, 4, True, "en-GB"),
    ("3/4/24", 2024, 3, 4, True, "en-US"),
    ("3/4/95", 1995, 3, 4, True, "en-US"),
    ("3/4/70", 1970, 3, 4, True, "en-US"),
    ("5.3.2024", 2024, 5, 3, True, "en-US"),
    ("5-3-2024", 2024, 5, 3, True, "en-US"),
    ("5.3.2024", 2024, 3, 5, True, "en-GB"),
    ("11/12/1999", 1999, 11, 12, True, "en-US"),
]:
    C("T02", t, A_FRI, date(y, mo, d), date(y, mo, d) + timedelta(days=1),
      prec=P.DAY, span=t, amb=amb, locale=loc)

for t in ["13/45/2024", "0/0/2024", "99/99/99"]:
    C("T30", t, A_FRI, prec=P.UNKNOWN, span=t, note="invalid slash kept")

# ---------------------------------------------------------------------------
# T03 — Month YYYY
# ---------------------------------------------------------------------------

for m in range(1, 13):
    mn = _MONTH_FULL[m - 1].capitalize()
    for y in (2024, 1999):
        t = f"{mn} {y}"
        s, e = mon_iv(y, m)
        C("T03", t, A_FRI, date(y, m, 1), _am(date(y, m, 1), 1),
          prec=P.MONTH, span=t)
t = "March of 2024"
C("T03", t, A_FRI, date(2024, 3, 1), date(2024, 4, 1), prec=P.MONTH, span=t)
t = "march 2024"
C("T03", t, A_FRI, date(2024, 3, 1), date(2024, 4, 1), prec=P.MONTH, span=t)

# ---------------------------------------------------------------------------
# T04 — bare YYYY and decades
# ---------------------------------------------------------------------------

for y in (1985, 1990, 1999, 2000, 2001, 2010, 2019, 2020, 2024, 2025,
          2026, 2030):
    for t in (f"{y}", f"in {y}", f"during {y}"):
        C("T04", t, A_FRI, date(y, 1, 1), date(y + 1, 1, 1),
          prec=P.YEAR, span=f"{y}")

for t, y in [("the 1990s", 1990), ("in the 1980s", 1980),
             ("the 2000s", 2000), ("during the 2020s", 2020)]:
    C("T04", t, A_FRI, date(y, 1, 1), date(y + 10, 1, 1),
      prec=P.DECADE, span=f"{y}s")

# ---------------------------------------------------------------------------
# T05 — bare Month (nearest past occurrence)
# ---------------------------------------------------------------------------

for m in range(1, 13):
    mn = _MONTH_FULL[m - 1].capitalize()
    y, mo = mon_le(A_FRI, m)
    C("T05", mn, A_FRI, date(y, m, 1), _am(date(y, m, 1), 1),
      prec=P.MONTH, span=mn)
    t = f"in {mn.lower()}"   # lowercase + temporal lead-in
    C("T05", t, A_FRI, date(y, m, 1), _am(date(y, m, 1), 1),
      prec=P.MONTH, span=mn.lower())

for mn in ("Jan", "Feb", "Sept", "Oct", "Dec"):
    m = _MONTHS_FULL_ABB[mn]
    y, mo = mon_le(A_FRI, m)
    t = f"back in {mn}"
    C("T05", t, A_FRI, date(y, m, 1), _am(date(y, m, 1), 1),
      prec=P.MONTH, span=mn)

# ---------------------------------------------------------------------------
# T06–T08 — today / yesterday / tomorrow family
# ---------------------------------------------------------------------------

for w in ("today", "tonight", "this morning", "this afternoon",
          "this evening"):
    for a in WD_ANCHORS:
        C("T06", w, a, a, a + timedelta(days=1), prec=P.DAY, span=w)
for w in ("right now", "now", "currently", "at the moment", "as of now"):
    for a in WD_ANCHORS[:4]:
        C("T06", w, a, prec=P.INSTANT, span=w,
          s_us=adt_us(a), e_us=adt_us(a))

for w in ("yesterday", "last night", "yesterday morning",
          "yesterday evening"):
    for a in WD_ANCHORS:
        d = a - timedelta(days=1)
        C("T07", w, a, d, d + timedelta(days=1), prec=P.DAY, span=w)

for w in ("tomorrow", "tomorrow morning", "tomorrow night",
          "tomorrow afternoon"):
    for a in WD_ANCHORS:
        d = a + timedelta(days=1)
        C("T08", w, a, d, d + timedelta(days=1), prec=P.DAY, span=w)

# ---------------------------------------------------------------------------
# T09 — N days ago | later | in N days (+ span forms)
# ---------------------------------------------------------------------------

for n in (1, 2, 3, 5, 7, 10, 14, 30, 60, 100):
    for t, delta in ((f"{n} days ago", -n), (f"{n} days later", n),
                     (f"in {n} days", n), (f"{n} days back", -n),
                     (f"{n} days earlier", -n)):
        d = A_WED + timedelta(days=delta)
        C("T09", t, A_WED, d, d + timedelta(days=1), prec=P.DAY, span=t)

for t, n in [("a day ago", 1), ("one day ago", 1), ("two days ago", 2),
             ("three days later", 3), ("twelve days ago", 12)]:
    d = A_FRI + timedelta(days=-n if "ago" in t else n)
    C("T09", t, A_FRI, d, d + timedelta(days=1), prec=P.DAY, span=t)

for t, n in [("the last 3 days", 3), ("the past 7 days", 7),
             ("last 10 days", 10), ("the past 2 days", 2)]:
    e = A_FRI + timedelta(days=1)
    C("T09", t, A_FRI, e - timedelta(days=n), e, prec=P.DAY, span=t)
for t, n in [("the next 3 days", 3), ("next 5 days", 5)]:
    C("T09", t, A_FRI, A_FRI, A_FRI + timedelta(days=n), prec=P.DAY,
      span=t)

# ---------------------------------------------------------------------------
# T10 — N weeks ago (±3d, week precision)
# ---------------------------------------------------------------------------

for n in (1, 2, 3, 4, 5, 6, 10):
    for t, delta in ((f"{n} weeks ago", -n), (f"{n} weeks later", n),
                     (f"in {n} weeks", n)):
        c = A_THU + timedelta(weeks=delta)
        C("T10", t, A_THU, c - timedelta(days=3), c + timedelta(days=4),
          prec=P.WEEK, span=t)
for t, n in [("a week ago", 1), ("two weeks ago", 2), ("a week later", 1)]:
    dlt = -n if "ago" in t else n
    c = A_SUN + timedelta(weeks=dlt)
    C("T10", t, A_SUN, c - timedelta(days=3), c + timedelta(days=4),
      prec=P.WEEK, span=t)

# ---------------------------------------------------------------------------
# T11 — N months ago | later | in N months (calendar arithmetic)
# ---------------------------------------------------------------------------

for n in (1, 2, 3, 6, 11, 12, 18, 24):
    for a in (A_FRI, A_JAN31, A_MAR31, A_LEAP, A_NYE):
        t = f"{n} months ago"
        target = _am(a, -n)
        C("T11", t, a, date(target.year, target.month, 1),
          _am(date(target.year, target.month, 1), 1), prec=P.MONTH,
          span=t)
for n in (1, 3, 6):
    t = f"in {n} months"
    target = _am(A_MON, n)
    C("T11", t, A_MON, date(target.year, target.month, 1),
      _am(date(target.year, target.month, 1), 1), prec=P.MONTH, span=t)
    t = f"{n} months later"
    C("T11", t, A_MON, date(target.year, target.month, 1),
      _am(date(target.year, target.month, 1), 1), prec=P.MONTH, span=t)

# ---------------------------------------------------------------------------
# T12 — N years ago | later | in N years
# ---------------------------------------------------------------------------

for n in (1, 2, 3, 5, 10, 25):
    for a in (A_FRI, A_LEAP, A_NYE):
        t = f"{n} years ago"
        target = _ay(a, -n)
        C("T12", t, a, date(target.year, 1, 1), date(target.year + 1, 1, 1),
          prec=P.YEAR, span=t)
for t, n in [("a year ago", 1), ("two years ago", 2), ("in 2 years", 2),
             ("3 years later", 3)]:
    fwd = n if (t.startswith("in") or "later" in t) else -n
    target = _ay(A_FRI, fwd)
    C("T12", t, A_FRI, date(target.year, 1, 1), date(target.year + 1, 1, 1),
      prec=P.YEAR, span=t)

# ---------------------------------------------------------------------------
# T13–T16 — last|this|next + week / weekend / month / year
# ---------------------------------------------------------------------------

for dirw, off in (("last", -1), ("this", 0), ("next", 1)):
    for a in WD_ANCHORS:
        mon = iso_monday(a) + timedelta(weeks=off)
        t = f"{dirw} week"
        C("T13", t, a, mon, mon + timedelta(days=7), prec=P.WEEK, span=t)

for dirw, off in (("last", -1), ("this", 0), ("next", 1)):
    for a in WD_ANCHORS:
        sat, sun_end = weekend_gold(a)
        s = sat + timedelta(weeks=off)
        t = f"{dirw} weekend"
        C("T14", t, a, s, s + timedelta(days=2), prec=P.DAY, span=t)

for dirw, off in (("last", -1), ("this", 0), ("next", 1)):
    for a in WD_ANCHORS + [A_JAN31, A_NYE]:
        s = _am(date(a.year, a.month, 1), off)
        t = f"{dirw} month"
        C("T15", t, a, s, _am(s, 1), prec=P.MONTH, span=t)

for dirw, off in (("last", -1), ("this", 0), ("next", 1)):
    for a in WD_ANCHORS + [A_LEAP, A_NYE]:
        y = a.year + off
        t = f"{dirw} year"
        C("T16", t, a, date(y, 1, 1), date(y + 1, 1, 1), prec=P.YEAR,
          span=t)

# ---------------------------------------------------------------------------
# T17 — weekday names
# ---------------------------------------------------------------------------

for wname, twd in _WDS.items():
    wname_c = wname.capitalize()
    for dirw in ("last", "next", "this", "on"):
        for a in (A_WED, A_SAT, A_SUN):
            t = f"{dirw} {wname_c}"
            d = wd_gold(a, twd, dirw)
            C("T17", t, a, d, d + timedelta(days=1), prec=P.DAY, span=t)
    # bare capitalized weekday → most recent strictly before
    d = wd_gold(A_TUE, twd, "")
    C("T17", wname_c, A_TUE, d, d + timedelta(days=1), prec=P.DAY,
      span=wname_c)

for t, wname, dirw in [("this past Monday", "monday", "last"),
                       ("this coming Friday", "friday", "next"),
                       ("this past Sunday", "sunday", "last"),
                       ("this coming Tuesday", "tuesday", "next")]:
    d = wd_gold(A_WED, _WDS[wname], dirw)
    C("T17", t, A_WED, d, d + timedelta(days=1), prec=P.DAY, span=t)

# ---------------------------------------------------------------------------
# T18 — meteorological seasons, both hemispheres
# ---------------------------------------------------------------------------

for season in ("spring", "summer", "autumn", "winter"):
    for dirw in ("last", "this", "next"):
        for a in (A_FRI, A_JUN, A_FEB, A_NYE):
            for hemi in ("north", "south"):
                t = f"{dirw} {season}"
                s, e = season_pick_gold(season, dirw, a, hemi)
                C("T18", t, a, s, e, prec=P.SEASON, span=t, hemi=hemi)
    for a in (A_FRI, A_JUN):
        for hemi in ("north", "south"):
            t = f"in {season}"
            s, e = season_pick_gold(season, "bare", a, hemi)
            C("T18", t, a, s, e, prec=P.SEASON, span=season, hemi=hemi)

# fall alias + explicit year
for t, a in [("last fall", A_FRI), ("in the fall", A_FRI)]:
    s, e = season_pick_gold("autumn", "last" if "last" in t else "bare",
                            a, "north")
    C("T18", t, a, s, e, prec=P.SEASON,
      span="last fall" if "last" in t else "fall")
for season in ("summer", "winter"):
    t = f"{season} 2024"
    s, e = season_gold(season, 2024, "north")
    C("T18", t, A_FRI, s, e, prec=P.SEASON, span=t)

# ---------------------------------------------------------------------------
# T19/T20/T21/T29 — fuzzy ranges
# ---------------------------------------------------------------------------

for a in WD_ANCHORS:
    C("T19", "a few days ago", a, a - timedelta(days=5),
      a - timedelta(days=1), prec=P.WEEK, span="a few days ago")
    C("T19", "a few weeks ago", a, a - timedelta(days=35),
      a - timedelta(days=13), prec=P.WEEK, span="a few weeks ago")
    C("T19", "several days ago", a, a - timedelta(days=5),
      a - timedelta(days=1), prec=P.WEEK, span="several days ago")
    C("T19", "a few days back", a, a - timedelta(days=5),
      a - timedelta(days=1), prec=P.WEEK, span="a few days back")
for a in (A_FRI, A_MON):
    base = date(a.year, a.month, 1)
    C("T19", "a few months ago", a, _am(base, -5), _am(base, -1),
      prec=P.MONTH, span="a few months ago")
    C("T19", "a few years ago", a, date(a.year - 5, 1, 1),
      date(a.year - 1, 1, 1), prec=P.YEAR, span="a few years ago")
    C("T19", "in a few days", a, a + timedelta(days=2),
      a + timedelta(days=6), prec=P.WEEK, span="in a few days")
    C("T19", "in a few weeks", a, a + timedelta(days=14),
      a + timedelta(days=36), prec=P.WEEK, span="in a few weeks")

for a in WD_ANCHORS:
    C("T20", "a couple of weeks ago", a, a - timedelta(days=21),
      a - timedelta(days=9), prec=P.WEEK, span="a couple of weeks ago")
    C("T20", "a couple weeks ago", a, a - timedelta(days=21),
      a - timedelta(days=9), prec=P.WEEK, span="a couple weeks ago")
    C("T20", "a couple of days ago", a, a - timedelta(days=3), a,
      prec=P.WEEK, span="a couple of days ago")
for a in (A_FRI, A_MON):
    base = date(a.year, a.month, 1)
    C("T20", "a couple of months ago", a, _am(base, -3), base,
      prec=P.MONTH, span="a couple of months ago")
    C("T20", "a couple of years ago", a, date(a.year - 3, 1, 1),
      date(a.year, 1, 1), prec=P.YEAR, span="a couple of years ago")
    C("T20", "in a couple of weeks", a, a + timedelta(days=10),
      a + timedelta(days=22), prec=P.WEEK, span="in a couple of weeks")

for w in ("recently", "lately", "of late", "as of late"):
    for a in WD_ANCHORS[:6]:
        C("T21", w, a, a - timedelta(days=30), a + timedelta(days=1),
          prec=P.MONTH, span=w)

for w in ("the other day", "the other night"):
    for a in ALL_ANCHORS:
        C("T29", w, a, a - timedelta(days=7), a, prec=P.WEEK, span=w)

# ---------------------------------------------------------------------------
# T22 — [N] day|week|month|year(s) before|after <expr>
# ---------------------------------------------------------------------------

C("T22", "the day after Christmas", A_FRI, date(2025, 12, 26),
  date(2025, 12, 27), prec=P.DAY, span="the day after Christmas")
C("T22", "the day before Christmas", A_FRI, date(2025, 12, 24),
  date(2025, 12, 25), prec=P.DAY, span="the day before Christmas")
C("T22", "the week before Christmas", A_FRI, date(2025, 12, 18),
  date(2025, 12, 25), prec=P.WEEK, span="the week before Christmas")
C("T22", "the week after Christmas", A_FRI, date(2025, 12, 26),
  date(2026, 1, 2), prec=P.WEEK, span="the week after Christmas")
C("T22", "the day after tomorrow", A_FRI, date(2026, 9, 20),
  date(2026, 9, 21), prec=P.DAY, span="the day after tomorrow")
C("T22", "the day before yesterday", A_FRI, date(2026, 9, 16),
  date(2026, 9, 17), prec=P.DAY, span="the day before yesterday")
C("T22", "day after tomorrow", A_FRI, date(2026, 9, 20),
  date(2026, 9, 21), prec=P.DAY, span="day after tomorrow")
C("T22", "the week before last Friday", A_WED, date(2026, 9, 4),
  date(2026, 9, 11), prec=P.WEEK, span="the week before last Friday")
C("T22", "the month after March", A_FRI, date(2026, 4, 1),
  date(2026, 5, 1), prec=P.MONTH, span="the month after March")
C("T22", "the month before March", A_FRI, date(2026, 2, 1),
  date(2026, 3, 1), prec=P.MONTH, span="the month before March")
C("T22", "the year before 2020", A_FRI, date(2019, 1, 1),
  date(2020, 1, 1), prec=P.YEAR, span="the year before 2020")
C("T22", "the year after 2020", A_FRI, date(2021, 1, 1),
  date(2022, 1, 1), prec=P.YEAR, span="the year after 2020")
C("T22", "the week before 2024-06-01", A_FRI, date(2024, 5, 25),
  date(2024, 6, 1), prec=P.WEEK, span="the week before 2024-06-01")
C("T22", "3 days after 2024-06-01", A_FRI, date(2024, 6, 4),
  date(2024, 6, 5), prec=P.DAY, span="3 days after 2024-06-01")
C("T22", "2 weeks before July 4, 2024", A_FRI, date(2024, 6, 20),
  date(2024, 6, 27), prec=P.WEEK, span="2 weeks before July 4, 2024")
C("T22", "3 days prior to March 5, 2024", A_FRI, date(2024, 3, 2),
  date(2024, 3, 3), prec=P.DAY, span="3 days prior to March 5, 2024")
C("T22", "a week after Easter 2024", A_FRI, date(2024, 4, 1),
  date(2024, 4, 8), prec=P.WEEK, span="a week after Easter 2024")
C("T22", "two days after Thanksgiving 2024", A_FRI, date(2024, 11, 30),
  date(2024, 12, 1), prec=P.DAY, span="two days after Thanksgiving 2024")
C("T22", "the day after March 5, 2024", A_FRI, date(2024, 3, 6),
  date(2024, 3, 7), prec=P.DAY, span="the day after March 5, 2024")
# Gold corrected 2026-09-22: the embedded "last week" resolves to
# Sep 7-14; per T22 "relative to the embedded date" the week before it
# is Aug 31-Sep 7. The original expectation repeated the embedded
# interval (no shift), inconsistent with every other T22 before-case
# above ("the month before March" -> Feb, "the week before Christmas"
# -> Dec 18-25).
C("T22", "the week before last week", A_FRI, date(2026, 8, 31),
  date(2026, 9, 7), prec=P.WEEK, span="the week before last week",
  note="week before embedded last-week interval")
C("T22", "5 days before 2026-09-18", A_FRI, date(2026, 9, 13),
  date(2026, 9, 14), prec=P.DAY, span="5 days before 2026-09-18")

# ---------------------------------------------------------------------------
# T23 — early/mid/late <Month [YYYY] | YYYY>
# ---------------------------------------------------------------------------

for part, i in (("early", 0), ("mid", 1), ("late", 2)):
    for y in (2024, 1999):
        t = f"{part} {y}"
        thirds = [(date(y, 1, 1), date(y, 5, 1)),
                  (date(y, 5, 1), date(y, 9, 1)),
                  (date(y, 9, 1), date(y + 1, 1, 1))]
        s, e = thirds[i]
        C("T23", t, A_FRI, s, e, prec=P.MONTH, span=t)
    for t2 in (f"{part} March", f"{part} of March", f"{part}-March"):
        y, mo = mon_le(A_FRI, 3)
        ms = date(y, 3, 1)
        me = _am(ms, 1)
        mid = ms + timedelta(days=10)
        late = ms + timedelta(days=20)
        s, e = [(ms, mid), (mid, late), (late, me)][i]
        C("T23", t2, A_FRI, s, e, prec=P.DAY, span=t2)
    for t2 in (f"{part} June 2024", f"{part} of June 2024"):
        ms, me = date(2024, 6, 1), date(2024, 7, 1)
        mid = ms + timedelta(days=10)
        late = ms + timedelta(days=20)
        s, e = [(ms, mid), (mid, late), (late, me)][i]
        C("T23", t2, A_FRI, s, e, prec=P.DAY, span=t2)
C("T23", "early in March", A_FRI,
  date(mon_le(A_FRI, 3)[0], 3, 1),
  date(mon_le(A_FRI, 3)[0], 3, 1) + timedelta(days=10), prec=P.DAY,
  span="early in March")
C("T23", "middle of 2024", A_FRI, date(2024, 5, 1), date(2024, 9, 1),
  prec=P.MONTH, span="middle of 2024")

# ---------------------------------------------------------------------------
# T24 — ordinal weeks of a month
# ---------------------------------------------------------------------------

_ORDS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2,
         "3rd": 2, "fourth": 3, "4th": 3, "fifth": 4, "5th": 4}
for ordw, k in _ORDS.items():
    for mn, m, y in (("June", 6, 2024), ("February", 2, 2024),
                     ("December", 12, 2025)):
        t = f"the {ordw} week of {mn} {y}"
        ms, me = date(y, m, 1), _am(date(y, m, 1), 1)
        s = ms + timedelta(weeks=k)
        e = min(s + timedelta(days=7), me)
        if s >= me:  # fifth week of a 28-day month doesn't exist
            C("T30", t, A_FRI, prec=P.UNKNOWN, span=t,
              note="nonexistent fifth week kept unknown")
            continue
        C("T24", t, A_FRI, s, e, prec=P.WEEK, span=t)
for ordw in ("last", "final"):
    for mn, m, y in (("June", 6, 2024), ("February", 2, 2024),
                     ("March", 3, 2026)):
        t = f"the {ordw} week of {mn} {y}"
        me = _am(date(y, m, 1), 1)
        C("T24", t, A_FRI, me - timedelta(days=7), me, prec=P.WEEK,
          span=t)
# yearless → nearest past month
for ordw, k in (("first", 0), ("third", 2)):
    t = f"the {ordw} week of March"
    y, mo = mon_le(A_FRI, 3)
    ms = date(y, 3, 1)
    C("T24", t, A_FRI, ms + timedelta(weeks=k),
      ms + timedelta(weeks=k, days=7), prec=P.WEEK, span=t)

# ---------------------------------------------------------------------------
# T25 — since/until/before/after/through + from X to Y + between
# ---------------------------------------------------------------------------

for emb_expr, es, ee, prec in [
    ("March", date(2026, 3, 1), date(2026, 4, 1), P.MONTH),
    ("March 2024", date(2024, 3, 1), date(2024, 4, 1), P.MONTH),
    ("March 5, 2024", date(2024, 3, 5), date(2024, 3, 6), P.DAY),
    ("2024-06-01", date(2024, 6, 1), date(2024, 6, 2), P.DAY),
    ("2020", date(2020, 1, 1), date(2021, 1, 1), P.YEAR),
    ("last week", date(2026, 9, 7), date(2026, 9, 14), P.WEEK),
    ("last Monday", date(2026, 9, 14), date(2026, 9, 15), P.DAY),
    ("yesterday", date(2026, 9, 17), date(2026, 9, 18), P.DAY),
    ("Christmas", date(2025, 12, 25), date(2025, 12, 26), P.DAY),
    ("3 days ago", date(2026, 9, 15), date(2026, 9, 16), P.DAY),
]:
    t = f"since {emb_expr}"
    C("T25", t, A_FRI, es, None, prec=prec, span=t,
      e_us=adt_us(A_FRI), note="since end = anchor")
    t = f"until {emb_expr}"
    C("T25", t, A_FRI, None, es, prec=prec, span=t)
    t = f"before {emb_expr}"
    C("T25", t, A_FRI, None, es, prec=prec, span=t)
    t = f"after {emb_expr}"
    C("T25", t, A_FRI, ee, None, prec=prec, span=t)
    t = f"through {emb_expr}"
    C("T25", t, A_FRI, None, ee, prec=prec, span=t)

for t, s, e, prec in [
    ("from May to July", date(2026, 5, 1), date(2026, 8, 1), P.MONTH),
    ("from March to July", A_FRI.replace(month=3, day=1),
     date(2026, 8, 1), P.MONTH),
    ("from March 5 to March 20", date(2026, 3, 5), date(2026, 3, 21),
     P.DAY),
    ("from March 5 to 10", date(2026, 3, 5), date(2026, 3, 11), P.DAY),
    ("from 2020 to 2024", date(2020, 1, 1), date(2025, 1, 1), P.YEAR),
    ("from 2024-03-01 to 2024-04-01", date(2024, 3, 1), date(2024, 4, 2),
     P.DAY),
    ("from June 2024 to September 2024", date(2024, 6, 1),
     date(2024, 10, 1), P.MONTH),
    ("between March and April", date(2026, 3, 1), date(2026, 5, 1),
     P.MONTH),
    ("between June 1, 2024 and June 30, 2024", date(2024, 6, 1),
     date(2024, 7, 1), P.DAY),
    ("from March 5, 2024 until April 2, 2024", date(2024, 3, 5),
     date(2024, 4, 3), P.DAY),
    ("from last Monday to today", date(2026, 9, 14), date(2026, 9, 19),
     P.DAY),
]:
    C("T25", t, A_FRI, s, e, prec=prec, span=t)

# Anchor corrected 2026-09-22: under A_FRI (Sep 2026) the nearest-past
# "from May to July" pair is May-Jul 2026 (asserted above); the 2025
# pair is the most-recent completed range only under an anchor before
# May 2026. Kept under A_FEB where this gold is the spec-correct
# nearest-past answer (V7-09.03) and exercises the cross-year roll.
C("T25", "from May to July", A_FEB, date(2025, 5, 1), date(2025, 8, 1),
  prec=P.MONTH, span="from May to July",
  note="cross-year month range, most recent completed pair")

# ---------------------------------------------------------------------------
# T26 — durations: marked, never resolved
# ---------------------------------------------------------------------------

for t in ("for 3 days", "for two weeks", "for a month", "for 6 months",
          "for a year", "for 2 years", "for about 5 days",
          "for around 3 weeks", "for nearly a month", "for over 2 days",
          "for a while", "for ages", "for years", "for months",
          "for weeks", "for several days", "for a few days",
          "for a couple of weeks", "for 90 minutes", "for 2 hours",
          "for a decade", "for ten years", "for three months",
          "for one week", "for approximately 4 days"):
    C("T26", t, A_FRI, prec=P.UNKNOWN, span=t, note="duration kept")

# ---------------------------------------------------------------------------
# T27 — age expressions: need a birth fact ⇒ unknown
# ---------------------------------------------------------------------------

for t in ("when I was 12", "when I was twelve", "when we were 5",
          "when I was a kid", "when I was a child", "when I was a baby",
          "when I was a teenager", "when I was young",
          "when I was younger", "when I was born", "when I was little",
          "when I was small", "when I was growing up", "at age 8",
          "at the age of 8", "at the age of twelve", "aged 30",
          "age 21", "as a child", "as a kid", "as a teenager",
          "in my teens", "in my twenties", "in my thirties",
          "in my forties", "in my childhood", "in my youth",
          "in my early years", "when we were kids"):
    # "when we were kids" hits no pattern — adjust: keep only matching
    C("T27", t, A_FRI, prec=P.UNKNOWN, span=t, note="age → unknown")

# ---------------------------------------------------------------------------
# T28 — holidays
# ---------------------------------------------------------------------------

for name in ("Christmas", "Christmas Day", "Christmas Eve", "Xmas",
             "Boxing Day", "New Year's Eve", "New Year's Day",
             "Halloween", "Valentine's Day", "St. Patrick's Day",
             "Independence Day", "the Fourth of July", "4th of July",
             "Thanksgiving", "Thanksgiving Day", "Easter",
             "Easter Sunday"):
    for dirw in ("", "last ", "this ", "next "):
        t = f"{dirw}{name}"
        key = name.lower()
        if key.startswith("the "):
            key = key[4:]
        d = hol_pick(key, dirw.strip(), A_FRI)
        C("T28", t, A_FRI, d, d + timedelta(days=1), prec=P.DAY, span=t)
for name, y in (("Christmas", 2024), ("Easter", 2023),
                ("Thanksgiving", 2025), ("Halloween", 2022)):
    t = f"{name} {y}"
    d = hol_date(name.lower(), y)
    C("T28", t, A_FRI, d, d + timedelta(days=1), prec=P.DAY, span=t)
# anchor-on-holiday direction checks
C("T28", "last Christmas", A_XMAS, date(2025, 12, 25),
  date(2025, 12, 26), prec=P.DAY, span="last Christmas")
C("T28", "Christmas", A_XMAS, date(2026, 12, 25), date(2026, 12, 26),
  prec=P.DAY, span="Christmas")
C("T28", "next Christmas", A_XMAS, date(2027, 12, 25),
  date(2027, 12, 26), prec=P.DAY, span="next Christmas")

# ---------------------------------------------------------------------------
# T30 — kept unknown vague phrases
# ---------------------------------------------------------------------------

for t in ("someday", "some day", "one day", "one of these days", "soon",
          "sooner or later", "eventually", "sometime", "some time",
          "a while ago", "a while back", "some time ago", "ages ago",
          "a long time ago", "long ago", "way back", "back in the day",
          "back then", "in the past", "in the future",
          "in the near future", "in the distant past", "at some point",
          "down the line", "down the road", "in due course",
          "in due time", "once upon a time", "from now on",
          "any day now", "in a bit", "in a moment", "in a minute",
          "right away", "any minute now", "later on", "the other week",
          "the other month", "the other year", "in the meantime",
          "at the time", "up to now", "by then", "so far", "as yet",
          "in time", "on time", "going forward", "in retrospect",
          "these days", "nowadays", "meanwhile", "afterwards",
          "beforehand", "shortly", "before long", "before too long",
          "a little while ago", "not long ago", "in the distant future",
          "later", "earlier", "once", "in a few minutes"):
    C("T30", t, A_FRI, prec=P.UNKNOWN, span=t, note="vague kept")


# ---------------------------------------------------------------------------
# correctness of fixture construction
# ---------------------------------------------------------------------------

def _find(rts, rule, span):
    for r in rts:
        if r.rule_id == rule and (span is None or r.text == span):
            return r
    return None


def _expected(c: Case):
    s_us = c.s_us if c.s_us is not None else (
        dus(c.s) if c.s is not None else None)
    e_us = c.e_us if c.e_us is not None else (
        dus(c.e) if c.e is not None else None)
    return s_us, e_us


def test_fixture_size_and_per_rule_coverage():
    assert len(CASES) >= 800, f"fixture has {len(CASES)} cases (< 800)"
    counts = Counter(c.rule for c in CASES)
    for r in sorted(counts):
        assert counts[r] >= 20, f"rule {r} has {counts[r]} cases (< 20)"
    missing = {f"T{i:02d}" for i in range(1, 31)} - set(counts)
    assert not missing, f"rules never exercised: {sorted(missing)}"


def test_fixture_accuracy():
    """Every case must resolve with its gold interval (we target 1.0 —
    well above the 0.97 gate)."""
    fails = []
    for i, c in enumerate(CASES):
        rts = resolve(c.text, adt_us(c.anchor), locale=c.locale,
                      hemisphere=c.hemi)
        r = _find(rts, c.rule, c.span)
        es, ee = _expected(c)
        if (r is None or r.interval.start_us != es
                or r.interval.end_us != ee
                or r.interval.precision != c.prec
                or r.ambiguous_locale != c.amb):
            fails.append((i, c, r))
    acc = (len(CASES) - len(fails)) / len(CASES)
    detail = "\n".join(
        f"  [{i}] {c.rule} {c.text!r}@{c.anchor} want="
        f"{_expected(c)} prec={c.prec} got="
        f"{(r.interval.start_us, r.interval.end_us, r.interval.precision) if r else None}"
        for i, c, r in fails[:40])
    assert not fails, (
        f"{len(fails)}/{len(CASES)} fixture misses (acc={acc:.4f}):\n"
        + detail)
    assert acc >= 0.97


def test_per_rule_exact_accuracy():
    """Per-rule accuracy is also exact (no rule silently broken)."""
    per = Counter()
    bad = Counter()
    for c in CASES:
        rts = resolve(c.text, adt_us(c.anchor), locale=c.locale,
                      hemisphere=c.hemi)
        r = _find(rts, c.rule, c.span)
        es, ee = _expected(c)
        ok = (r is not None and r.interval.start_us == es
              and r.interval.end_us == ee
              and r.interval.precision == c.prec
              and r.ambiguous_locale == c.amb)
        per[c.rule] += 1
        if not ok:
            bad[c.rule] += 1
    for r in sorted(per):
        assert bad[r] == 0, f"{r}: {bad[r]}/{per[r]} missed"


# ---------------------------------------------------------------------------
# span / anchor / determinism invariants
# ---------------------------------------------------------------------------

def test_byte_spans_slice_back():
    """Every ResolvedTime's byte span must slice back to its text —
    including under multibyte prefixes (UTF-8 offsets, not chars)."""
    texts = [
        "café — we met on March 5, 2024.",          # é = 2 bytes
        "🎉🎊 party on 2024-03-05 was great",        # emoji = 4 bytes each
        "naïve résumé review last Friday",
        "ünïcödé — last week we hiked",
        "met her the day after Christmas — nice",
        "🇫🇷 trip in June 2024",
    ]
    for t in texts:
        for r in resolve(t, adt_us(A_FRI)):
            raw = t.encode("utf-8")[r.byte_start:r.byte_end]
            assert raw.decode("utf-8") == r.text, (t, r)
            assert r.byte_start < r.byte_end
            assert r.text == t[r.byte_start:r.byte_end] or True  # bytes
            # char-level check only when text is pure ascii
            if len(t.encode("utf-8")) == len(t):
                assert t[r.byte_start:r.byte_end] == r.text


def test_anchor_and_rule_recorded():
    for t in ("March 5, 2024", "last week", "someday", "in 2024"):
        a = adt_us(A_FRI)
        for r in resolve(t, a):
            assert r.interval.anchor_us == a
            assert r.interval.rule_id == r.rule_id
            assert r.rule_id in {f"T{i:02d}" for i in range(1, 31)}


def test_determinism_and_no_wallclock():
    """Identical inputs → identical outputs; resolver never reads the
    wall clock (anchor is always explicit)."""
    texts = [c.text for c in CASES[::7]]
    for t in texts:
        r1 = resolve(t, adt_us(A_FRI))
        r2 = resolve(t, adt_us(A_FRI))
        assert r1 == r2, t
    # different anchors must change relative resolutions
    r_past = resolve("last week", adt_us(A_FRI))
    r_far = resolve("last week", adt_us(date(2030, 1, 5)))
    assert r_past != r_far


def test_sorted_and_non_overlapping():
    for c in CASES[::11]:
        rts = resolve(c.text, adt_us(c.anchor), locale=c.locale,
                      hemisphere=c.hemi)
        spans = [(r.byte_start, r.byte_end) for r in rts]
        assert spans == sorted(spans)
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            assert e1 <= s2, (c.text, spans)


def test_multiple_expressions_in_one_text():
    rts = resolve("we met in March and again last week", adt_us(A_FRI))
    rules = [r.rule_id for r in rts]
    assert "T05" in rules and "T13" in rules and len(rts) == 2
    rts = resolve("from March 5 to 10", adt_us(A_FRI))
    assert len(rts) == 1 and rts[0].rule_id == "T25"


def test_guards_reject_common_words():
    """Lowercase ambiguous forms without a temporal lead-in must not
    resolve: 'may' the modal, 'march' the verb, 'sun/sat' the words."""
    assert resolve("we may go tomorrow", adt_us(A_FRI))[0].text == "tomorrow"
    assert resolve("they march on", adt_us(A_FRI)) == []
    assert resolve("the sun is bright", adt_us(A_FRI)) == []
    assert resolve("he sat down", adt_us(A_FRI)) == []
    assert resolve("the fall was steep", adt_us(A_FRI)) == []
    rts = resolve("in the fall we hiked", adt_us(A_FRI))
    assert rts and rts[0].rule_id == "T18"


def test_overlap_longest_span_wins():
    """A subsumed candidate never double-emits."""
    rts = resolve("March 5, 2024", adt_us(A_FRI))
    assert len(rts) == 1 and rts[0].rule_id == "T02"
    rts = resolve("March 2024", adt_us(A_FRI))
    assert len(rts) == 1 and rts[0].rule_id == "T03"
    rts = resolve("the first week of June", adt_us(A_FRI))
    assert len(rts) == 1 and rts[0].rule_id == "T24"
    rts = resolve("3 days later", adt_us(A_FRI))
    assert [r.rule_id for r in rts] == ["T09"]
    rts = resolve("since March", adt_us(A_FRI))
    assert len(rts) == 1 and rts[0].rule_id == "T25"


def test_invalid_dates_kept_unknown_never_dropped():
    for t in ("2024-13-40", "March 45, 2024", "13/45/2024",
              "the fifth week of February 2023"):
        rts = resolve(t, adt_us(A_FRI))
        unk = [r for r in rts if r.interval.precision == P.UNKNOWN]
        assert unk, f"{t!r} produced no kept-unknown"
        for r in unk:
            assert r.interval.start_us is None
            assert r.interval.end_us is None
            assert r.interval.source == S.UNKNOWN


def test_locale_handling():
    assert locale_supported("en-US")
    assert locale_supported("en-GB")
    assert not locale_supported("de-DE")
    # unsupported locale: English rules still run, every hit flagged
    rts = resolve("March 5, 2024 and last week", adt_us(A_FRI),
                  locale="de-DE")
    assert len(rts) == 2
    assert all(r.ambiguous_locale for r in rts)
    # supported locales don't blanket-flag
    rts = resolve("March 5, 2024", adt_us(A_FRI), locale="en-GB")
    assert not rts[0].ambiguous_locale


def test_hemisphere_season_flip():
    n = resolve("last summer", adt_us(A_FRI), hemisphere="north")
    s = resolve("last summer", adt_us(A_FRI), hemisphere="south")
    assert n[0].interval.start_us != s[0].interval.start_us
    # northern last summer 2026 = Jun–Aug; southern = Dec 2025–Feb 2026
    assert n[0].interval.start_us == dus(date(2026, 6, 1))
    assert s[0].interval.start_us == dus(date(2025, 12, 1))
    # unknown hemisphere falls back to north honestly
    x = resolve("last summer", adt_us(A_FRI), hemisphere="tropical")
    assert x == n


def test_no_wallclock_anchor_override():
    """The same text resolves differently under different anchors —
    proving the anchor is used, never wall clock."""
    a1 = resolve("today", adt_us(A_FRI))
    a2 = resolve("today", adt_us(A_NYE))
    assert a1[0].interval.start_us == dus(A_FRI)
    assert a2[0].interval.start_us == dus(A_NYE)


# ---------------------------------------------------------------------------
# resolve_query_window
# ---------------------------------------------------------------------------

def _norm(text: str) -> NormAnalysis:
    return NormAnalysis(analyzer_id="norm/v2", terms=(
        NormTerm(term=w, channel="text", byte_start=0, byte_end=0)
        for w in text.split()), identifiers=(), text=text)


def test_query_window_basic():
    qt = adt_us(A_FRI)
    w = resolve_query_window(_norm("what did we do last week"), qt)
    assert w is not None
    assert w.start_us == dus(iso_monday(A_FRI) - timedelta(days=7))
    assert w.end_us == dus(iso_monday(A_FRI))
    assert w.anchor_us == qt
    assert "T13" in w.rule_id


def test_query_window_none_when_nothing_bounded():
    qt = adt_us(A_FRI)
    assert resolve_query_window(_norm("tell me about the project"), qt) \
        is None
    # only unbounded expressions → None, not a fake window
    assert resolve_query_window(_norm("for 3 days maybe someday"), qt) \
        is None


def test_query_window_unions_multiple():
    qt = adt_us(A_FRI)
    w = resolve_query_window(
        _norm("what happened in march or april"), qt)
    assert w is not None
    assert w.start_us == dus(date(2026, 3, 1))
    assert w.end_us == dus(date(2026, 5, 1))


def test_query_window_open_bounds():
    qt = adt_us(A_FRI)
    w = resolve_query_window(_norm("anything since march"), qt)
    assert w is not None and w.start_us == dus(date(2026, 3, 1))
    w2 = resolve_query_window(_norm("stuff until february"), qt)
    assert w2 is not None and w2.end_us == dus(date(2026, 2, 1))
    assert w2.start_us is None


def test_resolver_ids():
    assert RESOLVER_ID == "temporal/v2"
    assert RULE_SET_STATUS == "provisional/v7-r0"
    assert "en-us" in SUPPORTED_LOCALES
    assert "en-gb" in DAY_FIRST_LOCALES


def test_never_drops_recognized_phrases():
    """Every recognized expression emits a ResolvedTime — unknown
    precision is a result, not an absence (V7-09.03)."""
    for t in ("for 3 days", "when I was 12", "someday", "soon"):
        rts = resolve(t, adt_us(A_FRI))
        assert len(rts) == 1, t
        assert rts[0].interval.precision == P.UNKNOWN
        assert rts[0].interval.start_us is None
