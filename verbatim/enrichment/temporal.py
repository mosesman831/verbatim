"""Deterministic temporal enrichment — ``temporal/v1`` (§7, V5-30.10–13).

``parse_temporal(text, anchor)`` extracts explicit English temporal
expressions — ISO dates/times, month names (+day/year), weekdays,
calendar-relative words, ``N units ago`` / ``in N units``, ``since X`` —
and resolves them against an RFC3339 ``anchor`` (capture time or a
caller-supplied ``metadata.event_time``).

Contract rules honored here:

- **Ambiguity resolves to ``unknown``, never a guessed date**
  (V5-30.10). Slash dates valid under both MDY and DMY, month names
  without a temporal preposition, and expressions that fail validation
  all degrade to ``unknown`` — or are simply not produced.
- **Both the resolved interval and the anchor are recorded**
  (V5-30.11): ``event_at``/``event_end`` is a half-open interval
  ``[start, end)`` — ``end == start`` marks an instant — and
  ``anchor_at`` echoes the canonical anchor used. Changing the anchor
  changes the output; the enrichment is a recomputable derived view.
- **Status comes only from aspect markers** (V5-30.12): the expression's
  own marker (``ago`` → completed, ``tomorrow``/``in N`` → planned,
  ``since``/``this`` → ongoing) or else markers in the enclosing
  sentence; conflicting markers ⇒ ``unknown``.

Published parser assumptions (V5-30.13): English month/weekday names
(full + 3-letter abbreviations), Gregorian calendar, naive ISO times
interpreted as UTC, month-without-year resolved to the **nearest**
matching occurrence (ties prefer the past), bare weekday direction
resolved by sentence aspect markers then by the unique nearest
occurrence, "since X" binds the *most recent* occurrence of X ≤ anchor.
Unsupported/ambiguous forms report ``unknown`` rather than mis-parse.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from verbatim.memory.types import ENRICHMENT_VERSION, TimePrecision, TimeStatus
from .normalize import utf8_offsets

PARSER_VERSION = "temporal/v1"
PARSER_LOCALE = "en"      # only English month/weekday names supported
PARSER_CALENDAR = "gregorian"


@dataclass(frozen=True)
class TemporalResult:
    """Resolution of the dominant temporal expression in a record.

    ``start``/``end`` are UTF-8 byte offsets of ``expression`` in the
    original text (-1 when no expression was found). ``event_at`` /
    ``event_end`` form a half-open interval ``[start, end)`` in RFC3339:
    timestamps for instants, ``YYYY-MM-DD`` date form for day/month/year
    precisions; ``event_end == event_at`` marks a resolved instant.
    Empty strings when unresolved.
    """

    expression: str = ""
    start: int = -1
    end: int = -1
    precision: str = TimePrecision.UNKNOWN.value
    status: str = TimeStatus.UNKNOWN.value
    event_at: str = ""
    event_end: str = ""
    anchor_at: str = ""
    rule: str = "none"
    parser: str = PARSER_VERSION
    producer: str = ENRICHMENT_VERSION

    def to_dict(self) -> dict:
        return {
            "expression": self.expression,
            "start": self.start, "end": self.end,
            "precision": self.precision, "status": self.status,
            "event_at": self.event_at, "event_end": self.event_end,
            "anchor_at": self.anchor_at, "rule": self.rule,
            "parser": self.parser, "producer": self.producer,
        }


# ---------------------------------------------------------------- anchor

def _parse_anchor(anchor: str) -> datetime:
    """Strict RFC3339 parse — an explicit offset is required (naive
    anchors would bake a host timezone into a derived artifact)."""
    s = str(anchor or "").strip()
    if not s:
        raise ValueError("anchor must be an RFC3339 timestamp")
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as exc:
        raise ValueError(f"anchor is not RFC3339: {anchor!r}") from exc
    if dt.tzinfo is None:
        raise ValueError(
            f"anchor must carry an explicit offset: {anchor!r}"
        )
    return dt


def _canon(dt: datetime) -> str:
    """Canonical RFC3339 UTC form for a datetime."""
    dt = dt.astimezone(timezone.utc)
    if dt.microsecond:
        return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def _fmt_d(d) -> str:
    return d.isoformat()


# ---------------------------------------------------------------- lexicon

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
_NUM_RE = r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"


# ---------------------------------------------------------------- status

#: Sentence-level aspect markers (V5-30.12). At most one distinct status
#: may appear in the clause — conflicting markers resolve to unknown.
_PLANNED_RE = re.compile(
    r"\b(?:will|'ll|’ll|plan(?:s|ned|ning)?\s+(?:to|for|on)|"
    r"going\s+to|gonna|"
    r"intend(?:s|ed|ing)?\s+to|schedul(?:ed|es)\b|upcoming|due\b|"
    r"to\s+be\s+held|set\s+for|expect(?:ed|s|ing)?\s+to)\b",
    re.IGNORECASE,
)
_ONGOING_RE = re.compile(
    r"\b(?:since|currently|right\s+now|as\s+of|at\s+the\s+moment|still|"
    r"ongoing|in\s+progress|underway|continues?)\b",
    re.IGNORECASE,
)
_COMPLETED_RE = re.compile(
    r"\b(?:ago|yesterday|completed|finished|happened|occurred|was|were|"
    r"had\s+been|had|deployed|shipped|released|launched|landed|merged|"
    r"retired|removed|ended|wrapped|met|went|crashed|failed|broke|"
    r"spoke|saw|sent|told|got|received|"
    r"announced|rolled\s+back|rolled\s+out|done\b)\b",
    re.IGNORECASE,
)

_SENT_BOUNDARY_RE = re.compile(r"[.!?;\n]")
_SINCE_PREFIX_RE = re.compile(r"\bsince\s+$", re.IGNORECASE)


def _sentence_bounds(text: str, start: int, end: int) -> Tuple[int, int]:
    """Enclosing sentence bounds for a span (boundaries: .!?; newline)."""
    lo = 0
    for m in _SENT_BOUNDARY_RE.finditer(text[:start]):
        lo = m.end()
    hi = len(text)
    nxt = _SENT_BOUNDARY_RE.search(text, end)
    if nxt is not None:
        hi = nxt.start()
    return lo, hi


def _sentence_status(text: str, start: int, end: int) -> str:
    """Unique aspect marker in the enclosing sentence; conflicting
    markers ⇒ unknown (V5-30.12)."""
    lo, hi = _sentence_bounds(text, start, end)
    seg = text[lo:hi]
    found = set()
    if _COMPLETED_RE.search(seg):
        found.add(TimeStatus.COMPLETED.value)
    if _PLANNED_RE.search(seg):
        found.add(TimeStatus.PLANNED.value)
    if _ONGOING_RE.search(seg):
        found.add(TimeStatus.ONGOING.value)
    if len(found) == 1:
        return next(iter(found))
    return TimeStatus.UNKNOWN.value


# ---------------------------------------------------------------- dates

def _add_months(dt: datetime, months: int) -> datetime:
    """Calendar month arithmetic with day clamping (Jan 31 − 1mo =
    Dec 31 of the prior year)."""
    total = dt.year * 12 + (dt.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def _add_years(dt: datetime, years: int) -> datetime:
    try:
        return dt.replace(year=dt.year + years)
    except ValueError:  # Feb 29 → Feb 28
        return dt.replace(year=dt.year + years, day=28)


def _valid_date(y: int, m: int, d: int):
    try:
        return datetime(y, m, d).date()
    except ValueError:
        return None


class _Cand:
    """Internal resolved-expression candidate."""

    __slots__ = ("cs", "ce", "precision", "status", "event_at",
                 "event_end", "rule")

    def __init__(self, cs, ce, precision, status, event_at, event_end,
                 rule):
        self.cs = cs
        self.ce = ce
        self.precision = precision
        self.status = status
        self.event_at = event_at
        self.event_end = event_end
        self.rule = rule


# ---------------------------------------------------------------- patterns

_ISO_DT_RE = re.compile(
    r"\b(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})"
    r"(?:[Tt ](?P<h>\d{2}):(?P<mi>\d{2})"
    r"(?::(?P<s>\d{2})(?:\.(?P<us>\d{1,6}))?)?"
    r"\s*(?P<tz>[Zz]|[+-]\d{2}:?\d{2})?)?\b"
)
_ISO_MONTH_RE = re.compile(r"\b(?P<y>\d{4})-(?P<mo>\d{2})\b(?![-\d])")
_SLASH_RE = re.compile(
    r"\b(?P<a>\d{1,2})/(?P<b>\d{1,2})/(?P<y>\d{2,4})\b"
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
    rf"\b(?P<mn>{_MONTH_RE})\s+(?P<y>(?:19|20)\d{{2}})\b",
    re.IGNORECASE,
)
_PREP_MONTH_RE = re.compile(
    rf"\b(?P<prep>in|since|until|before|after|during|from|of)\s+"
    rf"(?P<mn>{_MONTH_RE})(?:\s+(?P<y>(?:19|20)\d{{2}})\b)?",
    re.IGNORECASE,
)
_IN_YEAR_RE = re.compile(
    r"\b(?P<prep>in|since|until|before|after|during)\s+"
    r"(?P<y>(?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_WEEKDAY_RE = re.compile(
    rf"\b(?:(?P<dir>last|next|this|on)\s+)?(?P<wd>{_WD_RE})\b",
    re.IGNORECASE,
)
_RELWORD_RE = re.compile(
    r"\b(?P<w>yesterday|today|tomorrow|tonight|"
    r"last\s+week|this\s+week|next\s+week|"
    r"last\s+month|this\s+month|next\s+month|"
    r"last\s+year|this\s+year|next\s+year|"
    r"last\s+weekend|next\s+weekend|"
    r"right\s+now|at\s+the\s+moment|currently|as\s+of\s+now)\b",
    re.IGNORECASE,
)
_AGO_RE = re.compile(
    rf"\b(?P<n>{_NUM_RE})\s+"
    rf"(?P<u>minutes?|hours?|days?|weeks?|months?|years?)\s+ago\b",
    re.IGNORECASE,
)
_IN_RE = re.compile(
    rf"\bin\s+(?P<n>{_NUM_RE})\s+"
    rf"(?P<u>minutes?|hours?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)


def _num(tok: str) -> int:
    tok = tok.lower()
    if tok.isdigit():
        return int(tok)
    return _NUMWORDS[tok]


def _nearest_month_day(anchor: datetime, month: int, day: int):
    """Nearest month+day occurrence to the anchor (ties prefer past).

    Candidates in anchor.year−1 … anchor.year+1; minimum absolute
    distance wins, the past wins exact ties."""
    best = None
    for y in (anchor.year - 1, anchor.year, anchor.year + 1):
        d = _valid_date(y, month, day)
        if d is None:
            continue
        delta = (d - anchor.date()).days
        if best is None or abs(delta) < abs(best[1]) or (
            abs(delta) == abs(best[1]) and delta < best[1]
        ):
            best = (d, delta)
    return best[0] if best else None


def _nearest_month(anchor: datetime, month: int):
    """(year, month) of the month-name occurrence nearest the anchor;
    ties prefer the past."""
    best = None
    for y in (anchor.year - 1, anchor.year, anchor.year + 1):
        delta = (y - anchor.year) * 12 + (month - anchor.month)
        if best is None or abs(delta) < abs(best[1]) or (
            abs(delta) == abs(best[1]) and delta < best[1]
        ):
            best = ((y, month), delta)
    return best[0]


def _recent_month(anchor: datetime, month: int):
    """Most recent occurrence of a bare month ≤ the anchor (for
    ``since``/``until`` bounds — never a future guess)."""
    year = anchor.year if month <= anchor.month else anchor.year - 1
    return (year, month)


def _month_bounds(year: int, month: int):
    start = datetime(year, month, 1).date()
    nxt = _add_months(datetime(year, month, 1), 1).date()
    return start, nxt


def _candidates(text: str, anchor: datetime) -> List[_Cand]:
    """All resolvable temporal-expression candidates (char offsets)."""
    out: List[_Cand] = []
    aday = anchor.date()

    # --- ISO datetime / date -----------------------------------------
    for m in _ISO_DT_RE.finditer(text):
        y, mo, d = int(m["y"]), int(m["mo"]), int(m["d"])
        if _valid_date(y, mo, d) is None:
            continue
        if m["h"] is not None:
            tz = m["tz"]
            if tz and tz.upper() == "Z":
                off = "+00:00"
            elif tz and ":" not in tz:
                off = tz[:3] + ":" + tz[3:]
            else:
                off = tz or "+00:00"  # naive time ⇒ UTC (published rule)
            iso = (f"{y:04d}-{mo:02d}-{d:02d}T{m['h']}:{m['mi']}"
                   f":{m['s'] or '00'}")
            if m["us"]:
                iso += "." + m["us"].ljust(6, "0")
            try:
                dt = datetime.fromisoformat(iso + off)
            except ValueError:
                continue
            out.append(_Cand(m.start(), m.end(), TimePrecision.EXACT.value,
                             "", _canon(dt), _canon(dt), "iso_datetime"))
        else:
            day = _valid_date(y, mo, d)
            out.append(_Cand(m.start(), m.end(), TimePrecision.DAY.value,
                             "", _fmt_d(day),
                             _fmt_d(day + timedelta(days=1)),
                             "iso_date"))

    # --- ISO year-month ------------------------------------------------
    for m in _ISO_MONTH_RE.finditer(text):
        y, mo = int(m["y"]), int(m["mo"])
        if not 1 <= mo <= 12:
            continue
        start, end = _month_bounds(y, mo)
        out.append(_Cand(m.start(), m.end(), TimePrecision.MONTH.value,
                         "", _fmt_d(start), _fmt_d(end), "iso_month"))

    # --- slash dates: resolve only when a single reading is valid -----
    for m in _SLASH_RE.finditer(text):
        a, b, y = int(m["a"]), int(m["b"]), int(m["y"])
        if y < 100:
            y += 2000 if y <= 69 else 1900
        mdy = _valid_date(y, a, b)      # MM/DD
        dmy = _valid_date(y, b, a)      # DD/MM
        if mdy and dmy:
            out.append(_Cand(m.start(), m.end(),
                             TimePrecision.UNKNOWN.value, "", "", "",
                             "slash_date_ambiguous"))
        else:
            d = mdy or dmy
            if d is None:
                continue
            out.append(_Cand(m.start(), m.end(), TimePrecision.DAY.value,
                             "", _fmt_d(d),
                             _fmt_d(d + timedelta(days=1)),
                             "slash_date"))

    # --- month-name forms ----------------------------------------------
    for m in _MDY_RE.finditer(text):
        month, day, year = _MONTHS[m["mn"].lower()], int(m["d"]), m["y"]
        if year:
            d = _valid_date(int(year), month, day)
            if d is None:
                continue
        else:
            d = _nearest_month_day(anchor, month, day)
            if d is None:
                continue
        out.append(_Cand(m.start(), m.end(), TimePrecision.DAY.value,
                         "", _fmt_d(d),
                         _fmt_d(d + timedelta(days=1)), "month_day"))
    for m in _DMY_RE.finditer(text):
        month, day, year = _MONTHS[m["mn"].lower()], int(m["d"]), m["y"]
        if day > 31:
            continue
        if year:
            d = _valid_date(int(year), month, day)
            if d is None:
                continue
        else:
            d = _nearest_month_day(anchor, month, day)
            if d is None:
                continue
        out.append(_Cand(m.start(), m.end(), TimePrecision.DAY.value,
                         "", _fmt_d(d), _fmt_d(d + timedelta(days=1)),
                         "day_month"))
    for m in _MY_RE.finditer(text):
        start, end = _month_bounds(int(m["y"]), _MONTHS[m["mn"].lower()])
        out.append(_Cand(m.start(), m.end(), TimePrecision.MONTH.value,
                         "", _fmt_d(start), _fmt_d(end), "month_year"))
    for m in _PREP_MONTH_RE.finditer(text):
        month = _MONTHS[m["mn"].lower()]
        prep = m["prep"].lower()
        if prep == "since":
            if m["y"]:
                y, mo = int(m["y"]), month
            else:
                y, mo = _recent_month(anchor, month)
            start, _end = _month_bounds(y, mo)
            out.append(_Cand(m.start(), m.end(), TimePrecision.MONTH.value,
                             TimeStatus.ONGOING.value, _fmt_d(start),
                             _canon(anchor), "since_month"))
        elif prep in ("in", "of", "during", "from"):
            if m["y"]:
                y, mo = int(m["y"]), month
            else:
                y, mo = _nearest_month(anchor, month)
            start, end = _month_bounds(y, mo)
            out.append(_Cand(m.start(), m.end(), TimePrecision.MONTH.value,
                             "", _fmt_d(start), _fmt_d(end),
                             "prep_month"))
        else:
            # until/before/after bound a range; the event point itself
            # stays unknown — never invent it.
            out.append(_Cand(m.start(), m.end(),
                             TimePrecision.UNKNOWN.value, "", "", "",
                             "bound_month"))

    # --- "in 2025" / "since 2020" ---------------------------------------
    for m in _IN_YEAR_RE.finditer(text):
        y = int(m["y"])
        if m["prep"].lower() == "since":
            out.append(_Cand(m.start(), m.end(), TimePrecision.YEAR.value,
                             TimeStatus.ONGOING.value, f"{y:04d}-01-01",
                             _canon(anchor), "since_year"))
        else:
            out.append(_Cand(m.start(), m.end(), TimePrecision.YEAR.value,
                             "", f"{y:04d}-01-01", f"{y + 1:04d}-01-01",
                             "prep_year"))

    # --- weekdays ------------------------------------------------------
    for m in _WEEKDAY_RE.finditer(text):
        twd = _WEEKDAYS[m["wd"].lower()]
        awd = aday.weekday()
        direction = (m["dir"] or "").lower()
        if direction == "last":
            back = (awd - twd) % 7 or 7
            delta = -back
            status = TimeStatus.COMPLETED.value
        elif direction == "next":
            fwd = (twd - awd) % 7 or 7
            delta = fwd
            status = TimeStatus.PLANNED.value
        elif direction == "this":
            delta = twd - awd
            status = ""
        else:
            # bare / "on" — direction disambiguated by the sentence's
            # own aspect markers (V5-30.12): completed markers pick the
            # most recent past occurrence, planned markers the next one;
            # otherwise the unique nearest occurrence (≤3 days either
            # way — 7 is odd, so no tie is possible).
            sstatus = _sentence_status(text, m.start(), m.end())
            if sstatus == TimeStatus.COMPLETED.value:
                delta = -((awd - twd) % 7 or 7)
            elif sstatus == TimeStatus.PLANNED.value:
                delta = (twd - awd) % 7 or 7
            else:
                delta = (twd - awd) % 7
                if delta > 3:
                    delta -= 7
            status = ""
        d = aday + timedelta(days=delta)
        rule = f"weekday_{direction or 'bare'}"
        out.append(_Cand(m.start(), m.end(), TimePrecision.DAY.value,
                         status, _fmt_d(d),
                         _fmt_d(d + timedelta(days=1)), rule))

    # --- calendar-relative words ---------------------------------------
    for m in _RELWORD_RE.finditer(text):
        w = re.sub(r"\s+", " ", m["w"].lower())
        status, precision, rule = "", "", "relword"
        ev_at = ev_end = ""
        if w == "yesterday":
            d = aday - timedelta(days=1)
            precision, status = TimePrecision.DAY.value, \
                TimeStatus.COMPLETED.value
            ev_at, ev_end = _fmt_d(d), _fmt_d(d + timedelta(days=1))
        elif w == "today":
            precision = TimePrecision.DAY.value
            ev_at = _fmt_d(aday)
            ev_end = _fmt_d(aday + timedelta(days=1))
        elif w == "tomorrow":
            d = aday + timedelta(days=1)
            precision, status = TimePrecision.DAY.value, \
                TimeStatus.PLANNED.value
            ev_at, ev_end = _fmt_d(d), _fmt_d(d + timedelta(days=1))
        elif w == "tonight":
            precision = TimePrecision.DAY.value
            ev_at = _fmt_d(aday)
            ev_end = _fmt_d(aday + timedelta(days=1))
        elif w.endswith("week"):
            monday = aday - timedelta(days=aday.weekday())
            if w.startswith("last"):
                start = monday - timedelta(days=7)
                status = TimeStatus.COMPLETED.value
            elif w.startswith("next"):
                start = monday + timedelta(days=7)
                status = TimeStatus.PLANNED.value
            else:
                start = monday
                status = TimeStatus.ONGOING.value
            precision = TimePrecision.RELATIVE.value
            ev_at = _fmt_d(start)
            ev_end = _fmt_d(start + timedelta(days=7))
        elif w.endswith("weekend"):
            sat = aday + timedelta(days=(5 - aday.weekday()) % 7)
            if w.startswith("last"):
                start = sat - timedelta(days=7)
                status = TimeStatus.COMPLETED.value
            elif w.startswith("next"):
                start = sat + timedelta(days=7)
                status = TimeStatus.PLANNED.value
            else:
                start = sat
            precision = TimePrecision.RELATIVE.value
            ev_at = _fmt_d(start)
            ev_end = _fmt_d(start + timedelta(days=2))
        elif w.endswith("month"):
            base = datetime(anchor.year, anchor.month, 1)
            if w.startswith("last"):
                start = _add_months(base, -1).date()
                status = TimeStatus.COMPLETED.value
            elif w.startswith("next"):
                start = _add_months(base, 1).date()
                status = TimeStatus.PLANNED.value
            else:
                start = base.date()
                status = TimeStatus.ONGOING.value
            precision = TimePrecision.MONTH.value
            ev_at = _fmt_d(start)
            ev_end = _fmt_d(_add_months(
                datetime(start.year, start.month, 1), 1).date())
        elif w.endswith("year"):
            if w.startswith("last"):
                y = anchor.year - 1
                status = TimeStatus.COMPLETED.value
            elif w.startswith("next"):
                y = anchor.year + 1
                status = TimeStatus.PLANNED.value
            else:
                y = anchor.year
                status = TimeStatus.ONGOING.value
            precision = TimePrecision.YEAR.value
            ev_at, ev_end = f"{y:04d}-01-01", f"{y + 1:04d}-01-01"
        else:  # right now / at the moment / currently / as of now
            precision = TimePrecision.RELATIVE.value
            status = TimeStatus.ONGOING.value
            ev_at = ev_end = _canon(anchor)
        out.append(_Cand(m.start(), m.end(), precision, status,
                         ev_at, ev_end, rule))

    # --- N units ago / in N units --------------------------------------
    for m in _AGO_RE.finditer(text):
        n, unit = _num(m["n"]), m["u"].lower().rstrip("s")
        cand = _shift(anchor, -n, unit, m.start(), m.end(), "ago",
                      TimeStatus.COMPLETED.value)
        if cand:
            out.append(cand)
    for m in _IN_RE.finditer(text):
        n, unit = _num(m["n"]), m["u"].lower().rstrip("s")
        cand = _shift(anchor, n, unit, m.start(), m.end(), "in_units",
                      TimeStatus.PLANNED.value)
        if cand:
            out.append(cand)

    # --- "since <expr>" → ongoing interval bound -----------------------
    for c in out:
        if c.rule.startswith("since_"):
            continue
        prefix = text[:c.cs]
        if _SINCE_PREFIX_RE.search(prefix):
            c.status = TimeStatus.ONGOING.value
            c.event_end = _canon(anchor)
            c.rule = "since_" + c.rule

    return out


def _shift(anchor: datetime, n: int, unit: str, cs: int, ce: int,
           rule: str, status: str) -> Optional[_Cand]:
    """Resolve ``n`` units back/forward from the anchor."""
    if unit in ("minute", "hour"):
        delta = timedelta(minutes=n) if unit == "minute" \
            else timedelta(hours=n)
        dt = anchor + delta
        return _Cand(cs, ce, TimePrecision.RELATIVE.value, status,
                     _canon(dt), _canon(dt), rule)
    if unit == "day":
        d = anchor.date() + timedelta(days=n)
    elif unit == "week":
        d = anchor.date() + timedelta(weeks=n)
    elif unit == "month":
        d = _add_months(anchor, n).date()
    elif unit == "year":
        d = _add_years(anchor, n).date()
    else:
        return None
    return _Cand(cs, ce, TimePrecision.DAY.value, status,
                 _fmt_d(d), _fmt_d(d + timedelta(days=1)), rule)


_PRECISION_RANK = {
    TimePrecision.EXACT.value: 0,
    TimePrecision.DAY.value: 1,
    TimePrecision.MONTH.value: 2,
    TimePrecision.YEAR.value: 3,
    TimePrecision.RELATIVE.value: 4,
    TimePrecision.UNKNOWN.value: 5,
}


def _resolve_temporal_overlaps(cands: List[_Cand]) -> List[_Cand]:
    """Overlapping candidates (e.g. "March 2025" ⊃ "in March"…) keep the
    longest span; ties prefer the better precision then earliest start."""
    order = sorted(
        cands,
        key=lambda c: (-(c.ce - c.cs), _PRECISION_RANK[c.precision],
                       c.cs),
    )
    kept: List[_Cand] = []
    for c in order:
        if any(c.cs < k.ce and k.cs < c.ce for k in kept):
            continue
        kept.append(c)
    return kept


def parse_temporal(text: str, anchor: str) -> TemporalResult:
    """Resolve the dominant temporal expression in ``text``.

    ``anchor`` is an RFC3339 timestamp (capture time or explicit
    ``metadata.event_time``); it is echoed back as ``anchor_at`` so the
    resolution can be invalidated+recomputed when the anchor changes
    (V5-30.11). When several expressions appear, the most precise wins
    (ties: earliest); when none resolve, precision/status are ``unknown``
    and no date is invented (V5-30.10).
    """
    t = str(text if text is not None else "")
    anchor_dt = _parse_anchor(anchor)
    anchor_at = _canon(anchor_dt)

    cands = _resolve_temporal_overlaps(_candidates(t, anchor_dt))
    if not cands:
        return TemporalResult(anchor_at=anchor_at)

    best = min(
        cands,
        key=lambda c: (_PRECISION_RANK[c.precision], c.cs),
    )
    status = best.status or _sentence_status(t, best.cs, best.ce)

    # Sanity: an interval wholly in the future can never be "completed";
    # conflicting evidence yields unknown rather than a guess.
    if (
        status == TimeStatus.COMPLETED.value
        and best.event_at
        and best.event_at[:10] > anchor_dt.date().isoformat()
    ):
        status = TimeStatus.UNKNOWN.value

    offsets = utf8_offsets(t)
    return TemporalResult(
        expression=t[best.cs:best.ce],
        start=offsets[best.cs],
        end=offsets[best.ce],
        precision=best.precision,
        status=status,
        event_at=best.event_at,
        event_end=best.event_end,
        anchor_at=anchor_at,
        rule=best.rule,
    )
