"""Deterministic interval algebra over ``IntervalUs`` (V7-09.10, V7-09.12).

Pure-stdlib calendar arithmetic. No randomness, no wall-clock reads, no I/O:
identical inputs always produce identical outputs.

Interval semantics
------------------
- Every ``IntervalUs`` is the **half-open** interval ``[start_us, end_us)`` in
  UTC epoch microseconds (repo convention, ``TimeInterval`` in v2).
- ``None`` endpoints split into two honest cases:
    * **fully unknown** (``start_us is None and end_us is None``, T30 output):
      opaque. Every relation returns ``False``, ``duration`` returns ``nan``,
      ``intersect`` returns ``None``, ``union_span`` returns the other operand,
      ``allen_relation`` returns ``"unknown"``, ``format_interval`` returns
      ``"unknown"``, and ``shift`` returns the interval unchanged. Nothing is
      fabricated and nothing crashes.
    * **open-ended** (exactly one bound ``None``, e.g. T25 ``since``/``until``):
      the missing bound is ``-inf``/``+inf`` for comparisons and is preserved
      as ``None`` in derived intervals.
- Degenerate intervals (``end_us <= start_us``) are tolerated: they contain
  nothing, overlap nothing, and ``intersect`` returns ``None`` for them.

Conventions
-----------
- ``before``/``meets`` are strict Allen predicates: ``a.end < b.start`` /
  ``a.end == b.start``. ``after``/``met_by`` are the inverses.
- ``overlaps`` is the *any-intersection* predicate (non-empty intersection);
  Allen's narrower ``overlaps`` relation is reported by ``allen_relation``.
- ``duration(a, b, unit)`` is the **signed** elapsed time from ``a.start_us``
  to ``b.start_us`` (the canonical representative instant of each interval);
  negative when ``b`` precedes ``a``. ``gap(a, b, unit)`` measures
  ``b.start - a.end`` for "how long between the end of A and the start of B";
  ``span(iv, unit)`` measures an interval's own length.
- ``months``/``years`` are calendar-correct: the result is ``k + f`` where
  ``k`` is the number of *whole* calendar units from the earlier instant and
  ``f`` is the fraction of the in-progress unit, ``f = (t - A_k) /
  (A_{k+1} - A_k)`` with ``A_j`` = the instant ``j`` calendar units after the
  earlier endpoint. ``Jan 31 -> Feb 28`` is exactly ``1.0`` month (clamped
  one-month-later date). Fixed units (``us``/``ms``/``s``/``min``/``h``/
  ``day``/``week``) are exact ratios; ``week`` = 7 days.
- ``shift`` moves each bound independently; ``months``/``years`` shift by
  calendar with day-of-month clamping (``Jan 31 + 1mo -> Feb 28``,
  ``Feb 29 + 1y -> Feb 28``). Calendar shifts are **not** guaranteed
  invertible when clamping fires; fixed-unit shifts always are. Metadata
  (``precision``/``source``/``rule_id``/``anchor_us``) is preserved.
- ``relative(iv, n, unit, side)`` implements V7-09.10 relative phrasing:
  ``relative(day(2023,5,7), 2, "weeks", "before")`` = the day 2023-04-23,
  precision carried along.
- ``age_at_date(birth, at_us)`` returns the ordinal of the most recent
  anniversary of ``birth.start_us`` — the floor of elapsed calendar years —
  or ``None`` when the birth start is unknown. Negative values precede the
  birth. ``Feb 29`` anniversaries are observed on ``Feb 28`` in common years.
  ``age_bounds_at_date`` returns the ``(min, max)`` honest bounds over the
  birth interval's uncertainty (``None`` on an unbounded side).
- ``precision_floor`` implements V7-09.12: the coarser precision wins, ranked
  ``instant < day < week < month < season < year < decade < unknown``.
- ``intersect``/``union_span`` derive intervals with ``precision =
  precision_floor(a, b)``, ``source`` = the common source else ``UNKNOWN``,
  ``anchor_us`` = the common anchor else ``None``, ``rule_id = None`` (algebra
  is not a resolver rule).
- ``format_interval`` renders **no sharper than the declared precision**
  (V7-09.12): a month-precision interval renders ``"2023-05"``, never a day;
  multi-period spans render ``"lo..hi"`` ranges over the covered period.
"""

from __future__ import annotations

import math
import operator
from calendar import monthrange
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Union

from verbatim.core.types_v7 import IntervalUs, OccurredPrecision, OccurredSource

ALGEBRA_ID = "interval_algebra/v1"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECOND_US = 1_000_000
MINUTE_US = 60 * SECOND_US
HOUR_US = 60 * MINUTE_US
DAY_US = 24 * HOUR_US
WEEK_US = 7 * DAY_US

#: Canonical unit name -> microseconds, for fixed (non-calendar) units.
UNIT_US: dict[str, int] = {
    "us": 1,
    "ms": 1_000,
    "second": SECOND_US,
    "minute": MINUTE_US,
    "hour": HOUR_US,
    "day": DAY_US,
    "week": WEEK_US,
}

_UNIT_ALIASES: dict[str, str] = {
    "us": "us",
    "microsecond": "us",
    "microseconds": "us",
    "usec": "us",
    "ms": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    "s": "second",
    "sec": "second",
    "secs": "second",
    "second": "second",
    "seconds": "second",
    "m": "minute",
    "min": "minute",
    "mins": "minute",
    "minute": "minute",
    "minutes": "minute",
    "h": "hour",
    "hr": "hour",
    "hrs": "hour",
    "hour": "hour",
    "hours": "hour",
    "d": "day",
    "day": "day",
    "days": "day",
    "w": "week",
    "week": "week",
    "weeks": "week",
    "month": "month",
    "months": "month",
    "y": "year",
    "yr": "year",
    "yrs": "year",
    "year": "year",
    "years": "year",
}

CALENDAR_UNITS: Tuple[str, ...] = ("month", "year")
DURATION_UNITS: Tuple[str, ...] = tuple(sorted(set(_UNIT_ALIASES) | {"month", "months", "year", "years"}))

#: Fineness order for V7-09.12; higher rank = coarser.
_PRECISION_RANK: dict[OccurredPrecision, int] = {
    OccurredPrecision.INSTANT: 0,
    OccurredPrecision.DAY: 1,
    OccurredPrecision.WEEK: 2,
    OccurredPrecision.MONTH: 3,
    OccurredPrecision.SEASON: 4,
    OccurredPrecision.YEAR: 5,
    OccurredPrecision.DECADE: 6,
    OccurredPrecision.UNKNOWN: 7,
}

#: All precision grades ordered finest -> coarsest.
PRECISION_ORDER: Tuple[OccurredPrecision, ...] = tuple(
    sorted(_PRECISION_RANK, key=lambda p: _PRECISION_RANK[p])
)

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_US_TD = timedelta(microseconds=1)
_NEG_INF = float("-inf")
_POS_INF = float("inf")

ALLEN_RELATIONS: Tuple[str, ...] = (
    "before",
    "meets",
    "overlaps",
    "starts",
    "during",
    "finishes",
    "equals",
    "finished_by",
    "contains",
    "started_by",
    "overlapped_by",
    "met_by",
    "after",
)

__all__ = [
    "ALGEBRA_ID",
    "SECOND_US",
    "MINUTE_US",
    "HOUR_US",
    "DAY_US",
    "WEEK_US",
    "UNIT_US",
    "CALENDAR_UNITS",
    "DURATION_UNITS",
    "PRECISION_ORDER",
    "ALLEN_RELATIONS",
    "is_unknown",
    "is_open",
    "contains",
    "overlaps",
    "before",
    "after",
    "meets",
    "met_by",
    "allen_relation",
    "intersect",
    "union_span",
    "precision_floor",
    "midpoint",
    "duration",
    "gap",
    "span",
    "shift",
    "relative",
    "age_at_date",
    "age_bounds_at_date",
    "format_interval",
]


# ---------------------------------------------------------------------------
# Calendar helpers (exact epoch-us <-> UTC datetime, no float rounding)
# ---------------------------------------------------------------------------


def _to_dt(us: int) -> datetime:
    return _EPOCH + timedelta(microseconds=int(us))


def _to_us(dt: datetime) -> int:
    return (dt - _EPOCH) // _US_TD


def _add_months(dt: datetime, n: int) -> datetime:
    """Calendar-month shift with day-of-month clamping (Jan 31 + 1mo -> Feb 28)."""
    m0 = dt.month - 1 + n
    y = dt.year + m0 // 12
    m = m0 % 12 + 1
    d = min(dt.day, monthrange(y, m)[1])
    return dt.replace(year=y, month=m, day=d)


def _add_years(dt: datetime, n: int) -> datetime:
    """Calendar-year shift; Feb 29 -> Feb 28 on non-leap target years."""
    try:
        return dt.replace(year=dt.year + n)
    except ValueError:
        # Feb 29 onto a common year.
        return dt.replace(year=dt.year + n, month=2, day=28)


def _canon_unit(unit: str) -> str:
    try:
        key = unit.strip().lower()
    except AttributeError:
        raise ValueError(f"unit must be a string, got {unit!r}") from None
    canon = _UNIT_ALIASES.get(key)
    if canon is None:
        raise ValueError(f"unknown duration/shift unit {unit!r}")
    return canon


# ---------------------------------------------------------------------------
# Bound helpers
# ---------------------------------------------------------------------------


def is_unknown(iv: IntervalUs) -> bool:
    """Fully-unknown interval (T30): both bounds absent. Opaque to algebra."""
    return iv.start_us is None and iv.end_us is None


def is_open(iv: IntervalUs) -> bool:
    """Open-ended interval: exactly one bound present (e.g. ``since May``)."""
    return (iv.start_us is None) != (iv.end_us is None)


def _lo(iv: IntervalUs) -> float:
    return _NEG_INF if iv.start_us is None else float(iv.start_us)


def _hi(iv: IntervalUs) -> float:
    return _POS_INF if iv.end_us is None else float(iv.end_us)


def _prec(x: Union[IntervalUs, OccurredPrecision, str]) -> OccurredPrecision:
    if isinstance(x, IntervalUs):
        return x.precision if isinstance(x.precision, OccurredPrecision) else OccurredPrecision(x.precision)
    return x if isinstance(x, OccurredPrecision) else OccurredPrecision(x)


def _merge_source(a: IntervalUs, b: IntervalUs) -> OccurredSource:
    sa = a.source if isinstance(a.source, OccurredSource) else OccurredSource(a.source)
    sb = b.source if isinstance(b.source, OccurredSource) else OccurredSource(b.source)
    return sa if sa == sb else OccurredSource.UNKNOWN


def _merge_anchor(a: IntervalUs, b: IntervalUs) -> Optional[int]:
    return a.anchor_us if a.anchor_us == b.anchor_us else None


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------


def contains(outer: IntervalUs, inner: IntervalUs) -> bool:
    """``outer.start <= inner.start`` and ``inner.end <= outer.end`` (open
    bounds are infinities). ``False`` when either operand is fully unknown."""
    if is_unknown(outer) or is_unknown(inner):
        return False
    return _lo(outer) <= _lo(inner) and _hi(inner) <= _hi(outer)


def overlaps(a: IntervalUs, b: IntervalUs) -> bool:
    """Non-empty intersection: ``max(starts) < min(ends)``. Correctly ``False``
    for degenerate (empty) intervals and fully-unknown operands."""
    if is_unknown(a) or is_unknown(b):
        return False
    return max(_lo(a), _lo(b)) < min(_hi(a), _hi(b))


def before(a: IntervalUs, b: IntervalUs) -> bool:
    """Strict Allen ``before``: ``a.end < b.start``. ``False`` when the needed
    bound is open or either operand is fully unknown."""
    if is_unknown(a) or is_unknown(b):
        return False
    ea, sb = a.end_us, b.start_us
    return ea is not None and sb is not None and ea < sb


def after(a: IntervalUs, b: IntervalUs) -> bool:
    """Inverse of :func:`before`."""
    return before(b, a)


def meets(a: IntervalUs, b: IntervalUs) -> bool:
    """Allen ``meets``: ``a.end == b.start`` (a ends exactly where b begins)."""
    if is_unknown(a) or is_unknown(b):
        return False
    return a.end_us is not None and a.end_us == b.start_us


def met_by(a: IntervalUs, b: IntervalUs) -> bool:
    """Inverse of :func:`meets`."""
    return meets(b, a)


def allen_relation(a: IntervalUs, b: IntervalUs) -> str:
    """Classify the Allen interval relation of ``a`` to ``b``.

    Returns one of the 13 names in :data:`ALLEN_RELATIONS`, or ``"unknown"``
    when either operand is fully unknown. Open bounds participate as
    ``-inf``/``+inf``, so e.g. ``[May, +inf)`` vs a June day reports
    ``"contains"``.
    """
    if is_unknown(a) or is_unknown(b):
        return "unknown"
    sa, ea = _lo(a), _hi(a)
    sb, eb = _lo(b), _hi(b)
    if ea < sb:
        return "before"
    if ea == sb:
        return "meets"
    if eb < sa:
        return "after"
    if eb == sa:
        return "met_by"
    if sa == sb:
        if ea == eb:
            return "equals"
        return "started_by" if ea > eb else "starts"
    if ea == eb:
        return "finished_by" if sa < sb else "finishes"
    if sa < sb:
        return "overlaps" if ea < eb else "contains"
    # sa > sb
    return "overlapped_by" if ea > eb else "during"


# ---------------------------------------------------------------------------
# Derived intervals
# ---------------------------------------------------------------------------


def precision_floor(
    a: Union[IntervalUs, OccurredPrecision, str],
    b: Union[IntervalUs, OccurredPrecision, str],
) -> OccurredPrecision:
    """The coarser of two precision grades (V7-09.12). Accepts intervals,
    ``OccurredPrecision`` members, or their string values."""
    pa, pb = _prec(a), _prec(b)
    return pa if _PRECISION_RANK[pa] >= _PRECISION_RANK[pb] else pb


def intersect(a: IntervalUs, b: IntervalUs) -> Optional[IntervalUs]:
    """The intersection interval, or ``None`` when the operands do not overlap
    (or either is fully unknown). Open bounds propagate as ``None``."""
    if is_unknown(a) or is_unknown(b):
        return None
    starts = [v for v in (a.start_us, b.start_us) if v is not None]
    ends = [v for v in (a.end_us, b.end_us) if v is not None]
    start = max(starts) if starts else None
    end = min(ends) if ends else None
    if start is not None and end is not None and start >= end:
        return None
    return IntervalUs(
        start_us=start,
        end_us=end,
        precision=precision_floor(a, b),
        source=_merge_source(a, b),
        rule_id=None,
        anchor_us=_merge_anchor(a, b),
    )


def union_span(a: IntervalUs, b: IntervalUs) -> IntervalUs:
    """The bounding interval covering both operands. A fully-unknown operand
    contributes no bounds (union of unknown and ``x`` is ``x``); two unknowns
    return a fully-unknown interval."""
    if is_unknown(a):
        return b
    if is_unknown(b):
        return a
    # An open bound is unbounded (-inf/+inf): it wins in a union.
    start = None if (a.start_us is None or b.start_us is None) else min(a.start_us, b.start_us)
    end = None if (a.end_us is None or b.end_us is None) else max(a.end_us, b.end_us)
    return IntervalUs(
        start_us=start,
        end_us=end,
        precision=precision_floor(a, b),
        source=_merge_source(a, b),
        rule_id=None,
        anchor_us=_merge_anchor(a, b),
    )


def midpoint(iv: IntervalUs) -> Optional[int]:
    """Midpoint microsecond; ``None`` unless both bounds are known."""
    if iv.start_us is None or iv.end_us is None:
        return None
    return (iv.start_us + iv.end_us) // 2


# ---------------------------------------------------------------------------
# Durations (calendar-correct)
# ---------------------------------------------------------------------------


def _cal_between(lo_dt: datetime, hi_dt: datetime, step_months: int) -> float:
    """Whole ``step_months`` calendar units from ``lo_dt`` to ``hi_dt`` plus the
    fraction of the in-progress unit. Requires ``lo_dt <= hi_dt``."""
    def adv(j: int) -> datetime:
        return _add_months(lo_dt, j * step_months)

    k = ((hi_dt.year - lo_dt.year) * 12 + (hi_dt.month - lo_dt.month)) // step_months
    while adv(k + 1) <= hi_dt:
        k += 1
    while adv(k) > hi_dt:
        k -= 1
    cur, nxt = adv(k), adv(k + 1)
    frac = (hi_dt - cur) / (nxt - cur)
    return k + frac


def _delta_units(lo_us: int, hi_us: int, canon_unit: str) -> float:
    """Signed duration from ``lo_us`` to ``hi_us`` in ``canon_unit``."""
    delta = hi_us - lo_us
    fixed = UNIT_US.get(canon_unit)
    if fixed is not None:
        return delta / fixed
    if delta == 0:
        return 0.0
    step = 1 if canon_unit == "month" else 12
    if delta > 0:
        return _cal_between(_to_dt(lo_us), _to_dt(hi_us), step)
    return -_cal_between(_to_dt(hi_us), _to_dt(lo_us), step)


def duration(a: IntervalUs, b: IntervalUs, unit: str) -> float:
    """Signed elapsed time from ``a.start_us`` to ``b.start_us`` in ``unit``
    (positive when ``b`` starts after ``a``). ``nan`` when either start is
    unknown. Units: us/ms/second/minute/hour/day/week (exact) and
    month/year (calendar-correct)."""
    canon = _canon_unit(unit)
    if a.start_us is None or b.start_us is None:
        return math.nan
    return _delta_units(a.start_us, b.start_us, canon)


def gap(a: IntervalUs, b: IntervalUs, unit: str) -> float:
    """Signed elapsed time from ``a.end_us`` to ``b.start_us`` — the literal
    "how long between the end of A and the start of B". Negative when the
    intervals overlap; ``nan`` when either needed bound is unknown."""
    canon = _canon_unit(unit)
    if a.end_us is None or b.start_us is None:
        return math.nan
    return _delta_units(a.end_us, b.start_us, canon)


def span(iv: IntervalUs, unit: str) -> float:
    """The interval's own length in ``unit``; ``nan`` when unbounded."""
    canon = _canon_unit(unit)
    if iv.start_us is None or iv.end_us is None:
        return math.nan
    return _delta_units(iv.start_us, iv.end_us, canon)


# ---------------------------------------------------------------------------
# Shifts / relative phrasing
# ---------------------------------------------------------------------------


def shift(iv: IntervalUs, n: int, unit: str) -> IntervalUs:
    """Move both bounds by ``n`` ``unit``s (``n`` may be negative).

    Fixed units are exact integer addition; ``months``/``years`` shift by
    calendar with day-of-month clamping. Open/unknown bounds stay ``None``;
    metadata is preserved. Raises ``ValueError`` on unknown units or results
    outside the representable ``datetime`` range (years 1..9999)."""
    canon = _canon_unit(unit)
    n = operator.index(n)

    def _move(us: Optional[int]) -> Optional[int]:
        if us is None:
            return None
        if canon in UNIT_US:
            return us + n * UNIT_US[canon]
        step = 1 if canon == "month" else 12
        try:
            return _to_us(_add_months(_to_dt(us), n * step))
        except OverflowError as exc:
            raise ValueError(f"shift result out of representable range: {exc}") from exc

    return IntervalUs(
        start_us=_move(iv.start_us),
        end_us=_move(iv.end_us),
        precision=iv.precision,
        source=iv.source,
        rule_id=iv.rule_id,
        anchor_us=iv.anchor_us,
    )


def relative(iv: IntervalUs, n: int, unit: str, side: str = "before") -> IntervalUs:
    """V7-09.10 relative phrasing: ``relative(day(2023,5,7), 2, "weeks",
    "before")`` is the day-precision interval for 2023-04-23. ``side`` is
    ``"before"`` or ``"after"``."""
    if side == "before":
        return shift(iv, -n, unit)
    if side == "after":
        return shift(iv, n, unit)
    raise ValueError(f"side must be 'before' or 'after', got {side!r}")


# ---------------------------------------------------------------------------
# Age
# ---------------------------------------------------------------------------


def _age_from(born_us: int, at_us: int) -> int:
    """Ordinal of the most recent anniversary of ``born_us`` at ``at_us``."""
    b, d = _to_dt(born_us), _to_dt(at_us)
    k = d.year - b.year
    if d < _add_years(b, k):
        k -= 1
    return k


def age_at_date(birth: IntervalUs, at_us: int) -> Optional[int]:
    """Whole calendar years elapsed from ``birth.start_us`` to ``at_us``
    (floor; negative before birth). ``None`` when the birth start is unknown.
    For coarse-precision births the result is the *earliest-possible* age;
    use :func:`age_bounds_at_date` for the honest interval."""
    if birth.start_us is None:
        return None
    return _age_from(birth.start_us, at_us)


def age_bounds_at_date(
    birth: IntervalUs, at_us: int
) -> Optional[Tuple[Optional[int], Optional[int]]]:
    """``(min_age, max_age)`` over the birth interval's uncertainty: the
    earliest possible birth instant is ``start_us``, the latest is
    ``end_us - 1us``. ``None`` on an unbounded side; ``None`` overall when the
    interval is fully unknown."""
    if is_unknown(birth):
        return None
    latest = birth.end_us - 1 if birth.end_us is not None else None
    earliest = birth.start_us
    min_age = _age_from(latest, at_us) if latest is not None else None
    max_age = _age_from(earliest, at_us) if earliest is not None else None
    return (min_age, max_age)


# ---------------------------------------------------------------------------
# Rendering (never sharper than declared precision — V7-09.12)
# ---------------------------------------------------------------------------


def _fmt_ts(dt: datetime) -> str:
    base = dt.strftime("%Y-%m-%dT%H:%M:%S")
    if dt.microsecond:
        base += f".{dt.microsecond:06d}"
    return base + "Z"


def _fmt_bound(dt: datetime, precision: OccurredPrecision) -> str:
    """Format a single bound at the precision's granularity."""
    if precision == OccurredPrecision.INSTANT:
        return _fmt_ts(dt)
    if precision in (OccurredPrecision.MONTH, OccurredPrecision.SEASON):
        return dt.strftime("%Y-%m")
    if precision in (OccurredPrecision.YEAR, OccurredPrecision.DECADE):
        return dt.strftime("%Y")
    return dt.strftime("%Y-%m-%d")


def _iso_week_str(dt: datetime) -> str:
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _is_month_start(dt: datetime) -> bool:
    return (dt.day, dt.hour, dt.minute, dt.second, dt.microsecond) == (1, 0, 0, 0, 0)


def _is_year_start(dt: datetime) -> bool:
    return _is_month_start(dt) and dt.month == 1


def _is_week_start(dt: datetime) -> bool:
    return dt.weekday() == 0 and (dt.hour, dt.minute, dt.second, dt.microsecond) == (0, 0, 0, 0)


def format_interval(iv: IntervalUs) -> str:
    """Render ``iv`` honoring ``precision`` — never sharper (V7-09.12).

    - fully unknown -> ``"unknown"``; open bounds -> ``"since X"``/``"until X"``
      with the bound rendered at the precision's granularity;
    - ``instant`` -> UTC ISO-8601 timestamp (range when the span exceeds 1 s);
    - ``day`` -> ``YYYY-MM-DD`` (day range over covered days when longer);
    - ``week`` -> ISO ``YYYY-Www`` when exactly one ISO week, else day range;
    - ``month`` -> ``YYYY-MM`` (month range when longer), never a day;
    - ``season`` -> covered month range ``YYYY-MM..YYYY-MM``;
    - ``year`` -> ``YYYY`` (year range when longer);
    - ``decade`` -> ``YYYYs`` for aligned 10-year spans, else year range;
    - ``unknown`` precision -> covered day range (a factual span, no claim).
    """
    precision = iv.precision if isinstance(iv.precision, OccurredPrecision) else OccurredPrecision(iv.precision)

    if is_unknown(iv):
        return "unknown"
    if iv.start_us is None:
        # open-start: covered through end_us - 1us (end_us is not None here)
        return "until " + _fmt_bound(_to_dt(iv.end_us - 1), precision)
    if iv.end_us is None:
        return "since " + _fmt_bound(_to_dt(iv.start_us), precision)

    s_us, e_us = iv.start_us, iv.end_us
    cov_us = e_us - 1 if e_us > s_us else s_us  # last covered microsecond
    s_dt, c_dt = _to_dt(s_us), _to_dt(cov_us)

    if precision == OccurredPrecision.INSTANT:
        if e_us - s_us <= SECOND_US:
            return _fmt_ts(s_dt)
        return f"{_fmt_ts(s_dt)}..{_fmt_ts(_to_dt(e_us))}"

    if precision == OccurredPrecision.DAY or precision == OccurredPrecision.UNKNOWN:
        s_d = s_dt.strftime("%Y-%m-%d")
        c_d = c_dt.strftime("%Y-%m-%d")
        return s_d if s_d == c_d else f"{s_d}..{c_d}"

    if precision == OccurredPrecision.WEEK:
        if _is_week_start(s_dt) and e_us - s_us == WEEK_US:
            return _iso_week_str(s_dt)
        s_d = s_dt.strftime("%Y-%m-%d")
        c_d = c_dt.strftime("%Y-%m-%d")
        return s_d if s_d == c_d else f"{s_d}..{c_d}"

    if precision in (OccurredPrecision.MONTH, OccurredPrecision.SEASON):
        s_m = s_dt.strftime("%Y-%m")
        c_m = c_dt.strftime("%Y-%m")
        if precision == OccurredPrecision.MONTH and s_m == c_m:
            return s_m
        return f"{s_m}..{c_m}" if s_m != c_m else s_m

    if precision == OccurredPrecision.YEAR:
        if _is_year_start(s_dt) and _to_dt(e_us) == _add_years(s_dt, 1):
            return s_dt.strftime("%Y")
        s_y, c_y = s_dt.strftime("%Y"), c_dt.strftime("%Y")
        return s_y if s_y == c_y else f"{s_y}..{c_y}"

    if precision == OccurredPrecision.DECADE:
        if (
            _is_year_start(s_dt)
            and s_dt.year % 10 == 0
            and _to_dt(e_us) == _add_years(s_dt, 10)
        ):
            return f"{s_dt.year}s"
        s_y, c_y = s_dt.strftime("%Y"), c_dt.strftime("%Y")
        return s_y if s_y == c_y else f"{s_y}..{c_y}"

    return "unknown"
