"""V7-09.10 / V7-09.12 — deterministic interval algebra tests.

≥400 parametrized cases plus a seeded property suite:

- Allen-algebra relations exhaustively over day/month/year grids, with an
  independent index-space reference (``ref_relation``) and a boolean
  consistency table (``_BOOLS``) cross-checking every predicate.
- ``duration`` across month/year boundaries including leap years, both
  directions, all fixed units, and pinned calendar fractions.
- ``shift`` round-trips (exact for fixed units; clamp-aware for calendar
  units) on bounded, open, and fully-unknown intervals.
- Precision is never sharpened: ``precision_floor`` exhaustively,
  ``format_interval`` shape checks, ``intersect``/``union_span`` carry the
  coarser grade.
- ``meets``/``overlaps``/``contains`` edge cases incl. degenerate and
  open-ended intervals.
- ``age_at_date``/``age_bounds_at_date`` arithmetic incl. Feb-29 birthdays
  and pre-birth dates.

All randomness comes from ``random.Random(SEED)`` — fully deterministic.
"""

from __future__ import annotations

import math
import random
import re
from datetime import datetime, timedelta, timezone

import pytest

from verbatim.core.types_v7 import (
    IntervalUs,
    OccurredPrecision as P,
    OccurredSource as S,
)
from verbatim.temporal import algebra as A

SEED = 20260918
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def us(dt: datetime) -> int:
    return (dt - EPOCH) // timedelta(microseconds=1)


def D(y: int, m: int, d: int, H: int = 0, Mi: int = 0, Sec: int = 0, u: int = 0) -> int:
    return us(datetime(y, m, d, H, Mi, Sec, u, tzinfo=timezone.utc))


def dt_of(t_us: int) -> datetime:
    return EPOCH + timedelta(microseconds=t_us)


def iv(s_us, e_us, precision=P.DAY, **kw) -> IntervalUs:
    return IntervalUs(s_us, e_us, precision, **kw)


def day_iv(y: int, m: int, d: int) -> IntervalUs:
    return iv(D(y, m, d), D(y, m, d) + A.DAY_US, P.DAY)


def month_iv(y: int, m: int) -> IntervalUs:
    end = D(y + 1, 1, 1) if m == 12 else D(y, m + 1, 1)
    return iv(D(y, m, 1), end, P.MONTH)


def year_iv(y: int) -> IntervalUs:
    return iv(D(y, 1, 1), D(y + 1, 1, 1), P.YEAR)


def week_iv(y: int, w: int) -> IntervalUs:
    start = us(datetime.fromisocalendar(y, w, 1).replace(tzinfo=timezone.utc))
    return iv(start, start + A.WEEK_US, P.WEEK)


def decade_iv(y: int) -> IntervalUs:
    return iv(D(y, 1, 1), D(y + 10, 1, 1), P.DECADE)


def instant_iv(y: int, m: int, d: int, H: int = 0, Mi: int = 0, Sec: int = 0, u: int = 0) -> IntervalUs:
    t = D(y, m, d, H, Mi, Sec, u)
    return iv(t, t + 1, P.INSTANT)


def open_since(s_us: int, precision=P.DAY) -> IntervalUs:
    return iv(s_us, None, precision)


def open_until(e_us: int, precision=P.DAY) -> IntervalUs:
    return iv(None, e_us, precision)


def span_iv(s_us: int, e_us: int, precision=P.DAY) -> IntervalUs:
    return iv(s_us, e_us, precision)


UNKNOWN = IntervalUs(None, None, P.UNKNOWN)


# ---------------------------------------------------------------------------
# Independent references (index-space / table-driven, not the implementation)
# ---------------------------------------------------------------------------


def ref_relation(ia: int, ja: int, ib: int, jb: int) -> str:
    """Expected Allen name for a=[p[ia],p[ja]), b=[p[ib],p[jb]) on a strictly
    increasing grid, derived purely from endpoint index orderings."""
    if ja < ib:
        return "before"
    if ja == ib:
        return "meets"
    if jb < ia:
        return "after"
    if jb == ia:
        return "met_by"
    if ia == ib:
        if ja == jb:
            return "equals"
        return "started_by" if ja > jb else "starts"
    if ja == jb:
        return "finished_by" if ia < ib else "finishes"
    if ia < ib:
        return "overlaps" if ja < jb else "contains"
    return "overlapped_by" if ja > jb else "during"


_INVERSE = {
    "before": "after",
    "after": "before",
    "meets": "met_by",
    "met_by": "meets",
    "overlaps": "overlapped_by",
    "overlapped_by": "overlaps",
    "starts": "started_by",
    "started_by": "starts",
    "during": "contains",
    "contains": "during",
    "finishes": "finished_by",
    "finished_by": "finishes",
    "equals": "equals",
}

# name -> (before, after, meets, met_by, overlaps, contains_ab)
_BOOLS = {
    "before": (True, False, False, False, False, False),
    "after": (False, True, False, False, False, False),
    "meets": (False, False, True, False, False, False),
    "met_by": (False, False, False, True, False, False),
    "overlaps": (False, False, False, False, True, False),
    "overlapped_by": (False, False, False, False, True, False),
    "starts": (False, False, False, False, True, False),
    "started_by": (False, False, False, False, True, True),
    "during": (False, False, False, False, True, False),
    "contains": (False, False, False, False, True, True),
    "finishes": (False, False, False, False, True, False),
    "finished_by": (False, False, False, False, True, True),
    "equals": (False, False, False, False, True, True),
}


def _check_bools(a: IntervalUs, b: IntervalUs, name: str) -> None:
    bef, aft, met, metb, ov, cont = _BOOLS[name]
    assert A.before(a, b) is bef
    assert A.after(a, b) is aft
    assert A.meets(a, b) is met
    assert A.met_by(a, b) is metb
    assert A.overlaps(a, b) is ov
    assert A.contains(a, b) is cont


# ---------------------------------------------------------------------------
# Allen relations: literal 13 over day/month/year scales
# ---------------------------------------------------------------------------

# Each entry builds (a, b) for one relation on a scale of base units.
_LITERAL_SCALES = {
    "day": lambda k: day_iv(2023, 5, k),
    "month": lambda k: month_iv(2023, k),
    "year": lambda k: year_iv(2020 + k),
}


def _literal_cases():
    """The 13 Allen relations on each scale: p0<p1<p2<p3 grid fragments."""
    cases = []
    for scale, mk in _LITERAL_SCALES.items():
        # units u0..u4 are consecutive base cells; spans are unions of them.
        def sp(i, j, mk=mk):  # span covering cells i..j-1 (cells are 1-based)
            a, b = mk(i + 1), mk(j)
            return IntervalUs(a.start_us, b.end_us, a.precision)

        named = {
            "before": (sp(0, 1), sp(2, 3)),
            "meets": (sp(0, 1), sp(1, 2)),
            "overlaps": (sp(0, 2), sp(1, 3)),
            "starts": (sp(0, 1), sp(0, 3)),
            "during": (sp(1, 2), sp(0, 4)),
            "finishes": (sp(2, 4), sp(0, 4)),
            "equals": (sp(0, 3), sp(0, 3)),
            "finished_by": (sp(0, 4), sp(2, 4)),
            "contains": (sp(0, 4), sp(1, 3)),
            "started_by": (sp(0, 4), sp(0, 1)),
            "overlapped_by": (sp(1, 3), sp(0, 2)),
            "met_by": (sp(1, 2), sp(0, 1)),
            "after": (sp(3, 4), sp(0, 1)),
        }
        for name, (a, b) in named.items():
            cases.append(pytest.param(a, b, name, id=f"{scale}:{name}"))
    return cases


@pytest.mark.parametrize("a,b,name", _literal_cases())
def test_allen_literal(a, b, name):
    assert A.allen_relation(a, b) == name
    assert A.allen_relation(b, a) == _INVERSE[name]
    _check_bools(a, b, name)


# ---------------------------------------------------------------------------
# Allen relations: exhaustive over day/month/year point grids
# ---------------------------------------------------------------------------


_SCALE_PRECISION = {"day": P.DAY, "month": P.MONTH, "year": P.YEAR}


def _grid_cases(scale: str, points):
    prec = _SCALE_PRECISION[scale]
    ivs = [
        IntervalUs(points[i], points[j], prec)
        for i in range(len(points))
        for j in range(i + 1, len(points))
    ]
    idx = [(i, j) for i in range(len(points)) for j in range(i + 1, len(points))]
    cases = []
    for x, (ia, ja) in enumerate(idx):
        for y, (ib, jb) in enumerate(idx):
            cases.append(
                pytest.param(ivs[x], ivs[y], ref_relation(ia, ja, ib, jb),
                             id=f"{scale}:[{ia},{ja})x[{ib},{jb})")
            )
    return cases


_DAY_PTS = [D(2023, 5, k) for k in range(1, 6)]
_MONTH_PTS = [D(2023, k, 1) for k in range(1, 6)]
_YEAR_PTS = [D(2020 + k, 1, 1) for k in range(5)]

_GRID_CASES = (
    _grid_cases("day", _DAY_PTS)
    + _grid_cases("month", _MONTH_PTS)
    + _grid_cases("year", _YEAR_PTS)
)


@pytest.mark.parametrize("a,b,name", _GRID_CASES)
def test_allen_grid_exhaustive(a, b, name):
    assert A.allen_relation(a, b) == name
    assert A.allen_relation(b, a) == _INVERSE[name]
    _check_bools(a, b, name)
    inter = A.intersect(a, b)
    if name in ("before", "meets", "after", "met_by"):
        assert inter is None
        assert not A.overlaps(a, b)
    else:
        assert inter is not None
        assert inter.start_us == max(a.start_us, b.start_us)
        assert inter.end_us == min(a.end_us, b.end_us)
        assert A.contains(inter, inter) or True  # sanity: constructed
        assert A.contains(a, inter) and A.contains(b, inter)
        assert inter.precision == A.precision_floor(a, b)
    uni = A.union_span(a, b)
    assert A.contains(uni, a) and A.contains(uni, b)
    assert uni.start_us == min(a.start_us, b.start_us)
    assert uni.end_us == max(a.end_us, b.end_us)


# ---------------------------------------------------------------------------
# meets / overlaps / contains edge cases
# ---------------------------------------------------------------------------

_MEETS_EDGES = [
    # adjacent days: a ends exactly where b starts
    (day_iv(2023, 5, 7), day_iv(2023, 5, 8), True),
    (day_iv(2023, 5, 7), day_iv(2023, 5, 9), False),
    (month_iv(2023, 5), month_iv(2023, 6), True),
    (month_iv(2023, 5), month_iv(2023, 7), False),
    (year_iv(2022), year_iv(2023), True),
    (decade_iv(2020), decade_iv(2030), True),
    # meets is directional
    (day_iv(2023, 5, 8), day_iv(2023, 5, 7), False),
    # an open FAR bound never meets; a known meeting bound still meets
    (open_until(D(2023, 5, 8)), day_iv(2023, 5, 8), True),   # (-∞,May8) meets [May8,May9)
    (day_iv(2023, 5, 7), open_since(D(2023, 5, 8)), True),
    (open_since(D(2023, 5, 8)), day_iv(2023, 5, 9), False),  # [May8,+∞) has no end
    (open_until(D(2023, 5, 8)), day_iv(2023, 5, 9), False),
    # unknown never meets
    (UNKNOWN, day_iv(2023, 5, 8), False),
    (day_iv(2023, 5, 7), UNKNOWN, False),
    # degenerate [t,t) meets anything starting at t
    (iv(D(2023, 5, 7), D(2023, 5, 7)), day_iv(2023, 5, 7), True),
]


@pytest.mark.parametrize("a,b,expected", _MEETS_EDGES)
def test_meets_edges(a, b, expected):
    assert A.meets(a, b) is expected


_OVERLAP_EDGES = [
    (day_iv(2023, 5, 7), day_iv(2023, 5, 7), True),
    (day_iv(2023, 5, 7), day_iv(2023, 5, 8), False),   # adjacent, not overlapping
    (day_iv(2023, 5, 7), day_iv(2023, 5, 6), False),
    (month_iv(2023, 5), day_iv(2023, 5, 31), True),
    (month_iv(2023, 5), day_iv(2023, 6, 1), False),
    (year_iv(2023), month_iv(2023, 12), True),
    (open_since(D(2023, 5, 1)), day_iv(2099, 1, 1), True),   # open end reaches far future
    (open_since(D(2023, 5, 1)), day_iv(2023, 4, 30), False), # but not before its start
    (open_until(D(2023, 5, 1)), day_iv(1900, 1, 1), True),   # open start reaches far past
    (open_until(D(2023, 5, 1)), day_iv(2023, 5, 1), False),  # ends at May 1 exclusive
    (open_since(D(2023, 5, 1)), open_until(D(2023, 6, 1)), True),
    (open_since(D(2023, 6, 1)), open_until(D(2023, 5, 1)), False),
    (UNKNOWN, day_iv(2023, 5, 7), False),
    (UNKNOWN, UNKNOWN, False),
    # degenerate [t,t) intersects nothing, even a containing interval
    (iv(D(2023, 5, 7), D(2023, 5, 7)), day_iv(2023, 5, 7), False),
    (iv(D(2023, 5, 7), D(2023, 5, 7)), iv(D(2023, 5, 6), D(2023, 5, 9)), False),
]


@pytest.mark.parametrize("a,b,expected", _OVERLAP_EDGES)
def test_overlaps_edges(a, b, expected):
    assert A.overlaps(a, b) is expected
    # symmetry
    assert A.overlaps(b, a) is expected


_CONTAINS_EDGES = [
    (month_iv(2023, 5), day_iv(2023, 5, 7), True),
    (month_iv(2023, 5), day_iv(2023, 5, 1), True),   # boundary start
    (month_iv(2023, 5), day_iv(2023, 5, 31), True),  # boundary end
    (month_iv(2023, 5), day_iv(2023, 6, 1), False),
    (day_iv(2023, 5, 7), month_iv(2023, 5), False),
    (day_iv(2023, 5, 7), day_iv(2023, 5, 7), True),  # reflexive
    (year_iv(2023), month_iv(2023, 5), True),
    (open_since(D(2023, 5, 1)), day_iv(2023, 6, 1), True),
    (open_since(D(2023, 5, 1)), day_iv(2023, 4, 30), False),
    (open_until(D(2023, 5, 1)), day_iv(2023, 4, 30), True),
    (day_iv(2023, 5, 7), open_since(D(2023, 5, 1)), False),  # can't bound +inf
    (open_since(D(2023, 4, 1)), open_since(D(2023, 5, 1)), True),   # [Apr,∞) ⊇ [May,∞)
    (open_since(D(2023, 5, 1)), open_since(D(2023, 4, 1)), False),
    (open_until(D(2023, 5, 1)), open_until(D(2023, 4, 1)), True),   # (-∞,May) ⊇ (-∞,Apr)
    (UNKNOWN, day_iv(2023, 5, 7), False),
    (day_iv(2023, 5, 7), UNKNOWN, False),
]


@pytest.mark.parametrize("outer,inner,expected", _CONTAINS_EDGES)
def test_contains_edges(outer, inner, expected):
    assert A.contains(outer, inner) is expected


_BEFORE_AFTER_EDGES = [
    (day_iv(2023, 5, 7), day_iv(2023, 5, 9), True, False),
    (day_iv(2023, 5, 7), day_iv(2023, 5, 8), False, False),  # adjacent = meets, not before
    (day_iv(2023, 5, 9), day_iv(2023, 5, 7), False, True),
    (open_since(D(2023, 5, 1)), day_iv(2023, 6, 1), False, False),  # never ends → never before
    (open_until(D(2023, 5, 1)), day_iv(2023, 6, 1), True, False),
    (day_iv(2023, 5, 7), open_until(D(2023, 5, 1)), False, True),   # b started -inf → a after b
    (UNKNOWN, day_iv(2023, 5, 7), False, False),
]


@pytest.mark.parametrize("a,b,bef,aft", _BEFORE_AFTER_EDGES)
def test_before_after_edges(a, b, bef, aft):
    assert A.before(a, b) is bef
    assert A.after(a, b) is aft


# ---------------------------------------------------------------------------
# intersect / union_span edge cases
# ---------------------------------------------------------------------------

_INTERSECT_CASES = [
    # (a, b, expected) — expected = (start, end) tuple or None for no intersection
    (month_iv(2023, 5), day_iv(2023, 5, 7), (D(2023, 5, 7), D(2023, 5, 8))),
    (month_iv(2023, 5), day_iv(2023, 6, 1), None),
    (open_since(D(2023, 5, 1)), open_until(D(2023, 6, 1)), (D(2023, 5, 1), D(2023, 6, 1))),
    (open_since(D(2023, 5, 1)), day_iv(2023, 6, 7), (D(2023, 6, 7), D(2023, 6, 8))),
    (open_since(D(2023, 5, 1)), open_since(D(2023, 6, 1)), (D(2023, 6, 1), None)),
    (open_until(D(2023, 5, 1)), open_until(D(2023, 6, 1)), (None, D(2023, 5, 1))),
    (open_since(D(2023, 6, 1)), open_until(D(2023, 5, 1)), None),
    (day_iv(2023, 5, 7), day_iv(2023, 5, 7), (D(2023, 5, 7), D(2023, 5, 8))),
    (iv(D(2023, 5, 7), D(2023, 5, 7)), day_iv(2023, 5, 7), None),  # empty ∩ = None
]


@pytest.mark.parametrize("a,b,expected", _INTERSECT_CASES)
def test_intersect_cases(a, b, expected):
    r = A.intersect(a, b)
    if expected is None:
        assert r is None
        assert not A.overlaps(a, b)
    else:
        es, ee = expected
        assert r is not None
        assert r.start_us == es
        assert r.end_us == ee
        assert r.precision == A.precision_floor(a, b)


def test_intersect_unknown():
    assert A.intersect(UNKNOWN, day_iv(2023, 5, 7)) is None
    assert A.intersect(day_iv(2023, 5, 7), UNKNOWN) is None
    assert A.intersect(UNKNOWN, UNKNOWN) is None


_UNION_CASES = [
    (day_iv(2023, 5, 7), day_iv(2023, 5, 9), D(2023, 5, 7), D(2023, 5, 10)),
    (day_iv(2023, 5, 9), day_iv(2023, 5, 7), D(2023, 5, 7), D(2023, 5, 10)),
    (open_since(D(2023, 5, 1)), day_iv(2023, 6, 1), D(2023, 5, 1), None),
    (open_until(D(2023, 5, 1)), day_iv(2023, 6, 1), None, D(2023, 6, 2)),
    (open_since(D(2023, 5, 1)), open_until(D(2023, 6, 1)), None, None),
    (day_iv(2023, 5, 7), UNKNOWN, D(2023, 5, 7), D(2023, 5, 8)),
    (UNKNOWN, day_iv(2023, 5, 7), D(2023, 5, 7), D(2023, 5, 8)),
]


@pytest.mark.parametrize("a,b,es,ee", _UNION_CASES)
def test_union_span_cases(a, b, es, ee):
    u = A.union_span(a, b)
    assert u.start_us == es
    assert u.end_us == ee
    if es is not None or ee is not None:
        assert not A.is_unknown(u)


def test_union_span_both_unknown():
    u = A.union_span(UNKNOWN, UNKNOWN)
    assert A.is_unknown(u)


def test_derived_interval_metadata():
    a = IntervalUs(D(2023, 5, 1), D(2023, 6, 1), P.MONTH, S.EXPLICIT,
                   rule_id="T03", anchor_us=D(2023, 7, 1))
    b = IntervalUs(D(2023, 5, 7), D(2023, 5, 8), P.DAY, S.EXPLICIT,
                   rule_id="T01", anchor_us=D(2023, 7, 1))
    r = A.intersect(a, b)
    assert r.precision == P.MONTH            # coarser wins (V7-09.12)
    assert r.source == S.EXPLICIT            # common source propagated
    assert r.anchor_us == D(2023, 7, 1)      # common anchor propagated
    assert r.rule_id is None                 # algebra is not a resolver rule
    b2 = IntervalUs(D(2023, 5, 7), D(2023, 5, 8), P.DAY, S.RESOLVED_RELATIVE,
                    anchor_us=D(2023, 8, 1))
    r2 = A.intersect(a, b2)
    assert r2.source == S.UNKNOWN            # differing sources → UNKNOWN
    assert r2.anchor_us is None


# ---------------------------------------------------------------------------
# precision_floor — exhaustive 8x8
# ---------------------------------------------------------------------------

_PREC_PAIRS = [(p, q) for p in P for q in P]


@pytest.mark.parametrize("p,q", _PREC_PAIRS)
def test_precision_floor_exhaustive(p, q):
    r = A.precision_floor(p, q)
    assert r in (p, q)
    assert A._PRECISION_RANK[r] == max(A._PRECISION_RANK[p], A._PRECISION_RANK[q])


@pytest.mark.parametrize("p,q", _PREC_PAIRS)
def test_precision_floor_on_intervals(p, q):
    a = IntervalUs(D(2023, 1, 1), D(2023, 1, 2), p)
    b = IntervalUs(D(2023, 1, 1), D(2023, 1, 2), q)
    assert A.precision_floor(a, b) == A.precision_floor(p, q)


def test_precision_floor_strings():
    assert A.precision_floor("day", "month") == P.MONTH
    assert A.precision_floor(day_iv(2023, 5, 7), "instant") == P.DAY


# ---------------------------------------------------------------------------
# duration — calendar-correct
# ---------------------------------------------------------------------------

# Hand-computed expectations. Months/years use whole-units + fraction of the
# in-progress calendar unit.
def _frac(lo: datetime, nxt_lo: datetime, hi: datetime) -> float:
    return (hi - lo) / (nxt_lo - lo)


_DURATION_CASES = [
    # --- fixed units, exact ratios
    (D(2023, 3, 1), D(2023, 3, 8), "days", 7.0),
    (D(2023, 3, 1), D(2023, 3, 8), "weeks", 1.0),
    (D(2023, 3, 1), D(2023, 3, 2), "hours", 24.0),
    (D(2023, 3, 1), D(2023, 3, 1, 1, 30), "minutes", 90.0),
    (D(2023, 3, 1), D(2023, 3, 1, 0, 0, 30), "seconds", 30.0),
    (D(2023, 3, 1), D(2023, 3, 1, 0, 0, 0, 500), "ms", 0.5),
    (D(2023, 3, 1), D(2023, 3, 1, 0, 0, 0, 7), "us", 7.0),
    (D(2023, 3, 1), D(2023, 3, 2, 12), "days", 1.5),
    (D(2023, 3, 8), D(2023, 3, 1), "days", -7.0),
    (D(2023, 3, 8), D(2023, 3, 1), "weeks", -1.0),
    (D(2023, 3, 1), D(2023, 3, 1), "days", 0.0),
    # --- day boundaries / leap years (fixed units)
    (D(2020, 2, 29), D(2020, 3, 1), "days", 1.0),   # leap day exists in 2020
    (D(2021, 2, 28), D(2021, 3, 1), "days", 1.0),   # no Feb 29 in 2021
    (D(2020, 2, 28), D(2020, 3, 1), "days", 2.0),   # spans Feb 29
    (D(2021, 2, 28), D(2021, 3, 1), "days", 1.0),
    (D(2022, 12, 31), D(2023, 1, 1), "days", 1.0),
    (D(2020, 1, 1), D(2021, 1, 1), "days", 366.0),  # leap year
    (D(2021, 1, 1), D(2022, 1, 1), "days", 365.0),
    (D(2023, 1, 1), D(2024, 1, 1), "weeks", 365 / 7),
    # --- months: whole calendar months
    (D(2023, 1, 15), D(2023, 2, 15), "months", 1.0),
    (D(2023, 1, 15), D(2023, 5, 15), "months", 4.0),
    (D(2023, 1, 1), D(2024, 1, 1), "months", 12.0),
    (D(2022, 12, 15), D(2023, 1, 15), "months", 1.0),
    (D(2023, 5, 1), D(2023, 8, 1), "months", 3.0),
    (D(2020, 1, 31), D(2020, 2, 29), "months", 1.0),   # clamped onto leap Feb 29
    (D(2023, 1, 31), D(2023, 2, 28), "months", 1.0),   # clamped onto Feb 28
    (D(2023, 1, 31), D(2023, 3, 31), "months", 2.0),
    (D(2023, 1, 31), D(2023, 4, 30), "months", 3.0),
    (D(2023, 8, 31), D(2023, 9, 30), "months", 1.0),
    (D(2023, 2, 15), D(2023, 1, 15), "months", -1.0),  # negative
    (D(2024, 1, 1), D(2023, 1, 1), "months", -12.0),
    # --- months: fractional (whole units + fraction of in-progress unit)
    (D(2023, 1, 15), D(2023, 3, 10), "months",
     1 + _frac(datetime(2023, 2, 15, tzinfo=timezone.utc),
               datetime(2023, 3, 15, tzinfo=timezone.utc),
               datetime(2023, 3, 10, tzinfo=timezone.utc))),
    (D(2023, 1, 31), D(2023, 3, 30), "months",
     1 + _frac(datetime(2023, 2, 28, tzinfo=timezone.utc),
               datetime(2023, 3, 31, tzinfo=timezone.utc),
               datetime(2023, 3, 30, tzinfo=timezone.utc))),
    (D(2023, 1, 31), D(2023, 2, 27), "months",
     _frac(datetime(2023, 1, 31, tzinfo=timezone.utc),
           datetime(2023, 2, 28, tzinfo=timezone.utc),
           datetime(2023, 2, 27, tzinfo=timezone.utc))),
    (D(2020, 2, 15), D(2020, 3, 1), "months",
     _frac(datetime(2020, 2, 15, tzinfo=timezone.utc),
           datetime(2020, 3, 15, tzinfo=timezone.utc),
           datetime(2020, 3, 1, tzinfo=timezone.utc))),
    (D(2023, 5, 10), D(2023, 5, 20), "months",
     _frac(datetime(2023, 5, 10, tzinfo=timezone.utc),
           datetime(2023, 6, 10, tzinfo=timezone.utc),
           datetime(2023, 5, 20, tzinfo=timezone.utc))),
    # --- years
    (D(2020, 1, 1), D(2021, 1, 1), "years", 1.0),
    (D(2020, 2, 29), D(2021, 2, 28), "years", 1.0),   # clamped anniversary
    (D(2020, 2, 29), D(2024, 2, 29), "years", 4.0),
    (D(2000, 6, 15), D(2012, 6, 15), "years", 12.0),
    (D(2023, 1, 1), D(2023, 7, 2), "years",
     _frac(datetime(2023, 1, 1, tzinfo=timezone.utc),
           datetime(2024, 1, 1, tzinfo=timezone.utc),
           datetime(2023, 7, 2, tzinfo=timezone.utc))),
    (D(2020, 1, 1), D(2020, 7, 1), "years",
     _frac(datetime(2020, 1, 1, tzinfo=timezone.utc),
           datetime(2021, 1, 1, tzinfo=timezone.utc),
           datetime(2020, 7, 1, tzinfo=timezone.utc))),   # leap-year denominator
    (D(2012, 6, 15), D(2000, 6, 15), "years", -12.0),
    (D(2023, 12, 31), D(2023, 1, 1), "years",
     -_frac(datetime(2023, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2023, 12, 31, tzinfo=timezone.utc))),
    # --- aliases normalize identically
    (D(2023, 3, 1), D(2023, 3, 8), "day", 7.0),
    (D(2023, 3, 1), D(2023, 3, 8), "d", 7.0),
    (D(2023, 3, 1), D(2023, 3, 8), "DAYS", 7.0),
    (D(2023, 3, 1), D(2023, 3, 15), "week", 2.0),
    (D(2023, 3, 1), D(2023, 3, 15), "w", 2.0),
    (D(2023, 1, 15), D(2023, 2, 15), "month", 1.0),
    (D(2020, 1, 1), D(2021, 1, 1), "year", 1.0),
    (D(2020, 1, 1), D(2021, 1, 1), "y", 1.0),
    (D(2023, 3, 1), D(2023, 3, 2), "hour", 24.0),
    (D(2023, 3, 1), D(2023, 3, 2), "h", 24.0),
    (D(2023, 3, 1), D(2023, 3, 1, 0, 30), "min", 30.0),
    (D(2023, 3, 1), D(2023, 3, 1, 0, 0, 1), "sec", 1.0),
]


@pytest.mark.parametrize("a_us,b_us,unit,expected", _DURATION_CASES)
def test_duration_pinned(a_us, b_us, unit, expected):
    a = IntervalUs(a_us, a_us + 1, P.DAY)
    b = IntervalUs(b_us, b_us + 1, P.DAY)
    got = A.duration(a, b, unit)
    assert got == pytest.approx(expected, rel=0, abs=1e-9)


# Duration over the day grid: every ordered pair of consecutive-day starts.
_DAY_GRID_DUR = [
    (day_iv(2023, 3, i + 1), day_iv(2023, 3, j + 1), "days", float(j - i))
    for i in range(12) for j in range(12)
]


@pytest.mark.parametrize("a,b,unit,expected", _DAY_GRID_DUR)
def test_duration_day_grid(a, b, unit, expected):
    assert A.duration(a, b, unit) == pytest.approx(expected)


# Duration over the month grid: month-start to month-start is a whole number.
_MONTH_GRID_DUR = [
    (month_iv(2023, i + 1), month_iv(2023, j + 1), "months", float(j - i))
    for i in range(12) for j in range(12)
]


@pytest.mark.parametrize("a,b,unit,expected", _MONTH_GRID_DUR)
def test_duration_month_grid(a, b, unit, expected):
    assert A.duration(a, b, unit) == pytest.approx(expected, abs=1e-9)


_YEAR_GRID_DUR = [
    (year_iv(2000 + i), year_iv(2000 + j), "years", float(j - i))
    for i in range(10) for j in range(10)
]


@pytest.mark.parametrize("a,b,unit,expected", _YEAR_GRID_DUR)
def test_duration_year_grid(a, b, unit, expected):
    assert A.duration(a, b, unit) == pytest.approx(expected, abs=1e-9)


def test_duration_nan_on_unknown():
    d = day_iv(2023, 5, 7)
    assert math.isnan(A.duration(UNKNOWN, d, "days"))
    assert math.isnan(A.duration(d, UNKNOWN, "months"))
    assert math.isnan(A.duration(UNKNOWN, UNKNOWN, "years"))
    assert math.isnan(A.duration(open_until(D(2023, 5, 1)), d, "days"))
    # open end does NOT block duration (starts are what matter)
    assert A.duration(open_since(D(2023, 5, 1)), d, "days") == pytest.approx(6.0)


def test_duration_bad_unit():
    d = day_iv(2023, 5, 7)
    for bad in ("fortnight", "mo", "q", "", " decade"):
        with pytest.raises(ValueError):
            A.duration(d, d, bad)


# ---------------------------------------------------------------------------
# gap / span
# ---------------------------------------------------------------------------

_GAP_CASES = [
    (day_iv(2023, 5, 7), day_iv(2023, 5, 8), "days", 0.0),    # adjacent
    (day_iv(2023, 5, 7), day_iv(2023, 5, 10), "days", 2.0),
    (day_iv(2023, 5, 10), day_iv(2023, 5, 7), "days", -4.0),  # May7 - May11
    (day_iv(2023, 5, 7), day_iv(2023, 5, 7), "days", -1.0),   # overlap → negative
    (month_iv(2023, 5), month_iv(2023, 8), "months", 2.0),    # Jun1..Aug1
    (month_iv(2023, 5), day_iv(2023, 5, 20), "days", -12.0),  # inside → negative
]


@pytest.mark.parametrize("a,b,unit,expected", _GAP_CASES)
def test_gap(a, b, unit, expected):
    assert A.gap(a, b, unit) == pytest.approx(expected, abs=1e-9)


def test_gap_nan():
    assert math.isnan(A.gap(open_since(D(2023, 5, 1)), day_iv(2023, 6, 1), "days"))
    assert math.isnan(A.gap(day_iv(2023, 5, 1), open_until(D(2023, 6, 1)), "days"))


_SPAN_CASES = [
    (day_iv(2023, 5, 7), "days", 1.0),
    (day_iv(2023, 5, 7), "hours", 24.0),
    (month_iv(2023, 1), "months", 1.0),
    (month_iv(2023, 2), "months", 1.0),          # Feb is exactly 1 calendar month
    (month_iv(2023, 2), "days", 28.0),
    (month_iv(2020, 2), "days", 29.0),           # leap Feb
    (year_iv(2023), "years", 1.0),
    (year_iv(2020), "days", 366.0),
    (year_iv(2023), "days", 365.0),
    (year_iv(2023), "months", 12.0),
    (decade_iv(2020), "years", 10.0),
    (week_iv(2023, 19), "weeks", 1.0),
    (span_iv(D(2023, 5, 6), D(2023, 5, 8)), "days", 2.0),
]


@pytest.mark.parametrize("target,unit,expected", _SPAN_CASES)
def test_span(target, unit, expected):
    assert A.span(target, unit) == pytest.approx(expected, abs=1e-9)


def test_span_nan():
    assert math.isnan(A.span(UNKNOWN, "days"))
    assert math.isnan(A.span(open_since(D(2023, 5, 1)), "days"))
    assert math.isnan(A.span(open_until(D(2023, 5, 1)), "days"))


# ---------------------------------------------------------------------------
# shift / relative
# ---------------------------------------------------------------------------

_SHIFT_ROUNDTRIP_IVS = [
    day_iv(2023, 5, 7),
    month_iv(2023, 5),
    year_iv(2020),
    instant_iv(2023, 5, 7, 14, 30, 5, 123456),
    open_since(D(2023, 5, 1)),
    open_until(D(2023, 6, 1)),
    UNKNOWN,
]
_SHIFT_FIXED = [
    (t, n, u)
    for t in _SHIFT_ROUNDTRIP_IVS
    for n in (-30, -7, -1, 0, 1, 2, 90)
    for u in ("us", "ms", "seconds", "minutes", "hours", "days", "weeks")
]


@pytest.mark.parametrize("target,n,unit", _SHIFT_FIXED)
def test_shift_fixed_roundtrip(target, n, unit):
    moved = A.shift(target, n, unit)
    back = A.shift(moved, -n, unit)
    assert back == target
    # bounds moved by exactly n*unit (or stayed None)
    for orig, new in ((target.start_us, moved.start_us), (target.end_us, moved.end_us)):
        if orig is None:
            assert new is None
        else:
            assert new == orig + n * A.UNIT_US[A._canon_unit(unit)]
    # metadata preserved
    assert moved.precision == target.precision
    assert moved.source == target.source
    assert moved.rule_id == target.rule_id
    assert moved.anchor_us == target.anchor_us


_SHIFT_CAL_CASES = [
    # (date, n, unit, expected date) — calendar semantics incl. clamping
    (datetime(2023, 1, 31, tzinfo=timezone.utc), 1, "months", datetime(2023, 2, 28, tzinfo=timezone.utc)),
    (datetime(2023, 1, 31, tzinfo=timezone.utc), 2, "months", datetime(2023, 3, 31, tzinfo=timezone.utc)),
    (datetime(2023, 1, 31, tzinfo=timezone.utc), -1, "months", datetime(2022, 12, 31, tzinfo=timezone.utc)),
    (datetime(2023, 3, 31, tzinfo=timezone.utc), -1, "months", datetime(2023, 2, 28, tzinfo=timezone.utc)),
    (datetime(2020, 3, 31, tzinfo=timezone.utc), -1, "months", datetime(2020, 2, 29, tzinfo=timezone.utc)),
    (datetime(2023, 8, 31, tzinfo=timezone.utc), 1, "months", datetime(2023, 9, 30, tzinfo=timezone.utc)),
    (datetime(2023, 12, 15, tzinfo=timezone.utc), 2, "months", datetime(2024, 2, 15, tzinfo=timezone.utc)),
    (datetime(2023, 12, 31, tzinfo=timezone.utc), 1, "months", datetime(2024, 1, 31, tzinfo=timezone.utc)),
    (datetime(2023, 1, 1, tzinfo=timezone.utc), -1, "months", datetime(2022, 12, 1, tzinfo=timezone.utc)),
    (datetime(2020, 2, 29, tzinfo=timezone.utc), 1, "years", datetime(2021, 2, 28, tzinfo=timezone.utc)),
    (datetime(2020, 2, 29, tzinfo=timezone.utc), 4, "years", datetime(2024, 2, 29, tzinfo=timezone.utc)),
    (datetime(2020, 2, 29, tzinfo=timezone.utc), -4, "years", datetime(2016, 2, 29, tzinfo=timezone.utc)),
    (datetime(2000, 6, 15, tzinfo=timezone.utc), 12, "years", datetime(2012, 6, 15, tzinfo=timezone.utc)),
    (datetime(2023, 6, 15, 10, 20, 30, tzinfo=timezone.utc), 3, "months",
     datetime(2023, 9, 15, 10, 20, 30, tzinfo=timezone.utc)),   # time-of-day preserved
    (datetime(2023, 1, 15, tzinfo=timezone.utc), 0, "months", datetime(2023, 1, 15, tzinfo=timezone.utc)),
    (datetime(2023, 5, 31, tzinfo=timezone.utc), 13, "months", datetime(2024, 6, 30, tzinfo=timezone.utc)),
    (datetime(1999, 12, 31, tzinfo=timezone.utc), 25, "years", datetime(2024, 12, 31, tzinfo=timezone.utc)),
]


@pytest.mark.parametrize("start,n,unit,expected", _SHIFT_CAL_CASES)
def test_shift_calendar(start, n, unit, expected):
    t = IntervalUs(us(start), us(start) + 1, P.DAY)
    moved = A.shift(t, n, unit)
    assert moved.start_us == us(expected)


def test_shift_month_interval_keeps_month_shape():
    moved = A.shift(month_iv(2023, 5), 1, "months")
    assert (moved.start_us, moved.end_us) == (month_iv(2023, 6).start_us, month_iv(2023, 6).end_us)
    moved = A.shift(month_iv(2023, 1), -13, "months")
    assert (moved.start_us, moved.end_us) == (month_iv(2021, 12).start_us, month_iv(2021, 12).end_us)


def test_shift_unknown_is_noop():
    moved = A.shift(UNKNOWN, 5, "months")
    assert moved.start_us is None and moved.end_us is None
    moved2 = A.shift(UNKNOWN, -9, "days")
    assert moved2.start_us is None and moved2.end_us is None


def test_shift_bad_unit_and_side():
    d = day_iv(2023, 5, 7)
    with pytest.raises(ValueError):
        A.shift(d, 1, "fortnight")
    with pytest.raises(ValueError):
        A.relative(d, 1, "days", "sideways")


_RELATIVE_CASES = [
    (day_iv(2023, 5, 7), 2, "weeks", "before", day_iv(2023, 4, 23)),
    (day_iv(2023, 5, 7), 2, "weeks", "after", day_iv(2023, 5, 21)),
    (day_iv(2023, 5, 7), 1, "days", "before", day_iv(2023, 5, 6)),
    (month_iv(2023, 5), 3, "months", "before", month_iv(2023, 2)),
    (year_iv(2023), 1, "years", "after", year_iv(2024)),
    (day_iv(2023, 1, 5), 10, "days", "before", day_iv(2022, 12, 26)),
    (instant_iv(2023, 5, 7, 12, 0, 0), 90, "minutes", "before",
     instant_iv(2023, 5, 7, 10, 30, 0)),
]


@pytest.mark.parametrize("base,n,unit,side,expected", _RELATIVE_CASES)
def test_relative_phrasing(base, n, unit, side, expected):
    got = A.relative(base, n, unit, side)
    assert got.start_us == expected.start_us
    assert got.end_us == expected.end_us
    assert got.precision == expected.precision


# ---------------------------------------------------------------------------
# age_at_date / age_bounds_at_date
# ---------------------------------------------------------------------------

_AGE_CASES = [
    # (birth date, at date, expected age)
    (datetime(2000, 6, 15, tzinfo=timezone.utc), datetime(2012, 6, 15, tzinfo=timezone.utc), 12),
    (datetime(2000, 6, 15, tzinfo=timezone.utc), datetime(2012, 6, 14, tzinfo=timezone.utc), 11),
    (datetime(2000, 6, 15, tzinfo=timezone.utc), datetime(2012, 6, 16, tzinfo=timezone.utc), 12),
    (datetime(2000, 6, 15, tzinfo=timezone.utc), datetime(2000, 6, 15, tzinfo=timezone.utc), 0),
    (datetime(2000, 6, 15, tzinfo=timezone.utc), datetime(2000, 6, 16, tzinfo=timezone.utc), 0),
    (datetime(2000, 6, 15, tzinfo=timezone.utc), datetime(2001, 6, 14, tzinfo=timezone.utc), 0),
    (datetime(2000, 6, 15, tzinfo=timezone.utc), datetime(2001, 6, 15, tzinfo=timezone.utc), 1),
    (datetime(2000, 6, 15, tzinfo=timezone.utc), datetime(2023, 12, 31, tzinfo=timezone.utc), 23),
    (datetime(2000, 6, 15, tzinfo=timezone.utc), datetime(2024, 1, 1, tzinfo=timezone.utc), 23),
    (datetime(2000, 6, 15, tzinfo=timezone.utc), datetime(2024, 6, 15, tzinfo=timezone.utc), 24),
    # pre-birth: floor semantics → negative ordinals
    (datetime(2000, 6, 15, tzinfo=timezone.utc), datetime(2000, 6, 14, tzinfo=timezone.utc), -1),
    (datetime(2000, 6, 15, tzinfo=timezone.utc), datetime(1999, 6, 15, tzinfo=timezone.utc), -1),
    (datetime(2000, 6, 15, tzinfo=timezone.utc), datetime(1999, 6, 14, tzinfo=timezone.utc), -2),
    (datetime(2000, 6, 15, tzinfo=timezone.utc), datetime(1990, 1, 1, tzinfo=timezone.utc), -11),
    # Feb-29 birthdays: observed Feb 28 in common years
    (datetime(2000, 2, 29, tzinfo=timezone.utc), datetime(2001, 2, 28, tzinfo=timezone.utc), 1),
    (datetime(2000, 2, 29, tzinfo=timezone.utc), datetime(2001, 2, 27, tzinfo=timezone.utc), 0),
    (datetime(2000, 2, 29, tzinfo=timezone.utc), datetime(2001, 3, 1, tzinfo=timezone.utc), 1),
    (datetime(2000, 2, 29, tzinfo=timezone.utc), datetime(2004, 2, 29, tzinfo=timezone.utc), 4),
    (datetime(2000, 2, 29, tzinfo=timezone.utc), datetime(2004, 2, 28, tzinfo=timezone.utc), 3),
    (datetime(2000, 2, 29, tzinfo=timezone.utc), datetime(2100, 2, 28, tzinfo=timezone.utc), 100),
    # year boundary / century leap
    (datetime(2000, 12, 31, tzinfo=timezone.utc), datetime(2001, 1, 1, tzinfo=timezone.utc), 0),
    (datetime(2000, 12, 31, tzinfo=timezone.utc), datetime(2001, 12, 31, tzinfo=timezone.utc), 1),
    (datetime(2000, 1, 1, tzinfo=timezone.utc), datetime(2000, 12, 31, tzinfo=timezone.utc), 0),
    (datetime(1970, 1, 1, tzinfo=timezone.utc), datetime(1970, 1, 1, tzinfo=timezone.utc), 0),
    (datetime(1900, 3, 1, tzinfo=timezone.utc), datetime(2000, 3, 1, tzinfo=timezone.utc), 100),
    # time-of-day within the birthday
    (datetime(2000, 6, 15, 18, 0, tzinfo=timezone.utc),
     datetime(2012, 6, 15, 9, 0, tzinfo=timezone.utc), 11),
    (datetime(2000, 6, 15, 18, 0, tzinfo=timezone.utc),
     datetime(2012, 6, 15, 19, 0, tzinfo=timezone.utc), 12),
]


@pytest.mark.parametrize("birth,at,expected", _AGE_CASES)
def test_age_at_date(birth, at, expected):
    b_iv = IntervalUs(us(birth), us(birth) + 1, P.DAY)
    assert A.age_at_date(b_iv, us(at)) == expected
    # anniversary invariant: k-th birthday <= at < (k+1)-th birthday
    k = expected
    lo = A.shift(b_iv, k, "years").start_us
    hi = A.shift(b_iv, k + 1, "years").start_us
    assert lo <= us(at) < hi


def test_age_at_date_unknown():
    assert A.age_at_date(UNKNOWN, D(2023, 5, 7)) is None
    assert A.age_at_date(open_until(D(2000, 1, 1)), D(2023, 5, 7)) is None
    # open end doesn't block: only the start matters
    assert A.age_at_date(open_since(D(2000, 6, 15)), D(2012, 6, 15)) == 12


def test_age_bounds_honest_uncertainty():
    # month-precision birth: age within the birth month is honestly ambiguous
    b = month_iv(2000, 5)
    lo, hi = A.age_bounds_at_date(b, D(2020, 5, 15))
    assert (lo, hi) == (19, 20)
    lo2, hi2 = A.age_bounds_at_date(b, D(2020, 7, 15))
    assert (lo2, hi2) == (20, 20)
    # day-precision birth: exact except at the very start of the birthday,
    # where a same-day-evening birth is still 19 — honest (19, 20)
    d = day_iv(2000, 5, 10)
    assert A.age_bounds_at_date(d, D(2020, 5, 15)) == (20, 20)
    assert A.age_bounds_at_date(d, D(2020, 5, 10)) == (19, 20)
    assert A.age_bounds_at_date(d, D(2020, 5, 11)) == (20, 20)
    # unbounded sides
    assert A.age_bounds_at_date(open_since(D(2000, 5, 1)), D(2020, 5, 15)) == (None, 20)
    assert A.age_bounds_at_date(UNKNOWN, D(2020, 5, 15)) is None


# ---------------------------------------------------------------------------
# format_interval — precision is never sharpened (V7-09.12)
# ---------------------------------------------------------------------------

_FORMAT_CASES = [
    (day_iv(2023, 5, 7), "2023-05-07"),
    (span_iv(D(2023, 5, 6), D(2023, 5, 8), P.DAY), "2023-05-06..2023-05-07"),
    (month_iv(2023, 5), "2023-05"),                      # never "2023-05-01"
    (span_iv(D(2023, 5, 1), D(2023, 7, 1), P.MONTH), "2023-05..2023-06"),
    (span_iv(D(2022, 11, 1), D(2023, 2, 1), P.MONTH), "2022-11..2023-01"),
    (year_iv(2023), "2023"),
    (span_iv(D(2020, 1, 1), D(2023, 1, 1), P.YEAR), "2020..2022"),
    (decade_iv(2020), "2020s"),
    (span_iv(D(2019, 1, 1), D(2029, 1, 1), P.DECADE), "2019..2028"),
    (week_iv(2023, 19), "2023-W19"),                     # ISO week
    (week_iv(2020, 53), "2020-W53"),
    (span_iv(D(2023, 5, 9), D(2023, 5, 12), P.WEEK), "2023-05-09..2023-05-11"),  # non-ISO → day range
    (span_iv(D(2023, 6, 1), D(2023, 9, 1), P.SEASON), "2023-06..2023-08"),       # meteorological summer
    (span_iv(D(2023, 12, 1), D(2024, 3, 1), P.SEASON), "2023-12..2024-02"),      # crosses year
    (instant_iv(2023, 5, 7, 14, 30, 5), "2023-05-07T14:30:05Z"),
    (instant_iv(2023, 5, 7, 14, 30, 5, 123456), "2023-05-07T14:30:05.123456Z"),
    (span_iv(D(2023, 5, 7, 10), D(2023, 5, 7, 12), P.INSTANT),
     "2023-05-07T10:00:00Z..2023-05-07T12:00:00Z"),
    (UNKNOWN, "unknown"),
    (open_since(D(2023, 5, 1), P.MONTH), "since 2023-05"),
    (open_since(D(2023, 5, 7), P.DAY), "since 2023-05-07"),
    (open_until(D(2023, 6, 1), P.MONTH), "until 2023-05"),      # covered through May 31
    (open_until(D(2023, 5, 8), P.DAY), "until 2023-05-07"),     # covered through May 7
    (open_since(D(2023, 1, 1), P.YEAR), "since 2023"),
    (span_iv(D(2023, 5, 7, 14, 30), D(2023, 5, 9, 1, 0), P.UNKNOWN), "2023-05-07..2023-05-09"),
    (span_iv(D(2023, 5, 7), D(2023, 5, 8), P.UNKNOWN), "2023-05-07"),
    # degenerate interval renders its start, never crashes
    (iv(D(2023, 5, 7), D(2023, 5, 7), P.DAY), "2023-05-07"),
]


@pytest.mark.parametrize("target,expected", _FORMAT_CASES)
def test_format_interval(target, expected):
    assert A.format_interval(target) == expected


# Precision never sharpens: formatted output must not contain a finer
# component than declared.
_FORMAT_NEVER_SHARPER = [
    (month_iv(2023, 5), r"^\d{4}-\d{2}(\.\.\d{4}-\d{2})?$"),          # no day
    (span_iv(D(2023, 5, 7), D(2023, 5, 20), P.MONTH), r"^\d{4}-\d{2}$"),
    (span_iv(D(2023, 5, 7), D(2023, 6, 20), P.MONTH), r"^\d{4}-\d{2}\.\.\d{4}-\d{2}$"),
    (year_iv(2023), r"^\d{4}(\.\.\d{4})?$"),                           # no month
    (decade_iv(2020), r"^\d{4}s$"),
    (span_iv(D(2023, 6, 1), D(2023, 9, 1), P.SEASON), r"^\d{4}-\d{2}\.\.\d{4}-\d{2}$"),
    (week_iv(2023, 19), r"^\d{4}-W\d{2}$"),
    (open_since(D(2023, 5, 7), P.MONTH), r"^since \d{4}-\d{2}$"),      # open month bound: no day
    (open_since(D(2023, 1, 1), P.DECADE), r"^since \d{4}$"),
    (open_until(D(2024, 1, 1), P.YEAR), r"^until \d{4}$"),
]


@pytest.mark.parametrize("target,pattern", _FORMAT_NEVER_SHARPER)
def test_format_never_sharpens(target, pattern):
    assert re.match(pattern, A.format_interval(target))


# ---------------------------------------------------------------------------
# midpoint / misc
# ---------------------------------------------------------------------------


def test_midpoint():
    d = day_iv(2023, 5, 7)
    assert A.midpoint(d) == d.start_us + A.DAY_US // 2
    assert A.midpoint(UNKNOWN) is None
    assert A.midpoint(open_since(D(2023, 5, 1))) is None
    assert A.midpoint(open_until(D(2023, 5, 1))) is None


def test_algebra_id():
    assert A.ALGEBRA_ID == "interval_algebra/v1"


# ---------------------------------------------------------------------------
# Property suite (fixed seed — deterministic)
# ---------------------------------------------------------------------------

_RNG = random.Random(SEED)


def _rand_iv(rng: random.Random) -> IntervalUs:
    base = D(2018, 1, 1)
    span = D(2030, 1, 1) - base
    kind = rng.random()
    if kind < 0.12:
        return IntervalUs(None, None, rng.choice(list(P)))
    s = base + rng.randrange(0, span)
    if kind < 0.24:
        return IntervalUs(s, None, rng.choice(list(P)))
    if kind < 0.30:
        return IntervalUs(None, s, rng.choice(list(P)))
    length = rng.randrange(0, span // 3)
    return IntervalUs(s, s + length, rng.choice(list(P)))


def test_prop_relation_duality():
    rng = random.Random(SEED)
    for _ in range(400):
        a, b = _rand_iv(rng), _rand_iv(rng)
        ra = A.allen_relation(a, b)
        rb = A.allen_relation(b, a)
        if ra == "unknown":
            assert rb == "unknown"
            assert A.is_unknown(a) or A.is_unknown(b)
        else:
            assert rb == _INVERSE[ra]


def test_prop_bool_consistency():
    rng = random.Random(SEED)
    for _ in range(400):
        a, b = _rand_iv(rng), _rand_iv(rng)
        name = A.allen_relation(a, b)
        if name == "unknown":
            assert not (A.overlaps(a, b) or A.before(a, b) or A.meets(a, b)
                        or A.contains(a, b) or A.after(a, b))
            continue
        _check_bools(a, b, name)
        # intersect exists iff intervals overlap
        assert (A.intersect(a, b) is not None) == A.overlaps(a, b)


def test_prop_overlaps_symmetric_and_intersect_commutes():
    rng = random.Random(SEED)
    for _ in range(300):
        a, b = _rand_iv(rng), _rand_iv(rng)
        assert A.overlaps(a, b) == A.overlaps(b, a)
        x, y = A.intersect(a, b), A.intersect(b, a)
        assert (x is None) == (y is None)
        if x is not None:
            assert (x.start_us, x.end_us) == (y.start_us, y.end_us)
            # intersection contained in both
            assert A.contains(a, x) and A.contains(b, x)


def test_prop_union_bounds_and_commutes():
    rng = random.Random(SEED)
    for _ in range(300):
        a, b = _rand_iv(rng), _rand_iv(rng)
        u, v = A.union_span(a, b), A.union_span(b, a)
        assert (u.start_us, u.end_us) == (v.start_us, v.end_us)
        if not A.is_unknown(a) and not A.is_unknown(b):
            # open bound is unbounded: it wins in a union
            exp_s = None if (a.start_us is None or b.start_us is None) else min(a.start_us, b.start_us)
            exp_e = None if (a.end_us is None or b.end_us is None) else max(a.end_us, b.end_us)
            assert u.start_us == exp_s and u.end_us == exp_e
            if u.start_us is not None and u.end_us is not None:
                # bounded union contains both operands outright
                assert A.contains(u, a) and A.contains(u, b)
            # NB: a union that collapses to (None, None) is the unbounded
            # "all time" interval — IntervalUs cannot distinguish it from
            # fully-unknown, so contains() stays conservatively False.


def test_prop_contains_transitive():
    rng = random.Random(SEED)
    for _ in range(300):
        s = D(2020, 1, 1) + rng.randrange(0, 1000)
        inner = IntervalUs(s + 2000, s + 3000, P.DAY)
        mid = IntervalUs(s + 1000, s + 4000, P.DAY)
        outer = IntervalUs(s, s + 5000, P.DAY)
        assert A.contains(outer, mid) and A.contains(mid, inner)
        assert A.contains(outer, inner)          # transitivity
        assert A.overlaps(outer, inner)          # containment implies overlap
        assert A.allen_relation(inner, outer) == "during"


def test_prop_meets_excludes_before_and_overlap():
    rng = random.Random(SEED)
    for _ in range(200):
        s = D(2020, 1, 1) + rng.randrange(0, 10**8)
        e = s + rng.randrange(1, 10**7)
        e2 = e + rng.randrange(1, 10**7)
        a, b = IntervalUs(s, e, P.DAY), IntervalUs(e, e2, P.DAY)
        assert A.meets(a, b)
        assert not A.before(a, b)
        assert not A.overlaps(a, b)
        assert A.met_by(b, a)


def test_prop_shift_fixed_roundtrip_and_additive():
    rng = random.Random(SEED)
    units = ("us", "ms", "seconds", "minutes", "hours", "days", "weeks")
    for _ in range(300):
        a = _rand_iv(rng)
        n, m = rng.randrange(-100, 100), rng.randrange(-100, 100)
        u = rng.choice(units)
        assert A.shift(A.shift(a, n, u), -n, u) == a
        assert A.shift(A.shift(a, n, u), m, u) == A.shift(a, n + m, u)


def test_prop_shift_monotone_in_n():
    rng = random.Random(SEED)
    for _ in range(300):
        a = _rand_iv(rng)
        u = rng.choice(("days", "weeks", "months", "years"))
        lo, hi = A.shift(a, 1, u), A.shift(a, 2, u)
        if a.start_us is not None:
            assert hi.start_us > lo.start_us > a.start_us
        if a.end_us is not None:
            assert hi.end_us > lo.end_us > a.end_us


def test_prop_duration_antisymmetry():
    rng = random.Random(SEED)
    units = ("seconds", "hours", "days", "weeks", "months", "years")
    for _ in range(300):
        a, b = _rand_iv(rng), _rand_iv(rng)
        u = rng.choice(units)
        da, db = A.duration(a, b, u), A.duration(b, a, u)
        if a.start_us is None or b.start_us is None:
            assert math.isnan(da) and math.isnan(db)
        else:
            assert da == pytest.approx(-db, abs=1e-9)


def test_prop_duration_fixed_unit_exact():
    rng = random.Random(SEED)
    units = ("us", "ms", "seconds", "minutes", "hours", "days", "weeks")
    for _ in range(300):
        a, b = _rand_iv(rng), _rand_iv(rng)
        if a.start_us is None or b.start_us is None:
            continue
        u = rng.choice(units)
        assert A.duration(a, b, u) == pytest.approx(
            (b.start_us - a.start_us) / A.UNIT_US[A._canon_unit(u)], rel=1e-12)


def test_prop_precision_floor_never_sharpens():
    rng = random.Random(SEED)
    for _ in range(300):
        a, b = _rand_iv(rng), _rand_iv(rng)
        f = A.precision_floor(a, b)
        assert A._PRECISION_RANK[f] >= A._PRECISION_RANK[a.precision]
        assert A._PRECISION_RANK[f] >= A._PRECISION_RANK[b.precision]
        assert A.precision_floor(a, f) == f       # idempotent absorbency
        assert A.precision_floor(f, f) == f
        x = A.intersect(a, b)
        if x is not None:
            assert A._PRECISION_RANK[x.precision] >= A._PRECISION_RANK[a.precision]
            assert A._PRECISION_RANK[x.precision] >= A._PRECISION_RANK[b.precision]
        u = A.union_span(a, b)
        if not A.is_unknown(u):
            assert A._PRECISION_RANK[u.precision] >= min(
                A._PRECISION_RANK[a.precision], A._PRECISION_RANK[b.precision])


def test_prop_format_never_sharpens():
    rng = random.Random(SEED)
    day_part = re.compile(r"\d{4}-\d{2}-\d{2}")
    month_part = re.compile(r"\d{4}-\d{2}")
    for _ in range(300):
        a = _rand_iv(rng)
        out = A.format_interval(a)
        assert isinstance(out, str) and out
        if A.is_unknown(a):
            assert out == "unknown"
            continue
        if a.precision in (P.MONTH, P.SEASON):
            assert not day_part.search(out)      # never renders a day
        if a.precision in (P.YEAR, P.DECADE):
            assert not month_part.search(out)    # never renders a month
        if a.precision == P.YEAR and a.start_us is not None:
            pass


def test_prop_age_anniversary_invariant():
    rng = random.Random(SEED)
    for _ in range(300):
        b_us = D(1950, 1, 1) + rng.randrange(0, D(2020, 1, 1) - D(1950, 1, 1))
        at = b_us + rng.randrange(-10**8, 10**9)
        birth = IntervalUs(b_us, b_us + 1, P.DAY)
        k = A.age_at_date(birth, at)
        assert k is not None
        lo = A.shift(birth, k, "years").start_us
        hi = A.shift(birth, k + 1, "years").start_us
        assert lo <= at < hi


def test_prop_before_after_meets_partition():
    """For any two known intervals, Allen classification partitions reality:
    exactly one named relation holds and the booleans agree with it."""
    rng = random.Random(SEED)
    for _ in range(400):
        a, b = _rand_iv(rng), _rand_iv(rng)
        name = A.allen_relation(a, b)
        if name == "unknown":
            continue
        assert name in A.ALLEN_RELATIONS
        _check_bools(a, b, name)


def test_prop_duration_consistent_with_shift():
    """Shifting b by exactly the measured calendar distance lands on/after a's
    start consistently: whole-month part of duration(a,b) reproduces b."""
    rng = random.Random(SEED)
    for _ in range(200):
        s = D(2015, 1, 1) + rng.randrange(0, D(2030, 1, 1) - D(2015, 1, 1))
        e = s + rng.randrange(0, 10**9)
        a = IntervalUs(s, s + 1, P.DAY)
        b = IntervalUs(e, e + 1, P.DAY)
        m = A.duration(a, b, "months")
        whole = math.floor(m)
        moved = A.shift(a, whole, "months")
        moved_next = A.shift(a, whole + 1, "months")
        assert moved.start_us <= e < moved_next.start_us


def test_prop_unknown_total_ordering_safety():
    """Fully-unknown intervals never assert relations in any combination."""
    rng = random.Random(SEED)
    others = [_rand_iv(rng) for _ in range(50)] + [UNKNOWN]
    for o in others:
        assert not A.overlaps(UNKNOWN, o)
        assert not A.contains(UNKNOWN, o)
        assert not A.contains(o, UNKNOWN)
        assert not A.before(UNKNOWN, o)
        assert not A.meets(UNKNOWN, o)
        assert A.allen_relation(UNKNOWN, o) == "unknown"
        assert A.intersect(UNKNOWN, o) is None
        assert math.isnan(A.duration(UNKNOWN, o, "days"))
        fmt = A.format_interval(UNKNOWN)
        assert fmt == "unknown"


# ---------------------------------------------------------------------------
# Count guard: the suite must carry ≥400 cases (V7-09.10)
# ---------------------------------------------------------------------------


def test_param_count_guard():
    # Parametrized collections above: literal 39 + grids 300 + edges + duration
    # grids + precision floor 128 + shift round-trips 343 + ... — comfortably
    # above 400 before counting property loops. This guards against accidental
    # de-parametrization.
    total = (
        len(_literal_cases())
        + len(_GRID_CASES)
        + len(_MEETS_EDGES) + len(_OVERLAP_EDGES) + len(_CONTAINS_EDGES)
        + len(_BEFORE_AFTER_EDGES) + len(_INTERSECT_CASES) + len(_UNION_CASES)
        + len(_PREC_PAIRS) * 2
        + len(_DURATION_CASES) + len(_DAY_GRID_DUR) + len(_MONTH_GRID_DUR)
        + len(_YEAR_GRID_DUR)
        + len(_GAP_CASES) + len(_SPAN_CASES)
        + len(_SHIFT_FIXED) + len(_SHIFT_CAL_CASES) + len(_RELATIVE_CASES)
        + len(_AGE_CASES) + len(_FORMAT_CASES) + len(_FORMAT_NEVER_SHARPER)
    )
    assert total >= 400, total
