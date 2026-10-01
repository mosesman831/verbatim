"""Deterministic time handling: epoch micros, RFC 3339, precisions, intervals.

Calendar arithmetic, date ordering, interval overlap, and numeric comparison
are code responsibilities — never delegated to a model (SPEC §14). Unknown
times stay explicitly unknown.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone, date
from typing import Optional

from .types import ErrorCode, Precision, TimeInterval, VerbatimError


def now_us() -> int:
    return time.time_ns() // 1000


def wall_us() -> int:
    """Actual wall-clock reading; recorded separately from logical order."""
    return now_us()


def rfc3339(us: int) -> str:
    dt = datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def parse_rfc3339(text: str) -> int:
    s = text.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as exc:
        raise VerbatimError(ErrorCode.VALIDATION, f"invalid RFC 3339: {text!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)


def contains(iv: TimeInterval, t_us: int) -> Optional[bool]:
    """Whether T falls in [from, until). None = unknown endpoint prevents a decision."""
    if iv.from_us is None or iv.until_us is None:
        return None
    return iv.from_us <= t_us < iv.until_us


def overlaps(a: TimeInterval, b: TimeInterval) -> Optional[bool]:
    """Whether two half-open intervals overlap; None if unknowable."""
    if any(x is None for x in (a.from_us, a.until_us, b.from_us, b.until_us)):
        return None
    return a.from_us < b.until_us and b.from_us < a.until_us


_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_ISO_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_ISO_YEAR = re.compile(r"^(\d{4})$")


def _day_bounds(d: date) -> tuple[int, int]:
    start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return int(start.timestamp() * 1_000_000), int(end.timestamp() * 1_000_000)


def parse_time_expression(text: str, reference_us: int) -> Optional[TimeInterval]:
    """Parse a conservative set of explicit time expressions.

    Returns a half-open interval with declared precision, or None when the
    expression is ambiguous ("last Friday", locale-dependent forms) — callers
    must keep it unresolved rather than guess (SPEC §14).
    """
    s = text.strip().lower()
    ref = datetime.fromtimestamp(reference_us / 1_000_000, tz=timezone.utc).date()

    m = _ISO_DATE.match(s)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
        a, b = _day_bounds(d)
        return TimeInterval(a, b, Precision.DAY, "UTC", "explicit_date")

    m = _ISO_MONTH.match(s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if not 1 <= mo <= 12:
            return None
        first = datetime(y, mo, 1, tzinfo=timezone.utc)
        nxt = datetime(y + (mo == 12), (mo % 12) + 1, 1, tzinfo=timezone.utc)
        return TimeInterval(
            int(first.timestamp() * 1_000_000), int(nxt.timestamp() * 1_000_000),
            Precision.MONTH, "UTC", "explicit_month",
        )

    m = _ISO_YEAR.match(s)
    if m:
        y = int(m.group(1))
        first = datetime(y, 1, 1, tzinfo=timezone.utc)
        nxt = datetime(y + 1, 1, 1, tzinfo=timezone.utc)
        return TimeInterval(
            int(first.timestamp() * 1_000_000), int(nxt.timestamp() * 1_000_000),
            Precision.YEAR, "UTC", "explicit_year",
        )

    rel_days = {"today": 0, "yesterday": -1, "tomorrow": 1}
    if s in rel_days:
        d = ref + timedelta(days=rel_days[s])
        a, b = _day_bounds(d)
        return TimeInterval(a, b, Precision.DAY, "UTC", "relative_day")

    if s == "last week":
        start = ref - timedelta(days=ref.weekday() + 7)
        a, _ = _day_bounds(start)
        _, b = _day_bounds(start + timedelta(days=7))
        return TimeInterval(a, b, Precision.DAY, "UTC", "relative_week")

    return None


def precision_label(iv: TimeInterval) -> str:
    """Human-facing validity label; unknown stays unknown (SPEC §31)."""
    if iv.from_us is None and iv.until_us is None:
        return "unknown"
    if iv.until_us is None:
        return f"from {rfc3339(iv.from_us)[:10]}"
    if iv.from_us is None:
        return f"until {rfc3339(iv.until_us)[:10]}"
    return f"{rfc3339(iv.from_us)[:10]}..{rfc3339(iv.until_us)[:10]}"
