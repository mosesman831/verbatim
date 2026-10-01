"""Write-time relative-time mention resolution — ``reltime/v1``.

SPEC_V8 V8-09.04 / §21.6: a T0 write-path pass resolves *relative*
temporal expressions inside a turn unit's pinned text — "yesterday",
"last week", "two days ago", "next month", "tomorrow", "this weekend",
"on Friday", "last summer" — anchored on the unit's own ``occurred``
instant, and the writer persists one ``unit_time_mentions`` row per
resolved mention (half-open ``[start_us, end_us)``, byte spans into the
unit's pinned bytes, the anchor used, and the resolver version).

Design decisions (documented per the spec's determinism requirement):

- **Relative only.** The grammar is ``temporal/v2``'s own candidate
  grammar; a mention is emitted only when its ``IntervalUs.source`` is
  ``resolved_relative`` — the expression needed the unit's anchor to
  resolve. ``explicit`` expressions ("May 8, 2023", "2077") are absolute
  facts, not anchor-relative mentions, and are never written here; this
  also keeps bare-year misanchors (V8-09.03's "Cyberpunk 2077" class)
  out of the mention plane entirely.
- **Precision-first.** An ambiguous or unresolvable expression emits no
  row (the empty-set fallback, V7-09 carried): ``ambiguous_locale``
  hits, kept-unknown phrases (``None`` bounds, ``unknown`` precision),
  open/half-bounded intervals, and degenerate ``end <= start`` instants
  are all skipped — the table's ``end_us > start_us`` CHECK cannot
  represent them and a widened bound would be fabricated precision.
- **Anchor.** The anchor instant is the unit's ``occurred`` interval's
  start (``occurred_start_us``), the event-time floor; the end is used
  only when the start is unknown. A unit with no occurred bounds gets
  no rows — never a wall-clock guess.
- **Spans.** ``span_start``/``span_end`` are UTF-8 byte offsets *within*
  the unit's pinned bytes (0 = ``units.byte_start``), the same
  unit-relative convention ``entity_mentions`` and event ``pins`` use.

``RESOLVER_VERSION`` is bump-pinned: any change to the grammar filter,
the anchor rule, or ``temporal/v2``'s rule constants requires a bump so
a rebuild marks stale rows via ``derivation_coverage_v8`` and the fence
serves only same-version rows (V8-19.05, V8-13.06).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.types_v7 import OccurredPrecision, OccurredSource
from . import temporal_v2 as _tv2

#: Bump-pin: recorded on every emitted row. Any grammar/anchor change
#: bumps the tag; ``temporal/v2`` itself is the grammar (``RESOLVER_ID``).
RESOLVER_VERSION = "reltime/v1"

#: Precisions the ``unit_time_mentions`` CHECK accepts (§19 DDL).
ALLOWED_PRECISIONS = frozenset(
    {
        OccurredPrecision.INSTANT.value,
        OccurredPrecision.DAY.value,
        OccurredPrecision.WEEK.value,
        OccurredPrecision.MONTH.value,
        OccurredPrecision.SEASON.value,
        OccurredPrecision.YEAR.value,
    }
)


def unit_anchor_us(unit: Dict[str, Any]) -> Optional[int]:
    """The unit's occurred anchor instant, or ``None`` when no occurred
    bound exists (V8-09.04: such units get no rows)."""
    s = unit.get("occurred_start_us")
    if isinstance(s, int) and not isinstance(s, bool):
        return s
    e = unit.get("occurred_end_us")
    if isinstance(e, int) and not isinstance(e, bool):
        return e
    return None


def resolve_mentions(
    text: str,
    anchor_us: int,
    *,
    locale: str = "en-US",
    hemisphere: str = "north",
) -> List[Dict[str, Any]]:
    """Resolved relative-time mentions for one unit's pinned text.

    Returns ``[{ord, start_us, end_us, precision, span_start, span_end}]``
    in byte order — deterministic, no clock, pure over (text, anchor).
    """
    out: List[Dict[str, Any]] = []
    for rt in _tv2.resolve(
        text, int(anchor_us), locale=locale, hemisphere=hemisphere
    ):
        if rt.ambiguous_locale:
            continue  # ambiguous → not resolved (empty-set fallback)
        iv = rt.interval
        if iv.source is not OccurredSource.RESOLVED_RELATIVE:
            continue
        s, e = iv.start_us, iv.end_us
        if (
            not isinstance(s, int)
            or not isinstance(e, int)
            or isinstance(s, bool)
            or isinstance(e, bool)
            or e <= s
        ):
            continue  # unbounded / open / degenerate → not storable
        prec = (
            iv.precision.value
            if isinstance(iv.precision, OccurredPrecision)
            else str(iv.precision)
        )
        if prec not in ALLOWED_PRECISIONS:
            continue
        out.append(
            {
                "start_us": s,
                "end_us": e,
                "precision": prec,
                "span_start": int(rt.byte_start),
                "span_end": int(rt.byte_end),
            }
        )
    out.sort(key=lambda r: (r["span_start"], r["span_end"], r["start_us"]))
    for i, r in enumerate(out):
        r["ord"] = i
    return out


__all__ = [
    "RESOLVER_VERSION",
    "ALLOWED_PRECISIONS",
    "unit_anchor_us",
    "resolve_mentions",
]
