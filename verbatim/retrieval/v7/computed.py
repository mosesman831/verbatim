"""Computed answers for context packs (SPEC_V7 §12, V7-12.09; §31.03).

For ``temporal_point``, ``temporal_order``, ``duration``,
``count_aggregate`` and ``current_value`` intents the pack carries a
``computed`` item: a timeline, an ordering, interval arithmetic, a
count of distinct supporting refs, or a current value with its
valid-from and superseded predecessor.

Hard rule (V7-12.09, the pack mutation target): every ``inputs`` ref
MUST be a delivered item — evidence or context. A computed item never
states anything its inputs do not support; the invariant is enforced
by an assert at emission. Undelivered predecessor refs named in item
pins are dropped from the text, never cited.

Deterministic: same items + same query -> byte-identical output.
Pure stdlib; no sibling imports outside ``core.types_v7``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional

from ...core.types_v7 import (
    ComputedItem,
    IntentClass,
    LifecycleLabel,
    OccurredPrecision,
    PackItemV7,
    QueryViewV7,
)

COMPUTED_VERSION = "computed/v1"

_MAX_COMPUTED = 3            # a pack never needs a wall of derivations
_LABEL_WORDS = 8             # short event label cap (words)
_LABEL_CHARS = 80            # short event label cap (chars)

# Intents that produce a computed item (V7-12.09).
_COMPUTED_INTENTS = frozenset(
    {
        IntentClass.TEMPORAL_POINT,
        IntentClass.TEMPORAL_ORDER,
        IntentClass.DURATION,
        IntentClass.COUNT_AGGREGATE,
        IntentClass.CURRENT_VALUE,
    }
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _text(item: PackItemV7) -> str:
    """Quote bytes as text (deterministic; defensive on mis-shapes)."""
    q = item.quote
    if isinstance(q, str):
        return q
    if q is None:
        return ""
    return bytes(q).decode("utf-8", "replace")


def _flatten(items: Iterable[PackItemV7]) -> list[PackItemV7]:
    """Delivered parents + their context items, ref-deduped, order kept."""
    out: list[PackItemV7] = []
    seen: set[str] = set()
    for item in items or ():
        for cand in (item, *tuple(item.context or ())):
            if cand.ref in seen:
                continue
            seen.add(cand.ref)
            out.append(cand)
    return out


def _context_ids(items: Iterable[PackItemV7]) -> set[int]:
    ids: set[int] = set()
    for item in items or ():
        for c in item.context or ():
            ids.add(id(c))
    return ids


def _is_context(item: PackItemV7, ctx_ids: set[int]) -> bool:
    return id(item) in ctx_ids


def _dated(items: Iterable[PackItemV7]) -> list[PackItemV7]:
    """Items with a known occurred start, ordered by (start, ref)."""
    dated = [
        i
        for i in items
        if i.occurred is not None and i.occurred.start_us is not None
    ]
    dated.sort(
        key=lambda i: (
            i.occurred.start_us,
            i.occurred.end_us if i.occurred.end_us is not None else -1,
            i.ref,
        )
    )
    return dated


def _date_of(us: int, precision: OccurredPrecision) -> str:
    """Deterministic UTC date string honoring declared precision."""
    dt = datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc)
    if precision is OccurredPrecision.MONTH:
        return f"{dt.year:04d}-{dt.month:02d}"
    if precision in (
        OccurredPrecision.YEAR,
        OccurredPrecision.DECADE,
        OccurredPrecision.SEASON,
    ):
        return f"{dt.year:04d}"
    return dt.date().isoformat()


def _rec_date(item: PackItemV7) -> Optional[str]:
    rec = item.recorded_at
    if isinstance(rec, str) and rec:
        return rec.split("T", 1)[0].split(" ", 1)[0]
    return None


def _label(item: PackItemV7) -> str:
    """Short event label: pinned ``event_label``/``label`` wins, else a
    bounded quote prefix. Never invents content."""
    for key in ("event_label", "label", "summary"):
        v = item.pins.get(key) if item.pins else None
        if isinstance(v, str) and v.strip():
            return v.strip()
    words = _text(item).split()
    label = " ".join(words[:_LABEL_WORDS])
    if len(label) > _LABEL_CHARS:
        label = label[:_LABEL_CHARS].rsplit(" ", 1)[0]
    return label


def _entry(item: PackItemV7) -> str:
    """One timeline entry: ``<label> <date> (<precision>) [<ref>]``."""
    occ = item.occurred
    prec = occ.precision if occ is not None else OccurredPrecision.UNKNOWN
    prec_s = prec.value if isinstance(prec, OccurredPrecision) else str(prec)
    date = _date_of(occ.start_us, prec) if occ is not None else "unknown"
    return f"{_label(item)} {date} ({prec_s}) [{item.ref}]"


def _state_value(item: PackItemV7) -> Optional[str]:
    pins = item.pins or {}
    for key in ("state_value", "value"):
        v = pins.get(key)
        if isinstance(v, str) and v:
            return v
    return None


def _state_key(item: PackItemV7) -> Optional[str]:
    v = (item.pins or {}).get("state_key")
    return v if isinstance(v, str) and v else None


def _valid_from(item: PackItemV7) -> Optional[str]:
    pins = item.pins or {}
    v = pins.get("valid_from")
    if isinstance(v, str) and v:
        return v
    vus = pins.get("valid_from_us")
    if isinstance(vus, int):
        return _date_of(vus, OccurredPrecision.DAY)
    if item.occurred is not None and item.occurred.start_us is not None:
        return _date_of(item.occurred.start_us, item.occurred.precision)
    return _rec_date(item)


# ---------------------------------------------------------------------------
# per-intent builders — each returns a ComputedItem or None
# ---------------------------------------------------------------------------


def _timeline(flat: list[PackItemV7]) -> Optional[ComputedItem]:
    dated = _dated(flat)
    if not dated:
        return None
    return ComputedItem(
        kind="timeline",
        text="; ".join(_entry(i) for i in dated),
        inputs=tuple(i.ref for i in dated),
        formula=f"{COMPUTED_VERSION}:timeline:sort(occurred.start_us)",
    )


def _order(flat: list[PackItemV7]) -> Optional[ComputedItem]:
    dated = _dated(flat)
    if len(dated) < 2:
        return None  # an order claim needs two pinned points
    first, second = dated[0], dated[1]
    text = "; ".join(_entry(i) for i in dated)
    text += f"; order: {_label(first)} before {_label(second)}"
    return ComputedItem(
        kind="order",
        text=text,
        inputs=tuple(i.ref for i in dated),
        formula=f"{COMPUTED_VERSION}:order:sort(occurred.start_us)",
    )


def _duration(flat: list[PackItemV7]) -> Optional[ComputedItem]:
    dated = _dated(flat)
    if len(dated) < 2:
        return None
    first, last = dated[0], dated[-1]
    d0 = datetime.fromtimestamp(
        first.occurred.start_us / 1_000_000, tz=timezone.utc
    ).date()
    d1 = datetime.fromtimestamp(
        last.occurred.start_us / 1_000_000, tz=timezone.utc
    ).date()
    days = (d1 - d0).days
    text = (
        f"{days} days between {_label(first)} "
        f"({_date_of(first.occurred.start_us, first.occurred.precision)}) "
        f"[{first.ref}] and {_label(last)} "
        f"({_date_of(last.occurred.start_us, last.occurred.precision)}) "
        f"[{last.ref}]"
    )
    return ComputedItem(
        kind="duration",
        text=text,
        inputs=(first.ref, last.ref),
        formula=f"{COMPUTED_VERSION}:duration:days(start_last-start_first)",
    )


def _count(flat: list[PackItemV7], ctx_ids: set[int]) -> Optional[ComputedItem]:
    # Counts run over distinct delivered *evidence* units: context-only
    # items and derived views restate evidence and are never counted.
    ev = [
        i
        for i in flat
        if not _is_context(i, ctx_ids) and not i.derived
    ]
    if len(ev) < 2:
        return None
    refs = tuple(i.ref for i in ev)
    return ComputedItem(
        kind="count",
        text=(
            f"{len(refs)} distinct supporting items "
            + " ".join(f"[{r}]" for r in refs)
        ),
        inputs=refs,
        formula=f"{COMPUTED_VERSION}:count:distinct(delivered_refs)",
    )


def _current_value(
    flat: list[PackItemV7], ctx_ids: set[int]
) -> Optional[ComputedItem]:
    currents = [
        i
        for i in flat
        if not _is_context(i, ctx_ids)
        and i.lifecycle is LifecycleLabel.CURRENT
    ]
    if not currents:
        return None
    top = currents[0]  # flat preserves pack order: first = highest ranked
    sk = _state_key(top)
    val = _state_value(top)
    # A superseded predecessor counts only when actually delivered.
    preds = [
        i
        for i in flat
        if i is not top
        and i.lifecycle is LifecycleLabel.SUPERSEDED
        and sk is not None
        and _state_key(i) == sk
    ]
    inputs: list[str] = [top.ref]
    if sk and val:
        text = f"{sk}: {val}"
        parts: list[str] = []
        vf = _valid_from(top)
        parts.append(f"current since {vf}" if vf else "current")
        for p in preds:
            pv = _state_value(p) or _label(p)
            parts.append(f"previously {pv} [{p.ref}]")
            inputs.append(p.ref)
        text += " (" + "; ".join(parts) + ")"
    else:
        # Unstructured fallback: the top current item's own text is the
        # value; its recorded date is the honest "as of".
        text = f"{_label(top)} [{top.ref}]"
        rd = _rec_date(top)
        if rd:
            text += f" (recorded {rd})"
    return ComputedItem(
        kind="current_value",
        text=text,
        inputs=tuple(inputs),
        formula=f"{COMPUTED_VERSION}:current_value:latest_current",
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def computed_items(
    query: QueryViewV7, items: Iterable[PackItemV7]
) -> list[ComputedItem]:
    """Build computed answers for ``query`` over delivered ``items``.

    ``items`` are the pack's delivered items (their ``context`` members
    count as delivered too). Emits at most one computed item per
    applicable intent class, in ``intent.classes`` order (primary
    first), capped at ``_MAX_COMPUTED``. Every emitted item's inputs
    are asserted to be a subset of the delivered refs — a builder that
    ever cites an undelivered ref fails loudly, never silently ships.
    """
    flat = _flatten(items)
    delivered_refs = {i.ref for i in flat}
    ctx_ids = _context_ids(items)

    intent = getattr(query, "intent", None)
    classes: list[IntentClass] = []
    if intent is not None:
        for ic in (intent.primary,) + tuple(intent.classes or ()):
            if ic in classes:
                continue
            classes.append(ic)

    out: list[ComputedItem] = []
    for ic in classes:
        if len(out) >= _MAX_COMPUTED:
            break
        if ic not in _COMPUTED_INTENTS:
            continue
        if ic is IntentClass.TEMPORAL_POINT:
            ci = _timeline(flat)
        elif ic is IntentClass.TEMPORAL_ORDER:
            ci = _order(flat)
        elif ic is IntentClass.DURATION:
            ci = _duration(flat)
        elif ic is IntentClass.COUNT_AGGREGATE:
            ci = _count(flat, ctx_ids)
        elif ic is IntentClass.CURRENT_VALUE:
            ci = _current_value(flat, ctx_ids)
        else:  # pragma: no cover - unreachable given the set check
            ci = None
        if ci is None:
            continue
        assert set(ci.inputs) <= delivered_refs, (
            f"computed item cites undelivered input: "
            f"{sorted(set(ci.inputs) - delivered_refs)}"
        )
        out.append(ci)
    return out


__all__ = ["COMPUTED_VERSION", "computed_items"]
