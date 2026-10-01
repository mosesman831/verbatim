"""Canonical reader view serializer — `pack_render/v1` (SPEC_V7 §12,
V7-12.10/12.13/12.14).

Renders a :class:`~verbatim.retrieval.v7.pack.PackResult` into the
compact text block handed to readers::

    # MEMORY (as of <query_time>)
    ## COMPUTED                      (only when computed items exist)
    - <kind>: <text>
    ## FACTS
    - [<ref>] <text> (proof N; <lifecycle>[ since <d>][; previously <v>]) ← [<ref>] ...
    ## EVIDENCE
    ### session <id> · <date> · <speaker1, speaker2>
    - [<ref>] <date> <speaker>: <quote> [(event <date>, <precision>)] [(context)] [(truncated)]
    ## CONFLICTS
    - <key>: [<ref>] [<ref>]          (or `- none`)
    ## MISSING
    - <facet>: <v1>, <v2>             (or `- none`)

Contract rules:

- Context-only items (neighbors, derived-item supports) are marked
  ``(context)`` — never indistinguishable from ranked evidence
  (V7-12.14).
- No internal scores, lane names, or policy details ever appear —
  those stay in ``explain`` (V7-12.14).
- Byte-deterministic for a fixed manifest (V7-12.13): all ordering is
  resolved at pack time; rendering adds no volatility — no wall clock,
  no random ids, refs are the only identifiers.
- Prompt-cacheable: fixed section order, stable ordering within.
- Free-text fields (quotes, speakers, session labels, computed text,
  facet values) are whitespace-flattened onto one line so stored text
  cannot corrupt the line-oriented block; the items' stored bytes and
  pins stay byte-exact — this is a view, not the evidence.
- The serialized view's own token total is `estimate_tokens(view)`
  (V7-12.03 reports it separately from the item budget).

Pure stdlib; deterministic.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from ...core.types_v7 import (
    LifecycleLabel,
    OccurredPrecision,
    PackItemV7,
)
from .pack import PackResult, RENDER_ID

_WS = re.compile(r"\s+")


def _date_us(us: int) -> str:
    return datetime.fromtimestamp(
        us / 1_000_000, tz=timezone.utc
    ).date().isoformat()


def _inline(text: str) -> str:
    """Quotes render on one line — embedded newlines would corrupt the
    line-oriented block. Whitespace runs collapse to single spaces;
    the item's stored bytes stay byte-exact (pins carry truth)."""
    return _WS.sub(" ", text).strip()


def _text(item: PackItemV7) -> str:
    q = item.quote
    if isinstance(q, str):
        return q
    if q is None:
        return ""
    return bytes(q).decode("utf-8", "replace")


def _rec_date(item: PackItemV7) -> Optional[str]:
    rec = item.recorded_at
    if isinstance(rec, str) and rec:
        return rec.split("T", 1)[0].split(" ", 1)[0]
    return None


def _occ_date(item: PackItemV7) -> Optional[str]:
    if item.occurred is not None and item.occurred.start_us is not None:
        occ = item.occurred
        prec = occ.precision
        us = occ.start_us
        if prec is OccurredPrecision.MONTH:
            return _date_us(us)[:7]
        if prec in (
            OccurredPrecision.YEAR,
            OccurredPrecision.DECADE,
            OccurredPrecision.SEASON,
        ):
            return _date_us(us)[:4]
        return _date_us(us)
    return None


def _item_date(item: PackItemV7) -> str:
    return _rec_date(item) or _occ_date(item) or "unknown"


def _support_refs(item: PackItemV7, delivered: set) -> list:
    """Support refs from pins that are actually delivered in this
    pack — a reader view never cites a ref it did not ship."""
    pins = item.pins or {}
    sups = pins.get("supports") or pins.get("support_refs") or []
    refs: list = []
    for s in sups:
        r = s.ref if isinstance(s, PackItemV7) else str(s)
        if r in delivered and r not in refs:
            refs.append(r)
    return refs


def _fact_line(item: PackItemV7, delivered: set) -> str:
    """`## FACTS` line for a derived item (V7-12.08/12.14): ref, text,
    proof count + lifecycle, valid-from, predecessor, then ← pins."""
    parts: list = []
    if item.proof_count:
        parts.append(f"proof {item.proof_count}")
    lc = item.lifecycle.value if item.lifecycle else "current"
    vf = (item.pins or {}).get("valid_from")
    if vf:
        parts.append(f"{lc} since {_inline(str(vf))}")
    else:
        parts.append(lc)
    prev = (item.pins or {}).get("previously") or (
        item.pins or {}
    ).get("predecessor_text")
    if prev:
        parts.append(f"previously {_inline(str(prev))}")
    line = f"- [{item.ref}] {_inline(_text(item))} ({'; '.join(parts)})"
    refs = _support_refs(item, delivered)
    if refs:
        line += " ← " + " ".join(f"[{r}]" for r in refs)
    return line


def _evidence_line(item: PackItemV7, context: bool) -> str:
    """`## EVIDENCE` line: ``- [<ref>] <date> <speaker>: <quote>``
    plus honest annotations (event time/precision, non-current
    lifecycle, ``(context)``, ``(truncated)``)."""
    line = f"- [{item.ref}] {_item_date(item)}"
    if item.speaker:
        line += f" {_inline(str(item.speaker))}:"
    else:
        line += ":"
    body = _inline(_text(item))
    if body:
        line += f" {body}"
    if item.occurred is not None and item.occurred.start_us is not None:
        prec = item.occurred.precision
        prec_s = (
            prec.value if isinstance(prec, OccurredPrecision) else str(prec)
        )
        line += f" (event {_occ_date(item)}, {prec_s})"
    if item.lifecycle not in (None, LifecycleLabel.CURRENT):
        lc = (
            item.lifecycle.value
            if isinstance(item.lifecycle, LifecycleLabel)
            else str(item.lifecycle)
        )
        line += f" ({lc})"
    if context:
        line += " (context)"
    if (item.pins or {}).get("truncated"):
        line += " (truncated)"
    return line


def render_reader_view(
    pack: PackResult, query_time_us: Optional[int] = None
) -> str:
    """Render ``pack`` as the `pack_render/v1` text block.

    ``query_time_us`` falls back to ``pack.query_time_us``; when both
    are absent the header honestly says ``unknown`` rather than
    inventing a "now".
    """
    qt = query_time_us if query_time_us is not None else pack.query_time_us
    as_of = _date_us(qt) if isinstance(qt, int) else "unknown"

    delivered_refs = {i.ref for i in pack.items}
    for i in pack.items:
        delivered_refs.update(c.ref for c in (i.context or ()))

    lines: list = [f"# MEMORY (as of {as_of})"]

    # -- COMPUTED (optional, only when present) -------------------------
    if pack.computed:
        lines.append("## COMPUTED")
        for c in pack.computed:
            lines.append(f"- {c.kind}: {_inline(c.text)}")

    # -- FACTS (derived items only) -------------------------------------
    lines.append("## FACTS")
    derived = [i for i in pack.items if i.derived]
    if derived:
        for i in derived:
            lines.append(_fact_line(i, delivered_refs))
    else:
        lines.append("- none")

    # -- EVIDENCE (non-derived units, grouped by session) ---------------
    lines.append("## EVIDENCE")
    wrote = False
    for g in pack.groups:
        entries = [e for e in g.entries if not e.item.derived]
        if not entries:
            continue
        if g.session_id is not None:
            head = f"### session {_inline(str(g.session_id))}"
            segs = [
                s
                for s in (
                    _inline(str(g.date)) if g.date else "",
                    ", ".join(_inline(str(sp)) for sp in g.speakers),
                )
                if s
            ]
            if segs:
                head += " · " + " · ".join(segs)
            lines.append(head)
        for e in entries:
            lines.append(_evidence_line(e.item, e.context))
            wrote = True
    if not wrote:
        lines.append("- none")

    # -- CONFLICTS -------------------------------------------------------
    lines.append("## CONFLICTS")
    any_conflict = False
    for key, refs in pack.conflicts:
        lines.append(
            f"- {_inline(str(key))}: "
            + " ".join(f"[{r}]" for r in refs)
        )
        any_conflict = True
    for key, n in pack.unresolved_conflicts:
        lines.append(
            f"- unresolved: {_inline(str(key))} ({n} items withheld)"
        )
        any_conflict = True
    if not any_conflict:
        lines.append("- none")

    # -- MISSING ---------------------------------------------------------
    lines.append("## MISSING")
    missing = pack.missing
    wrote_missing = False
    if missing is not None:
        for kind in sorted((missing.facets or {}).keys()):
            vals = missing.facets[kind]
            lines.append(
                f"- {_inline(str(kind))}: "
                + ", ".join(_inline(str(v)) for v in vals)
            )
            wrote_missing = True
        if missing.note:
            lines.append(f"- {_inline(missing.note)}")
            wrote_missing = True
    if not wrote_missing:
        lines.append("- none")

    return "\n".join(lines) + "\n"


__all__ = ["render_reader_view", "RENDER_ID"]
