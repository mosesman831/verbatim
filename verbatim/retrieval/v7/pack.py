"""S8 context packs — the fewest tokens that let any reader answer.

SPEC_V7 §12 (V7-12.01–12.14), `tok/v1` §32.15.

Pipeline position: S8 consumes the final ranking (``ScoredCandidate``
objects, or pre-materialized ``PackItemV7`` objects) and produces a
:class:`PackResult` — delivered items (with neighbor ``context``
attached), session groups, computed answers, and honest flags.

Rules implemented here:

- ``estimate_tokens`` — `tok/v1` (§32.15):
  ``ceil(utf8bytes/4 + 0.25 * word-boundary-punctuation)``, clamped at
  ``>= 0.75 * word_count``. Dependency-free and pinned.
- Greedy packing over the final ranking with **skip-long-continue**
  (an item too large for the remaining budget is skipped; packing
  continues) and **never-empty** (if nothing fits, the top item is
  truncated at a sentence boundary and flagged) — V7-12.02.
- Metadata never counts against ``max_tokens``; only delivered content
  bytes do (V7-12.03). The reader view reports its own total
  separately (``render.py``).
- Duplicate collapse by normalized text signature with a
  ``collapsed_duplicates`` pin; conflicting/superseding evidence is
  never collapsed — it ships as an atomic group or the result flags
  ``conflict_unresolved`` (V7-12.04).
- Neighbor turns attach as ``context`` items: they count against the
  token budget but are never ranked independently (V7-12.05).
- Session grouping under one header per session; groups ordered by
  relevance by default, chronologically for temporal / ``history_of``
  intents or ``order="chronological"`` (V7-12.06).
- Every item carries the V7-12.07 field set; missing fields stay
  ``None``, never invented.
- Derived items (``derived=True``) carry their proof count and are
  followed, budget permitting, by at least one pinned supporting unit
  attached as context (V7-12.08).
- Computed answers via :mod:`.computed` (V7-12.09).

Determinism (V7-12.13): identical inputs produce byte-identical packs.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from ...core.types_v7 import (
    FORMULA_STATUS_PROVISIONAL,
    POOLS,
    IntentClass,
    IntervalUs,
    LifecycleLabel,
    MissingDescriptor,
    OccurredPrecision,
    OccurredSource,
    PackItemV7,
    Perspective,
    QueryViewV7,
    ScoredCandidate,
    SupportLabel,
)
from .computed import computed_items

TOKEN_ESTIMATOR_ID = "tok/v1"
PACK_VERSION = "pack/v1"
RENDER_ID = "pack_render/v1"

# Budget presets (V7-12.11): comparability rows name one of these.
TOKEN_PRESETS: dict = {
    "tokens_800": 800,
    "tokens_2000": 2000,   # consumer default
    "tokens_4096": 4096,   # Hindsight default parity
    "tokens_7000": 7000,   # Mem0 parity
}

# V7-12.01 also allows an exact `o200k_base`-compatible count reported
# beside the estimate "when tokenizers is installed". That requires a
# provisioned, hash-verified tokenizer artifact; none exists in wave A,
# so no side-count is emitted here — the estimator alone is the pinned
# budget measure (honest absence, not a silent 0).

# Intents that flip group ordering to chronological (V7-12.06).
_CHRONO_INTENTS = frozenset(
    {
        IntentClass.TEMPORAL_POINT,
        IntentClass.TEMPORAL_RANGE,
        IntentClass.TEMPORAL_ORDER,
        IntentClass.DURATION,
        IntentClass.HISTORY_OF,
    }
)

# Lifecycles that are evidence of change/conflict — never collapsed.
_PROTECTED_LIFECYCLES = frozenset(
    {LifecycleLabel.SUPERSEDED, LifecycleLabel.DISPUTED}
)

# Pin keys promoted out of ScoredCandidate.detail onto the item's pins.
_PIN_KEYS = (
    "conflict_group",
    "state_key",
    "state_value",
    "value",
    "valid_from",
    "valid_from_us",
    "predecessor_ref",
    "event_label",
    "label",
    "summary",
    "supports",
    "support_refs",
    "previously",
    "predecessor_text",
    "truncated",
)

# Sentence boundary for never-empty truncation: terminal punctuation
# followed by whitespace/end, or a newline run.
_SENT_END = re.compile(r"[.!?]+(?=\s|$)|\n+")


# ---------------------------------------------------------------------------
# tok/v1 (§32.15)
# ---------------------------------------------------------------------------


def estimate_tokens(text: Any) -> int:
    """`tok/v1` pinned estimator.

    ``ceil(len(utf8(text)) / 4 + 0.25 * p)`` where ``p`` counts
    punctuation characters adjacent to a word character (the
    word-boundary punctuation), clamped at ``>= 0.75 * word_count``.
    """
    if text is None:
        return 0
    if isinstance(text, (bytes, bytearray)):
        text = bytes(text).decode("utf-8", "replace")
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return 0
    nbytes = len(text.encode("utf-8"))
    words = text.split()
    nwords = len(words)

    def _word_char(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    punct = 0
    n = len(text)
    for i, ch in enumerate(text):
        if unicodedata.category(ch).startswith("P"):
            prev_w = i > 0 and _word_char(text[i - 1])
            next_w = i + 1 < n and _word_char(text[i + 1])
            if prev_w or next_w:
                punct += 1
    est = math.ceil(nbytes / 4 + 0.25 * punct)
    floor = math.ceil(0.75 * nwords)
    return max(est, floor)


# ---------------------------------------------------------------------------
# result types (owned here; V7-31.03 fixes ComputedItem in types_v7)
# ---------------------------------------------------------------------------


@dataclass
class GroupEntry:
    """One rendered line inside a session group: a ranked member or a
    context-only (neighbor / supporting) unit — never ranked
    independently (V7-12.05)."""

    item: PackItemV7
    context: bool = False


@dataclass
class SessionGroup:
    """Items sharing one session, rendered under a single
    ``### session`` header (V7-12.06). ``session_id=None`` is the
    ungrouped bucket — rendered without a session header."""

    session_id: Optional[str]
    date: Optional[str] = None
    speakers: tuple = ()
    entries: list = field(default_factory=list)  # list[GroupEntry]
    first_rank: int = 1 << 30  # best pack position of a ranked member


@dataclass
class PackResult:
    """S8 output. ``items`` are the ranked, delivered items in pack
    order (context units nest under each parent's ``.context`` and in
    ``groups`` entries marked ``context``). Metadata fields never count
    against ``max_tokens`` — ``tokens_items`` sums delivered content
    bytes only (V7-12.03)."""

    items: list = field(default_factory=list)          # list[PackItemV7]
    groups: list = field(default_factory=list)         # list[SessionGroup]
    computed: list = field(default_factory=list)       # list[ComputedItem]
    conflicts: list = field(default_factory=list)      # [(key, (refs))]
    unresolved_conflicts: list = field(default_factory=list)  # [(key, size)]
    missing: Optional[MissingDescriptor] = None
    tokens_items: int = 0
    truncated: bool = False
    truncated_refs: list = field(default_factory=list)
    collapsed_duplicates: int = 0
    omitted: int = 0
    warnings: list = field(default_factory=list)
    max_tokens: Optional[int] = None
    limit: Optional[int] = None
    order: str = "relevance"
    query_time_us: Optional[int] = None
    group_maxpool: bool = False          # V8-10.05 arm state actually applied
    estimator: str = TOKEN_ESTIMATOR_ID
    render_id: str = RENDER_ID
    pack_version: str = PACK_VERSION
    formula_status: str = FORMULA_STATUS_PROVISIONAL

    @property
    def conflict_unresolved(self) -> bool:
        return bool(self.unresolved_conflicts)


# ---------------------------------------------------------------------------
# item materialization
# ---------------------------------------------------------------------------


def _text(item: PackItemV7) -> str:
    q = item.quote
    if isinstance(q, str):
        return q
    if q is None:
        return ""
    return bytes(q).decode("utf-8", "replace")


def _as_enum(value: Any, cls: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, cls):
        return value
    try:
        return cls(value)
    except Exception:
        try:
            return cls[str(value)]
        except Exception:
            return default


def _as_interval(value: Any) -> Optional[IntervalUs]:
    if value is None or isinstance(value, IntervalUs):
        return value
    if isinstance(value, dict):
        return IntervalUs(
            start_us=value.get("start_us"),
            end_us=value.get("end_us"),
            precision=_as_enum(
                value.get("precision"),
                OccurredPrecision,
                OccurredPrecision.UNKNOWN,
            ),
            source=_as_enum(
                value.get("source"),
                OccurredSource,
                OccurredSource.UNKNOWN,
            ),
            rule_id=value.get("rule_id"),
            anchor_us=value.get("anchor_us"),
        )
    return None


def _as_item(entry: Any) -> tuple:
    """Normalize one ranking element to ``(PackItemV7, score|None)``.

    Accepts a pre-materialized :class:`PackItemV7` (kept verbatim,
    input order = rank) or a :class:`ScoredCandidate` whose ``detail``
    either carries a ready item under ``detail["item"]`` or the field
    values themselves. Missing V7-12.07 fields stay ``None`` — never
    invented.
    """
    if isinstance(entry, PackItemV7):
        return entry, None
    if not isinstance(entry, ScoredCandidate):
        raise TypeError(
            f"pack element must be PackItemV7 or ScoredCandidate, "
            f"got {type(entry).__name__}"
        )
    d = entry.detail or {}
    it = d.get("item")
    if isinstance(it, PackItemV7):
        return it, entry.score
    quote = d.get("quote", b"")
    if isinstance(quote, str):
        quote = quote.encode("utf-8")
    pins = dict(d.get("pins") or {})
    for k in _PIN_KEYS:
        if k in d and k not in pins:
            pins[k] = d[k]
    item = PackItemV7(
        ref=str(d.get("ref") or f"u:{entry.unit_id}"),
        unit_id=entry.unit_id,
        quote=quote,
        speaker=d.get("speaker"),
        recorded_at=d.get("recorded_at"),
        occurred=_as_interval(d.get("occurred")),
        session=d.get("session"),
        lifecycle=_as_enum(
            d.get("lifecycle"), LifecycleLabel, LifecycleLabel.CURRENT
        ),
        support=_as_enum(
            d.get("support"), SupportLabel, SupportLabel.SUPPORTED
        ),
        perspective=_as_enum(d.get("perspective"), Perspective, None),
        derived=bool(d.get("derived", False)),
        proof_count=int(d.get("proof_count") or 0),
        pins=pins,
    )
    return item, entry.score


@dataclass
class _Entry:
    item: PackItemV7
    pos: int  # rank position after ordering


# ---------------------------------------------------------------------------
# duplicate collapse + conflict grouping (V7-12.04)
# ---------------------------------------------------------------------------


#: ASCII chars whose Unicode category starts with "P" (punctuation) —
#: the per-char ``category()`` check is skipped entirely for ASCII text.
_ASCII_PUNCT_DELETE = {
    i: None
    for i in range(128)
    if unicodedata.category(chr(i)).startswith("P")
}


def _signature(item: PackItemV7) -> str:
    """Normalized text signature for near-duplicate detection."""
    folded = unicodedata.normalize("NFKC", _text(item)).casefold()
    if folded.isascii():
        stripped = folded.translate(_ASCII_PUNCT_DELETE)
    else:
        stripped = "".join(
            ch for ch in folded if not unicodedata.category(ch).startswith("P")
        )
    return " ".join(stripped.split())


def _protected(item: PackItemV7) -> bool:
    """Items that must never be duplicate-collapsed (V7-12.04):
    superseded/disputed evidence and anything in a declared conflict or
    state-key group."""
    pins = item.pins or {}
    return (
        item.lifecycle in _PROTECTED_LIFECYCLES
        or bool(pins.get("conflict_group"))
        or bool(pins.get("state_key"))
    )


def _collapse(entries: list) -> tuple:
    """Fold near-duplicates into their best-ranked representative.

    Returns ``(kept_entries, collapsed_count)``. Protected items are
    never folded; two unprotected items collapse only when both the
    signature and the lifecycle match.
    """
    seen: dict = {}
    kept: list = []
    collapsed = 0
    for e in entries:
        item = e.item
        if _protected(item):
            kept.append(e)
            continue
        # V8.5 — an empty signature means the item's text was never
        # materialized, not that it duplicates anything: unmaterialized
        # items key on their own unit_id so they can never fold together.
        sig = _signature(item)
        key = (sig if sig else f"unit-none:{item.unit_id or e.pos}", item.lifecycle)
        rep = seen.get(key)
        if rep is not None:
            collapsed += 1
            rep.item.pins["collapsed_duplicates"] = (
                int(rep.item.pins.get("collapsed_duplicates") or 0) + 1
            )
            continue
        seen[key] = e
        kept.append(e)
    return kept, collapsed


def _group_key(item: PackItemV7) -> Optional[str]:
    pins = item.pins or {}
    cg = pins.get("conflict_group")
    if cg:
        return f"conflict:{cg}"
    sk = pins.get("state_key")
    if sk:
        return f"state:{sk}"
    return None


def _atomic_units(entries: list) -> list:
    """Partition entries into atomic packing units: items sharing a
    conflict/state key form one all-or-nothing group ordered by its
    best member's rank (V7-12.04)."""
    groups: dict = {}
    order: list = []
    for e in entries:
        k = _group_key(e.item)
        if k is None:
            order.append([e])
        else:
            if k not in groups:
                groups[k] = []
                order.append(groups[k])
            groups[k].append(e)
    order.sort(key=lambda members: min(m.pos for m in members))
    return order


def _session_maxpool(atomic: list) -> list:
    """V8-10.05 ``pack.group_maxpool`` — group→member max-pool.

    Atomic units sharing a session cluster into ONE pack unit scored
    (positioned) by its best member's rank; members then expand in rank
    order inside the cluster. Conflict/state-key atomic groups keep
    their all-or-nothing semantics — the cluster only re-orders, it
    never splits or merges an atomic group. Session-less units each
    remain their own pack unit.
    """
    clusters: dict = {}
    order: list = []
    for agroup in atomic:
        key = _session_key(agroup[0].item)
        if key is None:
            order.append([agroup])
        else:
            if key not in clusters:
                clusters[key] = []
                order.append(clusters[key])
            clusters[key].append(agroup)
    order.sort(key=lambda ags: min(m.pos for g in ags for m in g))
    return order


# ---------------------------------------------------------------------------
# never-empty truncation (V7-12.02)
# ---------------------------------------------------------------------------


def _truncate_to_budget(text: str, budget_tokens: float) -> str:
    """Longest prefix of ``text`` within ``budget_tokens`` ending at a
    sentence boundary; falls back to a word boundary, then a hard
    character cut. estimate_tokens is monotone in prefix length, so the
    char-level fallback binary-searches."""
    if estimate_tokens(text) <= budget_tokens:
        return text
    best = ""
    for m in _SENT_END.finditer(text):
        cand = text[: m.end()].rstrip()
        if estimate_tokens(cand) <= budget_tokens:
            best = cand
        else:
            break  # prefixes only grow — once over budget, stay over
    if best:
        return best
    acc = ""
    for w in text.split(" "):
        cand = w if not acc else acc + " " + w
        if estimate_tokens(cand) <= budget_tokens:
            acc = cand
        else:
            break
    if acc:
        return acc
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= budget_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


# ---------------------------------------------------------------------------
# session grouping (V7-12.06)
# ---------------------------------------------------------------------------


def _session_key(item: PackItemV7) -> Optional[str]:
    s = item.session
    if not isinstance(s, dict):
        return None
    for k in ("id", "session_id", "label"):
        v = s.get(k)
        if v:
            return str(v)
    return None


def _session_date(item: PackItemV7) -> Optional[str]:
    s = item.session or {}
    if isinstance(s, dict):
        for k in ("date", "started_at", "started"):
            v = s.get(k)
            if isinstance(v, str) and v:
                return v.split("T", 1)[0].split(" ", 1)[0]
        vus = s.get("started_us")
        if isinstance(vus, int):
            return _date_us(vus)
    return None


def _date_us(us: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(
        us / 1_000_000, tz=timezone.utc
    ).date().isoformat()


def _occ_date(item: PackItemV7) -> Optional[str]:
    if item.occurred is not None and item.occurred.start_us is not None:
        return _date_us(item.occurred.start_us)
    return None


def _rec_date(item: PackItemV7) -> Optional[str]:
    rec = item.recorded_at
    if isinstance(rec, str) and rec:
        return rec.split("T", 1)[0].split(" ", 1)[0]
    return None


def _entry_date_key(item: PackItemV7) -> tuple:
    return (_rec_date(item) or _occ_date(item) or "9999-99-99", item.ref)


def _build_groups(items: list, chronological: bool) -> list:
    """Group delivered parents and their context units by session.

    Ranked members keep pack order for group ranking; context units
    group by their *own* session. Within a group, entries render
    chronologically (a session transcript reads in time order);
    context entries carry their flag through to the renderer.
    """
    gmap: dict = {}

    def _group(item: PackItemV7) -> SessionGroup:
        k = _session_key(item)
        if k not in gmap:
            gmap[k] = SessionGroup(session_id=k)
        return gmap[k]

    for idx, item in enumerate(items):
        if item.derived:
            # derived items render under FACTS, never under a session
            # header — but their context units still group by session
            for c in item.context or ():
                _group(c).entries.append(GroupEntry(item=c, context=True))
            continue
        g = _group(item)
        g.entries.append(GroupEntry(item=item, context=False))
        g.first_rank = min(g.first_rank, idx)
        for c in item.context or ():
            g2 = _group(c)
            g2.entries.append(GroupEntry(item=c, context=True))

    groups = list(gmap.values())
    for g in groups:
        g.entries.sort(
            key=lambda e: (_entry_date_key(e.item), e.context)
        )
        # header metadata: declared session fields win, else derive
        dates = [
            _session_date(e.item)
            or _rec_date(e.item)
            or _occ_date(e.item)
            for e in g.entries
        ]
        g.date = next((d for d in dates if d), None)
        speakers: list = []
        sess = next(
            (
                e.item.session
                for e in g.entries
                if isinstance(e.item.session, dict)
            ),
            {},
        )
        declared = sess.get("speakers") if isinstance(sess, dict) else None
        if isinstance(declared, (list, tuple)) and declared:
            speakers = [str(s) for s in declared]
        else:
            for e in g.entries:
                sp = e.item.speaker
                if sp and sp not in speakers:
                    speakers.append(str(sp))
        g.speakers = tuple(speakers)

    def _group_date(g: SessionGroup) -> str:
        return g.date or "9999-99-99"

    if chronological:
        groups.sort(
            key=lambda g: (
                _group_date(g),
                g.session_id or "",
                g.first_rank,
            )
        )
    else:
        groups.sort(key=lambda g: (g.first_rank, g.session_id or ""))
    return groups


# ---------------------------------------------------------------------------
# assemble_pack (V7-12.01–12.09)
# ---------------------------------------------------------------------------


def _pool_neighbor_window(ctx: Any) -> int:
    """Neighbor window from the caller's context budget (§32.3) — 1 on
    mid/default, 0 on low, 2 on high."""
    budget = getattr(ctx, "budget", None) if ctx is not None else None
    pool = POOLS.get(budget)
    return int(pool.neighbor_window) if pool is not None else 1


def _ctx_flag(ctx: Any, name: str, default: Any = None) -> Any:
    """§23 arm resolution (V8): tolerates whichever carrier the policy
    plumbing lands on — a ``<section>_<key>`` attribute on the policy
    object, a ``params``/``knobs``/``arms`` mapping keyed
    ``section.key``/``section_key``, or the lane manifest — else the
    spec default. Mirrors the dense lane's resolver (V8-08.04)."""
    if ctx is None:
        return default
    pol = getattr(ctx, "policy", None)
    val = getattr(pol, name.replace(".", "_"), None)
    if val is not None:
        return val
    for carrier in ("params", "knobs", "arms"):
        mapping = getattr(pol, carrier, None)
        if isinstance(mapping, dict):
            for key in (name, name.replace(".", "_"), name.split(".")[-1]):
                if key in mapping:
                    return mapping[key]
    manifest = getattr(ctx, "manifest", None) or {}
    if isinstance(manifest, dict):
        for key in (name, name.replace(".", "_")):
            if key in manifest:
                return manifest[key]
        head, _, tail = name.partition(".")
        sub = manifest.get(head)
        if isinstance(sub, dict) and tail in sub:
            return sub[tail]
    return default


def assemble_pack(
    scored: Iterable,
    query: QueryViewV7,
    max_tokens: Optional[int] = None,
    limit: Optional[int] = None,
    neighbor_window: Optional[int] = None,
    neighbors_fn: Optional[Callable] = None,
    order: str = "relevance",
    ctx: Any = None,
    group_maxpool: Optional[bool] = None,
) -> PackResult:
    """Greedy-budget pack assembly (V7-12.02).

    ``scored`` is the final ranking — :class:`ScoredCandidate` objects
    (sorted by descending score, ties keep input order) or
    pre-materialized :class:`PackItemV7` objects (input order is the
    ranking). ``neighbors_fn(item, n)`` returns up to ``n`` neighbor
    turns per side within the item's session; they attach as
    ``context`` — counted against ``max_tokens``, never ranked
    independently, deduped against delivered items.
    """
    # Tolerate the frozen-contract positional order
    # ``(scored, query, ctx, max_tokens, limit, neighbor_n)``: when the
    # third argument is a lane-context-shaped object rather than a
    # number, rebind accordingly.
    if max_tokens is not None and not isinstance(
        max_tokens, (int, float)
    ) and (hasattr(max_tokens, "policy") or hasattr(max_tokens, "budget")):
        _ctx = max_tokens
        _mt = limit
        _lim = neighbor_window
        _nw = neighbors_fn if isinstance(neighbors_fn, int) else None
        ctx = _ctx
        max_tokens = _mt if isinstance(_mt, (int, float)) else None
        limit = _lim if isinstance(_lim, int) else None
        neighbor_window = _nw
        neighbors_fn = None
    if order not in ("relevance", "chronological"):
        raise ValueError(f"unknown order {order!r}")
    if neighbor_window is None:
        neighbor_window = _pool_neighbor_window(ctx)
    budget = math.inf if max_tokens is None else float(max_tokens)
    item_cap = math.inf if limit is None else int(limit)

    # -- materialize + order ------------------------------------------------
    raw = list(scored or ())
    scored_flag = any(isinstance(s, ScoredCandidate) for s in raw)
    pairs = [_as_item(s) for s in raw]
    if scored_flag:
        # stable: ties keep input order
        pairs.sort(key=lambda p: -(p[1] if p[1] is not None else 0.0))
    seen_units: set = set()
    entries: list = []
    for item, _score in pairs:
        # a unit is delivered once, however many times it scored
        uk = item.unit_id
        if uk in seen_units:
            continue
        seen_units.add(uk)
        entries.append(_Entry(item=item, pos=len(entries)))

    # -- duplicate collapse (V7-12.04) --------------------------------------
    entries, n_collapsed = _collapse(entries)
    for i, e in enumerate(entries):
        e.pos = i
    atomic = _atomic_units(entries)

    # V8-10.05 ``pack.group_maxpool`` (§23, default off): when armed,
    # session clusters become the pack units — a group scores by its
    # best member and its members expand in rank order.
    if group_maxpool is None:
        group_maxpool = bool(_ctx_flag(ctx, "pack.group_maxpool", False))
    pack_units = _session_maxpool(atomic) if group_maxpool else [[g] for g in atomic]

    # -- greedy packing: skip-long-continue ---------------------------------
    delivered: list = []
    used = 0
    n_items = 0
    conflicts: list = []
    unresolved: list = []
    omitted = 0

    for agroups in pack_units:
        for members in agroups:
            n = len(members)
            gkey = _group_key(members[0].item) if n > 1 else None
            if n_items + n > item_cap:
                omitted += n
                if gkey is not None:
                    unresolved.append((gkey, n))
                continue
            cost = sum(estimate_tokens(_text(m.item)) for m in members)
            if used + cost > budget:
                omitted += n
                if gkey is not None:
                    unresolved.append((gkey, n))
                continue
            for m in members:
                delivered.append(m.item)
            used += cost
            n_items += n
            if gkey is not None:
                conflicts.append(
                    (gkey, tuple(m.item.ref for m in members))
                )

    # -- never-empty ---------------------------------------------------------
    truncated = False
    truncated_refs: list = []
    if not delivered and pack_units and item_cap != 0:
        top = pack_units[0][0][0].item
        new_text = _truncate_to_budget(_text(top), budget)
        if new_text != _text(top):
            top.quote = new_text.encode("utf-8")
            top.pins["truncated"] = True
            truncated = True
            truncated_refs.append(top.ref)
        delivered.append(top)
        used += estimate_tokens(_text(top))
        n_items += 1
        first_atomic = pack_units[0][0]
        if len(first_atomic) > 1:
            gkey = _group_key(first_atomic[0].item)
            if all(u[0] != gkey for u in unresolved):
                unresolved.append((gkey, len(first_atomic)))
                omitted += len(first_atomic) - 1

    seen_ctx_units: set = {i.unit_id for i in delivered}

    # -- derived items: pinned support follows (V7-12.08) --------------------
    warnings: list = []
    for item in delivered:
        if not item.derived:
            continue
        sups = (item.pins or {}).get("supports") or (
            item.pins or {}
        ).get("support_refs") or []
        sup_refs = {
            s.ref if isinstance(s, PackItemV7) else str(s) for s in sups
        }
        have = any(
            r in sup_refs
            for r in (
                [i.ref for i in delivered]
                + [c.ref for i in delivered for c in (i.context or ())]
            )
        )
        if have:
            continue
        attached = False
        for s in sups:
            if not isinstance(s, PackItemV7):
                continue  # a bare ref cannot be materialized here
            if s.unit_id in seen_ctx_units:
                continue
            cost = estimate_tokens(_text(s))
            if used + cost > budget:
                continue  # budget permitting — skip-long-continue
            item.context.append(s)
            seen_ctx_units.add(s.unit_id)
            used += cost
            attached = True
            break
        if not attached:
            warnings.append(f"derived_without_support:{item.ref}")

    # -- neighbor context (V7-12.05) -----------------------------------------
    if neighbor_window and neighbor_window > 0 and neighbors_fn is not None:
        for item in delivered:
            nbs = neighbors_fn(item, neighbor_window) or []
            for nb in nbs:
                if not isinstance(nb, PackItemV7):
                    continue
                if nb.unit_id in seen_ctx_units:
                    continue
                cost = estimate_tokens(_text(nb))
                if used + cost > budget:
                    continue  # context obeys skip-long-continue too
                item.context.append(nb)
                seen_ctx_units.add(nb.unit_id)
                used += cost

    # -- ordering + groups ----------------------------------------------------
    intent = getattr(query, "intent", None)
    primary = getattr(intent, "primary", None)
    chronological = order == "chronological" or primary in _CHRONO_INTENTS
    groups = _build_groups(delivered, chronological)

    computed = computed_items(query, delivered)

    return PackResult(
        items=delivered,
        groups=groups,
        computed=computed,
        conflicts=conflicts,
        unresolved_conflicts=[u for u in unresolved if u[0] is not None],
        tokens_items=used,
        truncated=truncated,
        truncated_refs=truncated_refs,
        collapsed_duplicates=n_collapsed,
        omitted=omitted,
        warnings=warnings,
        max_tokens=max_tokens,
        limit=limit,
        order=("chronological" if chronological else "relevance"),
        query_time_us=getattr(query, "query_time_us", None),
        group_maxpool=bool(group_maxpool),
    )


__all__ = [
    "TOKEN_ESTIMATOR_ID",
    "TOKEN_PRESETS",
    "PACK_VERSION",
    "RENDER_ID",
    "GroupEntry",
    "SessionGroup",
    "PackResult",
    "estimate_tokens",
    "assemble_pack",
]
