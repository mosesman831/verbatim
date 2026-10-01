"""Deterministic query analysis — ``query_analysis/v1`` (SPEC_V5 §31.1,
V5-31.01/31.02; worker contract docs/v5_contracts.md §8).

``analyze(query)`` is a pure, versioned, model-free classifier. A query
may carry several classes at once ("when was deploy-v2 released" is
temporal *and* identifier-bearing), so the result keeps the full ordered
``classes`` tuple plus a ``primary`` class chosen by a fixed precedence.
Temporal intent resolves to a bitemporal hint (``current`` | ``timeline``
| ``known_at``) — it may add a temporal score at ranking time and may
filter only when the query is explicit (V5-31.02).

No clock, no I/O, no randomness: identical input yields identical output,
and the serialized form is safe to log on the delivery receipt.

Enrichment seam (contracts §7): identifier/entity/normalization work is
delegated to ``verbatim.enrichment`` when importable. That package is
built by a parallel worker — this module resolves each contract function
at call time (package root first, then landed submodules, then the local
deterministic fallbacks below) so ``query_analysis/v1`` behaves the same
before and after the seam lands. If the real extractor's output ever
shifts a classification, bump ``QUERY_ANALYSIS_VERSION`` in
``memory/types.py`` — classification is versioned, never silently
changed.
"""

from __future__ import annotations

import enum
import importlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

from ..memory.types import (
    QUERY_ANALYSIS_VERSION,
    MemoryType,
    Polarity,
    QueryClass,
    TimePrecision,
    TimeStatus,
)

#: Bitemporal hint vocabulary (V5-31.02): current-state eligibility,
#: timeline read, or known-at point-in-time semantics.
HINT_CURRENT = "current"
HINT_TIMELINE = "timeline"
HINT_KNOWN_AT = "known_at"


# ---------------------------------------------------------------------
# enrichment seam (contracts §7)
# ---------------------------------------------------------------------
#
# ``verbatim.enrichment`` is owned by a parallel worker and may be absent
# or partially landed. Resolution order per contract function:
#   1. attribute on the ``verbatim.enrichment`` package root,
#   2. attribute on any landed submodule (identifiers, normalize,
#      temporal, polarity, typing),
#   3. the local fallback implementation in this file.
# The fallback keeps the same signatures so callers never branch.


@dataclass(frozen=True)
class Mention:
    """Normalized view of one enrichment mention: ``(kind, value, start,
    end)`` with UTF-8 byte offsets, case preserved (V5-30.14/30.17)."""

    kind: str
    value: str
    start: int = 0
    end: int = 0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "value": self.value,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class TemporalView:
    """Normalized view of ``enrichment.parse_temporal`` output."""

    precision: str = TimePrecision.UNKNOWN.value
    status: str = TimeStatus.UNKNOWN.value
    event_at: Optional[str] = None
    anchor_at: Optional[str] = None


def _enrich_func(name: str) -> Optional[Callable]:
    """Resolve a §7 contract function from ``verbatim.enrichment``."""
    try:
        pkg = importlib.import_module("verbatim.enrichment")
    except ImportError:
        pkg = None
    if pkg is not None:
        fn = getattr(pkg, name, None)
        if callable(fn):
            return fn
    for submod in (
        "identifiers",
        "normalize",
        "entities",
        "temporal",
        "polarity",
        "typing",
    ):
        try:
            mod = importlib.import_module(f"verbatim.enrichment.{submod}")
        except ImportError:
            continue
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    return None


def _attr(obj: Any, *names: str, index: Optional[int] = None) -> Any:
    """Duck-typed field access: attribute names first, then tuple index."""
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
        if hasattr(obj, n):
            return getattr(obj, n)
    if index is not None and isinstance(obj, (tuple, list)) and len(obj) > index:
        return obj[index]
    return None


def _mention_view(obj: Any, *, default_kind: str) -> Optional[Mention]:
    """Coerce an Identifier/Entity/dict/tuple into ``Mention``."""
    value = _attr(obj, "value", "text", "name", "label", "span")
    if value is None:
        if isinstance(obj, (tuple, list)) and len(obj) >= 2:
            value = obj[1]
        elif isinstance(obj, str):
            value = obj
    if not value:
        return None
    kind = _attr(obj, "kind", "type", index=0)
    start = _attr(obj, "start", index=2)
    end = _attr(obj, "end", index=3)
    return Mention(
        str(kind or default_kind),
        str(value),
        int(start or 0),
        int(end or 0),
    )


def _enum_value(obj: Any) -> Optional[str]:
    """Plain-string value of an enum/str/dict field. ``str``-mixin enums
    must be unwrapped before the ``isinstance(str)`` shortcut — ``str()``
    on them renders ``Class.MEMBER``, not the value."""
    if obj is None:
        return None
    if isinstance(obj, enum.Enum):
        val = obj.value
        return val if isinstance(val, str) else str(val)
    if isinstance(obj, str):
        return obj
    val = getattr(obj, "value", None)
    return str(val) if val is not None else str(obj)


# --- local fallbacks ----------------------------------------------------
#
# Used only while ``verbatim.enrichment`` lacks a function. Deterministic,
# stdlib-only, and intentionally conservative — they exist so this worker
# is runnable and testable mid-flight, not to compete with the pinned
# producer (``ENRICHMENT_VERSION`` is still the row producer label).

_IDENT_FALLBACK: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("url", re.compile(r"(?:https?|ftp|file)://[^\s<>\"'`]+", re.IGNORECASE)),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("handle", re.compile(r"(?<![\w.+-])@[A-Za-z_][A-Za-z0-9_-]{0,30}\b")),
    (
        "path",
        re.compile(
            r"(?<![\w@])(?:~|\.{1,2})?(?:/[A-Za-z0-9._@+~-]+){2,}/?"
            r"|[A-Za-z]:\\(?:[^\x00-\x1f<>\"|?*\\/]+\\?)+"
        ),
    ),
    ("ticket", re.compile(r"\b[A-Z][A-Z0-9]{1,11}-\d{1,7}\b")),
    ("hash", re.compile(r"\b(?:0x[0-9a-fA-F]{4,64}|[0-9a-fA-F]{7,64})\b")),
    (
        "version",
        re.compile(
            r"\b[A-Za-z][A-Za-z0-9_.]*-[vV]\d+(?:\.\d+)*\b"
            r"|\b[vV]\d+(?:\.\d+){1,3}(?:-[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*)?\b"
            r"|\b[vV]\d+\b"
            r"|\b\d+\.\d+(?:\.\d+){0,2}(?:-[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*)?\b"
        ),
    ),
    (
        "quoted",
        re.compile(
            r"\"[^\"\n]{1,500}\"|“[^”\n]{1,500}”|‘[^’\n]{1,500}’"
            r"|(?<![\w'])'[^'\n]{1,500}'(?![\w'])|`[^`\n]{1,500}`"
        ),
    ),
    (
        "code",
        re.compile(
            r"\b\w+(?:\.\w+)+\b|\b[\w-]*_[\w-]*\b"
            r"|\b[a-z0-9_]+(?:[A-Z][a-zA-Z0-9]*)+\b"
            r"|\b[A-Z][a-z0-9]+(?:[A-Z][a-zA-Z0-9]*)+\b"
            r"|\b[A-Za-z]+\d+[A-Za-z0-9]*\b|\b\d+[A-Za-z]+[A-Za-z0-9]*\b"
            r"|\b(?=[\w.-]*-)(?=[\w.-]*\d)[\w.-]+\b"
        ),
    ),
)

_IDENT_PRIORITY = {
    "url": 0, "email": 1, "path": 2, "handle": 3, "ticket": 4,
    "hash": 5, "version": 6, "code": 7, "quoted": 8,
}

#: Capitalized-run extraction for entities; sentence-initial function
#: words are suppressed by _SENT_STOP below.
_ENTITY_WORD_RE = re.compile(r"[^\W_][\w'’.-]*", re.UNICODE)
_ENTITY_CONNECTORS = frozenset({
    "of", "the", "de", "del", "della", "van", "von", "der", "den", "da",
    "di", "la", "le", "and",
})
_SENT_STOP = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "i", "we", "you", "they", "he", "she", "in", "on", "at", "for",
    "with", "by", "to", "from", "as", "but", "and", "or", "nor", "if",
    "when", "while", "however", "then", "so", "not", "no", "yes", "do",
    "does", "did", "is", "are", "was", "were", "be", "been", "can",
    "could", "will", "would", "should", "may", "might", "must", "shall",
    "there", "here", "what", "who", "how", "why", "which", "after",
    "before", "during", "my", "our", "your", "his", "her", "their",
    "let", "lets", "also", "just", "now", "today", "yesterday",
    "tomorrow", "note", "todo", "fixme", "per", "via", "see",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december",
})
_ENTITY_NEVER = frozenset({
    "i", "a", "i'm", "i'll", "i'd", "i've", "i’ll", "i’d", "i’ve",
})
_SENT_BOUNDARY_CHARS = frozenset(".!?\n")


def _fallback_extract_identifiers(text: str) -> List[Mention]:
    t = str(text or "")
    cands: List[Tuple[str, int, int]] = []
    for kind, rx in _IDENT_FALLBACK:
        for m in rx.finditer(t):
            s, e = m.span()
            while e > s and t[e - 1] in ".,;:!?)]}>\"'”’" and kind in {
                "url",
                "path",
            }:
                e -= 1
            if kind == "hash":
                tok = m.group(0)
                if not tok.lower().startswith("0x") and not any(
                    "a" <= c.lower() <= "f" for c in tok
                ):
                    continue
                if not tok.lower().startswith("0x") and not any(
                    c.isdigit() for c in tok
                ):
                    continue
            cands.append((kind, s, e))
    order = sorted(
        set(cands),
        key=lambda c: (_IDENT_PRIORITY[c[0]], c[1], -(c[2] - c[1])),
    )
    kept: List[Tuple[str, int, int]] = []
    for kind, s, e in order:
        if any(s < ke and ks < e for _k, ks, ke in kept):
            continue
        kept.append((kind, s, e))
    kept.sort(key=lambda c: (c[1], c[2]))
    offsets = _utf8_offsets(t)
    return [
        Mention(kind, t[s:e], offsets[s], offsets[e]) for kind, s, e in kept
    ]


def _sentence_initial(text: str, start: int) -> bool:
    i = start - 1
    while i >= 0 and text[i] in " \t\r":
        i -= 1
    return i < 0 or text[i] in _SENT_BOUNDARY_CHARS


def _fallback_extract_entities(text: str) -> List[Mention]:
    t = str(text or "")
    words = [(m.start(), m.end(), m.group(0)) for m in _ENTITY_WORD_RE.finditer(t)]
    cands: List[Tuple[str, int, int]] = []
    run_start = run_end = None  # type: ignore[assignment]
    pending = None
    for ws, we, wtok in words:
        first_upper = wtok[0].isupper()
        is_never = wtok.lower() in _ENTITY_NEVER
        is_conn = wtok.lower() in _ENTITY_CONNECTORS and not first_upper
        suppressed = _sentence_initial(t, ws) and wtok.lower() in _SENT_STOP
        participates = (first_upper and not is_never and not suppressed) or is_conn
        if participates:
            if run_start is None:
                run_start, run_end = ws, we
            else:
                run_end = we
            pending = (ws, we) if is_conn else None
        else:
            if run_start is not None:
                end = pending[0] if pending else run_end
                if end > run_start:
                    cands.append(("capitalized_span", run_start, end))
                run_start = run_end = pending = None
    if run_start is not None:
        end = pending[0] if pending else run_end
        if end > run_start:
            cands.append(("capitalized_span", run_start, end))
    cands.sort(key=lambda c: (c[1], c[2]))
    offsets = _utf8_offsets(t)
    return [
        Mention(kind, t[s:e], offsets[s], offsets[e]) for kind, s, e in cands
    ]


def _utf8_offsets(text: str) -> List[int]:
    offsets = [0] * (len(text) + 1)
    pos = 0
    for i, ch in enumerate(text):
        pos += len(ch.encode("utf-8"))
        offsets[i + 1] = pos
    return offsets


def _fallback_normalize_text(text: str, *, version: str = "norm/v1") -> str:
    if version != "norm/v1":
        raise ValueError(f"unsupported normalization version {version!r}")
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = unicodedata.normalize("NFKD", s)
    out: List[str] = []
    for ch in s:
        group = unicodedata.category(ch)[0]
        if group == "M":
            continue
        if group in frozenset({"P", "S", "C", "Z"}):
            out.append(" ")
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out).casefold()).strip()


_QUOTED_SPAN_RE = dict(
    (kind, rx) for kind, rx in _IDENT_FALLBACK
)["quoted"]  # the fallback "quoted" pattern

_NEGATION_RE = re.compile(
    r"\b(?:no longer|never|nothing|none|nobody|nowhere|nor|without"
    r"|do not|does not|did not|don'?t|doesn'?t|didn'?t|is not|isn'?t"
    r"|are not|aren'?t|was not|wasn'?t|were not|weren'?t|cannot|can'?t"
    r"|could not|couldn'?t|will not|won'?t|would not|wouldn'?t"
    r"|should not|shouldn'?t|not anymore|stopped|quits?|quit|gave up"
    r"|gives? up|dropped|dropped out|no more)\b",
    re.IGNORECASE,
)

_HEDGE_RE = re.compile(
    r"\b(?:maybe|perhaps|possibly|probably|reportedly|apparently"
    r"|alleged(?:ly)?|rumou?red|supposedly|seemingly|arguably|unsure"
    r"|uncertain|i think|i guess|i believe|i feel like|not sure"
    r"|don'?t think|do not think|might be|seems like|sort of|kind of"
    r"|whether)\b",
    re.IGNORECASE,
)

_HYPOTHETICAL_RE = re.compile(
    r"\b(?:what if|if we|if i|suppose|supposing|let'?s say|lets say"
    r"|imagine|hypothetically|in theory|theoretically|might|could"
    r"|would|shall we|should we|could we|would we|one day|someday"
    r"|plan(?:s|ned|ning)? to|intend(?:s|ed|ing)? to|hope(?:s|d)? to"
    r"|hoping to|thinking about|considering|contemplating"
    r"|want(?:s|ed)? to|wish(?:es|ed)?)\b",
    re.IGNORECASE,
)

_HEARSAY_RE = re.compile(
    r"\b(?:rumou?rs?\s+(?:says|has it)|word is|they say|i heard"
    r"|people say|report has it|according to|she said|he said"
    r"|they said|docs? says?|documentation says|the doc says"
    r"|runbook says|readme says)\b",
    re.IGNORECASE,
)


def _quoted_majority(text: str) -> bool:
    """True when ≥half the alphanumeric content sits inside quotes."""
    total = sum(1 for c in text if c.isalnum())
    if not total:
        return False
    quoted = 0
    for m in _QUOTED_SPAN_RE.finditer(text):
        quoted += sum(1 for c in m.group(0) if c.isalnum())
    if quoted * 2 >= total and quoted > 0:
        return True
    # hearsay verb + any quoted span: attributed, not asserted
    return bool(quoted and _HEARSAY_RE.search(text))


def _fallback_polarity(text: str) -> str:
    t = str(text or "")
    if _quoted_majority(t):
        return Polarity.QUOTED.value
    if _HYPOTHETICAL_RE.search(t):
        return Polarity.HYPOTHETICAL.value
    if _HEDGE_RE.search(t):
        return Polarity.HEDGED.value
    if _NEGATION_RE.search(t):
        return Polarity.NEGATED.value
    return Polarity.AFFIRMATIVE.value


_TYPE_RULES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (
        MemoryType.PREFERENCE.value,
        re.compile(
            r"\b(?:i|we|my|our)?\s*(?:really\s+)?(?:like|love|enjoy|hate"
            r"|dislike|prefer)|\b(?:favorite|favourite)\b",
            re.IGNORECASE,
        ),
    ),
    (
        MemoryType.DECISION.value,
        re.compile(
            r"\b(?:decided|decision|chose|chosen|picked|went with"
            r"|go with|going with|let'?s use|we'?ll use|agreed"
            r"|settled on|approved)\b",
            re.IGNORECASE,
        ),
    ),
    (
        MemoryType.PLAN.value,
        re.compile(
            r"\b(?:plan(?:s|ned|ning)? to|intend(?:s|ing)? to|roadmap"
            r"|next step|todo|we will|i will|we'?ll|i'?ll"
            r"|going to|aim(?:s|ing)? to)\b",
            re.IGNORECASE,
        ),
    ),
    (
        MemoryType.ABSENCE.value,
        re.compile(
            r"\b(?:no longer|never used|never had|nothing|do not have"
            r"|don'?t have|does not have|stopped|quit|no more"
            r"|ran out|removed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        MemoryType.PROCEDURE_HINT.value,
        re.compile(
            r"\b(?:how to|steps? to|procedure for|first\b.{1,80}\bthen"
            r"|to deploy\b|run\b.{1,40}\bthen)\b",
            re.IGNORECASE,
        ),
    ),
    (
        MemoryType.RELATIONSHIP.value,
        re.compile(
            r"\b(?:reports? to|works? (?:with|under|for)|married to"
            r"|manages|managed by|mentor(?:s|ed)?|colleague|teammate"
            r"|partner of|friend of|sibling|spouse)\b",
            re.IGNORECASE,
        ),
    ),
    (
        MemoryType.EVENT.value,
        re.compile(
            r"\b(?:yesterday|ago|happened|met with|released|shipped"
            r"|deployed|launched|migrated|rolled out|was held"
            r"|took place)\b|\b\d{4}-\d{2}-\d{2}\b",
            re.IGNORECASE,
        ),
    ),
    (
        MemoryType.STATE.value,
        re.compile(
            r"\b(?:currently|is running|lives? in|works? at|based in"
            r"|located (?:in|at)|status is|remains|still is)\b",
            re.IGNORECASE,
        ),
    ),
)


def _fallback_classify_type(text: str) -> str:
    """Typing heuristic (V5-30.25): interpretation label, never authority.
    ``untyped`` is the honest default when no marker fires."""
    t = str(text or "")
    if not t.strip():
        return MemoryType.UNTYPED.value
    for type_value, rx in _TYPE_RULES:
        if rx.search(t):
            return type_value
    if len(_content_terms(t)) < 2:
        return MemoryType.UNTYPED.value
    return MemoryType.FACT.value


_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_ISO_MONTH_RE = re.compile(r"\b(\d{4})-(\d{2})\b")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_RELATIVE_TIME_RE = re.compile(
    r"\b(?:yesterday|today|tomorrow|last\s+(?:week|month|year|night)"
    r"|this\s+(?:week|month|year|morning|afternoon|evening)"
    r"|next\s+(?:week|month|year)|\d+\s+(?:day|week|month|year)s?\s+ago"
    r"|since\s+(?:last\s+)?\w+)\b",
    re.IGNORECASE,
)
_MONTH_NAME_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september"
    r"|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?"
    r"(?:,?\s+\d{4})?\b",
    re.IGNORECASE,
)
_COMPLETED_MARKERS = re.compile(
    r"\b(?:was|were|did|happened|finished|completed|done|released"
    r"|shipped|deployed|moved|changed|switched|migrated|ended|ago)\b",
    re.IGNORECASE,
)
_ONGOING_MARKERS = re.compile(
    r"\b(?:is|are|am|currently|still|ongoing|remains|keeps?|lives?"
    r"|works?|uses?|runs?)\b",
    re.IGNORECASE,
)
_PLANNED_MARKERS = re.compile(
    r"\b(?:will|planned|plans?|planning|intends?|scheduled|upcoming"
    r"|going to|next)\b",
    re.IGNORECASE,
)


def _fallback_parse_temporal(text: str, anchor: Optional[str]) -> TemporalView:
    """Ambiguity → ``unknown``, never a guessed date (V5-30.10)."""
    t = str(text or "")
    anchor_at = anchor or None
    precision = TimePrecision.UNKNOWN.value
    event_at: Optional[str] = None
    m = _ISO_DATE_RE.search(t)
    if m:
        precision = TimePrecision.DAY.value
        event_at = m.group(0)
    else:
        m = _MONTH_NAME_RE.search(t)
        if m:
            precision = TimePrecision.DAY.value if re.search(
                r"\d{1,2}", m.group(0)
            ) else TimePrecision.MONTH.value
            event_at = m.group(0)
        else:
            m = _ISO_MONTH_RE.search(t)
            if m:
                precision = TimePrecision.MONTH.value
                event_at = m.group(0)
            elif _RELATIVE_TIME_RE.search(t):
                precision = TimePrecision.RELATIVE.value
                # resolved interval needs the anchor; the fallback records
                # the expression's existence, not a fabricated instant.
                event_at = None
            else:
                m = _YEAR_RE.search(t)
                if m:
                    precision = TimePrecision.YEAR.value
                    event_at = m.group(0)
    if _PLANNED_MARKERS.search(t):
        status = TimeStatus.PLANNED.value
    elif _COMPLETED_MARKERS.search(t):
        status = TimeStatus.COMPLETED.value
    elif _ONGOING_MARKERS.search(t):
        status = TimeStatus.ONGOING.value
    else:
        status = TimeStatus.UNKNOWN.value
    return TemporalView(precision, status, event_at, anchor_at)


# --- resolved entry points ----------------------------------------------
# These wrap whichever implementation was resolved; they are what the
# classifier and the update detector call.


def extract_identifiers(text: str) -> List[Mention]:
    fn = _enrich_func("extract_identifiers")
    if fn is None:
        return _fallback_extract_identifiers(text)
    out = []
    for item in fn(str(text or "")):
        v = _mention_view(item, default_kind="identifier")
        if v is not None:
            out.append(v)
    return out


def extract_entities(text: str) -> List[Mention]:
    fn = _enrich_func("extract_entities")
    if fn is None:
        return _fallback_extract_entities(text)
    out = []
    for item in fn(str(text or "")):
        v = _mention_view(item, default_kind="entity")
        if v is not None:
            out.append(v)
    return out


def normalize_text(text: str) -> str:
    fn = _enrich_func("normalize_text")
    if fn is None:
        return _fallback_normalize_text(text)
    try:
        return str(fn(text))
    except TypeError:
        return str(fn(str(text or ""), version="norm/v1"))


def text_polarity(text: str) -> str:
    fn = _enrich_func("polarity")
    if fn is None:
        return _fallback_polarity(text)
    return _enum_value(fn(str(text or ""))) or Polarity.AFFIRMATIVE.value


def classify_type(text: str) -> str:
    fn = _enrich_func("classify_type")
    if fn is None:
        return _fallback_classify_type(text)
    return _enum_value(fn(str(text or ""))) or MemoryType.UNTYPED.value


def parse_temporal(text: str, anchor: Optional[str]) -> TemporalView:
    """§7 ``parse_temporal(text, anchor)``. The landed parser requires a
    strict RFC3339 anchor (naive anchors bake host timezones into derived
    artifacts); with no anchor we use the local fallback, which resolves
    anchor-free explicit expressions and leaves relatives unresolved —
    the honest answer when no anchor exists to resolve against."""
    fn = _enrich_func("parse_temporal")
    if fn is None or not anchor:
        return _fallback_parse_temporal(text, anchor)
    try:
        res = fn(str(text or ""), anchor)
    except (ValueError, TypeError):
        return _fallback_parse_temporal(text, anchor)
    return TemporalView(
        _enum_value(_attr(res, "precision"))
        or TimePrecision.UNKNOWN.value,
        _enum_value(_attr(res, "status")) or TimeStatus.UNKNOWN.value,
        _attr(res, "event_at") or None,
        _attr(res, "anchor_at") or anchor,
    )


# ---------------------------------------------------------------------
# query analysis (query_analysis/v1)
# ---------------------------------------------------------------------

#: Function words — no retrieval signal. Negation-bearing tokens
#: (no/not/never/nor/without/none/longer) are deliberately absent:
#: dropping them would silently invert claim polarity.
_STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has
    have he her hers him his how i if in into is it its me my of on or our
    ours she so such than that the their theirs them then there these they
    this those to too was we were what when where which who whom why will
    with would you your yours about after again against all also am any
    because before being between both each few further here once only other
    out over own same should some under until up very
    """.split()
)

#: Conversational meta-verbs address the memory system itself ("tell me",
#: "do I remember") — no content signal. Same proven list as the v2
#: analyzer, pinned here so query_analysis/v1 never drifts with another
#: module's edits. Negation stays load-bearing.
_META_TERMS = frozenset(
    """
    tell told remember recall remind show find search look list mention
    mentioned say said ask asked talk talked speak spoke know knew think
    thought wondering want wanted need needed give gave get got please help
    mean meant happen happened anything something everything everyone
    someone anybody somebody thing things stuff way ways kind kinda sort
    really actually just even still yet maybe perhaps probably definitely
    basically literally honestly hey hi hello ok okay yeah yes nope thanks
    thank please dude btw fyi
    """.split()
) - frozenset({"no", "not", "never", "nor", "without", "none"})

_ALNUM_RE = re.compile(r"[^\W_]", re.UNICODE)
_TOKEN_RE = re.compile(r"[^\W_][\w'’.-]*", re.UNICODE)

#: Temporal intent → bitemporal hint (V5-31.02). Checked in this order:
#: ``as of`` is the most explicit (known-at), ``when/history/before/after``
#: read as timeline, ``current/latest`` is current-state eligibility.
_KNOWN_AT_RE = re.compile(
    r"\b(?:as\s+of|as-of|known[\s_-]?at|at the time|point in time"
    r"|back (?:in|on)|as it was)\b",
    re.IGNORECASE,
)
_TIMELINE_RE = re.compile(
    r"\b(?:when|history|historical|timeline|before|after|earlier"
    r"|previously|originally|used to|since|ever|ago|during"
    r"|last time|first time|over time|evolution|changed)\b",
    re.IGNORECASE,
)
_CURRENT_RE = re.compile(
    r"\b(?:current|currently|latest|now|today|right now|present"
    r"|presently|nowadays|as it stands|most recent|newest)\b",
    re.IGNORECASE,
)

_PREFERENCE_RE = re.compile(
    r"\b(?:favorite|favourite|prefer|prefers|preferred|preference"
    r"|preferences|like best|decided|decide|decides|decision|decisions"
    r"|chose|chosen|picked|pick|went with|go with|going with|choice"
    r"|opinion|stand on|policy on|rule on)\b",
    re.IGNORECASE,
)

_PROCEDURAL_RE = re.compile(
    r"\bhow\s+(?:did|do|does|to|can|could|should|would|might|shall)\b"
    r"|\b(?:steps?|procedure|process|recipe|workflow|instructions?|runbook)"
    r"\s+(?:to|for|of|behind)\b"
    r"|\bwhat\s+(?:is|was|are|were)\s+the\s+(?:steps?|procedure|process)\b",
    re.IGNORECASE,
)

_ENTITY_FRAME_RE = re.compile(
    r"\btell\s+(?:me|us)\s+about\b"
    r"|\b(?:what|who)\s+(?:do|did|does)\s+(?:we|i|you|they)\s+know\s+about\b"
    r"|\bwho\s+(?:is|was|are|were)\b"
    r"|\bwhat\s+(?:is|was|are|were)\s+[A-Z]"
    r"|\b(?:anything|something|everything|info|information|details?|notes?)"
    r"\s+(?:about|on|regarding)\s+[A-Z]"
    r"|\babout\s+[A-Z][a-z]",
)

#: "Likely no-answer" markers (§31.1): the query asserts the requested
#: content is absent, so an honest ``insufficient`` beats a least-bad hit
#: (V5-31.09). Versioned marker list — extend only with a version bump.
_IMPOSSIBLE_RES = (
    re.compile(
        r"\b(?:i|we)\s+(?:never|didn'?t|did not|haven'?t|have not"
        r"|hasn'?t|has not)\s+(?:told|tell|said|say|mentioned|mention"
        r"|shared|share|gave|give|given|reported|written|wrote)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+(?:didn'?t|did not|haven'?t|have not)\s+(?:i|we)"
        r"\s+(?:say|tell|mention|share|write|note)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:something|anything|everything)\s+(?:i|we)\s+"
        r"(?:never|haven'?t|didn'?t)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:you\s+(?:don'?t|do not)\s+know|nobody told|no record of)\b",
        re.IGNORECASE,
    ),
)

#: Fixed precedence for the ``primary`` class. ``no_answer_likely`` wins
#: whenever present (V5-31.09 — honest insufficient); ``temporal`` beats
#: ``identifier`` because bitemporal intent changes delivery semantics,
#: while the identifier still drives exact-match dominance inside the
#: class (V5-30.17). ``factual`` is the declared default.
_PRECEDENCE: Tuple[QueryClass, ...] = (
    QueryClass.NO_ANSWER_LIKELY,
    QueryClass.TEMPORAL,
    QueryClass.IDENTIFIER,
    QueryClass.PREFERENCE,
    QueryClass.PROCEDURAL,
    QueryClass.ENTITY,
    QueryClass.FACTUAL,
)


def _content_terms(text: str) -> List[str]:
    """Signal tokens: alnum-bearing, deduplicated, stop/meta filtered."""
    terms: List[str] = []
    seen = set()
    for tok in _TOKEN_RE.findall(str(text or "")):
        tok = tok.strip("'’.-_")
        if not tok or not _ALNUM_RE.search(tok):
            continue
        low = tok.lower()
        if low in _STOPWORDS or low in _META_TERMS:
            continue
        if low in seen:
            continue
        seen.add(low)
        terms.append(low)
    return terms


def _temporal_intent(query: str) -> Tuple[Tuple[str, ...], Optional[str]]:
    """All matched temporal markers plus the resolved bitemporal hint."""
    markers: List[str] = []
    for hint, rx in (
        (HINT_KNOWN_AT, _KNOWN_AT_RE),
        (HINT_TIMELINE, _TIMELINE_RE),
        (HINT_CURRENT, _CURRENT_RE),
    ):
        for m in rx.finditer(query):
            val = re.sub(r"\s+", " ", m.group(0).lower()).strip()
            if val not in markers:
                markers.append(val)
    hint: Optional[str] = None
    if _KNOWN_AT_RE.search(query):
        hint = HINT_KNOWN_AT
    elif _TIMELINE_RE.search(query):
        hint = HINT_TIMELINE
    elif _CURRENT_RE.search(query):
        hint = HINT_CURRENT
    return tuple(markers), hint


@dataclass(frozen=True)
class QueryAnalysis:
    """Versioned deterministic query classification (``query_analysis/v1``).

    ``classes`` is the precedence-ordered tuple of every matched
    ``QueryClass`` value; ``primary == classes[0]``. ``temporal_hint`` is
    set only when ``temporal`` is among the classes: ``current`` |
    ``timeline`` | ``known_at`` (V5-31.02). ``identifiers``/``entities``
    carry the deterministic §30.4 extractions that drove classification.
    """

    version: str = QUERY_ANALYSIS_VERSION
    classes: Tuple[str, ...] = (QueryClass.FACTUAL.value,)
    primary: str = QueryClass.FACTUAL.value
    identifiers: Tuple[Tuple[str, str], ...] = ()  # (kind, value)
    entities: Tuple[str, ...] = ()
    terms: Tuple[str, ...] = ()
    temporal_hint: Optional[str] = None
    temporal_markers: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """Receipt-loggable serialization (V5-31.01)."""
        return {
            "version": self.version,
            "classes": list(self.classes),
            "primary": self.primary,
            "identifiers": [
                {"kind": k, "value": v} for k, v in self.identifiers
            ],
            "entities": list(self.entities),
            "terms": list(self.terms),
            "temporal_hint": self.temporal_hint,
            "temporal_markers": list(self.temporal_markers),
            "warnings": list(self.warnings),
        }


def analyze(query: str) -> QueryAnalysis:
    """Classify ``query`` into ``QueryClass`` values — deterministic and
    side-effect free (query_analysis/v1).

    Multi-label: every matched class is reported in precedence order;
    ``primary`` is the head of that order. An identifier-bearing temporal
    query (``when was deploy-v2 released``) reports both. A query with no
    content signal, or one asserting its own absence ("what did I never
    tell you"), is ``no_answer_likely``.
    """
    text = str(query or "")
    warnings: List[str] = []

    identifiers = tuple(extract_identifiers(text))
    entities = tuple(extract_entities(text))
    terms = tuple(_content_terms(text))
    markers, hint = _temporal_intent(text)

    detected = set()
    empty = not terms and not identifiers and not entities
    if empty:
        detected.add(QueryClass.NO_ANSWER_LIKELY)
        warnings.append("no_signal")
    else:
        if any(rx.search(text) for rx in _IMPOSSIBLE_RES):
            detected.add(QueryClass.NO_ANSWER_LIKELY)
        if hint is not None:
            detected.add(QueryClass.TEMPORAL)
        if identifiers:
            detected.add(QueryClass.IDENTIFIER)
        if _PREFERENCE_RE.search(text):
            detected.add(QueryClass.PREFERENCE)
        if _PROCEDURAL_RE.search(text):
            detected.add(QueryClass.PROCEDURAL)
        if entities or _ENTITY_FRAME_RE.search(text):
            detected.add(QueryClass.ENTITY)
        if not detected:
            detected.add(QueryClass.FACTUAL)

    classes = tuple(c.value for c in _PRECEDENCE if c in detected)
    return QueryAnalysis(
        version=QUERY_ANALYSIS_VERSION,
        classes=classes,
        primary=classes[0],
        identifiers=tuple((m.kind, m.value) for m in identifiers),
        entities=tuple(m.value for m in entities),
        terms=terms,
        temporal_hint=hint if QueryClass.TEMPORAL in detected else None,
        temporal_markers=markers,
        warnings=tuple(warnings),
    )
