"""Deterministic query analysis (SPEC §29).

The analyzer never calls a generative rewriting service. It preserves
quoted phrases and exact identifiers (paths, camelCase, numbers), applies
only conservative explicit time cues, and returns a typed empty plan for
no-signal input — a query of stopwords or bare symbols must never become
"return everything" (SPEC §29). Ambiguous dates produce warnings, never
fabricated intervals (SPEC §14).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..core.time import parse_time_expression
from ..core.types import RecallRequest

# Common English function words carry no retrieval signal. Negation-bearing
# tokens (no/not/never/nor/without/none) are deliberately absent: dropping
# them would silently invert claim polarity (SPEC §5, §29).
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

# Conversational meta-verbs: tokens that address the memory system itself
# ("tell me about X", "do I remember Y", "what happened with Z"). They carry
# no content signal — stored evidence never contains them — and treating
# them as terms both dilutes the MATCH query and breaks term-coverage
# abstention (V2-29). Negation stays load-bearing; "forget" is a negation
# cue in disguise ("I forget whether…"), so it stays too.
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

# Signal test is Unicode-aware: any letter/number in any script counts as
# signal (V2-25.08 preserves script). An earlier ASCII-only test silently
# emptied Japanese/Arabic/Cyrillic queries — a no-signal result for a
# meaningful query is worse than a rare symbol false-positive.
_ALNUM_RE = re.compile(r"[^\W_]", re.UNICODE)
# Only double quotes delimit phrases: single quotes appear inside ordinary
# words ("don't"), where treating them as delimiters would fabricate
# nonsense phrases. This also matches FTS5's own phrase syntax.
_PHRASE_RE = re.compile(r'"([^"]*)"')

# Explicit, unambiguous time cues only: "in 2024-03", "in 2024", bare ISO
# dates/months, and the small relative set supported by
# ``parse_time_expression`` (yesterday/today/tomorrow/last week).
_TIME_CUE_RES = (
    re.compile(r"\bin\s+(\d{4}-\d{2}-\d{2})(?![-\d])"),
    re.compile(r"\bin\s+(\d{4}-\d{2})(?![-\d])"),
    re.compile(r"\bin\s+(\d{4})(?![-\d])"),
    re.compile(r"\b(\d{4}-\d{2}-\d{2})(?![-\d])"),
    re.compile(r"\b(\d{4}-\d{2})(?![-\d])"),
    re.compile(r"\b(yesterday|today|tomorrow|last week)\b"),
)

# Time-flavoured expressions ``parse_time_expression`` deliberately leaves
# unresolved. Detecting them lets the analyzer warn instead of guessing.
_AMBIGUOUS_TIME_RE = re.compile(
    r"\b(?:last|next|this)\s+"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"week|month|year|weekend|morning|evening|night)\b"
    r"|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"|\bin\s+(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b"
    r"|\b\d+\s+(?:day|week|month|year)s?\s+ago\b"
    r"|\bago\b",
    re.IGNORECASE,
)

# Bound on MATCH-string length so a maximal 8 KiB query cannot explode the
# FTS5 expression. Remaining terms stay available for fallback scoring.
_MAX_MATCH_TERMS = 64

_PREDICATE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("project_database", ("project database", "database", "datastore", "db")),
    ("editor", ("code editor", "text editor", "editor")),
    ("ide", ("integrated development environment", "ide")),
    (
        "residence",
        (
            "where do i live",
            "residence",
            "location",
            "address",
            "city",
            "live",
            "lives",
            "living",
        ),
    ),
    ("preference", ("preference", "prefer", "prefers", "favorite", "favourite")),
    ("schedule", ("schedule", "meeting", "appointment", "calendar")),
    ("language", ("programming language", "language")),
    ("os", ("operating system", "os")),
    ("tool_use", ("tool use", "tools")),
)


@dataclass
class QueryPlan:
    """Immutable-in-spirit output of ``analyze`` consumed by ``gather``.

    ``terms`` keeps every signal token (for fallback lexical scoring);
    ``match_query`` is the sanitized FTS5 expression built only from quoted
    literals — user syntax is treated as text, never as operators (§29).
    ``empty``/``reason`` type the no-signal case so callers return a typed
    empty result instead of scanning all memory.
    """

    terms: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    match_query: Optional[str] = None
    valid_at_us: Optional[int] = None
    valid_until_us: Optional[int] = None
    known_at_seq: Optional[int] = None
    entity_ids: tuple = ()
    predicates: tuple = ()
    predicate_terms: tuple = ()
    warnings: tuple = ()
    empty: bool = False
    reason: Optional[str] = None


def _fts_quote(token: str) -> str:
    """Wrap one token as an FTS5 quoted literal; ``"`` inside is doubled."""
    return '"' + token.replace('"', '""') + '"'


def _extract_phrases(query: str) -> tuple[list[str], str]:
    """Pull quoted phrases verbatim; return (phrases, remaining text)."""
    phrases: list[str] = []
    seen: set[str] = set()

    def _keep(match: re.Match) -> str:
        text = match.group(1).strip()
        if text and _ALNUM_RE.search(text) and text not in seen:
            seen.add(text)
            phrases.append(text)
        return " "

    rest = _PHRASE_RE.sub(_keep, query)
    return phrases, rest


def _extract_terms(text: str) -> list[str]:
    """Split on whitespace; drop stopwords and symbol-only tokens.

    Identifiers are preserved byte-for-byte — the FTS5 unicode61 tokenizer
    applies its own normalization, so no lossy folding happens here.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for token in text.split():
        token = token.strip()
        if not token or not _ALNUM_RE.search(token):
            continue  # symbol-only token: no signal
        lowered = token.lower()
        if lowered in _STOPWORDS or lowered in _META_TERMS:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        terms.append(token)
    return terms


def _extract_predicates(query: str, terms: list[str]) -> tuple[tuple, tuple]:
    lowered = query.casefold()
    term_keys = {}
    for term in terms:
        raw = term.casefold()
        term_keys[raw.strip(".,;:!?\"'()[]{}")] = raw
    predicates: list[str] = []
    pairs: list[tuple[str, str]] = []
    for predicate, aliases in _PREDICATE_ALIASES:
        for alias in aliases:
            if not re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", lowered):
                continue
            if predicate not in predicates:
                predicates.append(predicate)
            for word in alias.split():
                if word in term_keys:
                    pair = (term_keys[word], predicate)
                    if pair not in pairs:
                        pairs.append(pair)
            break
    return tuple(predicates), tuple(pairs)


def _extract_valid_at(query: str, request: RecallRequest, now_us: int,
                      warnings: list[str]) -> tuple[Optional[int], Optional[int]]:
    """Resolve the valid-time filter: explicit request fields first, then
    conservative query cues (SPEC §29 ordering).

    Returns ``(point_us, until_us)``. ``point_us`` is the range start (or a
    lone point); ``until_us`` is the range end when one is known — an
    explicit ``request.valid_until_us``, or the parsed interval's own end
    ("in 2024-03" means *during March*, not only at its first instant,
    SPEC_V2 §16). Conflicting parses and unparseable date language warn
    rather than guess (SPEC §14).
    """
    if request.valid_at_us is not None or request.valid_until_us is not None:
        return request.valid_at_us, request.valid_until_us

    intervals: list[tuple[int, int, Optional[int]]] = []
    for cue_re in _TIME_CUE_RES:
        for match in cue_re.finditer(query):
            iv = parse_time_expression(match.group(1), now_us)
            if iv is not None and iv.from_us is not None:
                pair = (iv.from_us, iv.until_us or -1, iv.until_us)
                if pair not in intervals:
                    intervals.append(pair)

    ambiguous = False
    for match in _AMBIGUOUS_TIME_RE.finditer(query):
        if parse_time_expression(match.group(0), now_us) is None:
            ambiguous = True
            break

    if len(intervals) > 1:
        ambiguous = True
    if ambiguous:
        warnings.append("ambiguous_time")
    # A single distinct interpretation is safe to use; several are not.
    if len(intervals) == 1:
        return intervals[0][0], intervals[0][2]
    return None, None


def analyze(query: str, request: RecallRequest, now_us: int) -> QueryPlan:
    """Build a QueryPlan from raw text plus the explicit request fields.

    Deterministic and side-effect free; ``now_us`` is injected so tests and
    replays observe identical plans.
    """
    warnings: list[str] = []
    phrases, rest = _extract_phrases(query)
    terms = _extract_terms(rest)
    predicates, predicate_terms = _extract_predicates(query, terms)
    valid_at_us, valid_until_us = _extract_valid_at(
        query, request, now_us, warnings
    )

    match_terms = (phrases + terms)[:_MAX_MATCH_TERMS]
    match_query = " OR ".join(_fts_quote(t) for t in match_terms) or None

    entity_ids = tuple(request.entity_ids)
    empty = match_query is None and not entity_ids
    return QueryPlan(
        terms=terms,
        phrases=phrases,
        match_query=match_query,
        valid_at_us=valid_at_us,
        valid_until_us=valid_until_us,
        known_at_seq=request.known_at_seq,
        entity_ids=entity_ids,
        predicates=predicates,
        predicate_terms=predicate_terms,
        warnings=tuple(warnings),
        empty=empty,
        reason="no_signal" if empty else None,
    )
