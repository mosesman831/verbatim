"""Deterministic identifier + candidate-entity extraction (§7, V5-30.14).

Pure regex/rule extraction over the raw text — no store, no model, no
normalization of the emitted value. Every hit is ``(kind, value, start,
end)`` where ``start``/``end`` are UTF-8 **byte** offsets into the
original text and ``value`` is the exact byte slice (case and
punctuation preserved exactly, V5-30.17).

Identifier kinds: ``url``, ``email``, ``path``, ``handle``, ``ticket``,
``hash``, ``version``, ``code``, ``quoted``. Overlapping candidate spans
are deduplicated deterministically: a fixed kind priority decides
cross-kind overlaps (specific structural kinds beat the generic
``quoted`` wrapper so ``"ABC-123"`` still yields the ticket), same-kind
overlaps keep the longest span, ties prefer the earliest start.

Entity kinds: ``capitalized_span`` (maximal run of capitalized tokens,
optionally joined by lowercase connectors like ``of``/``van``) and
``dictionary`` (a small pinned known-entity list — matching inputs only,
never identity merges, V5-30.15). Sentence-initial function words
("The", "However", ...) are suppressed by a fixed stop list.

Parser/producer version: ``extract/v1`` — pinned on emitted rows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from .normalize import utf8_offsets

EXTRACT_VERSION = "extract/v1"


@dataclass(frozen=True)
class Identifier:
    """One extracted identifier mention (§7). ``start``/``end`` are UTF-8
    byte offsets into the source text; ``value`` is the exact span."""

    kind: str
    value: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "value": self.value,
            "start": self.start, "end": self.end,
        }


@dataclass(frozen=True)
class Entity:
    """One candidate named-entity mention. ``kind`` is
    ``capitalized_span`` or ``dictionary``."""

    kind: str
    value: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "value": self.value,
            "start": self.start, "end": self.end,
        }


# ---------------------------------------------------------------- patterns

_URL_RE = re.compile(r"(?:https?|ftp|file)://[^\s<>\"'`]+", re.IGNORECASE)
_URL_TRAIL = ".,;:!?)]}>\"'”’"  # sentence punctuation wrongly captured at end

_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b"
)

#: @handle — must not be glued to a preceding word char / email local part.
_HANDLE_RE = re.compile(r"(?<![\w.+-])@[A-Za-z_][A-Za-z0-9_-]{0,30}\b")

#: Unix relative/absolute or Windows drive paths. Candidates are
#: post-validated (``_is_path``) so "and/or" is never a path. The first
#: alternative is a leading segment + ≥1 slash segment (``src/main.py``);
#: the second is anchored (``/etc/hosts``, ``~/x``, ``./x``, ``../x``);
#: the third is a Windows drive path.
_PATH_RE = re.compile(
    r"(?<![\w@/.])"
    r"(?:"
    r"[\w.@+~-]+(?:/[\w.@+~-]+)+/?"
    r"|(?:~|\.{1,2})?(?:/[\w.@+~-]+)+/?"
    r"|[A-Za-z]:\\(?:[^\x00-\x1f<>\"|?*\\/]+\\?)+"
    r")"
)
_PATH_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,10}/?$")

#: JIRA-style ticket keys: ≥2 uppercase/digit chars, dash, digits.
_TICKET_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,11}-\d{1,7}\b")

#: Hex hashes: ≥7 hex chars containing at least one digit, or a 0x-prefixed
#: hex string (≥4 digits after the prefix). The digit requirement keeps
#: English words made of a–f letters ("facaded", "acceded") out.
_HASH_RE = re.compile(r"\b(?:0x[0-9a-fA-F]{4,64}|[0-9a-fA-F]{7,64})\b")

#: Version strings: "v1.2.3", "1.2.3-rc.1", bare "v2", and name-suffixed
#: "deploy-v2" (whole token captured — the name segment is part of the
#: identifier for exact-match purposes, V5-30.17).
_VERSION_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9_.]*-[vV]\d+(?:\.\d+)*\b"
    r"|\b[vV]\d+(?:\.\d+){1,3}(?:-[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*)?"
    r"(?:\+[0-9A-Za-z.-]+)?\b"
    r"|\b[vV]\d+\b"
    r"|\b\d+\.\d+(?:\.\d+){0,2}(?:-[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*)?"
    r"(?:\+[0-9A-Za-z.-]+)?\b"
)

#: Quoted strings. Apostrophes adjacent to word chars can't open/close a
#: single-quoted span, so "don't" never pairs with a later quote.
_QUOTED_RE = re.compile(
    r"\"[^\"\n]{1,500}\""
    r"|“[^”\n]{1,500}”"
    r"|‘[^’\n]{1,500}’"
    r"|(?<![\w'])'[^'\n]{1,500}'(?![\w'])"
    r"|`[^`\n]{1,500}`"
)

#: Code-ish tokens: snake_case, dotted names (foo.bar, main.py),
#: lowerCamel / UpperCamel, letter+digit mixes (utf8, 2fa), and kebab
#: tokens containing a digit (node-18, py-3.11). A bare hyphenated word
#: without digits ("well-known") is deliberately not code.
_CODE_RES = (
    re.compile(r"\b\w+(?:\.\w+)+\b"),
    re.compile(r"\b[\w-]*_[\w-]*\b"),
    re.compile(r"\b[a-z0-9_]+(?:[A-Z][a-zA-Z0-9]*)+\b"),
    re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-zA-Z0-9]*)+\b"),
    re.compile(r"\b[A-Za-z]+\d+[A-Za-z0-9]*\b"),
    re.compile(r"\b\d+[A-Za-z]+[A-Za-z0-9]*\b"),
    re.compile(r"\b(?=[\w.-]*-)(?=[\w.-]*\d)[\w.-]+\b"),
)

#: Overlap precedence — lower wins. ``quoted`` is last: a quoted span is
#: emitted only when it wraps nothing more specific, so `"ABC-123"`
#: still produces the ticket identifier (specific beats wrapper).
_KIND_PRIORITY = {
    "url": 0, "email": 1, "path": 2, "handle": 3, "ticket": 4,
    "hash": 5, "version": 6, "code": 7, "quoted": 8,
}


def _is_path(span: str) -> bool:
    """Post-validate a unix-form path candidate (Windows forms always pass).

    Accepted: ``~``- or ``.``-relative (``~/x``, ``./x``, ``../x``),
    absolute paths with ≥2 segments (``/etc/hosts``), any path with ≥3
    segments (``a/b/c``), or a final segment with an extension
    (``src/main.py``). Rejected: bare ``/etc`` and two-segment words like
    ``and/or``.
    """
    if "\\" in span or ":" in span[:2]:
        return True
    rest = span
    anchored = rest.startswith(("/", "~", "."))
    if rest.startswith("~"):
        rest = rest[1:]
    elif rest.startswith("."):
        rest = rest.lstrip(".")
    segs = [s for s in rest.split("/") if s]
    if segs and all(s.isdigit() for s in segs):
        return False  # "2025/03/14" is a date-like token, never a path
    if not anchored and len(segs) < 3 and not _PATH_EXT_RE.search(rest):
        return False
    if rest.startswith("/") and not span.startswith(("~", ".")) \
            and len(segs) < 2:
        return False
    return True


def _collect_identifiers(text: str) -> List[Tuple[str, int, int]]:
    """All raw (kind, char_start, char_end) candidates, before overlap
    resolution."""
    cands: List[Tuple[str, int, int]] = []

    for m in _URL_RE.finditer(text):
        s, e = m.span()
        while e > s and text[e - 1] in _URL_TRAIL:
            e -= 1
        # an unbalanced "(" glued to the tail is prose punctuation, not
        # part of the URL ("(see https://x.io/a)")
        if text[s:e].count("(") > text[s:e].count(")"):
            e = text.rindex("(", s, e)
            while e > s and text[e - 1] in _URL_TRAIL:
                e -= 1
        if e > s + len("http://"):
            cands.append(("url", s, e))
    for m in _EMAIL_RE.finditer(text):
        cands.append(("email", *m.span()))
    for m in _HANDLE_RE.finditer(text):
        cands.append(("handle", *m.span()))
    for m in _PATH_RE.finditer(text):
        s, e = m.span()
        span = text[s:e]
        if "\\" in span:
            span = span.rstrip("\\")
            e = s + len(span)
        if span and _is_path(span):
            cands.append(("path", s, e))
    for m in _TICKET_RE.finditer(text):
        cands.append(("ticket", *m.span()))
    for m in _HASH_RE.finditer(text):
        tok = m.group(0)
        if tok.lower().startswith("0x"):
            cands.append(("hash", *m.span()))
        elif any("a" <= c.lower() <= "f" for c in tok) and any(
            c.isdigit() for c in tok
        ):
            # non-0x hex must carry BOTH a hex letter and a digit —
            # pure numbers ("1234567") are just numbers and pure
            # a–f strings ("facaded", "acceded") are just words.
            cands.append(("hash", *m.span()))
    for m in _VERSION_RE.finditer(text):
        cands.append(("version", *m.span()))
    for rx in _CODE_RES:
        for m in rx.finditer(text):
            cands.append(("code", *m.span()))
    for m in _QUOTED_RE.finditer(text):
        cands.append(("quoted", *m.span()))
    return cands


def _resolve_overlaps(
    cands: List[Tuple[str, int, int]],
) -> List[Tuple[str, int, int]]:
    """Deterministic non-overlap resolution.

    Order: kind priority, then earliest start, then longest span; keep a
    candidate only if it overlaps nothing already kept. Result is sorted
    by start.
    """
    order = sorted(
        set(cands),
        key=lambda c: (_KIND_PRIORITY[c[0]], c[1], -(c[2] - c[1])),
    )
    kept: List[Tuple[str, int, int]] = []
    for cand in order:
        _kind, s, e = cand
        if any(s < ke and ks < e for _k, ks, ke in kept):
            continue
        kept.append(cand)
    return sorted(kept, key=lambda c: (c[1], c[2]))


def extract_identifiers(text: str) -> List[Identifier]:
    """Extract all identifier mentions with UTF-8 byte offsets.

    Case and punctuation are preserved exactly (V5-30.17): ``value`` is
    always ``text.encode('utf-8')[start:end].decode('utf-8')``.
    Overlapping spans are deduplicated by fixed kind priority; the same
    surface span can never yield two identifiers.
    """
    t = str(text if text is not None else "")
    offsets = utf8_offsets(t)
    out = []
    for kind, cs, ce in _resolve_overlaps(_collect_identifiers(t)):
        out.append(Identifier(kind, t[cs:ce], offsets[cs], offsets[ce]))
    return out


# ---------------------------------------------------------------- entities

#: Letter-led tokens (may contain internal ' ’ - . — "O'Brien",
#: "Smith-Jones", "St. John").
_WORD_RE = re.compile(r"[^\W_][\w'’.-]*", re.UNICODE)

#: Lowercase connectors allowed strictly inside a capitalized run:
#: "Ruth van der Berg", "United States of America", "Alexander the
#: Great". ``and`` is deliberately excluded — "Alice and Bob" is a list
#: of two entities, never one.
_CONNECTORS = frozenset({
    "of", "the", "de", "del", "della", "van", "von", "der", "den", "da",
    "di", "la", "le",
})

#: Sentence-initial function words suppressed as run starters. Applied
#: only at a sentence boundary — mid-sentence "The" in "The Beatles"
#: still joins a run.
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

#: Tokens that are never entities anywhere ("I", "I'm", "A").
_NEVER = frozenset({"i", "a", "i'm", "i’ll", "i'll", "i’d", "i'd",
                    "i’ve", "i've"})

#: Small pinned known-entity dictionary (V5-30.14 "known-entity
#: dictionary hits"). Matching signal only — never an identity merge.
KNOWN_ENTITIES = frozenset({
    "python", "javascript", "typescript", "java", "rust", "golang",
    "ruby", "php", "swift", "kotlin", "scala", "c++", "c#",
    "objective-c",
    "redis", "postgresql", "postgres", "mysql", "sqlite", "mariadb",
    "mongodb", "elasticsearch", "kafka", "rabbitmq", "clickhouse",
    "linux", "ubuntu", "debian", "fedora", "macos", "windows",
    "android", "ios",
    "aws", "gcp", "azure", "google cloud", "ec2", "s3", "lambda",
    "docker", "kubernetes", "k8s", "terraform", "ansible", "jenkins",
    "github", "gitlab", "bitbucket", "jira", "confluence", "slack",
    "notion", "linear", "figma",
    "django", "flask", "fastapi", "rails", "react", "vue", "angular",
    "svelte", "nextjs", "node.js", "nodejs", "express",
    "pytorch", "tensorflow", "openai", "anthropic", "google",
    "microsoft", "apple", "amazon", "meta", "netflix", "nvidia",
    "new york", "san francisco", "los angeles", "san diego", "london",
    "paris", "berlin", "tokyo",
})

_DICT_RE = re.compile(
    r"(?<![\w])(?:"
    + "|".join(
        re.escape(e) for e in sorted(KNOWN_ENTITIES, key=len, reverse=True)
    )
    + r")(?![\w])",
    re.IGNORECASE,
)

_SENT_BOUNDARY = frozenset(".!?\n")


def _sentence_initial(text: str, start: int) -> bool:
    """True when the token at ``start`` opens a sentence: start of text or
    immediately after . ! ? or a newline."""
    i = start - 1
    while i >= 0 and text[i] in " \t\r":
        i -= 1
    return i < 0 or text[i] in _SENT_BOUNDARY


def _entity_overlaps(
    cands: List[Tuple[str, int, int]],
) -> List[Tuple[str, int, int]]:
    """Overlap resolution for entity candidates: longest span wins; ties
    prefer ``dictionary`` (more informative) then earliest start."""
    order = sorted(
        set(cands),
        key=lambda c: (-(c[2] - c[1]), 0 if c[0] == "dictionary" else 1,
                       c[1]),
    )
    kept: List[Tuple[str, int, int]] = []
    for cand in order:
        _kind, s, e = cand
        if any(s < ke and ks < e for _k, ks, ke in kept):
            continue
        kept.append(cand)
    return sorted(kept, key=lambda c: (c[1], c[2]))


def extract_entities(text: str) -> List[Entity]:
    """Candidate named entities with UTF-8 byte offsets.

    Two deterministic signals: maximal runs of capitalized tokens
    (optionally joined by ``of``/``van``-class connectors), and hits from
    the pinned ``KNOWN_ENTITIES`` dictionary (matched case-insensitively,
    surface preserved). Sentence-initial function words never start a
    run. Candidates only — merging/splitting is a reviewed operation
    elsewhere (V5-30.15).
    """
    t = str(text if text is not None else "")
    offsets = utf8_offsets(t)
    words = []
    for m in _WORD_RE.finditer(t):
        ws, wtok = m.start(), m.group(0)
        # Trailing punctuation glued to the token ("Berg.", "St.") is
        # not part of the word — trim it from the span end.
        trimmed = wtok.rstrip(".-")
        if not trimmed:
            continue
        words.append((ws, ws + len(trimmed), trimmed))
    cands: List[Tuple[str, int, int]] = []

    # capitalized runs — a run never crosses a sentence boundary and a
    # connector can only extend an open run (never start one).
    run_start = None  # type: int | None
    run_end = None    # type: int | None
    pending = None    # connector tail to trim: (start,end)
    for ws, we, wtok in words:
        first_upper = wtok[0].isupper()
        is_never = wtok.lower() in _NEVER
        is_conn = (
            wtok.lower() in _CONNECTORS
            and not first_upper
            and run_start is not None
        )
        sent_init = _sentence_initial(t, ws)
        suppressed = sent_init and wtok.lower() in _SENT_STOP
        boundary_break = run_start is not None and any(
            c in _SENT_BOUNDARY for c in t[run_end:ws]
        )
        participates = (
            (first_upper and not is_never and not suppressed) or is_conn
        ) and not boundary_break
        if participates:
            if run_start is None:
                run_start = ws
            run_end = we
            if is_conn:
                # first connector of a trailing chain — the run trims
                # back to its start if no capitalized word follows
                pending = pending or (ws, we)
            else:
                pending = None
        else:
            if run_start is not None:
                end = pending[0] if pending else run_end
                if end > run_start:
                    cands.append(("capitalized_span",
                                  run_start, end))
                run_start = run_end = pending = None
            # a non-participating token may still open a new run
            if (first_upper and not is_never and not suppressed
                    and not is_conn):
                run_start, run_end, pending = ws, we, None
    if run_start is not None:
        end = pending[0] if pending else run_end
        if end > run_start:
            cands.append(("capitalized_span", run_start, end))

    # dictionary hits
    for m in _DICT_RE.finditer(t):
        cands.append(("dictionary", *m.span()))

    out = []
    for kind, cs, ce in _entity_overlaps(cands):
        out.append(Entity(kind, t[cs:ce], offsets[cs], offsets[ce]))
    return out
