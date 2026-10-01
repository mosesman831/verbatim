"""Deterministic verbatim span harvester (SPEC §10, §11).

The harvester proposes *exact* byte spans of an accepted source revision. It
never paraphrases, repairs grammar, expands pronouns, or invents subjects —
every emitted candidate reproduces the stored quotation exactly, and every
offset lands on a UTF-8 character boundary. Hint flags (negation, condition,
modality, sensitivity, context-dependence) are *advisory metadata* for the
policy reducer, not semantic judgments; downstream code may not treat them as
proof of anything.

All regexes and the segmentation procedure are versioned together as
``HARVESTER_VERSION`` so a candidate set is reproducible for a given version.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Optional

from .types import (
    ErrorCode,
    Provenance,
    SourceEnvelope,
    SourceKind,
    SpanRef,
    VerbatimError,
    require_id,
)

HARVESTER_VERSION = "harvest-1"

#: Member roles a context group accepts (SPEC_V2 §12.05): the persisted group
#: vocabulary — a span may be primary evidence and simultaneously carry the
#: facet roles its interpretation depends on.
CONTEXT_MEMBER_ROLES = (
    "primary",
    "attribution",
    "antecedent",
    "condition",
    "negation",
    "temporal",
)


@dataclass(frozen=True)
class Candidate:
    """One proposed verbatim span plus advisory hint flags (SPEC §11).

    ``start_byte``/``end_byte`` index the *source payload bytes* and reproduce
    the exact quotation when decoded as UTF-8.

    ``kind`` is one of:

    - ``"sentence"``   — a piece produced by sentence splitting that ends at a
      detected sentence boundary.
    - ``"statement"``  — a piece of a split paragraph that has no sentence
      terminator (unterminated tail).
    - ``"paragraph"``  — a paragraph retained whole (either never split, or
      split pieces merged back because splitting would orphan a fragment).
    - ``"list_item"``  — a line carrying a list marker, kept whole.

    V2 context hints (SPEC_V2 §12): ``context_roles`` lists extra
    context-member roles this span carries for a group (``negation``,
    ``condition``, ``temporal``) — the same advisory flags, named so
    persistence can record *which* interpretation facets the span supplies.
    ``needs_antecedent`` marks a span that references an earlier statement
    (short reply, pronoun-dominant fragment) and cannot stand alone.
    """

    start_byte: int
    end_byte: int
    kind: str
    context_needed: bool
    sensitive_hint: bool
    negated: bool
    has_condition: bool
    modality_hint: Optional[str]
    reason: Optional[str]
    context_roles: tuple = ()
    needs_antecedent: bool = False


@dataclass(frozen=True)
class HarvestResult:
    """Harvester output: candidates in source order plus overflow accounting.

    ``overflow_count`` is the number of eligible candidates beyond
    ``max_candidates``; they must enter a local overflow job, never be silently
    discarded (SPEC §11). ``skipped`` counts segments that were segmented but
    not emitted — fenced code blocks plus pieces dropped by the size bounds —
    so rejection accounting stays visible.
    """

    candidates: tuple[Candidate, ...]
    overflow_count: int
    skipped: int = 0


@dataclass(frozen=True)
class PersistedHarvest:
    """Receipt of :func:`persist_harvest` (SPEC_V2 §12, §37).

    ``spans`` are the persisted span references in candidate order — the
    ingest worker feeds them to ``claims.propose`` directly. ``created`` is
    False when the operation key replayed an already-committed harvest, so a
    retry never reports a second logical effect.
    """

    group_id: str
    spans: tuple[SpanRef, ...]
    completeness: str
    created: bool
    skipped: int
    overflow_count: int


# ---------------------------------------------------------------------------
# Versioned hint regexes (harvest-1)
# ---------------------------------------------------------------------------
# These are deliberately simple and documented. They are *hints*: a false
# negative leaves an unflagged quotation (safe), a false positive only adds
# conservative metadata downstream.

#: Negation cues. ``n't\b`` catches contracted forms (don't, can't, won't) that
#: ``\bnot\b`` cannot see because the apostrophe breaks the word boundary.
NEGATION_RE = re.compile(
    r"n't\b|\bnot\b|\bno\s+longer\b|\bnever\b|\bwithout\b|\bcannot\b"
    r"|\bnobody\b|\bnothing\b|\bnowhere\b|\bneither\b",
    re.IGNORECASE,
)

#: Condition clauses that map to a canonical context value (SPEC §11 requires
#: preserving "at work", "unless", "when travelling", "for personal projects").
#: Only clauses with a bounded, mechanically comparable reading are listed;
#: anything else stays a cue on CONDITION_OTHER_RE and yields no structure.
CONDITION_CUES: tuple[tuple["re.Pattern[str]", str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), value)
    for pattern, value in (
        (r"\bat\s+work\b", "work"),
        (r"\bfor\s+work\b", "work"),
        (r"\bat\s+(?:the\s+)?office\b", "work"),
        (r"\bduring\s+(?:the\s+)?work\s*hours?\b", "work"),
        (r"\bat\s+home\b", "home"),
        (r"\bfrom\s+home\b", "home"),
        (r"\bfor\s+(?:my\s+)?personal\s+projects?\b", "personal_projects"),
        (r"\bfor\s+(?:my\s+)?side\s+projects?\b", "personal_projects"),
        (r"\bwhen\s+(?:I(?:'m|\s+am)\s+)?travell?ing\b", "travel"),
        (r"\bwhile\s+travell?ing\b", "travel"),
        (r"\bon\s+vacation\b", "travel"),
        (r"\bon\s+weekends?\b", "weekends"),
        (r"\bon\s+weekdays?\b", "weekdays"),
    )
)

#: Condition cues that are real qualifiers but have no bounded mapping —
#: presence is recorded (``has_condition``) yet the claim condition stays
#: ``None`` rather than an implicit unconditional interpretation (SPEC §12).
CONDITION_OTHER_RE = re.compile(
    r"\bunless\b|\bif\s+I\b|\bduring\b|\bwhen\b|\bwhenever\b",
    re.IGNORECASE,
)

#: Modality cues → 'hypothetical' | 'habitual' | 'uncertain'. The leftmost
#: matching cue wins, on the theory that the earliest hedge frames the whole
#: statement ("I think I might …" reads as uncertain-first is debatable, but
#: deterministic either way — the ordering is part of the versioned contract).
MODALITY_CUES: tuple[tuple["re.Pattern[str]", str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), value)
    for pattern, value in (
        (r"\bused\s+to\b", "habitual"),
        (r"\busually\b|\bgenerally\b|\btend\s+to\b", "habitual"),
        (
            r"\bmight\b|\bmay\b|\bcould\b|\bwould\b|\bplan\s+to\b"
            r"|\bplanning\s+to\b|\bhope\s+to\b|\bconsidering\b|\bif\s+I\b|\bwill\b",
            "hypothetical",
        ),
        (
            r"\bI\s+think\b|\bI\s+guess\b|\bI\s+believe\b|\bprobably\b"
            r"|\bmaybe\b|\bperhaps\b|\bnot\s+sure\b",
            "uncertain",
        ),
    )
)

#: Sensitive-content hints. HEURISTIC ONLY (SPEC §13): this catches obvious
#: self-declared secrets ("my api key", "password", PEM headers) and common
#: card-number shapes. It cannot detect arbitrary credentials, rotated or
#: obfuscated secrets, or non-English disclosures, and MUST NOT be presented
#: as comprehensive secret detection — it exists to route likely-sensitive
#: material through human review, not to certify anything as clean.
SENSITIVE_RE = re.compile(
    r"\bpassword\b|\bpasswd\b|\bssn\b|\bsocial\s+security\b"
    r"|\bapi[_ -]?key\b|\btoken\s*[:=]|\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"
    r"|\bcredit\s+card\b|\bcard\s+number\b|\bprivate\s+key\b|\bseed\s+phrase\b"
    r"|\bsecret\b|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|\b(?:\d[ -]?){13,16}\b",
    re.IGNORECASE,
)

#: Explicit time cues (dates, relative days) that make a statement carry a
#: temporal facet. Versioned with the harvester; the *interpretation* of the
#: cue belongs to ``core.time`` — this only flags that one is present.
TIME_CUE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}|\d{4}-\d{2}|\b\d{4}\b"
    r"|\byesterday\b|\btoday\b|\btomorrow\b|\blast\s+week\b",
    re.IGNORECASE,
)

#: Short replies that cannot stand alone as facts (SPEC §11: resolve "yes",
#: "the latter", pronouns only through explicit context links).
_SHORT_REPLY_RE = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|no|nope|nah|sure|correct|right|exactly"
    r"|absolutely|the\s+latter|the\s+former|both|neither|it|its|this|that"
    r"|these|those|he|she|they|we|i)\b",
    re.IGNORECASE,
)

_CHAT_NOISE_RE = re.compile(
    r"^\s*(?:ok(?:ay)?(?:\s+(?:hi|bye|then|thanks?|cool|great))?"
    r"|thanks?|thank\s+you|lol|lmao|rofl|ha(?:ha)+|hehe"
    r"|sounds?\s+good|got\s+it|understood|noted|nice|cool|great"
    r"|awesome|perfect|alright|all\s+right|no\s+problem"
    r"|you(?:'re|\s+are)\s+welcome|hi|hello|hey|bye|goodbye"
    r"|good\s+(?:morning|afternoon|evening|night))\s*[.!?]*\s*$",
    re.IGNORECASE,
)

_PRONOUNS = frozenset(
    "i me my mine you your yours he him his she her hers it its "
    "they them their theirs we us our ours this that these those one ones".split()
)
_WORD_RE = re.compile(r"[A-Za-z']+")

#: List-item markers: ``-``, ``*``, ``•``, or ``1.``/``2)`` style numerals.
_LIST_RE = re.compile(r"^[ \t]*(?:[-*•]|\d{1,3}[.)])(?=[ \t]|$)")

#: Blank-line paragraph boundary.
_BLANK_RE = re.compile(r"\n[ \t\r]*\n")

#: Characters that can terminate a sentence.
_ASCII_TERMINATORS = ".!?"
_CJK_TERMINATORS = "。！？"
_ELLIPSIS = "…"
_TERMINATORS = _ASCII_TERMINATORS + _CJK_TERMINATORS + _ELLIPSIS

#: Closing quotes/brackets absorbed into the end of a sentence, plus repeat
#: emphasis marks ("Really?!").
_CLOSERS = set("\"'”’)]}»》」』）】?!")

#: Characters that mark a piece as *terminated* — a complete bounded
#: statement rather than a dangling fragment. Includes closers so a sentence
#: ending `"done."` still counts as terminated after quote absorption.
_TERMINATED_END = set(_TERMINATORS) | _CLOSERS

#: Abbreviations whose trailing period is not a sentence boundary. Versioned
#: with the harvester; heuristics, not a dictionary of English.
_ABBREVIATIONS = frozenset(
    (
        "mr mrs ms dr st vs jr sr prof gen col sgt capt lt cmdr rev hon pres "
        "gov sen rep fig figs approx appt apt dept est min max inc ltd corp co "
        "etc viz cf al ca ed eds vol vols pp p ibid op cit no nos "
        "e.g i.e a.m p.m u.s u.k u.n e.u d.c "
        "jan feb mar apr jun jul aug sep sept oct nov dec "
        "mon tue tues wed thu thur thurs fri sat sun"
    ).split()
)


def _byte_offsets(text: str) -> list[int]:
    """Cumulative UTF-8 byte offset for every char index (incl. ``len(text)``).

    Guarantees emitted byte offsets are exact character boundaries by
    construction rather than by post-hoc validation.
    """
    offsets = [0] * (len(text) + 1)
    total = 0
    for i, ch in enumerate(text):
        total += len(ch.encode("utf-8"))
        offsets[i + 1] = total
    return offsets


def _code_regions(text: str) -> list[tuple[int, int]]:
    """Char ranges of fenced ``` ... ``` blocks (unclosed fence runs to EOF).

    Fenced code is excluded from personal-fact harvesting (SPEC §11). Regions
    are still *counted* so skipped-span accounting is complete and offsets of
    later candidates are never rebased.
    """
    regions: list[tuple[int, int]] = []
    start: Optional[int] = None
    pos = 0
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            if start is None:
                start = pos
            else:
                regions.append((start, pos + len(line)))
                start = None
        pos += len(line)
    if start is not None:
        regions.append((start, len(text)))
    return regions


def _non_code_segments(text: str, regions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Complement of ``regions`` within ``[0, len(text))``, in order."""
    out: list[tuple[int, int]] = []
    cursor = 0
    for a, b in regions:
        if cursor < a:
            out.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < len(text):
        out.append((cursor, len(text)))
    return out


def _paragraphs(text: str, lo: int, hi: int) -> list[tuple[int, int]]:
    """Split ``text[lo:hi]`` on blank lines."""
    out: list[tuple[int, int]] = []
    start = lo
    for m in _BLANK_RE.finditer(text, lo, hi):
        if m.start() > start:
            out.append((start, m.start()))
        start = m.end()
    if start < hi:
        out.append((start, hi))
    return out


def _units(text: str, lo: int, hi: int) -> list[tuple[int, int, bool]]:
    """Split a paragraph into list-item lines and runs of plain text."""
    out: list[tuple[int, int, bool]] = []
    pos = lo
    buf: Optional[int] = None
    while pos < hi:
        nl = text.find("\n", pos, hi)
        line_end = hi if nl == -1 else nl + 1
        if _LIST_RE.match(text[pos:line_end]):
            if buf is not None:
                out.append((buf, pos, False))
                buf = None
            out.append((pos, line_end, True))
        elif buf is None:
            buf = pos
        pos = line_end
    if buf is not None:
        out.append((buf, hi, False))
    return out


def _after_ok(text: str, j: int, hi: int) -> bool:
    """Whether a terminator ending at ``j`` is followed by a real new sentence.

    Requires end-of-text or whitespace + an uppercase/digit (optionally behind
    an opening quote). Lowercase letters and caseless scripts return False —
    conservative: missing a boundary is safer than inventing one.
    """
    k = j
    while k < hi and text[k] in " \t\n\r":
        k += 1
    if k >= hi:
        return True
    c = text[k]
    if c.isupper() or c.isdigit():
        return True
    if c in "\"'“‘([{<":
        k += 1
        while k < hi and text[k] in " \t":
            k += 1
        return k < hi and (text[k].isupper() or text[k].isdigit())
    return False


def _dot_boundary(text: str, i: int, hi: int) -> bool:
    """Whether the '.' at ``i`` ends a sentence (SPEC §11 heuristics).

    Declines the boundary on ellipses, decimals (3.14), versioned
    abbreviations (Dr., U.S., i.e.), and single-capital initials (J. R. R.).
    URL internals never reach here because they are not followed by
    whitespace+capital.
    """
    if i + 1 < hi and text[i + 1] == ".":
        return False  # start of ".." ellipsis
    if i > 0 and text[i - 1] == ".":
        return False  # inside ".."
    if i > 0 and i + 1 < hi and text[i - 1].isdigit() and text[i + 1].isdigit():
        return False  # decimal
    j = i - 1
    while j >= 0 and (text[j].isalpha() or text[j] == "."):
        j -= 1
    token = text[j + 1 : i]
    if token.lower() in _ABBREVIATIONS:
        return False
    if len(token) == 1 and token.isupper():
        return False  # personal initial
    return _after_ok(text, i + 1, hi)


def _consume_close(text: str, j: int, hi: int) -> int:
    """Extend a sentence end past closing quotes/brackets and repeat ?! marks."""
    while j < hi and text[j] in _CLOSERS:
        j += 1
    return j


def _sentence_spans(text: str, lo: int, hi: int) -> list[tuple[int, int]]:
    """Sentence char-spans inside ``text[lo:hi]``; the unterminated tail
    (if non-blank) is appended as a final span so nothing is silently lost."""
    spans: list[tuple[int, int]] = []
    start = lo
    i = lo
    while i < hi:
        ch = text[i]
        if ch == ".":
            if _dot_boundary(text, i, hi):
                end = _consume_close(text, i + 1, hi)
                spans.append((start, end))
                start = end
                i = end
                continue
        elif ch in _CJK_TERMINATORS:
            end = _consume_close(text, i + 1, hi)
            spans.append((start, end))
            start = end
            i = end
            continue
        elif ch == _ELLIPSIS:
            if _after_ok(text, i + 1, hi):
                end = _consume_close(text, i + 1, hi)
                spans.append((start, end))
                start = end
                i = end
                continue
        elif ch in "?!":
            # A mixed run like "?!" is expressive emphasis, not a sentence
            # boundary — "What?! That changed" is one bounded statement, and
            # splitting it would orphan a meaningless fragment (SPEC §11).
            # Pure runs ("Really??", "Stop!") still terminate normally.
            j = i
            saw_q = saw_x = False
            while j < hi and text[j] in "?!":
                saw_q = saw_q or text[j] == "?"
                saw_x = saw_x or text[j] == "!"
                j += 1
            if not (saw_q and saw_x) and _after_ok(text, j, hi):
                end = _consume_close(text, j, hi)
                spans.append((start, end))
                start = end
                i = end
                continue
            i = j
            continue
        i += 1
    if start < hi and text[start:hi].strip():
        spans.append((start, hi))
    return spans


def _strip_span(text: str, a: int, b: int) -> tuple[int, int]:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def _context_needed(text_slice: str, min_len: int) -> bool:
    """Short-reply rule: below ``min_len`` AND (yes/no/latter starter OR
    pronoun-dominant). Longer pieces are assumed self-contained for candidacy;
    deeper resolution is left to explicit context links (SPEC §11)."""
    if len(text_slice) >= min_len:
        return False
    if _SHORT_REPLY_RE.match(text_slice):
        return True
    words = _WORD_RE.findall(text_slice.lower())
    if not words or len(words) > 8:
        return False
    pronouns = sum(1 for w in words if w in _PRONOUNS)
    return pronouns / len(words) >= 0.5


def _modality_hint(text_slice: str) -> Optional[str]:
    """Leftmost modality cue wins (deterministic framing order)."""
    best: Optional[tuple[int, str]] = None
    for rx, value in MODALITY_CUES:
        m = rx.search(text_slice)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), value)
    return best[1] if best else None


def _has_condition(text_slice: str) -> bool:
    return any(rx.search(text_slice) for rx, _ in CONDITION_CUES) or bool(
        CONDITION_OTHER_RE.search(text_slice)
    )


def _context_roles(text_slice: str, negated: bool, conditioned: bool) -> tuple:
    """Facet roles this span carries for its context group (SPEC_V2 §12.05).

    Deterministic order (negation, condition, temporal) — the tuple is part of
    the versioned harvest contract so persisted member rows are reproducible.
    """
    roles: list[str] = []
    if negated:
        roles.append("negation")
    if conditioned:
        roles.append("condition")
    if TIME_CUE_RE.search(text_slice):
        roles.append("temporal")
    return tuple(roles)


def harvest(
    payload: bytes,
    *,
    max_candidates: int = 32,
    min_len: int = 32,
    max_len: int = 1200,
) -> HarvestResult:
    """Segment ``payload`` into verbatim candidates (SPEC §11).

    ``payload`` MUST be well-formed UTF-8: strict decoding is deliberate — a
    malformed source is rejected by the caller rather than harvested with
    guessed offsets (SPEC §10). ``min_len``/``max_len`` are character bounds;
    byte offsets are derived through the cumulative offset table so they always
    land on character boundaries.

    Size policy: meaningful short chat statements are retained even without
    terminal punctuation; a bounded acknowledgement/greeting set is skipped.
    Short replies that need prior context remain candidates with
    ``context_needed=True``. Pieces over ``max_len`` are not truncated — they
    are counted in ``skipped`` and left for bounded re-segmentation, since a
    truncated quotation would be a false representation of the source.
    """
    if max_candidates < 1 or min_len < 1 or max_len < min_len:
        raise ValueError("invalid candidate bounds")
    text = payload.decode("utf-8")  # strict: UnicodeDecodeError propagates
    byte_off = _byte_offsets(text)
    code = _code_regions(text)
    skipped = len(code)  # fenced blocks are segmented but never emitted

    found: list[Candidate] = []
    for seg_a, seg_b in _non_code_segments(text, code):
        for p_a, p_b in _paragraphs(text, seg_a, seg_b):
            for u_a, u_b, is_list in _units(text, p_a, p_b):
                if is_list:
                    pieces = [(_strip_span(text, u_a, u_b), "list_item")]
                else:
                    spans = [
                        s
                        for s in (
                            _strip_span(text, a, b)
                            for a, b in _sentence_spans(text, u_a, u_b)
                        )
                        if s[0] < s[1]
                    ]
                    if not spans:
                        continue
                    # A *terminated* trailing sentence below min_len is a
                    # fragment of the preceding statement, not a candidate of
                    # its own — merge it back (SPEC §11 "retain the whole
                    # bounded statement"). An unterminated tail instead stays
                    # separate as 'statement': merging it would fabricate a
                    # boundary the source does not have.
                    if len(spans) > 1:
                        ta, tb = spans[-1]
                        if (
                            tb - ta < min_len
                            and text[tb - 1] in _TERMINATED_END
                        ):
                            spans[-2] = (spans[-2][0], tb)
                            spans.pop()
                    single = len(spans) == 1
                    pieces = [
                        (
                            (a, b),
                            "paragraph"
                            if single
                            else (
                                "sentence"
                                if text[b - 1] in _TERMINATED_END
                                else "statement"
                            ),
                        )
                        for a, b in spans
                    ]
                for (a, b), kind in pieces:
                    if a >= b:
                        continue
                    piece = text[a:b]
                    need_ctx = _context_needed(piece, min_len)
                    # Size policy (SPEC §11): punctuation is not required in
                    # chat, so a short unterminated statement is evidence too.
                    # Drop only a bounded acknowledgement/greeting vocabulary;
                    # ambiguous replies still persist with an antecedent need.
                    if (
                        kind != "list_item"
                        and len(piece) < min_len
                        and len(pieces) == 1
                        and _CHAT_NOISE_RE.fullmatch(piece)
                    ):
                        skipped += 1
                        continue
                    if len(piece) > max_len:
                        skipped += 1
                        continue
                    negated = bool(NEGATION_RE.search(piece))
                    conditioned = _has_condition(piece)
                    found.append(
                        Candidate(
                            start_byte=byte_off[a],
                            end_byte=byte_off[b],
                            kind=kind,
                            context_needed=need_ctx,
                            sensitive_hint=bool(SENSITIVE_RE.search(piece)),
                            negated=negated,
                            has_condition=conditioned,
                            modality_hint=_modality_hint(piece),
                            reason="short_reply" if need_ctx else None,
                            context_roles=_context_roles(piece, negated, conditioned),
                            needs_antecedent=need_ctx,
                        )
                    )
    return HarvestResult(
        candidates=tuple(found[:max_candidates]),
        overflow_count=max(0, len(found) - max_candidates),
        skipped=skipped,
    )


#: Source kinds eligible for automatic harvesting by default: user-authored
#: text plus explicitly imported/operator records (SPEC §10: "automatic
#: harvesting admits user-authored text by default"). Assistant and tool
#: output require explicit caller opt-in through ``allow_kinds``.
_DEFAULT_ALLOW_KINDS = frozenset(
    {SourceKind.USER_MESSAGE, SourceKind.IMPORT, SourceKind.OPERATOR_RECORD}
)


def harvest_source(
    source: SourceEnvelope,
    *,
    allow_kinds: Optional[frozenset] = None,
    max_candidates: int = 32,
    min_len: int = 32,
    max_len: int = 1200,
) -> HarvestResult:
    """Policy-checked wrapper around :func:`harvest`.

    ``source_kind`` must be in ``allow_kinds`` (default: user messages,
    imports, operator records). Assistant-provenance material additionally
    requires ``SourceKind.ASSISTANT_MESSAGE`` in ``allow_kinds`` — provenance
    is an input to policy, and assistant text is optional contextual evidence
    only, never harvested by default (SPEC §10). Ineligible sources return an
    empty result rather than raising: rejection is a policy outcome, not a
    crash.
    """
    kinds = frozenset(allow_kinds) if allow_kinds is not None else _DEFAULT_ALLOW_KINDS
    if source.source_kind not in kinds:
        return HarvestResult((), 0)
    if (
        source.provenance == Provenance.ASSISTANT_GENERATED
        and SourceKind.ASSISTANT_MESSAGE not in kinds
    ):
        return HarvestResult((), 0)
    return harvest(
        source.payload,
        max_candidates=max_candidates,
        min_len=min_len,
        max_len=max_len,
    )


# ---------------------------------------------------------------------------
# Context-group persistence (SPEC_V2 §12, §37)
# ---------------------------------------------------------------------------


def _span_id_for(
    source_id: str, revision: int, start_byte: int, end_byte: int, parser_version: str
) -> str:
    """Deterministic span identity (SPEC_V2 §12.12).

    Replaying a source under the same harvester version must yield identical
    span ids — so identity derives from (source, revision, bounds, parser)
    rather than a random UUID. The excerpt bytes are bound separately by the
    ``spans.excerpt_hmac`` checksum.
    """
    digest = hashlib.sha256(
        f"{source_id}\x00{revision}\x00{start_byte}\x00{end_byte}"
        f"\x00{parser_version}".encode("utf-8")
    ).hexdigest()
    return f"sp_{digest[:32]}"


def _group_id_for(scope_id: str, operation_key: str) -> str:
    """Deterministic group identity for one (scope, operation) pair."""
    digest = hashlib.sha256(f"{scope_id}\x00{operation_key}".encode("utf-8")).hexdigest()
    return f"cg_{digest[:32]}"


def persist_harvest(
    store: Any,
    conn: sqlite3.Connection,
    envelope: SourceEnvelope,
    result: HarvestResult,
    operation_key: str,
    *,
    parser_version: str = HARVESTER_VERSION,
) -> PersistedHarvest:
    """Persist a harvest: spans + one context group + members, atomically.

    Runs entirely inside the caller's write transaction ``conn`` (SPEC_V2
    §11.13: source persistence, the acceptance event, and the processing
    obligation commit together — this call supplies the span/group side of
    that obligation). The ingest worker is expected to call this where it
    currently inserts spans by hand:

    .. code-block:: python

        with store.tx() as conn:
            ph = persist_harvest(store, conn, envelope, result,
                                 f"harvest:{source_id}:{revision}:{HARVESTER_VERSION}")

    Idempotence (SPEC_V2 §12.12): span ids and the group id derive
    deterministically from source/revision/bounds/parser and the operation
    key, so a replay under the same version produces identical identities and
    no additional rows. ``operation_key`` is also written onto each new span
    row for downstream dedup/audit.

    A candidate flagged ``needs_antecedent`` links the immediately preceding
    candidate span as an ``antecedent`` member — a deterministic local link,
    marked ``required=0`` because the true antecedent may live in another
    source. If no preceding span exists the group records the unresolved
    dependency and degrades to ``partial`` completeness (SPEC_V2 §12.06).
    """
    from ..storage import repos as _repos
    from ..storage import repos_v2 as _repos_v2

    if envelope.source_id is None:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "persist_harvest requires envelope.source_id (persist the source first)",
        )
    require_id(operation_key, "operation_key")
    require_id(parser_version, "parser_version")
    source_id = envelope.source_id
    # The authoritative partition is the source's persisted scope_id —
    # re-deriving from the caller's envelope.scope could map to a
    # different scope row (normalization, principal/visibility variants)
    # and would plant derived evidence across the partition boundary.
    row = conn.execute(
        "SELECT scope_id FROM sources WHERE source_id = ?", (source_id,)
    ).fetchone()
    if row is None:
        # The source row was not persisted in this tx — fall back to the
        # envelope-derived scope so standalone callers still work.
        scope_id = _repos.ensure_scope(store, conn, envelope.scope)
    else:
        scope_id = row[0]
        _repos.ensure_scope(store, conn, envelope.scope)
    spans_repo = _repos.SpansRepo(store)
    groups = _repos_v2.ContextGroupsRepo(store)

    # --- spans ---------------------------------------------------------------
    span_refs: list[SpanRef] = []
    for cand in result.candidates:
        sid = _span_id_for(
            source_id, envelope.revision, cand.start_byte, cand.end_byte, parser_version
        )
        exists = conn.execute(
            "SELECT 1 FROM spans WHERE span_id = ?", (sid,)
        ).fetchone()
        if exists is None:
            spans_repo.insert(
                sid,
                source_id,
                envelope.revision,
                cand.start_byte,
                cand.end_byte,
                parser_version,
                conn=conn,
            )
            conn.execute(
                "UPDATE spans SET operation_key = ? WHERE span_id = ?",
                (operation_key, sid),
            )
        span_refs.append(
            SpanRef(sid, source_id, envelope.revision, cand.start_byte, cand.end_byte)
        )

    # --- completeness ----------------------------------------------------------
    # Missing local antecedents and overflowed material both degrade the
    # group to 'partial' — the receipt distinguishes those from deliberately
    # excluded (skipped, reported by the caller) and deferred ranges.
    unresolved_antecedent = any(
        i == 0 and c.needs_antecedent for i, c in enumerate(result.candidates)
    )
    completeness = (
        "partial"
        if (unresolved_antecedent or result.overflow_count > 0)
        else "complete"
    )

    # --- group + members -------------------------------------------------------
    created = (
        conn.execute(
            "SELECT 1 FROM context_groups WHERE scope_id = ? AND operation_key = ?",
            (scope_id, operation_key),
        ).fetchone()
        is None
    )
    group_id = groups.create(
        conn,
        scope_id,
        source_id,
        envelope.revision,
        parser_version=parser_version,
        operation_key=operation_key,
        completeness=completeness,
        group_id=_group_id_for(scope_id, operation_key),
    )
    previous_span: Optional[str] = None
    for i, cand in enumerate(result.candidates):
        sid = span_refs[i].span_id
        reason = cand.reason
        if cand.needs_antecedent and not reason:
            reason = "needs_antecedent"
        groups.add_member(
            conn,
            group_id,
            sid,
            role="primary",
            required=True,
            ord=i,
            dependency_reason=reason,
        )
        for role in cand.context_roles:
            if role not in CONTEXT_MEMBER_ROLES or role == "primary":
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    f"invalid context role {role!r} on candidate {i}",
                )
            groups.add_member(
                conn,
                group_id,
                sid,
                role=role,
                required=False,
                ord=i,
                dependency_reason="advisory_hint",
            )
        if cand.needs_antecedent and previous_span is not None:
            groups.add_member(
                conn,
                group_id,
                previous_span,
                role="antecedent",
                required=False,
                ord=i - 1,
                dependency_reason="prior_candidate",
            )
        previous_span = sid

    return PersistedHarvest(
        group_id=group_id,
        spans=tuple(span_refs),
        completeness=completeness,
        created=created,
        skipped=result.skipped,
        overflow_count=result.overflow_count,
    )
