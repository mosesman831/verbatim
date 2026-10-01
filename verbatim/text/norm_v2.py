"""`norm/v2` — the single shared text analyzer for query AND document
sides (SPEC_V7 §32.1, V7-05.10/11).

The analyzer is a pure, deterministic projection — it never rewrites
stored bytes and never emits offsets that do not round-trip into the
original UTF-8 encoding of the input. Pipeline (§32.1 steps 1–7):

1. Strict UTF-8: ``bytes`` input is decoded strictly; ``str`` input must
   re-encode cleanly (unpaired surrogates raise ``UnicodeEncodeError``).
   Invalid sequences are rejected, never repaired.
2. The **identifier channel** is extracted first, on the *raw* text:
   URLs, emails, file paths, ``@handles``, ``#tags``, ticket ids
   (``[A-Z]{2,10}-\\d+``), hex hashes (>= 7), semantic versions, quoted
   code spans, and numbers with units. Identifier ``NormTerm`` rows keep
   the exact surface bytes + byte offsets and are excluded from folding;
   they are returned in ``NormAnalysis.identifiers`` (channel
   ``"identifier"``), never inside ``terms``.
3. The rest of the text is projected: NFKC -> casefold -> diacritic
   strip. Byte offsets of every emitted term map back to source bytes
   through a per-character provenance index + UTF-8 offset table.
4. Tokenize on Unicode word boundaries; ``'`` and ``’`` inside a word
   are clitic boundaries.
5. Clitic table (§32.1 rule 5): ``'s``/``’s`` -> drop, ``'d`` -> drop,
   ``n't`` -> ``not``, ``'re`` -> ``are``, ``'ll`` -> ``will``,
   ``'ve`` -> ``have``, ``'m`` -> ``am``. A piece left with <= 1
   character after splitting is never emitted as a term (D7-03).
6. Terms are emitted on channel ``"text"``; Porter stems on channel
   ``"stem"`` (same offsets). The Porter stemmer is a compact stdlib
   port of the tartarus.org reference algorithm — no NLTK.
7. Stopwords are indexed, never dropped (V7-05.11 — their weight is
   IDF-derived downstream).

``NormAnalysis.text`` is the matching-projection debug string: folded
text with identifier spans left verbatim (exact bytes), whitespace
collapsed.

§32.0: this analyzer's rule table is ``provisional/v7-r0``; artifacts
record ``analyzer_id == "norm/v2"`` (V7-32.01).

Deviations / interpretations (documented for the wave report):

- ``n't``: in the written form ("can't", "don't") the apostrophe sits
  between the n and the t, so the clitic is detected as a post-apostrophe
  piece ``t`` whose preceding piece ends in ``n``; the base keeps its n
  ("can't" -> ``can`` + ``not``). The emitted ``not`` term pins the
  orthographic ``n't`` span (n + ' + t).
- Clitic-expanded terms (``not``/``are``/``will``/``have``/``am``) pin
  the clitic's written span (including the apostrophe), so a byte
  round-trip yields e.g. ``'re``, not the expansion.
- Hash: a non-``0x`` hex run must contain BOTH a hex letter and a digit
  (reuses the ``extract/v1`` guard — a pure-digit string is a number,
  not a hash; "facaded" is a word).
- ``quoted code spans`` = backtick code spans only; quoted prose stays
  foldable so its words still index. A specific identifier inside a
  code span wins over the wrapper ("`` `ABC-123` ``" yields the ticket),
  mirroring ``extract/v1`` precedence.
- ``numbers with units`` uses a pinned unit table (SI + common) plus
  ``%`` and ``°C/°F/°K`` forms; the optional single space (or NBSP) is
  part of the identifier span.
- Only ``'`` (U+0027) and ``’`` (U+2019) are clitic boundaries, per spec.
- The <= 1-character guard applies to pieces produced by clitic
  splitting (D7-03); standalone one-character tokens still index —
  stopwords are never dropped (§32.1 rule 7).
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Sequence, Tuple

from verbatim.core.types_v7 import (
    FORMULA_STATUS_PROVISIONAL,
    NormAnalysis,
    NormTerm,
)

ANALYZER_ID = "norm/v2"

#: §32.0 — every §32 rule table is provisional until the formula search.
FORMULA_STATUS = FORMULA_STATUS_PROVISIONAL

__all__ = ["ANALYZER_ID", "FORMULA_STATUS", "analyze", "fold"]


# ---------------------------------------------------------------------------
# Step 1 — strict UTF-8
# ---------------------------------------------------------------------------


def _as_text(text) -> str:
    """Coerce input to ``str`` under strict UTF-8 discipline (§32.1.1).

    ``bytes``/``bytearray``/``memoryview`` are decoded strictly — invalid
    sequences raise ``UnicodeDecodeError``. ``str`` must re-encode
    strictly — unpaired surrogates raise ``UnicodeEncodeError``.
    """
    if isinstance(text, (bytes, bytearray, memoryview)):
        return bytes(text).decode("utf-8", "strict")
    s = str(text if text is not None else "")
    s.encode("utf-8", "strict")  # reject unpaired surrogates
    return s


def _utf8_offsets(text: str) -> List[int]:
    """Char-index -> UTF-8 byte-offset table for ``text``."""
    offsets = [0] * (len(text) + 1)
    pos = 0
    for i, ch in enumerate(text):
        pos += len(ch.encode("utf-8"))
        offsets[i + 1] = pos
    return offsets


# ---------------------------------------------------------------------------
# Step 2 — identifier channel (raw text; patterns reuse the repo's
# extract/v1 shapes from verbatim/enrichment/identifiers.py)
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"(?:https?|ftp|file)://[^\s<>\"'`]+", re.IGNORECASE)
_URL_TRAIL = ".,;:!?)]}>\"'”’"  # trailing prose punctuation wrongly captured

_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b"
)

#: @handle — not glued to a preceding word char / email local part.
_HANDLE_RE = re.compile(r"(?<![\w.+-])@[A-Za-z_][A-Za-z0-9_-]{0,30}\b")

#: #tag — not glued to a word char, ``&`` (HTML entity) or another ``#``.
_TAG_RE = re.compile(r"(?<![\w&#])#[A-Za-z0-9_]+")

#: Unix relative/absolute or Windows drive paths; post-validated by
#: ``_is_path`` so "and/or" and "2025/03/14" are never paths.
_PATH_RE = re.compile(
    r"(?<![\w@/.])"
    r"(?:"
    r"[\w.@+~-]+(?:/[\w.@+~-]+)+/?"
    r"|(?:~|\.{1,2})?(?:/[\w.@+~-]+)+/?"
    r"|[A-Za-z]:\\(?:[^\x00-\x1f<>\"|?*\\/]+\\?)+"
    r")"
)
_PATH_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,10}/?$")

#: Ticket ids per §32.1: ``[A-Z]{2,10}-\d+``.
_TICKET_RE = re.compile(r"\b[A-Z]{2,10}-\d+\b")

#: Hex hashes >= 7; a non-0x run must carry both a hex letter and a digit.
_HASH_RE = re.compile(r"\b(?:0x[0-9a-fA-F]{4,64}|[0-9a-fA-F]{7,64})\b")

#: Semantic versions (extract/v1 shapes): "v1.2.3", "1.2.3-rc.1",
#: "deploy-v2", bare "v2", dotted "3.5".
_VERSION_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9_.]*-[vV]\d+(?:\.\d+)*\b"
    r"|\b[vV]\d+(?:\.\d+){1,3}(?:-[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*)?"
    r"(?:\+[0-9A-Za-z.-]+)?\b"
    r"|\b[vV]\d+\b"
    r"|\b\d+\.\d+(?:\.\d+){0,2}(?:-[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*)?"
    r"(?:\+[0-9A-Za-z.-]+)?\b"
)

#: Quoted code spans — backtick spans only.
_CODE_SPAN_RE = re.compile(r"`[^`\n]{1,500}`")

#: Units for "numbers with units" (§32.1). Pinned, deterministic.
_MEASURE_UNITS = frozenset({
    # mass
    "kg", "g", "mg", "mcg", "ug", "µg", "lb", "lbs", "oz",
    # length
    "km", "m", "cm", "mm", "um", "µm", "nm", "mi", "yd", "ft",
    # volume
    "l", "ml", "cl", "dl", "gal",
    # time
    "ms", "us", "µs", "ns", "min", "hr", "hrs", "sec", "secs", "s",
    # frequency
    "hz", "khz", "mhz", "ghz",
    # data volume / rate
    "kb", "mb", "gb", "tb", "pb", "kib", "mib", "gib", "tib",
    "kbps", "mbps", "gbps",
    # electrical / power / energy
    "v", "mv", "kv", "ma", "amp", "w", "kw", "mw", "gw",
    "wh", "kwh", "j", "kj", "mj", "cal", "kcal",
    # pressure
    "pa", "kpa", "mpa", "bar", "psi", "atm",
    # speed
    "mph", "kph", "kmph", "km/h", "m/s", "fps", "rpm", "kn",
    # display / imaging / misc
    "px", "pt", "em", "rem", "vh", "vw", "dpi", "ppi", "mp", "db",
})
_UNIT_ALT = "|".join(
    re.escape(u) for u in sorted(_MEASURE_UNITS, key=lambda u: (-len(u), u))
)
_MEASURE_RE = re.compile(
    r"\b\d+(?:[.,]\d+)*"
    r"(?:[ \u00a0]?(?:" + _UNIT_ALT + r")(?![\w])"
    r"|[ \u00a0]?%"
    r"|[ \u00a0]?°[cfk](?![\w]))",
    re.IGNORECASE,
)

#: Overlap precedence — lower wins. ``code`` (the quoted wrapper) is
#: last so a specific identifier inside a code span still wins
#: ("`` `ABC-123` ``" yields the ticket), mirroring extract/v1.
_KIND_PRIORITY = {
    "url": 0, "email": 1, "path": 2, "handle": 3, "hashtag": 4,
    "ticket": 5, "hash": 6, "measure": 7, "version": 8, "code": 9,
}


def _is_path(span: str) -> bool:
    """Post-validate a unix-form path candidate (Windows forms always
    pass). Same rules as ``extract/v1``: ``~``/``.``-relative, absolute
    with >= 2 segments, any >= 3-segment path, or a final segment with an
    extension. Rejects all-digit segments (dates) and "and/or"."""
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


def _collect_identifiers(t: str) -> List[Tuple[str, int, int]]:
    """All raw (kind, char_start, char_end) candidates, pre-resolution."""
    cands: List[Tuple[str, int, int]] = []
    for m in _URL_RE.finditer(t):
        s, e = m.span()
        while e > s and t[e - 1] in _URL_TRAIL:
            e -= 1
        # an unbalanced "(" glued to the tail is prose punctuation
        if t[s:e].count("(") > t[s:e].count(")"):
            e = t.rindex("(", s, e)
            while e > s and t[e - 1] in _URL_TRAIL:
                e -= 1
        if e > s + len("http://"):
            cands.append(("url", s, e))
    for m in _EMAIL_RE.finditer(t):
        cands.append(("email", *m.span()))
    for m in _HANDLE_RE.finditer(t):
        cands.append(("handle", *m.span()))
    for m in _TAG_RE.finditer(t):
        cands.append(("hashtag", *m.span()))
    for m in _PATH_RE.finditer(t):
        s, e = m.span()
        span = t[s:e]
        if "\\" in span:
            span = span.rstrip("\\")
            e = s + len(span)
        if span and _is_path(span):
            cands.append(("path", s, e))
    for m in _TICKET_RE.finditer(t):
        cands.append(("ticket", *m.span()))
    for m in _HASH_RE.finditer(t):
        tok = m.group(0)
        if tok.lower().startswith("0x"):
            cands.append(("hash", *m.span()))
        elif any("a" <= c.lower() <= "f" for c in tok) and any(
            c.isdigit() for c in tok
        ):
            # non-0x hex needs BOTH a hex letter and a digit — pure
            # numbers ("1234567") and pure a–f words ("facaded") stay
            # text terms.
            cands.append(("hash", *m.span()))
    for m in _MEASURE_RE.finditer(t):
        cands.append(("measure", *m.span()))
    for m in _VERSION_RE.finditer(t):
        cands.append(("version", *m.span()))
    for m in _CODE_SPAN_RE.finditer(t):
        cands.append(("code", *m.span()))
    return cands


def _resolve_overlaps(
    cands: Sequence[Tuple[str, int, int]],
) -> List[Tuple[str, int, int]]:
    """Deterministic non-overlap resolution: kind priority, then earliest
    start, then longest span; sorted by start at the end."""
    order = sorted(
        set(cands),
        key=lambda c: (_KIND_PRIORITY[c[0]], c[1], -(c[2] - c[1])),
    )
    kept: List[Tuple[str, int, int]] = []
    for _kind, s, e in order:
        if any(s < ke and ks < e for _k, ks, ke in kept):
            continue
        kept.append((_kind, s, e))
    return sorted(kept, key=lambda c: (c[1], c[2]))


# ---------------------------------------------------------------------------
# Step 3 — matching projection: NFKC -> casefold -> diacritic strip
# ---------------------------------------------------------------------------


def _fold_piece(piece: str) -> str:
    """NFKC -> casefold -> strip combining marks (§32.1 step 3)."""
    s = unicodedata.normalize("NFKC", piece).casefold()
    s = unicodedata.normalize("NFD", s)
    return "".join(
        c for c in s if not unicodedata.category(c).startswith("M")
    )


def fold(surface: str) -> str:
    """Matching projection of one surface: NFKC + casefold + diacritic
    strip (the fold applied to term surfaces; identifiers are never
    folded)."""
    return _fold_piece(_as_text(surface))


def _project(t: str, masked: Sequence[bool]):
    """Build the folded projection with per-char provenance.

    Returns ``(proj_terms, proj_disp, src_index)``. ``proj_terms`` is the
    string tokenized for terms: identifier chars are blanked to spaces so
    they can never join or seed a term. ``proj_disp`` is the debug
    projection (identifier spans verbatim). ``src_index[p]`` is the
    source char index that produced projection char ``p``. Both
    projections have identical length.
    """
    terms_chars: List[str] = []
    disp_chars: List[str] = []
    src_index: List[int] = []
    for i, ch in enumerate(t):
        if masked[i]:
            terms_chars.append(" ")
            disp_chars.append(ch)
            src_index.append(i)
            continue
        for fc in _fold_piece(ch):
            terms_chars.append(fc)
            disp_chars.append(fc)
            src_index.append(i)
    return "".join(terms_chars), "".join(disp_chars), src_index


# ---------------------------------------------------------------------------
# Steps 4-6 — word-boundary tokenize, clitic split, text + stem channels
# ---------------------------------------------------------------------------

#: Unicode word tokens; ``'``/``’`` between word chars stay inside the
#: token so the clitic table sees the pieces (a trailing apostrophe —
#: "James'" — is not inside a word and drops out naturally).
_TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*")

_APOSTROPHES = ("'", "’")

#: §32.1 rule 5 — dropped clitics: 's/’s (possessive or "is"), 'd.
_CLITIC_DROP = frozenset({"s", "d"})

#: §32.1 rule 5 — expanded clitics ('t handled separately: it fires as
#: ``n't`` -> ``not`` only when the preceding piece ends in ``n``).
_CLITIC_MAP = {"re": "are", "ll": "will", "ve": "have", "m": "am"}

_WS_RE = re.compile(r"\s+")


def _emit(
    terms: List[NormTerm],
    term: str,
    ps: int,
    pe: int,
    src_index: Sequence[int],
    offsets: Sequence[int],
) -> None:
    """Emit (text, stem) term pair; ``ps``/``pe`` are projection char
    offsets which are mapped back to source byte offsets."""
    cs = src_index[ps]
    ce = src_index[pe - 1] + 1
    bs, be = offsets[cs], offsets[ce]
    terms.append(NormTerm(term, "text", bs, be))
    terms.append(NormTerm(_porter_stem(term), "stem", bs, be))


def _terms_from(
    proj: str,
    src_index: Sequence[int],
    offsets: Sequence[int],
) -> List[NormTerm]:
    terms: List[NormTerm] = []
    for m in _TOKEN_RE.finditer(proj):
        ta, tb = m.span()
        # Split the token at clitic boundaries (apostrophes). Between two
        # pieces sits exactly one apostrophe char.
        pieces: List[Tuple[int, int]] = []
        last = ta
        for i in range(ta, tb):
            if proj[i] in _APOSTROPHES:
                pieces.append((last, i))
                last = i + 1
        pieces.append((last, tb))
        split = len(pieces) > 1
        prev_text = ""
        for pi, (ps, pe) in enumerate(pieces):
            ptext = proj[ps:pe]
            if pi == 0:
                # base piece; a <=1-char piece left by a split is never
                # emitted (D7-03); standalone short tokens index fine.
                if ptext and (not split or len(ptext) > 1):
                    _emit(terms, ptext, ps, pe, src_index, offsets)
            elif ptext in _CLITIC_DROP:
                pass  # 's / ’s and 'd -> drop the clitic
            elif ptext == "t" and prev_text.endswith("n"):
                # n't -> not; the orthographic clitic is n + ' + t.
                _emit(terms, "not", ps - 2, pe, src_index, offsets)
            elif ptext in _CLITIC_MAP:
                _emit(
                    terms, _CLITIC_MAP[ptext], ps - 1, pe,
                    src_index, offsets,
                )
            elif len(ptext) > 1:
                _emit(terms, ptext, ps, pe, src_index, offsets)
            prev_text = ptext
    return terms


# ---------------------------------------------------------------------------
# Step 6 helper — Porter (1980) stemmer, tartarus.org reference semantics
# ---------------------------------------------------------------------------


def _porter_stem(w: str) -> str:
    """Classic Porter stemming algorithm (stdlib-only port).

    Reference semantics: steps 1a/1b/1c/2/3/4/5a/5b with the tartarus.org
    rule tables (incl. ``bli -> ble`` and ``logi -> log``). Words of <= 2
    chars are returned unchanged.
    """
    if len(w) <= 2:
        return w
    b = list(w)
    k = len(b) - 1
    j = 0  # after ends(): b[j+1..k] is the matched suffix

    def cons(i: int) -> bool:
        ch = b[i]
        if ch in "aeiou":
            return False
        if ch == "y":
            return i == 0 or not cons(i - 1)
        return True

    def m() -> int:
        """Measure of b[0..j]: [C](VC)^m[V]."""
        n = 0
        i = 0
        while True:
            if i > j:
                return n
            if not cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > j:
                    return n
                if cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > j:
                    return n
                if not cons(i):
                    break
                i += 1
            i += 1

    def vowel_in_stem() -> bool:
        return any(not cons(i) for i in range(j + 1))

    def doublec(jj: int) -> bool:
        return jj >= 1 and b[jj] == b[jj - 1] and cons(jj)

    def cvc(i: int) -> bool:
        if i < 2 or not cons(i) or cons(i - 1) or not cons(i - 2):
            return False
        return b[i] not in "wxy"

    def ends(s: str) -> bool:
        nonlocal j
        n = len(s)
        if n > k + 1 or "".join(b[k - n + 1: k + 1]) != s:
            return False
        j = k - n
        return True

    def setto(s: str) -> None:
        nonlocal k
        n = len(s)
        b[j + 1: k + 1] = list(s)
        k = j + n

    def r(s: str) -> None:
        if m() > 0:
            setto(s)

    # step 1a — plurals -----------------------------------------------
    if b[k] == "s":
        if ends("sses"):
            k -= 2
        elif ends("ies"):
            setto("i")
        elif b[k - 1] != "s":
            k -= 1

    # step 1b — eed / ed / ing -----------------------------------------
    if ends("eed"):
        if m() > 0:
            k -= 1
    elif (ends("ed") and vowel_in_stem()) or (
        ends("ing") and vowel_in_stem()
    ):
        k = j  # delete the suffix
        if ends("at"):
            setto("ate")
        elif ends("bl"):
            setto("ble")
        elif ends("iz"):
            setto("ize")
        elif doublec(k):
            k -= 1
            if b[k] in "lsz":
                k += 1
        elif m() == 1 and cvc(k):
            setto("e")

    # step 1c — terminal y ----------------------------------------------
    if ends("y") and vowel_in_stem():
        b[k] = "i"

    # step 2 — derivational suffixes (condition m > 0) ------------------
    for suf, rep in _PORTER_STEP2:
        if ends(suf):
            r(rep)
            break

    # step 3 — more derivational suffixes (m > 0) -----------------------
    for suf, rep in _PORTER_STEP3:
        if ends(suf):
            r(rep)
            break

    # step 4 — suffix deletion (m > 1) ----------------------------------
    if (
        ends("al") or ends("ance") or ends("ence") or ends("er")
        or ends("ic") or ends("able") or ends("ible") or ends("ant")
        or ends("ement") or ends("ment") or ends("ent")
        or (ends("ion") and j >= 0 and b[j] in "st")
        or ends("ou") or ends("ism") or ends("ate") or ends("iti")
        or ends("ous") or ends("ive") or ends("ize")
    ):
        if m() > 1:
            k = j

    # step 5 ------------------------------------------------------------
    j = k
    if b[k] == "e":
        a = m()
        if a > 1 or (a == 1 and not cvc(k - 1)):
            k -= 1
    if b[k] == "l" and doublec(k) and m() > 1:
        k -= 1

    return "".join(b[: k + 1])


#: Step-2 table in canonical dispatch order (overlapping suffixes like
#: "ational"/"tional" and "ization"/"ation" resolve by listing order).
_PORTER_STEP2 = (
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
    ("anci", "ance"), ("izer", "ize"), ("bli", "ble"), ("alli", "al"),
    ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
    ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
    ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"),
    ("biliti", "ble"), ("logi", "log"),
)

_PORTER_STEP3 = (
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
)


# ---------------------------------------------------------------------------
# Public analyzer
# ---------------------------------------------------------------------------


def analyze(text: str) -> NormAnalysis:
    """Analyze ``text`` under `norm/v2` (§32.1). Deterministic and pure.

    Returns ``NormAnalysis`` with text/stem ``terms`` (byte offsets into
    the original UTF-8 encoding) and the exact-byte ``identifiers``
    channel. Raises ``UnicodeDecodeError``/``UnicodeEncodeError`` on
    invalid UTF-8 / unpaired surrogates (step 1).
    """
    t = _as_text(text)
    offsets = _utf8_offsets(t)

    resolved = _resolve_overlaps(_collect_identifiers(t))
    masked = [False] * len(t)
    for _kind, cs, ce in resolved:
        for i in range(cs, ce):
            masked[i] = True

    proj_terms, proj_disp, src_index = _project(t, masked)
    terms = _terms_from(proj_terms, src_index, offsets)
    identifiers = tuple(
        NormTerm(t[cs:ce], "identifier", offsets[cs], offsets[ce])
        for _kind, cs, ce in resolved
    )
    return NormAnalysis(
        analyzer_id=ANALYZER_ID,
        terms=tuple(terms),
        identifiers=identifiers,
        text=_WS_RE.sub(" ", proj_disp).strip(),
    )
