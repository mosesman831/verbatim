"""Canonical entity mentions + deterministic alias rules (V7-08.01–06, §32.7).

``entities/v2`` — the V7 entity pass. Three jobs, all pure stdlib and
deterministic:

1. **Canonical keys** (V7-08.01, D7-02): ``canon(surface)`` is the matching
   key — possessive stripped, NFKC, casefolded, diacritics dropped,
   punctuation/separator categories folded to a single space, whitespace
   collapsed. ``"Caroline's"``, ``"CAROLINE"`` and ``"caroline"`` all land
   on the same canon. The exact surface is always kept beside the canon.

2. **Mention extraction** (write side + query side): ``extract_mentions``
   emits ``EntityMention`` rows with byte offsets into the source text —
   multi-word capitalized runs (``"Ruth van der Berg"``) and single
   capitalized tokens ≥ 2 chars; the turn's ``speaker`` becomes an
   additional mention with ``role=speaker``. Canon hygiene (V8-13.03):
   an apostrophe-contraction tail ends a run (``"Can't"`` never mints
   ``"can t"``) and owned vocative tokens never join one (``"Thanks
   Nate"`` → ``"nate"``). ``extract_query_entities``
   (V7-08.02) never requires capitalization: any query token or n-gram of
   ≤ 4 tokens whose canon is in the scope's entity vocabulary matches,
   plus capitalized runs; longest match wins at each position.

3. **Alias proposals** (§32.7 ``alias/v1``, V7-08.03): ``propose_aliases``
   applies rules A1–A6 exactly — token-subset, initial forms, explicit
   statements ("call me Mel", "Melanie (Mel)", …), near-spelling with
   co-mention support, caller-supplied rows — and the A6 conflict rule:
   any alias that resolves to ≥ 2 plausible canonicals becomes a
   reviewable ``candidate``, never auto-applied (same-name-different-
   person abstention, V7-08.13/H101).

Also here: ``expand_query`` (V7-08.04, bounded active-alias expansion) and
``idf_weight``/``entity_weight`` (V7-08.06, scope-level IDF with the
dominant-canon cap — a canon in > 30% of units contributes at most 0.1 of
a rare canon's weight).

**Consumed input contract.** ``NormAnalysis`` carries ``terms`` (folded)
plus ``text``. The entity pass needs surfaces and capitalization, so it
reads the unit/query source text from ``norm.text`` — the same field
``extract_events`` pins into and ``temporal/v2`` resolves against. If a
producer stores the folded projection there instead, capitalized-run
detection degrades honestly to zero text mentions (canons and speaker
mentions still work); no fabricated surfaces are ever emitted.

Constants are ``provisional/v7-r0`` (§32.0); ``ALIAS_RULES_VERSION`` and
``EXTRACTOR_ID`` pin the rule snapshot on every artifact.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Container, Dict, Iterable, List, Optional, Tuple

from ..core.types_v7 import (
    AliasMethod,
    AliasRow,
    AliasState,
    EntityMention,
    MentionRole,
)
from .normalize import utf8_offsets

if TYPE_CHECKING:  # type-only; the analyzer module is owned by w-norm
    from ..core.types_v7 import NormAnalysis

#: Version tag pinning the extraction rule snapshot (provisional §32.0).
#: v2 → v2.1: V8-13.03 canon hygiene — apostrophe-contraction tails end a
#: capitalized run; owned vocative tokens never join one. Derived
#: ``entity_mentions``/``entity_canon`` rows change only under this tag.
EXTRACTOR_ID = "entities/v2.1"
ALIAS_RULES_VERSION = "alias/v1"
FORMULA_STATUS = "provisional/v7-r0"

#: Query-side n-gram bound (V7-08.02).
MAX_QUERY_NGRAM = 4

#: Query expansion bound per mention (V7-08.04).
DEFAULT_EXPANSION_LIMIT = 8

#: Dominant-canon IDF cap (V7-08.06): a canon present in strictly more
#: than this fraction of units contributes at most ``_DOMINANT_CAP`` of a
#: rare (df = 1) canon's weight.
DOMINANT_DF_RATIO = 0.30
DOMINANT_CAP = 0.10


# ---------------------------------------------------------------------------
# canon — V7-08.01 matching projection
# ---------------------------------------------------------------------------

#: Possessive/clitic tail on one token: ``'s`` ``'S`` ``’s`` ``’S`` or a
#: bare trailing apostrophe (``James'`` → ``James``).
_POSSESSIVE_TAIL_RE = re.compile(r"(?:['’][sS]|['’])$")

#: Unicode general-category groups folded to a single space (same
#: matching-projection convention as ``normalize.normalize_text``):
#: punctuation, symbols, control/format, separators. Marks are dropped
#: (diacritic strip); letters and numbers stay.
_FOLD_GROUP = frozenset({"P", "S", "C", "Z"})

_WS_RE = re.compile(r"\s+")


def strip_possessive(surface: str) -> str:
    """Drop the possessive/clitic tail from every whitespace-separated
    token of ``surface`` (``"Caroline's"`` → ``"Caroline"``,
    ``"James'"`` → ``"James"``). Internal apostrophes survive —
    ``"O'Brien"`` keeps its ``'B``. Applied per token so
    ``"Caroline's Mural"`` canonicalizes as ``caroline mural``."""
    s = str(surface if surface is not None else "")
    return " ".join(_POSSESSIVE_TAIL_RE.sub("", tok) for tok in s.split())


def _fold(text: str) -> str:
    """``norm/v2`` fold (V7-05.10, §32.1): NFKC, casefold, diacritic strip,
    punctuation/symbol/separator categories folded to a single space,
    whitespace collapsed. Local copy of the matching projection so this
    module has no import dependency on ``text/norm_v2.py`` (another
    worker's file); the pipeline is the spec-pinned one."""
    s = unicodedata.normalize("NFKC", str(text if text is not None else ""))
    s = s.casefold()
    s = unicodedata.normalize("NFKD", s)
    out: List[str] = []
    for ch in s:
        if unicodedata.combining(ch):
            continue
        if unicodedata.category(ch)[0] in _FOLD_GROUP:
            out.append(" ")
        else:
            out.append(ch)
    return _WS_RE.sub(" ", "".join(out)).strip()


def canon(surface: str) -> str:
    """Canonical entity key (V7-08.01):
    ``fold(NFKC(casefold(strip_possessive(surface))))``.

    The inner NFKC+casefold is subsumed by ``_fold`` (both are
    idempotent), so this is ``_fold(strip_possessive(surface))``.
    Idempotent: ``canon(canon(x)) == canon(x)``.
    """
    return _fold(strip_possessive(surface))


# ---------------------------------------------------------------------------
# mention extraction (write side)
# ---------------------------------------------------------------------------

#: Letter-led tokens (may contain internal ' ’ - . — "O'Brien",
#: "Smith-Jones", "St. John"). Same pattern family as
#: ``identifiers._WORD_RE`` (proven extract/v1, reused not edited).
_WORD_RE = re.compile(r"[^\W_][\w'’.-]*", re.UNICODE)

#: Lowercase connectors allowed strictly inside a capitalized run —
#: "Ruth van der Berg", "United States of America". ``and`` is excluded:
#: "Alice and Bob" is two entities, never one.
_CONNECTORS = frozenset({
    "of", "the", "de", "del", "della", "van", "von", "der", "den", "da",
    "di", "la", "le",
})

#: Sentence-initial function words suppressed as run starters (applied
#: only at a sentence boundary; mid-sentence "The" still joins a run —
#: "The Beatles"). Mirrors identifiers.py's proven list.
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

#: Owned contraction-suffix inventory (V8-13.03): an apostrophe + one of
#: these tails marks a contraction, never an entity. Without a boundary
#: the fold maps the apostrophe to a space and "Can't"/"You're"/"Don't"
#: mint "can t"/"you re"/"don t" canons; keeping the whole token out of
#: a run also keeps the head ("Can", "You", "Don") from minting a
#: pronoun-stem canon.
_CONTRACTION_SUFFIXES = frozenset({"t", "s", "re", "ve", "ll", "d", "m"})

#: Tail regex over the owned list minus ``s``: an 's tail is already
#: ended at canon level by ``strip_possessive`` ("Caroline's" → canon
#: "caroline", surface kept byte-exact), so the run keeps 's tokens and
#: real possessive mentions survive. The remaining tails are
#: unambiguous contractions — the token does not participate and closes
#: any open run (the run ends at the apostrophe boundary).
_CONTRACTION_TAIL_RE = re.compile(
    r"['’](?:%s)$" % "|".join(sorted(_CONTRACTION_SUFFIXES - {"s"})),
    re.IGNORECASE,
)

#: Owned leading vocative/greeting/interjection list (V8-13.03) — these
#: tokens never join a capitalized run: "Thanks Nate" yields canon
#: "nate", never "thanks nate"; "Hey Gina" → "gina". Applied at any run
#: position (a mid-run vocative closes the run rather than gluing two
#: names into one junk canon).
_VOCATIVE = frozenset({
    "thanks", "thank", "hey", "hi", "hello", "congrats",
    "congratulations", "wow", "oh", "yes", "no", "okay", "ok", "dear",
    "good", "well", "sorry", "please",
})

#: A colon ends the speaker prefix in chat transcripts — "Jon: Gina went
#: …" must not fuse into a "jon gina" run (beat-it r6).
_SENT_BOUNDARY = frozenset(".!?\n:")


def _sentence_initial(text: str, start: int) -> bool:
    """True when the token at ``start`` opens a sentence: start of text or
    immediately after . ! ? or a newline."""
    i = start - 1
    while i >= 0 and text[i] in " \t\r":
        i -= 1
    return i < 0 or text[i] in _SENT_BOUNDARY


def _words(text: str) -> List[Tuple[int, int, str]]:
    """(char_start, char_end, token) for each word; trailing ".-" trimmed
    ("Berg." → "Berg", "St." → "St")."""
    out = []
    for m in _WORD_RE.finditer(text):
        tok = m.group(0).rstrip(".-")
        if tok:
            out.append((m.start(), m.start() + len(tok), tok))
    return out


def _cap_spans(text: str) -> List[Tuple[int, int]]:
    """Capitalized-mention char spans in ``text``.

    Maximal runs of capitalized tokens, optionally joined by
    ``_CONNECTORS`` (a connector extends an open run, never starts one,
    and a trailing connector is trimmed back). Single-token runs emit
    only when the token is ≥ 2 chars — "I" and "A" never mention.
    Sentence-initial function words never start a run; a run never
    crosses a sentence boundary. A token ending in an apostrophe-
    contraction tail (V8-13.03a) never participates — "Can't" ends a
    run at its apostrophe boundary rather than minting "can t" — and an
    owned vocative token (V8-13.03b) never joins a run, so "Thanks
    Nate" yields "Nate" alone.
    """
    words = _words(text)
    spans: List[Tuple[int, int]] = []
    run_start: Optional[int] = None
    run_end: Optional[int] = None   # end of the run incl. connectors
    real_end: Optional[int] = None  # end of the last real (non-connector) token
    n_real = 0          # non-connector tokens in the open run
    first_len = 0       # length of the run's first real token
    for ws, we, wtok in words:
        first_upper = wtok[0].isupper()
        is_never = wtok.casefold() in _NEVER
        # V8-13.03(a): apostrophe-contraction boundary — "Can't" is a
        # contraction, not a capitalized name token.
        is_contraction = _CONTRACTION_TAIL_RE.search(wtok) is not None
        # V8-13.03(b): leading vocative/greeting/interjection strip —
        # "Thanks"/"Hey" never glue onto the real name that follows.
        is_vocative = wtok.casefold() in _VOCATIVE
        is_conn = (
            wtok.casefold() in _CONNECTORS
            and not first_upper
            and run_start is not None
        )
        suppressed = _sentence_initial(text, ws) and wtok.casefold() in _SENT_STOP
        boundary_break = (
            run_start is not None
            and any(c in _SENT_BOUNDARY for c in text[run_end:ws])
        )
        entity_token = (
            first_upper and not is_never and not suppressed
            and not is_contraction and not is_vocative
        )
        participates = (entity_token or is_conn) and not boundary_break
        if participates:
            if run_start is None:
                run_start, n_real, first_len = ws, 0, 0
            run_end = we
            if not is_conn:
                n_real += 1
                real_end = we
                if n_real == 1:
                    first_len = we - ws
        else:
            if run_start is not None:
                # trailing connector chain trims back to the last real
                # token — "Alice van" emits "Alice", surface-exact.
                if real_end > run_start and (n_real > 1 or first_len >= 2):
                    spans.append((run_start, real_end))
                run_start = run_end = real_end = None
                n_real = first_len = 0
            if entity_token and not is_conn:
                run_start, run_end, real_end = ws, we, we
                n_real, first_len = 1, we - ws
    if run_start is not None:
        if real_end > run_start and (n_real > 1 or first_len >= 2):
            spans.append((run_start, real_end))
    return spans


def extract_mentions(
    norm: "NormAnalysis",
    unit_id: str,
    speaker: Optional[str] = None,
) -> List[EntityMention]:
    """Entity mentions in one unit (V7-08.01).

    Scans ``norm.text`` — the analyzed source text — for multi-word
    capitalized runs and single capitalized tokens (≥ 2 chars). Each hit
    yields an ``EntityMention`` with the exact surface, its ``canon``,
    and UTF-8 byte offsets into the source text. ``speaker`` (surface or
    canon) adds a ``role=speaker`` mention; it is metadata, not a text
    span, so its byte offsets are ``(0, 0)`` — distinguishable from every
    real span and never fabricated. Text-extracted mentions carry the
    default ``role=mention``; subject/object attribution belongs to the
    event pass, not the tokenizer.
    """
    text = getattr(norm, "text", "") or ""
    unit = str(unit_id)
    out: List[EntityMention] = []
    if text:
        offsets = utf8_offsets(text)
        for cs, ce in _cap_spans(text):
            surface = text[cs:ce]
            c = canon(surface)
            if not c:
                continue
            out.append(EntityMention(
                canon=c,
                surface=surface,
                unit_id=unit,
                byte_start=offsets[cs],
                byte_end=offsets[ce],
                role=MentionRole.MENTION,
            ))
    if speaker:
        sc = canon(speaker)
        if sc:
            out.append(EntityMention(
                canon=sc,
                surface=str(speaker),
                unit_id=unit,
                byte_start=0,
                byte_end=0,
                role=MentionRole.SPEAKER,
            ))
    return out


# ---------------------------------------------------------------------------
# query-side extraction (V7-08.02)
# ---------------------------------------------------------------------------


def _has_boundary(text: str, end_a: int, start_b: int) -> bool:
    """True when a sentence boundary lies between two word spans."""
    return any(c in _SENT_BOUNDARY for c in text[end_a:start_b])


def extract_query_entities(
    norm: "NormAnalysis",
    known_canons: Container[str],
) -> List[str]:
    """Query-side entity canons (V7-08.02) — no capitalization required.

    Two signals over the query source text (``norm.text``):

    - any token or n-gram of ≤ ``MAX_QUERY_NGRAM`` tokens whose canon is
      in ``known_canons`` (the scope's entity vocabulary), longest match
      first at each position — so "alice chen" beats "alice" + "chen",
      and n-grams never cross a sentence boundary;
    - capitalized runs (same rules as the write side) so a capitalized
      out-of-vocabulary name still yields an entity signal.

    Returns distinct canons in first-seen order.
    """
    text = getattr(norm, "text", "") or ""
    known = {c for c in (canon(k) for k in (known_canons or ())) if c}
    out: List[str] = []
    if not text:
        return out
    words = _words(text)
    i = 0
    while i < len(words):
        matched = 0
        for n in range(min(MAX_QUERY_NGRAM, len(words) - i), 0, -1):
            if n > 1 and any(
                _has_boundary(text, words[i + k - 1][1], words[i + k][0])
                for k in range(1, n)
            ):
                continue
            gram = " ".join(w[2] for w in words[i:i + n])
            c = canon(gram)
            if c and c in known:
                out.append(c)
                matched = n
                break
        i += matched if matched else 1
    for cs, ce in _cap_spans(text):
        c = canon(text[cs:ce])
        if c:
            out.append(c)
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# alias rules A1–A6 (§32.7, alias/v1)
# ---------------------------------------------------------------------------

#: Rule precedence when two rules propose the same (canon, alias) pair:
#: the more authoritative rule keeps the row (evidence counts merge by
#: max). A6 is last — it only ever mediates conflicts.
_PRECEDENCE = {"A5": 0, "A3": 1, "A1": 2, "A2": 3, "A4": 4, "A6": 5}

#: A3 explicit-statement patterns. Each yields ``(canonical_surface,
#: alias_surface)``; a ``None`` canonical means "the unit's speaker" and
#: resolves only when exactly one speaker canon is present.
_PAREN_RE = re.compile(
    r"([A-Z][\w.'’-]*(?:\s+[A-Z][\w.'’-]*){0,2})\s*"
    r"\(\s*([A-Z][\w.'’-]*)\s*\)"
)
_CALL_ME_RE = re.compile(
    r"\bcall\s+me\s+([A-Za-z][\w.'’-]*(?:\s+[A-Z][\w.'’-]*){0,2})",
    re.IGNORECASE,
)
_MY_NAME_RE = re.compile(
    r"\bmy\s+name\s+is\s+([A-Za-z][\w.'’-]*(?:\s+[A-Z][\w.'’-]*){0,2})",
    re.IGNORECASE,
)
_GOES_BY_RE = re.compile(
    r"\b(?:([A-Z][\w.'’-]*(?:\s+[A-Z][\w.'’-]*){0,2})\s+)?"
    r"goes\s+by\s+([A-Za-z][\w.'’-]*)",
    re.IGNORECASE,
)
_FOR_SHORT_RE = re.compile(
    # the alias token is capitalized — "abbreviated for short" idioms
    # with a lowercase X are not name statements
    r"\b([A-Z][\w.'’-]*)\s+for\s+short\b",
)


def _trim_surface(s: str) -> str:
    return s.strip().rstrip(".-").strip()


def _edit_le1(a: str, b: str) -> bool:
    """True when Levenshtein distance between ``a`` and ``b`` is ≤ 1
    (substitution, insertion, or deletion). Short-circuit, O(len)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = 0
        for x, y in zip(a, b):
            if x != y:
                diff += 1
                if diff > 1:
                    return False
        return True
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = j = 0
    edits = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            edits += 1
            j += 1
            if edits > 1:
                return False
    return True  # any leftover tail char in ``b`` is the single edit


def _nearest_cap_before(text: str, pos: int) -> Optional[str]:
    """The last capitalized token ending strictly before ``pos`` —
    antecedent for "X for short" (``"Melanie, Mel for short"``)."""
    found: Optional[str] = None
    for cs, ce, tok in _words(text):
        if ce > pos:
            break
        if tok[0].isupper() and tok.casefold() not in _NEVER:
            found = tok
    return found


def _explicit_pairs(
    text: str,
    speaker_canons: Tuple[str, ...],
) -> List[Tuple[str, str]]:
    """A3 explicit alias statements → (canonical_canon, alias_canon).

    Both names pinned by the pattern ("Melanie (Mel)", "Melanie goes by
    Mel") resolve directly; speaker-relative patterns ("call me Mel",
    "my name is Mel", bare "goes by Mel", "Mel for short" with no
    antecedent) resolve only when the mentions carry exactly one
    distinct speaker canon — otherwise the pair is skipped, never
    guessed.
    """
    pairs: List[Tuple[str, str]] = []
    speaker = speaker_canons[0] if len(speaker_canons) == 1 else None

    for m in _PAREN_RE.finditer(text):
        cn = canon(_trim_surface(m.group(1)))
        al = canon(_trim_surface(m.group(2)))
        if cn and al:
            pairs.append((cn, al))
    for m in _CALL_ME_RE.finditer(text):
        al = canon(_trim_surface(m.group(1)))
        if speaker and al and al not in _NEVER:
            pairs.append((speaker, al))
    for m in _MY_NAME_RE.finditer(text):
        al = canon(_trim_surface(m.group(1)))
        if speaker and al and al not in _NEVER:
            pairs.append((speaker, al))
    for m in _GOES_BY_RE.finditer(text):
        y, x = m.group(1), m.group(2)
        cn = canon(_trim_surface(y)) if y else None
        if not cn or cn in _NEVER:
            # "I goes by Mel" / "she goes by Mel" — the subject is the
            # speaker, not a name
            cn = speaker
        al = canon(_trim_surface(x))
        if cn and al and al not in _NEVER:
            pairs.append((cn, al))
    for m in _FOR_SHORT_RE.finditer(text):
        al = canon(_trim_surface(m.group(1)))
        antecedent = _nearest_cap_before(text, m.start(1))
        cn = canon(antecedent) if antecedent else speaker
        if cn and al and al not in _NEVER:
            pairs.append((cn, al))
    return [
        (cn, al) for cn, al in pairs
        if cn and al and cn != al and cn not in _NEVER
    ]


def propose_aliases(
    mentions: Iterable[EntityMention],
    known_canons: Iterable[str],
    *,
    now_generation: int = 0,
    texts: Optional[Iterable[str]] = None,
    scope_id: str = "",
    caller_aliases: Optional[Iterable[Tuple[str, str]]] = None,
) -> List[AliasRow]:
    """Propose ``entity_aliases`` rows under §32.7 ``alias/v1``.

    - **A1 token-subset**: alias tokens ⊂ canonical tokens and exactly
      one canonical in scope contains them → ``active``. Canonicals are
      ``known_canons`` ∪ multi-token mention canons (a multi-token
      mention is itself a canonical candidate in scope).
    - **A2 initial forms**: ``first last-initial`` ("alice c.") or
      ``f. last`` ("a. chen") matching exactly one canonical → ``active``.
    - **A3 explicit statements**: "call me X", "my name is X",
      "Y goes by X" / "goes by X", "Y (X)", "X for short" over ``texts``
      (raw unit texts — optional; absent texts simply yield no A3 rows).
      Speaker-relative patterns need exactly one speaker canon.
    - **A4 near-spelling**: edit distance ≤ 1, both names ≥ 6 chars,
      ≥ 2 shared co-mentioned canons → ``candidate`` (never auto-active).
    - **A5 caller-supplied**: ``caller_aliases`` pairs → ``active``,
      ``method=caller``.
    - **A6 conflict**: whenever any rule yields ≥ 2 plausible canonicals
      for the same alias, every such row is rewritten ``rule_id="A6"``,
      ``state=candidate`` — same-name-different-person pairs are
      reviewable, never auto-applied (V7-08.03/13, H101).

    Rows dedup on ``(canon, alias_canon)`` keeping the highest-precedence
    rule; output is sorted — identical inputs give identical rows.
    ``scope_id`` is a parameter because the frozen signature has none;
    the persisting layer stamps the real scope at insert.
    """
    mentions = [m for m in (mentions or ()) if getattr(m, "canon", "")]
    known = {c for c in (canon(k) for k in (known_canons or ())) if c}
    mention_counts: Counter = Counter(m.canon for m in mentions)
    mention_canons = set(mention_counts)

    # Canonical universe: the scope vocabulary plus multi-token mention
    # canons (single-token mentions are never alias *targets* — a proper
    # token subset cannot exist).
    canonicals = known | {c for c in mention_canons if " " in c}
    canon_tokens: Dict[str, frozenset] = {
        c: frozenset(c.split()) for c in canonicals
    }

    # Co-mention sets: canon -> other canons seen in the same unit.
    by_unit: Dict[str, set] = defaultdict(set)
    for m in mentions:
        by_unit[m.unit_id].add(m.canon)
    comention: Dict[str, set] = defaultdict(set)
    for uset in by_unit.values():
        for c in uset:
            comention[c] |= uset - {c}

    speaker_canons = tuple(sorted({
        m.canon for m in mentions if m.role == MentionRole.SPEAKER
    }))

    # (canon, alias_canon) -> [rule_id, state, method, evidence_count]
    proposals: Dict[Tuple[str, str], list] = {}

    def _emit(cn: str, al: str, rule: str, state: AliasState,
              method: AliasMethod, evidence: int) -> None:
        if not cn or not al or cn == al:
            return
        key = (cn, al)
        old = proposals.get(key)
        if old is None:
            proposals[key] = [rule, state, method, evidence]
            return
        if _PRECEDENCE[rule] < _PRECEDENCE[old[0]]:
            old[0], old[1], old[2] = rule, state, method
        old[3] = max(old[3], evidence)

    # -- A1: token-subset ------------------------------------------------
    for mc in sorted(mention_canons):
        mt = frozenset(mc.split())
        if not mt:
            continue
        supersets = [k for k in sorted(canonicals) if mt < canon_tokens[k]]
        for k in supersets:
            _emit(k, mc, "A1", AliasState.ACTIVE, AliasMethod.RULE,
                  mention_counts[mc])

    # -- A2: initial forms ------------------------------------------------
    for mc in sorted(mention_canons):
        toks = mc.split()
        if len(toks) != 2:
            continue
        t1, t2 = toks
        matches: List[str] = []
        if len(t2) == 1 and len(t1) > 1:
            # "alice c." → first name + last initial
            matches = [
                k for k in sorted(canonicals)
                if len(k.split()) >= 2
                and k.split()[0] == t1 and k.split()[-1].startswith(t2)
            ]
        elif len(t1) == 1 and len(t2) > 1:
            # "a. chen" → first initial + last name
            matches = [
                k for k in sorted(canonicals)
                if len(k.split()) >= 2
                and k.split()[0].startswith(t1) and k.split()[-1] == t2
            ]
        for k in matches:
            _emit(k, mc, "A2", AliasState.ACTIVE, AliasMethod.RULE,
                  mention_counts[mc])

    # -- A3: explicit statements ------------------------------------------
    for text in texts or ():
        for cn, al in _explicit_pairs(str(text), speaker_canons):
            _emit(cn, al, "A3", AliasState.ACTIVE, AliasMethod.RULE, 1)

    # -- A4: near-spelling -------------------------------------------------
    pool = sorted(known | mention_canons)
    for i, a in enumerate(pool):
        if len(a) < 6:
            continue
        for b in pool[i + 1:]:
            if len(b) < 6 or not _edit_le1(a, b):
                continue
            if len(comention.get(a, set()) & comention.get(b, set())) < 2:
                continue
            # direction: an established canon stays the canonical;
            # otherwise more-observed, then lexicographic — deterministic.
            ka, kb = a in known, b in known
            if ka != kb:
                cn, al = (a, b) if ka else (b, a)
            elif mention_counts[a] != mention_counts[b]:
                cn, al = (a, b) if mention_counts[a] > mention_counts[b] \
                    else (b, a)
            else:
                cn, al = a, b  # a < b lexicographically
            _emit(cn, al, "A4", AliasState.CANDIDATE, AliasMethod.RULE,
                  max(mention_counts[a], mention_counts[b]))

    # -- A5: caller-supplied ------------------------------------------------
    for pair in caller_aliases or ():
        cn, al = canon(pair[0]), canon(pair[1])
        _emit(cn, al, "A5", AliasState.ACTIVE, AliasMethod.CALLER, 1)

    # -- A6: conflict → candidate, never auto-applied -----------------------
    by_alias: Dict[str, set] = defaultdict(set)
    for cn, al in proposals:
        by_alias[al].add(cn)
    for (cn, al), rec in proposals.items():
        if len(by_alias[al]) > 1:
            rec[0] = "A6"
            rec[1] = AliasState.CANDIDATE

    return [
        AliasRow(
            scope_id=str(scope_id),
            canon=cn,
            alias_canon=al,
            rule_id=rec[0],
            evidence_count=rec[3],
            method=rec[2],
            state=rec[1],
            generation=int(now_generation),
        )
        for (cn, al), rec in sorted(proposals.items())
    ]


# ---------------------------------------------------------------------------
# query expansion (V7-08.04)
# ---------------------------------------------------------------------------


def expand_query(
    canons: Iterable[str],
    aliases: Iterable[AliasRow],
    limit: int = DEFAULT_EXPANSION_LIMIT,
) -> List[str]:
    """Bounded alias expansion of query canons (V7-08.04).

    ``active`` aliases expand in both directions — a query naming the
    alias reaches the canonical's postings and vice versa (merges are
    links, never rewrites). ``candidate``/``rejected`` rows never expand:
    unreviewed merges cannot widen a query. At most ``limit`` expansions
    per input canon; output preserves first-seen order, deduplicated.
    """
    fwd: Dict[str, set] = defaultdict(set)   # alias_canon -> canon
    rev: Dict[str, set] = defaultdict(set)   # canon -> alias_canon
    for a in aliases or ():
        if a.state != AliasState.ACTIVE:
            continue
        cn, al = canon(a.canon), canon(a.alias_canon)
        if not cn or not al or cn == al:
            continue
        fwd[al].add(cn)
        rev[cn].add(al)
    out: List[str] = []
    cap = max(0, int(limit))
    for c in canons or ():
        cc = canon(c)
        if not cc:
            continue
        out.append(cc)
        exp = sorted((fwd.get(cc, set()) | rev.get(cc, set())) - {cc})
        out.extend(exp[:cap])
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# entity IDF (V7-08.06)
# ---------------------------------------------------------------------------


def idf_weight(canon_df: int, n_units: int) -> float:
    """Scope-level entity IDF — the §32.2 BM25 idf applied to unit df:

    ``ln(1 + (N − df + 0.5) / (df + 0.5))``, floored at 0. ``df`` above
    ``N`` (bad input) yields 0 rather than a negative weight.
    """
    n = float(max(0, int(n_units)))
    df = float(max(0, int(canon_df)))
    if df > n:
        return 0.0
    return max(0.0, math.log(1.0 + (n - df + 0.5) / (df + 0.5)))


def entity_weight(
    canon_str: str,
    df_map: Dict[str, int],
    n_units: int,
) -> float:
    """Ranking weight of one canon (V7-08.06, D7-07).

    ``df_map`` is keyed by canon (a surface key is normalized before
    lookup). A canon present in strictly more than ``DOMINANT_DF_RATIO``
    of units contributes at most ``DOMINANT_CAP`` × the weight of a rare
    (df = 1) canon — a frequent speaker name can never dominate ranking
    the way a flat entity-overlap bonus did.
    """
    n = max(0, int(n_units))
    df = int(df_map.get(canon(canon_str), 0))
    base = idf_weight(df, n)
    if n > 0 and df / n > DOMINANT_DF_RATIO:
        return min(base, DOMINANT_CAP * idf_weight(1, n))
    return base


__all__ = [
    "ALIAS_RULES_VERSION",
    "DEFAULT_EXPANSION_LIMIT",
    "DOMINANT_CAP",
    "DOMINANT_DF_RATIO",
    "EXTRACTOR_ID",
    "FORMULA_STATUS",
    "MAX_QUERY_NGRAM",
    "canon",
    "entity_weight",
    "expand_query",
    "extract_mentions",
    "extract_query_entities",
    "idf_weight",
    "propose_aliases",
    "strip_possessive",
]
