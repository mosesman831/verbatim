"""Standing-rule detection — ``rules_detect/v1`` (SPEC_V7 §32.13, V7-20.02,
V7-20.05).

Implicit-recall rules: normative / imperative sentences ("always X",
"never X", "only until …", "don't … unless", channel and file-path
conventions) detected at T0 over one unit's ``norm/v2`` analysis, pinned
to the unit's raw bytes, scoped by entity/topic, and rendered for agents
through :func:`prefetch_block` as a compact ``## STANDING RULES`` block.

Pipeline (per unit, pure — no store access):

1. ``norm.text`` (the folded matching projection — punctuation and
   identifier surfaces preserved, §32.1) is split into sentences on
   terminal punctuation followed by whitespace or end-of-text, so
   ``v1.2`` and ``tests/`` never split mid-token.
2. ``RULE_PATTERNS_V1`` — an ordered tuple of ``(rule_id, regex,
   extractor)`` — scans each sentence, most specific first; a match
   whose span overlaps an already-accepted rule is skipped.
3. Precision-first guards (§32.13's excluded-forms rule, carried from
   ``pref/v1``): a ``?`` terminator, a hedge/modal/quotative marker
   before the trigger keyword, an interrogative-auxiliary lead, and
   explicit revocation vocabulary ("never mind", "no longer",
   "… anymore") all veto detection.
4. Each accepted rule is byte-pinned. With ``raw_text`` the module
   rebuilds the §32.1 projection locally and maps the match through
   per-char provenance onto exact UTF-8 source bytes
   (``pinned=raw_verified``); without it, a tolerant anchor walk locates
   each ``NormTerm`` inside ``norm.text`` and the pin spans the matched
   terms' source byte offsets (``pinned=term_offsets``). Rules that
   cannot be pinned are dropped and counted — never emitted unpinned.
5. Conditional clauses are captured, not resolved: ``until/before/after
   X`` becomes ``valid_until_expr`` (the verbatim expression is the
   contract — never a date); ``unless/if/once/when X`` is recorded under
   ``signals.condition``. At detection the expr is offered to
   ``temporal_v2.resolve`` against ``anchor_us`` (default ``now_us``);
   only a cleanly-resolved bound lands in ``until_end_us`` — anything
   else keeps the rule active with ``signals.until_unresolved``.
6. ``trigger_entities`` (identifier surfaces — ``#channel``s, paths,
   handles — plus capitalized mentions and extractor-declared
   destinations, all ``entities_v2.canon``-folded) and
   ``trigger_topics`` (the span's content terms) scope the rule. A rule
   with neither is *global* and matches every task.

Lifecycle (``standing_rules.status`` ∈ active/expired/revoked):

- :func:`detect_revocations` spots explicit revocation phrases and
  returns the revoked target text; :func:`apply_revocations` flips
  same-scope active rules to ``revoked`` only at ≥ 0.7 content-term
  overlap — a vague "never mind" never nukes unrelated rules.
- :func:`sweep_expired` flips ``active`` → ``expired`` once ``now_us``
  passes a resolved ``until_end_us``; unresolved exprs never expire.
- ``signals.agent_stated``: §32.13 restricts governing rules to
  ``user_stated``/``document`` perspectives; rules detected in
  agent-stated units are kept but labeled (``perspective=`` argument)
  and ranked after user rules in prefetch — they never override.

Write-path integration (unit → rows) belongs to the caller
(``source_jobs``); this module ships pure detectors plus small
conn-taking helpers (:func:`insert_rules`, :func:`apply_revocations`,
:func:`sweep_expired`) and the read-side :func:`prefetch_block`.

Deterministic, pure stdlib. ``rules_detect/v1`` is
``provisional/v7-r0`` per V7-32.01; the tag rides on
``FORMULA_STATUS``.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from verbatim.core.types import json_dumps, safe_json_loads
from verbatim.core.types_v7 import NormAnalysis, NormTerm
from verbatim.enrichment.normalize import utf8_offsets

EXTRACTOR_ID = "rules_detect/v1"
FORMULA_STATUS = "provisional/v7-r0"

#: Cumulative honest counters (the ``event/v1`` convention). Callers may
#: also pass a per-call ``stats`` dict to ``detect_rules``.
COUNTERS: dict[str, int] = {
    "units": 0,
    "sentences": 0,
    "candidates": 0,
    "emitted": 0,
    "dropped_hedged": 0,
    "dropped_empty_body": 0,
    "dropped_unpinned": 0,
    "revocations": 0,
}

#: ``standing_rules`` column order (for ``insert_rules`` / tests).
RULE_COLUMNS: tuple[str, ...] = (
    "rule_id",
    "scope_id",
    "unit_id",
    "text_pin",
    "trigger_entities",
    "trigger_topics",
    "valid_until_expr",
    "status",
    "generation",
)

RULE_STATUSES: tuple[str, ...] = ("active", "expired", "revoked")

#: Conservative content-term containment required for a revocation
#: phrase to retire an existing rule (spec: ≥ 0.7 text overlap).
REVOCATION_MIN_OVERLAP = 0.7

#: Validity-bound connectives whose clause becomes ``valid_until_expr``.
_UNTIL_HEADS = frozenset({"until", "before", "after"})

#: Conditional connectives recorded under ``signals.condition`` — they
#: qualify the rule but are never a temporal expiry.
_CONDITION_HEADS = frozenset({"unless", "if", "once", "when", "whenever"})

_PERSPECTIVES_AGENT = frozenset({
    "agent_stated", "agent_action", "tool_output", "assistant",
})

_MAX_TOPICS = 8


def reset_counters() -> None:
    for k in COUNTERS:
        COUNTERS[k] = 0


# ---------------------------------------------------------------------------
# Guard vocabularies
# ---------------------------------------------------------------------------

#: Function words + the trigger keywords themselves — excluded from
#: ``trigger_topics`` so a rule is never "triggered" by its own marker.
_STOPWORDS = frozenset(
    "the a an and or but nor of to in on at for with by from as is are "
    "was were be been being do does did done doing don dont have has had "
    "having will would can could shall should may might must ought ll "
    "ve re i you he she it we they me him her us them my your his their "
    "our its mine yours hers ours theirs this that these those there "
    "here not no just also very really quite so too than then when "
    "while if unless until before after once whenever wherever where "
    "who whom whose which what how why all any both each few more most "
    "other others some such own same s t am aren isn wasn weren won "
    "wont let lets about above across against along among around below "
    "beneath beside between beyond down during inside into near off "
    "onto out outside over past per since through throughout toward "
    "towards under underneath up upon via within without again back "
    "further furthermore hence however instead meanwhile moreover "
    "nevertheless nonetheless otherwise still thereby therefore thus "
    "yet already always never only even ever soon now today yesterday "
    "tomorrow tonight ago later every anything something nothing "
    "everything anyone someone everyone nobody somebody everybody "
    "please thanks thank ok okay yeah yes nope kind sort thing things "
    "stuff way ways lot lots bit sure make makes made remember forget "
    "forgot forgotten rule rules policy policies convention conventions "
    "standard standards guideline guidelines practice practices protocol "
    "protocols norm norms prefer prefers preferred preferring suppose "
    "supposed go goes going gone went live lives lived belong belongs "
    "belonged put puts putting keep keeps kept store stores stored add "
    "adds added write writes writing wrote written create creates "
    "created save saves saved move moves moved file files filed send "
    "sends sent post posts posted share shares shared use uses used "
    "using need needs needed want wants wanted get gets got getting "
    "take takes took taken give gives given say says said tell tells "
    "told ask asks asked try tries tried seem seems seemed look looks "
    "looked feel feels felt think thought believe believes believed "
    "know knows knew known see sees saw seen mean means meant like "
    "likes liked love loves loved hate hates hated good bad new old "
    "right wrong big small long short high low early late much many "
    "little enough several whole half part side end beginning middle "
    "number one two three first second third last next previous "
    "following current recent final default".split()
)

#: Animate subjects that license ``always``/``never`` as normatives.
#: "we always deploy" is a rule; "the build is always green" is a
#: description. Sentence-initial keywords need no subject.
_ANIMATE_SUBJ = frozenset({
    "i", "we", "you", "they", "people", "everyone", "everybody",
    "someone", "anyone", "team", "teams", "devs", "developers",
    "engineers", "agents", "agent", "reviewers", "contributors",
    "maintainers", "ops", "eng", "engineering", "folks", "yall", "u",
    "one", "members", "admins", "users", "qa",
})

#: Hedge / modal markers before the trigger keyword veto the match
#: ("maybe never deploy", "i think we always…"). Precision-first.
_HEDGE_WORDS = frozenset({
    "maybe", "perhaps", "possibly", "probably", "likely", "might",
    "could", "would", "should", "may", "hopefully", "supposedly",
    "apparently", "tempted", "unsure", "uncertain", "unclear",
    "wonder", "wondering", "wondered", "wonders", "guess", "guessing",
    "thinking", "think", "thought", "believe", "believed", "suppose",
    "doubt", "doubts", "considering", "contemplating", "leaning",
    "inclined", "iffy", "hesitant", "seems", "seemed", "seem",
    "appears", "appeared",
})
_HEDGE_RE = re.compile(
    r"\b(?:" + "|".join(sorted(_HEDGE_WORDS)) + r"|used\s+to|tends?\s+to"
    r"|planning\s+to)\b",
    re.IGNORECASE,
)

#: Quotative / reported-speech markers before the keyword veto ("she
#: said never deploy on fridays" reports her rule — it is not ours).
_QUOTATIVE_RE = re.compile(
    r"\b(?:said|says|say|told|tells|tell|asked|asks|ask|wrote|writes"
    r"|written|quoted|quoting|quotes|claimed|claims|claim|suggested"
    r"|suggests|suggest|recommended|recommends|recommend|according"
    r"|joked|jokes|joking|kidding|insisted|insists|argued|argues"
    r"|mentioned|mentions|noted|reads|heard|reported)\b",
    re.IGNORECASE,
)

#: Sentence-initial words marking an interrogative or hypothetical frame
#: even when the "?" was dropped ("should we always…", "if we always…").
_LEAD_VETO = frozenset({
    "do", "does", "did", "is", "are", "was", "were", "can", "could",
    "shall", "should", "will", "would", "may", "might", "have", "has",
    "what", "whats", "why", "how", "who", "whom", "whose", "which",
    "whether", "suppose", "imagine", "assuming", "given", "say",
    "wonder", "wondering", "if",
})

#: Revocation vocabulary vetoes rule detection for the whole sentence —
#: "we no longer post to #eng" mints a revocation (see
#: ``detect_revocations``), never a fresh "post to #eng" rule.
_VETO_RE = re.compile(
    r"\b(?:never\s+mind|no\s+longer|anymore|any\s+more|scratch\s+that"
    r"|forget\s+(?:that|it|about|what)|disregard|ignore\s+that"
    r"|rule\s+(?:is\s+)?(?:gone|dead|over|done|cancelled|canceled"
    r"|rescinded|retired|obsolete)|no\s+longer\s+applies"
    r"|stopped\s+(?:doing|using|posting|sending|deploying)"
    r"|quit\s+(?:doing|using)|dropped\s+(?:that|the)\s+"
    r"(?:rule|policy|convention))\b",
    re.IGNORECASE,
)

#: Clause scanner for validity bounds / conditions: a clause ends at a
#: comma, semicolon, or the sentence end ("until X, do Y" → "until X").
_COND_RE = re.compile(
    r"\b(until|before|after|unless|if|once|when|whenever)\s+(.+?)"
    r"(?:\s*[,;]|$)",
    re.IGNORECASE,
)

#: Sentence boundaries: terminal punctuation followed by whitespace or
#: end-of-text (never "3.9"), or a newline run.
_BOUNDARY_RE = re.compile(r"[.!?…]+(?=\s|$)|\n+")

#: Path-ish surfaces (file-path conventions, V7-20.05): at least one
#: interior "/", or a "~"/"$VAR" lead.
_PATH_RE = re.compile(
    r"^(?:[\w*$.-]+/)+[\w*$.-]*$|^~[\w/.-]+$|^\$[\w.]+")

#: Base verbs ending in "ed" — the never-past guard must not reject
#: "never embed secrets".
_ED_BASE_VERBS = frozenset({
    "embed", "embeds", "need", "needs", "feed", "feeds", "seed",
    "seeds", "speed", "speeds", "breed", "breeds", "shed", "sheds",
    "shred", "shreds", "wed", "heed", "deed", "blend", "blends",
    "amend", "amends", "mend", "mends", "trend", "trends", "depend",
    "depends", "append", "appends", "attend", "attends", "extend",
    "extends", "intend", "intends", "pretend", "pretends", "contend",
    "suspend", "suspends", "descend", "descends", "ascend", "ascends",
    "commend", "commends", "recommend", "recommends", "transcend",
    "defend", "defends", "offend", "offends", "end", "ends", "friend",
    "upend", "vend", "expend", "impend", "misfeed", "redo",
})

#: Irregular past-tense forms — "she never called" recounts history;
#: only base-form objects make ``never`` a rule. Ambiguous bases
#: ("read", "shed", "left") stay out — they err toward recall.
_IRREG_PAST = frozenset({
    "went", "saw", "said", "told", "came", "took", "made", "did",
    "was", "were", "had", "found", "met", "ran", "ate", "slept",
    "spoke", "wrote", "bought", "brought", "thought", "felt", "heard",
    "knew", "grew", "threw", "flew", "forgot", "began", "chose",
    "broke", "stole", "woke", "wore", "tore", "swore", "rode", "drove",
    "fell", "stood", "won", "lost", "sold", "held", "paid", "spent",
    "built", "kept", "shot", "swam", "sang", "rang", "sank", "shrank",
    "sprang", "stung", "swung", "lent", "meant", "got", "sat", "dug",
    "stuck", "struck", "snuck", "blew", "drew", "forsook", "mistook",
    "partook", "shook", "underwent", "withdrew", "forbade", "gave",
})

#: Broader past-tense set for the ``cond/time_tail`` head guard: unlike
#: the ``never``-body check (which errs toward recall on ambiguous
#: bases), a head-final "left"/"read"/"led" in "he left before friday"
#: is almost always narrative.
_PAST_FORMS = _IRREG_PAST | frozenset({
    "left", "read", "led", "fed", "fled", "sped", "bled", "bred",
    "slid", "hid", "bid", "beat", "quit", "hit", "split", "shut",
    "cut", "put", "set", "bet", "hurt", "burst", "cost", "cast",
    "hurt", "knelt", "leapt", "lit", "dreamt", "dealt", "dwelt",
    "knit", "wet", "offset", "upset", "sweat",
})

#: Copula/stative verbs inside a ``head`` ("X only until Y", "X unless
#: Y") mark description, not rule: "the report is only useful when
#: fresh" — but imperative heads ("post notes only until green") carry
#: none.
_HEAD_STATIVE = frozenset({
    "is", "are", "was", "were", "been", "being", "seem", "seems",
    "seemed", "look", "looks", "looked", "feel", "feels", "felt",
    "sound", "sounds", "work", "works", "worked", "happen", "happens",
    "happened", "exist", "exists", "existed", "matter", "matters",
    "help", "helps", "helped", "cost", "costs", "remain", "remains",
    "tend", "tends", "become", "becomes", "became", "get", "gets",
    "got", "stay", "stays", "stayed", "keep", "keeps", "kept",
    "smell", "smells", "taste", "tastes", "reads", "read",
})

#: Follows of ``always``/``never`` that describe states ("always late")
#: rather than mandate actions.
_ALWAYS_BLOCK = frozenset({
    "the", "a", "an", "my", "your", "our", "his", "her", "their", "its",
    "on", "in", "at", "of", "for", "with", "to", "from", "there", "here",
    "late", "happy", "sad", "tired", "busy", "hungry", "sick", "ready",
    "available", "asleep", "awake", "online", "offline", "home", "alone",
    "together", "right", "wrong", "true", "false", "green", "red",
    "open", "closed", "full", "empty", "same", "different", "present",
})

#: Bodies that are verdicts about rules, not rules ("the rule is
#: simple", "rules are meant to be broken").
_BODY_VETO = frozenset({
    "meant", "made", "broken", "simple", "complex", "easy", "difficult",
    "hard", "clear", "unclear", "strict", "basic", "optional", "there",
    "here", "impossible", "stupid", "dumb", "silly", "ridiculous",
    "outdated", "obsolete", "gone", "dead", "over", "done", "moot",
    "void", "null", "none", "nothing", "maybe", "perhaps", "probably",
})


# ---------------------------------------------------------------------------
# Span helpers — sentences, projection mapping, anchors
# ---------------------------------------------------------------------------

def _sentences(text: str) -> list[tuple[int, int, str]]:
    """Split ``norm.text`` into ``(start, end, terminator)`` sentence
    spans; the body span excludes the terminator."""
    out: list[tuple[int, int, str]] = []
    prev = 0
    for m in _BOUNDARY_RE.finditer(text):
        out.append((prev, m.start(), m.group(0)))
        prev = m.end()
    if prev < len(text):
        out.append((prev, len(text), ""))
    return [(s, e, t) for s, e, t in out if text[s:e].strip()]


def _first_word(sentence: str) -> str:
    m = re.match(r"\s*[^\W_]+", sentence)
    return m.group(0).strip().casefold() if m else ""


class _NormToRaw:
    """Exact ``norm.text`` char → raw-UTF-8-byte map, rebuilt from
    ``raw_text`` with a local copy of the §32.1 step-3 fold (the
    ``entities_v2._fold`` precedent — no private import from a sibling
    worker file). ``ok=False`` when the reconstruction disagrees with
    ``norm.text`` (identifier case aside) — the caller then falls back
    to term anchors."""

    def __init__(self, norm_text: str, raw_text: str) -> None:
        proj_chars: list[str] = []
        src_index: list[int] = []
        for i, ch in enumerate(raw_text):
            for fc in self._fold_char(ch):
                proj_chars.append(fc)
                src_index.append(i)
        # Reproduce the disp-projection whitespace collapse + strip.
        cmap: list[int] = []
        collapsed: list[str] = []
        for i, c in enumerate(proj_chars):
            if c.isspace():
                if collapsed and collapsed[-1] != " ":
                    collapsed.append(" ")
                    cmap.append(i)
            else:
                collapsed.append(c)
                cmap.append(i)
        s = "".join(collapsed)
        lead = len(s) - len(s.lstrip())
        stripped = s.strip()
        self.ok = bool(stripped) and (
            stripped.casefold() == norm_text.casefold())
        self._cmap = cmap[lead:lead + len(stripped)] if self.ok else []
        self._src_index = src_index
        self._raw_offsets = utf8_offsets(raw_text)
        self._raw_len = len(raw_text)

    @staticmethod
    def _fold_char(ch: str) -> str:
        s = unicodedata.normalize("NFKC", ch).casefold()
        s = unicodedata.normalize("NFD", s)
        return "".join(
            c for c in s if not unicodedata.category(c).startswith("M")
        )

    def span(self, cs: int, ce: int) -> Optional[tuple[int, int]]:
        """``norm.text`` char span → raw ``(byte_start, byte_end)``."""
        if not self.ok or cs < 0 or ce > len(self._cmap) or cs >= ce:
            return None
        ps = self._cmap[cs]
        pe = self._cmap[ce - 1]
        rs = self._src_index[ps]
        re_ = self._src_index[pe] + 1
        if rs < 0 or re_ > self._raw_len or rs >= re_:
            return None
        return (self._raw_offsets[rs], self._raw_offsets[re_])


def _isword(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _boundary_find(text: str, needle: str, start: int) -> Optional[int]:
    """``text.find(needle, start)`` with word-edge guards — "is" must not
    match inside "this". Needles with non-word edges ("#Eng") skip the
    corresponding check."""
    if not needle:
        return None
    pos = text.find(needle, start)
    while pos >= 0:
        left_ok = (
            pos == 0 or not _isword(text[pos - 1])
            or not _isword(needle[0])
        )
        end = pos + len(needle)
        right_ok = (
            end >= len(text) or not _isword(text[end])
            or not _isword(needle[-1])
        )
        if left_ok and right_ok:
            return pos
        pos = text.find(needle, pos + 1)
    return None


def _anchors(
    norm: NormAnalysis, raw_text: Optional[str],
) -> list[tuple[int, int, NormTerm]]:
    """Approximate ``norm.text`` char span of every term (text +
    identifier channels, byte order). ``find`` advances monotonically;
    clitic-generated terms ("not" out of "don't") fall back to their raw
    byte slice when ``raw_text`` is known, else drop out — interior
    misses never break pinning since only the outermost matched terms
    set the span."""
    text = norm.text or ""
    terms = [t for t in norm.terms if t.channel == "text"]
    terms += [t for t in norm.identifiers]
    terms.sort(key=lambda t: (t.byte_start, t.byte_end))
    raw = raw_text or ""
    raw_b = raw.encode("utf-8") if raw else b""
    out: list[tuple[int, int, NormTerm]] = []
    cursor = 0
    for t in terms:
        pos = _boundary_find(text, t.term, cursor)
        if pos is None and raw_b:
            try:
                surf = raw_b[t.byte_start:t.byte_end].decode("utf-8")
            except UnicodeDecodeError:
                surf = ""
            if surf and surf != t.term:
                pos = text.find(surf, cursor)
        if pos is None:
            continue
        out.append((pos, pos + len(t.term), t))
        cursor = pos + max(1, len(t.term))
    return out


def _pin_via_anchors(
    anchors: list[tuple[int, int, NormTerm]], cs: int, ce: int,
) -> Optional[tuple[int, int]]:
    """Byte span covering the anchors that intersect ``[cs, ce)``."""
    hits = [t for a, b, t in anchors if a < ce and b > cs]
    if not hits:
        return None
    return (min(t.byte_start for t in hits),
            max(t.byte_end for t in hits))


# ---------------------------------------------------------------------------
# Extracted-rule plumbing
# ---------------------------------------------------------------------------

@dataclass
class _Extracted:
    """What a pattern extractor returns. ``start``/``end`` are
    sentence-relative char offsets of the canonical rule text; ``kw`` is
    the trigger-keyword offset (hedge/quotative window =
    ``sentence[:kw]``); ``entities`` are extra surfaces to canon
    (channel dest, path, …); ``speaker`` marks first-person-scoped
    rules."""

    start: int
    end: int
    kw: int
    entities: tuple[str, ...] = ()
    speaker: bool = False


def _body_ok(sentence: str, start: int, end: int) -> bool:
    """§32.13's "with an action verb and an object": the span must carry
    ≥ 1 non-stopword token or an identifier-ish surface. Keeps bare
    "always be" / "make sure" out."""
    frag = sentence[start:end]
    for tok in re.findall(r"[\w#@$~/.'’-]+", frag):
        if tok.startswith(("#", "@", "$", "~/")) or "/" in tok:
            return True
        t = tok.strip(".'’-").casefold()
        if t and t not in _STOPWORDS and len(t) > 1:
            return True
    return False


def _hedge_hit(sentence: str, kw: int) -> bool:
    """True when the match is hedged, questioned-by-lead, quoted, or
    immediately undermined ("always maybe deploy"). The window is
    ``sentence[:kw]`` plus the token right after the keyword."""
    prefix = sentence[:kw]
    if _HEDGE_RE.search(prefix) or _QUOTATIVE_RE.search(prefix):
        return True
    if _first_word(sentence) in _LEAD_VETO:
        return True
    toks = re.findall(r"[^\W_]+", sentence[kw:kw + 40])
    if len(toks) > 1 and toks[1].casefold() in _HEDGE_WORDS:
        return True
    return False


def _kw(m: re.Match) -> int:
    """Trigger-keyword offset — the named ``kw`` group when the pattern
    declares one (head-forms), else the match start."""
    try:
        s = m.start("kw")
    except (IndexError, KeyError):
        s = -1
    return s if s >= 0 else m.start()


def _x_full(m: re.Match, sentence: str) -> Optional[_Extracted]:
    """Rule text = the whole match — the normative marker ("never",
    "the rule is", "make sure") is part of the standing instruction and
    stays inside the pinned span."""
    return _Extracted(start=m.start(), end=m.end(), kw=_kw(m))


def _x_decl(m: re.Match, sentence: str) -> Optional[_Extracted]:
    """``_x_full`` + a veto on verdict bodies ("the rule is simple",
    "rules are meant to be broken") — checked on the ``body`` group's
    first content word."""
    body = m.groupdict().get("body") or ""
    toks = re.findall(r"[^\W_]+", body)
    if toks and toks[0].casefold() in _BODY_VETO:
        return None
    return _x_full(m, sentence)


def _x_whole(m: re.Match, sentence: str) -> Optional[_Extracted]:
    return _Extracted(start=m.start(), end=m.end(), kw=_kw(m))


def _x_head(m: re.Match, sentence: str) -> Optional[_Extracted]:
    """Head-forms ("X only until Y", "X unless Y"): a stative verb in
    the head marks a description, not a rule."""
    head = m.groupdict().get("head") or ""
    if any(w.casefold() in _HEAD_STATIVE
           for w in re.findall(r"[^\W_]+", head)):
        return None
    return _x_whole(m, sentence)


def _x_only(m: re.Match, sentence: str) -> Optional[_Extracted]:
    """Lead-form "only X until/when Y": a determiner or quantity noun
    before ``only`` makes it adjectival ("the only way", "our only
    option")."""
    before = sentence[:m.start()].strip()
    if before:
        w = before.split()[-1].strip(",;:").casefold()
        if w in {"the", "a", "an", "that", "this", "my", "your", "its",
                 "their", "our", "one", "his", "her"}:
            return None
    nxt = re.match(r"\s*([^\W_]+)", sentence[m.end("kw"):])
    if nxt and nxt.group(1).casefold() in {
        "way", "thing", "person", "people", "reason", "time", "place",
        "option", "choice", "problem", "issue", "part", "downside",
        "catch", "difference", "one", "ones", "rule", "solution",
    }:
        return None
    return _x_whole(m, sentence)


#: Trailing continuation marker for convention rules — "post X to #a
#: and cc #b", "put tests in tests/ and fixtures in tests/fixtures/".
#: When the tail carries another destination/path surface, the rule span
#: extends to cover it.
_TAIL_SURFACE_RE = re.compile(r"#[\w-]+|@[\w-]+|(?:[\w*$.-]+/)+[\w*$.-]*")


def _extend_tail(m: re.Match, sentence: str) -> tuple[int, tuple[str, ...]]:
    """If the sentence tail after the match carries more destination or
    path surfaces, extend the rule to sentence end and return the extra
    surfaces for entity extraction."""
    tail = sentence[m.end():]
    extra = tuple(_TAIL_SURFACE_RE.findall(tail))
    if extra:
        return len(sentence), extra
    return m.end(), ()


def _x_channel(m: re.Match, sentence: str) -> Optional[_Extracted]:
    dest = m.group("dest")
    if not re.search(r"#|@|\bchannel\b|\bchat\b|\bthread\b", dest, re.I):
        return None
    end, extra = _extend_tail(m, sentence)
    return _Extracted(
        start=m.start(), end=end, kw=_kw(m),
        entities=(dest,) + extra)


def _x_path(m: re.Match, sentence: str) -> Optional[_Extracted]:
    path = m.group("path").rstrip(".")
    if not _PATH_RE.match(path):
        return None
    end, extra = _extend_tail(m, sentence)
    return _Extracted(
        start=m.start(), end=end, kw=_kw(m),
        entities=(path,) + extra)


#: Adverbial lead-ins that license ``always``/``never`` like a subject
#: ("please always run tests", "on fridays, never deploy").
_CLAUSE_LEAD = frozenset({"please", "just", "also", "kindly", "do"})
_CLAUSE_PUNCT = (",", ";", ":", "—", "–", "-", ")")

#: Unambiguous past forms for the ``always`` narrative veto ("i always
#: loved that place", "we always had lunch"). Deliberately narrower
#: than ``_PAST_FORMS``: body-position "cut"/"set"/"put"/"read" keep
#: their base-verb reading ("always put tests in tests/").
_ALWAYS_PAST = _IRREG_PAST | frozenset({
    "left", "led", "hid", "slid", "fed", "fled", "sped", "bled",
    "bred", "knelt", "leapt", "lit", "dealt", "dwelt", "dreamt",
})


def _subject_ok(sentence: str, kw: int) -> bool:
    """The word before ``always``/``never`` is an animate subject, a
    politeness lead-in, or a clause boundary; anything else ("the
    server is", "it") marks a description."""
    before = sentence[:kw].strip()
    if not before:
        return True
    if before.endswith(_CLAUSE_PUNCT):
        return True
    subj = before.split()[-1].strip(",;:").casefold()
    return subj in _ANIMATE_SUBJ or subj in _CLAUSE_LEAD


def _x_always(m: re.Match, sentence: str) -> Optional[_Extracted]:
    """``always`` needs an animate subject or a clause-initial seat;
    adjectives/prepositions right after it mean a description ("always
    late"), and a past-tense body recounts history ("we always had
    lunch"), not a rule."""
    if not _subject_ok(sentence, _kw(m)):
        return None
    nxt = m.group("body").split()
    if nxt:
        w = nxt[0].strip(".,;'").casefold()
        if w in _ALWAYS_BLOCK:
            return None
        if w in _ALWAYS_PAST:
            return None
        if w.endswith("ed") and len(w) > 3 and w not in _ED_BASE_VERBS:
            return None
    return _x_full(m, sentence)


def _x_never(m: re.Match, sentence: str) -> Optional[_Extracted]:
    """``never`` = ``always``'s guard plus a narrative veto: "i have
    never been", "she never called" recount history. An "-ed" surface
    rejects (base verbs ending in "ed" are whitelisted)."""
    before = sentence[:m.start()].strip()
    if before:
        toks = before.split()
        if toks[-1].casefold() in ("have", "has", "had", "having"):
            return None
        if not _subject_ok(sentence, _kw(m)):
            return None
    body = m.group("body").strip()
    if body.casefold().startswith("mind"):
        return None  # "never mind" is a revocation marker
    nxt = body.split()
    if nxt:
        w = nxt[0].strip(".,;'").casefold()
        if w in _ALWAYS_BLOCK:
            return None
        if w in _IRREG_PAST:
            return None
        if w.endswith("ed") and len(w) > 3 and w not in _ED_BASE_VERBS:
            return None
    return _x_full(m, sentence)


def _x_must(m: re.Match, sentence: str) -> Optional[_Extracted]:
    """Epistemic veto: "that must be nice" is inference, not a rule —
    "must be <w>" survives only for passive ("must be restarted") and
    "able"/"allowed" forms."""
    body = m.groupdict().get("body") or ""
    mm = re.match(r"be\s+([^\W_]+)", body)
    if mm:
        w = mm.group(1).casefold()
        if not (w.endswith("ed") or w.endswith("en")
                or w in {"able", "allowed", "kept", "done", "made"}):
            return None
    return _x_full(m, sentence)


def _x_whenever(m: re.Match, sentence: str) -> Optional[_Extracted]:
    """``whenever`` opens a rule only at a clause seat (sentence start
    or after punctuation); "i smile whenever i see her" makes the
    whenever-clause a condition, never the rule."""
    i = m.start()
    if i > 0:
        j = i - 1
        while j >= 0 and sentence[j] == " ":
            j -= 1
        if j >= 0 and sentence[j] not in ",;:—–-(":
            return None
    return _x_full(m, sentence)


#: Pronouns that precede a past-tense verb in a narrative head
#: ("i deployed the build before friday").
_NARR_SUBJ = frozenset({
    "i", "we", "you", "he", "she", "it", "they", "someone", "somebody",
})


def _is_past_word(w: str) -> bool:
    w = w.casefold()
    if w in _PAST_FORMS:
        return True
    return w.endswith("ed") and len(w) > 3 and w not in _ED_BASE_VERBS


def _x_time_tail(m: re.Match, sentence: str) -> Optional[_Extracted]:
    """Bare trailing bound ("deploy to prod before friday"): the head
    must read imperative — a stative verb anywhere ("the deadline is
    before friday") or a past-tense verb after a pronoun/animate
    subject ("he left", "i deployed the build") marks narrative fact,
    not a rule. An "-ed" adjective with no subject ("post updated
    docs before friday") survives."""
    ext = _x_head(m, sentence)
    if ext is None:
        return None
    toks = [w.casefold()
            for w in re.findall(r"[^\W_]+",
                                m.groupdict().get("head") or "")]
    if not toks:
        return None
    if _is_past_word(toks[-1]):
        return None
    for i, w in enumerate(toks[1:], start=1):
        if _is_past_word(w) and \
                (toks[i - 1] in _NARR_SUBJ
                 or toks[i - 1] in _ANIMATE_SUBJ):
            return None
    return ext


def _x_prefer(m: re.Match, sentence: str) -> Optional[_Extracted]:
    subj = (m.groupdict().get("subj") or "").casefold()
    speaker = subj in {"i", "we", "our team", "the team", "my team"}
    ext = _x_full(m, sentence)
    if ext is not None:
        ext = _Extracted(ext.start, ext.end, ext.kw, ext.entities,
                         speaker)
    return ext


# ---------------------------------------------------------------------------
# RULE_PATTERNS_V1 — ordered, most specific first (§32.13)
# ---------------------------------------------------------------------------

RULE_PATTERNS_V1: tuple[
    tuple[str, "re.Pattern[str]",
          Callable[[re.Match, str], Optional[_Extracted]]], ...
] = (
    # --- conditional negation: "don't X unless Y" ----------------------
    ("cond/dont_unless",
     re.compile(r"\b(?P<kw>do\s+not|don['’]t|dont|does\s+not"
                r"|doesn['’]t|doesnt)\s+(?P<body>.+?)(?:\s+unless\s+"
                r"(?P<cond>.+))?$", re.IGNORECASE),
     _x_full),
    # --- bounded: "post notes in #eng only until deploy is green" ------
    ("cond/only_until_head",
     re.compile(r"^(?P<head>.{3,80}?)\s+(?P<kw>only)\s+"
                r"(?P<sub>until|when|if|once|after)\s+(?P<cond>.+)$",
                re.IGNORECASE),
     _x_head),
    # --- bounded: "only ship when tests pass" ---------------------------
    ("cond/only_lead",
     re.compile(r"\b(?P<kw>only)\s+(?P<body>.+?)\s+"
                r"(?P<sub>until|when|if|unless|after|before|once)\s+"
                r"(?P<cond>.+)$", re.IGNORECASE),
     _x_only),
    # --- conditional: "ship it unless it breaks" ------------------------
    ("cond/unless_head",
     re.compile(r"^(?P<head>.{3,80}?)\s+(?P<kw>unless)\s+(?P<cond>.+)$",
                re.IGNORECASE),
     _x_head),
    # --- leading bound: "until the deploy is green, put notes in #eng" --
    ("cond/until_lead",
     re.compile(r"^(?P<kw>until|before|after)\s+(?P<cond>.+?)\s*,\s*"
                r"(?P<body>.+)$", re.IGNORECASE),
     _x_whole),
    # --- channel convention (V7-20.02): "post X to #chan" ---------------
    ("conv/channel",
     re.compile(r"\b(?P<kw>post|put|send|file|drop|share|announce"
                r"|publish|report|submit|deliver|route|forward|log|dump"
                r"|cc|bcc|ping|message|dm|email|tag|mirror|relay"
                r"|crosspost)\b(?P<mid>[^.!?]*?)\s+(?:to|in|into|on"
                r"|onto|under)\s+(?P<dest>#[\w.-]+|@[\w.-]+|(?:the\s+)"
                r"?[\w-]+\s+channel|channel\s+[\w-]+|[\w-]+\s+chat"
                r"|[\w-]+\s+thread)", re.IGNORECASE),
     _x_channel),
    # --- file-path conventions (V7-20.05) --------------------------------
    ("conv/path_put",
     re.compile(r"\b(?P<kw>put|keep|place|store|add|write|create|save"
                r"|move|commit|house|organize|organise|stick|toss|throw"
                r"|drop|file|nest|tuck|stash)\b(?P<mid>[^.!?]*?)\s+"
                r"(?:in|into|under|inside|to|at|within)\s+"
                r"(?P<path>(?:[\w*$.-]+/)+[\w*$.-]*|~[\w/.-]+|\$[\w.]+"
                r"(?:/[\w*$.-]*)*)", re.IGNORECASE),
     _x_path),
    ("conv/path_go",
     re.compile(r"\b(?P<thing>[\w][\w .'-]{0,40}?)\s+(?P<kw>go|goes"
                r"|live|lives|belong|belongs|sit|sits|stay|stays"
                r"|reside|resides)\s+(?:in|into|under|inside|within)\s+"
                r"(?P<path>(?:[\w*$.-]+/)+[\w*$.-]*)", re.IGNORECASE),
     _x_path),
    ("conv/path_for",
     re.compile(r"\b(?P<path>(?:[\w*$.-]+/)+[\w*$.-]*)\s+(?P<kw>is|are)"
                r"\s+(?:for|where|the\s+place\s+for|home\s+for)\s+"
                r"(?P<body>.+)$", re.IGNORECASE),
     _x_path),
    # --- declared rules: "the rule is X", "as a rule" --------------------
    ("decl/rule_is",
     re.compile(r"\b(?:(?:the|our|their|this|that|a|one|team|project"
                r"|house|general|golden|first|cardinal|main|standing"
                r"|ground|basic|fundamental|unspoken|unwritten"
                r"|explicit|implicit|hard|simple)\s+)*(?P<kw>rules?)"
                r"\s*(?:\([^)]*\))?\s*(?:is|are|:|says?|states?|reads?"
                r"|goes|—|–|-)\s*(?:that\s+|of\s+thumb\s*(?:is|:)?\s*)?"
                r"(?P<body>.+)$", re.IGNORECASE),
     _x_decl),
    ("decl/as_a_rule",
     re.compile(r"\b(?P<kw>as\s+a\s+(?:general\s+|house\s+|team\s+)"
                r"?rule(?:\s+of\s+thumb)?)\s*,?\s*(?P<body>.+)$",
                re.IGNORECASE),
     _x_decl),
    # --- declared conventions: "the convention is X" ---------------------
    ("decl/convention",
     re.compile(r"\b(?:(?:the|our|their|this|that|team|project|house"
                r"|company|org|repo|codebase|standing|local|standard"
                r"|usual|normal|general)\s+)*(?P<kw>convention|policy"
                r"|guideline|protocol|standard|practice|norm|motto"
                r"|house\s+rule|rule\s+of\s+thumb|style\s+rule|sla"
                r"|slo)s?\s+(?:is|are|:|says?|states?|reads?|requires?"
                r"|mandates?|dictates?|goes|demands?)\s*:?\s*"
                r"(?:that\s+)?(?P<body>.+)$", re.IGNORECASE),
     _x_decl),
    # --- deontic copula: "tests are required before merge" ----------------
    ("decl/required",
     re.compile(r"^(?P<head>.{2,80}?)\s+(?P<kw>is|are|was|were"
                r"|must\s+be|must\s+remain|remains?|stays?|shall\s+be)"
                r"\s+(?P<cls>required|mandatory|obligatory|compulsory"
                r"|forbidden|prohibited|banned|disallowed|not\s+allowed"
                r"|never\s+allowed|off[\s-]limits|verboten|expected"
                r"|allowed|permitted|optional|encouraged|discouraged)"
                r"\b(?P<tail>.*)$", re.IGNORECASE),
     _x_head),
    # --- normative frame: "we're supposed to X" ----------------------------
    ("decl/supposed_to",
     re.compile(r"\b(?P<subj>[\w][\w .'-]{0,30}?)\s+(?P<kw>am|is|are"
                r"|was|were|'m|'re|'s|’m|’re|’s|aren['’]t|isn['’]t"
                r"|wasn['’]t|weren['’]t|ain['’]t)\s+"
                r"(?P<neg>not\s+|never\s+)?supposed\s+to\s+"
                r"(?P<body>.+)$", re.IGNORECASE),
     _x_full),
    # --- preference-as-rule: "i prefer X for Y" ----------------------------
    ("pref/prefer",
     re.compile(r"\b(?P<subj>i|we|the team|our team|my team|everyone"
                r"|they|people|folks|the org|the company)\s+"
                r"(?P<kw>prefer)\s+(?P<body>.+)$", re.IGNORECASE),
     _x_prefer),
    ("pref/we_use",
     re.compile(r"\b(?P<subj>we|the team|our team|my team|the org"
                r"|the company|everyone|they|people)\s+"
                r"(?P<kw>use|uses|stick\s+with|go\s+with|run|runs"
                r"|standardize\s+on|standardise\s+on|standardized\s+on"
                r"|standardised\s+on)\s+(?P<body>.+?)\s+for\s+"
                r"(?P<ctx>.+)$", re.IGNORECASE),
     _x_prefer),
    # --- imperatives -----------------------------------------------------
    ("imp/make_sure",
     re.compile(r"\b(?P<kw>make\s+sure)\s+(?:that\s+|to\s+)?"
                r"(?P<body>.+)$", re.IGNORECASE),
     _x_full),
    ("imp/remember_to",
     re.compile(r"\b(?P<kw>remember\s+to)\s+(?P<body>.+)$",
                re.IGNORECASE),
     _x_full),
    ("imp/from_now_on",
     re.compile(r"\b(?P<kw>from\s+now\s+on|going\s+forward|henceforth"
                r"|henceforward|from\s+here\s+on(?:\s+out)?|in\s+the"
                r"\s+future|moving\s+forward)\s*,?\s*(?P<body>.+)$",
                re.IGNORECASE),
     _x_full),
    ("imp/whenever",
     re.compile(r"\b(?P<kw>whenever)\s+(?P<body>.+)$", re.IGNORECASE),
     _x_whenever),
    ("imp/by_default",
     re.compile(r"\b(?P<kw>by\s+default|as\s+a\s+default|the\s+default"
                r"\s+(?:is|rule\s+is|convention\s+is)|default\s+to"
                r"|defaults?\s+to)\s*,?\s*(?P<body>.+)$",
                re.IGNORECASE),
     _x_full),
    # --- modal normatives --------------------------------------------------
    ("modal/must",
     re.compile(r"\b(?:(?P<subj>[\w][\w .'-]{0,40}?)\s+)?(?P<kw>must)"
                r"\s+(?P<neg>not\s+|never\s+|n['’]t\s+)?(?P<body>.+)$",
                re.IGNORECASE),
     _x_must),
    ("modal/need_to",
     re.compile(r"\b(?P<subj>you|we|the team|our team|everyone|they"
                r"|people|devs|developers|engineers|agents|reviewers"
                r"|contributors|maintainers)\s+(?P<kw>have\s+to"
                r"|has\s+to|need\s+to|needs\s+to|got\s+to|gotta)\s+"
                r"(?P<body>.+)$", re.IGNORECASE),
     _x_full),
    # --- quantified normatives ----------------------------------------------
    ("kw/always",
     re.compile(r"\b(?P<kw>always)\s+(?P<body>.+)$", re.IGNORECASE),
     _x_always),
    ("kw/never",
     re.compile(r"\b(?P<kw>never)\s+(?P<body>.+)$", re.IGNORECASE),
     _x_never),
    ("kw/no_rule",
     re.compile(r"\b(?P<kw>no)\s+(?P<body>.+?)\s+(?:allowed|permitted"
                r"|past|beyond|after|before|on|without|during|over"
                r"|under|outside|except)\s+(?P<cond>.+)$",
                re.IGNORECASE),
     _x_full),
    ("kw/avoid",
     re.compile(r"\b(?P<kw>avoid)\s+(?P<body>.+)$", re.IGNORECASE),
     _x_full),
    # --- bare trailing bound: "deploy to prod before friday" -------------
    # Catch-all, runs last: keyword patterns above own their own spans.
    ("cond/time_tail",
     re.compile(r"^(?P<head>.{3,80}?)\s+(?P<kw>until|before|after"
                r"|when|if|once|whenever)\s+(?P<cond>.+)$",
                re.IGNORECASE),
     _x_time_tail),
)

PATTERN_IDS: tuple[str, ...] = tuple(p[0] for p in RULE_PATTERNS_V1)


# ---------------------------------------------------------------------------
# Revocation patterns
# ---------------------------------------------------------------------------

#: ``(marker_id, regex)`` — the captured ``tail`` is the revoked
#: content; empty tails (a bare "never mind") stay honest: they match
#: nothing in ``apply_revocations``.
REVOKE_PATTERNS_V1: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("rev/never_mind",
     re.compile(r"\bnever\s+mind\b[:\s]*(?P<tail>.*)", re.IGNORECASE)),
    ("rev/no_longer",
     re.compile(r"\b(?:[\w][\w .,'-]{0,40}?\s+)?no\s+longer\s+"
                r"(?P<tail>.+)$",
                re.IGNORECASE)),
    ("rev/not_anymore",
     re.compile(r"\b(?:do\s+not|don['’]t|dont|does\s+not|doesn['’]t"
                r"|doesnt|isn['’]t|aren['’]t|wasn['’]t|weren['’]t)\s+"
                r"(?P<tail>.+?)\s+any\s*more\b|\b(?P<tail2>.+?)\s+"
                r"anymore\b", re.IGNORECASE)),
    ("rev/stop_doing",
     re.compile(r"\b(?:stop|stopped|stops|quit|quitted)\s+"
                r"(?:doing\s+|using\s+)?(?P<tail>.+)$", re.IGNORECASE)),
    ("rev/forget_that",
     re.compile(r"\b(?:forget|scratch|disregard|ignore|drop|scrap"
                r"|cancel|rescind|rescinded)\s+"
                r"(?:(?:about|that|the|this|our|it|what\s+i\s+said"
                r"|rule|policy|convention|idea)\s+)*(?P<tail>.+)$",
                re.IGNORECASE)),
    ("rev/rule_gone",
     re.compile(r"\b(?:(?:that|the|this|our|old)\s+)?(?:rule|policy"
                r"|convention|guideline|requirement)\s+"
                r"(?:is|are|was|has\s+been|no\s+longer)\s+"
                r"(?:gone|dead|over|done|cancelled|canceled|rescinded"
                r"|retired|obsolete|deprecated|moot|void|null"
                r"|applies|apply)\b[:\s]*(?P<tail>.*)", re.IGNORECASE)),
)

_REV_TAIL_STRIP = frozenset({
    "that", "the", "this", "our", "a", "an", "we", "i", "you", "they",
    "rule", "rules", "policy", "convention", "about", "of", "to", "do",
    "doing", "don", "dont", "not", "no", "it", "its", "anymore", "any",
    "more", "longer", "please", "just", "also", "now", "is", "are",
    "on", "in", "at", "for", "with", "by", "from", "into", "onto",
})


# ---------------------------------------------------------------------------
# Term / entity helpers
# ---------------------------------------------------------------------------

def _terms_in_span(
    norm: NormAnalysis, bs: int, be: int,
) -> tuple[list[NormTerm], list[NormTerm]]:
    """``(text_terms, identifier_terms)`` intersecting the byte span."""
    txt = [t for t in norm.terms
           if t.channel == "text"
           and t.byte_start < be and t.byte_end > bs]
    ids = [t for t in norm.identifiers
           if t.byte_start < be and t.byte_end > bs]
    return txt, ids


def _canon_fn(canon_fn: Optional[Callable[[str], str]]):
    if canon_fn is not None:
        return canon_fn
    try:
        from verbatim.enrichment.entities_v2 import canon  # lazy
        return canon
    except Exception:
        return lambda s: str(s or "").strip().casefold()


def _content_terms(terms: Sequence[NormTerm], cap: int = _MAX_TOPICS
                   ) -> list[str]:
    """Ordered unique non-stopword terms — the rule's topic signature."""
    out: list[str] = []
    for t in terms:
        w = t.term.casefold()
        if w in _STOPWORDS or len(w) < 2:
            continue
        if w not in out:
            out.append(w)
        if len(out) >= cap:
            break
    return out


def _mention_canons(
    norm: NormAnalysis, unit_id: str, ms: int, me: int,
) -> list[str]:
    """``entities_v2.extract_mentions`` filtered to the match's
    ``norm.text`` span (mention byte offsets index ``norm.text``)."""
    try:
        from verbatim.enrichment.entities_v2 import extract_mentions
    except Exception:
        return []
    try:
        mentions = extract_mentions(norm, unit_id)
    except Exception:
        return []
    norm_off = utf8_offsets(norm.text or "")
    bs, be = norm_off[ms], norm_off[me]
    out: list[str] = []
    for mm in mentions:
        if mm.byte_start < be and mm.byte_end > bs and mm.canon:
            if mm.canon not in out:
                out.append(mm.canon)
    return out


#: Separator bytes that join an identifier head to a following term
#: ("#Eng" + "-releases" → ``eng releases``). A space gap does NOT join
#: ("#ops until" is two tokens, never "ops until").
_JOIN_SEPS = frozenset("-/.:_")


def _identifier_canons(
    span_ids: Sequence[NormTerm],
    norm: NormAnalysis,
    canon: Callable[[str], str],
    gap_char: Callable[[NormTerm], str],
) -> list[str]:
    """Canon each identifier in the span, plus the separator-joined
    continuation when the byte right after the identifier is one of
    ``-/.:_`` and a text term follows it."""
    out: list[str] = []
    text_terms = sorted(
        (t for t in norm.terms if t.channel == "text"),
        key=lambda t: t.byte_start)
    for t in span_ids:
        c = canon(t.term)
        if c and c not in out:
            out.append(c)
        nxt = next(
            (x for x in text_terms
             if x.byte_start >= t.byte_end
             and x.byte_start - t.byte_end <= 1),
            None,
        )
        if nxt is not None and nxt.byte_start > t.byte_end \
                and gap_char(t) in _JOIN_SEPS:
            cc = canon(t.term + " " + nxt.term)
            if cc and cc not in out:
                out.append(cc)
        elif nxt is not None and nxt.byte_start == t.byte_end:
            cc = canon(t.term + " " + nxt.term)
            if cc and cc not in out:
                out.append(cc)
    return out


# ---------------------------------------------------------------------------
# Until-expr resolution (temporal_v2 — clean-resolution-only contract)
# ---------------------------------------------------------------------------

def _resolve_until(expr: str, anchor_us: Optional[int]) -> Optional[int]:
    """Offer ``expr`` to ``temporal_v2.resolve``. A bound exists only
    when a candidate produced a known interval edge; the expiry is the
    latest resolvable edge. ``None`` = unresolved — the caller keeps
    the rule active and flags ``until_unresolved``."""
    if anchor_us is None or not expr:
        return None
    try:
        from verbatim.enrichment import temporal_v2
    except Exception:
        return None
    try:
        rts = temporal_v2.resolve(expr, int(anchor_us))
    except Exception:
        return None
    edges: list[int] = []
    for rt in rts:
        iv = rt.interval
        for v in (iv.end_us, iv.start_us):
            if v is not None:
                edges.append(int(v))
    return max(edges) if edges else None


def _scan_condition(
    text: str, sent_s: int, sent_e: int, kw_abs: int,
) -> tuple[Optional[tuple[int, int, str]], Optional[tuple[int, int, str]]]:
    """Scan the containing sentence for the first until/before/after
    clause → ``((abs_s, abs_e, expr_text), (abs_s, abs_e, cond_text))``.
    Both keep their connective verbatim. A clause opening exactly at the
    trigger keyword is skipped (the ``whenever`` in a whenever-rule is
    the marker, not a captured condition)."""
    until: Optional[tuple[int, int, str]] = None
    cond: Optional[tuple[int, int, str]] = None
    for m in _COND_RE.finditer(text[sent_s:sent_e]):
        head = m.group(1).casefold()
        abs_s = sent_s + m.start()
        abs_e = sent_s + m.end()
        if abs_s == kw_abs and head in _CONDITION_HEADS:
            continue  # the trigger keyword itself ("whenever …")
        frag = text[abs_s:abs_e].strip()
        if head in _UNTIL_HEADS and until is None:
            until = (abs_s, abs_e, frag)
        elif head in _CONDITION_HEADS and cond is None:
            cond = (abs_s, abs_e, frag)
    return until, cond


# ---------------------------------------------------------------------------
# Public API — detection
# ---------------------------------------------------------------------------

def detect_rules(
    norm: NormAnalysis,
    unit_id: str,
    speaker_canon: Optional[str] = None,
    raw_text: Optional[str] = None,
    *,
    scope_id: str = "",
    generation: int = 0,
    now_us: Optional[int] = None,
    anchor_us: Optional[int] = None,
    perspective: Optional[str] = None,
    canon_fn: Optional[Callable[[str], str]] = None,
    stats: Optional[dict[str, int]] = None,
) -> list[dict]:
    """Detect standing rules in one unit's ``norm/v2`` analysis.

    Returns ``standing_rules`` row dicts (``RULE_COLUMNS`` keys).
    ``raw_text`` is the unit's source text — when supplied, pins are
    rebuilt against exact UTF-8 bytes (``raw_verified``); without it the
    pin spans the matched terms' source byte offsets (``term_offsets``).
    ``now_us`` timestamps the row (``created_us``) and is the default
    ``anchor_us`` for the ``valid_until`` resolution attempt (§32.9's
    anchor A is the caller's occurred value — pass it explicitly).
    ``perspective`` labels agent-stated rules; they never override user
    rules (§32.13)."""
    local = {
        "sentences": 0, "candidates": 0, "emitted": 0,
        "dropped_hedged": 0, "dropped_empty_body": 0,
        "dropped_unpinned": 0,
    }
    COUNTERS["units"] += 1

    text = (norm.text or "") if norm is not None else ""
    if not text.strip():
        return []

    canon = _canon_fn(canon_fn)
    anchor = anchor_us if anchor_us is not None else now_us
    raw_bytes = raw_text.encode("utf-8") if raw_text is not None else None

    mapper = _NormToRaw(text, raw_text) if raw_text is not None else None
    if mapper is not None and not mapper.ok:
        mapper = None
    anchor_list = _anchors(norm, raw_text)
    # gap-char resolver for the identifier-join rule: the raw byte when
    # known, else the ``norm.text`` char after the term's anchor.
    _anchor_pos = {id(t): (a, b) for a, b, t in anchor_list}

    def gap_char(t: NormTerm) -> str:
        if raw_bytes is not None and t.byte_end < len(raw_bytes):
            try:
                return raw_bytes[t.byte_end:t.byte_end + 1].decode(
                    "ascii")
            except UnicodeDecodeError:
                return ""
        ab = _anchor_pos.get(id(t))
        if ab is not None and ab[1] < len(text):
            return text[ab[1]]
        return ""

    def pin_of(cs: int, ce: int) -> Optional[tuple[int, int]]:
        if mapper is not None:
            return mapper.span(cs, ce)
        return _pin_via_anchors(anchor_list, cs, ce)

    def slice_text(cs: int, ce: int,
                   bs_be: Optional[tuple[int, int]]) -> str:
        if raw_bytes is not None and bs_be is not None:
            s, e = bs_be
            if 0 <= s < e <= len(raw_bytes):
                try:
                    return raw_bytes[s:e].decode("utf-8")
                except UnicodeDecodeError:
                    pass
        return text[cs:ce].strip()

    rows: list[dict] = []
    accepted: list[tuple[int, int]] = []

    for sent_s, sent_e, term in _sentences(text):
        local["sentences"] += 1
        sentence = text[sent_s:sent_e]
        if "?" in term or "?" in sentence:
            continue  # interrogative — never a rule
        if _VETO_RE.search(sentence):
            continue  # revocation vocabulary — see detect_revocations

        for pid, rx, extractor in RULE_PATTERNS_V1:
            for m in rx.finditer(sentence):
                local["candidates"] += 1
                ext = extractor(m, sentence)
                if ext is None:
                    continue
                rs, re_ = ext.start, ext.end
                if rs >= re_:
                    continue
                if _hedge_hit(sentence, ext.kw):
                    local["dropped_hedged"] += 1
                    continue
                if not _body_ok(sentence, rs, re_):
                    local["dropped_empty_body"] += 1
                    continue
                abs_s, abs_e = sent_s + rs, sent_s + re_
                if any(abs_s < ae and abs_e > as_
                       for as_, ae in accepted):
                    continue  # a more specific pattern already owns it

                # conditional clauses anywhere in the sentence bind to
                # this rule; a trailing clause extends the pinned span.
                until, cond = _scan_condition(
                    text, sent_s, sent_e, sent_s + ext.kw)
                if until is not None and until[0] >= abs_s \
                        and until[1] > abs_e:
                    abs_e = until[1]
                elif cond is not None and cond[0] >= abs_s \
                        and cond[1] > abs_e:
                    abs_e = cond[1]

                bs_be = pin_of(abs_s, abs_e)
                if bs_be is None:
                    local["dropped_unpinned"] += 1
                    continue
                bs, be = bs_be
                rule_text = slice_text(abs_s, abs_e, bs_be)
                if not rule_text.strip():
                    local["dropped_unpinned"] += 1
                    continue

                until_expr: Optional[str] = None
                until_end_us: Optional[int] = None
                until_unresolved = False
                if until is not None:
                    us_, ue, uexpr = until
                    upin = pin_of(us_, ue)
                    until_expr = (
                        slice_text(us_, ue, upin) if upin else uexpr
                    ).strip().rstrip(",;").strip() or uexpr.strip(
                    ).rstrip(",;").strip()
                    until_end_us = _resolve_until(until_expr, anchor)
                    until_unresolved = until_end_us is None
                cond_text = cond[2] if cond is not None else None
                if cond_text is None:
                    # head-patterns whose kw IS the connective
                    # ("ship it unless red") are skipped by _COND_RE;
                    # rebuild the clause verbatim from the match.
                    try:
                        cend = m.end("cond")
                    except IndexError:
                        cend = -1
                    if cend > 0:
                        pre = sentence[:m.start("cond")].rstrip()
                        ctok = (pre.split()[-1].casefold()
                                if pre else "")
                        if ctok in _UNTIL_HEADS | _CONDITION_HEADS:
                            cstart = len(pre) - len(ctok)
                            cond_text = sentence[cstart:cend].strip()

                # trigger scoping
                span_terms, span_ids = _terms_in_span(norm, bs, be)
                entities: list[str] = []
                for c in _identifier_canons(span_ids, norm, canon,
                                            gap_char):
                    if c not in entities:
                        entities.append(c)
                for surf in ext.entities:
                    c = canon(surf)
                    if c and c not in entities:
                        entities.append(c)
                for c in _mention_canons(norm, unit_id, abs_s, abs_e):
                    if c not in entities:
                        entities.append(c)
                if ext.speaker and speaker_canon:
                    sc = canon(speaker_canon)
                    if sc and sc not in entities:
                        entities.append(sc)
                topics = _content_terms(span_terms)
                for t in span_ids:
                    tc = canon(t.term)
                    for piece in tc.split():
                        if piece and piece not in _STOPWORDS \
                                and piece not in topics \
                                and len(topics) < _MAX_TOPICS:
                            topics.append(piece)

                signals: dict[str, Any] = {
                    "pattern": pid,
                    "pinned": ("raw_verified" if mapper is not None
                               else "term_offsets"),
                }
                if cond_text:
                    signals["condition"] = cond_text
                if until_unresolved:
                    signals["until_unresolved"] = True
                if perspective and str(perspective).casefold() in (
                    _PERSPECTIVES_AGENT
                ):
                    signals["agent_stated"] = True
                if ext.speaker:
                    signals["speaker_scoped"] = True

                status = "active"
                if until_end_us is not None and now_us is not None \
                        and until_end_us <= now_us:
                    status = "expired"

                pin = {
                    "byte_start": bs,
                    "byte_end": be,
                    "unit_id": unit_id,
                    "text": rule_text,
                    "pattern": pid,
                    "created_us": now_us,
                    "anchor_us": anchor,
                    "until_end_us": until_end_us,
                    "signals": signals,
                }
                rid = "rule7:" + hashlib.sha256(json_dumps({
                    "v": EXTRACTOR_ID,
                    "unit": unit_id,
                    "pat": pid,
                    "s": bs,
                    "e": be,
                }).encode("utf-8")).hexdigest()[:24]

                rows.append({
                    "rule_id": rid,
                    "scope_id": scope_id,
                    "unit_id": unit_id,
                    "text_pin": json_dumps(pin),
                    "trigger_entities": json_dumps(entities),
                    "trigger_topics": json_dumps(topics),
                    "valid_until_expr": until_expr,
                    "status": status,
                    "generation": generation,
                })
                accepted.append((abs_s, abs_e))
                local["emitted"] += 1
                break  # one pattern per match region

    def _bs(r: dict) -> int:
        pin = safe_json_loads(r["text_pin"])
        return int(pin.get("byte_start") or 0)

    rows.sort(key=lambda r: (_bs(r), r["rule_id"]))
    for k, v in local.items():
        COUNTERS[k] += v
    if stats is not None:
        stats.update(local)
    return rows


# ---------------------------------------------------------------------------
# Revocation detection (pure) + application (conn)
# ---------------------------------------------------------------------------

def detect_revocations(
    norm: NormAnalysis,
    unit_id: str,
    raw_text: Optional[str] = None,
) -> list[dict]:
    """Explicit revocation phrases in one unit. Returns
    ``{marker, target_text, span, unit_id, raw_tail}`` dicts —
    ``target_text`` is the revoked content cleaned of stopword noise,
    ``span`` the byte pin of the revocation sentence (when mappable).
    Application is :func:`apply_revocations`."""
    text = (norm.text or "") if norm is not None else ""
    if not text.strip():
        return []
    mapper = _NormToRaw(text, raw_text) if raw_text is not None else None
    if mapper is not None and not mapper.ok:
        mapper = None
    anchor_list = None if mapper is not None else _anchors(norm, raw_text)

    out: list[dict] = []
    for sent_s, sent_e, _term in _sentences(text):
        sentence = text[sent_s:sent_e]
        for rid, rx in REVOKE_PATTERNS_V1:
            m = rx.search(sentence)
            if not m:
                continue
            gd = m.groupdict()
            tail = gd.get("tail") or gd.get("tail2") or ""
            tail = tail.strip(" \t.,;:!?\"'")
            words = [w for w in tail.split()
                     if w.strip(".,;'").casefold() not in _REV_TAIL_STRIP]
            target = " ".join(words).strip()
            abs_s = sent_s + m.start()
            abs_e = sent_s + m.end()
            if mapper is not None:
                span = mapper.span(abs_s, abs_e)
            else:
                span = _pin_via_anchors(anchor_list, abs_s, abs_e)
            out.append({
                "marker": rid,
                "target_text": target,
                "unit_id": unit_id,
                "span": span,
                "raw_tail": tail,
            })
            COUNTERS["revocations"] += 1
            break  # one marker per sentence
    return out


def _content_set(text_: str) -> frozenset:
    try:
        from verbatim.text.norm_v2 import fold
        t = fold(text_)
    except Exception:
        t = str(text_ or "").casefold()
    return frozenset(
        w for w in re.findall(r"[^\W_]+", t)
        if w not in _STOPWORDS and len(w) > 1
    )


def _overlap(a: frozenset, b: frozenset) -> float:
    """Containment overlap ``|a∩b| / min(|a|,|b|)`` — the revocation
    need only cover the smaller term set (the spec's ≥ 0.7 text-overlap
    bound, conservative side chosen)."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _fetch_active(
    conn, scope_id: str, generation: Optional[int] = None,
) -> list[dict]:
    sql = (
        "SELECT rule_id, unit_id, text_pin, trigger_entities,"
        " trigger_topics, valid_until_expr, status, generation"
        " FROM standing_rules WHERE scope_id = ? AND status = 'active'"
    )
    args: list[Any] = [scope_id]
    if generation is not None:
        sql += " AND generation <= ?"
        args.append(int(generation))
    rows = []
    for r in conn.execute(sql, args):
        rows.append({
            "rule_id": r[0], "unit_id": r[1],
            "pin": safe_json_loads(r[2] or "{}"),
            "trigger_entities": safe_json_loads(r[3] or "[]"),
            "trigger_topics": safe_json_loads(r[4] or "[]"),
            "valid_until_expr": r[5], "status": r[6],
            "generation": r[7],
        })
    return rows


def apply_revocations(
    conn,
    *,
    scope_id: str,
    revocations: Sequence[dict],
    min_overlap: float = REVOCATION_MIN_OVERLAP,
    generation: Optional[int] = None,
) -> list[str]:
    """Mark same-scope active rules ``revoked`` when a revocation's
    content terms overlap the rule's pinned text ≥ ``min_overlap``.
    Returns the revoked rule_ids in deterministic order."""
    if not revocations:
        return []
    rules = _fetch_active(conn, scope_id, generation)
    revoked: list[str] = []
    for rev in revocations:
        target = _content_set(rev.get("target_text") or "")
        if not target:
            continue
        for rule in rules:
            if rule["rule_id"] in revoked:
                continue
            pin_text = str(rule["pin"].get("text") or "")
            if _overlap(target, _content_set(pin_text)) >= min_overlap:
                conn.execute(
                    "UPDATE standing_rules SET status = 'revoked'"
                    " WHERE rule_id = ?", (rule["rule_id"],))
                revoked.append(rule["rule_id"])
    return sorted(revoked)


def sweep_expired(
    conn,
    *,
    now_us: int,
    scope_id: Optional[str] = None,
    generation: Optional[int] = None,
) -> int:
    """``active`` → ``expired`` where a resolved ``until_end_us`` has
    passed. Unresolved exprs never expire. Returns the flipped count."""
    sql = ("SELECT rule_id, text_pin FROM standing_rules"
           " WHERE status = 'active'")
    args: list[Any] = []
    if scope_id is not None:
        sql += " AND scope_id = ?"
        args.append(scope_id)
    if generation is not None:
        sql += " AND generation <= ?"
        args.append(int(generation))
    n = 0
    for rid, pin_json in conn.execute(sql, args).fetchall():
        pin = safe_json_loads(pin_json or "{}")
        end = pin.get("until_end_us")
        if end is not None and int(end) <= int(now_us):
            conn.execute(
                "UPDATE standing_rules SET status = 'expired'"
                " WHERE rule_id = ?", (rid,))
            n += 1
    return n


def insert_rules(conn, rows: Sequence[dict]) -> int:
    """Insert ``detect_rules`` row dicts. The caller owns the
    transaction; returns the inserted count."""
    cols = ", ".join(RULE_COLUMNS)
    ph = ", ".join("?" for _ in RULE_COLUMNS)
    n = 0
    for r in rows:
        conn.execute(
            f"INSERT OR REPLACE INTO standing_rules ({cols})"
            f" VALUES ({ph})",
            tuple(r[c] for c in RULE_COLUMNS),
        )
        n += 1
    return n


# ---------------------------------------------------------------------------
# Prefetch render — the "## STANDING RULES" block (V7-20.02)
# ---------------------------------------------------------------------------

def _task_signals(
    task_text: str,
    entities: Sequence[str],
    topics: Sequence[str],
    canon: Callable[[str], str],
) -> tuple[set, set]:
    """``(entity_canons, topic_terms)`` for the task side: caller args
    (canon'd) + identifier surfaces + path components + capitalized
    mentions + adjacency-extended identifier canons + content terms +
    n-gram canons (≤ 4 tokens) so multi-word rule entities match."""
    ent = {c for c in (canon(e) for e in entities) if c}
    top = {c for c in (canon(t) for t in topics) if c}
    text = str(task_text or "")
    if not text.strip():
        return ent, top

    toks: list[str] = []
    raw_b = text.encode("utf-8")
    try:
        from verbatim.text.norm_v2 import analyze
        tn = analyze(text)
        toks = [t.term for t in tn.terms if t.channel == "text"]
        idents = list(tn.identifiers)
        for t in idents:
            c = canon(t.term)
            if c:
                ent.add(c)
        # separator-joined identifier canons ("#eng" + "-releases" →
        # "eng releases"); a space gap never joins.
        for t in idents:
            gap = (chr(raw_b[t.byte_end])
                   if t.byte_end < len(raw_b)
                   and raw_b[t.byte_end] < 128 else "")
            if gap not in _JOIN_SEPS:
                continue
            for nxt in tn.terms:
                if nxt.channel == "text" \
                        and nxt.byte_start == t.byte_end + 1:
                    cc = canon(t.term + " " + nxt.term)
                    if cc:
                        ent.add(cc)
                    break
        try:
            from verbatim.enrichment.entities_v2 import (
                extract_mentions,
            )
            for mm in extract_mentions(tn, "task"):
                if mm.canon:
                    ent.add(mm.canon)
        except Exception:
            pass
    except Exception:
        try:
            from verbatim.text.norm_v2 import fold
            toks = re.findall(r"[^\W_]+", fold(text))
        except Exception:
            toks = re.findall(r"[^\W_]+", text.casefold())

    # path components: "tests/foo.py" scopes file-path rules
    for raw_tok in re.findall(r"[\w$~.-]+(?:/[\w*$.-]+)+/?", text):
        c = canon(raw_tok)
        if c:
            ent.add(c)
        for piece in raw_tok.split("/"):
            pc = canon(piece)
            if pc:
                top.add(pc)

    for w in toks:
        cw = canon(w)
        if cw:
            ent.add(cw)
        if w not in _STOPWORDS and len(w) > 1:
            top.add(w)
    # n-gram canons for multi-token entities ("eng releases")
    for n in (2, 3, 4):
        for i in range(0, len(toks) - n + 1):
            c = canon(" ".join(toks[i:i + n]))
            if c:
                ent.add(c)
    return ent, top


def prefetch_block(
    conn,
    *,
    scope_id: str,
    generation: int,
    task_text: str = "",
    entities: Sequence[str] = (),
    topics: Sequence[str] = (),
    limit: int = 8,
    now_us: Optional[int] = None,
    canon_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """Render the ``## STANDING RULES`` block for a task.

    Rules whose ``trigger_entities``/``trigger_topics`` overlap the task
    rank above global (untriggered) rules; ties break on the
    agent-stated label (user rules first — §32.13) then recency
    (``created_us``), then ``rule_id`` for determinism. Expired
    (resolved ``until_end_us`` ≤ ``now_us``) and non-active rows are
    excluded; unresolved exprs stay active with ``until_unresolved``.
    Returns ``""`` when nothing applies — never a fabricated block."""
    canon = _canon_fn(canon_fn)
    rows = _fetch_active(conn, scope_id, generation)
    if not rows:
        return ""

    tent, ttop = _task_signals(task_text, entities, topics, canon)

    scored: list[tuple[tuple, dict]] = []
    for row in rows:
        pin = row["pin"]
        end = pin.get("until_end_us")
        if end is not None and now_us is not None \
                and int(end) <= int(now_us):
            continue  # expired (lazy; sweep_expired owns the UPDATE)
        rents = {c for c in
                 (canon(e) for e in row["trigger_entities"]) if c}
        rtops = {c for c in
                 (canon(t) for t in row["trigger_topics"]) if c}
        if rents or rtops:
            triggered = bool((rents & tent) or (rtops & ttop))
            if not triggered:
                continue  # scoped rule, no overlap
        else:
            triggered = False  # global rule — always deliverable
        agent = bool(pin.get("signals", {}).get("agent_stated"))
        created = pin.get("created_us") or 0
        key = (
            0 if triggered else 1,   # specificity: triggered > global
            1 if agent else 0,       # user rules never overridden
            -int(created),           # recency
            row["rule_id"],          # determinism
        )
        scored.append((key, row))

    scored.sort(key=lambda kr: kr[0])
    lines: list[str] = []
    for _key, row in scored[:max(1, int(limit))]:
        pin = row["pin"]
        text_ = str(pin.get("text") or "").strip()
        if not text_:
            continue
        unit = row["unit_id"] or pin.get("unit_id") or row["rule_id"]
        ref = (f"[{unit}:{pin.get('byte_start', '?')}"
               f"-{pin.get('byte_end', '?')}]")
        expr = row["valid_until_expr"]
        suffix = ""
        if expr and str(expr).strip() \
                and str(expr).casefold() not in text_.casefold():
            suffix = f" ({str(expr).strip()})"
        lines.append(f"- {ref} {text_}{suffix}")
    if not lines:
        return ""
    return "## STANDING RULES\n" + "\n".join(lines)


__all__ = [
    "EXTRACTOR_ID",
    "FORMULA_STATUS",
    "COUNTERS",
    "reset_counters",
    "RULE_COLUMNS",
    "RULE_STATUSES",
    "RULE_PATTERNS_V1",
    "PATTERN_IDS",
    "REVOKE_PATTERNS_V1",
    "REVOCATION_MIN_OVERLAP",
    "detect_rules",
    "detect_revocations",
    "apply_revocations",
    "sweep_expired",
    "insert_rules",
    "prefetch_block",
]
