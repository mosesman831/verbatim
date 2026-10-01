"""Deterministic polarity classification (§7, SPEC_V5 §30.1).

``polarity(text)`` maps a record's surface markers to one
``Polarity`` value. Marker precedence is fixed and documented —
``quoted`` → ``hypothetical`` → ``negated`` → ``hedged`` →
``affirmative``:

- **quoted** beats everything: quotation marks or reported-speech verbs
  ("she said …", "according to …") mean the content is attributed to
  another speaker; its inner polarity belongs to them.
- **hypothetical** beats negated: a conditional/modal frame
  ("if we deploy", "we would migrate", "could be") conditions the whole
  assertion, including any negation inside it.
- **negated** beats hedged: explicit negation ("not", "never",
  "no longer", n't-contractions, "without", "none") is the strongest
  remaining signal — except hedge *frames* checked first, because they
  wrap the negation itself: complementizer frames ("whether or not"),
  hearsay ("i heard", "word is"), and negated cognition ("don't think",
  "not sure", "doesn't seem").
- **hedged** markers port the clause-marker semantics of
  ``verbatim/evidence/supersession.py::_hedged``: hedge adverbs
  ("maybe", "reportedly"), bare "whether" (always hedges — it only ever
  introduces an indirect question), modal verbs that take a
  complementizer ("check whether", "confirm that" — a bare "ruff check"
  never hedges), hearsay ("they say", "word is"), and epistemic
  first-person forms ("i think", "i guess", "seems", "probably").
- otherwise **affirmative**.

Never invents a signal: a record with no markers is affirmative.
"""

from __future__ import annotations

import re

from verbatim.memory.types import Polarity

# ---------------------------------------------------------------- quoted

#: Balanced quote pairs. Apostrophes adjacent to word chars can never
#: open/close a single-quoted span, so "don't" can't pair with a later
#: quote. Backticks excluded — they mark code, not reported speech.
_QUOTE_PAIR_RE = re.compile(
    r'"[^"\n]+"|“[^”\n]+”|‘[^’\n]+’'
    r"|(?<![\w'])'[^'\n]+'(?![\w'])"
)
_REPORTED_RE = re.compile(
    r"\b(?:said|says|say|wrote|write|told|claimed|claims|stated|states|"
    r"announced|announces|reported|reports|tweeted|posted|commented|"
    r"according\s+to|as\s+reported\s+by|per\s+\w+|in\s+\w+'?s\s+words)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------- hypothetical

#: Conditional subordinators — the assertion they wrap never happened.
_CONDITIONAL_RE = re.compile(
    r"\b(?:if|unless|in\s+case|what\s+if|provided\s+that|providing\s+that|"
    r"assuming|suppose|supposing|imagine|hypothetically|let'?s\s+say|"
    r"lets\s+say|say\s+we|as\s+if|even\s+if)\b",
    re.IGNORECASE,
)

#: Conditional/possibility modals. "may" is deliberately excluded — it
#: collides with the month name; "should"/"shall" are excluded — they
#: mark recommendations, which typing treats as plan/decision signal.
_MODAL_HYPO_RE = re.compile(
    r"\b(?:would|could|might|would['’]?ve|could['’]?ve|might['’]?ve)\b"
    r"|\b\w+['’]d\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------- negated

#: Negated cognition — "don't think", "not sure", "doesn't seem" express
#: a hedged belief, not a negated fact; checked before plain negation.
_SOFT_NEG_RE = re.compile(
    r"\b(?:do(?:es|id)?\s+not|don['’]?t|doesn['’]?t|didn['’]?t)\s+"
    r"(?:think|believe|feel|seem|look|sound|appear|reckon|suppose|"
    r"expect|imagine|guess)\b"
    r"|\b(?:not|never)\s+(?:sure|certain|clear|convinced|confident|"
    r"positive)\b",
    re.IGNORECASE,
)

_NEGATION_RE = re.compile(
    r"\b(?:no\s+longer|not|never|cannot|can\s*not|without|none|nobody|"
    r"no\s+one|nothing|nowhere|neither|nor|hardly|barely|scarcely|"
    r"no|lack(?:s|ed|ing)?|devoid\s+of|free\s+of)\b"
    r"|\b\w+n['’]t\b|\bain['’]t\b|\bshan['’]t\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------- hedged

#: Hedge adverbs — hedge wherever they appear (ported from
#: supersession.py::_HEDGE_ADV_RE, extended with epistemic first-person
#: forms and likelihood adverbs).
_HEDGE_ADV_RE = re.compile(
    r"\b(?:maybe|perhaps|possibly|probably|likely|unlikely|reportedly|"
    r"rumou?red(?:ly)?|alleged(?:ly)?|apparently|supposedly|seemingly|"
    r"arguably|unsure|uncertain|unclear|presumably|conceivably|"
    r"i\s+think|i\s+guess|i\s+believe|i\s+suppose|i'?m\s+not\s+sure|"
    r"seems?|appears?|suggests?|tends?\s+to|in\s+theory)\b",
    re.IGNORECASE,
)

#: Modal verbs hedge only with an explicit complementizer ("check
#: whether X", "confirm that Y"); a bare verb never hedges — "ruff
#: check" is a command. Ported from supersession.py::_MODAL_VERB_RE.
_MODAL_VERB_RE = re.compile(
    r"\b(?:wonder(?:ing|ed)?|check(?:ing|ed)?|confirm(?:ing|ed)?|"
    r"verif(?:y|ying|ied)|ask(?:ing|ed)?|discuss(?:ing|ed)?|"
    r"question(?:ing|ed)?|consider(?:ing|ed)?|unsure|curious)\b",
    re.IGNORECASE,
)
_COMPLEMENT_RE = re.compile(r"\b(?:whether|if|that|about)\b",
                          re.IGNORECASE)

#: "whether" always hedges — it only ever introduces an indirect
#: question (supersession.py rule, same semantics at record level).
_WHETHER_RE = re.compile(r"\bwhether\b", re.IGNORECASE)

#: Hearsay attribution — reports a claim without asserting it.
_HEARSAY_RE = re.compile(
    r"\b(?:rumou?rs?\s+(?:says|has\s+it)|word\s+is|they\s+say|"
    r"i\s+heard|people\s+say|report\s+has\s+it|i'?m\s+told)\b",
    re.IGNORECASE,
)


def _is_quoted(text: str) -> bool:
    if _QUOTE_PAIR_RE.search(text):
        return True
    return bool(_REPORTED_RE.search(text))


def _is_hypothetical(text: str) -> bool:
    if _CONDITIONAL_RE.search(text):
        return True
    return bool(_MODAL_HYPO_RE.search(text))


def _is_negated(text: str) -> bool:
    return bool(_NEGATION_RE.search(text))


def _hedge_frame(text: str) -> bool:
    """Hedge frames that wrap any inner negation — checked BEFORE the
    negation pass: "whether or not it ran" is a question, "word is it
    broke" is hearsay; the ``not`` inside neither asserts a negated
    fact."""
    return bool(_WHETHER_RE.search(text) or _HEARSAY_RE.search(text))


def _is_hedged(text: str) -> bool:
    if _HEDGE_ADV_RE.search(text):
        return True
    for m in _MODAL_VERB_RE.finditer(text):
        if _COMPLEMENT_RE.search(text, m.end()):
            return True
    return False


def polarity(text: str) -> Polarity:
    """Deterministic marker polarity — see module docstring for the
    precedence contract."""
    t = str(text if text is not None else "")
    if _is_quoted(t):
        return Polarity.QUOTED
    if _is_hypothetical(t):
        return Polarity.HYPOTHETICAL
    if _hedge_frame(t):
        return Polarity.HEDGED
    if _SOFT_NEG_RE.search(t):
        return Polarity.HEDGED
    if _is_negated(t):
        return Polarity.NEGATED
    if _is_hedged(t):
        return Polarity.HEDGED
    return Polarity.AFFIRMATIVE
