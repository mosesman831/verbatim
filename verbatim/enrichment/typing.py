"""Deterministic memory-type classification — ``enrich/v1`` (§7,
V5-30.25).

``classify_type(text)`` maps marker rules to a ``MemoryType``
interpretation label — never authority. Records whose text matches no
marker set are honestly ``untyped``; ``fact`` is never assigned by
markers (no reliable deterministic marker exists — V5-30.25).

Precedence when several marker sets fire (fixed, documented):

    preference > decision > plan > relationship > procedure_hint
        > absence > state > event > untyped

Rationale: stance verbs ("I no longer like X") keep ``preference`` so a
value/polarity change still lands in the same type for update-candidate
matching (V5-30.18); "we decided to remove X" is a ``decision`` record
about a removal, not an ``absence`` fact; "Alice no longer reports to
Bob" stays ``relationship``.
"""

from __future__ import annotations

import re

from verbatim.memory.types import ENRICHMENT_VERSION, MemoryType

TYPING_PRODUCER = ENRICHMENT_VERSION  # producer label stamped on rows

#: Cessation/removal vocabulary — "there is no X", "X was removed".
_ABSENCE_RE = re.compile(
    r"\b(?:no\s+longer|there\s+(?:is|are|was|were)\s+no\b|"
    r"(?:is|are|was|were|been|being)\s+(?:removed|deprecated|retired|"
    r"deleted|dropped|decommissioned|sunset(?:ted)?|obsoleted?|gone|"
    r"discontinued|end[-\s]of[-\s]life)|"
    r"(?:we|i|they)\s+(?:removed|deleted|dropped|deprecated|retired|"
    r"decommissioned|stopped\s+using|killed|axed)|"
    r"removed|removed\s+support\s+for|dropped\s+support\s+for|"
    r"no\s+more|does\s+not\s+exist|doesn['’]?t\s+exist|"
    r"went\s+away|is\s+gone|are\s+gone|no\s+support\s+for)\b",
    re.IGNORECASE,
)

#: First-person (and first-plural) stance verbs — negation/degree
#: adverbs may intervene ("I no longer like X" stays a preference so the
#: polarity flip lands in the same type for update-candidate matching).
_PREFERENCE_RE = re.compile(
    r"\b(?:i|we)\s+"
    r"(?:(?:really|still|also|just|no\s+longer|not|never|don['’]?t|"
    r"do\s+not|didn['’]?t|did\s+not|used\s+to|kind\s+of|sort\s+of|"
    r"absolutely|definitely|always)\s+)*"
    r"(?:like|liked|love|loved|prefer|prefers|preferred|enjoy|enjoyed|"
    r"hate|hated|dislike|disliked|adore|adored|detest|detested|fancy|"
    r"fancied|favou?r|favou?red|want|wanted|miss|missed|"
    r"can['’]?t\s+stand)\b"
    r"|\b(?:my|our)\s+(?:favou?rite|preferred)\b"
    r"|\bi['’]d\s+(?:rather|love|prefer)\b|\bi['’]m\s+(?:a\s+fan|into)\b"
    r"|\bfavou?rite\b|\bpref(?:er|ers|erred|erence|erences)\b",
    re.IGNORECASE,
)

#: Decision markers — explicit decision verbs and imperative adoption.
_DECISION_RE = re.compile(
    r"\b(?:we|i|they|the\s+team)\s+(?:decided|chose|agreed|settled\s+on|"
    r"opted|picked|committed\s+to)\b"
    r"|\blet['’]?s\s+(?:use|go\s+with|pick|choose|stick\s+with|adopt|"
    r"standardi[sz]e\s+on|move\s+to|switch\s+to)\b"
    r"|\bit\s+was\s+decided\b|\bdecision\s+is\b|\bdecided\s+to\b"
    r"|\bwe['’]ll\s+go\s+with\b|\bagreed\s+(?:on|to)\b"
    r"|\bthe\s+decision\b",
    re.IGNORECASE,
)

#: Future intent markers.
_PLAN_RE = re.compile(
    r"\b(?:plan(?:s|ned|ning)?\s+to|the\s+plan\s+is|"
    r"intend(?:s|ed|ing)?\s+to|going\s+to|gonna|aim(?:s|ed|ing)?\s+to|"
    r"(?:i|we|they|he|she)\s+will\b|\b(?:i|we|they|he|she)['’]ll\b|"
    r"schedul(?:ed|es)\s+(?:for|to|on)|to-?do\b|next\s+steps?\b|"
    r"roadmap\b|upcoming\b|will\s+(?:deploy|ship|migrate|move|start|"
    r"implement|use|happen|be))\b",
    re.IGNORECASE,
)

#: Relationship markers — who works/reports/collaborates with whom.
_RELATIONSHIP_RE = re.compile(
    r"\b(?:works?\s+(?:with|for|under)|reports?\s+(?:to|under)|"
    r"report\s+to|managed\s+by|manages(?!\s+to\b)|manager\s+of|"
    r"mentor(?:s|ed|ing)?\b|colleague|teammate|coworker|co-?worker|"
    r"married\s+to|partner(?:ed)?\s+with|collaborat\w+\s+with|"
    r"is\s+(?:my|our|his|her|their)\s+(?:manager|mentor|lead|boss|"
    r"colleague|teammate|report|partner|spouse)|"
    r"(?:my|our|his|her|their)\s+(?:manager|mentor|lead|boss|"
    r"colleague|teammate|report|spouse|partner)\b|"
    r"on\s+\w+['’]?s\s+team|member\s+of|leads(?!\s+to\b)|"
    r"leads\s+the|line\s+manager)\b",
    re.IGNORECASE,
)

#: Procedure hints — how-to / steps / runbook vocabulary.
_PROCEDURE_RE = re.compile(
    r"\b(?:how\s+to|steps?\s+to|procedure|runbook|recipe|instructions?|"
    r"workaround|to\s+fix|to\s+run|to\s+deploy|steps?:|"
    r"first\b.+\bthen\b|you\s+run|then\s+run|the\s+command\s+is|"
    r"run\s+the\s+command|to\s+restart|to\s+reset|checklist)\b",
    re.IGNORECASE,
)

#: Current-state markers.
_STATE_RE = re.compile(
    r"\b(?:currently|right\s+now|at\s+the\s+moment|as\s+of|these\s+days|"
    r"at\s+present|presently|for\s+now|is\s+now|are\s+now|now\s+uses?|"
    r"now\s+supports?|still\s+(?:uses?|running|on|is|are))\b",
    re.IGNORECASE,
)

#: Event markers — past-tense outcome verbs / incident vocabulary.
_EVENT_RE = re.compile(
    r"\b(?:happened|occurred|deployed|shipped|released|launched|"
    r"outage|incident|migrated|rolled\s*back|rolled\s*out|landed|"
    r"merged|crashed|failed|broke|went\s+down|completed|finished|"
    r"announced|met\s+with|took\s+place|fired|hired|joined|left\b|"
    r"resigned|graduated|got\s+(?:married|promoted|hired|fired))\b",
    re.IGNORECASE,
)

_RULES = (
    (MemoryType.PREFERENCE, _PREFERENCE_RE),
    (MemoryType.DECISION, _DECISION_RE),
    (MemoryType.PLAN, _PLAN_RE),
    (MemoryType.RELATIONSHIP, _RELATIONSHIP_RE),
    (MemoryType.PROCEDURE_HINT, _PROCEDURE_RE),
    (MemoryType.ABSENCE, _ABSENCE_RE),
    (MemoryType.STATE, _STATE_RE),
    (MemoryType.EVENT, _EVENT_RE),
)


def classify_type(text: str) -> MemoryType:
    """First matching marker set wins, in the documented precedence
    order. No markers → ``MemoryType.UNTYPED`` (an honest label)."""
    t = str(text if text is not None else "")
    for mtype, rx in _RULES:
        if rx.search(t):
            return mtype
    return MemoryType.UNTYPED
