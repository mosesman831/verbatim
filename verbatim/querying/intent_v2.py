"""Query intent classification — ``intent/v2`` (SPEC_V7 §32.14, V7-05.12/13).

Deterministic rules + owned, versioned, generic lexicons. No models, no I/O,
no clock: identical ``NormAnalysis`` input yields identical ``IntentResult``.

Rule order (first match wins for ``primary``; every matching rule is also
recorded in ``classes`` and ``rule_trace``, in evaluation order):

  R01 identifier        identifier-channel tokens present
  R02 temporal_point    when-questions / what date|time|day / how long ago /
                        how many <unit> ago
  R03 temporal_order    before|after|first|earlier|later|which came with
                        >= 2 distinct event phrases
  R04 duration          how long (not ago) / how many <unit> between / since /
                        duration
  R05 count_aggregate   how many|how much|count|number of|total
  R06 current_value     now|currently|current|still|latest|these days (+siblings)
  R07 history_of        used to|history|over time|changed|previously (+siblings)
  R08 preference        prefer|favorite|like better|would i enjoy|like /
                        recommend|suggest … for me
  R09 comparison        compare|difference|versus|vs|both|which of /
                        comparative adjective + than|or
  R10 why/causal        why|because|reason|what led (+siblings)
  R11 temporal_range    bounded-span questions: between/from..to/during/in/
                        on/over + temporal marker; this|last|next|past|coming +
                        marker; absolute day tokens; <period> ago; the other day
  R12 open_domain       would|likely|might|probably (+could|should|possibly)
                        with a persona (first-person) subject
  R13 abstain_likely    ever-verification / existence frames (have i ever,
                        did i mention, is there any, ...) or a query with no
                        content terms at all
  R14 multi_hop         >= 2 distinct entity canons, or a relative-clause chain
  R15 lookup            fallback

R11 and R13 are the two classes V7-05.12 requires that the §32.14 ordered list
does not explicitly enumerate. They are inserted where they cannot shadow an
enumerated rule: R11 sits after every question-type rule so that a temporal
span that merely *modifies* a count/current/history/preference/comparison/why
question stays secondary ("temporal windows are attached to any class" —
§32.14), and fires only when the question's frame is the bounded span itself.
R13 sits after open_domain so modal-persona questions keep their class, and
before multi_hop because an ever-verification frame dominates a 2-entity
mention ("have i ever met alice and bob" abstains, it does not hop).

``IntentResult.window`` is left ``None`` here: window resolution needs
``query_time_us`` (the caller's anchor) which ``classify`` does not receive;
the pipeline attaches it via ``temporal/v2`` ``resolve_query_window``
(V7-09.05). Range/point classes still *detect* the temporal frame.

``decompose(norm, intent)`` implements V7-05.13: deterministic entity- and
conjunction-based facet sub-queries (<= 3) for ``multi_hop`` and
``comparison`` primaries. Facets are built from the original ``NormTerm``
objects so byte offsets stay honest. A singleton ``(norm,)`` return means "no
split possible / not applicable" — callers may iterate the result
unconditionally.
"""

from __future__ import annotations

import re
from typing import Optional

from verbatim.core.types_v7 import (
    FORMULA_STATUS_PROVISIONAL,
    IntentClass,
    IntentResult,
    NormAnalysis,
    NormTerm,
)

INTENT_ID = "intent/v2"
LEXICON_ID = "intent_lex/v1"
FORMULA_STATUS = FORMULA_STATUS_PROVISIONAL
MAX_FACETS = 3


# ---------------------------------------------------------------------------
# Lexicons (owned, versioned, generic — no benchmark-derived content)
# ---------------------------------------------------------------------------

# Event predicates/nouns for the temporal_order "two event phrases" clause.
# Lemmas and common inflections drawn from the §32.10 event families plus
# generic event nouns; matched on whole folded tokens only.
_EVENT_TERMS = frozenset({
    # life/activity verbs (infinitive + common inflections)
    "move", "moved", "moving", "relocate", "relocated", "relocating",
    "start", "started", "starting", "begin", "began", "begun", "beginning",
    "join", "joined", "joining", "enroll", "enrolled", "enrolling",
    "quit", "quitting", "leave", "left", "leaving", "resign", "resigned",
    "graduate", "graduated", "graduating", "finish", "finished",
    "complete", "completed", "marry", "married", "marrying", "engage",
    "engaged", "divorce", "divorced", "date", "dated", "dating",
    "buy", "bought", "buying", "purchase", "purchased", "order", "ordered",
    "sell", "sold", "selling", "visit", "visited", "visiting", "travel",
    "traveled", "travelled", "traveling", "go", "went", "gone", "going",
    "fly", "flew", "flown", "flying", "return", "returned", "adopt",
    "adopted", "paint", "painted", "draw", "drew", "drawn", "write",
    "wrote", "written", "writing", "publish", "published", "record",
    "recorded", "run", "ran", "running", "race", "raced", "compete",
    "competed", "win", "won", "winning", "lose", "lost", "losing",
    "attend", "attended", "meet", "met", "meeting", "reunite", "reunited",
    "host", "hosted", "throw", "threw", "thrown", "celebrate",
    "celebrated", "cook", "cooked", "bake", "baked", "learn", "learned",
    "learnt", "take", "took", "taken", "volunteer", "volunteered",
    "donate", "donated", "mentor", "mentored", "recover", "recovered",
    "injure", "injured", "diagnose", "diagnosed", "hire", "hired",
    "promote", "promoted", "fire", "fired", "interview", "interviewed",
    "launch", "launched", "ship", "shipped", "release", "released",
    "book", "booked", "reserve", "reserved", "cancel", "cancelled",
    "canceled", "call", "called", "text", "texted", "email", "emailed",
    "read", "watch", "watched", "listen", "listened", "plant", "planted",
    "grow", "grew", "grown", "garden", "gardened", "renovate",
    "renovated", "build", "built", "fix", "fixed", "arrive", "arrived",
    "come", "came", "coming", "depart", "departed", "land", "landed",
    "open", "opened", "close", "closed", "sign", "signed", "break",
    "broke", "broken", "pay", "paid", "eat", "ate", "eaten", "drink",
    "drank", "drunk", "sleep", "slept", "wake", "woke", "woken", "try",
    "tried", "test", "tested", "deploy", "deployed", "submit",
    "submitted", "approve", "approved", "merge", "merged", "review",
    "reviewed", "present", "presented", "give", "gave", "given",
    "receive", "received", "send", "sent", "see", "saw", "seen", "talk",
    "talked", "speak", "spoke", "spoken", "discuss", "discussed", "chat",
    "chatted", "plan", "planned", "decide", "decided", "choose", "chose",
    "chosen", "change", "changed", "switch", "switched", "update",
    "updated", "upgrade", "upgraded", "install", "installed", "rent",
    "rented", "lease", "leased", "borrow", "borrowed", "lend", "lent",
    "find", "found", "miss", "missed", "pass", "passed", "fail", "failed",
    "study", "studied", "teach", "taught", "practice", "practiced",
    "perform", "performed", "play", "played", "hear", "heard", "feel",
    "felt", "happen", "happened", "occur", "occurred", "hold", "held",
    "schedule", "scheduled", "announce", "announced", "propose",
    "proposed", "accept", "accepted", "reject", "rejected", "decline",
    "declined", "agree", "agreed", "disagree", "confirm", "confirmed",
    "postpone", "postponed", "delay", "delayed", "reschedule",
    "rescheduled", "end", "ended", "stop", "stopped", "pause", "paused",
    "resume", "resumed", "continue", "continued", "repeat", "repeated",
    "transfer", "transferred", "apply", "applied", "onboard",
    "onboarded", "retire", "retired", "demote", "demoted", "evacuate",
    "evacuated", "enter", "entered", "exit", "exited", "reach",
    "reached", "cross", "crossed", "board", "boarded", "crash",
    "crashed", "deliver", "delivered", "file", "filed", "register",
    "registered", "vote", "voted", "score", "scored", "qualify",
    "qualified", "advance", "advanced", "eliminate", "eliminated",
    "organize", "organized", "arrange", "arranged", "prepare",
    "prepared", "pack", "packed", "unpack", "clean", "cleaned", "wash",
    "washed", "repair", "repaired", "replace", "replaced", "demolish",
    "demolished", "shift", "shifted", "slip", "slipped",
    # event nouns (generic — an event phrase may be nominal)
    "relocation", "graduation", "wedding", "marriage", "divorce",
    "purchase", "sale", "trip", "flight", "interview", "promotion",
    "launch", "release", "party", "concert", "race", "marathon",
    "surgery", "vacation", "appointment", "deadline", "exam",
    "renovation", "arrival", "departure", "birth", "birthday",
    "anniversary", "conference", "summit", "retreat", "onboarding",
    "workshop", "offsite", "seminar", "webinar", "demo", "deployment",
    "incident", "outage", "ceremony", "reception", "dinner", "lunch",
    "breakfast", "brunch", "hike", "festival", "fair", "exhibition",
    "tournament", "match", "game", "move",
})

# Temporal markers for the temporal_range detector (§32.9 families).
_MONTHS = frozenset({
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
})
_SEASONS = frozenset({"spring", "summer", "fall", "autumn", "winter"})
_WEEKDAYS = frozenset({
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "mon", "tue", "tues", "wed", "thu", "thur", "thurs",
    "fri", "sat", "sun",
})
_PERIOD_WORDS = frozenset({
    "day", "days", "week", "weeks", "weekend", "weekends", "month",
    "months", "year", "years", "quarter", "quarters", "decade",
    "decades", "hour", "hours", "morning", "afternoon", "evening",
    "night", "semester", "semesters",
})
_ABSOLUTE_DAYS = frozenset({"today", "yesterday", "tomorrow", "tonight"})
_HOLIDAY_WORDS = frozenset({
    "holiday", "holidays", "christmas", "easter", "thanksgiving",
    "halloween",
})

_RELATIVE_MARKERS = frozenset({"that", "who", "whom", "which", "whose"})

_ORDER_TRIGGERS = frozenset({"before", "after", "first", "earlier", "later"})

_CURRENT_TOKENS = frozenset({
    "now", "currently", "current", "still", "latest", "nowadays",
})

_HISTORY_TOKENS = frozenset({
    "history", "historical", "change", "changes", "changed",
    "previously", "previous", "timeline", "progression", "evolution",
    "evolved", "originally", "original",
})

_PREFERENCE_TOKENS = frozenset({
    "prefer", "prefers", "preferred", "preferring", "preference",
    "preferences", "favorite", "favorites", "favourite", "favourites",
})

_COMPARISON_TOKENS = frozenset({
    "compare", "compares", "compared", "comparing", "comparison",
    "comparisons", "difference", "differences", "differ", "differs",
    "differed", "versus", "vs", "both",
})

_COMPARATIVES = frozenset({
    "better", "worse", "more", "less", "fewer", "cheaper", "faster",
    "slower", "bigger", "smaller", "larger", "easier", "harder",
    "happier", "sadder", "longer", "shorter", "taller", "older",
    "newer", "earlier", "later", "closer", "farther", "further",
    "healthier", "stronger", "weaker", "richer", "poorer", "safer",
    "smarter", "warmer", "colder", "cooler", "heavier", "lighter",
    "brighter", "darker", "quieter", "louder", "nicer", "simpler",
    "deeper", "higher", "lower", "wider", "narrower", "thicker",
    "thinner", "greener", "cleaner",
})

_CAUSAL_TOKENS = frozenset({
    "why", "because", "reason", "reasons", "cause", "caused", "causes",
})

_OPEN_MODALS = frozenset({
    "would", "likely", "might", "probably", "possibly", "could", "should",
})

_FIRST_PERSON = frozenset({
    "i", "me", "my", "mine", "we", "us", "our", "ours", "myself",
    "ourselves",
})

# Function/scaffold vocabulary: (a) stripped from facet edges in decompose,
# (b) a query whose tokens are all function words has no content to retrieve
# -> abstain_likely (R13 second clause).
_FUNCTION_TOKENS = frozenset({
    "what", "which", "who", "whom", "whose", "when", "where", "why",
    "how", "is", "are", "was", "were", "will", "would", "do", "does",
    "did", "have", "has", "had", "can", "could", "should", "shall",
    "may", "might", "must", "be", "been", "being", "am", "the", "a",
    "an", "of", "to", "in", "on", "for", "at", "by", "with", "from",
    "about", "as", "into", "over", "after", "before", "between", "and",
    "or", "but", "not", "no", "i", "me", "my", "mine", "we", "us",
    "our", "ours", "you", "your", "yours", "it", "its", "that", "this",
    "these", "those", "there", "here", "any", "some", "all", "both",
    "ever", "still", "now", "so", "if", "than", "then", "just", "much",
    "many", "more", "most", "other", "another", "own", "same", "very",
    "really", "again", "back", "up", "out", "off", "please", "tell",
    "show", "give", "find", "get", "got", "know", "think", "say",
    "said", "see", "let", "make", "made", "take", "took", "go", "went",
    "come", "came", "thing", "things", "way", "something", "anything",
    "nothing", "everything", "someone", "anyone", "nobody", "somebody",
    "he", "she", "his", "her", "him", "they", "them", "their", "one",
    "s", "t", "re", "ll", "ve", "d", "m",
    # comparison scaffold (stripped from facet edges in decompose)
    "compare", "compares", "compared", "comparing", "comparison",
    "comparisons", "difference", "differences", "versus", "vs",
})

# Facet machinery (V7-05.13)
_COMPARISON_SPLITS = frozenset({"and", "or", "versus", "vs", "than"})
_MULTIHOP_SPLITS = frozenset({
    "and", "or", "that", "who", "whom", "which", "whose",
})
_TAIL_PREPOSITIONS = frozenset({
    "for", "to", "in", "on", "at", "with", "as", "about", "of",
})


# ---------------------------------------------------------------------------
# Phrase patterns (evaluated against the space-joined folded token stream)
# ---------------------------------------------------------------------------

_RX_WHEN_AUX = re.compile(
    r"\bwhen (did|does|do|is|was|were|will|would|have|has|had|are|"
    r"shall|should|can|could|may|might|must|am|the|a|an|my|our|your|"
    r"his|her|their|its|i|you|he|she|we|they|it|there|s)\b"
)
_RX_DATE_WH = re.compile(
    r"\b(what|which) (day|date|time|week|month|year)\b"
)
_RX_HOW_LONG_AGO = re.compile(r"\bhow long ago\b")
_RX_HOW_MANY_AGO = re.compile(
    r"\bhow many (days|weeks|months|years) ago\b"
)
_RX_HOW_LONG = re.compile(r"\bhow long (?!ago\b)")
_RX_DURATION_BETWEEN = re.compile(
    r"\bhow many (days|weeks|months|years) between\b"
)
_RX_HOW_MANY = re.compile(
    r"\bhow many (?!(?:days|weeks|months|years) (?:between|ago)\b)"
)
_RX_HOW_MUCH = re.compile(r"\bhow much\b")
_RX_NUMBER_OF = re.compile(r"\bnumber of\b")
_RX_THESE_DAYS = re.compile(r"\bthese days\b")
_RX_AT_THE_MOMENT = re.compile(r"\bat (the )?moment\b")
_RX_AT_PRESENT = re.compile(r"\bat present\b")
_RX_SO_FAR = re.compile(r"\bso far\b")
_RX_USED_TO = re.compile(r"\bused to\b")
_RX_OVER_TIME = re.compile(r"\bover time\b")
_RX_IN_THE_PAST = re.compile(r"\bin the past\b")
_RX_LIKE_BETTER = re.compile(r"\blike better\b")
_RX_WOULD_I_ENJOY = re.compile(r"\bwould (i|we) (enjoy|like|love|prefer)\b")
_RX_RECOMMEND_ME = re.compile(
    r"\b(recommend|recommendation|recommendations|recommending|suggest|"
    r"suggestion|suggestions|suggesting)\b[^?]*\b(for|to) (me|us)\b"
)
_RX_WHICH_OF = re.compile(r"\bwhich of\b")
_RX_WHAT_LED = re.compile(r"\bwhat (led|leads|lead)\b")
_RX_HOW_COME = re.compile(r"\bhow come\b")
_RX_WHAT_MADE = re.compile(r"\bwhat (made|makes|cause|causes|caused)\b")
_RX_DUE_TO = re.compile(r"\bdue to\b")
_RX_WHICH_CAME = re.compile(r"\bwhich (came|comes|come)\b")
_RX_THE_OTHER_DAY = re.compile(r"\bthe other day\b")
_RX_ABSTAIN_FRAME = re.compile(
    r"\b(?:"
    r"(?:have|has|did|do) (?:i|we) ever"
    r"|(?:have|has) (?:i|we) been"
    r"|(?:have|has|did|do) (?:i|we) "
    r"(?:mention|mentioned|tell|told|say|said|talk|talked|discuss|"
    r"discussed|bring|brought|write|wrote|note|noted|record|recorded|"
    r"cover|covered|describe|described|report|reported)"
    r"|(?:is|are|was|were|has|have|had) there (?:any|been|ever)"
    r"|(?:do|did|have|has) (?:i|we) (?:have|had|got|gotten) any"
    r"|do you (?:know|recall|remember) (?:if|whether)"
    r")\b"
)
# Aux-initial yes/no frames ("did i lock the door", "have we paid the
# invoice"): existence/verification questions that a memory store either
# confirms or abstains on. Wh-initial questions ("what did i say") are
# content recall, not this frame — enforced by checking position 0.
_ABSTAIN_AUX = frozenset({
    "is", "are", "was", "were", "do", "does", "did", "have", "has", "had",
})
_ABSTAIN_PERSON = frozenset({
    "i", "we", "you", "they", "he", "she", "it", "there",
})
_ABSTAIN_MARKERS = frozenset({
    "ever", "any", "been", "already", "yet", "once", "twice", "again",
    "anything", "something", "nothing", "everything", "none",
})
# "do you remember/think/know …" frames are discourse markers on a recall
# question, not existence checks — they must not trigger the content clause.
_DISCOURSE_VERBS = frozenset({
    "remember", "recall", "know", "think", "believe", "suppose",
    "notice", "wonder", "realize", "forget", "forgot", "imagine",
    "guess", "feel", "mean", "say",
})
_WH_START = frozenset({
    "what", "which", "who", "whom", "whose", "when", "where", "why",
    "how",
})
_RX_YEAR = re.compile(r"(?:19|20)\d\d")
_RX_QUARTER = re.compile(r"q[1-4]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lex_terms(norm: NormAnalysis) -> list[NormTerm]:
    """Folded lexical terms for rule matching.

    The identifier channel is deliberately excluded: code spans, URLs, and
    version strings may literally contain trigger words ("current_value",
    "still-123") and must not feed phrase matching. Identifier presence is
    rule R01's own signal, taken from the identifier channel directly.
    """
    text_terms = [t for t in norm.terms if t.channel == "text"]
    if text_terms:
        return list(text_terms)
    seen: set[str] = set()
    out: list[NormTerm] = []
    for t in norm.terms:
        if t.channel == "identifier":
            continue
        if t.term not in seen:
            seen.add(t.term)
            out.append(t)
    return out


def _joined(tokens: list[str]) -> str:
    return " " + " ".join(tokens) + " "


def _is_temporal_marker(tok: str) -> bool:
    if tok in _MONTHS or tok in _SEASONS or tok in _WEEKDAYS:
        return True
    if tok in _PERIOD_WORDS or tok in _ABSOLUTE_DAYS or tok in _HOLIDAY_WORDS:
        return True
    if _RX_YEAR.fullmatch(tok) or _RX_QUARTER.fullmatch(tok):
        return True
    return False


# ---------------------------------------------------------------------------
# Rule predicates — each is a pure function of (tokens, joined, canons, ids)
# ---------------------------------------------------------------------------


def _r01_identifier(t, j, canons, ids) -> bool:
    return bool(ids)


def _r02_temporal_point(t, j, canons, ids) -> bool:
    if _RX_HOW_LONG_AGO.search(j) or _RX_HOW_MANY_AGO.search(j):
        return True
    if _RX_DATE_WH.search(j):
        return True
    if t and t[0] == "when":
        return True
    return _RX_WHEN_AUX.search(j) is not None


def _r03_temporal_order(t, j, canons, ids) -> bool:
    trigger = any(x in _ORDER_TRIGGERS for x in t) or bool(
        _RX_WHICH_CAME.search(j)
    )
    if not trigger:
        return False
    events = {x for x in t if x in _EVENT_TERMS}
    return len(events) >= 2


def _r04_duration(t, j, canons, ids) -> bool:
    if _RX_HOW_LONG.search(j) or _RX_DURATION_BETWEEN.search(j):
        return True
    if "since" in t or "duration" in t:
        return True
    return False


def _r05_count_aggregate(t, j, canons, ids) -> bool:
    if _RX_HOW_MANY.search(j) or _RX_HOW_MUCH.search(j):
        return True
    if _RX_NUMBER_OF.search(j):
        return True
    return "count" in t or "total" in t or "totals" in t


def _r06_current_value(t, j, canons, ids) -> bool:
    if any(x in _CURRENT_TOKENS for x in t):
        return True
    return bool(
        _RX_THESE_DAYS.search(j)
        or _RX_AT_THE_MOMENT.search(j)
        or _RX_AT_PRESENT.search(j)
        or _RX_SO_FAR.search(j)
    )


def _r07_history_of(t, j, canons, ids) -> bool:
    if any(x in _HISTORY_TOKENS for x in t):
        return True
    return bool(
        _RX_USED_TO.search(j)
        or _RX_OVER_TIME.search(j)
        or _RX_IN_THE_PAST.search(j)
    )


def _r08_preference(t, j, canons, ids) -> bool:
    if any(x in _PREFERENCE_TOKENS for x in t):
        return True
    return bool(
        _RX_LIKE_BETTER.search(j)
        or _RX_WOULD_I_ENJOY.search(j)
        or _RX_RECOMMEND_ME.search(j)
    )


def _r09_comparison(t, j, canons, ids) -> bool:
    if any(x in _COMPARISON_TOKENS for x in t):
        return True
    if _RX_WHICH_OF.search(j):
        return True
    # comparative adjective + than/or ("is the train cheaper than flying",
    # "which is better, x or y")
    if ("than" in t or "or" in t) and any(x in _COMPARATIVES for x in t):
        return True
    return False


def _r10_why_causal(t, j, canons, ids) -> bool:
    if any(x in _CAUSAL_TOKENS for x in t):
        return True
    return bool(
        _RX_WHAT_LED.search(j)
        or _RX_HOW_COME.search(j)
        or _RX_WHAT_MADE.search(j)
        or _RX_DUE_TO.search(j)
    )


def _r11_temporal_range(t, j, canons, ids) -> bool:
    """Bounded-span question frame (inserted — see module docstring)."""
    if any(x in _ABSOLUTE_DAYS for x in t):
        return True
    if _RX_THE_OTHER_DAY.search(j):
        return True
    n = len(t)
    for i, x in enumerate(t):
        if x == "between":
            rest = t[i + 1 :]
            if "and" in rest and any(_is_temporal_marker(u) for u in rest):
                return True
        elif x == "from":
            rest = t[i + 1 :]
            if any(u in {"to", "until", "till", "through"} for u in rest) and any(
                _is_temporal_marker(u) for u in rest
            ):
                return True
        elif x in {"during", "in", "on", "at", "over", "throughout", "within"}:
            if any(_is_temporal_marker(u) for u in t[i + 1 : i + 5]):
                return True
        elif x in {"this", "last", "next", "past", "coming"}:
            if any(_is_temporal_marker(u) for u in t[i + 1 : i + 5]):
                return True
        elif x in _PERIOD_WORDS and i + 1 < n and t[i + 1] == "ago":
            return True
    return False


def _r12_open_domain(t, j, canons, ids) -> bool:
    return any(x in _OPEN_MODALS for x in t) and any(
        x in _FIRST_PERSON for x in t
    )


def _r13_abstain_likely(t, j, canons, ids) -> bool:
    # wh-initial questions are content recall, not existence checks —
    # "what did i say about the deadline" is a lookup, not abstain_likely.
    if t and t[0] not in _WH_START and _RX_ABSTAIN_FRAME.search(j):
        return True
    # aux-initial yes/no verification frame: "did i lock the door",
    # "have we paid the invoice", "is there any milk left"
    if (
        len(t) > 2
        and t[0] in _ABSTAIN_AUX
        and t[1] in _ABSTAIN_PERSON
    ):
        rest = t[2:]
        if any(x in _ABSTAIN_MARKERS for x in rest):
            return True
        if rest[0] not in _DISCOURSE_VERBS and any(
            x not in _FUNCTION_TOKENS for x in rest
        ):
            return True
    # no content terms and no identifier -> nothing to retrieve; an honest
    # abstention candidate. (An identifier IS content — its miss is handled
    # by verdict trigger (a), not this class.)
    return not ids and all(x in _FUNCTION_TOKENS for x in t)


def _r14_multi_hop(t, j, canons, ids) -> bool:
    if len(set(canons)) >= 2:
        return True
    # relative-clause chain: >= 2 relative markers (non-initial position),
    # or one marker binding a named entity into a chained clause
    rel = [i for i, x in enumerate(t) if x in _RELATIVE_MARKERS and i > 0]
    return len(rel) >= 2 or (len(rel) >= 1 and len(set(canons)) >= 1)


_RULES = (
    ("R01", IntentClass.IDENTIFIER, _r01_identifier),
    ("R02", IntentClass.TEMPORAL_POINT, _r02_temporal_point),
    ("R03", IntentClass.TEMPORAL_ORDER, _r03_temporal_order),
    ("R04", IntentClass.DURATION, _r04_duration),
    ("R05", IntentClass.COUNT_AGGREGATE, _r05_count_aggregate),
    ("R06", IntentClass.CURRENT_VALUE, _r06_current_value),
    ("R07", IntentClass.HISTORY_OF, _r07_history_of),
    ("R08", IntentClass.PREFERENCE, _r08_preference),
    ("R09", IntentClass.COMPARISON, _r09_comparison),
    ("R10", IntentClass.WHY_CAUSAL, _r10_why_causal),
    ("R11", IntentClass.TEMPORAL_RANGE, _r11_temporal_range),
    ("R12", IntentClass.OPEN_DOMAIN, _r12_open_domain),
    ("R13", IntentClass.ABSTAIN_LIKELY, _r13_abstain_likely),
    ("R14", IntentClass.MULTI_HOP, _r14_multi_hop),
)
# R15 `lookup` is the else-branch: it is appended only when no earlier rule
# fired, so ``classes`` records *detected* classes rather than a constant
# trailing lookup on every query.


# ---------------------------------------------------------------------------
# Public API (frozen — docs/v7_contracts.md)
# ---------------------------------------------------------------------------


def classify(
    norm: NormAnalysis,
    entity_canons: tuple[str, ...] = (),
    identifiers: tuple[NormTerm, ...] = (),
) -> IntentResult:
    """Classify a normalized query into the §32.14 intent classes.

    Rules evaluate in fixed order; the first match is ``primary`` and every
    match is recorded in ``classes``/``rule_trace`` (``R##.<class>`` ids).
    ``window`` is left ``None`` — window attachment happens downstream in the
    pipeline via ``temporal/v2`` once ``query_time_us`` is known (V7-09.05).
    """
    terms = _lex_terms(norm)
    toks = [t.term for t in terms]
    joined = _joined(toks)
    ids: tuple[NormTerm, ...] = tuple(identifiers) or tuple(norm.identifiers)

    classes: list[IntentClass] = []
    trace: list[str] = []
    for rule_id, klass, fn in _RULES:
        if fn(toks, joined, entity_canons, ids):
            classes.append(klass)
            trace.append(f"{rule_id}.{klass.value}")
    if not classes:
        classes.append(IntentClass.LOOKUP)
        trace.append(f"R15.{IntentClass.LOOKUP.value}")

    primary = classes[0]
    return IntentResult(
        primary=primary,
        classes=tuple(classes),
        window=None,
        rule_trace=tuple(trace),
    )


def _strip_function_edges(terms: tuple[NormTerm, ...]) -> tuple[NormTerm, ...]:
    """Remove leading/trailing function (scaffold) tokens from a segment."""
    start = 0
    end = len(terms)
    while start < end and terms[start].term in _FUNCTION_TOKENS:
        start += 1
    while end > start and terms[end - 1].term in _FUNCTION_TOKENS:
        end -= 1
    return terms[start:end]


def _facet_analysis(norm: NormAnalysis, terms: tuple[NormTerm, ...]) -> NormAnalysis:
    return NormAnalysis(
        analyzer_id=norm.analyzer_id,
        terms=tuple(terms),
        identifiers=tuple(t for t in terms if t.channel == "identifier"),
        text=" ".join(t.term for t in terms),
    )


def decompose(
    norm: NormAnalysis,
    intent: IntentResult,
    *,
    max_facets: int = MAX_FACETS,
) -> tuple[NormAnalysis, ...]:
    """Deterministic facet sub-queries for ``multi_hop``/``comparison``
    primaries (V7-05.13): split on conjunctions (comparison: and/or/versus/
    vs/than) and relative-clause markers (multi_hop: and/or/that/who/whom/
    which/whose), strip question scaffold from segment edges, and for
    comparisons hoist a trailing preposition phrase ("for a honeymoon") into
    every facet as shared context.

    Returns ``(norm,)`` when the primary is not decomposable or no split is
    possible — callers may always iterate the result.
    """
    if intent.primary not in (IntentClass.MULTI_HOP, IntentClass.COMPARISON):
        return (norm,)

    terms = _lex_terms(norm)
    splits = (
        _COMPARISON_SPLITS
        if intent.primary == IntentClass.COMPARISON
        else _MULTIHOP_SPLITS
    )
    cut = [
        i
        for i, t in enumerate(terms)
        if t.term in splits and 0 < i < len(terms) - 1
    ]
    if not cut:
        return (norm,)

    segments: list[tuple[NormTerm, ...]] = []
    prev = 0
    for i in cut:
        segments.append(tuple(terms[prev:i]))
        prev = i
    segments.append(tuple(terms[prev:]))

    # Comparison shared-context tail: "compare X and Y for Z" -> the
    # preposition phrase after the last side belongs to every facet.
    tail: tuple[NormTerm, ...] = ()
    if intent.primary == IntentClass.COMPARISON and len(segments) >= 2:
        last = segments[-1]
        tail_cut: Optional[int] = next(
            (
                j
                for j, t in enumerate(last)
                if t.term in _TAIL_PREPOSITIONS and j > 0
            ),
            None,
        )
        if tail_cut is not None:
            segments[-1] = last[:tail_cut]
            tail = _strip_function_edges(last[tail_cut:])

    facets: list[tuple[NormTerm, ...]] = []
    for seg in segments:
        core = _strip_function_edges(seg)
        if tail:
            core = core + tail
        if core:
            facets.append(core)

    # dedupe identical facet term sequences, preserve order
    seen: set[tuple[str, ...]] = set()
    unique: list[tuple[NormTerm, ...]] = []
    for f in facets:
        key = tuple(t.term for t in f)
        if key and key not in seen:
            seen.add(key)
            unique.append(f)

    if len(unique) < 2:
        return (norm,)
    return tuple(
        _facet_analysis(norm, f) for f in unique[: max(1, max_facets)]
    )


# ---------------------------------------------------------------------------
# Structural facet decomposition (V8-10.01)
# ---------------------------------------------------------------------------
#
# Coordination is detected on the resolved-canon layout, not on the intent
# class: ``decompose`` only fires for ``multi_hop``/``comparison`` primaries,
# so a coordinated-subject question classified ``lookup`` ("How did Melanie
# and Caroline each spend the summer?") used to ship zero facets (D8-15).
# The structural path fires whenever >= 2 resolved canons (entity or
# speaker — the caller supplies the merged, ordered set) are joined by a
# coordinator, independent of ``intent.primary``.

#: Owned, versioned coordination lexicon id for coverage/explain.
COORD_LEXICON_ID = "coord_lex/v1"

#: Binary coordinators: join the canon spans on either side ("X and Y",
#: "X or Y", "X vs Y", "X as well as Y", "neither X nor Y"). ``nor``
#: completes the correlative ``neither … nor`` — the spec's owned list
#: names ``neither``; its pair token is required for the second conjunct
#: to bind.
_COORD_JOIN = frozenset({"and", "or", "nor", "vs", "versus"})
#: Distributive/collective markers: bound the contiguous canon cluster
#: they touch as coordinated participants ("both X and Y", "X and Y
#: each", "either X or Y", "between X and Y").
_COORD_MARK = frozenset({"both", "each", "either", "neither", "between"})
#: The multi-token coordinator "as well as" (single join site).
_AS_WELL_AS = ("as", "well", "as")

#: Owned function-word stoplist for the V8.5 real-canon gate
#: (V85-05.05): a canon that *is* a closed-class scaffold token can
#: never seed structural decomposition — junk canon-like resolutions
#: ("of", "both", "each") firing decomposition produced the measured
#: c4 multi-hop all@10 regression. The list is the union of this
#: module's owned closed-class vocabularies (function tokens +
#: coordinator joins/marks + the "as well as" pieces) — fixed code,
#: not a policy arm.
FUNC_WORD_STOPLIST: frozenset = frozenset(
    _FUNCTION_TOKENS | _COORD_JOIN | _COORD_MARK | {"well"}
)


def _canon_occurrences(toks: list[str], canon: str) -> list[tuple[int, int]]:
    """Every ``(start, end)`` term-window where ``canon``'s folded token
    sequence appears in the query term stream."""
    ct = str(canon).split()
    n = len(ct)
    if not n:
        return []
    return [
        (i, i + n)
        for i in range(len(toks) - n + 1)
        if toks[i : i + n] == ct
    ]


def _drop_overlapping_spans(
    spans: dict[str, list[tuple[int, int]]],
) -> dict[str, list[tuple[int, int]]]:
    """Keep non-overlapping canon occurrences — longest span first, then
    leftmost (the same greedy shape ``extract_query_entities`` applies).

    Entity- and speaker-channel resolutions can overlap ("melanie park"
    entity vs "melanie" speaker); the longer canon owns the overlap."""
    order = sorted(
        (
            (c, s, e)
            for c, occs in spans.items()
            for s, e in occs
        ),
        key=lambda x: (-(x[2] - x[1]), x[1], x[0]),
    )
    taken: list[tuple[int, int]] = []
    accepted: dict[str, list[tuple[int, int]]] = {}
    for c, s, e in order:
        if any(s < te and e > ts for ts, te in taken):
            continue
        taken.append((s, e))
        accepted.setdefault(c, []).append((s, e))
    return accepted


def _coord_sites(
    toks: list[str],
    spans: dict[str, list[tuple[int, int]]],
) -> list[tuple[int, int, str]]:
    """Coordinator sites as ``(start, end, kind)`` term ranges.

    ``kind`` is ``"join"`` (and/or/nor/vs/versus/as well as — binds the
    spans on both sides) or ``"mark"`` (both/each/either/neither/between —
    bounds a contiguous canon cluster). A coordinator token *inside* a
    resolved canon span is part of the name ("rock and roll"), never a
    site.
    """
    covered: set[int] = set()
    for occs in spans.values():
        for s, e in occs:
            covered.update(range(s, e))
    sites: list[tuple[int, int, str]] = []
    i = 0
    n = len(toks)
    while i < n:
        if i in covered:
            i += 1
            continue
        if (
            toks[i : i + 3] == list(_AS_WELL_AS)
            and all(j not in covered for j in (i, i + 1, i + 2))
        ):
            sites.append((i, i + 3, "join"))
            i += 3
            continue
        if toks[i] in _COORD_JOIN:
            sites.append((i, i + 1, "join"))
        elif toks[i] in _COORD_MARK:
            sites.append((i, i + 1, "mark"))
        i += 1
    return sites


def _cluster(
    pos2canon: dict[int, tuple[str, int, int]],
    p: int,
    step: int,
) -> list[tuple[str, int, int]]:
    """Contiguous canon occurrences walking ``step`` (-1 left / +1 right)
    from position ``p`` — the spans a coordinator site binds on one side."""
    out: list[tuple[str, int, int]] = []
    while p in pos2canon:
        canon, s, e = pos2canon[p]
        out.append((canon, s, e))
        p = (s - 1) if step < 0 else e
    return out


def _surface_bytes(surface: object) -> Optional[bytes]:
    """The raw query surface as UTF-8 bytes — ``NormTerm.byte_start``/
    ``byte_end`` index into this encoding. ``None`` when no usable
    surface was supplied (the capitalization check then cannot prove a
    canon real — fail-closed, V85-05.05)."""
    if isinstance(surface, str):
        try:
            return surface.encode("utf-8", "strict")
        except UnicodeEncodeError:
            return None
    if isinstance(surface, (bytes, bytearray, memoryview)):
        return bytes(surface)
    return None


def _capitalized_fragment(frag: str) -> bool:
    """The first *cased* character is uppercase. Digits, punctuation and
    uncased script carry no capitalization evidence → ``False``."""
    for ch in frag:
        if ch.isupper():
            return True
        if ch.islower():
            return False
    return False


def _occurrence_capitalized(
    terms: list,
    span: tuple[int, int],
    surface: Optional[bytes],
) -> bool:
    """The occurrence's raw surface slice begins with a capital letter.

    Fail-closed on every unverifiable edge: no surface, out-of-range or
    non-int byte offsets, invalid UTF-8 — an occurrence whose
    capitalization cannot be proven is not a real canon (V85-05.05).
    """
    if surface is None:
        return False
    s, e = span
    if not (0 <= s < e <= len(terms)):
        return False
    bs = terms[s].byte_start
    be = terms[e - 1].byte_end
    if not isinstance(bs, int) or not isinstance(be, int):
        return False
    if bs < 0 or be < bs or be > len(surface):
        return False
    try:
        frag = surface[bs:be].decode("utf-8", "strict")
    except UnicodeDecodeError:
        return False
    return _capitalized_fragment(frag)


def decompose_structural(
    norm: NormAnalysis,
    canons: tuple[str, ...] = (),
    *,
    max_facets: int = MAX_FACETS,
    surface: object = None,
) -> tuple[tuple[NormAnalysis, str], ...]:
    """Canon-bound facet sub-queries for coordinated subjects (V8-10.01).

    Fires when >= 2 resolved canons (entity or speaker — merged upstream)
    are joined by a coordinator from the owned ``coord_lex/v1`` list
    (and, or, nor, vs/versus, as well as — joins; both, each, either,
    neither, between — cluster markers). Each facet is the query
    restricted to one canon: that canon is kept (and returned as the
    facet's entity seed — the caller pins it on the sub-view), every
    *other* joined canon's occurrences are removed from the terms, and
    the coordinator sites inside the coordination zone are dropped so no
    dangling "and"/"each" remains. Decomposition is intent-independent
    and one level deep (the caller builds sub-views with
    ``allow_facets=False``).

    V85-05.05 real-canon gate: a canon seeds only through occurrences
    that are *real* — the canon string is not on the owned
    :data:`FUNC_WORD_STOPLIST` and the occurrence's raw ``surface``
    slice begins with a capital letter (the caller passes the original
    query text; ``NormTerm`` byte offsets index into it). Without a
    surface the capitalization half cannot be proven, so the gate stays
    closed — a lowercase or unverifiable mention is the common word,
    never the entity.

    Returns ``((facet_norm, seed_canon), ...)`` in query order, or ``()``
    when no coordination binds >= 2 resolved canons.
    """
    terms = _lex_terms(norm)
    toks = [t.term for t in terms]
    surf = _surface_bytes(surface)
    raw: dict[str, list[tuple[int, int]]] = {}
    for c in dict.fromkeys(canons or ()):
        canon = str(c)
        # (a) a function-word canon is never real.
        if canon in FUNC_WORD_STOPLIST:
            continue
        # (b) only occurrences capitalized in the raw surface count.
        occs = [
            occ
            for occ in _canon_occurrences(toks, canon)
            if _occurrence_capitalized(terms, occ, surf)
        ]
        if occs:
            raw[canon] = occs
    if len(raw) < 2:
        return ()
    spans = _drop_overlapping_spans(raw)
    if len(spans) < 2:
        return ()

    sites = _coord_sites(toks, spans)
    if not sites:
        return ()
    pos2canon: dict[int, tuple[str, int, int]] = {}
    for c, occs in spans.items():
        for s, e in occs:
            for p in range(s, e):
                pos2canon[p] = (c, s, e)

    # Joined occurrences: every canon occurrence bound to a coordinator
    # site (joins bind both sides; markers bind the touching cluster).
    joined: list[tuple[str, int, int]] = []
    for s, e, _kind in sites:
        for hit in _cluster(pos2canon, s - 1, -1):
            if hit not in joined:
                joined.append(hit)
        for hit in _cluster(pos2canon, e, +1):
            if hit not in joined:
                joined.append(hit)
    if not joined:
        return ()

    # Facet order follows the coordination order in the query (rollover
    # is "in facet order" downstream); joined canons dedupe by their
    # earliest bound occurrence.
    joined.sort(key=lambda x: (x[1], x[0]))
    seeds = list(dict.fromkeys(c for c, _s, _e in joined))
    if len(seeds) < 2:
        return ()

    # Coordination zone: first-to-last joined occurrence, widened across
    # cluster markers touching its edges ("both X and Y", "X and Y each",
    # "between X and Y"). Sites inside the zone are removed from every
    # facet — their joining job is done.
    zs = min(s for _c, s, _e in joined)
    ze = max(e for _c, _s, e in joined)
    moved = True
    while moved:
        moved = False
        for s, e, kind in sites:
            if kind != "mark":
                continue
            if e == zs:
                zs = s
                moved = True
            if s == ze:
                ze = e
                moved = True
    zone_sites = [
        (s, e) for s, e, _k in sites if s >= zs and e <= ze
    ]

    facets: list[tuple[NormTerm, ...]] = []
    facet_seeds: list[str] = []
    for seed in seeds[: max(1, int(max_facets))]:
        remove: set[int] = set()
        # only the *other joined* canons leave the facet — an
        # uncoordinated resolved canon ("what did X and Y tell Z") is
        # shared predicate context, so it stays in every facet.
        for c in seeds:
            if c == seed:
                continue
            for s, e in spans.get(c, ()):
                remove.update(range(s, e))
        for s, e in zone_sites:
            remove.update(range(s, e))
        core = _strip_function_edges(
            tuple(t for i, t in enumerate(terms) if i not in remove)
        )
        if core:
            facets.append(core)
            facet_seeds.append(seed)

    # dedupe identical facet term sequences, preserve order
    seen: set[tuple[str, ...]] = set()
    unique: list[tuple[NormAnalysis, str]] = []
    for core, seed in zip(facets, facet_seeds):
        key = tuple(t.term for t in core)
        if key and key not in seen:
            seen.add(key)
            unique.append((_facet_analysis(norm, core), seed))

    if len(unique) < 2:
        return ()
    return tuple(unique[: max(1, int(max_facets))])


__all__ = [
    "INTENT_ID",
    "LEXICON_ID",
    "FORMULA_STATUS",
    "MAX_FACETS",
    "COORD_LEXICON_ID",
    "FUNC_WORD_STOPLIST",
    "classify",
    "decompose",
    "decompose_structural",
]
