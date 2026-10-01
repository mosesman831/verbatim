"""Typed claim proposals: deterministic, conservative slot mapping (SPEC §12).

Structure is OPTIONAL — a proposal may keep ``predicate=None`` and remain a
searchable verbatim quotation; structure is an interpretation, never a
prerequisite for evidence retrieval. Every emitted object points at exact
evidence byte offsets, never generated prose.

Conservative bias throughout: when a pattern does not clearly apply, the
proposal stays unstructured (``predicate=None``) rather than being forced
into the nearest supported slot.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .harvest import (
    CONDITION_CUES,
    MODALITY_CUES,
    NEGATION_RE,
    TIME_CUE_RE,
)
from .time import parse_time_expression
from .types import (
    ClaimProposal,
    Condition,
    InterpretationStatus,
    Modality,
    Polarity,
    SourceEnvelope,
    SpanRef,
    TimeInterval,
    Precision,
)

PROPOSER_VERSION = "propose-1"

#: Method tag for explicit "remember …" requests: they bypass the pattern
#: table (SPEC §13) but stay grounded to the span — no invented structure.
EXPLICIT_REMEMBER = "explicit_remember"

#: Time-cue detector kept under its historical private name; the canonical
#: compiled pattern now lives with the other versioned harvest regexes.
_TIME_CUE_RE = TIME_CUE_RE

# ---------------------------------------------------------------------------
# Slot registry (SPEC §12)
# ---------------------------------------------------------------------------
#: Supported predicates → {multi_valued, mutable, sensitive}.
#:
#: - ``multi_valued``: set-valued slot; a new value does NOT compete with
#:   existing ones ("adding one hobby does not remove other hobbies").
#: - ``mutable``: values may legitimately change over time (supersession
#:   candidate); immutable values use correction semantics instead.
#: - ``sensitive``: personal-data category that defaults to review-gated
#:   capture (SPEC §13: sensitive categories get no automatic persistence).
SLOT_REGISTRY: dict[str, dict[str, bool]] = {
    "editor": {"multi_valued": False, "mutable": True, "sensitive": False},
    "ide": {"multi_valued": False, "mutable": True, "sensitive": False},
    "project_database": {"multi_valued": False, "mutable": True, "sensitive": False},
    "residence": {"multi_valued": False, "mutable": True, "sensitive": True},
    "preference": {"multi_valued": True, "mutable": True, "sensitive": False},
    "schedule": {"multi_valued": True, "mutable": True, "sensitive": False},
    "language": {"multi_valued": True, "mutable": False, "sensitive": False},
    "os": {"multi_valued": False, "mutable": True, "sensitive": False},
    "tool_use": {"multi_valued": True, "mutable": True, "sensitive": False},
}

_LITERAL_DEFAULTS = {"multi_valued": False, "mutable": True, "sensitive": False}
_LITERAL_KEY_RE = re.compile(r"^literal:[a-z][a-z0-9_]{0,39}$")


def slot_for(predicate: Optional[str]) -> Optional[dict[str, bool]]:
    """Registry lookup; ``literal:<key>`` is a generic opt-in slot.

    Unknown predicates return ``None`` — callers must keep them as searchable
    quotations, not coerce them into a near-enough slot (SPEC §12).
    """
    if predicate is None:
        return None
    if predicate in SLOT_REGISTRY:
        return SLOT_REGISTRY[predicate]
    if _LITERAL_KEY_RE.match(predicate):
        return _LITERAL_DEFAULTS
    return None


# ---------------------------------------------------------------------------
# Built-in predicate registry (SPEC_V2 §13.09, §16)
# ---------------------------------------------------------------------------

#: Namespace for all predicates this engine ships. Third parties MUST NOT
#: register into it (SPEC_V2 §16.05).
BUILTIN_REGISTRY_NAMESPACE = "vb"

#: Bump when entries change; written to ``claim_revisions.registry_version``
#: so the rule set that produced a claim is auditable (SPEC_V2 §16.02).
BUILTIN_REGISTRY_VERSION = 1

#: (name, cardinality, mutable, sensitive) — the durable predicate table
#: seeded into ``predicate_definitions`` on first use. Cardinality mirrors
#: ``SLOT_REGISTRY`` ("single"/"set"); the static slot table remains the
#: in-process fallback when no store is consulted.
BUILTIN_PREDICATES: tuple[tuple[str, str, bool, bool], ...] = (
    ("editor", "single", True, False),
    ("ide", "single", True, False),
    ("project_database", "single", True, False),
    ("residence", "single", True, True),
    ("preference", "set", True, False),
    ("schedule", "set", True, False),
    ("language", "set", False, False),
    ("os", "single", True, False),
    ("tool_use", "set", True, False),
    # The generic opt-in family for "my <key> is <value>" statements —
    # registered so literal:<key> claims also consult one durable table.
    ("literal", "single", True, False),
)

#: Registry name shared by every ``literal:<key>`` predicate.
LITERAL_PREDICATE = "literal"


# ---------------------------------------------------------------------------
# Deterministic proposal patterns (propose-1)
# ---------------------------------------------------------------------------

#: Known editor/IDE names. Only "I use X" maps to the single-valued ``editor``
#: slot when X is a recognized editor — a generic tool stays ``tool_use``
#: (set-valued) because "I use Docker" is not an editor statement. Deliberately
#: small; misses degrade to ``tool_use``, never to a wrong slot.
_EDITOR_NAMES = frozenset(
    {
        "neovim", "nvim", "vim", "emacs", "spacemacs", "doom emacs",
        "vscode", "vs code", "visual studio code", "visual studio",
        "helix", "hx", "sublime", "sublime text", "intellij",
        "intellij idea", "pycharm", "webstorm", "goland", "clion",
        "rider", "rubymine", "phpstorm", "datagrip", "dataspell",
        "android studio", "xcode", "atom", "nano", "notepad++",
        "zed", "cursor", "eclipse", "fleet", "kate", "gedit",
        "lapce", "nova", "bbedit", "textmate", "micro", "kakoune",
        "rstudio", "netbeans",
    }
)

_OS_NAMES = frozenset(
    {
        "linux",
        "arch",
        "arch linux",
        "ubuntu",
        "debian",
        "fedora",
        "macos",
        "mac os",
        "mac os x",
        "osx",
        "windows",
        "windows 10",
        "windows 11",
        "nixos",
        "gentoo",
        "freebsd",
        "openbsd",
        "opensuse",
        "opensuse tumbleweed",
        "manjaro",
        "pop os",
        "pop!_os",
        "centos",
        "alpine",
        "rhel",
        "void",
        "void linux",
    }
)

#: Objects that carry no referent — proposing a slot value for "it" would be
#: fabrication, so these produce an unstructured quotation (SPEC §11).
_VAGUE_OBJECTS = frozenset(
    {"it", "this", "that", "them", "they", "something", "stuff", "things", "one", "some"}
)

#: "I work at/for X" values that are work *arrangements*, not employers.
_NON_EMPLOYER = frozenset({"home", "remote", "remotely", "from home", "freelance"})

#: Keys that must never become a ``literal:`` slot — a structure-free span is
#: safer for likely secrets (the sensitive hint still routes it to review).
_SENSITIVE_LITERAL_KEYS = frozenset(
    {"password", "passwd", "ssn", "secret", "token", "api_key", "api key", "pin", "credit_card"}
)

_REMEMBER_RE = re.compile(
    r"^\s*(?:please\s+)?remember\b(?:\s+that\b|\s*:\s*|\s+this\b)?", re.IGNORECASE
)

# Shared object-boundary lookahead: a captured object ends at a clause cue,
# a time cue, terminal punctuation, or end of text — so "to Helix yesterday"
# never swallows the date into the object.
_TAIL = (
    r"(?=\s+(?:for|at|when|while|unless|during|if|on|but|and|because"
    r"|every|each|yesterday|today|tomorrow|tonight"
    r"|last\s+(?:week|month|year|night))\b"
    r"|[,.;!?。！？]|$)"
)

_USE_RE = re.compile(
    r"\bI\s+(?:also\s+|still\s+|mainly\s+|mostly\s+|now\s+|currently\s+"
    r"|typically\s+|usually\s+|always\s+|often\s+|sometimes\s+|generally\s+"
    r"|never\s+|rarely\s+|no\s+longer\s+|used\s+to\s+)*use\s+"
    r"(.+?)" + _TAIL,
    re.IGNORECASE,
)

_SWITCH_RE = re.compile(
    r"\bI\s+(?:just\s+|recently\s+|finally\s+)*"
    r"(?:switched|moved|changed|migrated|upgraded)\s+"
    r"(?:over\s+)?(?:from\s+\S+(?:\s+\S+){0,3}?\s+to\s+|to\s+)(.+?)" + _TAIL,
    re.IGNORECASE,
)

#: "I switched my personal projects from Neovim to Helix" — object is the
#: post-"to" value; the intervening text stays part of the quotation.
_SWITCH_FROM_TO_RE = re.compile(
    r"\bI\s+(?:just\s+|recently\s+|finally\s+)*"
    r"(?:switched|moved|changed|migrated|upgraded)\s+"
    r".*?\bfrom\s+.+?\s+to\s+(.+?)" + _TAIL,
    re.IGNORECASE,
)

_RETURN_TO_EDITOR_RE = re.compile(
    r"(?:\bI\s+)?(?:went|switched|moved|changed)\s+back\s+to\s+(.+?)" + _TAIL,
    re.IGNORECASE,
)

_RESIDENCE_RE = re.compile(
    r"\bI\s+(?:'ve\s+|have\s+|just\s+|recently\s+|currently\s+)*"
    r"(live\s+in|live\s+at|moved\s+to|relocated\s+to|'m\s+based\s+in|am\s+based\s+in)"
    r"\s+(.+?)" + _TAIL,
    re.IGNORECASE,
)

_DB_TEAM_RE = re.compile(
    r"\b(?:we|our\s+team|my\s+team|the\s+team)\s+"
    r"(?:use|uses|are\s+using|switched\s+to|migrated\s+to|moved\s+to|chose|picked|standardized\s+on)\s+"
    r"(.+?)\s+(?:as\s+|for\s+)(?:the\s+|our\s+)?(?:project\s+)?database\b",
    re.IGNORECASE,
)
_DB_IS_RE = re.compile(
    r"\b(?:our|the|my)\s+(?:project\s+)?database\s+is\s+(.+?)" + _TAIL,
    re.IGNORECASE,
)
_DB_MIGRATE_RE = re.compile(
    r"\bwe\s+(?:just\s+|recently\s+)?migrated\s+(?:the\s+|our\s+)?database\s+to\s+(.+?)" + _TAIL,
    re.IGNORECASE,
)

_EMPLOYER_RE = re.compile(
    r"\bI\s+work\s+(?:at|for)\s+(.+?)" + _TAIL,
    re.IGNORECASE,
)

_PREFER_RE = re.compile(
    r"\bI\s+(?:really\s+|much\s+)?prefer\s+(.+?)" + _TAIL,
    re.IGNORECASE,
)

#: "I (don't) like/love/enjoy/hate/dislike/can't stand X" — a preference or
#: aversion statement (SPEC_V2 §13.11). The captured verb decides whether the
#: proposition is affirmative or negated *independent* of cue words, so
#: "I hate cilantro" and "I don't like cilantro" share one polarity and can
#: be recognized as equivalent rather than contradictory.
_LIKE_RE = re.compile(
    r"\bI\s+(?:really\s+|much\s+|truly\s+|do\s+not\s+|don'?t\s+|never\s+)*"
    r"(like|love|enjoy|hate|dislike|can'?t\s+stand|cannot\s+stand|cant\s+stand)"
    r"\s+(.+?)" + _TAIL,
    re.IGNORECASE,
)

#: Verbs that are themselves negations — their affirmative surface form still
#: produces a negated preference proposition.
_NEGATIVE_LIKE_VERBS = frozenset(
    {"hate", "dislike", "can't stand", "cant stand", "cannot stand"}
)

_SPEAK_RE = re.compile(r"\bI\s+speak\s+(.+?)" + _TAIL, re.IGNORECASE)

_RUN_OS_RE = re.compile(
    r"\bI\s+(?:also\s+|still\s+|used\s+to\s+)?run\s+(.+?)" + _TAIL,
    re.IGNORECASE,
)

# Ordered "my X is Y" patterns; the generic literal fallback is last.
_MY_SLOT_RES: tuple[tuple["re.Pattern[str]", str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), predicate)
    for pattern, predicate in (
        (
            r"\bmy\s+(?:main\s+|primary\s+|default\s+|preferred\s+)?"
            r"(?:editor|text\s+editor|code\s+editor)\s+is\s+(.+?)" + _TAIL,
            "editor",
        ),
        (
            r"\bmy\s+(?:main\s+|primary\s+|default\s+)?ide\s+is\s+(.+?)" + _TAIL,
            "ide",
        ),
        (
            r"\bmy\s+(?:main\s+|primary\s+|default\s+)?"
            r"(?:os|operating\s+system)\s+is\s+(.+?)" + _TAIL,
            "os",
        ),
        (
            r"\bmy\s+(?:native\s+|preferred\s+|primary\s+)?language\s+is\s+(.+?)" + _TAIL,
            "language",
        ),
        (
            r"\bmy\s+(?:work\s+|daily\s+)?(?:schedule|hours|routine)\s+(?:is|are)\s+(.+?)" + _TAIL,
            "schedule",
        ),
        (
            r"\bmy\s+(?:home|house|apartment|flat)\s+is\s+in\s+(.+?)" + _TAIL,
            "residence",
        ),
    )
)

_MY_FAVORITE_RE = re.compile(
    r"\bmy\s+(?:favou?rite|preferred|beloved)\s+(.+?)\s+is\s+(.+?)" + _TAIL,
    re.IGNORECASE,
)

_MY_LITERAL_RE = re.compile(
    r"\bmy\s+([a-z][a-z _-]{0,29})\s+is\s+(.+?)" + _TAIL,
    re.IGNORECASE,
)

#: Speech-attribution verbs; combined with an inside-quotes test they block
#: reported speech ("Sam said 'I live in Bristol'") from becoming a personal
#: slot claim (SPEC §5, §11).
_SPEECH_VERB_RE = re.compile(
    r"\b(?:said|says|say|told|asked|wrote|claims?|mentioned|reported|thinks?)\b",
    re.IGNORECASE,
)

_CHANGE_VERB_RE = re.compile(
    r"\b(?:switched|moved|changed|migrated|relocated|upgraded|updated)\b"
    r"|\bwent\s+back\b|\bnow\b|\brecently\b",
    re.IGNORECASE,
)


def _clean_object(raw: str) -> Optional[str]:
    """Trim a captured object; reject empty, oversized, or referent-free text."""
    obj = raw.strip().strip("\"'“”‘’").strip()
    obj = obj.rstrip(".,;:!?。！？)]}»”’").strip()
    if not obj or len(obj) > 120:
        return None
    if obj.lower() in _VAGUE_OBJECTS:
        return None
    return obj


def _inside_quotes(text: str, pos: int) -> bool:
    """Whether ``pos`` sits inside quote marks (heuristic: odd count before)."""
    before = text[:pos]
    if before.count('"') % 2 == 1:
        return True
    if (before.count("“") - before.count("”")) % 2 == 1:
        return True
    if (before.count("‘") - before.count("’")) % 2 == 1:
        return True
    # Apostrophes in contractions make "'" noisy; require the speech-verb
    # check downstream to fire before this ever suppresses a claim.
    return before.count("'") % 2 == 1


def _is_reported_speech(text: str, match_pos: int) -> bool:
    """Match is inside quotation marks AND a speech-attribution verb precedes.

    Both conditions are required: quoting oneself ("I said I use vim" has no
    quote marks) or quoting without attribution stays eligible — but any
    ambiguity suppresses structure, never the underlying evidence.
    """
    return _inside_quotes(text, match_pos) and bool(
        _SPEECH_VERB_RE.search(text[:match_pos])
    )


def _extract_condition(text: str) -> Optional[Condition]:
    """Map detected condition clauses to ``eq`` leaves on key 'context'.

    Multiple distinct clauses combine under an ``all`` node; cues without a
    bounded mapping ("unless", "if I", bare "during") yield ``None`` — an
    unparseable qualifier must not silently become unconditional (SPEC §12).
    """
    leaves: list[Condition] = []
    seen: set[str] = set()
    for rx, value in CONDITION_CUES:
        if value not in seen and rx.search(text):
            seen.add(value)
            leaves.append(Condition("eq", key="context", value=value))
    if not leaves:
        return None
    if len(leaves) == 1:
        return leaves[0]
    return Condition("all", children=tuple(leaves))


def _extract_modality(text: str) -> Modality:
    best: Optional[tuple[int, str]] = None
    for rx, value in MODALITY_CUES:
        m = rx.search(text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), value)
    if best is None:
        return Modality.ASSERTED
    return {
        "hypothetical": Modality.HYPOTHETICAL,
        "habitual": Modality.HABITUAL,
        "uncertain": Modality.UNCERTAIN,
    }[best[1]]


def _extract_valid(text: str, event_us: int) -> TimeInterval:
    """Valid-time interpretation (SPEC §14).

    - Explicit date cues are parsed with :func:`parse_time_expression` using
      the source event time as reference — never the worker's clock.
    - Change verbs ("switched", "moved", "now") yield an
      ``asserted_current_at`` basis: the new state applies from the stated
      date if one parsed, else from the event time, open-ended.
    - Anything else stays explicitly unknown; unknown valid time is not
      treated as negative infinity or definitive current truth.
    """
    parsed: Optional[TimeInterval] = None
    for m in _TIME_CUE_RE.finditer(text):
        parsed = parse_time_expression(m.group(0), event_us)
        if parsed is not None:
            break
    if _CHANGE_VERB_RE.search(text):
        if parsed is not None and parsed.from_us is not None:
            return TimeInterval(
                from_us=parsed.from_us,
                until_us=None,
                precision=parsed.precision,
                timezone=parsed.timezone,
                basis="asserted_current_at",
            )
        return TimeInterval(
            from_us=event_us,
            until_us=None,
            precision=Precision.INSTANT,
            basis="asserted_current_at",
        )
    if parsed is not None:
        return parsed
    return TimeInterval()


def _match_structure(
    text: str,
) -> Optional[tuple[str, str, int, Optional[str], Optional[Polarity]]]:
    """First matching structural pattern → (predicate, object, pos, topic, polarity).

    Patterns are tried in a fixed order — most specific first — so the result
    is deterministic. ``topic`` is set only for "my favourite X is Y".
    ``polarity`` is ``None`` unless the pattern's own semantics fix the
    proposition's polarity (e.g. "I hate X" — the surface carries no negation
    cue, but the proposition is a negated preference). Returns ``None`` for
    no match; the caller then keeps the proposal unstructured.
    """
    m = _USE_RE.search(text)
    if m:
        obj = _clean_object(m.group(1))
        if obj:
            pred = "editor" if obj.lower() in _EDITOR_NAMES else "tool_use"
            return pred, obj, m.start(), None, None
    m = _RESIDENCE_RE.search(text)
    if m:
        obj = _clean_object(m.group(2))
        if obj:
            return "residence", obj, m.start(), None, None
    m = _RETURN_TO_EDITOR_RE.search(text)
    if m:
        obj = _clean_object(m.group(1))
        if obj and obj.lower() in _EDITOR_NAMES:
            return "editor", obj, m.start(), None, None
    for rx in (_SWITCH_RE, _SWITCH_FROM_TO_RE):
        m = rx.search(text)
        if m:
            obj = _clean_object(m.group(1))
            if obj:
                pred = "editor" if obj.lower() in _EDITOR_NAMES else "tool_use"
                return pred, obj, m.start(), None, None
    for rx in (_DB_TEAM_RE, _DB_IS_RE, _DB_MIGRATE_RE):
        m = rx.search(text)
        if m:
            obj = _clean_object(m.group(1))
            if obj:
                return "project_database", obj, m.start(), None, None
    m = _EMPLOYER_RE.search(text)
    if m:
        obj = _clean_object(m.group(1))
        if obj and obj.lower() not in _NON_EMPLOYER:
            return "literal:employer", obj, m.start(), None, None
    m = _PREFER_RE.search(text)
    if m:
        obj = _clean_object(m.group(1))
        if obj:
            return "preference", obj, m.start(), None, None
    m = _LIKE_RE.search(text)
    if m:
        verb = m.group(1).lower()
        obj = _clean_object(m.group(2))
        if obj:
            # Only an inherently negative verb overrides the cue check:
            # "I hate X" has no negation cue but is a negated preference,
            # while "I don't like X" is caught by NEGATION_RE downstream.
            polarity = (
                Polarity.NEGATED if verb in _NEGATIVE_LIKE_VERBS else None
            )
            return "preference", obj, m.start(), None, polarity
    m = _SPEAK_RE.search(text)
    if m:
        obj = _clean_object(m.group(1))
        if obj:
            return "language", obj, m.start(), None, None
    m = _RUN_OS_RE.search(text)
    if m:
        obj = _clean_object(m.group(1))
        if obj and obj.lower() in _OS_NAMES:
            return "os", obj, m.start(), None, None
    for rx, predicate in _MY_SLOT_RES:
        m = rx.search(text)
        if m:
            obj = _clean_object(m.group(1))
            if obj:
                return predicate, obj, m.start(), None, None
    m = _MY_FAVORITE_RE.search(text)
    if m:
        topic = _clean_object(m.group(1))
        obj = _clean_object(m.group(2))
        if obj and topic:
            return "preference", obj, m.start(), topic, None
    m = _MY_LITERAL_RE.search(text)
    if m:
        key = m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        obj = _clean_object(m.group(2))
        if obj and key not in _SENSITIVE_LITERAL_KEYS and len(key) <= 40:
            return f"literal:{key}", obj, m.start(), None, None
    return None


def _object_json(text: str, obj: str, span: SpanRef, envelope: SourceEnvelope, topic: Optional[str] = None) -> dict[str, Any]:
    """Structured object pointing at exact evidence offsets (SPEC §12).

    ``byte_start``/``byte_end`` locate the object substring inside the source
    payload when it can be found verbatim; they are ``None`` otherwise — never
    a fabricated location.
    """
    out: dict[str, Any] = {"kind": "literal", "text": obj}
    if topic:
        out["topic"] = topic
    try:
        region = envelope.payload[span.start_byte : span.end_byte]
        idx = region.find(obj.encode("utf-8"))
    except Exception:
        idx = -1
    if idx >= 0:
        out["byte_start"] = span.start_byte + idx
        out["byte_end"] = span.start_byte + idx + len(obj.encode("utf-8"))
    else:
        out["byte_start"] = None
        out["byte_end"] = None
    return out


def propose(
    candidate_text: str,
    span: SpanRef,
    envelope: SourceEnvelope,
    event_us: int,
) -> ClaimProposal:
    """Build a deterministic, conservative :class:`ClaimProposal` for a span.

    Structure is derived only from ``candidate_text`` (the exact span
    quotation). Reported speech, referent-free objects, sensitive literal
    keys, and unparseable qualifiers all suppress structure — the proposal
    remains a searchable quotation either way (SPEC §12, §13).

    ``event_us`` is the *source event* time, used as the basis for relative
    dates and ``asserted_current_at`` interpretations (SPEC §14).
    """
    predicate: Optional[str] = None
    object_json: Optional[dict[str, Any]] = None
    method = PROPOSER_VERSION
    pattern_polarity: Optional[Polarity] = None

    if _REMEMBER_RE.search(candidate_text):
        # Explicit "remember …" bypasses the pattern table entirely — the
        # operator asked for storage, not for our guess at its structure.
        method = EXPLICIT_REMEMBER
    else:
        match = _match_structure(candidate_text)
        if match is not None:
            pred, obj, pos, topic, pattern_polarity = match
            if not _is_reported_speech(candidate_text, pos):
                predicate = pred
                object_json = _object_json(candidate_text, obj, span, envelope, topic)
            else:
                pattern_polarity = None

    polarity = pattern_polarity or (
        Polarity.NEGATED
        if NEGATION_RE.search(candidate_text)
        else Polarity.AFFIRMATIVE
    )

    return ClaimProposal(
        evidence=(span,),
        predicate=predicate,
        object_json=object_json,
        subject_entity_id=None,
        polarity=polarity,
        modality=_extract_modality(candidate_text),
        condition=_extract_condition(candidate_text),
        valid=_extract_valid(candidate_text, event_us),
        method=method,
        confidence_kind="rule",
    )


def is_explicit_remember(proposal: ClaimProposal) -> bool:
    """Whether the proposal came from an explicit "remember …" request."""
    return proposal.method == EXPLICIT_REMEMBER


def interpretation_status(proposal: ClaimProposal) -> str:
    """SPEC_V2 §13.06-§13.08 interpretation status for a proposal.

    ``structured`` when a typed predicate was derived, ``unstructured``
    otherwise. ``partial`` is reserved for future lanes that extract a
    predicate without a usable object — no current pattern produces it.
    """
    return (
        InterpretationStatus.STRUCTURED
        if proposal.predicate is not None
        else InterpretationStatus.UNSTRUCTURED
    )
