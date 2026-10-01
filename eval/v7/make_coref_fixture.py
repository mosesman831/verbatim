"""Owned coreference fixture generator (SPEC_V7 V7-13.20, SPEC_V7_5 §05 Q8).

Deterministic, stdlib-only generator producing the owned fixture the Q8
formula-search arm is gated on: >= 400 pronoun / definite-description
resolution cases written entirely from owned text (never benchmark gold),
including >= 100 cases whose antecedent is *not* in the immediately
previous turn and >= 50 two-candidate traps where the correct behavior
is abstention.  The gate metric is sieve precision >= 0.95; recall is
published (V7-13.20).

The cases exercise ``verbatim.enrichment.coref_sieve``
(``coref_sieve/v1``) through its frozen contract::

    resolve_antecedent(mention, unit_index, session_turns, lookback=6,
                       *, gender=None, kind=None, number=None,
                       last_resort_previous_turn=False)

``session_turns[i]`` = ``{"speaker": <canon|None>, "speaker_canon":
<canon|None>, "text": str, "canon_mentions": [canon, ...]}`` — the same
shape ``units_jobs._session_context`` produces (both speaker keys, canon
mentions in surface order, SPEAKER-role mentions excluded).  Optional
richer keys (``mention_roles``, ``subjects``) appear on a subset of
turns so the fixture covers the full documented input surface.

Record model (one JSON object per line, preceded by one ``fixture``
header line)::

    case_id             ``cf-0001`` ...
    stratum             primary slice tag (per-stratum reporting)
    strata              every applicable tag (distance, outcome, class)
    mention             the pronoun / description surface queried
    unit_index          index into ``turns`` of the mention's unit
    span                ``{"start","end"}`` char offsets of ``mention``
                        inside ``turns[unit_index]["text"]``
    turns               session turns (sieve contract shape, above)
    expected            antecedent canon, or ``"abstain"``
    intended_antecedent when ``expected == "abstain"`` but the text has a
                        linguistically-intended referent the sieve cannot
                        safely reach (decoys, group ambiguity)
    antecedent_turn     index of the nearest turn <= ``unit_index``
                        mentioning ``expected`` / ``intended_antecedent``
    antecedent_distance ``unit_index - antecedent_turn`` (0 = same unit)
    context             optional ``{"gender":..,"kind":..,"number":..}``
                        maps — the sieve's explicit-statement kwargs
    rationale           free-text why the label is what it is

Labels are assigned by *construction*, never by running the sieve:
single-survivor constructions are labeled with the survivor;
constructions that leave >= 2 class-compatible candidates inside the
declared margin are labeled ``"abstain"``; antecedents planted beyond
the lookback window are labeled ``"abstain"`` because the sieve cannot
reach them — including the ``window_decoy`` variants, where an in-window
same-class rival *will* be wrongly resolved.  Those misses stay visible
in the harness report rather than being designed away.

Pure enumeration over fixed pools — no RNG, no clock, no network.
Regenerating reproduces the committed fixture byte-identically (tests
assert this).  ``main()`` additionally runs a self-audit that prints the
sieve's own statuses and label disagreements to stderr without touching
labels (``python -m eval.v7.make_coref_fixture --audit``).
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

GENERATOR_ID = "coref_owned/v1"
FIXTURE_NAME = "coref_owned.jsonl"
FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, FIXTURE_NAME)

#: Declared sieve lookback (V7-13.20: N=6 — previous N turns + same unit).
LOOKBACK = 6
ABSTAIN = "abstain"

# ---------------------------------------------------------------------------
# Owned canon pools — invented entities only.  Speaker canons stay disjoint
# from mentionable canons so every case's semantics is unambiguous (a
# participant may still be *mentioned* by name in another speaker's turn —
# the speaker_exclusion stratum uses exactly that).
# ---------------------------------------------------------------------------

SPEAKERS: Tuple[str, ...] = (
    "lena", "arjun", "sofia", "kenji", "rosa", "elias",
    "nadia", "owen", "iris", "mateo", "sana", "dario",
)

#: Bare given names — the sieve never guesses gender from names, so all of
#: these are gender-unknown person canons.  They are split by conventional
#: reading only so pronoun pairings read naturally to a human reviewer.
NAMES_F: Tuple[str, ...] = (
    "maya", "alina", "sara", "priya", "nina", "julia", "emma", "zoe",
    "lily", "irene", "cleo", "dora", "elsa", "fiona", "greta", "vera",
    "tessa", "mira", "noor", "isla",
)
NAMES_M: Tuple[str, ...] = (
    "dan", "marcus", "tom", "raj", "omar", "kevin", "paul", "liam",
    "noah", "ethan", "felix", "hugo", "ivan", "jorge", "karl", "stefan",
    "bruno", "cole", "axel",
)
#: Conventionally ambiguous names — paired with epicene forms only.
NAMES_U: Tuple[str, ...] = (
    "alex", "sam", "jordan", "casey", "robin", "jamie", "quinn", "avery",
    "riley", "devon",
)
NAMES: Tuple[str, ...] = NAMES_F + NAMES_M + NAMES_U

#: Canons whose own surface carries a gendered noun (morphology-verified
#: gender — the only kind the sieve trusts besides an explicit map).
GENDER_M: Tuple[str, ...] = (
    "her brother", "my dad", "the new guy", "my uncle", "her husband",
    "her nephew", "the man", "his grandfather", "the waiter", "my stepdad",
    "her boyfriend", "the chairman", "my son", "his father", "the groom",
    "the policeman",
)
GENDER_F: Tuple[str, ...] = (
    "his sister", "their mom", "her aunt", "his wife", "the woman",
    "his niece", "their grandmother", "the waitress", "my stepmom",
    "his girlfriend", "the chairwoman", "my daughter", "her mother",
    "the bride", "the policewoman", "the landlady",
)
#: Person-noun canons — person evidence, no gender.
PERSON_N: Tuple[str, ...] = (
    "the manager", "my doctor", "her coach", "the neighbor", "my landlord",
    "the intern", "his mentor", "the recruiter", "the plumber", "her tutor",
    "the consultant", "my lawyer", "the photographer", "the electrician",
    "her therapist", "the firefighter",
)
#: Singular non-person canons (thing evidence in surface).
THINGS: Tuple[str, ...] = (
    "the report", "my bike", "the blue folder", "our old server",
    "the recipe", "her laptop", "the proposal", "the garden shed",
    "the rental car", "the coffee machine", "the budget", "my phone",
    "the project plan", "the presentation", "his truck", "the contract",
    "the invoice", "the meeting", "the piano", "the desk", "the monitor",
    "the printer", "the umbrella", "the guitar",
)
#: Plural canons (>1 entity).
PLURALS: Tuple[str, ...] = (
    "my parents", "the kids", "mom and dad", "the neighbors",
    "his siblings", "both dogs", "the twins", "my grandparents",
    "her coworkers", "the smiths", "our friends", "his cousins",
    "the visitors", "my nieces", "the clients",
)
#: Collective canons — grammatically singular groups (it/they-compatible,
#: never he/she).
COLLECTIVES: Tuple[str, ...] = (
    "the team", "her family", "the committee", "the board", "our group",
    "the band", "the staff", "the couple", "the jury", "the crew",
    "the department", "the panel",
)


def _canon_kind(canon: str) -> str:
    """Coarse kind for template selection (not a sieve concept)."""
    if canon in NAMES + GENDER_M + GENDER_F + PERSON_N + PLURALS \
            + COLLECTIVES:
        return "person" if canon in NAMES + GENDER_M + GENDER_F \
            + PERSON_N else "group"
    return "thing"


# ---------------------------------------------------------------------------
# Owned text templates — {S}/{s} slots take the canon surface
# (sentence-initial capitalized vs mid-sentence lower).
# ---------------------------------------------------------------------------

SUBJ_PERSON_T: Tuple[str, ...] = (
    "{S} called earlier", "{S} stopped by", "{S} sent a note",
    "{S} joined the call", "{S} spoke first", "{S} asked a question",
    "{S} left early", "{S} brought news", "{S} volunteered",
    "{S} was here", "{S} gave an update", "{S} checked in",
)
OBJ_PERSON_T: Tuple[str, ...] = (
    "we ran into {s} yesterday", "the call was about {s}",
    "I heard from {s}", "everyone asked about {s}",
    "we compared notes with {s}", "a message came from {s}",
    "we waited for {s}", "the update involved {s}",
    "the group missed {s}", "we quoted {s}",
)
SUBJ_THING_T: Tuple[str, ...] = (
    "{S} came up", "{S} needs attention", "{S} was on the agenda",
    "{S} broke down", "{S} arrived in the mail", "{S} got mentioned",
    "{S} is ready", "{S} took most of the hour", "{S} stalled",
    "{S} went missing",
)
OBJ_THING_T: Tuple[str, ...] = (
    "we discussed {s}", "the talk turned to {s}", "I looked over {s}",
    "we reviewed {s}", "we compared {s} with the old one",
    "the meeting covered {s}", "we booked {s}", "we wrapped up {s}",
    "we priced {s}", "we skimmed {s}",
)
PAIR_PERSON_T: Tuple[str, ...] = (
    "{A} met {B}", "{A} and {B} arrived together", "{A} introduced {B}",
    "{A} saw {B} at the party", "{A} called {B}", "{A} walked in with {B}",
)
PAIR_THING_T: Tuple[str, ...] = (
    "we compared {a} and {b}", "{A} and {b} both came up",
    "we discussed {a} and {b}", "the review covered {a} and {b}",
)
#: Mixed-kind pairs (person + thing / group + thing ...).
PAIR_MIX_T: Tuple[str, ...] = (
    "we heard about {a} and {b}", "the call covered {a} and {b}",
    "we discussed {a} and {b}", "{A} and {b} both came up",
    "the update mentioned {a} and {b}",
)
FILLER_T: Tuple[str, ...] = (
    "we chatted about nothing much", "a short pause followed",
    "the topic drifted", "we ordered coffee", "small talk filled the gap",
    "the conversation moved on", "we caught up on odds and ends",
    "a quiet stretch followed", "the discussion wandered",
    "we wrapped up loose ends", "we laughed about old times",
    "the mood stayed light",
)

#: Mention-embedding templates for the queried turn.  Every template
#: places the mention exactly once as a standalone token; the span is
#: located by word-boundary regex (never a bare substring hit inside
#: e.g. "with"/"meanwhile").
_PRE = ("then ", "and ", "later ", "so ", "meanwhile ", "next ")
_POST = (" left", " agreed", " called back", " smiled", " nodded",
         " stayed quiet", " spoke up", " explained", " was right",
         " said so", " laughed", " replied")
_ATTRIB_NOUN = ("phone", "idea", "plan", "car", "reply", "turn",
                "question", "story", "offer", "dog", "project", "wallet")
_ATTRIB_VERB = ("rang", "won", "worked", "stalled", "arrived", "came",
                "landed", "barked", "helped", "surfaced")
_POSS_T = ("the credit was {m}", "the choice stayed {m}",
           "the win was {m}", "the last seat was {m}",
           "the praise was {m}", "the idea was {m}")
_REFL_T = ("and fixed it by {m}", "then handled it by {m}",
           "and sorted it by {m}", "then did it by {m}",
           "and managed it by {m}")
_UNSUPPORTED_T: Dict[str, Tuple[str, ...]] = {
    "this": ("and this came up too", "then this mattered",
             "so this changed things"),
    "that": ("and that surfaced again", "then that was odd",
             "so that changed things"),
    "these": ("and these came up too", "then these mattered"),
    "those": ("and those resurfaced", "then those came up"),
    "who": ("and who was that again", "then who answered"),
    "which": ("and which one was it", "then which arrived first"),
    "someone": ("then someone knocked", "and someone called out"),
    "each": ("then each took a turn", "and each had a say"),
    "everyone": ("then everyone laughed", "and everyone agreed"),
    "one": ("then one stood out", "and one was missing"),
}
_DESC_POST = (" spoke up", " finally arrived", " said hello",
              " joined us", " asked a question", " took the floor")

_ATTRIB_FORMS = frozenset({"his", "her", "their", "its", "my", "your", "our"})
_POSS_FORMS = frozenset({"mine", "yours", "hers", "ours", "theirs"})
_REFL_FORMS = frozenset({
    "himself", "herself", "itself", "themselves", "myself", "yourself",
    "ourselves", "themself", "yourselves",
})

#: Pronoun pools keyed by sieve class.
M_PRON: Tuple[str, ...] = ("he", "him", "his", "himself")
F_PRON: Tuple[str, ...] = ("she", "her", "hers", "herself")
THEY: Tuple[str, ...] = ("they", "them", "their", "theirs", "themselves")
IT: Tuple[str, ...] = ("it", "its", "itself")
FIRST: Tuple[str, ...] = ("i", "me", "my", "mine", "myself")
SECOND: Tuple[str, ...] = ("you", "your", "yours", "yourself")
FIRST_PL: Tuple[str, ...] = ("we", "us", "our", "ours", "ourselves")
UNSUPPORTED: Tuple[str, ...] = tuple(_UNSUPPORTED_T)


def _class_of(mention: str) -> str:
    m = mention.lower()
    if m in M_PRON:
        return "masc"
    if m in F_PRON:
        return "fem"
    if m in THEY:
        return "epicene"
    if m in IT:
        return "neut"
    if m in FIRST:
        return "first_sg"
    if m in FIRST_PL:
        return "first_pl"
    if m in SECOND:
        return "second"
    if m in _UNSUPPORTED_T:
        return "unsupported"
    return "description"


def _surf(canon: str) -> str:
    """Canon -> sentence-initial surface (cosmetic capitalization)."""
    return canon[:1].upper() + canon[1:]


def _span_of(text: str, mention: str) -> Dict[str, int]:
    """Word-boundary span of the mention inside ``text`` — must occur
    exactly once (the assertion is a construction check)."""
    hits = list(re.finditer(r"\b" + re.escape(mention) + r"\b", text))
    assert len(hits) == 1, (mention, text)
    return {"start": hits[0].start(), "end": hits[0].end()}


# ---------------------------------------------------------------------------
# Turn builders
# ---------------------------------------------------------------------------

def _turn(speaker: Optional[str], text: str,
          canons: Iterable[str] = (),
          roles: Optional[Dict[str, str]] = None,
          subjects: Optional[Sequence[str]] = None,
          raw_speaker: bool = False) -> Dict[str, Any]:
    """Production-shaped turn (``units_jobs._session_context`` emits both
    ``speaker`` and ``speaker_canon``).  ``raw_speaker=True`` keeps a raw
    id in ``speaker`` and the canon in ``speaker_canon`` — the override
    path the sieve prefers."""
    t: Dict[str, Any] = {"text": text, "canon_mentions": list(canons)}
    if raw_speaker and speaker is not None:
        t["speaker"] = f"user:{speaker}"
        t["speaker_canon"] = speaker
    else:
        t["speaker"] = speaker
        t["speaker_canon"] = speaker
    if roles is not None:
        t["mention_roles"] = dict(roles)
    if subjects is not None:
        t["subjects"] = list(subjects)
    return t


def _canon_turn(i: int, speaker: Optional[str], canon: str,
                role: str = "subject", raw_speaker: bool = False) -> Dict[str, Any]:
    """Single-canon mention turn; text embeds the canon surface."""
    if _canon_kind(canon) == "thing":
        bank = SUBJ_THING_T if role == "subject" else OBJ_THING_T
    else:
        bank = SUBJ_PERSON_T if role == "subject" else OBJ_PERSON_T
    text = bank[i % len(bank)].format(S=_surf(canon), s=canon)
    # Alternate explicit ``mention_roles`` emission so both the
    # optional-key path and the production position-fallback path are
    # covered; role content stays honest either way.
    if role == "object":
        roles = {canon: "object"} if i % 2 == 0 else None
    else:
        roles = {canon: "subject"} if i % 5 == 0 else None
    return _turn(speaker, text, [canon], roles=roles, raw_speaker=raw_speaker)


def _pair_turn(i: int, speaker: Optional[str], a: str, b: str,
               both_subject: bool = False,
               use_subjects: bool = False) -> Dict[str, Any]:
    """Two-canon turn; ``a`` surfaces before ``b``."""
    ka, kb = _canon_kind(a), _canon_kind(b)
    if ka == "person" and kb == "person":
        text = PAIR_PERSON_T[i % len(PAIR_PERSON_T)].format(
            A=_surf(a), a=a, B=_surf(b), b=b)
    elif ka == "thing" and kb == "thing":
        text = PAIR_THING_T[i % len(PAIR_THING_T)].format(
            A=_surf(a), a=a, B=_surf(b), b=b)
    else:
        text = PAIR_MIX_T[i % len(PAIR_MIX_T)].format(
            A=_surf(a), a=a, B=_surf(b), b=b)
    if both_subject:
        return _turn(speaker, text, [a, b],
                     roles={a: "subject", b: "subject"})
    if use_subjects:
        # ``subjects`` variant of the role contract: listed canons are
        # subject-position, everything else is object-position.
        if ka == kb == "person":
            return _turn(speaker, text, [a, b], subjects=[a])
        return _turn(speaker, text, [a, b], subjects=[])
    if ka == kb:
        # same-kind pair: first mention subject-position, second object
        roles = {a: "subject", b: "object"} if ka == "person" \
            else {a: "object", b: "object"}
    else:
        roles = {a: "object", b: "object"}
    return _turn(speaker, text, [a, b], roles=roles)


def _empty_turn(i: int, speaker: Optional[str]) -> Dict[str, Any]:
    return _turn(speaker, FILLER_T[i % len(FILLER_T)], [])


#: Filler canon pools that can never survive the target class's agreement
#: filter (things/collectives/plurals are invisible to masc/fem; persons
#: and plurals are invisible to ``it``; only pure singular things are
#: invisible to ``they``; ``any`` for mention classes whose label does
#: not depend on the candidate filter).
_FILLER_POOLS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "masc": (THINGS, COLLECTIVES, PLURALS, GENDER_F),
    "fem": (THINGS, COLLECTIVES, PLURALS, GENDER_M),
    "neut": (NAMES, PERSON_N, GENDER_M, GENDER_F, PLURALS),
    "epicene": (THINGS,),
    "any": (THINGS, PLURALS, COLLECTIVES, NAMES, PERSON_N),
}


def _filler_canon(cls: str, i: int, j: int) -> Optional[str]:
    pools = _FILLER_POOLS.get(cls, _FILLER_POOLS["any"])
    if i % 3 == 0:  # every third gap stays plain chatter
        return None
    pool = pools[(i + j) % len(pools)]
    return pool[(i + j) % len(pool)]


def _filler_turn(i: int, speaker: Optional[str], canon: Optional[str],
                 role: str = "object") -> Dict[str, Any]:
    if canon is None:
        return _empty_turn(i, speaker)
    return _canon_turn(i, speaker, canon, role)


def _mention_text(mention: str, i: int) -> str:
    """Current-unit text embedding ``mention`` exactly once."""
    m = mention.lower()
    if _class_of(m) == "description":
        return _PRE[i % len(_PRE)] + m + _DESC_POST[i % len(_DESC_POST)]
    if m in _UNSUPPORTED_T:
        bank = _UNSUPPORTED_T[m]
        return bank[i % len(bank)]
    if m in _ATTRIB_FORMS:
        return (_PRE[i % len(_PRE)] + m + " "
                + _ATTRIB_NOUN[i % len(_ATTRIB_NOUN)] + " "
                + _ATTRIB_VERB[i % len(_ATTRIB_VERB)])
    if m in _POSS_FORMS:
        return _POSS_T[i % len(_POSS_T)].format(m=m)
    if m in _REFL_FORMS:
        return _REFL_T[i % len(_REFL_T)].format(m=m)
    return _PRE[i % len(_PRE)] + m + _POST[i % len(_POST)]


def _current_turn(i: int, speaker: Optional[str], mention: str,
                  canons: Sequence[str] = (),
                  roles: Optional[Dict[str, str]] = None,
                  subjects: Optional[Sequence[str]] = None,
                  raw_speaker: bool = False
                  ) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Mention unit + byte span of ``mention`` inside its text."""
    if canons:
        c = canons[0]
        if len(canons) == 1:
            if mention in _ATTRIB_FORMS:
                text = f"{_surf(c)} called and {mention} phone rang"
            elif mention in _POSS_FORMS:
                text = f"{_surf(c)} spoke but the credit was {mention}"
            elif mention in _REFL_FORMS:
                text = f"{_surf(c)} called and fixed it by {mention}"
            else:
                bank = (
                    "{S} called and {m} was happy",
                    "{S} spoke first and {m} smiled",
                    "we hired {s} and {m} starts monday",
                    "{S} arrived and {m} sat down",
                    "{S} replied and {m} laughed",
                    "we met {s} and {m} agreed",
                    "{S} answered and {m} nodded",
                )
                text = bank[i % len(bank)].format(S=_surf(c), s=c, m=mention)
        else:
            a, b = canons[0], canons[1]
            if mention in _ATTRIB_FORMS:
                text = f"{_surf(a)} met {_surf(b)} and {mention} phone rang"
            else:
                bank = (
                    "{A} met {B} and {m} laughed",
                    "{A} saw {B} and {m} waved",
                    "{A} called {B} and {m} answered",
                    "{A} introduced {B} and {m} smiled",
                    "we compared {a} and {b} and {m} won",
                )
                text = bank[i % len(bank)].format(
                    A=_surf(a), a=a, B=_surf(b), b=b, m=mention)
    else:
        text = _mention_text(mention, i)
    span = _span_of(text, mention)
    turn = _turn(speaker, text, canons, roles=roles, subjects=subjects,
                 raw_speaker=raw_speaker)
    return turn, span


def _session(speakers: Sequence[Optional[str]], n_turns: int,
             slots: Dict[int, Dict[str, Any]],
             filler_cls: str = "any") -> List[Dict[str, Any]]:
    """Assemble ``n_turns`` turns: ``slots`` wins, gaps get safe fillers.

    ``filler_cls`` selects filler canons that cannot survive the target
    pronoun's agreement filter (``'any'`` = unconstrained)."""
    turns: List[Dict[str, Any]] = []
    for idx in range(n_turns):
        if idx in slots:
            turns.append(slots[idx])
            continue
        sp = speakers[idx % len(speakers)]
        canon = _filler_canon(filler_cls, idx, n_turns)
        turns.append(_filler_turn(idx, sp, canon))
    return turns


# ---------------------------------------------------------------------------
# Case assembly
# ---------------------------------------------------------------------------

class _Builder:
    def __init__(self) -> None:
        self.cases: List[Dict[str, Any]] = []

    def add(self, stratum: str, strata: Iterable[str], mention: str,
            turns: List[Dict[str, Any]], expected: str,
            ui: Optional[int] = None, span: Optional[Dict[str, int]] = None,
            intended: Optional[str] = None,
            context: Optional[Dict[str, Dict[str, str]]] = None,
            rationale: str = "") -> None:
        ui = len(turns) - 1 if ui is None else ui
        assert 0 <= ui < len(turns)
        if span is None:
            span = _span_of(turns[ui]["text"], mention)
        target = intended if expected == ABSTAIN else expected
        antecedent_turn = antecedent_distance = None
        if target:
            for ti in range(ui, -1, -1):
                if target in (turns[ti].get("canon_mentions") or ()):
                    antecedent_turn = ti
                    antecedent_distance = ui - ti
                    break
        if expected != ABSTAIN and antecedent_turn is None:
            # speaker/addressee resolutions point at a speaker canon,
            # which is never a canon_mention — verify that instead.
            speakers = {
                (t.get("speaker_canon") or t.get("speaker"))
                for t in turns}
            assert expected in speakers, (
                "resolve target neither mentioned nor a speaker", expected)
        rec: Dict[str, Any] = {
            "case_id": f"cf-{len(self.cases) + 1:04d}",
            "stratum": stratum,
            "strata": sorted(set(strata) | {stratum}),
            "mention": mention,
            "unit_index": ui,
            "span": span,
            "turns": turns,
            "expected": expected,
            "antecedent_turn": antecedent_turn,
            "antecedent_distance": antecedent_distance,
            "rationale": rationale,
        }
        if expected == ABSTAIN and intended:
            rec["intended_antecedent"] = intended
        if context:
            rec["context"] = context
        self.cases.append(rec)


def _pair_speakers(i: int, k: int = 2) -> List[str]:
    return [SPEAKERS[(i + j) % len(SPEAKERS)] for j in range(k)]


def _pron_for(canon: str, i: int) -> str:
    """Conventionally sensible pronoun for a canon (label-irrelevant for
    single-survivor cases; keeps the fixture readable)."""
    if canon in NAMES_U:
        return THEY[i % 5]
    if canon in NAMES_M or canon in GENDER_M:
        return (M_PRON + THEY[:2])[i % 6]
    if canon in NAMES_F or canon in GENDER_F:
        return (F_PRON + THEY[:2])[i % 6]
    if canon in THINGS:
        return IT[i % 3]
    return THEY[i % 5]


# ---------------------------------------------------------------------------
# Stratum builders — labels by construction, never by running the sieve
# ---------------------------------------------------------------------------

def _s_single_prev(b: _Builder) -> None:
    """d=1, exactly one class-compatible candidate -> resolve (64)."""
    tags = ["antecedent_prev_turn", "expected_resolve", "single_candidate"]
    for j, name in enumerate(NAMES_F[:10] + NAMES_M[:10] + NAMES_U[:4]):
        pron = _pron_for(name, j)
        sp = _pair_speakers(j)
        cur, span = _current_turn(j, sp[1], pron)
        turns = [_canon_turn(j, sp[0], name), cur]
        b.add("single_prev_turn", tags + [f"class:{_class_of(pron)}"],
              pron, turns, name, span=span,
              rationale="single bare-name antecedent at d=1 resolves")
    for j, canon in enumerate(GENDER_M[:6] + GENDER_F[:6]):
        pron = M_PRON[j % 4] if canon in GENDER_M else F_PRON[j % 4]
        sp = _pair_speakers(30 + j)
        cur, span = _current_turn(j, sp[1], pron)
        turns = [_canon_turn(j, sp[0], canon,
                             "object" if j % 3 == 0 else "subject"), cur]
        b.add("single_prev_turn", tags + ["gendered_canon",
                                          f"class:{_class_of(pron)}"],
              pron, turns, canon, span=span,
              rationale="gendered-noun canon; single survivor")
    for j, canon in enumerate(THINGS[:10]):
        pron = IT[j % 3]
        sp = _pair_speakers(50 + j)
        cur, span = _current_turn(j, sp[1], pron)
        turns = [_canon_turn(j, sp[0], canon,
                             "object" if j % 2 else "subject"), cur]
        b.add("single_prev_turn", tags + [f"class:{_class_of(pron)}"],
              pron, turns, canon, span=span,
              rationale="thing canon; it resolves to the only survivor")
    for j, canon in enumerate(COLLECTIVES[:6] + PLURALS[:6]):
        pron = IT[j % 3] if canon in COLLECTIVES and j % 2 == 0 \
            else THEY[j % 5]
        sp = _pair_speakers(70 + j)
        cur, span = _current_turn(j, sp[1], pron)
        turns = [_canon_turn(j, sp[0], canon), cur]
        b.add("single_prev_turn", tags + [f"class:{_class_of(pron)}"],
              pron, turns, canon, span=span,
              rationale="collective/plural canon takes it or they")
    for j, canon in enumerate(PERSON_N[:6]):
        pron = (THEY + M_PRON[:1] + F_PRON[:1])[j % 6]
        sp = _pair_speakers(90 + j)
        cur, span = _current_turn(j, sp[1], pron)
        turns = [_canon_turn(j, sp[0], canon), cur]
        b.add("single_prev_turn", tags + [f"class:{_class_of(pron)}"],
              pron, turns, canon, span=span,
              rationale="person-noun canon, single survivor (unverified gender ok)")


def _s_same_turn(b: _Builder) -> None:
    """d=0 — antecedent inside the mention's own unit (26)."""
    pool = list(NAMES_F[:3]) + list(NAMES_M[:3]) + list(GENDER_F[:2]) \
        + list(GENDER_M[:2]) + list(THINGS[:2])
    prons = ["she", "her", "she", "he", "his", "him",
             "she", "her", "he", "his", "it", "its"]
    for j, canon in enumerate(pool):
        pron = prons[j]
        sp = _pair_speakers(140 + j)
        cur, span = _current_turn(j, sp[0], pron, canons=[canon])
        b.add("same_turn", ["antecedent_same_turn", "expected_resolve",
                            "single_candidate", f"class:{_class_of(pron)}"],
              pron, [cur], canon, span=span,
              rationale="same-unit mention resolves (d=0 candidate)")
    # traps: two same-class canons inside the mention unit.
    trap_prons = ("she", "her", "they", "she", "her", "they")
    for j in range(6):
        a, c = NAMES_F[j], NAMES_F[(j + 5) % len(NAMES_F)]
        pron = trap_prons[j]
        sp = _pair_speakers(160 + j)
        cur, span = _current_turn(j, sp[0], pron, canons=[a, c],
                                  roles={a: "subject", c: "object"})
        b.add("same_turn", ["antecedent_same_turn", "expected_abstain",
                            "two_candidate_trap",
                            f"class:{_class_of(pron)}"],
              pron, [cur], ABSTAIN, span=span,
              rationale="two same-class canons in the mention unit -> abstain")
    # speaker self-name: the current speaker's canon is excluded.
    for j, pron in enumerate(("she", "he", "she", "they")):
        sp = _pair_speakers(170 + j)
        cur, span = _current_turn(j, sp[0], pron, canons=[sp[0]])
        b.add("same_turn", ["antecedent_same_turn", "expected_abstain",
                            "speaker_exclusion",
                            f"class:{_class_of(pron)}"],
              pron, [cur], ABSTAIN, span=span,
              rationale="current speaker excluded from third-person candidates")
    # current-turn competitor traps: antecedent at d=1 + rival at d=0.
    for j in range(4):
        a, c = NAMES_M[j], NAMES_M[(j + 4) % len(NAMES_M)]
        sp = _pair_speakers(180 + j)
        cur, span = _current_turn(j, sp[0], "he", canons=[c])
        turns = [_canon_turn(j, sp[1], a), cur]
        b.add("same_turn", ["antecedent_prev_turn", "expected_abstain",
                            "two_candidate_trap", "current_turn_competitor",
                            "class:masc"],
              "he", turns, ABSTAIN, span=span, intended=a,
              rationale="same-turn rival outscores the d=1 antecedent; unverified coin flip")


def _s_deep_resolve(b: _Builder) -> None:
    """Antecedent at d in 2..6, resolvable (130)."""
    pool = list(NAMES_F[:12]) + list(NAMES_M[:12]) + list(NAMES_U[:4]) \
        + list(GENDER_M[:8]) + list(GENDER_F[:8]) + list(PERSON_N[:8]) \
        + list(THINGS[:12]) + list(PLURALS[:4]) + list(COLLECTIVES[:4])
    # 80 single-survivor deep cases with safe fillers.
    for j in range(80):
        canon = pool[j % len(pool)]
        pron = _pron_for(canon, j)
        cls = _class_of(pron)
        d = 2 + j % 5
        sp = _pair_speakers(200 + j, 2 + (j % 2))
        slots = {0: _canon_turn(j, sp[0], canon,
                                "object" if j % 4 == 0 else "subject")}
        turns = _session(sp, d + 1, slots, filler_cls=cls)
        cur, span = _current_turn(j, sp[1], pron)
        turns[-1] = cur
        b.add("deep_resolve", ["antecedent_beyond_prev", "expected_resolve",
                               "single_candidate", f"class:{cls}"],
              pron, turns, canon, span=span,
              rationale=f"antecedent at d={d} inside lookback; fillers carry no same-class candidate")
    # 20 re-mentioned: canon at d=4..6 and again at d=2 (nearest = 2).
    for j in range(20):
        canon = pool[(j * 7) % len(pool)]
        pron = _pron_for(canon, j)
        cls = _class_of(pron)
        d = 4 + j % 3
        sp = _pair_speakers(300 + j, 2 + (j % 2))
        slots = {0: _canon_turn(j, sp[0], canon),
                 d - 2: _canon_turn(j + 1, sp[1 % len(sp)], canon,
                                    "object" if j % 2 else "subject")}
        turns = _session(sp, d + 1, slots, filler_cls=cls)
        cur, span = _current_turn(j, sp[1], pron)
        turns[-1] = cur
        b.add("deep_resolve", ["antecedent_beyond_prev", "expected_resolve",
                               "single_candidate", "re_mentioned",
                               f"class:{cls}"],
              pron, turns, canon, span=span,
              rationale="canon re-mentioned at d=2; nearest antecedent beyond prev turn")
    # 15 decisive-margin: verified winner d=2..3 beats same-class rival
    # at d=5..6 by more than the margin.
    for j in range(15):
        if j % 3 == 0:
            win = THINGS[j % len(THINGS)]
            rival = THINGS[(j + 5) % len(THINGS)]
            pron = IT[j % 3]
        elif j % 3 == 1:
            win = GENDER_M[j % len(GENDER_M)]
            rival = GENDER_M[(j + 6) % len(GENDER_M)]
            pron = M_PRON[j % 4]
        else:
            win = GENDER_F[j % len(GENDER_F)]
            rival = GENDER_F[(j + 7) % len(GENDER_F)]
            pron = F_PRON[j % 4]
        cls = _class_of(pron)
        dw = 2 + j % 2                      # winner distance 2..3
        dr = min(dw + 3 + j % 3, LOOKBACK)  # rival distance 5..6
        sp = _pair_speakers(320 + j)
        slots = {0: _canon_turn(j, sp[0], rival),
                 dr - dw: _canon_turn(j + 1, sp[1 % len(sp)], win)}
        turns = _session(sp, dr + 1, slots, filler_cls=cls)
        cur, span = _current_turn(j, sp[1], pron)
        turns[-1] = cur
        b.add("deep_resolve", ["antecedent_beyond_prev", "expected_resolve",
                               "decisive_margin", f"class:{cls}"],
              pron, turns, win, span=span,
              rationale=f"verified winner at d={dw} beats same-class rival at d={dr} past the margin")
    # 15 mid-session: the mention unit is not the session's last turn.
    for j in range(15):
        canon = pool[(j * 11) % len(pool)]
        pron = _pron_for(canon, j)
        cls = _class_of(pron)
        d = 2 + j % 4  # d=2..5 relative to ui
        ui = d
        sp = _pair_speakers(340 + j, 2 + (j % 2))
        slots = {0: _canon_turn(j, sp[0], canon)}
        turns = _session(sp, ui + 1 + (1 + j % 2), slots, filler_cls=cls)
        cur, span = _current_turn(j, sp[1], pron)
        turns[ui] = cur
        b.add("deep_resolve", ["antecedent_beyond_prev", "expected_resolve",
                               "single_candidate", "mid_session",
                               f"class:{cls}"],
              pron, turns, canon, ui=ui, span=span,
              rationale="mention unit mid-session; antecedent d>=2 back")


def _s_deep_abstain(b: _Builder) -> None:
    """Deep intended referents that still abstain (28)."""
    # 12 verified same-class rivals at adjacent depths >= d=2 — the score
    # gap is exactly the margin boundary (<= _MARGIN) -> abstain.
    for j in range(12):
        if j % 2:
            a = GENDER_M[j % len(GENDER_M)]
            c = GENDER_M[(j + 4) % len(GENDER_M)]
            pron = M_PRON[j % 4]
        else:
            a = GENDER_F[j % len(GENDER_F)]
            c = GENDER_F[(j + 5) % len(GENDER_F)]
            pron = F_PRON[j % 4]
        cls = _class_of(pron)
        d = 3 + j % 3  # a at d-1 in 2..4, rival at d in 3..5
        sp = _pair_speakers(360 + j)
        slots = {0: _canon_turn(j, sp[0], c),
                 1: _canon_turn(j + 1, sp[1 % len(sp)], a)}
        turns = _session(sp, d + 1, slots, filler_cls=cls)
        cur, span = _current_turn(j, sp[1], pron)
        turns[-1] = cur
        b.add("deep_abstain", ["antecedent_beyond_prev", "expected_abstain",
                               "two_candidate_trap", "margin_boundary",
                               f"class:{cls}"],
              pron, turns, ABSTAIN, span=span, intended=a,
              rationale="two verified same-class candidates inside the margin -> abstain")
    # 8 unverified-winner: nearer unverified rival outscores the deep
    # target; an unverified winner cannot take a gendered pronoun.
    for j in range(8):
        target = NAMES_F[j] if j % 2 else NAMES_M[j]
        rival = NAMES_U[j % len(NAMES_U)]
        pron = F_PRON[j % 4] if j % 2 else M_PRON[j % 4]
        cls = _class_of(pron)
        d = 3 + j % 3  # target at d=3..5, rival at d=1
        sp = _pair_speakers(380 + j)
        slots = {0: _canon_turn(j, sp[0], target),
                 d - 1: _canon_turn(j + 1, sp[1 % len(sp)], rival)}
        turns = _session(sp, d + 1, slots, filler_cls=cls)
        cur, span = _current_turn(j, sp[1], pron)
        turns[-1] = cur
        b.add("deep_abstain", ["antecedent_beyond_prev", "expected_abstain",
                               "two_candidate_trap", "unverified_winner",
                               f"class:{cls}"],
              pron, turns, ABSTAIN, span=span, intended=target,
              rationale="nearer rival wins on recency but is unverified -> abstain")
    # 8 epicene group ambiguity: two survivors at any depth -> abstain.
    for j in range(8):
        if j % 3 == 0:
            cands = (NAMES_F[j], NAMES_M[j])
        elif j % 3 == 1:
            cands = (PLURALS[j % len(PLURALS)],
                     NAMES_F[(j + 3) % len(NAMES_F)])
        else:
            cands = (COLLECTIVES[j % len(COLLECTIVES)],
                     NAMES_M[(j + 2) % len(NAMES_M)])
        pron = THEY[j % 5]
        d = 3 + j % 3
        sp = _pair_speakers(400 + j)
        slots = {0: _canon_turn(j, sp[0], cands[0]),
                 d - 2: _canon_turn(j + 1, sp[1 % len(sp)], cands[1])}
        turns = _session(sp, d + 1, slots, filler_cls="epicene")
        cur, span = _current_turn(j, sp[1], pron)
        turns[-1] = cur
        b.add("deep_abstain", ["antecedent_beyond_prev", "expected_abstain",
                               "two_candidate_trap", "group_ambiguity",
                               "class:epicene"],
              pron, turns, ABSTAIN, span=span,
              rationale="they may denote the union of two survivors -> abstain outright")


def _s_beyond_window(b: _Builder) -> None:
    """Antecedent outside the lookback window, d > 6 (30)."""
    pool = list(NAMES_F[:6]) + list(NAMES_M[:6]) + list(GENDER_F[:4]) \
        + list(GENDER_M[:4]) + list(THINGS[:2])
    for j in range(22):
        canon = pool[j % len(pool)]
        pron = _pron_for(canon, j)
        cls = _class_of(pron)
        d = LOOKBACK + 1 + j % 4  # d=7..10
        sp = _pair_speakers(420 + j)
        slots = {0: _canon_turn(j, sp[0], canon)}
        turns = _session(sp, d + 1, slots, filler_cls=cls)
        cur, span = _current_turn(j, sp[1], pron)
        turns[-1] = cur
        b.add("beyond_window", ["antecedent_beyond_window", "expected_abstain",
                                "no_window_candidate", f"class:{cls}"],
              pron, turns, ABSTAIN, span=span, intended=canon,
              rationale=f"antecedent at d={d} beyond lookback {LOOKBACK}; unreachable -> abstain")
    # 8 decoy cases: out-of-window target + in-window same-class decoy.
    # The sieve *will* resolve the decoy — a measured precision error,
    # kept deliberately: the honest label for an unreachable referent
    # with a live rival is abstain.
    for j in range(8):
        if j % 2:
            target = NAMES_F[(j + 8) % len(NAMES_F)]
            decoy = NAMES_F[(j + 2) % len(NAMES_F)]
            pron = F_PRON[j % 4]
        else:
            target = NAMES_M[(j + 8) % len(NAMES_M)]
            decoy = NAMES_M[(j + 2) % len(NAMES_M)]
            pron = M_PRON[j % 4]
        cls = _class_of(pron)
        d = LOOKBACK + 2 + j % 3  # target at d=8..10
        dd = 1 + j % 2            # decoy at d=1..2
        sp = _pair_speakers(450 + j)
        slots = {0: _canon_turn(j, sp[0], target),
                 d - dd: _canon_turn(j + 1, sp[1 % len(sp)], decoy)}
        turns = _session(sp, d + 1, slots, filler_cls=cls)
        cur, span = _current_turn(j, sp[1], pron)
        turns[-1] = cur
        b.add("beyond_window", ["antecedent_beyond_window", "expected_abstain",
                                "window_decoy", f"class:{cls}"],
              pron, turns, ABSTAIN, span=span, intended=target,
              rationale="true referent beyond window; the in-window decoy is a wrong merge")


def _s_traps(b: _Builder) -> None:
    """Two-candidate traps at d=1 (46)."""
    # 16 previous-turn person pairs.
    for j in range(16):
        pron = THEY[j % 5] if j % 5 == 0 else (F_PRON + M_PRON)[j % 8]
        if pron in M_PRON:
            a, c = NAMES_M[j % len(NAMES_M)], \
                NAMES_M[(j + 7) % len(NAMES_M)]
        elif pron in F_PRON:
            a, c = NAMES_F[j % len(NAMES_F)], \
                NAMES_F[(j + 6) % len(NAMES_F)]
        else:
            a, c = NAMES[j % len(NAMES)], NAMES[(j + 9) % len(NAMES)]
        sp = _pair_speakers(470 + j)
        cur, span = _current_turn(j, sp[1], pron)
        turns = [_pair_turn(j, sp[0], a, c, use_subjects=j % 4 == 0), cur]
        b.add("trap", ["antecedent_prev_turn", "expected_abstain",
                       "two_candidate_trap", f"class:{_class_of(pron)}"],
              pron, turns, ABSTAIN, span=span,
              rationale="two same-class canons in the previous turn -> abstain")
    # 10 verified-gender pairs.
    for j in range(10):
        if j % 2:
            a = GENDER_M[j % len(GENDER_M)]
            c = GENDER_M[(j + 3) % len(GENDER_M)]
            pron = M_PRON[j % 4]
        else:
            a = GENDER_F[j % len(GENDER_F)]
            c = GENDER_F[(j + 5) % len(GENDER_F)]
            pron = F_PRON[j % 4]
        sp = _pair_speakers(490 + j)
        cur, span = _current_turn(j, sp[1], pron)
        turns = [_pair_turn(j, sp[0], a, c, both_subject=j % 3 == 0), cur]
        b.add("trap", ["antecedent_prev_turn", "expected_abstain",
                       "two_candidate_trap", "verified_pair",
                       f"class:{_class_of(pron)}"],
              pron, turns, ABSTAIN, span=span,
              rationale="two verified same-gender canons inside the margin -> abstain")
    # 12 it-pairs: two things in one turn.
    for j in range(12):
        a = THINGS[j % len(THINGS)]
        c = THINGS[(j + 4) % len(THINGS)]
        pron = IT[j % 3]
        sp = _pair_speakers(510 + j)
        cur, span = _current_turn(j, sp[1], pron)
        turns = [_pair_turn(j, sp[0], a, c), cur]
        b.add("trap", ["antecedent_prev_turn", "expected_abstain",
                       "two_candidate_trap", "class:neut"],
              pron, turns, ABSTAIN, span=span,
              rationale="two thing canons inside the margin -> abstain")
    # 8 they-pairs across classes.
    combos = [
        (NAMES_F[0], NAMES_M[0]), (PLURALS[0], PLURALS[4]),
        (NAMES_M[1], COLLECTIVES[0]), (PERSON_N[0], NAMES_F[1]),
        (NAMES_F[2], NAMES_M[2]), (PLURALS[1], PLURALS[5]),
        (NAMES_M[3], COLLECTIVES[1]), (PERSON_N[1], NAMES_F[3]),
    ]
    for j, (a, c) in enumerate(combos):
        pron = THEY[j % 5]
        sp = _pair_speakers(530 + j)
        cur, span = _current_turn(j, sp[1], pron)
        turns = [_pair_turn(j, sp[0], a, c), cur]
        b.add("trap", ["antecedent_prev_turn", "expected_abstain",
                       "two_candidate_trap", "group_ambiguity",
                       "class:epicene"],
              pron, turns, ABSTAIN, span=span,
              rationale="they with two survivors may denote the union -> abstain")


def _s_mixed_class(b: _Builder) -> None:
    """Mixed candidates; the agreement filter leaves one survivor (42)."""
    for j in range(12):  # {person, thing} + gendered pronoun -> person
        person = NAMES[j % len(NAMES)]
        thing = THINGS[(j + 2) % len(THINGS)]
        pron = _pron_for(person, j)
        if pron in THEY:
            pron = M_PRON[j % 4] if person in NAMES_M else F_PRON[j % 4]
        sp = _pair_speakers(550 + j)
        cur, span = _current_turn(j, sp[1], pron)
        turns = [_pair_turn(j, sp[0], person, thing), cur]
        b.add("mixed_class", ["antecedent_prev_turn", "expected_resolve",
                              f"class:{_class_of(pron)}"],
              pron, turns, person, span=span,
              rationale="thing fails the gendered filter; person is the only survivor")
    for j in range(10):  # {person, thing} + it -> thing
        person = NAMES_F[(j + 4) % len(NAMES_F)]
        thing = THINGS[(j + 7) % len(THINGS)]
        sp = _pair_speakers(570 + j)
        cur, span = _current_turn(j, sp[1], IT[j % 3])
        turns = [_pair_turn(j, sp[0], person, thing), cur]
        b.add("mixed_class", ["antecedent_prev_turn", "expected_resolve",
                              "class:neut"],
              IT[j % 3], turns, thing, span=span,
              rationale="person fails the it filter; the thing is the only survivor")
    for j in range(8):  # {male, female} verified pair + matching pronoun
        m_c = GENDER_M[j % len(GENDER_M)]
        f_c = GENDER_F[j % len(GENDER_F)]
        pron = M_PRON[j % 4] if j % 2 else F_PRON[j % 4]
        win = m_c if pron in M_PRON else f_c
        sp = _pair_speakers(590 + j)
        cur, span = _current_turn(j, sp[1], pron)
        turns = [_pair_turn(j, sp[0], m_c, f_c), cur]
        b.add("mixed_class", ["antecedent_prev_turn", "expected_resolve",
                              "gender_filter", f"class:{_class_of(pron)}"],
              pron, turns, win, span=span,
              rationale="verified opposite-gender canon is filtered out")
    for j in range(6):  # {plural, thing} + it -> thing
        pl = PLURALS[j % len(PLURALS)]
        th = THINGS[(j + 9) % len(THINGS)]
        sp = _pair_speakers(610 + j)
        cur, span = _current_turn(j, sp[1], IT[j % 3])
        turns = [_pair_turn(j, sp[0], pl, th), cur]
        b.add("mixed_class", ["antecedent_prev_turn", "expected_resolve",
                              "number_filter", "class:neut"],
              IT[j % 3], turns, th, span=span,
              rationale="plural fails the it number filter")
    for j in range(6):  # {person, collective} + it -> collective
        person = NAMES_M[(j + 3) % len(NAMES_M)]
        coll = COLLECTIVES[(j + 2) % len(COLLECTIVES)]
        sp = _pair_speakers(620 + j)
        cur, span = _current_turn(j, sp[1], IT[j % 3])
        turns = [_pair_turn(j, sp[0], person, coll), cur]
        b.add("mixed_class", ["antecedent_prev_turn", "expected_resolve",
                              "class:neut"],
              IT[j % 3], turns, coll, span=span,
              rationale="collective is it-compatible; the person is not")


def _s_gender(b: _Builder) -> None:
    """Gender agreement (20)."""
    # 8 pronoun vs only opposite-verified canons -> zero survivors.
    for j in range(8):
        if j % 2:
            canon, pron = GENDER_M[j % len(GENDER_M)], F_PRON[j % 4]
            other = GENDER_M[(j + 6) % len(GENDER_M)]
        else:
            canon, pron = GENDER_F[j % len(GENDER_F)], M_PRON[j % 4]
            other = GENDER_F[(j + 6) % len(GENDER_F)]
        cls = _class_of(pron)
        sp = _pair_speakers(640 + j)
        slots = {0: _canon_turn(j, sp[0], canon),
                 1: _canon_turn(j + 1, sp[1 % len(sp)], other)}
        turns = _session(sp, 3 + j % 3, slots, filler_cls=cls)
        cur, span = _current_turn(j, sp[1], pron)
        turns[-1] = cur
        b.add("gender_agreement", ["antecedent_prev_turn", "expected_abstain",
                                   "gender_mismatch", "zero_survivors",
                                   f"class:{cls}"],
              pron, turns, ABSTAIN, span=span,
              rationale="all candidates carry verified opposite gender -> no survivors")
    # 6 verified winner at d=1 beats an unverified rival at d=3..4.
    for j in range(6):
        if j % 2:
            win, pron = GENDER_F[j % len(GENDER_F)], F_PRON[j % 4]
        else:
            win, pron = GENDER_M[j % len(GENDER_M)], M_PRON[j % 4]
        rival = NAMES_U[(j + 3) % len(NAMES_U)]
        cls = _class_of(pron)
        sp = _pair_speakers(660 + j)
        slots = {0: _canon_turn(j, sp[0], rival),
                 2 + j % 2: _canon_turn(j + 1, sp[1 % len(sp)], win)}
        turns = _session(sp, 4 + j % 2, slots, filler_cls=cls)
        cur, span = _current_turn(j, sp[1], pron)
        turns[-1] = cur
        b.add("gender_agreement", ["antecedent_prev_turn", "expected_resolve",
                                   "verified_winner", f"class:{cls}"],
              pron, turns, win, span=span,
              rationale="verified-gender winner beats an unverified rival past the margin")
    # 6 unverified top vs verified runner -> abstain.
    for j in range(6):
        top = NAMES_U[(j + 5) % len(NAMES_U)]
        if j % 2:
            runner, pron = GENDER_M[j % len(GENDER_M)], M_PRON[j % 4]
        else:
            runner, pron = GENDER_F[j % len(GENDER_F)], F_PRON[j % 4]
        cls = _class_of(pron)
        sp = _pair_speakers(670 + j)
        slots = {0: _canon_turn(j, sp[0], runner),
                 2: _canon_turn(j + 1, sp[1 % len(sp)], top)}
        turns = _session(sp, 4, slots, filler_cls=cls)
        cur, span = _current_turn(j, sp[1], pron)
        turns[-1] = cur
        b.add("gender_agreement", ["antecedent_prev_turn", "expected_abstain",
                                   "two_candidate_trap", "unverified_winner",
                                   f"class:{cls}"],
              pron, turns, ABSTAIN, span=span, intended=runner,
              rationale="unverified winner over a verified rival still abstains")


def _s_number(b: _Builder) -> None:
    """Number agreement (24)."""
    for j in range(8):  # singular gendered pronoun + plural only
        canon = PLURALS[j % len(PLURALS)]
        pron = (M_PRON + F_PRON)[j % 8]
        cls = _class_of(pron)
        sp = _pair_speakers(680 + j)
        slots = {0: _canon_turn(j, sp[0], canon)}
        turns = _session(sp, 2 + j % 2, slots, filler_cls=cls)
        cur, span = _current_turn(j, sp[1], pron)
        turns[-1] = cur
        b.add("number_agreement", ["antecedent_prev_turn", "expected_abstain",
                                   "number_mismatch", "zero_survivors",
                                   f"class:{cls}"],
              pron, turns, ABSTAIN, span=span,
              rationale="singular pronoun never resolves to a plural canon")
    for j in range(6):  # it + plural only
        canon = PLURALS[(j + 5) % len(PLURALS)]
        sp = _pair_speakers(690 + j)
        slots = {0: _canon_turn(j, sp[0], canon)}
        turns = _session(sp, 2 + j % 2, slots, filler_cls="neut")
        cur, span = _current_turn(j, sp[1], IT[j % 3])
        turns[-1] = cur
        b.add("number_agreement", ["antecedent_prev_turn", "expected_abstain",
                                   "number_mismatch", "zero_survivors",
                                   "class:neut"],
              IT[j % 3], turns, ABSTAIN, span=span,
              rationale="it never resolves to a plural canon")
    for j in range(6):  # he/she + collective only
        canon = COLLECTIVES[(j + 3) % len(COLLECTIVES)]
        pron = (M_PRON + F_PRON)[j % 8]
        cls = _class_of(pron)
        sp = _pair_speakers(700 + j)
        slots = {0: _canon_turn(j, sp[0], canon)}
        turns = _session(sp, 2 + j % 2, slots, filler_cls=cls)
        cur, span = _current_turn(j, sp[1], pron)
        turns[-1] = cur
        b.add("number_agreement", ["antecedent_prev_turn", "expected_abstain",
                                   "collective_mismatch", "zero_survivors",
                                   f"class:{cls}"],
              pron, turns, ABSTAIN, span=span,
              rationale="collectives take it/they, never he/she")
    for j in range(4):  # they + plural -> resolve
        canon = PLURALS[(j + 8) % len(PLURALS)]
        sp = _pair_speakers(710 + j)
        slots = {0: _canon_turn(j, sp[0], canon)}
        turns = _session(sp, 2 + j % 2, slots, filler_cls="epicene")
        cur, span = _current_turn(j, sp[1], THEY[j % 5])
        turns[-1] = cur
        b.add("number_agreement", ["antecedent_prev_turn", "expected_resolve",
                                   "class:epicene"],
              THEY[j % 5], turns, canon, span=span,
              rationale="they resolves to a plural canon")


def _s_speaker(b: _Builder) -> None:
    """Speaker alternation: i -> speaker, you -> addressee, we -> group (42)."""
    # 10 i-forms -> speaker canon; a third-party canon sits in the
    # previous turn — "i" never inherits it.
    for j in range(10):
        form = FIRST[j % 5]
        name = NAMES[j % len(NAMES)]
        sp = _pair_speakers(720 + j)
        raw = j % 4 == 0  # a quarter exercise the speaker_canon key path
        cur, span = _current_turn(j, sp[1], form, raw_speaker=raw)
        turns = [_canon_turn(j, sp[0], name), cur]
        b.add("speaker_deixis", ["expected_resolve", "i_never_inherits",
                                 "speaker_referent", "class:first_sg"],
              form, turns, sp[1], span=span,
              rationale="first-person forms resolve to the speaker canon, never a prior mention")
    # 5 i with no speaker canon -> abstain (never inherits the mention).
    for j in range(5):
        form = FIRST[j % 5]
        name = NAMES[(j + 12) % len(NAMES)]
        sp = _pair_speakers(740 + j)
        turns = [_canon_turn(j, sp[0], name),
                 _turn(None, _mention_text(form, j))]
        b.add("speaker_deixis", ["expected_abstain", "i_never_inherits",
                                 "no_speaker", "no_antecedent",
                                 "class:first_sg"],
              form, turns, ABSTAIN, intended=None,
              rationale="no speaker canon; i abstains rather than inheriting a prior canon")
    # 8 you -> unique addressee.
    for j in range(8):
        form = SECOND[j % 4]
        sp = _pair_speakers(750 + j)
        turns = [_empty_turn(j, sp[0]), _empty_turn(j + 1, sp[0])]
        cur, span = _current_turn(j, sp[1], form)
        turns.append(cur)
        b.add("speaker_deixis", ["expected_resolve", "addressee",
                                 "speaker_referent", "class:second"],
              form, turns, sp[0], span=span,
              rationale="you resolves to the unique other speaker in the window")
    # 6 you -> two other speakers in window -> abstain.
    for j in range(6):
        sp = _pair_speakers(770 + j, 3)
        turns = [_empty_turn(j, sp[1]), _empty_turn(j + 1, sp[2])]
        cur, span = _current_turn(j, sp[0], SECOND[j % 4])
        turns.append(cur)
        b.add("speaker_deixis", ["expected_abstain", "addressee",
                                 "ambiguous_addressee", "class:second"],
              SECOND[j % 4], turns, ABSTAIN, span=span,
              rationale="two distinct other speakers in window -> ambiguous addressee")
    # 4 you -> no other speaker -> abstain (monologue).
    for j in range(4):
        sp = _pair_speakers(780 + j, 1)
        turns = [_empty_turn(j, sp[0]), _empty_turn(j + 1, sp[0])]
        cur, span = _current_turn(j, sp[0], SECOND[j % 4])
        turns.append(cur)
        b.add("speaker_deixis", ["expected_abstain", "addressee",
                                 "no_addressee", "class:second"],
              SECOND[j % 4], turns, ABSTAIN, span=span,
              rationale="no other speaker exists in the window")
    # 3 you -> the only other speaker is beyond the lookback -> abstain.
    for j in range(3):
        sp = _pair_speakers(790 + j)
        turns = [_empty_turn(j, sp[0])] + \
            [_empty_turn(j + 1 + k, sp[1]) for k in range(LOOKBACK + 1)]
        cur, span = _current_turn(j, sp[1], "you")
        turns[-1] = cur
        b.add("speaker_deixis", ["expected_abstain", "addressee",
                                 "no_addressee", "antecedent_beyond_window",
                                 "class:second"],
              "you", turns, ABSTAIN, span=span,
              rationale="the only other speaker sits outside the window")
    # 6 we/us/our/ours/ourselves -> group reference -> abstain.
    for j in range(6):
        form = FIRST_PL[j % 5]
        sp = _pair_speakers(800 + j)
        name = NAMES[(j + 20) % len(NAMES)]
        cur, span = _current_turn(j, sp[1], form)
        turns = [_canon_turn(j, sp[0], name), cur]
        b.add("speaker_deixis", ["expected_abstain", "group_reference",
                                 "class:first_pl"],
              form, turns, ABSTAIN, span=span,
              rationale="we/us/our is a group referent; never resolves to one canon")


def _s_unsupported(b: _Builder) -> None:
    """Forms the sieve deliberately does not handle -> abstain (10)."""
    for j, form in enumerate(UNSUPPORTED):
        sp = _pair_speakers(820 + j)
        canon = NAMES[j % len(NAMES)] if j % 2 else THINGS[j % len(THINGS)]
        cur, span = _current_turn(j, sp[1], form)
        turns = [_canon_turn(j, sp[0], canon), cur]
        b.add("unsupported_form", ["antecedent_prev_turn", "expected_abstain",
                                   "class:unsupported"],
              form, turns, ABSTAIN, span=span,
              rationale=f"'{form}' is an unsupported mention class -> honest abstain")


def _s_descriptions(b: _Builder) -> None:
    """Definite descriptions — unique attribute match resolves (40)."""
    # (canon template, description mention, zero-overlap rival template)
    desc_bank = [
        ("{n} from work", "the woman from work", "{r} from school"),
        ("{n} from work", "the man from work", "{r} from school"),
        ("{n} from accounting", "the guy from accounting", "{r} from sales"),
        ("{n} from accounting", "the woman from accounting", "{r} from sales"),
        ("{n} from the gym", "the coach from the gym", "{r} from the office"),
        ("{n} with the dog", "the woman with the dog", "{r} with the cat"),
        ("{n} from downtown", "the neighbor from downtown", "{r} from uptown"),
        ("{n} from the night shift", "the nurse from the night shift", "{r} from the day shift"),
        ("{n} in accounting", "the man in accounting", "{r} in sales"),
        ("{n} from the clinic", "the nurse from the clinic", "{r} from the hospital"),
        ("{n} from work", "the manager from work", "{r} from school"),
        ("{n} with the dog", "the guy with the dog", "{r} with the cat"),
        ("{n} from the gym", "the trainer from the gym", "{r} from the studio"),
        ("{n} from the night shift", "the nurse from the night shift", "{r} from the clinic"),
    ]
    for j, (ct, desc, rt) in enumerate(desc_bank):
        name = NAMES[(j * 3) % len(NAMES)]
        rival = NAMES[(j * 3 + 17) % len(NAMES)]
        canon = ct.format(n=name)
        rival_c = rt.format(r=rival)
        sp = _pair_speakers(840 + j)
        cur, span = _current_turn(j, sp[1], desc)
        turns = [_pair_turn(j, sp[0], canon, rival_c), cur]
        b.add("description", ["antecedent_prev_turn", "expected_resolve",
                              "unique_attribute", "class:description"],
              desc, turns, canon, span=span,
              rationale=f"'{desc}' discriminates '{canon}' on attributes")
    # 8 attribute ties -> abstain.
    for j in range(8):
        a = f"{NAMES_M[j]} from work"
        c = f"{NAMES_M[(j + 5) % len(NAMES_M)]} from work"
        desc = ("the guy from work", "the man from work")[j % 2]
        sp = _pair_speakers(860 + j)
        cur, span = _current_turn(j, sp[1], desc)
        turns = [_pair_turn(j, sp[0], a, c), cur]
        b.add("description", ["antecedent_prev_turn", "expected_abstain",
                              "attribute_tie", "class:description"],
              desc, turns, ABSTAIN, span=span,
              rationale="two candidates tie on attribute overlap -> abstain")
    # 6 zero-overlap -> abstain.
    for j in range(6):
        canon = NAMES[j % len(NAMES)] if j % 2 else THINGS[j % len(THINGS)]
        desc = desc_bank[j][1]
        sp = _pair_speakers(880 + j)
        cur, span = _current_turn(j, sp[1], desc)
        turns = [_canon_turn(j, sp[0], canon), cur]
        b.add("description", ["antecedent_prev_turn", "expected_abstain",
                              "no_attribute_match", "class:description"],
              desc, turns, ABSTAIN, span=span,
              rationale="no candidate shares an attribute term -> abstain")
    # plural descriptions.
    pl_bank = [("the brothers", "the two brothers"),
               ("the sisters", "the two sisters"),
               ("the cousins", "the three cousins"),
               ("the twins", "the two twins")]
    for j, (canon, desc) in enumerate(pl_bank):  # 4 resolve
        rival = NAMES_M[(j + 3) % len(NAMES_M)]
        sp = _pair_speakers(890 + j)
        cur, span = _current_turn(j, sp[1], desc)
        turns = [_pair_turn(j, sp[0], canon, rival), cur]
        b.add("description", ["antecedent_prev_turn", "expected_resolve",
                              "plural_description", "class:description"],
              desc, turns, canon, span=span,
              rationale="plural-marked description matches the plural canon only")
    for j, (_pl, desc) in enumerate(pl_bank):  # 4 singular-only -> abstain
        canon = NAMES_M[(j + 7) % len(NAMES_M)]
        sp = _pair_speakers(900 + j)
        cur, span = _current_turn(j, sp[1], desc)
        turns = [_canon_turn(j, sp[0], canon), cur]
        b.add("description", ["antecedent_prev_turn", "expected_abstain",
                              "plural_description", "class:description"],
              desc, turns, ABSTAIN, span=span,
              rationale="plural-marked description excludes the singular canon")
    # 4 gender-filtered descriptions -> abstain.
    for j in range(4):
        canon = GENDER_F[(j + 4) % len(GENDER_F)]
        desc = ("the man", "the man from work", "the guy", "the waiter")[j]
        sp = _pair_speakers(910 + j)
        cur, span = _current_turn(j, sp[1], desc)
        turns = [_canon_turn(j, sp[0], canon), cur]
        b.add("description", ["antecedent_prev_turn", "expected_abstain",
                              "gender_filter", "class:description"],
              desc, turns, ABSTAIN, span=span,
              rationale="masculine description filters out the female canon")


def _s_zero_candidates(b: _Builder) -> None:
    """No canon mentions anywhere in the window -> abstain (8)."""
    prons = ("he", "she", "they", "it", "his", "her", "their", "its")
    for j, pron in enumerate(prons):
        sp = _pair_speakers(920 + j, 2 + (j % 2))
        turns = [_empty_turn(j + k, sp[k % len(sp)])
                 for k in range(2 + j % 3)]
        cur, span = _current_turn(j, sp[(len(turns)) % len(sp)], pron)
        turns.append(cur)
        b.add("zero_candidates", ["expected_abstain", "no_antecedent",
                                  f"class:{_class_of(pron)}"],
              pron, turns, ABSTAIN, span=span,
              rationale="empty candidate set -> abstain")


def _s_they_extras(b: _Builder) -> None:
    """Epicene coverage: speaker exclusion, rarer forms (12)."""
    # 4 speaker-exclusion: current speaker is one of two candidates.
    for j in range(4):
        sp = _pair_speakers(940 + j)
        other = NAMES_M[(j + 5) % len(NAMES_M)]
        pron = THEY[j % 5]
        cur, span = _current_turn(j, sp[0], pron)
        turns = [_empty_turn(j, sp[0]),
                 _pair_turn(j + 1, sp[1], sp[0], other), cur]
        b.add("they_epicene", ["antecedent_prev_turn", "expected_resolve",
                               "speaker_exclusion", "class:epicene"],
              pron, turns, other, span=span,
              rationale="current speaker excluded; the other canon is the sole survivor")
    # 4 single-canon they with reflexive/standalone-possessive forms.
    for j, pron in enumerate(("themselves", "themself", "theirs", "them")):
        canon = NAMES[(j + 25) % len(NAMES)]
        sp = _pair_speakers(950 + j)
        cur, span = _current_turn(j, sp[1], pron)
        turns = [_canon_turn(j, sp[0], canon), cur]
        b.add("they_epicene", ["antecedent_prev_turn", "expected_resolve",
                               "single_candidate", "class:epicene"],
              pron, turns, canon, span=span,
              rationale=f"epicene {pron} resolves to the only candidate")
    # 4 they + lone singular thing -> abstain (pure things are not they).
    for j in range(4):
        canon = THINGS[(j + 12) % len(THINGS)]
        sp = _pair_speakers(960 + j)
        cur, span = _current_turn(j, sp[1], THEY[j % 5])
        turns = [_canon_turn(j, sp[0], canon), cur]
        b.add("they_epicene", ["antecedent_prev_turn", "expected_abstain",
                               "zero_survivors", "class:epicene"],
              THEY[j % 5], turns, ABSTAIN, span=span,
              rationale="a lone singular thing is 'it', never 'they'")


def _s_context_maps(b: _Builder) -> None:
    """Explicit-statement maps: gender / kind / number kwargs (12)."""
    # 5 gender map: an explicit statement grounds gender for a bare name;
    # the verified name sits at d=1, the unverified rival at d=2.
    for j in range(5):
        name = NAMES_U[j % len(NAMES_U)]
        rival = NAMES_U[(j + 4) % len(NAMES_U)]
        g = "f" if j % 2 else "m"
        pron = F_PRON[j % 4] if g == "f" else M_PRON[j % 4]
        noun = "a woman" if g == "f" else "a man"
        sp = _pair_speakers(970 + j)
        cur, span = _current_turn(j, sp[1], pron)
        turns = [_canon_turn(j, sp[0], rival),
                 _turn(sp[0],
                       f"{_surf(name)}, {noun}, called earlier", [name]),
                 cur]
        b.add("context_map", ["antecedent_prev_turn", "expected_resolve",
                              "explicit_gender", f"class:{_class_of(pron)}"],
              pron, turns, name, span=span,
              context={"gender": {name: g}},
              rationale="explicit statement grounds gender; verified winner wins the margin")
    # 4 kind map: thing-kind on a non-lexicon canon enables 'it'.
    for j, canon in enumerate(("the prototype", "the widget",
                               "the gadget", "the contraption")):
        rival = NAMES_F[(j + 2) % len(NAMES_F)]
        sp = _pair_speakers(980 + j)
        cur, span = _current_turn(j, sp[1], IT[j % 3])
        turns = [_pair_turn(j, sp[0], canon, rival), cur]
        b.add("context_map", ["antecedent_prev_turn", "expected_resolve",
                              "explicit_kind", "class:neut"],
              IT[j % 3], turns, canon, span=span,
              context={"kind": {canon: "thing"}},
              rationale="kind map supplies thing evidence; person rival filtered")
    # 3 number map: sg override on a plural-looking canon.
    for j, canon in enumerate(("the smiths", "the joneses", "the brooks")):
        sp = _pair_speakers(990 + j)
        cur, span = _current_turn(j, sp[1], M_PRON[j % 4])
        turns = [_canon_turn(j, sp[0], canon), cur]
        b.add("context_map", ["antecedent_prev_turn", "expected_resolve",
                              "explicit_number", "class:masc"],
              M_PRON[j % 4], turns, canon, span=span,
              context={"number": {canon: "sg"}},
              rationale="number map overrides the plural-looking head token")


# ---------------------------------------------------------------------------
# Inventory verification (generation-time; catches lexicon drift)
# ---------------------------------------------------------------------------

def _verify_inventory() -> None:
    """Assert canon pools match the sieve's agreement lexicons — this
    guards against silent drift between this fixture and
    ``coref_sieve``'s noun tables.  Labels are still by construction;
    this only checks that the pools mean what the builders assume."""
    from verbatim.enrichment import coref_sieve as cs
    empty: Dict[str, str] = {}
    no_speakers: frozenset = frozenset()
    for c in NAMES:
        t = cs._canon_traits(c, no_speakers, empty, empty, empty)
        assert not t.person and not t.thing and not t.plural \
            and not t.collective and t.gender is None, \
            ("bare name drifted", c)
    for c in GENDER_M:
        t = cs._canon_traits(c, no_speakers, empty, empty, empty)
        assert t.gender == "m" and t.person and not t.plural \
            and not t.collective, ("GENDER_M drifted", c)
    for c in GENDER_F:
        t = cs._canon_traits(c, no_speakers, empty, empty, empty)
        assert t.gender == "f" and t.person and not t.plural \
            and not t.collective, ("GENDER_F drifted", c)
    for c in PERSON_N:
        t = cs._canon_traits(c, no_speakers, empty, empty, empty)
        assert t.person and t.gender is None and not t.plural \
            and not t.collective and not t.thing, ("PERSON_N drifted", c)
    for c in THINGS:
        t = cs._canon_traits(c, no_speakers, empty, empty, empty)
        assert t.thing and not t.person and not t.plural \
            and not t.collective and t.gender is None, ("THINGS drifted", c)
    for c in PLURALS:
        t = cs._canon_traits(c, no_speakers, empty, empty, empty)
        assert t.plural, ("PLURALS drifted", c)
    for c in COLLECTIVES:
        t = cs._canon_traits(c, no_speakers, empty, empty, empty)
        assert t.collective and not t.plural and not t.person, \
            ("COLLECTIVES drifted", c)
    spk = frozenset(SPEAKERS)
    for c in NAMES + GENDER_M + GENDER_F + PERSON_N + THINGS + PLURALS \
            + COLLECTIVES:
        assert c not in spk, ("canon collides with a speaker", c)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def generate() -> List[Dict[str, Any]]:
    """Enumerate the full fixture.  Pure enumeration — deterministic."""
    _verify_inventory()
    b = _Builder()
    _s_single_prev(b)
    _s_same_turn(b)
    _s_deep_resolve(b)
    _s_deep_abstain(b)
    _s_beyond_window(b)
    _s_traps(b)
    _s_mixed_class(b)
    _s_gender(b)
    _s_number(b)
    _s_speaker(b)
    _s_unsupported(b)
    _s_descriptions(b)
    _s_zero_candidates(b)
    _s_they_extras(b)
    _s_context_maps(b)
    ids = [c["case_id"] for c in b.cases]
    assert len(ids) == len(set(ids)), "duplicate case_id"
    return b.cases


def header(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "record": "fixture",
        "generator": GENERATOR_ID,
        "spec": "V7-13.20",
        "arm_gate": "SPEC_V7_5 Q8",
        "lookback": LOOKBACK,
        "cases": len(cases),
        "abstain_label": ABSTAIN,
    }


def serialize(cases: Sequence[Dict[str, Any]]) -> str:
    lines = [json.dumps(header(cases), ensure_ascii=False, sort_keys=True)]
    lines += [json.dumps(c, ensure_ascii=False, sort_keys=True)
              for c in cases]
    return "\n".join(lines) + "\n"


def write_fixture(path: Optional[str] = None) -> str:
    """Write the JSONL fixture; returns the path written."""
    path = path or FIXTURE_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(serialize(generate()))
    return path


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _audit(cases: Sequence[Dict[str, Any]]) -> None:
    """Self-audit: run the sieve over every case and report statuses +
    label disagreements on stderr.  Never rewrites labels — the fixture
    ships with construction-time labels; disagreements are findings."""
    from collections import Counter
    from verbatim.enrichment import coref_sieve as cs
    stats = Counter()
    wrong = []
    for c in cases:
        ctx = dict(c.get("context") or {})
        out = cs.explain(c["mention"], c["unit_index"], c["turns"],
                         lookback=LOOKBACK, **ctx)
        stats[out["status"]] += 1
        pred = out["canon"]
        exp = None if c["expected"] == ABSTAIN else c["expected"]
        if pred != exp:
            wrong.append((c["case_id"], c["stratum"], c["mention"],
                          exp, pred, out["status"]))
    print("== audit: sieve statuses ==", file=sys.stderr)
    for s, n in stats.most_common():
        print(f"  {s:32s} {n}", file=sys.stderr)
    resolved = sum(n for s, n in stats.items()
                   if s.startswith("resolved")
                   or s in ("identity", "speaker", "addressee"))
    print(f"  resolved predictions: {resolved}; "
          f"label disagreements: {len(wrong)}", file=sys.stderr)
    for w in wrong:
        print("  DISAGREE:", w, file=sys.stderr)


def main() -> None:
    path = write_fixture()
    cases = generate()
    from collections import Counter
    strata = Counter(c["stratum"] for c in cases)
    deep = sum(1 for c in cases
               if c["expected"] != ABSTAIN
               and (c["antecedent_distance"] or 0) >= 2)
    traps = sum(1 for c in cases
                if "two_candidate_trap" in c["strata"]
                and c["expected"] == ABSTAIN)
    print(f"wrote {len(cases)} cases -> {path}")
    print(f"  deep (d>=2) resolve cases: {deep} (need >=100)")
    print(f"  two-candidate traps:       {traps} (need >=50)")
    for s in sorted(strata):
        print(f"  {s:24s} {strata[s]}")
    if "--audit" in sys.argv:
        _audit(cases)


if __name__ == "__main__":
    main()
