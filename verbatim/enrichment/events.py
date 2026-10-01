"""Deterministic event extraction — ``event/v1`` (SPEC_V7 §32.10, V7-09.08).

The Chronos mechanism without a model: from one unit's ``norm/v2``
analysis, extract ``(subject_canon, predicate_lemma, object_text,
polarity, occurred, pins)`` tuples — a deterministic event calendar.

Rules over the normalized term stream:

- **Predicate matching is lemma-based.** §32.10's predicate lexicon is
  organized into families ("move/relocate", "start/begin/join/enroll/
  sign up", …); each lexeme's head verb is expanded into its inflections
  (generated regular forms ∪ a pinned irregular table) and the emitted
  ``predicate_lemma`` is the *family-canonical* lemma, so "relocated to
  Oslo" and "moved to Berlin" share one event predicate. The matched
  surface lexeme is recorded in ``pins["lexeme"]`` and ``rule_id``.
- **Subject resolution** (§32.10): first-person pronouns → the speaker
  canon; second-person → the addressee canon when the caller knows it;
  third-person pronouns and definite descriptions ("the manager") → the
  caller-supplied ``sieve`` callable (adapting ``coref_sieve/v1`` —
  ``resolve_antecedent`` with its unit/session context bound); explicit
  names → ``canon_fn`` (default: lazy ``entities_v2.canon``). Anything
  unresolved stays ``subject_canon=None`` (the spec's ``subject=unknown``).
  A coordinated subject ("alice and bob moved") emits one tuple per
  subject; a predicate joined by a coordinator or following a light
  verb ("i moved and started", "i got engaged") reuses the subject
  resolved for the preceding predicate.
- **Polarity** comes from the local negation/modal window between the
  subject and the predicate: ``not``/``never``/``n't``-clitic ``not`` →
  ``"negate"``; modals and infinitive frames ("will move", "want to
  move") → ``"hypothetical"``; otherwise ``"affirm"``. The V5 polarity
  vocabulary is retained; hypothetical tuples are emitted, not dropped —
  they are stated plans, and the honest label is theirs.
- **Every element is span-pinned** to byte offsets into the unit's raw
  text, taken from the ``NormTerm`` offsets (which §32.1 pins to source
  bytes). When ``raw_text`` is supplied the pins are verified against the
  UTF-8 bytes (bounds + decodability); tuples that cannot be fully
  verified are dropped and counted. Reused subjects pin the original
  mention span.
- ``occurred`` is passed through verbatim — temporal resolution belongs
  to ``temporal/v2``, not this extractor.

Pure stdlib, deterministic, no store access. Provisional
``provisional/v7-r0`` (V7-32.01); the tag rides on ``FORMULA_STATUS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from verbatim.core.types_v7 import (
    EventTuple,
    IntervalUs,
    NormAnalysis,
    NormTerm,
)

EXTRACTOR_ID = "event/v1"
FORMULA_STATUS = "provisional/v7-r0"

#: Cumulative honest counters. ``dropped_unpinned`` is the §32.10
#: "tuples failing pin verification are dropped" counter; callers and
#: tests can also pass a per-call ``stats`` dict to ``extract_events``.
COUNTERS: dict[str, int] = {
    "units": 0,
    "predicates_matched": 0,
    "emitted": 0,
    "dropped_unpinned": 0,
    "canon_unavailable": 0,
}


def reset_counters() -> None:
    for k in COUNTERS:
        COUNTERS[k] = 0


# ---------------------------------------------------------------------------
# Predicate lexicon (§32.10) — family-canonical lemmas
# ---------------------------------------------------------------------------

_VOWELS = frozenset("aeiou")


@dataclass(frozen=True)
class _Lexeme:
    """One predicate surface rule. ``head`` is the inflecting verb;
    ``particles`` are required trailing particles in order ("sign up" →
    head ``sign``, particles ``("up",)``). Gates:

    - ``aux_left``: a term in this set must appear ≤ 3 terms to the left
      (adjective predicates like "sick" in "got sick" / "was born");
      the predicate span covers aux..head.
    - ``obj_gate``: first content term of the object (skipping
      possessive determiners) must be in the set ("get" + pet noun →
      adopt; "have" + birthday → celebrate).
    - ``obj_num``: first object term must be a bare number ("turned 30").
    - ``obj_personish``: first post-predicate term must be person-like —
      a pronoun, a possessive, or a bare name ("saw alice" fires,
      "saw the movie" does not).
    - ``subj_person``: subject must resolve to a person form — pronoun
      or explicit name — never a description ("our date was fun" is not
      a dating event).
    - ``reject_next``: the term immediately after the head must not be
      in the set ("grew sick" is the sick event, not a gardening one).
    """

    lexeme: str
    family: str
    head: str
    particles: tuple[str, ...] = ()
    forms: tuple[str, ...] = ()
    aux_left: Optional[frozenset[str]] = None
    obj_gate: Optional[frozenset[str]] = None
    obj_num: bool = False
    obj_personish: bool = False
    subj_person: bool = False
    reject_next: Optional[frozenset[str]] = None


#: Irregular English forms the generator cannot derive. Extra forms are
#: always safe — a surface maps onto its head and nothing else.
_IRREGULAR: dict[str, tuple[str, ...]] = {
    "go": ("went", "gone"),
    "see": ("saw", "seen"),
    "meet": ("met",),
    "run": ("ran",),
    "win": ("won",),
    "lose": ("lost",),
    "break": ("broke", "broken"),
    "buy": ("bought",),
    "sell": ("sold",),
    "get": ("got", "gotten"),
    "take": ("took", "taken"),
    "throw": ("threw", "thrown"),
    "write": ("wrote", "written"),
    "draw": ("drew", "drawn"),
    "grow": ("grew", "grown"),
    "begin": ("began", "begun"),
    "leave": ("left",),
    "quit": ("quit", "quitted", "quitting"),
    "read": ("read",),
    "have": ("has", "had"),
    "catch": ("caught",),
    "fly": ("flew", "flown"),
    "come": ("came",),
    "lay": ("laid",),
    "fall": ("fell", "fallen"),
    "feel": ("felt",),
    "build": ("built",),
    "wed": ("wed", "wedded", "wedding"),
    "learn": ("learnt",),
    "drive": ("drove", "driven"),
    "ring": ("rang", "rung"),
    "binge": ("binged", "bingeing", "binging"),
    "enroll": ("enrolled", "enrolling"),
    "enrol": ("enrolled", "enrolling", "enroled", "enroling"),
    "travel": ("travelled", "travelling"),
    "cancel": ("cancelled", "cancelling"),
    "remodel": ("remodelled", "remodelling"),
    "dm": ("dmed", "dming", "dmmed"),
    "sow": ("sown", "sowed"),
    "sick": ("sick", "sicker", "sickest"),
    "ill": ("ill", "iller", "illest"),
    "unwell": ("unwell",),
    "born": ("born", "borne"),
    "die": ("died", "dying", "dies"),
    "pass": ("passed",),
    "redo": ("redid", "redone", "redoes"),
    "study": ("studied", "studying"),
    "show": ("showed", "shown"),
    "split": ("split", "splitting"),
    "step": ("stepped", "stepping"),
    "drop": ("dropped", "dropping"),
    "wrap": ("wrapped", "wrapping"),
    "premiere": ("premiered",),
}


def _cvc_double(head: str) -> bool:
    """Short single-syllable CVC verbs double the final consonant
    (run→running, drop→dropped). Bounded to len ≤ 4 so "visit",
    "enroll", "cancel" stay single — those get explicit doubled forms
    via ``_IRREGULAR`` when legal."""
    return (
        2 < len(head) <= 4
        and head[-1] in "bcdfghjklmnpqrstvz"
        and head[-2] in _VOWELS
        and head[-3] not in _VOWELS
    )


def _gen_forms(head: str) -> tuple[str, ...]:
    """Regular inflections of ``head``: base, 3rd-person, past/-ed, -ing,
    unioned with the pinned irregular table."""
    out = {head}
    if head.endswith(("s", "x", "z", "ch", "sh", "o")):
        out.add(head + "es")
    elif head.endswith("y") and len(head) > 1 and head[-2] not in _VOWELS:
        out.add(head[:-1] + "ies")
    else:
        out.add(head + "s")
    if head.endswith("e"):
        out.add(head + "d")
    elif head.endswith("y") and len(head) > 1 and head[-2] not in _VOWELS:
        out.add(head[:-1] + "ied")
    elif _cvc_double(head):
        out.add(head + head[-1] + "ed")
    else:
        out.add(head + "ed")
    if head.endswith("ie"):
        out.add(head[:-2] + "ying")
    elif head.endswith("e") and not head.endswith(("ee", "ye", "oe")):
        out.add(head[:-1] + "ing")
    elif _cvc_double(head):
        out.add(head + head[-1] + "ing")
    else:
        out.add(head + "ing")
    out.update(_IRREGULAR.get(head, ()))
    return tuple(sorted(out))


# --- object gates ----------------------------------------------------------

#: "adopt/get (a pet)" — pet nouns.
_PETS = frozenset({
    "dog", "dogs", "cat", "cats", "puppy", "puppies", "kitten",
    "kittens", "pet", "pets", "bird", "birds", "fish", "hamster",
    "rabbit", "bunny", "parrot", "turtle", "lizard", "snake", "gecko",
    "ferret", "horse", "pony", "guinea",
})

#: "attend/go to (an event)" — event nouns distinguish "went to a
#: concert" (attend) from "went to berlin" (visit).
_EVENT_NOUNS = frozenset({
    "meeting", "meetings", "conference", "concert", "concerts", "party",
    "wedding", "weddings", "funeral", "class", "course", "workshop",
    "seminar", "event", "events", "rally", "match", "matches", "game",
    "games", "show", "shows", "festival", "ceremony", "lecture",
    "gathering", "meetup", "summit", "gig", "practice", "rehearsal",
    "interview", "appointment", "session", "service", "mass", "parade",
    "protest", "demo", "talk", "keynote", "retreat", "fair", "expo",
})

#: "learn/take (a class)".
_CLASS_NOUNS = frozenset({
    "class", "classes", "course", "courses", "lesson", "lessons",
    "workshop", "seminar", "training", "degree", "program", "programme",
    "certification", "bootcamp", "tutorial", "lecture", "module",
})

#: "host/throw (a party)".
_PARTY_NOUNS = frozenset({
    "party", "parties", "celebration", "reception", "shower", "bash",
    "dinner", "barbecue", "bbq", "potluck", "gathering", "fete",
    "brunch", "picnic", "ceremony",
})

#: "celebrate/have (a birthday)".
_BIRTHDAY_NOUNS = frozenset({
    "birthday", "birthdays", "anniversary", "anniversaries",
})

#: "injure/break (a bone)".
_BODY_NOUNS = frozenset({
    "arm", "arms", "leg", "legs", "bone", "bones", "wrist", "ankle",
    "rib", "ribs", "finger", "fingers", "toe", "toes", "hip", "shoulder",
    "neck", "back", "hand", "hands", "foot", "feet", "thumb", "skull",
    "jaw", "elbow", "knee", "knees", "nose", "collarbone", "tailbone",
    "tooth", "teeth",
})

#: "get sick / catch (an illness)".
_ILLNESS_NOUNS = frozenset({
    "cold", "flu", "covid", "fever", "virus", "infection", "bug",
    "pneumonia", "mono", "measles", "mumps", "chickenpox", "bronchitis",
    "strep", "norovirus", "shingles", "rsv",
})

#: "hire/offer (a job)".
_JOB_NOUNS = frozenset({
    "job", "jobs", "position", "positions", "role", "roles",
    "promotion", "promotions", "contract", "contracts", "offer",
    "offers", "gig", "internship",
})

#: "renovate/paint (a room)" — household objects pull "paint" out of the
#: art family into renovation.
_HOME_NOUNS = frozenset({
    "house", "room", "rooms", "kitchen", "bathroom", "bedroom",
    "bedrooms", "wall", "walls", "fence", "deck", "garage", "basement",
    "ceiling", "ceilings", "cabinet", "cabinets", "shed", "porch",
    "apartment", "flat", "office", "door", "doors", "floor", "floors",
    "stairs", "roof", "hallway", "attic", "driveway",
})

#: "book/confirm (a reservation)".
_REZ_NOUNS = frozenset({
    "reservation", "reservations", "booking", "bookings", "flight",
    "flights", "hotel", "hotels", "table", "tables", "ticket", "tickets",
    "room", "rooms", "appointment", "appointments",
})

#: Auxiliaries that license the adjective predicates ("sick", "ill",
#: "unwell"): get/be/become/feel/fall/seem/look/sound/grow/come.
_SICK_AUX = frozenset({
    "get", "got", "gets", "getting", "gotten",
    "be", "am", "is", "are", "was", "were", "been", "being",
    "become", "became", "becomes", "becoming",
    "feel", "feels", "felt", "feeling",
    "fall", "falls", "fell", "fallen", "falling",
    "seem", "seems", "seemed", "seeming",
    "look", "looks", "looked", "looking",
    "sound", "sounds", "sounded", "sounding",
    "come", "came", "comes", "coming",
    "grow", "grew", "grows", "grown", "growing",
})

_BORN_AUX = frozenset({
    "be", "am", "is", "are", "was", "were", "been", "being",
})


#: The §32.10 families, "extended conservatively". Order matters only as
#: a deterministic tiebreak — candidate ordering is (particle count,
#: gated-first, declaration index).
_LEXICON: tuple[_Lexeme, ...] = (
    # move/relocate -----------------------------------------------------
    _Lexeme("move", "move", "move"),
    _Lexeme("relocate", "move", "relocate"),
    _Lexeme("migrate", "move", "migrate"),
    _Lexeme("move in", "move", "move", ("in",)),
    _Lexeme("move out", "move", "move", ("out",)),
    # start/begin/join/enroll/sign up -----------------------------------
    _Lexeme("start", "start", "start"),
    _Lexeme("begin", "start", "begin"),
    _Lexeme("join", "start", "join"),
    _Lexeme("enroll", "start", "enroll"),
    _Lexeme("enrol", "start", "enrol"),
    _Lexeme("sign up", "start", "sign", ("up",)),
    _Lexeme("commence", "start", "commence"),
    # quit/leave/resign/drop out ----------------------------------------
    _Lexeme("quit", "quit", "quit"),
    _Lexeme("leave", "quit", "leave"),
    _Lexeme("resign", "quit", "resign"),
    _Lexeme("drop out", "quit", "drop", ("out",)),
    _Lexeme("step down", "quit", "step", ("down",)),
    _Lexeme("walk out", "quit", "walk", ("out",)),
    # retire -------------------------------------------------------------
    _Lexeme("retire", "retire", "retire"),
    # graduate/finish/complete ------------------------------------------
    _Lexeme("graduate", "graduate", "graduate"),
    _Lexeme("finish", "graduate", "finish"),
    _Lexeme("complete", "graduate", "complete"),
    _Lexeme("wrap up", "graduate", "wrap", ("up",)),
    # marry/engage/divorce/date/break up ---------------------------------
    _Lexeme("marry", "marry", "marry"),
    _Lexeme("remarry", "marry", "remarry"),
    _Lexeme("wed", "marry", "wed"),
    _Lexeme("engage", "marry", "engage"),
    _Lexeme("divorce", "marry", "divorce"),
    _Lexeme("date", "marry", "date", subj_person=True),
    _Lexeme("break up", "marry", "break", ("up",)),
    _Lexeme("split up", "marry", "split", ("up",)),
    _Lexeme("propose", "marry", "propose", ("to",)),
    # buy/purchase/order/sell -------------------------------------------
    _Lexeme("buy", "buy", "buy"),
    _Lexeme("purchase", "buy", "purchase"),
    _Lexeme("order", "buy", "order"),
    _Lexeme("sell", "buy", "sell"),
    # visit/travel/go to/fly to/return from ------------------------------
    _Lexeme("visit", "visit", "visit"),
    _Lexeme("travel", "visit", "travel"),
    _Lexeme("fly to", "visit", "fly", ("to",)),
    _Lexeme("drive to", "visit", "drive", ("to",)),
    _Lexeme("head to", "visit", "head", ("to",)),
    _Lexeme("return from", "visit", "return", ("from",)),
    _Lexeme("return to", "visit", "return", ("to",)),
    _Lexeme("come back", "visit", "come", ("back",)),
    _Lexeme("arrive in", "visit", "arrive", ("in",)),
    _Lexeme("arrive at", "visit", "arrive", ("at",)),
    # the gated sibling of "go to" — event objects mean attendance,
    # anything else is a visit ("went to a concert" vs "went to berlin")
    _Lexeme("go to", "attend", "go", ("to",), obj_gate=_EVENT_NOUNS),
    _Lexeme("go to", "visit", "go", ("to",)),
    # adopt/get (a pet) ---------------------------------------------------
    _Lexeme("adopt", "adopt", "adopt"),
    _Lexeme("foster", "adopt", "foster", obj_gate=_PETS),
    _Lexeme("rescue", "adopt", "rescue", obj_gate=_PETS),
    _Lexeme("get", "adopt", "get", obj_gate=_PETS),
    # paint/draw/write/publish/record -------------------------------------
    _Lexeme("draw", "paint", "draw"),
    _Lexeme("sketch", "paint", "sketch"),
    _Lexeme("write", "paint", "write"),
    _Lexeme("publish", "paint", "publish"),
    _Lexeme("record", "paint", "record"),
    _Lexeme("compose", "paint", "compose"),
    _Lexeme("illustrate", "paint", "illustrate"),
    # "paint" is shared with renovate — the household gate decides; the
    # ungated (art) entry is declared first so sorting tries the gated
    # one first (gated before ungated in candidate order).
    _Lexeme("paint", "renovate", "paint", obj_gate=_HOME_NOUNS),
    _Lexeme("paint", "paint", "paint"),
    # run/race/compete/win/lose -------------------------------------------
    _Lexeme("run into", "meet", "run", ("into",)),
    _Lexeme("run", "run", "run"),
    _Lexeme("race", "run", "race"),
    _Lexeme("compete", "run", "compete"),
    _Lexeme("win", "run", "win"),
    _Lexeme("lose", "run", "lose"),
    _Lexeme("participate in", "run", "participate", ("in",)),
    # attend --------------------------------------------------------------
    _Lexeme("attend", "attend", "attend"),
    _Lexeme("show up", "attend", "show", ("up",)),
    # meet/see/reunite ----------------------------------------------------
    _Lexeme("meet up", "meet", "meet", ("up",)),
    _Lexeme("meet", "meet", "meet"),
    _Lexeme("see", "meet", "see", obj_personish=True),
    _Lexeme("reunite", "meet", "reunite"),
    _Lexeme("catch up", "meet", "catch", ("up",)),
    _Lexeme("bump into", "meet", "bump", ("into",)),
    # host/throw (a party) -------------------------------------------------
    _Lexeme("host", "host", "host"),
    _Lexeme("throw", "host", "throw", obj_gate=_PARTY_NOUNS),
    _Lexeme("organize", "host", "organize"),
    _Lexeme("organise", "host", "organise"),
    _Lexeme("arrange", "host", "arrange"),
    # celebrate/have (a birthday) -------------------------------------------
    _Lexeme("celebrate", "celebrate", "celebrate"),
    _Lexeme("have", "celebrate", "have", obj_gate=_BIRTHDAY_NOUNS),
    _Lexeme("turn", "celebrate", "turn", obj_num=True),
    # cook/bake -------------------------------------------------------------
    _Lexeme("cook", "cook", "cook"),
    _Lexeme("bake", "cook", "bake"),
    _Lexeme("grill", "cook", "grill"),
    _Lexeme("roast", "cook", "roast"),
    _Lexeme("brew", "cook", "brew"),
    # learn/take (a class) ----------------------------------------------------
    _Lexeme("learn", "learn", "learn"),
    _Lexeme("study", "learn", "study"),
    _Lexeme("take", "learn", "take", obj_gate=_CLASS_NOUNS),
    _Lexeme("practice", "learn", "practice"),
    _Lexeme("practise", "learn", "practise"),
    # volunteer/donate/mentor --------------------------------------------------
    _Lexeme("volunteer", "volunteer", "volunteer"),
    _Lexeme("donate", "volunteer", "donate"),
    _Lexeme("mentor", "volunteer", "mentor"),
    _Lexeme("tutor", "volunteer", "tutor"),
    _Lexeme("coach", "volunteer", "coach"),
    # get sick/recover/injure/diagnose ------------------------------------------
    _Lexeme("sick", "sick", "sick", aux_left=_SICK_AUX),
    _Lexeme("ill", "sick", "ill", aux_left=_SICK_AUX),
    _Lexeme("unwell", "sick", "unwell", aux_left=_SICK_AUX),
    _Lexeme("recover", "sick", "recover"),
    _Lexeme("injure", "sick", "injure"),
    _Lexeme("diagnose", "sick", "diagnose"),
    _Lexeme("break", "sick", "break", obj_gate=_BODY_NOUNS),
    _Lexeme("catch", "sick", "catch", obj_gate=_ILLNESS_NOUNS),
    # born / die — conservative life-event additions ----------------------------
    _Lexeme("born", "born", "born", aux_left=_BORN_AUX),
    _Lexeme("die", "die", "die"),
    _Lexeme("pass away", "die", "pass", ("away",)),
    # hire/promote/fire/interview -------------------------------------------------
    _Lexeme("hire", "hire", "hire"),
    _Lexeme("promote", "hire", "promote"),
    _Lexeme("demote", "hire", "demote"),
    _Lexeme("fire", "hire", "fire"),
    _Lexeme("interview", "hire", "interview"),
    _Lexeme("recruit", "hire", "recruit"),
    _Lexeme("lay off", "hire", "lay", ("off",)),
    _Lexeme("offer", "hire", "offer", obj_gate=_JOB_NOUNS),
    # launch/ship/release ------------------------------------------------------------
    _Lexeme("launch", "launch", "launch"),
    _Lexeme("ship", "launch", "ship"),
    _Lexeme("release", "launch", "release"),
    _Lexeme("deploy", "launch", "deploy"),
    _Lexeme("unveil", "launch", "unveil"),
    _Lexeme("premiere", "launch", "premiere"),
    # book/reserve/cancel --------------------------------------------------------------
    _Lexeme("book", "book", "book"),
    _Lexeme("reserve", "book", "reserve"),
    _Lexeme("cancel", "book", "cancel"),
    _Lexeme("reschedule", "book", "reschedule"),
    _Lexeme("confirm", "book", "confirm", obj_gate=_REZ_NOUNS),
    # call/text/email ---------------------------------------------------------------------
    _Lexeme("call", "call", "call"),
    _Lexeme("phone", "call", "phone"),
    _Lexeme("text", "call", "text"),
    _Lexeme("email", "call", "email"),
    _Lexeme("message", "call", "message"),
    _Lexeme("dm", "call", "dm"),
    _Lexeme("ring", "call", "ring", obj_personish=True),
    # read/watch/listen to -------------------------------------------------------------------
    _Lexeme("read", "read", "read"),
    _Lexeme("watch", "read", "watch"),
    _Lexeme("listen to", "read", "listen", ("to",)),
    _Lexeme("stream", "read", "stream"),
    _Lexeme("binge", "read", "binge"),
    # plant/grow/garden ------------------------------------------------------------------------
    _Lexeme("plant", "plant", "plant"),
    _Lexeme("grow", "plant", "grow",
            reject_next=frozenset({"sick", "ill", "unwell", "tired",
                                   "old", "older", "bored"})),
    _Lexeme("garden", "plant", "garden"),
    _Lexeme("harvest", "plant", "harvest"),
    _Lexeme("sow", "plant", "sow"),
    _Lexeme("prune", "plant", "prune"),
    # renovate/build/fix --------------------------------------------------------------------------
    _Lexeme("renovate", "renovate", "renovate"),
    _Lexeme("remodel", "renovate", "remodel"),
    _Lexeme("build", "renovate", "build"),
    _Lexeme("fix", "renovate", "fix"),
    _Lexeme("repair", "renovate", "repair"),
    _Lexeme("redo", "renovate", "redo"),
)

LEXICON_FAMILIES: tuple[str, ...] = tuple(
    dict.fromkeys(lx.family for lx in _LEXICON))
LEXEME_COUNT: int = len(_LEXICON)


def _build_head_index() -> dict[str, list[int]]:
    """surface form -> lexeme indexes, ordered longest-particles first,
    gated before ungated, then declaration order."""
    out: dict[str, list[int]] = {}
    for idx, lx in enumerate(_LEXICON):
        forms = lx.forms + _gen_forms(lx.head)
        for f in forms:
            out.setdefault(f, []).append(idx)
    for key, idxs in out.items():
        idxs.sort(key=lambda i: (
            -len(_LEXICON[i].particles),
            0 if (_LEXICON[i].obj_gate or _LEXICON[i].obj_num
                  or _LEXICON[i].obj_personish or _LEXICON[i].aux_left) else 1,
            i,
        ))
    return out


_HEAD_INDEX: dict[str, list[int]] = _build_head_index()
_ALL_FORMS: frozenset[str] = frozenset(_HEAD_INDEX)


# ---------------------------------------------------------------------------
# Word classes for the shallow clause walk
# ---------------------------------------------------------------------------

#: Auxiliaries / modals / negation / adverbs skipped when scanning left
#: for a subject — they form the polarity window. Light verbs
#: (get/become/feel/come…) are included so "i got engaged" resolves "i"
#: for the ``engaged`` predicate.
_AUX_VERBS = frozenset({
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing", "done",
    "will", "would", "can", "could", "shall", "should",
    "may", "might", "must", "ought",
    # n't-clitic stems: norm/v2 splits "didn't" → "didn" + "not"
    "didn", "don", "doesn", "isn", "aren", "wasn", "weren", "won", "wo",
    "couldn", "shouldn", "wouldn", "mustn", "needn", "daren", "ain",
    "shan", "mightn", "oughtn", "haven", "hasn", "hadn", "cannot",
    "cant", "dare", "need",
    # light verbs licensing adjective predicates / participles
    "get", "gets", "got", "getting", "gotten",
    "become", "became", "becomes", "becoming",
    "feel", "feels", "felt", "feeling",
    "seem", "seems", "seemed", "seeming",
    "look", "looks", "looked", "looking",
    "sound", "sounds", "sounded", "sounding",
    "remain", "remains", "remained", "remaining",
    "stay", "stays", "stayed", "staying",
    "keep", "keeps", "kept", "keeping",
})

_ADVERBS_LEFT = frozenset({
    "just", "really", "actually", "probably", "definitely", "also",
    "still", "already", "soon", "quite", "very", "truly", "finally",
    "recently", "simply", "nearly", "almost", "ever", "always", "often",
    "sometimes", "rarely", "even", "now", "then", "yet", "never", "not",
    "hardly", "barely", "scarcely", "no", "longer", "anymore",
    "immediately", "currently", "previously", "originally", "officially",
    "eventually", "suddenly", "quickly", "slowly", "later", "usually",
    "mostly", "mainly", "partly", "hopefully", "apparently", "certainly",
    "obviously", "clearly", "possibly", "perhaps", "maybe",
    "absolutely", "totally", "completely", "entirely", "fully",
    "pretty", "rather", "fairly", "directly", "straight", "right",
})

_SKIP_LEFT = _AUX_VERBS | _ADVERBS_LEFT

#: Clause-initial words that can never sit inside a subject NP.
_COORD = frozenset({
    "and", "but", "or", "so", "yet", "nor", "because", "if", "when",
    "while", "though", "although", "since", "until", "unless", "as",
    "then", "therefore", "however", "thus", "hence", "meanwhile",
    "otherwise", "instead", "besides", "moreover", "furthermore",
    "whereas", "whenever", "wherever", "whether",
})

_PREP = frozenset({
    "to", "in", "on", "at", "by", "about", "into", "onto", "over",
    "under", "between", "through", "during", "before", "after",
    "around", "across", "toward", "towards", "upon", "within",
    "without", "against", "along", "behind", "beyond", "near", "off",
    "outside", "inside", "despite", "except", "like", "unlike", "via",
    "per", "versus", "vs", "than", "of", "from", "with", "for",
})

#: Temporal nouns/adverbs that terminate objects and never sit inside a
#: subject NP ("visited paris last week" → object "paris").
_TEMP_ADV = frozenset({
    "yesterday", "today", "tomorrow", "tonight", "ago", "later",
    "soon", "recently", "now", "then", "afterward", "afterwards",
    "already", "currently", "finally", "here", "there", "everywhere",
    "somewhere", "anywhere", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "january", "february",
    "march", "april", "june", "july", "august", "september",
    "october", "november", "december", "weekend", "weekends",
    "morning", "afternoon", "evening", "night",
    "weekly", "monthly", "yearly", "daily", "hourly",
    "last", "next",
})

#: Control verbs that take "to" infinitives — for subject recovery
#: across an infinitive boundary ("i want to move" → subject "i",
#: polarity hypothetical).
_CONTROL = frozenset({
    "want", "wants", "wanted", "wanting",
    "plan", "plans", "planned", "planning",
    "decide", "decides", "decided", "deciding",
    "hope", "hopes", "hoped", "hoping",
    "need", "needs", "needed", "needing",
    "like", "likes", "liked", "liking",
    "love", "loves", "loved", "loving",
    "hate", "hates", "hated", "hating",
    "try", "tries", "tried", "trying",
    "manage", "manages", "managed", "managing",
    "forget", "forgets", "forgot", "forgetting",
    "remember", "remembers", "remembered", "remembering",
    "intend", "intends", "intended", "intending",
    "mean", "means", "meant", "meaning",
    "expect", "expects", "expected", "expecting",
    "choose", "chooses", "chose", "chosen", "choosing",
    "prefer", "prefers", "preferred", "preferring",
    "wish", "wishes", "wished", "wishing",
    "agree", "agrees", "agreed", "agreeing",
    "refuse", "refuses", "refused", "refusing",
    "appear", "appears", "appeared", "appearing",
    "tend", "tends", "tended", "tending",
    "afford", "affords", "afforded", "affording",
    "promise", "promises", "promised", "promising",
    "threaten", "threatens", "threatened", "threatening",
    "deserve", "deserves", "deserved", "deserving",
    "attempt", "attempts", "attempted", "attempting",
    "fail", "fails", "failed", "failing",
    "help", "helps", "helped", "helping",
    "wait", "waits", "waited", "waiting",
    "used", "use", "uses", "using",
    "ask", "asks", "asked", "asking",
    "beg", "begs", "begged", "begging",
    "aim", "aims", "aimed", "aiming",
    "prepare", "prepares", "prepared", "preparing",
    "continue", "continues", "continued", "continuing",
})

#: Terms that block subject-NP collection.
_NP_BLOCK = _COORD | _PREP | _TEMP_ADV | _CONTROL | {
    "there", "here", "course", "time",
}

#: NP leads → "definite description" subjects (sieve/unknown). Any
#: determiner also closes a collected NP from the left.
_HARD_DET = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "some", "any",
    "no", "every", "each", "such", "another", "either", "neither",
    "both", "all", "few", "many", "several", "much", "more", "most",
    "other", "others", "enough",
})
_POSS_DET = frozenset({"my", "our", "your", "his", "her", "their", "its"})
_DET_ALL = _HARD_DET | _POSS_DET

_FIRST = frozenset({"i", "we"})
_SECOND = frozenset({"you", "yall", "ye"})
_THIRD = frozenset({"he", "she", "they", "it"})
_PRONOUN_ALL = _FIRST | _SECOND | _THIRD | frozenset({
    "me", "us", "him", "them", "who", "whoever",
})

#: Negation markers in the polarity window (norm/v2 maps n't → "not").
_NEG = frozenset({
    "not", "never", "no", "hardly", "barely", "scarcely", "without",
    "none", "nobody", "nothing", "nowhere", "neither", "nor",
    "cannot", "cant", "aint",
})

#: Modals → "hypothetical" polarity when no negation is present.
_MODAL = frozenset({
    "will", "would", "can", "could", "shall", "should", "may",
    "might", "must", "ought", "ll", "wont",
})

#: Subordinators adjacent left of the subject mark the clause
#: conditional ("if i moved").
_HYP_SUB = frozenset({"if", "unless", "whether", "whenever"})

#: Object collection stops.
_NOM_PRON = frozenset({"i", "we", "he", "she", "they"})
_PARTICLE_WORDS = frozenset({
    "up", "down", "out", "away", "back", "off", "over", "together",
    "home", "along", "around", "abroad", "aboard", "downtown", "aside",
    "apart", "forward", "on",
})
_OBJ_STOP = (_AUX_VERBS | _NEG | _TEMP_ADV | _PREP | _NOM_PRON
             | _COORD | _PARTICLE_WORDS)

#: Leading function words stripped from an object NP ("moved to berlin"
#: → object "berlin"). Possessive determiners are NOT stripped —
#: "visited my mother" keeps object "my mother" and "saw her" keeps
#: "her".
_LEAD_STRIP = _HARD_DET | _PREP | _PARTICLE_WORDS | frozenset({
    "really", "just", "also", "only", "even", "quite", "very",
    "somewhere", "anywhere",
})

#: Coordinators that let a predicate reuse the previous subject.
_COORD_REUSE = frozenset({"and", "but", "or"})

#: Byte gap between consecutive terms that implies punctuation — a
#: clause boundary ("moved. she" — the ". " is 2 bytes).
_BOUNDARY_GAP = 2

#: Caps to keep scans bounded.
_MAX_SUBJ_TERMS = 5
_MAX_OBJ_TERMS = 8
_MAX_PARTICLE_SKIP = 3
_MAX_AUX_LOOKBACK = 3
_MAX_REUSE_LOOKBACK = 6


# ---------------------------------------------------------------------------
# Term-stream helpers
# ---------------------------------------------------------------------------

def _linked(terms: list[NormTerm], i: int) -> bool:
    """``terms[i]`` continues the same clause as ``terms[i-1]`` — the
    byte gap between them is smaller than one punctuation mark."""
    if i <= 0 or i >= len(terms):
        return False
    return terms[i].byte_start - terms[i - 1].byte_end < _BOUNDARY_GAP


def _text_terms(norm: NormAnalysis) -> list[NormTerm]:
    """The text channel in byte order (stem and identifier channels are
    ignored — matching is on the folded surface)."""
    terms = [t for t in norm.terms if t.channel == "text"]
    terms.sort(key=lambda t: (t.byte_start, t.byte_end, t.term))
    return terms


def _collectable(term: str) -> bool:
    """A term that may sit inside a subject NP chunk."""
    return (
        term not in _SKIP_LEFT
        and term not in _NP_BLOCK
        and term not in _ALL_FORMS
    )


# ---------------------------------------------------------------------------
# Subject resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Subject:
    canon: Optional[str]
    form: str                      # first_person|second_person|pronoun|
                                   # description|name|reused|none
    span: Optional[tuple[int, int]]
    window: tuple[NormTerm, ...]   # aux/neg terms between subject & head
    infinitive: bool = False


def _collect_np(
    terms: list[NormTerm],
    j: int,
    skip: frozenset[str],
) -> tuple[list[NormTerm], int, Optional[NormTerm]]:
    """Collect the subject NP ending at term ``j`` scanning left.
    Returns (chunk, stop_index, blocker). A determiner closes the chunk
    from the left ("the manager" is one description); an ``of``- or
    ``and``-bridge continues it into another collectable term ("the
    manager of acme", "alice and bob")."""
    chunk: list[NormTerm] = []
    blocker: Optional[NormTerm] = None
    bridged_of = False
    while j >= 0 and _linked(terms, j + 1):
        w = terms[j].term
        if w in skip or w in _ALL_FORMS:
            blocker = terms[j]
            break
        if w in ("of", "and", "or") and chunk and j > 0 \
                and _collectable(terms[j - 1].term):
            # bridge only into a collectable term — "alice and bob"
            # joins two NPs; "moved and quit" does not (left of "and"
            # is a verb form).
            if w == "of":
                if bridged_of:
                    blocker = terms[j]
                    break
                bridged_of = True
            chunk.insert(0, terms[j])
            j -= 1
            continue
        if w in _NP_BLOCK:
            blocker = terms[j]
            break
        chunk.insert(0, terms[j])
        j -= 1
        if w in _DET_ALL:
            break  # determiner closes the NP on the left
        if len(chunk) >= _MAX_SUBJ_TERMS:
            break
    return chunk, j, blocker


def _split_subjects(chunk: list[NormTerm]) -> list[list[NormTerm]]:
    """Split a collected chunk on coordinator terms: "alice and bob" →
    [[alice], [bob]] — one event per subject."""
    out: list[list[NormTerm]] = []
    cur: list[NormTerm] = []
    for t in chunk:
        if t.term in ("and", "or"):
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        out.append(cur)
    return out or [chunk]


def _classify_chunk(
    chunk: list[NormTerm],
    window: tuple[NormTerm, ...],
    infinitive: bool,
    speaker_canon: Optional[str],
    addressee_canon: Optional[str],
    sieve: Optional[Callable[[str], Optional[str]]],
    canon_fn: Optional[Callable[[str], str]],
    raw_slice: Callable[[int, int], Optional[str]],
    stats: dict[str, int],
) -> _Subject:
    """Map one subject NP chunk to a canon (or abstain → None)."""
    span = (chunk[0].byte_start, chunk[-1].byte_end)
    surface = " ".join(t.term for t in chunk)
    surface_raw = raw_slice(*span) or surface
    w0 = chunk[0].term

    if len(chunk) == 1 and w0 in _FIRST:
        return _Subject(speaker_canon, "first_person", span,
                        window, infinitive)
    if len(chunk) == 1 and w0 in _SECOND:
        return _Subject(addressee_canon, "second_person", span,
                        window, infinitive)
    if len(chunk) == 1 and w0 in _THIRD:
        canon = sieve(surface_raw) if sieve is not None else None
        return _Subject(canon, "pronoun", span, window, infinitive)
    if w0 in _DET_ALL or w0 in _PRONOUN_ALL:
        # definite / possessive description → coref_sieve/v1
        canon = sieve(surface_raw) if sieve is not None else None
        return _Subject(canon, "description", span, window, infinitive)

    # bare lexical NP → explicit name → canon_fn (entities_v2.canon)
    if canon_fn is not None:
        canon = canon_fn(surface_raw)
        if canon:
            return _Subject(canon, "name", span, window, infinitive)
    else:
        stats["canon_unavailable"] += 1
    return _Subject(None, "name", span, window, infinitive)


def _resolve_subjects(
    terms: list[NormTerm],
    head_idx: int,
    speaker_canon: Optional[str],
    addressee_canon: Optional[str],
    sieve: Optional[Callable[[str], Optional[str]]],
    canon_fn: Optional[Callable[[str], str]],
    last_subject: Optional[_Subject],
    raw_slice: Callable[[int, int], Optional[str]],
    stats: dict[str, int],
) -> list[_Subject]:
    """§32.10 subject resolution: skip the aux/adverb window, collect
    the NP chunk, split coordinators, classify each part. Always
    returns ≥ 1 subject (the ``none`` form when unresolved)."""
    j = head_idx - 1
    window: list[NormTerm] = []
    while j >= 0 and _linked(terms, j + 1) and terms[j].term in _SKIP_LEFT:
        window.insert(0, terms[j])
        j -= 1

    infinitive = False
    chunk, stop_j, blocker = _collect_np(terms, j, _SKIP_LEFT)

    if not chunk and blocker is not None and blocker.term == "to":
        # infinitive predicate ("want to move", "going to move") —
        # recover the matrix subject across "to" + control/aux terms.
        infinitive = True
        j2 = stop_j - 1
        while (
            j2 >= 0 and _linked(terms, j2 + 1)
            and terms[j2].term in (_SKIP_LEFT | _CONTROL | {"to"})
        ):
            j2 -= 1
        chunk, stop_j, blocker = _collect_np(terms, j2, _SKIP_LEFT)

    if not chunk:
        # coordinator / serial-predicate reuse: "i moved and started",
        # "i moved, then started", "i moved. went to berlin." — scan
        # left (bounded) for a coordinator or a previous predicate form.
        k = stop_j
        hops = 0
        reuse = False
        while k >= 0 and hops < _MAX_REUSE_LOOKBACK:
            w = terms[k].term
            if w in _SKIP_LEFT:
                k -= 1
                hops += 1
                continue
            if w in _COORD_REUSE or w in _ALL_FORMS:
                reuse = True
            break
        if reuse and last_subject is not None:
            return [_Subject(
                canon=last_subject.canon,
                form="reused",
                span=last_subject.span,
                window=tuple(window),
                infinitive=infinitive,
            )]
        return [_Subject(None, "none", None, tuple(window), infinitive)]

    win = tuple(window)
    return [
        _classify_chunk(part, win, infinitive, speaker_canon,
                        addressee_canon, sieve, canon_fn, raw_slice,
                        stats)
        for part in _split_subjects(chunk)
    ]


# ---------------------------------------------------------------------------
# Object extraction
# ---------------------------------------------------------------------------

def _scan_object(terms: list[NormTerm], start: int) -> list[NormTerm]:
    """Collect the object NP after the predicate: strip leading function
    words, then take content terms until a clause boundary, coordinator,
    auxiliary, temporal adverb, or another predicate head. ``and``/``or``
    continue the object only when followed by an NP-ish term that is not
    itself a new subject ("bread and milk" keeps both; "house and bob
    sold" does not swallow ``bob``)."""
    k = start
    while k < len(terms) and _linked(terms, k) \
            and terms[k].term in _LEAD_STRIP:
        k += 1
    obj: list[NormTerm] = []
    while k < len(terms) and _linked(terms, k):
        w = terms[k].term
        if w in _COORD:
            if (
                w in ("and", "or")
                and k + 1 < len(terms) and _linked(terms, k + 1)
                and _collectable(terms[k + 1].term)
                and not (
                    k + 2 < len(terms) and _linked(terms, k + 2)
                    and terms[k + 2].term in _ALL_FORMS
                )
            ):
                obj.append(terms[k])
                k += 1
                continue
            break
        if w in _OBJ_STOP or w in _ALL_FORMS:
            break
        obj.append(terms[k])
        k += 1
        if len(obj) >= _MAX_OBJ_TERMS:
            break
    return obj


# ---------------------------------------------------------------------------
# Polarity
# ---------------------------------------------------------------------------

def _polarity(
    window: tuple[NormTerm, ...],
    infinitive: bool,
    terms: list[NormTerm],
    subject_span: Optional[tuple[int, int]],
) -> tuple[str, Optional[tuple[int, int]]]:
    """``negate`` if a negation marker sits in the window between the
    subject and the predicate; else ``hypothetical`` for modal /
    infinitive / conditional frames; else ``affirm``. Returns the
    polarity and the pin of the triggering negation term, if any."""
    neg_pin = None
    modal = infinitive
    for t in window:
        if t.term in _NEG:
            neg_pin = (t.byte_start, t.byte_end)
            break
        if t.term in _MODAL or t.term in _HYP_SUB:
            modal = True
    if neg_pin is not None:
        return "negate", neg_pin
    if subject_span is not None:
        # a subordinator directly left of the subject marks the clause
        # conditional ("if i moved")
        for i, t in enumerate(terms):
            if t.byte_start == subject_span[0]:
                if i > 0 and _linked(terms, i) \
                        and terms[i - 1].term in _HYP_SUB:
                    modal = True
                break
    if modal:
        return "hypothetical", None
    return "affirm", None


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Match:
    lexeme: _Lexeme
    head_idx: int
    pred_start: int            # byte start of predicate span
    pred_end: int              # byte end of predicate span
    after_idx: int             # first term index after the predicate
    mid_terms: tuple[NormTerm, ...]   # terms skipped between head & particle


def _try_particles(
    terms: list[NormTerm],
    head_idx: int,
    lx: _Lexeme,
) -> Optional[tuple[int, int, tuple[NormTerm, ...]]]:
    """Match the lexeme's particles in order, each within
    ``_MAX_PARTICLE_SKIP`` terms. Intervening terms (separable phrasal
    verbs: "broke it up", "laid him off") become ``mid_terms`` and are
    covered by the predicate span. Returns (pred_end, after_idx, mid)."""
    k = head_idx + 1
    mid: list[NormTerm] = []
    for particle in lx.particles:
        found = False
        skipped = 0
        while k < len(terms) and skipped <= _MAX_PARTICLE_SKIP:
            if not _linked(terms, k):
                return None
            w = terms[k].term
            if w == particle:
                found = True
                k += 1
                break
            if (
                w in _OBJ_STOP or w in _ALL_FORMS or w in _DET_ALL
            ):
                return None
            mid.append(terms[k])
            k += 1
            skipped += 1
        if not found:
            return None
    return (terms[k - 1].byte_end, k, tuple(mid))


def _match_lexeme(
    terms: list[NormTerm],
    head_idx: int,
    lx: _Lexeme,
) -> Optional[_Match]:
    """Structural checks that decide whether the lexeme fires at
    ``head_idx``: particle match, aux-left requirement, the
    reject-next-term guard, and the noun-position guard (a determiner
    directly before the head means the head is a noun — "the book was
    great" never emits "book")."""
    prev = head_idx - 1
    if prev >= 0 and _linked(terms, head_idx) \
            and terms[prev].term in _DET_ALL:
        return None
    if (
        lx.reject_next is not None
        and head_idx + 1 < len(terms)
        and _linked(terms, head_idx + 1)
        and terms[head_idx + 1].term in lx.reject_next
    ):
        return None

    pred_start = terms[head_idx].byte_start
    after_idx = head_idx + 1
    mid: tuple[NormTerm, ...] = ()

    if lx.particles:
        res = _try_particles(terms, head_idx, lx)
        if res is None:
            return None
        pred_end, after_idx, mid = res
    else:
        pred_end = terms[head_idx].byte_end

    if lx.aux_left is not None:
        j = head_idx - 1
        back = 0
        aux_start: Optional[int] = None
        while j >= 0 and _linked(terms, j + 1) \
                and back <= _MAX_AUX_LOOKBACK:
            w = terms[j].term
            if w in lx.aux_left:
                aux_start = terms[j].byte_start
                break
            if w in _COORD or w in _PREP:
                break
            j -= 1
            back += 1
        if aux_start is None:
            return None
        pred_start = aux_start

    return _Match(lx, head_idx, pred_start, pred_end, after_idx, mid)


def _object_ok(
    lx: _Lexeme,
    terms: list[NormTerm],
    after_idx: int,
    obj: list[NormTerm],
) -> bool:
    """Lexeme gates on the object: noun-set gates, numeric gate,
    person-ish gate."""
    if lx.obj_personish:
        if after_idx >= len(terms) or not _linked(terms, after_idx):
            return False
        w = terms[after_idx].term
        if w in _HARD_DET or w in _OBJ_STOP or w in _ALL_FORMS:
            return False
        return True
    # first content term of the object, skipping possessive dets
    first: Optional[str] = None
    for t in obj:
        if t.term in _POSS_DET:
            continue
        first = t.term
        break
    if lx.obj_num:
        return bool(first and first.isdigit())
    if lx.obj_gate is not None:
        return first in lx.obj_gate
    return True


def _pins_ok(pins: dict[str, Any], raw_len: Optional[int]) -> bool:
    """Pin verification (V7-09.08): every present span has ``s < e`` and
    stays inside the raw byte length when known. Failing tuples are
    dropped and counted."""
    pred = pins.get("predicate")
    if pred is None or not (pred[0] < pred[1]):
        return False
    for key in ("subject", "object", "negation", "span"):
        span = pins.get(key)
        if span is not None and not (span[0] < span[1]):
            return False
    if raw_len is not None:
        for key in ("subject", "predicate", "object", "negation", "span"):
            span = pins.get(key)
            if span is not None and span[1] > raw_len:
                return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_events(
    norm: NormAnalysis,
    unit_id: str,
    speaker_canon: Optional[str],
    occurred: IntervalUs,
    sieve: Optional[Callable[[str], Optional[str]]] = None,
    raw_text: Optional[str] = None,
    *,
    addressee_canon: Optional[str] = None,
    canon_fn: Optional[Callable[[str], str]] = None,
    stats: Optional[dict[str, int]] = None,
) -> list[EventTuple]:
    """Extract event tuples from one unit's ``norm/v2`` analysis.

    ``sieve`` is a ``Callable[[str], Optional[str]]`` mapping a mention
    surface ("she", "the manager") to a canon — the caller adapts
    ``coref_sieve.resolve_antecedent`` by binding its unit index and
    session turns. ``canon_fn`` maps an explicit-name surface to a
    canon and defaults to ``entities_v2.canon`` (lazy import; when
    unavailable, name subjects stay ``None`` and the miss is counted).
    ``stats``, when given, receives per-call counters including
    ``dropped_unpinned`` (tuples that failed pin verification).
    """
    local = {
        "predicates_matched": 0,
        "emitted": 0,
        "dropped_unpinned": 0,
        "canon_unavailable": 0,
    }
    COUNTERS["units"] += 1

    if canon_fn is None:
        try:
            from verbatim.enrichment.entities_v2 import (
                canon as _canon,
            )
            canon_fn = _canon
        except Exception:
            canon_fn = None

    raw_bytes = raw_text.encode("utf-8") if raw_text is not None else None
    raw_len = len(raw_bytes) if raw_bytes is not None else None

    def raw_slice(s: int, e: int) -> Optional[str]:
        if raw_bytes is None or s < 0 or e > raw_len or s >= e:
            return None
        try:
            return raw_bytes[s:e].decode("utf-8")
        except UnicodeDecodeError:
            return None

    terms = _text_terms(norm)
    events: list[EventTuple] = []
    last_subject: Optional[_Subject] = None
    sieve_cache: dict[str, Optional[str]] = {}

    def _sieve(surface: str) -> Optional[str]:
        if sieve is None:
            return None
        if surface not in sieve_cache:
            sieve_cache[surface] = sieve(surface)
        return sieve_cache[surface]

    for i, t in enumerate(terms):
        cand_idx = _HEAD_INDEX.get(t.term)
        if not cand_idx:
            continue
        # the subject scan is shared by all lexeme candidates at this
        # head — resolve lazily, once.
        subjects: Optional[list[_Subject]] = None
        for lx_i in cand_idx:
            lx = _LEXICON[lx_i]
            m = _match_lexeme(terms, i, lx)
            if m is None:
                continue
            local["predicates_matched"] += 1
            if subjects is None:
                subjects = _resolve_subjects(
                    terms, i, speaker_canon, addressee_canon, _sieve,
                    canon_fn, last_subject, raw_slice, local,
                )
                # the subject persists across the clause even when this
                # candidate's gates reject it ("i got engaged" — the
                # failed "get" still supplies "i" to "engaged")
                if subjects[0].form != "none":
                    last_subject = subjects[0]

            # object: separable-particle mid-terms win over the
            # post-particle NP ("laid him off" → object "him"); adverb
            # mid-terms never become the object.
            obj = _scan_object(terms, m.after_idx)
            if m.mid_terms:
                mid_obj = [
                    x for x in m.mid_terms
                    if x.term not in _ADVERBS_LEFT
                    and x.term not in _TEMP_ADV
                    and x.term not in _NEG
                ]
                if mid_obj:
                    obj = mid_obj
            if not _object_ok(lx, terms, m.after_idx, obj):
                continue

            obj_text = ""
            obj_span: Optional[tuple[int, int]] = None
            if obj:
                obj_span = (obj[0].byte_start, obj[-1].byte_end)
                obj_text = raw_slice(*obj_span) or " ".join(
                    x.term for x in obj)

            for subj in subjects:
                if lx.subj_person and subj.form not in (
                    "first_person", "second_person", "pronoun", "name",
                    "reused",
                ):
                    continue
                polarity, neg_pin = _polarity(
                    subj.window, subj.infinitive, terms, subj.span)

                span_s = min(
                    x for x in (
                        subj.span[0] if subj.span else None,
                        neg_pin[0] if neg_pin else None,
                        m.pred_start,
                        obj_span[0] if obj_span else None,
                    ) if x is not None
                )
                span_e = max(
                    x for x in (
                        m.pred_end,
                        obj_span[1] if obj_span else None,
                        neg_pin[1] if neg_pin else None,
                    ) if x is not None
                )

                pins: dict[str, Any] = {
                    "subject": subj.span,
                    "predicate": (m.pred_start, m.pred_end),
                    "object": obj_span,
                    "negation": neg_pin,
                    "span": (span_s, span_e),
                    "lexeme": lx.lexeme,
                    "subject_form": subj.form,
                    "pinned": "raw_verified" if raw_bytes is not None
                              else "term_offsets",
                }
                if not _pins_ok(pins, raw_len):
                    local["dropped_unpinned"] += 1
                    continue

                events.append(EventTuple(
                    unit_id=unit_id,
                    subject_canon=subj.canon,
                    predicate_lemma=lx.family,
                    object_text=obj_text,
                    polarity=polarity,
                    occurred=occurred,
                    pins=pins,
                    rule_id=f"{lx.family}/{lx.lexeme}",
                ))
                local["emitted"] += 1
            break  # one lexeme per head term

    for k, v in local.items():
        COUNTERS[k] += v
    if stats is not None:
        stats.update(local)
    return events


__all__ = [
    "EXTRACTOR_ID",
    "FORMULA_STATUS",
    "COUNTERS",
    "reset_counters",
    "extract_events",
    "LEXICON_FAMILIES",
    "LEXEME_COUNT",
]
