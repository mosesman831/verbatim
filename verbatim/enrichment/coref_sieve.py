"""Deterministic coreference sieve — ``coref_sieve/v1`` (SPEC_V7 R1, V7-13.20).

Model-free pronoun and definite-description resolution over one session.
The sieve is *conservative by construction*: abstention is free, a wrong
merge is expensive. A mention resolves only when exactly one candidate
canon survives the agreement filters with a decisive score; any residual
competition returns ``None`` (callers record ``subject=unknown``,
V7-09.08 / §32.10).

Contract (``docs/v7_contracts.md`` frozen APIs)::

    resolve_antecedent(mention, unit_index, session_turns, lookback=6)
        -> Optional[str]        # canon or None (abstain)

``session_turns[i]`` = ``{"speaker": <canon|None>, "text": str,
"canon_mentions": [canon, ...]}`` — canon keys mentioned in that turn, in
surface order. Optional richer turn keys are honored when present, never
required: ``"subjects": [canon, ...]`` or ``"mention_roles":
{canon: "subject"|"object"}`` supply grammatical role; otherwise the
first ``canon_mentions`` entry is treated as the subject-position
mention. ``"speaker_canon"`` overrides ``"speaker"`` as the speaker's
canon key.

Rules, in order (V7-13.20):

(a) immediately-previous-turn subject/object preference — mentions in the
    previous turn carry a bonus, subject-position mentions more than
    object-position ones;
(b) mention recency inside the lookback window (default N = 6 previous
    turns plus the current unit at ``d = 0``);
(c) number agreement — singular pronouns never resolve to plural
    entities; ``they/them`` may be singular and also resolve to
    collectives ("the team");
(d) gender agreement only where morphology makes it reliable —
    ``he/him/his`` and ``she/her/hers`` filter out candidates whose
    gender is *known* to differ. Canon gender comes only from the
    optional ``gender`` map or from gendered nouns in the canon's own
    surface ("her brother" → m); gender is NEVER guessed from names;
(e) co-occurrence boost — a canon previously co-mentioned with a canon
    of the current turn scores higher;
(f) abstention — when ≥2 candidates remain plausible (score gap within
    the declared ``_MARGIN``) the sieve returns ``None``. For gendered
    pronouns an *unverified-gender* winner abstains on ANY surviving
    competition: picking between two unknown-gender canons is a coin
    flip. ``they/them`` with ≥2 survivors abstains outright — the union
    of two canons is itself a live referent no single canon carries.

Speaker alternation (V7-13.20, §32.10): first-person forms resolve to the
current speaker canon; ``you``-forms resolve to the addressee — the
unique *other* speaker inside the window (≥2 others ⇒ abstain, none ⇒
``None``); ``we/us/our`` is a group and never resolves. The current
speaker is excluded from third-person candidates.

``it`` requires *thing evidence*: a thing-noun or collective noun in the
canon surface, or explicit ``kind="thing"`` / ``gender="n"`` metadata.
Person canons (person noun, session speaker, known m/f gender) are never
``it``.

Definite descriptions ("the woman from work") resolve via
``resolve_descriptions`` only when a single candidate discriminates on
attribute terms — a tie on attribute overlap abstains even when recency
differs.

``last_resort_previous_turn=True`` reproduces the R0 previous-turn-only
rule retained as the Q8 baseline arm (candidates restricted to ``d = 1``;
same-unit mentions excluded).

All lexicons below are owned, English-only, and versioned with the sieve
— no benchmark-derived entries (V7-22.18). Constants are
``provisional/v7-r0``. Pure stdlib; deterministic.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Dict, List, Optional, Tuple

SIEVE_ID = "coref_sieve/v1"
FORMULA_STATUS = "provisional/v7-r0"

#: Declared lookback window N (V7-13.20): previous N turns + same unit.
DEFAULT_LOOKBACK = 6

# Score weights (declared §32.10 pre-search snapshot; provisional/v7-r0).
_W_TURN = 10          # recency: per-turn weight inside the window
_W_SUBJECT = 4        # subject-role bonus within a turn
_W_OBJECT = 1         # object/other-role bonus within a turn
_W_PREV_TURN = 6      # immediately-previous-turn preference
_W_COOC = 3           # co-occurrence with a current-turn canon
_W_FREQ_CAP = 3       # repeat-mention bonus cap

#: Declared abstention margin (V7-13.20(f)): a winner must beat the
#: runner-up by strictly more than this to resolve.
_MARGIN = 10

# ---------------------------------------------------------------------------
# Lexicons (owned, English; surface morphology only — never name guessing)
# ---------------------------------------------------------------------------

_CLASS_FORMS: Dict[str, frozenset] = {
    "first_sg": frozenset({"i", "me", "my", "mine", "myself"}),
    "first_pl": frozenset({"we", "us", "our", "ours", "ourselves"}),
    "second": frozenset(
        {"you", "your", "yours", "yourself", "yourselves", "u", "ya"}
    ),
    "masc": frozenset({"he", "him", "his", "himself"}),
    "fem": frozenset({"she", "her", "hers", "herself"}),
    "neut": frozenset({"it", "its", "itself"}),
    "epicene": frozenset(
        {"they", "them", "their", "theirs", "themself", "themselves"}
    ),
}
_FORM_CLASS: Dict[str, str] = {
    f: c for c, forms in _CLASS_FORMS.items() for f in forms
}
_REQ_GENDER = {"masc": "m", "fem": "f", "neut": "n", "epicene": None}

#: Mentions the sieve deliberately does not handle — an honest
#: ``unsupported`` rather than a guess.
_UNSUPPORTED_FORMS = frozenset({
    "who", "whom", "whose", "which", "that", "this", "these", "those",
    "there", "here", "what", "whatever", "whoever", "whichever",
    "someone", "anyone", "everyone", "nobody", "somebody", "anybody",
    "everybody", "something", "anything", "everything", "nothing",
    "each", "either", "neither", "both", "all", "none", "one", "ones",
    "such", "same", "others", "another",
})

_MASC_NOUNS = frozenset({
    "man", "guy", "boy", "dude", "gentleman", "father", "dad", "daddy",
    "papa", "brother", "uncle", "son", "grandfather", "grandpa",
    "grandson", "husband", "boyfriend", "fiance", "nephew", "king",
    "prince", "groom", "widower", "waiter", "actor", "chairman",
    "spokesman", "policeman", "fireman", "salesman", "stepdad",
    "stepfather", "stepson", "lad", "bloke", "fella", "monk", "lord",
    "sir", "mister", "mr", "male", "bachelor", "hero", "host",
    "businessman", "cameraman", "mailman", "milkman", "fisherman",
    "foreman", "godfather", "stepbrother",
})
_FEM_NOUNS = frozenset({
    "woman", "girl", "lady", "gal", "mother", "mom", "mommy", "mama",
    "mum", "sister", "aunt", "daughter", "grandmother", "grandma",
    "granddaughter", "wife", "girlfriend", "fiancee", "niece", "queen",
    "princess", "bride", "widow", "waitress", "actress", "chairwoman",
    "spokeswoman", "policewoman", "saleswoman", "stepmom",
    "stepmother", "stepdaughter", "nun", "madam", "mrs", "ms", "miss",
    "female", "heroine", "hostess", "businesswoman", "goddaughter",
    "stepsister", "landlady",
})
#: Gender-neutral person nouns (kind evidence, no gender).
_PERSON_NOUNS = frozenset({
    "person", "friend", "partner", "spouse", "sibling", "cousin",
    "parent", "child", "kid", "baby", "toddler", "neighbor", "neighbour",
    "roommate", "flatmate", "housemate", "classmate", "coworker",
    "co-worker", "colleague", "teammate", "workmate", "manager", "boss",
    "lead", "supervisor", "mentor", "mentee", "intern", "employee",
    "employer", "doctor", "nurse", "teacher", "professor", "prof",
    "coach", "chef", "guest", "client", "customer", "patient",
    "therapist", "engineer", "developer", "designer", "programmer",
    "artist", "musician", "writer", "author", "singer", "dancer",
    "driver", "pilot", "lawyer", "agent", "officer", "director",
    "founder", "owner", "volunteer", "student", "pupil", "advisor",
    "adviser", "consultant", "photographer", "plumber", "electrician",
    "mechanic", "barista", "landlord", "tenant", "assistant",
    "receptionist", "scientist", "analyst", "judge", "captain",
    "firefighter", "soldier", "vet", "veterinarian", "tutor",
    "principal", "dean", "mayor", "minister", "senator", "gardener",
    "librarian", "technician", "freelancer", "blogger", "podcaster",
    "clerk", "salesperson", "secretary", "nurse", "pm", "ceo", "cto",
    "vp", "dev", "op", "admin", "recruiter", "candidate", "witness",
    "relative", "roomie", "buddy", "pal", "mate", "grandparent",
    "grandchild", "stepparent", "in-law", "fiancé", "fiancée",
})
#: Collective nouns — grammatically singular but group referents.
#: ``it``- and ``they``-compatible, never ``he/she``.
_COLLECTIVE_NOUNS = frozenset({
    "team", "family", "group", "committee", "couple", "pair", "trio",
    "staff", "crew", "class", "board", "crowd", "audience", "company",
    "band", "gang", "jury", "panel", "council", "department", "club",
    "organization", "organisation", "firm", "agency", "government",
    "squad", "platoon", "orchestra", "choir", "cast", "faculty",
    "management", "leadership", "public", "community", "union",
})
#: Plural nouns — surface evidence that a canon denotes >1 entity.
_PLURAL_NOUNS = frozenset({
    "parents", "children", "kids", "friends", "siblings", "twins",
    "coworkers", "colleagues", "teammates", "classmates", "roommates",
    "flatmates", "grandparents", "grandchildren", "folks", "people",
    "men", "women", "boys", "girls", "guys", "partners", "spouses",
    "sons", "daughters", "brothers", "sisters", "uncles", "aunts",
    "cousins", "nieces", "nephews", "moms", "dads", "mothers",
    "fathers", "wives", "husbands", "babies", "toddlers", "neighbors",
    "neighbours", "guests", "visitors", "clients", "customers",
    "members", "players", "employees", "students", "teachers",
    "doctors", "nurses", "managers", "bosses", "mentors", "interns",
    "volunteers", "lawyers", "agents", "officers", "directors",
    "founders", "owners", "advisors", "consultants", "assistants",
    "scientists", "analysts", "judges", "soldiers", "drivers",
    "pilots", "chefs", "waiters", "waitresses", "actors", "actresses",
    "singers", "dancers", "photographers", "developers", "engineers",
    "designers", "programmers", "artists", "musicians", "writers",
    "authors", "landlords", "tenants", "librarians", "technicians",
    "tutors", "supervisors", "principals", "deans", "mayors",
    "ministers", "senators", "firefighters", "guards", "farmers",
    "dogs", "cats", "pets", "puppies", "kittens", "birds", "horses",
    "rabbits", "others", "both",
})
#: Singular-in-spite-of-trailing-s exceptions for the head-token
#: heuristic ("the smiths" is plural; "the news" is not).
_S_SG_EXCEPTIONS = frozenset({
    "news", "series", "species", "physics", "maths", "mathematics",
    "economics", "politics", "ethics", "gymnastics", "athletics",
    "linguistics", "statistics", "lens", "bus", "gas", "plus", "canvas",
    "chaos", "bias", "atlas", "cosmos", "this", "his", "its", "yes",
    "thus", "us", "vs", "etc",
})
#: Thing nouns — positive evidence a canon is a non-person referent.
_THING_NOUNS = frozenset({
    "car", "bike", "bicycle", "motorcycle", "truck", "van", "bus",
    "train", "plane", "flight", "boat", "ship", "scooter", "house",
    "home", "apartment", "condo", "flat", "room", "office", "building",
    "garage", "garden", "yard", "shed", "studio", "cabin", "computer",
    "laptop", "phone", "tablet", "tv", "television", "camera", "watch",
    "device", "machine", "printer", "server", "app", "application",
    "software", "program", "website", "site", "email", "message",
    "letter", "package", "box", "bag", "backpack", "suitcase",
    "luggage", "book", "notebook", "journal", "diary", "movie", "film",
    "show", "series", "song", "album", "podcast", "game", "toy",
    "report", "document", "file", "folder", "binder", "paper",
    "presentation", "slides", "spreadsheet", "budget", "project",
    "task", "ticket", "issue", "bug", "feature", "code", "script",
    "tool", "plan", "idea", "proposal", "contract", "invoice",
    "meeting", "appointment", "event", "party", "dinner", "lunch",
    "breakfast", "meal", "recipe", "food", "cake", "coffee", "tea",
    "drink", "wine", "beer", "gift", "present", "card", "photo",
    "picture", "painting", "drawing", "desk", "chair", "table",
    "couch", "sofa", "bed", "lamp", "door", "window", "key", "keys",
    "wallet", "purse", "umbrella", "jacket", "coat", "shirt", "dress",
    "shoes", "glasses", "ring", "necklace", "gym", "park", "store",
    "shop", "restaurant", "cafe", "cafeteria", "bar", "hotel",
    "school", "hospital", "church", "library", "museum", "city",
    "country", "town", "village", "street", "road", "bridge", "dog",
    "cat", "pet", "puppy", "kitten", "bird", "fish", "hamster",
    "rabbit", "horse", "turtle", "plant", "tree", "flower", "charger",
    "cable", "keyboard", "mouse", "monitor", "screen", "speaker",
    "headphones", "earbuds", "job", "role", "position", "interview",
    "exam", "test", "class", "course", "lesson", "degree", "thesis",
    "essay", "article", "blog", "post", "tweet", "video", "channel",
    "account", "password", "wifi", "router", "battery", "engine",
    "brakes", "tires", "guitar", "piano", "violin", "drums",
    "basketball", "football", "soccer", "tennis", "marathon", "race",
    "concert", "festival", "wedding", "funeral", "graduation",
    "birthday", "holiday", "vacation", "trip", "flight", "visa",
    "passport", "license", "insurance", "loan", "mortgage", "rent",
    "bill", "bills", "taxes", "salary", "bonus", "raise",
})
#: Number words / quantifiers that make a description plural.
_PLURAL_MARKERS = frozenset({
    "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "several", "many", "few", "both", "couple", "pair", "all",
    "some",
})
#: Determiners/possessives stripped from a description before attribute
#: extraction.
_DESC_STOP = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "my", "your",
    "his", "her", "our", "their", "its", "new", "old", "other",
    "another", "same", "former", "previous", "current", "ex", "from",
    "of", "at", "in", "on", "with", "for", "to", "who", "whom",
    "whose", "which", "one", "ones",
})

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _fold(text: object) -> str:
    """NFKC + casefold — the canon's matching projection."""
    return unicodedata.normalize("NFKC", str(text if text is not None else "")).casefold()


def _tokens(text: str) -> Tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(_fold(text)))


def _norm_map(m: Optional[Mapping[str, str]]) -> Dict[str, str]:
    if not m:
        return {}
    return {_fold(k): str(v).lower() for k, v in m.items()}


def _turn_canons(turn: Mapping) -> List[str]:
    return [_fold(c) for c in (turn.get("canon_mentions") or ())]


def _turn_speaker(turn: Mapping) -> Optional[str]:
    s = turn.get("speaker_canon") or turn.get("speaker")
    f = _fold(s).strip()
    return f or None


def _role(turn: Mapping, canon: str, position: int) -> str:
    """subject|mention — explicit keys win; else first mention counts as
    the subject-position mention (documented fallback)."""
    roles = turn.get("mention_roles")
    if roles:
        for k, v in roles.items():
            if _fold(k) == canon:
                return "subject" if str(v).lower() == "subject" else "object"
    subs = turn.get("subjects")
    if subs is not None:
        return "subject" if canon in {_fold(s) for s in subs} else "object"
    return "subject" if position == 0 else "object"


class _Traits:
    """Agreement features of one candidate canon."""

    __slots__ = ("canon", "gender", "plural", "collective", "person",
                 "thing")

    def __init__(self, canon: str, gender: Optional[str], plural: bool,
                 collective: bool, person: bool, thing: bool):
        self.canon = canon
        self.gender = gender          # "m"|"f"|"n"|None
        self.plural = plural
        self.collective = collective
        self.person = person
        self.thing = thing


def _canon_traits(canon: str, speakers: frozenset,
                  gmap: Dict[str, str], kmap: Dict[str, str],
                  nmap: Dict[str, str]) -> _Traits:
    toks = set(_tokens(canon))
    last = tuple(_tokens(canon))[-1] if toks else ""

    # --- number --------------------------------------------------------
    plural = bool(toks & _PLURAL_NOUNS) or " and " in f" {canon} " \
        or canon.startswith("both ")
    if not plural and toks and len(toks) >= 2 and last.endswith("s") \
            and not last.endswith(("ss", "us", "is")) \
            and last not in _S_SG_EXCEPTIONS:
        plural = True  # "the smiths", "my glasses" — head-token heuristic
    n = nmap.get(canon)
    if n == "pl":
        plural = True
    elif n == "sg":
        plural = False

    # --- gender (morphology or explicit map only — never from names) ---
    gender = gmap.get(canon)
    if gender not in ("m", "f", "n"):
        gender = None
    if gender is None:
        m_hit = bool(toks & _MASC_NOUNS)
        f_hit = bool(toks & _FEM_NOUNS)
        if m_hit and not f_hit:
            gender = "m"
        elif f_hit and not m_hit:
            gender = "f"
        # both or neither -> unknown

    # --- kind ----------------------------------------------------------
    collective = bool(toks & _COLLECTIVE_NOUNS)
    person = bool(toks & (_PERSON_NOUNS | _MASC_NOUNS | _FEM_NOUNS)) \
        or canon in speakers or gender in ("m", "f")
    thing = bool(toks & _THING_NOUNS)
    k = kmap.get(canon)
    if k == "person":
        person, thing = True, False
    elif k == "thing":
        thing, person = True, False
    elif k == "group":
        collective, person = True, False
    if gender == "n":
        thing, person = True, False
    return _Traits(canon, gender, plural, collective, person, thing)


def _passes(t: _Traits, cls: str) -> bool:
    """Agreement filters (V7-13.20 c/d). ``cls`` is the pronoun class."""
    if cls == "masc" or cls == "fem":
        if t.plural or t.collective:
            return False
        if t.gender == "n":
            return False
        if cls == "masc" and t.gender == "f":
            return False
        if cls == "fem" and t.gender == "m":
            return False
        if t.thing and not t.person:
            return False
        return True
    if cls == "neut":
        if t.plural or t.person:
            return False
        return bool(t.thing or t.collective or t.gender == "n")
    if cls == "epicene":
        # they/them may be singular or plural, person or group — but a
        # *single* thing is ``it``, never ``they``: pure singular things
        # (thing evidence, not person/plural/collective) are excluded.
        return not (t.thing and not (t.plural or t.collective or t.person))
    return False


def _window_indices(ui: int, n_turns: int, lookback: int,
                    prev_only: bool) -> List[int]:
    """Turn indices that may contribute candidates, oldest first.

    Includes the current unit (``d = 0``) unless ``prev_only`` restricts
    to the immediately previous turn (R0 baseline arm)."""
    if prev_only:
        i = ui - 1
        return [i] if 0 <= i < n_turns else []
    lo = max(0, ui - lookback)
    hi = min(ui, n_turns - 1)
    return list(range(lo, hi + 1))


def _score(canon: str, window: Sequence[int], turns: Sequence[Mapping],
           ui: int, current_canons: frozenset, lookback: int) -> int:
    """Recency + role + previous-turn + co-occurrence score.

    Best single-turn mention dominates (a pronoun binds the nearest
    salient referent); repeat mentions add a capped bonus."""
    best = 0
    hits = 0
    for ti in window:
        turn = turns[ti]
        cm = _turn_canons(turn)
        if canon not in cm:
            continue
        hits += 1
        d = ui - ti
        pos = cm.index(canon)
        s = _W_TURN * (lookback + 1 - d)
        s += _W_SUBJECT if _role(turn, canon, pos) == "subject" else _W_OBJECT
        if d == 1:
            s += _W_PREV_TURN
        if d >= 1 and current_canons and (set(cm) & current_canons):
            s += _W_COOC
        if s > best:
            best = s
    if hits:
        best += min(hits, _W_FREQ_CAP)
    return best


def _resolve(mention: str, unit_index: int,
             session_turns: Sequence[Mapping], lookback: int,
             gender: Optional[Mapping[str, str]],
             kind: Optional[Mapping[str, str]],
             number: Optional[Mapping[str, str]],
             prev_only: bool) -> Tuple[Optional[str], str, dict]:
    """Core sieve → (canon|None, status, detail)."""
    m = _fold(mention).strip()
    turns = [t if isinstance(t, Mapping) else {} for t in (session_turns or ())]
    n = len(turns)
    if not m:
        return None, "no_input", {}
    if n == 0:
        return None, "no_session", {}
    ui = int(unit_index)
    if ui < 0:
        return None, "bad_unit_index", {}
    lookback = max(0, int(lookback))
    gmap, kmap, nmap = _norm_map(gender), _norm_map(kind), _norm_map(number)

    current = turns[ui] if ui < n else {}
    speaker = _turn_speaker(current)
    current_canons = frozenset(_turn_canons(current))
    window = _window_indices(ui, n, lookback, prev_only)

    # Ordered, de-duplicated candidate canons inside the window.
    seen: Dict[str, None] = {}
    for ti in window:
        for c in _turn_canons(turns[ti]):
            if c and c not in seen:
                seen[c] = None
    candidates = list(seen)

    speakers = frozenset(
        s for s in (_turn_speaker(t) for t in turns) if s)

    cls = _FORM_CLASS.get(m)
    if cls is None:
        if m in candidates:
            return m, "identity", {"canon": m}
        if m in _UNSUPPORTED_FORMS:
            return None, "unsupported_form", {}
        det = _tokens(m)[:1]
        if det and det[0] in _DESC_STOP:
            return _resolve_description(m, ui, turns, window,
                                        current_canons, speaker,
                                        lookback, gmap, kmap, nmap,
                                        speakers)
        return None, "unsupported_form", {}

    if cls == "first_sg":
        return (speaker, "speaker" if speaker else "no_speaker",
                {"canon": speaker})
    if cls == "first_pl":
        return None, "group_reference", {}
    if cls == "second":
        others = []
        for ti in window:
            if ti == ui:
                continue
            s = _turn_speaker(turns[ti])
            if s and s != speaker and s not in others:
                others.append(s)
        if len(others) == 1:
            return others[0], "addressee", {"canon": others[0]}
        return None, ("no_addressee" if not others
                      else "ambiguous_addressee"), {"others": others}

    # ---- third-person classes -----------------------------------------
    req_gender = _REQ_GENDER[cls]
    cand = [c for c in candidates if c != speaker]
    traits = {c: _canon_traits(c, speakers, gmap, kmap, nmap)
              for c in cand}
    survivors = [c for c in cand if _passes(traits[c], cls)]
    detail = {"class": cls, "survivors": survivors,
              "candidates": cand}
    if not survivors:
        return None, "no_candidates", detail
    if len(survivors) == 1:
        return survivors[0], "resolved_single", detail | {
            "canon": survivors[0]}
    if cls == "epicene":
        # With ≥2 surviving candidates "they" may denote the *union* of
        # them — a referent no single canon carries. Resolving to one
        # member is the classic over-merge; the sieve abstains outright.
        return None, "group_ambiguity", detail

    scored = sorted(
        ((c, _score(c, window, turns, ui, current_canons, lookback))
         for c in survivors),
        key=lambda cs: (-cs[1], cs[0]))
    detail["scores"] = {c: s for c, s in scored}
    top, top_s = scored[0]
    runner, runner_s = scored[1]
    # (f) abstention: gendered pronoun whose winner's gender is not
    # verified-matching abstains on ANY surviving competition — picking
    # between two unknown-gender canons is a coin flip.
    if req_gender in ("m", "f") and traits[top].gender != req_gender:
        return None, "unverified_gender_competition", detail
    if top_s - runner_s <= _MARGIN:
        return None, "margin", detail
    return top, "resolved_margin", detail | {"canon": top}


def resolve_antecedent(mention: str, unit_index: int,
                       session_turns: Sequence[Mapping],
                       lookback: int = DEFAULT_LOOKBACK, *,
                       gender: Optional[Mapping[str, str]] = None,
                       kind: Optional[Mapping[str, str]] = None,
                       number: Optional[Mapping[str, str]] = None,
                       last_resort_previous_turn: bool = False
                       ) -> Optional[str]:
    """Resolve a pronoun/description mention to a session canon.

    Returns the canon key or ``None`` when the sieve abstains (≥2
    plausible candidates, no candidate, or an unsupported form).
    ``last_resort_previous_turn=True`` is the R0 previous-turn-only rule,
    retained as the Q8 baseline arm.
    """
    canon, _status, _detail = _resolve(
        mention, unit_index, session_turns, lookback,
        gender, kind, number, last_resort_previous_turn)
    return canon


def explain(mention: str, unit_index: int,
            session_turns: Sequence[Mapping],
            lookback: int = DEFAULT_LOOKBACK, *,
            gender: Optional[Mapping[str, str]] = None,
            kind: Optional[Mapping[str, str]] = None,
            number: Optional[Mapping[str, str]] = None,
            last_resort_previous_turn: bool = False) -> dict:
    """Debug/test view: ``{"canon", "status", "detail", "sieve"}``."""
    canon, status, detail = _resolve(
        mention, unit_index, session_turns, lookback,
        gender, kind, number, last_resort_previous_turn)
    return {"sieve": SIEVE_ID, "mention": mention, "canon": canon,
            "status": status, "detail": detail}


def _resolve_description(m: str, ui: int, turns: Sequence[Mapping],
                         window: Sequence[int],
                         current_canons: frozenset,
                         speaker: Optional[str], lookback: int,
                         gmap: Dict[str, str], kmap: Dict[str, str],
                         nmap: Dict[str, str],
                         speakers: frozenset) -> Tuple[Optional[str], str, dict]:
    """Definite descriptions ("the woman from work").

    A description resolves only when exactly one candidate canon
    discriminates on attribute terms — strictly greater attribute
    overlap than every rival. Ties abstain; zero overlap abstains."""
    toks = _tokens(m)
    attrs = {t for t in toks if t not in _DESC_STOP}
    if not attrs:
        return None, "no_attributes", {}

    # gender/number hints carried by the description's own morphology
    req_gender = None
    if attrs & _FEM_NOUNS and not attrs & _MASC_NOUNS:
        req_gender = "f"
    elif attrs & _MASC_NOUNS and not attrs & _FEM_NOUNS:
        req_gender = "m"
    plural_req = bool(attrs & (_PLURAL_NOUNS | _PLURAL_MARKERS))

    cand: List[str] = []
    seen: Dict[str, None] = {}
    for ti in window:
        for c in _turn_canons(turns[ti]):
            if c and c not in seen:
                seen[c] = None
                cand.append(c)

    rows = []
    for c in cand:
        t = _canon_traits(c, speakers, gmap, kmap, nmap)
        if req_gender == "m" and t.gender == "f":
            continue
        if req_gender == "f" and t.gender == "m":
            continue
        if req_gender and (t.plural or t.collective):
            continue
        if plural_req:
            if not (t.plural or t.collective):
                continue
        elif t.plural:
            continue
        overlap = len(attrs & set(_tokens(c)))
        if overlap < 1:
            continue
        sc = _score(c, window, turns, ui, current_canons, lookback)
        rows.append((c, overlap, sc))

    if not rows:
        return None, "no_attribute_match", {"attrs": sorted(attrs)}
    rows.sort(key=lambda r: (-r[1], -r[2], r[0]))
    if len(rows) > 1 and rows[0][1] == rows[1][1]:
        return None, "attribute_tie", {
            "attrs": sorted(attrs),
            "tied": [r[0] for r in rows if r[1] == rows[0][1]]}
    return rows[0][0], "resolved_description", {
        "canon": rows[0][0], "attrs": sorted(attrs),
        "overlap": rows[0][1]}


def resolve_descriptions(mention: str, unit_index: int,
                         session_turns: Sequence[Mapping],
                         lookback: int = DEFAULT_LOOKBACK, *,
                         gender: Optional[Mapping[str, str]] = None,
                         kind: Optional[Mapping[str, str]] = None,
                         number: Optional[Mapping[str, str]] = None
                         ) -> Optional[str]:
    """Resolve a definite description to a canon, or ``None`` (abstain).

    Only a unique attribute-overlap winner resolves — never a coin
    flip."""
    canon, _status, _detail = _resolve_description(
        _fold(mention).strip(), int(unit_index),
        [t if isinstance(t, Mapping) else {} for t in (session_turns or ())],
        _window_indices(int(unit_index), len(session_turns or ()),
                        max(0, int(lookback)), False),
        frozenset(_turn_canons(session_turns[unit_index]))
        if 0 <= unit_index < len(session_turns or ()) else frozenset(),
        _turn_speaker(session_turns[unit_index])
        if 0 <= unit_index < len(session_turns or ()) else None,
        max(0, int(lookback)), _norm_map(gender), _norm_map(kind),
        _norm_map(number),
        frozenset(s for s in (_turn_speaker(t)
                              for t in (session_turns or ())) if s))
    return canon


__all__ = [
    "SIEVE_ID",
    "FORMULA_STATUS",
    "DEFAULT_LOOKBACK",
    "resolve_antecedent",
    "resolve_descriptions",
    "explain",
]
