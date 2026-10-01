"""Owned entity-resolution fixture generator (SPEC_V7 V7-08.13, SPEC_V7_5 §05 Q4).

Deterministic, stdlib-only generator producing the owned fixture the Q4
formula-search arm is gated on: same-name-different-person pairs (over-merge
gate: ≤ 0.01 over ≥ 100 pairs) and nickname/initial/alias variants that
SHOULD merge (correct-merge recall over ≥ 100 pairs), plus adversarial
near-misses around the ``alias/v1`` decision boundary and pronoun-only
mentions (V7-08.13 requires them in the fixture).

Record model (one JSON object per line)::

    pair_id        unique id, ``sn-*`` same-name / ``mg-*`` merge / ``adv-*``
    stratum        fine-grained slice tag (reported per stratum)
    expected       "merge" | "no_merge" | "abstain"  (ground truth outcome)
    surface_a      surface form of referent A as written in ``context_a``
    surface_b      surface form of referent B as written in ``context_b``
    context_a      unit text containing surface_a
    context_b      unit text containing surface_b
    speaker_a/b    optional speaker for that unit (drives A3 speaker forms)
    extra_units    optional [{text, speaker}] scaffolding units
    known_canons   optional scope vocabulary (entity_canon stand-in)
    caller_aliases optional [[canonical, alias]] caller-supplied rows (A5)
    canons_a/b     resolver-visible canon set attributed to each side —
    rationale      free-text why the pair is labeled as it is
    pronoun_mention  context contains a pronoun co-reference (not load-bearing)
    pronoun_only   one side is attested only by a pronoun (undecidable pair)

``canons_a``/``canons_b`` are authored, not derived: for surfaces the
extractor mangles on purpose (``"C. Vance"`` → mention ``"vance"`` — the
period reads as a sentence boundary) the authored set records what the
resolver will actually attribute to that mention, so pair scoring stays
honest about mechanism.

The generator is pure enumeration over fixed pools — no RNG, no clock, no
I/O beyond the output write.  Regenerating must reproduce the committed
fixture byte-identically (tests assert this).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

GENERATOR_ID = "er_fixture/v1"
FIXTURE_NAME = "entity_resolution_owned.jsonl"
FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

# ---------------------------------------------------------------------------
# pools
# ---------------------------------------------------------------------------

#: Given names used for same-name-different-person strata.
GIVEN = [
    "Caroline", "Ruth", "Daniel", "Priya", "Marcus", "Elena", "Sofia",
    "David", "Nadia", "Felix", "Grace", "Omar", "Ingrid", "Pavel",
    "Lucia", "Hassan", "Miriam", "Jonas", "Tessa", "Ruben", "Alicia",
    "Vera", "Simon", "Leah", "Dmitri", "Clara", "Oscar", "Farah",
    "Hugo", "Bianca", "Stefan", "Amara", "Noor", "Celine", "Mateo",
    "Iris", "Anton", "Selma", "Victor", "Zara",
]

SURNAMES = [
    "Vance", "Chen", "Okafor", "Marsh", "Delacroix", "Whitfield",
    "Nakamura", "Petrov", "Lindqvist", "Osei", "Fitzgerald", "Reyes",
    "Kowalski", "Tanaka", "Beaumont", "Hartley", "Quimby", "Zielinski",
    "Ferreira", "Aldridge", "Novak", "Castellano", "Iwu", "Marchetti",
    "Solano", "Thornbury", "Ellison", "Duval", "McAllister", "Rosenfeld",
    "Holloway", "Brandt",
]

#: (given_A, given_B, shared surname) for shared-initial / shared-surname
#: adversarial strata — the two persons collide on first initial.
SHARED_SURNAME = [
    ("Clara", "Caroline", "Vance"),
    ("Daphne", "Daniel", "Marsh"),
    ("Petra", "Pavel", "Okafor"),
    ("Miriam", "Marco", "Reyes"),
    ("Tessa", "Tomas", "Novak"),
    ("Sofia", "Simon", "Brandt"),
    ("Olive", "Oscar", "Hartley"),
    ("Helena", "Hugo", "Duval"),
    ("Ingrid", "Iris", "Solano"),
    ("Renata", "Ruben", "Castellano"),
    ("Leah", "Lucia", "Beaumont"),
    ("Bianca", "Bruno", "Ferreira"),
    ("Zara", "Zoe", "Whitfield"),
    ("Nadia", "Noor", "Zielinski"),
]

CITIES = [
    "Dover", "Oslo", "Lisbon", "Austin", "Bergen", "Kyoto", "Harare",
    "Galway", "Turin", "Cusco", "Riga", "Tampere", "Oaxaca", "Bruges",
    "Sapporo", "Windhoek",
]

ORGS = [
    "Meridian Health", "Northwind Labs", "Atlas Freight", "Juniper Works",
    "Beacon Point", "Kestrel Bio", "Solstice Fund", "Ironwood Group",
]

PROJECTS = [
    "Falcon", "Zephyr", "Quartz", "Lantern", "Cobalt", "Harvest",
    "Pioneer", "Summit",
]

#: Partner surnames used as co-mentioned entities (A4 needs ≥ 2 shared).
PARTNERS = [
    "Reyes", "Malik", "Petrov", "Okonkwo", "Whitfield", "Novak",
    "Brandt", "Solano",
]

#: Canonical → nickname list (for merge strata).  All invented-person
#: friendly, no benchmark text.
NICKS = [
    ("Katherine", ["Kate", "Katie", "Kat", "Kathy"]),
    ("Elizabeth", ["Liz", "Beth", "Lizzie", "Eliza", "Betsy"]),
    ("William", ["Bill", "Will", "Liam", "Billy"]),
    ("Robert", ["Rob", "Bob", "Bobby", "Robbie"]),
    ("Margaret", ["Meg", "Maggie", "Peg", "Peggy"]),
    ("Jennifer", ["Jen", "Jenny"]),
    ("Michael", ["Mike", "Mikey"]),
    ("Christopher", ["Chris", "Topher", "Kit"]),
    ("Patricia", ["Pat", "Patty", "Trish"]),
    ("Richard", ["Rick", "Ricky", "Rich"]),
    ("James", ["Jim", "Jimmy", "Jamie"]),
    ("Susan", ["Sue", "Susie", "Suzy"]),
    ("Deborah", ["Deb", "Debbie"]),
    ("Rebecca", ["Becca", "Becky"]),
    ("Nicholas", ["Nick", "Nicky"]),
    ("Anthony", ["Tony"]),
    ("Joseph", ["Joe", "Joey"]),
    ("Benjamin", ["Ben", "Benji", "Benny"]),
    ("Samuel", ["Sam", "Sammy"]),
    ("Daniel", ["Dan", "Danny"]),
    ("Matthew", ["Matt", "Matty"]),
    ("Andrew", ["Andy", "Drew"]),
    ("Caroline", ["Carrie", "Caro", "Lina"]),
    ("Victoria", ["Vicky", "Tori"]),
    ("Jonathan", ["Jon", "Jonny"]),
    ("Stephanie", ["Steph"]),
    ("Gabrielle", ["Gabby"]),
    ("Isabelle", ["Izzy", "Belle"]),
    ("Nathaniel", ["Nate", "Nat"]),
    ("Theodore", ["Ted", "Teddy", "Theo"]),
    ("Penelope", ["Penny"]),
    ("Alexandra", ["Alex", "Lexi"]),
    ("Maximilian", ["Max"]),
    ("Dominic", ["Dom", "Nic"]),
    ("Sebastian", ["Seb", "Bastian"]),
    ("Gwendolyn", ["Gwen", "Wendy"]),
    ("Jacqueline", ["Jackie"]),
    ("Madeleine", ["Maddie"]),
    ("Timothy", ["Tim", "Timmy"]),
]

#: Same-person spelling variants: edit distance ≤ 1 on the full canon AND
#: every surface ≥ 6 chars → inside A4's documented envelope.
SPELL_VARIANTS = [
    ("Katherine", "Katharine"),
    ("Isobel", "Isabel"),
    ("Kristine", "Kristin"),
    ("Susanne", "Suzanne"),
    ("Nicolle", "Nicole"),
    ("Rachael", "Rachel"),
    ("Debora", "Deborah"),
    ("Michele", "Michelle"),
    ("Carolina", "Caroline"),
    ("Annemarie", "Annemaria"),
    ("Charlotta", "Charlotte"),
    ("Katarina", "Katerina"),
]

#: Object/place names that embed a given name (name-as-object stratum).
OBJECT_TEMPLATES = [
    "Cafe {}", "Project {}", "{} Foundation", "Team {}",
    "{} Gallery", "{} Press",
]

#: Generational suffixes (junior/senior stratum).
GEN_SUFFIXES = ["Jr", "Sr", "II", "III", "IV"]

TITLES = ["Dr", "Prof", "Ms", "Rev", "Sister", "Fr"]


# ---------------------------------------------------------------------------
# context templates (surfaces land exactly as written; co-mentioned orgs,
# projects, partner surnames and cities are capitalized entities the
# extractor will see — they are the "context" a smarter arm could use)
# ---------------------------------------------------------------------------

def _work(name: str, i: int) -> str:
    return (
        f"{name} led the {PROJECTS[i % len(PROJECTS)]} audit at "
        f"{ORGS[i % len(ORGS)]} with {PARTNERS[i % len(PARTNERS)]}."
    )


def _work2(name: str, i: int) -> str:
    return (
        f"{name} presented the {PROJECTS[(i + 3) % len(PROJECTS)]} findings "
        f"to {PARTNERS[(i + 1) % len(PARTNERS)]} and "
        f"{PARTNERS[(i + 5) % len(PARTNERS)]} in {CITIES[i % len(CITIES)]}."
    )


def _family(name: str, i: int) -> str:
    return (
        f"{name} called her sister about the {CITIES[i % len(CITIES)]} "
        f"reunion and the twins' birthday."
    )


def _family2(name: str, i: int) -> str:
    return (
        f"{name} is planning Thanksgiving in "
        f"{CITIES[(i + 7) % len(CITIES)]} with her cousin "
        f"{PARTNERS[(i + 3) % len(PARTNERS)]}."
    )


def _civic(name: str, i: int) -> str:
    return (
        f"{name} coaches the {CITIES[(i + 2) % len(CITIES)]} little league "
        f"team with {PARTNERS[(i + 6) % len(PARTNERS)]}."
    )


# ---------------------------------------------------------------------------
# record builder
# ---------------------------------------------------------------------------

def _rec(
    pair_id: str,
    stratum: str,
    expected: str,
    surface_a: str,
    surface_b: str,
    context_a: str,
    context_b: str,
    *,
    canons_a: Optional[List[str]] = None,
    canons_b: Optional[List[str]] = None,
    speaker_a: Optional[str] = None,
    speaker_b: Optional[str] = None,
    extra_units: Optional[List[Dict[str, Any]]] = None,
    known_canons: Optional[List[str]] = None,
    caller_aliases: Optional[List[List[str]]] = None,
    pronoun_mention: bool = False,
    pronoun_only: bool = False,
    rationale: str = "",
) -> Dict[str, Any]:
    """One fixture record.  ``canons_*`` default to the naive canon of each
    surface (computed locally, same fold as the resolver) — overridden only
    when the extractor demonstrably attributes a different key."""
    # Local copy of canon() so the generator has no verbatim import — the
    # authored values must mirror entities_v2.canon semantics:
    #   possessive strip → NFKC → casefold → diacritic drop →
    #   punctuation/symbol/separator → single space → collapse.
    import unicodedata
    import re as _re

    def _canon(surface: str) -> str:
        s = str(surface or "")
        s = " ".join(
            _re.sub(r"(?:['’][sS]|['’])$", "", tok) for tok in s.split()
        )
        s = unicodedata.normalize("NFKC", s).casefold()
        s = unicodedata.normalize("NFKD", s)
        out = []
        for ch in s:
            if unicodedata.combining(ch):
                continue
            if unicodedata.category(ch)[0] in ("P", "S", "C", "Z"):
                out.append(" ")
            else:
                out.append(ch)
        return _re.sub(r"\s+", " ", "".join(out)).strip()

    return {
        "pair_id": pair_id,
        "stratum": stratum,
        "expected": expected,
        "surface_a": surface_a,
        "surface_b": surface_b,
        "context_a": context_a,
        "context_b": context_b,
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "extra_units": list(extra_units or []),
        "known_canons": list(known_canons or []),
        "caller_aliases": list(caller_aliases or []),
        "canons_a": list(canons_a) if canons_a else [_canon(surface_a)],
        "canons_b": list(canons_b) if canons_b else [_canon(surface_b)],
        "pronoun_mention": bool(pronoun_mention),
        "pronoun_only": bool(pronoun_only),
        "rationale": rationale,
        "generator": GENERATOR_ID,
    }


# ---------------------------------------------------------------------------
# stratum builders — same-name / no-merge side
# ---------------------------------------------------------------------------

def _sn_identical_surface(recs: List[Dict[str, Any]]) -> None:
    """Bare identical given name, two different people, disjoint contexts.

    The canon is identical so the alias layer cannot even represent the
    distinction — scored as ``same_canon`` (canon-level conflation), which
    the report shows separately from active-alias over-merges.
    """
    for i in range(16):
        g = GIVEN[i]
        recs.append(_rec(
            f"sn-iso-{i:03d}", "sn_identical_surface", "no_merge",
            g, g,
            _work(g, i),
            _family(g, i),
            rationale=(
                f"Two distinct {g}s (workplace vs family), identical bare "
                f"surface — canon() already conflates them; the alias layer "
                f"has no decision to make."
            ),
        ))


def _sn_disjoint_full(recs: List[Dict[str, Any]]) -> None:
    """Both full names attested, no shared surface — trivially separate."""
    for i in range(10):
        g = GIVEN[i]
        s1 = SURNAMES[i % len(SURNAMES)]
        s2 = SURNAMES[(i + 11) % len(SURNAMES)]
        a, b = f"{g} {s1}", f"{g} {s2}"
        recs.append(_rec(
            f"sn-dis-{i:03d}", "sn_disjoint_full", "no_merge",
            a, b,
            _work(a, i),
            _family(b, i + 4),
            rationale=(
                f"{a} and {b} are different people; only full forms are "
                f"attested, so no rule can link them."
            ),
        ))


def _sn_subset_overmerge(recs: List[Dict[str, Any]]) -> None:
    """Bare first name of person B vs full name of person A — only A's full
    form is in scope, so A1 token-subset actively merges the bare mention
    into the WRONG canonical.  Ground truth no_merge; the resolver is
    context-blind here by design (the Q4 arm exists because of this)."""
    for i in range(24):
        g = GIVEN[(i + 8) % len(GIVEN)]
        s1 = SURNAMES[(i * 3 + 1) % len(SURNAMES)]
        a = f"{g} {s1}"
        # disjoint co-mentions so a context-aware arm could separate them
        recs.append(_rec(
            f"sn-sub-om-{i:03d}", "sn_subset_overmerge", "no_merge",
            a, g,
            _work(a, i),
            _family(g, i + 6),
            rationale=(
                f"Bare '{g}' in the family context is {g} "
                f"{SURNAMES[(i * 3 + 9) % len(SURNAMES)]} (full form never "
                f"stated in scope); '{a}' is a different person.  A1 "
                f"token-subset still emits an active merge — expected "
                f"over-merge."
            ),
        ))


def _sn_subset_abstain(recs: List[Dict[str, Any]]) -> None:
    """Bare first name + BOTH plausible full canonicals in scope → A1
    proposes twice, A6 demotes both to candidate — the documented
    abstention path (V7-08.13/H101)."""
    for i in range(20):
        g = GIVEN[(i + 4) % len(GIVEN)]
        s1 = SURNAMES[(i * 2 + 2) % len(SURNAMES)]
        s2 = SURNAMES[(i * 2 + 17) % len(SURNAMES)]
        a = f"{g} {s1}"
        recs.append(_rec(
            f"sn-sub-ab-{i:03d}", "sn_subset_abstain", "abstain",
            a, g,
            _work(a, i),
            _family(g, i + 3),
            known_canons=[f"{g} {s1}".lower(), f"{g} {s2}".lower()],
            rationale=(
                f"Bare '{g}' is ambiguous between {g} {s1} and {g} {s2} "
                f"(both canonicals in scope); the correct outcome is a "
                f"reviewable candidate, not a merge."
            ),
        ))


def _sn_junior_senior(recs: List[Dict[str, Any]]) -> None:
    """'Robert Harris' vs 'Robert Harris Jr' — A1 subset merges father and
    son.  Expected no_merge; measured over-merge (hard case by design)."""
    for i in range(12):
        g = GIVEN[(i + 20) % len(GIVEN)]
        s = SURNAMES[(i * 5 + 3) % len(SURNAMES)]
        suf = GEN_SUFFIXES[i % len(GEN_SUFFIXES)]
        a, b = f"{g} {s}", f"{g} {s} {suf}"
        recs.append(_rec(
            f"sn-gen-{i:03d}", "sn_junior_senior", "no_merge",
            a, b,
            _family(a, i),
            _work(b, i),
            rationale=(
                f"'{b}' is a different person from '{a}' ({suf} suffix); "
                f"A1 token-subset cannot see the suffix as disqualifying."
            ),
        ))


def _sn_object_person(recs: List[Dict[str, Any]]) -> None:
    """Name-as-object vs person: 'Cafe Caroline'/'Project Caroline' etc.

    Subset variant: only the object is multi-token and the bare person
    name has no other canonical → A1 actively merges person into object
    (over-merge).  Known variant: person's full form is in scope → A6
    abstain.  Full variant: both multi-token → separate.
    """
    for i in range(6):
        g = GIVEN[(i + 28) % len(GIVEN)]
        obj = OBJECT_TEMPLATES[i % len(OBJECT_TEMPLATES)].format(g)
        recs.append(_rec(
            f"sn-obj-om-{i:03d}", "sn_object_vs_person", "no_merge",
            obj, g,
            f"{obj} reopened on Hill Street and already has a line.",
            _family(g, i + 9),
            rationale=(
                f"'{obj}' is a place/project; bare '{g}' is a person whose "
                f"surname is never stated — A1 merges her into the object "
                f"canon (over-merge)."
            ),
        ))
    for i in range(3):
        g = GIVEN[(i + 33) % len(GIVEN)]
        s = SURNAMES[(i * 7 + 5) % len(SURNAMES)]
        obj = OBJECT_TEMPLATES[(i + 2) % len(OBJECT_TEMPLATES)].format(g)
        recs.append(_rec(
            f"sn-obj-ab-{i:03d}", "sn_object_vs_person", "abstain",
            obj, g,
            f"{obj} reopened on Hill Street and already has a line.",
            _family(g, i + 11),
            known_canons=[f"{g} {s}".lower()],
            rationale=(
                f"'{obj}' (object) and '{g} {s}' (person) both contain the "
                f"bare mention → A6 conflict → correct abstention."
            ),
        ))
    for i in range(3):
        g = GIVEN[(i + 36) % len(GIVEN)]
        s = SURNAMES[(i * 4 + 8) % len(SURNAMES)]
        obj = OBJECT_TEMPLATES[(i + 4) % len(OBJECT_TEMPLATES)].format(g)
        recs.append(_rec(
            f"sn-obj-sep-{i:03d}", "sn_object_vs_person", "no_merge",
            obj, f"{g} {s}",
            f"{obj} reopened on Hill Street and already has a line.",
            _work(f"{g} {s}", i),
            rationale=(
                f"'{obj}' (object) vs '{g} {s}' (person): only multi-token "
                f"forms attested → no rule applies → correct separation."
            ),
        ))


def _sn_shared_surname(recs: List[Dict[str, Any]]) -> None:
    """'C Vance'-style shared-initial pairs (no period — the extractor can
    only form 'c vance' when the period is absent).

    abstain variant: both plausible full canonicals present → A2 fires
    twice → A6.  no_merge variant: only the wrong person is a canonical →
    A2 actively merges the initial form into the wrong canonical.
    """
    for i in range(7):
        ga, gb, s = SHARED_SURNAME[i % len(SHARED_SURNAME)]
        # surface_a = full name of person A (mentioned in work context);
        # surface_b = the ambiguous initial form whose truth is person B
        recs.append(_rec(
            f"sn-surn-ab-{i:03d}", "sn_shared_surname", "abstain",
            f"{ga} {s}", f"{ga[0]} {s}",
            _work(f"{ga} {s}", i),
            f"The invoice came from {ga[0]} {s} on Tuesday.",
            known_canons=[f"{ga} {s}".lower(), f"{gb} {s}".lower()],
            rationale=(
                f"'{ga[0]} {s}' is really {gb} {s}; both {ga} and {gb} "
                f"{s} are canonicals → A2 proposes both → A6 abstain."
            ),
        ))
    for i in range(7):
        ga, gb, s = SHARED_SURNAME[(i + 7) % len(SHARED_SURNAME)]
        # "I Solano"-style: initials "a"/"i" are in _NEVER, so the
        # extractor emits the bare surname — the authored canon set
        # records what the resolver will actually see.
        never_initial = gb[0].lower() in ("a", "i")
        recs.append(_rec(
            f"sn-surn-om-{i:03d}", "sn_shared_surname", "no_merge",
            f"{ga} {s}", f"{gb[0]} {s}",
            _work(f"{ga} {s}", i + 2),
            f"The invoice came from {gb[0]} {s} on Tuesday.",
            canons_b=[s.lower()] if never_initial else None,
            rationale=(
                f"'{gb[0]} {s}' is really {gb} {s}, but only '{ga} {s}' "
                f"is a canonical — the initial form merges the invoice "
                f"sender into the wrong person (over-merge)."
                + (" The initial 'I' is in the extractor's never-list, "
                   "so the mention lands as the bare surname and merges "
                   "via A1, not A2." if never_initial else "")
            ),
        ))


def _sn_initial_period(recs: List[Dict[str, Any]]) -> None:
    """'C. Vance' — the period reads as a sentence boundary, so the
    extractor emits the bare surname 'vance' (never 'c vance'); A2's
    documented 'f. last' form is unreachable with a period.  The surname
    mention then A1-subsets into every matching canonical."""
    for i in range(4):
        ga, gb, s = SHARED_SURNAME[i % len(SHARED_SURNAME)]
        recs.append(_rec(
            f"sn-ini-ab-{i:03d}", "sn_initial_period", "abstain",
            f"{ga} {s}", f"{gb[0]}. {s}",
            _work(f"{ga} {s}", i + 1),
            f"The invoice came from {gb[0]}. {s} on Tuesday.",
            canons_b=[s.lower()],
            known_canons=[f"{ga} {s}".lower(), f"{gb} {s}".lower()],
            rationale=(
                f"'{gb[0]}. {s}' extracts as bare '{s}' (period breaks "
                f"the run); both {ga}/{gb} {s} canonicals → A6 abstain."
            ),
        ))
    for i in range(4):
        ga, gb, s = SHARED_SURNAME[(i + 4) % len(SHARED_SURNAME)]
        recs.append(_rec(
            f"sn-ini-om-{i:03d}", "sn_initial_period", "no_merge",
            f"{ga} {s}", f"{gb[0]}. {s}",
            _work(f"{ga} {s}", i + 5),
            f"The invoice came from {gb[0]}. {s} on Tuesday.",
            canons_b=[s.lower()],
            rationale=(
                f"'{gb[0]}. {s}' extracts as bare '{s}'; only '{ga} {s}' "
                f"is a canonical → A1 actively merges person {gb}'s "
                f"mention into {ga}'s canonical (over-merge)."
            ),
        ))


def _sn_near_spelling_strangers(recs: List[Dict[str, Any]]) -> None:
    """Edit-distance-1 full names of DIFFERENT people.  With ≥ 2 shared
    co-mentions A4 proposes a candidate (abstain — correct); without them
    no rule fires (separate)."""
    for i in range(6):
        va, vb = SPELL_VARIANTS[i % len(SPELL_VARIANTS)]
        s = SURNAMES[(i * 6 + 1) % len(SURNAMES)]
        p1, p2 = PARTNERS[i % len(PARTNERS)], PARTNERS[(i + 3) % len(PARTNERS)]
        a, b = f"{va} {s}", f"{vb} {s}"
        recs.append(_rec(
            f"sn-spell-ab-{i:03d}", "sn_near_spelling_strangers", "abstain",
            a, b,
            f"{a} chaired the review with {p1} and {p2}.",
            f"{b} chaired the review with {p1} and {p2}.",
            rationale=(
                f"{a} and {b} are two different people (twin analysts); "
                f"the identical co-mentions make them A4 candidates — "
                f"abstain is the documented, correct outcome."
            ),
        ))
    for i in range(4):
        va, vb = SPELL_VARIANTS[(i + 6) % len(SPELL_VARIANTS)]
        s = SURNAMES[(i * 6 + 7) % len(SURNAMES)]
        a, b = f"{va} {s}", f"{vb} {s}"
        recs.append(_rec(
            f"sn-spell-sep-{i:03d}", "sn_near_spelling_strangers", "no_merge",
            a, b,
            _work(a, i),
            _family(b, i + 2),
            rationale=(
                f"{a} and {b} are different people in disjoint contexts; "
                f"A4 needs ≥ 2 shared co-mentions and has none → separate."
            ),
        ))


def _sn_pronoun_only(recs: List[Dict[str, Any]]) -> None:
    """Person B attested only by a pronoun — undecidable pair.  Expected
    abstain (leave separate); any active link is a serious error."""
    for i in range(8):
        g = GIVEN[(i + 12) % len(GIVEN)]
        s = SURNAMES[(i * 3 + 4) % len(SURNAMES)]
        a = f"{g} {s}"
        recs.append(_rec(
            f"sn-pron-{i:03d}", "sn_pronoun_only", "abstain",
            a, g,
            _work(a, i),
            "She brought the twins to the picnic and stayed until dark.",
            pronoun_only=True,
            rationale=(
                f"'{a}' is attested; the other '{g}' appears only as "
                f"'She' — a pronoun carries no canon, so the pair is "
                f"undecidable and must stay separate."
            ),
        ))


def _sn_boundary_misc(recs: List[Dict[str, Any]]) -> None:
    """One-token-different / hyphenation boundary cases."""
    cases = [
        # hyphenated vs base surname, DIFFERENT people → A1 over-merge
        ("Caroline Vance-Lee", "Caroline Vance", "no_merge",
         "Caroline Vance-Lee argued the appeal before the panel.",
         "Caroline Vance filed the dissent in the same court.",
         "Hyphenated double-barrel surname vs the base surname — two "
         "different attorneys; A1 token-subset merges them."),
        ("Michael Reyes-Petrov", "Michael Reyes", "no_merge",
         "Michael Reyes-Petrov chaired the benefits committee.",
         "Michael Reyes chaired the audit committee.",
         "Compound vs simple surname, different people → A1 over-merge."),
        ("Sofia Brandt-Keller", "Sofia Brandt", "no_merge",
         "Sofia Brandt-Keller led the Zurich office.",
         "Sofia Brandt led the Geneva office.",
         "Compound vs simple surname, different people → A1 over-merge."),
        # same structure but single canonical → first-name A1 over-merge
        ("Daniel Marsh", "Daniel Marsh-Lopez", "no_merge",
         "Daniel Marsh retired from the firm in March.",
         "Daniel Marsh-Lopez joined the firm in April.",
         "Father-son style shared name, different people → A1 merges "
         "'daniel marsh' into 'daniel marsh lopez'."),
        # genuinely separable near-misses
        ("Ann Marie Solano", "Annemarie Solano", "no_merge",
         "Ann Marie Solano presented the poster.",
         "Annemarie Solano presented the keynote.",
         "Two-token 'Ann Marie' vs one-token 'Annemarie' — different "
         "people; canons differ by more than one edit → separate."),
        ("Caroline Mae Vance", "Caroline Rae Vance", "no_merge",
         "Caroline Mae Vance signed the deed.",
         "Caroline Rae Vance witnessed the deed.",
         "Identical first+last, different middle name — two sisters; "
         "no rule links them → separate."),
        ("Maria del Carmen", "Maria Carmen Reyes", "no_merge",
         "Maria del Carmen chaired the session.",
         "Maria Carmen Reyes chaired the panel.",
         "Particle surname vs compound — different people; 'maria' is "
         "not a proper subset issue because both canons are multi-token "
         "and no bare form is attested."),
        ("Jan Smits", "Jan Smit", "no_merge",
         "Jan Smits reviewed the manuscript.",
         "Jan Smit reviewed the galley.",
         "Edit-1 surnames, different people; full canons 'jan smits' vs "
         "'jan smit' are ≥ 6 chars and edit ≤ 1 → A4 needs shared "
         "co-mentions it does not have → separate."),
    ]
    for i, (a, b, exp, ca, cb, rat) in enumerate(cases):
        recs.append(_rec(
            f"sn-bnd-{i:03d}", "sn_boundary_misc", exp,
            a, b, ca, cb, rationale=rat,
        ))


# ---------------------------------------------------------------------------
# stratum builders — merge side (nickname / alias / initial variants)
# ---------------------------------------------------------------------------

def _mg_token_subset(recs: List[Dict[str, Any]]) -> None:
    """'Alice Chen' ↔ 'Alice' — same person; A1 active merge."""
    for i in range(20):
        g = GIVEN[(i + 6) % len(GIVEN)]
        s = SURNAMES[(i * 2 + 6) % len(SURNAMES)]
        full = f"{g} {s}"
        proj = PROJECTS[i % len(PROJECTS)]
        recs.append(_rec(
            f"mg-sub-{i:03d}", "mg_token_subset", "merge",
            full, g,
            _work(full, i),
            f"{g} filed the {proj} minutes afterwards.",
            rationale=(
                f"'{g}' is '{full}' — the only {g}-canonical in scope; "
                f"A1 token-subset is the documented merge."
            ),
        ))


def _mg_middle_name(recs: List[Dict[str, Any]]) -> None:
    """'Mary Jane Watson' ↔ 'Mary Jane' / 'Mary' — A1 subset chain."""
    mids = ["Jane", "Louise", "Anne", "Rae", "Mae", "Grace"]
    for i in range(6):
        g = GIVEN[(i + 2) % len(GIVEN)]
        m = mids[i % len(mids)]
        s = SURNAMES[(i * 4 + 3) % len(SURNAMES)]
        full = f"{g} {m} {s}"
        partial = f"{g} {m}"
        recs.append(_rec(
            f"mg-mid-{i:03d}", "mg_middle_name", "merge",
            full, partial,
            _work(full, i),
            f"{partial} took the notes for the session.",
            rationale=(
                f"'{partial}' is '{full}'; first+middle ⊂ "
                f"first+middle+last → A1 merge."
            ),
        ))


def _mg_first_plus_initial(recs: List[Dict[str, Any]]) -> None:
    """'Alice C' ↔ 'Alice Chen' — A2 first+last-initial (documented form
    works whether or not a trailing period is present)."""
    # surnames whose initial is "a"/"i" are excluded — those initials are
    # in the extractor's _NEVER list, so "Alice I" can never form
    # (covered as a documented miss under adv_initial_never instead).
    ok_surnames = [s for s in SURNAMES if s[0].lower() not in ("a", "i")]
    for i in range(12):
        g = GIVEN[(i + 10) % len(GIVEN)]
        s = ok_surnames[(i * 5 + 2) % len(ok_surnames)]
        full = f"{g} {s}"
        ini = f"{g} {s[0]}"
        recs.append(_rec(
            f"mg-fini-{i:03d}", "mg_first_plus_initial", "merge",
            full, ini,
            _work(full, i),
            f"The invoice came from {ini} on Tuesday.",
            rationale=(
                f"'{ini}' is '{full}'; A2 first+last-initial matches the "
                f"unique canonical."
            ),
        ))


def _mg_initial_plus_last(recs: List[Dict[str, Any]]) -> None:
    """'J Chen' ↔ 'Jordan Chen' — A2 first-initial+last, NO period (the
    period breaks the cap run — see sn_initial_period/adv_initial_period)."""
    pool = [
        ("Jordan", "Chen"), ("Marcus", "Reyes"), ("Ruth", "Novak"),
        ("Tessa", "Brandt"), ("Victor", "Marsh"), ("Leah", "Okafor"),
        ("Hugo", "Petrov"), ("Selma", "Vance"), ("Mateo", "Duval"),
        ("Farah", "Ellison"), ("Simon", "Iwu"), ("Clara", "Solano"),
    ]
    for i, (g, s) in enumerate(pool):
        full = f"{g} {s}"
        ini = f"{g[0]} {s}"
        recs.append(_rec(
            f"mg-lini-{i:03d}", "mg_initial_plus_last", "merge",
            full, ini,
            _work(full, i),
            f"The package went to {ini} first.",
            rationale=(
                f"'{ini}' is '{full}'; A2 first-initial+last matches the "
                f"unique canonical (period-free form, since 'X. {s}' "
                f"extracts as bare '{s}')."
            ),
        ))


def _mg_goes_by(recs: List[Dict[str, Any]]) -> None:
    """'Y goes by X' — A3 named-subject explicit statement."""
    pairs = []
    for canon_name, nicks in NICKS:
        for n in nicks:
            pairs.append((canon_name, n))
    for i in range(12):
        g, n = pairs[i]
        s = SURNAMES[(i * 3 + 7) % len(SURNAMES)]
        full = f"{g} {s}"
        recs.append(_rec(
            f"mg-goes-{i:03d}", "mg_goes_by", "merge",
            full, n,
            f"{full} goes by {n} at the studio.",
            f"{n} booked the studio for Friday.",
            rationale=(
                f"Explicit 'goes by' statement binds '{n}' to '{full}' "
                f"(A3 active); '{n}' appears again on its own."
            ),
        ))


def _mg_call_me(recs: List[Dict[str, Any]]) -> None:
    """'call me X' with exactly one speaker canon → A3 active."""
    pairs = []
    for canon_name, nicks in NICKS:
        for n in nicks:
            pairs.append((canon_name, n))
    for i in range(10):
        g, n = pairs[(i + 12) % len(pairs)]
        s = SURNAMES[(i * 2 + 4) % len(SURNAMES)]
        full = f"{g} {s}"
        recs.append(_rec(
            f"mg-call-{i:03d}", "mg_call_me", "merge",
            full, n,
            f"Hi, call me {n}.",
            f"{n} confirmed the booking.",
            speaker_a=full,
            rationale=(
                f"Speaker-relative 'call me {n}' resolves against the "
                f"single speaker canon '{full}' (A3 active)."
            ),
        ))


def _mg_my_name(recs: List[Dict[str, Any]]) -> None:
    """'my name is X' with one speaker → A3 active."""
    pairs = []
    for canon_name, nicks in NICKS:
        for n in nicks:
            pairs.append((canon_name, n))
    for i in range(8):
        g, n = pairs[(i + 40) % len(pairs)]
        s = SURNAMES[(i * 6 + 9) % len(SURNAMES)]
        full = f"{g} {s}"
        recs.append(_rec(
            f"mg-name-{i:03d}", "mg_my_name", "merge",
            full, n,
            f"Actually, my name is {n}.",
            f"{n} chaired the call.",
            speaker_a=full,
            rationale=(
                f"'my name is {n}' resolves against the single speaker "
                f"canon '{full}' (A3 active)."
            ),
        ))


def _mg_paren(recs: List[Dict[str, Any]]) -> None:
    """'Y (X)' — documented A3 pattern, but the cap-run extractor glues
    the parenthetical into the name run ('Katherine Vance (Kate)' →
    mention 'katherine vance kate'), creating a phantom canonical that
    A6-conflicts the explicit pair → candidate, never active.  Ground
    truth merge; resolver abstains — measured as an abstained miss."""
    pairs = []
    for canon_name, nicks in NICKS:
        for n in nicks:
            pairs.append((canon_name, n))
    for i in range(10):
        g, n = pairs[(i + 60) % len(pairs)]
        s = SURNAMES[(i * 4 + 1) % len(SURNAMES)]
        full = f"{g} {s}"
        recs.append(_rec(
            f"mg-paren-{i:03d}", "mg_paren", "merge",
            full, n,
            f"{full} ({n}) led the review.",
            f"{n} filed the minutes.",
            rationale=(
                f"'{full} ({n})' is an explicit statement (A3), but the "
                f"extractor glues the parenthetical into the cap run and "
                f"the phantom canonical forces A6 → abstain, not merge."
            ),
        ))


def _mg_for_short(recs: List[Dict[str, Any]]) -> None:
    """'X for short' — antecedent is the nearest capitalized TOKEN, so it
    binds single-token canonicals correctly ('Melanie prefers Mel for
    short' → melanie→mel)."""
    single = [
        ("Melanie", "Mel"), ("Katherine", "Kat"), ("Jennifer", "Jen"),
        ("Rebecca", "Becca"), ("Stephanie", "Steph"), ("Gabrielle", "Gabby"),
        ("Penelope", "Penny"), ("Madeleine", "Maddie"),
    ]
    for i, (g, n) in enumerate(single):
        recs.append(_rec(
            f"mg-short-{i:03d}", "mg_for_short", "merge",
            g, n,
            f"{g} prefers {n} for short.",
            f"{n} booked the studio for Friday.",
            rationale=(
                f"'{g} prefers {n} for short' — antecedent '{g}' is the "
                f"nearest capitalized token → A3 active merge."
            ),
        ))


def _mg_caller(recs: List[Dict[str, Any]]) -> None:
    """Caller-supplied alias rows → A5 active (highest precedence rule)."""
    pairs = []
    for canon_name, nicks in NICKS:
        for n in nicks:
            pairs.append((canon_name, n))
    for i in range(10):
        g, n = pairs[(i + 75) % len(pairs)]
        s = SURNAMES[(i * 3 + 2) % len(SURNAMES)]
        full = f"{g} {s}"
        recs.append(_rec(
            f"mg-caller-{i:03d}", "mg_caller", "merge",
            full, n,
            _work(full, i),
            f"{n} confirmed the booking.",
            caller_aliases=[[full, n]],
            rationale=(
                f"The caller asserted '{n}' is '{full}' → A5 active, "
                f"method=caller."
            ),
        ))


def _mg_near_spelling(recs: List[Dict[str, Any]]) -> None:
    """Same-person spelling variants inside A4's envelope (edit ≤ 1, both
    ≥ 6 chars, ≥ 2 shared co-mentions) — A4 emits a CANDIDATE by design,
    so the resolver abstains on ground-truth merges."""
    for i, (va, vb) in enumerate(SPELL_VARIANTS):
        s = SURNAMES[(i * 3 + 5) % len(SURNAMES)]
        p1, p2 = PARTNERS[i % len(PARTNERS)], PARTNERS[(i + 3) % len(PARTNERS)]
        a, b = f"{va} {s}", f"{vb} {s}"
        recs.append(_rec(
            f"mg-spell-{i:03d}", "mg_near_spelling", "merge",
            a, b,
            f"{a} chaired the review with {p1} and {p2}.",
            f"{b} chaired the review with {p1} and {p2}.",
            rationale=(
                f"'{a}' and '{b}' are the same person spelled two ways; "
                f"A4 proposes a candidate (never auto-active) → abstained "
                f"miss against ground truth."
            ),
        ))


def _mg_nickname_unsignaled(recs: List[Dict[str, Any]]) -> None:
    """Nickname ↔ formal with NO explicit statement — 'Kate'/'Katherine'
    share no rule path (not subset, not initial, edit > 1 or < 6 chars),
    so the resolver stays silent: the recall gap the Q4 nickname level
    exists to close."""
    pairs = []
    for canon_name, nicks in NICKS:
        for n in nicks:
            pairs.append((canon_name, n))
    for i in range(16):
        g, n = pairs[(i + 25) % len(pairs)]
        s = SURNAMES[(i * 4 + 6) % len(SURNAMES)]
        p1, p2 = PARTNERS[i % len(PARTNERS)], PARTNERS[(i + 4) % len(PARTNERS)]
        full = f"{g} {s}"
        recs.append(_rec(
            f"mg-nick-{i:03d}", "mg_nickname_unsignaled", "merge",
            full, n,
            f"{full} ran the {PROJECTS[i % len(PROJECTS)]} review with "
            f"{p1} and {p2}.",
            f"{n} ran the {PROJECTS[i % len(PROJECTS)]} review with "
            f"{p1} and {p2}.",
            rationale=(
                f"'{n}' is '{full}' — identical context, but no alias/v1 "
                f"rule covers unsignaled nicknames → silent miss."
            ),
        ))


def _mg_surface_fold(recs: List[Dict[str, Any]]) -> None:
    """Surface variants that fold to the same canon — canon-level merges,
    trivially correct (case, possessive, hyphen-as-separator)."""
    cases = [
        ("CAROLINE", "Caroline", "CAROLINE sent the memo.",
         "Caroline sent the memo.", "casefold"),
        ("Caroline's", "Caroline", "Caroline's mural won the prize.",
         "Caroline accepted the award.", "possessive strip"),
        ("Mary-Kate", "Mary Kate", "Mary-Kate opened the show.",
         "Mary Kate opened the show.", "hyphen → separator fold"),
        ("RENÉE", "Renée", "RENÉE chaired the panel.",
         "Renée chaired the panel.", "casefold + diacritic identity"),
        ("Osei", "OSEI", "Osei filed the report.",
         "OSEI filed the report.", "casefold"),
        ("Van der Berg", "VAN DER BERG",
         "The ledger went to Van der Berg.",
         "VAN DER BERG signed the ledger.",
         "casefold multi-token particle surname"),
        ("McAllister", "Mcallister", "McAllister joined the board.",
         "Mcallister joined the board.", "casefold"),
        ("James'", "James", "That was James' desk.",
         "James cleared his desk.", "bare trailing apostrophe strip"),
    ]
    for i, (a, b, ca, cb, why) in enumerate(cases):
        recs.append(_rec(
            f"mg-fold-{i:03d}", "mg_surface_fold", "merge",
            a, b, ca, cb,
            rationale=f"Same person; surfaces coincide under canon() ({why}).",
        ))


def _mg_title_prefix(recs: List[Dict[str, Any]]) -> None:
    """'Dr Caroline Vance' ↔ 'Caroline Vance' — the titled form is a
    token superset → A1 merges it into the untitled canonical."""
    for i in range(6):
        g = GIVEN[(i + 14) % len(GIVEN)]
        s = SURNAMES[(i * 5 + 7) % len(SURNAMES)]
        t = TITLES[i % len(TITLES)]
        full = f"{g} {s}"
        titled = f"{t} {full}"
        recs.append(_rec(
            f"mg-title-{i:03d}", "mg_title_prefix", "merge",
            titled, full,
            f"{titled} chaired the panel.",
            f"{full} chaired the panel.",
            rationale=(
                f"'{titled}' is '{full}' with a title; A1 token-subset "
                f"merges the longer canon into the base one."
            ),
        ))


# ---------------------------------------------------------------------------
# adversarial near-misses around the decision boundary
# ---------------------------------------------------------------------------

def _adv(recs: List[Dict[str, Any]]) -> None:
    """Cases that sit exactly on (or just past) alias/v1's decision
    boundary, including documented-quirk shapes.  Expected labels reflect
    ground truth; several will be misses by design."""
    pairs = []
    for canon_name, nicks in NICKS:
        for n in nicks:
            pairs.append((canon_name, n))

    # 1-3: 'call me X <word>' — IGNORECASE lets the capture group swallow
    # a following lowercase word → alias 'kate tomorrow' never links 'kate'.
    for i, (g, n) in enumerate([pairs[3], pairs[17], pairs[41]]):
        s = SURNAMES[(i * 5 + 3) % len(SURNAMES)]
        full = f"{g} {s}"
        recs.append(_rec(
            f"adv-call-{i:03d}", "adv_callme_trailing", "merge",
            full, n,
            f"Hi, call me {n} tomorrow.",
            f"{n} confirmed the booking.",
            speaker_a=full,
            rationale=(
                f"'call me {n} tomorrow' — the A3 capture swallows the "
                f"trailing lowercase word (IGNORECASE reaches the "
                f"capitalized-continuation class) → alias '{n} tomorrow' "
                f"never links '{n}' — a documented-rule violation worth "
                f"reporting."
            ),
        ))

    # 4-6: 'J. Chen' period-initial, same person — extractor emits the
    # bare surname; A1 merges it into the unique matching canonical.
    # Correct outcome, wrong mechanism (any 'Chen' would match).
    for i, (g, s) in enumerate([("Jordan", "Chen"), ("Maria", "Reyes"),
                                ("Tomas", "Novak")]):
        full = f"{g} {s}"
        recs.append(_rec(
            f"adv-peri-{i:03d}", "adv_initial_period", "merge",
            full, f"{g[0]}. {s}",
            _work(full, i + 10),
            f"The package went to {g[0]}. {s} first.",
            canons_b=[s.lower()],
            rationale=(
                f"'{g[0]}. {s}' extracts as bare '{s}' (period boundary) "
                f"→ A1 surname merge, not A2 — correct merge via a looser "
                f"mechanism."
            ),
        ))

    # 7-8: 'A'-initial never extracts (in _NEVER) → silent miss.
    for i, (g, s) in enumerate([("Alicia", "Chen"), ("Anna", "Reyes")]):
        full = f"{g} {s}"
        recs.append(_rec(
            f"adv-anever-{i:03d}", "adv_initial_never", "merge",
            full, g[0],
            _work(full, i + 20),
            f"{g[0]} sent the invoice on Tuesday.",
            rationale=(
                f"'{g[0]}' is a single-letter mention of '{full}'; the "
                f"extractor suppresses 'a'/standalone initials → silent "
                f"miss."
            ),
        ))

    # 9-10: 'She goes by X' — the regex binds the PRONOUN as canonical
    # (comment says it should resolve to the speaker) → row (she→x),
    # pair misses and 'she' collects a garbage alias.
    for i, (g, n) in enumerate([("Melanie", "Mel"), ("Katherine", "Kate")]):
        s = SURNAMES[(i * 7 + 4) % len(SURNAMES)]
        full = f"{g} {s}"
        recs.append(_rec(
            f"adv-she-{i:03d}", "adv_pronoun_goes_by", "merge",
            full, n,
            f"She goes by {n} at the studio.",
            f"{n} booked the studio.",
            speaker_a=full,
            rationale=(
                f"'She goes by {n}' binds canon 'she' as the canonical "
                f"(documented intent was the speaker) → pair misses and "
                f"a stray (she→{n.lower()}) row appears."
            ),
        ))

    # 11-12: 'X for short' with a multi-token name — antecedent is the
    # nearest capitalized TOKEN ('Cross'), not the full name → miss.
    for i, (g, s, n) in enumerate([("Melanie", "Cross", "Mel"),
                                   ("Katherine", "Vance", "Kate")]):
        full = f"{g} {s}"
        recs.append(_rec(
            f"adv-short-{i:03d}", "adv_for_short_antecedent", "merge",
            full, n,
            f"{full}, {n} for short, runs the studio.",
            f"{n} booked the studio.",
            rationale=(
                f"'{full}, {n} for short' — the antecedent resolves to "
                f"the single token '{s}', so the row is "
                f"({s.lower()}→{n.lower()}), never the pair link."
            ),
        ))

    # 13: apostrophe folding — 'O'Brien' vs 'OBrien' are different canons.
    recs.append(_rec(
        "adv-apos-000", "adv_apostrophe", "merge",
        "O'Brien", "OBrien",
        "O'Brien filed the brief.",
        "OBrien filed the brief.",
        rationale=(
            "Same person; canon('O'Brien')='o brien' ≠ canon('OBrien')="
            "'obrien' — the apostrophe is an internal separator, so the "
            "surfaces never meet."
        ),
    ))

    # 14: 'St. John' — the period breaks the cap run, so the surname is
    # never extracted as a unit → silent miss against the same person.
    recs.append(_rec(
        "adv-st-000", "adv_st_name", "merge",
        "Caroline St. John", "Caroline St John",
        "Caroline St. John chaired the session.",
        "Caroline St John chaired the session.",
        canons_a=["caroline st", "john"],
        rationale=(
            "Same person; 'St.' reads as a sentence boundary so the "
            "extractor emits 'caroline st'+'john', never the unit surname "
            "'caroline st john' — the pair still merges, via an A1 "
            "subset on 'caroline st'/'john', not the intended name."
        ),
    ))

    # 15: two nicknames of the same person — no rule path.
    recs.append(_rec(
        "adv-nick-000", "adv_nick_to_nick", "merge",
        "Liz", "Beth",
        "Liz ran the Falcon review with Reyes and Malik.",
        "Beth ran the Falcon review with Reyes and Malik.",
        rationale=(
            "'Liz' and 'Beth' are both Elizabeth Marsh; no shared "
            "canonical exists and no rule covers nickname↔nickname."
        ),
    ))


# ---------------------------------------------------------------------------
# build + write
# ---------------------------------------------------------------------------

def generate() -> List[Dict[str, Any]]:
    """All fixture records in deterministic order."""
    recs: List[Dict[str, Any]] = []
    _sn_identical_surface(recs)
    _sn_disjoint_full(recs)
    _sn_subset_overmerge(recs)
    _sn_subset_abstain(recs)
    _sn_junior_senior(recs)
    _sn_object_person(recs)
    _sn_shared_surname(recs)
    _sn_initial_period(recs)
    _sn_near_spelling_strangers(recs)
    _sn_pronoun_only(recs)
    _sn_boundary_misc(recs)
    _mg_token_subset(recs)
    _mg_middle_name(recs)
    _mg_first_plus_initial(recs)
    _mg_initial_plus_last(recs)
    _mg_goes_by(recs)
    _mg_call_me(recs)
    _mg_my_name(recs)
    _mg_paren(recs)
    _mg_for_short(recs)
    _mg_caller(recs)
    _mg_near_spelling(recs)
    _mg_nickname_unsignaled(recs)
    _mg_surface_fold(recs)
    _mg_title_prefix(recs)
    _adv(recs)
    ids = [r["pair_id"] for r in recs]
    assert len(ids) == len(set(ids)), "duplicate pair_id"
    return recs


def write_fixture(path: Optional[str] = None) -> str:
    """Write the JSONL fixture; returns the path written."""
    path = path or os.path.join(FIXTURE_DIR, FIXTURE_NAME)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    recs = generate()
    with open(path, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True))
            fh.write("\n")
    return path


def main() -> None:
    path = write_fixture()
    recs = generate()
    from collections import Counter
    strata = Counter(r["stratum"] for r in recs)
    exp = Counter(r["expected"] for r in recs)
    print(f"wrote {len(recs)} records → {path}")
    print("expected:", dict(exp))
    for s in sorted(strata):
        print(f"  {s:32s} {strata[s]}")


if __name__ == "__main__":
    main()
