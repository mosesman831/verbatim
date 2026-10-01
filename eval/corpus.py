"""Owned, deterministic, synthetic evaluation corpus for Verbatim v2.

The corpus is generated locally from seeded pseudo-random choices over
hand-written template and vocabulary pools.  It contains no external
benchmark data — no MemConflict, LongMemEval, LoCoMo, or any other
third-party fixture — and this module intentionally has **no** ``verbatim``
imports so the corpus stays independent of the engine under test.

Generation contract
-------------------
* ``generate_corpus()`` returns a fully materialized :class:`Corpus`.
* The same ``seed`` always yields byte-identical output; ``Corpus.sha256()``
  is the reproducibility fingerprint recorded in reports.
* Every statement carries gold metadata: the statement id (which the
  harness maps to the ingested ``source_id``), its category, the salient
  tokens a query may use, and pairing links for updates/contradictions.
* Timeline is fixed: statement *i* occurs at ``T0 + i * STEP_US`` so
  ordering, validity intervals, and historical queries are reproducible
  and never wall-clock dependent.

Scale targets (SPEC_V2 §54 measurement posture): ~1000 memory statements
including ~300 update pairs and ~200 no-answer probes, spanning
preferences, possessions, relationships, schedules, locations, work,
health-adjacent facts, projects, opinions, negations, conditionals,
temporal facts, and explicit contradictions.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# fixed timeline
# ---------------------------------------------------------------------------

T0_US = 1_700_000_000_000_000          # 2023-11-14T22:13:20Z, arbitrary epoch
STEP_US = 60_000_000                    # one minute between statements
DAY_US = 86_400_000_000
UPDATE_LAG_US = 30 * DAY_US             # update successor arrives a month later

# ---------------------------------------------------------------------------
# vocabulary pools — original content, written for this corpus
# ---------------------------------------------------------------------------

FIRST_NAMES: Tuple[str, ...] = (
    "Anika", "Mateo", "Priya", "Tomas", "Yuki", "Nadia", "Omar", "Ingrid",
    "Kwame", "Saoirse", "Diego", "Maren", "Hiro", "Leila", "Piotr", "Freya",
    "Andre", "Mei", "Viktor", "Zola", "Ravi", "Elif", "Jonas", "Amara",
    "Kenji", "Solveig", "Dmitri", "Lucia", "Tariq", "Wren", "Bruno",
    "Selin", "Kofi", "Irena", "Marco", "Asha", "Nils", "Rosa", "Emeka",
    "Talia",
)

CITIES: Tuple[str, ...] = (
    "Lisbon", "Osaka", "Turin", "Gdańsk", "Adelaide", "Oaxaca", "Tampere",
    "Kigali", "Valparaíso", "Ljubljana", "Halifax", "Bergen", "Chiang Mai",
    "Montevideo", "Tbilisi", "Wellington", "Bruges", "Salvador", "Innsbruck",
    "Plovdiv", "Reykjavik", "Dakar", "Kaohsiung", "Mendoza", "Tallinn",
    "Galway", "Sapporo", "Marrakesh", "Utrecht", "Cusco", "Bergamo",
    "Stellenbosch", "Da Nang", "Québec City", "Rotterdam", "Zanzibar",
    "Puebla", "Aarhus", "Coimbra", "Tartu", "Valdivia", "Kotor", "Fukuoka",
    "Windhoek", "Punta Arenas", "Skopje", "Christchurch", "Lviv", "Manaus",
)

# held-out pools used ONLY for no-answer queries — disjoint by construction
HELD_OUT_NAMES: Tuple[str, ...] = (
    "Bartholomew", "Quintessa", "Zephyrine", "Ignatius", "Persephone",
    "Leopold", "Seraphina", "Maximilian", "Odette", "Casimir",
)
HELD_OUT_CITIES: Tuple[str, ...] = (
    "Ulverston", "Pucallpa", "Mariental", "Kirkenes", "Ouarzazate",
    "Tocopilla", "Yellowknife", "Svalbard", "Ittoqqortoormiit", "Norilsk",
)
HELD_OUT_TOOLS: Tuple[str, ...] = (
    "quillpad", "frostbyte", "lumenstack", "verdigris", "thornfield",
    "mosaicdb", "cinderlang", "opaline", "driftwell", "kestrelwm",
)

EDITORS: Tuple[str, ...] = (
    "vim", "neovim", "emacs", "vscode", "sublime text", "helix", "nano", "zed",
)
OSES: Tuple[str, ...] = (
    "arch linux", "debian", "fedora", "ubuntu", "nixos", "macos",
    "windows 11", "freebsd",
)
LANGUAGES: Tuple[str, ...] = (
    "python", "rust", "go", "typescript", "julia", "zig", "haskell",
    "elixir", "kotlin", "ocaml",
)
FRAMEWORKS: Tuple[str, ...] = (
    "django", "flask", "fastapi", "rails", "laravel", "axum", "phoenix",
    "nextjs", "sveltekit",
)
DATABASES: Tuple[str, ...] = (
    "postgresql", "sqlite", "mysql", "duckdb", "mongodb", "redis", "mariadb",
)
CI_TOOLS: Tuple[str, ...] = (
    "github actions", "gitlab ci", "buildkite", "jenkins", "circleci",
    "woodpecker ci",
)
NOTE_TOOLS: Tuple[str, ...] = (
    "obsidian", "logseq", "joplin", "org-mode", "notion", "foam",
)
DOC_TOOLS: Tuple[str, ...] = (
    "mkdocs", "sphinx", "docusaurus", "vitepress", "mdbook",
)
FOODS: Tuple[str, ...] = (
    "mushroom risotto", "okonomiyaki", "shakshuka", "pozole", "bibimbap",
    "moussaka", "arepas", "laksa", "pierogi", "tagine", "gado-gado",
    "feijoada", "katsudon", "ratatouille", "injera",
)
DRINKS: Tuple[str, ...] = (
    "oolong tea", "espresso", "kombucha", "sparkling water", "yerba mate",
    "horchata", "matcha", "chai", "kefir", "barley tea",
)
HOBBIES: Tuple[str, ...] = (
    "beekeeping", "letterpress printing", "birdwatching", "bouldering",
    "calligraphy", "astrophotography", "sourdough baking", "whittling",
    "orienteering", "linocut printing", "mushroom foraging", "kite making",
)
VEHICLES: Tuple[str, ...] = (
    "cargo bike", "hatchback", "scooter", "pickup truck", "minivan",
    "motorcycle", "e-bike", "station wagon",
)
INSTRUMENTS: Tuple[str, ...] = (
    "upright bass", "clarinet", "modular synthesizer", "banjo", "cello",
    "accordion", "drum kit", "fretless guitar",
)
SPORTS: Tuple[str, ...] = (
    "futsal", "bouldering", "ultimate frisbee", "rowing", "fencing",
    "trail running", "badminton", "kitesurfing",
)
COMPANIES: Tuple[str, ...] = (
    "Northwind Traders", "Acme Analytics", "Bluefin Systems", "Kelpware",
    "Ferrous Labs", "Mistral Data", "Cindercone", "Alder & Pine",
    "Quarryhouse", "Tidewater Instruments", "Vantablack Studios",
    "Solstice Grid",
)
ROLES: Tuple[str, ...] = (
    "platform engineer", "data engineer", "research scientist",
    "site reliability engineer", "technical writer", "product engineer",
    "security engineer", "developer advocate",
)
PROJECTS: Tuple[str, ...] = (
    "Project Sandpiper", "Project Juniper", "Project Tarn", "Project Larkspur",
    "Project Cobalt", "Project Fen", "Project Whetstone", "Project Alder",
    "Project Quasar", "Project Tinder", "Project Rill", "Project Marlow",
)
WEEKDAYS: Tuple[str, ...] = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday",
)
MEETING_KINDS: Tuple[str, ...] = (
    "standup", "planning", "retrospective", "design review", "book club",
    "board games night", "running group", "study group",
)
PETS: Tuple[str, ...] = (
    "tabby cat", "border collie", "bearded dragon", "senegal parrot",
    "rex rabbit", "betta fish", "maine coon", "cockatiel",
)
COLORS: Tuple[str, ...] = (
    "terracotta", "sage green", "ultramarine", "ochre", "slate grey",
    "burgundy", "teal", "mustard yellow",
)
GIBBERISH: Tuple[str, ...] = (
    "qwzxvbn", "plughx", "zarfle", "mordacq", "thwim", "glorp", "snizzle",
    "quavok", "brintle", "fazzum",
)

# ---------------------------------------------------------------------------
# corpus record types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Statement:
    """One ingestable evidence sentence plus gold metadata."""

    stmt_id: str
    text: str
    category: str
    event_us: int
    speaker_id: str = "me"
    #: distinctive tokens guaranteed to appear in the statement text; the
    #: harness derives recall queries from them.
    tokens: Tuple[str, ...] = ()
    #: 'fact' | 'update_old' | 'update_new' | 'contradiction_a' |
    #: 'contradiction_b' | 'conditional' | 'negation' | 'ambiguous'
    role: str = "fact"
    #: links update_old↔update_new and contradiction_a↔contradiction_b
    pair_id: Optional[str] = None
    #: mutable slot name for update pairs (e.g. "editor", "city")
    slot: Optional[str] = None


@dataclass(frozen=True)
class Query:
    """A recall probe with gold expected-evidence links."""

    query_id: str
    text: str
    #: statement ids whose evidence should be returned
    expect: Tuple[str, ...] = ()
    #: 'point' | 'current' | 'historical' | 'no_answer'
    kind: str = "point"
    #: explicit valid-time for temporal probes (None = engine default)
    valid_at_us: Optional[int] = None


@dataclass(frozen=True)
class UpdatePair:
    """old fact superseded by a newer fact about the same slot."""

    pair_id: str
    slot: str
    old_stmt_id: str
    new_stmt_id: str
    #: event time of the successor — the cut instant for temporal probes
    change_us: int


@dataclass(frozen=True)
class Corpus:
    """The full generated corpus."""

    seed: int
    statements: Tuple[Statement, ...]
    queries: Tuple[Query, ...]
    update_pairs: Tuple[UpdatePair, ...]
    contradiction_pairs: Tuple[Tuple[str, str], ...] = ()

    @property
    def no_answer_queries(self) -> Tuple[Query, ...]:
        return tuple(q for q in self.queries if q.kind == "no_answer")

    @property
    def expected_queries(self) -> Tuple[Query, ...]:
        return tuple(q for q in self.queries if q.kind != "no_answer")

    def by_id(self) -> dict[str, Statement]:
        return {s.stmt_id: s for s in self.statements}

    def sha256(self) -> str:
        """Reproducibility fingerprint over all generated content."""
        canon = {
            "seed": self.seed,
            "statements": [
                [s.stmt_id, s.text, s.category, s.event_us, s.role,
                 s.pair_id, s.slot, list(s.tokens)]
                for s in self.statements
            ],
            "queries": [
                [q.query_id, q.text, list(q.expect), q.kind, q.valid_at_us]
                for q in self.queries
            ],
            "update_pairs": [
                [u.pair_id, u.slot, u.old_stmt_id, u.new_stmt_id, u.change_us]
                for u in self.update_pairs
            ],
            "contradiction_pairs": [list(p) for p in self.contradiction_pairs],
        }
        blob = json.dumps(canon, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# template builders — each returns (text, tokens) for a chosen value set
# ---------------------------------------------------------------------------
#
# Templates deliberately mix extraction-friendly phrasings ("My editor is X",
# "I prefer X", "I live in X") with unstructured prose so the harness
# measures real coverage rather than matching the proposer's known grammar.


_TOKEN_SPLIT_RE = re.compile(r"[a-z0-9]+")


def _tok(*words: str) -> Tuple[str, ...]:
    """Salient lowercase alphanumeric tokens; hyphenated values split too."""
    out = []
    for w in words:
        out.extend(_TOKEN_SPLIT_RE.findall(w.lower()))
    return tuple(dict.fromkeys(out))


def _t_preference(rng: random.Random) -> Tuple[str, Tuple[str, ...]]:
    thing = rng.choice(EDITORS + LANGUAGES + FOODS + DRINKS + NOTE_TOOLS)
    alt = rng.choice(tuple(x for x in EDITORS + LANGUAGES + FOODS if x != thing))
    kind = rng.choice(("work", "daily use", "focus time", "weekends"))
    forms = (
        f"I prefer {thing} over {alt} for {kind}.",
        f"My favorite choice for {kind} is {thing}.",
        f"Given the option, I always pick {thing} for {kind}.",
    )
    return rng.choice(forms), _tok(thing)


def _t_possession(rng: random.Random) -> Tuple[str, Tuple[str, ...]]:
    v = rng.choice(VEHICLES + INSTRUMENTS + PETS)
    forms = (
        f"I own a {v} that I keep in good shape.",
        f"I have a {v} at home.",
        f"My {v} is one of my favorite possessions.",
    )
    return rng.choice(forms), _tok(v)


def _t_relationship(rng: random.Random) -> Tuple[str, Tuple[str, ...]]:
    name = rng.choice(FIRST_NAMES)
    role = rng.choice(("colleague", "sister", "neighbor", "climbing partner",
                       "mentor", "flatmate", "cousin"))
    act = rng.choice((
        "leads the design guild", "runs a small bakery",
        "teaches evening classes", "restores old radios",
        "coaches the junior team", "writes a weekly newsletter",
    ))
    return f"My {role} {name} {act}.", _tok(name, role.split()[-1])


def _t_schedule(rng: random.Random) -> Tuple[str, Tuple[str, ...]]:
    kind = rng.choice(MEETING_KINDS)
    day = rng.choice(WEEKDAYS)
    hour = rng.choice(("9:00", "10:30", "14:00", "16:15", "18:30"))
    forms = (
        f"My {kind} is every {day} at {hour}.",
        f"Every {day} at {hour} I have my {kind}.",
        f"On {day}s I go to {kind} at {hour}.",
    )
    return rng.choice(forms), _tok(kind.split()[-1], day)


def _t_location(rng: random.Random) -> Tuple[str, Tuple[str, ...]]:
    city = rng.choice(CITIES)
    forms = (
        f"I live in {city}.",
        f"My apartment is in {city}.",
        f"I moved to {city} a while back.",
        f"My office is located in {city}.",
    )
    return rng.choice(forms), _tok(city)


def _t_work(rng: random.Random) -> Tuple[str, Tuple[str, ...]]:
    company = rng.choice(COMPANIES)
    role = rng.choice(ROLES)
    forms = (
        f"I work at {company} as a {role}.",
        f"My day job is {role} at {company}.",
        f"I joined {company} as a {role}.",
    )
    return rng.choice(forms), _tok(company)


def _t_health(rng: random.Random) -> Tuple[str, Tuple[str, ...]]:
    forms = (
        (f"I walk {rng.randint(3, 9)} kilometers every morning.", ("walk",)),
        (f"I sleep about {rng.randint(6, 9)} hours on weeknights.", ("sleep",)),
        (f"I run {rng.randint(2, 5)} days a week before work.", ("run",)),
        (f"I stretch for {rng.randint(10, 30)} minutes after lunch.", ("stretch",)),
    )
    text, tk = rng.choice(forms)
    return text, tk


def _t_project(rng: random.Random) -> Tuple[str, Tuple[str, ...]]:
    proj = rng.choice(PROJECTS)
    desc = rng.choice((
        "batch job scheduler", "personal knowledge index",
        "home sensor dashboard", "markdown publishing tool",
        "distributed log tailer", "retro game database",
    ))
    forms = (
        f"I'm building {proj}, a {desc}.",
        f"My side project {proj} is a {desc}.",
        f"{proj} is my current project — a {desc}.",
    )
    return rng.choice(forms), _tok(proj.split()[-1])


def _t_opinion(rng: random.Random) -> Tuple[str, Tuple[str, ...]]:
    tool = rng.choice(FRAMEWORKS + DATABASES + CI_TOOLS + DOC_TOOLS)
    forms = (
        f"I think {tool} is overrated for small teams.",
        f"{tool} is the most dependable tool I've used this year.",
        f"In my experience, {tool} is great until it isn't.",
    )
    return rng.choice(forms), _tok(tool)


def _t_negation(rng: random.Random) -> Tuple[str, Tuple[str, ...]]:
    thing = rng.choice(EDITORS + FOODS + DRINKS + SPORTS)
    forms = (
        f"I do not use {thing} anymore.",
        f"I never liked {thing}.",
        f"I don't drink {thing}." if thing in DRINKS else f"I don't eat {thing}.",
    )
    return rng.choice(forms), _tok(thing)


def _t_conditional(rng: random.Random) -> Tuple[str, Tuple[str, ...]]:
    cond, act = rng.choice((
        ("it rains", f"I take the {rng.choice(VEHICLES)}"),
        ("I'm traveling", f"I use {rng.choice(NOTE_TOOLS)} for notes"),
        ("it's a weekday", f"I start work at {rng.choice(('8:00', '8:30', '9:00'))}"),
        ("I'm at the office", "I take the metro home"),
        ("it's sunny", f"I go {rng.choice(SPORTS)}"),
    ))
    forms = (
        f"When {cond}, {act}.",
        f"If {cond}, {act}.",
    )
    return rng.choice(forms), _tok(act)


def _t_temporal(rng: random.Random) -> Tuple[str, Tuple[str, ...]]:
    year = rng.randint(2011, 2024)
    what = rng.choice((
        f"I started learning {rng.choice(LANGUAGES)}",
        f"I moved to {rng.choice(CITIES)}",
        f"I adopted my {rng.choice(PETS)}",
        f"I joined {rng.choice(COMPANIES)}",
    ))
    return f"In {year}, {what}.", _tok(what)


def _t_ambiguous(rng: random.Random) -> Tuple[str, Tuple[str, ...]]:
    forms = (
        ("She told him it was finally ready.", ("ready",)),
        ("He said the thing with the parts got sorted out.", ("parts",)),
        ("They mentioned it would happen again next month.", ("again",)),
        ("She said it changed everything for the group.", ("changed",)),
    )
    return rng.choice(forms)


def _t_misc_unstructured(rng: random.Random) -> Tuple[str, Tuple[str, ...]]:
    forms = (
        ("The hallway lights flicker whenever the kettle runs.", ("flicker", "kettle")),
        ("A neighbor's rooster still wakes the whole block at dawn.", ("rooster",)),
        ("The library on the corner closes early on Wednesdays.", ("library",)),
        ("Someone keeps leaving chalk drawings on the pavement.", ("chalk",)),
        ("The old ferry only runs when the tide is high enough.", ("ferry",)),
        ("Our building's elevator makes a clicking sound between floors.", ("elevator",)),
    )
    return rng.choice(forms)


#: update slots — each entry: (slot name, value pool, template function)
_UPDATE_SLOTS: Sequence[Tuple[str, Tuple[str, ...], str]] = (
    ("editor", EDITORS, "My editor is {v}."),
    ("os", OSES, "My main operating system is {v}."),
    ("language", LANGUAGES, "The language I reach for most is {v}."),
    ("framework", FRAMEWORKS, "Our backend is built on {v}."),
    ("database", DATABASES, "The database for the service is {v}."),
    ("ci", CI_TOOLS, "Our CI runs on {v}."),
    ("notes", NOTE_TOOLS, "My notes live in {v}."),
    ("docs", DOC_TOOLS, "We publish docs with {v}."),
    ("city", CITIES, "I live in {v}."),
    ("company", COMPANIES, "I work at {v}."),
)

_CATEGORY_BUILDERS = {
    "preference": _t_preference,
    "possession": _t_possession,
    "relationship": _t_relationship,
    "schedule": _t_schedule,
    "location": _t_location,
    "work": _t_work,
    "health_adjacent": _t_health,
    "project": _t_project,
    "opinion": _t_opinion,
    "negation": _t_negation,
    "conditional": _t_conditional,
    "temporal_fact": _t_temporal,
    "ambiguous_reference": _t_ambiguous,
    "unstructured": _t_misc_unstructured,
}

# weights over non-update/non-contradiction categories
_CATEGORY_WEIGHTS = (
    ("preference", 12),
    ("possession", 8),
    ("relationship", 10),
    ("schedule", 10),
    ("location", 9),
    ("work", 9),
    ("health_adjacent", 6),
    ("project", 8),
    ("opinion", 8),
    ("negation", 8),
    ("conditional", 8),
    ("temporal_fact", 6),
    ("ambiguous_reference", 4),
    ("unstructured", 8),
)


def _pick_category(rng: random.Random) -> str:
    total = sum(w for _, w in _CATEGORY_WEIGHTS)
    r = rng.randrange(total)
    acc = 0
    for name, w in _CATEGORY_WEIGHTS:
        acc += w
        if r < acc:
            return name
    return _CATEGORY_WEIGHTS[-1][0]


def _pick_update_slot(rng: random.Random) -> Tuple[str, Tuple[str, ...], str]:
    return _UPDATE_SLOTS[rng.randrange(len(_UPDATE_SLOTS))]


# ---------------------------------------------------------------------------
# no-answer query builders
# ---------------------------------------------------------------------------


def _no_answer_queries(rng: random.Random, count: int, start_idx: int) -> list[Query]:
    out: list[Query] = []
    templates = (
        "what does {} prefer",
        "where does {} live",
        "when is the {} meeting",
        "how often do I use {}",
        "what is my {} configuration",
        "did I ever mention {}",
        "what happened with {}",
    )
    probes = HELD_OUT_NAMES + HELD_OUT_CITIES + HELD_OUT_TOOLS + GIBBERISH
    for i in range(count):
        subject = probes[i % len(probes)]
        tmpl = templates[i % len(templates)]
        text = tmpl.format(subject)
        out.append(Query(
            query_id=f"na{start_idx + i:04d}",
            text=text,
            expect=(),
            kind="no_answer",
        ))
    return out


# ---------------------------------------------------------------------------
# point / temporal query builders
# ---------------------------------------------------------------------------


def _point_query(rng: random.Random, idx: int, stmt: Statement) -> Query:
    """Build a query from a statement's salient tokens.

    Uses 1-2 tokens so FTS5 BM25 has real lexical signal; the gold link is
    the statement id (mapped to the ingested source_id by the harness), not
    a fuzzy text match.
    """
    toks = [t for t in stmt.tokens if len(t) > 2]
    if not toks:
        toks = list(stmt.tokens) or ["memory"]
    q_tokens = toks[:2] if len(toks) > 1 else toks
    lead = rng.choice(("what about", "do I remember", "tell me about", ""))
    text = " ".join([lead, *q_tokens]).strip()
    return Query(
        query_id=f"q{idx:04d}",
        text=text,
        expect=(stmt.stmt_id,),
        kind="point",
    )


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def generate_corpus(size: int = 1000, seed: int = 42) -> Corpus:
    """Generate the deterministic corpus.

    ``size`` is the target statement count; the actual count may differ
    slightly because update pairs and contradiction pairs contribute two
    statements each.
    """
    rng = random.Random(seed)

    n_update_pairs = min(300, max(0, size // 3))
    n_contra_pairs = max(4, size // 50)
    n_regular = size - 2 * n_update_pairs - 2 * n_contra_pairs
    if n_regular < 0:
        n_regular = 0
        n_update_pairs = max(0, (size - 2 * n_contra_pairs) // 2)

    statements: list[Statement] = []
    update_pairs: list[UpdatePair] = []
    contra_pairs: list[Tuple[str, str]] = []

    idx = 0

    def _next_id() -> str:
        nonlocal idx
        idx += 1
        return f"s{idx:04d}"

    def _event_us() -> int:
        return T0_US + idx * STEP_US

    # --- update pairs: old fact first, new fact later in the stream -------
    # Interleave: emit old statements into the first ~60% of the stream and
    # new statements into the last ~40%, so successors genuinely arrive later.
    pair_plan: list[Tuple[str, str, str, str, str]] = []  # pid, slot, old, new, tmpl
    for i in range(n_update_pairs):
        slot, pool, tmpl = _pick_update_slot(rng)
        old_v, new_v = rng.sample(pool, 2)
        pid = f"u{i:04d}"
        pair_plan.append((pid, slot, old_v, new_v, tmpl))

    # --- contradiction pairs: same slot, different value, NO successor ----
    contra_plan: list[Tuple[str, str, str, str, str]] = []
    for i in range(n_contra_pairs):
        slot, pool, tmpl = _pick_update_slot(rng)
        a_v, b_v = rng.sample(pool, 2)
        pid = f"c{i:04d}"
        contra_plan.append((pid, slot, a_v, b_v, tmpl))

    # --- regular statements ------------------------------------------------
    regular: list[Tuple[str, str, Tuple[str, ...], str]] = []  # cat, text, toks, role
    for _ in range(n_regular):
        cat = _pick_category(rng)
        text, toks = _CATEGORY_BUILDERS[cat](rng)
        role = {
            "negation": "negation",
            "conditional": "conditional",
            "ambiguous_reference": "ambiguous",
        }.get(cat, "fact")
        regular.append((cat, text, toks, role))

    # Assemble in a deterministic interleaved order:
    #   [regular/old/contra_a shuffled] ... [new/contra_b appended later]
    first_half: list[Tuple[str, str, Tuple[str, ...], str, Optional[str], Optional[str]]] = []
    second_half: list[Tuple[str, str, Tuple[str, ...], str, Optional[str], Optional[str]]] = []

    for cat, text, toks, role in regular:
        first_half.append((cat, text, toks, role, None, None))

    for pid, slot, old_v, new_v, tmpl in pair_plan:
        first_half.append((f"update:{slot}", tmpl.format(v=old_v), _tok(old_v),
                           "update_old", pid, slot))
        second_half.append((f"update:{slot}", tmpl.format(v=new_v), _tok(new_v),
                            "update_new", pid, slot))

    for pid, slot, a_v, b_v, tmpl in contra_plan:
        first_half.append((f"contradiction:{slot}", tmpl.format(v=a_v), _tok(a_v),
                           "contradiction_a", pid, slot))
        second_half.append((f"contradiction:{slot}", tmpl.format(v=b_v), _tok(b_v),
                            "contradiction_b", pid, slot))

    rng.shuffle(first_half)

    for cat, text, toks, role, pid, slot in first_half:
        sid = _next_id()
        statements.append(Statement(
            stmt_id=sid, text=text, category=cat, event_us=_event_us(),
            tokens=toks, role=role, pair_id=pid, slot=slot,
        ))

    first_half_ids = {s.stmt_id for s in statements}
    for cat, text, toks, role, pid, slot in second_half:
        sid = _next_id()
        statements.append(Statement(
            stmt_id=sid, text=text, category=cat, event_us=_event_us(),
            tokens=toks, role=role, pair_id=pid, slot=slot,
        ))

    # update pair records: change instant = the successor's event time
    by_pair = {}
    for s in statements:
        if s.pair_id:
            by_pair.setdefault(s.pair_id, {})[s.role] = s
    for pid, slot, _old_v, _new_v, _tmpl in pair_plan:
        pair = by_pair[pid]
        update_pairs.append(UpdatePair(
            pair_id=pid, slot=slot,
            old_stmt_id=pair["update_old"].stmt_id,
            new_stmt_id=pair["update_new"].stmt_id,
            change_us=pair["update_new"].event_us,
        ))
    for pid, slot, _a, _b, _tmpl in contra_plan:
        pair = by_pair[pid]
        contra_pairs.append(
            (pair["contradiction_a"].stmt_id, pair["contradiction_b"].stmt_id)
        )

    # --- queries ------------------------------------------------------------
    queries: list[Query] = []
    qi = 0
    by_id = {s.stmt_id: s for s in statements}

    # point queries over a deterministic sample of non-update statements
    sampleable = [s for s in statements if s.role in
                  ("fact", "negation", "conditional", "ambiguous")]
    rng.shuffle(sampleable)
    n_point = min(250, len(sampleable))
    for s in sampleable[:n_point]:
        queries.append(_point_query(rng, qi, s))
        qi += 1

    # temporal probes for a sample of update pairs
    for u in update_pairs[:200]:
        new_s, old_s = by_id[u.new_stmt_id], by_id[u.old_stmt_id]
        queries.append(Query(
            query_id=f"q{qi:04d}",
            text=" ".join(new_s.tokens[:2]) or new_s.text,
            expect=(u.new_stmt_id,),
            kind="current",
            valid_at_us=u.change_us + DAY_US,
        ))
        qi += 1
        queries.append(Query(
            query_id=f"q{qi:04d}",
            text=" ".join(old_s.tokens[:2]) or old_s.text,
            expect=(u.old_stmt_id,),
            kind="historical",
            valid_at_us=u.change_us - DAY_US,
        ))
        qi += 1

    # no-answer probes (~200)
    queries.extend(_no_answer_queries(rng, 200, qi))

    corpus = Corpus(
        seed=seed,
        statements=tuple(statements),
        queries=tuple(queries),
        update_pairs=tuple(update_pairs),
        contradiction_pairs=tuple(contra_pairs),
    )
    return corpus


def generate_realistic_chat_corpus() -> Corpus:
    texts = (
        (
            "chat0001", "My name is Bob", "identity", ("name", "Bob"),
            "fact", None, None,
        ),
        (
            "chat0002", "hey so I just moved to Berlin last month",
            "location", ("Berlin",), "fact", None, None,
        ),
        (
            "chat0003", "oh and btw I switched from VS Code to Neovim",
            "editor", ("Neovim",), "update_old", "chat-editor", "editor",
        ),
        (
            "chat0004",
            "Actually scratch that about Neovim, I went back to VS Code",
            "editor", ("VS", "Code"), "update_new", "chat-editor", "editor",
        ),
        (
            "chat0005", "I don't use Docker anymore", "negation",
            ("Docker",), "negation", None, None,
        ),
        (
            "chat0006", "my manager Sarah wants the Q3 report by Friday",
            "work", ("Sarah", "report"), "fact", None, None,
        ),
        (
            "chat0007", "I'm allergic to shellfish", "health",
            ("allergic", "shellfish"), "fact", None, None,
        ),
        (
            "chat0008", "My daughter Emma turns 7 next week", "relationship",
            ("Emma",), "fact", None, None,
        ),
        (
            "chat0009", "the API key rotates every 90 days", "operations",
            ("API", "key"), "fact", None, None,
        ),
        (
            "chat0010", "私は東京に住んでいます", "location", ("東京",),
            "fact", None, None,
        ),
    )
    statements = tuple(
        Statement(
            stmt_id=sid,
            text=text,
            category=category,
            event_us=T0_US + i * STEP_US,
            tokens=tuple(token.casefold() for token in tokens),
            role=role,
            pair_id=pair_id,
            slot=slot,
        )
        for i, (sid, text, category, tokens, role, pair_id, slot)
        in enumerate(texts, start=1)
    )
    change_us = statements[3].event_us
    queries = (
        Query("chatq001", "what is my name", ("chat0001",), "point"),
        Query("chatq002", "where do I live?", ("chat0002",), "point"),
        Query("chatq003", "editor?", ("chat0004",), "current", change_us + DAY_US),
        Query("chatq004", "editor?", ("chat0003",), "historical", change_us - DAY_US),
        Query("chatq005", "Docker", ("chat0005",), "point"),
        Query("chatq006", "Sarah report", ("chat0006",), "point"),
        Query("chatq007", "what am I allergic to", ("chat0007",), "point"),
        Query("chatq008", "Emma", ("chat0008",), "point"),
        Query("chatq009", "API key", ("chat0009",), "point"),
        Query("chatq010", "東京", ("chat0010",), "point"),
        Query("chatq011", "Bartholomew editor", (), "no_answer"),
        Query("chatq012", "zeppelin configuration", (), "no_answer"),
    )
    return Corpus(
        seed=0,
        statements=statements,
        queries=queries,
        update_pairs=(
            UpdatePair(
                pair_id="chat-editor",
                slot="editor",
                old_stmt_id="chat0003",
                new_stmt_id="chat0004",
                change_us=change_us,
            ),
        ),
    )


def corpus_stats(corpus: Corpus) -> dict[str, object]:
    """Category/role/query counts for the report."""
    cats: dict[str, int] = {}
    roles: dict[str, int] = {}
    for s in corpus.statements:
        cats[s.category] = cats.get(s.category, 0) + 1
        roles[s.role] = roles.get(s.role, 0) + 1
    qk: dict[str, int] = {}
    for q in corpus.queries:
        qk[q.kind] = qk.get(q.kind, 0) + 1
    return {
        "seed": corpus.seed,
        "sha256": corpus.sha256(),
        "statements": len(corpus.statements),
        "categories": cats,
        "roles": roles,
        "queries": qk,
        "update_pairs": len(corpus.update_pairs),
        "contradiction_pairs": len(corpus.contradiction_pairs),
    }
