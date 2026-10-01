"""Fixture-driven tests for ``intent/v2`` (SPEC_V7 §32.14, V7-05.12/13).

The labeled fixture is *generated* deterministically at module load (seeded
``random.Random`` over template x slot-product families) — it is code, not a
committed blob of benchmark text. ``NormAnalysis`` objects are constructed
directly with a small local tokenizer; ``norm/v2`` itself is owned by a
concurrent worker and is not imported here.

Coverage asserted:
- >= 600 labeled queries spanning all 15 ``IntentClass`` values
- macro-accuracy (mean per-class primary recall) >= 0.90
- per-class primary correctness floor
- ``rule_trace`` populated, ``classes[0] == primary``, classes unique
- ``decompose`` returns <= 3 facets for multi_hop/comparison primaries and
  the ``(norm,)`` singleton otherwise
- determinism: identical inputs -> identical outputs
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import pytest

from verbatim.core.types_v7 import (
    IntentClass,
    IntentResult,
    NormAnalysis,
    NormTerm,
)
from verbatim.querying.intent_v2 import (
    INTENT_ID,
    MAX_FACETS,
    classify,
    decompose,
)


# ---------------------------------------------------------------------------
# Local fixture tokenizer (NOT norm/v2 — byte offsets into the folded string)
# ---------------------------------------------------------------------------

_TOK_RX = re.compile(r"[a-z0-9]+(?:[._/:\-][a-z0-9]+)*")


def _looks_like_identifier(tok: str) -> bool:
    if "://" in tok:
        return True
    if not any(c.isdigit() for c in tok):
        return False
    if not any(c.isalpha() for c in tok):
        return False  # pure numbers are ordinary terms
    if any(c in "-./:" for c in tok):
        return True
    return len(tok) >= 6 and re.fullmatch(r"[a-z0-9]+", tok) is not None


def mk(text: str) -> NormAnalysis:
    """Tokenize like the contract describes: folded text, identifiers on
    their own channel, clitics split ('s -> s)."""
    terms: list[NormTerm] = []
    ids: list[NormTerm] = []
    for wm in re.finditer(r"\S+", text.lower()):
        raw = wm.group(0)
        base = wm.start()
        if _looks_like_identifier(raw):
            nt = NormTerm(raw, "identifier", base, base + len(raw))
            terms.append(nt)
            ids.append(nt)
            continue
        folded = raw.replace("'", " ")
        for m in _TOK_RX.finditer(folded):
            terms.append(
                NormTerm(m.group(0), "text", base + m.start(), base + m.end())
            )
    return NormAnalysis(
        analyzer_id="norm/v2",
        terms=tuple(terms),
        identifiers=tuple(ids),
        text=text.lower(),
    )


def run(text: str, canons: tuple[str, ...] = ()) -> IntentResult:
    norm = mk(text)
    return classify(norm, canons, norm.identifiers)


# ---------------------------------------------------------------------------
# Slot pools (generic English; no benchmark-derived content)
# ---------------------------------------------------------------------------

_NAMES = [
    "alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi",
    "ivan", "judy", "mallory", "nia", "olivia", "peggy", "quentin",
    "rupert", "sybil", "trent", "uma", "victor", "wendy", "xander",
    "yvonne", "zach", "marco", "lena", "priya", "kenji", "sofia", "ravi",
]

_PLACES = [
    "paris", "lisbon", "denver", "tokyo", "oslo", "austin", "berlin",
    "madrid", "chicago", "seattle", "nairobi", "quito", "helsinki",
    "lyon", "porto", "tucson", "boulder", "galway", "bruges", "osaka",
]

# content noun phrases — deliberately free of rule-trigger words
_THINGS = [
    "project falcon", "garden project", "budget spreadsheet",
    "onboarding doc", "photography course", "tax return",
    "warranty document", "lease agreement", "reading list",
    "packing list", "insurance claim", "bank statement",
    "recipe folder", "slide deck", "status report", "expense report",
    "grocery list", "car insurance", "phone bill", "gym membership",
    "library card", "pitch deck", "roadmap", "design doc",
    "inventory list", "contact list", "checklist", "mailing list",
]

# event phrases — every entry contains >= 1 term from the event lexicon
_EVENTS = [
    "move to denver", "relocation", "graduation", "wedding",
    "product launch", "conference", "surgery", "marathon",
    "job interview", "team offsite", "kitchen renovation",
    "flight to oslo", "birthday party", "workshop", "book club meeting",
    "sailing trip", "deployment", "company retreat", "deadline", "exam",
    "anniversary dinner", "tournament", "demo day", "apartment purchase",
    "onboarding",
]

# verbs — all in the classifier's event lexicon (infinitive forms)
_VERBS = [
    "move", "start", "join", "quit", "leave", "resign", "finish",
    "marry", "buy", "sell", "visit", "travel", "fly", "return", "adopt",
    "write", "publish", "record", "run", "compete", "attend", "meet",
    "host", "celebrate", "cook", "learn", "volunteer", "donate", "hire",
    "promote", "interview", "launch", "ship", "release", "book",
    "reserve", "cancel", "call", "email", "watch", "plant", "renovate",
    "build", "fix", "arrive", "depart", "land", "open", "sign", "pay",
    "deploy", "submit", "approve", "merge", "present", "schedule",
    "announce", "accept", "reject", "confirm", "postpone", "delay",
    "apply", "register", "vote", "organize", "prepare", "pack", "repair",
]

_PPS = [  # past participles / past forms, all in the event lexicon
    "moved", "started", "joined", "left", "resigned", "graduated",
    "finished", "married", "bought", "sold", "visited", "traveled",
    "returned", "adopted", "wrote", "published", "recorded", "won",
    "lost", "attended", "met", "hosted", "celebrated", "learned",
    "volunteered", "donated", "hired", "promoted", "fired",
    "interviewed", "launched", "shipped", "released", "booked",
    "cancelled", "called", "emailed", "watched", "planted", "renovated",
    "built", "fixed", "arrived", "departed", "landed", "opened",
    "closed", "signed", "paid", "tested", "deployed", "submitted",
    "approved", "merged", "presented", "gave", "received", "sent",
    "discussed", "planned", "decided", "changed", "switched",
    "updated", "rented", "borrowed", "found", "missed", "passed",
    "failed", "studied", "scheduled", "announced", "proposed",
    "confirmed", "postponed", "delayed", "applied", "registered",
    "voted", "organized", "prepared", "packed", "repaired", "replaced",
]

_GERUNDS = [
    "moving", "starting", "leaving", "graduating", "finishing",
    "buying", "selling", "visiting", "traveling", "flying", "returning",
    "writing", "recording", "running", "competing", "attending",
    "hosting", "celebrating", "cooking", "learning", "volunteering",
    "donating", "recovering", "interviewing", "launching", "shipping",
    "booking", "calling", "watching", "listening", "planting",
    "renovating", "building", "fixing", "departing", "opening",
    "closing", "signing", "paying", "eating", "sleeping", "testing",
    "deploying", "reviewing", "presenting", "sending", "talking",
    "discussing", "planning", "changing", "updating", "renting",
    "studying", "teaching", "playing", "scheduling", "preparing",
    "packing", "cleaning", "repairing", "working", "living", "staying",
    "dating", "training", "reading",
]

_NOUNS = [
    "meetings", "emails", "photos", "tickets", "commits",
    "pull requests", "invoices", "receipts", "workouts", "recipes",
    "books", "songs", "episodes", "messages", "calls", "packages",
    "plants", "candles", "stamps", "documents", "files", "slides",
    "tasks", "bugs", "features", "sprints", "leads", "reports",
]

_ATTRS = [
    "employer", "job title", "address", "phone number", "manager",
    "team", "car", "plan", "subscription", "project", "commute",
    "office", "salary", "timezone", "editor", "programming language",
    "workout routine", "diet", "schedule", "budget", "goal", "mentor",
    "landlord", "roommate", "primary doctor", "insurance provider",
]

_ROLES = [
    "chef", "manager", "lawyer", "landlord", "neighbor", "cousin",
    "trainer", "broker", "agent", "tenant", "dentist", "plumber",
    "accountant", "roommate", "barista", "mechanic", "tutor", "nanny",
    "coworker", "intern",
]

_MEALS = ["breakfast", "lunch", "dinner", "snacks", "brunch"]

_CMP_PAIRS = [
    ("python", "ruby"), ("ipad", "kindle"), ("honda", "toyota"),
    ("yoga", "running"), ("the train", "flying"), ("the apartment", "the house"),
    ("mac", "pc"), ("coffee", "tea"), ("paris", "rome"),
    ("spotify", "apple music"), ("whatsapp", "signal"), ("vi", "emacs"),
    ("subaru", "mazda"), ("nike", "adidas"), ("sushi", "sashimi"),
    ("the ebook", "the paperback"), ("docker", "podman"),
    ("mysql", "postgres"), ("the sedan", "the hatchback"),
    ("swift", "kotlin"), ("the hostel", "the hotel"),
]

_MONTHS = [
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]

_SEASONS = ["spring", "summer", "fall", "autumn", "winter"]

_PERIODS = ["week", "month", "year", "weekend", "quarter"]

_PERIOD_PLURALS = ["weeks", "months", "years", "days"]

_WEEKDAYS = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday",
]

_ABSDAYS = ["yesterday", "today", "tomorrow"]

_YEARS = ["2015", "2016", "2017", "2018", "2019", "2020", "2021",
          "2022", "2023", "2024", "2025"]

_NUMS = ["two", "three", "four", "five", "3", "5", "7", "10"]

_IDFMTS = [
    "GH-{n}", "INC-{n}", "ORD-{n}", "v{a}.{b}.{c}",
    "https://ex.com/{w}", "{x}f{n}ab{n}f", "ticket-{n}", "case-{n}",
    "bug-{n}", "PR-{n}", "rev-{x}c{n}", "doc-{n}",
]


# ---------------------------------------------------------------------------
# Template families — every template is engineered so its label is the class
# whose rule must fire first; collisions are exercised in the edge tests.
# ---------------------------------------------------------------------------

_ID = IntentClass

FAMILIES: dict[IntentClass, list[str]] = {
    _ID.IDENTIFIER: [
        "what is {id}", "show me {id}", "find {id}",
        "where does {id} appear", "details for {id}",
        "any notes on {id}", "what does {id} refer to", "look up {id}",
        "status of {id}", "who filed {id}", "when was {id} created",
        "what does {id} mean", "open {id}", "pull up {id}",
        "remind me what {id} is", "how do i use {id}",
    ],
    _ID.TEMPORAL_POINT: [
        "when did {e} {v}", "when did we {v} the {t}",
        "when was the {ev}", "when is the {ev}", "when will {e} {v}",
        "when does the {ev} start", "when should i {v} the {t}",
        "what date did we {v}", "what time does the {ev} start",
        "which day was the {ev}", "which day did {e} {v}",
        "how long ago did we {v}", "how long ago was the {ev}",
        "how many days ago did {e} {v}", "how many weeks ago was the {ev}",
        "what year did {e} {v}", "on what date did the {ev} happen",
        "what month was the {ev}", "when did {e} and {e2} {v}",
        "remind me when the {ev} starts", "tell me when the {ev} ends",
    ],
    _ID.TEMPORAL_ORDER: [
        "did the {ev} happen before or after the {ev2}",
        "which came first, the {ev} or the {ev2}",
        "was the {ev} earlier than the {ev2}",
        "did i {v} before i {v2}",
        "what happened first, the {ev} or the {ev2}",
        "the {ev} before or after the {ev2}",
        "did we {v} after we {v2}",
        "did the {ev} come after the {ev2}",
        "which happened later, the {ev} or the {ev2}",
        "was the {ev} later than the {ev2}",
        "did the {ev} happen after the {ev2}",
        "the {ev} first or the {ev2}",
    ],
    _ID.DURATION: [
        "how long did the {ev} last", "how long was the {ev}",
        "how long have i had the {t}",
        "how many days between the {ev} and the {ev2}",
        "how many weeks between the {ev} and the {ev2}",
        "how many months between {ev} and {ev2}",
        "how many years between {ev} and {ev2}",
        "how long has it been since the {ev}", "how long since i {v}",
        "for how long did the {ev} run", "how long did we {v}",
        "it's been a while since the {ev}",
        "what's the duration of the {ev}", "how long was i {g}",
        "i have had the {t} since {y}",
    ],
    _ID.COUNT_AGGREGATE: [
        "how many {n} did i {v}", "how many times did {e} {v}",
        "how many {n} have we {pp}", "how much did the {t} cost",
        "how much time did the {ev} take", "count the {n}",
        "what's the total number of {n}", "the number of {n} on the list",
        "total {n} this {pd}", "how many {n} are on the {t} list",
        "how many times have i {pp} the {t}",
        "how much do i owe on the {t}",
        "how many {n} did we get last {pd}",
        "how many {n} does {e} have",
    ],
    _ID.CURRENT_VALUE: [
        "what is my current {a}", "where do i live now",
        "what's the latest on the {t}", "do i still have the {t}",
        "what is the current {a}", "what am i {g} these days",
        "what's my {a} now", "right now what is the {a}",
        "at the moment what is my {a}", "what's the {a} nowadays",
        "is {e} still {g}", "the latest {a}",
        "what's the status of the {t} now", "what do i still {v}",
        "what's my {a} at present", "what have we done so far on the {t}",
        "who is my {a} these days",
    ],
    _ID.HISTORY_OF: [
        "how has my {a} changed", "the history of my {a}",
        "what did i used to {v}", "how did the {t} change over time",
        "what was my {a} previously", "how has the {t} evolved",
        "timeline of the {t}", "what's the history of the {t}",
        "my previous {a}", "how did my {a} change",
        "in the past what was my {a}", "what was the {a} progression",
        "the evolution of my {a}", "what was my original {a}",
    ],
    _ID.PREFERENCE: [
        "what is my favorite {n}", "do i prefer {x} or {y}",
        "which {n} do i like better", "would i enjoy the {t}",
        "would i like the {t}", "what do i prefer for {meal}",
        "recommend a {n} for me", "what's my preferred {n}",
        "my favorite {n}", "do i prefer {x} to {y}",
        "suggest a {n} for me", "what are my {n} preferences",
        "which do i prefer, {x} or {y}", "would we enjoy the {t}",
        "recommend {x} or {y} for me", "what's my favorite {n}",
    ],
    _ID.COMPARISON: [
        "compare {x} and {y}",
        "what's the difference between {x} and {y}",
        "{x} versus {y}", "{x} vs {y}", "which is better, {x} or {y}",
        "is {x} cheaper than {y}", "which of {x} and {y} should i pick",
        "do both {x} and {y} work", "how do {x} and {y} differ",
        "differences between {x} and {y}", "is {x} faster than {y}",
        "{x} or {y}, which is better", "compare the {x} to the {y}",
        "which is worse, {x} or {y}", "how does {x} compare to {y}",
        "which is quieter, {x} or {y}", "is {x} bigger than {y}",
    ],
    _ID.WHY_CAUSAL: [
        "why did i {v}", "why did we {v} the {t}", "why does {e} {v}",
        "what's the reason for the {ev}", "what led to the {ev}",
        "how come the {ev} happened", "what caused the {ev}",
        "what made me {v}", "the reason i {pp}",
        "did we cancel because of the {ev}", "why is {e} {g}",
        "the reason we {pp}", "what's the reason behind the {ev}",
        "why do i {v}",
    ],
    _ID.OPEN_DOMAIN: [
        "would i be happy in {p}", "might i regret {g}",
        "would we need a bigger {t}", "is it likely that i {v}",
        "will i probably {v} the {t}", "would my {a} suit {p}",
        "might we {v} the {t}", "could i {v} a {t}",
        "should i {v} the {t}", "would a {t} work for me",
        "is it likely we {v}", "would i survive a {t}",
        "might i {v} the {t}", "could we {v} the {ev}",
        "would it be hard for me to {v}",
        "is it likely that we {v}",
    ],
    _ID.ABSTAIN_LIKELY: [
        "have i ever {pp} the {t}", "have i ever been to {p}",
        "did i ever mention the {t}", "have i told you about the {t}",
        "do i have any {n}", "is there any {n} on the {t}",
        "are there any {n}", "have i talked about the {t}",
        "did i say anything about the {t}", "has there been any {n}",
        "do we have any {n}", "have we ever {pp} {p}",
        "did i tell you about the {t}", "have i been {g}",
        "did we ever cover the {t}", "have i already {pp} the {t}",
        "did i lock the {t}", "did i send the {t}",
        "do you know if i {pp} the {t}",
    ],
    _ID.MULTI_HOP: [
        "what did {e} say to {e2} about the {t}",
        "who introduced {e} to {e2}", "the {t} that {e} recommended",
        "what is the connection between {e} and {e2}",
        "which {role} did {e} recommend to {e2}",
        "what does {e} think of {e2}",
        "the {t} that the {role} who {pp} {e} wrote",
        "where does {e} work with {e2}",
        "did {e} meet {e2} at the {ev}",
        "the {t} {e} gave to {e2}",
        "what did {e} bring {e2} from {p}",
        "who does {e} report to besides {e2}",
        "the {n} that {e} mentioned to {e2}",
        "what do {e} and {e2} have in common",
        "how are {e} and {e2} related",
        "did {e} and {e2} attend the {ev}",
        "the {t} {e} showed {e2}",
    ],
    _ID.TEMPORAL_RANGE: [
        "what did we do last {pd}", "what happened between {m} and {m2}",
        "what did i work on from {m} to {m2}",
        "what happened during the {s}", "what did we discuss in {m}",
        "what happened this {pd}", "what did i do over the past few {pds}",
        "what happened over the {s}", "what did we ship from {y} to {y2}",
        "what did we do {abs}", "what's planned for next {pd}",
        "what happened {num} days ago", "what did we cover on {wd}",
        "what happened the other day", "what did we do in {y}",
        "what came up in the {s}", "what did we decide during {m}",
        "what happened over {m}", "what did we do this past {pd}",
        "what did we discuss on {wd}", "what's happening {abs}",
        "what did i work on in {m} and {m2}",
    ],
    _ID.LOOKUP: [
        "what is the {t}", "where is the {t}", "tell me about the {t}",
        "notes on the {t}", "find the {t} details",
        "what's the {t} called", "show the {t} information",
        "the {t} overview", "what does the {t} include",
        "describe the {t}", "details about the {t}", "who is {e}",
        "where is {e} seated", "what does {e} do", "is the {t} ready",
        "the {t} documentation", "what does the {t} say",
        "more on the {t}",
    ],
}


# ---------------------------------------------------------------------------
# Deterministic fixture generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixtureCase:
    text: str
    expected: IntentClass
    canons: tuple[str, ...]


def _id_value(rng: random.Random, fmt: str) -> str:
    return fmt.format(
        n=rng.randint(100, 99999),
        a=rng.randint(0, 9),
        b=rng.randint(0, 20),
        c=rng.randint(0, 9),
        w=rng.choice(["alpha", "beta", "gamma", "delta", "omega"]),
        x=rng.choice("abcdef"),
    )


def _distinct(rng: random.Random, pool: list, exclude: tuple = ()) -> str:
    for _ in range(64):
        v = rng.choice(pool)
        if v not in exclude:
            return v
    return next(v for v in pool if v not in exclude)


def build_fixture(per_class: int = 45, seed: int = 20260918) -> list[FixtureCase]:
    rng = random.Random(seed)
    cases: list[FixtureCase] = []
    for klass, templates in FAMILIES.items():
        for i in range(per_class):
            tmpl = templates[i % len(templates)]
            e = _distinct(rng, _NAMES)
            e2 = _distinct(rng, _NAMES, (e,))
            x, y = rng.choice(_CMP_PAIRS)
            ev = rng.choice(_EVENTS)
            ev2 = _distinct(rng, _EVENTS, (ev,))
            v = rng.choice(_VERBS)
            v2 = _distinct(rng, _VERBS, (v,))
            m = rng.choice(_MONTHS)
            m2 = _distinct(rng, _MONTHS, (m,))
            yr = rng.choice(_YEARS)
            yr2 = _distinct(rng, _YEARS, (yr,))
            fills = {
                "e": e, "e2": e2, "p": rng.choice(_PLACES),
                "t": rng.choice(_THINGS), "ev": ev, "ev2": ev2,
                "v": v, "v2": v2, "pp": rng.choice(_PPS),
                "g": rng.choice(_GERUNDS), "n": rng.choice(_NOUNS),
                "a": rng.choice(_ATTRS), "role": rng.choice(_ROLES),
                "meal": rng.choice(_MEALS), "x": x, "y": y,
                "m": m, "m2": m2, "s": rng.choice(_SEASONS),
                "pd": rng.choice(_PERIODS),
                "pds": rng.choice(_PERIOD_PLURALS),
                "wd": rng.choice(_WEEKDAYS),
                "abs": rng.choice(_ABSDAYS), "y": yr, "y2": yr2,
                "num": rng.choice(_NUMS),
                "id": _id_value(rng, rng.choice(_IDFMTS)),
            }
            text = tmpl.format(**fills)
            # entity canons are the named-entity slots actually rendered
            canons = tuple(
                dict.fromkeys(
                    fills[k] for k in ("e", "e2", "p") if "{" + k + "}" in tmpl
                )
            )
            cases.append(FixtureCase(text, klass, canons))
    return cases


CASES: list[FixtureCase] = build_fixture()
RESULTS: list[tuple[FixtureCase, IntentResult]] = [
    (c, run(c.text, c.canons)) for c in CASES
]


# ---------------------------------------------------------------------------
# Fixture-level assertions
# ---------------------------------------------------------------------------


def test_intent_id_and_fixture_size():
    assert INTENT_ID == "intent/v2"
    assert len(CASES) >= 600
    present = {c.expected for c in CASES}
    assert present == set(IntentClass), (
        f"fixture misses classes: {set(IntentClass) - present}"
    )
    per_class = {k: 0 for k in IntentClass}
    for c in CASES:
        per_class[c.expected] += 1
    assert min(per_class.values()) >= 30


def test_macro_accuracy_and_per_class_floor():
    correct = {k: 0 for k in IntentClass}
    total = {k: 0 for k in IntentClass}
    misses: list[tuple[str, IntentClass, IntentClass]] = []
    for case, res in RESULTS:
        total[case.expected] += 1
        if res.primary == case.expected:
            correct[case.expected] += 1
        else:
            misses.append((case.text, case.expected, res.primary))
    recalls = {k: correct[k] / total[k] for k in IntentClass}
    macro = sum(recalls.values()) / len(recalls)
    detail = "; ".join(f"{k.value}={v:.3f}" for k, v in recalls.items())
    assert macro >= 0.90, (
        f"macro-accuracy {macro:.3f} < 0.90\nper-class: {detail}\n"
        f"misses (first 25): {misses[:25]}"
    )
    for k, r in recalls.items():
        assert r >= 0.85, f"class {k.value} recall {r:.3f} < 0.85"


def test_primary_first_and_rule_trace_populated():
    for case, res in RESULTS:
        assert res.classes, case.text
        assert res.classes[0] == res.primary, case.text
        assert len(set(res.classes)) == len(res.classes), case.text
        assert res.rule_trace, case.text
        assert res.rule_trace[0].endswith(res.primary.value), (
            case.text,
            res.rule_trace,
        )


def test_determinism():
    for case in CASES[::7]:
        a = run(case.text, case.canons)
        b = run(case.text, case.canons)
        assert a == b
        na = mk(case.text)
        assert decompose(na, a) == decompose(na, b)


# ---------------------------------------------------------------------------
# Hand-written edge cases (precedence, secondary classes, channel exclusion)
# ---------------------------------------------------------------------------


def test_identifier_channel_excluded_from_phrase_matching():
    # "still-0817" lives on the identifier channel; the "still" inside it
    # must not produce a current_value class.
    res = run("what is still-0817")
    assert res.primary == IntentClass.IDENTIFIER
    assert res.classes == (IntentClass.IDENTIFIER,)
    assert IntentClass.CURRENT_VALUE not in res.classes


def test_temporal_point_preempts_multi_hop():
    res = run("when did alice and bob meet", ("alice", "bob"))
    assert res.primary == IntentClass.TEMPORAL_POINT
    assert res.classes == (IntentClass.TEMPORAL_POINT, IntentClass.MULTI_HOP)


def test_duration_between_beats_count():
    res = run("how many days between the move and the wedding")
    assert res.primary == IntentClass.DURATION
    assert IntentClass.COUNT_AGGREGATE not in res.classes


def test_how_many_days_ago_is_point_not_count_or_duration():
    res = run("how many days ago did the conference start")
    assert res.primary == IntentClass.TEMPORAL_POINT
    assert IntentClass.COUNT_AGGREGATE not in res.classes
    assert IntentClass.DURATION not in res.classes


def test_comparison_beats_temporal_range_modifier():
    res = run("what's the difference between the march and june reports")
    assert res.primary == IntentClass.COMPARISON
    assert IntentClass.TEMPORAL_RANGE in res.classes


def test_would_i_enjoy_is_preference_not_open_domain():
    res = run("would i enjoy the sailing trip")
    assert res.primary == IntentClass.PREFERENCE
    assert IntentClass.OPEN_DOMAIN in res.classes


def test_abstain_preempts_multi_hop():
    res = run("have i ever met alice and bob", ("alice", "bob"))
    assert res.primary == IntentClass.ABSTAIN_LIKELY
    assert IntentClass.MULTI_HOP in res.classes


def test_wh_initial_is_not_abstain():
    res = run("what did i say about the deadline")
    assert res.primary == IntentClass.LOOKUP


def test_aux_initial_verification_is_abstain():
    assert run("did i lock the door").primary == IntentClass.ABSTAIN_LIKELY
    assert run("have we paid the invoice").primary == IntentClass.ABSTAIN_LIKELY


def test_no_content_query_is_abstain():
    res = run("what is it")
    assert res.primary == IntentClass.ABSTAIN_LIKELY


def test_empty_query_is_abstain():
    res = classify(NormAnalysis("norm/v2", (), (), ""), (), ())
    assert res.primary == IntentClass.ABSTAIN_LIKELY


def test_single_canon_relative_chain_is_multi_hop():
    res = run("the cafe that marco recommended", ("marco",))
    assert res.primary == IntentClass.MULTI_HOP


def test_who_initial_is_not_relative_chain():
    res = run("who is alice", ("alice",))
    assert res.primary == IntentClass.LOOKUP


def test_open_domain_modal_plus_persona():
    assert run("should i buy the macbook").primary == IntentClass.OPEN_DOMAIN
    assert run("is it likely that we move").primary == IntentClass.OPEN_DOMAIN


def test_non_persona_modal_is_not_open_domain():
    # modal without first-person subject -> no open_domain rule
    res = run("would the launch slip again")
    assert res.primary != IntentClass.OPEN_DOMAIN


def test_lookup_fallback():
    assert run("tell me about the garden project").primary == IntentClass.LOOKUP
    assert run("the warranty document").primary == IntentClass.LOOKUP


def test_count_with_range_secondary():
    res = run("how many flights did i take this year")
    assert res.primary == IntentClass.COUNT_AGGREGATE
    assert IntentClass.TEMPORAL_RANGE in res.classes


def test_temporal_range_frames():
    for q in (
        "what did we do last week",
        "what happened between march and june",
        "what did i work on from january to april",
        "what happened during the summer",
        "what did we do yesterday",
        "what did we cover on friday",
        "what happened the other day",
    ):
        assert run(q).primary == IntentClass.TEMPORAL_RANGE, q


def test_window_is_none_until_pipeline_resolves():
    # classify() receives no query_time anchor; the window is attached
    # downstream by temporal/v2 (V7-09.05).
    for _, res in RESULTS[:50]:
        assert res.window is None


# ---------------------------------------------------------------------------
# decompose (V7-05.13)
# ---------------------------------------------------------------------------


def test_decompose_comparison_splits_sides():
    norm = mk("compare python and ruby")
    res = classify(norm, (), ())
    assert res.primary == IntentClass.COMPARISON
    facets = decompose(norm, res)
    assert 2 <= len(facets) <= MAX_FACETS
    texts = [f.text for f in facets]
    assert "python" in texts[0] and "ruby" in texts[1]


def test_decompose_comparison_shared_tail():
    norm = mk("compare the ipad and the kindle for reading")
    res = classify(norm, (), ())
    facets = decompose(norm, res)
    assert 2 <= len(facets) <= MAX_FACETS
    for f in facets:
        assert "reading" in f.text.split()


def test_decompose_multi_hop_relative_clause():
    norm = mk("the cafe that marco recommended")
    res = classify(norm, ("marco",), ())
    assert res.primary == IntentClass.MULTI_HOP
    facets = decompose(norm, res)
    assert len(facets) >= 2
    assert facets[0].text.split() == ["cafe"]
    assert "marco" in facets[1].text.split()


def test_decompose_caps_at_max_facets():
    norm = mk("compare alpha and beta and gamma and delta")
    res = classify(norm, (), ())
    facets = decompose(norm, res)
    assert len(facets) == MAX_FACETS


def test_decompose_non_decomposable_returns_singleton():
    for q, canons in (
        ("what is the roadmap", ()),
        ("what did alice say to bob about the trip", ("alice", "bob")),
        ("when did we move", ()),
    ):
        norm = mk(q)
        res = classify(norm, canons, ())
        facets = decompose(norm, res)
        assert facets == (norm,), q


def test_decompose_facets_carry_original_terms():
    for case, res in RESULTS:
        if res.primary not in (IntentClass.MULTI_HOP, IntentClass.COMPARISON):
            continue
        norm = mk(case.text)
        facets = decompose(norm, res)
        assert 1 <= len(facets) <= MAX_FACETS, case.text
        norm_terms = set(t.term for t in norm.terms)
        for f in facets:
            assert isinstance(f, NormAnalysis)
            assert f.analyzer_id == norm.analyzer_id
            assert f.terms, case.text
            assert all(t.term in norm_terms for t in f.terms), case.text
