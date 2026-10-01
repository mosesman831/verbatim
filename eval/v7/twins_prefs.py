"""Owned preference-heavy twin generator (V7-24.10 ``owned_preferences``).

Seeded, fully-invented corpus of first-person preference statements spread
across sessions, built on the §32.12 ``pref/v1`` pattern families:

* positive — ``love|like|enjoy|adore|prefer|am into|am a fan of|
  can't get enough of``
* negative — ``hate|dislike|can't stand|am not a fan of|avoid|never``
* habitual — ``usually|always|tend to|typically``
* comparative — ``my favorite X is Y`` / ``X is my favorite`` /
  ``I'd rather X than Y``
* constraint — ``I'm allergic to|I don't eat|I'm vegetarian|vegan``

Strength ordering follows §32.12 (``constraint > favorite > love_hate >
like_dislike > habitual``) and is recorded per unit in
``unit["meta"]["pref"]`` so a downstream preference extractor can be scored
field-by-field (subject canon, polarity, strength, object span).

Task kinds — each carries the ``preference`` annotation the spec reports as
its own Track R/Q row (V7-24.11):

* ``preference_recall`` — "what's my favorite X" / "do I like X" probes;
  gold = the statement units for that slot.
* ``preference_apply`` — LME ``single-session-preference`` twin: a scenario
  request ("suggest a lunch spot") whose correct answer depends on several
  earlier preference/constraint statements; gold = all governing units.
* ``preference_update`` — an old favorite replaced by a newer statement;
  ``gold_unit_ids`` is the *current* statement and ``historical_ids`` the
  superseded one (``current_value`` semantics, V7-24.2 knowledge-update).
* ``preference_comparative`` — "would I rather A or B" against an explicit
  comparative statement.
* ``preference_negative`` — only §32.12-excluded forms exist for the asked
  object (hypothetical, quoted, hedged, negated-hypothetical);
  ``expected_abstain`` is true and the excluded units are listed under
  ``distractor_ids`` — never gold.

Filler text avoids the §32.12 lexicon entirely so the only preference-bearing
units are the planted ones. Deterministic per ``(seed, counts)``; stdlib
only; no benchmark text.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List, Optional, Tuple

GENERATOR_ID = "twins_prefs/v1"
CONSTANTS_TAG = "provisional/v7-r0"
CORPUS_NAME = "owned_preferences"
DEFAULT_SEED = 20260923

BASE_US = 1_704_067_200_000_000  # 2024-01-01T00:00:00Z
DAY_US = 86_400_000_000
HOUR_US = 3_600_000_000

#: §32.12 strength labels (mirrors PreferenceFact.strength).
STRENGTHS: Tuple[str, ...] = (
    "constraint",
    "favorite",
    "love_hate",
    "like_dislike",
    "habitual",
)

# ---------------------------------------------------------------------------
# slot catalog — invented objects only
# ---------------------------------------------------------------------------

SLOTS: Dict[str, Dict[str, Any]] = {
    "favorite_food": {
        "noun": "food",
        "objects": [
            "saffron risotto", "miso eggplant", "cacio e pepe",
            "charred leeks", "plum galette", "smoked trout",
            "peanut stew", "fennel salad", "barley soup", "fig tart",
        ],
    },
    "favorite_color": {
        "noun": "color",
        "objects": [
            "deep teal", "ochre", "slate blue", "terracotta", "sage green",
            "maroon", "periwinkle", "charcoal", "mustard", "seafoam",
        ],
    },
    "favorite_music": {
        "noun": "music",
        "objects": [
            "modal jazz", "ambient techno", "baroque pop", "desert blues",
            "chamber folk", "afrobeat", "post-rock", "bossa nova",
        ],
    },
    "favorite_book": {
        "noun": "book",
        "objects": [
            "The Salt Meridian", "Cartographer's Winter", "Low Orbit",
            "The Briar Archive", "Salt and Cinder", "Quiet Machines",
        ],
    },
    "favorite_movie": {
        "noun": "movie",
        "objects": [
            "The Long Estuary", "Paper Lanterns", "Night Ferry",
            "The Glass Orchard", "Winter Signals", "Harbor Lights",
        ],
    },
    "drink": {
        "noun": "drink",
        "objects": [
            "oat flat white", "genmaicha", "sparkling water with lime",
            "cold brew", "rooibos", "ginger tea",
        ],
    },
    "hobbies": {
        "noun": "hobby",
        "objects": [
            "bouldering", "letterpress printing", "birdwatching",
            "sourdough baking", "urban sketching", "trail running",
            "chess puzzles", "kayaking",
        ],
    },
    "sport": {
        "noun": "sport",
        "objects": [
            "climbing", "swimming", "table tennis", "rowing", "fencing",
        ],
    },
    "restaurant_cuisine": {
        "noun": "cuisine",
        "objects": [
            "Ethiopian", "Oaxacan", "Basque", "Vietnamese", "Sicilian",
            "Gujarati",
        ],
    },
    "meeting_time": {
        "noun": "meeting time",
        "objects": [
            "early morning", "right after lunch", "late afternoon",
            "mid-morning",
        ],
    },
    # constraint-only slots (§32.12 constraint forms; §32.11 families)
    "diet": {
        "noun": "diet",
        "objects": ["vegetarian", "pescatarian", "vegan"],
        "constraint_only": True,
    },
    "allergies": {
        "noun": "allergy",
        "objects": ["peanuts", "shellfish", "sesame", "tree nuts", "dairy"],
        "constraint_only": True,
    },
}

DIET_OBJECTS: Tuple[str, ...] = tuple(SLOTS["diet"]["objects"])
ALLERGY_OBJECTS: Tuple[str, ...] = tuple(SLOTS["allergies"]["objects"])
TIMES: Tuple[str, ...] = ("noon", "12:30", "11:45", "13:00", "12:15")

# ---------------------------------------------------------------------------
# statement templates per §32.12 form
# ---------------------------------------------------------------------------

def _stmt_positive(obj: str, rng: random.Random) -> Tuple[str, str]:
    tpl = rng.choice(
        (
            "I love {o}.",
            "I really like {o}.",
            "I enjoy {o}.",
            "I adore {o}.",
            "I prefer {o}.",
            "I'm into {o}.",
            "I'm a fan of {o}.",
            "I can't get enough of {o}.",
        )
    )
    text = tpl.format(o=obj)
    strength = "love_hate" if any(
        w in text for w in ("love", "adore", "can't get enough")
    ) else "like_dislike"
    return text, strength


def _stmt_negative(obj: str, rng: random.Random) -> Tuple[str, str]:
    tpl = rng.choice(
        (
            "I hate {o}.",
            "I dislike {o}.",
            "I can't stand {o}.",
            "I'm not a fan of {o}.",
            "I avoid {o}.",
            "I never order {o}.",
        )
    )
    text = tpl.format(o=obj)
    strength = "love_hate" if any(
        w in text for w in ("hate", "can't stand")
    ) else "like_dislike"
    return text, strength


def _stmt_habitual(verb_phrase: str, rng: random.Random) -> Tuple[str, str]:
    tpl = rng.choice(
        (
            "I usually {v}.",
            "I always {v}.",
            "I tend to {v}.",
            "I typically {v}.",
        )
    )
    return tpl.format(v=verb_phrase), "habitual"


def _stmt_favorite(slot_noun: str, obj: str, rng: random.Random) -> Tuple[str, str]:
    tpl = rng.choice(
        (
            "My favorite {n} is {o}.",
            "{o} is my favorite {n}.",
        )
    )
    return tpl.format(n=slot_noun, o=obj), "favorite"


def _stmt_comparative(a: str, b_: str) -> Tuple[str, str]:
    return f"I'd rather have {a} than {b_}.", "favorite"


def _stmt_constraint(obj: str, rng: random.Random) -> Tuple[str, str]:
    if obj in DIET_OBJECTS:
        tpl = rng.choice(("I'm {o}.", "I eat {o}.", "I keep a {o} diet."))
        return tpl.format(o=obj), "constraint"
    tpl = rng.choice(
        ("I'm allergic to {o}.", "I don't eat {o}.", "I have to avoid {o}.")
    )
    return tpl.format(o=obj), "constraint"


#: §32.12 excluded forms — planted as distractors, never gold.
def _stmt_excluded(obj: str, rng: random.Random) -> Tuple[str, str]:
    tpl, form = rng.choice(
        (
            ("If I liked {o}, I'd bring it up all the time.", "hypothetical"),
            ("Maren said \"I love {o}\" at lunch.", "quoted"),
            ("I might like {o}; not sure yet.", "hedged"),
            ("It's not that I love {o} — it just grew on people.", "negated_hypothetical"),
        )
    )
    return tpl.format(o=obj), form


#: cue-free filler — no §32.12 lexemes (asserted in tests).
_FILLER: Tuple[str, ...] = (
    "The {s} deploy wrapped early.",
    "Standup moved to {t} tomorrow.",
    "Can you skim the {w} before Friday?",
    "The {s} graphs look calm this week.",
    "Package arrived; the {w} is on my desk.",
    "Quiet afternoon — inbox is almost empty.",
    "The {s} canary looks clean so far.",
    "Weekend was restful; back to the {w} now.",
)

_FILLER_NOUNS: Tuple[str, ...] = (
    "report", "spreadsheet", "diagram", "proposal", "prototype",
)
_FILLER_SERVICES: Tuple[str, ...] = (
    "osprey-api", "tern-worker", "magpie-web", "puffin-ml", "wren-db",
)

#: §32.12 lexemes that must never appear in filler.
PREF_LEXEMES: Tuple[str, ...] = (
    "love", "like", "enjoy", "adore", "prefer", "fan of", "hate",
    "dislike", "can't stand", "avoid", "never", "usually", "always",
    "tend to", "typically", "favorite", "rather", "allergic",
    "vegetarian", "vegan",
)


class _Builder:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.units: List[Dict[str, Any]] = []
        self.tasks: List[Dict[str, Any]] = []
        self._seq = 0
        self._sess_seq = 0
        self._us = BASE_US

    def _next_session_us(self, lo: int = 2, hi: int = 9) -> int:
        self._us += self.rng.randint(lo, hi) * DAY_US
        self._us += self.rng.randint(8, 19) * HOUR_US
        return self._us

    def _tick(self, us: int) -> int:
        return us + self.rng.randint(60, 900) * 1_000_000

    def _session(self) -> str:
        self._sess_seq += 1
        return f"sess-{self._sess_seq:04d}"

    def add_unit(
        self,
        *,
        speaker: str,
        session_id: str,
        text: str,
        occurred_us: int,
        perspective: str,
        kind: str = "turn",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._seq += 1
        unit = {
            "id": f"u-{self._seq:05d}",
            "kind": kind,
            "speaker": speaker,
            "session_id": session_id,
            "text": text,
            "occurred_us": occurred_us,
            "perspective": perspective,
        }
        if meta:
            unit["meta"] = meta
        self.units.append(unit)
        return unit

    def add_pref(
        self,
        session_id: str,
        us: int,
        *,
        slot: str,
        obj: str,
        polarity: str,
        strength: str,
        form: str,
        text: str,
        excluded_form: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.add_unit(
            speaker="user",
            session_id=session_id,
            text=text,
            occurred_us=us,
            perspective="user_stated",
            meta={
                "pref": {
                    "slot": slot,
                    "object_text": obj,
                    "polarity": polarity,
                    "strength": strength,
                    "form": form,
                    "excluded_form": excluded_form,
                }
            },
        )

    def add_filler(self, session_id: str, us: int) -> Dict[str, Any]:
        text = self.rng.choice(_FILLER).format(
            s=self.rng.choice(_FILLER_SERVICES),
            w=self.rng.choice(_FILLER_NOUNS),
            t=self.rng.choice(TIMES),
        )
        return self.add_unit(
            speaker="user" if self.rng.random() < 0.6 else "assistant",
            session_id=session_id,
            text=text,
            occurred_us=us,
            perspective=(
                "user_stated" if self.rng.random() < 0.75 else "agent_stated"
            ),
        )


def corpus_digest(corpus: Dict[str, Any]) -> str:
    canon = {
        "name": corpus.get("name"),
        "generator": corpus.get("generator"),
        "seed": corpus.get("seed"),
        "units": corpus.get("units"),
        "tasks": corpus.get("tasks"),
    }
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# emitters
# ---------------------------------------------------------------------------

_SLOT_KEYS: Tuple[str, ...] = tuple(SLOTS.keys())
_FAVORABLE_KEYS: Tuple[str, ...] = tuple(
    k for k, v in SLOTS.items() if not v.get("constraint_only")
)


def _slot(i: int) -> Tuple[str, Dict[str, Any]]:
    key = _SLOT_KEYS[i % len(_SLOT_KEYS)]
    return key, SLOTS[key]


def _favorable_slot(i: int) -> Tuple[str, Dict[str, Any]]:
    key = _FAVORABLE_KEYS[i % len(_FAVORABLE_KEYS)]
    return key, SLOTS[key]


def _emit_recall(b: _Builder, i: int) -> None:
    slot_key, slot = _slot(i)
    obj = slot["objects"][i % len(slot["objects"])]
    if slot.get("constraint_only"):
        text, strength = _stmt_constraint(obj, b.rng)
        polarity, form = "constraint", "constraint"
    elif slot_key == "meeting_time":
        text, strength = _stmt_habitual(
            f"take my focus block in the {obj}", b.rng
        )
        polarity, form = "habitual", "habitual"
    else:
        polarity = b.rng.choice(
            ("positive", "positive", "negative", "habitual")
        )
        if polarity == "positive":
            text, strength = _stmt_positive(obj, b.rng)
        elif polarity == "negative":
            text, strength = _stmt_negative(obj, b.rng)
        else:
            text, strength = _stmt_habitual(f"pick {obj}", b.rng)
        form = polarity

    session = b._session()
    us = b._next_session_us()
    unit = b.add_pref(
        session,
        us,
        slot=slot_key,
        obj=obj,
        polarity=polarity,
        strength=strength,
        form=form,
        text=text,
    )
    for _ in range(b.rng.randint(1, 3)):
        us = b._tick(us)
        b.add_filler(session, us)

    if polarity == "negative":
        q = f"How do I feel about {obj}?"
    elif slot_key == "meeting_time":
        q = "When do I usually take my focus block?"
    elif slot_key.startswith("favorite_"):
        q = f"What's my favorite {slot['noun']}?"
    else:
        q = f"Do I like {obj}?"
    b.tasks.append(
        {
            "task_id": f"pref-rec-{i:04d}",
            "kind": "preference_recall",
            "query": q,
            "task_text": q,
            "gold_unit_ids": [unit["id"]],
            "expected_abstain": False,
            "preference": {
                "subject": "user",
                "slot": slot_key,
                "polarity": polarity,
                "strength": strength,
                "object_text": obj,
            },
            "meta": {},
        }
    )


#: scenario → slots whose preferences govern the right answer
_APPLY_SCENARIOS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Suggest a lunch spot for tomorrow.", ("favorite_food", "restaurant_cuisine", "diet", "allergies")),
    ("Pick a playlist for my evening run.", ("favorite_music", "sport")),
    ("Recommend a weekend activity.", ("hobbies", "sport")),
    ("Choose a gift for me to bring a friend.", ("hobbies", "favorite_book")),
    ("Propose a dinner reservation.", ("restaurant_cuisine", "diet", "allergies", "favorite_food")),
    ("When should we book my focus block?", ("meeting_time",)),
)


def _emit_apply(b: _Builder, i: int) -> None:
    scenario, slot_keys = _APPLY_SCENARIOS[i % len(_APPLY_SCENARIOS)]
    session = b._session()
    us = b._next_session_us()
    gold: List[str] = []
    prefs_used: List[Dict[str, Any]] = []
    for j, sk in enumerate(slot_keys):
        if sk == "diet":
            obj = DIET_OBJECTS[(i + j) % len(DIET_OBJECTS)]
            text, strength = _stmt_constraint(obj, b.rng)
            form, polarity = "constraint", "constraint"
        elif sk == "allergies":
            obj = ALLERGY_OBJECTS[(i + j) % len(ALLERGY_OBJECTS)]
            text, strength = _stmt_constraint(obj, b.rng)
            form, polarity = "constraint", "constraint"
        elif sk == "meeting_time":
            obj = TIMES[(i + j) % len(TIMES)]
            text, strength = _stmt_habitual(
                f"take my focus block at {obj}", b.rng
            )
            form, polarity = "habitual", "habitual"
        else:
            slot = SLOTS[sk]
            obj = slot["objects"][(i + j) % len(slot["objects"])]
            if sk.startswith("favorite_"):
                text, strength = _stmt_favorite(slot["noun"], obj, b.rng)
                form, polarity = "favorite", "positive"
            else:
                text, strength = _stmt_positive(obj, b.rng)
                form, polarity = "positive", "positive"
        u = b.add_pref(
            session,
            us,
            slot=sk,
            obj=obj,
            polarity=polarity,
            strength=strength,
            form=form,
            text=text,
        )
        gold.append(u["id"])
        prefs_used.append(
            {"slot": sk, "object_text": obj, "strength": strength}
        )
        us = b._tick(us)
    for _ in range(b.rng.randint(1, 2)):
        us = b._tick(us)
        b.add_filler(session, us)

    b.tasks.append(
        {
            "task_id": f"pref-app-{i:04d}",
            "kind": "preference_apply",
            "query": scenario,
            "task_text": scenario,
            "gold_unit_ids": gold,
            "expected_abstain": False,
            "preference": {
                "subject": "user",
                "slots": [p["slot"] for p in prefs_used],
                "facts": prefs_used,
            },
            "meta": {"scenario_index": i % len(_APPLY_SCENARIOS)},
        }
    )


def _emit_update(b: _Builder, i: int) -> None:
    slot_key, slot = _favorable_slot(i + 3)
    objs = slot["objects"]
    old_obj = objs[(i * 2) % len(objs)]
    new_obj = objs[(i * 2 + 1) % len(objs)]
    if new_obj == old_obj:
        new_obj = objs[(i * 2 + 2) % len(objs)]

    s1 = b._session()
    us = b._next_session_us()
    old_text, _ = _stmt_favorite(slot["noun"], old_obj, b.rng)
    old_unit = b.add_pref(
        s1,
        us,
        slot=slot_key,
        obj=old_obj,
        polarity="positive",
        strength="favorite",
        form="favorite",
        text=old_text,
    )
    for _ in range(b.rng.randint(1, 3)):
        us = b._tick(us)
        b.add_filler(s1, us)

    # the update lands many sessions later (knowledge-update spacing);
    # dated later than the original but not dragging the global clock
    s2 = b._session()
    us2 = us + b.rng.randint(60, 240) * DAY_US
    new_tpl = b.rng.choice(
        (
            "My favorite {n} is now {o}.",
            "I've switched — my new favorite {n} is {o}.",
            "These days my favorite {n} is {o}.",
        )
    )
    new_text = new_tpl.format(n=slot["noun"], o=new_obj)
    new_unit = b.add_pref(
        s2,
        us2,
        slot=slot_key,
        obj=new_obj,
        polarity="positive",
        strength="favorite",
        form="favorite_update",
        text=new_text,
    )
    for _ in range(b.rng.randint(1, 2)):
        us2 = b._tick(us2)
        b.add_filler(s2, us2)

    q = f"What's my current favorite {slot['noun']}?"
    b.tasks.append(
        {
            "task_id": f"pref-upd-{i:04d}",
            "kind": "preference_update",
            "query": q,
            "task_text": q,
            "gold_unit_ids": [new_unit["id"]],
            "historical_ids": [old_unit["id"]],
            "expected_abstain": False,
            "expect": {"current_object": new_obj, "supersedes": old_unit["id"]},
            "preference": {
                "subject": "user",
                "slot": slot_key,
                "polarity": "positive",
                "strength": "favorite",
                "object_text": new_obj,
            },
            "meta": {"update": True},
        }
    )


def _emit_comparative(b: _Builder, i: int) -> None:
    slot_key, slot = _favorable_slot(i + 7)
    objs = slot["objects"]
    a = objs[(i * 3) % len(objs)]
    bb = objs[(i * 3 + 1) % len(objs)]
    if a == bb:
        bb = objs[(i * 3 + 2) % len(objs)]
    text, strength = _stmt_comparative(a, bb)
    session = b._session()
    us = b._next_session_us()
    unit = b.add_pref(
        session,
        us,
        slot=slot_key,
        obj=a,
        polarity="positive",
        strength=strength,
        form="comparative",
        text=text,
    )
    for _ in range(b.rng.randint(1, 3)):
        us = b._tick(us)
        b.add_filler(session, us)
    q = f"Would I rather have {a} or {bb}?"
    b.tasks.append(
        {
            "task_id": f"pref-cmp-{i:04d}",
            "kind": "preference_comparative",
            "query": q,
            "task_text": q,
            "gold_unit_ids": [unit["id"]],
            "expected_abstain": False,
            "preference": {
                "subject": "user",
                "slot": slot_key,
                "polarity": "positive",
                "strength": strength,
                "object_text": a,
                "compared": [a, bb],
            },
            "meta": {},
        }
    )


def _emit_negative(b: _Builder, i: int) -> None:
    """Only §32.12-excluded mentions exist → the honest answer is abstain."""
    slot_key, slot = _slot(i + 5)
    obj = slot["objects"][(i * 7 + 3) % len(slot["objects"])]
    session = b._session()
    us = b._next_session_us()
    text, form = _stmt_excluded(obj, b.rng)
    unit = b.add_pref(
        session,
        us,
        slot=slot_key,
        obj=obj,
        polarity="excluded",
        strength="excluded",
        form=form,
        text=text,
        excluded_form=form,
    )
    for _ in range(b.rng.randint(1, 3)):
        us = b._tick(us)
        b.add_filler(session, us)
    q = f"Do I like {obj}?"
    b.tasks.append(
        {
            "task_id": f"pref-neg-{i:04d}",
            "kind": "preference_negative",
            "query": q,
            "task_text": q,
            "gold_unit_ids": [],
            "distractor_ids": [unit["id"]],
            "expected_abstain": True,
            "preference": {
                "subject": "user",
                "slot": slot_key,
                "polarity": "excluded",
                "excluded_form": form,
                "object_text": obj,
            },
            "meta": {"negative_probe": True},
        }
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def generate(
    seed: int = DEFAULT_SEED,
    *,
    n_recall: int = 110,
    n_apply: int = 80,
    n_update: int = 70,
    n_comparative: int = 50,
    n_negative: int = 20,
    filler_sessions: int = 30,
) -> Dict[str, Any]:
    """Generate the owned preference corpus (defaults: 330 tasks ≥ 300)."""
    for name, val in (
        ("n_recall", n_recall),
        ("n_apply", n_apply),
        ("n_update", n_update),
        ("n_comparative", n_comparative),
        ("n_negative", n_negative),
        ("filler_sessions", filler_sessions),
    ):
        if not isinstance(val, int) or val < 0:
            raise ValueError(f"{name} must be a non-negative int, got {val!r}")

    b = _Builder(seed)
    for i in range(n_recall):
        _emit_recall(b, i)
    for i in range(n_apply):
        _emit_apply(b, i)
    for i in range(n_update):
        _emit_update(b, i)
    for i in range(n_comparative):
        _emit_comparative(b, i)
    for i in range(n_negative):
        _emit_negative(b, i)
    for _ in range(filler_sessions):
        session = b._session()
        us = b._next_session_us()
        for _ in range(b.rng.randint(2, 6)):
            us = b._tick(us)
            b.add_filler(session, us)

    b.units.sort(key=lambda u: (u["occurred_us"], u["id"]))
    corpus = {
        "name": CORPUS_NAME,
        "generator": GENERATOR_ID,
        "constants": CONSTANTS_TAG,
        "seed": seed,
        "units": b.units,
        "tasks": b.tasks,
        "stats": {
            "n_units": len(b.units),
            "n_tasks": len(b.tasks),
            "n_pref_units": sum(
                1 for u in b.units if "pref" in u.get("meta", {})
            ),
            "task_kinds": {
                k: sum(1 for t in b.tasks if t["kind"] == k)
                for k in sorted({t["kind"] for t in b.tasks})
            },
        },
    }
    corpus["digest"] = corpus_digest(corpus)
    return corpus


__all__ = [
    "GENERATOR_ID",
    "CONSTANTS_TAG",
    "CORPUS_NAME",
    "DEFAULT_SEED",
    "PREF_LEXEMES",
    "SLOTS",
    "STRENGTHS",
    "corpus_digest",
    "generate",
]
