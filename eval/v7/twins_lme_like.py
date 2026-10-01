"""Owned LongMemEval-S twin corpus — deterministic seeded generator.

SPEC_V7 §22 (V7-22.05): every external benchmark needs an owned,
license-free twin that CI can run without network or restricted data,
mirroring the benchmark's categories at >= 500 questions per suite. This
module generates the ``owned_lme_like`` twin of LongMemEval-S with
all-invented content (no benchmark text — V7-22.18 anti-gaming).

A generated corpus is a list of self-contained questions. Each question
owns a *haystack*: ``sessions_per_user`` dated chat sessions between a
``user`` and an ``assistant`` (the LongMemEval shape: session ids, turn
ids, speakers, epoch timestamps, ``haystack_dates``, and a
``question_date`` the adapter passes as ``query_time`` per V7-24.04).
Evidence-bearing turns are planted at known positions inside an
otherwise generic filler conversation; every question carries its gold
as ``{session_id, turn_ids}`` references so Track R can score turn- and
session-level recall exactly (V7-22.12).

Categories mirror LongMemEval-S (§24.2) under the owned-twin names:

* ``information_extraction`` — one planted fact in one session, spoken
  by the user (~60%) or the assistant (~40%; recommendations and
  lists). ``subtype`` records ``single_session_user`` /
  ``single_session_assistant``.
* ``multi_session`` — a repeated activity across >= 2 distinct
  sessions; ``count`` ("how many") and ``list`` ("which") variants;
  gold is ALL of the planted refs (recall_all is the §24.2 target).
* ``temporal_reasoning`` — an absolute event date in evidence; the
  query uses relative phrasing ("how many days ago", "what day of the
  week") resolved against ``question_date``; a ``temporal`` annotation
  carries the mode, event date, day delta, and weekday.
* ``knowledge_updates`` — an old value planted in an earlier session
  and a superseding value in a later one; the answer is the NEW value
  while gold covers BOTH turns (§24.2 recall_all of both sessions) and
  the old turn doubles as the distractor. A ``knowledge_update``
  annotation names predecessor/successor refs and both values.
* ``single_session_preference`` — a stated preference applied to a new
  request; carries the ``preference`` sub-annotation (the spec reports
  preference quality separately).
* ``multi_session_user`` — user profile/preference facets aggregated
  across >= 3 distinct sessions; also carries ``preference``.
* ``abstention`` — unanswerable (the ``_abs`` subset): zero supporting
  evidence; about half plant a topically-adjacent near-miss turn as a
  retrieval lure. ``checks.absent_topic`` is a token guaranteed absent
  from every turn.

Determinism contract: ``generate(seed, ...)`` is byte-deterministic —
one seeded ``random.Random`` drives everything, dates derive from a
fixed base via ``datetime.date`` arithmetic, epoch timestamps come from
``calendar.timegm`` (UTC, platform-independent), and ``to_jsonl`` /
``corpus_digest`` serialize with sorted keys. Stdlib only; no network.

CLI: ``python -m eval.v7.twins_lme_like --seed 42 --n 500 \
    --sessions 10 --out owned_lme_like.jsonl``
"""

from __future__ import annotations

import argparse
import calendar
import datetime
import hashlib
import json
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

GENERATOR_ID = "twins_lme_like/v1"
CORPUS_NAME = "owned_lme_like"

DEFAULT_SEED = 42
DEFAULT_N = 500
DEFAULT_SESSIONS = 10
MIN_SESSIONS = 4

#: Session dates are sampled without replacement from day-offsets
#: ``range(_MIN_OFFSET_DAYS, _MAX_OFFSET_DAYS + 1)`` before the question
#: date, so ``sessions_per_user`` is bounded by that range's size.
_MIN_OFFSET_DAYS = 3
_MAX_OFFSET_DAYS = 78
MAX_SESSIONS = _MAX_OFFSET_DAYS - _MIN_OFFSET_DAYS + 1

_BASE_DATE = datetime.date(2025, 4, 1)

# ---------------------------------------------------------------------------
# categories — owned-twin names mirroring the LongMemEval-S type table (§24.2)
# ---------------------------------------------------------------------------

CATEGORY_IE = "information_extraction"
CATEGORY_MS = "multi_session"
CATEGORY_TR = "temporal_reasoning"
CATEGORY_KU = "knowledge_updates"
CATEGORY_SSP = "single_session_preference"
CATEGORY_MSU = "multi_session_user"
CATEGORY_ABS = "abstention"

CATEGORIES: Tuple[str, ...] = (
    CATEGORY_IE,
    CATEGORY_MS,
    CATEGORY_TR,
    CATEGORY_KU,
    CATEGORY_SSP,
    CATEGORY_MSU,
    CATEGORY_ABS,
)

#: Category weights per 100 questions, mirroring the §24.2 LongMemEval-S
#: proportions (single-session-user+assistant 126 -> ie, multi-session
#: 133 -> ms, temporal 133 -> tr, knowledge-update 78 -> ku,
#: single-session-preference 30 -> ssp, _abs 30 -> abs) with the owned
#: profile-aggregation slice (msu) carved modestly out of the
#: multi-session/ie mass. At the default n=500 each weight yields
#: exactly 5*w questions.
CATEGORY_WEIGHTS: Dict[str, int] = {
    CATEGORY_IE: 21,
    CATEGORY_MS: 23,
    CATEGORY_TR: 23,
    CATEGORY_KU: 16,
    CATEGORY_SSP: 6,
    CATEGORY_MSU: 6,
    CATEGORY_ABS: 5,
}

_MONTHS = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)
_WEEKDAYS = (
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
)


def _us(d: datetime.date, hour: int, minute: int) -> int:
    """Epoch microseconds via UTC — platform-independent determinism."""
    dt = datetime.datetime(d.year, d.month, d.day, hour, minute)
    return calendar.timegm(dt.timetuple()) * 1_000_000


def _fmt_day(d: datetime.date) -> str:
    """``March 3`` — the in-text date style used by planted evidence."""
    return f"{_MONTHS[d.month - 1]} {d.day}"


# ---------------------------------------------------------------------------
# filler exchanges — deliberately vague small talk. They carry no names,
# numbers, dates, places, or preference markers, so a filler turn can
# never be evidence for any planted question and never duplicates a gold
# token (the per-question ``checks`` invariants rely on that).
# ---------------------------------------------------------------------------

_FILLERS: Tuple[Tuple[str, str], ...] = (
    ("Hey, good morning!", "Morning! What's on your mind today?"),
    ("Not much, just checking in.", "Always happy to chat."),
    ("Ugh, what a week.", "Tell me about it. Anything I can help with?"),
    ("I'm so behind on laundry.", "One load at a time — start a quick wash now?"),
    ("The weather's been weird lately.", "Right? Keeps you guessing."),
    ("I need to get more sleep.", "A consistent bedtime helps more than you'd think."),
    ("My inbox is a disaster.", "Try a fifteen-minute triage: delete, delegate, do, defer."),
    ("I'm thinking of repainting the hallway.", "Nice — light colors open up a hallway."),
    ("Can't decide what to make for dinner.", "What ingredients do you have around?"),
    ("I binged way too much TV this weekend.", "Happens to the best of us."),
    ("My desk is a mess.", "A five-minute tidy does wonders for focus."),
    ("I keep forgetting to water the plants.", "A weekly reminder on your phone might help."),
    ("Traffic was brutal this morning.", "Hopefully the drive home is kinder."),
    ("I'm craving something sweet.", "Fruit, or go straight for dessert?"),
    ("My phone battery keeps dying.", "Might be time to check battery health in settings."),
    ("I finally organized the garage.", "That feels so satisfying, doesn't it?"),
    ("I need a vacation.", "Even a long weekend can recharge you."),
    ("My keyboard is double-typing.", "Compressed air sometimes fixes a sticky key."),
    ("I keep meaning to call the bank.", "Put it first on tomorrow's list — small wins."),
    ("The grocery store was packed.", "Sunday afternoons are the worst for that."),
    ("I started a jigsaw puzzle yesterday.", "Fun — how many pieces?"),
    ("My headphones went through the wash.", "Oh no! Let them dry fully before testing."),
    ("I might take a walk later.", "A short walk is always a good reset."),
    ("The internet was down all morning.", "Frustrating — did it come back on its own?"),
    ("I'm trying to drink more water.", "Keeping a bottle on your desk really helps."),
    ("I forgot my umbrella again.", "Classic. Keeping a spare at work is a lifesaver."),
    ("The neighbors were up late again.", "Hopefully tonight is quieter."),
    ("I'm learning to juggle.", "Ha! Start with two balls, then add the third."),
    ("My houseplants are thriving.", "Look at you, plant parent of the year."),
    ("I need new shoes soon.", "Comfy soles first — everything else follows."),
    ("The meeting ran so long.", "Meetings that could've been emails, right?"),
    ("My tea went cold again.", "A mug warmer might change your life."),
    ("I can't find my keys.", "Check your pockets — that's where they hide."),
    ("The library hold line is huge.", "Weekend rush — try a weekday evening."),
    ("I'm finally caught up on shows.", "Nice! Anything worth recommending?"),
    ("My commute playlist needs a refresh.", "Shuffle a new genre this week — keeps it fresh."),
    ("I need to clean out the fridge.", "Shelf by shelf — fifteen minutes and it's done."),
    ("My back is sore from gardening.", "A good stretch and a warm shower help."),
    ("I might start journaling.", "Even three lines a day builds the habit."),
    ("The car needs an oil change.", "Worth doing before the next road trip."),
)

# ---------------------------------------------------------------------------
# information_extraction — single planted fact, single gold turn.
# ``speaker`` mirrors the single-session-user vs single-session-assistant
# split in §24.2. ``{sdate}`` resolves to the planting session's date.
# ---------------------------------------------------------------------------

_IE_USER_FACTS: Tuple[Dict[str, Any], ...] = (
    {
        "key": "pet",
        "speaker": "user",
        "params": {
            "sp": ["dog", "cat", "rabbit", "parrot", "bearded dragon"],
            "val": ["Biscuit", "Miso", "Pepper", "Waffles", "Juno",
                    "Pickles", "Maple", "Ziggy", "Clover", "Noodle"],
            "when": ["last weekend", "two weeks ago",
                     "earlier this month"],
        },
        "text": "I adopted a {sp} named {val} {when}.",
        "queries": ["What did I name my {sp}?",
                    "What is my {sp}'s name?"],
        "answer": "{val}",
    },
    {
        "key": "car",
        "speaker": "user",
        "params": {
            "color": ["teal", "crimson", "graphite", "pearl white",
                      "midnight blue"],
            "model": ["Kestrel K2", "Auriga Cross", "Borealis GT",
                      "Vanta S", "Corvid XR", "Lumen E5",
                      "Solano Trail", "Driftline Coupe"],
        },
        "text": "I finally bought a {color} {model}.",
        "queries": ["What car did I buy?",
                    "What kind of car did I end up getting?"],
        "answer": "{color} {model}",
    },
    {
        "key": "job",
        "speaker": "user",
        "params": {
            "title": ["data engineer", "pastry chef", "park ranger",
                      "technical writer", "nurse practitioner",
                      "sound designer"],
            "company": ["Northwind Analytics", "Ferrostack",
                        "Bluepine Health", "Copperleaf Labs",
                        "Solstice Energy", "Pinecrest Bakery",
                        "Marlowe Studios"],
        },
        "text": "I started a new job as a {title} at {company}.",
        "queries": ["Where do I work now?",
                    "What company did I start at?"],
        "answer": "{company}",
    },
    {
        "key": "race",
        "speaker": "user",
        "params": {
            "town": ["Riverton", "Maple Grove", "Cedar Falls",
                     "Harborview", "Larkspur"],
            "dist": ["5K", "10K", "half marathon"],
            "mins": [str(m) for m in range(19, 61)],
        },
        "text": "I ran the {town} {dist} on {sdate} and finished in "
                "{mins} minutes.",
        "queries": ["What was my finish time in the {town} {dist}?",
                    "How fast did I run the {town} {dist}?"],
        "answer": "{mins} minutes",
    },
    {
        "key": "repair",
        "speaker": "user",
        "params": {
            "appliance": ["water heater", "dishwasher",
                          "washing machine", "furnace",
                          "garage door opener"],
            "cost": [str(c) for c in range(120, 901, 15)],
        },
        "text": "The {appliance} repair was finally done — cost me "
                "${cost}.",
        "queries": ["How much did the {appliance} repair cost?",
                    "What did I pay for the {appliance} repair?"],
        "answer": "${cost}",
    },
    {
        "key": "class",
        "speaker": "user",
        "params": {
            "subject": ["pottery", "salsa dancing",
                        "watercolor painting", "sourdough baking",
                        "rock climbing"],
            "venue": ["the community center", "Brightside Studio",
                      "the rec hall", "the loft downtown"],
        },
        "text": "I signed up for a {subject} class at {venue}.",
        "queries": ["What class did I sign up for?",
                    "Which class did I register for?"],
        "answer": "{subject} class",
    },
    {
        "key": "volunteer",
        "speaker": "user",
        "params": {
            "place": ["the animal shelter", "the food bank",
                      "the community garden", "the library"],
        },
        "text": "I started volunteering at {place} on Saturdays.",
        "queries": ["Where did I start volunteering?",
                    "Where do I volunteer now?"],
        "answer": "{place}",
    },
    {
        "key": "garden",
        "speaker": "user",
        "params": {
            "plant": ["heirloom tomatoes", "sunflowers",
                      "lavender bushes", "a dwarf apple tree",
                      "jalapeño peppers"],
        },
        "text": "I planted {plant} in the garden on {sdate}.",
        "queries": ["What did I plant in the garden this year?",
                    "What went into the garden this spring?"],
        "answer": "{plant}",
    },
    {
        "key": "concert",
        "speaker": "user",
        "params": {
            "band": ["The Paper Lanterns", "Echo Vale",
                     "Northern Static", "Juniper Falls",
                     "Velvet Meridian", "The Quiet Parade"],
            "month": ["April", "May", "June", "July", "August"],
        },
        "text": "I scored tickets to see {band} in {month}.",
        "queries": ["Which band do I have tickets to see?",
                    "Who am I seeing in concert in {month}?"],
        "answer": "{band}",
    },
    {
        "key": "trip",
        "speaker": "user",
        "params": {
            "place": ["the Oregon coast", "Santa Fe", "Lake Placid",
                      "the Blue Ridge Mountains", "Mendocino", "Taos"],
            "month": ["May", "June", "July", "September", "October"],
        },
        "text": "I booked a trip to {place} for {month}.",
        "queries": ["Where am I traveling in {month}?",
                    "What trip did I book?"],
        "answer": "{place}",
    },
    {
        "key": "instrument",
        "speaker": "user",
        "params": {
            "instr": ["acoustic guitar", "electronic drum kit",
                      "upright piano", "mandolin", "cello"],
        },
        "text": "I picked up a used {instr} from the classifieds.",
        "queries": ["What instrument did I buy?",
                    "Which instrument did I pick up?"],
        "answer": "{instr}",
    },
    {
        "key": "instrument_cost",
        "speaker": "user",
        "params": {
            "instr": ["acoustic guitar", "electronic drum kit",
                      "upright piano", "mandolin", "cello"],
            "cost": [str(c) for c in range(80, 1201, 20)],
        },
        "text": "I paid ${cost} for a used {instr}.",
        "queries": ["How much did I pay for the {instr}?",
                    "What did the used {instr} cost me?"],
        "answer": "${cost}",
    },
    {
        "key": "language",
        "speaker": "user",
        "params": {
            "lang": ["Portuguese", "Japanese", "Italian", "Korean",
                     "Swahili", "Norwegian"],
            "app": ["LingoNest", "PhraseFox", "Vocablio", "TalkSprout"],
        },
        "text": "I started learning {lang} with the {app} app.",
        "queries": ["Which language am I learning?",
                    "What language did I start learning?"],
        "answer": "{lang}",
    },
    {
        "key": "sister_wedding",
        "speaker": "user",
        "params": {
            "name": ["Priya", "Elena", "Tara", "Sofia", "Naomi",
                     "Imogen"],
            "month": ["June", "July", "September", "October",
                      "December"],
        },
        "text": "My sister {name} is getting married in {month}.",
        "queries": ["When is my sister's wedding?",
                    "What month is {name}'s wedding?"],
        "answer": "{month}",
    },
    {
        "key": "bike",
        "speaker": "user",
        "params": {
            "color": ["forest green", "matte black", "sunset orange",
                      "slate grey"],
            "brand": ["Wren Cycles", "Alder & Pine", "Foxglove",
                      "Stonepath"],
        },
        "text": "My new bike is a {color} {brand}.",
        "queries": ["What brand is my new bike?",
                    "What bike did I get?"],
        "answer": "{brand}",
    },
    {
        "key": "hobby",
        "speaker": "user",
        "params": {
            "hobby": ["birdwatching", "origami", "chess",
                      "calligraphy", "whittling", "astrophotography"],
        },
        "text": "I've gotten really into {hobby} lately.",
        "queries": ["What new hobby have I picked up?",
                    "Which hobby have I gotten into recently?"],
        "answer": "{hobby}",
    },
)

_IE_ASSISTANT_FACTS: Tuple[Dict[str, Any], ...] = (
    {
        "key": "book_rec",
        "speaker": "assistant",
        "params": {
            "val": ["The Glass Meridian", "Salt and Cinder",
                    "A Lantern in the Pines",
                    "The Cartographer's Daughter", "Wintering Bees",
                    "The Ninth Orchard"],
            "author": ["Iris Calloway", "Theo Marsh", "Bryn Okafor",
                       "Elena Vidal", "Sam Oduya", "Clara Beaumont"],
        },
        "text": "Based on what you enjoyed before, you'd probably "
                "love {val} by {author}.",
        "queries": ["Which book did you recommend?",
                    "What was that book you suggested I read?"],
        "answer": "{val}",
    },
    {
        "key": "restaurant_rec",
        "speaker": "assistant",
        "params": {
            "val": ["Golden Basil", "Saffron Orchid", "Lotus & Lime",
                    "The Copper Skillet", "Marlowe's Table",
                    "Pepperwood"],
            "cuisine": ["Thai", "Ethiopian", "Lebanese", "Vietnamese"],
            "street": ["Alder Street", "Fifth Avenue", "Harbor Lane",
                       "Willow Row"],
        },
        "text": "If you're in the mood for {cuisine} food, {val} on "
                "{street} gets great reviews.",
        "queries": ["What restaurant did you suggest?",
                    "Which restaurant did you recommend?"],
        "answer": "{val}",
    },
    {
        "key": "show_rec",
        "speaker": "assistant",
        "params": {
            "val": ["Harbor Lights", "The Long Way Home", "Coldwater",
                    "Meridian Station", "Farrows End",
                    "The Beekeeper's Map"],
        },
        "text": "You should start with {val} — it's a miniseries, "
                "only six episodes.",
        "queries": ["Which show did you tell me to watch?",
                    "What was that miniseries you recommended?"],
        "answer": "{val}",
    },
    {
        "key": "gear_rec",
        "speaker": "assistant",
        "params": {
            "item": ["tent", "backpack", "headlamp", "camera",
                     "daypack"],
            "val": ["NorthPeak NP-2", "SummitLine 40L", "Aurora B9",
                    "Klarheit K3", "Fieldstone F7"],
            "price": [str(p) for p in range(40, 301, 10)],
        },
        "text": "For your budget, the {val} {item} is the best pick — "
                "solid reviews and under ${price}.",
        "queries": ["Which {item} did you recommend?",
                    "What {item} should I get?"],
        "answer": "{val} {item}",
    },
    {
        "key": "app_rec",
        "speaker": "assistant",
        "params": {
            "val": ["HabitSeed", "Trailmark", "QuietFocus",
                    "MealMosaic", "LedgerLeaf", "PacePal"],
            "purpose": ["tracking habits", "logging workouts",
                        "meal planning", "budget tracking"],
        },
        "text": "The {val} app is great for {purpose}.",
        "queries": ["What app did you suggest?",
                    "Which app did you recommend for {purpose}?"],
        "answer": "{val}",
    },
    {
        "key": "trail_rec",
        "speaker": "assistant",
        "params": {
            "val": ["Fern Hollow Loop", "Eagle Ridge",
                    "Cascade Pass Trail", "Birchwood Loop",
                    "Otter Creek Trail"],
            "miles": ["2", "3", "4", "5", "6"],
        },
        "text": "If you want a beginner-friendly hike, {val} is "
                "lovely — about {miles} miles round trip.",
        "queries": ["Which trail did you recommend?",
                    "What was that beginner hike you suggested?"],
        "answer": "{val}",
    },
    {
        "key": "recipe_rec",
        "speaker": "assistant",
        "params": {
            "val": ["shakshuka", "mushroom risotto",
                    "sheet-pan gnocchi", "coconut lentil curry"],
            "ing": ["eggs", "arborio rice", "gnocchi", "red lentils"],
        },
        "text": "A good recipe for that is {val} — it's built around "
                "{ing}.",
        "queries": ["What recipe did you suggest?",
                    "Which dish did you recommend?"],
        "answer": "{val}",
    },
    {
        "key": "podcast_rec",
        "speaker": "assistant",
        "params": {
            "val": ["The Quiet Ledger", "Signal & Noise",
                    "The Understory", "Night Freight Radio"],
            "genre": ["history", "science", "true-story",
                      "interview"],
        },
        "text": "If you like {genre} podcasts, {val} is worth a "
                "listen.",
        "queries": ["Which podcast did you recommend?",
                    "What was that podcast you suggested?"],
        "answer": "{val}",
    },
)

# ---------------------------------------------------------------------------
# multi_session — one mention planted in k distinct sessions. ``token``
# is a substring unique to the planted turns (filler never contains it)
# so a counter's recall is exactly measurable; ``list_answer="dates"``
# renders the list answer from the planted session dates instead of the
# item values.
# ---------------------------------------------------------------------------

_MS_ACTIVITIES: Tuple[Dict[str, Any], ...] = (
    {
        "key": "hiking", "token": "hiking",
        "turn": "I went hiking at {item} on {sdate}.",
        "q_count": ["How many times have I mentioned going hiking?",
                    "How many hikes have I told you about?"],
        "q_list": "Which trails have I told you about?",
        "places": ["Eagle Ridge", "Fern Hollow", "Cascade Pass",
                   "Birchwood Loop", "Otter Creek", "Marlin Falls",
                   "Stonebridge Trail", "Wren Canyon",
                   "Larkspur Ridge"],
    },
    {
        "key": "books", "token": "finished reading",
        "turn": "I finished reading {item} last night.",
        "q_count": ["How many books have I mentioned finishing?",
                    "How many books did I tell you I read?"],
        "q_list": "Which books have I told you I finished?",
        "places": ["The Glass Meridian", "Salt and Cinder",
                   "A Lantern in the Pines", "Wintering Bees",
                   "The Ninth Orchard", "Harbor of Small Hours",
                   "The Juniper Archive", "Prairie Lights"],
    },
    {
        "key": "coffee", "token": "coffee place",
        "turn": "Tried a new coffee place called {item} — the "
                "{drink} was excellent.",
        "q_count": ["How many new coffee places have I tried?",
                    "How many coffee shops have I mentioned?"],
        "q_list": "Which coffee places have I told you about?",
        "places": ["Ember & Oak", "Copper Cup", "Marble Bean",
                   "Fogcatcher", "The Daily Grindstone",
                   "Ninth & Pine", "Blue Kettle", "Cardamom House"],
        "extra": {"drink": ["oat latte", "pour-over", "cortado",
                            "cold brew", "flat white"]},
    },
    {
        "key": "yoga", "token": "yoga",
        "turn": "Went to a {item} yoga class on {sdate}.",
        "q_count": ["How many yoga classes have I mentioned?"],
        "q_list": "Which yoga styles have I tried?",
        "places": ["vinyasa", "yin", "power", "restorative", "hatha",
                   "ashtanga"],
    },
    {
        "key": "swim", "token": "laps",
        "turn": "Did {item} laps at the pool on {sdate}.",
        "q_count": ["How many times have I mentioned swimming laps?",
                    "How many swim sessions have I told you about?"],
        "q_list": "On which dates have I been swimming?",
        "places": ["20", "24", "30", "36", "40", "48"],
        "list_answer": "dates",
    },
    {
        "key": "aunt", "token": "Aunt",
        "turn": "Had a long call with Aunt {item} on {sdate}.",
        "q_count": ["How many times have I mentioned calling an aunt?"],
        "q_list": "Which aunts have I called?",
        "places": ["Miriam", "Dolores", "June", "Bess", "Agatha",
                   "Pearl"],
    },
    {
        "key": "game_night", "token": "game night",
        "turn": "Played {item} at game night on {sdate}.",
        "q_count": ["How many game nights have I mentioned?"],
        "q_list": "Which games have I played at game night?",
        "places": ["Copper & Clover", "The Underhill Game",
                   "Brine & Bone", "Signal Fires", "Wool & Whimsy",
                   "Nine Bridges"],
    },
    {
        "key": "farmers_market", "token": "farmers market",
        "turn": "Stopped by the {item} farmers market on {sdate}.",
        "q_count": ["How many farmers markets have I mentioned?"],
        "q_list": "Which farmers markets have I visited?",
        "places": ["Riverside", "Old Town", "Hillcrest",
                   "Marigold Square", "Depot District"],
    },
)

# ---------------------------------------------------------------------------
# temporal_reasoning — an absolute event date inside evidence; the query
# phrases the ask relative to ``question_date``. ``{edate}`` renders the
# event date; ``{thing}`` picks the event kind.
# ---------------------------------------------------------------------------

_TR_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "key": "class_start", "mode": "days_ago",
        "thing": ["pottery", "spin", "fencing", "calligraphy",
                  "improv"],
        "turn": "I started my {thing} class on {edate}.",
        "queries": ["How many days ago did my {thing} class start?",
                    "How many days has it been since my {thing} "
                    "class started?"],
    },
    {
        "key": "joined", "mode": "weeks_ago",
        "thing": ["the climbing gym", "a book club",
                  "the rec soccer league", "a community choir"],
        "turn": "I joined {thing} on {edate}.",
        "queries": ["How many weeks have I been part of {thing}?",
                    "For how many weeks have I been in {thing}?"],
    },
    {
        "key": "appointment", "mode": "weekday",
        "thing": ["doctor's", "vet", "tax prep", "optometrist",
                  "car service"],
        "turn": "My {thing} appointment is on {edate}.",
        "queries": ["What day of the week is my {thing} appointment?"],
    },
    {
        "key": "event_until", "mode": "days_until",
        "thing": ["charity gala", "wine tasting", "half marathon",
                  "conference", "art fair", "family reunion"],
        "turn": "The {thing} is coming up on {edate}.",
        "queries": ["How many days until the {thing}?",
                    "How many days away is the {thing}?"],
    },
    {
        "key": "last_visit", "mode": "days_ago",
        "thing": ["my aunt Rosa", "my old neighbor Sam",
                  "my friend Priya", "my mentor Dale"],
        "turn": "I last saw {thing} on {edate}.",
        "queries": ["How many days ago did I last see {thing}?"],
    },
)

# ---------------------------------------------------------------------------
# knowledge_updates — ``old`` planted in an earlier session, ``new`` in
# a later one. The answer is always the NEW value; both turns are gold
# (the old turn is the deliberate distractor). Templates may reference
# {a} (old value), {b} (new value), {odate}, {ndate}.
# ---------------------------------------------------------------------------

_KU_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "key": "city",
        "vals": ["Denver", "Portland", "Austin", "Boulder",
                 "Sacramento", "Asheville", "Boise", "Madison"],
        "old": "I live in {a} — been here a while now.",
        "new": "Quick update: I moved to {b}! Still unpacking boxes.",
        "queries": ["Where do I live now?",
                    "What city am I living in?"],
        "answer": "{b}",
    },
    {
        "key": "employer",
        "vals": ["Northwind Analytics", "Ferrostack",
                 "Bluepine Health", "Copperleaf Labs",
                 "Solstice Energy", "Marlowe Studios"],
        "old": "Work at {a} is keeping me busy.",
        "new": "Big news — I just started a new role at {b}!",
        "queries": ["Where do I work?",
                    "What's my current employer?"],
        "answer": "{b}",
    },
    {
        "key": "phone_plan",
        "vals": ["the basic prepaid plan", "the family plan",
                 "the unlimited plan", "the budget 5G plan"],
        "old": "I'm on {a} for my phone.",
        "new": "I finally switched my phone plan to {b} — better "
               "coverage.",
        "queries": ["What phone plan am I on?",
                    "Which phone plan do I have now?"],
        "answer": "{b}",
    },
    {
        "key": "favorite_restaurant",
        "vals": ["Golden Basil", "The Copper Skillet",
                 "Marlowe's Table", "Saffron Orchid", "Lotus & Lime",
                 "Pepperwood"],
        "old": "{a} is still my favorite restaurant.",
        "new": "Official announcement: {b} has dethroned {a} as my "
               "favorite restaurant.",
        "queries": ["What's my favorite restaurant?",
                    "Which restaurant is my favorite now?"],
        "answer": "{b}",
    },
    {
        "key": "gym",
        "vals": ["IronWorks", "the YMCA", "Peak Fitness",
                 "Greenline Gym", "the rec center"],
        "old": "I work out at {a} most mornings.",
        "new": "I canceled my {a} membership and joined {b} — it's "
               "closer to work.",
        "queries": ["Which gym do I go to?",
                    "Where do I work out now?"],
        "answer": "{b}",
    },
    {
        "key": "car",
        "vals": ["a teal Kestrel K2", "a graphite Vanta S",
                 "a white Auriga Cross", "a blue Driftline Coupe",
                 "a silver Borealis GT"],
        "old": "I drive {a} — still going strong.",
        "new": "Traded in the old ride — I drive {b} now.",
        "queries": ["What car do I drive?",
                    "What's my current car?"],
        "answer": "{b}",
    },
    {
        "key": "project_name",
        "vals": ["Nightjar", "Cloudberry", "Halfmoon",
                 "Foxglove Labs", "Dovetail"],
        "old": "My side project is called {a} for now.",
        "new": "I renamed the side project — it's {b} now.",
        "queries": ["What's my side project called?",
                    "What's the name of my side project?"],
        "answer": "{b}",
    },
    {
        "key": "wake_time",
        "vals": ["5:30 AM", "6:00 AM", "6:45 AM", "7:15 AM",
                 "5:45 AM"],
        "old": "My alarm goes off at {a} every weekday.",
        "new": "I've shifted my wake-up — alarm's at {b} now.",
        "queries": ["What time do I wake up?",
                    "When does my alarm go off now?"],
        "answer": "{b}",
    },
    {
        "key": "laptop",
        "vals": ["a Corvidbook Pro", "a Wren 14", "a Fieldwork X1",
                 "a Lumen Slate", "a Driftware 15"],
        "old": "My daily driver is {a}.",
        "new": "New machine day — I upgraded to {b}.",
        "queries": ["What laptop do I use?",
                    "What's my current laptop?"],
        "answer": "{b}",
    },
    {
        "key": "dining_budget",
        "vals": ["$150", "$200", "$250", "$300", "$120"],
        "old": "My dining-out budget is {a} a month.",
        "new": "Tightening up — the dining budget is now {b} a month.",
        "queries": ["What's my monthly dining budget?",
                    "How much do I budget for dining out?"],
        "answer": "{b}",
    },
    {
        "key": "race_goal",
        "vals": ["a 10K", "a half marathon", "a full marathon",
                 "a sprint triathlon"],
        "old": "I'm training for {a} this fall.",
        "new": "Bumped the goal up — now I'm training for {b}.",
        "queries": ["What race am I training for?",
                    "What's my current race goal?"],
        "answer": "{b}",
    },
    {
        "key": "hair",
        "vals": ["dark brown", "auburn", "jet black", "copper red"],
        "old": "My hair is {a} right now.",
        "new": "Impulse decision — dyed my hair {b} yesterday.",
        "queries": ["What color is my hair?",
                    "What's my hair color now?"],
        "answer": "{b}",
    },
    {
        "key": "music_service",
        "vals": ["Tunewire", "Melodia", "Ampfield", "Chordbox"],
        "old": "I subscribe to {a} for music.",
        "new": "I canceled {a} and moved to {b} for music.",
        "queries": ["Which music service do I use?",
                    "What music subscription do I have now?"],
        "answer": "{b}",
    },
)

# ---------------------------------------------------------------------------
# single_session_preference — a stated preference applied to a new
# request. ``check`` is the substring guaranteed unique to the planted
# turn (kept distinct from ``value`` where the value alone is too
# generic, e.g. "in the morning" vs. fillers' "this morning").
# ---------------------------------------------------------------------------

_SSP_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "key": "airplane_seat", "value": "window seat",
        "check": "window seat",
        "text": "I always prefer a window seat when I fly.",
        "queries": ["Help me pick a seat for my upcoming flight.",
                    "I'm booking a flight — which seat should I "
                    "choose?"],
        "answer": "window seat",
    },
    {
        "key": "coffee_order", "value": "large oat-milk latte",
        "check": "oat-milk latte",
        "text": "My go-to coffee order is a large oat-milk latte.",
        "queries": ["I'm grabbing coffee — what should I order?",
                    "What do I usually get at the coffee shop?"],
        "answer": "a large oat-milk latte",
    },
    {
        "key": "notifications", "value": "email",
        "check": "by email",
        "text": "I prefer getting updates by email instead of text "
                "messages.",
        "queries": ["How should the clinic send me appointment "
                    "reminders?",
                    "Which notification channel should I pick?"],
        "answer": "by email",
    },
    {
        "key": "workout_time", "value": "morning",
        "check": "in the morning",
        "text": "I like exercising first thing in the morning.",
        "queries": ["When should I schedule my workouts?",
                    "What's the best time for me to exercise?"],
        "answer": "in the morning",
    },
    {
        "key": "diet", "value": "vegetarian",
        "check": "vegetarian",
        "text": "I eat vegetarian most days of the week.",
        "queries": ["Suggest a dinner recipe for tonight.",
                    "What kind of restaurant should I pick for "
                    "dinner?"],
        "answer": "vegetarian",
    },
    {
        "key": "temp_units", "value": "Celsius",
        "check": "Celsius",
        "text": "Please give me temperatures in Celsius.",
        "queries": ["What's the weather going to be like this "
                    "weekend?",
                    "How hot will it get tomorrow?"],
        "answer": "in Celsius",
    },
    {
        "key": "meeting_time", "value": "after 10am",
        "check": "after 10am",
        "text": "I prefer meetings after 10am — I'm useless before "
                "that.",
        "queries": ["When should I book the team sync?",
                    "What time works best for my meetings?"],
        "answer": "after 10am",
    },
    {
        "key": "music_work", "value": "instrumental",
        "check": "instrumental",
        "text": "I prefer instrumental music while I'm working.",
        "queries": ["Put together a work playlist for me.",
                    "What music should I play while I work?"],
        "answer": "instrumental",
    },
    {
        "key": "hotel", "value": "a hotel with a gym",
        "check": "hotel with a gym",
        "text": "When I travel I always pick a hotel with a gym.",
        "queries": ["Which hotel should I book for the conference?",
                    "What should I look for in a hotel?"],
        "answer": "one with a gym",
    },
    {
        "key": "followup", "value": "a quick call",
        "check": "quick call",
        "text": "I'd rather do a quick call than a long email "
                "thread.",
        "queries": ["How should I follow up with the contractor?",
                    "What's the best way to check in with the "
                    "designer?"],
        "answer": "with a quick call",
    },
    {
        "key": "budget_dining", "value": "budget-friendly",
        "check": "budget-friendly",
        "text": "I prefer budget-friendly places when eating out.",
        "queries": ["Pick a spot for lunch tomorrow.",
                    "Where should I grab lunch?"],
        "answer": "budget-friendly",
    },
    {
        "key": "reading_format", "value": "physical books",
        "check": "physical books",
        "text": "I prefer physical books over e-readers.",
        "queries": ["Should I get the ebook or the paperback?",
                    "Which format should I buy the novel in?"],
        "answer": "physical book / paperback",
    },
)

# ---------------------------------------------------------------------------
# multi_session_user — profile/preference facets, one planted per
# session across >= 3 distinct sessions. ``slots`` are substring tokens
# that appear literally in their facet text (checked case-insensitively)
# so the gold set is self-verifying.
# ---------------------------------------------------------------------------

_MSU_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "key": "dietary_profile",
        "queries": ["What are my dietary restrictions and "
                    "preferences?",
                    "Summarize my diet preferences."],
        "slots": ["vegetarian", "dairy", "sodium", "tree nuts"],
        "facets": {
            "vegetarian": "I've been mostly vegetarian for about "
                          "five years now.",
            "dairy": "Dairy doesn't agree with me, so I skip milk "
                     "and cream.",
            "sodium": "I keep my sodium low — doctor's orders.",
            "tree nuts": "I'm mildly allergic to tree nuts, so I "
                         "avoid them.",
        },
    },
    {
        "key": "running_profile",
        "queries": ["Summarize my running routine.",
                    "What do you know about my running habits?"],
        "slots": ["three times", "Saturday", "9:30", "trail"],
        "facets": {
            "three times": "I usually run three times a week.",
            "Saturday": "Saturday mornings are my long-run days.",
            "9:30": "My comfortable pace is around 9:30 per mile.",
            "trail": "I prefer trail runs over road running.",
        },
    },
    {
        "key": "work_setup",
        "queries": ["Describe my work setup.",
                    "What's my work arrangement?"],
        "slots": ["remotely", "time zones", "platform"],
        "facets": {
            "remotely": "I work remotely three days a week.",
            "time zones": "My team is spread across four time "
                          "zones.",
            "platform": "I lead the platform group at my company.",
        },
    },
    {
        "key": "family",
        "queries": ["What do you know about my family?",
                    "Tell me about my family."],
        "slots": ["sister", "kids", "dog"],
        "facets": {
            "sister": "My sister Maya lives in Chicago.",
            "kids": "I have two kids — a seven-year-old and a "
                    "toddler.",
            "dog": "Our dog Rex is part husky.",
        },
    },
    {
        "key": "travel_style",
        "queries": ["What are my travel habits?",
                    "How do I usually travel?"],
        "slots": ["carry-on", "refundable", "airport"],
        "facets": {
            "carry-on": "I pack light — carry-on only, every trip.",
            "refundable": "I always book refundable tickets just "
                          "in case.",
            "airport": "I get to the airport way early; crowds "
                       "stress me out.",
        },
    },
    {
        "key": "music_hobby",
        "queries": ["What do you know about my music hobby?",
                    "Describe my guitar playing."],
        "slots": ["guitar", "fingerstyle", "jam"],
        "facets": {
            "guitar": "I play guitar most evenings after dinner.",
            "fingerstyle": "Lately I've been practicing fingerstyle "
                           "arrangements.",
            "jam": "Sunday evenings I jam with two friends from "
                   "work.",
        },
    },
)

# ---------------------------------------------------------------------------
# abstention (_abs) — queries the haystack cannot answer. ``topic`` is a
# token guaranteed absent from every turn; ``near`` is a topically
# adjacent lure planted in ~half of the items.
# ---------------------------------------------------------------------------

_ABS_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "topic": "sailboat",
        "q": "What's my sailboat's name?",
        "near": "I spent Saturday kayaking on the lake.",
    },
    {
        "topic": "dentist",
        "q": "When is my next dentist appointment?",
        "near": "I need to book a haircut this week.",
    },
    {
        "topic": "stamp collection",
        "q": "What did I tell you about my stamp collection?",
        "near": "I've been reorganizing my bookshelf lately.",
    },
    {
        "topic": "roommate",
        "q": "What's my college roommate's name?",
        "near": "I ran into an old classmate at the store.",
    },
    {
        "topic": "sushi",
        "q": "What's my favorite sushi place?",
        "near": "I made pasta for dinner last night.",
    },
    {
        "topic": "confirmation number",
        "q": "What's my flight confirmation number?",
        "near": "I should double-check my flight times.",
    },
    {
        "topic": "locker",
        "q": "What's my gym locker combination?",
        "near": "I forgot my water bottle at the gym.",
    },
    {
        "topic": "passport",
        "q": "When does my passport expire?",
        "near": "I need to renew my driver's license soon.",
    },
    {
        "topic": "cousin",
        "q": "Where does my cousin live?",
        "near": "My neighbor just moved in from out of state.",
    },
    {
        "topic": "piano teacher",
        "q": "What's my piano teacher's phone number?",
        "near": "My neighbor plays the violin beautifully.",
    },
    {
        "topic": "wedding gift",
        "q": "What wedding gift did I say I'd bring?",
        "near": "I have a birthday party to shop for.",
    },
    {
        "topic": "wifi password",
        "q": "What's my wifi password?",
        "near": "My router needed a restart this morning.",
    },
)

# ---------------------------------------------------------------------------
# generation machinery
# ---------------------------------------------------------------------------


def _plant(ctx: Dict[str, Any], si: int, speaker: str, text: str
           ) -> Tuple[int, int]:
    """Queue a planted turn for session ``si``; returns its handle."""
    plants = ctx["plants"].setdefault(si, [])
    plants.append((speaker, text))
    return (si, len(plants) - 1)


def _make_sessions(rng: random.Random, qid: str,
                   dates: List[datetime.date],
                   plants: Dict[int, List[Tuple[str, str]]],
                   lo: int, hi: int
                   ) -> Tuple[List[Dict[str, Any]],
                              Dict[Tuple[int, int], Tuple[str, str]]]:
    """Materialize sessions: filler pair blocks interleaved with planted
    turns. Returns (sessions, refmap) where ``refmap[(si, k)]`` is the
    planted turn's ``(session_id, turn_id)``."""
    sessions: List[Dict[str, Any]] = []
    refmap: Dict[Tuple[int, int], Tuple[str, str]] = {}
    for si, d in enumerate(dates):
        sid = f"{qid}-s{si:02d}"
        npairs = rng.randint(lo, hi)
        pairs = rng.sample(list(_FILLERS), npairs)
        blocks: List[Tuple[str, Any]] = [("pair", p) for p in pairs]
        for k, (speaker, text) in enumerate(plants.get(si, [])):
            pos = rng.randint(0, len(blocks))
            blocks.insert(pos, ("plant", (speaker, text, k)))
        specs: List[Tuple[str, str, Optional[int]]] = []
        for kind, blk in blocks:
            if kind == "pair":
                u_text, a_text = blk
                specs.append(("user", u_text, None))
                specs.append(("assistant", a_text, None))
            else:
                speaker, text, k = blk
                specs.append((speaker, text, k))
        hour = rng.randint(7, 20)
        minute = rng.randint(0, 59)
        started_us = _us(d, hour, minute)
        t_us = started_us
        turns: List[Dict[str, Any]] = []
        for ti, (speaker, text, k) in enumerate(specs):
            t_us += rng.randint(45, 240) * 1_000_000
            tid = f"{sid}-t{ti:02d}"
            turns.append({
                "turn_id": tid,
                "speaker": speaker,
                "text": text,
                "timestamp_us": t_us,
            })
            if k is not None:
                refmap[(si, k)] = (sid, tid)
        sessions.append({
            "session_id": sid,
            "date": d.isoformat(),
            "started_us": started_us,
            "turns": turns,
        })
    return sessions, refmap


def _b_information_extraction(rng: random.Random,
                              ctx: Dict[str, Any]) -> None:
    # ~60/40 user/assistant mirrors the LME single-session split.
    pool = (_IE_USER_FACTS if rng.random() < 0.6
            else _IE_ASSISTANT_FACTS)
    fact = rng.choice(pool)
    fmt = {k: rng.choice(v) for k, v in fact["params"].items()}
    si = rng.randrange(len(ctx["dates"]))
    fmt["sdate"] = _fmt_day(ctx["dates"][si])
    h = _plant(ctx, si, fact["speaker"], fact["text"].format(**fmt))
    ctx["query"] = rng.choice(fact["queries"]).format(**fmt)
    ctx["answer"] = fact["answer"].format(**fmt)
    ctx["gold_handles"] = [h]
    ctx["subtype"] = ("single_session_user"
                      if fact["speaker"] == "user"
                      else "single_session_assistant")
    ctx["checks"]["evidence_substrings"] = [ctx["answer"]]


def _b_multi_session(rng: random.Random, ctx: Dict[str, Any]) -> None:
    act = rng.choice(_MS_ACTIVITIES)
    n_sessions = len(ctx["dates"])
    k = min(rng.choice([2, 3, 4]), n_sessions, len(act["places"]))
    sis = sorted(rng.sample(range(n_sessions), k))
    items = rng.sample(act["places"], k)
    handles = []
    for si, item in zip(sis, items):
        fmt = {"item": item, "sdate": _fmt_day(ctx["dates"][si])}
        for ek, ev in act.get("extra", {}).items():
            fmt[ek] = rng.choice(ev)
        handles.append(
            _plant(ctx, si, "user", act["turn"].format(**fmt)))
    if act.get("q_list") and rng.random() < 0.5:
        ctx["query"] = act["q_list"]
        if act.get("list_answer") == "dates":
            ctx["answer"] = ", ".join(
                _fmt_day(ctx["dates"][si]) for si in sis)
        else:
            ctx["answer"] = ", ".join(str(v) for v in items)
        ctx["subtype"] = "multi_session_list"
    else:
        ctx["query"] = rng.choice(act["q_count"])
        ctx["answer"] = str(k)
        ctx["subtype"] = "multi_session_count"
    ctx["gold_handles"] = handles
    ctx["checks"]["count_token"] = act["token"]
    ctx["checks"]["expected_count"] = k


def _b_temporal_reasoning(rng: random.Random,
                          ctx: Dict[str, Any]) -> None:
    spec = rng.choice(_TR_SPECS)
    mode = spec["mode"]
    fmt = {"thing": rng.choice(spec["thing"])}
    si = rng.randrange(len(ctx["dates"]))
    if mode in ("days_ago", "weeks_ago"):
        edate = ctx["dates"][si]  # event on the planting session's date
    else:
        edate = ctx["qdate"] + datetime.timedelta(
            days=rng.randint(3, 30))
    fmt["edate"] = _fmt_day(edate)
    h = _plant(ctx, si, "user", spec["turn"].format(**fmt))
    delta = (ctx["qdate"] - edate).days
    if mode == "days_ago":
        answer = f"{delta} days"
    elif mode == "weeks_ago":
        answer = f"about {delta // 7} weeks"
    elif mode == "days_until":
        answer = f"{-delta} days"
    else:  # weekday
        answer = _WEEKDAYS[edate.weekday()]
    ctx["query"] = rng.choice(spec["queries"]).format(**fmt)
    ctx["answer"] = answer
    ctx["subtype"] = f"temporal_{mode}"
    ctx["gold_handles"] = [h]
    ctx["checks"]["evidence_substrings"] = [_fmt_day(edate)]
    ctx["temporal_pending"] = {
        "mode": mode,
        "event_date": edate.isoformat(),
        "question_date": ctx["qdate"].isoformat(),
        "days": abs(delta),
        "weekday": _WEEKDAYS[edate.weekday()],
    }


def _b_knowledge_updates(rng: random.Random,
                         ctx: Dict[str, Any]) -> None:
    spec = rng.choice(_KU_SPECS)
    a, b = rng.sample(spec["vals"], 2)
    si_old, si_new = sorted(
        rng.sample(range(len(ctx["dates"])), 2))
    fmt = {
        "a": a, "b": b,
        "odate": _fmt_day(ctx["dates"][si_old]),
        "ndate": _fmt_day(ctx["dates"][si_new]),
    }
    h_old = _plant(ctx, si_old, "user", spec["old"].format(**fmt))
    h_new = _plant(ctx, si_new, "user", spec["new"].format(**fmt))
    ctx["query"] = rng.choice(spec["queries"])
    ctx["answer"] = spec["answer"].format(**fmt)
    ctx["gold_handles"] = [h_old, h_new]
    ctx["ku_pending"] = {
        "key": spec["key"], "a": a, "b": b,
        "h_old": h_old, "h_new": h_new,
        "old_date": ctx["dates"][si_old].isoformat(),
        "new_date": ctx["dates"][si_new].isoformat(),
    }


def _b_single_session_preference(rng: random.Random,
                                 ctx: Dict[str, Any]) -> None:
    spec = rng.choice(_SSP_SPECS)
    si = rng.randrange(len(ctx["dates"]))
    h = _plant(ctx, si, "user", spec["text"])
    ctx["query"] = rng.choice(spec["queries"])
    ctx["answer"] = spec["answer"]
    ctx["gold_handles"] = [h]
    ctx["checks"]["evidence_substrings"] = [spec["check"]]
    ctx["pref_pending"] = {
        "mode": "single_session",
        "key": spec["key"],
        "values": [spec["value"]],
        "handles": [h],
    }


def _b_multi_session_user(rng: random.Random,
                          ctx: Dict[str, Any]) -> None:
    spec = rng.choice(_MSU_SPECS)
    slots = list(spec["slots"])
    rng.shuffle(slots)
    if len(slots) > 3:
        slots = slots[:3]  # keep >=3 facets; cap for session pressure
    sis = sorted(rng.sample(range(len(ctx["dates"])), len(slots)))
    handles = [
        _plant(ctx, si, "user", spec["facets"][slot])
        for si, slot in zip(sis, slots)
    ]
    ctx["query"] = rng.choice(spec["queries"])
    ctx["answer"] = ", ".join(slots)
    ctx["gold_handles"] = list(handles)
    ctx["checks"]["evidence_substrings"] = list(slots)
    ctx["pref_pending"] = {
        "mode": "multi_session",
        "key": spec["key"],
        "values": list(slots),
        "handles": list(handles),
    }


def _b_abstention(rng: random.Random, ctx: Dict[str, Any]) -> None:
    spec = rng.choice(_ABS_SPECS)
    if rng.random() < 0.5:
        si = rng.randrange(len(ctx["dates"]))
        _plant(ctx, si, "user", spec["near"])
        ctx["checks"]["near_miss"] = True
    else:
        ctx["checks"]["near_miss"] = False
    ctx["query"] = spec["q"]
    ctx["answer"] = None
    ctx["answerable"] = False
    ctx["expected_abstain"] = True
    ctx["gold_handles"] = []
    ctx["checks"]["absent_topic"] = spec["topic"]


_BUILDERS = {
    CATEGORY_IE: _b_information_extraction,
    CATEGORY_MS: _b_multi_session,
    CATEGORY_TR: _b_temporal_reasoning,
    CATEGORY_KU: _b_knowledge_updates,
    CATEGORY_SSP: _b_single_session_preference,
    CATEGORY_MSU: _b_multi_session_user,
    CATEGORY_ABS: _b_abstention,
}


def _finalize(ctx: Dict[str, Any], sessions: List[Dict[str, Any]],
              refmap: Dict[Tuple[int, int], Tuple[str, str]]
              ) -> Dict[str, Any]:
    """Resolve plant handles into gold refs and pending annotations."""
    order = {s["session_id"]: i for i, s in enumerate(sessions)}
    by_session: Dict[str, List[str]] = {}
    for h in ctx["gold_handles"]:
        sid, tid = refmap[h]
        by_session.setdefault(sid, []).append(tid)
    gold = [
        {"session_id": sid, "turn_ids": tids}
        for sid, tids in sorted(
            by_session.items(), key=lambda kv: order[kv[0]])
    ]

    knowledge_update = None
    if ctx.get("ku_pending"):
        ku = ctx["ku_pending"]
        osid, otid = refmap[ku["h_old"]]
        nsid, ntid = refmap[ku["h_new"]]
        knowledge_update = {
            "key": ku["key"],
            "old_value": ku["a"],
            "new_value": ku["b"],
            "old_session_id": osid,
            "old_turn_id": otid,
            "old_date": ku["old_date"],
            "new_session_id": nsid,
            "new_turn_id": ntid,
            "new_date": ku["new_date"],
        }

    preference = None
    if ctx.get("pref_pending"):
        p = ctx["pref_pending"]
        preference = {
            "mode": p["mode"],
            "key": p["key"],
            "values": p["values"],
            "stated": [
                {"session_id": refmap[h][0], "turn_id": refmap[h][1]}
                for h in p["handles"]
            ],
        }

    return {
        "id": ctx["qid"],
        "category": ctx["cat"],
        "subtype": ctx["subtype"],
        "query": ctx["query"],
        "question_date": ctx["qdate"].isoformat(),
        "question_time_us": _us(ctx["qdate"], 12, 0),
        "answer": ctx["answer"],
        "answerable": ctx["answerable"],
        "expected_abstain": ctx["expected_abstain"],
        "gold_evidence": gold,
        "gold_session_ids": [g["session_id"] for g in gold],
        "gold_turn_ids": [t for g in gold for t in g["turn_ids"]],
        "preference": preference,
        "knowledge_update": knowledge_update,
        "temporal": ctx.get("temporal_pending"),
        "checks": ctx["checks"],
        "haystack_dates": [s["date"] for s in sessions],
        "sessions": sessions,
    }


def _build_question(rng: random.Random, cat: str, idx: int,
                    n_sessions: int, lo: int, hi: int
                    ) -> Dict[str, Any]:
    qid = f"lme-{idx:04d}"
    qdate = _BASE_DATE + datetime.timedelta(days=rng.randint(0, 90))
    offsets = sorted(
        rng.sample(range(_MIN_OFFSET_DAYS, _MAX_OFFSET_DAYS + 1),
                   n_sessions),
        reverse=True,
    )
    dates = [qdate - datetime.timedelta(days=o) for o in offsets]
    ctx: Dict[str, Any] = {
        "qid": qid,
        "cat": cat,
        "qdate": qdate,
        "dates": dates,
        "plants": {},
        "query": None,
        "answer": None,
        "answerable": True,
        "expected_abstain": False,
        "subtype": cat,
        "gold_handles": [],
        "preference": None,
        "temporal_pending": None,
        "checks": {},
    }
    _BUILDERS[cat](rng, ctx)
    sessions, refmap = _make_sessions(rng, qid, dates,
                                      ctx["plants"], lo, hi)
    return _finalize(ctx, sessions, refmap)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def generate(seed: int = DEFAULT_SEED,
             n_questions: int = DEFAULT_N,
             sessions_per_user: int = DEFAULT_SESSIONS,
             min_filler_pairs: int = 2,
             max_filler_pairs: int = 5) -> Dict[str, Any]:
    """Deterministic LongMemEval-S twin corpus.

    Parameters
    ----------
    seed
        Drives the whole corpus — identical seeds produce
        byte-identical output.
    n_questions
        Question count. At the default 500 each category weight yields
        exactly ``5 * w`` questions (105/115/115/80/30/30/25).
    sessions_per_user
        Dated sessions per question haystack (4..76). LME-S runs ~40–50;
        the default 10 keeps the owned twin CI-fast while retaining the
        multi-session structure the categories exercise.
    min_filler_pairs, max_filler_pairs
        Per-session filler exchange bounds (each pair = user turn +
        assistant turn). Planted evidence is interleaved among them.
    """
    if n_questions < 1:
        raise ValueError("n_questions must be >= 1")
    if not (MIN_SESSIONS <= sessions_per_user <= MAX_SESSIONS):
        raise ValueError(
            f"sessions_per_user must be in "
            f"[{MIN_SESSIONS}, {MAX_SESSIONS}]"
        )
    if min_filler_pairs < 1 or max_filler_pairs < min_filler_pairs:
        raise ValueError(
            "require 1 <= min_filler_pairs <= max_filler_pairs")
    if max_filler_pairs > len(_FILLERS):
        raise ValueError(
            f"max_filler_pairs <= {len(_FILLERS)} (filler pool size)")

    rng = random.Random(seed)
    bag: List[str] = [
        cat for cat, w in CATEGORY_WEIGHTS.items() for _ in range(w)
    ]
    rng.shuffle(bag)

    questions = []
    for i in range(n_questions):
        cat = bag[i % len(bag)]
        qrng = random.Random(rng.getrandbits(64))
        questions.append(
            _build_question(qrng, cat, i, sessions_per_user,
                            min_filler_pairs, max_filler_pairs))

    counts = {c: 0 for c in CATEGORIES}
    for q in questions:
        counts[q["category"]] += 1

    return {
        "corpus": CORPUS_NAME,
        "generator": GENERATOR_ID,
        "seed": seed,
        "params": {
            "n_questions": n_questions,
            "sessions_per_user": sessions_per_user,
            "min_filler_pairs": min_filler_pairs,
            "max_filler_pairs": max_filler_pairs,
            "base_date": _BASE_DATE.isoformat(),
        },
        "category_counts": counts,
        "questions": questions,
    }


def to_jsonl(corpus: Dict[str, Any]) -> str:
    """Canonical serialization — sorted keys, byte-deterministic."""
    meta = {k: corpus[k] for k in
            ("corpus", "generator", "seed", "params",
             "category_counts")}
    lines = [
        json.dumps({"_meta": meta}, sort_keys=True,
                   separators=(",", ":"))
    ]
    for q in corpus["questions"]:
        lines.append(json.dumps(q, sort_keys=True,
                                separators=(",", ":")))
    return "\n".join(lines) + "\n"


def corpus_digest(corpus: Dict[str, Any]) -> str:
    """sha256 over the canonical serialization (manifest identity)."""
    return hashlib.sha256(
        to_jsonl(corpus).encode("utf-8")).hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="question count")
    ap.add_argument("--sessions", type=int,
                    default=DEFAULT_SESSIONS,
                    help="dated sessions per question haystack")
    ap.add_argument("--out", default=None,
                    help="write JSONL here (default: stdout)")
    args = ap.parse_args(argv)
    corpus = generate(seed=args.seed, n_questions=args.n,
                      sessions_per_user=args.sessions)
    blob = to_jsonl(corpus)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(blob)
    else:
        sys.stdout.write(blob)
    print(
        f"{CORPUS_NAME} ({GENERATOR_ID}) seed={args.seed} "
        f"n={args.n} sessions={args.sessions} "
        f"digest={corpus_digest(corpus)}",
        file=sys.stderr,
    )
    for cat, n in corpus["category_counts"].items():
        print(f"  {cat}: {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
