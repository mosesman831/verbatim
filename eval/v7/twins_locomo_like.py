"""Owned LoCoMo-like twin corpus generator (V7-22.05, ``owned_locomo_like``).

Deterministic, seeded generator of a dialogue corpus that mirrors the LoCoMo
task categories — single-hop, multi-hop, temporal, open-domain, and an
abstention slice covering LoCoMo's adversarial category (cat 5: false
premises, usually speaker mis-attribution, per §24.1 / V7-24.01) — using
100% invented entities.  No benchmark text, names, questions, or answer
strings appear here; every name, place, hobby, date, and fact comes from
the owned word lists below.

Structure mirrors the released LoCoMo shape:

* two speakers (roles ``user`` / ``assistant``) holding dated sessions of
  speaker turns spread over months;
* facts planted inside specific turns so every question's gold evidence is
  known exactly (LoCoMo-style ``D<session>:<turn>`` ids);
* question records carry ``qid``, ``category``, ``subtype``, ``query``,
  ``answer``, ``answerable``, ``evidence`` (gold turn ids — empty for
  abstain items), and ``question_time``.

Categories and their attack surfaces (§24.1):

* ``single_hop``   — the answer is stated outright in one turn.
* ``multi_hop``    — the answer needs >= 2 turns, always across sessions:
  shared hobby/city/routine, shop->street attribute chains,
  festival-city->relative chains, revisit chains, and who-ran-X-first
  comparisons.
* ``temporal``     — when / ordering / duration questions.  Evidence uses
  explicit dates and relative expressions ("yesterday", "last week",
  "two months ago", "last Tuesday") resolved against the turn's own
  session date (LoCoMo answers are session-date-relative, V7-24.03), plus
  "on what date did X mention Y" session-date questions.
* ``open_domain``  — paraphrased / indirect questions whose evidence turn
  uses deliberately different wording (the hard lexical slice).
* ``abstain``      — ``never_discussed`` (topic nowhere in the corpus),
  ``speaker_mismatch`` (fact exists but was said by the *other* speaker —
  the adversarial twin), and ``out_of_timeline`` (dates decades before the
  dialogue).  All carry ``evidence: []`` and ``answerable: False``.

Ambiguity control: value pools enforce uniqueness where a question keys on
the value alone (e.g. festival names are global; shops are unique per
subject; "visited" cities are globally unique so "which city have both
speakers visited" has exactly one supporting pair).  A final verification
pass re-checks every abstain topic is absent and every session-date mention
is unique, and swaps or drops the rare offender.

Byte-determinism: every choice flows from one ``random.Random(seed)`` in
fixed program order; sets are membership-tested but never iterated; no
``hash()``, no wall clock.  ``to_jsonl`` writes canonical JSON
(``sort_keys``, compact separators) — identical bytes for a fixed seed.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

GENERATOR_ID = "twins_locomo_like/v1"
CONSTANTS_TAG = "provisional/v7-r0"
CORPUS_NAME = "owned_locomo_like"
DEFAULT_SEED = 20260923

CATEGORIES: Tuple[str, ...] = (
    "single_hop",
    "multi_hop",
    "temporal",
    "open_domain",
    "abstain",
)


class _PoolEmpty(Exception):
    """A value pool cannot supply a uniqueness-satisfying draw."""


# ---------------------------------------------------------------------------
# Owned word lists — invented entities only
# ---------------------------------------------------------------------------

FIRST_NAMES: Tuple[str, ...] = (
    "Ansel", "Bryn", "Callum", "Doria", "Eamon", "Fia", "Gideon", "Hattie",
    "Ivo", "Junia", "Kellan", "Lior", "Maren", "Nils", "Odette", "Piers",
    "Quilla", "Ronan", "Sable", "Thayer", "Uma", "Vada", "Wren", "Xiomara",
    "Yannick", "Zora", "Marlowe", "Ines", "Tova", "Bram", "Sorrel", "Petra",
)

HOBBIES: Tuple[str, ...] = (
    "pottery", "archery", "beekeeping", "salsa dancing", "bouldering",
    "calligraphy", "birdwatching", "origami", "sourdough baking",
    "kayaking", "watercolor painting", "fencing", "glassblowing",
    "quilting", "stargazing", "woodcarving", "tai chi", "pickleball",
    "needle felting", "mushroom foraging", "rock climbing",
    "model railroading", "geocaching", "embroidery", "roller skating",
    "metal detecting", "basket weaving", "whittling", "soap carving",
    "letterboxing", "kite building", "chainmail weaving",
)

PET_SPECIES: Tuple[str, ...] = (
    "greyhound", "tabby cat", "spaniel", "rabbit", "ferret", "tortoise",
    "parrot", "hedgehog", "beagle", "cockatiel", "pygmy goat", "axolotl",
    "chinchilla", "gecko", "miniature donkey", "hermit crab", "budgerigar",
    "potbellied pig",
)

PET_NAMES: Tuple[str, ...] = (
    "Plover", "Biscuit", "Noodle", "Captain", "Zinnia", "Mochi", "Tilde",
    "Bramble", "Comet", "Fig", "Junip", "Kip", "Lentil", "Maple", "Nori",
    "Orbit", "Pepper", "Quill", "Rye", "Sprout", "Tansy", "Ube", "Vesper",
    "Waffle", "Yucca", "Zephyr", "Acorn", "Basil", "Clover", "Doodle",
)

CITIES: Tuple[str, ...] = (
    "Port Vasco", "Brambleton", "Calder Falls", "Netherton", "Larkspur Bay",
    "Ivenhold", "Marrowgate", "Thessaly Point", "Quillon Harbor",
    "Dunmore Heath", "East Barrow", "Felicity Springs", "Garroway",
    "Hallowick", "Ismere", "Jordans Mill", "Kelp Harbor", "Low Wexley",
    "Millbrook Vale", "Norwick", "Osterley Fen", "Penmark", "Rillwater",
    "Tumbledown", "Vetchfield", "Wexford Hollow", "Yarrow Beach",
    "Zennor Wick", "Ashcombe", "Bellwater", "Cinderford", "Dovecot",
    "Elmswell", "Fernhollow", "Grimsby Docks", "Hartley Vale",
    "Inkberrow", "Jackdaw Point", "Kestrel Bay", "Loxmere", "Mudlark",
    "Nethergate", "Owlshead", "Pinfold", "Quail Run", "Russett",
    "Starbridge", "Thornwick", "Undercliff", "Vesperton", "Whelk Bay",
    "Yewtree", "Amberleigh", "Brightholm", "Coveley", "Driftline",
    "Eelgrass", "Foxglove End", "Gullcrest", "Hollowmere", "Ivystone",
    "Juniper Falls", "Knotweed", "Lambswool", "Mistral Quay", "Newhaven Gap",
    "Otterbourne", "Ploverfield", "Quimby", "Ropewalk", "Shale Cove",
    "Tidewater End", "Ulfston", "Vervain", "Woolgarth", "Yonderlea",
)

STREETS: Tuple[str, ...] = (
    "Bramble Street", "Quilter Lane", "Anchor Road", "Mallow Crescent",
    "Tinker Alley", "Juniper Row", "Cobb Lane", "Alder Court",
    "Ferryman Walk", "Gull Street", "Harrow Lane", "Inkberry Way",
    "Lobster Row", "Millstone Close", "Nettlebed Lane", "Oxcart Road",
    "Puddle Dock", "Ropewalk Street", "Saltern Way", "Thimble Court",
)

SHOPS: Tuple[str, ...] = (
    "Tinker & Bale", "the Copper Kettle", "Ferro & Sons", "Mallow's",
    "the Odd Crate", "Brightwall Supply", "Pike & Pearl", "the Long Shelf",
    "Sunderland's", "Wicklow Exchange", "the Brass Button", "Nettle & Co.",
    "Fathom Books", "the Salted Rim", "Cordwainer Hall", "Moss & Bone",
    "the Tin Whistle", "Aldercraft", "Glimmer & Sons", "the Dusty Ledger",
    "Pinchbeck's", "the Woolen Sail", "Harrier's", "Quince & Quill",
    "the Bent Anchor", "Stonecrop Market", "Lark & Loom", "Featherbed Lane",
)

ITEMS: Tuple[str, ...] = (
    "upright piano", "gravel bike", "canoe", "wood lathe",
    "espresso machine", "brass telescope", "spinning wheel", "rowboat",
    "violin", "ceramic kiln", "oak dresser", "typewriter", "butter churn",
    "field accordion", "drafting table", "antique globe", "chess clock",
    "cedar chest", "hand truck", "wormery", "storm lantern", "ice axe",
    "letterpress", "mandolin", "orchid press", "seed drill", "tide clock",
    "winnowing basket", "yurt kit", "zephyr vane",
)

LANDMARKS: Tuple[str, ...] = (
    "the old lighthouse", "the tide pools", "the bell tower",
    "the fossil cliffs", "the botanical garden", "the night market",
    "the observatory", "the ferry museum", "the canyon overlook",
    "the lavender farm", "the grain mill", "the salt baths",
    "the hedge maze", "the rope bridge", "the marble quarry",
    "the clock museum", "the dunes boardwalk", "the slate caverns",
    "the orchid house", "the ruins on the hill", "the canal locks",
    "the windmill museum", "the glassworks", "the cormorant rookery",
    "the seaweed baths", "the signal station", "the deer park",
    "the spring grotto", "the ice house", "the arboretum",
)

FESTIVALS: Tuple[str, ...] = (
    "the Lantern Tide Festival", "the Harvest Moon Regatta",
    "the Winterlight Parade", "the Ember Days Fair", "the Frost Fair",
    "the Marigold Market", "the Starfall Music Weekend",
    "the Coracle Races", "the Kite Festival", "the Orchard Blossom Walk",
    "the Driftwood Bonfire", "the Salt Harvest Feast",
    "the Foghorn Concert", "the Equinox Swim", "the Scarecrow Parade",
    "the Wool Fair", "the Currach Cup", " the Bellringers' Meet".strip(),
    "the Candlelit Regatta", "the Bracken Ball", " the Periwinkle Pageant".strip(),
    "the Longest Day Picnic", " the Shoal Run".strip(),
    "the Barometer Ball", "the Combe Fair", "the Fog Festival",
    "the Gullwing Games", "the Herring Supper", "the Ivy Crown Fete",
    "the Jubilee Bonfire", "the Kettle Brine Contest", "the Lantern Ebb",
)

RACES: Tuple[str, ...] = (
    "marathon", "half marathon", "10K", "trail race", "triathlon",
    "duathlon", "ultramarathon", "relay race", "fun run",
    "obstacle race", "cross-country race", "sprint triathlon",
    "night race", "charity run", "parkrun", "fell race", "swim-run",
    "aquathlon", "tower climb", "sand race",
)

RELATIONS: Tuple[str, ...] = (
    "sister", "brother", "cousin", "aunt", "uncle", "niece", "nephew",
    "godmother", "grandmother", "college roommate", "sister-in-law",
    "godfather", "stepsister", "great-aunt",
)

ALLERGENS: Tuple[str, ...] = (
    "shellfish", "kiwi", "walnuts", "sesame", "strawberries", "pineapple",
    "bee stings", "penicillin", "lavender", "nickel", "hazelnuts",
    "dust mites", "peaches", "mustard",
)

JOBS: Tuple[str, ...] = (
    "audio archivist", "ferry scheduler", "botanical illustrator",
    "clockmaker", "pastry instructor", "wildlife surveyor",
    "lighthouse keeper", "court stenographer", "violin luthier",
    "tea blender", "map engraver", "aquarium curator",
)

EMPLOYERS: Tuple[str, ...] = (
    "the Maritime Archive", "Kestrel & Vine Bakery", "the North Pier Trust",
    "Hollis Conservatory", "the Tideworks Cooperative", "Bramble & Fenn",
    "the Glasshouse Museum", "Saltline Radio", "the Herbarium",
    "Port Authority Annex", "the Clocktower Guild", "Mirehouse Press",
)

BOOKS: Tuple[Tuple[str, str], ...] = (
    ("The Salt Cartographer", "Iris Fenwick"),
    ("Winter Arithmetic", "Pavel Ost"),
    ("A Field Guide to Vanishing", "Marta Quell"),
    ("The Clockwork Orchard", "Tamsin Gale"),
    ("Lanterns Over Stillwater", "Ezra Hollow"),
    ("The Beekeeper's Ledger", "Junia Peale"),
    ("Maps for Lost Harbors", "Osric Vane"),
    ("The Upholsterer's Daughter", "Petra Lindqvist"),
    ("Signal Fires", "Alder Crane"),
    ("Nine Kinds of Rain", "Wren Solace"),
    ("The Cartwright Inheritance", "Mabel Quinn"),
    ("Echoes in the Granary", "Theo Marchetti"),
    ("The Bone Lantern", "Silas Vetch"),
    ("A Winter of Foxes", "Odile Marsh"),
    ("The Cartographer's Debt", "Pia Romero"),
    ("Tide Lines", "Henrik Voss"),
    ("The Apiary Murders", "Constance Yu"),
    ("Half a Kingdom", "Dmitri Ash"),
    ("The Orchardists", "Ruth Calloway"),
    ("Vespers at Sea", "Lionel Frost"),
)

# (verb_phrase, head_verb, place) — "Every Tuesday morning I <verb_phrase>."
ROUTINES: Tuple[Tuple[str, str, str], ...] = (
    ("jog along the river path", "jog", "the river path"),
    ("swim laps at the quarry pool", "swim", "the quarry pool"),
    ("bike to the market square", "bike", "the market square"),
    ("do tai chi in the park", "do tai chi", "the park"),
    ("walk the dogs on the heath", "walk the dogs", "the heath"),
    ("row on the lake", "row", "the lake"),
    ("do yoga at the studio", "do yoga", "the studio"),
    ("sketch at the harbor wall", "sketch", "the harbor wall"),
    ("shoot hoops at the rec court", "shoot hoops", "the rec court"),
    ("birdwatch at the marsh blinds", "birdwatch", "the marsh blinds"),
    ("run stairs at the stadium", "run stairs", "the stadium"),
    ("lift weights at the gym", "lift weights", "the gym"),
    ("stretch on the roof deck", "stretch", "the roof deck"),
    ("swim at the cove", "swim at the cove", "the cove"),
    ("practice kicks at the dojo", "practice kicks", "the dojo"),
    ("walk the seawall", "walk the seawall", "the seawall"),
    ("meditate in the garden", "meditate", "the garden"),
    ("throw pots at the studio", "throw pots", "the pottery studio"),
    ("play chess in the square", "play chess", "the square"),
    ("run the cliff trail", "run the cliff trail", "the cliff trail"),
)

WEEKDAYS: Tuple[str, ...] = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday",
)

# (action_id, past_clause, query_clause) for dated temporal events.
DATED_ACTIONS: Tuple[Tuple[str, str, str], ...] = (
    ("lease", "signed the lease on the workshop",
     "sign the lease on the workshop"),
    ("driving", "passed the driving test", "pass the driving test"),
    ("manuscript", "sent off the manuscript", "send off the manuscript"),
    ("ferry", "bought the ferry tickets", "buy the ferry tickets"),
    ("cast", "got the cast off", "get the cast off"),
    ("orchard", "planted the apple orchard", "plant the apple orchard"),
    ("permit", "filed the permit paperwork", "file the permit paperwork"),
    ("kittens", "adopted the foster kittens", "adopt the foster kittens"),
    ("boat", "started the boat restoration", "start the boat restoration"),
    ("savings", "opened the savings account", "open the savings account"),
    ("resignation", "handed in the resignation letter",
     "hand in the resignation letter"),
    ("cabin", "booked the cabin for autumn", "book the cabin for autumn"),
    ("attic", "finished the attic renovation", "finish the attic renovation"),
    ("window", "ordered the replacement window",
     "order the replacement window"),
    ("passport", "renewed the passport", "renew the passport"),
    ("bees", "set up the second hive", "set up the second hive"),
)

# Relative expressions resolved against the *evidence turn's* session date
# (V7-24.03 session-date-relative).  kind: "days" | "months" | "weekday".
REL_PHRASES: Tuple[Tuple[str, str, int], ...] = (
    ("yesterday", "days", 1),
    ("the day before yesterday", "days", 2),
    ("three days ago", "days", 3),
    ("four days ago", "days", 4),
    ("a week ago", "days", 7),
    ("last week", "days", 7),
    ("two weeks ago", "days", 14),
    ("three weeks ago", "days", 21),
    ("a month ago", "months", 1),
    ("two months ago", "months", 2),
    ("three months ago", "months", 3),
    ("last {weekday}", "weekday", 0),
    ("this past {weekday}", "weekday", 0),
)

# (gerund_phrase, word, n, unit) — "I have been <g> for <word> <unit> now."
DURATIONS: Tuple[Tuple[str, str, int, str], ...] = (
    ("playing the cello", "six", 6, "years"),
    ("studying Italian", "eight", 8, "months"),
    ("volunteering at the shelter", "two", 2, "years"),
    ("taking evening classes", "nine", 9, "months"),
    ("keeping a sketchbook diary", "five", 5, "years"),
    ("running the neighborhood book club", "three", 3, "years"),
    ("coaching the junior swim team", "four", 4, "years"),
    ("restoring the farmhouse", "eighteen", 18, "months"),
    ("writing the newsletter", "seven", 7, "months"),
    ("chairing the harbor committee", "two", 2, "years"),
    ("growing orchids", "six", 6, "months"),
    ("baking for the market stall", "eleven", 11, "months"),
    ("hosting the trivia night", "three", 3, "years"),
    ("maintaining the town archive", "ten", 10, "years"),
)

# Open-domain pairs: evidence phrasing deliberately diverges from question
# phrasing (the hard lexical slice).  ``vars`` lists value tuples so
# instantiations differ in surface; when the var list cycles the evidence
# gains a lead-in so no two evidence turns are byte-identical.
OD_PAIRS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "weekend_project",
        "ev": "I spent all of Saturday elbow-deep in the {v0}, swapping out the {v1}.",
        "q": "What does {n} do to unwind when the work week ends?",
        "ans": "working on the {v0}",
        "mention": "{v1}",
        "vars": [("1971 motorcycle", "carburetor"),
                 ("1968 pickup", "fuel pump"),
                 ("old sailboat", "rigging"),
                 ("derelict rowboat", "transom")],
    },
    {
        "name": "rec_center",
        "ev": "The rec center kids finally beat me at chess, twice in one afternoon.",
        "q": "How does {n} spend time with kids in the neighborhood?",
        "ans": "playing chess at the rec center",
        "mention": "rec center kids",
        "vars": [("",)],
    },
    {
        "name": "volunteer",
        "ev": "I have been shelving donations at the {v0} every {v1}.",
        "q": "How does {n} volunteer in the community?",
        "ans": "shelving donations at the {v0}",
        "mention": "{v0}",
        "vars": [("thrift shop", "Thursday"),
                 ("food pantry", "Tuesday"),
                 ("animal shelter", "Saturday"),
                 ("book depot", "Monday")],
    },
    {
        "name": "heirloom",
        "ev": "My grandmother's recipe box is my treasure. I have cooked through half of it.",
        "q": "What heirloom does {n} value most?",
        "ans": "{n}'s grandmother's recipe box",
        "mention": "recipe box",
        "vars": [("",)],
    },
    {
        "name": "training",
        "ev": "I did {v0} before breakfast this morning. The {v1} will not know what hit it.",
        "q": "What athletic goal is {n} chasing?",
        "ans": "the {v1}",
        "mention": "{v1}",
        "vars": [("sixteen miles", "Saltwind Marathon"),
                 ("twenty kilometers", "Harborlight Triathlon"),
                 ("ten miles", "Fellfoot Trail Race"),
                 ("eighteen miles", "Cinder Path Ultra")],
    },
    {
        "name": "potluck",
        "ev": "We host a porch potluck every first Friday. The neighbors bring the wildest casseroles.",
        "q": "What tradition does {n} keep with their neighbors?",
        "ans": "a monthly porch potluck",
        "mention": "porch potluck",
        "vars": [("",)],
    },
    {
        "name": "pages",
        "ev": "I write three pages by hand before bed, every single night. I never break the rule.",
        "q": "What nightly habit does {n} follow?",
        "ans": "writing three pages by hand",
        "mention": "three pages",
        "vars": [("",)],
    },
    {
        "name": "choir",
        "ev": "The choir rehearses in the church basement. I am the only tenor under sixty.",
        "q": "What musical group does {n} belong to?",
        "ans": "a choir",
        "mention": "choir",
        "vars": [("",)],
    },
    {
        "name": "balcony",
        "ev": "My balcony is basically a farm: tomatoes, chilies, and a fig tree in a bucket.",
        "q": "What does {n} grow at home?",
        "ans": "tomatoes, chilies, and a fig tree",
        "mention": "fig tree",
        "vars": [("",)],
    },
    {
        "name": "penpal",
        "ev": "I write to a pen pal in {v0}, paper, stamps, the whole ritual.",
        "q": "Who does {n} correspond with by mail?",
        "ans": "a pen pal in {v0}",
        "mention": "pen pal",
        "vars": [("Ismere",), ("Norwick",), ("Garroway",), ("Tumbledown",)],
    },
    {
        "name": "nets",
        "ev": "I mend the fishermen's nets down at the dock on weekends for pocket money.",
        "q": "How does {n} earn extra money?",
        "ans": "mending nets at the dock",
        "mention": "nets",
        "vars": [("",)],
    },
    {
        "name": "syrup",
        "ev": "Every year I tap the maples on the north fence and boil syrup in the yard.",
        "q": "What seasonal project does {n} take on?",
        "ans": "boiling maple syrup",
        "mention": "maples",
        "vars": [("",)],
    },
    {
        "name": "radio",
        "ev": "I have been reading bedtime stories on the community radio on Thursdays.",
        "q": "Where can people hear {n}'s voice?",
        "ans": "on community radio",
        "mention": "community radio",
        "vars": [("",)],
    },
    {
        "name": "logbook",
        "ev": "I keep the lighthouse logbook for the historical society.",
        "q": "What does {n} do for the historical society?",
        "ans": "keeps the lighthouse logbook",
        "mention": "logbook",
        "vars": [("",)],
    },
)

# Abstain topics: (substring key, display phrase).  The key MUST NOT appear
# in any generated turn text; the verifier re-checks every use and swaps.
ABSTAIN_TOPICS: Tuple[Tuple[str, str], ...] = (
    ("hot air balloon", "a hot air balloon ride"),
    ("antarctica", "a trip to Antarctica"),
    ("polka", "the county polka championship"),
    ("tarantula", "a pet tarantula"),
    ("home brewery", "their home brewery"),
    ("stock market", "the stock market"),
    ("postage stamps", "vintage postage stamps"),
    ("karaoke", "a karaoke machine"),
    ("iguana", "an iguana"),
    ("pilot's license", "their pilot's license"),
    ("lunar eclipse", "the lunar eclipse viewing party"),
    ("food truck", "a food truck business"),
    ("chess rating", "their chess rating"),
    ("rooftop garden", "a rooftop garden"),
    ("saxophone", "the saxophone"),
    ("compost", "a composting system"),
    ("windsurfing", "windsurfing lessons"),
    ("tattoo", "a new tattoo"),
    ("drone", "a drone"),
    ("monastery", "a monastery retreat"),
    ("ice fishing", "an ice fishing trip"),
    ("paragliding", "paragliding"),
    ("the opera", "the opera"),
    ("bidet", "a bidet"),
    ("vinyl", "a vinyl collection"),
    ("beehive", "a backyard beehive"),
    ("slot canyon", "a slot canyon hike"),
    ("harpsichord", "a harpsichord"),
    ("unicycle", "a unicycle"),
    ("ostrich farm", "an ostrich farm"),
    ("welding", "a welding course"),
    ("igloo", "an igloo-building weekend"),
)

FILLERS: Tuple[str, ...] = (
    "How has your week been treating you?",
    "The weather cannot make up its mind today.",
    "I made the best soup last night, I swear.",
    "My commute was a nightmare this morning.",
    "I am so ready for the weekend.",
    "The garden is finally coming together.",
    "I finished that puzzle, the one with the windmill.",
    "We should plan a hike soon.",
    "The coffee place on the corner changed its name again.",
    "I lost an hour to a recipe rabbit hole.",
    "Traffic was unreal today.",
    "I found a box of old mixtapes in the closet.",
    "The neighbor's rooster is at it again.",
    "I have been meaning to organize the garage.",
    "I finally fixed the squeaky door.",
    "The library book sale is this weekend.",
    "I am thinking about a haircut.",
    "My phone battery dies by lunch every day.",
    "I saw the northern lights forecast for Friday.",
    "The farmers market had the first peaches of the season.",
    "Tell me something good that happened today.",
    "That sounds wonderful, tell me more.",
    "How did that go?",
    "No way, really?",
    "What happened next?",
    "Pictures or it did not happen.",
    "Ha, that is very on brand for you.",
    "I cannot wait to hear how it turns out.",
    "Good for you, honestly.",
    "What a week it has been.",
    "Same old, same old over here.",
    "You always have the best stories.",
    "Let me know how it goes.",
    "I am proud of you for sticking with it.",
    "Wild, I would never have guessed.",
    "Anyway, enough about me.",
    "Oh, before I forget.",
    "How is everything on your end?",
    "I need a nap just thinking about it.",
    "The bus was twenty minutes late again.",
    "I finally tried that new noodle place.",
    "My inbox is a disaster zone.",
    "Caught the sunrise on my walk today.",
)

LEAD_INS: Tuple[str, ...] = (
    "Oh, that reminds me -",
    "Speaking of which,",
    "You know what?",
    "Before I forget,",
    "Funny you should ask,",
    "Guess what!",
    "Small update:",
    "Real talk:",
)

CAPTIONS: Tuple[str, ...] = (
    "a sunset over the harbor", "a pottery kiln mid-firing",
    "a plate of dumplings", "a hand-drawn trail map",
    "a sleeping cat in a sunbeam", "a half-finished quilt",
    "a crowded farmers market stall", "a lighthouse at dusk",
    "a stack of library books", "a muddy pair of boots",
    "a bicycle leaning on a fence", "a bowl of cherries",
)

#: Static turn text — a mention/topic can never collide with these.
_STATIC_TEXTS: Tuple[str, ...] = FILLERS + LEAD_INS + CAPTIONS


def _has_phrase(text: str, phrase: str) -> bool:
    """Case-insensitive word-boundary containment for a literal phrase."""
    return re.search(r"\b" + re.escape(phrase) + r"\b",
                     text, re.IGNORECASE) is not None

#: Restricted families allow at most one fact per subject so undiscriminated
#: questions ("what is X allergic to?") stay unambiguous.
_RESTRICTED = frozenset({"hobby", "job", "allergy"})

#: Families the single-hop generator may draw (subtype == family name).
SH_FAMILIES: Tuple[str, ...] = (
    "hobby", "pet", "job", "purchase", "visit", "visit_planned",
    "relation", "race", "allergy", "routine", "book",
)

MH_SHAPES: Tuple[str, ...] = (
    "shared_hobby", "shared_city", "paired_routine", "compare_race",
    "purchase_chain", "event_attr", "repeat_visit",
)

T_SHAPES: Tuple[str, ...] = (
    "relative", "explicit", "ordering", "duration", "session_date",
)

AB_SHAPES: Tuple[str, ...] = (
    "never_discussed", "speaker_mismatch", "out_of_timeline",
)

_MONTHS: Tuple[str, ...] = (
    "", "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)

_MDAYS = (0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _leap(y: int) -> bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def _shift_months(d: date, months: int) -> date:
    """Calendar-correct month shift with day clamping."""
    total = d.year * 12 + (d.month - 1) + months
    y, m0 = divmod(total, 12)
    m = m0 + 1
    last = _MDAYS[m] + (1 if m == 2 and _leap(y) else 0)
    return date(y, m, min(d.day, last))


def _last_weekday(d: date, weekday: int) -> date:
    """Most recent ``weekday`` strictly before ``d``."""
    delta = (d.weekday() - weekday) % 7
    return d - timedelta(days=delta or 7)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _us(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)


def _date_phrase(d: date) -> str:
    return f"{d.day} {_MONTHS[d.month]} {d.year}"


def corpus_digest(corpus: Dict[str, Any]) -> str:
    canon = {
        "name": corpus.get("name"),
        "generator": corpus.get("generator"),
        "seed": corpus.get("seed"),
        "params": corpus.get("params"),
        "speakers": corpus.get("speakers"),
        "sessions": corpus.get("sessions"),
        "turns": corpus.get("turns"),
        "facts": corpus.get("facts"),
        "questions": corpus.get("questions"),
    }
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class _Builder:
    def __init__(self, seed: int, n_sessions: int, base: date,
                 distractor_rate: float) -> None:
        self.rng = random.Random(seed)
        self.n_sessions = n_sessions
        self.distractor_rate = distractor_rate

        # Two named speakers: a user and an assistant.
        self.user, self.assistant = self.rng.sample(list(FIRST_NAMES), 2)
        self.speakers = (self.user, self.assistant)
        self.role = {self.user: "user", self.assistant: "assistant"}

        # Session timeline: roughly weekly-to-fortnightly over months.
        self.session_starts: List[datetime] = []
        cur = datetime.combine(
            base + timedelta(days=self.rng.randint(0, 6)),
            time(self.rng.randint(9, 18), self.rng.choice((0, 15, 30, 45))),
            tzinfo=timezone.utc,
        )
        for _ in range(n_sessions):
            self.session_starts.append(cur)
            cur += timedelta(
                days=self.rng.randint(7, 16), hours=self.rng.randint(-2, 4)
            )

        self.beats: List[Dict[str, Any]] = []
        self.session_load = [0] * n_sessions
        self.facts: List[Dict[str, Any]] = []
        self.drafts: List[Dict[str, Any]] = []
        self.dated_facts: List[Dict[str, Any]] = []
        self._pools: Dict[str, List[Any]] = {}
        self._subj_family: set = set()
        self._subj_actions: set = set()
        self._used_topics: set = set()
        self._fact_seq = 0
        self._beat_seq = 0

        # --- uniqueness bookkeeping (membership-tested only, never iterated)
        self._visit_cities: set = set()     # cities someone visited
        self._race_cities: set = set()      # cities with a race (may repeat)
        self._rel_cities: set = set()       # cities a relative lives in
        self._shared_cities: set = set()    # cities both speakers visited
        self.went: Dict[str, set] = {self.user: set(), self.assistant: set()}
        self._visit_facts: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._visit_count: Dict[Tuple[str, str], int] = {}
        self._used_items: set = set()       # (subject, item)
        self._used_shops: set = set()       # (subject, shop)
        self._shop_street: Dict[str, str] = {}
        self._used_landmarks: set = set()   # (subject, landmark)
        self._used_festivals: set = set()
        self._used_races: set = set()
        self._used_rel: set = set()         # (subject, relation word)
        self._used_rnames: set = set()      # (subject, relative name)
        self._used_species: set = set()     # (subject, species)
        self._used_petnames: set = set()
        self._used_jobs: set = set()
        self._used_employers: set = set()
        self._used_allergens: set = set()
        self._used_books: set = set()
        self._used_hobbies: set = set()
        self._used_routines: set = set()    # (subject, routine tuple)
        self._used_weekdays: set = set()    # (subject, weekday)
        self._used_durations: set = set()

    # -- randomness helpers -------------------------------------------------

    def _pick(self, items):
        return items[self.rng.randrange(len(items))]

    def _pool(self, name: str, items, refill: bool = True):
        if name not in self._pools:
            lst = list(items)
            self.rng.shuffle(lst)
            self._pools[name] = lst
        if not self._pools[name]:
            if not refill:
                raise _PoolEmpty(name)
            lst = list(items)
            self.rng.shuffle(lst)
            self._pools[name] = lst
        return self._pools[name].pop()

    def _draw_unique(self, name: str, items, used: set,
                     key=None, extra_block: Optional[set] = None,
                     tries: int = 16):
        """Draw a value whose ``key`` is not in ``used`` (nor in
        ``extra_block``); refills the pool, retries, then gives up."""
        key = key or (lambda v: v)
        for _ in range(tries):
            v = self._pool(name, items, refill=True)
            k = key(v)
            if k in used or (extra_block and v in extra_block):
                continue
            used.add(k)
            return v
        raise _PoolEmpty(name)

    def _pick_session(self, exclude: frozenset = frozenset()) -> int:
        """Least-loaded eligible session (keeps beats balanced)."""
        best = min(
            self.session_load[s]
            for s in range(self.n_sessions)
            if s not in exclude
        )
        cand = [
            s for s in range(self.n_sessions)
            if self.session_load[s] == best and s not in exclude
        ]
        s = self.rng.choice(cand)
        self.session_load[s] += 1
        return s

    def _two_sessions(self) -> Tuple[int, int]:
        a = self._pick_session()
        if self.n_sessions < 2:
            return a, a
        b = self._pick_session(exclude=frozenset({a}))
        return a, b

    def _other(self, speaker: str) -> str:
        return self.assistant if speaker == self.user else self.user

    def _session_date(self, s: int) -> date:
        return self.session_starts[s].date()

    def _past_date(self, s: int, lo: int = 10, hi: int = 600) -> date:
        return self._session_date(s) - timedelta(
            days=self.rng.randint(lo, hi))

    def _fresh_weekday(self, subjects: Tuple[str, ...]) -> str:
        for _ in range(8):
            wd = self._pick(WEEKDAYS)
            if all((s, wd) not in self._used_weekdays for s in subjects):
                for s in subjects:
                    self._used_weekdays.add((s, wd))
                return wd
        raise _PoolEmpty("weekdays")

    # -- records -------------------------------------------------------------

    def _add_beat(self, session: int, speaker: str, text: str,
                  kind: str) -> Dict[str, Any]:
        self._beat_seq += 1
        beat = {
            "b": self._beat_seq,
            "session": session,
            "speaker": speaker,
            "text": text,
            "kind": kind,
            "turn_id": None,
        }
        self.beats.append(beat)
        return beat

    def _add_fact(self, kind: str, subject: str, beat: Dict[str, Any],
                  must_contain: Iterable[str], value: Dict[str, Any],
                  *, mention: Optional[str] = None,
                  when: Optional[date] = None,
                  verb: Optional[str] = None,
                  verb_past: Optional[str] = None,
                  orderable: bool = True,
                  shared: bool = False,
                  related: Optional[str] = None) -> Dict[str, Any]:
        self._fact_seq += 1
        fact = {
            "fact_id": f"f{self._fact_seq:03d}",
            "kind": kind,
            "subject": subject,
            "beat": beat,
            "must_contain": list(must_contain),
            "value": dict(value),
            "mention": mention,
            "shared_across_speakers": shared,
            "related_fact": related,
        }
        self.facts.append(fact)
        if when is not None:
            fact["date"] = when.isoformat()
            fact["verb"] = verb
            fact["verb_past"] = verb_past or verb
            fact["orderable"] = orderable
            self.dated_facts.append(fact)
        return fact

    def _add_question(self, category: str, subtype: str, query: str,
                      answer: Optional[str], beats: List[Dict[str, Any]],
                      *, answerable: bool = True,
                      trap: Optional[Dict[str, Any]] = None) -> None:
        self.drafts.append({
            "category": category,
            "subtype": subtype,
            "query": query,
            "answer": answer,
            "answerable": answerable,
            "beats": list(beats),
            "trap": trap,
        })

    def _distract(self, fact: Dict[str, Any], text: str,
                  must: Iterable[str]) -> None:
        """Plant a near-miss turn (same speaker, different session)."""
        if self.rng.random() >= self.distractor_rate:
            return
        s = self._pick_session(
            exclude=frozenset({fact["beat"]["session"]})
            if self.n_sessions > 1 else frozenset())
        beat = self._add_beat(s, fact["subject"], text, "distractor")
        self._add_fact("distractor", fact["subject"], beat, must,
                       {"echo_of": fact["fact_id"]},
                       related=fact["fact_id"])

    # -- city bookkeeping -----------------------------------------------------

    def _fresh_visit_city(self, block_rel: bool = False) -> str:
        """A globally fresh 'was physically there' city."""
        blocked = self._visit_cities | self._race_cities | self._shared_cities
        if block_rel:
            blocked = blocked | self._rel_cities
        for _ in range(16):
            c = self._pool("vcity", CITIES, refill=True)
            if c not in blocked:
                return c
        raise _PoolEmpty("visit cities")

    def _claim_visit(self, subject: str, city: str,
                     fact: Dict[str, Any]) -> None:
        self._visit_cities.add(city)
        self.went[subject].add(city)
        self._visit_facts[(subject, city)] = fact
        self._visit_count[(subject, city)] = 1

    def _fresh_race_city(self) -> str:
        blocked = self._visit_cities | self._shared_cities
        for _ in range(16):
            c = self._pool("rcity", CITIES, refill=True)
            if c not in blocked:
                return c
        raise _PoolEmpty("race cities")

    def _fresh_rel_city(self) -> str:
        blocked = self._rel_cities | self._visit_cities
        for _ in range(16):
            c = self._pool("relcity", CITIES, refill=True)
            if c not in blocked:
                self._rel_cities.add(c)
                return c
        raise _PoolEmpty("rel cities")

    # -- fact families --------------------------------------------------------

    def f_hobby(self, subject: str, session: Optional[int] = None,
                hobby: Optional[str] = None,
                shared: bool = False) -> Dict[str, Any]:
        if hobby is None:
            hobby = self._draw_unique("hobbies", HOBBIES,
                                      self._used_hobbies)
        s = session if session is not None else self._pick_session()
        tpl = self._pick((
            "I have gotten really into {h} lately, I practice most evenings now.",
            "Small confession: I started {h} classes at the community hall.",
            "My newest obsession is {h}. I can bore anyone about it for an hour.",
        ))
        beat = self._add_beat(s, subject, tpl.format(h=hobby), "evidence")
        fact = self._add_fact("hobby", subject, beat, [hobby],
                              {"hobby": hobby}, mention=hobby, shared=shared)
        self._subj_family.add((subject, "hobby"))
        self._distract(
            fact,
            f"There was a flyer at the library for a "
            f"{self._pick(HOBBIES)} circle, not really my speed.",
            ["flyer"])
        return fact

    def f_pet(self, subject: str) -> Dict[str, Any]:
        sp = self._draw_unique(
            "pet_species", PET_SPECIES, self._used_species,
            key=lambda v: (subject, v))
        pet = self._draw_unique("pet_names", PET_NAMES, self._used_petnames)
        s = self._pick_session()
        tpl = self._pick((
            "We adopted a {sp}! The name we picked is {pet}.",
            "Big news at our place: a {sp} named {pet} joined the family.",
            "Meet {pet}, our newly adopted {sp}. Already runs the house.",
        ))
        beat = self._add_beat(s, subject, tpl.format(sp=sp, pet=pet),
                              "evidence")
        fact = self._add_fact("pet", subject, beat, [sp, pet],
                              {"species": sp, "name": pet}, mention=pet)
        self._distract(
            fact,
            f"A stray {sp} showed up at the park again, bold as anything.",
            [sp])
        return fact

    def f_job(self, subject: str) -> Dict[str, Any]:
        job = self._draw_unique("jobs", JOBS, self._used_jobs)
        emp = self._draw_unique("employers", EMPLOYERS,
                                self._used_employers)
        s = self._pick_session()
        tpl = self._pick((
            "I started a new job as a {job} at {emp}. First week done.",
            "Work news: I am now a {job} over at {emp}.",
        ))
        beat = self._add_beat(s, subject, tpl.format(job=job, emp=emp),
                              "evidence")
        fact = self._add_fact("job", subject, beat, [job, emp],
                              {"job": job, "employer": emp}, mention=emp)
        self._subj_family.add((subject, "job"))
        self._distract(
            fact,
            f"My cousin is a {job} too, over in {self._pick(CITIES)}.",
            [job])
        return fact

    def f_purchase(self, subject: str, session: Optional[int] = None,
                   item: Optional[str] = None,
                   shop: Optional[str] = None,
                   distract: bool = True) -> Dict[str, Any]:
        if item is None:
            item = self._draw_unique(
                "items", ITEMS, self._used_items,
                key=lambda v: (subject, v))
        if shop is None:
            shop = self._draw_unique(
                "shops", SHOPS, self._used_shops,
                key=lambda v: (subject, v))
        s = session if session is not None else self._pick_session()
        tpl = self._pick((
            "I found a {item} at {shop} and finally bought it.",
            "Picked up a {item} from {shop} this morning, very pleased.",
        ))
        beat = self._add_beat(s, subject, tpl.format(item=item, shop=shop),
                              "evidence")
        fact = self._add_fact("purchase", subject, beat, [item, shop],
                              {"item": item, "shop": shop}, mention=item)
        if distract:
            self._distract(
                fact,
                f"{shop} was closed when I walked by, typical Monday.",
                [shop])
        return fact

    def f_shoploc(self, subject: str, session: int, shop: str,
                  street: str, related: str) -> Dict[str, Any]:
        tpl = self._pick((
            "{shop}? It is over on {street}, next to the bakery.",
            "I walked past {shop} on {street} today, cute window display.",
        ))
        beat = self._add_beat(session, subject,
                              tpl.format(shop=shop, street=street),
                              "evidence")
        return self._add_fact("shoploc", subject, beat, [shop, street],
                              {"shop": shop, "street": street},
                              mention=street, related=related)

    def _visit_beat(self, subject: str, s: int, city: str, lm: str,
                    d: date, revisit: bool = False) -> Dict[str, Any]:
        dp = _date_phrase(d)
        if revisit:
            tpl = self._pick((
                "I went back to {city} on {dp}, this time for {lm}.",
                "Second trip to {city}! On {dp} I finally saw {lm}.",
            ))
        else:
            tpl = self._pick((
                "Back on {dp} I visited {lm} in {city}.",
                "I still think about {lm} in {city} - I was there on {dp}.",
            ))
        return self._add_beat(
            s, subject, tpl.format(dp=dp, lm=lm, city=city), "evidence")

    def f_visit(self, subject: str, session: Optional[int] = None,
                city: Optional[str] = None) -> Dict[str, Any]:
        if city is None:
            city = self._fresh_visit_city()
        lm = self._draw_unique(
            "landmarks", LANDMARKS, self._used_landmarks,
            key=lambda v: (subject, v))
        s = session if session is not None else self._pick_session()
        d = self._past_date(s)
        beat = self._visit_beat(subject, s, city, lm, d)
        fact = self._add_fact("visit", subject, beat, [lm, city],
                              {"landmark": lm, "city": city,
                               "date": d.isoformat()},
                              mention=lm, when=d,
                              verb=f"visit {lm}",
                              verb_past=f"visited {lm}")
        fact["must_contain"].append(_date_phrase(d))
        self._claim_visit(subject, city, fact)
        self._distract(fact,
                       f"{city} keeps coming up in the news lately.",
                       [city])
        return fact

    def f_revisit(self, subject: str, first: Dict[str, Any],
                  session: int) -> Dict[str, Any]:
        city = first["value"]["city"]
        lm = self._draw_unique(
            "landmarks", LANDMARKS, self._used_landmarks,
            key=lambda v: (subject, v))
        d1 = date.fromisoformat(first["value"]["date"])
        d = d1 + timedelta(days=self.rng.randint(10, 180))
        beat = self._visit_beat(subject, session, city, lm, d,
                                revisit=True)
        self._visit_count[(subject, city)] = 2
        return self._add_fact("revisit", subject, beat,
                              [city, lm, _date_phrase(d)],
                              {"landmark": lm, "city": city,
                               "date": d.isoformat()},
                              mention=lm, when=d, verb=f"return to {city}",
                              verb_past=f"returned to {city}",
                              related=first["fact_id"])

    def f_visit_planned(self, subject: str) -> Dict[str, Any]:
        city = self._fresh_visit_city()
        lm = self._draw_unique(
            "landmarks", LANDMARKS, self._used_landmarks,
            key=lambda v: (subject, v))
        s = self._pick_session()
        d = self._session_date(s) + timedelta(days=self.rng.randint(10, 200))
        dp = _date_phrase(d)
        tpl = self._pick((
            "I am going to {lm} in {city} on {dp}, already packed.",
            "Big plans: {lm} in {city} on {dp}.",
        ))
        beat = self._add_beat(s, subject,
                              tpl.format(dp=dp, lm=lm, city=city),
                              "evidence")
        fact = self._add_fact("visit_planned", subject, beat,
                              [lm, city, dp],
                              {"landmark": lm, "city": city,
                               "date": d.isoformat()},
                              mention=lm, when=d, verb=f"go to {lm}",
                              verb_past=f"went to {lm}",
                              orderable=False)
        self._visit_cities.add(city)  # claim city, but not a "was there"
        self._distract(fact,
                       f"{city} hotels are already booking up for the "
                       "season.", [city])
        return fact

    def f_festival(self, subject: str, session: int,
                   city: str) -> Dict[str, Any]:
        fest = self._draw_unique("festivals", FESTIVALS,
                                 self._used_festivals)
        d = self._past_date(session)
        dp = _date_phrase(d)
        tpl = self._pick((
            "I went to {city} for {fest} on {dp}.",
            "Loved {fest} - I was in {city} for it on {dp}.",
        ))
        beat = self._add_beat(session, subject,
                              tpl.format(city=city, fest=fest, dp=dp),
                              "evidence")
        fact = self._add_fact("festival", subject, beat, [fest, city, dp],
                              {"festival": fest, "city": city,
                               "date": d.isoformat()},
                              mention=fest, when=d,
                              verb=f"attend {fest}",
                              verb_past=f"attended {fest}")
        self._claim_visit(subject, city, fact)
        return fact

    def f_race(self, subject: str, session: int, city: str,
               race: str, avoid: Optional[date] = None) -> Dict[str, Any]:
        d = self._past_date(session, lo=20, hi=400)
        if avoid is not None and d == avoid:
            d = avoid - timedelta(days=31)
        dp = _date_phrase(d)
        tpl = self._pick((
            "I ran the {city} {race} on {dp}. My legs are still mad at me.",
            "{dp}: the {city} {race}. Finished, barely.",
        ))
        beat = self._add_beat(session, subject,
                              tpl.format(city=city, race=race, dp=dp),
                              "evidence")
        fact = self._add_fact("race", subject, beat, [city, race, dp],
                              {"city": city, "race": race,
                               "date": d.isoformat()},
                              mention=f"the {city} {race}", when=d,
                              verb=f"run the {city} {race}",
                              verb_past=f"ran the {city} {race}")
        self._race_cities.add(city)
        self.went[subject].add(city)
        return fact

    def f_relation(self, subject: str, session: Optional[int] = None,
                   city: Optional[str] = None,
                   rel: Optional[str] = None,
                   rname: Optional[str] = None) -> Dict[str, Any]:
        if rel is None:
            rel = self._draw_unique(
                "relations", RELATIONS, self._used_rel,
                key=lambda v: (subject, v))
        if rname is None:
            rname = self._draw_unique(
                "rel_names",
                [n for n in FIRST_NAMES if n not in self.speakers],
                self._used_rnames, key=lambda v: (subject, v))
        if city is None:
            city = self._fresh_rel_city()
        else:
            self._rel_cities.add(city)
        s = session if session is not None else self._pick_session()
        tpl = self._pick((
            "My {rel} {rn} just moved to {city}, near the old dockyard.",
            "{rn}, my {rel}, is living in {city} now.",
        ))
        beat = self._add_beat(s, subject,
                              tpl.format(rel=rel, rn=rname, city=city),
                              "evidence")
        fact = self._add_fact("relation", subject, beat, [rel, rname, city],
                              {"relation": rel, "name": rname,
                               "city": city}, mention=rname)
        self._distract(fact,
                       f"My {rel} called me yesterday, we talked for an "
                       "hour.", [rel])
        return fact

    def f_allergy(self, subject: str) -> Dict[str, Any]:
        al = self._draw_unique("allergens", ALLERGENS,
                               self._used_allergens)
        s = self._pick_session()
        tpl = self._pick((
            "Turns out I am allergic to {al}. All these years unexplained.",
            "Newly diagnosed: {al} allergy. That explains the picnics.",
        ))
        beat = self._add_beat(s, subject, tpl.format(al=al), "evidence")
        fact = self._add_fact("allergy", subject, beat, [al],
                              {"allergen": al}, mention=al)
        self._subj_family.add((subject, "allergy"))
        self._distract(
            fact,
            f"Everything at that cafe has {self._pick(ALLERGENS)} in it, "
            "I checked.", ["cafe"])
        return fact

    def f_routine(self, subject: str, session: Optional[int] = None,
                  routine: Optional[Tuple[str, str, str]] = None,
                  weekday: Optional[str] = None) -> Dict[str, Any]:
        if routine is None:
            routine = self._draw_unique(
                "routines", ROUTINES, self._used_routines,
                key=lambda v: (subject, v))
        if weekday is None:
            weekday = self._fresh_weekday((subject,))
        else:
            self._used_weekdays.add((subject, weekday))
        vp, head, place = routine
        s = session if session is not None else self._pick_session()
        tpl = self._pick((
            "Every {wd} morning I {vp}. Rain or shine.",
            "I {vp} every {wd} morning without fail.",
        ))
        beat = self._add_beat(s, subject, tpl.format(wd=weekday, vp=vp),
                              "evidence")
        fact = self._add_fact("routine", subject, beat, [vp, weekday],
                              {"routine": vp, "weekday": weekday,
                               "place": place, "head": head},
                              mention=place)
        self._distract(fact,
                       f"I passed {place} this morning, it was packed.",
                       [place])
        return fact

    def f_book(self, subject: str) -> Dict[str, Any]:
        title, author = self._draw_unique("books", BOOKS, self._used_books)
        s = self._pick_session()
        tpl = self._pick((
            "I finally finished {t} by {a}. Could not put it down.",
            "Just closed {t} by {a}, what an ending.",
        ))
        beat = self._add_beat(s, subject, tpl.format(t=title, a=author),
                              "evidence")
        fact = self._add_fact("book", subject, beat, [title, author],
                              {"title": title, "author": author},
                              mention=title)
        self._distract(fact,
                       f"{author} has a new book coming out this winter.",
                       [author])
        return fact

    def f_rel_event(self, subject: str, session: int,
                    action: Tuple[str, str, str],
                    rel: Tuple[str, str, int]) -> Dict[str, Any]:
        aid, past, query = action
        surface, kind, n = rel
        turn_d = self._session_date(session)
        if kind == "days":
            resolved = turn_d - timedelta(days=n)
            surf = surface
        elif kind == "months":
            resolved = _shift_months(turn_d, -n)
            surf = surface
        else:  # weekday
            wd_name = self._pick(WEEKDAYS)
            resolved = _last_weekday(turn_d, WEEKDAYS.index(wd_name))
            surf = surface.format(weekday=wd_name)
        surf_text = surf[0].upper() + surf[1:]
        text = f"{surf_text}, I {past}."
        beat = self._add_beat(session, subject, text, "evidence")
        self._subj_actions.add((subject, aid))
        return self._add_fact("rel_event", subject, beat, [surf_text, past],
                              {"action": aid, "relative": surf,
                               "resolved": resolved.isoformat()},
                              when=resolved, verb=query, verb_past=past)

    def f_explicit_event(self, subject: str, session: int,
                         action: Tuple[str, str, str]) -> Dict[str, Any]:
        aid, past, query = action
        d = self._past_date(session, lo=30, hi=800)
        dp = _date_phrase(d)
        text = f"On {dp}, I {past}."
        beat = self._add_beat(session, subject, text, "evidence")
        self._subj_actions.add((subject, aid))
        return self._add_fact("explicit_event", subject, beat, [dp, past],
                              {"action": aid, "date": d.isoformat()},
                              when=d, verb=query, verb_past=past)

    def f_duration(self, subject: str,
                   entry: Tuple[str, str, int, str]) -> Dict[str, Any]:
        g, word, n, unit = entry
        s = self._pick_session()
        text = f"I have been {g} for {word} {unit} now."
        beat = self._add_beat(s, subject, text, "evidence")
        return self._add_fact("duration", subject, beat,
                              [g, f"{word} {unit}"],
                              {"activity": g, "n": n, "unit": unit,
                               "answer": f"{word} {unit}"},
                              mention=g)

    def f_duration_since(self, subject: str,
                         entry: Tuple[str, str, int, str]) -> Dict[str, Any]:
        g, _w, n, unit = entry
        s = self._pick_session()
        if unit == "months":
            start = _shift_months(self._session_date(s), -n)
        else:
            start = date(self._session_date(s).year - n,
                         self._session_date(s).month, 1)
        text = f"I have been {g} since {_MONTHS[start.month]} {start.year}."
        beat = self._add_beat(s, subject, text, "evidence")
        months = (self._session_date(s).year - start.year) * 12 + \
            (self._session_date(s).month - start.month)
        return self._add_fact("duration", subject, beat,
                              [g, f"{_MONTHS[start.month]} {start.year}"],
                              {"activity": g, "since": start.isoformat(),
                               "answer": f"about {months} months"},
                              mention=g)


# ---------------------------------------------------------------------------
# Question emitters
# ---------------------------------------------------------------------------


def _subject(rng: random.Random, speakers: Tuple[str, str]) -> str:
    return speakers[rng.randrange(2)]


def _drive(b: _Builder, count: int, shapes: Tuple[str, ...],
           one) -> int:
    """Cycle ``shapes`` (shuffled once) until ``count`` emissions or every
    shape keeps declining (pool-capped shapes return False)."""
    order = list(shapes)
    b.rng.shuffle(order)
    emitted = 0
    idx = 0
    fails = 0
    cap = max(12, len(order) * 4)
    while emitted < count and fails < cap:
        shape = order[idx % len(order)]
        idx += 1
        try:
            ok = one(b, shape)
        except _PoolEmpty:
            ok = False
        if ok:
            emitted += 1
            fails = 0
        else:
            fails += 1
    return emitted


def _mh_one(b: _Builder, shape: str) -> bool:
    user, asst = b.user, b.assistant
    if shape == "shared_hobby":
        if ((user, "hobby") in b._subj_family
                or (asst, "hobby") in b._subj_family):
            return False
        hobby = b._draw_unique("hobbies", HOBBIES, b._used_hobbies)
        s1, s2 = b._two_sessions()
        fA = b.f_hobby(user, session=s1, hobby=hobby, shared=True)
        fB = b.f_hobby(asst, session=s2, hobby=hobby, shared=True)
        q = b._pick((
            "What hobby do {A} and {B} have in common?",
            "Which pastime do both {A} and {B} share?",
        )).format(A=user, B=asst)
        b._add_question("multi_hop", shape, q, hobby,
                        [fA["beat"], fB["beat"]])
        return True
    if shape == "shared_city":
        city = b._fresh_visit_city(block_rel=True)
        s1, s2 = b._two_sessions()
        fA = b.f_visit(user, session=s1, city=city)
        fB = b.f_visit(asst, session=s2, city=city)
        b._shared_cities.add(city)
        q = b._pick((
            "Which city have both {A} and {B} visited?",
            "{A} and {B} have both been to which city?",
        )).format(A=user, B=asst)
        b._add_question("multi_hop", shape, q, city,
                        [fA["beat"], fB["beat"]])
        return True
    if shape == "paired_routine":
        wd = b._fresh_weekday(b.speakers)
        rA = b._draw_unique("routines", ROUTINES, b._used_routines,
                            key=lambda v: (user, v))
        rB = b._draw_unique("routines", ROUTINES, b._used_routines,
                            key=lambda v: (asst, v))
        s1, s2 = b._two_sessions()
        fA = b.f_routine(user, session=s1, routine=rA, weekday=wd)
        fB = b.f_routine(asst, session=s2, routine=rB, weekday=wd)
        q = f"What does each speaker do on {wd} mornings?"
        ans = f"{user}: {rA[0]}; {asst}: {rB[0]}"
        b._add_question("multi_hop", shape, q, ans,
                        [fA["beat"], fB["beat"]])
        return True
    if shape == "compare_race":
        race = b._draw_unique("races", RACES, b._used_races)
        cA = b._fresh_race_city()
        cB = b._fresh_race_city()
        s1, s2 = b._two_sessions()
        fA = b.f_race(user, s1, cA, race)
        fB = b.f_race(asst, s2, cB, race,
                      avoid=date.fromisoformat(fA["date"]))
        dA, dB = fA["date"], fB["date"]
        first, fd = (user, dA) if dA < dB else (asst, dB)
        q = b._pick((
            "Who ran a {race} first, {A} or {B}?",
            "Which speaker ran a {race} earlier, {A} or {B}?",
        )).format(race=race, A=user, B=asst)
        b._add_question("multi_hop", shape, q, f"{first} ({fd})",
                        [fA["beat"], fB["beat"]])
        return True
    if shape == "purchase_chain":
        buyer = _subject(b.rng, b.speakers)
        item = b._draw_unique("items", ITEMS, b._used_items,
                              key=lambda v: (buyer, v))
        shop = b._draw_unique("shops", SHOPS, b._used_shops,
                              key=lambda v: (buyer, v))
        if shop in b._shop_street:
            street = b._shop_street[shop]
        else:
            street = b._pool("streets", STREETS, refill=True)
            b._shop_street[shop] = street
        s1, s2 = b._two_sessions()
        f1 = b.f_purchase(buyer, session=s1, item=item, shop=shop)
        speaker2 = b._pick(b.speakers)
        f2 = b.f_shoploc(speaker2, s2, shop, street, f1["fact_id"])
        q = b._pick((
            "On which street is the shop where {n} bought the {item}?",
            "Where is the shop {n} bought the {item} from located?",
        )).format(n=buyer, item=item)
        b._add_question("multi_hop", shape, q, street,
                        [f1["beat"], f2["beat"]])
        return True
    if shape == "event_attr":
        visitor = _subject(b.rng, b.speakers)
        owner = b._pick(b.speakers)
        city = b._fresh_visit_city(block_rel=True)
        s1, s2 = b._two_sessions()
        if b.rng.random() < 0.5:
            f1 = b.f_festival(visitor, s1, city)
            anchor = f1["value"]["festival"]
            q = ("Whose relative lives in the city where {v} attended "
                 "{a}?").format(v=visitor, a=anchor)
        else:
            f1 = b.f_visit(visitor, session=s1, city=city)
            anchor = f1["value"]["landmark"]
            q = ("Whose relative lives in the city where {v} visited "
                 "{a}?").format(v=visitor, a=anchor)
        f2 = b.f_relation(owner, session=s2, city=city)
        rel, rname = f2["value"]["relation"], f2["value"]["name"]
        ans = f"{owner}'s {rel} {rname}"
        b._add_question("multi_hop", shape, q, ans,
                        [f1["beat"], f2["beat"]])
        return True
    if shape == "repeat_visit":
        subj = _subject(b.rng, b.speakers)
        cands = [
            k for k, f in b._visit_facts.items()
            if k[0] == subj
            and b._visit_count[k] < 2
            and k[1] not in b._shared_cities
        ]
        if cands:
            key = b._pick(tuple(cands))
            first = b._visit_facts[key]
            city = key[1]
        else:
            first = b.f_visit(subj)
            city = first["value"]["city"]
        first_s = first["beat"]["session"]
        if first_s < b.n_sessions - 1:
            s2 = b._pick_session(
                exclude=frozenset(range(first_s)))
        else:
            s2 = first_s
        f2 = b.f_revisit(subj, first, s2)
        q = b._pick((
            "What did {n} do on their second trip to {city}?",
            "The second time {n} went to {city}, what did they see?",
        )).format(n=subj, city=city)
        b._add_question("multi_hop", shape, q,
                        f"visited {f2['value']['landmark']}",
                        [first["beat"], f2["beat"]])
        return True
    return False


def emit_multi_hop(b: _Builder, i: int) -> None:
    _drive(b, i, MH_SHAPES, _mh_one)


def _sh_fact(b: _Builder, fam: str, subj: str) -> Optional[Dict[str, Any]]:
    if fam in _RESTRICTED and (subj, fam) in b._subj_family:
        return None
    try:
        if fam == "hobby":
            return b.f_hobby(subj)
        if fam == "pet":
            return b.f_pet(subj)
        if fam == "job":
            return b.f_job(subj)
        if fam == "purchase":
            return b.f_purchase(subj)
        if fam == "visit":
            return b.f_visit(subj)
        if fam == "visit_planned":
            return b.f_visit_planned(subj)
        if fam == "relation":
            return b.f_relation(subj)
        if fam == "allergy":
            return b.f_allergy(subj)
        if fam == "routine":
            return b.f_routine(subj)
        if fam == "book":
            return b.f_book(subj)
        if fam == "race":
            s = b._pick_session()
            race = b._draw_unique("races", RACES, b._used_races)
            return b.f_race(subj, s, b._fresh_race_city(), race)
    except _PoolEmpty:
        return None
    return None


def emit_single_hop(b: _Builder, i: int) -> None:
    # Build facts across families/subjects, then distribute questions.
    combos = [(fam, subj) for fam in SH_FAMILIES for subj in b.speakers]
    b.rng.shuffle(combos)
    facts: List[Dict[str, Any]] = []
    for fam, subj in combos:
        n_facts = 1 if fam in _RESTRICTED else b.rng.choice((1, 2))
        for _ in range(n_facts):
            f = _sh_fact(b, fam, subj)
            if f is not None:
                facts.append(f)
    if not facts:
        return
    b.rng.shuffle(facts)
    qcount: Dict[str, int] = {}
    for k in range(i):
        fact = facts[k % len(facts)]
        fsubs = _sh_qtemplates(fact)
        if not fsubs:
            continue
        ti = qcount.get(fact["fact_id"], 0) % len(fsubs)
        qcount[fact["fact_id"]] = ti + 1
        q_fmt, ans_fmt = fsubs[ti]
        fmt = {"n": fact["subject"], "mention": fact.get("mention"),
               **fact["value"]}
        b._add_question("single_hop", fact["kind"], q_fmt.format(**fmt),
                        ans_fmt.format(**fmt), [fact["beat"]])


def _sh_qtemplates(fact: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(query_template, answer_template) pairs per fact kind."""
    kind = fact["kind"]
    if kind == "hobby":
        return [
            ("What hobby has {n} gotten into lately?", "{hobby}"),
            ("What does {n} practice most evenings?", "{hobby}"),
            ("What new pastime did {n} mention?", "{hobby}"),
            ("What did {n} say about {mention}?", "{hobby}"),
        ]
    if kind == "pet":
        return [
            ("What did {n} name their {species}?", "{name}"),
            ("What kind of animal is {name}?", "{species}"),
            ("What is {n}'s {species} called?", "{name}"),
            ("What did {n} say about {mention}?", "{name}"),
        ]
    if kind == "job":
        return [
            ("What is {n}'s new job?", "{job} at {employer}"),
            ("Where does {n} work now?", "{employer}"),
            ("What does {n} do for a living?", "{job} at {employer}"),
            ("What did {n} say about {mention}?", "{job} at {employer}"),
        ]
    if kind == "purchase":
        return [
            ("What did {n} buy at {shop}?", "{item}"),
            ("Where did {n} find the {item}?", "{shop}"),
            ("What did {n} say about {mention}?", "{item}"),
        ]
    if kind == "visit":
        return [
            ("What did {n} visit in {city}?", "{landmark}"),
            ("In which city did {n} visit {landmark}?", "{city}"),
            ("What did {n} say about {mention}?", "{landmark}"),
        ]
    if kind == "visit_planned":
        return [
            ("What is {n} planning to visit in {city}?", "{landmark}"),
            ("What did {n} say about {mention}?", "{landmark}"),
        ]
    if kind == "relation":
        return [
            ("Where does {n}'s {relation} {name} live?", "{city}"),
            ("How is {name} related to {n}?", "{relation}"),
            ("What did {n} say about {mention}?", "{relation}"),
        ]
    if kind == "race":
        return [
            ("In which city did {n} run their {race}?", "{city}"),
            ("What did {n} say about {mention}?", "{city}"),
        ]
    if kind == "allergy":
        return [
            ("What is {n} allergic to?", "{allergen}"),
            ("What allergy does {n} have?", "{allergen}"),
            ("What did {n} say about {mention}?", "{allergen}"),
        ]
    if kind == "routine":
        return [
            ("What does {n} do every {weekday} morning?", "{routine}"),
            ("On which morning does {n} {head}?", "{weekday}"),
            ("What did {n} say about their {weekday} routine?",
             "{routine}"),
        ]
    if kind == "book":
        return [
            ("Who wrote {title}, the book {n} finished?", "{author}"),
            ("What book by {author} did {n} finish?", "{title}"),
            ("What did {n} say about {mention}?", "{title}"),
        ]
    return []


def emit_temporal(b: _Builder, i: int) -> None:
    _drive(b, i, T_SHAPES, _t_one)


def _t_one(b: _Builder, shape: str) -> bool:
    if shape in ("relative", "explicit"):
        subj = _subject(b.rng, b.speakers)
        avail = [a for a in DATED_ACTIONS
                 if (subj, a[0]) not in b._subj_actions]
        if not avail:
            return False
        action = b._pick(tuple(avail))
        s = b._pick_session()
        if shape == "relative":
            rel = b._pick(REL_PHRASES)
            fact = b.f_rel_event(subj, s, action, rel)
            ans = fact["value"]["resolved"]
        else:
            fact = b.f_explicit_event(subj, s, action)
            ans = fact["value"]["date"]
        q = f"When did {subj} {action[2]}?"
        b._add_question("temporal", shape, q, ans, [fact["beat"]])
        return True
    if shape == "ordering":
        cands = [f for f in b.dated_facts if f.get("orderable")]
        b.rng.shuffle(cands)
        pair = None
        for x in range(len(cands)):
            for y in range(x + 1, len(cands)):
                if cands[x]["date"] != cands[y]["date"]:
                    pair = (cands[x], cands[y])
                    break
            if pair:
                break
        if not pair:
            return False
        f1, f2 = pair
        s1, s2 = f1["subject"], f2["subject"]
        v1, v2p = f1["verb"], f2.get("verb_past") or f2["verb"]
        d1, d2 = f1["date"], f2["date"]
        if s1 == s2:
            q = f"Did {s1} {v1} before or after they {v2p}?"
        else:
            q = f"Did {s1} {v1} before or after {s2} {v2p}?"
        ans = "before" if d1 < d2 else "after"
        b._add_question("temporal", shape, q, ans,
                        [f1["beat"], f2["beat"]])
        return True
    if shape == "duration":
        entry = b._draw_unique("durations", DURATIONS,
                               b._used_durations)
        subj = _subject(b.rng, b.speakers)
        if b.rng.random() < 0.4:
            fact = b.f_duration_since(subj, entry)
        else:
            fact = b.f_duration(subj, entry)
        q = f"How long has {subj} been {entry[0]}?"
        b._add_question("temporal", shape, q, fact["value"]["answer"],
                        [fact["beat"]])
        return True
    if shape == "session_date":
        cands = [f for f in b.facts
                 if f.get("mention") and f["kind"] != "distractor"
                 and not f.get("shared_across_speakers")]
        b.rng.shuffle(cands)
        for fact in cands:
            men = fact["mention"]
            # mention must be unique to its gold turn: not in any other
            # planned beat and not in the static filler/caption text.
            if any(_has_phrase(x["text"], men) for x in b.beats
                   if x is not fact["beat"]):
                continue
            if any(_has_phrase(t, men) for t in _STATIC_TEXTS):
                continue
            subj = fact["subject"]
            q = b._pick((
                "On what date did {n} mention {m}?",
                "What was the date when {n} talked about {m}?",
            )).format(n=subj, m=men)
            beat = fact["beat"]
            ans = b._session_date(beat["session"]).isoformat()
            b._add_question("temporal", shape, q, ans, [beat])
            b.drafts[-1]["_check_unique"] = men
            return True
        return False
    return False


def emit_open_domain(b: _Builder, i: int) -> None:
    for k in range(i):
        spec = OD_PAIRS[k % len(OD_PAIRS)]
        var = spec["vars"][(k // len(OD_PAIRS)) % len(spec["vars"])]
        subj = _subject(b.rng, b.speakers)
        vals = {f"v{j}": v for j, v in enumerate(var) if v}
        s = b._pick_session()
        text = spec["ev"].format(**vals)
        if k >= len(OD_PAIRS) * len(spec["vars"]):
            text = f"{b._pick(LEAD_INS)} {text}"
        beat = b._add_beat(s, subj, text, "evidence")
        mention = spec["mention"].format(**vals)
        fact = b._add_fact("open_domain", subj, beat,
                           [m for m in (mention, *vals.values()) if m],
                           {"theme": spec["name"], **vals},
                           mention=mention)
        q = spec["q"].format(n=subj)
        ans = spec["ans"].format(n=subj, **vals)
        b._add_question("open_domain", spec["name"], q, ans, [fact["beat"]])


def emit_abstain(b: _Builder, i: int) -> None:
    _drive(b, i, AB_SHAPES, _ab_one)


def _ab_one(b: _Builder, shape: str) -> bool:
    if shape == "never_discussed":
        unused = [(k, d) for k, d in ABSTAIN_TOPICS
                  if k not in b._used_topics]
        if not unused:
            return False
        key, disp = b._pick(tuple(unused))
        b._used_topics.add(key)
        subj = _subject(b.rng, b.speakers)
        q = b._pick((
            "What did {n} say about {t}?",
            "Did either speaker mention {t}?",
            "Has {n} ever talked about {t}?",
        )).format(n=subj, t=disp)
        b._add_question("abstain", shape, q, None, [],
                        answerable=False,
                        trap={"type": "never_discussed", "topic_key": key,
                              "topic": disp})
        b.drafts[-1]["_check_absent"] = key
        return True
    if shape == "speaker_mismatch":
        cands = [f for f in b.facts
                 if f.get("mention") and not f.get("shared_across_speakers")
                 and f["kind"] != "distractor"]
        b.rng.shuffle(cands)
        for fact in cands:
            other = b._other(fact["subject"])
            men = fact["mention"]
            # the trap is valid only if the asked speaker never uttered the
            # mention (abstain runs last, so all beats already exist).
            if any(_has_phrase(x["text"], men) for x in b.beats
                   if x["speaker"] == other):
                continue
            if any(_has_phrase(t, men) for t in _STATIC_TEXTS):
                continue
            q = b._pick((
                "What did {o} say about {m}?",
                "When did {o} mention {m}?",
                "What was {o}'s take on {m}?",
            )).format(o=other, m=men)
            b._add_question("abstain", shape, q, None, [],
                            answerable=False,
                            trap={"type": "speaker_mismatch",
                                  "actual_speaker": fact["subject"],
                                  "related_fact": fact["fact_id"],
                                  "mention": men})
            b.drafts[-1]["_check_speaker_absent"] = (men, other)
            return True
        return False
    if shape == "out_of_timeline":
        subj = _subject(b.rng, b.speakers)
        far = b.session_starts[0].date() - timedelta(
            days=b.rng.randint(12 * 365, 40 * 365))
        q = b._pick((
            "What did {n} do on {d}?",
            "What was {n} up to in {y}?",
            "Where was {n} living in {y}?",
        )).format(n=subj, d=_date_phrase(far), y=far.year)
        b._add_question("abstain", shape, q, None, [],
                        answerable=False,
                        trap={"type": "out_of_timeline",
                              "date": far.isoformat()})
        return True
    return False


# ---------------------------------------------------------------------------
# Layout + finalize
# ---------------------------------------------------------------------------


def _layout(b: _Builder, tps: int) -> List[Dict[str, Any]]:
    """Place beats into session slots, fill the rest with filler turns."""
    by_session: List[List[Dict[str, Any]]] = [
        [] for _ in range(b.n_sessions)]
    for beat in b.beats:
        by_session[beat["session"]].append(beat)

    turns: List[Dict[str, Any]] = []
    filler_pool: List[str] = []
    for s in range(b.n_sessions):
        beats_s = by_session[s]
        if len(beats_s) > tps:
            raise ValueError(
                f"session {s} needs {len(beats_s)} turns > "
                f"turns_per_session={tps}; raise turns_per_session or "
                f"n_sessions, or lower questions_per_category")
        idxs = sorted(b.rng.sample(range(tps), len(beats_s)))
        slots: List[Optional[Dict[str, Any]]] = [None] * tps
        for beat, idx in zip(beats_s, idxs):
            slots[idx] = beat
        step = b.rng.choice((3, 4, 5))
        start = b.session_starts[s]
        last: Optional[str] = None
        for i in range(tps):
            beat = slots[i]
            if beat is not None:
                speaker, text, kind = (
                    beat["speaker"], beat["text"], beat["kind"])
            else:
                speaker = b._other(last) if last else b.user
                if not filler_pool:
                    filler_pool = list(FILLERS)
                    b.rng.shuffle(filler_pool)
                text = filler_pool.pop()
                if b.rng.random() < 0.12:
                    text = f"{b._pick(LEAD_INS)} {text[0].lower()}{text[1:]}"
                kind = "filler"
            ts = start + timedelta(minutes=step * i)
            turn_id = f"D{s + 1}:{i + 1}"
            if beat is not None:
                beat["turn_id"] = turn_id
            turn = {
                "turn_id": turn_id,
                "session_id": f"S{s + 1:02d}",
                "turn_index": i + 1,
                "speaker": speaker,
                "role": b.role[speaker],
                "timestamp": _iso(ts),
                "timestamp_us": _us(ts),
                "text": text,
                "kind": kind,
            }
            if kind == "filler" and b.rng.random() < 0.08:
                turn["image_caption"] = f"a photo of {b._pick(CAPTIONS)}"
            turns.append(turn)
            last = speaker
    return turns


def _has_phrase(text: str, phrase: str) -> bool:
    """Case-insensitive word-boundary containment for a literal phrase."""
    return re.search(r"\b" + re.escape(phrase) + r"\b",
                     text, re.IGNORECASE) is not None


def _verify_and_finalize(b: _Builder, turns: List[Dict[str, Any]],
                         question_time: datetime) -> List[Dict[str, Any]]:
    """Emit public questions; enforce abstain/uniqueness invariants."""
    by_id = {t["turn_id"]: t for t in turns}
    counters: Dict[str, int] = {}
    questions: List[Dict[str, Any]] = []
    for d in b.drafts:
        # Swap a colliding never_discussed topic for a clean reserve one.
        key = d.pop("_check_absent", None)
        if key is not None:
            if any(key in t["text"].lower() for t in turns):
                swapped = False
                for k2, disp in ABSTAIN_TOPICS:
                    if k2 in b._used_topics:
                        continue
                    if any(k2 in t["text"].lower() for t in turns):
                        continue
                    b._used_topics.add(k2)
                    d["query"] = d["query"].replace(
                        d["trap"]["topic"], disp)
                    d["trap"] = {"type": "never_discussed",
                                 "topic_key": k2, "topic": disp}
                    swapped = True
                    break
                if not swapped:
                    continue  # drop: no clean topic left
        # Drop session_date questions whose mention leaks into a second turn.
        men = d.pop("_check_unique", None)
        if men is not None:
            hits = [t for t in turns if _has_phrase(t["text"], men)]
            gold = {beat["turn_id"] for beat in d["beats"]}
            if any(t["turn_id"] not in gold for t in hits):
                continue  # mention not unique -> ambiguous question
        # Drop mismatch traps whose mention was also spoken by the asked
        # speaker (can only happen via an unexpected echo; construction
        # already routes distractors through the fact's own subject).
        chk = d.pop("_check_speaker_absent", None)
        if chk is not None:
            men2, asked = chk
            if any(_has_phrase(t["text"], men2) for t in turns
                   if t["speaker"] == asked):
                continue
        cat = d["category"]
        counters[cat] = counters.get(cat, 0) + 1
        q = {
            "qid": f"{cat}-{counters[cat]:03d}",
            "category": cat,
            "subtype": d["subtype"],
            "query": d["query"],
            "answer": d["answer"],
            "answerable": d["answerable"],
            "evidence": [beat["turn_id"] for beat in d["beats"]],
            "question_time": _iso(question_time),
            "question_time_us": _us(question_time),
        }
        if d["trap"] is not None:
            trap = dict(d["trap"])
            rf = trap.get("related_fact")
            if rf:
                for f in b.facts:
                    if f["fact_id"] == rf:
                        trap["related_turn"] = f["beat"]["turn_id"]
                        break
            q["trap"] = trap
        questions.append(q)
    # All evidence refs must resolve (belt-and-suspenders).
    for q in questions:
        for tid in q["evidence"]:
            assert tid in by_id, f"gold evidence {tid} missing"
    return questions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate(
    seed: int = DEFAULT_SEED,
    n_sessions: int = 12,
    turns_per_session: Any = 30,
    questions_per_category: int = 24,
    *,
    category_counts: Optional[Dict[str, int]] = None,
    distractor_rate: float = 0.35,
    base_date: str = "2024-01-08",
) -> Dict[str, Any]:
    """Generate a LoCoMo-like owned corpus.  Byte-deterministic per seed.

    ``turns_per_session`` may be ``"auto"``: then it is sized from the
    planned evidence beats (~60% fill).  With the defaults (12x30 turns,
    24 questions/category) the corpus is ~360 turns / 120 questions.  For
    the >=500-question suite (V7-22.05) use e.g.
    ``n_sessions=30, turns_per_session=45, questions_per_category=105``
    or ``turns_per_session="auto"``.
    """
    for name, val in (("n_sessions", n_sessions),
                      ("questions_per_category", questions_per_category)):
        if not isinstance(val, int) or val < 1:
            raise ValueError(f"{name} must be a positive int, got {val!r}")
    if not (isinstance(turns_per_session, int) and turns_per_session > 0) \
            and turns_per_session != "auto":
        raise ValueError(
            "turns_per_session must be a positive int or 'auto', got "
            f"{turns_per_session!r}")
    counts = {c: questions_per_category for c in CATEGORIES}
    if category_counts:
        for k, v in category_counts.items():
            if k not in CATEGORIES:
                raise ValueError(f"unknown category {k!r}")
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"category count for {k} must be >= 0")
            counts[k] = v

    b = _Builder(seed, n_sessions, date.fromisoformat(base_date),
                 distractor_rate)

    # Fixed emission order (multi-hop claims shared resources first).
    emit_multi_hop(b, counts["multi_hop"])
    emit_single_hop(b, counts["single_hop"])
    emit_temporal(b, counts["temporal"])
    emit_open_domain(b, counts["open_domain"])
    emit_abstain(b, counts["abstain"])

    if turns_per_session == "auto":
        peak = max(b.session_load) if b.session_load else 1
        tps = max(10, int(peak / 0.6) + 1)
    else:
        tps = turns_per_session
    turns = _layout(b, tps)

    last_end = b.session_starts[-1] + timedelta(minutes=5 * tps)
    question_time = last_end + timedelta(days=1)
    questions = _verify_and_finalize(b, turns, question_time)

    facts_pub = []
    for f in b.facts:
        pub = {
            "fact_id": f["fact_id"],
            "kind": f["kind"],
            "subject": f["subject"],
            "turn_id": f["beat"]["turn_id"],
            "must_contain": f["must_contain"],
            "value": f["value"],
        }
        if f.get("mention"):
            pub["mention"] = f["mention"]
        if f.get("related_fact"):
            pub["related_fact"] = f["related_fact"]
        if f.get("shared_across_speakers"):
            pub["shared_across_speakers"] = True
        if "date" in f:
            pub["date"] = f["date"]
            pub["verb"] = f["verb"]
            pub["orderable"] = f["orderable"]
        facts_pub.append(pub)

    sessions = [
        {
            "session_id": f"S{s + 1:02d}",
            "index": s + 1,
            "started_at": _iso(b.session_starts[s]),
            "started_us": _us(b.session_starts[s]),
            "date": b._session_date(s).isoformat(),
            "turn_count": tps,
        }
        for s in range(b.n_sessions)
    ]

    per_category = {
        c: sum(1 for q in questions if q["category"] == c)
        for c in CATEGORIES
    }
    per_subtype = {
        st: sum(1 for q in questions if q["subtype"] == st)
        for st in sorted({q["subtype"] for q in questions})
    }
    corpus = {
        "name": CORPUS_NAME,
        "generator": GENERATOR_ID,
        "constants": CONSTANTS_TAG,
        "seed": seed,
        "params": {
            "n_sessions": n_sessions,
            "turns_per_session": tps,
            "questions_per_category": questions_per_category,
            "category_counts": category_counts,
            "distractor_rate": distractor_rate,
            "base_date": base_date,
        },
        "speakers": [
            {"name": b.user, "role": "user"},
            {"name": b.assistant, "role": "assistant"},
        ],
        "timeline": {
            "start": _iso(b.session_starts[0]),
            "end": _iso(last_end),
            "question_time": _iso(question_time),
        },
        "sessions": sessions,
        "turns": turns,
        "facts": facts_pub,
        "questions": questions,
        "stats": {
            "n_sessions": b.n_sessions,
            "n_turns": len(turns),
            "n_evidence_turns": sum(1 for t in turns
                                    if t["kind"] == "evidence"),
            "n_distractors": sum(1 for t in turns
                                 if t["kind"] == "distractor"),
            "n_facts": len(facts_pub),
            "n_questions": len(questions),
            "per_category": per_category,
            "per_subtype": per_subtype,
        },
    }
    corpus["digest"] = corpus_digest(corpus)
    return corpus


def to_jsonl(corpus: Dict[str, Any], path: Any) -> Path:
    """Write the corpus as canonical JSONL (byte-deterministic).

    Line order: one ``corpus`` meta record, then ``session``, ``turn``,
    ``fact``, and ``question`` records — a stream an ingester can replay
    in order.
    """
    p = Path(path)
    lines = [
        {"record": "corpus", "name": corpus["name"],
         "generator": corpus["generator"], "constants": corpus["constants"],
         "seed": corpus["seed"], "params": corpus["params"],
         "speakers": corpus["speakers"], "timeline": corpus["timeline"],
         "stats": corpus["stats"], "digest": corpus["digest"]},
        *({"record": "session", **s} for s in corpus["sessions"]),
        *({"record": "turn", **t} for t in corpus["turns"]),
        *({"record": "fact", **f} for f in corpus["facts"]),
        *({"record": "question", **q} for q in corpus["questions"]),
    ]
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in lines:
            fh.write(json.dumps(rec, sort_keys=True,
                                separators=(",", ":")) + "\n")
    return p


def iter_jsonl(path: Any) -> Iterator[Dict[str, Any]]:
    """Stream records written by :func:`to_jsonl`."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


__all__ = [
    "ABSTAIN_TOPICS",
    "CATEGORIES",
    "CONSTANTS_TAG",
    "CORPUS_NAME",
    "DEFAULT_SEED",
    "GENERATOR_ID",
    "corpus_digest",
    "generate",
    "iter_jsonl",
    "to_jsonl",
]
