"""pref_state/v1 — V7 wave-A deterministic preference & state extraction (§32.11/§32.12).

This module implements the frozen V7 contracts:

* ``extract_preferences(norm, unit_id, speaker_canon) -> list[PreferenceFact]``
* ``extract_state_facts(norm, unit_id, speaker_canon, occurred) -> list[StateFact]``
* ``state_compatible(key, a, b) -> bool``

The frozen signatures lack ``scope_id`` and (for preferences) ``occurred``;
both are accepted as keyword-only extensions:

* ``scope_id: str = ""`` — stamped verbatim onto every emitted artifact's
  ``scope_id`` field.  The persister stamps the real scope; extraction stays
  scope-blind by default.
* ``occurred: IntervalUs | None = None`` (``extract_preferences`` only) —
  the frozen contract does not pass occurrence time to preference extraction;
  callers that have it may supply it keyword-only.  Default is the unknown
  interval ``IntervalUs(None, None)``.
* ``raw_text: str | None = None`` — optional original surface.  Term byte
  offsets always pin the raw unit bytes; when ``raw_text`` is supplied (or
  ``norm.text`` is byte-aligned with the terms, e.g. ASCII projections or the
  wave convention of storing the unit text there), surfaces and guards that
  need raw bytes (capitalization, quote spans, dropped clitics like ``'d``)
  become available.  Without any aligned text the extractor degrades
  honestly: capitalized-name subjects and capitalization-gated values are
  skipped rather than guessed.
* ``canon_fn: Callable[[str], str] | None`` — canonicalizer for third-person
  named subjects (defaults to ``entities_v2.canon``, lazy-imported).

Determinism: no wall-clock, no randomness, no I/O, no models.  Same input
produces byte-identical output.

== Preferences (§32.12, pref/v1) ==

Positive  ``love|adore`` -> love_hate; ``like|enjoy|prefer|am into|
am a fan of`` -> like_dislike; ``can't get enough of`` -> love_hate.
Negative  ``hate|detest`` -> love_hate; ``dislike|can't stand|
am not a fan of|avoid|miss``-class -> like_dislike; ``never`` -> habitual.
Habitual  ``usually|always|typically|tend to|often|normally`` -> habitual.
Favorite/comparative  ``my favorite X is Y`` / ``X is my favorite`` /
``I'd rather X than Y`` -> favorite.
Constraint  ``I'm allergic to|I don't/can't eat or drink|I'm vegetarian/
vegan/...|gluten-free|lactose intolerant`` -> constraint.

Strength order (for dedupe): constraint > favorite > love_hate >
like_dislike > habitual.  Same (subject, object-span, polarity) keeps the
strongest.

Polarity is ``"positive"``/``"negative"``.  A negation window flips a
positive-base rule to ``"negative"``.  Negation applied to a negative-base
rule ("I don't hate X") ABSTAINS — a non-hate is not a liking signal.
Hypothetical ("would love", ``'d``-clitic, "might like"), quoted/reported,
hedged ("I think", "maybe"), conditional ("if I liked"), and interrogative
("do I like?") clauses are all excluded per §32.12 and the V5 polarity
vocabulary — implemented locally on the term stream because the V5 hedge
lexicon marks "tend to" hedged, which is a legitimate habitual trigger here.

Speaker scoping (V8-13.04 decided contract): first-person subjects
("i"/"we"/"my") bind ``speaker_canon`` — multi-party correctness; when the
input is unattributed (empty ``speaker_canon``) they bind ``"me"`` so
single-user facts land on ``me/<slot>`` keys.  "you" and unresolvable
third-person pronouns are skipped (no coref input in this contract); a
third-person NP binds ``entities_v2.canon(surface)`` only when its surface
is a capitalized name, else skipped.

Objects: the NP after the trigger, with and/or-of coordination kept and
identifiers (``norm.identifiers``) as legitimate objects — "I like ABC-123"
emits object_text "ABC-123".  Anaphoric objects ("I love it") are EMITTED
with ``object_text="it"`` — documented choice: the stated object is
literally "it"; resolution belongs to the sieve downstream, not extraction.

== State facts (§32.11, state_keys/v1) ==

~50 families with trigger lexicons, value extraction and compatibility
semantics; state_key = ``f"{subject_canon}/{family}"`` — ``speaker:<canon>/``
when a speaker canon is known, ``me/`` for unattributed input, bare entity
canon for named third-person subjects (V8-13.04).  Extraction emits
``StateFactStatus.CURRENT`` candidates only — lifecycle marking
(historical/disputed) belongs to the persister.  ``valid_from_us`` =
``occurred.start_us``; ``valid_to_us`` is left open (None).

Compatibility (``state_compatible``): accumulate-mode families (pets,
pet_names, children, hobbies, languages, allergies, health_condition,
medication, sport, programming_language, subscription, diet, goal_current,
schedule_regular, device) never conflict.  All other families are
replace-mode: equal normalized values are compatible, differing values
conflict.  ``age`` compares numerically.  Unknown families default to
replace semantics — honest and conservative.

Formula status: everything here is ``provisional/v7-r0`` (frozen constant
``FORMULA_STATUS_PROVISIONAL``).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from verbatim.core.types_v7 import (
    FORMULA_STATUS_PROVISIONAL,
    IntervalUs,
    NormAnalysis,
    NormTerm,
    PreferenceFact,
    StateFact,
    StateFactStatus,
)

MODULE_ID = "pref_state/v1"
EXTRACTOR_ID = MODULE_ID
FORMULA_STATUS = FORMULA_STATUS_PROVISIONAL  # "provisional/v7-r0"
PREF_RULES_VERSION = "pref/v1"
STATE_KEYS_VERSION = "state_keys/v1"

COUNTERS: dict[str, int] = {
    "units_pref": 0,
    "units_state": 0,
    "emitted_pref": 0,
    "emitted_state": 0,
    "guarded": 0,            # candidates dropped by polarity guards
    "subject_skip": 0,       # candidates dropped: no bindable subject
    "value_skip": 0,         # candidates dropped: no/invalid value
    "deduped": 0,
    "canon_unavailable": 0,
    "name_uncapitalized": 0,  # third-person surface not capitalized -> skipped
}


def reset_counters() -> None:
    for k in COUNTERS:
        COUNTERS[k] = 0


# ---------------------------------------------------------------------------
# Local text helpers (stdlib only)
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _foldish(text: str) -> str:
    """NFKC + casefold + diacritic-strip + punctuation->space + ws collapse.

    Mirrors entities_v2._fold semantics (punct -> space) so a raw slice can be
    consistency-checked against folded terms.  Local copy: sibling internals
    are private API.
    """
    norm = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", text).casefold())
    out: list[str] = []
    for ch in norm:
        cat = unicodedata.category(ch)
        if cat.startswith("M"):
            continue
        if cat[0] in ("P", "S", "C", "Z"):
            out.append(" ")
        else:
            out.append(ch)
    return _WS.sub(" ", "".join(out)).strip()


def _vnorm(text: str) -> str:
    """value_norm: folded form with a leading *standalone* determiner stripped
    ("a civic" -> "civic" but "a@b.com" keeps its 'a' — the det must be a
    whole word in the source)."""
    f = _foldish(text)
    if re.match(r"(?i)^(a|an|the)\s", text.strip()):
        parts = f.split()
        while parts and parts[0] in {"a", "an", "the"}:
            parts.pop(0)
        f = " ".join(parts)
    return f


# ---------------------------------------------------------------------------
# Word classes (local vocabulary; mirrors polarity.py/events.py conventions)
# ---------------------------------------------------------------------------

_AUX = frozenset({
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing", "done",
    "will", "would", "can", "could", "shall", "should", "may", "might",
    "must", "ought", "cannot", "cant",
    "didn", "don", "doesn", "isn", "aren", "wasn", "weren", "won", "wo",
    "couldn", "shouldn", "wouldn", "mustn", "needn", "haven", "hasn",
    "hadn", "daren", "ain", "shan", "mightn", "oughtn",
})

# adverbs + neg markers skipped between subject and trigger
_ADV = frozenset({
    "just", "really", "actually", "probably", "definitely", "also", "still",
    "already", "soon", "quite", "very", "truly", "finally", "recently",
    "simply", "nearly", "almost", "ever", "always", "often", "sometimes",
    "rarely", "seldom", "even", "now", "then", "yet", "never", "not", "no",
    "longer", "anymore", "immediately", "currently", "previously",
    "originally", "eventually", "suddenly", "quickly", "slowly", "later",
    "usually", "normally", "generally", "mostly", "mainly", "hopefully",
    "apparently", "certainly", "obviously", "clearly", "possibly",
    "perhaps", "maybe", "absolutely", "totally", "completely", "entirely",
    "fully", "pretty", "rather", "fairly", "directly", "straight", "right",
    "kind", "sort", "typically", "honestly", "frankly", "seriously",
    "basically", "literally", "genuinely", "truly", "mostly",
})

_SKIP = _AUX | _ADV  # terms allowed between subject and trigger

_NEG = frozenset({
    "not", "never", "no", "hardly", "barely", "scarcely", "without",
    "none", "nobody", "nothing", "nowhere", "neither", "nor", "cannot",
    "cant", "ain", "dont", "doesnt", "didnt", "isnt", "arent", "wasnt",
    "werent", "wont", "couldnt", "shouldnt", "wouldnt", "mustnt",
})

_MODAL = frozenset({
    "will", "would", "can", "could", "shall", "should", "may", "might",
    "must", "ought",
})

# control-ish verbs before a trigger make the clause hypothetical-ish
_CONTROL = frozenset({
    "want", "wants", "wanted", "wanting", "wish", "wishes", "wished",
    "hope", "hopes", "hoped", "hoping", "try", "tries", "tried", "trying",
    "need", "needs", "needed", "plan", "plans", "planned", "planning",
    "intend", "intends", "intended", "used", "decide", "decides",
    "decided", "expect", "expects", "expected", "manage", "manages",
    "managed", "forget", "forgets", "forgot", "remember", "remembers",
    "remembered", "choose", "chooses", "chose", "deserve", "deserves",
    "attempt", "attempts", "attempted", "wait", "waits", "waited",
    "going", "gonna", "aim", "aims", "aimed", "meaning", "meant",
})

_COND = frozenset({
    "if", "unless", "whether", "suppose", "supposing", "imagine",
    "assuming", "provided", "providing", "whenever", "lest",
})

_COG = frozenset({
    "think", "thinks", "thought", "believe", "believes", "believed",
    "guess", "guesses", "guessed", "reckon", "reckons", "seem", "seems",
    "seemed", "feel", "feels", "felt", "appear", "appears", "appeared",
    "wonder", "wonders", "wondered", "wondering", "hope", "hopes", "hoped",
    "assume", "assumes", "assumed", "figure", "figures", "figured",
    "expect", "expects", "expected", "imagine", "imagines", "imagined",
    "suppose", "supposes", "supposed", "doubt", "doubts", "doubted",
    "afraid", "sure", "certain", "uncertain", "unsure",
})

_REPORTED = frozenset({
    "said", "says", "say", "tell", "tells", "told", "wrote", "writes",
    "written", "claim", "claims", "claimed", "heard", "hear", "hears",
    "announce", "announced", "announces", "report", "reports", "reported",
    "tweet", "tweeted", "posted", "post", "posts", "comment", "commented",
    "according", "quote", "quoted", "mention", "mentions", "mentioned",
    "joke", "jokes", "joked", "joking", "kidding", "kids", "texted",
    "replied", "replies", "reply", "responded", "whispered", "shouted",
    "yelled", "asked", "ask", "asks",
})

_HEDGE_ADV = frozenset({
    "maybe", "perhaps", "possibly", "probably", "likely", "unlikely",
    "reportedly", "allegedly", "apparently", "supposedly", "seemingly",
    "arguably", "presumably", "conceivably", "technically",
})

_QAUX = frozenset({
    "do", "does", "did", "can", "could", "will", "would", "shall",
    "should", "may", "might", "must", "is", "are", "was", "were", "am",
    "have", "has", "had",
})

_QWORDS = frozenset({
    "what", "who", "whom", "whose", "how", "why", "when", "where",
    "which",
})

_COORD = frozenset({"and", "or", "but", "nor", "yet", "so"})

_PREP = frozenset({
    "to", "in", "on", "at", "by", "about", "into", "onto", "over",
    "under", "between", "through", "during", "before", "after", "around",
    "across", "toward", "towards", "upon", "within", "without", "against",
    "along", "behind", "beyond", "near", "off", "outside", "inside",
    "despite", "except", "unlike", "via", "per", "versus", "vs", "than",
    "from", "with", "for", "of", "as", "like",
})

_TEMP = frozenset({
    "yesterday", "today", "tomorrow", "tonight", "ago", "later", "soon",
    "recently", "now", "then", "afterward", "afterwards", "already",
    "currently", "finally", "here", "there", "everywhere", "somewhere",
    "anywhere", "weekend", "weekends", "weekly", "monthly", "yearly",
    "daily", "hourly", "nightly", "last", "next", "late", "early",
    "home", "abroad", "together", "alone", "online", "remotely",
    "forever",
})

_DET = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "some", "any",
    "each", "every", "all", "both", "no", "another", "either", "neither",
    "few", "many", "several", "much", "more", "most", "other", "others",
    "enough", "such",
})

_POSS = frozenset({"my", "our", "your", "his", "her", "their", "its"})

_NOM = frozenset({"i", "we", "you", "he", "she", "they", "it"})
_OPRON = frozenset({"me", "us", "him", "her", "them", "yall", "ya"})

_PART = frozenset({
    "up", "down", "out", "away", "back", "aside", "apart", "off",
    "through", "over", "around", "about",
})

_SUBORD = frozenset({
    "when", "while", "because", "since", "if", "although", "though",
    "until", "unless", "where", "whereas", "whenever", "wherever",
    "whether",
})

# object stop set: aux | neg | temp | coord | prep(minus of) | nominative
# pronouns (minus it/you — those are legitimate stated objects) |
# particles | subordinators | wh-words
_VSTOP = _AUX | _NEG | _TEMP | _COORD | (_PREP - {"of"}) | \
    frozenset({"i", "we", "he", "she", "they"}) | _PART | _SUBORD | _QWORDS

_LEAD_STRIP = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "some", "any",
    "each", "every", "all", "to",
})

_COP = frozenset({"am", "is", "are", "was", "were", "be", "been", "being"})

# blockers for subject-NP chunk collection
_NP_BLOCK = _COORD | _PREP | _TEMP | _CONTROL | _AUX | frozenset({
    "there", "here", "course", "time", "way", "lot",
})


# ---------------------------------------------------------------------------
# Family lexicons (provisional/v7-r0; deterministic, owned, extendable)
# ---------------------------------------------------------------------------

_COUNTRIES = frozenset({
    "afghanistan", "argentina", "australia", "austria", "belgium", "brazil",
    "bulgaria", "canada", "chile", "china", "colombia", "croatia", "cuba",
    "cyprus", "czechia", "denmark", "egypt", "estonia", "ethiopia",
    "finland", "france", "germany", "greece", "hungary", "iceland",
    "india", "indonesia", "iran", "iraq", "ireland", "israel", "italy",
    "japan", "jordan", "kenya", "korea", "latvia", "lebanon", "lithuania",
    "malaysia", "mexico", "morocco", "netherlands", "nigeria", "norway",
    "pakistan", "peru", "philippines", "poland", "portugal", "romania",
    "russia", "serbia", "singapore", "slovakia", "slovenia", "spain",
    "sweden", "switzerland", "taiwan", "thailand", "turkey", "ukraine",
    "uk", "usa", "america", "britain", "england", "scotland", "wales",
    "vietnam", "venezuela", "uruguay", "zealand",
})

_LANGS = frozenset({
    "english", "spanish", "french", "german", "italian", "portuguese",
    "russian", "chinese", "mandarin", "cantonese", "japanese", "korean",
    "arabic", "hindi", "urdu", "bengali", "turkish", "dutch", "swedish",
    "norwegian", "danish", "finnish", "polish", "greek", "hebrew", "thai",
    "vietnamese", "indonesian", "malay", "tagalog", "ukrainian", "czech",
    "slovak", "hungarian", "romanian", "bulgarian", "croatian", "serbian",
    "swahili", "persian", "farsi", "tamil", "telugu", "kannada",
    "malayalam", "gujarati", "punjabi", "marathi", "catalan", "basque",
    "welsh", "irish", "latin", "esperanto", "asl", "yoruba", "igbo",
    "hausa", "amharic", "burmese", "khmer", "lao", "nepali", "sinhala",
})

_DEMONYMS = frozenset({
    "american", "canadian", "mexican", "brazilian", "argentinian",
    "chilean", "colombian", "peruvian", "venezuelan", "british", "english",
    "scottish", "welsh", "irish", "french", "german", "italian", "spanish",
    "portuguese", "dutch", "belgian", "swiss", "austrian", "swedish",
    "norwegian", "danish", "finnish", "icelandic", "polish", "czech",
    "slovak", "hungarian", "romanian", "bulgarian", "greek", "croatian",
    "serbian", "slovenian", "estonian", "latvian", "lithuanian",
    "ukrainian", "russian", "turkish", "georgian", "armenian",
    "azerbaijani", "israeli", "palestinian", "lebanese", "jordanian",
    "syrian", "iraqi", "iranian", "saudi", "emirati", "egyptian",
    "moroccan", "algerian", "tunisian", "nigerian", "kenyan", "ethiopian",
    "ghanaian", "somali", "indian", "pakistani", "bangladeshi", "nepali",
    "chinese", "japanese", "korean", "vietnamese", "thai", "filipino",
    "indonesian", "malaysian", "singaporean", "australian", "kiwi",
    "haitian", "jamaican", "cuban", "dominican", "puerto",
})

_ROLE_NOUNS = frozenset({
    "engineer", "developer", "programmer", "designer", "manager",
    "teacher", "nurse", "doctor", "lawyer", "accountant", "analyst",
    "scientist", "writer", "artist", "student", "intern", "consultant",
    "freelancer", "chef", "barista", "pilot", "mechanic", "electrician",
    "plumber", "photographer", "therapist", "researcher", "professor",
    "architect", "surgeon", "dentist", "pharmacist", "musician", "actor",
    "actress", "director", "founder", "ceo", "cto", "cfo", "vp",
    "librarian", "technician", "admin", "administrator", "recruiter",
    "salesperson", "marketer", "journalist", "editor", "translator",
    "farmer", "driver", "carpenter", "waiter", "waitress", "bartender",
    "coach", "trainer", "tutor", "judge", "officer", "soldier",
    "firefighter", "policeman", "detective", "vet", "veterinarian",
    "gardener", "janitor", "cashier", "clerk", "secretary", "assistant",
    "volunteer", "attorney", "broker", "agent", "instructor", "mentor",
    "blogger", "podcaster", "streamer", "singer", "dancer", "model",
    "rancher", "fisherman", "receptionist", "paramedic", "nanny",
    "housekeeper", "economist", "psychologist", "psychiatrist", "devops",
    "sre", "qa", "pm", "em", "retiree",
})

_PETS = frozenset({
    "dog", "cat", "puppy", "kitten", "bird", "parrot", "fish", "hamster",
    "rabbit", "bunny", "guinea", "pig", "turtle", "tortoise", "snake",
    "lizard", "gecko", "ferret", "chinchilla", "rat", "mouse", "hedgehog",
    "horse", "pony", "goat", "chicken", "duck", "frog", "axolotl",
    "lab", "labrador", "retriever", "poodle", "beagle", "bulldog",
    "husky", "corgi", "shihtzu", "chihuahua", "dachshund", "pomeranian",
    "maltese", "cockapoo", "goldendoodle", "persian", "siamese", "tabby",
    "maine", "ragdoll", "bengal", "sphynx", "canary", "cockatiel",
    "parakeet", "budgie", "gerbil",
})

_KIDS = frozenset({
    "son", "daughter", "child", "kid", "kids", "children", "baby",
    "toddler", "twins", "newborn", "infant", "stepchild", "stepson",
    "stepdaughter", "boy", "girl", "boys", "girls", "sons", "daughters",
})

_STATUS_WORDS = frozenset({
    "single", "married", "engaged", "divorced", "separated", "widowed",
    "taken", "partnered", "attached", "dating", "polyamorous", "solo",
    "remarried",
})

_DIET_IDS = frozenset({
    "vegetarian", "vegan", "pescatarian", "pescetarian", "keto",
    "ketogenic", "paleo", "kosher", "halal", "carnivore", "omnivore",
    "flexitarian", "whole30", "mediterranean", "vegetarianism", "raw",
    "plantbased", "plant",
})

_CONDS = frozenset({
    "asthma", "diabetes", "hypertension", "anxiety", "depression", "adhd",
    "autism", "epilepsy", "migraine", "migraines", "arthritis",
    "insomnia", "anemia", "celiac", "coeliac", "crohns", "ibs", "ocd",
    "ptsd", "bipolar", "eczema", "psoriasis", "cholesterol",
    "prediabetes", "osteoporosis", "scoliosis", "tinnitus", "vertigo",
    "narcolepsy", "fibromyalgia", "anemia", "dyslexia", "dyspraxia",
    "endometriosis", "pcos", "lupus", "astigmatism", "add", "bpd",
    "thyroid", "hashimotos", "anemia", "gout", "apnea",
})

_SPORTS = frozenset({
    "soccer", "football", "basketball", "tennis", "golf", "hockey",
    "baseball", "volleyball", "cricket", "rugby", "swimming", "running",
    "cycling", "skiing", "snowboarding", "surfing", "skating", "boxing",
    "wrestling", "karate", "judo", "taekwondo", "fencing", "rowing",
    "sailing", "climbing", "badminton", "squash", "lacrosse", "softball",
    "marathon", "triathlon", "pickleball", "handball", "table", "archery",
    "weightlifting", "powerlifting", "bodybuilding", "gymnastics",
})

_GAMES = frozenset({
    "chess", "guitar", "piano", "violin", "drums", "flute", "cello",
    "trumpet", "saxophone", "clarinet", "ukulele", "banjo", "mandolin",
    "harmonica", "games", "videogames", "minecraft", "fortnite", "dnd",
    "dungeons", "chess", "poker", "bridge", "piano", "guitar",
})

_OS_LEX = frozenset({
    "linux", "windows", "macos", "osx", "ios", "android", "ubuntu",
    "debian", "fedora", "arch", "archlinux", "nixos", "freebsd",
    "openbsd", "chromeos", "raspbian", "mint", "manjaro", "gentoo",
    "centos", "redhat", "suse", "solaris", "unix", "pop",
})

_EDITOR_LEX = frozenset({
    "vim", "neovim", "nvim", "emacs", "vscode", "vscodium", "sublime",
    "atom", "nano", "helix", "zed", "cursor", "intellij", "pycharm",
    "webstorm", "goland", "rider", "clion", "phpstorm", "eclipse",
    "xcode", "netbeans", "notepad", "kate", "gedit", "geany", "brackets",
    "textmate", "nova", "lapce", "windsurf", "idea",
})

_PL_LEX = frozenset({
    "python", "javascript", "typescript", "rust", "go", "golang", "java",
    "kotlin", "swift", "c", "cpp", "csharp", "ruby", "php", "perl",
    "scala", "haskell", "ocaml", "elixir", "erlang", "clojure", "lua",
    "r", "julia", "dart", "zig", "fortran", "cobol", "matlab", "bash",
    "shell", "powershell", "groovy", "lisp", "scheme", "racket", "sql",
    "html", "css", "assembly", "wasm", "solidity", "verilog",
})

_TZ_LEX = frozenset({
    "pst", "pdt", "est", "edt", "cst", "cdt", "mst", "mdt", "gmt", "utc",
    "bst", "cet", "cest", "eet", "jst", "kst", "ist", "aest", "aedt",
    "nzst", "nzdt", "pt", "et", "ct", "mt", "akst", "hst", "ast",
})

_SUB_LEX = frozenset({
    "netflix", "spotify", "hulu", "disney", "amazon", "prime", "icloud",
    "youtube", "hbo", "max", "paramount", "peacock", "apple", "audible",
    "patreon", "chatgpt", "openai", "notion", "figma", "github", "adobe",
    "dropbox", "scribd", "crunchyroll", "tidal", "deezer", "substack",
    "medium", "linkedin", "strava", "duolingo",
})

_PAY_LEX = frozenset({
    "paypal", "venmo", "zelle", "cashapp", "wise", "revolut", "payoneer",
    "stripe", "applepay", "gpay", "bitcoin", "crypto", "ethereum",
})

_DEVICE_NOUNS = frozenset({
    "phone", "laptop", "computer", "tablet", "device", "pc", "mac",
    "ipad", "watch", "console", "desktop", "server", "kindle", "chromebook",
    "thinkpad", "macbook", "iphone", "android", "pixel", "galaxy",
})

_PRONOUN_LEX = frozenset({
    "he", "him", "his", "she", "her", "hers", "they", "them", "their",
    "ze", "zem", "zir", "xe", "xem", "xyr", "ae", "aer", "ve", "ver",
    "ey", "em", "eir", "fae", "faer",
})

_SUBSTANCES = frozenset({
    "gluten", "dairy", "nut", "nuts", "sugar", "caffeine", "alcohol",
    "meat", "fish", "soy", "wheat", "lactose", "fat", "shellfish",
    "peanut", "peanuts", "egg", "eggs", "dairy",
})

_DEGREE_WORDS = frozenset({
    "bachelor", "bachelors", "master", "masters", "phd", "doctorate",
    "mba", "bs", "ba", "ms", "ma", "bsc", "msc", "associate",
    "associates", "jd", "md", "edd", "dphil", "llm", "degree",
})

_STREET_WORDS = frozenset({
    "street", "st", "avenue", "ave", "road", "rd", "lane", "ln", "drive",
    "dr", "boulevard", "blvd", "way", "court", "ct", "place", "pl",
    "square", "apt", "apartment", "suite", "unit", "floor", "block",
    "broadway", "highway", "hwy", "terrace", "circle",
})

_MED_TAIL = frozenset({
    "for", "daily", "every", "day", "night", "morning", "evening",
    "weekly", "monthly", "regularly", "as", "twice", "once",
})

# common generic/brand drug names — lets np_then accept "i take metformin"
# with no schedule tail while still rejecting "i take the bus".
_MEDS = frozenset({
    "metformin", "insulin", "aspirin", "ibuprofen", "advil", "tylenol",
    "acetaminophen", "naproxen", "aleve", "lexapro", "escitalopram",
    "sertraline", "zoloft", "prozac", "fluoxetine", "adderall",
    "vyvanse", "ritalin", "atorvastatin", "lipitor", "simvastatin",
    "lisinopril", "amlodipine", "metoprolol", "levothyroxine",
    "synthroid", "omeprazole", "prilosec", "albuterol", "ventolin",
    "prednisone", "warfarin", "coumadin", "xanax", "alprazolam",
    "zyrtec", "cetirizine", "claritin", "loratadine", "benadryl",
    "diphenhydramine", "ozempic", "semaglutide", "wegovy", "humira",
    "amoxicillin", "penicillin", "azithromycin", "ciprofloxacin",
    "gabapentin", "tramadol", "hydrochlorothiazide", "losartan",
    "pantoprazole", "melatonin", "vitamins", "vitamin",
})

_COLOR_LEX = frozenset({
    "red", "blue", "green", "black", "white", "silver", "gray", "grey",
    "yellow", "orange", "purple", "pink", "brown", "gold", "beige",
    "maroon", "navy", "teal", "turquoise", "violet", "magenta", "cyan",
})

_SCHED_ALL = frozenset({
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "mondays", "tuesdays", "wednesdays", "thursdays", "fridays",
    "saturdays", "sundays", "morning", "mornings", "afternoon",
    "afternoons", "evening", "evenings", "night", "nights", "weekend",
    "weekends", "weekday", "weekdays", "day", "days", "week", "weeks",
    "month", "months", "year", "years", "daily", "weekly", "monthly",
    "yearly", "nightly", "hourly",
})

_SCHED_PLURAL = frozenset({
    "mondays", "tuesdays", "wednesdays", "thursdays", "fridays",
    "saturdays", "sundays", "mornings", "afternoons", "evenings",
    "nights", "weekends", "weekdays", "days", "nights",
})

_NUMWORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}

_WORKOUT = frozenset({
    "yoga", "pilates", "crossfit", "zumba", "cardio", "lifting",
    "spinning", "aerobics", "judo", "karate", "taekwondo", "gymnastics",
    "hiit", "barre", "calisthenics", "stretching", "spin", "barre",
})

_VERBISH = frozenset({
    "be", "do", "have", "get", "go", "sleep", "eat", "work", "rest",
    "cry", "run", "walk", "sit", "stand", "dance", "sing", "study",
    "shower", "nap", "play", "watch", "read", "write", "talk", "speak",
    "start", "stop", "try", "make", "take", "give", "buy", "pay", "cook",
    "clean", "wash", "call", "text", "email", "meet", "help", "look",
    "feel", "think", "know", "say", "tell", "ask", "use", "find", "leave",
    "stay", "come", "see", "move", "fly", "travel", "drive", "visit",
    "learn", "practice", "finish", "begin", "continue", "keep",
})

_VALUE_BLOCK = frozenset({
    "it", "this", "that", "something", "anything", "everything", "stuff",
    "things", "nothing", "somewhere", "anywhere",
})

# favorite_* category -> state family (unmapped cats emit no state fact)
_FAV_CAT = {
    "food": "favorite_food", "meal": "favorite_food",
    "dish": "favorite_food", "cuisine": "favorite_food",
    "restaurant": "favorite_food", "snack": "favorite_food",
    "drink": "favorite_food", "dessert": "favorite_food",
    "color": "favorite_color", "colour": "favorite_color",
    "music": "favorite_music", "song": "favorite_music",
    "band": "favorite_music", "artist": "favorite_music",
    "singer": "favorite_music", "album": "favorite_music",
    "genre": "favorite_music", "musician": "favorite_music",
    "book": "favorite_book", "novel": "favorite_book",
    "author": "favorite_book", "poet": "favorite_book",
    "movie": "favorite_movie", "film": "favorite_movie",
    "documentary": "favorite_movie", "show": "favorite_movie",
    "series": "favorite_movie", "anime": "favorite_movie",
    "actor": "favorite_movie", "actress": "favorite_movie",
    "sport": "sport", "team": "team", "hobby": "hobbies",
    "game": "hobbies",
}

_ACCUMULATE = frozenset({
    "pets", "pet_names", "children", "hobbies", "languages", "allergies",
    "health_condition", "medication", "sport", "programming_language",
    "subscription", "diet", "goal_current", "schedule_regular", "device",
    "workout_routine", "degree", "plan_upcoming", "travel_upcoming",
})

_NUMERIC_FAMILIES = frozenset({"age"})


# ---------------------------------------------------------------------------
# Extraction context
# ---------------------------------------------------------------------------

_QUOTE_PAIR_RE = re.compile(
    r'"[^"\n]+"|\u201c[^\u201d\n]+\u201d|\u2018[^\u2019\n]+\u2019'
    r"|(?<![\w'\u2019])'[^'\n]+'(?![\w'\u2019])"
)

_CLITIC_GAP_RE = re.compile(
    r"^\s*(?:['\u2019](s|d|m|re|ll|ve)|['\u2019])\s*$", re.IGNORECASE
)

_RECOVER_RE = re.compile(
    r"\b(i|we|you|yall|he|she|it|they|me|us|her|him|them)"
    r"\s*(?:['\u2019]\s*(m|d|ll|re|ve|s))?\s*$",
    re.IGNORECASE,
)


def _utf8_offsets(text: str) -> list[int]:
    """Cumulative UTF-8 byte offsets; lazy-imports normalize's when present."""
    try:  # preferred: share the wave-A helper
        from .normalize import utf8_offsets  # type: ignore

        return utf8_offsets(text)
    except Exception:
        out = [0]
        n = 0
        for ch in text:
            n += len(ch.encode("utf-8"))
            out.append(n)
        return out


@dataclass
class _Ctx:
    raw_bytes: Optional[bytes]
    hay: Optional[str]              # text aligned (or not) with term bytes
    hay_bytes: Optional[bytes]
    hay_ok: bool                    # hay verified byte-aligned with terms
    canon: Optional[Callable[[str], str]]
    quotes: tuple[tuple[int, int], ...] = ()


def _stream(norm: NormAnalysis) -> list[NormTerm]:
    """Merged, byte-sorted stream: text-channel terms + identifier terms."""
    terms = [t for t in norm.terms if t.channel == "text"]
    terms.extend(getattr(norm, "identifiers", ()) or ())
    terms.sort(key=lambda t: (t.byte_start, t.byte_end, t.term))
    return terms


def _mk_ctx(norm: NormAnalysis, raw_text: Optional[str],
            canon_fn: Optional[Callable[[str], str]]) -> _Ctx:
    raw_bytes: Optional[bytes] = None
    if raw_text is not None:
        try:
            raw_bytes = raw_text.encode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            raw_bytes = None
    hay: Optional[str] = raw_text if raw_bytes is not None else getattr(
        norm, "text", None)
    hay_bytes: Optional[bytes] = None
    if hay is not None:
        try:
            hay_bytes = hay.encode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            hay_bytes = None
    ctx = _Ctx(raw_bytes, hay, hay_bytes, False, canon_fn)
    # alignment probe: a sample of terms must slice cleanly and fold-match.
    # Clitic remnants ("am" covering 'm, "not" covering n't) are tolerated.
    terms = _stream(norm)
    if hay_bytes is not None and terms:
        probes = [terms[0], terms[len(terms) // 2], terms[-1]]
        ok = True
        checked = 0
        for t in probes:
            if t.byte_end > len(hay_bytes):
                ok = False
                break
            try:
                piece = hay_bytes[t.byte_start:t.byte_end].decode("utf-8")
            except UnicodeDecodeError:
                ok = False
                break
            if t.channel == "identifier":
                ok = piece == t.term
            else:
                ok = _foldish(piece) == t.term or t.term in _SKIP
            checked += 1
            if not ok:
                break
        ctx.hay_ok = ok if checked else False
    if ctx.hay_ok and hay is not None and hay_bytes is not None:
        off = _utf8_offsets(hay)
        spans = []
        for m in _QUOTE_PAIR_RE.finditer(hay):
            spans.append((off[m.start()], off[m.end()]))
        ctx.quotes = tuple(spans)
    if canon_fn is None:
        canon_fn = _default_canon()
        ctx.canon = canon_fn
    return ctx


def _default_canon() -> Optional[Callable[[str], str]]:
    try:
        from .entities_v2 import canon  # lazy sibling import

        return canon
    except Exception:
        return None


def _slice(ctx: _Ctx, s: int, e: int) -> Optional[str]:
    if ctx.hay_bytes is None or not (0 <= s < e <= len(ctx.hay_bytes)):
        return None
    try:
        return ctx.hay_bytes[s:e].decode("utf-8")
    except UnicodeDecodeError:
        return None


def _surface(ctx: _Ctx, ts: Sequence[NormTerm]) -> str:
    """Best surface for a term span: verified slice else folded join."""
    if not ts:
        return ""
    joined = " ".join(t.term for t in ts)
    if ctx.hay_ok:
        piece = _slice(ctx, ts[0].byte_start, ts[-1].byte_end)
        if piece is not None and _foldish(piece) == _foldish(joined):
            return piece
    return joined


def _term_surface(ctx: _Ctx, t: NormTerm) -> str:
    if t.channel == "identifier":
        return t.term  # exact bytes by construction
    return _surface(ctx, (t,))


def _is_cap_surface(ctx: _Ctx, ts: Sequence[NormTerm]) -> bool:
    """True iff some content term's surface starts uppercase."""
    if not ctx.hay_ok:
        return False
    for t in ts:
        piece = _slice(ctx, t.byte_start, t.byte_end)
        if piece:
            for ch in piece:
                if ch.isalpha():
                    if ch.isupper():
                        return True
                    break
    return False


def _gap_clitic(ctx: _Ctx, prev_end: int, next_start: int) -> str:
    """Return the clitic letter bridging a small byte gap ('s','d','m','re',
    'll','ve') or "" when the gap is punctuation/plain space."""
    gap = next_start - prev_end
    if gap < 2 or gap > 6 or not ctx.hay_ok or ctx.hay_bytes is None:
        return ""
    piece = _slice(ctx, prev_end, next_start)
    if piece is None:
        return ""
    m = _CLITIC_GAP_RE.match(piece)
    if not m:
        return ""
    return (m.group(1) or "s").lower()


def _linked(ctx: _Ctx, terms: Sequence[NormTerm], i: int) -> bool:
    """terms[i-1] and terms[i] are clause-linked (gap <2 or clitic bridge)."""
    if i <= 0:
        return False
    gap = terms[i].byte_start - terms[i - 1].byte_end
    if gap < 2:
        return True
    return _gap_clitic(ctx, terms[i - 1].byte_end, terms[i].byte_start) != ""


def _clause_bounds(ctx: _Ctx, terms: Sequence[NormTerm], i: int
                   ) -> tuple[int, int]:
    lo = i
    while lo > 0 and _linked(ctx, terms, lo):
        lo -= 1
    hi = i
    while hi + 1 < len(terms) and _linked(ctx, terms, hi + 1):
        hi += 1
    return lo, hi + 1


# ---------------------------------------------------------------------------
# Subject resolution
# ---------------------------------------------------------------------------


@dataclass
class _Subj:
    canon: Optional[str]
    form: str                       # speaker | name | second | pronoun | desc
    span: Optional[tuple[int, int]]
    window: tuple[NormTerm, ...]
    clitic: str = ""                # 'd/'ll/'ve/'s/'m seen after subject
    neg: Optional[NormTerm] = None


def _collectable(t: NormTerm) -> bool:
    w = t.term
    return not (
        w in _SKIP or w in _NP_BLOCK or w in _DET or w in _POSS
        or w in _QWORDS
    )


def _subject(ctx: _Ctx, terms: Sequence[NormTerm], i: int, lo: int,
             speaker: Optional[str]) -> Optional[_Subj]:
    """Resolve the subject left of index ``i`` (bounded by clause ``lo``)."""
    j = i - 1
    window: list[NormTerm] = []
    clitic = ""
    # window: aux/adv/neg terms adjacent to the trigger
    while j >= lo:
        if not _linked(ctx, terms, j + 1):
            break
        if terms[j].term in _SKIP:
            window.insert(0, terms[j])
            j -= 1
            continue
        break
    # chunk: contiguous collectable terms (the subject NP)
    chunk: list[NormTerm] = []
    while j >= lo:
        prev = j + 1
        cl = _gap_clitic(ctx, terms[j].byte_end, terms[prev].byte_start)
        if cl:
            clitic = cl  # 's/'d/'m between candidate subject and the rest
        elif not _linked(ctx, terms, prev):
            break
        t = terms[j]
        if not _collectable(t) or t.channel == "identifier":
            break
        chunk.insert(0, t)
        j -= 1
        if len(chunk) >= 4:
            break
    neg = next((t for t in window if t.term in _NEG), None)
    if not chunk:
        # clitic-eaten subject recovery: "i'm X" / "i'd X" drop the "i" term.
        if ctx.hay_ok and ctx.hay_bytes is not None and j + 1 < len(terms):
            pos = terms[j + 1].byte_start
            piece = _slice(ctx, 0, pos)
            if piece:
                m = _RECOVER_RE.search(piece)
                if m:
                    pron = m.group(1).lower()
                    cl = (m.group(2) or "").lower()
                    span = (m.start(1), m.end(1))
                    # only adjacent: nothing between match end and pos
                    if pron in ("i", "we"):
                        return _Subj(speaker, "speaker", span,
                                     tuple(window), cl, neg)
                    if pron in ("you", "yall"):
                        return _Subj(None, "second", span,
                                     tuple(window), cl, neg)
                    return _Subj(None, "pronoun", span,
                                 tuple(window), cl, neg)
        return None
    w0 = chunk[0].term
    if len(chunk) == 1 and w0 in ("i", "we"):
        return _Subj(speaker, "speaker", (chunk[0].byte_start,
                                          chunk[0].byte_end),
                     tuple(window), clitic, neg)
    if w0 in ("you", "yall") or (len(chunk) == 1 and w0 in ("your",)):
        return _Subj(None, "second", (chunk[0].byte_start,
                                      chunk[-1].byte_end),
                     tuple(window), clitic, neg)
    if w0 in _NOM or w0 in _OPRON:
        return _Subj(None, "pronoun", (chunk[0].byte_start,
                                       chunk[-1].byte_end),
                     tuple(window), clitic, neg)
    if w0 in _DET or w0 in _POSS:
        return _Subj(None, "desc", (chunk[0].byte_start, chunk[-1].byte_end),
                     tuple(window), clitic, neg)
    # named NP -> capitalized-name check
    if ctx.canon is None:
        COUNTERS["canon_unavailable"] += 1
        return None
    if not _is_cap_surface(ctx, chunk):
        COUNTERS["name_uncapitalized"] += 1
        return None
    surface = _surface(ctx, chunk)
    canon = ctx.canon(surface)
    if not canon:
        return None
    return _Subj(canon, "name", (chunk[0].byte_start, chunk[-1].byte_end),
                 tuple(window), clitic, neg)


def _possessor(ctx: _Ctx, terms: Sequence[NormTerm], i: int, lo: int,
               speaker: Optional[str]) -> Optional[_Subj]:
    """Possessor directly before an attr-noun: 'my X is' / 'Maria's X is'."""
    j = i - 1
    if j < lo:
        return None
    t = terms[j]
    if t.term in ("my", "our") and _linked(ctx, terms, i):
        return _Subj(speaker, "speaker", (t.byte_start, t.byte_end), (), "")
    # name + 's-gap
    cl = _gap_clitic(ctx, t.byte_end, terms[i].byte_start)
    if cl == "s" and _collectable(t) and t.term not in _POSS \
            and t.term not in _NOM and t.term not in _OPRON:
        if ctx.canon is not None and _is_cap_surface(ctx, (t,)):
            return _Subj(ctx.canon(_term_surface(ctx, t)), "name",
                         (t.byte_start, t.byte_end), (), "s")
        if not (ctx.canon is not None and _is_cap_surface(ctx, (t,))):
            COUNTERS["name_uncapitalized"] += 1
        return None
    return None


# ---------------------------------------------------------------------------
# Guards (§32.12 exclusion classes, adapted to state facts too)
# ---------------------------------------------------------------------------


def _guarded(ctx: _Ctx, terms: Sequence[NormTerm], lo: int, hi: int,
             subj: Optional[_Subj], head_i: int, allow_d: bool,
             ) -> Optional[str]:
    """Return a skip reason or None.  Negation is NOT a skip — caller decides."""
    clause = terms[lo:hi]
    # question form: clause-initial wh-word always; clause-initial aux unless
    # it is a clitic remnant ("i'm X" — the eaten pronoun precedes 'am').
    if clause and clause[0].term in _QWORDS:
        return "question"
    if clause and clause[0].term in _QAUX:
        remnant = False
        if ctx.hay_ok:
            piece = _slice(ctx, 0, clause[0].byte_start) or ""
            remnant = _RECOVER_RE.search(piece) is not None
        if not remnant:
            return "question"
    # '?' immediately after the clause's last term (aligned text only)
    if ctx.hay_ok and ctx.hay_bytes is not None:
        k = terms[hi - 1].byte_end
        while k < len(ctx.hay_bytes) and ctx.hay_bytes[k:k + 1].isspace():
            k += 1
        if k < len(ctx.hay_bytes) and ctx.hay_bytes[k:k + 1] == b"?":
            return "question"
    # left-context guards: conditional / cognition / reported / hedge terms
    subj_start = subj.span[0] if subj is not None and subj.span else \
        terms[head_i].byte_start
    for t in clause:
        if t.byte_start >= subj_start:
            break
        w = t.term
        if w in _COND:
            return "conditional"
        if w in _COG:
            return "hedged"
        if w in _REPORTED:
            return "reported"
        if w in _HEDGE_ADV:
            return "hedged"
    # window guards: modal / control / 'd-clitic.  can/could paired with a
    # negation is the "can't X" idiom — negative, not hypothetical.
    if subj is not None:
        has_neg = any(t.term in _NEG for t in subj.window)
        for t in subj.window:
            if t.term in _HEDGE_ADV:
                return "hedged"
            if t.term in ("can", "could", "cannot", "cant") and has_neg:
                continue
            if t.term in _MODAL or t.term in _CONTROL:
                return "hypothetical"
        if subj.clitic in ("d", "ll") and not allow_d:
            return "hypothetical"
    # quoted span containing the trigger
    hs = terms[head_i].byte_start
    for qs, qe in ctx.quotes:
        if qs <= hs < qe:
            return "quoted"
    # postposed report tag: "…, she said" — the clause is a quotation
    # attributed to the tag's subject, not this unit's speaker.  A
    # coordinated clause ("i love pizza and she said hi") stays linked and
    # never reaches this check.
    if hi + 1 < len(terms) and terms[hi + 1].term in _REPORTED:
        tag = terms[hi]
        if tag.term in {"i", "we", "you", "he", "she", "they"} or \
                _is_cap_surface(ctx, (tag,)) or \
                tag.term not in (_AUX | _ADV | _PREP | _COORD | _QWORDS |
                                 _NEG | _TEMP | _PART | _SUBORD | _COND |
                                 _REPORTED):
            return "reported"
    return None


# ---------------------------------------------------------------------------
# Value / object collection
# ---------------------------------------------------------------------------


def _np(ctx: _Ctx, terms: Sequence[NormTerm], k: int, hi: int,
        extra_stop: frozenset = frozenset(),
        lead_strip: frozenset = _LEAD_STRIP,
        stop: frozenset = _VSTOP) -> tuple[list[NormTerm], int]:
    """Collect an NP-ish run of terms starting at k. Returns (terms, next)."""
    while k < hi and _linked(ctx, terms, k) and \
            terms[k].term in lead_strip:
        k += 1
    out: list[NormTerm] = []
    while k < hi and _linked(ctx, terms, k):
        w = terms[k].term
        if w in extra_stop:
            break
        if w in ("and", "or"):
            if k + 1 < hi and _linked(ctx, terms, k + 1) and \
                    _obj_ok(terms[k + 1], extra_stop, stop):
                out.append(terms[k])
                k += 1
                continue
            break
        if w == "of":
            if k + 1 < hi and _linked(ctx, terms, k + 1) and \
                    _obj_ok(terms[k + 1], extra_stop, stop):
                out.append(terms[k])
                k += 1
                continue
            break
        if not _obj_ok(terms[k], extra_stop, stop):
            break
        out.append(terms[k])
        k += 1
        if len(out) >= 8:
            break
    # strip trailing coordinators/preps that snuck in
    while out and out[-1].term in ("and", "or", "of"):
        out.pop()
    return out, k


def _obj_ok(t: NormTerm, extra_stop: frozenset,
            stop: frozenset = _VSTOP) -> bool:
    return t.term not in stop and t.term not in extra_stop


# ---------------------------------------------------------------------------
# Rule tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Trig:
    """Trigger-class rule: subject + window + trigger + literal chain + value."""
    rid: str
    fam: str                    # "" => preference rule
    verbs: frozenset
    lit: tuple = ()             # required literal sets after trigger
    opt: tuple = ()             # optional literal sets
    vspec: str = "np"           # np|np_cap|np_terms|np_any|np_from|np_then|num|self
    arg: frozenset = frozenset()
    reject: frozenset = frozenset()   # next term after lits must NOT be in set
    need_cop: bool = False      # window must contain copula or 's/'m/'re gap
    need_neg: bool = False      # window must contain a negation
    need: frozenset = frozenset()     # required window term(s)
    strength: str = ""
    pol: str = "positive"
    allow_d: bool = False
    vblock: frozenset = frozenset()
    skiplead: frozenset = frozenset()
    gate: str = ""              # geo|sport|pronoun|phone_device|going
    emits: tuple = ()           # extra (family, "trigger"|"value") fixed facts


@dataclass(frozen=True)
class _Poss:
    """Possessor rule: {my|our|Name's} + noun chain + copula/'s + value."""
    rid: str
    fam: str
    chain: tuple                # noun chain after possessor
    copula: bool = True
    lit: tuple = ()             # literals between copula and value
    opt: tuple = ()
    vspec: str = "np"
    arg: frozenset = frozenset()
    gate: str = ""              # cap|digit|phone_device|lang_pl|petname|nodigit
    skiplead: frozenset = frozenset()
    emits: tuple = ()


_PREF_RULES: tuple[_Trig, ...] = (
    # --- constraints (strongest; declared first) ---
    _Trig("pref/allergic", "", frozenset({"allergic"}), lit=(frozenset({"to"}),),
          need_cop=True, strength="constraint", pol="negative"),
    _Trig("pref/dont_eat", "",
          frozenset({"eat", "eats", "ate", "eaten", "eating", "drink",
                     "drinks", "drank", "drinking", "consume", "consumes",
                     "touch", "touches"}),
          need_neg=True, strength="constraint", pol="negative"),
    _Trig("pref/diet_id", "", _DIET_IDS, need_cop=True, vspec="self",
          strength="constraint", pol="positive"),
    # --- favorite / comparative ---
    # 'my favorite X is Y' and 'X is my favorite' are matched by the
    # possessive machinery below (favorite head needs possessor handling).
    # --- love/hate ---
    _Trig("pref/love", "",
          frozenset({"love", "loves", "loved", "loving", "adore", "adores",
                     "adored", "adoring"}),
          strength="love_hate", pol="positive"),
    _Trig("pref/hate", "",
          frozenset({"hate", "hates", "hated", "hating", "detest",
                     "detests", "detested"}),
          strength="love_hate", pol="negative"),
    # --- like/dislike ---
    _Trig("pref/like", "",
          frozenset({"like", "likes", "liked", "liking", "enjoy", "enjoys",
                     "enjoyed", "enjoying", "prefer", "prefers",
                     "preferred", "preferring", "fancy", "fancies",
                     "fancied", "dig", "digs", "miss", "misses", "missed"}),
          strength="like_dislike", pol="positive"),
    _Trig("pref/dislike", "",
          frozenset({"dislike", "dislikes", "disliked", "disliking"}),
          strength="like_dislike", pol="negative"),
    _Trig("pref/avoid", "",
          frozenset({"avoid", "avoids", "avoided", "avoiding"}),
          strength="like_dislike", pol="negative"),
    _Trig("pref/stand", "", frozenset({"stand", "stands", "stood"}),
          need=frozenset({"can", "could", "cannot", "cant"}),
          need_neg=True, strength="like_dislike", pol="negative"),
    # --- habitual markers (verb+object follow the marker) ---
    _Trig("pref/habit", "",
          frozenset({"usually", "always", "typically", "normally",
                     "generally", "often"}),
          vspec="habit", strength="habitual", pol="positive"),
    _Trig("pref/habit_neg", "", frozenset({"never", "rarely", "seldom"}),
          vspec="habit", strength="habitual", pol="negative"),
    _Trig("pref/tend", "", frozenset({"tend", "tends", "tended"}),
          lit=(frozenset({"to"}),), vspec="habit",
          strength="habitual", pol="positive"),
)


@dataclass(frozen=True)
class _Seq:
    """Sequence rule: anchored term + backward pre-slots + forward slots."""
    rid: str
    fam: str
    anchor: frozenset
    pre: tuple = ()             # backward slots: ("w",set)|("ow",set)|
                                # ("adv*",)|("cop",)|("s",)
    post: tuple = ()            # forward slots after anchor
    strength: str = ""
    pol: str = "positive"
    allow_d: bool = False
    vblock: frozenset = frozenset()
    gate: str = ""
    emits: tuple = ()


def _w(s): return ("w", frozenset(s))


def _ow(s): return ("ow", frozenset(s))


_PREF_SEQ: tuple[_Seq, ...] = (
    _Seq("pref/fan", "", frozenset({"fan"}),
         pre=(("s",), ("cop",), ("adv*",), _w({"a", "an", "the"}),
              ("owx",)),
         post=(_w({"of"}), ("valu", "np", ())),
         strength="like_dislike", pol="positive"),
    _Seq("pref/into", "", frozenset({"into"}),
         pre=(("s",), ("cop",), ("adv*",)),
         post=(("valu", "np", ()),),
         strength="like_dislike", pol="positive"),
    # "i can't get enough of X" — the neg is idiom-internal, positive pref
    _Seq("pref/enough", "", frozenset({"enough"}),
         pre=(("s",), _w({"can", "could", "cannot", "cant"}),
              _w({"not", "never"}), _w({"get", "gets", "getting", "got"})),
         post=(_w({"of"}), ("valu", "np", ())),
         strength="love_hate", pol="positive"),
    _Seq("pref/rather", "", frozenset({"rather"}),
         pre=(("s",),),
         post=(("valu", "np", ()), _w({"than"}), ("val2", "np", ())),
         strength="favorite", pol="positive", allow_d=True),
    # "i'm lactose intolerant" / "i'm gluten free" — substance + trigger
    _Seq("pref/intolerant", "", frozenset({"intolerant"}),
         pre=(("s",), ("cop",), ("adv*",), ("val",)),
         post=(), strength="constraint", pol="negative",
         gate="substance", emits=(("value", "prev+anchor"),)),
    _Seq("pref/free", "", frozenset({"free"}),
         pre=(("s",), ("cop",), ("adv*",), ("val",)),
         post=(), strength="constraint", pol="negative",
         gate="substance", emits=(("value", "prev+anchor"),)),
)


# favorite: 'my favorite <cat?> is <v>' / '<v> is my favorite <cat?>'
# handled as possessive/reverse special cases in code.

_STATE_TRIG_RULES: tuple[_Trig, ...] = (
    _Trig("state/live_in", "home_city",
          frozenset({"live", "lives", "lived", "living", "reside",
                     "resides", "resided"}),
          lit=(frozenset({"in"}),), gate="geo"),
    _Trig("state/move_to", "home_city",
          frozenset({"move", "moves", "moved", "moving", "relocate",
                     "relocates", "relocated"}),
          lit=(frozenset({"to"}),), gate="geo"),
    _Trig("state/from", "home_city", frozenset({"from"}),
          need_cop=True, gate="geo"),
    _Trig("state/live_at", "address",
          frozenset({"live", "lives", "lived", "reside", "resides"}),
          lit=(frozenset({"at", "on"}),), vspec="np_any",
          arg=_STREET_WORDS),
    _Trig("state/work_at", "employer",
          frozenset({"work", "works", "worked", "working"}),
          lit=(frozenset({"at", "for"}),)),
    _Trig("state/work_as", "job_title",
          frozenset({"work", "works", "working"}),
          lit=(frozenset({"as"}),), vspec="np_any", arg=_ROLE_NOUNS),
    _Trig("state/work_on", "project_current",
          frozenset({"work", "works", "working"}),
          lit=(frozenset({"on"}),)),
    _Trig("state/work_out", "workout_routine",
          frozenset({"work", "works", "worked"}),
          lit=(frozenset({"out"}),), vspec="tail_or_self"),
    _Trig("state/joined", "employer",
          frozenset({"join", "joins", "joined", "joining"}),
          vspec="np_cap"),
    _Trig("state/report_to", "manager",
          frozenset({"report", "reports", "reported", "reporting"}),
          lit=(frozenset({"to"}),), vspec="np_cap"),
    _Trig("state/married_to", "partner", frozenset({"married"}),
          lit=(frozenset({"to"}),), vspec="np_cap", need_cop=True,
          emits=(("relationship_status", "trigger"),)),
    _Trig("state/engaged_to", "partner", frozenset({"engaged"}),
          lit=(frozenset({"to"}),), vspec="np_cap", need_cop=True,
          emits=(("relationship_status", "trigger"),)),
    _Trig("state/dating", "partner", frozenset({"dating", "date", "dates"}),
          vspec="np_cap", emits=(("relationship_status", "trigger"),)),
    _Trig("state/study_at", "school",
          frozenset({"study", "studies", "studied", "studying"}),
          lit=(frozenset({"at"}),), vspec="np_cap"),
    _Trig("state/study", "major",
          frozenset({"study", "studies", "studied", "studying"}),
          reject=frozenset({"at", "in"})),
    _Trig("state/major_in", "major",
          frozenset({"major", "majors", "majored", "majoring"}),
          lit=(frozenset({"in"}),)),
    _Trig("state/attend", "school",
          frozenset({"attend", "attends", "attended", "attending"}),
          vspec="np_cap"),
    _Trig("state/speak", "languages",
          frozenset({"speak", "speaks", "spoke", "spoken"}),
          vspec="np_terms", arg=_LANGS),
    _Trig("state/learn_lang", "languages",
          frozenset({"learn", "learns", "learning", "learned", "practice",
                     "practicing", "studying"}),
          vspec="np_any", arg=_LANGS),
    _Trig("state/fluent", "languages", frozenset({"fluent"}),
          lit=(frozenset({"in"}),), vspec="np_any", arg=_LANGS,
          need_cop=True),
    _Trig("state/drive", "car",
          frozenset({"drive", "drives", "drove", "driving"}),
          lit=(frozenset({"a", "an", "the", "my", "new", "used"}),),
          skiplead=_COLOR_LEX),
    _Trig("state/drive_to", "travel_upcoming",
          frozenset({"fly", "flying", "travel", "traveling", "travelling",
                     "driving", "head", "heading", "visiting"}),
          lit=(frozenset({"to"}),), vspec="np_cap", need_cop=True),
    _Trig("state/going_to", "travel_upcoming",
          frozenset({"going"}),
          lit=(frozenset({"to"}),), gate="going", need_cop=True),
    _Trig("state/plan_to", "plan_upcoming",
          frozenset({"plan", "plans", "planning", "planned"}),
          lit=(frozenset({"to", "on"}),)),
    _Trig("state/want_to", "goal_current",
          frozenset({"want", "wants", "hope", "hopes", "aim", "aims",
                     "intend", "intends", "trying", "try", "tries"}),
          lit=(frozenset({"to"}),)),
    _Trig("state/dating_status", "relationship_status",
          frozenset({"dating"}), vspec="self", need_cop=True),
    _Trig("state/take_med", "medication",
          frozenset({"take", "takes", "took", "taking"}),
          vspec="np_then", arg=_MED_TAIL, vblock=_VALUE_BLOCK),
    _Trig("state/prescribed", "medication", frozenset({"prescribed"}),
          need_cop=True),
    _Trig("state/have_pet", "pets",
          frozenset({"have", "has", "got", "adopt", "adopted", "rescue",
                     "rescued", "foster", "fosters", "own", "owns",
                     "owned"}),
          vspec="np_pet", arg=_PETS),
    _Trig("state/have_kid", "children",
          frozenset({"have", "has", "got", "expect", "expecting"}),
          vspec="np_any", arg=_KIDS),
    _Trig("state/have_cond", "health_condition",
          frozenset({"have", "has", "got", "developed", "suffer"}),
          vspec="np_any", arg=_CONDS),
    _Trig("state/suffer", "health_condition", frozenset({"suffer",
                                                         "suffers",
                                                         "suffering"}),
          lit=(frozenset({"from"}),), vspec="np_any", arg=_CONDS),
    _Trig("state/diagnosed", "health_condition", frozenset({"diagnosed"}),
          lit=(frozenset({"with"}),), need_cop=True),
    _Trig("state/allergic", "allergies", frozenset({"allergic"}),
          lit=(frozenset({"to"}),), need_cop=True),
    _Trig("state/diet_id", "diet", _DIET_IDS, vspec="self", need_cop=True),
    _Trig("state/status_id", "relationship_status", _STATUS_WORDS,
          vspec="self", need_cop=True),
    _Trig("state/nationality_id", "nationality", _DEMONYMS, vspec="self",
          need_cop=True),
    _Trig("state/play", "hobbies",
          frozenset({"play", "plays", "played", "playing"}),
          vspec="np", gate="sport"),
    _Trig("state/do_workout", "workout_routine",
          frozenset({"do", "does", "doing"}), vspec="np_any",
          arg=_WORKOUT),
    _Trig("state/run_os", "os", frozenset({"run", "runs", "running"}),
          vspec="np_any", arg=_OS_LEX),
    _Trig("state/run_workout", "workout_routine",
          frozenset({"run", "runs", "jog", "jogs", "jogging", "lift",
                     "lifts", "swim", "swims", "swimming", "cycle",
                     "cycles", "cycling", "hike", "hikes", "hiking",
                     "climb", "climbs", "row", "rows", "boxing"}),
          vspec="np_opt", vblock=_SCHED_ALL),
    _Trig("state/use_editor", "editor",
          frozenset({"use", "uses", "used", "using"}),
          vspec="np_any", arg=_EDITOR_LEX),
    _Trig("state/use_os", "os",
          frozenset({"use", "uses", "used", "using"}),
          vspec="np_any", arg=_OS_LEX),
    _Trig("state/use_pl", "programming_language",
          frozenset({"use", "uses", "used", "using"}),
          vspec="np_any", arg=_PL_LEX),
    _Trig("state/use_pay", "bank_or_payment",
          frozenset({"use", "uses", "used", "using"}),
          vspec="np_any", arg=_PAY_LEX),
    _Trig("state/code_in", "programming_language",
          frozenset({"code", "codes", "coding", "program", "programs",
                     "programming", "develop", "developing", "hack",
                     "hacking"}),
          lit=(frozenset({"in"}),), vspec="np_any", arg=_PL_LEX),
    _Trig("state/write_pl", "programming_language",
          frozenset({"write", "writes", "wrote", "writing"}),
          vspec="np_any", arg=_PL_LEX),
    _Trig("state/subscribe", "subscription",
          frozenset({"subscribe", "subscribes", "subscribed",
                     "subscribing"}),
          lit=(frozenset({"to"}),)),
    _Trig("state/pay_for", "subscription",
          frozenset({"pay", "pays", "paid"}),
          lit=(frozenset({"for"}),), vspec="np_any", arg=_SUB_LEX),
    _Trig("state/bank_with", "bank_or_payment",
          frozenset({"bank", "banks", "banked"}),
          lit=(frozenset({"with"}),)),
    _Trig("state/reading", "reading_current",
          frozenset({"reading", "rereading"}), need_cop=True),
    _Trig("state/watching", "show_current",
          frozenset({"watching", "rewatching", "binging", "bingeing",
                     "binged", "streaming"}),
          need_cop=True, vblock=_OPRON | frozenset({"you", "me", "us"})),
    _Trig("state/working_on", "project_current",
          frozenset({"building", "hacking"}), vspec="np"),
    _Trig("state/collect", "hobbies",
          frozenset({"collect", "collects", "collected", "collecting"})),
    _Trig("state/born", "birthday", frozenset({"born"}),
          opt=(frozenset({"in", "on"}),)),
    _Trig("state/turned", "age", frozenset({"turned", "turn", "turns"}),
          vspec="num"),
)


# possessive state rules: possessor + noun chain + [copula] + [lit] + value
_STATE_POSS: tuple[_Poss, ...] = (
    _Poss("state/p_home_country", "home_country",
          (frozenset({"country"}),)),
    _Poss("state/p_home_country2", "home_country",
          (frozenset({"home"}), frozenset({"country"}))),
    _Poss("state/p_home_city", "home_city",
          (frozenset({"city", "town", "hometown"}),)),
    _Poss("state/p_home_in", "home_city", (frozenset({"home"}),),
          lit=(frozenset({"in"}),), gate="geo"),
    _Poss("state/p_address", "address", (frozenset({"address"}),)),
    _Poss("state/p_employer", "employer",
          (frozenset({"employer", "company", "organization", "org"}),)),
    _Poss("state/p_job", "job_title",
          (frozenset({"job", "title", "role", "position", "profession",
                      "occupation"}),)),
    _Poss("state/p_team", "team", (frozenset({"team"}),)),
    _Poss("state/p_manager", "manager",
          (frozenset({"manager", "boss", "lead", "supervisor", "mentor"}),),
          gate="cap"),
    _Poss("state/p_school", "school",
          (frozenset({"school", "college", "university"}),)),
    _Poss("state/p_major", "major", (frozenset({"major"}),)),
    _Poss("state/p_degree", "degree", (frozenset({"degree"}),),
          opt=(frozenset({"in"}),)),
    _Poss("state/p_partner", "partner",
          (frozenset({"partner", "husband", "wife", "boyfriend",
                      "girlfriend", "fiance", "fiancee", "spouse", "so"}),),
          gate="cap"),
    _Poss("state/p_children", "children",
          (frozenset({"son", "daughter", "kid", "child", "baby",
                      "kids", "children", "toddler", "twins", "newborn"}),),
          copula=False, gate="chainval"),
    _Poss("state/p_petname", "pet_names",
          (frozenset({"dog", "cat", "puppy", "kitten", "bird", "parrot",
                      "fish", "hamster", "rabbit", "bunny", "turtle",
                      "snake", "lizard", "gecko", "ferret", "chinchilla",
                      "rat", "mouse", "hedgehog", "horse", "goat"}),),
          gate="petname"),
    _Poss("state/p_petname_app", "pet_names",
          (frozenset({"dog", "cat", "puppy", "kitten", "bird", "parrot",
                      "fish", "hamster", "rabbit", "bunny", "turtle",
                      "snake", "lizard", "gecko", "ferret", "chinchilla",
                      "rat", "mouse", "hedgehog", "horse", "goat"}),),
          copula=False, gate="petname"),
    _Poss("state/p_petnamed", "pet_names",
          (frozenset({"dog", "cat", "puppy", "kitten", "bird", "parrot",
                      "fish", "hamster", "rabbit", "bunny", "turtle",
                      "snake", "lizard", "gecko", "ferret", "chinchilla",
                      "rat", "mouse", "hedgehog", "horse", "goat"}),),
          lit=(frozenset({"named", "called"}),), gate="petname_named"),
    _Poss("state/p_petname2", "pet_names",
          (frozenset({"dog", "cat", "puppy", "kitten", "bird", "parrot",
                      "fish", "hamster", "rabbit", "bunny", "turtle",
                      "snake", "lizard", "gecko", "ferret", "chinchilla",
                      "rat", "mouse", "hedgehog", "horse", "goat"}),
           frozenset({"name"})),
          gate="petname_named"),
    _Poss("state/p_birthday", "birthday", (frozenset({"birthday"}),),
          opt=(frozenset({"in", "on"}),)),
    _Poss("state/p_age", "age", (frozenset({"age"}),), vspec="num"),
    _Poss("state/p_nationality", "nationality",
          (frozenset({"nationality"}),)),
    _Poss("state/p_phone", "phone",
          (frozenset({"phone"}), frozenset({"number"}))),
    _Poss("state/p_phone2", "phone", (frozenset({"number"}),),
          gate="digit"),
    _Poss("state/p_phone_dev", "device", (frozenset({"phone"}),),
          gate="phone_device"),
    _Poss("state/p_email", "email",
          (frozenset({"email", "mail"}),)),
    _Poss("state/p_email2", "email",
          (frozenset({"email"}), frozenset({"address"}))),
    _Poss("state/p_hobby", "hobbies",
          (frozenset({"hobby", "hobbies", "pastime"}),)),
    _Poss("state/p_sport", "sport", (frozenset({"sport"}),)),
    _Poss("state/p_diet", "diet", (frozenset({"diet"}),)),
    _Poss("state/p_allergy", "allergies",
          (frozenset({"allergy", "allergies"}),),
          opt=(frozenset({"to"}),)),
    _Poss("state/p_condition", "health_condition",
          (frozenset({"condition", "diagnosis"}),)),
    _Poss("state/p_medication", "medication",
          (frozenset({"medication", "medications", "meds",
                      "prescription"}),)),
    _Poss("state/p_car", "car",
          (frozenset({"car", "truck", "vehicle", "bike", "motorcycle",
                      "scooter", "van", "suv"}),),
          skiplead=_COLOR_LEX),
    _Poss("state/p_device", "device",
          (frozenset({"laptop", "computer", "tablet", "device", "pc",
                      "mac", "ipad", "watch", "console", "desktop",
                      "kindle", "chromebook", "thinkpad", "macbook"}),),
          gate="nodigit"),
    _Poss("state/p_os", "os", (frozenset({"os", "system"}),)),
    _Poss("state/p_editor", "editor", (frozenset({"editor", "ide"}),)),
    _Poss("state/p_lang", "programming_language",
          (frozenset({"language", "languages"}),), gate="lang_pl"),
    _Poss("state/p_lang2", "programming_language",
          (frozenset({"main", "favorite", "preferred", "primary"}),
           frozenset({"language", "languages"})), gate="lang_pl"),
    _Poss("state/p_project", "project_current",
          (frozenset({"project"}),)),
    _Poss("state/p_project2", "project_current",
          (frozenset({"current"}), frozenset({"project"}))),
    _Poss("state/p_goal", "goal_current",
          (frozenset({"goal"}),), opt=(frozenset({"to"}),)),
    _Poss("state/p_goal2", "goal_current",
          (frozenset({"current", "main"}), frozenset({"goal"}),),
          opt=(frozenset({"to"}),)),
    _Poss("state/p_plan", "plan_upcoming",
          (frozenset({"plan", "plans"}),), opt=(frozenset({"to"}),)),
    _Poss("state/p_trip", "travel_upcoming",
          (frozenset({"trip", "flight", "vacation", "holiday"}),),
          copula=False, lit=(frozenset({"to"}),)),
    _Poss("state/p_schedule", "schedule_regular",
          (frozenset({"schedule"}),)),
    _Poss("state/p_subscription", "subscription",
          (frozenset({"subscription", "subscriptions"}),)),
    _Poss("state/p_bank", "bank_or_payment",
          (frozenset({"bank"}),)),
    _Poss("state/p_payacct", "bank_or_payment",
          (frozenset({"paypal", "venmo", "zelle", "cashapp"}),)),
    _Poss("state/p_timezone", "timezone",
          (frozenset({"timezone"}),)),
    _Poss("state/p_timezone2", "timezone",
          (frozenset({"time"}), frozenset({"zone"}))),
    _Poss("state/p_name", "preferred_name", (frozenset({"name"}),)),
    _Poss("state/p_pronouns", "pronouns", (frozenset({"pronouns"}),),
          vspec="np_open"),
    _Poss("state/p_workout", "workout_routine",
          (frozenset({"workout"}),)),
    _Poss("state/p_workout2", "workout_routine",
          (frozenset({"workout"}), frozenset({"routine"}))),
    _Poss("state/p_book", "reading_current",
          (frozenset({"book", "read"}),)),
    _Poss("state/p_book2", "reading_current",
          (frozenset({"current"}), frozenset({"book", "read"}))),
    _Poss("state/p_show", "show_current",
          (frozenset({"show", "series"}),)),
    _Poss("state/p_show2", "show_current",
          (frozenset({"current"}), frozenset({"show", "series"}))),
)


# favorite possessive state rule handled via _FAV_CAT mapping.

_STATE_SEQ: tuple[_Seq, ...] = (
    # "i'm a <role>" — job_title; value = modifier-terms + role word
    _Seq("state/job", "job_title", _ROLE_NOUNS,
         pre=(("s",), ("cop",), ("adv*",), _ow({"a", "an", "the", "my"}),
              ("val",)),
         post=(), emits=(("value", "prev+anchor"),)),
    # "i'm on the X diet"
    _Seq("state/on_diet", "diet", frozenset({"diet"}),
         pre=(("s",), ("cop",), ("adv*",), _w({"on"}),
              _ow({"a", "an", "the", "my"}), ("val",)),
         post=(), emits=(("value", "prev"),)),
    # "i have a peanut allergy"
    _Seq("state/have_allergy", "allergies", frozenset({"allergy"}),
         pre=(("s",), ("adv*",), _w({"have", "has", "got"}),
              _ow({"a", "an", "the", "my"}), ("val",)),
         post=(), emits=(("value", "prev"),)),
    # "i'm a german citizen"
    _Seq("state/citizen", "nationality",
         frozenset({"citizen", "national"}),
         pre=(("s",), ("cop",), ("adv*",), _ow({"a", "an", "the"}),
              ("val",)),
         post=(), emits=(("value", "prev"),)),
    # "i'm lactose intolerant" / "i'm gluten free"
    _Seq("state/intolerant", "allergies", frozenset({"intolerant"}),
         pre=(("s",), ("cop",), ("adv*",), ("val",)),
         post=(), gate="substance",
         emits=(("value", "prev+anchor"),)),
    _Seq("state/free", "diet", frozenset({"free"}),
         pre=(("s",), ("cop",), ("adv*",), ("val",)),
         post=(), gate="substance",
         emits=(("value", "prev+anchor"),)),
    # "i'm on lexapro for anxiety"
    _Seq("state/on_med", "medication", frozenset({"on"}),
         pre=(("s",), ("cop",), ("adv*",)),
         post=(("valu", "np_then", _MED_TAIL),),
         emits=(("value", "slot"),)),
    # "i'm in pacific time" / "i'm on est"
    _Seq("state/tz_time", "timezone", frozenset({"time"}),
         pre=(("s",), ("cop",), ("adv*",), _w({"in", "on"}),
              _ow({"a", "an", "the", "my"}), ("val",)),
         post=(), emits=(("value", "prev"),)),
    _Seq("state/tz_zone", "timezone",
         _TZ_LEX,
         pre=(("s",), ("cop",), ("adv*",), _w({"in", "on"})),
         post=(), emits=(("value", "anchor"),)),
    # "i'm in Berlin" (capitalized place; TZ names excluded)
    _Seq("state/in_city", "home_city", frozenset({"in"}),
         pre=(("s",), ("cop",), ("adv*",)),
         post=(("valu", "np_cap", ()),), vblock=_TZ_LEX,
         emits=(("value", "slot"),)),
    # "i'm in a relationship"
    _Seq("state/in_rel", "relationship_status",
         frozenset({"relationship"}),
         pre=(("s",), ("cop",), ("adv*",), _w({"in"}),
              _ow({"a", "an"})),
         post=(), emits=(("value", "anchor"),)),
    # "i'm on the platform team"
    _Seq("state/on_team", "team", frozenset({"team"}),
         pre=(("s",), ("cop",), ("adv*",), _w({"on"}),
              _ow({"the", "a", "an", "my"}), ("val",)),
         post=(), emits=(("value", "prev"),)),
    # "i use she/her pronouns"
    _Seq("state/use_pronouns", "pronouns", frozenset({"pronouns"}),
         pre=(("s",), ("adv*",), _w({"use", "uses"}), ("val",)),
         post=(), emits=(("value", "prev"),)),
    # "i go by Mel" / "i go by they/them"
    _Seq("state/go_by", "preferred_name",
         frozenset({"by"}),
         pre=(("s",), ("adv*",), _w({"go", "goes", "going"})),
         post=(("valu", "np_open", ()),), gate="pronoun",
         emits=(("value", "slot"),)),
    # "call me Mel"
    _Seq("state/call_me", "preferred_name",
         frozenset({"call", "calls"}),
         pre=(),
         post=(_w({"me"}), ("valu", "np", ())),
         emits=(("value", "slot"),)),
    # "i have a phd in physics" / "i'm doing my masters"
    _Seq("state/degree", "degree", _DEGREE_WORDS,
         pre=(("s",), ("adv*",),
              _w({"have", "has", "got", "earned", "completed", "finished",
                  "hold", "holds", "doing", "pursuing", "studying"}),
              _ow({"a", "an", "the", "my"})),
         post=(_ow({"in", "of"}), ("val2o", "np", ())),
         emits=(("value", "avspan"),)),
)


# ---------------------------------------------------------------------------
# Candidate + emit plumbing
# ---------------------------------------------------------------------------


@dataclass
class _Cand:
    fam: str
    strength: str
    pol: str
    subj: _Subj
    trig_span: tuple[int, int]
    obj_terms: list[NormTerm]
    obj_span: Optional[tuple[int, int]]
    rule: str
    extra: dict


def _numval(t: NormTerm) -> Optional[int]:
    if t.term.isdigit():
        return int(t.term)
    return _NUMWORDS.get(t.term)


def _emit_pref(out: list[PreferenceFact], ctx: _Ctx, c: _Cand,
               unit_id: str, speaker: Optional[str], scope: str,
               occurred: IntervalUs) -> None:
    if c.subj.canon is None:
        COUNTERS["subject_skip"] += 1
        return
    if not c.obj_terms or c.obj_span is None:
        COUNTERS["value_skip"] += 1
        return
    obj_text = _surface(ctx, c.obj_terms)
    if not obj_text:
        COUNTERS["value_skip"] += 1
        return
    pins: dict[str, Any] = {
        "subject": c.subj.span,
        "trigger": c.trig_span,
        "object": c.obj_span,
        "rule": c.rule,
        "extractor": EXTRACTOR_ID,
        "formula": FORMULA_STATUS,
        "pinned": "raw_verified" if (ctx.raw_bytes is not None
                                     and ctx.hay_ok) else
                  ("text_verified" if ctx.hay_ok else "term_offsets"),
    }
    pins.update(c.extra)
    f = PreferenceFact(
        scope_id=scope,
        subject_canon=c.subj.canon,
        unit_id=unit_id,
        object_text=obj_text,
        polarity=c.pol,
        strength=c.strength,
        occurred=occurred,
        pins=pins,
    )
    out.append(f)
    COUNTERS["emitted_pref"] += 1


def _emit_state(out: list[StateFact], ctx: _Ctx, c: _Cand,
                unit_id: str, speaker: Optional[str], scope: str,
                occurred: Optional[IntervalUs]) -> None:
    if c.subj.canon is None:
        COUNTERS["subject_skip"] += 1
        return
    if c.obj_span is None or not c.obj_terms:
        COUNTERS["value_skip"] += 1
        return
    value_text = _surface(ctx, c.obj_terms)
    if not value_text:
        COUNTERS["value_skip"] += 1
        return
    pins: dict[str, Any] = {
        "subject": c.subj.span,
        "trigger": c.trig_span,
        "value": c.obj_span,
        "family": c.fam,
        "rule": c.rule,
        "extractor": EXTRACTOR_ID,
        "formula": FORMULA_STATUS,
        "pinned": "raw_verified" if (ctx.raw_bytes is not None
                                     and ctx.hay_ok) else
                  ("text_verified" if ctx.hay_ok else "term_offsets"),
    }
    pins.update(c.extra)
    f = StateFact(
        scope_id=scope,
        state_key=f"{c.subj.canon}/{c.fam}",
        unit_id=unit_id,
        value_text=value_text,
        value_norm=_vnorm(value_text),
        valid_from_us=occurred.start_us if occurred is not None else None,
        valid_to_us=None,
        status=StateFactStatus.CURRENT,
        producer=EXTRACTOR_ID,
        pins=pins,
    )
    out.append(f)
    COUNTERS["emitted_state"] += 1


# ---------------------------------------------------------------------------
# Value extraction (vspec dispatch)
# ---------------------------------------------------------------------------


def _value(ctx: _Ctx, terms: Sequence[NormTerm], k: int, hi: int,
           rule) -> list[tuple[list[NormTerm], dict]]:
    """Return list of (value_terms, extra_pins) or [] on failure."""
    spec = rule.vspec
    arg = getattr(rule, "arg", frozenset()) or frozenset()
    vblock = getattr(rule, "vblock", frozenset()) or frozenset()
    skiplead = getattr(rule, "skiplead", frozenset()) or frozenset()

    if spec == "self":
        return [([terms[k - 1]] if k - 1 >= 0 else [], {})]
    if spec == "num":
        if k < hi and _linked(ctx, terms, k) and \
                _numval(terms[k]) is not None:
            n = _numval(terms[k])
            if n is not None and 0 < n <= 120:
                return [([terms[k]], {})]
        return []
    if spec == "habit":
        # after marker: skip aux/neg/adv -> verb -> np
        j = k
        while j < hi and _linked(ctx, terms, j) and \
                terms[j].term in _SKIP:
            j += 1
        if j >= hi or not _linked(ctx, terms, j):
            return []
        verb = terms[j]
        if verb.term in _COP or verb.term in _CONTROL:
            return []
        val, _ = _np(ctx, terms, j + 1, hi, vblock)
        if not val and j + 1 < hi and _linked(ctx, terms, j + 1) and \
                terms[j + 1].term == "home" and not (
                j + 2 < hi and _linked(ctx, terms, j + 2)
                and _obj_ok(terms[j + 2], vblock)):
            # standalone deictic object ("walk home") — _TEMP membership
            # keeps "home" out of general NP collection but it is the
            # literal stated object here (V8-13.04).  A following
            # collectable ("hit home runs") abstains rather than
            # mis-slicing.
            val = [terms[j + 1]]
        if not val or val[0].term in _VALUE_BLOCK:
            return []
        return [(val, {"verb": (verb.byte_start, verb.byte_end)})]
    if spec == "tail_or_self":
        val, _ = _np(ctx, terms, k, hi, vblock)
        if not val:
            return []
        return [(val, {})]

    if spec == "np_open":
        # pronouns allowed in the value (go-by pronouns, "she/her", ...)
        stop = _VSTOP - frozenset({"i", "we", "he", "she", "they"})
        val, k2 = _np(ctx, terms, k, hi, vblock,
                      lead_strip=_LEAD_STRIP | skiplead,
                      stop=stop)
        return [(val, {})] if val else []
    val, k2 = _np(ctx, terms, k, hi, vblock,
                  lead_strip=_LEAD_STRIP | skiplead)
    if spec == "np":
        return [(val, {})] if val else []
    if spec == "np_opt":
        return [(val, {})] if val else [([terms[k - 1]], {})]
    if spec == "np_cap":
        if val and _is_cap_surface(ctx, val[:1]):
            return [(val, {})]
        return []
    if spec == "np_terms":
        hits = [t for t in val if t.term in arg]
        return [([t], {}) for t in hits]
    if spec == "np_any":
        if val and any(t.term in arg for t in val):
            return [(val, {})]
        return []
    if spec == "np_from":
        for idx, t in enumerate(val):
            if t.term in arg:
                return [(val[idx:], {})]
        return []
    if spec == "np_then":
        if not val:
            return []
        if val[0].term in _VALUE_BLOCK:
            return []
        # tail marker (schedule/"for") OR a recognized med name suffices;
        # otherwise abstain ("i take the bus" is not medication).
        if k2 < hi and _linked(ctx, terms, k2) and terms[k2].term in arg:
            return [(val, {"marker": (terms[k2].byte_start,
                                      terms[k2].byte_end)})]
        if any(t.term in _MEDS for t in val):
            return [(val, {"marker": None})]
        return []
    if spec == "np_pet":
        # np until {named,called} -> split conjuncts on and/or; pet noun
        # inside each conjunct -> emit; 'named X' tail -> pet_names fact.
        stop = frozenset({"named", "called"})
        val2, k3 = _np(ctx, terms, k, hi, stop)
        out: list[tuple[list[NormTerm], dict]] = []
        if val2:
            # split on and/or
            conj: list[NormTerm] = []
            parts: list[list[NormTerm]] = []
            for t in val2:
                if t.term in ("and", "or"):
                    if conj:
                        parts.append(conj)
                    conj = []
                else:
                    conj.append(t)
            if conj:
                parts.append(conj)
            for p in parts:
                core = [t for t in p if t.term not in _LEAD_STRIP]
                if core and any(t.term in arg for t in core):
                    out.append((core, {}))
        if k3 < hi and _linked(ctx, terms, k3) and \
                terms[k3].term in ("named", "called"):
            nm, _ = _np(ctx, terms, k3 + 1, hi)
            if nm:
                out.append((nm, {"family_override": "pet_names"}))
        return out
    return []


# ---------------------------------------------------------------------------
# Trigger-rule matcher (verb-class: subject + window + trigger + lits + value)
# ---------------------------------------------------------------------------


def _match_trig(ctx: _Ctx, terms: Sequence[NormTerm], i: int, lo: int,
                hi: int, rule: _Trig, speaker: Optional[str],
                negs_ok: bool) -> list[_Cand]:
    k = i + 1
    for wset in rule.lit:
        if k >= hi or not _linked(ctx, terms, k) or \
                terms[k].term not in wset:
            return []
        k += 1
    for wset in rule.opt:
        if k < hi and _linked(ctx, terms, k) and terms[k].term in wset:
            k += 1
    if rule.reject and k < hi and _linked(ctx, terms, k) and \
            terms[k].term in rule.reject:
        return []
    subj = _subject(ctx, terms, i, lo, speaker)
    if subj is None or subj.canon is None:
        return []
    trig_span = (terms[i].byte_start,
                 terms[k - 1].byte_end if k > i + 1 else terms[i].byte_end)
    reason = _guarded(ctx, terms, lo, hi, subj, i, rule.allow_d)
    if reason:
        COUNTERS["guarded"] += 1
        return []
    neg = subj.neg is not None
    if rule.need_neg and not neg:
        return []
    if rule.need and not any(t.term in rule.need for t in subj.window):
        return []
    if rule.need_cop:
        cop_ok = any(t.term in _COP for t in subj.window) or \
            subj.clitic in ("s", "m", "re") or \
            any(t.term in _AUX for t in subj.window)
        if not cop_ok:
            return []
    pol = rule.pol
    if neg and not rule.need_neg:
        if rule.fam:
            # negated state asserts nothing — abstain entirely
            COUNTERS["guarded"] += 1
            return []
        if pol == "negative":
            # double negation -> abstain (documented)
            COUNTERS["guarded"] += 1
            return []
        pol = "negative"
    vals = _value(ctx, terms, k, hi, rule)
    if not vals and rule.vspec == "tail_or_self" and k > i:
        # "i work out daily" — the verb phrase itself is the routine
        vals = [([x for x in terms[i:k]], {})]
    cands: list[_Cand] = []
    for vt, extra in vals:
        if not vt:
            continue
        if vblocked(rule, vt):
            continue
        fam = extra.pop("family_override", rule.fam)
        if rule.gate == "geo":
            fam = "home_country" if any(
                t.term in _COUNTRIES for t in vt) else "home_city"
        if rule.gate == "sport":
            if all(t.term in _SPORTS for t in vt
                   if t.term not in ("and", "or", "of")):
                fam = "sport"
            elif any(t.term in _GAMES or t.term in _SPORTS for t in vt):
                fam = "hobbies"
            else:
                fam = "hobbies"
        if rule.gate == "going":
            # "i'm going to <verb>" -> plan; "i'm going to <Place>" -> travel
            if vt and vt[0].term in _VERBISH:
                fam = "plan_upcoming"
            elif _is_cap_surface(ctx, vt[:1]):
                fam = "travel_upcoming"
            else:
                continue
        cands.append(_Cand(
            fam=fam, strength=rule.strength, pol=pol, subj=subj,
            trig_span=trig_span, obj_terms=vt,
            obj_span=(vt[0].byte_start, vt[-1].byte_end),
            rule=rule.rid, extra=dict(extra)))
        for fam_ov, _which in getattr(rule, "emits", ()):
            cands.append(_Cand(
                fam=fam_ov, strength=rule.strength, pol=pol, subj=subj,
                trig_span=trig_span, obj_terms=[terms[i]],
                obj_span=(terms[i].byte_start, terms[i].byte_end),
                rule=rule.rid, extra={}))
    return cands


def vblocked(rule, vt) -> bool:
    vb = getattr(rule, "vblock", frozenset()) or frozenset()
    return bool(vb) and len(vt) == 1 and vt[0].term in vb


# ---------------------------------------------------------------------------
# Sequence-rule matcher (anchor + backward pre-slots + forward post-slots)
# ---------------------------------------------------------------------------


def _match_seq(ctx: _Ctx, terms: Sequence[NormTerm], i: int, lo: int,
               hi: int, rule: _Seq, speaker: Optional[str]) -> list[_Cand]:
    """Match anchor terms[i]; pre-slots matched backward; post forward.

    Pre-slot kinds (processed right-to-left from the anchor):
      ("s",)      subject resolution (window+chunk+clitic recovery)
      ("cop",)    copula term or 's/'m/'re gap
      ("adv*",)   zero+ adverb/neg terms (neg terms recorded -> negation)
      ("w", set)  required literal term
      ("ow", set) optional literal term
      ("val",)    collect 0-3 collectable terms backward -> pre-value
      ("owx",)    optional single collectable term
    Post-slot kinds:
      ("w"|"ow", set)      literal after anchor
      ("valu"/"val"/"val2", spec, arg)  forward value collection
    Emits entries: (family-or-"value", "slot"|"anchor"|"prev"|"prev+anchor"
    |"val2").
    """
    j = i - 1
    window: list[NormTerm] = []
    pre_val: list[NormTerm] = []     # collected right-to-left
    subj: Optional[_Subj] = None
    clitic = ""
    neg: Optional[NormTerm] = None
    for slot in reversed(rule.pre):
        kind = slot[0]
        if kind == "adv*":
            while j >= lo and _linked(ctx, terms, j + 1) and \
                    terms[j].term in _ADV:
                if terms[j].term in _NEG:
                    neg = terms[j]
                window.insert(0, terms[j])
                j -= 1
            continue
        if kind == "owx":
            if j >= lo and _linked(ctx, terms, j + 1) and \
                    _collectable(terms[j]):
                j -= 1
            continue
        if kind == "val":
            n = 0
            while j >= lo and n < 3 and _linked(ctx, terms, j + 1) and \
                    _collectable(terms[j]):
                pre_val.insert(0, terms[j])
                j -= 1
                n += 1
            continue
        if kind == "cop":
            if j >= lo and _linked(ctx, terms, j + 1) and \
                    terms[j].term in _COP:
                window.insert(0, terms[j])
                j -= 1
            else:
                cl = _gap_clitic(ctx, terms[j].byte_end,
                                 terms[j + 1].byte_start) \
                    if j >= lo and j + 1 < len(terms) else ""
                if cl in ("s", "m", "re"):
                    clitic = cl
                else:
                    return []
            continue
        if kind == "s":
            subj = _subject(ctx, terms, j + 1, lo, speaker)
            if subj is None:
                return []
            if subj.window:
                window = list(subj.window) + window
            if subj.neg is not None:
                neg = subj.neg
            if subj.clitic:
                clitic = subj.clitic
            continue
        # literal slot (w|ow)
        _, wset = slot
        if j >= lo and _linked(ctx, terms, j + 1) and \
                terms[j].term in wset:
            window.insert(0, terms[j])
            j -= 1
        elif kind == "w":
            return []
    if subj is None:
        for t in terms[lo:i]:
            if t.term in _NEG:
                neg = t
        subj = _Subj(speaker, "speaker", None, tuple(window), clitic, neg)
    else:
        subj = _Subj(subj.canon, subj.form, subj.span,
                     tuple(window), clitic, neg)
    # forward post-slots
    k = i + 1
    value_terms: list[NormTerm] = []
    value2: list[NormTerm] = []
    for slot in rule.post:
        kind = slot[0]
        if kind in ("w", "ow"):
            _, wset = slot
            if k < hi and _linked(ctx, terms, k) and \
                    terms[k].term in wset:
                k += 1
            elif kind == "w":
                return []
            continue
        if kind in ("valu", "val", "val2", "val2o"):
            spec = slot[1]
            arg = slot[2] if len(slot) > 2 else frozenset()
            fake = _Trig("", "", frozenset(), vspec=spec, arg=arg,
                         vblock=rule.vblock)
            got = _value(ctx, terms, k, hi, fake)
            vt = got[0][0] if got else []
            if kind in ("valu", "val", "val2") and not vt:
                return []
            if vt:
                k = _index_of(terms, vt[-1]) + 1
            if kind in ("val2", "val2o"):
                value2 = vt
            else:
                value_terms = vt
            continue
        return []
    reason = _guarded(ctx, terms, lo, hi, subj, i, rule.allow_d)
    if reason:
        COUNTERS["guarded"] += 1
        return []
    pol = rule.pol
    if neg is not None:
        if pol == "negative":
            COUNTERS["guarded"] += 1
            return []
        pol = "negative"
    emits = rule.emits or (("value", "slot"),)
    cands: list[_Cand] = []
    for fam_ov, which in emits:
        fam = rule.fam if fam_ov == "value" else fam_ov
        if which in ("slot", "valu", "val"):
            vt = value_terms
        elif which == "val2":
            vt = value2
        elif which == "anchor":
            vt = [terms[i]]
        elif which == "prev":
            vt = pre_val
        elif which == "prev+anchor":
            vt = pre_val + [terms[i]]
        elif which == "avspan":
            # value = anchor through the end of val2 (incl connectors)
            if value2:
                vt = list(terms[i:_index_of(terms, value2[-1]) + 1])
            else:
                vt = [terms[i]]
        else:
            vt = value_terms
        if not vt:
            continue
        if rule.gate == "pronoun":
            fam = "pronouns" if all(t.term in _PRONOUN_LEX
                                    for t in vt) else "preferred_name"
        if rule.gate == "substance" and \
                not any(t.term in _SUBSTANCES for t in pre_val):
            continue
        if vblocked(rule, vt):
            continue
        extra: dict[str, Any] = {}
        if value2:
            extra["over"] = (value2[0].byte_start, value2[-1].byte_end)
            extra["over_text"] = _surface(ctx, value2)
        cands.append(_Cand(
            fam=fam, strength=rule.strength, pol=pol, subj=subj,
            trig_span=(terms[i].byte_start, terms[i].byte_end),
            obj_terms=vt, obj_span=(vt[0].byte_start, vt[-1].byte_end),
            rule=rule.rid, extra=extra))
    return cands


def _index_of(terms: Sequence[NormTerm], t: NormTerm) -> int:
    for idx, x in enumerate(terms):
        if x is t or (x.byte_start == t.byte_start and
                      x.byte_end == t.byte_end and x.term == t.term):
            return idx
    return len(terms) - 1


# ---------------------------------------------------------------------------
# Possessive-rule matcher
# ---------------------------------------------------------------------------


def _match_poss(ctx: _Ctx, terms: Sequence[NormTerm], i: int, lo: int,
                hi: int, rule: _Poss, speaker: Optional[str]) -> list[_Cand]:
    """terms[i] == chain[0]; possessor at i-1; chain + copula + lit + value."""
    subj = _possessor(ctx, terms, i, lo, speaker)
    if subj is None or subj.canon is None:
        return []
    k = i
    for wset in rule.chain:
        if k >= hi or terms[k].term not in wset:
            return []
        if k > i and not _linked(ctx, terms, k):
            cl = _gap_clitic(ctx, terms[k - 1].byte_end,
                             terms[k].byte_start)
            if not cl:
                return []
        k += 1
    chain_end = k
    # copula slot: term in COP or 's-gap before value
    cop_ok = False
    if rule.copula:
        if k < hi and _linked(ctx, terms, k) and terms[k].term in _COP:
            k += 1
            cop_ok = True
        elif k < hi:
            cl = _gap_clitic(ctx, terms[k - 1].byte_end,
                             terms[k].byte_start)
            if cl in ("s", "m", "re"):
                cop_ok = True
        if not cop_ok:
            return []
    for wset in rule.lit:
        if k >= hi or not _linked(ctx, terms, k) or \
                terms[k].term not in wset:
            return []
        k += 1
    for wset in rule.opt:
        if k < hi and _linked(ctx, terms, k) and terms[k].term in wset:
            k += 1
    reason = _guarded(ctx, terms, lo, hi, subj, i, False)
    if reason:
        COUNTERS["guarded"] += 1
        return []
    if subj.neg is not None:
        return []  # negated state asserts nothing
    gate = rule.gate
    cands: list[_Cand] = []
    if gate == "chainval":
        # the chain noun itself is the stated fact ("my son is 5" -> the
        # child descriptor "son"; the trailing value is age/etc, not a fact)
        cands.append(_Cand(
            fam=rule.fam, strength="", pol="positive", subj=subj,
            trig_span=(terms[i].byte_start,
                       terms[chain_end - 1].byte_end),
            obj_terms=[terms[chain_end - 1]],
            obj_span=(terms[chain_end - 1].byte_start,
                      terms[chain_end - 1].byte_end),
            rule=rule.rid, extra={}))
        return cands
    if gate in ("petname", "petname_named"):
        # "my dog is X" / "my dog X" / "my dog is named X" — ownership
        # itself yields a pets fact (the pet noun is the value).
        cands.append(_Cand(
            fam="pets", strength="", pol="positive", subj=subj,
            trig_span=(terms[i].byte_start,
                       terms[chain_end - 1].byte_end),
            obj_terms=[terms[i]],
            obj_span=(terms[i].byte_start, terms[i].byte_end),
            rule=rule.rid + "/pet", extra={}))
    fake = _Trig("", "", frozenset(), vspec=rule.vspec, arg=rule.arg,
                 skiplead=getattr(rule, "skiplead", frozenset()))
    vals = _value(ctx, terms, k, hi, fake)
    for vt, extra in vals:
        if not vt:
            continue
        fam = rule.fam
        if gate == "cap" and not _is_cap_surface(ctx, vt[:1]):
            continue
        if gate == "digit":
            if not any(ch.isdigit() for t in vt for ch in t.term):
                continue
        if gate == "nodigit":
            if any(ch.isdigit() for t in vt for ch in t.term) or \
                    any("@" in t.term for t in vt):
                continue
        if gate == "phone_device":
            if any(ch.isdigit() for t in vt for ch in t.term) or \
                    any("@" in t.term for t in vt):
                fam = "phone"
            else:
                fam = "device"
        if gate == "lang_pl":
            if any(t.term in _PL_LEX for t in vt):
                fam = "programming_language"
            elif any(t.term in _LANGS for t in vt):
                fam = "languages"
            else:
                continue
        if gate == "petname" and not _is_cap_surface(ctx, vt[:1]):
            continue
        # petname_named: 'named/called' itself asserts the name — no
        # capitalization evidence required.
        cands.append(_Cand(
            fam=fam, strength="", pol="positive", subj=subj,
            trig_span=(terms[i].byte_start, terms[chain_end - 1].byte_end),
            obj_terms=vt, obj_span=(vt[0].byte_start, vt[-1].byte_end),
            rule=rule.rid, extra=dict(extra)))
    return cands


# ---------------------------------------------------------------------------
# Favorite machinery (pref + state share the same shape)
# ---------------------------------------------------------------------------


def _match_favorite(ctx: _Ctx, terms: Sequence[NormTerm], i: int,
                    lo: int, hi: int, speaker: Optional[str],
                    want_state: bool) -> list[_Cand]:
    """'my favorite <cat> is <v>' and '<v> is my favorite <cat>'."""
    out: list[_Cand] = []
    t = terms[i]
    if t.term not in ("favorite", "favourite"):
        return out
    # --- forward form: possessor + favorite + cat* + cop + value ---
    poss = _possessor(ctx, terms, i, lo, speaker)
    if poss is not None and poss.canon is not None:
        k = i + 1
        cat: list[NormTerm] = []
        while k < hi and len(cat) < 2 and _linked(ctx, terms, k) and \
                _collectable(terms[k]) and terms[k].term not in _COP and \
                terms[k].term not in _AUX:
            cat.append(terms[k])
            k += 1
        cop_ok = False
        if k < hi and _linked(ctx, terms, k) and terms[k].term in _COP:
            k += 1
            cop_ok = True
        elif k < hi:
            cl = _gap_clitic(ctx, terms[k - 1].byte_end,
                             terms[k].byte_start)
            cop_ok = cl in ("s", "m", "re")
        if cop_ok:
            val, _k2 = _np(ctx, terms, k, hi)
            if val:
                reason = _guarded(ctx, terms, lo, hi, poss, i, False)
                if not reason and poss.neg is None:
                    cat_txt = " ".join(x.term for x in cat) or None
                    extra: dict[str, Any] = {}
                    if cat_txt:
                        extra["category"] = (
                            cat[0].byte_start, cat[-1].byte_end)
                        extra["category_text"] = cat_txt
                    fam = _FAV_CAT.get(cat_txt or "", "")
                    if not want_state or fam:
                        out.append(_Cand(
                            fam=fam, strength="favorite", pol="positive",
                            subj=poss,
                            trig_span=(t.byte_start, terms[k - 1].byte_end),
                            obj_terms=val,
                            obj_span=(val[0].byte_start,
                                      val[-1].byte_end),
                            rule="pref/fav_poss" if not want_state else
                            "state/fav_poss", extra=extra))
                elif reason:
                    COUNTERS["guarded"] += 1
    # --- reverse form: <v> is my favorite [cat] ---
    if i >= lo + 2 and terms[i - 1].term in ("my", "our") and \
            terms[i - 2].term in _COP and _linked(ctx, terms, i - 1) and \
            _linked(ctx, terms, i):
        cop_i = i - 2
        # collect NP left of copula
        chunk: list[NormTerm] = []
        j = cop_i - 1
        while j >= lo and _linked(ctx, terms, j + 1) and \
                _collectable(terms[j]):
            chunk.insert(0, terms[j])
            j -= 1
            if len(chunk) >= 4:
                break
        while chunk and chunk[0].term in _LEAD_STRIP:
            chunk.pop(0)
        # optional category after favorite
        cat2: list[NormTerm] = []
        k = i + 1
        while k < hi and len(cat2) < 2 and _linked(ctx, terms, k) and \
                _collectable(terms[k]):
            cat2.append(terms[k])
            k += 1
        if chunk:
            subj = _Subj(speaker, "speaker",
                         (terms[i - 1].byte_start, terms[i - 1].byte_end),
                         (), "")
            reason = _guarded(ctx, terms, lo, hi, subj, i, False)
            if not reason:
                cat_txt = " ".join(x.term for x in cat2) or None
                extra = {}
                if cat_txt:
                    extra["category"] = (cat2[0].byte_start,
                                         cat2[-1].byte_end)
                    extra["category_text"] = cat_txt
                fam = _FAV_CAT.get(cat_txt or "", "")
                if not want_state or fam:
                    out.append(_Cand(
                        fam=fam, strength="favorite", pol="positive",
                        subj=subj,
                        trig_span=(terms[cop_i].byte_start, t.byte_end),
                        obj_terms=chunk,
                        obj_span=(chunk[0].byte_start, chunk[-1].byte_end),
                        rule="pref/fav_rev" if not want_state else
                        "state/fav_rev", extra=extra))
            else:
                COUNTERS["guarded"] += 1
    return out


# ---------------------------------------------------------------------------
# Special seq rules implemented in code (schedule, age, bare-name)
# ---------------------------------------------------------------------------


_SCHED_ACT_AUX = frozenset({"have", "has", "had", "do", "does", "did"})


def _match_sched(ctx: _Ctx, terms: Sequence[NormTerm], lo: int, hi: int,
                 speaker: Optional[str]) -> list[_Cand]:
    """'i <verb> <np?> (on|every) <sched>' and 'on/every <sched> i <verb>'."""
    out: list[_Cand] = []

    def _act(span_terms) -> Optional[list[NormTerm]]:
        # reject spans with negation/coordination; keep content words plus
        # verb particles, preps and dets that are part of the phrase
        # ("go to the gym"), and have/do aux carriers ("have class").
        if any(t.term in _NEG or t.term in _COORD or t.term in _MODAL
               or t.term in _SUBORD for t in span_terms):
            return None
        act = [t for t in span_terms
               if t.term not in _AUX or t.term in _SCHED_ACT_AUX]
        if not act or not any(t.term not in _AUX and t.term not in _DET
                              and t.term not in _PREP for t in act):
            return None
        return act

    # form A: marker inside the clause
    for m in range(lo, hi):
        if terms[m].term not in ("on", "every"):
            continue
        if m + 1 >= hi or not _linked(ctx, terms, m + 1):
            continue
        sched = terms[m + 1]
        if terms[m].term == "on" and sched.term not in _SCHED_PLURAL:
            continue
        if terms[m].term == "every" and sched.term not in _SCHED_ALL:
            continue
        # subject must resolve somewhere left of the marker; the activity
        # is the content span between subject-end and the marker.
        subj = None
        vi = None
        for j in range(lo, m):
            s = _subject(ctx, terms, j, lo, speaker)
            if s is not None and s.canon is not None:
                subj = s
                vi = j
                break
        if subj is None or vi is None:
            continue
        act = _act(terms[vi:m])
        if not act:
            continue
        if subj.neg is not None:
            continue
        if _guarded(ctx, terms, lo, hi, subj, m, False):
            continue
        out.append(_Cand(
            fam="schedule_regular", strength="", pol="positive",
            subj=subj,
            trig_span=(terms[vi].byte_start, sched.byte_end),
            obj_terms=act,
            obj_span=(act[0].byte_start, act[-1].byte_end),
            rule="state/sched_a",
            extra={"schedule": (terms[m].byte_start, sched.byte_end)}))

    # form B: 'every tuesday i play chess' / 'on fridays i work out'
    if hi - lo >= 4 and terms[lo].term in ("on", "every") and \
            terms[lo + 1].term in _SCHED_ALL and \
            _linked(ctx, terms, lo + 1):
        if terms[lo].term == "on" and \
                terms[lo + 1].term not in _SCHED_PLURAL:
            return out
        si = lo + 2
        if si < hi and _linked(ctx, terms, si) and \
                terms[si].term in ("i", "we"):
            act = _act(terms[si + 1:hi])
            if act:
                subj = _Subj(speaker, "speaker",
                             (terms[si].byte_start, terms[si].byte_end),
                             (), "")
                out.append(_Cand(
                    fam="schedule_regular", strength="", pol="positive",
                    subj=subj,
                    trig_span=(terms[lo].byte_start,
                               terms[lo + 1].byte_end),
                    obj_terms=act,
                    obj_span=(act[0].byte_start, act[-1].byte_end),
                    rule="state/sched_b",
                    extra={"schedule": (terms[lo].byte_start,
                                        terms[lo + 1].byte_end)}))
    return out


def _match_age(ctx: _Ctx, terms: Sequence[NormTerm], lo: int, hi: int,
               speaker: Optional[str]) -> list[_Cand]:
    """'i'm 30' (clause-final) and 'i'm 30 years old'."""
    out: list[_Cand] = []
    for i in range(lo, hi):
        n = _numval(terms[i])
        if n is None or not (0 < n <= 120):
            continue
        # must follow copula-window + subject
        subj = _subject(ctx, terms, i, lo, speaker)
        if subj is None or subj.canon is None:
            continue
        if not any(t.term in _COP for t in subj.window) and \
                subj.clitic not in ("s", "m", "re"):
            continue
        # clause-final OR followed by 'year(s) old'
        if i + 1 == hi:
            pass
        elif i + 2 < hi and terms[i + 1].term in ("year", "years") and \
                terms[i + 2].term == "old" and \
                _linked(ctx, terms, i + 1) and _linked(ctx, terms, i + 2):
            pass
        elif i + 1 < hi and terms[i + 1].term in ("year", "years") and \
                _linked(ctx, terms, i + 1):
            pass
        else:
            continue
        if subj.neg is not None:
            continue
        reason = _guarded(ctx, terms, lo, hi, subj, i, False)
        if reason:
            continue
        out.append(_Cand(
            fam="age", strength="", pol="positive", subj=subj,
            trig_span=(terms[i].byte_start, terms[i].byte_end),
            obj_terms=[terms[i]],
            obj_span=(terms[i].byte_start, terms[i].byte_end),
            rule="state/age_num", extra={}))
    return out


def _match_im_name(ctx: _Ctx, terms: Sequence[NormTerm], lo: int, hi: int,
                   speaker: Optional[str]) -> list[_Cand]:
    """'i'm Mel' — copula + capitalized bare value -> preferred_name."""
    out: list[_Cand] = []
    for i in range(lo, hi):
        t = terms[i]
        if t.term in _SKIP or t.term in _NP_BLOCK or t.term in _DET or \
                t.term in _POSS or t.term in _NOM or t.channel == "identifier":
            continue
        subj = _subject(ctx, terms, i, lo, speaker)
        if subj is None or subj.canon is None or subj.form != "speaker":
            continue
        if not any(x.term in _COP for x in subj.window) and \
                subj.clitic not in ("s", "m", "re"):
            continue
        if not _is_cap_surface(ctx, (t,)):
            continue
        if subj.neg is not None:
            continue
        reason = _guarded(ctx, terms, lo, hi, subj, i, False)
        if reason:
            continue
        # avoid stealing values other rules own (diet/status/demonym/role)
        if t.term in _DIET_IDS or t.term in _STATUS_WORDS or \
                t.term in _DEMONYMS or t.term in _TZ_LEX:
            continue
        out.append(_Cand(
            fam="preferred_name", strength="", pol="positive", subj=subj,
            trig_span=(t.byte_start, t.byte_end), obj_terms=[t],
            obj_span=(t.byte_start, t.byte_end),
            rule="state/im_name", extra={}))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_STRENGTH_RANK = {
    "constraint": 4, "favorite": 3, "love_hate": 2, "like_dislike": 1,
    "habitual": 0, "": 0,
}


def extract_preferences(
    norm: NormAnalysis,
    unit_id: str,
    speaker_canon: str,
    *,
    scope_id: str = "",
    occurred: Optional[IntervalUs] = None,
    raw_text: Optional[str] = None,
    canon_fn: Optional[Callable[[str], str]] = None,
    stats: Optional[Any] = None,
) -> list[PreferenceFact]:
    """§32.12 pref/v1: deterministic preference extraction.

    The frozen contract is ``(norm, unit_id, speaker_canon)``; ``scope_id``
    and ``occurred`` are keyword-only extensions (the persister stamps the
    real scope and the unit's occurrence interval — ``extract_preferences``
    has no ``occurred`` in the frozen signature, so it defaults to the
    unknown interval).  ``raw_text`` enables exact surfaces + the
    capitalization/quote/clitic guards; without it the extractor degrades
    honestly (folded surfaces, no name-binding).  ``stats``, when a dict is
    supplied, receives this call's counter deltas."""
    base = dict(COUNTERS)
    COUNTERS["units_pref"] += 1
    if occurred is None:
        occurred = IntervalUs(None, None)
    ctx = _mk_ctx(norm, raw_text, canon_fn)
    terms = _stream(norm)
    # V8-13.04: unattributed single-user input binds first-person subjects
    # to "me"; a known speaker canon stays verbatim (multi-party).
    speaker = speaker_canon or "me"
    cands: list[_Cand] = []

    trig_by: dict[str, list[_Trig]] = {}
    for r in _PREF_RULES:
        for v in r.verbs:
            trig_by.setdefault(v, []).append(r)
    seq_by: dict[str, list[_Seq]] = {}
    for r in _PREF_SEQ:
        for v in r.anchor:
            seq_by.setdefault(v, []).append(r)

    for i, t in enumerate(terms):
        w = t.term
        lo, hi = _clause_bounds(ctx, terms, i)
        for r in trig_by.get(w, ()):
            cands.extend(_match_trig(ctx, terms, i, lo, hi, r, speaker,
                                     True))
        for r in seq_by.get(w, ()):
            cands.extend(_match_seq(ctx, terms, i, lo, hi, r, speaker))
        if w in ("favorite", "favourite"):
            cands.extend(_match_favorite(ctx, terms, i, lo, hi, speaker,
                                         False))

    out: list[PreferenceFact] = []
    # dedupe: same (subject, object span, polarity) -> keep strongest
    best: dict[tuple, _Cand] = {}
    order: dict[tuple, int] = {}
    for c in cands:
        if c.obj_span is None:
            continue
        key = (c.subj.canon, c.obj_span, c.pol)
        rank = _STRENGTH_RANK.get(c.strength, 0)
        if key not in best:
            best[key] = c
            order[key] = c.trig_span[0]
        else:
            if rank > _STRENGTH_RANK.get(best[key].strength, 0):
                best[key] = c
            COUNTERS["deduped"] += 1
    for key, c in sorted(best.items(),
                         key=lambda kv: (kv[1].trig_span[0],
                                         kv[1].obj_span or (0, 0),
                                         kv[1].rule)):
        _emit_pref(out, ctx, c, unit_id, speaker or "", scope_id,
                   occurred)
    if stats is not None:
        stats.update({k: v - base[k] for k, v in COUNTERS.items()
                      if v != base[k]})
    return out


def extract_state_facts(
    norm: NormAnalysis,
    unit_id: str,
    speaker_canon: str,
    occurred: IntervalUs,
    *,
    scope_id: str = "",
    raw_text: Optional[str] = None,
    canon_fn: Optional[Callable[[str], str]] = None,
    stats: Optional[Any] = None,
) -> list[StateFact]:
    """§32.11 state_keys/v1: deterministic state-fact extraction.

    Emits ``StateFactStatus.CURRENT`` candidates; ``state_key`` =
    ``f"{subject_canon}/{family}"``; ``valid_from_us`` =
    ``occurred.start_us``.  ``scope_id`` is a keyword-only extension stamped
    onto each fact's ``scope_id`` (the frozen signature lacks it).
    ``stats``, when a dict is supplied, receives this call's counter
    deltas."""
    base = dict(COUNTERS)
    COUNTERS["units_state"] += 1
    ctx = _mk_ctx(norm, raw_text, canon_fn)
    terms = _stream(norm)
    # V8-13.04: unattributed single-user input binds first-person subjects
    # to "me"; a known speaker canon stays verbatim (multi-party).
    speaker = speaker_canon or "me"
    cands: list[_Cand] = []

    trig_by: dict[str, list[_Trig]] = {}
    for r in _STATE_TRIG_RULES:
        for v in r.verbs:
            trig_by.setdefault(v, []).append(r)
    poss_by: dict[str, list[_Poss]] = {}
    for r in _STATE_POSS:
        for v in r.chain[0]:
            poss_by.setdefault(v, []).append(r)
    seq_by: dict[str, list[_Seq]] = {}
    for r in _STATE_SEQ:
        for v in r.anchor:
            seq_by.setdefault(v, []).append(r)

    for i, t in enumerate(terms):
        w = t.term
        lo, hi = _clause_bounds(ctx, terms, i)
        for r in trig_by.get(w, ()):
            cands.extend(_match_trig(ctx, terms, i, lo, hi, r, speaker,
                                     False))
        for r in seq_by.get(w, ()):
            cands.extend(_match_seq(ctx, terms, i, lo, hi, r, speaker))
        if w in poss_by:
            for r in poss_by[w]:
                cands.extend(_match_poss(ctx, terms, i, lo, hi, r,
                                         speaker))
        if w in ("favorite", "favourite"):
            cands.extend(_match_favorite(ctx, terms, i, lo, hi, speaker,
                                         True))
    # clause-level specials
    seen_clauses: set[tuple[int, int]] = set()
    for i, t in enumerate(terms):
        lo, hi = _clause_bounds(ctx, terms, i)
        if (lo, hi) in seen_clauses:
            continue
        seen_clauses.add((lo, hi))
        cands.extend(_match_sched(ctx, terms, lo, hi, speaker))
        cands.extend(_match_age(ctx, terms, lo, hi, speaker))
        cands.extend(_match_im_name(ctx, terms, lo, hi, speaker))

    # geo routing for live/move rules
    out: list[StateFact] = []
    best: dict[tuple, _Cand] = {}
    for c in cands:
        if c.obj_span is None:
            continue
        fam = c.fam
        if fam == "home_city" and c.rule in ("state/live_in",
                                             "state/move_to",
                                             "state/from"):
            if any(x.term in _COUNTRIES for x in c.obj_terms):
                fam = "home_country"
        c.fam = fam
        key = (c.subj.canon, fam, c.obj_span)
        if key not in best:
            best[key] = c
        else:
            COUNTERS["deduped"] += 1
    for key, c in sorted(best.items(),
                         key=lambda kv: (kv[1].trig_span[0],
                                         kv[1].obj_span or (0, 0),
                                         kv[1].rule)):
        _emit_state(out, ctx, c, unit_id, speaker or "", scope_id,
                    occurred)
    if stats is not None:
        stats.update({k: v - base[k] for k, v in COUNTERS.items()
                      if v != base[k]})
    return out


def state_compatible(key: str, a: str, b: str) -> bool:
    """§32.11: do two values under ``key`` coexist without conflict?

    ``key`` is ``"<subject_canon>/<family>"`` (the family suffix is used).
    Accumulate families (pets, pet_names, children, hobbies, languages,
    allergies, health_condition, medication, sport, programming_language,
    subscription, diet, goal_current, schedule_regular, device) never
    conflict.  Replace families are compatible iff the normalized values
    are equal; differing values conflict (the persister marks the older
    fact historical).  ``age`` compares numerically.  Unknown families
    default to replace semantics."""
    # family keys are canonical identifiers ("home_city"); casefold only —
    # punctuation-folding would turn the underscore into a space.
    fam = (key.rsplit("/", 1)[-1] if "/" in key else key).strip().casefold()
    if fam in _ACCUMULATE:
        # exception: a bare kid *count* replaces rather than accumulates
        # ("two kids" vs "three kids" conflict; "son"/"daughter" coexist).
        if fam == "children":
            ca, cb = _countof(a), _countof(b)
            if ca is not None and cb is not None:
                return ca == cb
        return True
    na, nb = _vnorm(a), _vnorm(b)
    if fam in _NUMERIC_FAMILIES:
        da, db = _digits(na), _digits(nb)
        if da is not None and db is not None:
            return da == db
    return na == nb


def _digits(s: str) -> Optional[int]:
    d = "".join(ch for ch in s if ch.isdigit())
    if d:
        return int(d)
    return _NUMWORDS.get(s)


def _countof(s: str) -> Optional[int]:
    """Leading numeral of a folded value: '2 kids'/'two kids' -> 2."""
    f = _vnorm(s)
    tok = f.split()[0] if f else ""
    if tok.isdigit():
        return int(tok)
    return _NUMWORDS.get(tok)
