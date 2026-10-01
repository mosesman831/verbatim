#!/usr/bin/env python3
"""SCRATCH — synthetic LoCoMo-like dialogue slice for formula search.

Self-contained: generates a dialogue corpus (2 speakers/session, evolving
attributes, aliases, temporal anchors, filler chatter with lexical confusors),
then measures any@10/any@20 for candidate retrieval formulas:

  A. plain BM25 over turn text                      (the baseline we lose to)
  B. BM25F over {text, speaker, entities, tlabel}
  C. lanes -> flat RRF, k in {10,30,60,90}
  D. lanes -> weighted RRF {1.0/0.75/0.5} provisional shape
  E. lanes -> min-max normalized CombSUM
  F. best fusion + bounded boosts (recency, temporal, proof-ish)
  G. BM25F fields + hashing-embedding lane fused by RRF

All numbers are CALCULATED on an invented corpus — not benchmark evidence.
"""
import json, math, random, re, sqlite3, time, hashlib, os, sys
from collections import Counter, defaultdict

RNG = random.Random(20260922)

# ---------------------------------------------------------------- corpus gen
FIRST_NAMES = [
    ("Alice", "Ali"), ("Carlos", "Carl"), ("Priya", "Pri"), ("Dmitri", "Dim"),
    ("Rosa", "Rosie"), ("Kenji", "Ken"), ("Fatima", "Fati"), ("Tom", "Tommy"),
    ("Ingrid", "Inga"), ("Marco", "Marc"), ("Leila", "Lei"), ("Hugo", "Hu"),
    ("Anya", "An"), ("Bruno", "Bru"), ("Mei", "May"), ("Oscar", "Oz"),
]
CITIES = ["Boston", "Seattle", "Austin", "Denver", "Oslo", "Kyoto", "Lisbon",
          "Prague", "Toronto", "Melbourne", "Nairobi", "Helsinki", "Valencia",
          "Portland", "Marseille", "Daegu", "Cusco", "Tallinn"]
JOBS = [("nurse", "Riverview Clinic"), ("data analyst", "Ferrolytics"),
        ("teacher", "Hillcrest School"), ("chef", "Casa Verde"),
        ("barista", "Mud Cup"), ("carpenter", "Old Pine Workshop"),
        ("pharmacist", "MedLane"), ("journalist", "City Ledger"),
        ("mechanic", "Gears Auto"), ("pilot", "Sky Harbor Air"),
        ("librarian", "Northgate Library"), ("dancer", "Studio Lume")]
HOBBIES = ["hiking", "pottery", "birdwatching", "bouldering", "piano",
           "gardening", "salsa dancing", "astrophotography", "kayaking",
           "embroidery", "table tennis", "sourdough baking", "fencing",
           "origami", "longboarding", "chess"]
PETS = [("cat", ["Whiskers", "Miso", "Sable", "Pico", "Noodle"]),
        ("dog", ["Biscuit", "Kiko", "Rex", "Taffy", "Mango"]),
        ("parrot", ["Kiwi", "Zazu", "Milo", "Pesto"]),
        ("rabbit", ["Clover", "Buttons", "Hazel"])]
FOODS = ["ramen", "biryani", "tacos", "pad see ew", "shakshuka",
         "pierogi", "moussaka", "okonomiyaki", "ceviche", "bibimbap",
         "gnocchi", "falafel wraps", "jollof rice", "arepas"]
SIBS = ["sister", "brother", "cousin"]
SIB_NAMES = ["Nina", "Elias", "Tara", "Jonas", "Marta", "Felix", "Sana", "Pavel"]
TLABELS_WEEK = ["last weekend", "a few days ago", "this past Friday",
                "the other day", "last Tuesday", "this past week"]
TLABELS_MONTH = ["a couple of weeks ago", "last month", "back in March",
                 "earlier this year", "two months ago"]

FILLER = [
    "Haha that's so true.", "lol no way", "Wait, seriously?",
    "That reminds me of that meme.", "ok but hear me out",
    "Haha fair enough.", "Totally agree with you there.",
    "Can't believe it's already Friday.", "Anyway, coffee?",
    "So random but it works.", "We should plan something soon.",
    "You always know what to say.", "Oh for sure, same.",
    "That's the funniest thing today.", "Omg yes, tell me more",
    "Not me reading this twice.", "Right?? Exactly.",
    "I'm crying, stop it.", "That escalated fast lol",
    "bless up", "good vibes only today",
]

TEMPLATES = {
    "pet": ["Guess what! I adopted a {0[0]} {1} — her name is {0[1]}.",
            "We finally brought {0[1]} home {1}, cutest {0[0]} ever.",
            "{0[1]} is settling in well since we adopted her {1}. She's a sweet {0[0]}.",
            "I'm officially a pet owner now — adopted a {0[0]} {1}, named her {0[1]}.",
            "My new pet {0[0]} {0[1]} came home {1}.",
            "This pet of mine, a {0[0]} called {0[1]}, is the best decision {1}."],
    "reside_new": ["I just moved to {0} {1}! Still unpacking boxes.",
                   "So {1} I relocated to {0}. New apartment, new neighborhood.",
                   "Officially a {0} resident as of {1}.",
                   "I'm now living in {0}, moved {1}.",
                   "Settled into life in {0} after the move {1}.",
                   "My current city is {0} as of {1}.",
                   "Living in {0} these days — the move was {1}."],
    "reside_old": ["Back when I lived in {0}, everything was different.",
                   "I spent a few years in {0} before this. Loved it there.",
                   "My old apartment in {0} was tiny but cozy.",
                   "When I was living in {0} I used to bike everywhere."],
    "job_new": ["Big news — I start as a {0[0]} at {0[1]} {1}.",
                "Just signed with {0[1]}; I'll be their new {0[0]} starting {1}.",
                "New gig locked in: {0[0]} at {0[1]}, starting {1}.",
                "I work at {0[1]} now, as a {0[0]} — started {1}.",
                "I'm working at {0[1]} as a {0[0]} these days, since {1}."],
    "job_old": ["I used to work as a {0[0]} at {0[1]}.",
                "My old job at {0[1]} as a {0[0]} taught me a lot.",
                "Before this I was a {0[0]} over at {0[1]}.",
                "I worked at {0[1]} for years as a {0[0]}."],
    "hobby": ["I've been getting really into {0} lately, {1} was my third time this week.",
              "{1} I spent the whole afternoon on {0}. Obsessed now.",
              "You'd laugh but {0} is my new thing — started {1}.",
              "My current hobby is {0} — picked it up {1}.",
              "New hobby unlocked: {0}. Have been at it since {1}.",
              "For fun these days I do {0}, started {1}."],
    "food": ["Hands down, {0} is my favorite food. Had it {1}.",
             "I could eat {0} every day — grabbed some {1}.",
             "{1} I made {0} from scratch. Still my all-time favorite.",
             "I love eating {0} — had it again {1}."],
    "sibling": ["My {0[0]} {0[1]} lives in {0[2]} now, moved there {1}.",
                "{0[1]}, my {0[0]}, is in {0[2]} these days — since {1}.",
                "Talking to my {0[0]} {0[1]} in {0[2]} {1}, they're doing great.",
                "Family update: my {0[0]} {0[1]} is living in {0[2]} now."],
    "visit": ["I'm visiting family in {0} next month, can't wait.",
              "Booked my trip to see family in {0} — going next month.",
              "Next month I'll be in {0} visiting family. Excited!"],
    "vacation": ["Dreaming about a trip to {0} {1}.",
                 "{1} I started planning a vacation to {0}.",
                 "I really want to visit {0} someday — was looking at flights {1}."],
    "allergy": ["Found out {1} that I'm allergic to {0}. Explains a lot.",
                "Apparently I have a {0} allergy — learned that {1}.",
                "Doctor confirmed {1}: allergic to {0}. Ugh."],
    "car": ["I bought a {0} {1}! First big purchase.",
            "{1} I finally got the {0} I've been saving for.",
            "New {0} owner as of {1}. Nervous driver mode on.",
            "Decided to buy a {0} — picked it up {1}."],
    "project": ["I'm building a {0} at home, started {1}.",
                "Side project {1}: working on a {0}. It's coming along.",
                "Making slow progress on my {0} since {1}."],
    "book": ["Started reading {0} {1}, it's incredible.",
             "Can't put {0} down since {1}.",
             "{1} I picked up {0} — already halfway through.",
             "The book I'm reading right now is {0}, since {1}."],
}

def fact_text(attr, val, spk, when):
    return RNG.choice(TEMPLATES[attr]).format(val, when)

QTEMPLATES = {
    "pet_name": ["What is the name of {0}'s pet?",
                 "What did {0} name their {1}?"],
    "pet_kind": ["What kind of pet does {0} have?"],
    "reside_now": ["Where does {0} live now?",
                   "What city is {0} currently living in?"],
    "job_now": ["What is {0}'s current job?",
                "Where does {0} work now?"],
    "hobby_now": ["What hobby is {0} into these days?",
                  "What does {0} do for fun lately?"],
    "fav_food": ["What is {0}'s favorite food?",
                 "What does {0} love to eat?"],
    "sibling_city": ["Where does {0}'s {1[0]} live?",
                     "What city does {0}'s {1[0]} {1[1]} live in?"],
    "visit_city": ["Which city is {0} visiting family in?",
                   "Where will {0} go to see family?"],
    "allergy": ["What is {0} allergic to?",
                "Does {0} have any allergies?"],
    "car": ["What did {0} buy recently?",
            "What big purchase did {0} make?"],
    "project": ["What is {0} building?",
                "What project is {0} working on at home?"],
    "book": ["What book is {0} reading?",
             "What is {0} currently reading?"],
    "outdoor_pref": ["Does {0} like outdoor activities?",
                     "Is {0} into nature stuff?"],
}

def question_text(qtype, spk_name, extra):
    return RNG.choice(QTEMPLATES[qtype]).format(spk_name, extra)

FACTMAP = {}

def gen_corpus():
    """returns (turns, questions)
    turn: dict(id, session, speaker, alias, ts, text, entities, tlabel, suppressed)
    question: dict(qid, text, evidence_ids, qtype, wants_latest)
    evidence ids are registered EXPLICITLY when a fact turn is emitted —
    never reconstructed by grepping text afterwards."""
    turns, questions = [], []
    tid = 0
    sessions = [RNG.sample(FIRST_NAMES, 2) for _ in range(20)]
    for si, (s1, s2) in enumerate(sessions):
        n_turns = RNG.randint(33, 40)
        tss = sorted(RNG.randint(600, 60 * 86400) for _ in range(n_turns))
        speakers = [RNG.choice((s1, s2)) for _ in range(n_turns)]
        attrs = RNG.sample(
            ["pet", "reside", "job", "hobby", "food", "sibling_visit",
             "allergy", "car", "project", "book", "vacation"], 8)
        fact_slots = {}
        for a in attrs:
            k = {"pet": 2, "reside": 3, "job": 3, "sibling_visit": 2}.get(a, 2)
            fact_slots[a] = sorted(RNG.sample(range(2, n_turns - 1), k))
        # evmap[fullname][attr] = {"ids": [tid...], "latest_ids": [tid...]}
        spk_state = {}
        def reg(name, attr, tid, latest=None, val=None):
            st = spk_state.setdefault(name, {})
            e = st.setdefault(attr, {"ids": [], "latest_ids": [], "val": val})
            e["ids"].append(tid)
            if latest:
                e["latest_ids"].append(tid)
            if val is not None:
                e["val"] = val
        for i in range(n_turns):
            spk, spkalias = speakers[i]
            when = RNG.choice(TLABELS_WEEK + TLABELS_MONTH)
            text, entities = None, []
            for a, idxs in fact_slots.items():
                if i not in idxs:
                    continue
                pos = idxs.index(i)
                st = spk_state.setdefault(spk, {})
                if a == "pet":
                    kind, pname = RNG.choice(
                        [(k, RNG.choice(n)) for k, n in PETS])
                    text = fact_text("pet", (kind, pname), spk, when)
                    entities = [pname, kind]
                    reg(spk, "pet", tid, val=(kind, pname))
                elif a == "reside":
                    if pos == 0:
                        oc = RNG.choice(CITIES)
                        text = fact_text("reside_old", oc, spk, when)
                        entities = [oc]; st["oldcity"] = oc
                    elif pos == len(idxs) - 1:
                        nc = RNG.choice([c for c in CITIES
                                         if c != st.get("oldcity")])
                        text = fact_text("reside_new", nc, spk, when)
                        entities = [nc]
                        reg(spk, "reside", tid, latest=True, val=nc)
                    else:
                        mid = st.get("oldcity") or RNG.choice(CITIES)
                        text = RNG.choice([
                            f"I still think about {mid} sometimes.",
                            f"{mid} has the best coffee, I swear.",
                            f"Friends in {mid} keep inviting me back.",
                            f"I miss living in {mid} honestly."])
                        entities = [mid]
                elif a == "job":
                    if pos == 0:
                        oj = RNG.choice(JOBS)
                        text = fact_text("job_old", oj, spk, when)
                        entities = [oj[0], oj[1]]; st["oldjob"] = oj
                    elif pos == len(idxs) - 1:
                        nj = RNG.choice([j for j in JOBS
                                         if j != st.get("oldjob")])
                        text = fact_text("job_new", nj, spk, when)
                        entities = [nj[0], nj[1]]
                        reg(spk, "job", tid, latest=True, val=nj)
                    else:
                        oj = st.get("oldjob") or RNG.choice(JOBS)
                        text = RNG.choice([
                            f"Working as a {oj[0]} back then was exhausting.",
                            f"I still have friends at {oj[1]}.",
                            f"{oj[1]} was a grind but the people were nice."])
                        entities = [oj[0], oj[1]]
                elif a == "hobby":
                    h = st.setdefault("hobby_val", RNG.choice(HOBBIES))
                    if pos == 0:
                        text = fact_text("hobby", h, spk, when)
                    else:
                        text = RNG.choice([
                            f"Another round of {h} {when}. Never gets old.",
                            f"Spent {when} on {h} again, no regrets.",
                            f"{h.capitalize()} update: still obsessed ({when})."])
                    entities = [h]
                    reg(spk, "hobby", tid, val=h)
                elif a == "food":
                    f = RNG.choice(FOODS)
                    text = fact_text("food", f, spk, when)
                    entities = [f]
                    reg(spk, "food", tid, val=f)
                elif a == "sibling_visit":
                    if pos == 0:
                        sib, sname = RNG.choice(SIBS), RNG.choice(SIB_NAMES)
                        sc = RNG.choice(CITIES)
                        val = (sib, sname, sc)
                        text = fact_text("sibling", val, spk, when)
                        entities = [sname, sc]
                        reg(spk, "sib", tid, val=val)
                    else:
                        sc = st.get("sib", {}).get(
                            "val", (None, None, RNG.choice(CITIES)))[2]
                        text = fact_text("visit", sc, spk, when)
                        entities = [sc]
                        reg(spk, "visit", tid, val=sc)
                elif a == "vacation" and pos == 0:
                    vc = RNG.choice(CITIES)
                    text = fact_text("vacation", vc, spk, when)
                    entities = [vc]
                    reg(spk, "vacation", tid, val=vc)
                elif a == "allergy" and pos == 0:
                    al = RNG.choice(["peanuts", "pollen", "shellfish",
                                     "cats", "gluten"])
                    text = fact_text("allergy", al, spk, when)
                    entities = [al]
                    reg(spk, "allergy", tid, val=al)
                elif a == "car" and pos == 0:
                    c = RNG.choice(["used Honda Civic", "mountain bike",
                                    "Vespa scooter", "Toyota RAV4", "e-bike"])
                    text = fact_text("car", c, spk, when)
                    entities = [c]
                    reg(spk, "car", tid, val=c)
                elif a == "project" and pos == 0:
                    p = RNG.choice(["treehouse", "home espresso bar",
                                    "tiny robot arm", "fermentation cellar",
                                    "retro arcade cabinet"])
                    text = fact_text("project", p, spk, when)
                    entities = [p]
                    reg(spk, "project", tid, val=p)
                elif a == "book" and pos == 0:
                    b = RNG.choice(["Dune", "Project Hail Mary",
                                    "Klara and the Sun",
                                    "The Ministry for the Future",
                                    "Piranesi"])
                    text = fact_text("book", b, spk, when)
                    entities = [b]
                    reg(spk, "book", tid, val=b)
            if text is None:
                text = RNG.choice(FILLER)
                if RNG.random() < 0.12:
                    c = RNG.choice(CITIES)
                    text = RNG.choice([
                        f"Oh {c} is lovely this time of year.",
                        f"I heard {c} is expensive though.",
                        f"Someone I know just moved to {c}."])
                    entities = [c]
            is_alias_turn = False
            if RNG.random() < 0.045 and spkalias:
                is_alias_turn = True
                text = f"Everyone just calls me {spkalias}, honestly."
                entities = entities + [spkalias]
                reg(spk, "alias", tid, val=spkalias)
            turns.append(dict(id=tid, session=si, speaker=spk,
                              alias=spkalias, ts=tss[i], text=text,
                              entities=entities,
                              tlabel=(when if RNG.random() < 0.5 else None),
                              suppressed=(RNG.random() < 0.04)))
            tid += 1
        # ---- emit questions
        qn = 0
        def ask(qtype, name, extra, ids, wants_latest=False, qcat=None, attr=None):
            nonlocal qn
            if not ids:
                return
            questions.append(dict(qid=f"s{si}q{qn}", qtype=qcat or qtype,
                                  text=question_text(qtype, name, extra),
                                  evidence=list(ids),
                                  wants_latest=wants_latest,
                                  qattr=attr or qtype,
                                  qname=name))
            qn += 1
        OUTDOOR = ("hiking", "bouldering", "birdwatching", "kayaking",
                   "gardening", "longboarding")
        for name, st in spk_state.items():
            if name == "_alias_ev":
                continue
            if "pet" in st:
                ask("pet_name", name, st["pet"]["val"][0], st["pet"]["ids"])
                ask("pet_kind", name, None, st["pet"]["ids"])
            if "reside" in st:
                ask("reside_now", name, None, st["reside"]["latest_ids"],
                    wants_latest=True)
            if "job" in st:
                ask("job_now", name, None, st["job"]["latest_ids"],
                    wants_latest=True)
            if "hobby" in st:
                ask("hobby_now", name, None, st["hobby"]["ids"])
                if st["hobby"]["val"] in OUTDOOR:
                    ask("outdoor_pref", name, None, st["hobby"]["ids"],
                        qcat="openish")
            if "food" in st:
                ask("fav_food", name, None, st["food"]["ids"])
            if "sib" in st:
                v = st["sib"]["val"]
                ask("sibling_city", name, (v[0], v[1]), st["sib"]["ids"],
                    qcat="multihop")
                if "visit" in st:
                    ask("visit_city", name, None,
                        st["sib"]["ids"] + st["visit"]["ids"],
                        qcat="multihop")
            if "allergy" in st:
                ask("allergy", name, None, st["allergy"]["ids"])
            if "car" in st:
                ask("car", name, None, st["car"]["ids"])
            if "project" in st:
                ask("project", name, None, st["project"]["ids"])
            if "book" in st:
                ask("book", name, None, st["book"]["ids"])
            if "vacation" in st:
                ask("visit_city", name, None, st["vacation"]["ids"])
        # alias-based questions (entity category): ask via the nickname
        for name, st in spk_state.items():
            if name == "_alias_ev" or "alias" not in st:
                continue
            spkt = s1 if s1[0] == name else s2
            alias = st["alias"]["val"]
            for attr in ("pet", "food", "hobby"):
                if attr in st:
                    extra = st[attr]["val"][0] if attr == "pet" else None
                    ask("pet_name" if attr == "pet" else
                        ("fav_food" if attr == "food" else "hobby_now"),
                        alias, extra, st[attr]["ids"] + st["alias"]["ids"],
                        qcat="entity")
                    break
        # snapshot this session's fact-state into the oracle map
        for name, st in spk_state.items():
            if name == "_alias_ev":
                continue
            FACTMAP[name] = {a: {"ids": e["ids"],
                                 "latest_ids": e["latest_ids"],
                                 "val": (list(e["val"])
                                        if isinstance(e.get("val"), tuple)
                                        else e.get("val"))}
                             for a, e in st.items() if isinstance(e, dict)}
    for q in questions:
        ev = [e for e in q["evidence"] if not turns[e]["suppressed"]]
        q["evidence"] = ev or q["evidence"]
    questions = [q for q in questions if q["evidence"]]
    # persist the fact-state map = oracle fact index for e4 analysis —
    # accumulate a copy per session BEFORE loop ends (spk_state is per-session)
    return turns, questions

# ---------------------------------------------------------------- retrieval
TOK = re.compile(r"[a-z0-9']+")
def _norm(w):
    """crude symmetric morphological normalizer (calculated, not a real stemmer):
    applied identically to index and query terms so inflections collide:
    lives/living/live -> liv, moved/move -> mov, named/name -> nam."""
    w = w.lower()
    if len(w) > 4:
        for suf in ("ies", "ing", "est", "ers", "er", "ed", "ly", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)]
                break
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    if len(w) > 3 and len(w) >= 2 and w[-1] == w[-2] and w[-1] in "bcdfglmnprstz":
        w = w[:-1]
    return w

def toks(s): return [_norm(w) for w in TOK.findall(s.lower())]
def raw_toks(s): return TOK.findall(s.lower())

class Corpus:
    def __init__(self, turns):
        self.turns = [t for t in turns if not t["suppressed"]]
        self.id_of_idx = [t["id"] for t in self.turns]
        self.idx_of_id = {t["id"]: i for i, t in enumerate(self.turns)}
        self.tok_text = [toks(t["text"]) for t in self.turns]
        self.tok_ent = [[_norm(w) for e in t["entities"] for w in e.lower().split()]
                        for t in self.turns]
        self.speaker = [_norm(t["speaker"]) for t in self.turns]
        self.alias = [_norm(t.get("alias") or "") for t in self.turns]
        self.raw_speaker = [t["speaker"].lower() for t in self.turns]
        self.raw_alias = [(t.get("alias") or "").lower() for t in self.turns]
        self.tlabel = [(t.get("tlabel") or "") for t in self.turns]
        self.N = len(self.turns)
        self.dl = [len(x) for x in self.tok_text]
        self.avgdl = sum(self.dl) / max(1, self.N)
        self.df = Counter()
        for tt in self.tok_text:
            for w in set(tt):
                self.df[w] += 1
        # speaker name index: token -> set(turn idx) — mirrors speaker field search
        self.name_index = defaultdict(set)
        for i in range(self.N):
            self.name_index[_norm(self.raw_speaker[i])].add(i)
            if self.raw_alias[i]:
                self.name_index[_norm(self.raw_alias[i])].add(i)

        # hash-embedding lane vectors (blake2b subword n-grams, 384 dims — mirrors hashing.py)
        self.D = 384
        import numpy as np
        self.vecs = np.zeros((self.N, self.D), dtype=np.float32)
        for i, tt in enumerate(self.tok_text):
            self.vecs[i] = self._hash_vec(tt)
        norms = np.linalg.norm(self.vecs, axis=1, keepdims=True)
        self.vecs = self.vecs / np.maximum(norms, 1e-9)

    def speaker_mask(self, qt):
        idx = set()
        for w in set(qt):
            idx |= self.name_index.get(w.lower(), set())
        return idx

    def _hash_vec(self, tokens):
        import numpy as np
        v = np.zeros(self.D, dtype=np.float32)
        def h(b):
            return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")
        for w in tokens:
            v[h(w.encode()) % self.D] += 1.0
            for n in (3, 4, 5):
                for j in range(len(w) - n + 1):
                    v[h(w[j:j+n].encode()) % self.D] += 0.6
        return v

    def idf(self, w):
        df = self.df.get(w, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def bm25(self, qt, k1=1.2, b=0.75):
        s = np.zeros(self.N)
        for w in set(qt):
            idf = self.idf(w)
            for i, tt in enumerate(self.tok_text):
                f = tt.count(w)
                if f:
                    s[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * self.dl[i] / self.avgdl))
        return s

    def bm25f(self, qt, fields=("text", "speaker", "entities", "tlabel"),
              weights=(1.0, 1.8, 2.2, 0.4), k1=1.2):
        # per-field saturation variant (Lucene-consistent): Σ_f w_f · BM25_f
        ent_docs = self.tok_ent
        allw = set(qt)
        s = np.zeros(self.N)
        # precompute field dls
        ent_dl = [len(e) for e in ent_docs]
        ent_avg = (sum(ent_dl) or 1) / max(1, self.N)
        tl_docs = [toks(x) for x in self.tlabel]
        tl_dl = [len(x) for x in tl_docs]
        tl_avg = (sum(tl_dl) or 1) / max(1, self.N)
        sp_docs = [toks(x) for x in self.raw_speaker]
        sp_dl = [len(x) for x in sp_docs]
        sp_avg = (sum(sp_dl) or 1) / max(1, self.N)
        # df per field for idf
        dfe, dft, dfs = Counter(), Counter(), Counter()
        for i in range(self.N):
            for w in set(ent_docs[i]): dfe[w] += 1
            for w in set(tl_docs[i]): dft[w] += 1
            for w in set(sp_docs[i]): dfs[w] += 1
        def fidf(df, w):
            d = df.get(w, 0)
            return math.log(1 + (self.N - d + 0.5) / (d + 0.5))
        qtl = [w.lower() for w in qt]
        for w in allw:
            wl = w.lower()
            # text
            s += weights[0] * self.bm25([w], k1=k1) * 1.0
            # entities: exact token or substring match on entity strings
            iw = fidf(dfe, wl)
            for i, es in enumerate(ent_docs):
                f = sum(1 for e in es for et in e.split() if et == wl) + \
                    sum(1 for e in es if wl in e and len(wl) > 3)
                if f:
                    sat = f * (k1 + 1) / (f + k1 * (1 - 0.75 + 0.75 * ent_dl[i] / ent_avg))
                    s[i] += weights[2] * iw * sat * 0.5
            # speaker: exact name or alias
            iw = fidf(dfs, wl)
            for i in range(self.N):
                f = sp_docs[i].count(wl) + (1 if self.alias[i] == wl else 0)
                if f:
                    sat = f * (k1 + 1) / (f + k1 * (1 - 0.75 + 0.75 * sp_dl[i] / sp_avg))
                    s[i] += weights[1] * iw * sat * 0.5
            # tlabel
            iw = fidf(dft, wl)
            for i, td in enumerate(tl_docs):
                f = td.count(wl)
                if f:
                    sat = f * (k1 + 1) / (f + k1 * (1 - 0.75 + 0.75 * tl_dl[i] / tl_avg))
                    s[i] += weights[3] * iw * sat * 0.5
        return s

import numpy as np

def topk(score, k):
    idx = np.argpartition(-score, min(k, len(score) - 1))[:k]
    return idx[np.argsort(-score[idx])].tolist()

def rrf(lists, k=60, weights=None):
    s = defaultdict(float)
    for li, lst in enumerate(lists):
        w = 1.0 if weights is None else weights[li]
        for r, d in enumerate(lst, 1):
            s[d] += w / (k + r)
    return sorted(s, key=lambda d: -s[d])

def combsum(lanes, k=20):
    # min-max normalize each lane's scores over its candidate set, sum
    s = defaultdict(float)
    for sc in lanes:
        vals = np.asarray(sc)
        hi = vals.max()
        if hi <= 0:
            continue
        for i, v in enumerate(vals):
            if v > 0:
                s[i] += float(v / hi)
    return sorted(s, key=lambda d: -s[d])

def run_eval(turns, questions, system):
    corp = Corpus(turns)
    LANE_N = 40
    def lanes_for(qt):
        # lexical lane
        lex = corp.bm25(qt)
        l_lex = topk(lex, LANE_N)
        # entity lane: score = #entity matches (weight by idf)
        ent_sc = np.zeros(corp.N)
        ql = set(w.lower() for w in qt)
        for i in range(corp.N):
            m = 0.0
            for e in corp.tok_ent[i]:
                for et in e.split():
                    if et in ql:
                        m += corp.idf(et)
            ent_sc[i] = m
        l_ent = topk(ent_sc, LANE_N)
        # speaker lane
        sp_sc = np.zeros(corp.N)
        for i in range(corp.N):
            if corp.speaker[i] in ql or corp.alias[i] in ql:
                sp_sc[i] = 1.0
        l_spk = topk(sp_sc, LANE_N)
        # temporal lane: recency + tlabel match
        tmp_sc = np.zeros(corp.N)
        twords = {_norm(w) for w in ("now", "current", "currently", "lately",
                  "these", "days", "today", "recent", "recently", "this",
                  "week", "month", "new", "latest")}
        wants_time = bool(ql & twords)
        for i, t in enumerate(corp.turns):
            rec = 1.0 / (1.0 + t["ts"] / 86400.0 / 30.0)  # monthly decay
            tl = 1.0 if (wants_time and t.get("tlabel")) else 0.0
            tmp_sc[i] = rec + tl
        l_tmp = topk(tmp_sc, LANE_N)
        # hashing-embedding lane (cosine on hashed n-gram vectors)
        qv = corp._hash_vec(list(qt))
        qv = qv / max(np.linalg.norm(qv), 1e-9)
        den_sc = corp.vecs @ qv
        l_den = topk(den_sc, LANE_N)
        return dict(lex=lex, ent=ent_sc, spk=sp_sc, tmp=tmp_sc, den=den_sc,
                    lists=[l_lex, l_ent, l_spk, l_tmp, l_den])
    def masked_topk(score, mask, k):
        s = np.asarray(score).copy()
        if mask is not None and len(mask):
            keep = np.zeros(corp.N, dtype=bool)
            keep[list(mask)] = True
            s[~keep] = -1e9
        return topk(s, k)

    res = defaultdict(list)
    t_all = []
    for q in questions:
        qt = toks(q["text"])
        t0 = time.perf_counter()
        if system == "bm25":
            ranked = topk(corp.bm25(qt), 20)
        elif system == "bm25f":
            ranked = topk(corp.bm25f(qt), 20)
        elif system == "spk_bm25":
            m = corp.speaker_mask(qt)
            ranked = masked_topk(corp.bm25(qt), m or None, 20)
        elif system == "spk_bm25f":
            m = corp.speaker_mask(qt)
            ranked = masked_topk(corp.bm25f(qt), m or None, 20)
        elif system == "spk_bm25f_temp":
            m = corp.speaker_mask(qt)
            sc = corp.bm25f(qt)
            twords = {_norm(w) for w in ("now", "current", "currently",
                      "lately", "these", "days", "today", "recent",
                      "recently", "new", "latest")}
            wants_time = bool(set(qt) & twords)
            for i, t in enumerate(corp.turns):
                rec = 1.0 / (1.0 + t["ts"] / 86400.0 / 30.0)
                mult = 1.0 + 0.10 * rec
                if wants_time and t.get("tlabel"):
                    mult *= 1.10
                sc[i] *= mult
            ranked = masked_topk(sc, m or None, 20)
        elif system.startswith("rrf_"):
            k = int(system.split("_")[1])
            L = lanes_for(qt)
            ranked = rrf(L["lists"], k=k)[:20]
        elif system == "wrrf60":
            L = lanes_for(qt)
            ranked = rrf(L["lists"], k=60, weights=[1.0, 0.75, 0.5, 0.5, 1.0])[:20]
        elif system == "combsum":
            L = lanes_for(qt)
            ranked = combsum([L["lex"], L["ent"], L["spk"], L["tmp"], L["den"]])[:20]
        elif system == "rrf60_boosts":
            L = lanes_for(qt)
            base = rrf(L["lists"], k=60)
            pos = {d: r + 1 for r, d in enumerate(base)}
            twords = {_norm(w) for w in ("now", "current", "currently",
                      "lately", "these", "days", "today", "recent",
                      "recently", "new", "latest")}
            wants_time = bool(set(qt) & twords)
            def boosted(d):
                t = corp.turns[d]
                v = 1.0 + 0.10 * (1.0 / (1.0 + t["ts"] / 86400.0 / 30.0))
                if wants_time and t.get("tlabel"):
                    v *= 1.10
                if set(corp.tok_ent[d]) & set(qt):
                    v *= 1.05  # entity-corroborated (proof-ish)
                return (1.0 / (60.0 + pos[d])) * v
            ranked = sorted(base, key=lambda d: -boosted(d))[:20]
        elif system == "spk_bm25f_ctemp":
            # conditioned temporal: recency+tlabel boosts apply ONLY to candidates
            # that already match the query on entity or speaker tokens
            m = corp.speaker_mask(qt)
            sc = corp.bm25f(qt)
            ql = set(qt)
            twords = {_norm(w) for w in ("now", "current", "currently",
                      "lately", "these", "days", "today", "recent",
                      "recently", "new", "latest")}
            wants_time = bool(ql & twords)
            for i, t in enumerate(corp.turns):
                ent_hit = bool(set(corp.tok_ent[i]) & ql) or \
                          corp.speaker[i] in ql or corp.alias[i] in ql
                if not ent_hit:
                    continue
                rec = 1.0 / (1.0 + t["ts"] / 86400.0 / 30.0)
                sc[i] *= 1.0 + 0.10 * rec
                if wants_time and t.get("tlabel"):
                    sc[i] *= 1.10
            ranked = masked_topk(sc, m or None, 20)
        elif system == "spk_csum":
            # scope mask + calibrated-ish combsum inside scope (bm25f + entity lanes)
            m = corp.speaker_mask(qt)
            lex = corp.bm25f(qt)
            ql = set(qt)
            ent_sc = np.zeros(corp.N)
            for i in range(corp.N):
                for e in corp.tok_ent[i]:
                    for et in e.split():
                        if et in ql:
                            ent_sc[i] += corp.idf(et)
            tmp_sc = np.zeros(corp.N)
            for i, t in enumerate(corp.turns):
                tmp_sc[i] = 1.0 / (1.0 + t["ts"] / 86400.0 / 30.0)
            lanes = [np.where(np.isin(np.arange(corp.N), list(m or range(corp.N))), x, 0.0)
                     for x in (lex, ent_sc, tmp_sc)]
            fused = np.zeros(corp.N)
            for x in lanes:
                hi = x.max()
                if hi > 0:
                    fused += x / hi
            ranked = masked_topk(fused, m or None, 20)
        elif system == "bm25f_den_rrf":
            lex = corp.bm25f(qt)
            l_lex = topk(lex, 40)
            qv = corp._hash_vec(list(qt))
            qv = qv / max(np.linalg.norm(qv), 1e-9)
            l_den = topk(corp.vecs @ qv, 40)
            ranked = rrf([l_lex, l_den], k=60)[:20]
        t_all.append(time.perf_counter() - t0)
        ids10 = {corp.id_of_idx[r] for r in ranked[:10]}
        ids20 = {corp.id_of_idx[r] for r in ranked[:20]}
        hit10 = bool(ids10 & set(q["evidence"]))
        hit20 = bool(ids20 & set(q["evidence"]))
        res[q["qtype"]].append((hit10, hit20))
        res["_all"].append((hit10, hit20))
    out = {}
    for cat, hits in res.items():
        n = len(hits)
        out[cat] = dict(n=n,
                        any10=sum(h[0] for h in hits) / n,
                        any20=sum(h[1] for h in hits) / n)
    out["_latency_ms"] = dict(p50=float(np.percentile(t_all, 50) * 1000),
                              p95=float(np.percentile(t_all, 95) * 1000))
    return out

def main():
    turns, questions = gen_corpus()
    n_sup = sum(1 for t in turns if t["suppressed"])
    print(f"corpus: {len(turns)} turns ({n_sup} suppressed = eligibility-filtered), "
          f"{len(questions)} questions",
          file=sys.stderr)
    os.makedirs("/workspace/memorysys/eval/v7/scratch", exist_ok=True)
    with open("/workspace/memorysys/eval/v7/scratch/synth_corpus.json", "w") as f:
        json.dump(dict(turns=turns, questions=questions), f, indent=1)
    systems = ["bm25", "bm25f", "spk_bm25", "spk_bm25f", "spk_bm25f_temp",
               "bm25f_den_rrf",
               "spk_bm25f_ctemp", "spk_csum",
               "rrf_10", "rrf_30", "rrf_60", "rrf_90",
               "wrrf60", "combsum", "rrf60_boosts"]
    allres = {}
    for s in systems:
        r = run_eval(turns, questions, s)
        allres[s] = r
        lat = r.pop("_latency_ms")
        line = " ".join(f"{c}:{v['any10']:.3f}/{v['any20']:.3f}(n{v['n']})"
                        for c, v in sorted(r.items()))
        print(f"{s:16s} all:{r['_all']['any10']:.3f}/{r['_all']['any20']:.3f} "
              f"| {line} | p50={lat['p50']:.0f}ms", file=sys.stderr)
    with open("/workspace/memorysys/eval/v7/scratch/synth_results.json", "w") as f:
        json.dump(allres, f, indent=1)

if __name__ == "__main__":
    main()
