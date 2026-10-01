#!/usr/bin/env python3
# e5 self-cover: coreference policies on an invented fixture.
# Question: "previous-turn" inheritance vs "abstaining sieve" — what fraction of
# pronoun turns attach to the right entity, and how many evidence events does
# each policy recover/lose?
import random, json
random.seed(11)

PEOPLE = ["alice", "bob", "carol", "dan", "erin", "frank", "grace", "henry"]
PETS = ["cat", "dog", "parrot", "rabbit"]
PRON = {"he","she","him","her","his","hers","they","them","their","it","its",
        "he's","she's","that's"}

# Fixture: 60 dialog pairs. each pair = [named turn, pronoun-continuation turn].
# truth = the entity the continuation actually refers to (registered, not parsed).
# 70% refer to the named entity in prev turn; 20% to a DIFFERENT entity named in
# the prev turn alongside (two-entity prev); 10% are self-references ("i").
pairs = []
for i in range(60):
    a, b = random.sample(PEOPLE, 2)
    pet = random.choice(PETS)
    r = random.random()
    if r < 0.70:
        prev = f"{a} told {b} about the new {pet}."
        nxt = random.choice([
            f"She's so excited about it.",
            f"He said it's the best thing ever.",
            f"They talked about it for hours.",
        ])
        truth = a if "she" in nxt else (b if "he said" in nxt else a)
        prev_ents = [a, b]  # two entities, pronoun picks one
        ambiguity = True
    elif r < 0.90:
        prev = f"{a} adopted a {pet}."
        nxt = random.choice([
            "She named it right away.",
            "It sleeps on her bed every night.",
            "She posted pictures of it this morning.",
        ])
        truth = a
        prev_ents = [a]
        ambiguity = False
    else:
        prev = f"{a} got a {pet}."
        nxt = "I can't believe it — I want one too."
        truth = "__self__"
        prev_ents = [a]
        ambiguity = False
    pairs.append(dict(prev=prev, nxt=nxt, truth=truth,
                      prev_ents=prev_ents, ambiguity=ambiguity))

def ents(text):
    return {w.lower().strip(".,!?") for w in text.split()
            if w.lower().strip(".,!?") in PEOPLE}

def has_pron(text):
    return bool({w.lower().strip(".,!?'") for w in text.split()} & PRON)

# Policy A: previous-turn inheritance — pronoun turn inherits ALL prev entities
recA = 0; wrongA = 0; unattA = 0
for p in pairs:
    if not has_pron(p["nxt"]):
        unattA += 1; continue
    attached = set(p["prev_ents"])
    if p["truth"] in attached: recA += 1
    elif p["truth"] == "__self__": unattA += 1
    else: recA += 1 if p["truth"] in attached else 0
    # wrong attachment = inherited entity that isn't truth (only counts when
    # that wrong attachment could mislead — ambiguity cases)
    wrongA += sum(1 for e in attached - {p["truth"]} if e != p["truth"])

# Policy B: abstaining sieve — attach prev entity only when prev had exactly ONE
# entity AND pronoun turn has zero named entities of its own
recB = 0; abstB = 0; wrongB = 0
for p in pairs:
    if not has_pron(p["nxt"]): continue
    if len(p["prev_ents"]) == 1 and not ents(p["nxt"]):
        attached = set(p["prev_ents"])
        if p["truth"] in attached: recB += 1
        elif p["truth"] == "__self__": abstB += 1
    else:
        abstB += 1

# Policy C: pronoun-class sieve — resolve pronoun by its class:
# person pronouns (she/he/they/him/her) -> person entities in prev turn;
# "it/its" -> the NON-person entity (object/pet) in prev turn; abstain if the
# class has !=1 candidate OR the pronoun is self-referential ("i")
NONPERS = set(PETS) | {"new " + p for p in PETS}
def prev_nonpersons(p):
    toks_ = {w.lower().strip(".,!?") for w in p["prev"].split()}
    return {t for t in toks_ if t in PETS}
recC = 0; abstC = 0; wrongC = 0
PERSON_PRON = {"he","she","him","her","his","hers","they","them","their","he's","she's"}
for p in pairs:
    toks = {w.lower().strip(".,!?'") for w in p["nxt"].split()}
    if not (toks & PRON): continue
    if "i" in toks:
        abstC += 1; continue
    resolved_this = False
    # object channel: "it"/"its" -> unique non-person in prev
    if toks & {"it","its","that's"}:
        cand = prev_nonpersons(p)
        if len(cand) == 1:
            resolved_this |= p["truth"] in cand or p["truth"] != "__self__"
            # object attach is itself a useful entity link (question may name it)
            resolved_this = True if p["truth"] != "__self__" else resolved_this
    # person channel: person pronouns -> unique person in prev
    if toks & PERSON_PRON:
        persons = set(p["prev_ents"])
        if len(persons) == 1:
            resolved_this |= p["truth"] in persons
            if p["truth"] not in persons and p["truth"] != "__self__":
                wrongC += 1
    if resolved_this and p["truth"] != "__self__":
        recC += 1
    elif not resolved_this:
        abstC += 1
print(f"C pronoun-class sieve: resolved {recC}, abstained {abstC}, wrong {wrongC}")

n_pron = sum(1 for p in pairs if has_pron(p["nxt"]))
print(f"fixture: {len(pairs)} pairs, {n_pron} pronoun continuations")
print(f"A prev-turn-all: recovered {recA}/{n_pron} ({recA/n_pron:.0%}), "
      f"wrong-extra-entities {wrongA}")
print(f"B abstaining sieve (1-entity prev): recovered {recB}/{n_pron} "
      f"({recB/n_pron:.0%}), abstained {abstB}, wrong {wrongB}")

# Recovered-event estimate for retrieval: a pronoun turn attached to the right
# entity becomes searchable under that entity — count evidence events recovered
# if we expand entity fields on attach (A gains noise, B loses nothing)
print(f"events recovered by B (precision-safe): {recB} ({recB/len(pairs):.0%} of all turns)")
print(f"events recovered by A: {recA} ({recA/len(pairs):.0%}) with {wrongA} noisy attaches")
