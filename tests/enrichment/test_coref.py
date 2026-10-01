"""coref_sieve/v1 — conservative deterministic coreference (V7-13.20).

Owned fixture (≥400 pronoun/description cases, written from owned text —
never benchmark gold, V7-22.18): single-candidate resolution, ≥50
two-candidate traps, ≥100 antecedents beyond the immediately previous
turn, number/gender agreement, ``it`` objects-vs-people, speaker
self-reference and addressee resolution, ambiguous ``they``, and
definite descriptions. Precision target ≥ 0.95 on resolved predictions
(H98); recall is measured and reported, never gated.
"""

from __future__ import annotations

import pytest

from verbatim.enrichment.coref_sieve import (
    DEFAULT_LOOKBACK,
    SIEVE_ID,
    explain,
    resolve_antecedent,
    resolve_descriptions,
)


# ---------------------------------------------------------------------------
# Fixture construction (programmatic, deterministic)
# ---------------------------------------------------------------------------

_MALE = ["dan", "marcus", "tom", "raj", "omar", "kevin", "paul",
         "liam", "noah", "ethan", "felix", "hugo", "ivan", "jorge",
         "karl"]
_FEM = ["maya", "alina", "sara", "priya", "nina", "julia", "emma",
        "zoe", "lily", "irene", "cleo", "dora", "elsa", "fiona",
        "greta"]
_GENDERED_CANONS = [  # canon surface carries its own gender noun
    ("her brother", "m"), ("his sister", "f"), ("my dad", "m"),
    ("their mom", "f"), ("the new guy", "m"), ("her aunt", "f"),
    ("my uncle", "m"), ("his wife", "f"), ("her husband", "m"),
    ("the woman", "f"), ("the man", "m"), ("my cousin", None),
    ("her nephew", "m"), ("his niece", "f"), ("their grandmother", "f"),
]
_THINGS = ["the report", "my bike", "the blue folder", "our old server",
           "the recipe", "her laptop", "the proposal", "the garden shed",
           "the rental car", "the coffee machine", "the budget",
           "my phone", "the project plan", "the presentation",
           "his truck"]
_PLURALS = ["my parents", "the kids", "mom and dad", "the neighbors",
            "his siblings", "our friends", "both dogs", "the twins",
            "my grandparents", "the kids next door"]
_COLLECTIVES = ["the team", "her family", "the committee", "the board",
                "our group", "the band", "the staff", "the couple",
                "the jury", "the crew"]
_PERSON_NOUN_CANONS = ["her brother", "the manager", "my doctor",
                       "his roommate", "the neighbor", "her coach",
                       "my landlord", "the intern", "his mentor",
                       "the accountant"]

_FILLER = ["we talked about the weather", "the conversation drifted",
           "it was a quiet afternoon", "we caught up on the news",
           "nothing much happened", "we planned the weekend",
           "work came up briefly", "lunch plans came up",
           "the topic moved on", "a short pause followed"]

_M_PRON = ["he", "him", "his", "himself"]
_F_PRON = ["she", "her", "hers", "herself"]
_THEY = ["they", "them", "their", "themselves"]
_IT = ["it", "its", "itself"]
_FIRST = ["i", "me", "my", "mine", "myself"]
_SECOND = ["you", "your", "yours", "yourself"]
_FIRST_PL = ["we", "us", "our", "ours", "ourselves"]


def _t(speaker, text, canons=(), roles=None):
    d = {"speaker": speaker, "text": text,
         "canon_mentions": list(canons)}
    if roles:
        d["mention_roles"] = dict(roles)
    return d


def _filler(n, speakers=("u1", "u2"), canons=()):
    """n filler turns, alternating speakers, no canons by default."""
    return [_t(speakers[i % len(speakers)], _FILLER[i % len(_FILLER)],
               canons[i % len(canons)] if canons else [])
            for i in range(n)]


def _session(antecedent_turn, depth, current_speaker="u1",
             mention_text="then it came up again", current_canons=()):
    """antecedent turn + (depth-1) fillers + current turn."""
    turns = [antecedent_turn] + _filler(depth - 1)
    turns.append(_t(current_speaker, mention_text, current_canons))
    return turns


FIXTURE: list = []


def _case(cls, mention, turns, expected, ui=None, **kw):
    FIXTURE.append({
        "id": f"{cls}-{len(FIXTURE)}", "cls": cls, "mention": mention,
        "turns": turns, "ui": len(turns) - 1 if ui is None else ui,
        "expected": expected, "kw": kw})


def _build_fixture():
    names = _MALE + _FEM

    # --- single-candidate: resolves ------------------------------------
    for i, name in enumerate(names):                       # 30
        g = "m" if i < len(_MALE) else "f"
        pron = (_M_PRON if g == "m" else _F_PRON)[i % 4]
        d = 1 + i % 3
        _case("single", pron,
              _session(_t("u2", f"{name} was just mentioned", [name]),
                       d), name)
    for i, th in enumerate(_THINGS):                       # 15
        _case("single", _IT[i % 3],
              _session(_t("u1", f"we discussed {th}", [th]),
                       1 + i % 2), th)
    for i, name in enumerate(names[:15]):                  # 15
        _case("single", _THEY[i % 4],
              _session(_t("u2", f"{name} said hi", [name]),
                       1 + i % 2), name)
    for i, (cn, g) in enumerate(_GENDERED_CANONS):         # 15
        if g is None:
            pron = _THEY[i % 4]
        else:
            pron = (_M_PRON if g == "m" else _F_PRON)[i % 4]
        _case("single", pron,
              _session(_t("u1", f"we talked about {cn}", [cn]),
                       1 + i % 3), cn)

    # --- two-candidate traps: abstain ----------------------------------
    for i in range(20):                                    # 20 same turn
        a, b = names[i % len(names)], names[(i + 7) % len(names)]
        pron = _THEY[i % 4] if i % 3 == 0 else (
            _M_PRON[i % 4] if i % 3 == 1 else _F_PRON[i % 4])
        turns = [_t("u1", f"{a} met {b}", [a, b]),
                 _t("u2", f"later {pron} left", [])]
        _case("trap", pron, turns, None)
    for i in range(15):                                    # 15 verified same-gender
        a, b = _FEM[i % len(_FEM)], _FEM[(i + 5) % len(_FEM)]
        turns = [_t("u1", f"{a} saw {b}", [a, b]),
                 _t("u2", "then she smiled", [])]
        _case("trap", "she", turns, None,
              gender={a: "f", b: "f"})
    for i in range(15):                                    # 15 current-turn competitor
        a, b = names[i % len(names)], names[(i + 3) % len(names)]
        turns = [_t("u1", f"{a} arrived", [a]),
                 _t("u2", "small talk", []),
                 _t("u1", f"while talking to {b}, he nodded", [b])]
        _case("trap", "he", turns, None)
    for i in range(15):                                    # 15 two things, "it"
        a, b = _THINGS[i % len(_THINGS)], \
            _THINGS[(i + 4) % len(_THINGS)]
        turns = [_t("u1", f"we compared {a} and {b}", [a, b]),
                 _t("u2", "it broke anyway", [])]
        _case("trap", "it", turns, None)
    for i in range(15):                                    # 15 unverified nearer rival
        a, b = _FEM[i % len(_FEM)], names[(i + 2) % len(names)]
        turns = [_t("u1", f"{a} spoke first", [a]),
                 _t("u2", "a pause", []),
                 _t("u1", f"{b} answered", [b]),
                 _t("u2", "she agreed", [])]
        # b unknown-gender wins on recency but is unverified -> abstain
        _case("trap", "she", turns, None, gender={a: "f"})

    # --- deep antecedents (beyond the previous turn) --------------------
    for i in range(60):                                    # 60 clean d2..d6
        name = names[i % len(names)]
        g = "m" if name in _MALE else "f"
        pool = _M_PRON if g == "m" else _F_PRON
        pron = pool[i % 4] if i % 5 else _THEY[i % 4]
        d = 2 + i % 5
        _case("deep", pron,
              _session(_t("u2", f"{name} was mentioned", [name]), d),
              name)
    for i in range(30):                                    # 30 thing-only fillers
        name = names[(i * 2) % len(names)]
        g = "m" if name in _MALE else "f"
        pron = (_M_PRON if g == "m" else _F_PRON)[i % 4]
        d = 2 + i % 4
        fillers = [_t("u2", f"about {th}", [th])
                   for th in [_THINGS[(i + j) % len(_THINGS)]
                              for j in range(d - 1)]]
        turns = [_t("u1", f"{name} spoke", [name])] + fillers + \
            [_t("u2", f"then {pron} left", [])]
        _case("deep", pron, turns, name)
    for i in range(30):                                    # 30 opposite-gender fillers
        a = _FEM[i % len(_FEM)]
        b = _MALE[i % len(_MALE)]
        d = 2 + i % 4
        fillers = [_t("u2", f"{b} interjected", [b])] + \
            _filler(d - 2)
        turns = [_t("u1", f"{a} led off", [a])] + fillers + \
            [_t("u1", "she continued", [])]
        _case("deep", "she", turns, a,
              gender={a: "f", b: "m"})
    for i in range(20):                                    # 20 beyond lookback
        name = names[i % len(names)]
        g = "m" if name in _MALE else "f"
        pron = (_M_PRON if g == "m" else _F_PRON)[i % 4]
        d = DEFAULT_LOOKBACK + 1 + i % 4
        _case("deep", pron,
              _session(_t("u1", f"{name} was mentioned", [name]), d),
              None)

    # --- number agreement ------------------------------------------------
    for i in range(15):                                    # 15 sg -> pl rejected
        pl = _PLURALS[i % len(_PLURALS)]
        pron = (_M_PRON + _F_PRON)[i % 8]
        _case("num", pron,
              _session(_t("u1", f"we saw {pl}", [pl]), 1 + i % 2),
              None)
    for i in range(10):                                    # 10 it -> pl rejected
        pl = _PLURALS[i % len(_PLURALS)]
        _case("num", _IT[i % 3],
              _session(_t("u1", f"we saw {pl}", [pl]), 1), None)
    for i in range(15):                                    # 15 they -> pl ok
        pl = (_PLURALS + _COLLECTIVES)[i % len(_PLURALS + _COLLECTIVES)]
        _case("num", _THEY[i % 4],
              _session(_t("u1", f"we saw {pl}", [pl]), 1 + i % 2), pl)
    for i in range(10):                                    # 10 he/she -> collective rejected
        coll = _COLLECTIVES[i % len(_COLLECTIVES)]
        pron = (_M_PRON + _F_PRON)[i % 8]
        _case("num", pron,
              _session(_t("u1", f"{coll} met", [coll]), 1), None)

    # --- "it": objects vs people -----------------------------------------
    for i in range(15):                                    # 15 it -> thing
        th = _THINGS[i % len(_THINGS)]
        _case("it", _IT[i % 3],
              _session(_t("u1", f"about {th}", [th]), 1 + i % 2), th)
    for i in range(10):                                    # 10 it -> thing over person
        th = _THINGS[i % len(_THINGS)]
        nm = _FEM[i % len(_FEM)]
        turns = [_t("u1", f"{nm} brought {th}", [nm, th]),
                 _t("u2", "it broke", [])]
        _case("it", "it", turns, th)
    for i in range(15):                                    # 15 it + only people -> None
        cn = _PERSON_NOUN_CANONS[i % len(_PERSON_NOUN_CANONS)]
        _case("it", _IT[i % 3],
              _session(_t("u1", f"{cn} called", [cn]), 1 + i % 2),
              None)
    for i in range(10):                                    # 10 it + bare name -> None
        nm = names[i % len(names)]
        _case("it", "it",
              _session(_t("u1", f"{nm} called", [nm]), 1), None)

    # --- speaker self-reference / addressee ------------------------------
    for i in range(15):                                    # 15 i-forms -> speaker
        form = _FIRST[i % 5]
        turns = _filler(1, speakers=("u2",)) + \
            [_t("u1", f"{form} disagree", [])]
        _case("speaker", form, turns, "u1")
    for i in range(15):                                    # 15 you -> addressee
        form = _SECOND[i % 4]
        turns = [_t("u2", "u2 opened", []),
                 _t("u1", f"{form} said so", [])]
        _case("speaker", form, turns, "u2")
    for i in range(10):                                    # 10 ambiguous addressee
        turns = [_t("u2", "u2 spoke", []),
                 _t("u3", "u3 spoke", []),
                 _t("u1", "you both did", [])]
        _case("speaker", "you", turns, None)
    for i in range(5):                                     # 5 no addressee
        turns = [_t("u1", "u1 alone", []),
                 _t("u1", "you never know", [])]
        _case("speaker", "you", turns, None)
    for i in range(10):                                    # 10 we/us -> group -> None
        form = _FIRST_PL[i % 5]
        turns = _filler(1, speakers=("u2",)) + \
            [_t("u1", f"{form} agreed", ["the team"])]
        _case("speaker", form, turns, None)

    # --- ambiguous they ---------------------------------------------------
    for i in range(20):                                    # 20 ambiguous they
        if i % 3 == 0:
            cands = [_PLURALS[i % len(_PLURALS)],
                     _PLURALS[(i + 3) % len(_PLURALS)]]
        elif i % 3 == 1:
            cands = [names[i % len(names)],
                     names[(i + 6) % len(names)]]
        else:
            cands = [names[i % len(names)],
                     _PLURALS[i % len(_PLURALS)]]
        turns = [_t("u1", "both were discussed", cands),
                 _t("u2", f"{_THEY[i % 4]} left", [])]
        _case("they", _THEY[i % 4], turns, None)
    for i in range(10):                                    # 10 single-canon they
        nm = names[(i * 3) % len(names)]
        _case("they", _THEY[i % 4],
              _session(_t("u2", f"{nm} spoke", [nm]), 1 + i % 3), nm)
    for i in range(10):                                    # 10 they w/ speaker excluded
        other = _MALE[i % len(_MALE)]  # never "maya" (the speaker)
        turns = [_t("maya", "maya was there", ["maya"]),
                 _t("u1", f"so was {other}", [other]),
                 _t("maya", "they agreed", [])]
        # current speaker maya excluded -> single survivor
        _case("they", "they", turns, other)

    # --- definite descriptions ---------------------------------------------
    for i in range(15):                                    # 15 unique attr match
        cn = f"{_FEM[i % len(_FEM)]} from work"
        other = _MALE[i % len(_MALE)]
        turns = [_t("u1", f"{cn} and {other} joined", [cn, other]),
                 _t("u2", "the woman from work spoke", [])]
        _case("desc", "the woman from work", turns, cn)
    for i in range(15):                                    # 15 attribute tie -> None
        a = f"{_MALE[i % len(_MALE)]} from work"
        b = f"{_MALE[(i + 4) % len(_MALE)]} from work"
        turns = [_t("u1", f"{a} and {b} joined", [a, b]),
                 _t("u2", "the guy from work spoke", [])]
        _case("desc", "the guy from work", turns, None)
    for i in range(10):                                    # 10 zero overlap -> None
        cn = names[i % len(names)]
        turns = [_t("u1", f"{cn} joined", [cn]),
                 _t("u2", "the woman from work spoke", [])]
        _case("desc", "the woman from work", turns, None)
    for i in range(10):                                    # 10 gender-filtered desc
        cn = _FEM[i % len(_FEM)]
        turns = [_t("u1", f"{cn} joined", [cn]),
                 _t("u2", "the man spoke", [])]
        _case("desc", "the man", turns, None, gender={cn: "f"})
    for i in range(5):                                     # 5 plural desc resolve
        cn = "the brothers"
        turns = [_t("u1", f"{cn} visited", [cn]),
                 _t("u2", "the two brothers laughed", [])]
        _case("desc", "the two brothers", turns, cn)
    for i in range(5):                                     # 5 plural desc vs sg -> None
        cn = _MALE[i % len(_MALE)]
        turns = [_t("u1", f"{cn} visited", [cn]),
                 _t("u2", "the two brothers laughed", [])]
        _case("desc", "the two brothers", turns, None)


_build_fixture()


def _run(case):
    kw = dict(case["kw"])
    return resolve_antecedent(case["mention"], case["ui"], case["turns"],
                            lookback=kw.pop("lookback", DEFAULT_LOOKBACK),
                            **kw)


def _stats():
    resolved = errors = 0
    expected_res = expected_abs = got_res = 0
    wrong = []
    for c in FIXTURE:
        pred = _run(c)
        if c["expected"] is not None:
            expected_res += 1
        else:
            expected_abs += 1
        if pred is not None:
            got_res += 1
            if pred == c["expected"]:
                resolved += 1
            else:
                errors += 1
                wrong.append((c["id"], c["cls"], c["mention"],
                              pred, c["expected"]))
    return resolved, errors, got_res, expected_res, expected_abs, wrong


# ---------------------------------------------------------------------------
# Fixture-level requirements
# ---------------------------------------------------------------------------

def test_fixture_composition():
    assert len(FIXTURE) >= 400
    deep = [c for c in FIXTURE
            if c["cls"] == "deep" and c["expected"] is not None]
    assert len(deep) >= 100, "≥100 antecedents beyond previous turn"
    traps = [c for c in FIXTURE if c["cls"] == "trap"]
    assert len(traps) >= 50, "≥50 two-candidate traps"


def test_fixture_precision_and_recall():
    resolved, errors, got_res, exp_res, exp_abs, wrong = _stats()
    precision = resolved / got_res if got_res else 1.0
    recall = resolved / exp_res if exp_res else 0.0
    print(f"\ncoref fixture: {len(FIXTURE)} cases | resolved {got_res} "
          f"| precision {precision:.3f} | recall {recall:.3f} "
          f"| errors {errors}")
    for w in wrong:
        print("  WRONG:", w)
    assert precision >= 0.95, f"precision {precision} < 0.95: {wrong[:5]}"
    assert errors == 0


def test_traps_always_abstain():
    for c in FIXTURE:
        if c["cls"] == "trap":
            assert _run(c) is None, f"trap resolved: {c['mention']}"


def test_deep_within_lookback_resolves():
    for c in FIXTURE:
        if c["cls"] == "deep" and c["expected"] is not None:
            assert _run(c) == c["expected"], \
                f"deep case failed: {c['mention']}"


def test_beyond_lookback_abstains():
    for c in FIXTURE:
        if c["cls"] == "deep" and c["expected"] is None:
            assert _run(c) is None


def test_number_agreement():
    for c in FIXTURE:
        if c["cls"] == "num":
            assert _run(c) == c["expected"], \
                f"number case failed: {c['mention']} -> {_run(c)}"


def test_speaker_rules():
    for c in FIXTURE:
        if c["cls"] == "speaker":
            assert _run(c) == c["expected"], \
                f"speaker case failed: {c['mention']} -> {_run(c)}"


def test_descriptions():
    for c in FIXTURE:
        if c["cls"] == "desc":
            assert _run(c) == c["expected"], \
                f"desc case failed: {c['mention']} -> {_run(c)}"


# ---------------------------------------------------------------------------
# Unit-level semantics
# ---------------------------------------------------------------------------

class TestSpeakerAlternation:
    def test_i_resolves_to_speaker(self):
        turns = [_t("u1", "u1 spoke"), _t("u2", "I disagree")]
        assert resolve_antecedent("I", 1, turns) == "u2"

    def test_you_resolves_to_addressee(self):
        turns = [_t("alice", "hi"), _t("bob", "you said so")]
        assert resolve_antecedent("you", 1, turns) == "alice"

    def test_we_never_resolves(self):
        turns = [_t("alice", "hi"), _t("bob", "we decided")]
        assert resolve_antecedent("we", 1, turns) is None

    def test_speaker_excluded_from_third_person(self):
        turns = [_t("maya", "maya arrived", ["maya"]),
                 _t("maya", "she left", [])]
        # speaker canon excluded -> no candidates -> abstain
        assert resolve_antecedent("she", 1, turns) is None


class TestAgreementFilters:
    def test_singular_rejects_plural(self):
        turns = [_t("u1", "my parents visited", ["my parents"]),
                 _t("u2", "he called", [])]
        assert resolve_antecedent("he", 1, turns) is None

    def test_they_may_be_singular(self):
        turns = [_t("u1", "maya arrived", ["maya"]),
                 _t("u2", "they left", [])]
        assert resolve_antecedent("they", 1, turns) == "maya"

    def test_gender_map_disambiguates(self):
        turns = [_t("u1", "alina and dan met", ["alina", "dan"]),
                 _t("u2", "she left", [])]
        assert resolve_antecedent(
            "she", 1, turns, gender={"alina": "f", "dan": "m"}) == "alina"

    def test_never_guess_gender_from_names(self):
        # alina/dan without gender info -> both plausible -> abstain
        turns = [_t("u1", "alina and dan met", ["alina", "dan"]),
                 _t("u2", "she left", [])]
        assert resolve_antecedent("she", 1, turns) is None

    def test_it_prefers_things(self):
        turns = [_t("u1", "maya brought the bike", ["maya", "the bike"]),
                 _t("u2", "it broke", [])]
        assert resolve_antecedent("it", 1, turns) == "the bike"

    def test_it_rejects_people(self):
        turns = [_t("u1", "her brother called", ["her brother"]),
                 _t("u2", "it broke", [])]
        assert resolve_antecedent("it", 1, turns) is None


class TestRecencyAndLookback:
    def test_recency_dominates(self):
        turns = [_t("u1", "the report came up", ["the report"]),
                 _t("u2", "filler", []),
                 _t("u1", "the folder surfaced", ["the folder"]),
                 _t("u2", "it broke", [])]
        # folder at d1 vs report at d3: gap 26 > margin -> folder
        assert resolve_antecedent("it", 3, turns) == "the folder"

    def test_they_two_candidates_is_group_ambiguity(self):
        turns = [_t("u1", "maya spoke", ["maya"]),
                 _t("u2", "filler", []),
                 _t("u1", "dan replied", ["dan"]),
                 _t("u2", "they continued", [])]
        # "they" may denote {maya, dan} — a referent no canon carries
        assert resolve_antecedent("they", 3, turns) is None

    def test_they_never_resolves_to_single_thing(self):
        turns = [_t("u1", "the report arrived", ["the report"]),
                 _t("u2", "they are done", [])]
        # a lone singular *thing* is "it", not "they" -> no candidates
        assert resolve_antecedent("they", 1, turns) is None

    def test_they_singular_when_only_alternative_is_thing(self):
        turns = [_t("u1", "maya reviewed the report",
                    ["maya", "the report"]),
                 _t("u2", "they left early", [])]
        # the report is a pure singular thing -> excluded; maya is the
        # sole survivor -> singular they
        assert resolve_antecedent("they", 1, turns) == "maya"

    def test_same_unit_candidates(self):
        turns = [_t("u1", "earlier", []),
                 _t("u2", "alina told him", ["alina"])]
        # d=0 canon is a candidate
        assert resolve_antecedent("she", 1, turns) == "alina"

    def test_beyond_lookback_abstains(self):
        turns = [_t("u1", "maya spoke", ["maya"])] + _filler(7)
        turns.append(_t("u2", "she left", []))
        assert resolve_antecedent("she", 8, turns, lookback=6) is None

    def test_lookback_param(self):
        turns = [_t("u1", "maya spoke", ["maya"])] + _filler(2)
        turns.append(_t("u2", "she left", []))
        assert resolve_antecedent("she", 3, turns, lookback=2) is None
        assert resolve_antecedent("she", 3, turns, lookback=3) == "maya"


class TestPreviousTurnBaseline:
    def test_flag_restricts_to_previous_turn(self):
        turns = [_t("u1", "maya spoke", ["maya"]),
                 _t("u2", "filler", []),
                 _t("u1", "she left", [])]
        assert resolve_antecedent("she", 2, turns) == "maya"
        assert resolve_antecedent("she", 2, turns,
                                  last_resort_previous_turn=True) is None

    def test_flag_still_resolves_d1(self):
        turns = [_t("u1", "maya spoke", ["maya"]),
                 _t("u2", "she left", [])]
        assert resolve_antecedent("she", 1, turns,
                                  last_resort_previous_turn=True) == "maya"

    def test_flag_still_abstains_on_traps(self):
        turns = [_t("u1", "alina met sara", ["alina", "sara"]),
                 _t("u2", "she left", [])]
        assert resolve_antecedent("she", 1, turns,
                                  last_resort_previous_turn=True) is None


class TestDescriptions:
    def test_unique_attribute_match(self):
        turns = [_t("u1", "nina from work and dan joined",
                    ["nina from work", "dan"]),
                 _t("u2", "the woman from work spoke", [])]
        assert resolve_descriptions("the woman from work", 1, turns) \
            == "nina from work"

    def test_attribute_tie_abstains(self):
        turns = [_t("u1", "dan from work and omar from work joined",
                    ["dan from work", "omar from work"]),
                 _t("u2", "the guy from work spoke", [])]
        assert resolve_descriptions("the guy from work", 1, turns) is None

    def test_no_attribute_match(self):
        turns = [_t("u1", "dan joined", ["dan"]),
                 _t("u2", "the woman from work spoke", [])]
        assert resolve_descriptions("the woman from work", 1, turns) is None

    def test_antecedent_via_description_in_text(self):
        # "the new PM"-class nominal resolves through antecedent API too
        turns = [_t("u1", "the new pm started", ["the new pm"]),
                 _t("u2", "the new pm spoke", [])]
        assert resolve_antecedent("the new pm", 1, turns) == "the new pm"


class TestEdgeCases:
    def test_empty_session(self):
        assert resolve_antecedent("he", 0, []) is None

    def test_empty_mention(self):
        turns = [_t("u1", "maya", ["maya"])]
        assert resolve_antecedent("", 0, turns) is None
        assert resolve_antecedent(None, 0, turns) is None

    def test_unsupported_forms(self):
        turns = [_t("u1", "maya spoke", ["maya"]),
                 _t("u2", "who left", [])]
        for m in ("who", "which", "someone", "everyone", "this"):
            assert resolve_antecedent(m, 1, turns) is None, m

    def test_identity_mention(self):
        turns = [_t("u1", "maya spoke", ["maya"]),
                 _t("u2", "about maya", ["maya"])]
        assert resolve_antecedent("maya", 1, turns) == "maya"

    def test_canon_object_mentions(self):
        # non-dict turns degrade gracefully
        assert resolve_antecedent("he", 0, [None, {}]) is None

    def test_determinism(self):
        c = FIXTURE[0]
        assert _run(c) == _run(c) == _run(c)


class TestExplain:
    def test_explain_reports_status(self):
        turns = [_t("u1", "alina met sara", ["alina", "sara"]),
                 _t("u2", "she left", [])]
        out = explain("she", 1, turns)
        assert out["sieve"] == SIEVE_ID
        assert out["canon"] is None
        assert out["status"] == "unverified_gender_competition"

    def test_explain_resolved(self):
        turns = [_t("u1", "maya spoke", ["maya"]),
                 _t("u2", "she left", [])]
        out = explain("she", 1, turns)
        assert out["canon"] == "maya"
        assert out["status"] == "resolved_single"


def test_sieve_id():
    assert SIEVE_ID == "coref_sieve/v1"
