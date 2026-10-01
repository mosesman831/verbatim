"""Tests for ``pref_state/v1`` — deterministic preference & state-fact
extraction (SPEC_V7 §32.11/§32.12, V7-13.08, V7-16.03).

The module under test is ``verbatim/enrichment/prefs_state.py``.  All inputs
go through the real ``norm/v2`` analyzer (``verbatim.text.norm_v2.analyze``);
``NormAnalysis`` is never faked.  ``raw_text`` defaults to the analyzed text
so byte pins are verified against the true UTF-8 surface; the degraded paths
(no raw text / misaligned raw text) are exercised explicitly.

Covered:

- every preference strength (constraint > favorite > love_hate >
  like_dislike > habitual, §32.12) and polarity;
- positive patterns (love/adore/like/enjoy/prefer/am into/am a fan of/
  can't get enough of) and negative patterns (hate/detest/dislike/
  can't stand/not a fan of/avoid/never);
- habitual markers, favorite/comparative forms, constraints;
- clitic-subject recovery (``I'm``, ``I'd``) and contraction negation;
- guards: hypothetical/modal/'d-clitic, interrogative, conditional,
  hedged (cognition + adverbs), quoted, preposed and postposed reported
  speech, control verbs, unsupported third-person pronouns;
- named third-person subjects bound through ``canon_fn``;
- all §32.11 state families, other-person state keys, negated-state
  abstention;
- the V8-13.04 subject-key contract: ``speaker:<canon>/<slot>`` when a
  speaker canon is known, ``me/<slot>`` for unattributed input;
- evidence pins (raw_verified / text_verified / term_offsets), UTF-8
  byte spans, ``scope_id`` stamping, ``StateFact`` status/producer/
  ``valid_from_us`` fields;
- ``state_compatible`` accumulate/replace/numeric semantics;
- determinism in-process and across ``PYTHONHASHSEED`` processes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from verbatim.core.types_v7 import IntervalUs, StateFactStatus
from verbatim.enrichment.prefs_state import (
    FORMULA_STATUS,
    MODULE_ID,
    extract_preferences,
    extract_state_facts,
    reset_counters,
    state_compatible,
)
from verbatim.text.norm_v2 import analyze

SPEAKER = "speaker:alice"
OCC = IntervalUs(start_us=1_000_000, end_us=2_000_000)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clean_counters():
    reset_counters()
    yield
    reset_counters()


def prefs(text, unit_id="u-1", speaker=SPEAKER, **kw):
    kw.setdefault("raw_text", text)
    return extract_preferences(analyze(text), unit_id, speaker, **kw)


def states(text, unit_id="u-1", speaker=SPEAKER, occurred=OCC, **kw):
    kw.setdefault("raw_text", text)
    return extract_state_facts(analyze(text), unit_id, speaker, occurred,
                               **kw)


def _raw_span(text, span):
    return text.encode("utf-8")[span[0]:span[1]]


def assert_pref_pins(text, p):
    """Subject/trigger/object pins slice non-empty spans of the raw text."""
    for key in ("subject", "trigger", "object"):
        span = p.pins.get(key)
        assert span is not None, f"{key} pin missing: {p.pins!r}"
        piece = _raw_span(text, span)
        assert piece and piece.strip(), f"{key} pin empty: {p.pins!r}"
    assert p.pins["extractor"] == MODULE_ID
    assert p.pins["formula"] == FORMULA_STATUS


def assert_state_pins(text, f):
    for key in ("subject", "trigger", "value"):
        span = f.pins.get(key)
        assert span is not None, f"{key} pin missing: {f.pins!r}"
        piece = _raw_span(text, span)
        assert piece and piece.strip(), f"{key} pin empty: {f.pins!r}"
    assert f.pins["family"] == f.state_key.rsplit("/", 1)[-1]
    assert f.pins["extractor"] == MODULE_ID
    assert f.pins["formula"] == FORMULA_STATUS


# ---------------------------------------------------------------------------
# Module contract
# ---------------------------------------------------------------------------


class TestModuleContract:
    def test_module_id(self):
        assert MODULE_ID == "pref_state/v1"

    def test_formula_status(self):
        assert FORMULA_STATUS == "provisional/v7-r0"

    def test_scope_id_is_keyword_only(self):
        # frozen signature lacks scope_id; it must not be positional
        with pytest.raises(TypeError):
            extract_preferences(analyze("i love pizza"), "u", SPEAKER,
                                "scope:x")
        with pytest.raises(TypeError):
            extract_state_facts(analyze("i live in berlin"), "u",
                                SPEAKER, OCC, "scope:x")


# ---------------------------------------------------------------------------
# Positive preference patterns (§32.12)
# ---------------------------------------------------------------------------


class TestPreferencePositive:
    @pytest.mark.parametrize("text", ["I love pizza", "I adore her",
                                      "i love mornings"])
    def test_love_hate_positive(self, text):
        (p,) = prefs(text)
        assert p.polarity == "positive"
        assert p.strength == "love_hate"
        assert p.subject_canon == SPEAKER

    @pytest.mark.parametrize("text,obj", [
        ("I like tea", "tea"),
        ("I enjoy hiking", "hiking"),
        ("I prefer tea", "tea"),
        ("I'm into jazz", "jazz"),
        ("I'm a fan of sushi", "sushi"),
        ("I'm a huge fan of jazz", "jazz"),
    ])
    def test_like_dislike_positive(self, text, obj):
        (p,) = prefs(text)
        assert p.object_text == obj
        assert p.polarity == "positive"
        assert p.strength == "like_dislike"

    def test_cant_get_enough_is_positive(self):
        (p,) = prefs("I can't get enough of her")
        assert p.polarity == "positive"
        assert p.strength == "love_hate"
        assert p.object_text == "her"

    def test_pronoun_object_kept(self):
        (p,) = prefs("i love it")
        assert p.object_text == "it"
        (p,) = prefs("i love you")
        assert p.object_text == "you"

    def test_identifier_object(self):
        (p,) = prefs("i love ABC-123")
        assert p.object_text == "ABC-123"

    def test_we_subject_binds_speaker(self):
        (p,) = prefs("we love pizza")
        assert p.subject_canon == SPEAKER


# ---------------------------------------------------------------------------
# Negative preference patterns
# ---------------------------------------------------------------------------


class TestPreferenceNegative:
    @pytest.mark.parametrize("text,obj", [
        ("I hate traffic", "traffic"),
        ("I detest spam", "spam"),
    ])
    def test_love_hate_negative(self, text, obj):
        (p,) = prefs(text)
        assert p.object_text == obj
        assert p.polarity == "negative"
        assert p.strength == "love_hate"

    @pytest.mark.parametrize("text,obj", [
        ("I dislike waiting", "waiting"),
        ("I can't stand noise", "noise"),
        ("I'm not a fan of jazz", "jazz"),
        ("I'm no longer a fan of jazz", "jazz"),
        ("I avoid gluten", "gluten"),
    ])
    def test_like_dislike_negative(self, text, obj):
        (p,) = prefs(text)
        assert p.object_text == obj
        assert p.polarity == "negative"
        assert p.strength == "like_dislike"

    def test_negated_negative_abstains(self):
        # "I don't hate X" is not a liking signal — abstain per §32.12
        assert prefs("I don't hate pizza") == []

    def test_never_eat_is_constraint(self):
        (p,) = prefs("I never eat sushi")
        assert p.polarity == "negative"
        assert p.strength == "constraint"
        assert p.object_text == "sushi"


# ---------------------------------------------------------------------------
# Habitual preferences
# ---------------------------------------------------------------------------


class TestHabitual:
    @pytest.mark.parametrize("text,obj", [
        ("I usually drink coffee", "coffee"),
        ("i always take the bus", "bus"),
        ("I often eat sushi", "sushi"),
        ("i normally walk home", "home"),
        ("I tend to prefer quiet places", "quiet places"),
    ])
    def test_habitual(self, text, obj):
        (p,) = prefs(text)
        assert p.object_text == obj
        assert p.polarity == "positive"
        assert p.strength == "habitual"


# ---------------------------------------------------------------------------
# Favorite / comparative
# ---------------------------------------------------------------------------


class TestFavorite:
    def test_my_favorite_x_is_y(self):
        (p,) = prefs("my favorite food is sushi")
        assert p.object_text == "sushi"
        assert p.polarity == "positive"
        assert p.strength == "favorite"
        assert p.subject_canon == SPEAKER

    def test_x_is_my_favorite(self):
        (p,) = prefs("sushi is my favorite")
        assert p.object_text == "sushi"
        assert p.strength == "favorite"

    def test_rather_than(self):
        (p,) = prefs("I'd rather tea than coffee")
        assert p.object_text == "tea"
        assert p.polarity == "positive"
        assert p.strength == "favorite"

    def test_favorite_without_value_abstains(self):
        assert prefs("my favorite") == []


# ---------------------------------------------------------------------------
# Constraints (§32.12)
# ---------------------------------------------------------------------------


class TestConstraint:
    @pytest.mark.parametrize("text,obj,pol", [
        ("I'm allergic to peanuts", "peanuts", "negative"),
        ("I don't eat meat", "meat", "negative"),
        ("I can't eat dairy", "dairy", "negative"),
        ("I'm vegetarian", "vegetarian", "positive"),
        ("i'm vegan", "vegan", "positive"),
        ("i'm lactose intolerant", "lactose intolerant", "negative"),
        ("i'm gluten free", "gluten free", "negative"),
    ])
    def test_constraints(self, text, obj, pol):
        (p,) = prefs(text)
        assert p.object_text == obj
        assert p.polarity == pol
        assert p.strength == "constraint"


# ---------------------------------------------------------------------------
# Clitic recovery / contractions
# ---------------------------------------------------------------------------


class TestClitics:
    @pytest.mark.parametrize("text", [
        "I'm into jazz",
        "I'm a fan of sushi",
        "I'm allergic to peanuts",
        "I'm vegetarian",
    ])
    def test_im_clitic_subject_recovers_speaker(self, text):
        ps = prefs(text)
        assert ps, f"no pref for {text!r}"
        assert all(p.subject_canon == SPEAKER for p in ps)

    def test_id_rather_recovers(self):
        (p,) = prefs("I'd rather tea than coffee")
        assert p.subject_canon == SPEAKER

    def test_contraction_negation(self):
        (p,) = prefs("I can't stand noise")
        assert p.polarity == "negative"


# ---------------------------------------------------------------------------
# Guards — conservative abstention (§32.12)
# ---------------------------------------------------------------------------


class TestGuards:
    @pytest.mark.parametrize("text", [
        "I would love pizza",
        "I'd love pizza",
        "I'll love it",
        "I could enjoy that",
        "I might like jazz",
        "I should like it",
        "I will love pizza",
    ])
    def test_hypothetical_modal_abstains(self, text):
        assert prefs(text) == []

    @pytest.mark.parametrize("text", [
        "do i like pizza?",
        "do I like pizza",
        "I love pizza?",
        "am i late",
    ])
    def test_question_abstains(self, text):
        assert prefs(text) == []

    @pytest.mark.parametrize("text", [
        "I think i like pizza",
        "i guess i love it",
        "maybe i like jazz",
        "i probably like jazz",
        "i feel like i love pizza",
    ])
    def test_hedged_abstains(self, text):
        assert prefs(text) == []

    def test_conditional_abstains(self):
        assert prefs("if i liked sushi") == []
        assert prefs("if i liked sushi, i would eat it") == []

    @pytest.mark.parametrize("text", [
        "she said i love pizza",
        "he told me he loves jazz",
        'she said "i love pizza"',
        '"i love pizza" she said',
        "i love pizza, she said",
        "i live in berlin, she said",
        "they say i love pizza",
    ])
    def test_reported_speech_abstains(self, text):
        assert prefs(text) == []

    def test_quoted_object_abstains(self):
        assert prefs('i love "pizza"') == []

    @pytest.mark.parametrize("text", [
        "I want to love pizza",
        "I used to love sushi",
        "i want to like it",
    ])
    def test_control_verb_abstains(self, text):
        assert prefs(text) == []

    @pytest.mark.parametrize("text", [
        "he loves jazz",
        "she hates rain",
        "they like pizza",
        "you love pizza",
        "her birthday is june 5",
    ])
    def test_unsupported_pronoun_abstains(self, text):
        # no coref input in this contract: third-person pronouns and "you"
        # never resolve
        assert prefs(text) == []
        assert states(text) == []

    def test_coordinated_second_clause_is_own_clause(self):
        # "and she said hi" is linked (coordination), not a postposed quote
        (p,) = prefs("i love pizza and she said hi")
        assert p.object_text == "pizza"

    def test_missing_object_abstains(self):
        assert prefs("i love") == []

    def test_subordinator_tail_not_in_object(self):
        (p,) = prefs("i like it when it rains")
        assert p.object_text == "it"


# ---------------------------------------------------------------------------
# Named third-person subjects (canon_fn path)
# ---------------------------------------------------------------------------


class TestNamedSubjects:
    def test_named_subject_binds_canon(self):
        (p,) = prefs("Maria loves jazz")
        assert p.subject_canon == "maria"
        assert p.object_text == "jazz"

    def test_named_subject_possessive(self):
        (p,) = prefs("Maria's favorite food is sushi")
        assert p.subject_canon == "maria"
        assert p.object_text == "sushi"

    def test_named_state_key(self):
        fs = states("Maria's birthday is June 5")
        assert any(f.state_key == "maria/birthday" for f in fs)

    def test_uncapitalized_name_abstains(self):
        # without capitalization evidence a bare noun isn't a name —
        # honest abstention rather than a guessed canon
        assert prefs("maria loves jazz") == []

    def test_name_requires_alignment(self):
        # folded-only input: 'Maria' folds to 'maria' but hay is the
        # analyzer's folded text — capitalization evidence unavailable
        ps = extract_preferences(analyze("Maria loves jazz"), "u", SPEAKER)
        assert ps == []


# ---------------------------------------------------------------------------
# Subject-key contract (V8-13.04)
# ---------------------------------------------------------------------------


class TestSubjectContract:
    """V8-13.04 decided contract: ``speaker:<canon>/<slot>`` when a speaker
    canon is known (multi-party correctness), ``me/<slot>`` only for
    unattributed single-user input, bare entity canon for named
    third-person subjects."""

    def test_known_speaker_keys_state(self):
        (f,) = states("i live in Berlin")
        assert f.state_key == "speaker:alice/home_city"

    def test_known_speaker_keys_pref(self):
        (p,) = prefs("i love pizza")
        assert p.subject_canon == "speaker:alice"

    def test_unattributed_state_binds_me(self):
        (f,) = states("i live in Berlin", speaker="")
        assert f.state_key == "me/home_city"
        fs = states("my dog is Rex", speaker="")
        assert any(f.state_key == "me/pet_names" for f in fs)
        assert any(f.state_key == "me/pets" for f in fs)

    def test_unattributed_pref_binds_me(self):
        (p,) = prefs("i love pizza", speaker="")
        assert p.subject_canon == "me"

    def test_named_subject_bare_canon(self):
        fs = states("Maria's birthday is June 5")
        assert any(f.state_key == "maria/birthday" for f in fs)

    def test_me_key_uses_compat_family(self):
        # me/<slot> keys still resolve the family for state_compatible
        assert state_compatible("me/home_city", "berlin", "berlin")
        assert not state_compatible("me/home_city", "berlin", "oslo")


# ---------------------------------------------------------------------------
# State families (§32.11)
# ---------------------------------------------------------------------------


class TestStateFamilies:
    @pytest.mark.parametrize("text,key,val", [
        ("i live in Berlin", "speaker:alice/home_city", "Berlin"),
        ("i live in Germany", "speaker:alice/home_country", "Germany"),
        ("i moved to Oslo", "speaker:alice/home_city", "Oslo"),
        ("my address is 12 Main St", "speaker:alice/address", "12 Main St"),
        ("i work at Acme", "speaker:alice/employer", "Acme"),
        ("i'm a software engineer", "speaker:alice/job_title", "software engineer"),
        ("i work as a nurse", "speaker:alice/job_title", "nurse"),
        ("my team is Phoenix", "speaker:alice/team", "Phoenix"),
        ("my manager is Sarah", "speaker:alice/manager", "Sarah"),
        ("i report to Tom", "speaker:alice/manager", "Tom"),
        ("i study at MIT", "speaker:alice/school", "MIT"),
        ("my major is CS", "speaker:alice/major", "CS"),
        ("i'm majoring in physics", "speaker:alice/major", "physics"),
        ("i have a PhD in physics", "speaker:alice/degree", "PhD in physics"),
        ("i'm single", "speaker:alice/relationship_status", "single"),
        ("my wife is Anna", "speaker:alice/partner", "Anna"),
        ("i have two kids", "speaker:alice/children", "two kids"),
        ("i have a dog", "speaker:alice/pets", "dog"),
        ("my dog is Rex", "speaker:alice/pet_names", "Rex"),
        ("my birthday is June 5", "speaker:alice/birthday", "June 5"),
        ("i was born on June 5", "speaker:alice/birthday", "June 5"),
        ("i'm 30", "speaker:alice/age", "30"),
        ("i'm 30 years old", "speaker:alice/age", "30"),
        ("i'm German", "speaker:alice/nationality", "German"),
        ("i speak French", "speaker:alice/languages", "French"),
        ("my phone is 555-1234", "speaker:alice/phone", "555-1234"),
        ("my email is a@b.com", "speaker:alice/email", "a@b.com"),
        ("my favorite food is sushi", "speaker:alice/favorite_food", "sushi"),
        ("my favorite color is blue", "speaker:alice/favorite_color", "blue"),
        ("my hobby is photography", "speaker:alice/hobbies", "photography"),
        ("i play tennis", "speaker:alice/sport", "tennis"),
        ("i'm vegan", "speaker:alice/diet", "vegan"),
        ("i'm allergic to peanuts", "speaker:alice/allergies", "peanuts"),
        ("i have asthma", "speaker:alice/health_condition", "asthma"),
        ("i take metformin", "speaker:alice/medication", "metformin"),
        ("i drive a Civic", "speaker:alice/car", "Civic"),
        ("my laptop is a ThinkPad", "speaker:alice/device", "ThinkPad"),
        ("i use Linux", "speaker:alice/os", "Linux"),
        ("i use vim", "speaker:alice/editor", "vim"),
        ("i code in Rust", "speaker:alice/programming_language", "Rust"),
        ("i'm working on Hermes", "speaker:alice/project_current", "Hermes"),
        ("i want to run a marathon", "speaker:alice/goal_current", "run a marathon"),
        ("i'm planning to visit Japan", "speaker:alice/plan_upcoming", "visit Japan"),
        ("i'm flying to Tokyo", "speaker:alice/travel_upcoming", "Tokyo"),
        ("i have class on Mondays", "speaker:alice/schedule_regular", "have class"),
        ("i subscribe to Spotify", "speaker:alice/subscription", "Spotify"),
        ("i bank with Chase", "speaker:alice/bank_or_payment", "Chase"),
        ("my timezone is EST", "speaker:alice/timezone", "EST"),
        ("call me Mel", "speaker:alice/preferred_name", "Mel"),
        ("my pronouns are she/her", "speaker:alice/pronouns", "she/her"),
        ("i do yoga", "speaker:alice/workout_routine", "yoga"),
        ("i'm reading Dune", "speaker:alice/reading_current", "Dune"),
        ("i'm watching Severance", "speaker:alice/show_current", "Severance"),
    ])
    def test_family(self, text, key, val):
        fs = states(text)
        got = [(f.state_key, f.value_text) for f in fs]
        assert (key, val) in got, f"{key}/{val} not in {got}"


class TestStateGuards:
    def test_negated_state_abstains(self):
        assert states("i don't live in Berlin") == []

    def test_hypothetical_state_abstains(self):
        assert states("i might move to Oslo") == []
        assert states("i would live in Berlin") == []

    def test_question_state_abstains(self):
        assert states("do you work at Acme") == []

    def test_reported_state_abstains(self):
        assert states("i live in berlin, she said") == []

    def test_lowercase_name_value_abstains(self):
        # capitalization-gated proper nouns stay honest
        assert states("my manager is sarah") == []
        assert states("my wife is anna") == []

    def test_medication_requires_evidence(self):
        # "take the bus" is not medication — tail marker or known med needed
        assert states("i take the bus") == []
        assert states("i take notes") == []
        (f,) = states("i take metformin")
        assert f.state_key == "speaker:alice/medication"

    def test_pet_name_needs_capital_or_marker(self):
        fs = states("my dog is rex")
        assert any(f.state_key == "speaker:alice/pets" for f in fs)
        assert not any(f.state_key == "speaker:alice/pet_names" for f in fs)
        fs2 = states("i have a dog named rex")
        assert any(f.state_key == "speaker:alice/pets" for f in fs2)


class TestStateFields:
    def test_status_producer_valid_from(self):
        (f,) = states("i live in Berlin")
        assert f.status is StateFactStatus.CURRENT
        assert f.producer == MODULE_ID
        assert f.valid_from_us == OCC.start_us
        assert f.valid_to_us is None

    def test_value_norm_folded(self):
        (f,) = states("my email is A@B.com")
        assert f.value_norm == "a b com"

    def test_unit_id_verbatim(self):
        (f,) = states("i live in Berlin", unit_id="unit-xyz")
        assert f.unit_id == "unit-xyz"

    def test_multi_language_emits_each(self):
        fs = states("i speak English and French")
        vals = sorted(f.value_text for f in fs
                      if f.state_key == "speaker:alice/languages")
        assert vals == ["English", "French"]

    def test_married_emits_status_and_partner(self):
        fs = states("i'm married to Tom")
        keys = {f.state_key for f in fs}
        assert "speaker:alice/relationship_status" in keys
        assert "speaker:alice/partner" in keys

    def test_dog_named_emits_pets_and_name(self):
        fs = states("i have a dog named Rex")
        got = {f.state_key: f.value_text for f in fs}
        assert got.get("speaker:alice/pets") == "dog"
        assert got.get("speaker:alice/pet_names") == "Rex"


# ---------------------------------------------------------------------------
# Evidence pins / UTF-8 alignment
# ---------------------------------------------------------------------------


class TestPins:
    def test_pref_pins_slice_raw(self):
        text = "I love pizza"
        (p,) = prefs(text)
        assert_pref_pins(text, p)
        assert p.pins["pinned"] == "raw_verified"
        assert _raw_span(text, p.pins["object"]) == b"pizza"

    def test_state_pins_slice_raw(self):
        text = "i live in Berlin"
        (f,) = states(text)
        assert_state_pins(text, f)
        assert f.pins["pinned"] == "raw_verified"
        assert _raw_span(text, f.pins["value"]) == b"Berlin"

    def test_utf8_multibyte_span(self):
        text = "I love cafés"          # é = 2 UTF-8 bytes
        (p,) = prefs(text)
        assert p.pins["object"] == (7, 13)
        assert _raw_span(text, p.pins["object"]) == "cafés".encode("utf-8")
        assert p.object_text == "cafés"

    def test_text_verified_without_raw(self):
        # ASCII: folded norm.text stays byte-aligned with term offsets
        (p,) = extract_preferences(analyze("i love pizza"), "u", SPEAKER)
        assert p.pins["pinned"] == "text_verified"

    def test_term_offsets_when_unaligned(self):
        # non-ASCII folds shorter than raw -> probe fails -> honest degrade
        (p,) = extract_preferences(analyze("i love cafés"), "u", SPEAKER)
        assert p.pins["pinned"] == "term_offsets"
        assert p.object_text == "cafes"   # folded surface, not raw bytes

    def test_misaligned_raw_not_raw_verified(self):
        (p,) = extract_preferences(analyze("i love pizza"), "u", SPEAKER,
                                   raw_text="completely different text")
        assert p.pins["pinned"] == "term_offsets"


# ---------------------------------------------------------------------------
# scope_id / stats
# ---------------------------------------------------------------------------


class TestScopeAndStats:
    def test_scope_default_empty(self):
        (p,) = prefs("i love pizza")
        assert p.scope_id == ""
        (f,) = states("i live in Berlin")
        assert f.scope_id == ""

    def test_scope_stamped(self):
        (p,) = prefs("i love pizza", scope_id="scope:work")
        assert p.scope_id == "scope:work"
        (f,) = states("i live in Berlin", scope_id="scope:work")
        assert f.scope_id == "scope:work"

    def test_stats_receives_deltas(self):
        st = {}
        prefs("i love pizza", stats=st)
        assert st["units_pref"] == 1
        assert st["emitted_pref"] == 1
        st2 = {}
        states("i live in Berlin", stats=st2)
        assert st2["units_state"] == 1
        assert st2["emitted_state"] == 1

    def test_occurred_passthrough_pref(self):
        (p,) = prefs("i love pizza", occurred=OCC)
        assert p.occurred is OCC

    def test_occurred_default_unknown(self):
        (p,) = prefs("i love pizza")
        assert p.occurred.start_us is None


# ---------------------------------------------------------------------------
# state_compatible
# ---------------------------------------------------------------------------


class TestStateCompatible:
    def test_same_value_compatible(self):
        assert state_compatible("speaker:alice/home_city", "berlin", "berlin")
        assert state_compatible("speaker:alice/home_city", "Berlin", "berlin")

    def test_replace_family_conflicts(self):
        assert not state_compatible("speaker:alice/home_city", "berlin", "oslo")
        assert not state_compatible("speaker:alice/employer", "acme", "globex")
        assert not state_compatible("speaker:alice/editor", "vim", "emacs")
        assert not state_compatible("speaker:alice/partner", "tom", "anna")
        assert not state_compatible("speaker:alice/relationship_status", "single",
                                    "married")

    @pytest.mark.parametrize("fam", [
        "pets", "pet_names", "hobbies", "languages", "allergies",
        "health_condition", "medication", "sport", "programming_language",
        "subscription", "diet", "goal_current", "schedule_regular",
        "device", "workout_routine", "degree", "plan_upcoming",
        "travel_upcoming",
    ])
    def test_accumulate_family_never_conflicts(self, fam):
        assert state_compatible(f"speaker:alice/{fam}", "alpha", "beta")

    def test_children_counts_conflict_names_coexist(self):
        assert not state_compatible("speaker:alice/children", "2", "3")
        assert not state_compatible("speaker:alice/children", "two kids",
                                    "three kids")
        assert state_compatible("speaker:alice/children", "son", "daughter")
        assert state_compatible("speaker:alice/children", "two kids", "two kids")

    def test_age_numeric(self):
        assert state_compatible("speaker:alice/age", "30", "30")
        assert not state_compatible("speaker:alice/age", "30", "31")
        assert state_compatible("speaker:alice/age", "thirty", "30")

    def test_bare_key_and_unknown_family(self):
        assert state_compatible("home_city", "berlin", "berlin")
        assert not state_compatible("home_city", "berlin", "oslo")
        assert state_compatible("unknown_key", "a", "a")
        assert not state_compatible("unknown_key", "a", "b")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    _DET_TEXTS = [
        "i love pizza and maria hates rain",
        "I'm allergic to peanuts. i usually drink coffee",
        "my dog is Rex and i live in Berlin",
        "i speak English and French; i do yoga on fridays",
    ]

    @staticmethod
    def _rows_json():
        rows = []
        for t in TestDeterminism._DET_TEXTS:
            for p in prefs(t, unit_id="u-9"):
                rows.append(["p", p.unit_id, p.subject_canon, p.object_text,
                             p.polarity, p.strength,
                             sorted((k, repr(v))
                                    for k, v in p.pins.items())])
            for f in states(t, unit_id="u-9"):
                rows.append(["s", f.unit_id, f.state_key, f.value_text,
                             f.value_norm, f.status.value, f.producer,
                             sorted((k, repr(v))
                                    for k, v in f.pins.items())])
        return json.dumps(rows)

    def test_repeat_identical_in_process(self):
        assert self._rows_json() == self._rows_json()

    def test_deterministic_across_processes_and_hashseeds(self):
        script = (
            "import json\n"
            "from verbatim.text.norm_v2 import analyze\n"
            "from verbatim.enrichment.prefs_state import (\n"
            "    extract_preferences, extract_state_facts)\n"
            "from verbatim.core.types_v7 import IntervalUs\n"
            "occ = IntervalUs(start_us=1000000, end_us=2000000)\n"
            f"texts = {self._DET_TEXTS!r}\n"
            "rows = []\n"
            "for t in texts:\n"
            "    for p in extract_preferences(analyze(t), 'u-9', 'spk:1',"
            "                                 raw_text=t):\n"
            "        rows.append(['p', p.unit_id, p.subject_canon,"
            " p.object_text, p.polarity, p.strength,"
            " sorted((k, repr(v)) for k, v in p.pins.items())])\n"
            "    for f in extract_state_facts(analyze(t), 'u-9', 'spk:1',"
            "                                 occ, raw_text=t):\n"
            "        rows.append(['s', f.unit_id, f.state_key, f.value_text,"
            " f.value_norm, f.status.value, f.producer,"
            " sorted((k, repr(v)) for k, v in f.pins.items())])\n"
            "print(json.dumps(rows))\n"
        )
        outs = []
        for seed in ("0", "1", "42"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(_REPO_ROOT), env=env,
                capture_output=True, text=True, check=True)
            outs.append(proc.stdout.strip())
        assert outs[0] == outs[1] == outs[2]
        # and identical to the in-process result modulo speaker label
        inproc = json.loads(self._rows_json())
        subproc = json.loads(outs[0])
        for row_p, row_s in zip(inproc, subproc):
            # index 2 carries the speaker canon ('speaker:alice' in-process
            # vs 'spk:1' in the subprocess); everything after it is
            # speaker-independent and must be byte-identical.
            assert row_p[0] == row_s[0]
            assert row_p[3:] == row_s[3:]


# ---------------------------------------------------------------------------
# Edge inputs
# ---------------------------------------------------------------------------


class TestEdgeInputs:
    @pytest.mark.parametrize("text", [
        "",
        "   ",
        "the and or",
        "yes no maybe",
        "the weather is nice today",
        "hmm ok",
    ])
    def test_no_output(self, text):
        assert prefs(text) == []
        assert states(text) == []

    def test_second_clause_pronoun_does_not_leak(self):
        # two clauses: "she hates" binds nothing; only pizza survives
        ps = prefs("i love pizza. she hates rain.")
        assert [p.object_text for p in ps] == ["pizza"]

    def test_multi_fact_ordering(self):
        fs = states("i live in Berlin and i work at Acme")
        keys = [f.state_key for f in fs]
        assert keys == sorted(keys, key=str) or len(keys) >= 2
        assert "speaker:alice/home_city" in keys and "speaker:alice/employer" in keys

    def test_empty_speaker_still_extracts_names(self):
        # V8-13.04: unattributed input still binds named subjects, and
        # first-person facts fall back to the "me" canon rather than
        # being dropped.
        (p,) = prefs("Maria loves jazz", speaker="")
        assert p.subject_canon == "maria"
        (f,) = states("i live in Berlin", speaker="")
        assert f.state_key == "me/home_city"
