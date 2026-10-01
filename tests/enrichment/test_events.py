"""Tests for ``event/v1`` — deterministic event extraction
(SPEC_V7 §32.10, V7-09.08; sieve integration per V7-13.20).

The module under test is ``verbatim/enrichment/events.py``. All inputs go
through the real ``norm/v2`` analyzer (``verbatim.text.norm_v2.analyze``);
``NormAnalysis`` is never faked. Expected values below were derived from
the spec'd rules (family lexicon, subject resolution, polarity window,
gates) and verified against the landed implementation.

Covered:

- every §32.10 predicate family, lemma-level (``predicate_lemma`` is the
  family-canonical lemma; ``rule_id`` carries ``family/lexeme``);
- pin verification against ``raw_text`` UTF-8 bytes, including the
  dropped-and-counted path (``dropped_unpinned``);
- honest ``subject_canon=None`` abstention (``subject=unknown``) for
  unresolved pronouns/descriptions/missing subjects;
- speaker/addressee substitution, caller ``sieve`` for third person and
  definite descriptions, ``canon_fn`` for explicit names;
- coordinated subjects (one tuple per subject), subject reuse across
  coordinators and sentence boundaries, light-verb and infinitive frames;
- polarity ``affirm``/``negate``/``hypothetical`` (V5 vocabulary);
- object gates (personish, pets, event/class/body/illness/job nouns,
  numeric), ``reject_next`` guard, noun-position guard;
- ``occurred`` pass-through, ``unit_id`` verbatim, counters/stats,
  determinism (in-process and across ``PYTHONHASHSEED`` processes);
- empty/stopword-only inputs.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from verbatim.core.types_v7 import EventTuple, IntervalUs
from verbatim.enrichment.coref_sieve import resolve_antecedent
from verbatim.enrichment.events import (
    COUNTERS,
    EXTRACTOR_ID,
    FORMULA_STATUS,
    LEXEME_COUNT,
    LEXICON_FAMILIES,
    extract_events,
    reset_counters,
)
from verbatim.text.norm_v2 import analyze

SPEAKER = "speaker:alice"
ADDRESSEE = "speaker:bob"
OCC = IntervalUs(start_us=1_000_000, end_us=2_000_000)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clean_counters():
    reset_counters()
    yield
    reset_counters()


def extract(text, unit_id="u-1", speaker=SPEAKER, occurred=OCC, **kw):
    """Analyze ``text`` with the real ``norm/v2`` analyzer and extract.
    ``raw_text`` defaults to the analyzed text so pins are verified."""
    kw.setdefault("raw_text", text)
    return extract_events(analyze(text), unit_id, speaker, occurred, **kw)


def _raw_span(text, span):
    return text.encode("utf-8")[span[0]:span[1]]


def assert_pins_verify(text, ev):
    """Every non-None pin span must slice decodable, non-empty UTF-8
    bytes out of ``raw_text`` and lie inside ``pins["span"]``."""
    raw = text.encode("utf-8")
    pins = ev.pins
    assert pins["pinned"] == "raw_verified"
    seen = []
    for key in ("subject", "predicate", "object", "negation"):
        span = pins.get(key)
        if span is None:
            continue
        s, e = span
        assert 0 <= s < e <= len(raw), (key, span)
        frag = raw[s:e].decode("utf-8")
        assert frag, (key, span)
        seen.append(span)
    assert seen, "a tuple must pin at least one element"
    span_s, span_e = pins["span"]
    assert 0 <= span_s < span_e <= len(raw)
    for s, e in seen:
        assert span_s <= s and e <= span_e


# ---------------------------------------------------------------------------
# Contract / metadata
# ---------------------------------------------------------------------------


class TestContract:
    def test_extractor_metadata(self):
        assert EXTRACTOR_ID == "event/v1"
        assert FORMULA_STATUS == "provisional/v7-r0"  # V7-32.01
        assert LEXEME_COUNT > 0

    def test_lexicon_covers_spec_families(self):
        # §32.10's declared families, "extended conservatively".
        expected = {
            "move", "start", "quit", "retire", "graduate", "marry",
            "buy", "visit", "attend", "adopt", "paint", "renovate",
            "run", "meet", "host", "celebrate", "cook", "learn",
            "volunteer", "sick", "born", "die", "hire", "launch",
            "book", "call", "read", "plant",
        }
        assert expected <= set(LEXICON_FAMILIES)

    def test_returns_list_of_event_tuples(self):
        evs = extract("i moved to berlin")
        assert isinstance(evs, list)
        assert all(isinstance(e, EventTuple) for e in evs)

    def test_tuple_fields(self):
        (ev,) = extract("i moved to berlin")
        assert ev.unit_id == "u-1"
        assert ev.subject_canon == SPEAKER
        assert ev.predicate_lemma == "move"
        assert ev.object_text == "berlin"
        assert ev.polarity == "affirm"
        assert ev.rule_id == "move/move"
        for key in ("subject", "predicate", "object", "negation",
                    "span", "lexeme", "subject_form", "pinned"):
            assert key in ev.pins

    def test_rule_id_is_family_slash_lexeme(self):
        (ev,) = extract("alice left the company")
        assert ev.rule_id == "quit/leave"
        assert ev.pins["lexeme"] == "leave"
        assert ev.predicate_lemma == "quit"

    def test_unit_id_carried_verbatim(self):
        evs = extract("i moved. went to berlin.", unit_id="unit-XYZ-99")
        assert len(evs) == 2
        assert all(e.unit_id == "unit-XYZ-99" for e in evs)

    def test_occurred_passed_through_verbatim(self):
        occ = IntervalUs(start_us=42, end_us=77)
        evs = extract("i moved to berlin", occurred=occ)
        assert evs[0].occurred is occ
        other = IntervalUs(start_us=5, end_us=6, precision="day",
                           source="stated", rule_id="T00")
        evs = extract("i moved to berlin", occurred=other)
        assert evs[0].occurred == other


# ---------------------------------------------------------------------------
# §32.10 family coverage — (text, family, rule_id, object, subject_canon)
# ---------------------------------------------------------------------------

FAMILY_CASES = [
    # move/relocate --------------------------------------------------
    ("i moved to berlin", "move", "move/move", "berlin", SPEAKER),
    ("alice relocated to oslo", "move", "move/relocate", "oslo", "alice"),
    ("we migrated to canada", "move", "move/migrate", "canada", SPEAKER),
    ("we moved in together", "move", "move/move in", "", SPEAKER),
    ("we moved out", "move", "move/move out", "", SPEAKER),
    # start/begin/join/enroll/sign up ---------------------------------
    ("i started a new job", "start", "start/start", "new job", SPEAKER),
    ("she joined the gym", "start", "start/join", "gym", None),
    ("he enrolled in college", "start", "start/enroll", "college", None),
    ("i signed up for the marathon", "start", "start/sign up",
     "marathon", SPEAKER),
    ("alice began work", "start", "start/begin", "work", "alice"),
    ("the semester commenced", "start", "start/commence", "", None),
    # quit/leave/resign/drop out --------------------------------------
    ("i quit my job", "quit", "quit/quit", "my job", SPEAKER),
    ("alice left the company", "quit", "quit/leave", "company", "alice"),
    ("bob resigned", "quit", "quit/resign", "", "bob"),
    ("i dropped out of school", "quit", "quit/drop out", "school",
     SPEAKER),
    ("she stepped down", "quit", "quit/step down", "", None),
    ("they walked out", "quit", "quit/walk out", "", None),
    # retire ------------------------------------------------------------
    ("i retired", "retire", "retire/retire", "", SPEAKER),
    # graduate/finish/complete ------------------------------------------
    ("i graduated", "graduate", "graduate/graduate", "", SPEAKER),
    ("alice finished college", "graduate", "graduate/finish", "college",
     "alice"),
    ("i completed my degree", "graduate", "graduate/complete",
     "my degree", SPEAKER),
    ("we wrapped up the project", "graduate", "graduate/wrap up",
     "project", SPEAKER),
    # marry/engage/divorce/date/break up ---------------------------------
    ("i married alice", "marry", "marry/marry", "alice", SPEAKER),
    ("they got engaged", "marry", "marry/engage", "", None),
    ("i dated alice", "marry", "marry/date", "alice", SPEAKER),
    ("they divorced", "marry", "marry/divorce", "", None),
    ("we broke up", "marry", "marry/break up", "", SPEAKER),
    ("bob proposed to alice", "marry", "marry/propose", "alice", "bob"),
    # buy/purchase/order/sell ---------------------------------------------
    ("i bought a house", "buy", "buy/buy", "house", SPEAKER),
    ("alice purchased a car", "buy", "buy/purchase", "car", "alice"),
    ("i ordered pizza", "buy", "buy/order", "pizza", SPEAKER),
    ("bob sold his bike", "buy", "buy/sell", "his bike", "bob"),
    # visit/travel/go to/fly to/return from ---------------------------------
    ("i visited paris", "visit", "visit/visit", "paris", SPEAKER),
    ("we traveled to japan", "visit", "visit/travel", "japan", SPEAKER),
    ("i flew to london", "visit", "visit/fly to", "london", SPEAKER),
    ("she drove to work", "visit", "visit/drive to", "work", None),
    ("i went to berlin", "visit", "visit/go to", "berlin", SPEAKER),
    ("i returned from paris", "visit", "visit/return from", "paris",
     SPEAKER),
    ("we arrived in rome", "visit", "visit/arrive in", "rome", SPEAKER),
    ("i came back", "visit", "visit/come back", "", SPEAKER),
    # attend / go-to(event) ---------------------------------------------------
    ("i went to a concert", "attend", "attend/go to", "concert", SPEAKER),
    ("i attended the meeting", "attend", "attend/attend", "", SPEAKER),
    ("she showed up", "attend", "attend/show up", "", None),
    # adopt/get (a pet) ---------------------------------------------------------
    ("we adopted a puppy", "adopt", "adopt/adopt", "puppy", SPEAKER),
    ("i got a dog", "adopt", "adopt/get", "dog", SPEAKER),
    ("they fostered a kitten", "adopt", "adopt/foster", "kitten", None),
    ("i rescued a cat", "adopt", "adopt/rescue", "cat", SPEAKER),
    # paint/draw/write/publish/record -------------------------------------------
    ("i painted a portrait", "paint", "paint/paint", "portrait", SPEAKER),
    ("she wrote a novel", "paint", "paint/write", "novel", None),
    ("he published a paper", "paint", "paint/publish", "paper", None),
    ("i recorded a song", "paint", "paint/record", "song", SPEAKER),
    # renovate/build/fix (paint is gated into this family by home nouns) --------
    ("i painted the kitchen", "renovate", "renovate/paint", "kitchen",
     SPEAKER),
    ("we renovated the bathroom", "renovate", "renovate/renovate",
     "bathroom", SPEAKER),
    ("bob built a deck", "renovate", "renovate/build", "deck", "bob"),
    ("i fixed the fence", "renovate", "renovate/fix", "fence", SPEAKER),
    ("she repaired the roof", "renovate", "renovate/repair", "roof", None),
    # run/race/compete/win/lose ---------------------------------------------------
    ("i ran a marathon", "run", "run/run", "marathon", SPEAKER),
    ("alice won the race", "run", "run/win", "", "alice"),
    ("we lost the game", "run", "run/lose", "game", SPEAKER),
    ("she competed", "run", "run/compete", "", None),
    ("i participated in the triathlon", "run", "run/participate in",
     "triathlon", SPEAKER),
    # meet/see/reunite --------------------------------------------------------------
    ("i met alice", "meet", "meet/meet", "alice", SPEAKER),
    ("i saw alice", "meet", "meet/see", "alice", SPEAKER),
    ("we met up", "meet", "meet/meet up", "", SPEAKER),
    ("i bumped into bob", "meet", "meet/bump into", "bob", SPEAKER),
    ("they reunited", "meet", "meet/reunite", "", None),
    ("we caught up", "meet", "meet/catch up", "", SPEAKER),
    # host/throw (a party) ------------------------------------------------------------
    ("i hosted a dinner", "host", "host/host", "dinner", SPEAKER),
    ("she threw a party", "host", "host/throw", "party", None),
    ("we organized a meetup", "host", "host/organize", "meetup", SPEAKER),
    # celebrate/have (a birthday) -------------------------------------------------------
    ("i celebrated my birthday", "celebrate", "celebrate/celebrate",
     "my birthday", SPEAKER),
    ("we had a birthday", "celebrate", "celebrate/have", "birthday",
     SPEAKER),
    ("i turned 30", "celebrate", "celebrate/turn", "30", SPEAKER),
    # cook/bake --------------------------------------------------------------------------
    ("i cooked dinner", "cook", "cook/cook", "dinner", SPEAKER),
    ("alice baked a cake", "cook", "cook/bake", "cake", "alice"),
    ("i grilled steaks", "cook", "cook/grill", "steaks", SPEAKER),
    ("she brewed beer", "cook", "cook/brew", "beer", None),
    # learn/take (a class) ------------------------------------------------------------------
    ("i learned python", "learn", "learn/learn", "python", SPEAKER),
    ("she studied french", "learn", "learn/study", "french", None),
    ("i took a class", "learn", "learn/take", "class", SPEAKER),
    ("we practiced yoga", "learn", "learn/practice", "yoga", SPEAKER),
    ("he practised scales", "learn", "learn/practise", "scales", None),
    # volunteer/donate/mentor --------------------------------------------------------------
    ("i volunteered at the shelter", "volunteer", "volunteer/volunteer",
     "shelter", SPEAKER),
    ("she donated blood", "volunteer", "volunteer/donate", "blood", None),
    ("he mentors students", "volunteer", "volunteer/mentor", "students",
     None),
    ("i tutored kids", "volunteer", "volunteer/tutor", "kids", SPEAKER),
    ("we coached the team", "volunteer", "volunteer/coach", "team",
     SPEAKER),
    # get sick/recover/injure/diagnose ------------------------------------------------------
    ("i got sick", "sick", "sick/sick", "", SPEAKER),
    ("alice recovered", "sick", "sick/recover", "", "alice"),
    ("i broke my arm", "sick", "sick/break", "my arm", SPEAKER),
    ("i caught a cold", "sick", "sick/catch", "cold", SPEAKER),
    ("i injured my knee", "sick", "sick/injure", "my knee", SPEAKER),
    ("he felt unwell", "sick", "sick/unwell", "", None),
    ("she looks ill", "sick", "sick/ill", "", None),
    # born / die -----------------------------------------------------------------------------
    ("i was born in oslo", "born", "born/born", "oslo", SPEAKER),
    ("alice was born", "born", "born/born", "", "alice"),
    ("my grandfather died", "die", "die/die", "", None),
    ("alice passed away", "die", "die/pass away", "", "alice"),
    # hire/promote/fire/interview -------------------------------------------------------------
    ("i hired a lawyer", "hire", "hire/hire", "lawyer", SPEAKER),
    ("alice was promoted", "hire", "hire/promote", "", "alice"),
    ("i fired him", "hire", "hire/fire", "him", SPEAKER),
    ("acme laid him off", "hire", "hire/lay off", "him", "acme"),
    ("i interviewed bob", "hire", "hire/interview", "bob", SPEAKER),
    ("acme offered a position", "hire", "hire/offer", "position", "acme"),
    ("she recruited me", "hire", "hire/recruit", "me", None),
    # launch/ship/release -----------------------------------------------------------------------
    ("she launched a startup", "launch", "launch/launch", "startup", None),
    ("we shipped the feature", "launch", "launch/ship", "feature",
     SPEAKER),
    ("they released the album", "launch", "launch/release", "album", None),
    ("i deployed the update", "launch", "launch/deploy", "update",
     SPEAKER),
    ("the film premiered", "launch", "launch/premiere", "", None),
    # book/reserve/cancel -------------------------------------------------------------------------
    ("i booked a flight", "book", "book/book", "flight", SPEAKER),
    ("alice reserved a table", "book", "book/reserve", "table", "alice"),
    ("we canceled the trip", "book", "book/cancel", "trip", SPEAKER),
    ("i confirmed the reservation", "book", "book/confirm", "reservation",
     SPEAKER),
    ("i rescheduled the meeting", "book", "book/reschedule", "", SPEAKER),
    # call/text/email ------------------------------------------------------------------------------
    ("i called my mom", "call", "call/call", "my mom", SPEAKER),
    ("she texted me", "call", "call/text", "me", None),
    ("he emailed the team", "call", "call/email", "team", None),
    ("i phoned bob", "call", "call/phone", "bob", SPEAKER),
    ("i messaged alice", "call", "call/message", "alice", SPEAKER),
    ("i dmed alice", "call", "call/dm", "alice", SPEAKER),
    # read/watch/listen to ----------------------------------------------------------------------------
    ("i read the novel", "read", "read/read", "novel", SPEAKER),
    ("i watched the movie", "read", "read/watch", "movie", SPEAKER),
    ("i listened to the podcast", "read", "read/listen to", "podcast",
     SPEAKER),
    ("i streamed the series", "read", "read/stream", "series", SPEAKER),
    ("we binged the season", "read", "read/binge", "season", SPEAKER),
    # plant/grow/garden ---------------------------------------------------------------------------------
    ("i planted tomatoes", "plant", "plant/plant", "tomatoes", SPEAKER),
    ("we grew herbs", "plant", "plant/grow", "herbs", SPEAKER),
    ("i harvested apples", "plant", "plant/harvest", "apples", SPEAKER),
    ("she pruned the roses", "plant", "plant/prune", "roses", None),
    ("we gardened", "plant", "plant/garden", "", SPEAKER),
    ("i sowed seeds", "plant", "plant/sow", "seeds", SPEAKER),
]


class TestPredicateFamilies:
    """§32.10 happy path: one tuple per case; ``predicate_lemma`` is the
    family-canonical lemma, ``rule_id`` records ``family/lexeme``."""

    @pytest.mark.parametrize(
        "text,family,rule_id,obj,subj", FAMILY_CASES,
        ids=[c[0] for c in FAMILY_CASES])
    def test_family_case(self, text, family, rule_id, obj, subj):
        evs = extract(text)
        assert len(evs) == 1
        ev = evs[0]
        assert ev.predicate_lemma == family
        assert ev.rule_id == rule_id
        assert ev.object_text == obj
        assert ev.subject_canon == subj
        assert ev.polarity == "affirm"
        assert ev.occurred == OCC
        assert_pins_verify(text, ev)


# ---------------------------------------------------------------------------
# Pin verification (V7-09.08: tuples failing pin verification are dropped)
# ---------------------------------------------------------------------------


class TestPins:
    PIN_CORPUS = [
        "I moved to Berlin",
        "Alice bought a house",
        "She hired a new manager",
        "i moved to berlin and started a new job",
        "they laid him off",
        "bob signed alice up",
        "i did not move to berlin",
        "the manager resigned",
        "alice and bob moved to oslo",
        "we got married in june",
        "i was born in oslo",
        "my grandmother passed away",
        "i broke my arm",
        "i turned 30",
        "i want to visit paris",
    ]

    def test_pins_byte_verify_across_corpus(self):
        for text in self.PIN_CORPUS:
            for ev in extract(text):
                assert_pins_verify(text, ev)

    def test_subject_predicate_object_exact_bytes(self):
        text = "I moved to Berlin"
        (ev,) = extract(text)
        assert _raw_span(text, ev.pins["subject"]) == b"I"
        assert _raw_span(text, ev.pins["predicate"]) == b"moved"
        assert _raw_span(text, ev.pins["object"]) == b"Berlin"

    def test_object_text_is_raw_case_preserved(self):
        # object_text comes from the raw slice, not the folded term.
        (ev,) = extract("i visited Paris")
        assert ev.object_text == "Paris"
        (ev,) = extract("i visited Paris", raw_text=None)
        assert ev.object_text == "paris"

    def test_predicate_span_covers_separable_particle_object(self):
        text = "they laid him off"
        (ev,) = extract(text)
        assert _raw_span(text, ev.pins["predicate"]) == b"laid him off"
        assert _raw_span(text, ev.pins["object"]) == b"him"

    def test_predicate_span_covers_aux(self):
        text = "i was born in oslo"
        (ev,) = extract(text)
        assert _raw_span(text, ev.pins["predicate"]) == b"was born"

    def test_unicode_pin_byte_offsets(self):
        text = "Alice is moving to Zürich"
        (ev,) = extract(text)
        assert ev.object_text == "Zürich"
        assert _raw_span(text, ev.pins["object"]) == "Zürich".encode()

    def test_negation_pin_exact_bytes(self):
        text = "I did not move to Berlin"
        (ev,) = extract(text)
        assert ev.polarity == "negate"
        assert _raw_span(text, ev.pins["negation"]) == b"not"

    def test_dropped_unpinned_counted(self):
        # raw_text truncated so the object span (11,17) exceeds raw_len:
        # the tuple cannot verify and is dropped, not silently emitted.
        stats = {}
        evs = extract_events(
            analyze("I moved to Berlin"), "u-1", SPEAKER, OCC,
            raw_text="I moved to", stats=stats)
        assert evs == []
        assert stats["dropped_unpinned"] == 1
        assert stats["emitted"] == 0
        assert COUNTERS["dropped_unpinned"] == 1

    def test_dropped_when_raw_text_empty(self):
        stats = {}
        evs = extract_events(
            analyze("i moved to berlin"), "u-1", SPEAKER, OCC,
            raw_text="", stats=stats)
        assert evs == []
        assert stats["dropped_unpinned"] == 1

    def test_unverified_without_raw_text(self):
        # No raw_text: pins are term offsets, marked honestly, still emitted.
        stats = {}
        evs = extract_events(
            analyze("i moved to berlin"), "u-1", SPEAKER, OCC,
            stats=stats)
        assert len(evs) == 1
        assert evs[0].pins["pinned"] == "term_offsets"
        assert stats["dropped_unpinned"] == 0
        assert stats["emitted"] == 1


# ---------------------------------------------------------------------------
# Subject resolution (§32.10, V7-13.20)
# ---------------------------------------------------------------------------


class TestSubjects:
    def test_first_person_i_is_speaker(self):
        (ev,) = extract("i moved to berlin")
        assert ev.subject_canon == SPEAKER
        assert ev.pins["subject_form"] == "first_person"

    def test_first_person_we_is_speaker(self):
        (ev,) = extract("we moved to berlin")
        assert ev.subject_canon == SPEAKER
        assert ev.pins["subject_form"] == "first_person"

    def test_first_person_without_speaker_canon_abstains(self):
        (ev,) = extract("i moved to berlin", speaker=None)
        assert ev.subject_canon is None
        assert ev.pins["subject_form"] == "first_person"

    def test_second_person_is_addressee(self):
        (ev,) = extract("you moved to berlin", addressee_canon=ADDRESSEE)
        assert ev.subject_canon == ADDRESSEE
        assert ev.pins["subject_form"] == "second_person"

    def test_second_person_without_addressee_abstains(self):
        (ev,) = extract("you moved to berlin")
        assert ev.subject_canon is None
        assert ev.pins["subject_form"] == "second_person"

    def test_third_person_pronoun_without_sieve_abstains(self):
        (ev,) = extract("she met him at the party")
        assert ev.subject_canon is None
        assert ev.pins["subject_form"] == "pronoun"
        assert_pins_verify("she met him at the party", ev)

    def test_definite_description_without_sieve_abstains(self):
        (ev,) = extract("the manager resigned")
        assert ev.subject_canon is None
        assert ev.pins["subject_form"] == "description"

    def test_possessive_description_abstains(self):
        (ev,) = extract("my grandfather died")
        assert ev.subject_canon is None
        assert ev.pins["subject_form"] == "description"

    def test_no_subject_form_none(self):
        (ev,) = extract("went to berlin")
        assert ev.subject_canon is None
        assert ev.pins["subject_form"] == "none"
        assert ev.pins["subject"] is None
        assert ev.predicate_lemma == "visit"

    def test_explicit_name_via_default_canon_fn(self):
        # entities_v2.canon: fold(NFKC(casefold(strip_possessive(x))))
        (ev,) = extract("Alice bought a house")
        assert ev.subject_canon == "alice"
        assert ev.pins["subject_form"] == "name"

    def test_canon_fn_injected(self):
        (ev,) = extract("Alice bought a house",
                        canon_fn=lambda s: "canon/" + s.casefold())
        assert ev.subject_canon == "canon/alice"

    def test_canon_fn_falsy_means_unresolved_not_fabricated(self):
        stats = {}
        (ev,) = extract_events(
            analyze("Alice bought a house"), "u-1", SPEAKER, OCC,
            raw_text="Alice bought a house",
            canon_fn=lambda s: "", stats=stats)
        assert ev.subject_canon is None
        # the miss is not the "canon unavailable" counter — the fn ran.
        assert stats["canon_unavailable"] == 0

    def test_canon_fn_unavailable(self, monkeypatch):
        # entities_v2 import failing → canon_fn=None → name subjects stay
        # None (no fabrication) and the miss is counted.
        monkeypatch.setitem(
            sys.modules, "verbatim.enrichment.entities_v2", None)
        stats = {}
        (ev,) = extract_events(
            analyze("Alice bought a house"), "u-1", SPEAKER, OCC,
            raw_text="Alice bought a house", stats=stats)
        assert ev.subject_canon is None
        assert ev.pins["subject_form"] == "name"
        assert stats["canon_unavailable"] == 1
        assert COUNTERS["canon_unavailable"] == 1

    def test_coordinated_subject_emits_one_tuple_each(self):
        text = "Alice and Bob moved to Berlin"
        evs = extract(text)
        assert len(evs) == 2
        assert {e.subject_canon for e in evs} == {"alice", "bob"}
        assert all(e.predicate_lemma == "move" for e in evs)
        # each subject pins its own mention span
        spans = {e.subject_canon: e.pins["subject"] for e in evs}
        assert _raw_span(text, spans["alice"]) == b"Alice"
        assert _raw_span(text, spans["bob"]) == b"Bob"

    def test_subject_reuse_across_coordinator(self):
        text = "i moved to berlin and bought a house"
        evs = extract(text)
        assert [e.predicate_lemma for e in evs] == ["move", "buy"]
        assert all(e.subject_canon == SPEAKER for e in evs)
        reused = evs[1]
        assert reused.pins["subject_form"] == "reused"
        # reused subjects pin the ORIGINAL mention span ("i")
        assert _raw_span(text, reused.pins["subject"]) == b"i"

    def test_subject_reuse_across_sentence_boundary(self):
        text = "i moved. went to berlin."
        evs = extract(text)
        assert [e.predicate_lemma for e in evs] == ["move", "visit"]
        assert evs[1].pins["subject_form"] == "reused"
        assert evs[1].subject_canon == SPEAKER

    def test_light_verb_supplies_subject(self):
        # "i got engaged": the failed "get" candidate still resolves "i".
        (ev,) = extract("i got engaged")
        assert ev.predicate_lemma == "marry"
        assert ev.rule_id == "marry/engage"
        assert ev.subject_canon == SPEAKER
        assert ev.pins["subject_form"] == "first_person"

    def test_infinitive_recovers_matrix_subject(self):
        (ev,) = extract("i want to visit paris")
        assert ev.predicate_lemma == "visit"
        assert ev.subject_canon == SPEAKER
        assert ev.polarity == "hypothetical"

    def test_sieve_resolves_pronoun(self):
        calls = []

        def sieve(surface):
            calls.append(surface)
            return {"she": "alice"}.get(surface.casefold())

        (ev,) = extract("She met him at the party", sieve=sieve)
        assert ev.subject_canon == "alice"
        assert ev.pins["subject_form"] == "pronoun"
        # the sieve is handed the RAW mention surface, not the folded term
        assert calls == ["She"]

    def test_sieve_resolves_description(self):
        sieve = lambda s: {"the manager": "bob"}.get(s.casefold())
        (ev,) = extract("the manager resigned", sieve=sieve)
        assert ev.subject_canon == "bob"
        assert ev.pins["subject_form"] == "description"

    def test_sieve_abstains_keeps_tuple_with_unknown(self):
        sieve = lambda s: None
        (ev,) = extract("she left the company", sieve=sieve)
        assert ev.subject_canon is None
        assert ev.predicate_lemma == "quit"

    def test_sieve_not_consulted_for_first_person(self):
        def boom(surface):  # pragma: no cover - must never be called
            raise AssertionError("sieve consulted for first person")

        (ev,) = extract("i moved to berlin", sieve=boom)
        assert ev.subject_canon == SPEAKER

    def test_real_coref_sieve_integration(self):
        # V7-13.20: the caller adapts resolve_antecedent by binding unit
        # index + session turns. Single candidate → resolves.
        turns = [
            {"speaker": "alice", "text": "Alice got a dog",
             "canon_mentions": ["alice"]},
            {"speaker": "carol", "text": "she moved to berlin",
             "canon_mentions": []},
        ]
        sieve = functools.partial(
            resolve_antecedent, unit_index=1, session_turns=turns)
        (ev,) = extract("she moved to berlin", speaker="carol",
                        sieve=sieve)
        assert ev.subject_canon == "alice"

    def test_real_coref_sieve_abstention_integration(self):
        # Two surviving candidates → the sieve abstains; the tuple is
        # kept with subject=unknown (§32.10 / V7-13.20(f)).
        turns = [
            {"speaker": "alice", "text": "Alice met Bob",
             "canon_mentions": ["alice", "bob"]},
            {"speaker": "carol", "text": "he left", "canon_mentions": []},
        ]
        sieve = functools.partial(
            resolve_antecedent, unit_index=1, session_turns=turns)
        (ev,) = extract("he left", speaker="carol", sieve=sieve)
        assert ev.subject_canon is None
        assert ev.pins["subject_form"] == "pronoun"

    def test_subj_person_gate_rejects_description(self):
        # "date" requires a person-form subject: "our date" is a
        # description, not a dating event.
        assert extract("our date was fun") == []
        (ev,) = extract("i dated alice")
        assert ev.predicate_lemma == "marry"


# ---------------------------------------------------------------------------
# Polarity (V5 vocabulary retained: affirm | negate | hypothetical)
# ---------------------------------------------------------------------------


class TestPolarity:
    def test_affirm(self):
        (ev,) = extract("i moved to berlin")
        assert ev.polarity == "affirm"
        assert ev.pins["negation"] is None

    def test_negate_not(self):
        (ev,) = extract("i did not move to berlin")
        assert ev.polarity == "negate"
        assert ev.pins["negation"] is not None

    def test_negate_nt_clitic(self):
        # norm/v2 splits "didn't" → "didn" + "not"
        (ev,) = extract("i didn't move to berlin")
        assert ev.polarity == "negate"
        s, e = ev.pins["negation"]
        assert "i didn't move to berlin".encode("utf-8")[s:e].decode()

    def test_negate_never(self):
        text = "i never left"
        (ev,) = extract(text)
        assert ev.polarity == "negate"
        assert _raw_span(text, ev.pins["negation"]) == b"never"

    def test_negate_am_not(self):
        (ev,) = extract("i am not leaving")
        assert ev.polarity == "negate"
        assert ev.predicate_lemma == "quit"

    def test_hypothetical_modal(self):
        (ev,) = extract("i will move to berlin")
        assert ev.polarity == "hypothetical"
        assert ev.pins["negation"] is None

    def test_hypothetical_modal_can(self):
        (ev,) = extract("i can move")
        assert ev.polarity == "hypothetical"

    def test_hypothetical_infinitive(self):
        (ev,) = extract("i want to move to berlin")
        assert ev.polarity == "hypothetical"

    def test_hypothetical_subordinator(self):
        (ev,) = extract("if i moved to berlin")
        assert ev.polarity == "hypothetical"

    def test_negated_modal_is_negate_not_hypothetical(self):
        (ev,) = extract("i will not move")
        assert ev.polarity == "negate"


# ---------------------------------------------------------------------------
# Lexeme gates and guards
# ---------------------------------------------------------------------------


class TestGates:
    def test_obj_personish_fires_on_name(self):
        (ev,) = extract("i saw alice")
        assert ev.rule_id == "meet/see"
        assert ev.object_text == "alice"

    def test_obj_personish_rejects_det_np(self):
        assert extract("i saw the movie") == []

    def test_pet_gate_fires(self):
        (ev,) = extract("i got a dog")
        assert ev.predicate_lemma == "adopt"

    def test_pet_gate_rejects_non_pet(self):
        assert extract("i got a raise") == []

    def test_event_noun_gate_picks_attend(self):
        (ev,) = extract("i went to a concert")
        assert ev.predicate_lemma == "attend"
        assert ev.rule_id == "attend/go to"

    def test_event_noun_gate_falls_back_to_visit(self):
        (ev,) = extract("i went to berlin")
        assert ev.predicate_lemma == "visit"
        assert ev.rule_id == "visit/go to"

    def test_obj_num_gate_fires(self):
        (ev,) = extract("i turned 30")
        assert ev.rule_id == "celebrate/turn"
        assert ev.object_text == "30"

    def test_obj_num_gate_rejects_non_number(self):
        assert extract("i turned the corner") == []

    def test_illness_gate_fires(self):
        (ev,) = extract("i caught a cold")
        assert ev.predicate_lemma == "sick"

    def test_illness_gate_rejects_non_illness(self):
        assert extract("i caught the ball") == []

    def test_body_gate_fires(self):
        (ev,) = extract("i broke my arm")
        assert ev.rule_id == "sick/break"

    def test_offer_job_gate(self):
        assert extract("acme offered a position") != []
        # first object term "me" is not a job noun → gated out
        assert extract("they offered me a job") == []

    def test_reject_next_guard(self):
        # "grew sick" is the sick event, not a gardening one.
        (ev,) = extract("i grew sick")
        assert ev.predicate_lemma == "sick"
        assert ev.rule_id == "sick/sick"

    def test_noun_position_guard(self):
        # a determiner directly before the head means the head is a noun
        assert extract("the book was great") == []
        assert extract("the move was sudden") == []

    def test_noun_object_that_is_also_a_head(self):
        # "record" is a predicate head; as a determiner-led noun it must
        # not emit an art event, and it blocks "broke"'s object so the
        # sick gate fails too → zero tuples.
        assert extract("i broke the record") == []


# ---------------------------------------------------------------------------
# Counters and per-call stats
# ---------------------------------------------------------------------------


class TestCounters:
    def test_reset_counters_zeroes(self):
        extract("i moved to berlin")
        reset_counters()
        assert all(v == 0 for v in COUNTERS.values())

    def test_units_counted(self):
        extract("i moved to berlin")
        extract("nothing eventful here")
        assert COUNTERS["units"] == 2

    def test_emitted_and_predicates_matched(self):
        extract("i moved to berlin and bought a house")
        assert COUNTERS["emitted"] == 2
        assert COUNTERS["predicates_matched"] == 2

    def test_stats_dict_receives_per_call_counters(self):
        stats = {}
        extract_events(analyze("i moved to berlin"), "u-1", SPEAKER, OCC,
                       raw_text="i moved to berlin", stats=stats)
        assert stats == {"predicates_matched": 1, "emitted": 1,
                         "dropped_unpinned": 0, "canon_unavailable": 0}

    def test_stats_on_empty_input(self):
        stats = {}
        extract_events(analyze(""), "u-1", SPEAKER, OCC, stats=stats)
        assert stats["emitted"] == 0
        assert stats["predicates_matched"] == 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_output(self):
        norm = analyze("alice and bob moved to berlin and bought a house")
        a = extract_events(norm, "u-1", SPEAKER, OCC,
                           raw_text="alice and bob moved to berlin "
                                    "and bought a house")
        b = extract_events(norm, "u-1", SPEAKER, OCC,
                           raw_text="alice and bob moved to berlin "
                                    "and bought a house")
        assert a == b
        assert a is not b

    def test_fresh_analysis_same_output(self):
        text = "she did not sign up for the marathon"
        a = extract(text)
        b = extract(text)
        assert a == b

    _DET_TEXTS = [
        "i moved to berlin and bought a house",
        "alice and bob moved to oslo",
        "she did not sign up for the marathon",
        "i want to visit paris",
        "the manager resigned",
        "we got married. i will never leave.",
    ]

    @staticmethod
    def _rows_json():
        rows = []
        for t in TestDeterminism._DET_TEXTS:
            for e in extract(t, unit_id="u-9", speaker="spk:1"):
                rows.append([e.unit_id, e.subject_canon,
                             e.predicate_lemma, e.object_text,
                             e.polarity, e.rule_id,
                             sorted((k, repr(v))
                                    for k, v in e.pins.items())])
        return json.dumps(rows)

    def test_deterministic_across_processes_and_hashseeds(self):
        script = (
            "import json\n"
            "from verbatim.text.norm_v2 import analyze\n"
            "from verbatim.enrichment.events import extract_events\n"
            "from verbatim.core.types_v7 import IntervalUs\n"
            "occ = IntervalUs(start_us=1000000, end_us=2000000)\n"
            f"texts = {self._DET_TEXTS!r}\n"
            "rows = []\n"
            "for t in texts:\n"
            "    for e in extract_events(analyze(t), 'u-9', 'spk:1', occ,"
            "                            raw_text=t):\n"
            "        rows.append([e.unit_id, e.subject_canon,"
            " e.predicate_lemma, e.object_text, e.polarity, e.rule_id,"
            " sorted((k, repr(v)) for k, v in e.pins.items())])\n"
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
        # and identical to the in-process result
        assert outs[0] == self._rows_json()


# ---------------------------------------------------------------------------
# Edge inputs and multi-predicate ordering
# ---------------------------------------------------------------------------


class TestEdgeInputs:
    @pytest.mark.parametrize("text", [
        "",
        "   ",
        "the",
        "and or the",
        "yes no maybe",
        "the weather is nice today",
        "hmm ok",
    ])
    def test_no_events(self, text):
        assert extract(text) == []

    def test_predicates_emit_in_text_order(self):
        evs = extract("i moved to berlin and bought a house "
                      "and adopted a dog")
        assert [e.predicate_lemma for e in evs] == ["move", "buy", "adopt"]

    def test_lexical_object_strips_leading_function_words(self):
        (ev,) = extract("i moved to berlin")
        assert ev.object_text == "berlin"
        (ev,) = extract("i arrived in rome")
        assert ev.object_text == "rome"

    def test_possessive_det_kept_in_object(self):
        (ev,) = extract("i visited my mother")
        assert ev.object_text == "my mother"

    def test_coordinated_object(self):
        (ev,) = extract("i bought bread and milk")
        assert ev.predicate_lemma == "buy"
        assert ev.object_text == "bread and milk"

    def test_temporal_adverb_terminates_object(self):
        (ev,) = extract("i visited paris last week")
        assert ev.object_text == "paris"
        (ev,) = extract("we got married in june")
        assert ev.object_text == ""
