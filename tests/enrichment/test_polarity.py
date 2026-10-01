"""polarity — deterministic marker polarity (§7, §30.1).

Precedence: quoted > hypothetical > negated > hedged > affirmative.
The marker semantics for hedged are ported from
``verbatim/evidence/supersession.py::_hedged``.
"""

from __future__ import annotations

from verbatim.enrichment import polarity
from verbatim.memory.types import Polarity


class TestAffirmative:
    def test_plain(self):
        assert polarity("the deploy worked") is Polarity.AFFIRMATIVE

    def test_ruff_check_not_hedged(self):
        # bare modal verb, no complementizer — a command, not a hedge
        assert polarity("ruff check passed") is Polarity.AFFIRMATIVE

    def test_empty(self):
        assert polarity("") is Polarity.AFFIRMATIVE
        assert polarity(None) is Polarity.AFFIRMATIVE


class TestNegated:
    def test_no_longer(self):
        assert polarity("I no longer like X") is Polarity.NEGATED

    def test_not_never(self):
        assert polarity("we do not use tabs") is Polarity.NEGATED
        assert polarity("we never deploy Friday") is Polarity.NEGATED

    def test_contractions(self):
        assert polarity("it doesn't work") is Polarity.NEGATED
        assert polarity("we didn't ship") is Polarity.NEGATED
        assert polarity("it isn’t ready") is Polarity.NEGATED  # curly ’

    def test_cannot_without_none(self):
        assert polarity("we cannot migrate yet") is Polarity.NEGATED
        assert polarity("it works without keys") is Polarity.NEGATED
        assert polarity("none of them failed") is Polarity.NEGATED

    def test_adversarial_polarity_differs(self):
        # V5-30.07 — the canonical pair must differ
        assert polarity("I like X") is not polarity("I no longer like X")
        assert polarity("deploy worked") is not polarity(
            "deploy did not work")


class TestHedged:
    def test_adverbs(self):
        assert polarity("maybe it works") is Polarity.HEDGED
        assert polarity("it reportedly fails") is Polarity.HEDGED
        assert polarity("it probably works") is Polarity.HEDGED
        assert polarity("allegedly broken") is Polarity.HEDGED

    def test_whether_always_hedges(self):
        assert polarity("check whether it works") is Polarity.HEDGED
        assert polarity("whether or not it ran") is Polarity.HEDGED

    def test_modal_plus_complementizer(self):
        assert polarity("confirm that it works") is Polarity.HEDGED
        assert polarity("verify that it holds") is Polarity.HEDGED

    def test_verify_if_is_hypothetical(self):
        # "if" is a conditional frame — hypothetical outranks hedged
        assert polarity("verify if it holds") is Polarity.HYPOTHETICAL

    def test_epistemic_first_person(self):
        assert polarity("i think it works") is Polarity.HEDGED
        assert polarity("it seems fine") is Polarity.HEDGED

    def test_hearsay(self):
        # unverifiable report without a quoting verb → hedged; "they
        # say" IS a reported-speech verb → quoted (see TestQuoted)
        assert polarity("i heard it shipped") is Polarity.HEDGED
        assert polarity("word is it broke") is Polarity.HEDGED

    def test_hedge_frame_wraps_negation(self):
        # "whether or not" — the "not" is inside the question, not a
        # negated assertion
        assert polarity("whether or not it ran") is Polarity.HEDGED
        assert polarity("i heard it never worked") is Polarity.HEDGED

    def test_negated_cognition_is_hedged(self):
        # "don't think" hedges belief — not a negated fact
        assert polarity("I don't think it's ready") is Polarity.HEDGED
        assert polarity("not sure it works") is Polarity.HEDGED


class TestHypothetical:
    def test_conditionals(self):
        assert polarity("if we deploy, it breaks") is \
            Polarity.HYPOTHETICAL
        assert polarity("unless it fails, we ship") is \
            Polarity.HYPOTHETICAL
        assert polarity("suppose it broke") is Polarity.HYPOTHETICAL
        assert polarity("imagine we migrated") is Polarity.HYPOTHETICAL

    def test_modals(self):
        assert polarity("we would migrate") is Polarity.HYPOTHETICAL
        assert polarity("it could be fine") is Polarity.HYPOTHETICAL
        assert polarity("it might work") is Polarity.HYPOTHETICAL
        assert polarity("we'd try it") is Polarity.HYPOTHETICAL

    def test_hypothetical_beats_negated(self):
        # documented precedence — a conditioned negation stays
        # hypothetical, not negated
        assert polarity("we would not do that") is Polarity.HYPOTHETICAL
        assert polarity("if it doesn't work") is Polarity.HYPOTHETICAL


class TestQuoted:
    def test_quotation_marks(self):
        assert polarity('she said "no way"') is Polarity.QUOTED
        assert polarity('the note read “deprecated”') is Polarity.QUOTED
        assert polarity("doc says 'run it twice'") is Polarity.QUOTED

    def test_reported_speech(self):
        assert polarity("she said it works") is Polarity.QUOTED
        assert polarity("according to the docs it broke") is \
            Polarity.QUOTED
        assert polarity("he claimed it shipped") is Polarity.QUOTED

    def test_quoted_beats_inner_negation(self):
        # quoted first — inner polarity belongs to the quoted speaker
        assert polarity('"I no longer like X", she wrote') is \
            Polarity.QUOTED


class TestPrecedenceAndDeterminism:
    def test_full_precedence_chain(self):
        # quoted wins over all; hypothetical over negated; negated over
        # hedged
        assert polarity('she said "if it breaks"') is Polarity.QUOTED
        assert polarity("if it never worked") is Polarity.HYPOTHETICAL
        assert polarity("maybe we never shipped") is Polarity.NEGATED

    def test_determinism(self):
        t = 'maybe she said "we could not ship" yesterday'
        assert polarity(t) == polarity(t)

    def test_returns_enum(self):
        assert isinstance(polarity("x"), Polarity)
