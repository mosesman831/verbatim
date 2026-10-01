"""classify_type — deterministic marker rules → MemoryType (V5-30.25).

Precedence: preference > decision > plan > relationship >
procedure_hint > absence > state > event > untyped. ``fact`` is never
emitted — no reliable deterministic marker exists; ``untyped`` is the
honest fallback.
"""

from __future__ import annotations

from verbatim.enrichment import classify_type
from verbatim.enrichment.typing import TYPING_PRODUCER
from verbatim.memory.types import MemoryType


class TestMarkers:
    def test_preference(self):
        for t in ("I like tabs", "we prefer postgres",
                  "my favorite editor is vim", "I'd rather use YAML",
                  "I can't stand meetings"):
            assert classify_type(t) is MemoryType.PREFERENCE, t

    def test_preference_through_negation(self):
        # negation adverbs don't break the stance match — the type stays
        # preference so "I like X"/"I no longer like X" share type for
        # update-candidate matching (V5-30.18)
        assert classify_type("I no longer like X") is \
            MemoryType.PREFERENCE
        assert classify_type("I don't like YAML") is MemoryType.PREFERENCE
        assert classify_type("we never liked XML") is MemoryType.PREFERENCE

    def test_decision(self):
        for t in ("we decided to use Redis", "let's use Postgres",
                  "the team chose React", "we'll go with option B",
                  "they settled on kafka"):
            assert classify_type(t) is MemoryType.DECISION, t

    def test_plan(self):
        for t in ("plan to deploy Friday", "we will migrate in Q3",
                  "going to rewrite it", "scheduled for next week",
                  "todo: fix the flaky test", "the next step is cleanup"):
            assert classify_type(t) is MemoryType.PLAN, t

    def test_relationship(self):
        for t in ("Alice reports to Bob", "Carol works with Dave",
                  "Eve is my mentor", "Frank manages the infra team",
                  "she's married to a pilot"):
            assert classify_type(t) is MemoryType.RELATIONSHIP, t

    def test_procedure_hint(self):
        for t in ("how to restart the cache", "steps to deploy:",
                  "the runbook says run the command",
                  "workaround for the login bug"):
            assert classify_type(t) is MemoryType.PROCEDURE_HINT, t

    def test_absence(self):
        for t in ("the endpoint was removed", "we dropped support",
                  "there is no staging env", "it was deprecated",
                  "the flag is gone", "we decommissioned the box"):
            assert classify_type(t) is MemoryType.ABSENCE, t

    def test_state(self):
        for t in ("currently on v2", "as of now we use OIDC",
                  "right now the build is red", "it still uses md5"):
            assert classify_type(t) is MemoryType.STATE, t

    def test_event(self):
        for t in ("the deploy failed yesterday", "outage on friday",
                  "v2 shipped last week", "the migration happened",
                  "she joined the team", "build crashed"):
            assert classify_type(t) is MemoryType.EVENT, t

    def test_untyped(self):
        for t in ("the sky is blue", "water boils at 100C",
                  "a b c", ""):
            assert classify_type(t) is MemoryType.UNTYPED, t


class TestPrecedence:
    def test_decision_beats_absence(self):
        # "we decided to remove X" is a decision record about a removal
        assert classify_type("we decided to remove X") is \
            MemoryType.DECISION

    def test_relationship_beats_absence(self):
        # "no longer reports to" stays relationship — the pair shares
        # type with the affirmative form for update candidates
        assert classify_type("Alice no longer reports to Bob") is \
            MemoryType.RELATIONSHIP

    def test_preference_beats_absence(self):
        assert classify_type("I no longer like X") is \
            MemoryType.PREFERENCE

    def test_plan_beats_event(self):
        assert classify_type("plan to ship v3 next month") is \
            MemoryType.PLAN

    def test_procedure_beats_absence(self):
        assert classify_type("how to remove the cache") is \
            MemoryType.PROCEDURE_HINT


class TestContract:
    def test_producer_label(self):
        # the frozen producer version stamped on rows (§7)
        assert TYPING_PRODUCER == "enrich/v1"

    def test_determinism(self):
        t = "we decided to maybe remove the old endpoint"
        assert classify_type(t) == classify_type(t)

    def test_fact_never_emitted(self):
        # no marker set maps to fact — untyped is the honest label
        for t in ("it is a fact", "the fact remains", "note that x=1"):
            assert classify_type(t) is not MemoryType.FACT
