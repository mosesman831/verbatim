"""Durable tests for the owned entity-resolution fixture + harness.

SPEC_V7 V7-08.13 / SPEC_V7_5 §05 Q4: the Q4 arm gate needs an owned
fixture with ≥ 100 same-name-different-person pairs and ≥ 100
nickname/alias pairs (plus pronoun-only mentions), and measurement
machinery that runs the current ``alias/v1`` resolver over it.

Covers: fixture schema + stratum counts; generator determinism (the
committed JSONL regenerates byte-identically); harness mechanics on
hand-built records (merge/abstain/separate/same_canon classification);
and the full-fixture run producing real, deterministic numbers.  Nothing
here asserts the resolver meets the gate — the current over-merge rate
is the *measurement* the Q4 arm exists to improve, so the tests pin the
machinery, not a target score.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from eval.v7 import er_eval, make_er_fixture as gen

FIXTURE = os.path.join(
    os.path.dirname(er_eval.__file__), "fixtures", "entity_resolution_owned.jsonl"
)

REQUIRED_FIELDS = {
    "pair_id", "stratum", "expected", "surface_a", "surface_b",
    "context_a", "context_b", "canons_a", "canons_b", "rationale",
    "known_canons", "caller_aliases", "extra_units",
    "speaker_a", "speaker_b", "pronoun_mention", "pronoun_only",
    "generator",
}

EXPECTATIONS = {"merge", "no_merge", "abstain"}
OUTCOMES = {"merge", "abstain", "separate", "same_canon"}


@pytest.fixture(scope="module")
def records():
    return gen.generate()


@pytest.fixture(scope="module")
def report(records):
    return er_eval.evaluate(records)


# ---------------------------------------------------------------------------
# fixture shape + stratum counts
# ---------------------------------------------------------------------------

class TestFixtureShape:
    def test_required_fields_present(self, records):
        for r in records:
            missing = REQUIRED_FIELDS - set(r)
            assert not missing, f"{r.get('pair_id')}: missing {missing}"

    def test_expected_values_valid(self, records):
        for r in records:
            assert r["expected"] in EXPECTATIONS, r["pair_id"]

    def test_pair_ids_unique(self, records):
        ids = [r["pair_id"] for r in records]
        assert len(ids) == len(set(ids))

    def test_surfaces_nonempty_and_contexts_carry_them(self, records):
        for r in records:
            assert r["surface_a"] and r["surface_b"]
            assert r["context_a"] and r["context_b"]
            # each surface is literally attested — in its context text, or
            # as that unit's speaker (speaker-relative A3 pairs).  The
            # pronoun_only stratum is exempt by definition.
            assert (
                r["surface_a"] in r["context_a"]
                or r["speaker_a"] == r["surface_a"]
            ), r["pair_id"]
            if not r["pronoun_only"]:
                assert (
                    r["surface_b"] in r["context_b"]
                    or r["speaker_b"] == r["surface_b"]
                ), r["pair_id"]

    def test_gate_counts(self, records):
        sn = [r for r in records if r["stratum"].startswith("sn_")]
        mg = [r for r in records if r["expected"] == "merge"]
        assert len(sn) >= 100, "V7-08.13 needs ≥ 100 same-name pairs"
        assert len(mg) >= 100, "V7-08.13 needs ≥ 100 nickname/alias pairs"
        # most same-name pairs must be real decisions, not canon-fold
        # trivialities — keep identical-surface pairs a minority
        ident = [r for r in sn if r["stratum"] == "sn_identical_surface"]
        assert len(ident) <= 0.25 * len(sn)

    def test_pronoun_only_mentions_present(self, records):
        assert any(r["pronoun_only"] for r in records), (
            "V7-08.13 requires pronoun-only mentions in the fixture"
        )

    def test_no_merge_pairs_have_distinct_or_flagged_canons(self, records):
        for r in records:
            if r["expected"] in ("no_merge", "abstain"):
                assert r["canons_a"] and r["canons_b"], r["pair_id"]

    def test_every_record_has_rationale(self, records):
        for r in records:
            assert len(r["rationale"]) >= 20, r["pair_id"]


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_generate_is_pure(self):
        a = gen.generate()
        b = gen.generate()
        assert a == b

    def test_committed_fixture_matches(self, tmp_path):
        regen = tmp_path / "regen.jsonl"
        gen.write_fixture(str(regen))
        with open(FIXTURE, "rb") as fh:
            committed = fh.read()
        with open(str(regen), "rb") as fh:
            assert fh.read() == committed, (
                "fixture drifted — rerun eval/v7/make_er_fixture.py"
            )

    def test_harness_deterministic(self, records):
        r1 = er_eval.evaluate(records)
        r2 = er_eval.evaluate(records)
        assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


# ---------------------------------------------------------------------------
# harness mechanics on hand-built records
# ---------------------------------------------------------------------------

def _mini(**kw):
    base = {
        "pair_id": "t-0", "stratum": "t", "expected": "merge",
        "surface_a": "Alice Chen", "surface_b": "Alice",
        "context_a": "Alice Chen led the review.",
        "context_b": "Alice filed the minutes.",
        "speaker_a": None, "speaker_b": None, "extra_units": [],
        "known_canons": [], "caller_aliases": [],
        "canons_a": ["alice chen"], "canons_b": ["alice"],
        "pronoun_mention": False, "pronoun_only": False,
        "rationale": "mini", "generator": "test",
    }
    base.update(kw)
    return base


class TestHarnessMechanics:
    def test_active_link_is_merge(self):
        d = er_eval.decide_pair(_mini())
        assert d["outcome"] == "merge"
        assert d["links"] and d["links"][0]["state"] == "active"

    def test_conflicted_link_is_abstain(self):
        d = er_eval.decide_pair(_mini(known_canons=["alice chen", "alice wu"]))
        assert d["outcome"] == "abstain"
        assert all(l["state"] == "candidate" for l in d["links"])

    def test_no_link_is_separate(self):
        d = er_eval.decide_pair(_mini(
            surface_b="Kate", context_b="Kate filed the minutes.",
            canons_b=["kate"]))
        assert d["outcome"] == "separate"

    def test_same_canon_detected(self):
        d = er_eval.decide_pair(_mini(
            surface_b="ALICE CHEN",
            context_b="ALICE CHEN filed the minutes.",
            canons_b=["alice chen"]))
        assert d["outcome"] == "same_canon"

    def test_caller_alias_merge(self):
        d = er_eval.decide_pair(_mini(
            surface_b="Kate", context_b="Kate filed the minutes.",
            canons_b=["kate"],
            caller_aliases=[["Alice Chen", "Kate"]]))
        assert d["outcome"] == "merge"

    def test_explicit_statement_merge(self):
        d = er_eval.decide_pair(_mini(
            context_a="Alice Chen goes by Ace at the lab.",
            surface_b="Ace", context_b="Ace filed the minutes.",
            canons_b=["ace"]))
        assert d["outcome"] == "merge"
        assert any(l["rule_id"] == "A3" for l in d["links"])

    def test_metrics_direction(self):
        # a resolver that merges everything must over-merge; one that
        # merges nothing must have zero recall — the metric sees both.
        recs = [
            _mini(pair_id="p1", expected="no_merge"),          # A1 over-merge
            _mini(pair_id="p2", expected="merge"),
            _mini(pair_id="p3", expected="merge",
                  surface_b="Kate", context_b="Kate filed.",
                  canons_b=["kate"]),                          # silent miss
        ]
        rep = er_eval.evaluate(recs)
        assert rep["over_merge_rate"] == 1.0
        assert rep["merge_recall"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# full-fixture run — real numbers, sane bounds
# ---------------------------------------------------------------------------

class TestFullRun:
    def test_report_shape(self, report):
        for key in ("over_merge_rate", "merge_recall", "abstain_rate",
                    "per_stratum", "details", "over_merge_pairs",
                    "same_name_over_merge_rate"):
            assert key in report
        assert report["resolver"]["alias_rules"] == "alias/v1"
        assert report["n_pairs"] >= 200

    def test_metrics_in_bounds(self, report):
        for m in ("over_merge_rate", "same_name_over_merge_rate",
                  "merge_recall", "abstain_rate"):
            assert 0.0 <= report[m] <= 1.0, m

    def test_every_detail_classified(self, report):
        for d in report["details"]:
            assert d["outcome"] in OUTCOMES, d["pair_id"]

    def test_same_name_denominator_meets_gate_size(self, report):
        assert report["same_name_n"] >= 100
        assert report["merge_denominator"] >= 100

    def test_abstention_path_exercised(self, report):
        # the fixture must produce real candidate rows somewhere —
        # otherwise the abstention metric is vacuous
        assert any(d["outcome"] == "abstain" for d in report["details"])
        assert report["abstain_rate"] > 0.0

    def test_harness_detects_overmerges(self, report):
        # alias/v1 is context-blind on subset/initial collisions by design;
        # if this is ever zero the fixture is probably broken, not the
        # resolver fixed — assert the machinery sees real over-merges.
        assert report["over_merge_n"] > 0
        assert all(p["links"] for p in report["over_merge_pairs"])
