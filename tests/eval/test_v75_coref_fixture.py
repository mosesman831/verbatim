"""Owned coreference fixture tests (SPEC_V7 V7-13.20, SPEC_V7_5 §05 Q8).

Covers ``eval/v7/fixtures/coref_owned.jsonl`` and its generator
``eval/v7/make_coref_fixture.py`` plus the ``eval/v7/coref_eval.py``
harness:

* fixture shape — required keys, byte-pinned mention spans, canon
  surfaces actually present in turn text, referential integrity of the
  ``expected`` label (a mentioned canon or a session speaker);
* stratum counts — >= 400 cases, >= 100 resolve-labeled cases whose
  antecedent is beyond the immediately previous turn, >= 50
  two-candidate abstain traps (V7-13.20);
* generator determinism — ``generate()`` + ``serialize()`` reproduce the
  committed fixture byte-identically;
* harness — both Q8 arms run, counters are internally consistent, and
  the measured precision/recall/abstention are *reported* (never
  hardcoded): the test asserts the plumbing and prints the real numbers.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from eval.v7 import coref_eval, make_coref_fixture as gen
from verbatim.enrichment import coref_sieve

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "eval" / "v7" / "fixtures" / "coref_owned.jsonl"

REQUIRED_CASE_KEYS = {
    "case_id", "stratum", "strata", "mention", "unit_index", "span",
    "turns", "expected", "antecedent_turn", "antecedent_distance",
    "rationale",
}
REQUIRED_TURN_KEYS = {"text", "canon_mentions"}


@pytest.fixture(scope="module")
def cases():
    return coref_eval.load_cases(str(FIXTURE))


# ---------------------------------------------------------------------------
# Fixture shape
# ---------------------------------------------------------------------------

def test_fixture_file_exists():
    assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"


def test_fixture_header(cases):
    recs = list(gen.iter_jsonl(str(FIXTURE)))
    assert recs[0]["record"] == "fixture"
    assert recs[0]["generator"] == gen.GENERATOR_ID
    assert recs[0]["cases"] == len(cases)


def test_case_schema(cases):
    ids = set()
    for c in cases:
        assert REQUIRED_CASE_KEYS <= set(c), c["case_id"]
        assert re.fullmatch(r"cf-\d{4}", c["case_id"])
        ids.add(c["case_id"])
        assert isinstance(c["turns"], list) and c["turns"]
        assert isinstance(c["unit_index"], int)
        assert 0 <= c["unit_index"] < len(c["turns"])
        assert isinstance(c["strata"], list) and c["stratum"] in c["strata"]
        assert isinstance(c["mention"], str) and c["mention"]
        assert isinstance(c["rationale"], str)
        for t in c["turns"]:
            assert REQUIRED_TURN_KEYS <= set(t)
            assert isinstance(t["canon_mentions"], list)
            assert all(isinstance(x, str) for x in t["canon_mentions"])
    assert len(ids) == len(cases), "duplicate case_id"


def test_span_pins_mention(cases):
    for c in cases:
        text = c["turns"][c["unit_index"]]["text"]
        span = c["span"]
        assert text[span["start"]:span["end"]] == c["mention"], c["case_id"]
        # word-boundary integrity: the span is a whole token
        assert re.fullmatch(r"\w+", c["mention"]) or " " in c["mention"]


def test_canon_surfaces_present_in_text(cases):
    """Every canon_mentions entry surfaces in its turn text (folded)."""
    for c in cases:
        for t in c["turns"]:
            folded = t["text"].casefold()
            for canon in t["canon_mentions"]:
                assert canon.casefold() in folded, (c["case_id"], canon)


def test_expected_label_integrity(cases):
    """``expected`` is 'abstain' or a canon that is actually reachable in
    principle — mentioned in some turn or a session speaker canon."""
    for c in cases:
        exp = c["expected"]
        if exp == gen.ABSTAIN:
            continue
        mentioned = {m for t in c["turns"] for m in t["canon_mentions"]}
        speakers = {(t.get("speaker_canon") or t.get("speaker"))
                    for t in c["turns"]} - {None}
        assert exp in mentioned or exp in speakers, (c["case_id"], exp)


# ---------------------------------------------------------------------------
# Stratum requirements (V7-13.20)
# ---------------------------------------------------------------------------

def test_total_count(cases):
    assert len(cases) >= 400


def test_beyond_previous_turn_count(cases):
    deep = [c for c in cases
            if c["expected"] != gen.ABSTAIN
            and (c["antecedent_distance"] or 0) >= 2]
    assert len(deep) >= 100, "need >=100 antecedents beyond prev turn"
    # recompute from raw data: the expected canon must be absent from the
    # immediately previous turn and present deeper.
    for c in deep:
        ui = c["unit_index"]
        prev_mentions = c["turns"][ui - 1]["canon_mentions"] if ui else []
        assert c["expected"] not in prev_mentions, c["case_id"]


def test_two_candidate_trap_count(cases):
    traps = [c for c in cases
             if "two_candidate_trap" in c["strata"]
             and c["expected"] == gen.ABSTAIN]
    assert len(traps) >= 50, "need >=50 two-candidate abstain traps"


def test_traps_present_two_survivors(cases):
    """A 'two-candidate trap' must literally offer >= 2 class-compatible
    candidates — the sieve's own survivor list proves the construction."""
    for c in cases:
        if "two_candidate_trap" not in c["strata"]:
            continue
        out = coref_sieve.explain(c["mention"], c["unit_index"],
                                  c["turns"], lookback=gen.LOOKBACK,
                                  **dict(c.get("context") or {}))
        survivors = out["detail"].get("survivors")
        assert survivors is not None and len(survivors) >= 2, \
            (c["case_id"], c["mention"], survivors)


def test_pronoun_form_coverage(cases):
    """Required pronoun coverage: he/she/they/it/this/that/his/her/their."""
    mentions = {c["mention"] for c in cases}
    for m in ("he", "she", "they", "it", "this", "that",
              "his", "her", "their"):
        assert m in mentions, f"pronoun form uncovered: {m}"


def test_distance_coverage(cases):
    dists = {c["antecedent_distance"] for c in cases
             if c["antecedent_distance"] is not None}
    assert 0 in dists and 1 in dists
    assert {2, 3} <= dists, "mid-depth antecedents missing"
    assert max(dists) > gen.LOOKBACK, "beyond-window antecedents missing"


def test_candidate_count_coverage(cases):
    tags = {t for c in cases for t in c["strata"]}
    assert "zero_candidates" in tags or "no_antecedent" in tags
    assert "single_candidate" in tags
    assert "two_candidate_trap" in tags
    assert "mixed_class" in tags


def test_i_never_inherits_coverage(cases):
    tagged = [c for c in cases if "i_never_inherits" in c["strata"]]
    assert len(tagged) >= 10
    # at least one no-speaker case: 'i' must not inherit a prior canon
    assert any("no_speaker" in c["strata"] for c in tagged)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_generator_determinism():
    a, b = gen.generate(), gen.generate()
    assert a == b
    assert gen.serialize(a) == gen.serialize(b)


def test_generator_reproduces_committed_fixture():
    text = gen.serialize(gen.generate())
    assert text.encode("utf-8") == FIXTURE.read_bytes()


def test_lookback_matches_sieve():
    assert gen.LOOKBACK == coref_sieve.DEFAULT_LOOKBACK


# ---------------------------------------------------------------------------
# Harness — measured numbers are reported, never hardcoded
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def result(cases):
    return coref_eval.run(str(FIXTURE))


def test_harness_runs_both_arms(result):
    assert set(result["arms"]) == {"sieve", "previous_turn"}
    assert result["cases"] >= 400
    for arm, m in result["arms"].items():
        assert m["total"] == result["cases"]
        assert m["resolved"] + m["abstained"] == m["total"]
        assert m["correct"] + m["wrong"] == m["resolved"]
        assert (m["correct_abstain"] + m["missed"]) == m["abstained"]
        assert (m["expected_resolvable"] + m["expected_abstain"]
                ) == m["total"]
        if m["resolved"]:
            assert 0.0 <= m["precision"] <= 1.0
        if m["expected_resolvable"]:
            assert 0.0 <= m["recall"] <= 1.0


def test_harness_per_stratum_consistent(result, cases):
    tag_totals = {}
    for c in cases:
        for t in c["strata"]:
            tag_totals[t] = tag_totals.get(t, 0) + 1
    m = result["arms"]["sieve"]
    for tag, c in m["per_stratum"].items():
        assert c["total"] == tag_totals[tag], tag
        assert c["resolved"] + c["abstained"] == c["total"]


def test_measured_metrics_reported(result, capsys):
    """Print the real measured metrics — the Q8 gate numbers.  No quality
    assertion here: the fixture is the measuring instrument, and any
    sieve regression must surface as a reported number, not a rigged
    pass."""
    for arm, m in result["arms"].items():
        print(f"\n[coref fixture] arm={arm}: "
              f"precision={m['precision']}, recall={m['recall']}, "
              f"abstain={m['abstain_rate']}, "
              f"resolved={m['resolved']}, wrong={m['wrong']}, "
              f"missed={m['missed']}")
    out = capsys.readouterr().out
    assert "arm=sieve" in out and "arm=previous_turn" in out
    # the error list is reported verbatim for review
    errors = result["arms"]["sieve"]["errors"]
    for e in errors:
        print("  disagreement:", e)


def test_format_report_smoke(result):
    text = coref_eval.format_report(result)
    assert "ARM sieve" in text and "ARM previous_turn" in text
    assert "precision" in text
