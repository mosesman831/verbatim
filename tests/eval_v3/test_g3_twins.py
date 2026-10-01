"""G3 deep twin corpus — durable harness test.

The G3 gate requires the false-supersession upper-95% bound <0.01; at
~460 near-miss twins the rule-of-three bound reaches ~0.0065 when the
detectors stay clean. ``eval.v3.g3_twins`` generates the corpus
deterministically and runs every case through the real admission-time
detectors in a real Store — this test asserts the measurement itself.
"""

from __future__ import annotations

import pytest

from eval.v3.g3_twins import gen_cases, run_case
from verbatim.storage.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "g3.db"))
    yield s
    s.close()


def test_corpus_is_deterministic_and_sized_for_the_bound():
    a, ta = gen_cases(42)
    b, tb = gen_cases(42)
    assert [c["id"] for c in a] == [c["id"] for c in b]
    assert [c["new_text"] for c in a] == [c["new_text"] for c in b]
    assert len(a) >= 460
    assert len(ta) == len(tb) > 0
    assert {c["category"] for c in a} >= {
        "hedged_question", "hearsay", "negated", "future", "conditional",
        "noun_form", "reinstated", "imperative", "corroborating",
        "family_name", "other_retired", "imperative_neg",
    }


def test_zero_false_positives_over_full_corpus(store):
    """Every twin through both real detectors — any proposal is a false
    positive. The G3 bound this test preserves: 3/460 ≈ 0.0065."""
    twins, trues = gen_cases(42)
    flagged = []
    for case in twins:
        sup, rel = run_case(store, case, case["id"])
        if sup + rel:
            flagged.append((case["id"], case["category"]))
    assert flagged == []


def test_true_pairs_still_detected(store):
    """A corpus everything passes proves nothing — true retirement
    assertions must still bind their counterparties."""
    _, trues = gen_cases(42)
    missed = [
        case["id"] for case in trues
        if not sum(run_case(store, case, case["id"]))
    ]
    assert missed == []
