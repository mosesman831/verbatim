"""Durable tests for V7-33.12 proportional recall@k (SPEC_V7_5
V75-04.05, acceptance J14).

Pins LoCoMo's official retrieval R@k:

* ``metrics.evidence_proportional_at_k`` — per-question
  ``|G ∩ R_k| / |G|`` over the first-k *distinct* delivered units,
  hand-computed on partial-overlap, k<|G|, k>|G|, dedup, and
  multi-ref-unit fixtures; empty G returns ``None`` and is excluded
  from the aggregate mean while still counted via ``n``/``n_gold``;
* ``aggregate``/``per_category`` carry ``prop@k`` beside ``any@k``/
  ``all@k`` plus ``gold_mean`` (mean |G| over every question in the
  slice, empty-gold questions counting 0);
* the Track R report prints all three metric families side by side
  with ``mean|G|`` and LoCoMo "category id name" row labels.
"""

from __future__ import annotations

import pytest

from eval.v7 import metrics as M
from eval.v7.arms import DictCorpus, QueryOutcome
from eval.v7.track_r import render_markdown, run_track_r


# ---------------------------------------------------------------------------
# unit level — hand-computed proportional recall
# ---------------------------------------------------------------------------


class TestProportionalAtK:
    def test_partial_overlap_fraction(self):
        # G={a,b,c,d}; R_4 covers {a,b} → 2/4
        assert M.evidence_proportional_at_k(
            {"a", "b", "c", "d"}, ["a", "x", "b", "y"], 4
        ) == pytest.approx(0.5)

    def test_diverges_from_any_and_all(self):
        # the metric's raison d'être: any@k saturates at 1.0 while the
        # question is only a quarter answered (V75-04.05 multi-hop)
        g = {"a", "b", "c", "d"}
        assert M.evidence_proportional_at_k(g, ["a"], 10) == pytest.approx(
            0.25
        )
        assert M.evidence_any_at_k(g, ["a"], 10) == 1.0
        assert M.evidence_all_at_k(g, ["a"], 10) == 0.0

    def test_full_and_zero_coverage(self):
        assert M.evidence_proportional_at_k(
            {"a", "b"}, ["a", "b"], 10
        ) == 1.0
        assert M.evidence_proportional_at_k({"a", "b"}, ["x", "y"], 10) == 0.0
        assert M.evidence_proportional_at_k({"a"}, [], 10) == 0.0

    def test_k_smaller_than_gold(self):
        g = {"a", "b", "c"}
        assert M.evidence_proportional_at_k(g, ["a"], 1) == pytest.approx(
            1 / 3
        )
        assert M.evidence_proportional_at_k(g, ["a", "b"], 2) == pytest.approx(
            2 / 3
        )
        # k cuts the prefix: third gold ref sits at rank 3
        assert M.evidence_proportional_at_k(
            g, ["a", "b", "c"], 2
        ) == pytest.approx(2 / 3)

    def test_k_larger_than_gold(self):
        assert M.evidence_proportional_at_k(
            {"a"}, ["x", "y", "a"], 20
        ) == 1.0
        assert M.evidence_proportional_at_k(
            {"a", "b"}, ["x", "a"], 20
        ) == pytest.approx(0.5)

    def test_distinct_units_dedup(self):
        # same scalar ref redelivered is ONE unit — the duplicate does
        # not spend a slot of the k-prefix
        g = {"a", "b"}
        assert M.evidence_proportional_at_k(g, ["a", "a", "b"], 2) == 1.0
        assert M.evidence_proportional_at_k(g, ["a", "a"], 2) == 0.5
        # ...but identical *set* units are distinct units (they share
        # coverage, not identity) — matching any@k/all@k semantics
        assert M.evidence_proportional_at_k(
            {"a"}, [{"s"}, {"s"}, {"a"}], 2
        ) == 0.0

    def test_multi_ref_unit_coverage(self):
        # a delivered unit covering two gold refs credits both
        assert M.evidence_proportional_at_k(
            {"a", "b", "c"}, [{"a", "b"}], 1
        ) == pytest.approx(2 / 3)

    def test_empty_gold_is_none(self):
        assert M.evidence_proportional_at_k(set(), ["a"], 10) is None
        assert M.evidence_proportional_at_k(None, ["a"], 10) is None
        assert M.evidence_proportional_at_k({}, [], 10) is None

    def _rec(self, **kw):
        base = dict(
            task_id="t", arm="a", answerable=True,
            gold={"g": 1.0}, delivered=(), n_items=0,
        )
        base.update(kw)
        return M.TaskScore(**base)

    def test_aggregate_excludes_empty_gold(self):
        recs = [
            self._rec(task_id="a", gold={"x": 1, "y": 1},
                      delivered=("x", "y")),          # prop 1.0
            self._rec(task_id="b", gold={"x": 1, "y": 1},
                      delivered=("x",)),              # prop 0.5
            self._rec(task_id="c", gold={}, delivered=("x",)),  # excluded
        ]
        agg = M.aggregate(recs, (10,))
        assert agg["prop@10"] == pytest.approx(0.75)
        assert agg["n"] == 3 and agg["n_gold"] == 2
        # mean |G| counts every question — the empty-gold one is 0
        assert agg["gold_mean"] == pytest.approx(4 / 3)

    def test_aggregate_every_k_and_per_category(self):
        recs = [
            self._rec(task_id="a", gold={"x": 1, "y": 1},
                      delivered=("x",), category="multi_hop"),
            self._rec(task_id="b", gold={"z": 1},
                      delivered=("z",), category="single_hop"),
        ]
        agg = M.aggregate(recs, (10, 20))
        assert agg["prop@10"] == pytest.approx(0.75)
        assert agg["prop@20"] == pytest.approx(0.75)
        cats = M.per_category(recs, (10,))
        assert cats["multi_hop"]["prop@10"] == pytest.approx(0.5)
        assert cats["multi_hop"]["gold_mean"] == pytest.approx(2.0)
        assert cats["single_hop"]["prop@10"] == pytest.approx(1.0)
        assert cats["single_hop"]["gold_mean"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# track_r level — report carries all three families + mean |G| + ids
# ---------------------------------------------------------------------------

ITEMS = [
    {"id": f"d{i}", "text": f"evidence text number {i}",
     "session_id": "s1"}
    for i in range(1, 7)
]

TASKS = [
    {"task_id": "q1", "query": "multi hop question one",
     "category": "multi_hop", "gold_evidence": ["d1", "d2", "d3"],
     "metadata": {"category_id": 1}},
    {"task_id": "q2", "query": "multi hop question two",
     "category": "multi_hop", "gold_evidence": ["d4"],
     "metadata": {"category_id": 1}},
    {"task_id": "q3", "query": "single hop question",
     "category": "single_hop", "gold_evidence": ["d5"],
     "metadata": {"category_id": 4}},
    {"task_id": "q4", "query": "unanswerable probe",
     "category": "adversarial", "answerable": False,
     "metadata": {"category_id": 5}},
]


class ScriptedArm:
    """Deterministic arm — returns a fixed ref list per task id so the
    report's prop@k is hand-computable."""

    name = "scripted"

    def __init__(self, refs_by_task):
        self._refs = dict(refs_by_task)

    def ingest(self, corpus):
        return {"indexed": len(ITEMS)}

    def indexed_refs(self):
        return {f"d{i}" for i in range(1, 7)}

    def query(self, task, k):
        refs = list(self._refs.get(task.task_id, ()))[: int(k)]
        return QueryOutcome(refs=refs, n_items=len(refs), k=int(k))


@pytest.fixture()
def report():
    corpus = DictCorpus(ITEMS, TASKS, name="tiny", dataset_id="tiny-dict")
    arm = ScriptedArm({
        "q1": ["d1", "x", "d2"],   # covers 2 of 3 gold → prop 2/3
        "q2": ["d4"],              # covers 1 of 1     → prop 1.0
        "q3": ["d5", "d6"],        # covers 1 of 1     → prop 1.0
        "q4": ["d6"],              # unanswerable — spurious delivery
    })
    return run_track_r(corpus, [arm], k_list=(10, 20), seed=0)


class TestTrackRProportional:
    def test_aggregate_values(self, report):
        arm = report["arms"]["scripted"]
        ov = arm["overall"]
        # any@k saturates while prop@k reports the honest fraction:
        # (2/3 + 1 + 1)/3 = 8/9
        assert ov["any@10"] == pytest.approx(1.0)
        assert ov["all@10"] == pytest.approx(2 / 3)
        assert ov["prop@10"] == pytest.approx(8 / 9)
        assert ov["prop@20"] == pytest.approx(8 / 9)
        # mean |G| over all four questions: (3 + 1 + 1 + 0)/4 = 1.25
        assert ov["gold_mean"] == pytest.approx(1.25)

        cats = arm["categories"]
        assert cats["multi_hop"]["prop@10"] == pytest.approx(
            (2 / 3 + 1.0) / 2
        )
        assert cats["multi_hop"]["gold_mean"] == pytest.approx(2.0)
        assert cats["adversarial"]["n_gold"] == 0
        assert cats["adversarial"]["prop@10"] is None
        assert cats["adversarial"]["gold_mean"] == pytest.approx(0.0)

    def test_category_ids_recorded(self, report):
        assert report["dataset"]["category_ids"] == {
            "multi_hop": 1, "single_hop": 4, "adversarial": 5,
        }

    def test_markdown_prints_all_metric_families(self, report):
        md = render_markdown(report)
        header = next(
            ln for ln in md.splitlines() if "any@10" in ln and "|" in ln
        )
        # every existing column kept, prop@k inserted beside any/all
        # (mean|G| is markdown-escaped so it survives the pipe split)
        for col in ("n_gold", "mean\\|G\\|",
                    "any@10", "any@20",
                    "all@10", "all@20",
                    "prop@10", "prop@20",
                    "ndcg@10", "mrr@10", "zero", "abstain"):
            assert col in header, col
        # side-by-side ordering: any < all < prop < ndcg
        assert header.index("any@10") < header.index("all@10") \
            < header.index("prop@10") < header.index("ndcg@10")

    def test_markdown_category_id_and_name(self, report):
        md = render_markdown(report)
        assert "| scripted | 1 multi_hop |" in md
        assert "| scripted | 4 single_hop |" in md
        assert "| scripted | 5 adversarial |" in md
        # hand-computed cells visible in the row
        row = next(
            ln for ln in md.splitlines() if "| 1 multi_hop |" in ln
        )
        cells = [c.strip() for c in row.split("|")]
        assert "2.000" in cells          # mean |G| = (3+1)/2
        assert "0.833" in cells          # prop@10 = (2/3+1)/2

    def test_name_fallback_without_observed_ids(self):
        # a dict corpus with LoCoMo category *names* but no metadata
        # ids still prints "id name" via the static V7-22.07 table;
        # non-LoCoMo categories render bare
        tasks = [
            {"task_id": "t1", "query": "q", "category": "temporal",
             "gold_evidence": ["d1"]},
            {"task_id": "t2", "query": "q", "category": "custom_lane",
             "gold_evidence": ["d2"]},
        ]
        corpus = DictCorpus(ITEMS, tasks, name="n", dataset_id="n")
        rep = run_track_r(corpus, [ScriptedArm({})], k_list=(10,))
        md = render_markdown(rep)
        assert "| scripted | 2 temporal |" in md
        assert "| scripted | custom_lane |" in md
