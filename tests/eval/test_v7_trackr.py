"""Durable tests for the V7 Track R harness (SPEC_V7 §22.3, §33).

Pins the retrieval-only measurement contract:

* metric functions reproduce hand-computed any@k / all@k / NDCG
  (binary AND graded gold), MRR, zero-result and refusal rates — with
  empty-gold questions honestly excluded from denominators;
* the reference arms execute for real on a tiny inline dict corpus —
  flat BM25 and FTS5 must surface planted evidence;
* the verbatim arm drives the real ``Memory`` facade end-to-end
  (add → drain → settle → search) — no fixture store;
* the report carries per-arm overall + per-category + session
  granularity + attribution + manifest-stub fields, and two runs of
  deterministic inputs produce identical metrics;
* attribution classes are correct on synthetic diagnostics;
* ``lanes_disabled`` plumbing reports applied vs unapplied honestly.
"""

from __future__ import annotations

import math

import pytest

from eval.v7 import metrics as M
from eval.v7.arms import (
    DictCorpus,
    FTS5Arm,
    FlatBM25Arm,
    QueryOutcome,
    VerbatimArm,
    VerbatimClaimsArm,
    arm_name,
    arm_task,
    corpus_items,
    item_document,
    item_ref,
    make_arm,
    _group_anchors,
    _task_as_of,
)
from eval.v7.attribution import (
    aggregate_table,
    attribute,
    attribute_detail,
)
from eval.v7.track_r import render_markdown, run_track_r


# ---------------------------------------------------------------------------
# fixtures — a tiny inline dict corpus (no external data)
# ---------------------------------------------------------------------------

ITEMS = [
    {"id": "d1", "text": "Caroline adopted a rescue dog named Biscuit "
                         "last March.", "speaker": "Caroline",
     "session_id": "s1", "when": "2023-05-01"},
    {"id": "d2", "text": "Melanie paints watercolors on weekends.",
     "speaker": "Melanie", "session_id": "s1", "when": "2023-05-01"},
    {"id": "d3", "text": "Caroline's dog Biscuit is a golden retriever "
                         "mix.", "speaker": "Caroline",
     "session_id": "s2", "when": "2023-05-08"},
    {"id": "d4", "text": "The pottery studio opens at 9am on Saturdays.",
     "speaker": "Melanie", "session_id": "s2", "when": "2023-05-08"},
    {"id": "d5", "text": "Filler about trains and timestamps and lunch "
                         "menus.", "speaker": "Caroline",
     "session_id": "s3", "when": "2023-05-15"},
]

TASKS = [
    {"task_id": "q1", "query": "what kind of dog did Caroline adopt",
     "category": "single_hop", "gold_evidence": ["d1"],
     "evidence_session_ids": ["s1"]},
    {"task_id": "q2", "query": "when does the pottery studio open",
     "category": "temporal", "gold_evidence": ["d4"]},
    {"task_id": "q3", "query": "what breed is Biscuit",
     "category": "single_hop", "gold_evidence": ["d1", "d3"]},
    {"task_id": "q4", "query": "who won the chess tournament",
     "category": "adversarial", "answerable": False},
]


@pytest.fixture()
def corpus():
    return DictCorpus(ITEMS, TASKS, name="tiny", dataset_id="tiny-dict")


def _strip_volatile(rep):
    """Remove measured-time fields so two runs compare byte-equal."""
    import copy

    r = copy.deepcopy(rep)
    r.pop("environment", None)
    for arm in r["arms"].values():
        blocks = [arm.get("overall") or {}]
        blocks += list((arm.get("categories") or {}).values())
        blocks += list((arm.get("split") or {}).values())
        sess = arm.get("session") or {}
        blocks.append(sess.get("overall") or {})
        blocks += list((sess.get("categories") or {}).values())
        for blk in blocks:
            blk.pop("latency_ms", None)
        (arm.get("ingest") or {}).pop("ingest_ms", None)
        (arm.get("ingest") or {}).pop("settle_waited_s", None)
    for rec in r.get("per_task", []):
        rec.pop("latency_ms", None)
    return r


# ---------------------------------------------------------------------------
# metrics — hand-computed fixtures
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_any_at_k(self):
        g = {"a", "b"}
        assert M.evidence_any_at_k(g, ["x", "a"], 1) == 0.0
        assert M.evidence_any_at_k(g, ["x", "a"], 2) == 1.0
        assert M.evidence_any_at_k(g, ["x", "y"], 20) == 0.0

    def test_all_at_k(self):
        g = {"a", "b"}
        assert M.evidence_all_at_k(g, ["a", "b", "x"], 2) == 1.0
        assert M.evidence_all_at_k(g, ["a", "x", "b"], 2) == 0.0
        assert M.evidence_all_at_k(g, ["a"], 10) == 0.0

    def test_ndcg_binary_hand_computed(self):
        # gold {a,b}; delivered [a,x]: DCG = 1/log2(2) = 1.0
        # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.630930 = 1.630930
        got = M.ndcg_at_k({"a", "b"}, ["a", "x"], 2)
        assert got == pytest.approx(1.0 / 1.6309297535714575, abs=1e-9)

    def test_ndcg_perfect_and_zero(self):
        assert M.ndcg_at_k({"a", "b"}, ["a", "b"], 2) == pytest.approx(1.0)
        assert M.ndcg_at_k({"a", "b"}, ["x", "y"], 2) == 0.0

    def test_ndcg_graded_partial_credit(self):
        # gold grades a:2, b:1; delivered [b,a]
        # DCG = 1/log2(2) + 2/log2(3) = 1 + 1.261860 = 2.261860
        # IDCG = 2/log2(2) + 1/log2(3) = 2 + 0.630930 = 2.630930
        got = M.ndcg_at_k({"a": 2.0, "b": 1.0}, ["b", "a"], 2)
        assert got == pytest.approx(2.2618595071429146 / 2.6309297535714574,
                                    abs=1e-9)

    def test_k05_ndcg_session_le_one(self):
        # K05 / V8-15.01 / §21.8 — three delivered turns from one gold
        # session at session granularity: the first covering unit
        # consumes the gold ref; the later two earn no further credit.
        # Pre-fix DCG = 1 + 0.630930 + 0.5 = 2.130930 > IDCG → nDCG 2.13.
        gold = {"s1"}
        turns = [{"d1", "s1"}, {"d2", "s1"}, {"d3", "s1"}]
        got = M.ndcg_at_k(gold, turns, 10)
        assert got == pytest.approx(1.0)
        assert got <= 1.0
        # 1.0 only when the first delivered unit is gold — a non-gold
        # head pushes the credit to rank 2's discount.
        late = [{"x"}, {"d1", "s1"}, {"d2", "s1"}]
        assert M.ndcg_at_k(gold, late, 10) == pytest.approx(
            1.0 / math.log2(3))
        # no delivered unit covers the session → 0
        assert M.ndcg_at_k(gold, [{"x"}, {"y"}], 10) == 0.0

    def test_k05_ndcg_session_credit_once_boundary(self):
        # Credit-once boundary: the second unit in the same gold session
        # adds nothing; a fresh gold ref at rank 3 still credits.
        # gold {s1, s2}; delivered covers s1, s1 again, then s2.
        # DCG = 1/log2(2) + 0 + 1/log2(4) = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1.630930  (pre-fix nDCG ≈ 1.31)
        gold = {"s1", "s2"}
        turns = [{"d1", "s1"}, {"d2", "s1"}, {"d3", "s2"}]
        expect = 1.5 / (1.0 + 1.0 / math.log2(3))
        got = M.ndcg_at_k(gold, turns, 10)
        assert got == pytest.approx(expect)
        assert got <= 1.0
        # graded gold: the covering unit consumes every fresh ref it
        # covers, not just the argmax — a later unit re-covering the
        # weaker ref earns 0.
        graded = {"s1": 2.0, "s2": 1.0}
        dup = [{"d1", "s1", "s2"}, {"d2", "s2"}]
        # DCG = 2/log2(2) + 0; IDCG = 2 + 1/log2(3)
        assert M.ndcg_at_k(graded, dup, 10) == pytest.approx(
            2.0 / (2.0 + 1.0 / math.log2(3)))

    def test_mrr(self):
        assert M.mrr_at_k({"b"}, ["x", "y", "b"], 3) == pytest.approx(1 / 3)
        assert M.mrr_at_k({"b"}, ["x", "y", "b"], 2) == 0.0
        assert M.mrr_at_k({"b"}, ["b"], 1) == 1.0

    def test_distinct_units_and_multi_ref_coverage(self):
        # duplicate scalar ref collapses — same unit redelivered
        assert M.evidence_any_at_k({"a"}, ["x", "x", "a"], 2) == 1.0
        # a unit may cover several gold refs (session granularity sets)
        units = [{"d1", "x"}, {"d2", "s1"}]
        assert M.evidence_any_at_k({"s1"}, units, 2) == 1.0
        assert M.evidence_all_at_k({"d1", "s1"}, units, 2) == 1.0
        # ...but identical set elements are distinct units — NOT deduped
        assert M.evidence_any_at_k({"a"}, [{"s"}, {"s"}, {"a"}], 2) == 0.0

    def test_empty_gold_excluded(self):
        assert M.evidence_any_at_k(set(), ["a"], 10) is None
        assert M.evidence_all_at_k([], ["a"], 10) is None
        assert M.ndcg_at_k({}, ["a"], 10) is None
        assert M.mrr_at_k(None, ["a"], 10) is None

    def _rec(self, **kw):
        base = dict(task_id="t", arm="a", answerable=True,
                    gold={"g": 1.0}, delivered=(), n_items=0)
        base.update(kw)
        return M.TaskScore(**base)

    def test_zero_result_rate(self):
        recs = [
            self._rec(task_id="a", n_items=0),
            self._rec(task_id="b", n_items=2),
            self._rec(task_id="c", gold={}, n_items=0),     # no gold: excluded
            self._rec(task_id="d", answerable=False, n_items=0),  # excluded
        ]
        assert M.zero_result_rate(recs) == pytest.approx(0.5)

    def test_refusal_and_abstention_rates(self):
        recs = [
            self._rec(task_id="a", withheld=True),                    # answerable withheld
            self._rec(task_id="b", n_items=3),
            self._rec(task_id="c", answerable=False, withheld=True),  # refused
            self._rec(task_id="d", answerable=False, n_items=1),      # spurious
            self._rec(task_id="e", answerable=False, n_items=0),      # empty = refused
        ]
        assert M.false_abstention_rate(recs) == pytest.approx(0.5)
        assert M.correct_refusal_rate(recs) == pytest.approx(2 / 3)
        assert M.abstention_rate(recs) == pytest.approx(2 / 5)

    def test_aggregate_denominators(self):
        recs = [
            self._rec(task_id="a", delivered=("g",), n_items=1,
                      category="c1"),
            self._rec(task_id="b", delivered=("z",), n_items=1,
                      category="c1"),
            self._rec(task_id="c", gold={}, n_items=0, category="c2"),
        ]
        agg = M.aggregate(recs, (10,))
        assert agg["n"] == 3 and agg["n_gold"] == 2
        assert agg["any@10"] == pytest.approx(0.5)
        assert agg["ndcg@10"] == pytest.approx(0.5)
        cats = M.per_category(recs, (10,))
        assert cats["c1"]["n_gold"] == 2
        assert cats["c2"]["n_gold"] == 0
        assert cats["c2"]["any@10"] is None

    def test_estimate_tokens_monotone_nonempty(self):
        assert M.estimate_tokens("") == 0
        a = M.estimate_tokens("hello world")
        b = M.estimate_tokens("hello world, this is a longer sentence!")
        assert 0 < a < b


# ---------------------------------------------------------------------------
# arms — the reference arms on the tiny corpus
# ---------------------------------------------------------------------------


class TestFlatArms:
    @pytest.mark.parametrize("cls", [FlatBM25Arm, FTS5Arm])
    def test_planted_evidence_found(self, corpus, cls):
        arm = cls()
        rep = arm.ingest(corpus)
        assert rep["indexed"] == len(ITEMS)
        assert arm.indexed_refs() == {"d1", "d2", "d3", "d4", "d5"}
        for t in TASKS[:3]:
            out = arm.query(arm_task(t), 10)
            assert isinstance(out, QueryOutcome)
            assert list(out) == out.refs          # it IS the ref list
            assert out.status == "ok"
            assert not out.abstained
            gold = set(t["gold_evidence"])
            assert gold & set(out.refs), (t["task_id"], out.refs)
        # multi-gold task: both refs surface
        out = arm.query(arm_task(TASKS[2]), 10)
        assert {"d1", "d3"} <= set(out.refs)
        # unanswerable probe: arms never withhold (no verdict machinery)
        out = arm.query(arm_task(TASKS[3]), 10)
        assert out.abstained is False

    def test_flat_bm25_deterministic(self, corpus):
        a, b = FlatBM25Arm(), FlatBM25Arm()
        a.ingest(corpus)
        b.ingest(corpus)
        for t in TASKS:
            assert a.query(arm_task(t), 10).refs == \
                b.query(arm_task(t), 10).refs

    def test_arm_sees_no_gold(self, corpus):
        at = arm_task(TASKS[0])
        assert at.task_id == "q1" and at.query
        assert not hasattr(at, "gold_evidence")
        assert not hasattr(at, "answerable")


# ---------------------------------------------------------------------------
# attribution — synthetic diagnostics hit every class
# ---------------------------------------------------------------------------


class TestAttribution:
    def _diag(self, **kw):
        base = dict(refs=[], surfaced=[], status="ok", abstained=False,
                    suppressed=0, omitted=0, error=None, k=10)
        base.update(kw)
        return base

    def test_delivered(self):
        t = {"gold_evidence": ["a"], "answerable": True}
        assert attribute(t, self._diag(refs=["a"], surfaced=["a"])) \
            == "delivered"

    def test_unsupported_no_gold_or_unanswerable(self):
        assert attribute({"answerable": False}, self._diag()) == "unsupported"
        assert attribute({"gold_evidence": []}, self._diag()) == "unsupported"

    def test_unsupported_not_indexed(self):
        t = {"gold_evidence": ["g"], "answerable": True}
        assert attribute(t, self._diag(), indexed={"x", "y"}) \
            == "unsupported"

    def test_abstain_verdict_withheld(self):
        t = {"gold_evidence": ["g"], "answerable": True}
        assert attribute(
            t, self._diag(status="insufficient", abstained=True,
                          suppressed=4),
            indexed={"g"}) == "abstain"

    def test_lane_miss(self):
        t = {"gold_evidence": ["g"], "answerable": True}
        d = self._diag(refs=["x"], surfaced=["x", "y"])
        assert attribute(t, d, indexed={"g"}) == "lane_miss"

    def test_rank_shift(self):
        t = {"gold_evidence": ["g"], "answerable": True}
        d = self._diag(refs=["x"], surfaced=["x", "g"], k=1)
        assert attribute(t, d, indexed={"g"}) == "rank_shift"

    def test_packed_out(self):
        t = {"gold_evidence": ["g"], "answerable": True}
        # gold surfaced at rank 1 (<=k) yet not delivered — the pack
        # stage dropped a top-k candidate
        d = self._diag(refs=["x"], surfaced=["g", "x"], k=1)
        assert attribute(t, d, indexed={"g"}) == "packed_out"

    def test_unattributed_invisible_tail(self):
        t = {"gold_evidence": ["g"], "answerable": True}
        # gold not in the observed pool but the verdict suppressed
        # candidates we cannot see — never guess
        assert attribute(
            t, self._diag(refs=["x"], surfaced=["x"], suppressed=3),
            indexed={"g"}) == "unattributed"
        # pool not observable at all
        assert attribute(
            t, self._diag(refs=["x"], surfaced=None),
            indexed={"g"}) == "unattributed"
        # arm error
        assert attribute(
            t, self._diag(refs=[], surfaced=None, error="boom"),
            indexed={"g"}) == "unattributed"

    def test_aggregate_table(self):
        recs = [
            M.TaskScore(task_id="a", arm="x", category="c1",
                        attribution="delivered"),
            M.TaskScore(task_id="b", arm="x", category="c1",
                        attribution="lane_miss"),
            M.TaskScore(task_id="c", arm="x", category="c2",
                        attribution="abstain"),
        ]
        tab = aggregate_table(recs)
        assert tab["total"] == 3 and tab["misses"] == 2
        assert tab["by_class"]["delivered"] == 1
        assert tab["by_category"]["c1"]["lane_miss"] == 1
        miss = aggregate_table(recs, misses_only=True)
        assert miss["total"] == 2 and miss["by_class"]["delivered"] == 0


# ---------------------------------------------------------------------------
# track_r — report shape, slicing, determinism
# ---------------------------------------------------------------------------


class TestTrackR:
    def test_report_shape_and_categories(self, corpus):
        rep = run_track_r(corpus, [FlatBM25Arm(), FTS5Arm()],
                          k_list=(10, 20), seed=0)
        assert rep["schema"] == "track_r/v7-a"
        assert rep["constants_tag"] == "provisional/v7-r0"
        assert rep["dataset"]["n_tasks"] == len(TASKS)
        assert rep["manifest"]["status"] == "unpinned"
        assert set(rep["arms"]) == {"flat_bm25", "fts5"}
        assert len(rep["per_task"]) == 2 * len(TASKS)

        ov = rep["arms"]["flat_bm25"]["overall"]
        for key in ("any@10", "any@20", "all@10", "ndcg@10", "mrr@10",
                    "zero_rate", "abstain_rate", "latency_ms", "n",
                    "n_gold"):
            assert key in ov, key
        assert ov["n_gold"] == 3           # q4 has no gold — excluded
        assert ov["any@10"] == 1.0         # all 3 answerable hit
        assert ov["all@10"] == 1.0         # q3 covers both refs

        cats = rep["arms"]["flat_bm25"]["categories"]
        assert set(cats) == {"single_hop", "temporal", "adversarial"}
        assert cats["single_hop"]["n"] == 2
        assert cats["adversarial"]["n_gold"] == 0

        # session granularity recorded when tasks carry session gold
        sess = rep["arms"]["flat_bm25"]["session"]
        assert sess and sess["overall"]["n_gold"] == 1
        assert sess["overall"]["any@10"] == 1.0

        # correct refusal: flat arms return q4 a spurious hit (no
        # verdict machinery) → refusal 0 unless empty
        assert rep["arms"]["flat_bm25"]["overall"]["correct_refusal"] \
            in (0.0, 1.0)

        att = rep["arms"]["flat_bm25"]["attribution"]["by_class"]
        assert att["delivered"] == 3 and att["unsupported"] == 1

    def test_determinism(self, corpus):
        r1 = run_track_r(corpus, [FlatBM25Arm()], k_list=(10,))
        r2 = run_track_r(corpus, [FlatBM25Arm()], k_list=(10,))
        assert _strip_volatile(r1) == _strip_volatile(r2)

    def test_manifest_pinning(self, corpus):
        rep = run_track_r(corpus, [FlatBM25Arm()], manifest="sha256:abc")
        assert rep["manifest"]["status"] == "pinned"
        assert rep["manifest"]["digest"] == "sha256:abc"

    def test_lanes_disabled_plumbing(self, corpus):
        flat = FlatBM25Arm()
        rep = run_track_r(corpus, [flat], lanes_disabled={"dense", "bogus"})
        info = rep["arms"]["flat_bm25"]["lanes_disabled"]
        assert info["requested"] == ["bogus", "dense"]
        assert info["applied"] == []
        assert set(info["unapplied"]) == {"dense", "bogus"}

    def test_markdown_renders(self, corpus):
        rep = run_track_r(corpus, [FlatBM25Arm()], k_list=(10, 20))
        md = render_markdown(rep)
        assert "any@10" in md and "flat_bm25" in md
        assert "Attribution" in md and "single_hop" in md

    def test_raw_dict_corpus(self):
        rep = run_track_r(
            {"items": ITEMS, "tasks": TASKS}, [FlatBM25Arm()])
        assert rep["arms"]["flat_bm25"]["overall"]["any@10"] == 1.0


# ---------------------------------------------------------------------------
# the verbatim arm — real Memory facade, no fixture store
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def verbatim_report():
    """One real Memory store + one Track R run, shared by the class —
    the facade path is real (add→drain→search), not a fixture store."""
    corpus = DictCorpus(ITEMS, TASKS, name="tiny", dataset_id="tiny-dict")
    arm = VerbatimArm(settle_timeout_s=60.0)
    try:
        rep = run_track_r(corpus, [arm], k_list=(10, 20), seed=0)
    finally:
        arm.close()
    return rep


class TestVerbatimArm:

    def test_ingest_settles(self, verbatim_report):
        ing = verbatim_report["arms"]["verbatim"]["ingest"]
        assert ing["indexed"] == len(ITEMS)
        assert ing["add_errors"] == 0
        assert ing["settle_state"] == "ready"
        assert ing["drain"]["failed"] == 0

    def test_real_retrieval_path(self, verbatim_report):
        ov = verbatim_report["arms"]["verbatim"]["overall"]
        # the real Memory path delivered every answerable task's gold
        assert ov["n_gold"] == 3
        assert ov["any@10"] == 1.0
        assert ov["all@10"] == 1.0
        # The unanswerable probe is correctly refused: under the V85
        # ship set (``LANES_V85`` = lex/fuzzy/dense/time) ``graph`` is
        # not in the default policy tuple, so q4's weak FTS seeds
        # produce no candidates at all — the verdict is ``insufficient``
        # (``answerability: no_evidence``) and the arm abstains.
        assert ov["correct_refusal"] == 1.0

    def test_verbatim_attribution(self, verbatim_report):
        att = verbatim_report["arms"]["verbatim"]["attribution"]["by_class"]
        assert att["delivered"] == 3
        assert att["unsupported"] == 1     # q4: unanswerable
        assert att["unattributed"] == 0

    def test_verbatim_session_granularity(self, verbatim_report):
        sess = verbatim_report["arms"]["verbatim"]["session"]
        assert sess and sess["overall"]["n_gold"] == 1
        assert sess["overall"]["any@10"] == 1.0

    def test_verbatim_lanes_disabled(self, corpus):
        arm = VerbatimArm()
        try:
            rep = run_track_r(corpus, [arm],
                              lanes_disabled={"dense", "nonlane"})
        finally:
            arm.close()
        info = rep["arms"]["verbatim"]["lanes_disabled"]
        assert info["applied"] == ["dense"]
        assert info["unapplied"] == ["nonlane"]
        assert info["applied_via"] == "config.v3.retrieval.<lane>=false"


# ---------------------------------------------------------------------------
# V8-13.01 / K74 — the verbatim_claims arm (admission.require_review=False)
# ---------------------------------------------------------------------------


class TestVerbatimClaimsArm:

    def test_k74_registry_resolves(self):
        arm = make_arm("verbatim_claims")
        assert isinstance(arm, VerbatimClaimsArm)
        assert isinstance(arm, VerbatimArm)
        assert arm_name(arm) == "verbatim_claims"
        assert arm._policy_overrides["admission"]["require_review"] is False
        arm.close()

    def test_k74_product_default_untouched(self):
        # Invariant 1 — the eval arm's pin never moves the product
        # default (config.py:29 / V8-13.01).
        from verbatim.config import AdmissionConfig, VerbatimConfig

        assert AdmissionConfig().require_review is True
        assert VerbatimConfig().admission.require_review is True
        # ... and the plain arm carries no admission override
        assert "admission" not in VerbatimArm()._policy_overrides

    def test_k74_caller_kwargs_still_apply(self):
        # registry kwargs pass through; the pin wins over a caller's
        # conflicting admission mapping (the arm IS the override).
        arm = make_arm(
            "verbatim_claims",
            settle_timeout_s=30.0,
            policy_overrides={
                "admission": {"require_review": True},
                "v3": {"retrieval": {"dense": False}},
            },
        )
        try:
            assert arm._policy_overrides["admission"]["require_review"] is False
            assert arm._policy_overrides["v3"]["retrieval"]["dense"] is False
        finally:
            arm.close()

    def test_k74_manifest_records_override(self, corpus):
        # K74 + K13: the run's arm row names the override, the built
        # store really runs require_review=False, and product default
        # stays True.
        arm = make_arm("verbatim_claims", settle_timeout_s=60.0)
        try:
            rep = run_track_r(corpus, [arm], k_list=(10,))
            cfg = rep["arms"]["verbatim_claims"]["ingest"]["config_overrides"]
            assert cfg["admission"]["require_review"] is False
            assert arm._mem._cfg.admission.require_review is False
            assert any("require_review" in n for n in
                       rep["arms"]["verbatim_claims"]["notes"])
            # behavioral difference is real: the blanket
            # ``review_required`` park is unreachable when the flag is
            # off (policy.py:838 — D8-18), so no review row may carry
            # that reason regardless of what the corpus proposed.
            with arm._mem._store.read() as conn:
                rr = conn.execute(
                    "SELECT COUNT(*) FROM reviews"
                    " WHERE proposed_effect_json"
                    " LIKE '%\"review_required\"%'"
                ).fetchone()[0]
            assert rr == 0
        finally:
            arm.close()

    def test_default_arm_records_no_admission_override(self, corpus):
        arm = VerbatimArm(settle_timeout_s=60.0)
        try:
            rep = run_track_r(corpus, [arm], k_list=(10,))
            assert arm._mem._cfg.admission.require_review is True
            # the contrast K74 measures: under the product default the
            # same corpus parks claims through ``review_required``
            with arm._mem._store.read() as conn:
                n_claims = conn.execute(
                    "SELECT COUNT(*) FROM claims"
                ).fetchone()[0]
                rr = conn.execute(
                    "SELECT COUNT(*) FROM reviews"
                    " WHERE proposed_effect_json"
                    " LIKE '%\"review_required\"%'"
                ).fetchone()[0]
            assert n_claims > 0     # fixture exercises the admission ladder
            assert rr > 0
        finally:
            arm.close()
        cfg = rep["arms"]["verbatim"]["ingest"]["config_overrides"]
        assert "admission" not in cfg


# ---------------------------------------------------------------------------
# V8-09.02 / K49-K50 — the as_of question-time anchor pass-through
# ---------------------------------------------------------------------------


class _StubResult:
    items: list = []
    coverage: dict = {}
    status: str = "ready"
    warnings: list = []


_UNSET = object()


class _StubMem:
    """``Memory.search`` with the V8-20.01 signature — records whether
    the caller actually sent ``as_of`` (sentinel, not a None default)."""

    def __init__(self):
        self.as_of_seen = []

    def search(self, query, *, limit=8, filters=None, after=None,
               consistency="session", ready_timeout_ms=None,
               timeout_ms=500, strict=False, retrieval="auto",
               as_of=_UNSET):
        self.as_of_seen.append(as_of)
        return _StubResult()


class _StubMemLegacy:
    """Pre-V8-20.01 signature — no ``as_of`` parameter at all."""

    def __init__(self):
        self.calls = []

    def search(self, query, *, limit=8, consistency="session",
               timeout_ms=500):
        self.calls.append((query, limit))
        return _StubResult()


def _armed_stub(mem):
    arm = VerbatimArm()
    arm._mem = mem
    return arm


class TestAsOfAnchor:

    def test_task_as_of_declared_keys(self):
        # metadata question_time_us → literal µs
        t = {"task_id": "q", "query": "x",
             "metadata": {"question_time_us": 1684540800000000,
                          "answer": "gold-stays-scorer-side"}}
        assert _task_as_of(t) == 1684540800000000
        # query_time ISO date → resolved µs (LongMemEval question_date)
        t = {"task_id": "q", "query": "x",
             "metadata": {"query_time": "2023-05-20"}}
        assert _task_as_of(t) == 1684540800000000
        # top-level question_date wins over nothing; RFC3339 resolves
        t = {"task_id": "q", "query": "x", "question_date": "2023-05-20"}
        assert _task_as_of(t) == 1684540800000000
        # no anchor → None keeps current behavior
        assert _task_as_of({"task_id": "q", "query": "x"}) is None
        # unresolvable declared value passes raw — the facade's own
        # VALIDATION judges it (V8-09.01/20.05), never a guessed clock
        t = {"task_id": "q", "query": "x",
             "metadata": {"question_time": "sometime later"}}
        assert _task_as_of(t) == "sometime later"

    def test_task_as_of_through_raw_view(self):
        # The scorer-side view wraps the corpus task — metadata rides
        # through ``raw`` (track_r.TaskView shape).
        class Raw:
            metadata = {"question_time_us": 1684540800000000}

        class View:
            task_id = "q"
            query = "x"
            raw = Raw()

        assert _task_as_of(View()) == 1684540800000000
        at = arm_task(View())
        assert at.as_of == 1684540800000000
        # gold surface stays closed — declared keys only
        assert not hasattr(at, "metadata")
        assert not hasattr(at, "answer")

    def test_group_anchors_final_session(self):
        items = [
            {"id": "a", "group_id": "conv", "when": "8 May, 2023"},
            {"id": "b", "group_id": "conv",
             "when": "1:56 pm on 19 May, 2023"},
            {"id": "c", "group_id": "conv", "when": "unparseable"},
            {"id": "d", "when": "9 May, 2023"},        # no group — skipped
        ]
        got = _group_anchors(items)
        assert got == {"conv": 1684504560000000}       # 19 May 13:56 UTC

    def test_as_of_sent_only_when_supplied(self):
        mem = _StubMem()
        arm = _armed_stub(mem)
        arm._group_as_of = {}
        out = arm.query(arm_task({
            "task_id": "q", "query": "x",
            "metadata": {"question_time_us": 1684540800000000}}), 10)
        assert out.status == "ready"
        assert mem.as_of_seen[0] == 1684540800000000
        assert out.diag["as_of"] == {
            "requested": 1684540800000000, "source": "task",
            "applied": True, "reason": None}
        assert "as_of_unsupported" not in out.warnings
        # no anchor → kwarg never sent (sentinel proves absence)
        out = arm.query(arm_task({"task_id": "q", "query": "x"}), 10)
        assert mem.as_of_seen[-1] is _UNSET
        assert out.diag["as_of"]["applied"] is False
        assert out.diag["as_of"]["requested"] is None
        # the unmeasured pool probe shares the measured call's anchor —
        # prefix-consistency requires the same reference time
        mem.as_of_seen.clear()
        arm.query(arm_task({
            "task_id": "q", "query": "x",
            "metadata": {"question_time_us": 1684540800000000}}), 10)
        assert mem.as_of_seen == [1684540800000000] * len(mem.as_of_seen)

    def test_as_of_group_fallback(self):
        mem = _StubMem()
        arm = _armed_stub(mem)
        arm._group_as_of = {"conv1": 1683504000000000}
        out = arm.query(arm_task(
            {"task_id": "q", "query": "x", "group_id": "conv1"}), 10)
        assert mem.as_of_seen[0] == 1683504000000000
        assert out.diag["as_of"]["source"] == "group_final_session"
        # a task-declared anchor beats the group fallback (V8-09.02
        # precedence — caller's clock wins)
        out = arm.query(arm_task({
            "task_id": "q", "query": "x", "group_id": "conv1",
            "metadata": {"question_time_us": 1684540800000000}}), 10)
        assert out.diag["as_of"]["source"] == "task"
        assert out.diag["as_of"]["requested"] == 1684540800000000

    def test_as_of_legacy_facade_reports_not_applied(self):
        # V8-20.01 not yet landed: the anchor is reported dropped, the
        # query still runs — never silent, never a crash.
        mem = _StubMemLegacy()
        arm = _armed_stub(mem)
        arm._group_as_of = {"conv1": 1683504000000000}
        out = arm.query(arm_task(
            {"task_id": "q", "query": "x", "group_id": "conv1"}), 10)
        assert out.status == "ready"
        assert out.diag["as_of"]["applied"] is False
        assert out.diag["as_of"]["requested"] == 1683504000000000
        assert "as_of_unsupported" in out.warnings
        # and nothing reached the legacy signature
        assert mem.calls[0] == ("x", 10)

    def test_as_of_policy_recorded(self, verbatim_report):
        ing = verbatim_report["arms"]["verbatim"]["ingest"]
        assert "as_of_policy" in ing
        assert ing["group_anchors"] == 0   # tiny corpus carries no group_id


# ---------------------------------------------------------------------------
# V8-15.03 attribution v2 (K07) + V8-15.02 split reporting (K06)
# ---------------------------------------------------------------------------


class TestAttributionV2:
    """K07 — a withheld question records ``abstain`` AND the underlying
    miss stage; gold delivered under a withheld verdict is its own
    label (D8-26); delivered gold carries provenance tags."""

    def _diag(self, **kw):
        base = dict(refs=[], surfaced=[], status="ok", abstained=False,
                    suppressed=0, omitted=0, error=None, k=10)
        base.update(kw)
        return base

    def test_k07_delivered_withheld_is_not_delivered(self):
        # Gold shipped inside the pack under an INSUFFICIENT verdict —
        # a trust/status defect, never a recall success.
        t = {"gold_evidence": ["g"], "answerable": True}
        d = self._diag(refs=["g"], surfaced=["g"],
                       status="insufficient", abstained=True)
        det = attribute_detail(t, d, indexed={"g"})
        assert det["attribution"] == "delivered_withheld"
        assert det["withheld"] is True
        assert det["gold_delivered"] is True
        assert det["miss_stage"] is None
        # the string API agrees with the record's label
        assert attribute(t, d, indexed={"g"}) == "delivered_withheld"
        # status-only withhold (flag unset, status in ABSTAIN_STATUSES)
        d2 = self._diag(refs=["g"], surfaced=["g"],
                        status="insufficient")
        assert attribute(t, d2, indexed={"g"}) == "delivered_withheld"
        # same refs, verdict not withheld → plain delivered
        d3 = self._diag(refs=["g"], surfaced=["g"])
        det3 = attribute_detail(t, d3, indexed={"g"})
        assert det3["attribution"] == "delivered"
        assert det3["withheld"] is False

    def test_k07_withheld_keeps_lane_miss(self):
        # withheld + gold indexed but never surfaced → abstain +
        # lane_miss (the scenario's literal pair)
        t = {"gold_evidence": ["g"], "answerable": True}
        d = self._diag(status="insufficient", abstained=True,
                       refs=["x"], surfaced=["x", "y"])
        det = attribute_detail(t, d, indexed={"g"})
        assert det["attribution"] == "abstain"
        assert det["withheld"] is True
        assert det["miss_stage"] == "lane_miss"

    def test_k07_withheld_keeps_rank_shift_and_packed_out(self):
        t = {"gold_evidence": ["g"], "answerable": True}
        d = self._diag(status="insufficient", refs=["x"],
                       surfaced=["x", "g"], k=1)
        det = attribute_detail(t, d, indexed={"g"})
        assert det["attribution"] == "abstain"
        assert det["miss_stage"] == "rank_shift"
        d = self._diag(status="insufficient", refs=["x"],
                       surfaced=["g", "x"], k=1)
        det = attribute_detail(t, d, indexed={"g"})
        assert det["attribution"] == "abstain"
        assert det["miss_stage"] == "packed_out"

    def test_k07_withheld_invisible_tail_stays_honest(self):
        # suppressed tail under a withhold: label abstain, stage
        # unattributed — never a guessed lane_miss
        t = {"gold_evidence": ["g"], "answerable": True}
        d = self._diag(status="insufficient", refs=["x"],
                       surfaced=["x"], suppressed=3)
        det = attribute_detail(t, d, indexed={"g"})
        assert det["attribution"] == "abstain"
        assert det["miss_stage"] == "unattributed"

    def test_k07_hard_stages_outrank_the_abstain_label(self):
        # v1 precedence preserved: arm error and never-ingested gold
        # keep their own labels even when the verdict withheld
        t = {"gold_evidence": ["g"], "answerable": True}
        det = attribute_detail(
            t, self._diag(status="insufficient", abstained=True,
                          error="boom", surfaced=None),
            indexed={"g"})
        assert det["attribution"] == "unattributed"
        assert det["withheld"] is True
        assert det["miss_stage"] == "unattributed"
        det = attribute_detail(
            t, self._diag(status="insufficient", abstained=True),
            indexed={"x", "y"})
        assert det["attribution"] == "unsupported"
        assert det["miss_stage"] == "unsupported"

    def test_k07_provenance_ctx_injected(self):
        # gold delivered via injection records ctx_injected (K07's
        # literal pair); non-gold units' tags are not collected
        t = {"gold_evidence": ["g"], "answerable": True}
        d = self._diag(
            refs=["g", "x"], surfaced=["g", "x"],
            item_explain={
                "g": {"ctx_from": "u-parent", "dense_slot": 1,
                      "rescue": True},
                "x": {"joint": True},
            })
        det = attribute_detail(t, d, indexed={"g"})
        assert det["attribution"] == "delivered"
        assert det["provenance"] == ["ctx_injected", "dense_slot",
                                     "rescue"]

    def test_k07_provenance_list_shape_and_mention(self):
        # list-shaped explain records resolve their ref under any of
        # the accepted keys; mention_interval maps to ``mention``
        t = {"gold_evidence": ["g1", "g2"], "answerable": True}
        d = self._diag(
            refs=["g1", "g2"], surfaced=["g1", "g2"],
            explain=[{"ref": "g1", "mention_interval": [3, 9]},
                     {"unit_id": "g2", "facet": "f", "joint": True}])
        det = attribute_detail(t, d, indexed={"g1", "g2"})
        assert det["provenance"] == ["facet", "joint", "mention"]

    def test_k07_aggregate_table_v2_views(self):
        recs = [
            {"category": "c1", "attribution": "delivered",
             "withheld": False, "miss_stage": None,
             "provenance": ["ctx_injected"]},
            {"category": "c1", "attribution": "delivered_withheld",
             "withheld": True, "miss_stage": None, "provenance": []},
            {"category": "c2", "attribution": "abstain",
             "withheld": True, "miss_stage": "lane_miss",
             "provenance": []},
        ]
        tab = aggregate_table(recs)
        assert tab["by_class"]["delivered_withheld"] == 1
        assert tab["withheld"] == 2
        assert tab["by_stage"]["lane_miss"] == 1
        assert tab["withheld_by_stage"]["lane_miss"] == 1
        assert tab["provenance"] == {"ctx_injected": 1}
        # delivered_withheld counts as a defect (miss), not a delivery
        assert tab["misses"] == 2
        # v1 views preserved
        assert tab["total"] == 3
        assert tab["by_category"]["c1"]["delivered"] == 1


# ---------------------------------------------------------------------------
# the split — a scripted verdict arm over cat 1–4 + cat-5 tasks
# ---------------------------------------------------------------------------

SPLIT_ITEMS = [
    {"id": "p1", "text": "premise turn about racing bikes",
     "speaker": "A", "session_id": "s1"},
    {"id": "p2", "text": "premise turn about pottery",
     "speaker": "B", "session_id": "s1"},
    {"id": "e1", "text": "evidence about a rescue dog",
     "speaker": "A", "session_id": "s2"},
    {"id": "e2", "text": "evidence about watercolor painting",
     "speaker": "B", "session_id": "s2"},
    {"id": "e3", "text": "evidence about lunch menus",
     "speaker": "A", "session_id": "s3"},
    {"id": "x1", "text": "filler about trains and timestamps",
     "speaker": "B", "session_id": "s3"},
]

SPLIT_TASKS = [
    {"task_id": "a1", "query": "dog", "category": "single_hop",
     "gold_evidence": ["e1"], "metadata": {"category_id": 4}},
    {"task_id": "a2", "query": "paint", "category": "temporal",
     "gold_evidence": ["e2"], "metadata": {"category_id": 2}},
    {"task_id": "a3", "query": "lunch", "category": "single_hop",
     "gold_evidence": ["e3"], "metadata": {"category_id": 4}},
    {"task_id": "c1", "query": "bikes", "category": "adversarial",
     "answerable": False, "gold_evidence": ["p1"],
     "metadata": {"category_id": 5}},
    {"task_id": "c2", "query": "pottery", "category": "adversarial",
     "answerable": False, "gold_evidence": ["p2"],
     "metadata": {"category_id": 5}},
]

# a2's gold sits at rank 25 of the surfaced pool — below the k=20 cut
_A2_SURFACE = ["u%02d" % i for i in range(24)] + ["e2"]

SPLIT_SCRIPT = {
    "a1": {"refs": ["e1", "x1"], "surfaced": ["e1", "x1"]},
    # answerable, withheld, gold surfaced below the cut → abstain +
    # rank_shift
    "a2": {"refs": ["x1"], "surfaced": _A2_SURFACE,
           "status": "insufficient", "abstained": True,
           "text": "filler about trains and timestamps"},
    # answerable, gold shipped at rank 2 under an insufficient verdict
    # → delivered_withheld (D8-26), tokens_first_gold = x1's estimate
    "a3": {"refs": ["x1", "e3"], "surfaced": ["x1", "e3"],
           "status": "insufficient", "abstained": True},
    # cat-5: premise turn delivered AND verdict refused → both facts
    "c1": {"refs": ["p1"], "surfaced": ["p1"],
           "status": "insufficient", "abstained": True},
    # cat-5: premise turn missed and no refusal → spurious delivery
    "c2": {"refs": ["x1"], "surfaced": ["x1"]},
}


class _VerdictArm:
    """Scripted arm — per-task refs/status/abstained/surfaced so the
    split rows and attribution classes are hand-computable."""

    name = "verdict_script"

    def __init__(self, script):
        self._script = dict(script)
        self._indexed = set()

    def ingest(self, corpus):
        items = corpus_items(corpus)
        self._indexed = {
            item_ref(it, i) for i, it in enumerate(items)
        }
        return {"indexed": len(self._indexed)}

    def indexed_refs(self):
        return set(self._indexed)

    def query(self, task, k):
        spec = self._script.get(task.task_id, {})
        refs = list(spec.get("refs", ()))[: int(k)]
        return QueryOutcome(
            refs=refs,
            surfaced=spec.get("surfaced", refs),
            status=spec.get("status", "ok"),
            abstained=spec.get("abstained", False),
            n_items=len(refs),
            delivered_text=spec.get("text", ""),
            diag=dict(spec.get("diag", {})),
            k=int(k),
        )


@pytest.fixture()
def split_corpus():
    return DictCorpus(SPLIT_ITEMS, SPLIT_TASKS, name="split",
                      dataset_id="split-dict")


@pytest.fixture()
def split_report(split_corpus):
    return run_track_r(split_corpus, [_VerdictArm(SPLIT_SCRIPT)],
                       k_list=(10, 20), seed=0)


class TestTrackRSplitV8:
    """K06 — answerable / cat5_premise / all row groups with the full
    V8-15.02 column set; cat-5 premise delivery and correct refusal on
    separate rows; attribution keeps abstain + stage + provenance."""

    def test_k06_split_groups_present(self, split_report):
        sp = split_report["arms"]["verdict_script"]["split"]
        assert set(sp) == {"answerable", "cat5_premise", "all"}
        assert sp["answerable"]["n"] == 3        # a1, a2, a3
        assert sp["cat5_premise"]["n"] == 2      # c1, c2
        assert sp["all"]["n"] == 5
        # cat-5 row carries its LoCoMo id + name
        assert sp["cat5_premise"]["cat_label"] == "5 adversarial"

    def test_k06_split_metrics_hand_computed(self, split_report):
        sp = split_report["arms"]["verdict_script"]["split"]
        ans, cat5, allg = (
            sp["answerable"], sp["cat5_premise"], sp["all"])
        # answerable any@10: a1 + a3 hit, a2 missed → 2/3
        assert ans["any@10"] == pytest.approx(2 / 3)
        assert ans["n_gold"] == 3
        # false-insufficient (V8-22.03): a2 + a3 withheld of 3 → 2/3
        assert ans["false_insufficient"] == pytest.approx(2 / 3)
        assert ans["correct_refusal"] is None    # no unanswerable here
        # cat-5 premise delivery any@10 (V8-22.02): c1 shipped p1 → 1/2
        assert cat5["any@10"] == pytest.approx(0.5)
        assert cat5["n_gold"] == 2
        # correct refusal (V8-22.04): c1 withheld of 2 cat-5 → 0.5 —
        # a DIFFERENT number from premise any@10, on its own row
        assert cat5["correct_refusal"] == pytest.approx(0.5)
        assert cat5["false_insufficient"] is None
        assert allg["any@10"] == pytest.approx(3 / 5)
        assert allg["correct_refusal"] == pytest.approx(0.5)
        assert allg["false_insufficient"] == pytest.approx(2 / 3)

    def test_k06_markdown_row_groups(self, split_report):
        md = render_markdown(split_report)
        assert "### Answerability split" in md
        assert "| verdict_script | answerable | 1–4 |" in md
        assert "| verdict_script | cat5_premise | 5 adversarial |" in md
        # premise delivery and refusal are different numbers on
        # different rows
        assert "| verdict_script | cat5_premise refusal " \
            "| 5 adversarial |" in md
        assert "| verdict_script | all | all |" in md
        # every V8-15.02 column renders
        header = next(
            ln for ln in md.splitlines() if "false_ins" in ln
        )
        for col in ("mean\\|G\\|", "any@10", "any@20", "all@10",
                    "all@20", "prop@10", "prop@20", "ndcg@10", "mrr@10",
                    "zero", "false_ins", "corr_ref", "tok→G", "tok→G90",
                    "p50ms", "p95ms"):
            assert col in header, col
        # the refusal row carries its rate under corr_ref
        row = next(
            ln for ln in md.splitlines()
            if "cat5_premise refusal" in ln
        )
        cells = [c.strip() for c in row.split("|")]
        assert "0.500" in cells

    def test_k06_no_cat5_corpus_still_prints_group(self):
        items = [{"id": "e1", "text": "only evidence"}]
        tasks = [{"task_id": "a1", "query": "e",
                  "category": "single_hop", "gold_evidence": ["e1"]}]
        corpus = DictCorpus(items, tasks, name="n", dataset_id="n")
        rep = run_track_r(corpus, [_VerdictArm({"a1": {"refs": ["e1"]}})])
        sp = rep["arms"]["verdict_script"]["split"]
        assert sp["cat5_premise"]["n"] == 0
        assert sp["answerable"]["n"] == 1
        md = render_markdown(rep)
        assert "| verdict_script | cat5_premise " in md

    def test_k06_tokens_to_first_gold(self, split_report):
        per_task = {
            r["task_id"]: r
            for r in split_report["per_task"]
            if r["arm"] == "verdict_script"
        }
        # a1: gold first → zero tokens preceding
        assert per_task["a1"]["tokens_first_gold"] == 0.0
        # a2/c2: gold never delivered → null, counted separately
        assert per_task["a2"]["tokens_first_gold"] is None
        assert per_task["c2"]["tokens_first_gold"] is None
        # a3: gold at rank 2 → x1's item-document estimate precedes it
        x1_doc = next(i for i in SPLIT_ITEMS if i["id"] == "x1")
        expect = float(M.estimate_tokens(item_document(x1_doc)))
        assert per_task["a3"]["tokens_first_gold"] == pytest.approx(
            expect)
        # aggregate: median over delivered only; undelivered counted
        tfg = split_report["arms"]["verdict_script"]["split"]["all"][
            "tokens_first_gold"]
        assert tfg["n_gold_delivered"] == 3      # a1, a3, c1
        assert tfg["n_gold_undelivered"] == 2    # a2, c2
        assert tfg["median"] == 0.0              # [0, 0, x1_est]
        assert tfg["p90"] == pytest.approx(expect)
        basis = split_report["arms"]["verdict_script"][
            "tokens_first_gold_basis"]
        assert basis == ["corpus_item_document"]

    def test_k06_unit_texts_override_basis(self, split_corpus):
        # an arm supplying per-unit delivered texts measures the real
        # bytes, not the corpus-document estimate
        script = {
            "a1": {"refs": ["x1", "e1"], "surfaced": ["x1", "e1"],
                   "diag": {"unit_texts": ["alpha beta gamma",
                                          "delta"]}},
        }
        rep = run_track_r(split_corpus, [_VerdictArm(script)],
                          k_list=(10,))
        pt = next(r for r in rep["per_task"] if r["task_id"] == "a1")
        assert pt["tokens_first_gold"] == pytest.approx(
            float(M.estimate_tokens("alpha beta gamma")))
        assert rep["arms"]["verdict_script"]["tokens_first_gold_basis"] \
            == ["delivered_unit_text"]

    def test_k07_report_attribution_split(self, split_report):
        att = split_report["arms"]["verdict_script"]["attribution"]
        bc = att["by_class"]
        # a1 delivered; a3 gold shipped under withhold (D8-26); a2
        # abstain; c1/c2 unanswerable → unsupported (their withhold is
        # a correct refusal, kept out of the miss classes)
        assert bc["delivered"] == 1
        assert bc["delivered_withheld"] == 1
        assert bc["abstain"] == 1
        assert bc["unsupported"] == 2
        assert att["withheld"] == 3          # a2, a3, c1
        # a2's stage survives beneath the abstain label
        assert att["by_stage"]["rank_shift"] == 1
        assert att["withheld_by_stage"]["rank_shift"] == 1
        # per-task records carry the v2 fields
        pt = {r["task_id"]: r for r in split_report["per_task"]}
        assert pt["a2"]["attribution"] == "abstain"
        assert pt["a2"]["miss_stage"] == "rank_shift"
        assert pt["a3"]["attribution"] == "delivered_withheld"
        assert pt["a3"]["miss_stage"] is None
        assert pt["c1"]["attribution"] == "unsupported"

    def test_k07_markdown_stage_table(self, split_report):
        md = render_markdown(split_report)
        assert "delivered_withheld" in md
        assert "Withheld questions — underlying miss stage" in md
        stage_table = md.split(
            "Withheld questions — underlying miss stage"
        )[1]
        line = next(
            ln for ln in stage_table.splitlines()
            if ln.startswith("| verdict_script")
        )
        cells = [c.strip() for c in line.split("|")]
        # STAGES order: lane_miss rank_shift packed_out unsupported
        # unattributed → cells: "", arm, lane_miss, rank_shift, ...
        assert cells[3] == "1"   # rank_shift holds a2's withheld miss

    def test_k07_per_corpus_provenance_renders_when_present(
            self, split_corpus):
        script = {
            "a1": {"refs": ["e1"], "surfaced": ["e1"],
                   "diag": {"item_explain": {
                       "e1": {"ctx_from": "p", "rescue": True}}}},
        }
        rep = run_track_r(split_corpus, [_VerdictArm(script)],
                          k_list=(10,))
        att = rep["arms"]["verdict_script"]["attribution"]
        assert att["provenance"] == {"ctx_injected": 1, "rescue": 1}
        md = render_markdown(rep)
        assert "Delivered-gold provenance" in md
        assert "ctx_injected" in md
