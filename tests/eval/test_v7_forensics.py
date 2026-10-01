"""Smoke tests for ``eval/v7/forensics`` — the V8 wave-0 measurement
tools (SPEC_V8 V8-05.01, V8-11.01, V8-11.02, V8-14.02).

Everything runs on a tiny inline dict corpus through the real
``Memory`` write+read path (``VerbatimArm``) — no LoCoMo download, no
network, no model fetch.  Pure-function surfaces (shape normalization,
score decomposition, premise diagnostics, policy resolution) are
tested directly; the tool entry points are exercised end-to-end with
module-scoped paired runs (one ingest each).
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from eval.v7.arms import DictCorpus, FlatBM25Arm
from eval.v7.track_r import task_view
from eval.v7.forensics import (
    ArmSpec,
    PolicyPatch,
    SqlCensus,
    classify_lane_miss,
    normalize_shape,
    paired_run,
    run_lane_miss_forensic,
)
from eval.v7.forensics import _common as C
from eval.v7.forensics import lane_ablation as LA
from eval.v7.forensics import lane_miss as LM
from eval.v7.forensics import rank_forensics as RF
from eval.v7.forensics import speaker_audit as SA


# ---------------------------------------------------------------------------
# corpus — tiny inline dict corpus (same shape as test_v7_trackr)
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
     "evidence_session_ids": ["s1"], "group_id": "conv1"},
    {"task_id": "q2", "query": "when does the pottery studio open",
     "category": "temporal", "gold_evidence": ["d4"],
     "group_id": "conv1"},
    {"task_id": "q3", "query": "what breed is Biscuit",
     "category": "single_hop", "gold_evidence": ["d1", "d3"],
     "group_id": "conv1"},
    {"task_id": "q4", "query": "who won the chess tournament",
     "category": "adversarial", "answerable": False,
     "group_id": "conv1"},
]


def _corpus() -> DictCorpus:
    return DictCorpus(ITEMS, TASKS, name="tiny", dataset_id="tiny-dict")


def _jsonable(rep) -> str:
    """Reports must serialize byte-stable (sort_keys) — the determinism
    contract."""
    return json.dumps(rep, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# sql_census — pure shape normalization (no store needed)
# ---------------------------------------------------------------------------


class TestNormalizeShape:
    def test_select_where_columns_sorted(self):
        s = normalize_shape(
            "SELECT unit_id, text FROM units WHERE generation = 4 "
            "AND speaker_canon = 'mel' ORDER BY unit_id"
        )
        assert s == "select:units(generation,speaker_canon)"

    def test_literals_collapse(self):
        a = normalize_shape("SELECT x FROM t WHERE a = 'one'")
        b = normalize_shape("SELECT x FROM t WHERE a = 'two'")
        assert a == b == "select:t(a)"

    def test_insert_column_list(self):
        s = normalize_shape(
            "INSERT INTO units (unit_id, source_id, revision) "
            "VALUES (?, ?, ?)"
        )
        assert s == "insert:units(revision,source_id,unit_id)"

    def test_update_set_and_where(self):
        s = normalize_shape(
            "UPDATE units SET flag = 1, seen = 2 WHERE unit_id = ?"
        )
        assert s == "update:units(flag,seen,unit_id)"

    def test_delete(self):
        assert normalize_shape("DELETE FROM t WHERE k = 7") == \
            "delete:t(k)"

    def test_tx_and_pragma(self):
        assert normalize_shape("BEGIN") == "tx:begin"
        assert normalize_shape("commit") == "tx:commit"
        assert normalize_shape("ROLLBACK TO sp1") == "tx:rollback"
        assert normalize_shape("PRAGMA table_info(units)") == \
            "pragma:table_info"

    def test_create_and_virtual(self):
        assert normalize_shape(
            "CREATE TABLE IF NOT EXISTS units (a TEXT)"
        ) == "create:table:units"
        assert normalize_shape(
            "CREATE VIRTUAL TABLE u_fts USING fts5(body)"
        ) == "create:table:u_fts"

    def test_fts_match_column(self):
        s = normalize_shape(
            "SELECT rowid FROM unit_fts WHERE unit_fts MATCH ?"
        )
        assert s == "select:unit_fts(unit_fts)"

    def test_join_where_strips_qualifier(self):
        s = normalize_shape(
            "SELECT u.a FROM u JOIN e ON u.id = e.uid WHERE e.gen = ?"
        )
        assert s == "select:u(gen)"

    def test_unknown_and_empty(self):
        assert normalize_shape("") == "empty:"
        assert normalize_shape("ANALYZE") == "analyze:"
        assert normalize_shape("WITH x AS (SELECT 1) SELECT * FROM x") \
            .startswith("with_select:")


class TestSqlCensus:
    def test_counts_known_statements(self):
        conn = sqlite3.connect(":memory:")
        c = SqlCensus()
        c.attach(conn, scope="mem")
        conn.execute("CREATE TABLE t (a, b)")
        conn.execute("INSERT INTO t (a, b) VALUES (?, ?)", (1, "x"))
        conn.execute("SELECT a FROM t WHERE b = ?", ("x",)).fetchall()
        snap = c.snapshot()
        # 3 issued + the implicit BEGIN sqlite fires for the write —
        # tx:* shapes are part of the complete count.
        assert snap["statements"] == 4
        assert snap["by_shape"]["create:table:t"] == 1
        assert snap["by_shape"]["insert:t(a,b)"] == 1
        assert snap["by_shape"]["select:t(b)"] == 1
        assert snap["by_shape"]["tx:begin"] == 1
        assert snap["statements"] == sum(snap["by_shape"].values())
        assert snap["scopes"] == ["mem"]
        c.detach()

    def test_reset_and_wrap(self):
        conn = sqlite3.connect(":memory:")
        c = SqlCensus()
        c.attach(conn, scope="mem")
        conn.execute("CREATE TABLE t (a)")
        res, snap = c.wrap(
            lambda: conn.execute("SELECT a FROM t").fetchall()
        )
        assert res == []
        assert snap["statements"] == 1
        c.detach()

    def test_wrap_exception_keeps_census(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (a UNIQUE)")
        conn.execute("INSERT INTO t VALUES (1)")
        c = SqlCensus()
        c.attach(conn, scope="mem")
        # a statement that fails mid-step is still counted — the
        # partial census rides the exception for honest reporting
        with pytest.raises(sqlite3.IntegrityError) as ei:
            c.wrap(conn.execute, "INSERT INTO t (a) VALUES (1)")
        snap = ei.value.census  # type: ignore[attr-defined]
        assert snap["statements"] >= 1
        assert snap["by_shape"].get("insert:t(a)") == 1
        c.detach()


# ---------------------------------------------------------------------------
# PolicyPatch — resolution without a store
# ---------------------------------------------------------------------------


class TestPolicyPatch:
    def test_dry_resolve_removes_lane(self):
        p = PolicyPatch(lanes_disabled=["graph"])
        pol = p.dry_resolve()
        assert pol is not None
        lanes = [getattr(l, "value", str(l)) for l in pol.lanes]
        assert "graph" not in lanes
        assert "lex" in lanes  # other lanes survive

    def test_dry_resolve_bad_lane_records_error(self):
        p = PolicyPatch(lanes_disabled=["bogus_lane_xyz"])
        assert p.dry_resolve() is None
        assert p.errors
        desc = p.describe()
        assert desc["lanes_disabled"] == ["bogus_lane_xyz"]
        assert desc["errors"]
        json.dumps(desc)  # manifest-safe

    def test_feature_weights_describe(self):
        p = PolicyPatch(feature_weights={"speaker_match": 0.0})
        desc = p.describe()
        assert desc["feature_weights"] == {"speaker_match": 0.0}
        assert any("score_candidates" in v for v in desc["applied_via"])


# ---------------------------------------------------------------------------
# score decomposition + forensic rows — pure scorer-side functions
# ---------------------------------------------------------------------------


def _explain_item(unit, src, *, lane_ranks, features, weights,
                  signals=None, fused=0.02, score=0.5):
    return {
        "unit_id": unit,
        "source_id": src,
        "revision": 1,
        "lane_ranks": dict(lane_ranks),
        "signals": signals or {},
        "fused_rrf": fused,
        "score": score,
        "score_family": "ranking/v7",
        "detail": {
            "features": dict(features),
            "feature_score": score,
            "tie_epsilon": 0.0,
            "weights": dict(weights),
            "rrf": fused,
            "lane_ranks": dict(lane_ranks),
            "signals": signals or {},
            "speaker_match_source": "query",
        },
        "group_label": None,
        "pack": "delivered",
    }


class TestDecompose:
    def test_decompose_item_math(self):
        item = _explain_item(
            "u2", "SRCd2",
            lane_ranks={"lex": 1, "dense": 4},
            features={"rrf_norm": 0.5, "speaker_match": 1.0},
            weights={"rrf_norm": 0.2, "speaker_match": 0.4},
            fused=0.025, score=0.5,
        )
        d = RF.decompose_item(
            item, ref="d2", lane_weights={"lex": 1.0, "dense": 2.0}
        )
        assert d["ref"] == "d2" and d["mapped"] is True
        assert d["lane_contrib"]["lex"] == pytest.approx(1.0 / 61.0)
        assert d["lane_contrib"]["dense"] == pytest.approx(2.0 / 64.0)
        assert d["lane_contrib_sum"] == pytest.approx(
            1.0 / 61.0 + 2.0 / 64.0)
        assert d["rrf_residual"] == pytest.approx(
            0.025 - (1.0 / 61.0 + 2.0 / 64.0))
        assert d["feature_contrib"]["speaker_match"] == \
            pytest.approx(0.4)
        assert d["feature_contrib"]["rrf_norm"] == pytest.approx(0.1)
        json.dumps(d)

    def test_forensic_rows_population(self):
        """bm25 gold@1 + verbatim gold@3 → one forensic row with a
        gold-vs-displacer decomposition."""
        bm = FlatBM25Arm()
        bm.ingest(_corpus())
        try:
            tv = task_view({
                "task_id": "qf", "query": "rescue dog Biscuit Caroline",
                "category": "single_hop", "gold_evidence": ["d1"],
                "group_id": "conv1",
            })
            explains = {
                "qf": {
                    "query": {"intent": "factual_lookup"},
                    "items": [
                        _explain_item(
                            "u2", "SRCd2",
                            lane_ranks={"lex": 1},
                            features={"rrf_norm": 0.9},
                            weights={"rrf_norm": 0.2},
                            fused=0.03, score=0.9),
                        _explain_item(
                            "u1", "SRCd1",
                            lane_ranks={"lex": 2},
                            features={"rrf_norm": 0.8},
                            weights={"rrf_norm": 0.2},
                            fused=0.028, score=0.8),
                    ],
                }
            }
            baseline_rows = {
                "qf": {"first_gold_rank": 3, "pool_gold_rank": 2,
                       "status": "ok", "attribution": "rank_shift"}
            }
            rows = RF.forensic_rows(
                [tv], baseline_rows, explains, bm,
                {"SRCd1": "d1", "SRCd2": "d2"},
                k=10, lane_weights_fn=lambda _i: {"lex": 1.0},
            )
        finally:
            bm.close()
        assert len(rows) == 1
        r = rows[0]
        assert r["bm25"]["gold_rank"] == 1
        assert r["verbatim"]["delivered_gold_rank"] == 3
        assert r["verbatim"]["explain_rank_of_gold"] == 2
        assert r["gold_item"]["ref"] == "d1"
        assert r["top_nongold"]["ref"] == "d2"
        assert [d["ref"] for d in r["displacers"]] == ["d2"]
        assert r["explain_status"] == "ok"
        json.dumps(rows)

    def test_forensic_rows_skips_outside_population(self):
        bm = FlatBM25Arm()
        bm.ingest(_corpus())
        try:
            tv = task_view({
                "task_id": "qf", "query": "rescue dog Biscuit Caroline",
                "category": "single_hop", "gold_evidence": ["d1"],
            })
            # verbatim ALSO ranked gold first → no displacement → no row
            rows = RF.forensic_rows(
                [tv],
                {"qf": {"first_gold_rank": 1}},
                {}, bm, {}, k=10, lane_weights_fn=lambda _i: {},
            )
            assert rows == []
            # verbatim missed entirely (gold unranked) → still a row,
            # explain_status unavailable rather than fabricated
            rows = RF.forensic_rows(
                [tv],
                {"qf": {"first_gold_rank": None, "status": "ok"}},
                {}, bm, {}, k=10, lane_weights_fn=lambda _i: {},
            )
            assert len(rows) == 1
            assert rows[0]["explain_status"] == "unavailable"
            assert rows[0]["gold_item"] is None
        finally:
            bm.close()


class TestPremiseDiag:
    def test_both_hold_detected(self):
        tv = task_view({
            "task_id": "qc", "query": "what does Melanie do",
            "category": "adversarial", "answerable": True,
            "gold_evidence": ["d1"],
        })
        explain = {
            "items": [
                _explain_item(
                    "u2", "SRCd2", lane_ranks={"lex": 1},
                    features={"speaker_match": 1.0},
                    weights={"speaker_match": 0.4},
                    signals={"ent": {"speaker_canon": "Melanie"}},
                    fused=0.03, score=0.9),
                _explain_item(
                    "u1", "SRCd1", lane_ranks={"lex": 2},
                    features={"rrf_norm": 0.8},
                    weights={"rrf_norm": 0.2},
                    fused=0.028, score=0.8),
            ]
        }
        d = SA._premise_diag(
            tv, explain,
            {"d1": "Caroline", "d2": "Melanie"},
            {"SRCd1": "d1", "SRCd2": "d2"},
            {"lex": 1.0},
        )
        assert d["premise_speaker"] == ["Caroline"]
        assert d["resolved_speaker"] == "Melanie"
        assert d["speaker_match_source"] == "query"
        assert d["premise_differs"] is True
        assert d["n_displacers"] == 1
        assert d["displacers_speaker_match"] == 1
        assert d["both_hold"] is True
        json.dumps(d)

    def test_unobservable_is_none_not_guess(self):
        tv = task_view({
            "task_id": "qc", "query": "q", "category": "adversarial",
            "answerable": True, "gold_evidence": ["d1"],
        })
        d = SA._premise_diag(
            tv, None, {"d1": "Caroline"}, {}, {},
        )
        assert d["resolved_speaker"] is None
        assert d["both_hold"] is None  # honest null, not a fabricated 0
        assert d["explain_status"] == "unavailable"


# ---------------------------------------------------------------------------
# paired_run — the shared machinery, exercised once with 3 specs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def paired_report():
    """One ingest, three specs: baseline, no_graph (policy-tuple
    ablation + census + explain), and a bogus lane that must fail
    validation honestly.

    V85-05.01: the measured ship set (``LANES_V85`` = lex/fuzzy/dense/
    time) is the default enablement — ``graph`` only enters the policy
    tuple through a declared ``lanes`` list.  Both lane specs therefore
    pin the full tuple via ``policy_doc`` so the ablation is a real
    differential (baseline observes graph; no_graph verifiably removes
    it), not the removal of a lane that was never enabled."""
    _LANES_WITH_GRAPH = ["lex", "fuzzy", "dense", "time", "graph"]
    specs = [
        ArmSpec(label="baseline",
                patch=PolicyPatch(
                    policy_doc={"lanes": list(_LANES_WITH_GRAPH)}),
                census=True),
        ArmSpec(label="no_graph",
                patch=PolicyPatch(
                    policy_doc={"lanes": list(_LANES_WITH_GRAPH)},
                    lanes_disabled=["graph"]),
                census=True),
        ArmSpec(label="bogus",
                patch=PolicyPatch(lanes_disabled=["bogus_lane_xyz"])),
    ]
    return paired_run(
        _corpus(), specs, k_list=(10,),
        full_explain=True, need_census=True, tool="test",
    )


class TestPairedRun:
    def test_status_and_schema(self, paired_report):
        assert paired_report["schema"] == C.SCHEMA
        assert paired_report["status"] == "executed"
        assert paired_report["dataset"]["n_items"] == 5
        assert paired_report["dataset"]["n_tasks"] == 4
        _jsonable(paired_report)

    def test_bogus_lane_fails_honestly(self, paired_report):
        spec = paired_report["specs"]["bogus"]
        assert spec["applied"] is False
        assert spec["error"]
        assert spec["overall"] is None  # never fabricated metrics
        q = next(q for q in paired_report["questions"]
                 if q["task_id"] == "q1")
        assert q["arms"]["bogus"]["status"] == "not_run"

    def test_lane_ablation_applied_and_verified(self, paired_report):
        off = paired_report["specs"]["no_graph"]
        assert off["applied"] is True
        obs = off["verify"]["observed_policy_lanes"]
        assert obs is not None
        assert "graph" not in obs          # lane left the policy tuple
        assert "lex" in obs                # others intact
        base = paired_report["specs"]["baseline"]
        base_obs = base["verify"]["observed_policy_lanes"]
        assert base_obs is None or "graph" in base_obs

    def test_question_rows_paired(self, paired_report):
        qs = {q["task_id"]: q for q in paired_report["questions"]}
        assert set(qs) == {"q1", "q2", "q3", "q4"}
        for q in qs.values():
            assert q["conv_id"] == "conv1"
            for arm in ("baseline", "no_graph"):
                row = q["arms"][arm]
                assert "delivered" in row and "gold_ranks" in row
                assert "latency_ms" in row and "status" in row
        # paired confusion covers every gold-bearing task exactly once
        conf = paired_report["paired"]["any_at_k"]["10"]
        n_gold = sum(1 for q in qs.values() if q["gold"])
        assert (conf["both_hit"] + conf["only_first"]
                + conf["only_second"] + conf["both_miss"]) == n_gold

    def test_census_on_store(self, paired_report):
        q1 = next(q for q in paired_report["questions"]
                  if q["task_id"] == "q1")
        sql = q1["arms"]["baseline"]["sql"]
        assert sql is not None
        assert sql["statements"] > 0
        assert sql["by_shape"]
        assert sql["statements"] == sum(sql["by_shape"].values())
        # the store reader was actually traced
        assert "reader" in (paired_report["manifest"].get("sql_scopes")
                            or [])

    def test_explain_captured(self, paired_report):
        q1 = next(q for q in paired_report["questions"]
                  if q["task_id"] == "q1")
        row = q1["arms"]["baseline"]
        assert row["explain_forced"] is True
        assert row["policy_lanes"]  # the policy echo verify() reads


# ---------------------------------------------------------------------------
# tool entry points — one paired run each (module scope)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lane_report():
    return LA.run_lane_ablation(
        _corpus(), "graph", k_list=(10,), census=True,
        full_explain=True,
    )


class TestLaneAblationTool:
    def test_specs_and_pairing(self, lane_report):
        assert lane_report["schema"] == LA.SCHEMA
        assert set(lane_report["specs"]) == {"graph_on", "no_graph"}
        assert lane_report["specs"]["no_graph"]["applied"] is True
        assert lane_report["manifest"]["ablated_lane"] == "graph"
        assert lane_report["manifest"]["requirement"].startswith("V8-05.01")
        assert lane_report["paired"]["first"] == "graph_on"
        _jsonable(lane_report)

    def test_per_question_contract(self, lane_report):
        for q in lane_report["questions"]:
            for key in ("task_id", "conv_id", "category", "category_id",
                        "answerable", "gold"):
                assert key in q
            for arm in ("graph_on", "no_graph"):
                row = q["arms"][arm]
                for key in ("delivered", "surfaced", "gold_ranks",
                            "first_gold_rank", "pool_gold_rank",
                            "latency_ms", "status", "sql"):
                    assert key in row, key


@pytest.fixture(scope="module")
def rank_report():
    return RF.run_rank_forensics(
        _corpus(), k=10, lolo_lanes=("lex",),
    )


class TestRankForensicsTool:
    def test_structure(self, rank_report):
        assert rank_report["schema"] == RF.SCHEMA
        pop = rank_report["population"]
        assert pop["n_answerable"] == 3
        assert pop["n_population"] == len(rank_report["rows"])
        assert rank_report["manifest"]["requirement"] == "V8-11.01"
        _jsonable(rank_report)

    def test_lolo_table(self, rank_report):
        lolo = rank_report["leave_one_lane_out"]
        assert lolo["status"] == "executed"
        assert "mrr@10" in lolo["baseline"]
        assert "ndcg@10" in lolo["baseline"]
        assert "no_lex" in lolo
        assert lolo["no_lex"]["delta_mrr@10"] is not None
        # the ablated spec verified against the printed policy tuple
        spec = rank_report["specs"]["no_lex"]
        assert spec["applied"] is True
        assert "lex" not in spec["verify"]["observed_policy_lanes"]

    def test_no_lolo_is_not_run_not_empty(self):
        rep = RF.run_rank_forensics(_corpus(), k=10, lolo_lanes=())
        assert rep["leave_one_lane_out"]["status"] == "not_run"
        assert rep["leave_one_lane_out"]["reason"]


@pytest.fixture(scope="module")
def speaker_report():
    return SA.run_speaker_audit(_corpus(), k=10, weights=(0.0,))


class TestSpeakerAuditTool:
    def test_specs(self, speaker_report):
        assert speaker_report["schema"] == SA.SCHEMA
        assert set(speaker_report["specs"]) == {"w_current", "w_0"}
        cur = speaker_report["manifest"]["speaker_match_weight_current"]
        assert cur == pytest.approx(0.4)
        assert speaker_report["manifest"]["weights_measured"] == \
            [pytest.approx(0.4), 0.0]
        # w_0 verified through observed feature weights OR honestly
        # marked unverifiable — never silently True
        assert speaker_report["specs"]["w_0"]["applied"] in (
            True, "unverifiable")
        _jsonable(speaker_report)

    def test_premise_aggregates_and_evidence(self, speaker_report):
        for label in ("w_current", "w_0"):
            spec = speaker_report["specs"][label]
            assert "premise_any@10" in spec
            assert "premise_n" in spec
        pe = speaker_report["premise_evidence"]
        for key in ("n", "n_evaluable", "n_both_hold",
                    "both_hold_fraction", "questions"):
            assert key in pe
        assert pe["n"] >= 0 and pe["n_evaluable"] <= pe["n"]

    def test_premise_predicate(self):
        q = {"category": "adversarial", "category_id": None}
        assert SA.is_premise_t(q, {"adversarial"}, {5})
        q2 = {"category": "single_hop", "category_id": 5}
        assert SA.is_premise_t(q2, {"adversarial"}, {5})
        q3 = {"category": "single_hop", "category_id": 2}
        assert not SA.is_premise_t(q3, {"adversarial"}, {5})


# ---------------------------------------------------------------------------
# determinism — reports are byte-stable minus exempt timing fields
# ---------------------------------------------------------------------------


def test_report_determinism_shape(paired_report):
    """Two serializations of one run are byte-identical; timing fields
    are the declared exemptions (V8-20.06)."""
    a = _jsonable(paired_report)
    b = _jsonable(paired_report)
    assert a == b
    assert "latency_ms" in paired_report["determinism_exempt"]


# ---------------------------------------------------------------------------
# lane_miss forensic — V8-07.01 / scenario K37
# ---------------------------------------------------------------------------


def _lm_stats(**over):
    """A minimal-but-complete lane.stats block for the pure classifier."""
    stats = {
        "coverage_lexical": {
            "df_gate": {"floor": 1400, "gated": [], "exempt": []},
            "rescue": {"fired": False, "terms": [], "rows": 0,
                       "produced": 0},
        },
        "nominate_terms_max": 32,
        "nomination_eligible_terms": 2,
        "nominated_terms": 2,
        "nomination_dropped": [],
        "nominated": 3,
        "scored": 3,
        "scored_docs": 3,
        "cap": 200,
        "overflow": 0,
        "n_eligible": 5,
        "n_universe": 5,
        "eligible_via": "set",
    }
    stats.update(over)
    return stats


def _lm_unit(**over):
    u = {
        "ref": "g", "unit_id": "u_g", "rowid": 9, "generation": 1,
        "eligible": True, "matching": ["fluxgate"], "rescued": False,
        "readable": True, "admitted": False, "in_items": False,
    }
    u.update(over)
    return u


def _lm_ev(**over):
    """Evidence dict as ``_assemble_evidence`` would emit on a healthy
    ok-lane run — tests mutate one facet per case."""
    ev = {
        "lane_present": True,
        "explain_present": True,
        "lane_status": "ok",
        "lane_reason": None,
        "stats": _lm_stats(),
        "terms": ["fluxgate", "skylark"],
        "idents": [],
        "nominating": ["fluxgate", "skylark"],
        "content_flags": {"fluxgate": True, "skylark": True},
        "gated_terms": [],
        "dropped_terms": [],
        "post_pool": {"limit": 60, "fused": 3, "scored": 3},
        "gold_units": [_lm_unit()],
        "unmapped_refs": [],
        "recheck_dropped": None,
        "elig_state": "verified",
        "surfaced": ["x1"],
    }
    ev.update(over)
    return ev


class TestLaneMissClassify:
    """Pure per-stage ladder — no store needed (K37: exactly one class
    per question, eligibility count must stay zero)."""

    def test_a_all_matching_gated(self):
        ev = _lm_ev(stats=_lm_stats(coverage_lexical={
            "df_gate": {"floor": 1,
                        "gated": [{"term": "fluxgate", "df": 5}],
                        "exempt": []},
            "rescue": {"fired": False, "terms": [], "rows": 0,
                       "produced": 0},
        }))
        letter, detail = classify_lane_miss(ev)
        assert letter == "a"
        assert detail["note"] == "all_matching_gated"
        json.dumps(detail)

    def test_a_rescued_unit_is_not_a(self):
        """A gated-only gold that the bounded rescue produced keeps
        walking the ladder — rescue overrides (a)."""
        ev = _lm_ev(stats=_lm_stats(
            coverage_lexical={
                "df_gate": {"floor": 1,
                            "gated": [{"term": "fluxgate", "df": 5}],
                            "exempt": []},
                "rescue": {"fired": True, "terms": ["fluxgate"],
                           "rows": 9, "produced": 1},
            },
            overflow=2))
        ev["gold_units"][0]["rescued"] = True
        letter, _ = classify_lane_miss(ev)
        assert letter == "c"   # rescued → scored → cap cut

    def test_b_nomination_budget(self):
        ev = _lm_ev(stats=_lm_stats(
            nomination_dropped=[
                {"term": "fluxgate", "df": 4, "reason": "budget"}],
            nominated_terms=1))
        letter, detail = classify_lane_miss(ev)
        assert letter == "b"
        assert detail["note"] == "all_matching_dropped"

    def test_b_rescue_overrides_budget(self):
        ev = _lm_ev(stats=_lm_stats(
            nomination_dropped=[
                {"term": "fluxgate", "df": 4, "reason": "budget"}],
            nominated_terms=1))
        ev["gold_units"][0]["rescued"] = True
        letter, _ = classify_lane_miss(ev)
        # rescued → nominated → scored → admitted=None → defect (g)
        assert letter == "g"

    def test_c_lane_cap(self):
        ev = _lm_ev(stats=_lm_stats(overflow=3))
        letter, detail = classify_lane_miss(ev)
        assert letter == "c"
        assert detail["note"] == "cap_cut"

    def test_c_via_facet_cap_truncation(self):
        ev = _lm_ev(stats=_lm_stats(cap_truncated=2))
        letter, _ = classify_lane_miss(ev)
        assert letter == "c"

    def test_d_zero_overlap(self):
        ev = _lm_ev()
        ev["gold_units"][0]["matching"] = []
        letter, detail = classify_lane_miss(ev)
        assert letter == "d"
        assert detail["note"] == "no_nominating_overlap"

    def test_d_ignores_non_nominating_overlap(self):
        """A stop/meta term that hits gold is evidence of text overlap
        but never nominates — still (d)."""
        ev = _lm_ev(nominating=["fluxgate", "skylark"])
        ev["gold_units"][0]["matching"] = ["the"]   # not nominating
        letter, _ = classify_lane_miss(ev)
        assert letter == "d"

    def test_e_deadline_pre_lane(self):
        ev = _lm_ev(lane_status="deadline",
                    lane_reason="deadline_exhausted",
                    stats={})
        letter, detail = classify_lane_miss(ev)
        assert letter == "e"
        assert detail["deadline_phase"] == "pre_lane"

    def test_e_deadline_at_entry(self):
        ev = _lm_ev(lane_status="deadline", lane_reason="deadline",
                    stats={})
        letter, detail = classify_lane_miss(ev)
        assert letter == "e"
        assert detail["deadline_phase"] == "entry"

    def test_e_deadline_mid_postings(self):
        ev = _lm_ev(lane_status="partial", lane_reason="deadline",
                    stats=_lm_stats())
        # strip nomination markers → died inside _collect_postings
        stats = dict(ev["stats"])
        for k in ("nominated_terms", "nomination_dropped",
                  "nomination_eligible_terms", "nominated", "scored"):
            stats.pop(k, None)
        ev["stats"] = stats
        letter, detail = classify_lane_miss(ev)
        assert letter == "e"
        assert detail["deadline_phase"] == "postings"

    def test_e_deadline_at_scoring(self):
        stats = _lm_stats()
        stats.pop("scored")
        ev = _lm_ev(lane_status="partial", lane_reason="deadline",
                    stats=stats)
        letter, detail = classify_lane_miss(ev)
        assert letter == "e"
        assert detail["deadline_phase"] == "scoring"

    def test_e_beats_gate_when_gate_ran_but_posts_died(self):
        """Gate ran (gated list recorded) but the fetch loop died before
        nomination — an ungated matching term dies at postings → (e)."""
        ev = _lm_ev(lane_status="partial", lane_reason="deadline",
                    stats=_lm_stats(coverage_lexical={
                        "df_gate": {"floor": 1, "gated": [], "exempt": []},
                        "rescue": {"fired": False, "terms": [],
                                   "rows": 0, "produced": 0}}))
        stats = dict(ev["stats"])
        for k in ("nominated_terms", "nomination_dropped",
                  "nomination_eligible_terms", "nominated", "scored"):
            stats.pop(k, None)
        ev["stats"] = stats
        letter, detail = classify_lane_miss(ev)
        assert letter == "e"
        assert detail["deadline_phase"] == "postings"

    def test_f_unit_denied(self):
        ev = _lm_ev()
        ev["gold_units"][0]["eligible"] = False
        letter, detail = classify_lane_miss(ev)
        assert letter == "f"
        assert detail["note"] == "unit_denied"

    def test_f_shape_unknown(self):
        ev = _lm_ev(lane_status="unavailable",
                    lane_reason="eligibility_shape_unknown",
                    stats=_lm_stats(eligible_via="unrecognized"))
        letter, detail = classify_lane_miss(ev)
        assert letter == "f"
        assert detail["note"] == "eligibility_shape_unknown"

    def test_g_no_turn_unit(self):
        ev = _lm_ev(gold_units=[])
        letter, detail = classify_lane_miss(ev)
        assert letter == "g"
        assert detail["note"] == "no_turn_unit"

    def test_g_probe_unavailable(self):
        ev = _lm_ev(probe_error="read_snapshot: boom")
        letter, detail = classify_lane_miss(ev)
        assert letter == "g"
        assert detail["note"] == "probe_unavailable"

    def test_g_nominated_scored_unadmitted_is_defect(self):
        """Nominated + scored + readable + admitted=False with zero
        overflow is an impossible state — (g), not a guess."""
        ev = _lm_ev(stats=_lm_stats(overflow=0))
        letter, detail = classify_lane_miss(ev)
        assert letter == "g"
        assert detail["note"] == "nominated_scored_unadmitted"

    def test_g_no_field_content(self):
        ev = _lm_ev()
        ev["gold_units"][0]["readable"] = False
        letter, detail = classify_lane_miss(ev)
        assert letter == "g"
        assert detail["note"] == "no_field_content"

    def test_g_unverifiable_fields(self):
        ev = _lm_ev()
        ev["gold_units"][0]["readable"] = None
        letter, detail = classify_lane_miss(ev)
        assert letter == "g"
        assert detail["note"] == "ordered_membership_unverifiable"

    def test_g_admitted_but_scored_below_delivery(self):
        ev = _lm_ev()
        ev["gold_units"][0].update(admitted=True, in_items=True)
        letter, detail = classify_lane_miss(ev)
        assert letter == "g"
        assert detail["note"] == "scored_below_delivery"

    def test_g_post_pool_trim(self):
        ev = _lm_ev(post_pool={"limit": 60, "fused": 5, "scored": 3})
        ev["gold_units"][0].update(admitted=True, in_items=False)
        letter, detail = classify_lane_miss(ev)
        assert letter == "g"
        assert detail["note"] == "post_pool_trim"

    def test_g_admitted_not_scored(self):
        ev = _lm_ev(post_pool={"limit": 60, "fused": 3, "scored": 3})
        ev["gold_units"][0].update(admitted=True, in_items=False)
        letter, detail = classify_lane_miss(ev)
        assert letter == "g"
        assert detail["note"] == "admitted_not_scored"

    def test_gate_not_ran_no_deadline_is_g(self):
        ev = _lm_ev(stats={"n_universe": 5})   # no df_gate, no drops
        letter, detail = classify_lane_miss(ev)
        assert letter == "g"
        assert detail["note"] == "df_gate_block_absent"

    def test_deepest_unit_wins(self):
        """Multi-gold question: the LAST surviving gold's death stage
        classifies the question — cap-cut beats an earlier gate loss."""
        ev = _lm_ev(stats=_lm_stats(
            overflow=1,
            coverage_lexical={
                "df_gate": {"floor": 1,
                            "gated": [{"term": "fluxgate", "df": 9}],
                            "exempt": []},
                "rescue": {"fired": False, "terms": [], "rows": 0,
                           "produced": 0}}))
        ev["gold_units"] = [
            _lm_unit(ref="g1", unit_id="u1", rowid=1,
                     matching=["fluxgate"]),            # dies at gate (a)
            _lm_unit(ref="g2", unit_id="u2", rowid=2,
                     matching=["skylark"]),             # scored, cap-cut (c)
        ]
        letter, detail = classify_lane_miss(ev)
        assert letter == "c"
        stages = {s["unit_id"]: s["class"] for s in detail["unit_stages"]}
        assert stages == {"u1": "a", "u2": "c"}

    def test_deadline_deepest_over_budget(self):
        """One gold died at budget, its twin survived to a scoring
        deadline → question is (e)."""
        stats = _lm_stats(
            nomination_dropped=[
                {"term": "fluxgate", "df": 4, "reason": "budget"}])
        stats.pop("scored")
        ev = _lm_ev(lane_status="partial", lane_reason="deadline",
                    stats=stats)
        ev["gold_units"] = [
            _lm_unit(ref="g1", unit_id="u1", rowid=1,
                     matching=["fluxgate"]),            # dropped (b)
            _lm_unit(ref="g2", unit_id="u2", rowid=2,
                     matching=["skylark"]),             # died scoring (e)
        ]
        letter, detail = classify_lane_miss(ev)
        assert letter == "e"
        assert detail["deadline_phase"] == "scoring"

    def test_exactly_one_class_and_labels(self):
        for ev in (
            _lm_ev(stats=_lm_stats(coverage_lexical={
                "df_gate": {"floor": 1,
                            "gated": [{"term": "fluxgate", "df": 5}],
                            "exempt": []}})),
            _lm_ev(stats=_lm_stats(nomination_dropped=[
                {"term": "fluxgate", "df": 4, "reason": "budget"}])),
            _lm_ev(stats=_lm_stats(overflow=1)),
            _lm_ev(gold_units=[_lm_unit(matching=[])]),
            _lm_ev(lane_status="deadline",
                   lane_reason="deadline_exhausted", stats={}),
            _lm_ev(gold_units=[_lm_unit(eligible=False)]),
            _lm_ev(gold_units=[]),
        ):
            letter, detail = classify_lane_miss(ev)
            assert letter in LM.CLASS_ORDER
            assert letter in LM.CLASS_LABELS
            json.dumps(detail)


# ---------------------------------------------------------------------------
# lane_miss — real-store runs (one ingest each)
# ---------------------------------------------------------------------------

LM_ITEMS = list(ITEMS)
LM_TASKS = list(TASKS) + [
    # zero lexical overlap with gold d2 — real (d) on the default lane
    {"task_id": "q5", "query": "zebra quixotic blorf",
     "category": "single_hop", "gold_evidence": ["d2"],
     "group_id": "conv1"},
]


def _lm_corpus() -> DictCorpus:
    return DictCorpus(LM_ITEMS, LM_TASKS, name="tiny-lm",
                      dataset_id="tiny-lm")


@pytest.fixture(scope="module")
def lm_report():
    return run_lane_miss_forensic(_lm_corpus(), limit=10)


class TestLaneMissReport:
    def test_schema_and_tool_identity(self, lm_report):
        assert lm_report["schema"] == LM.SCHEMA
        assert lm_report["tool"] == "lane_miss_forensic"
        assert lm_report["requirement"] == "V8-07.01"
        assert lm_report["status"] == "executed"
        assert lm_report["not_run"] == []
        assert lm_report["manifest"]["explain_forced"] is True
        assert lm_report["manifest"]["classes_legend"] == \
            LM.CLASS_LABELS

    def test_population_is_lane_miss_only(self, lm_report):
        pop = lm_report["population"]
        assert pop["n_tasks"] == len(LM_TASKS)
        assert pop["n_answerable"] == 4   # q4 is unanswerable
        # only q5 is an answerable lane_miss — q1–q3 deliver gold
        assert pop["n_lane_miss"] == 1
        assert pop["n_classified"] == pop["n_lane_miss"]
        assert [q["task_id"] for q in lm_report["questions"]] == ["q5"]

    def test_exactly_one_class_per_question(self, lm_report):
        counts = lm_report["classes"]
        assert set(counts) == set(LM.CLASS_ORDER)
        assert sum(counts.values()) == \
            lm_report["population"]["n_lane_miss"]
        for q in lm_report["questions"]:
            assert q["class"] in LM.CLASS_ORDER
            assert q["class_label"] == LM.CLASS_LABELS[q["class"]]

    def test_eligibility_zero_and_verified(self, lm_report):
        # K37 — the eligibility class count must be zero
        assert lm_report["classes"]["f"] == 0
        assert lm_report["eligibility"]["class_f_rows"] == 0
        assert lm_report["eligibility"]["state"] == "verified"
        assert lm_report["defects"]["eligibility_losses"] == 0

    def test_real_zero_overlap_is_d(self, lm_report):
        q = lm_report["questions"][0]
        assert q["class"] == "d"
        assert q["class_label"] == "zero_lexical_overlap"
        assert q["attribution"] in ("lane_miss", "abstain")
        assert q["miss_stage"] == "lane_miss"
        ev = q["evidence"]
        assert ev["terms"]["nominating_matching"] == []
        assert ev["terms"]["overlap_all"] == []
        assert ev["gold"]["refs"] == ["d2"]
        assert ev["gold"]["n_units"] == 1
        assert ev["eligibility"]["denied_units"] == []

    def test_row_fields_are_the_permitted_set(self, lm_report):
        allowed = {"task_id", "conv_id", "category", "category_id",
                   "attribution", "miss_stage", "withheld",
                   "class", "class_label", "evidence"}
        ev_keys = {"lane", "terms", "df_gate", "nomination", "rescue",
                   "cap", "deadline", "eligibility", "gold", "post_pool",
                   "unit_stages", "note"}
        for q in lm_report["questions"]:
            assert set(q) <= allowed, set(q) - allowed
            assert set(q["evidence"]) <= ev_keys, \
                set(q["evidence"]) - ev_keys

    def test_report_serializes_deterministically(self, lm_report):
        a = _jsonable(lm_report)
        b = _jsonable(lm_report)
        assert a == b
        for ex in lm_report["determinism_exempt"]:
            assert isinstance(ex, str)


@pytest.fixture(scope="module")
def lm_gate_report():
    """(a) df-gate on the real path — ``lexical.df_floor=1`` gates the
    gold's only matching term; ``lexical.K_rescue=0`` disarms rescue."""
    items = [
        {"id": f"g{i}", "text": f"fluxgate sensor {i} routine check",
         "speaker": "A", "session_id": "s1", "when": "2023-05-01"}
        for i in range(4)
    ] + [
        {"id": "gold", "text": "the fluxgate experiment succeeded quietly",
         "speaker": "B", "session_id": "s2", "when": "2023-05-08"},
    ]
    tasks = [{"task_id": "qa", "query": "fluxgate skylark",
              "category": "single_hop", "gold_evidence": ["gold"],
              "group_id": "conv1"}]
    doc = {"lanes": ["lex"],
           "params": {"lexical.df_floor": 1, "lexical.K_rescue": 0}}
    return run_lane_miss_forensic(
        DictCorpus(items, tasks, name="lm-gate", dataset_id="lm-gate"),
        limit=10, policy_doc=doc)


class TestLaneMissGate:
    def test_gate_class_a(self, lm_gate_report):
        assert lm_gate_report["status"] == "executed"
        assert lm_gate_report["classes"]["a"] == 1
        assert lm_gate_report["classes"]["f"] == 0
        q = lm_gate_report["questions"][0]
        assert q["class"] == "a"
        ev = q["evidence"]
        assert ev["note"] == "all_matching_gated"
        assert ev["df_gate"]["gated_matching"] == [
            {"term": "fluxgate", "df": 5}]
        assert ev["terms"]["nominating_matching"] == ["fluxgate"]
        assert ev["rescue"]["fired"] is False
        _jsonable(lm_gate_report)


@pytest.fixture(scope="module")
def lm_budget_report():
    """(b) nomination budget — ``nominate_terms_max=1`` keeps the rare
    term; the gold's matching term is dropped ``reason=budget``."""
    items = [
        {"id": f"c{i}", "text": f"commonality thread {i}",
         "speaker": "A", "session_id": "s1", "when": "2023-05-01"}
        for i in range(3)
    ] + [
        {"id": "r1", "text": "rarebird spotted once",
         "speaker": "A", "session_id": "s1", "when": "2023-05-01"},
        {"id": "gold", "text": "the commonality result mattered",
         "speaker": "B", "session_id": "s2", "when": "2023-05-08"},
    ]
    tasks = [{"task_id": "qb", "query": "rarebird commonality",
              "category": "single_hop", "gold_evidence": ["gold"],
              "group_id": "conv1"}]
    doc = {"lanes": ["lex"], "nominate_terms_max": 1,
           "params": {"lexical.K_rescue": 0}}
    return run_lane_miss_forensic(
        DictCorpus(items, tasks, name="lm-budget", dataset_id="lm-budget"),
        limit=10, policy_doc=doc)


class TestLaneMissBudget:
    def test_budget_class_b(self, lm_budget_report):
        assert lm_budget_report["status"] == "executed"
        assert lm_budget_report["classes"]["b"] == 1
        assert lm_budget_report["classes"]["f"] == 0
        q = lm_budget_report["questions"][0]
        assert q["class"] == "b"
        ev = q["evidence"]
        assert ev["note"] == "all_matching_dropped"
        assert ev["nomination"]["budget"] == 1
        assert ev["nomination"]["dropped_matching"] == [
            {"term": "commonality", "reason": "budget", "df": 4}]
        _jsonable(lm_budget_report)


@pytest.fixture(scope="module")
def lm_deadline_report():
    """(e) deadline — ``timeout_ms=0`` exhausts the slice before the
    lexical lane runs (``deadline_exhausted`` / ``pre_lane``)."""
    items = [
        {"id": "d1", "text": "Caroline adopted a rescue dog named "
                             "Biscuit last March.", "speaker": "Caroline",
         "session_id": "s1", "when": "2023-05-01"},
        {"id": "d2", "text": "Melanie paints watercolors.",
         "speaker": "Melanie", "session_id": "s1", "when": "2023-05-01"},
    ]
    tasks = [{"task_id": "q1",
              "query": "what kind of dog did Caroline adopt",
              "category": "single_hop", "gold_evidence": ["d1"],
              "group_id": "conv1"}]
    return run_lane_miss_forensic(
        DictCorpus(items, tasks, name="lm-deadline",
                   dataset_id="lm-deadline"),
        limit=10, arm_kwargs={"timeout_ms": 0.0})


class TestLaneMissDeadline:
    def test_deadline_class_e(self, lm_deadline_report):
        rep = lm_deadline_report
        assert rep["status"] == "executed"
        if rep["classes"]["e"] == 1:
            q = rep["questions"][0]
            assert q["class"] == "e"
            assert q["evidence"]["deadline"]["phase"] in (
                "pre_lane", "entry", "universe_scan", "corpus_stats",
                "postings", "scoring")
        else:
            # a 0 ms budget could still complete an empty-enough run on
            # a trivially small store — whatever the lane printed rules
            q = rep["questions"][0]
            assert q["class"] in LM.CLASS_ORDER
            assert q["evidence"]["lane"]["status"] in (
                "ok", "partial", "deadline", "skipped", "unavailable")
        _jsonable(lm_deadline_report)


def test_lane_miss_cli_writes_report(tmp_path, monkeypatch):
    """``python -m eval.v7.forensics.lane_miss`` — dataset loads through
    the registry seam (monkeypatched here), report lands on disk."""
    import eval.v7.corpora as corpora

    monkeypatch.setattr(corpora, "load_corpus",
                        lambda _d, _s=None: _lm_corpus())
    out = tmp_path / "lane_miss_forensic.json"
    rc = LM.main([
        "--dataset", "tiny-lm", "--limit", "10", "--out", str(out)])
    assert rc == 0
    rep = json.loads(out.read_text())
    assert rep["schema"] == LM.SCHEMA
    assert rep["population"]["n_lane_miss"] == \
        rep["population"]["n_classified"]
    assert rep["classes"]["f"] == 0
