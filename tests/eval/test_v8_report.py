"""Durable tests for the V8 report renderer (SPEC_V8 V8-03.01,
V8-15.04, V8-15.05, V8-15.17 — scenarios K02, K03, K08, K19).

Pins the honesty rules: ``not_run`` renders verbatim in every metric
position (a gate input carrying a number is still ``not_run``), a
scoreboard value without a manifest digest is refused behind
``unpinned``, missing SB8 rows render ``not_run`` rather than being
omitted, and the paired-statistics block prints per-conversation
deltas plus the cluster count.
"""

from __future__ import annotations

from eval.v8 import ledger as L
from eval.v8 import report as R
from eval.v8 import stats as S


def _metric_set(any10=0.60):
    row = {"any@10": any10, "any@20": 0.7, "all@10": 0.1,
           "prop@10": 0.4, "mrr@10": 0.3, "ndcg@10": 0.4}
    return {
        "answerable": {"overall": dict(row),
                       "categories": {"1": dict(row)}},
        "cat5_premise": {"delivery_any@10": 0.4,
                         "correct_refusal": 0.2},
        "false_insufficient": 0.17,
        "latency_500ms": {"p50": 300.0, "p95": 600.0},
        "sql_statements_per_query_median": 4000.0,
    }


def _record(**over):
    rec = {
        "requirement_ids": ["V8-06.01"],
        "defect_ids": ["D8-05"],
        "base_commit": "a" * 40,
        "head_commit": "b" * 40,
        "base_manifest_digest": "1" * 64,
        "candidate_manifest_digest": "2" * 64,
        "metrics": {"base": _metric_set(), "candidate": _metric_set(0.62)},
        "decision": "default_on",
        "decision_record": "eval/v8/decisions/V8-06.01.md",
    }
    rec.update(over)
    return L.validate_record(rec)


# ---------------------------------------------------------------------------
# metric cell honesty (the K19/K03 primitive)
# ---------------------------------------------------------------------------


def test_metric_cell_not_run_verbatim():
    assert R.metric_cell(0.97, status="not_run") == "not_run"
    assert R.metric_cell("not_run") == "not_run"
    assert R.metric_cell(None, status="not_run") == "not_run"


def test_metric_cell_unpinned_without_digest():
    assert R.metric_cell(0.92, status="passed") == "unpinned"
    assert R.metric_cell(0.92, status="passed",
                         manifest_digest="f" * 64) == "0.920"


# ---------------------------------------------------------------------------
# scoreboard (V8-03.01)
# ---------------------------------------------------------------------------


def test_scoreboard_missing_rows_render_not_run():
    md = R.render_scoreboard({})
    assert "SB8-01" in md and "SB8-16" in md  # never omitted
    for line in md.splitlines():
        if line.startswith("| SB8-"):
            assert "not_run" in line


def test_scoreboard_not_run_row_verbatim():
    md = R.render_scoreboard({
        "SB8-01": {"status": "not_run", "value": 0.99,
                   "manifest_digest": "f" * 64},
    })
    line = next(l for l in md.splitlines() if "SB8-01" in l)
    assert "not_run" in line
    assert "0.99" not in line  # the number never leaks through


def test_scoreboard_unpinned_value_refused():
    md = R.render_scoreboard({
        "SB8-12": {"status": "passed", "value": 0.925,
                   "target": "≥ 0.925"},
    })
    line = next(l for l in md.splitlines() if "SB8-12" in l)
    assert "unpinned" in line
    assert "0.925" not in line.split("|")[3]  # value cell refuses


def test_scoreboard_pinned_value_renders():
    md = R.render_scoreboard({
        "SB8-01": {"status": "passed", "value": 0.72,
                   "target": "≥ 0.72", "reference": "BM25 0.589",
                   "protocol": "Track R dev",
                   "manifest_digest": "f" * 64},
    })
    line = next(l for l in md.splitlines() if "SB8-01" in l)
    assert "passed" in line and "0.720" in line


def test_scoreboard_blocked_status():
    md = R.render_scoreboard({
        "SB8-13": {"status": "blocked_on_authorization",
                   "target": "≥ 0.95"},
    })
    line = next(l for l in md.splitlines() if "SB8-13" in l)
    assert "blocked_on_authorization" in line


# ---------------------------------------------------------------------------
# gates (K19 — not_run inputs never show a metric)
# ---------------------------------------------------------------------------


def test_gates_not_run_input_renders_not_run():
    gates = {
        "G8-01": {
            "name": "Harness integrity",
            "status": "not_run",
            "inputs_ready": 0, "inputs_total": 2,
            "inputs": [
                {"artifact": "v8.ndcg_fix", "status": "not_run",
                 "value": 1.0, "manifest_digest": "e" * 64},
                {"artifact": "v8.ledger", "status": "passed",
                 "value": 1.0, "manifest_digest": "f" * 64},
            ],
        },
    }
    md = R.render_gates_table(gates)
    lines = md.splitlines()
    gate_line = next(l for l in lines if "G8-01" in l)
    assert "not_run" in gate_line
    # the not_run input row renders not_run — not its carried value
    inp = next(l for l in lines if "v8.ndcg_fix" in l)
    assert "not_run" in inp
    assert "1.000" not in inp
    ok = next(l for l in lines if "v8.ledger" in l)
    assert "passed" in ok and "1.000" in ok


# ---------------------------------------------------------------------------
# ledger table (V8-15.04 — K02)
# ---------------------------------------------------------------------------


def test_ledger_table_renders_keep_decision():
    recs = [
        _record(),
        _record(requirement_ids=["V8-11.03"], decision="flag_off",
                rejecting_metric="answerable.any@10",
                base_commit="b" * 40, head_commit="c" * 40,
                base_manifest_digest="3" * 64,
                candidate_manifest_digest="4" * 64),
    ]
    md = R.render_ledger_table(recs)
    assert "V8-06.01" in md and "default_on" in md
    line = next(l for l in md.splitlines() if "V8-11.03" in l)
    assert "flag_off" in line
    assert "answerable.any@10" in line  # rejecting metric named (K02)
    assert "+0.0200" in md              # Δany@10 column


# ---------------------------------------------------------------------------
# paired statistics (V8-15.05 — K08)
# ---------------------------------------------------------------------------


def test_comparison_renders_clusters_and_per_conv_deltas():
    pairs = [(0.0, 1.0)] * 8 + [(1.0, 1.0)] * 12
    conv = ["conv_a"] * 10 + ["conv_b"] * 10
    comp = S.paired_bootstrap(
        pairs, resamples=200, seed=1, conversation_ids=conv)
    comp = dict(comp)
    comp["mcnemar"] = S.mcnemar_from_pairs(pairs)
    md = R.render_comparison("V8-06.01 any@10", comp)
    assert "clusters: 2 conversation(s)" in md
    assert "conv_a" in md and "conv_b" in md
    assert "delta" in md and "ci_lower" in md
    assert "McNemar" in md


def test_comparison_not_run():
    md = R.render_comparison("V8-99.99", {})
    assert "not_run" in md


# ---------------------------------------------------------------------------
# full report composition (V8-15.17)
# ---------------------------------------------------------------------------


def test_render_report_composes_sections():
    comp = S.paired_bootstrap(
        [(0.0, 1.0)] * 5 + [(1.0, 1.0)] * 5, resamples=100, seed=0,
        conversation_ids=["c1"] * 5 + ["c2"] * 5)
    md = R.render_report(
        ledger_records=[_record()],
        scoreboard={},
        gates={"G8-01": {"name": "Harness", "status": "not_run",
                         "inputs_ready": 0, "inputs_total": 1,
                         "inputs": []}},
        comparisons={"V8-06.01": comp},
        header={"date": "2026-09-18", "git_rev": "f" * 40,
                "dirty": False, "machine": "testbox",
                "manifests": ["1" * 64]},
        blocked=["SB8-12: awaiting O6"],
    )
    assert "report_v8.md" in md
    assert "2026-09-18" in md and "clean" in md
    assert "Scoreboard" in md and "Gates" in md
    assert "Requirement ledger" in md
    assert "Paired statistics" in md
    assert "Blocked / deferred" in md
    assert "clusters: 2" in md
