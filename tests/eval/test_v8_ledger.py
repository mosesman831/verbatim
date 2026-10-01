"""Durable tests for the V8 paired-run ledger (SPEC_V8 V8-15.04,
V8-00.05/00.06 — scenarios K01, K02, K04).

Pins the append-time contract: required fields are enforced, missing
or malformed manifest digests are refused outright, ``flag_off`` /
``reverted`` decisions must name the rejecting metric, metric slots
accept numbers or the honest ``not_run`` label (never silent zeros),
and the K04 pairing chain catches a record whose baseline is not the
immediately-previous head.
"""

from __future__ import annotations

import json
import os

import pytest

from eval.v8 import ledger as L


def _metric_set(any10=0.60, any20=0.70, all10=0.10, prop10=0.40,
                mrr10=0.35, ndcg10=0.40):
    cat_row = {
        "any@10": any10, "any@20": any20, "all@10": all10,
        "prop@10": prop10, "mrr@10": mrr10, "ndcg@10": ndcg10,
    }
    return {
        "answerable": {
            "overall": dict(cat_row),
            "categories": {"1": dict(cat_row), "2": dict(cat_row)},
        },
        "cat5_premise": {
            "delivery_any@10": 0.41, "correct_refusal": 0.22,
        },
        "false_insufficient": 0.17,
        "latency_500ms": {"p50": 320.0, "p95": 610.0},
        "sql_statements_per_query_median": 4500.0,
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
    return rec


# ---------------------------------------------------------------------------
# happy path + normalization
# ---------------------------------------------------------------------------


def test_append_read_round_trip(tmp_path):
    p = str(tmp_path / "ledger.jsonl")
    rec = L.append_record(p, _record())
    assert rec["schema"] == L.RECORD_SCHEMA
    assert rec["rejecting_metric"] is None
    got = L.read_ledger(p)
    assert len(got) == 1
    assert got[0]["requirement_ids"] == ["V8-06.01"]
    assert got[0]["metrics"]["candidate"]["answerable"]["overall"]["any@10"] == 0.62


def test_append_creates_parent_dirs(tmp_path):
    p = str(tmp_path / "sub" / "dir" / "ledger.jsonl")
    L.append_record(p, _record())
    assert os.path.exists(p)


def test_read_missing_file_is_empty(tmp_path):
    assert L.read_ledger(str(tmp_path / "none.jsonl")) == []
    assert L.latest(str(tmp_path / "none.jsonl")) is None


def test_append_is_append_only(tmp_path):
    p = str(tmp_path / "ledger.jsonl")
    L.append_record(p, _record())
    L.append_record(p, _record(head_commit="c" * 40,
                               base_commit="b" * 40,
                               requirement_ids=["V8-06.02"]))
    recs = L.read_ledger(p)
    assert len(recs) == 2
    assert recs[1]["requirement_ids"] == ["V8-06.02"]


# ---------------------------------------------------------------------------
# required-field validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", [
    "requirement_ids", "base_commit", "head_commit", "metrics",
    "decision", "decision_record",
])
def test_missing_required_field_refused(tmp_path, key):
    rec = _record()
    del rec[key]
    with pytest.raises(L.LedgerError) as ei:
        L.append_record(str(tmp_path / "l.jsonl"), rec)
    assert key in str(ei.value)


def test_empty_requirement_ids_refused(tmp_path):
    with pytest.raises(L.LedgerError):
        L.append_record(str(tmp_path / "l.jsonl"),
                        _record(requirement_ids=[]))


@pytest.mark.parametrize("bad", ["V7-06.01", "V8-6.1", "V8-06", "v8-06.01", 601])
def test_bad_requirement_id_refused(tmp_path, bad):
    with pytest.raises(L.LedgerError):
        L.append_record(str(tmp_path / "l.jsonl"),
                        _record(requirement_ids=[bad]))


def test_multiple_requirement_ids_accepted(tmp_path):
    rec = L.append_record(str(tmp_path / "l.jsonl"),
                          _record(requirement_ids=["V8-06.01", "V8-06.02"]))
    assert rec["requirement_ids"] == ["V8-06.01", "V8-06.02"]


def test_bad_defect_id_refused(tmp_path):
    with pytest.raises(L.LedgerError):
        L.append_record(str(tmp_path / "l.jsonl"),
                        _record(defect_ids=["D7-01"]))


# ---------------------------------------------------------------------------
# digest enforcement (the hard rule — unpinned evidence is refused)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", [
    "base_manifest_digest", "candidate_manifest_digest",
])
@pytest.mark.parametrize("bad", [None, "", "unpinned", "not a digest",
                                 "z" * 64, 12345])
def test_missing_or_bad_digest_refused(tmp_path, key, bad):
    with pytest.raises(L.LedgerError) as ei:
        L.append_record(str(tmp_path / "l.jsonl"), _record(**{key: bad}))
    assert key in str(ei.value)


def test_nothing_written_on_refusal(tmp_path):
    p = str(tmp_path / "l.jsonl")
    with pytest.raises(L.LedgerError):
        L.append_record(p, _record(candidate_manifest_digest=""))
    assert L.read_ledger(p) == []


# ---------------------------------------------------------------------------
# decision discipline (V8-00.06 / V8-15.04)
# ---------------------------------------------------------------------------


def test_unknown_decision_refused(tmp_path):
    with pytest.raises(L.LedgerError):
        L.append_record(str(tmp_path / "l.jsonl"),
                        _record(decision="shipped"))


@pytest.mark.parametrize("decision", ["flag_off", "reverted"])
def test_rejecting_decisions_require_metric(tmp_path, decision):
    with pytest.raises(L.LedgerError) as ei:
        L.append_record(str(tmp_path / "l.jsonl"),
                        _record(decision=decision))
    assert "rejecting_metric" in str(ei.value)


@pytest.mark.parametrize("decision", ["flag_off", "reverted"])
def test_rejecting_decisions_accepted_with_metric(tmp_path, decision):
    rec = L.append_record(
        str(tmp_path / "l.jsonl"),
        _record(decision=decision, rejecting_metric="answerable.any@10"))
    assert rec["decision"] == decision


def test_structural_decision_accepted(tmp_path):
    rec = L.append_record(str(tmp_path / "l.jsonl"),
                          _record(decision="structural"))
    assert rec["decision"] == "structural"


# ---------------------------------------------------------------------------
# metric-set validation
# ---------------------------------------------------------------------------


def test_missing_metric_refused(tmp_path):
    ms = _metric_set()
    del ms["answerable"]["overall"]["any@10"]
    with pytest.raises(L.LedgerError) as ei:
        L.append_record(str(tmp_path / "l.jsonl"),
                        _record(metrics={"base": ms,
                                         "candidate": _metric_set()}))
    assert "any@10" in str(ei.value)


def test_missing_category_metric_refused(tmp_path):
    ms = _metric_set()
    del ms["answerable"]["categories"]["1"]["ndcg@10"]
    with pytest.raises(L.LedgerError):
        L.append_record(str(tmp_path / "l.jsonl"),
                        _record(metrics={"base": ms,
                                         "candidate": _metric_set()}))


def test_missing_cat5_field_refused(tmp_path):
    ms = _metric_set()
    del ms["cat5_premise"]["delivery_any@10"]
    with pytest.raises(L.LedgerError):
        L.append_record(str(tmp_path / "l.jsonl"),
                        _record(metrics={"base": ms,
                                         "candidate": _metric_set()}))


def test_missing_latency_field_refused(tmp_path):
    ms = _metric_set()
    del ms["latency_500ms"]["p95"]
    with pytest.raises(L.LedgerError):
        L.append_record(str(tmp_path / "l.jsonl"),
                        _record(metrics={"base": ms,
                                         "candidate": _metric_set()}))


def test_not_run_metric_accepted(tmp_path):
    ms = _metric_set()
    ms["latency_500ms"] = {"p50": "not_run", "p95": "not_run"}
    ms["sql_statements_per_query_median"] = "not_run"
    rec = L.append_record(str(tmp_path / "l.jsonl"),
                          _record(metrics={"base": ms,
                                           "candidate": _metric_set()}))
    assert rec["metrics"]["base"]["latency_500ms"]["p50"] == "not_run"


def test_missing_base_or_candidate_side_refused(tmp_path):
    with pytest.raises(L.LedgerError):
        L.append_record(str(tmp_path / "l.jsonl"),
                        _record(metrics={"candidate": _metric_set()}))


# ---------------------------------------------------------------------------
# file integrity + the K04 pairing chain
# ---------------------------------------------------------------------------


def test_malformed_line_is_an_error_not_a_skip(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text(json.dumps(_record()) + "\nnot json\n", encoding="utf-8")
    with pytest.raises(L.LedgerError):
        L.read_ledger(str(p))


def test_strict_read_revalidates(tmp_path):
    p = tmp_path / "l.jsonl"
    bad = _record()
    bad["decision"] = "shipped"
    p.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(L.LedgerError):
        L.read_ledger(str(p), strict=True)
    # non-strict read still parses — callers choose the audit level
    assert len(L.read_ledger(str(p))) == 1


def test_check_chain_clean(tmp_path):
    r1 = _record()
    r2 = _record(requirement_ids=["V8-06.02"],
                 base_commit="b" * 40, head_commit="c" * 40)
    assert L.check_chain([r1, r2]) == []


def test_check_chain_flags_skipped_head(tmp_path):
    r1 = _record()
    # base is the wave start, not r1's head — the K04 violation
    r2 = _record(requirement_ids=["V8-06.02"],
                 base_commit="a" * 40, head_commit="c" * 40)
    violations = L.check_chain([r1, r2])
    assert len(violations) == 1
    assert "base_commit" in violations[0]


# ---------------------------------------------------------------------------
# metric helpers consumed by report + keep rule
# ---------------------------------------------------------------------------


def test_delta_any10():
    rec = _record()  # candidate any@10 0.62 vs base 0.60
    assert L.delta_any10(rec) == pytest.approx(0.02)


def test_delta_any10_not_run():
    rec = _record()
    rec["metrics"]["base"]["answerable"]["overall"]["any@10"] = "not_run"
    assert L.delta_any10(rec) is None


def test_category_recall_deltas():
    rec = _record()
    assert L.category_recall_deltas(rec) == {
        "1": pytest.approx(0.02), "2": pytest.approx(0.02),
    }
