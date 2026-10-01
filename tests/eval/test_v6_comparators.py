"""Durable tests for the V6 comparator registry (SPEC_V6 §06,
V6-06.01–05).

The registry must EXECUTE its offline arms for real — real ``seed()``
and ``answer()`` against the shared V5 consumer corpus — and report its
unrunnable competitor rows by name with honest reasons, never
fabricated metrics. These tests pin that contract:

* the five always-run arms execute green end-to-end (small corpus);
* every declared row is named with status + reason + the 13 pins;
* the mem0 rows are tested-or-unavailable honestly — the probe is
  real, so whichever status the environment yields is asserted, never
  assumed;
* ``compare()`` refusal paths are preserved verbatim;
* corpus lifecycle events (supersedes/forget) are applied through each
  arm's real capability and disclosed in the row notes.
"""

from __future__ import annotations

import pytest

from eval.v5.comparators import REQUIRED_PINS
from eval.v5.corpus import seed_corpus

from eval.v6 import comparators as C


@pytest.fixture(scope="module")
def corpus():
    # 24 retained items: the full core mix (identifiers, unicode,
    # update pair, forget distractor) + 8 fillers + probes.
    return seed_corpus(memories=24, seed=42)


@pytest.fixture(scope="module")
def registry(corpus, tmp_path_factory):
    workdir = str(tmp_path_factory.mktemp("v6reg"))
    return C.run_registry(corpus, workdir)


def _rows(registry):
    return {r["arm"]: r for r in registry["rows"]}


# ---------------------------------------------------------------------------
# registry completeness + offline execution (V6-06.01/03/04)
# ---------------------------------------------------------------------------


def test_registry_names_every_declared_row(registry):
    rows = _rows(registry)
    for name in (
        "verbatim_memory", "verbatim_v2", "naive_fts", "vector_rag",
        "no_memory", "mem0_oss", "mem0_oss_inferfalse", "graphiti_oss",
        "holographic", "zep_hosted", "mem0_platform",
    ):
        assert name in rows, f"row {name} dropped from the registry"
        assert rows[name]["status"] in C.STATUSES
    assert set(registry["registry"]) == set(rows)


def test_always_run_arms_executed(registry, corpus):
    rows = _rows(registry)
    for name in C.ALWAYS_RUN_ARMS:
        r = rows[name]
        assert r["status"] == "tested", (
            f"{name} should execute offline: {r.get('reason')}"
        )
        assert r["executed"] is True
        m = r["metrics"]
        assert m["tasks"] == len(corpus.tasks)
        assert m["accuracy"] is not None
        assert m["task_success"] == m["accuracy"]
        assert m["recall_at_k"] is not None
        assert m["precision_at_k"] is not None
        assert m["abstain"]["n"] >= 1  # corpus carries no-answer probes
        assert (m["latency_ms"] or {}).get("n") == len(corpus.tasks)
        # claim-eligible: all 13 pins populated (V6-06.04)
        assert r["pins_missing"] == []
        # partial cost accounting is labeled partial (V6-06.05)
        assert r["costs"]["total"]["complete"] is False
        assert r["costs"]["total"]["unmeasured"]


def test_offline_arms_produce_real_metrics(registry):
    rows = _rows(registry)
    vm = rows["verbatim_memory"]["metrics"]
    nm = rows["no_memory"]["metrics"]
    # the consumer route actually retrieves — not a floor arm's score
    assert vm["accuracy"] > nm["accuracy"]
    assert vm["recall_at_k"] > 0.0
    # lifecycle honesty: the forgotten/superseded items stay out
    assert vm["forbidden_hits"] == 0
    # control arm: only the no-answer/empty-expectation tasks score
    assert nm["correct"] <= nm["abstain"]["n"] + 2


def test_lifecycle_applied_and_disclosed(registry):
    rows = _rows(registry)
    for name in ("verbatim_memory", "verbatim_v2", "naive_fts",
                 "vector_rag"):
            notes = " ".join(rows[name].get("notes") or [])
            assert "supersede core-meeting-v1->core-meeting-v2" in notes, (
                f"{name}: supersede event not applied/disclosed: {notes}"
            )
            assert "forget core-forgotten" in notes, (
                f"{name}: forget event not applied/disclosed: {notes}"
            )
    # and it shows in measured behavior: no arm may deliver the
    # forgotten or superseded item once the lifecycle ran
    for name in ("verbatim_memory", "verbatim_v2", "naive_fts",
                 "vector_rag"):
        assert rows[name]["metrics"]["forbidden_hits"] == 0


def test_no_memory_is_empty_floor(registry):
    m = _rows(registry)["no_memory"]["metrics"]
    assert m["recall_at_k"] == 0.0
    assert m["precision_at_k"] == 0.0
    assert m["forbidden_hits"] == 0


# ---------------------------------------------------------------------------
# probed rows — honest status, never fabricated (V6-06.02/03)
# ---------------------------------------------------------------------------


def test_mem0_rows_honest_status(registry):
    rows = _rows(registry)
    probe = C.probe_mem0()
    for name in ("mem0_oss", "mem0_oss_inferfalse"):
        r = rows[name]
        assert r["status"] in ("unavailable", "tested")
        if probe["available"]:
            # an importable mem0 still needs its pinned local
            # deployment; either outcome must carry a reason or metrics
            assert r["status"] == "tested" or r["reason"]
            if r["status"] == "tested":
                assert r["metrics"]["tasks"] > 0
        else:
            assert r["status"] == "unavailable"
            assert "mem0" in r["reason"].lower() or \
                "ModuleNotFoundError" in r["reason"]
    # the infer=False arm is a disclosed separate row, never default mem0
    inf = rows["mem0_oss_inferfalse"]
    disclosure = " ".join(inf.get("notes") or []) + \
        str((inf.get("pin") or {}).get("extractor"))
    assert "infer=False" in disclosure


def test_named_rows_reported_not_hidden(registry):
    rows = _rows(registry)
    for name in ("graphiti_oss", "holographic", "zep_hosted",
                 "mem0_platform"):
        r = rows[name]
        assert r["status"] in ("unavailable", "out_of_scope", "tested")
        if r["status"] != "tested":
            assert r["reason"], f"{name}: unrunnable row needs a reason"
    assert rows["zep_hosted"]["status"] == "out_of_scope"
    assert rows["mem0_platform"]["status"] == "out_of_scope"
    if rows["graphiti_oss"]["status"] == "unavailable":
        assert "graphiti" in rows["graphiti_oss"]["reason"]


def test_pins_populated_on_every_row(registry):
    for r in registry["rows"]:
        pin = r.get("pin") or {}
        for field_name in REQUIRED_PINS:
            assert field_name in pin, (
                f"{r['arm']}: pin field {field_name} absent"
            )
            assert pin[field_name] is not None, (
                f"{r['arm']}: pin field {field_name} left null — "
                "unrunnable rows mark 'n/a' honestly instead"
            )


# ---------------------------------------------------------------------------
# comparisons — verbatim vs each arm, refusals preserved (V6-06.04/05)
# ---------------------------------------------------------------------------


def test_comparisons_cover_every_other_row(registry):
    rows = _rows(registry)
    comps = registry["comparisons"]
    assert len(comps) == len(rows) - 1
    by_b = {c["b"]: c for c in comps}
    for name, r in rows.items():
        if name == "verbatim_memory":
            continue
        c = by_b[name]
        assert c["a"] == "verbatim_memory"
        if r["status"] == "tested":
            # a real comparison — either a valid verdict or an
            # enforced refusal reason, never silence
            if not c["valid"]:
                assert c["reason"]
        else:
            assert c["valid"] is False
            assert r["status"] in c["reason"] or r["reason"] in c["reason"]


def test_compare_refusal_paths_preserved():
    a = C.ComparatorRow(
        name="a", status="tested",
        pin=C.ComparatorPin(
            edition="verbatim", revision="x", deployment="local",
            extractor="e", embedder="hashing", reader="r", judge="j",
            prompts="p", settings="s", indexes="i",
            readiness_policy="rp", hardware="h", pricing_date="d"),
        metrics={"accuracy": 0.9})
    unavailable = C.ComparatorRow(
        name="b", status="unavailable", reason="dep absent")
    v = C.compare(a, unavailable)
    assert not v["valid"] and "untested" in v["reason"]

    unpinned = C.ComparatorRow(
        name="c", status="tested",
        pin=C.ComparatorPin(edition="verbatim"),
        metrics={"accuracy": 0.1})
    v = C.compare(a, unpinned)
    assert not v["valid"] and "unpinned" in v["reason"]

    foreign = C.ComparatorRow(
        name="d", status="tested", track="other_track",
        pin=a.pin, metrics={"accuracy": 0.1})
    v = C.compare(a, foreign)
    assert not v["valid"] and "track_mismatch" in v["reason"]

    no_metric = C.ComparatorRow(
        name="e", status="tested", pin=a.pin, metrics={})
    v = C.compare(a, no_metric)
    assert not v["valid"] and "not measured" in v["reason"]

    # and a valid comparison between two pinned tested rows stands
    b2 = C.ComparatorRow(
        name="f", status="tested", pin=a.pin,
        metrics={"accuracy": 0.4})
    v = C.compare(a, b2)
    assert v["valid"] and v["winner"] == "a" and v["delta"] > 0


# ---------------------------------------------------------------------------
# single-arm plumbing + portfolio tail
# ---------------------------------------------------------------------------


def test_run_arm_single(corpus, tmp_path):
    row = C.run_arm("naive_fts", corpus, workdir=str(tmp_path / "fts"))
    assert row.status == "tested"
    assert row.metrics["tasks"] == len(corpus.tasks)
    assert row.pin.pinned()
    # an unknown arm is named out_of_scope, never silently dropped
    row = C.run_arm("does_not_exist", corpus)
    assert row.status == "out_of_scope" and row.reason


def test_fts_arm_is_its_own_index(corpus, tmp_path):
    """naive_fts keeps its own FTS5 store — delete really removes."""
    from eval.v5.corpus import public_item

    arm = C.NaiveFtsArm(corpus, workdir=str(tmp_path / "fts2"))
    items = list(corpus.items)
    arm.seed([public_item(i) for i in items[:3]])
    try:
        tid = items[0].id
        arm.delete(tid)

        class _T:
            task_id = "x"
            query = items[0].text
        rec = arm.answer(_T(), 5)
        assert tid not in rec["returned_ids"]
    finally:
        arm.close()


def test_v5_tail_counts_unresolved():
    import json
    import os

    from eval.v6.portfolio import run_v5_tail
    out = run_v5_tail()
    assert out["requirements_total"] > 0
    assert sum(out["status_counts"].values()) == \
        out["requirements_total"]
    assert out["unresolved_count"] <= out["requirements_total"]
    assert len(out["unresolved_ids"]) == min(
        out["unresolved_count"], 200)
    # every listed id really is unresolved in the source file
    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "eval", "v5", "dispositions_v5.json")
    with open(src, "r", encoding="utf-8") as fh:
        reqs = json.load(fh)["requirements"]
    for rid in out["unresolved_ids"]:
        assert (reqs[rid].get("qualification_status") or "unset") \
            not in out["resolved_statuses"]


def test_portfolio_smoke(tmp_path):
    from eval.v6.portfolio import run_portfolio
    out = run_portfolio(quick=True, seed=42, suites=["v5_tail"],
                        workdir=str(tmp_path))
    assert out["portfolio"] == "v6"
    assert out["suites"]["v5_tail"]["status"] == "executed"
    assert out["verdict"] in ("passed", "executed", "inconclusive")
