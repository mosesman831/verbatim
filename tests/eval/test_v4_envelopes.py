"""Durable tests for the SPEC_V4 §56 workload-envelope harness
(V4-56.01/56.02/56.05/56.06 anchors).

These run a TINY scaled-down envelope — the harness shape, report
contract, warmup exclusion, percentile math, and honest miss reporting are
the durable properties; the real S0/S1 qualification numbers live in
``eval/v4/envelope_*_report.*`` (generated artifacts, not test asserts).
"""

from __future__ import annotations

import dataclasses
import json
import os

import pytest

from eval.v4 import envelopes as env


def _tiny_spec(**over) -> env.EnvelopeSpec:
    """A scaled-down envelope: same code path, ~2 orders smaller."""
    spec = dataclasses.replace(
        env.S0_SPEC,
        envelope="tiny",
        claims=10,
        spans=30,
        episodes=2,
        procedures=0,
        recall_clients=2,
        captures_per_s=5.0,
        capture_payload_bytes=512,
        queries=24,
        warmup=5,
        restarts=2,
        embedding_backend="none",
        readiness_deadline_s=3.0,
        targets={
            "recall.p95_ms": 25.0,
            "recall.p99_ms": 75.0,
            "capture.ack_p95_ms": 50.0,
        },
    )
    for k, v in over.items():
        spec = dataclasses.replace(spec, **{k: v})
    return spec


def test_tiny_envelope_runs_and_report_shape(tmp_path):
    """V4-56.01/56.02: the harness drives capture→drain→recall through the
    real public path and returns the full report contract."""
    report = env.run_envelope(_tiny_spec(), str(tmp_path))
    assert report["envelope"] == "tiny"
    assert report["samples"]["measured_queries"] == 24
    assert report["samples"]["restart_repetitions"] == 2
    assert len(report["samples"]["per_repetition"]) == 2
    obs = report["observed_volumes"]
    assert obs["claims_total"] >= 10
    assert obs["spans"] >= 30
    assert obs["episodes"] >= 2
    assert report["metrics"]["recall"]["clients"] == 2
    assert report["metrics"]["recall"]["n"] == 24
    for key in ("p50_ms", "p95_ms", "p99_ms", "max_ms", "mean_ms"):
        assert report["metrics"]["recall"][key] is not None
    # every declared target evaluated into a row
    assert {t["target"] for t in report["targets"]} == {
        "recall.p95_ms", "recall.p99_ms", "capture.ack_p95_ms"
    }
    assert isinstance(report["met"], bool)
    assert report["environment"]["cpu_count"] >= 1
    assert os.path.exists(
        os.path.join(str(tmp_path), "tiny", "env.db")
    )


def test_warmup_excluded_from_measured_samples(tmp_path):
    """V4-56.01: warmup queries run but never enter the sample set."""
    report = env.run_envelope(
        _tiny_spec(queries=16, warmup=7, restarts=1), str(tmp_path)
    )
    assert report["samples"]["measured_queries"] == 16
    assert report["samples"]["warmup_queries"] == 7
    assert report["samples"]["warmup_excluded"] is True
    assert report["metrics"]["recall"]["n"] == 16


def test_percentile_math_nearest_rank():
    """V4-55.09: p50/p95/p99/max follow the nearest-rank convention."""
    p = env.percentiles([float(i) for i in range(1, 101)])
    assert p["n"] == 100
    assert p["p50_ms"] == 50.0
    assert p["p95_ms"] == 95.0
    assert p["p99_ms"] == 99.0
    assert p["max_ms"] == 100.0
    assert p["mean_ms"] == pytest.approx(50.5)
    empty = env.percentiles([])
    assert empty["n"] == 0 and empty["p95_ms"] is None
    one = env.percentiles([3.0])
    assert one["p99_ms"] == 3.0


def test_target_miss_is_reported_not_hidden(tmp_path):
    """V4-56.11 / honesty: an impossible bound produces met=False with the
    measured value beside it — the harness never suppresses a miss."""
    spec = _tiny_spec(
        queries=12, warmup=3, restarts=1,
        targets={"recall.p95_ms": 1e-9},
    )
    report = env.run_envelope(spec, str(tmp_path))
    assert report["met"] is False
    row = report["targets"][0]
    assert row["met"] is False
    assert row["measured_ms"] is not None and row["measured_ms"] > 1e-9


def test_reports_written_json_and_md(tmp_path):
    """envelope_<id>_report.{json,md} land with matching numbers."""
    report = env.run_envelope(
        _tiny_spec(queries=8, warmup=2, restarts=1), str(tmp_path)
    )
    out = tmp_path / "out"
    paths = env.write_reports(report, str(out))
    assert os.path.exists(paths["json"]) and os.path.exists(paths["md"])
    loaded = json.load(open(paths["json"]))
    assert loaded["samples"]["measured_queries"] == report[
        "samples"]["measured_queries"]
    md = open(paths["md"]).read()
    assert "Workload envelope" in md and "Targets" in md


def test_restart_repetitions_reopen_store(tmp_path):
    """V4-56.01: each repetition reopens the store — per-repetition sample
    counts are recorded so a dead restart cannot hide."""
    report = env.run_envelope(
        _tiny_spec(queries=20, warmup=2, restarts=3, recall_clients=2),
        str(tmp_path),
    )
    reps = report["samples"]["per_repetition"]
    assert len(reps) == 3
    assert sum(r["measured"] for r in reps) == 20
