"""Durable tests for the V6 envelope harness (SPEC_V6 §02.2,
V6-02.11–02.16).

These run TINY scaled-down envelopes — the harness shape, the budget
structure, bulk-seed projection honesty, capability reporting, and
honest-miss accounting are the durable properties; the real envelope
numbers live in generated reports, not test asserts.
"""

from __future__ import annotations

from eval.v5.corpus import seed_corpus
from eval.v5.harness import bulk_seed, seed_corpus_env, settle
from eval.v6 import envelopes as env6


# ---------------------------------------------------------------------------
# structure / imports
# ---------------------------------------------------------------------------


def test_module_and_budgets_structure():
    """BUDGETS carries all five §02.2 envelopes with their spec keys."""
    assert set(env6.BUDGETS) == {"A0", "A0_CACHE", "A0_NEURAL", "A1", "A3"}
    a0 = env6.BUDGETS["A0"]
    assert a0["search_p95_ms"] == 25.0
    assert a0["search_p99_ms"] == 75.0
    assert a0["add_ack_p95_ms"] == 50.0
    assert a0["memories"] == 1000
    assert env6.BUDGETS["A0_CACHE"]["search_p95_ms"] == 8.0
    an = env6.BUDGETS["A0_NEURAL"]
    assert an["search_p95_ms"] == 40.0 and an["search_p99_ms"] == 120.0
    a1 = env6.BUDGETS["A1"]
    assert a1["search_p95_ms"] == 40.0 and a1["add_ack_p95_ms"] == 50.0
    assert a1["writer_ops_per_s"] == 2.0
    assert a1["reader_queries_per_s"] == 10.0
    assert a1["unbounded_backlog"] is False
    a3 = env6.BUDGETS["A3"]
    assert a3["memories"] == 20_000
    assert a3["search_p95_ms"] == 150.0
    assert a3["add_ack_p95_ms"] == 80.0
    assert a3["peak_rss_mib"] == 1536.0


def test_envelope_result_shape():
    """Per-envelope dicts carry the portfolio-facing contract keys."""
    res = env6.EnvelopeResult(
        name="X", qualification="locally_measured", scale={},
        repetitions=1, add_ack_ms={}, search_ms={}, budgets={})
    d = res.to_dict()
    for key in ("status", "verdict", "measurements", "misses",
                "search_ms", "add_ack_ms", "budgets", "qualification",
                "environment", "errors"):
        assert key in d


# ---------------------------------------------------------------------------
# bulk_seed
# ---------------------------------------------------------------------------


def test_bulk_seed_produces_projected_sources(tmp_path):
    """bulk_seed returns real counts and the seeded texts are found by
    the public search after settle — harness-side seeding, public reads."""
    corpus = seed_corpus(memories=24, seed=7)
    env = seed_corpus_env(corpus, workdir=str(tmp_path / "seeded"),
                          worker="external")
    try:
        texts = [f"bulk seeded note {i}: garage token bg-{5000 + i}"
                 for i in range(40)]
        rep = bulk_seed(env, texts, drain_every=16)
        assert rep["added"] == 40
        assert rep["errors"] == 0
        assert rep["wall_s"] > 0
        assert rep["drain_totals"]["processed"] > 0
        st = settle(env)
        assert st["state"] in (
            "ready", "partial", "pending", "blocked", "unavailable")
        # Bounded poll: obligations resolve asynchronously — the
        # projection must land, but the wall time is load-dependent.
        import time
        deadline = time.monotonic() + 10.0
        found = False
        while time.monotonic() < deadline:
            hits = env.memory.search("bg-5017", limit=4)
            if any("bg-5017" in (h.quote or "")
                   for h in (hits.items or [])):
                found = True
                break
            time.sleep(0.2)
        assert found, "bulk-seeded token never became searchable"
    finally:
        env.close()


def test_bulk_seed_records_failures_visibly(tmp_path):
    """A failing add lands in env.add_errors — never silently skipped."""
    corpus = seed_corpus(memories=20, seed=8)
    env = seed_corpus_env(corpus, workdir=str(tmp_path / "seeded"),
                          worker="external")
    try:
        env.close()  # closed store → every add errors
        rep = bulk_seed(env, ["x"] * 3, drain_every=2)
        assert rep["added"] == 0
        assert rep["errors"] == 3
        assert len(env.add_errors) >= 3
    finally:
        env.close()


# ---------------------------------------------------------------------------
# A0
# ---------------------------------------------------------------------------


def test_measure_a0_runs_quick_scale(tmp_path):
    """A0 at toy scale produces real percentiles and the full result
    contract — without hanging."""
    r = env6.measure_a0(memories=48, seed=42, queries=16,
                        workdir=str(tmp_path / "a0"), immediate_probes=2)
    assert r["name"] == "A0"
    assert r["status"] == "measured"
    assert r["verdict"] in ("passed", "missed")
    assert r["qualification"] == "locally_measured"
    assert r["search_ms"]["n"] == 16
    assert r["search_ms"]["p95"] is not None
    assert r["add_ack_ms"]["n"] == 48
    assert r["budgets"]["search_p95_ms"] == 25.0
    assert isinstance(r["misses"], list)
    assert "ready" in r["measurements"]["search_status"] or \
        "insufficient" in r["measurements"]["search_status"]
    # V6-02.12: the immediate add→search boundary is visible beside
    # settled search.
    assert "immediate_add_search_ms" in r["measurements"]
    assert r["environment"]["cpu_count_logical"] >= 1


def test_miss_reporting_is_honest(tmp_path, monkeypatch):
    """An impossible budget produces a named miss — the harness never
    suppresses a loss to keep a number clean."""
    monkeypatch.setitem(env6.BUDGETS["A0"], "search_p95_ms", 1e-9)
    r = env6.measure_a0(memories=40, seed=42, queries=12,
                        workdir=str(tmp_path / "a0miss"),
                        immediate_probes=0)
    assert r["verdict"] == "missed"
    assert any("search p95" in m for m in r["misses"])
    assert r["search_ms"]["p95"] is not None


# ---------------------------------------------------------------------------
# A0-cache / A0-neural capability honesty
# ---------------------------------------------------------------------------


def test_measure_a0_cache_reports_real_counters(tmp_path):
    """Cache-on envelope either measures with real stats_for counters
    or reports unavailable — never a simulated hit rate."""
    r = env6.measure_a0_cache(memories=40, seed=42, queries=12, passes=2,
                              workdir=str(tmp_path / "a0c"))
    assert r["name"] == "A0_CACHE"
    assert r["status"] in ("measured", "unavailable")
    if r["status"] == "measured":
        cache = r["measurements"]["cache"]
        assert cache.get("enabled") is True
        assert "lookups" in cache and "hits" in cache
        assert r["support"]["pass_rows"][0]["role"] == "cold"
    else:
        assert r["support"]["unavailable_reason"]
        assert r["verdict"] == "unavailable"


def test_measure_a0_neural_capability_report(tmp_path):
    """Neural envelope runs only on a real provisioned artifact —
    unavailable carries the actual reason."""
    r = env6.measure_a0_neural(memories=40, seed=42, queries=8,
                               workdir=str(tmp_path / "a0n"))
    assert r["name"] == "A0_NEURAL"
    if r["status"] == "unavailable":
        assert r["verdict"] == "unavailable"
        assert r["errors"]
        assert "artifact" in r["support"]["unavailable_reason"].lower()
        probe = r["support"]["capability_probe"]
        assert probe["artifact_encoder_importable"] is True
    else:  # sibling artifact landed — real measurement required
        assert r["search_ms"]["n"] == 8
        assert r["support"]["capability_probe"]["encoder_available"]


# ---------------------------------------------------------------------------
# A1
# ---------------------------------------------------------------------------


def test_measure_a1_live_load(tmp_path):
    """A1 runs the writer/reader/managed-drain pattern and reports
    backlog + V6-02.11 same-session visibility."""
    r = env6.measure_a1(memories=48, seed=43, reader_queries=10,
                        writer_ops=5, visibility_probes=3,
                        workdir=str(tmp_path / "a1"))
    assert r["name"] == "A1"
    assert r["status"] == "measured"
    assert r["search_ms"]["n"] == 10
    assert "unbounded" in r["backlog"]
    vis = r["measurements"]["same_session_visibility"]
    assert vis["probes"] == 3
    assert vis["observed"] + vis["honest_pending"] \
        + vis["unobserved_ready"] == 3
    assert isinstance(r["misses"], list)


# ---------------------------------------------------------------------------
# A3 + add-ack scaling
# ---------------------------------------------------------------------------


def test_measure_a3_tiny_scale(tmp_path):
    """A3 at toy scale: bulk-seeded sources, per-checkpoint stage
    profile, scaling exponents, honest disclosure."""
    r = env6.measure_a3(memories=96, seed=42, queries=12,
                        checkpoints=(24, 48, 96), profile_adds=6,
                        checkpoint_queries=6, drain_every=32,
                        workdir=str(tmp_path / "a3"))
    assert r["name"] == "A3"
    assert r["status"] == "measured"
    assert r["search_ms"]["n"] == 12
    assert "scale_disclosure" in r["scale"]
    m = r["measurements"]
    assert m["seed"]["added"] == 96
    rows = m["add_ack_profile"]["checkpoints"]
    assert [row["n"] for row in rows] == [24, 48, 96]
    # the reachable stages actually produced samples
    stage_names = set()
    for row in rows:
        stage_names.update((row["stages_ms"] or {}).keys())
    assert "commit_tx" in stage_names
    assert "obligations" in stage_names
    assert isinstance(m["superlinear_stages"], list)
    assert m["index_growth"]["alpha"] is not None
    assert m["peak_rss_mib"] is None or m["peak_rss_mib"] > 0


def test_measure_add_ack_scaling_standalone(tmp_path):
    """The standalone V6-02.16 probe emits per-n p50/p95 plus the
    documented superlinear heuristic."""
    r = env6.measure_add_ack_scaling(
        memories_list=(24, 48), profile_adds=6,
        workdir=str(tmp_path / "ack"))
    assert r["suite"] == "add_ack_scaling"
    assert [c["n"] for c in r["checkpoints"]] == [24, 48]
    for c in r["checkpoints"]:
        assert c["add_ack_ms"]["n"] == 6
        assert c["add_ack_ms"]["p50"] is not None
        assert c["add_ack_ms"]["p95"] is not None
    assert isinstance(r["superlinear_stages"], list)
    assert "alpha" in r["stage_scaling"]["method"] or \
        "alpha" in env6.SUPERLINEAR_METHOD


def test_run_envelopes_shape(tmp_path):
    """run_envelopes returns the per-envelope {status, measurements,
    misses, verdict} contract for portfolio consumption."""
    out = env6.run_envelopes(
        scale="quick",
        params={
            "a0_memories": 40, "a0_queries": 10,
            "cache_memories": 40, "cache_queries": 8, "cache_passes": 2,
            "neural_memories": 40, "neural_queries": 8,
        },
        suites=["a0", "a0_cache", "a0_neural"],
        workdir=str(tmp_path / "run"),
    )
    assert out["suite"] == "envelopes_v6"
    assert set(out["envelopes"]) == {"a0", "a0_cache", "a0_neural"}
    for name, e in out["envelopes"].items():
        for key in ("status", "measurements", "misses", "verdict"):
            assert key in e, f"{name} missing {key}"
    assert out["verdict"] in ("passed", "missed", "inconclusive",
                              "failed")
    assert out["spec_budgets"]["A3"]["memories"] == 20_000
