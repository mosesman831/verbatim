"""Durable tests for the V7 stage profiler (SPEC_V7 §32.17,
``stage_profile/v7``) and the V7 mutation-target scaffold (§35,
V7-35.01).

Covers: record producers and their field validation, the ``ProfiledCall``
context manager (mark/mark_lane timing against an injected deterministic
clock, injected SQL/snapshot/byte counters, callable wrapping, async add
fields, exception paths), envelope aggregation math (nearest-rank
percentiles, honest histograms for label fields, per-field denominators),
JSON round-trip, and ``eval/v7/mutations.yaml`` loading under the *same*
``tools/run_mutation.py`` loader as the v4 suite.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from eval.v7 import stage_profile as sp

REPO_ROOT = Path(__file__).resolve().parents[2]
MUTATIONS_YAML = REPO_ROOT / "eval" / "v7" / "mutations.yaml"
RUNNER = REPO_ROOT / "tools" / "run_mutation.py"


# ---------------------------------------------------------------------------
# deterministic instruments
# ---------------------------------------------------------------------------


class FakeClock:
    """Injectable ``()``-returns-seconds clock for ProfiledCall tests."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class CountBox:
    """``.count``-attribute counter (the ``eval.v5.timers.SqlCounter``
    shape) — read at enter and exit, delta recorded."""

    def __init__(self) -> None:
        self.count = 0


def _load_runner():
    """Import ``tools/run_mutation.py`` by path (tools/ is not a package) —
    the same loader the v4 mutation suite uses."""
    spec = importlib.util.spec_from_file_location("run_mutation", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    # dataclass() resolves cls.__module__ via sys.modules — register first.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# record producers
# ---------------------------------------------------------------------------


def test_record_search_declared_fields():
    rec = sp.record_search(
        t_total=12.5,
        t_barrier=1.0,
        t_analyze=0.4,
        sql_statements=7,
        snapshots=1,
        bytes_read=4096,
        pool_R=100,
        items=6,
        tokens=812,
        status="ready",
        coverage_digest="deadbeef",
        t_lane={"lexical": 3.0, "dense": 5},
        candidates={"lexical": 40, "dense": 17},
    )
    assert rec.kind == "search"
    assert rec.fields["t_total"] == 12.5
    assert rec.fields["status"] == "ready"
    assert rec.fields["coverage_digest"] == "deadbeef"
    # spec t_lane[<name>] / candidates[<lane>] live in their own dicts
    assert rec.t_lane == {"lexical": 3.0, "dense": 5.0}
    assert rec.candidates == {"lexical": 40, "dense": 17}
    # unset fields stay absent — no fabricated zeros
    assert "t_rrf" not in rec.fields
    assert "t_lane" not in rec.fields


def test_record_search_mapping_and_kwargs_merge():
    rec = sp.record_search({"t_total": 1.0}, t_total=2.0, items=3)
    assert rec.fields["t_total"] == 2.0  # keyword wins on collision
    assert rec.fields["items"] == 3


def test_record_search_validation():
    with pytest.raises(ValueError):
        sp.record_search(t_nope=1.0)  # undeclared field rejected
    with pytest.raises(ValueError):
        sp.record_search(t_total=-1.0)  # negative time
    with pytest.raises(ValueError):
        sp.record_search(items=True)  # bool is not a number here
    with pytest.raises(ValueError):
        sp.record_search(t_total=float("nan"))
    with pytest.raises(ValueError):
        sp.record_search(t_total=float("inf"))
    with pytest.raises(TypeError):
        sp.record_search(status=3)  # status is a label field
    with pytest.raises(TypeError):
        sp.record_search(coverage_digest=123)
    with pytest.raises(ValueError):
        sp.record_search(t_lane={"lexical": -0.5})
    with pytest.raises(ValueError):
        sp.record_search(t_lane={"": 1.0})
    with pytest.raises(ValueError):
        sp.record_search(candidates={"lexical": 1.5})
    with pytest.raises(ValueError):
        sp.record_search(candidates={"lexical": -1})
    # search fields on an add record are rejected (field sets are disjoint)
    with pytest.raises(ValueError):
        sp.record_add(t_total=1.0)
    with pytest.raises(ValueError):
        sp.record_search(t_ack=1.0)


def test_record_add_fields():
    rec = sp.record_add(
        t_ack=6.0, t_screen=1.0, t_tx=3.0, t_enqueue=0.5
    )
    assert rec.kind == "add"
    assert rec.fields["t_ack"] == 6.0
    # async fields absent until the post-ack signal arrives
    assert "t_visible" not in rec.fields
    assert "t_t0" not in rec.fields
    rec2 = sp.record_add(t_ack=6.0, t_visible=12.0, t_t0=30.0)
    assert rec2.fields["t_visible"] == 12.0


# ---------------------------------------------------------------------------
# ProfiledCall
# ---------------------------------------------------------------------------


def test_profiled_call_marks_lanes_counters():
    clk = FakeClock()
    sql, snaps = CountBox(), CountBox()
    with sp.ProfiledCall(
        "search", clock=clk, sql_counter=sql, snapshot_counter=snaps
    ) as pc:
        clk.advance(0.002)  # pre-first-mark time: honest residual
        pc.mark("analyze")
        clk.advance(0.010)
        pc.mark_lane("lexical")
        clk.advance(0.020)
        pc.mark("union")  # closes the lexical interval
        clk.advance(0.004)  # union accrues mark -> exit
        sql.count += 7
        snaps.count += 1
        pc.set_candidates("lexical", 23)
        pc.set_field("status", "ready")
    rec = pc.record
    assert rec.fields["t_analyze"] == pytest.approx(10.0)
    assert rec.t_lane["lexical"] == pytest.approx(20.0)
    assert rec.fields["t_union"] == pytest.approx(4.0)
    assert rec.fields["t_total"] == pytest.approx(36.0)
    assert rec.fields["sql_statements"] == 7
    assert rec.fields["snapshots"] == 1
    assert rec.candidates == {"lexical": 23}
    assert rec.fields["status"] == "ready"


def test_profiled_call_repeated_marks_accumulate():
    clk = FakeClock()
    with sp.ProfiledCall("search", clock=clk) as pc:
        pc.mark_lane("dense")
        clk.advance(0.001)
        pc.mark("pack")
        clk.advance(0.002)
        pc.mark_lane("dense")  # a second dense interval adds, not replaces
        clk.advance(0.003)
    rec = pc.record
    assert rec.t_lane["dense"] == pytest.approx(4.0)
    assert rec.fields["t_pack"] == pytest.approx(2.0)
    assert rec.fields["t_total"] == pytest.approx(6.0)


def test_profiled_call_add_ack_and_async_fields():
    clk = FakeClock()
    with sp.ProfiledCall("add", clock=clk) as pc:
        clk.advance(0.002)
        pc.mark("screen")
        clk.advance(0.001)
        pc.mark("tx")
        clk.advance(0.003)  # tx accrues mark -> exit
    rec = pc.record
    # for adds the context's wall time is t_ack, not t_total
    assert rec.fields["t_ack"] == pytest.approx(6.0)
    assert "t_total" not in rec.fields
    assert rec.fields["t_screen"] == pytest.approx(1.0)
    assert rec.fields["t_tx"] == pytest.approx(3.0)
    # async fields filled post-exit, when the visibility signal lands
    pc.set_field("t_visible", 12.0)
    pc.set_field("t_t0", 30.0)
    assert pc.record is rec
    assert rec.fields["t_visible"] == 12.0
    assert rec.fields["t_t0"] == 30.0


def test_profiled_call_wraps_callable():
    clk = FakeClock()
    with sp.ProfiledCall("search", clock=clk) as pc:
        out = pc(lambda q: q * 2, 21)
    assert out == 42
    assert pc.record.kind == "search"


def test_profiled_call_static_and_callable_counters():
    clk = FakeClock()
    calls = {"n": 0}

    def polled() -> int:
        calls["n"] += 1
        return calls["n"] * 10

    with sp.ProfiledCall(
        "search",
        clock=clk,
        sql_counter=polled,  # callable: delta = exit - enter
        snapshot_counter=4,  # static number: recorded as absolute
    ) as pc:
        pass
    assert pc.record.fields["sql_statements"] == 10
    assert pc.record.fields["snapshots"] == 4


def test_profiled_call_rejects_unknown_and_reserved_stages():
    clk = FakeClock()
    with sp.ProfiledCall("search", clock=clk) as pc:
        with pytest.raises(ValueError):
            pc.mark("analize")  # typo can never mint an undeclared field
        with pytest.raises(ValueError):
            pc.mark("total")  # t_total is context-owned, not a stage
        with pytest.raises(ValueError):
            pc.mark("screen")  # add stage on a search record
        with pytest.raises(ValueError):
            pc.mark_lane("")  # lane names are non-empty
        with pytest.raises(ValueError):
            pc.set_field("nope", 1)
    # marks are only valid inside the context
    pc2 = sp.ProfiledCall("search")
    with pytest.raises(RuntimeError):
        pc2.mark("analyze")
    with pytest.raises(RuntimeError):
        _ = pc2.record
    with pytest.raises(ValueError):
        sp.ProfiledCall("bogus")  # kind validated at construction


def test_profiled_call_exception_still_finalizes():
    clk = FakeClock()
    pc = None
    with pytest.raises(RuntimeError):
        with sp.ProfiledCall("search", clock=clk) as pc:
            pc.mark("analyze")
            clk.advance(0.001)
            raise RuntimeError("boom")
    assert pc is not None
    rec = pc.record  # the record still exists and is finalized
    assert rec.fields["t_analyze"] == pytest.approx(1.0)
    assert rec.fields["t_total"] == pytest.approx(1.0)
    with pytest.raises(RuntimeError):
        pc.mark("analyze")  # closed after exit


def test_profiled_call_caller_total_kept():
    clk = FakeClock()
    with sp.ProfiledCall("search", clock=clk) as pc:
        pc.set_field("t_total", 999.0)  # externally measured total wins
        clk.advance(0.001)
    assert pc.record.fields["t_total"] == 999.0


# ---------------------------------------------------------------------------
# aggregate()
# ---------------------------------------------------------------------------


def test_aggregate_percentile_math():
    recs = [
        sp.record_search(
            t_total=v,
            sql_statements=3,
            status="ready" if i % 2 == 0 else "insufficient",
            coverage_digest="abc",
            t_lane={"lexical": v * 2},
            candidates={"lexical": i},
        )
        for i, v in enumerate([10, 20, 30, 40, 50])
    ]
    agg = sp.aggregate(recs)
    # n=5, v5 nearest-rank: idx = round(q * (n-1))
    assert agg["t_total"] == {
        "n": 5, "p50": 30.0, "p95": 50.0, "p99": 50.0,
        "mean": 30.0, "max": 50.0,
    }
    assert agg["t_lane.lexical"] == {
        "n": 5, "p50": 60.0, "p95": 100.0, "p99": 100.0,
        "mean": 60.0, "max": 100.0,
    }
    assert agg["candidates.lexical"]["n"] == 5
    assert agg["candidates.lexical"]["mean"] == 2.0
    assert agg["sql_statements"]["p50"] == 3.0
    # label fields aggregate to honest histograms, not percentiles
    assert agg["status"] == {
        "n": 5, "counts": {"insufficient": 2, "ready": 3}
    }
    # one digest bucket = deterministic coverage across the window
    assert agg["coverage_digest"] == {"n": 5, "counts": {"abc": 5}}
    # keys sorted for deterministic artifacts
    assert list(agg) == sorted(agg)


def test_aggregate_honest_denominators_and_empty():
    recs = [
        sp.record_search(t_total=10),
        sp.record_search(t_total=20, status="ready"),
    ]
    agg = sp.aggregate(recs)
    assert agg["t_total"]["n"] == 2
    # status only has one contributor — the denominator says so
    assert agg["status"] == {"n": 1, "counts": {"ready": 1}}
    assert "t_analyze" not in agg  # absent stays absent
    assert sp.aggregate([]) == {}


def test_aggregate_small_n_and_add_records():
    agg = sp.aggregate([sp.record_add(t_ack=6.0, t_screen=1.0)])
    assert agg["t_ack"] == {
        "n": 1, "p50": 6.0, "p95": 6.0, "p99": 6.0, "mean": 6.0, "max": 6.0
    }
    pair = sp.aggregate(
        [sp.record_add(t_ack=6.0), sp.record_add(t_ack=10.0)]
    )
    # n=2: p50 idx round(0.5)=1? no — round(0.5)=0 (banker's) -> idx 0
    assert pair["t_ack"]["p50"] == 6.0
    assert pair["t_ack"]["p95"] == 10.0
    assert pair["t_ack"]["mean"] == 8.0


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def _sample_records():
    return [
        sp.record_search(
            t_total=12.5,
            t_barrier=1.0,
            t_lane={"dense": 3.0},
            candidates={"dense": 17},
            status="ready",
            coverage_digest="deadbeef",
        ),
        sp.record_add(t_ack=6.0, t_tx=3.0),
    ]


def test_json_roundtrip_single_and_list():
    rec = _sample_records()[0]
    rec2 = sp.from_json(sp.to_json(rec))
    assert rec2 == rec
    assert rec2.kind == "search"
    assert rec2.t_lane["dense"] == 3.0

    recs = _sample_records()
    back = sp.from_json(sp.to_json(recs))
    assert isinstance(back, list) and back == recs


def test_to_json_deterministic_and_strict():
    rec = _sample_records()[0]
    assert sp.to_json(rec) == sp.to_json(sp.from_json(sp.to_json(rec)))
    doc = json.loads(sp.to_json(rec))
    assert doc["format"] == sp.RECORD_FORMAT == "stage_profile/v7"
    assert doc["kind"] == "search"
    # aggregate envelopes pass through to_json unchanged (already plain)
    agg = sp.aggregate(_sample_records())
    assert json.loads(sp.to_json(agg)) == agg


def test_from_json_rejects_non_records():
    with pytest.raises(ValueError):
        sp.from_json('{"t_total": 1.0}')  # no kind — not a record
    with pytest.raises(ValueError):
        sp.from_json('{"kind": "bogus"}')
    with pytest.raises(ValueError):
        # a hand-edited artifact with an undeclared field fails loudly
        sp.from_json('{"kind": "search", "fields": {"t_nope": 1}}')
    with pytest.raises(ValueError):
        sp.from_json("42")


# ---------------------------------------------------------------------------
# mutations.yaml — consumable by the SAME runner as eval/v4
# ---------------------------------------------------------------------------

# The 23 §35 mutation targets as <area>/<mutation> slugs — every row of
# the §35 "Mutation targets" column, in table order.
EXPECTED_TARGETS = [
    "analyzer/skip-clitic-split",
    "analyzer/fold-identifiers",
    "lexical/universe-stats",
    "lexical/skip-held-subtraction",
    "fuzzy/respell-identifiers",
    "dense/ann-without-eligibility-recheck",
    "entities/auto-apply-candidate-aliases",
    "graph/traverse-held-units",
    "temporal/drop-precision",
    "temporal/wrong-anchor",
    "events/unpinned-tuple-delivered",
    "fusion/constant-signal-weight",
    "fusion/unbounded-boost",
    "verdict/floor-based-deletion",
    "packs/computed-item-undelivered-input",
    "consolidation/stale-observation-alone",
    "standing/stale-pack-after-hold",
    "t2/accept-unverified-quote",
    "reflect/cite-undelivered-ref",
    "trust/skip-rescan",
    "trust/drop-blocked-silently",
    "performance/extra-snapshot-per-search",
    "eval/tripwire-bypass",
]


def test_mutations_yaml_loads_under_v4_runner():
    """The same ``tools/run_mutation.py load_mutations`` consumes the v7
    file unchanged: promoted targets load as runnable Mutation records and
    the loader ignores the ``pending:`` scaffold."""
    rm = _load_runner()
    muts = rm.load_mutations(MUTATIONS_YAML)
    assert muts, "no promoted mutants — the §35 set regressed to scaffold"
    ids = [m.id for m in muts]
    assert len(ids) == len(set(ids)), "duplicate runnable mutation ids"
    for m in muts:
        assert m.anchor and m.anchor != m.replacement, m.id
        assert m.killed_by, m.id


def test_mutations_yaml_pending_scaffold_is_wellformed():
    doc = yaml.safe_load(MUTATIONS_YAML.read_text(encoding="utf-8"))
    assert doc["version"] == 1
    assert doc["runner"] == "tools/run_mutation.py"
    promoted = doc["mutations"]
    pending = doc["pending"]
    assert isinstance(promoted, list) and isinstance(pending, list)
    # Promoted + pending together cover the complete §35 column, in table
    # order — no target may vanish from the declaration.
    all_ids = [p["id"] for p in promoted] + [p["id"] for p in pending]
    assert len(set(all_ids)) == len(all_ids), "duplicate ids across lists"
    by_id = {p["id"]: p for p in promoted + pending}
    targets = [by_id[pid]["target"] for pid in sorted(
        by_id, key=lambda i: int(i.rsplit("-", 1)[1]))]
    assert targets == EXPECTED_TARGETS  # table order, complete §35 column
    for p in promoted:
        assert p["id"].startswith("MUT-V7-")
        assert isinstance(p["anchor"], str) and p["anchor"]
        assert isinstance(p["replacement"], str)
        assert p["anchor"] != p["replacement"]
        assert isinstance(p["killed_by"], list) and p["killed_by"]
        assert isinstance(p["file"], str) and p["file"]
        assert isinstance(p["check"], str) and p["check"]
        assert "V7-35.01" in p["requirements"]
        assert any(r != "V7-35.01" for r in p["requirements"])
    for p in pending:
        assert p["id"].startswith("MUT-V7-")
        assert p["status"] == "pending_anchor"
        # schema-aligned placeholders: promotion fills these, nothing else
        assert p["anchor"] is None
        assert p["replacement"] is None
        assert p["killed_by"] == []
        assert isinstance(p["file"], str) and p["file"]
        assert isinstance(p["check"], str) and p["check"]
        # a pending entry must say *why* it cannot run yet
        assert isinstance(p.get("pending_reason"), str) \
            and p["pending_reason"].strip()
        assert "V7-35.01" in p["requirements"]
        # every target cites at least one behavioral V7 requirement
        assert any(r != "V7-35.01" for r in p["requirements"])
