"""w-policy tests — `retrieval_policy/v7` table + S2 deadline slices.

Covers SPEC_V7 V7-05.02 (declared policy table), V7-05.05 (per-lane
slices from remaining budget, no sequential re-derating), V7-05.07 (pool
caps per budget), V7-05.09 (ablation switch), and the §32.3 weight/pool
tables verbatim under their `provisional/v7-r0` stamp (V7-32.01).
"""

from __future__ import annotations

import json
import math

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v7 import (
    FORMULA_STATUS_PROVISIONAL,
    POOLS,
    BudgetClass,
    IntentClass,
    LaneName,
    LaneSlice,
    PoolProfile,
)
from verbatim.retrieval.v7 import deadline as dl
from verbatim.retrieval.v7 import policy as pol

# §32.3 column order — kept independent of the module constant on purpose:
# this list is the test's own transcription of the spec table.
_SPEC_LANES = [
    LaneName.LEX,
    LaneName.FUZZY,
    LaneName.DENSE,
    LaneName.ENT,
    LaneName.TIME,
    LaneName.GRAPH,
    LaneName.TYPED,
    LaneName.OBS,
]

# §32.3 `retrieval_policy/v7` rows, verbatim (independent transcription).
_SPEC_WEIGHTS = {
    IntentClass.LOOKUP: [1.0, 0.3, 1.0, 0.8, 0.3, 0.5, 0.8, 0.5],
    IntentClass.IDENTIFIER: [1.5, 0.2, 0.3, 1.0, 0.1, 0.2, 1.0, 0.2],
    IntentClass.TEMPORAL_POINT: [0.8, 0.2, 0.8, 0.8, 1.5, 0.4, 0.8, 0.3],
    IntentClass.TEMPORAL_RANGE: [0.8, 0.2, 0.8, 0.8, 1.5, 0.4, 0.8, 0.3],
    IntentClass.TEMPORAL_ORDER: [0.8, 0.2, 0.8, 0.8, 1.5, 0.4, 0.8, 0.3],
    IntentClass.DURATION: [0.8, 0.2, 0.8, 0.8, 1.5, 0.4, 0.8, 0.3],
    IntentClass.COUNT_AGGREGATE: [1.0, 0.2, 0.8, 1.0, 0.8, 0.6, 0.8, 0.6],
    IntentClass.CURRENT_VALUE: [0.9, 0.2, 0.8, 1.0, 0.8, 0.4, 1.0, 0.8],
    IntentClass.HISTORY_OF: [0.9, 0.2, 0.8, 1.0, 0.8, 0.4, 1.0, 0.8],
    IntentClass.PREFERENCE: [0.7, 0.2, 1.2, 0.6, 0.3, 0.4, 0.8, 1.2],
    IntentClass.MULTI_HOP: [1.0, 0.3, 1.0, 1.0, 0.5, 1.2, 0.8, 0.6],
    IntentClass.COMPARISON: [1.0, 0.3, 1.0, 1.0, 0.5, 1.2, 0.8, 0.6],
    IntentClass.OPEN_DOMAIN: [0.8, 0.3, 1.4, 0.6, 0.3, 0.8, 0.6, 1.0],
    IntentClass.WHY_CAUSAL: [0.9, 0.2, 1.0, 0.8, 0.4, 1.2, 0.6, 0.6],
}


def _policy() -> object:
    return pol.load_policy("local_memory")


def _policy_all() -> object:
    """A policy enabling all eight §32.3 lanes — the pre-V85-05.01
    default set — for tests that exercise non-default lanes directly."""
    return pol.load_policy(
        "local_memory", {"lanes": [lane.value for lane in pol.LANES_V1]})


# ---------------------------------------------------------------------------
# §32.3 lane-weight table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "intent,expected",
    sorted(_SPEC_WEIGHTS.items(), key=lambda kv: kv[0].value),
)
def test_lane_weights_match_spec_table(intent, expected):
    row = pol.LANE_WEIGHTS_V1[intent]
    assert row == dict(zip(_SPEC_LANES, expected))
    assert list(row) == _SPEC_LANES  # declared column order preserved


def test_weight_table_covers_exactly_the_spec_rows():
    assert set(pol.LANE_WEIGHTS_V1) == set(_SPEC_WEIGHTS)
    # abstain_likely has no §32.3 row -> default-weight fallback.
    assert IntentClass.ABSTAIN_LIKELY not in pol.LANE_WEIGHTS_V1
    # exact_id / source have no §32.3 column.
    for row in pol.LANE_WEIGHTS_V1.values():
        assert LaneName.EXACT_ID not in row
        assert LaneName.SOURCE not in row
        assert len(row) == 8


def test_grouped_rows_share_values():
    w = pol.LANE_WEIGHTS_V1
    temporal = w[IntentClass.TEMPORAL_POINT]
    for ic in (
        IntentClass.TEMPORAL_RANGE,
        IntentClass.TEMPORAL_ORDER,
        IntentClass.DURATION,
    ):
        assert w[ic] == temporal
    assert w[IntentClass.CURRENT_VALUE] == w[IntentClass.HISTORY_OF]
    assert w[IntentClass.MULTI_HOP] == w[IntentClass.COMPARISON]
    # grouped members own independent dicts — no shared mutable rows
    assert w[IntentClass.TEMPORAL_POINT] is not w[IntentClass.DURATION]


def test_lane_costs_match_04_2_budgets():
    # Beat-it r2 recalibration: the 2–6ms table under-declared the heavy
    # lanes ~30–50× against measured §04.2 costs (~150ms lex/dense at 7k
    # units), starving them mid-scan inside a 500ms deadline.
    assert pol.LANE_COSTS_V1 == {
        LaneName.LEX: 200.0,
        LaneName.FUZZY: 25.0,
        LaneName.DENSE: 200.0,
        LaneName.ENT: 20.0,
        LaneName.TIME: 15.0,
        LaneName.GRAPH: 40.0,
        LaneName.TYPED: 25.0,
        LaneName.OBS: 15.0,
    }


# ---------------------------------------------------------------------------
# Pools (V7-05.07, §32.3 pools rows)
# ---------------------------------------------------------------------------


def test_pool_profiles_exact():
    assert pol.pool_for(BudgetClass.LOW) == PoolProfile(50, 40, 0, 0, 1)
    assert pol.pool_for(BudgetClass.MID) == PoolProfile(200, 100, 16, 1, 2)
    assert pol.pool_for(BudgetClass.HIGH) == PoolProfile(800, 300, 32, 2, 3)
    # the frozen contract POOLS table agrees
    assert pol.pool_for("low") is POOLS[BudgetClass.LOW]
    assert pol.pool_for("mid") is POOLS[BudgetClass.MID]
    assert pol.pool_for("high") is POOLS[BudgetClass.HIGH]


def test_pool_for_unknown_budget_rejected():
    with pytest.raises(VerbatimError) as exc:
        pol.pool_for("extreme")
    assert exc.value.code == ErrorCode.VALIDATION


# ---------------------------------------------------------------------------
# load_policy (V7-05.02)
# ---------------------------------------------------------------------------


def test_load_policy_defaults():
    p = _policy()
    assert p.policy_id == "retrieval_policy/v7"
    assert p.profile == "local_memory"
    # V85-05.01: the default enablement is the measured ship set —
    # (lex, fuzzy, dense, time); LANES_V1 still names the full §32.3
    # eight the weight/cost tables are written in.
    assert p.lanes == pol.LANES_V85 == (
        LaneName.LEX, LaneName.FUZZY, LaneName.DENSE, LaneName.TIME)
    assert tuple(_SPEC_LANES) == pol.LANES_V1
    assert p.formula_status == FORMULA_STATUS_PROVISIONAL == "provisional/v7-r0"
    # materialized table: every intent x every enabled lane
    assert set(p.lane_weights) == set(IntentClass)
    for intent in IntentClass:
        assert set(p.lane_weights[intent]) == set(p.lanes)
    # V75-03.03: pre-O9 default is flat 1.0 — the §32.3 table is the
    # named arm `retrieval_policy/v7-intent-weighted`, not the default.
    for intent in IntentClass:
        assert all(
            w == 1.0 for w in p.lane_weights[intent].values()
        ), intent
    # the named arm still materializes the §32.3 rows verbatim — here
    # with the full eight declared so every column lands
    arm = pol.load_policy(
        "local_memory",
        {"policy_id": pol.POLICY_ID_INTENT_WEIGHTED,
         "lanes": [lane.value for lane in _SPEC_LANES]},
    )
    for intent, expected in _SPEC_WEIGHTS.items():
        assert arm.lane_weights[intent] == dict(zip(_SPEC_LANES, expected))


def test_load_policy_overrides_lanes():
    p = pol.load_policy("local_memory", {"lanes": ["lex", "dense"]})
    assert p.lanes == (LaneName.LEX, LaneName.DENSE)
    # weight rows shrink to the enabled set, keeping spec values
    assert p.lane_weights[IntentClass.LOOKUP] == {
        LaneName.LEX: 1.0,
        LaneName.DENSE: 1.0,
    }


def test_load_policy_overrides_weights_merge():
    # ``obs`` sits outside the V85-05.01 default tuple — declare it so
    # its weight override materializes (weights on disabled lanes are
    # validated but inert by contract).
    p = pol.load_policy(
        "local_memory",
        {"lanes": ["lex", "fuzzy", "dense", "time", "obs"],
         "lane_weights": {"lookup": {"dense": 2.0}, "abstain_likely": {"obs": 0.4}}},
    )
    assert p.lane_weights[IntentClass.LOOKUP][LaneName.DENSE] == 2.0
    # untouched cells keep the §32.3 value
    assert p.lane_weights[IntentClass.LOOKUP][LaneName.LEX] == 1.0
    # override lands on an intent with no default row
    assert p.lane_weights[IntentClass.ABSTAIN_LIKELY][LaneName.OBS] == 0.4
    assert p.lane_weights[IntentClass.ABSTAIN_LIKELY][LaneName.LEX] == 1.0
    # the module-level r0 table is untouched
    assert pol.LANE_WEIGHTS_V1[IntentClass.LOOKUP][LaneName.DENSE] == 1.0


def test_load_policy_profile_overlay():
    doc = {
        "lane_weights": {"lookup": {"lex": 1.4}},
        "profiles": {
            "local_memory": {"lanes": ["lex", "ent"]},
            "local_memory_max": {"lane_weights": {"lookup": {"dense": 1.6}}},
        },
    }
    p = pol.load_policy("local_memory", doc)
    assert p.lanes == (LaneName.LEX, LaneName.ENT)
    assert p.lane_weights[IntentClass.LOOKUP][LaneName.LEX] == 1.4
    p2 = pol.load_policy("local_memory_max", doc)
    assert p2.lanes == pol.LANES_V85
    assert p2.lane_weights[IntentClass.LOOKUP][LaneName.DENSE] == 1.6
    assert p2.lane_weights[IntentClass.LOOKUP][LaneName.LEX] == 1.4
    p3 = pol.load_policy("other_profile", doc)
    assert p3.lanes == pol.LANES_V85
    assert p3.lane_weights[IntentClass.LOOKUP][LaneName.LEX] == 1.4


def test_load_policy_custom_id_and_status():
    p = pol.load_policy(
        "local_memory",
        {"policy_id": "retrieval_policy/v9", "formula_status": "selected/v9"},
    )
    assert p.policy_id == "retrieval_policy/v9"
    assert p.formula_status == "selected/v9"


@pytest.mark.parametrize(
    "doc",
    [
        {"lanes": ["lex", "bogus"]},
        {"lanes": []},
        {"lanes": ["lex", "lex"]},
        {"lanes": "lex"},
        {"lane_weights": {"bogus_intent": {"lex": 1.0}}},
        {"lane_weights": {"lookup": {"bogus": 1.0}}},
        {"lane_weights": {"lookup": {"lex": -0.5}}},
        {"lane_weights": {"lookup": {"lex": float("nan")}}},
        {"lane_weights": {"lookup": {"lex": float("inf")}}},
        {"lane_weights": {"lookup": {"lex": True}}},
        {"lane_weights": {"lookup": {"lex": "high"}}},
        {"lane_weights": {"lookup": [1.0]}},
        {"lane_weights": [("lookup", {})]},
        {"policy_id": ""},
        {"formula_status": 7},
        {"nonsense": {}},
        {"profiles": {"local_memory": {"bogus_key": []}}},
        {"profiles": {"local_memory": ["lex"]}},
        {"profiles": ["local_memory"]},
    ],
)
def test_load_policy_rejects_bad_tables(doc):
    with pytest.raises(VerbatimError) as exc:
        pol.load_policy("local_memory", doc)
    assert exc.value.code == ErrorCode.VALIDATION


def test_load_policy_rejects_non_dict_and_empty_profile():
    with pytest.raises(VerbatimError):
        pol.load_policy("local_memory", [("lanes", ["lex"])])
    with pytest.raises(VerbatimError):
        pol.load_policy("")
    with pytest.raises(VerbatimError):
        pol.load_policy("   ")


# ---------------------------------------------------------------------------
# lanes_for / weights_for / ablation (V7-05.01/09/12)
# ---------------------------------------------------------------------------


def test_lanes_for_returns_enabled_set_for_every_intent():
    p = _policy()
    for intent in IntentClass:
        assert pol.lanes_for(intent, p) == p.lanes
    # unparseable intent still returns the declared set
    assert pol.lanes_for("not-an-intent", p) == p.lanes


def test_weights_for_known_and_unknown_intent():
    p = _policy()
    default_row = {lane: 1.0 for lane in p.lanes}
    # V75-03.03: pre-O9 default is flat for every intent
    assert pol.weights_for(IntentClass.IDENTIFIER, p) == default_row
    # intent class with no §32.3 row
    assert pol.weights_for(IntentClass.ABSTAIN_LIKELY, p) == default_row
    # unrecognized intent value -> defaults, never a crash
    assert pol.weights_for("totally_unknown", p) == default_row
    # the §32.3 rows resolve under the named arm only
    arm = pol.load_policy(
        "local_memory", {"policy_id": pol.POLICY_ID_INTENT_WEIGHTED}
    )
    assert pol.weights_for("identifier", arm)[LaneName.LEX] == 1.5


def test_ablation_removes_lanes():
    p = _policy_all()   # graph/dense both enabled → both real ablations
    ablated = pol.ablation_lanes(p, ["dense", LaneName.GRAPH])
    assert LaneName.DENSE not in ablated.lanes
    assert LaneName.GRAPH not in ablated.lanes
    assert ablated.lanes == tuple(
        lane for lane in p.lanes if lane not in {LaneName.DENSE, LaneName.GRAPH}
    )
    # weight rows no longer carry the ablated lanes
    for row in ablated.lane_weights.values():
        assert LaneName.DENSE not in row
        assert LaneName.GRAPH not in row
    # ... so fusion weights and slice sets exclude them
    assert LaneName.DENSE not in pol.weights_for(IntentClass.LOOKUP, ablated)
    assert LaneName.DENSE not in pol.lanes_for(IntentClass.LOOKUP, ablated)
    # original frozen policy unchanged
    assert LaneName.DENSE in p.lanes
    # ablation keeps the table's identity + stamp
    assert ablated.policy_id == p.policy_id
    assert ablated.formula_status == FORMULA_STATUS_PROVISIONAL


def test_ablation_unknown_lane_rejected_and_noop_is_safe():
    p = _policy()
    with pytest.raises(VerbatimError):
        pol.ablation_lanes(p, ["dence"])  # typo must not silently no-op
    # disabling a valid-but-not-enabled lane is a no-op
    same = pol.ablation_lanes(p, [LaneName.EXACT_ID])
    assert same.lanes == p.lanes


# ---------------------------------------------------------------------------
# allocate (V7-05.05)
# ---------------------------------------------------------------------------


def test_allocate_proportional_to_declared_cost():
    p = _policy_all()   # ent is outside the V85-05.01 default tuple
    # two lanes, costs 200 + 20 = 220; headroom (4.0 - 2*0.5 = 3.0) splits
    # proportionally to the declared §04.2 costs (~10:1 lex over ent)
    out = dl.allocate(4.0, [LaneName.LEX, LaneName.ENT], p)
    assert out[LaneName.LEX].deadline_ms == pytest.approx(0.5 + 3.0 * 200 / 220)
    assert out[LaneName.ENT].deadline_ms == pytest.approx(0.5 + 3.0 * 20 / 220)
    # proportionality holds for the headroom component
    lex_over = out[LaneName.LEX].deadline_ms - dl.SLICE_FLOOR_MS
    ent_over = out[LaneName.ENT].deadline_ms - dl.SLICE_FLOOR_MS
    assert lex_over / ent_over == pytest.approx(200.0 / 20.0)


def test_allocate_floor_plus_proportional_headroom():
    p = _policy_all()
    total = sum(pol.LANE_COSTS_V1.values())  # 540 ms of declared cost
    out = dl.allocate(total / 2, list(pol.LANES_V1), p)
    for lane in pol.LANES_V1:
        assert out[lane].deadline_ms == pytest.approx(
            dl.SLICE_FLOOR_MS
            + (total / 2 - 8 * dl.SLICE_FLOOR_MS)
            * pol.LANE_COSTS_V1[lane]
            / total
        )


def test_allocate_floor_respected_when_affordable():
    p = _policy_all()
    out = dl.allocate(10.0, list(pol.LANES_V1), p)
    assert all(s.deadline_ms >= dl.SLICE_FLOOR_MS for s in out.values())


def test_allocate_starved_budget_splits_proportionally():
    p = _policy_all()   # ent is outside the V85-05.01 default tuple
    # 2 lanes * 0.5 floor = 1.0 ms; a 0.9 ms budget cannot afford floors
    out = dl.allocate(0.9, [LaneName.LEX, LaneName.ENT], p)
    assert out[LaneName.LEX].deadline_ms == pytest.approx(0.9 * 200 / 220)
    assert out[LaneName.ENT].deadline_ms == pytest.approx(0.9 * 20 / 220)


@pytest.mark.parametrize("rem", [0.01, 0.5, 1.0, 4.0, 10.0, 16.5, 33.0, 100.0, 1000.0])
def test_allocate_sums_within_remaining(rem):
    p = _policy_all()
    out = dl.allocate(rem, list(pol.LANES_V1), p)
    total = sum(s.deadline_ms for s in out.values())
    assert total <= rem + 1e-9
    assert all(s.deadline_ms >= 0.0 for s in out.values())


def test_allocate_proportional_share_not_capped_at_cost():
    p = _policy_all()
    out = dl.allocate(1000.0, list(pol.LANES_V1), p)
    # ``_cost`` is the proportional weight, NOT a ceiling: a lane capped at
    # its declared §04.2 p95 cannot finish on a corpus larger than the
    # reference scale and would report deadline/produced=0.  The share is
    # floor + headroom·cost/Σcost — deterministic, sums to <= remaining.
    headroom = 1000.0 - len(pol.LANES_V1) * dl.SLICE_FLOOR_MS
    total_cost = sum(pol.LANE_COSTS_V1[l] for l in pol.LANES_V1)
    for lane, s in out.items():
        expect = dl.SLICE_FLOOR_MS + headroom * pol.LANE_COSTS_V1[lane] / total_cost
        assert s.deadline_ms == pytest.approx(expect)
    assert sum(s.deadline_ms for s in out.values()) == pytest.approx(1000.0)


def test_allocate_proportional_boundary():
    p = _policy_all()
    # at rem=33: share = 0.5 + (33-4)·cost/540 — uncapped proportional.
    out = dl.allocate(33.0, list(pol.LANES_V1), p)
    assert out[LaneName.OBS].deadline_ms == pytest.approx(0.5 + 29.0 * 15 / 540)
    assert out[LaneName.LEX].deadline_ms == pytest.approx(0.5 + 29.0 * 200 / 540)
    assert sum(s.deadline_ms for s in out.values()) == pytest.approx(33.0)


def test_allocate_zero_and_negative_remaining():
    p = _policy_all()
    for rem in (0.0, -5.0):
        out = dl.allocate(rem, list(pol.LANES_V1), p)
        assert all(s.deadline_ms == 0.0 for s in out.values())


def test_allocate_infinite_remaining_gives_generous_bound():
    p = _policy_all()
    out = dl.allocate(math.inf, list(pol.LANES_V1), p)
    # Unbounded budget grants a large finite sentinel per lane — the
    # declared p95 costs are proportion weights, not ceilings, so an
    # "infinite" request does not starve a lane at 200 ms on a real corpus.
    for lane, s in out.items():
        assert s.deadline_ms == dl._UNBOUNDED_SLICE_MS


def test_allocate_nan_rejected():
    p = _policy_all()
    with pytest.raises(VerbatimError):
        dl.allocate(float("nan"), list(pol.LANES_V1), p)


def test_allocate_disabled_lane_absent_and_rejected():
    p = pol.ablation_lanes(_policy(), ["dense"])
    out = dl.allocate(20.0, list(p.lanes), p)
    assert LaneName.DENSE not in out
    assert set(out) == set(p.lanes)
    # asking for a slice of a disabled lane is a policy violation
    with pytest.raises(VerbatimError) as exc:
        dl.allocate(20.0, [LaneName.DENSE], p)
    assert exc.value.code == ErrorCode.VALIDATION
    with pytest.raises(VerbatimError):
        dl.allocate(20.0, ["not_a_lane"], p)


def test_allocate_candidate_cap_sources():
    p = _policy()
    # explicit pool wins
    out = dl.allocate(10.0, [LaneName.LEX], p, pool=pol.pool_for("high"))
    assert out[LaneName.LEX].cap == 800
    # explicit cap wins over nothing
    out = dl.allocate(10.0, [LaneName.LEX], p, cap=17)
    assert out[LaneName.LEX].cap == 17
    # default is the declared mid-pool lane cap
    out = dl.allocate(10.0, [LaneName.LEX], p)
    assert out[LaneName.LEX].cap == POOLS[BudgetClass.MID].lane_cap


def test_allocate_dedupes_and_preserves_request_order():
    p = _policy_all()   # ent is outside the V85-05.01 default tuple
    out = dl.allocate(10.0, [LaneName.ENT, LaneName.LEX, LaneName.ENT], p)
    assert list(out) == [LaneName.ENT, LaneName.LEX]


# ---------------------------------------------------------------------------
# SliceBudget polling
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_slice_budget_polls_clock():
    clock = _Clock()
    b = dl.SliceBudget(5.0, cap=50, clock=clock)
    assert b.cap == 50
    assert b.spent() == pytest.approx(0.0)
    assert b.remaining() == pytest.approx(5.0)
    assert not b.exceeded()
    clock.t += 0.003  # +3 ms
    assert b.spent() == pytest.approx(3.0)
    assert b.remaining() == pytest.approx(2.0)
    assert not b.exceeded()
    clock.t += 0.002  # ~5 ms total -> float noise sits at the boundary
    assert b.remaining() == pytest.approx(0.0, abs=1e-9)
    clock.t += 0.0005  # clearly past the slice
    assert b.exceeded()
    clock.t += 0.004  # overrun reads negative, honestly
    assert b.remaining() == pytest.approx(-4.5)
    assert b.spent() == pytest.approx(9.5)


def test_slice_budget_from_slice_and_zero_deadline():
    clock = _Clock()
    b = dl.SliceBudget.from_slice(LaneSlice(deadline_ms=2.0, cap=7), clock=clock)
    assert b.limit_ms == 2.0
    assert b.cap == 7
    zero = dl.SliceBudget.from_slice(LaneSlice(deadline_ms=0.0, cap=50), clock=clock)
    assert zero.exceeded()  # a starved lane reports deadline immediately


# ---------------------------------------------------------------------------
# coverage view + determinism
# ---------------------------------------------------------------------------


def test_policy_view_is_jsonable_and_complete():
    p = _policy()
    view = pol.policy_view(p)
    assert view["policy_id"] == "retrieval_policy/v7"
    assert view["profile"] == "local_memory"
    assert view["formula_status"] == "provisional/v7-r0"
    # V85-05.01 — the applied (default) table prints the ship set.
    assert view["lanes"] == ["lex", "fuzzy", "dense", "time"]
    assert view["lane_weights"]["lookup"]["lex"] == 1.0
    # JSON round-trips — the printed table is byte-stable coverage data
    assert json.loads(json.dumps(view)) == view


def test_determinism():
    p1 = _policy()
    p2 = _policy()
    p1_all = _policy_all()
    p2_all = _policy_all()
    assert p1 == p2
    assert p1_all == p2_all
    doc = {"lanes": ["lex", "ent", "obs"], "lane_weights": {"lookup": {"ent": 1.7}}}
    assert pol.load_policy("local_memory", doc) == pol.load_policy("local_memory", doc)
    a = dl.allocate(16.5, list(pol.LANES_V1), p1_all)
    b = dl.allocate(16.5, list(pol.LANES_V1), p2_all)
    assert a == b
    assert list(a) == list(pol.LANES_V1)  # stable ordering
