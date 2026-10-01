"""S2 per-lane deadline slices — two-phase scheduler (V8-14.03, §21.2).

``allocate`` computes one phase's lane slices from a budget figure —
proportional to the declared per-lane cost (``LANE_COSTS_V1``, the §04.2
p95 slice bounds), floored at ``SLICE_FLOOR_MS``. V8-14.03 supersedes
the single-shot V7-05.05 allocation (V8-00.01a): the pipeline calls
``allocate`` once for the S2a core set against ``budget − R_post`` and
then re-derates each S2b expansion lane's slice from the budget
remaining *at its start* (:func:`expansion_slice_ms`), so headroom a
core lane leaves unspent flows to the expansion lanes that still need
it. Expansion lanes are need-gated (:func:`need_gate`) — a lane whose
inputs show no need reports ``skipped(not_needed)`` and consumes
nothing.

Allocation rule (``provisional/v7-r0``):

- ``remaining_ms <= 0`` → every requested, policy-enabled lane gets a
  ``0.0`` slice so it can report ``deadline`` honestly (V7-04.03).
- ``remaining <= n * SLICE_FLOOR_MS`` → pure cost-proportional split;
  the floor is unaffordable, so lanes get tiny honest slices and mostly
  degrade to ``deadline``.
- otherwise → ``floor + headroom * cost / Σcost`` per lane, capped at the
  lane's declared §04.2 budget (``LANE_COSTS_V1[lane]``); granting more
  would plan past the lane's own p95 contract. Capped surplus stays as
  headroom for later stages.

Invariants: slices sum to ``<= remaining_ms``; every slice is ``>= 0``;
the result contains exactly the requested lanes that ``policy.lanes``
enables (requesting a disabled lane is a policy violation — V7-05.02 —
and fails validation rather than silently running it).

``SliceBudget`` is the lane-side poller: lanes check ``exceeded()``
inside their scan loops and return ``PARTIAL`` with ``reason="deadline"``
plus honest ``examined``/``eligible`` counts when it fires (V7-04.03).
The clock is injectable so tests stay deterministic.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Iterable, Optional

from ...core.types import ErrorCode, VerbatimError
from ...core.types_v7 import (
    POOLS,
    BudgetClass,
    IntentClass,
    LaneName,
    LaneSlice,
    PoolProfile,
    QueryViewV7,
    RetrievalPolicyV7,
)
from .policy import DEFAULT_LANE_COST_MS, LANE_COSTS_V1, policy_param

#: Minimum slice a lane is granted when the budget can afford floors —
#: below this a lane cannot do meaningful work and degrades honestly.
SLICE_FLOOR_MS = 0.5

#: Per-lane bound under an unbounded request budget — effectively "no
#: meaningful deadline" while staying finite for the SliceWatch clock.
_UNBOUNDED_SLICE_MS = 3_600_000.0  # 1 hour

#: Candidate cap used when the caller does not pass the request's pool —
#: the mid-tier lane cap. The pipeline normally passes ``pool=`` from the
#: request's ``budget`` (V7-05.07); the default exists for tools/tests.
DEFAULT_SLICE_CAP = POOLS[BudgetClass.MID].lane_cap


def _fail(message: str) -> None:
    raise VerbatimError(ErrorCode.VALIDATION, message)


def _cost(lane: LaneName) -> float:
    return LANE_COSTS_V1.get(lane, DEFAULT_LANE_COST_MS)


def allocate(
    remaining_ms: float,
    lanes: Iterable[LaneName],
    policy: RetrievalPolicyV7,
    *,
    cap: Optional[int] = None,
    pool: Optional[PoolProfile] = None,
) -> dict[LaneName, LaneSlice]:
    """Compute per-lane ``LaneSlice`` values at S2 entry (V7-05.05).

    ``lanes`` is the caller's intended lane set (typically
    ``lanes_for(intent, policy)``); every requested lane must be enabled
    by ``policy.lanes`` — asking for a slice of a disabled lane fails
    ``VALIDATION`` because lane enablement lives only in the declared
    table (V7-05.02). ``cap``/``pool`` set the candidate cap on each
    slice; pass the request pool (``pool_for(budget)``) on the hot path.
    """

    rem = float(remaining_ms)
    if math.isnan(rem):
        _fail("remaining_ms must not be NaN")
    if rem < 0.0:
        rem = 0.0  # an overrun barrier degrades lanes to 0-slices honestly

    if cap is not None:
        cand_cap = cap
    elif pool is not None:
        cand_cap = pool.lane_cap
    else:
        cand_cap = DEFAULT_SLICE_CAP
    if cand_cap < 0:
        _fail(f"slice candidate cap must be >= 0, got {cand_cap!r}")

    enabled = set(policy.lanes)
    lane_list: list[LaneName] = []
    for raw in lanes:
        try:
            lane = raw if isinstance(raw, LaneName) else LaneName(str(raw))
        except ValueError:
            _fail(f"unknown lane for slicing: {raw!r}")
        if lane not in enabled:
            _fail(
                f"lane {lane.value!r} is not enabled by policy "
                f"{policy.policy_id!r} — enablement is the declared table "
                f"(V7-05.02)"
            )
        if lane not in lane_list:
            lane_list.append(lane)

    out: dict[LaneName, LaneSlice] = {}
    if not lane_list:
        return out
    if rem <= 0.0:
        for lane in lane_list:
            out[lane] = LaneSlice(deadline_ms=0.0, cap=cand_cap)
        return out

    total_cost = sum(_cost(lane) for lane in lane_list)
    if math.isinf(rem):
        # Unbounded budget: grant a generous per-lane bound. The declared
        # §04.2 costs are p95 *targets* at the reference scale, not work
        # ceilings — capping at them starves any corpus larger than the
        # calibrated point, so the bound is a large finite sentinel.
        for lane in lane_list:
            out[lane] = LaneSlice(deadline_ms=_UNBOUNDED_SLICE_MS, cap=cand_cap)
        return out

    n = len(lane_list)
    if rem <= n * SLICE_FLOOR_MS:
        # Starved budget — floors unaffordable: pure cost-proportional
        # split (sums to rem exactly; lanes degrade honestly).
        for lane in lane_list:
            share = rem * _cost(lane) / total_cost
            out[lane] = LaneSlice(deadline_ms=share, cap=cand_cap)
        return out

    headroom = rem - n * SLICE_FLOOR_MS
    for lane in lane_list:
        # ``_cost`` is the proportional weight, never a ceiling — a lane
        # capped at its declared p95 cannot finish on a larger corpus and
        # reports deadline/produced=0. The full share is the slice.
        share = SLICE_FLOOR_MS + headroom * _cost(lane) / total_cost
        out[lane] = LaneSlice(deadline_ms=share, cap=cand_cap)
    return out


class SliceBudget:
    """Wall-clock view of one ``LaneSlice`` for a lane to poll (V7-05.05).

    ``clock`` returns seconds (``time.monotonic`` by default); tests inject
    a fake. ``spent``/``remaining`` are milliseconds; ``exceeded()`` is
    true once the slice is consumed — a ``0.0`` slice is exceeded
    immediately, which is how a starved lane reports ``deadline``.
    """

    __slots__ = ("_limit_ms", "_cap", "_clock", "_start")

    def __init__(
        self,
        limit_ms: float,
        cap: int = 0,
        *,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._limit_ms = float(limit_ms)
        self._cap = int(cap)
        self._clock = clock if clock is not None else time.monotonic
        self._start = self._clock()

    @classmethod
    def from_slice(
        cls,
        slice_: LaneSlice,
        *,
        clock: Optional[Callable[[], float]] = None,
    ) -> "SliceBudget":
        return cls(slice_.deadline_ms, slice_.cap, clock=clock)

    @property
    def limit_ms(self) -> float:
        return self._limit_ms

    @property
    def cap(self) -> int:
        return self._cap

    def spent(self) -> float:
        """Milliseconds consumed since construction."""

        return max(0.0, (self._clock() - self._start) * 1000.0)

    def remaining(self) -> float:
        """Milliseconds left in the slice; negative means overdrawn."""

        return self._limit_ms - self.spent()

    def exceeded(self) -> bool:
        """True when the slice is fully consumed (``remaining <= 0``)."""

        return self.remaining() <= 0.0


# ---------------------------------------------------------------------------
# V8-14.03 — two-phase re-derated scheduling (§21.2 normative)
# ---------------------------------------------------------------------------

#: Phase S2a core lanes (§21.2 ``[lex, fuzzy, dense, ent]``). Sliced once
#: from ``budget − R_post``; fuzzy keeps its ``no_oov_terms`` self-skip
#: inside the lane.
CORE_LANES_V8: frozenset = frozenset(
    {LaneName.LEX, LaneName.FUZZY, LaneName.DENSE, LaneName.ENT}
)

#: Phase S2b expansion lanes in declaration order (§21.2
#: ``[time, typed, obs, graph]``). Each is need-gated; the run order is
#: ``(LANE_COSTS_V1[lane], name)`` — see :func:`expansion_order`.
EXPANSION_LANES_V8: tuple = (
    LaneName.TIME,
    LaneName.TYPED,
    LaneName.OBS,
    LaneName.GRAPH,
)
EXPANSION_SET_V8: frozenset = frozenset(EXPANSION_LANES_V8)

#: ``scheduler.R_post`` arm prior (§23): reserved post-lane budget —
#: 60 ms at the 500 ms reference profile. Lanes are sliced against
#: ``budget − R_post`` so fusion/rerank/verdict/pack always have room.
DEFAULT_R_POST_MS = 60.0

#: ``scheduler.post_pool`` arm prior (§23, V8-14.05): the fused pool
#: handed to rerank is trimmed to ``max(POST_POOL_LIMIT_MULT × limit,
#: POST_POOL_MIN)``.
POST_POOL_LIMIT_MULT = 4
POST_POOL_MIN = 64

#: Per-post-stage p95 estimates in ms (V8-14.04) — when the budget
#: remaining at a post stage's start falls below its estimate, the stage
#: runs its degraded variant instead of risking the overrun. r0
#: provisional estimates pending the V8-14.09 profile artifact; the
#: hard floor is always ``remaining <= 0`` → the cheapest honest variant.
POST_STAGE_P95_MS: dict = {
    "rerank_feat": 30.0,   # provider-backed feature lookups are the cost
    "rerank_ce": 40.0,     # optional cross-encoder hook
    "boost": 5.0,          # pure arithmetic — skipped only at the floor
    "verdict": 20.0,       # group classification → structural fallback
    "explain": 10.0,       # per-item payload → optional fields dropped
}


def resolve_r_post(policy: RetrievalPolicyV7) -> float:
    """Effective ``scheduler.R_post`` reservation in ms (V8-14.03).

    ``None``/absent → :data:`DEFAULT_R_POST_MS`. A declared value must be
    a finite number ≥ 0 — a malformed reservation is a policy defect and
    fails validation, like any other bad table entry (V7-05.02).
    """

    raw = policy_param(policy, "scheduler.R_post")
    if raw is None:
        return DEFAULT_R_POST_MS
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        _fail(f"scheduler.R_post must be a number, got {raw!r}")
    r_post = float(raw)
    if not math.isfinite(r_post) or r_post < 0.0:
        _fail(f"scheduler.R_post must be finite and >= 0, got {raw!r}")
    return r_post


def resolve_post_pool(policy: RetrievalPolicyV7, limit: int) -> int:
    """Effective ``scheduler.post_pool`` bound (V8-14.05).

    Absent → ``max(4 × limit, 64)`` (§23 prior). A declared value must be
    an int ≥ 1.
    """

    raw = policy_param(policy, "scheduler.post_pool")
    if raw is None:
        return max(POST_POOL_LIMIT_MULT * int(limit), POST_POOL_MIN)
    if isinstance(raw, bool) or not isinstance(raw, int):
        _fail(f"scheduler.post_pool must be an int, got {raw!r}")
    if raw < 1:
        _fail(f"scheduler.post_pool must be >= 1, got {raw!r}")
    return int(raw)


def graph_gate_enabled(policy: RetrievalPolicyV7) -> bool:
    """``graph.gate`` flag (§23, V8-05.05): need-gated unless the arm
    disabled the gate. Default ON — the §05 exit measurement keeps graph
    need-gated unless its ablation proves ungated wins."""

    raw = policy_param(policy, "graph.gate")
    if raw is None:
        return True
    if not isinstance(raw, bool):
        _fail(f"graph.gate must be a bool, got {raw!r}")
    return raw


def two_phase_enabled(policy: RetrievalPolicyV7) -> bool:
    """``scheduler.two_phase`` flag (V85-05.07): the §21.2 two-phase
    schedule (S2a core slices + need-gated re-derated S2b expansion)
    unless the arm selects single-phase — every enabled lane sliced once
    from ``budget − R_post``, in policy order, with no need gate (the
    pre-V8-14.03 shape; each lane's own in-lane skip still applies).

    Default ON — the flag is kept only while a paired run shows any@10
    drops no more than 0.005 with zero over-deadline queries at the
    500 ms envelope (the c3 measurement found 17–45 overshoots). A
    declared non-bool is a malformed table entry and fails validation
    (V7-05.02)."""

    raw = policy_param(policy, "scheduler.two_phase")
    if raw is None:
        return True
    if not isinstance(raw, bool):
        _fail(f"scheduler.two_phase must be a bool, got {raw!r}")
    return raw


def expansion_order(lanes: Iterable) -> list:
    """S2b run order (§21.2): ``(LANE_COSTS_V1[lane], name)`` ascending —
    cheap lanes first, name as the deterministic tie-break."""

    return sorted(
        lanes, key=lambda lane: (_cost(lane), str(getattr(lane, "value", lane)))
    )


def expansion_slice_ms(rem_ms: float, lane: LaneName, pending: Iterable) -> float:
    """One expansion lane's re-derated slice (§21.2):

    ``slice = rem · cost(l) / Σ_{m in exp not yet run} cost(m)``

    ``pending`` must contain ``lane`` plus every expansion lane still
    waiting — the last lane's share is the whole remaining budget. The
    caller floors at ``SLICE_FLOOR_MS`` and never calls this with
    ``rem <= 0``; non-finite budgets degrade to the unbounded sentinel
    (mirroring ``allocate``)."""

    pending = list(pending)
    total = sum(_cost(m) for m in pending)
    if total <= 0.0:
        return float(rem_ms)
    share = float(rem_ms) * _cost(lane) / total
    if math.isnan(share):
        _fail("expansion slice computed NaN")
    return share if math.isfinite(share) else _UNBOUNDED_SLICE_MS


# --- need gates (§21.2 gate rules) ---------------------------------------
#
# The intent sets below mirror each lane's *own* rule so the scheduler's
# verdict matches what the lane would decide itself — the gate exists to
# spend zero budget on a lane that would only self-skip.

#: ``time`` — parsed window or temporal intent. Mirrors temporal.py's
#: ``_TEMPORAL_INTENTS`` (the lane's own ``no_window`` skip condition).
_TIME_INTENTS = frozenset(
    {
        IntentClass.TEMPORAL_POINT,
        IntentClass.TEMPORAL_ORDER,
        IntentClass.DURATION,
        IntentClass.COUNT_AGGREGATE,
        IntentClass.TEMPORAL_RANGE,
        IntentClass.HISTORY_OF,
        IntentClass.CURRENT_VALUE,
    }
)

#: ``typed`` — the lane's own ``intent_not_typed`` rule (typed.py
#: ``_TYPED_INTENTS``): fact-shaped intents only.
_TYPED_INTENTS = frozenset(
    {
        IntentClass.CURRENT_VALUE,
        IntentClass.HISTORY_OF,
        IntentClass.PREFERENCE,
        IntentClass.COMPARISON,
        IntentClass.OPEN_DOMAIN,
        IntentClass.TEMPORAL_POINT,
        IntentClass.TEMPORAL_RANGE,
    }
)

#: ``obs`` — the lane's own intent rule (obs.py): never identifier-only,
#: never temporal-primary / temporal-only classes.
_OBS_IDENTIFIER_ONLY = frozenset({IntentClass.IDENTIFIER})
_OBS_TEMPORAL = frozenset(
    {
        IntentClass.TEMPORAL_POINT,
        IntentClass.TEMPORAL_ORDER,
        IntentClass.DURATION,
        IntentClass.COUNT_AGGREGATE,
        IntentClass.TEMPORAL_RANGE,
        IntentClass.HISTORY_OF,
        IntentClass.CURRENT_VALUE,
    }
)

#: ``graph`` — V8-05.05(a): multi-hop or comparison intent.
_GRAPH_INTENTS = frozenset({IntentClass.MULTI_HOP, IntentClass.COMPARISON})


def _gate_intent_classes(query: QueryViewV7) -> set:
    intent = getattr(query, "intent", None)
    if intent is None:
        return set()
    classes = set(getattr(intent, "classes", ()) or ())
    primary = getattr(intent, "primary", None)
    if primary is not None:
        classes.add(primary)
    return classes


def _gate_has_window(query: QueryViewV7) -> bool:
    window = getattr(getattr(query, "intent", None), "window", None)
    if window is None:
        return False
    return (
        getattr(window, "start_us", None) is not None
        or getattr(window, "end_us", None) is not None
    )


def need_gate(
    lane: LaneName,
    query: QueryViewV7,
    *,
    core_union: int,
    union_floor: int,
    requested: frozenset = frozenset(),
    graph_gate: bool = True,
) -> tuple:
    """The §21.2 need gate for one expansion lane.

    Returns ``(needed, inputs)`` — ``inputs`` is the plain-JSON dict of
    every signal the verdict consumed, logged verbatim in
    ``coverage.lanes.<lane>.gate.inputs`` (V8-05.05's "gate inputs are
    logged in explain"). A lane gated out reports
    ``skipped(not_needed)`` and costs nothing.

    - ``time``: a parsed window or a temporal intent (the lane's own
      ``no_window`` rule).
    - ``typed``: a fact-shaped intent (the lane's own
      ``intent_not_typed`` rule).
    - ``obs``: the lane's own intent rule *and* the S2a core union below
      ``union_floor`` (= 4 × limit) — obs is a rescue lane.
    - ``graph``: V8-05.05 — (a) multi-hop/comparison intent or ≥ 2 entity
      canons, (b) the core union below ``union_floor``, or (c) the caller
      requested ``lanes=["graph", …]`` (``ctx.manifest["lanes"]``). The
      ``graph.gate`` flag off → always needed, inputs still reported.
    """

    lane = LaneName(lane)
    classes = _gate_intent_classes(query)
    core_union = int(core_union)
    union_floor = int(union_floor)

    if lane is LaneName.TIME:
        window = _gate_has_window(query)
        temporal = bool(classes & _TIME_INTENTS)
        return (
            bool(window or temporal),
            {"window": window, "temporal_intent": temporal},
        )

    if lane is LaneName.TYPED:
        typed = bool(classes & _TYPED_INTENTS)
        return typed, {"typed_intent": typed}

    if lane is LaneName.OBS:
        intent = getattr(query, "intent", None)
        primary = getattr(intent, "primary", None)
        identifier_only = bool(classes) and classes <= _OBS_IDENTIFIER_ONLY
        temporal = (primary in _OBS_TEMPORAL) or (
            bool(classes) and classes <= _OBS_TEMPORAL
        )
        intent_ok = not identifier_only and not temporal
        below = core_union < union_floor
        inputs = {
            "intent_ok": intent_ok,
            "identifier_only": identifier_only,
            "temporal": temporal,
            "core_union": core_union,
            "union_below": below,
            "union_floor": union_floor,
        }
        return bool(intent_ok and below), inputs

    if lane is LaneName.GRAPH:
        multi = bool(classes & _GRAPH_INTENTS)
        canons = len(getattr(query, "entity_canons", ()) or ())
        many_canons = canons >= 2
        below = core_union < union_floor
        requested_flag = lane.value in set(requested or ())
        inputs = {
            "multihop_or_comparison": multi,
            "entity_canons": canons,
            "entity_canons_multi": many_canons,
            "core_union": core_union,
            "union_below": below,
            "union_floor": union_floor,
            "requested": requested_flag,
            "gate_flag": "on" if graph_gate else "off",
        }
        if not graph_gate:
            # V8-05.05's flag arm: gate disabled → the lane always runs,
            # inputs still disclosed (declared-but-dead flags are defects).
            return True, inputs
        return bool(multi or many_canons or below or requested_flag), inputs

    # No declared gate rule — run the lane (conservative: a lane without a
    # spec'd gate is never starved by a guess).
    return True, {"gate": "no_rule"}


__all__ = [
    "CORE_LANES_V8",
    "DEFAULT_R_POST_MS",
    "DEFAULT_SLICE_CAP",
    "EXPANSION_LANES_V8",
    "EXPANSION_SET_V8",
    "POST_POOL_LIMIT_MULT",
    "POST_POOL_MIN",
    "POST_STAGE_P95_MS",
    "SLICE_FLOOR_MS",
    "SliceBudget",
    "allocate",
    "expansion_order",
    "expansion_slice_ms",
    "graph_gate_enabled",
    "need_gate",
    "resolve_post_pool",
    "resolve_r_post",
    "two_phase_enabled",
]
