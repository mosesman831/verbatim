"""§04.2 read-path orchestrator — the one function family (V7-04.04).

``run_search`` is THE S1–S8 pipeline: ``Memory.search``, the service,
MCP, and every adapter route through here; no surface assembles its own
ranking (V7-04.04). Ownership: w-pipeline (``docs/v7_contracts.md``).

Stage map (§04.2 order; per-stage honesty per V7-04.03):

- **S0 barrier** — upstream: the caller pins the read snapshot, epoch,
  and eligibility handle before calling. ``ctx.store`` /
  ``ctx.generation`` / ``ctx.eligible`` arrive resolved. ``t_barrier``
  records ``0.0`` here by construction.
- **S1 analyze** — upstream: ``query`` arrives as a fully-analyzed
  :class:`QueryViewV7` (norm/v2 terms + identifier channel, intent/v2
  classes, entity canons, resolved temporal window, decomposition
  facets). ``t_analyze`` records ``0.0``.
- **S2 lanes** — EVERY lane enabled in ``ctx.policy.lanes`` runs as a
  peer with its own :class:`LaneSlice` computed once at S2 entry by
  ``deadline.allocate`` (equal-share fallback when the module is
  absent). There is no typed-thin gate — D7-09 is dead by construction —
  and no sequential re-derating — D7-14 is dead by construction: slices
  are dealt at entry, a slow lane consumes only its own slice, and a
  lane that overruns is downgraded ``partial`` by ``run_one``. Lanes
  that have not started when the total request deadline is exhausted
  report ``deadline`` — the request Deadline is a hard bound
  (V7-04.03). Decomposition facets (``query.facets``, V7-05.13) run
  through the same lanes inside the lane's divided slice; their
  candidates merge into the lane's ranked list tagged
  ``signals["facet"]`` so fusion/rerank can apply the facet coverage
  bonus (bonus itself is fusion's job, V7-10). V85-05.07: the declared
  ``scheduler.two_phase`` arm selects between the §21.2 two-phase
  schedule (default) and the single-phase slice-once shape; the mode
  and the request-deadline outcome are disclosed on
  ``coverage.budget`` (``two_phase``, ``elapsed_ms``,
  ``over_deadline``/``over_by_ms`` — the c3 ``over25`` criterion).
- **S3 union + fuse** — eligibility re-check on the merged candidate set
  (defense in depth; lanes already enforce eligibility before rank,
  V7-05.08) then ``fusion.rrf_fuse`` with the policy's intent-keyed lane
  weights (k=60). Module absent -> deterministic minimal union +
  unweighted-RRF fallback, marked ``unavailable`` in coverage.
- **S4 feature rerank** — ``rerank_features.score_candidates`` cut to
  ``pool.rerank_pool``. Absent -> RRF-order passthrough scored
  ``score_family="rrf/fusion-fallback"`` (the family honestly declares
  the score did not come from ``ranking/v7``).
- **S5 cross-encoder** — optional hook ``rerank_ce`` (unfrozen seam;
  expected ``rerank_ce.rerank(candidates, query, ctx) ->
  list[ScoredCandidate]`` re-scoring the head). Runs only when
  ``pool.ce_pool > 0``. Absent -> ``coverage.rerank =
  {status: unavailable, reason: no_ce}``. Never crashes.
- **S6 boosts** — ``boosts.apply_boosts`` (bounded multiplicative
  recency/proximity/proof). Absent -> scores pass through unchanged.
  V85-05.06: under the default ``temporal.as_of_scope="window"`` the
  recency clock is wall time — the caller's ``as_of`` anchors window
  resolution (S1) and the verdict (S7) only; ``"global"`` restores the
  prior anchor-everything behavior for paired ablation.
- **S7 verdict** — ``querying.verdict_v2`` group + result verdict.
  Absent -> *minimal structural fallback*: trigger (b) of V7-11.01 —
  zero eligible candidates -> ``insufficient`` with a
  :class:`MissingDescriptor`; otherwise ``ready`` flagged
  ``provisional`` in coverage (chosen over a blind ``ready`` so an empty
  result still abstains honestly).
- **S8 pack** — ``pack.assemble_pack``. Absent/failed -> raw scored
  items passthrough in ``result.pack`` (``mode="raw_passthrough"``,
  bounded by ``ctx.manifest["limit"]``) with a coverage note.
- **S9 exposure journal** — post-delivery, owned by the caller/facade,
  not this function; ``t_post`` records ``0.0``.

Caller parameters beyond the frozen signature travel on
``ctx.manifest``: ``limit`` (default 10), ``max_tokens``,
``calibration`` (handed to ``result_verdict``), ``explain`` (alternative
to the kwarg). The graph lane's seeds (§32.6: top-10 eligible of L-ent ∪
L-lex) are written to ``ctx.manifest["seeds"]`` as lanes complete.

Lazy-import rule: every engine module — ``deadline``, ``policy``,
``fusion``, ``rerank_features``, ``rerank_ce``, ``boosts``,
``querying.verdict_v2``, ``pack`` — is imported inside ``run_search``
through :func:`_lazy_import`. Wave-A modules land concurrently; absence
degrades honestly into coverage and never crashes the pipeline.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import time
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from ...core.types import ErrorCode, VerbatimError
from ...core.types_v7 import (
    POOLS,
    BudgetClass,
    CandidateV7,
    CoverageV7,
    FusedCandidate,
    IntentClass,
    LaneContextV7,
    LaneName,
    LaneOutput,
    LaneSlice,
    LaneStatus,
    MissingDescriptor,
    PoolProfile,
    QueryViewV7,
    ResultStatus,
    ScoredCandidate,
    StageRecord,
)
from .eligibility import _currency_verdict
from .lanes_base import FAIL_CLOSED_CODES, run_one
from . import turn_position as _tp
from ...core.time import now_us as _wall_now_us
from ...storage.repos import has_table

_RRF_K = 60
_DEFAULT_LIMIT = 10
_SEED_LANES = frozenset({LaneName.ENT, LaneName.LEX})
_SEED_TOP = 10

_MOD_DEADLINE = "verbatim.retrieval.v7.deadline"
_MOD_POLICY = "verbatim.retrieval.v7.policy"
_MOD_FUSION = "verbatim.retrieval.v7.fusion"
_MOD_RERANK = "verbatim.retrieval.v7.rerank_features"
_MOD_CE = "verbatim.retrieval.v7.rerank_ce"
_MOD_BOOSTS = "verbatim.retrieval.v7.boosts"
_MOD_VERDICT = "verbatim.querying.verdict_v2"
_MOD_PACK = "verbatim.retrieval.v7.pack"
_MOD_QV = "verbatim.querying.query_view"
_MOD_TEMPORAL = "verbatim.retrieval.v7.temporal"

# V8-14.03 — hard slack over the request deadline the whole pipeline is
# allowed (the §04 contract: ``search`` returns within
# ``timeout_ms + 25 ms``).
_HARD_SLACK_MS = 25.0

# Lane-phase constants duplicated as import fallbacks — the deadline
# module owns the canonical values; these keep the incremental-
# integration contract (module absent → §23 priors) alive.
_CORE_FALLBACK = frozenset(
    {LaneName.LEX, LaneName.FUZZY, LaneName.DENSE, LaneName.ENT}
)
_EXP_FALLBACK = frozenset(
    {LaneName.TIME, LaneName.TYPED, LaneName.OBS, LaneName.GRAPH}
)
_POST_P95_FALLBACK = {
    "rerank_feat": 30.0,
    "rerank_ce": 40.0,
    "boost": 5.0,
    "verdict": 20.0,
    "explain": 10.0,
}
_R_POST_FALLBACK = 60.0
_SLICE_FLOOR_FALLBACK = 0.5

# V8-10.06 — the facet fan-out runs on lex+ent by default; the
# ``facets.dense_per_facet`` arm adds dense. Other lanes answer the
# whole query only (a facet pass must not multiply heavy lanes).
_FACET_LANES_V8 = frozenset({LaneName.LEX, LaneName.ENT})

# Lane ``stats`` blocks grafted verbatim into ``coverage.<name>`` —
# ``coverage_*`` prefixed keys graft by suffix; the listed bare keys are
# the same shape under a shorter name (the graph lane's block is already
# the §20.03 coverage.graph payload; ``context`` merges with the fusion
# stage's S2c block).
_COVERAGE_GRAFT_EXTRA = frozenset({"graph", "context"})


@dataclass
class PipelineResult:
    """Product of ``run_search`` (frozen surface, docs/v7_contracts.md).

    The contract sketch lists ``lanes/fused/scored/verdict/missing/
    coverage/stage/explain``; ``pack`` is declared additionally because
    S8 must land its product somewhere stable — it carries the
    ``PackResult`` once ``retrieval/v7/pack.py`` exists, and the raw
    passthrough dict until then.
    """

    lanes: dict[str, LaneOutput]
    fused: list[FusedCandidate]
    scored: list[ScoredCandidate]
    verdict: ResultStatus
    missing: Optional[MissingDescriptor]
    coverage: CoverageV7
    stage: StageRecord
    explain: Optional[dict]
    pack: Any = None


def _lazy_import(modname: str) -> Any:
    """Import a wave-A engine module; ``None`` when absent/broken.

    A module that raises a fail-closed ``VerbatimError`` at import time
    still propagates — an integrity failure is never masked as
    "unavailable".
    """

    try:
        return importlib.import_module(modname)
    except VerbatimError as exc:
        if exc.code in FAIL_CLOSED_CODES:
            raise
        return None
    except Exception:  # noqa: BLE001 — includes ImportError
        return None


def _stage_call(fn: Callable, *args: Any, **kwargs: Any) -> tuple[bool, Any]:
    """Invoke an engine-stage callable. Returns ``(ok, value_or_repr)``.

    Fail-closed ``VerbatimError`` codes propagate (same rule as lanes);
    every other failure degrades to ``(False, repr(exc)[:200])``.
    """

    try:
        return True, fn(*args, **kwargs)
    except VerbatimError as exc:
        if exc.code in FAIL_CLOSED_CODES:
            raise
        return False, repr(exc)[:200]
    except Exception as exc:  # noqa: BLE001 — degradation is the contract
        return False, repr(exc)[:200]


def _ms(clock: Callable[[], float], t0: float) -> float:
    return (clock() - t0) * 1000.0


def _sched_call(fn: Callable, *args: Any, **kwargs: Any) -> tuple[bool, Any]:
    """``_stage_call`` for scheduler-param helpers: a malformed declared
    arm (``VerbatimError(VALIDATION)`` from ``policy_param`` resolution)
    is a policy defect — it propagates loudly instead of silently
    reverting to the prior (V7-05.02's declared-table rule)."""

    try:
        return True, fn(*args, **kwargs)
    except VerbatimError as exc:
        if exc.code in FAIL_CLOSED_CODES or exc.code == ErrorCode.VALIDATION:
            raise
        return False, repr(exc)[:200]
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)[:200]


def _pool_for(ctx: LaneContextV7) -> PoolProfile:
    """Pool profile for ``ctx.budget`` via ``policy.pool_for`` when the
    module has landed; the frozen ``POOLS`` table otherwise."""

    mod = _lazy_import(_MOD_POLICY)
    if mod is not None and callable(getattr(mod, "pool_for", None)):
        ok, val = _stage_call(mod.pool_for, ctx.budget)
        if ok and val is not None:
            return val
    try:
        budget = BudgetClass(ctx.budget)
    except ValueError:
        budget = BudgetClass.MID
    return POOLS.get(budget, POOLS[BudgetClass.MID])


def _allocate_slices(
    remaining_ms: float,
    lane_names: list[LaneName],
    ctx: LaneContextV7,
    pool: PoolProfile,
) -> tuple[dict[LaneName, LaneSlice], str]:
    """S2-entry slice computation (V7-05.05).

    Delegates to ``deadline.allocate`` when landed; equal-share fallback
    otherwise. Returns ``(slices, mode)`` — ``mode`` lands in
    ``coverage.budget["slices"]``.
    """

    mod = _lazy_import(_MOD_DEADLINE)
    if mod is not None and callable(getattr(mod, "allocate", None)):
        # ``pool=`` carries the request's lane cap into the slices; retry
        # positionally for an allocator built to the contract sketch.
        ok, val = _sched_call(
            mod.allocate, remaining_ms, lane_names, ctx.policy, pool=pool
        )
        if not ok:
            ok, val = _sched_call(mod.allocate, remaining_ms, lane_names, ctx.policy)
        if ok and isinstance(val, dict):
            out: dict[LaneName, LaneSlice] = {}
            for k, v in val.items():
                try:
                    out[LaneName(k)] = v
                except ValueError:
                    continue
            return out, "policy"
    share = remaining_ms / max(1, len(lane_names))
    return (
        {n: LaneSlice(deadline_ms=share, cap=pool.lane_cap) for n in lane_names},
        "fallback_equal",
    )


# ---------------------------------------------------------------------------
# V8-14.03/14.04/14.05 — two-phase scheduler plumbing
# ---------------------------------------------------------------------------


def _policy_param(policy: Any, name: str, default: Any = None) -> Any:
    """Read one §23 dotted arm off the policy's ``params`` map
    (``GatedPolicyV7``; absent on a bare ``RetrievalPolicyV7`` → the
    prior)."""
    params = getattr(policy, "params", None)
    if not isinstance(params, dict):
        return default
    return params.get(name, default)


def _request_limit(ctx: LaneContextV7) -> int:
    try:
        return max(1, int(ctx.manifest.get("limit", _DEFAULT_LIMIT)))
    except (TypeError, ValueError):
        return _DEFAULT_LIMIT


def _sched_fn(dl: Any, name: str) -> Optional[Callable]:
    return getattr(dl, name, None) if dl is not None else None


def _sched_r_post(dl: Any, policy: Any) -> float:
    fn = _sched_fn(dl, "resolve_r_post")
    if callable(fn):
        ok, val = _sched_call(fn, policy)
        if ok and isinstance(val, (int, float)):
            return float(val)
    return _R_POST_FALLBACK


def _sched_post_pool(dl: Any, policy: Any, limit: int) -> int:
    fn = _sched_fn(dl, "resolve_post_pool")
    if callable(fn):
        ok, val = _sched_call(fn, policy, limit)
        if ok and isinstance(val, int):
            return val
    return max(4 * int(limit), 64)


def _sched_graph_gate(dl: Any, policy: Any) -> bool:
    fn = _sched_fn(dl, "graph_gate_enabled")
    if callable(fn):
        ok, val = _sched_call(fn, policy)
        if ok:
            return bool(val)
    return True


def _sched_two_phase(dl: Any, policy: Any) -> bool:
    """``scheduler.two_phase`` (V85-05.07) — the §21.2 two-phase
    schedule unless the declared arm selects single-phase. Module
    absent → the prior (two-phase, current behavior); a malformed
    declared value propagates as VALIDATION like every scheduler arm."""
    fn = _sched_fn(dl, "two_phase_enabled")
    if callable(fn):
        ok, val = _sched_call(fn, policy)
        if ok:
            return bool(val)
    return True


def _as_of_scope(policy: Any) -> str:
    """``temporal.as_of_scope`` (V85-05.06) resolved through the
    temporal module's validator; ``"window"`` when the module is
    absent. A malformed declared value propagates as VALIDATION."""
    mod = _lazy_import(_MOD_TEMPORAL)
    fn = getattr(mod, "resolve_as_of_scope", None) if mod is not None else None
    if callable(fn):
        ok, val = _sched_call(fn, policy)
        if ok and isinstance(val, str):
            return val
    return "window"


def _sched_floor(dl: Any) -> float:
    v = getattr(dl, "SLICE_FLOOR_MS", None)
    return float(v) if isinstance(v, (int, float)) else _SLICE_FLOOR_FALLBACK


def _sched_p95(dl: Any) -> dict:
    v = getattr(dl, "POST_STAGE_P95_MS", None)
    return dict(v) if isinstance(v, dict) else dict(_POST_P95_FALLBACK)


def _expansion_order(dl: Any, names: list) -> list:
    fn = _sched_fn(dl, "expansion_order")
    if callable(fn):
        ok, val = _sched_call(fn, names)
        if ok and isinstance(val, list):
            return val
    return sorted(names, key=lambda n: str(getattr(n, "value", n)))


def _need_gate(dl: Any, name: LaneName, query: QueryViewV7, **kw: Any) -> tuple:
    """The §21.2 need gate — ``(needed, inputs)``. Module absent →
    conservative ``needed`` (a lane we cannot gate is never starved by
    a guess; its own in-lane skip still applies)."""

    fn = _sched_fn(dl, "need_gate")
    if callable(fn):
        ok, val = _sched_call(fn, name, query, **kw)
        if ok and isinstance(val, tuple) and len(val) == 2:
            return bool(val[0]), (val[1] if isinstance(val[1], dict) else {})
    return True, {"gate": "module_absent"}


def _expansion_slice(dl: Any, rem_ms: float, name: LaneName, pending: list) -> float:
    fn = _sched_fn(dl, "expansion_slice_ms")
    if callable(fn):
        ok, val = _sched_call(fn, rem_ms, name, pending)
        if ok and isinstance(val, (int, float)):
            return float(val)
    return float(rem_ms) / max(1, len(pending))


def _requested_lanes(ctx: LaneContextV7) -> frozenset:
    """Caller-requested lane names (§21.2 gate rule (c) — the
    ``lanes=[…]`` request channel travels on the manifest)."""

    man = getattr(ctx, "manifest", None) or {}
    out: set = set()
    for key in ("lanes", "requested_lanes", "lane_request"):
        v = man.get(key)
        if isinstance(v, str):
            out.add(v)
        elif isinstance(v, (list, tuple, set, frozenset)):
            out.update(str(x) for x in v)
    return frozenset(out)


def _graft_block(coverage: CoverageV7, name: str, block: Any) -> None:
    """Attach a §20.03 coverage block at ``coverage.<name>``.

    Declared ``CoverageV7`` dict fields are merged; names outside the
    contract (``post``/``graph``/``lexical``/``context``/``sql``/
    ``verdict`` — core/types_v7.py has no field for them yet) land as
    instance attributes so attribute readers see the spec path.  The
    stage digest serializes ``vars(coverage)`` so extension blocks are
    covered (see ``_coverage_digest``).
    """

    if not isinstance(block, dict):
        return
    cur = getattr(coverage, name, None)
    if isinstance(cur, dict):
        cur.update(block)
    else:
        setattr(coverage, name, dict(block))


def _core_gate_inputs(name: LaneName) -> dict:
    """§20.03 gate block for a phase-S2a lane: core lanes are ungated —
    ``inputs`` only discloses when a lane is an auxiliary lane outside
    the §21.2 core set (exact_id/source/scope run with the core)."""

    if name in _CORE_FALLBACK:
        return {}
    return {"auxiliary": True}


def _graft_lane_coverage(coverage: CoverageV7, out: LaneOutput) -> None:
    """Lift the lane's ``stats["coverage_*"]`` (and listed bare-name)
    blocks into ``coverage.*`` verbatim (V8-20.03)."""

    for key, val in (getattr(out, "stats", None) or {}).items():
        if not isinstance(val, dict):
            continue
        if key.startswith("coverage_"):
            _graft_block(coverage, key[len("coverage_"):], val)
        elif key in _COVERAGE_GRAFT_EXTRA:
            _graft_block(coverage, key, val)


def _run_lane(
    ctx: LaneContextV7,
    query: QueryViewV7,
    name: LaneName,
    slice_: LaneSlice,
    facets: list,
    facet_lanes: frozenset,
    pool: PoolProfile,
    clock: Callable[[], float],
    remaining_ms: Callable[[], float],
) -> LaneOutput:
    """One lane run including the V7-05.13 facet fan-out (same lane,
    divided slice, tagged merge).  Facets fan out only on
    ``facet_lanes`` (V8-10.06: lex+ent by default)."""

    _ensure_lane_loaded(name)
    out = run_one(ctx, query, name, slice_, clock=clock)
    if facets and name in facet_lanes:
        n_calls = 1 + len(facets)
        sub = slice_.deadline_ms / n_calls
        facet_stats = []
        merged = list(out.candidates)
        for fi, fq in enumerate(facets):
            if remaining_ms() <= 0:
                facet_stats.append({"facet": fi, "status": "deadline"})
                break
            fout = run_one(
                ctx,
                fq,
                name,
                LaneSlice(deadline_ms=sub, cap=slice_.cap),
                clock=clock,
            )
            appended = 0
            if fout.status in (LaneStatus.OK, LaneStatus.PARTIAL):
                for c in fout.candidates:
                    merged.append(
                        dc_replace(c, signals={**c.signals, "facet": fi})
                    )
                    appended += 1
            facet_stats.append(
                {"facet": fi, "status": fout.status.value, "produced": appended}
            )
        capv = slice_.cap if isinstance(slice_.cap, int) else len(merged)
        overflow = max(0, len(merged) - capv)
        if overflow:
            # V8-10.02 / §21.7 — reserved facet shares replace the
            # append-then-truncate merge (D8-16): the whole query keeps
            # the ``facets.whole_share`` share of the cap, each facet an
            # equal share of the rest, and unused shares roll over in
            # facet order — facet candidates no longer die whenever the
            # whole-query run fills the cap.
            fusion_mod = _lazy_import(_MOD_FUSION)
            reserve = getattr(fusion_mod, "reserve_facet_slots", None)
            if callable(reserve):
                share = _policy_param(ctx.policy, "facets.whole_share")
                rkw: dict[str, Any] = {"n_facets": len(facets)}
                if share is False:
                    rkw["whole_share"] = 1.0  # arm off — whole takes cap
                elif share not in (None, True):
                    rkw["whole_share"] = share
                ok, val = _sched_call(reserve, merged, capv, **rkw)
                if ok:
                    merged, slot_stats = val
                    per = slot_stats.get("per_facet") or {}
                    for entry in facet_stats:
                        fblk = per.get(str(entry.get("facet")))
                        if isinstance(fblk, dict):
                            entry["kept"] = fblk.get("kept", 0)
                            entry["rolled_over"] = fblk.get(
                                "rolled_over", 0
                            )
                    out.stats["facet_slots"] = slot_stats
                else:
                    merged = merged[:capv]
            else:
                merged = merged[:capv]
        else:
            merged = merged[:capv]
        out.candidates = [
            dc_replace(c, rank=i + 1) for i, c in enumerate(merged)
        ]
        out.stats["facets"] = facet_stats
        if overflow:
            out.stats["cap_truncated"] = out.stats.get("cap_truncated", 0) + overflow
    return out


def _context_inventory(ctx: LaneContextV7, outputs: list) -> list:
    """V8-06.07 — the §21.1 neighbor inventory: eligible-scope ``turn``
    rows in the sessions the nominated candidates touch, two batched
    reads (candidates→sessions, sessions→members).  The member read
    carries the V85-03 ordering columns (``occurred_start_us``,
    ``recorded_at_us``, ``byte_start``) so the context stage can index
    effective positions.  Honest ``[]`` when the store/table is
    unavailable — the context stage then runs applied with zero
    neighbor rows (no injections, no boosts; re-rank by identity keys
    only)."""

    conn = getattr(ctx, "store", None)
    if conn is None:
        return []
    try:
        if not has_table(conn, "units"):
            return []
    except Exception:
        return []
    ids = sorted(
        {str(c.unit_id) for o in outputs for c in (o.candidates or ())}
    )
    if not ids:
        return []
    gen = getattr(ctx, "generation", 0)
    scope = getattr(ctx, "scope_id", None)
    try:
        return _tp.context_inventory(conn, scope, ids, gen)
    except Exception:
        return []


# Lane name → module that implements ``lane_<name>``.  A module may
# self-register on import (``register_lane(LaneName.X, fn)`` — dense does);
# function-only modules are registered here on first use so the registry
# stays the single dispatch table.  Lanes with no module report
# ``unavailable/not_registered`` honestly — never fabricated.
_LANE_MODULES: dict[LaneName, str] = {
    LaneName.LEX: "verbatim.retrieval.v7.lexical",
    LaneName.FUZZY: "verbatim.retrieval.v7.fuzzy",
    LaneName.DENSE: "verbatim.retrieval.v7.dense",
    LaneName.TIME: "verbatim.retrieval.v7.temporal",
    LaneName.GRAPH: "verbatim.retrieval.v7.graph",
    LaneName.ENT: "verbatim.retrieval.v7.entity",
    LaneName.TYPED: "verbatim.retrieval.v7.typed",
    LaneName.OBS: "verbatim.retrieval.v7.obs",
    LaneName.EXACT_ID: "verbatim.retrieval.v7.exact_id",
    LaneName.SOURCE: "verbatim.retrieval.v7.source",
    # V75-04.01 scope lane (Q1 scope-form arm — default OFF, enabled only
    # via the declared policy lanes table).  The contract-frozen LaneName
    # enum has no SCOPE member; the module mints a LaneName instance
    # carrying "scope", and LaneName is a str enum — a "scope" key here
    # resolves identically for that member (same hash/eq).
    "scope": "verbatim.retrieval.v7.scope",
    # A lane whose module is absent or raises stays unregistered and
    # reports unavailable/not_registered (V7-04.03 honest coverage).
}

_LANE_FN = {
    LaneName.LEX: "lane_lexical",
    LaneName.FUZZY: "lane_fuzzy",
    LaneName.DENSE: "lane_dense",
    LaneName.TIME: "lane_temporal",
    LaneName.GRAPH: "lane_graph",
    LaneName.ENT: "lane_entity",
    LaneName.TYPED: "lane_typed",
    LaneName.OBS: "lane_obs",
    LaneName.EXACT_ID: "lane_exact_id",
    LaneName.SOURCE: "lane_source",
    "scope": "lane_scope",
}


def _ensure_lane_loaded(name: LaneName) -> None:
    """Import the lane's module and register its entry point if needed.

    Self-registering modules (``register_lane`` at import) are unaffected —
    re-registration of the same callable is idempotent.  A lane whose module
    is absent or raises stays unregistered; ``run_one`` then reports
    ``unavailable/not_registered`` per the lane contract.
    """

    from .lanes_base import LANE_REGISTRY, register_lane

    if name in LANE_REGISTRY:
        return
    modname = _LANE_MODULES.get(name)
    if modname is None:
        return
    mod = _lazy_import(modname)
    if mod is None:
        return
    if name in LANE_REGISTRY:  # module self-registered on import
        return
    fn = getattr(mod, _LANE_FN.get(name, ""), None)
    if callable(fn):
        try:
            register_lane(name, fn)
        except Exception:  # noqa: BLE001 — leave unregistered
            pass


def _policy_lanes(ctx: LaneContextV7) -> list[LaneName]:
    """Enabled lanes in declared policy order; invalid entries are
    dropped (they surface as coverage notes by the caller)."""

    out: list[LaneName] = []
    for raw in ctx.policy.lanes:
        try:
            out.append(LaneName(raw))
        except ValueError:
            continue
    return out


def _eligible_recheck(ctx: LaneContextV7, cand: CandidateV7) -> bool:
    """S3 eligibility re-check on one union candidate (V7-04.03/05.08).

    ``ctx.eligible`` is deliberately opaque (``callable(unit_row) ->
    bool`` or an eligible-set object). Best effort, never widening:
    callables are tried on the candidate then the ``unit_id``;
    containers are probed by membership. An unrecognized protocol
    passes the candidate through — lanes already enforced eligibility
    in candidate production, so this re-check can only tighten.
    A fail-closed ``VerbatimError`` from the eligibility handle
    propagates (never caught here).
    """

    el = getattr(ctx, "eligible", None)
    if el is None:
        return True
    if callable(el):
        for arg in (cand, cand.unit_id):
            try:
                return bool(el(arg))
            except (TypeError, KeyError, AttributeError, IndexError):
                continue
        return True
    try:
        return cand.unit_id in el
    except TypeError:
        try:
            return cand in el
        except TypeError:
            return True


def _fallback_fuse(outputs: list[LaneOutput]) -> list[FusedCandidate]:
    """Deterministic minimal union + unweighted RRF (k=60) used only
    when ``retrieval/v7/fusion.py`` has not landed. Merges on
    ``(unit_id, source_id, revision)``; lane ranks and raw signals are
    preserved per lane for explain. The real ``rrf_fuse`` adds the
    declared lane weights, no-ties rule (V7-10.02) and constant-signal
    veto (V7-10.03) — the fallback is labeled ``unavailable`` in
    coverage so consumers know which path produced the order."""

    merged: dict[tuple[str, str, int], FusedCandidate] = {}
    for out in outputs:
        for cand in out.candidates:
            key = (cand.unit_id, cand.source_id, cand.revision)
            add = 1.0 / (_RRF_K + cand.rank)
            prev = merged.get(key)
            if prev is None:
                merged[key] = FusedCandidate(
                    unit_id=cand.unit_id,
                    source_id=cand.source_id,
                    revision=cand.revision,
                    rrf=add,
                    lane_ranks={out.lane: cand.rank},
                    signals={out.lane: dict(cand.signals)},
                )
            else:
                prev.lane_ranks[out.lane] = cand.rank
                prev.signals[out.lane] = dict(cand.signals)
                merged[key] = dc_replace(prev, rrf=prev.rrf + add)
    return sorted(
        merged.values(),
        key=lambda f: (-f.rrf, f.unit_id, f.source_id, f.revision),
    )


def _fallback_scored(fused: list[FusedCandidate]) -> list[ScoredCandidate]:
    """S4 passthrough when ``rerank_features`` is absent: the fused RRF
    order becomes the score, honestly labeled by family."""

    return [
        ScoredCandidate(
            unit_id=f.unit_id,
            source_id=f.source_id,
            revision=f.revision,
            score=f.rrf,
            score_family="rrf/fusion-fallback",
            detail={"fallback": "rerank_features_absent", "lane_ranks": dict(f.lane_ranks)},
        )
        for f in fused
    ]


def _structural_verdict(
    scored: list[ScoredCandidate],
) -> tuple[ResultStatus, Optional[MissingDescriptor]]:
    """S7 minimal structural fallback (documented choice): only
    V7-11.01 trigger (b) — zero eligible candidates — is safe to compute
    without the calibrated module. Anything else is ``ready`` and the
    caller sees ``provisional`` in coverage."""

    if not scored:
        return (
            ResultStatus.INSUFFICIENT,
            MissingDescriptor(
                facets={"eligible": ("zero_candidates",)},
                note="structural fallback: no eligible candidates after S2-S6",
            ),
        )
    return ResultStatus.READY, None


def _update_seeds(ctx: LaneContextV7, name: LaneName, out: LaneOutput) -> None:
    """§32.6: graph-lane seeds = top-10 eligible hits of L-ent ∪ L-lex,
    propagated to later lanes via ``ctx.manifest["seeds"]``."""

    if name not in _SEED_LANES:
        return
    seeds = ctx.manifest.setdefault("seeds", [])
    for cand in out.candidates[:_SEED_TOP]:
        if cand.unit_id not in seeds:
            seeds.append(cand.unit_id)


def _coverage_digest(coverage: CoverageV7) -> float:
    """Numeric tag of the coverage block for the stage record (§32.17
    ``coverage_digest``). Deterministic over identical coverage.

    Serializes ``vars()`` — not ``dc_asdict`` — so §20.03 extension
    blocks grafted as attributes (``post``/``graph``/``context``/…)
    are covered by the digest exactly like declared fields."""

    try:
        blob = json.dumps(
            vars(coverage), sort_keys=True, default=str
        ).encode("utf-8")
    except (TypeError, ValueError):
        blob = repr(coverage).encode("utf-8")
    return float(int.from_bytes(hashlib.blake2b(blob, digest_size=8).digest(), "big"))


def _apply_source_currency(
    ctx: LaneContextV7, scored: list[ScoredCandidate], query: QueryViewV7
) -> tuple[list[ScoredCandidate], dict]:
    """S6b — source-lifecycle currency (V7-05.08's lifecycle leg;
    V5-14.12 window; V7-09.11 history semantics).

    Eligibility already withheld the never-answerable dispositions
    before rank; this stage applies the intent-dependent half where the
    query's intent is known — BEFORE S7 so group verdicts never count an
    out-of-window source as support:

    * non-history queries drop candidates whose source fails the
      currency window (a ``superseded`` source past ``effective_at``
      cannot answer a current question — the successor owns it);
    * ``history_of`` keeps them and the pack labels them
      (``superseded`` / closed-window ``active`` → ``historical``) —
      predecessors stay deliverable evidence (V75-04.08).

    Missing ``source_state`` rows are unresolved state, not an
    assertion of suppression — the candidate is admissible (V5-14.16).
    A source-level ``superseded`` label outranks fact-level lifecycle
    marks the lanes may have set; ``historical`` only fills a blank or
    ``current`` mark.
    """

    if not scored:
        return scored, {"status": "ok", "dropped": 0}
    conn = getattr(ctx, "store", None)
    if conn is None:
        return scored, {
            "status": "skipped",
            "reason": "no_read_snapshot",
            "dropped": 0,
        }
    try:
        if not has_table(conn, "source_state"):
            return scored, {
                "status": "skipped",
                "reason": "no_source_state",
                "dropped": 0,
            }
    except Exception:
        return scored, {
            "status": "skipped",
            "reason": "no_source_state",
            "dropped": 0,
        }

    classes = getattr(getattr(query, "intent", None), "classes", None) or ()
    history = IntentClass.HISTORY_OF in classes
    # Wall clock — identical to the source lane's ``_lifecycle_map`` so a
    # source's window verdict is the same wherever it is evaluated.
    now_ts = datetime.now(timezone.utc).timestamp()

    sids = sorted({str(s.source_id) for s in scored if s.source_id})
    disp_map: dict[str, tuple] = {}
    try:
        for i in range(0, len(sids), 400):
            part = sids[i : i + 400]
            ph = ",".join("?" * len(part))
            for row in conn.execute(
                "SELECT source_id, disposition, effective_at,"
                " valid_from, valid_to FROM source_state"
                f" WHERE source_id IN ({ph})",
                part,
            ):
                disp_map[str(row[0])] = (
                    str(row[1] or ""),
                    row[2],
                    row[3],
                    row[4],
                )
    except Exception as exc:
        # A read fault on the lifecycle plane is a skip, not a guess —
        # candidates keep their pre-stage deliverability and the note
        # discloses the gap.
        return scored, {
            "status": "skipped",
            "reason": f"lifecycle_read:{type(exc).__name__}",
            "dropped": 0,
        }

    kept: list[ScoredCandidate] = []
    dropped = 0
    labeled = 0
    for s in scored:
        row = disp_map.get(str(s.source_id)) if s.source_id else None
        if row is None:
            kept.append(s)
            continue
        ok, label = _currency_verdict(row[0], row[1], row[2], row[3], now_ts, history)
        if not ok:
            dropped += 1
            continue
        if label and isinstance(s.detail, dict):
            cur = s.detail.get("lifecycle")
            if label == "superseded" or cur in (None, "current"):
                s.detail["lifecycle"] = label
                labeled += 1
        kept.append(s)

    return kept, {
        "status": "ok",
        "history": bool(history),
        "checked_sources": len(disp_map),
        "dropped": dropped,
        "labeled": labeled,
    }


def _materialize_details(
    ctx: LaneContextV7,
    scored: list[ScoredCandidate],
    *,
    budget_exceeded: Optional[Callable[[], bool]] = None,
) -> int:
    """Fill ``detail`` with the unit's deliverable content for S8.

    ``pack.assemble_pack`` reads ``quote``/``speaker``/``session``/
    ``occurred``/``recorded_at`` off ``ScoredCandidate.detail`` — the
    lanes only carry ranking signals, so materialization lives here: one
    covering fetch of each candidate's ``units`` row plus its text
    (``unit_fts_content`` via the rowid carrier, else the pinned
    ``source_revisions.payload[byte_start:byte_end]`` slice).  Lane-set
    detail keys are never overwritten — a typed fact's
    ``state_key``/``value`` outranks a generic quote fill.

    ``budget_exceeded`` (V8-14.04) is polled inside the per-item fill
    loop: the first ``True`` stops the byte-slice enrichment tail —
    unfilled items keep their lane-level detail and the return value
    reports how many items went unfilled.
    """

    conn = getattr(ctx, "store", None)
    if conn is None or not scored:
        return 0
    # V8.5 — the pack can only deliver ``limit`` items: the deliverable
    # prefix fills unconditionally (delivery materialization is not the
    # droppable tail — unfilled items collapse onto an empty signature
    # downstream), and the tail beyond ``2*limit`` can never ship, so it
    # is not fetched at all.  ``budget_exceeded`` gates only the
    # collapse-backfill margin between ``limit`` and ``2*limit``
    # (V8-14.04's "enrichment tail").
    cap = int(ctx.manifest.get("limit", _DEFAULT_LIMIT) or _DEFAULT_LIMIT)
    head = [s for s in scored[:cap] if not (s.detail or {}).get("quote")]
    tail = [s for s in scored[cap : 2 * cap] if not (s.detail or {}).get("quote")]
    want = head + tail
    if not want:
        return 0
    tail_ids = {id(s) for s in tail}
    ids = [s.unit_id for s in want]
    gen = ctx.generation
    rows: dict[str, dict] = {}
    ph = ",".join("?" * len(ids))
    try:
        sql = (
            "SELECT u.unit_id, u.speaker_canon, u.session_id, "
            "u.recorded_at_us, u.occurred_start_us, u.occurred_end_us, "
            "u.occurred_precision, u.occurred_source, "
            "u.byte_start, u.byte_end, u.source_id, u.revision, "
            "c.text, u.generation "
            "FROM units u "
            "LEFT JOIN unit_fts_rows r ON r.unit_id = u.unit_id "
            "   AND r.generation <= ? "
            "LEFT JOIN unit_fts_content c ON c.fts_row_id = r.row_id "
            f"WHERE u.unit_id IN ({ph}) AND u.scope_id = ? "
            "AND u.generation <= ? "
            "ORDER BY u.generation DESC, r.generation DESC"
        )
        for r in conn.execute(sql, [gen, *ids, ctx.scope_id, gen]):
            uid = r[0]
            # ORDER BY … DESC ⇒ the first row per unit is its latest
            # generation ≤ fence (rebuild coexistence, V7-30.02).
            if uid not in rows:
                rows[uid] = {
                    "speaker": r[1],
                    "session_id": r[2],
                    "recorded_at_us": r[3],
                    "occurred_start_us": r[4],
                    "occurred_end_us": r[5],
                    "occurred_precision": r[6],
                    "occurred_source": r[7],
                    "byte_start": r[8],
                    "byte_end": r[9],
                    "source_id": r[10],
                    "revision": r[11],
                    "text": r[12],
                    "generation": r[13],
                }
    except Exception:
        rows = {}
    unfilled = 0
    for s in want:
        if budget_exceeded is not None and id(s) in tail_ids:
            try:
                if budget_exceeded():
                    unfilled += 1
                    continue
            except Exception:  # noqa: BLE001 — a bad clock never widens work
                pass
        u = rows.get(s.unit_id)
        if u is None:
            continue
        d = s.detail
        text = u.get("text")
        if not text and u.get("byte_start") is not None and u.get("byte_end") is not None:
            # byte-pinned fallback: slice the source payload
            try:
                prow = conn.execute(
                    "SELECT payload FROM source_revisions "
                    "WHERE source_id = ? AND revision = ?",
                    (u["source_id"], u["revision"]),
                ).fetchone()
                if prow and prow[0] is not None:
                    text = bytes(prow[0])[u["byte_start"]:u["byte_end"]].decode(
                        "utf-8", "replace"
                    )
            except Exception:
                text = None
        if text is not None:
            d.setdefault("quote", text.encode("utf-8"))
        if u.get("speaker"):
            d.setdefault("speaker", u["speaker"])
        if u.get("recorded_at_us") is not None:
            try:
                from ...core.time import rfc3339 as _rfc
                d.setdefault("recorded_at", _rfc(int(u["recorded_at_us"])))
            except Exception:
                pass
        if u.get("occurred_start_us") is not None:
            d.setdefault(
                "occurred",
                {
                    "start_us": u["occurred_start_us"],
                    "end_us": u.get("occurred_end_us"),
                    "precision": u.get("occurred_precision"),
                    "source": u.get("occurred_source"),
                },
            )
        if u.get("session_id"):
            sess = {"id": u["session_id"]}
            if u.get("occurred_start_us"):
                sess["started_us"] = u["occurred_start_us"]
            d.setdefault("session", sess)
    return unfilled


def _pack_item_ids(pack: Any) -> set[str]:
    """Unit ids delivered by the pack, best-effort over the PackResult
    shape (``pack.items`` of PackItemV7, or the passthrough dict)."""

    if isinstance(pack, dict):
        items = pack.get("items")
    else:
        items = getattr(pack, "items", None)
    ids: set[str] = set()
    if items is not None and not callable(items):
        for it in items:
            uid = getattr(it, "unit_id", None)
            if uid is None and isinstance(it, dict):
                uid = it.get("unit_id")
            if uid is not None:
                ids.add(uid)
    return ids


#: V8-20.04 — per-item explain fields surfaced from lane/fusion signals
#: onto the delivered item: ``support`` (V8-12.05), ``ctx_from`` +
#: ``pre_ctx_score`` (V8-06), ``dense_slot`` (V8-08.04), ``joint``
#: (V8-10.03), ``facet`` (V8-10.02), ``mention_interval`` (V8-09.08),
#: ``rescue`` (V8-07.03).  First writer wins across the item's lane
#: signal maps — ``FusedCandidate.signals`` iterates in contribution
#: order, which is deterministic.
_EXPLAIN_ITEM_SIGNAL_KEYS = (
    "support",
    "ctx_from",
    "pre_ctx_score",
    "dense_slot",
    "joint",
    "facet",
    "mention_interval",
    "rescue",
)


def _build_explain(
    ctx: LaneContextV7,
    query: QueryViewV7,
    lane_order: list[str],
    lane_outputs: dict[str, LaneOutput],
    fused: list[FusedCandidate],
    scored: list[ScoredCandidate],
    groups: list,
    verdict: ResultStatus,
    missing: Optional[MissingDescriptor],
    pack: Any,
    stage: StageRecord,
    slices: dict[LaneName, LaneSlice],
    post_pool: Optional[int] = None,
) -> dict:
    """V7-05.15 explain payload: per delivered item — lane ranks, raw
    signals, fused score, rerank score, boosts detail, verdict reason,
    pack decision. Plain JSON-able values; insertion order matches the
    delivered order so ``json.dumps(sort_keys=True)`` is byte-stable."""

    fused_map = {(f.unit_id, f.source_id, f.revision): f for f in fused}
    delivered = _pack_item_ids(pack)
    passthrough = isinstance(pack, dict) and pack.get("mode") == "raw_passthrough"

    group_map: dict[str, Any] = {}
    for g in groups:
        lab = getattr(getattr(g, "label", None), "value", getattr(g, "label", None))
        if lab is None:
            continue
        for uid in (getattr(g, "detail", None) or {}).get("members") or ():
            group_map[uid] = lab

    items = []
    for s in scored:
        f = fused_map.get((s.unit_id, s.source_id, s.revision))
        if passthrough:
            pack_decision = "passthrough"
        elif delivered:
            pack_decision = "delivered" if s.unit_id in delivered else "cut"
        else:
            pack_decision = "unknown"
        item: dict[str, Any] = {
            "unit_id": s.unit_id,
            "source_id": s.source_id,
            "revision": s.revision,
            "lane_ranks": dict(f.lane_ranks) if f else {},
            "signals": f.signals if f else {},
            "fused_rrf": f.rrf if f else None,
            "score": s.score,
            "score_family": s.score_family,
            "detail": s.detail,
            "group_label": group_map.get(s.unit_id),
            "pack": pack_decision,
        }
        if f is not None:
            # V8-20.04 — hoist the declared per-item fields out of the
            # per-lane signal maps so explain readers see them directly.
            for sig_map in f.signals.values():
                if not isinstance(sig_map, dict):
                    continue
                for fld in _EXPLAIN_ITEM_SIGNAL_KEYS:
                    if fld in sig_map and fld not in item:
                        item[fld] = sig_map[fld]
        items.append(item)

    return {
        "query": {
            "text": query.query,
            "intent": query.intent.primary.value,
            "classes": [c.value for c in query.intent.classes],
            "terms": [t.term for t in query.norm.terms],
            "identifiers": [t.term for t in query.norm.identifiers],
            "entity_canons": list(query.entity_canons),
            "facets": len(query.facets or ()),
        },
        "policy": {
            "policy_id": ctx.policy.policy_id,
            "profile": ctx.policy.profile,
            "formula_status": ctx.policy.formula_status,
            "lanes": lane_order,
        },
        "slices": {
            n.value: {"deadline_ms": s.deadline_ms, "cap": s.cap}
            for n, s in slices.items()
        },
        "lanes": {
            name: {
                "status": out.status.value,
                "reason": out.reason,
                "examined": out.examined,
                "eligible": out.eligible,
                "stats": out.stats,
                "candidates": [
                    {
                        "unit_id": c.unit_id,
                        "source_id": c.source_id,
                        "revision": c.revision,
                        "rank": c.rank,
                        "raw_score": c.raw_score,
                        "signals": c.signals,
                    }
                    for c in out.candidates
                ],
            }
            for name, out in lane_outputs.items()
        },
        "verdict": {
            "status": verdict.value,
            "groups": [
                {
                    "group_key": getattr(g, "group_key", None),
                    "label": getattr(getattr(g, "label", None), "value", getattr(g, "label", None)),
                    "trigger": getattr(g, "trigger", None),
                }
                for g in groups
            ],
            "missing": (
                {"facets": missing.facets, "note": missing.note} if missing else None
            ),
        },
        "items": items,
        # V8-14.05 — the post-pool trim boundary: ``fused`` counts the
        # full fused pool (explain-visible, never discarded); ``scored``
        # counts what rerank actually scored.
        "post_pool": {
            "limit": post_pool,
            "fused": len(fused),
            "scored": len(scored),
        },
        "stage": {"fields": dict(stage.fields), "t_lane": dict(stage.t_lane)},
    }


def run_search(
    ctx: LaneContextV7,
    query: QueryViewV7,
    deadline_ms: float,
    *,
    explain: bool = False,
    clock: Optional[Callable[[], float]] = None,
) -> PipelineResult:
    """Run S1–S8 and return a :class:`PipelineResult`.

    ``deadline_ms`` is the total request budget for stages S2–S8 (S0/S1
    are upstream — see module docstring). ``explain=True`` (or
    ``ctx.manifest["explain"]``) attaches the V7-05.15 per-item payload.
    ``clock`` is injectable for deterministic deadline tests; defaults
    to ``time.monotonic``.
    """

    clock = clock or time.monotonic
    explain = bool(explain or ctx.manifest.get("explain"))
    if deadline_ms is None:
        deadline_ms = float("inf")
    t_start = clock()

    def remaining_ms() -> float:
        return deadline_ms - _ms(clock, t_start)

    stage = StageRecord(kind="search")
    coverage = CoverageV7()
    coverage.policy = {
        "policy_id": ctx.policy.policy_id,
        "profile": ctx.policy.profile,
        "formula_status": ctx.policy.formula_status,
        "lanes": [getattr(n, "value", str(n)) for n in ctx.policy.lanes],
    }
    coverage.budget = {
        "class": getattr(ctx.budget, "value", str(ctx.budget)),
        "deadline_ms": deadline_ms,
    }

    pool = _pool_for(ctx)
    coverage.budget.update(
        {
            "lane_cap": pool.lane_cap,
            "rerank_pool": pool.rerank_pool,
            "ce_pool": pool.ce_pool,
            "neighbor_window": pool.neighbor_window,
        }
    )

    lane_names = _policy_lanes(ctx)
    facets = list(query.facets or ())[: max(0, pool.max_facets)]
    coverage.facets = {
        "declared": len(query.facets or ()),
        "ran": len(facets),
        "max_facets": pool.max_facets,
        # V75-03.04/J08: no unconditional "bonus" declaration — the key is
        # written after S3 only when fusion actually applied a facet bonus.
    }
    qv_mod = _lazy_import(_MOD_QV)
    if qv_mod is not None and callable(getattr(qv_mod, "facet_source", None)):
        ok, val = _stage_call(qv_mod.facet_source, query)
        if ok and val is not None:
            coverage.facets["source"] = val

    # ------------------------------------------------------------------
    # S2 — scheduling (V8-14.03, §21.2; V85-05.07 arm).
    #
    # Two-phase (default): S2a core lanes {lex, fuzzy, dense, ent}
    # (+ auxiliary lanes outside the §21.2 sets — exact_id/source/scope
    # run with the core) sliced once from ``budget − R_post`` and run in
    # policy order; S2b expansion lanes {time, typed, obs, graph} — each
    # need-gated on the query + the S2a core union, then re-derated: its
    # slice is computed from the budget remaining *when it starts*,
    # shared over the costs of the expansion lanes not yet run.  A
    # gated-out lane reports ``skipped(not_needed)``; one finding no
    # budget reports ``skipped(deadline_exhausted)``.
    #
    # Single-phase (``scheduler.two_phase=false``): every enabled lane is
    # sliced once from ``budget − R_post`` and run in policy order — no
    # need gates, no re-deration (the pre-V8-14.03 shape). Lanes keep
    # their own in-lane skips; the mode is disclosed on
    # ``coverage.budget["two_phase"]`` and each lane's gate inputs.
    # ------------------------------------------------------------------
    dl = _lazy_import(_MOD_DEADLINE)
    r_post = _sched_r_post(dl, ctx.policy)
    limit_req = _request_limit(ctx)
    post_pool = _sched_post_pool(dl, ctx.policy, limit_req)
    graph_gate = _sched_graph_gate(dl, ctx.policy)
    slice_floor = _sched_floor(dl)
    p95 = _sched_p95(dl)
    two_phase = _sched_two_phase(dl, ctx.policy)
    as_of_scope = _as_of_scope(ctx.policy)
    coverage.budget["two_phase"] = two_phase
    _graft_block(coverage, "temporal", {"as_of_scope": as_of_scope})

    exp_set = getattr(dl, "EXPANSION_SET_V8", None) or _EXP_FALLBACK
    core_names = [n for n in lane_names if n not in exp_set]
    exp_names = [n for n in lane_names if n in exp_set]

    # V85-05.05 — facets fan out only on the policy's enabled lanes ∩
    # {lex, ent} (``_FACET_LANES_V8``); ``facets.dense_per_facet`` still
    # opts dense in when the dense lane itself is enabled.
    facet_lanes = set(_FACET_LANES_V8) & set(lane_names)
    if _policy_param(ctx.policy, "facets.dense_per_facet"):
        if LaneName.DENSE in lane_names:
            facet_lanes.add(LaneName.DENSE)
    facet_lanes = frozenset(facet_lanes)

    lane_outputs: dict[str, LaneOutput] = {}
    lane_order: list[str] = []
    dealt: dict[LaneName, LaneSlice] = {}
    deadline_cut: set = set()

    def _record_lane(name, out, *, phase, gate, slice_ms, t_lane_ms):
        key = name.value
        lane_outputs[key] = out
        stage.t_lane[key] = round(t_lane_ms, 3)
        stage.candidates[key] = len(out.candidates)
        # Gate verdict lands on the lane's stats too — §21.2's "gate
        # inputs are logged in explain" (V8-05.05) reads the payload.
        out.stats.setdefault("gate", gate)
        if out.status == LaneStatus.DEADLINE or str(out.reason or "").startswith(
            "deadline"
        ) or out.reason == "no_slice_budget":
            # V8-12.06 — lanes the request deadline cut; the verdict
            # reads the set off ctx.manifest before classify_groups.
            deadline_cut.add(key)
        coverage.lane(
            key,
            out.status,
            out.reason,
            examined=out.examined,
            eligible=out.eligible,
            produced=len(out.candidates),
            cap=(dealt.get(name).cap if dealt.get(name) else pool.lane_cap),
            phase=phase,
            gate=gate,
            slice_ms=round(slice_ms, 3),
            t_ms=round(t_lane_ms, 3),
        )
        _graft_lane_coverage(coverage, out)
        _update_seeds(ctx, name, out)

    t_s2 = clock()

    # --- S2a — sliced once from budget − R_post (§21.2). Under the
    # single-phase arm (V85-05.07) this phase covers *every* enabled
    # lane; under two-phase it covers the core set only. -------------
    s2a_names = core_names if two_phase else list(lane_names)
    b_core = remaining_ms() - r_post
    slices, slice_mode = _allocate_slices(b_core, s2a_names, ctx, pool)
    coverage.budget["slices"] = slice_mode
    coverage.budget["r_post_ms"] = r_post
    dealt.update(slices)
    for name in s2a_names:
        key = name.value
        lane_order.append(key)
        lt0 = clock()
        if remaining_ms() <= 0:
            out = LaneOutput(
                lane=key, status=LaneStatus.DEADLINE, reason="deadline_exhausted"
            )
            slice_ms = slices.get(name).deadline_ms if slices.get(name) else 0.0
        else:
            slice_ = slices.get(name) or LaneSlice(
                deadline_ms=max(0.0, remaining_ms()), cap=pool.lane_cap
            )
            dealt.setdefault(name, slice_)
            slice_ms = slice_.deadline_ms
            out = _run_lane(
                ctx, query, name, slice_, facets, facet_lanes, pool, clock,
                remaining_ms,
            )
        gate_inputs = _core_gate_inputs(name)
        if not two_phase:
            gate_inputs = {**gate_inputs, "scheduler": "single_phase"}
        _record_lane(
            name, out, phase="core",
            gate={"needed": True, "inputs": gate_inputs},
            slice_ms=slice_ms, t_lane_ms=_ms(clock, lt0),
        )

    if two_phase:
        # --- S2b — expansion lanes, need-gated + re-derated (§21.2) ---
        # Distinct eligible units across the S2a core union — the gate
        # input for obs/graph ("core union below 4 × limit").
        core_union = 0
        seen_units: set = set()
        for name in core_names:
            out = lane_outputs.get(name.value)
            for c in (out.candidates if out is not None else ()):
                if c.unit_id in seen_units:
                    continue
                if _eligible_recheck(ctx, c):
                    seen_units.add(c.unit_id)
        core_union = len(seen_units)
        union_floor = 4 * limit_req
        requested = _requested_lanes(ctx)

        exp_order = _expansion_order(dl, exp_names)
        exp_gate: dict = {}
        exp_needed: list = []
        for name in exp_order:
            needed, inputs = _need_gate(
                dl, name, query,
                core_union=core_union, union_floor=union_floor,
                requested=requested, graph_gate=graph_gate,
            )
            exp_gate[name] = (needed, inputs)
            if needed:
                exp_needed.append(name)

        done: set = set()
        for name in exp_order:
            key = name.value
            lane_order.append(key)
            needed, inputs = exp_gate[name]
            lt0 = clock()
            if not needed:
                out = LaneOutput(
                    lane=key, status=LaneStatus.SKIPPED, reason="not_needed"
                )
                _record_lane(
                    name, out, phase="expansion",
                    gate={"needed": False, "inputs": inputs},
                    slice_ms=0.0, t_lane_ms=_ms(clock, lt0),
                )
                continue
            rem = remaining_ms() - r_post
            if rem <= slice_floor:
                out = LaneOutput(
                    lane=key, status=LaneStatus.SKIPPED,
                    reason="deadline_exhausted",
                )
                _record_lane(
                    name, out, phase="expansion",
                    gate={"needed": True, "inputs": inputs},
                    slice_ms=0.0, t_lane_ms=_ms(clock, lt0),
                )
                continue
            # slice = rem · cost(l) / Σ cost over needed lanes not yet
            # run — recomputed at each lane's start (V8-14.03).
            pending = [m for m in exp_needed if m not in done]
            share = _expansion_slice(dl, rem, name, pending)
            slice_ms = max(slice_floor, share)
            slice_ = LaneSlice(deadline_ms=slice_ms, cap=pool.lane_cap)
            dealt[name] = slice_
            out = _run_lane(
                ctx, query, name, slice_, facets, facet_lanes, pool, clock,
                remaining_ms,
            )
            done.add(name)
            _record_lane(
                name, out, phase="expansion",
                gate={"needed": True, "inputs": inputs},
                slice_ms=slice_ms, t_lane_ms=_ms(clock, lt0),
            )
    stage.fields["t_lanes"] = round(_ms(clock, t_s2), 3)

    # ------------------------------------------------------------------
    # V8-14.04 — post-lane window: S3–S8 bounded by R_post; the hard wall
    # is ``deadline_ms + 25 ms`` from request start (§04 contract).
    # ------------------------------------------------------------------
    t_post0 = clock()
    if math.isfinite(deadline_ms):
        post_end = min(
            t_post0 + r_post / 1000.0,
            t_start + (deadline_ms + _HARD_SLACK_MS) / 1000.0,
        )
    else:
        post_end = float("inf")

    def post_remaining_ms() -> float:
        return (post_end - clock()) * 1000.0

    def post_exceeded() -> bool:
        return post_remaining_ms() <= 0.0

    post_degraded: list = []

    def _degrade(tag: str) -> None:
        if tag not in post_degraded:
            post_degraded.append(tag)

    # ------------------------------------------------------------------
    # S3 — eligibility re-check on the union, then RRF fuse.  The
    # eligibility re-check is never deadline-gated (V8-14.04: pins and
    # eligibility are never dropped).
    # ------------------------------------------------------------------
    t = clock()
    kept_outputs: list[LaneOutput] = []
    recheck_dropped = 0
    for key in lane_order:
        out = lane_outputs[key]
        if not out.candidates:
            continue
        kept = [c for c in out.candidates if _eligible_recheck(ctx, c)]
        recheck_dropped += len(out.candidates) - len(kept)
        if len(kept) == len(out.candidates):
            kept_outputs.append(out)
        else:
            kept_outputs.append(
                LaneOutput(
                    lane=out.lane,
                    status=out.status,
                    candidates=kept,
                    reason=out.reason,
                    examined=out.examined,
                    eligible=out.eligible,
                    stats=dict(out.stats),
                )
            )
    stage.fields["t_union"] = round(_ms(clock, t), 3)
    if recheck_dropped:
        coverage.security["eligibility_recheck_dropped"] = recheck_dropped

    t = clock()
    weights = dict(ctx.policy.lane_weights.get(query.intent.primary, {}))
    # V75-04.03 weak-lane gates ride the policy object (``lane_gates``,
    # absent on a bare RetrievalPolicyV7 => nothing gated). V75-03.03:
    # ``ctx.policy.policy_id`` is the weight-table arm tag — the flat
    # default ``retrieval_policy/v7`` or the selected
    # ``retrieval_policy/v7-intent-weighted`` arm — echoed into the fusion
    # note and fusion stats so a paired Track R run attributes correctly.
    lane_gates = getattr(ctx.policy, "lane_gates", None) or None
    fusion_note: dict[str, Any] = {
        "status": "ok",
        "weights_arm": ctx.policy.policy_id,
    }
    if lane_gates:
        fusion_note["lane_gates"] = {
            getattr(k, "value", str(k)): v for k, v in lane_gates.items()
        }

    # V8 §23 fusion arms — resolved off the policy params map and passed
    # only when the landed fusion accepts them (``None`` → its internal
    # prior).  ``context`` (V8-06, stage S2c) runs inside ``rrf_fuse``
    # before lane ranks are consumed; its neighbor inventory is the one
    # batched read of V8-06.07.
    fusion_kwargs: dict[str, Any] = {
        "lane_gates": lane_gates,
        "weights_tag": ctx.policy.policy_id,
    }
    for _kw, _pname in (
        ("dense_slots", "dense.N_d"),
        ("facet_whole_share", "facets.whole_share"),
        ("lex_anchor", "fusion.lex_anchor"),
    ):
        _v = _policy_param(ctx.policy, _pname)
        if _v is not None:
            fusion_kwargs[_kw] = _v
    ctx_mode = _policy_param(ctx.policy, "context.mode", None)
    ctx_off = ctx_mode is not None and str(ctx_mode).lower() in (
        "off", "none", "false",
    )

    fusion = _lazy_import(_MOD_FUSION)
    fused: Any = []
    fstats: dict = {}
    if fusion is not None and callable(getattr(fusion, "rrf_fuse", None)):
        if ctx_off:
            fusion_kwargs["context"] = {"mode": "off"}
        else:
            # context.mode prior is propagate+inject (§23); undeclared
            # keys fall through to fusion's own priors — never pass a
            # null arm.  The neighbor inventory is the one batched read
            # of V8-06.07, computed only when fusion will consume it.
            ctx_map: dict[str, Any] = {
                "neighbors": _context_inventory(ctx, kept_outputs),
                "eligible": getattr(ctx, "eligible", None),
            }
            if ctx_mode is not None:
                ctx_map["mode"] = ctx_mode
            for _k, _pname in (
                ("w", "context.w"),
                ("W", "context.W"),
                ("M_ctx", "context.M_ctx"),
            ):
                _v = _policy_param(ctx.policy, _pname)
                if _v is not None:
                    ctx_map[_k] = _v
            fusion_kwargs["context"] = ctx_map
        # ``fusion.dense_form``/``dense_form_alpha`` (V8-08.06) resolve
        # off the policy inside ``rrf_fuse`` — the kwarg must carry it
        # or the declared arm is unreadable on the production path.
        fusion_kwargs["policy"] = ctx.policy
        # ``_sched_call`` (not ``_stage_call``): a VALIDATION raised on a
        # declared §23 arm (e.g. ``context.w`` outside (0, 1)) is a policy
        # defect — loud, never silently reverted to the prior.
        ok, val = _sched_call(
            fusion.rrf_fuse,
            kept_outputs,
            weights,
            _RRF_K,
            **fusion_kwargs,
        )
        if not ok:
            # Contract-sketch fusions accept (outputs, weights, k) only.
            ok, val = _stage_call(fusion.rrf_fuse, kept_outputs, weights, _RRF_K)
        if ok and val is not None:
            # Keep a returned FusedList — its ``stats`` (constant-signal
            # veto, truncation) feed the S4 ``suppress`` argument, and its
            # facet_bonus/lanes_gated/context blocks feed honest coverage.
            fused = val if isinstance(val, list) else list(val)
            fstats = getattr(val, "stats", None) or {}
            if isinstance(fstats, dict):
                fb = fstats.get("facet_bonus") or {}
                if fb.get("applied"):
                    # J08: declared only when a bonus was applied.
                    coverage.facets["bonus"] = fb.get("tag") or "applied"
                if fstats.get("lanes_gated"):
                    fusion_note["lanes_gated"] = fstats["lanes_gated"]
                if fstats.get("gates_bypassed"):
                    fusion_note["gates_bypassed"] = True
                # V8-20.03 — context + per-facet coverage blocks.
                if isinstance(fstats.get("context"), dict):
                    _graft_block(coverage, "context", fstats["context"])
                if isinstance(fstats.get("per_facet"), list):
                    coverage.facets["per_facet"] = fstats["per_facet"]
        else:
            fusion_note = {
                "status": "unavailable",
                "reason": f"fusion_error:{val}",
                "mode": "union_rrf_fallback",
                "weights_arm": ctx.policy.policy_id,
            }
            fused = _fallback_fuse(kept_outputs)
    else:
        fusion_note = {
            "status": "unavailable",
            "reason": "module_absent",
            "mode": "union_rrf_fallback",
            "weights_arm": ctx.policy.policy_id,
        }
        fused = _fallback_fuse(kept_outputs)
    stage.fields["t_rrf"] = round(_ms(clock, t), 3)

    # ------------------------------------------------------------------
    # S4 — feature rerank over the slim post pool (V8-14.05): the fused
    # pool is trimmed to ``scheduler.post_pool`` BEFORE scoring; items
    # outside the trim stay in ``fused`` for explain but are not scored.
    # Deadline-aware (V8-14.04): below the stage's p95 estimate the
    # scorer runs the cheap provider-free subset; at zero remaining it
    # is skipped outright (RRF passthrough).  The ``budget_exceeded``
    # hook is polled inside the scoring loop.
    # ------------------------------------------------------------------
    t = clock()
    fused_pool = list(fused)[:post_pool]
    rerank_note: dict[str, Any] = {"status": "ok"}
    if len(fused) > len(fused_pool):
        rerank_note["post_pool_trimmed"] = len(fused) - len(fused_pool)
    rf = _lazy_import(_MOD_RERANK)
    if rf is not None and callable(getattr(rf, "score_candidates", None)):
        suppress = fstats.get("constant_signals", ())
        rem = post_remaining_ms()
        cheap = rem < p95.get("rerank_feat", 30.0)
        if rem <= 0:
            rerank_note = {
                "status": "degraded",
                "reason": "deadline",
                "mode": "rrf_passthrough",
            }
            scored = _fallback_scored(fused_pool)
            _degrade("rerank_features")
        else:
            if cheap:
                _degrade("rerank_features_expensive")
            ok, val = _stage_call(
                rf.score_candidates,
                query,
                fused_pool,
                ctx=ctx,
                suppress=suppress,
                n_lanes=len(lane_names),
                cheap=cheap,
                budget_exceeded=post_exceeded,
            )
            if not ok:
                ok, val = _stage_call(
                    rf.score_candidates, query, fused_pool, ctx
                )
            if ok and val is not None:
                vstats = getattr(val, "stats", None) or {}
                if isinstance(vstats, dict):
                    if vstats.get("mode"):
                        rerank_note["mode"] = vstats["mode"]
                    elif cheap:
                        # Scorer accepted the call but reports no mode —
                        # record the pipeline's cheap intent honestly.
                        rerank_note["mode"] = "cheap_subset"
                    if vstats.get("deadline_degraded"):
                        rerank_note["deadline_degraded"] = vstats[
                            "deadline_degraded"
                        ]
                        _degrade("rerank_features_expensive")
                scored = list(val)[:post_pool]
            else:
                rerank_note = {
                    "status": "unavailable",
                    "reason": f"rerank_error:{val}",
                    "mode": "rrf_passthrough",
                }
                scored = _fallback_scored(fused_pool)
    else:
        rerank_note = {
            "status": "unavailable",
            "reason": "module_absent",
            "mode": "rrf_passthrough",
        }
        scored = _fallback_scored(fused_pool)
    stage.fields["t_rerank_feat"] = round(_ms(clock, t), 3)

    # ------------------------------------------------------------------
    # S5 — optional cross-encoder on the head (never crashes).
    # ------------------------------------------------------------------
    t = clock()
    if pool.ce_pool <= 0:
        coverage.rerank["status"] = "skipped"
        coverage.rerank["reason"] = "ce_pool_0"
    elif post_remaining_ms() < p95.get("rerank_ce", 40.0):
        # V8-14.04 — optional CE detail drops before verdict/pack.
        coverage.rerank["status"] = "skipped"
        coverage.rerank["reason"] = "deadline"
        _degrade("rerank_ce")
    else:
        ce = _lazy_import(_MOD_CE)
        hook = None
        if ce is not None:
            hook = getattr(ce, "rerank", None) or getattr(ce, "score_ce", None)
        if not callable(hook):
            coverage.rerank["status"] = "unavailable"
            coverage.rerank["reason"] = "no_ce"
        else:
            ok, val = _stage_call(hook, list(scored[: pool.ce_pool]), query, ctx)
            if ok and val is not None:
                head = list(val)
                scored = head + scored[len(head) :]
                coverage.rerank["status"] = "ok"
                coverage.rerank["pool"] = pool.ce_pool
            else:
                coverage.rerank["status"] = "unavailable"
                coverage.rerank["reason"] = f"ce_error:{val}"
    stage.fields["t_rerank_ce"] = round(_ms(clock, t), 3)

    # ------------------------------------------------------------------
    # S6 — bounded multiplicative boosts.
    # ------------------------------------------------------------------
    t = clock()
    boost_note: dict[str, Any] = {"status": "ok"}
    if post_exceeded():
        # V8-14.04 — boosts are optional detail; scores pass through.
        boost_note = {"status": "skipped", "reason": "deadline"}
        _degrade("boosts")
    else:
        bo = _lazy_import(_MOD_BOOSTS)
        if bo is not None and callable(getattr(bo, "apply_boosts", None)):
            now_us = ctx.query_time_us
            if now_us is None:
                now_us = query.query_time_us or 0
            if as_of_scope != "global":
                # V85-05.06 ``window`` scope — the caller's ``as_of``
                # anchor resolves the relative-expression window (S1)
                # and the verdict (S7, via ``ctx``) but must not
                # re-anchor the recency clock: boosts read wall time.
                # (The global re-anchor measured −0.029 temporal any@10
                # in the b3 arm.) With no ``as_of`` the ctx anchor is
                # already wall time, so this is a no-op off-arm.
                now_us = _wall_now_us()
            ok, val = _stage_call(bo.apply_boosts, scored, query, now_us)
            if ok and val is not None:
                scored = list(val)
            else:
                boost_note = {"status": "unavailable", "reason": f"boost_error:{val}"}
        else:
            boost_note = {"status": "unavailable", "reason": "module_absent"}
    stage.fields["t_boost"] = round(_ms(clock, t), 3)

    # ------------------------------------------------------------------
    # S6b — source-lifecycle currency.  Eligibility withheld the
    # never-answerable dispositions before rank (V7-05.08); the
    # intent-dependent window lands here, before S7, so group verdicts
    # never count an out-of-window source as support (V5-14.12/V7-09.11).
    # ------------------------------------------------------------------
    t = clock()
    scored, currency_note = _apply_source_currency(ctx, scored, query)
    stage.fields["t_currency"] = round(_ms(clock, t), 3)

    # ------------------------------------------------------------------
    # S7 — support verdict (structural triggers only).
    #
    # V8-12.06: lanes the request deadline cut are declared on
    # ``ctx.manifest['deadline_cut_lanes']`` BEFORE classify_groups so
    # their candidates still deliver but cannot change status or
    # answerability.  V8-14.04: below the stage's p95 estimate the group
    # classification (verdict detail) degrades to the structural
    # fallback — the verdict itself is never skipped.
    # ------------------------------------------------------------------
    t = clock()
    ctx.manifest["deadline_cut_lanes"] = sorted(deadline_cut)
    groups: list = []
    verdict = ResultStatus.READY
    missing: Optional[MissingDescriptor] = None
    verdict_note: dict[str, Any] = {"status": "ok", "provisional": False}
    vv = _lazy_import(_MOD_VERDICT)
    if post_remaining_ms() < p95.get("verdict", 20.0):
        verdict, missing = _structural_verdict(scored)
        verdict_note = {
            "status": "degraded",
            "reason": "deadline",
            "provisional": True,
            "mode": "structural_fallback",
        }
        _degrade("verdict_detail")
    elif vv is not None and callable(getattr(vv, "classify_groups", None)) and callable(
        getattr(vv, "result_verdict", None)
    ):
        ok, gv = _stage_call(vv.classify_groups, scored, query, ctx)
        if ok and gv is not None:
            groups = list(gv)
            ok2, rv = _stage_call(
                vv.result_verdict, groups, query,
                ctx.manifest.get("calibration"), ctx=ctx,
            )
            if not ok2:
                # Contract-sketch verdicts keep the 3-arg signature.
                ok2, rv = _stage_call(
                    vv.result_verdict, groups, query,
                    ctx.manifest.get("calibration"),
                )
            if ok2 and rv is not None:
                verdict, missing = rv
                # V8-20.03 — the verdict's own coverage block
                # (status_trigger/answerability/premise_speaker/
                # deadline_cut_lanes) grafts verbatim.
                vdet = getattr(rv, "detail", None)
                if isinstance(vdet, dict):
                    _graft_block(coverage, "verdict", vdet)
            else:
                verdict, missing = _structural_verdict(scored)
                verdict_note = {
                    "status": "unavailable",
                    "reason": f"verdict_error:{rv}",
                    "provisional": True,
                    "mode": "structural_fallback",
                }
        else:
            verdict, missing = _structural_verdict(scored)
            verdict_note = {
                "status": "unavailable",
                "reason": f"group_error:{gv}",
                "provisional": True,
                "mode": "structural_fallback",
            }
    else:
        verdict, missing = _structural_verdict(scored)
        verdict_note = {
            "status": "unavailable",
            "reason": "module_absent",
            "provisional": True,
            "mode": "structural_fallback",
        }
    stage.fields["t_verdict"] = round(_ms(clock, t), 3)

    # Propagate the group's support label onto its member candidates so the
    # pack delivers weak/partial evidence *labeled* (V7-11.03). Without
    # this the pack defaults every item to ``SUPPORTED`` regardless of the
    # verdict — a weak stopword-overlap match would ship as if supported.
    if groups:
        unit_label: dict[str, Any] = {}
        for g in groups:
            lab = getattr(getattr(g, "label", None), "value", getattr(g, "label", None))
            if lab is None:
                continue
            for uid in (getattr(g, "detail", None) or {}).get("members") or ():
                unit_label[uid] = lab
        for s in scored:
            lab = unit_label.get(s.unit_id)
            if lab is not None and isinstance(s.detail, dict):
                s.detail["support"] = lab

    # ------------------------------------------------------------------
    # S8 — pack assembly; raw passthrough when the module is absent.
    # Materialize each scored candidate's deliverable fields first — the
    # lanes only carry signals; ``detail`` gets the unit's
    # quote/speaker/session/occurred for the pack's byte-pinned render.
    # V8-14.04: the per-item enrichment tail is the last droppable work —
    # below the explain p95 it stops filling (``materialize`` degraded);
    # the pack itself is NEVER dropped.
    # ------------------------------------------------------------------
    t = clock()
    try:
        # The deadline hook is polled per item — at zero remaining every
        # fill is skipped, so the tail degrades rather than overruns.
        _unfilled = _materialize_details(
            ctx, scored, budget_exceeded=post_exceeded
        )
        if _unfilled:
            _degrade("materialize")
    except Exception:
        pass  # materialization is best-effort; missing fields stay None
    pack_note: dict[str, Any] = {"status": "ok"}
    pack: Any = None
    pk = _lazy_import(_MOD_PACK)
    if pk is not None and callable(getattr(pk, "assemble_pack", None)):
        ok, val = _stage_call(
            pk.assemble_pack,
            scored,
            query,
            ctx,
            ctx.manifest.get("max_tokens"),
            ctx.manifest.get("limit", _DEFAULT_LIMIT),
            pool.neighbor_window,
        )
        if ok and val is not None:
            pack = val
        else:
            pack_note = {
                "status": "unavailable",
                "reason": f"pack_error:{val}" if not ok else "pack_returned_none",
            }
    else:
        pack_note = {
            "status": "unavailable",
            "reason": "module_absent",
            "mode": "raw_passthrough",
        }
    if pack is None:
        limit = ctx.manifest.get("limit", _DEFAULT_LIMIT)
        pack = {
            "mode": "raw_passthrough",
            "items": list(scored[:limit]),
            "note": "pack module unavailable — scored items passthrough",
        }
    stage.fields["t_pack"] = round(_ms(clock, t), 3)

    # V8-20.03 — the post-lane coverage block: the reserved window plus
    # every stage capability degraded inside it, in drop order.
    post_block: dict[str, Any] = {"R_post_ms": r_post, "degraded": list(post_degraded)}
    over_ms = _ms(clock, t_post0) - r_post
    if over_ms > 0:
        post_block["overrun_ms"] = round(over_ms, 3)
    _graft_block(coverage, "post", post_block)
    # coverage.sql — statements this request issued (per-lane counts
    # summed; per-shape attribution lands when lanes report it).
    _graft_block(
        coverage,
        "sql",
        {
            "statements": float(
                sum(
                    o.stats.get("sql_statements", 0)
                    for o in lane_outputs.values()
                )
            ),
            "by_shape": None,
        },
    )
    # coverage.eligibility — the V8-14.01 snapshot-cache outcome rides
    # the eligible handle's stats when it exposes them.
    _elig = getattr(ctx, "eligible", None)
    _estats = getattr(_elig, "stats", None)
    if isinstance(_estats, dict) and (
        _estats.get("cache") is not None or _estats.get("cache_t_ms") is not None
    ):
        _graft_block(
            coverage,
            "eligibility",
            {"cache": _estats.get("cache"), "t_ms": _estats.get("cache_t_ms")},
        )
    # V8-20.03 — these blocks are present on every search; ``null``
    # where the owning stage did not run (honest "not applicable" —
    # never omitted, never fabricated).
    for _block_name in (
        "post", "sql", "context", "graph", "lexical", "verdict", "eligibility",
    ):
        if _block_name not in vars(coverage):
            setattr(coverage, _block_name, None)

    # Non-lane stage honesty lives under coverage.rerank["stages"] (the
    # ranking/decision machinery block; CE keeps the top-level
    # status/reason shape mandated by the brief).
    coverage.rerank["stages"] = {
        "fusion": fusion_note,
        "features": rerank_note,
        "boosts": boost_note,
        "currency": currency_note,
        "verdict": verdict_note,
        "pack": pack_note,
    }

    # ------------------------------------------------------------------
    # Stage record (§32.17) — S0/S1/S9 are upstream/owned elsewhere.
    # ------------------------------------------------------------------
    stage.fields["t_total"] = round(_ms(clock, t_start), 3)
    # V85-05.07 — request-deadline honesty: the measured over-deadline
    # criterion is elapsed > deadline + 25 ms hard slack (the §04 wall —
    # c3's ``over25``). The flag and overrun ride ``coverage.budget`` so
    # a paired arm run can count over-deadline queries per arm.
    elapsed_ms = _ms(clock, t_start)
    coverage.budget["elapsed_ms"] = round(elapsed_ms, 3)
    if math.isfinite(deadline_ms):
        hard_wall_ms = deadline_ms + _HARD_SLACK_MS
        over_ms = elapsed_ms - hard_wall_ms
        coverage.budget["hard_wall_ms"] = round(hard_wall_ms, 3)
        coverage.budget["over_deadline"] = bool(over_ms > 0.0)
        if over_ms > 0.0:
            coverage.budget["over_by_ms"] = round(over_ms, 3)
    else:
        coverage.budget["hard_wall_ms"] = None
        coverage.budget["over_deadline"] = False
    stage.fields["t_barrier"] = 0.0
    stage.fields["t_analyze"] = 0.0
    stage.fields["t_post"] = 0.0
    stage.fields["sql_statements"] = float(
        sum(o.stats.get("sql_statements", 0) for o in lane_outputs.values())
    )
    stage.fields["bytes_read"] = float(
        sum(o.stats.get("bytes_read", 0) for o in lane_outputs.values())
    )
    stage.fields["snapshots"] = 0.0  # caller's snapshot; we open none
    stage.fields["pool_R"] = float(pool.rerank_pool)
    stage.fields["post_pool"] = float(post_pool)
    stage.fields["items"] = float(len(_pack_item_ids(pack)) or len(scored))
    tokens = getattr(pack, "tokens", None)
    if tokens is None and isinstance(pack, dict):
        tokens = pack.get("tokens")
    stage.fields["tokens"] = float(tokens or 0)
    stage.fields["status"] = verdict.value
    stage.fields["coverage_digest"] = _coverage_digest(coverage)

    explain_payload = None
    if explain:
        explain_payload = _build_explain(
            ctx,
            query,
            lane_order,
            lane_outputs,
            fused,
            scored,
            groups,
            verdict,
            missing,
            pack,
            stage,
            dealt,
            post_pool,
        )

    return PipelineResult(
        lanes=lane_outputs,
        fused=fused,
        scored=scored,
        verdict=verdict,
        missing=missing,
        coverage=coverage,
        stage=stage,
        explain=explain_payload,
        pack=pack,
    )


__all__ = ["PipelineResult", "run_search"]
