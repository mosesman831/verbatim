"""Deterministic v3 controller (SPEC_V3 §26, V3-26.01–26.14).

``plan_routes`` maps a discretized request state — query class, task
context, action intent, store-size band, procedure availability, stuck
signal, session phase, freshness requirement, budget tier — through a
versioned policy table to a finite action set: routes, lane sets,
candidate caps, and pack budgets (V3-26.01/26.02).

Hard rules:

- The controller NEVER selects scopes, audiences, or verbs and never
  widens authorization (V3-26.07): its output is a read-path plan only.
- Tier values are ceilings intersected with the caller's stricter limits
  (V3-26.13): ``max_items``/``max_bytes``/``target_tokens``/``deadline_ms``
  on the request always win.
- Route priority (V3-26.13): explicit identifiers → current/historical
  flat retrieval → procedure/failure additions → verify additions →
  optional expansion. ``investigate`` is never selected implicitly.
- Every decision is logged to ``routing_decisions`` with the discretized
  state key, policy revision, route set, lane set, budgets, and (later)
  result sizes — replayable per §43 (V3-26.06).
- ``learned_shadow`` mode records what the learned policy WOULD have
  chosen — the deterministic plan is still the one returned and applied
  (V3-26.03/26.11). No network access, no synchronous learning.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional

from ...core.types import json_dumps
from ...core.types_v3 import (
    ActionIntent,
    BudgetTier,
    QueryClass,
    RecallRequestV3,
    Route,
)
from ...core.time import now_us

POLICY_REVISION = "routes_v3_r1"

# ---------------------------------------------------------------------------
# budget tiers (V3-26.05): lane caps, hop depths, rerank counts, deadlines
# ---------------------------------------------------------------------------

TIER_BUDGETS: dict[BudgetTier, dict] = {
    BudgetTier.LOW: {
        "lane_cap": 24,
        "candidate_cap": 64,
        "hop_depth": 0,
        "rerank": 0,
        "deadline_ms": 120,
        "max_items": 6,
        "max_bytes": 4096,
        "target_tokens": 1024,
    },
    BudgetTier.MID: {
        "lane_cap": 40,
        "candidate_cap": 128,
        "hop_depth": 1,
        "rerank": 16,
        "deadline_ms": 200,
        "max_items": 8,
        "max_bytes": 6000,
        "target_tokens": 1536,
    },
    BudgetTier.HIGH: {
        "lane_cap": 64,
        "candidate_cap": 256,
        "hop_depth": 2,
        "rerank": 32,
        "deadline_ms": 400,
        "max_items": 12,
        "max_bytes": 12000,
        "target_tokens": 3072,
    },
}


@dataclass(frozen=True)
class RouteSet:
    """The controller's plan: routes + lanes + effective budgets (§26.01)."""

    routes: tuple
    lanes: tuple
    budgets: dict
    state_key: str
    policy_revision: str = POLICY_REVISION
    shadow: Optional[dict] = None  # hypothetical learned choice (never applied)


@dataclass(frozen=True)
class _State:
    """Discretized controller inputs (V3-26.02 published table domain)."""

    query_class: QueryClass
    action_intent: ActionIntent
    size_band: str          # empty|small|medium|large
    procedures: bool        # authorized procedures exist
    stuck: bool
    session_phase: str      # start|continuation|stuck|end|none
    freshness_required: bool
    tier: BudgetTier
    has_identifiers: bool
    memory_kinds: tuple
    prefetch: bool          # automatic prefetch (not an explicit recall)


# ---------------------------------------------------------------------------
# the published decision table (V3-26.02)
# ---------------------------------------------------------------------------

_BASE_LANES = ("exact_id", "lexical", "structured")

_CLASS_ROUTES: dict[QueryClass, tuple] = {
    QueryClass.EXACT_LOOKUP: (Route.EXACT,),
    QueryClass.CURRENT_STATE: (Route.CURRENT,),
    QueryClass.PAST_STATE: (Route.HISTORICAL,),
    QueryClass.TIMELINE: (Route.HISTORICAL,),
    QueryClass.RELATIONSHIP: (Route.CURRENT, Route.GRAPH),
    QueryClass.CAUSE: (Route.CURRENT, Route.GRAPH),
    QueryClass.PROCEDURE: (Route.PROCEDURE, Route.CURRENT),
    QueryClass.FAILURE: (Route.FAILURE, Route.CURRENT),
    QueryClass.ARCHIVE: (Route.HISTORICAL,),
    QueryClass.EXPLORATORY: (Route.CURRENT,),
}

_CLASS_LANES: dict[QueryClass, tuple] = {
    QueryClass.EXACT_LOOKUP: ("exact_id", "structured", "lexical"),
    QueryClass.CURRENT_STATE: _BASE_LANES + ("temporal", "dense"),
    QueryClass.PAST_STATE: _BASE_LANES + ("temporal",),
    QueryClass.TIMELINE: ("temporal", "browse", "lexical"),
    QueryClass.RELATIONSHIP: _BASE_LANES + ("graph", "dense", "temporal"),
    QueryClass.CAUSE: _BASE_LANES + ("graph", "temporal", "episode_hierarchy"),
    QueryClass.PROCEDURE: (
        "procedural_signature", "exact_id", "structured", "lexical",
        "episode_hierarchy",
    ),
    QueryClass.FAILURE: (
        "procedural_signature", "exact_id", "lexical", "structured",
        "episode_hierarchy", "freshness_env",
    ),
    QueryClass.ARCHIVE: ("browse", "exact_id", "lexical", "temporal"),
    QueryClass.EXPLORATORY: _BASE_LANES + ("dense", "temporal",
                                           "episode_hierarchy"),
}


def _size_band(conn: sqlite3.Connection, scope_ids: list) -> str:
    """Discretized authorized store size — never a cross-scope count."""
    if not scope_ids:
        return "empty"
    marks = ",".join("?" * len(scope_ids))
    row = conn.execute(
        f"SELECT COUNT(*) FROM claims WHERE scope_id IN ({marks})",
        scope_ids,
    ).fetchone()
    n = int(row[0]) if row else 0
    if n == 0:
        return "empty"
    if n < 100:
        return "small"
    if n < 10000:
        return "medium"
    return "large"


def _procedures_available(conn: sqlite3.Connection, scope_ids: list) -> bool:
    if not scope_ids:
        return False
    try:
        marks = ",".join("?" * len(scope_ids))
        row = conn.execute(
            "SELECT 1 FROM procedures"
            f" WHERE scope_id IN ({marks})"
            " AND state NOT IN ('retired','deprecated') LIMIT 1",
            scope_ids,
        ).fetchone()
    except sqlite3.Error:
        return False
    return row is not None


def discretize(conn: sqlite3.Connection, request: RecallRequestV3,
               query_class: QueryClass, scope_ids: list,
               prefetch: bool = False) -> _State:
    """Reduce the request to the finite controller state (V3-26.02)."""
    task = request.task
    intent = task.action_intent if task is not None else ActionIntent.NONE
    phase = (task.session_phase if task is not None else None) or "none"
    stuck = bool(task.stuck) if task is not None else False
    return _State(
        query_class=query_class,
        action_intent=intent,
        size_band=_size_band(conn, scope_ids),
        procedures=_procedures_available(conn, scope_ids),
        stuck=stuck,
        session_phase=phase,
        freshness_required=request.freshness_required,
        tier=request.budget_tier,
        has_identifiers=bool(request.entity_ids),
        memory_kinds=tuple(request.memory_kinds),
        prefetch=prefetch,
    )


def _state_payload(state: _State) -> dict:
    """The replayable discretized state — same fields the key digests."""
    return {
        "qc": state.query_class.value,
        "intent": state.action_intent.value,
        "size": state.size_band,
        "procedures": state.procedures,
        "stuck": state.stuck,
        "phase": state.session_phase,
        "fresh": state.freshness_required,
        "tier": state.tier.value,
        "ids": state.has_identifiers,
        "kinds": sorted(k.value for k in state.memory_kinds),
        "prefetch": state.prefetch,
    }


def _state_json(state: _State, request: RecallRequestV3) -> str:
    """Full replay record: pinned state + the caller ceilings plan_routes
    intersects — enough to re-execute the identical decision offline
    (V3-26.06/§43: decisions are replayable, so their inputs must be
    recoverable; query text itself is never stored — metadata only)."""
    return json_dumps(
        {
            "state": _state_payload(state),
            "limits": {
                "max_items": request.max_items,
                "max_bytes": request.max_bytes,
                "target_tokens": request.target_tokens,
                "deadline_ms": request.deadline_ms,
            },
        }
    )


def _state_key(state: _State) -> str:
    """The discretized key recorded for replay (§26.06, §43)."""
    digest = hashlib.sha256(
        json_dumps(_state_payload(state)).encode("utf-8")
    ).hexdigest()[:16]
    return f"{state.query_class.value}:{state.size_band}:{digest}"


def plan_routes(request: RecallRequestV3, plan: Any,
                task_ctx: Any, state: _State,
                tables: Optional[dict] = None) -> RouteSet:
    """Pure deterministic table lookup → RouteSet (no I/O, no clock).

    ``plan`` is the analyzed query plan; ``task_ctx`` the optional task
    context; ``state`` the discretized inputs. The same state always maps
    to the same route set under one policy revision (V3-26.02).

    ``tables`` optionally overrides the published policy tables —
    ``{"class_routes": map, "class_lanes": map, "tier_budgets": map}`` —
    so the replay laboratory can re-execute a recorded decision under a
    candidate policy without mutating production constants (V3-43.02).
    """
    class_routes = _CLASS_ROUTES
    class_lanes = _CLASS_LANES
    tier_budgets = TIER_BUDGETS
    if tables is not None:
        class_routes = tables.get("class_routes", class_routes)
        class_lanes = tables.get("class_lanes", class_lanes)
        tier_budgets = tables.get("tier_budgets", tier_budgets)
    routes: list = list(class_routes.get(
        state.query_class, (Route.CURRENT,)
    ))
    lanes: list = list(class_lanes.get(
        state.query_class, _BASE_LANES
    ))

    # V3-26.08 adjustments (ordered by the §26.13 priority chain):
    # prefetch/no-memory → none; continuation → working; declared task
    # families → procedure; repeated-action/stuck → failure; consequential
    # intent → verify.
    if state.prefetch and state.size_band == "empty":
        routes = [Route.NONE]
        lanes = []
    if state.session_phase == "continuation" and Route.WORKING not in routes:
        routes.append(Route.WORKING)
        lanes.append("working")
    if state.procedures and state.query_class not in (
        QueryClass.PROCEDURE, QueryClass.FAILURE,
    ) and state.query_class in (
        QueryClass.CURRENT_STATE, QueryClass.EXPLORATORY, QueryClass.CAUSE,
    ):
        routes.append(Route.PROCEDURE)
        lanes.append("procedural_signature")
    if (state.stuck or state.session_phase == "stuck") and (
        Route.FAILURE not in routes
    ):
        routes.append(Route.FAILURE)
        if "freshness_env" not in lanes:
            lanes.append("freshness_env")
        if "procedural_signature" not in lanes:
            lanes.append("procedural_signature")
    if (
        state.freshness_required
        or state.action_intent in (ActionIntent.WRITE, ActionIntent.IRREVERSIBLE)
    ) and Route.VERIFY not in routes:
        routes.append(Route.VERIFY)
        if "freshness_env" not in lanes:
            lanes.append("freshness_env")
    if state.has_identifiers and Route.EXACT not in routes:
        routes.insert(0, Route.EXACT)
        if "exact_id" not in lanes:
            lanes.insert(0, "exact_id")
    if state.query_class == QueryClass.EXPLORATORY and "browse" not in lanes:
        lanes.append("browse")
    # capability-declared optional lanes stay declared; absence is reported
    # by the lane itself, never silently dropped from the plan (V3-28.06).

    budgets = dict(tier_budgets[state.tier])
    # ceilings intersect with the caller's stricter limits (V3-26.13)
    budgets["max_items"] = min(budgets["max_items"], request.max_items)
    budgets["max_bytes"] = min(budgets["max_bytes"], request.max_bytes)
    budgets["target_tokens"] = min(
        budgets["target_tokens"], request.target_tokens
    )
    budgets["deadline_ms"] = min(
        budgets["deadline_ms"], request.deadline_ms
    )
    return RouteSet(
        routes=tuple(routes),
        lanes=tuple(dict.fromkeys(lanes)),
        budgets=budgets,
        state_key=_state_key(state),
    )


# ---------------------------------------------------------------------------
# decision logging (V3-26.06) — inside the caller's write transaction
# ---------------------------------------------------------------------------


def log_decision(
    conn: sqlite3.Connection,
    scope_id: str,
    routeset: RouteSet,
    result_sizes: Optional[dict] = None,
    state: Optional[_State] = None,
    request: Optional[RecallRequestV3] = None,
) -> str:
    """Insert one ``routing_decisions`` row; returns ``decision_id``.

    The caller runs this inside ``store.tx()`` — this function performs no
    commit of its own so the log is atomic with the rest of the recall
    record. Idempotence: ``decision_id`` is content-derived; a replayed
    insert of the identical row is a PK no-op retry error the caller may
    treat as already-logged.

    ``state``+``request`` pin the replayable inputs into ``state_json``
    (V3-26.06): without them the row records the outcome but cannot be
    re-executed offline — rows predating the column report
    ``not_replayable`` in the laboratory rather than fabricating inputs.
    """
    created = now_us()
    digest = hashlib.sha256(
        (
            f"{scope_id}|{routeset.state_key}|{routeset.policy_revision}|"
            f"{','.join(r.value for r in routeset.routes)}|"
            f"{','.join(routeset.lanes)}|{created}"
        ).encode("utf-8")
    ).hexdigest()[:32]
    decision_id = f"rd_{digest}"
    conn.execute(
        "INSERT INTO routing_decisions"
        " (decision_id, scope_id, state_key, routes_json, lane_set_json,"
        "  budgets_json, state_json, policy_revision, result_sizes_json,"
        "  created_us)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            decision_id,
            scope_id,
            routeset.state_key,
            json_dumps([r.value for r in routeset.routes]),
            json_dumps(list(routeset.lanes)),
            json_dumps(routeset.budgets),
            (
                _state_json(state, request)
                if state is not None and request is not None
                else "{}"
            ),
            routeset.policy_revision,
            json_dumps(result_sizes or {}),
            created,
        ),
    )
    return decision_id


def record_shadow(
    conn: sqlite3.Connection,
    scope_id: str,
    routeset: RouteSet,
    state: Optional[_State] = None,
    request: Optional[RecallRequestV3] = None,
) -> Optional[str]:
    """``learned_shadow``: log the hypothetical learned choice only.

    The returned RouteSet still carries the deterministic plan — the
    shadow row records ``policy_revision='learned_shadow_v1'`` so replay
    can compare without ever applying the learned choice (V3-26.03).
    """
    if routeset.shadow is None:
        return None
    shadow = RouteSet(
        routes=tuple(routeset.shadow.get("routes", routeset.routes)),
        lanes=tuple(routeset.shadow.get("lanes", routeset.lanes)),
        budgets=dict(routeset.shadow.get("budgets", routeset.budgets)),
        state_key=routeset.state_key,
        policy_revision="learned_shadow_v1",
    )
    return log_decision(conn, scope_id, shadow,
                        result_sizes={"shadow": 1},
                        state=state, request=request)
