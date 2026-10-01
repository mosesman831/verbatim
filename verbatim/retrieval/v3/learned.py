"""Bounded learned controller policy (SPEC_V3 §26.03/26.04, §43.04/05).

A learned controller is a *bounded contextual policy*: it maps the same
discretized ``_State`` the deterministic controller sees onto the same
finite action set — never new routes, lanes, or budgets, only selections
among declared variants of the deterministic plan (V3-26.03).

Design:

- The action set is a fixed list of *plan deltas* applied to the
  deterministic ``RouteSet``: shift the effective budget tier one step,
  drop or add declared optional lanes, tighten item/byte ceilings.
  Nothing outside that list is expressible, so no learned choice can
  widen authorization, invent lanes, or exceed the caller's ceilings —
  deltas re-intersect with request limits after application (V3-26.07).
- The policy itself is a tabular Q model keyed by ``state_key`` (the
  published discretized key) with a small feature-prior fallback —
  bounded local statistical inference, no LLM call, microseconds per
  selection (V3-26.04's 5 ms learned bound is trivially met).
- Warm start: an unseen ``state_key`` always selects ``base`` — the
  deterministic plan — so the learned policy can only deviate where
  executed paired-arm measurements support it (V3-26.03 warm start,
  §43.04: no counterfactual outcome labels).
- Artifacts are versioned JSON carrying revision, action table, Q
  values, pull counts, training provenance, and a declared gate status.
  A missing/corrupt artifact yields ``None`` — the caller falls back to
  the deterministic plan and records the fallback honestly (B36).

This module never trains, never touches the network, and never writes;
``eval/v3/g8.py`` produces artifacts offline; ``paired_execution`` arms
and ``recall_v3`` consume them read-only.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

from ...core.types import json_dumps
from ...core.types_v3 import BudgetTier
from . import controller as _ctrl

ARTIFACT_KIND = "learned_policy_v1"

# ---------------------------------------------------------------------------
# bounded action set — RouteSet deltas only (V3-26.03 finite action set)
# ---------------------------------------------------------------------------

_TIER_ORDER = (BudgetTier.LOW, BudgetTier.MID, BudgetTier.HIGH)


def _tier_shift(tier: BudgetTier, delta: int) -> BudgetTier:
    i = _TIER_ORDER.index(tier) + delta
    return _TIER_ORDER[max(0, min(len(_TIER_ORDER) - 1, i))]


def apply_delta(base: _ctrl.RouteSet, state: _ctrl._State,
                 request: Any, action: str) -> _ctrl.RouteSet:
    """Apply one declared action to the deterministic plan.

    Every delta lands inside the caller's ceilings — budgets are
    re-intersected with the request's limits so a learned plan can never
    exceed what the deterministic plan was already allowed (V3-26.07).
    """
    routes = list(base.routes)
    lanes = list(base.lanes)
    budgets = dict(base.budgets)

    if action == "base":
        return base

    if action in ("tier_tighter", "tier_looser"):
        new_tier = _tier_shift(state.tier, -1 if action == "tier_tighter" else 1)
        raised = dict(_ctrl.TIER_BUDGETS[new_tier])
        for key in ("lane_cap", "candidate_cap", "hop_depth", "rerank"):
            budgets[key] = raised[key]
        # ceilings re-intersect with the caller's limits — never loosen
        # beyond what the request itself permits
        for key in ("max_items", "max_bytes", "target_tokens",
                    "deadline_ms"):
            req_limit = getattr(request, key, None)
            if req_limit is not None:
                budgets[key] = min(raised[key], req_limit)
            else:
                budgets[key] = raised[key]
    elif action == "items_half":
        budgets["max_items"] = max(1, budgets["max_items"] // 2)
    elif action == "bytes_half":
        budgets["max_bytes"] = max(512, budgets["max_bytes"] // 2)
        budgets["target_tokens"] = max(
            128, budgets["target_tokens"] // 2
        )
    elif action.startswith("drop_"):
        lane = action[5:]
        lanes = [l for l in lanes if l != lane]
    elif action == "lexical_only":
        lanes = [
            l for l in lanes
            if l in ("exact_id", "lexical", "structured")
        ]
    elif action.startswith("add_"):
        lane = action[4:]
        if lane not in lanes:
            lanes.append(lane)

    return dataclasses.replace(
        base, routes=tuple(routes), lanes=tuple(dict.fromkeys(lanes)),
        budgets=budgets,
    )


ACTIONS: tuple = (
    "base",
    "tier_tighter",
    "tier_looser",
    "items_half",
    "bytes_half",
    "drop_dense",
    "drop_temporal",
    "drop_episode_hierarchy",
    "lexical_only",
    "add_browse",
    "add_freshness_env",
)

def _feature_pairs(state: _ctrl._State) -> tuple[str, ...]:
    return (
        f"qc={state.query_class.value}",
        f"intent={state.action_intent.value}",
        f"size={state.size_band}",
        f"procedures={int(state.procedures)}",
        f"stuck={int(state.stuck)}",
        f"phase={state.session_phase}",
        f"fresh={int(state.freshness_required)}",
        f"tier={state.tier.value}",
        f"ids={int(state.has_identifiers)}",
        f"prefetch={int(state.prefetch)}",
    )


# ---------------------------------------------------------------------------
# artifact (versioned JSON — trained offline, loaded read-only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyArtifact:
    """A trained learned-policy artifact (bounded, auditable JSON).

    ``q`` maps ``state_key`` → ``action_id`` → mean executed reward;
    ``pulls`` records how many executed arms produced each estimate —
    an action with zero pulls at a seen state still scores via the
    feature priors, never a fabricated outcome.
    """

    revision: str
    actions: tuple = ACTIONS
    q: dict = field(default_factory=dict)       # state_key -> {action: reward}
    pulls: dict = field(default_factory=dict)   # state_key -> {action: n}
    priors: dict = field(default_factory=dict)  # feature -> {action: weight}
    trained_on: dict = field(default_factory=dict)
    gate: dict = field(default_factory=lambda: {"status": "not_evaluated"})
    created_us: int = 0

    def to_dict(self) -> dict:
        return {
            "kind": ARTIFACT_KIND,
            "revision": self.revision,
            "actions": list(self.actions),
            "q": self.q,
            "pulls": self.pulls,
            "priors": self.priors,
            "trained_on": self.trained_on,
            "gate": self.gate,
            "created_us": self.created_us,
        }

    def to_json(self) -> str:
        return json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyArtifact":
        if not isinstance(d, dict) or d.get("kind") != ARTIFACT_KIND:
            raise ValueError(f"not a {ARTIFACT_KIND} artifact")
        revision = d.get("revision")
        if not isinstance(revision, str) or not revision:
            raise ValueError("artifact missing revision")
        actions = tuple(d.get("actions") or ())
        if not actions or any(a not in ACTIONS for a in actions):
            raise ValueError("artifact declares actions outside the bounded set")
        for tbl in ("q", "pulls"):
            for skey, amap in (d.get(tbl) or {}).items():
                if not isinstance(amap, dict):
                    raise ValueError(f"{tbl}[{skey!r}] is not an action map")
                unknown = set(amap) - set(actions)
                if unknown:
                    raise ValueError(f"{tbl}[{skey!r}] names unknown actions {sorted(unknown)}")
        for feat, amap in (d.get("priors") or {}).items():
            if not isinstance(amap, dict):
                raise ValueError(f"priors[{feat!r}] is not an action map")
            unknown = set(amap) - set(actions)
            if unknown:
                raise ValueError(
                    f"priors[{feat!r}] names unknown actions {sorted(unknown)}"
                )
        # Non-finite values are silently exploitable (NaN beats max()'s
        # comparisons); every measured table entry must be a finite number.
        import math as _math
        for tbl in ("q", "priors"):
            for skey, amap in (d.get(tbl) or {}).items():
                for act, val in amap.items():
                    if not isinstance(val, (int, float)) or not _math.isfinite(val):
                        raise ValueError(f"{tbl}[{skey!r}][{act!r}] is not finite")
        for skey, amap in (d.get("pulls") or {}).items():
            for act, val in amap.items():
                if not isinstance(val, int) or val < 0:
                    raise ValueError(f"pulls[{skey!r}][{act!r}] is not a count")
        return cls(
            revision=revision,
            actions=actions,
            q=dict(d.get("q") or {}),
            pulls=dict(d.get("pulls") or {}),
            priors=dict(d.get("priors") or {}),
            trained_on=dict(d.get("trained_on") or {}),
            gate=dict(d.get("gate") or {"status": "not_evaluated"}),
            created_us=int(d.get("created_us") or 0),
        )

    @classmethod
    def from_json(cls, raw: str) -> "PolicyArtifact":
        import json as _json
        return cls.from_dict(_json.loads(raw))


def load_artifact(ref: Any) -> Optional[PolicyArtifact]:
    """Load an artifact from a dict payload, JSON string, or file path.

    Returns ``None`` on any missing/corrupt/underspecified input — the
    caller's deterministic fallback then applies (B36); the failure is
    never silently converted into a fake policy.
    """
    try:
        if ref is None:
            return None
        if isinstance(ref, PolicyArtifact):
            return ref
        if isinstance(ref, dict):
            return PolicyArtifact.from_dict(ref)
        if isinstance(ref, str):
            import json as _json
            import os
            if os.path.exists(ref):
                with open(ref, "r", encoding="utf-8") as fh:
                    return PolicyArtifact.from_dict(_json.load(fh))
            return PolicyArtifact.from_dict(_json.loads(ref))
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# selection — bounded local inference (V3-26.04)
# ---------------------------------------------------------------------------


def action_scores(state: _ctrl._State, artifact: PolicyArtifact) -> dict:
    """Score every declared action for ``state``.

    Score = mean executed reward at this ``state_key`` when pulls exist,
    else the sum of feature priors. ``base`` carries a +0 prior so the
    warm start is deterministic unless measured rewards say otherwise.
    """
    skey = _ctrl._state_key(state)
    q = artifact.q.get(skey, {})
    pulls = artifact.pulls.get(skey, {})
    feats = _feature_pairs(state)
    scores: dict[str, float] = {}
    for action in artifact.actions:
        if pulls.get(action, 0) > 0:
            scores[action] = float(q.get(action, 0.0))
        else:
            scores[action] = sum(
                float(artifact.priors.get(f, {}).get(action, 0.0))
                for f in feats
            )
    return scores


def choose(state: _ctrl._State, request: Any, base: _ctrl.RouteSet,
           artifact: PolicyArtifact) -> tuple:
    """Select the bounded action for ``state`` and apply it.

    Returns ``(routeset, action_id, scores)``. The deterministic plan is
    the default: ties and unseen states resolve to ``base`` — deviations
    require strictly positive measured advantage.
    """
    skey = _ctrl._state_key(state)
    if not any(artifact.pulls.get(skey, {}).values()):
        # Warm start (V3-26.03): a state with zero executed pulls selects
        # the deterministic plan — feature priors may inform ordering at
        # seen states, never substitute for a measurement at unseen ones.
        return base, "base", action_scores(state, artifact)
    scores = action_scores(state, artifact)
    best = max(
        artifact.actions,
        key=lambda a: (scores.get(a, 0.0), a == "base"),
    )
    pulls = artifact.pulls.get(skey, {})
    if (best == "base" or pulls.get(best, 0) == 0
            or scores.get(best, 0.0) <= scores.get("base", 0.0)):
        # base itself, an unmeasured action at this state, or no measured
        # advantage — the deterministic plan stands (V3-26.03/§43.04:
        # executed rewards only, never counterfactual labels)
        return base, "base", scores
    return apply_delta(base, state, request, best), best, scores
