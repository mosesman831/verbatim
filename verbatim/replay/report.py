"""What-if reports for the replay laboratory (SPEC_V3 V3-43.03).

A replay run compares recorded decisions against a candidate policy
executed inside a sandbox store. The report shows what *would* change —
routes, lanes, budgets — under the proposed policy. Counters that the
executed stage cannot produce stay honest ``None``s rather than
fabricated numbers; stages the laboratory does not execute are named
explicitly in ``stages_declared_not_executed``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# V3-43.02's replayable stage list. ``replay_routing`` executes the
# controller stages; ``paired_execution`` adds the real delivery path
# (lanes → union → fusion → abstention → packing); ``paired_admission``
# adds the write-side drain (harvesting → screening → admission →
# relation discovery → consolidation jobs). Nothing is declared-but-
# unexecuted at the stage level any more — what remains outside replay
# scope is fresh-inference exploration (V3-43.06/43.09), which is a
# product boundary, not a gap.
STAGES_EXECUTED = (
    "controller_routing",
    "lane_weights",
    "budgets",
    "abstention_thresholds",
    "packing",
    "harvesting",
    "screening",
    "admission",
    "relation_discovery",
    "consolidation_drain",
)
STAGES_DECLARED_NOT_EXECUTED: tuple = ()


@dataclass(frozen=True)
class DecisionDivergence:
    """One recorded decision vs its sandbox re-execution (V3-43.03)."""

    decision_id: str
    state_key: str
    baseline_routes: tuple
    variant_routes: tuple
    lanes_added: tuple
    lanes_removed: tuple
    budget_deltas: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "state_key": self.state_key,
            "baseline_routes": list(self.baseline_routes),
            "variant_routes": list(self.variant_routes),
            "lanes_added": list(self.lanes_added),
            "lanes_removed": list(self.lanes_removed),
            "budget_deltas": dict(self.budget_deltas),
        }


@dataclass(frozen=True)
class WhatIfReport:
    """Aggregate what-if report persisted into ``replay_runs`` (V3-43.03).

    ``model_calls`` is always 0: replay executes recorded decisions under
    local rules only — fresh inference needs explicit intent, consent, and
    budget (V3-43.06) and is not implemented here.
    """

    run_id: str
    scope_id: str
    sandbox_ref: str
    baseline_revision: str
    variant: dict
    decisions_total: int
    replayed: int
    not_replayable: int
    routes_changed: int
    lanes_added: int
    lanes_removed: int
    budget_divergences: int
    divergences: tuple
    latency_ms: float
    model_calls: int = 0
    review_burden: Optional[int] = None  # not computable at routing stage
    estimated_cost_microusd: Optional[int] = None  # local rules: no cost model
    stages_executed: tuple = STAGES_EXECUTED
    stages_declared_not_executed: tuple = STAGES_DECLARED_NOT_EXECUTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scope_id": self.scope_id,
            "sandbox_ref": self.sandbox_ref,
            "baseline_revision": self.baseline_revision,
            "variant": dict(self.variant),
            "decisions_total": self.decisions_total,
            "replayed": self.replayed,
            "not_replayable": self.not_replayable,
            "routes_changed": self.routes_changed,
            "lanes_added": self.lanes_added,
            "lanes_removed": self.lanes_removed,
            "budget_divergences": self.budget_divergences,
            "divergences": [d.to_dict() for d in self.divergences],
            "latency_ms": self.latency_ms,
            "model_calls": self.model_calls,
            "review_burden": self.review_burden,
            "estimated_cost_microusd": self.estimated_cost_microusd,
            "stages_executed": list(self.stages_executed),
            "stages_declared_not_executed": list(
                self.stages_declared_not_executed
            ),
        }
