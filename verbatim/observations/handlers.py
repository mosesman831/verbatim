"""Job handler for the ``consolidate`` pipeline kind (SPEC_V3 §23, §40;
SPEC_V4 §24, §42).

``handle_consolidate`` runs a consolidation pass over the job's scope
inside a single generation-fenced transaction. Idempotency is the
derivations contract, not the queue: a redelivered job re-derives the
same deterministic observation ids and evidence sets and writes nothing
new — a new revision appears only when the support set actually changed
(V3-17.02/17.03, V3-23.04).

``input_refs`` options (all optional):
- ``producer`` — producer id (default ``slot_aggregate_v1``)
- ``min_proof`` — family-distinct support threshold (default 2, V3-23.01)
- ``enabled`` — the ``observations.enabled`` gate (default true;
  false fails CAPABILITY_UNAVAILABLE, V3-23.09)
- ``window`` — trigger-scoped pass bounds (V4-24.01): ``since_seq``,
  ``claim_ids``, ``max_inputs``, ``max_outputs``, ``retention``, and
  ``expected_generation`` (compared to the job's lease generation — a
  stale trigger fails STALE_PROPOSAL instead of writing from an
  abandoned view, V4-24.09).
- ``reflect`` — optional bounded reflection after consolidation
  (V4-24.06): ``true`` runs with default bounds, or a dict of
  ``ReflectBudget`` fields (``max_iterations`` default 3 per V4-27.06,
  ``max_regions``, ``max_outputs``, ``deadline_s``). Reflection is
  read-only over memory inputs and emits labeled hypotheses only —
  it never touches grants, trust, receipts, or canonical bytes (C50).

Each pass records a ``consolidation_pass``/``reflection_pass`` event
with its report — the durable "why" for every derivative written or
retired (V4-24.07).
"""

from __future__ import annotations

from typing import Any, Optional

from ..core.types import ErrorCode, VerbatimError
from ..storage.repos import EventsRepo
from .aggregate import DEFAULT_MIN_PROOF, SLOT_AGGREGATE_V1
from .consolidate import ConsolidationWindow, consolidate, consolidate_windowed
from .reflect import ReflectBudget, reflect


def _window_from_refs(refs: dict[str, Any]) -> Optional[ConsolidationWindow]:
    raw = refs.get("window")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise VerbatimError(
            ErrorCode.VALIDATION, "input_refs.window must be an object"
        )
    claim_ids = raw.get("claim_ids") or ()
    if not isinstance(claim_ids, (list, tuple)):
        raise VerbatimError(
            ErrorCode.VALIDATION, "input_refs.window.claim_ids must be a list"
        )
    return ConsolidationWindow(
        since_seq=raw.get("since_seq"),
        claim_ids=tuple(str(c) for c in claim_ids),
        max_inputs=raw.get("max_inputs"),
        max_outputs=raw.get("max_outputs"),
        expected_generation=raw.get("expected_generation"),
        retention=raw.get("retention"),
    )


def _reflect_budget(refs: dict[str, Any]) -> Optional[ReflectBudget]:
    raw = refs.get("reflect")
    if raw is None or raw is False:
        return None
    if raw is True:
        return ReflectBudget()
    if not isinstance(raw, dict):
        raise VerbatimError(
            ErrorCode.VALIDATION, "input_refs.reflect must be true or an object"
        )
    return ReflectBudget(
        max_iterations=int(raw.get("max_iterations", 3)),
        max_regions=int(raw.get("max_regions", 256)),
        max_outputs=int(raw.get("max_outputs", 64)),
        deadline_s=raw.get("deadline_s"),
    )


def handle_consolidate(job: dict, owner: str, ingester) -> None:
    """Run ``consolidate`` for ``job["scope_id"]`` (kind ``consolidate``).

    The leased queue row supplies scope + options; ``ingester`` supplies
    the store and the job queue used for the commit-time lease fence.
    """
    refs = job.get("input_refs") or {}
    scope_id = job.get("scope_id")
    producer = refs.get("producer", SLOT_AGGREGATE_V1)
    min_proof = int(refs.get("min_proof", DEFAULT_MIN_PROOF))
    enabled = bool(refs.get("enabled", True))
    store = ingester.store
    with store.tx() as conn:
        jobs = getattr(ingester, "jobs", None)
        if (
            jobs is not None
            and job.get("job_id") is not None
            and job.get("generation") is not None
        ):
            jobs.assert_lease(conn, job["job_id"], owner, job["generation"])
        window = _window_from_refs(refs)
        if window is not None:
            expected = window.expected_generation
            if (
                expected is not None
                and job.get("generation") is not None
                and int(expected) != int(job["generation"])
            ):
                raise VerbatimError(
                    ErrorCode.STALE_PROPOSAL,
                    "consolidate trigger superseded — job generation "
                    f"{job['generation']} != expected {expected} (V4-24.09)",
                )
            rep = consolidate_windowed(
                conn,
                scope_id,
                window=window,
                producer=producer,
                min_proof=min_proof,
                enabled=enabled,
            )
            result: dict[str, Any] = rep.as_dict()
        else:
            result = consolidate(
                conn,
                scope_id,
                producer=producer,
                min_proof=min_proof,
                enabled=enabled,
            )
        job_id = job.get("job_id") or "consolidate"
        EventsRepo(store).append(
            conn,
            scope_id,
            "consolidation_pass",
            f"job:{job_id}",
            {
                "producer": producer,
                "min_proof": min_proof,
                "windowed": window is not None,
                "result": result if isinstance(result, dict) else result,
            },
            ingester.policy.policy_version,
        )
        budget = _reflect_budget(refs)
        if budget is not None:
            rep = reflect(
                conn, scope_id, budget=budget, enabled=enabled
            )
            EventsRepo(store).append(
                conn,
                scope_id,
                "reflection_pass",
                f"job:{job_id}",
                {"budget": {
                    "max_iterations": budget.max_iterations,
                    "max_regions": budget.max_regions,
                    "max_outputs": budget.max_outputs,
                    "deadline_s": budget.deadline_s,
                }, "result": rep.as_dict()},
                ingester.policy.policy_version,
            )
