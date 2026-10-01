"""Consumer-route quality suite (E64) — real ``verbatim.Memory`` runs.

Measures the §20.11 query mix on a seeded disposable corpus: recall/precision
per category, abstention discipline (``no_answer`` tasks), closure honesty
(``forget_distractor`` forbidden-id violations), supersession freshness
(``update`` tasks), and the add→ready→search→inspect→forget workflow.

Every task stays in its denominator as a scored ``ScoredQuery`` —
including ``error`` outcomes. Claims are emitted as preregistered
records for :func:`eval.v5.stats.evaluate_claim`; the suite itself
reports ``measured`` numbers and marks its support honestly — a small
local corpus is "locally measured", never "qualified" (V5-20.02).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

from .corpus import ConsumerCorpus, corpus_stats, seed_corpus
from .harness import (
    ConsumerEnv,
    ScoredQuery,
    percentiles,
    run_task,
    seed_corpus_env,
    settle,
)
from . import stats as st


@dataclass
class WorkflowProbe:
    """One add→ready→search→inspect→forget lifecycle measurement."""

    add_ms: float = 0.0
    ready_state: str = "unknown"
    ready_ms: float = 0.0
    search_found: bool = False
    inspect_found: bool = False
    inspect_provenance: bool = False
    forget_completed: bool = False
    post_forget_items: Optional[int] = None
    post_forget_warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "add_ms": round(self.add_ms, 3),
            "ready_state": self.ready_state,
            "ready_ms": round(self.ready_ms, 3),
            "search_found": self.search_found,
            "inspect_found": self.inspect_found,
            "inspect_provenance": self.inspect_provenance,
            "forget_completed": self.forget_completed,
            "post_forget_items": self.post_forget_items,
            "post_forget_warnings": list(self.post_forget_warnings),
            "errors": list(self.errors),
        }


def _workflow_probe(env: ConsumerEnv, text: str) -> WorkflowProbe:
    """Drive one full consumer lifecycle through the public facade."""
    p = WorkflowProbe()
    mem = env.memory
    try:
        t0 = time.perf_counter()
        res = mem.add(text, infer=True)
        p.add_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:  # noqa: BLE001
        p.errors = p.errors + (f"add:{type(exc).__name__}",)
        return p
    try:
        t0 = time.perf_counter()
        rd = mem.wait_ready(res, timeout_ms=5000)
        p.ready_ms = (time.perf_counter() - t0) * 1000.0
        p.ready_state = rd.state
    except Exception as exc:  # noqa: BLE001
        p.errors = p.errors + (f"ready:{type(exc).__name__}",)
    # Session-consistent search must see the just-added memory (V5-13).
    probe_tok = text.split()[2] if len(text.split()) > 2 else text
    try:
        sr = mem.search(probe_tok, limit=4)
        p.search_found = any(res.ref == h.ref for h in sr.items) or any(
            res.memory_id and res.memory_id == h.memory_id
            for h in sr.items
        )
    except Exception as exc:  # noqa: BLE001
        p.errors = p.errors + (f"search:{type(exc).__name__}",)
    try:
        ins = mem.inspect(res.ref)
        p.inspect_found = bool(ins.found)
        p.inspect_provenance = bool(ins.provenance)
    except Exception as exc:  # noqa: BLE001
        p.errors = p.errors + (f"inspect:{type(exc).__name__}",)
    try:
        fr = mem.forget(res.ref)
        p.forget_completed = str(getattr(fr, "status", "")) in (
            "completed", "suppressed", "closed",
        ) or bool(getattr(fr, "completed", False))
    except Exception as exc:  # noqa: BLE001
        p.errors = p.errors + (f"forget:{type(exc).__name__}",)
        return p
    try:
        sr2 = mem.search(probe_tok, limit=4)
        p.post_forget_items = len(sr2.items)
        p.post_forget_warnings = tuple(sr2.warnings)
        p.post_forget_items = sum(
            1 for h in sr2.items
            if h.ref == res.ref
            or (res.memory_id and h.memory_id == res.memory_id)
        )
    except Exception as exc:  # noqa: BLE001
        p.errors = p.errors + (f"post_forget:{type(exc).__name__}",)
    return p


def _category_rows(scored: Sequence[ScoredQuery]) -> dict:
    """Per-category recall/precision/abstention with denominators."""
    rows: dict[str, dict] = {}
    for cat in sorted({s.category for s in scored}):
        group = [s for s in scored if s.category == cat]
        recalls = [s.recall for s in group if s.recall is not None]
        precisions = [s.precision for s in group
                      if s.precision is not None]
        abstains = [s.abstain_correct for s in group
                    if s.abstain_correct is not None]
        rows[cat] = {
            "n": len(group),
            "errors": sum(1 for s in group if s.error),
            "recall_mean": (
                sum(recalls) / len(recalls) if recalls else None
            ),
            "precision_mean": (
                sum(precisions) / len(precisions) if precisions else None
            ),
            "abstain_correct": (
                sum(1 for a in abstains if a) if abstains else None
            ),
            "abstain_n": len(abstains),
            "forbidden_hits": sum(len(s.forbidden_hits) for s in group),
            "unclean": sum(1 for s in group if not s.clean),
        }
    return rows


def run_quality_suite(
    corpus: Optional[ConsumerCorpus] = None,
    *,
    memories: int = 64,
    seed: int = 42,
    k: int = 8,
    worker: str = "external",
    workflow_probes: int = 2,
    workdir: Optional[str] = None,
) -> dict:
    """Execute the consumer quality suite; return a report record.

    The record is self-describing: corpus digest, per-task outcomes,
    aggregate rows, workflow probes, support denominators, and the
    preregistered claims this run can (or cannot) evaluate.
    """
    corpus = corpus or seed_corpus(memories=memories, seed=seed)
    env = seed_corpus_env(corpus, workdir=workdir, worker=worker)
    scored: list[ScoredQuery] = []
    try:
        for task in corpus.tasks:
            scored.append(run_task(env, task, k=k))
        probes = [
            _workflow_probe(
                env,
                f"probe-{i} the boiler serial is BR-9{i}7{i}."
            )
            for i in range(workflow_probes)
        ]
        status = None
        try:
            status = env.memory.status()
        except Exception:
            status = None
    finally:
        env.close()

    rows = _category_rows(scored)
    recalls = [s.recall for s in scored if s.recall is not None]
    precisions = [s.precision for s in scored if s.precision is not None]
    n = len(scored)
    n_recall = len(recalls)
    answered = sum(1 for s in scored if s.returned_ids)
    forbidden_total = sum(len(s.forbidden_hits) for s in scored)
    abstain_tasks = [s for s in scored if s.expected_abstain]
    abstain_ok = sum(1 for s in abstain_tasks if s.abstained)
    update_tasks = [s for s in scored if s.category == "update"]
    update_clean = sum(
        1 for s in update_tasks if s.clean and s.recall == 1.0
    )

    latency = percentiles(s.latency_ms for s in scored)

    seeds = [corpus.seed]
    claims = [
        {
            "name": "closure:zero_forbidden_delivery",
            "kind": "zero_failures",
            "failures": forbidden_total,
            "n": n,
            # the claim is the measured upper bound itself (V5-23.08)
            "claimed_upper": st.upper_bound_zero(n) if n else None,
            "seeds_reported": seeds,
            "seeds_executed": seeds,
        },
        {
            "name": "update:current_beats_stale",
            "kind": "point",
            "estimate": (
                update_clean / len(update_tasks) if update_tasks else None
            ),
            "ci": list(
                st.wilson_interval(update_clean, len(update_tasks))
            ) if update_tasks else None,
            "failed": bool(update_tasks) and update_clean < len(update_tasks),
            "seeds_reported": seeds,
            "seeds_executed": seeds,
        },
    ]
    claim_verdicts = [st.evaluate_claim(c).to_dict() for c in claims]

    return {
        "suite": "quality",
        "arm": "verbatim_memory",
        "measured": True,
        "qualification": "locally_measured",
        "corpus": corpus_stats(corpus),
        "support": {
            "tasks_executed": n,
            "tasks_with_recall_gold": n_recall,
            "tasks_answered": answered,
            "errors": sum(1 for s in scored if s.error),
            "add_errors": dict(env.add_errors),
            "drain": dict(env.drain),
            "notes": list(env.notes),
        },
        "aggregate": {
            "recall_at_k": (
                sum(recalls) / len(recalls) if recalls else None
            ),
            "precision_at_k": (
                sum(precisions) / len(precisions) if precisions else None
            ),
            "abstain_correct": {
                "correct": abstain_ok,
                "n": len(abstain_tasks),
                "ci95": list(st.wilson_interval(abstain_ok,
                                               len(abstain_tasks))),
            },
            "forbidden_hits": forbidden_total,
            "unclean_queries": sum(1 for s in scored if not s.clean),
            "update_clean": {
                "clean": update_clean, "n": len(update_tasks),
            },
        },
        "categories": rows,
        "latency_ms": latency,
        "workflow_probes": [p.to_dict() for p in probes],
        "memory_status": (
            status.to_dict() if hasattr(status, "to_dict") else status
        ),
        "claims": claim_verdicts,
        "per_task": [
            {
                "task_id": s.task_id,
                "category": s.category,
                "status": s.status,
                "returned": list(s.returned_ids),
                "expected": list(s.expected_ids),
                "recall": s.recall,
                "abstained": s.abstained,
                "forbidden_hits": list(s.forbidden_hits),
                "latency_ms": round(s.latency_ms, 2),
                "delivered_bytes": s.delivered_bytes,
                "warnings": list(s.warnings),
                "error": s.error,
            }
            for s in scored
        ],
    }


__all__ = ["WorkflowProbe", "run_quality_suite"]
