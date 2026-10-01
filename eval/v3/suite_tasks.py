"""Paired task-execution suite (SPEC_V3 §53.14; gates G7/G8).

The contract: each ``task_runnable`` fixture is a scripted coding task —
the agent must pick exactly one command from a fixed ``choices`` list.
Two arms run under matched budgets on the *same* deterministic policy:

* **control** — no memory context: the scripted agent takes the
  documented ``naive_choice`` default.
* **memory** — a fresh store ingests the task's sources through the
  public path; the chosen baseline answers ``memory_query``; the agent
  picks the first choice whose command text appears in the returned
  evidence, in returned rank order (top-ranked evidence wins).  No
  lexical hit → the naive default.
* **oracle** (diagnostic) — the agent is handed the expected evidence
  text directly.  This arm exists to prove the fixture is winnable and
  to bound the measured effect; it is never counted as a system arm.

``mode="shadow"`` records what the memory arm *would* consult (the
retrieval outcome) without executing the choice — ``memory_correct``
stays ``None``, so shadow runs can never manufacture paired evidence
(§53.14, G5/G8: shadow logs are not task executions).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from . import metrics
from .baselines import (
    Capability,
    CaseEnv,
    QueryOutcome,
    SuiteRun,
    get_baseline,
    prepare_case,
    probe_capabilities,
)
from .corpus import Corpus, CorpusTask, public_task
from .metrics import ScoredRecord
from .suite_retrieval import _propagate_env_notes, _supersession_check


def _returned_texts(task: CorpusTask, out: QueryOutcome) -> List[str]:
    """Text of returned items, in rank order — the scripted agent's
    observable memory context."""
    raw = out.raw
    if raw is not None:
        items = getattr(raw, "items", None)
        if items:  # v2 RecallResult
            return [getattr(i, "text", "") for i in items]
        packs = getattr(raw, "packs", None)
        if packs:  # v3 RecallResultV3
            return [getattr(it, "text", "") for p in packs for it in p.items]
    # naive/vector arms return fixture ids — the raw source text is the item.
    by_id = {s.id: s.text for s in task.setup_sources}
    return [by_id[f] for f in out.returned_ids if f in by_id]


def _choose(texts: Sequence[str], task: CorpusTask) -> Optional[str]:
    """Deterministic policy: first choice command found in evidence text,
    scanning evidence in rank order then choices in fixture order."""
    assert task.task is not None
    for text in texts:
        for choice in task.task.choices:
            cmd = choice.get("command", "")
            if cmd and cmd in text:
                return choice["id"]
    return None


def _expected_texts(task: CorpusTask) -> List[str]:
    want = set(task.expected_evidence_ids)
    return [s.text for s in task.setup_sources if s.id in want]


def run_tasks_suite(
    corpus: Corpus,
    baseline_name: str = "verbatim_v2",
    *,
    k: int = 5,
    mode: str = "paired",
    task_ids: Optional[Sequence[str]] = None,
    capabilities: Optional[dict[str, Capability]] = None,
) -> SuiteRun:
    """Run the paired contract over every ``task_runnable`` fixture."""
    if mode not in ("paired", "shadow"):
        raise ValueError(f"mode must be 'paired' or 'shadow', got {mode!r}")
    caps = capabilities if capabilities is not None else probe_capabilities()
    baseline = get_baseline(baseline_name)
    run = SuiteRun(
        suite="tasks",
        baseline=baseline_name,
        capabilities=caps,
        k=k,
    )
    run.capabilities.update(baseline.capabilities())
    if mode == "shadow":
        run.notes.append(
            "shadow mode: memory-arm choices recorded but NOT executed — "
            "this run cannot support learned-controller usefulness claims"
        )

    supersession_checks: list[bool] = []
    tasks: Iterable[CorpusTask] = (
        (t for t in corpus.runnable if t.task_id in set(task_ids))
        if task_ids is not None
        else corpus.runnable
    )
    for task in tasks:
        assert task.task is not None
        naive = task.task.naive_choice or task.task.choices[0]["id"]
        control_correct = naive == task.task.correct_choice

        # oracle arm — expected evidence handed to the same policy.
        oracle_pick = _choose(_expected_texts(task), task)
        oracle_correct = (
            oracle_pick == task.task.correct_choice
            if oracle_pick is not None else False
        )

        env = prepare_case(
            task, capabilities=caps,
            ingest=getattr(baseline, "ingest_mode", "v2"),
        )
        # The memory arm's probe is the fixture's declared
        # ``task.task.memory_query`` — the retrieval the scripted agent
        # would issue (§53.14).  ``task.query`` is the fallback when a
        # fixture leaves it empty; the two coincide on today's corpus
        # but the fixture field is the contract, not the coincidence.
        mq = task.task.memory_query or task.query
        probe = (
            task if mq == task.query
            else dataclasses.replace(task, query=mq)
        )
        try:
            # Arms get the gold-free public view (correct_choice stays
            # with the scoring code below).
            out = baseline.query(env, public_task(probe), k=k)
        except Exception as exc:
            out = QueryOutcome(
                arm=baseline_name, task_id=task.task_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        sup = _supersession_check(env, task, baseline)
        try:
            if mode == "shadow" or out.unavailable or out.error:
                memory_correct: Optional[bool] = None
            else:
                pick = _choose(_returned_texts(task, out), task) or naive
                memory_correct = pick == task.task.correct_choice
        finally:
            # Preparation notes reach the report even when scoring
            # raised — a silent drain failure must not leave zeroed
            # records unexplained.
            _propagate_env_notes(run, env)
            env.close()

        note = (
            out.unavailable_reason
            or (f"error:{out.error}" if out.error else "")
            or ("shadow" if mode == "shadow" else "paired")
        )
        if sup is not None:
            note = f"{note}; {sup}"
            if not sup.startswith("n/a"):
                supersession_checks.append(
                    (task.supersession["expect"], sup.startswith("ok"))
                )
        rec = ScoredRecord(
            task_id=task.task_id,
            kind="paired_task",
            expected_ids=task.expected_evidence_ids,
            returned_ids=out.returned_ids[:k],
            abstained=out.abstained,
            expected_abstain=False,
            returned_items=out.n_items,
            memory_correct=memory_correct,
            control_correct=control_correct,
            oracle_correct=oracle_correct,
            note=note,
        )
        run.record(rec, out)

    run.metrics = metrics.summarize(run.records)
    run.metrics["execution_mode"] = {"value": mode}
    if supersession_checks:
        applied = [ok for k, ok in supersession_checks if k == "applied"]
        none_ = [ok for k, ok in supersession_checks if k == "none"]
        run.metrics["supersession"] = {
            "checked": len(supersession_checks),
            "correct": sum(ok for _, ok in supersession_checks),
            "applied": f"{sum(applied)}/{len(applied)}",
            "none": f"{sum(none_)}/{len(none_)}",
        }
    return run


__all__ = ["run_tasks_suite"]
