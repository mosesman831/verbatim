"""Retrieval-quality suite (SPEC_V3 §53 retrieval workload; gates G3/G5).

Per case: a **fresh store** is prepared through the public ingest path
(``prepare_case``), the selected baseline answers the task's query, and
the outcome is scored into a :class:`ScoredRecord`:

* ``returned_ids`` — fixture ids evidenced by returned items (k-truncated)
* ``abstained`` — typed abstention warnings only, never "empty list"
* ``unauthorized_items`` — items from the ``other`` scope (G1 violations)
* ``poisoned_returned`` — adversarial sources that reached results
* ``grounded_items`` — items carrying a resolvable evidence reference
  (claim/span/object ids — the deep derivation-root check lives in
  ``suite_grounding``)

Errors stay in the denominator as zero-scoring records; an unavailable
baseline produces records marked ``unavailable`` plus a lane entry in
``unavailable_lanes`` — explicit, never a silent skip.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from . import metrics
from .baselines import (
    Baseline,
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


def score_outcome(
    env: CaseEnv, task: CorpusTask, out: QueryOutcome, *, k: int
) -> ScoredRecord:
    """Map a baseline outcome to a scored record (shared by the grounding
    and security suites)."""
    returned = set(out.returned_ids)
    unauthorized = set(task.unauthorized_source_ids) & returned
    poisoned = set(task.poisoned_source_ids) & returned
    # Grounding: an item is grounded when its evidence ref resolves to a
    # real ingested source (fixture id or engine source_id present in the
    # case's source map / ingest receipts).
    real_sources = set(env.source_map.values())
    grounded = 0
    fabricated = 0
    for ref in out.evidence_refs:
        sid = ref.get("source_id")
        ext = ref.get("external_id")
        if (sid and sid in real_sources) or (ext and ext in env.source_map):
            grounded += 1
        elif sid is None and ext is None:
            fabricated += 1
        else:
            fabricated += 1
    note = out.unavailable_reason or (f"error:{out.error}" if out.error else "")
    return ScoredRecord(
        task_id=task.task_id,
        kind=task.kind,
        expected_ids=task.expected_evidence_ids,
        returned_ids=out.returned_ids[:k],
        abstained=out.abstained,
        expected_abstain=task.expected_abstain,
        returned_items=out.n_items,
        grounded_items=grounded,
        fabricated_items=fabricated,
        unauthorized_items=len(unauthorized),
        poisoned_returned=len(poisoned),
        poisoned_total=len(task.poisoned_source_ids),
        note=note,
    )


#: ``env.notes`` prefixes that mark *constant informational disclosures*
#: — notes repeated verbatim on every prepared case (the F27 reindex
#: rebuild).  They are emitted once per run; every other env note keeps
#: its ``<task_id>:`` attribution so a systemic failure renders once per
#: affected case — the blast radius stays visible.
_ENV_CONSTANT_DISCLOSURES = ("fts projection rebuilt",)


def _propagate_env_notes(run: SuiteRun, env: CaseEnv) -> None:
    """Propagate every case-preparation note into ``run.notes``.

    A silent drain/ingest/admission/review/quarantine failure must never
    leave a zeroed record unexplained in the report — so *no* env note
    is dropped.  Failure and measurement-invalidating notes (drain
    failures, unapplied reviews, other-scope partitions that produced
    no claims) keep per-case attribution; constant disclosures collapse
    to a single run-level entry so the notes section stays readable.
    """
    for n in env.notes:
        entry = (
            n if n.startswith(_ENV_CONSTANT_DISCLOSURES)
            else f"{env.task_id}: {n}"
        )
        if entry not in run.notes:
            run.notes.append(entry)


def _supersession_check(
    env: CaseEnv, task: CorpusTask, baseline: Any
) -> Optional[str]:
    """Verify the fixture's supersession expectation against the prepared
    store — the G3 corpus-level measurement.

    ``expect="applied"`` asserts the predecessor source's claim reached
    terminal ``superseded`` state after the operator review pass;
    ``expect="none"`` asserts the run created zero ``supersede``
    proposals (false-supersession bound). Only measurable on arms that
    ingest through the v3 write channel — the detector runs at admit
    time on v3-envelope claims; other arms report ``n/a`` honestly.
    """
    sup = task.supersession
    if sup is None:
        return None
    if getattr(baseline, "ingest_mode", "v2") != "v3":
        return "n/a:supersession(detector requires v3-ingest arm)"
    expect = sup["expect"]
    with env.store.read() as conn:
        if expect == "none":
            n = conn.execute(
                "SELECT COUNT(*) FROM reviews"
                " WHERE json_extract(proposed_effect_json, '$.effect')"
                "   = 'supersede'"
            ).fetchone()[0]
            return (
                "ok:supersede none" if n == 0
                else f"fail:supersede {n} proposals created, expected 0"
            )
        sid = env.source_map.get(sup["predecessor"])
        if sid is None:
            return "fail:predecessor source not ingested"
        rows = conn.execute(
            "SELECT DISTINCT ce.claim_id FROM claim_evidence ce"
            " JOIN spans s ON s.span_id = ce.span_id"
            " WHERE s.source_id = ?",
            (sid,),
        ).fetchall()
        if not rows:
            return "fail:no claim derived from predecessor source"
        states = []
        for (cid,) in rows:
            h = conn.execute(
                "SELECT state FROM claim_revisions"
                " WHERE claim_id = ? AND recorded_until IS NULL"
                " ORDER BY revision DESC LIMIT 1",
                (cid,),
            ).fetchone()
            states.append(h[0] if h else "missing")
        if all(s == "superseded" for s in states):
            return "ok:supersede applied"
        return f"fail:predecessor states {states}, expected superseded"


def run_retrieval_suite(
    corpus: Corpus,
    baseline_name: str,
    *,
    k: int = 5,
    task_ids: Optional[Sequence[str]] = None,
    capabilities: Optional[dict[str, Capability]] = None,
    cfg_overrides: Optional[dict[str, Any]] = None,
) -> SuiteRun:
    """Score one baseline over every corpus task with a query."""
    caps = capabilities if capabilities is not None else probe_capabilities()
    baseline = get_baseline(baseline_name)
    run = SuiteRun(
        suite="retrieval",
        baseline=baseline_name,
        capabilities=caps,
        k=k,
    )
    run.capabilities.update(baseline.capabilities())

    supersession_checks: list[bool] = []
    tasks: Iterable[CorpusTask] = (
        (t for t in corpus.tasks if t.task_id in set(task_ids))
        if task_ids is not None
        else corpus.tasks
    )
    for task in tasks:
        env = prepare_case(
            task, capabilities=caps, cfg_overrides=cfg_overrides,
            ingest=getattr(baseline, "ingest_mode", "v2"),
        )
        try:
            # Arms get the gold-free public view; the real CorpusTask
            # (expected ids, poison/scope gold, supersession) stays with
            # scoring below — an arm reading gold raises AttributeError.
            out = baseline.query(env, public_task(task), k=k)
        except Exception as exc:  # baseline crashed — keep the failure
            out = QueryOutcome(
                arm=baseline_name, task_id=task.task_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        rec = score_outcome(env, task, out, k=k)
        sup = _supersession_check(env, task, baseline)
        if sup is not None:
            import dataclasses
            rec = dataclasses.replace(
                rec, note=f"{rec.note}; {sup}" if rec.note else sup
            )
            if not sup.startswith("n/a"):
                supersession_checks.append(
                    (task.supersession["expect"], sup.startswith("ok"))
                )
        run.record(rec, out)
        # Every preparation note reaches the run — a silent drain
        # failure must not leave zeroed records unexplained.
        _propagate_env_notes(run, env)
        env.close()

    run.metrics = metrics.summarize(run.records)
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


__all__ = ["run_retrieval_suite", "score_outcome"]
