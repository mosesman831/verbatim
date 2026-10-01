"""Grounding & abstention suite (SPEC_V3 §29, §17.04; gates G4-adjacent/G3).

Two measurements per returned item and per abstention case:

* **Resolvable evidence.**  Every returned item must resolve to a real
  derivation root or a real source-level evidence id.  The check prefers
  ``verbatim.derivations.roots`` (evidence-plane ancestry); when the
  module is unavailable — or the store has no derivation edges for the
  object — the check falls back to source-level evidence ids and records
  the fallback as a capability note.  An item with no resolvable ref is
  counted ``fabricated`` — fabricated prose never counts as grounded.
* **Typed abstention.**  Cases marked ``expected_abstain`` must produce a
  typed abstention (``no_signal`` / ``insufficient_support`` /
  ``abstained_uncovered_terms`` warnings or the v3 ``abstained`` flag).
  Answering them with items is measured as spurious output.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable, Optional, Sequence, Tuple

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
from .suite_retrieval import _propagate_env_notes

_EVIDENCE_ROOT_KINDS = frozenset({"source", "source_revision", "span"})


def _derivations_module():
    try:
        import verbatim.derivations as drv
        return drv if hasattr(drv, "roots") else None
    except Exception:
        return None


def _object_exists(conn: sqlite3.Connection, kind: str, oid: str) -> bool:
    """Cheap existence probe on the object's own table (read-only)."""
    table, col = {
        "claim": ("claims", "claim_id"),
        "source": ("sources", "source_id"),
        "span": ("spans", "span_id"),
        "source_revision": ("source_revisions", "source_id"),
    }.get(kind, (None, None))
    if table is None:
        return False
    try:
        row = conn.execute(
            f"SELECT 1 FROM {table} WHERE {col} = ? LIMIT 1", (oid,)
        ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def _resolve_ref(
    conn: sqlite3.Connection,
    env: CaseEnv,
    ref: dict[str, Any],
    drv: Any,
) -> Tuple[bool, str]:
    """``(grounded, via)`` for one returned item's evidence ref.

    ``via`` ∈ ``derivation_roots`` | ``source_evidence`` | ``none``.
    """
    kind = ref.get("object_kind") or ("claim" if ref.get("claim_id") else "source")
    oid = ref.get("object_id") or ref.get("claim_id") or ref.get("source_id")
    rev = ref.get("revision") or ref.get("claim_revision")

    # 1. derivation roots — the §17.04 evidence-plane ancestry.
    if drv is not None and oid:
        try:
            roots = drv.roots(conn, (kind, oid, rev))
            for rkind, rid, _rrev in roots:
                if rkind in _EVIDENCE_ROOT_KINDS and (
                    rid != oid or rkind != kind
                ):
                    return True, "derivation_roots"
        except Exception:
            pass

    # 2. source-level evidence fallback — the item must still resolve to a
    # real ingested source.
    sid = ref.get("source_id") or ref.get("span_id")
    ext = ref.get("external_id")
    if sid and sid in set(env.source_map.values()):
        return True, "source_evidence"
    if ext and ext in env.source_map:
        return True, "source_evidence"
    if sid and _object_exists(conn, "source", sid):
        return True, "source_evidence"
    if oid and _object_exists(conn, kind, oid):
        # The object exists but no source link resolved — it is its own
        # root, which is *not* grounding.
        return False, "none"
    return False, "none"


def run_grounding_suite(
    corpus: Corpus,
    baseline_name: str,
    *,
    k: int = 5,
    task_ids: Optional[Sequence[str]] = None,
    capabilities: Optional[dict[str, Capability]] = None,
) -> SuiteRun:
    caps = capabilities if capabilities is not None else probe_capabilities()
    baseline = get_baseline(baseline_name)
    run = SuiteRun(
        suite="grounding",
        baseline=baseline_name,
        capabilities=caps,
        k=k,
    )
    run.capabilities.update(baseline.capabilities())
    drv = _derivations_module()
    if drv is None:
        run.notes.append(
            "capability unavailable: verbatim.derivations.roots — "
            "source-level evidence ids used as the grounding fallback"
        )
    else:
        run.notes.append(
            "derivation roots probed first; source-level evidence ids "
            "used where no derivation edges exist (v2-ingested claims)"
        )

    tasks: Iterable[CorpusTask] = (
        (t for t in corpus.tasks if t.task_id in set(task_ids))
        if task_ids is not None
        else corpus.tasks
    )
    for task in tasks:
        env = prepare_case(
            task, capabilities=caps,
            ingest=getattr(baseline, "ingest_mode", "v2"),
        )
        try:
            # Arms get the gold-free public view (expected ids stay
            # with the grounding checks below).
            out = baseline.query(env, public_task(task), k=k)
        except Exception as exc:
            out = QueryOutcome(
                arm=baseline_name, task_id=task.task_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        grounded = fabricated = 0
        via_roots = via_source = 0
        if not out.error and not out.unavailable:
            with env.store.read() as conn:
                for ref in out.evidence_refs:
                    ok, via = _resolve_ref(conn, env, ref, drv)
                    if ok:
                        grounded += 1
                        if via == "derivation_roots":
                            via_roots += 1
                        else:
                            via_source += 1
                    else:
                        fabricated += 1
        returned = set(out.returned_ids)
        rec = ScoredRecord(
            task_id=task.task_id,
            kind=task.kind,
            expected_ids=task.expected_evidence_ids,
            returned_ids=out.returned_ids[:k],
            abstained=out.abstained,
            expected_abstain=task.expected_abstain,
            returned_items=out.n_items,
            grounded_items=grounded,
            fabricated_items=fabricated,
            unauthorized_items=len(set(task.unauthorized_source_ids) & returned),
            poisoned_returned=len(set(task.poisoned_source_ids) & returned),
            poisoned_total=len(task.poisoned_source_ids),
            note=(
                out.unavailable_reason
                or (f"error:{out.error}" if out.error else "")
                or f"roots={via_roots},source={via_source}"
            ),
        )
        run.record(rec, out)
        # Every preparation note reaches the run — a silent drain
        # failure must not leave zeroed records unexplained.
        _propagate_env_notes(run, env)
        env.close()

    run.metrics = metrics.summarize(run.records)
    return run


__all__ = ["run_grounding_suite"]
