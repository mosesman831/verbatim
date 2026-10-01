"""Security & privacy suite (SPEC_V3 §14, §31, §34; gate G9).

Four measured lanes, all against real stores:

* **screening** — every setup source (poisoned *and* benign) goes through
  ``security.screen_content``.  ``poisoned_flagged``/``poisoned_total``
  measures the attack-detection rate; ``benign_flagged``/``benign_total``
  measures over-blocking — a screener that flags runbooks fails here.
* **retrieval exposure** — the chosen baseline answers the task query on
  the unprepared store (ingest path only); poisoned sources that reach
  results count as exposures, unauthorized-scope items as disclosures.
* **post-quarantine** — when ``verbatim.security`` is available, flagged
  sources get a ``security_labels`` row + ``open_quarantine`` holds on the
  source and its claims; the query runs again and residual exposure is
  measured.  Engines that ignore the quarantine table (v2, naive FTS)
  show their gap honestly.
* **governance** — consent issuance/requirement/revocation and a revoked
  READ grant → v3 recall must deny.  Recorded as explicit checks, never
  assumed.

Missing security/governance modules produce explicit capability gaps —
the lane reports ``capability unavailable`` and keeps going.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from . import metrics
from .baselines import (
    EVAL_CALLER,
    Capability,
    CaseEnv,
    QueryOutcome,
    SuiteRun,
    get_baseline,
    prepare_case,
    probe_capabilities,
)
from .corpus import (
    Corpus,
    CorpusSource,
    CorpusTask,
    SCOPE_OTHER,
    public_task,
)
from .metrics import ScoredRecord
from .suite_retrieval import _propagate_env_notes, score_outcome

_FLAGGED_RISKS = frozenset({"suspicious", "blocked"})


def _security_module():
    try:
        import verbatim.security as sec
        return sec if hasattr(sec, "screen_content") else None
    except Exception:
        return None


def _source_claims(conn: sqlite3.Connection, source_id: str) -> list[tuple[str, int]]:
    """(claim_id, head_revision) pairs derived from a source via spans —
    quarantine must hold the head, not an arbitrary revision."""
    rows = conn.execute(
        "SELECT ce.claim_id, h.mr FROM claim_evidence ce"
        " JOIN spans s ON s.span_id = ce.span_id"
        " JOIN (SELECT claim_id, MAX(revision) AS mr FROM claim_revisions"
        "       GROUP BY claim_id) h ON h.claim_id = ce.claim_id"
        " WHERE s.source_id = ?"
        " GROUP BY ce.claim_id",
        (source_id,),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _quarantine_flagged(env: CaseEnv, sec: Any, flagged: list[tuple[CorpusSource, Any]]) -> int:
    """Attach labels + open quarantine holds for screened-flagged sources.

    Returns the number of objects placed under quarantine."""
    holds = 0
    with env.store.tx() as conn:
        for src, verdict in flagged:
            sid = env.source_map.get(src.id)
            if not sid:
                continue
            scope_id = env.owner_scope_id
            findings = [dict(f) for f in verdict.findings]
            try:
                sec.attach_label(
                    conn,
                    scope_id,
                    source_trust="unknown",
                    content_form=verdict.content_form.value,
                    attack_risk=verdict.attack_risk.value,
                    review_state="quarantined",
                    findings=findings,
                    method=verdict.method,
                    rules_revision=verdict.rules_revision,
                )
                codes = [f"attack_risk:{verdict.attack_risk.value}"] + [
                    f"rule:{f.get('rule_id', 'unknown')}" for f in findings
                ]
                if sec.open_quarantine(
                    conn, ("source", sid, 1), codes, findings,
                    scope_id=scope_id,
                ):
                    holds += 1
                for cid, rev in _source_claims(conn, sid):
                    if sec.open_quarantine(
                        conn, ("claim", cid, rev), codes, findings,
                        scope_id=scope_id,
                    ):
                        holds += 1
            except Exception as exc:
                env.notes.append(
                    f"quarantine failed for {src.id}: {type(exc).__name__}: {exc}"
                )
    return holds


def _governance_lane(env: CaseEnv, run: SuiteRun) -> Dict[str, Any]:
    """Exercise consent + grant revocation on this case's store.

    The lane is **arm-independent**: it always measures v3 governance +
    ``recall_v3`` on the prepared store, so the identical result renders
    under every arm's security table.  ``arm_independence`` travels in
    the result so neither a report reader nor a JSON consumer attributes
    the outcome to the run's baseline (naive_fts/vector_rag included).
    """
    out: Dict[str, Any] = {
        "ran": False,
        "arm_independence": (
            "arm-independent: measures v3 governance + recall_v3 on the "
            "prepared store — identical under every arm; not a "
            "measurement of the run's baseline"
        ),
    }
    try:
        import verbatim.governance as gov
        from verbatim.core.types_v3 import EnvelopeKind, RecallRequestV3, Verb
    except Exception as exc:
        out["unavailable_reason"] = (
            f"capability unavailable: governance — {type(exc).__name__}: {exc}"
        )
        return out
    out["ran"] = True
    checks: Dict[str, Any] = {}
    sid = env.owner_scope_id
    try:
        with env.store.tx() as conn:
            before = gov.capture_authorized(
                conn, EVAL_CALLER, EnvelopeKind.USER_MESSAGE, sid
            )
            auth_id = gov.issue_capture_authorization(
                conn,
                principal_id=EVAL_CALLER,
                issuer_id="eval-v3-harness",
                allowed_kinds=[EnvelopeKind.USER_MESSAGE.value],
                retention_policy="eval",
                policy_revision="r1",
                scope_ids=[sid],
            )
            during = gov.capture_authorized(
                conn, EVAL_CALLER, EnvelopeKind.USER_MESSAGE, sid
            )
            gov.revoke_capture_authorization(conn, auth_id)
            after = gov.capture_authorized(
                conn, EVAL_CALLER, EnvelopeKind.USER_MESSAGE, sid
            )
        checks["consent_cycle"] = {
            "before": before, "during": during, "after_revoke": after,
            "ok": (not before) and during and (not after),
        }
    except Exception as exc:
        checks["consent_cycle"] = {"error": f"{type(exc).__name__}: {exc}"}

    # Grant revocation → recall_v3 must deny.
    try:
        from verbatim.retrieval.v3 import recall_v3

        req = RecallRequestV3(
            query="deploy", scope_id=sid, caller_id=EVAL_CALLER,
            purpose="eval",
        )
        with env.store.read() as conn:
            grants = gov.grants_for(conn, sid, EVAL_CALLER)
        revoked_ok = True
        try:
            with env.store.tx() as conn:
                for g in grants:
                    gov.revoke_grant(conn, g["grant_id"])
        except Exception as exc:
            checks["revoke_grant"] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            rr = recall_v3(env.store, req)
            revoked_ok = rr.abstained or not any(p.items for p in rr.packs)
            checks["post_revocation_recall"] = {
                "denied_or_empty": revoked_ok, "raised": False,
            }
        except Exception as exc:
            checks["post_revocation_recall"] = {
                "denied_or_empty": True, "raised": True,
                "error_type": type(exc).__name__,
            }
    except Exception as exc:
        checks["post_revocation_recall"] = {
            "unavailable": f"{type(exc).__name__}: {exc}"
        }
    out["checks"] = checks
    return out


def run_security_suite(
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
        suite="security",
        baseline=baseline_name,
        capabilities=caps,
        k=k,
    )
    run.capabilities.update(baseline.capabilities())
    sec = _security_module()
    if sec is None:
        run.notes.append(
            "capability unavailable: verbatim.security.screen_content — "
            "screening and quarantine lanes cannot run"
        )

    tasks: Iterable[CorpusTask] = (
        (t for t in corpus.tasks if t.task_id in set(task_ids))
        if task_ids is not None
        else corpus.tasks
    )
    governance_done = False
    post_cases: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        # Each arm is prepared through its own real write channel: the
        # verbatim_v3 arm ingests via ingest_envelope (screened at write
        # time, §34.01) while other arms share the legacy path — the
        # pre-quarantine columns then measure what each channel admits.
        env = prepare_case(
            task, capabilities=caps, ingest=baseline.ingest_mode
        )
        try:
            # ---- lane 1: screening ------------------------------------
            poisoned_flagged = benign_flagged = 0
            poisoned_total = len(task.poisoned_source_ids)
            benign_total = 0
            flagged: list[tuple[CorpusSource, Any]] = []
            if sec is not None:
                want = set(task.poisoned_source_ids)
                for src in task.setup_sources:
                    if src.scope == SCOPE_OTHER:
                        continue  # foreign-scope data isn't ours to screen
                    verdict = sec.screen_content(src.text)
                    if src.id in want:
                        if verdict.attack_risk.value in _FLAGGED_RISKS:
                            poisoned_flagged += 1
                            flagged.append((src, verdict))
                    else:
                        benign_total += 1
                        if verdict.attack_risk.value in _FLAGGED_RISKS:
                            benign_flagged += 1

            # ---- lane 2: retrieval exposure ---------------------------
            try:
                # Arms get the gold-free public view (poisoned/
                # unauthorized gold stays with scoring below).
                out = baseline.query(env, public_task(task), k=k)
            except Exception as exc:
                out = QueryOutcome(
                    arm=baseline_name, task_id=task.task_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
            rec = score_outcome(env, task, out, k=k)
            rec = ScoredRecord(
                **{**rec.__dict__,
                   "poisoned_flagged": poisoned_flagged,
                   "benign_flagged": benign_flagged,
                   "benign_total": benign_total}
            )
            run.record(rec, out)

            # ---- lane 3: post-quarantine exposure ---------------------
            # Kept OUT of ScoredRecords — folding post-quarantine counts
            # into the shared denominators would blend pre/post rates.
            if sec is not None and (task.poisoned_source_ids or flagged):
                holds = _quarantine_flagged(env, sec, flagged)
                try:
                    out_q = baseline.query(env, public_task(task), k=k)
                except Exception as exc:
                    out_q = QueryOutcome(
                        arm=baseline_name, task_id=task.task_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                rec_q = score_outcome(env, task, out_q, k=k)
                post_cases[task.task_id] = {
                    "poisoned_total": rec_q.poisoned_total,
                    "poisoned_returned": rec_q.poisoned_returned,
                    "returned_ids": list(rec_q.returned_ids),
                    "holds": holds,
                    "error": out_q.error,
                }

            # ---- lane 4: governance (once) ----------------------------
            if not governance_done:
                gov_res = _governance_lane(env, run)
                run.metrics["governance"] = gov_res
                governance_done = True
                # The identical result renders under every arm's table
                # — label it so a reader never attributes it to this
                # run's baseline.
                run.notes.append(
                    "governance checks are arm-independent: they measure "
                    "v3 governance + recall_v3 on the prepared store — "
                    "the same result appears under every arm's security "
                    "table and is not a measurement of the run's baseline"
                )
                if gov_res.get("unavailable_reason"):
                    run.notes.append(gov_res["unavailable_reason"])
        finally:
            # Preparation + quarantine notes must reach the report even
            # when a lane raised — a silent drain failure must not leave
            # zeroed records unexplained.
            _propagate_env_notes(run, env)
            env.close()

    run.metrics.update(metrics.summarize(run.records))
    # post-quarantine lane — its own denominator (§54.12 stage reporting)
    post_total = sum(c["poisoned_total"] for c in post_cases.values())
    post_exposed = sum(c["poisoned_returned"] for c in post_cases.values())
    run.metrics["post_quarantine_exposure"] = {
        "value": (post_exposed / post_total) if post_total else None,
        "cases": post_cases,
    }
    return run


__all__ = ["run_security_suite"]
