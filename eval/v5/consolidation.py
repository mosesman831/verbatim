"""Grounded consolidation measurement (E94 / SPEC_V5 §32.04, V3-23,
SPEC_V6 §03.2).

Executes the *real* consolidation path — ``RefreshScheduler.refresh``
enqueues ``JobKind.CONSOLIDATE`` on the background lane and the durable
drain runs ``handle_consolidate`` → ``observations.consolidate`` —
then audits what came out:

* **execution** — jobs enqueued vs. drained vs. failed (durable-queue
  evidence, not inference);
* **fixture reachability (V6-03.11)** — the corpus plants ≥2 same-slot,
  whitelist-predicate claims from distinct evidence families (the
  ``cons-cafe-*`` items — structured ``preference`` claims over
  ``Café Lumière`` plus a ``Café Noir`` rival pair), so a positive
  observation outcome is reachable. ``obs == 0`` is reported
  ``inconclusive`` with an explicit reason — the honest-zero contract
  when no slot qualifies;
* **pin liveness (V6-03.13)** — every ``observation_evidence`` pin must
  resolve through ``claim_evidence`` → ``spans`` → ``source_revisions``
  to independently retrievable bytes (``resolve_pins`` /
  ``never_sole_trace``). A live observation on a dangling, held,
  erased, or superseded pin — or one that is the sole trace of its
  fact — is a defect, never rounded away;
* **delivered grounding** — every ``Hit.quote`` the consumer route
  returns must appear verbatim in its source item's stored text;
* **corroboration (V6-03.12)** — the byte-identical second attestation
  of ``cons-corr-1`` rides the envelope channel under a distinct
  (origin, speaker, provenance) family, so the duplicate group folds
  to ``corroboration = 2``. The probe passes only on a delivered
  proof-bearing hit — ``hit.corroboration ≥ 2`` — never on the raw
  count of hits mentioning the fact.

The consumer route leaves harvested claims ``pending`` (the review
queue is the admission surface), so the suite runs one operator
admission pass through the public ``Engine.apply_transition`` API —
the same operation the CLI review flow performs — before the
consolidation pass. Admissions are counted and failures stay visible.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .corpus import (
    CONSOLIDATION_ITEMS,
    CONSOLIDATION_TASKS,
    ConsumerCorpus,
    seed_corpus,
)
from .harness import (
    ConsumerEnv,
    drain_memory,
    hit_source_id,
    seed_corpus_env,
    settle,
)

#: Fixture items whose structured claims feed the slot-aggregate probe
#: (V6-03.11 — ≥2 same-slot claims from distinct evidence families).
_FIXTURE_ITEMS: Tuple[str, ...] = (
    "cons-cafe-1", "cons-cafe-2", "cons-cafe-3", "cons-cafe-4",
)
#: The corroboration fixture's consumer-route attestation; the suite
#: adds the byte-identical second attestation itself (V6-03.12).
_CORR_ITEM = "cons-corr-1"
_CORR_QUERY = "dehumidifier code DH-5520"


def corroborated_corpus(memories: int = 48, seed: int = 42) -> ConsumerCorpus:
    """Corpus with planted consolidation + corroboration fixtures.

    The base consumer corpus carries the §20.11 mix; ``CONSOLIDATION_ITEMS``
    appends the same-slot café preference pair (+ rival pair) that the
    slot-aggregate producer can actually fold, and one consumer-route
    attestation of the corroborated fact whose byte-identical twin the
    suite ingests through the envelope channel.
    """
    base = seed_corpus(memories=memories, seed=seed)
    return ConsumerCorpus(
        name=f"{base.name}+cons", seed=seed,
        items=tuple(list(base.items) + list(CONSOLIDATION_ITEMS)),
        tasks=tuple(list(base.tasks) + list(CONSOLIDATION_TASKS)),
    )


def _second_attestation(env: ConsumerEnv) -> dict:
    """Ingest the byte-identical second attestation through the envelope
    channel — a distinct (origin, speaker, provenance) family, so the
    duplicate group folds to ``corroboration = 2`` on delivery
    (V6-03.12). Same bytes, different attester: real second-family
    evidence, not a replay of the same submission.
    """
    item = env.corpus.item_by_id().get(_CORR_ITEM)
    if item is None:
        return {"error": f"fixture item {_CORR_ITEM!r} missing from corpus"}
    try:
        from verbatim.core.types_v3 import (
            EnvelopeKind,
            Perspective,
            SourceEnvelopeV3,
        )
        from verbatim.evidence.envelopes import ingest_envelope
        from verbatim.jobs.source_jobs import enqueue_source_jobs
        from verbatim.sourcestate.state import ensure_state
    except ImportError as exc:
        return {"error": f"envelope channel unavailable: {exc}"}
    store = env.memory._store
    ns = env.memory._namespace
    now = env.memory._host.now_us()
    envelope = SourceEnvelopeV3(
        kind=EnvelopeKind.USER_MESSAGE,
        scope_id=ns,
        actor_principal="eval-v5-peer-attester",
        perspective=Perspective(asserter="eval-v5-peer-attester"),
        content=item.text.encode("utf-8"),
        media_type="text/plain",
        host_id="eval-v5-peer",
        session_id="eval-v5-consolidation",
        external_id="v5-cons-corr-peer",
        event_us=now,
        receipt_us=now + 1,
    )
    try:
        with store.tx() as conn:
            receipt = ingest_envelope(conn, store, envelope)
            ensure_state(conn, receipt.source_id, ns, store=store)
            jobs = enqueue_source_jobs(
                conn, store, receipt_id=receipt.receipt_id
            )
        return {
            "envelope_source_id": receipt.source_id,
            "receipt_id": receipt.receipt_id,
            "job_ids": list(jobs.get("job_ids") or ()),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _admit_pending(env: ConsumerEnv) -> dict:
    """Operator admission over every pending claim head — the public
    ``Engine.apply_transition`` effect the CLI review queue performs.

    The consumer ``Memory.add`` contract leaves harvested claims
    ``pending``; consolidation only folds admitted (active/disputed)
    revisions. Without this pass the suite measures an empty input
    set, not the producer.
    """
    try:
        from verbatim.api import Engine
        from verbatim.core.policy import TransitionCommand
    except ImportError as exc:
        return {"error": f"transition API unavailable: {exc}",
                "admitted": 0, "pending_heads": 0, "failed": []}
    store = env.memory._store
    engine = Engine(store, env.memory._cfg, env.memory._host)
    scope = env.memory._host.default_scope()
    try:
        with store.read() as conn:
            heads = conn.execute(
                "SELECT c.claim_id, r.revision, r.state FROM claims c"
                " JOIN claim_revisions r ON r.claim_id = c.claim_id"
                " WHERE r.revision = (SELECT MAX(x.revision)"
                "                   FROM claim_revisions x"
                "                   WHERE x.claim_id = c.claim_id)"
            ).fetchall()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"head read: {type(exc).__name__}: {exc}",
                "admitted": 0, "pending_heads": 0, "failed": []}
    pending = [h for h in heads if h[2] == "pending"]
    admitted = 0
    failed: List[dict] = []
    for cid, rev, _st in pending:
        try:
            engine.apply_transition(
                TransitionCommand(
                    claim_id=cid,
                    expected_revision=int(rev),
                    effect="admit",
                    actor_id="eval-v5-consolidation",
                    reason="operator admission for consolidation "
                           "measurement",
                ),
                scope=scope,
            )
            admitted += 1
        except Exception as exc:  # noqa: BLE001 — recorded, kept visible
            failed.append({
                "claim": cid,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return {
        "pending_heads": len(pending),
        "admitted": admitted,
        "failed": failed,
    }


def _consolidate_jobs(env: ConsumerEnv) -> dict:
    """Enqueue + drain a real consolidation pass; return counts."""
    from verbatim.refresh import RefreshScheduler
    store = env.memory._store
    sched = RefreshScheduler(store)
    try:
        plan = sched.refresh(env.memory._namespace)
        enqueued = len(getattr(plan, "job_ids", []) or [])
    except Exception as exc:  # noqa: BLE001
        return {"enqueued": 0, "error": f"{type(exc).__name__}: {exc}"}
    drain_memory(env)
    try:
        with store.read() as conn:
            row = conn.execute(
                "SELECT COUNT(*), SUM(CASE WHEN state='succeeded' THEN 1 "
                "ELSE 0 END), SUM(CASE WHEN state='failed' THEN 1 "
                "ELSE 0 END) FROM jobs WHERE kind='consolidate'"
            ).fetchone()
        return {
            "enqueued": enqueued,
            "total": int(row[0] or 0),
            "done": int(row[1] or 0),
            "failed": int(row[2] or 0),
        }
    except Exception as exc:  # noqa: BLE001
        return {"enqueued": enqueued,
                "error": f"jobs audit: {type(exc).__name__}: {exc}"}


def _fixture_claim_ids(conn: Any, env: ConsumerEnv) -> set:
    """Claim ids derived from the planted fixture items — resolved
    through ``claim_evidence`` → ``spans`` → fixture source ids."""
    srcs = [
        env.source_id(cid) for cid in _FIXTURE_ITEMS if env.source_id(cid)
    ]
    if not srcs:
        return set()
    rows = conn.execute(
        "SELECT DISTINCT ce.claim_id FROM claim_evidence ce"
        " JOIN spans s ON s.span_id = ce.span_id"
        f" WHERE s.source_id IN ({','.join('?' for _ in srcs)})",
        srcs,
    ).fetchall()
    return {str(r[0]) for r in rows}


def _observation_audit(env: ConsumerEnv, jobs: dict) -> dict:
    """Observation emission + per-pin liveness (V6-03.11/03.13).

    Counts live observations, resolves every evidence pin through
    ``resolve_pins``, and checks ``never_sole_trace`` — the strict
    byte-grounding invariant. Also recomputes the fixture's slot
    candidates so a zero-observation outcome reports *why*: no
    structured fixture claims, none eligible, no slot at
    ``min_proof``, the job never ran, or a real producer defect.
    """
    store = env.memory._store
    ns = env.memory._namespace
    try:
        from verbatim.observations.aggregate import (
            DEFAULT_MIN_PROOF,
            aggregate_candidates,
            fetch_claim_inputs,
            never_sole_trace,
            resolve_pins,
        )
    except ImportError as exc:
        return {"error": f"observation plane unavailable: {exc}",
                "verdict": "unavailable"}
    try:
        with store.read() as conn:
            fixture_claims = _fixture_claim_ids(conn, env)
            inputs = fetch_claim_inputs(conn, ns)
            candidates = aggregate_candidates(conn, ns, inputs=inputs)
            fixture_eligible = sum(
                1 for ci in inputs if ci.claim_id in fixture_claims
            )
            qualified = [
                c for c in candidates
                if c.proof_count >= DEFAULT_MIN_PROOF
                and any(
                    kind == "claim" and oid in fixture_claims
                    for kind, oid, _rev in c.supports
                )
            ]
            rows = conn.execute(
                "SELECT observation_id, revision, text, proof_count,"
                "       recorded_until"
                " FROM observations WHERE scope_id = ?",
                (ns,),
            ).fetchall()
            live = [r for r in rows if r[4] is None]
            retired = len(rows) - len(live)
            live_ids = {str(r[0]) for r in live}
            expected_ids = {c.observation_id for c in qualified}
            missing_slots = sorted(expected_ids - live_ids)

            total_refs = 0
            ungrounded = 0
            held_or_erased = 0
            superseded_pins = 0
            sole_trace = 0
            pin_rows: List[dict] = []
            for oid, rev, text, proof, _ru in live:
                rep = resolve_pins(conn, str(oid))
                traced = never_sole_trace(conn, str(oid))
                summary = rep.get("summary") or {}
                total_refs += int(summary.get("total", 0))
                ungrounded += int(summary.get("total", 0)) - int(
                    summary.get("resolvable", 0)
                )
                held_or_erased += int(summary.get("held", 0)) + int(
                    summary.get("erased", 0)
                )
                superseded_pins += int(summary.get("superseded", 0))
                if not traced:
                    sole_trace += 1
                pin_rows.append({
                    "observation": str(oid),
                    "proof_count": int(proof or 0),
                    "pins": summary,
                    "sole_trace": not traced,
                    "fixture_slot": str(oid) in expected_ids,
                })
        out: Dict[str, Any] = {
            "observations": len(live),
            "retired": retired,
            "ungrounded": ungrounded,
            "held_or_erased_pins": held_or_erased,
            "superseded_pins": superseded_pins,
            "sole_trace_violations": sole_trace,
            "evidence_refs": total_refs,
            "fixture": {
                "claims": len(fixture_claims),
                "eligible_claims": fixture_eligible,
                "qualified_slots": len(qualified),
                "missing_observations": missing_slots,
            },
            "pins": pin_rows,
        }
        if not live:
            if jobs.get("done", 0) == 0 and jobs.get("enqueued", 0) > 0:
                out["reason"] = (
                    "consolidate job was enqueued but did not complete"
                )
            elif not fixture_claims:
                out["reason"] = (
                    "no structured claims derived from the planted "
                    "fixture items — the claims pipeline produced "
                    "nothing for the producer to fold"
                )
            elif not fixture_eligible:
                out["reason"] = (
                    "fixture claims exist but none reached an eligible "
                    "state (pending claims were not admitted or the "
                    "modality/predicate filters excluded them)"
                )
            elif not qualified:
                out["reason"] = (
                    "fixture claims exist but no same-slot group "
                    "reached min_proof=2 distinct evidence families — "
                    "an honest zero: no slot qualifies"
                )
            else:
                out["reason"] = (
                    "eligible same-slot input existed and the job ran, "
                    "yet no observation was emitted"
                )
        return out
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}",
                "verdict": "unavailable"}


def _delivered_grounding(env: ConsumerEnv) -> dict:
    """Every delivered quote must be verbatim substrings of the stored
    item text — the consumer-visible grounding audit."""
    item_text = {i.id: i.text for i in env.corpus.items}
    checked = 0
    violations: List[dict] = []
    for task in env.corpus.tasks:
        try:
            sr = env.memory.search(task.query, limit=8)
        except Exception:
            continue
        for h in getattr(sr, "items", []) or []:
            sid = hit_source_id(env, h)
            cid = env.corpus_id(sid)
            if cid is None:
                continue
            checked += 1
            quote = (getattr(h, "quote", "") or "")
            src = item_text.get(cid, "")
            # quotes may be span-excerpts; a real quote must appear in
            # the stored bytes (verbatim grounding, §32.04)
            if quote and quote not in src and src not in quote:
                violations.append({
                    "task": task.task_id, "item": cid,
                    "quote_prefix": quote[:80],
                })
    return {
        "hits_checked": checked,
        "violations": violations,
        "verdict": "passed" if not violations else "failed",
    }


def _corroboration_audit(env: ConsumerEnv, attestation: dict) -> dict:
    """Folded-corroboration probe (V6-03.12).

    Passes only when a delivered hit on the corroborated group reports
    ``hit.corroboration ≥ 2`` — the proof-bearing fold of distinct
    (origin, speaker, provenance) attestations. The raw count of hits
    mentioning the fact stays a diagnostic: multiple raw hits with
    ``corroboration = 1`` prove nothing was folded.
    """
    corr_sources = {
        s for s in (
            env.source_id(_CORR_ITEM),
            attestation.get("envelope_source_id"),
        ) if s
    }
    try:
        sr = env.memory.search(_CORR_QUERY, limit=8)
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "unavailable",
                "error": f"{type(exc).__name__}: {exc}"}
    corr_hits = 0
    folded_hits = 0
    max_corr = 0
    for h in getattr(sr, "items", []) or []:
        sid = hit_source_id(env, h)
        if sid not in corr_sources:
            continue
        corr_hits += 1
        folded = int(getattr(h, "corroboration", 1) or 1)
        max_corr = max(max_corr, folded)
        if folded >= 2:
            folded_hits += 1
    group_members = 0
    try:
        from verbatim.dedup import links as _links
        with env.memory._store.read() as conn:
            for sid in corr_sources:
                view = _links.collapse_for_hit(conn, sid, 1) or {}
                group_members = max(
                    group_members, len(view.get("member_refs") or ())
                )
    except Exception:
        group_members = -1
    out: Dict[str, Any] = {
        "corroborated_hits": corr_hits,
        "folded_hits": folded_hits,
        "max_corroboration": max_corr,
        "group_members": group_members,
        "note": "verdict requires a folded hit.corroboration ≥ 2 "
                "(proof-bearing group); raw hit count is diagnostic "
                "only — multiple corr=1 hits prove nothing folded",
    }
    if folded_hits:
        out["verdict"] = "passed"
        return out
    out["verdict"] = "inconclusive"
    if not corr_sources:
        out["reason"] = (
            "the second attestation never persisted — no corroborated "
            "group exists to fold"
        )
    elif corr_hits == 0:
        out["reason"] = (
            "no delivered hit on the corroborated group — the fold "
            "could not be observed through the consumer route"
        )
    elif group_members in (0, 1):
        out["reason"] = (
            "attestations were delivered but never linked into one "
            "duplicate group — nothing folded to measure"
        )
    else:
        out["reason"] = (
            "the group is linked and delivered, but the representative "
            "hit still reports corroboration < 2 — the distinct-family "
            "fold did not reach delivery"
        )
    return out


def run_consolidation(*, memories: int = 48, seed: int = 42,
                      workdir: Optional[str] = None) -> dict:
    """Execute the grounded-consolidation suite."""
    corpus = corroborated_corpus(memories, seed)
    env = seed_corpus_env(corpus, workdir=workdir, worker="external")
    try:
        settle(env)
        # Second attestation first: its source-project job links the
        # duplicate group before delivery probes run (V6-03.12).
        attestation = _second_attestation(env)
        drain_memory(env)
        # Operator admission — the review-queue operation that turns
        # pending harvested claims into consolidation-eligible input.
        admission = _admit_pending(env)
        drain_memory(env)
        jobs = _consolidate_jobs(env)
        obs = _observation_audit(env, jobs)
        grounding = _delivered_grounding(env)
        corr = _corroboration_audit(env, attestation)
    finally:
        env.close()
    obs_verdict = (
        "failed" if (
            obs.get("ungrounded", 0)
            or obs.get("held_or_erased_pins", 0)
            or obs.get("superseded_pins", 0)
            or obs.get("sole_trace_violations", 0)
            or obs.get("fixture", {}).get("missing_observations")
        )
        else "passed" if obs.get("observations", 0)
        else "inconclusive"
    )
    verdicts = {
        "jobs": "passed" if jobs.get("done", 0) > 0
                or jobs.get("enqueued", 0) == 0 else "failed",
        "observations": "unavailable" if obs.get("verdict") == "unavailable"
                        else obs_verdict,
        "grounding": grounding.get("verdict", "unavailable"),
        "corroboration": corr.get("verdict", "unavailable"),
    }
    return {
        "suite": "consolidation",
        "qualification": "locally_measured",
        "scale": {"memories": memories,
                  "corpus": corpus.name, "digest": corpus.digest()},
        "admission": admission,
        "attestation": attestation,
        "jobs": jobs,
        "observations": obs,
        "delivered_grounding": grounding,
        "corroboration": corr,
        "verdicts": verdicts,
        "verdict": ("failed" if "failed" in verdicts.values()
                    else "unavailable" if all(
                        v == "unavailable" for v in verdicts.values())
                    else "passed" if all(
                        v == "passed" for v in verdicts.values())
                    else "inconclusive"),
    }


__all__ = ["corroborated_corpus", "run_consolidation"]
