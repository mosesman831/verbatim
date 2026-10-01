"""G8 learned-controller training + paired evaluation (§54 G8, §43.04/05).

The G8 gate compares a learned controller against the deterministic
table on *executed* outcomes only — paired sandbox arms that actually
deliver each policy's context (V3-43.05). Nothing here scores a
counterfactual: every Q value in the artifact comes from an arm that
physically ran ``recall_v3`` inside its own seeded sandbox.

Two phases, both offline and model-free:

``train`` — for every corpus task, build the task's store through the
real v3 ingest path, then run ``paired_execution`` with one arm per
bounded action (``forced:<action>`` — documented action support,
V3-43.04). Each arm's delivered set is scored against the task's gold:
``success`` = expected-evidence coverage (or abstain correctness for
abstention tasks) and ``tokens`` = delivered text bytes/4 — the declared
downstream-token proxy. Rewards fold into a per-``state_key`` sample
mean → ``PolicyArtifact`` with full provenance.

``evaluate`` — deterministic-vs-learned paired runs over the corpus:
per-task success diff and token reduction, aggregated into paired 95%
intervals. The verdict reports the measured numbers against §54's G8
criterion — success lower-95% interval > −0.01 with ≥ 10% token
reduction, OR success +0.03 with positive interval at equal tokens —
without asserting a pass the data did not earn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from typing import Any, Optional, Sequence

from verbatim.core.types_v3 import RecallRequestV3
from verbatim.replay import ReplayLab
from verbatim.retrieval.v3 import controller as _ctrl
from verbatim.retrieval.v3 import learned as _learned
from verbatim.storage.store import Store

from . import baselines as _bl
from .corpus import CorpusTask, load_seed_corpus

# reward = success − λ · (delivered_bytes / tier_budget_bytes): the
# success-vs-token tradeoff the learned policy optimizes, declared in the
# artifact's provenance (V3-26.04's cost-bound intent).
LAMBDA = 0.5

_G8 = (
    "success lower paired 95% interval > -0.01 with >= 10% total "
    "task-token reduction, or success +0.03 with positive interval at "
    "equal tokens (|token_reduction| < 5%)"
)


# ---------------------------------------------------------------------------
# scoring — actual delivered sets vs declared gold
# ---------------------------------------------------------------------------


def _claim_source_map(env: _bl.CaseEnv, store: Store) -> dict:
    """claim_id → fixture gold id — the production store's evidence
    chains resolve which declared source each claim carries."""
    out: dict[str, str] = {}
    by_real = {real: fx for fx, real in env.source_map.items()}
    with store.read() as conn:
        rows = conn.execute(
            "SELECT ce.claim_id, s.source_id FROM claim_evidence ce"
            " JOIN spans s ON s.span_id = ce.span_id"
            " ORDER BY ce.claim_id, ce.span_id"  # deterministic attribution
        ).fetchall()
    for claim_id, real_source in rows:
        fx = by_real.get(real_source)
        if fx is not None and claim_id not in out:
            out[claim_id] = fx
    return out


def _score_delivery(
    delivery: dict, task: CorpusTask, claim_map: dict
) -> tuple[float, int]:
    """(success, delivered_bytes) for one arm's delivery.

    Abstention tasks score the abstain flag itself; evidence tasks score
    gold coverage — a poisoned/unexpected item never counts toward it.
    """
    if task.expected_abstain:
        return (1.0 if delivery["abstained"] else 0.0), sum(
            delivery.get("item_bytes", {}).values()
        )
    expected = set(task.expected_evidence_ids)
    if not expected:
        # a non-abstain task with no declared gold is a corpus defect —
        # scoring it 1.0 would fabricate success (the corpus validator
        # already forbids this; fail loudly if one slips through)
        raise ValueError(
            f"{task.task_id}: non-abstain task with no expected evidence"
        )
    matched: set[str] = set()
    for item_id in delivery["items"]:
        claim_id = item_id.split(":", 1)[1].rsplit("@", 1)[0]
        fx = claim_map.get(claim_id)
        if fx in expected:
            matched.add(fx)
    return len(matched) / len(expected), sum(
        delivery.get("item_bytes", {}).values()
    )


def _request_for(task: CorpusTask, scope_id: str) -> RecallRequestV3:
    return RecallRequestV3(
        query=task.query,
        scope_id=scope_id,
        caller_id="replay-g8",
        purpose="evaluate",
    )


def _arm_state_key(
    sandbox_ref: str, scope_id: str, request: RecallRequestV3
) -> Optional[str]:
    """The state_key the arm's routing decision recorded — read from the
    executed sandbox. Early-return recalls (no_signal/processing_pending)
    exit before the decision row is written; for those the key is
    recomputed through the same classify→discretize path the arm took,
    so the earned reward isn't silently dropped."""
    try:
        sb = Store.open(sandbox_ref)
        try:
            with sb.read() as conn:
                row = conn.execute(
                    "SELECT state_key FROM routing_decisions"
                    " WHERE scope_id = ? ORDER BY created_us DESC LIMIT 1",
                    (scope_id,),
                ).fetchone()
                if row:
                    return row[0]
                # no decision row → the recall early-returned; recompute
                # the identical discretized key on the arm's own snapshot
                from verbatim.core.time import now_us
                from verbatim.retrieval.query import analyze
                from verbatim.retrieval.v3.recall import classify
                plan = analyze(request.query, request, now_us())
                qclass = classify(request, plan)
                state = _ctrl.discretize(
                    conn, request, qclass, [scope_id]
                )
                return _ctrl._state_key(state)
        finally:
            sb.close()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# train — executed forced-action arms → tabular Q
# ---------------------------------------------------------------------------


def train(
    tasks: Sequence[CorpusTask],
    work_dir: str,
    *,
    lam: float = LAMBDA,
    ingest: str = "v3",
    actions: Sequence[str] = _learned.ACTIONS,
) -> _learned.PolicyArtifact:
    """Execute every bounded action as a paired arm per task; fold the
    measured rewards into per-state_key sample means (V3-43.04/05)."""
    sums: dict[str, dict[str, float]] = {}
    pulls: dict[str, dict[str, int]] = {}
    runs: list[str] = []
    for task in tasks:
        env = _bl.prepare_case(task, ingest=ingest)
        try:
            store = env.store
            scope_id = env.owner_scope_id
            claim_map = _claim_source_map(env, store)
            lab = ReplayLab(store)
            arms = [
                {
                    "name": a,
                    "variant": {
                        "controller": {"kind": "forced", "action": a}
                    },
                }
                for a in actions
            ]
            sandbox_dir = os.path.join(
                work_dir, f"g8-train-{task.task_id}"
            )
            rep = lab.paired_execution(
                scope_id, [_request_for(task, scope_id)],
                arms=arms, sandbox_dir=sandbox_dir,
            )
            runs.append(rep["run_id"])
            # reward denominator: the request tier's byte ceiling —
            # λ·(consumed/ceiling) is the declared success-vs-cost
            # tradeoff, stable whether or not the baseline delivered
            tier_budget = _ctrl.TIER_BUDGETS[
                _request_for(task, scope_id).budget_tier
            ]["max_bytes"]
            req = _request_for(task, scope_id)
            for name, arm in rep["per_arm"].items():
                delivery = arm["deliveries"][0]
                success, nbytes = _score_delivery(
                    delivery, task, claim_map
                )
                reward = success - lam * (nbytes / tier_budget)
                skey = _arm_state_key(arm["sandbox_ref"], scope_id, req)
                if skey is None:
                    continue
                sums.setdefault(skey, {})
                pulls.setdefault(skey, {})
                pulls[skey][name] = pulls[skey].get(name, 0) + 1
                prev_n = pulls[skey][name] - 1
                prev = sums[skey].get(name, 0.0) * prev_n
                sums[skey][name] = (prev + reward) / pulls[skey][name]
        finally:
            env.close()
    digest = hashlib.sha256(
        "|".join(t.task_id for t in tasks).encode("utf-8")
    ).hexdigest()[:16]
    return _learned.PolicyArtifact(
        revision=f"g8w1_{digest}",
        actions=tuple(actions),
        q=sums,
        pulls=pulls,
        trained_on={
            "method": "executed forced-action paired arms (V3-43.04/05)",
            "tasks": len(tasks),
            "lambda": lam,
            "run_ids": runs,
            "ingest": ingest,
        },
        gate={"status": "pending_evaluation"},
    )


# ---------------------------------------------------------------------------
# evaluate — deterministic vs learned paired runs → G8 verdict
# ---------------------------------------------------------------------------


def _ci95(diffs: list[float]) -> tuple[float, float]:
    n = len(diffs)
    if n < 2:
        return (diffs[0], diffs[0]) if diffs else (0.0, 0.0)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    half = 1.96 * math.sqrt(var / n)
    return mean - half, mean + half


def evaluate(
    tasks: Sequence[CorpusTask],
    artifact: _learned.PolicyArtifact,
    work_dir: str,
    *,
    ingest: str = "v3",
) -> dict:
    """Paired deterministic-vs-learned executions; paired 95% intervals
    on success diff and token reduction — the §54 G8 measurement."""
    diffs: list[float] = []
    token_reductions: list[float] = []
    fallback_checks: list[bool] = []
    per_task: list[dict] = []
    det_bytes = learned_bytes = 0
    det_success = learned_success = 0.0
    for task in tasks:
        env = _bl.prepare_case(task, ingest=ingest)
        try:
            store = env.store
            scope_id = env.owner_scope_id
            claim_map = _claim_source_map(env, store)
            lab = ReplayLab(store)
            rep = lab.paired_execution(
                scope_id, [_request_for(task, scope_id)],
                arms=[
                    {"name": "deterministic", "variant": None},
                    {
                        "name": "learned",
                        "variant": {
                            "controller": {
                                "kind": "replay_learned",
                                "artifact": artifact.to_dict(),
                            }
                        },
                    },
                    {
                        # §54 G8 "deterministic fallback verified": a
                        # missing artifact must produce deterministic-
                        # identical deliveries through the same channel
                        "name": "learned_fallback",
                        "variant": {
                            "controller": {
                                "kind": "replay_learned",
                                "artifact": None,
                            }
                        },
                    },
                ],
                sandbox_dir=os.path.join(
                    work_dir, f"g8-eval-{task.task_id}"
                ),
            )
            d_det = rep["per_arm"]["deterministic"]["deliveries"][0]
            d_lrn = rep["per_arm"]["learned"]["deliveries"][0]
            d_fb = rep["per_arm"]["learned_fallback"]["deliveries"][0]
            s_det, b_det = _score_delivery(d_det, task, claim_map)
            s_lrn, b_lrn = _score_delivery(d_lrn, task, claim_map)
            diffs.append(s_lrn - s_det)
            # det_bytes=0 sentinel: learned spending where deterministic
            # spent nothing is a regression (−1.0), never "equal tokens"
            red = (
                (b_det - b_lrn) / b_det if b_det
                else (0.0 if b_lrn == 0 else -1.0)
            )
            token_reductions.append(red)
            fallback_ok = (
                d_fb["items"] == d_det["items"]
                and d_fb["abstained"] == d_det["abstained"]
                and any(
                    "learned_artifact_missing" in w
                    for w in d_fb["warnings"]
                )
            )
            fallback_checks.append(fallback_ok)
            det_bytes += b_det
            learned_bytes += b_lrn
            det_success += s_det
            learned_success += s_lrn
            per_task.append({
                "task_id": task.task_id,
                "det": {"success": s_det, "bytes": b_det},
                "learned": {"success": s_lrn, "bytes": b_lrn},
                "learned_warnings": d_lrn["warnings"],
            })
        finally:
            env.close()
    lo, hi = _ci95(diffs)
    tlo, thi = _ci95(token_reductions)
    mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
    token_red = (
        (det_bytes - learned_bytes) / det_bytes if det_bytes
        else (0.0 if learned_bytes == 0 else -1.0)
    )
    n = len(tasks)
    fallback_verified = bool(fallback_checks) and all(fallback_checks)
    # V3-53.14: task-family-clustered intervals alongside the pooled one
    by_kind: dict[str, list[float]] = {}
    for task, d in zip(tasks, diffs):
        by_kind.setdefault(getattr(task, "kind", None) or "unknown", []).append(d)
    kind_intervals = {
        k: {"n": len(v), "success_diff_ci95": [round(x, 4) for x in _ci95(v)]}
        for k, v in sorted(by_kind.items())
    }
    verdict = {
        "tasks": n,
        "success_diff": round(mean_diff, 4),
        "success_diff_ci95": [round(lo, 4), round(hi, 4)],
        "det_success": round(det_success / n, 4) if n else 0.0,
        "learned_success": round(learned_success / n, 4) if n else 0.0,
        "token_reduction": round(token_red, 4),
        "token_reduction_ci95": [round(tlo, 4), round(thi, 4)],
        # delivered byte sums — the declared token proxy is bytes/4;
        # the reduction ratio is identical either way
        "det_bytes": det_bytes,
        "learned_bytes": learned_bytes,
        "fallback_verified": fallback_verified,
        "per_kind_ci95": kind_intervals,
        "disclosures": {
            "train_eval_same_corpus": True,
            "seeds": "1 (deterministic harness — no seed variance to repeat)",
        },
        "criterion": _G8,
        "g8_pass": bool(
            n >= 2 and fallback_verified and (
                (lo > -0.01 and token_red >= 0.10)
                or (mean_diff >= 0.03 and lo > 0.0
                    and abs(token_red) < 0.05)
            )
        ),
        "per_task": per_task,
    }
    return verdict


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=int, default=0,
                   help="limit corpus tasks (0 = all)")
    p.add_argument("--train-out", default="eval/v3/g8_policy.json")
    p.add_argument("--report-out", default="eval/v3/g8_report.json")
    p.add_argument("--work-dir", default=None)
    p.add_argument("--skip-train", action="store_true",
                   help="evaluate an existing --train-out artifact")
    args = p.parse_args(argv)

    corpus = load_seed_corpus()
    tasks = list(corpus.tasks)
    if args.tasks:
        tasks = tasks[: args.tasks]
    work = args.work_dir or tempfile.mkdtemp(prefix="verbatim-g8-")

    if args.skip_train:
        artifact = _learned.load_artifact(args.train_out)
        if artifact is None:
            raise SystemExit(
                f"no valid artifact at {args.train_out}"
            )
    else:
        artifact = train(tasks, work)
        with open(args.train_out, "w", encoding="utf-8") as fh:
            fh.write(artifact.to_json())
        print(f"[g8] artifact → {args.train_out} "
              f"({len(artifact.q)} state_keys trained)")

    verdict = evaluate(tasks, artifact, work)
    artifact_dict = artifact.to_dict()
    artifact_dict["gate"] = {
        "status": "measured",
        "g8_pass": verdict["g8_pass"],
        "verdict": {
            k: v for k, v in verdict.items() if k != "per_task"
        },
    }
    with open(args.train_out, "w", encoding="utf-8") as fh:
        fh.write(_learned.PolicyArtifact.from_dict(artifact_dict).to_json())
    with open(args.report_out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "artifact_revision": artifact.revision,
                "trained_on": artifact.trained_on,
                "verdict": verdict,
            },
            fh, indent=2, sort_keys=True,
        )
    print(f"[g8] report → {args.report_out}")
    print(f"[g8] success_diff {verdict['success_diff']} "
          f"ci95 {verdict['success_diff_ci95']} "
          f"token_reduction {verdict['token_reduction']} "
          f"g8_pass={verdict['g8_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
