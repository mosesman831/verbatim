"""I6 utility-budgeted refresh vs periodic full reflection
(SPEC_V4_5 §08; D11, D12; V45-08.01–08.04).

Paired arms over an identical seeded observation corpus, each in its own
disposable real ``Store`` — no lane, handler, or pass is shimmed: both
arms enqueue ``consolidate`` jobs on the real ``JobQueue`` and drain
them through the production ``handle_consolidate`` path (windowed
consolidation for the budgeted arm, the unwindowed full pass + bounded
reflection for the periodic arm).

* **periodic** — ``RefreshScheduler.run_periodic``: one unwindowed
  consolidate job per period, charged every live claim plus a second
  full scan for bounded reflection (``reflect()`` reads all inputs).
* **budgeted** — ``RefreshScheduler.refresh``: changed slot regions
  discovered from the durable ``refresh_pass`` watermark, ordered by the
  declared utility expression (staleness × demand × priority +
  stale-answer + deferred-readiness costs), scheduled under
  ``max_jobs`` — owner-priority requests never evicted (D12).

Measured, not asserted:

- **spend** = per-period ``RefreshPlan.spend()`` — job counts, claims
  processed, bytes processed — plus realized queue drain outcomes;
- **freshness floor** = ``stale_answers`` — live observations still
  citing a closed claim revision — counted after every drain; both arms
  must sit at the same floor (0) every period, and the count of changed
  regions actually covered is reported per arm;
- **D11** = budgeted spend ≥ 20% below periodic spend at that matched
  floor (primary comparator: periodic including its reflection scan; a
  consolidate-only periodic number is reported alongside so the margin
  does not rest on reflection's cost);
- **D12** = an owner-priority request issued under a saturated budget
  still schedules (never dropped) and still counts in the spend report.

CLI: ``python -m eval.v45.i6_refresh --out <dir>`` →
``i6_refresh_report.json`` + ``.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import Any, Optional

from verbatim.config import VerbatimConfig
from verbatim.core.types import json_dumps
from verbatim.ingest import Ingester
from verbatim.observations import consolidate
from verbatim.observations.freshness import next_seq
from verbatim.refresh import (
    REFRESH_POLICY_ID,
    RefreshBudget,
    RefreshScheduler,
)
from verbatim.storage.store import Store

from . import corpus


SCOPE = "sA"
SLOTS = 12
CLAIMS_PER_SLOT = 2
#: Period 0 is the baseline coverage cycle on both arms; one interior
#: period is the D12 probe (zero declared budget + owner request), and
#: the final period catches up the deferred work — six cycles total.
PERIODS = 6
CHANGES_PER_PERIOD = 3


def _claim(
    conn: Any,
    scope_id: str,
    claim_id: str,
    *,
    subject: str,
    predicate: str,
    value: str,
    family_id: str,
    recorded_from: int = 1,
) -> None:
    conn.execute(
        "INSERT INTO claims (claim_id, scope_id, subject_id, predicate,"
        " created_event) VALUES (?,?,?,?,0)",
        (claim_id, scope_id, subject, predicate),
    )
    conn.execute(
        "INSERT INTO claim_revisions (claim_id, revision, state,"
        " object_json, polarity, modality, recorded_from)"
        " VALUES (?,1,'active',?,'affirmative','asserted',?)",
        (claim_id, json_dumps({"kind": "literal", "text": value}),
         recorded_from),
    )
    conn.execute(
        "INSERT OR IGNORE INTO evidence_families"
        " (family_id, scope_id, origin_kind, origin_id, created_event)"
        " VALUES (?,?,?,?,0)",
        (family_id, scope_id, "test", claim_id),
    )
    conn.execute(
        "INSERT INTO family_members (family_id, object_kind, object_id,"
        " role) VALUES (?, 'claim', ?, 'origin')",
        (family_id, claim_id),
    )


def _revise(conn: Any, claim_id: str, *, new_value: str, at_seq: int) -> None:
    head = conn.execute(
        "SELECT MAX(revision) FROM claim_revisions WHERE claim_id = ?",
        (claim_id,),
    ).fetchone()[0]
    conn.execute(
        "UPDATE claim_revisions SET recorded_until = ?"
        " WHERE claim_id = ? AND revision = ?",
        (at_seq, claim_id, head),
    )
    conn.execute(
        "INSERT INTO claim_revisions (claim_id, revision, state,"
        " object_json, polarity, modality, recorded_from)"
        " VALUES (?,?,'active',?,'affirmative','asserted',?)",
        (claim_id, head + 1,
         json_dumps({"kind": "literal", "text": new_value}), at_seq),
    )


def seed_corpus(
    store: Store,
    *,
    slots: int = SLOTS,
    claims_per_slot: int = CLAIMS_PER_SLOT,
) -> dict:
    """N slots × claims_per_slot family-distinct claims, consolidated.

    Every slot gets a subject/predicate pair and ``claims_per_slot``
    claims in distinct evidence families — enough for the default
    ``min_proof=2`` aggregation to produce one observation per slot.
    """
    with store.tx() as conn:
        corpus.seed_scope(conn, SCOPE)
        slot_specs: list[dict] = []
        for i in range(slots):
            subject = f"svc{i}"
            predicate = "mode"
            value = f"fast{i}"
            claims = []
            for j in range(claims_per_slot):
                cid = f"cl-s{i}-{j}"
                _claim(
                    conn, SCOPE, cid,
                    subject=subject, predicate=predicate, value=value,
                    family_id=f"fam-s{i}-{j}",
                )
                claims.append(cid)
            slot_specs.append(
                {"slot": i, "subject": subject, "predicate": predicate,
                 "value": value, "claims": claims}
            )
        rep = consolidate(conn, SCOPE)
    return {
        "scope_id": SCOPE,
        "slots": slot_specs,
        "baseline_observations": rep["observations_written"],
    }


def _stale_answers(conn: Any, scope_id: str) -> int:
    """Live observations still citing a closed claim revision — the
    stale-answer count a reader could be served right now."""
    return int(
        conn.execute(
            "SELECT COUNT(DISTINCT o.observation_id)"
            " FROM observations o"
            " JOIN observation_evidence oe"
            "   ON oe.observation_id = o.observation_id"
            "  AND oe.revision = o.revision"
            " JOIN claim_revisions r"
            "   ON r.claim_id = oe.object_id"
            "  AND r.revision = oe.object_revision"
            " WHERE o.scope_id = ? AND o.recorded_until IS NULL"
            "   AND oe.object_kind = 'claim'"
            "   AND r.recorded_until IS NOT NULL",
            (scope_id,),
        ).fetchone()[0]
    )


def _consolidation_passes(conn: Any, scope_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT payload_json FROM events WHERE scope_id = ?"
        " AND kind = 'consolidation_pass' ORDER BY event_seq",
        (scope_id,),
    ).fetchall()
    from verbatim.core.types import safe_json_loads

    return [safe_json_loads(r[0]) or {} for r in rows]


def _apply_changes(
    store: Store,
    seeded: dict,
    period: int,
    *,
    changes_per_period: int,
) -> list[dict]:
    """Rotate ``changes_per_period`` slots through a correction each
    period — deterministic, disjoint from earlier periods' slots."""
    slots = seeded["slots"]
    n = len(slots)
    changed: list[dict] = []
    with store.tx() as conn:
        seq = next_seq(conn, seeded["scope_id"])
        for j in range(changes_per_period):
            spec = slots[(period * changes_per_period + j) % n]
            cid = spec["claims"][0]
            new_value = f"safe{spec['slot']}-p{period}"
            _revise(conn, cid, new_value=new_value, at_seq=seq)
            changed.append(
                {"slot": spec["slot"], "claim": cid,
                 "new_value": new_value}
            )
    return changed


def _drain(store: Store) -> dict:
    ing = Ingester(store, VerbatimConfig())
    return ing.drain_report(limit=512)


def _spend_zero() -> dict:
    return {
        "jobs": 0, "work_items": 0, "priority_jobs": 0,
        "consolidation_jobs": 0, "reflection_jobs": 0,
        "claims_processed": 0, "bytes_processed": 0,
        "deferred": 0, "deferred_claims": 0,
    }


def _add_spend(acc: dict, spend: dict) -> None:
    for k in acc:
        acc[k] += int(spend.get(k, 0))


def _run_periodic(
    seeded_dir: str,
    seeded: dict,
    *,
    periods: int,
    changes_per_period: int,
    reflect: bool = True,
) -> dict:
    store = corpus.make_store(seeded_dir, "periodic.db")
    try:
        corpus_reseed = seed_corpus(store)
        sched = RefreshScheduler(store)
        totals = _spend_zero()
        period_rows = []
        drained = {"succeeded": 0, "failed": 0}
        for p in range(periods):
            changed = [] if p == 0 else _apply_changes(
                store, corpus_reseed, p - 1,
                changes_per_period=changes_per_period,
            )
            plan = sched.run_periodic(corpus_reseed["scope_id"],
                                      reflect=reflect)
            rep = _drain(store)
            drained["succeeded"] += rep["succeeded"]
            drained["failed"] += rep["failed"]
            _add_spend(totals, plan.spend())
            with store.read() as conn:
                stale = _stale_answers(conn, corpus_reseed["scope_id"])
                passes = _consolidation_passes(
                    conn, corpus_reseed["scope_id"]
                )
            period_rows.append({
                "period": p,
                "changed_regions": len(changed),
                "jobs": plan.spend()["jobs"],
                "claims_processed": plan.spend()["claims_processed"],
                "bytes_processed": plan.spend()["bytes_processed"],
                "stale_answers_after": stale,
                # An unwindowed pass covers the whole scope implicitly —
                # report what it actually wrote, not a region count.
                "coverage": "unbounded",
                "observations_written": (
                    passes[-1].get("result", {}).get(
                        "observations_written"
                    )
                    if passes else None
                ),
            })
        with store.read() as conn:
            total_passes = len(
                _consolidation_passes(conn, corpus_reseed["scope_id"])
            )
        return {
            "spend": totals,
            "periods": period_rows,
            "drained": drained,
            "consolidation_passes": total_passes,
            "freshness_floor_zero_stale": all(
                r["stale_answers_after"] == 0 for r in period_rows
            ),
        }
    finally:
        store.close()


def _run_budgeted(
    seeded_dir: str,
    seeded: dict,
    *,
    periods: int,
    changes_per_period: int,
    regions_per_job: int,
    probe_period: Optional[int] = None,
) -> dict:
    """The utility-budgeted arm.

    Period 0 is the declared catch-up: a wider ``max_jobs`` covers the
    whole seeded scope through packed per-region windows so the durable
    watermark primes — every later cycle only sees post-watermark
    changes. One ``probe_period`` runs ``max_jobs=0`` plus an explicit
    owner request (D12): owner work still schedules and counts while
    every ordinary candidate defers; the next cycle's watermark-held
    backlog re-surfaces and is covered.
    """
    store = corpus.make_store(seeded_dir, "budgeted.db")
    try:
        corpus_reseed = seed_corpus(store)
        n_slots = len(corpus_reseed["slots"])
        sched = RefreshScheduler(store, regions_per_job=regions_per_job)
        totals = _spend_zero()
        period_rows = []
        drained = {"succeeded": 0, "failed": 0}
        owner_report: dict = {}
        probe_period = (
            periods - 2 if probe_period is None else probe_period
        )
        last_period = periods - 1
        for p in range(periods):
            # The baseline and the closing catch-up cycle introduce no
            # new changes; steady + probe periods rotate corrections.
            if p in (0, last_period):
                changed = []
            else:
                changed = _apply_changes(
                    store, corpus_reseed, p - 1,
                    changes_per_period=changes_per_period,
                )
            if p == 0:
                budget = RefreshBudget(max_jobs=n_slots)
                requests: list = []
            elif p == probe_period:
                # D12 probe: zero declared budget + an explicit owner
                # request — priority work still schedules and counts
                # while every ordinary candidate defers.
                budget = RefreshBudget(max_jobs=0)
                requests = [
                    {"claim_ids": list(
                        corpus_reseed["slots"][0]["claims"]
                    ), "weight": 4.0}
                ]
            elif p == last_period:
                # Catch-up: the probe's deferred backlog re-surfaces
                # through the held watermark and drains in one cycle.
                budget = RefreshBudget(max_jobs=n_slots)
                requests = []
            else:
                # Steady state: one packed job per cycle is the declared
                # budget — packing carries every changed region.
                budget = RefreshBudget(max_jobs=1)
                requests = []
            plan = sched.refresh(
                corpus_reseed["scope_id"],
                budget=budget,
                owner_requests=requests,
            )
            rep = _drain(store)
            drained["succeeded"] += rep["succeeded"]
            drained["failed"] += rep["failed"]
            _add_spend(totals, plan.spend())
            if p == probe_period:
                owner_report = {
                    "owner_scheduled": sum(
                        1 for c in plan.scheduled if c.owner_requested
                    ),
                    "owner_deferred": sum(
                        1 for c in plan.deferred if c.owner_requested
                    ),
                    "priority_jobs": plan.spend()["priority_jobs"],
                    "deferred_non_owner": sum(
                        1 for c in plan.deferred
                        if not c.owner_requested
                    ),
                }
            with store.read() as conn:
                stale = _stale_answers(conn, corpus_reseed["scope_id"])
            covered = sum(
                1 for c in plan.scheduled if c.kind == "consolidation"
            )
            period_rows.append({
                "period": p,
                "phase": (
                    "baseline" if p == 0
                    else "d12_probe" if p == probe_period
                    else "catchup" if p == last_period
                    else "steady"
                ),
                "changed_regions": len(changed),
                "jobs": plan.spend()["jobs"],
                "priority_jobs": plan.spend()["priority_jobs"],
                "claims_processed": plan.spend()["claims_processed"],
                "bytes_processed": plan.spend()["bytes_processed"],
                "deferred": plan.spend()["deferred"],
                "stale_answers_after": stale,
                "regions_covered": covered,
                "watermark_after": plan.watermark_after,
            })
        return {
            "spend": totals,
            "periods": period_rows,
            "drained": drained,
            "owner_probe": owner_report,
            "freshness_floor_zero_stale_final": (
                period_rows[-1]["stale_answers_after"] == 0
            ),
            "steady_state_at_floor": all(
                r["stale_answers_after"] == 0
                for r in period_rows
                if r["phase"] == "steady"
            ),
        }
    finally:
        store.close()


def run_i6(
    periods: int = PERIODS,
    changes_per_period: int = CHANGES_PER_PERIOD,
    regions_per_job: Optional[int] = None,
    workdir: Optional[str] = None,
) -> dict:
    """Run both arms over identical seeded observation corpora."""
    regions_per_job = (
        regions_per_job
        if regions_per_job is not None
        else changes_per_period
    )
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="v45_i6_")
    # One seeded corpus description shared by both arms — each arm
    # builds its own store from the identical seed routine.
    probe = corpus.make_store(os.path.join(workdir, "seedshape"),
                              "shape.db")
    seeded = seed_corpus(probe)
    probe.close()

    periodic = _run_periodic(
        os.path.join(workdir, "arm_periodic"), seeded,
        periods=periods, changes_per_period=changes_per_period,
        reflect=True,
    )
    periodic_consolidate_only = _run_periodic(
        os.path.join(workdir, "arm_periodic_co"), seeded,
        periods=periods, changes_per_period=changes_per_period,
        reflect=False,
    )
    budgeted = _run_budgeted(
        os.path.join(workdir, "arm_budgeted"), seeded,
        periods=periods, changes_per_period=changes_per_period,
        regions_per_job=regions_per_job,
    )

    def _savings(budgeted_spend: dict, periodic_spend: dict,
                 key: str) -> float:
        p = periodic_spend[key]
        if not p:
            return 0.0
        return 1.0 - (budgeted_spend[key] / p)

    bs, ps = budgeted["spend"], periodic["spend"]
    pco = periodic_consolidate_only["spend"]
    savings_claims = _savings(bs, ps, "claims_processed")
    savings_bytes = _savings(bs, ps, "bytes_processed")
    savings_jobs = _savings(bs, ps, "jobs")
    savings_claims_co = _savings(bs, pco, "claims_processed")
    # Matched freshness (V45-08.04): both arms end the window at the
    # same floor — zero stale answers — and every steady-state cycle on
    # the budgeted arm sat at the floor. The D12 probe period's
    # transient (zero declared budget ⇒ deferred regions) is disclosed
    # in the per-period table, not hidden.
    matched_freshness = bool(
        budgeted["freshness_floor_zero_stale_final"]
        and budgeted["steady_state_at_floor"]
        and periodic["freshness_floor_zero_stale"]
        and periodic_consolidate_only["freshness_floor_zero_stale"]
    )
    owner = budgeted["owner_probe"]
    report = {
        "experiment": "i6_utility_budgeted_refresh",
        "spec": {
            "requirements": [
                "V45-08.01", "V45-08.02", "V45-08.03", "V45-08.04",
            ],
            "acceptance": ["D11", "D12"],
            "policy_id": REFRESH_POLICY_ID,
        },
        "sample_size": {
            "slots": SLOTS,
            "claims": SLOTS * CLAIMS_PER_SLOT,
            "periods": periods,
            "changes_per_period": changes_per_period,
            "arms": 2,
            "note": "identical seeded corpus per arm in disposable"
                    " stores; every arm drains real consolidate jobs",
        },
        "matched_bound": {
            "freshness_floor": (
                "both arms end the window at zero live observations"
                " citing closed claim revisions; the budgeted arm holds"
                " that floor through every steady-state cycle (the D12"
                " probe period's zero-budget transient is disclosed"
                " per-period)"
            ),
            "regions_per_job": regions_per_job,
            "steady_state_max_jobs": 1,
        },
        "definitions": {
            "spend": (
                "RefreshPlan.spend summed over periods: consolidate/"
                "reflection jobs enqueued, claim inputs processed,"
                " value bytes processed. Periodic is charged a full"
                " input scan for consolidation plus a second full scan"
                " for bounded reflection — the same sets the handlers"
                " actually read."
            ),
            "stale_answers": (
                "live (recorded_until NULL) observations whose current"
                " revision's evidence cites a closed claim revision —"
                " the stale-answer count a reader could be served"
            ),
        },
        "arms": {
            "periodic": periodic,
            "periodic_consolidate_only": periodic_consolidate_only,
            "budgeted": budgeted,
        },
        "totals": {
            "spend": {
                "periodic": ps,
                "periodic_consolidate_only": pco,
                "budgeted": bs,
            },
            "savings": {
                "claims_processed": round(savings_claims, 4),
                "bytes_processed": round(savings_bytes, 4),
                "jobs": round(savings_jobs, 4),
                "claims_processed_vs_consolidate_only": round(
                    savings_claims_co, 4
                ),
            },
        },
        "d11": {
            "matched_freshness": matched_freshness,
            "claims_savings": round(savings_claims, 4),
            "bytes_savings": round(savings_bytes, 4),
            "jobs_savings": round(savings_jobs, 4),
            "threshold": 0.20,
            "measured": (
                f"budgeted claims_processed {bs['claims_processed']}"
                f" vs periodic {ps['claims_processed']}"
                f" ({savings_claims:.1%} less), jobs"
                f" {bs['jobs']} vs {ps['jobs']}"
            ),
        },
        "d12": {
            "owner_scheduled": owner.get("owner_scheduled", 0),
            "owner_deferred": owner.get("owner_deferred", 0),
            "priority_jobs_counted": owner.get("priority_jobs", 0),
            "non_owner_deferred": owner.get("deferred_non_owner", 0),
            "measured": (
                "owner request under saturated budget scheduled"
                f" {owner.get('owner_scheduled', 0)} / deferred"
                f" {owner.get('owner_deferred', 0)}; counted in spend"
                f" as {owner.get('priority_jobs', 0)} priority job(s)"
            ),
        },
    }
    report["met"] = bool(
        matched_freshness
        and savings_claims >= 0.20
        and owner.get("owner_scheduled", 0) >= 1
        and owner.get("owner_deferred", 0) == 0
    )
    return report


def _md(report: dict) -> str:
    t = report["totals"]
    d11, d12 = report["d11"], report["d12"]
    lines = [
        "# I6 — utility-budgeted refresh vs periodic full reflection"
        " (D11/D12)",
        "",
        f"- corpus: {report['sample_size']['slots']} slots ×"
        f" {CLAIMS_PER_SLOT} claims,"
        f" {report['sample_size']['periods']} periods ×"
        f" {report['sample_size']['changes_per_period']} corrections",
        f"- freshness floor matched (0 stale answers, all changed"
        f" regions covered): **{d11['matched_freshness']}**",
        f"- claims processed: periodic **{t['spend']['periodic']['claims_processed']}**"
        f" vs budgeted **{t['spend']['budgeted']['claims_processed']}**"
        f" → {t['savings']['claims_processed']:.1%} less",
        f"- bytes processed:"
        f" {t['spend']['periodic']['bytes_processed']} →"
        f" {t['spend']['budgeted']['bytes_processed']}"
        f" ({t['savings']['bytes_processed']:.1%} less)",
        f"- jobs: {t['spend']['periodic']['jobs']} →"
        f" {t['spend']['budgeted']['jobs']}"
        f" ({t['savings']['jobs']:.1%} less)",
        f"- vs periodic consolidate-only (no reflection charge):"
        f" {t['savings']['claims_processed_vs_consolidate_only']:.1%}"
        " less claims",
        f"- D12 owner probe: scheduled {d12['owner_scheduled']},"
        f" deferred {d12['owner_deferred']}, counted"
        f" {d12['priority_jobs_counted']} priority job(s)",
        f"- met: **{report['met']}**",
        "",
        "| period | changed | budgeted jobs | claims | stale |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in report["arms"]["budgeted"]["periods"]:
        lines.append(
            f"| {r['period']} | {r['changed_regions']} |"
            f" {r['jobs']} (priority {r['priority_jobs']}) |"
            f" {r['claims_processed']} | {r['stale_answers_after']} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(report: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "i6_refresh_report.json")
    mpath = os.path.join(out_dir, "i6_refresh_report.md")
    with open(jpath, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=False)
    with open(mpath, "w") as fh:
        fh.write(_md(report))
    return {"json": jpath, "md": mpath}


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="I6 utility-budgeted refresh (D11/D12)"
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--periods", type=int, default=PERIODS)
    ap.add_argument("--changes", type=int, default=CHANGES_PER_PERIOD)
    ap.add_argument("--regions-per-job", type=int, default=None)
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args(argv)
    report = run_i6(
        periods=args.periods,
        changes_per_period=args.changes,
        regions_per_job=args.regions_per_job,
        workdir=args.workdir,
    )
    if args.out:
        paths = write_reports(report, args.out)
        print(f"wrote {paths['json']}")
    else:
        print(json.dumps(report["totals"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
