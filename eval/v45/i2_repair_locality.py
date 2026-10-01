"""I2 repair-local recomputation vs full-scope rebuild
(SPEC_V4_5 §04; D03, D04; V45-04.01–04.04).

Paired arms over an identical seeded derivation graph, each in its own
disposable real ``Store`` — no producer is shimmed: observations come
from the real ``consolidate_windowed`` pass (which writes
``observation_evidence`` provenance rows), views from real
``Synthesizer.compose`` calls (which write ``dependency_edges``), so
``impact_closure`` reaches every dependent through the same graph
surfaces the kernel and the deletion engine walk.

* **repair** — one claim corrected (rev1 closed, rev2 opened), then the
  real pipeline: ``invalidate_dependents`` inside the correction's own
  transaction (epoch bump + held/stale marks), ``plan_repair`` +
  ``persist_plan`` (digest-bound ``repair_plan`` object), then
  ``apply_repair`` — the bounded executor that re-walks the closure,
  fences the plan, reconsolidates the touched slot, and recomposes the
  bound view.
* **full** — the identical seed and the identical correction, then
  ``full_rebuild``: the same producers over the whole scope.

Measured, not asserted:

- **work** = ``objects_scanned`` / ``objects_evaluated`` /
  ``objects_recomputed`` — the same counters on both arms;
- **D03** = the repair arm evaluates and recomputes at least 50% fewer
  objects than the rebuild arm on the same post-correction input
  (V45-04.04), at equal post-state (the corrected observation text is
  live on both arms);
- **D04** = two fence probes: a plan whose declared forecast omits a
  real dependent is ``complete=False`` and refuses with
  ``CONTEXT_INCOMPLETE``, and a plan whose closure grew between plan and
  apply is re-walked and refused the same way — with zero committed
  writes in both cases.

CLI: ``python -m eval.v45.i2_repair_locality --out <dir>`` →
``i2_repair_report.json`` + ``.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import Any, Optional

from verbatim.core.types import ErrorCode, VerbatimError, json_dumps
from verbatim.governance import CallerV3, create_grant, register_principal, seed_purposes
from verbatim.kernel import Kernel
from verbatim.observations.consolidate import (
    ConsolidationWindow,
    consolidate_windowed,
)
from verbatim.repair import (
    apply_repair,
    full_rebuild,
    invalidate_dependents,
    persist_plan,
    plan_repair,
)
from verbatim.storage.store import Store
from verbatim.synthesis import Synthesizer

from . import corpus


SCOPE = "sA"
SLOTS = 16
MIN_PROOF = 1
#: One claim slot is corrected; everything else must be untouched.
CORRECTED_SLOT = 0
T0 = 1_700_000_000_000_000


def _caller() -> CallerV3:
    return CallerV3(principal_id="human:alice")


def _claim(conn: Any, claim_id: str, *, slot: int, value: str,
           recorded_from: int = 1, rev: int = 1,
           recorded_until: Optional[int] = None) -> None:
    """A consolidation-eligible structured claim — the unit the
    recomputation counters measure."""
    conn.execute(
        "INSERT INTO claims (claim_id, scope_id, subject_id, predicate,"
        " created_event) VALUES (?,?,?,?,0)",
        (claim_id, SCOPE, f"svc{slot}", "mode"),
    )
    conn.execute(
        "INSERT INTO claim_revisions (claim_id, revision, state,"
        " object_json, polarity, modality, recorded_from,"
        " recorded_until) VALUES (?,?, 'active', ?, 'affirmative',"
        " 'asserted', ?, ?)",
        (claim_id, rev,
         json_dumps({"kind": "literal", "text": value}),
         recorded_from, recorded_until),
    )


def _correct(conn: Any, claim_id: str, new_value: str, *, at_seq: int) -> None:
    """The localized correction — close the head revision, open the
    corrected one (the same physical shape the lifecycle machine
    produces)."""
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


def _seed(store: Store, *, slots: int = SLOTS) -> dict:
    """The identical seed both arms build: ``slots`` independent
    claim slots, one real observation + one persisted view each.

    Every slot's claim carries its own (subject, predicate) pair so a
    localized correction touches exactly one consolidation slot — the
    honest locality surface D03 measures.
    """
    kernel = Kernel(store)
    synth = Synthesizer(store, kernel=kernel)
    with store.tx() as conn:
        corpus.seed_scope(conn, SCOPE)
        seed_purposes(conn)
        register_principal(conn, kind="human", principal_id="human:alice")
        create_grant(
            conn, scope_id=SCOPE, principal_id="human:alice",
            verbs={"read", "quote", "derive", "review", "admin"},
            issuer_id="human:alice",
            purposes=["recall", "admin", "review", "derive"],
        )
        synth.register_producer(conn)
        for i in range(slots):
            _claim(conn, f"c{i}", slot=i, value=f"value {i}")
        # Real producer pass — writes observations + observation_evidence
        # rows the impact walk reaches through the registered side walker.
        consolidate_windowed(
            conn, SCOPE, window=ConsolidationWindow(since_seq=0),
            min_proof=MIN_PROOF,
        )
    views: dict[str, str] = {}
    for i in range(slots):
        view = synth.compose(
            SCOPE, caller=_caller(), purpose="recall",
            view_kind="typed_summary",
            inputs=[{"kind": "claim", "id": f"c{i}", "revision": 1}],
            now_us=T0 + i,
        )
        views[f"c{i}"] = view.view_id
    return {"scope_id": SCOPE, "slots": slots, "views": views,
            "synth": synth, "kernel": kernel}


def _live_observations(store: Store) -> set:
    with store.read() as conn:
        return {
            r[0]
            for r in conn.execute(
                "SELECT text FROM observations"
                " WHERE scope_id = ? AND recorded_until IS NULL",
                (SCOPE,),
            ).fetchall()
        }


def _run_repair_arm(workdir: str, *, slots: int) -> dict:
    store = corpus.make_store(os.path.join(workdir, "arm_repair"), "rep.db")
    try:
        seeded = _seed(store, slots=slots)
        synth, kernel = seeded["synth"], seeded["kernel"]
        cid = f"c{CORRECTED_SLOT}"
        # Correction + plan + immediate invalidation — one commit, so
        # held/stale marks land atomically with the corrected revision.
        with store.tx() as conn:
            _correct(conn, cid, f"corrected value {CORRECTED_SLOT}",
                     at_seq=50)
            plan = plan_repair(conn, SCOPE, [("claim", cid, 1)])
            plan_id = persist_plan(store, conn, plan)
            inv = invalidate_dependents(
                conn, kernel, SCOPE, [("claim", cid, 1)],
                synthesizer=synth,
            )
            held_in_tx = dict(inv["marked_held"])
        rep = apply_repair(
            store, plan_id, caller=_caller(), synthesizer=synth,
            min_proof=MIN_PROOF,
        )
        return {
            "plan_id": plan_id,
            "plan_complete": plan["complete"],
            "plan_targets": len(plan["targets"]),
            "held_marks_in_correction_tx": held_in_tx,
            "report": rep,
            "post_state_texts": sorted(_live_observations(store)),
        }
    finally:
        store.close()


def _run_full_arm(workdir: str, *, slots: int) -> dict:
    store = corpus.make_store(os.path.join(workdir, "arm_full"), "full.db")
    try:
        seeded = _seed(store, slots=slots)
        synth = seeded["synth"]
        cid = f"c{CORRECTED_SLOT}"
        # Same input (V45-04.04): the rebuild store receives the
        # identical correction; it just recomputes the whole scope.
        with store.tx() as conn:
            _correct(conn, cid, f"corrected value {CORRECTED_SLOT}",
                     at_seq=50)
        rep = full_rebuild(
            store, SCOPE, caller=_caller(), synthesizer=synth,
            min_proof=MIN_PROOF,
        )
        return {"report": rep,
                "post_state_texts": sorted(_live_observations(store))}
    finally:
        store.close()


def _run_d04_probes(workdir: str) -> dict:
    """Both incomplete-plan fences on a fresh seeded store.

    Probe 1 — declared forecast omits a real dependent: the plan itself
    is ``complete=False`` and apply refuses before touching anything.

    Probe 2 — the closure grows between plan and apply (a new persisted
    view binds the corrected claim): the apply-time re-walk trips the
    same ``CONTEXT_INCOMPLETE`` fence; nothing commits.
    """
    store = corpus.make_store(os.path.join(workdir, "arm_d04"), "d04.db")
    try:
        seeded = _seed(store, slots=4)
        synth = seeded["synth"]
        out: dict[str, Any] = {}
        # Probe 1: forecast names only the seed — the observation and the
        # view dependents are missing → incomplete plan.
        with store.tx() as conn:
            _correct(conn, "c0", "d04 corrected 0", at_seq=50)
            bad = plan_repair(
                conn, SCOPE, [("claim", "c0", 1)],
                declared=[("claim", "c0", 1)],
            )
        out["incomplete_forecast"] = {
            "complete": bad["complete"],
            "missing": len(bad["missing"]),
        }
        try:
            apply_repair(store, bad)
            out["incomplete_forecast"]["apply_refused"] = False
            out["incomplete_forecast"]["error_code"] = None
        except VerbatimError as exc:
            out["incomplete_forecast"]["apply_refused"] = True
            out["incomplete_forecast"]["error_code"] = exc.code.value

        # Probe 2: a complete plan, then the graph grows — a fresh view
        # binds the SAME pinned revision the plan forecasted, AFTER the
        # forecast was fixed. (An identical-input compose would return
        # the same view object idempotently; a genuinely new dependent
        # needs a new input set over the planned revision.)
        with store.tx() as conn:
            good = plan_repair(conn, SCOPE, [("claim", "c1", 1)])
        assert good["complete"] is True
        late = synth.compose(
            SCOPE, caller=_caller(), purpose="recall",
            view_kind="typed_summary",
            inputs=[{"kind": "claim", "id": "c1", "revision": 1},
                    {"kind": "claim", "id": "c2", "revision": 1}],
            now_us=T0 + 999,
        )
        try:
            apply_repair(
                store, good, caller=_caller(), synthesizer=synth,
                min_proof=MIN_PROOF,
            )
            out["grown_closure"] = {"apply_refused": False,
                                    "error_code": None}
        except VerbatimError as exc:
            out["grown_closure"] = {"apply_refused": True,
                                    "error_code": exc.code.value}
        # The refused apply committed nothing — the hidden dependent was
        # neither recomputed nor marked held by a half-executed job.
        from verbatim.storage import repos_v4
        with store.read() as conn:
            late_obj = repos_v4.get(
                conn, "objects",
                {"kind": "derived_view", "object_id": late.view_id},
            )
        out["late_view_disposition"] = (
            late_obj["disposition"] if late_obj else None
        )
        out["partial_writes"] = int(
            late_obj is not None
            and late_obj["disposition"] != "active"
        )
        return out
    finally:
        store.close()


def run_i2(slots: int = SLOTS,
           workdir: Optional[str] = None) -> dict:
    """Run the paired repair/rebuild arms plus the D04 fence probes."""
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="v45_i2_")

    repair = _run_repair_arm(workdir, slots=slots)
    full = _run_full_arm(workdir, slots=slots)
    d04 = _run_d04_probes(workdir)

    rr, fr = repair["report"], full["report"]
    eval_red = (
        1.0 - rr["objects_evaluated"] / fr["objects_evaluated"]
        if fr["objects_evaluated"] else 0.0
    )
    rec_red = (
        1.0 - rr["objects_recomputed"] / fr["objects_recomputed"]
        if fr["objects_recomputed"] else 0.0
    )
    scan_red = (
        1.0 - rr["objects_scanned"] / fr["objects_scanned"]
        if fr["objects_scanned"] else 0.0
    )
    # Equal post-state: the corrected value is live on both arms — the
    # cheaper arm produced the same answer, not a smaller one.
    marker = f"corrected value {CORRECTED_SLOT}"
    post_equal = (
        any(marker in t for t in repair["post_state_texts"])
        and repair["post_state_texts"] == full["post_state_texts"]
    )
    d03 = bool(
        eval_red >= 0.5 and rec_red >= 0.5 and post_equal
        and rr["objects_recomputed"] >= 1
        and repair["held_marks_in_correction_tx"].get("views", 0) >= 1
        and repair["held_marks_in_correction_tx"].get(
            "observations", 0) >= 1
    )
    d04_probes = d04
    d04 = bool(
        d04_probes["incomplete_forecast"]["apply_refused"]
        and d04_probes["incomplete_forecast"]["error_code"]
        == ErrorCode.CONTEXT_INCOMPLETE.value
        and d04_probes["grown_closure"]["apply_refused"]
        and d04_probes["grown_closure"]["error_code"]
        == ErrorCode.CONTEXT_INCOMPLETE.value
        and d04_probes["partial_writes"] == 0
    )
    report = {
        "experiment": "i2_repair_locality",
        "spec": {
            "requirements": [
                "V45-04.01", "V45-04.02", "V45-04.03", "V45-04.04",
            ],
            "acceptance": ["D03", "D04"],
        },
        "sample_size": {
            "slots": slots,
            "claims": slots,
            "views": slots,
            "corrected_claims": 1,
            "arms": 2,
            "note": "identical seeded graph per arm in disposable"
                    " stores; the rebuild arm receives the identical"
                    " correction (same input, V45-04.04)",
        },
        "definitions": {
            "objects_evaluated": (
                "producer work units: consolidation slot-value"
                " candidates re-derived + scenes reconciled + views"
                " recomposed + profile entries scanned"
            ),
            "objects_recomputed": (
                "durable writes: observation writes/retirements +"
                " refreshed scenes + recomposed views + profile"
                " mutations"
            ),
            "objects_scanned": (
                "touched surface: slots touched + scenes/views"
                " dispatched + (full arm) every claim/observation row"
            ),
        },
        "arms": {
            "repair": {
                "plan_id": repair["plan_id"],
                "plan_complete": repair["plan_complete"],
                "plan_targets": repair["plan_targets"],
                "held_marks_in_correction_tx":
                    repair["held_marks_in_correction_tx"],
                "objects_scanned": rr["objects_scanned"],
                "objects_evaluated": rr["objects_evaluated"],
                "objects_recomputed": rr["objects_recomputed"],
                "marked_held": rr["marked_held"],
                "consolidation": rr["consolidation"],
                "views": rr["views"],
                "failures": rr["failures"],
            },
            "full_rebuild": {
                "objects_scanned": fr["objects_scanned"],
                "objects_evaluated": fr["objects_evaluated"],
                "objects_recomputed": fr["objects_recomputed"],
                "consolidation": fr["consolidation"],
                "views_recomposed": len(fr["views"]),
                "failures": fr["failures"],
            },
        },
        "totals": {
            "reduction": {
                "objects_scanned": round(scan_red, 4),
                "objects_evaluated": round(eval_red, 4),
                "objects_recomputed": round(rec_red, 4),
            },
        },
        "d03": {
            "threshold": 0.50,
            "evaluated_reduction": round(eval_red, 4),
            "recomputed_reduction": round(rec_red, 4),
            "post_state_equal": post_equal,
            "measured": (
                f"repair evaluated {rr['objects_evaluated']} vs full"
                f" {fr['objects_evaluated']} ({eval_red:.1%} less);"
                f" recomputed {rr['objects_recomputed']} vs"
                f" {fr['objects_recomputed']} ({rec_red:.1%} less);"
                " corrected observation live on both arms"
            ),
        },
        "d04": d04_probes,
    }
    report["met"] = bool(d03 and d04)
    report["d03"]["met"] = d03
    report["d04_met"] = d04
    return report


def _md(report: dict) -> str:
    t = report["totals"]["reduction"]
    d03, d04 = report["d03"], report["d04"]
    rep_arm = report["arms"]["repair"]
    full_arm = report["arms"]["full_rebuild"]
    lines = [
        "# I2 — repair-local recomputation vs full rebuild (D03/D04)",
        "",
        f"- corpus: {report['sample_size']['slots']} claim slots,"
        f" {report['sample_size']['views']} persisted views,"
        " 1 localized correction",
        f"- objects scanned: repair **{rep_arm['objects_scanned']}**"
        f" vs rebuild **{full_arm['objects_scanned']}**"
        f" → {t['objects_scanned']:.1%} less",
        f"- objects evaluated: **{rep_arm['objects_evaluated']}**"
        f" vs **{full_arm['objects_evaluated']}**"
        f" → {t['objects_evaluated']:.1%} less",
        f"- objects recomputed: **{rep_arm['objects_recomputed']}**"
        f" vs **{full_arm['objects_recomputed']}**"
        f" → {t['objects_recomputed']:.1%} less",
        f"- post-state equal (corrected value live on both arms):"
        f" **{d03['post_state_equal']}**",
        f"- held/stale marks inside the correction commit:"
        f" {rep_arm['held_marks_in_correction_tx']}",
        f"- D03 met (≥50% on evaluated+recomputed): **{d03['met']}**",
        f"- D04 incomplete-forecast refused:"
        f" {d04['incomplete_forecast']['apply_refused']}"
        f" ({d04['incomplete_forecast']['error_code']})",
        f"- D04 grown-closure refused:"
        f" {d04['grown_closure']['apply_refused']}"
        f" ({d04['grown_closure']['error_code']}),"
        f" partial writes {d04['partial_writes']}",
        f"- met: **{report['met']}**",
        "",
        "| arm | scanned | evaluated | recomputed |",
        "| --- | --- | --- | --- |",
        f"| repair | {rep_arm['objects_scanned']} |"
        f" {rep_arm['objects_evaluated']} |"
        f" {rep_arm['objects_recomputed']} |",
        f"| full rebuild | {full_arm['objects_scanned']} |"
        f" {full_arm['objects_evaluated']} |"
        f" {full_arm['objects_recomputed']} |",
    ]
    return "\n".join(lines) + "\n"


def write_reports(report: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "i2_repair_report.json")
    mpath = os.path.join(out_dir, "i2_repair_report.md")
    with open(jpath, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=False)
    with open(mpath, "w") as fh:
        fh.write(_md(report))
    return {"json": jpath, "md": mpath}


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="I2 repair-local recomputation (D03/D04)"
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--slots", type=int, default=SLOTS)
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args(argv)
    report = run_i2(slots=args.slots, workdir=args.workdir)
    if args.out:
        paths = write_reports(report, args.out)
        print(f"wrote {paths['json']}")
    else:
        print(json.dumps(report["totals"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
