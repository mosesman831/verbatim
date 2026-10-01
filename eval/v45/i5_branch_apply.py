"""I5 reversible memory branches — measured isolation, review-gated
apply, and purge propagation (SPEC_V4_5 §07; D09, D10; V45-07.*).

Every scenario runs the real production surfaces in disposable
``Store`` instances: ``BranchService`` create/submit/apply/rebase/
abandon/diff, the real ``LifecycleMachine`` for parent movement, the
real ``ReviewsRepo`` review fence, ``kernel.invalidate`` +
``note_invalidation`` for live holds, the public
``VerbatimV3.delete_source`` suppression path, and the real
``plan_purge``/``execute_purge`` closure engine for erasure. Nothing is
simulated; every gate outcome is measured from returned errors and
committed state.

Measured scenarios:

- **d09_isolation** — a proposed ``archive`` op is invisible to live
  ``recall_v3`` until the reviewed apply lands; apply without an open
  review refuses ``INVALID_TRANSITION``; after apply the claim is
  archived and recall drops it; a replayed apply returns the stored
  receipt (``replayed=True``) rather than re-running effects.
- **moved_parent** — a parent advanced past its pinned revision is
  ``STALE_PROPOSAL`` at submit and again at apply under the open review;
  ``rebase`` re-pins honestly.
- **d10_suppressed** — ``delete_source`` suppression of the pinned
  parent's evidence makes apply refuse
  ``NOT_FOUND_OR_UNAUTHORIZED``; the failed apply commits nothing
  (branch stays live, proposed text never lands).
- **d10_purged** — ``execute_purge`` on the sole parent tombstones the
  branch through the deletion closure (disposition ``erased``, revision
  docs carry no proposed text); apply refuses with a typed error; no
  ``operation_receipts`` row exists; the purged claim's revisions carry
  no resurrected payload.
- **held_rebase** — a kernel invalidation on the pinned parent flips the
  branch to ``held`` inside the same transaction; submit/apply refuse
  ``INVALID_TRANSITION`` until an operator ``rebase`` returns it live.
- **abandon_diff** — ``diff`` reports per-op health honestly; an
  abandoned branch cannot apply and its audit record persists.

CLI: ``python -m eval.v45.i5_branch_apply --out <dir>`` →
``i5_branch_report.json`` + ``.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import Any, Optional

from verbatim.api_v3.facade import VerbatimV3
from verbatim.branches import BranchService
from verbatim.core.lifecycle import LifecycleMachine, TransitionCommand, read_claim_head
from verbatim.core.types import (
    ErrorCode,
    VerbatimError,
    safe_json_loads,
)
from verbatim.core.types_v3 import RecallRequestV3
from verbatim.governance import CallerV3, create_grant, register_principal, seed_purposes
from verbatim.kernel import Kernel
from verbatim.purge import execute_purge, plan_purge
from verbatim.retrieval.v3 import recall_v3
from verbatim.storage import repos_v4
from verbatim.storage.store import Store

from . import corpus


SCOPE = "sA"
T0 = 1_700_000_000_000_000


def _caller() -> CallerV3:
    return CallerV3(principal_id="human:alice")


def _bootstrap(conn: Any, scope_id: str = SCOPE) -> None:
    corpus.seed_scope(conn, scope_id)
    seed_purposes(conn)
    register_principal(conn, kind="human", principal_id="human:alice")
    create_grant(
        conn, scope_id=scope_id, principal_id="human:alice",
        verbs={"read", "quote", "derive", "review", "admin"},
        issuer_id="human:alice",
        purposes=None,
    )


def _seed(store: Store, claim_ids=("c1", "c2")) -> None:
    gen = store.projection_generation()
    with store.tx() as conn:
        _bootstrap(conn)
        for i, cid in enumerate(claim_ids):
            corpus.seed_claim(
                conn, cid, SCOPE, f"src:{cid}", f"sp:{cid}",
                f"claim body {cid} deployment state value {i}", gen,
            )


def _transition(store: Store, claim_id: str, effect: str, *,
                expected: int = 1) -> int:
    """A real lifecycle transition — the 'world moved' probe."""
    machine = LifecycleMachine(store)
    with store.tx() as conn:
        return machine.apply(
            TransitionCommand(
                claim_id=claim_id,
                expected_revision=expected,
                effect=effect,
                actor_id="human:alice",
                reason="eval transition",
            ),
            conn,
        )


def _purge(store: Store, *targets) -> dict:
    preview = plan_purge(store, SCOPE, list(targets), actor="alice")
    return execute_purge(store, preview["purge_id"])


def _refused(fn) -> Optional[str]:
    """Run a gate probe; return the typed error code or None."""
    try:
        fn()
    except VerbatimError as exc:
        return exc.code.value
    return None


def _delivered(store: Store, query: str) -> set:
    res = recall_v3(
        store,
        RecallRequestV3(
            query=query, scope_id=SCOPE, caller_id="human:alice",
            purpose="recall",
        ),
    )
    return {i.handle.object_id for p in res.packs for i in p.items}


def _objects_row(store: Store, object_id: str) -> Optional[dict]:
    with store.read() as conn:
        return repos_v4.get(
            conn, "objects", {"kind": "branch", "object_id": object_id}
        )


def _claim_head_state(store: Store, claim_id: str) -> Optional[str]:
    with store.read() as conn:
        head = read_claim_head(conn, claim_id)
        return head.state.value if head is not None else None


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------

def _s_d09(workdir: str) -> dict:
    store = corpus.make_store(os.path.join(workdir, "s_d09"), "d09.db")
    try:
        _seed(store, ("c1",))
        svc = BranchService(store)
        query = "claim body c1 deployment state"
        before = _delivered(store, query)
        bid = svc.create(
            _caller(), SCOPE, name="retire-c1",
            ops=[{"effect": "archive", "claim_id": "c1"}],
        )["branch_id"]
        mid = _delivered(store, query)
        gate = _refused(lambda: svc.apply(_caller(), bid))
        sub = svc.submit(_caller(), bid)
        applied = svc.apply(_caller(), bid)
        after = _delivered(store, query)
        replay = svc.apply(_caller(), bid)
        return {
            "isolated_before_apply": "c1" in before and "c1" in mid,
            "review_gate_code": gate,
            "review_opened": bool(sub["review_id"]),
            "forecast_recorded": bool(sub["forecast"]),
            "applied": applied["state"] == "applied"
            and applied["replayed"] is False,
            "head_state_after": _claim_head_state(store, "c1"),
            "recall_dropped_after_apply": "c1" not in after,
            "idempotent_replay": replay["replayed"] is True,
        }
    finally:
        store.close()


def _s_moved_parent(workdir: str) -> dict:
    store = corpus.make_store(os.path.join(workdir, "s_moved"), "mv.db")
    try:
        _seed(store, ("c1",))
        svc = BranchService(store)
        bid = svc.create(
            _caller(), SCOPE,
            ops=[{"effect": "archive", "claim_id": "c1"}],
        )["branch_id"]
        _transition(store, "c1", "archive")
        submit_code = _refused(lambda: svc.submit(_caller(), bid))
        svc.rebase(_caller(), bid)
        svc.submit(_caller(), bid)
        _transition(store, "c1", "restore", expected=2)
        apply_code = _refused(lambda: svc.apply(_caller(), bid))
        return {
            "moved_submit_code": submit_code,
            "moved_apply_code": apply_code,
            "rebase_repinned": True,
        }
    finally:
        store.close()


def _s_d10_suppressed(workdir: str) -> dict:
    store = corpus.make_store(os.path.join(workdir, "s_supp"), "sup.db")
    try:
        _seed(store, ("c1",))
        svc = BranchService(store)
        bid = svc.create(
            _caller(), SCOPE, name="restore-src",
            ops=[{"effect": "archive", "claim_id": "c1",
                  "new_object": {"kind": "literal",
                                 "text": "resurrect payload"}}],
        )["branch_id"]
        svc.submit(_caller(), bid)
        facade = VerbatimV3(store)
        out = facade.delete_source("src:c1", principal_id="human:alice")
        apply_code = _refused(lambda: svc.apply(_caller(), bid))
        doc = svc.get(_caller(), bid)
        return {
            "suppression_state": out.get("status") or out.get("state"),
            "apply_code": apply_code,
            "branch_state_after": doc["state"],
            "no_restore": _claim_head_state(store, "c1") != "archived",
        }
    finally:
        store.close()


def _s_d10_purged(workdir: str) -> dict:
    store = corpus.make_store(os.path.join(workdir, "s_purge"), "pg.db")
    try:
        _seed(store, ("c1",))
        svc = BranchService(store)
        bid = svc.create(
            _caller(), SCOPE,
            ops=[{"effect": "archive", "claim_id": "c1",
                  "new_object": {"kind": "literal",
                                 "text": "resurrect payload"}}],
        )["branch_id"]
        svc.submit(_caller(), bid)
        _purge(store, ("claim", "c1"))
        apply_code = _refused(lambda: svc.apply(_caller(), bid))
        with store.read() as conn:
            revs = conn.execute(
                "SELECT revision, state, object_json FROM claim_revisions"
                " WHERE claim_id='c1' ORDER BY revision"
            ).fetchall()
            receipt = repos_v4.get(
                conn, "operation_receipts",
                {"operation_id": f"branch-apply:{bid}"},
            )
            branch_revs = conn.execute(
                "SELECT metadata_json FROM object_revisions"
                " WHERE kind='branch' AND object_id = ?",
                (bid,),
            ).fetchall()
        obj = _objects_row(store, bid)
        tombstones_carry_no_text = all(
            "resurrect payload" not in (r[0] or "")
            and (safe_json_loads(r[0]) or {}).get("erased") is True
            for r in branch_revs
        )
        return {
            "apply_code": apply_code,
            "apply_refused_typed": apply_code in (
                ErrorCode.INVALID_TRANSITION.value,
                ErrorCode.NOT_FOUND_OR_UNAUTHORIZED.value,
                ErrorCode.NOT_FOUND_OR_FORBIDDEN.value,
            ),
            "claim_revisions_states": [r[1] for r in revs],
            "no_resurrect_text": all(
                "resurrect" not in (r[2] or "") for r in revs
            ),
            "no_apply_receipt": receipt is None,
            "branch_disposition": obj["disposition"] if obj else None,
            "tombstones_carry_no_text": tombstones_carry_no_text,
        }
    finally:
        store.close()


def _s_held_rebase(workdir: str) -> dict:
    store = corpus.make_store(os.path.join(workdir, "s_held"), "hd.db")
    try:
        _seed(store, ("c1",))
        svc = BranchService(store)
        kernel = Kernel(store)
        bid = svc.create(
            _caller(), SCOPE,
            ops=[{"effect": "archive", "claim_id": "c1"}],
        )["branch_id"]
        with store.tx() as conn:
            report = kernel.invalidate(
                conn,
                {"kind": "correction", "scope_ids": [SCOPE],
                 "object_refs": [["claim", "c1", 1]]},
            )
            marked = svc.note_invalidation(conn, report)
            _obj, doc = svc._doc(conn, bid)
            held_in_tx = doc["state"]
        doc_after = svc.get(_caller(), bid)
        submit_code = _refused(lambda: svc.submit(_caller(), bid))
        apply_code = _refused(lambda: svc.apply(_caller(), bid))
        rebased = svc.rebase(_caller(), bid)
        return {
            "held_in_invalidation_tx": held_in_tx,
            "marked": marked,
            "held_durable": doc_after["state"],
            "submit_code": submit_code,
            "apply_code": apply_code,
            "rebase_state": rebased["state"],
        }
    finally:
        store.close()


def _s_abandon_diff(workdir: str) -> dict:
    store = corpus.make_store(os.path.join(workdir, "s_ab"), "ab.db")
    try:
        _seed(store, ("c1", "c2"))
        svc = BranchService(store)
        bid = svc.create(
            _caller(), SCOPE, name="wont-fix",
            ops=[{"effect": "archive", "claim_id": "c1"},
                 {"effect": "archive", "claim_id": "c2"}],
        )["branch_id"]
        d0 = svc.diff(_caller(), bid)
        _transition(store, "c1", "archive")
        d1 = svc.diff(_caller(), bid)
        health = {o["claim_id"]: o["health"] for o in d1["ops"]}
        out = svc.abandon(_caller(), bid, reason="eval superseded")
        apply_code = _refused(lambda: svc.apply(_caller(), bid))
        obj = _objects_row(store, bid)
        return {
            "diff_initial_health": sorted(
                {o["health"] for o in d0["ops"]}
            ),
            "diff_moved_health": health.get("c1"),
            "abandon_state": out["state"],
            "apply_code": apply_code,
            "audit_preserved": bool(
                svc.get(_caller(), bid)["base"]["objects"]
            ),
            "disposition": obj["disposition"] if obj else None,
            "parent_untouched": _claim_head_state(store, "c2") == "active",
        }
    finally:
        store.close()


def run_i5(workdir: Optional[str] = None) -> dict:
    """Run every measured branch scenario on disposable stores."""
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="v45_i5_")

    d09 = _s_d09(workdir)
    moved = _s_moved_parent(workdir)
    supp = _s_d10_suppressed(workdir)
    purged = _s_d10_purged(workdir)
    held = _s_held_rebase(workdir)
    abandon = _s_abandon_diff(workdir)

    d09_met = bool(
        d09["isolated_before_apply"]
        and d09["review_gate_code"] == ErrorCode.INVALID_TRANSITION.value
        and d09["review_opened"] and d09["forecast_recorded"]
        and d09["applied"]
        and d09["head_state_after"] == "archived"
        and d09["recall_dropped_after_apply"]
        and d09["idempotent_replay"]
    )
    moved_met = bool(
        moved["moved_submit_code"] == ErrorCode.STALE_PROPOSAL.value
        and moved["moved_apply_code"] == ErrorCode.STALE_PROPOSAL.value
    )
    supp_met = bool(
        supp["apply_code"] == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED.value
        and supp["branch_state_after"] == "live"
        and supp["no_restore"]
    )
    purged_met = bool(
        purged["apply_refused_typed"]
        and purged["claim_revisions_states"] == ["active", "erased"]
        and purged["no_resurrect_text"]
        and purged["no_apply_receipt"]
        and purged["branch_disposition"] == "erased"
        and purged["tombstones_carry_no_text"]
    )
    held_met = bool(
        held["held_in_invalidation_tx"] == "held"
        and held["marked"] == 1
        and held["held_durable"] == "held"
        and held["submit_code"] == ErrorCode.INVALID_TRANSITION.value
        and held["apply_code"] == ErrorCode.INVALID_TRANSITION.value
        and held["rebase_state"] == "live"
    )
    abandon_met = bool(
        abandon["diff_initial_health"] == ["ok"]
        and abandon["diff_moved_health"] == "moved"
        and abandon["abandon_state"] == "abandoned"
        and abandon["apply_code"] is not None
        and abandon["audit_preserved"]
        and abandon["disposition"] == "archived"
        and abandon["parent_untouched"]
    )

    report = {
        "experiment": "i5_branch_apply",
        "spec": {
            "requirements": [
                "V45-07.01", "V45-07.02", "V45-07.03", "V45-07.04",
                "V45-12.03",
            ],
            "acceptance": ["D09", "D10"],
        },
        "sample_size": {
            "scenarios": 6,
            "note": "each scenario in its own disposable Store; every"
                    " gate measured from the typed error the real path"
                    " returns and from committed post-state",
        },
        "scenarios": {
            "d09_isolation_reviewed_apply": d09,
            "moved_parent_fencing": moved,
            "d10_suppressed_parent": supp,
            "d10_purged_parent": purged,
            "held_by_invalidation_then_rebase": held,
            "abandon_and_diff": abandon,
        },
        "checks": {
            "d09": d09_met,
            "moved_parent": moved_met,
            "d10_suppressed": supp_met,
            "d10_purged": purged_met,
            "held_rebase": held_met,
            "abandon_diff": abandon_met,
        },
    }
    report["met"] = all(report["checks"].values())
    return report


def _md(report: dict) -> str:
    s = report["scenarios"]
    c = report["checks"]
    lines = [
        "# I5 — reversible memory branches (D09/D10)",
        "",
        f"- D09 isolation: claim visible before apply"
        f" **{s['d09_isolation_reviewed_apply']['isolated_before_apply']}**,"
        f" review gate `{s['d09_isolation_reviewed_apply']['review_gate_code']}`,"
        f" applied → `{s['d09_isolation_reviewed_apply']['head_state_after']}`,"
        f" recall dropped"
        f" **{s['d09_isolation_reviewed_apply']['recall_dropped_after_apply']}**,"
        f" replay `{s['d09_isolation_reviewed_apply']['idempotent_replay']}`"
        f" → met **{c['d09']}**",
        f"- moved parent: submit `{s['moved_parent_fencing']['moved_submit_code']}`,"
        f" apply `{s['moved_parent_fencing']['moved_apply_code']}`"
        f" → met **{c['moved_parent']}**",
        f"- D10 suppressed parent: apply"
        f" `{s['d10_suppressed_parent']['apply_code']}`, branch stays"
        f" `{s['d10_suppressed_parent']['branch_state_after']}`"
        f" → met **{c['d10_suppressed']}**",
        f"- D10 purged parent: apply"
        f" `{s['d10_purged_parent']['apply_code']}`, revisions"
        f" {s['d10_purged_parent']['claim_revisions_states']},"
        f" branch `{s['d10_purged_parent']['branch_disposition']}`,"
        f" no receipt **{s['d10_purged_parent']['no_apply_receipt']}**"
        f" → met **{c['d10_purged']}**",
        f"- held→rebase: held in tx"
        f" `{s['held_by_invalidation_then_rebase']['held_in_invalidation_tx']}`,"
        f" apply `{s['held_by_invalidation_then_rebase']['apply_code']}`,"
        f" rebase `{s['held_by_invalidation_then_rebase']['rebase_state']}`"
        f" → met **{c['held_rebase']}**",
        f"- abandon/diff: diff"
        f" {s['abandon_and_diff']['diff_moved_health']}, apply"
        f" `{s['abandon_and_diff']['apply_code']}`, disposition"
        f" `{s['abandon_and_diff']['disposition']}`"
        f" → met **{c['abandon_diff']}**",
        "",
        f"**met: {report['met']}**",
    ]
    return "\n".join(lines) + "\n"


def write_reports(report: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "i5_branch_report.json")
    mpath = os.path.join(out_dir, "i5_branch_report.md")
    with open(jpath, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=False)
    with open(mpath, "w") as fh:
        fh.write(_md(report))
    return {"json": jpath, "md": mpath}


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="I5 reversible memory branches (D09/D10)"
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args(argv)
    report = run_i5(workdir=args.workdir)
    if args.out:
        paths = write_reports(report, args.out)
        print(f"wrote {paths['json']}")
    else:
        print(json.dumps(report["checks"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
