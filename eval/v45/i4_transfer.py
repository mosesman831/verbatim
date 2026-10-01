"""I4 environment- and counterexample-qualified transfer vs
positive-only reuse (SPEC_V4_5 §06; D07, D08; V45-06.01–06.05).

Paired arms over an identical compiled-procedure corpus, each in its
own disposable ``Store`` — procedures are compiled through the real
``compile_episode`` producer path, delivered through
``procedures.deliver_procedure`` in ``positive_only`` /
``failure_aware`` modes, and success is measured by
``procedures.transfer_success`` against real outcome envelopes — the
same host-attested checker-receipt rules episode outcomes use.

Per task the harness plays the **host**: it delivers the card per the
arm's mode, then writes the task's outcome envelope from a *declared*
ground truth (recorded in ``host_ground_truth``):

- ``in_env`` — delivery environment matches the compiled environment;
  the reuse succeeds (attested ``success``).
- ``heldout`` — environment mismatches (platform flip). Positive-only
  ships the card and the reuse *fails* (attested ``failure`` → negative
  transfer). Failure-aware blocks, the host runs the task without
  memory, and the no-memory path *succeeds* — the V45-06.03 case where
  no-memory wins and unqualified reuse loses.
- ``unknown_env`` — no environment supplied; failure-aware delivers
  loudly qualified, the reuse still succeeds.
- ``exposed_only`` / ``self_report`` — D08 probes: delivery with no
  outcome envelope, and delivery with an agent-reported "success"
  envelope — neither may count as transfer success.

Measured, not asserted: deliveries, blocks, qualified deliveries,
counterexample attachments (with evidence refs), per-task
``transfer_success`` verdicts, negative transfer (delivered ∧ attested
failure), and task-level success (attested success whatever the path).

CLI: ``python -m eval.v45.i4_transfer --out <dir>`` →
``i4_transfer_report.json`` + ``.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from typing import Any, Optional

from verbatim.core.types import json_dumps, new_id, safe_json_loads
from verbatim.procedures import compile_episode
from verbatim.procedures.transfer import (
    TRANSFER_POLICY_ID,
    deliver_procedure,
    transfer_success,
)
from verbatim.storage.store import Store

from . import corpus


SCOPE = "scope:transfer_eval"

#: The compiled-from environment and its held-out flip.
ENV_ROWS = {
    "repo_id": "repo-1",
    "repo_revision": "abc123",
    "platform": "linux",
    "runtime.python": "3.12",
    "tool.pytest": "8.0",
}
HELD_OUT_ENV = dict(ENV_ROWS, platform="darwin")

PROCEDURES = 6

#: Host ground truth per scenario — declared, never measured.
#: ``delivered`` / ``no_memory`` give the checker outcome the host
#: writes for each path (None → no outcome envelope).
GROUND_TRUTH = {
    "in_env": {"delivered": "success", "no_memory": "success"},
    "heldout": {"delivered": "failure", "no_memory": "success"},
    "unknown_env": {"delivered": "success", "no_memory": "success"},
    "exposed_only": {"delivered": None, "no_memory": "success"},
    "self_report": {"delivered": "success", "no_memory": "success"},
}
SCENARIOS = tuple(GROUND_TRUTH)


def _envelope(
    conn: sqlite3.Connection,
    store: Store,
    scope_id: str,
    kind: str,
    metadata: Optional[dict] = None,
    task_id: str = "",
) -> str:
    eid, sid = new_id(), new_id()
    conn.execute(
        "INSERT INTO sources (source_id, origin, source_kind, scope_id,"
        " created_us) VALUES (?, 'test', 'tool_output', ?, 0)",
        (sid, scope_id),
    )
    body = b"env-payload"
    conn.execute(
        "INSERT INTO source_revisions (source_id, revision, payload,"
        " payload_hmac, event_us, captured_us, provenance)"
        " VALUES (?, 1, ?, ?, 0, 0, 'approved_tool')",
        (sid, body, store.hmac(body)),
    )
    conn.execute(
        "INSERT INTO source_envelopes"
        " (envelope_id, source_id, revision, scope_id, envelope_kind,"
        "  actor_principal, event_us, receipt_us, trust_class, task_id,"
        "  metadata_json)"
        " VALUES (?,?,?,?,?,'agent',0,0,'host_observed',?,?)",
        (eid, sid, 1, scope_id, kind, task_id, json_dumps(metadata or {})),
    )
    return eid


def _seed_episode(
    conn: sqlite3.Connection,
    store: Store,
    scope_id: str,
    *,
    errors: Optional[list[dict]] = None,
    env: Optional[dict] = None,
) -> str:
    episode_id = f"ep:{new_id()[:12]}"
    traj_id = f"traj:{new_id()[:12]}"
    conn.execute(
        "INSERT INTO episodes (episode_id, scope_id, revision,"
        " host_task_id, kind, label, recorded_from, recorded_until,"
        " boundary_rule, outcome)"
        " VALUES (?,?,1,'task-1','task','fix-tests',1,99,'task_id','success')",
        (episode_id, scope_id),
    )
    conn.execute(
        "INSERT INTO trajectories (trajectory_id, scope_id, task_id,"
        " boundary_rule, created_event, completed_event, metadata_json)"
        " VALUES (?,?,'task-1','task_id',1,2,?)",
        (traj_id, scope_id, json_dumps({"goal_class": "fix-tests"})),
    )
    ops = [
        {"tool": "read_file", "args": {"path": "src/a.py"}},
        {"tool": "apply_patch",
         "args": {"path": "src/a.py", "patch": "@@ -1 +1 @@\n-x\n+y"}},
        {"tool": "run_check",
         "args": {"argv": ["pytest", "tests/test_a.py", "-x"]}},
    ]
    step_ids: list[str] = []
    for i, op in enumerate(ops):
        step_id = f"step:{new_id()[:12]}"
        eid = _envelope(
            conn, store, scope_id, "tool_call",
            metadata={"tool": op["tool"], "args": op["args"]},
        )
        conn.execute(
            "INSERT INTO trajectory_steps (step_id, trajectory_id,"
            " scope_id, ord, action_envelope_id) VALUES (?,?,?,?,?)",
            (step_id, traj_id, scope_id, i + 1, eid),
        )
        step_ids.append(step_id)
        for j, err in enumerate(errors or []):
            oid = _envelope(conn, store, scope_id, "error", metadata=err)
            conn.execute(
                "INSERT INTO step_observations (step_id, envelope_id, ord)"
                " VALUES (?,?,?)",
                (step_id, oid, j),
            )
    checker_env = _envelope(
        conn, store, scope_id, "test_result",
        metadata={"checker": "pytest", "outcome": "success",
                  "exit_code": 0, "completed": True},
    )
    for i, step_id in enumerate(step_ids):
        conn.execute(
            "INSERT INTO transitions (transition_id, episode_id, scope_id,"
            " ord, action_step_id, checker_ref, edge, created_event)"
            " VALUES (?,?,?,?,?,?,'observed_after',?)",
            (
                f"tr:{new_id()[:12]}", episode_id, scope_id, i + 1,
                step_id,
                checker_env if i == len(step_ids) - 1 else None,
                i,
            ),
        )
    if env:
        for k, v in env.items():
            conn.execute(
                "INSERT OR REPLACE INTO environment_state"
                " (scope_id, key, value, observed_us, volatile)"
                " VALUES (?,?,?,0,1)",
                (scope_id, k, v),
            )
    return episode_id


def _attested_outcome(
    conn: sqlite3.Connection,
    store: Store,
    scope_id: str,
    task_id: str,
    *,
    outcome: str,
    agent_report: bool = False,
) -> str:
    meta: dict[str, Any] = {
        "outcome": outcome,
        "exit_code": 0 if outcome == "success" else 1,
        "completed": True,
        "checker_receipt": {
            "checker_id": "pytest",
            "host_attested": not agent_report,
            "invocation_id": f"inv-{task_id}",
            "task_id": task_id,
            "scope_id": scope_id,
            "exit_code": 0 if outcome == "success" else 1,
            "completed": True,
        },
    }
    if agent_report:
        meta["checker_receipt"]["agent_report"] = True
    return _envelope(
        conn, store, scope_id, "test_result",
        metadata=meta, task_id=task_id,
    )


def _bindings_for(conn: sqlite3.Connection, pid: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT bindings_json FROM procedures WHERE procedure_id = ?",
        (pid,),
    ).fetchone()
    return {
        b["name"]: b.get("observed_value")
        for b in (safe_json_loads(row[0]) or [])
        if isinstance(b, dict) and b.get("required") and b.get("name")
    }


def seed_corpus(store: Store, *, procedures: int = PROCEDURES) -> dict:
    """Compile ``procedures`` real procedures in ENV_ROWS — every other
    one carries a recorded failure mode (a counterexample that must
    ride along on failure-aware deliveries)."""
    pids: list[str] = []
    with store.tx() as conn:
        corpus.seed_scope(conn, SCOPE)
        for i in range(procedures):
            errors = (
                [{"kind": "test_failure",
                  "summary": f"pytest failed on darwin: assert {i}",
                  "tool": "run_check"}]
                if i % 2 == 0 else None
            )
            ep = _seed_episode(
                conn, store, SCOPE, errors=errors, env=dict(ENV_ROWS)
            )
            result = compile_episode(conn, ep, hmac_fn=store.hmac)
            if not result.procedure_id:
                raise RuntimeError(f"compile gated: {result}")
            conn.execute(
                "UPDATE procedures SET state='active'"
                " WHERE procedure_id = ?",
                (result.procedure_id,),
            )
            pids.append(result.procedure_id)
    return {"scope_id": SCOPE, "procedure_ids": pids}


def _env_for(scenario: str) -> Any:
    if scenario == "in_env":
        return dict(ENV_ROWS)
    if scenario == "heldout":
        return dict(HELD_OUT_ENV)
    return None  # unknown_env and the D08 probes deliver unscoped-env


def _run_arm(
    workdir: str,
    name: str,
    *,
    mode: str,
    procedures: int,
) -> dict:
    store = corpus.make_store(workdir, name)
    try:
        seeded = seed_corpus(store, procedures=procedures)
        sid = seeded["scope_id"]
        rows: list[dict] = []
        with store.tx() as conn:
            for i, pid in enumerate(seeded["procedure_ids"]):
                bindings = _bindings_for(conn, pid)
                for scenario in SCENARIOS:
                    task_id = f"proc{i}:{scenario}"
                    d = deliver_procedure(
                        conn, pid, task_id=task_id,
                        environment=_env_for(scenario),
                        bindings=bindings,
                        mode=mode,
                        experiment_id="i4",
                        arm=mode,
                    )
                    truth = GROUND_TRUTH[scenario]
                    outcome = (
                        truth["delivered"] if d.deliverable
                        else truth["no_memory"]
                    )
                    if outcome is not None:
                        _attested_outcome(
                            conn, store, sid, task_id,
                            outcome=outcome,
                            agent_report=(scenario == "self_report"),
                        )
                    res = transfer_success(conn, pid, task_id)
                    rows.append({
                        "task_id": task_id,
                        "scenario": scenario,
                        "deliverable": d.deliverable,
                        "disposition": d.disposition,
                        "environment_match": d.environment_match,
                        "counterexamples": len(d.counterexamples),
                        "counterexample_evidence_refs": sum(
                            len(c.evidence_refs)
                            for c in d.counterexamples
                        ),
                        "exposure_recorded": d.exposure_id is not None,
                        "host_outcome": outcome,
                        "delivered": res["delivered"],
                        "attested_outcome": res["attested_outcome"],
                        "transfer_success": res["transfer_success"],
                        "negative_transfer": bool(
                            res["delivered"]
                            and res["attested_outcome"] == "failure"
                        ),
                        "task_success": res["attested_outcome"] == "success",
                    })
        totals = {
            "deliveries": sum(1 for r in rows if r["deliverable"]),
            "blocked": sum(
                1 for r in rows if r["disposition"] == "blocked"
            ),
            "qualified": sum(
                1 for r in rows if r["disposition"] == "qualified"
            ),
            "transfer_success": sum(
                1 for r in rows if r["transfer_success"]
            ),
            "negative_transfer": sum(
                1 for r in rows if r["negative_transfer"]
            ),
            "task_success": sum(1 for r in rows if r["task_success"]),
            "counterexample_attachments": sum(
                1 for r in rows if r["counterexamples"]
            ),
            "counterexample_evidence_refs": sum(
                r["counterexample_evidence_refs"] for r in rows
            ),
        }
        by_scenario = {
            sc: {
                "deliveries": sum(
                    1 for r in rows
                    if r["scenario"] == sc and r["deliverable"]
                ),
                "blocked": sum(
                    1 for r in rows
                    if r["scenario"] == sc
                    and r["disposition"] == "blocked"
                ),
                "transfer_success": sum(
                    1 for r in rows
                    if r["scenario"] == sc and r["transfer_success"]
                ),
                "negative_transfer": sum(
                    1 for r in rows
                    if r["scenario"] == sc and r["negative_transfer"]
                ),
                "task_success": sum(
                    1 for r in rows
                    if r["scenario"] == sc and r["task_success"]
                ),
            }
            for sc in SCENARIOS
        }
        return {"mode": mode, "rows": rows, "totals": totals,
                "by_scenario": by_scenario}
    finally:
        store.close()


def run_i4(
    procedures: int = PROCEDURES,
    workdir: Optional[str] = None,
) -> dict:
    """Run both delivery arms over identical compiled corpora."""
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="v45_i4_")
    positive = _run_arm(
        os.path.join(workdir, "arm_positive"), "positive.db",
        mode="positive_only", procedures=procedures,
    )
    aware = _run_arm(
        os.path.join(workdir, "arm_aware"), "aware.db",
        mode="failure_aware", procedures=procedures,
    )

    def _sc(arm: dict, scenario: str, key: str) -> int:
        return arm["by_scenario"][scenario][key]

    d07 = {
        "heldout_negative_transfer": {
            "positive_only": _sc(positive, "heldout", "negative_transfer"),
            "failure_aware": _sc(aware, "heldout", "negative_transfer"),
        },
        "heldout_blocked": _sc(aware, "heldout", "blocked"),
        "heldout_task_success_aware": _sc(aware, "heldout", "task_success"),
        "inenv_success_preserved": bool(
            _sc(aware, "in_env", "transfer_success")
            == _sc(positive, "in_env", "transfer_success")
            == procedures
        ),
        "measured": (
            f"held-out negative transfer positive_only="
            f"{_sc(positive, 'heldout', 'negative_transfer')}"
            f" vs failure_aware="
            f"{_sc(aware, 'heldout', 'negative_transfer')}"
            f" (blocked {_sc(aware, 'heldout', 'blocked')},"
            f" no-memory task success"
            f" {_sc(aware, 'heldout', 'task_success')})"
        ),
    }
    aware_rows = aware["rows"]
    d08 = {
        "exposed_only_not_counted": sum(
            1 for r in aware_rows
            if r["scenario"] == "exposed_only"
            and r["delivered"] and not r["transfer_success"]
        ),
        "self_report_not_counted": sum(
            1 for r in aware_rows
            if r["scenario"] == "self_report"
            and r["delivered"]
            and r["attested_outcome"] == "none"
            and not r["transfer_success"]
        ),
        "cases": procedures,
        "measured": (
            "delivery without an outcome envelope and delivery with an"
            " agent-reported envelope both report transfer_success"
            " False on every case"
        ),
    }
    report = {
        "experiment": "i4_qualified_transfer",
        "spec": {
            "requirements": [
                "V45-06.01", "V45-06.02", "V45-06.03", "V45-06.04",
                "V45-06.05",
            ],
            "acceptance": ["D07", "D08"],
            "policy_id": TRANSFER_POLICY_ID,
        },
        "sample_size": {
            "procedures_per_arm": procedures,
            "scenarios": len(SCENARIOS),
            "tasks_per_arm": procedures * len(SCENARIOS),
            "arms": 2,
            "note": (
                "identical compiled corpus per arm in disposable"
                " stores; procedures compile through the real"
                " compile_episode path, deliveries through"
                " deliver_procedure, success through transfer_success"
            ),
        },
        "environment": {
            "compiled": dict(ENV_ROWS),
            "held_out": dict(HELD_OUT_ENV),
        },
        "host_ground_truth": GROUND_TRUTH,
        "definitions": {
            "negative_transfer": (
                "the procedure was delivered for the task AND the"
                " task's host-attested outcome resolved to failure"
            ),
            "transfer_success": (
                "transfer_success(): a procedure_exposures row binds"
                " the procedure to the task AND a host-attested"
                " checker receipt resolves success — exposures, agent"
                " reports, and hypothetical trials never count"
            ),
            "task_success": (
                "the task's attested outcome is success regardless of"
                " the path — includes no-memory completions after a"
                " blocked delivery"
            ),
            "qualified": (
                "delivered loudly qualified — unknown applicability"
                " verdict or attached counterexamples"
            ),
        },
        "arms": {
            "positive_only": positive,
            "failure_aware": aware,
        },
        "totals": {
            "positive_only": positive["totals"],
            "failure_aware": aware["totals"],
        },
        "d07": d07,
        "d08": d08,
    }
    report["met"] = bool(
        d07["heldout_negative_transfer"]["positive_only"] > 0
        and d07["heldout_negative_transfer"]["failure_aware"] == 0
        and d07["heldout_blocked"] == procedures
        and d07["heldout_task_success_aware"] == procedures
        and d07["inenv_success_preserved"]
        and d08["exposed_only_not_counted"] == procedures
        and d08["self_report_not_counted"] == procedures
        and aware["totals"]["counterexample_attachments"] > 0
        and aware["totals"]["counterexample_evidence_refs"] > 0
    )
    return report


def _md(report: dict) -> str:
    t = report["totals"]
    d07, d08 = report["d07"], report["d08"]
    n = report["sample_size"]["procedures_per_arm"]
    lines = [
        "# I4 — environment-/counterexample-qualified transfer (D07/D08)",
        "",
        f"- corpus: {n} compiled procedures ×"
        f" {report['sample_size']['scenarios']} scenarios per arm",
        f"- held-out negative transfer: positive_only"
        f" **{d07['heldout_negative_transfer']['positive_only']}**"
        f" vs failure_aware"
        f" **{d07['heldout_negative_transfer']['failure_aware']}**",
        f"- held-out blocked on failure-aware: {d07['heldout_blocked']}"
        f" (no-memory task success {d07['heldout_task_success_aware']})",
        f"- in-env success preserved both arms:"
        f" **{d07['inenv_success_preserved']}**",
        f"- D08: exposure-only not counted"
        f" {d08['exposed_only_not_counted']}/{n}, self-report not"
        f" counted {d08['self_report_not_counted']}/{n}",
        f"- counterexample attachments on failure-aware:"
        f" {t['failure_aware']['counterexample_attachments']}"
        f" deliveries /"
        f" {t['failure_aware']['counterexample_evidence_refs']}"
        " evidence refs",
        f"- met: **{report['met']}**",
        "",
        "| scenario | arm | delivered | blocked | transfer_success"
        " | negative_transfer | task_success |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for sc in SCENARIOS:
        for arm in ("positive_only", "failure_aware"):
            s = report["arms"][arm]["by_scenario"][sc]
            lines.append(
                f"| {sc} | {arm} | {s['deliveries']} | {s['blocked']}"
                f" | {s['transfer_success']} | {s['negative_transfer']}"
                f" | {s['task_success']} |"
            )
    return "\n".join(lines) + "\n"


def write_reports(report: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "i4_transfer_report.json")
    mpath = os.path.join(out_dir, "i4_transfer_report.md")
    with open(jpath, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=False)
    with open(mpath, "w") as fh:
        fh.write(_md(report))
    return {"json": jpath, "md": mpath}


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="I4 qualified procedure transfer (D07/D08)"
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--procedures", type=int, default=PROCEDURES)
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args(argv)
    report = run_i4(procedures=args.procedures, workdir=args.workdir)
    if args.out:
        paths = write_reports(report, args.out)
        print(f"wrote {paths['json']}")
    else:
        print(json.dumps(report["totals"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
