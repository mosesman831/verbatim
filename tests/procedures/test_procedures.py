"""``coding_rules_v1`` — the bounded procedure compiler (SPEC_V3 §21–§22).

Covers the reference producer contract: completed coding episodes compile
into ``candidate`` procedures with abstracted operations, extracted
bindings, environment fingerprints, failure modes, signature dedup,
derivation edges — then the review ladder, three-valued applicability,
contrastive-refinement hypotheses, exposure accounting, and the durable
job handlers. Everything is seeded directly through the table contracts;
the episode builder is a separate producer.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

import pytest

from verbatim.config import VerbatimConfig
from verbatim.core.types import (
    ErrorCode,
    JobKind,
    VerbatimError,
    json_dumps,
    new_id,
    safe_json_loads,
)
from verbatim.core.types_v3 import (
    ApplicabilityVerdict,
    CompilationStatus,
    EnvironmentFingerprint,
    OperationClass,
)
from verbatim.ingest import Ingester
from verbatim.procedures import (
    ProcedureCompiler,
    activate,
    check_applicability,
    classify,
    compile_episode,
    compute_signature,
    exposures_for,
    find_by_signature,
    index_procedure,
    index_scope,
    record_exposure,
    refine_procedure,
    requires_paired_evidence,
    reuse_stats,
    review,
    suspend,
)
from verbatim.procedures.compiler import _environment_for
from verbatim.procedures.signatures import COMPILER_MANIFEST
from verbatim.storage.store import Store

SCOPE = "scope:proc"
ENV_ROWS = {
    "repo_id": "repo-1",
    "repo_revision": "abc123",
    "platform": "linux",
    "runtime.python": "3.12",
    "tool.pytest": "8.0",
}
ENV = EnvironmentFingerprint(
    repo_id="repo-1",
    repo_revision="abc123",
    runtime_versions=(("python", "3.12"),),
    tool_schema_versions=(("pytest", "8.0"),),
    platform="linux",
)


@pytest.fixture()
def store(tmp_path):
    s = Store.create(str(tmp_path / "proc.db"))
    yield s
    s.close()


@pytest.fixture()
def scope_id(store):
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES (?, 'prof', 'owner')",
            (SCOPE,),
        )
    return SCOPE


# ---------------------------------------------------------------------------
# seeding helpers (table contracts only)
# ---------------------------------------------------------------------------


def _envelope(
    conn: sqlite3.Connection,
    store: Store,
    scope_id: str,
    kind: str,
    metadata: Optional[dict] = None,
    payload: Optional[dict] = None,
    task_id: str = "",
) -> str:
    """One source + revision + envelope row; returns envelope_id."""
    eid, sid = new_id(), new_id()
    body = (
        json_dumps(payload).encode("utf-8")
        if payload is not None
        else b"raw-bytes"
    )
    conn.execute(
        "INSERT INTO sources (source_id, origin, source_kind, scope_id,"
        " created_us) VALUES (?, 'test', 'tool_output', ?, 0)",
        (sid, scope_id),
    )
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
    ops: list[dict[str, Any]],
    *,
    completed: bool = True,
    outcome: str = "success",
    label: str = "fix-tests",
    goal_class: Optional[str] = "fix-tests",
    checker: bool = True,
    checker_outcome: str = "success",
    checker_name: str = "pytest",
    env: Optional[dict] = None,
    insert_order_shuffled: bool = False,
) -> dict[str, Any]:
    """Seed a coding episode: episode + trajectory + steps + tool-call
    envelopes + transitions (+ optional checker receipt + env state)."""
    episode_id = f"ep:{new_id()[:12]}"
    traj_id = f"traj:{new_id()[:12]}"
    conn.execute(
        "INSERT INTO episodes (episode_id, scope_id, revision,"
        " host_task_id, kind, label, recorded_from, recorded_until,"
        " boundary_rule, outcome)"
        " VALUES (?,?,1,'task-1','task',?,1,?,?,?)",
        (
            episode_id, scope_id, label,
            99 if completed else None,
            "task_id", outcome,
        ),
    )
    conn.execute(
        "INSERT INTO trajectories (trajectory_id, scope_id, task_id,"
        " boundary_rule, created_event, completed_event, metadata_json)"
        " VALUES (?,?,'task-1','task_id',1,2,?)",
        (traj_id, scope_id, json_dumps(
            {"goal_class": goal_class} if goal_class else {}
        )),
    )
    env_ids: list[str] = []
    transition_ids: list[str] = []
    steps: list[str] = []
    for i, op in enumerate(ops):
        eid = _envelope(
            conn, store, scope_id, "tool_call",
            metadata={"tool": op["tool"], "args": op.get("args", {})},
        )
        env_ids.append(eid)
        step_id = f"step:{new_id()[:12]}"
        conn.execute(
            "INSERT INTO trajectory_steps (step_id, trajectory_id,"
            " scope_id, ord, action_envelope_id)"
            " VALUES (?,?,?,?,?)",
            (step_id, traj_id, scope_id, i + 1, eid),
        )
        steps.append(step_id)
        for j, err in enumerate(op.get("errors") or []):
            oid = _envelope(conn, store, scope_id, "error", metadata=err)
            conn.execute(
                "INSERT INTO step_observations (step_id, envelope_id, ord)"
                " VALUES (?,?,?)",
                (step_id, oid, j),
            )
            env_ids.append(oid)
        for j, rec in enumerate(op.get("recoveries") or []):
            oid = _envelope(conn, store, scope_id, "recovery", metadata=rec)
            conn.execute(
                "INSERT INTO step_observations (step_id, envelope_id, ord)"
                " VALUES (?,?,?)",
                (step_id, oid, 10 + j),
            )
            env_ids.append(oid)
    checker_env = None
    if checker:
        checker_env = _envelope(
            conn, store, scope_id, "test_result",
            metadata={
                "checker": checker_name,
                "outcome": checker_outcome,
                "selected_tests": ["tests/test_a.py"],
                "exit_code": 0 if checker_outcome == "success" else 1,
                "completed": True,
            },
        )
        env_ids.append(checker_env)
    ords = list(range(len(ops)))
    if insert_order_shuffled:
        ords = list(reversed(ords))
    for pos, i in enumerate(ords):
        tid = f"tr:{new_id()[:12]}"
        last = pos == len(ords) - 1
        conn.execute(
            "INSERT INTO transitions (transition_id, episode_id, scope_id,"
            " ord, action_step_id, checker_ref, edge, created_event)"
            " VALUES (?,?,?,?,?,?,'observed_after',?)",
            (
                tid, episode_id, scope_id, i + 1, steps[i],
                checker_env if (checker and last) else None, pos,
            ),
        )
        transition_ids.append(tid)
    if env is not None:
        for k, v in env.items():
            conn.execute(
                "INSERT OR REPLACE INTO environment_state"
                " (scope_id, key, value, observed_us, volatile)"
                " VALUES (?,?,?,0,1)",
                (scope_id, k, v),
            )
    return {
        "episode_id": episode_id,
        "trajectory_id": traj_id,
        "envelope_ids": env_ids,
        "transition_ids": transition_ids,
        "step_ids": steps,
        "checker_env": checker_env,
    }


GOOD_OPS = [
    {"tool": "read_file", "args": {"path": "src/a.py"}},
    {"tool": "grep", "args": {"pattern": "def solve", "path": "src"}},
    {"tool": "apply_patch",
     "args": {"path": "src/a.py", "patch": "@@ -1 +1 @@\n-x\n+y"}},
    {"tool": "run_check",
     "args": {"argv": ["pytest", "tests/test_a.py", "-x"]}},
]


def _seed_good(conn, store, scope_id, **kw):
    return _seed_episode(
        conn, store, scope_id, GOOD_OPS, env=dict(ENV_ROWS), **kw
    )


def _proc_row(conn, pid):
    """Fetch a procedures row on the caller's connection (sees the open tx)."""
    cur = conn.execute(
        "SELECT * FROM procedures WHERE procedure_id = ?", (pid,)
    )
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    return dict(zip(cols, row)) if row else None


def _j(row, key, default=None):
    v = row.get(key)
    if v is None:
        return default
    return safe_json_loads(v)


# ---------------------------------------------------------------------------
# classify()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool,args,want",
    [
        ("read_file", {"path": "a.py"}, OperationClass.INSPECT_FILE),
        ("Cat", {"path": "a.py"}, OperationClass.INSPECT_FILE),
        ("fs.read_file", {"path": "a.py"}, OperationClass.INSPECT_FILE),
        ("mcp/fs/read_file", {"path": "a.py"}, OperationClass.INSPECT_FILE),
        ("grep", {"pattern": "x"}, OperationClass.SEARCH_REPO),
        ("ripgrep", {"pattern": "x"}, OperationClass.SEARCH_REPO),
        ("find", {}, OperationClass.SEARCH_REPO),
        ("apply_patch", {"path": "a"}, OperationClass.APPLY_PATCH),
        ("edit_file", {"path": "a"}, OperationClass.APPLY_PATCH),
        ("write", {"path": "a"}, OperationClass.APPLY_PATCH),
        ("run_check", {"argv": ["pytest", "t.py"]},
         OperationClass.RUN_CHECK),
        ("run_check", {"argv": ["python", "-m", "pytest", "t.py"]},
         OperationClass.RUN_CHECK),
        ("run_check", {"argv": ["npm", "test"]}, OperationClass.RUN_CHECK),
        ("run_check", {"argv": ["pnpm", "test"]}, OperationClass.RUN_CHECK),
        ("run_check", {"argv": ["make", "check"]}, OperationClass.RUN_CHECK),
        ("pytest", {"argv": ["pytest"]}, OperationClass.RUN_CHECK),
        # shell-form commands demote to opaque
        ("run_check", {"command": "pytest t.py"}, OperationClass.OPAQUE),
        ("run_check", {"cmd": "make check"}, OperationClass.OPAQUE),
        ("run_check", {"script": "sh build.sh"}, OperationClass.OPAQUE),
        ("run_check", {"argv": ["python", "script.py"]},
         OperationClass.OPAQUE),
        ("run_check", {"argv": ["npm", "install"]}, OperationClass.OPAQUE),
        ("run_check", {}, OperationClass.OPAQUE),
        ("shell", {"command": "ls"}, OperationClass.OPAQUE),
        ("browser_click", {}, OperationClass.OPAQUE),
        ("deploy", {}, OperationClass.OPAQUE),
        ("sudo", {"command": "x"}, OperationClass.OPAQUE),
        ("", {}, OperationClass.OPAQUE),
        (None, {}, OperationClass.OPAQUE),
        ("totally_unknown_tool", {}, OperationClass.OPAQUE),
    ],
)
def test_classify(tool, args, want):
    assert classify(tool, args) == want


# ---------------------------------------------------------------------------
# compile_episode — gates
# ---------------------------------------------------------------------------


def test_compile_happy_path(store, scope_id):
    """§22.01/22.13: ops+bindings+checker → candidate with edges."""
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        result = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac)
        assert result.status == CompilationStatus.CANDIDATE
        assert result.procedure_id
        pid = result.procedure_id

        row = _proc_row(conn, pid)
        assert row["state"] == "candidate"
        assert row["compiler_manifest"] == COMPILER_MANIFEST
        assert row["risk_class"] == "medium"  # apply_patch present
        ops = _j(row, "operations_json")
        assert [o["op_class"] for o in ops] == [
            "inspect_file", "search_repo", "apply_patch", "run_check",
        ]
        # Ops keep transition ord order; templates hold binding slots only.
        assert [o["ord"] for o in ops] == [1, 2, 3, 4]
        for o in ops:
            for v in o["param_template"].values():
                assert "$binding" in v
        bindings = _j(row, "bindings_json")
        by_kind = {b["kind"] for b in bindings}
        assert {"repo_path", "test_target", "revision"} <= by_kind
        values = {b["observed_value"] for b in bindings}
        assert "src/a.py" in values
        assert any("tests/test_a.py" in (v or "") for v in values)
        assert "abc123" in values
        intent = _j(row, "intent_signature_json")
        assert intent["goal_class"] == "fix-tests"
        assert intent["check_kind"] == "pytest"
        verification = _j(row, "verification_json")
        assert verification[0]["checker"] == "pytest"
        assert verification[0]["outcome"] == "success"
        fm = _j(row, "failure_modes_json")
        assert fm["status"] == "no_failure_evidence"  # V3-21.05
        assert fm["modes"] == []
        prov = _j(row, "provenance_json")
        assert prov["source_episodes"] == [seed["episode_id"]]
        assert prov["compiler"] == COMPILER_MANIFEST
        env = _j(row, "environment_json")
        assert env["repo_id"] == "repo-1"
        assert env["platform"] == "linux"
        assert row["evidence_family_id"]
        # V3-14.06: the compiled procedure is a screened write channel — a
        # real label is attached at compile time, and benign content earns
        # no_findings with no quarantine hold.
        assert row["security_label_id"]
        lab = conn.execute(
            "SELECT attack_risk, review_state FROM security_labels"
            " WHERE label_id = ?",
            (row["security_label_id"],),
        ).fetchone()
        assert lab is not None
        assert lab[0] == "no_findings"
        assert conn.execute(
            "SELECT COUNT(*) FROM quarantine WHERE object_kind='procedure'"
            " AND object_id = ?",
            (pid,),
        ).fetchone()[0] == 0

        # Signature row + derivations edges.
        sig = find_by_signature(
            conn, scope_id, _sig_of(conn, scope_id, pid))
        assert sig is not None
        assert sig["procedure_id"] == pid
        edges = conn.execute(
            "SELECT parent_kind, parent_id FROM derivations"
            " WHERE child_kind='procedure' AND child_id=?",
            (pid,),
        ).fetchall()
        kinds = {k for k, _ in edges}
        assert {"episode", "transition", "envelope"} <= kinds
        assert (("episode", seed["episode_id"]) in edges)


def _sig_of(conn, scope_id, pid):
    row = conn.execute(
        "SELECT signature_digest FROM procedure_signatures"
        " WHERE procedure_id = ? ORDER BY revision DESC LIMIT 1",
        (pid,),
    ).fetchone()
    return row[0] if row else ""


def test_compile_no_checker_receipt_unsupported(store, scope_id):
    """§22 boundary row: no checker evidence → unsupported, no row."""
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id, checker=False)
        result = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac)
        assert result.status == CompilationStatus.UNSUPPORTED
        assert result.procedure_id is None
        assert result.reason == "no_checker_receipt"
        n = conn.execute("SELECT COUNT(*) FROM procedures").fetchone()[0]
        assert n == 0


def test_compile_unresolvable_checker_ref_unsupported(store, scope_id):
    """A checker_ref pointing at a non-receipt envelope is not evidence."""
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        bogus = _envelope(conn, store, scope_id, "tool_result", metadata={})
        conn.execute(
            "UPDATE transitions SET checker_ref = ? WHERE episode_id = ?"
            " AND checker_ref IS NOT NULL",
            (bogus, seed["episode_id"]),
        )
        result = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac)
        assert result.status == CompilationStatus.UNSUPPORTED
        assert result.reason == "no_checker_receipt"


def test_compile_all_opaque_unsupported(store, scope_id):
    """<2 recognized ops → unsupported (§22 op-mapping row)."""
    ops = [
        {"tool": "shell", "args": {"command": "ls"}},
        {"tool": "browser_click", "args": {}},
        {"tool": "deploy", "args": {}},
    ]
    with store.tx() as conn:
        seed = _seed_episode(conn, store, scope_id, ops)
        result = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac)
        assert result.status == CompilationStatus.UNSUPPORTED
        assert result.reason == "opaque_operations_present"
        n = conn.execute("SELECT COUNT(*) FROM procedures").fetchone()[0]
        assert n == 0


def test_compile_mixed_with_opaque_blocked(store, scope_id):
    """Opaque evidence blocks automatic compilation even with 4 good ops."""
    ops = GOOD_OPS + [{"tool": "mystery_tool", "args": {}}]
    with store.tx() as conn:
        seed = _seed_episode(conn, store, scope_id, ops)
        result = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac)
        assert result.status == CompilationStatus.UNSUPPORTED
        assert result.reason == "opaque_operations_present"


def test_compile_incomplete_episode(store, scope_id):
    """Missing end marker → incomplete, not a procedure (§22 boundary)."""
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id, completed=False)
        result = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac)
        assert result.status == CompilationStatus.INCOMPLETE
        assert result.reason == "episode_not_completed"


def test_compile_episode_not_found(store, scope_id):
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            compile_episode(conn, "ep:nope", hmac_fn=store.hmac)
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_compile_unresolved_action_incomplete(store, scope_id):
    """A transition whose action step cannot be resolved → partial."""
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        conn.execute(
            "INSERT INTO transitions (transition_id, episode_id, scope_id,"
            " ord, action_step_id, edge, created_event)"
            " VALUES ('tr:ghost', ?, ?, 5, 'step:missing',"
            " 'observed_after', 9)",
            (seed["episode_id"], scope_id),
        )
        result = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac)
        assert result.status == CompilationStatus.INCOMPLETE
        assert result.reason == "unresolved_action_evidence"


def test_compile_binding_limit_unsupported(store, scope_id):
    """Over-limit episodes stay evidence-only (§22 candidate row)."""
    ops = [
        {"tool": "read_file", "args": {"path": f"src/f{i}.py"}}
        for i in range(40)
    ] + [{"tool": "run_check", "args": {"argv": ["pytest"]}}]
    with store.tx() as conn:
        seed = _seed_episode(conn, store, scope_id, ops)
        result = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac)
        assert result.status == CompilationStatus.UNSUPPORTED
        assert result.reason == "binding_limit_exceeded"


# ---------------------------------------------------------------------------
# failure modes
# ---------------------------------------------------------------------------


def test_failure_modes_recorded(store, scope_id):
    """§21.05: a failed action step produces a FailureMode with evidence."""
    ops = [dict(GOOD_OPS[0]), dict(GOOD_OPS[1]), dict(GOOD_OPS[2]),
           dict(GOOD_OPS[3])]
    ops[1]["errors"] = [{"error_type": "PermissionDenied",
                         "message": "cannot open src"}]
    with store.tx() as conn:
        seed = _seed_episode(conn, store, scope_id, ops,
                             checker_outcome="failure", outcome="failure")
        result = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac)
        assert result.status == CompilationStatus.CANDIDATE
        row = _proc_row(conn, result.procedure_id)
        fm = _j(row, "failure_modes_json")
        assert fm["status"] == "observed"
        sigs = {m["signature"] for m in fm["modes"]}
        assert len(sigs) == len(fm["modes"])
        # One mode for the step error, one for the checker failure.
        assert len(fm["modes"]) == 2
        for m in fm["modes"]:
            assert m["evidence_refs"]
        descs = " ".join(m["description"] for m in fm["modes"])
        assert "PermissionDenied" in descs
        assert "failure" in descs


def test_failure_modes_none_marked(store, scope_id):
    """Zero contrary evidence → ``no_failure_evidence``, not 'safe'."""
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        pid = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac).procedure_id
        fm = _j(_proc_row(conn, pid), "failure_modes_json")
        assert fm["status"] == "no_failure_evidence"


# ---------------------------------------------------------------------------
# idempotency + signature dedup
# ---------------------------------------------------------------------------


def test_recompile_idempotent(store, scope_id):
    """V3-22.13: same episode → same procedure, no new revision."""
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        r1 = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac)
        r2 = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac)
        assert r1.procedure_id == r2.procedure_id
        assert r2.reason == "already_compiled"
        row = _proc_row(conn, r1.procedure_id)
        assert row["revision"] == 1
        n = conn.execute("SELECT COUNT(*) FROM procedures").fetchone()[0]
        assert n == 1
        n = conn.execute(
            "SELECT COUNT(*) FROM procedure_signatures").fetchone()[0]
        assert n == 1


def test_signature_dedup_bumps_revision(store, scope_id):
    """Same signature+scope from a *different* episode → revision bump."""
    with store.tx() as conn:
        s1 = _seed_good(conn, store, scope_id)
        s2 = _seed_good(conn, store, scope_id)  # same goal + op classes
        r1 = compile_episode(conn, s1["episode_id"], hmac_fn=store.hmac)
        r2 = compile_episode(conn, s2["episode_id"], hmac_fn=store.hmac)
        assert r1.procedure_id == r2.procedure_id
        assert r2.reason == "revision_bump"
        row = _proc_row(conn, r1.procedure_id)
        assert row["revision"] == 2
        prov = _j(row, "provenance_json")
        assert set(prov["source_episodes"]) == {
            s1["episode_id"], s2["episode_id"]}
        sigs = conn.execute(
            "SELECT revision FROM procedure_signatures"
            " WHERE procedure_id = ? ORDER BY revision",
            (r1.procedure_id,),
        ).fetchall()
        assert [r[0] for r in sigs] == [1, 2]


def test_different_goal_new_procedure(store, scope_id):
    """Unknown/different goal class does not join the family (§22)."""
    with store.tx() as conn:
        s1 = _seed_good(conn, store, scope_id, goal_class="fix-tests",
                        label="fix-tests")
        s2 = _seed_episode(conn, store, scope_id, GOOD_OPS,
                           goal_class="refactor-auth", label="refactor-auth",
                           env=dict(ENV_ROWS))
        r1 = compile_episode(conn, s1["episode_id"], hmac_fn=store.hmac)
        r2 = compile_episode(conn, s2["episode_id"], hmac_fn=store.hmac)
        assert r1.procedure_id != r2.procedure_id
        n = conn.execute("SELECT COUNT(*) FROM procedures").fetchone()[0]
        assert n == 2


# ---------------------------------------------------------------------------
# no executable content in templates (V3-21.01, V3-22.14)
# ---------------------------------------------------------------------------


def test_operations_hold_no_shell_text(store, scope_id):
    """Templates carry op_class + binding slots; argv/patch text stays
    inside the episode's evidence envelopes."""
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        pid = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac).procedure_id
        row = _proc_row(conn, pid)
        ops_text = row["operations_json"]
        # Instance values live only in bindings_json, never in the template.
        assert "src/a.py" not in ops_text
        assert "tests/test_a.py" not in ops_text
        assert "pytest tests" not in ops_text
        assert "@@ -1 +1 @@" not in ops_text
        assert '"$binding"' in ops_text
        ops = _j(row, "operations_json")
        for o in ops:
            assert set(o) == {
                "op_class", "tool", "param_template", "ord",
                "evidence_refs",
            }


# ---------------------------------------------------------------------------
# review ladder (V3-22.04)
# ---------------------------------------------------------------------------


def test_review_ladder(store, scope_id):
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        pid = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac).procedure_id

        # candidate → activate refused: v3.0 requires explicit review.
        with pytest.raises(VerbatimError) as exc:
            activate(conn, pid, "op-1")
        assert exc.value.code == ErrorCode.INVALID_TRANSITION

        r = review(conn, pid, "approve", "rev-1", notes="lgtm")
        assert r["state"] == "reviewed"
        r = activate(conn, pid, "op-1")
        assert r["state"] == "active"
        row = _proc_row(conn, pid)
        prov = _j(row, "provenance_json")
        assert prov["reviews"][0]["decision"] == "approve"
        assert prov["reviews"][0]["reviewer_id"] == "rev-1"
        assert prov["activations"][0]["activator_id"] == "op-1"

        r = suspend(conn, pid, "verified applicability violation")
        assert r["state"] == "deprecated"


def test_review_reject_retires(store, scope_id):
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        pid = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac).procedure_id
        r = review(conn, pid, "reject", "rev-1", "not reusable")
        assert r["state"] == "retired"
        with pytest.raises(VerbatimError):
            review(conn, pid, "approve", "rev-2")


def test_activate_without_review_evidence_refused(store, scope_id):
    """A 'reviewed' row lacking review evidence cannot activate."""
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        pid = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac).procedure_id
        conn.execute(
            "UPDATE procedures SET state='reviewed' WHERE procedure_id=?",
            (pid,),
        )
        with pytest.raises(VerbatimError) as exc:
            activate(conn, pid, "op-1")
        assert exc.value.code == ErrorCode.INVALID_TRANSITION


def test_suspend_requires_active(store, scope_id):
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        pid = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac).procedure_id
        with pytest.raises(VerbatimError):
            suspend(conn, pid, "too early")


def test_requires_paired_evidence_contract():
    reqs = requires_paired_evidence()
    assert reqs["explicit_review_required_v3_0"] is True
    assert reqs["auto_promotion_implemented"] is False
    assert reqs["paired_executions_required"] is True


# ---------------------------------------------------------------------------
# applicability (V3-22.05/22.15)
# ---------------------------------------------------------------------------


def _bind_all(row) -> dict[str, str]:
    out = {}
    for b in _j(row, "bindings_json"):
        if b.get("required"):
            out[b["name"]] = "refilled-value"
    return out


def test_applicability_applies(store, scope_id):
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        pid = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac).procedure_id
        row = _proc_row(conn, pid)
        res = check_applicability(conn, pid, ENV, _bind_all(row))
        assert res.verdict == ApplicabilityVerdict.APPLIES


def test_applicability_environment_mismatch(store, scope_id):
    """V3-22.05: drifted fingerprint → does_not_apply + reason."""
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        pid = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac).procedure_id
        row = _proc_row(conn, pid)
        other = EnvironmentFingerprint(
            repo_id="repo-OTHER",
            repo_revision="zzz",
            runtime_versions=(("python", "3.9"),),
            platform="windows",
        )
        res = check_applicability(conn, pid, other, _bind_all(row))
        assert res.verdict == ApplicabilityVerdict.DOES_NOT_APPLY
        assert "environment_mismatch" in res.reasons
        assert "environment" in res.mismatched


def test_applicability_missing_binding_unknown(store, scope_id):
    """Missing required binding → unknown, never a wildcard (V3-21.07)."""
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        pid = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac).procedure_id
        res = check_applicability(conn, pid, ENV, {})
        assert res.verdict == ApplicabilityVerdict.UNKNOWN
        assert res.missing_bindings


def test_applicability_no_env_unknown(store, scope_id):
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        pid = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac).procedure_id
        row = _proc_row(conn, pid)
        res = check_applicability(conn, pid, None, _bind_all(row))
        assert res.verdict == ApplicabilityVerdict.UNKNOWN
        assert "environment_not_provided" in res.reasons


# ---------------------------------------------------------------------------
# contrastive refinement (V3-22.03)
# ---------------------------------------------------------------------------


def test_refine_records_hypotheses(store, scope_id):
    """Differences become hypothesis conditions — never proven."""
    with store.tx() as conn:
        s1 = _seed_good(conn, store, scope_id)
        pid = compile_episode(conn, s1["episode_id"], hmac_fn=store.hmac).procedure_id
        # Same signature, but the patch touches a different path → the
        # binding's observed value differs.
        ops2 = [dict(GOOD_OPS[0]), dict(GOOD_OPS[1]),
                {"tool": "apply_patch",
                 "args": {"path": "src/other.py", "patch": "@@"}},
                dict(GOOD_OPS[3])]
        s2 = _seed_episode(conn, store, scope_id, ops2, outcome="failure",
                           checker_outcome="failure", env=dict(ENV_ROWS))
        res = refine_procedure(conn, pid, s2["episode_id"], hmac_fn=store.hmac)
        assert res["refined"] is True
        row = _proc_row(conn, pid)
        assert row["revision"] == 2
        conds = _j(row, "applicability_json")
        hyps = [c for c in conds if c.get("provenance") == "hypothesis"]
        assert hyps
        for h in hyps:
            assert h["validated"] is False
        prov = _j(row, "provenance_json")
        assert prov["refinements"][-1]["contrasting_episode_id"] == (
            s2["episode_id"])


def test_refine_signature_mismatch_noop(store, scope_id):
    with store.tx() as conn:
        s1 = _seed_good(conn, store, scope_id)
        pid = compile_episode(conn, s1["episode_id"], hmac_fn=store.hmac).procedure_id
        s2 = _seed_episode(conn, store, scope_id, GOOD_OPS,
                           goal_class="other-task", label="other-task",
                           env=dict(ENV_ROWS))
        res = refine_procedure(conn, pid, s2["episode_id"], hmac_fn=store.hmac)
        assert res["refined"] is False
        assert res["reason"] == "signature_mismatch"
        assert _proc_row(conn, pid)["revision"] == 1


# ---------------------------------------------------------------------------
# ops order
# ---------------------------------------------------------------------------


def test_ops_follow_transition_ord(store, scope_id):
    """Ops order comes from transition ord, not insertion order."""
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id, insert_order_shuffled=True)
        pid = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac).procedure_id
        ops = _j(_proc_row(conn, pid), "operations_json")
        assert [o["ord"] for o in ops] == [1, 2, 3, 4]
        assert [o["op_class"] for o in ops] == [
            "inspect_file", "search_repo", "apply_patch", "run_check",
        ]


# ---------------------------------------------------------------------------
# exposures / reuse stats (V3-22.06/22.07)
# ---------------------------------------------------------------------------


def test_record_exposure_and_stats(store, scope_id):
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        pid = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac).procedure_id
        e1 = record_exposure(conn, pid, "task-1", "exposed",
                             environment_digest="envd1")
        record_exposure(conn, pid, "task-1", "applicable",
                        environment_digest="envd1")
        record_exposure(conn, pid, "task-1", "adopted",
                        environment_digest="envd1")
        record_exposure(conn, pid, "task-1", "success",
                        environment_digest="envd1")
        record_exposure(conn, pid, "task-2", "failure",
                        environment_digest="envd2")
        record_exposure(conn, pid, "task-3", "success",
                        experiment_id="exp-1", arm="with",
                        environment_digest="envd1")
        record_exposure(conn, pid, "task-3", "failure",
                        experiment_id="exp-1", arm="without",
                        environment_digest="envd1")
        rows = exposures_for(conn, pid)
        assert len(rows) == 7
        stats = reuse_stats(conn, pid)
        assert stats["total"] == 7
        assert stats["exposed"] == 1
        assert stats["success"] == 2
        assert stats["failure"] == 2
        assert stats["by_environment"]["envd1"]["total"] == 6
        assert stats["by_environment"]["envd2"]["failure"] == 1
        # Paired-arm bookkeeping for negative-transfer measurement.
        assert stats["paired"]["exp-1"]["with"]["success"] == 1
        assert stats["paired"]["exp-1"]["without"]["failure"] == 1


def test_record_exposure_validation(store, scope_id):
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        pid = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac).procedure_id
        with pytest.raises(VerbatimError):
            record_exposure(conn, pid, "t", "bogus")
        with pytest.raises(VerbatimError):
            record_exposure(conn, pid, "t", "success",
                            experiment_id="e")  # arm missing
        with pytest.raises(VerbatimError):
            record_exposure(conn, "proc:nope", "t", "success")


# ---------------------------------------------------------------------------
# signatures / index jobs
# ---------------------------------------------------------------------------


def test_signature_index_idempotent(store, scope_id):
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        pid = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac).procedure_id
        conn.execute("DELETE FROM procedure_signatures")
        assert index_procedure(conn, pid) == 1
        assert index_scope(conn, scope_id) == 1
        n = conn.execute(
            "SELECT COUNT(*) FROM procedure_signatures").fetchone()[0]
        assert n == 1
        row = conn.execute(
            "SELECT ordered_ops_json FROM procedure_signatures"
            " WHERE procedure_id=?",
            (pid,)).fetchone()
        ops = safe_json_loads(row[0])
        assert ops == [
            "inspect_file", "search_repo", "apply_patch", "run_check",
        ]


def test_compute_signature_deterministic():
    a = compute_signature('{"goal_class":"g"}', ["inspect_file"])
    b = compute_signature('{"goal_class":"g"}', ["inspect_file"])
    c = compute_signature('{"goal_class":"g"}', ["search_repo"])
    assert a == b != c


# ---------------------------------------------------------------------------
# handlers through the durable queue (§40)
# ---------------------------------------------------------------------------


def _ingester(store) -> Ingester:
    return Ingester(store, VerbatimConfig())


def test_compile_job_drains(store, scope_id):
    ing = _ingester(store)
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        jid = ing.jobs.enqueue(
            conn, scope_id, JobKind.PROCEDURE_COMPILE,
            {"episode_id": seed["episode_id"]},
            operation_key=f"compile:{seed['episode_id']}",
        )
    n = ing.run_pending(scope=None)
    assert n >= 1
    with store.read() as conn:
        job = conn.execute(
            "SELECT state FROM jobs WHERE job_id = ?", (jid,)
        ).fetchone()
        assert job[0] == "succeeded"
        proc = conn.execute("SELECT state FROM procedures").fetchone()
        assert proc[0] == "candidate"


def test_compile_job_replays_receipt(store, scope_id):
    """A redelivered compile job replays its receipt, not a second write."""
    ing = _ingester(store)
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        jid = ing.jobs.enqueue(
            conn, scope_id, JobKind.PROCEDURE_COMPILE,
            {"episode_id": seed["episode_id"]},
            operation_key=f"compile:{seed['episode_id']}",
        )
    ing.run_pending(scope=None)
    # Directly re-drive the handler as a redelivery with the same job row:
    # the operations ledger replays instead of double-compiling.
    with store.read() as conn:
        cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (jid,))
        cols = [d[0] for d in cur.description]
        job = dict(zip(cols, cur.fetchone()))
    job["input_refs"] = {"episode_id": seed["episode_id"]}
    job["generation"] = job.get("generation") or 0
    with store.tx() as conn:
        # Simulate a fenced re-execution against a stale lease: assert_lease
        # must refuse because the job already succeeded.
        with pytest.raises(VerbatimError):
            ing.jobs.assert_lease(
                conn, jid, "inline", job["generation"]
            )
    with store.read() as conn:
        n = conn.execute("SELECT COUNT(*) FROM procedures").fetchone()[0]
        assert n == 1


def test_refine_job_drains(store, scope_id):
    ing = _ingester(store)
    with store.tx() as conn:
        s1 = _seed_good(conn, store, scope_id)
        pid = compile_episode(conn, s1["episode_id"], hmac_fn=store.hmac).procedure_id
        s2 = _seed_episode(conn, store, scope_id, GOOD_OPS, outcome="failure",
                           checker_outcome="failure", env=dict(ENV_ROWS))
        jid = ing.jobs.enqueue(
            conn, scope_id, JobKind.PROCEDURE_REFINE,
            {"procedure_id": pid,
             "contrasting_episode_id": s2["episode_id"]},
        )
    ing.run_pending(scope=None)
    with store.read() as conn:
        job = conn.execute(
            "SELECT state FROM jobs WHERE job_id = ?", (jid,)
        ).fetchone()
        assert job[0] == "succeeded"
        rev = conn.execute(
            "SELECT revision FROM procedures WHERE procedure_id=?",
            (pid,)).fetchone()
        assert rev[0] == 2


def test_signature_index_job(store, scope_id):
    ing = _ingester(store)
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        pid = compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac).procedure_id
        conn.execute("DELETE FROM procedure_signatures")
        jid = ing.jobs.enqueue(
            conn, scope_id, JobKind.SIGNATURE_INDEX,
            {"procedure_id": pid},
        )
    ing.run_pending(scope=None)
    with store.read() as conn:
        job = conn.execute(
            "SELECT state FROM jobs WHERE job_id = ?", (jid,)
        ).fetchone()
        assert job[0] == "succeeded"
        n = conn.execute(
            "SELECT COUNT(*) FROM procedure_signatures").fetchone()[0]
        assert n == 1


def test_compiler_class_wrapper(store, scope_id):
    with store.tx() as conn:
        seed = _seed_good(conn, store, scope_id)
        compiler = ProcedureCompiler()
        r = compiler.compile_episode(conn, seed["episode_id"], hmac_fn=store.hmac)
        assert r.status == CompilationStatus.CANDIDATE
        env = _environment_for(conn, scope_id)
        assert env.repo_id == "repo-1"
        assert dict(env.runtime_versions) == {"python": "3.12"}
        assert dict(env.tool_schema_versions) == {"pytest": "8.0"}
