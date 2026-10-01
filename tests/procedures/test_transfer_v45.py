"""Environment- and counterexample-qualified procedure transfer
(SPEC_V4_5 §06, V45-06.*; D07/D08 honesty semantics).

Pinned behaviors:

- a mismatched environment *blocks* delivery — the host never receives
  the reuse card, only the verdict and the counterexamples;
- an unknown verdict delivers but loudly qualifies (never a wildcard
  pass, V3-22.15);
- recorded failure modes ride along as counterexamples with their
  evidence references, and ``no_failure_evidence`` is labeled rather
  than presumed safe (V45-06.03);
- ``positive_only`` is the honest comparator arm — identical card, no
  qualification — while liveness still gates in both modes;
- ``transfer_success`` requires a delivery record AND a host-attested
  positive outcome bound to the task — exposure counts, adoption, and
  agent self-reports never count (V45-06.02, D08).
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

import pytest

from verbatim.core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    new_id,
)
from verbatim.core.types_v3 import EnvironmentFingerprint
from verbatim.procedures import (
    compile_episode,
    deliver_procedure,
    transfer_success,
)
from verbatim.procedures.exposures import exposures_for, reuse_stats
from verbatim.procedures.transfer import (
    TRANSFER_POLICY_ID,
    counterexamples,
)
from verbatim.storage.store import Store

SCOPE = "scope:transfer"
ENV_ROWS = {
    "repo_id": "repo-1",
    "repo_revision": "abc123",
    "platform": "linux",
    "runtime.python": "3.12",
    "tool.pytest": "8.0",
}
HELD_OUT_ENV = dict(ENV_ROWS, platform="darwin")


@pytest.fixture()
def store(tmp_path):
    s = Store.create(str(tmp_path / "transfer.db"))
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
    """Minimal compileable episode: trajectory + steps + checker."""
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


def _compile(conn, store, scope_id, *, errors=None, env=None) -> str:
    """Compile one episode through the real producer path on the
    caller's transaction, then promote to ``active`` so the reuse
    surface sees a live procedure."""
    ep = _seed_episode(conn, store, scope_id, errors=errors, env=env)
    result = compile_episode(conn, ep, hmac_fn=store.hmac)
    assert result.procedure_id, f"compile gated: {result}"
    conn.execute(
        "UPDATE procedures SET state='active' WHERE procedure_id=?",
        (result.procedure_id,),
    )
    return result.procedure_id


def safe_loads(raw):
    from verbatim.core.types import safe_json_loads

    return safe_json_loads(raw)


def _bindings_for(conn, pid) -> dict[str, Any]:
    row = conn.execute(
        "SELECT bindings_json FROM procedures WHERE procedure_id=?",
        (pid,),
    ).fetchone()
    return {
        b["name"]: b.get("observed_value")
        for b in (safe_loads(row[0]) or [])
        if isinstance(b, dict) and b.get("required") and b.get("name")
    }


# ---------------------------------------------------------------------------
# environment qualification (V45-06.01)
# ---------------------------------------------------------------------------


def test_environment_mismatch_blocks_delivery(store, scope_id):
    """Held-out environment: does_not_apply → blocked, no card ships."""
    with store.tx() as conn:
        pid = _compile(conn, store, scope_id, env=dict(ENV_ROWS))
        d = deliver_procedure(
            conn, pid, task_id="t1", environment=HELD_OUT_ENV,
        )
        assert d.deliverable is False
        assert d.disposition == "blocked"
        assert d.environment_match == "mismatch"
        assert d.card is None
        assert d.exposure_id is None
        assert d.applicability["verdict"] == "does_not_apply"
        assert "environment_mismatch" in d.applicability["reasons"]
        # A blocked delivery records no exposure — the host never saw
        # the procedure, and the denominators stay honest.
        assert exposures_for(conn, pid) == []


def test_matching_environment_delivers(store, scope_id):
    with store.tx() as conn:
        pid = _compile(conn, store, scope_id, env=dict(ENV_ROWS))
        d = deliver_procedure(
            conn, pid, task_id="t1", environment=dict(ENV_ROWS),
            bindings=_bindings_for(conn, pid),
        )
        assert d.deliverable is True
        assert d.disposition == "delivered"
        assert d.environment_match == "match"
        assert d.card is not None
        assert d.card["procedure_id"] == pid
        assert d.exposure_id is not None
        exp = exposures_for(conn, pid)
        assert len(exp) == 1 and exp[0]["outcome"] == "applicable"
        assert exp[0]["task_id"] == "t1"


def test_unknown_environment_qualifies_loudly(store, scope_id):
    """Missing environment evidence is unknown — qualified, not a pass."""
    with store.tx() as conn:
        pid = _compile(conn, store, scope_id, env=dict(ENV_ROWS))
        d = deliver_procedure(
            conn, pid, task_id="t1",
            environment=None,  # caller didn't supply one
            bindings=_bindings_for(conn, pid),
        )
        assert d.deliverable is True
        assert d.disposition == "qualified"
        assert d.environment_match == "unknown"
        assert any(w.startswith("applicability_unknown") for w in d.warnings)
        exp = exposures_for(conn, pid)
        assert len(exp) == 1 and exp[0]["outcome"] == "exposed"


def test_missing_required_binding_qualifies(store, scope_id):
    with store.tx() as conn:
        pid = _compile(conn, store, scope_id, env=dict(ENV_ROWS))
        d = deliver_procedure(
            conn, pid, environment=dict(ENV_ROWS), bindings={},
        )
        assert d.deliverable is True
        assert d.disposition == "qualified"
        assert "missing_required_bindings" in d.applicability["reasons"]


def test_positive_only_arm_delivers_unqualified(store, scope_id):
    """The comparator arm ships the same card with zero qualification —
    even into a held-out environment (that is the measured harm)."""
    with store.tx() as conn:
        pid = _compile(conn, store, scope_id, env=dict(ENV_ROWS))
        d = deliver_procedure(
            conn, pid, task_id="t1", environment=HELD_OUT_ENV,
            mode="positive_only",
        )
        assert d.deliverable is True
        assert d.disposition == "delivered"
        assert d.environment_match == "unqualified"
        assert d.counterexamples == ()
        assert d.failure_evidence == "unqualified"
        assert d.card is not None
        exp = exposures_for(conn, pid)
        assert len(exp) == 1 and exp[0]["outcome"] == "exposed"


def test_retired_procedure_blocked_in_both_modes(store, scope_id):
    with store.tx() as conn:
        pid = _compile(conn, store, scope_id, env=dict(ENV_ROWS))
        conn.execute(
            "UPDATE procedures SET state='retired' WHERE procedure_id=?",
            (pid,),
        )
        for mode in ("failure_aware", "positive_only"):
            d = deliver_procedure(
                conn, pid, environment=dict(ENV_ROWS), mode=mode,
            )
            assert d.deliverable is False
            assert d.disposition == "blocked"
            assert d.card is None


def test_invalid_mode_rejected(store, scope_id):
    with store.tx() as conn:
        pid = _compile(conn, store, scope_id, env=dict(ENV_ROWS))
        with pytest.raises(VerbatimError) as ei:
            deliver_procedure(conn, pid, mode="yolo")
    assert ei.value.code == ErrorCode.VALIDATION


# ---------------------------------------------------------------------------
# counterexample attachment (V45-06.03)
# ---------------------------------------------------------------------------


def test_counterexamples_attach_with_evidence_refs(store, scope_id):
    """Recorded failure modes ride along with their evidence references."""
    errors = [
        {"kind": "test_failure", "summary": "pytest failed: assert x",
         "tool": "run_check"},
    ]
    with store.tx() as conn:
        pid = _compile(conn, store, scope_id, errors=errors, env=dict(ENV_ROWS))
        ces = counterexamples(conn, pid)
        assert ces, "compiler should have extracted the error envelope"
        ce = ces[0]
        assert ce.signature and ce.evidence_refs
        assert all("id" in r for r in ce.evidence_refs)

        d = deliver_procedure(
            conn, pid, environment=dict(ENV_ROWS),
            bindings=_bindings_for(conn, pid),
        )
        assert d.counterexamples == ces
        assert d.failure_evidence == "observed"
        # Attached counterexamples loudly qualify the delivery.
        assert d.disposition == "qualified"
        assert any(
            w.startswith("counterexample:") for w in d.warnings
        )


def test_no_failure_evidence_is_labeled(store, scope_id):
    with store.tx() as conn:
        pid = _compile(conn, store, scope_id, env=dict(ENV_ROWS))
        assert counterexamples(conn, pid) == ()
        d = deliver_procedure(
            conn, pid, environment=dict(ENV_ROWS),
            bindings=_bindings_for(conn, pid),
        )
        assert d.counterexamples == ()
        assert d.failure_evidence == "no_failure_evidence"


# ---------------------------------------------------------------------------
# honest transfer success (V45-06.02, D08)
# ---------------------------------------------------------------------------


def _attested_outcome(
    conn, store, scope_id, task_id, *, outcome="success", bound_task=None,
    host_attested=True, invocation=True, agent_report=False,
):
    meta: dict[str, Any] = {
        "outcome": outcome,
        "exit_code": 0 if outcome == "success" else 1,
        "completed": True,
        "checker_receipt": {
            "checker_id": "pytest",
            "host_attested": host_attested,
            "invocation_id": "inv-1" if invocation else "",
            "task_id": bound_task if bound_task is not None else task_id,
            "scope_id": scope_id,
            "exit_code": 0 if outcome == "success" else 1,
            "completed": True,
        },
    }
    if agent_report:
        meta["checker_receipt"]["agent_report"] = True
    return _envelope(conn, store, scope_id, "test_result",
                     metadata=meta, task_id=task_id)


def test_transfer_success_requires_delivery_and_attestation(store, scope_id):
    """Delivered + host-attested success → transfer_success True."""
    with store.tx() as conn:
        pid = _compile(conn, store, scope_id, env=dict(ENV_ROWS))
        deliver_procedure(
            conn, pid, task_id="t-ok", environment=dict(ENV_ROWS),
            bindings=_bindings_for(conn, pid),
        )
        _attested_outcome(conn, store, scope_id, "t-ok")
        res = transfer_success(conn, pid, "t-ok")
        assert res["delivered_procedure"] is True
        assert res["attested_outcome"] == "success"
        assert res["transfer_success"] is True


def test_delivery_alone_is_not_success(store, scope_id):
    """Exposure without any outcome envelope is not transfer success."""
    with store.tx() as conn:
        pid = _compile(conn, store, scope_id, env=dict(ENV_ROWS))
        deliver_procedure(
            conn, pid, task_id="t-none", environment=dict(ENV_ROWS),
            bindings=_bindings_for(conn, pid),
        )
        res = transfer_success(conn, pid, "t-none")
        assert res["delivered_procedure"] is True
        assert res["attested_outcome"] == "none"
        assert res["transfer_success"] is False
        assert "no_outcome_envelopes" in res["reasons"]


def test_agent_report_is_not_attestation(store, scope_id):
    """A self-reported outcome can never satisfy the attestation clause."""
    with store.tx() as conn:
        pid = _compile(conn, store, scope_id, env=dict(ENV_ROWS))
        deliver_procedure(
            conn, pid, task_id="t-agent", environment=dict(ENV_ROWS),
            bindings=_bindings_for(conn, pid),
        )
        _attested_outcome(
            conn, store, scope_id, "t-agent", agent_report=True,
        )
        res = transfer_success(conn, pid, "t-agent")
        assert res["attested_outcome"] == "none"
        assert res["attested_envelopes"] == 0
        assert res["non_attested_envelopes"] == 1
        assert res["transfer_success"] is False
        assert "no_attested_outcome" in res["reasons"]


def test_no_invocation_id_is_not_attestation(store, scope_id):
    with store.tx() as conn:
        pid = _compile(conn, store, scope_id, env=dict(ENV_ROWS))
        deliver_procedure(
            conn, pid, task_id="t-noinv", environment=dict(ENV_ROWS),
            bindings=_bindings_for(conn, pid),
        )
        _attested_outcome(
            conn, store, scope_id, "t-noinv", invocation=False,
        )
        res = transfer_success(conn, pid, "t-noinv")
        assert res["transfer_success"] is False
        assert res["attested_outcome"] == "none"


def test_attested_failure_is_not_success(store, scope_id):
    with store.tx() as conn:
        pid = _compile(conn, store, scope_id, env=dict(ENV_ROWS))
        deliver_procedure(
            conn, pid, task_id="t-fail", environment=dict(ENV_ROWS),
            bindings=_bindings_for(conn, pid),
        )
        _attested_outcome(
            conn, store, scope_id, "t-fail", outcome="failure",
        )
        res = transfer_success(conn, pid, "t-fail")
        assert res["attested_outcome"] == "failure"
        assert res["transfer_success"] is False
        assert "attested_failure" in res["reasons"]


def test_receipt_bound_to_other_task_does_not_certify(store, scope_id):
    """A checker receipt pinned to a different task cannot certify this
    one (V4-23.01 binding) — even though the envelope row names it."""
    with store.tx() as conn:
        pid = _compile(conn, store, scope_id, env=dict(ENV_ROWS))
        deliver_procedure(
            conn, pid, task_id="t-1", environment=dict(ENV_ROWS),
            bindings=_bindings_for(conn, pid),
        )
        _attested_outcome(
            conn, store, scope_id, "t-1", bound_task="t-2",
        )
        res = transfer_success(conn, pid, "t-1")
        assert res["attested_outcome"] == "none"
        assert res["transfer_success"] is False


def test_attestation_without_delivery_is_not_success(store, scope_id):
    """The outcome must pair with an actual procedure delivery for the
    task — a stray success envelope is not transfer."""
    with store.tx() as conn:
        pid = _compile(conn, store, scope_id, env=dict(ENV_ROWS))
        _attested_outcome(conn, store, scope_id, "t-orphan")
        res = transfer_success(conn, pid, "t-orphan")
        assert res["delivered_procedure"] is False
        assert res["attested_outcome"] == "success"
        assert res["transfer_success"] is False
        assert "no_delivery_record" in res["reasons"]


def test_delivery_record_blocks_exposure_inflation(store, scope_id):
    """Blocked deliveries never mint exposure rows, so a held-out task
    cannot accumulate 'delivered' evidence it never received."""
    with store.tx() as conn:
        pid = _compile(conn, store, scope_id, env=dict(ENV_ROWS))
        d = deliver_procedure(
            conn, pid, task_id="t-heldout", environment=HELD_OUT_ENV,
        )
        assert d.deliverable is False
        _attested_outcome(conn, store, scope_id, "t-heldout")
        res = transfer_success(conn, pid, "t-heldout")
        # Even with a positive attested outcome, there was no delivery —
        # the pair reports failure honestly rather than claiming reuse.
        assert res["transfer_success"] is False


def test_reuse_stats_reflect_delivery_outcomes(store, scope_id):
    with store.tx() as conn:
        pid = _compile(conn, store, scope_id, env=dict(ENV_ROWS))
        deliver_procedure(
            conn, pid, task_id="t1", environment=dict(ENV_ROWS),
            bindings=_bindings_for(conn, pid),
        )
        stats = reuse_stats(conn, pid)
        assert stats["applicable"] == 1 and stats["total"] == 1
