"""V3 transition derivation tests (SPEC_V3 §20.02–§20.04, §17.02, §20.08).

Pinned behaviors:

- A completed 3-step episode derives exactly one transition per step;
  the checker-observed step's edge is ``verified_by`` with
  ``checker_ref`` set, the rest are ``observed_after`` — order, never
  causation (V3-20.04).
- Anchor windows: pre = anchors on strictly earlier steps; post =
  anchors on this step or the next (V3-20.03).
- Replay yields identical transition identity (deterministic
  ``transition_id``, ``ord`` = step ord) and never duplicates anchor
  links or derivations edges (V3-20.08, V3-17.02).
"""

from __future__ import annotations

import pytest

from verbatim.config import VerbatimConfig
from verbatim.core.types import (
    ErrorCode,
    JobKind,
    VerbatimError,
    json_dumps,
)
from verbatim.core.types_v3 import EnvironmentFingerprint, TransitionEdge
from verbatim.experience import anchors as anch
from verbatim.experience import episodes_v3, transitions
from verbatim.ingest import Ingester
from verbatim.storage import repos_v3
from verbatim.storage.store import Store

from . import v3_seed as seed


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "v3tr.db"))
    yield s
    s.close()


@pytest.fixture
def scope_id(store):
    sid = "scope:v3tr"
    with store.tx() as conn:
        seed.seed_scope(conn, sid)
    return sid


@pytest.fixture
def ingester(store):
    return Ingester(store, VerbatimConfig())


@pytest.fixture
def episode_id(store, scope_id):
    """A built episode over the canonical 3-step trajectory."""
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id)
        return episodes_v3.build_episode(conn, "traj-1")


def _dicts(cur):
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _transitions(store, episode_id):
    with store.read() as conn:
        return transitions.transitions_for_episode(conn, episode_id)


def _links(store, transition_id):
    with store.read() as conn:
        return repos_v3.query(
            conn, "transition_anchors", {"transition_id": transition_id}
        )


def _derivations(store, child_id):
    with store.read() as conn:
        return repos_v3.query(
            conn, "derivations", {"child_kind": "transition", "child_id": child_id}
        )


def _jobs(store, kind=None):
    with store.read() as conn:
        if kind:
            cur = conn.execute(
                "SELECT * FROM jobs WHERE kind = ? ORDER BY rowid", (kind,)
            )
        else:
            cur = conn.execute("SELECT * FROM jobs ORDER BY rowid")
        return _dicts(cur)


def _anchor_at(store, scope_id, step_id, tag):
    with store.tx() as conn:
        return anch.record_anchor(
            conn, scope_id, "file_revision", f"ref-{tag}", f"dg-{tag}", step_id
        )


# ---------------------------------------------------------------------------
# core derivation (V3-20.02, V3-20.04)
# ---------------------------------------------------------------------------


def test_three_step_episode_derives_three_transitions(store, episode_id):
    with store.tx() as conn:
        count = transitions.build_transitions(conn, episode_id)
    assert count == 3
    rows = _transitions(store, episode_id)
    assert [r["ord"] for r in rows] == [0, 1, 2]
    assert [r["action_step_id"] for r in rows] == ["step-1", "step-2", "step-3"]
    assert [r["edge"] for r in rows] == [
        TransitionEdge.OBSERVED_AFTER.value,
        TransitionEdge.OBSERVED_AFTER.value,
        TransitionEdge.VERIFIED_BY.value,
    ]
    # the checker-observed step points at the outcome envelope
    assert rows[2]["checker_ref"] == "env-test"
    assert rows[0]["checker_ref"] is None
    assert rows[1]["checker_ref"] is None
    # deterministic identity + environment digest propagation
    for row, step_id in zip(rows, ("step-1", "step-2", "step-3")):
        assert row["transition_id"] == transitions.transition_id_for(
            episode_id, step_id
        )
        assert row["environment_digest"] == "env-digest-1"
        assert row["episode_id"] == episode_id
        assert row["scope_id"] == "scope:v3tr"


def test_no_checker_all_observed_after(store, scope_id):
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id, checker=False)
        eid = episodes_v3.build_episode(conn, "traj-1")
        assert transitions.build_transitions(conn, eid) == 3
    rows = _transitions(store, eid)
    assert all(r["edge"] == TransitionEdge.OBSERVED_AFTER.value for r in rows)
    assert all(r["checker_ref"] is None for r in rows)


def test_outcome_envelope_without_receipt_stays_observed(store, scope_id):
    """A test_result with no identified checker does not verify anything."""
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id, checker=False)
        conn.execute(
            "UPDATE source_envelopes SET metadata_json = ?"
            " WHERE envelope_id = 'env-test'",
            (json_dumps({"outcome": "success"}),),
        )
        eid = episodes_v3.build_episode(conn, "traj-1")
        transitions.build_transitions(conn, eid)
    rows = _transitions(store, eid)
    assert rows[2]["edge"] == TransitionEdge.OBSERVED_AFTER.value
    assert rows[2]["checker_ref"] is None


def test_verification_kind_also_verifies(store, scope_id):
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id, checker=False)
        conn.execute(
            "UPDATE source_envelopes SET envelope_kind = 'verification',"
            " metadata_json = ? WHERE envelope_id = 'env-test'",
            (json_dumps(seed.checker_metadata("success", exit_code=0)),),
        )
        eid = episodes_v3.build_episode(conn, "traj-1")
        transitions.build_transitions(conn, eid)
    assert _transitions(store, eid)[2]["edge"] == (
        TransitionEdge.VERIFIED_BY.value
    )


def test_causal_hypothesis_never_emitted(store, episode_id):
    """V3-20.04: this producer writes observed_after/verified_by only —
    causal claims need a separately labeled method."""
    with store.tx() as conn:
        transitions.build_transitions(conn, episode_id)
    rows = _transitions(store, episode_id)
    assert {r["edge"] for r in rows} <= {
        TransitionEdge.OBSERVED_AFTER.value,
        TransitionEdge.VERIFIED_BY.value,
    }


# ---------------------------------------------------------------------------
# anchor windows (V3-20.03)
# ---------------------------------------------------------------------------


def test_anchor_windows_partition_pre_and_post(store, scope_id, episode_id):
    a1 = _anchor_at(store, scope_id, "step-1", "a1")
    a2 = _anchor_at(store, scope_id, "step-2", "a2")
    a3 = _anchor_at(store, scope_id, "step-3", "a3")
    with store.tx() as conn:
        transitions.build_transitions(conn, episode_id)

    t1 = transitions.transition_id_for(episode_id, "step-1")
    t2 = transitions.transition_id_for(episode_id, "step-2")
    t3 = transitions.transition_id_for(episode_id, "step-3")

    with store.read() as conn:
        w1 = transitions.anchors_for_transition(conn, t1)
        w2 = transitions.anchors_for_transition(conn, t2)
        w3 = transitions.anchors_for_transition(conn, t3)

    # step-1: nothing before it; post = this step + next (a1, a2)
    assert [a["anchor_id"] for a in w1["pre"]] == []
    assert [a["anchor_id"] for a in w1["post"]] == [a1, a2]
    # step-2: pre = earlier steps (a1); post = this + next (a2, a3)
    assert [a["anchor_id"] for a in w2["pre"]] == [a1]
    assert [a["anchor_id"] for a in w2["post"]] == [a2, a3]
    # step-3: pre = a1, a2; post = just its own step (no next)
    assert [a["anchor_id"] for a in w3["pre"]] == [a1, a2]
    assert [a["anchor_id"] for a in w3["post"]] == [a3]


def test_transition_anchor_rows_carry_role_and_ord(store, scope_id, episode_id):
    a1 = _anchor_at(store, scope_id, "step-1", "x1")
    _anchor_at(store, scope_id, "step-3", "x3")
    with store.tx() as conn:
        transitions.build_transitions(conn, episode_id)
    t3 = transitions.transition_id_for(episode_id, "step-3")
    links = _links(store, t3)
    roles = {(l["role"], l["anchor_id"]) for l in links}
    assert ("pre", a1) in roles
    assert all(l["transition_id"] == t3 for l in links)


def test_anchor_outside_trajectory_ignored(store, scope_id, episode_id):
    """An anchor attached to a step of ANOTHER trajectory never leaks in."""
    with store.tx() as conn:
        seed.seed_trajectory(
            conn,
            scope_id,
            "traj-other",
            steps=[{"step_id": "other-1", "ord": 0}],
            task_id="task-other",
        )
    other = _anchor_at(store, scope_id, "other-1", "other")
    mine = _anchor_at(store, scope_id, "step-2", "mine")
    with store.tx() as conn:
        transitions.build_transitions(conn, episode_id)
    t2 = transitions.transition_id_for(episode_id, "step-2")
    with store.read() as conn:
        w2 = transitions.anchors_for_transition(conn, t2)
    assert [a["anchor_id"] for a in w2["post"]] == [mine]
    assert other not in [a["anchor_id"] for a in w2["pre"] + w2["post"]]


def test_derivation_edges_cover_step_envelopes_and_anchors(
    store, scope_id, episode_id
):
    """V3-17.02: every transition input is a recorded parent."""
    a1 = _anchor_at(store, scope_id, "step-1", "d1")
    a3 = _anchor_at(store, scope_id, "step-3", "d3")
    with store.tx() as conn:
        transitions.build_transitions(conn, episode_id)
    t3 = transitions.transition_id_for(episode_id, "step-3")
    edges = _derivations(store, t3)
    parents = {(e["parent_kind"], e["parent_id"]) for e in edges}
    assert ("trajectory_step", "step-3") in parents
    assert ("envelope", "env-run") in parents             # action envelope
    assert ("envelope", "env-test") in parents            # checker envelope
    assert ("state_anchor", a1) in parents                # pre anchor
    assert ("state_anchor", a3) in parents                # post anchor
    assert all(e["producer_id"] == transitions.TRANSITION_PRODUCER_ID for e in edges)
    assert all(e["scope_id"] == "scope:v3tr" for e in edges)
    # seq values are a dense deterministic ordering
    assert sorted(e["seq"] for e in edges) == list(range(len(edges)))


# ---------------------------------------------------------------------------
# idempotency + late evidence (V3-20.08, V3-17.03)
# ---------------------------------------------------------------------------


def test_rebuild_is_idempotent(store, scope_id, episode_id):
    a1 = _anchor_at(store, scope_id, "step-1", "r1")
    with store.tx() as conn:
        assert transitions.build_transitions(conn, episode_id) == 3
    before = _transitions(store, episode_id)
    with store.tx() as conn:
        assert transitions.build_transitions(conn, episode_id) == 3
    after = _transitions(store, episode_id)
    assert after == before
    t2 = transitions.transition_id_for(episode_id, "step-2")
    # anchor links + derivations unchanged on replay
    assert _links(store, t2) == _links(store, t2)
    with store.read() as conn:
        n_links = conn.execute(
            "SELECT COUNT(*) FROM transition_anchors"
        ).fetchone()[0]
        n_deriv = conn.execute(
            "SELECT COUNT(*) FROM derivations WHERE child_kind = 'transition'"
        ).fetchone()[0]
    # a1@step-1 is t1.post, t2.pre, t3.pre — three links total
    assert n_links == 3
    # t1: step+env+1 post anchor=3 ; t2: step+env+1 pre anchor=3 ;
    # t3: step+env+checker env+1 pre anchor=4
    assert n_deriv == 3 + 3 + 4


def test_late_checker_refines_edge_in_place(store, scope_id):
    """A checker receipt linked after the first build upgrades the same
    transition's edge to verified_by — one row per step, stable id."""
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id, checker=False)
        eid = episodes_v3.build_episode(conn, "traj-1")
        transitions.build_transitions(conn, eid)
    t3 = transitions.transition_id_for(eid, "step-3")
    assert _transitions(store, eid)[2]["edge"] == (
        TransitionEdge.OBSERVED_AFTER.value
    )
    with store.tx() as conn:
        conn.execute(
            "UPDATE source_envelopes SET metadata_json = ?"
            " WHERE envelope_id = 'env-test'",
            (json_dumps(seed.checker_metadata("success", exit_code=0)),),
        )
        assert transitions.build_transitions(conn, eid) == 3
    rows = _transitions(store, eid)
    assert len(rows) == 3
    assert rows[2]["transition_id"] == t3
    assert rows[2]["edge"] == TransitionEdge.VERIFIED_BY.value
    assert rows[2]["checker_ref"] == "env-test"


def test_new_anchor_joins_existing_transition_on_rebuild(
    store, scope_id, episode_id
):
    """Anchors recorded between builds join the window on rebuild —
    links are additive, never duplicated."""
    with store.tx() as conn:
        transitions.build_transitions(conn, episode_id)
    a2 = _anchor_at(store, scope_id, "step-2", "late")
    with store.tx() as conn:
        transitions.build_transitions(conn, episode_id)
    t2 = transitions.transition_id_for(episode_id, "step-2")
    with store.read() as conn:
        w2 = transitions.anchors_for_transition(conn, t2)
    assert [a["anchor_id"] for a in w2["post"]] == [a2]
    # and it is also step-3's pre-state
    t3 = transitions.transition_id_for(episode_id, "step-3")
    with store.read() as conn:
        w3 = transitions.anchors_for_transition(conn, t3)
    assert [a["anchor_id"] for a in w3["pre"]] == [a2]


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_missing_episode_refused(store):
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            transitions.build_transitions(conn, "ep-nope")
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_episode_without_trajectory_derivation_refused(store, scope_id):
    """An episode row with no trajectory parent cannot derive transitions."""
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO episodes (episode_id, scope_id, kind)"
            " VALUES ('ep-bare', ?, 'task')",
            (scope_id,),
        )
        with pytest.raises(VerbatimError) as exc:
            transitions.build_transitions(conn, "ep-bare")
    assert exc.value.code == ErrorCode.VALIDATION


def test_step_env_digest_overrides_trajectory(store, scope_id):
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id, environment_digest="traj-env")
        conn.execute(
            "UPDATE trajectory_steps SET environment_digest = 'step-env'"
            " WHERE step_id = 'step-2'"
        )
        eid = episodes_v3.build_episode(conn, "traj-1")
        transitions.build_transitions(conn, eid)
    rows = _transitions(store, eid)
    assert rows[0]["environment_digest"] == "traj-env"
    assert rows[1]["environment_digest"] == "step-env"


# ---------------------------------------------------------------------------
# anchors.py helpers
# ---------------------------------------------------------------------------


def test_record_anchor_deterministic_and_idempotent(store, scope_id):
    with store.tx() as conn:
        seed.seed_trajectory(
            conn, scope_id, "traj-a", steps=[{"step_id": "s1", "ord": 0}]
        )
        a1 = anch.record_anchor(conn, scope_id, "file_revision", "r", "d", "s1")
        assert anch.record_anchor(conn, scope_id, "file_revision", "r", "d", "s1") == a1
        rows = repos_v3.query(conn, "state_anchors", {"anchor_id": a1})
    assert len(rows) == 1
    assert rows[0]["kind"] == "file_revision"
    assert rows[0]["step_id"] == "s1"


def test_record_anchor_divergent_fields_integrity(store, scope_id):
    with store.tx() as conn:
        seed.seed_trajectory(
            conn, scope_id, "traj-a", steps=[{"step_id": "s1", "ord": 0}]
        )
        aid = anch.record_anchor(conn, scope_id, "file_revision", "r", "d", "s1")
        with pytest.raises(VerbatimError) as exc:
            anch.record_anchor(
                conn, scope_id, "file_revision", "r", "DIFFERENT", "s1",
                anchor_id=aid,
            )
    assert exc.value.code == ErrorCode.INTEGRITY


def test_anchor_validation(store, scope_id):
    with store.tx() as conn:
        with pytest.raises(VerbatimError):
            anch.record_anchor(conn, scope_id, "", "r", "d", "s1")
        with pytest.raises(VerbatimError):
            anch.record_anchor(conn, scope_id, "kind", "", "d", "s1")
        with pytest.raises(VerbatimError):
            anch.record_anchor(conn, scope_id, "kind", "r", "", "s1")


def test_anchors_near_before_after(store, scope_id):
    with store.tx() as conn:
        seed.seed_trajectory(
            conn,
            scope_id,
            "traj-n",
            steps=[
                {"step_id": "n1", "ord": 0},
                {"step_id": "n2", "ord": 5},
                {"step_id": "n3", "ord": 9},
            ],
        )
        a1 = anch.record_anchor(conn, scope_id, "file_revision", "r1", "d1", "n1")
        a2 = anch.record_anchor(conn, scope_id, "file_revision", "r2", "d2", "n2")
        a3 = anch.record_anchor(conn, scope_id, "file_revision", "r3", "d3", "n3")

        before = anch.anchors_near(conn, "n2", "before")
        after = anch.anchors_near(conn, "n2", "after")
        at = anch.anchors_for_step(conn, "n2")
        ords = anch.anchors_for_ords(conn, "traj-n", {5, 9})

    assert [a["anchor_id"] for a in before] == [a1]
    assert [a["anchor_id"] for a in after] == [a3]
    assert [a["anchor_id"] for a in at] == [a2]
    assert [a["anchor_id"] for a in ords] == [a2, a3]


def test_anchors_near_validation(store, scope_id):
    with store.tx() as conn:
        seed.seed_trajectory(
            conn, scope_id, "traj-n", steps=[{"step_id": "n1", "ord": 0}]
        )
        with pytest.raises(VerbatimError) as exc:
            anch.anchors_near(conn, "n1", "sideways")
        assert exc.value.code == ErrorCode.VALIDATION
        with pytest.raises(VerbatimError) as exc:
            anch.anchors_near(conn, "ghost-step", "before")
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_environment_digest_helpers():
    fp = EnvironmentFingerprint(
        repo_id="repo",
        repo_revision="abc",
        runtime_versions=(("python", "3.11"),),
        tool_schema_versions=(("pytest", "8"),),
        platform="linux",
    )
    d_obj = anch.environment_digest(fp)
    d_map = anch.environment_digest(
        {
            "repo_id": "repo",
            "repo_revision": "abc",
            "runtime_versions": [["python", "3.11"]],
            "tool_schema_versions": {"pytest": "8"},
            "platform": "linux",
        }
    )
    assert d_obj == d_map == fp.digest()
    # missing fields digest differently — unknown is never a wildcard
    assert anch.environment_digest(EnvironmentFingerprint()) != d_obj
    with pytest.raises(VerbatimError):
        anch.environment_digest("not-a-fingerprint")


# ---------------------------------------------------------------------------
# job handler (§40)
# ---------------------------------------------------------------------------


def _enqueue(ing, sid, kind, refs, op):
    with ing.store.tx() as conn:
        return ing.jobs.enqueue(
            conn,
            sid,
            kind,
            refs,
            dedup_key=ing.store.hmac(op.encode()),
            operation_key=op,
        )


def test_transition_build_job(store, scope_id, episode_id, ingester):
    _enqueue(
        ingester,
        scope_id,
        JobKind.TRANSITION_BUILD,
        {"episode_id": episode_id},
        f"transition_build:{episode_id}",
    )
    assert ingester.run_pending(scope=scope_id) == 1
    assert len(_transitions(store, episode_id)) == 3
    job = _jobs(store, "transition_build")[0]
    assert job["state"] == "succeeded"


def test_transition_build_job_rejects_missing_ref(store, scope_id, ingester):
    _enqueue(ingester, scope_id, JobKind.TRANSITION_BUILD, {}, "op-bad-t")
    assert ingester.run_pending(scope=scope_id) == 1
    job = _jobs(store, "transition_build")[0]
    assert job["state"] == "failed"
    assert job["error_code"] == ErrorCode.VALIDATION.value


def test_transition_build_job_fails_missing_episode(store, scope_id, ingester):
    _enqueue(
        ingester,
        scope_id,
        JobKind.TRANSITION_BUILD,
        {"episode_id": "ep-ghost"},
        "op-ghost",
    )
    assert ingester.run_pending(scope=scope_id) == 1
    job = _jobs(store, "transition_build")[0]
    assert job["state"] == "failed"
    assert job["error_code"] == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED.value
