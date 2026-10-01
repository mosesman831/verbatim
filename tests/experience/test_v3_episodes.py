"""V3 episode building tests (SPEC_V3 §20, §17.02, §12.03).

Pinned behaviors:

- A completed 3-step trajectory builds one episode (boundary rule +
  outcome + environment digest on the v3 columns) with member envelopes
  and derivations edges (V3-20.01/20.02, V3-17.02).
- Replay yields identical episode identity (V3-20.08) — deterministic
  ``episode_id`` plus derivation/member dedup.
- Incomplete trajectories refuse derivation: incomplete evidence stays
  evidence.
- Outcome resolves only from checker-attested outcome envelopes —
  failure episodes are retained as negative experience (V3-20.09), and
  an agent's "it worked" claim is never an outcome (V3-12.03).
"""

from __future__ import annotations

import pytest

from verbatim.config import VerbatimConfig
from verbatim.core.types import ErrorCode, JobKind, VerbatimError, json_dumps
from verbatim.core.types_v3 import OutcomeClass
from verbatim.experience import episodes_v3
from verbatim.ingest import Ingester
from verbatim.storage import repos_v3
from verbatim.storage.store import Store

from . import v3_seed as seed


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "v3ep.db"))
    yield s
    s.close()


@pytest.fixture
def scope_id(store):
    sid = "scope:v3ep"
    with store.tx() as conn:
        seed.seed_scope(conn, sid)
    return sid


@pytest.fixture
def ingester(store):
    return Ingester(store, VerbatimConfig())


def _dicts(cur):
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _ep_row(store, episode_id):
    with store.read() as conn:
        rows = _dicts(
            conn.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
            )
        )
    return rows[0] if rows else None


def _members(store, episode_id):
    with store.read() as conn:
        return _dicts(
            conn.execute(
                "SELECT * FROM episode_members WHERE episode_id = ?"
                " ORDER BY ord, object_id",
                (episode_id,),
            )
        )


def _derivations(store, child_id, child_kind="episode"):
    with store.read() as conn:
        return repos_v3.query(
            conn,
            "derivations",
            {"child_kind": child_kind, "child_id": child_id},
        )


def _episodes(store):
    with store.read() as conn:
        return _dicts(conn.execute("SELECT * FROM episodes"))


def _jobs(store, kind=None):
    with store.read() as conn:
        if kind:
            cur = conn.execute(
                "SELECT * FROM jobs WHERE kind = ? ORDER BY rowid", (kind,)
            )
        else:
            cur = conn.execute("SELECT * FROM jobs ORDER BY rowid")
        return _dicts(cur)


def _set_metadata(store, envelope_id, metadata):
    with store.tx() as conn:
        conn.execute(
            "UPDATE source_envelopes SET metadata_json = ?"
            " WHERE envelope_id = ?",
            (json_dumps(metadata), envelope_id),
        )


# ---------------------------------------------------------------------------
# happy path (V3-20.01/20.02)
# ---------------------------------------------------------------------------


def test_build_episode_groups_completed_trajectory(store, scope_id):
    with store.tx() as conn:
        ids = seed.seed_three_step_task(conn, scope_id)
        episode_id = episodes_v3.build_episode(conn, "traj-1")

    assert episode_id == episodes_v3.episode_id_for("traj-1")
    episode = _ep_row(store, episode_id)
    assert episode is not None
    assert episode["scope_id"] == scope_id
    assert episode["kind"] == "task"
    assert episode["boundary_rule"] == "task_id"
    assert episode["outcome"] == "success"
    assert episode["environment_digest"] == "env-digest-1"
    assert episode["host_task_id"] == "task-1"
    assert episode["host_session_id"] == "sess-1"
    assert episode["revision"] == 1
    assert episode["recorded_until"] is None

    members = _members(store, episode_id)
    assert [m["object_id"] for m in members] == [
        ids["tool_call"],
        ids["file_diff"],
        ids["run_check"],
        ids["test_result"],
    ]
    assert all(m["object_kind"] == "envelope" for m in members)
    assert [m["ord"] for m in members] == [0, 1, 2, 3]

    edges = _derivations(store, episode_id)
    parents = {(e["parent_kind"], e["parent_id"]) for e in edges}
    assert ("trajectory", "traj-1") in parents
    for eid in ids.values():
        assert ("envelope", eid) in parents
    assert all(e["producer_id"] == episodes_v3.EPISODE_PRODUCER_ID for e in edges)
    assert all(e["scope_id"] == scope_id for e in edges)
    traj_edge = [e for e in edges if e["parent_kind"] == "trajectory"][0]
    assert traj_edge["seq"] == 0
    # member envelope revisions flow into the edge
    env_edge = [
        e for e in edges if e["parent_id"] == ids["test_result"]
    ][0]
    assert env_edge["parent_revision"] == 1


def test_episode_for_trajectory_roundtrip(store, scope_id):
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id)
        episode_id = episodes_v3.build_episode(conn, "traj-1")
        assert episodes_v3.episode_for_trajectory(conn, "traj-1") == episode_id
        assert episodes_v3.episode_for_trajectory(conn, "traj-absent") is None


def test_build_episode_idempotent_replay(store, scope_id):
    """V3-20.08: replaying the trajectory yields identical episode identity
    and never duplicates members or edges."""
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id)
        first = episodes_v3.build_episode(conn, "traj-1")
    members_before = _members(store, first)
    with store.tx() as conn:
        second = episodes_v3.build_episode(conn, "traj-1")
    assert second == first
    assert len(_episodes(store)) == 1
    assert len(_members(store, first)) == 4
    assert len(_derivations(store, first)) == 5
    # membership history is not churned by the rebuild (V3-20.06)
    assert _members(store, first) == members_before


def test_build_episode_records_boundary_rule(store, scope_id):
    """V3-20.01: the declared boundary rule travels with the episode."""
    with store.tx() as conn:
        seed.seed_trajectory(
            conn,
            scope_id,
            "traj-seg",
            steps=[{"step_id": "s-1", "ord": 0}],
            boundary_rule="session_segment",
            task_id="",
        )
        eid = episodes_v3.build_episode(conn, "traj-seg")
    assert _ep_row(store, eid)["boundary_rule"] == "session_segment"


def test_distinct_trajectories_distinct_episodes(store, scope_id):
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id, "traj-1")
        seed.seed_three_step_task(
            conn, scope_id, "traj-2", task_id="task-2", prefix="t2-"
        )
        e1 = episodes_v3.build_episode(conn, "traj-1")
        e2 = episodes_v3.build_episode(conn, "traj-2")
    assert e1 != e2
    assert len(_episodes(store)) == 2


def test_foreign_producer_edge_does_not_redirect_lookup(store, scope_id):
    """A different producer's episode→trajectory edge is never adopted as
    this pipeline's build target — deterministic identity governs."""
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id)
        conn.execute(
            "INSERT INTO derivations"
            " (child_kind, child_id, child_revision, parent_kind, parent_id,"
            "  parent_revision, producer_kind, producer_id, seq, scope_id)"
            " VALUES ('episode', 'ep-foreign', 1, 'trajectory', 'traj-1', 1,"
            "         'producer', 'other.builder', 0, ?)",
            (scope_id,),
        )
        eid = episodes_v3.build_episode(conn, "traj-1")
    assert eid == episodes_v3.episode_id_for("traj-1")
    assert eid != "ep-foreign"
    assert len(_derivations(store, "ep-foreign")) == 1


def test_derivation_edge_producer_conflict_is_integrity(store, scope_id):
    """The identical edge under another producer is an INTEGRITY failure —
    derivation edges are immutable, never silently adopted (V3-17.02)."""
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id)
        eid = episodes_v3.episode_id_for("traj-1")
        conn.execute(
            "INSERT INTO derivations"
            " (child_kind, child_id, child_revision, parent_kind, parent_id,"
            "  parent_revision, producer_kind, producer_id, seq, scope_id)"
            " VALUES ('episode', ?, 1, 'trajectory', 'traj-1', 1,"
            "         'producer', 'other.builder', 0, ?)",
            (eid, scope_id),
        )
    # the failure propagates out of the caller's tx → the partial build
    # rolls back atomically (episode + members + edges all-or-nothing)
    with pytest.raises(VerbatimError) as exc:
        with store.tx() as conn:
            episodes_v3.build_episode(conn, "traj-1")
    assert exc.value.code == ErrorCode.INTEGRITY
    assert _episodes(store) == []


# ---------------------------------------------------------------------------
# refusals (incomplete evidence stays evidence)
# ---------------------------------------------------------------------------


def test_incomplete_trajectory_refused(store, scope_id):
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id, completed=False)
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            episodes_v3.build_episode(conn, "traj-1")
    assert exc.value.code == ErrorCode.VALIDATION
    assert _episodes(store) == []


def test_missing_trajectory_refused(store, scope_id):
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            episodes_v3.build_episode(conn, "traj-nope")
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


# ---------------------------------------------------------------------------
# outcome resolution (§12.03, §20.09, §13.05)
# ---------------------------------------------------------------------------


def test_failure_episode_retained_as_negative_experience(store, scope_id):
    """V3-20.09: a checker-attested failure still produces the episode."""
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id, outcome="failure")
        episode_id = episodes_v3.build_episode(conn, "traj-1")
    assert _ep_row(store, episode_id)["outcome"] == "failure"


def test_partial_outcome_recorded(store, scope_id):
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id, outcome="partial")
        eid = episodes_v3.build_episode(conn, "traj-1")
    assert _ep_row(store, eid)["outcome"] == "partial"


def test_outcome_from_checker_exit_code(store, scope_id):
    """No declared outcome: the receipt's exit_code derives it."""
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id, outcome=None)
    _set_metadata(store, "env-test", seed.checker_metadata(None, exit_code=1))
    with store.tx() as conn:
        eid = episodes_v3.build_episode(conn, "traj-1")
    assert _ep_row(store, eid)["outcome"] == "failure"


def test_no_checker_means_unknown_outcome(store, scope_id):
    """A test_result envelope WITHOUT an identified checker receipt is not
    outcome evidence — even one declaring success (V3-42.08)."""
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id, checker=False)
    _set_metadata(store, "env-test", {"outcome": "success"})
    with store.tx() as conn:
        eid = episodes_v3.build_episode(conn, "traj-1")
    assert _ep_row(store, eid)["outcome"] == "unknown"


def test_agent_claim_is_never_an_outcome(store, scope_id):
    """V3-12.03: an assistant_message saying it worked changes nothing."""
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id, checker=False)
        seed.seed_envelope(
            conn,
            scope_id,
            "env-claim",
            "assistant_message",
            metadata={"outcome": "success", "text": "it worked"},
            task_id="task-1",
            step_id="step-3",
        )
        conn.execute(
            "INSERT INTO step_observations (step_id, envelope_id, ord)"
            " VALUES ('step-3', 'env-claim', 1)"
        )
        eid = episodes_v3.build_episode(conn, "traj-1")
    assert _ep_row(store, eid)["outcome"] == "unknown"


def test_last_checker_verdict_wins(store, scope_id):
    """The final checker outcome in member order is the task outcome."""
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id, outcome="failure")
        seed.seed_envelope(
            conn,
            scope_id,
            "env-retest",
            "verification",
            metadata=seed.checker_metadata(
                "success", exit_code=0, invocation_id="inv-2"
            ),
            task_id="task-1",
            step_id="step-3",
        )
        conn.execute(
            "INSERT INTO step_observations (step_id, envelope_id, ord)"
            " VALUES ('step-3', 'env-retest', 1)"
        )
        eid = episodes_v3.build_episode(conn, "traj-1")
    assert _ep_row(store, eid)["outcome"] == "success"


def test_outcome_heals_forward_on_rebuild(store, scope_id):
    """A checker receipt arriving after the first build refines the
    episode outcome in place; identity is untouched (V3-17.03)."""
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id, checker=False)
        eid = episodes_v3.build_episode(conn, "traj-1")
    assert _ep_row(store, eid)["outcome"] == "unknown"
    _set_metadata(
        store, "env-test", seed.checker_metadata("success", exit_code=0)
    )
    with store.tx() as conn:
        assert episodes_v3.build_episode(conn, "traj-1") == eid
    assert _ep_row(store, eid)["outcome"] == "success"


# ---------------------------------------------------------------------------
# envelope_outcome / checker_receipt unit checks
# ---------------------------------------------------------------------------


def test_checker_receipt_requires_identified_checker():
    assert episodes_v3.checker_receipt({}) is None
    assert episodes_v3.checker_receipt({"checker": {}}) is None
    assert episodes_v3.checker_receipt({"checker": {"checker_id": ""}}) is None
    assert episodes_v3.checker_receipt({"checker": "pytest"}) is None
    receipt = episodes_v3.checker_receipt(seed.checker_metadata("success"))
    assert receipt is not None and receipt["checker_id"] == "pytest"
    md = {"checker_receipt": {"checker_id": "npm test"}}
    assert episodes_v3.checker_receipt(md)["checker_id"] == "npm test"
    # the SDK descriptor's plural list form resolves too
    md = {
        "checker_receipt": None,
        "checker_receipts": [{"checker_id": "mypy"}, {"checker_id": "ruff"}],
    }
    assert episodes_v3.checker_receipt(md)["checker_id"] == "mypy"
    assert episodes_v3.checker_receipt({"checker_receipts": [{}]}) is None


def test_envelope_outcome_values():
    assert episodes_v3.envelope_outcome({}) is None
    assert episodes_v3.envelope_outcome({"outcome": "success"}) is None
    md = seed.checker_metadata("partial")
    assert episodes_v3.envelope_outcome(md) == OutcomeClass.PARTIAL
    md = seed.checker_metadata(None, exit_code=0)
    assert episodes_v3.envelope_outcome(md) == OutcomeClass.SUCCESS
    md = seed.checker_metadata(None, exit_code=2)
    assert episodes_v3.envelope_outcome(md) == OutcomeClass.FAILURE
    md = seed.checker_metadata(None, exit_code=None, completed=False)
    assert episodes_v3.envelope_outcome(md) == OutcomeClass.UNKNOWN
    # bogus declared outcome falls back to receipt data
    md = seed.checker_metadata("ludicrous", exit_code=0)
    assert episodes_v3.envelope_outcome(md) == OutcomeClass.SUCCESS


# ---------------------------------------------------------------------------
# job handler (§40, V3-20.08)
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


def test_episode_build_job_builds_and_chains(store, scope_id, ingester):
    """episode_build → episode + auto-enqueued transition_build →
    transitions, all inside the durable job pipeline."""
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id)
    _enqueue(
        ingester,
        scope_id,
        JobKind.EPISODE_BUILD,
        {"trajectory_id": "traj-1"},
        "episode_build:traj-1",
    )
    # one drain runs episode_build then the chained transition_build job
    assert ingester.run_pending(scope=scope_id) == 2

    episode_id = episodes_v3.episode_id_for("traj-1")
    assert _ep_row(store, episode_id) is not None

    # the handler chained transition_build for the same episode
    tjobs = _jobs(store, "transition_build")
    assert len(tjobs) == 1
    assert episode_id in tjobs[0]["input_refs_json"]

    with store.read() as conn:
        rows = repos_v3.query(conn, "transitions", {"episode_id": episode_id})
    assert len(rows) == 3
    assert all(j["state"] == "succeeded" for j in _jobs(store))


def test_episode_build_job_rejects_missing_ref(store, scope_id, ingester):
    _enqueue(ingester, scope_id, JobKind.EPISODE_BUILD, {}, "op-bad")
    assert ingester.run_pending(scope=scope_id) == 1
    job = _jobs(store, "episode_build")[0]
    assert job["state"] == "failed"
    assert job["error_code"] == ErrorCode.VALIDATION.value


def test_episode_build_job_fails_on_incomplete(store, scope_id, ingester):
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id, completed=False)
    _enqueue(
        ingester,
        scope_id,
        JobKind.EPISODE_BUILD,
        {"trajectory_id": "traj-1"},
        "op-incomplete",
    )
    assert ingester.run_pending(scope=scope_id) == 1
    job = _jobs(store, "episode_build")[0]
    assert job["state"] == "failed"
    assert job["error_code"] == ErrorCode.VALIDATION.value


def test_episode_build_job_replayed_receipt(store, scope_id, ingester):
    """A committed operation receipt records the derived episode id —
    redelivery replays it instead of re-deriving (V2-39.10 convention)."""
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id)
    _enqueue(
        ingester,
        scope_id,
        JobKind.EPISODE_BUILD,
        {"trajectory_id": "traj-1"},
        "episode_build:traj-1",
    )
    assert ingester.run_pending(scope=scope_id) == 2
    episode_id = episodes_v3.episode_id_for("traj-1")
    with store.read() as conn:
        rows = _dicts(
            conn.execute(
                "SELECT * FROM operations WHERE effect_kind = 'episode_build'"
            )
        )
    assert len(rows) == 1
    assert episode_id in rows[0]["receipt_json"]
