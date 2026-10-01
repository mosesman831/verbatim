"""V4 task-family scenes over v3 episodes (SPEC_V4 §22).

Pinned behaviors:

- Scenes are ``episodes`` rows with ``kind='scene'`` whose membership
  lives in ``episode_members`` — no parallel scene table; the scene's
  ``revision`` IS its membership revision (V4-22.05).
- Grouping is *declared*, never similarity-inferred: the family key comes
  from trajectory ``metadata.task_family``, else ``task_id``, else the
  episode label/host_task_id; no declared family → ungrouped, and an
  episode can never cross scope boundaries into another scope's scene
  (V4-22.05 — grouping never broadens an audience).
- Membership is revisioned and evidence-linked: every membership change
  bumps the scene revision and records the member set as immutable
  derivations edges at that revision; member rows keep their own
  recorded interval, so removal is prospective (V4-22.05, V3-17.02).
- Conflicting members coexist — a failure episode stays in the scene
  beside its successful siblings (V4-22.04 negative experience).
- Construction is incremental, bounded, and resumable (V4-22.09):
  ``build_scenes`` returns cursors + ``truncated`` honestly.
- Lookup surface (V4-22.07): exact episode→scene, task-family search,
  and temporal episode sequence.
"""

from __future__ import annotations

import pytest

from verbatim.config import VerbatimConfig
from verbatim.core.types import ErrorCode, JobKind, VerbatimError
from verbatim.experience import episodes_v3, hierarchy, scenes
from verbatim.ingest import Ingester
from verbatim.storage.store import Store

from . import v3_seed as seed


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "v4scenes.db"))
    yield s
    s.close()


@pytest.fixture
def ingester(store):
    return Ingester(store, VerbatimConfig())


@pytest.fixture
def scope_id(store):
    sid = "scope:v4scenes"
    with store.tx() as conn:
        seed.seed_scope(conn, sid)
    return sid


def _dicts(cur):
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _build(store, scope_id, traj_id, **kw):
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id, traj_id, **kw)
        return episodes_v3.build_episode(conn, traj_id)


def _ep(store, episode_id):
    with store.read() as conn:
        rows = _dicts(
            conn.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
            )
        )
    return rows[0] if rows else None


def _member_edges(store, scene_id, revision):
    with store.read() as conn:
        return _dicts(
            conn.execute(
                "SELECT * FROM derivations WHERE child_kind = 'episode'"
                " AND child_id = ? AND child_revision = ?"
                " ORDER BY parent_id",
                (scene_id, revision),
            )
        )


# ---------------------------------------------------------------------------
# attachment + family grouping (V4-22.05)
# ---------------------------------------------------------------------------


def test_same_family_episodes_share_one_scene(store, scope_id):
    """Two episodes whose trajectories share a task_id land in the same
    deterministic scene; members attach through episode_members."""
    e1 = _build(store, scope_id, "traj-1", task_id="fix-bug", prefix="a-")
    e2 = _build(store, scope_id, "traj-2", task_id="fix-bug", prefix="b-")
    with store.tx() as conn:
        r1 = scenes.assign_episode(conn, e1)
        r2 = scenes.assign_episode(conn, e2)
    assert r1["assigned"] and r1["changed"]
    assert r2["assigned"] and r2["changed"]
    assert r1["scene_id"] == r2["scene_id"]
    assert r1["scene_id"] == scenes.scene_id_for(scope_id, "fix-bug")
    with store.read() as conn:
        members = scenes.scene_members(conn, r1["scene_id"])
    assert {m["episode_id"] for m in members} == {e1, e2}


def test_declared_task_family_metadata_wins(store, scope_id):
    """metadata.task_family on the trajectory outranks task_id
    (V4-22.05 — the host's declared family is the grouping rule)."""
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id, "traj-f", task_id="raw-id")
        # re-declare the family explicitly on the trajectory row
        conn.execute(
            "UPDATE trajectories SET metadata_json = ?"
            " WHERE trajectory_id = 'traj-f'",
            ('{"task_family": "db-migrations"}',),
        )
        ep = episodes_v3.build_episode(conn, "traj-f")
        res = scenes.assign_episode(conn, ep)
    assert res["family_key"] == "db-migrations"
    assert res["scene_id"] == scenes.scene_id_for(scope_id, "db-migrations")


def test_no_declared_family_stays_ungrouped(store, scope_id):
    """An episode with no task_family, no task_id, and no label has no
    justifiable scene — it is never force-grouped (V4-22.05)."""
    e = _build(store, scope_id, "traj-x", task_id="", prefix="x-")
    with store.tx() as conn:
        res = scenes.assign_episode(conn, e)
    assert res == {"assigned": False, "reason": "no_declared_family"}
    with store.read() as conn:
        assert scenes.scenes_for_scope(conn, scope_id) == []


def test_scene_of_scene_rejected(store, scope_id):
    e = _build(store, scope_id, "traj-1", task_id="fix-bug")
    with store.tx() as conn:
        scenes.assign_episode(conn, e)
        scene_id = scenes.scene_for_episode(conn, e)["episode_id"]
        with pytest.raises(VerbatimError) as ei:
            scenes.assign_episode(conn, scene_id)
    assert ei.value.code == ErrorCode.VALIDATION


def test_scenes_never_cross_scope(store):
    """Same family key in two scopes → two distinct scenes; membership
    is bounded by the member's own scope (V4-22.05 no audience widening)."""
    s1, s2 = "scope:A", "scope:B"
    with store.tx() as conn:
        seed.seed_scope(conn, s1)
        seed.seed_scope(conn, s2)
        seed.seed_three_step_task(conn, s1, "t-a", task_id="fam", prefix="a-")
        seed.seed_three_step_task(conn, s2, "t-b", task_id="fam", prefix="b-")
        ea = episodes_v3.build_episode(conn, "t-a")
        eb = episodes_v3.build_episode(conn, "t-b")
        ra = scenes.assign_episode(conn, ea)
        rb = scenes.assign_episode(conn, eb)
    assert ra["scene_id"] != rb["scene_id"]
    with store.read() as conn:
        ma = scenes.scene_members(conn, ra["scene_id"])
        mb = scenes.scene_members(conn, rb["scene_id"])
    assert [m["episode_id"] for m in ma] == [ea]
    assert [m["episode_id"] for m in mb] == [eb]


# ---------------------------------------------------------------------------
# membership revisioning + derivations (V4-22.05, V3-17.02)
# ---------------------------------------------------------------------------


def test_membership_revision_history(store, scope_id):
    """Each membership change bumps the scene revision and writes the
    member edges AT that revision — history is reconstructable."""
    e1 = _build(store, scope_id, "traj-1", task_id="fam", prefix="a-")
    e2 = _build(store, scope_id, "traj-2", task_id="fam", prefix="b-")
    with store.tx() as conn:
        r1 = scenes.assign_episode(conn, e1)
        r2 = scenes.assign_episode(conn, e2)
    scene_id = r1["scene_id"]
    assert r1["revision"] == 2  # creation rev 1 + first membership
    assert r2["revision"] == 3
    # rev-2 edges: only e1; rev-3 edges: e1 + e2 — the graph itself is
    # the membership history.
    rev2 = _member_edges(store, scene_id, 2)
    rev3 = _member_edges(store, scene_id, 3)
    assert {e["parent_id"] for e in rev2} == {e1}
    assert {e["parent_id"] for e in rev3} == {e1, e2}
    assert all(e["producer_id"] == scenes.SCENE_PRODUCER_ID for e in rev3)


def test_remove_member_is_prospective(store, scope_id):
    """Removing a member closes its interval prospectively — the row
    keeps recorded_from and gains recorded_until; the scene revision
    bumps again (V4-22.05 revisable membership)."""
    e1 = _build(store, scope_id, "traj-1", task_id="fam", prefix="a-")
    e2 = _build(store, scope_id, "traj-2", task_id="fam", prefix="b-")
    with store.tx() as conn:
        scenes.assign_episode(conn, e1)
        r2 = scenes.assign_episode(conn, e2)
        scene_id = r2["scene_id"]
        rm = scenes.remove_member(conn, scene_id, e1)
    assert rm["removed"] and rm["revision"] == 4
    with store.read() as conn:
        members_all = scenes.scene_members(conn, scene_id, include_closed=True)
        members_live = scenes.scene_members(conn, scene_id)
    assert len(members_all) == 2 and len(members_live) == 1
    closed = [m for m in members_all if m["episode_id"] == e1][0]
    assert closed["member_recorded_until"] is not None
    assert closed["member_recorded_from"] is not None
    # rev-4 derivations name only the surviving member.
    rev4 = _member_edges(store, scene_id, 4)
    assert {e["parent_id"] for e in rev4} == {e2}


def test_reassign_is_idempotent(store, scope_id):
    e = _build(store, scope_id, "traj-1", task_id="fam")
    with store.tx() as conn:
        scenes.assign_episode(conn, e)
        again = scenes.assign_episode(conn, e)
    assert again["assigned"] and again["changed"] is False


def test_conflicting_members_coexist(store, scope_id):
    """A failed episode stays in the family scene beside a successful
    sibling — scenes never merge or erase conflict (V4-22.04/22.05)."""
    e1 = _build(store, scope_id, "traj-ok", task_id="fam", prefix="a-",
                outcome="success")
    e2 = _build(store, scope_id, "traj-bad", task_id="fam", prefix="b-",
                outcome="failure")
    with store.tx() as conn:
        scenes.assign_episode(conn, e1)
        scenes.assign_episode(conn, e2)
    with store.read() as conn:
        members = scenes.scene_members(
            conn, scenes.scene_for_episode(conn, e1)["episode_id"]
        )
    assert {m["episode_id"]: m["outcome"] for m in members} == {
        e1: "success",
        e2: "failure",
    }


# ---------------------------------------------------------------------------
# bounded incremental construction (V4-22.09)
# ---------------------------------------------------------------------------


def test_build_scenes_bounded_and_resumable(store, scope_id):
    """build_scenes processes a bounded slice per call and returns a
    resumption cursor; a second pass over settled state writes nothing."""
    eps = [
        _build(store, scope_id, f"traj-{i}", task_id="fam", prefix=f"{i}-")
        for i in range(3)
    ]
    with store.tx() as conn:
        rep = scenes.build_scenes(conn, scope_id, limit=2)
    assert rep["assigned"] == 2 and rep["truncated"]
    cursor = rep["cursor"]
    with store.tx() as conn:
        rep2 = scenes.build_scenes(conn, scope_id, limit=8, cursor=cursor)
    assert rep2["assigned"] == 1
    with store.read() as conn:
        assert len(scenes.scenes_for_scope(conn, scope_id)) == 1
        members = scenes.scene_members(
            conn, scenes.scene_for_episode(conn, eps[0])["episode_id"]
        )
    assert {m["episode_id"] for m in members} == set(eps)
    # Idempotent re-pass: nothing new to assign, nothing to drop.
    with store.tx() as conn:
        rep3 = scenes.build_scenes(conn, scope_id)
    assert rep3["assigned"] == 0 and rep3["members_dropped"] == 0
    assert rep3["scenes_created"] == 0


def test_build_scenes_disabled_fails_loud(store, scope_id):
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as ei:
            scenes.build_scenes(conn, scope_id, enabled=False)
    assert ei.value.code == ErrorCode.CAPABILITY_UNAVAILABLE


def test_refresh_drops_closed_members(store, scope_id):
    """A member whose episode row is closed drops out prospectively; an
    emptied scene is retired, not deleted (V4-22.09)."""
    e1 = _build(store, scope_id, "traj-1", task_id="fam")
    with store.tx() as conn:
        scene_id = scenes.assign_episode(conn, e1)["scene_id"]
        conn.execute(
            "UPDATE episodes SET recorded_until = 999 WHERE episode_id = ?",
            (e1,),
        )
        res = scenes.refresh_scene(conn, scene_id)
    assert res["changed"] and res["dropped"] == [e1]
    scene = _ep(store, scene_id)
    assert scene["recorded_until"] is not None  # retired, still present


# ---------------------------------------------------------------------------
# lookup surface (V4-22.07)
# ---------------------------------------------------------------------------


def test_exact_family_and_sequence_queries(store, scope_id):
    e1 = _build(store, scope_id, "traj-1", task_id="fam-a", prefix="a-")
    e2 = _build(store, scope_id, "traj-2", task_id="fam-a", prefix="b-")
    e3 = _build(store, scope_id, "traj-3", task_id="fam-b", prefix="c-")
    with store.tx() as conn:
        # Pin distinct recorded_from so the temporal sequence is real
        # (seeded builds share MAX(events)+1 without event writes).
        for i, e in enumerate((e1, e2, e3), start=1):
            conn.execute(
                "UPDATE episodes SET recorded_from = ? WHERE episode_id = ?",
                (100 + i, e),
            )
        for e in (e1, e2, e3):
            scenes.assign_episode(conn, e)
        fam_a = scenes.episodes_for_family(conn, scope_id, "fam-a")
        fam_b = scenes.episodes_for_family(conn, scope_id, "fam-b")
        seq_all = scenes.episode_sequence(conn, scope_id)
        seq_a = scenes.episode_sequence(conn, scope_id, family_key="fam-a")
    assert {m["episode_id"] for m in fam_a} == {e1, e2}
    assert [m["episode_id"] for m in fam_b] == [e3]
    # Temporal sequence: creation order, scenes excluded.
    assert [r["episode_id"] for r in seq_all] == [e1, e2, e3]
    assert [m["episode_id"] for m in seq_a] == [e1, e2]
    with store.read() as conn:
        # Exact episode→scene lookup (V4-22.07).
        assert scenes.scene_for_episode(conn, e2)["episode_id"] == (
            scenes.scene_id_for(scope_id, "fam-a")
        )
        assert scenes.trajectory_of_episode(conn, e1) == "traj-1"


def test_assign_validates_ids(store, scope_id):
    with store.tx() as conn:
        with pytest.raises(VerbatimError):
            scenes.assign_episode(conn, "ep-nonexistent")
        with pytest.raises(VerbatimError):
            scenes.episodes_for_family(conn, scope_id, "   ")


# ---------------------------------------------------------------------------
# durable-job chain: episode_build → scene assign + overview (V4-22.09, §42)
# ---------------------------------------------------------------------------


def test_episode_build_job_chains_scene_assignment(store, scope_id, ingester):
    """The episode_build job assigns the new episode to its family scene
    and refreshes the episode + scene overviews inline — all inside the
    same fenced transaction, before the chained transition_build."""
    with store.tx() as conn:
        seed.seed_three_step_task(conn, scope_id, "traj-1",
                                  task_id="fix-bug")
    with store.tx() as conn:
        ingester.jobs.enqueue(
            conn, scope_id, JobKind.EPISODE_BUILD,
            {"trajectory_id": "traj-1"},
        )
    assert ingester.run_pending(scope=scope_id) == 2
    ep = episodes_v3.episode_id_for("traj-1")
    with store.read() as conn:
        scene = scenes.scene_for_episode(conn, ep)
        assert scene is not None
        assert scene["episode_id"] == scenes.scene_id_for(
            scope_id, "fix-bug"
        )
        # Both overviews derived inline by the same job.
        assert hierarchy.overview_for(conn, scope_id, "episode", ep)
        assert hierarchy.overview_for(
            conn, scope_id, "episode", scene["episode_id"]
        )
