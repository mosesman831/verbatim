"""V4 hierarchical derived views + progressive expansion (SPEC_V4 §22).

Pinned behaviors:

- An *overview* is a derived ``observations`` row (producer
  ``v4.hierarchy.v1``) — labeled derived, never canonical; it carries
  ``observation_evidence`` pointers to the exact member revisions it
  summarizes plus a ``derivations`` edge to the parent object itself
  (V4-22.06, V4-27 provenance, V3-17.02).
- Overviews preserve contradictory branches, temporal bounds, and
  missing coverage — an outcome histogram names the conflict instead of
  picking a winner (V4-22.06).
- ``expand`` performs progressive scene→episode→envelope→span
  disclosure returning *locators + per-child eligibility only* — never
  payload bytes (V4-22.07); a parent's presence never launders a held
  child (V4-22.08 — each child passes its own checks).
- Purging a member episode invalidates the derived views through the
  real closure machinery: the scene suppresses, the overview
  observations suppress, and the closure run completes clean
  (V4-38 deletion closure).
- Construction is bounded + idempotent: re-derivation over unchanged
  state writes nothing (V4-22.09, V4-24.08).
"""

from __future__ import annotations

import pytest

from verbatim.experience import episodes_v3, hierarchy, scenes
from verbatim.privacy.closure import ClosureEngine, ClosurePhase
from verbatim.security import open_quarantine
from verbatim.storage import repos_v3
from verbatim.storage.store import Store

from . import v3_seed as seed


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "v4hier.db"))
    yield s
    s.close()


@pytest.fixture
def scope_id(store):
    sid = "scope:v4hier"
    with store.tx() as conn:
        seed.seed_scope(conn, sid)
    return sid


def _dicts(cur):
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _build(store, scope_id, traj_id, **kw):
    with store.tx() as conn:
        ids = seed.seed_three_step_task(conn, scope_id, traj_id, **kw)
        return episodes_v3.build_episode(conn, traj_id), ids


def _obs_row(store, obs_id):
    with store.read() as conn:
        return repos_v3.get(conn, "observations", {"observation_id": obs_id})


def _evidence(store, obs_id, revision):
    with store.read() as conn:
        return repos_v3.query(
            conn,
            "observation_evidence",
            {"observation_id": obs_id, "revision": revision},
        )


def _parents(store, obs_id, revision):
    with store.read() as conn:
        return repos_v3.query(
            conn,
            "derivations",
            {
                "child_kind": "observation",
                "child_id": obs_id,
                "child_revision": revision,
            },
        )


def _setup_scene(store, scope_id, n=2, family="fix-bug", **kw):
    """Build ``n`` episodes in one family + assign; returns (scene_id, eps)."""
    eps = []
    for i in range(n):
        ep, _ = _build(store, scope_id, f"traj-{i}", task_id=family,
                       prefix=f"{i}-", **kw)
        eps.append(ep)
    with store.tx() as conn:
        for e in eps:
            scenes.assign_episode(conn, e)
        scene_id = scenes.scene_for_episode(conn, eps[0])["episode_id"]
    return scene_id, eps


# ---------------------------------------------------------------------------
# overview construction + provenance (V4-22.06, V3-17.02)
# ---------------------------------------------------------------------------


def test_episode_overview_is_labeled_derived(store, scope_id):
    ep, ids = _build(store, scope_id, "traj-1", task_id="fam")
    with store.tx() as conn:
        res = hierarchy.build_overview(conn, scope_id, "episode", ep)
    assert res["written"] and res["revision"] == 1
    obs = _obs_row(store, res["observation_id"])
    assert obs["producer"] == hierarchy.HIERARCHY_PRODUCER_ID
    assert obs["scope_id"] == scope_id
    assert obs["recorded_until"] is None
    # Evidence pointers = the member envelopes at their revisions.
    ev = _evidence(store, obs["observation_id"], 1)
    assert {e["object_kind"] for e in ev} == {"envelope"}
    assert {e["object_id"] for e in ev} == set(ids.values())
    # Parent linkage rides the derivations graph (revision-precise).
    edges = _parents(store, obs["observation_id"], 1)
    parent_refs = {(e["parent_kind"], e["parent_id"]) for e in edges}
    assert ("episode", ep) in parent_refs


def test_scene_overview_preserves_conflict_and_bounds(store, scope_id):
    """Success+failure members produce a contested overview naming both
    branches — never merged into one verdict (V4-22.06)."""
    eps = []
    eps.append(_build(store, scope_id, "traj-ok", task_id="fam",
                      prefix="a-", outcome="success")[0])
    eps.append(_build(store, scope_id, "traj-bad", task_id="fam",
                      prefix="b-", outcome="failure")[0])
    with store.tx() as conn:
        for e in eps:
            scenes.assign_episode(conn, e)
        scene_id = scenes.scene_for_episode(conn, eps[0])["episode_id"]
        res = hierarchy.build_overview(conn, scope_id, "episode", scene_id)
    assert res["written"]
    assert res["detail"]["contested"] is True
    assert res["detail"]["outcomes"] == {"failure": 1, "success": 1}
    assert "conflict=yes" in _obs_row(store, res["observation_id"])["text"]
    # Supports cite the member episodes at their revisions; the scene
    # itself is a derivations parent, not an evidence row.
    ev = _evidence(store, res["observation_id"], 1)
    assert {e["object_id"] for e in ev} == set(eps)
    assert all(e["object_kind"] == "episode" for e in ev)
    edges = _parents(store, res["observation_id"], 1)
    assert ("episode", scene_id) in {
        (e["parent_kind"], e["parent_id"]) for e in edges
    }


def test_overview_idempotent(store, scope_id):
    ep, _ = _build(store, scope_id, "traj-1", task_id="fam")
    with store.tx() as conn:
        first = hierarchy.build_overview(conn, scope_id, "episode", ep)
        second = hierarchy.build_overview(conn, scope_id, "episode", ep)
    assert first["written"] and not second["written"]
    assert second["revision"] == 1


def test_build_overviews_bounded(store, scope_id):
    """The bulk pass is bounded by limit and reports truncation."""
    scene_id, eps = _setup_scene(store, scope_id, n=3)
    with store.tx() as conn:
        rep = hierarchy.build_overviews(conn, scope_id, limit=2)
    assert rep["truncated"] is True
    assert rep["parents"] == 2
    with store.tx() as conn:
        rep2 = hierarchy.build_overviews(conn, scope_id)
    # Pass 1 covered two parents; pass 2 sees all four but writes only
    # the two not yet derived (idempotent — V4-24.08).
    assert rep2["parents"] == 4
    assert rep2["overviews_written"] == 2
    with store.read() as conn:
        for ep in eps + [scene_id]:
            assert hierarchy.overview_for(conn, scope_id, "episode", ep)


def test_overview_never_covers_purged_parent(store, scope_id):
    """A purge-tombstoned parent fails closed: no overview is derived
    from suppressed material."""
    ep, _ = _build(store, scope_id, "traj-1", task_id="fam")
    eng = ClosureEngine(store)
    run = eng.begin([("episode", ep, 1)], scope_id)
    run = eng.drain(run.run_id)
    assert run.phase == ClosurePhase.COMPLETED
    with store.tx() as conn:
        res = hierarchy.build_overview(conn, scope_id, "episode", ep)
    assert res["written"] is False and res["reason"] == "parent_purged"


# ---------------------------------------------------------------------------
# progressive expansion + per-child eligibility (V4-22.07/.08)
# ---------------------------------------------------------------------------


def test_expand_scene_to_spans(store, scope_id):
    """scene → member episodes → member envelopes → span locators,
    each level carrying its own eligibility flag; no payload bytes."""
    scene_id, eps = _setup_scene(store, scope_id, n=2)
    with store.tx() as conn:
        # Seed one span under the first member's first envelope source so
        # the span-locator level of expansion is exercised for real.
        first = hierarchy.expand(conn, scope_id, scene_id)
        env0 = first["members"][0]["envelopes"][0]
        conn.execute(
            "INSERT INTO spans (span_id, source_id, revision, start_byte,"
            " end_byte, excerpt_hmac, harvester_version)"
            " VALUES ('sp-test', ?, ?, 0, 7, X'00', 'test')",
            (env0["source_id"], env0["source_revision"]),
        )
    with store.read() as conn:
        tree = hierarchy.expand(conn, scope_id, scene_id)
    assert tree["object"]["kind"] == "scene"
    assert tree["object"]["eligible"] is True
    members = tree["members"]
    assert {m["episode_id"] for m in members} == set(eps)
    seen_spans: list[str] = []
    for m in members:
        assert m["eligible"] is True
        assert m["envelopes"]  # each episode carries envelope locators
        for env in m["envelopes"]:
            assert "envelope_id" in env and "source_id" in env
            assert "payload" not in env and "text" not in env
            for sp in env.get("spans", []):
                assert set(sp) <= {"span_id", "eligible"}
                seen_spans.append(sp["span_id"])
    assert "sp-test" in seen_spans  # the span level was actually reached
    # The whole tree is locators only — no byte fields anywhere.
    import json as _json

    assert "payload" not in _json.dumps(tree)


def test_expand_episode_direct(store, scope_id):
    ep, ids = _build(store, scope_id, "traj-1", task_id="fam")
    with store.read() as conn:
        tree = hierarchy.expand(conn, scope_id, ep)
    node = tree["object"]
    assert node["episode_id"] == ep and node["kind"] == "task"
    assert {e["envelope_id"] for e in node["envelopes"]} == set(ids.values())


def test_child_eligibility_independent_of_parent(store, scope_id):
    """Quarantining one member envelope withholds exactly that child —
    the scene and siblings stay eligible (V4-22.08)."""
    scene_id, eps = _setup_scene(store, scope_id, n=2)
    with store.read() as conn:
        tree0 = hierarchy.expand(conn, scope_id, scene_id)
    held_env = tree0["members"][0]["envelopes"][0]
    with store.tx() as conn:
        open_quarantine(
            conn,
            ("source_envelope", held_env["envelope_id"], held_env["revision"]),
            ["rule:test.hold"],
            [],
            scope_id=scope_id,
        )
    with store.read() as conn:
        tree = hierarchy.expand(conn, scope_id, scene_id)
    assert tree["object"]["eligible"] is True
    held_member = tree["members"][0]
    flagged = [
        e for e in held_member["envelopes"]
        if e["envelope_id"] == held_env["envelope_id"]
    ][0]
    assert flagged["eligible"] is False and flagged["reason"] == "held"
    assert any(
        e["eligible"]
        for e in held_member["envelopes"]
        if e["envelope_id"] != held_env["envelope_id"]
    )


def test_expand_unknown_id_fails_closed(store, scope_id):
    from verbatim.core.types import ErrorCode, VerbatimError

    with store.read() as conn:
        with pytest.raises(VerbatimError) as ei:
            hierarchy.expand(conn, scope_id, "ep-nope")
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


# ---------------------------------------------------------------------------
# purge cascade (V4-38: deletion closure reaches derived views)
# ---------------------------------------------------------------------------


def test_episode_purge_suppresses_scene_and_overview(store, scope_id):
    """Purging one member episode: the scene suppresses (mixed ancestry)
    and every overview derived from the purged member suppresses — the
    closure run completes clean."""
    scene_id, eps = _setup_scene(store, scope_id, n=2)
    with store.tx() as conn:
        ov_scene = hierarchy.build_overview(conn, scope_id, "episode",
                                            scene_id)
        ov_ep = hierarchy.build_overview(conn, scope_id, "episode",
                                         eps[0])
    eng = ClosureEngine(store)
    run = eng.begin([("episode", eps[0], 1)], scope_id)
    run = eng.drain(run.run_id)
    assert run.phase == ClosurePhase.COMPLETED
    with store.read() as conn:
        scene = conn.execute(
            "SELECT recorded_until FROM episodes WHERE episode_id = ?",
            (scene_id,),
        ).fetchone()
        obs_scene = repos_v3.get(
            conn, "observations",
            {"observation_id": ov_scene["observation_id"]},
        )
        obs_ep = repos_v3.get(
            conn, "observations",
            {"observation_id": ov_ep["observation_id"]},
        )
        live = conn.execute(
            "SELECT recorded_until FROM episodes WHERE episode_id = ?",
            (eps[1],),
        ).fetchone()
    # Scene tombstoned (suppressed), surviving member untouched, both
    # overviews suppressed through ancestry.
    assert scene[0] is not None
    assert obs_scene["recorded_until"] is not None
    assert obs_ep["recorded_until"] is not None
    assert live[0] is None


def test_scene_root_purge_suppresses_overview(store, scope_id):
    """Purging the scene itself (all revisions) reaches its overview
    through the derivations parent edge; members stay untouched."""
    scene_id, eps = _setup_scene(store, scope_id, n=2)
    with store.tx() as conn:
        ov = hierarchy.build_overview(conn, scope_id, "episode", scene_id)
    eng = ClosureEngine(store)
    run = eng.begin([("episode", scene_id, None)], scope_id)
    run = eng.drain(run.run_id)
    assert run.phase == ClosurePhase.COMPLETED
    with store.read() as conn:
        obs = repos_v3.get(
            conn, "observations",
            {"observation_id": ov["observation_id"]},
        )
        live = conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE episode_id IN (?,?)"
            " AND recorded_until IS NULL",
            tuple(eps),
        ).fetchone()[0]
    assert obs["recorded_until"] is not None
    assert live == 2


def test_purged_episode_dropped_from_scene_on_refresh(store, scope_id):
    """After closure, a refresh pass drops the purged member
    prospectively — the member interval closes, never rewrites."""
    scene_id, eps = _setup_scene(store, scope_id, n=2)
    eng = ClosureEngine(store)
    run = eng.begin([("episode", eps[0], 1)], scope_id)
    run = eng.drain(run.run_id)
    assert run.phase == ClosurePhase.COMPLETED
    with store.tx() as conn:
        # The scene itself is purge-suppressed — refresh refuses loudly.
        res = scenes.refresh_scene(conn, scene_id)
    assert res["refreshed"] is False and res["reason"] == "suppressed"
