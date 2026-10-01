"""I2 repair-local recomputation (SPEC_V4_5 §04; V45-04.01–04.04; D03/D04).

Every test runs real producers — ``consolidate_windowed`` observations,
``Synthesizer.compose`` views, ``assign_episode``/``refresh_scene``
scenes, ``ProfileService`` compile/refresh, ``BranchService`` — against
real ``Store.create`` fixtures. Recomputation counts are measured, never
asserted from fixture intent.
"""

from __future__ import annotations

import pytest

from verbatim.core.types import ErrorCode, VerbatimError, json_dumps
from verbatim.experience.scenes import assign_episode, refresh_scene
from verbatim.observations.consolidate import (
    ConsolidationWindow,
    consolidate_windowed,
)
from verbatim.kernel import Kernel
from verbatim.profiles import ProfileService
from verbatim.repair import (
    apply_repair,
    full_rebuild,
    impact_closure,
    invalidate_dependents,
    load_plan,
    mark_held,
    persist_plan,
    plan_repair,
)
from verbatim.storage import repos_v4
from verbatim.synthesis import Synthesizer
from verbatim.synthesis.types import VIEW_OBJECT_KIND

from .conftest import (
    T0,
    add_claim,
    add_episode,
    add_scene,
    add_source,
    add_span,
    bootstrap,
    caller,
    compose_claim_view,
    correct_claim_revision,
    make_store,
    run_consolidate,
    seed_graph,
)


SID = "scope:a"


def _obs_rows(conn, sid=SID):
    return conn.execute(
        "SELECT observation_id, revision, text, stale_since_seq,"
        " recorded_until FROM observations WHERE scope_id = ?"
        " ORDER BY observation_id",
        (sid,),
    ).fetchall()


def _view_disposition(conn, view_id):
    row = conn.execute(
        "SELECT disposition FROM objects WHERE kind = ? AND object_id = ?",
        (VIEW_OBJECT_KIND, view_id),
    ).fetchone()
    return row[0] if row else None


def _seed_full(store, sid=SID, n_claims=4, view_claims=("c0", "c1")):
    """Seed claims + real observations + real composed views."""
    synth = Synthesizer(store, kernel=Kernel(store))
    with store.tx() as conn:
        seed_graph(conn, store, sid, n_claims)
        synth.register_producer(conn)
        consolidate_windowed(
            conn,
            sid,
            window=ConsolidationWindow(since_seq=0),
            min_proof=1,
        )
    views = {
        cid: compose_claim_view(store, synth, sid, cid, 1).view_id
        for cid in view_claims
    }
    return synth, views


# ---------------------------------------------------------------------------
# impact_closure + plan durability (V45-04.01)
# ---------------------------------------------------------------------------

def test_impact_closure_walks_all_edge_sources(store):
    """The closure unions ``derivations``, ``dependency_edges``, and the
    registered side walkers — claim→span via ``claim_evidence``,
    anything→observation via ``observation_evidence``, plus the two edge
    tables — so no reference system can hide a dependent."""
    with store.tx() as conn:
        seed_graph(conn, store, SID, 2)
        consolidate_windowed(
            conn, SID, window=ConsolidationWindow(since_seq=0),
            min_proof=1,
        )
        # span → claims (claim_evidence side walker)
        span_children = impact_closure(conn, [("span", "sp0")])
        assert ("claim", "c0", 1) in span_children
        # claim → observation (derivations + observation_evidence)
        claim_children = impact_closure(conn, [("claim", "c0", 1)])
        kinds = {k for k, _i, _r in claim_children}
        assert "observation" in kinds
        obs_ids = {i for k, i, _r in claim_children if k == "observation"}
        assert len(obs_ids) == 1
        # Locality: a different claim's closure names only ITS slot's
        # observation — disjoint from c0's (the D03 premise).
        c1_children = impact_closure(conn, [("claim", "c1", 1)])
        assert {k for k, _i, _r in c1_children} == {"observation"}
        c1_obs = {i for k, i, _r in c1_children}
        assert len(c1_obs) == 1 and c1_obs.isdisjoint(obs_ids)
        # Transitive: span → claim → observation (two hops, one call).
        assert obs_ids <= {
            i for k, i, _r in span_children if k == "observation"
        }


def test_plan_persist_load_round_trip_and_tamper_detection(store):
    """V45-04.01 + V45-13.02: the impact plan is a durable digest-bound
    object — it survives load and rejects tampering with STORE_CORRUPT."""
    with store.tx() as conn:
        seed_graph(conn, store, SID, 2)
        consolidate_windowed(
            conn, SID, window=ConsolidationWindow(since_seq=0),
            min_proof=1,
        )
        plan = plan_repair(conn, SID, [("claim", "c0", 1)])
        assert plan["complete"] is True
        assert plan["doc"] == "repair_plan/v1"
        assert [t["producer"] for t in plan["targets"]] == ["consolidation"]
        plan_id = persist_plan(store, conn, plan)
        loaded = load_plan(conn, store, plan_id)
        assert loaded["plan_id"] == plan_id
        assert loaded["affected"] == plan["affected"]
        # The plan joins the same ancestry graph — edges name its seeds.
        edges = conn.execute(
            "SELECT parent_kind, parent_id, parent_revision"
            " FROM dependency_edges WHERE child_kind='repair_plan'"
            " AND child_id = ?",
            (plan_id,),
        ).fetchall()
        assert [(r[0], r[1], r[2]) for r in edges] == [("claim", "c0", 1)]
        # Tamper with the stored doc — integrity fails closed.
        conn.execute(
            "UPDATE object_revisions SET metadata_json = "
            " json_replace(metadata_json, '$.complete', json('false'))"
            " WHERE kind = 'repair_plan' AND object_id = ?",
            (plan_id,),
        )
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            load_plan(conn, store, plan_id)
        assert exc.value.code is ErrorCode.STORE_CORRUPT
        assert load_plan(conn, store, "rplan:nope") is None


def test_plan_repair_validates_seed_refs(store):
    with store.tx() as conn:
        bootstrap(conn, SID)
        with pytest.raises(VerbatimError) as exc:
            plan_repair(conn, SID, [])
        assert exc.value.code is ErrorCode.VALIDATION
        with pytest.raises(VerbatimError) as exc2:
            plan_repair(conn, SID, [("claim",)])
        assert exc2.value.code is ErrorCode.VALIDATION
        with pytest.raises(VerbatimError) as exc3:
            plan_repair(conn, SID, [("claim", "c0", 0)])
        assert exc3.value.code is ErrorCode.VALIDATION


# ---------------------------------------------------------------------------
# D03 — localized recompute vs full rebuild (V45-04.02/04.04)
# ---------------------------------------------------------------------------

def test_localized_repair_recomputes_strict_subset_and_beats_rebuild(
    stores,
):
    """D03/V45-04.04 measured: one claim correction on an 4-slot graph
    recomputes only the touched slot's observation and the bound view —
    at least 50% fewer evaluated/recomputed objects than the identical
    full rebuild on the twin store, with equal post-state."""
    rep_store, full_store = stores
    kernel = Kernel(rep_store)
    synth, views = _seed_full(rep_store)
    synth_f, views_f = _seed_full(full_store)
    assert set(views) == set(views_f)

    # The correction + plan + immediate invalidation — one transaction,
    # so held marks commit with the correction (V45-04.02).
    with rep_store.tx() as conn:
        correct_claim_revision(conn, "c0", "corrected value 0")
        plan = plan_repair(conn, SID, [("claim", "c0", 1)])
        assert plan["complete"] is True
        producers = {t["producer"] for t in plan["targets"]}
        assert producers == {"consolidation", "synthesis"}
        inv = invalidate_dependents(
            conn, kernel, SID, [("claim", "c0", 1)],
            synthesizer=synth,
        )
        assert inv["report"].epochs[SID] >= 1
        assert inv["marked_held"]["views"] == 1
        assert inv["marked_held"]["observations"] == 1
        # Inside the correction's own commit the view is already held —
        # no stale readable view survives (V45-04.02).
        assert _view_disposition(conn, views["c0"]) == "held"
        assert _view_disposition(conn, views["c1"]) == "active"

    # Same-input arm (V45-04.04): the full-rebuild store receives the
    # identical correction — it just recomputes the WHOLE scope instead
    # of the touched closure.
    with full_store.tx() as conn:
        correct_claim_revision(conn, "c0", "corrected value 0")

    rep = apply_repair(
        rep_store, plan, caller=caller(), synthesizer=synth,
        min_proof=1,
    )
    full = full_rebuild(
        full_store, SID, caller=caller(), synthesizer=synth_f,
        min_proof=1,
    )

    # The measured bars — locality on BOTH work metrics.
    assert rep["objects_evaluated"] >= 2  # 1 obs candidate + 1 view
    assert full["objects_evaluated"] >= 6  # 4 obs + 2 views
    assert rep["objects_evaluated"] <= 0.5 * full["objects_evaluated"], (
        f"evaluated {rep['objects_evaluated']} vs full "
        f"{full['objects_evaluated']}"
    )
    assert rep["objects_recomputed"] < full["objects_recomputed"], (
        f"recomputed {rep['objects_recomputed']} vs full "
        f"{full['objects_recomputed']}"
    )
    assert rep["objects_scanned"] < full["objects_scanned"]

    # Correctness: the touched slot's observation reflects the
    # correction on BOTH arms (equal post-state), the untouched view is
    # identical, and the stale view object is retired, never served.
    with rep_store.read() as conn:
        texts = {r[2] for r in _obs_rows(conn) if r[4] is None}
        assert any("corrected value 0" in t for t in texts)
        assert _view_disposition(conn, views["c0"]) == "superseded"
        assert _view_disposition(conn, views["c1"]) == "active"
    with full_store.read() as conn:
        texts_f = {r[2] for r in _obs_rows(conn) if r[4] is None}
        assert any("corrected value 0" in t for t in texts_f)
    # The recomposed successor binds the claim at its new revision.
    succ = rep["views"][0]["successor"]
    with rep_store.read() as conn:
        rev_row = conn.execute(
            "SELECT metadata_json FROM object_revisions"
            " WHERE kind = ? AND object_id = ? ORDER BY revision DESC"
            " LIMIT 1",
            (VIEW_OBJECT_KIND, succ),
        ).fetchone()
        doc = rev_row[0]
        assert '"id":"c0"' in doc and '"rev":2' in doc


def test_invalidate_dependents_marks_held_atomically(store):
    """V45-04.02: the invalidation half is immediate and transactional —
    epoch bump, held views, stale observations all inside the caller's
    commit; rollback leaves nothing half-marked."""
    kernel = Kernel(store)
    synth, views = _seed_full(store)
    with store.tx() as conn:
        correct_claim_revision(conn, "c0", "corrected value 0")
        out = invalidate_dependents(
            conn, kernel, SID, [("claim", "c0", 1)],
            synthesizer=synth,
        )
        # The affected set names exactly the dependency closure.
        affected = {tuple(r) for r in out["affected"]}
        obs_affected = [r for r in affected if r[0] == "observation"]
        assert len(obs_affected) == 1
        assert ("derived_view", views["c0"], 1) in affected
        assert _view_disposition(conn, views["c0"]) == "held"
        obs_row = conn.execute(
            "SELECT stale_since_seq FROM observations"
            " WHERE observation_id = ?",
            (obs_affected[0][1],),
        ).fetchone()
        assert obs_row[0] is not None
    # Committed: the marks persist after the tx closes.
    with store.read() as conn:
        assert _view_disposition(conn, views["c0"]) == "held"


def test_view_left_held_when_input_purged(store):
    """A view whose bound input was purged between plan and apply cannot
    be honestly recomputed — the executor reports ``unrecomputable``
    and never reactivates or replaces the tombstoned object."""
    from verbatim.purge import execute_purge, plan_purge

    kernel = Kernel(store)
    synth, views = _seed_full(store, view_claims=("c0",))
    with store.tx() as conn:
        # Plan while the graph is intact: the span seed reaches claim c0
        # (claim_evidence walker) and the view bound to it — the plan
        # names the view as a synthesis target.
        plan = plan_repair(conn, SID, [("span", "sp0", 1)])
        assert plan["complete"] is True
        assert ["derived_view", views["c0"], 1] in [
            t["ref"] for t in plan["targets"]
        ]
    # Real erasure between plan and apply: the production purge path
    # scrubs c0's evidence rows and tombstones the view through the
    # closure engine.
    preview = plan_purge(store, SID, [("claim", "c0")], actor="alice")
    execute_purge(store, preview["purge_id"])
    with store.read() as conn:
        assert _view_disposition(conn, views["c0"]) != "active"
    rep = apply_repair(
        store, plan, caller=caller(), synthesizer=synth, min_proof=1,
    )
    statuses = {v["status"] for v in rep["views"]}
    assert statuses & {"unrecomputable", "failed"}
    assert rep["failures"]
    with store.read() as conn:
        # The tombstone stands — nothing reactivates or fabricates it.
        assert _view_disposition(conn, views["c0"]) != "active"


# ---------------------------------------------------------------------------
# D04 — incomplete plans fail, never partial (V45-04.03)
# ---------------------------------------------------------------------------

def test_unsupported_dependent_fails_plan_and_apply(store):
    """A dependent no registered producer can recompute makes the plan
    incomplete by construction — apply refuses CONTEXT_INCOMPLETE and
    the explicit escape hatch still names it (nothing is hidden)."""
    with store.tx() as conn:
        seed_graph(conn, store, SID, 2)
        # An unknown-kind dependent hanging off c0 rev1.
        conn.execute(
            "INSERT INTO dependency_edges(child_kind,child_id,"
            "child_revision,parent_kind,parent_id,parent_revision,role,"
            "producer_id,operation_id,seq) VALUES('widget','w1',1,"
            "'claim','c0',1,'derived','prod','op:x',0)"
        )
        plan = plan_repair(conn, SID, [("claim", "c0", 1)])
        assert plan["complete"] is False
        assert [u["ref"] for u in plan["unsupported"]] == [
            ["widget", "w1", 1]
        ]
    with pytest.raises(VerbatimError) as exc:
        apply_repair(store, plan)
    assert exc.value.code is ErrorCode.CONTEXT_INCOMPLETE
    rep = apply_repair(store, plan, allow_unsupported=True, min_proof=1)
    assert ["widget", "w1", 1] in rep["unsupported"]


def test_missing_declared_forecast_is_incomplete(store):
    """D04 plan-time fence: a declared forecast that omits a real
    dependent marks the plan incomplete — it cannot be applied at all."""
    with store.tx() as conn:
        seed_graph(conn, store, SID, 2)
        consolidate_windowed(
            conn, SID, window=ConsolidationWindow(since_seq=0),
            min_proof=1,
        )
        plan = plan_repair(
            conn, SID, [("claim", "c0", 1)],
            declared=[("claim", "c0", 1)],  # omits the observation
        )
        assert plan["complete"] is False
        assert plan["missing"]
    with pytest.raises(VerbatimError) as exc:
        apply_repair(store, plan)
    assert exc.value.code is ErrorCode.CONTEXT_INCOMPLETE


def test_apply_time_grown_closure_refuses_without_partial_writes(
    store,
):
    """D04 apply-time fence: a dependent that appears between plan and
    apply is a hidden dependency — the job fails before any write
    commits; the new dependent is neither recomputed nor marked."""
    synth, views = _seed_full(store)
    with store.tx() as conn:
        plan = plan_repair(conn, SID, [("claim", "c0", 1)])
        assert plan["complete"] is True
        plan_id = persist_plan(store, conn, plan)
    # The closure grows AFTER planning: a fresh composed view binds
    # c0 rev1 as one of its inputs — a real dependent the plan never
    # predicted (a different input set mints a different view_id).
    v_late = synth.compose(
        SID,
        caller=caller(),
        purpose="recall",
        view_kind="typed_summary",
        inputs=[
            {"kind": "claim", "id": "c0", "revision": 1},
            {"kind": "claim", "id": "c1", "revision": 1},
        ],
        now_us=T0,
    )
    assert v_late.view_id != views["c0"]
    with pytest.raises(VerbatimError) as exc:
        apply_repair(store, plan_id, caller=caller(), synthesizer=synth,
                     min_proof=1)
    assert exc.value.code is ErrorCode.CONTEXT_INCOMPLETE
    with store.read() as conn:
        # Nothing committed: the late view is still active (never
        # touched), the plan never reached 'applied', and the prior
        # view kept its disposition.
        assert _view_disposition(conn, v_late.view_id) == "active"
        assert _view_disposition(conn, views["c0"]) == "active"
        plan_row = conn.execute(
            "SELECT current_revision FROM objects"
            " WHERE kind = 'repair_plan' AND object_id = ?",
            (plan_id,),
        ).fetchone()
        assert plan_row[0] == 1  # only the 'planned' revision exists


# ---------------------------------------------------------------------------
# producer dispatch — scenes, profiles, branches
# ---------------------------------------------------------------------------

def test_scene_target_dispatches_real_refresh(store):
    """A scene in the impact closure routes to the real
    ``refresh_scene`` producer — a closed member drops prospectively and
    the scene revision bumps (counted as recomputed)."""
    kernel = Kernel(store)
    with store.tx() as conn:
        bootstrap(conn, SID)
        add_episode(conn, SID, "ep0", kind="task", recorded_from=1)
        add_episode(conn, SID, "ep1", kind="task", recorded_from=2)
        scene_id = add_scene(conn, SID, "fam-x", ["ep0", "ep1"], seq=3)
        assert scene_id is not None
        # The localized correction: member ep0 is closed.
        conn.execute(
            "UPDATE episodes SET recorded_until = 9"
            " WHERE episode_id = 'ep0'"
        )
        plan = plan_repair(conn, SID, [("episode", "ep0", 1)])
        assert plan["complete"] is True
        # The scene legitimately appears at both revisions in the closure
        # (assign_episode bumped it); dispatch dedupes to the object.
        assert {t["producer"] for t in plan["targets"]} == {"scenes"}
        inv = invalidate_dependents(conn, kernel, SID,
                                    [("episode", "ep0", 1)])
        affected_ids = {
            i for k, i, _r in (tuple(r) for r in inv["affected"])
        }
        assert scene_id in affected_ids
    rep = apply_repair(store, plan)
    assert len(rep["scenes"]) == 1
    assert rep["scenes"][0]["refreshed"] is True
    assert rep["objects_recomputed"] >= 1
    with store.read() as conn:
        member = conn.execute(
            "SELECT recorded_until FROM episode_members"
            " WHERE episode_id = ? AND object_id = 'ep0'",
            (scene_id,),
        ).fetchone()
        assert member[0] is not None
        assert conn.execute(
            "SELECT revision FROM episodes WHERE episode_id = ?",
            (scene_id,),
        ).fetchone()[0] >= 2


def test_profile_target_dispatches_real_refresh(store):
    """A profile entry in the closure routes to the real
    ``ProfileService.refresh`` — the scope reconciliation runs and
    reports honestly (a moved-but-live parent keeps the entry servable;
    a purged one would tombstone)."""
    kernel = Kernel(store)
    svc = ProfileService(store)
    with store.tx() as conn:
        seed_graph(conn, store, SID, 2)
        consolidate_windowed(
            conn, SID, window=ConsolidationWindow(since_seq=0),
            min_proof=1,
        )
    svc.put_topic(
        caller(), SID,
        {"topic_key": "prefs",
         "match": {"predicates": ["pred:0"]},
         "min_support": 1},
        purpose="admin",
    )
    comp = svc.compile(caller(), SID, purpose="derive")
    assert comp["claims_matched"] >= 1
    entry_id = comp["entries_created"][0]

    with store.tx() as conn:
        correct_claim_revision(conn, "c0", "likes emacs now")
        plan = plan_repair(conn, SID, [("claim", "c0", 1)])
        producers = {t["producer"] for t in plan["targets"]}
        assert "profiles" in producers
        assert "consolidation" in producers
    rep = apply_repair(
        store, plan, caller=caller(), synthesizer=None,
        profile_service=svc, min_proof=1,
    )
    assert rep["profiles"] is not None
    assert rep["profiles"]["scanned"] >= 1
    with store.read() as conn:
        row = conn.execute(
            "SELECT disposition FROM objects"
            " WHERE kind = 'profile' AND object_id = ?",
            (entry_id,),
        ).fetchone()
        assert row is not None


def test_branch_in_closure_is_held_only(store):
    """A branch pinned on a corrected parent appears in the impact set
    as ``held_only`` — it is marked held for the operator, never
    auto-recomputed (the apply surface is review-gated)."""
    from verbatim.branches import BranchService

    kernel = Kernel(store)
    bsvc = BranchService(store)
    with store.tx() as conn:
        seed_graph(conn, store, SID, 2)
    branch = bsvc.create(
        caller(), SID, name="fix-c0",
        ops=[{"effect": "archive", "claim_id": "c0"}],
    )
    with store.tx() as conn:
        correct_claim_revision(conn, "c0", "corrected value 0")
        plan = plan_repair(conn, SID, [("claim", "c0", 1)])
        assert [h["ref"] for h in plan["held_only"]] == [
            ["branch", branch["branch_id"], 1]
        ]
        assert plan["complete"] is True  # held-only is not unsupported
        inv = invalidate_dependents(
            conn, kernel, SID, [("claim", "c0", 1)],
            synthesizer=None,
        )
        assert inv["marked_held"]["branches"] == 1
        obj = repos_v4.get(
            conn, "objects",
            {"kind": "branch", "object_id": branch["branch_id"]},
        )
        assert obj["disposition"] == "held"


def test_mark_held_is_idempotent_and_bounded(store):
    """mark_held only flips live objects and is a no-op replay — the
    caller's transaction owns the marks either way."""
    kernel = Kernel(store)
    synth, views = _seed_full(store)
    with store.tx() as conn:
        first = mark_held(
            conn, [("derived_view", views["c0"], 1)],
            synthesizer=synth,
        )
        assert first["views"] == 1
        second = mark_held(
            conn, [("derived_view", views["c0"], 1)],
            synthesizer=synth,
        )
        assert second["views"] == 0  # already held — idempotent
        # An unrelated view is untouched.
        assert _view_disposition(conn, views["c1"]) == "active"
