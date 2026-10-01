"""Utility-budgeted refresh scheduling (SPEC_V4_5 §08, V45-08.*).

Pinned behaviors:

- candidates are ordered by the declared utility expression —
  staleness × demand × owner priority, plus stale-answer and
  deferred-readiness costs (V45-08.01/03);
- ``max_jobs``/``max_claims``/``min_utility`` bounds defer the
  lowest-utility non-owner work and *report* it — deferred regions
  resurface because the watermark only advances past covered work
  (V45-08.03/04);
- owner-requested work is never evicted by the budget but still counts
  in the spend report (V45-08.02, D12);
- scheduled work lands on the real ``JobQueue`` as ``consolidate`` jobs
  on the background lane and drains through the production
  ``handle_consolidate`` path — windowed consolidation that writes real
  observations and ``consolidation_pass`` events;
- utility scores are advisory ordering only — enqueueing a plan writes
  no claims, observations, derivations, or provenance of its own;
- ``periodic_plan`` is the honest D11 comparator: one unwindowed
  consolidate job (+ bounded reflection) charged a full input scan.
"""

from __future__ import annotations

import pytest

from verbatim.config import VerbatimConfig
from verbatim.core.types import (
    ErrorCode,
    JobKind,
    VerbatimError,
    json_dumps,
    safe_json_loads,
)
from verbatim.ingest import Ingester
from verbatim.observations import SLOT_AGGREGATE_V1, consolidate
from verbatim.observations.aggregate import observation_id_for
from verbatim.observations.freshness import next_seq
from verbatim.refresh import (
    REFRESH_EVENT_KIND,
    REFRESH_POLICY_ID,
    RefreshBudget,
    RefreshScheduler,
    last_watermark,
)
from verbatim.storage import repos_v3
from verbatim.storage.store import Store


@pytest.fixture()
def store(tmp_path):
    s = Store.create(str(tmp_path / "refresh.db"))
    yield s
    s.close()


@pytest.fixture()
def scope_id(store):
    sid = "scope:refresh"
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES (?, 'prof', 'owner')",
            (sid,),
        )
    return sid


def _claim(
    conn,
    scope_id,
    claim_id,
    *,
    subject="alice",
    predicate="residence",
    value="london",
    family_id=None,
    recorded_from=1,
    state="active",
):
    conn.execute(
        "INSERT INTO claims (claim_id, scope_id, subject_id, predicate,"
        " created_event) VALUES (?,?,?,?,0)",
        (claim_id, scope_id, subject, predicate),
    )
    conn.execute(
        "INSERT INTO claim_revisions (claim_id, revision, state,"
        " object_json, polarity, modality, recorded_from)"
        " VALUES (?,1,?,?, 'affirmative','asserted',?)",
        (claim_id, state,
         json_dumps({"kind": "literal", "text": value}), recorded_from),
    )
    if family_id is not None:
        conn.execute(
            "INSERT OR IGNORE INTO evidence_families"
            " (family_id, scope_id, origin_kind, origin_id, created_event)"
            " VALUES (?,?,?,?,0)",
            (family_id, scope_id, "test", claim_id),
        )
        conn.execute(
            "INSERT INTO family_members (family_id, object_kind, object_id,"
            " role) VALUES (?, 'claim', ?, 'origin')",
            (family_id, claim_id),
        )


def _slot(conn, scope_id, subject, predicate, value, n=2, prefix=None):
    """One slot with ``n`` family-distinct claims (proof threshold)."""
    prefix = prefix or f"{subject}-{predicate}-{value}"
    for i in range(n):
        _claim(
            conn, scope_id, f"{prefix}-{i}",
            subject=subject, predicate=predicate, value=value,
            family_id=f"{prefix}-fam-{i}",
        )


def _revise(conn, claim_id, *, new_value, at_seq):
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
        " VALUES (?,?, 'active', ?, 'affirmative','asserted',?)",
        (claim_id, head + 1,
         json_dumps({"kind": "literal", "text": new_value}), at_seq),
    )


def _deliveries(conn, scope_id, claim_ids, n=3):
    """Observed demand: influence rows naming the claims."""
    from verbatim.core.types import new_id

    for i, cid in enumerate(claim_ids):
        for j in range(n):
            conn.execute(
                "INSERT INTO influence (handle_id, receipt_id, scope_id,"
                " caller_id, epoch, pack, object_kind, object_id,"
                " revision, created_us) VALUES (?,?,?, 'host', 0,"
                " 'answer', 'claim', ?, 1, 0)",
                (new_id(), f"rc:{i}-{j}", scope_id, cid),
            )


def _scheduler(store):
    return RefreshScheduler(store)


def _drain(store, scope=None):
    ing = Ingester(store, VerbatimConfig())
    return ing.drain_report(scope=scope, limit=256)


def _events(conn, scope_id, kind):
    rows = conn.execute(
        "SELECT payload_json FROM events WHERE scope_id = ? AND kind = ?"
        " ORDER BY event_seq",
        (scope_id, kind),
    ).fetchall()
    return [safe_json_loads(r[0]) or {} for r in rows]


# ---------------------------------------------------------------------------
# planning: changed regions + utility ordering
# ---------------------------------------------------------------------------


def test_plan_discovers_only_changed_regions(store, scope_id):
    """Baseline consolidate, then one correction — the plan names only
    the touched slot, not the whole scope."""
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
        _slot(conn, scope_id, "bob", "role", "admin")
        consolidate(conn, scope_id)
        sched = _scheduler(store)
        # First plan covers everything (watermark 0).
        plan = sched.plan(conn, scope_id)
        assert len(plan.scheduled) == 2
        seq = next_seq(conn, scope_id)
        _revise(conn, "bob-role-admin-0", new_value="viewer", at_seq=seq)
        plan = sched.plan(conn, scope_id, since_seq=seq - 1)
        assert len(plan.scheduled) == 1
        assert plan.scheduled[0].first_change_seq == seq
        assert plan.scheduled[0].staleness == 0


def test_utility_orders_by_staleness_demand_priority(store, scope_id):
    """Higher demand and older changes rank above fresh quiet ones."""
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
        _slot(conn, scope_id, "bob", "role", "admin")
        _slot(conn, scope_id, "carol", "dept", "eng")
        _deliveries(conn, scope_id, ["carol-dept-eng-0",
                                     "carol-dept-eng-1"], n=5)
        sched = _scheduler(store)
        plan = sched.plan(conn, scope_id)
        utils = [c.utility for c in plan.scheduled]
        assert utils == sorted(utils, reverse=True)
        top = plan.scheduled[0]
        assert "carol" in top.claim_ids[0] or top.demand == 10
        assert top.demand == 10


def test_plan_reports_spend(store, scope_id):
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
        sched = _scheduler(store)
        plan = sched.plan(conn, scope_id)
        spend = plan.spend()
        assert spend["jobs"] == 1
        assert spend["claims_processed"] == 2
        assert spend["bytes_processed"] > 0
        assert spend["deferred"] == 0


# ---------------------------------------------------------------------------
# budget enforcement (V45-08.04)
# ---------------------------------------------------------------------------


def test_max_jobs_defers_lowest_utility(store, scope_id):
    """Three changed regions, budget 2 — the lowest-utility region
    defers and is *reported*, never dropped silently."""
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
        _slot(conn, scope_id, "bob", "role", "admin")
        _slot(conn, scope_id, "carol", "dept", "eng")
        _deliveries(conn, scope_id, ["alice-residence-london-0"], n=9)
        sched = _scheduler(store)
        plan = sched.plan(
            conn, scope_id, budget=RefreshBudget(max_jobs=2),
        )
        assert len(plan.scheduled) == 2
        assert len(plan.deferred) == 1
        deferred = plan.deferred[0]
        assert plan.deferred_reasons[deferred.region_key] == "max_jobs"
        # The lowest-utility candidate is the one deferred.
        assert deferred.utility <= min(
            c.utility for c in plan.scheduled
        )
        assert plan.watermark_after <= plan.current_seq


def test_deferred_regions_resurface_next_cycle(store, scope_id):
    """Watermark only advances past covered work — a deferred change
    re-appears on the next plan (V45-08.03)."""
    sched = _scheduler(store)
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
        _slot(conn, scope_id, "bob", "role", "admin")
        _slot(conn, scope_id, "carol", "dept", "eng")
    plan = sched.refresh(
        scope_id, budget=RefreshBudget(max_jobs=1),
    )
    assert len(plan.scheduled) == 1
    assert len(plan.deferred) == 2
    first_deferred = min(
        c.first_change_seq for c in plan.deferred
        if c.first_change_seq is not None
    )
    with store.read() as conn:
        wm = last_watermark(conn, scope_id)
    # The watermark holds just below the oldest deferred change — the
    # deferred work stays discoverable, the covered part stays done.
    assert wm == first_deferred - 1
    # Next cycle re-discovers the deferred regions.
    with store.read() as conn:
        plan2 = sched.plan(conn, scope_id)
        assert len(plan2.scheduled) + len(plan2.deferred) >= 2
        keys = {c.region_key for c in plan2.scheduled + plan2.deferred}
        deferred_keys = {c.region_key for c in plan.deferred}
        assert deferred_keys <= keys


def test_max_claims_bounds_bytes(store, scope_id):
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london", n=3)
        _slot(conn, scope_id, "bob", "role", "admin", n=3)
        sched = _scheduler(store)
        plan = sched.plan(
            conn, scope_id, budget=RefreshBudget(max_claims=3),
        )
        assert len(plan.scheduled) == 1
        assert plan.scheduled[0].cost_claims <= 3
        assert len(plan.deferred) == 1
        assert plan.deferred_reasons[
            plan.deferred[0].region_key
        ] == "max_claims"


def test_min_utility_floor(store, scope_id):
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
        sched = _scheduler(store)
        plan = sched.plan(
            conn, scope_id,
            budget=RefreshBudget(min_utility=10**9),
        )
        assert plan.scheduled == []
        assert len(plan.deferred) == 1
        assert plan.deferred_reasons[
            plan.deferred[0].region_key
        ] == "below_min_utility"


def test_budget_validation(store):
    with pytest.raises(Exception):
        RefreshBudget(max_jobs=-1)
    with pytest.raises(Exception):
        RefreshBudget(max_claims=0)


# ---------------------------------------------------------------------------
# owner priority (V45-08.02, D12)
# ---------------------------------------------------------------------------


def test_owner_request_bypasses_budget_but_counts_spend(store, scope_id):
    """Budget exhausted by scanner candidates — the owner's explicit
    request still schedules (never demoted) and still reports cost."""
    sched = _scheduler(store)
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
        _slot(conn, scope_id, "bob", "role", "admin")
        _slot(conn, scope_id, "carol", "dept", "eng")
    # First cycle covers everything; watermark advances to current_seq.
    sched.refresh(scope_id)
    with store.tx() as conn:
        seq = next_seq(conn, scope_id)
        # Two slots change; carol's stays untouched — the owner still
        # asks for it explicitly.
        _revise(conn, "alice-residence-london-0",
                new_value="paris", at_seq=seq)
        _revise(conn, "bob-role-admin-0",
                new_value="viewer", at_seq=seq)
        plan = sched.plan(
            conn, scope_id,
            budget=RefreshBudget(max_jobs=1),
            owner_requests=[{"claim_ids": ["carol-dept-eng-0"],
                             "weight": 1.0}],
        )
        owner = [c for c in plan.scheduled if c.owner_requested]
        assert len(owner) == 1
        assert "owner_request" in owner[0].reasons
        # Two changed candidates compete for the single budget slot —
        # one defers — and the owner job schedules around it entirely.
        assert len(plan.scheduled) == 2
        assert len(plan.deferred) == 1
        assert not plan.deferred[0].owner_requested
        spend = plan.spend()
        assert spend["jobs"] == 2
        assert spend["priority_jobs"] == 1
        assert spend["claims_processed"] == (
            sum(c.cost_claims for c in plan.scheduled)
        )


def test_owner_weight_boosts_order(store, scope_id):
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
        _slot(conn, scope_id, "bob", "role", "admin")
        sched = _scheduler(store)
        plain = sched.plan(conn, scope_id)
        boosted = sched.plan(
            conn, scope_id,
            owner_requests=[{"claim_ids": ["alice-residence-london-0"],
                             "weight": 5.0}],
        )
        owner = [c for c in boosted.scheduled if c.owner_requested][0]
        assert owner.priority == 5.0
        non_owner_util = max(
            c.utility for c in boosted.scheduled if not c.owner_requested
        )
        assert owner.utility > non_owner_util
        # The owner's region now orders first.
        assert boosted.scheduled[0].owner_requested


# ---------------------------------------------------------------------------
# execution through the real queue + handler
# ---------------------------------------------------------------------------


def test_refresh_enqueues_real_consolidate_jobs(store, scope_id):
    """Scheduled work lands on the queue as background-lane
    ``consolidate`` jobs carrying the region window."""
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
        sched = _scheduler(store)
        plan = sched.plan(conn, scope_id)
        sched.schedule(conn, plan)
        assert len(plan.job_ids) == 1
        job = conn.execute(
            "SELECT kind, lane, input_refs_json FROM jobs WHERE job_id = ?",
            (plan.job_ids[0],),
        ).fetchone()
        assert job[0] == JobKind.CONSOLIDATE.value
        assert job[1] == "background"
        refs = safe_json_loads(job[2])
        assert refs["window"]["claim_ids"]
        assert refs["refresh"]["policy_id"] == REFRESH_POLICY_ID


def test_scheduled_jobs_drain_through_handler(store, scope_id):
    """Draining the queue executes the real windowed consolidation —
    the corrected slot's observation revises, the untouched one stays."""
    sched = _scheduler(store)
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
        _slot(conn, scope_id, "bob", "role", "admin")
        consolidate(conn, scope_id)
    # Baseline refresh covers the whole scope and advances the
    # watermark — the next cycle sees only the correction.
    sched.refresh(scope_id)
    _drain(store)
    with store.tx() as conn:
        seq = next_seq(conn, scope_id)
        _revise(conn, "alice-residence-london-0",
                new_value="paris", at_seq=seq)
    plan = sched.refresh(scope_id)
    assert len(plan.job_ids) == 1
    report = _drain(store)
    assert report["succeeded"] >= 1
    with store.read() as conn:
        london_obs = observation_id_for(
            scope_id, "alice", "residence", None, "", "affirmative:london"
        )
        paris_obs = observation_id_for(
            scope_id, "alice", "residence", None, "", "affirmative:paris"
        )
        bob_obs = observation_id_for(
            scope_id, "bob", "role", None, "", "affirmative:admin"
        )
        london = repos_v3.get(
            conn, "observations", {"observation_id": london_obs}
        )
        bob = repos_v3.get(
            conn, "observations", {"observation_id": bob_obs}
        )
        # london lost a proof → retired by the windowed pass; paris is
        # still below threshold (single family) so no new observation.
        assert london["recorded_until"] is not None
        assert bob["recorded_until"] is None
        # The durable pass events exist — consolidation + refresh plan.
        passes = _events(conn, scope_id, "consolidation_pass")
        assert passes
        refresh_events = _events(conn, scope_id, REFRESH_EVENT_KIND)
        assert len(refresh_events) == 2  # baseline + correction cycle
        last = refresh_events[-1]
        assert last["watermark_after"] == plan.watermark_after
        assert last["watermark_before"] == refresh_events[0]["watermark_after"]
        assert last["spend"]["jobs"] == 1


def test_refresh_event_records_plan_and_spend(store, scope_id):
    sched = _scheduler(store)
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
        _slot(conn, scope_id, "bob", "role", "admin")
    plan = sched.refresh(scope_id, budget=RefreshBudget(max_jobs=1))
    with store.read() as conn:
        ev = _events(conn, scope_id, REFRESH_EVENT_KIND)[0]
    assert ev["policy_id"] == REFRESH_POLICY_ID
    assert ev["scope_id"] == scope_id
    assert ev["spend"]["jobs"] == len(plan.scheduled) == 1
    assert ev["spend"]["deferred"] == 1
    assert all("utility" in c for c in ev["scheduled"])
    assert all("reasons" in c for c in ev["scheduled"])
    assert ev["deferred"][0]["deferred_reason"] == "max_jobs"


def test_dedup_makes_replan_idempotent(store, scope_id):
    """Same scope + same seq → same dedup key → the existing job id is
    returned rather than a duplicate enqueue."""
    sched = _scheduler(store)
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
        plan = sched.plan(conn, scope_id)
        sched.schedule(conn, plan)
        again = sched.plan(conn, scope_id)
        sched.schedule(conn, again)
        assert again.job_ids == plan.job_ids


def test_advisory_only_no_domain_writes(store, scope_id):
    """Enqueueing a plan writes jobs + the audit event only — no
    claims, observations, or derivations move (utility is ordering,
    never evidence)."""
    sched = _scheduler(store)
    tables = ("claims", "claim_revisions", "observations",
              "observation_evidence", "derivations")
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
    with store.read() as conn:
        before = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in tables
        }
    plan = sched.refresh(scope_id)
    with store.read() as conn:
        after = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in tables
        }
        assert before == after
        assert plan.job_ids


def test_seq_window_reaches_fully_closed_region(store, scope_id):
    """A slot emptied by deletion can't be reached by claim_ids —
    the job falls back to a since_seq window that retires its stale
    observation."""
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
        _slot(conn, scope_id, "bob", "role", "admin")
        consolidate(conn, scope_id)
        seq = next_seq(conn, scope_id)
        # Delete alice's whole slot: close both heads with no reopen.
        conn.execute(
            "UPDATE claim_revisions SET recorded_until = ?"
            " WHERE claim_id IN"
            " ('alice-residence-london-0','alice-residence-london-1')",
            (seq,),
        )
        sched = _scheduler(store)
        plan = sched.plan(conn, scope_id, since_seq=seq - 1)
        target = [
            c for c in plan.scheduled
            if c.seq_window is not None
        ]
        assert len(target) == 1
        assert target[0].seq_window == seq - 1
        sched.schedule(conn, plan)
    _drain(store)
    with store.read() as conn:
        london = repos_v3.get(
            conn, "observations",
            {"observation_id": observation_id_for(
                scope_id, "alice", "residence", None, "",
                "affirmative:london")},
        )
        assert london["recorded_until"] is not None


# ---------------------------------------------------------------------------
# periodic comparator (D11 baseline)
# ---------------------------------------------------------------------------


def test_periodic_plan_single_full_job(store, scope_id):
    """The comparator is one unwindowed consolidate+reflect job charged
    a double input scan — the honest periodic cost."""
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
        _slot(conn, scope_id, "bob", "role", "admin")
        sched = _scheduler(store)
        plan = sched.periodic_plan(conn, scope_id, reflect=True)
        assert len(plan.scheduled) == 1
        cand = plan.scheduled[0]
        assert cand.kind == "periodic_full"
        assert cand.cost_claims == 8  # 4 claims × (consolidate+reflect)
        sched.schedule(conn, plan)
        job = conn.execute(
            "SELECT input_refs_json FROM jobs WHERE job_id = ?",
            (plan.job_ids[0],),
        ).fetchone()
        refs = safe_json_loads(job[0])
        assert "window" not in refs  # unwindowed = full scan
        assert refs["reflect"]


def test_periodic_plan_executes_full_pass(store, scope_id):
    sched = _scheduler(store)
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
        plan = sched.periodic_plan(conn, scope_id)
        sched.schedule(conn, plan)
    _drain(store)
    with store.read() as conn:
        obs = repos_v3.query(
            conn, "observations", {"scope_id": scope_id}
        )
        assert obs  # the full pass consolidated the slot


# ---------------------------------------------------------------------------
# region packing (V45-08.04 — emitted jobs are the bounded unit)
# ---------------------------------------------------------------------------


def test_regions_pack_into_one_job(store, scope_id):
    """regions_per_job packs consecutive claim-id-windowed regions into
    a single emitted job — identical domain work, less queue overhead."""
    sched = RefreshScheduler(store, regions_per_job=3)
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
        _slot(conn, scope_id, "bob", "role", "admin")
        _slot(conn, scope_id, "carol", "dept", "eng")
        plan = sched.plan(conn, scope_id)
        assert len(plan.scheduled) == 3
        assert plan.projected_jobs == 1
        sched.schedule(conn, plan)
        assert len(plan.job_ids) == 1
        job = conn.execute(
            "SELECT input_refs_json FROM jobs WHERE job_id = ?",
            (plan.job_ids[0],),
        ).fetchone()
        refs = safe_json_loads(job[0])
        # The packed window names the union of all three regions' claims.
        assert len(refs["window"]["claim_ids"]) == 6
        assert len(refs["refresh"]["regions"]) == 3
        assert sorted(refs["refresh"]["regions"]) == sorted(
            c.region_key for c in plan.scheduled
        )
    assert plan.spend()["jobs"] == 1
    assert plan.spend()["work_items"] == 3


def test_packed_job_drains_all_regions(store, scope_id):
    """One packed job re-derives every packed slot through the real
    windowed-consolidate handler."""
    sched = RefreshScheduler(store, regions_per_job=2)
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
        _slot(conn, scope_id, "bob", "role", "admin")
        _slot(conn, scope_id, "carol", "dept", "eng")
        _slot(conn, scope_id, "dan", "lang", "go")
    plan = sched.refresh(scope_id)
    assert len(plan.job_ids) == 2  # 4 regions / 2 per job
    report = _drain(store)
    assert report["succeeded"] == 2
    with store.read() as conn:
        obs = repos_v3.query(
            conn, "observations", {"scope_id": scope_id}
        )
        assert len(obs) == 4  # every packed slot derived


def test_max_jobs_bounds_emitted_jobs_not_regions(store, scope_id):
    """The budget bounds emitted jobs: packing lets more regions fit
    inside the same declared job count."""
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
        _slot(conn, scope_id, "bob", "role", "admin")
        _slot(conn, scope_id, "carol", "dept", "eng")
        packed = RefreshScheduler(store, regions_per_job=3)
        plan = packed.plan(
            conn, scope_id, budget=RefreshBudget(max_jobs=1),
        )
        # All three packable regions fit inside one emitted job.
        assert len(plan.scheduled) == 3
        assert len(plan.deferred) == 0
        assert plan.projected_jobs == 1
        unpacked = RefreshScheduler(store, regions_per_job=1)
        plan2 = unpacked.plan(
            conn, scope_id, budget=RefreshBudget(max_jobs=1),
        )
        assert len(plan2.scheduled) == 1
        assert len(plan2.deferred) == 2


def test_owner_jobs_stay_unpacked(store, scope_id):
    """Owner-requested work emits its own job — priority stays
    explicitly attributed even alongside packable regions."""
    sched = RefreshScheduler(store, regions_per_job=4)
    with store.tx() as conn:
        _slot(conn, scope_id, "alice", "residence", "london")
        _slot(conn, scope_id, "bob", "role", "admin")
        plan = sched.plan(
            conn, scope_id,
            owner_requests=[
                {"claim_ids": ["bob-role-admin-0"], "weight": 4.0}
            ],
        )
        sched.schedule(conn, plan)
        # bob's slot merged into the owner request — it is owner work,
        # not a packable region; alice's region packs alone.
        assert len(plan.scheduled) == 2
        assert len(plan.job_ids) == 2
        jobs = {
            r[0]: safe_json_loads(r[1])
            for r in conn.execute(
                "SELECT job_id, input_refs_json FROM jobs"
            ).fetchall()
        }
        owner_jobs = [
            refs for refs in jobs.values()
            if refs["refresh"].get("region_key")
        ]
        assert len(owner_jobs) == 1
        pack_jobs = [
            refs for refs in jobs.values()
            if refs["refresh"].get("regions")
        ]
        assert len(pack_jobs) == 1


def test_regions_per_job_validation(store):
    for bad in (0, -1, 2.5, "3", True):
        with pytest.raises(VerbatimError) as ei:
            RefreshScheduler(store, regions_per_job=bad)
        assert ei.value.code == ErrorCode.VALIDATION
