"""Coordinator tests: atomic effect application, idempotent receipts,
fencing, epoch gates, expected revisions, and follow-up job obligations
(SPEC_V4 §09 — V4-09.01..V4-09.10).
"""

from __future__ import annotations

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v4 import Effect, EffectKind, EffectPlan, JobRequest
from verbatim.jobs.coordinator import Coordinator
from verbatim.storage.store import Store


@pytest.fixture()
def coord(tmp_path):
    store = Store.create(str(tmp_path / "v4.db"))
    yield Coordinator(store)
    store.close()


def _plan(**kw) -> EffectPlan:
    base = dict(
        operation_id="op-1",
        scope_id="scope-a",
        producer_id="test-producer",
        input_digests=("sha256:in-1",),
        effects=(
            Effect(
                EffectKind.INSERT_OBJECT,
                "objects",
                {
                    "object_id": "obj-1",
                    "kind": "claim",
                    "scope_id": "scope-a",
                    "current_revision": 1,
                    "created_event": 1,
                },
            ),
        ),
    )
    base.update(kw)
    return EffectPlan(**base)


def test_apply_plan_commits_effects_and_returns_receipt(coord):
    receipt = coord.apply_plan(_plan())
    assert receipt.operation_id == "op-1"
    assert receipt.effects_applied == 1
    with coord.store.read() as conn:
        assert conn.execute(
            "SELECT object_id FROM objects WHERE object_id='obj-1'"
        ).fetchall() == [("obj-1",)]


def test_replay_returns_prior_receipt_without_reapplying(coord):
    first = coord.apply_plan(_plan())
    second = coord.apply_plan(_plan())
    assert second.applied_seq == first.applied_seq
    # The effect ran once: the object row exists exactly once.
    with coord.store.read() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM objects WHERE object_id='obj-1'"
        ).fetchone()[0] == 1


def test_operation_key_reuse_with_different_inputs_conflicts(coord):
    coord.apply_plan(_plan())
    with pytest.raises(VerbatimError) as exc:
        coord.apply_plan(_plan(input_digests=("sha256:OTHER",)))
    assert exc.value.code == ErrorCode.OPERATION_CONFLICT


def test_follow_up_jobs_enqueue_atomically(coord):
    receipt = coord.apply_plan(
        _plan(follow_ups=(JobRequest("harvest", "ordinary", {"x": 1}),))
    )
    assert len(receipt.jobs_enqueued) == 1
    with coord.store.read() as conn:
        row = conn.execute(
            "SELECT kind, state, operation_key, lane FROM jobs"
        ).fetchone()
    assert row == ("harvest", "queued", "op-1", "ordinary")


def test_failed_effect_rolls_back_follow_up_jobs(coord):
    """A bad effect must not leave its enqueued jobs behind (V4-09.03)."""
    bad = _plan(
        effects=(
            Effect(EffectKind.INSERT_OBJECT, "objects", {"object_id": "x"}),
        ),
        follow_ups=(JobRequest("harvest", "ordinary", {}),),
    )
    with pytest.raises(VerbatimError):
        coord.apply_plan(bad)
    with coord.store.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM operation_receipts"
        ).fetchone()[0] == 0


def test_epoch_gate_rejects_stale_pin(coord):
    plan = _plan(epoch_vector={"policy_epoch:scope-a": 7})
    with pytest.raises(VerbatimError) as exc:
        coord.apply_plan(plan)
    assert exc.value.code == ErrorCode.STALE_EPOCH


def test_expected_revision_gate(coord):
    coord.apply_plan(_plan())
    ok = _plan(
        operation_id="op-2",
        effects=(
            Effect(
                EffectKind.UPDATE_LIFECYCLE,
                "objects",
                {
                    "object_id": "obj-1",
                    "disposition": "superseded",
                    "current_revision": 2,
                },
            ),
        ),
        expected_revisions={"obj-1": 1},
    )
    coord.apply_plan(ok)
    stale = _plan(
        operation_id="op-3",
        effects=(
            Effect(
                EffectKind.UPDATE_LIFECYCLE,
                "objects",
                {"object_id": "obj-1", "disposition": "archived"},
            ),
        ),
        expected_revisions={"obj-1": 1},  # still pinned at the old revision
    )
    with pytest.raises(VerbatimError) as exc:
        coord.apply_plan(stale)
    assert exc.value.code == ErrorCode.STALE_DEPENDENCY


def test_expected_revision_missing_object_is_stale(coord):
    plan = _plan(expected_revisions={"obj-never": 3})
    with pytest.raises(VerbatimError) as exc:
        coord.apply_plan(plan)
    assert exc.value.code == ErrorCode.STALE_DEPENDENCY


def test_lease_fence_rejects_terminal_job(coord):
    """A cancelled/terminal job's worker must commit nothing (V4-09.05)."""
    with coord.store.tx() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, scope_id, kind, state, dedup_key,"
            " input_refs_json, policy_epoch, attempts, not_before_us,"
            " deadline_us, generation, operation_key, lane)"
            " VALUES ('j1','scope-a','harvest','cancelled',NULL,'{}',0,0,0,0,0,'op-x','ordinary')"
        )
    with pytest.raises(VerbatimError) as exc:
        coord.apply_plan(
            _plan(operation_id="op-x"), job_id="j1", expected_lease=0
        )
    assert exc.value.code == ErrorCode.CANCELLED


def test_lease_fence_rejects_stale_generation(coord):
    with coord.store.tx() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, scope_id, kind, state, dedup_key,"
            " input_refs_json, policy_epoch, attempts, not_before_us,"
            " deadline_us, generation, operation_key, lane)"
            " VALUES ('j2','scope-a','harvest','leased',NULL,'{}',0,0,0,0,5,'op-y','ordinary')"
        )
    with pytest.raises(VerbatimError) as exc:
        coord.apply_plan(
            _plan(operation_id="op-y"), job_id="j2", expected_lease=3
        )
    assert exc.value.code == ErrorCode.LEASE_LOST
