"""JobQueue tests: fencing, reclaim, backoff, dedup, backpressure, purge lane."""

from __future__ import annotations

import random

import pytest

from verbatim.core.types import ErrorCode, JobKind, JobState, VerbatimError
from verbatim.jobs.queue import JobQueue


@pytest.fixture()
def queue(store, clock):
    return JobQueue(store, cap=100, clock=clock, rng=random.Random(7))


def test_scope_key_consistent_across_modules(scope):
    """jobs.scope_key and privacy.scope_key must render identically —
    consents and jobs share the scope_id TEXT namespace — and both must
    match the canonical core.identity rendering used by ingest and the CLI."""
    from verbatim.core.identity import scope_key as canonical
    from verbatim.jobs.queue import scope_key as qk
    from verbatim.privacy.egress import scope_key as ek

    assert qk(scope) == ek(scope) == canonical(scope)
    sid = canonical(scope)
    assert qk(sid) == sid


def _enq(queue, store, scope_id, kind=JobKind.COMPARE, **kw):
    with store.tx() as conn:
        return queue.enqueue(conn, scope_id, kind, {"a": 1}, **kw)


# ---------------------------------------------------------------- enqueue


def test_enqueue_and_list(queue, store, scope_id):
    jid = _enq(queue, store, scope_id)
    jobs = queue.list(scope_id)
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == jid
    assert jobs[0]["state"] == "queued"
    assert jobs[0]["kind"] == "compare"
    assert jobs[0]["input_refs"] == {"a": 1}  # private _enqueued_us stripped


def test_enqueue_dedup_returns_existing(queue, store, scope_id):
    key = b"dedup-1"
    a = _enq(queue, store, scope_id, dedup_key=key)
    b = _enq(queue, store, scope_id, dedup_key=key)
    assert a == b
    assert len(queue.list(scope_id)) == 1


def test_enqueue_dedup_retry_wait_returns_existing(queue, store, clock, scope_id):
    """In-flight dedup covers retry_wait too — a backed-off job is still
    the durable work item for its key."""
    key = b"dedup-rw"
    a = _enq(queue, store, scope_id, dedup_key=key)
    job = queue.lease(scope_id, ["compare"], owner="w1")[0]
    with store.tx() as conn:
        st = queue.fail(
            conn, a, "w1", job["generation"], "BUSY", retryable=True
        )
    assert st == JobState.RETRY_WAIT
    b = _enq(queue, store, scope_id, dedup_key=key)
    assert b == a
    assert len(queue.list(scope_id)) == 1


def test_enqueue_dedup_succeeded_returns_existing(queue, store, scope_id):
    """A succeeded dedup key is the durable receipt — re-enqueue converges
    on it rather than re-running finished work."""
    key = b"dedup-ok"
    a = _enq(queue, store, scope_id, dedup_key=key)
    job = queue.lease(scope_id, ["compare"], owner="w1")[0]
    with store.tx() as conn:
        assert queue.complete(conn, a, "w1", job["generation"]) is True
    b = _enq(queue, store, scope_id, dedup_key=key)
    assert b == a
    assert len(queue.list(scope_id)) == 1


def test_enqueue_dedup_failed_redrives(queue, store, scope_id):
    """A terminal-failed job must not pin its dedup key forever: re-enqueue
    mints a fresh runnable job while the dead attempt stays inspectable."""
    key = b"dedup-fail"
    a = _enq(queue, store, scope_id, dedup_key=key)
    job = queue.lease(scope_id, ["compare"], owner="w1")[0]
    with store.tx() as conn:
        st = queue.fail(conn, a, "w1", job["generation"], "FATAL", retryable=False)
    assert st == JobState.FAILED
    b = _enq(queue, store, scope_id, dedup_key=key)
    assert b != a  # a *different* runnable job, not the dead row
    jobs = {j["job_id"]: j for j in queue.list(scope_id)}
    assert len(jobs) == 2
    assert jobs[b]["state"] == "queued"
    # the dead attempt keeps its terminal state and history; the canonical
    # dedup key moved to the fresh attempt
    assert jobs[a]["state"] == "failed"
    assert jobs[a]["dedup_key"] != key
    assert jobs[b]["dedup_key"] == key
    # the fresh attempt leases and runs to completion normally
    leased = queue.lease(scope_id, ["compare"], owner="w2")[0]
    assert leased["job_id"] == b
    with store.tx() as conn:
        assert queue.complete(conn, b, "w2", leased["generation"]) is True
    # and once succeeded, the dedup key converges again
    c = _enq(queue, store, scope_id, dedup_key=key)
    assert c == b


def test_enqueue_dedup_cancelled_redrives(queue, store, scope_id):
    """Cancelled is terminal-unfinished like failed — the key frees for a
    fresh attempt instead of stranding the work item."""
    key = b"dedup-cancel"
    a = _enq(queue, store, scope_id, dedup_key=key)
    with store.tx() as conn:
        assert queue.cancel(conn, a) is True
    b = _enq(queue, store, scope_id, dedup_key=key)
    assert b != a
    jobs = {j["job_id"]: j for j in queue.list(scope_id)}
    assert jobs[b]["state"] == "queued"
    assert jobs[a]["state"] == "cancelled"


def test_enqueue_dedup_distinct_kinds_ok(queue, store, scope_id):
    a = _enq(queue, store, scope_id, dedup_key=b"k1")
    b = _enq(queue, store, scope_id, kind=JobKind.EMBED, dedup_key=b"k2")
    assert a != b


def test_backpressure_cap(queue, store, scope_id):
    small = JobQueue(store, cap=3, clock=queue._now, rng=random.Random(1))
    for _ in range(3):
        _enq(small, store, scope_id)
    with store.tx() as conn, pytest.raises(VerbatimError) as ei:
        small.enqueue(conn, scope_id, JobKind.COMPARE, {})
    assert ei.value.code == ErrorCode.BACKPRESSURE
    assert ei.value.retryable is True


def test_purge_bypasses_cap(store, scope_id, clock):
    q = JobQueue(store, cap=2, clock=clock, rng=random.Random(1))
    _enq(q, store, scope_id)
    _enq(q, store, scope_id)
    pid = _enq(q, store, scope_id, kind=JobKind.PURGE)  # must not raise
    assert pid


# ------------------------------------------------------------------ lease


def test_lease_sets_owner_generation_attempts(queue, store, scope_id):
    jid = _enq(queue, store, scope_id)
    leased = queue.lease(scope_id, [JobKind.COMPARE], owner="w1", lease_s=30)
    assert len(leased) == 1
    job = leased[0]
    assert job["job_id"] == jid
    assert job["state"] == "leased"
    assert job["lease_owner"] == "w1"
    assert job["generation"] == 1
    assert job["attempts"] == 1
    assert job["lease_until_us"] == queue._now() + 30_000_000
    # second lease sees nothing
    assert queue.lease(scope_id, [JobKind.COMPARE], owner="w2") == []


def test_lease_fifo_and_purge_lane(queue, store, scope_id):
    for _ in range(3):
        _enq(queue, store, scope_id, kind=JobKind.COMPARE)
    purge = _enq(queue, store, scope_id, kind=JobKind.PURGE)
    leased = queue.lease(
        scope_id, [JobKind.COMPARE, JobKind.PURGE], owner="w1", limit=1
    )
    # purge leases ahead of older enrichment jobs
    assert leased[0]["job_id"] == purge
    assert leased[0]["kind"] == "purge"


def test_lease_respects_not_before(queue, store, clock, scope_id):
    future = clock() + 10_000_000
    _enq(queue, store, scope_id, not_before_us=future)
    assert queue.lease(scope_id, ["compare"], owner="w1") == []
    clock.advance(11)
    assert len(queue.lease(scope_id, ["compare"], owner="w1")) == 1


# ------------------------------------------------- fencing / commit paths


def test_commit_if_current_and_complete(queue, store, scope_id):
    jid = _enq(queue, store, scope_id)
    job = queue.lease(scope_id, ["compare"], owner="w1")[0]
    with store.tx() as conn:
        assert queue.commit_if_current(conn, jid, "w1", job["generation"]) is True
        assert queue.commit_if_current(conn, jid, "w2", job["generation"]) is False
        assert queue.commit_if_current(conn, jid, "w1", job["generation"] + 9) is False
        assert queue.complete(conn, jid, "w1", job["generation"]) is True
    assert queue.list(scope_id, state="succeeded")[0]["job_id"] == jid


def test_stale_worker_cannot_overwrite(queue, store, clock, scope_id):
    """Expired lease → reclaimed → re-leased: old worker is fenced out."""
    jid = _enq(queue, store, scope_id)
    old = queue.lease(scope_id, ["compare"], owner="old", lease_s=5)[0]
    clock.advance(10)  # lease expires
    with store.tx() as conn:
        assert queue.reclaim_expired(conn, clock()) == 1
    new = queue.lease(scope_id, ["compare"], owner="new")[0]
    assert new["generation"] == old["generation"] + 2  # reclaim + re-lease
    with store.tx() as conn:
        # the paused old worker wakes up — its generation no longer matches
        assert queue.commit_if_current(conn, jid, "old", old["generation"]) is False
        assert queue.complete(conn, jid, "old", old["generation"]) is False
        assert queue.complete(conn, jid, "new", new["generation"]) is True
    assert queue.list(scope_id, state="succeeded")[0]["job_id"] == jid


def test_reclaim_expired(queue, store, clock, scope_id):
    jid = _enq(queue, store, scope_id)
    queue.lease(scope_id, ["compare"], owner="w1", lease_s=5)
    clock.advance(6)
    with store.tx() as conn:
        n = queue.reclaim_expired(conn, clock())
    assert n == 1
    job = queue.list(scope_id)[0]
    assert job["state"] == "retry_wait"
    assert job["not_before_us"] == clock()
    assert job["lease_owner"] is None


def test_reclaim_ignores_live_leases(queue, store, clock, scope_id):
    _enq(queue, store, scope_id)
    queue.lease(scope_id, ["compare"], owner="w1", lease_s=60)
    with store.tx() as conn:
        assert queue.reclaim_expired(conn, clock()) == 0


# -------------------------------------------------------------------- fail


def test_fail_retryable_backoff(queue, store, clock, scope_id):
    jid = _enq(queue, store, scope_id)
    job = queue.lease(scope_id, ["compare"], owner="w1")[0]
    with store.tx() as conn:
        st = queue.fail(
            conn, jid, "w1", job["generation"], "REMOTE_BUSY", retryable=True
        )
    assert st == JobState.RETRY_WAIT
    row = queue.list(scope_id)[0]
    # base 2s * jitter(0.5..1.5) → between 1s and 3s out
    delta = row["not_before_us"] - clock()
    assert 1_000_000 <= delta <= 3_000_000
    assert row["error_code"] == "REMOTE_BUSY"


def test_fail_max_attempts_then_failed(queue, store, clock, scope_id):
    jid = _enq(queue, store, scope_id)
    # exhaust attempts: lease+fail 3 times (max_attempts=3)
    for i in range(3):
        clock.advance(400)
        job = queue.lease(scope_id, ["compare"], owner="w1")[0]
        with store.tx() as conn:
            st = queue.fail(
                conn, jid, "w1", job["generation"], "X", retryable=True, max_attempts=3
            )
        if i < 2:
            assert st == JobState.RETRY_WAIT
        else:
            assert st == JobState.FAILED
    assert queue.list(scope_id)[0]["state"] == "failed"


def test_fail_non_retryable(queue, store, scope_id):
    jid = _enq(queue, store, scope_id)
    job = queue.lease(scope_id, ["compare"], owner="w1")[0]
    with store.tx() as conn:
        st = queue.fail(
            conn, jid, "w1", job["generation"], "REMOTE_AUTH", retryable=False
        )
    assert st == JobState.FAILED


def test_fail_stale_returns_none(queue, store, clock, scope_id):
    jid = _enq(queue, store, scope_id)
    job = queue.lease(scope_id, ["compare"], owner="w1", lease_s=1)[0]
    clock.advance(2)
    with store.tx() as conn:
        queue.reclaim_expired(conn, clock())
        assert (
            queue.fail(conn, jid, "w1", job["generation"], "X", retryable=True) is None
        )


# ------------------------------------------------------------------ cancel


def test_cancel_queued(queue, store, scope_id):
    jid = _enq(queue, store, scope_id)
    with store.tx() as conn:
        assert queue.cancel(conn, jid) is True
        assert queue.cancel(conn, jid) is False  # already terminal
    assert queue.list(scope_id)[0]["state"] == "cancelled"


def test_cancel_fences_lease_holder(queue, store, scope_id):
    jid = _enq(queue, store, scope_id)
    job = queue.lease(scope_id, ["compare"], owner="w1")[0]
    with store.tx() as conn:
        assert queue.cancel(conn, jid) is True
        assert queue.complete(conn, jid, "w1", job["generation"]) is False


def test_cancel_unknown_raises(queue, store):
    with store.tx() as conn, pytest.raises(VerbatimError) as ei:
        queue.cancel(conn, "nonexistent")
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


# ------------------------------------------------------------------- stats


def test_stats(queue, store, clock, scope_id):
    _enq(queue, store, scope_id)          # stays pending; enqueued at t0
    clock.advance(30)
    _enq(queue, store, scope_id)          # leased+failed below
    s0 = queue.stats(scope_id)
    assert s0["pending"] == 2
    assert s0["oldest_age_us"] is not None and s0["oldest_age_us"] >= 30_000_000
    j = queue.lease(scope_id, ["compare"], owner="w1", limit=1)[0]
    with store.tx() as conn:
        queue.fail(conn, j["job_id"], "w1", j["generation"], "E", retryable=False)
    s = queue.stats(scope_id)
    assert s["pending"] == 1
    assert s["leased"] == 0
    assert s["failed"] == 1
    # FIFO leased the older job; the remaining pending job is the newer one
    assert s["oldest_age_us"] is not None and s["oldest_age_us"] < 1_000_000


def test_stats_empty(queue, scope_id):
    s = queue.stats(scope_id)
    assert s == {"pending": 0, "oldest_age_us": None, "leased": 0, "failed": 0}


# -------------------------------------------- self-transacting call shapes


def test_lease_none_drains_all_scopes(store, clock):
    q = JobQueue(store, clock=clock, rng=random.Random(1))
    with store.tx() as conn:
        j1 = q.enqueue(conn, "scope_a", JobKind.COMPARE, {})
        j2 = q.enqueue(conn, "scope_b", JobKind.COMPARE, {})
    leased = q.lease(None, ["compare"], owner="w1", limit=10)
    assert {j["job_id"] for j in leased} == {j1, j2}


def test_fail_complete_cancel_without_conn(store, clock, scope_id):
    """The ingest-style call shape: methods open their own tx."""
    q = JobQueue(store, clock=clock, rng=random.Random(1))
    with store.tx() as conn:
        jid = q.enqueue(conn, scope_id, JobKind.COMPARE, {})
    job = q.lease(scope_id, ["compare"], owner="w1")[0]
    st = q.fail(jid, "w1", job["generation"], "REMOTE_BUSY", True)
    assert st == JobState.RETRY_WAIT

    with store.tx() as conn:
        j2 = q.enqueue(conn, scope_id, JobKind.COMPARE, {})
    job2 = q.lease(scope_id, ["compare"], owner="w1")[0]
    assert q.complete(j2, "w1", job2["generation"]) is True

    with store.tx() as conn:
        j3 = q.enqueue(conn, scope_id, JobKind.COMPARE, {})
    assert q.cancel(j3) is True


def test_reclaim_without_conn(store, clock, scope_id):
    q = JobQueue(store, clock=clock, rng=random.Random(1))
    with store.tx() as conn:
        jid = q.enqueue(conn, scope_id, JobKind.COMPARE, {})
    job = q.lease(scope_id, ["compare"], owner="w1", lease_s=5)[0]
    clock.advance(10)
    assert q.reclaim_expired() == 1
    with store.tx() as conn:
        assert q.complete(conn, jid, "w1", job["generation"]) is False


def test_max_pending_ctor_alias(store):
    a = JobQueue(store, max_pending=7)
    b = JobQueue(store, cap=7)
    assert a._cap == b._cap == 7


# ------------------------------------------------------------- job events


def test_job_events_bounded_and_ordered(queue, store, clock, scope_id):
    jid = _enq(queue, store, scope_id)
    for _ in range(3):
        clock.advance(400)
        job = queue.lease(scope_id, ["compare"], owner="w1")[0]
        with store.tx() as conn:
            queue.fail(conn, jid, "w1", job["generation"], "E", retryable=True)
    with store.read() as conn:
        from tests.conftest import qrows

        rows = qrows(
            conn,
            "SELECT state FROM job_events WHERE job_id = ? ORDER BY event_seq",
            (jid,),
        )
    states = [r["state"] for r in rows]
    assert states[0] == "queued"
    assert states.count("leased") == 3
    # third fail hits max_attempts=3 → 'failed', so only two retry_waits
    assert states.count("retry_wait") == 2
    assert states[-1] == "failed"
