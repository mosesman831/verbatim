"""Managed worker lifecycle tests (SPEC_V5 §09, V5-33.08/09; E17–E21).

Real on-disk ``Store`` + real ``JobQueue``/``Ingester`` machinery — no
kernel mocks. The facade is a parallel build; these tests exercise the
worker layer's own contract: ref-counted shared drain per store, enrolled
namespaces only, bounded honest close, crash/SIGKILL resume through the
queue's durable lease machinery, and fork safety (subprocess).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace

import pytest

from verbatim.config import VerbatimConfig
from verbatim.core.identity import scope_key
from verbatim.core.time import now_us
from verbatim.core.types import (
    ErrorCode,
    JobKind,
    Provenance,
    Scope,
    SourceEnvelope,
    SourceKind,
    VerbatimError,
)
from verbatim.ingest import Ingester
from verbatim.jobs.queue import JobQueue
from verbatim.memory import worker as W
from verbatim.storage.store import Store

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCOPE = Scope(profile_id="mem", principal_id="alice", conversation_id="c1")
SID = scope_key(SCOPE)
OTHER = scope_key(Scope(profile_id="mem", principal_id="bob", conversation_id="c9"))


@pytest.fixture()
def cfg() -> VerbatimConfig:
    return VerbatimConfig()


@pytest.fixture()
def store(tmp_path):
    s = Store.create(str(tmp_path / "mem.db"))
    yield s
    s.close()


@pytest.fixture()
def ingester(store, cfg) -> Ingester:
    return Ingester(store, cfg)


@pytest.fixture(autouse=True)
def _registry_clean():
    yield
    # Never leak daemon threads between tests.
    W._reset_registry()


def _enq(ingester: Ingester, store, sid=SID, kind=JobKind.EPISODE_INDEX,
         refs=None, **kw) -> str:
    with store.tx() as conn:
        return ingester.jobs.enqueue(conn, sid, kind, refs or {"n": 1}, **kw)


def _job_state(store, job_id: str):
    with store.read() as conn:
        row = conn.execute(
            "SELECT state, error_code, lease_owner FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    return row[0] if row else None


def _job_row(store, job_id: str) -> dict:
    with store.read() as conn:
        cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()
    return dict(zip(cols, row)) if row else {}


def _wait(pred, timeout_s=15.0, interval_s=0.01) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval_s)
    return bool(pred())


def _wait_state(store, job_id, state, timeout_s=15.0) -> bool:
    return _wait(lambda: _job_state(store, job_id) == state, timeout_s)


def _drained(store, sid) -> bool:
    """Honest drained: nothing queued/retrying AND nothing leased mid-flight.

    ``pending`` alone is wrong — a leased job has already left the pending
    count but hasn't committed yet, so reading downstream tables while it
    is leased races its commit.
    """
    st = JobQueue(store).stats(sid)
    return st["pending"] == 0 and st["leased"] == 0


# ------------------------------------------------------------------
# drain behavior (V5-09.01)
# ------------------------------------------------------------------


def test_drains_pending_job_enqueued_before_acquire(store, ingester, cfg):
    """Existing pending obligations resume on open — before any new add
    (V5-09.01): the job was enqueued with no worker running at all."""
    jid = _enq(ingester, store)
    assert _job_state(store, jid) == "queued"

    h = W.acquire(store, cfg, scope_ids=[SID])
    try:
        assert _wait_state(store, jid, "succeeded")
        # Counters update when the drain pass returns — poll, don't race it.
        assert _wait(lambda: h.status()["jobs_processed"] >= 1, timeout_s=5)
        st = h.status()
        assert st["running"] is True
        assert st["mode"] == "managed"
        assert st["jobs_succeeded"] >= 1
        # The leased row was claimed by this worker's owner token.
        events = JobQueue(store).list(SID)
        assert events[0]["state"] == "succeeded"
    finally:
        rep = h.release()
    assert rep.closed and rep.worker_stopped and rep.drained


def test_drains_job_enqueued_after_acquire_and_wake(store, ingester, cfg):
    h = W.acquire(store, cfg, scope_ids=[SID])
    try:
        t0 = time.monotonic()
        jid = _enq(ingester, store)
        h.wake()  # the facade's add() pokes this after enqueue
        assert _wait_state(store, jid, "succeeded")
        pickup_ms = (time.monotonic() - t0) * 1000.0
        # ≤50 ms cadence is the target (V5-33.08); assert a robust bound and
        # keep the measured value visible in output for tuning.
        assert pickup_ms < 2000.0, f"pickup {pickup_ms:.1f}ms"
        print(f"\n[measured enqueue→succeeded pickup: {pickup_ms:.1f} ms]")
        assert W.DEFAULT_IDLE_MS <= 50.0
    finally:
        h.release()


def test_idle_poll_alone_picks_up_work(store, ingester, cfg):
    """No wake() call: the idle cadence still bounds pickup (V5-33.08)."""
    h = W.acquire(store, cfg, scope_ids=[SID])
    try:
        t0 = time.monotonic()
        jid = _enq(ingester, store)  # no wake — pure poll cadence
        assert _wait_state(store, jid, "succeeded")
        pickup_ms = (time.monotonic() - t0) * 1000.0
        assert pickup_ms < 2000.0
        print(f"\n[measured unpoked pickup: {pickup_ms:.1f} ms]")
    finally:
        h.release()


def test_batch_bound_and_all_jobs_drain(store, ingester, cfg):
    h = W.acquire(store, cfg, scope_ids=[SID], batch_size=2)
    try:
        jids = [_enq(ingester, store, refs={"n": i}) for i in range(9)]
        for jid in jids:
            assert _wait_state(store, jid, "succeeded")
        # Counters update when the drain pass returns — poll, don't race it.
        assert _wait(lambda: h.status()["jobs_processed"] >= 9, timeout_s=5)
        st = h.status()
        assert st["jobs_succeeded"] >= 9
        assert st["passes"] >= 2  # bounded batches, multiple passes
    finally:
        h.release()


def test_pipeline_ingest_drains_end_to_end(tmp_path, cfg):
    """A real source capture drains harvest → admit → embed under the
    managed worker — no manual run_pending call."""
    # require_review=False lands the admitted claim ACTIVE — only an ACTIVE
    # claim publishes FTS + the EMBED obligation, so the full
    # harvest→admit→embed chain is exercised end to end.
    cap_cfg = replace(
        cfg,
        capture=replace(cfg.capture, enabled=True),
        embedding=replace(cfg.embedding, backend="hashing"),
        admission=replace(cfg.admission, require_review=False),
    )
    store = Store.create(str(tmp_path / "pipe.db"))
    ingester = Ingester(store, cap_cfg)
    from verbatim.embeddings.encoder import get_encoder

    encoder = get_encoder(cap_cfg)
    store.encoder = encoder  # Engine convention — the worker picks it up
    try:
        env = SourceEnvelope(
            origin="test:worker",
            source_kind=SourceKind.USER_MESSAGE,
            scope=SCOPE,
            speaker_id="alice",
            payload=b"my editor is neovim and my database is postgres",
            event_us=now_us(),
            captured_us=now_us(),
            provenance=Provenance.DIRECT_USER,
        )
        receipt = ingester.ingest(env)
        assert receipt.accepted and receipt.job_ids
        harvest_id = receipt.job_ids[0]

        h = W.acquire(store, cap_cfg, scope_ids=[SID])
        try:
            assert _wait_state(store, harvest_id, "succeeded")
            # Follow-on admit/embed work was enqueued inside the harvest
            # commit — the worker drains those too. Wait for a real terminal
            # drain (nothing queued AND nothing leased mid-flight), not just
            # an empty pending count that races a leased commit.
            assert _wait(lambda: _drained(store, SID))
            with store.read() as conn:
                n_spans = conn.execute(
                    "SELECT COUNT(*) FROM spans"
                ).fetchone()[0]
                n_claims = conn.execute(
                    "SELECT COUNT(*) FROM claims"
                ).fetchone()[0]
            assert n_spans >= 1
            assert n_claims >= 1
            st = h.status()
            assert st["jobs_failed"] == 0, st["last_error"]
            # Embed ran through the store-attached encoder.
            with store.read() as conn:
                n_emb = conn.execute(
                    "SELECT COUNT(*) FROM embeddings"
                ).fetchone()[0]
            assert n_emb >= 1
        finally:
            rep = h.release()
        assert rep.worker_stopped
    finally:
        store.close()


def test_source_job_kind_reaches_terminal_state(store, ingester, cfg):
    """v5 source_* kinds are leased by the managed worker and dispatch to
    the real source-jobs handlers. A job naming a source that does not
    exist reaches the honest terminal state — EVIDENCE_UNAVAILABLE, never
    a silent no-op, never an infinite retry."""
    jid = _enq(ingester, store, kind=JobKind.SOURCE_PROJECT,
               refs={"source_id": "src-1", "revision": 1,
                     "namespace": SID, "producer": "test"})
    h = W.acquire(store, cfg, scope_ids=[SID])
    try:
        assert _wait(lambda: _job_state(store, jid) in ("succeeded", "failed"))
        row = _job_row(store, jid)
        # The registered handler ran and refused a missing source
        # revision — terminal, typed, non-retryable.
        assert row["state"] == "failed"
        assert row["error_code"] == ErrorCode.EVIDENCE_UNAVAILABLE.value
        assert h.status()["running"]
    finally:
        h.release()


# ------------------------------------------------------------------
# shared registry (V5-09.02/09.07)
# ------------------------------------------------------------------


def test_two_facades_share_one_worker(store, cfg, tmp_path):
    store2 = Store.open(store.path, hmac_key_path=store.key_path)
    try:
        h1 = W.acquire(store, cfg, scope_ids=[SID])
        h2 = W.acquire(store2, cfg, scope_ids=[SID])
        assert W.registry_size() == 1
        assert h1.worker is h2.worker

        r1 = h1.release()
        # Non-last close: detached, but the sibling's worker keeps running.
        assert r1.closed and not r1.worker_stopped
        assert any("shared" in w for w in r1.warnings)
        assert h2.status()["running"]
        assert W.registry_size() == 1

        # The surviving owner still drains — closing h1 stopped nothing.
        ing = Ingester(store2, cfg)
        jid = _enq(ing, store2)
        h2.wake()
        assert _wait_state(store, jid, "succeeded")

        r2 = h2.release()
        assert r2.closed and r2.worker_stopped
        assert W.registry_size() == 0
        # The worker's own store is closed only after its thread joined.
        assert getattr(h2.worker.store, "_closed", False)
    finally:
        store2.close()


def test_distinct_stores_get_distinct_workers(tmp_path, cfg):
    a = Store.create(str(tmp_path / "a.db"))
    b = Store.create(str(tmp_path / "b.db"))
    try:
        ha = W.acquire(a, cfg, scope_ids=[SID])
        hb = W.acquire(b, cfg, scope_ids=[SID])
        assert W.registry_size() == 2
        assert ha.worker is not hb.worker
        ra, rb = ha.release(), hb.release()
        assert ra.worker_stopped and rb.worker_stopped
        assert W.registry_size() == 0
    finally:
        a.close()
        b.close()


def test_acquire_after_last_release_starts_fresh_worker(store, cfg):
    h1 = W.acquire(store, cfg, scope_ids=[SID])
    w1 = h1.worker
    h1.release()
    assert W.registry_size() == 0
    h2 = W.acquire(store, cfg, scope_ids=[SID])
    try:
        assert h2.worker is not w1
        assert h2.status()["running"]
    finally:
        h2.release()


def test_store_identity_stable_across_connections(store, cfg, tmp_path):
    store2 = Store.open(store.path, hmac_key_path=store.key_path)
    try:
        assert W.store_identity(store) == W.store_identity(store2)
    finally:
        store2.close()
    other = Store.create(str(tmp_path / "other.db"))
    try:
        assert W.store_identity(store) != W.store_identity(other)
    finally:
        other.close()


# ------------------------------------------------------------------
# enrolled namespaces only (V5-09.13)
# ------------------------------------------------------------------


def test_unenrolled_namespace_is_never_leased(store, ingester, cfg):
    foreign = _enq(ingester, store, sid=OTHER)
    h = W.acquire(store, cfg, scope_ids=[SID])
    try:
        own = _enq(ingester, store)
        h.wake()
        assert _wait_state(store, own, "succeeded")
        # Give the loop several passes — the foreign job must remain
        # untouched: the worker leases only enrolled namespaces.
        time.sleep(0.4)
        row = _job_row(store, foreign)
        assert row["state"] == "queued"
        assert row["lease_owner"] is None
        # Backlog introspection never sees the other namespace either.
        assert h.status()["backlog_estimate"]["jobs_pending"] == 0
        # Enrolling the namespace lets the same worker drain it.
        h.worker.enroll(OTHER)
        assert _wait_state(store, foreign, "succeeded")
    finally:
        h.release()


def test_remote_and_optional_kinds_are_not_leased(store, ingester, cfg):
    """V5-09.04: connector_pull / projection_sync are outside the
    local_memory permitted set — they stay honestly queued, never failed
    or silently executed by an unauthorized helper."""
    jid = _enq(ingester, store, kind=JobKind.CONNECTOR_PULL,
               refs={"connector": "x"})
    h = W.acquire(store, cfg, scope_ids=[SID])
    try:
        time.sleep(0.4)
        row = _job_row(store, jid)
        assert row["state"] == "queued"
        assert row["lease_owner"] is None
        assert JobKind.CONNECTOR_PULL not in W.MANAGED_JOB_KINDS
        assert JobKind.PROJECTION_SYNC not in W.MANAGED_JOB_KINDS
    finally:
        rep = h.release()
    # The unleased job counts honestly as pending backlog at close.
    assert not rep.drained
    assert "pending_jobs" in rep.incomplete


# ------------------------------------------------------------------
# errors, honesty, close semantics (V5-09.05/06/07/08/09)
# ------------------------------------------------------------------


def test_failed_job_is_terminal_and_worker_survives(store, ingester, cfg):
    bad = _enq(ingester, store, kind=JobKind.COMPARE, refs={"a": 1})
    good = _enq(ingester, store)
    h = W.acquire(store, cfg, scope_ids=[SID])
    try:
        # 'compare' has no handler in this build — the dispatcher fails it
        # loudly (CAPABILITY_UNAVAILABLE is non-retryable → terminal).
        assert _wait_state(store, bad, "failed")
        assert _wait_state(store, good, "succeeded")
        assert _wait(lambda: h.status()["jobs_failed"] >= 1, timeout_s=5)
        st = h.status()
        assert st["running"]
        assert st["jobs_failed"] >= 1
        assert st["jobs_succeeded"] >= 1
        assert st["last_error"]["code"] == ErrorCode.CAPABILITY_UNAVAILABLE.value
        # Error reporting is code+kind only — no payload text leaks.
        assert "input_refs" not in json.dumps(st["last_error"])
    finally:
        h.release()


def test_privacy_lane_drains_ahead_of_ordinary(store, ingester, cfg):
    """V5-09.05/V5-33.09: control/privacy lanes outrank ordinary work in
    dequeue order — verified through the real queue ordering, not a mock."""
    ordinary = [_enq(ingester, store, refs={"n": i}) for i in range(6)]
    ctrl = _enq(ingester, store, kind=JobKind.REINDEX, refs={})

    order: list[str] = []
    w = W.ManagedWorker(store, cfg)  # direct construction (test seam)
    orig = w._ingester._execute

    def _record(job, owner):
        order.append(job["kind"])
        return orig(job, owner)

    w._ingester._execute = _record
    w.enroll(SID)
    w.start()
    try:
        assert _wait(lambda: len(order) >= 7)
    finally:
        rep = w.stop()
    assert rep["worker_stopped"]
    assert order[0] == JobKind.REINDEX.value  # control lane first
    assert _job_state(store, ctrl) == "succeeded"
    for jid in ordinary:
        assert _job_state(store, jid) == "succeeded"


def test_close_bounded_and_idempotent(store, ingester, cfg):
    h = W.acquire(store, cfg, scope_ids=[SID])
    _enq(ingester, store)
    assert _wait(lambda: h.status()["jobs_processed"] >= 1)
    t0 = time.monotonic()
    r1 = h.release(timeout_ms=5000)
    elapsed = time.monotonic() - t0
    assert r1.closed and r1.worker_stopped and r1.drained
    assert elapsed < 5.0  # bounded close (typically ~ms)
    r2 = h.release(timeout_ms=5000)  # repeated close is safe + replays
    assert r2.closed and r2.worker_stopped
    assert r1.incomplete == r2.incomplete


def test_close_reports_pending_backlog_honestly(store, ingester, cfg):
    """A not-yet-due job cannot be claimed drained — the close report
    carries it as pending rather than promising an emptied queue
    (V5-09.07)."""
    _enq(ingester, store, not_before_us=now_us() + 3600 * 10**6)
    h = W.acquire(store, cfg, scope_ids=[SID])
    rep = h.release(timeout_ms=5000)
    assert rep.closed and rep.worker_stopped
    assert rep.drained is False
    assert "pending_jobs" in rep.incomplete
    assert rep.pending_obligations >= 1


def test_join_timeout_reports_incomplete_then_recovers(store, ingester, cfg):
    """E19/V5-09.08: a cooperative shutdown that misses its deadline stays
    honest — worker_stopped=False, incomplete marker, resources retained;
    a repeated close retries the join and completes."""
    h = W.acquire(store, cfg, scope_ids=[SID])
    w = h.worker
    orig = w._ingester.drain_report

    def _slow(*a, **kw):
        time.sleep(0.4)
        return orig(*a, **kw)

    w._ingester.drain_report = _slow
    w.wake()
    time.sleep(0.05)  # let the loop enter the slowed pass
    t0 = time.monotonic()
    r1 = h.release(timeout_ms=30)
    assert time.monotonic() - t0 < 1.0  # bounded, never waits the 0.4s pass
    assert r1.closed
    assert r1.worker_stopped is False
    assert "worker_join_timeout" in r1.incomplete
    # Store NOT closed while the worker may still be using it (V5-09.08).
    assert not getattr(w.store, "_closed", True)
    # Repeated close retries the join with a fresh budget.
    r2 = h.release(timeout_ms=5000)
    assert r2.worker_stopped is True
    assert getattr(w.store, "_closed", False)


def test_external_mode_status_and_no_helper(store, ingester, cfg):
    """worker='external' starts nothing — the facade simply never calls
    acquire; obligations stay honestly pending (V5-09.03)."""
    st = W.external_status()
    assert st["mode"] == "external" and st["running"] is False
    assert W.registry_size() == 0
    jid = _enq(ingester, store)
    time.sleep(0.3)
    assert _job_state(store, jid) == "queued"  # honest pending, not hidden


def test_acquire_validates(store, cfg, tmp_path):
    with pytest.raises(VerbatimError) as ei:
        W.acquire(object(), cfg, scope_ids=[SID])
    assert ei.value.code is ErrorCode.VALIDATION

    ro = Store.open(store.path, hmac_key_path=store.key_path, readonly=True)
    try:
        with pytest.raises(VerbatimError) as ei2:
            W.acquire(ro, cfg, scope_ids=[SID])
        assert ei2.value.code is ErrorCode.CONFIG_INVALID
    finally:
        ro.close()

    with pytest.raises(VerbatimError):
        W.acquire(store, cfg, scope_ids=["bad scope!!"])


def test_status_shape_and_no_foreign_data(store, ingester, cfg):
    _enq(ingester, store, sid=OTHER)  # foreign backlog — must not leak
    h = W.acquire(store, cfg, scope_ids=[SID])
    try:
        st = h.status()
        for k in ("mode", "running", "state", "owner", "jobs_processed",
                  "jobs_succeeded", "jobs_failed", "last_error",
                  "backlog_estimate", "uptime_ms"):
            assert k in st
        assert st["backlog_estimate"]["jobs_pending"] == 0
        assert st["namespaces_enrolled"] == 1
        assert OTHER not in json.dumps(st)
    finally:
        h.release()


# ------------------------------------------------------------------
# crash / SIGKILL resume (V5-09.11, E21) — subprocess isolation
# ------------------------------------------------------------------

_SIGKILL_CHILD = r"""
import os, signal, sys
sys.path.insert(0, %r)
from verbatim.config import VerbatimConfig
from verbatim.core.types import JobKind
from verbatim.ingest import Ingester
from verbatim.storage.store import Store

mode, path, sid, marker = sys.argv[1:5]
store = Store.open(path)
ing = Ingester(store, VerbatimConfig())
with store.tx() as conn:
    jid = ing.jobs.enqueue(conn, sid, JobKind.EPISODE_INDEX, {"mode": mode})
if mode == "leased":
    jobs = ing.jobs.lease(sid, [JobKind.EPISODE_INDEX], owner="crash-worker",
                          lease_s=0.5)
    assert jobs and jobs[0]["job_id"] == jid, jobs
with open(marker, "w") as f:
    f.write(jid)
    f.flush()
    os.fsync(f.fileno())
os.kill(os.getpid(), signal.SIGKILL)
"""


def _run_sigkill_child(tmp_path, mode: str):
    script = tmp_path / "crash_child.py"
    script.write_text(_SIGKILL_CHILD % REPO)
    marker = tmp_path / "job_id.txt"
    proc = subprocess.run(
        [sys.executable, str(script), mode,
         str(tmp_path / "mem.db"), SID, str(marker)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == -signal.SIGKILL, proc.stderr
    assert marker.exists()
    return marker.read_text().strip()


def test_sigkill_queued_job_resumes_on_reopen(store, ingester, cfg, tmp_path):
    """Accepted work survives a SIGKILL: the durable queued job drains the
    moment a managed worker opens the store (V5-09.11)."""
    jid = _run_sigkill_child(tmp_path, "queued")
    assert _job_state(store, jid) == "queued"
    h = W.acquire(store, cfg, scope_ids=[SID])
    try:
        assert _wait_state(store, jid, "succeeded")
        assert _wait(lambda: h.status()["jobs_succeeded"] >= 1, timeout_s=5)
    finally:
        h.release()


def test_sigkill_mid_lease_is_fenced_and_reclaimed(store, ingester, cfg,
                                                   tmp_path):
    """A worker killed mid-lease leaves a durable obligation: the persisted
    lease expiry + generation fencing reclaim it and a fresh worker commits
    the effects — the dead owner can never overwrite them."""
    jid = _run_sigkill_child(tmp_path, "leased")
    row = _job_row(store, jid)
    assert row["state"] == "leased"
    assert row["lease_owner"] == "crash-worker"

    h = W.acquire(store, cfg, scope_ids=[SID])
    try:
        # The 0.5s lease expires; reclaim_expired (inside every drain pass)
        # returns it to retry_wait and the worker completes it.
        assert _wait_state(store, jid, "succeeded")
        row = _job_row(store, jid)
        assert row["state"] == "succeeded"
        assert row["generation"] >= 2  # reclaimed + re-leased, fenced
    finally:
        h.release()


# ------------------------------------------------------------------
# fork safety (V5-09.10, E20) — real os.fork in a disposable subprocess
# ------------------------------------------------------------------

_FORK_CHILD = r"""
import json, os, sys, time
import faulthandler
faulthandler.dump_traceback_later(90, exit=True)
sys.path.insert(0, %r)
from verbatim.config import VerbatimConfig
from verbatim.core.types import JobKind
from verbatim.ingest import Ingester
from verbatim.storage.store import Store
from verbatim.memory import worker as W

path, sid, out = sys.argv[1:4]
cfg = VerbatimConfig()
store = Store.create(path)
ing = Ingester(store, cfg)
h = W.acquire(store, cfg, scope_ids=[sid])

def state_of(s, jid):
    with s.read() as conn:
        return conn.execute("SELECT state FROM jobs WHERE job_id=?",
                            (jid,)).fetchone()[0]

def wait(s, jid, want, t=15.0):
    end = time.monotonic() + t
    st = "?"
    while time.monotonic() < end:
        st = state_of(s, jid)
        if st == want:
            return True
        time.sleep(0.02)
    return False

with store.tx() as conn:
    j1 = ing.jobs.enqueue(conn, sid, JobKind.EPISODE_INDEX, {"n": 1})
h.wake()
result = {"pre_fork_drained": wait(store, j1, "succeeded")}

pid = os.fork()
if pid == 0:
    child = {}
    try:
        # The whole V5-09.10 contract for an inherited object: report the
        # forked state honestly and detach WITHOUT touching any inherited
        # lock, thread object, or SQLite connection. The child deliberately
        # does NOT Store.open/acquire/thread-start here — running arbitrary
        # code in a live-forked child of a multithreaded process can
        # deadlock on locks a dead-in-child thread held at the fork
        # instant (CPython warns; sqlite forbids carrying connections
        # across fork). "A clean child process opens the store fresh and
        # drains durable work" is covered by the SIGKILL subprocess tests
        # below, which exercise the same reopen path safely.
        st = h.status()
        rep = h.release()
        child = {
            "inherited_status_forked": bool(st.get("forked")),
            "inherited_state": st.get("state"),
            "release_incomplete": list(rep.incomplete),
            "release_worker_stopped": rep.worker_stopped,
        }
    except Exception as exc:
        child = {"error": type(exc).__name__ + ": " + str(exc)}
    with open(out + ".child", "w") as f:
        json.dump(child, f)
    os._exit(0)   # skip atexit — never run parent-inherited cleanup

_, status = os.waitpid(pid, 0)
# Parent's worker must be unaffected by anything the child did — it still
# drains new work. Wait for the drain BEFORE release: release stops the
# worker, so waiting after it could never observe the commit.
with store.tx() as conn:
    j3 = ing.jobs.enqueue(conn, sid, JobKind.EPISODE_INDEX, {"n": 3})
h.wake()
post_fork_drained = wait(store, j3, "succeeded")
rep = h.release(timeout_ms=5000)
result["parent"] = {
    "wait_status": status,
    "post_fork_drained": post_fork_drained,
    "worker_stopped": rep.worker_stopped,
}
store.close()
with open(out, "w") as f:
    json.dump(result, f)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX-only")
def test_fork_inherited_handle_is_inert_parent_unaffected(tmp_path):
    script = tmp_path / "fork_child.py"
    script.write_text(_FORK_CHILD % REPO)
    out = tmp_path / "fork_result.json"
    proc = subprocess.run(
        [sys.executable, str(script),
         str(tmp_path / "fork.db"), SID, str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text())
    child = json.loads((tmp_path / "fork_result.json.child").read_text())

    assert result["pre_fork_drained"] is True
    # The child observed the inherited handle as forked and detached without
    # touching any inherited lock/thread/connection — its release is a
    # no-op that must NOT report the parent's worker stopped.
    assert child["inherited_status_forked"] is True
    assert child["inherited_state"] == "forked_child"
    assert "forked_child" in child["release_incomplete"]
    assert child["release_worker_stopped"] is False
    # Parent kept draining through its own worker after the fork — the
    # child's inert detach touched nothing the parent relies on.
    assert result["parent"]["post_fork_drained"] is True
    assert result["parent"]["worker_stopped"] is True


def test_pid_mismatch_marks_worker_defunct_in_process(store, cfg):
    """The loop's PID check is defense-in-depth: simulate the post-fork
    observation by mutating the recorded PID — the worker reports
    forked_child and stops without touching the store."""
    h = W.acquire(store, cfg, scope_ids=[SID])
    w = h.worker
    w._pid = w._pid + 10**6  # simulate foreign-PID observation
    assert w.status()["state"] == "forked_child"
    assert _wait(lambda: not w.status()["running"])
    h.release()


# ------------------------------------------------------------------
# unblock-first drain (V6-02.08, v6_contracts §3) — barrier-marked
# sources drain ahead of ordinary work, never ahead of privacy lanes;
# marks are reaped only once their obligations settle.
# ------------------------------------------------------------------


def _cap_cfg(cfg) -> VerbatimConfig:
    return replace(cfg, capture=replace(cfg.capture, enabled=True))


def _capture_source(ingester: Ingester, text: str = "blocked source") -> str:
    """Real capture + its v5 source jobs (source_project/source_embed);
    returns the source_id."""
    from verbatim.jobs import source_jobs as sj
    from verbatim.readiness import ingest_receipt_id

    env = SourceEnvelope(
        origin="test:worker-prio",
        source_kind=SourceKind.USER_MESSAGE,
        scope=SCOPE,
        speaker_id="alice",
        payload=text.encode("utf-8"),
        event_us=now_us(),
        captured_us=now_us(),
        provenance=Provenance.DIRECT_USER,
    )
    r = ingester.ingest(env)
    sid = r.accepted[0]
    with ingester.store.tx() as conn:
        sj.enqueue_source_jobs(
            conn, ingester.store, receipt_id=ingest_receipt_id(sid, 1)
        )
    return sid


def test_worker_drains_blocked_source_first_and_clears_mark(store, cfg):
    """V6-02.08 end-to-end: a barrier-marked source's projection jobs run
    ahead of ordinary work enqueued before them; once the source's
    obligations settle the mark is reaped (v6_contracts §3)."""
    from verbatim.storage import commit_notify

    cap_cfg = _cap_cfg(cfg)
    ing = Ingester(store, cap_cfg)
    _enq(ing, store, refs={"n": 1})          # pre-existing ordinary work
    _enq(ing, store, refs={"n": 2})
    sid = _capture_source(ing)
    commit_notify.note_barrier_sources(store.path, [sid])
    assert sid in commit_notify.blocked_sources(store.path)

    order: list[str] = []
    w = W.ManagedWorker(store, cap_cfg)
    orig = w._ingester._execute

    def _rec(job, owner):
        order.append(job["kind"])
        return orig(job, owner)

    w._ingester._execute = _rec
    w.enroll(SID)
    w.start()
    try:
        # source_project + source_embed + 2 ordinary + harvest + admit.
        assert _wait(lambda: len(order) >= 6)
        assert _wait(
            lambda: commit_notify.blocked_sources(store.path) == frozenset()
        )
    finally:
        rep = w.stop()
    assert rep["worker_stopped"]
    assert order[0] == "source_project"
    assert order[1] == "source_embed"
    assert order.index("episode_index") > 1


def test_worker_privacy_lane_still_beats_marked_source(store, cfg):
    """A marked source never jumps ahead of a privacy/control job —
    the unblock-first pass runs after the non-ordinary scan."""
    from verbatim.storage import commit_notify

    cap_cfg = _cap_cfg(cfg)
    ing = Ingester(store, cap_cfg)
    _enq(ing, store, refs={"n": 1})
    _enq(ing, store, kind=JobKind.REINDEX, refs={})
    sid = _capture_source(ing)
    commit_notify.note_barrier_sources(store.path, [sid])

    order: list[str] = []
    w = W.ManagedWorker(store, cap_cfg)
    orig = w._ingester._execute
    w._ingester._execute = lambda job, owner: (
        order.append(job["kind"]), orig(job, owner))[1]
    w.enroll(SID)
    w.start()
    try:
        assert _wait(lambda: len(order) >= 6)
    finally:
        rep = w.stop()
    assert rep["worker_stopped"]
    assert order[0] == "reindex"
    assert order[1] == "source_project"


def test_worker_keeps_mark_while_source_job_pending(store, cfg):
    """Partial settle reaps only the settled ids: a marked source whose
    projection job is still owed keeps its mark (advisory hint, never a
    stale clear)."""
    from verbatim.storage import commit_notify

    cap_cfg = _cap_cfg(cfg)
    ing = Ingester(store, cap_cfg)
    settled_src = _capture_source(ing, "settled source")
    held_src = "src-held-future"
    _enq(
        ing,
        store,
        kind=JobKind.SOURCE_PROJECT,
        refs={
            "source_id": held_src,
            "revision": 1,
            "namespace": SID,
            "producer": "test",
        },
        not_before_us=now_us() + 3600 * 10**6,
    )
    commit_notify.note_barrier_sources(store.path, [settled_src, held_src])

    w = W.ManagedWorker(store, cap_cfg)
    w.enroll(SID)
    w.start()
    try:
        # The settled source's mark is reaped; the held source — whose
        # job cannot drain yet — keeps its mark.
        assert _wait(
            lambda: commit_notify.blocked_sources(store.path)
            == frozenset({held_src})
        )
    finally:
        rep = w.stop()
    assert rep["worker_stopped"]
    assert commit_notify.blocked_sources(store.path) == frozenset({held_src})
    commit_notify.clear_barrier_sources(store.path)


def test_worker_tolerates_commit_notify_failure(store, ingester, cfg,
                                                monkeypatch):
    """The mark is advisory ordering, never correctness: a
    ``blocked_sources`` registry that raises reads as 'nothing blocked'
    and the drain proceeds normally."""
    import verbatim.storage.commit_notify as cn

    def _boom(path):
        raise RuntimeError("registry exploded")

    monkeypatch.setattr(cn, "blocked_sources", _boom)
    jid = _enq(ingester, store)
    w = W.ManagedWorker(store, cfg)
    w.enroll(SID)
    w.start()
    try:
        assert _wait_state(store, jid, "succeeded")
    finally:
        rep = w.stop()
    assert rep["worker_stopped"]
