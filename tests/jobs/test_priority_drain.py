"""Unblock-first priority drain (SPEC_V6 V6-02.08, v6_contracts §3).

Real on-disk ``Store`` databases — the priority pass runs through the
production ``drain_report`` → ``JobQueue.lease_priority`` path, never a
mock. The contract: marked ``source_project``/``source_embed`` jobs (the
sources a live session barrier waits on) drain ahead of ordinary work,
bounded by ``limit``, under identical generation fencing — while
privacy/correctness lanes still dequeue first and the mark is never a
second scheduler.
"""

from __future__ import annotations

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
from verbatim.jobs import source_jobs as sj
from verbatim.readiness import ingest_receipt_id
from verbatim.storage.store import Store

SCOPE = Scope(profile_id="p", principal_id="alice", conversation_id="c1")
SID = scope_key(SCOPE)
OTHER = scope_key(Scope(profile_id="p", principal_id="bob", conversation_id="c9"))

NON_ORDINARY = ("control", "privacy_control", "maintenance", "background")


@pytest.fixture()
def cfg() -> VerbatimConfig:
    c = VerbatimConfig()
    return replace(c, capture=replace(c.capture, enabled=True))


@pytest.fixture()
def store(tmp_path):
    s = Store.create(str(tmp_path / "prio.db"))
    yield s
    s.close()


@pytest.fixture()
def ingester(store, cfg) -> Ingester:
    return Ingester(store, cfg)


def _enq(ingester: Ingester, store, sid=SID, kind=JobKind.EPISODE_INDEX,
         refs=None, **kw) -> str:
    with store.tx() as conn:
        return ingester.jobs.enqueue(conn, sid, kind, refs or {"n": 1}, **kw)


def _capture(ingester: Ingester, text: str = "a marked source") -> str:
    """Real capture + its v5 source jobs; returns the source_id."""
    env = SourceEnvelope(
        origin="test",
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
        sj.enqueue_source_jobs(conn, ingester.store,
                               receipt_id=ingest_receipt_id(sid, 1))
    return sid


def _job_state(store, job_id: str):
    with store.read() as conn:
        row = conn.execute(
            "SELECT state FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    return row[0] if row else None


def _states(store) -> dict:
    with store.read() as conn:
        return {
            r[0]: r[1]
            for r in conn.execute("SELECT job_id, state FROM jobs").fetchall()
        }


# ------------------------------------------------------------------
# lease_priority (queue-level)
# ------------------------------------------------------------------


def test_lease_priority_filters_by_source_id(store, ingester):
    """Only jobs whose payload ``$.source_id`` is marked are leased."""
    ja = _enq(ingester, store, kind=JobKind.SOURCE_PROJECT,
              refs={"source_id": "src-a", "revision": 1})
    jb = _enq(ingester, store, kind=JobKind.SOURCE_PROJECT,
              refs={"source_id": "src-b", "revision": 1})
    _enq(ingester, store, kind=JobKind.COMPARE, refs={"a": 1})

    leased = ingester.jobs.lease_priority(
        SID, [JobKind.SOURCE_PROJECT, JobKind.COMPARE],
        source_ids={"src-a"}, owner="w1", limit=4,
    )
    assert [j["job_id"] for j in leased] == [ja]
    job = leased[0]
    # Same fencing shape as lease(): owner stamped, generation bumped.
    assert job["state"] == "leased"
    assert job["lease_owner"] == "w1"
    assert job["generation"] == 1
    assert job["attempts"] == 1
    # The unmarked sibling and the non-source job stay queued.
    assert _job_state(store, jb) == "queued"
    # And the leased row completes under the same fence.
    with store.tx() as conn:
        assert ingester.jobs.complete(conn, ja, "w1", job["generation"])


def test_lease_priority_empty_and_foreign(store, ingester):
    jb = _enq(ingester, store, kind=JobKind.SOURCE_PROJECT,
              refs={"source_id": "src-b", "revision": 1})
    _enq(ingester, store, sid=OTHER, kind=JobKind.SOURCE_PROJECT,
         refs={"source_id": "src-a", "revision": 1})
    q = ingester.jobs
    # An empty or all-invalid set matches nothing.
    assert q.lease_priority(SID, [JobKind.SOURCE_PROJECT],
                          source_ids=set(), owner="w1") == []
    assert q.lease_priority(SID, [JobKind.SOURCE_PROJECT],
                          source_ids={None, 7}, owner="w1") == []
    # A marked id in ANOTHER scope is invisible to this scope's lease.
    assert q.lease_priority(SID, [JobKind.SOURCE_PROJECT],
                          source_ids={"src-a"}, owner="w1") == []
    leased = q.lease_priority(SID, [JobKind.SOURCE_PROJECT],
                            source_ids={"src-b"}, owner="w1")
    assert [j["job_id"] for j in leased] == [jb]


def test_lease_priority_lane_sets_and_order(store, ingester):
    """Lane filtering mirrors lease(): control work first, then the
    marked source job inside the ordinary lane."""
    ctrl = _enq(ingester, store, kind=JobKind.REINDEX, refs={})
    jp = _enq(ingester, store, kind=JobKind.SOURCE_PROJECT,
              refs={"source_id": "src-a", "revision": 1})
    _enq(ingester, store, kind=JobKind.EPISODE_INDEX, refs={"n": 1})

    q = ingester.jobs
    kinds = [JobKind.REINDEX, JobKind.SOURCE_PROJECT, JobKind.EPISODE_INDEX]
    non_ord = q.lease_priority(SID, kinds, owner="w1", lane=NON_ORDINARY)
    assert [j["job_id"] for j in non_ord] == [ctrl]
    marked = q.lease_priority(
        SID, [JobKind.SOURCE_PROJECT], source_ids={"src-a"},
        owner="w1", lane="ordinary",
    )
    assert [j["job_id"] for j in marked] == [jp]
    # Everything already leased → nothing left but the ordinary job.
    rest = q.lease_priority(SID, kinds, owner="w1", lane="ordinary")
    assert len(rest) == 1 and rest[0]["kind"] == "episode_index"


def test_lease_priority_validates(store, ingester):
    q = ingester.jobs
    with pytest.raises(VerbatimError) as ei:
        q.lease_priority(SID, [JobKind.COMPARE], source_ids={"s"},
                         owner="w1", lane="bogus")
    assert ei.value.code == ErrorCode.VALIDATION
    with pytest.raises(VerbatimError):
        q.lease_priority(SID, [], source_ids={"s"}, owner="w1")
    with pytest.raises(VerbatimError):
        q.lease_priority(SID, [JobKind.COMPARE], source_ids={"s"},
                         owner="w1", limit=0)


# ------------------------------------------------------------------
# drain_report(priority_sources=...)
# ------------------------------------------------------------------


def test_drain_marked_source_jobs_run_first(store, ingester):
    """The marked source's jobs dispatch ahead of pre-existing ordinary
    work — before the ordinary dequeue, after the privacy scan."""
    ordinary = [
        _enq(ingester, store, refs={"n": i}) for i in range(3)
    ]
    sid = _capture(ingester)  # enqueues source_project + source_embed last

    order: list = []
    orig = ingester._execute

    def _rec(job, owner):
        order.append((job["kind"], job["job_id"]))
        return orig(job, owner)

    ingester._execute = _rec
    rep = ingester.drain_report(
        scope=SCOPE, limit=16, owner="t", priority_sources={sid}
    )
    assert rep["failed"] == 0, rep["errors"]
    assert rep["processed"] == rep["succeeded"]
    # Both marked source jobs ran via the unblock-first pass, lexical
    # (barrier-settling) job first — ahead of the ordinary jobs that
    # were queued long before them.
    assert order[0][0] == "source_project"
    assert order[1][0] == "source_embed"
    assert rep["priority_processed"] == 2
    kinds = [k for k, _ in order]
    first_ordinary = kinds.index("episode_index")
    assert first_ordinary > 1
    for jid in ordinary:
        assert _job_state(store, jid) == "succeeded"


def test_drain_unmarked_source_not_prioritized(store, ingester):
    """A one-job drain takes ONLY the marked source's job; the unmarked
    sibling source job and ordinary work stay queued."""
    marked = _capture(ingester, "marked source text")
    unmarked = _capture(ingester, "unmarked source text")
    # Find the two sources' source_project job ids.
    with store.read() as conn:
        rows = conn.execute(
            "SELECT job_id, json_extract(input_refs_json, '$.source_id')"
            " FROM jobs WHERE kind = 'source_project'"
        ).fetchall()
    by_src = {r[1]: r[0] for r in rows}
    j_marked, j_unmarked = by_src[marked], by_src[unmarked]

    rep = ingester.drain_report(
        scope=SCOPE, limit=1, owner="t", priority_sources={marked}
    )
    assert rep["processed"] == 1
    assert rep["priority_processed"] == 1
    assert _job_state(store, j_marked) == "succeeded"
    # Unmarked work — including the sibling's own source_project —
    # remains honestly queued.
    assert _job_state(store, j_unmarked) == "queued"
    states = _states(store)
    assert sum(1 for s in states.values() if s == "queued") >= 3


def test_drain_privacy_lane_beats_priority(store, ingester):
    """privacy/control work still dequeues ahead of a marked source —
    the unblock-first pass can never demote a correctness lane."""
    ordinary = _enq(ingester, store, refs={"n": 1})
    ctrl = _enq(ingester, store, kind=JobKind.REINDEX, refs={})
    sid = _capture(ingester)

    order: list = []
    orig = ingester._execute
    ingester._execute = lambda job, owner: (
        order.append(job["kind"]), orig(job, owner))[1]
    rep = ingester.drain_report(
        scope=SCOPE, limit=16, owner="t", priority_sources={sid}
    )
    assert rep["failed"] == 0, rep["errors"]
    assert order[0] == "reindex"          # control lane first
    assert order[1] == "source_project"   # then the marked source
    assert order[2] == "source_embed"
    assert rep["priority_processed"] == 2
    assert _job_state(store, ctrl) == "succeeded"
    assert _job_state(store, ordinary) == "succeeded"


def test_drain_priority_respects_kinds_filter(store, ingester):
    """A drain whose kind set excludes the source kinds cannot pull
    them through the priority pass — the mark is never a backdoor."""
    sid = _capture(ingester)
    ordinary = _enq(ingester, store, refs={"n": 1})
    rep = ingester.drain_report(
        scope=SCOPE, limit=8, owner="t", kinds=[JobKind.EPISODE_INDEX],
        priority_sources={sid},
    )
    assert rep["processed"] == 1
    assert rep["priority_processed"] == 0
    assert _job_state(store, ordinary) == "succeeded"
    with store.read() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE kind = 'source_project'"
            " AND state = 'queued'"
        ).fetchone()[0]
    assert n == 1


def test_drain_priority_with_pinned_lane(store, ingester):
    """``lane='ordinary'`` + marked sources: the marked jobs still go
    first inside the pinned lane; nothing outside it is touched."""
    ordinary = _enq(ingester, store, refs={"n": 1})
    ctrl = _enq(ingester, store, kind=JobKind.REINDEX, refs={})
    sid = _capture(ingester)

    order: list = []
    orig = ingester._execute
    ingester._execute = lambda job, owner: (
        order.append(job["kind"]), orig(job, owner))[1]
    rep = ingester.drain_report(
        scope=SCOPE, limit=8, owner="t", lane="ordinary",
        priority_sources={sid},
    )
    assert order[0] == "source_project"
    assert rep["priority_processed"] == 2
    # The control-lane job was never eligible under the pinned lane.
    assert _job_state(store, ctrl) == "queued"
    assert _job_state(store, ordinary) == "succeeded"


def test_drain_without_priority_matches_lane_order(store, ingester):
    """No mark → the single global lane scan, unchanged: control work
    first, then ordinary throughput in enqueue order."""
    j1 = _enq(ingester, store, refs={"n": 1})
    j2 = _enq(ingester, store, refs={"n": 2})
    ctrl = _enq(ingester, store, kind=JobKind.REINDEX, refs={})

    order: list = []
    orig = ingester._execute
    ingester._execute = lambda job, owner: (
        order.append(job["kind"]), orig(job, owner))[1]
    rep = ingester.drain_report(scope=SCOPE, limit=8, owner="t")
    assert order == ["reindex", "episode_index", "episode_index"]
    assert rep["priority_processed"] == 0
    assert rep["processed"] == 3


def test_drain_empty_marked_set_is_noop(store, ingester):
    """``priority_sources=set()`` behaves exactly like no mark at all."""
    _enq(ingester, store, refs={"n": 1})
    rep = ingester.drain_report(
        scope=SCOPE, limit=8, owner="t", priority_sources=set()
    )
    assert rep["processed"] == 1
    assert rep["priority_processed"] == 0


def test_priority_jobs_still_settle_normally(store, ingester):
    """A marked job that fails reaches the same terminal outcome as any
    drain — the priority pass changes ordering, never settlement."""
    sid = "src-missing"
    jf = _enq(ingester, store, kind=JobKind.SOURCE_PROJECT,
              refs={"source_id": sid, "revision": 1, "namespace": SID,
                    "producer": "test"})
    rep = ingester.drain_report(
        scope=SCOPE, limit=8, owner="t", priority_sources={sid}
    )
    # EVIDENCE_UNAVAILABLE is terminal-nonretryable — failed, not looped.
    assert rep["processed"] == 1
    assert rep["priority_processed"] == 1
    assert rep["failed"] == 1
    assert _job_state(store, jf) == "failed"
