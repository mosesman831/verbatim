"""V2 queue behavior: reserved lanes, operation keys, commit-time fencing.

The production ``jobs.kind`` CHECK in ``storage/schema.py`` still lists only
the seven original kinds — ``episode_index`` / ``procedure_validate`` /
``replay`` are valid ``JobKind`` members the shipped DDL rejects (a storage
schema gap owned elsewhere; ``enqueue`` surfaces it as ``SCHEMA_UNSUPPORTED``
rather than silently dropping work). ``V2Store`` below rebuilds the same DDL
with the CHECK widened to the full enum so lane/fencing behavior can be
exercised for every registered kind.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import random
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Optional

import pytest

from verbatim.core.time import now_us
from verbatim.core.types import (
    ErrorCode,
    JobKind,
    VerbatimError,
    json_dumps,
    safe_json_loads,
)
from verbatim.jobs.queue import JobQueue
from verbatim.storage.schema import (
    DDL_FTS5,
    DDL_V1,
    DDL_V2,
    DDL_V2_ALTER,
    FTS_TRIGGERS,
)

_V1_KIND_CHECK = (
    "('harvest','admit','compare','embed','reindex','review_apply','purge')"
)
# Derived from the enum itself so v3 kinds (screen, procedure_compile, …)
# stay covered when the shim widens the v1 CHECK.
_ALL_KIND_CHECK = "(" + ",".join(f"'{k.value}'" for k in JobKind) + ")"
# Same for the v2 lane CHECK: v3 adds background/privacy_control/maintenance.
_LANE_CHECK_V2 = "(lane IN ('ordinary','control'))"
_LANE_CHECK_ALL = "(lane IN (" + ",".join(
    f"'{l}'" for l in ("ordinary", "background", "privacy_control",
                      "maintenance", "control")
) + "))"


class V2Store:
    """In-memory store honoring the ``tx()``/``read()``/meta contract at
    schema v2 — the shared ``TestStore`` fixture is v1-only."""

    fts_enabled = True

    def __init__(self) -> None:
        self._conn = sqlite3.connect(
            ":memory:", isolation_level=None, check_same_thread=False
        )
        self._conn.execute("PRAGMA foreign_keys = ON")
        if _V1_KIND_CHECK not in DDL_V1:
            raise AssertionError(
                "DDL_V1 jobs CHECK drifted — refresh the widened patch"
            )
        self._conn.executescript(DDL_V1.replace(_V1_KIND_CHECK, _ALL_KIND_CHECK))
        self._conn.executescript(DDL_V2)
        for stmt in DDL_V2_ALTER.split(";"):
            s = stmt.strip()
            if s:
                self._conn.execute(s.replace(_LANE_CHECK_V2, _LANE_CHECK_ALL))
        self._conn.executescript(DDL_FTS5)
        self._conn.executescript(FTS_TRIGGERS)
        for key_, value in (
            ("schema_version", 2),
            ("projection_generation", 1),
            ("policy_epoch", 0),
        ):
            self._conn.execute(
                "INSERT INTO meta(key, value_json) VALUES (?, ?)",
                (key_, json_dumps(value)),
            )
        self._key = b"test-store-hmac-key"
        self._tx_lock = threading.RLock()
        self._last_event_us = 0

    @contextmanager
    def tx(self):
        with self._tx_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @contextmanager
    def read(self):
        yield self._conn

    def hmac(self, data: bytes) -> bytes:
        return _hmac.new(self._key, data, hashlib.sha256).digest()

    def _meta_get(self, conn: sqlite3.Connection, key: str) -> Any:
        row = conn.execute(
            "SELECT value_json FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return safe_json_loads(row[0]) if row else None

    def _meta_set(self, conn: sqlite3.Connection, key: str, value: Any) -> None:
        conn.execute(
            "INSERT INTO meta(key, value_json) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
            (key, json_dumps(value)),
        )

    def projection_generation(self) -> int:
        v = self._meta_get(self._conn, "projection_generation")
        return int(v) if v is not None else 0

    def bump_generation(self, conn: sqlite3.Connection) -> int:
        nxt = int(self._meta_get(conn, "projection_generation") or 0) + 1
        self._meta_set(conn, "projection_generation", nxt)
        return nxt

    def policy_epoch(self) -> int:
        return int(self._meta_get(self._conn, "policy_epoch") or 0)

    def next_event_us(self) -> int:
        nxt = max(now_us(), self._last_event_us + 1)
        self._last_event_us = nxt
        return nxt


@pytest.fixture()
def v2store() -> V2Store:
    return V2Store()


@pytest.fixture()
def v2queue(v2store: V2Store, clock: Any) -> JobQueue:
    return JobQueue(v2store, cap=100, clock=clock, rng=random.Random(7))


def _enq(queue: JobQueue, store: Any, scope_id: str,
         kind: JobKind = JobKind.COMPARE, refs: Optional[dict] = None,
         **kw: Any) -> str:
    with store.tx() as conn:
        return queue.enqueue(conn, scope_id, kind, refs or {"a": 1}, **kw)


def _job_row(store: Any, job_id: str) -> dict[str, Any]:
    with store.read() as conn:
        cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()
    return dict(zip(cols, row))


# ------------------------------------------------------------------ lanes


def test_control_kinds_default_to_control_lane(v2queue, v2store, scope_id):
    for kind in (JobKind.PURGE, JobKind.REINDEX, JobKind.REVIEW_APPLY):
        jid = _enq(v2queue, v2store, scope_id, kind)
        assert _job_row(v2store, jid)["lane"] == "control", kind
    for kind in (JobKind.COMPARE, JobKind.EMBED, JobKind.EPISODE_INDEX):
        jid = _enq(v2queue, v2store, scope_id, kind)
        assert _job_row(v2store, jid)["lane"] == "ordinary", kind


def test_lane_override_and_validation(v2queue, v2store, scope_id):
    jid = _enq(v2queue, v2store, scope_id, JobKind.COMPARE, lane="control")
    assert _job_row(v2store, jid)["lane"] == "control"
    with v2store.tx() as conn, pytest.raises(VerbatimError) as ei:
        v2queue.enqueue(conn, scope_id, JobKind.COMPARE, {}, lane="vip")
    assert ei.value.code == ErrorCode.VALIDATION


def test_control_lane_leases_ahead_of_earlier_ordinary(v2queue, v2store, scope_id):
    ordinary = _enq(v2queue, v2store, scope_id, JobKind.COMPARE)
    control = _enq(v2queue, v2store, scope_id, JobKind.REINDEX)
    leased = v2queue.lease(scope_id, list(JobKind), owner="w1", limit=2)
    assert [j["job_id"] for j in leased] == [control, ordinary]
    assert leased[0]["lane"] == "control"


def test_lane_filter(v2queue, v2store, scope_id):
    _enq(v2queue, v2store, scope_id, JobKind.COMPARE)
    ctrl = _enq(v2queue, v2store, scope_id, JobKind.REVIEW_APPLY)
    assert [
        j["job_id"]
        for j in v2queue.lease(scope_id, list(JobKind), owner="w1", lane="control")
    ] == [ctrl]
    assert v2queue.lease(scope_id, list(JobKind), owner="w1", lane="control") == []
    leased = v2queue.lease(scope_id, list(JobKind), owner="w2", lane="ordinary")
    assert len(leased) == 1 and leased[0]["kind"] == "compare"


def test_control_enqueue_bypasses_backpressure(v2store, scope_id):
    q = JobQueue(v2store, cap=2, rng=random.Random(1))
    _enq(q, v2store, scope_id, JobKind.COMPARE)
    _enq(q, v2store, scope_id, JobKind.COMPARE)
    with v2store.tx() as conn, pytest.raises(VerbatimError) as ei:
        q.enqueue(conn, scope_id, JobKind.COMPARE, {})
    assert ei.value.code == ErrorCode.BACKPRESSURE
    # The reserved lane is never starved by ordinary backpressure.
    for kind in (JobKind.PURGE, JobKind.REINDEX, JobKind.REVIEW_APPLY):
        _enq(q, v2store, scope_id, kind)


def test_all_kinds_register_and_lease(v2queue, v2store, scope_id):
    for kind in JobKind:
        jid = _enq(v2queue, v2store, scope_id, kind)
        assert _job_row(v2store, jid)["kind"] == kind.value
    leased = v2queue.lease(scope_id, list(JobKind), owner="w", limit=32)
    assert len(leased) == len(JobKind)
    assert leased[0]["lane"] == "control"


# ------------------------------------------------------- operation keys


def test_operation_key_persisted_and_leased(v2queue, v2store, scope_id):
    jid = _enq(v2queue, v2store, scope_id, JobKind.EMBED,
               operation_key="op:embed:1")
    assert _job_row(v2store, jid)["operation_key"] == "op:embed:1"
    job = v2queue.lease(scope_id, [JobKind.EMBED], owner="w1")[0]
    assert job["operation_key"] == "op:embed:1"


def test_operation_key_rejected_on_v1_schema(store, scope_id):
    """A pre-v2 jobs table must refuse the durability key, not drop it."""
    q = JobQueue(store)
    assert not q.supports_durability
    with store.tx() as conn, pytest.raises(VerbatimError) as ei:
        q.enqueue(conn, scope_id, JobKind.COMPARE, {}, operation_key="op:1")
    assert ei.value.code == ErrorCode.SCHEMA_UNSUPPORTED
    with store.tx() as conn, pytest.raises(VerbatimError) as ei:
        q.enqueue(conn, scope_id, JobKind.COMPARE, {}, lane="control")
    assert ei.value.code == ErrorCode.SCHEMA_UNSUPPORTED


def test_new_kinds_rejected_on_v1_schema(store, scope_id):
    """The v1 seven-kind CHECK rejects valid newer JobKinds — that is a
    schema gap, surfaced as SCHEMA_UNSUPPORTED, never a silent drop."""
    q = JobQueue(store)
    for kind in (JobKind.EPISODE_INDEX, JobKind.PROCEDURE_VALIDATE, JobKind.REPLAY):
        with store.tx() as conn, pytest.raises(VerbatimError) as ei:
            q.enqueue(conn, scope_id, kind, {})
        assert ei.value.code == ErrorCode.SCHEMA_UNSUPPORTED, kind


def test_v1_store_lane_semantics_via_kinds(store, clock, scope_id):
    """Without the lane column the control kind set still sorts first."""
    q = JobQueue(store, clock=clock, rng=random.Random(3))
    ordinary = _enq(q, store, scope_id, JobKind.COMPARE)
    purge = _enq(q, store, scope_id, JobKind.PURGE)
    leased = q.lease(scope_id, ["compare", "purge"], owner="w1", limit=2)
    assert [j["job_id"] for j in leased] == [purge, ordinary]


# ------------------------------------------------------------- fencing


def test_assert_lease_accepts_current(v2queue, v2store, scope_id):
    jid = _enq(v2queue, v2store, scope_id)
    job = v2queue.lease(scope_id, ["compare"], owner="w1")[0]
    with v2store.tx() as conn:
        v2queue.assert_lease(conn, jid, "w1", job["generation"])  # no raise


def test_assert_lease_rejects_wrong_owner_and_generation(v2queue, v2store, scope_id):
    jid = _enq(v2queue, v2store, scope_id)
    job = v2queue.lease(scope_id, ["compare"], owner="w1")[0]
    with v2store.tx() as conn, pytest.raises(VerbatimError) as ei:
        v2queue.assert_lease(conn, jid, "w2", job["generation"])
    assert ei.value.code == ErrorCode.LEASE_LOST
    with v2store.tx() as conn, pytest.raises(VerbatimError) as ei:
        v2queue.assert_lease(conn, jid, "w1", job["generation"] + 1)
    assert ei.value.code == ErrorCode.LEASE_LOST


def test_assert_lease_rejects_expired(v2queue, v2store, scope_id, clock):
    jid = _enq(v2queue, v2store, scope_id)
    job = v2queue.lease(scope_id, ["compare"], owner="w1", lease_s=5)[0]
    clock.advance(6)
    with v2store.tx() as conn, pytest.raises(VerbatimError) as ei:
        v2queue.assert_lease(conn, jid, "w1", job["generation"], now_us=clock())
    assert ei.value.code == ErrorCode.LEASE_LOST
    assert ei.value.retryable is True


def test_assert_lease_rejects_cancelled(v2queue, v2store, scope_id):
    jid = _enq(v2queue, v2store, scope_id)
    job = v2queue.lease(scope_id, ["compare"], owner="w1")[0]
    assert v2queue.cancel(jid) is True
    with v2store.tx() as conn, pytest.raises(VerbatimError) as ei:
        v2queue.assert_lease(conn, jid, "w1", job["generation"])
    assert ei.value.code == ErrorCode.LEASE_LOST


def test_stale_worker_fenced_while_replacement_commits(v2queue, v2store, scope_id, clock):
    """Worker-1's lease expires; worker-2 re-leases at a bumped generation
    and commits; worker-1's staged commit is fenced out."""
    jid = _enq(v2queue, v2store, scope_id)
    stale = v2queue.lease(scope_id, ["compare"], owner="w1", lease_s=5)[0]
    clock.advance(6)
    assert v2queue.reclaim_expired(now_us=clock()) == 1
    fresh = v2queue.lease(scope_id, ["compare"], owner="w2", lease_s=5)[0]
    # reclaim bumps the fencing generation, and the re-lease bumps it again
    assert fresh["generation"] > stale["generation"]

    with v2store.tx() as conn, pytest.raises(VerbatimError) as ei:
        v2queue.assert_lease(
            conn, jid, "w1", stale["generation"], now_us=clock()
        )
    assert ei.value.code == ErrorCode.LEASE_LOST

    with v2store.tx() as conn:
        v2queue.assert_lease(conn, jid, "w2", fresh["generation"], now_us=clock())
        assert v2queue.complete(conn, jid, "w2", fresh["generation"]) is True
    assert _job_row(v2store, jid)["state"] == "succeeded"


def test_lease_events_recorded(v2queue, v2store, scope_id):
    jid = _enq(v2queue, v2store, scope_id)
    job = v2queue.lease(scope_id, ["compare"], owner="w1")[0]
    with v2store.tx() as conn:
        assert v2queue.complete(conn, jid, "w1", job["generation"])
    with v2store.read() as conn:
        states = [
            r[0]
            for r in conn.execute(
                "SELECT state FROM job_events WHERE job_id = ? ORDER BY event_seq",
                (jid,),
            )
        ]
    assert states == ["queued", "leased", "succeeded"]
