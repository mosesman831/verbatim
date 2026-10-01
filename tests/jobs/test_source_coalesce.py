"""Ingest coalescing bounds (SPEC_V8 V8-13.05).

Real on-disk ``Store`` databases — the shared sibling commit runs
through the production ``drain_report`` → ``_drain_siblings`` path.

The contract: same-scope same-kind pending siblings may commit inside
the leased job's fenced write tx, but (a) a wall-clock bound releases
the claimed-but-unprocessed tail back to ``queued`` so foreground
writers are never starved past the busy cap, and (b) a priority-leased
job coalesces only siblings that serve the same marked barrier
(V6-02.08 — a bounded priority drain takes only marked work).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from verbatim.config import VerbatimConfig
from verbatim.core.identity import scope_key
from verbatim.core.time import now_us
from verbatim.core.types import (
    JobKind,
    Provenance,
    Scope,
    SourceEnvelope,
    SourceKind,
)
from verbatim.ingest import Ingester
from verbatim.jobs import source_jobs as sj
from verbatim.readiness import ingest_receipt_id
from verbatim.storage.store import Store

SCOPE = Scope(profile_id="p", principal_id="alice", conversation_id="c1")
SID = scope_key(SCOPE)


@pytest.fixture()
def cfg() -> VerbatimConfig:
    c = VerbatimConfig()
    return replace(c, capture=replace(c.capture, enabled=True))


@pytest.fixture()
def store(tmp_path):
    s = Store.create(str(tmp_path / "coal.db"))
    yield s
    s.close()


@pytest.fixture()
def ingester(store, cfg) -> Ingester:
    return Ingester(store, cfg)


def _capture(ingester: Ingester, text: str = "coalesced source") -> str:
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
        sj.enqueue_source_jobs(
            conn, ingester.store, receipt_id=ingest_receipt_id(sid, 1)
        )
    return sid


def _job_state(store, job_id: str):
    with store.read() as conn:
        row = conn.execute(
            "SELECT state FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    return row[0] if row else None


def _project_jobs(store) -> list[str]:
    with store.read() as conn:
        return [
            r[0]
            for r in conn.execute(
                "SELECT job_id FROM jobs WHERE kind = 'source_project'"
            ).fetchall()
        ]


def _job_row(store, job_id: str):
    with store.read() as conn:
        return conn.execute(
            "SELECT state, attempts, generation FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()


def test_ordinary_drain_coalesces_siblings(store, ingester):
    """One leased drain commits same-scope ``source_project`` siblings
    inside its own fenced tx — the V8-13.05 batching win."""
    a = _capture(ingester, "alpha source text")
    b = _capture(ingester, "beta source text")
    c = _capture(ingester, "gamma source text")
    jobs = _project_jobs(store)
    assert len(jobs) == 3

    rep = ingester.drain_report(scope=SCOPE, limit=1, owner="t")
    assert rep["processed"] == 1
    assert rep["failed"] == 0, rep["errors"]
    # The leased job plus both staged siblings committed in one pass —
    # siblings ride the shared commit but don't count as processed leases.
    states = {_job_state(store, j) for j in jobs}
    assert states == {"succeeded"}


def test_tx_budget_releases_claimed_tail(store, cfg):
    """``source_coalesce_tx_ms = 0`` bounds the shared commit to the
    first sibling; the claimed tail is released to ``queued`` — never
    lost, never double-applied — and drains on a later pass."""
    cfg0 = replace(
        cfg, jobs=replace(cfg.jobs, source_coalesce_tx_ms=0.0)
    )
    ing = Ingester(store, cfg0)
    _capture(ing, "alpha source text")
    _capture(ing, "beta source text")
    _capture(ing, "gamma source text")
    jobs = _project_jobs(store)
    assert len(jobs) == 3

    rep = ing.drain_report(scope=SCOPE, limit=1, owner="t")
    assert rep["processed"] == 1
    assert rep["failed"] == 0, rep["errors"]
    states = [_job_state(store, j) for j in jobs]
    # The leased job committed; at most one sibling rode the shared
    # commit before the budget tripped — the rest released to queued.
    assert states.count("succeeded") <= 2
    assert states.count("queued") >= 1

    # Released siblings re-lease and commit on later passes — at-least-
    # once delivery with no loss and no stale-generation replay.
    rep2 = ing.drain_report(scope=SCOPE, limit=16, owner="t")
    assert rep2["failed"] == 0, rep2["errors"]
    assert {_job_state(store, j) for j in jobs} == {"succeeded"}


def test_release_siblings_restores_queue_row(store, ingester):
    """``release_siblings`` is owner-conditional and undoes the claim's
    attempt bump while still advancing the generation fence."""
    _capture(ingester, "alpha source text")
    _capture(ingester, "beta source text")
    _capture(ingester, "gamma source text")
    jobs = _project_jobs(store)
    parent_id, claimed_a, claimed_b = jobs

    with store.tx() as conn:
        parent = {
            "job_id": parent_id,
            "kind": "source_project",
            "scope_id": SID,
        }
        claimed = ingester.jobs.claim_siblings(
            conn,
            parent,
            owner="w1",
            limit=2,
            job_ids=[claimed_a, claimed_b],
        )
        assert {c["job_id"] for c in claimed} == {claimed_a, claimed_b}
        # In-tx read — the claim is uncommitted and invisible to a
        # separate store.read() connection.
        in_tx_state = lambda jid: conn.execute(
            "SELECT state FROM jobs WHERE job_id = ?", (jid,)
        ).fetchone()[0]
        assert in_tx_state(claimed_a) == "leased"

        # A different owner's release is a no-op — lease fencing.
        assert (
            ingester.jobs.release_siblings(conn, [claimed_a], owner="w2")
            == 0
        )
        assert in_tx_state(claimed_a) == "leased"

        released = ingester.jobs.release_siblings(
            conn, [claimed_a, claimed_b], owner="w1"
        )
        assert released == 2
        assert in_tx_state(claimed_a) == "queued"

    for jid in (claimed_a, claimed_b):
        state, attempts, gen = _job_row(store, jid)
        assert state == "queued"
        assert attempts == 0  # claim's bump undone — never a real attempt
        assert gen > 1  # fencing token advanced — a stale gen can't replay


def test_priority_drain_keeps_unmarked_siblings_queued(store, ingester):
    """A priority-leased job coalesces only siblings inside the marked
    set — the unmarked source's ``source_project`` stays queued even
    though it is due, same-scope, and same-kind."""
    marked = _capture(ingester, "marked source text")
    unmarked = _capture(ingester, "unmarked source text")
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
    assert rep["failed"] == 0, rep["errors"]
    assert _job_state(store, j_marked) == "succeeded"
    assert _job_state(store, j_unmarked) == "queued"


def test_coalesce_zero_disables_shared_commit(store, cfg):
    """``source_coalesce = 0`` keeps the pre-V8 serial drain — every
    sibling leases and commits on its own transaction."""
    cfg0 = replace(cfg, jobs=replace(cfg.jobs, source_coalesce=0))
    ing = Ingester(store, cfg0)
    _capture(ing, "alpha source text")
    _capture(ing, "beta source text")
    jobs = _project_jobs(store)

    rep = ing.drain_report(scope=SCOPE, limit=1, owner="t")
    assert rep["processed"] == 1
    states = [_job_state(store, j) for j in jobs]
    assert states.count("succeeded") == 1
    assert states.count("queued") == 1
