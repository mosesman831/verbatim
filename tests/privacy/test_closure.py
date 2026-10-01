"""Resumable deletion-closure engine tests (SPEC_V4 §38, F4-07).

F4-07: the old bounded cascade (``_MAX_CASCADE = 512``) could stop
partway through a purge while reporting success. The closure engine
replaces it with durable runs + frontiers: ``begin()`` suppresses the
roots and advances the erasure epoch atomically (V4-38.01), descendants
are excluded by ancestry before enumeration reaches them (V4-38.02),
``step()`` is bounded and never reports terminal success while edges
remain (V4-38.04), and ``resume()`` converges crashed/failed runs without
lifting suppression (V4-38.11).

Regression coverage:

* C55 / V4-05.02 — a purge with 513 descendants finishes truthfully
  across multiple bounded steps (the old code truncated at 512).
* C56 — 50K descendants, crash mid-run, a *fresh* engine resumes from
  durable state only, and a sibling-scope object survives untouched.
* C57 / V4-38.06 — mixed-ancestry derivatives suppress (tombstoned, not
  deleted) until recomputed; their children revalidate.
* V4-05.05 — ``handle_purge_derived`` and the ordinary purge path are
  exercised separately while sharing ``ClosureEngine``.
"""

from __future__ import annotations

import json

import pytest

from verbatim.config import VerbatimConfig
from verbatim.core.types import ErrorCode, JobKind, VerbatimError
from verbatim.core.types_v4 import ClosurePhase
from verbatim.ingest import Ingester
from verbatim.privacy.closure import ClosureEngine
from verbatim.purge import execute_purge, plan_purge
from verbatim.storage.repos_v2 import ErasureRepo
from verbatim.storage.store import Store


# ---------------------------------------------------------------------------
# fixtures + builders
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "v4.db"))
    yield s
    s.close()


@pytest.fixture
def scope_id(store):
    sid = "scope:closure"
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES (?, 'prof', 'owner')",
            (sid,),
        )
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES ('scope:other', 'prof', 'owner')"
        )
    return sid


def qrows(conn, sql, params=()):
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def qrow(conn, sql, params=()):
    rows = qrows(conn, sql, params)
    return rows[0] if rows else None


def make_source(conn, store, sid, src_id, payload=b"sensitive body text"):
    conn.execute(
        "INSERT INTO sources (source_id, origin, source_kind, scope_id,"
        " created_us) VALUES (?, 'test', 'user_message', ?, 1)",
        (src_id, sid),
    )
    conn.execute(
        "INSERT INTO source_revisions"
        "(source_id, revision, payload, payload_hmac, event_us, captured_us,"
        " provenance) VALUES (?, 1, ?, ?, 1, 1, 'direct_user')",
        (src_id, payload, store.hmac(payload)),
    )


def make_observation(conn, sid, oid, rev=1):
    conn.execute(
        "INSERT INTO observations (observation_id, scope_id, revision, text)"
        " VALUES (?, ?, ?, 'observed pattern')",
        (oid, sid, rev),
    )


def edge(conn, child, parent, sid, seq):
    conn.execute(
        "INSERT INTO derivations"
        "(child_kind, child_id, child_revision, parent_kind, parent_id,"
        " parent_revision, producer_kind, producer_id, seq, scope_id)"
        " VALUES (?,?,?,?,?,?, 'job', 'producer-1', ?, ?)",
        (*child, *parent, seq, sid),
    )


def fan_graph(conn, sid, root=("source", "src-1", 1), mids=10, leaves=10):
    """root → mids → leaves: mids*(leaves+1) descendants, all observations."""
    seq = 1
    for m in range(mids):
        mid = f"mid-{m}"
        make_observation(conn, sid, mid)
        edge(conn, ("observation", mid, 1), root, sid, seq)
        seq += 1
        for l in range(leaves):
            leaf = f"leaf-{m}-{l}"
            make_observation(conn, sid, leaf)
            edge(conn, ("observation", leaf, 1), ("observation", mid, 1),
                 sid, seq)
            seq += 1


# ---------------------------------------------------------------------------
# begin — immediate suppression (V4-38.01/02)
# ---------------------------------------------------------------------------


def test_begin_suppresses_roots_and_excludes_descendants(store, scope_id):
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src-1")
        make_observation(conn, scope_id, "obs-a")
        make_observation(conn, scope_id, "obs-b")
        make_observation(conn, scope_id, "obs-x")
        edge(conn, ("observation", "obs-a", 1), ("source", "src-1", 1),
             scope_id, 1)
        edge(conn, ("observation", "obs-b", 1), ("observation", "obs-a", 1),
             scope_id, 2)

    eng = ClosureEngine(store)
    run = eng.begin([("source", "src-1", 1)], scope_id)
    assert run.phase is ClosurePhase.CLEANING
    assert run.erasure_epoch >= 1

    with store.read() as conn:
        # durable run + frontier committed by begin (V4-38.03)
        assert qrow(
            conn,
            "SELECT phase FROM closure_runs WHERE run_id = ?",
            (run.run_id,),
        )["phase"] == "cleaning"
        frontier = qrows(
            conn,
            "SELECT action, state FROM closure_frontier WHERE run_id = ?",
            (run.run_id,),
        )
        assert frontier == [
            {"action": "enumerate", "state": "pending"}
        ]
        # suppression is live before any cleanup step ran (V4-38.01/02)
        assert eng.is_excluded(conn, ("source", "src-1", 1))
        assert eng.is_excluded(conn, ("observation", "obs-a", 1))
        assert eng.is_excluded(conn, ("observation", "obs-b", 1))
        assert not eng.is_excluded(conn, ("observation", "obs-x", 1))
        # the purge registry carries the tombstone intent
        assert qrow(
            conn,
            "SELECT 1 AS x FROM purge_targets pt JOIN purges p"
            " ON p.purge_id = pt.purge_id"
            " WHERE pt.object_id = 'src-1'"
            " AND p.state IN ('suppressed','purging')",
        ) is not None


def test_begin_is_idempotent_and_rejects_different_roots(store, scope_id):
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src-1")
    eng = ClosureEngine(store)
    run = eng.begin(
        [("source", "src-1", 1)], scope_id, run_id="run-fixed"
    )
    again = eng.begin(
        [("source", "src-1", 1)], scope_id, run_id="run-fixed"
    )
    assert again.run_id == run.run_id
    assert again.phase is run.phase
    with pytest.raises(VerbatimError) as exc:
        eng.begin(
            [("source", "src-other", 1)], scope_id, run_id="run-fixed"
        )
    assert exc.value.code == ErrorCode.INTEGRITY


# ---------------------------------------------------------------------------
# bounded steps — never a false completion (V4-38.04, C55/V4-05.02)
# ---------------------------------------------------------------------------


def test_bounded_steps_stay_cleaning_until_frontier_drains(
    store, scope_id
):
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src-1")
        for i in range(8):
            make_observation(conn, scope_id, f"c-{i}")
            edge(conn, ("observation", f"c-{i}", 1),
                 ("source", "src-1", 1), scope_id, i + 1)

    eng = ClosureEngine(store)
    run = eng.begin([("source", "src-1", 1)], scope_id)
    seen_pending = False
    for _ in range(60):
        run = eng.step(run.run_id, budget=2)
        if run.phase is ClosurePhase.COMPLETED:
            break
        # never terminal success while work remains
        assert run.phase is ClosurePhase.CLEANING
        assert run.boundary.get("pending", 0) > 0
        seen_pending = True
    assert seen_pending
    assert run.phase is ClosurePhase.COMPLETED
    assert run.verification["closed"] is True
    with store.read() as conn:
        assert qrows(conn, "SELECT 1 AS x FROM observations") == []
        assert qrows(conn, "SELECT 1 AS x FROM derivations") == []


def test_completed_run_is_a_durable_noop(store, scope_id):
    eng = ClosureEngine(store)
    run = eng.begin([("source", "src-gone", 1)], scope_id)
    run = eng.drain(run.run_id)
    assert run.phase is ClosurePhase.COMPLETED
    # stepping a completed run changes nothing
    run2 = eng.step(run.run_id, budget=4)
    assert run2.phase is ClosurePhase.COMPLETED
    assert eng.verify(run.run_id)["closed"] is True


def test_c55_513_descendants_complete_across_steps(store, scope_id):
    """The 512-cascade regression (V4-05.02): 513 descendants all erase."""
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src-1")
        # root → 513 direct children
        for i in range(513):
            make_observation(conn, scope_id, f"c-{i}")
            edge(conn, ("observation", f"c-{i}", 1),
                 ("source", "src-1", 1), scope_id, i + 1)

    eng = ClosureEngine(store)
    run = eng.begin([("source", "src-1", 1)], scope_id)
    # small budget forces many steps — the run must converge, not truncate
    steps = 0
    while run.phase not in (ClosurePhase.COMPLETED, ClosurePhase.FAILED):
        run = eng.step(run.run_id, budget=64)
        steps += 1
        assert steps < 500, "closure failed to converge"
    assert steps > 1  # multi-step by construction
    assert run.phase is ClosurePhase.COMPLETED
    assert run.verification["closed"] is True
    assert run.verification["counts"]["members"] == 514

    rec = eng.receipt(run.run_id)
    assert rec["phase"] == "completed"
    assert rec["pending_work"] == {"pending": 0, "failed": 0}
    assert len(rec["erased"]) == 514
    assert rec["suppressed"] == []
    with store.read() as conn:
        assert qrows(conn, "SELECT 1 AS x FROM observations") == []
        assert qrows(conn, "SELECT 1 AS x FROM derivations") == []
        ledger = qrows(
            conn,
            "SELECT object_kind FROM erasure_ledger WHERE scope_id = ?",
            (scope_id,),
        )
        assert len(ledger) == 514


# ---------------------------------------------------------------------------
# C56 — scale, crash-resume, sibling-scope isolation
# ---------------------------------------------------------------------------


def test_c56_50k_descendants_crash_resume_sibling_scope(store, scope_id):
    """50K descendants: a mid-run crash leaves durable state a fresh
    engine resumes; the sibling scope is never touched."""
    n_mid, n_leaf = 500, 99  # 500 + 49_500 descendants
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src-1")
        conn.executemany(
            "INSERT INTO observations (observation_id, scope_id, text)"
            " VALUES (?, ?, 'bulk')",
            [(f"m-{m}", scope_id) for m in range(n_mid)]
            + [
                (f"l-{m}-{l}", scope_id)
                for m in range(n_mid)
                for l in range(n_leaf)
            ],
        )
        conn.executemany(
            "INSERT INTO derivations"
            "(child_kind, child_id, child_revision, parent_kind, parent_id,"
            " parent_revision, producer_kind, producer_id, seq, scope_id)"
            " VALUES ('observation', ?, 1, ?, ?, ?, 'job', 'p', ?, ?)",
            [
                (f"m-{m}", "source", "src-1", 1, m + 1, scope_id)
                for m in range(n_mid)
            ]
            + [
                (f"l-{m}-{l}", "observation", f"m-{m}", 1,
                 n_mid + m * n_leaf + l + 1, scope_id)
                for m in range(n_mid)
                for l in range(n_leaf)
            ],
        )
        # sibling scope: must survive untouched
        conn.execute(
            "INSERT INTO sources (source_id, origin, source_kind, scope_id,"
            " created_us) VALUES ('src-sib', 'test', 'user_message',"
            " 'scope:other', 1)"
        )
        conn.execute(
            "INSERT INTO observations"
            " (observation_id, scope_id, text) VALUES ('sib-obs',"
            " 'scope:other', 'keep me')"
        )

    eng = ClosureEngine(store)
    run = eng.begin([("source", "src-1", 1)], scope_id)
    assert run.phase is ClosurePhase.CLEANING

    # "crash" mid-enumeration: partial committed steps, then the engine
    # object is discarded — resume runs on durable state alone.
    for _ in range(3):
        run = eng.step(run.run_id, budget=500)
        assert run.phase is ClosurePhase.CLEANING
    del eng

    resumed = ClosureEngine(store)
    run = resumed.drain(run.run_id)
    assert run.phase is ClosurePhase.COMPLETED
    assert run.verification["closed"] is True
    assert run.verification["counts"]["members"] == 1 + n_mid * (n_leaf + 1)

    with store.read() as conn:
        assert qrow(
            conn,
            "SELECT COUNT(*) AS n FROM observations WHERE scope_id = ?",
            (scope_id,),
        )["n"] == 0
        assert qrow(
            conn, "SELECT COUNT(*) AS n FROM derivations"
        )["n"] == 0
        sib = qrow(
            conn,
            "SELECT text FROM observations WHERE observation_id = 'sib-obs'",
        )
        assert sib == {"text": "keep me"}


# ---------------------------------------------------------------------------
# C57 — mixed ancestry (V4-38.06)
# ---------------------------------------------------------------------------


def test_c57_mixed_ancestry_suppressed_until_recomputed(store, scope_id):
    """obs-m derives from purged obs-a *and* surviving src-2 — it stays
    suppressed (tombstoned, withheld), never erased and never left live."""
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src-1")
        make_source(conn, store, scope_id, "src-2")
        make_observation(conn, scope_id, "obs-a")
        make_observation(conn, scope_id, "obs-b")
        make_observation(conn, scope_id, "obs-m")
        make_observation(conn, scope_id, "obs-g")
        make_observation(conn, scope_id, "obs-f")  # foreign-scope child
        conn.execute(
            "UPDATE observations SET scope_id = 'scope:other'"
            " WHERE observation_id = 'obs-f'"
        )
        edge(conn, ("observation", "obs-a", 1), ("source", "src-1", 1),
             scope_id, 1)
        edge(conn, ("observation", "obs-b", 1), ("observation", "obs-a", 1),
             scope_id, 2)
        edge(conn, ("observation", "obs-m", 1), ("observation", "obs-a", 1),
             scope_id, 3)
        edge(conn, ("observation", "obs-m", 1), ("source", "src-2", 1),
             scope_id, 4)
        edge(conn, ("observation", "obs-g", 1), ("observation", "obs-m", 1),
             scope_id, 5)
        edge(conn, ("observation", "obs-f", 1), ("source", "src-1", 1),
             scope_id, 6)

    eng = ClosureEngine(store)
    run = eng.begin([("source", "src-1", 1)], scope_id)
    run = eng.drain(run.run_id)
    assert run.phase is ClosurePhase.COMPLETED

    rec = eng.receipt(run.run_id)
    erased = {tuple(r) for r in rec["erased"]}
    assert ("observation", "obs-a", 1) in erased
    assert ("observation", "obs-b", 1) in erased
    assert ("source", "src-1", 1) in erased
    # suppressed, not erased — disclosed on the receipt
    assert [tuple(r) for r in rec["suppressed"]] == [
        ("observation", "obs-m", 1)
    ]
    # the grandchild of a suppressed member revalidates
    assert [tuple(r) for r in rec["revalidate"]] == [
        ("observation", "obs-g", 1)
    ]
    # foreign-scope member is reported, never silently claimed
    assert rec["outside_boundary"] == [
        {"ref": ["observation", "obs-f", 1], "reason": "foreign_scope"}
    ]

    with store.read() as conn:
        # suppressed row survives, tombstoned and withheld
        row = qrow(
            conn,
            "SELECT recorded_until FROM observations"
            " WHERE observation_id = 'obs-m'",
        )
        assert row["recorded_until"] is not None
        assert eng.is_excluded(conn, ("observation", "obs-m", 1))
        # revalidation flag + staleness marker on the grandchild
        assert qrow(
            conn,
            "SELECT class FROM freshness"
            " WHERE object_kind = 'observation' AND object_id = 'obs-g'",
        )["class"] == "revalidate_after"
        assert qrow(
            conn,
            "SELECT stale_since_seq FROM observations"
            " WHERE observation_id = 'obs-g'",
        )["stale_since_seq"] is not None
        # the foreign-scope row is untouched
        assert qrow(
            conn,
            "SELECT observation_id, recorded_until FROM observations"
            " WHERE observation_id = 'obs-f'",
        )["recorded_until"] is None
        # edges into purged material are gone; surviving ancestry remains
        survivors = qrows(
            conn,
            "SELECT child_id, parent_id FROM derivations ORDER BY child_id",
        )
        assert [tuple(r.values()) for r in survivors] == [
            ("obs-g", "obs-m"),
            ("obs-m", "src-2"),
        ]


# ---------------------------------------------------------------------------
# crash / failure recovery (V4-38.11)
# ---------------------------------------------------------------------------


def test_failed_finalize_row_resumes_to_completion(
    store, scope_id, monkeypatch
):
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src-1")
        fan_graph(conn, scope_id, mids=5, leaves=4)

    eng = ClosureEngine(store)
    run = eng.begin([("source", "src-1", 1)], scope_id)

    def boom(self, conn, row, item, members, ctx):
        raise RuntimeError("injected finalize crash")

    monkeypatch.setattr(ClosureEngine, "_finalize", boom)
    run = eng.step(run.run_id, budget=100_000)
    assert run.phase is ClosurePhase.FAILED
    assert "finalize" in (run.error or "")
    monkeypatch.undo()

    with store.read() as conn:
        # the failure is durable and suppression is retained (V4-38.11)
        failed = qrows(
            conn,
            "SELECT state, detail_json FROM closure_frontier"
            " WHERE run_id = ? AND state = 'failed'",
            (run.run_id,),
        )
        assert len(failed) == 1
        # the batch's key list survives the failure marker — resume
        # re-runs the SAME unit, not a truncated one
        assert '"keys"' in failed[0]["detail_json"]
        assert eng.is_excluded(conn, ("source", "src-1", 1))
        assert eng.is_excluded(conn, ("observation", "mid-0", 1))
        # a failed run refuses further steps until resumed
        with pytest.raises(VerbatimError) as exc:
            eng.step(run.run_id, conn=conn)
        assert exc.value.code == ErrorCode.INVALID_TRANSITION

    run = eng.resume(run.run_id)
    assert run.phase is ClosurePhase.CLEANING
    run = eng.drain(run.run_id)
    assert run.phase is ClosurePhase.COMPLETED
    with store.read() as conn:
        assert qrows(conn, "SELECT 1 AS x FROM observations") == []
        assert qrows(conn, "SELECT 1 AS x FROM derivations") == []


def test_begin_on_failed_run_resumes_for_redelivery(
    store, scope_id, monkeypatch
):
    """Job redelivery replays ``begin`` on the same run id — a failed run
    is resumed (V4-38.11), never orphaned and never falsely completed."""
    roots = [("source", "src-1", 1)]
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src-1")
        make_observation(conn, scope_id, "obs-a")
        edge(conn, ("observation", "obs-a", 1), ("source", "src-1", 1),
             scope_id, 1)

    eng = ClosureEngine(store)
    run = eng.begin(roots, scope_id, run_id="purge_derived:job-1")

    def boom(self, conn, row, item, members, ctx):
        raise RuntimeError("injected finalize crash")

    monkeypatch.setattr(ClosureEngine, "_finalize", boom)
    run = eng.step(run.run_id, budget=100_000)
    assert run.phase is ClosurePhase.FAILED
    monkeypatch.undo()

    # redelivery: begin on the same run id resumes, does not error
    run = eng.begin(roots, scope_id, run_id="purge_derived:job-1")
    assert run.phase is ClosurePhase.CLEANING
    run = eng.drain(run.run_id)
    assert run.phase is ClosurePhase.COMPLETED
    with store.read() as conn:
        assert qrows(conn, "SELECT 1 AS x FROM observations") == []


def test_step_exception_rolls_back_and_next_step_continues(
    store, scope_id, monkeypatch
):
    """A crash inside a step aborts that transaction; the run stays
    ``cleaning`` and the next step continues — nothing is lost, nothing
    is falsely terminal."""
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src-1")
        fan_graph(conn, scope_id, mids=4, leaves=3)

    eng = ClosureEngine(store)
    run = eng.begin([("source", "src-1", 1)], scope_id)

    def boom(self, conn, row, page, members, ctx):
        raise RuntimeError("injected enumerate crash")

    monkeypatch.setattr(ClosureEngine, "_enumerate_page", boom)
    with pytest.raises(RuntimeError, match="enumerate crash"):
        eng.step(run.run_id, budget=10_000)
    monkeypatch.undo()

    run = eng.status(run.run_id)
    # never a false failure and never a false completion — just resumable
    assert run.phase is ClosurePhase.CLEANING
    run = eng.drain(run.run_id)
    assert run.phase is ClosurePhase.COMPLETED
    with store.read() as conn:
        assert qrows(conn, "SELECT 1 AS x FROM observations") == []


def test_verify_failure_is_honest_and_resumable(
    store, scope_id, monkeypatch
):
    """When physical erasure cannot be verified the run lands
    ``failed_cleanup`` — suppression retained, obligations retryable —
    and ``resume`` re-plans the run to convergence."""
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src-1")
        fan_graph(conn, scope_id, mids=4, leaves=2)

    eng = ClosureEngine(store)
    run = eng.begin([("source", "src-1", 1)], scope_id)

    def noop(self, conn, keys, sid, purge_id, ctx):
        return None

    monkeypatch.setattr(ClosureEngine, "_delete_batch", noop)
    run = eng.drain(run.run_id)
    assert run.phase is ClosurePhase.FAILED
    assert "verification failed" in (run.error or "")
    assert run.verification["closed"] is False
    assert run.verification["orphans"]
    monkeypatch.undo()

    with store.read() as conn:
        # suppression was never lifted to improve availability
        assert eng.is_excluded(conn, ("source", "src-1", 1))
        assert eng.is_excluded(conn, ("observation", "mid-0", 1))
        # rows still exist — the failure is real, not hidden
        assert qrows(conn, "SELECT 1 AS x FROM observations")

    run = eng.resume(run.run_id)
    assert run.phase is ClosurePhase.CLEANING
    run = eng.drain(run.run_id)
    assert run.phase is ClosurePhase.COMPLETED
    assert run.verification["closed"] is True
    with store.read() as conn:
        assert qrows(conn, "SELECT 1 AS x FROM observations") == []


# ---------------------------------------------------------------------------
# job cancellation — erasure lane exempt (F4-07 regression)
# ---------------------------------------------------------------------------


def test_pending_jobs_cancelled_but_erasure_lane_exempt(store, scope_id):
    ing = Ingester(store, VerbatimConfig())
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src-1")
        make_observation(conn, scope_id, "obs-a")
        edge(conn, ("observation", "obs-a", 1), ("source", "src-1", 1),
             scope_id, 1)
        # an ordinary job referencing the purged object → cancelled
        j_embed = ing.jobs.enqueue(
            conn, scope_id, JobKind.EMBED, {"source_id": "src-1"}
        )
        # an erasure-lane job referencing it → exempt (it IS the erasure)
        j_pd = ing.jobs.enqueue(
            conn, scope_id, JobKind.PURGE_DERIVED,
            {"parents": [{"kind": "observation", "id": "obs-a"}]},
        )

    eng = ClosureEngine(store)
    run = eng.begin([("source", "src-1", 1)], scope_id)
    run = eng.drain(run.run_id)
    assert run.phase is ClosurePhase.COMPLETED

    with store.read() as conn:
        assert qrow(
            conn, "SELECT state FROM jobs WHERE job_id = ?", (j_embed,)
        )["state"] == "cancelled"
        assert qrow(
            conn, "SELECT state FROM jobs WHERE job_id = ?", (j_pd,)
        )["state"] == "queued"


# ---------------------------------------------------------------------------
# V4-05.05 — both entry points, separately, on the same engine
# ---------------------------------------------------------------------------


def _drain_job(ing: Ingester, kind, refs, scope_id, **enqueue_kw):
    with ing.store.tx() as conn:
        jid = ing.jobs.enqueue(conn, scope_id, kind, refs, **enqueue_kw)
    n = ing.run_pending(limit=8, owner="test-worker")
    assert n >= 1
    with ing.store.read() as conn:
        job = qrow(conn, "SELECT * FROM jobs WHERE job_id = ?", (jid,))
    return jid, job


def test_standalone_purge_derived_handler(store, scope_id):
    """``handle_purge_derived`` drives a ClosureEngine run to completion
    inside the job lifecycle — no truncation, durable run id bound to the
    job."""
    ing = Ingester(store, VerbatimConfig())
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src-1")
        make_observation(conn, scope_id, "obs-1")
        make_observation(conn, scope_id, "obs-2")
        edge(conn, ("observation", "obs-1", 1), ("source", "src-1", 1),
             scope_id, 1)
        edge(conn, ("observation", "obs-2", 1), ("observation", "obs-1", 1),
             scope_id, 2)

    jid, job = _drain_job(
        ing, JobKind.PURGE_DERIVED,
        {"parents": [{"kind": "source", "id": "src-1", "revision": 1}]},
        scope_id,
        operation_key="op-pd-1",
    )
    assert job["state"] == "succeeded"

    with store.read() as conn:
        assert qrows(conn, "SELECT 1 AS x FROM observations") == []
        assert qrows(conn, "SELECT 1 AS x FROM derivations") == []
        run_row = qrow(
            conn,
            "SELECT phase FROM closure_runs WHERE run_id = ?",
            (f"purge_derived:{jid}",),
        )
        assert run_row["phase"] == "completed"
        # the operation receipt reports the closure honestly
        op = qrow(
            conn,
            "SELECT receipt_json FROM operations"
            " WHERE scope_id = ? AND effect_kind = 'purge_derived'",
            (scope_id,),
        )
        assert op is not None
        rec = json.loads(op["receipt_json"])
        assert rec["truncated"] is False
        assert rec["closure_run_id"] == f"purge_derived:{jid}"
        assert rec["pending_work"] == {"pending": 0, "failed": 0}
        deleted = {
            (c["kind"], c["id"], c["revision"])
            for c in rec["children_deleted"]
        }
        assert deleted == {
            ("observation", "obs-1", 1),
            ("observation", "obs-2", 1),
        }


def test_ordinary_purge_runs_the_same_engine(store, scope_id):
    """The ordinary purge path runs closure through ClosureEngine with
    run id ``purge:{purge_id}`` — same engine, same durable lifecycle."""
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src-1")
        make_observation(conn, scope_id, "obs-1")
        edge(conn, ("observation", "obs-1", 1), ("source", "src-1", 1),
             scope_id, 1)

    plan = plan_purge(store, scope_id, [("source", "src-1")], actor="alice")
    result = execute_purge(store, plan["purge_id"])
    assert result["state"] == "completed"

    with store.read() as conn:
        assert qrows(conn, "SELECT 1 AS x FROM observations") == []
        run_row = qrow(
            conn,
            "SELECT phase FROM closure_runs WHERE run_id = ?",
            (f"purge:{plan['purge_id']}",),
        )
        assert run_row is not None
        assert run_row["phase"] == "completed"
        assert ErasureRepo(store).is_erased(conn, scope_id, "source", "src-1")


def test_both_entry_points_share_one_engine(store, scope_id, monkeypatch):
    """V4-05.05: prove both paths dispatch through ``ClosureEngine``."""
    calls: list[str] = []
    orig_begin = ClosureEngine.begin
    orig_drain = ClosureEngine.drain

    def spy_begin(self, roots, scope, **kw):
        calls.append("begin")
        return orig_begin(self, roots, scope, **kw)

    def spy_drain(self, run_id, **kw):
        calls.append("drain")
        return orig_drain(self, run_id, **kw)

    monkeypatch.setattr(ClosureEngine, "begin", spy_begin)
    monkeypatch.setattr(ClosureEngine, "drain", spy_drain)

    # path 1: standalone handler job
    ing = Ingester(store, VerbatimConfig())
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src-1")
        make_observation(conn, scope_id, "obs-1")
        edge(conn, ("observation", "obs-1", 1), ("source", "src-1", 1),
             scope_id, 1)
    _jid, job = _drain_job(
        ing, JobKind.PURGE_DERIVED,
        {"parents": [{"kind": "source", "id": "src-1", "revision": 1}]},
        scope_id,
    )
    assert job["state"] == "succeeded"

    # path 2: ordinary purge
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src-2")
        make_observation(conn, scope_id, "obs-2")
        edge(conn, ("observation", "obs-2", 1), ("source", "src-2", 1),
             scope_id, 10)
    plan = plan_purge(store, scope_id, [("source", "src-2")], actor="alice")
    execute_purge(store, plan["purge_id"])

    assert calls == ["begin", "drain", "begin", "drain"]


# ---------------------------------------------------------------------------
# honest reporting — outside boundary, propagations, receipts (V4-38.05/08)
# ---------------------------------------------------------------------------


def test_unresolvable_child_kind_reported_not_silently_skipped(
    store, scope_id
):
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src-9")
        edge(conn, ("mystery_kind", "m-1", 1), ("source", "src-9", 1),
             scope_id, 1)

    eng = ClosureEngine(store)
    run = eng.begin([("source", "src-9", 1)], scope_id)
    run = eng.drain(run.run_id)
    assert run.phase is ClosurePhase.COMPLETED
    rec = eng.receipt(run.run_id)
    assert rec["outside_boundary"] == [
        {"ref": ["mystery_kind", "m-1", 1], "reason": "unresolvable_kind"}
    ]
    with store.read() as conn:
        # the edge into purged material is still removed
        assert qrows(conn, "SELECT 1 AS x FROM derivations") == []


def test_vault_entry_descendant_erased_via_closure(store, scope_id):
    """A ``vault_entry`` child resolves through the engine — tombstoned
    via ``erase_entry`` semantics (``erased_event``), vault_refs scrubbed,
    and an opaque ledger digest written (V4-38.05 — same coverage the old
    ``_delete_child_object`` special case gave)."""
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src-1")
        conn.execute(
            "INSERT INTO vault_entries"
            "(entry_id, scope_id, revision, sensitivity, placeholder,"
            " algorithm, key_version, wrap_key_version, nonce, ciphertext,"
            " wrapped_key, aad_digest, detection)"
            " VALUES ('ve-1', ?, 1, 's2', '[VAULT:S2:1]',"
            " 'AES-256-GCM', 1, 1, X'00', X'01', X'02', X'03', 'detected')",
            (scope_id,),
        )
        conn.execute(
            "INSERT INTO vault_refs"
            "(placeholder, entry_id, scope_id, view_id, start_byte,"
            " end_byte) VALUES"
            " ('[VAULT:S2:1]', 've-1', ?, 'view-1', 0, 4)",
            (scope_id,),
        )
        edge(conn, ("vault_entry", "ve-1", 1), ("source", "src-1", 1),
             scope_id, 1)

    eng = ClosureEngine(store)
    run = eng.drain(eng.begin([("source", "src-1", 1)], scope_id).run_id)
    assert run.phase is ClosurePhase.COMPLETED
    rec = eng.receipt(run.run_id)
    assert ["vault_entry", "ve-1", 1] in rec["erased"]

    with store.read() as conn:
        row = qrow(
            conn,
            "SELECT erased_event FROM vault_entries WHERE entry_id = 've-1'",
        )
        assert row["erased_event"] is not None
        assert qrows(
            conn, "SELECT 1 AS x FROM vault_refs WHERE entry_id = 've-1'"
        ) == []
        assert ErasureRepo(store).is_erased(
            conn, scope_id, "vault_entry", "ve-1"
        )


def test_propagated_copies_revoked_and_disclosed(store, scope_id):
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src-1")
        make_observation(conn, scope_id, "obs-a")
        edge(conn, ("observation", "obs-a", 1), ("source", "src-1", 1),
             scope_id, 1)
        conn.execute(
            "INSERT INTO propagations"
            "(propagation_id, scope_id, object_kind, object_id, revision,"
            " recipient_id, verbs_json, purpose, epoch, created_us)"
            " VALUES ('prop-1', ?, 'observation', 'obs-a', 1, 'recip-9',"
            " '[]', 'care', 1, 1)",
            (scope_id,),
        )

    eng = ClosureEngine(store)
    run = eng.drain(eng.begin([("source", "src-1", 1)], scope_id).run_id)
    assert run.phase is ClosurePhase.COMPLETED

    rec = eng.receipt(run.run_id)
    assert rec["external_copies"] == [
        {
            "ref": ["observation", "obs-a", 1],
            "reason": "propagated_copy",
            "detail": "recipient recip-9 holds a copy",
            "propagation_id": "prop-1",
            "recipient_id": "recip-9",
        }
    ]
    with store.read() as conn:
        assert qrow(
            conn,
            "SELECT revoked_seq FROM propagations"
            " WHERE propagation_id = 'prop-1'",
        )["revoked_seq"] is not None


def test_receipt_reports_pending_work_mid_run(store, scope_id):
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src-1")
        fan_graph(conn, scope_id, mids=4, leaves=3)

    eng = ClosureEngine(store)
    run = eng.begin([("source", "src-1", 1)], scope_id)
    rec = eng.receipt(run.run_id)
    assert rec["phase"] == "cleaning"
    assert rec["pending_work"]["pending"] >= 1
    assert rec["verification"] == {}
    assert rec["cryptographic_erasure"].startswith("unproven")
    assert "backup" in rec["backup_obligation"]

    run = eng.drain(run.run_id)
    rec = eng.receipt(run.run_id)
    assert rec["phase"] == "completed"
    assert rec["pending_work"] == {"pending": 0, "failed": 0}
    assert rec["verification"]["closed"] is True
    assert rec["members"] == 17
