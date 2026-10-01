"""F4-09: fenced admission — a cancelled, stale, or reclaimed worker commits
no partial domain effects.

Covers V4-09.02/09.04 (lease + dependency re-verification inside the same
transaction that writes effects), V4-42.01 (generation fencing tokens),
V4-42.03 (cancellation fences publication AND queued follow-up work), and
V4-42.08 (expired leases reclaimed after restart while stale workers stay
fenced).

Scenarios:

- cancel between dequeue and commit → LEASE_LOST, zero domain effects
  (no claim, revision, evidence link, review, edge, index row, receipt,
  embed follow-up job, or audit event).
- cancel landing after the cheap pre-fence but before the commit tx → the
  in-transaction ``assert_lease`` still fences it.
- cancel between the admission commit and the relation follow-up → the
  claim commits (its commit was legitimately fenced) but the follow-up's
  own fenced tx rolls back — no edges, no pair event, no completion.
- lease expiry → reclaim → replacement worker commits → the stale worker's
  generation can never commit (C23).
- crash between admission commit and job completion → redelivery replays
  the operation receipt and heals the follow-on phase instead of minting
  a duplicate claim.
- an erasure tombstone or quarantine hold landing between the read-time
  pre-check and the commit still vetoes the admission inside the tx.

The v2-schema ``V2Store`` shim exercises the full operation-receipt +
generation machinery; the real ``Store`` (schema v4) covers the quarantine
and derivations surfaces the shim lacks.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Optional

import pytest

from tests.jobs.test_v2_jobs import V2Store
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
from verbatim.purge import suppress
from verbatim.storage.store import Store
import verbatim.ingest as ingest_mod


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _cfg(require_review: bool = True) -> VerbatimConfig:
    cfg = VerbatimConfig()
    cfg = replace(cfg, capture=replace(cfg.capture, enabled=True))
    if not require_review:
        cfg = replace(
            cfg, admission=replace(cfg.admission, require_review=False)
        )
    return cfg


def _env(scope: Scope, text: str = "My editor is neovim.") -> SourceEnvelope:
    return SourceEnvelope(
        origin="test:f409",
        source_kind=SourceKind.USER_MESSAGE,
        scope=scope,
        speaker_id=scope.principal_id,
        payload=text.encode("utf-8"),
        event_us=now_us(),
        captured_us=now_us(),
        provenance=Provenance.DIRECT_USER,
    )


SCOPE = Scope(profile_id="p", principal_id="alice", conversation_id="c1")


def _harvested(ing: Ingester, text: str = "My editor is neovim.") -> str:
    """Ingest one source and drain only its harvest; returns source_id."""
    receipt = ing.ingest(_env(SCOPE, text))
    (sid,) = receipt.accepted
    ing.run_pending(scope=SCOPE, kinds=[JobKind.HARVEST])
    return sid


def _lease_admit(
    ing: Ingester, owner: str, **kw: Any
) -> dict[str, Any]:
    leased = ing.jobs.lease(
        scope_key(SCOPE), [JobKind.ADMIT], owner=owner, limit=1, **kw
    )
    assert leased, "expected a leased admit job"
    return leased[0]


def _counts(store: Any) -> dict[str, int]:
    tables = (
        "claims",
        "claim_revisions",
        "claim_evidence",
        "reviews",
        "edges",
        "fts_rows",
        "operations",
    )
    out: dict[str, int] = {}
    with store.read() as conn:
        for t in tables:
            try:
                out[t] = conn.execute(
                    f"SELECT COUNT(*) FROM {t}"
                ).fetchone()[0]
            except Exception:
                out[t] = -1  # table absent on this schema
        out["embed_jobs"] = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE kind = 'embed'"
        ).fetchone()[0]
        out["events"] = conn.execute(
            "SELECT COUNT(*) FROM events"
            " WHERE kind IN ('claim_proposed','admitted','pair_comparison')"
        ).fetchone()[0]
    return out


def _job_row(store: Any, job_id: str) -> dict[str, Any]:
    with store.read() as conn:
        cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, cur.fetchone()))


class _FakeEncoder:
    """Minimal encoder: makes ``_enqueue_embed`` create real EMBED jobs so
    the "no follow-up job" assertions are load-bearing."""

    @property
    def encoder_id(self) -> str:
        return "fake:f409:v1"

    def available(self) -> bool:
        return True

    def encode(self, texts: list[str]) -> list[bytes]:
        return [b"\x00" * 16 for _ in texts]

    def manifest(self) -> dict[str, Any]:
        return {"preprocessing_version": "p1"}


@pytest.fixture()
def v2store() -> V2Store:
    return V2Store()


@pytest.fixture()
def real_store(tmp_path):
    s = Store.create(str(tmp_path / "f409.db"))
    yield s
    s.close()


# --------------------------------------------------------------------------
# cancellation races
# --------------------------------------------------------------------------


def test_cancel_between_lease_and_commit_leaves_no_effects(v2store):
    """C21: cancel lands between dequeue and apply — the worker commits
    nothing: no claim rows, no index publication, no receipt, no embed
    follow-up job, no audit event (V4-09.04/42.03).

    ``require_review=False`` + a configured encoder make the follow-up
    assertions load-bearing: an unfenced commit would have produced an
    ACTIVE claim, an FTS row, and a queued EMBED job."""
    ing = Ingester(v2store, _cfg(require_review=False), encoder=_FakeEncoder())
    _harvested(ing)
    job = _lease_admit(ing, "w1")

    assert ing.jobs.cancel(job["job_id"]) is True
    with pytest.raises(VerbatimError) as ei:
        ing._do_admit(job, "w1")
    assert ei.value.code is ErrorCode.LEASE_LOST

    got = _counts(v2store)
    assert got["claims"] == 0
    assert got["claim_revisions"] == 0
    assert got["claim_evidence"] == 0
    assert got["reviews"] == 0
    assert got["edges"] == 0
    assert got["fts_rows"] == 0
    # The only receipt is the harvest job's; nothing for admit.
    with v2store.read() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM operations WHERE operation_key LIKE 'admit:%'"
        ).fetchone()[0] == 0
    assert got["embed_jobs"] == 0
    assert got["events"] == 0
    assert _job_row(v2store, job["job_id"])["state"] == "cancelled"


def test_embed_followup_enqueued_atomically_when_lease_live(v2store):
    """Control: with a live lease the ACTIVE claim, its FTS row, and the
    EMBED job all land in one commit — so the cancelled-path assertions
    above are not vacuous."""
    ing = Ingester(v2store, _cfg(require_review=False), encoder=_FakeEncoder())
    _harvested(ing)
    ing._do_admit(_lease_admit(ing, "w1"), "w1")
    got = _counts(v2store)
    assert got["claims"] == 1
    assert got["fts_rows"] == 1
    assert got["embed_jobs"] == 1


def test_cancel_after_pre_fence_is_fenced_inside_commit_tx(
    v2store, monkeypatch
):
    """The early ``_pre_fence`` is an optimization only — a cancel that lands
    after it (mid-evaluation, before the commit tx) is still caught by the
    in-transaction ``assert_lease`` (V4-09.02)."""
    ing = Ingester(v2store, _cfg())
    _harvested(ing)
    job = _lease_admit(ing, "w1")

    real_propose = ingest_mod.propose

    def cancelling_propose(*a: Any, **kw: Any) -> Any:
        # The cancel commits after _pre_fence passed but before TX1 opens.
        assert ing.jobs.cancel(job["job_id"]) is True
        return real_propose(*a, **kw)

    monkeypatch.setattr(ingest_mod, "propose", cancelling_propose)
    with pytest.raises(VerbatimError) as ei:
        ing._do_admit(job, "w1")
    assert ei.value.code is ErrorCode.LEASE_LOST

    got = _counts(v2store)
    assert got["claims"] == 0
    assert got["claim_revisions"] == 0
    assert got["claim_evidence"] == 0
    assert got["reviews"] == 0
    assert got["events"] == 0
    assert got["embed_jobs"] == 0


def test_cancel_during_followup_rolls_back_relation_effects(real_store):
    """Cancel between the admission commit and the relation pass: the claim
    itself committed under a live lease (legitimate), but the follow-up's
    fenced tx rolls back — no edge, no supersession review, no pair event —
    and the terminal completion cannot mark a cancelled job succeeded
    (V4-42.03)."""
    ing = Ingester(real_store, _cfg(require_review=False))
    # First claim admitted and completed normally so the second claim has a
    # same-predicate ACTIVE candidate — relate would mint a conflicts_with
    # edge + supersede review for this pair.
    _harvested(ing, "My editor is neovim.")
    ing._do_admit(_lease_admit(ing, "w1"), "w1")
    with real_store.read() as conn:
        n_pairs = conn.execute(
            "SELECT COUNT(*) FROM events WHERE kind = 'pair_comparison'"
        ).fetchone()[0]
    assert n_pairs == 1

    _harvested(ing, "My editor is emacs.")
    job = _lease_admit(ing, "w2")

    real_relate = ingest_mod.relate
    cancelled = {"done": False}

    def cancelling_relate(store: Any, claim_id: str, **kw: Any) -> Any:
        # Cancel lands after the admission commit, inside the follow-up.
        assert ing.jobs.cancel(job["job_id"]) is True
        cancelled["done"] = True
        return real_relate(store, claim_id, **kw)

    # Patch the name _do_admit resolves; _LeaseFencedStore.tx() then raises
    # inside relate's commit tx.
    import unittest.mock as mock

    with mock.patch.object(ingest_mod, "relate", cancelling_relate):
        with pytest.raises(VerbatimError) as ei:
            ing._do_admit(job, "w2")
    assert ei.value.code is ErrorCode.LEASE_LOST
    assert cancelled["done"] is True

    with real_store.read() as conn:
        # The admission committed while the lease was live: claim2 exists…
        n_claims = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        assert n_claims == 2
        # …but every follow-up artifact rolled back.
        assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM reviews"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM events WHERE kind = 'pair_comparison'"
        ).fetchone()[0] == n_pairs  # still just the first claim's
    # The cancelled job can never be reported succeeded by the stale worker.
    assert _job_row(real_store, job["job_id"])["state"] == "cancelled"


def test_conflicting_pair_produces_edge_when_lease_live(real_store):
    """Control for the test above: with a live lease the same pair yields a
    conflicts_with edge + review, proving the rollback assertions are
    meaningful."""
    ing = Ingester(real_store, _cfg(require_review=False))
    _harvested(ing, "My editor is neovim.")
    ing._do_admit(_lease_admit(ing, "w1"), "w1")
    _harvested(ing, "My editor is emacs.")
    ing._do_admit(_lease_admit(ing, "w2"), "w2")
    with real_store.read() as conn:
        kinds = [
            r[0]
            for r in conn.execute("SELECT edge_type FROM edges")
        ]
        assert "conflicts_with" in kinds
        assert conn.execute(
            "SELECT COUNT(*) FROM reviews"
        ).fetchone()[0] >= 1
        assert conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE kind='admit' AND state='succeeded'"
        ).fetchone()[0] == 2


# --------------------------------------------------------------------------
# stale generation / reclaim (C23)
# --------------------------------------------------------------------------


def test_stale_worker_after_reclaim_cannot_commit(v2store):
    """Lease expires → ``reclaim_expired`` bumps the generation → a
    replacement worker leases and commits → the original worker's fenced
    commit raises LEASE_LOST and writes nothing (V4-42.08)."""
    ing = Ingester(v2store, _cfg())
    _harvested(ing)
    old = _lease_admit(ing, "old-worker", lease_s=0.05)
    assert old["generation"] >= 1

    time.sleep(0.06)
    assert ing.jobs.reclaim_expired() == 1
    new = _lease_admit(ing, "new-worker")
    assert new["job_id"] == old["job_id"]
    assert new["generation"] > old["generation"]

    ing._do_admit(new, "new-worker")
    assert _job_row(v2store, new["job_id"])["state"] == "succeeded"

    with pytest.raises(VerbatimError) as ei:
        ing._do_admit(old, "old-worker")
    assert ei.value.code is ErrorCode.LEASE_LOST

    got = _counts(v2store)
    assert got["claims"] == 1
    assert got["claim_revisions"] == 1
    assert got["claim_evidence"] == 1
    with v2store.read() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM operations WHERE operation_key LIKE 'admit:%'"
        ).fetchone()[0] == 1


def test_reclaim_makes_cancelled_worker_unable_to_complete(v2store):
    """A worker fenced out mid-run cannot even fail/complete its job — the
    queue-level transitions are generation-checked too."""
    ing = Ingester(v2store, _cfg())
    _harvested(ing)
    job = _lease_admit(ing, "w1", lease_s=0.05)
    time.sleep(0.06)
    assert ing.jobs.reclaim_expired() == 1

    with pytest.raises(VerbatimError) as ei:
        ing._do_admit(job, "w1")
    assert ei.value.code is ErrorCode.LEASE_LOST
    # fail() under the stale generation is a no-op — the job is retryable
    # work for a NEW lease, not failed.
    assert ing.jobs.fail(
        job["job_id"], "w1", job["generation"], "INTERNAL", True
    ) is None
    assert _job_row(v2store, job["job_id"])["state"] == "retry_wait"


# --------------------------------------------------------------------------
# redelivery / receipt replay
# --------------------------------------------------------------------------


def test_redelivery_after_crash_replays_receipt_and_heals(v2store):
    """Crash between the admission commit and job completion: the receipt
    committed, so redelivery replays it (no duplicate claim/revision/event)
    and still runs the fenced follow-on + completion."""
    ing = Ingester(v2store, _cfg())
    _harvested(ing)
    job = _lease_admit(ing, "w1")

    real_relate = ingest_mod.relate
    crashed = {"once": False}

    def crashing_relate(store: Any, claim_id: str, **kw: Any) -> Any:
        if not crashed["once"]:
            crashed["once"] = True
            raise RuntimeError("worker died mid-followup")
        return real_relate(store, claim_id, **kw)

    import unittest.mock as mock

    with mock.patch.object(ingest_mod, "relate", crashing_relate):
        with pytest.raises(RuntimeError):
            ing._do_admit(job, "w1")
        # The admission commit + receipt survived the crash; the job is
        # still leased (no completion ran).
        got = _counts(v2store)
        assert got["claims"] == 1
        assert _job_row(v2store, job["job_id"])["state"] == "leased"
        # Simulate the worker framework: retryable fail → re-lease.
        assert ing.jobs.fail(
            job["job_id"], "w1", job["generation"], "INTERNAL", True
        ) is not None
        future = now_us() + 10_000_000
        job2 = _lease_admit(ing, "w2", now_us=future, lease_s=600)
        assert job2["generation"] > job["generation"]
        ing._do_admit(job2, "w2")

    got = _counts(v2store)
    assert got["claims"] == 1
    assert got["claim_revisions"] == 1
    # one claim_proposed + one admitted + pair events — none duplicated
    with v2store.read() as conn:
        kinds = [
            r[0]
            for r in conn.execute(
                "SELECT kind FROM events ORDER BY event_seq"
            )
        ]
        assert kinds.count("claim_proposed") == 1
        assert kinds.count("admitted") == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM operations WHERE operation_key LIKE 'admit:%'"
        ).fetchone()[0] == 1
    assert _job_row(v2store, job["job_id"])["state"] == "succeeded"


# --------------------------------------------------------------------------
# in-transaction dependency re-verification
# --------------------------------------------------------------------------


def test_suppression_landing_mid_evaluation_fences_commit(v2store):
    """An erasure tombstone committed between the read-time pre-check and
    the effects tx still vetoes the admission — the in-tx
    ``_source_suppressed`` re-check catches what the snapshot could not
    (V4-42.03)."""
    ing = Ingester(v2store, _cfg())
    sid = _harvested(ing)
    job = _lease_admit(ing, "w1")

    real_eval = ingest_mod._admit_evaluate

    def suppressing_eval(*a: Any, **kw: Any) -> Any:
        plan = real_eval(*a, **kw)
        suppress(v2store, SCOPE, [("source", sid)], actor="test")
        return plan

    import unittest.mock as mock

    with mock.patch.object(ingest_mod, "_admit_evaluate", suppressing_eval):
        with pytest.raises(VerbatimError) as ei:
            ing._do_admit(job, "w1")
    assert ei.value.code is ErrorCode.EVIDENCE_UNAVAILABLE

    got = _counts(v2store)
    assert got["claims"] == 0
    assert got["claim_revisions"] == 0
    assert got["events"] == 0


def test_quarantine_hold_landing_mid_evaluation_fences_commit(real_store):
    """Same race on the quarantine surface (schema-v4 store): a hold opened
    after the pre-check still wins inside the commit tx (V3-14.10/V4-09.02).
    """
    from verbatim.security.quarantine import open_quarantine

    ing = Ingester(real_store, _cfg())
    sid = _harvested(ing)
    job = _lease_admit(ing, "w1")
    with real_store.read() as conn:
        scope_id = conn.execute(
            "SELECT scope_id FROM sources WHERE source_id = ?", (sid,)
        ).fetchone()[0]

    real_eval = ingest_mod._admit_evaluate

    def quarantining_eval(*a: Any, **kw: Any) -> Any:
        plan = real_eval(*a, **kw)
        with real_store.tx() as conn:
            open_quarantine(
                conn,
                ("source", sid, 1),
                ["attack_risk:blocked"],
                [],
                scope_id=scope_id,
            )
        return plan

    import unittest.mock as mock

    with mock.patch.object(ingest_mod, "_admit_evaluate", quarantining_eval):
        with pytest.raises(VerbatimError) as ei:
            ing._do_admit(job, "w1")
    assert ei.value.code is ErrorCode.QUARANTINED

    got = _counts(real_store)
    assert got["claims"] == 0
    assert got["claim_revisions"] == 0
    assert got["events"] == 0


# --------------------------------------------------------------------------
# public admit() compatibility
# --------------------------------------------------------------------------


def test_direct_admit_call_still_commits_atomically(v2store):
    """``admit(store, proposal, envelope)`` keeps its direct-call contract
    for ``api_ingest.remember`` and review callers: evaluation outside the
    tx, effects inside one self-managed commit."""
    from verbatim.core.claims import propose
    from verbatim.core.policy import admit
    from verbatim.core.types import SpanRef
    from verbatim.ingest import envelope_for
    from verbatim.storage.repos import SourcesRepo, SpansRepo

    env = _env(SCOPE)
    with v2store.tx() as conn:
        sid, _ = SourcesRepo(v2store).insert(env, conn=conn)
        SpansRepo(v2store).insert(
            "sp_direct", sid, 1, 0, len(env.payload),
            "test-direct", conn=conn,
        )
    envelope = envelope_for(v2store, sid, 1)
    text = env.payload.decode("utf-8")
    span = SpanRef(
        span_id="sp_direct", source_id=sid, revision=1,
        start_byte=0, end_byte=len(env.payload),
    )
    proposal = propose(text, span, envelope, envelope.event_us)
    outcome = admit(v2store, proposal, envelope, ctx=Ingester(v2store, _cfg()).policy)
    assert outcome.claim_id
    got = _counts(v2store)
    assert got["claims"] == 1
    assert got["claim_revisions"] == 1
