"""Durable readiness subsystem tests (SPEC_V4 §14, V4-14.01–14.11).

Covers: atomic obligation recording at capture, per-receipt DAG state,
dependency enforcement, failure cascade + visibility, restart
persistence, ``wait_ready`` honesty (pending/failed/deadline), drain
count truthfulness, mid-drain priority reconsideration, session-end
failure visibility, and the v2/v3 capture-path integrations.

Conformance anchors: C26 (unrelated later events never satisfy a
receipt), C27 (drain reports follow-up work honestly), C83 (session-end
drain failure stays visible), C88 (global sequence numbers are not
readiness proof).

Fixtures use real ``Store.create`` databases — the conftest TestStore
shim is DDL_V1-only and carries no v4 readiness tables.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

import pytest

from verbatim.api import Engine
from verbatim.api_v3 import VerbatimV3
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
from verbatim.core.types_v3 import (
    EnvelopeKind,
    Perspective,
    SourceEnvelopeV3,
)
from verbatim.core.types_v4 import CapabilityName, ReadinessState
from verbatim.embeddings.hashing import HashingEncoder
from verbatim.evidence import ingest_envelope
from verbatim.host import LocalHost
from verbatim.ingest import Ingester
from verbatim.provider import VerbatimMemoryProvider
from verbatim.readiness import (
    ReadinessEngine,
    ingest_receipt_id,
)
from verbatim.storage.store import Store


# ---------------------------------------------------------------------
# fixtures + helpers
# ---------------------------------------------------------------------

SCOPE = Scope(profile_id="p", principal_id="alice", conversation_id="c1")
SCOPE2 = Scope(profile_id="p", principal_id="bob", conversation_id="c2")
V3_SCOPE = "scope:readiness"
AGENT = "agent-readiness"


@pytest.fixture
def cfg() -> VerbatimConfig:
    c = VerbatimConfig()
    c = replace(c, capture=replace(c.capture, enabled=True))
    # Claims land ACTIVE on admission — no review round-trip needed for
    # pipeline-settle assertions.
    c = replace(c, admission=replace(c.admission, require_review=False))
    return c


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "v4.db"))
    yield s
    s.close()


def _env(
    text: str,
    scope: Scope = SCOPE,
    ext: str | None = None,
    kind: SourceKind = SourceKind.USER_MESSAGE,
) -> SourceEnvelope:
    return SourceEnvelope(
        origin="test",
        source_kind=kind,
        scope=scope,
        speaker_id=scope.principal_id,
        payload=text.encode("utf-8"),
        event_us=now_us(),
        captured_us=now_us(),
        provenance=Provenance.DIRECT_USER,
        external_id=ext,
    )


def _states(snap: dict) -> dict[str, str]:
    return {k: v["state"] for k, v in snap["states"].items()}


def _obligation_rows(store: Store, receipt_id: str) -> dict[str, dict]:
    with store.read() as conn:
        rows = conn.execute(
            "SELECT * FROM readiness_obligations WHERE receipt_id = ?",
            (receipt_id,),
        ).fetchall()
    cols = [
        "obligation_id", "receipt_id", "scope_id", "capability",
        "depends_on_json", "state", "error", "created_us", "updated_us",
    ]
    return {r[3]: dict(zip(cols, r)) for r in rows}


def _events(store: Store, kind: str):
    with store.read() as conn:
        return conn.execute(
            "SELECT kind, payload_json FROM events WHERE kind = ?", (kind,)
        ).fetchall()


class _StubCapture:
    """Minimal stand-in for the provider's adapter: calls the supplied
    drain seam, records failures the same shape ``HermesV3Adapter``
    reports them."""

    def __init__(self) -> None:
        self.last_hook_error = None

    def on_session_end(self, messages, drain=None):
        summary: dict = {"ended": True}
        try:
            summary["drained"] = (drain or (lambda **k: 0))(limit=64)
        except Exception as exc:  # noqa: BLE001 — mirrors adapter catch
            summary["drain_error"] = str(exc)
            rec = {"error": str(exc)}
            code = getattr(exc, "code", None)
            if code is not None:
                rec["code"] = getattr(code, "value", code)
            self.last_hook_error = rec
        return summary


def _provider_on(engine: Engine) -> VerbatimMemoryProvider:
    prov = VerbatimMemoryProvider()
    prov._engine = engine
    prov._host = LocalHost(profile_id="p", principal_id="alice")
    prov._capture = _StubCapture()
    return prov


# ---------------------------------------------------------------------
# V4-14.01/02 — obligations record atomically with capture
# ---------------------------------------------------------------------


def test_capture_records_obligation_dag_atomically(store, cfg):
    ing = Ingester(store, cfg)
    r = ing.ingest(_env("My editor is neovim."))
    rid = ingest_receipt_id(r.accepted[0], 1)

    snap = ReadinessEngine(store).receipt_state(rid)
    st = _states(snap)
    assert st["accepted"] == "succeeded"
    for cap in (
        "screened", "lexical_ready", "semantic_ready",
        "derived_ready", "failed",
    ):
        assert st[cap] == "pending", cap
    assert snap["ready"] is False and snap["complete"] is False

    rows = _obligation_rows(store, rid)
    assert len(rows) == 6
    # Sibling fan-out: semantic + derived both gate on lexical only.
    assert json.loads(rows["semantic_ready"]["depends_on_json"]) == [
        f"ro:{rid}:lexical_ready"
    ]
    assert json.loads(rows["derived_ready"]["depends_on_json"]) == [
        f"ro:{rid}:lexical_ready"
    ]
    assert json.loads(rows["lexical_ready"]["depends_on_json"]) == [
        f"ro:{rid}:screened"
    ]
    assert json.loads(rows["failed"]["depends_on_json"]) == []


def test_duplicate_capture_converges(store, cfg):
    """V4-14.08: an identical re-capture dedups — no duplicated
    obligation rows, same receipt identity."""
    ing = Ingester(store, cfg)
    env = _env("identical payload", ext="dup-1")
    r1 = ing.ingest(env)
    r2 = ing.ingest(env)
    assert r2.duplicate is True
    rid = ingest_receipt_id(r1.accepted[0], 1)
    # The replay converged onto the same source revision: still exactly
    # one DAG — six rows, no duplicated obligations.
    assert len(_obligation_rows(store, rid)) == 6
    with store.read() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM readiness_obligations"
        ).fetchone()[0]
    assert total == 6


# ---------------------------------------------------------------------
# V4-14.03 — wait_ready honesty
# ---------------------------------------------------------------------


def test_wait_ready_deadline_returns_pending_not_invented(store, cfg):
    ing = Ingester(store, cfg)
    r = ing.ingest(_env("a fact left undrained"))
    rid = ingest_receipt_id(r.accepted[0], 1)
    eng = ReadinessEngine(store)

    snap = eng.wait_ready(rid, deadline_us=now_us() + 150_000)
    assert snap["deadline_exceeded"] is True
    assert snap["ready"] is False
    assert "screened" in snap["pending"]
    # The wait never fabricated readiness: rows are still pending.
    assert _states(eng.receipt_state(rid))["screened"] == "pending"


def test_wait_ready_unknown_receipt_and_scope_fence(store, cfg):
    eng = ReadinessEngine(store)
    with pytest.raises(VerbatimError) as ei:
        eng.receipt_state("rc_ingest:no-such-source:1")
    assert ei.value.code is ErrorCode.NOT_FOUND_OR_FORBIDDEN

    ing = Ingester(store, cfg)
    r = ing.ingest(_env("scoped fact"))
    rid = ingest_receipt_id(r.accepted[0], 1)
    # A mismatched scope is indistinguishable from unknown (§9).
    with pytest.raises(VerbatimError) as ei:
        eng.receipt_state(rid, scope_id=scope_key(SCOPE2))
    assert ei.value.code is ErrorCode.NOT_FOUND_OR_FORBIDDEN
    with pytest.raises(VerbatimError) as ei:
        eng.wait_ready(
            rid,
            deadline_us=now_us() + 50_000,
            scope_id=scope_key(SCOPE2),
        )
    assert ei.value.code is ErrorCode.NOT_FOUND_OR_FORBIDDEN


# ---------------------------------------------------------------------
# drain settlement + dependency handling
# ---------------------------------------------------------------------


def test_drain_settles_full_pipeline(store, cfg):
    ing = Ingester(store, cfg, encoder=HashingEncoder(cfg.embedding))
    r = ing.ingest(_env("My editor is neovim."))
    rid = ingest_receipt_id(r.accepted[0], 1)

    rep = ing.drain_report(scope=None, limit=64)
    assert rep["processed"] > 0
    assert rep["still_pending"] == 0
    assert rep["complete"] is True

    snap = ReadinessEngine(store).receipt_state(rid)
    assert _states(snap) == {
        "accepted": "succeeded",
        "screened": "succeeded",
        "lexical_ready": "succeeded",
        "semantic_ready": "succeeded",
        "derived_ready": "succeeded",
        "failed": "succeeded",
    }
    assert snap["ready"] is True and snap["complete"] is True


def test_semantic_defers_without_encoder_sibling_unblocked(store, cfg):
    """No encoder provisioned: ``semantic_ready`` defers while
    ``derived_ready`` — its sibling on ``lexical_ready`` — still
    succeeds (the DAG is not a linear chain)."""
    ing = Ingester(store, cfg)  # no encoder
    r = ing.ingest(_env("My shell is zsh."))
    rid = ingest_receipt_id(r.accepted[0], 1)
    ing.drain_report(scope=None, limit=64)

    snap = ReadinessEngine(store).receipt_state(rid)
    st = _states(snap)
    assert st["lexical_ready"] == "succeeded"
    assert st["derived_ready"] == "succeeded"
    assert st["semantic_ready"] == "deferred"
    assert snap["states"]["semantic_ready"]["error"] == "encoder_unavailable"
    # Deferred is settled — the receipt is ready, honestly labeled.
    assert snap["ready"] is True
    assert "semantic_ready" in snap["deferred"]


def test_dependency_enforcement_typed_errors(store, cfg):
    """The engine refuses out-of-order fulfillment — durable deps are
    enforced, not advisory (V4-14.03)."""
    ing = Ingester(store, cfg)
    r = ing.ingest(_env("dependency ordering test"))
    rid = ingest_receipt_id(r.accepted[0], 1)
    eng = ReadinessEngine(store)

    with store.tx() as conn:
        # lexical before screened settles -> STALE_DEPENDENCY
        with pytest.raises(VerbatimError) as ei:
            eng.fulfill(conn, rid, CapabilityName.LEXICAL_READY)
        assert ei.value.code is ErrorCode.STALE_DEPENDENCY
        # A dep on a nonexistent obligation is INTEGRITY.
        eng.record_obligations(
            conn, "rc_x:1", scope_key(SCOPE),
            [CapabilityName.SCREENED],
            depends_on=[f"ro:{rid}:nonexistent_cap"],
        )
        with pytest.raises(VerbatimError) as ei:
            eng.fulfill(conn, "rc_x:1", CapabilityName.SCREENED)
        assert ei.value.code is ErrorCode.INTEGRITY
        # In-order transitions succeed.
        eng.fulfill(conn, rid, CapabilityName.SCREENED)
        eng.fulfill(conn, rid, CapabilityName.LEXICAL_READY)
        assert (
            eng.state_of(conn, rid, CapabilityName.LEXICAL_READY)
            == "succeeded"
        )
        # A succeeded obligation cannot be failed afterwards.
        with pytest.raises(VerbatimError) as ei:
            eng.fail(conn, rid, CapabilityName.LEXICAL_READY, "late")
        assert ei.value.code is ErrorCode.INVALID_TRANSITION


# ---------------------------------------------------------------------
# failure cascade + visibility (V4-14.07)
# ---------------------------------------------------------------------


def test_terminal_failure_cascades_and_stays_visible(store, cfg):
    ing = Ingester(store, cfg)
    r = ing.ingest(_env("payload will vanish before harvest"))
    sid = r.accepted[0]
    rid = ingest_receipt_id(sid, 1)
    # Destroy the evidence after acceptance — harvest must fail
    # terminally, not retry forever.
    with store.tx() as conn:
        conn.execute(
            "DELETE FROM source_revisions WHERE source_id = ?", (sid,)
        )

    rep = ing.drain_report(scope=None, limit=64)
    assert rep["failed"] >= 1
    assert rep["errors"]

    eng = ReadinessEngine(store)
    snap = eng.receipt_state(rid)
    st = _states(snap)
    assert st["accepted"] == "succeeded"  # committed work never rewritten
    assert st["screened"] == "failed"
    assert st["lexical_ready"] == "cancelled"
    assert st["semantic_ready"] == "cancelled"
    assert st["derived_ready"] == "cancelled"
    assert st["failed"] == "failed"
    assert snap["complete"] is True and snap["ready"] is False

    # wait_ready reports the failure promptly — no waiting on a
    # terminally-settled receipt.
    res = eng.wait_ready(rid, deadline_us=now_us() + 50_000)
    assert res["complete"] is True
    assert "screened" in res["failed"]
    assert "failed" in res["failed"]
    assert res.get("deadline_exceeded") is None


# ---------------------------------------------------------------------
# restart persistence (V4-14.07 — obligations are durable rows)
# ---------------------------------------------------------------------


def test_obligations_survive_restart(tmp_path, cfg):
    path = str(tmp_path / "restart.db")
    store = Store.create(path)
    ing = Ingester(store, cfg)
    r = ing.ingest(_env("restart persistence fact"))
    rid = ingest_receipt_id(r.accepted[0], 1)
    store.close()

    store2 = Store.open(path)
    try:
        eng = ReadinessEngine(store2)
        snap = eng.receipt_state(rid)
        st = _states(snap)
        assert st["accepted"] == "succeeded"
        assert st["screened"] == "pending"
        assert snap["ready"] is False

        # The same obligations drive completion after restart.
        Ingester(store2, cfg).drain_report(scope=None, limit=64)
        snap = eng.receipt_state(rid)
        assert snap["complete"] is True
        assert _states(snap)["screened"] == "succeeded"
    finally:
        store2.close()


# ---------------------------------------------------------------------
# C26/C88 — receipt isolation; unrelated work is never proof
# ---------------------------------------------------------------------


def test_unrelated_events_and_receipts_not_readiness_proof(store, cfg):
    """Another scope's capture+drain moves global counters — receipt A's
    readiness is untouched (C26/C88)."""
    ing = Ingester(store, cfg)
    rA = ing.ingest(_env("receipt A about apples", scope=SCOPE, ext="a1"))
    ridA = ingest_receipt_id(rA.accepted[0], 1)
    rB = ing.ingest(_env("receipt B about bananas", scope=SCOPE2, ext="b1"))
    ridB = ingest_receipt_id(rB.accepted[0], 1)

    eng = ReadinessEngine(store)
    # Drain B's scope only — all of B's jobs settle, A's stay queued.
    rep = ing.drain_report(scope=SCOPE2, limit=64)
    assert rep["processed"] > 0
    snapB = eng.receipt_state(ridB)
    assert snapB["ready"] is True

    # Global progress (events, completed jobs, other receipts) does not
    # move A: its own rows are still pending.
    with store.read() as conn:
        seq = conn.execute("SELECT MAX(event_seq) FROM events").fetchone()[0]
        done = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE state='succeeded'"
        ).fetchone()[0]
    assert seq and seq > 0 and done > 0
    snapA = eng.wait_ready(ridA, deadline_us=now_us() + 150_000)
    assert snapA["ready"] is False
    assert snapA["deadline_exceeded"] is True
    assert "screened" in snapA["pending"]

    # And A settles only when its own obligations do.
    ing.drain_report(scope=SCOPE, limit=64)
    snapA = eng.receipt_state(ridA)
    assert snapA["ready"] is True


# ---------------------------------------------------------------------
# V4-14.05/06 — honest drain reporting + priority reconsideration
# ---------------------------------------------------------------------


def test_drain_report_limit_is_not_completion(store, cfg):
    ing = Ingester(store, cfg)
    ing.ingest(_env("first fact", ext="d1"))
    ing.ingest(_env("second fact", ext="d2"))

    rep = ing.drain_report(scope=None, limit=1)
    assert rep["processed"] == 1
    assert rep["still_pending"] >= 1
    assert rep["complete"] is False

    rep2 = ing.drain_report(scope=None, limit=64)
    assert rep2["still_pending"] == 0
    assert rep2["complete"] is True
    assert rep2["succeeded"] == rep2["processed"]
    assert rep2["failed"] == 0
    assert "pending_obligations" in rep2


def test_drain_reconsiders_higher_priority_followups(store, cfg):
    """V4-14.06: a control-lane job enqueued MID-drain runs before the
    remaining ordinary work — the priority scan is per-job, not per-call."""
    ing = Ingester(store, cfg)
    sid = scope_key(SCOPE)
    with store.tx() as conn:
        ing.jobs.enqueue(conn, sid, JobKind.EPISODE_INDEX, {"n": 1})
        ing.jobs.enqueue(conn, sid, JobKind.EPISODE_INDEX, {"n": 2})

    order: list[int] = []
    real_execute = ing._execute
    fired = {"enqueued": False}

    def spy(job, owner):
        order.append(job["input_refs"].get("n"))
        if not fired["enqueued"]:
            fired["enqueued"] = True
            # Newly-arrived higher-priority work, enqueued mid-drain.
            with store.tx() as conn:
                ing.jobs.enqueue(
                    conn, sid, JobKind.EPISODE_INDEX, {"n": 3},
                    lane="control",
                )
        return real_execute(job, owner)

    ing._execute = spy
    try:
        rep = ing.drain_report(scope=SCOPE, limit=8)
    finally:
        ing._execute = real_execute
    assert rep["processed"] == 3 and rep["succeeded"] == 3
    assert order == [1, 3, 2]


# ---------------------------------------------------------------------
# C83 / V4-14.07/11 — session-end failure visibility
# ---------------------------------------------------------------------


def test_session_end_drain_failure_visible_and_durable(store, cfg):
    engine = Engine(
        store, cfg,
        LocalHost(profile_id="p", principal_id="alice", conversation_id="c1"),
    )
    engine.ingest(_env("session-end fact"))
    prov = _provider_on(engine)

    real = engine._ingester.run_pending

    def boom(*a, **k):
        raise VerbatimError(ErrorCode.STORE_BUSY, "injected drain failure")

    engine._ingester.run_pending = boom
    try:
        prov.on_session_end([])  # must not raise into the host
    finally:
        engine._ingester.run_pending = real

    # Visible: typed error + pending counts on the status surface.
    err = prov.drain_error
    assert err is not None
    assert err["code"] == ErrorCode.STORE_BUSY.value
    assert "injected drain failure" in err["error"]
    status = prov.drain_status()
    assert status["drain_error"] == err
    assert status["pending_jobs"] >= 1
    assert status["readiness"]["pending"] >= 1

    # Durable: journaled; obligations stay queued, not discarded.
    assert _events(store, "session_end_drain_failed")
    rid = ingest_receipt_id(_source_id(store), 1)
    assert _states(ReadinessEngine(store).receipt_state(rid))[
        "screened"
    ] == "pending"

    # Recovery: the same seam completes honestly once the fault clears.
    prov.on_session_end([])
    assert prov.drain_error is None
    status = prov.drain_status()
    assert status["pending_jobs"] == 0
    assert status["readiness"]["pending"] == 0
    report = status["last_drain_report"]
    assert report is not None
    assert report["processed"] > 0
    assert "succeeded" in report and "failed" in report


def _source_id(store: Store) -> str:
    with store.read() as conn:
        return conn.execute(
            "SELECT source_id FROM sources ORDER BY created_us LIMIT 1"
        ).fetchone()[0]


# ---------------------------------------------------------------------
# v2 Engine path — wait_ready through the public API
# ---------------------------------------------------------------------


def test_engine_wait_ready_and_drain_report(store, cfg):
    engine = Engine(
        store, cfg,
        LocalHost(profile_id="p", principal_id="alice", conversation_id="c1"),
    )
    rec = engine.ingest(_env("My editor is emacs."))
    rid = ingest_receipt_id(rec.accepted[0], 1)

    # Receipt identity by (source_id, revision) resolves too.
    snap = engine.receipt_state(source_id=rec.accepted[0], revision=1)
    assert snap["receipt_id"] == rid
    assert _states(snap)["screened"] == "pending"

    rep = engine.drain_report(limit=64)
    assert rep["complete"] is True

    snap = engine.wait_ready(rid, timeout_s=1.0)
    assert snap["ready"] is True
    assert _states(snap)["lexical_ready"] == "succeeded"


def test_engine_wait_ready_denies_without_grant(store, cfg):
    """A bound caller without READ_EVIDENCE never reads readiness."""
    from verbatim.core.types import CallerContext, GrantKind

    engine = Engine(
        store, cfg,
        LocalHost(profile_id="p", principal_id="alice", conversation_id="c1"),
    )
    rec = engine.ingest(_env("grant-gated fact"))
    rid = ingest_receipt_id(rec.accepted[0], 1)
    no_read = CallerContext(
        profile_id="p",
        principal_id="mallory",
        grants=frozenset({GrantKind.INGEST}),
    )
    with pytest.raises(VerbatimError) as ei:
        engine.wait_ready(rid, caller=no_read, timeout_s=0.1)
    assert ei.value.code is ErrorCode.NOT_FOUND_OR_FORBIDDEN


# ---------------------------------------------------------------------
# v3 facade path — capture + wait_ready (V4-14.01/03/09)
# ---------------------------------------------------------------------


def test_v3_facade_agent_note_settles_at_capture(store):
    """Agent-authored kinds never enter derivation (V3-13.11): the DAG
    records with downstream stages deferred at capture, not pending
    forever."""
    facade = VerbatimV3(store)
    facade.issue_capture_authorization(AGENT, V3_SCOPE, granted_by=AGENT)
    sid = facade.capture_submitted(AGENT, V3_SCOPE, "restart the gateway first")

    snap = facade.wait_ready(
        source_id=sid, timeout_s=0.5,
        principal_id=AGENT, scope_id=V3_SCOPE,
    )
    assert snap["ready"] is True
    st = _states(snap)
    assert st["accepted"] == "succeeded"
    assert st["screened"] == "succeeded"
    assert st["lexical_ready"] == "deferred"
    assert st["failed"] == "succeeded"


def test_v3_facade_pipeline_kind_drains_to_ready(store, cfg):
    """A pipeline-eligible v3 capture pends until its own jobs drain —
    then reports ready through the same wait surface."""
    facade = VerbatimV3(store)
    facade.issue_capture_authorization(AGENT, V3_SCOPE, granted_by=AGENT)
    sid = facade.capture_submitted(
        AGENT, V3_SCOPE,
        "pytest run finished: 3 passed, 1 failed",
        declared_type="tool_result",
    )
    eng = ReadinessEngine(store)
    rid = ingest_receipt_id(sid, 1)
    snap = eng.receipt_state(rid)
    assert _states(snap)["screened"] == "pending"

    Ingester(store, cfg).drain_report(scope=None, limit=64)
    snap = facade.wait_ready(
        source_id=sid, timeout_s=0.5,
        principal_id=AGENT, scope_id=V3_SCOPE,
    )
    assert snap["ready"] is True
    assert _states(snap)["screened"] == "succeeded"


def test_v3_direct_ingest_lazy_convergence(store, cfg):
    """A capture that bypassed obligation recording still converges:
    the first drain/read materializes the durable DAG from the
    persisted source rows (no second capture path)."""
    env = SourceEnvelopeV3(
        kind=EnvelopeKind.USER_MESSAGE,
        scope_id=V3_SCOPE,
        actor_principal="alice",
        perspective=Perspective(asserter="alice", observer="alice"),
        content=b"the launch checklist is on the wiki",
        media_type="text/plain",
        host_id="h1",
        session_id="s1",
        event_us=now_us(),
        receipt_us=0,
        metadata={},
    )
    with store.tx() as conn:
        receipt = ingest_envelope(conn, store, env)
    rid = ingest_receipt_id(receipt.source_id, int(receipt.revision))
    # Capture did not record obligations — the table is still empty.
    assert _obligation_rows(store, rid) == {}

    # First job touch converges: drain materializes + settles the DAG.
    rep = Ingester(store, cfg).drain_report(scope=None, limit=64)
    assert rep["processed"] > 0
    snap = ReadinessEngine(store).receipt_state(rid)
    assert snap["ready"] is True
    assert _states(snap)["accepted"] == "succeeded"


def test_v3_wait_ready_denies_other_scope(store):
    """Cross-scope readiness reads are indistinguishable from unknown."""
    facade = VerbatimV3(store)
    facade.issue_capture_authorization(AGENT, V3_SCOPE, granted_by=AGENT)
    sid = facade.capture_submitted(AGENT, V3_SCOPE, "a scoped note")
    rid = ingest_receipt_id(sid, 1)
    with pytest.raises(VerbatimError) as ei:
        facade.receipt_state(
            rid, principal_id=AGENT, scope_id="scope:other"
        )
    assert ei.value.code in (
        ErrorCode.NOT_FOUND_OR_FORBIDDEN,
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
    )
