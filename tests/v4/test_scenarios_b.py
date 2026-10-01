"""SPEC_V4 §58 adversarial acceptance scenarios C17–C32.

Every scenario drives the REAL public path on a real file-backed
``Store.create(tmp_path)`` — no isolated mocks, no simulated seams.
Deterministic interleavings stand in for races: injected clocks, explicit
transaction boundaries, held locks, store close/reopen, and patched seam
hooks (the same technique ``tests/test_f409_admit.py`` uses). No sleeps.

Covered:

- C17  revocation between inference and commit fences both plan
       publication (Coordinator.apply_plan) and permit issuance
       (Kernel.seal_delivery) — V4-09.04/09.07, V4-11.07, V4-42.01.
- C18  expired / replayed / wrong-recipient dispatch permits cannot send
       content — V4-12.02/12.03 (TransportBroker).
- C19  a made-up checker name + declared success stays an agent report,
       never a verified episode outcome — V4-23.01/23.02.
- C20  a valid receipt bound to another task/scope/artifact/invocation
       cannot certify this task — V4-23.01.
- C21  cancellation before admission leaves no claim, edge, index, or
       receipt — V4-09.04/42.03.
- C22  a crash between domain application and response yields exactly
       one committed effect and a replayable receipt — V4-09.03/09.05.
- C23  restart reclaims an abandoned lease; the stale worker commits
       nothing — V4-42.01/42.08.
- C24  duplicate operation lookup succeeds under queue saturation —
       V4-42.06.
- C25  privacy work progresses under enrichment saturation without
       starving another scope — V4-42.04.
- C26  unrelated later events do not satisfy readiness for an unfinished
       captured input — V4-14.02/03.
- C27  a drain re-checks earlier-priority lanes for follow-up work and
       never reports premature completion — V4-14.05/06, V4-42.02.
- C28  writer contention cannot hold a foreground request past its
       admitted deadline — V4-40.02/03/10.
- C29  a missing mandatory handler/capability is a durable typed failure,
       not silent success — V4-40.08, §03 honest-state.
- C30  unauthorized and absent are indistinguishable: errors, metadata
       counts, and handles reveal no existence — V4-08.06, V4-10.04.
- C31  >160 held lexical matches cannot hide the next eligible claim —
       V4-28.02.
- C32  private alpha-heavy documents cannot reorder an unchanged public
       result set — V4-28.03, V4-28.10 (metamorphic bound, §58.03).
"""

from __future__ import annotations

import json
import random
import sqlite3
import time
from dataclasses import replace
from decimal import Decimal
from typing import Any, Optional

import pytest

from verbatim.api_v3 import VerbatimV3
from verbatim.config import EmbeddingConfig, JudgeConfig, VerbatimConfig
from verbatim.core.time import now_us
from verbatim.core.types import (
    ErrorCode,
    JobKind,
    Mode,
    Provenance,
    Scope,
    SourceEnvelope,
    SourceKind,
    VerbatimError,
)
from verbatim.core.types_v3 import (
    EnvelopeKind,
    OutcomeClass,
    Perspective,
    RecallRequestV3,
    SourceEnvelopeV3,
    TrustClass,
    Verb,
)
from verbatim.core.types_v4 import (
    DispatchPermit,
    Effect,
    EffectKind,
    EffectPlan,
    EvidenceLocator,
    JobRequest,
)
from verbatim.evidence import ingest_envelope
from verbatim.evidence.receipts import receipt_for_envelope
from verbatim.experience import episodes_v3
from verbatim.governance import (
    CallerV3,
    create_grant,
    register_principal,
    revoke_grant,
    seed_purposes,
)
from verbatim.ingest import Ingester
from verbatim.jobs.coordinator import Coordinator
from verbatim.jobs.queue import JobQueue
from verbatim.kernel import Kernel
from verbatim.privacy.broker import (
    EndpointDescriptor,
    TransportBroker,
    payload_digest,
)
from verbatim.readiness import ingest_receipt_id
from verbatim.retrieval.v3 import recall_v3
from verbatim.storage import repos_v3
from verbatim.storage.repos import ConsentsRepo
from verbatim.storage.store import Store
from tests.conftest import FakeClock, qrow


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

SCOPE = Scope(profile_id="p", principal_id="alice", conversation_id="c1")
API_SCOPE = "scope:v4-c"
AGENT = "agent-1"


@pytest.fixture()
def store(tmp_path):
    s = Store.create(str(tmp_path / "v4.db"))
    yield s
    s.close()


def _capture_cfg(require_review: bool = True) -> VerbatimConfig:
    cfg = VerbatimConfig()
    cfg = replace(
        cfg,
        capture=replace(
            cfg.capture,
            enabled=True,
            user_messages=True,
            assistant_context=True,
            tool_outputs=True,
        ),
    )
    if not require_review:
        cfg = replace(cfg, admission=replace(cfg.admission, require_review=False))
    return cfg


def _env(scope: Scope, text: str, *, source_id: Optional[str] = None) -> SourceEnvelope:
    return SourceEnvelope(
        origin="test:c-scenarios",
        source_kind=SourceKind.USER_MESSAGE,
        scope=scope,
        speaker_id=scope.principal_id,
        payload=text.encode("utf-8"),
        event_us=now_us(),
        captured_us=now_us(),
        provenance=Provenance.DIRECT_USER,
        source_id=source_id,
    )


def _seed_governance(conn, *scope_ids: str, principals=("human:alice",)) -> None:
    """Scope rows + purpose registry + principals (kernel conftest shape)."""
    for sid in scope_ids:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES (?, 'prof', 'owner')",
            (sid,),
        )
    seed_purposes(conn)
    for pid in principals:
        kind = pid.split(":", 1)[0] if ":" in pid else "human"
        register_principal(conn, kind=kind, principal_id=pid)


def _grant(conn, scope_id: str, principal_id: str, verbs, purposes=("recall",)) -> str:
    return create_grant(
        conn,
        scope_id=scope_id,
        principal_id=principal_id,
        verbs=verbs,
        issuer_id=principal_id,
        purposes=list(purposes),
    )


def _job_row(store: Store, job_id: str) -> dict[str, Any]:
    with store.read() as conn:
        row = qrow(conn, "SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    assert row is not None
    return row


def _table_count(store: Store, table: str, where: str = "1=1") -> int:
    with store.read() as conn:
        return conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {where}"
        ).fetchone()[0]


def _err_of(fn) -> tuple[ErrorCode, str]:
    with pytest.raises(VerbatimError) as ei:
        fn()
    return (ei.value.code, str(ei.value))


class _FakeEncoder:
    """Makes admission's embed follow-up real so its absence is load-bearing."""

    @property
    def encoder_id(self) -> str:
        return "fake:c-scenarios:v1"

    def available(self) -> bool:
        return True

    def encode(self, texts: list[str]) -> list[bytes]:
        return [b"\x00" * 16 for _ in texts]

    def manifest(self) -> dict[str, Any]:
        return {"preprocessing_version": "p1"}


def _plan(operation_id: str, scope_id: str, **kw: Any) -> EffectPlan:
    base: dict[str, Any] = dict(
        operation_id=operation_id,
        scope_id=scope_id,
        producer_id="producer:c-scenarios",
        input_digests=(f"sha256:{operation_id}",),
        effects=(
            Effect(
                EffectKind.INSERT_OBJECT,
                "objects",
                {
                    "object_id": f"obj:{operation_id}",
                    "kind": "claim",
                    "scope_id": scope_id,
                    "current_revision": 1,
                    "created_event": 1,
                },
            ),
        ),
    )
    base.update(kw)
    return EffectPlan(**base)


# ---------------------------------------------------------------------------
# C17 — revocation between inference and commit (V4-09.04/09.07, V4-11.07)
# ---------------------------------------------------------------------------


def test_c17_revocation_between_plan_and_apply_fences_everything(store):
    """C17 — a grant revoked after access resolution but before commit must
    block BOTH the fenced plan application AND delivery-permit sealing
    (V4-09.04 stale worker commits nothing; V4-09.07 seal re-authorizes;
    V4-11.07 the pinned scope epoch is the fence)."""
    sid = "scope:c17"
    with store.tx() as conn:
        _seed_governance(conn, sid)
        gid = _grant(conn, sid, "human:alice", {"read", "quote"})

    kernel = Kernel(store)
    caller = CallerV3(principal_id="human:alice")
    with store.read() as conn:
        lease = kernel.resolve_access(conn, caller, Verb.QUOTE, "recall", [sid])
    assert not lease.denied
    pinned = dict(lease.epoch_vector)
    assert pinned == {sid: 0}

    # A worker holds a live lease on a real queue job — the plan is
    # worker-driven, fenced by generation AND epoch.
    queue = JobQueue(store, clock=FakeClock(), rng=random.Random(1))
    with store.tx() as conn:
        job_id = queue.enqueue(conn, sid, JobKind.EPISODE_INDEX, {"w": 1})
    leased = queue.lease(sid, list(JobKind), owner="w-inference", limit=1)
    generation = leased[0]["generation"]

    coord = Coordinator(store)

    # The revocation lands while inference is still in flight.
    with store.tx() as conn:
        receipt = revoke_grant(conn, gid)
    assert receipt["epoch"] == 1

    # The plan computed under the old epoch can no longer publish —
    # the lease is still live, but the epoch pin is stale.
    with pytest.raises(VerbatimError) as ei:
        coord.apply_plan(
            _plan("op:c17-a", sid, epoch_vector=pinned),
            job_id=job_id,
            expected_lease=generation,
        )
    assert ei.value.code is ErrorCode.STALE_EPOCH

    # …and the delivery permit cannot be sealed either: seal rechecks the
    # lease inside the permit transaction — the epoch drift fences before
    # the per-scope re-authorization even runs (V4-09.07).
    with pytest.raises(VerbatimError) as ei:
        with store.tx() as conn:
            kernel.seal_delivery(conn, lease, b"pack bytes")
    assert ei.value.code in (
        ErrorCode.STALE_EPOCH,
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
    )

    # Nothing committed anywhere: no object, no receipt, no permit.
    assert _table_count(store, "objects") == 0
    assert _table_count(store, "operation_receipts") == 0
    assert _table_count(store, "delivery_permits") == 0

    # Control: a plan pinned at the POST-revocation epoch is not stale —
    # the fence fires for pre-revocation pins only, not for the scope.
    receipt2 = coord.apply_plan(
        _plan("op:c17-b", sid, epoch_vector={sid: 1}),
        job_id=job_id,
        expected_lease=generation,
    )
    assert receipt2.effects_applied == 1
    assert _table_count(store, "objects") == 1


# ---------------------------------------------------------------------------
# C18 — expired / replayed / wrong-recipient dispatch permits (V4-12.02/03)
# ---------------------------------------------------------------------------

CF_ID = "cloudflare:api.cloudflare.com"
OL_ID = "ollama:127.0.0.1:11434"
EGRESS_PURPOSE = "embed_document"
CF_DESC = EndpointDescriptor(
    endpoint_id=CF_ID, origin="https://api.cloudflare.com", account="acct-1"
)
OL_DESC = EndpointDescriptor(
    endpoint_id=OL_ID, origin="http://127.0.0.1:11434", require_tls=False
)


def _egress_cfg() -> VerbatimConfig:
    return VerbatimConfig(
        mode=Mode.REMOTE_ASSISTED,
        judge=JudgeConfig(backend="jev", daily_budget_usd=Decimal("1.00")),
        embedding=EmbeddingConfig(backend="cloudflare", account_id="acct-1"),
    )


def _broker(store: Store, clock: FakeClock) -> TransportBroker:
    return TransportBroker(
        store, _egress_cfg(), endpoints=(CF_DESC, OL_DESC), clock=clock
    )


def _digest(*texts: str) -> str:
    return payload_digest(json.dumps({"input": list(texts)}).encode())


def _open_dispatch(broker, store, sid, **kw):
    args = dict(
        caller="ingest.embed",
        recipient=CF_ID,
        purpose=EGRESS_PURPOSE,
        payload_digest=_digest("hello"),
        scope_ids=[sid],
        max_spend=0.001,
        est_tokens=4,
    )
    args.update(kw)
    return broker.open_dispatch(store, **args)


def test_c18_expired_replayed_wrong_recipient_permits_cannot_dispatch(tmp_path):
    """C18 — a dispatch permit is one-use, short-lived, and bound to its
    recipient + payload digest: an expired, replayed, wrong-recipient,
    digest-mismatched, or forged permit can never carry content off-box
    (V4-12.02 issuance, V4-12.03 dispatch-time recheck)."""
    store = Store.create(str(tmp_path / "c18.db"))
    clock = FakeClock()
    sid = "scope:c18"
    try:
        with store.tx() as conn:
            ConsentsRepo(store).grant(conn, sid, CF_ID, EGRESS_PURPOSE, "digest-1")
        broker = _broker(store, clock)

        # --- wrong recipient: the permit names CF_ID; presenting it to the
        # Ollama transport denies without consuming it.
        permit = _open_dispatch(broker, store, sid)
        with pytest.raises(VerbatimError) as ei:
            broker.dispatch(
                permit, recipient=OL_ID, payload_digest=permit.payload_digest
            )
        assert ei.value.code is ErrorCode.EGRESS_DENIED
        assert broker.permit(permit.permit_id).state == "open"

        # --- digest mismatch: the permit does not cover a tampered body.
        with pytest.raises(VerbatimError) as ei:
            broker.dispatch(
                permit, payload_digest=_digest("tampered-payload")
            )
        assert ei.value.code is ErrorCode.EGRESS_DENIED
        assert broker.permit(permit.permit_id).state == "open"

        # --- correct handoff consumes the permit exactly once.
        out = broker.dispatch(
            permit, recipient=CF_ID, payload_digest=permit.payload_digest
        )
        assert out.state == "dispatched"

        # --- replay: the same permit can never dispatch twice.
        with pytest.raises(VerbatimError) as ei:
            broker.dispatch(
                permit, recipient=CF_ID, payload_digest=permit.payload_digest
            )
        assert ei.value.code is ErrorCode.EGRESS_DENIED
        assert broker.permit(permit.permit_id).state == "dispatched"

        # --- expired: a permit that outlives its bound lifetime cannot send.
        p2 = _open_dispatch(broker, store, sid)
        clock.advance(2.0)  # past DEFAULT_MAX_PERMIT_AGE_US (1s)
        with pytest.raises(VerbatimError) as ei:
            broker.dispatch(p2, recipient=CF_ID, payload_digest=p2.payload_digest)
        assert ei.value.code is ErrorCode.PERMIT_EXPIRED
        assert broker.permit(p2.permit_id).state == "expired"

        # --- forged: a caller-constructed DispatchPermit was never minted.
        forged = DispatchPermit(
            permit_id="dpermit:forged",
            recipient=CF_ID,
            purpose=EGRESS_PURPOSE,
            payload_digest=_digest("hello"),
            scope_ids=(sid,),
            consent_refs=(),
            reservation_id="res:forged",
            max_spend=0.001,
            issued_us=clock(),
            expires_us=clock() + 1_000_000,
        )
        with pytest.raises(VerbatimError) as ei:
            broker.dispatch(forged)
        assert ei.value.code is ErrorCode.EGRESS_DENIED

        # Exactly one dispatch ever left the box.
        assert _table_count(
            store, "dispatch_permits", "state = 'dispatched'"
        ) == 1
    finally:
        store.close()


# ---------------------------------------------------------------------------
# C19 — made-up checker stays self-reported (V4-23.01/02)
# C20 — receipts bound elsewhere cannot certify (V4-23.01)
# ---------------------------------------------------------------------------


def _bootstrap(facade: VerbatimV3) -> None:
    facade.issue_capture_authorization(AGENT, API_SCOPE, granted_by=AGENT)


def _ep_outcome(conn, episode_id: str) -> Optional[str]:
    row = conn.execute(
        "SELECT outcome FROM episodes WHERE episode_id = ?", (episode_id,)
    ).fetchone()
    return row[0] if row else None


class _Resolver:
    """Host checker resolver: the ONLY source of host-observed receipts."""

    resolver_id = "host-checker-resolver:test"

    def __init__(self, receipts):
        self._receipts = dict(receipts)

    def resolve_checker(self, invocation_id):
        return self._receipts.get(invocation_id)


def _receipt(inv: str = "inv-1", **over) -> dict[str, Any]:
    base = {
        "checker_id": "pytest-runner",
        "checker_version": "1.0",
        "invocation_id": inv,
        "completed": True,
        "exit_code": 0,
        "scope_id": API_SCOPE,
        "task_id": "task-1",
        "artifact_digest": "art-1",
        "environment_digest": "env-1",
        "nonce": "n-1",
        "issued_us": 42,
    }
    base.update(over)
    return base


def test_c19_made_up_checker_stays_self_reported(store):
    """C19 — with no host checker resolver wired, a caller-named checker
    and a declared ``success`` persist as an agent-generated report that
    can never upgrade an episode outcome (V4-23.01/23.02)."""
    facade = VerbatimV3(store)  # no checker_resolver — nothing is attested
    _bootstrap(facade)
    out = facade.submit_outcome(
        API_SCOPE,
        principal_id=AGENT,
        outcome="success",
        checker_id="totally-made-up-checker",
        task_id="task-1",
    )
    assert out["attested"] is False
    assert out["agent_report"] is True

    with store.read() as conn:
        env = repos_v3.get(
            conn, "source_envelopes", {"envelope_id": out["envelope_id"]}
        )
        meta = repos_v3.json_field(env, "metadata_json", {})
    assert env["trust_class"] == TrustClass.AGENT_GENERATED.value
    assert meta["checker"]["host_attested"] is False
    assert meta["checker"]["agent_report"] is True
    # The episode machinery itself does not count a self-report.
    assert episodes_v3.envelope_outcome(meta) is None

    # End-to-end: the report joins the trajectory as evidence but the
    # episode outcome stays ``unknown`` — never "success".
    traj = facade.submit_trajectory(
        API_SCOPE,
        principal_id=AGENT,
        task_id="task-1",
        steps=[{"observation_envelope_ids": [out["envelope_id"]]}],
    )
    with store.tx() as conn:
        episode_id = episodes_v3.build_episode(conn, traj["trajectory_id"])
        outcome = _ep_outcome(conn, episode_id)
    assert outcome == "unknown"


@pytest.mark.parametrize(
    "over,submit_kw",
    [
        ({"scope_id": "scope:other"}, {}),                 # wrong scope
        ({"task_id": "other-task"}, {}),                   # wrong task
        ({"artifact_digest": "art-other"}, {}),            # wrong artifact
        ({"invocation_id": "inv-2"}, {}),                  # wrong invocation
        ({}, {"artifact_digest": "art-other"}),            # ask differs
        ({}, {"task_id": "task-2"}),                       # task ask differs
    ],
)
def test_c20_receipt_bound_elsewhere_cannot_certify(store, over, submit_kw):
    """C20 — a perfectly valid host receipt bound to another task, scope,
    artifact, or invocation cannot certify THIS submission: the binding
    check rejects it before any envelope persists (V4-23.01)."""
    receipt = _receipt(**over)
    facade = VerbatimV3(store, checker_resolver=_Resolver({"inv-1": receipt}))
    _bootstrap(facade)
    kw = {
        "principal_id": AGENT,
        "outcome": "success",
        "invocation_id": "inv-1",
        "artifact_digest": "art-1",
        "task_id": "task-1",
    }
    kw.update(submit_kw)
    with pytest.raises(VerbatimError) as ei:
        facade.submit_outcome(API_SCOPE, **kw)
    assert ei.value.code is ErrorCode.VALIDATION
    # No host-observed envelope escaped the binding check.
    with store.read() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM source_envelopes"
            " WHERE trust_class = ?",
            (TrustClass.HOST_OBSERVED.value,),
        ).fetchone()[0]
    assert n == 0


def test_c20_episode_ignores_receipt_bound_to_other_task(store):
    """C20 at episode build: even a host-attested-shaped receipt bound to
    a different task leaves this trajectory's outcome ``unknown``."""
    facade = VerbatimV3(store)
    _bootstrap(facade)
    meta = {
        "outcome": "success",
        "checker": {
            "checker_id": "host-checker",
            "invocation_id": "inv-9",
            "host_attested": True,
            "task_id": "other-task",  # bound elsewhere
            "scope_id": API_SCOPE,
        },
    }
    env = SourceEnvelopeV3(
        kind=EnvelopeKind.VERIFICATION,
        scope_id=API_SCOPE,
        actor_principal="host",
        perspective=Perspective(asserter="host"),
        event_us=now_us(),
        receipt_us=0,
        content=b'{"note": "host outcome write"}',
        media_type="application/json",
        trust_class=TrustClass.HOST_OBSERVED,
        task_id="task-1",
        metadata=meta,
    )
    with store.tx() as conn:
        rec = ingest_envelope(conn, store, env)
    traj = facade.submit_trajectory(
        API_SCOPE,
        principal_id=AGENT,
        task_id="task-1",
        steps=[{"observation_envelope_ids": [rec.envelope_id]}],
    )
    with store.tx() as conn:
        episode_id = episodes_v3.build_episode(conn, traj["trajectory_id"])
        outcome = _ep_outcome(conn, episode_id)
    assert outcome == "unknown"


# ---------------------------------------------------------------------------
# C21 — cancellation before admission leaves nothing behind (V4-09.04/42.03)
# ---------------------------------------------------------------------------


def _harvested(ing: Ingester, text: str = "My editor is neovim.") -> str:
    receipt = ing.ingest(_env(SCOPE, text))
    (sid,) = receipt.accepted
    ing.run_pending(scope=SCOPE, kinds=[JobKind.HARVEST])
    return sid


def _domain_counts(store: Store) -> dict[str, int]:
    return {
        "claims": _table_count(store, "claims"),
        "claim_revisions": _table_count(store, "claim_revisions"),
        "claim_evidence": _table_count(store, "claim_evidence"),
        "reviews": _table_count(store, "reviews"),
        "edges": _table_count(store, "edges"),
        "fts_rows": _table_count(store, "fts_rows"),
        "admit_ops": _table_count(
            store, "operations", "operation_key LIKE 'admit:%'"
        ),
        "operation_receipts": _table_count(store, "operation_receipts"),
        "delivery_permits": _table_count(store, "delivery_permits"),
        "embed_jobs": _table_count(store, "jobs", "kind = 'embed'"),
        "claim_events": _table_count(
            store, "events",
            "kind IN ('claim_proposed','admitted','pair_comparison')",
        ),
    }


def test_c21_cancel_before_admission_leaves_no_effects(store):
    """C21 — a cancel landing between dequeue and the admission commit
    leaves zero domain effects: no claim, revision, evidence link, review,
    edge, index row, operation receipt, embed follow-up, or audit event
    (V4-09.04 lease fencing, V4-42.03 cancellation fences publication)."""
    ing = Ingester(
        store, _capture_cfg(require_review=False), encoder=_FakeEncoder()
    )
    source_id = _harvested(ing)
    leased = ing.jobs.lease(
        SCOPE, [JobKind.ADMIT], owner="w-cancel", limit=1
    )
    assert leased, "expected an admit job after harvest"
    job = leased[0]

    assert ing.jobs.cancel(job["job_id"]) is True
    with pytest.raises(VerbatimError) as ei:
        ing._do_admit(job, "w-cancel")
    assert ei.value.code is ErrorCode.LEASE_LOST

    counts = _domain_counts(store)
    assert counts == {k: 0 for k in counts}, counts
    assert _job_row(store, job["job_id"])["state"] == "cancelled"

    # The readiness DAG is honest too: durable acceptance committed with
    # the capture, but the admission-stage capabilities the cancelled
    # worker would have fulfilled never reached ``succeeded``.
    snap = ing.readiness_engine().receipt_state(
        ingest_receipt_id(source_id, 1)
    )
    states = snap["states"]
    assert states["accepted"]["state"] == "succeeded"
    for cap in ("lexical_ready", "derived_ready", "semantic_ready"):
        assert states[cap]["state"] != "succeeded"


def test_c21_control_live_lease_admits_atomically(store):
    """C21 control — with the lease live, the same admission DOES produce
    the claim + FTS row + embed job in one commit, so the cancelled-path
    zeroes above are not vacuous."""
    ing = Ingester(
        store, _capture_cfg(require_review=False), encoder=_FakeEncoder()
    )
    _harvested(ing)
    job = ing.jobs.lease(SCOPE, [JobKind.ADMIT], owner="w-ok", limit=1)[0]
    ing._do_admit(job, "w-ok")
    counts = _domain_counts(store)
    assert counts["claims"] == 1
    assert counts["fts_rows"] == 1
    assert counts["embed_jobs"] == 1


# ---------------------------------------------------------------------------
# C22 — crash between apply and response: one effect, replayable receipt
# ---------------------------------------------------------------------------


def test_c22_crash_between_apply_and_response_replays_one_receipt(tmp_path):
    """C22 — the apply transaction commits; the process dies before the
    caller sees the response; after reopen, replaying the same operation
    returns the identical durable receipt and applies NOTHING twice
    (V4-09.03 atomicity, V4-09.05 idempotent replay)."""
    path = str(tmp_path / "c22.db")
    store = Store.create(path)
    coord = Coordinator(store)
    plan = _plan(
        "op:c22",
        "scope:c22",
        follow_ups=(JobRequest("harvest", "ordinary", {"why": "c22"}),),
    )

    first = coord.apply_plan(plan)
    # Crash here: the response never reached the caller.
    store.close()

    reopened = Store.open(path)
    try:
        second = Coordinator(reopened).apply_plan(plan)
        assert second.operation_id == first.operation_id
        assert second.applied_seq == first.applied_seq
        assert second.effects_applied == first.effects_applied == 1
        assert second.jobs_enqueued == first.jobs_enqueued
        assert second.input_digest == first.input_digest
        with reopened.read() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM objects"
            ).fetchone()[0] == 1
            assert conn.execute(
                "SELECT COUNT(*) FROM operation_receipts"
            ).fetchone()[0] == 1
            # The follow-up obligation also replayed once, not twice.
            assert conn.execute(
                "SELECT COUNT(*) FROM jobs"
            ).fetchone()[0] == 1
    finally:
        reopened.close()


# ---------------------------------------------------------------------------
# C23 — restart reclaims an abandoned lease; stale workers cannot commit
# ---------------------------------------------------------------------------


def test_c23_restart_reclaims_lease_and_fences_stale_worker(tmp_path):
    """C23 — a worker crashes holding a lease; after reopen, the queue
    reclaims it (bumped generation fences the dead worker), a replacement
    leases and commits, and the stale worker's generation can commit
    nothing ever again (V4-42.01 generation fencing, V4-42.08 reclaim)."""
    path = str(tmp_path / "c23.db")
    clock = FakeClock()
    sid = "scope:c23"

    store = Store.create(path)
    q1 = JobQueue(store, clock=clock, rng=random.Random(1))
    with store.tx() as conn:
        job_id = q1.enqueue(conn, sid, JobKind.EPISODE_INDEX, {"n": 1})
    old = q1.lease(sid, list(JobKind), owner="w-old", limit=1, lease_s=5.0)[0]
    old_gen = old["generation"]
    store.close()  # crash: w-old never commits, never releases

    reopened = Store.open(path)
    try:
        q2 = JobQueue(reopened, clock=clock, rng=random.Random(1))
        clock.advance(10.0)  # past the 5s lease
        assert q2.reclaim_expired() == 1

        row = _job_row(reopened, job_id)
        assert row["state"] == "retry_wait"
        assert row["generation"] != old_gen

        new = q2.lease(sid, list(JobKind), owner="w-new", limit=1, lease_s=60.0)[0]
        new_gen = new["generation"]
        assert new_gen != old_gen

        coord = Coordinator(reopened)
        # The stale worker's fencing token is dead.
        with pytest.raises(VerbatimError) as ei:
            coord.apply_plan(
                _plan("op:c23-stale", sid), job_id=job_id, expected_lease=old_gen
            )
        assert ei.value.code is ErrorCode.LEASE_LOST
        # Its completion lands on nothing either.
        assert q2.complete(job_id, "w-old", old_gen) is False

        # The replacement worker commits exactly once.
        receipt = coord.apply_plan(
            _plan("op:c23", sid), job_id=job_id, expected_lease=new_gen
        )
        assert receipt.effects_applied == 1
        assert q2.complete(job_id, "w-new", new_gen) is True

        # Post-commit, the stale worker's plan hits the terminal fence.
        with pytest.raises(VerbatimError) as ei:
            coord.apply_plan(
                _plan("op:c23-stale2", sid),
                job_id=job_id,
                expected_lease=old_gen,
            )
        assert ei.value.code is ErrorCode.CANCELLED

        assert _table_count(reopened, "objects") == 1
        assert _table_count(reopened, "operation_receipts") == 1
    finally:
        reopened.close()


# ---------------------------------------------------------------------------
# C24 — dedup lookup under saturation; C25 — privacy lane under saturation
# ---------------------------------------------------------------------------


def test_c24_duplicate_lookup_succeeds_under_queue_saturation(store):
    """C24 — V4-42.06: deduplicated retries resolve the EXISTING
    obligation before new-item backpressure applies — a full ordinary
    lane cannot make receipt lookup non-idempotent."""
    q = JobQueue(store, cap=2, clock=FakeClock(), rng=random.Random(1))
    sid = "scope:c24"
    with store.tx() as conn:
        j1 = q.enqueue(
            conn, sid, JobKind.EPISODE_INDEX, {"n": 1}, dedup_key=b"c24:one"
        )
        q.enqueue(
            conn, sid, JobKind.EPISODE_INDEX, {"n": 2}, dedup_key=b"c24:two"
        )
        # The lane is at cap: genuinely NEW work is backpressured…
        with pytest.raises(VerbatimError) as ei:
            q.enqueue(
                conn, sid, JobKind.EPISODE_INDEX, {"n": 3},
                dedup_key=b"c24:new",
            )
        assert ei.value.code is ErrorCode.BACKPRESSURE
        assert ei.value.retryable is True
        # …but re-presenting an existing dedup key resolves the durable
        # job — saturation cannot turn idempotent lookup into an error.
        again = q.enqueue(
            conn, sid, JobKind.EPISODE_INDEX, {"n": 1}, dedup_key=b"c24:one"
        )
        assert again == j1
        assert conn.execute(
            "SELECT COUNT(*) FROM jobs"
        ).fetchone()[0] == 2


def test_c25_privacy_work_progresses_under_enrichment_saturation(store):
    """C25 — V4-42.04: privacy/control work has reserved capacity — a
    saturated ordinary lane cannot block it — and per-scope caps keep one
    scope's saturation from starving another scope."""
    q = JobQueue(store, cap=2, clock=FakeClock(), rng=random.Random(1))
    with store.tx() as conn:
        q.enqueue(conn, "scope:c25-a", JobKind.EPISODE_INDEX, {"n": 1})
        q.enqueue(conn, "scope:c25-a", JobKind.EPISODE_INDEX, {"n": 2})
        # scope:c25-a's ordinary lane is saturated.
        with pytest.raises(VerbatimError) as ei:
            q.enqueue(conn, "scope:c25-a", JobKind.EPISODE_INDEX, {"n": 3})
        assert ei.value.code is ErrorCode.BACKPRESSURE

        # Reserved lane: privacy work enqueues anyway.
        priv = q.enqueue(
            conn, "scope:c25-a", JobKind.QUARANTINE_REVIEW, {"q": 1}
        )
        # A different scope's ordinary work is untouched by the
        # saturation — no cross-scope starvation.
        other = q.enqueue(conn, "scope:c25-b", JobKind.EPISODE_INDEX, {"n": 9})

    # Priority order inside the saturated scope: privacy first.
    leased = q.lease("scope:c25-a", list(JobKind), owner="w", limit=1)
    assert leased[0]["job_id"] == priv
    assert leased[0]["lane"] == "privacy_control"
    # The other scope's work leases normally.
    leased_b = q.lease("scope:c25-b", list(JobKind), owner="w", limit=1)
    assert leased_b[0]["job_id"] == other


# ---------------------------------------------------------------------------
# C26 — unrelated later events never satisfy readiness (V4-14.02/03)
# ---------------------------------------------------------------------------


def test_c26_unrelated_events_do_not_satisfy_readiness(store):
    """C26 — a captured input's readiness is its OWN durable obligation
    DAG (V4-14.02): later unrelated events — a same-scope capture drained
    all the way to readiness — can never satisfy it (V4-14.03). An
    expired ``wait_ready`` deadline returns the honest pending snapshot,
    never an error and never fabricated readiness."""
    ing = Ingester(
        store, _capture_cfg(require_review=False), encoder=_FakeEncoder()
    )
    engine = ing.readiness_engine()

    rec_x = ing.ingest(_env(SCOPE, "the unfinished captured input"))
    src_x = rec_x.accepted[0]
    rid_x = ingest_receipt_id(src_x, 1)

    # A stuck worker holds X's only pipeline job — X's DAG is genuinely
    # in flight while unrelated work proceeds around it.
    held = ing.jobs.lease(SCOPE, [JobKind.HARVEST], owner="w-stuck", limit=1)
    assert held and held[0]["input_refs"]["source_id"] == src_x

    # Unrelated later events: a second capture in the same scope, drained
    # to completion through the public pipeline.
    rec_y = ing.ingest(_env(SCOPE, "an unrelated later event"))
    rid_y = ingest_receipt_id(rec_y.accepted[0], 1)
    done = ing.run_pending(scope=SCOPE, owner="w-live")
    assert done >= 2  # Y's harvest + admit (+ embed) really ran

    # Y's own committed work settled Y's DAG — the unrelated events DID
    # happen and DID reach readiness; nothing was suppressed globally.
    snap_y = engine.receipt_state(rid_y)
    assert snap_y["complete"] is True
    assert snap_y["pending"] == []
    assert snap_y["failed"] == []
    assert snap_y["ready"] is True

    # X is untouched: durable acceptance committed at capture, but every
    # stage the held job still owes remains pending — Y's settled DAG
    # contributed nothing to X.
    snap_x = engine.receipt_state(rid_x)
    assert snap_x["ready"] is False
    assert snap_x["complete"] is False
    assert snap_x["states"]["accepted"]["state"] == "succeeded"
    for cap in ("screened", "lexical_ready", "derived_ready"):
        assert cap in snap_x["pending"]
        assert snap_x["states"][cap]["state"] == "pending"

    # A deadline-expired wait reports the honest pending snapshot.
    wait = engine.wait_ready(rid_x, deadline_us=now_us())
    assert wait["ready"] is False
    assert wait.get("deadline_exceeded") is True
    assert wait["pending"]

    # Readiness answers stay scope-bound: X's receipt queried under a
    # foreign scope is indistinguishable from unknown (§9).
    with pytest.raises(VerbatimError) as ei:
        engine.receipt_state(rid_x, scope_id="scope:foreign")
    assert ei.value.code is ErrorCode.NOT_FOUND_OR_FORBIDDEN


# ---------------------------------------------------------------------------
# C27 — drain re-checks earlier lanes; no premature completion (V4-14.05/06)
# ---------------------------------------------------------------------------


def test_c27_drain_rechecks_earlier_lanes_for_followups(store, monkeypatch):
    """C27 — a job that enqueues follow-up work into an EARLIER-priority
    lane mid-drain must see that work reconsidered before the drain
    finishes: lane order is re-evaluated per job, and the drain's report
    never claims completion while work remains (V4-14.05/06, V4-42.02)."""
    ing = Ingester(store, VerbatimConfig())
    sid = "scope:c27"
    with store.tx() as conn:
        t_id = ing.jobs.enqueue(
            conn, sid, JobKind.EPISODE_INDEX, {"tag": "trigger"}
        )
        l_id = ing.jobs.enqueue(
            conn, sid, JobKind.EPISODE_INDEX, {"tag": "later"}
        )

    executed: list[tuple[str, Any]] = []

    def spy_execute(job: dict[str, Any], owner: str) -> None:
        executed.append((job["kind"], job.get("lane"), job["job_id"]))
        if job["job_id"] == t_id:
            # Follow-up lands on the privacy_control lane — higher
            # priority than the ordinary job still queued behind us.
            with store.tx() as conn:
                ing.jobs.enqueue(
                    conn, sid, JobKind.QUARANTINE_REVIEW,
                    {"tag": "followup"},
                )

    monkeypatch.setattr(ing, "_execute", spy_execute)
    report = ing.drain_report(owner="w-c27")

    kinds = [k for k, _ln, _jid in executed]
    assert kinds == [
        JobKind.EPISODE_INDEX.value,
        JobKind.QUARANTINE_REVIEW.value,
        JobKind.EPISODE_INDEX.value,
    ]
    # The drain report is honest: it counted every job it actually ran —
    # and left nothing pending rather than declaring completion early
    # (V4-14.05/06).
    assert report["processed"] == 3
    assert report["succeeded"] == 3
    assert report["failed"] == 0
    assert report["still_pending"] == 0
    assert report["complete"] is True
    assert _table_count(
        store, "jobs", "state IN ('queued','retry_wait','leased')"
    ) == 0
    assert _job_row(store, l_id)["state"] == "succeeded"


# ---------------------------------------------------------------------------
# C28 — writer contention bounded by the admitted deadline (V4-40.02/03/10)
# ---------------------------------------------------------------------------


def test_c28_writer_contention_bounded_by_deadline(tmp_path):
    """C28 — a foreground request's admitted deadline bounds writer
    admission: in-process lock contention → retryable BACKPRESSURE inside
    the budget, a passed deadline fails immediately, and a cross-process
    busy writer → retryable STORE_BUSY inside the budget (V4-40.02/03/10)."""
    path = str(tmp_path / "c28.db")
    store = Store.create(path)
    try:
        # In-process writer contention: the lock is held, the request's
        # budget is 50ms — the wait must end in BACKPRESSURE, not a hang.
        assert store._write_lock.acquire(timeout=1)
        try:
            t0 = time.monotonic()
            with pytest.raises(VerbatimError) as ei:
                with store.tx(budget_ms=50):
                    pass
            elapsed = time.monotonic() - t0
            assert ei.value.code is ErrorCode.BACKPRESSURE
            assert ei.value.retryable is True
            assert elapsed < 5.0
        finally:
            store._write_lock.release()

        # An already-passed absolute deadline never even touches the lock.
        with pytest.raises(VerbatimError) as ei:
            with store.tx(deadline_us=now_us() - 1):
                pass
        assert ei.value.code is ErrorCode.BACKPRESSURE

        # Cross-process writer: another connection holds the write txn —
        # the request's busy budget is bounded by its remaining deadline.
        other = sqlite3.connect(path)
        other.execute("PRAGMA busy_timeout = 0")
        other.execute("BEGIN IMMEDIATE")
        try:
            t0 = time.monotonic()
            with pytest.raises(VerbatimError) as ei:
                with store.tx(budget_ms=40):
                    pass
            assert ei.value.code is ErrorCode.STORE_BUSY
            assert ei.value.retryable is True
            assert time.monotonic() - t0 < 5.0
        finally:
            other.rollback()
            other.close()

        # Once contention clears, the same bounded write proceeds.
        with store.tx(budget_ms=1000) as conn:
            conn.execute(
                "INSERT INTO scopes (scope_id, profile_id, visibility)"
                " VALUES ('scope:c28', 'prof', 'owner')"
            )
        diag = store.diagnostics()
        assert diag["write_tx"]["admission_timeouts"] >= 1
        assert diag["write_tx"]["busy_errors"] >= 1
    finally:
        store.close()


# ---------------------------------------------------------------------------
# C29 — missing handler/capability is a durable typed failure (V4-40.08)
# ---------------------------------------------------------------------------


def test_c29_missing_handler_fails_closed_durably(store):
    """C29 — a job whose kind has no provisioned handler must become a
    durable, typed failure — never a silent success and never a retry
    storm (V4-40.08, §03 honest-unavailable). Covers a declared-but-
    unimplemented kind, a handler module that is absent, and an unknown
    kind string refused at the door."""
    ing = Ingester(store, VerbatimConfig())
    sid = "scope:c29"
    with store.tx() as conn:
        # Unknown kinds are typed errors BEFORE a row exists.
        with pytest.raises(VerbatimError) as ei:
            ing.jobs.enqueue(conn, sid, "teleport_to_production", {})
        assert ei.value.code is ErrorCode.VALIDATION

        j_compare = ing.jobs.enqueue(conn, sid, JobKind.COMPARE, {})
        j_projection = ing.jobs.enqueue(conn, sid, JobKind.PROJECTION_SYNC, {})
        # CONNECTOR_PULL is provisioned (V4-48 — verbatim/connectors), so
        # an empty-refs pull job fails typed VALIDATION, not
        # CAPABILITY_UNAVAILABLE — still durable, still never a no-op.
        j_connector = ing.jobs.enqueue(conn, sid, JobKind.CONNECTOR_PULL, {})

    done = ing.run_pending(owner="w-c29")
    assert done == 3

    for jid in (j_compare, j_projection):
        row = _job_row(store, jid)
        assert row["state"] == "failed"
        assert row["error_code"] == ErrorCode.CAPABILITY_UNAVAILABLE.value
    row = _job_row(store, j_connector)
    assert row["state"] == "failed"
    assert row["error_code"] == ErrorCode.VALIDATION.value

    # The failure is durable and typed in the event log too.
    with store.read() as conn:
        for jid, code in (
            (j_compare, ErrorCode.CAPABILITY_UNAVAILABLE.value),
            (j_projection, ErrorCode.CAPABILITY_UNAVAILABLE.value),
            (j_connector, ErrorCode.VALIDATION.value),
        ):
            ev = conn.execute(
                "SELECT state, error_code FROM job_events"
                " WHERE job_id = ? AND state = 'failed'",
                (jid,),
            ).fetchone()
            assert ev == ("failed", code)

    # A second drain changes nothing — terminal failures stay failed.
    assert ing.run_pending(owner="w-c29") == 0
    assert _job_row(store, j_compare)["state"] == "failed"


# ---------------------------------------------------------------------------
# C30 — unauthorized ≡ absent: errors, metadata, handles (V4-08.06/10.04)
# ---------------------------------------------------------------------------


def test_c30_denied_and_absent_are_indistinguishable(store):
    """C30 — requesting a private object and requesting a nonexistent one
    produce the SAME denial: identical error code and message, identical
    denied-lease shape, identical metadata withholding — no existence
    oracle anywhere on the public surface (V4-08.06, V4-10.04)."""
    sid_a, sid_b = "scope:c30-a", "scope:c30-b"
    private_payload = b"private payload bytes"
    with store.tx() as conn:
        _seed_governance(conn, sid_a, sid_b)
        _grant(conn, sid_a, "human:alice", {"read", "quote"})
        # A real object exists in scope:c30-b; alice holds nothing there.
        conn.execute(
            "INSERT INTO sources"
            "(source_id, origin, external_id, source_kind, scope_id,"
            " speaker_id, created_us)"
            " VALUES ('src-private', 'test', NULL, 'user_message', ?, NULL, 1)",
            (sid_b,),
        )
        conn.execute(
            "INSERT INTO source_revisions"
            "(source_id, revision, payload, payload_hmac, event_us,"
            " captured_us, timezone, provenance, metadata_json)"
            " VALUES ('src-private', 1, ?, ?, 1, 1, 'UTC', 'direct_user', '{}')",
            (private_payload, store.hmac(private_payload)),
        )

    kernel = Kernel(store)
    alice = CallerV3(principal_id="human:alice")

    # --- resolve_access: private-object ref vs missing-object ref → the
    # same denied lease shape (no refs, no epochs, no detail).
    with store.read() as conn:
        denied_private = kernel.resolve_access(
            conn, alice, Verb.QUOTE, "recall", [sid_a],
            [("source", "src-private", 1)],
        )
        denied_missing = kernel.resolve_access(
            conn, alice, Verb.QUOTE, "recall", [sid_a],
            [("source", "src-missing", 1)],
        )
    assert denied_private.denied is True
    assert denied_missing.denied is True
    assert denied_private.object_refs == denied_missing.object_refs == ()
    assert denied_private.epoch_vector == denied_missing.epoch_vector == {}

    # --- read_verified under a LIVE lease on scope:a: a locator into the
    # private scope and a locator to nothing fail identically.
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, alice, Verb.QUOTE, "recall", [sid_a]
        )
    assert not lease.denied
    with store.read() as conn:
        e_private = _err_of(
            lambda: kernel.read_verified(
                conn, lease,
                [EvidenceLocator(
                    object_id="src-private", revision=1,
                    start_byte=0, end_byte=len(private_payload),
                )],
            )
        )
        e_missing = _err_of(
            lambda: kernel.read_verified(
                conn, lease,
                [EvidenceLocator(
                    object_id="src-missing", revision=1,
                    start_byte=0, end_byte=8,
                )],
            )
        )
    assert e_private == e_missing == (
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "not found or unauthorized"
    )

    # --- metadata review: denied leases withhold counts/identifiers
    # identically whether the underlying object was private or absent.
    with store.read() as conn:
        m_private = _err_of(
            lambda: kernel.review_metadata(
                conn, denied_private, {"count": 5, "title": "x"}
            )
        )
        m_missing = _err_of(
            lambda: kernel.review_metadata(
                conn, denied_missing, {"count": 5, "title": "x"}
            )
        )
    assert m_private == m_missing == (
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "not found or unauthorized"
    )

    # --- delivery-permit handles: a consumed permit is indistinguishable
    # from one that never existed.
    with store.tx() as conn:
        live = kernel.resolve_access(conn, alice, Verb.QUOTE, "recall", [sid_a])
        permit = kernel.seal_delivery(conn, live, b"pack")
        kernel.mark_delivered(conn, permit.permit_id)
    with store.read() as conn:
        p_absent = _err_of(
            lambda: kernel.verify_delivery(conn, "dpermit:never-existed")
        )
        p_consumed = _err_of(
            lambda: kernel.verify_delivery(conn, permit.permit_id)
        )
    assert p_absent == p_consumed == (
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "not found or unauthorized"
    )


# ---------------------------------------------------------------------------
# C31/C32 — metamorphic retrieval bounds (V4-28.02/28.03/28.10)
# ---------------------------------------------------------------------------


def _seed_scope(conn, scope_id: str) -> None:
    conn.execute(
        "INSERT INTO scopes(scope_id,profile_id,principal_id,workspace_id,"
        "conversation_id,visibility,acl_revision) VALUES(?,?,?,?,?,?,0)",
        (scope_id, "prof", "p1", "ws", "c1", "conversation"),
    )


def _seed_auth(conn, scope_id: str, pid: str = "human:alice") -> None:
    seed_purposes(conn)
    register_principal(conn, kind="human", principal_id=pid)
    create_grant(
        conn, scope_id=scope_id, principal_id=pid, verbs={"read"},
        issuer_id=pid, purposes=["recall"],
    )


def _seed_claim(
    conn,
    store: Store,
    claim_id: str,
    scope_id: str,
    source_id: str,
    span_id: str,
    text: str,
    gen: int,
    state: str = "active",
) -> None:
    """Minimal claim corpus row set (mirrors tests/retrieval/v3 helpers,
    keyed by the store's own HMAC)."""
    payload = text.encode("utf-8")
    conn.execute(
        "INSERT INTO sources(source_id,origin,source_kind,scope_id,"
        "speaker_id,created_us) VALUES(?,?,?,?,?,1)",
        (source_id, "test", "user_message", scope_id, "u1"),
    )
    conn.execute(
        "INSERT INTO source_revisions(source_id,revision,payload,"
        "payload_hmac,event_us,captured_us,timezone,provenance,metadata_json)"
        " VALUES(?,1,?,?,1,1,'UTC','direct_user','{}')",
        (source_id, payload, store.hmac(payload)),
    )
    conn.execute(
        "INSERT INTO spans(span_id,source_id,revision,start_byte,end_byte,"
        "excerpt_hmac,harvester_version) VALUES(?,?,?,?,?,?,'t')",
        (span_id, source_id, 1, 0, len(payload), store.hmac(payload)),
    )
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,NULL,NULL,1,1)",
        (claim_id, scope_id),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,condition_json,"
        "recorded_from,recorded_until,perspective_id,freshness)"
        " VALUES(?,1,?,NULL,1,NULL,NULL,NULL)",
        (claim_id, state),
    )
    conn.execute(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role,family_id) VALUES(?,1,?,'primary',NULL)",
        (claim_id, span_id),
    )
    cur = conn.execute(
        "INSERT INTO fts_rows(claim_id,claim_revision,scope_id,"
        "projection_generation) VALUES(?,1,?,?)",
        (claim_id, scope_id, gen),
    )
    conn.execute(
        "INSERT INTO facts_fts(fts_row_id,text) VALUES(?,?)",
        (cur.lastrowid, text),
    )


def _hold_claim(conn, claim_id: str, scope_id: str) -> None:
    conn.execute(
        "INSERT INTO quarantine(object_kind,object_id,revision,"
        "scope_id,state,opened_event) VALUES('claim',?,1,?,'pending',1)",
        (claim_id, scope_id),
    )


def _recall_request(query: str, scope_id: str = "sA", **kw) -> RecallRequestV3:
    kw.setdefault("purpose", "recall")
    return RecallRequestV3(
        query=query, scope_id=scope_id, caller_id="human:alice", **kw
    )


def _result_items(result) -> list[tuple[str, str]]:
    return [
        (i.handle.object_id, i.text) for p in result.packs for i in p.items
    ]


def test_c31_held_matches_cannot_hide_eligible_claim(store):
    """C31 — V4-28.02: eligibility precedes bounded rank selection — a
    fetch window full of ineligible (quarantine-held) lexical matches must
    page past them until the eligible bound; >160 held matches cannot
    hide the one eligible claim."""
    held_n = 170  # > the historical fixed 160-row oversample window
    with store.tx() as conn:
        _seed_scope(conn, "sA")
        _seed_auth(conn, "sA")
        gen = store.projection_generation()
        for i in range(held_n):
            cid = f"held{i:04d}"
            _seed_claim(
                conn, store, cid, "sA", f"srcH{i}", f"spH{i}",
                f"guardterm held variant {i}", gen,
            )
            _hold_claim(conn, cid, "sA")
        # The eligible match is deliberately bm25-worst for the term.
        padded = "guardterm " + " ".join(f"filler{i}" for i in range(64))
        _seed_claim(
            conn, store, "eligible-1", "sA", "srcE", "spE", padded, gen
        )

    res = recall_v3(store, _recall_request("guardterm"))
    ids = {i.handle.object_id for p in res.packs for i in p.items}
    assert "eligible-1" in ids
    assert not any(i.startswith("held") for i in ids)


def test_c32_private_docs_cannot_reorder_public_results(store):
    """C32 — V4-28.03/28.10 (metamorphic, §58.03): a private corpus of
    alpha-heavy documents landing in a scope the caller cannot read must
    leave the public-scope result set byte-for-byte unchanged — same
    items, same order, same abstention."""
    with store.tx() as conn:
        _seed_scope(conn, "sA")
        _seed_auth(conn, "sA")
        _seed_scope(conn, "sB")  # private scope — alice holds no grant
        gen = store.projection_generation()
        _seed_claim(
            conn, store, "pub-alpha-1", "sA", "srcP1", "spP1",
            "alpha beta public report", gen,
        )
        _seed_claim(
            conn, store, "pub-alpha-2", "sA", "srcP2", "spP2",
            "alpha beta public minutes with extra context", gen,
        )

    req = _recall_request("alpha beta")
    before = recall_v3(store, req)
    items_before = _result_items(before)
    assert {oid for oid, _t in items_before} == {"pub-alpha-1", "pub-alpha-2"}

    # The private alpha flood lands — entirely outside the caller's grant.
    with store.tx() as conn:
        for i in range(80):
            _seed_claim(
                conn, store, f"priv{i:03d}", "sB", f"srcQ{i}", f"spQ{i}",
                f"alpha alpha alpha private flood document {i}", gen,
            )

    after = recall_v3(store, req)
    items_after = _result_items(after)
    assert items_after == items_before
    assert after.omitted == before.omitted
    assert after.abstained == before.abstained
