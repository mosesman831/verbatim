"""Capture-SDK conformance tests (SPEC_V3 §13, §11.11, §12, §14, §20.08).

Runs the real SDK over a real ``Store.create`` (the tests/conftest shim is
v1-DDL only, so fixtures here mirror tests/governance — a disposable
SQLite file per test). The covered contract:

- session lifecycle: begin → capture → end → ``episode_build`` enqueued
  in the same transaction (§20.08);
- the two gates on every write: the ``ingest`` verb grant
  (``NOT_FOUND_OR_UNAUTHORIZED`` — existence never leaks, §09.09) and the
  §11.11 capture authorization (``CONSENT_REQUIRED``, §12.10);
- honest provenance: agent-submitted content is never human testimony
  (§13.11); host-attested envelopes carry their declared trust;
- trajectory + checker-attested outcome capture (§12.02–§12.03);
- idempotent session close and revocation/expiry denial at close
  (§13.07, §48.12).
"""

from __future__ import annotations

import pytest

from verbatim.config import config_from_mapping
from verbatim.core.types import ErrorCode, JobKind, VerbatimError
from verbatim.core.types_v3 import EnvelopeKind, TrustClass
from verbatim.governance import create_grant, revoke_capture_authorization
from verbatim.sdk import CaptureClient, EnvelopeBuilder
from verbatim.storage import repos_v3
from verbatim.storage.store import Store

PRINCIPAL = "agent:w11"
ISSUER = "ops:test"
SCOPE = "scope:sdk-test"


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "sdk.db"))
    yield s
    s.close()


@pytest.fixture
def client(store):
    return CaptureClient(store, principal_id=PRINCIPAL, host_id="sdk-test")


@pytest.fixture
def capture_client(store):
    """A client whose config admits the captured kinds — the default
    ``capture.enabled=False`` honestly rejects claims at admit time
    (jobs still succeed; nothing materializes)."""
    cfg = config_from_mapping(
        {"capture": {"enabled": True, "user_messages": True}}
    )
    return CaptureClient(
        store, config=cfg, principal_id=PRINCIPAL, host_id="sdk-test"
    )


def _grant(store, scope=SCOPE, principal=PRINCIPAL, verbs=("ingest",)):
    """Host-side provisioning: a live ingest grant for the principal."""
    with store.tx() as conn:
        return create_grant(
            conn,
            scope_id=scope,
            principal_id=principal,
            verbs=list(verbs),
            issuer_id=ISSUER,
        )


def _ready(client, store, scope=SCOPE):
    """grant + authorization + open session → (scope, auth_id, session_id)."""
    _grant(store, scope)
    aid = client.authorize(scope, ISSUER, principal_id=PRINCIPAL)
    sid = client.begin_session(
        PRINCIPAL, "sdk-test", metadata={"scope_id": scope, "task_id": "task:t1"}
    )
    return scope, aid, sid


def _envelope_row(store, source_id):
    with store.read() as conn:
        rows = repos_v3.query(
            conn, "source_envelopes", {"source_id": source_id}, order="revision DESC"
        )
    return rows[0] if rows else None


def _jobs(store, kind=None):
    sql = "SELECT job_id, scope_id, kind, state, input_refs_json FROM jobs"
    params = ()
    if kind is not None:
        sql += " WHERE kind = ?"
        params = (kind,)
    with store.read() as conn:
        cols = [d[0] for d in conn.execute(sql, params).description]
        return [dict(zip(cols, r)) for r in conn.execute(sql, params).fetchall()]


def _trajectory(store, trajectory_id):
    with store.read() as conn:
        return repos_v3.get(conn, "trajectories", {"trajectory_id": trajectory_id})


def _steps(store, trajectory_id):
    with store.read() as conn:
        return repos_v3.query(
            conn, "trajectory_steps", {"trajectory_id": trajectory_id}, order="ord"
        )


# ---------------------------------------------------------------------
# sessions + authorization
# ---------------------------------------------------------------------


def test_begin_session_binds_scope_and_opens_trajectory(client, store):
    scope, aid, sid = _ready(client, store)
    state = client.session_state(sid)
    assert state["session_id"] == sid
    assert state["scope_id"] == scope
    assert state["closed"] is False
    assert state["trajectory_id"] is not None
    traj = _trajectory(store, state["trajectory_id"])
    assert traj is not None
    assert traj["scope_id"] == scope
    assert traj["host_id"] == "sdk-test"
    assert traj["completed_event"] is None


def test_begin_session_without_scope_defers_trajectory(client, store):
    sid = client.begin_session(PRINCIPAL, "sdk-test")
    state = client.session_state(sid)
    assert state["scope_id"] is None
    assert state["trajectory_id"] is None


def test_authorize_returns_authorization_row(client, store):
    _grant(store)
    aid = client.authorize(SCOPE, ISSUER, principal_id=PRINCIPAL)
    with store.read() as conn:
        row = repos_v3.get(
            conn, "capture_authorizations", {"authorization_id": aid}
        )
    assert row is not None
    assert row["issuer_id"] == ISSUER
    assert row["principal_id"] == PRINCIPAL
    assert SCOPE in (repos_v3.json_field(row, "scope_ids_json") or [])


# ---------------------------------------------------------------------
# denial surface (§09.09, §11.11, §13.01)
# ---------------------------------------------------------------------


def test_submit_source_without_ingest_grant_denied(client, store):
    """§13.01: no verb grant → the public indistinguishable denial."""
    aid = client.authorize(SCOPE, ISSUER, principal_id=PRINCIPAL)
    sid = client.begin_session(PRINCIPAL, "sdk-test")  # no scope bound yet
    with pytest.raises(VerbatimError) as exc:
        client.submit_source(sid, SCOPE, "hi", declared_type="agent_note")
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_submit_source_without_capture_auth_is_consent_required(client, store):
    """§12.10: verb grant but no §11.11 authorization → CONSENT_REQUIRED
    (consent is a separate record, never implied by permission)."""
    _grant(store)
    sid = client.begin_session(PRINCIPAL, "sdk-test")
    with pytest.raises(VerbatimError) as exc:
        client.submit_source(sid, SCOPE, "hi", declared_type="agent_note")
    assert exc.value.code == ErrorCode.CONSENT_REQUIRED


def test_unknown_session_denied_indistinguishably(client, store):
    with pytest.raises(VerbatimError) as exc:
        client.submit_source("sess:nope", SCOPE, "x", declared_type="agent_note")
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    with pytest.raises(VerbatimError) as exc2:
        client.end_session("sess:nope")
    assert exc2.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_kind_narrowed_authorization_denies_other_kinds(client, store):
    _grant(store)
    client.authorize(
        SCOPE, ISSUER, kinds=["agent_note"], principal_id=PRINCIPAL
    )
    sid = client.begin_session(
        PRINCIPAL, "sdk-test", metadata={"scope_id": SCOPE}
    )
    client.submit_source(sid, SCOPE, "note", declared_type="agent_note")
    with pytest.raises(VerbatimError) as exc:
        client.capture_envelope(
            sid,
            EnvelopeBuilder()
            .kind("tool_result")
            .scope(SCOPE)
            .actor(PRINCIPAL)
            .content("obs")
            .trust("host_observed")
            .build(),
        )
    assert exc.value.code == ErrorCode.CONSENT_REQUIRED


def test_expired_authorization_denies(client, store):
    _grant(store)
    aid = client.authorize(SCOPE, ISSUER, principal_id=PRINCIPAL)
    # Backdate expiry — deterministic (wall-clock races never flake).
    with store.tx() as conn:
        repos_v3.update(
            conn,
            "capture_authorizations",
            {"expires_us": 1},
            {"authorization_id": aid},
        )
    sid = client.begin_session(
        PRINCIPAL, "sdk-test", metadata={"scope_id": SCOPE}
    )
    with pytest.raises(VerbatimError) as exc:
        client.submit_source(sid, SCOPE, "x", declared_type="agent_note")
    assert exc.value.code == ErrorCode.CONSENT_REQUIRED


# ---------------------------------------------------------------------
# provenance (§13.11, §14)
# ---------------------------------------------------------------------


def test_submit_source_records_agent_provenance(client, store):
    scope, aid, sid = _ready(client, store)
    source_id = client.submit_source(
        sid, scope, "the migration is reversible", declared_type="agent_note"
    )
    row = _envelope_row(store, source_id)
    assert row["envelope_kind"] == "agent_note"
    assert row["trust_class"] == TrustClass.AGENT_GENERATED.value
    assert row["actor_principal"] == PRINCIPAL
    assert row["session_id"] == sid
    assert row["task_id"] == "task:t1"
    # §11.11: the resolved authorization is bound as capture_proof.
    assert row["capture_proof"] == aid


def test_submit_source_rejects_human_testimony(client, store):
    scope, aid, sid = _ready(client, store)
    with pytest.raises(VerbatimError) as exc:
        client.submit_source(sid, scope, "i said this", declared_type="user_message")
    assert exc.value.code == ErrorCode.VALIDATION


def test_submit_source_assistant_kind_never_upgraded(client, store):
    scope, aid, sid = _ready(client, store)
    source_id = client.submit_source(
        sid, scope, "draft answer", declared_type="assistant_message"
    )
    row = _envelope_row(store, source_id)
    assert row["trust_class"] == TrustClass.AGENT_GENERATED.value


def test_capture_envelope_host_attested_trust(client, store):
    """§13.03: the host attests actual authorship — a user turn persists
    as principal_direct through the envelope path (the submit path could
    never carry this)."""
    scope, aid, sid = _ready(client, store)
    receipt = client.capture_envelope(
        sid,
        EnvelopeBuilder()
        .kind("user_message")
        .scope(scope)
        .actor("user:alice")
        .perspective({"asserter": "user:alice", "observer": "sdk-test"})
        .content("deploy on friday")
        .trust("principal_direct")
        .build(),
    )
    row = _envelope_row(store, receipt.source_id)
    assert row["trust_class"] == TrustClass.PRINCIPAL_DIRECT.value
    assert row["actor_principal"] == "user:alice"


def test_submit_source_invokes_screening(client, store):
    """§14.01: every persisted envelope carries a security label in the
    same transaction (taint dimensions stay independent)."""
    scope, aid, sid = _ready(client, store)
    source_id = client.submit_source(
        sid, scope, "remember to rotate keys", declared_type="agent_note"
    )
    with store.read() as conn:
        labels = repos_v3.query(conn, "security_labels", {"scope_id": scope})
    assert labels, "no security label persisted for the envelope"
    assert labels[0]["source_trust"] == TrustClass.AGENT_GENERATED.value


# ---------------------------------------------------------------------
# trajectory steps + outcomes (§12.02–§12.03)
# ---------------------------------------------------------------------


def test_record_step_appends_dense_ords(client, store):
    scope, aid, sid = _ready(client, store)
    s1 = client.submit_source(sid, scope, "first", declared_type="agent_note")
    s2 = client.submit_source(sid, scope, "second", declared_type="agent_note")
    st1 = client.record_step(sid, s1, {})
    st2 = client.record_step(sid, s2, {})
    state = client.session_state(sid)
    steps = _steps(store, state["trajectory_id"])
    assert [s["ord"] for s in steps] == [0, 1]
    assert {s["step_id"] for s in steps} == {st1, st2}


def test_record_step_sparse_ord_rejected(client, store):
    scope, aid, sid = _ready(client, store)
    s1 = client.submit_source(sid, scope, "first", declared_type="agent_note")
    with pytest.raises(VerbatimError):
        client.record_step(sid, s1, {"ord": 7})  # gap — not dense/monotonic


def test_record_outcome_persists_checker_evidence(client, store):
    scope, aid, sid = _ready(client, store)
    s1 = client.submit_source(sid, scope, "fix applied", declared_type="agent_note")
    outcome_id = client.record_outcome(
        sid,
        s1,
        {"outcome": "success"},
        receipts=[
            {
                "checker_id": "pytest",
                "checker_version": "8.4.2",
                "invocation_id": "inv-1",
                "selected_tests": ["tests/test_x.py"],
                "completed": True,
                "exit_code": 0,
                "result_json": {"passed": 3},
            }
        ],
    )
    with store.read() as conn:
        env = repos_v3.get(conn, "source_envelopes", {"envelope_id": outcome_id})
    assert env is not None
    assert env["envelope_kind"] == "test_result"
    meta = repos_v3.json_field(env, "metadata_json") or {}
    assert meta.get("verifies_source_id") == s1
    # The outcome lands on the trajectory as an observation-only step.
    state = client.session_state(sid)
    with store.read() as conn:
        obs = repos_v3.query(
            conn,
            "step_observations",
            {"envelope_id": outcome_id},
        )
        steps = repos_v3.query(
            conn, "trajectory_steps", {"trajectory_id": state["trajectory_id"]}
        )
    step_ids = {s["step_id"] for s in steps}
    assert obs and obs[0]["step_id"] in step_ids


def test_record_outcome_unknown_source_denied(client, store):
    scope, aid, sid = _ready(client, store)
    with pytest.raises(VerbatimError) as exc:
        client.record_outcome(sid, "src:nope", {"outcome": "success"})
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


# ---------------------------------------------------------------------
# session end → episode pipeline (§20.08)
# ---------------------------------------------------------------------


def test_end_session_completes_trajectory_and_enqueues(client, store):
    scope, aid, sid = _ready(client, store)
    s1 = client.submit_source(sid, scope, "work done", declared_type="agent_note")
    client.record_step(sid, s1, {})
    summary = client.end_session(sid, status="complete")
    assert summary["closed"] is True
    assert summary["status"] == "complete"
    assert summary["completed_event"] is not None
    assert summary["episode_build_job_id"] is not None

    traj = _trajectory(store, summary["trajectory_id"])
    assert traj["completed_event"] is not None
    meta = repos_v3.json_field(traj, "metadata_json") or {}
    # Lifecycle status is journaled — it is NOT an episode outcome.
    assert meta.get("session_status") == "complete"

    jobs = _jobs(store, "episode_build")
    assert len(jobs) == 1
    import json

    refs = json.loads(jobs[0]["input_refs_json"])
    assert refs["trajectory_id"] == summary["trajectory_id"]


def test_end_session_idempotent_single_job(client, store):
    scope, aid, sid = _ready(client, store)
    s1 = client.submit_source(sid, scope, "x", declared_type="agent_note")
    client.record_step(sid, s1, {})
    first = client.end_session(sid)
    second = client.end_session(sid)
    assert first["trajectory_id"] == second["trajectory_id"]
    assert second["closed"] is True
    assert len(_jobs(store, "episode_build")) == 1


def test_end_session_empty_session_no_job(client, store):
    sid = client.begin_session(PRINCIPAL, "sdk-test")
    summary = client.end_session(sid)
    assert summary["closed"] is True
    assert "episode_build_job_id" not in summary
    assert _jobs(store) == []


# ---------------------------------------------------------------------
# drain_pending — the host's explicit execution seam (§40)
# ---------------------------------------------------------------------


def _capture_user_turn(client, scope, sid):
    return client.capture_envelope(
        sid,
        EnvelopeBuilder()
        .kind("user_message")
        .scope(scope)
        .actor("user:alice")
        .perspective({"asserter": "user:alice", "observer": "sdk-test"})
        .content("the deploy runbook lives at docs/deploy.md")
        .trust("principal_direct")
        .build(),
    )


def test_drain_pending_executes_queued_pipeline(capture_client, store):
    """Captures only *enqueue* — ``drain_pending`` is the host-side
    execution seam: harvest→admit runs inline and claims materialize."""
    scope, aid, sid = _ready(capture_client, store)
    _capture_user_turn(capture_client, scope, sid)
    queued = [j for j in _jobs(store, "harvest") if j["state"] == "queued"]
    assert queued, "no harvest job enqueued for the capture"

    drained = capture_client.drain_pending()
    assert drained >= 2  # harvest + admit chain
    states = {j["kind"]: j["state"] for j in _jobs(store)}
    assert states.get("harvest") == "succeeded"
    assert states.get("admit") == "succeeded"
    with store.read() as conn:
        claims = conn.execute(
            "SELECT claim_id FROM claim_revisions"
        ).fetchall()
    assert claims, "drained pipeline materialized no claims"


def test_drain_pending_kinds_and_scope_filters(capture_client, store):
    """``kinds`` narrows the drain; ``scope`` binds it to one partition;
    an empty queue drains zero."""
    assert capture_client.drain_pending() == 0
    scope, aid, sid = _ready(capture_client, store)
    _capture_user_turn(capture_client, scope, sid)

    drained = capture_client.drain_pending(kinds=[JobKind.HARVEST])
    assert drained == 1
    assert _jobs(store, "harvest")[0]["state"] == "succeeded"
    # The admit chain stays queued until the host drains it.
    queued = [j for j in _jobs(store, "admit") if j["state"] == "queued"]
    assert queued
    # A foreign scope sees nothing; the bound scope drains the rest.
    assert capture_client.drain_pending("scope:other") == 0
    assert capture_client.drain_pending(scope) >= 1
    assert _jobs(store, "admit")[0]["state"] == "succeeded"


def test_drain_pending_runs_episode_build(client, store):
    """Session close enqueues ``episode_build`` atomically (§20.08); the
    same host drain executes it — no separate worker needed."""
    scope, aid, sid = _ready(client, store)
    s1 = client.submit_source(sid, scope, "work done", declared_type="agent_note")
    client.record_step(sid, s1, {})
    client.end_session(sid)
    assert len(_jobs(store, "episode_build")) == 1
    drained = client.drain_pending()
    assert drained >= 1
    assert _jobs(store, "episode_build")[0]["state"] == "succeeded"


def test_end_session_denies_after_authorization_revoked(client, store):
    """§13.07/§48.12: consent is rechecked at close — a mid-session
    revocation denies the close exactly like any other capture event."""
    scope, aid, sid = _ready(client, store)
    client.submit_source(sid, scope, "x", declared_type="agent_note")
    with store.tx() as conn:
        assert revoke_capture_authorization(conn, aid) is True
    with pytest.raises(VerbatimError) as exc:
        client.end_session(sid)
    assert exc.value.code == ErrorCode.CONSENT_REQUIRED
    assert client.session_state(sid)["closed"] is False


def test_writes_after_close_denied(client, store):
    scope, aid, sid = _ready(client, store)
    client.end_session(sid)
    with pytest.raises(VerbatimError) as exc:
        client.submit_source(sid, scope, "x", declared_type="agent_note")
    assert exc.value.code == ErrorCode.VALIDATION


# ---------------------------------------------------------------------
# EnvelopeBuilder round-trip (§13.02 event form)
# ---------------------------------------------------------------------


def test_envelope_builder_dict_roundtrip():
    env = (
        EnvelopeBuilder()
        .kind("tool_result")
        .scope("scope:rt")
        .actor("agent:a")
        .content("payload-bytes")
        .trust("host_observed")
        .host("hermes")
        .session("sess:1")
        .task("task:1")
        .event_us(123)
        .metadata(k="v")
        .build()
    )
    event = EnvelopeBuilder.from_dict(
        EnvelopeBuilder().kind("tool_result").scope("scope:rt")
        .actor("agent:a").content("payload-bytes").trust("host_observed")
        .host("hermes").session("sess:1").task("task:1").event_us(123)
        .metadata(k="v").to_dict()
    ).build()
    assert event.kind == env.kind
    assert event.scope_id == env.scope_id
    assert event.actor_principal == env.actor_principal
    assert event.content == env.content
    assert event.trust_class == env.trust_class
    assert event.session_id == env.session_id
    assert event.metadata["k"] == "v"
