"""Evidence-plane tests: envelope ingest, receipts, trajectories, replay.

Covers SPEC_V3 §11.11 (capture authorization), §12 (envelope ingest,
idempotence, event/receipt time split), §13 (agent attribution), §14
(trust classes), §43 (replay manifests, sandbox-only), §46 (typed errors),
and the v2-compat disposition (V3-62): the coarse ``sources.source_kind``
keeps v2 read paths valid.

Fixtures use a real ``Store.create`` (schema v3) — the conftest TestStore
shim is DDL_V1-only and cannot run v3 tables.
"""

from __future__ import annotations

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v3 import (
    CaptureAuthorization,
    EnvelopeKind,
    EnvironmentFingerprint,
    Perspective,
    SourceEnvelopeV3,
    StateAnchor,
    TrajectoryRecord,
    TrajectoryStep,
    TrustClass,
)
from verbatim.evidence import (
    add_step,
    anchor,
    build_replay_manifest,
    complete,
    get_trajectory,
    ingest_envelope,
    record_trajectory,
    steps,
    verify_receipt,
)
from verbatim.evidence import detect, envelopes as env_mod
from verbatim.storage import repos_v3
from verbatim.storage.repos import SourcesRepo
from verbatim.storage.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "v3.db"))
    yield s
    s.close()


SCOPE = "scope:evidence"


def _env(kind, *, scope_id=SCOPE, content=b"payload", actor="user-1", **kw):
    kw.setdefault("perspective", Perspective(asserter=actor))
    kw.setdefault("event_us", 1_000)
    kw.setdefault("receipt_us", 1_001)
    kw.setdefault("content", content)
    return SourceEnvelopeV3(
        kind=kind, scope_id=scope_id, actor_principal=actor, **kw
    )


def _auth(kinds, scopes=(SCOPE,), **kw):
    kw.setdefault("authorization_id", "authz-1")
    kw.setdefault("issuer_id", "operator-1")
    kw.setdefault("principal_id", "agent-1")
    kw.setdefault("retention_policy", "task")
    kw.setdefault("policy_revision", "pol-1")
    kw.setdefault("issued_us", 1)
    return CaptureAuthorization(
        allowed_kinds=frozenset(kinds),
        scope_ids=frozenset(scopes),
        **kw,
    )


def _ingest(store, env, *, authorization=None):
    with store.tx() as conn:
        return ingest_envelope(conn, store, env, authorization=authorization)


def _table_counts(store):
    with store.read() as conn:
        names = (
            "scopes", "sources", "source_revisions", "spans",
            "source_envelopes", "perspectives", "events",
        )
        return {
            n: conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0]
            for n in names
        }


# ------------------------------------------------------------ fresh capture


def test_fresh_store_capture_user_message(store):
    r = _ingest(store, _env(EnvelopeKind.USER_MESSAGE, content=b"hello world"))
    assert r.source_id and r.envelope_id and r.receipt_id
    assert r.envelope_kind == "user_message"
    assert r.revision == 1
    assert r.event_seq > 0
    with store.read() as conn:
        src = conn.execute(
            "SELECT * FROM sources WHERE source_id = ?", (r.source_id,)
        ).fetchone()
        assert src is not None
        rev = conn.execute(
            "SELECT payload, event_us, captured_us FROM source_revisions"
            " WHERE source_id = ? AND revision = 1",
            (r.source_id,),
        ).fetchone()
        assert bytes(rev[0]) == b"hello world"
        # event clock vs receipt clock stay distinct (V3-12.11)
        assert rev[1] == 1_000 and rev[2] == 1_001
        sp = conn.execute(
            "SELECT start_byte, end_byte, harvester_version FROM spans"
            " WHERE source_id = ? AND revision = 1",
            (r.source_id,),
        ).fetchone()
        assert (sp[0], sp[1]) == (0, len(b"hello world"))
        env = repos_v3.get(conn, "source_envelopes", {"envelope_id": r.envelope_id})
        assert env["envelope_kind"] == "user_message"
        assert env["actor_principal"] == "user-1"
        assert env["scope_id"] == SCOPE
        assert env["event_us"] == 1_000 and env["receipt_us"] == 1_001
        assert env["trust_class"] == "principal_direct"
        assert env["perspective_id"] is not None


def test_receipt_verifies_against_store(store):
    r = _ingest(store, _env(EnvelopeKind.USER_MESSAGE))
    with store.read() as conn:
        verified = verify_receipt(conn, r)
    assert verified.receipt_id == r.receipt_id
    assert verified.event_seq == r.event_seq


def test_verify_receipt_missing_envelope(store):
    r = _ingest(store, _env(EnvelopeKind.USER_MESSAGE))
    forged = type(r)(
        receipt_id=r.receipt_id, envelope_id="env_missing",
        source_id=r.source_id, revision=1, scope_id=SCOPE,
        envelope_kind="user_message", event_seq=r.event_seq,
        dedup_key=r.dedup_key, accepted_bytes=r.accepted_bytes,
    )
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            verify_receipt(conn, forged)
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_verify_receipt_detects_forged_field(store):
    r = _ingest(store, _env(EnvelopeKind.USER_MESSAGE))
    forged = type(r)(
        receipt_id=r.receipt_id, envelope_id=r.envelope_id,
        source_id=r.source_id, revision=1, scope_id="scope:other",
        envelope_kind="user_message", event_seq=r.event_seq,
        dedup_key=r.dedup_key, accepted_bytes=r.accepted_bytes,
    )
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            verify_receipt(conn, forged)
    assert exc.value.code == ErrorCode.INTEGRITY


# ------------------------------------------------------- consent / capture


def test_agent_note_without_authorization_rejected(store):
    env = _env(EnvelopeKind.AGENT_NOTE, actor="agent-1", content=b"note to self")
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            ingest_envelope(conn, store, env)
    assert exc.value.code == ErrorCode.CONSENT_REQUIRED
    # and the refusal wrote nothing
    with store.read() as conn:
        n = conn.execute("SELECT COUNT(*) FROM source_envelopes").fetchone()[0]
        assert n == 0


def test_lesson_and_tool_kinds_require_authorization(store):
    for kind in (
        EnvelopeKind.LESSON, EnvelopeKind.TOOL_CALL, EnvelopeKind.TOOL_RESULT,
    ):
        with store.tx() as conn:
            with pytest.raises(VerbatimError) as exc:
                ingest_envelope(conn, store, _env(kind, actor="agent-1"))
            assert exc.value.code == ErrorCode.CONSENT_REQUIRED


def test_agent_note_with_authorization_succeeds(store):
    env = _env(EnvelopeKind.AGENT_NOTE, actor="agent-1", content=b"remember this")
    r = _ingest(
        store, env,
        authorization=_auth({EnvelopeKind.AGENT_NOTE}),
    )
    with store.read() as conn:
        env_row = repos_v3.get(conn, "source_envelopes", {"envelope_id": r.envelope_id})
        # never human testimony (V3-13.11)
        assert env_row["trust_class"] == "agent_generated"
        assert env_row["actor_principal"] == "agent-1"


def test_agent_note_cannot_claim_principal_trust(store):
    env = _env(
        EnvelopeKind.AGENT_NOTE, actor="agent-1",
        trust_class=TrustClass.PRINCIPAL_DIRECT,
    )
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            ingest_envelope(
                conn, store, env,
                authorization=_auth({EnvelopeKind.AGENT_NOTE}),
            )
    assert exc.value.code == ErrorCode.VALIDATION


def test_capture_proof_resolves_authorization_row(store):
    with store.tx() as conn:
        repos_v3.insert(conn, "capture_authorizations", {
            "authorization_id": "cap-1", "issuer_id": "op-1",
            "principal_id": "agent-1",
            "allowed_kinds_json": ["agent_note", "tool_call", "tool_result"],
            "scope_ids_json": [SCOPE],
            "retention_policy": "task", "policy_revision": "pol-9",
            "issued_us": 1,
        })
    env = _env(EnvelopeKind.AGENT_NOTE, actor="agent-1", capture_proof="cap-1")
    r = _ingest(store, env)
    with store.read() as conn:
        assert repos_v3.get(conn, "source_envelopes", {"envelope_id": r.envelope_id})["capture_proof"] == "cap-1"


def test_capture_proof_wrong_scope_denied(store):
    with store.tx() as conn:
        repos_v3.insert(conn, "capture_authorizations", {
            "authorization_id": "cap-2", "issuer_id": "op-1",
            "principal_id": "agent-1",
            "allowed_kinds_json": ["agent_note"],
            "scope_ids_json": ["scope:elsewhere"],
            "retention_policy": "task", "policy_revision": "pol-9",
            "issued_us": 1,
        })
    env = _env(EnvelopeKind.AGENT_NOTE, actor="agent-1", capture_proof="cap-2")
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            ingest_envelope(conn, store, env)
    assert exc.value.code == ErrorCode.CONSENT_REQUIRED


def test_capture_proof_expired_denied(store):
    with store.tx() as conn:
        repos_v3.insert(conn, "capture_authorizations", {
            "authorization_id": "cap-3", "issuer_id": "op-1",
            "principal_id": "agent-1",
            "allowed_kinds_json": ["agent_note"],
            "scope_ids_json": [],
            "retention_policy": "task", "policy_revision": "pol-9",
            "issued_us": 1, "expires_us": 500,
        })
    env = _env(
        EnvelopeKind.AGENT_NOTE, actor="agent-1",
        capture_proof="cap-3", receipt_us=1_000,
    )
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            ingest_envelope(conn, store, env)
    assert exc.value.code == ErrorCode.CONSENT_REQUIRED


def test_capture_proof_revoked_denied(store):
    with store.tx() as conn:
        repos_v3.insert(conn, "capture_authorizations", {
            "authorization_id": "cap-4", "issuer_id": "op-1",
            "principal_id": "agent-1",
            "allowed_kinds_json": ["agent_note"],
            "scope_ids_json": [],
            "retention_policy": "task", "policy_revision": "pol-9",
            "issued_us": 1, "revoked_us": 50,
        })
    env = _env(EnvelopeKind.AGENT_NOTE, actor="agent-1", capture_proof="cap-4")
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            ingest_envelope(conn, store, env)
    assert exc.value.code == ErrorCode.CONSENT_REQUIRED


def test_authorization_wrong_kind_denied(store):
    env = _env(EnvelopeKind.AGENT_NOTE, actor="agent-1")
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            ingest_envelope(
                conn, store, env,
                authorization=_auth({EnvelopeKind.LESSON}),
            )
    assert exc.value.code == ErrorCode.CONSENT_REQUIRED


def test_capture_proof_must_match_passed_authorization(store):
    env = _env(EnvelopeKind.AGENT_NOTE, actor="agent-1", capture_proof="cap-other")
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            ingest_envelope(
                conn, store, env,
                authorization=_auth({EnvelopeKind.AGENT_NOTE}),
            )
    assert exc.value.code == ErrorCode.VALIDATION


def test_failed_closed_envelope_not_persisted(store):
    env = _env(EnvelopeKind.USER_MESSAGE, redaction_status="failed_closed")
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            ingest_envelope(conn, store, env)
    assert exc.value.code == ErrorCode.VALIDATION


# ------------------------------------------------------------- idempotence


def test_ingest_same_envelope_twice_is_idempotent(store):
    env = _env(
        EnvelopeKind.USER_MESSAGE, content=b"same bytes",
        external_id="msg-1", host_id="host-a",
    )
    r1 = _ingest(store, env)
    r2 = _ingest(store, env)
    assert r1.receipt_id == r2.receipt_id
    assert r1.envelope_id == r2.envelope_id
    assert r1.event_seq == r2.event_seq
    assert r1.span_ids == r2.span_ids
    with store.read() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM source_envelopes WHERE source_id = ?",
            (r1.source_id,),
        ).fetchone()[0]
        assert n == 1
        n_events = conn.execute(
            "SELECT COUNT(*) FROM events WHERE kind = 'envelope_captured'"
        ).fetchone()[0]
        assert n_events == 1


def test_conflicting_payload_under_same_key_is_typed_error(store):
    _ingest(store, _env(
        EnvelopeKind.USER_MESSAGE, content=b"bytes-1",
        external_id="dup-1", host_id="host-a",
    ))
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            ingest_envelope(conn, store, _env(
                EnvelopeKind.USER_MESSAGE, content=b"bytes-2",
                external_id="dup-1", host_id="host-a",
            ))
    assert exc.value.code == ErrorCode.VALIDATION


# --------------------------------------------------------------- trajectory


def _tool_envelopes(store):
    auth = _auth({EnvelopeKind.TOOL_CALL, EnvelopeKind.TOOL_RESULT, EnvelopeKind.TEST_RESULT})
    call = _ingest(store, _env(
        EnvelopeKind.TOOL_CALL, actor="agent-1", content=b"pytest -q",
        external_id="call-1", host_id="h1",
    ), authorization=auth)
    result = _ingest(store, _env(
        EnvelopeKind.TOOL_RESULT, actor="agent-1", content=b"12 passed",
        external_id="res-1", host_id="h1",
    ), authorization=auth)
    test = _ingest(store, _env(
        EnvelopeKind.TEST_RESULT, actor="agent-1", content=b"suite green",
        external_id="test-1", host_id="h1",
    ), authorization=auth)
    return call, result, test


def test_trajectory_steps_and_anchors(store):
    call, result, test = _tool_envelopes(store)
    envp = EnvironmentFingerprint(repo_id="repo", repo_revision="abc123")
    with store.tx() as conn:
        tid = record_trajectory(conn, TrajectoryRecord(
            trajectory_id="traj-1", scope_id=SCOPE,
            host_id="h1", task_id="task-1",
        ))
        assert tid == "traj-1"
        add_step(conn, TrajectoryStep(
            step_id="s-1", trajectory_id=tid, ord=0,
            action_envelope_id=call.envelope_id,
            observation_envelope_ids=(result.envelope_id, test.envelope_id),
            environment=envp,
        ))
        add_step(conn, TrajectoryStep(
            step_id="s-2", trajectory_id=tid, ord=1,
            action_envelope_id=test.envelope_id,
        ))
        anchor(conn, StateAnchor(
            anchor_id="a-1", kind="file_revision",
            ref="repo@abc123", digest="d" * 8,
        ), step_id="s-1")
        row = complete(conn, tid, "env-digest-1")

    assert row["completed_event"] is not None
    assert row["environment_digest"] == "env-digest-1"
    with store.read() as conn:
        traj = get_trajectory(conn, tid)
        assert traj["created_event"] > 0
        assert traj["boundary_rule"] == "task_id"
        srows = steps(conn, tid)
        assert [s["ord"] for s in srows] == [0, 1]
        assert srows[0]["action_envelope_id"] == call.envelope_id
        assert srows[0]["observation_envelope_ids"] == [
            result.envelope_id, test.envelope_id,
        ]
        assert srows[0]["environment_digest"] == envp.digest()
        a = repos_v3.get(conn, "state_anchors", {"anchor_id": "a-1"})
        assert a["step_id"] == "s-1" and a["scope_id"] == SCOPE


def test_step_ord_must_be_dense(store):
    with store.tx() as conn:
        tid = record_trajectory(conn, TrajectoryRecord(
            trajectory_id="traj-dense", scope_id=SCOPE,
        ))
        with pytest.raises(VerbatimError) as exc:
            add_step(conn, TrajectoryStep(
                step_id="s-skip", trajectory_id=tid, ord=3,
            ))
        assert exc.value.code == ErrorCode.VALIDATION


def test_step_ord_uniqueness_enforced(store):
    with store.tx() as conn:
        tid = record_trajectory(conn, TrajectoryRecord(
            trajectory_id="traj-uniq", scope_id=SCOPE,
        ))
        add_step(conn, TrajectoryStep(step_id="s-a", trajectory_id=tid, ord=0))
        # ord 0 again → dense check fires before the UNIQUE constraint
        with pytest.raises(VerbatimError):
            add_step(conn, TrajectoryStep(step_id="s-b", trajectory_id=tid, ord=0))
        # unknown trajectory
        with pytest.raises(VerbatimError) as exc:
            add_step(conn, TrajectoryStep(step_id="s-c", trajectory_id="nope", ord=0))
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_step_rejects_unknown_observation_envelope(store):
    with store.tx() as conn:
        tid = record_trajectory(conn, TrajectoryRecord(
            trajectory_id="traj-obs", scope_id=SCOPE,
        ))
        with pytest.raises(VerbatimError) as exc:
            add_step(conn, TrajectoryStep(
                step_id="s-obs", trajectory_id=tid, ord=0,
                observation_envelope_ids=("env_nonexistent",),
            ))
        assert exc.value.code == ErrorCode.VALIDATION


def test_complete_is_idempotent(store):
    with store.tx() as conn:
        tid = record_trajectory(conn, TrajectoryRecord(
            trajectory_id="traj-done", scope_id=SCOPE,
        ))
        r1 = complete(conn, tid, "d1")
        r2 = complete(conn, tid, "d2")
        assert r1["completed_event"] == r2["completed_event"]
        assert r2["environment_digest"] == "d1"


# -------------------------------------------------------------------- replay


def test_replay_manifest(store):
    call, result, test = _tool_envelopes(store)
    with store.tx() as conn:
        tid = record_trajectory(conn, TrajectoryRecord(
            trajectory_id="traj-r", scope_id=SCOPE, host_id="h1",
        ))
        add_step(conn, TrajectoryStep(
            step_id="rs-1", trajectory_id=tid, ord=0,
            action_envelope_id=call.envelope_id,
            observation_envelope_ids=(result.envelope_id,),
        ))
        add_step(conn, TrajectoryStep(
            step_id="rs-2", trajectory_id=tid, ord=1,
            action_envelope_id=test.envelope_id,
        ))
        anchor(conn, StateAnchor(
            anchor_id="ra-1", kind="artifact_digest", ref="art://x", digest="dd",
        ), step_id="rs-2")
        complete(conn, tid, "envd")
        manifest = build_replay_manifest(conn, tid)
    assert manifest.manifest_id
    assert manifest.scope_ids == (SCOPE,)
    assert manifest.source_range[0] > 0
    assert manifest.source_range[1] >= manifest.source_range[0]
    with store.read() as conn:
        run = repos_v3.get(conn, "replay_runs", {"run_id": manifest.manifest_id})
        assert run is not None
        assert run["sandbox_ref"].startswith("sandbox://replay/")
        detail = repos_v3.json_field(run, "manifest_json")
        assert [s["ord"] for s in detail["steps"]] == [0, 1]
        assert detail["anchors"][0]["anchor_id"] == "ra-1"
        assert {r["envelope_kind"] for r in detail["receipts"]} == {
            "tool_call", "tool_result", "test_result",
        }
        assert "sandbox" in detail["note"]
        report = repos_v3.json_field(run, "report_json")
        assert report["step_count"] == 2


def test_replay_refuses_live_store_ref(store):
    with store.tx() as conn:
        tid = record_trajectory(conn, TrajectoryRecord(
            trajectory_id="traj-live", scope_id=SCOPE,
        ))
        with pytest.raises(VerbatimError) as exc:
            build_replay_manifest(
                conn, tid, sandbox_ref=store.path, store=store,
            )
        assert exc.value.code == ErrorCode.VALIDATION


def test_replay_manifest_is_deterministic(store):
    with store.tx() as conn:
        tid = record_trajectory(conn, TrajectoryRecord(
            trajectory_id="traj-det", scope_id=SCOPE,
        ))
        m1 = build_replay_manifest(conn, tid)
        m2 = build_replay_manifest(conn, tid)
        assert m1.manifest_id == m2.manifest_id
        n = conn.execute("SELECT COUNT(*) FROM replay_runs").fetchone()[0]
        assert n == 1


# --------------------------------------------------------------- atomicity


def test_mid_ingest_failure_leaves_no_partial_rows(store, monkeypatch):
    # Force a failure AFTER sources/revision/span have written: the event
    # journal append raises inside the same tx.
    def boom(conn, *a, **k):
        raise RuntimeError("forced mid-ingest failure")

    monkeypatch.setattr(env_mod, "_record_event", boom)
    before = _table_counts(store)
    with pytest.raises(RuntimeError):
        with store.tx() as conn:
            ingest_envelope(conn, store, _env(EnvelopeKind.USER_MESSAGE))
    after = _table_counts(store)
    assert before == after


def test_consent_failure_leaves_no_rows(store):
    before = _table_counts(store)
    with store.tx() as conn:
        with pytest.raises(VerbatimError):
            ingest_envelope(conn, store, _env(EnvelopeKind.AGENT_NOTE, actor="agent-1"))
    assert _table_counts(store) == before


# ----------------------------------------------------------------- v2 compat


def test_sources_row_carries_v2_kind(store):
    cases = [
        (EnvelopeKind.USER_MESSAGE, "user_message"),
        (EnvelopeKind.ASSISTANT_MESSAGE, "assistant_message"),
        (EnvelopeKind.TOOL_CALL, "tool_output"),
        (EnvelopeKind.TEST_RESULT, "tool_output"),
        (EnvelopeKind.DOCUMENT, "import"),
        (EnvelopeKind.HANDOFF, "import"),
        (EnvelopeKind.SYSTEM_EVENT, "operator_record"),
        (EnvelopeKind.AGENT_NOTE, "assistant_message"),
        (EnvelopeKind.LESSON, "assistant_message"),
        (EnvelopeKind.FILE_DIFF, "tool_output"),
        (EnvelopeKind.ERROR, "tool_output"),
        (EnvelopeKind.IMPORT, "import"),
        (EnvelopeKind.CONNECTOR_ITEM, "import"),
        (EnvelopeKind.DELEGATION, "import"),
        (EnvelopeKind.PLAN, "assistant_message"),
        (EnvelopeKind.SUBGOAL, "assistant_message"),
        (EnvelopeKind.DECISION, "assistant_message"),
        (EnvelopeKind.VERIFICATION, "tool_output"),
        (EnvelopeKind.BROWSER_STATE, "tool_output"),
        (EnvelopeKind.SCREENSHOT_REF, "tool_output"),
        (EnvelopeKind.RECOVERY, "tool_output"),
        (EnvelopeKind.FILE_SNAPSHOT_REF, "tool_output"),
    ]
    auth = _auth(set(EnvelopeKind))
    repo = SourcesRepo(store)
    for i, (kind, v2_kind) in enumerate(cases):
        r = _ingest(
            store,
            _env(kind, actor="agent-1", content=f"payload-{i}".encode(),
                 external_id=f"ext-{i}"),
            authorization=auth,
        )
        src = repo.get(r.source_id)
        assert src["source_kind"] == v2_kind, f"{kind} -> {src['source_kind']}"
        assert src["scope_id"] == SCOPE
        # v2 read paths still resolve payload + revision
        assert repo.payload(r.source_id, 1) == f"payload-{i}".encode()
        assert repo.get_revision(r.source_id, 1)["payload_bytes"] == len(f"payload-{i}")


def test_v2_kind_mapping_is_total():
    assert set(env_mod._V2_KIND) == set(EnvelopeKind)


# ------------------------------------------------------------------ helpers


def test_detect_whole_and_line_spans():
    payload = b"line one\n\nline two\nline three"
    assert detect.whole_payload_span(payload) == (0, len(payload))
    lines = detect.line_spans(payload)
    # blank line at offset 9 is skipped — spans carry exact byte ranges
    assert lines == [(0, 8), (10, 18), (19, 29)]
    spans = detect.spans_for(payload, "text/plain", include_lines=True)
    assert spans[0] == (0, len(payload))
    assert (0, 8) in spans


def test_detect_span_id_deterministic():
    a = detect.span_id_for("src", 1, 0, 5)
    b = detect.span_id_for("src", 1, 0, 5)
    assert a == b and a.startswith("sp_")
    assert detect.span_id_for("src", 1, 0, 6) != a


def test_artifact_ref_envelope_persists_descriptor(store):
    env = _env(
        EnvelopeKind.DOCUMENT, content=None, artifact_ref="blob://sha256:abc",
        media_type="application/pdf",
    )
    r = _ingest(store, env)
    with store.read() as conn:
        env_row = repos_v3.get(conn, "source_envelopes", {"envelope_id": r.envelope_id})
        assert env_row["artifact_ref"] == "blob://sha256:abc"
        assert env_row["trust_class"] == "external_content"
        payload = conn.execute(
            "SELECT payload FROM source_revisions WHERE source_id = ?",
            (r.source_id,),
        ).fetchone()[0]
        assert b"blob://sha256:abc" in bytes(payload)
