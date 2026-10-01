"""V2 purge orchestration: plan → execute, suppression, restore fencing.

Runs against the real ``Store`` (schema v2) in a tmp_path — erasure ledger,
handoff, disclosure, and operations tables all exist there.
"""

from __future__ import annotations

import sqlite3

import pytest

from verbatim.core.lifecycle import read_claim_head
from verbatim.core.types import (
    CallerContext,
    ErrorCode,
    GrantKind,
    Lifecycle,
    Modality,
    Polarity,
    Provenance,
    Scope,
    SourceEnvelope,
    SourceKind,
    VerbatimError,
    json_dumps,
    new_id,
)
from verbatim.purge import (
    check_restore_fence,
    execute_purge,
    lift_suppression,
    plan_purge,
    suppress,
)
from verbatim.storage.repos import (
    ClaimsRepo,
    EventsRepo,
    SourcesRepo,
    SpansRepo,
    ensure_scope,
)
from verbatim.storage.repos_v2 import ErasureRepo
from verbatim.storage.store import Store


# ----------------------------------------------------------------------
# shared helpers (imported by test_v2_export / test_v2_privacy)
# ----------------------------------------------------------------------


def make_store(tmp_path, name="v2.db") -> Store:
    return Store.create(str(tmp_path / name))


def qrows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def qrow(conn: sqlite3.Connection, sql: str, params: tuple = ()):
    rows = qrows(conn, sql, params)
    return rows[0] if rows else None


def make_scope(
    profile="prof",
    principal="alice",
    workspace=None,
    conversation="conv1",
) -> Scope:
    return Scope(
        profile_id=profile,
        principal_id=principal,
        workspace_id=workspace,
        conversation_id=conversation,
    )


def make_caller(
    profile="prof",
    principal="alice",
    *,
    grants=(GrantKind.SHARE, GrantKind.EXPORT, GrantKind.PURGE, GrantKind.SUPPRESS),
    agent_id="agent-1",
    workspace=None,
    conversation="conv1",
) -> CallerContext:
    return CallerContext(
        profile_id=profile,
        principal_id=principal,
        agent_id=agent_id,
        workspace_id=workspace,
        conversation_id=conversation,
        grants=frozenset(grants),
    )


def make_evidence(
    store: Store,
    scope: Scope,
    text: str = "Alice met Bob at the club on Tuesday",
    *,
    predicate: str = "met",
) -> dict:
    """Persist source + span + claim(+revision/evidence) in one tx."""
    payload = text.encode("utf-8")
    with store.tx() as conn:
        sid = ensure_scope(store, conn, scope)
        src_id, created = SourcesRepo(store).insert(
            SourceEnvelope(
                origin="test",
                source_kind=SourceKind.USER_MESSAGE,
                scope=scope,
                speaker_id="alice",
                payload=payload,
                event_us=1,
                captured_us=1,
                provenance=Provenance.DIRECT_USER,
                external_id=f"ext-{new_id()[:8]}",
            ),
            conn=conn,
        )
        assert created
        span_id = new_id()
        SpansRepo(store).insert(
            span_id, src_id, 1, 0, len(payload), "test-harvest-1", conn=conn
        )
        claim_id = ClaimsRepo(store).create(sid, None, predicate, conn)
        seq = conn.execute(
            "SELECT COALESCE(MAX(event_seq), 0) + 1 FROM events"
        ).fetchone()[0]
        ClaimsRepo(store).add_revision(
            claim_id,
            Lifecycle.ACTIVE,
            {"text": text},
            Polarity.AFFIRMATIVE,
            Modality.ASSERTED,
            None,
            {"note": "derived"},
            [],
            [(span_id, "primary")],
            seq,
            conn,
        )
        EventsRepo(store).append(
            conn, sid, "ingested", "test", {"source_id": src_id}, "policy-1"
        )
    return {
        "scope_id": sid,
        "source_id": src_id,
        "span_id": span_id,
        "claim_id": claim_id,
        "payload": payload,
    }


# ----------------------------------------------------------------------
# plan → execute
# ----------------------------------------------------------------------


def test_plan_then_execute_erases_claim_and_writes_ledger(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    sid, cid = ev["scope_id"], ev["claim_id"]

    plan = plan_purge(store, scope, [("claim", cid)], actor="alice")
    assert plan["state"] == "previewed"
    assert plan["purge_id"]

    result = execute_purge(store, plan["purge_id"])
    assert result["state"] == "completed"
    assert ("claim", cid) in result["erased"]
    assert result["projection_generation"] >= 2
    assert result["event_seq"] > 0

    with store.read() as conn:
        head = read_claim_head(conn, cid)
        assert head is not None and head.state == Lifecycle.ERASED
        rows = qrows(
            conn,
            "SELECT object_json, interpretation_json FROM claim_revisions"
            " WHERE claim_id = ?",
            (cid,),
        )
        assert all(r["object_json"] is None for r in rows)
        assert ErasureRepo(store).is_erased(conn, sid, "claim", cid)


def test_purge_source_scrubs_payload_and_dependents(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    sid, src, cid = ev["scope_id"], ev["source_id"], ev["claim_id"]

    plan = plan_purge(store, sid, [("source", src)], actor="alice")
    # dependency closure pulls the claim citing the source's span into targets
    assert ("claim", cid) in [tuple(t) for t in plan["targets"]]

    execute_purge(store, plan["purge_id"])
    with store.read() as conn:
        rev = qrow(
            conn,
            "SELECT payload, payload_hmac FROM source_revisions"
            " WHERE source_id = ? AND revision = 1",
            (src,),
        )
        assert bytes(rev["payload"]) == b""
        head = read_claim_head(conn, cid)
        assert head is not None and head.state == Lifecycle.ERASED
        assert ErasureRepo(store).is_erased(conn, sid, "source", src)
        assert ErasureRepo(store).is_erased(conn, sid, "source_revision", f"{src}:1")


def test_revision_scoped_purge_leaves_other_revisions(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope, "first body text")
    src = ev["source_id"]
    # a true second revision on the same source
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO source_revisions"
            "(source_id, revision, payload, payload_hmac, event_us,"
            " captured_us, timezone, provenance, metadata_json)"
            " VALUES (?, 2, ?, ?, 2, 2, NULL, 'direct_user', '{}')",
            (src, b"second body text", store.hmac(b"second body text")),
        )

    plan = plan_purge(store, ev["scope_id"], [("source", src, 1)], actor="alice")
    execute_purge(store, plan["purge_id"])
    with store.read() as conn:
        r1 = qrow(
            conn,
            "SELECT payload FROM source_revisions WHERE source_id = ? AND revision = 1",
            (src,),
        )
        r2 = qrow(
            conn,
            "SELECT payload FROM source_revisions WHERE source_id = ? AND revision = 2",
            (src,),
        )
        assert bytes(r1["payload"]) == b""
        assert bytes(r2["payload"]) == b"second body text"
        assert ErasureRepo(store).is_erased(
            conn, ev["scope_id"], "source_revision", f"{src}:1"
        )
        assert not ErasureRepo(store).is_erased(
            conn, ev["scope_id"], "source_revision", f"{src}:2"
        )


def test_plan_rejects_unknown_kind_and_foreign_scope(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    other = make_scope(principal="bob", conversation="conv2")

    with pytest.raises(VerbatimError) as ei:
        plan_purge(store, scope, [("mystery", ev["claim_id"])], actor="alice")
    assert ei.value.code == ErrorCode.VALIDATION

    with pytest.raises(VerbatimError) as ei:
        plan_purge(store, other, [("claim", ev["claim_id"])], actor="bob")
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN

    with pytest.raises(VerbatimError) as ei:
        plan_purge(store, scope, [("claim", "nonexistent-id")], actor="alice")
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


# ----------------------------------------------------------------------
# reversible suppression
# ----------------------------------------------------------------------


def test_suppress_hides_then_lift_restores(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    sid, cid = ev["scope_id"], ev["claim_id"]

    res = suppress(store, scope, [("claim", cid)], actor="alice")
    assert res["state"] == "suppressed" and res["reversible"] is True

    with store.read() as conn:
        head = read_claim_head(conn, cid)
        assert head is not None and head.recorded_until is not None
        # tombstone set is visible through the purge registry
        sup = qrow(
            conn,
            "SELECT state FROM purges WHERE purge_id = ?",
            (res["purge_id"],),
        )
        assert sup["state"] == "suppressed"
        # suppression writes NO erasure-ledger rows
        assert not ErasureRepo(store).is_erased(conn, sid, "claim", cid)

    lifted = lift_suppression(store, res["purge_id"], actor="alice")
    assert lifted["lifted"] is True and lifted["restored"] >= 1
    with store.read() as conn:
        head = read_claim_head(conn, cid)
        assert head is not None and head.recorded_until is None
        assert qrow(conn, "SELECT 1 AS x FROM purges WHERE purge_id = ?",
                    (res["purge_id"],)) is None


def test_lift_after_execute_is_denied(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    plan = plan_purge(store, scope, [("claim", ev["claim_id"])], actor="alice")
    execute_purge(store, plan["purge_id"])
    with pytest.raises(VerbatimError) as ei:
        lift_suppression(store, plan["purge_id"], actor="alice")
    assert ei.value.code == ErrorCode.INVALID_TRANSITION


# ----------------------------------------------------------------------
# restore fence
# ----------------------------------------------------------------------


def test_restore_fence_blocks_resurrection(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    sid, cid = ev["scope_id"], ev["claim_id"]

    plan = plan_purge(store, scope, [("claim", cid)], actor="alice")
    execute_purge(store, plan["purge_id"])

    with store.read() as conn:
        assert check_restore_fence(store, conn, sid, "claim", cid) is True
        assert check_restore_fence(store, conn, sid, "claim", "other-id") is False
        # ledger stores only opaque digests — the raw id never appears
        rows = qrows(conn, "SELECT * FROM erasure_ledger WHERE scope_id = ?", (sid,))
        assert rows
        assert all(cid not in str(r.values()) for r in rows)
        assert all(r["object_kind"] == "claim" for r in rows)
        with pytest.raises(VerbatimError) as ei:
            check_restore_fence(store, conn, sid, "bogus", cid)
        assert ei.value.code == ErrorCode.VALIDATION


def test_purge_source_scrubs_metadata_envelope_and_identity(tmp_path):
    """Physical erasure covers revision metadata, v3 envelope content, and
    submitter-controlled source identity — not just the payload bytes
    (V2-41.10, V3-36.02). The receipt reports derived coverage plus the
    honest ``unhandled`` audit list."""
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    sid, src = ev["scope_id"], ev["source_id"]
    with store.tx() as conn:
        conn.execute(
            "UPDATE source_revisions SET metadata_json = ?"
            " WHERE source_id = ? AND revision = 1",
            (
                json_dumps({"note": "sensitive context", "author": "alice"}),
                src,
            ),
        )
        conn.execute(
            "INSERT INTO source_envelopes (envelope_id, source_id, revision,"
            " scope_id, envelope_kind, actor_principal, host_id, session_id,"
            " task_id, capture_proof, artifact_ref, metadata_json)"
            " VALUES ('env1', ?, 1, ?, 'user_message', 'alice', 'host-1',"
            " 'sess-9', 'task-7', 'proof-bytes', 'artifact://x', ?)",
            (src, sid, json_dumps({"context": "caller supplied"})),
        )

    plan = plan_purge(store, sid, [("source", src)], actor="alice")
    result = execute_purge(store, plan["purge_id"])
    assert result["state"] == "completed"
    assert isinstance(result["derived"], dict)
    assert isinstance(result["unhandled"], list)

    with store.read() as conn:
        rev = qrow(
            conn,
            "SELECT payload, metadata_json FROM source_revisions"
            " WHERE source_id = ? AND revision = 1",
            (src,),
        )
        assert bytes(rev["payload"]) == b""
        assert rev["metadata_json"] == "{}"
        # the covering envelope row goes entirely (v3 closure semantics)
        assert (
            qrow(
                conn,
                "SELECT 1 AS x FROM source_envelopes WHERE envelope_id = 'env1'",
            )
            is None
        )
        srow = qrow(
            conn,
            "SELECT external_id, speaker_id FROM sources WHERE source_id = ?",
            (src,),
        )
        assert srow["external_id"] is None
        assert srow["speaker_id"] is None
        # audit payloads naming purged ids are disclosed, not rewritten —
        # the purge_previewed/suppressed events legitimately remain
        assert any(
            u.get("table") == "events" for u in result["unhandled"]
        )


def test_purge_claim_deletes_graph_derived_observation(tmp_path):
    """An observation derived from a purged claim is deleted through the
    derivations closure — derived content does not outlive its evidence
    (V3-17.03, V3-36.02)."""
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    sid, cid = ev["scope_id"], ev["claim_id"]
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO observations (observation_id, scope_id, revision,"
            " text, proof_count, recorded_from)"
            " VALUES ('obs1', ?, 1, 'derived note', 1, 1)",
            (sid,),
        )
        conn.execute(
            "INSERT INTO observation_evidence (observation_id, revision,"
            " role, object_kind, object_id, object_revision)"
            " VALUES ('obs1', 1, 'supports', 'claim', ?, 1)",
            (cid,),
        )
        conn.execute(
            "INSERT INTO derivations (child_kind, child_id, child_revision,"
            " parent_kind, parent_id, parent_revision, producer_kind,"
            " producer_id, seq, scope_id)"
            " VALUES ('observation', 'obs1', 1, 'claim', ?, 1,"
            " 'slot_aggregate_v1', 'obs:obs1', 1, ?)",
            (cid, sid),
        )

    plan = plan_purge(store, sid, [("claim", cid)], actor="alice")
    result = execute_purge(store, plan["purge_id"])
    assert result["state"] == "completed"
    assert result["derived"].get("derived_observation") == 1
    with store.read() as conn:
        assert (
            qrow(conn, "SELECT 1 AS x FROM observations"
                       " WHERE observation_id = 'obs1'")
            is None
        )
        assert (
            qrow(conn, "SELECT 1 AS x FROM observation_evidence"
                       " WHERE observation_id = 'obs1'")
            is None
        )
        assert (
            qrow(conn, "SELECT 1 AS x FROM derivations"
                       " WHERE child_id = 'obs1'")
            is None
        )


def test_purge_rejects_revision_mismatch(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    with pytest.raises(VerbatimError) as ei:
        plan_purge(store, scope, [("claim", ev["claim_id"], 99)], actor="alice")
    assert ei.value.code in (ErrorCode.VALIDATION, ErrorCode.NOT_FOUND_OR_FORBIDDEN)
