"""Regression test for the purge suppression-commit seam (MUT-16 evidence).

``verbatim/purge.py::_execute`` calls ``purges.confirm_suppress`` — the
previewed → suppressed transition that stamps ``approved_us`` — *before*
any physical cleanup. The mutation that replaces that call with ``pass``
leaves the purge row without its suppression-commit marker while payloads
are still scrubbed: erasure evidence with no committed tombstone record.

This test exercises the real ``Store.create`` path (schema v2, real
transactions) and asserts the suppression marker is durable after
``execute_purge`` — under the mutant the assertion fails.
"""

from __future__ import annotations

from verbatim.core.lifecycle import read_claim_head
from verbatim.core.types import (
    Lifecycle,
    Modality,
    Polarity,
    Provenance,
    Scope,
    SourceEnvelope,
    SourceKind,
    new_id,
)
from verbatim.purge import execute_purge, plan_purge
from verbatim.storage.repos import (
    ClaimsRepo,
    EventsRepo,
    PurgesRepo,
    SourcesRepo,
    SpansRepo,
    ensure_scope,
)
from verbatim.storage.store import Store


def _evidence(store: Store, scope: Scope) -> dict:
    """Persist source + span + claim(+revision) through the real repos."""
    payload = b"mutation-evidence: suppression precedes erasure"
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
                external_id=f"mut-{new_id()[:8]}",
            ),
            conn=conn,
        )
        assert created
        span_id = new_id()
        SpansRepo(store).insert(
            span_id, src_id, 1, 0, len(payload), "mut-harvest", conn=conn
        )
        claim_id = ClaimsRepo(store).create(sid, None, "met", conn)
        seq = conn.execute(
            "SELECT COALESCE(MAX(event_seq), 0) + 1 FROM events"
        ).fetchone()[0]
        ClaimsRepo(store).add_revision(
            claim_id,
            Lifecycle.ACTIVE,
            {"text": payload.decode()},
            Polarity.AFFIRMATIVE,
            Modality.ASSERTED,
            None,
            {"note": "mutation-seeded"},
            [],
            [(span_id, "primary")],
            seq,
            conn,
        )
        EventsRepo(store).append(
            conn, sid, "ingested", "mutation", {"source_id": src_id}, "policy-1"
        )
    return {"scope_id": sid, "claim_id": claim_id}


def test_execute_purge_commits_suppression_marker(tmp_path):
    store = Store.create(str(tmp_path / "mut16.db"))
    scope = Scope(
        profile_id="prof",
        principal_id="alice",
        workspace_id=None,
        conversation_id="conv1",
    )
    ev = _evidence(store, scope)

    plan = plan_purge(store, scope, [("claim", ev["claim_id"])], actor="alice")
    result = execute_purge(store, plan["purge_id"])
    assert result["state"] == "completed"

    row = PurgesRepo(store).get(plan["purge_id"])
    assert row is not None and row["state"] == "completed"
    # The suppression commit marker: previewed → suppressed stamped this.
    # Under MUT-16 (confirm_suppress skipped) it stays NULL — killed.
    assert row["approved_us"] is not None
    assert row["completed_us"] is not None
    assert row["approved_us"] <= row["completed_us"]

    with store.read() as conn:
        head = read_claim_head(conn, ev["claim_id"])
        assert head is not None and head.state == Lifecycle.ERASED
