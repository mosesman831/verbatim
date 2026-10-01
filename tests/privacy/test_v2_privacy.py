"""V2 privacy: handoff capsules (share/consume/revoke/expire) and egress
disclosure receipts."""

from __future__ import annotations

from decimal import Decimal

import pytest

from verbatim.config import JudgeConfig, VerbatimConfig
from verbatim.core.time import wall_us
from verbatim.core.types import (
    CallerContext,
    ErrorCode,
    GrantKind,
    Mode,
    VerbatimError,
)
from verbatim.privacy.egress import EgressGate
from verbatim.purge import execute_purge, plan_purge, suppress
from verbatim.sharing import (
    consume_handoff,
    create_handoff,
    revoke_handoff,
)
from tests.conftest import grant_consent
from tests.privacy.test_v2_purge import (
    make_caller,
    make_evidence,
    make_scope,
    make_store,
    qrow,
    qrows,
)


def _cfg(mode=Mode.JEV_ASSISTED, budget="1.00", backend="jev"):
    return VerbatimConfig(
        mode=mode,
        judge=JudgeConfig(backend=backend, daily_budget_usd=Decimal(budget)),
    )


# ----------------------------------------------------------------------
# handoff capsules
# ----------------------------------------------------------------------


def test_create_and_consume_handoff(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    caller = make_caller()

    made = create_handoff(
        store,
        caller,
        "bob",
        [("claim", ev["claim_id"]), ("span", ev["span_id"])],
        permission="read_evidence",
    )
    assert made["status"] == "open"
    assert made["member_count"] == 2
    # no plaintext is copied into the capsule
    with store.read() as conn:
        cap = qrow(conn, "SELECT * FROM handoff_capsules WHERE capsule_id = ?",
                   (made["capsule_id"],))
        assert cap["snapshot_json"] in (None, "{}")
        members = qrows(conn, "SELECT * FROM capsule_members WHERE capsule_id = ?",
                        (made["capsule_id"],))
        assert len(members) == 2

    recipient = CallerContext(
        profile_id="prof", principal_id="bob",
        grants=frozenset({GrantKind.READ_EVIDENCE}),
    )
    got = consume_handoff(store, recipient, made["capsule_id"])
    assert got["status"] == "consumed"
    assert got["role"] == "recipient"
    kinds = {(m["object_kind"], m["object_id"]) for m in got["members"]}
    assert ("claim", ev["claim_id"]) in kinds
    assert ("span", ev["span_id"]) in kinds
    assert got["dropped"] == []


def test_consume_drops_members_under_source_hold(tmp_path):
    """A quarantine hold on the source revision taints both the claim
    citing its span and the span itself — consumption reports them under
    ``dropped`` rather than shipping held references (V3-14.10)."""
    import verbatim.security as security

    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    caller = make_caller()
    made = create_handoff(
        store,
        caller,
        "bob",
        [("claim", ev["claim_id"]), ("span", ev["span_id"])],
        permission="read_evidence",
    )
    with store.tx() as conn:
        security.open_quarantine(
            conn, ("source", ev["source_id"], 1),
            ["attack_risk:blocked"], [], scope_id=ev["scope_id"],
        )
    recipient = CallerContext(
        profile_id="prof", principal_id="bob",
        grants=frozenset({GrantKind.READ_EVIDENCE}),
    )
    got = consume_handoff(store, recipient, made["capsule_id"])
    assert got["members"] == []
    assert len(got["dropped"]) == 2


def test_create_handoff_refuses_quarantined_member(tmp_path):
    """A quarantined object fails create-time disclosure the same way a
    suppressed or missing one does (V3-14.10)."""
    import verbatim.security as security

    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    caller = make_caller()
    with store.tx() as conn:
        security.open_quarantine(
            conn, ("claim", ev["claim_id"], 1),
            ["attack_risk:blocked"], [], scope_id=ev["scope_id"],
        )
    with pytest.raises(VerbatimError) as ei:
        create_handoff(store, caller, "bob", [("claim", ev["claim_id"])])
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


def test_create_handoff_requires_share_grant(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    caller = make_caller(grants=())  # no SHARE grant
    with pytest.raises(VerbatimError) as ei:
        create_handoff(store, caller, "bob", [("claim", ev["claim_id"])])
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


def test_create_handoff_rejects_foreign_object(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    caller = make_caller()
    other_caller = make_caller(principal="bob")
    # bob's home scope does not contain alice's claim
    with pytest.raises(VerbatimError) as ei:
        create_handoff(store, other_caller, "carol", [("claim", ev["claim_id"])])
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


def test_revoked_capsule_fails_closed(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    caller = make_caller()
    made = create_handoff(store, caller, "bob", [("claim", ev["claim_id"])])

    revoke = revoke_handoff(store, caller, made["capsule_id"])
    assert revoke["status"] == "revoked"

    recipient = CallerContext(profile_id="prof", principal_id="bob",
                              grants=frozenset({GrantKind.READ_EVIDENCE}))
    with pytest.raises(VerbatimError) as ei:
        consume_handoff(store, recipient, made["capsule_id"])
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


def test_expired_capsule_denies_consumption(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    caller = make_caller()
    made = create_handoff(
        store, caller, "bob", [("claim", ev["claim_id"])],
        expires_us=wall_us() + 60_000_000,
    )
    # force expiry by backdating the capsule row
    with store.tx() as conn:
        conn.execute(
            "UPDATE handoff_capsules SET expires_us = ? WHERE capsule_id = ?",
            (wall_us() - 1, made["capsule_id"]),
        )
    recipient = CallerContext(profile_id="prof", principal_id="bob",
                              grants=frozenset({GrantKind.READ_EVIDENCE}))
    with pytest.raises(VerbatimError) as ei:
        consume_handoff(store, recipient, made["capsule_id"])
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN
    with store.read() as conn:
        cap = qrow(conn, "SELECT status FROM handoff_capsules WHERE capsule_id = ?",
                   (made["capsule_id"],))
        assert cap["status"] == "expired"


def test_non_party_cannot_consume(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    caller = make_caller()
    made = create_handoff(store, caller, "bob", [("claim", ev["claim_id"])])
    stranger = CallerContext(profile_id="prof", principal_id="mallory",
                             grants=frozenset({GrantKind.READ_EVIDENCE}))
    with pytest.raises(VerbatimError) as ei:
        consume_handoff(store, stranger, made["capsule_id"])
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


def test_consume_drops_suppressed_members(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    caller = make_caller()
    made = create_handoff(
        store, caller, "bob",
        [("claim", ev["claim_id"]), ("span", ev["span_id"])],
    )
    # suppress the claim after sharing — consumption re-checks availability
    # and reports it dropped rather than disclosing it.
    suppress(store, scope, [("claim", ev["claim_id"])], actor="alice")

    recipient = CallerContext(profile_id="prof", principal_id="bob",
                              grants=frozenset({GrantKind.READ_EVIDENCE}))
    got = consume_handoff(store, recipient, made["capsule_id"])
    live_ids = {m["object_id"] for m in got["members"]}
    assert ev["claim_id"] not in live_ids
    assert ev["span_id"] in live_ids  # the span stays disclosable
    assert any(d["object_id"] == ev["claim_id"] for d in got["dropped"])
    assert got["warnings"]


def test_purge_removes_capsule_membership(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    caller = make_caller()
    made = create_handoff(store, caller, "bob", [("claim", ev["claim_id"])])

    plan = plan_purge(store, ev["scope_id"], [("claim", ev["claim_id"])],
                    actor="alice")
    execute_purge(store, plan["purge_id"])

    # physical erasure scrubs the capsule membership reference itself —
    # a later consumer cannot even tell the claim was ever a member.
    with store.read() as conn:
        members = qrows(conn, "SELECT * FROM capsule_members"
                              " WHERE capsule_id = ?", (made["capsule_id"],))
        assert all(m["object_id"] != ev["claim_id"] for m in members)


def test_suppressed_member_is_not_shared(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    caller = make_caller()
    suppress(store, scope, [("claim", ev["claim_id"])], actor="alice")
    with pytest.raises(VerbatimError) as ei:
        create_handoff(store, caller, "bob", [("claim", ev["claim_id"])])
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


# ----------------------------------------------------------------------
# egress disclosure receipts
# ----------------------------------------------------------------------


def test_authorize_writes_disclosure_receipt(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    sid = ev["scope_id"]

    grant_consent(store, sid)
    gate = EgressGate(store, _cfg())
    rid = gate.authorize(
        sid, "typesafe", "candidate_curation", 1000,
        input_refs=[f"claim:{ev['claim_id']}", f"span:{ev['span_id']}"],
    )
    assert rid
    with store.read() as conn:
        rows = qrows(conn, "SELECT * FROM disclosures WHERE scope_id = ?", (sid,))
        assert len(rows) == 1
        d = rows[0]
        assert d["processor"] == "typesafe"
        assert d["purpose"] == "candidate_curation"
        assert d["outcome"] == "reserved"
        import json as _j

        refs = _j.loads(d["input_refs_json"])
        assert f"claim:{ev['claim_id']}" in refs
        # the receipt carries ids only — no payload text
        assert "Alice met Bob" not in d["input_refs_json"]

    gate.settle(rid, 900)
    with store.read() as conn:
        d = qrow(conn, "SELECT outcome, usage_json FROM disclosures"
                       " WHERE scope_id = ?", (sid,))
        assert d["outcome"] == "settled"
        usage = qrow(conn, "SELECT count FROM usage_aggregates"
                           " WHERE scope_id = ? AND object_kind = 'claim'"
                           " AND kind = 'egress'", (sid,))
        assert usage is not None and usage["count"] == 1


def test_expire_updates_disclosure_outcome(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    sid = ev["scope_id"]
    grant_consent(store, sid)
    gate = EgressGate(store, _cfg())
    rid = gate.authorize(sid, "typesafe", "candidate_curation", 100)
    gate.expire(rid)
    with store.read() as conn:
        d = qrow(conn, "SELECT outcome FROM disclosures WHERE scope_id = ?", (sid,))
        assert d["outcome"] == "expired"


def test_denied_authorize_writes_no_receipt(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    sid = ev["scope_id"]
    gate = EgressGate(store, _cfg())  # no consent granted
    with pytest.raises(VerbatimError) as ei:
        gate.authorize(sid, "typesafe", "candidate_curation", 100)
    assert ei.value.code == ErrorCode.EGRESS_DISABLED
    with store.read() as conn:
        assert qrows(conn, "SELECT * FROM disclosures") == []


def test_suppressed_scope_pauses_egress(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    sid = ev["scope_id"]
    grant_consent(store, sid)
    gate = EgressGate(store, _cfg())
    suppress(store, scope, [("claim", ev["claim_id"])], actor="alice")
    # scope-level suppression tombstone blocks dispatch
    with pytest.raises(VerbatimError) as ei:
        gate.authorize(sid, "typesafe", "candidate_curation", 100)
    assert ei.value.code == ErrorCode.EGRESS_DISABLED
