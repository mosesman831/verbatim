"""Handoff capsules — SPEC_V4 V4-47.07/08 (C74).

Capsules bind recipient + purpose + expiry + exact object revisions +
allowed operations; they can never delegate past the issuer's grants;
offline presentation reports revocation honestly and is read-only.
"""

from __future__ import annotations

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.projections import (
    ProjectionAuthority,
    create_capsule,
    open_capsule,
    present_offline,
)
from verbatim.storage.repos_v2 import HandoffRepo

from .conftest import grant, scope_row, seed_claim

T0 = 1_700_000_000_000_000


def _auth(pid="human:alice", purpose="recall"):
    return ProjectionAuthority(principal_id=pid, purpose=purpose)


def _seeded(store):
    with store.tx() as conn:
        scope_row(conn, "sA")
        grant(conn, "sA")                               # alice: read+quote
        grant(conn, "sA", pid="agent:bob")              # bob: read+quote
        grant(conn, "sA", pid="human:carol",
              verbs=("read",))                        # carol: read only
        seed_claim(conn, "cl1", "sA", "src1", "sp1", "handoff fact",
                   obj={"text": "hf"})


def _members():
    return [("claim", "cl1", 1), ("span", "sp1", 1)]


class TestBinding:
    def test_capsule_binds_all_fields(self, store):
        _seeded(store)
        cap = create_capsule(
            store, issuer=_auth(), scope_id="sA",
            recipient_id="agent:bob", members=_members(),
            verbs=("read", "quote"), expires_us=T0 + 10_000,
            now_us=T0,
        )
        row = HandoffRepo(store).get(cap["capsule_id"])
        assert row["recipient_id"] == "agent:bob"
        assert row["issuer_id"] == "human:alice"
        assert row["expires_us"] == T0 + 10_000
        assert row["purpose"] == "recall"
        import json
        assert set(json.loads(row["verbs_json"])) == {"read", "quote"}
        members = HandoffRepo(store).members(cap["capsule_id"])
        assert {
            (m["object_kind"], m["object_id"], m["revision"])
            for m in members
        } == {("claim", "cl1", 1), ("span", "sp1", 1)}

    def test_expiry_required(self, store):
        _seeded(store)
        with pytest.raises(VerbatimError) as ei:
            create_capsule(
                store, issuer=_auth(), scope_id="sA",
                recipient_id="agent:bob", members=_members(),
                expires_us=None, now_us=T0,
            )
        assert ei.value.code is ErrorCode.VALIDATION

    def test_no_authority_expansion(self, store):
        """V4-47.07: an issuer without quote cannot delegate quote."""
        _seeded(store)
        with pytest.raises(VerbatimError) as ei:
            create_capsule(
                store, issuer=_auth("human:carol"), scope_id="sA",
                recipient_id="agent:bob", members=_members(),
                verbs=("quote",), expires_us=T0 + 10_000, now_us=T0,
            )
        assert ei.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    def test_prohibited_verbs_rejected(self, store):
        """V4-47.08: act/hydrate/promote can never ride a capsule."""
        _seeded(store)
        for bad in ("act", "hydrate", "promote", "derive", "admin"):
            with pytest.raises(VerbatimError) as ei:
                create_capsule(
                    store, issuer=_auth(), scope_id="sA",
                    recipient_id="agent:bob", members=_members(),
                    verbs=(bad,), expires_us=T0 + 10_000, now_us=T0,
                )
            assert ei.value.code is ErrorCode.VALIDATION

    def test_foreign_member_rejected(self, store):
        """Members outside the scope (or nonexistent) deny the capsule."""
        _seeded(store)
        with store.tx() as conn:
            scope_row(conn, "sB")
            grant(conn, "sB")
            seed_claim(conn, "clX", "sB", "srcX", "spX", "other scope")
        with pytest.raises(VerbatimError) as ei:
            create_capsule(
                store, issuer=_auth(), scope_id="sA",
                recipient_id="agent:bob",
                members=[("claim", "clX", 1)],
                expires_us=T0 + 10_000, now_us=T0,
            )
        assert ei.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


class TestConsumption:
    def test_recipient_reauthorized_and_leased(self, store):
        _seeded(store)
        cap = create_capsule(
            store, issuer=_auth(), scope_id="sA",
            recipient_id="agent:bob", members=_members(),
            verbs=("read", "quote"), expires_us=T0 + 10_000, now_us=T0,
        )
        out = open_capsule(
            store, cap["capsule_id"], recipient=_auth("agent:bob"),
            now_us=T0 + 1,
        )
        assert out["granted_verbs"] == ["quote", "read"]
        assert set(out["leases"]) == {"read", "quote"}
        assert HandoffRepo(store).get(cap["capsule_id"])["status"] == "consumed"

    def test_capsule_narrows_to_recipient_grants(self, store):
        """Recipient without quote gets a read-only lease, never more."""
        _seeded(store)
        with store.tx() as conn:
            grant(conn, "sA", pid="agent:dave", verbs=("read",))
        cap = create_capsule(
            store, issuer=_auth(), scope_id="sA",
            recipient_id="agent:dave", members=_members(),
            verbs=("read", "quote"), expires_us=T0 + 10_000, now_us=T0,
        )
        out = open_capsule(
            store, cap["capsule_id"], recipient=_auth("agent:dave"),
            now_us=T0 + 1,
        )
        assert out["granted_verbs"] == ["read"]
        assert out["denied_verbs"] == ["quote"]

    def test_wrong_recipient_denied(self, store):
        _seeded(store)
        cap = create_capsule(
            store, issuer=_auth(), scope_id="sA",
            recipient_id="agent:bob", members=_members(),
            expires_us=T0 + 10_000, now_us=T0,
        )
        with pytest.raises(VerbatimError) as ei:
            open_capsule(
                store, cap["capsule_id"], recipient=_auth("human:carol"),
                now_us=T0 + 1,
            )
        assert ei.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    def test_expired_capsule_denied_and_marked(self, store):
        _seeded(store)
        cap = create_capsule(
            store, issuer=_auth(), scope_id="sA",
            recipient_id="agent:bob", members=_members(),
            expires_us=T0 + 10, now_us=T0,
        )
        with pytest.raises(VerbatimError) as ei:
            open_capsule(
                store, cap["capsule_id"], recipient=_auth("agent:bob"),
                now_us=T0 + 11,
            )
        assert ei.value.code is ErrorCode.PERMIT_EXPIRED
        assert HandoffRepo(store).get(cap["capsule_id"])["status"] == "expired"


class TestOffline:
    def test_offline_reports_unknown_revocation_short_lease(self, store):
        """V4-47.08: unknown revocation + bounded cached lease."""
        _seeded(store)
        cap = create_capsule(
            store, issuer=_auth(), scope_id="sA",
            recipient_id="agent:bob", members=_members(),
            verbs=("read", "quote"),
            expires_us=T0 + 30 * 24 * 3600 * 1_000_000,  # 30d
            portable=True, now_us=T0,
        )
        import json

        snap = json.loads(
            HandoffRepo(store).get(cap["capsule_id"])["snapshot_json"]
        )
        view = present_offline(snap, now_us=T0 + 60_000_000)
        assert view["valid"] is True
        assert view["revocation_state"] == "unknown-offline"
        # Cached lease is bounded far inside the capsule expiry.
        assert view["cached_authorization_until"] <= T0 + 3_600_000_000
        assert "act" in view["prohibited"]
        assert "hydrate" in view["prohibited"]

    def test_offline_expired_invalid(self, store):
        _seeded(store)
        cap = create_capsule(
            store, issuer=_auth(), scope_id="sA",
            recipient_id="agent:bob", members=_members(),
            expires_us=T0 + 10, portable=True, now_us=T0,
        )
        import json

        snap = json.loads(
            HandoffRepo(store).get(cap["capsule_id"])["snapshot_json"]
        )
        view = present_offline(snap, now_us=T0 + 11)
        assert view["valid"] is False
        assert view["expired"] is True
