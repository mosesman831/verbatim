"""Kernel tests (SPEC_V4 §08–§09): leases, verified reads, disclosure
linearization, dispatch delegation, invalidation, metadata review.

All fixtures run on a real ``Store.create`` (v4 schema). Time is injected
through the ``now_us`` parameters so lease/permit expiry is deterministic.
"""

from __future__ import annotations

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v4 import (
    DispatchPermit,
    EvidenceLocator,
    PurposeTag,
)
from verbatim.governance import bump_epoch, create_grant, revoke_grant
from verbatim.kernel import Kernel
from verbatim.security.quarantine import open_quarantine
from verbatim.storage import repos_v4

from .conftest import T0, add_source, add_span, add_view, caller, grant

PAYLOAD = b"the quick brown fox jumps over the lazy dog"


def _src(store, conn, sid="scope:a", oid="src:s1", payload=PAYLOAD):
    return add_source(conn, store, sid, oid, payload)


def _loc(oid, rev=1, start=0, end=len(PAYLOAD), view=None):
    return EvidenceLocator(
        object_id=oid, revision=rev, start_byte=start, end_byte=end,
        view_id=view,
    )


# ----------------------------------------------------------------------
# resolve_access — lease binding, expiry, indistinguishable denial
# ----------------------------------------------------------------------


def test_lease_binds_caller_scope_purpose_epochs(store, kernel, seeded):
    with store.tx() as conn:
        _src(store, conn)
        grant(conn, "scope:a", "human:alice", ["read", "quote"])
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller(), "quote", "recall", ["scope:a"],
            object_refs=[("source", "src:s1", 1)], now_us=T0,
        )
    assert not lease.denied
    assert lease.caller_id == "human:alice"
    assert lease.verb.value == "quote"
    assert lease.purpose == "recall"
    assert lease.scope_ids == ("scope:a",)
    assert lease.object_refs == (("src:s1", 1),)
    assert lease.epoch_vector == {"scope:a": 0}
    assert lease.issued_us == T0
    # V4-09.08-class default: one second.
    assert lease.expires_us - lease.issued_us == 1_000_000


def test_lease_expiry_blocks_reads(store, kernel, seeded):
    with store.tx() as conn:
        _src(store, conn)
        grant(conn, "scope:a", "human:alice", ["quote"])
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller(), "quote", "recall", ["scope:a"], now_us=T0
        )
        with pytest.raises(VerbatimError) as exc:
            kernel.read_verified(
                conn, lease, [_loc("src:s1")], now_us=T0 + 2_000_000
            )
        assert exc.value.code == ErrorCode.PERMIT_EXPIRED


def test_resolve_denied_indistinguishable(store, kernel, seeded):
    """Absent scope and unauthorized scope produce identical denials."""
    with store.tx() as conn:
        grant(conn, "scope:a", "human:alice", ["read"])
    with store.read() as conn:
        no_grant = kernel.resolve_access(
            conn, caller(), "read", "recall", ["scope:b"], now_us=T0
        )
        no_scope = kernel.resolve_access(
            conn, caller(), "read", "recall", ["scope:nope"], now_us=T0
        )
        no_verb = kernel.resolve_access(
            conn, caller(), "quote", "recall", ["scope:a"], now_us=T0
        )
    for lease in (no_grant, no_scope, no_verb):
        assert lease.denied
        # Denial carries no store-derived detail: no epochs, no refs,
        # no which-scope-failed signal — only the caller's inputs echo back.
        assert lease.epoch_vector == {}
        assert lease.object_refs == ()
    # Absent vs unauthorized scope: identical response shape — the only
    # difference is the caller-supplied scope name echoed back.
    shape = lambda l: (l.denied, l.verb, l.purpose, l.epoch_vector, l.object_refs)
    assert shape(no_grant) == shape(no_scope)


def test_cross_scope_requires_per_scope_purpose_auth(store, kernel, seeded):
    """V4-10.05/11.04: every contributing scope must authorize the purpose."""
    with store.tx() as conn:
        grant(conn, "scope:a", "human:alice", ["read"], purposes=["recall"])
        # scope:b grant exists but not for the requested purpose.
        grant(conn, "scope:b", "human:alice", ["read"], purposes=["review"])
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller(), "read", "recall", ["scope:a", "scope:b"],
            now_us=T0,
        )
    assert lease.denied
    assert lease.epoch_vector == {}


def test_pinned_epoch_fences_stale_caller(store, kernel, seeded):
    with store.tx() as conn:
        grant(conn, "scope:a", "human:alice", ["read"])
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            kernel.resolve_access(
                conn, caller(epoch=7), "read", "recall", ["scope:a"],
                now_us=T0,
            )
        assert exc.value.code == ErrorCode.STALE_EPOCH


def test_resolve_requires_explicit_scopes(store, kernel, seeded):
    """V4-10.04: an empty scope filter is never global access."""
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            kernel.resolve_access(
                conn, caller(), "read", "recall", [], now_us=T0
            )
        assert exc.value.code == ErrorCode.VALIDATION


def test_unregistered_purpose_is_caller_bug(store, kernel, seeded):
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            kernel.resolve_access(
                conn, caller(), "read", "not-a-purpose", ["scope:a"],
                now_us=T0,
            )
        assert exc.value.code == ErrorCode.VALIDATION


# ----------------------------------------------------------------------
# object-level constraints (V4-08.01)
# ----------------------------------------------------------------------


def test_quarantined_object_denied(store, kernel, seeded):
    with store.tx() as conn:
        _src(store, conn)
        grant(conn, "scope:a", "human:alice", ["quote"])
        open_quarantine(
            conn, ("source", "src:s1", 1), ["attack_risk:blocked"], [],
            scope_id="scope:a",
        )
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller(), "quote", "recall", ["scope:a"],
            object_refs=[("source", "src:s1", 1)], now_us=T0,
        )
        assert lease.denied  # a hold resolves like absence — no detail


def test_held_span_source_denies_span_read(store, kernel, seeded):
    """Cascade: a hold on the covering source withholds its span."""
    with store.tx() as conn:
        _src(store, conn)
        add_span(conn, store, "span:s1", "src:s1", 1, 0, 9, PAYLOAD)
        grant(conn, "scope:a", "human:alice", ["quote"])
        open_quarantine(
            conn, ("source", "src:s1", 1), ["rule:test"], [],
            scope_id="scope:a",
        )
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller(), "quote", "recall", ["scope:a"],
            object_refs=[("span", "span:s1", 1)], now_us=T0,
        )
        assert lease.denied


def test_suppressed_object_denied(store, kernel, seeded):
    """A purge in a suppressing state withholds the target."""
    with store.tx() as conn:
        _src(store, conn)
        grant(conn, "scope:a", "human:alice", ["quote"])
        conn.execute(
            "INSERT INTO purges (purge_id, selection_digest, scope_id,"
            " state, requested_us) VALUES ('purge:1', X'00', 'scope:a',"
            " 'suppressed', ?)",
            (T0,),
        )
        conn.execute(
            "INSERT INTO purge_targets (purge_id, object_kind, object_id)"
            " VALUES ('purge:1', 'source', 'src:s1')",
        )
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller(), "quote", "recall", ["scope:a"],
            object_refs=[("source", "src:s1", 1)], now_us=T0,
        )
        assert lease.denied


def test_registry_lifecycle_gate(store, kernel, seeded):
    """v4 objects.disposition gates: held objects deny indistinguishably."""
    with store.tx() as conn:
        _src(store, conn)
        grant(conn, "scope:a", "human:alice", ["quote"])
        repos_v4.insert(
            conn, "objects",
            {"object_id": "src:s1", "kind": "source", "scope_id": "scope:a",
             "current_revision": 1, "disposition": "held",
             "created_event": 0},
        )
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller(), "quote", "recall", ["scope:a"],
            object_refs=[("source", "src:s1", 1)], now_us=T0,
        )
        assert lease.denied


# ----------------------------------------------------------------------
# read_verified — verb gate, integrity, group verdict
# ----------------------------------------------------------------------


def test_quote_read_returns_verified_bytes(store, kernel, seeded):
    with store.tx() as conn:
        _src(store, conn)
        grant(conn, "scope:a", "human:alice", ["quote"])
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller(), "quote", "recall", ["scope:a"], now_us=T0
        )
        slices = kernel.read_verified(
            conn, lease, [_loc("src:s1", start=0, end=9)], now_us=T0
        )
    assert len(slices) == 1
    s = slices[0]
    assert s.data == b"the quick"
    assert s.verification == "verified"
    assert s.algorithm == "hmac-sha256"
    # The slice digest is domain-separated, not a bare content HMAC.
    assert s.digest and s.digest != store.hmac(s.data).hex()


def test_read_verb_yields_metadata_only(store, kernel, seeded):
    """V4-08.02: read permits metadata + approved views, never raw bytes."""
    with store.tx() as conn:
        _src(store, conn)
        grant(conn, "scope:a", "human:alice", ["read"])
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller(), "read", "recall", ["scope:a"], now_us=T0
        )
        slices = kernel.read_verified(
            conn, lease, [_loc("src:s1")], now_us=T0
        )
    assert slices[0].data == b""
    assert slices[0].verification == "metadata_only"
    # Authorized metadata still flows.
    assert slices[0].provenance["byte_length"] == len(PAYLOAD)


def test_read_verb_serves_derived_view_bytes(store, kernel, seeded):
    with store.tx() as conn:
        _src(store, conn)
        add_view(conn, store, "src:s1", 1, "view:norm", b"normalized text")
        grant(conn, "scope:a", "human:alice", ["read"])
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller(), "read", "recall", ["scope:a"], now_us=T0
        )
        slices = kernel.read_verified(
            conn,
            lease,
            [_loc("src:s1", start=0, end=10, view="view:norm")],
            now_us=T0,
        )
    assert slices[0].data == b"normalized"
    assert slices[0].verification == "verified"


def test_primary_view_under_read_is_metadata_only(store, kernel, seeded):
    """A 'primary' view is still canonical bytes — quote required."""
    with store.tx() as conn:
        _src(store, conn)
        conn.execute(
            "INSERT INTO source_views"
            "(source_id, revision, view_id, media_type, view_kind,"
            " transformer_revision, locator_json)"
            " VALUES ('src:s1', 1, 'primary', 'text/plain', 'primary',"
            " NULL, '{}')",
        )
        grant(conn, "scope:a", "human:alice", ["read"])
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller(), "read", "recall", ["scope:a"], now_us=T0
        )
        slices = kernel.read_verified(
            conn,
            lease,
            [_loc("src:s1", start=0, end=4, view="primary")],
            now_us=T0,
        )
    assert slices[0].data == b""
    assert slices[0].verification == "metadata_only"


def test_tampered_payload_fails_closed(store, kernel, seeded):
    """V4-08.03/C03: mutated bytes are corruption, never content."""
    with store.tx() as conn:
        _src(store, conn)
        grant(conn, "scope:a", "human:alice", ["quote"])
    with store.tx() as conn:
        conn.execute(
            "UPDATE source_revisions SET payload = ?"
            " WHERE source_id = 'src:s1' AND revision = 1",
            (b"attacker controlled bytes",),
        )
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller(), "quote", "recall", ["scope:a"], now_us=T0
        )
        with pytest.raises(VerbatimError) as exc:
            kernel.read_verified(conn, lease, [_loc("src:s1")], now_us=T0)
        assert exc.value.code == ErrorCode.STORE_CORRUPT


def test_legacy_null_digest_is_unverified(store, kernel, seeded):
    """V4-13.05/C04: NULL digest → legacy_unverified, never verified."""
    with store.tx() as conn:
        _src(store, conn)
        add_view(
            conn, store, "src:s1", 1, "view:old", b"legacy bytes",
            legacy_digest=True,
        )
        grant(conn, "scope:a", "human:alice", ["read", "quote"])
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller(), "quote", "recall", ["scope:a"], now_us=T0
        )
        slices = kernel.read_verified(
            conn,
            lease,
            [_loc("src:s1", start=0, end=6, view="view:old")],
            now_us=T0,
        )
    assert slices[0].data == b"legacy"
    assert slices[0].verification == "legacy_unverified"


def test_span_read_exact_range(store, kernel, seeded):
    """Span locators must equal the stored range — excerpt_hmac binds it."""
    with store.tx() as conn:
        _src(store, conn)
        add_span(conn, store, "span:s1", "src:s1", 1, 4, 9, PAYLOAD)
        grant(conn, "scope:a", "human:alice", ["quote"])
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller(), "quote", "recall", ["scope:a"], now_us=T0
        )
        ok = kernel.read_verified(
            conn, lease, [_loc("span:s1", 1, 4, 9)], now_us=T0
        )
        assert ok[0].data == b"quick"
        assert ok[0].verification == "verified"
        with pytest.raises(VerbatimError) as exc:
            kernel.read_verified(
                conn, lease, [_loc("span:s1", 1, 0, 9)], now_us=T0
            )
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_group_verdict_no_partial_subsets(store, kernel, seeded):
    """V4-08.05: one denied locator invalidates the whole read."""
    with store.tx() as conn:
        _src(store, conn)
        _src(store, conn, oid="src:s2", payload=b"second evidence")
        grant(conn, "scope:a", "human:alice", ["quote"])
        open_quarantine(
            conn, ("source", "src:s2", 1), ["rule:test"], [],
            scope_id="scope:a",
        )
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller(), "quote", "recall", ["scope:a"], now_us=T0
        )
        with pytest.raises(VerbatimError) as exc:
            kernel.read_verified(
                conn, lease, [_loc("src:s1"), _loc("src:s2")], now_us=T0
            )
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_locator_outside_lease_scope_denied(store, kernel, seeded):
    """A lease for scope:a must not read objects living in scope:b."""
    with store.tx() as conn:
        _src(store, conn)
        _src(store, conn, sid="scope:b", oid="src:b1")
        grant(conn, "scope:a", "human:alice", ["quote"])
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller(), "quote", "recall", ["scope:a"], now_us=T0
        )
        with pytest.raises(VerbatimError) as exc:
            kernel.read_verified(conn, lease, [_loc("src:b1")], now_us=T0)
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_denied_lease_reads_deny(store, kernel, seeded):
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller(), "quote", "recall", ["scope:a"], now_us=T0
        )
        assert lease.denied
        with pytest.raises(VerbatimError) as exc:
            kernel.read_verified(conn, lease, [_loc("src:s1")], now_us=T0)
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_epoch_drift_stales_lease(store, kernel, seeded):
    with store.tx() as conn:
        _src(store, conn)
        grant(conn, "scope:a", "human:alice", ["quote"])
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller(), "quote", "recall", ["scope:a"], now_us=T0
        )
    with store.tx() as conn:
        bump_epoch(conn, "scope:a")
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            kernel.read_verified(conn, lease, [_loc("src:s1")], now_us=T0)
        assert exc.value.code == ErrorCode.STALE_EPOCH


# ----------------------------------------------------------------------
# seal_delivery — disclosure linearization (V4-09.06/07/08)
# ----------------------------------------------------------------------


def _sealed_lease(store, kernel, conn, verbs=("quote",)):
    grant(conn, "scope:a", "human:alice", list(verbs))
    return kernel.resolve_access(
        conn, caller(), verbs[0], "recall", ["scope:a"], now_us=T0
    )


def test_seal_delivery_persists_permit(store, kernel, seeded):
    with store.tx() as conn:
        _src(store, conn)
        lease = _sealed_lease(store, kernel, conn)
        permit = kernel.seal_delivery(
            conn, lease, b"serialized-pack", {"src:s1": 1}, now_us=T0
        )
    assert permit.state == "sealed"
    assert permit.expires_us - permit.issued_us == 1_000_000
    assert permit.dependency_versions == {"src:s1": 1}
    assert permit.epoch_vector == {"scope:a": 0}
    # Canonical egress digest form — broker-compatible binding.
    assert permit.payload_digest.startswith("sha256:")
    with store.read() as conn:
        row = repos_v4.get(
            conn, "delivery_permits", {"permit_id": permit.permit_id}
        )
    assert row is not None and row["state"] == "sealed"


def test_seal_rechecks_revocation(store, kernel, seeded):
    """V4-09.06/07: revoke between resolve and seal → no permit."""
    with store.tx() as conn:
        _src(store, conn)
        gid = grant(conn, "scope:a", "human:alice", ["quote"])
        lease = kernel.resolve_access(
            conn, caller(), "quote", "recall", ["scope:a"], now_us=T0
        )
    with store.tx() as conn:
        revoke_grant(conn, gid)
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            kernel.seal_delivery(
                conn, lease, b"pack", {}, now_us=T0
            )
        # Epoch drift (STALE_EPOCH) or the revoked grant itself denies.
        assert exc.value.code in (
            ErrorCode.STALE_EPOCH,
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
        )
    with store.read() as conn:
        assert repos_v4.query(conn, "delivery_permits") == []


def test_seal_rechecks_late_quarantine(store, kernel, seeded):
    """A hold opened after resolve blocks the seal inside the same tx."""
    with store.tx() as conn:
        _src(store, conn)
        lease = _sealed_lease(store, kernel, conn, verbs=("quote",))
        lease = kernel.resolve_access(
            conn, caller(), "quote", "recall", ["scope:a"],
            object_refs=[("source", "src:s1", 1)], now_us=T0,
        )
    with store.tx() as conn:
        open_quarantine(
            conn, ("source", "src:s1", 1), ["rule:test"], [],
            scope_id="scope:a",
        )
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            kernel.seal_delivery(conn, lease, b"pack", {}, now_us=T0)
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_stale_dependency_blocks_seal(store, kernel, seeded):
    with store.tx() as conn:
        _src(store, conn)
        lease = _sealed_lease(store, kernel, conn)
        with pytest.raises(VerbatimError) as exc:
            kernel.seal_delivery(
                conn, lease, b"pack", {"src:s1": 7}, now_us=T0
            )
        assert exc.value.code == ErrorCode.STALE_DEPENDENCY


def test_expired_lease_cannot_seal(store, kernel, seeded):
    with store.tx() as conn:
        _src(store, conn)
        lease = _sealed_lease(store, kernel, conn)
        with pytest.raises(VerbatimError) as exc:
            kernel.seal_delivery(
                conn, lease, b"pack", {}, now_us=T0 + 2_000_000
            )
        assert exc.value.code == ErrorCode.PERMIT_EXPIRED


def test_expired_permit_denied(store, kernel, seeded):
    """V4-09.08: an unhanded permit expires and needs reauthorization."""
    with store.tx() as conn:
        _src(store, conn)
        lease = _sealed_lease(store, kernel, conn)
        permit = kernel.seal_delivery(conn, lease, b"pack", {}, now_us=T0)
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            kernel.verify_delivery(
                conn, permit, now_us=T0 + 2_000_000
            )
        assert exc.value.code == ErrorCode.PERMIT_EXPIRED
        # A different caller's forged permit id denies indistinguishably.
        with pytest.raises(VerbatimError) as exc2:
            kernel.verify_delivery(conn, "dpermit:nonexistent", now_us=T0)
        assert exc2.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_mark_delivered_lifecycle(store, kernel, seeded):
    with store.tx() as conn:
        _src(store, conn)
        lease = _sealed_lease(store, kernel, conn)
        permit = kernel.seal_delivery(conn, lease, b"pack", {}, now_us=T0)
        done = kernel.mark_delivered(conn, permit.permit_id, now_us=T0)
        assert done.state == "delivered"
        with pytest.raises(VerbatimError):
            kernel.verify_delivery(conn, permit.permit_id, now_us=T0)


# ----------------------------------------------------------------------
# open_dispatch — broker delegation
# ----------------------------------------------------------------------


def test_open_dispatch_unavailable_without_broker(store, kernel, seeded):
    """No provisioned broker → honest CAPABILITY_UNAVAILABLE, no permit."""
    with store.tx() as conn:
        _src(store, conn)
        lease = _sealed_lease(store, kernel, conn)
        permit = kernel.seal_delivery(conn, lease, b"pack", {}, now_us=T0)
        with pytest.raises(VerbatimError) as exc:
            kernel.open_dispatch(
                conn, permit, recipient="endpoint:x", max_spend=0.5,
                now_us=T0,
            )
        assert exc.value.code == ErrorCode.CAPABILITY_UNAVAILABLE


def test_open_dispatch_delegates_and_marks_delivered(store, kernel, seeded):
    """A provisioned broker receives the digest-bound handoff and the
    delivery permit records 'delivered' in the same transaction."""
    captured = {}

    class FakeBroker:
        def open_dispatch(self, conn, **kwargs):
            captured.update(kwargs)
            return DispatchPermit(
                permit_id="xpermit:1",
                recipient=kwargs["recipient"],
                purpose=kwargs["purpose"],
                payload_digest=kwargs["payload_digest"],
                scope_ids=kwargs["scope_ids"],
                consent_refs=kwargs["consent_refs"],
                reservation_id="rsv:1",
                max_spend=kwargs["max_spend"],
                issued_us=T0,
                expires_us=kwargs["deadline_us"],
            )

    with store.tx() as conn:
        _src(store, conn)
        lease = _sealed_lease(store, kernel, conn)
        permit = kernel.seal_delivery(conn, lease, b"pack", {}, now_us=T0)
        dispatch = kernel.open_dispatch(
            conn, permit, recipient="endpoint:x", max_spend=0.25,
            consent_refs=["consent:1"], payload_digest=permit.payload_digest,
            broker=FakeBroker(), now_us=T0,
        )
        assert isinstance(dispatch, DispatchPermit)
        assert captured["payload_digest"] == permit.payload_digest
        assert captured["payload_digest"].startswith("sha256:")
        assert captured["caller"] == "human:alice"
        assert captured["scope_ids"] == ("scope:a",)
        assert captured["recipient"] == "endpoint:x"
        row = repos_v4.get(
            conn, "delivery_permits", {"permit_id": permit.permit_id}
        )
        assert row["state"] == "delivered"


def test_open_dispatch_rejects_digest_mismatch(store, kernel, seeded):
    with store.tx() as conn:
        _src(store, conn)
        lease = _sealed_lease(store, kernel, conn)
        permit = kernel.seal_delivery(conn, lease, b"pack", {}, now_us=T0)
        with pytest.raises(VerbatimError) as exc:
            kernel.open_dispatch(
                conn, permit, recipient="endpoint:x", max_spend=0.5,
                payload_digest="sha256:" + "00" * 32, now_us=T0,
            )
        assert exc.value.code == ErrorCode.VALIDATION


# ----------------------------------------------------------------------
# derive_inputs — verified bundle + inherited restrictions
# ----------------------------------------------------------------------


def test_derive_inputs_full_flow(store, kernel, seeded):
    """V4-10.07: derived audience/purposes = ∩ of contributing parents."""
    with store.tx() as conn:
        _src(store, conn)
        _src(store, conn, sid="scope:b", oid="src:b1", payload=b"b-bytes")
        gid = create_grant(
            conn, scope_id="scope:a", principal_id="agent:prod",
            verbs=["derive"], purposes=["derive"], issuer_id="human:alice",
        )
        create_grant(
            conn, scope_id="scope:a", principal_id="agent:prod",
            verbs=["quote"], purposes=["derive", "recall"],
            issuer_id="human:alice",
        )
        create_grant(
            conn, scope_id="scope:b", principal_id="agent:prod",
            verbs=["quote"], purposes=["derive"], issuer_id="human:alice",
        )
        create_grant(
            conn, scope_id="scope:a", principal_id="human:bob",
            verbs=["read"], purposes=["derive"], issuer_id="human:alice",
        )
        create_grant(
            conn, scope_id="scope:b", principal_id="human:bob",
            verbs=["read"], purposes=["derive"], issuer_id="human:alice",
        )
        bundle = kernel.derive_inputs(
            conn, gid,
            [("src:s1", 1), ("src:b1", 1)],
            output_audience=["human:bob", "human:mallory"],
            output_purpose="derive",
            now_us=T0,
        )
    assert bundle.producer_id == "agent:prod"
    assert len(bundle.inputs) == 2
    assert {i.data for i in bundle.inputs} == {PAYLOAD, b"b-bytes"}
    assert set(bundle.scope_ids) == {"scope:a", "scope:b"}
    # Purpose intersection: {derive,recall} ∩ {derive} = {derive}.
    assert bundle.allowed_purposes.tag is PurposeTag.SET
    assert bundle.allowed_purposes.values == frozenset({"derive"})
    # Audience intersection: bob authorized in both, mallory in neither.
    assert bundle.effective_audience == ("human:bob",)


def test_derive_does_not_imply_read(store, kernel, seeded):
    """V4-11.05: a derive grant without separate input access denies."""
    with store.tx() as conn:
        _src(store, conn)
        gid = create_grant(
            conn, scope_id="scope:a", principal_id="agent:prod",
            verbs=["derive"], purposes=["derive"], issuer_id="human:alice",
        )
        with pytest.raises(VerbatimError) as exc:
            kernel.derive_inputs(
                conn, gid, [("src:s1", 1)],
                output_purpose="derive", now_us=T0,
            )
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


# ----------------------------------------------------------------------
# invalidate — epoch bumps + obligations
# ----------------------------------------------------------------------


def test_invalidate_bumps_epochs_and_enumerates(store, kernel, seeded):
    with store.tx() as conn:
        _src(store, conn)
        lease = _sealed_lease(store, kernel, conn)
        permit = kernel.seal_delivery(conn, lease, b"pack", {}, now_us=T0)
        report = kernel.invalidate(
            conn,
            {"kind": "revocation", "scope_ids": ["scope:a"],
             "object_refs": [("source", "src:s1", 1)]},
            now_us=T0,
        )
    assert report.event_kind == "revocation"
    assert report.epochs == {"scope:a": 1}
    # The committed permit is an honestly irreversible in-flight disclosure.
    assert report.in_flight_permits == (permit.permit_id,)
    assert report.expired_permits == ()
    assert ("source", "src:s1", 1) in report.affected


def test_invalidate_expires_lapsed_permits(store, kernel, seeded):
    with store.tx() as conn:
        _src(store, conn)
        lease = _sealed_lease(store, kernel, conn)
        permit = kernel.seal_delivery(conn, lease, b"pack", {}, now_us=T0)
        report = kernel.invalidate(
            conn, {"kind": "revocation", "scope_ids": ["scope:a"]},
            now_us=T0 + 2_000_000,
        )
        row = repos_v4.get(
            conn, "delivery_permits", {"permit_id": permit.permit_id}
        )
        assert row["state"] == "expired"
    assert report.expired_permits == (permit.permit_id,)
    assert report.in_flight_permits == ()


def test_invalidate_cascades_to_dependents(store, kernel, seeded):
    with store.tx() as conn:
        _src(store, conn)
        repos_v4.insert(
            conn, "dependency_edges",
            {"child_kind": "claim", "child_id": "claim:c1",
             "child_revision": 2, "parent_kind": "source",
             "parent_id": "src:s1", "parent_revision": 1,
             "role": "evidence", "producer_id": "prod",
             "operation_id": "op:1", "seq": 0},
        )
        report = kernel.invalidate(
            conn, {"kind": "correction", "scope_ids": ["scope:a"],
                   "object_refs": [("source", "src:s1", 1)]},
            now_us=T0,
        )
    kinds = {(k, i) for k, i, _ in report.affected}
    assert ("source", "src:s1") in kinds
    assert ("claim", "claim:c1") in kinds


# ----------------------------------------------------------------------
# review_metadata (V4-08.06)
# ----------------------------------------------------------------------


def test_metadata_review_follows_lease(store, kernel, seeded):
    with store.tx() as conn:
        _src(store, conn)
        grant(conn, "scope:a", "human:alice", ["read"])
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller(), "read", "recall", ["scope:a"], now_us=T0
        )
        reviewed = kernel.review_metadata(
            conn, lease, {"count": 3, "title": "evidence"}, now_us=T0
        )
        assert reviewed.fields == {"count": 3, "title": "evidence"}
        assert reviewed.lease_id == lease.lease_id
        # Metadata scoped outside the lease denies like content.
        with pytest.raises(VerbatimError) as exc:
            kernel.review_metadata(
                conn, lease, {"count": 1}, scope_id="scope:b", now_us=T0
            )
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
        # Expired lease → no metadata.
        with pytest.raises(VerbatimError) as exc2:
            kernel.review_metadata(
                conn, lease, {"count": 1}, now_us=T0 + 2_000_000
            )
        assert exc2.value.code == ErrorCode.PERMIT_EXPIRED
