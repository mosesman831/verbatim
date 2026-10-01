"""Governance tests (SPEC_V3 §08–§11, §46).

Covers the required behaviors: effective-verb authorization with purpose
binding, indistinguishable denial, attenuated delegation + revocation
cascade, pinned-epoch fencing (STALE_EPOCH), capture authorization, the
purpose registry, propagation ledger / blast radius, and the
revocation_notify job handler — all against a real ``Store.create`` in
tmp_path (the tests/conftest.py shim is DDL_V1-only and lacks v3 tables).
"""

from __future__ import annotations

import sqlite3

import pytest

from verbatim.config import config_from_mapping
from verbatim.core.time import now_us
from verbatim.core.types import ErrorCode, JobKind, VerbatimError
from verbatim.core.types_v3 import Propagation
from verbatim.governance import (
    BUILTIN_PURPOSES,
    CallerV3,
    authorize,
    blast_radius,
    bump_epoch,
    capture_authorized,
    check_epoch,
    consent_for_hydration,
    create_grant,
    create_perspective,
    current_epoch,
    delegate_grant,
    effective_verbs,
    get_or_create_perspective,
    get_principal,
    handle_revocation_notify,
    is_registered,
    issue_capture_authorization,
    list_principals,
    list_purposes,
    mark_propagations_revoked,
    propagations_for,
    recipients_of,
    record_propagation,
    register_principal,
    register_purpose,
    require_capture_authorization,
    require_hydration_consent,
    require_purpose,
    resolve_perspective,
    retire_principal,
    revoke_capture_authorization,
    revoke_delegation,
    revoke_grant,
    seed_purposes,
    subjects_of,
)
from verbatim.governance import purposes as purpose_mod
from verbatim.ingest import Ingester
from verbatim.storage import repos_v3
from verbatim.storage.store import Store

US = 1_000_000
T0 = 1_700_000_000_000_000  # fixed past base (2023) for deterministic rows


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "gov.db"))
    yield s
    s.close()


@pytest.fixture
def scope_id(store):
    sid = "scope:gov"
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES (?, 'prof', 'owner')",
            (sid,),
        )
        seed_purposes(conn)
    return sid


@pytest.fixture
def alice(store):
    with store.tx() as conn:
        return register_principal(conn, kind="human", principal_id="human:alice")


def _caller(pid="human:alice", epoch=None):
    return CallerV3(principal_id=pid, epoch=epoch)


def _denied(store, sid, pid="human:alice", verb="read", purpose=None, epoch=None):
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            authorize(conn, _caller(pid, epoch), sid, verb, purpose=purpose)
        return exc.value


# ---------------------------------------------------------------------
# principals (§08)
# ---------------------------------------------------------------------


def test_register_and_get_principal(store):
    with store.tx() as conn:
        pid = register_principal(
            conn,
            kind="agent",
            principal_id="agent:a1",
            display_name="Agent One",
            host_binding="host:hermes",
        )
        row = get_principal(conn, pid)
        assert row["kind"] == "agent"
        assert row["retired"] == 0
        # idempotent identical re-registration
        assert register_principal(conn, kind="agent", principal_id=pid) == pid


def test_register_principal_kind_conflict(store):
    with store.tx() as conn:
        register_principal(conn, kind="human", principal_id="p:x")
        with pytest.raises(VerbatimError) as exc:
            register_principal(conn, kind="agent", principal_id="p:x")
        assert exc.value.code == ErrorCode.VALIDATION


def test_register_all_principal_kinds(store):
    kinds = [
        "human", "agent", "service", "workspace", "organization",
        "external_party",
    ]
    with store.tx() as conn:
        for i, k in enumerate(kinds):
            register_principal(conn, kind=k, principal_id=f"p:{k}")
        rows = list_principals(conn)
        assert {r["kind"] for r in rows} == set(kinds)


def test_retire_principal_revokes_grants(store, scope_id, alice):
    with store.tx() as conn:
        create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, issuer_id=alice,
        )
    with store.tx() as conn:
        receipt = retire_principal(conn, alice)
        assert receipt["retired"] is True
        assert receipt["grants_revoked"] == 1
        assert receipt["scope_epochs"][scope_id] == 1
    err = _denied(store, scope_id, pid=alice)
    assert err.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    # idempotent
    with store.tx() as conn:
        again = retire_principal(conn, alice)
        assert again["already_retired"] is True


def test_retire_missing_principal(store):
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            retire_principal(conn, "p:ghost")
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


# ---------------------------------------------------------------------
# perspectives (§08.03–08.04)
# ---------------------------------------------------------------------


def test_perspective_roundtrip(store, scope_id):
    with store.tx() as conn:
        pid = create_perspective(
            conn,
            scope_id=scope_id,
            asserter="human:alice",
            observer="agent:a1",
            subjects=["human:bob", "human:bob", "ws:team"],
            audience=["ws:team"],
        )
        p = resolve_perspective(conn, pid)
        assert p.asserter == "human:alice"
        assert p.observer == "agent:a1"
        assert p.subjects == ("human:bob", "ws:team")
        assert p.audience == ("ws:team",)
        assert subjects_of(conn, pid) == ("human:bob", "ws:team")


def test_perspective_dedupe(store, scope_id):
    with store.tx() as conn:
        a = get_or_create_perspective(
            conn, scope_id=scope_id, asserter="human:alice",
            subjects=["human:bob"], audience=["ws:team"],
        )
        b = get_or_create_perspective(
            conn, scope_id=scope_id, asserter="human:alice",
            subjects=["human:bob"], audience=["ws:team"],
        )
        assert a == b
        c = get_or_create_perspective(
            conn, scope_id=scope_id, asserter="human:alice",
            subjects=["human:carol"], audience=["ws:team"],
        )
        assert c != a


def test_perspective_none_means_unrecorded(store, scope_id):
    with store.tx() as conn:
        pid = create_perspective(conn, scope_id=scope_id)
        p = resolve_perspective(conn, pid)
        assert p.asserter is None
        assert p.observer is None
        assert p.subjects == ()
        assert p.audience == ()


# ---------------------------------------------------------------------
# purposes (§11.06)
# ---------------------------------------------------------------------


def test_builtin_purposes_seeded(store, scope_id):
    with store.read() as conn:
        names = {r["purpose"] for r in list_purposes(conn)}
    assert names == set(BUILTIN_PURPOSES)
    assert {"recall", "derive", "share", "hydrate", "review", "ingest",
            "admin", "evaluate"} <= names


def test_seed_purposes_idempotent(store, scope_id):
    with store.tx() as conn:
        assert seed_purposes(conn) == 0


def test_register_and_require_purpose(store):
    with store.tx() as conn:
        register_purpose(conn, "debugging", "dev diagnostics")
        assert is_registered(conn, "debugging")
        require_purpose(conn, "debugging")
        with pytest.raises(VerbatimError) as exc:
            require_purpose(conn, "telepathy")
        assert exc.value.code == ErrorCode.VALIDATION


def test_unregistered_purpose_grant_rejected(store, scope_id, alice):
    """governance_strict: free-text purposes are refused (§11.06)."""
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            create_grant(
                conn, scope_id=scope_id, principal_id=alice,
                verbs={"read"}, purposes={"telepathy"}, issuer_id=alice,
            )
        assert exc.value.code == ErrorCode.VALIDATION
        # strict=False is the governance_strict=off escape hatch
        gid = create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, purposes={"telepathy"}, issuer_id=alice,
            strict=False,
        )
        assert gid


# ---------------------------------------------------------------------
# grants: authorize (§09.01–09.09)
# ---------------------------------------------------------------------


def test_authorize_allows_covering_grant(store, scope_id, alice):
    with store.tx() as conn:
        create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read", "quote"}, issuer_id=alice,
        )
    with store.read() as conn:
        assert authorize(conn, _caller(), scope_id, "read") is None
        assert authorize(conn, _caller(), scope_id, "quote") is None


def test_verb_containment_not_equality(store, scope_id, alice):
    """Grant {read,quote} covers read and quote but not share (§09)."""
    with store.tx() as conn:
        create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read", "quote"}, issuer_id=alice,
        )
    err = _denied(store, scope_id, verb="share")
    assert err.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_authorize_denies_without_grant(store, scope_id, alice):
    err = _denied(store, scope_id)
    assert err.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_denied_and_nonexistent_identical(store, scope_id, alice):
    """Absent vs forbidden produce one public error class (§09.09/§10.05)."""
    with store.tx() as conn:
        create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, issuer_id=alice,
        )
    with store.read() as conn:
        # denied: no grant covers 'share'
        with pytest.raises(VerbatimError) as e1:
            authorize(conn, _caller(), scope_id, "share")
        # nonexistent: scope with no rows at all
        with pytest.raises(VerbatimError) as e2:
            authorize(conn, _caller(), "scope:ghost", "read")
        # nonexistent principal
        with pytest.raises(VerbatimError) as e3:
            authorize(conn, _caller("human:ghost"), scope_id, "read")
        # nonexistent object_ref never dereferenced → same denial
        with pytest.raises(VerbatimError) as e4:
            authorize(
                conn, _caller(), scope_id, "share",
                object_ref=("claim", "c:missing", 1),
            )
    for e in (e1, e2, e3, e4):
        assert e.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
        assert e.value.message == e1.value.message


def test_expired_grant_denied(store, scope_id, alice):
    with store.tx() as conn:
        create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, issuer_id=alice,
            issued_us=T0, expires_us=T0 + 10,
        )
    assert _denied(store, scope_id).code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_revoked_grant_denied(store, scope_id, alice):
    with store.tx() as conn:
        gid = create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, issuer_id=alice,
        )
    with store.read() as conn:
        authorize(conn, _caller(), scope_id, "read")
    with store.tx() as conn:
        revoke_grant(conn, gid)
    assert _denied(store, scope_id).code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_purpose_binding(store, scope_id, alice):
    """A purpose-bound grant answers only declared purposes (§09.08)."""
    with store.tx() as conn:
        create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, purposes={"recall"}, issuer_id=alice,
        )
    with store.read() as conn:
        assert authorize(
            conn, _caller(), scope_id, "read", purpose="recall"
        ) is None
    assert _denied(store, scope_id, purpose="derive").code == (
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    )
    # declaring no purpose never satisfies a purpose-bound grant
    assert _denied(store, scope_id).code == (
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    )


def test_default_purposes_grant_covers_any(store, scope_id, alice):
    """Compat (V4-05.04): omitting ``purposes`` keeps the historical
    unrestricted root grant — but the row now persists an explicit
    ``purpose_tag='any'``, not the ambiguous untagged empty set (F4-03).
    Passing an *empty iterable* is an explicit empty SET and normalizes
    to NONE — see test_purpose_constraints.py."""
    with store.tx() as conn:
        gid = create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, issuer_id=alice,
        )
        row = repos_v3.get(conn, "grants_v3", {"grant_id": gid})
        assert row["purpose_tag"] == "any"
        assert repos_v3.json_field(row, "purposes_json") == []
    with store.read() as conn:
        assert authorize(
            conn, _caller(), scope_id, "read", purpose="derive"
        ) is None
        assert authorize(conn, _caller(), scope_id, "read") is None


def test_other_scope_grant_does_not_leak(store, scope_id, alice):
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES ('scope:other', 'prof', 'owner')"
        )
        create_grant(
            conn, scope_id="scope:other", principal_id=alice,
            verbs={"read"}, issuer_id=alice,
        )
    assert _denied(store, scope_id).code == (
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    )


def test_grant_epoch_pins_visibility(store, scope_id, alice):
    """A grant stamped at a future epoch is invisible to a current caller."""
    with store.tx() as conn:
        repos_v3.insert(conn, "grants_v3", {
            "grant_id": "g:future", "scope_id": scope_id,
            "principal_id": alice, "verbs_json": ["read"],
            "purposes_json": [], "caveats_json": [], "delegation_depth": 0,
            "issuer_id": alice, "issued_us": T0, "epoch": 99,
        })
    assert _denied(store, scope_id).code == (
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    )


def test_invalid_verb_is_validation(store, scope_id, alice):
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            authorize(conn, _caller(), scope_id, "teleport")
        assert exc.value.code == ErrorCode.VALIDATION


# ---------------------------------------------------------------------
# epochs (§09.04, §46)
# ---------------------------------------------------------------------


def test_epoch_bump_and_stale_pin(store, scope_id):
    with store.read() as conn:
        assert current_epoch(conn, scope_id) == 0
    with store.tx() as conn:
        assert bump_epoch(conn, scope_id) == 1
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            check_epoch(conn, 0, scope_id)
        assert exc.value.code == ErrorCode.STALE_EPOCH
        assert exc.value.retryable is True
        # current pin passes
        check_epoch(conn, 1, scope_id)
        check_epoch(conn, None, scope_id)


def test_authorize_stale_epoch_after_revocation(store, scope_id, alice):
    """caller.epoch < current after a revocation bump → STALE_EPOCH."""
    with store.tx() as conn:
        gid = create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, issuer_id=alice,
        )
    with store.read() as conn:
        authorize(conn, _caller(epoch=0), scope_id, "read")
    with store.tx() as conn:
        revoke_grant(conn, gid)
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            authorize(conn, _caller(epoch=0), scope_id, "read")
        assert exc.value.code == ErrorCode.STALE_EPOCH
        # rebind at the new epoch: evaluates current state (grant is dead)
        with pytest.raises(VerbatimError) as exc2:
            authorize(conn, _caller(epoch=1), scope_id, "read")
        assert exc2.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_stale_epoch_beats_grant_eval(store, scope_id, alice):
    """Epoch fencing precedes grant evaluation (§09.04)."""
    with store.tx() as conn:
        create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, issuer_id=alice,
        )
        bump_epoch(conn, scope_id)
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            effective_verbs(conn, _caller(epoch=0), scope_id)
        assert exc.value.code == ErrorCode.STALE_EPOCH
        assert effective_verbs(conn, _caller(epoch=1), scope_id) == {"read"}


# ---------------------------------------------------------------------
# delegation (§08.02, §09.03)
# ---------------------------------------------------------------------


def _parent(store, sid, alice, verbs=frozenset({"read", "quote"}), depth=1):
    with store.tx() as conn:
        return create_grant(
            conn, scope_id=sid, principal_id=alice,
            verbs=verbs, purposes={"recall"}, caveats=("no-export",),
            issuer_id=alice, delegation_depth=depth,
        )


def test_delegate_grant_attenuates(store, scope_id, alice):
    parent = _parent(store, scope_id, alice)
    with store.tx() as conn:
        did, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
            verbs={"read"}, purposes={"recall"}, caveats=("no-export",),
        )
        link = repos_v3.get(conn, "delegations", {"delegation_id": did})
        assert link["parent_grant_id"] == parent
        assert link["child_grant_id"] == child
        assert link["delegate_id"] == "agent:a1"
        grow = repos_v3.get(conn, "grants_v3", {"grant_id": child})
        assert grow["delegation_depth"] == 0  # decremented from parent's 1
        assert repos_v3.json_field(grow, "verbs_json") == ["read"]


def test_delegate_widening_verbs_rejected(store, scope_id, alice):
    """Child verbs ⊄ parent → rejected at creation (§09.03)."""
    parent = _parent(store, scope_id, alice)
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            delegate_grant(
                conn, parent_grant_id=parent, delegate_id="agent:a1",
                verbs={"read", "share"},
            )
        assert exc.value.code == ErrorCode.VALIDATION


def test_delegate_widening_purposes_rejected(store, scope_id, alice):
    parent = _parent(store, scope_id, alice)
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            delegate_grant(
                conn, parent_grant_id=parent, delegate_id="agent:a1",
                verbs={"read"}, purposes={"recall", "derive"},
            )
        assert exc.value.code == ErrorCode.VALIDATION


def test_delegate_dropping_caveats_rejected(store, scope_id, alice):
    """V4-11.03 (compat note): caveats are conjunctive restrictions —
    the child must retain every parent caveat. The pre-v4 contract ran
    ordinary subset logic here, which let a delegation *drop* a parent
    restriction: silent authority expansion. Dropping now rejects;
    adding restrictions is attenuation and is allowed."""
    parent = _parent(store, scope_id, alice)
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            delegate_grant(
                conn, parent_grant_id=parent, delegate_id="agent:a1",
                verbs={"read"}, caveats=(),
            )
        assert exc.value.code == ErrorCode.VALIDATION
        # adding a restriction strengthens and is permitted
        _, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
            verbs={"read"}, caveats=("no-export", "extra-caveat"),
        )
        grow = repos_v3.get(conn, "grants_v3", {"grant_id": child})
        assert repos_v3.json_field(grow, "caveats_json") == [
            "no-export", "extra-caveat",
        ]


def test_delegate_depth_exhausted(store, scope_id, alice):
    """A depth-0 grant carries no delegation budget (§09.13 overflow)."""
    parent = _parent(store, scope_id, alice, depth=0)
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            delegate_grant(
                conn, parent_grant_id=parent, delegate_id="agent:a1",
                verbs={"read"},
            )
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_delegate_missing_parent_denied(store, scope_id):
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            delegate_grant(
                conn, parent_grant_id="g:ghost", delegate_id="agent:a1",
            )
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_delegated_grant_authorizes(store, scope_id, alice):
    """delegate_id's depth>0 grant is effective while the chain lives."""
    parent = _parent(store, scope_id, alice)
    with store.tx() as conn:
        delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
            verbs={"read"},
        )
    with store.read() as conn:
        assert authorize(
            conn, _caller("agent:a1"), scope_id, "read", purpose="recall"
        ) is None
        # the attenuated grant never widened beyond 'read'
        with pytest.raises(VerbatimError):
            authorize(conn, _caller("agent:a1"), scope_id, "quote")
        # and purposes stay bound
        with pytest.raises(VerbatimError):
            authorize(
                conn, _caller("agent:a1"), scope_id, "read",
                purpose="derive",
            )


def test_revoke_parent_cascades_to_children(store, scope_id, alice):
    """A child never outlives its parent's revocation (§09/§10)."""
    parent = _parent(store, scope_id, alice, depth=2)
    with store.tx() as conn:
        _, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
            verbs={"read"},
        )
        _, grandchild = delegate_grant(
            conn, parent_grant_id=child, delegate_id="agent:a2",
            verbs={"read"},
        )
    with store.read() as conn:
        authorize(conn, _caller("agent:a2"), scope_id, "read",
                  purpose="recall")
    with store.tx() as conn:
        receipt = revoke_grant(conn, parent)
        assert receipt["grants_revoked"] == 3  # parent + child + grandchild
        assert receipt["delegations_revoked"] == 2
    for pid in ("agent:a1", "agent:a2"):
        assert _denied(store, scope_id, pid=pid).code == (
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
        )


def test_revoke_delegation_kills_child(store, scope_id, alice):
    parent = _parent(store, scope_id, alice)
    with store.tx() as conn:
        did, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
            verbs={"read"},
        )
    with store.tx() as conn:
        receipt = revoke_delegation(conn, did)
        assert receipt["grants_revoked"] == 1
        dead = repos_v3.get(conn, "grants_v3", {"grant_id": child})
        assert dead["revoked_us"] is not None
    assert _denied(store, scope_id, pid="agent:a1").code == (
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    )


def test_revoke_grant_idempotent(store, scope_id, alice):
    with store.tx() as conn:
        gid = create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, issuer_id=alice,
        )
        first = revoke_grant(conn, gid)
        assert first["revoked"] is True and first["epoch"] == 1
        second = revoke_grant(conn, gid)
        assert second["already_revoked"] is True
        assert second["epoch"] == 1  # no extra bump


def test_expired_delegation_edge_dead(store, scope_id, alice):
    """An expired delegation edge cannot carry the child (§09.03)."""
    parent = _parent(store, scope_id, alice)
    now = now_us()
    with store.tx() as conn:
        did, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
            verbs={"read"},
        )
        # force the edge (not the grant) to be already-expired
        repos_v3.update(
            conn, "delegations",
            {"expires_us": now - 1}, {"delegation_id": did},
        )
    assert _denied(store, scope_id, pid="agent:a1").code == (
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    )


# ---------------------------------------------------------------------
# effective_verbs
# ---------------------------------------------------------------------


def test_effective_verbs_union(store, scope_id, alice):
    with store.tx() as conn:
        create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read", "quote"}, issuer_id=alice,
        )
        create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"derive", "ingest"}, issuer_id=alice,
            grant_id="g:two",
        )
    with store.read() as conn:
        verbs = effective_verbs(conn, _caller(), scope_id)
        assert verbs == frozenset({"read", "quote", "derive", "ingest"})


def test_effective_verbs_excludes_dead(store, scope_id, alice):
    with store.tx() as conn:
        create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, issuer_id=alice,
        )
        create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"share"}, issuer_id=alice,
            issued_us=T0, expires_us=T0 + 10,
        )
        gid = create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"admin"}, issuer_id=alice,
        )
        revoke_grant(conn, gid)
    with store.read() as conn:
        assert effective_verbs(conn, _caller(), scope_id) == frozenset({"read"})


def test_effective_verbs_empty_without_grants(store, scope_id):
    with store.read() as conn:
        assert effective_verbs(conn, _caller("human:ghost"), scope_id) == (
            frozenset()
        )


# ---------------------------------------------------------------------
# capture authorization + hydration consent (§11)
# ---------------------------------------------------------------------


def _cauth(store, sid, pid="human:alice", kinds=("user_message",),
           scopes=("scope:gov",), expires=None):
    with store.tx() as conn:
        return issue_capture_authorization(
            conn, principal_id=pid, issuer_id="operator:o1",
            allowed_kinds=kinds, scope_ids=scopes,
            retention_policy="keep-30d", policy_revision="r1",
            issued_us=T0, expires_us=expires,
        )


def test_capture_authorized_happy_path(store, scope_id):
    _cauth(store, scope_id)
    with store.read() as conn:
        assert capture_authorized(
            conn, "human:alice", "user_message", scope_id
        ) is True
        assert capture_authorized(
            conn, "human:alice", "user_message", scope_id,
            retention_policy="keep-30d",
        ) is True
        # mismatched retention policy denies
        assert capture_authorized(
            conn, "human:alice", "user_message", scope_id,
            retention_policy="keep-forever",
        ) is False


def test_capture_kind_not_allowed_denied(store, scope_id):
    _cauth(store, scope_id, kinds=("user_message",))
    with store.read() as conn:
        assert capture_authorized(
            conn, "human:alice", "tool_call", scope_id
        ) is False


def test_capture_expired_denied(store, scope_id):
    _cauth(store, scope_id, expires=T0 + 10)  # already expired vs wall clock
    with store.read() as conn:
        assert capture_authorized(
            conn, "human:alice", "user_message", scope_id
        ) is False


def test_capture_revoked_denied(store, scope_id):
    aid = _cauth(store, scope_id)
    with store.tx() as conn:
        assert revoke_capture_authorization(conn, aid) is True
        assert revoke_capture_authorization(conn, aid) is False  # idempotent
    with store.read() as conn:
        assert capture_authorized(
            conn, "human:alice", "user_message", scope_id
        ) is False


def test_capture_scope_miss_denied(store, scope_id):
    _cauth(store, scope_id, scopes=("scope:gov",))
    with store.read() as conn:
        assert capture_authorized(
            conn, "human:alice", "user_message", "scope:other"
        ) is False


def test_capture_empty_scopes_unconstrained(store, scope_id):
    _cauth(store, scope_id, scopes=())
    with store.read() as conn:
        assert capture_authorized(
            conn, "human:alice", "user_message", "scope:anywhere"
        ) is True


def test_require_capture_authorization_raises(store, scope_id):
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            require_capture_authorization(
                conn, "human:alice", "tool_call", scope_id
            )
        assert exc.value.code == ErrorCode.CONSENT_REQUIRED


def _consent(store, sid, purpose="hydrate", processor="proc:p1",
             expires=None, revoked=None):
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO consents (consent_id, scope_id, processor, purpose,"
            " granted_us, revoked_us, policy_digest, expires_us,"
            " data_classes_json, sanitization, retention_promise,"
            " budget_microusd)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"c:{purpose}:{processor}", sid, processor, purpose,
                T0, revoked, "digest-1", expires,
                '["s2"]', "sanitized", "30d", 5000,
            ),
        )


def test_hydration_consent_match(store, scope_id):
    _consent(store, scope_id)
    with store.read() as conn:
        row = require_hydration_consent(conn, scope_id, "hydrate")
        assert row["processor"] == "proc:p1"
        assert row["sanitization"] == "sanitized"
        assert consent_for_hydration(conn, scope_id, "hydrate") is not None


def test_hydration_consent_purpose_mismatch(store, scope_id):
    _consent(store, scope_id, purpose="hydrate")
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            require_hydration_consent(conn, scope_id, "derive")
        assert exc.value.code == ErrorCode.CONSENT_REQUIRED


def test_hydration_consent_expired(store, scope_id):
    _consent(store, scope_id, expires=T0 + 10)
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            require_hydration_consent(conn, scope_id, "hydrate")
        assert exc.value.code == ErrorCode.CONSENT_REQUIRED


def test_hydration_consent_revoked(store, scope_id):
    _consent(store, scope_id, revoked=T0 + 5)
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            require_hydration_consent(conn, scope_id, "hydrate")
        assert exc.value.code == ErrorCode.CONSENT_REQUIRED


# ---------------------------------------------------------------------
# propagation ledger (§10.04–10.05)
# ---------------------------------------------------------------------


def _propagation(pid="prop:p1", recipient="agent:a1", epoch=0):
    return Propagation(
        propagation_id=pid,
        scope_id="scope:gov",
        object_kind="source",
        object_id="src:1",
        revision=1,
        recipient_id=recipient,
        verbs=frozenset({"read", "quote"}),
        purpose="share",
        epoch=epoch,
        created_us=T0,
        capsule_id="cap:c1",
    )


def test_record_propagation_persists(store, scope_id):
    with store.tx() as conn:
        pid = record_propagation(conn, _propagation())
        rows = propagations_for(conn, "source", "src:1")
        assert len(rows) == 1
        row = rows[0]
        assert row["propagation_id"] == pid
        assert row["recipient_id"] == "agent:a1"
        assert repos_v3.json_field(row, "verbs_json") == ["quote", "read"]
        assert row["purpose"] == "share"
        assert row["epoch"] == 0
        assert row["capsule_id"] == "cap:c1"


def test_blast_radius_lists_recipients(store, scope_id):
    with store.tx() as conn:
        record_propagation(conn, _propagation("prop:p1", "agent:a1"))
        record_propagation(
            conn, _propagation("prop:p2", "workspace:w1", epoch=1)
        )
        radius = blast_radius(conn, "source", "src:1")
        assert [r["recipient_id"] for r in radius] == [
            "agent:a1", "workspace:w1",
        ]
        assert radius[0]["verbs"] == ["quote", "read"]
        assert recipients_of(conn, "source", "src:1") == frozenset(
            {"agent:a1", "workspace:w1"}
        )


def test_mark_propagations_revoked(store, scope_id):
    with store.tx() as conn:
        record_propagation(conn, _propagation("prop:p1", "agent:a1"))
        record_propagation(conn, _propagation("prop:p2", "agent:a2"))
        n = mark_propagations_revoked(
            conn, seq=7, scope_id=scope_id, recipient_id="agent:a1"
        )
        assert n == 1
        live = propagations_for(conn, "source", "src:1")
        assert [r["recipient_id"] for r in live] == ["agent:a2"]
        radius = blast_radius(conn, "source", "src:1")
        revoked = {r["recipient_id"]: r["revoked"] for r in radius}
        assert revoked == {"agent:a1": True, "agent:a2": False}
        # blanket marking refused
        with pytest.raises(VerbatimError):
            mark_propagations_revoked(conn, seq=9)


# ---------------------------------------------------------------------
# revocation_notify handler (§10.05, §40)
# ---------------------------------------------------------------------


@pytest.fixture
def ingester(store):
    return Ingester(store, config_from_mapping({}))


def _lease(store, ingester, sid, refs, dedup=None):
    with store.tx() as conn:
        jid = ingester.jobs.enqueue(
            conn, sid, JobKind.REVOCATION_NOTIFY, refs, dedup_key=dedup
        )
    leased = ingester.jobs.lease(
        sid, [JobKind.REVOCATION_NOTIFY], owner="w1"
    )
    assert leased and leased[0]["job_id"] == jid
    return leased[0], jid


def test_handler_revokes_grant_and_cascade(store, scope_id, alice, ingester):
    parent = _parent(store, scope_id, alice)
    with store.tx() as conn:
        _, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
            verbs={"read"},
        )
    job, _ = _lease(store, ingester, scope_id, {"grant_id": parent})
    handle_revocation_notify(job, "w1", ingester)
    with store.read() as conn:
        for gid in (parent, child):
            row = repos_v3.get(conn, "grants_v3", {"grant_id": gid})
            assert row["revoked_us"] is not None
        assert current_epoch(conn, scope_id) == 1
        ev = conn.execute(
            "SELECT kind, payload_json FROM events"
            " WHERE kind = 'revocation_applied'"
        ).fetchall()
        assert len(ev) == 1
    assert _denied(store, scope_id, pid="agent:a1").code == (
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    )


def test_handler_fences_propagations_and_capsules(
    store, scope_id, alice, ingester
):
    with store.tx() as conn:
        gid = create_grant(
            conn, scope_id=scope_id, principal_id="agent:a1",
            verbs={"read"}, issuer_id=alice,
        )
        record_propagation(conn, _propagation("prop:p1", "agent:a1"))
        conn.execute(
            "INSERT INTO handoff_capsules"
            " (capsule_id, scope_id, recipient_id, issuer_id, status)"
            " VALUES ('cap:x', ?, 'agent:a1', 'human:alice', 'open')",
            (scope_id,),
        )
    job, _ = _lease(store, ingester, scope_id, {"grant_id": gid})
    handle_revocation_notify(job, "w1", ingester)
    with store.read() as conn:
        row = repos_v3.get(conn, "propagations", {"propagation_id": "prop:p1"})
        assert row["revoked_seq"] == 1  # fenced at the new epoch
        cap = conn.execute(
            "SELECT status FROM handoff_capsules WHERE capsule_id = 'cap:x'"
        ).fetchone()
        assert cap[0] == "revoked"


def test_handler_delegation_path(store, scope_id, alice, ingester):
    parent = _parent(store, scope_id, alice)
    with store.tx() as conn:
        did, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
            verbs={"read"},
        )
    job, _ = _lease(store, ingester, scope_id, {"delegation_id": did})
    handle_revocation_notify(job, "w1", ingester)
    with store.read() as conn:
        dead = repos_v3.get(conn, "grants_v3", {"grant_id": child})
        assert dead["revoked_us"] is not None


def test_handler_idempotent_and_dedup(store, scope_id, alice, ingester):
    with store.tx() as conn:
        gid = create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, issuer_id=alice,
        )
    dedup = store.hmac(b"revoke:g")
    with store.tx() as conn:
        j1 = ingester.jobs.enqueue(
            conn, scope_id, JobKind.REVOCATION_NOTIFY,
            {"grant_id": gid}, dedup_key=dedup,
        )
        j2 = ingester.jobs.enqueue(
            conn, scope_id, JobKind.REVOCATION_NOTIFY,
            {"grant_id": gid}, dedup_key=dedup,
        )
    assert j1 == j2  # job-level dedup converged on one row
    leased = ingester.jobs.lease(
        scope_id, [JobKind.REVOCATION_NOTIFY], owner="w1"
    )
    assert leased and leased[0]["job_id"] == j1
    job = leased[0]
    handle_revocation_notify(job, "w1", ingester)
    # a redelivered handler call is a no-op beyond the first
    handle_revocation_notify(job, "w1", ingester)
    with store.read() as conn:
        assert current_epoch(conn, scope_id) == 1
        evs = conn.execute(
            "SELECT COUNT(*) FROM events WHERE kind = 'revocation_applied'"
        ).fetchone()[0]
        assert evs == 2  # each handler call audits; state stayed revoked


def test_handler_requires_exactly_one_ref(store, scope_id, ingester):
    job, _ = _lease(store, ingester, scope_id, {})
    with pytest.raises(VerbatimError) as exc:
        handle_revocation_notify(job, "w1", ingester)
    assert exc.value.code == ErrorCode.VALIDATION


def test_handler_dispatch_registered():
    """The v3 dispatcher resolves our handler path (v3_contracts.md)."""
    from verbatim.ingest import _V3_KIND_HANDLERS

    mod, fn = _V3_KIND_HANDLERS[JobKind.REVOCATION_NOTIFY]
    assert mod == "verbatim.governance.handlers"
    assert fn == "handle_revocation_notify"
    import importlib

    assert callable(getattr(importlib.import_module(mod), fn))


# ---------------------------------------------------------------------
# tx-scoping sanity: everything above already mutates via store.tx();
# this test proves a failed mutation rolls back atomically.
# ---------------------------------------------------------------------


def test_failed_grant_rolls_back(store, scope_id, alice):
    with pytest.raises(VerbatimError):
        with store.tx() as conn:
            create_grant(
                conn, scope_id=scope_id, principal_id=alice,
                verbs={"read"}, issuer_id=alice, grant_id="g:partial",
            )
            create_grant(
                conn, scope_id=scope_id, principal_id=alice,
                verbs={"read"}, issuer_id=alice, grant_id="g:partial",
            )  # duplicate PK → abort tx
    with store.read() as conn:
        assert repos_v3.get(conn, "grants_v3", {"grant_id": "g:partial"}) is None
