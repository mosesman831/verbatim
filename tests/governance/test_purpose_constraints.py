"""Tagged purpose constraints and semantic delegation attenuation.

SPEC_V4 §11 (V4-11.01–V4-11.10), finding F4-03, scenario C09, migration
contract V4-62.05. These are the durable public-surface regressions for
"empty delegated purposes become unrestricted":

- ``grants_v3`` rows persist an explicit ``purpose_tag`` — any | set |
  none — next to ``purposes_json``; an explicitly empty purpose set is a
  NONE constraint, never an untagged ANY (V4-11.01).
- ``delegate_grant`` proves child ⊆ parent semantically across verbs,
  scope, purposes (``PurposeConstraint.is_subset_of``), expiry, and depth
  (V4-11.02); caveats are conjunctive — retained or strengthened, never
  dropped (V4-11.03).
- Cyclic chains, excessive depth, ambiguous constraints, malformed or
  non-finite values are rejected at creation and fail closed at
  evaluation (V4-11.09); the sweep test searches for authority expansion
  over grant/delegation combinations (V4-11.10).
- ``legacy_empty``/NULL-tagged rows keep their historical empty-means-ANY
  evaluation but are listed by ``legacy_purpose_grants`` for
  owner-reviewed remediation (V4-62.05).
"""

from __future__ import annotations

import pytest

from verbatim.core.time import now_us
from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v4 import PurposeConstraint
from verbatim.governance import (
    CallerV3,
    authorize,
    create_grant,
    delegate_grant,
    effective_verbs,
    get_grant,
    grant_purpose_constraint,
    legacy_purpose_grants,
    register_principal,
    revoke_delegation,
    seed_purposes,
)
from verbatim.storage import repos_v3
from verbatim.storage.store import Store

T0 = 1_700_000_000_000_000  # fixed past base (2023) for deterministic rows


def FAR_FUTURE() -> int:
    """A live expiry bound — computed per call: a module-level constant
    can fall behind wall clock when the full suite runs for a while."""
    return now_us() + 10**12  # ~11 days out; safe under any suite duration


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "pc.db"))
    yield s
    s.close()


@pytest.fixture
def scope_id(store):
    sid = "scope:pc"
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


def _allows(store, sid, pid, verb, purpose=None) -> bool:
    with store.read() as conn:
        try:
            authorize(conn, _caller(pid), sid, verb, purpose=purpose)
            return True
        except VerbatimError as exc:
            assert exc.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
            return False


def _denied(store, sid, pid="human:alice", verb="read", purpose=None):
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            authorize(conn, _caller(pid), sid, verb, purpose=purpose)
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
        return exc.value


# ---------------------------------------------------------------------
# tag persistence + evaluation (V4-11.01)
# ---------------------------------------------------------------------


def test_set_purposes_persist_tagged(store, scope_id, alice):
    with store.tx() as conn:
        gid = create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, purposes={"derive", "recall"}, issuer_id=alice,
        )
        row = get_grant(conn, gid)
        assert row["purpose_tag"] == "set"
        assert repos_v3.json_field(row, "purposes_json") == [
            "derive", "recall",
        ]
        assert grant_purpose_constraint(row) == PurposeConstraint.set(
            {"derive", "recall"}
        )


def test_explicit_any_persisted(store, scope_id, alice):
    with store.tx() as conn:
        gid = create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, purposes=PurposeConstraint.any(),
            issuer_id=alice,
        )
        row = get_grant(conn, gid)
        assert row["purpose_tag"] == "any"
        assert repos_v3.json_field(row, "purposes_json") == []
        assert grant_purpose_constraint(row) == PurposeConstraint.any()


def test_empty_iterable_purposes_normalize_to_none(store, scope_id, alice):
    """V4-11.01: an explicitly empty purpose set is a SET of zero members
    → normalized to NONE at persistence; it authorizes nothing, not even
    a request that declares no purpose."""
    with store.tx() as conn:
        gid = create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, purposes=[], issuer_id=alice,
        )
        row = get_grant(conn, gid)
        assert row["purpose_tag"] == "none"
        assert repos_v3.json_field(row, "purposes_json") == []
        assert grant_purpose_constraint(row) == PurposeConstraint.none()
    for purpose in (None, "recall", "evaluate"):
        _denied(store, scope_id, purpose=purpose)


def test_none_constraint_denies_everything(store, scope_id, alice):
    with store.tx() as conn:
        gid = create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, purposes=PurposeConstraint.none(),
            issuer_id=alice,
        )
        assert get_grant(conn, gid)["purpose_tag"] == "none"
    _denied(store, scope_id)
    _denied(store, scope_id, purpose="recall")


def test_set_grant_binds_declared_purpose(store, scope_id, alice):
    """SET permits only listed purposes; a request declaring no purpose
    never satisfies a SET constraint (§09.08)."""
    with store.tx() as conn:
        create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, purposes={"recall"}, issuer_id=alice,
        )
    assert _allows(store, scope_id, "human:alice", "read", "recall")
    assert not _allows(store, scope_id, "human:alice", "read", "derive")
    assert not _allows(store, scope_id, "human:alice", "read", None)


def test_effective_verbs_union_is_purpose_blind_diagnostic(
    store, scope_id, alice
):
    """V4-11.04: ``effective_verbs`` is a diagnostic union — a NONE grant
    still contributes its verbs there while ``authorize`` denies it."""
    with store.tx() as conn:
        create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, purposes=PurposeConstraint.none(),
            issuer_id=alice,
        )
    with store.read() as conn:
        assert "read" in effective_verbs(conn, _caller(), scope_id)
    _denied(store, scope_id, purpose="recall")


# ---------------------------------------------------------------------
# malformed / ambiguous input rejection (V4-11.09)
# ---------------------------------------------------------------------


def test_create_grant_rejects_ambiguous_purposes(store, scope_id, alice):
    """Bare strings, raw dicts, non-iterables, and non-string members are
    ambiguous constraint forms — rejected, never guessed."""
    bad = [
        "recall",                 # one string ≠ an iterable of names
        b"recall",
        {"tag": "any", "values": []},
        42,
        ["recall", ""],           # empty member
        ["recall", None],         # non-string member
    ]
    with store.tx() as conn:
        for value in bad:
            with pytest.raises(VerbatimError) as exc:
                create_grant(
                    conn, scope_id=scope_id, principal_id=alice,
                    verbs={"read"}, purposes=value, issuer_id=alice,
                )
            assert exc.value.code == ErrorCode.VALIDATION, value


def test_create_grant_rejects_nonfinite_and_malformed(store, scope_id, alice):
    with store.tx() as conn:
        for bad_expiry in (float("inf"), float("nan"), "soon", 1.5, True):
            with pytest.raises(VerbatimError) as exc:
                create_grant(
                    conn, scope_id=scope_id, principal_id=alice,
                    verbs={"read"}, issuer_id=alice, expires_us=bad_expiry,
                )
            assert exc.value.code == ErrorCode.VALIDATION, bad_expiry
        for bad_issued in ("x", float("nan"), object()):
            with pytest.raises(VerbatimError) as exc:
                create_grant(
                    conn, scope_id=scope_id, principal_id=alice,
                    verbs={"read"}, issuer_id=alice, issued_us=bad_issued,
                )
            assert exc.value.code == ErrorCode.VALIDATION, bad_issued
        for bad_epoch in (-1, "x", 2.5):
            with pytest.raises(VerbatimError) as exc:
                create_grant(
                    conn, scope_id=scope_id, principal_id=alice,
                    verbs={"read"}, issuer_id=alice, epoch=bad_epoch,
                )
            assert exc.value.code == ErrorCode.VALIDATION, bad_epoch


def test_create_grant_depth_bound(store, scope_id, alice):
    """V4-11.09 documented bound: delegation-depth budget is capped at 8;
    real chains can never exceed it since each delegation decrements."""
    with store.tx() as conn:
        gid = create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, issuer_id=alice, delegation_depth=8,
        )
        assert get_grant(conn, gid)["delegation_depth"] == 8
        for bad in (9, 100, -1, 2.5, "4", True):
            with pytest.raises(VerbatimError) as exc:
                create_grant(
                    conn, scope_id=scope_id, principal_id=alice,
                    verbs={"read"}, issuer_id=alice, delegation_depth=bad,
                )
            assert exc.value.code == ErrorCode.VALIDATION, bad


def test_malformed_stored_purpose_tag_fails_closed(store, scope_id, alice):
    """An unknown persisted tag authorizes nothing (V4-11.09)."""
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO grants_v3 (grant_id, scope_id, principal_id,"
            " verbs_json, purposes_json, purpose_tag, issuer_id,"
            " issued_us, epoch)"
            " VALUES ('g:bad', ?, ?, '[\"read\"]', '[]', 'bogus', ?, ?, 0)",
            (scope_id, alice, alice, T0),
        )
    _denied(store, scope_id)
    _denied(store, scope_id, purpose="recall")
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            grant_purpose_constraint(
                get_grant(conn, "g:bad")
            )
        assert exc.value.code == ErrorCode.VALIDATION


def test_contradictory_tag_values_fail_closed(store, scope_id, alice):
    """``purpose_tag='any'`` carrying SET values is a self-contradictory
    persisted form — dead, not silently ANY."""
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO grants_v3 (grant_id, scope_id, principal_id,"
            " verbs_json, purposes_json, purpose_tag, issuer_id,"
            " issued_us, epoch)"
            " VALUES ('g:contr', ?, ?, '[\"read\"]', '[\"recall\"]',"
            " 'any', ?, ?, 0)",
            (scope_id, alice, alice, T0),
        )
    _denied(store, scope_id, purpose="recall")


def test_set_tag_with_empty_values_evaluates_none(store, scope_id, alice):
    """A 'set'-tagged row whose JSON values are empty normalizes to NONE
    at evaluation — the empty set can never resurrect ANY."""
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO grants_v3 (grant_id, scope_id, principal_id,"
            " verbs_json, purposes_json, purpose_tag, issuer_id,"
            " issued_us, epoch)"
            " VALUES ('g:emptyset', ?, ?, '[\"read\"]', '[]', 'set',"
            " ?, ?, 0)",
            (scope_id, alice, alice, T0),
        )
    _denied(store, scope_id)
    _denied(store, scope_id, purpose="recall")


def test_malformed_stored_expiry_dead(store, scope_id, alice):
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO grants_v3 (grant_id, scope_id, principal_id,"
            " verbs_json, purposes_json, purpose_tag, issuer_id,"
            " issued_us, expires_us, epoch)"
            " VALUES ('g:badexp', ?, ?, '[\"read\"]', '[]', 'any',"
            " ?, ?, 'not-a-number', 0)",
            (scope_id, alice, alice, T0),
        )
    _denied(store, scope_id)


def test_corrupt_purposes_json_fails_closed_and_reports(
    store, scope_id, alice
):
    """Undecodable purposes_json on an untagged row is dead authority —
    denied, not crashed — and still listed for remediation."""
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO grants_v3 (grant_id, scope_id, principal_id,"
            " verbs_json, purposes_json, purpose_tag, issuer_id,"
            " issued_us, epoch)"
            " VALUES ('g:corrupt', ?, ?, '[\"read\"]', 'not json', NULL,"
            " ?, ?, 0)",
            (scope_id, alice, alice, T0),
        )
        rows = legacy_purpose_grants(conn)
        assert rows[0]["grant_id"] == "g:corrupt"
        assert rows[0]["malformed_purposes_json"] is True
        assert rows[0]["effective_purposes"] == "denied"
    _denied(store, scope_id, purpose="recall")
    _denied(store, scope_id)


# ---------------------------------------------------------------------
# delegation: purposes (V4-11.02, C09)
# ---------------------------------------------------------------------


def _parent(store, sid, alice, purposes=("recall",), depth=2,
            verbs=("read", "quote"), expires_us=None, caveats=("no-export",)):
    with store.tx() as conn:
        return create_grant(
            conn, scope_id=sid, principal_id=alice, verbs=set(verbs),
            purposes=purposes, caveats=caveats, issuer_id=alice,
            delegation_depth=depth, expires_us=expires_us,
        )


def test_c09_empty_delegated_purposes_grant_no_authority(
    store, scope_id, alice
):
    """C09 / F4-03: delegating an empty purpose set from an evaluate-only
    grant yields a NONE-purpose child — no recall authority, ever. The
    pre-v4 row encoded ``[]`` which the evaluator read as unrestricted."""
    parent = _parent(store, scope_id, alice, purposes={"evaluate"})
    with store.tx() as conn:
        _, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
            purposes=[],
        )
        row = get_grant(conn, child)
        assert row["purpose_tag"] == "none"
        assert repos_v3.json_field(row, "purposes_json") == []
    with store.read() as conn:
        # parent retains its evaluate-only authority
        authorize(conn, _caller(), scope_id, "read", purpose="evaluate")
    # the child holds no purpose authority at all
    for purpose in (None, "evaluate", "recall"):
        _denied(store, scope_id, pid="agent:a1", verb="read",
                purpose=purpose)


def test_delegate_empty_set_and_none_constraint_equivalent(
    store, scope_id, alice
):
    """``[]``, ``set()``, and ``PurposeConstraint.none()`` all produce a
    NONE child — three spellings, one dead authority."""
    for empty in ([], set(), frozenset(), PurposeConstraint.none()):
        parent = _parent(store, scope_id, alice, purposes={"evaluate"})
        with store.tx() as conn:
            _, child = delegate_grant(
                conn, parent_grant_id=parent, delegate_id="agent:a1",
                purposes=empty,
            )
            assert get_grant(conn, child)["purpose_tag"] == "none"
        _denied(store, scope_id, pid="agent:a1", verb="read",
                purpose="evaluate")


def test_delegate_any_parent_to_set_child(store, scope_id, alice):
    parent = _parent(
        store, scope_id, alice, purposes=PurposeConstraint.any(),
        caveats=(),
    )
    with store.tx() as conn:
        _, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
            purposes={"recall"},
        )
        row = get_grant(conn, child)
        assert row["purpose_tag"] == "set"
        assert repos_v3.json_field(row, "purposes_json") == ["recall"]
    assert _allows(store, scope_id, "agent:a1", "read", "recall")
    assert _allows(store, scope_id, "agent:a1", "quote", "recall")
    assert not _allows(store, scope_id, "agent:a1", "read", "derive")
    assert not _allows(store, scope_id, "agent:a1", "read", None)


def test_delegate_inherits_parent_constraint(store, scope_id, alice):
    parent = _parent(store, scope_id, alice, purposes={"recall", "derive"})
    with store.tx() as conn:
        _, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
        )
        row = get_grant(conn, child)
        assert row["purpose_tag"] == "set"
        assert set(repos_v3.json_field(row, "purposes_json")) == {
            "recall", "derive",
        }
    assert _allows(store, scope_id, "agent:a1", "read", "derive")
    assert not _allows(store, scope_id, "agent:a1", "read", "evaluate")


def test_delegate_any_parent_inherits_any(store, scope_id, alice):
    parent = _parent(
        store, scope_id, alice, purposes=PurposeConstraint.any(),
        caveats=(),
    )
    with store.tx() as conn:
        _, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
        )
        assert get_grant(conn, child)["purpose_tag"] == "any"
    assert _allows(store, scope_id, "agent:a1", "read", "evaluate")


def test_delegate_cannot_widen_set_to_any(store, scope_id, alice):
    parent = _parent(store, scope_id, alice, purposes={"recall"})
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            delegate_grant(
                conn, parent_grant_id=parent, delegate_id="agent:a1",
                purposes=PurposeConstraint.any(),
            )
        assert exc.value.code == ErrorCode.VALIDATION


def test_delegate_cannot_add_purpose(store, scope_id, alice):
    parent = _parent(store, scope_id, alice, purposes={"recall"})
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            delegate_grant(
                conn, parent_grant_id=parent, delegate_id="agent:a1",
                purposes={"recall", "derive"},
            )
        assert exc.value.code == ErrorCode.VALIDATION


def test_delegate_none_parent_accepts_only_none(store, scope_id, alice):
    parent = _parent(
        store, scope_id, alice, purposes=PurposeConstraint.none(),
        caveats=(),
    )
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            delegate_grant(
                conn, parent_grant_id=parent, delegate_id="agent:a1",
                purposes={"recall"},
            )
        assert exc.value.code == ErrorCode.VALIDATION
        # inheriting (or restating) NONE is legal — still dead authority
        _, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
        )
        assert get_grant(conn, child)["purpose_tag"] == "none"


def test_delegate_verbs_still_attenuation_only(store, scope_id, alice):
    parent = _parent(store, scope_id, alice)
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            delegate_grant(
                conn, parent_grant_id=parent, delegate_id="agent:a1",
                verbs={"read", "share"},
            )
        assert exc.value.code == ErrorCode.VALIDATION


def test_delegate_scope_inherited_verbatim(store, scope_id, alice):
    """Resource dimension of V4-11.02: a child always binds the parent's
    scope — delegation can never re-target another partition."""
    parent = _parent(store, scope_id, alice)
    with store.tx() as conn:
        _, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
        )
        assert get_grant(conn, child)["scope_id"] == scope_id


# ---------------------------------------------------------------------
# delegation: expiry, depth, caveats
# ---------------------------------------------------------------------


def test_delegate_expiry_bounded_by_parent(store, scope_id, alice):
    far = FAR_FUTURE()
    parent = _parent(store, scope_id, alice, expires_us=far)
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            delegate_grant(
                conn, parent_grant_id=parent, delegate_id="agent:a1",
                expires_us=far + 1,
            )
        assert exc.value.code == ErrorCode.VALIDATION
        # omission inherits the parent's bound — never widens
        _, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
        )
        assert get_grant(conn, child)["expires_us"] == far
        _, child2 = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a2",
            expires_us=far - 1,
        )
        assert get_grant(conn, child2)["expires_us"] == far - 1


def test_delegate_unbounded_parent_allows_child_expiry(
    store, scope_id, alice
):
    far = FAR_FUTURE()
    parent = _parent(store, scope_id, alice)
    with store.tx() as conn:
        _, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
            expires_us=far,
        )
        assert get_grant(conn, child)["expires_us"] == far


def test_delegate_rejects_malformed_expiry(store, scope_id, alice):
    parent = _parent(store, scope_id, alice)
    with store.tx() as conn:
        for bad in (float("inf"), float("nan"), "soon", True):
            with pytest.raises(VerbatimError) as exc:
                delegate_grant(
                    conn, parent_grant_id=parent, delegate_id="agent:a1",
                    expires_us=bad,
                )
            assert exc.value.code == ErrorCode.VALIDATION, bad


def test_delegate_depth_decrements(store, scope_id, alice):
    parent = _parent(store, scope_id, alice, depth=3)
    with store.tx() as conn:
        _, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
        )
        assert get_grant(conn, child)["delegation_depth"] == 2


def test_delegate_caveats_retained_or_strengthened(store, scope_id, alice):
    """V4-11.03: conjunctive caveats — the child keeps every parent
    caveat; adding more is attenuation, dropping any is expansion."""
    parent = _parent(store, scope_id, alice, caveats=("no-export",))
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            delegate_grant(
                conn, parent_grant_id=parent, delegate_id="agent:a1",
                caveats=(),
            )
        assert exc.value.code == ErrorCode.VALIDATION
        with pytest.raises(VerbatimError) as exc:
            delegate_grant(
                conn, parent_grant_id=parent, delegate_id="agent:a1",
                caveats=("different",),
            )
        assert exc.value.code == ErrorCode.VALIDATION
        _, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
            caveats=("no-export", "watermark"),
        )
        assert repos_v3.json_field(
            get_grant(conn, child), "caveats_json"
        ) == ["no-export", "watermark"]
        # omission inherits all parent caveats
        _, child2 = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a2",
        )
        assert repos_v3.json_field(
            get_grant(conn, child2), "caveats_json"
        ) == ["no-export"]


def test_delegate_under_dead_chain_parent_denied(store, scope_id, alice):
    """A delegated parent whose chain broke holds no effective authority
    to pass on — delegation under it denies like absent authority."""
    parent = _parent(store, scope_id, alice, depth=2)
    with store.tx() as conn:
        did, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
        )
        # expire the edge (not the grant): the child row stays live but
        # its only chain to a root is dead — it holds no authority to
        # delegate onward
        repos_v3.update(
            conn, "delegations", {"expires_us": now_us() - 1},
            {"delegation_id": did},
        )
        with pytest.raises(VerbatimError) as exc:
            delegate_grant(
                conn, parent_grant_id=child, delegate_id="agent:a3",
            )
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    _denied(store, scope_id, pid="agent:a1", verb="read")


def test_delegate_under_cyclic_parent_denied(store, scope_id, alice):
    """V4-11.09: a cyclic delegation chain is dead authority — the cyclic
    parent cannot delegate and its principal authorizes nothing."""
    with store.tx() as conn:
        ga = create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, issuer_id=alice, delegation_depth=2,
        )
        gb = create_grant(
            conn, scope_id=scope_id, principal_id="agent:a1",
            verbs={"read"}, issuer_id=alice, delegation_depth=2,
        )
        # hand-written cycle: a→b and b→a — no path reaches a live root
        repos_v3.insert(conn, "delegations", {
            "delegation_id": "dlg:cyc1", "parent_grant_id": ga,
            "child_grant_id": gb, "delegator_id": alice,
            "delegate_id": "agent:a1", "created_us": T0,
        })
        repos_v3.insert(conn, "delegations", {
            "delegation_id": "dlg:cyc2", "parent_grant_id": gb,
            "child_grant_id": ga, "delegator_id": "agent:a1",
            "delegate_id": alice, "created_us": T0,
        })
    _denied(store, scope_id, pid="agent:a1", verb="read")
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            delegate_grant(
                conn, parent_grant_id=gb, delegate_id="agent:a3",
            )
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_delegate_rejects_reused_grant_ids(store, scope_id, alice):
    """V4-11.09: a delegation child must be a fresh grant — naming an
    existing grant (the parent itself, or any ancestor) would close a
    cycle and kill the chain. Rejected as caller error, not a crash."""
    with store.tx() as conn:
        parent = create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, issuer_id=alice, delegation_depth=2,
        )
        _, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:a1",
        )
        # self-loop attempt: parent as its own child
        with pytest.raises(VerbatimError) as exc:
            delegate_grant(
                conn, parent_grant_id=parent, delegate_id="agent:a2",
                child_grant_id=parent,
            )
        assert exc.value.code == ErrorCode.VALIDATION
        # cycle attempt: an existing descendant as the new child id
        with pytest.raises(VerbatimError) as exc:
            delegate_grant(
                conn, parent_grant_id=parent, delegate_id="agent:a2",
                child_grant_id=child,
            )
        assert exc.value.code == ErrorCode.VALIDATION
        # delegation_id collision likewise
        repos_v3.insert(conn, "delegations", {
            "delegation_id": "dlg:taken", "parent_grant_id": parent,
            "child_grant_id": child, "delegator_id": alice,
            "delegate_id": "agent:a1", "created_us": T0,
        })
        with pytest.raises(VerbatimError) as exc:
            delegate_grant(
                conn, parent_grant_id=parent, delegate_id="agent:a2",
                delegation_id="dlg:taken",
            )
        assert exc.value.code == ErrorCode.VALIDATION


def test_excessive_chain_is_dead_authority(store, scope_id, alice):
    """V4-11.09: a hand-written chain beyond the evaluation walk bound
    never reaches its root — the leaf grant authorizes nothing."""
    n = 40  # > _MAX_CHAIN
    with store.tx() as conn:
        ids = []
        for i in range(n):
            gid = f"g:chain{i}"
            conn.execute(
                "INSERT INTO grants_v3 (grant_id, scope_id, principal_id,"
                " verbs_json, purposes_json, purpose_tag, issuer_id,"
                " issued_us, delegation_depth, epoch)"
                " VALUES (?, ?, ?, '[\"read\"]', '[]', 'any', ?, ?, 64, 0)",
                (gid, scope_id, alice if i == 0 else f"agent:c{i}",
                 alice, T0),
            )
            if i:
                conn.execute(
                    "INSERT INTO delegations (delegation_id,"
                    " parent_grant_id, child_grant_id, delegator_id,"
                    " delegate_id, created_us)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (f"dlg:chain{i}", ids[-1], gid, "agent", f"agent:c{i}",
                     T0),
                )
            ids.append(gid)
    _denied(store, scope_id, pid=f"agent:c{n - 1}", verb="read")


def test_multi_parent_chain_alternate_path(store, scope_id, alice):
    """Multiple parents (V4-11.10): a second delegation edge is an
    alternate liveness path — killing edge one leaves the child alive via
    edge two, and the child's constraint stays what its creator proved."""
    with store.tx() as conn:
        p1 = create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read", "quote"}, purposes={"recall"},
            issuer_id=alice, delegation_depth=2,
        )
        p2 = create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, purposes={"recall", "derive"},
            issuer_id=alice, delegation_depth=2,
        )
        d1, child = delegate_grant(
            conn, parent_grant_id=p1, delegate_id="agent:a1",
            verbs={"read"}, purposes={"recall"},
        )
        # second endorsing parent edge (as an out-of-band row could add)
        repos_v3.insert(conn, "delegations", {
            "delegation_id": "dlg:second", "parent_grant_id": p2,
            "child_grant_id": child, "delegator_id": alice,
            "delegate_id": "agent:a1", "created_us": T0,
        })
    assert _allows(store, scope_id, "agent:a1", "read", "recall")
    with store.tx() as conn:
        revoke_delegation(conn, d1)
    # still live through p2 — and still only ever a SET{recall} child
    assert _allows(store, scope_id, "agent:a1", "read", "recall")
    assert not _allows(store, scope_id, "agent:a1", "read", "derive")
    assert not _allows(store, scope_id, "agent:a1", "quote", "recall")


# ---------------------------------------------------------------------
# legacy rows: historical semantics + remediation report (V4-62.05)
# ---------------------------------------------------------------------


def _insert_legacy(conn, sid, pid, gid, purposes_json, tag, **kw):
    conn.execute(
        "INSERT INTO grants_v3 (grant_id, scope_id, principal_id,"
        " verbs_json, purposes_json, purpose_tag, issuer_id, issued_us,"
        " delegation_depth, expires_us, revoked_us, epoch)"
        " VALUES (?, ?, ?, '[\"read\"]', ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            gid, sid, pid, purposes_json, tag, kw.get("issuer", pid),
            kw.get("issued_us", T0), kw.get("depth", 0),
            kw.get("expires_us"), kw.get("revoked_us"), kw.get("epoch", 0),
        ),
    )


def test_legacy_untagged_empty_row_keeps_any_semantics(
    store, scope_id, alice
):
    """Pre-v4 rows (NULL tag) evaluate exactly as before — an empty stored
    set means ANY — so migration never silently kills existing authority."""
    with store.tx() as conn:
        _insert_legacy(conn, scope_id, alice, "g:legacy", "[]", None)
    assert _allows(store, scope_id, "human:alice", "read", "recall")
    assert _allows(store, scope_id, "human:alice", "read", "derive")
    assert _allows(store, scope_id, "human:alice", "read", None)


def test_legacy_empty_tagged_row_keeps_any_semantics(
    store, scope_id, alice
):
    """Rows stamped 'legacy_empty' by the 3→4 migration behave the same."""
    with store.tx() as conn:
        _insert_legacy(
            conn, scope_id, alice, "g:legacy2", "[]", "legacy_empty"
        )
    assert _allows(store, scope_id, "human:alice", "read", "evaluate")


def test_legacy_untagged_nonempty_row_keeps_set_semantics(
    store, scope_id, alice
):
    """Historical membership check on a non-empty stored set is preserved
    for untagged rows too."""
    with store.tx() as conn:
        _insert_legacy(
            conn, scope_id, alice, "g:legacy3", '["recall"]', None
        )
    assert _allows(store, scope_id, "human:alice", "read", "recall")
    assert not _allows(store, scope_id, "human:alice", "read", "derive")
    assert not _allows(store, scope_id, "human:alice", "read", None)


def test_legacy_purpose_grants_report(store, scope_id, alice):
    """V4-62.05: every ambiguously-tagged row is surfaced for owner
    review; explicitly tagged rows never appear."""
    with store.tx() as conn:
        _insert_legacy(conn, scope_id, alice, "g:l1", "[]", None)
        _insert_legacy(conn, scope_id, alice, "g:l2", "[]", "legacy_empty")
        _insert_legacy(conn, scope_id, alice, "g:l3", '["recall"]', None)
        create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, purposes={"recall"}, issuer_id=alice,
        )
        create_grant(
            conn, scope_id=scope_id, principal_id=alice,
            verbs={"read"}, purposes=PurposeConstraint.any(),
            issuer_id=alice,
        )
        rows = legacy_purpose_grants(conn)
        assert {r["grant_id"] for r in rows} == {"g:l1", "g:l2", "g:l3"}
        by_id = {r["grant_id"]: r for r in rows}
        assert by_id["g:l1"]["effective_purposes"] == "any"
        assert by_id["g:l3"]["effective_purposes"] == "set"
        assert all(r["live"] for r in rows)
        assert all("PurposeConstraint" in r["remediation"] for r in rows)


def test_remediated_legacy_row_leaves_report(store, scope_id, alice):
    """Owner remediation: retag the row explicitly and it drops out of
    the report — with the new evaluation attached."""
    with store.tx() as conn:
        _insert_legacy(conn, scope_id, alice, "g:l9", "[]", "legacy_empty")
        conn.execute(
            "UPDATE grants_v3 SET purpose_tag = 'set',"
            " purposes_json = '[\"recall\"]' WHERE grant_id = 'g:l9'"
        )
        assert legacy_purpose_grants(conn) == []
    assert _allows(store, scope_id, "human:alice", "read", "recall")
    assert not _allows(store, scope_id, "human:alice", "read", "derive")


def test_revoked_legacy_rows_hidden_by_default(store, scope_id, alice):
    with store.tx() as conn:
        _insert_legacy(
            conn, scope_id, alice, "g:ldead", "[]", "legacy_empty",
            revoked_us=T0 + 1,
        )
        assert legacy_purpose_grants(conn) == []
        assert len(legacy_purpose_grants(conn, include_revoked=True)) == 1


def test_delegate_from_legacy_parent(store, scope_id, alice):
    """A legacy ANY-evaluating parent delegates normally: SET children
    attenuate it, explicit empties yield NONE children."""
    with store.tx() as conn:
        _insert_legacy(
            conn, scope_id, alice, "g:lpar", "[]", None, depth=2
        )
        _, child = delegate_grant(
            conn, parent_grant_id="g:lpar", delegate_id="agent:a1",
            purposes={"recall"},
        )
        assert get_grant(conn, child)["purpose_tag"] == "set"
        _, dead = delegate_grant(
            conn, parent_grant_id="g:lpar", delegate_id="agent:a2",
            purposes=[],
        )
        assert get_grant(conn, dead)["purpose_tag"] == "none"
    assert _allows(store, scope_id, "agent:a1", "read", "recall")
    assert not _allows(store, scope_id, "agent:a2", "read", "recall")


def test_delegate_under_malformed_constraint_parent_denied(
    store, scope_id, alice
):
    with store.tx() as conn:
        _insert_legacy(
            conn, scope_id, alice, "g:badpar", "[]", "bogus", depth=2
        )
        with pytest.raises(VerbatimError) as exc:
            delegate_grant(
                conn, parent_grant_id="g:badpar", delegate_id="agent:a1",
            )
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


# ---------------------------------------------------------------------
# V4-11.10: authority-expansion sweep over grant/delegation combos
# ---------------------------------------------------------------------


def test_no_authority_expansion_sweep(store, scope_id, alice):
    """Property-style search: for every parent constraint × delegation
    request combination, either the delegation is rejected outright or
    every (verb, purpose) the child authorizes is authorized for the
    parent — including empty sets, tag mixing, and expiry edges."""
    far = FAR_FUTURE()
    parent_specs = [
        {"purposes": PurposeConstraint.any()},
        {"purposes": ["recall", "derive"]},
        {"purposes": ["recall"]},
        {"purposes": PurposeConstraint.none()},
        {"purposes": []},                      # explicit empty → NONE
        {"expires_us": far},
        {"purposes": ["recall"], "expires_us": far},
        {"verbs": {"read"}},                   # single-verb ANY parent
    ]
    child_specs = [
        {},
        {"purposes": []},
        {"purposes": ["recall"]},
        {"purposes": ["recall", "derive"]},
        {"purposes": ["evaluate"]},
        {"purposes": PurposeConstraint.any()},
        {"purposes": PurposeConstraint.none()},
        {"purposes": PurposeConstraint.set(["recall"])},
        {"verbs": {"read"}},
        {"verbs": {"read", "share"}},
        {"expires_us": far - 1},
        {"expires_us": far + 1},
        {"expires_us": far},
    ]
    verbs_matrix = ("read", "quote", "share", "derive", "ingest")
    purposes_matrix = (None, "recall", "derive", "evaluate")

    def _ok(conn, pid, verb, purpose):
        try:
            authorize(conn, _caller(pid), scope_id, verb, purpose=purpose)
            return True
        except VerbatimError as exc:
            assert exc.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
            return False

    combos = delegated = rejected = 0
    for pi, pspec in enumerate(parent_specs):
        with store.tx() as conn:
            parent = create_grant(
                conn, scope_id=scope_id, principal_id=alice,
                verbs=pspec.get("verbs", {"read", "quote"}),
                purposes=pspec.get("purposes", PurposeConstraint.any()),
                expires_us=pspec.get("expires_us"),
                issuer_id=alice, delegation_depth=2,
            )
        for ci, cspec in enumerate(child_specs):
            combos += 1
            delegatee = f"agent:s{pi}x{ci}"
            with store.tx() as conn:
                try:
                    _, child = delegate_grant(
                        conn, parent_grant_id=parent,
                        delegate_id=delegatee,
                        verbs=cspec.get("verbs"),
                        purposes=cspec.get("purposes"),
                        expires_us=cspec.get("expires_us"),
                    )
                except VerbatimError as exc:
                    rejected += 1
                    assert exc.code in (
                        ErrorCode.VALIDATION,
                        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
                    )
                    continue
            delegated += 1
            with store.read() as conn:
                crow = get_grant(conn, child)
                for verb in verbs_matrix:
                    for purpose in purposes_matrix:
                        child_ok = _ok(conn, delegatee, verb, purpose)
                        parent_ok = _ok(conn, "human:alice", verb, purpose)
                        assert not child_ok or parent_ok, (
                            f"authority expansion: parent={pspec} "
                            f"child={cspec} allowed ({verb}, {purpose})"
                        )
            # the child row itself must carry a real tag
            assert crow["purpose_tag"] in ("any", "set", "none")
    # the sweep must actually exercise both outcomes
    assert delegated > 0 and rejected > 0
    assert combos == len(parent_specs) * len(child_specs)
