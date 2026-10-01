"""Trusted-owner bootstrap tests (SPEC_V5 §05).

Real on-disk ``Store`` — bootstrap writes principals, scope rows, grants,
the alias record, and the persisted binding inside one transaction.
Convergent reopen and revoked-authority denial are exercised for real.
"""

from __future__ import annotations

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.governance import CallerV3, effective_verbs, get_principal
from verbatim.memory.bootstrap import (
    BINDING_KEY,
    ROOT_VERBS,
    BootstrapInfo,
    ensure_bootstrap,
)
from verbatim.storage.repos import has_table
from verbatim.storage.store import Store

OWNER = "local-owner"
PROFILE = "local"
HOST = "cli"


@pytest.fixture()
def store(tmp_path):
    s = Store.create(str(tmp_path / "boot.db"))
    yield s
    s.close()


def _bootstrap(store, owner=OWNER, label="default"):
    return ensure_bootstrap(
        store,
        owner=owner,
        owner_kind="human",
        profile_id=PROFILE,
        host_name=HOST,
        alias_label=label,
    )


class TestEstablish:
    def test_first_bootstrap_establishes_atomically(self, store):
        info = _bootstrap(store)
        assert isinstance(info, BootstrapInfo)
        assert info.created is True
        assert info.owner == OWNER
        assert info.namespace.startswith("ns_")
        with store.read() as conn:
            binding = store._meta_get(conn, BINDING_KEY)
            assert binding["owner"] == OWNER
            assert binding["profile_id"] == PROFILE
            principal = get_principal(conn, OWNER)
            assert principal is not None and principal["kind"] == "human"
            row = conn.execute(
                "SELECT owner_principal_id FROM scopes WHERE scope_id = ?",
                (info.namespace,),
            ).fetchone()
            assert row[0] == OWNER
            verbs = effective_verbs(
                conn, CallerV3(principal_id=OWNER), info.namespace
            )
            assert set(ROOT_VERBS) <= {str(v) for v in verbs}

    def test_reopen_converges(self, store):
        first = _bootstrap(store)
        second = _bootstrap(store)
        assert second.created is False
        assert second.namespace == first.namespace
        assert second.owner == first.owner

    def test_no_agent_capture_authorization_minted(self, store):
        """Ownership never arrives via agent-capture consent (V5-05.05)."""
        _bootstrap(store)
        with store.read() as conn:
            if not has_table(conn, "capture_authorizations"):
                return
            rows = conn.execute(
                "SELECT COUNT(*) FROM capture_authorizations"
            ).fetchone()
            assert rows[0] == 0


class TestReopenDenial:
    def test_foreign_owner_denied(self, store):
        _bootstrap(store)
        with pytest.raises(VerbatimError) as e:
            _bootstrap(store, owner="mallory")
        assert e.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    def test_revoked_grant_is_never_resurrected(self, store):
        info = _bootstrap(store)
        with store.tx() as conn:
            conn.execute(
                "UPDATE grants_v3 SET revoked_us = ? WHERE scope_id = ?",
                (1, info.namespace),
            )
        with pytest.raises(VerbatimError) as e:
            _bootstrap(store)
        assert e.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    def test_corrupt_binding_denied(self, store):
        _bootstrap(store)
        with store.tx() as conn:
            store._meta_set(conn, BINDING_KEY, {"v": 99})
        with pytest.raises(VerbatimError) as e:
            _bootstrap(store)
        assert e.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    def test_owner_field_is_never_rewritten(self, store):
        """A pre-provisioned scope keeps its established owner."""
        info = _bootstrap(store)
        with store.tx() as conn:
            conn.execute(
                "UPDATE scopes SET owner_principal_id = 'mallory'"
                " WHERE scope_id = ?",
                (info.namespace,),
            )
            # The alias provisioning guard (IS NULL) cannot rewrite an
            # established owner — the UPDATE is a no-op on a bound row.
            conn.execute(
                "UPDATE scopes SET owner_principal_id = ?"
                " WHERE scope_id = ? AND owner_principal_id IS NULL",
                (OWNER, info.namespace),
            )
            row = conn.execute(
                "SELECT owner_principal_id FROM scopes WHERE scope_id = ?",
                (info.namespace,),
            ).fetchone()
            assert row[0] == "mallory"  # hostile rewrite took effect
        # ...and reopen still binds by the persisted binding + grant check,
        # not by the scope row's owner field alone.
        with pytest.raises(VerbatimError):
            _bootstrap(store, owner="mallory")


class TestAliases:
    def test_second_user_alias_provisions_new_namespace(self, store):
        first = _bootstrap(store, label="alice")
        second = _bootstrap(store, label="bob")
        assert second.created is False  # same binding, not a new owner
        assert second.namespace != first.namespace
        again = _bootstrap(store, label="alice")
        assert again.namespace == first.namespace
