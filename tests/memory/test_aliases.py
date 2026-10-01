"""Namespace-alias registry tests (SPEC_V5 §05.12–§05.13).

Real on-disk ``Store`` — alias records live in the ``meta`` KV table and
namespace provisioning writes real scope rows + grants inside the caller's
transaction.  No mocks.
"""

from __future__ import annotations

import pytest

from verbatim.core.time import now_us
from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.governance import CallerV3, effective_verbs
from verbatim.memory import aliases
from verbatim.storage.store import Store

OWNER = "local-owner"
PROFILE = "local"


@pytest.fixture()
def store(tmp_path):
    s = Store.create(str(tmp_path / "alias.db"))
    yield s
    s.close()


def _provision(store, **kw):
    kw.setdefault("owner", OWNER)
    kw.setdefault("profile_id", PROFILE)
    kw.setdefault("verbs", ("read", "quote", "ingest", "derive", "review", "admin"))
    with store.tx() as conn:
        import verbatim.governance as gov

        gov.seed_purposes(conn)
        return aliases.provision(conn, store, **kw)


def _resolve(store, **kw):
    kw.setdefault("owner", OWNER)
    kw.setdefault("profile_id", PROFILE)
    with store.read() as conn:
        return aliases.resolve(conn, store, **kw)


class TestValidation:
    def test_label_must_be_string(self):
        with pytest.raises(VerbatimError) as e:
            aliases.validate_label(42, kind="user")
        assert e.value.code == ErrorCode.VALIDATION

    def test_label_not_empty(self):
        for bad in ("", "   ", "\t\n"):
            with pytest.raises(VerbatimError):
                aliases.validate_label(bad, kind="user")

    def test_label_no_control_chars(self):
        with pytest.raises(VerbatimError):
            aliases.validate_label("bad\x00label", kind="user")
        with pytest.raises(VerbatimError):
            aliases.validate_label("bad\x7flabel", kind="run")

    def test_label_bounded(self):
        with pytest.raises(VerbatimError):
            aliases.validate_label("x" * 300, kind="user")

    def test_kind_membership(self):
        for bad in ("admin", "", "USER", None, 5):
            with pytest.raises(VerbatimError):
                aliases.validate_kind(bad)
        for good in aliases.KINDS:
            assert aliases.validate_kind(good) == good

    def test_label_stripped(self):
        assert aliases.validate_label("  alice  ", kind="user") == "alice"


class TestProvision:
    def test_provision_creates_namespace_and_grant(self, store):
        rec = _provision(store, kind="user", label="alice")
        ns = rec["namespace"]
        assert ns.startswith("ns_")
        with store.read() as conn:
            row = conn.execute(
                "SELECT owner_principal_id FROM scopes WHERE scope_id = ?", (ns,)
            ).fetchone()
            assert row is not None and row[0] == OWNER
            verbs = effective_verbs(conn, CallerV3(principal_id=OWNER), ns)
            assert {"read", "quote", "ingest", "derive", "review", "admin"} <= {
                str(v) for v in verbs
            }

    def test_resolve_roundtrip(self, store):
        rec = _provision(store, kind="user", label="alice")
        got = _resolve(store, kind="user", label="alice")
        assert got is not None and got["namespace"] == rec["namespace"]

    def test_labels_are_opaque_on_disk(self, store):
        _provision(store, kind="user", label="hunter2-secret-label")
        with store.read() as conn:
            keys = [
                r[0]
                for r in conn.execute(
                    "SELECT key FROM meta WHERE key LIKE 'memory.alias.%'"
                )
            ]
        assert keys and all("hunter2" not in k for k in keys)

    def test_alias_is_owner_scoped(self, store):
        rec = _provision(store, kind="user", label="shared-name")
        other = _resolve(store, owner="someone-else", kind="user", label="shared-name")
        assert other is None and rec["owner"] == OWNER

    def test_namespace_ids_are_opaque(self, store):
        rec = _provision(store, kind="agent", label="agent-007")
        assert "agent-007" not in rec["namespace"]


class TestRunAliases:
    def test_run_alias_expires(self, store):
        now = now_us()
        rec = _provision(store, kind="run", label="task-1", ttl_s=60, now=now)
        assert rec["expires_us"] == now + 60 * 1_000_000
        assert _resolve(store, kind="run", label="task-1", now=now) is not None
        assert _resolve(store, kind="run", label="task-1", now=now + 61 * 1_000_000) is None

    def test_run_default_ttl(self, store):
        now = now_us()
        rec = _provision(store, kind="run", label="task-2", now=now)
        assert rec["expires_us"] == now + aliases.RUN_DEFAULT_TTL_S * 1_000_000

    def test_run_alias_not_durable_search(self, store):
        rec = _provision(store, kind="run", label="task-3")
        assert rec["ephemeral"] is True
        assert rec["durable_search"] is False
        rec2 = _provision(store, kind="user", label="alice")
        assert rec2["durable_search"] is True
        assert rec2["ephemeral"] is False

    def test_ttl_rejected_for_non_run(self, store):
        with pytest.raises(VerbatimError) as e:
            _provision(store, kind="user", label="alice", ttl_s=10)
        assert e.value.code == ErrorCode.VALIDATION

    def test_ttl_must_be_positive(self, store):
        for bad in (0, -5, "soon"):
            with pytest.raises(VerbatimError):
                _provision(store, kind="run", label="t", ttl_s=bad)


class TestListing:
    def test_list_for_owner(self, store):
        _provision(store, kind="user", label="a")
        _provision(store, kind="agent", label="b")
        _provision(store, kind="run", label="c")
        with store.read() as conn:
            mine = aliases.list_for_owner(conn, store, OWNER)
            foreign = aliases.list_for_owner(conn, store, "nobody")
        assert {r["kind"] for r in mine} == {"user", "agent", "run"}
        assert foreign == []
