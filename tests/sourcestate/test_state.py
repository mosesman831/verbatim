"""source_state/v1 — registration, serialization, integrity, recovery.

Covers the V5-03.06/V5-14.09 handler surface: install (migration),
kernel access registration, provenance edges, digest integrity,
doc-based recovery, idempotent ensure, and honest adoption defaults.
"""

from __future__ import annotations

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v3 import Verb
from verbatim.governance import CallerV3, seed_purposes
from verbatim.kernel import Kernel
from verbatim.sourcestate import (
    HEAD_UNRESOLVED,
    OBJECT_KIND,
    PRODUCER_ID,
    SOURCE_STATE_KIND,
    SourceState,
    adopt,
    artifact_id,
    current_state,
    ensure_state,
    get_state,
    history,
    install_schema,
    recover,
    serialize_doc,
    doc_digest,
    parse_doc,
    verify,
)
from verbatim.storage import repos_v4
from verbatim.storage.repos import has_table

from .conftest import NS, SID, bootstrap_principal, seed_scope, seed_source


def _ensure(conn, store, sid=SID, ns=NS, **kw):
    return ensure_state(conn, sid, ns, store=store, **kw)


class TestInstallAndEnsure:
    def test_install_creates_table_idempotently(self, store):
        """Migration handler: creates the §3 table when absent (e.g. a
        store that skipped the v5 migration), no-ops when present."""
        with store.tx() as conn:
            # The v5 storage migration already created it on this store —
            # drop it inside the tx to exercise the handler genuinely.
            conn.execute("DROP TABLE source_state")
            assert not has_table(conn, "source_state")
            install_schema(conn)
            assert has_table(conn, "source_state")
            install_schema(conn)  # idempotent
            marker = conn.execute(
                "SELECT value_json FROM meta WHERE key = 'source_state.install'"
            ).fetchone()
            assert marker is not None

    def test_install_noop_on_migrated_store(self, store):
        with store.tx() as conn:
            assert has_table(conn, "source_state")
            install_schema(conn)  # must not clobber/conflict
            assert has_table(conn, "source_state")

    def test_reads_fail_closed_without_table(self, store):
        with store.tx() as conn:
            conn.execute("DROP TABLE source_state")
            with pytest.raises(VerbatimError) as ei:
                get_state(conn, SID)
            assert ei.value.code is ErrorCode.SCHEMA_UNSUPPORTED

    def test_get_state_absent(self, installed):
        with installed.read() as conn:
            assert get_state(conn, SID) is None

    def test_ensure_creates_control_version_zero(self, installed):
        with installed.tx() as conn:
            seed_source(conn, installed, SID, NS)
            st = _ensure(conn, installed)
        assert st.control_version == 0
        assert st.disposition == "active"
        assert st.mutation_head == "1"  # bound to the appended revision
        assert st.namespace == NS
        assert st.producer == PRODUCER_ID
        with installed.read() as conn:
            again = get_state(conn, SID)
        assert again == st

    def test_ensure_explicit_head(self, installed):
        with installed.tx() as conn:
            seed_source(conn, installed, SID, NS)
            st = _ensure(conn, installed, head=3)
        assert st.mutation_head == "3"
        assert st.head_revision == 3

    def test_ensure_unknown_source_head_unresolved(self, installed):
        with installed.tx() as conn:
            seed_scope(conn)
            st = _ensure(conn, installed)
        assert st.mutation_head == HEAD_UNRESOLVED
        assert st.head_revision is None

    def test_ensure_idempotent_no_version_bump(self, installed):
        with installed.tx() as conn:
            seed_source(conn, installed, SID, NS)
            first = _ensure(conn, installed)
            second = _ensure(conn, installed)
        assert first == second
        with installed.read() as conn:
            assert len(history(conn, SID)) == 1

    def test_ensure_namespace_conflict_is_integrity(self, installed):
        with installed.tx() as conn:
            seed_source(conn, installed, SID, NS)
            _ensure(conn, installed)
        with installed.tx() as conn:
            with pytest.raises(VerbatimError) as ei:
                ensure_state(conn, SID, "ns:other")
            assert ei.value.code is ErrorCode.INTEGRITY

    def test_adopt_records_unresolved_not_active(self, installed):
        with installed.tx() as conn:
            seed_source(conn, installed, SID, NS)
            st = adopt(conn, SID, NS, store=installed)
        # V5-14.16: unknown legacy lifecycle stays unresolved — the real
        # head revision exists but adoption must not infer approval.
        assert st.disposition == "recorded"
        assert st.mutation_head == HEAD_UNRESOLVED
        with installed.read() as conn:
            cur = current_state(conn, SID)
        assert cur.label == "recorded"
        assert cur.current is False


class TestReaderSeam:
    def test_get_returns_contract_row_dict(self, installed):
        """The facade's duck-typed probe (``get``/``current``/…) binds
        this dict-returning reader (memory/controls.py)."""
        from verbatim.sourcestate import get

        with installed.tx() as conn:
            seed_source(conn, installed, SID, NS)
            _ensure(conn, installed)
        with installed.read() as conn:
            row = get(conn, SID)
            assert row["source_id"] == SID
            assert row["control_version"] == 0
            assert row["disposition"] == "active"
            assert get(conn, "src9999") is None


class TestRegistration:
    """The artifact registers under the existing object/revision/
    coordinator interfaces (V5-14.09) — kernel access included."""

    def test_registry_rows_and_producer(self, installed):
        with installed.tx() as conn:
            seed_source(conn, installed, SID, NS)
            _ensure(conn, installed)
            oid = artifact_id(SID)
            obj = repos_v4.get(
                conn, "objects", {"kind": OBJECT_KIND, "object_id": oid}
            )
            assert obj is not None
            assert obj["scope_id"] == NS
            assert obj["disposition"] == "active"
            assert obj["current_revision"] == 1  # cv0 -> doc rev 1
            rev = repos_v4.get(
                conn,
                "object_revisions",
                {"kind": OBJECT_KIND, "object_id": oid, "revision": 1},
            )
            assert rev is not None and rev["digest"].startswith("sha256:")
            prod = repos_v4.get(
                conn, "producer_manifests", {"producer_id": PRODUCER_ID}
            )
            assert prod is not None and prod["health"] == "available"

    def test_provenance_edges(self, installed):
        with installed.tx() as conn:
            seed_source(conn, installed, SID, NS)
            _ensure(conn, installed)
            oid = artifact_id(SID)
            parents = conn.execute(
                "SELECT parent_kind, parent_id, parent_revision"
                " FROM derivations"
                " WHERE child_kind = ? AND child_id = ?",
                (OBJECT_KIND, oid),
            ).fetchall()
        assert ("source_revision", SID, 1) in [
            (k, i, r) for k, i, r in parents
        ]

    def test_kernel_access_resolves_artifact_not_source(self, installed):
        """Registration must not shadow the source id (kernel ambiguity)."""
        pid = "human:alice"
        with installed.tx() as conn:
            seed_source(conn, installed, SID, NS)
            bootstrap_principal(conn)
            _ensure(conn, installed)
        kernel = Kernel(installed)
        caller = CallerV3(principal_id=pid)
        with installed.read() as conn:
            # The artifact resolves under its own kind…
            lease = kernel.resolve_access(
                conn,
                caller,
                Verb.READ,
                "recall",
                [NS],
                [("source_state", artifact_id(SID), 1)],
            )
            assert not lease.denied
            # …and the source id still resolves cleanly as a source —
            # no kind ambiguity introduced (kernel._resolve denies
            # multi-kind ids).
            lease2 = kernel.resolve_access(
                conn,
                caller,
                Verb.QUOTE,
                "recall",
                [NS],
                [("source", SID, 1)],
            )
            assert not lease2.denied

    def test_kernel_access_denied_without_grant(self, installed):
        with installed.tx() as conn:
            seed_source(conn, installed, SID, NS)
            seed_purposes(conn)
            _ensure(conn, installed)
        kernel = Kernel(installed)
        caller = CallerV3(principal_id="human:mallory")
        with installed.read() as conn:
            lease = kernel.resolve_access(
                conn,
                caller,
                Verb.READ,
                "recall",
                [NS],
                [("source_state", artifact_id(SID), 1)],
            )
            assert lease.denied  # indistinguishable denial


class TestSerializationIntegrityRecovery:
    def test_state_roundtrip(self, installed):
        with installed.tx() as conn:
            seed_source(conn, installed, SID, NS)
            st = _ensure(conn, installed)
        assert SourceState.from_dict(st.to_dict()) == st

    def test_doc_roundtrip_and_digest_stable(self, installed):
        with installed.tx() as conn:
            seed_source(conn, installed, SID, NS)
            _ensure(conn, installed)
            (doc,) = history(conn, SID)
        assert doc["kind"] == SOURCE_STATE_KIND
        assert doc["change"] == "create"
        assert doc["control_version"] == 0
        assert doc["registry_revision"] == 1
        text = serialize_doc(doc)
        assert parse_doc(text) == doc
        assert doc_digest(doc) == doc_digest(parse_doc(text))

    def test_parse_doc_rejects_foreign_kind(self):
        with pytest.raises(VerbatimError) as ei:
            parse_doc('{"kind": "branch/v1"}')
        assert ei.value.code is ErrorCode.INTEGRITY

    def test_tampered_doc_fails_verify(self, installed):
        with installed.tx() as conn:
            seed_source(conn, installed, SID, NS)
            _ensure(conn, installed)
        with installed.tx() as conn:
            conn.execute(
                "UPDATE object_revisions SET metadata_json = ?"
                " WHERE kind = ? AND object_id = ?",
                (
                    serialize_doc(
                        {"kind": SOURCE_STATE_KIND, "source_id": SID,
                         "disposition": "erased", "control_version": 0,
                         "mutation_head": "1", "namespace": NS,
                         "known_at": "x", "updated_at": "x",
                         "producer": "evil", "superseded_by": None,
                         "effective_at": None, "valid_from": None,
                         "valid_to": None}
                    ),
                    OBJECT_KIND,
                    artifact_id(SID),
                ),
            )
        with installed.read() as conn:
            with pytest.raises(VerbatimError) as ei:
                verify(conn, SID)
            assert ei.value.code is ErrorCode.STORE_CORRUPT

    def test_recover_rebuilds_lost_row(self, installed):
        with installed.tx() as conn:
            seed_source(conn, installed, SID, NS)
            st = _ensure(conn, installed)
        with installed.tx() as conn:
            conn.execute(
                "DELETE FROM source_state WHERE source_id = ?", (SID,)
            )
            assert get_state(conn, SID) is None
            # docs survive the lost row — ensure_state must not pretend
            # this is a fresh create (typed INTEGRITY, recover() instead).
            with pytest.raises(VerbatimError) as ei:
                _ensure(conn, installed)
            assert ei.value.code is ErrorCode.INTEGRITY
            rebuilt = recover(conn, SID, store=installed)
        assert rebuilt == st

    def test_recover_flags_divergent_row(self, installed):
        with installed.tx() as conn:
            seed_source(conn, installed, SID, NS)
            _ensure(conn, installed)
        with installed.tx() as conn:
            conn.execute(
                "UPDATE source_state SET disposition = 'retracted'"
                " WHERE source_id = ?",
                (SID,),
            )
            with pytest.raises(VerbatimError) as ei:
                recover(conn, SID)
            assert ei.value.code is ErrorCode.INTEGRITY

    def test_recover_absent_history_returns_none(self, installed):
        with installed.read() as conn:
            assert recover(conn, SID) is None
