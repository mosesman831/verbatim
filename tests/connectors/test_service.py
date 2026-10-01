"""Pull-engine invariants (V4-48.03/48.04/48.05/48.06): consent gates,
crash-safe cursor+batch atomicity, remote-unavailable honesty, and the
ledger identity model.
"""

from __future__ import annotations

import json

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.connectors import (
    ConnectorService,
    RemoteConnectorStub,
    get_connector,
    register,
)
from verbatim.connectors.service import ledger_connector_id

from tests.connectors.conftest import PRINCIPAL, provision, qrows, write_file


def test_unauthorized_scope_denied_writes_nothing(store, service, tmp_path):
    root = tmp_path / "inbox"
    write_file(root, "a.txt", b"alpha\n")
    # No grant at all → indistinguishable denial (§10.05).
    with pytest.raises(VerbatimError) as ei:
        service.pull(
            "local_files",
            {"dir": str(root)},
            scope_id="scopeA",
            principal_id=PRINCIPAL,
        )
    assert ei.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    with store.read() as conn:
        for table in (
            "sources",
            "ingest_batches",
            "connector_cursors",
            "source_envelopes",
        ):
            assert (
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                == 0
            ), table


def test_grant_without_capture_consent_denied(store, service, tmp_path):
    provision(store, consent=False)
    root = tmp_path / "inbox"
    write_file(root, "a.txt", b"alpha\n")
    with pytest.raises(VerbatimError) as ei:
        service.pull(
            "local_files",
            {"dir": str(root)},
            scope_id="scopeA",
            principal_id=PRINCIPAL,
        )
    assert ei.value.code is ErrorCode.CONSENT_REQUIRED


def test_dry_run_also_requires_ingest_grant(store, service, tmp_path):
    root = tmp_path / "inbox"
    write_file(root, "a.txt", b"alpha\n")
    with pytest.raises(VerbatimError) as ei:
        service.pull(
            "local_files",
            {"dir": str(root)},
            scope_id="scopeA",
            principal_id=PRINCIPAL,
            dry_run=True,
        )
    assert ei.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_crash_mid_pull_resumes_without_duplicates(
    store, service, tmp_path
):
    provision(store)
    root = tmp_path / "inbox"
    for name in ("a.txt", "b.txt", "c.txt"):
        write_file(root, f"{name}", f"{name}\n".encode())

    # Simulate a crash after the first committed page: the fence raises
    # on every tx after tx #2 (tx1 = batch create; page1 commits a.txt +
    # cursor; page2's tx is fenced out and rolls back).
    calls = {"n": 0}

    def _crashy(conn):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("simulated crash mid-pull")

    with pytest.raises(RuntimeError):
        service.pull(
            "local_files",
            {"dir": str(root)},
            scope_id="scopeA",
            principal_id=PRINCIPAL,
            batch_size=1,
            fence=_crashy,
        )
    with store.read() as conn:
        cur = conn.execute(
            "SELECT cursor_value FROM connector_cursors"
        ).fetchone()
        assert cur is not None and cur[0] == "a.txt"
        n_sources = conn.execute(
            "SELECT COUNT(*) FROM sources"
        ).fetchone()[0]
        assert n_sources == 1

    # Retry from the durable cursor: b+c insert, a.txt is NOT re-scanned.
    report = service.pull(
        "local_files",
        {"dir": str(root)},
        scope_id="scopeA",
        principal_id=PRINCIPAL,
        batch_size=1,
    )
    assert report.cursor_before == "a.txt"
    assert report.inserted == 2
    assert report.duplicates == 0
    with store.read() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM sources"
        ).fetchone()[0] == 3
        # And a full rescan still dedups — retries can never duplicate.
    again = service.pull(
        "local_files",
        {"dir": str(root)},
        scope_id="scopeA",
        principal_id=PRINCIPAL,
        resume_from="",
    )
    assert again.duplicates == 3 and again.inserted == 0


def test_remote_connector_declared_unavailable(store, service, tmp_path):
    register(RemoteConnectorStub("acme_cloud", "Acme Cloud (remote)"))
    provision(store)
    with pytest.raises(VerbatimError) as ei:
        service.pull(
            "acme_cloud",
            {"endpoint": "https://example.invalid"},
            scope_id="scopeA",
            principal_id=PRINCIPAL,
        )
    assert ei.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    desc = get_connector("acme_cloud").descriptor()
    assert desc.remote is True


def test_unknown_connector_denied(service):
    with pytest.raises(VerbatimError) as ei:
        get_connector("nope")
    assert ei.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_batch_manifest_records_identity_and_permission(
    store, service, tmp_path
):
    aid = provision(store)
    root = tmp_path / "inbox"
    write_file(root, "a.txt", b"alpha\n")
    report = service.pull(
        "local_files",
        {"dir": str(root)},
        scope_id="scopeA",
        principal_id=PRINCIPAL,
        purpose="migration",
    )
    batch = service.batches.get_batch(report.batch_id)
    assert batch is not None
    manifest = json.loads(batch["manifest_json"])
    assert manifest["connector_id"] == "local_files"
    assert manifest["connector_version"] == "1.0"
    assert manifest["cursor_before"] is None
    assert manifest["cursor_after"] == "a.txt"
    perm = manifest["permission"]
    assert perm["authorization_id"] == aid
    assert perm["principal_id"] == PRINCIPAL
    assert perm["purpose"] == "migration"
    assert manifest["retention_policy"] == "keep"


def test_pull_report_is_json_serializable(store, service, tmp_path):
    provision(store)
    root = tmp_path / "inbox"
    write_file(root, "a.txt", b"alpha\n")
    report = service.pull(
        "local_files",
        {"dir": str(root)},
        scope_id="scopeA",
        principal_id=PRINCIPAL,
    )
    json.dumps(report.to_dict())  # receipt shape must serialize


def test_ledger_id_discriminates_source(service):
    a = ledger_connector_id("local_files", {"dir": "/x"})
    b = ledger_connector_id("local_files", {"dir": "/y"})
    assert a != b and a.startswith("local_files@")


def test_max_items_partial_pull(store, service, tmp_path):
    provision(store)
    root = tmp_path / "inbox"
    for i in range(4):
        write_file(root, f"f{i}.txt", f"{i}\n".encode())
    report = service.pull(
        "local_files",
        {"dir": str(root)},
        scope_id="scopeA",
        principal_id=PRINCIPAL,
        max_items=2,
    )
    assert report.state == "partial"
    assert report.scanned == 2
    # Resume finishes the rest.
    rest = service.pull(
        "local_files",
        {"dir": str(root)},
        scope_id="scopeA",
        principal_id=PRINCIPAL,
    )
    assert rest.scanned == 2 and rest.state == "complete"
    with store.read() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM sources"
        ).fetchone()[0] == 4


def test_schedule_pull_enqueues_durable_job(store, service, tmp_path):
    provision(store)
    root = tmp_path / "inbox"
    write_file(root, "a.txt", b"alpha\n")
    jid = service.schedule_pull(
        "local_files",
        {"dir": str(root)},
        scope_id="scopeA",
        principal_id=PRINCIPAL,
    )
    with store.read() as conn:
        row = qrows(
            conn, "SELECT * FROM jobs WHERE job_id=?", (jid,)
        )[0]
        assert row["kind"] == "connector_pull"
        assert row["state"] == "queued"
        assert row["lane"] == "maintenance"
        refs = json.loads(row["input_refs_json"])
        assert refs["connector_id"] == "local_files"
        assert refs["scope_id"] == "scopeA"
        assert row["operation_key"]


def test_schedule_pull_requires_consent_too(store, service, tmp_path):
    provision(store, consent=False)
    with pytest.raises(VerbatimError) as ei:
        service.schedule_pull(
            "local_files",
            {"dir": str(tmp_path)},
            scope_id="scopeA",
            principal_id=PRINCIPAL,
        )
    assert ei.value.code is ErrorCode.CONSENT_REQUIRED
