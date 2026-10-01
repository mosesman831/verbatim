"""connector_pull durable-job path (V4-48.05/48.09): enqueue → drain →
real pull with per-page lease fencing; malformed jobs fail loudly.
"""

from __future__ import annotations

import json

from verbatim.config import VerbatimConfig
from verbatim.core.types import JobKind
from verbatim.ingest import Ingester

from tests.connectors.conftest import PRINCIPAL, provision, qrows, write_file


def _ingester(store):
    return Ingester(store, VerbatimConfig())


def test_connector_pull_job_drains_to_completion(store, service, tmp_path):
    aid = provision(store)
    root = tmp_path / "inbox"
    write_file(root, "a.txt", b"alpha\n")
    write_file(root, "b.txt", b"beta\n")

    jid = service.schedule_pull(
        "local_files",
        {"dir": str(root)},
        scope_id="scopeA",
        principal_id=PRINCIPAL,
        authorization_id=aid,
    )
    ing = _ingester(store)
    done = ing.run_pending(owner="w-test", limit=32)
    assert done >= 1

    with store.read() as conn:
        job = qrows(conn, "SELECT * FROM jobs WHERE job_id=?", (jid,))[0]
        assert job["state"] == "succeeded"
        # The pull really ran: sources + batch + cursor committed.
        assert conn.execute(
            "SELECT COUNT(*) FROM sources WHERE scope_id='scopeA'"
        ).fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM ingest_batches"
        ).fetchone()[0] == 1
        cur = conn.execute(
            "SELECT cursor_value FROM connector_cursors"
        ).fetchone()
        assert cur is not None and cur[0] == "b.txt"
        # Operation receipt journaled the pull report (V2-39.10).
        op = qrows(
            conn,
            "SELECT * FROM operations WHERE operation_key=?",
            (job["operation_key"],),
        )
        assert len(op) == 1
        receipt = json.loads(op[0]["receipt_json"])
        assert receipt["connector_id"] == "local_files"
        assert receipt["inserted"] == 2
        assert receipt["batch_id"]


def test_connector_pull_job_replay_is_idempotent(
    store, service, tmp_path
):
    provision(store)
    root = tmp_path / "inbox"
    write_file(root, "a.txt", b"alpha\n")
    jid = service.schedule_pull(
        "local_files",
        {"dir": str(root)},
        scope_id="scopeA",
        principal_id=PRINCIPAL,
    )
    ing = _ingester(store)
    ing.run_pending(owner="w1", limit=32)
    # A second drain finds nothing queued; the job stays succeeded.
    assert ing.run_pending(owner="w1", limit=32) == 0
    with store.read() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM sources"
        ).fetchone()[0] == 1
        job = qrows(conn, "SELECT state FROM jobs WHERE job_id=?", (jid,))
        assert job[0]["state"] == "succeeded"


def test_connector_pull_malformed_refs_fail_typed(store, service):
    provision(store)
    ing = _ingester(store)
    with store.tx() as conn:
        jid = ing.jobs.enqueue(
            conn,
            "scopeA",
            JobKind.CONNECTOR_PULL,
            {"connector_id": "local_files"},  # missing source/scope/principal
        )
    ing.run_pending(owner="w-test", limit=8)
    with store.read() as conn:
        row = qrows(conn, "SELECT * FROM jobs WHERE job_id=?", (jid,))[0]
        assert row["state"] == "failed"
        assert row["error_code"] == "VALIDATION"


def test_connector_pull_unknown_connector_fails_typed(
    store, service, tmp_path
):
    provision(store)
    root = tmp_path / "inbox"
    write_file(root, "a.txt", b"alpha\n")
    ing = _ingester(store)
    with store.tx() as conn:
        jid = ing.jobs.enqueue(
            conn,
            "scopeA",
            JobKind.CONNECTOR_PULL,
            {
                "connector_id": "does_not_exist",
                "source": {"dir": str(root)},
                "scope_id": "scopeA",
                "principal_id": PRINCIPAL,
            },
        )
    ing.run_pending(owner="w-test", limit=8)
    with store.read() as conn:
        row = qrows(conn, "SELECT * FROM jobs WHERE job_id=?", (jid,))[0]
        assert row["state"] == "failed"
        assert row["error_code"] == "NOT_FOUND_OR_UNAUTHORIZED"
