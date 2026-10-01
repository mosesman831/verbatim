"""V3 foundation tests: schema v3 creation, 2→3 migration, queue lanes,
v3 config sections, and the shared repos layer (SPEC_V3 §39–§40, §57).

These tests pin the additive contract: a v3 store must accept every v3
kind/state while every v2 path keeps working on the same schema.
"""

from __future__ import annotations

import sqlite3

import pytest

from verbatim.config import config_from_mapping
from verbatim.core.types import ErrorCode, JobKind, VerbatimError
from verbatim.jobs.queue import JobQueue
from verbatim.storage import migrations, repos_v3
from verbatim.storage.schema import (
    DDL_FTS5,
    DDL_V1,
    DDL_V2,
    DDL_V2_ALTER,
    DDL_V2_JOBS_REBUILD,
    FTS_TRIGGERS,
    SCHEMA_VERSION,
)
from verbatim.storage.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "v3.db"))
    yield s
    s.close()


@pytest.fixture
def scope_id(store):
    sid = "scope:foundation"
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES (?, 'prof', 'owner')",
            (sid,),
        )
        # 's1' is the throwaway scope several tests write against.
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES ('s1', 'prof', 'owner')"
        )
    return sid


def _table_names(conn):
    return {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


# --------------------------------------------------------------------- schema


def test_fresh_store_is_schema_v3(store):
    assert store.schema_version == SCHEMA_VERSION


def test_fresh_store_has_all_v3_tables(store):
    with store.read() as conn:
        names = _table_names(conn)
    expected = {
        "principals", "perspectives", "perspective_subjects", "purposes",
        "grants_v3", "delegations", "capture_authorizations",
        "source_envelopes", "trajectories", "trajectory_steps",
        "step_observations", "state_anchors", "transitions",
        "transition_anchors", "procedure_signatures", "procedure_exposures",
        "observations", "observation_evidence", "derivations",
        "security_labels", "quarantine", "vault_entries", "vault_refs",
        "action_tickets", "ticket_objects", "value_handles",
        "redaction_spans", "propagations", "routing_decisions",
        "routing_stats", "influence", "freshness", "environment_state",
        "working_sets", "working_set_items", "social_memory",
        "learning_snapshots", "index_generations", "replay_runs",
    }
    missing = expected - names
    assert not missing, f"missing v3 tables: {sorted(missing)}"


def test_procedures_check_accepts_v3_ladder(store, scope_id):
    with store.tx() as conn:
        for state in ("candidate", "reviewed", "active", "deprecated", "retired"):
            conn.execute(
                "INSERT INTO procedures (procedure_id, scope_id, task_label, state)"
                " VALUES (?, 's1', 't', ?)",
                (f"p-{state}", state),
            )
        # legacy v2 states still valid (additive contract)
        conn.execute(
            "INSERT INTO procedures (procedure_id, scope_id, task_label, state)"
            " VALUES ('p-legacy', 's1', 't', 'proposed')"
        )


def test_procedures_rejects_unknown_state(store, scope_id):
    with store.tx() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO procedures (procedure_id, scope_id, task_label, state)"
                " VALUES ('p-bad', 's1', 't', 'vaporware')"
            )


def test_jobs_accepts_all_v3_kinds(store, scope_id):
    v3_kinds = [
        "screen", "sparse_index", "late_index", "signature_index",
        "episode_build", "transition_build", "procedure_compile",
        "procedure_refine", "consolidate", "purge_derived", "purge_vault",
        "quarantine_review", "revocation_notify", "vault_rotate",
        "projection_sync", "connector_pull",
    ]
    with store.tx() as conn:
        for kind in v3_kinds:
            conn.execute(
                "INSERT INTO jobs (job_id, scope_id, kind, state)"
                " VALUES (?, ?, ?, 'queued')",
                (f"j-{kind}", scope_id, kind),
            )


def test_jobs_accepts_v3_lanes(store, scope_id):
    with store.tx() as conn:
        for lane in ("ordinary", "background", "privacy_control", "maintenance"):
            conn.execute(
                "INSERT INTO jobs (job_id, scope_id, kind, state, lane)"
                " VALUES (?, ?, 'harvest', 'queued', ?)",
                (f"j-lane-{lane}", scope_id, lane),
            )


# ----------------------------------------------------------------- migration


def _make_v2_store(path: str) -> None:
    """Build a genuine schema-v2 database (pre-v3 DDL) for migration tests."""
    import os
    with open(path + ".key", "wb") as fh:
        fh.write(os.urandom(32))
    os.chmod(path + ".key", 0o600)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(DDL_V1)
    conn.executescript(DDL_V2)
    for stmt in DDL_V2_ALTER.split(";"):
        if stmt.strip():
            conn.execute(stmt)
    for stmt in DDL_V2_JOBS_REBUILD.split(";"):
        if stmt.strip():
            conn.execute(stmt)
    conn.executescript(DDL_FTS5)
    conn.executescript(FTS_TRIGGERS)
    from verbatim.core.types import json_dumps
    from verbatim.core.time import now_us

    for key, value in (
        ("schema_version", 2),
        ("db_id", "db-v2-fixture"),
        ("created_us", now_us()),
        ("projection_generation", 1),
        ("policy_epoch", 0),
    ):
        conn.execute(
            "INSERT INTO meta(key, value_json) VALUES (?, ?)",
            (key, json_dumps(value)),
        )
    conn.execute(
        "INSERT INTO scopes (scope_id, profile_id, visibility)"
        " VALUES ('s1', 'prof', 'owner')"
    )
    # A v2 procedure in the legacy state must map forward on migration.
    conn.execute(
        "INSERT INTO procedures (procedure_id, scope_id, task_label, state)"
        " VALUES ('legacy-proc', 's1', 't', 'proposed')"
    )
    conn.execute(
        "INSERT INTO procedures (procedure_id, scope_id, task_label, state)"
        " VALUES ('review-proc', 's1', 't', 'review')"
    )
    conn.commit()
    conn.close()


def test_migrate_v2_to_v3(tmp_path):
    db = str(tmp_path / "v2.db")
    _make_v2_store(db)
    # A writable Store.open migrates in place (SPEC §21) — apply() is
    # idempotent and its sniffers also resume crash-interrupted rebuilds.
    store = Store.open(db)
    assert store.schema_version == SCHEMA_VERSION
    version = migrations.apply(store)
    assert version == SCHEMA_VERSION
    with store.read() as conn:
        names = _table_names(conn)
        assert "vault_entries" in names
        assert "derivations" in names
        states = {
            r[0]
            for r in conn.execute("SELECT state FROM procedures")
        }
    # V3-57.05: proposed→candidate, review→reviewed
    assert states == {"candidate", "reviewed"}
    store.close()


def test_migrated_v2_jobs_accept_v3_kinds(tmp_path):
    db = str(tmp_path / "v2b.db")
    _make_v2_store(db)
    store = Store.open(db)
    migrations.apply(store)
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, scope_id, kind, state)"
            " VALUES ('j1', 's1', 'procedure_compile', 'queued')"
        )
        conn.execute(
            "INSERT INTO jobs (job_id, scope_id, kind, state, lane)"
            " VALUES ('j2', 's1', 'harvest', 'queued', 'privacy_control')"
        )
    store.close()


# -------------------------------------------------------------------- queue


def _enq(queue, store, sid, kind):
    with store.tx() as conn:
        return queue.enqueue(conn, sid, kind, {})


def test_v3_lane_defaults(store, scope_id):
    q = JobQueue(store)
    expected = {
        JobKind.PURGE_VAULT: "privacy_control",
        JobKind.QUARANTINE_REVIEW: "privacy_control",
        JobKind.REVOCATION_NOTIFY: "privacy_control",
        JobKind.VAULT_ROTATE: "privacy_control",
        JobKind.PURGE_DERIVED: "privacy_control",
        JobKind.SPARSE_INDEX: "maintenance",
        JobKind.PROJECTION_SYNC: "maintenance",
        JobKind.CONNECTOR_PULL: "maintenance",
        JobKind.PROCEDURE_COMPILE: "background",
        JobKind.CONSOLIDATE: "background",
        JobKind.EPISODE_BUILD: "background",
        JobKind.SCREEN: "privacy_control",
        # legacy v2 control kinds keep their reserved lane
        JobKind.PURGE: "control",
        JobKind.REINDEX: "control",
    }
    with store.tx() as conn:
        for kind, lane in expected.items():
            jid = q.enqueue(conn, scope_id, kind, {})
            row = conn.execute(
                "SELECT lane FROM jobs WHERE job_id = ?", (jid,)
            ).fetchone()
            assert row[0] == lane, f"{kind.value} -> {row[0]}, want {lane}"


def test_privacy_control_leases_before_ordinary(store, scope_id):
    q = JobQueue(store)
    _enq(q, store, scope_id, JobKind.COMPARE)          # ordinary, enqueued first
    _enq(q, store, scope_id, JobKind.PURGE_VAULT)     # privacy_control
    leased = q.lease(scope_id, [JobKind.COMPARE, JobKind.PURGE_VAULT], owner="w1")
    assert leased[0]["lane"] == "privacy_control"


def test_lane_filter_accepts_v3_lanes(store, scope_id):
    q = JobQueue(store)
    _enq(q, store, scope_id, JobKind.PROCEDURE_COMPILE)
    _enq(q, store, scope_id, JobKind.COMPARE)
    bg = q.lease(scope_id, list(JobKind), owner="w1", lane="background")
    assert [j["kind"] for j in bg] == ["procedure_compile"]


# ------------------------------------------------------------------- config


def test_v3_config_defaults():
    cfg = config_from_mapping({})
    assert cfg.v3.profile == "embedded"
    assert cfg.v3.retrieval.controller == "deterministic"
    assert cfg.v3.vault.enabled is False
    assert cfg.v3.capture_depth == 0


def test_v3_config_nested_sections():
    cfg = config_from_mapping({
        "v3": {
            "profile": "local_semantic",
            "capture_depth": 2,
            "retrieval": {"dense": True, "candidate_cap": 64},
            "vault": {"enabled": True, "key_source": "file", "key_file": "/tmp/k"},
        }
    })
    assert cfg.v3.profile == "local_semantic"
    assert cfg.v3.retrieval.dense is True
    assert cfg.v3.retrieval.candidate_cap == 64
    assert cfg.v3.vault.enabled is True
    assert cfg.v3.vault.key_file == "/tmp/k"


def test_v3_config_rejects_unknown_nested_key():
    with pytest.raises(VerbatimError):
        config_from_mapping({"v3": {"retrieval": {"telepathy": True}}})


def test_v3_config_profile_mode_consistency():
    with pytest.raises(VerbatimError) as exc:
        config_from_mapping({"mode": "offline_rules", "v3": {"profile": "split_privacy"}})
    assert exc.value.code == ErrorCode.CONFIG_INVALID
    cfg = config_from_mapping({
        "mode": "remote_assisted", "v3": {"profile": "split_privacy"},
    })
    assert cfg.v3.profile == "split_privacy"


def test_learned_active_controller_refused_until_g5():
    with pytest.raises(VerbatimError):
        config_from_mapping({"v3": {"retrieval": {"controller": "learned_active"}}})


# ------------------------------------------------------------------- repos


def test_repos_v3_roundtrip(store, scope_id):
    with store.tx() as conn:
        repos_v3.insert(conn, "principals", {
            "principal_id": "p1", "kind": "agent",
            "display_name": "tester", "created_us": 1,
        })
        repos_v3.insert(conn, "grants_v3", {
            "grant_id": "g1", "scope_id": scope_id, "principal_id": "p1",
            "verbs_json": ["read", "quote"],  # native list, serialized by repo
            "issuer_id": "p1", "issued_us": 1, "epoch": 0,
        })
        row = repos_v3.get(conn, "grants_v3", {"grant_id": "g1"})
        assert row["scope_id"] == scope_id
        assert repos_v3.json_field(row, "verbs_json") == ["read", "quote"]
        n = repos_v3.update(conn, "grants_v3",
                            {"revoked_us": 99}, {"grant_id": "g1"})
        assert n == 1
        row = repos_v3.get(conn, "grants_v3", {"grant_id": "g1"})
        assert row["revoked_us"] == 99


def test_repos_v3_rejects_unknown_column(store):
    with store.tx() as conn:
        with pytest.raises(VerbatimError):
            repos_v3.insert(conn, "principals", {"principal_id": "x", "backdoor": 1})
        with pytest.raises(VerbatimError):
            repos_v3.query(conn, "grants_v3", {"sql_injection": "x"})
        with pytest.raises(VerbatimError):
            repos_v3.insert(conn, "no_such_table", {"a": 1})


def test_repos_v3_delete_requires_where(store):
    with store.tx() as conn:
        with pytest.raises(VerbatimError):
            repos_v3.delete(conn, "principals", {})
