"""Profile-plane fixtures — real ``Store.create`` (v1..v4 schema).

Mirrors ``tests/kernel/conftest.py``: the shared ``tests/conftest.py``
TestStore shim only carries DDL_V1, and profiles need the v3 governance
tables (grants/purposes/perspectives/quarantine) plus the v4 registry
(objects/object_revisions/dependency_edges).
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from verbatim.core.types_v3 import Verb
from verbatim.governance import (
    CallerV3,
    create_grant,
    register_principal,
    seed_purposes,
)
from verbatim.profiles import ProfileService
from verbatim.storage.store import Store

T0 = 1_700_000_000_000_000


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "profiles.db"))
    yield s
    s.close()


@pytest.fixture
def svc(store):
    return ProfileService(store)


@pytest.fixture
def seeded(store):
    """Two scopes, seeded purposes, three principals, owner grants."""
    with store.tx() as conn:
        for sid in ("scope:a", "scope:b"):
            conn.execute(
                "INSERT INTO scopes (scope_id, profile_id, visibility)"
                " VALUES (?, 'prof', 'owner')",
                (sid,),
            )
        seed_purposes(conn)
        for pid in ("human:alice", "human:bob", "agent:prod"):
            register_principal(
                conn,
                kind="agent" if pid.startswith("agent") else "human",
                principal_id=pid,
            )
        # alice owns scope:a (full verbs, any purpose); bob is a peer in
        # scope:a for the group-session tests; agent has read+quote only.
        create_grant(
            conn,
            scope_id="scope:a",
            principal_id="human:alice",
            verbs=["read", "quote", "derive", "admin", "share"],
            purposes=None,  # PurposeConstraint.any()
            issuer_id="human:alice",
        )
        create_grant(
            conn,
            scope_id="scope:b",
            principal_id="human:alice",
            verbs=["read", "quote", "derive", "admin"],
            purposes=None,
            issuer_id="human:alice",
        )
        create_grant(
            conn,
            scope_id="scope:a",
            principal_id="human:bob",
            verbs=["read", "quote", "derive"],
            purposes=None,
            issuer_id="human:alice",
        )
        create_grant(
            conn,
            scope_id="scope:a",
            principal_id="agent:prod",
            verbs=["read", "quote"],
            purposes=None,
            issuer_id="human:alice",
        )
    return {"a": "scope:a", "b": "scope:b"}


def caller(pid: str = "human:alice", epoch=None) -> CallerV3:
    return CallerV3(principal_id=pid, epoch=epoch)


def add_source(
    conn: sqlite3.Connection,
    store: Store,
    scope_id: str,
    source_id: str,
    payload: bytes,
    *,
    revision: int = 1,
) -> str:
    conn.execute(
        "INSERT INTO sources"
        "(source_id, origin, external_id, source_kind, scope_id,"
        " speaker_id, created_us)"
        " VALUES (?, 'test', NULL, 'user_message', ?, NULL, ?)",
        (source_id, scope_id, T0),
    )
    conn.execute(
        "INSERT INTO source_revisions"
        "(source_id, revision, payload, payload_hmac, event_us,"
        " captured_us, timezone, provenance, metadata_json)"
        " VALUES (?, ?, ?, ?, ?, ?, NULL, 'direct_user', '{}')",
        (source_id, revision, payload, store.hmac(payload), T0, T0),
    )
    return source_id


def add_span(
    conn: sqlite3.Connection,
    store: Store,
    span_id: str,
    source_id: str,
    revision: int,
    payload: bytes,
) -> str:
    conn.execute(
        "INSERT INTO spans"
        "(span_id, source_id, revision, start_byte, end_byte,"
        " excerpt_hmac, harvester_version)"
        " VALUES (?, ?, ?, 0, ?, ?, 'test-v1')",
        (span_id, source_id, revision, len(payload),
         store.hmac(payload)),
    )
    return span_id


def seed_claim(
    conn: sqlite3.Connection,
    store: Store,
    scope_id: str,
    claim_id: str,
    subject_id: str,
    predicate: str,
    obj_text: str,
    *,
    state: str = "active",
    revision: int = 1,
    with_span: bool = True,
    recorded_from: int = 1,
) -> str:
    """A claim row + revision + (optional) span evidence — the seed a
    compile run derives entries from."""
    payload = f"{subject_id} {predicate} {obj_text}".encode()
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,?,?,?,1)",
        (claim_id, scope_id, subject_id, predicate, recorded_from),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,object_json,"
        "recorded_from) VALUES(?,?,?,?,?)",
        (claim_id, revision, state,
         json.dumps({"text": obj_text}), recorded_from),
    )
    if with_span:
        sid = f"src:{claim_id}"
        spid = f"span:{claim_id}"
        add_source(conn, store, scope_id, sid, payload)
        add_span(conn, store, spid, sid, revision, payload)
        conn.execute(
            "INSERT INTO claim_evidence(claim_id,revision,span_id,"
            "evidence_role) VALUES(?,?,?,'primary')",
            (claim_id, revision, spid),
        )
    return claim_id


def topic(**kw) -> dict:
    base = {
        "topic_key": "editor",
        "match": {"keywords": ["editor", "vim", "emacs", "vscode"]},
    }
    base.update(kw)
    return base
