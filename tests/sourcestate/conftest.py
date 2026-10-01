"""source_state test fixtures — real on-disk ``Store`` (SPEC_V5 §13 rule 6:
no kernel mocks), the artifact table installed through the module's own
``install_schema`` migration handler, and real scope/source seeding so
provenance edges and erasure composition exercise genuine schema.
"""

from __future__ import annotations

import pytest

from verbatim.core import time as _time
from verbatim.core.types import json_dumps
from verbatim.governance import (
    create_grant,
    register_principal,
    seed_purposes,
)
from verbatim.sourcestate import install_schema
from verbatim.storage.store import Store

NS = "ns:alice"
SID = "src0001"


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "ss.db"))
    yield s
    s.close()


@pytest.fixture
def installed(store):
    """A real store with the source_state artifact installed."""
    with store.tx() as conn:
        install_schema(conn)
    yield store


def seed_scope(conn, sid: str = NS) -> str:
    conn.execute(
        "INSERT INTO scopes (scope_id, profile_id, visibility)"
        " VALUES (?, 'prof', 'owner')",
        (sid,),
    )
    return sid


def seed_source(
    conn,
    store,
    source_id: str = SID,
    scope_id: str = NS,
    text: str = "the deploy command is deploy-v1",
    revision: int = 1,
) -> str:
    """Real ``sources`` + ``source_revisions`` rows (digest-bound bytes)."""
    seed_scope(conn, scope_id)
    conn.execute(
        "INSERT INTO sources"
        "(source_id, origin, external_id, source_kind, scope_id,"
        " speaker_id, created_us)"
        " VALUES (?, 'test', NULL, 'user_message', ?, NULL, ?)",
        (source_id, scope_id, _time.now_us()),
    )
    add_revision(conn, store, source_id, revision, text)
    return source_id


def add_revision(
    conn, store, source_id: str, revision: int, text: str
) -> None:
    payload = text.encode("utf-8")
    now = _time.now_us()
    conn.execute(
        "INSERT INTO source_revisions"
        "(source_id, revision, payload, payload_hmac, event_us,"
        " captured_us, timezone, provenance, metadata_json)"
        " VALUES (?, ?, ?, ?, ?, ?, NULL, 'direct_user', ?)",
        (
            source_id,
            revision,
            payload,
            store.hmac(payload),
            now,
            now,
            json_dumps({}),
        ),
    )


def bootstrap_principal(conn, sid: str = NS, pid: str = "human:alice") -> str:
    """Scope + purposes + principal + full grant (kernel-access tests)."""
    seed_purposes(conn)
    register_principal(conn, kind="human", principal_id=pid)
    create_grant(
        conn,
        scope_id=sid,
        principal_id=pid,
        verbs=["read", "quote", "derive", "review", "admin"],
        purposes=["recall", "admin", "review"],
        issuer_id=pid,
    )
    return pid
