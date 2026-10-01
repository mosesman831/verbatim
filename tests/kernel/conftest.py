"""Kernel test fixtures — real ``Store.create`` (v1+v2+v3+v4 schema).

The shared ``tests/conftest.py`` TestStore shim carries DDL_V1 only — the
v4 kernel tables (``objects``, ``delivery_permits``, ``dependency_edges``)
and the v3 governance tables do not exist there, so these tests run
against a real file-backed store in ``tmp_path``.
"""

from __future__ import annotations

import sqlite3

import pytest

from verbatim.core.types_v3 import Verb
from verbatim.governance import (
    CallerV3,
    create_grant,
    register_principal,
    seed_purposes,
)
from verbatim.kernel import Kernel
from verbatim.storage.store import Store

US = 1_000_000
T0 = 1_700_000_000_000_000


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "kernel.db"))
    yield s
    s.close()


@pytest.fixture
def kernel(store):
    return Kernel(store)


@pytest.fixture
def seeded(store):
    """Two scopes + purpose registry + two principals. Returns ids."""
    with store.tx() as conn:
        for sid in ("scope:a", "scope:b"):
            conn.execute(
                "INSERT INTO scopes (scope_id, profile_id, visibility)"
                " VALUES (?, 'prof', 'owner')",
                (sid,),
            )
        seed_purposes(conn)
        register_principal(conn, kind="human", principal_id="human:alice")
        register_principal(conn, kind="agent", principal_id="agent:prod")
        register_principal(conn, kind="human", principal_id="human:bob")
    return {"a": "scope:a", "b": "scope:b"}


def add_source(
    conn: sqlite3.Connection,
    store: Store,
    scope_id: str,
    source_id: str,
    payload: bytes,
    *,
    revision: int = 1,
    provenance: str = "direct_user",
) -> str:
    """Insert a source + revision row with the real keyed HMAC."""
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
        " VALUES (?, ?, ?, ?, ?, ?, NULL, ?, '{}')",
        (source_id, revision, payload, store.hmac(payload), T0, T0, provenance),
    )
    return source_id


def add_span(
    conn: sqlite3.Connection,
    store: Store,
    span_id: str,
    source_id: str,
    revision: int,
    start: int,
    end: int,
    payload: bytes,
) -> str:
    """Insert a span with its excerpt HMAC over the real stored bytes."""
    excerpt = payload[start:end]
    conn.execute(
        "INSERT INTO spans"
        "(span_id, source_id, revision, start_byte, end_byte,"
        " excerpt_hmac, harvester_version)"
        " VALUES (?, ?, ?, ?, ?, ?, 'test-v1')",
        (span_id, source_id, revision, start, end, store.hmac(excerpt)),
    )
    return span_id


def add_view(
    conn: sqlite3.Connection,
    store: Store,
    source_id: str,
    revision: int,
    view_id: str,
    derived: bytes,
    *,
    view_kind: str = "normalized",
    legacy_digest: bool = False,
) -> str:
    """Insert a derived view; ``legacy_digest=True`` leaves integrity NULL."""
    digest = None if legacy_digest else store.hmac(derived)
    conn.execute(
        "INSERT INTO source_views"
        "(source_id, revision, view_id, media_type, view_kind,"
        " transformer_revision, locator_json, derived_bytes, integrity_digest)"
        " VALUES (?, ?, ?, 'text/plain', ?, 'xf-v1', '{}', ?, ?)",
        (source_id, revision, view_id, view_kind, derived, digest),
    )
    return view_id


def grant(
    conn: sqlite3.Connection,
    scope_id: str,
    principal_id: str,
    verbs,
    purposes=("recall",),
) -> str:
    return create_grant(
        conn,
        scope_id=scope_id,
        principal_id=principal_id,
        verbs=list(verbs),
        purposes=list(purposes),
        issuer_id="human:alice",
    )


def caller(pid: str = "human:alice", epoch=None) -> CallerV3:
    return CallerV3(principal_id=pid, epoch=epoch)
