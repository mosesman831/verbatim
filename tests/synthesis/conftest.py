"""Synthesis test fixtures — real ``Store.create`` (v1..v4 schema).

Same convention as ``tests/v4``: the profile HMAC key is pinned so
seeded digests verify under ``store.hmac``, and kernel/synthesis timing
goes through explicit ``now_us`` parameters — no wall-clock sleeps.
"""

from __future__ import annotations

import pytest

from verbatim.governance import (
    create_grant,
    register_principal,
    seed_purposes,
)
from verbatim.kernel import Kernel
from verbatim.storage.store import Store
from verbatim.synthesis import Synthesizer

from tests.kernel.conftest import T0, add_source, add_span, caller, grant

_TEST_HMAC_KEY = b"test-hmac-key-padded-to-32-bytes"


@pytest.fixture
def store(tmp_path):
    (tmp_path / "syn.db.key").write_bytes(_TEST_HMAC_KEY)
    s = Store.create(str(tmp_path / "syn.db"))
    yield s
    s.close()


@pytest.fixture
def kernel(store):
    return Kernel(store)


@pytest.fixture
def synth(store, kernel):
    s = Synthesizer(store, kernel=kernel)
    with store.tx() as conn:
        s.register_producer(conn)
    return s


def scope(conn, sid, profile="prof"):
    conn.execute(
        "INSERT INTO scopes (scope_id, profile_id, visibility)"
        " VALUES (?, ?, 'owner')",
        (sid, profile),
    )


def bootstrap(conn, sids=("scope:a",)):
    for sid in sids:
        scope(conn, sid)
    seed_purposes(conn)
    register_principal(conn, kind="human", principal_id="human:alice")
    register_principal(conn, kind="human", principal_id="human:bob")
    register_principal(conn, kind="agent", principal_id="agent:prod")


def add_claim(conn, claim_id, scope_id, span_id, *, rev=1, state="active"):
    """Claim + revision + evidence link (metadata rows only)."""
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,'subj:test','is_about',1,1)",
        (claim_id, scope_id),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,recorded_from)"
        " VALUES(?,?,?,1)",
        (claim_id, rev, state),
    )
    conn.execute(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role) VALUES(?,?,?,'primary')",
        (claim_id, rev, span_id),
    )


__all__ = [
    "T0",
    "add_claim",
    "add_source",
    "add_span",
    "bootstrap",
    "caller",
    "create_grant",
    "grant",
    "register_principal",
    "scope",
    "seed_purposes",
]
