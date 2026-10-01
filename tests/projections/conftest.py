"""Fixtures for tests/projections — SPEC_V4 §43/§47.

Same conventions as tests/v4: a pinned profile HMAC key so seeded
digests verify under ``store.hmac``, scopes/grants through the real
governance helpers, and seeded rows carrying real excerpt/payload HMACs
because every read path verifies them.
"""

from __future__ import annotations

import hashlib
import hmac

import pytest

from verbatim.governance import (
    create_grant,
    register_principal,
    seed_purposes,
)
from verbatim.storage.store import Store

_TEST_HMAC_KEY = b"test-hmac-key-padded-to-32-bytes"


def _h(data: bytes) -> bytes:
    return hmac.new(_TEST_HMAC_KEY, data, hashlib.sha256).digest()


@pytest.fixture
def store(tmp_path):
    (tmp_path / "v4.db.key").write_bytes(_TEST_HMAC_KEY)
    s = Store.create(str(tmp_path / "v4.db"))
    yield s
    s.close()


@pytest.fixture
def store_b(tmp_path):
    """A second, independent store — the foreign-import target."""
    (tmp_path / "v4b.db.key").write_bytes(_TEST_HMAC_KEY)
    s = Store.create(str(tmp_path / "v4b.db"))
    yield s
    s.close()


def scope_row(conn, sid, principal="p1", conv="c1", profile="prof",
              vis="conversation"):
    conn.execute(
        "INSERT INTO scopes(scope_id,profile_id,principal_id,workspace_id,"
        "conversation_id,visibility,acl_revision) VALUES(?,?,?,?,?,?,0)",
        (sid, profile, principal, "ws", conv, vis),
    )


def grant(conn, sid, pid="human:alice", purposes=("recall",),
          verbs=("read", "quote")):
    seed_purposes(conn)
    register_principal(conn, kind="human", principal_id=pid)
    return create_grant(
        conn, scope_id=sid, principal_id=pid, verbs=set(verbs),
        issuer_id=pid,
        purposes=None if purposes is None else list(purposes),
    )


def add_source(conn, source_id, scope_id, payload: bytes, speaker="u1"):
    conn.execute(
        "INSERT INTO sources(source_id,origin,source_kind,scope_id,"
        "speaker_id,created_us) VALUES(?,?,?,?,?,1)",
        (source_id, "test", "user_message", scope_id, speaker),
    )
    conn.execute(
        "INSERT INTO source_revisions(source_id,revision,payload,"
        "payload_hmac,event_us,captured_us,timezone,provenance,metadata_json)"
        " VALUES(?,1,?,?,1,1,'UTC','direct_user','{}')",
        (source_id, payload, _h(payload)),
    )


def add_span(conn, span_id, source_id, start, end, rev=1):
    payload = bytes(
        conn.execute(
            "SELECT payload FROM source_revisions"
            " WHERE source_id = ? AND revision = ?",
            (source_id, rev),
        ).fetchone()[0]
    )
    conn.execute(
        "INSERT INTO spans(span_id,source_id,revision,start_byte,end_byte,"
        "excerpt_hmac,harvester_version) VALUES(?,?,?,?,?,?,'t')",
        (span_id, source_id, rev, start, end, _h(payload[start:end])),
    )


def seed_claim(conn, claim_id, scope_id, source_id, span_id, text,
               state="active", recorded_from=1, recorded_until=None,
               rev=1, obj=None, predicate=None, subject_id=None):
    """Claim + head revision + primary evidence (real digests)."""
    payload = text.encode("utf-8")
    add_source(conn, source_id, scope_id, payload)
    add_span(conn, span_id, source_id, 0, len(payload))
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,?,?,?,1)",
        (claim_id, scope_id, subject_id, predicate, recorded_from),
    )
    obj_json = (
        None
        if obj is None
        else __import__("json").dumps(obj, sort_keys=True)
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,object_json,"
        "recorded_from,recorded_until) VALUES(?,?,?,?,?,?)",
        (claim_id, rev, state, obj_json, recorded_from, recorded_until),
    )
    conn.execute(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role) VALUES(?,?,?,'primary')",
        (claim_id, rev, span_id),
    )


def bump_head(conn, claim_id, rev, state="active", recorded_from=2,
              obj=None):
    """Move a claim head forward — the conflict-detection fixture."""
    import json as _json

    conn.execute(
        "UPDATE claim_revisions SET recorded_until = ?"
        " WHERE claim_id = ? AND recorded_until IS NULL",
        (recorded_from, claim_id),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,object_json,"
        "recorded_from,recorded_until) VALUES(?,?,?,?,?,NULL)",
        (
            claim_id,
            rev,
            state,
            None if obj is None else _json.dumps(obj, sort_keys=True),
            recorded_from,
        ),
    )
