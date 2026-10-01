"""Branch test fixtures — real ``Store.create`` + real claim lifecycles.

Claims are seeded with real source/span/evidence chains so every
``LifecycleMachine`` precondition is genuinely satisfiable — a branch
apply runs the same transition path ``_do_review_apply`` uses.
"""

from __future__ import annotations

import hashlib
import hmac

import pytest

from verbatim.core.lifecycle import LifecycleMachine, TransitionCommand
from verbatim.core.types import json_dumps
from verbatim.governance import (
    CallerV3,
    create_grant,
    register_principal,
    seed_purposes,
)
from verbatim.kernel import Kernel
from verbatim.storage.store import Store
from verbatim.synthesis import Synthesizer

_TEST_HMAC_KEY = b"test-hmac-key-padded-to-32-bytes"
T0 = 1_700_000_000_000_000


def _h(data: bytes) -> bytes:
    return hmac.new(_TEST_HMAC_KEY, data, hashlib.sha256).digest()


@pytest.fixture
def store(tmp_path):
    (tmp_path / "br.db.key").write_bytes(_TEST_HMAC_KEY)
    s = Store.create(str(tmp_path / "br.db"))
    yield s
    s.close()


@pytest.fixture
def kernel(store):
    return Kernel(store)


@pytest.fixture
def synth(store, kernel):
    return Synthesizer(store, kernel=kernel)


def caller(pid: str = "human:alice") -> CallerV3:
    return CallerV3(principal_id=pid)


def bootstrap(conn, sid="scope:a"):
    conn.execute(
        "INSERT INTO scopes (scope_id, profile_id, visibility)"
        " VALUES (?, 'prof', 'owner')",
        (sid,),
    )
    seed_purposes(conn)
    register_principal(conn, kind="human", principal_id="human:alice")
    create_grant(
        conn,
        scope_id=sid,
        principal_id="human:alice",
        verbs=["read", "quote", "derive", "review", "admin"],
        purposes=["recall", "admin", "review"],
        issuer_id="human:alice",
    )
    return "human:alice"


def add_claim(conn, claim_id, sid, text, *, rev=1, state="active",
              recorded_from=1):
    """A claim standing on real source/span evidence (the foreign-key
    chain ``claim_evidence → spans → source_revisions`` is real)."""
    payload = text.encode("utf-8")
    sid_src = f"src:{claim_id}"
    spid = f"span:{claim_id}"
    conn.execute(
        "INSERT INTO sources(source_id,origin,source_kind,scope_id,"
        "speaker_id,created_us) VALUES(?, 'test', 'user_message', ?,"
        " 'u1', ?)",
        (sid_src, sid, T0),
    )
    conn.execute(
        "INSERT INTO source_revisions(source_id,revision,payload,"
        "payload_hmac,event_us,captured_us,timezone,provenance,"
        "metadata_json) VALUES(?,1,?,?,?,?,'UTC','direct_user','{}')",
        (sid_src, payload, _h(payload), T0, T0),
    )
    conn.execute(
        "INSERT INTO spans(span_id,source_id,revision,start_byte,"
        "end_byte,excerpt_hmac,harvester_version) VALUES(?,?,1,0,?,?,"
        "'test-v1')",
        (spid, sid_src, len(payload), _h(payload)),
    )
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,?,?,?,1)",
        (claim_id, sid, f"subj:{claim_id}", "pred", recorded_from),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,object_json,"
        "polarity,modality,recorded_from) VALUES(?,?,?,?,"
        "'affirmative','asserted',?)",
        (claim_id, rev, state,
         json_dumps({"kind": "literal", "text": text}), recorded_from),
    )
    conn.execute(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role) VALUES(?,?,?,'primary')",
        (claim_id, rev, spid),
    )


def transition(store, claim_id, effect, *, expected=1, successor=None,
               reason="test transition"):
    """Drive a real lifecycle transition outside a review (the "the
    world moved under the branch" probe)."""
    machine = LifecycleMachine(store)
    with store.tx() as conn:
        return machine.apply(
            TransitionCommand(
                claim_id=claim_id,
                expected_revision=expected,
                effect=effect,
                actor_id="human:alice",
                reason=reason,
                successor_claim_id=successor,
            ),
            conn,
        )


__all__ = [
    "T0",
    "add_claim",
    "bootstrap",
    "caller",
    "transition",
]
