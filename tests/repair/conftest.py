"""Repair test fixtures — real ``Store.create`` + real producers.

Same convention as ``tests/v45/test_scenarios_d1.py``: file-backed
stores, pinned HMAC, explicit ``now_us``/``recorded_from``. The graph
builder seeds consolidation-eligible claims (subject/predicate +
``object_json`` value + asserted modality), real ``consolidate``
observations, real ``compose`` views, and scenes — so repair plans and
recomputation counts measure production paths, never shims.
"""

from __future__ import annotations

import pytest

from verbatim.core.types import json_dumps
from verbatim.governance import (
    CallerV3,
    create_grant,
    register_principal,
    seed_purposes,
)
from verbatim.kernel import Kernel
from verbatim.observations.consolidate import consolidate_windowed
from verbatim.storage.store import Store
from verbatim.synthesis import Synthesizer

_TEST_HMAC_KEY = b"test-hmac-key-padded-to-32-bytes"
T0 = 1_700_000_000_000_000


def make_store(path: str) -> Store:
    key_path = path + ".key"
    with open(key_path, "wb") as fh:
        fh.write(_TEST_HMAC_KEY)
    return Store.create(path)


@pytest.fixture
def store(tmp_path):
    s = make_store(str(tmp_path / "rep.db"))
    yield s
    s.close()


@pytest.fixture
def stores(tmp_path):
    """Two disposable stores — the honest twin-arm comparator: repair
    runs on one, full rebuild on the other, both over an identically
    seeded graph (V45-04.04's same-input requirement)."""
    a = make_store(str(tmp_path / "rep_a.db"))
    b = make_store(str(tmp_path / "rep_b.db"))
    yield a, b
    a.close()
    b.close()


@pytest.fixture
def kernel(store):
    return Kernel(store)


@pytest.fixture
def synth(store, kernel):
    s = Synthesizer(store, kernel=kernel)
    with store.tx() as conn:
        s.register_producer(conn)
    return s


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
    register_principal(conn, kind="agent", principal_id="agent:prod")
    create_grant(
        conn,
        scope_id=sid,
        principal_id="human:alice",
        verbs=["read", "quote", "derive", "review", "admin"],
        purposes=["recall", "admin", "review", "derive"],
        issuer_id="human:alice",
    )


def add_source(conn, store, sid, source_id, payload):
    conn.execute(
        "INSERT INTO sources(source_id,origin,source_kind,scope_id,"
        "speaker_id,created_us) VALUES(?, 'test', 'user_message', ?,"
        " 'u1', ?)",
        (source_id, sid, T0),
    )
    conn.execute(
        "INSERT INTO source_revisions(source_id,revision,payload,"
        "payload_hmac,event_us,captured_us,timezone,provenance,"
        "metadata_json) VALUES(?, 1, ?, ?, ?, ?, 'UTC', 'direct_user',"
        " '{}')",
        (source_id, payload, store.hmac(payload), T0, T0),
    )


def add_span(conn, store, span_id, source_id, start, end, payload):
    excerpt = payload[start:end]
    conn.execute(
        "INSERT INTO spans(span_id,source_id,revision,start_byte,"
        "end_byte,excerpt_hmac,harvester_version)"
        " VALUES(?, ?, 1, ?, ?, ?, 'test-v1')",
        (span_id, source_id, start, end, store.hmac(excerpt)),
    )


def add_claim(
    conn,
    claim_id,
    sid,
    subject,
    predicate,
    text,
    span_id,
    *,
    rev=1,
    state="active",
    recorded_from=1,
):
    """A consolidation-eligible structured claim (subject/predicate +
    ``object_json`` value + asserted modality) standing on real
    evidence — the unit repair-local recomputation is measured over."""
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,?,?,?,1)",
        (claim_id, sid, subject, predicate, recorded_from),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,object_json,"
        "polarity,modality,recorded_from) VALUES(?,?,?,?,"
        "'affirmative','asserted',?)",
        (
            claim_id,
            rev,
            state,
            json_dumps({"kind": "literal", "text": text}),
            recorded_from,
        ),
    )
    if span_id is not None:
        conn.execute(
            "INSERT INTO claim_evidence(claim_id,revision,span_id,"
            "evidence_role) VALUES(?,?,?,'primary')",
            (claim_id, rev, span_id),
        )


def seed_graph(conn, store, sid="scope:a", n_claims=4):
    """N independent claim slots — each its own consolidation slot, so a
    localized correction touches exactly one slot's observations."""
    bootstrap(conn, sid)
    for i in range(n_claims):
        payload = f"evidence bytes for claim {i}".encode()
        add_source(conn, store, sid, f"src{i}", payload)
        add_span(conn, store, f"sp{i}", f"src{i}", 0, len(payload), payload)
        add_claim(
            conn, f"c{i}", sid, f"subj:{i}", f"pred:{i}",
            f"value {i}", f"sp{i}", recorded_from=1 + i,
        )


def run_consolidate(store, sid="scope:a", min_proof=1):
    """The real full-surface consolidation producer pass (unbounded
    ``since_seq=0`` window touches every slot)."""
    from verbatim.observations.consolidate import ConsolidationWindow

    with store.tx() as conn:
        return consolidate_windowed(
            conn,
            sid,
            window=ConsolidationWindow(since_seq=0),
            min_proof=min_proof,
        )


def compose_claim_view(store, synth, sid, claim_id, rev, *, now_us=T0):
    """A real persisted view bound to one claim input — its
    ``dependency_edges`` hang it off exactly that claim revision."""
    return synth.compose(
        sid,
        caller=caller(),
        purpose="recall",
        view_kind="typed_summary",
        inputs=[{"kind": "claim", "id": claim_id, "revision": rev}],
        now_us=now_us,
    )


def add_episode(conn, sid, episode_id, *, kind="task", recorded_from=1):
    conn.execute(
        "INSERT INTO episodes(episode_id,scope_id,revision,kind,"
        "recorded_from) VALUES(?,?,1,?,?)",
        (episode_id, sid, kind, recorded_from),
    )


def add_scene(conn, sid, family_key, member_ids, *, seq=1):
    """A real scene built through ``assign_episode`` — the production
    producer that creates the scene row, writes ``episode_members``,
    and records ``derivations`` edges scene→member on revision bumps."""
    from verbatim.experience.scenes import assign_episode

    scene_id = None
    for mid in member_ids:
        res = assign_episode(
            conn, mid, family_key=family_key, seq=seq
        )
        assert res["assigned"], res
        scene_id = res["scene_id"]
    return scene_id


def correct_claim_revision(
    conn, claim_id, new_text, *, new_rev=2, recorded_from=50
):
    """A localized correction: close the head revision, append the
    corrected one — the same physical shape the ingester/lifecycle
    machine produces."""
    conn.execute(
        "UPDATE claim_revisions SET recorded_until = ?"
        " WHERE claim_id = ? AND revision = ?",
        (recorded_from, claim_id, new_rev - 1),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,object_json,"
        "polarity,modality,recorded_from) VALUES(?,?,'active',?,"
        "'affirmative','asserted',?)",
        (
            claim_id,
            new_rev,
            json_dumps({"kind": "literal", "text": new_text}),
            recorded_from,
        ),
    )
    conn.execute(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role) SELECT claim_id, ?, span_id, evidence_role"
        " FROM claim_evidence WHERE claim_id = ? AND revision = ?",
        (new_rev, claim_id, new_rev - 1),
    )


__all__ = [
    "T0",
    "add_claim",
    "add_episode",
    "add_scene",
    "add_source",
    "add_span",
    "bootstrap",
    "caller",
    "compose_claim_view",
    "correct_claim_revision",
    "make_store",
    "run_consolidate",
    "seed_graph",
]
