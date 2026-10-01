"""X2 entity timeline + X8 exact-intersection reporting (SPEC_V4_5 §09;
V45-09.02, V45-09.08; D14).

X2: a later mention is never an automatic correction — only an explicit
lifecycle transition ends the current view, and ``known_at_seq`` keeps
the prior revision visible in the historical view (parent §18 lifecycle
guards remain binding).

X8: associative/typed-edge composition is reported AGAINST the exact
entity intersection (:func:`compose_entities`, the V4-04.02 reference) —
a candidate accelerator can never mint facts (D14).
"""

from __future__ import annotations

import hashlib
import hmac

import pytest

from verbatim.core.lifecycle import (
    LifecycleMachine,
    TimeInterval,
    TransitionCommand,
    read_claim_head,
)
from verbatim.core.types_v3 import QueryClass, RecallRequestV3
from verbatim.governance import (
    create_grant,
    register_principal,
    seed_purposes,
)
from verbatim.retrieval.query import analyze
from verbatim.retrieval.v3 import lanes as _lanes
from verbatim.retrieval.v3 import recall_v3
from verbatim.storage.repos import EventsRepo
from verbatim.storage.store import Store

_TEST_HMAC_KEY = b"test-hmac-key-padded-to-32-bytes"


def _h(data: bytes) -> bytes:
    return hmac.new(_TEST_HMAC_KEY, data, hashlib.sha256).digest()


@pytest.fixture
def store(tmp_path):
    (tmp_path / "x2.db.key").write_bytes(_TEST_HMAC_KEY)
    s = Store.create(str(tmp_path / "x2.db"))
    yield s
    s.close()


def _gen(store) -> int:
    return store.projection_generation()


def seed_scope(conn, scope_id, principal="p1", conv="c1"):
    conn.execute(
        "INSERT INTO scopes(scope_id,profile_id,principal_id,workspace_id,"
        "conversation_id,visibility,acl_revision) VALUES(?,?,?,?,?,?,0)",
        (scope_id, "prof", principal, "ws", conv, "conversation"),
    )


def seed_auth(conn, scope_id, pid="human:alice"):
    seed_purposes(conn)
    register_principal(conn, kind="human", principal_id=pid)
    create_grant(
        conn, scope_id=scope_id, principal_id=pid,
        verbs={"read", "quote"}, issuer_id=pid, purposes=["recall"],
    )


def seed_claim(conn, claim_id, scope_id, text, gen, state="active",
               recorded_from=1, recorded_until=None, rev=1):
    src, sp = f"src-{claim_id}", f"sp-{claim_id}"
    payload = text.encode("utf-8")
    conn.execute(
        "INSERT INTO sources(source_id,origin,source_kind,scope_id,"
        "speaker_id,created_us) VALUES(?,?,?,?,?,1)",
        (src, "test", "user_message", scope_id, "u1"),
    )
    conn.execute(
        "INSERT INTO source_revisions(source_id,revision,payload,"
        "payload_hmac,event_us,captured_us,timezone,provenance,metadata_json)"
        " VALUES(?,1,?,?,1,1,'UTC','direct_user','{}')",
        (src, payload, _h(payload)),
    )
    conn.execute(
        "INSERT INTO spans(span_id,source_id,revision,start_byte,end_byte,"
        "excerpt_hmac,harvester_version) VALUES(?,?,?,?,?,?,'t')",
        (sp, src, 1, 0, len(payload), _h(payload)),
    )
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,'ent-x','lives_in',1,1)",
        (claim_id, scope_id),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,condition_json,"
        "recorded_from,recorded_until,perspective_id,freshness)"
        " VALUES(?,?,?,NULL,?,?,NULL,NULL)",
        (claim_id, rev, state, recorded_from, recorded_until),
    )
    conn.execute(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role,family_id) VALUES(?,?,?,'primary',NULL)",
        (claim_id, rev, sp),
    )
    cur = conn.execute(
        "INSERT INTO fts_rows(claim_id,claim_revision,scope_id,"
        "projection_generation) VALUES(?,?,?,?)",
        (claim_id, rev, scope_id, gen),
    )
    conn.execute(
        "INSERT INTO facts_fts(fts_row_id,text) VALUES(?,?)",
        (cur.lastrowid, text),
    )


def add_entity(conn, entity_id, scope_id, label, kind="person"):
    conn.execute(
        "INSERT INTO entities(entity_id,scope_id,kind,label,created_event)"
        " VALUES(?,?,?,?,1)",
        (entity_id, scope_id, kind, label),
    )


def link_entity(conn, claim_id, entity_id, role="subject"):
    conn.execute(
        "INSERT OR IGNORE INTO claim_entities(claim_id,entity_id,role,"
        "span_id) VALUES(?,?,?,NULL)",
        (claim_id, entity_id, role),
    )


def _request(query, **kw):
    kw.setdefault("purpose", "recall")
    return RecallRequestV3(
        query=query, scope_id="sA", caller_id="human:alice", **kw
    )


def _ids(result):
    return [i.handle.object_id for p in result.packs for i in p.items]


class _Cfg:
    graph = True
    dense = False
    sparse = False
    late_interaction = False
    causal = False


def _ctx(conn, store, seeds, scope_ids=("sA",)):
    req = _request("timeline probe")
    plan = analyze("timeline probe", req, 1)
    return _lanes.LaneContext(
        conn=conn, store=store, request=req, plan=plan,
        query_class=QueryClass.CURRENT_STATE,
        scope_ids=tuple(scope_ids), generation=_gen(store),
        deadline=_lanes._cand.Deadline(None), lane_cap=40,
        seeds=dict.fromkeys(seeds, 1), cfg=_Cfg(),
    )


# ---------------------------------------------------------------------------
# X2 — later mention ≠ automatic correction (V45-09.02)
# ---------------------------------------------------------------------------


def test_later_mention_does_not_invalidate(store):
    """Two claims naming the same entity with conflicting values: the
    later-recorded one does NOT supersede the earlier — both stay
    active until an explicit transition says otherwise."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        add_entity(conn, "ent-x", "sA", "Entity X")
        seed_claim(conn, "clOld", "sA", "entity x lives in berlin", gen,
                   recorded_from=5)
        seed_claim(conn, "clNew", "sA", "entity x lives in madrid", gen,
                   recorded_from=50)
        link_entity(conn, "clOld", "ent-x")
        link_entity(conn, "clNew", "ent-x")
    with store.read() as conn:
        # No transition was ever applied: both heads stay active —
        # mention order is not invalidation (V45-09.02).
        assert read_claim_head(conn, "clOld").state.value == "active"
        assert read_claim_head(conn, "clNew").state.value == "active"
        # No supersession edge was minted by mere mention order.
        assert conn.execute(
            "SELECT COUNT(*) FROM edges WHERE edge_type='supersedes'"
        ).fetchone()[0] == 0


def test_explicit_invalidation_known_at_vs_current(store):
    """An explicit supersede ends the current view of the predecessor —
    while known_at still resolves the earlier revision (§18 bitemporal
    guards; V45-09.02)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        add_entity(conn, "ent-x", "sA", "Entity X")
        seed_claim(conn, "clOld", "sA", "entity x lives in berlin", gen,
                   recorded_from=5)
        seed_claim(conn, "clNew", "sA", "entity x lives in madrid", gen,
                   recorded_from=40)
        link_entity(conn, "clOld", "ent-x")
        link_entity(conn, "clNew", "ent-x")
    events = EventsRepo(store)
    with store.tx() as conn:
        while events.latest_seq(conn) < 45:
            events.append(conn, "sA", "seed_pad", "test", {}, "x2-test")
        seq = LifecycleMachine(store).apply(
            TransitionCommand(
                claim_id="clOld", expected_revision=1,
                effect="supersede", successor_claim_id="clNew",
                actor_id="reviewer:r1", reason="explicit correction",
                interval=TimeInterval(from_us=5000, basis="explicit"),
            ),
            conn,
        )
    assert seq > 40
    with store.read() as conn:
        assert read_claim_head(conn, "clOld").state.value == "superseded"
    # Current view: the successor is served, the predecessor is not.
    now_ids = _ids(recall_v3(store, _request("entity x lives")))
    assert "clNew" in now_ids
    assert "clOld" not in now_ids
    # Known-at before the transition: the prior revision still resolves —
    # history is never rewritten by the later state.
    hist_ids = _ids(
        recall_v3(store, _request("entity x lives", known_at_seq=seq - 1))
    )
    assert "clOld" in hist_ids


# ---------------------------------------------------------------------------
# X8 — exact intersection remains the reference (V45-09.08, D14)
# ---------------------------------------------------------------------------


def test_compose_report_exact_intersection_reference(store):
    """D14: associative candidates are reported against the exact
    intersection — the unconfirmed tail is labeled, never minted."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        add_entity(conn, "ent-a", "sA", "Alpha")
        add_entity(conn, "ent-b", "sA", "Beta")
        # cl1 links to BOTH entities — the exact-intersection member.
        seed_claim(conn, "cl1", "sA", "alpha met beta in lisbon", gen)
        link_entity(conn, "cl1", "ent-a")
        link_entity(conn, "cl1", "ent-b")
        # cl2/cl3 link to one entity each — associative-only candidates.
        seed_claim(conn, "cl2", "sA", "alpha alone", gen)
        link_entity(conn, "cl2", "ent-a")
        seed_claim(conn, "cl3", "sA", "beta alone", gen)
        link_entity(conn, "cl3", "ent-b")
    with store.read() as conn:
        ctx = _ctx(conn, store, [])
        report = _lanes.compose_report(ctx, ["ent-a", "ent-b"])
    assert report["reference"] == "exact_intersection"
    assert report["exact"] == ["cl1"]
    # Only the exact-intersection member is verified — cl2/cl3 are
    # candidate associations, not composed facts.
    assert report["verified"] == ["cl1"]
    assert report["associative_only"] == ["cl2", "cl3"]
    assert report["n_exact"] == 1
    assert report["n_associative"] == 3
    assert report["n_associative_only"] == 2
    assert report["binding"] == "structural_typed_edge"


def test_compose_report_empty_when_no_exact(store):
    """No claim links all entities → exact empty, every candidate is
    associative_only — the report never fabricates a composition."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        add_entity(conn, "ent-a", "sA", "Alpha")
        add_entity(conn, "ent-b", "sA", "Beta")
        seed_claim(conn, "cl1", "sA", "alpha only", gen)
        link_entity(conn, "cl1", "ent-a")
    with store.read() as conn:
        ctx = _ctx(conn, store, [])
        report = _lanes.compose_report(ctx, ["ent-a", "ent-b"])
    assert report["exact"] == []
    assert report["verified"] == []
    assert report["associative_only"] == ["cl1"]
    assert report["n_exact"] == 0
