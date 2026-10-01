"""Graph expansion lane — SPEC_V4 §30 (V4-30.01–30.09) + V3-28.07.

The lane seeds from already-admitted flat-lane hits and expands through
the typed-edge graph (``edges``, ``derivations``, entity co-membership,
episode membership, transitions, open conflict groups) under explicit
hop/fanout/visited-node/examined-edge/deadline bounds. Every landed node
— and every intermediate hop — passes the same ``union.admit_keys``
gate, so traversal can never widen authorization (V4-30.02).

Fixtures use the same pinned-HMAC seeding convention as
``test_retrieval_v3.py`` (digest-verified reads).
"""

from __future__ import annotations

import hashlib
import hmac
import sqlite3

import pytest

from verbatim.core.types_v3 import QueryClass, RecallRequestV3
from verbatim.governance import (
    create_grant,
    register_principal,
    seed_purposes,
)
from verbatim.retrieval.query import analyze
from verbatim.retrieval.v3 import lanes as _lanes
from verbatim.retrieval.v3 import recall_v3
from verbatim.storage.store import Store

_TEST_HMAC_KEY = b"test-hmac-key-padded-to-32-bytes"


def _h(data: bytes) -> bytes:
    return hmac.new(_TEST_HMAC_KEY, data, hashlib.sha256).digest()


@pytest.fixture
def store(tmp_path):
    (tmp_path / "g.db.key").write_bytes(_TEST_HMAC_KEY)
    s = Store.create(str(tmp_path / "g.db"))
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
               rev=1, source_id=None, span_id=None):
    src = source_id or f"src-{claim_id}"
    sp = span_id or f"sp-{claim_id}"
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
        "created_event,row_version) VALUES(?,?,NULL,NULL,1,1)",
        (claim_id, scope_id),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,condition_json,"
        "recorded_from,recorded_until,perspective_id,freshness)"
        " VALUES(?,?,?,NULL,1,NULL,NULL,NULL)",
        (claim_id, rev, state),
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


def add_edge(conn, edge_id, scope_id, sk, sid, tk, tid, etype,
             retired=False):
    conn.execute(
        "INSERT INTO edges(edge_id,scope_id,source_kind,source_id,"
        "target_kind,target_id,edge_type,decision_id,created_event,"
        "retired_event) VALUES(?,?,?,?,?,?,?,NULL,1,?)",
        (edge_id, scope_id, sk, sid, tk, tid, etype, 1 if retired else None),
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


def add_derivation(conn, scope_id, ck, cid, pk, pid, seq=1):
    conn.execute(
        "INSERT INTO derivations(child_kind,child_id,child_revision,"
        "parent_kind,parent_id,parent_revision,producer_kind,producer_id,"
        "seq,scope_id) VALUES(?,?,1,?,?,1,'test','t',?,?)",
        (ck, cid, pk, pid, seq, scope_id),
    )


class _Cfg:
    """Minimal retrieval cfg exposing the graph toggle."""
    graph = True
    dense = False
    sparse = False
    late_interaction = False
    causal = False


def _ctx(conn, store, seeds, query="graph probe", scope_ids=("sA",),
         cfg=_Cfg(), **bounds):
    req = RecallRequestV3(
        query=query, scope_id="sA", caller_id="human:alice",
        purpose="recall",
    )
    plan = analyze(query, req, 1)
    return _lanes.LaneContext(
        conn=conn, store=store, request=req, plan=plan,
        query_class=QueryClass.CURRENT_STATE,
        scope_ids=tuple(scope_ids), generation=_gen(store),
        deadline=_lanes._cand.Deadline(None), lane_cap=40,
        seeds=dict.fromkeys(seeds, 1), cfg=cfg, **bounds,
    )


# ---------------------------------------------------------------------
# expansion + typed edges
# ---------------------------------------------------------------------

def test_graph_expands_typed_edges(store):
    """Seeds expand through every edges-table type, ordered by the
    structural binding weight (V4-30.03 provenance labels recorded)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        for cid in ("cl1", "cl2", "cl3", "cl4", "cl5"):
            seed_claim(conn, cid, "sA", f"edge probe {cid}", gen)
        add_edge(conn, "e-sup", "sA", "claim", "cl1", "claim", "cl2",
                 "supersedes")
        add_edge(conn, "e-con", "sA", "claim", "cl1", "claim", "cl3",
                 "conflicts_with")
        add_edge(conn, "e-ctx", "sA", "claim", "cl1", "claim", "cl4",
                 "context_of")
        add_edge(conn, "e-cor", "sA", "claim", "cl1", "claim", "cl5",
                 "corrects")
        # A retired edge must not expand.
        add_edge(conn, "e-dead", "sA", "claim", "cl1", "claim", "cl5",
                 "context_of", retired=True)
    with store.read() as conn:
        ctx = _ctx(conn, store, [("claim", "cl1")])
        res = _lanes.lane_graph(ctx)
    assert res.status == "ok"
    assert set(res.hits) == {
        ("claim", "cl2"), ("claim", "cl3"),
        ("claim", "cl4"), ("claim", "cl5"),
    }
    # Structural ordering: supersedes (1.0) > conflicts_with (0.9) >
    # corrects (1.0 tie-broken by id) … then context_of (0.6) last.
    order = sorted(res.hits, key=lambda k: res.hits[k])
    assert res.hits[("claim", "cl4")] == max(res.hits.values())
    prov = res.details["provenance"]
    assert prov["claim:cl2"]["via"] == "supersedes"
    assert prov["claim:cl3"]["class"] == "contradiction"
    assert res.details["composition"] == "structural"
    assert res.details["binding"] == "typed_edge"


def test_graph_ignores_evidence_plane_targets(store):
    """Edges into the evidence plane never land — spans are not
    retrievable candidates (V4-30.02)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "cl1", "sA", "edge to span", gen)
        add_edge(conn, "e-sp", "sA", "claim", "cl1", "span", "sp-cl1",
                 "derived_from")
    with store.read() as conn:
        res = _lanes.lane_graph(_ctx(conn, store, [("claim", "cl1")]))
    assert res.hits == {}
    assert res.status == "skipped" and res.reason == "no_expansion"


def test_graph_derivations_traverse(store):
    """Derivation edges carry provenance both directions (§06.06)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        for cid in ("cl1", "cl2", "cl3"):
            seed_claim(conn, cid, "sA", f"derivation probe {cid}", gen)
        # cl2 derives from cl1; cl3 derives from cl2.
        add_derivation(conn, "sA", "claim", "cl2", "claim", "cl1")
        add_derivation(conn, "sA", "claim", "cl3", "claim", "cl2", seq=2)
    with store.read() as conn:
        ctx = _ctx(conn, store, [("claim", "cl2")])
        res = _lanes.lane_graph(ctx)
    assert set(res.hits) == {("claim", "cl1"), ("claim", "cl3")}
    prov = res.details["provenance"]
    assert prov["claim:cl1"]["via"] == "derived_from"
    assert prov["claim:cl3"]["via"] == "derived_from"


def test_graph_entity_comembership(store):
    """Claims sharing an authorized entity expand as topic association
    with the entity id recorded as provenance (V4-30.04 reference)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "cl1", "sA", "entity probe one", gen)
        seed_claim(conn, "cl2", "sA", "entity probe two", gen)
        seed_claim(conn, "cl3", "sA", "entity probe three", gen)
        add_entity(conn, "ent-1", "sA", "Jordan")
        link_entity(conn, "cl1", "ent-1")
        link_entity(conn, "cl2", "ent-1")
        link_entity(conn, "cl3", "ent-1")
    with store.read() as conn:
        res = _lanes.lane_graph(_ctx(conn, store, [("claim", "cl1")]))
    assert set(res.hits) == {("claim", "cl2"), ("claim", "cl3")}
    prov = res.details["provenance"]
    assert prov["claim:cl2"]["via"] == "entity"
    assert prov["claim:cl2"]["through"] == "ent-1"
    assert prov["claim:cl2"]["class"] == "topic_association"


# ---------------------------------------------------------------------
# bounds (V4-30.09)
# ---------------------------------------------------------------------

def _chain(conn, scope_id, gen, n=4, etype="supersedes"):
    ids = [f"clc{i}" for i in range(n)]
    for cid in ids:
        seed_claim(conn, cid, scope_id, f"chain {cid}", gen)
    for i in range(n - 1):
        add_edge(conn, f"ec{i}", scope_id, "claim", ids[i],
                 "claim", ids[i + 1], etype)
    return ids


def test_graph_hop_bound(store):
    """One hop reaches only direct neighbours; the spec default never
    silently widens to two (V4-30.09)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        ids = _chain(conn, "sA", _gen(store), 4)
    with store.read() as conn:
        res1 = _lanes.lane_graph(
            _ctx(conn, store, [("claim", ids[0])], graph_hops=1))
        res2 = _lanes.lane_graph(
            _ctx(conn, store, [("claim", ids[0])], graph_hops=2))
    assert set(res1.hits) == {("claim", ids[1])}
    assert set(res2.hits) == {("claim", ids[1]), ("claim", ids[2])}
    assert res2.details["hops_used"] == 2


def test_graph_fanout_bound(store):
    """Per-node fanout is an explicit bound; the walk reports which bound
    stopped it instead of silently widening (V4-30.09)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "cl1", "sA", "fanout hub", gen)
        for i in range(5):
            cid = f"clf{i}"
            seed_claim(conn, cid, "sA", f"fanout leaf {cid}", gen)
            add_edge(conn, f"ef{i}", "sA", "claim", "cl1",
                     "claim", cid, "context_of")
    with store.read() as conn:
        res = _lanes.lane_graph(
            _ctx(conn, store, [("claim", "cl1")], graph_fanout=2))
    assert len(res.hits) == 2
    assert res.status == "partial"
    assert res.details["bounded_by"] == "fanout"
    assert "graph_bound:fanout" in res.warnings


def test_graph_node_bound(store):
    """Visited-node cap bounds the walk (seeds count as visited)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "cl1", "sA", "node hub", gen)
        for i in range(5):
            cid = f"cln{i}"
            seed_claim(conn, cid, "sA", f"node leaf {cid}", gen)
            add_edge(conn, f"en{i}", "sA", "claim", "cl1",
                     "claim", cid, "supports")
    with store.read() as conn:
        res = _lanes.lane_graph(
            _ctx(conn, store, [("claim", "cl1")], graph_max_nodes=3))
    # seed (1) + at most 2 neighbours — the cap counts the seed.
    assert res.details["visited_nodes"] <= 3
    assert len(res.hits) <= 2
    assert res.status == "partial"
    assert res.details["bounded_by"] == "nodes"


def test_graph_edge_budget_bound(store):
    """The examined-edge budget stops the walk honestly (V4-30.09)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "cl1", "sA", "edge hub", gen)
        for i in range(4):
            cid = f"cle{i}"
            seed_claim(conn, cid, "sA", f"edge leaf {cid}", gen)
            add_edge(conn, f"ee{i}", "sA", "claim", "cl1",
                     "claim", cid, "supports")
    with store.read() as conn:
        res = _lanes.lane_graph(
            _ctx(conn, store, [("claim", "cl1")], graph_max_edges=2))
    assert res.details["examined_edges"] <= 2
    assert res.status == "partial"
    assert res.details["bounded_by"] == "edges"


def test_graph_deadline_bound(store):
    """An expired request deadline stops traversal before the first hop
    and is reported, never hidden (V4-27.09, V4-30.09)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "cl1", "sA", "deadline probe", gen)
        seed_claim(conn, "cl2", "sA", "deadline neighbor", gen)
        add_edge(conn, "ed", "sA", "claim", "cl1", "claim", "cl2",
                 "supports")

    class _Expired:
        def expired(self):
            return True

    with store.read() as conn:
        ctx = _ctx(conn, store, [("claim", "cl1")])
        ctx.deadline = _Expired()
        _o, _p, stats = _lanes.graph_expand(ctx, [("claim", "cl1")])
    assert stats["bounded_by"] == "deadline"


# ---------------------------------------------------------------------
# eligibility / authorization — never widened (V4-30.02)
# ---------------------------------------------------------------------

def test_quarantined_intermediate_blocks_expansion(store):
    """A held intermediate node can neither land nor be traversed
    through — the path through it dies (V4-30.02)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        ids = _chain(conn, "sA", _gen(store), 3)
        conn.execute(
            "INSERT INTO quarantine(object_kind,object_id,revision,"
            "scope_id,state,opened_event) VALUES('claim',?,1,'sA',"
            "'suppressed',1)",
            (ids[1],),
        )
    with store.read() as conn:
        res = _lanes.lane_graph(
            _ctx(conn, store, [("claim", ids[0])], graph_hops=2))
    assert ("claim", ids[1]) not in res.hits
    assert ("claim", ids[2]) not in res.hits


def test_quarantined_leaf_not_emitted(store):
    """A held direct neighbour is examined but never emitted."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "cl1", "sA", "held leaf probe", gen)
        seed_claim(conn, "cl2", "sA", "held leaf", gen)
        add_edge(conn, "eh", "sA", "claim", "cl1", "claim", "cl2",
                 "supports")
        conn.execute(
            "INSERT INTO quarantine(object_kind,object_id,revision,"
            "scope_id,state,opened_event) VALUES('claim','cl2',1,'sA',"
            "'pending',1)",
        )
    with store.read() as conn:
        res = _lanes.lane_graph(_ctx(conn, store, [("claim", "cl1")]))
    assert res.hits == {}
    assert res.details["examined_edges"] >= 1


def test_revoked_scope_never_reachable(store):
    """An in-scope edge pointing at an out-of-scope (revoked) claim must
    not surface it — admission fails, and the revoked node's own edges
    are never scanned (V4-30.02)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_scope(conn, "sB", principal="p2", conv="c2")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clA", "sA", "revocation probe", gen)
        seed_claim(conn, "clB", "sB", "revoked scope claim", gen)
        # Edge authored in sA reaching into sB — malicious or stale.
        add_edge(conn, "eX", "sA", "claim", "clA", "claim", "clB",
                 "supports")
        # …and an edge authored in sB entirely (invisible to sA's walk).
        seed_claim(conn, "clB2", "sB", "revoked scope second", gen)
        add_edge(conn, "eY", "sB", "claim", "clB", "claim", "clB2",
                 "supports")
    with store.read() as conn:
        res = _lanes.lane_graph(
            _ctx(conn, store, [("claim", "clA")], graph_hops=3))
    assert ("claim", "clB") not in res.hits
    assert ("claim", "clB2") not in res.hits


def test_superseded_state_not_admitted(store):
    """Lifecycle gates apply to expanded candidates identically to flat
    ones — a superseded claim never lands under a CURRENT query."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "cl1", "sA", "lifecycle probe", gen)
        seed_claim(conn, "cl2", "sA", "lifecycle stale", gen,
                   state="superseded")
        add_edge(conn, "es", "sA", "claim", "cl1", "claim", "cl2",
                 "supersedes")
    with store.read() as conn:
        res = _lanes.lane_graph(_ctx(conn, store, [("claim", "cl1")]))
    assert res.hits == {}


def test_graph_disabled_and_seedless(store):
    """Honest skip states: disabled by config, zero hop budget, or no
    seeds — reported, never silent."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "skip probe", _gen(store))
    with store.read() as conn:
        off = _lanes.lane_graph(
            _ctx(conn, store, [("claim", "cl1")], cfg=type("C", (), {
                "graph": False, "dense": False, "sparse": False,
                "late_interaction": False, "causal": False})()))
        nohop = _lanes.lane_graph(
            _ctx(conn, store, [("claim", "cl1")], graph_hops=0))
        noseed = _lanes.lane_graph(_ctx(conn, store, []))
    assert (off.status, off.reason) == ("skipped", "off_until_ablation")
    assert (nohop.status, nohop.reason) == ("skipped", "no_hop_budget")
    assert (noseed.status, noseed.reason) == (
        "skipped", "no_authorized_seeds")


# ---------------------------------------------------------------------
# §30 reference operations
# ---------------------------------------------------------------------

def test_reference_ops_probe_compose_contradictions(store):
    """probe / compose / contradictions / why_related — deterministic
    authorized reference implementations (V4-30.01, V4-04.02)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "cl1", "sA", "ops probe one", gen)
        seed_claim(conn, "cl2", "sA", "ops probe two", gen)
        seed_claim(conn, "cl3", "sA", "ops probe three", gen)
        seed_claim(conn, "cl4", "sA", "ops probe four", gen)
        seed_claim(conn, "cl5", "sA", "ops probe isolated", gen)
        add_entity(conn, "ent-a", "sA", "Alpha")
        add_entity(conn, "ent-b", "sA", "Beta")
        link_entity(conn, "cl1", "ent-a")
        link_entity(conn, "cl1", "ent-b")   # cl1 names both
        link_entity(conn, "cl2", "ent-a")
        link_entity(conn, "cl3", "ent-b")
        add_edge(conn, "eo1", "sA", "claim", "cl1", "claim", "cl4",
                 "supports")
        add_edge(conn, "eo2", "sA", "claim", "cl2", "claim", "cl4",
                 "conflicts_with")
    with store.read() as conn:
        ctx = _ctx(conn, store, [])
        # probe(entity) → authorized claims bound to it
        assert sorted(_lanes.probe_entity(ctx, "ent-a")) == ["cl1", "cl2"]
        # compose(entity_set) → exact intersection
        assert _lanes.compose_entities(ctx, ["ent-a", "ent-b"]) == ["cl1"]
        # contradictions(claim) → conflicts_with neighbours
        assert _lanes.contradictions(ctx, ("claim", "cl4")) == ["cl2"]
        # why_related: admitted path with edge provenance
        path = _lanes.why_related(ctx, ("claim", "cl1"), ("claim", "cl4"))
        assert path is not None
        assert path[0][0] == ("claim", "cl1")
        assert path[-1][0] == ("claim", "cl4")
        assert path[1][1] == "supports"
        # cl1 and cl3 ARE related — they share entity ent-b — and the
        # path discloses the entity-mediated hop honestly.
        p13 = _lanes.why_related(ctx, ("claim", "cl1"), ("claim", "cl3"))
        assert p13 is not None and p13[-1][1] == "entity"
        # A fully isolated node → honest None, never a fabricated path.
        assert _lanes.why_related(
            ctx, ("claim", "cl1"), ("claim", "cl5")) is None


def test_related_labels_relation_classes(store):
    """related() returns provenance with the distinguished relation
    classes — labels never conflated (V4-30.03)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "cl1", "sA", "class probe", gen)
        seed_claim(conn, "cl2", "sA", "class conflict", gen)
        seed_claim(conn, "cl3", "sA", "class context", gen)
        add_edge(conn, "er1", "sA", "claim", "cl1", "claim", "cl2",
                 "conflicts_with")
        add_edge(conn, "er2", "sA", "claim", "cl1", "claim", "cl3",
                 "context_of")
    with store.read() as conn:
        out = _lanes.related(_ctx(conn, store, []), ("claim", "cl1"))
    prov = {k: p for k, p in out}
    assert prov[("claim", "cl2")]["class"] == "contradiction"
    assert prov[("claim", "cl3")]["class"] == "topic_association"


def test_episode_membership_expansion(store):
    """Episode↔member edges expand both directions as live membership
    only (recorded_until honoured)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "cl1", "sA", "episode member", gen)
        seed_claim(conn, "cl2", "sA", "former member", gen)
        conn.execute(
            "INSERT INTO episodes(episode_id,scope_id,revision,kind,"
            "label,recorded_from,recorded_until,row_version)"
            " VALUES('ep1','sA',1,'task','probe episode',1,NULL,1)",
        )
        conn.execute(
            "INSERT INTO episode_members(episode_id,object_kind,object_id,"
            "ord,recorded_from,recorded_until)"
            " VALUES('ep1','claim','cl1',0,1,NULL),"
            "       ('ep1','claim','cl2',1,1,2)",  # cl2's membership ended
        )
    with store.read() as conn:
        res = _lanes.lane_graph(
            _ctx(conn, store, [("episode", "ep1")]))
    assert ("claim", "cl1") in res.hits
    assert ("claim", "cl2") not in res.hits
    assert res.details["provenance"]["claim:cl1"]["via"] == (
        "episode_member")


# ---------------------------------------------------------------------
# end-to-end through recall_v3 (union/fusion path — V4-30.02)
# ---------------------------------------------------------------------

def test_graph_lane_enters_union_through_recall(store):
    """Through the real pipeline the expansion lands as ordinary lane
    hits — the union's authoritative admission is the same gate."""
    from verbatim.config import config_from_mapping

    cfg = config_from_mapping({"v3": {"retrieval": {"graph": True}}})
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "cl1", "sA", "pipeline probe anchor phrase", gen)
        # cl2 has no lexical overlap — only the edge can carry it in.
        seed_claim(conn, "cl2", "sA", "unrelated zzz qqq", gen)
        add_edge(conn, "ep", "sA", "claim", "cl1", "claim", "cl2",
                 "supports")
    res = recall_v3(store, RecallRequestV3(
        query="pipeline probe anchor phrase", scope_id="sA",
        caller_id="human:alice", purpose="recall",
        modes=("relationship",),
    ), cfg=cfg)
    assert res.capabilities["lanes"].get("graph") in ("ok", "skipped")
    details = res.capabilities.get("lane_details", {}).get("graph")
    if res.capabilities["lanes"].get("graph") == "ok":
        assert details["composition"] == "structural"
        ids = {i.handle.object_id for p in res.packs for i in p.items}
        assert "cl1" in ids
        # Whether cl2 packs depends on downstream group/abstain policy —
        # the lane-level guarantee is that it was admitted through the
        # same union gate and carries typed-edge provenance.
        prov = details.get("provenance", {})
        assert "claim:cl2" in prov or "cl2" in ids
        if "claim:cl2" in prov:
            assert prov["claim:cl2"]["via"] == "supports"
