"""F4-01 / F4-02 authorization regressions (SPEC_V4 §05, §08–§11).

C07 — purpose-aware scope expansion (V4-10.05, V4-11.04): a caller
holding read/recall on scope A and read+quote/evaluate on scope B can
never pull B's content into a recall-purpose query on A. The pre-fix
``_authorized_scopes`` consulted the purpose-blind ``effective_verbs``
union, which admitted B outright — every contributing scope now passes
``governance.authorize`` for the request's own purpose, and exclusion
is indistinguishable from absence (no error, count, or label names it).

C08 — quote enforcement at delivery (V4-08.02): a read-only caller
receives metadata / derived-view items only; exact source bytes never
appear in pack items, in serialized item bodies, or on the
``AssemblyResult.groups`` inspection surface. The check lives at the
render point (``pack._render_item``), so it cannot be skipped by an
upstream flag — or by inspecting the assembly result instead of packs.
"""

from __future__ import annotations

import itertools
import json

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v3 import (
    ContextRequestV3,
    PackKind,
    QueryClass,
)
from verbatim.governance import (
    create_grant,
    revoke_grant,
)
from verbatim.retrieval.query import analyze
from verbatim.retrieval.v3 import context_v3, recall_v3
from verbatim.retrieval.v3 import fusion_v3 as _fuse
from verbatim.retrieval.v3 import influence as _infl
from verbatim.retrieval.v3 import lanes as _lanes
from verbatim.retrieval.v3 import pack as _pack
from verbatim.retrieval.v3 import union as _union
from verbatim.storage.store import Store

from tests.retrieval.v3.test_retrieval_v3 import (
    _TEST_HMAC_KEY,
    _gen,
    add_source,
    add_span,
    add_fts,
    items_of,
    request,
    seed_auth,
    seed_claim,
    seed_scope,
    texts_of,
)


@pytest.fixture
def store(tmp_path):
    (tmp_path / "v3.db.key").write_bytes(_TEST_HMAC_KEY)
    s = Store.create(str(tmp_path / "v3.db"))
    yield s
    s.close()


def _bodies(result):
    """Parsed JSON body of every serialized pack item."""
    head, tail = _pack.UNTRUSTED_HEADER, _pack.UNTRUSTED_FOOTER
    out = []
    for p in result.packs:
        for i in p.items:
            assert i.text.startswith(head) and i.text.endswith(tail)
            out.append(json.loads(i.text[len(head):-len(tail)]))
    return out


def _ids(result):
    return {i.handle.object_id for i in items_of(result)}


# ---------------------------------------------------------------------
# C07 — F4-01: purpose-aware scope expansion
# ---------------------------------------------------------------------

def test_c07_evaluate_scope_cannot_leak_into_recall(store):
    """read+quote/recall on A plus read+quote/evaluate on B: a
    recall-purpose query on A must not disclose B — under the old
    purpose-blind verb union B contributed outright (V4-10.05, V4-11.04,
    C07)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_scope(conn, "sB", principal="p2", conv="c2")
        seed_auth(conn, "sA")                              # read+quote@recall
        seed_auth(conn, "sB", purposes=("evaluate",))      # read+quote@evaluate
        gen = _gen(store)
        seed_claim(conn, "clA", "sA", "srcA", "spA",
                   "shared probe phrase alpha", gen)
        seed_claim(conn, "clB", "sB", "srcB", "spB",
                   "shared probe phrase beta secret", gen)
    res = recall_v3(store, request("shared probe phrase"))
    ids = _ids(res)
    assert "clA" in ids
    assert "clB" not in ids                       # B never contributes
    assert not any("beta secret" in t for t in texts_of(res))
    # indistinguishable denial — no error, warning, or count names sB
    assert not any("sB" in w for w in res.warnings)


def test_c07_matching_purpose_admits_the_scope(store):
    """Same grants, but the request declares ``evaluate`` — which A also
    holds: B then contributes normally. The boundary is the request's
    purpose pinned per scope, not a blanket cross-scope ban."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_scope(conn, "sB", principal="p2", conv="c2")
        seed_auth(conn, "sA", purposes=("recall", "evaluate"))
        seed_auth(conn, "sB", purposes=("evaluate",))
        gen = _gen(store)
        seed_claim(conn, "clA", "sA", "srcA", "spA",
                   "shared probe phrase alpha", gen)
        seed_claim(conn, "clB", "sB", "srcB", "spB",
                   "shared probe phrase beta", gen)
    res = recall_v3(store, request("shared probe phrase",
                                   purpose="evaluate"))
    assert "clB" in _ids(res)
    assert any("beta" in t for t in texts_of(res))


def test_c07_request_scope_denied_for_wrong_purpose(store):
    """The requesting scope itself must authorize the request purpose:
    a recall-purpose query under an evaluate-only grant on sA denies
    indistinguishably (§10.05)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA", purposes=("evaluate",))
    with pytest.raises(VerbatimError) as exc:
        recall_v3(store, request("anything"))
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_c07_revocation_drops_scope_contribution(store):
    """Epoch-advancing revocation removes B from subsequent multi-scope
    recalls — silently, mid-stream (V4-11.06, epoch-vector pinning)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_scope(conn, "sB", principal="p2", conv="c2")
        seed_auth(conn, "sA")
        gid_b = create_grant(
            conn, scope_id="sB", principal_id="human:alice",
            verbs={"read", "quote"}, issuer_id="human:alice",
            purposes=["recall"],
        )
        gen = _gen(store)
        seed_claim(conn, "clA", "sA", "srcA", "spA",
                   "shared probe phrase alpha", gen)
        seed_claim(conn, "clB", "sB", "srcB", "spB",
                   "shared probe phrase beta", gen)
    res1 = recall_v3(store, request("shared probe phrase"))
    assert "clB" in _ids(res1)  # authorized pre-revocation

    with store.tx() as conn:
        revoke_grant(conn, gid_b)
    res2 = recall_v3(store, request("shared probe phrase"))
    assert "clB" not in _ids(res2)
    assert not any("beta" in t for t in texts_of(res2))
    assert "clA" in _ids(res2)


# ---------------------------------------------------------------------
# C08 — F4-02: quote enforcement at delivery
# ---------------------------------------------------------------------

def test_c08_read_only_gets_metadata_view_only(store):
    """Read-only authority: the pack still carries the authorized
    metadata view — identity, lifecycle, locator, reasons — but every
    payload-slice item is ``quote_withheld`` with empty text; the raw
    quotation bytes appear nowhere (V4-08.02, C08)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA", verbs=("read",))
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "verbatim secret payload text", _gen(store))
    res = recall_v3(store, request("verbatim secret payload"))
    assert items_of(res)  # the metadata view still ships
    bodies = _bodies(res)
    assert bodies
    for b in bodies:
        assert b.get("quote_authorized") is False
        assert b.get("quote_withheld") is True
        assert b["text"] == ""
        assert b["claim_id"] == "cl1"        # metadata remains
        assert b["span"]["span_id"] == "sp1"
    assert not any(
        "verbatim secret payload text" in t for t in texts_of(res)
    )
    assert "quote_withheld" in res.warnings
    assert any("quote_withheld" in p.warnings for p in res.packs)


def test_c08_quote_holder_still_gets_bytes(store):
    """read+quote authority: the quotation path is unchanged — exact
    bytes ship byte-identically, marked ``quote_authorized``."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")                 # read+quote
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "quotable payload text", _gen(store))
    res = recall_v3(store, request("quotable payload"))
    bodies = _bodies(res)
    assert any(
        b["text"] == "quotable payload text"
        and b.get("quote_authorized") is True
        for b in bodies
    )
    assert not any(b.get("quote_withheld") for b in bodies)


def test_c08_pack_level_inspection_carries_no_bytes(store):
    """``assemble_packs``' own result is an inspection surface too:
    ``AssemblyResult.groups`` must not retain denied bytes — the gate is
    consulted at render, not delegated to an upstream flag (C08)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA", verbs=("read",))
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "inspection surface payload", _gen(store))
    with store.read() as conn:
        req = request("inspection surface payload")
        plan = analyze(req.query, req, 1)
        ctx = _lanes.LaneContext(
            conn=conn, store=store, request=req, plan=plan,
            query_class=QueryClass.CURRENT_STATE, scope_ids=("sA",),
            generation=_gen(store), deadline=_lanes._cand.Deadline(None),
            lane_cap=40,
        )
        lane_res = _lanes.lane_lexical(ctx)
        union = _union.build_union(
            ctx, [lane_res], candidate_cap=64, lane_weights={}
        )
        ranked = _fuse.fuse(
            union, covering_keys=set(union.keys()), query_point_us=None
        )
        seq = itertools.count(1)
        assembled = _pack.assemble_packs(
            ctx, union, ranked, {},
            lambda pk, kind, oid, rev: _infl.mint_handle(
                "rc_test", "human:alice", 0, pk, kind, oid, rev,
                seq=next(seq),
            ),
        )
    assert assembled.quote_withheld > 0
    # no raw bytes on the groups surface …
    assert all(
        not item.text for g in assembled.groups for item in g.items
    )
    # … and none inside serialized pack items either
    assert not any(
        "inspection surface payload" in i.text
        for p in assembled.packs for i in p.items
    )


def test_c08_mixed_quote_authority_per_scope(store):
    """read+quote on A, read-only on B: B contributes at read level (its
    item ids and metadata are authorized) while B's exact bytes are
    withheld and A's render normally — per-scope output-kind
    authorization (V4-10.05)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_scope(conn, "sB", principal="p2", conv="c2")
        seed_auth(conn, "sA")                      # read+quote on A
        seed_auth(conn, "sB", verbs=("read",))     # read-only on B
        gen = _gen(store)
        seed_claim(conn, "clA", "sA", "srcA", "spA",
                   "shared probe phrase alpha", gen)
        seed_claim(conn, "clB", "sB", "srcB", "spB",
                   "shared probe phrase beta bytes", gen)
    res = recall_v3(store, request("shared probe phrase"))
    ids = _ids(res)
    assert "clA" in ids and "clB" in ids
    assert any("alpha" in t for t in texts_of(res))
    assert not any("beta bytes" in t for t in texts_of(res))


def test_c08_required_context_denied_omits_group(store):
    """A required context member whose source bytes the caller may not
    quote invalidates the dependent group (V4-08.05): here the
    attribution span's source lives in read-only scope B, so the claim
    group omits whole rather than shipping a stripped subset."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_scope(conn, "sB", principal="p2", conv="c2")
        seed_auth(conn, "sA")                      # read+quote on A
        seed_auth(conn, "sB", verbs=("read",))     # read-only on B
        gen = _gen(store)
        payload_a = b"the deploy is blocked"
        add_source(conn, "srcA", "sA", payload_a)
        add_span(conn, "spMain", "srcA", 0, len(payload_a))
        payload_b = b"Alice said"
        add_source(conn, "srcB", "sB", payload_b)
        add_span(conn, "spAttr", "srcB", 0, len(payload_b))
        conn.execute(
            "INSERT INTO claims(claim_id,scope_id,created_event)"
            " VALUES('clCtx','sA',1)",
        )
        conn.execute(
            "INSERT INTO claim_revisions(claim_id,revision,state,"
            "recorded_from) VALUES('clCtx',1,'active',1)",
        )
        conn.execute(
            "INSERT INTO claim_evidence(claim_id,revision,span_id,"
            "evidence_role) VALUES('clCtx',1,'spMain','primary')",
        )
        add_fts(conn, "clCtx", 1, "sA", "deploy blocked", gen)
        conn.execute(
            "INSERT INTO context_groups(group_id,scope_id,source_id,"
            "revision,parser_version,operation_key,completeness,"
            "recorded_from) VALUES('cg1','sA','srcA',1,'p1','k1',"
            "'complete',1)",
        )
        conn.execute(
            "INSERT INTO context_members(group_id,span_id,role,required,"
            "ord) VALUES('cg1','spMain','primary',1,0),"
            "('cg1','spAttr','attribution',1,1)",
        )
    res = recall_v3(store, request("deploy blocked"))
    assert "clCtx" not in _ids(res)  # whole group omitted, never half-shipped

    # granting quote on B repairs representability — the boundary is the
    # verb, not the data shape
    with store.tx() as conn:
        create_grant(
            conn, scope_id="sB", principal_id="human:alice",
            verbs={"quote"}, issuer_id="human:alice",
            purposes=["recall"],
        )
    res2 = recall_v3(store, request("deploy blocked"))
    assert "clCtx" in _ids(res2)
    assert any("Alice said" in t for t in texts_of(res2))


def test_c08_derived_views_ship_under_read(store):
    """``read`` permits approved derived views: an episode's stored
    label is the object's own content — not a payload slice — so it
    still ships for a read-only caller (V4-08.02)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA", verbs=("read",))
        conn.execute(
            "INSERT INTO observations(observation_id,scope_id,revision,"
            "text,proof_count,freshness,recorded_from) VALUES('ob1','sA',"
            "1,'derived observation probe',3,'stable',1)",
        )
        conn.execute(
            "INSERT INTO episodes(episode_id,scope_id,revision,kind,"
            "label,recorded_from) VALUES('ep1','sA',1,'task',"
            "'obs episode',1)",
        )
        conn.execute(
            "INSERT INTO episode_members(episode_id,object_kind,object_id,"
            "ord,recorded_from) VALUES('ep1','observation','ob1',0,1)",
        )
    res = recall_v3(store, request("obs episode", modes=("exploratory",)))
    assert "ep1" in _ids(res)
    assert any("obs episode" in t for t in texts_of(res))


def test_c08_context_v3_same_gate(store):
    """``context_v3`` shares the recall pipeline — a read-only caller
    gets the same withheld-metadata items there too (C08)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA", verbs=("read",))
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "context surface payload", _gen(store))
    res = context_v3(
        store,
        ContextRequestV3(
            query="context surface payload", scope_id="sA",
            caller_id="human:alice", purpose="recall",
            packs=(PackKind.EVIDENCE_BUNDLE,),
        ),
    )
    assert not any(
        "context surface payload" in i.text
        for p in res.packs for i in p.items
    )
