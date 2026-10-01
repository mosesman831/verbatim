"""Final-pack recall cache (SPEC_V4 §33; V4-33.01–33.08).

Real ``Store.create`` fixtures (v3 schema) with direct-SQL seeding —
same fixture shape as ``tests/retrieval/v3/test_retrieval_v3.py``.

The cache is OFF by default; every test here enables it explicitly via
``retrieval.cache.enabled`` on the config passed to ``recall_v3``.

Covered:
  - default-off gating and explicit enablement;
  - repeated identical recall serves a hit with identical pack content
    (handles/bytes preserved);
  - grant edits, purge suppression, quarantine holds, content writes,
    and projection-generation bumps all prevent reuse;
  - callers never share entries; disclosure tiers isolate entries;
  - capacity-bounded LRU eviction; abstentions never cached;
  - honest stats + status surface.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from verbatim.config import (
    RetrievalCacheConfig,
    RetrievalConfig,
    VerbatimConfig,
)
from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v3 import RecallRequestV3
from verbatim.governance import (
    create_grant,
    register_principal,
    revoke_grant,
    seed_purposes,
)
from verbatim.retrieval import cache as _rcache
from verbatim.retrieval.v3 import recall_v3
from verbatim.storage.store import Store


_TEST_HMAC_KEY = b"test-hmac-key-padded-to-32-bytes"


def _h(data: bytes) -> bytes:
    return hmac.new(_TEST_HMAC_KEY, data, hashlib.sha256).digest()


@pytest.fixture
def store(tmp_path):
    (tmp_path / "v3.db.key").write_bytes(_TEST_HMAC_KEY)
    s = Store.create(str(tmp_path / "v3.db"))
    yield s
    s.close()


def _gen(store) -> int:
    return store.projection_generation()


def cache_cfg(capacity=256, enabled=True) -> VerbatimConfig:
    return VerbatimConfig(
        retrieval=RetrievalConfig(
            cache=RetrievalCacheConfig(
                enabled=enabled, capacity=capacity
            )
        )
    )


# ---------------------------------------------------------------------
# seeding helpers (mirrors tests/retrieval/v3/test_retrieval_v3.py)
# ---------------------------------------------------------------------

def seed_scope(conn, scope_id, principal="p1", conv="c1", profile="prof",
               vis="conversation"):
    conn.execute(
        "INSERT INTO scopes(scope_id,profile_id,principal_id,workspace_id,"
        "conversation_id,visibility,acl_revision) VALUES(?,?,?,?,?,?,0)",
        (scope_id, profile, principal, "ws", conv, vis),
    )


def seed_auth(conn, scope_id, pid="human:alice", purposes=("recall",),
              verbs=("read", "quote")):
    seed_purposes(conn)
    register_principal(conn, kind="human", principal_id=pid)
    return create_grant(
        conn, scope_id=scope_id, principal_id=pid, verbs=set(verbs),
        issuer_id=pid, purposes=list(purposes),
    )


def add_source(conn, source_id, scope_id, payload: bytes, speaker="u1",
               provenance="direct_user"):
    conn.execute(
        "INSERT INTO sources(source_id,origin,source_kind,scope_id,"
        "speaker_id,created_us) VALUES(?,?,?,?,?,1)",
        (source_id, "test", "user_message", scope_id, speaker),
    )
    conn.execute(
        "INSERT INTO source_revisions(source_id,revision,payload,"
        "payload_hmac,event_us,captured_us,timezone,provenance,metadata_json)"
        " VALUES(?,1,?,?,1,1,'UTC',?,'{}')",
        (source_id, payload, _h(payload), provenance),
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


def add_fts(conn, claim_id, rev, scope_id, text, gen):
    cur = conn.execute(
        "INSERT INTO fts_rows(claim_id,claim_revision,scope_id,"
        "projection_generation) VALUES(?,?,?,?)",
        (claim_id, rev, scope_id, gen),
    )
    conn.execute(
        "INSERT INTO facts_fts(fts_row_id,text) VALUES(?,?)",
        (cur.lastrowid, text),
    )


def seed_claim(conn, claim_id, scope_id, source_id, span_id, text, gen,
               subject=None, predicate=None, state="active"):
    payload = text.encode("utf-8")
    add_source(conn, source_id, scope_id, payload)
    add_span(conn, span_id, source_id, 0, len(payload))
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,?,?,?,1)",
        (claim_id, scope_id, subject, predicate, 1),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,condition_json,"
        "recorded_from,recorded_until,perspective_id,freshness)"
        " VALUES(?,?,?,NULL,1,NULL,NULL,NULL)",
        (claim_id, 1, state),
    )
    conn.execute(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role,family_id) VALUES(?,?,?,'primary',NULL)",
        (claim_id, 1, span_id),
    )
    add_fts(conn, claim_id, 1, scope_id, text, gen)


def request(query="term", scope_id="sA", caller="human:alice", **kw):
    kw.setdefault("purpose", "recall")
    return RecallRequestV3(
        query=query, scope_id=scope_id, caller_id=caller, **kw
    )


def items_of(result):
    return [i for p in result.packs for i in p.items]


def texts_of(result):
    return [i.text for p in result.packs for i in p.items]


def _stats(store):
    cache = getattr(store, "_recall_cache_v4", None)
    return cache.stats() if cache is not None else {}


def _hit(result) -> bool:
    return bool(result.capabilities.get("cache", {}).get("hit"))


# ---------------------------------------------------------------------
# default-off + hit/miss fundamentals
# ---------------------------------------------------------------------

def test_cache_off_by_default(store):
    """No ``retrieval.cache.enabled`` → every recall recomputes; no cache
    object is even attached to the store (V4-33 opt-in)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "default off probe", _gen(store))
    res1 = recall_v3(store, request("default off probe"))
    res2 = recall_v3(store, request("default off probe"))
    assert items_of(res1) and items_of(res2)
    assert not _hit(res1) and not _hit(res2)
    assert getattr(store, "_recall_cache_v4", None) is None


def test_repeated_recall_serves_identical_hit(store):
    """Second identical recall → ``cache.hit`` with byte-identical pack
    content (the stored result is replayed) while each serve mints its
    own fresh delivery handles — a replay is a distinct exposure, never
    folded into the first delivery's influence rows (V4-33.02)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "repeatable probe body", _gen(store))
    cfg = cache_cfg()
    res1 = recall_v3(store, request("repeatable probe"), cfg=cfg)
    res2 = recall_v3(store, request("repeatable probe"), cfg=cfg)
    assert items_of(res1)
    assert not _hit(res1)
    assert _hit(res2)
    assert "cache_hit" in res2.warnings
    assert texts_of(res1) == texts_of(res2)
    h1 = {i.handle.handle_id for i in items_of(res1)}
    h2 = {i.handle.handle_id for i in items_of(res2)}
    assert h1 and h2 and h1.isdisjoint(h2)
    # both deliveries recorded — the hit is real exposure accounting
    with store.read() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM influence WHERE object_id = 'cl1'"
        ).fetchone()[0]
    assert n >= 2
    st = _stats(store)
    assert st["lookups"] == 2 and st["hits"] == 1 and st["misses"] == 1
    assert st["stores"] == 1


def test_query_normalization_shares_entry(store):
    """Exact normalization only: case/whitespace variants of the same
    query share the entry (V4-33.04) — nothing fuzzier is ever equated."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "normalized probe body", _gen(store))
    cfg = cache_cfg()
    res1 = recall_v3(store, request("normalized probe"), cfg=cfg)
    res2 = recall_v3(store, request("  Normalized   PROBE "), cfg=cfg)
    assert _hit(res2)
    assert texts_of(res1) == texts_of(res2)


def test_different_query_never_hits(store):
    """A different question can never reuse another's answer (V4-33.04)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "alpha probe body", _gen(store))
        seed_claim(conn, "cl2", "sA", "src2", "sp2",
                   "omega probe body", _gen(store))
    cfg = cache_cfg()
    res1 = recall_v3(store, request("alpha probe"), cfg=cfg)
    res2 = recall_v3(store, request("omega probe"), cfg=cfg)
    assert items_of(res1) and items_of(res2)
    assert not _hit(res2)
    st = _stats(store)
    assert st["hits"] == 0 and st["misses"] == 2


def test_different_callers_never_share(store):
    """Caller identity is key material — bob's identical query is a
    miss, never alice's served entry (V4-33.08)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA", pid="human:alice")
        seed_auth(conn, "sA", pid="human:bob")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "caller isolation body", _gen(store))
    cfg = cache_cfg()
    res_a = recall_v3(store, request("caller isolation"), cfg=cfg)
    res_b = recall_v3(
        store, request("caller isolation", caller="human:bob"), cfg=cfg
    )
    assert items_of(res_a) and items_of(res_b)
    assert not _hit(res_b)
    st = _stats(store)
    assert st["misses"] == 2 and st["hits"] == 0 and st["stores"] == 2


def test_tier_isolates_entries(store):
    """The disclosure tier is key material — l0 and l1 of the same query
    are distinct entries (V4-33.01)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "tiered probe body", _gen(store),
                   subject="tiered", predicate="probe")
    cfg = cache_cfg()
    res0 = recall_v3(store, request("tiered probe"), cfg=cfg,
                     detail_tier="l0")
    res1 = recall_v3(store, request("tiered probe"), cfg=cfg,
                     detail_tier="l1")
    res0b = recall_v3(store, request("tiered probe"), cfg=cfg,
                      detail_tier="l0")
    assert items_of(res0) and items_of(res1)
    assert not _hit(res1)      # different tier → different key
    assert _hit(res0b)         # same tier → hit
    st = _stats(store)
    assert st["stores"] == 2 and st["hits"] == 1


def test_budget_isolates_entries(store):
    """Different budget buckets never cross-serve (V4-33.01)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "budget probe body", _gen(store))
    cfg = cache_cfg()
    res1 = recall_v3(store, request("budget probe", max_items=8),
                     cfg=cfg)
    res2 = recall_v3(store, request("budget probe", max_items=4),
                     cfg=cfg)
    assert items_of(res1) and items_of(res2)
    assert not _hit(res2)


def test_abstentions_never_cached(store):
    """An abstaining recall stores nothing and never serves — a cached
    entry can only ever replay delivered packs (V4-33.02)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
    cfg = cache_cfg()
    res1 = recall_v3(store, request("nonexistent probe phrase"),
                     cfg=cfg)
    res2 = recall_v3(store, request("nonexistent probe phrase"),
                     cfg=cfg)
    assert res1.abstained and res2.abstained
    st = _stats(store)
    assert st["stores"] == 0 and st["hits"] == 0


# ---------------------------------------------------------------------
# invalidation discipline (V4-33.02/33.03)
# ---------------------------------------------------------------------

def test_grant_edit_invalidates_fingerprint(store):
    """A grant change on the same scope set bumps the scope's epoch —
    the stored fingerprint no longer matches and the entry invalidates
    (counted) rather than serving (V4-33.02)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_scope(conn, "sB", principal="p2", conv="c2")
        seed_auth(conn, "sA")
        gid_b = seed_auth(conn, "sB", pid="human:alice")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "epoch probe body", _gen(store))
    cfg = cache_cfg()
    res1 = recall_v3(store, request("epoch probe"), cfg=cfg)
    assert items_of(res1)
    with store.tx() as conn:
        revoke_grant(conn, gid_b)   # bumps sB's authz_revision (epoch)
    res2 = recall_v3(store, request("epoch probe"), cfg=cfg)
    # sB leaves the authorized set → different scope digest → the stale
    # entry is unreachable; a same-set epoch bump is covered below.
    assert not _hit(res2)


def test_epoch_bump_same_scopes_invalidates(store):
    """Same scope-set digest, changed epoch vector → fingerprint
    mismatch → counted invalidation, fresh compute (V4-33.01)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "epoch bump probe body", _gen(store))
    cfg = cache_cfg()
    res1 = recall_v3(store, request("epoch bump probe"), cfg=cfg)
    assert items_of(res1)
    with store.tx() as conn:
        # Any grant edit bumps the scope's authz_revision.
        create_grant(
            conn, scope_id="sA", principal_id="human:alice",
            verbs={"read"}, issuer_id="human:alice",
            purposes=["recall"],
        )
    res2 = recall_v3(store, request("epoch bump probe"), cfg=cfg)
    assert not _hit(res2)
    assert items_of(res2)
    st = _stats(store)
    assert st["invalidations"] >= 1


def test_revoked_request_scope_never_serves(store):
    """Revoking the requesting scope's own grant between recalls denies
    indistinguishably — the cache cannot resurrect authority (§10.05)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        gid = seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "revoke probe body", _gen(store))
    cfg = cache_cfg()
    res1 = recall_v3(store, request("revoke probe"), cfg=cfg)
    assert items_of(res1)
    with store.tx() as conn:
        revoke_grant(conn, gid)
    with pytest.raises(VerbatimError) as exc:
        recall_v3(store, request("revoke probe"), cfg=cfg)
    assert exc.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_purge_suppression_invalidates(store):
    """A suppressing purge after caching withholds the object — the
    per-ref revalidation on the hit path (and the content watermark)
    both catch it; the recomputed result omits the claim (V4-33.03)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "purge probe alpha body", _gen(store))
        seed_claim(conn, "cl2", "sA", "src2", "sp2",
                   "purge probe beta body", _gen(store))
    cfg = cache_cfg()
    res1 = recall_v3(store, request("purge probe"), cfg=cfg)
    ids1 = {i.handle.object_id for i in items_of(res1)}
    assert "cl1" in ids1
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO purges(purge_id,selection_digest,scope_id,state,"
            "requested_us,approved_us) VALUES('pg1',X'00','sA',"
            "'suppressed',1,2)",
        )
        conn.execute(
            "INSERT INTO purge_targets(purge_id,object_kind,object_id)"
            " VALUES('pg1','claim','cl1')",
        )
    res2 = recall_v3(store, request("purge probe"), cfg=cfg)
    assert not _hit(res2)
    ids2 = {i.handle.object_id for i in items_of(res2)}
    assert "cl1" not in ids2
    assert "cl2" in ids2
    assert not any("alpha" in t for t in texts_of(res2))
    st = _stats(store)
    assert st["invalidations"] + st["ref_denied"] >= 1


def test_quarantine_hold_invalidates(store):
    """A quarantine hold opened after caching suppresses the object —
    the stored pack can never serve held content (V4-33.03)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "held probe alpha body", _gen(store))
        seed_claim(conn, "cl2", "sA", "src2", "sp2",
                   "held probe beta body", _gen(store))
    cfg = cache_cfg()
    res1 = recall_v3(store, request("held probe"), cfg=cfg)
    ids1 = {i.handle.object_id for i in items_of(res1)}
    assert "cl1" in ids1
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO quarantine(object_kind,object_id,revision,"
            "scope_id,state,opened_event) VALUES('claim','cl1',1,'sA',"
            "'pending',2)",
        )
    res2 = recall_v3(store, request("held probe"), cfg=cfg)
    assert not _hit(res2)
    ids2 = {i.handle.object_id for i in items_of(res2)}
    assert "cl1" not in ids2
    assert "cl2" in ids2


def test_content_write_invalidates(store):
    """A new claim committed between recalls changes the content
    watermark — the stale entry invalidates and the fresh result
    includes the new object (V4-33.02)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "freshness probe alpha body", _gen(store))
    cfg = cache_cfg()
    res1 = recall_v3(store, request("freshness probe"), cfg=cfg)
    assert items_of(res1)
    with store.tx() as conn:
        seed_claim(conn, "cl2", "sA", "src2", "sp2",
                   "freshness probe beta body", _gen(store))
    res2 = recall_v3(store, request("freshness probe"), cfg=cfg)
    assert not _hit(res2)
    ids2 = {i.handle.object_id for i in items_of(res2)}
    assert "cl2" in ids2
    st = _stats(store)
    assert st["invalidations"] >= 1


def test_generation_bump_invalidates(store):
    """A projection-generation bump is fingerprinted — cached packs can
    never outlive the index generation they were built against."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "generation probe body", _gen(store))
    cfg = cache_cfg()
    res1 = recall_v3(store, request("generation probe"), cfg=cfg)
    assert items_of(res1)
    with store.tx() as conn:
        conn.execute(
            "UPDATE meta SET value_json = json(999)"
            " WHERE key = 'projection_generation'"
        )
        seed_claim(conn, "cl2", "sA", "src2", "sp2",
                   "generation probe body", 999)
    res2 = recall_v3(store, request("generation probe"), cfg=cfg)
    assert not _hit(res2)
    st = _stats(store)
    assert st["invalidations"] >= 1


# ---------------------------------------------------------------------
# capacity + stats honesty
# ---------------------------------------------------------------------

def test_capacity_bounded_lru_eviction(store):
    """Capacity is a hard bound — a third distinct query evicts the
    oldest entry and counts the eviction (V4-33.05)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "capacity probe alpha body", _gen(store))
        seed_claim(conn, "cl2", "sA", "src2", "sp2",
                   "capacity probe beta body", _gen(store))
        seed_claim(conn, "cl3", "sA", "src3", "sp3",
                   "capacity probe gamma body", _gen(store))
    cfg = cache_cfg(capacity=2)
    for term in ("alpha", "beta", "gamma"):
        res = recall_v3(store, request(f"capacity probe {term}"),
                        cfg=cfg)
        assert items_of(res)
    st = _stats(store)
    assert st["entries"] <= 2
    assert st["evictions"] >= 1
    assert st["stores"] == 3


def test_stats_report_real_counters(store):
    """Hits, misses, invalidations, evictions and hit-rate are the real
    tallies — nothing fabricated (V4-33.07)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "stats probe body", _gen(store))
    cfg = cache_cfg()
    recall_v3(store, request("stats probe"), cfg=cfg)
    recall_v3(store, request("stats probe"), cfg=cfg)
    recall_v3(store, request("stats probe"), cfg=cfg)
    st = _stats(store)
    assert st["lookups"] == 3
    assert st["hits"] == 2
    assert st["misses"] == 1
    assert st["hit_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert st["invalidations"] == 0
    assert st["ref_denied"] == 0


def test_status_surface_reports_cache(store):
    """``stats_for`` exposes enabled flag + real counters; the facade
    capabilities lane reports the same honestly (V4-33.07)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "status probe body", _gen(store))
    cfg = cache_cfg()
    # Off by default — honest "configured but disabled" state.
    off = _rcache.stats_for(store, VerbatimConfig())
    assert off["enabled"] is False
    assert off["degraded_reason"]
    recall_v3(store, request("status probe"), cfg=cfg)
    recall_v3(store, request("status probe"), cfg=cfg)
    on = _rcache.stats_for(store, cfg)
    assert on["enabled"] is True
    assert on["state"] == "healthy"
    assert on["hits"] == 1 and on["misses"] == 1

    from verbatim.api_v3.facade import VerbatimV3

    facade = VerbatimV3(store, cfg)
    caps = facade.capabilities()
    lane = caps["recall_cache"]
    assert lane["rung"] == "healthy"
    assert lane["details"]["hits"] == 1
    # the cache is a status surface, not a query lane (C53)
    assert not any("cache" in name for name in caps["lanes"])
    pd = caps["lanes"]["progressive_disclosure"]
    assert pd["rung"] == "healthy"
    assert pd["details"]["tiers"] == ["l0", "l1", "l2"]


def test_policy_tables_bypass_cache(store):
    """Replay arms carry their own variant tables — they are never
    cache-served or cache-storing (each arm must run its own tables)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "replay probe body", _gen(store))
    cfg = cache_cfg()
    tables = {"abstain": {"term_floor_ratio": 0.4}}
    res1 = recall_v3(store, request("replay probe"), cfg=cfg,
                     policy_tables=tables)
    res2 = recall_v3(store, request("replay probe"), cfg=cfg,
                     policy_tables=tables)
    assert not _hit(res1) and not _hit(res2)
    st = _stats(store)
    assert st.get("stores", 0) == 0
    # and a production recall afterwards populates normally
    res3 = recall_v3(store, request("replay probe"), cfg=cfg)
    res4 = recall_v3(store, request("replay probe"), cfg=cfg)
    assert _hit(res4)
