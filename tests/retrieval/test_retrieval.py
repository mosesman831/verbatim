"""Behavioral tests for the retrieval core (SPEC §29–§31).

The storage repo layer is built in parallel, so tests seed the schema
directly through DDL and exercise the same contract ``search`` relies on:
``store.read()`` yielding a connection and ``store.projection_generation()``.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import sqlite3
import struct

import pytest

from verbatim.core.time import parse_rfc3339
from verbatim.core.types import (
    Lifecycle,
    RecallMode,
    RecallRequest,
    Scope,
)
from verbatim.retrieval import analyze, search
from verbatim.retrieval.candidates import CandidateHit, CandidateMap, gather
from verbatim.retrieval.fusion import rrf
from verbatim.retrieval.package import serialize_item
from verbatim.storage.schema import DDL_FTS5, DDL_V1, FTS_TRIGGERS


# Content digests are verified on reads (SpansRepo.text, SourcesRepo.payload,
# Store._verify_content_digests), so seeders must write real profile-keyed
# HMACs, not placeholder bytes. The key is exactly 32 bytes so a real
# ``Store.create`` can also load it from a ``*.key`` file.
_TEST_HMAC_KEY = b"test-hmac-key-padded-to-32-bytes"


def _h(data: bytes) -> bytes:
    return hmac.new(_TEST_HMAC_KEY, data, hashlib.sha256).digest()


# ---------------------------------------------------------------- store shim

class StoreShim:
    """Minimal stand-in for the parallel-built Store.

    Exposes the exact surface retrieval uses: ``read()``, 
    ``projection_generation()``, and ``fts_enabled`` (which the real
    ``FtsRepo`` consults before searching).
    """

    def __init__(self, conn, generation: int = 1, fts_enabled: bool = True,
                 encode_query=None):
        self._conn = conn
        self._gen = generation
        self.fts_enabled = fts_enabled
        if encode_query is not None:
            self.encode_query = encode_query

    @contextlib.contextmanager
    def read(self):
        yield self._conn

    def projection_generation(self) -> int:
        return self._gen

    def hmac(self, data: bytes) -> bytes:
        """Same shape as ``Store.hmac`` — verified reads delegate here."""
        return _h(data)


def make_conn(fts5: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(DDL_V1)
    if fts5:
        conn.executescript(DDL_FTS5)
        conn.executescript(FTS_TRIGGERS)
    conn.execute(
        "INSERT INTO meta(key, value_json) VALUES('projection_generation','1')"
    )
    return conn


def make_store(conn=None, fts5: bool = True, **kw) -> StoreShim:
    kw.setdefault("fts_enabled", fts5)
    return StoreShim(conn or make_conn(fts5=fts5), **kw)


# ------------------------------------------------------------------ seeders

def us(text: str) -> int:
    return parse_rfc3339(text)


def add_scope(conn, scope_id, principal=None, workspace=None, conv=None,
              vis="conversation", profile="prof"):
    conn.execute(
        "INSERT INTO scopes(scope_id,profile_id,principal_id,workspace_id,"
        "conversation_id,visibility,acl_revision) VALUES(?,?,?,?,?,?,0)",
        (scope_id, profile, principal, workspace, conv, vis),
    )


def add_source(conn, source_id, scope_id, payload: bytes, speaker="user1",
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
    # excerpt_hmac covers exactly the stored byte slice payload[start:end],
    # which may be a sub-range of the revision payload.
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


def add_claim(conn, claim_id, scope_id, created_event=1, entities=(),
              predicate=None):
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,NULL,?,?,1)",
        (claim_id, scope_id, predicate, created_event),
    )
    for eid in entities:
        conn.execute(
            "INSERT OR IGNORE INTO entities(entity_id,scope_id,kind,label,"
            "created_event) VALUES(?,?,NULL,?,0)",
            (eid, scope_id, eid),
        )
        conn.execute(
            "INSERT INTO claim_entities(claim_id,entity_id,role)"
            " VALUES(?,?,'mention')",
            (claim_id, eid),
        )


def add_revision(conn, claim_id, rev=1, state="active", recorded_from=1,
                 recorded_until=None, span_ids=(), intervals=()):
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,recorded_from,"
        "recorded_until) VALUES(?,?,?,?,?)",
        (claim_id, rev, state, recorded_from, recorded_until),
    )
    for sid in span_ids:
        conn.execute(
            "INSERT INTO claim_evidence(claim_id,revision,span_id,"
            "evidence_role) VALUES(?,?,?,'primary')",
            (claim_id, rev, sid),
        )
    for i, (from_us, until_us, precision) in enumerate(intervals):
        conn.execute(
            "INSERT INTO valid_intervals(claim_id,revision,interval_no,"
            "from_us,until_us,precision,basis) VALUES(?,?,?,?,?,?,'test')",
            (claim_id, rev, i, from_us, until_us, precision),
        )


def add_fts(conn, claim_id, rev, scope_id, text, gen=1):
    cur = conn.execute(
        "INSERT INTO fts_rows(claim_id,claim_revision,scope_id,"
        "projection_generation) VALUES(?,?,?,?)",
        (claim_id, rev, scope_id, gen),
    )
    conn.execute(
        "INSERT INTO facts_fts(fts_row_id,text) VALUES(?,?)",
        (cur.lastrowid, text),
    )


def add_conflict(conn, group_id, scope_id, claim_ids):
    conn.execute(
        "INSERT INTO conflict_groups(group_id,scope_id,status)"
        " VALUES(?,?,'open')",
        (group_id, scope_id),
    )
    for cid in claim_ids:
        conn.execute(
            "INSERT INTO conflict_members(group_id,claim_id) VALUES(?,?)",
            (group_id, cid),
        )


def add_purge(conn, purge_id, scope_id, state, targets):
    conn.execute(
        "INSERT INTO purges(purge_id,selection_digest,scope_id,state,"
        "requested_us) VALUES(?,?,?,?,1)",
        (purge_id, b"d", scope_id, state),
    )
    for kind, oid in targets:
        conn.execute(
            "INSERT INTO purge_targets(purge_id,object_kind,object_id)"
            " VALUES(?,?,?)",
            (purge_id, kind, oid),
        )


def add_edge(conn, edge_id, scope_id, source_id, target_id,
             edge_type="conflicts_with", created=1):
    conn.execute(
        "INSERT INTO edges(edge_id,scope_id,source_kind,source_id,"
        "target_kind,target_id,edge_type,created_event)"
        " VALUES(?,?,'claim',?,'claim',?,?,?)",
        (edge_id, scope_id, source_id, target_id, edge_type, created),
    )


def seed_claim_with_text(conn, claim_id, scope_id, source_id, span_id,
                         text: str, **rev_kw):
    """One scope'd claim whose single primary span quotes ``text``."""
    payload = text.encode("utf-8")
    add_source(conn, source_id, scope_id, payload)
    add_span(conn, span_id, source_id, 0, len(payload))
    add_claim(conn, claim_id, scope_id,
              created_event=rev_kw.get("recorded_from", 1),
              entities=rev_kw.pop("entities", ()))
    add_revision(conn, claim_id, span_ids=(span_id,), **rev_kw)
    add_fts(conn, claim_id, rev_kw.get("rev", 1), scope_id, text)


READER = Scope(profile_id="prof", principal_id="p1", conversation_id="c1")


def request(query="neovim", **kw) -> RecallRequest:
    kw.setdefault("scope", READER)
    return RecallRequest(query=query, **kw)


# ------------------------------------------------------------ lexical/scope

def test_lexical_search_respects_scope_boundaries():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    add_scope(conn, "sB", principal="p1", conv="c2")
    add_scope(conn, "sO", principal="p1", vis="owner")
    seed_claim_with_text(conn, "clA", "sA", "srcA", "spA",
                         "I use neovim for personal projects")
    seed_claim_with_text(conn, "clB", "sB", "srcB", "spB",
                         "neovim in another conversation")
    seed_claim_with_text(conn, "clO", "sO", "srcO", "spO",
                         "neovim note visible to owner")

    result = search(make_store(conn), request("neovim"))
    ids = {i.claim_id for i in result.items}
    # conversation c1 sees its own scope plus owner-visible; c2 stays hidden.
    assert ids == {"clA", "clO"}
    assert all("LEXICAL_MATCH" in i.reasons for i in result.items)
    assert result.capabilities["rerank"] is False


def test_lexical_fallback_without_fts5_index():
    conn = make_conn(fts5=False)
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim_with_text(conn, "clA", "sA", "srcA", "spA",
                         "fallback path finds neovim too")
    result = search(make_store(conn), request("neovim"))
    assert {i.claim_id for i in result.items} == {"clA"}


def test_empty_and_stopword_queries_return_no_signal():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim_with_text(conn, "clA", "sA", "srcA", "spA", "stored memory")
    store = make_store(conn)
    for query in ("", "   ", "the and of", "!@#$%^"):
        result = search(store, request(query))
        assert result.items == ()
        assert "no_signal" in result.warnings
        assert result.omitted == 0


def test_uncovered_term_abstains_instead_of_partial_match():
    """V2-29: a query term absent from all indexed evidence means no
    stored span can satisfy the full query — OR-matching the generic
    remainder must not return evidence for a different question."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim_with_text(conn, "clA", "sA", "srcA", "spA",
                         "I prefer neovim over vscode")
    store = make_store(conn)
    # "nonexistent" is nowhere in evidence; "prefer" alone would match clA.
    result = search(store, request("nonexistent prefer"))
    assert result.items == ()
    assert "abstained_uncovered_terms" in result.warnings
    # all terms covered -> normal recall still works
    hit = search(store, request("neovim prefer"))
    assert {i.claim_id for i in hit.items} == {"clA"}


def test_meta_verbs_do_not_block_coverage():
    """Conversational meta-verbs are not content terms: "tell me about X"
    must be judged only on whether X has evidence."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim_with_text(conn, "clA", "sA", "srcA", "spA",
                         "the ferry runs on weekends")
    store = make_store(conn)
    result = search(store, request("tell me about ferry"))
    assert {i.claim_id for i in result.items} == {"clA"}


@pytest.mark.parametrize(
    "predicate,evidence,query",
    [
        ("editor", "I use neovim", "editor?"),
        ("residence", "moved to Berlin", "where do I live?"),
        ("project_database", "we switched to postgres", "project database"),
    ],
)
def test_predicate_query_reaches_structured_claim(predicate, evidence, query):
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    add_source(conn, "srcA", "sA", evidence.encode())
    add_span(conn, "spA", "srcA", 0, len(evidence.encode()))
    add_claim(conn, "clA", "sA", predicate=predicate)
    add_revision(conn, "clA", span_ids=("spA",))
    add_fts(conn, "clA", 1, "sA", evidence)
    result = search(make_store(conn), request(query))
    assert {i.claim_id for i in result.items} == {"clA"}
    assert "abstained_uncovered_terms" not in result.warnings


def test_missing_subject_still_abstains_when_predicate_exists():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    evidence = "I use neovim"
    add_source(conn, "srcA", "sA", evidence.encode())
    add_span(conn, "spA", "srcA", 0, len(evidence))
    add_claim(conn, "clA", "sA", predicate="editor")
    add_revision(conn, "clA", span_ids=("spA",))
    add_fts(conn, "clA", 1, "sA", evidence)
    result = search(make_store(conn), request("Bartholomew editor"))
    assert result.items == ()
    assert "abstained_uncovered_terms" in result.warnings


# ------------------------------------------------------------- bitemporal

def test_known_at_cutoff_resolves_earlier_revision():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    add_source(conn, "src1", "sA", b"db is sqlite")
    add_span(conn, "sp1", "src1", 0, len(b"db is sqlite"))
    add_source(conn, "src2", "sA", b"db is postgres")
    add_span(conn, "sp2", "src2", 0, len(b"db is postgres"))
    add_claim(conn, "clX", "sA")
    add_revision(conn, "clX", rev=1, state="active", recorded_from=1,
                 recorded_until=5, span_ids=("sp1",))
    add_revision(conn, "clX", rev=2, state="superseded", recorded_from=5,
                 span_ids=("sp2",))
    add_fts(conn, "clX", 1, "sA", "db is sqlite")
    add_fts(conn, "clX", 2, "sA", "db is postgres")
    store = make_store(conn)

    # known_at=3: only rev1 was recorded → packaged revision is 1.
    result = search(store, request("db", known_at_seq=3))
    assert [(i.claim_id, i.claim_revision) for i in result.items] == [("clX", 1)]
    assert result.items[0].text == "db is sqlite"

    # no cutoff: latest revision is superseded → invisible in current mode…
    result = search(store, request("db"))
    assert result.items == ()
    # …but visible in historical mode, labeled historical.
    result = search(store, request("db", mode=RecallMode.HISTORICAL))
    assert len(result.items) == 1
    assert result.items[0].claim_revision == 2
    assert result.items[0].lifecycle == Lifecycle.SUPERSEDED
    assert result.items[0].historical is True


def test_known_at_excludes_not_yet_recorded_claim():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim_with_text(conn, "clLate", "sA", "srcL", "spL",
                         "late arriving neovim fact", recorded_from=10)
    result = search(make_store(conn), request("neovim", known_at_seq=3))
    assert result.items == ()


def test_valid_at_filters_applicability():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim_with_text(conn, "clJan", "sA", "srcJ", "spJ",
                         "valid fact alpha", intervals=[
                             (us("2024-01-01T00:00:00Z"),
                              us("2024-02-01T00:00:00Z"), "day")])
    seed_claim_with_text(conn, "clMar", "sA", "srcM", "spM",
                         "valid fact alpha too", intervals=[
                             (us("2024-03-01T00:00:00Z"),
                              us("2024-04-01T00:00:00Z"), "day")])
    seed_claim_with_text(conn, "clU", "sA", "srcU", "spU",
                         "valid fact alpha unknown when")

    result = search(
        make_store(conn),
        request("valid fact", valid_at_us=us("2024-01-15T00:00:00Z")),
    )
    by_id = {i.claim_id: i for i in result.items}
    assert set(by_id) == {"clJan", "clU"}
    assert "TIME_COMPATIBLE" in by_id["clJan"].reasons
    assert "TIME_UNKNOWN" in by_id["clU"].reasons
    assert "time_unknown" in result.warnings
    # In historical mode the out-of-interval claim stays as evidence.
    result = search(
        make_store(conn),
        request("valid fact", valid_at_us=us("2024-01-15T00:00:00Z"),
                mode=RecallMode.HISTORICAL),
    )
    assert "clMar" in {i.claim_id for i in result.items}


def test_query_time_cue_sets_valid_at():
    req = request("editor in 2024-03")
    plan = analyze(req.query, req, us("2024-09-01T00:00:00Z"))
    assert plan.valid_at_us == us("2024-03-01T00:00:00Z")
    assert "2024-03" in plan.terms  # identifier preserved for lexical match


def test_ambiguous_time_cue_warns_not_guesses():
    req = request("what editor last friday")
    plan = analyze(req.query, req, us("2024-09-18T00:00:00Z"))
    assert plan.valid_at_us is None
    assert "ambiguous_time" in plan.warnings


# --------------------------------------------------------- dispute grouping

def test_disputed_pair_emitted_adjacent_with_warning():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim_with_text(conn, "clD1", "sA", "srcD1", "spD1",
                         "graduated in 2019 disputedclaim")
    seed_claim_with_text(conn, "clD2", "sA", "srcD2", "spD2",
                         "graduated in 2021 disputedclaim",
                         entities=("eX",))
    seed_claim_with_text(conn, "clX", "sA", "srcX", "spX",
                         "unrelated disputedclaim filler")
    add_conflict(conn, "g1", "sA", ("clD1", "clD2"))
    conn.execute("UPDATE claim_revisions SET state='disputed'"
                 " WHERE claim_id IN ('clD1','clD2')")

    result = search(
        make_store(conn),
        request("disputedclaim", entity_ids=("eX",)),
    )
    ids = [i.claim_id for i in result.items]
    # clD2 is structured-tier first; its group partner must sit next to it.
    i1, i2 = ids.index("clD1"), ids.index("clD2")
    assert abs(i1 - i2) == 1
    flagged = {i.claim_id: i.disputed for i in result.items}
    assert flagged["clD1"] is True and flagged["clD2"] is True
    assert "unresolved_conflict" in result.warnings


def test_dispute_group_omitted_atomically_when_over_budget():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim_with_text(conn, "clD1", "sA", "srcD1", "spD1",
                         "disputed alpha evidence")
    seed_claim_with_text(conn, "clD2", "sA", "srcD2", "spD2",
                         "disputed beta evidence")
    conn.execute("UPDATE claim_revisions SET state='disputed'")
    add_conflict(conn, "g1", "sA", ("clD1", "clD2"))

    result = search(make_store(conn), request("disputed", limit=1))
    assert result.items == ()
    assert result.omitted == 2
    assert "EVIDENCE_TOO_LARGE" in result.warnings


# --------------------------------------------------------------- budgets

def test_item_limit_enforced():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    for n in range(5):
        seed_claim_with_text(conn, f"cl{n}", "sA", f"src{n}", f"sp{n}",
                             f"commonterm evidence number {n}")
    result = search(make_store(conn), request("commonterm", limit=2))
    assert len(result.items) == 2
    assert result.omitted == 3


def test_byte_budget_drops_whole_items_keeps_warnings():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    # Equal document lengths so local BM25 ties and row-id order applies —
    # the ordering is incidental to what this test exercises.
    texts = [
        "budgetterm aaaa " + "a" * 100,
        "budgetterm bbbb " + "b" * 100,
        "budgetterm cccc " + "c" * 100,
    ]
    for n, t in enumerate(texts):
        seed_claim_with_text(conn, f"cl{n}", "sA", f"src{n}", f"sp{n}", t)
    store = make_store(conn)

    full = search(store, request("budgetterm", limit=8, max_bytes=24_000))
    assert len(full.items) == 3
    sizes = [len(serialize_item(i)) for i in full.items]

    capped = search(
        store, request("budgetterm", limit=8, max_bytes=sizes[0] + sizes[1] + 1)
    )
    assert [i.claim_id for i in capped.items] == [
        i.claim_id for i in full.items[:2]
    ]
    assert capped.omitted >= 1
    assert "EVIDENCE_TOO_LARGE" in capped.warnings
    # surviving items are complete — never truncated mid-quotation.
    assert capped.items[0].text == texts[0]


def test_byte_exact_evidence_reconstruction():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    payload = "prefix héllo wörld suffix".encode("utf-8")
    start = payload.index("héllo".encode("utf-8"))
    end = payload.index(" suffix".encode("utf-8"))
    add_source(conn, "srcU", "sA", payload)
    add_span(conn, "spU", "srcU", start, end)
    add_claim(conn, "clU", "sA")
    add_revision(conn, "clU", span_ids=("spU",))
    add_fts(conn, "clU", 1, "sA", "utf8 evidence")

    result = search(make_store(conn), request("utf8"))
    assert result.items[0].text == payload[start:end].decode("utf-8")
    assert result.items[0].text == "héllo wörld"


def test_purged_payload_skips_item_with_warning():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim_with_text(conn, "clOk", "sA", "srcOk", "spOk",
                         "purgecase surviving evidence")
    # payload erased by the purge workflow: the revision row survives with
    # an empty blob (column is NOT NULL), while the span keeps stale offsets.
    add_source(conn, "srcP", "sA", b"")
    add_span(conn, "spP", "srcP", 0, 1)
    add_claim(conn, "clP", "sA")
    add_revision(conn, "clP", span_ids=("spP",))
    add_fts(conn, "clP", 1, "sA", "purgecase erased evidence")

    result = search(make_store(conn), request("purgecase"))
    assert {i.claim_id for i in result.items} == {"clOk"}
    assert "evidence_unavailable" in result.warnings
    assert result.omitted >= 1


# -------------------------------------------------------------- suppression

def test_suppressed_claim_and_span_excluded():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim_with_text(conn, "clKeep", "sA", "srcK", "spK",
                         "supcase keep me")
    seed_claim_with_text(conn, "clSup", "sA", "srcS", "spS",
                         "supcase suppressed claim")
    seed_claim_with_text(conn, "clSpanSup", "sA", "srcSS", "spSS",
                         "supcase suppressed span claim")
    seed_claim_with_text(conn, "clPrev", "sA", "srcPV", "spPV",
                         "supcase preview only purge")
    add_purge(conn, "pg1", "sA", "suppressed", [("claim", "clSup")])
    add_purge(conn, "pg2", "sA", "purging", [("span", "spSS")])
    add_purge(conn, "pg3", "sA", "previewed", [("claim", "clPrev")])

    result = search(make_store(conn), request("supcase"))
    ids = {i.claim_id for i in result.items}
    assert "clKeep" in ids and "clPrev" in ids
    assert "clSup" not in ids and "clSpanSup" not in ids


# ------------------------------------------------------------ other sources

def test_structured_entity_intersect_requires_all_entities():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim_with_text(conn, "clBoth", "sA", "srcB", "spB",
                         "entitycase claim", entities=("e1", "e2"))
    seed_claim_with_text(conn, "clOne", "sA", "srcO", "spO",
                         "entitycase claim too", entities=("e1",))

    store = make_store(conn)
    req = request("entitycase", entity_ids=("e1", "e2"))
    plan = analyze(req.query, req, us("2024-01-01T00:00:00Z"))
    with store.read() as conn_ro:
        hits = gather(conn_ro, store, plan, req, 1)
    assert "structured" in hits["clBoth"].source_ranks
    assert "structured" not in hits["clOne"].source_ranks
    # And the exact-match tier puts the structured hit first.
    result = search(store, req)
    assert result.items[0].claim_id == "clBoth"
    assert {"ENTITY_MATCH", "EXACT_SLOT"} <= set(result.items[0].reasons)


def test_graph_neighbor_retrieved_one_hop():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim_with_text(conn, "clSeed", "sA", "srcA", "spA",
                         "graphcase anchor evidence")
    seed_claim_with_text(conn, "clNext", "sA", "srcN", "spN",
                         "unrelated vocabulary neighbor")
    add_edge(conn, "edge1", "sA", "clSeed", "clNext", "conflicts_with")

    result = search(make_store(conn), request("graphcase"))
    by_id = {i.claim_id: i for i in result.items}
    assert "clNext" in by_id
    assert "GRAPH_NEIGHBOR" in by_id["clNext"].reasons


def test_semantic_source_when_embeddings_and_encoder_present():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim_with_text(conn, "clSem", "sA", "srcS", "spS",
                         "unmatched lexical text")
    vec = struct.pack("<3f", 1.0, 0.0, 0.0)
    conn.execute(
        "INSERT INTO embeddings(span_id,encoder_id,preprocessing_version,"
        "dimensions,dtype,vector,source_hmac) VALUES(?,?,?,?,?,?,?)",
        # source_hmac must equal the span's persisted excerpt_hmac —
        # the write path keys both on the exact embedded bytes and the
        # scan now excludes stale-input rows (V4-29.05).
        ("spS", "enc1", "pp1", 3, "float32le", vec,
         _h(b"unmatched lexical text")),
    )
    store = make_store(conn, encode_query=lambda q: [0.9, 0.1, 0.0])
    result = search(store, request("anything"))
    by_id = {i.claim_id: i for i in result.items}
    assert "clSem" in by_id
    assert "SEMANTIC_MATCH" in by_id["clSem"].reasons
    assert result.capabilities["semantic"] is True

    # Same path via an Encoder-protocol object on the store (the real
    # verbatim.embeddings.Encoder shape: encode(texts)->[bytes]+encoder_id).
    class FakeEncoder:
        encoder_id = "enc1"

        def encode(self, texts):
            return [struct.pack("<3f", 0.9, 0.1, 0.0)]

    store2 = make_store(conn)
    store2.encoder = FakeEncoder()
    result2 = search(store2, request("anything"))
    assert "clSem" in {i.claim_id for i in result2.items}
    assert result2.capabilities["semantic"] is True


def test_semantic_degraded_without_encoder_or_rows():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim_with_text(conn, "clA", "sA", "srcA", "spA", "plain lexical hit")
    result = search(make_store(conn), request("lexical"))
    assert result.capabilities["semantic"] is False
    assert "semantic" in result.capabilities["degraded"]
    # still returns lexical evidence
    assert result.items


def test_phrase_hit_takes_exact_match_tier():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim_with_text(conn, "clPhrase", "sA", "srcP", "spP",
                         "the alpha beta thing mentioned")
    seed_claim_with_text(conn, "clTerm", "sA", "srcT", "spT",
                         "only gamma lives here alpha")
    result = search(make_store(conn), request('"alpha beta" gamma'))
    assert result.items[0].claim_id == "clPhrase"
    assert "PHRASE_MATCH" in result.items[0].reasons


# ------------------------------------------------------------------- fusion

def test_rrf_math_and_deterministic_tiebreaks():
    hits = CandidateMap()
    a = CandidateHit("clA")
    a.source_ranks = {"lexical": 1}
    a.recorded_from = 5
    b = CandidateHit("clB")
    b.source_ranks = {"lexical": 2, "graph": 1}
    b.recorded_from = 5
    c = CandidateHit("clC")
    c.source_ranks = {"structured": 1}
    c.recorded_from = 1
    for h in (a, b, c):
        hits[h.claim_id] = h

    ranked = rrf(hits)
    order = [cid for cid, _ in ranked]
    assert order == ["clC", "clB", "clA"]  # tier first, then fused score
    scores = dict(ranked)
    assert scores["clA"] == pytest.approx(1 / 61)
    assert scores["clB"] == pytest.approx(1 / 62 + 0.5 / 61)
    assert scores["clC"] == pytest.approx(1 / 61)

    # equal fused scores → recorded_from desc, then claim_id asc.
    hits2 = CandidateMap()
    d = CandidateHit("clD")
    d.source_ranks = {"lexical": 1}
    d.recorded_from = 10
    e = CandidateHit("clE")
    e.source_ranks = {"semantic": 1}
    e.recorded_from = 5
    f = CandidateHit("clF")
    f.source_ranks = {"structured": 2}
    f.recorded_from = 5
    for h in (d, e, f):
        hits2[h.claim_id] = h
    ranked2 = rrf(hits2)
    assert [cid for cid, _ in ranked2] == ["clF", "clD", "clE"]

    g = CandidateHit("clG")
    g.source_ranks = {"lexical": 1}
    g.recorded_from = 10
    hits3 = CandidateMap()
    for h in (d, g):
        hits3[h.claim_id] = h
    assert [cid for cid, _ in rrf(hits3)] == ["clD", "clG"]


# -------------------------------------------------------------- query plan

def test_analyze_preserves_identifiers_and_phrases():
    req = request('what about "exact words" src/main.py MyClass var_2 NEAR')
    plan = analyze(req.query, req, us("2024-01-01T00:00:00Z"))
    assert plan.phrases == ["exact words"]
    for tok in ("src/main.py", "MyClass", "var_2", "NEAR"):
        assert tok in plan.terms
    # user "operators" become quoted literals — never FTS5 syntax.
    assert '"NEAR"' in plan.match_query
    assert '"exact words"' in plan.match_query
    assert not plan.empty


# --------------------------------------------------------- real Store e2e

def test_search_against_real_store(tmp_path):
    """End-to-end: Store.create + tx seeding + search via the real repos."""
    from verbatim.storage.store import Store

    if getattr(Store, "create", None) is None or getattr(Store, "read", None) is None:
        # tests/core/conftest.py installs a duck-typed stub into sys.modules
        # when it collects before the real storage modules load.
        pytest.skip("real storage Store unavailable in this import order")
    # Pin the profile key so the seeders' _h() digests match store.hmac.
    (tmp_path / "mem.db.key").write_bytes(_TEST_HMAC_KEY)
    store = Store.create(str(tmp_path / "mem.db"))
    try:
        with store.tx() as conn:
            add_scope(conn, "sA", principal="p1", conv="c1")
            add_scope(conn, "sB", principal="p1", conv="c2")
            seed_claim_with_text(conn, "clA", "sA", "srcA", "spA",
                                 "e2e neovim with plugins")
            seed_claim_with_text(conn, "clB", "sB", "srcB", "spB",
                                 "e2e neovim hidden conversation")
        result = search(store, request("neovim"))
        assert {i.claim_id for i in result.items} == {"clA"}
        assert result.items[0].text == "e2e neovim with plugins"
        assert result.projection_generation == store.projection_generation()
        assert "semantic" in result.capabilities["degraded"]
    finally:
        store.close()
