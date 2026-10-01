"""F4-11 / F4-12 regression tests (SPEC_V4 §05, §28–§29).

F4-11 (C32): FTS5's GLOBAL bm25() statistics let unauthorized documents
move authorized rankings. The fix computes BM25 over the query's eligible
authorized corpus only — FTS MATCH generates identifiers, never scores.
These tests pin noninterference: additions, deletions, and term-frequency
changes in an unauthorized scope leave the authorized order AND the
authorization-local scores bit-identical.

F4-12 (C33): the dense lane rejected corpora beyond 4,096 eligible spans
because ``load_matrix`` enforced a whole-input bound. The streaming exact
scan (bounded batches + bounded top-k heap) now covers any eligible set —
exercised at 4,096 / 4,097 / >10K spans — and a deadline-limited scan
reports examined/eligible coverage + partial status instead of silently
truncating (V4-29.05/06). Stale-input and vector-space mismatches are
excluded and counted, never scored.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import sqlite3

from verbatim.core.time import parse_rfc3339
from verbatim.core.types import RecallRequest, Scope
from verbatim.embeddings.codec import Float32Codec
from verbatim.embeddings import vectors as _vec
from verbatim.retrieval import analyze, search
from verbatim.retrieval.candidates import Deadline, gather
from verbatim.storage.schema import DDL_FTS5, DDL_V1, DDL_V2, FTS_TRIGGERS


_TEST_HMAC_KEY = b"test-hmac-key-padded-to-32-bytes"


def _h(data: bytes) -> bytes:
    return hmac.new(_TEST_HMAC_KEY, data, hashlib.sha256).digest()


class StoreShim:
    """Same minimal store surface as test_retrieval.py's shim."""

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
        return _h(data)


def make_conn(fts5: bool = True, v2: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(DDL_V1)
    if fts5:
        conn.executescript(DDL_FTS5)
        conn.executescript(FTS_TRIGGERS)
    if v2:
        conn.executescript(DDL_V2)
    conn.execute(
        "INSERT INTO meta(key, value_json) VALUES('projection_generation','1')"
    )
    return conn


def make_store(conn=None, fts5: bool = True, **kw) -> StoreShim:
    kw.setdefault("fts_enabled", fts5)
    return StoreShim(conn or make_conn(fts5=fts5), **kw)


def us(text: str) -> int:
    return parse_rfc3339(text)


def add_scope(conn, scope_id, principal=None, workspace=None, conv=None,
              vis="conversation", profile="prof"):
    conn.execute(
        "INSERT INTO scopes(scope_id,profile_id,principal_id,workspace_id,"
        "conversation_id,visibility,acl_revision) VALUES(?,?,?,?,?,?,0)",
        (scope_id, profile, principal, workspace, conv, vis),
    )


def add_source(conn, source_id, scope_id, payload: bytes, speaker="user1"):
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


def add_claim(conn, claim_id, scope_id, created_event=1):
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,NULL,NULL,?,1)",
        (claim_id, scope_id, created_event),
    )


def add_revision(conn, claim_id, rev=1, state="active", recorded_from=1,
                 recorded_until=None, span_ids=()):
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


def seed_claim_with_text(conn, claim_id, scope_id, source_id, span_id,
                         text: str, **rev_kw):
    payload = text.encode("utf-8")
    add_source(conn, source_id, scope_id, payload)
    add_span(conn, span_id, source_id, 0, len(payload))
    add_claim(conn, claim_id, scope_id,
              created_event=rev_kw.get("recorded_from", 1))
    add_revision(conn, claim_id, span_ids=(span_id,), **rev_kw)
    add_fts(conn, claim_id, rev_kw.get("rev", 1), scope_id, text)


def seed_dense_claim(conn, claim_id, scope_id, source_id, span_id,
                     text: str, vector, encoder_id="enc1"):
    """Claim + span + embedding row with a *consistent* source digest."""
    payload = text.encode("utf-8")
    add_source(conn, source_id, scope_id, payload)
    add_span(conn, span_id, source_id, 0, len(payload))
    add_claim(conn, claim_id, scope_id)
    add_revision(conn, claim_id, span_ids=(span_id,))
    blob = Float32Codec.pack(list(vector))
    conn.execute(
        "INSERT INTO embeddings(span_id,encoder_id,preprocessing_version,"
        "dimensions,dtype,vector,source_hmac) VALUES(?,?,?,?,?,?,?)",
        (span_id, encoder_id, "pp1", len(vector), "float32le", blob,
         _h(payload)),
    )


READER = Scope(profile_id="prof", principal_id="p1", conversation_id="c1")


def request(query="term", **kw) -> RecallRequest:
    kw.setdefault("scope", READER)
    return RecallRequest(query=query, **kw)


class FakeDeadline(Deadline):
    """Deterministic expiry after ``checks`` calls to ``expired``."""

    def __init__(self, checks: int):
        super().__init__(None)
        self._end = 1.0
        self._left = checks

    def expired(self) -> bool:
        if self._left <= 0:
            self.exceeded = True
            return True
        self._left -= 1
        return False


def _lexical_note(res):
    return res.capabilities["notes"]["lexical"]


# ---------------------------------------------------------------------------
# F4-11 — authorization-local ranking statistics (C32)
# ---------------------------------------------------------------------------

def test_private_alpha_flood_cannot_move_authorized_ranking():
    """C32: private-scope alpha-heavy additions must not reorder or rescore
    the authorized alpha/beta results (V4-28.03/04, F4-11)."""
    conn = make_conn()
    add_scope(conn, "sPub", principal="p1", conv="c1")
    add_scope(conn, "sPriv", principal="p2", conv="c9")
    seed_claim_with_text(conn, "clA", "sPub", "srcA", "spA",
                         "alpha beta shared topic note")
    seed_claim_with_text(conn, "clB", "sPub", "srcB", "spB",
                         "alpha alpha alpha focused note")
    seed_claim_with_text(conn, "clC", "sPub", "srcC", "spC",
                         "beta beta other note entirely")
    store = make_store(conn)

    before = search(store, request("alpha beta"))
    order_before = [i.claim_id for i in before.items]
    scores_before = dict(_lexical_note(before)["scores"])
    n_before = _lexical_note(before)["eligible"]
    assert set(order_before) == {"clA", "clB", "clC"}
    assert scores_before  # local scores are reported, not opaque

    # Flood an UNAUTHORIZED scope with alpha-heavy documents in the SAME
    # shared FTS index — under global bm25() this changes df/N/avgdl for
    # every query.
    for i in range(80):
        seed_claim_with_text(conn, f"clP{i}", "sPriv", f"srcP{i}",
                             f"spP{i}", "alpha " * 30 + f"beta flood {i}")

    after = search(store, request("alpha beta"))
    assert [i.claim_id for i in after.items] == order_before
    assert _lexical_note(after)["scores"] == scores_before
    assert _lexical_note(after)["eligible"] == n_before
    # private documents never surface — and are not even hinted at
    assert not any(i.claim_id.startswith("clP") for i in after.items)


def test_unauthorized_tf_changes_and_deletions_are_invisible():
    """Metamorphic: edits/deletes inside the unauthorized corpus leave the
    authorized ordering and scores bit-identical (V4-29.04)."""
    conn = make_conn()
    add_scope(conn, "sPub", principal="p1", conv="c1")
    add_scope(conn, "sPriv", principal="p2", conv="c9")
    seed_claim_with_text(conn, "clA", "sPub", "srcA", "spA",
                         "alpha beta shared topic")
    seed_claim_with_text(conn, "clB", "sPub", "srcB", "spB",
                         "alpha alpha alpha focused")
    for i in range(40):
        seed_claim_with_text(conn, f"clP{i}", "sPriv", f"srcP{i}",
                             f"spP{i}", "alpha " * 20 + f"private {i}")
    store = make_store(conn)
    base = search(store, request("alpha beta"))
    order0 = [i.claim_id for i in base.items]
    scores0 = dict(_lexical_note(base)["scores"])

    # Term-frequency changes on private documents (fires the FTS update
    # trigger — the global index really does change).
    for i in range(40):
        conn.execute(
            "UPDATE facts_fts SET text = ?"
            " WHERE fts_row_id = (SELECT row_id FROM fts_rows"
            "                    WHERE claim_id = ?)",
            ("alpha " * 400, f"clP{i}"),
        )
    mutated = search(store, request("alpha beta"))
    assert [i.claim_id for i in mutated.items] == order0
    assert _lexical_note(mutated)["scores"] == scores0

    # Deletions of private documents (FTS delete trigger fires).
    for i in range(20):
        conn.execute(
            "DELETE FROM facts_fts WHERE fts_row_id ="
            " (SELECT row_id FROM fts_rows WHERE claim_id = ?)",
            (f"clP{i}",),
        )
        conn.execute("DELETE FROM fts_rows WHERE claim_id = ?",
                     (f"clP{i}",))
    deleted = search(store, request("alpha beta"))
    assert [i.claim_id for i in deleted.items] == order0
    assert _lexical_note(deleted)["scores"] == scores0


def test_stats_corpus_excludes_ineligible_authorized_docs():
    """The statistics corpus is the ELIGIBLE set, not merely the scoped
    set: a purge-suppressed same-scope document leaves df/N/avgdl
    untouched while suppressed, and rejoins when the suppression is
    cleared."""
    conn = make_conn()
    add_scope(conn, "sPub", principal="p1", conv="c1")
    seed_claim_with_text(conn, "clA", "sPub", "srcA", "spA",
                         "alpha beta shared topic")
    seed_claim_with_text(conn, "clB", "sPub", "srcB", "spB",
                         "alpha alpha focused note")
    # Same-scope document under a suppressing purge — authorized scope,
    # withheld claim (SPEC §40 suppression semantics).
    seed_claim_with_text(conn, "clHeld", "sPub", "srcH", "spH",
                         "alpha " * 200)
    add_purge(conn, "purge1", "sPub", "suppressed",
              [("claim", "clHeld")])
    store = make_store(conn)
    res = search(store, request("alpha"))
    assert {i.claim_id for i in res.items} == {"clA", "clB"}
    note = _lexical_note(res)
    assert note["eligible"] == 2  # the suppressed document is not corpus

    # Clearing the suppression grows the corpus — and honestly rescores.
    conn.execute(
        "UPDATE purges SET state='previewed' WHERE purge_id='purge1'"
    )
    res2 = search(store, request("alpha"))
    assert _lexical_note(res2)["eligible"] == 3
    assert "clHeld" in {i.claim_id for i in res2.items}


def test_lexical_stats_reported_in_capability_notes():
    """Provenance of the local statistics is visible for audit: candidate
    counts, corpus size, completeness — not the raw global bm25 values."""
    conn = make_conn()
    add_scope(conn, "sPub", principal="p1", conv="c1")
    seed_claim_with_text(conn, "clA", "sPub", "srcA", "spA",
                         "alpha beta shared topic")
    res = search(make_store(conn), request("alpha"))
    note = _lexical_note(res)
    assert note["stats_complete"] is True
    assert note["eligible"] == 1
    assert note["candidates"] == 1 and note["scored"] == 1
    assert res.items[0].claim_id == "clA"


# ---------------------------------------------------------------------------
# F4-12 — streaming exact dense search (C33)
# ---------------------------------------------------------------------------

def _seed_dense_corpus(conn, n: int, scope_id="sA"):
    """``n`` claims, each with a span over one shared source payload and an
    ``enc1`` embedding — bulk inserts keep the fixture fast at 10K scale.
    Every span slices the same bytes, so excerpt_hmac == source_hmac and
    the staleness check passes for all rows."""
    payload = b"shared dense evidence"
    digest = _h(payload)
    add_source(conn, "srcDense", scope_id, payload)
    conn.executemany(
        "INSERT INTO spans(span_id,source_id,revision,start_byte,end_byte,"
        "excerpt_hmac,harvester_version)"
        " VALUES(?, 'srcDense', 1, 0, ?, ?, 't')",
        [(f"sp{i}", len(payload), digest) for i in range(n)],
    )
    conn.executemany(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?, ?, NULL, NULL, 1, 1)",
        [(f"cl{i}", scope_id) for i in range(n)],
    )
    conn.executemany(
        "INSERT INTO claim_revisions(claim_id,revision,state,"
        "recorded_from,recorded_until) VALUES(?, 1, 'active', 1, NULL)",
        [(f"cl{i}",) for i in range(n)],
    )
    conn.executemany(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role) VALUES(?, 1, ?, 'primary')",
        [(f"cl{i}", f"sp{i}") for i in range(n)],
    )
    conn.executemany(
        "INSERT INTO embeddings(span_id,encoder_id,preprocessing_version,"
        "dimensions,dtype,vector,source_hmac) VALUES(?, 'enc1', 'pp1', 3,"
        " 'float32le', ?, ?)",
        [
            (f"sp{i}", Float32Codec.pack([1.0, i * 1e-4, 0.0]), digest)
            for i in range(n)
        ],
    )


def _clear_dense_corpus(conn):
    for table in ("embeddings", "claim_evidence", "claim_revisions",
                  "claims", "spans", "source_revisions", "sources"):
        conn.execute(f"DELETE FROM {table}")


def test_dense_scan_complete_at_4096_4097_and_10k():
    """C33: eligible sets at/past the old 4,096 bound scan completely."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")

    for n in (4096, 4097, 10240):
        _clear_dense_corpus(conn)
        _seed_dense_corpus(conn, n)
        store = make_store(conn, encode_query=lambda q: [1.0, 0.0, 0.0])
        # max request deadline — the test asserts completeness, not speed
        res = search(store, request("anything", deadline_ms=10_000))
        scan = res.capabilities["notes"]["semantic_scan"]
        assert scan["eligible"] == n, n
        assert scan["examined"] == n and scan["scored"] == n
        assert scan["partial"] is False
        assert res.capabilities["semantic"] is True
        # closest vector is span 0 — exact search found it in the stream
        assert res.items[0].claim_id == "cl0", n


def test_dense_deadline_reports_partial_coverage():
    """A deadline cutting the batch scan marks the result partial and
    reports examined < eligible — never claims a completed exact scan."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    n = 1025  # > one 512-row batch
    _seed_dense_corpus(conn, n)
    store = make_store(conn, encode_query=lambda q: [1.0, 0.0, 0.0])
    req = request("anything")
    plan = analyze(req.query, req, us("2024-01-01T00:00:00Z"))
    # expired() call order: lexical gate, structured gate, semantic gate,
    # then one check per 512-row batch inside the scan.
    deadline = FakeDeadline(4)
    with store.read() as conn_ro:
        hits = gather(conn_ro, store, plan, req, 1, deadline=deadline)
    assert "semantic_partial" in hits.warnings
    assert hits.deadline_exceeded is True
    scan = hits.capability_notes["semantic_scan"]
    assert scan["partial"] is True
    assert scan["examined"] == 512 and scan["eligible"] == n
    assert scan["examined"] < scan["eligible"]


def test_dense_scan_excludes_stale_generation_and_malformed():
    """V4-29.05: rows answering a different input or vector space are
    excluded AND counted — never silently scored, never silently dropped."""
    conn = make_conn(v2=True)
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_dense_claim(conn, "clGood", "sA", "srcG", "spG", "good text",
                     [1.0, 0.0, 0.0])
    # stale: embedded text changed since encoding (hmac mismatch)
    seed_dense_claim(conn, "clStale", "sA", "srcS", "spS", "stale text",
                     [1.0, 0.0, 0.0])
    conn.execute("UPDATE embeddings SET source_hmac=? WHERE span_id='spS'",
                 (b"outdated",))
    # foreign generation: 4-dim row cannot score against a 3-dim query
    seed_dense_claim(conn, "clDims", "sA", "srcD", "spD", "dims text",
                     [1.0, 0.0, 0.0, 0.0])
    # malformed: blob shorter than declared dimensions
    seed_dense_claim(conn, "clBad", "sA", "srcB", "spB", "bad text",
                     [1.0, 0.0, 0.0])
    conn.execute(
        "UPDATE embeddings SET vector=? WHERE span_id='spB'",
        (b"\x00\x00",),
    )
    # ledger: input recorded after the known-at cutoff (v2 schema present)
    seed_dense_claim(conn, "clFuture", "sA", "srcF", "spF", "future text",
                     [1.0, 0.0, 0.0])
    conn.execute(
        "INSERT INTO embedding_inputs(input_id,span_id,encoder_id,"
        "preprocessing_version,dependency_digest,input_known_seq)"
        " VALUES('in1','spF','enc1','pp1',?,99)",
        (_h(b"future text"),),
    )

    store = make_store(conn, encode_query=lambda q: [1.0, 0.0, 0.0])
    res = search(store, request("anything", known_at_seq=50))
    scan = res.capabilities["notes"]["semantic_scan"]
    assert scan["eligible"] == 5
    assert scan["scored"] == 1
    assert scan["excluded"] == {
        "stale": 1, "generation": 2, "malformed": 1, "missing": 0,
    }
    assert "semantic_rows_excluded" in res.warnings
    assert [i.claim_id for i in res.items] == ["clGood"]


def test_vectors_search_unit_level_bounds():
    """``_vec.search`` itself: bounded heap, full coverage, honest partial."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    add_source(conn, "srcAll", "sA", b"x")
    # All spans share one source slice so excerpt_hmac == source_hmac for
    # every row — the staleness check passes and vectors actually score.
    for i in range(700):
        conn.execute(
            "INSERT INTO spans(span_id,source_id,revision,start_byte,"
            "end_byte,excerpt_hmac,harvester_version)"
            " VALUES(?, 'srcAll', 1, 0, 1, ?, 't')",
            (f"sp{i}", _h(b"x")),
        )
        conn.execute(
            "INSERT INTO embeddings(span_id,encoder_id,preprocessing_version,"
            "dimensions,dtype,vector,source_hmac) VALUES(?,?,?,?,?,?,?)",
            (f"sp{i}", "enc1", "pp1", 2, "float32le",
             Float32Codec.pack([1.0, i * 1e-3]), _h(b"x")),
        )
    q = Float32Codec.pack([1.0, 0.0])
    ids = [f"sp{i}" for i in range(700)]

    hits, cov = _vec.search(conn, ids, "enc1", q, top_k=10)
    assert cov.eligible == cov.examined == cov.scored == 700
    assert not cov.partial and cov.returned == 10
    # hits are (span_id, score) sorted score-desc, id-asc
    assert hits[0][0] == "sp0"
    assert [h[1] for h in hits] == sorted(
        (h[1] for h in hits), reverse=True
    )

    # deadline before the second batch → partial, examined prefix only
    hits2, cov2 = _vec.search(conn, ids, "enc1", q, top_k=10,
                              deadline=FakeDeadline(1))
    assert cov2.partial is True
    assert cov2.examined == 512 < cov2.eligible
    assert hits2 and len(hits2) == 10

    # batch_size smaller than the corpus still covers everything
    hits3, cov3 = _vec.search(conn, ids, "enc1", q, top_k=5,
                              batch_size=64)
    assert cov3.scored == 700 and not cov3.partial
    assert hits3[0][0] == "sp0"
