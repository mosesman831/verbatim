"""V5 source-projection lane tests (SPEC_V5 §10–§12, §30, §31).

Exercises a real on-disk ``Store`` with the v5 contract tables created
verbatim from docs/v5_contracts.md §3 (schema_v5/repos_v5 land in
parallel — the lane must degrade honestly when they are absent, and
score exactly when they are present).

Covered:

- eligible-corpus BM25 exactness vs a naive reference implementation
  (N/df/avgdl over E — never candidate-set or global stats, V5-11.03/04);
- ``eligible_ids`` and namespace restriction;
- exact identifier postings — case/punctuation/version preserved
  (V5-30.17);
- blocked exact cosine over ``source_vectors`` for the pinned encoder;
- ``entity_timeline`` ordering + ``source_state`` lifecycle labels
  (§30.16, §14.3);
- honest coverage: ``truncated``, ``partial``/``deadline_exceeded``,
  ``unavailable``.
"""

from __future__ import annotations

import math
from collections import Counter

import pytest

from verbatim.embeddings.codec import Float32Codec
from verbatim.retrieval.candidates import Deadline
from verbatim.retrieval.v3 import source_lane as lane
from verbatim.storage.store import Store


_TEST_HMAC_KEY = b"test-hmac-key-padded-to-32-bytes"


@pytest.fixture
def store(tmp_path):
    """Real on-disk store — schema v5 (``schema_v5.py``) supplies the
    source-projection tables verbatim from docs/v5_contracts.md §3."""
    (tmp_path / "v5.db.key").write_bytes(_TEST_HMAC_KEY)
    s = Store.create(str(tmp_path / "v5.db"))
    yield s
    s.close()


@pytest.fixture
def no_fts_store(tmp_path):
    """A real store whose lexical projection table was dropped — the
    pre-v5/partial-migration shape the lane must degrade on."""
    (tmp_path / "nofts.db.key").write_bytes(_TEST_HMAC_KEY)
    s = Store.create(str(tmp_path / "nofts.db"))
    with s.tx() as conn:
        conn.execute("DROP TABLE source_lexical_projection")
        conn.execute("DROP TABLE source_vectors")
        conn.execute("DROP TABLE entity_postings")
    yield s
    s.close()


def seed_projection(conn, source_id, revision, scope_id, tokens, gen=1):
    toks = tokens.split()
    conn.execute(
        "INSERT INTO source_lexical_projection"
        "(source_id,revision,scope_id,generation,tokens,doc_len,digest)"
        " VALUES(?,?,?,?,?,?,?)",
        (source_id, revision, scope_id, gen, tokens, len(toks),
         f"d-{source_id}-{revision}"),
    )


def seed_vector(conn, source_id, revision, namespace, vector,
                encoder="hashing:subword-ngram:v1", gen=1):
    conn.execute(
        "INSERT INTO source_vectors"
        "(source_id,revision,namespace,encoder,generation,vector,digest)"
        " VALUES(?,?,?,?,?,?,?)",
        (source_id, revision, namespace, encoder, gen,
         Float32Codec.pack(vector), f"v-{source_id}-{revision}"),
    )


def seed_posting(conn, namespace, entity, kind, source_id, revision,
                 offsets="[]", gen=1):
    conn.execute(
        "INSERT INTO entity_postings"
        "(namespace,entity,entity_kind,source_id,revision,offsets,"
        "generation) VALUES(?,?,?,?,?,?,?)",
        (namespace, entity, kind, source_id, revision, offsets, gen),
    )


def seed_state(conn, source_id, namespace, disposition="active",
               head="1", control=1, known_at="2026-01-01T00:00:00+00:00",
               superseded_by=None, effective_at=None):
    conn.execute(
        "INSERT INTO source_state"
        "(source_id,namespace,control_version,mutation_head,disposition,"
        "superseded_by,effective_at,known_at,valid_from,valid_to,"
        "updated_at,producer) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (source_id, namespace, control, head, disposition, superseded_by,
         effective_at, known_at, None, None, known_at, "test/v1"),
    )


def seed_source_rows(conn, source_id, scope_id, revision=1,
                     event_us=None, captured_us=None):
    """scopes + sources + source_revisions rows (real FK chain)."""
    conn.execute(
        "INSERT OR IGNORE INTO scopes"
        "(scope_id,profile_id,principal_id,workspace_id,conversation_id,"
        "visibility,acl_revision) VALUES(?,?,?,?,?,?,0)",
        (scope_id, "prof", "alice", "ws", "conv", "conversation"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO sources"
        "(source_id,origin,source_kind,scope_id,speaker_id,created_us)"
        " VALUES(?,?,?,?,?,1)",
        (source_id, "test", "user_message", scope_id, "u1"),
    )
    payload = f"payload-{source_id}-{revision}".encode()
    conn.execute(
        "INSERT OR REPLACE INTO source_revisions"
        "(source_id,revision,payload,payload_hmac,event_us,captured_us,"
        "timezone,provenance,metadata_json)"
        " VALUES(?,?,?,?,?,?,'UTC','direct_user','{}')",
        (source_id, revision, payload, b"\x00" * 32,
         event_us or 1, captured_us or 1),
    )


def seed_enrichment(conn, source_id, revision, event_at=None,
                    producer="enrich/v1", time_status=None,
                    time_precision=None):
    conn.execute(
        "INSERT INTO enrichment"
        "(source_id,revision,producer,type,polarity,time_precision,"
        "time_status,event_at,anchor_at,fields_json)"
        " VALUES(?,?,?,?,?,?,?,?,?,'{}')",
        (source_id, revision, producer, None, None, time_precision,
         time_status, event_at, None),
    )


def _naive_bm25(rows, terms, k1=1.2, b=0.75):
    """Reference BM25 over ``rows`` = [(sid, rev, tokens, doc_len)].

    Uses the same published constants + fsum as the lane, computed
    independently over whatever row set the caller passes — the test
    asserts the lane's E is the intended one.
    """
    n = len(rows)
    avgdl = (sum(r[3] for r in rows) / n) if n else 0.0
    df = Counter()
    for _sid, _rev, tokens, _dl in rows:
        for t in set(tokens.split()) & set(terms):
            df[t] += 1
    out = {}
    for sid, rev, tokens, dl in rows:
        counts = Counter(tokens.split())
        len_adj = (1.0 - b + b * dl / avgdl) if avgdl > 0 else 1.0
        denom = k1 * len_adj
        score = math.fsum(
            math.log(1.0 + (n - df[t] + 0.5) / (df[t] + 0.5))
            * (tf * (k1 + 1.0)) / (tf + denom)
            for t in terms
            for tf in (counts.get(t, 0),)
            if tf and df[t]
        )
        if score:
            out[(sid, rev)] = score
    return out


# ---------------------------------------------------------------------------
# lexical lane — exact eligible-corpus BM25
# ---------------------------------------------------------------------------


def test_lexical_bm25_matches_naive_reference(store):
    with store.tx() as conn:
        seed_projection(conn, "s1", 1, "ns1", "deploy api server nightly")
        seed_projection(conn, "s2", 1, "ns1", "deploy web frontend")
        seed_projection(conn, "s3", 1, "ns1", "database migration deploy plan")
        seed_projection(conn, "s4", 1, "ns1", "unrelated cooking recipe text")
        seed_projection(conn, "s5", 1, "ns2", "deploy api server")  # other ns
    with store.read() as conn:
        hits, stats = lane.lexical_candidates(
            conn, query_terms=["deploy", "server"], namespace="ns1",
            generation=1,
        )
    assert stats.status == "ok"
    assert stats.eligible == 4            # the ns1 corpus, not ns2
    expected = _naive_bm25(
        [
            ("s1", 1, "deploy api server nightly", 4),
            ("s2", 1, "deploy web frontend", 3),
            ("s3", 1, "database migration deploy plan", 4),
            ("s4", 1, "unrelated cooking recipe text", 4),
        ],
        ["deploy", "server"],
    )
    got = {(h.source_id, h.revision): h.signals["lexical"] for h in hits}
    assert set(got) == set(expected)
    for key, score in expected.items():
        assert got[key] == pytest.approx(score, abs=1e-12)
    # deterministic order: score desc, source_id asc, revision desc
    order = [(h.source_id, h.revision) for h in hits]
    assert order == sorted(
        got, key=lambda k: (-got[k], k[0], -k[1])
    )


def test_lexical_stats_cover_full_eligible_corpus(store):
    """A doc sharing no term still counts toward N and avgdl (V5-11.04)."""
    with store.tx() as conn:
        seed_projection(conn, "s1", 1, "ns1", "needle here")
        # 19 ineligible-by-term docs inflate N/avgdl — a candidate-set
        # statistics bug would score s1 as if they did not exist.
        for i in range(19):
            seed_projection(
                conn, f"filler-{i}", 1, "ns1",
                " ".join(["padding"] * 30),          # doc_len = 30
            )
    with store.read() as conn:
        hits, stats = lane.lexical_candidates(
            conn, query_terms=["needle"], namespace="ns1", generation=1,
        )
    assert stats.eligible == 20
    assert stats.details["n_docs"] == 20
    assert stats.details["stats_complete"] is True
    assert len(hits) == 1
    full = _naive_bm25(
        [("s1", 1, "needle here", 2)]
        + [(f"f{i}", 1, "", 30) for i in range(19)],  # seeded doc_len=30
        ["needle"],
    )
    cand_set_only = _naive_bm25([("s1", 1, "needle here", 2)], ["needle"])
    # Equal-N idf is identical either way for df; what diverges is avgdl.
    assert hits[0].signals["lexical"] == pytest.approx(
        full[("s1", 1)], abs=1e-12
    )
    assert hits[0].signals["lexical"] != pytest.approx(
        cand_set_only[("s1", 1)], abs=1e-9
    )


def test_eligible_ids_restrict_lexical(store):
    with store.tx() as conn:
        seed_projection(conn, "keep", 1, "ns1", "alpha beta")
        seed_projection(conn, "keep", 2, "ns1", "alpha beta gamma")
        seed_projection(conn, "drop", 1, "ns1", "alpha alpha alpha")
    with store.read() as conn:
        hits, stats = lane.lexical_candidates(
            conn, query_terms=["alpha"],
            eligible_ids={("keep", 2)},        # pair-pinned revision
            generation=1,
        )
        assert [(h.source_id, h.revision) for h in hits] == [("keep", 2)]
        assert stats.eligible == 1
        # bare source_id admits every revision
        hits, stats = lane.lexical_candidates(
            conn, query_terms=["alpha"], eligible_ids={"keep"},
            generation=1,
        )
        assert {(h.source_id, h.revision) for h in hits} == {
            ("keep", 1), ("keep", 2)
        }
        assert stats.eligible == 2


def test_lexical_generation_fence(store):
    with store.tx() as conn:
        seed_projection(conn, "s1", 1, "ns1", "alpha", gen=1)
        seed_projection(conn, "s1", 2, "ns1", "alpha alpha", gen=2)
    with store.read() as conn:
        hits, stats = lane.lexical_candidates(
            conn, query_terms=["alpha"], namespace="ns1", generation=1,
        )
        assert [(h.source_id, h.revision) for h in hits] == [("s1", 1)]


def test_lexical_truncated_flag(store):
    with store.tx() as conn:
        for i in range(10):
            seed_projection(conn, f"s{i:02d}", 1, "ns1", "alpha doc")
    with store.read() as conn:
        hits, stats = lane.lexical_candidates(
            conn, query_terms=["alpha"], namespace="ns1",
            generation=1, limit=4,
        )
    assert len(hits) == 4
    assert stats.truncated is True
    assert stats.scored == 10


def test_lexical_deadline_partial(store):
    with store.tx() as conn:
        for i in range(600):
            seed_projection(conn, f"s{i}", 1, "ns1", "alpha doc")
    with store.read() as conn:
        hits, stats = lane.lexical_candidates(
            conn, query_terms=["alpha"], namespace="ns1",
            generation=1, deadline=Deadline(0.0),
        )
    assert stats.status == "partial"
    assert stats.deadline_exceeded is True
    assert "scan_incomplete:deadline" in stats.warnings
    assert stats.details["stats_complete"] is False


def test_lexical_missing_table_unavailable(no_fts_store):
    with no_fts_store.read() as conn:
        hits, stats = lane.lexical_candidates(
            conn, query_terms=["x"], namespace="ns1", generation=1,
        )
    assert hits == []
    assert stats.status == "unavailable"
    assert stats.reason == "no_source_lexical_projection"


# ---------------------------------------------------------------------------
# vector lane — blocked exact cosine over source_vectors
# ---------------------------------------------------------------------------


def test_vector_candidates_exact_cosine(store):
    with store.tx() as conn:
        seed_vector(conn, "s1", 1, "ns1", [1.0, 0.0, 0.0])
        seed_vector(conn, "s2", 1, "ns1", [0.9, 0.1, 0.0])
        seed_vector(conn, "s3", 1, "ns1", [0.0, 1.0, 0.0])
        seed_vector(conn, "s4", 1, "ns2", [1.0, 0.0, 0.0])  # other ns
    with store.read() as conn:
        hits, stats = lane.vector_candidates(
            conn, query_vector=[1.0, 0.0, 0.0], namespace="ns1",
            generation=1,
        )
    assert stats.status == "ok"
    assert stats.eligible == 3            # namespace-scoped denominator
    assert [h.source_id for h in hits] == ["s1", "s2", "s3"]
    assert hits[0].signals["similarity"] == pytest.approx(1.0)
    assert hits[1].signals["similarity"] == pytest.approx(0.9 / math.sqrt(0.82))
    assert hits[2].signals["similarity"] == pytest.approx(0.0, abs=1e-12)


def test_vector_eligible_ids_and_revisions(store):
    with store.tx() as conn:
        seed_vector(conn, "a", 1, "ns1", [1.0, 0.0])
        seed_vector(conn, "a", 2, "ns1", [1.0, 0.0])
        seed_vector(conn, "b", 1, "ns1", [1.0, 0.0])
    with store.read() as conn:
        hits, stats = lane.vector_candidates(
            conn, query_vector=[1.0, 0.0],
            eligible_ids={("a", 2), "b"}, namespace="ns1", generation=1,
        )
    assert {(h.source_id, h.revision) for h in hits} == {("a", 2), ("b", 1)}


def test_vector_encoder_pinning_and_malformed(store):
    with store.tx() as conn:
        seed_vector(conn, "s1", 1, "ns1", [1.0, 0.0], encoder="encA")
        seed_vector(conn, "s2", 1, "ns1", [1.0, 0.0], encoder="encB")
        # malformed: blob length not a multiple of dims
        conn.execute(
            "INSERT INTO source_vectors"
            "(source_id,revision,namespace,encoder,generation,vector,digest)"
            " VALUES('bad',1,'ns1','encA',1,?, 'd')",
            (b"\x00\x01",),
        )
    with store.read() as conn:
        # two encoders present, none pinned -> unresolved
        hits, stats = lane.vector_candidates(
            conn, query_vector=[1.0, 0.0], namespace="ns1", generation=1,
        )
        assert stats.status == "unavailable"
        assert stats.reason == "encoder_unresolved"
        # pinned encoder scans only its own space
        hits, stats = lane.vector_candidates(
            conn, query_vector=[1.0, 0.0], namespace="ns1",
            encoder="encA", generation=1,
        )
        assert [h.source_id for h in hits] == ["s1"]
        assert stats.details["excluded_malformed"] == 1


def test_vector_deadline_partial(store):
    with store.tx() as conn:
        for i in range(600):
            seed_vector(conn, f"s{i}", 1, "ns1", [1.0, 0.0])
    with store.read() as conn:
        hits, stats = lane.vector_candidates(
            conn, query_vector=[1.0, 0.0], namespace="ns1",
            generation=1, deadline=Deadline(0.0),
        )
    assert stats.status == "partial"
    assert stats.deadline_exceeded is True


def test_vector_missing_table_unavailable(no_fts_store):
    with no_fts_store.read() as conn:
        hits, stats = lane.vector_candidates(
            conn, query_vector=[1.0], namespace="ns1",
        )
    assert hits == [] and stats.status == "unavailable"


# ---------------------------------------------------------------------------
# identifier / entity postings — exact match only (V5-30.17)
# ---------------------------------------------------------------------------


def test_identifier_exact_match(store):
    with store.tx() as conn:
        seed_posting(conn, "ns1", "Deploy-v2", "version", "s1", 1)
        seed_posting(conn, "ns1", "deploy-v2", "version", "s2", 1)
        seed_posting(conn, "ns1", "deploy-v2.0", "version", "s3", 1)
        seed_posting(conn, "ns1", "/src/deploy-v2.py", "path", "s4", 1)
        seed_posting(conn, "ns2", "deploy-v2", "version", "s5", 1)
    with store.read() as conn:
        hits, stats = lane.identifier_candidates(
            conn, identifiers=["deploy-v2"], namespace="ns1", generation=1,
        )
    got = {(h.source_id, h.revision) for h in hits}
    # case and version segments preserved exactly: only s2 matches
    assert got == {("s2", 1)}
    assert hits[0].signals["identifier_hit"] == 1.0


def test_identifier_and_entity_signals(store):
    with store.tx() as conn:
        seed_posting(conn, "ns1", "api-server", "identifier", "s1", 1)
        seed_posting(conn, "ns1", "Deploy Pipeline", "entity", "s1", 1)
        seed_posting(conn, "ns1", "Deploy Pipeline", "entity", "s2", 1)
        seed_posting(conn, "ns1", "Other Entity", "entity", "s2", 1)
    with store.read() as conn:
        hits, _stats = lane.identifier_candidates(
            conn, identifiers=["api-server"],
            entities=["Deploy Pipeline", "Other Entity"],
            namespace="ns1", generation=1,
        )
    by_id = {h.source_id: h for h in hits}
    assert by_id["s1"].signals["identifier_hit"] == 1.0
    assert by_id["s1"].signals["entity_overlap"] == 1.0
    assert "identifier_hit" not in by_id["s2"].signals
    assert by_id["s2"].signals["entity_overlap"] == 2.0


def test_identifier_eligible_ids_restrict(store):
    with store.tx() as conn:
        seed_posting(conn, "ns1", "deploy-v2", "version", "ok", 1)
        seed_posting(conn, "ns1", "deploy-v2", "version", "held", 1)
    with store.read() as conn:
        hits, _s = lane.identifier_candidates(
            conn, identifiers=["deploy-v2"], eligible_ids={"ok"},
            namespace="ns1", generation=1,
        )
    assert [h.source_id for h in hits] == ["ok"]


# ---------------------------------------------------------------------------
# entity_timeline — §30.16 ordering + lifecycle labels
# ---------------------------------------------------------------------------


def test_entity_timeline_order_and_labels(store):
    with store.tx() as conn:
        for sid, rev, ev in (("s1", 1, 300), ("s2", 1, 100), ("s3", 1, 200)):
            seed_source_rows(conn, sid, "ns1", rev,
                             event_us=ev, captured_us=ev)
            seed_posting(conn, "ns1", "Deploy Pipeline", "entity", sid, rev)
        seed_state(conn, "s1", "ns1", disposition="active", head="1")
        seed_state(conn, "s2", "ns1", disposition="superseded",
                   head="1", superseded_by="s9")
        # s3 has no source_state row -> lifecycle "unknown"
    with store.read() as conn:
        entries, stats = lane.entity_timeline(
            conn, "ns1", "Deploy Pipeline", generation=1,
        )
    assert stats.status == "ok" and stats.returned == 3
    # ascending recorded/event time
    assert [e["source_id"] for e in entries] == ["s2", "s3", "s1"]
    labels = {e["source_id"]: e["lifecycle"] for e in entries}
    assert labels == {"s1": "active", "s2": "superseded", "s3": "unknown"}
    by_id = {e["source_id"]: e for e in entries}
    assert by_id["s1"]["current"] is True
    assert by_id["s2"]["current"] is False
    assert by_id["s2"]["superseded_by"] == "s9"


def test_entity_timeline_prefers_enrichment_event_time(store):
    with store.tx() as conn:
        # recorded order s1 < s2; enrichment reverses the event order
        seed_source_rows(conn, "s1", "ns1", 1, event_us=100, captured_us=100)
        seed_source_rows(conn, "s2", "ns1", 1, event_us=200, captured_us=200)
        seed_posting(conn, "ns1", "X", "entity", "s1", 1)
        seed_posting(conn, "ns1", "X", "entity", "s2", 1)
        seed_state(conn, "s1", "ns1")
        seed_state(conn, "s2", "ns1")
        seed_enrichment(conn, "s1", 1, event_at="2026-03-02T00:00:00+00:00")
        seed_enrichment(conn, "s2", 1, event_at="2026-03-01T00:00:00+00:00")
    with store.read() as conn:
        entries, _ = lane.entity_timeline(conn, "ns1", "X", generation=1)
    # event_at (both parseable) wins over recorded_us for ordering
    assert [e["source_id"] for e in entries] == ["s2", "s1"]
    assert entries[0]["event_at"] == "2026-03-01T00:00:00+00:00"


def test_entity_timeline_exact_match(store):
    with store.tx() as conn:
        seed_posting(conn, "ns1", "Deploy", "entity", "s1", 1)
        seed_posting(conn, "ns1", "deploy", "entity", "s2", 1)
    with store.read() as conn:
        entries, _ = lane.entity_timeline(conn, "ns1", "Deploy")
        assert [e["source_id"] for e in entries] == ["s1"]


# ---------------------------------------------------------------------------
# orchestrator — source_candidates merges the three sub-lanes
# ---------------------------------------------------------------------------


def test_source_candidates_merges_signals(store):
    with store.tx() as conn:
        seed_projection(conn, "s1", 1, "ns1", "deploy api server")
        seed_projection(conn, "s2", 1, "ns1", "deploy only words")
        seed_posting(conn, "ns1", "deploy-v2", "version", "s1", 1)
        seed_vector(conn, "s1", 1, "ns1", [1.0, 0.0])
        seed_vector(conn, "s3", 1, "ns1", [1.0, 0.0])
    with store.read() as conn:
        hits, stats = lane.source_candidates(
            store, conn,
            query_terms=["deploy"], identifiers=["deploy-v2"],
            query_vector=[1.0, 0.0],
            namespace="ns1", snapshot=1, limit=10,
        )
    assert stats.status == "ok"
    by_id = {h.source_id: h for h in hits}
    assert set(by_id) == {"s1", "s2", "s3"}
    # s1 carries all three signals; s2 lexical only; s3 similarity only
    assert set(by_id["s1"].signals) == {
        "lexical", "identifier_hit", "similarity"
    }
    assert set(by_id["s2"].signals) == {"lexical"}
    assert set(by_id["s3"].signals) == {"similarity"}
    # per-sublane coverage is exposed for diagnostics
    assert stats.details["lexical"]["eligible"] == 2
    assert stats.details["vector"]["returned"] == 2
    assert stats.details["identifier"]["returned"] == 1


def test_source_candidates_eligible_ids_restrict_all_lanes(store):
    with store.tx() as conn:
        seed_projection(conn, "in", 1, "ns1", "alpha")
        seed_projection(conn, "out", 1, "ns1", "alpha")
        seed_vector(conn, "in", 1, "ns1", [1.0, 0.0])
        seed_vector(conn, "out", 1, "ns1", [1.0, 0.0])
        seed_posting(conn, "ns1", "id-1", "identifier", "in", 1)
        seed_posting(conn, "ns1", "id-1", "identifier", "out", 1)
    with store.read() as conn:
        hits, stats = lane.source_candidates(
            store, conn, query_terms=["alpha"], identifiers=["id-1"],
            query_vector=[1.0, 0.0], eligible_ids={"in"},
            namespace="ns1", snapshot=1,
        )
    assert {h.source_id for h in hits} == {"in"}


def test_source_candidates_snapshot_object(store):
    """snapshot may be a mapping/object carrying the generation."""
    with store.tx() as conn:
        seed_projection(conn, "s1", 1, "ns1", "alpha", gen=3)
    with store.read() as conn:
        hits, stats = lane.source_candidates(
            store, conn, query_terms=["alpha"], namespace="ns1",
            snapshot={"projection_generation": 3},
        )
    assert [h.source_id for h in hits] == ["s1"]
    assert "generation_unfenced" not in stats.warnings


def test_source_candidates_no_tables_unavailable(no_fts_store):
    with no_fts_store.read() as conn:
        hits, stats = lane.source_candidates(
            no_fts_store, conn, query_terms=["alpha"], identifiers=["id-1"],
            namespace="ns1", snapshot=1,
        )
    assert hits == []
    assert stats.details["lexical"]["status"] == "unavailable"
    assert stats.details["identifier"]["status"] == "unavailable"
    assert stats.details["vector"]["status"] == "unavailable"
    # Every usable channel was unavailable → the lane itself is
    # unavailable, never a clean-looking empty "ok" (V5 honest states).
    assert stats.status == "unavailable"
    assert stats.reason == "channels_unavailable:lexical,identifier,vector"


def test_source_candidates_degraded_channel_warns(store):
    # entity_postings carries an exact hit but no query vector is
    # resolvable: the lane still returns the identifier hit and declares
    # the degraded vector channel in warnings instead of hiding it.
    with store.tx() as conn:
        seed_posting(conn, "ns1", "EXACT-1", "identifier", "s1", 1)
    with store.read() as conn:
        hits, stats = lane.source_candidates(
            store, conn, identifiers=["EXACT-1"],
            namespace="ns1", snapshot=1,
        )
    assert [h.source_id for h in hits] == ["s1"]
    assert stats.status == "ok"
    assert stats.details["vector"]["status"] == "unavailable"
    assert "degraded:vector" in stats.warnings
    assert stats.details["degraded_channels"] == ["vector"]
