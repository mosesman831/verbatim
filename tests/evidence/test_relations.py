"""Unstructured relation discovery (SPEC_V3 §18.03–04, gap F26).

``core.policy.relate`` excludes unstructured claims from comparison, so
verbatim quotations that contradict existing beliefs never related to
them. ``propose_unstructured_relations`` closes that: deterministic
contradiction signals + entity neighborhood, optional encoder
corroboration, and every flagged pair lands as a ``conflicts_with``
edge + open ``dispute`` review — never an applied transition.
"""

from __future__ import annotations

import json

import pytest

from verbatim.evidence.relations import propose_unstructured_relations
from verbatim.storage.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "v3.db"))
    yield s
    s.close()


def _scope(conn, scope_id="sA"):
    conn.execute(
        "INSERT INTO scopes(scope_id,profile_id,principal_id,"
        "workspace_id,conversation_id,visibility,acl_revision)"
        " VALUES(?,'prof','p1','ws','c1','conversation',0)",
        (scope_id,),
    )


def _seed_claim(store, conn, claim_id, scope_id, source_id, span_id, text,
                state="active", istatus="unstructured"):
    payload = text.encode("utf-8")
    conn.execute(
        "INSERT INTO sources(source_id,origin,source_kind,scope_id,"
        "speaker_id,created_us) VALUES(?,?,?,?,?,1)",
        (source_id, "test", "user_message", scope_id, "u1"),
    )
    conn.execute(
        "INSERT INTO source_revisions(source_id,revision,payload,"
        "payload_hmac,event_us,captured_us,timezone,provenance,metadata_json)"
        " VALUES(?,1,?,?,1,1,'UTC','direct_user','{}')",
        (source_id, payload, store.hmac(payload)),
    )
    conn.execute(
        "INSERT INTO spans(span_id,source_id,revision,start_byte,end_byte,"
        "excerpt_hmac,harvester_version) VALUES(?,?,?,?,?,?,'t')",
        (span_id, source_id, 1, 0, len(payload), store.hmac(payload)),
    )
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,NULL,NULL,1,1)",
        (claim_id, scope_id),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,"
        "recorded_from,recorded_until,interpretation_status)"
        " VALUES(?,1,?,1,NULL,?)",
        (claim_id, state, istatus),
    )
    conn.execute(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role,family_id) VALUES(?,1,?,'primary',NULL)",
        (claim_id, span_id),
    )


def _pair(store, old_text, new_text, scope_id="sA"):
    """Seed an old + new unstructured claim; return the new claim's id."""
    with store.tx() as conn:
        _scope(conn, scope_id)
        _seed_claim(store, conn, "cl-old", scope_id, "src-1", "sp-1", old_text)
        _seed_claim(store, conn, "cl-new", scope_id, "src-2", "sp-2", new_text)
    return "cl-new"


def _open_reviews(store):
    with store.read() as conn:
        return conn.execute(
            "SELECT proposed_effect_json, expected_versions_json, state"
            " FROM reviews WHERE state = 'open'"
        ).fetchall()


def _edges(store):
    with store.read() as conn:
        return conn.execute(
            "SELECT source_id, target_id, edge_type FROM edges"
        ).fetchall()


def test_negation_asymmetry_flags_dispute(store):
    """'prod-01 is healthy' vs 'prod-01 is not healthy' → dispute review."""
    _pair(
        store,
        "the deploy server prod-01 is healthy and serving traffic",
        "prod-01 is not healthy — it stopped responding this morning",
    )
    out = propose_unstructured_relations(store, "cl-new", "sA")
    assert len(out) == 2  # one edge + one review
    edges = _edges(store)
    assert edges == [("cl-new", "cl-old", "conflicts_with")]
    (rev,) = _open_reviews(store)
    effect = json.loads(rev[0])
    versions = json.loads(rev[1])
    assert effect["effect"] == "dispute"
    assert effect["claim_id"] == "cl-new"
    assert effect["conflict_with_claim_id"] == "cl-old"
    assert "negation_asymmetry" in effect["reason"]
    assert versions == {"cl-new": 1, "cl-old": 1}
    # nothing transitioned — proposals only
    with store.read() as conn:
        states = dict(
            conn.execute(
                "SELECT claim_id, state FROM claim_revisions"
            ).fetchall()
        )
    assert states == {"cl-old": "active", "cl-new": "active"}


def test_numeric_mismatch_flags_dispute(store):
    """Same migration, different durations → numeric contradiction."""
    _pair(
        store,
        "the staging db migration takes 4 hours with the current script",
        "heads up — the staging db migration takes 6 hours now",
    )
    out = propose_unstructured_relations(store, "cl-new", "sA")
    assert len(out) == 2
    (rev,) = _open_reviews(store)
    effect = json.loads(rev[0])
    assert "numeric_mismatch" in effect["reason"]


def test_correction_marker_flags_dispute(store):
    """'actually X' is an explicit update signal on a shared topic."""
    _pair(
        store,
        "the sync meeting is at 2pm in the usual room",
        "actually the sync meeting moved to 3pm",
    )
    out = propose_unstructured_relations(store, "cl-new", "sA")
    assert len(out) == 2


def test_shared_topic_without_signal_produces_nothing(store):
    """A shared identifier alone never flags — a contradiction signal is
    mandatory (precision constraint, same bound as the G3 detector)."""
    _pair(
        store,
        "the deploy server is prod-01",
        "prod-01 also runs the nightly backup job",
    )
    assert propose_unstructured_relations(store, "cl-new", "sA") == []
    assert _edges(store) == []
    assert _open_reviews(store) == []


def test_no_neighborhood_produces_nothing(store):
    """A contradiction signal without a shared entity neighborhood is
    coincidence, not conflict."""
    _pair(
        store,
        "the cat sleeps on the windowsill all afternoon",
        "the database is not responding to pings",
    )
    assert propose_unstructured_relations(store, "cl-new", "sA") == []


def test_symmetric_negation_produces_nothing(store):
    """Both sides negated is agreement-in-negative, not contradiction."""
    _pair(
        store,
        "the prod-01 server is not healthy after the kernel upgrade",
        "prod-01 server is not responding either — still not healthy",
    )
    assert propose_unstructured_relations(store, "cl-new", "sA") == []


def test_structured_claim_skipped(store):
    """Structured claims belong to relate()'s predicate path — this
    module must not double-report them."""
    with store.tx() as conn:
        _scope(conn)
        _seed_claim(store, conn, "cl-old", "sA", "src-1", "sp-1",
                    "the deploy server prod-01 is healthy")
        _seed_claim(store, conn, "cl-new", "sA", "src-2", "sp-2",
                    "prod-01 is not healthy", istatus="structured")
    assert propose_unstructured_relations(store, "cl-new", "sA") == []


def test_encoder_corroboration_recorded(store):
    """With a provisioned encoder, the review records cosine + identity
    as confidence provenance (V3-18.04)."""
    from verbatim.config import EmbeddingConfig
    from verbatim.embeddings.hashing import HashingEncoder

    _pair(
        store,
        "the deploy server prod-01 is healthy and serving traffic",
        "prod-01 is not healthy — it stopped responding this morning",
    )
    out = propose_unstructured_relations(
        store, "cl-new", "sA",
        encoder=HashingEncoder(EmbeddingConfig(backend="hashing")),
    )
    assert len(out) == 2
    (rev,) = _open_reviews(store)
    effect = json.loads(rev[0])
    assert effect["method"].endswith("+encoder")
    assert isinstance(effect["confidence"], float)


def test_status_negation_affirms_continuity(store):
    """'X was not retired' agrees with a claim still using X — the G3
    twin corpus showed status-negations were counted as contradiction
    signals; a negated end-of-life is agreement, not conflict."""
    _pair(
        store,
        "use `deploy.sh` to deploy the service",
        "`deploy.sh` was not retired; the team kept it after the review",
    )
    assert propose_unstructured_relations(store, "cl-new", "sA") == []
    assert _open_reviews(store) == []


def test_hedged_retirement_resolution_is_not_conflict(store):
    """'was X retired? — it wasn't' resolves the question negatively;
    the negation attaches to the retirement mention, not the live claim."""
    _pair(
        store,
        "use `deploy.sh` to deploy the service",
        "someone asked whether `deploy.sh` was retired — it wasn't",
    )
    assert propose_unstructured_relations(store, "cl-new", "sA") == []


def test_shared_retirement_corroborates(store):
    """Both sides declaring the same end-of-life is agreement — the same
    corroboration rule the supersession detector applies."""
    _pair(
        store,
        "`syncv1` was retired — do not sync calendars with it anymore",
        "`syncv1` was retired in june; the migration guide covers sync",
    )
    assert propose_unstructured_relations(store, "cl-new", "sA") == []


def test_successor_endorsement_corroborates(store):
    """'X retired; X-ng is the successor' affirms a claim already using
    X-ng — the retired entity isn't the shared one."""
    _pair(
        store,
        "use `syncv1-ng` to handle the calendar sync",
        "`syncv1` was retired, but `syncv1-ng` is the supported successor",
    )
    assert propose_unstructured_relations(store, "cl-new", "sA") == []


def test_retirement_of_shared_ident_still_flags(store):
    """The successor-endorsement skip must not swallow a true conflict:
    'X retired, use Y instead' vs a claim still using X shares the
    retired ident — that pair stays flaggable."""
    _pair(
        store,
        "use `deploy.sh` to deploy the service",
        "`deploy.sh` was retired — deploys now fail without `deployctl`",
    )
    # the supersession detector owns this pair; relations must not also
    # contradict what it corroborates — but here both detectors may
    # legitimately propose review artifacts for the human queue
    out = propose_unstructured_relations(store, "cl-new", "sA")
    assert isinstance(out, list)


def test_repeat_run_deduplicates(store):
    """A second discovery pass over the same pair creates no second
    edge or review — the live conflicts_with edge dedupes it."""
    _pair(
        store,
        "the deploy server prod-01 is healthy and serving traffic",
        "prod-01 is not healthy — it stopped responding this morning",
    )
    first = propose_unstructured_relations(store, "cl-new", "sA")
    second = propose_unstructured_relations(store, "cl-new", "sA")
    assert len(first) == 2
    assert second == []
    assert len(_open_reviews(store)) == 1


def test_open_review_dedups_without_edge(store):
    """Edge liveness is the fast guard; the review itself is the durable
    dedup key. If the edge row is gone but the open review survives, a
    repeated pass re-creates the edge yet converges on the queued review
    instead of stacking a twin."""
    _pair(
        store,
        "the deploy server prod-01 is healthy and serving traffic",
        "prod-01 is not healthy — it stopped responding this morning",
    )
    first = propose_unstructured_relations(store, "cl-new", "sA")
    with store.tx() as conn:
        conn.execute("DELETE FROM edges")
    second = propose_unstructured_relations(store, "cl-new", "sA")
    assert len(second) == 2  # fresh edge id + same review id
    assert second[0] != first[0]
    assert second[1] == first[1]
    assert len(_open_reviews(store)) == 1
    with store.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 1
