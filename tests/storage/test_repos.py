"""Repository tests: dedup, spans, bitemporal claims, scope isolation, graph."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.storage.conftest import make_envelope, make_scope

from verbatim.core.types import (
    ErrorCode,
    Lifecycle,
    Precision,
    Provenance,
    SourceKind,
    TimeInterval,
    VerbatimError,
    Visibility,
    new_id,
)
from verbatim.storage.repos import (
    ClaimsRepo,
    ConsentsRepo,
    DecisionsRepo,
    EdgesRepo,
    EntitiesRepo,
    EventsRepo,
    FeedbackRepo,
    FtsRepo,
    PurgesRepo,
    ReviewsRepo,
    SourcesRepo,
    SpansRepo,
    ensure_scope,
    readable_scope_ids,
)
from verbatim.storage.store import Store


# ----------------------------------------------------------------------
# sources
# ----------------------------------------------------------------------


def test_source_insert_dedup_and_payload_roundtrip(store: Store, scope) -> None:
    repo = SourcesRepo(store)
    payload = "I use Neovim — täglich".encode("utf-8")
    env = make_envelope(
        scope, payload=payload, external_id="msg-1",
        provenance=Provenance.DIRECT_USER,
    )
    source_id, created = repo.insert(env)
    assert created is True
    assert source_id

    # exact repeat of the dedup key + revision is a duplicate
    again_id, again_created = repo.insert(env)
    assert again_id == source_id
    assert again_created is False

    # payload returns byte-exact
    assert repo.payload(source_id, 1) == payload
    assert repo.latest_revision(source_id) == 1

    row = repo.get(source_id)
    assert row["origin"] == "test-harness"
    assert row["external_id"] == "msg-1"
    assert row["source_kind"] == SourceKind.USER_MESSAGE.value
    assert row["speaker_id"] == "alice"

    rev = repo.get_revision(source_id, 1)
    assert rev["payload_bytes"] == len(payload)
    assert rev["provenance"] == Provenance.DIRECT_USER.value
    assert "payload" not in rev  # metadata view never leaks bytes


def test_source_edit_creates_new_revision(store: Store, scope) -> None:
    repo = SourcesRepo(store)
    env = make_envelope(scope, payload=b"v1", external_id="m-9")
    source_id, _ = repo.insert(env)
    edited = replace(env, payload=b"v2", revision=2)
    same_id, created = repo.insert(edited)
    assert same_id == source_id
    assert created is True
    assert repo.latest_revision(source_id) == 2
    # prior revision keeps its original bytes
    assert repo.payload(source_id, 1) == b"v1"
    assert repo.payload(source_id, 2) == b"v2"


def test_source_dedup_conflict_rejected(store: Store, scope) -> None:
    repo = SourcesRepo(store)
    env = make_envelope(scope, payload=b"original", external_id="m-7")
    repo.insert(env)
    poisoned = replace(env, payload=b"different bytes")
    with pytest.raises(VerbatimError) as exc:
        repo.insert(poisoned)
    assert exc.value.code == ErrorCode.VALIDATION


def test_source_without_external_id_never_deduped(store: Store, scope) -> None:
    repo = SourcesRepo(store)
    env = make_envelope(scope, payload=b"identical text", external_id=None)
    first, _ = repo.insert(env)
    second, created = repo.insert(env)
    # identical text is not identity (SPEC §10)
    assert second != first
    assert created is True


def test_source_missing_reads(store: Store) -> None:
    repo = SourcesRepo(store)
    assert repo.get("nope") is None
    assert repo.get_revision("nope", 1) is None
    assert repo.payload("nope", 1) is None
    assert repo.latest_revision("nope") is None


# ----------------------------------------------------------------------
# content integrity: persisted HMACs are re-verified on read
# ----------------------------------------------------------------------


def test_source_payload_tampered_raises_corrupt(store: Store, scope) -> None:
    """Bytes that no longer match ``payload_hmac`` are corruption, never
    content — the read path raises instead of serving them silently."""
    repo = SourcesRepo(store)
    env = make_envelope(scope, payload=b"attack at dawn", external_id="t-1")
    source_id, _ = repo.insert(env)
    with store.tx() as conn:
        conn.execute(
            "UPDATE source_revisions SET payload = ?"
            " WHERE source_id = ? AND revision = 1",
            (b"attack at dusk", source_id),
        )
    with pytest.raises(VerbatimError) as exc:
        repo.payload(source_id, 1)
    assert exc.value.code == ErrorCode.STORE_CORRUPT


def test_source_payload_tampered_hmac_raises_corrupt(store: Store, scope) -> None:
    """A rewritten digest is corruption too — the stored digest is a
    checksum, not a trusted field."""
    repo = SourcesRepo(store)
    env = make_envelope(scope, payload=b"attack at dawn", external_id="t-1b")
    source_id, _ = repo.insert(env)
    with store.tx() as conn:
        conn.execute(
            "UPDATE source_revisions SET payload_hmac = ?"
            " WHERE source_id = ? AND revision = 1",
            (b"\x00" * 32, source_id),
        )
    with pytest.raises(VerbatimError) as exc:
        repo.payload(source_id, 1)
    assert exc.value.code == ErrorCode.STORE_CORRUPT


# NOTE: ``payload_hmac`` is ``NOT NULL`` in the current schema, so the
# NULL-digest (legacy) path can't be produced via UPDATE — the read-side
# NULL check is defensive, matching ``check_integrity``'s ``unverified``
# accounting (NULL digests are legacy rows, never condemned). The
# nullable ``source_views.integrity_digest`` NULL path *is* tested below.


def test_source_payload_purged_still_verifies(store: Store, scope) -> None:
    """Purge reseals the emptied payload with a matching digest — the read
    path verifies it and returns ``b''`` (erased, not corrupt)."""
    repo = SourcesRepo(store)
    env = make_envelope(scope, payload=b"gone", external_id="t-3")
    source_id, _ = repo.insert(env)
    with store.tx() as conn:
        conn.execute(
            "UPDATE source_revisions SET payload = X'', payload_hmac = ?"
            " WHERE source_id = ? AND revision = 1",
            (store.hmac(b""), source_id),
        )
    assert repo.payload(source_id, 1) == b""


# ----------------------------------------------------------------------
# spans
# ----------------------------------------------------------------------


def _source_with(store: Store, scope, payload: bytes) -> str:
    source_id, _ = SourcesRepo(store).insert(
        make_envelope(scope, payload=payload)
    )
    return source_id


def test_span_valid_and_text_roundtrip(store: Store, scope) -> None:
    payload = "I use Neovim — daily".encode("utf-8")
    source_id = _source_with(store, scope, payload)
    spans = SpansRepo(store)
    start = payload.index("—".encode("utf-8"))
    end = start + len("—".encode("utf-8"))
    spans.insert("span-1", source_id, 1, start, end, "harvester-1")
    assert spans.text("span-1") == "—"
    whole = spans.insert("span-2", source_id, 1, 0, len(payload), "harvester-1")
    assert spans.text(whole) == payload.decode("utf-8")


def test_span_rejects_mid_utf8_offsets(store: Store, scope) -> None:
    payload = "a—b".encode("utf-8")  # em dash occupies bytes 1..4
    source_id = _source_with(store, scope, payload)
    spans = SpansRepo(store)
    for bad in ((2, 4), (1, 3), (0, 2)):
        with pytest.raises(VerbatimError) as exc:
            spans.insert(new_id(), source_id, 1, bad[0], bad[1], "h-1")
        assert exc.value.code == ErrorCode.VALIDATION


def test_span_rejects_bad_ranges_and_revisions(store: Store, scope) -> None:
    payload = b"abc"
    source_id = _source_with(store, scope, payload)
    spans = SpansRepo(store)
    for args in ((0, 0), (2, 2), (3, 4), (-1, 2), (0, 99)):
        with pytest.raises(VerbatimError) as exc:
            spans.insert(new_id(), source_id, 1, args[0], args[1], "h-1")
        assert exc.value.code == ErrorCode.VALIDATION
    # nonexistent source revision reveals nothing
    with pytest.raises(VerbatimError) as exc:
        spans.insert(new_id(), source_id, 77, 0, 1, "h-1")
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


def test_span_text_missing_is_none(store: Store) -> None:
    assert SpansRepo(store).text("no-such-span") is None


def test_span_text_tampered_payload_raises_corrupt(store: Store, scope) -> None:
    """A valid-UTF-8 rewrite inside the span bounds still fails — the
    digest catches tampering that a decode check alone would serve."""
    payload = b"deploy prod-01 on friday"
    source_id = _source_with(store, scope, payload)
    spans = SpansRepo(store)
    spans.insert("sp-x", source_id, 1, 7, 14, "h-1")  # "prod-01"
    assert spans.text("sp-x") == "prod-01"
    with store.tx() as conn:
        conn.execute(
            "UPDATE source_revisions SET payload = ?"
            " WHERE source_id = ? AND revision = 1",
            (b"deploy prod-99 on friday", source_id),  # same byte length
        )
    with pytest.raises(VerbatimError) as exc:
        spans.text("sp-x")
    assert exc.value.code == ErrorCode.STORE_CORRUPT


def test_span_text_tampered_offsets_raise_corrupt(store: Store, scope) -> None:
    """Shifted offsets change the slice — same digest failure."""
    payload = b"deploy prod-01 on friday"
    source_id = _source_with(store, scope, payload)
    spans = SpansRepo(store)
    spans.insert("sp-x", source_id, 1, 7, 14, "h-1")
    with store.tx() as conn:
        conn.execute("UPDATE spans SET start_byte = 0 WHERE span_id = 'sp-x'")
    with pytest.raises(VerbatimError) as exc:
        spans.text("sp-x")
    assert exc.value.code == ErrorCode.STORE_CORRUPT


def test_span_text_purged_returns_none(store: Store, scope) -> None:
    """An emptied revision payload is a purge tombstone, not corruption —
    the span skeleton survives for lineage but the excerpt is gone."""
    payload = b"deploy prod-01 on friday"
    source_id = _source_with(store, scope, payload)
    spans = SpansRepo(store)
    spans.insert("sp-x", source_id, 1, 7, 14, "h-1")
    with store.tx() as conn:
        conn.execute(
            "UPDATE source_revisions SET payload = X'', payload_hmac = ?"
            " WHERE source_id = ? AND revision = 1",
            (store.hmac(b""), source_id),
        )
    assert spans.text("sp-x") is None


# ----------------------------------------------------------------------
# claims: bitemporal lifecycle
# ----------------------------------------------------------------------


@pytest.fixture()
def claim_setup(store: Store, scope, scope_id):
    """A claim with evidence span, plus repo handles and an event seq."""
    events = EventsRepo(store)
    claims = ClaimsRepo(store)
    source_id = _source_with(store, scope, b"the quick brown fox")
    SpansRepo(store).insert("sp-1", source_id, 1, 4, 9, "h-1")
    with store.tx() as conn:
        seq1 = events.append(conn, scope_id, "claim_created", "alice", {}, "pol-1")
        claim_id = claims.create(scope_id, "user:alice", "editor", conn)
        claims.add_revision(
            claim_id, Lifecycle.PENDING, {"value": "neovim"}, "affirmative",
            "asserted", None, None,
            [TimeInterval(from_us=100, precision=Precision.DAY, basis="explicit")],
            [("sp-1", "primary")], seq1, conn,
        )
    return {"claim_id": claim_id, "scope_id": scope_id, "seq1": seq1,
            "claims": claims, "events": events}


def test_claim_revision_lifecycle_and_known_at(store: Store, scope_id, claim_setup) -> None:
    claims = claim_setup["claims"]
    events = claim_setup["events"]
    cid = claim_setup["claim_id"]
    seq1 = claim_setup["seq1"]

    cur = claims.current(cid)
    assert cur["revision"] == 1
    assert cur["state"] == Lifecycle.PENDING.value
    assert cur["object"] == {"value": "neovim"}
    assert cur["valid_intervals"][0]["from_us"] == 100
    assert cur["evidence"] == [{"span_id": "sp-1", "evidence_role": "primary"}]

    with store.tx() as conn:
        seq2 = events.append(conn, scope_id, "claim_admitted", "alice", {}, "pol-1")
        rev2 = claims.add_revision(
            cid, Lifecycle.ACTIVE, {"value": "helix"}, "affirmative",
            "asserted", {"op": "eq", "key": "ctx", "value": "personal"},
            {"note": "operator approved"}, [], [], seq2, conn,
        )
    assert rev2 == 2

    # latest belief: revision 2
    assert claims.current(cid)["object"] == {"value": "helix"}
    # belief at seq1: revision 2 was not yet recorded
    past = claims.current(cid, known_at_seq=seq1)
    assert past["revision"] == 1
    assert past["object"] == {"value": "neovim"}
    # boundary: seq2-1 still sees rev1 (rev1 has no recorded_until yet)
    assert claims.current(cid, known_at_seq=seq2 - 1)["revision"] == 1

    # close rev1's recorded bound at seq2: it stops being known at seq2
    with store.tx() as conn:
        claims.set_recorded_until(cid, 1, seq2, conn)
    assert claims.current(cid, known_at_seq=seq2 - 1)["revision"] == 1
    assert claims.current(cid, known_at_seq=seq2)["revision"] == 2

    # closing rev2 too leaves nothing "currently believed"
    with store.tx() as conn:
        seq3 = events.append(conn, scope_id, "claim_retired", "alice", {}, "pol-1")
        claims.set_recorded_until(cid, 2, seq3, conn)
    assert claims.current(cid, known_at_seq=seq3) is None
    # but history at seq2 is unchanged — replay never resurrects nothing
    assert claims.current(cid, known_at_seq=seq2)["revision"] == 2


def test_claim_row_versions(store: Store, claim_setup) -> None:
    claims = claim_setup["claims"]
    cid = claim_setup["claim_id"]
    row = claims.get(cid)
    assert row["predicate"] == "editor"
    assert row["row_version"] == 2  # create + one revision
    with store.tx() as conn:
        claims.add_revision(
            cid, "active", None, "affirmative", "asserted", None, None,
            [], [], claim_setup["seq1"] + 1, conn,
        )
    assert claims.get(cid)["row_version"] == 3
    assert claims.get("missing") is None


def test_claim_revision_validates_enums(store: Store, claim_setup) -> None:
    claims = claim_setup["claims"]
    cid = claim_setup["claim_id"]
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            claims.add_revision(
                cid, "not-a-state", None, "affirmative", "asserted",
                None, None, [], [], 5, conn,
            )
        assert exc.value.code == ErrorCode.VALIDATION
        with pytest.raises(VerbatimError):
            claims.add_revision(
                cid, "active", None, "affirmative", "asserted",
                None, None, [], [("sp-1", "bogus-role")], 5, conn,
            )


def test_add_revision_unknown_claim(store: Store) -> None:
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            ClaimsRepo(store).add_revision(
                "no-claim", "active", None, "affirmative", "asserted",
                None, None, [], [], 1, conn,
            )
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


def test_list_current_filters_state_and_scope(store: Store, claim_setup) -> None:
    claims = claim_setup["claims"]
    cid = claim_setup["claim_id"]
    scope_id = claim_setup["scope_id"]
    # currently pending only
    rows = claims.list_current(scope_id, ("pending",))
    assert [r["claim_id"] for r in rows] == [cid]
    assert claims.list_current(scope_id, ("active",)) == []
    assert claims.list_current(scope_id, ()) == []


# ----------------------------------------------------------------------
# scope isolation via identity predicates
# ----------------------------------------------------------------------


def test_scope_isolation_list_current(store: Store) -> None:
    claims = ClaimsRepo(store)
    scope_a = make_scope(conversation_id="conv-A")
    scope_b = make_scope(conversation_id="conv-B")

    with store.tx() as conn:
        sid_a = ensure_scope(store, conn, scope_a)
        sid_b = ensure_scope(store, conn, scope_b)
        cid_a = claims.create(sid_a, "u", "editor", conn)
        claims.add_revision(cid_a, "active", None, "affirmative", "asserted",
                            None, None, [], [], 1, conn)
        cid_b = claims.create(sid_b, "u", "editor", conn)
        claims.add_revision(cid_b, "active", None, "affirmative", "asserted",
                            None, None, [], [], 1, conn)

    assert [r["claim_id"] for r in claims.list_current(sid_a, ("active",))] == [cid_a]
    assert [r["claim_id"] for r in claims.list_current(sid_b, ("active",))] == [cid_b]


def test_readable_scope_ids_apply_can_read(store: Store) -> None:
    """Other partitions stay invisible under the §9 predicates."""
    claims = ClaimsRepo(store)
    conv_a = make_scope(conversation_id="A", visibility=Visibility.CONVERSATION)
    conv_b = make_scope(conversation_id="B", visibility=Visibility.CONVERSATION)
    owner = make_scope(conversation_id=None, visibility=Visibility.OWNER)
    with store.tx() as conn:
        sid_a = ensure_scope(store, conn, conv_a)
        sid_b = ensure_scope(store, conn, conv_b)
        sid_owner = ensure_scope(store, conn, owner)

    with store.read() as conn:
        reader_a = make_scope(conversation_id="A")
        visible = set(readable_scope_ids(conn, reader_a))
        assert sid_a in visible
        assert sid_b not in visible          # different conversation
        assert sid_owner in visible          # owner vis, same principal

        # a different principal in conversation A sees A but not the owner partition
        stranger = make_scope(principal_id="mallory", conversation_id="A")
        visible_m = set(readable_scope_ids(conn, stranger))
        assert sid_a in visible_m            # conversation membership
        assert sid_b not in visible_m
        assert sid_owner not in visible_m    # missing principal never widens


# ----------------------------------------------------------------------
# events
# ----------------------------------------------------------------------


def test_event_appends_monotonic(store: Store, scope_id) -> None:
    events = EventsRepo(store)
    with store.tx() as conn:
        seqs = [
            events.append(conn, scope_id, "kind", "alice", {"i": i}, "pol-1")
            for i in range(3)
        ]
    assert seqs == sorted(seqs) == list(range(1, 4))
    with store.read() as conn:
        recorded = [
            r[0]
            for r in conn.execute(
                "SELECT recorded_us FROM events ORDER BY event_seq"
            ).fetchall()
        ]
    assert recorded == sorted(recorded)
    assert len(set(recorded)) == 3  # strictly increasing logical clock
    row = events.get(seqs[0])
    assert row["payload"] == {"i": 0}
    assert row["observed_wall_us"] > 0


def test_event_recorded_us_survives_clock_rollback(
    store: Store, scope_id, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = EventsRepo(store)
    with store.tx() as conn:
        events.append(conn, scope_id, "k", "a", {}, "p")
    monkeypatch.setattr("verbatim.storage.store.now_us", lambda: 1)
    with store.tx() as conn:
        events.append(conn, scope_id, "k", "a", {}, "p")
    with store.read() as conn:
        rec = [
            r[0]
            for r in conn.execute(
                "SELECT recorded_us FROM events ORDER BY event_seq"
            ).fetchall()
        ]
    assert rec[1] > rec[0]


# ----------------------------------------------------------------------
# entities, edges
# ----------------------------------------------------------------------


def test_entities_case_normalized_scoped_match(store: Store) -> None:
    ents = EntitiesRepo(store)
    claims = ClaimsRepo(store)
    scope_a = make_scope(conversation_id="A")
    scope_b = make_scope(conversation_id="B")
    with store.tx() as conn:
        sid_a = ensure_scope(store, conn, scope_a)
        sid_b = ensure_scope(store, conn, scope_b)
        e1 = ents.find_or_create(sid_a, "  Sam  ", conn=conn)
        e2 = ents.find_or_create(sid_a, "sam", conn=conn)
        e3 = ents.find_or_create(sid_b, "sam", conn=conn)
        assert e1 == e2            # normalized match within scope
        assert e3 != e1            # same label, different partition
        ents.add_alias(e1, "sam", None, conn)

        cid = claims.create(sid_a, None, None, conn)
        claims.add_revision(cid, "active", None, "affirmative", "asserted",
                            None, None, [], [], 1, conn)
        ents.link_claim(conn, cid, e1, role="subject")
        assert ents.claims_for_entities(sid_a, [e1], conn) == [cid]
        # the join never crosses partitions
        assert ents.claims_for_entities(sid_b, [e1], conn) == []
        assert ents.claims_for_entities(sid_a, [], conn) == []


def test_edges_canonical_conflict_and_retire(store: Store, scope_id) -> None:
    edges = EdgesRepo(store)
    with store.tx() as conn:
        fwd = edges.add(conn, scope_id, "claim", "A", "claim", "B", "conflicts_with")
        rev = edges.add(conn, scope_id, "claim", "B", "claim", "A", "conflicts_with")
        assert fwd == rev  # canonicalized: stored once
        same = edges.add(conn, scope_id, "claim", "A", "claim", "B", "conflicts_with")
        assert same == fwd  # idempotent while unretired
        neigh = edges.neighbors(scope_id, "claim", "A", ["conflicts_with"], conn)
        assert [n["edge_id"] for n in neigh] == [fwd]
        neigh_b = edges.neighbors(scope_id, "claim", "B", ["conflicts_with"], conn)
        assert [n["edge_id"] for n in neigh_b] == [fwd]
        with pytest.raises(VerbatimError) as exc:
            edges.add(conn, scope_id, "claim", "A", "claim", "A", "supports")
        assert exc.value.code == ErrorCode.VALIDATION
        edges.retire(fwd, 42, conn)
        assert edges.neighbors(scope_id, "claim", "A", ["conflicts_with"], conn) == []
        with pytest.raises(VerbatimError) as exc:
            edges.retire(fwd, 43, conn)
        assert exc.value.code == ErrorCode.STALE_PROPOSAL


# ----------------------------------------------------------------------
# decisions, reviews, feedback, consents, purges
# ----------------------------------------------------------------------


def test_decision_record_and_inputs(store: Store, scope_id) -> None:
    decisions = DecisionsRepo(store)
    with store.tx() as conn:
        did = decisions.record(
            conn, scope_id, "pair_relation", store.hmac(b"request"), "rules",
            None, "rubric-1", 0, {"label": "incompatible"},
            [("claim", "c1", 2, store.hmac(b"c1v2")), ("span", "s1", None, None)],
        )
    row = decisions.get(did)
    assert row["backend"] == "rules"
    assert row["result"] == {"label": "incompatible"}
    with store.read() as conn:
        inputs = conn.execute(
            "SELECT object_kind, object_id, revision FROM decision_inputs"
            " WHERE decision_id = ? ORDER BY object_id",
            (did,),
        ).fetchall()
    assert inputs == [("claim", "c1", 2), ("span", "s1", None)]


def test_reviews_open_and_resolve(store: Store, scope_id) -> None:
    reviews = ReviewsRepo(store)
    with store.tx() as conn:
        rid = reviews.create(
            conn, scope_id, {"effect": "supersede", "target": "c1"},
            {"c1": 2}, decision_id=None,
        )
    assert reviews.get(rid)["state"] == "open"
    assert [r["review_id"] for r in reviews.list_open(scope_id)] == [rid]
    with store.tx() as conn:
        reviews.resolve(conn, rid, "approved", 9)
    assert reviews.get(rid)["state"] == "approved"
    assert reviews.get(rid)["resolved_event"] == 9
    assert reviews.list_open(scope_id) == []
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            reviews.resolve(conn, rid, "rejected", 10)
        assert exc.value.code == ErrorCode.STALE_PROPOSAL


def test_reviews_create_dedup_returns_open_equivalent(store: Store, scope_id) -> None:
    """A redelivered proposal resolves to the queued open review instead of
    minting a duplicate — equivalence is effect kind + claim pair."""
    reviews = ReviewsRepo(store)
    effect = {"effect": "supersede", "predecessor_id": "c1", "successor_id": "c2"}
    with store.tx() as conn:
        first = reviews.create(
            conn, scope_id, effect, {"c1": 1, "c2": 1}, dedup=True
        )
        second = reviews.create(
            conn, scope_id, effect, {"c1": 1, "c2": 1}, dedup=True
        )
        # direction matters: the converse proposal is different work
        third = reviews.create(
            conn,
            scope_id,
            {"effect": "supersede", "predecessor_id": "c2", "successor_id": "c1"},
            {"c1": 1, "c2": 1},
            dedup=True,
        )
        # without ``dedup`` the append-always contract is unchanged
        fourth = reviews.create(conn, scope_id, effect, {"c1": 1, "c2": 1})
    assert second == first
    assert third != first
    assert fourth != first
    assert len(reviews.list_open(scope_id)) == 3


def test_reviews_create_dedup_field_shape_equivalent(store: Store, scope_id) -> None:
    """Producers name the counterparty under different keys
    (``counterparty_id`` vs ``conflict_with_claim_id``) — the pair set is
    the dedup identity, not the payload's field shape."""
    reviews = ReviewsRepo(store)
    with store.tx() as conn:
        a = reviews.create(
            conn,
            scope_id,
            {"effect": "dispute", "claim_id": "c1", "counterparty_id": "c2"},
            {"c1": 1, "c2": 1},
            dedup=True,
        )
        b = reviews.create(
            conn,
            scope_id,
            {"effect": "dispute", "claim_id": "c2", "conflict_with_claim_id": "c1"},
            {"c1": 1, "c2": 1},
            dedup=True,
        )
    assert b == a
    assert len(reviews.list_open(scope_id)) == 1


def test_reviews_create_dedup_resolved_allows_reproposal(store: Store, scope_id) -> None:
    """Only *open* reviews dedup — a decided pair may be re-proposed."""
    reviews = ReviewsRepo(store)
    effect = {"effect": "dispute", "claim_id": "c1", "counterparty_id": "c2"}
    with store.tx() as conn:
        first = reviews.create(
            conn, scope_id, effect, {"c1": 1, "c2": 1}, dedup=True
        )
        reviews.resolve(conn, first, "rejected", 9)
        second = reviews.create(
            conn, scope_id, effect, {"c1": 1, "c2": 1}, dedup=True
        )
    assert second != first
    assert [r["review_id"] for r in reviews.list_open(scope_id)] == [second]


def test_feedback_counts(store: Store, scope_id, claim_setup) -> None:
    fb = FeedbackRepo(store)
    cid = claim_setup["claim_id"]
    with store.tx() as conn:
        fb.add(conn, cid, "alice", "helpful")
        fb.add(conn, cid, "alice", "helpful")
        fb.add(conn, cid, "bob", "possibly_wrong", event_id="ev-1")
    counts = fb.counts(cid)
    assert counts["helpful"] == 2
    assert counts["possibly_wrong"] == 1
    assert counts["irrelevant"] == 0
    assert counts["total"] == 3


def test_consent_grant_revoke_active(store: Store, scope_id) -> None:
    consents = ConsentsRepo(store)
    assert consents.active(scope_id, "jev", "remote_decision") is None
    with store.tx() as conn:
        cid = consents.grant(conn, scope_id, "jev", "remote_decision", "digest-1")
    active = consents.active(scope_id, "jev", "remote_decision")
    assert active["consent_id"] == cid
    assert active["policy_digest"] == "digest-1"
    with store.tx() as conn:
        consents.revoke(conn, cid)
    assert consents.active(scope_id, "jev", "remote_decision") is None
    # other processor/purpose combos stay unaffected
    assert consents.active(scope_id, "other", "remote_decision") is None


def test_purge_suppression_lifecycle(store: Store, scope_id) -> None:
    purges = PurgesRepo(store)
    targets = [("source", "src-1"), ("span", "sp-1")]
    with store.tx() as conn:
        pid = purges.create_preview(conn, scope_id, targets, store.hmac(b"sel"))
    # previewed does not suppress yet
    assert purges.suppressed_ids("source", ["src-1"]) == set()
    with store.tx() as conn:
        purges.confirm_suppress(conn, pid)
    assert purges.suppressed_ids("source", ["src-1", "src-2"]) == {"src-1"}
    assert purges.suppressed_ids("span", ["sp-1"]) == {"sp-1"}
    assert purges.suppressed_ids("claim", ["sp-1"]) == set()
    with store.tx() as conn:
        purges.complete(conn, pid)
    # completed purges remain tombstones — suppression is permanent
    assert purges.suppressed_ids("source", ["src-1"]) == {"src-1"}
    row = purges.get(pid)
    assert row["state"] == "completed"
    assert sorted((t["object_kind"], t["object_id"]) for t in row["targets"]) == [
        ("source", "src-1"), ("span", "sp-1"),
    ]


def test_purge_confirm_twice_is_stale(store: Store, scope_id) -> None:
    purges = PurgesRepo(store)
    with store.tx() as conn:
        pid = purges.create_preview(conn, scope_id, [("source", "s")], b"d")
        purges.confirm_suppress(conn, pid)
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            purges.confirm_suppress(conn, pid)
        assert exc.value.code == ErrorCode.STALE_PROPOSAL


# ----------------------------------------------------------------------
# lexical index
# ----------------------------------------------------------------------


def test_fts_index_search_and_generation(store: Store) -> None:
    if not store.fts_enabled:
        pytest.skip("fts5 unavailable")
    fts = FtsRepo(store)
    scope_a = make_scope(conversation_id="A")
    scope_b = make_scope(conversation_id="B")
    with store.tx() as conn:
        sid_a = ensure_scope(store, conn, scope_a)
        sid_b = ensure_scope(store, conn, scope_b)
        fts.index(conn, "c1", 1, sid_a, 1, "neovim editor preference personal projects")
        fts.index(conn, "c2", 1, sid_a, 1, "postgres database migration schedule")
        fts.index(conn, "c3", 1, sid_b, 1, "neovim from another partition")
    hits = fts.search([sid_a], "neovim", 1)
    assert [(c, r) for c, r, _ in hits] == [("c1", 1)]
    # other scope's rows cannot leak into a different partition's results
    hits_all = fts.search([sid_a, sid_b], "neovim", 1)
    assert {c for c, _, _ in hits_all} == {"c1", "c3"}
    # unknown generation finds nothing — projections are generation-scoped
    assert fts.search([sid_a], "neovim", 2) == []
    assert fts.search([], "neovim", 1) == []
    assert fts.verify(1)["ok"] is True
    with store.tx() as conn:
        removed = fts.delete_for_generation(conn, 1)
    assert removed == 3
    assert fts.search([sid_a], "neovim", 1) == []
    assert fts.verify(1)["fts_rows"] == 0


def test_fts_reindex_replaces_row(store: Store) -> None:
    if not store.fts_enabled:
        pytest.skip("fts5 unavailable")
    fts = FtsRepo(store)
    with store.tx() as conn:
        sid = ensure_scope(store, conn, make_scope())
        fts.index(conn, "c1", 1, sid, 1, "first wording")
        fts.index(conn, "c1", 1, sid, 1, "second wording")
    hits = fts.search([sid], "second", 1)
    assert [(c, r) for c, r, _ in hits] == [("c1", 1)]
    assert fts.search([sid], "first", 1) == []
    v = fts.verify(1)
    assert v["ok"] and v["fts_rows"] == 1


# ----------------------------------------------------------------------
# transaction atomicity
# ----------------------------------------------------------------------


def test_tx_rolls_back_on_error(store: Store, scope) -> None:
    repo = SourcesRepo(store)
    env = make_envelope(scope, payload=b"atomic", external_id="tx-1")
    repo.insert(env)
    with pytest.raises(RuntimeError):
        with store.tx() as conn:
            conn.execute(
                "UPDATE sources SET origin = 'tampered' WHERE external_id = 'tx-1'"
            )
            raise RuntimeError("boom")
    # the failed tx left nothing behind
    with store.read() as conn:
        origin = conn.execute(
            "SELECT origin FROM sources WHERE external_id = 'tx-1'"
        ).fetchone()[0]
    assert origin == "test-harness"


# ----------------------------------------------------------------------
# source views: derived-bytes integrity digest re-verified on read
# ----------------------------------------------------------------------


def _seed_view(store: Store, scope, derived: bytes) -> str:
    from verbatim.storage.repos_v2 import SourceViewsRepo

    source_id, _ = SourcesRepo(store).insert(
        make_envelope(scope, payload=b"canonical")
    )
    views = SourceViewsRepo(store)
    with store.tx() as conn:
        views.ensure_primary(conn, source_id, 1)
        views.insert_derived(
            conn,
            source_id,
            1,
            "norm-1",
            media_type="text/plain",
            view_kind="normalized",
            transformer_revision="norm-v1",
            derived_bytes=derived,
        )
    return source_id


def test_source_view_derived_integrity_verified(store: Store, scope) -> None:
    """``get``/``for_revision`` re-check derived bytes against
    ``integrity_digest`` — tampered rows raise, never serve."""
    from verbatim.storage.repos_v2 import SourceViewsRepo

    source_id = _seed_view(store, scope, b"canonical")
    views = SourceViewsRepo(store)
    row = views.get(source_id, 1, "norm-1")
    assert row["derived_bytes"] == b"canonical"
    assert len(views.for_revision(source_id, 1)) == 2  # primary + derived
    with store.tx() as conn:
        conn.execute(
            "UPDATE source_views SET derived_bytes = ?"
            " WHERE source_id = ? AND revision = 1 AND view_id = 'norm-1'",
            (b"c4nonical", source_id),
        )
    with pytest.raises(VerbatimError) as exc:
        views.get(source_id, 1, "norm-1")
    assert exc.value.code == ErrorCode.STORE_CORRUPT
    with pytest.raises(VerbatimError) as exc:
        views.for_revision(source_id, 1)
    assert exc.value.code == ErrorCode.STORE_CORRUPT


def test_source_view_null_digest_is_legacy_unverified(store: Store, scope) -> None:
    """NULL ``integrity_digest`` marks a byte-less or legacy row — primary
    views carry no derived bytes at all, so they always read unverified."""
    from verbatim.storage.repos_v2 import SourceViewsRepo

    source_id, _ = SourcesRepo(store).insert(
        make_envelope(scope, payload=b"canonical")
    )
    views = SourceViewsRepo(store)
    with store.tx() as conn:
        views.ensure_primary(conn, source_id, 1)
    row = views.get(source_id, 1, "primary")
    assert row["view_kind"] == "primary"
    assert row["derived_bytes"] is None
    assert [r["view_id"] for r in views.for_revision(source_id, 1)] == ["primary"]
