"""V3 lane regressions for SPEC_V4 findings F4-10 and F4-13.

F4-10 / C31 (V4-28.02): eligibility precedes bounded rank selection — a
fetch window full of ineligible lexical matches must page through them
until the eligible bound, corpus exhaustion, or the deadline; >160 held
matches cannot hide the next eligible claim.

F4-13 (V4-27.07): lanes run sequentially on one read snapshot and report
``concurrency="sequential"`` truthfully on every LaneResult and on the
returned LaneRunReport.

V4-27.09: a deadline that stops a lane mid-corpus reports ``partial``
coverage — never silent truncation.
"""

from __future__ import annotations

import hashlib
import hmac

import pytest

from verbatim.core.types_v3 import (
    QueryClass,
    RecallRequestV3,
)
from verbatim.governance import (
    create_grant,
    register_principal,
    seed_purposes,
)
from verbatim.retrieval.candidates import Deadline
from verbatim.retrieval.query import analyze
from verbatim.retrieval.v3 import lanes as _lanes
from verbatim.retrieval.v3 import recall_v3
from verbatim.storage.store import Store


# Same pinned-key convention as test_retrieval_v3: seeded digests must
# verify against store.hmac() on reads.
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


def seed_scope(conn, scope_id, principal="p1", conv="c1", profile="prof",
               vis="conversation"):
    conn.execute(
        "INSERT INTO scopes(scope_id,profile_id,principal_id,workspace_id,"
        "conversation_id,visibility,acl_revision) VALUES(?,?,?,?,?,?,0)",
        (scope_id, profile, principal, "ws", conv, vis),
    )


def seed_auth(conn, scope_id, pid="human:alice", purposes=("recall",)):
    seed_purposes(conn)
    register_principal(conn, kind="human", principal_id=pid)
    create_grant(
        conn, scope_id=scope_id, principal_id=pid, verbs={"read"},
        issuer_id=pid, purposes=list(purposes),
    )


def add_source(conn, source_id, scope_id, payload: bytes, speaker="u1"):
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
               state="active", recorded_from=1, recorded_until=None, rev=1):
    payload = text.encode("utf-8")
    add_source(conn, source_id, scope_id, payload)
    add_span(conn, span_id, source_id, 0, len(payload))
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,NULL,NULL,?,1)",
        (claim_id, scope_id, recorded_from),
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
        (claim_id, rev, span_id),
    )
    add_fts(conn, claim_id, rev, scope_id, text, gen)


def hold_claim(conn, claim_id, scope_id="sA", rev=1, state="pending"):
    conn.execute(
        "INSERT INTO quarantine(object_kind,object_id,revision,"
        "scope_id,state,opened_event) VALUES('claim',?,?,?,?,1)",
        (claim_id, rev, scope_id, state),
    )


def request(query="term", scope_id="sA", caller="human:alice", **kw):
    kw.setdefault("purpose", "recall")
    return RecallRequestV3(
        query=query, scope_id=scope_id, caller_id=caller, **kw
    )


def lane_ctx(store, conn, query, lane_cap=40, deadline=None,
             scope_ids=("sA",), query_class=QueryClass.CURRENT_STATE):
    req = request(query)
    return _lanes.LaneContext(
        conn=conn, store=store,
        request=req,
        plan=analyze(query, req, 1),
        query_class=query_class,
        scope_ids=scope_ids,
        generation=_gen(store),
        deadline=deadline if deadline is not None else Deadline(None),
        lane_cap=lane_cap,
    )


class FakeDeadline(Deadline):
    """Deterministic expiry after ``checks`` calls to ``expired``."""

    def __init__(self, checks: int):
        super().__init__(None)
        self._end = 1.0  # armed; real clock value irrelevant
        self._left = checks

    def expired(self) -> bool:
        if self._left <= 0:
            self.exceeded = True
            return True
        self._left -= 1
        return False


def item_ids(result):
    return {i.handle.object_id for p in result.packs for i in p.items}


# ---------------------------------------------------------------------
# C31 / V4-28.02: >160 held lexical matches cannot hide the eligible one
# ---------------------------------------------------------------------

_HELD_N = 170  # > the old fixed 160-row oversample window


def _seed_held_corpus(conn, store, held_n=_HELD_N, term="guardterm"):
    """``held_n`` quarantined matches rank ahead of one eligible match.

    Held texts are deliberately short (best bm25 for the same term) and
    the eligible claim's text is padded so it ranks strictly last under
    both FTS5 orderings (``bm25(), row_id`` and bare ``ORDER BY rank``).
    """
    seed_scope(conn, "sA")
    seed_auth(conn, "sA")
    gen = _gen(store)
    for i in range(held_n):
        cid = f"held{i:04d}"
        seed_claim(conn, cid, "sA", f"srcH{i}", f"spH{i}",
                   f"{term} held variant {i}", gen)
        hold_claim(conn, cid)
    padded = f"{term} " + " ".join(f"filler{i}" for i in range(64))
    seed_claim(conn, "clGood", "sA", "srcG", "spG", padded, gen)
    return gen


def test_held_matches_do_not_hide_eligible_lane(store):
    """The eligible claim at lexical rank >160 still surfaces (V4-28.02).

    F4-11 note: the candidate stream is now eligibility-first — the local
    BM25 corpus excludes held rows before scoring, so they never consume
    fetch budget either. ``attempted`` therefore counts only eligible
    candidates; the stronger invariant is that the held tail cannot hide
    or reorder the eligible hit at all.
    """
    with store.tx() as conn:
        _seed_held_corpus(conn, store)
    with store.read() as conn:
        res = _lanes.lane_lexical(lane_ctx(store, conn, "guardterm"))
    assert res.status == "ok"
    assert res.hits.get(("claim", "clGood")) == 1
    # eligibility-first: the 170 held matches never enter the stream
    assert res.attempted == 1
    assert not any(k[1].startswith("held") for k in res.hits)


def test_held_matches_do_not_hide_eligible_recall(store):
    """End-to-end: the held matches never reach a pack; the eligible does."""
    with store.tx() as conn:
        _seed_held_corpus(conn, store)
    res = recall_v3(store, request("guardterm"))
    ids = item_ids(res)
    assert "clGood" in ids
    assert not any(i.startswith("held") for i in ids)


def test_eligible_bound_stops_paging(store):
    """Paging stops at the eligible bound — it does not drain the corpus."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        # Two short-text eligible claims rank strictly best; 90 held,
        # longer-text matches rank after — a single page suffices.
        seed_claim(conn, "clE1", "sA", "srcE1", "spE1", "guardterm a", gen)
        seed_claim(conn, "clE2", "sA", "srcE2", "spE2",
                   "guardterm alpha beta", gen)
        for i in range(90):
            cid = f"held{i:03d}"
            seed_claim(conn, cid, "sA", f"srcH{i}", f"spH{i}",
                       "guardterm " + "pad " * 40 + f"variant {i}", gen)
            hold_claim(conn, cid)
    with store.read() as conn:
        res = _lanes.lane_lexical(lane_ctx(store, conn, "guardterm",
                                           lane_cap=2))
    assert res.hits == {("claim", "clE1"): 1, ("claim", "clE2"): 2}
    # Eligibility-first candidate stream (F4-11): the two eligible claims
    # are the entire produced set — held rows are filtered before ranking
    # and no fetch window is needed at all.
    assert res.attempted == 2


def test_ineligible_rows_never_appear(store):
    """Metamorphic: held/rejected/superseded/out-of-scope matches never
    occupy lane hits or delivered items, however many there are."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_scope(conn, "sB", principal="p2", conv="c2")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clA1", "sA", "srcA1", "spA1", "guardterm a", gen)
        seed_claim(conn, "clA2", "sA", "srcA2", "spA2",
                   "guardterm alpha beta", gen)
        seed_claim(conn, "clA3", "sA", "srcA3", "spA3",
                   "guardterm alpha beta gamma", gen)
        # quarantined + wrong-lifecycle + out-of-scope matches
        for i in range(30):
            cid = f"held{i:03d}"
            seed_claim(conn, cid, "sA", f"srcH{i}", f"spH{i}",
                       f"guardterm heldvariant {i} " + "pad " * 30, gen)
            hold_claim(conn, cid)
        for i in range(10):
            seed_claim(conn, f"rej{i}", "sA", f"srcR{i}", f"spR{i}",
                       f"guardterm rejectedvariant {i} " + "pad " * 30,
                       gen, state="rejected")
        for i in range(10):
            seed_claim(conn, f"sup{i}", "sA", f"srcS{i}", f"spS{i}",
                       f"guardterm supersededvariant {i} " + "pad " * 30,
                       gen, state="superseded")
        seed_claim(conn, "clB", "sB", "srcB", "spB",
                   "guardterm other scope evidence", gen)
    with store.read() as conn:
        res = _lanes.lane_lexical(lane_ctx(store, conn, "guardterm"))
    # the three eligible claims surface, best-first, nothing else does
    assert set(res.hits) == {
        ("claim", "clA1"), ("claim", "clA2"), ("claim", "clA3")
    }
    assert res.hits[("claim", "clA1")] == 1
    res_r = recall_v3(store, request("guardterm"))
    ids = item_ids(res_r)
    assert ids <= {"clA1", "clA2", "clA3"}
    assert "clB" not in ids


def test_eligible_overflow_still_counted(store):
    """Eligible results beyond the cap remain honest overflow (V3-28.03)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        for i in range(3):
            seed_claim(conn, f"clE{i}", "sA", f"srcE{i}", f"spE{i}",
                       f"guardterm {'x ' * (i + 1)}{i}", gen)
        for i in range(6):
            cid = f"held{i}"
            seed_claim(conn, cid, "sA", f"srcH{i}", f"spH{i}",
                       "guardterm " + "pad " * 40 + str(i), gen)
            hold_claim(conn, cid)
    with store.read() as conn:
        res = _lanes.lane_lexical(lane_ctx(store, conn, "guardterm",
                                           lane_cap=1))
    assert len(res.hits) == 1
    assert res.overflow == 2  # the other two eligible matches
    assert not any(k[1].startswith("held") for k in res.hits)


# ---------------------------------------------------------------------
# V4-27.09: deadline-honest coverage
# ---------------------------------------------------------------------

def test_deadline_mid_scan_reports_partial(store):
    """A deadline stopping the scan mid-corpus reports incomplete coverage —
    the lane cannot silently truncate and claim ``ok``.

    The request deadline is propagated into the candidate fetch's
    corpus-statistics/scoring pass, so a pre-expired deadline cuts the
    very first window short of the 700-match corpus — the lane must
    report ``partial`` (V4-27.09) rather than a healthy empty result.
    """
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        for i in range(700):
            seed_claim(conn, f"clE{i:04d}", "sA", f"srcE{i}", f"spE{i}",
                       f"guardterm variant {i}", gen)
    with store.read() as conn:
        res = _lanes.lane_lexical(
            lane_ctx(store, conn, "guardterm", lane_cap=500,
                     deadline=FakeDeadline(0))
        )
    assert res.status == "partial"
    assert res.reason == "deadline"
    assert "scan_incomplete:deadline" in res.warnings
    assert res.attempted < 700  # provably short of the whole corpus


def test_deadline_not_reached_reports_complete(store):
    """Same corpus, one more deadline check: paging finishes the corpus
    and reports healthy completed coverage."""
    with store.tx() as conn:
        _seed_held_corpus(conn, store, held_n=350)
    # Budget covers the cooperative checks the stats pass now performs
    # (corpus stats, doc lengths, result text fetch) plus lane admission;
    # the point is completion under a live-but-sufficient deadline.
    with store.read() as conn:
        res = _lanes.lane_lexical(
            lane_ctx(store, conn, "guardterm", deadline=FakeDeadline(8))
        )
    assert res.status == "ok"
    assert res.hits.get(("claim", "clGood")) == 1


def test_browse_lane_pages_past_held(store):
    """Sibling lane with the same fixed-window pattern: the bounded browse
    listing pages through held rows to the eligible tail (V4-28.02)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        # held claims carry the newest created_event → they head the
        # DESC listing; the eligible claim sorts last.
        for i in range(90):
            cid = f"held{i:03d}"
            seed_claim(conn, cid, "sA", f"srcH{i}", f"spH{i}",
                       f"browse held {i}", gen, recorded_from=1000 + i)
            hold_claim(conn, cid)
        seed_claim(conn, "clTail", "sA", "srcT", "spT",
                   "browse eligible tail", gen, recorded_from=1)
    with store.read() as conn:
        res = _lanes.lane_browse(lane_ctx(store, conn, "ignored",
                                          lane_cap=20))
    assert ("claim", "clTail") in res.hits
    assert res.attempted > 80
    assert not any(k[1].startswith("held") for k in res.hits)


# ---------------------------------------------------------------------
# V4-27.07 / F4-13: truthful sequential execution reporting
# ---------------------------------------------------------------------

def test_run_lanes_reports_sequential(store):
    """run_lanes is serial and says so — on the report and every result."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "sequential reporting probe", _gen(store))
    with store.read() as conn:
        ctx = lane_ctx(store, conn, "sequential reporting probe")
        report = _lanes.run_lanes(ctx, ["lexical", "browse", "structured"])
    assert isinstance(report, _lanes.LaneRunReport)
    assert report.concurrency == "sequential"
    assert [r.lane for r in report] == ["lexical", "structured", "browse"]
    assert all(r.concurrency == "sequential" for r in report)
    assert report[0].hits.get(("claim", "cl1")) == 1


def test_run_lanes_deadline_skip_records_reason(store):
    """A pre-expired deadline skips later lanes with a recorded reason —
    never a healthy 'ok' for a lane that never ran (V4-27.09)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "deadline skip probe", _gen(store))
    with store.read() as conn:
        ctx = lane_ctx(store, conn, "deadline skip probe",
                       deadline=FakeDeadline(1))
        report = _lanes.run_lanes(ctx, ["exact_id", "lexical", "browse"])
    by_lane = {r.lane: r for r in report}
    # first expired() call lets exact_id run; the rest are cut
    assert by_lane["exact_id"].status == "ok"
    assert by_lane["lexical"].status == "skipped"
    assert by_lane["lexical"].reason == "deadline"
    assert by_lane["browse"].status == "skipped"
    assert by_lane["browse"].reason == "deadline"
    assert all(r.concurrency == "sequential" for r in report)


def test_recall_expired_deadline_reports_skipped_lanes(store):
    """Surface check: a blown request budget reaches lane diagnostics
    honestly — every selected lane reports ``skipped`` (V4-27.09)."""
    import verbatim.retrieval.candidates as cand

    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "zero deadline probe", _gen(store))
    real = cand.Deadline
    try:
        cand.Deadline = lambda _ms: FakeDeadline(0)
        res = recall_v3(store, request("zero deadline probe",
                                       deadline_ms=20))
    finally:
        cand.Deadline = real
    lane_status = res.capabilities["lanes"]
    assert lane_status  # lanes were selected and diagnosed
    assert all(s == "skipped" for s in lane_status.values())
