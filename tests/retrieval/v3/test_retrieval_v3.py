"""V3 retrieval/controller tests (SPEC_V3 §26–§32, §46).

Real ``Store.create`` fixtures (v3 schema) with direct SQL + governance
seeding. Covers: eligibility before lane truncation, cross-scope
authorization isolation, quarantine/suppression, known-at cutoff,
temporal filtering, hard-identifier abstention, group support verdicts,
dependency preservation, candidate cap, optional-lane degradation,
typed packs + shared budgets, atomic group omission, influence
handles/feedback/blast radius, deterministic routing + learned shadow,
decision logs, and sparse/late handler capability errors.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3

import pytest

from verbatim.core.types import ErrorCode, JobKind, VerbatimError
from verbatim.core.types_v3 import (
    ActionIntent,
    BudgetTier,
    GroupSupport,
    InfluenceFeedback,
    MemoryKindV3,
    PackKind,
    Perspective,
    QueryClass,
    RecallRequestV3,
    ContextRequestV3,
    Route,
    TaskContext,
)
from verbatim.governance import (
    CallerV3,
    create_grant,
    register_principal,
    seed_purposes,
)
from verbatim.retrieval.v3 import (
    blast_radius,
    context_v3,
    extract_hard_identifiers,
    plan_routes,
    recall_v3,
    record_feedback,
)
from verbatim.retrieval.query import analyze
from verbatim.retrieval.v3 import lanes as _lanes
from verbatim.retrieval.v3 import union as _union
from verbatim.retrieval.v3 import abstain as _abstain
from verbatim.retrieval.v3 import controller as _ctrl
from verbatim.retrieval.v3 import handlers as _handlers
from verbatim.retrieval.v3 import influence as _infl
from verbatim.storage.store import Store


# Content digests are verified on reads (SpansRepo.text, SourcesRepo.payload,
# Store._verify_content_digests), so seeders must write real profile-keyed
# HMACs, not placeholder bytes. The store fixture pins the profile key to
# this value (exactly 32 bytes, as Store key files require) so the conn-only
# seed helpers below produce digests store.hmac() re-verifies.
_TEST_HMAC_KEY = b"test-hmac-key-padded-to-32-bytes"


def _h(data: bytes) -> bytes:
    return hmac.new(_TEST_HMAC_KEY, data, hashlib.sha256).digest()


@pytest.fixture
def store(tmp_path):
    # Pin the profile HMAC key: Store.create loads <db>.key when it exists,
    # making store.hmac(data) identical to _h(data) for seeded digests.
    (tmp_path / "v3.db.key").write_bytes(_TEST_HMAC_KEY)
    s = Store.create(str(tmp_path / "v3.db"))
    yield s
    s.close()


GEN = None  # resolved per-test from store.projection_generation()


def _gen(store) -> int:
    return store.projection_generation()


# ---------------------------------------------------------------------
# seeding helpers
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
    """Principal + grant for the governance authorization path.

    Compat note (F4-02 / V4-08.02): pre-v4 revisions of this suite
    asserted exact quotation bytes under a ``read``-only grant — the
    unsafe contract the finding closed. Exact bytes now require
    ``quote``, so the default grant carries ``{"read", "quote"}`` and
    the byte-level assertions keep exercising the real quotation path.
    Read-only and purpose-narrowed grants are covered explicitly by the
    F4-01/F4-02 regressions in ``test_f4_authorization.py``.
    """
    seed_purposes(conn)
    register_principal(conn, kind="human", principal_id=pid)
    create_grant(
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
               state="active", recorded_from=1, recorded_until=None,
               rev=1, condition=None, intervals=(), family_id=None,
               perspective_id=None, freshness=None):
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
        " VALUES(?,?,?,?,?,?,?,?)",
        (claim_id, rev, state, condition, recorded_from, recorded_until,
         perspective_id, freshness),
    )
    conn.execute(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role,family_id) VALUES(?,?,?,'primary',?)",
        (claim_id, rev, span_id, family_id),
    )
    for i, (f_us, u_us) in enumerate(intervals):
        conn.execute(
            "INSERT INTO valid_intervals(claim_id,revision,interval_no,"
            "from_us,until_us,precision,basis) VALUES(?,?,?,?,?,'day','test')",
            (claim_id, rev, i, f_us, u_us),
        )
    add_fts(conn, claim_id, rev, scope_id, text, gen)


def request(query="term", scope_id="sA", caller="human:alice", **kw):
    kw.setdefault("purpose", "recall")
    return RecallRequestV3(
        query=query, scope_id=scope_id, caller_id=caller, **kw
    )


def items_of(result, kind=None):
    out = []
    for p in result.packs:
        if kind is None or p.kind == kind:
            out.extend(p.items)
    return out


def texts_of(result):
    return [i.text for p in result.packs for i in p.items]


# ---------------------------------------------------------------------
# §26 deterministic routing
# ---------------------------------------------------------------------

def test_route_current_state(store):
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "the release ships on Friday", _gen(store))
    res = recall_v3(store, request("when does the release ship"))
    assert res.decision_id
    row = None
    with store.read() as conn:
        row = conn.execute(
            "SELECT routes_json, lane_set_json, policy_revision,"
            " state_key FROM routing_decisions WHERE decision_id = ?",
            (res.decision_id,),
        ).fetchone()
    assert row is not None
    assert Route.CURRENT.value in row[0]
    assert row[2] == _ctrl.POLICY_REVISION


def test_route_procedure_class(store):
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
    with store.read() as conn:
        state = _ctrl.discretize(
            conn,
            request("how do i run the tests", modes=("procedure",)),
            QueryClass.PROCEDURE, ["sA"],
        )
        rs = plan_routes(
            request("how do i run the tests", modes=("procedure",)),
            None, None, state,
        )
    assert Route.PROCEDURE in rs.routes
    assert "procedural_signature" in rs.lanes


def test_route_stuck_adds_failure(store):
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
    task = TaskContext(stuck=True, session_phase="stuck")
    with store.read() as conn:
        state = _ctrl.discretize(
            conn, request("still broken", task=task),
            QueryClass.CURRENT_STATE, ["sA"],
        )
        rs = plan_routes(request("still broken", task=task), None, task,
                         state)
    assert Route.FAILURE in rs.routes


def test_route_irreversible_intent_adds_verify(store):
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
    task = TaskContext(action_intent=ActionIntent.IRREVERSIBLE)
    with store.read() as conn:
        state = _ctrl.discretize(
            conn, request("about to rm -rf", task=task),
            QueryClass.CURRENT_STATE, ["sA"],
        )
        rs = plan_routes(request("about to rm -rf", task=task), None,
                         task, state)
    assert Route.VERIFY in rs.routes
    assert "freshness_env" in rs.lanes


def test_learned_shadow_records_not_applies(store, tmp_path):
    """learned_shadow logs a hypothetical row; the deterministic plan is
    still the applied one (V3-26.03)."""
    from verbatim.config import config_from_mapping
    cfg = config_from_mapping(
        {"v3": {"retrieval": {"controller": "learned_shadow"}}}
    )
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "deploy runbook lives in infra.md", _gen(store))
    res = recall_v3(store, request("where is the deploy runbook"), cfg=cfg)
    assert res.decision_id
    with store.read() as conn:
        rows = conn.execute(
            "SELECT policy_revision FROM routing_decisions"
            " WHERE scope_id = 'sA' ORDER BY created_us"
        ).fetchall()
    revisions = {r[0] for r in rows}
    assert _ctrl.POLICY_REVISION in revisions
    assert "learned_shadow_v1" in revisions


# ---------------------------------------------------------------------
# §28 eligibility-first + lanes
# ---------------------------------------------------------------------

def test_quarantined_claim_not_delivered(store):
    """Quarantine is an admission predicate — the quarantined claim is
    invisible even though it textually matches (V3-28.01)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clQ", "sA", "srcQ", "spQ",
                   "secret quarantined token payload", gen)
        seed_claim(conn, "clOK", "sA", "srcK", "spK",
                   "secret visible token payload", gen)
        conn.execute(
            "INSERT INTO quarantine(object_kind,object_id,revision,"
            "scope_id,state,opened_event) VALUES('claim','clQ',1,'sA',"
            "'pending',1)",
        )
    res = recall_v3(store, request("secret token payload"))
    texts = texts_of(res)
    assert any("visible" in t for t in texts)
    assert not any("quarantined" in t for t in texts)


def test_quarantine_opened_between_recalls_is_observed(store):
    """A claim hold committed BETWEEN two recalls must suppress the claim
    on the second recall — per-snapshot admission verdicts are memoized
    inside one ``store.read()`` and must never answer a later snapshot
    (V3-28.01; SnapshotCache freshness regression)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clMid", "sA", "srcM", "spM",
                   "midstream probe phrase alpha", gen)
        seed_claim(conn, "clStay", "sA", "srcT", "spT",
                   "midstream probe phrase beta", gen)
    res1 = recall_v3(store, request("midstream probe phrase"))
    ids1 = {i.handle.object_id for i in items_of(res1)}
    assert "clMid" in ids1 and "clStay" in ids1

    # Hold lands after the first recall's read snapshot closed.
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO quarantine(object_kind,object_id,revision,"
            "scope_id,state,opened_event) VALUES('claim','clMid',1,'sA',"
            "'pending',2)",
        )
    res2 = recall_v3(store, request("midstream probe phrase"))
    ids2 = {i.handle.object_id for i in items_of(res2)}
    assert "clMid" not in ids2
    assert "clStay" in ids2
    assert not any("alpha" in t for t in texts_of(res2))


def test_span_hold_between_recalls_cascades(store):
    """A hold on the evidence SPAN — not the claim — opened mid-sequence
    withholds the dependent claim on the next recall: the quarantine
    cascade re-reads holds under each fresh snapshot, so a memoized
    span verdict can never outlive its snapshot (V3-14.10)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clDep", "sA", "srcD", "spD",
                   "cascade probe phrase gamma", gen)
        seed_claim(conn, "clOk", "sA", "srcO", "spO",
                   "cascade probe phrase delta", gen)
    res1 = recall_v3(store, request("cascade probe phrase"))
    ids1 = {i.handle.object_id for i in items_of(res1)}
    assert "clDep" in ids1 and "clOk" in ids1

    with store.tx() as conn:
        conn.execute(
            "INSERT INTO quarantine(object_kind,object_id,revision,"
            "scope_id,state,opened_event) VALUES('span','spD',1,'sA',"
            "'pending',2)",
        )
    res2 = recall_v3(store, request("cascade probe phrase"))
    ids2 = {i.handle.object_id for i in items_of(res2)}
    assert "clDep" not in ids2
    assert "clOk" in ids2


def test_revoked_read_between_recalls_fails_closed(store):
    """Revoking the requesting scope's own grant mid-sequence makes the
    NEXT recall deny indistinguishably — a cached admission can never
    resurrect authority committed away between snapshots (§10.05)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clR", "sA", "srcR", "spR",
                   "revocation probe phrase", gen)
    res1 = recall_v3(store, request("revocation probe phrase"))
    assert any(
        i.handle.object_id == "clR" for i in items_of(res1)
    )

    with store.tx() as conn:
        # strip every verb: revoke all live grants on the scope so it
        # contributes nothing for this purpose
        from verbatim.governance import revoke_grant
        for (g,) in conn.execute(
            "SELECT grant_id FROM grants_v3 WHERE scope_id='sA'"
            " AND principal_id='human:alice' AND revoked_us IS NULL",
        ).fetchall():
            revoke_grant(conn, g)
    with pytest.raises(VerbatimError) as exc:
        recall_v3(store, request("revocation probe phrase"))
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_eligibility_before_lane_truncation(store):
    """An ineligible candidate must NOT consume lane rank capacity: with
    lane_cap=1 the eligible second candidate still surfaces (V3-28.01)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        # clBad sorts first by claim_id and matches the query…
        seed_claim(conn, "cl1Bad", "sA", "srcB", "spB",
                   "rank capacity probe alpha", gen)
        seed_claim(conn, "cl2Good", "sA", "srcG", "spG",
                   "rank capacity probe alpha", gen)
        # …but it is quarantined, so it cannot occupy rank 1.
        conn.execute(
            "INSERT INTO quarantine(object_kind,object_id,revision,"
            "scope_id,state,opened_event) VALUES('claim','cl1Bad',1,'sA',"
            "'suppressed',1)",
        )
    with store.read() as conn:
        plan = analyze("rank capacity probe alpha",
                       request("rank capacity probe alpha"), 1)
        ctx = _lanes.LaneContext(
            conn=conn, store=store,
            request=request("rank capacity probe alpha"),
            plan=plan, query_class=QueryClass.CURRENT_STATE,
            scope_ids=("sA",), generation=_gen(store),
            deadline=_lanes._cand.Deadline(None), lane_cap=1,
        )
        res = _lanes.lane_lexical(ctx)
    assert res.hits == {("claim", "cl2Good"): 1}
    assert res.overflow == 0


def test_cross_scope_isolation(store):
    """Claims outside the caller's authorized scopes never surface
    (V3-28.04, §10.05)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_scope(conn, "sB", principal="p2", conv="c2")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clA", "sA", "srcA", "spA",
                   "alpha evidence in scope A", gen)
        seed_claim(conn, "clB", "sB", "srcB", "spB",
                   "alpha evidence in scope B", gen)
    res = recall_v3(store, request("alpha evidence"))
    texts = texts_of(res)
    assert any("scope A" in t for t in texts)
    assert not any("scope B" in t for t in texts)


def test_unauthorized_scope_fails_closed(store):
    """No grant → NOT_FOUND_OR_UNAUTHORIZED; existence never leaks."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_purposes(conn)
        register_principal(conn, kind="human", principal_id="human:alice")
    with pytest.raises(VerbatimError) as exc:
        recall_v3(store, request("anything"))
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_span_suppression_hides_claim(store):
    """A claim whose evidence span is purged withholds entirely (§14)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clS", "sA", "srcS", "spS",
                   "suppressed span content", gen)
        conn.execute(
            "INSERT INTO purges(purge_id,selection_digest,scope_id,state,"
            "requested_us) VALUES('pg1',X'00','sA','suppressed',1)"
        )
        conn.execute(
            "INSERT INTO purge_targets(purge_id,object_kind,object_id)"
            " VALUES('pg1','span','spS')",
        )
    res = recall_v3(store, request("suppressed span content"))
    assert not any("suppressed" in t for t in texts_of(res))


def test_known_at_cutoff(store):
    """known_at_seq excludes future-recorded revisions entirely
    (V3-28.11): nothing recorded after K may influence ranking."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clOld", "sA", "srcO", "spO",
                   "known at the cutoff marker", gen, recorded_from=5)
        seed_claim(conn, "clNew", "sA", "srcN", "spN",
                   "known at the cutoff marker", gen, recorded_from=50)
    res = recall_v3(
        store, request("cutoff marker", known_at_seq=10)
    )
    texts = texts_of(res)
    assert any("cutoff marker" in t for t in texts)
    # clNew recorded_from=50 > K=10 — must not appear
    items = items_of(res)
    ids = [i.handle.object_id for i in items]
    assert "clNew" not in ids
    assert "clOld" in ids


def test_temporal_valid_at(store):
    """valid_at_us filters by covering valid interval (V3-28.10)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clT1", "sA", "srcT", "spT",
                   "temporal window claim", gen,
                   intervals=[(100, 200)])
        seed_claim(conn, "clT2", "sA", "srcT2", "spT2",
                   "temporal window claim", gen,
                   intervals=[(900, 1000)])
    res = recall_v3(
        store,
        request("temporal window claim", valid_at_us=150,
                modes=("past_state",)),
    )
    ids = [i.handle.object_id for i in items_of(res)]
    assert "clT1" in ids


def test_exact_id_lane_entity(store):
    """entity_ids → exact lane hits without lexical signal (§28 row 1)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clE", "sA", "srcE", "spE",
                   "entity-bound evidence", gen)
        conn.execute(
            "INSERT INTO entities(entity_id,scope_id,label,created_event)"
            " VALUES('ent:deploy','sA','deploy-tool',1)",
        )
        conn.execute(
            "INSERT INTO claim_entities(claim_id,entity_id,role)"
            " VALUES('clE','ent:deploy','mention')",
        )
    res = recall_v3(
        store, request("deploy", entity_ids=("ent:deploy",))
    )
    ids = [i.handle.object_id for i in items_of(res)]
    assert "clE" in ids


def test_optional_lane_degrades(store):
    """Sparse/late lanes report unavailable honestly — never a fake empty
    success, and a declared-but-absent lane is never dropped from
    diagnostics (V3-28.06/28.12, V3-26.10)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "optional lane check", _gen(store))
    res = recall_v3(
        store,
        request("optional lane check"),
        # declared replay-lab override surface: puts the optional lanes in
        # this class's lane set so diagnostics must report them
        policy_tables={
            "class_lanes": {
                QueryClass.CURRENT_STATE: (
                    "exact_id", "lexical", "structured", "temporal",
                    "dense", "sparse", "late",
                ),
            },
        },
    )
    lane_status = res.capabilities["lanes"]
    # no provisioned artifacts → honest unavailability, in diagnostics
    assert lane_status["sparse"] == "unavailable"
    assert lane_status["late"] == "unavailable"
    # encoder-less store + empty embeddings table: dense is unavailable,
    # not silently ok (V3-28.12)
    assert lane_status["dense"] == "unavailable"


def test_handlers_sparse_late_unavailable(store):
    """Index jobs fail CAPABILITY_UNAVAILABLE through real dispatch —
    terminal ``failed``, never an INTERNAL crash loop (V3-28.06/28.12)."""
    from verbatim.config import VerbatimConfig
    from verbatim.ingest import Ingester

    ingester = Ingester(store, VerbatimConfig())
    with store.tx() as conn:
        seed_scope(conn, "sA")
        j_sparse = ingester.jobs.enqueue(
            conn, "sA", JobKind.SPARSE_INDEX, {"scope_id": "sA"}
        )
        j_late = ingester.jobs.enqueue(
            conn, "sA", JobKind.LATE_INDEX, {"scope_id": "sA"}
        )
    n = ingester.run_pending(
        scope="sA",
        kinds=[JobKind.SPARSE_INDEX, JobKind.LATE_INDEX],
        owner="w1",
    )
    assert n == 2
    with store.read() as conn:
        rows = {
            r[0]: (r[1], r[2])
            for r in conn.execute(
                "SELECT job_id, state, error_code FROM jobs"
                " WHERE job_id IN (?, ?)",
                (j_sparse, j_late),
            ).fetchall()
        }
    for jid in (j_sparse, j_late):
        # terminal 'failed' — a 'retry_wait' row would mean the handler
        # crashed where the honest capability error was required
        assert rows[jid] == (
            "failed", ErrorCode.CAPABILITY_UNAVAILABLE.value
        )

    # the frozen dispatch signature (job, owner, ingester) raises the
    # same typed, non-retryable error
    with pytest.raises(VerbatimError) as direct:
        _handlers.handle_sparse_index(
            {"job_id": "jx", "scope_id": "sA", "input_refs": {}},
            "w1",
            ingester,
        )
    assert direct.value.code == ErrorCode.CAPABILITY_UNAVAILABLE
    assert direct.value.retryable is False


# ---------------------------------------------------------------------
# §29 fusion + abstention
# ---------------------------------------------------------------------

def test_hard_identifier_extraction():
    ids = extract_hard_identifiers(
        "why does tests/test_x.py::test_y fail on src/foo/bar.py",
        ("ent:z",),
    )
    assert "tests/test_x.py::test_y" in ids or any(
        "::" in i for i in ids
    )
    assert "src/foo/bar.py" in ids
    assert "ent:z" in ids


def test_hard_identifier_abstention(store):
    """A required identifier absent from authorized evidence forces
    abstention (V3-29.05)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "unrelated content", _gen(store))
    res = recall_v3(
        store, request("why does src/missing/file.py fail")
    )
    assert res.abstained or "missing_hard_identifiers" in res.warnings
    assert not items_of(res)


def test_identifier_satisfied_no_abstain(store):
    """The same identifier present in authorized evidence does not
    abstain."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "fix for src/missing/file.py landed", _gen(store))
    res = recall_v3(store, request("fix for src/missing/file.py"))
    assert "missing_hard_identifiers" not in res.warnings


def test_group_verdict_supported(store):
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "verdict probe evidence", _gen(store))
    res = recall_v3(store, request("verdict probe evidence"))
    assert not res.abstained
    assert items_of(res)


def test_conflict_closure_atomic(store):
    """Both sides of an open conflict travel together (V3-26.13, §30.06)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clA", "sA", "srcA", "spA",
                   "conflict side alpha", gen)
        seed_claim(conn, "clB", "sA", "srcB", "spB",
                   "conflict side beta", gen)
        conn.execute(
            "INSERT INTO conflict_groups(group_id,scope_id,status)"
            " VALUES('cg1','sA','open')",
        )
        conn.execute(
            "INSERT INTO conflict_members(group_id,claim_id)"
            " VALUES('cg1','clA'),('cg1','clB')",
        )
    res = recall_v3(store, request("conflict side alpha"))
    ids = {i.handle.object_id for i in items_of(res)}
    assert "clA" in ids and "clB" in ids
    conflict_packs = [p for p in res.packs
                      if p.kind == PackKind.CONFLICT_PACK]
    assert conflict_packs


def test_conflict_member_unauthorized_omits_group(store):
    """A conflict member outside authorized scope makes the whole group
    incomplete → omitted, never half-shipped (V3-30.06)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_scope(conn, "sB", principal="p2", conv="c2")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clA", "sA", "srcA", "spA",
                   "disputed alpha", gen)
        seed_claim(conn, "clB", "sB", "srcB", "spB",
                   "disputed beta", gen)
        conn.execute(
            "INSERT INTO conflict_groups(group_id,scope_id,status)"
            " VALUES('cg1','sA','open')",
        )
        conn.execute(
            "INSERT INTO conflict_members(group_id,claim_id)"
            " VALUES('cg1','clA'),('cg1','clB')",
        )
    res = recall_v3(store, request("disputed alpha"))
    conflict_items = [
        i for p in res.packs if p.kind == PackKind.CONFLICT_PACK
        for i in p.items
    ]
    ids = {i.handle.object_id for i in conflict_items}
    assert "clB" not in ids


def test_candidate_cap(store):
    """Union cap bounds ordinary candidates; deps never count (V3-28.02)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        for i in range(10):
            seed_claim(conn, f"cl{i:02d}", "sA", f"src{i}", f"sp{i}",
                       "cap probe common term", gen)
    with store.read() as conn:
        plan = analyze("cap probe common term",
                       request("cap probe common term"), 1)
        ctx = _lanes.LaneContext(
            conn=conn, store=store,
            request=request("cap probe common term"),
            plan=plan, query_class=QueryClass.CURRENT_STATE,
            scope_ids=("sA",), generation=_gen(store),
            deadline=_lanes._cand.Deadline(None), lane_cap=40,
        )
        res = _lanes.lane_lexical(ctx)
        union = _union.build_union(ctx, [res], candidate_cap=4,
                                   lane_weights={})
    ordinary = [k for k, h in union.items() if not h.is_dependency]
    assert len(ordinary) <= 4
    assert union.overflow > 0


def test_family_dedup(store):
    """Same evidence family → best member kept, rest deduped (V3-29.07)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clF1", "sA", "srcF1", "spF1",
                   "family dedup probe", gen, family_id="fam1")
        seed_claim(conn, "clF2", "sA", "srcF2", "spF2",
                   "family dedup probe", gen, family_id="fam1")
        seed_claim(conn, "clG", "sA", "srcG", "spG",
                   "family dedup probe", gen, family_id="fam2")
    with store.read() as conn:
        plan = analyze("family dedup probe",
                       request("family dedup probe"), 1)
        ctx = _lanes.LaneContext(
            conn=conn, store=store,
            request=request("family dedup probe"),
            plan=plan, query_class=QueryClass.CURRENT_STATE,
            scope_ids=("sA",), generation=_gen(store),
            deadline=_lanes._cand.Deadline(None), lane_cap=40,
        )
        res = _lanes.lane_lexical(ctx)
        union = _union.build_union(ctx, [res], candidate_cap=64,
                                   lane_weights={})
    fams = [h.family_id for h in union.values() if h.family_id]
    assert fams.count("fam1") == 1  # fam1 collapsed to one member


def test_freshness_required_drops_volatile(store):
    """freshness_required filters volatile claims at admission (§24)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clV", "sA", "srcV", "spV",
                   "volatile freshness probe", gen, freshness="volatile")
        seed_claim(conn, "clS", "sA", "srcS", "spS",
                   "volatile freshness probe", gen, freshness="stable")
    res = recall_v3(
        store, request("volatile freshness probe", freshness_required=True)
    )
    ids = {i.handle.object_id for i in items_of(res)}
    assert "clV" not in ids
    assert "clS" in ids


# ---------------------------------------------------------------------
# §30 typed packs + budgets
# ---------------------------------------------------------------------

def test_pack_kinds_and_untrusted_framing(store):
    """Delivered items are wrapped as untrusted typed data (V3-30.08)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "pack framing probe", _gen(store))
    res = recall_v3(store, request("pack framing probe"))
    assert res.packs
    for p in res.packs:
        assert isinstance(p.kind, PackKind)
        for item in p.items:
            assert item.text.startswith("<memory_evidence")
            assert 'instruction="none"' in item.text
            assert item.handle.handle_id.startswith("ih_")


def test_shared_byte_budget(store):
    """A tiny max_bytes drops whole groups, never splits them (V3-30.02)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        for i in range(6):
            seed_claim(conn, f"cl{i}", "sA", f"src{i}", f"sp{i}",
                       f"budget probe item {i} with extra padding text",
                       gen)
    res = recall_v3(
        store, request("budget probe item", max_bytes=512)
    )
    # every serialized item honors the wrapper; total stays bounded
    total = sum(p.serialized_bytes for p in res.packs)
    assert total <= 512 or res.omitted > 0


def test_required_context_atomic(store):
    """Required context members ship with their claim or the group is
    omitted whole (V3-30.06)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        payload = b"Alice said: the deploy is blocked"
        add_source(conn, "src1", "sA", payload)
        add_span(conn, "spMain", "src1", 12, len(payload))
        add_span(conn, "spAttr", "src1", 0, 11)
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
            "recorded_from) VALUES('cg1','sA','src1',1,'p1','k1',"
            "'complete',1)",
        )
        conn.execute(
            "INSERT INTO context_members(group_id,span_id,role,required,"
            "ord) VALUES('cg1','spMain','primary',1,0),"
            "('cg1','spAttr','attribution',1,1)",
        )
    res = recall_v3(store, request("deploy blocked"))
    texts = texts_of(res)
    # required context member (attribution) ships alongside
    assert any("Alice said" in t for t in texts)
    assert any("deploy is blocked" in t for t in texts)


def test_procedure_pack(store):
    """Active procedures deliver via procedure_pack with exact steps
    (§21, §30.02)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        conn.execute(
            "INSERT INTO procedures(procedure_id,scope_id,revision,"
            "task_label,state,recorded_from) VALUES('pr1','sA',1,"
            "'run the unit tests','active',1)",
        )
        conn.execute(
            "INSERT INTO procedure_steps(procedure_id,revision,step_no,"
            "span_id,description) VALUES('pr1',1,1,NULL,"
            "'pytest tests/unit -x')",
        )
        conn.execute(
            "INSERT INTO procedure_signatures(procedure_id,revision,"
            "signature_digest,ordered_ops_json,intent_key,scope_id)"
            " VALUES('pr1',1,'d1','[\"run_check\"]','unit tests','sA')",
        )
    res = recall_v3(
        store, request("run the unit tests", modes=("procedure",))
    )
    proc_packs = [p for p in res.packs if p.kind == PackKind.PROCEDURE_PACK]
    assert proc_packs
    proc_texts = [i.text for p in proc_packs for i in p.items]
    assert any("pytest tests/unit -x" in t for t in proc_texts)


def test_procedure_pack_dict_shaped_failure_modes(store):
    """``failure_modes_json`` persists ``{"modes": [...], "status": ...}``
    (compiler payload); union's failure-ref walk must read the dict shape,
    never iterate dict keys (S1 envelope found the crash)."""
    import json as _json
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "clF", "sA", "srcF", "spF",
                   "fix-tests step failed on missing fixture", _gen(store))
        conn.execute(
            "INSERT INTO procedures(procedure_id,scope_id,revision,"
            "task_label,state,recorded_from,failure_modes_json)"
            " VALUES('prF','sA',1,'fix the tests steps','active',1,?)",
            (
                _json.dumps(
                    {
                        "modes": [
                            {
                                "signature": "missing-fixture",
                                "description": "fixture absent",
                                "evidence_refs": ["claim:clF"],
                                "recovery": "install fixture",
                                "occurrences": 1,
                            }
                        ],
                        "status": "observed",
                    }
                ),
            ),
        )
        conn.execute(
            "INSERT INTO procedure_steps(procedure_id,revision,step_no,"
            "span_id,description) VALUES('prF',1,1,NULL,"
            "'install the fixture')",
        )
        conn.execute(
            "INSERT INTO procedure_signatures(procedure_id,revision,"
            "signature_digest,ordered_ops_json,intent_key,scope_id)"
            " VALUES('prF',1,'dF','[\"run_check\"]','fix tests','sA')",
        )
    res = recall_v3(
        store, request("fix the tests steps", modes=("procedure",))
    )
    proc_packs = [p for p in res.packs if p.kind == PackKind.PROCEDURE_PACK]
    assert proc_packs
    proc_texts = [i.text for p in proc_packs for i in p.items]
    assert any("install the fixture" in t for t in proc_texts)


def test_verify_pack_for_volatile(store):
    """Volatile/stale objects earn verify_recommended items (§24.06)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clV", "sA", "srcV", "spV",
                   "volatile fact probe", gen)
        conn.execute(
            "INSERT INTO freshness(scope_id,object_kind,object_id,"
            "revision,class) VALUES('sA','claim','clV',1,'volatile')",
        )
    task = TaskContext(action_intent=ActionIntent.IRREVERSIBLE)
    res = recall_v3(store, request("volatile fact probe", task=task))
    verify_items = [
        i for p in res.packs if p.kind == PackKind.VERIFY_PACK
        for i in p.items
    ] + [
        i for p in res.packs for i in p.items if i.verify_recommended
    ]
    assert verify_items or any(
        i.verify_recommended for p in res.packs for i in p.items
    )


def test_context_v3_pack_filter(store):
    """context_v3 honors the caller's pack-kind restriction (§27.02)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "context filter probe", _gen(store))
    res = context_v3(
        store,
        ContextRequestV3(
            query="context filter probe", scope_id="sA",
            caller_id="human:alice", purpose="recall",
            packs=(PackKind.EVIDENCE_BUNDLE,),
        ),
    )
    for p in res.packs:
        assert p.kind == PackKind.EVIDENCE_BUNDLE


def test_screening_suppresses_secret(store):
    """Secret-shaped text inside an item is screened pre-delivery (§31)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clSec", "sA", "srcSec", "spSec",
                   "aws key AKIAABCDEFGHIJKLMNOP leaked", gen)
        seed_claim(conn, "clOk", "sA", "srcOk", "spOk",
                   "aws key rotation procedure", gen)
    res = recall_v3(store, request("aws key"))
    texts = texts_of(res)
    assert not any("AKIAABCDEFGHIJKLMNOP" in t for t in texts)


# ---------------------------------------------------------------------
# §32 influence
# ---------------------------------------------------------------------

def test_influence_handles_and_feedback(store):
    """Every delivered item carries a bound handle; feedback attaches."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "influence probe", _gen(store))
    res = recall_v3(store, request("influence probe"))
    assert items_of(res)
    handle = items_of(res)[0].handle
    assert handle.caller_id == "human:alice"
    assert handle.object_id == "cl1"
    assert handle.revision == 1
    with store.read() as conn:
        row = conn.execute(
            "SELECT receipt_id, scope_id, pack FROM influence"
            " WHERE handle_id = ?",
            (handle.handle_id,),
        ).fetchone()
    assert row is not None
    assert row[0] == handle.receipt_id
    with store.tx() as conn:
        record_feedback(conn, handle.handle_id, InfluenceFeedback.CITED,
                        action_receipt="ar:1")
    with store.read() as conn:
        fb = conn.execute(
            "SELECT feedback_kind, action_receipt FROM influence"
            " WHERE handle_id = ?",
            (handle.handle_id,),
        ).fetchone()
    assert fb == ("cited", "ar:1")


def test_blast_radius(store):
    """blast_radius joins influence + propagation exposure (V3-32.06)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "blast probe", _gen(store))
        conn.execute(
            "INSERT INTO propagations(propagation_id,scope_id,object_kind,"
            "object_id,revision,recipient_id,verbs_json,purpose,epoch,"
            "created_us) VALUES('pp1','sA','claim','cl1',1,'human:bob',"
            "'[\"read\"]','share',0,1)",
        )
    res = recall_v3(store, request("blast probe"))
    assert items_of(res)
    with store.read() as conn:
        br = blast_radius(conn, "claim", "cl1")
    recipients = br.recipients
    assert "human:alice" in recipients  # delivery exposure
    assert "human:bob" in recipients    # propagation exposure


def test_mint_handle_deterministic():
    h1 = _infl.mint_handle("rc1", "alice", 3, PackKind.EVIDENCE_BUNDLE,
                           "claim", "c1", 1, seq=1)
    h2 = _infl.mint_handle("rc1", "alice", 3, PackKind.EVIDENCE_BUNDLE,
                           "claim", "c1", 1, seq=1)
    assert h1 == h2
    assert h1.epoch == 3


# ---------------------------------------------------------------------
# misc pipeline behavior
# ---------------------------------------------------------------------

def test_no_signal_abstains(store):
    """A query with no lexical signal and no provisioned semantic lane
    abstains rather than scanning all memory (§29.12)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "some content", _gen(store))
    res = recall_v3(store, request("the and or"))
    assert res.abstained
    assert "no_signal" in res.warnings
    assert not items_of(res)


def test_min_ready_seq_pending(store):
    """Unmet readiness returns a pending result with receipt handle
    (V3-27.09)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
    res = recall_v3(
        store, request("anything", min_ready_seq=10 ** 9)
    )
    assert res.abstained
    assert "processing_pending" in res.warnings


def test_decision_log_replayable(store):
    """routing_decisions rows carry replay inputs (V3-26.06)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "replay probe", _gen(store))
    res = recall_v3(store, request("replay probe"))
    with store.read() as conn:
        row = conn.execute(
            "SELECT state_key, routes_json, budgets_json,"
            " result_sizes_json, state_json FROM routing_decisions"
            " WHERE decision_id = ?",
            (res.decision_id,),
        ).fetchone()
    assert row is not None
    assert row[0]  # state_key present
    assert "current" in row[1] or "exact" in row[1]
    # state_json is what makes the decision replayable — the one-way
    # state_key digest alone cannot reconstruct the discretized inputs.
    assert row[4]
    state_payload = json.loads(row[4])
    assert state_payload["state"]["qc"] == "current_state"
    assert state_payload["state"]["size"] in (
        "empty", "small", "medium", "large"
    )
    limits = state_payload["limits"]
    assert limits["max_items"] > 0 and limits["max_bytes"] > 0


def test_observation_derived_item(store):
    """Derived objects carry derived=True + proof_count (§30.04,
    V3-17.04). The episode surfaces as a derived item; the observation
    member is intentionally not propagated — observation injection is
    G4b-gated (V3-23.08) and no lane emits observation keys."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
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
    res = recall_v3(
        store, request("obs episode", modes=("exploratory",))
    )
    # the seeded episode MUST surface as a derived item — non-vacuous
    derived = [i for p in res.packs for i in p.items if i.derived]
    assert derived
    assert all(i.proof_count >= 0 for i in derived)
    ids = {i.handle.object_id for i in items_of(res)}
    assert "ep1" in ids
    # observation members are intentionally not propagated into delivery
    # (injection gate, V3-23.08); the derived marker rides the episode
    assert "ob1" not in ids


def test_dependency_claim_travels(store):
    """A failure/conflict edge pulls the dependency claim into the group
    (V3-28.02)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clMain", "sA", "srcM", "spM",
                   "main claim probe", gen)
        seed_claim(conn, "clDep", "sA", "srcD", "spD",
                   "dependency evidence", gen)
        conn.execute(
            "INSERT INTO edges(edge_id,scope_id,source_kind,source_id,"
            "target_kind,target_id,edge_type,created_event) VALUES('e1',"
            "'sA','claim','clMain','claim','clDep','corrects',1)",
        )
    res = recall_v3(store, request("main claim probe"))
    ids = {i.handle.object_id for i in items_of(res)}
    assert "clMain" in ids
    assert "clDep" in ids  # dependency travels inside the group
