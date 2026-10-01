"""SPEC_V4_5 §15 acceptance scenarios D13–D24.

Real ``Store.create`` fixtures (v3/v4 schema) with direct-SQL seeding —
the same harness shape as ``tests/v4/test_scenarios_*.py`` and the
sibling ``tests/v45/test_scenarios_d1.py`` (D01–D12). Every test
exercises a production path — the bake-off/disposition surfaces in
``eval/v45``, the graph lane's reference operations, the projection
builder/edit path, ``ProfileService``, the Holographic connector, the
working-memory ``promote`` gate, and the refresh scheduler — never a
helper shim (V45-15.01). No sleeps; explicit timestamps/sequence numbers
throughout.

Scenario status at authoring time:

- D13 — ``eval/v45/bakeoff.py`` is the real X1 harness; the test drives
  its registry/row/compare/cost APIs (the measured ``run()`` path is
  exercised by ``tests/eval/test_v45_bakeoff.py``).
- D14 — ``compose_report`` in ``verbatim/retrieval/v3/lanes.py`` is the
  X8 honest-composition surface; associative candidates are measured
  against the exact intersection, never instead of it.
- D15 — projection build/edit path: ``safe_relpath``/``check_inside``/
  ``prepare_root`` reject escapes; ``propose_edit`` is proposal-only.
- D16 — ``ProfileService`` packs bind the issuing caller; peer sessions
  never merge private models.
- D17 — the X5 guard is implemented: ``put_topic`` rejects
  exclusion-shaped match specs and ``compile`` names safety-bearing
  events under ``safety_events`` (steered, never suppressed).
- D18 — Holographic connector: non-original items persist as imported
  assertions; dry-run and accepted receipts name missing provenance.
- D19 — ``working.promote`` requires a persisted live
  ``capture_authorizations`` row before any durable write.
- D20 — structural scan: one Engine/Ingester/authorization/resolver;
  ``eval/v45`` opens only disposable stores through ``corpus.make_store``.
- D21 — ``eval/v45/dispositions.json`` keeps every I1–I6/X1–X8 row;
  failed/deferred/research mechanisms never enter recommended routing.
- D22 — the M5 claim set names every parent §03 registry comparator.
- D23 — refresh utility ordering is advisory: a scheduled pass touches
  only ``jobs`` + ``events``; grants/receipts/evidence bytes untouched.
- D24 — the honesty scan extended to v4.5 surfaces; I1–I6 carry no
  paired outcomes, so nothing describes a market win.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import os
import re
from pathlib import Path

import pytest

from verbatim.config import VerbatimConfig
from verbatim.core.time import now_us
from verbatim.core.types import (
    ErrorCode,
    JobKind,
    VerbatimError,
    json_dumps,
)
from verbatim.core.types_v3 import (
    CaptureAuthorization,
    EnvelopeKind,
    QueryClass,
    RecallRequestV3,
)
from verbatim.connectors import ConnectorService
from verbatim.governance import (
    CallerV3,
    create_grant,
    get_capture_authorization,
    issue_capture_authorization,
    register_principal,
    seed_purposes,
)
from verbatim.observations import working
from verbatim.profiles import ProfileService
from verbatim.projections import (
    ProjectionAuthority,
    build,
    propose_edit,
)
from verbatim.projections.paths import check_inside, safe_relpath
from verbatim.refresh import (
    REFRESH_EVENT_KIND,
    RefreshBudget,
    RefreshScheduler,
)
from verbatim.retrieval.query import analyze
from verbatim.retrieval.v3 import lanes as _lanes
from verbatim.storage.store import Store

from eval.v45 import bakeoff, dispositions


_TEST_HMAC_KEY = b"test-hmac-key-padded-to-32-bytes"
_REPO = Path(__file__).resolve().parents[2]

# Deterministic logical timestamps — never a wall clock or a sleep.
T0 = 1_700_000_000_000_000


def _h(data: bytes) -> bytes:
    return hmac.new(_TEST_HMAC_KEY, data, hashlib.sha256).digest()


@pytest.fixture
def store(tmp_path):
    (tmp_path / "v45.db.key").write_bytes(_TEST_HMAC_KEY)
    s = Store.create(str(tmp_path / "v45.db"))
    yield s
    s.close()


def _gen(store) -> int:
    return store.projection_generation()


# ---------------------------------------------------------------------------
# seeding helpers (mirror tests/v45/test_scenarios_d1.py)
# ---------------------------------------------------------------------------

def seed_scope(conn, scope_id, principal="p1", conv="c1", profile="prof",
               vis="conversation"):
    conn.execute(
        "INSERT INTO scopes(scope_id,profile_id,principal_id,workspace_id,"
        "conversation_id,visibility,acl_revision) VALUES(?,?,?,?,?,?,0)",
        (scope_id, profile, principal, "ws", conv, vis),
    )


def seed_auth(conn, scope_id, pid="human:alice", purposes=("recall",),
              verbs=("read", "quote"), issuer=None):
    seed_purposes(conn)
    register_principal(conn, kind="human", principal_id=pid)
    return create_grant(
        conn, scope_id=scope_id, principal_id=pid, verbs=set(verbs),
        issuer_id=issuer or pid,
        purposes=None if purposes is None else list(purposes),
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
               state="active", recorded_from=1, recorded_until=None,
               rev=1, subject=None, predicate=None):
    """Claim + source + span + claim_evidence + FTS row — the
    retrieval-path fixture. ``subject``/``predicate`` land on both the
    claims row and the head revision (``rev_predicate`` is the
    authoritative safety-classification input, V45-09.05)."""
    payload = text.encode("utf-8")
    add_source(conn, source_id, scope_id, payload)
    add_span(conn, span_id, source_id, 0, len(payload))
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,?,?,?,1)",
        (claim_id, scope_id, subject, predicate, recorded_from),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,object_json,"
        "condition_json,recorded_from,recorded_until,subject_id,predicate,"
        "perspective_id,freshness) VALUES(?,?,?,?,NULL,?,?,?,?,NULL,NULL)",
        (claim_id, rev, state,
         json_dumps({"text": text}) if text else None,
         recorded_from, recorded_until, subject, predicate),
    )
    conn.execute(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role,family_id) VALUES(?,?,?,'primary',NULL)",
        (claim_id, rev, span_id),
    )
    add_fts(conn, claim_id, rev, scope_id, text, gen)


def seed_structured_claim(conn, claim_id, scope_id, *, subject, predicate,
                          value, recorded_from, state="active",
                          recorded_until=None):
    """A consolidation-eligible structured claim (``object_json`` literal
    value) — the unit ``RefreshScheduler`` plans over."""
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event) VALUES(?,?,?,?,0)",
        (claim_id, scope_id, subject, predicate),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,object_json,"
        "polarity,modality,recorded_from,recorded_until)"
        " VALUES(?,1,?,?,'affirmative','asserted',?,?)",
        (claim_id, state,
         json_dumps({"kind": "literal", "text": value}),
         recorded_from, recorded_until),
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


def add_edge(conn, edge_id, scope_id, sk, sid, tk, tid, etype):
    conn.execute(
        "INSERT INTO edges(edge_id,scope_id,source_kind,source_id,"
        "target_kind,target_id,edge_type,decision_id,created_event,"
        "retired_event) VALUES(?,?,?,?,?,?,?,NULL,1,NULL)",
        (edge_id, scope_id, sk, sid, tk, tid, etype),
    )


class _Cfg:
    """Minimal retrieval cfg exposing the graph toggle."""
    graph = True
    dense = False
    sparse = False
    late_interaction = False
    causal = False


def _ctx(conn, store, seeds, query="v45 probe", scope_ids=("sA",),
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


def _row(cid, **kw):
    """A fully-pinned tested comparator row for D13."""
    base = dict(
        comparator_id=cid,
        edition="oss",
        track=bakeoff.TRACK_CONTROLLED,
        status=bakeoff.STATUS_TESTED,
        commit="deadbeef",
        deployment="local_in_process",
        models={"encoder": "hashing:subword-ngram:v1",
                "reader": "reader-v1"},
        prompts="p1",
        extraction={"harvester": "v3"},
        budgets={"k": 5},
        readiness="drained",
        pricing_date="2026-09-01",
        metrics={"aggregate": {"recall_at_k": 0.5}},
        costs=bakeoff.CostBreakdown(ingest=1.0),
    )
    base.update(kw)
    return bakeoff.ComparatorRow(**base)


def _qrows(conn, sql, params=()):
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# D13 — bake-off row honesty (V45-10.04, V45-10.05; X1)
# ---------------------------------------------------------------------------

def test_d13_bakeoff_rows_pin_readiness_budget_and_cost():
    """D13 / V45-10.04, V45-10.05: the X1 bake-off harness enforces the
    pin set structurally — a ``tested`` row that lacks edition, commit,
    deployment, reader, budgets, readiness, pricing date, or measured
    metrics is refused by :func:`validate_row`/:func:`pin`, a partial
    eight-category cost total is labeled ``complete: false``, and a
    weakened comparator (disabled embeddings, broken ingest, truncated
    context) can never count as a competitor loss."""
    rows = bakeoff.registry()
    # Every row is a named, statused comparator — edition is always
    # present, and non-tested rows always carry their reason.
    assert rows and all(r.comparator_id and r.edition for r in rows)
    assert all(
        r.track in bakeoff.TRACKS and r.status in bakeoff.STATUSES
        for r in rows
    )
    assert all(
        r.reason for r in rows if r.status != bakeoff.STATUS_TESTED
    )

    # The pin floor: a `tested` claim cannot stand without the whole
    # V4-54.01 set — edition/commit/deployment/models.reader/budgets/
    # readiness/pricing_date — plus measured metrics.
    good = _row("pinned")
    assert good.pinned()
    assert bakeoff.validate_row(good) == []
    for missing_kw in (
        {"commit": None},
        {"deployment": None},
        {"models": {}},
        {"budgets": {}},
        {"readiness": None},
        {"pricing_date": None},
        {"metrics": {}},
    ):
        bad = _row("loose", **missing_kw)
        assert bakeoff.validate_row(bad), missing_kw
    # ``pin`` refuses to mint an invalid row — the API itself enforces.
    with pytest.raises(ValueError):
        bakeoff.pin(dataclasses.replace(good, commit=None))
    refilled = bakeoff.pin(
        _row("was-bare", metrics={"aggregate": {"recall_at_k": 0.4}}))
    assert refilled.pinned() and bakeoff.validate_row(refilled) == []

    # The cost pin is the V45-10.05 eight-category accounting: missing
    # categories are named ``unmeasured``, never silently zeroed, and a
    # partial total is never presented complete.
    assert set(bakeoff.COST_CATEGORIES) == {
        "ingest", "extraction", "embeddings", "consolidation",
        "query_inference", "answer_reading", "storage", "maintenance",
    }
    partial = bakeoff.CostBreakdown(ingest=1.0, query_inference=2.0)
    assert partial.complete() is False
    assert partial.total()["complete"] is False
    assert set(partial.unmeasured()) == set(bakeoff.COST_CATEGORIES) - {
        "ingest", "query_inference"
    }
    full = bakeoff.CostBreakdown(
        **{c: 1.0 for c in bakeoff.COST_CATEGORIES}
    )
    assert full.complete() and full.total()["complete"] is True

    # A weakened comparator cannot lose — each declared weakener
    # invalidates the verdict and is named in the refusal reason.
    for weakener in ("disabled_embeddings", "broken_ingest",
                     "truncated_context"):
        weak = _row(
            "competitor",
            metrics={"aggregate": {"recall_at_k": 0.1}},
            weakened=(weakener,),
        )
        verdict = bakeoff.compare(good, weak)
        assert verdict["valid"] is False
        assert verdict["winner"] is None
        assert "weakened_comparator" in verdict["reason"]
        assert weakener in verdict["reason"]
    # The rule binds our own arm too — a weakened verbatim row cannot
    # manufacture a win either direction.
    verdict = bakeoff.compare(
        dataclasses.replace(good, weakened=("disabled_embeddings",)),
        _row("other", metrics={"aggregate": {"recall_at_k": 0.0}}),
    )
    assert verdict["valid"] is False
    # Untested comparators are never defeated (V45-15.02).
    gone = _row("hosted", status=bakeoff.STATUS_UNAVAILABLE,
                reason="hosted", metrics={})
    assert bakeoff.compare(good, gone)["valid"] is False
    assert "untested" in bakeoff.compare(good, gone)["reason"]
    # Nothing is ever inferred as defeated from the claim set alone.
    assert bakeoff.claim_set(rows)["defeated"] == []


# ---------------------------------------------------------------------------
# D14 — associative composition vs exact intersection (V45-09.08; X8)
# ---------------------------------------------------------------------------

def test_d14_associative_composition_reports_against_exact_intersection(
    store,
):
    """D14 / V45-09.08: holographic/associative composition is reported
    AGAINST the exact entity intersection — ``compose(entity_set)`` is
    the reference, candidates confirmed by it are ``verified``, and
    everything else stays ``associative_only`` (candidate accelerators,
    never minted facts). The graph lane's binding is structural
    typed-edge composition — no vector/HRR decoding is presented as
    canonical evidence."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "alpha beta shared claim", gen)
        seed_claim(conn, "cl2", "sA", "src2", "sp2",
                   "alpha side claim", gen)
        seed_claim(conn, "cl3", "sA", "src3", "sp3",
                   "beta side claim", gen)
        seed_claim(conn, "cl4", "sA", "src4", "sp4",
                   "edge neighbour claim", gen)
        add_entity(conn, "ent-a", "sA", "Alpha")
        add_entity(conn, "ent-b", "sA", "Beta")
        link_entity(conn, "cl1", "ent-a")
        link_entity(conn, "cl1", "ent-b")   # cl1 names BOTH entities
        link_entity(conn, "cl2", "ent-a")
        link_entity(conn, "cl3", "ent-b")
        add_edge(conn, "e1", "sA", "claim", "cl1", "claim", "cl4",
                 "supports")

    with store.read() as conn:
        ctx = _ctx(conn, store, [])
        # probe(entity) — authorized claims bound to one entity.
        assert sorted(_lanes.probe_entity(ctx, "ent-a")) == ["cl1", "cl2"]
        assert sorted(_lanes.probe_entity(ctx, "ent-b")) == ["cl1", "cl3"]
        # compose(entity_set) — the exact-intersection reference: ONLY
        # cl1 is linked to every entity in the set.
        assert _lanes.compose_entities(ctx, ["ent-a", "ent-b"]) == ["cl1"]
        # The X8 report measures the associative candidate set against
        # that intersection — a bounded graph walk included.
        rep = _lanes.compose_report(ctx, ["ent-a", "ent-b"], hops=1)

    assert rep["reference"] == "exact_intersection"
    assert rep["exact"] == ["cl1"]
    # Only what exact intersection confirms may be treated as composed
    # fact; the rest are labeled candidates with both denominators.
    assert rep["verified"] == ["cl1"]
    assert rep["associative_only"] == ["cl2", "cl3", "cl4"]
    assert rep["n_exact"] == 1 and rep["n_associative"] == 4
    assert rep["binding"] == "structural_typed_edge"
    assert rep["composition"] == "structural"
    assert "exact" in rep["note"]
    assert "never instead" in rep["note"]

    # The graph lane itself labels its composition honestly: typed-edge
    # structural binding, each hit carrying its edge/entity provenance
    # class — association is labeled, never conflated with the exact
    # intersection's verified set.
    with store.read() as conn:
        res = _lanes.lane_graph(_ctx(conn, store, [("claim", "cl1")]))
    assert res.details["composition"] == "structural"
    assert res.details["binding"] == "typed_edge"
    prov = res.details["provenance"]
    assert prov["claim:cl2"]["via"] == "entity"
    assert prov["claim:cl2"]["class"] == "topic_association"
    assert prov["claim:cl4"]["via"] == "supports"

    # X8 stays research: no decoding path mints a fact — the dispositions
    # table keeps HRR-class composition out of recommended routing until
    # it beats this reference plus hybrid retrieval (§16 stop condition).
    assert dispositions.is_recommended("X8") is False


# ---------------------------------------------------------------------------
# D15 — projection edit safety (V45-09.03; X3)
# ---------------------------------------------------------------------------

def test_d15_projection_edit_cannot_overwrite_or_escape(store, tmp_path):
    """D15 / V45-09.03: the Markdown projection is derived, never
    canonical — file names reject traversal/hidden/hook shapes, the root
    rejects symlink escapes, and a file edit lands as an open review
    proposal pinned to the revision the file showed; original evidence
    bytes and revisions are untouched."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA", verbs=("read", "quote", "review"))
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "the cat sat on the mat", _gen(store))
    auth = ProjectionAuthority(
        principal_id="human:alice", purpose="recall"
    )

    # ---- path safety: escapes, hidden/hook names, symlinked roots.
    for bad in ("../outside.md", "..", "a/b.md", ".hidden.md", "x.py"):
        with pytest.raises(VerbatimError) as ei:
            safe_relpath(bad)
        assert ei.value.code is ErrorCode.VALIDATION
    root = str(tmp_path / "proj")
    os.makedirs(root)
    with pytest.raises(VerbatimError):
        check_inside(root, str(tmp_path / "outside.md"))
    with pytest.raises(VerbatimError):
        check_inside(root, os.path.join(root, ".git", "hooks", "x"))
    real = tmp_path / "real-root"
    real.mkdir()
    link = tmp_path / "link-root"
    link.symlink_to(real)
    with pytest.raises(VerbatimError) as ei:
        build(store, str(link), authority=auth)
    assert ei.value.code is ErrorCode.VALIDATION

    # ---- the real build publishes a derived generation.
    res = build(store, root, authority=auth)
    scope_files = [
        f for f in res["files"] if f["path"].startswith("scope-")
    ]
    assert scope_files, "scope file must be projected"

    # Canonical evidence snapshot BEFORE the edit proposal.
    with store.read() as conn:
        src_before = conn.execute(
            "SELECT payload, payload_hmac FROM source_revisions"
            " WHERE source_id = 'src1'"
        ).fetchone()
        revs_before = conn.execute(
            "SELECT claim_id, revision, state, object_json"
            " FROM claim_revisions WHERE claim_id = 'cl1'"
            " ORDER BY revision"
        ).fetchall()

    # A file edit becomes a review proposal — never a write.
    out = propose_edit(
        store, authority=auth, scope_id="sA", claim_id="cl1",
        base_revision=1, new_object={"text": "cat on rug"},
        note="edit proposed from projected file",
    )
    assert out["applied"] is False
    assert out["conflict"] is False
    with store.read() as conn:
        row = conn.execute(
            "SELECT state, expected_versions_json FROM reviews"
            " WHERE review_id = ?",
            (out["review_id"],),
        ).fetchone()
        assert row is not None and row[0] == "open"
        assert json.loads(row[1]) == {"cl1": 1}
        # Original evidence is byte/revision immutable — the proposal
        # did not touch it.
        assert conn.execute(
            "SELECT payload, payload_hmac FROM source_revisions"
            " WHERE source_id = 'src1'"
        ).fetchone() == src_before
        assert conn.execute(
            "SELECT claim_id, revision, state, object_json"
            " FROM claim_revisions WHERE claim_id = 'cl1'"
            " ORDER BY revision"
        ).fetchall() == revs_before

    # A stale base is a surfaced conflict, never an auto-merge.
    with store.tx() as conn:
        conn.execute(
            "UPDATE claim_revisions SET recorded_until = 2"
            " WHERE claim_id = 'cl1' AND recorded_until IS NULL"
        )
        conn.execute(
            "INSERT INTO claim_revisions(claim_id,revision,state,"
            "object_json,recorded_from,recorded_until,subject_id,"
            "predicate) VALUES('cl1',2,'active',NULL,2,NULL,NULL,NULL)"
        )
    with pytest.raises(VerbatimError) as ei:
        propose_edit(
            store, authority=auth, scope_id="sA", claim_id="cl1",
            base_revision=1, new_object={"text": "stale edit"},
        )
    assert ei.value.code is ErrorCode.STALE_PROPOSAL
    # And a caller without review authority cannot propose at all.
    no_review = ProjectionAuthority(
        principal_id="human:mallory", purpose="recall"
    )
    with pytest.raises(VerbatimError) as ei:
        propose_edit(
            store, authority=no_review, scope_id="sA", claim_id="cl1",
            base_revision=2, new_object={"text": "x"},
        )
    assert ei.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


# ---------------------------------------------------------------------------
# D16 — session peers never share private perspective packs (V45-09.04; X4)
# ---------------------------------------------------------------------------

def test_d16_session_peers_no_shared_private_packs(store):
    """D16 / V45-09.04: two principals sharing one session scope keep
    partitioned private models — each receives its own pack, a pack
    binds the issuing caller (``pack_not_transferable`` for the peer),
    subject-scoped entries stay withheld from the peer, and shared
    membership never merges the two perspectives."""
    svc = ProfileService(store)
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_purposes(conn)
        register_principal(conn, kind="human", principal_id="human:alice")
        register_principal(conn, kind="human", principal_id="human:bob")
        create_grant(
            conn, scope_id="sA", principal_id="human:alice",
            verbs={"read", "quote", "derive", "admin"},
            issuer_id="human:alice", purposes=None,
        )
        create_grant(
            conn, scope_id="sA", principal_id="human:bob",
            verbs={"read", "quote", "derive"},
            issuer_id="human:alice", purposes=None,
        )
        seed_claim(conn, "ca", "sA", "srcA", "spA",
                   "alice prefers vim editor", _gen(store),
                   subject="human:alice", predicate="prefers")
        seed_claim(conn, "cb", "sA", "srcB", "spB",
                   "bob prefers emacs editor", _gen(store),
                   subject="human:bob", predicate="prefers")
    alice = CallerV3(principal_id="human:alice")
    bob = CallerV3(principal_id="human:bob")
    svc.put_topic(
        alice, "sA",
        {"topic_key": "editor", "match": {"keywords": ["editor"]}},
        purpose="admin",
    )
    svc.compile(alice, "sA", purpose="derive")

    # Entries stay partitioned inside the shared session: each peer
    # sees exactly its own subject model.
    a_view = svc.entries(alice, "sA", subject_id="human:alice")
    b_view = svc.entries(bob, "sA", subject_id="human:bob")
    assert len(a_view["entries"]) == 1
    assert len(b_view["entries"]) == 1
    assert (
        a_view["entries"][0]["subject_id"]
        != b_view["entries"][0]["subject_id"]
    )
    # Bob reaching for alice's private entries: withheld, not merged —
    # scope membership is not audience membership.
    b_on_a = svc.entries(bob, "sA", subject_id="human:alice")
    assert b_on_a["entries"] == []
    assert b_on_a["withheld"] >= 1

    # Each peer builds its own pack — the pack binds the issuing caller.
    apack = svc.build_perspective_pack(
        alice, "sA", subjects=["human:alice"], topics=["editor"],
        verbs=["read", "quote"], purpose="recall",
    )
    bpack = svc.build_perspective_pack(
        bob, "sA", subjects=["human:bob"], topics=["editor"],
        verbs=["read"], purpose="recall",
    )
    assert apack["caller_id"] == "human:alice"
    assert bpack["caller_id"] == "human:bob"

    candidates = [
        {"id": "x", "scope_id": "sA", "score": 1.0,
         "topics": ["editor"], "subjects": ["human:alice"]},
    ]
    # Alice's private pack cannot be applied by Bob — and vice versa.
    res = svc.apply_perspective(bob, apack, candidates)
    assert res["items"] == []
    assert res["reason"] == "pack_not_transferable"
    res = svc.apply_perspective(alice, bpack, candidates)
    assert res["items"] == []
    assert res["reason"] == "pack_not_transferable"

    # Each pack works for its own caller — attenuation only, no widening.
    res_a = svc.apply_perspective(alice, apack, candidates)
    assert res_a.get("reason") != "pack_not_transferable"
    assert res_a["items"]
    assert "read" in res_a["authorized_verbs"]
    res_b = svc.apply_perspective(bob, bpack, candidates)
    assert res_b["authorized_verbs"] == ["read"]
    # Bob's hints are HIS OWN model — alice's subject-only entries never
    # leak into a shared session's pack (V45-09.04: no merge).
    assert res_b["profile_hints"]
    assert all(
        h["subject_id"] == "human:bob" for h in res_b["profile_hints"]
    )


# ---------------------------------------------------------------------------
# D17 — topic policy cannot suppress safety events (V45-09.05; X5)
# ---------------------------------------------------------------------------

def test_d17_topic_policy_cannot_suppress_safety_events(store):
    """D17 / V45-09.05 (parent binding V4-16.06): a topic's ``match``
    spec is a positive steering filter only. Exclusion-shaped specs are
    refused at authoring time (they would persist as silent no-ops), and
    at compile time a deletion / consent-withdrawal / correction /
    security claim bypasses the filter — named under ``safety_events``,
    steered into topics, and never dropped."""
    svc = ProfileService(store)
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_purposes(conn)
        register_principal(conn, kind="human", principal_id="human:alice")
        create_grant(
            conn, scope_id="sA", principal_id="human:alice",
            verbs={"read", "quote", "derive", "admin"},
            issuer_id="human:alice", purposes=None,
        )
        # Safety-bearing claims — none carry the topic's narrow keyword.
        for cid, pred in (
            ("c-del", "data_deletion_request"),
            ("c-cons", "consent_withdrawal"),
            ("c-corr", "correction_issued"),
            ("c-sec", "security_incident"),
        ):
            seed_claim(
                conn, cid, "sA", f"src-{cid}", f"sp-{cid}",
                f"operator notice {cid}", _gen(store),
                subject="human:alice", predicate=pred,
            )
    alice = CallerV3(principal_id="human:alice")

    # An exclusion-shaped match spec is refused — the engine does not
    # honor negative filters, so persisting one would be a silent lie.
    for bad_match in (
        {"exclude_predicates": ["security_*"]},
        {"not_keywords": ["incident"]},
        {"drop": ["deletion"]},
        {"deny_predicates": ["consent_*"]},
    ):
        with pytest.raises(VerbatimError) as ei:
            svc.put_topic(
                alice, "sA",
                {"topic_key": "ops", "match": bad_match},
                purpose="admin",
            )
        assert ei.value.code is ErrorCode.VALIDATION

    # A narrow positive filter — legitimately steers. The safety claims
    # match NOTHING in it, yet the guard must keep them visible.
    svc.put_topic(
        alice, "sA",
        {"topic_key": "ops", "match": {"keywords": ["nonexistent-zzz"]}},
        purpose="admin",
    )
    report = svc.compile(alice, "sA", purpose="derive")
    flagged = {
        e["claim_id"]: e["category"] for e in report["safety_events"]
    }
    assert flagged == {
        "c-del": "deletion",
        "c-cons": "consent_withdrawal",
        "c-corr": "correction",
        "c-sec": "security",
    }
    # The safety events bypassed the keyword filter — steered into the
    # topic's derivation set, counted, and never suppressed.
    assert report["claims_matched"] >= 4
    assert report["entries_created"] or report["entries_revised"]

    # And nothing vanished: the claims and the derived entries remain
    # visible through the ordinary surfaces.
    entries = svc.entries(alice, "sA", subject_id="human:alice")
    assert entries["entries"], "safety events must remain visible"
    with store.read() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM claims WHERE scope_id = 'sA'"
        ).fetchone()[0] == 4


# ---------------------------------------------------------------------------
# D18 — import without originals stays an imported assertion (V45-09.06; X6)
# ---------------------------------------------------------------------------

def _export_jsonl(path, header, items):
    lines = []
    if header is not None:
        lines.append(json.dumps(header))
    lines += [json.dumps(i) for i in items]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_d18_import_without_originals_stays_imported_assertion(
    store, tmp_path
):
    """D18 / V45-09.06: a Holographic export item without
    ``original: true`` never becomes byte-exact evidence — the dry-run
    names its missing provenance per item, the accepted receipt keeps it
    ``imported``/``imported_assertion``, and unknown fields are counted
    as losses rather than silently absorbed."""
    service = ConnectorService(store, VerbatimConfig())
    scope_id, principal = "scopeA", "alice"
    with store.tx() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO scopes"
            "(scope_id,profile_id,principal_id,visibility)"
            " VALUES(?,?,?,'owner')",
            (scope_id, "v3", principal),
        )
        register_principal(conn, kind="human", principal_id=principal)
        create_grant(
            conn, scope_id=scope_id, principal_id=principal,
            verbs={"ingest"}, issuer_id=principal,
        )
        issue_capture_authorization(
            conn, principal_id=principal, issuer_id=principal,
            allowed_kinds={EnvelopeKind.CONNECTOR_ITEM},
            retention_policy="keep", policy_revision="pol1",
            scope_ids={scope_id},
        )
    export = tmp_path / "export.jsonl"
    _export_jsonl(
        export,
        {
            "format": "holographic-export-v1",
            "provider": "holographic",
            "exported_us": T0,
            "source_scope": "remote:conv-7",
        },
        [
            {
                "id": "ep-1", "kind": "episode",
                "text": "we deployed the fix at noon",
                "original": True, "actor": "bob", "event_us": T0,
            },
            {
                "id": "cl-1", "kind": "claim",
                "text": "the fix shipped in release 4.2",
                "refs": ["ep-1"],
                "unknown_field": {"provider": "specific"},
            },
        ],
    )

    # ---- dry run: enumerate + classify, write nothing, name the loss.
    dry = service.pull(
        "holographic", {"path": str(export)},
        scope_id=scope_id, principal_id=principal, dry_run=True,
    )
    assert dry.dry_run is True and dry.state == "dry_run"
    assert dry.inserted == 2  # would_insert count
    assert dry.missing_provenance == 1  # cl-1 lacks actor/event_us
    by_ext = {i["external_id"]: i for i in dry.items}
    assert by_ext["cl-1"]["imported_assertion"] is True
    assert by_ext["cl-1"]["missing_provenance"]
    assert by_ext["ep-1"]["imported_assertion"] is False
    assert by_ext["cl-1"]["losses"] == ["unknown_field"]
    assert dry.scope_mappings == {"remote:conv-7": scope_id}
    with store.read() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM sources"
        ).fetchone()[0] == 0

    # ---- accepted pull: the same honesty lands in the durable record.
    report = service.pull(
        "holographic", {"path": str(export)},
        scope_id=scope_id, principal_id=principal,
    )
    assert report.state == "complete" and report.inserted == 2
    assert report.missing_provenance == 1
    assert report.losses == {"unknown_field": 1}
    with store.read() as conn:
        joined = _qrows(
            conn,
            "SELECT se.*, s.external_id FROM source_envelopes se"
            " JOIN sources s ON s.source_id = se.source_id"
            " WHERE se.scope_id = ?",
            (scope_id,),
        )
    env = {r["external_id"]: r for r in joined}
    assert env["ep-1"]["trust_class"] == "external_content"
    assert env["cl-1"]["trust_class"] == "imported"
    meta = json.loads(env["cl-1"]["metadata_json"])
    assert meta["imported_assertion"] is True
    assert meta["original_present"] is False
    assert meta["connector_id"] == "holographic"
    assert meta["declared_not_verified"] == ["provider_format_fidelity"]
    # No byte-exact original evidence claim is ever made for cl-1.
    assert env["ep-1"]["actor_principal"] == "bob"


# ---------------------------------------------------------------------------
# D19 — working memory needs retention consent (V45-09.07; X7)
# ---------------------------------------------------------------------------

def test_d19_working_memory_promotion_requires_retention_consent(store):
    """D19 / V45-09.07: session working memory becomes durable only
    through an explicit, persisted, live ``capture_authorizations`` row.
    Tool permission (an ``ingest`` grant) is not that record; a minted
    but never-persisted ``CaptureAuthorization`` is not that record; the
    gate runs before any durable write. With real consent the same items
    commit through ``ingest_envelope`` with promotion provenance."""
    scope_id, session, principal = "scopeW", "sess-1", "alice"
    with store.tx() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO scopes"
            "(scope_id,profile_id,principal_id,visibility)"
            " VALUES(?,?,?,'owner')",
            (scope_id, "v3", principal),
        )
        register_principal(conn, kind="human", principal_id=principal)
        # Tool permission — an ingest grant — deliberately present.
        create_grant(
            conn, scope_id=scope_id, principal_id=principal,
            verbs={"ingest"}, issuer_id=principal,
        )
        set_id = working.create_set(
            conn, scope_id, session, 60_000_000, now=T0
        )
        working.add_item(
            conn, set_id, "decision", text="deploy plan approved",
            now=T0,
        )

    def _auth_obj(aid, **kw):
        base = dict(
            authorization_id=aid,
            issuer_id=principal,
            principal_id=principal,
            allowed_kinds={EnvelopeKind.SYSTEM_EVENT},
            scope_ids={scope_id},
            retention_policy="keep",
            policy_revision="pol1",
            issued_us=T0,
        )
        base.update(kw)
        return CaptureAuthorization(**base)

    # (a) No authorization object at all — the session stays in-memory.
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as ei:
            working.promote(
                conn, store, scope_id, session,
                principal_id=principal, authorization=None, now=T0,
            )
        assert ei.value.code is ErrorCode.CONSENT_REQUIRED
        assert conn.execute(
            "SELECT COUNT(*) FROM source_envelopes"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM sources"
        ).fetchone()[0] == 0

    # (b) A minted CaptureAuthorization that was never persisted —
    #     an in-memory claim is not consent.
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as ei:
            working.promote(
                conn, store, scope_id, session,
                principal_id=principal,
                authorization=_auth_obj("cauth:forged"), now=T0,
            )
        assert ei.value.code is ErrorCode.CONSENT_REQUIRED
        assert conn.execute(
            "SELECT COUNT(*) FROM sources"
        ).fetchone()[0] == 0

    # (c) Persisted consent — but for a different principal. The row is
    #     authoritative: consent issued FOR bob does not cover alice's
    #     session.
    with store.tx() as conn:
        bob_aid = issue_capture_authorization(
            conn, principal_id="bob", issuer_id=principal,
            allowed_kinds={EnvelopeKind.SYSTEM_EVENT},
            retention_policy="keep", policy_revision="pol1",
            scope_ids={scope_id}, issued_us=T0,
        )
        with pytest.raises(VerbatimError) as ei:
            working.promote(
                conn, store, scope_id, session,
                principal_id=principal,
                authorization=_auth_obj(bob_aid, principal_id="bob"),
                now=T0,
            )
        assert ei.value.code is ErrorCode.CONSENT_REQUIRED
        assert conn.execute(
            "SELECT COUNT(*) FROM sources"
        ).fetchone()[0] == 0

    # (d) Real persisted consent covering (principal, scope, kind,
    #     policy) — the promote path commits through ingest_envelope.
    with store.tx() as conn:
        aid = issue_capture_authorization(
            conn, principal_id=principal, issuer_id=principal,
            allowed_kinds={EnvelopeKind.SYSTEM_EVENT},
            retention_policy="keep", policy_revision="pol1",
            scope_ids={scope_id}, issued_us=T0,
        )
        # The consent is a durable row, field-matched to the object.
        row = get_capture_authorization(conn, aid)
        assert row is not None and row["principal_id"] == principal
    with store.tx() as conn:
        out = working.promote(
            conn, store, scope_id, session,
            principal_id=principal,
            authorization=_auth_obj(aid), now=T0,
        )
        assert out["envelope_kind"] == "system_event"
        assert out["authorization_id"] == aid
        assert out["retention_policy"] == "keep"
        assert len(out["promoted"]) == 1
        env = conn.execute(
            "SELECT envelope_kind, capture_proof, session_id,"
            " metadata_json FROM source_envelopes"
        ).fetchone()
        assert env is not None
        assert env[0] == "system_event"
        assert env[1] == aid               # capture_proof = consent id
        assert env[2] == session
        meta = json.loads(env[3])
        # Durable provenance: the record never forgets it was promoted
        # session working memory.
        assert meta["promoted_from"] == "working_set"
        assert meta["working_set_id"] == set_id
        assert meta["session_id"] == session
        assert meta["authorization_id"] == aid
        assert meta["retention_policy"] == "keep"


# ---------------------------------------------------------------------------
# D20 — one engine, one store policy, one authorization path
#       (V45-01.02, V45-13.04)
# ---------------------------------------------------------------------------

def test_d20_single_engine_store_authority_no_v45_sidecar():
    """D20 / V45-01.02, V45-13.04: constructing a separate M5 engine or
    store fails qualification — the structural scan pins it. Exactly one
    ``Engine``/``Ingester``/authorization evaluator/store-resolution
    policy exists; no ``EngineV45``, no ``schema_v45`` sidecar; and the
    ``eval/v45`` harness opens only disposable ``Store.create`` sandboxes
    through its corpus helper — never a second production path."""
    root = _REPO / "verbatim"
    engine_defs, ingester_defs = [], []
    authorize_defs, open_sites = [], {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        text = path.read_text(encoding="utf-8")
        if re.search(r"^class Engine\b", text, re.M):
            engine_defs.append(str(rel))
        if re.search(r"^class Ingester\b", text, re.M):
            ingester_defs.append(str(rel))
        for m in re.finditer(
            r"^def authorize\(|^    def authorize\(", text, re.M
        ):
            authorize_defs.append(str(rel))
            break
        if "Store.open(" in text or "Store.create(" in text:
            open_sites[str(rel)] = any(
                tok in text
                for tok in (
                    "resolve_store_path", "require_store_path",
                    "open_store(",
                )
            )

    # One engine, one ingester, one evaluator — same contract as C91.
    assert engine_defs == ["api.py"], engine_defs
    assert ingester_defs == ["ingest.py"], ingester_defs
    assert sorted(authorize_defs) == [
        "governance/__init__.py",
        "governance/grants.py",
        "privacy/egress.py",
        "sdk/capture.py",
    ], authorize_defs
    sanctioned = {
        "api.py": True,
        "api_v3/facade.py": False,
        "adapters/hermes_v3.py": True,
        "sdk/capture.py": False,
        "cli.py": True,
        "memory/facade.py": True,     # consumer facade → resolver
        "memory/worker.py": False,    # reopens the facade-resolved path
        "replay/lab.py": False,
        "storage/store.py": False,
    }
    assert set(open_sites) <= set(sanctioned), (
        set(open_sites) - set(sanctioned)
    )
    for rel, uses_resolver in open_sites.items():
        if rel in ("api_v3/facade.py", "sdk/capture.py",
                   "replay/lab.py", "memory/worker.py", "storage/store.py"):
            continue
        assert sanctioned.get(rel) == uses_resolver, rel
        assert uses_resolver, (
            f"{rel} opens a profile store without the resolver"
        )

    # No v4.5 parallel authority anywhere in the implementation trees
    # (production modules + the M5 eval harness itself).
    for tree in (_REPO / "verbatim", _REPO / "eval"):
        for path in sorted(tree.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            assert "EngineV45" not in text, path
            assert "schema_v45" not in text, path

    # The M5 harness opens only disposable eval stores — through the
    # corpus helper, never a profile-store resolver of its own.
    eval_root = _REPO / "eval" / "v45"
    assert eval_root.is_dir()
    for path in sorted(eval_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"^class Engine\b|^class Ingester\b",
                             text, re.M), path
        assert not re.search(
            r"^def (resolve_store_path|require_store_path|open_store"
            r"|authorize)\b",
            text, re.M,
        ), path
        if "Store.open(" in text or "Store.create(" in text:
            # Only the corpus helper mints disposable stores — the same
            # sanctioned category as replay/lab.py sandboxes.
            assert path.name == "corpus.py", path


# ---------------------------------------------------------------------------
# D21 — failed experiments stay in the table, out of routing (V45-09.09)
# ---------------------------------------------------------------------------

def test_d21_failed_experiments_stay_published_not_recommended():
    """D21 / V45-09.09, V45-16.01: the published disposition table keeps
    every I1–I6 and X1–X8 row — a failed, deferred, or research-only
    mechanism is *visible* and never enters recommended routing. The
    ``recommended`` flag is the consumer gate; presence of a runner file
    is not acceptance."""
    assert dispositions.check() == [], (
        "dispositions.json is stale or malformed"
    )
    rows = dispositions.load()
    assert dispositions.validate_rows(rows) == []

    # Every mechanism id exactly once, every row carrying the honesty
    # fields the table contract requires.
    assert {r["id"] for r in rows} == set(dispositions.MECHANISM_IDS)
    for r in rows:
        assert r["disposition"] in dispositions.DISPOSITIONS
        assert isinstance(r["evidence"], list)
        assert "measured" in r
        assert isinstance(r["recommended"], bool)
        # recommended ⟺ adopt — a failed/deferred/research row can
        # never carry the routing flag.
        assert r["recommended"] == (r["disposition"] == "adopt")

    # The I-experiments: deferred/unmeasured — kept in the table, out of
    # recommended routing. An unmeasured experiment is not a failed one
    # but the routing rule is identical: only `adopt` recommends.
    for iid in ("I1", "I2", "I3", "I4", "I5", "I6"):
        assert dispositions.is_recommended(iid) is False
    # X8 is the §16-prescribed research row: visible, never recommended.
    by_id = {r["id"]: r for r in rows}
    assert by_id["X8"]["disposition"] == "research_only"
    assert by_id["X8"]["recommended"] is False
    assert set(dispositions.recommended_mechanisms()) == {
        r["id"] for r in rows if r["disposition"] == "adopt"
    }

    # The consumer contract itself: a rejected (failed) mechanism must
    # fail validation if recommended, and a dropped mechanism fails the
    # completeness check — the table cannot silently lose a loser.
    rejected_recommended = [
        {
            "id": "I1",
            "mechanism": "counterevidence-first manifest packing",
            "disposition": "reject",
            "recommended": True,
            "evidence": [],
            "measured": "unmeasured",
        }
    ]
    problems = dispositions.validate_rows(rejected_recommended)
    assert any("recommended" in p for p in problems)
    dropped = [r for r in rows if r["id"] != "I4"]
    assert any(
        "missing" in p for p in dispositions.validate_rows(dropped)
    )

    # And the routing surface itself defaults to the un-experimental
    # path — deferred I-mechanisms are opt-in request flags, never the
    # shipped default.
    req = RecallRequestV3(
        query="q", scope_id="sA", caller_id="human:alice",
        purpose="recall",
    )
    assert req.manifest is None
    assert req.pack_mode == "standard"


# ---------------------------------------------------------------------------
# D22 — parent-registry competitor coverage (V45-10.01)
# ---------------------------------------------------------------------------

def _parent_registry_rows():
    """Parse the SPEC_V4.md §03 competitor table — the real parent
    registry, never a reduced hand-written copy."""
    text = (_REPO / "SPEC_V4.md").read_text(encoding="utf-8")
    section = text.split("## 03.", 1)[1].split("## 04.", 1)[0]
    rows = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|") or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells or cells[0] in ("Competitor / edition", ""):
            continue
        rows.append(cells[0])
    return rows


def test_d22_m5_report_names_every_parent_registry_comparator():
    """D22 / V45-10.01, V45-15.02: the M5 claim set names every
    competitor/edition in the parent §03 registry — each is ``tested``,
    ``unavailable``, or ``out_of_scope``, and every non-tested row keeps
    its reason. Unavailable comparators stay visible and never enter
    defeated counts."""
    parent = _parent_registry_rows()
    assert len(parent) >= 23, (
        f"parent registry shrank — only {len(parent)} rows parsed"
    )

    rows = bakeoff.registry()
    by_id = {r.comparator_id: r for r in rows}
    ids = set(by_id)

    # Every parent-registry row maps to ≥1 named comparator row (the
    # head token of the competitor cell is the registry slug prefix).
    uncovered = []
    for cell in parent:
        head = re.sub(r"[^a-z0-9]+", "", cell.split()[0].lower())
        if not any(head and head in cid for cid in ids):
            uncovered.append(cell)
    assert uncovered == [], (
        f"parent-registry competitors missing from the M5 claim set: "
        f"{uncovered}"
    )
    # Dual-edition competitors keep hosted and OSS rows distinct —
    # a hosted score is never attributed to the OSS edition (V45-10.03).
    for dual in (
        ("mem0_oss", "mem0_platform"),
        ("hindsight_oss", "hindsight_cloud"),
        ("honcho_self_hosted", "honcho_cloud"),
    ):
        assert all(d in ids for d in dual), dual
        assert len({by_id[d].edition for d in dual}) == 2

    # The claim set is the full roll-call: every comparator sorted into
    # exactly one status bucket, reasons attached, defeats never
    # inferred.
    cs = bakeoff.claim_set(rows)
    assert sorted(cs["tested"] + cs["unavailable"]
                  + cs["out_of_scope"]) == sorted(ids)
    assert cs["defeated"] == []
    for r in rows:
        assert r.status in bakeoff.STATUSES
        if r.status != bakeoff.STATUS_TESTED:
            assert r.reason, f"{r.comparator_id} missing reason"
    # Hosted/platform editions that cannot be executed here stay
    # out_of_scope with an explicit reason — never tested, never
    # defeated, never dropped from the list.
    for cid in ("mem0_platform", "zep_hosted", "aws_agentcore_memory",
                "google_memory_bank", "microsoft_foundry_memory_preview",
                "redis_agent_memory_cloud"):
        assert by_id[cid].status == bakeoff.STATUS_OUT_OF_SCOPE
        assert by_id[cid].reason


# ---------------------------------------------------------------------------
# D23 — learned/utility scores cannot touch authority (V45-12.02)
# ---------------------------------------------------------------------------

#: Tables whose rows a utility ordering must never rewrite: grants,
#: consent, provenance, checker receipts, evidence digests, canonical
#: bytes. The behavioral check snapshots them around a refresh pass.
_AUTHORITY_TABLES = (
    "grants_v3",
    "capture_authorizations",
    "sources",
    "source_revisions",
    "spans",
    "claim_evidence",
    "claim_revisions",
    "claims",
    "derivations",
    "reviews",
    "operation_receipts",
    "readiness_obligations",
)


def _snapshot(conn, tables):
    snap = {}
    for t in tables:
        snap[t] = conn.execute(
            f"SELECT * FROM {t} ORDER BY rowid"
        ).fetchall()
    return snap


def test_d23_utility_ordering_cannot_rewrite_authority(store):
    """D23 / V45-12.02: utility-budgeted refresh is *advisory ordering
    only* — it chooses which ``consolidate`` jobs enqueue and records a
    ``refresh_pass`` audit event. A scheduled pass leaves grants,
    consent rows, checker receipts, provenance, evidence digests, and
    canonical bytes byte-identical; the only growth is ``jobs`` +
    ``events``. ``learned.py`` likewise carries no write surface."""
    sched = RefreshScheduler(store)
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_purposes(conn)
        register_principal(conn, kind="human", principal_id="human:alice")
        create_grant(
            conn, scope_id="sA", principal_id="human:alice",
            verbs={"read", "quote", "derive"},
            issuer_id="human:alice", purposes=None,
        )
        issue_capture_authorization(
            conn, principal_id="human:alice", issuer_id="human:alice",
            allowed_kinds={EnvelopeKind.SYSTEM_EVENT},
            retention_policy="keep", policy_revision="pol1",
            scope_ids={"sA"}, issued_us=T0,
        )
        for i in range(4):
            seed_structured_claim(
                conn, f"cl{i}", "sA", subject="subjB",
                predicate=f"pred{i}", value=f"refresh probe {i}",
                recorded_from=10,
            )

    with store.read() as conn:
        before = _snapshot(conn, _AUTHORITY_TABLES)
        jobs_before = conn.execute(
            "SELECT COUNT(*) FROM jobs"
        ).fetchone()[0]
        events_before = conn.execute(
            "SELECT COUNT(*) FROM events"
        ).fetchone()[0]

    # The production path: plan → enqueue → refresh_pass event.
    plan = sched.refresh("sA", since_seq=5)
    assert plan.scheduled, "changed regions must schedule"

    with store.read() as conn:
        after = _snapshot(conn, _AUTHORITY_TABLES)
        jobs_after = conn.execute(
            "SELECT kind, lane, state FROM jobs ORDER BY rowid"
        ).fetchall()
        events_after = _qrows(
            conn,
            "SELECT kind, payload_json FROM events ORDER BY rowid",
        )
    # Every authority table is byte-identical — utility chose work, it
    # did not rewrite provenance, grants, receipts, digests, or bytes.
    assert after == before
    # The only growth: real consolidate jobs on the background lane and
    # the refresh_pass audit event.
    assert len(jobs_after) == jobs_before + len(plan.scheduled) or \
        len(jobs_after) > jobs_before
    assert all(j[0] == JobKind.CONSOLIDATE.value for j in jobs_after)
    assert all(j[1] == "background" for j in jobs_after)
    refresh_events = [
        e for e in events_after if e["kind"] == REFRESH_EVENT_KIND
    ]
    assert len(refresh_events) == 1
    payload = json.loads(refresh_events[0]["payload_json"])
    assert payload["policy_id"] == plan.policy_id
    assert payload["spend"]["claims_processed"] > 0

    # Structural bound: neither the scheduler nor the learned-policy
    # module may carry a write aimed at the authority tables.
    write_re = re.compile(
        r"(?:INSERT\s+INTO|REPLACE\s+INTO|UPDATE\s+|DELETE\s+FROM)\s+"
        r"(grants_v3|capture_authorizations|source_revisions|"
        r"claim_revisions|claim_evidence|spans|reviews|"
        r"operation_receipts|readiness_obligations|derivations|"
        r"sources|claims)\b",
        re.IGNORECASE,
    )
    for rel in ("verbatim/refresh.py",
                "verbatim/retrieval/v3/learned.py"):
        text = (_REPO / rel).read_text(encoding="utf-8")
        assert not write_re.search(text), (
            f"{rel} writes to an authority table — utility must stay "
            "advisory (V45-12.02)"
        )
    # The learned module's contract is read-only by declaration.
    learned = (_REPO / "verbatim/retrieval/v3/learned.py").read_text(
        encoding="utf-8"
    )
    assert "never writes" in learned


# ---------------------------------------------------------------------------
# D24 — market-claim honesty (V45-01.04, V45-01.05)
# ---------------------------------------------------------------------------

#: Competitor names drawn from the parent §03 registry (head tokens +
#: edition words). A line claiming superiority over any of these is a
#: defect unless the line is itself the normative prohibition.
_COMPETITOR_TOKENS = (
    "mem0", "holographic", "hermes", "graphiti", "zep", "hindsight",
    "honcho", "letta", "memos", "reme", "openviking", "supermemory",
    "cognee", "byterover", "retaindb", "vestige", "memobase",
    "agentcore", "memory bank", "foundry", "redis", "langgraph",
    "evermemos", "memgpt", "langmem",
)

_SUPERIORITY = re.compile(
    r"(?:beats?|outperform\w*|superior to|faster than|better than|"
    r"more accurate than|leads?|state-of-the-art|sota\b|"
    r"best-in-class|industry.leading|market.leading|defeats?)",
    re.IGNORECASE,
)

#: Normative/prohibition language — the spec, manifests, and honest
#: reports *describe* the rule ("never listed as defeated", "hosted
#: scores must not attribute", "remains research") without making the
#: claim. A line must contain competitor + superiority language AND
#: lack any normative framing to count as a violation.
_NORMATIVE = re.compile(
    r"(?:must|shall|never|cannot|can't|not\b|prohibit|forbid|honest|"
    r"untested|unavailable|refus|deny|denial|invalid|defeated|"
    r"research|defer|separate|no claim|remain)",
    re.IGNORECASE,
)


def _honesty_offenders(path: Path) -> list[str]:
    out = []
    for i, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        low = line.lower()
        if (
            any(c in low for c in _COMPETITOR_TOKENS)
            and _SUPERIORITY.search(line)
            and not _NORMATIVE.search(line)
        ):
            out.append(f"{path.name}:{i}: {line.strip()[:120]}")
    return out


def test_d24_no_market_victory_claims_without_g410_evidence():
    """D24 / V45-01.04, V45-01.05: shipping I1–I6 without G4-10 evidence
    cannot be described as beating the market. The C93 competitor-
    superiority scan extends to every v4.5 surface — the spec itself,
    the M5 report, disposition table, out/ artifacts, and the release
    manifest — and the manifest's M5/G4-10/G4-14 gates stay honestly
    unimplemented. Untested editions remain untested, never defeated."""
    surfaces = [
        _REPO / "SPEC_V4_5.md",
        _REPO / "RELEASE_MANIFEST_V4.json",
        _REPO / "README.md",
        _REPO / "REQUIREMENTS.md",
        _REPO / "REQUIREMENTS_V4.md",
        _REPO / "eval" / "v45" / "report_v45.md",
        _REPO / "eval" / "v45" / "dispositions.json",
    ]
    out_dir = _REPO / "eval" / "v45" / "out"
    if out_dir.is_dir():
        surfaces.extend(sorted(out_dir.glob("*.md")))
        surfaces.extend(sorted(out_dir.glob("*.json")))
    offenders = []
    for path in surfaces:
        if path.exists():
            offenders.extend(_honesty_offenders(path))
    assert offenders == [], offenders

    # The M5/leadership gates are honestly unimplemented — no v4.5
    # artifact may present them as passed.
    manifest = json.loads(
        (_REPO / "RELEASE_MANIFEST_V4.json").read_text(encoding="utf-8")
    )
    gates = manifest.get("gates", {})
    for gate in ("G4-10", "G4-12", "G4-14"):
        status = str(gates.get(gate, {}).get("status", "")).lower()
        assert status not in ("satisfied", "passed", "met", "verified"), (
            f"{gate} claims satisfaction without evidence"
        )
    m5 = manifest.get("milestone_status", {}).get("M5", {})
    assert str(m5.get("status", "")).lower() in (
        "unimplemented", "deferred", "in_progress", "partial",
    ), m5

    # Nothing in the published claim set is defeated; unavailable and
    # out-of-scope comparators stay exactly that.
    cs = bakeoff.claim_set(bakeoff.registry())
    assert cs["defeated"] == []
    assert cs["unavailable"] or cs["out_of_scope"]

    # And the I-experiments that would carry the claim are all deferred
    # and unmeasured — there is no paired outcome to dress as a win.
    for r in dispositions.load():
        if r["id"].startswith("I"):
            assert r["disposition"] == "defer"
            assert r["measured"] == "unmeasured"
            assert r["recommended"] is False
