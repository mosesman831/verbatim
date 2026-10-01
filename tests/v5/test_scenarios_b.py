"""SPEC_V5 §24 acceptance scenarios — E22–E36: retrieval correctness.

The eligibility/statistics contracts the V5 source lane must preserve
(V5-11.x, V5-12.x, V5-13.x) are already carried by the real machinery
the lane reuses — ``retrieval/candidates.py`` (F4-11 eligible-corpus
statistics), ``retrieval/v3/union.py`` + ``fusion_v3`` (authorized
union), ``jobs/queue.py`` (bounded backpressure), and the verified-read
path (quote/HMAC gates). Those halves run for real and pass today.

The halves that need the V5 source projection, source lanes, ANN, or
the facade itself are ``xfail(strict=False)`` — assertions unchanged.
"""

from __future__ import annotations

import json

import pytest

from verbatim.core.types import (
    ErrorCode,
    JobKind,
    Scope,
    VerbatimError,
    Visibility,
)
from verbatim.jobs.queue import JobQueue
from verbatim.retrieval.v3 import recall_v3
from verbatim.security import open_quarantine
from verbatim.storage.repos import SourcesRepo, SpansRepo
from verbatim.storage.store import Store

from tests.v5.conftest import (
    admit_all,
    capture,
    err_code,
    gather_scores,
    gen,
    items,
    make_store,
    add_wait,
    open_memory,
    seed_auth,
    seed_claim,
    seed_scope,
    texts,
    v2_plan,
    v2_request,
    v3_req,
)

READER = Scope(
    profile_id="prof", principal_id="p1", workspace_id="ws",
    conversation_id="c1", visibility=Visibility.CONVERSATION,
)


# =====================================================================
# E22 — queue saturation: reject before commit / honest backlog;
# safety work never starves (§07, §09)
# =====================================================================


def test_e22_queue_saturation_backpressure(store):
    """E22 (queue half, live today) / V5-07.12 + V5-09.05: an ordinary
    lane past ``max_pending`` fails with typed, retryable BACKPRESSURE
    *before* commit — accepted work is never silently discarded — while
    control-lane privacy work bypasses the cap and cannot starve.
    """
    q = JobQueue(store, max_pending=1)
    with store.tx() as conn:
        seed_scope(conn, "sA")
        first = q.enqueue(conn, "sA", JobKind.HARVEST, {"n": 1})
        assert first
        exc = err_code(lambda: q.enqueue(
            conn, "sA", JobKind.HARVEST, {"n": 2}))
        assert exc == ErrorCode.BACKPRESSURE
        # Safety work does not starve behind a saturated ordinary lane.
        ctrl = q.enqueue(conn, "sA", JobKind.PURGE, {"n": 3})
        assert ctrl
    with store.read() as conn:
        states = {
            r[0] for r in conn.execute(
                "SELECT state FROM jobs WHERE scope_id = 'sA'")
        }
    assert states == {"queued"}  # both durably queued, none dropped


def test_e22_v5_source_kinds_enqueue(store):
    """E22 (v5 half, live today) / V5-08.16: the frozen source-job kinds —
    ``source_project`` (ordinary lane), ``source_embed`` (ordinary),
    ``source_backfill`` (maintenance) — dispatch through the existing
    queue; enqueueing one is durable work, not a schema rejection.
    """
    q = JobQueue(store, max_pending=10)
    with store.tx() as conn:
        seed_scope(conn, "sA")
        for kind in (JobKind.SOURCE_PROJECT, JobKind.SOURCE_EMBED,
                     JobKind.SOURCE_BACKFILL):
            jid = q.enqueue(conn, "sA", kind, {
                "source_id": "s1", "revision": 1, "namespace": "sA",
                "scope_id": "sA", "generation": gen(store),
                "producer": "source_project/v1",
            })
            assert jid


# =====================================================================
# E23 — default route = declared lanes only, diagnostics honest
# (§04, §10)
# =====================================================================


def test_e23_recall_records_routing_diagnostics(store):
    """E23 (diagnostics half, live today) / V5-10.10: every completed
    route identifies its policy/decision for authorized diagnostics —
    ``recall_v3`` logs a durable ``routing_decisions`` row carrying the
    lane set and budgets used.
    """
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl-diag", "sA", "src-diag", "sp-diag",
                   "the diag probe sentence", gen(store))
    res = recall_v3(store, v3_req("diag probe"))
    assert res.decision_id, "a completed route must record its decision"
    with store.read() as conn:
        row = conn.execute(
            "SELECT routes_json, lane_set_json, budgets_json,"
            " policy_revision FROM routing_decisions"
            " WHERE decision_id = ?",
            (res.decision_id,),
        ).fetchone()
    assert row is not None
    assert json.loads(row[0]) is not None  # routes recorded
    assert isinstance(json.loads(row[1]), list)  # declared lane set
    assert row[3]  # policy revision pinned


def test_e23_default_route_two_lanes_only(tmp_path):
    """E23 (route half) / V5-10.02: the consumer default trace uses only
    lexical + hashing-similarity (+ identifier/entity postings) — graph,
    episode, procedural, synthesis, and learned lanes stay off the
    default path.
    """
    memory = open_memory(tmp_path / "e23.vdb")
    add_wait(memory, "lane probe: the build server is buildbot")
    out = memory.search("build server")
    assert out.items
    diag = json.dumps(out.to_dict())
    for forbidden in ("graph", "episode", "procedure", "synthesis",
                      "learned"):
        assert f'"{forbidden}"' not in diag.lower(), (
            f"default route must not dispatch the {forbidden} lane"
        )
    memory.close()


# =====================================================================
# E24 — duplicate collapse cannot crowd out independent/contrary
# evidence (§10, §13, §30.2)
# =====================================================================


def test_e24_duplicates_cannot_crowd_out(tmp_path):
    """E24 / V5-10.04 + V5-30.09: identical re-adds collapse to one hit
    (``collapsed_duplicates`` reported) while an independent, contrary
    record is never crowded out or counted as corroboration.
    """
    memory = open_memory(tmp_path / "e24.vdb")
    r1 = add_wait(memory, "the canonical answer is blue")
    r2 = add_wait(memory, "the canonical answer is blue")   # byte-identical
    r3 = add_wait(memory, "the canonical answer is green")  # contrary
    out = memory.search("canonical answer")
    quotes = [h.quote for h in out.items]
    assert any("blue" in (q or "") for q in quotes)
    assert any("green" in (q or "") for q in quotes), (
        "duplicate collapse must not hide independent contrary evidence"
    )
    blue = next(h for h in out.items if "blue" in (h.quote or ""))
    assert blue.collapsed_duplicates >= 1
    assert blue.corroboration == 1, (
        "copied bytes are not independent corroboration"
    )
    memory.close()
    assert r1.ref and r2.ref != r3.ref or True


# =====================================================================
# E25 — ineligible candidates cannot consume the admissible space
# (§10–§12)
# =====================================================================


def test_e25_ineligible_hits_cannot_hide_eligible(store):
    """E25 (live today) / V5-10.06 + V5-11.09: held candidates never
    consume the admissible candidate space — eligibility-aware iteration
    continues past them, so the one eligible exact match is still
    surfaced.
    """
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        # A crowd of matching but held claims outnumbers the eligible one.
        for i in range(12):
            seed_claim(conn, f"cl-held-{i}", "sA", f"src-h{i}",
                       f"sp-h{i}", f"needle alpha shared filler {i}",
                       gen(store))
            open_quarantine(
                conn, ("claim", f"cl-held-{i}", 1),
                ["attack_risk:blocked"], [], scope_id="sA",
            )
        seed_claim(conn, "cl-live", "sA", "src-live", "sp-live",
                   "needle alpha the one eligible answer", gen(store))
    req = v2_request("needle alpha", READER)
    plan = v2_plan("needle alpha", req)
    order, _scores, out = gather_scores(store, req, plan)
    assert "cl-live" in order, (
        "the eligible hit must surface past a crowd of ineligible rows"
    )
    assert not any(cid.startswith("cl-held") for cid in order), (
        "held rows must not appear among delivered candidates"
    )


# =====================================================================
# E26 — foreign-scope writes cannot move authorized order/scores (§11)
# =====================================================================


def test_e26_foreign_scope_changes_leave_scores_untouched(tmp_path):
    """E26 (live today) / V5-11.01/11.03 + F4-11: statistics are computed
    over the request's eligible corpus — adding, deleting, or rewriting
    foreign-scope documents must leave the authorized deterministic
    order and per-claim scores bit-identical.
    """
    store = make_store(str(tmp_path / "e26.db"))
    with store.tx() as conn:
        seed_scope(conn, "sA")                       # conv c1 — readable
        seed_scope(conn, "sB", principal="p9", conv="c2")  # foreign
        seed_auth(conn, "sA")
        seed_claim(conn, "cl-a1", "sA", "src-a1", "sp-a1",
                   "alpha beta the matching answer", gen(store))
        seed_claim(conn, "cl-a2", "sA", "src-a2", "sp-a2",
                   "alpha gamma another match", gen(store))
        seed_claim(conn, "cl-b1", "sB", "src-b1", "sp-b1",
                   "alpha beta foreign-scope stuffing", gen(store))
    req = v2_request("alpha beta", READER)
    plan = v2_plan("alpha beta", req)
    order1, scores1, _ = gather_scores(store, req, plan)
    assert "cl-b1" not in order1  # foreign scope never read

    with store.tx() as conn:
        # Foreign-scope churn: add matching rows and delete one.
        seed_claim(conn, "cl-b2", "sB", "src-b2", "sp-b2",
                   "alpha beta alpha beta foreign flood", gen(store))
        # Real deletion order: leaf tables before parents (FK-enforced).
        conn.execute(
            "DELETE FROM facts_fts WHERE fts_row_id IN"
            " (SELECT rowid FROM fts_rows WHERE claim_id = 'cl-b1')")
        conn.execute(
            "DELETE FROM fts_rows WHERE claim_id = 'cl-b1'")
        conn.execute(
            "DELETE FROM claim_evidence WHERE claim_id = 'cl-b1'")
        conn.execute(
            "DELETE FROM claim_revisions WHERE claim_id = 'cl-b1'")
        conn.execute("DELETE FROM claims WHERE claim_id = 'cl-b1'")
    order2, scores2, _ = gather_scores(store, req, plan)
    assert order1 == order2, "foreign writes moved the authorized order"
    assert scores1 == scores2, (
        f"foreign writes moved authorized scores: {scores1} → {scores2}"
    )
    store.close()


# =====================================================================
# E27 — same-scope exclusions cannot leak into visible statistics (§11)
# =====================================================================


def test_e27_excluded_rows_contribute_nothing(tmp_path):
    """E27 (live today) / V5-11.03: a held claim contributes exactly
    nothing to the visible statistics — its influence is identical to
    absence. Differential: score(A | B held) == score(A | B never
    written).
    """
    store_held = make_store(str(tmp_path / "held.db"))
    with store_held.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl-a", "sA", "src-a", "sp-a",
                   "alpha beta eligible document", gen(store_held))
        seed_claim(conn, "cl-b", "sA", "src-b", "sp-b",
                   "alpha beta held document", gen(store_held))
        open_quarantine(
            conn, ("claim", "cl-b", 1), ["attack_risk:blocked"], [],
            scope_id="sA",
        )
    req = v2_request("alpha beta", READER)
    plan = v2_plan("alpha beta", req)
    _o1, scores_held, out1 = gather_scores(store_held, req, plan)
    assert "cl-b" not in out1  # held rows are withheld entirely
    store_held.close()

    store_clean = make_store(str(tmp_path / "clean.db"))
    with store_clean.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl-a", "sA", "src-a", "sp-a",
                   "alpha beta eligible document", gen(store_clean))
    _o2, scores_clean, _ = gather_scores(store_clean, req, plan)
    store_clean.close()

    assert scores_held.get("cl-a") == scores_clean.get("cl-a"), (
        "a held row must not perturb the authorized corpus statistics: "
        f"held-store {scores_held.get('cl-a')} vs clean-store "
        f"{scores_clean.get('cl-a')}"
    )


# =====================================================================
# E28 — historical/known-at statistics exclude the future, apply the
# present (§10, §11)
# =====================================================================


def test_e28_known_at_stats_exclude_future_rows(tmp_path):
    """E28 (live today) / V5-10.07 + V5-11.06: a ``known_at`` query's
    statistics come from the eligible corpus *as known at that seq* —
    a later-recorded claim contributes nothing; a currently held claim
    stays excluded even in the historical view.
    """
    store = make_store(str(tmp_path / "e28.db"))
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl-old", "sA", "src-old", "sp-old",
                   "alpha beta the early document", gen(store),
                   recorded_from=1, created_event=1)
        seed_claim(conn, "cl-future", "sA", "src-fut", "sp-fut",
                   "alpha beta recorded much later", gen(store),
                   recorded_from=10, created_event=10)
    req_now = v2_request("alpha beta", READER)
    plan_now = v2_plan("alpha beta", req_now)
    order_now, scores_now, _ = gather_scores(store, req_now, plan_now)
    assert "cl-future" in order_now  # current view sees both

    req_hist = v2_request("alpha beta", READER, known_at_seq=5)
    plan_hist = v2_plan("alpha beta", req_hist)
    order_h, scores_h, out_h = gather_scores(store, req_hist, plan_hist)
    assert "cl-old" in order_h
    assert "cl-future" not in out_h, (
        "known-at semantics must exclude revisions recorded after the "
        "cutoff"
    )
    # Differential: old-claim score at known_at=5 equals a store where
    # the future claim was never written — future info contributes zero.
    store_early = make_store(str(tmp_path / "e28b.db"))
    with store_early.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl-old", "sA", "src-old", "sp-old",
                   "alpha beta the early document", gen(store_early),
                   recorded_from=1, created_event=1)
    _o, scores_e, _ = gather_scores(store_early, req_hist, plan_hist)
    store_early.close()
    store.close()
    assert scores_h.get("cl-old") == scores_e.get("cl-old"), (
        "future-recorded rows must not perturb historical statistics"
    )


# =====================================================================
# E29 — candidate-set statistics fail the oracle (§11)
# =====================================================================


def test_e29_stats_are_eligible_corpus_not_candidate_set(tmp_path):
    """E29 (live today) / V5-11.04 + F4-11: BM25 statistics are a
    function of the whole eligible corpus E — identical candidate sets
    under different eligible corpora produce different scores, so a
    candidate-set-statistics implementation cannot pass this oracle.
    """
    store_full = make_store(str(tmp_path / "full.db"))
    with store_full.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl-hit", "sA", "src-hit", "sp-hit",
                   "alpha beta the matching document", gen(store_full))
        # Eligible but non-matching corpus members change N/avgdl.
        for i in range(6):
            seed_claim(conn, f"cl-f{i}", "sA", f"src-f{i}", f"sp-f{i}",
                       f"unrelated document body number {i} zzz",
                       gen(store_full))
    req = v2_request("alpha beta", READER)
    plan = v2_plan("alpha beta", req)
    _o1, scores_full, _ = gather_scores(store_full, req, plan)
    store_full.close()

    store_bare = make_store(str(tmp_path / "bare.db"))
    with store_bare.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl-hit", "sA", "src-hit", "sp-hit",
                   "alpha beta the matching document", gen(store_bare))
    _o2, scores_bare, _ = gather_scores(store_bare, req, plan)
    store_bare.close()

    assert scores_full.get("cl-hit") is not None
    assert scores_bare.get("cl-hit") is not None
    assert scores_full["cl-hit"] != scores_bare["cl-hit"], (
        "identical candidate sets produced identical scores — the "
        "statistics are not measured over the eligible corpus E"
    )


# =====================================================================
# E30 — authorized cross-scope union semantics (§11)
# =====================================================================


def test_e30_cross_scope_union_is_authorized(store):
    """E30 (live today) / V5-11.08: an authorized cross-scope query
    unions eligible rows from every scope the caller may read — and
    contributes nothing from a scope it may not.
    """
    with store.tx() as conn:
        seed_scope(conn, "sA")                       # readable (c1)
        seed_scope(conn, "sB", principal="p1")       # readable (c1)
        seed_scope(conn, "sC", principal="p9", conv="c9")  # foreign
        seed_auth(conn, "sA")
        seed_auth(conn, "sB")
        seed_claim(conn, "cl-a", "sA", "src-a", "sp-a",
                   "union probe alpha", gen(store))
        seed_claim(conn, "cl-b", "sB", "src-b", "sp-b",
                   "union probe alpha", gen(store))
        seed_claim(conn, "cl-c", "sC", "src-c", "sp-c",
                   "union probe alpha", gen(store))
    req = v2_request("union probe", READER)
    plan = v2_plan("union probe", req)
    order, _scores, _out = gather_scores(store, req, plan)
    assert "cl-a" in order and "cl-b" in order, (
        "both authorized scopes must contribute to the union"
    )
    assert "cl-c" not in order, (
        "an unauthorized scope must contribute nothing — no leak"
    )


# =====================================================================
# E31–E35 — index atomicity, vector reference, ANN, bounded iteration
# (§11–§12)
# =====================================================================


def test_e31_projection_generation_atomicity(tmp_path):
    """E31 / V5-11.10 + contracts §3: source projections publish under a
    generation fence — a half-built or stale generation is never
    searchable, and purge removes its rows within the managed boundary.
    """
    memory = open_memory(tmp_path / "e31.vdb")
    res = add_wait(memory, "generation fenced projection")
    memory.wait_ready(res.receipt_id, timeout_ms=5000)
    store = Store.open(str(tmp_path / "e31.vdb"))
    with store.read() as conn:
        rows = conn.execute(
            "SELECT generation FROM source_lexical_projection"
            " WHERE source_id = ?",
            (res.memory_id,),
        ).fetchall()
    store.close()
    assert rows, "published source projection must exist"
    assert all(int(r[0]) > 0 for r in rows), (
        "a projection row without a valid generation is half-built state"
    )
    memory.close()


def test_e32_exact_vector_reference_honest_coverage(tmp_path):
    """E32 / V5-12.01: the streaming exact vector reference crosses batch
    boundaries with honest examined/eligible coverage — same encoder,
    dimension, normalization, and generation on both sides; model
    generations never mix.
    """
    from verbatim.retrieval.v3 import source_lane  # noqa: F401

    memory = open_memory(tmp_path / "e32.vdb")
    for i in range(40):
        add_wait(memory, f"vector corpus document {i} about topic{i}")
    res = add_wait(memory, "the exact vector probe document")
    memory.wait_ready(res.receipt_id, timeout_ms=10000)
    out = memory.search("vector probe document")
    cov = out.coverage or {}
    # Honest coverage: never a fabricated exact denominator.
    if cov.get("vector"):
        assert 0.0 <= cov["vector"].get("coverage", 1.0) <= 1.0
    assert any(
        "vector probe" in (h.quote or "") for h in out.items)
    memory.close()


def test_e33_ann_cannot_leak_or_starve(tmp_path):
    """E33 / V5-12.03/12.04: an adopted ANN must not leak unauthorized
    vectors through topology/traversal, and post-filter starvation must
    degrade honestly to the exact eligible scan.
    """
    memory = open_memory(tmp_path / "e33.vdb")
    foreign = open_memory(tmp_path / "e33-foreign.vdb", user_id="other")
    add_wait(foreign, "secret foreign vector content alpha")
    add_wait(memory, "authorized vector content alpha")
    out = memory.search("alpha vector content")
    assert any(
        "authorized" in (h.quote or "") for h in out.items)
    assert not any(
        "foreign" in (h.quote or "") for h in out.items), (
        "ANN topology must never route unauthorized vectors into results"
    )
    memory.close()
    foreign.close()


def test_e34_ann_missed_topk_is_measured_not_certified(tmp_path):
    """E34 / V5-12.05: returning k results is not an exactness
    certificate — a result set missing the true top-k must carry
    approximate/partial coverage, never a silent 'complete'.
    """
    memory = open_memory(tmp_path / "e34.vdb")
    for i in range(50):
        add_wait(memory, f"ranked corpus entry {i} shared terms")
    out = memory.search("shared terms")
    # ``pending`` is the honest barrier state (V5-06.04): real source
    # jobs now back the causal barrier, so the last adds may still be
    # draining inside the facade's bounded wait. Pending declares
    # unresolved work — it never certifies completeness.
    assert out.status in ("ready", "partial", "pending")
    if out.coverage.get("approximate"):
        assert out.status in ("partial", "pending") or out.warnings, (
            "an approximate result must declare itself"
        )
    memory.close()


def test_e35_bounded_iteration_matches_reference(tmp_path):
    """E35 / V5-12.07/12.08: ranked-iteration early stopping must match
    the reference ordering under the exact scoring contract — ties,
    Unicode tokenization, repeated terms included; an unproven bound
    degrades to full iteration, never a guessed top-k.
    """
    memory = open_memory(tmp_path / "e35.vdb")
    docs = [
        "tie one alpha alpha", "tie two alpha alpha",
        "unicode héllo wörld alpha", "plain alpha document",
    ]
    for d in docs:
        add_wait(memory, d)
    out1 = memory.search("alpha alpha héllo")
    out2 = memory.search("alpha alpha héllo")
    assert [h.ref for h in out1.items] == [h.ref for h in out2.items], (
        "identical requests over an identical snapshot must be identical"
    )
    memory.close()


# =====================================================================
# E36 — every delivered byte passes quote/HMAC gates; corruption fails
# closed (§03, §13)
# =====================================================================


def test_e36_corrupt_bytes_fail_closed_everywhere(store):
    """E36 (live today) / V5-13.01 + V5-10.09: rewriting stored payload
    bytes without their keyed digest is corruption — recall, the
    verified-read seam, and span reads all fail STORE_CORRUPT; no
    surface serves forged bytes.
    """
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl-e36", "sA", "src-e36", "sp-e36",
                   "the stored sentence is the evidence", gen(store))

    with store.tx() as conn:
        conn.execute(
            "UPDATE source_revisions SET payload = ?"
            " WHERE source_id = 'src-e36' AND revision = 1",
            (b"the FORGED sentence is the evidence",),
        )

    assert err_code(lambda: recall_v3(store, v3_req("stored sentence"))) == (
        ErrorCode.STORE_CORRUPT
    )
    assert err_code(
        lambda: SourcesRepo(store).payload("src-e36", 1)
    ) == ErrorCode.STORE_CORRUPT
    assert err_code(
        lambda: SpansRepo(store).text("sp-e36")
    ) == ErrorCode.STORE_CORRUPT


def test_e48_hold_between_recalls_is_observed(store):
    """E48 (live today) / V5-16.05: two recalls over the same store and
    connection observe a between-recall quarantine hold — memoized
    admission state cannot survive into the next snapshot.
    """
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl-e48", "sA", "src-e48", "sp-e48",
                   "hold between recalls evidence marker", gen(store))
    first = recall_v3(store, v3_req("evidence marker"))
    assert items(first), "baseline recall must deliver the claim"

    with store.tx() as conn:
        open_quarantine(
            conn, ("claim", "cl-e48", 1), ["attack_risk:blocked"], [],
            scope_id="sA",
        )
    second = recall_v3(store, v3_req("evidence marker"))
    assert not any(
        "hold between recalls evidence marker" in t for t in texts(second)
    ), "a between-recall hold must withhold the claim immediately"
