"""J09 / V75-03.05 — query-derived ``speaker_match`` in the S4 reranker.

``rerank_features.speaker_match`` previously fired only on the
caller-supplied ``speaker_hint`` (``QueryViewV7.speaker_canon``).  It now
also resolves which of the query's extracted entity canons are *speaker*
canons present in the queried scope (``units.speaker_canon`` at/below the
generation fence) — exactly one → 1.0 on that speaker's units, 0.0 on the
others; two or more → neutral 0.5 for all; zero → unchanged neutral.  A
caller hint always wins.  The resolution is a bounded feature only —
never an eligibility gate (V7-05.08/05.12).

The store side is the real §30 schema (``ensure_v7_additive``); only
``units`` is seeded.  The query side is the real S1 builder
(``build_query_view``) so the test exercises extraction → canon →
scope-speaker resolution end to end.
"""

from __future__ import annotations

import sqlite3

from verbatim.core.types_v7 import (
    BudgetClass,
    FusedCandidate,
    LaneContextV7,
    LaneName,
    RetrievalPolicyV7,
)
from verbatim.querying.query_view import build_query_view
from verbatim.retrieval.v7.entity import resolve_query_speaker_canons
from verbatim.retrieval.v7.rerank_features import (
    FeatureProviders,
    make_context,
    score_candidates,
)
from verbatim.storage.schema_v7 import ensure_v7_additive

SCOPE = "scope-a"
GEN = 1
T0 = 1_700_000_000_000_000

#: Scope vocabulary — both speakers plus one non-speaker entity.
KNOWN = ("caroline", "jon", "promotion")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def mk_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_v7_additive(conn)
    return conn


def add_unit(conn, uid, *, speaker=None, scope=SCOPE, gen=GEN, seq=None):
    conn.execute(
        "INSERT OR REPLACE INTO units (unit_id, source_id, revision,"
        " scope_id, kind, parent_unit_id, session_id, seq, speaker_canon,"
        " perspective, recorded_at_us, occurred_start_us, occurred_end_us,"
        " occurred_precision, occurred_source, byte_start, byte_end,"
        " generation) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            uid, f"src-{uid}", 1, scope, "turn", None, None, seq,
            speaker, "user_stated", T0, None, None, "unknown", "unknown",
            None, None, gen,
        ),
    )


def mk_ctx(conn, *, scope=SCOPE, gen=GEN) -> LaneContextV7:
    return LaneContextV7(
        store=conn,
        scope_id=scope,
        generation=gen,
        eligible=lambda row: True,  # noqa: E731 — allow-all fixture
        query_time_us=T0,
        profile="test",
        budget=BudgetClass.MID,
        policy=RetrievalPolicyV7(
            policy_id="retrieval_policy/v7",
            profile="test",
            lanes=(LaneName.LEX,),
            lane_weights={},
        ),
        manifest={},
    )


def mk_fused() -> list:
    """Two Caroline turns, one Jon turn, one unit with no speaker."""
    return [
        FusedCandidate(
            unit_id=u, source_id="src", revision=1,
            rrf=0.9 - i * 0.01, lane_ranks={"lex": i + 1},
            signals={"speaker_canon": sp} if sp else {},
        )
        for i, (u, sp) in enumerate(
            [("u_c1", "caroline"), ("u_j1", "jon"), ("u_c2", "caroline"),
             ("u_none", None)]
        )
    ]


def qv(query: str, **kw):
    return build_query_view(query, now_us=T0, known_canons=KNOWN, **kw)


def speaker_map(scored) -> dict:
    return {
        s.unit_id: s.detail["features"]["speaker_match"] for s in scored
    }


# ---------------------------------------------------------------------------
# J09 — the acceptance scenario
# ---------------------------------------------------------------------------


def test_j09_single_query_speaker_marks_her_turns():
    """"Where does Caroline live?" — Caroline is a speaker canon in
    scope: her turns get 1.0, the other speaker's 0.0."""
    conn = mk_conn()
    add_unit(conn, "u_c1", speaker="caroline")
    add_unit(conn, "u_c2", speaker="caroline")
    add_unit(conn, "u_j1", speaker="jon")

    view = qv("Where does Caroline live?")
    assert "caroline" in view.entity_canons
    assert view.speaker_canon is None  # no caller hint

    scored = score_candidates(view, mk_fused(), ctx=mk_ctx(conn))
    sm = speaker_map(scored)
    assert sm["u_c1"] == 1.0
    assert sm["u_c2"] == 1.0
    assert sm["u_j1"] == 0.0
    assert sm["u_none"] == 0.5  # unknown unit speaker → indeterminate
    assert scored.stats["speaker_match_source"] == "query"
    assert scored.stats["speaker_match_canon"] == "caroline"
    assert scored.stats["speaker_match_query_canons"] == ["caroline"]
    for s in scored:
        assert s.detail["speaker_match_source"] == "query"


def test_j09_two_named_speakers_are_neutral():
    """A question naming both speakers → 0.5 for every unit, never a
    guess; the source label reports the ambiguity honestly."""
    conn = mk_conn()
    add_unit(conn, "u_c1", speaker="caroline")
    add_unit(conn, "u_j1", speaker="jon")

    view = qv("What did Caroline and Jon discuss?")
    assert {"caroline", "jon"} <= set(view.entity_canons)

    scored = score_candidates(view, mk_fused(), ctx=mk_ctx(conn))
    sm = speaker_map(scored)
    assert sm == {u: 0.5 for u in sm}
    assert scored.stats["speaker_match_source"] == "ambiguous"
    assert scored.stats["speaker_match_canon"] is None
    assert scored.stats["speaker_match_query_canons"] == ["caroline", "jon"]
    for s in scored:
        assert s.detail["speaker_match_source"] == "ambiguous"


def test_j09_no_speaker_in_query_behaves_as_before():
    """A query naming no speaker keeps today's neutral/absent shape."""
    conn = mk_conn()
    add_unit(conn, "u_c1", speaker="caroline")
    add_unit(conn, "u_j1", speaker="jon")

    view = qv("Where does the team meet?")
    assert not ({"caroline", "jon"} & set(view.entity_canons))

    scored = score_candidates(view, mk_fused(), ctx=mk_ctx(conn))
    sm = speaker_map(scored)
    assert sm == {u: 0.5 for u in sm}
    assert scored.stats["speaker_match_source"] == "none"
    assert scored.stats["speaker_match_canon"] is None


def test_caller_hint_overrides_query_derivation():
    """speaker_hint wins even when the query names a different speaker —
    and even when the hint's canon is not a scope speaker."""
    conn = mk_conn()
    add_unit(conn, "u_c1", speaker="caroline")
    add_unit(conn, "u_j1", speaker="jon")

    view = qv("Where does Caroline live?", speaker_hint="Jon")
    assert view.speaker_canon == "jon"

    scored = score_candidates(view, mk_fused(), ctx=mk_ctx(conn))
    sm = speaker_map(scored)
    assert sm["u_j1"] == 1.0
    assert sm["u_c1"] == 0.0
    assert sm["u_c2"] == 0.0
    assert scored.stats["speaker_match_source"] == "hint"
    assert scored.stats["speaker_match_canon"] == "jon"
    for s in scored:
        assert s.detail["speaker_match_source"] == "hint"


def test_hint_not_a_scope_speaker_still_wins():
    conn = mk_conn()
    add_unit(conn, "u_c1", speaker="caroline")
    view = qv("Where does Caroline live?", speaker_hint="Zelda")
    scored = score_candidates(view, mk_fused(), ctx=mk_ctx(conn))
    sm = speaker_map(scored)
    assert sm["u_c1"] == 0.0  # hint honored verbatim — no speaker matches
    assert scored.stats["speaker_match_source"] == "hint"
    assert scored.stats["speaker_match_canon"] == "zelda"


# ---------------------------------------------------------------------------
# resolution semantics
# ---------------------------------------------------------------------------


def test_non_speaker_entity_canon_does_not_fire():
    """"promotion" is a scope entity canon but never a speaker → none."""
    conn = mk_conn()
    add_unit(conn, "u_c1", speaker="caroline")
    view = qv("What is the promotion timeline?")
    assert "promotion" in view.entity_canons
    scored = score_candidates(view, mk_fused(), ctx=mk_ctx(conn))
    assert scored.stats["speaker_match_source"] == "none"
    assert all(
        s.detail["features"]["speaker_match"] == 0.5 for s in scored
    )


def test_possessive_surface_folds_to_speaker_canon():
    """"Caroline's" resolves to canon "caroline" — same fold the lane
    applies (entities_v2.canon, D7-01/D7-02)."""
    conn = mk_conn()
    add_unit(conn, "u_c1", speaker="caroline")
    add_unit(conn, "u_j1", speaker="jon")
    view = qv("What did Caroline's report say?")
    scored = score_candidates(view, mk_fused(), ctx=mk_ctx(conn))
    assert scored.stats["speaker_match_source"] == "query"
    assert scored.stats["speaker_match_canon"] == "caroline"
    sm = speaker_map(scored)
    assert sm["u_c1"] == 1.0
    assert sm["u_j1"] == 0.0


def test_generation_fence_excludes_future_speakers():
    """A speaker canon only visible above the query's fence does not
    resolve — the probe honors the pinned snapshot's generation."""
    conn = mk_conn()
    add_unit(conn, "u_c1", speaker="caroline", gen=2)  # above GEN fence
    add_unit(conn, "u_j1", speaker="jon", gen=GEN)
    view = qv("Where does Caroline live?")
    scored = score_candidates(view, mk_fused(), ctx=mk_ctx(conn, gen=GEN))
    assert scored.stats["speaker_match_source"] == "none"
    # And at gen=2 it resolves.
    scored2 = score_candidates(view, mk_fused(), ctx=mk_ctx(conn, gen=2))
    assert scored2.stats["speaker_match_source"] == "query"
    assert scored2.stats["speaker_match_canon"] == "caroline"


def test_cross_scope_speakers_never_resolve():
    conn = mk_conn()
    add_unit(conn, "u_x", speaker="caroline", scope="scope-b")
    view = qv("Where does Caroline live?")
    scored = score_candidates(view, mk_fused(), ctx=mk_ctx(conn))
    assert scored.stats["speaker_match_source"] == "none"


def test_feature_never_gates_candidates():
    """Bounded feature only (V7-05.08/05.12): every fused candidate is
    still scored and returned — none removed, none re-eligible-checked."""
    conn = mk_conn()
    add_unit(conn, "u_c1", speaker="caroline")
    add_unit(conn, "u_j1", speaker="jon")
    fused = mk_fused()
    scored = score_candidates(
        qv("Where does Caroline live?"), fused, ctx=mk_ctx(conn)
    )
    assert len(scored) == len(fused)
    assert {s.unit_id for s in scored} == {c.unit_id for c in fused}


def test_deterministic_repeat_scoring():
    conn = mk_conn()
    add_unit(conn, "u_c1", speaker="caroline")
    add_unit(conn, "u_j1", speaker="jon")
    view = qv("Where does Caroline live?")
    a = score_candidates(view, mk_fused(), ctx=mk_ctx(conn))
    b = score_candidates(view, mk_fused(), ctx=mk_ctx(conn))
    assert [s.unit_id for s in a] == [s.unit_id for s in b]
    assert [s.score for s in a] == [s.score for s in b]


# ---------------------------------------------------------------------------
# wiring / back-compat
# ---------------------------------------------------------------------------


def test_no_ctx_keeps_prior_neutral_behavior():
    """No lane ctx and no providers → no store to resolve against → the
    pre-V75-03.05 behavior: neutral 0.5, source ``none``."""
    view = qv("Where does Caroline live?")
    scored = score_candidates(view, mk_fused())
    sm = speaker_map(scored)
    assert sm == {u: 0.5 for u in sm}
    assert scored.stats["speaker_match_source"] == "none"


def test_explicit_provider_wins_over_lane_ctx():
    """An injected ``query_speakers`` provider takes precedence over the
    lane-ctx synthesis — the injection seam stays authoritative."""
    conn = mk_conn()
    add_unit(conn, "u_c1", speaker="caroline")
    add_unit(conn, "u_j1", speaker="jon")
    prov = FeatureProviders(query_speakers=lambda q: ("zelda",))
    view = qv("Where does Caroline live?")
    scored = score_candidates(
        view, mk_fused(), ctx=mk_ctx(conn), providers=prov
    )
    assert scored.stats["speaker_match_source"] == "query"
    assert scored.stats["speaker_match_canon"] == "zelda"
    sm = speaker_map(scored)
    assert sm["u_c1"] == 0.0 and sm["u_j1"] == 0.0


def test_provider_via_plain_providers_arg_without_ctx():
    """The provider works with no lane ctx at all (evals/explain)."""
    prov = FeatureProviders(query_speakers=lambda q: ("caroline",))
    scored = score_candidates(
        qv("Where does Caroline live?"), mk_fused(), providers=prov
    )
    sm = speaker_map(scored)
    assert sm["u_c1"] == 1.0 and sm["u_j1"] == 0.0
    assert scored.stats["speaker_match_source"] == "query"


def test_provider_error_resolves_to_none_not_raise():
    def boom(q):
        raise RuntimeError("probe failed")

    prov = FeatureProviders(query_speakers=boom)
    scored = score_candidates(
        qv("Where does Caroline live?"), mk_fused(), providers=prov
    )
    assert scored.stats["speaker_match_source"] == "none"
    sm = speaker_map(scored)
    assert sm == {u: 0.5 for u in sm}


def test_make_context_reports_resolution():
    """The FeatureContext carries the resolved triple directly (the
    same object the pipeline uses for feature_vector explain calls)."""
    fctx = make_context(
        qv("Where does Caroline live?"),
        mk_fused(),
        providers=FeatureProviders(query_speakers=lambda q: ("caroline",)),
    )
    assert fctx.q_speaker == "caroline"
    assert fctx.q_speaker_source == "query"
    assert fctx.q_speakers == ("caroline",)


def test_resolve_query_speaker_canons_helper_directly():
    """The shared entity helper: fold, fence, dedup, honest empty."""
    conn = mk_conn()
    add_unit(conn, "u_c1", speaker="caroline")
    add_unit(conn, "u_j1", speaker="jon")
    assert resolve_query_speaker_canons(
        conn, SCOPE, GEN, ["Caroline's", "promotion"]
    ) == ["caroline"]
    assert resolve_query_speaker_canons(
        conn, SCOPE, GEN, ["caroline", "jon", "caroline"]
    ) == ["caroline", "jon"]
    assert resolve_query_speaker_canons(conn, SCOPE, GEN, []) == []
    assert resolve_query_speaker_canons(conn, "other-scope", GEN,
                                        ["caroline"]) == []
    # pre-migration store (no units table) → honest empty, never a guess
    bare = sqlite3.connect(":memory:")
    assert resolve_query_speaker_canons(bare, SCOPE, GEN, ["caroline"]) == []
