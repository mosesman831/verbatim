"""w-fusion tests: S3 RRF fusion + S4 feature reranker + S6 bounded boosts.

Hand-computed fixtures for every numeric claim (H13, H14, H16, H19, H20).
Constructs contract types directly — no sibling wave-A module imports.
"""

from __future__ import annotations

import math
import time

import pytest

from verbatim.core.types_v7 import (
    CandidateV7,
    FusedCandidate,
    IntentClass,
    IntentResult,
    IntervalUs,
    LaneName,
    LaneOutput,
    LaneStatus,
    NormAnalysis,
    NormTerm,
    OccurredPrecision,
    OccurredSource,
    QueryViewV7,
    ScoredCandidate,
)

from verbatim.retrieval.v7.fusion import (
    FUSION_MODEL_ID,
    RRF_K,
    FusedList,
    detect_constant_signals,
    rrf_fuse,
    score_detail,
)
from verbatim.retrieval.v7.rerank_features import (
    FEATURE_WEIGHTS_V1,
    RERANK_MODEL_ID,
    SCORE_FAMILY,
    TIE_EPSILON,
    FeatureProviders,
    ScoredList,
    score_candidates,
)
from verbatim.retrieval.v7.boosts import (
    ALPHA_PROOF,
    ALPHA_PROXIMITY,
    ALPHA_RECENCY,
    BOOSTS_MODEL_ID,
    SWING_MAX,
    SWING_MIN,
    apply_boosts,
)

DAY_US = 86_400_000_000
NOW_US = 1_700_000_000_000_000  # fixed epoch μs for all fixtures


def _cand(unit, lane, rank, source=None, rev=1, raw=1.0, signals=None):
    return CandidateV7(
        unit_id=unit,
        source_id=source or f"src-{unit}",
        revision=rev,
        lane=lane,
        rank=rank,
        raw_score=raw,
        signals=signals or {},
    )


def _lane(name, cands, status=LaneStatus.OK):
    return LaneOutput(lane=name, status=status, candidates=list(cands))


def _query(text="alpha beta", terms=None, intent=IntentClass.LOOKUP,
           window=None, canons=(), speaker=None, identifiers=()):
    terms = terms if terms is not None else [
        NormTerm(term=t, channel="text", byte_start=0, byte_end=len(t))
        for t in text.split()
    ]
    norm = NormAnalysis(
        analyzer_id="norm/v2",
        terms=tuple(terms),
        identifiers=tuple(identifiers),
        text=text,
    )
    return QueryViewV7(
        query=text,
        norm=norm,
        intent=IntentResult(primary=intent, classes=(intent,), window=window),
        entity_canons=tuple(canons),
        speaker_canon=speaker,
        query_time_us=NOW_US,
    )


def _fused(unit, rrf, lane_ranks, source=None, rev=1, signals=None):
    return FusedCandidate(
        unit_id=unit,
        source_id=source or f"src-{unit}",
        revision=rev,
        rrf=rrf,
        lane_ranks=dict(lane_ranks),
        signals=signals or {},
    )


def _scored(unit, score, source=None, rev=1, detail=None):
    return ScoredCandidate(
        unit_id=unit,
        source_id=source or f"src-{unit}",
        revision=rev,
        score=score,
        score_family=SCORE_FAMILY,
        detail=detail or {},
    )


# ---------------------------------------------------------------------------
# S3 RRF fusion
# ---------------------------------------------------------------------------


class TestRrfFuse:
    def test_hand_computed_equal_weights(self):
        """H16: rrf(d) = Σ w/(60+rank); hand-computed on a fixed fixture."""
        out = rrf_fuse([
            _lane("lex", [_cand("a", "lex", 1), _cand("b", "lex", 2)]),
            _lane("dense", [_cand("b", "dense", 1), _cand("c", "dense", 2)]),
        ], {})
        assert [c.unit_id for c in out] == ["b", "a", "c"]
        assert math.isclose(out[0].rrf, 1 / 61 + 1 / 62, rel_tol=1e-15)
        assert math.isclose(out[1].rrf, 1 / 61, rel_tol=1e-15)
        assert math.isclose(out[2].rrf, 1 / 62, rel_tol=1e-15)
        # provenance retained (V7-10.05)
        assert out[0].lane_ranks == {"dense": 1, "lex": 2}
        assert out[1].lane_ranks == {"lex": 1}
        assert out[2].lane_ranks == {"dense": 2}

    def test_hand_computed_weighted(self):
        out = rrf_fuse([
            _lane("lex", [_cand("a", "lex", 1), _cand("b", "lex", 2)]),
            _lane("dense", [_cand("b", "dense", 1), _cand("c", "dense", 2)]),
        ], {"dense": 2.0})
        assert [c.unit_id for c in out] == ["b", "c", "a"]
        assert math.isclose(out[0].rrf, 1 / 62 + 2 / 61, rel_tol=1e-15)
        assert math.isclose(out[1].rrf, 2 / 62, rel_tol=1e-15)
        assert math.isclose(out[2].rrf, 1 / 61, rel_tol=1e-15)
        # LaneName-enum keys normalize identically
        out2 = rrf_fuse([
            _lane("lex", [_cand("a", "lex", 1), _cand("b", "lex", 2)]),
            _lane("dense", [_cand("b", "dense", 1), _cand("c", "dense", 2)]),
        ], {LaneName.DENSE: 2.0})
        assert [(c.unit_id, c.rrf) for c in out2] == [(c.unit_id, c.rrf) for c in out]

    def test_absent_lane_contributes_zero(self):
        out = rrf_fuse([
            _lane("lex", [_cand("a", "lex", 1)]),
            _lane("dense", []),
        ], {})
        assert len(out) == 1
        assert math.isclose(out[0].rrf, 1 / 61, rel_tol=1e-15)

    def test_custom_k(self):
        out = rrf_fuse([_lane("lex", [_cand("a", "lex", 3)])], {}, k=10)
        assert math.isclose(out[0].rrf, 1 / 13, rel_tol=1e-15)

    def test_missing_lane_weight_defaults_to_one(self):
        out = rrf_fuse(
            [_lane("graph", [_cand("a", "graph", 1)])],
            {"lex": 5.0},
        )
        assert math.isclose(out[0].rrf, 1 / 61, rel_tol=1e-15)

    def test_zero_weight_lane_contributes_nothing(self):
        out = rrf_fuse([
            _lane("lex", [_cand("a", "lex", 1)]),
            _lane("dense", [_cand("a", "dense", 1)]),
        ], {"lex": 1.0, "dense": 0.0})
        assert math.isclose(out[0].rrf, 1 / 61, rel_tol=1e-15)
        # provenance still records the lane rank
        assert out[0].lane_ranks == {"dense": 1, "lex": 1}

    def test_degraded_lane_candidates_ignored(self):
        out = rrf_fuse([
            _lane("lex", [_cand("a", "lex", 1)]),
            _lane("dense", [_cand("z", "dense", 1)], status=LaneStatus.UNAVAILABLE),
            _lane("graph", [_cand("y", "graph", 1)], status=LaneStatus.SKIPPED),
            _lane("time", [_cand("x", "time", 1)], status=LaneStatus.DEADLINE),
        ], {})
        assert [c.unit_id for c in out] == ["a"]
        assert out.stats["lanes_ignored"] == {
            "dense": "unavailable", "graph": "skipped", "time": "deadline",
        }

    def test_partial_lane_contributes(self):
        out = rrf_fuse([
            _lane("lex", [_cand("a", "lex", 1)]),
            _lane("dense", [_cand("b", "dense", 1)], status=LaneStatus.PARTIAL),
        ], {})
        assert {c.unit_id for c in out} == {"a", "b"}

    def test_tie_break_deterministic(self):
        # Equal rrf -> (-rrf, source_id, unit_id)
        out = rrf_fuse([
            _lane("lex", [
                _cand("u-b", "lex", 1, source="s-b"),
                _cand("u-a", "lex", 2, source="s-a"),
                _cand("u-c", "lex", 3, source="s-a"),
            ]),
            _lane("dense", [
                _cand("u-a", "dense", 1, source="s-a"),
                _cand("u-c", "dense", 2, source="s-a"),
            ]),
        ], {})
        # u-a: 1/62+1/61 ≈ .03252; u-c: 1/63+1/62 ≈ .03200; u-b: 1/61 ≈ .01639
        assert [c.unit_id for c in out] == ["u-a", "u-c", "u-b"]
        assert math.isclose(out[0].rrf, 1 / 62 + 1 / 61, rel_tol=1e-15)
        assert math.isclose(out[1].rrf, 1 / 63 + 1 / 62, rel_tol=1e-15)
        assert math.isclose(out[2].rrf, 1 / 61, rel_tol=1e-15)
        # force an exact tie: two units, same single rank on one lane
        tied = rrf_fuse([
            _lane("lex", [_cand("u-x", "lex", 1, source="s-b")]),
            _lane("dense", [_cand("u-y", "dense", 1, source="s-a")]),
        ], {})
        assert [c.unit_id for c in tied] == ["u-y", "u-x"]  # source_id asc

    def test_duplicate_within_lane_keeps_best_rank(self):
        out = rrf_fuse([
            _lane("lex", [_cand("a", "lex", 4), _cand("a", "lex", 1)]),
        ], {})
        # NOTE: first occurrence mints the entry; better rank wins.
        assert math.isclose(out[0].rrf, 1 / 61, rel_tol=1e-15)
        assert out[0].lane_ranks == {"lex": 1}
        assert out.stats["duplicates_in_lane"] == 1

    def test_lane_local_signals_namespaced_no_alias(self):
        """A 'score' signal on two lanes must not alias across lanes."""
        out = rrf_fuse([
            _lane("lex", [_cand("a", "lex", 1, signals={"score": 9.9, "matched_terms": ["alpha"]})]),
            _lane("dense", [_cand("a", "dense", 1, signals={"score": 0.77, "matched_terms": ["beta"]})]),
        ], {})
        sig = out[0].signals
        assert "score" not in sig  # never flattened — only under _lanes
        assert sig["_lanes"]["lex"]["signals"]["score"] == 9.9
        assert sig["_lanes"]["dense"]["signals"]["score"] == 0.77
        assert set(sig["matched_terms"]) == {"alpha", "beta"}  # set union

    def test_unit_facts_merge_flat(self):
        out = rrf_fuse([
            _lane("lex", [_cand("a", "lex", 1, signals={"speaker_canon": "mel", "seq": 4})]),
            _lane("ent", [_cand("a", "ent", 2, signals={"speaker_canon": "mel", "lifecycle": "current"})]),
        ], {})
        sig = out[0].signals
        assert sig["speaker_canon"] == "mel"
        assert sig["lifecycle"] == "current"
        assert sig["seq"] == 4

    def test_limit_reports_truncation(self):
        out = rrf_fuse(
            [_lane("lex", [_cand(f"u{i}", "lex", i + 1) for i in range(5)])],
            {}, limit=3,
        )
        assert len(out) == 3
        assert out.stats["truncated"] is True

    def test_stats_and_constant_signals_attached(self):
        out = rrf_fuse([
            _lane("lex", [_cand("a", "lex", 1, signals={"const": 1.0})]),
        ], {"lex": 1.0})
        assert out.stats["model"] == FUSION_MODEL_ID
        assert out.stats["k"] == RRF_K
        assert "const" in out.stats["constant_signals"]

    def test_score_detail_explain(self):
        out = rrf_fuse([
            _lane("lex", [_cand("a", "lex", 2)]),
            _lane("ent", [_cand("a", "ent", 1)]),
        ], {"ent": 0.8})
        d = score_detail(out[0], {"ent": 0.8})
        assert math.isclose(d["rrf"], 1 / 62 + 0.8 / 61, rel_tol=1e-15)
        assert math.isclose(d["contributions"]["ent"], 0.8 / 61, rel_tol=1e-15)
        assert math.isclose(d["contributions"]["lex"], 1 / 62, rel_tol=1e-15)
        assert d["lane_ranks"] == {"ent": 1, "lex": 2}

    def test_byte_stable_across_input_orderings(self):
        lanes_a = [
            _lane("lex", [_cand("a", "lex", 1), _cand("b", "lex", 2)]),
            _lane("dense", [_cand("b", "dense", 1), _cand("a", "dense", 2)]),
        ]
        lanes_b = [lanes_a[1], lanes_a[0]]
        r1 = rrf_fuse(lanes_a, {})
        r2 = rrf_fuse(lanes_b, {})
        assert [(c.unit_id, c.rrf) for c in r1] == [(c.unit_id, c.rrf) for c in r2]
        # exact-tie case: swapped lane order still identical order
        t1 = rrf_fuse([
            _lane("lex", [_cand("x", "lex", 1, source="s1")]),
            _lane("dense", [_cand("y", "dense", 1, source="s2")]),
        ], {})
        t2 = rrf_fuse([
            _lane("dense", [_cand("y", "dense", 1, source="s2")]),
            _lane("lex", [_cand("x", "lex", 1, source="s1")]),
        ], {})
        assert [c.unit_id for c in t1] == [c.unit_id for c in t2]


# ---------------------------------------------------------------------------
# constant-signal detection (V7-10.03 / H14)
# ---------------------------------------------------------------------------


class TestConstantSignals:
    def test_detects_modal_value_at_threshold(self):
        # 9 of 10 rows share value 7 → constant
        cands = [_cand(f"u{i}", "lex", i + 1, signals={"bias": 7}) for i in range(9)]
        cands.append(_cand("u9x", "lex", 10, signals={"bias": 3}))
        got = detect_constant_signals([_lane("lex", cands)], threshold=0.9)
        assert "bias" in got

    def test_below_threshold_not_constant(self):
        cands = [_cand(f"u{i}", "lex", i + 1, signals={"bias": 7}) for i in range(8)]
        cands += [
            _cand("u8x", "lex", 9, signals={"bias": 3}),
            _cand("u9x", "lex", 10, signals={"bias": 5}),
        ]
        got = detect_constant_signals([_lane("lex", cands)], threshold=0.9)
        assert "bias" not in got

    def test_absent_on_some_still_constant_when_modal_covers(self):
        # present on 9/10 rows with same value → constant
        cands = [_cand(f"u{i}", "lex", i + 1, signals={"sig": "x"}) for i in range(9)]
        cands.append(_cand("u9x", "lex", 10, signals={}))
        got = detect_constant_signals([_lane("lex", cands)], threshold=0.9)
        assert "sig" in got

    def test_varied_values_not_constant(self):
        cands = [
            _cand(f"u{i}", "lex", i + 1, signals={"sig": i})
            for i in range(10)
        ]
        assert detect_constant_signals([_lane("lex", cands)]) == set()

    def test_empty_outputs(self):
        assert detect_constant_signals([]) == set()

    def test_degraded_lane_rows_not_counted(self):
        out = [
            _lane("lex", [_cand("a", "lex", 1, signals={"s": 1})]),
            _lane("bad", [_cand("b", "bad", 1, signals={"s": 2})],
                  status=LaneStatus.UNAVAILABLE),
        ]
        got = detect_constant_signals(out)
        assert "s" in got  # only the ok lane's row exists for the check


# ---------------------------------------------------------------------------
# S4 feature reranker
# ---------------------------------------------------------------------------


class TestFeatureWeights:
    def test_table_matches_spec_verbatim(self):
        # V8.5 §05.04 slim set — the measured ship table: ``ent_idf``
        # keeps its feature at coefficient 0; the seven dead features
        # are no longer computed at all (test_shipset_v85 pins the
        # absence in ``detail["features"]``).
        assert dict(FEATURE_WEIGHTS_V1) == {
            "rrf_norm": 1.0,
            "cov_idf": 0.9,
            "ent_idf": 0.0,
            "phrase": 0.3,
            "speaker_match": 0.4,
            "t_prox": 0.4,
        }

    def test_score_family_and_model_tags(self):
        q = _query()
        out = score_candidates(q, [_fused("a", 0.02, {"lex": 1})])
        assert isinstance(out, ScoredList)
        assert out[0].score_family == "ranking/v7"
        assert out[0].detail["model"] == RERANK_MODEL_ID
        assert out[0].detail["formula_status"] == "provisional/v7-r0"


class TestDefaultFeatures:
    def test_rrf_norm_always(self):
        q = _query()
        fused = [
            _fused("a", 0.04, {"lex": 1, "dense": 3}),
            _fused("b", 0.02, {"lex": 25}),
        ]
        out = score_candidates(q, fused, n_lanes=4)
        fa = out[0].detail["features"]
        assert math.isclose(fa["rrf_norm"], 1.0)
        fb = out[1].detail["features"]
        assert math.isclose(fb["rrf_norm"], 0.5)
        # V8.5 §05.04: lane_agree is a measured dead feature — never
        # computed, never a silent zero.
        assert "lane_agree" not in fa and "lane_agree" not in fb

    def test_cov_idf_from_matched_term_idf(self):
        q = _query(terms=[
            NormTerm("alpha", "text", 0, 5),
            NormTerm("beta", "text", 6, 10),
        ])
        cand = _fused("a", 0.01, {"lex": 1}, signals={
            "term_idf": {"alpha": 2.0, "beta": 1.0},
        })
        out = score_candidates(q, [cand])
        cov = out[0].detail["features"]["cov_idf"]
        # idf provider absent → uniform 1.0 → matched {alpha,beta} ⊇ qterms → 1.0
        assert math.isclose(cov, 1.0)

    def test_cov_idf_with_idf_provider(self):
        q = _query(terms=[
            NormTerm("rare", "text", 0, 4),
            NormTerm("common", "text", 5, 11),
        ])
        prov = FeatureProviders(term_idf=lambda t: {"rare": 4.0, "common": 0.5}[t])
        cand = _fused("a", 0.01, {"lex": 1}, signals={
            "matched_terms": ("common",),
        })
        out = score_candidates(q, [cand], providers=prov)
        cov = out[0].detail["features"]["cov_idf"]
        assert math.isclose(cov, 0.5 / 4.5, rel_tol=1e-12)

    def test_speaker_match_three_way(self):
        cand = _fused("a", 0.01, {"lex": 1}, signals={"speaker_canon": "mel"})
        # speaker in question + match
        q = _query(speaker="mel")
        assert score_candidates(q, [cand])[0].detail["features"]["speaker_match"] == 1.0
        # different speaker
        q = _query(speaker="caroline")
        assert score_candidates(q, [cand])[0].detail["features"]["speaker_match"] == 0.0
        # no speaker in question
        q = _query()
        assert score_candidates(q, [cand])[0].detail["features"]["speaker_match"] == 0.5
        # unknown unit speaker → neutral, not "different"
        cand2 = _fused("b", 0.01, {"lex": 1})
        q = _query(speaker="mel")
        assert score_candidates(q, [cand2])[0].detail["features"]["speaker_match"] == 0.5

    def test_ident_exact_never_computed(self):
        """V8.5 §05.04: ``ident_exact`` is a measured dead feature — the
        default function no longer emits it even when the identifier
        signals that used to drive it are present (the ``feature_fn``
        seam can still re-add it for a declared arm)."""
        q = _query(identifiers=[NormTerm("ABC-123", "identifier", 0, 7)])
        exact = _fused("a", 0.01, {"lex": 1},
                       signals={"identifiers": ["ABC-123"],
                                "ident_exact": 1.0,
                                "ident_match": "exact"})
        out = score_candidates(q, [exact])
        assert "ident_exact" not in out[0].detail["features"]
        assert "ident_exact" not in out.stats["features_seen"]

    def test_t_prox_window_centre(self):
        centre = NOW_US
        window = IntervalUs(
            start_us=centre - 10 * DAY_US,
            end_us=centre + 10 * DAY_US,
            precision=OccurredPrecision.DAY,
            source=OccurredSource.RESOLVED_RELATIVE,
        )
        q = _query(intent=IntentClass.TEMPORAL_RANGE, window=window)
        near = _fused("a", 0.01, {"time": 1}, signals={
            "occurred_start_us": centre - DAY_US,
            "occurred_end_us": centre - DAY_US,
        })
        far = _fused("b", 0.01, {"time": 2}, signals={
            "occurred_start_us": centre - 20 * DAY_US,
            "occurred_end_us": centre - 20 * DAY_US,
        })
        undated = _fused("c", 0.01, {"time": 3})
        out = score_candidates(q, [near, far, undated])
        feats = {c.unit_id: c.detail["features"]["t_prox"] for c in out}
        assert math.isclose(feats["a"], 1.0 - 1 / 10, rel_tol=1e-9)
        assert feats["b"] == 0.0  # |Δ|/half ≥ 1 → clamped
        assert feats["c"] == 0.5  # undated → neutral

    def test_t_prox_no_window_neutral(self):
        q = _query()
        cand = _fused("a", 0.01, {"lex": 1}, signals={"recorded_at_us": NOW_US})
        out = score_candidates(q, [cand])
        assert out[0].detail["features"]["t_prox"] == 0.5

    def test_life_state_never_computed(self):
        """V8.5 §05.04: ``life_state`` is dead — absent for every intent,
        signals or not."""
        cur = _fused("a", 0.01, {"lex": 1}, signals={"lifecycle": "current"})
        hist = _fused("b", 0.01, {"lex": 2}, signals={"lifecycle": "historical"})
        for intent in (IntentClass.CURRENT_VALUE, IntentClass.HISTORY_OF,
                       IntentClass.LOOKUP):
            out = score_candidates(_query(intent=intent), [cur, hist])
            assert all(
                "life_state" not in c.detail["features"] for c in out)

    def test_passthrough_lane_feature_signals(self):
        cand = _fused("a", 0.01, {"lex": 1}, signals={
            "phrase": 1.0, "event_pred": 1.0, "corroboration": 0.5,
        })
        q = _query()
        feats = score_candidates(q, [cand])[0].detail["features"]
        assert feats["phrase"] == 1.0
        # V8.5 §05.04 — lane-carried dead features are not passed through.
        assert "event_pred" not in feats
        assert "corrob" not in feats

    def test_missing_features_absent_not_invented(self):
        q = _query()
        cand = _fused("a", 0.01, {"lex": 1})
        feats = score_candidates(q, [cand])[0].detail["features"]
        for f in ("cov_idf", "phrase", "ent_idf"):
            assert f not in feats
        # score still computable — Σ w·present
        assert feats["rrf_norm"] == 1.0
        assert feats["t_prox"] == 0.5 and feats["speaker_match"] == 0.5

    def test_suppress_zeroes_constant_features(self):
        q = _query(terms=[NormTerm("alpha", "text", 0, 5)])
        fused = [_fused(f"u{i}", 0.02, {"lex": i + 1},
                        signals={"term_idf": {"alpha": 1.0}})
                 for i in range(4)]
        out = score_candidates(q, fused, suppress={"cov_idf"})
        for c in out:
            assert "cov_idf" not in c.detail["features"]
            assert "cov_idf" in c.detail["suppressed"]
        assert out.stats["suppressed_applied"] == ["cov_idf"]

    def test_score_is_weighted_sum_plus_epsilon(self):
        q = _query(speaker="mel")
        cand = _fused("a", 0.02, {"lex": 1}, signals={
            "speaker_canon": "mel", "phrase": 1.0,
        })
        out = score_candidates(q, [cand])
        d = out[0].detail
        expected = math.fsum((
            1.0 * 1.0,   # rrf_norm
            0.3 * 1.0,   # phrase
            0.4 * 1.0,   # speaker_match
            0.4 * 0.5,   # t_prox (no window)
        ))
        assert math.isclose(d["feature_score"], expected, rel_tol=1e-12)
        # untied candidate: score == Σw·feature bit-for-bit, epsilon 0
        assert d["tie_epsilon"] == 0.0
        assert out[0].score == expected

    def test_tied_group_gets_epsilon_disclosed(self):
        """Tied Σw·f with distinct signal vectors → distinct eps → distinct
        scores, and the epsilon is disclosed."""
        q = _query()
        a = _fused("u1", 0.02, {"lex": 1}, source="s-a",
                   signals={"matched_terms": ("x",)})
        b = _fused("u2", 0.02, {"lex": 1}, source="s-b",
                   signals={"matched_terms": ("y",)})
        out = score_candidates(q, [a, b])
        assert out[0].score != out[1].score
        assert 0.0 < out[0].detail["tie_epsilon"] < TIE_EPSILON
        assert out.stats["tied_groups"] == 1

    def test_tie_break_ordering(self):
        """Tied feature scores → rrf_norm → (source_id, revision, unit seq)."""
        q = _query()
        a = _fused("u1", 0.02, {"lex": 1}, source="s-b", rev=1)
        b = _fused("u2", 0.02, {"lex": 1}, source="s-a", rev=1)
        c = _fused("u3", 0.02, {"lex": 1}, source="s-a", rev=2)
        out = score_candidates(q, [a, b, c])
        # identical empty signals → identical vectors → same τ & rrf_norm;
        # falls to source_id asc, then revision desc.
        assert [x.unit_id for x in out] == ["u3", "u2", "u1"]

    def test_byte_stable_determinism(self):
        q = _query(terms=[NormTerm("alpha", "text", 0, 5)])
        fused = [
            _fused(f"u{i}", 0.02 - i * 1e-4, {"lex": i + 1},
                   signals={"matched_terms": ("alpha",), "seq": i})
            for i in range(20)
        ]
        r1 = [(c.unit_id, c.score) for c in score_candidates(q, fused)]
        r2 = [(c.unit_id, c.score) for c in score_candidates(q, fused)]
        assert r1 == r2


# ---------------------------------------------------------------------------
# no-ties property (V7-10.02 / H13)
# ---------------------------------------------------------------------------


class TestNoTies:
    def test_distinct_signal_vectors_give_distinct_scores(self):
        """N candidates with distinct raw signal vectors → N distinct final
        scores (no D7-06 degenerate collapse)."""
        q = _query(terms=[NormTerm("alpha", "text", 0, 5)])
        n = 50
        fused = [
            _fused(f"u{i:03d}", 0.02, {"lex": 1},
                   signals={"matched_terms": ("alpha",), "seq": i,
                            "matched_terms_ctx": ("alpha",) if i % 2 else ()})
            for i in range(n)
        ]
        out = score_candidates(q, fused)
        scores = [c.score for c in out]
        assert len(set(scores)) == n
        # and ordering is strictly monotone — a true ranking, not buckets
        assert all(scores[i] > scores[i + 1] for i in range(n - 1))

    def test_identical_vectors_still_ordered_deterministically(self):
        """Identical signal vectors may tie on score (spec-permitted), but
        ordering must remain total and byte-stable."""
        q = _query()
        fused = [_fused(f"u{i}", 0.02, {"lex": 1}) for i in range(10)]
        r1 = [c.unit_id for c in score_candidates(q, fused)]
        r2 = [c.unit_id for c in score_candidates(q, fused)]
        assert r1 == r2
        assert len(set(r1)) == 10

    def test_constant_signals_never_collapse_to_single_score(self):
        """The D7-06 defect shape: a signal constant across the pool MUST NOT
        collapse every candidate to one score."""
        q = _query(terms=[NormTerm("alpha", "text", 0, 5)])
        fused = [
            _fused(f"u{i}", 0.02, {"lex": i + 1},
                   signals={"matched_terms": ("alpha",), "seq": i})
            for i in range(10)
        ]
        out = score_candidates(q, fused)
        assert len({c.score for c in out}) == 10


# ---------------------------------------------------------------------------
# S6 bounded boosts
# ---------------------------------------------------------------------------


def _scored_with_signals(unit, score, signals, source=None, rev=1):
    return _scored(unit, score, source=source, rev=rev,
                   detail={"signals": signals})


class TestBoosts:
    def test_formula_hand_computed(self):
        """final = base × (1+.2(r−.5)) × (1+.2(p−.5)) × (1+.1(pr−.5))."""
        q = _query()
        # dated 36.5 days ago → recency = 1 − 0.1 = 0.9
        item = _scored_with_signals("a", 2.0, {
            "occurred_start_us": NOW_US - int(36.5 * DAY_US),
        })
        out = apply_boosts([item], q, NOW_US)
        d = out[0].detail["boosts"]
        assert d["base"] == 1.0  # single candidate → top of rank scale
        assert math.isclose(d["recency"], 0.9, rel_tol=1e-9)
        assert d["proximity"] == 0.5
        assert d["proof"] == 0.5
        expected = 1.0 * (1 + 0.2 * 0.4) * 1.0 * 1.0
        assert math.isclose(out[0].score, expected, rel_tol=1e-12)
        assert out[0].detail["boosts"]["model"] == BOOSTS_MODEL_ID

    def test_recency_floor_and_cap(self):
        q = _query()
        ancient = _scored_with_signals("a", 1.0, {
            "occurred_start_us": NOW_US - 10 * 365 * DAY_US,
        })
        future = _scored_with_signals("b", 1.0, {
            "occurred_start_us": NOW_US + 5 * DAY_US,
        })
        out = apply_boosts([ancient, future], q, NOW_US)
        feats = {c.unit_id: c.detail["boosts"]["recency"] for c in out}
        assert feats["a"] == 0.1   # clamped floor
        assert feats["b"] == 1.0   # future → >1 → clamped cap

    def test_recency_neutral_explicit_past_window(self):
        """V7-09.07: recency MUST NOT apply to explicit past windows."""
        window = IntervalUs(start_us=NOW_US - 40 * DAY_US,
                            end_us=NOW_US - 10 * DAY_US)
        q = _query(intent=IntentClass.TEMPORAL_RANGE, window=window)
        fresh = _scored_with_signals("a", 1.0, {
            "occurred_start_us": NOW_US - DAY_US,
        })
        old = _scored_with_signals("b", 1.0, {
            "occurred_start_us": NOW_US - 300 * DAY_US,
        })
        out = apply_boosts([fresh, old], q, NOW_US)
        recs = {c.unit_id: c.detail["boosts"]["recency"] for c in out}
        assert recs == {"a": 0.5, "b": 0.5}
        assert out.stats["recency_neutral"] == 2

    def test_recency_applies_without_window(self):
        q = _query()
        fresh = _scored_with_signals("a", 1.0, {
            "occurred_start_us": NOW_US - DAY_US,
        })
        out = apply_boosts([fresh], q, NOW_US)
        assert out[0].detail["boosts"]["recency"] > 0.99

    def test_undated_recency_neutral(self):
        q = _query()
        out = apply_boosts([_scored_with_signals("a", 1.0, {})], q, NOW_US)
        assert out[0].detail["boosts"]["recency"] == 0.5

    def test_recorded_fallback_when_no_occurred(self):
        q = _query()
        item = _scored_with_signals("a", 1.0, {
            "recorded_at_us": NOW_US - 365 * DAY_US,
        })
        out = apply_boosts([item], q, NOW_US)
        # 365 days → 1 − 365/365 = 0 → floor 0.1
        assert out[0].detail["boosts"]["recency"] == 0.1

    def test_occurred_preferred_over_recorded(self):
        q = _query()
        item = _scored_with_signals("a", 1.0, {
            "occurred_start_us": NOW_US - 30 * DAY_US,
            "occurred_end_us": NOW_US - 30 * DAY_US,
            "recorded_at_us": NOW_US - 300 * DAY_US,
        })
        out = apply_boosts([item], q, NOW_US)
        r = out[0].detail["boosts"]["recency"]
        assert math.isclose(r, 1.0 - 30 / 365, rel_tol=1e-9)

    def test_proximity_in_window(self):
        centre = NOW_US - 20 * DAY_US
        window = IntervalUs(start_us=centre - 10 * DAY_US,
                            end_us=centre + 10 * DAY_US)
        q = _query(intent=IntentClass.TEMPORAL_RANGE, window=window)
        at_centre = _scored_with_signals("a", 1.0, {
            "occurred_start_us": centre, "occurred_end_us": centre,
        })
        at_edge = _scored_with_signals("b", 1.0, {
            "occurred_start_us": centre + 10 * DAY_US,
            "occurred_end_us": centre + 10 * DAY_US,
        })
        out = apply_boosts([at_centre, at_edge], q, NOW_US)
        prox = {c.unit_id: c.detail["boosts"]["proximity"] for c in out}
        assert prox["a"] == 1.0
        assert prox["b"] == 0.0

    def test_proximity_neutral_no_window(self):
        q = _query()
        item = _scored_with_signals("a", 1.0, {"occurred_start_us": NOW_US})
        out = apply_boosts([item], q, NOW_US)
        assert out[0].detail["boosts"]["proximity"] == 0.5

    def test_proof_derived_only(self):
        q = _query()
        derived = _scored_with_signals("a", 1.0, {
            "derived": True, "proof_count": 50,
        })
        raw = _scored_with_signals("b", 1.0, {
            "derived": False, "proof_count": 50,
        })
        out = apply_boosts([derived, raw], q, NOW_US)
        proofs = {c.unit_id: c.detail["boosts"]["proof"] for c in out}
        assert math.isclose(proofs["a"], 0.5 + math.log(50) / 10, rel_tol=1e-12)
        assert proofs["b"] == 0.5  # raw units never boosted by proof_count

    def test_pass_through_base_never_constant(self):
        """H20/V7-10.11: identical S4 scores → distinct rank-scaled bases —
        boosts can't reduce the ranking to a recency sort."""
        q = _query()
        items = [
            _scored_with_signals(f"u{i}", 2.0, {
                "occurred_start_us": NOW_US - i * 30 * DAY_US,
            })
            for i in range(5)
        ]
        out = apply_boosts(items, q, NOW_US)
        bases = [c.detail["boosts"]["base"] for c in out]
        assert len(set(bases)) == 5  # never a constant
        assert max(bases) == 1.0 and math.isclose(min(bases), 0.1)
        finals = [c.score for c in out]
        assert len(set(finals)) == 5

    def test_ce_base_used_when_present(self):
        q = _query()
        item = _scored("a", 2.0, detail={"ce_score": 0.42})
        out = apply_boosts([item], q, NOW_US)
        assert out[0].detail["boosts"]["base"] == 0.42
        assert out[0].detail["boosts"]["base_kind"] == "ce"
        assert out.stats["ce_bases"] == 1

    def test_swing_bounds_at_extremes(self):
        """H19: combined swing stays in [−25%, +30%] on adversarial inputs."""
        q = _query()
        window = IntervalUs(start_us=NOW_US - 10 * DAY_US,
                            end_us=NOW_US + 10 * DAY_US)
        q_win = _query(intent=IntentClass.TEMPORAL_RANGE, window=window)
        # worst-down: ancient + far from window + no proof
        worst = _scored_with_signals("a", 1.0, {
            "occurred_start_us": NOW_US - 10 * 365 * DAY_US,
            "occurred_end_us": NOW_US - 10 * 365 * DAY_US,
        })
        # best-up: at centre + derived with huge proof count
        best = _scored_with_signals("b", 1.0, {
            "occurred_start_us": NOW_US, "occurred_end_us": NOW_US,
            "derived": True, "proof_count": 10**9,
        })
        out = apply_boosts([worst, best], q_win, NOW_US)
        for c in out:
            factor = c.detail["boosts"]["factor"]
            assert SWING_MIN <= factor <= SWING_MAX
        d = {c.unit_id: c.detail["boosts"] for c in out}
        # window ends in the future → NOT a past window → recency applies:
        # a: recency floor 0.1 → 0.92, proximity 0 → 0.9, proof 0.5 → 1.0
        assert d["a"]["factor"] == pytest.approx(0.92 * 0.9 * 1.0)
        # b: recency 1.0 → 1.1, proximity 1.0 → 1.1, proof 1.0 → 1.05
        assert d["b"]["factor"] == pytest.approx(1.1 * 1.1 * 1.05)

    def test_swing_bounds_no_window_extremes(self):
        """Without a window: recency floor + proof floor → min factor ≥ 0.75."""
        q = _query()
        items = [
            _scored_with_signals("a", 1.0, {
                "occurred_start_us": NOW_US - 100 * 365 * DAY_US,
            }),
            _scored_with_signals("b", 1.0, {
                "occurred_start_us": NOW_US,
                "derived": True, "proof_count": 10**12,
            }),
        ]
        out = apply_boosts(items, q, NOW_US)
        fa = next(c for c in out if c.unit_id == "a").detail["boosts"]["factor"]
        fb = next(c for c in out if c.unit_id == "b").detail["boosts"]["factor"]
        assert math.isclose(fa, (1 + ALPHA_RECENCY * (0.1 - 0.5))
                            * (1 + ALPHA_PROXIMITY * (0.5 - 0.5))
                            * (1 + ALPHA_PROOF * (0.5 - 0.5)), rel_tol=1e-12)
        assert fa >= SWING_MIN
        assert fb <= SWING_MAX
        assert 0.75 <= out.stats["factor_min"]
        assert out.stats["factor_max"] <= 1.30

    def test_boosts_can_reorder_within_band(self):
        """A mildly better-scored old item can outrank a fresh one only when
        the base gap fits inside the ±swing band."""
        q = _query()
        fresh = _scored_with_signals("fresh", 1.00, {
            "occurred_start_us": NOW_US - DAY_US,
        })
        old = _scored_with_signals("old", 1.20, {
            "occurred_start_us": NOW_US - 10 * 365 * DAY_US,
        })
        out = apply_boosts([fresh, old], q, NOW_US)
        # bases by pre-boost rank: old rank1 → 1.0, fresh rank2 → 0.1
        # old final = 1.0 × (1+0.2(0.1−0.5)) = 0.92
        # fresh final = 0.1 × (1+0.2·~0.5) ≈ 0.11
        assert out[0].unit_id == "old"
        d_old = out[0].detail["boosts"]
        assert math.isclose(d_old["base"], 1.0)
        assert math.isclose(d_old["factor"], 1 + 0.2 * (0.1 - 0.5), rel_tol=1e-9)

    def test_boosts_deterministic_byte_stable(self):
        q = _query()
        items = [
            _scored_with_signals(f"u{i}", 1.0 - i * 0.01, {
                "occurred_start_us": NOW_US - i * 7 * DAY_US,
                "derived": i % 2 == 0, "proof_count": i,
            })
            for i in range(15)
        ]
        r1 = [(c.unit_id, c.score) for c in apply_boosts(items, q, NOW_US)]
        r2 = [(c.unit_id, c.score) for c in apply_boosts(items, q, NOW_US)]
        assert r1 == r2
        # input order never matters — sorted deterministically on entry
        r3 = [(c.unit_id, c.score)
              for c in apply_boosts(list(reversed(items)), q, NOW_US)]
        assert r1 == r3

    def test_empty_input(self):
        out = apply_boosts([], _query(), NOW_US)
        assert list(out) == []
        assert out.stats["candidates"] == 0


# ---------------------------------------------------------------------------
# end-to-end + perf
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_fuse_score_boost_pipeline(self):
        """S3 → S4 → S6 wired in order; every artifact number explainable."""
        q = _query(terms=[NormTerm("alpha", "text", 0, 5)], speaker="mel")
        lanes = [
            _lane("lex", [
                _cand("a", "lex", 1, signals={
                    "matched_terms": ("alpha",), "speaker_canon": "mel",
                    "occurred_start_us": NOW_US - 5 * DAY_US,
                }),
                _cand("b", "lex", 2, signals={
                    "matched_terms": (), "speaker_canon": "caroline",
                    "occurred_start_us": NOW_US - 5 * DAY_US,
                }),
            ]),
            _lane("dense", [_cand("a", "dense", 1)]),
        ]
        fused = rrf_fuse(lanes, {})
        scored = score_candidates(q, fused, n_lanes=2)
        final = apply_boosts(scored, q, NOW_US)
        assert final[0].unit_id == "a"
        d = final[0].detail
        assert d["boosts"]["base"] == 1.0
        assert "features" in d and "lane_ranks" in d
        assert d["score_family"] if "score_family" in d else True
        assert final[0].score_family == "ranking/v7"

    def test_perf_s4_under_budget(self):
        """V7-10.07: ≤3 ms p95 at R=100 on the reference machine. Generous CI
        bound (10×) still catches a quadratic blow-up."""
        q = _query(terms=[NormTerm("alpha", "text", 0, 5),
                          NormTerm("beta", "text", 6, 10)])
        fused = [
            _fused(f"u{i:04d}", 0.02 - i * 1e-6,
                   {"lex": i + 1, "dense": (i % 7) + 1},
                   signals={"matched_terms": ("alpha", "beta"),
                            "term_idf": {"alpha": 2.0, "beta": 1.0},
                            "seq": i,
                            "speaker_canon": "mel",
                            "occurred_start_us": NOW_US - i * DAY_US})
            for i in range(100)
        ]
        best = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            score_candidates(q, fused, n_lanes=8)
            best = min(best, time.perf_counter() - t0)
        assert best < 0.015  # 15 ms CI bound; ~2 ms typical (spec ≤3 ms p95)
