"""V7.5 fusion conformance tests (SPEC_V7_5 §03/§04; J07/J08/J13/J17).

- V75-03.03 / J07 — pre-O9 default fusion weights are flat (``1.0`` on
  every lane, per V7-10.01's text); the §32.3 intent-weighted matrix is
  reachable only by selecting the named arm
  ``retrieval_policy/v7-intent-weighted``, and the applied table is
  honestly reported (the ``policy_id`` arm tag + fusion stats).
- V75-03.04 / J08 — ``coverage.facets.bonus`` is declared iff the
  ``facet_bonus/v1`` bounded rank-space term was actually applied, which
  requires a unit covered by >= 2 decomposition facets.
- V75-04.03 / J13 — per-profile ``lane_gates`` exclude a named lane from
  fusion or cap its contribution to top-N ranks; a gated lane still
  answers when it is the only lane with candidates.
- J17 — lane weights enter only as ``w/(k+rank)`` (or the divisor form
  ``1/(k+rank/w)``): raw heterogeneous lane scores are never scaled and
  summed.

Hand-computed fixtures; constructs contract types directly.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace as dc_replace

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v7 import (
    BudgetClass,
    CandidateV7,
    IntentClass,
    IntentResult,
    LaneContextV7,
    LaneName,
    LaneOutput,
    LaneStatus,
    NormAnalysis,
    NormTerm,
    QueryViewV7,
    RetrievalPolicyV7,
)
from verbatim.retrieval.v7 import policy as pol
from verbatim.retrieval.v7.fusion import (
    FACET_BONUS_BETA,
    FACET_BONUS_ID,
    FACET_BONUS_MAX,
    RRF_K,
    rrf_fuse,
    score_detail,
)
from verbatim.retrieval.v7.lanes_base import LANE_REGISTRY, register_lane
from verbatim.retrieval.v7.pipeline import run_search

NOW_US = 1_700_000_000_000_000  # fixed epoch µs


# ---------------------------------------------------------------------------
# helpers (same shapes as test_fusion.py / test_pipeline.py)
# ---------------------------------------------------------------------------


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


def _query(text="alpha beta", intent=IntentClass.LOOKUP, facets=()):
    terms = tuple(
        NormTerm(
            term=t, channel="text", byte_start=i * 8, byte_end=i * 8 + len(t)
        )
        for i, t in enumerate(text.split())
    )
    return QueryViewV7(
        query=text,
        norm=NormAnalysis(
            analyzer_id="norm/v2",
            terms=terms,
            identifiers=(),
            text=text,
        ),
        intent=IntentResult(primary=intent, classes=(intent,)),
        facets=tuple(facets),
        query_time_us=NOW_US,
    )


@pytest.fixture(autouse=True)
def clean_registry():
    saved = dict(LANE_REGISTRY)
    LANE_REGISTRY.clear()
    yield
    LANE_REGISTRY.clear()
    LANE_REGISTRY.update(saved)


def make_ctx(policy, *, budget=BudgetClass.MID, manifest=None):
    return LaneContextV7(
        store=None,
        scope_id="scope",
        generation=1,
        eligible=None,
        query_time_us=NOW_US,
        profile="test",
        budget=budget,
        policy=policy,
        manifest=manifest or {},
    )


def narrow_policy(policy, lanes):
    """Same policy with a smaller enabled lane set (keeps lane_gates)."""
    lanes = tuple(lanes)
    return dc_replace(
        policy,
        lanes=lanes,
        lane_weights={
            ic: {l: w for l, w in row.items() if l in lanes}
            for ic, row in policy.lane_weights.items()
        },
    )


def fake_lane(name, units, *, calls=None):
    """Fake lane emitting ``units`` at ranks 1..n on every invocation."""

    def fn(ctx, qv, slice):
        if calls is not None:
            calls.append(qv.query)
        return LaneOutput(
            lane=name,
            status=LaneStatus.OK,
            candidates=[
                _cand(u, name, i + 1, raw=1.0 - i * 0.1)
                for i, u in enumerate(units)
            ],
            examined=len(units),
            eligible=len(units),
        )

    return fn


def make_policy(
    policy_id="retrieval_policy/v7",
    lanes=(LaneName.LEX, LaneName.DENSE),
    weights=None,
    gates=None,
):
    """A directly-built policy for pipeline tests (no ``load_policy``)."""
    if gates is not None:
        return pol.GatedPolicyV7(
            policy_id=policy_id,
            profile="test",
            lanes=tuple(lanes),
            lane_weights=weights or {},
            lane_gates=gates,
        )
    return RetrievalPolicyV7(
        policy_id=policy_id,
        profile="test",
        lanes=tuple(lanes),
        lane_weights=weights or {},
    )


# ===========================================================================
# J07 — V75-03.03: flat pre-O9 default; intent-weighted table is a named arm
# ===========================================================================


class TestFlatDefault:
    def test_default_policy_is_flat_everywhere(self):
        """J07: the default table gives w = 1.0 on every (intent, lane)."""
        p = pol.load_policy("local_memory")
        assert p.policy_id == "retrieval_policy/v7"
        assert set(p.lane_weights) == set(IntentClass)
        for intent in IntentClass:
            row = p.lane_weights[intent]
            assert set(row) == set(p.lanes)
            assert all(w == 1.0 for w in row.values()), intent
        # weights_for agrees for every intent, incl. abstain_likely
        for intent in IntentClass:
            assert pol.weights_for(intent, p) == {lane: 1.0 for lane in p.lanes}

    def test_intent_weighted_arm_selectable_by_policy_id(self):
        """J07: the §32.3 table runs only when the named arm is selected."""
        # The §32.3 table is materialized over the enabled lanes; declare the
        # full eight-lane registry (V85 defaults enable only lex/fuzzy/dense/
        # time) so every arm row resolves against the complete weight table.
        p = pol.load_policy(
            "local_memory",
            {
                "policy_id": pol.POLICY_ID_INTENT_WEIGHTED,
                "lanes": [lane.value for lane in pol.LANES_V1],
            },
        )
        assert p.policy_id == "retrieval_policy/v7-intent-weighted"
        for intent, row in pol.LANE_WEIGHTS_V1.items():
            assert p.lane_weights[intent] == dict(row)
        # the arm matrix really is non-flat — the assertions above would
        # be vacuous if LANE_WEIGHTS_V1 were all ones
        assert any(
            w != 1.0 for row in pol.LANE_WEIGHTS_V1.values() for w in row.values()
        )
        # abstain_likely has no §32.3 row -> still flat under the arm
        assert all(
            w == 1.0 for w in p.lane_weights[IntentClass.ABSTAIN_LIKELY].values()
        )

    def test_intent_weighted_arm_not_reachable_by_default(self):
        """A bare load_policy must never silently produce the §32.3 table:
        every intent with a §32.3 row resolves flat instead."""
        p = pol.load_policy("local_memory")
        for intent, arm_row in pol.LANE_WEIGHTS_V1.items():
            assert p.lane_weights[intent] != dict(arm_row)
            assert all(w == 1.0 for w in p.lane_weights[intent].values())

    def test_overrides_merge_over_the_flat_base(self):
        """Explicit lane_weights still tune per-cell — over 1.0, not over
        the §32.3 row."""
        p = pol.load_policy(
            "local_memory", {"lane_weights": {"lookup": {"dense": 2.0}}}
        )
        assert p.lane_weights[IntentClass.LOOKUP][LaneName.DENSE] == 2.0
        # a cell the doc didn't name keeps the flat default, not §32.3's 1.4
        assert p.lane_weights[IntentClass.OPEN_DOMAIN][LaneName.DENSE] == 1.0

    def test_arm_overrides_merge_over_intent_weighted_base(self):
        """Inside the arm, overrides merge over the §32.3 row."""
        p = pol.load_policy(
            "local_memory",
            {
                "policy_id": pol.POLICY_ID_INTENT_WEIGHTED,
                "lane_weights": {"lookup": {"dense": 2.0}},
            },
        )
        assert p.lane_weights[IntentClass.LOOKUP][LaneName.DENSE] == 2.0
        assert p.lane_weights[IntentClass.LOOKUP][LaneName.LEX] == 1.0
        # an un-touched cell keeps the §32.3 arm value (open_domain dense 1.4)
        assert p.lane_weights[IntentClass.OPEN_DOMAIN][LaneName.DENSE] == 1.4

    def test_pipeline_default_run_reports_flat_arm(self):
        """J07 + attribution: a default run applies w=1 per lane and says so."""
        register_lane(LaneName.LEX, fake_lane("lex", ["a", "b"]))
        register_lane(LaneName.DENSE, fake_lane("dense", ["b", "c"]))
        ctx = make_ctx(pol.load_policy("test", {"lanes": ["lex", "dense"]}))
        result = run_search(ctx, _query(), 1000.0)
        stats = result.fused.stats
        assert stats["weights_tag"] == "retrieval_policy/v7"
        assert stats["weights"] == {"lex": 1.0, "dense": 1.0}
        note = result.coverage.rerank["stages"]["fusion"]
        assert note["status"] == "ok"
        assert note["weights_arm"] == "retrieval_policy/v7"
        # hand-computed: b in both lanes wins under flat RRF
        assert result.fused[0].unit_id == "b"
        assert math.isclose(result.fused[0].rrf, 1 / 62 + 1 / 61, rel_tol=1e-15)

    def test_pipeline_arm_run_reports_tag_and_spec_weights(self):
        """The selected arm applies its §32.3 row and reports the tag."""
        register_lane(LaneName.LEX, fake_lane("lex", ["a", "b"]))
        register_lane(LaneName.DENSE, fake_lane("dense", ["b", "c"]))
        ctx = make_ctx(
            pol.load_policy(
                "test",
                {
                    "policy_id": pol.POLICY_ID_INTENT_WEIGHTED,
                    "lanes": ["lex", "dense"],
                },
            )
        )
        result = run_search(ctx, _query(intent=IntentClass.OPEN_DOMAIN), 1000.0)
        stats = result.fused.stats
        assert stats["weights_tag"] == "retrieval_policy/v7-intent-weighted"
        # open_domain row: lex 0.8 / dense 1.4 (§32.3)
        assert stats["weights"] == {"lex": 0.8, "dense": 1.4}
        assert (
            result.coverage.rerank["stages"]["fusion"]["weights_arm"]
            == "retrieval_policy/v7-intent-weighted"
        )

    def test_arm_selection_changes_fused_scores(self):
        """Sanity: the two tables genuinely produce different RRF values."""
        outs = [
            _lane("lex", [_cand("a", "lex", 1)]),
            _lane(
                "dense",
                [_cand("b", "dense", 1), _cand("c", "dense", 2), _cand("a", "dense", 3)],
            ),
        ]
        flat = rrf_fuse(outs, {LaneName.LEX: 1.0, LaneName.DENSE: 1.0})
        arm = rrf_fuse(outs, {LaneName.LEX: 0.8, LaneName.DENSE: 1.4})
        # flat: a = 1/61 + 1/63; arm: a = 0.8/61 + 1.4/63
        assert math.isclose(flat[0].rrf, 1 / 61 + 1 / 63, rel_tol=1e-15)
        assert math.isclose(arm[0].rrf, 0.8 / 61 + 1.4 / 63, rel_tol=1e-15)
        assert arm[0].rrf != flat[0].rrf
        # and the dense-heavy arm closes the gap between a and b
        assert (arm[0].rrf - arm[1].rrf) < (flat[0].rrf - flat[1].rrf)


# ===========================================================================
# J17 — weights enter in rank space only, never by scaling raw lane scores
# ===========================================================================


class TestRankSpaceOnly:
    def test_raw_score_magnitude_cannot_change_fused_score(self):
        """J17: two lanes on incompatible score scales — the weight enters
        via the rank divisor; raw_score never participates."""
        big_raw = rrf_fuse(
            [
                _lane("lex", [_cand("a", "lex", 1, raw=9_842.7)]),
                _lane("dense", [_cand("b", "dense", 1, raw=0.0007)]),
            ],
            {"lex": 1.0, "dense": 1.0},
        )
        small_raw = rrf_fuse(
            [
                _lane("lex", [_cand("a", "lex", 1, raw=0.0001)]),
                _lane("dense", [_cand("b", "dense", 1, raw=4_000.0)]),
            ],
            {"lex": 1.0, "dense": 1.0},
        )
        # identical ranks -> identical fused scores, whatever raw said
        assert [(c.unit_id, c.rrf) for c in big_raw] == [
            (c.unit_id, c.rrf) for c in small_raw
        ]
        for c in big_raw:
            assert math.isclose(c.rrf, 1.0 / (RRF_K + 1), rel_tol=1e-15)

    def test_weight_enters_via_rank_divisor_only(self):
        """w/(k+rank) exactly: a 10^9x raw advantage loses to the divisor."""
        # lex rank 2 with an astronomically larger raw score still loses to
        # dense rank 1 under equal weights — only ranks decide.
        out = rrf_fuse(
            [
                _lane("lex", [_cand("a", "lex", 2, raw=1e9)]),
                _lane("dense", [_cand("b", "dense", 1, raw=1e-9)]),
            ],
            {},
        )
        assert [c.unit_id for c in out] == ["b", "a"]
        assert math.isclose(out[0].rrf, 1 / (RRF_K + 1), rel_tol=1e-15)
        assert math.isclose(out[1].rrf, 1 / (RRF_K + 2), rel_tol=1e-15)

    def test_weighted_lane_contribution_is_w_over_k_plus_rank(self):
        """w enters the numerator of 1/(k+rank) — never a raw-score factor."""
        out = rrf_fuse(
            [
                _lane("lex", [_cand("a", "lex", 1, raw=5.0)]),
                _lane("dense", [_cand("a", "dense", 1, raw=0.5)]),
            ],
            {"dense": 2.0},
        )
        assert math.isclose(out[0].rrf, 1 / 61 + 2 / 61, rel_tol=1e-15)
        detail = score_detail(out[0], {"dense": 2.0})
        assert math.isclose(detail["contributions"]["lex"], 1 / 61)
        assert math.isclose(detail["contributions"]["dense"], 2 / 61)

    def test_zero_raw_does_not_zero_the_lane(self):
        """A lane reporting raw_score=0.0 still contributes by rank — a
        score-space multiply would have zeroed it."""
        out = rrf_fuse(
            [_lane("lex", [_cand("a", "lex", 1, raw=0.0)])],
            {"lex": 1.0},
        )
        assert math.isclose(out[0].rrf, 1 / 61, rel_tol=1e-15)

    def test_heterogeneous_scales_same_order(self):
        """Permuting raw magnitudes across lanes leaves ordering untouched —
        proof that no raw-score term leaks into the sum."""
        outs_a = [
            _lane(
                "lex",
                [_cand("a", "lex", 1, raw=100.0), _cand("b", "lex", 2, raw=50.0)],
            ),
            _lane("dense", [_cand("b", "dense", 1, raw=0.01)]),
        ]
        outs_b = [
            _lane(
                "lex",
                [_cand("a", "lex", 1, raw=0.02), _cand("b", "lex", 2, raw=0.01)],
            ),
            _lane("dense", [_cand("b", "dense", 1, raw=9_999.0)]),
        ]
        r1 = rrf_fuse(outs_a, {"lex": 1.3, "dense": 0.7})
        r2 = rrf_fuse(outs_b, {"lex": 1.3, "dense": 0.7})
        assert [(c.unit_id, c.rrf) for c in r1] == [
            (c.unit_id, c.rrf) for c in r2
        ]


# ===========================================================================
# J08 — V75-03.04: facet coverage bonus, declared iff applied
# ===========================================================================


class TestFacetBonus:
    def test_two_facet_coverage_applies_bounded_bonus(self):
        """bonus = min(beta*(covered-1), cap) / (k + best_rank)."""
        out = rrf_fuse(
            [
                _lane(
                    "lex",
                    [
                        _cand("a", "lex", 1, signals={"facet": 0}),
                        _cand("b", "lex", 2),
                    ],
                ),
                _lane("dense", [_cand("a", "dense", 3, signals={"facet": 1})]),
            ],
            {},
        )
        top = out[0]
        assert top.unit_id == "a"
        best = min(top.lane_ranks.values())  # rank 1 on lex
        bonus = FACET_BONUS_BETA / (RRF_K + best)
        assert math.isclose(top.rrf, 1 / 61 + 1 / 63 + bonus, rel_tol=1e-15)
        assert top.signals["facet_coverage"] == (0, 1)
        assert math.isclose(top.signals["facet_bonus"], bonus)
        assert out.stats["facet_bonus"]["applied"] == 1
        assert out.stats["facet_bonus"]["tag"] == FACET_BONUS_ID
        # score_detail recomputes the term for explain — Σ parts == rrf
        detail = score_detail(top, {})
        assert detail["facet_bonus"]["tag"] == FACET_BONUS_ID
        assert math.isclose(detail["facet_bonus"]["value"], bonus)
        assert math.isclose(
            detail["rrf"],
            sum(detail["contributions"].values()) + detail["facet_bonus"]["value"],
            rel_tol=1e-15,
        )

    def test_single_facet_coverage_no_bonus(self):
        out = rrf_fuse(
            [
                _lane("lex", [_cand("a", "lex", 1, signals={"facet": 0})]),
                _lane("dense", [_cand("b", "dense", 1)]),
            ],
            {},
        )
        assert out.stats["facet_bonus"]["applied"] == 0
        assert all("facet_bonus" not in c.signals for c in out)

    def test_same_facet_twice_is_not_two_facets(self):
        """Coverage counts DISTINCT facets — facet 0 on two lanes = 1."""
        out = rrf_fuse(
            [
                _lane("lex", [_cand("a", "lex", 1, signals={"facet": 0})]),
                _lane("dense", [_cand("a", "dense", 1, signals={"facet": 0})]),
            ],
            {},
        )
        assert out.stats["facet_bonus"]["applied"] == 0

    def test_three_facet_coverage_capped_at_one_lane_max(self):
        """beta*(3-1) = 1.0 hits the cap — never exceeds a lane's top hit."""
        out = rrf_fuse(
            [
                _lane("lex", [_cand("a", "lex", 4, signals={"facet": 0})]),
                _lane("dense", [_cand("a", "dense", 2, signals={"facet": 1})]),
                _lane("ent", [_cand("a", "ent", 1, signals={"facet": 2})]),
            ],
            {},
        )
        top = out[0]
        best = min(top.lane_ranks.values())
        expected_w = min(FACET_BONUS_BETA * 2, FACET_BONUS_MAX)
        assert expected_w == FACET_BONUS_MAX == 1.0  # the cap fired
        assert math.isclose(top.signals["facet_bonus"], expected_w / (RRF_K + best))
        # the cap means: bonus <= one default lane's rank-1 contribution
        assert top.signals["facet_bonus"] <= 1.0 / (RRF_K + 1)

    def test_within_lane_duplicate_facet_counts(self):
        """A unit re-surfaced by facets inside the same lane still documents
        coverage even though its rank loses the within-lane dedup."""
        out = rrf_fuse(
            [
                _lane(
                    "lex",
                    [
                        _cand("a", "lex", 1),
                        _cand("a", "lex", 2, signals={"facet": 1}),
                        _cand("a", "lex", 3, signals={"facet": 0}),
                    ],
                ),
            ],
            {},
        )
        assert out[0].signals["facet_coverage"] == (0, 1)
        assert out.stats["facet_bonus"]["applied"] == 1

    def test_pipeline_declares_bonus_only_when_applied(self):
        """J08 end-to-end: multi-facet query declares; single/none don't."""

        def lane(ctx, qv, slice):
            uid = "main" if qv.query.startswith("main") else f"only-{qv.query}"
            return LaneOutput(
                lane="lex",
                status=LaneStatus.OK,
                candidates=[_cand(uid, "lex", 1), _cand("shared", "lex", 2)],
                examined=2,
                eligible=2,
            )

        register_lane(LaneName.LEX, lane)
        policy = narrow_policy(make_policy(), (LaneName.LEX,))

        # two facets -> "shared" is covered by facet 0 and facet 1
        f1 = _query("facet one")
        f2 = _query("facet two")
        q = _query("main query", facets=(f1, f2))
        result = run_search(make_ctx(policy), q, 1000.0)
        assert result.coverage.facets["ran"] == 2
        assert result.coverage.facets["bonus"] == FACET_BONUS_ID
        assert result.fused.stats["facet_bonus"]["applied"] >= 1

        # single facet -> no bonus declared (J08)
        result1 = run_search(
            make_ctx(policy), _query("main query", facets=(f1,)), 1000.0
        )
        assert result1.coverage.facets["ran"] == 1
        assert "bonus" not in result1.coverage.facets

        # no facets -> also nothing declared
        result0 = run_search(make_ctx(policy), _query("main query"), 1000.0)
        assert "bonus" not in result0.coverage.facets

    def test_pipeline_facet_fallback_fusion_declares_nothing(self, monkeypatch):
        """J08 honesty: the minimal fallback fusion never fabricates a
        bonus declaration (it does not implement the term)."""
        from verbatim.retrieval.v7 import pipeline as pl

        real = pl._lazy_import
        monkeypatch.setattr(
            pl,
            "_lazy_import",
            lambda name: None if name == pl._MOD_FUSION else real(name),
        )

        def lane(ctx, qv, slice):
            return LaneOutput(
                lane="lex",
                status=LaneStatus.OK,
                candidates=[_cand("shared", "lex", 1)],
                examined=1,
                eligible=1,
            )

        register_lane(LaneName.LEX, lane)
        policy = narrow_policy(make_policy(), (LaneName.LEX,))
        q = _query("main query", facets=(_query("facet one"), _query("facet two")))
        result = run_search(make_ctx(policy), q, 1000.0)
        assert result.coverage.facets["ran"] == 2
        assert "bonus" not in result.coverage.facets
        assert (
            result.coverage.rerank["stages"]["fusion"]["status"] == "unavailable"
        )


# ===========================================================================
# J13 — V75-04.03: weak-lane gates (machinery; default = nothing gated)
# ===========================================================================


class TestLaneGates:
    def test_default_nothing_gated(self):
        out = rrf_fuse(
            [
                _lane("lex", [_cand("a", "lex", 1)]),
                _lane("dense", [_cand("b", "dense", 1)]),
            ],
            {},
        )
        assert out.stats["lane_gates"] == {}
        assert out.stats["gates_bypassed"] is False
        assert out.stats["lanes_gated"] == {}
        assert {c.unit_id for c in out} == {"a", "b"}

    def test_exclude_removes_lane_from_fusion(self):
        out = rrf_fuse(
            [
                _lane("lex", [_cand("a", "lex", 1)]),
                _lane("dense", [_cand("b", "dense", 1), _cand("a", "dense", 2)]),
            ],
            {},
            lane_gates={"dense": "exclude"},
        )
        assert [c.unit_id for c in out] == ["a"]
        # the gated lane's corroborating rank is also gone from provenance
        assert out[0].lane_ranks == {"lex": 1}
        assert out.stats["lanes_gated"]["dense"] == {"gate": 0, "dropped": 2}

    def test_exclude_zero_form_identical(self):
        a = rrf_fuse(
            [
                _lane("dense", [_cand("b", "dense", 1)]),
                _lane("lex", [_cand("c", "lex", 1)]),
            ],
            {},
            lane_gates={"dense": 0},
        )
        b = rrf_fuse(
            [
                _lane("dense", [_cand("b", "dense", 1)]),
                _lane("lex", [_cand("c", "lex", 1)]),
            ],
            {},
            lane_gates={"dense": "exclude"},
        )
        assert [(c.unit_id, c.rrf) for c in a] == [(c.unit_id, c.rrf) for c in b]

    def test_top_n_caps_contribution(self):
        out = rrf_fuse(
            [
                _lane(
                    "dense",
                    [
                        _cand("d1", "dense", 1),
                        _cand("d2", "dense", 2),
                        _cand("d3", "dense", 3),
                    ],
                ),
                _lane("lex", [_cand("l1", "lex", 1)]),
            ],
            {},
            lane_gates={"dense": 2},
        )
        ids = {c.unit_id for c in out}
        assert ids == {"d1", "d2", "l1"}
        assert "d3" not in ids
        assert out.stats["lanes_gated"]["dense"] == {"gate": 2, "dropped": 1}
        # contributing ranks keep their exact RRF terms
        d1 = next(c for c in out if c.unit_id == "d1")
        assert math.isclose(d1.rrf, 1 / 61, rel_tol=1e-15)

    def test_top_n_dict_form(self):
        out = rrf_fuse(
            [
                _lane(
                    "dense", [_cand("d1", "dense", 1), _cand("d2", "dense", 2)]
                ),
                _lane("lex", [_cand("l1", "lex", 1)]),
            ],
            {},
            lane_gates={"dense": {"top_n": 1}},
        )
        assert {c.unit_id for c in out} == {"d1", "l1"}

    def test_gated_lane_still_answers_alone(self):
        """J13: the only lane with candidates is gated -> gates bypass."""
        out = rrf_fuse(
            [
                _lane(
                    "dense", [_cand("b", "dense", 1), _cand("c", "dense", 2)]
                ),
                _lane("lex", []),  # ungated but empty
            ],
            {},
            lane_gates={"dense": 0},
        )
        assert {c.unit_id for c in out} == {"b", "c"}
        assert out.stats["gates_bypassed"] is True
        assert out.stats["lane_gates"] == {"dense": 0}  # declared, bypassed
        assert out.stats["lanes_gated"] == {}  # nothing dropped

    def test_all_lanes_gated_bypasses(self):
        out = rrf_fuse(
            [
                _lane("dense", [_cand("b", "dense", 1)]),
                _lane("lex", [_cand("a", "lex", 1)]),
            ],
            {},
            lane_gates={"dense": 0, "lex": 1},
        )
        assert {c.unit_id for c in out} == {"a", "b"}
        assert out.stats["gates_bypassed"] is True

    def test_gate_does_not_bypass_when_ungated_lane_contributes(self):
        out = rrf_fuse(
            [
                _lane("dense", [_cand("b", "dense", 1)]),
                _lane("lex", [_cand("a", "lex", 1)]),
            ],
            {},
            lane_gates={"dense": 0},
        )
        assert {c.unit_id for c in out} == {"a"}
        assert out.stats["gates_bypassed"] is False

    def test_malformed_gate_fails_loud(self):
        for bad in (-1, True, 1.5, "sometimes", {"nope": 3}):
            with pytest.raises(ValueError):
                rrf_fuse(
                    [_lane("lex", [_cand("a", "lex", 1)])],
                    {},
                    lane_gates={"lex": bad},
                )

    def test_gate_on_degraded_lane_is_inert(self):
        out = rrf_fuse(
            [
                _lane(
                    "dense",
                    [_cand("b", "dense", 1)],
                    status=LaneStatus.UNAVAILABLE,
                ),
                _lane("lex", [_cand("a", "lex", 1)]),
            ],
            {},
            lane_gates={"dense": 0},
        )
        # dense was ignored for status, lex ungated and contributing
        assert {c.unit_id for c in out} == {"a"}
        assert out.stats["gates_bypassed"] is False


class TestPolicyLaneGates:
    def test_policy_doc_declares_gates(self):
        p = pol.load_policy(
            "local_memory",
            {
                "lane_gates": {
                    "dense": "exclude",
                    "fuzzy": {"top_n": 20},
                    "graph": 5,
                }
            },
        )
        assert p.lane_gates == {
            LaneName.DENSE: 0,
            LaneName.FUZZY: 20,
            LaneName.GRAPH: 5,
        }
        # default: nothing gated
        assert pol.load_policy("local_memory").lane_gates == {}

    def test_profile_overlay_merges_gates_per_lane(self):
        doc = {
            "lane_gates": {"dense": "exclude"},
            "profiles": {
                "local_memory": {
                    "lane_gates": {"dense": {"top_n": 8}, "obs": 2}
                },
                "other": {"lane_gates": {"lex": "exclude"}},
            },
        }
        p = pol.load_policy("local_memory", doc)
        assert p.lane_gates == {LaneName.DENSE: 8, LaneName.OBS: 2}
        p2 = pol.load_policy("other", doc)
        assert p2.lane_gates == {LaneName.DENSE: 0, LaneName.LEX: 0}

    @pytest.mark.parametrize(
        "gates",
        [
            {"bogus_lane": 0},
            {"dense": -1},
            {"dense": True},
            {"dense": 1.5},
            {"dense": "sometimes"},
            {"dense": {"nope": 3}},
            {"dense": {"top_n": -2}},
            {"dense": {"top_n": "many"}},
            ["dense"],
        ],
    )
    def test_bad_gates_rejected(self, gates):
        with pytest.raises(VerbatimError) as exc:
            pol.load_policy("local_memory", {"lane_gates": gates})
        assert exc.value.code == ErrorCode.VALIDATION

    def test_policy_view_reports_gates(self):
        p = pol.load_policy("local_memory", {"lane_gates": {"dense": 0}})
        view = pol.policy_view(p)
        assert view["lane_gates"] == {"dense": 0}
        assert json.loads(json.dumps(view)) == view  # byte-stable

    def test_ablation_keeps_gates_on_surviving_lanes(self):
        p = pol.load_policy(
            "local_memory", {"lane_gates": {"dense": 0, "lex": 5}}
        )
        ablated = pol.ablation_lanes(p, ["dense"])
        assert ablated.lane_gates == {LaneName.LEX: 5}
        assert p.lane_gates == {LaneName.DENSE: 0, LaneName.LEX: 5}

    def test_pipeline_excludes_gated_lane(self):
        """J13 pipeline-level: gated lane's unique units never fuse."""
        register_lane(LaneName.LEX, fake_lane("lex", ["a"]))
        register_lane(LaneName.DENSE, fake_lane("dense", ["b", "a"]))
        policy = pol.load_policy(
            "test",
            {"lanes": ["lex", "dense"], "lane_gates": {"dense": "exclude"}},
        )
        result = run_search(make_ctx(policy), _query(), 1000.0)
        assert {f.unit_id for f in result.fused} == {"a"}
        # the lane still RAN — gating is containment, not enablement
        assert result.lanes["dense"].status == LaneStatus.OK
        note = result.coverage.rerank["stages"]["fusion"]
        assert note["lane_gates"] == {"dense": 0}
        assert note["lanes_gated"]["dense"]["dropped"] == 2

    def test_pipeline_gated_lane_answers_alone(self):
        """J13 pipeline-level: with no ungated lane producing candidates,
        the gated lane still answers."""
        register_lane(LaneName.LEX, fake_lane("lex", []))
        register_lane(LaneName.DENSE, fake_lane("dense", ["b"]))
        policy = pol.load_policy(
            "test",
            {"lanes": ["lex", "dense"], "lane_gates": {"dense": "exclude"}},
        )
        result = run_search(make_ctx(policy), _query(), 1000.0)
        assert {f.unit_id for f in result.fused} == {"b"}
        note = result.coverage.rerank["stages"]["fusion"]
        assert note["gates_bypassed"] is True

    def test_pipeline_top_n_gate(self):
        register_lane(LaneName.LEX, fake_lane("lex", ["a"]))
        register_lane(LaneName.DENSE, fake_lane("dense", ["d1", "d2", "d3"]))
        policy = pol.load_policy(
            "test", {"lanes": ["lex", "dense"], "lane_gates": {"dense": 1}}
        )
        result = run_search(make_ctx(policy), _query(), 1000.0)
        ids = {f.unit_id for f in result.fused}
        assert "d1" in ids and "a" in ids
        assert "d2" not in ids and "d3" not in ids


# ===========================================================================
# Cross-cutting determinism
# ===========================================================================


def test_gates_and_bonus_deterministic_across_input_order():
    outs_a = [
        _lane("lex", [_cand("a", "lex", 1, signals={"facet": 0})]),
        _lane(
            "dense",
            [
                _cand("a", "dense", 1, signals={"facet": 1}),
                _cand("x", "dense", 2),
            ],
        ),
        _lane("ent", [_cand("z", "ent", 1)]),
    ]
    outs_b = [outs_a[2], outs_a[0], outs_a[1]]
    kw = {"lane_gates": {"ent": 0}, "weights_tag": "retrieval_policy/v7"}
    r1 = rrf_fuse(outs_a, {"lex": 1.2}, **kw)
    r2 = rrf_fuse(outs_b, {"lex": 1.2}, **kw)
    assert [(c.unit_id, c.rrf) for c in r1] == [(c.unit_id, c.rrf) for c in r2]
    assert r1.stats["facet_bonus"]["applied"] == r2.stats["facet_bonus"]["applied"]
