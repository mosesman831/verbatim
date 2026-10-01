"""``ranking/v1`` deterministic fusion contract tests (SPEC_V5 §31.2).

The fusion contract: per-signal min-max normalization over the admitted
candidate set, declared weight sum, declared tie-breaks
(score desc → source_id asc → revision desc), ``score_detail`` on every
hit, and pure determinism — identical inputs give identical output.
Fusion reorders an admitted list; it never admits or drops for
eligibility (V5-31.05).
"""

from __future__ import annotations

import math

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.retrieval.v3 import fusion_v1 as fusion
from verbatim.retrieval.v3.source_lane import SourceHit


def _hit(sid, rev=1, **signals):
    h = SourceHit(sid, rev)
    h.signals.update(signals)
    return h


# ---------------------------------------------------------------------------
# determinism + tie-breaks
# ---------------------------------------------------------------------------


def test_fuse_deterministic_identical_inputs():
    cands = [
        _hit("s1", 1, lexical=2.0, similarity=0.5),
        _hit("s2", 1, lexical=4.0, similarity=0.1),
        _hit("s3", 1, similarity=0.9, identifier_hit=1.0),
    ]
    first = fusion.fuse(cands)
    second = fusion.fuse(list(cands))
    assert [(h.source_id, h.revision, h.score) for h in first] == [
        (h.source_id, h.revision, h.score) for h in second
    ]
    # and input order cannot matter — shuffled input, same output
    third = fusion.fuse([cands[2], cands[0], cands[1]])
    assert [(h.source_id, h.revision, h.score) for h in first] == [
        (h.source_id, h.revision, h.score) for h in third
    ]


def test_tie_break_source_id_then_revision_desc():
    # identical signals → identical score → source_id asc decides
    cands = [
        _hit("b", 1, lexical=1.0),
        _hit("a", 1, lexical=1.0),
        _hit("c", 1, lexical=1.0),
    ]
    out = fusion.fuse(cands)
    assert [h.source_id for h in out] == ["a", "b", "c"]
    # same source_id, two revisions, equal score → revision desc
    cands = [_hit("s", 1, lexical=1.0), _hit("s", 3, lexical=1.0),
             _hit("s", 2, lexical=1.0)]
    out = fusion.fuse(cands)
    assert [h.revision for h in out] == [3, 2, 1]
    # mixed: score dominates, ties fall to source_id then revision
    cands = [
        _hit("z", 1, lexical=1.0),
        _hit("a", 2, lexical=2.0),
        _hit("a", 1, lexical=2.0),
    ]
    out = fusion.fuse(cands)
    assert [(h.source_id, h.revision) for h in out] == [
        ("a", 2), ("a", 1), ("z", 1)
    ]


def test_ranks_assigned_after_sort():
    out = fusion.fuse([_hit("b", 1, lexical=2.0), _hit("a", 1, lexical=1.0)])
    assert [h.rank for h in out] == [1, 2]
    assert out[0].source_id == "b"


# ---------------------------------------------------------------------------
# normalization + weights
# ---------------------------------------------------------------------------


def test_min_max_normalization_over_candidate_set():
    # lexical raw 1..3 → normalized 0, .5, 1; similarity constant →
    # degenerate range → all present-positive → normalized 1.0 each.
    cands = [
        _hit("s1", 1, lexical=1.0, similarity=0.4),
        _hit("s2", 1, lexical=2.0, similarity=0.4),
        _hit("s3", 1, lexical=3.0, similarity=0.4),
    ]
    out = {h.source_id: h for h in fusion.fuse(cands)}
    w_lex = fusion.RANKING_V1_WEIGHTS["lexical"]
    w_sim = fusion.RANKING_V1_WEIGHTS["similarity"]
    # s1's lexical normalized to 0 → not a contributing signal; its raw
    # value stays visible in ``signals``.
    assert "lexical" not in out["s1"].score_detail
    assert out["s1"].signals["lexical"] == 1.0
    assert out["s2"].score_detail["lexical"]["normalized"] == pytest.approx(0.5)
    assert out["s3"].score_detail["lexical"]["normalized"] == pytest.approx(1.0)
    assert out["s1"].score_detail["similarity"]["normalized"] == 1.0
    assert out["s1"].score == pytest.approx(0.0 * w_lex + 1.0 * w_sim)
    assert out["s3"].score == pytest.approx(1.0 * w_lex + 1.0 * w_sim)


def test_identifier_hit_dominates_on_identifier_class():
    """V5-30.17: exact identifier hit outranks any other signal mix."""
    cands = [
        _hit("fuzzy", 1, lexical=9.0, similarity=0.99,
             entity_overlap=3.0, temporal_match=1.0,
             type_affinity=1.0, corroboration=5.0, lifecycle_current=1.0),
        _hit("exact", 1, identifier_hit=1.0),
    ]
    out = fusion.fuse(cands, query_class="identifier")
    assert out[0].source_id == "exact"
    # default table: identifier weight 2.0 still dominates a similarity-
    # only near miss ("vector similarity cannot substitute", V5-10.05)
    out = fusion.fuse([_hit("near", 1, similarity=0.99),
                       _hit("exact", 1, identifier_hit=1.0)])
    assert out[0].source_id == "exact"


def test_weights_table_is_frozen():
    with pytest.raises(TypeError):
        fusion.RANKING_V1_WEIGHTS["lexical"] = 99.0
    with pytest.raises(TypeError):
        fusion.RANKING_V1_CLASS_WEIGHTS["identifier"]["lexical"] = 0.0


def test_explicit_weights_override():
    cands = [_hit("a", 1, lexical=1.0), _hit("b", 1, similarity=1.0)]
    out = fusion.fuse(cands, weights={"similarity": 5.0, "lexical": 0.1})
    assert out[0].source_id == "b"
    assert out.weights == {"similarity": 5.0, "lexical": 0.1}


# ---------------------------------------------------------------------------
# contract honesty
# ---------------------------------------------------------------------------


def test_fusion_never_admits_or_invents():
    cands = [_hit("a", 1, lexical=1.0), _hit("b", 1, lexical=2.0)]
    out = fusion.fuse(cands, signals={("c", 1): {"lexical": 99.0}})
    assert {h.source_id for h in out} == {"a", "b"}   # "c" never appears
    # limit truncates the reordered list — reported, not silent
    out = fusion.fuse(cands, limit=1)
    assert len(out) == 1 and out.truncated and out.stats["truncated"]


def test_score_detail_names_contributing_signals():
    out = fusion.fuse([
        _hit("s1", 1, lexical=2.0, identifier_hit=1.0),
        _hit("s2", 1, lexical=1.0),
    ])
    detail = out[0].score_detail
    for name, parts in detail.items():
        assert set(parts) == {"value", "normalized", "weight",
                              "contribution"}
        assert parts["contribution"] == pytest.approx(
            parts["weight"] * parts["normalized"]
        )
    # s1 contributed lexical + identifier_hit; a zero-normalized signal
    # is not "contributing" (it names contributing signals, §31.06)
    assert set(out[0].score_detail) == {"identifier_hit", "lexical"}
    assert set(out[1].score_detail) == set()   # all signals normalized 0


def test_signals_input_forms_equivalent():
    cands = [("a", 1), ("b", 1), ("c", 2)]
    major = {
        ("a", 1): {"lexical": 2.0},
        ("b", 1): {"lexical": 1.0},
        ("c", 2): {"lexical": 3.0},
    }
    by_signal = {"lexical": {("a", 1): 2.0, ("b", 1): 1.0, ("c", 2): 3.0}}
    carried = [_hit("a", 1, lexical=2.0), _hit("b", 1, lexical=1.0),
               _hit("c", 2, lexical=3.0)]
    out1 = [(h.key, h.score) for h in fusion.fuse(cands, major)]
    out2 = [(h.key, h.score) for h in fusion.fuse(cands, by_signal)]
    out3 = [(h.key, h.score) for h in fusion.fuse(carried)]
    assert out1 == out2 == out3


def test_missing_and_nonfinite_signals():
    cands = [
        _hit("a", 1, lexical=1.0),
        _hit("b", 1),                          # no signals at all
        _hit("c", 1, lexical=float("nan")),    # non-finite → dropped
    ]
    out = fusion.fuse(cands)
    assert out.stats["nonfinite_dropped"] == 1
    assert out[0].source_id == "a"
    # missing signal contributes 0 — never a negative share
    assert out[-1].score == 0.0


def test_duplicate_candidates_dropped_first_wins():
    cands = [_hit("a", 1, lexical=1.0), _hit("a", 1, lexical=9.0)]
    out = fusion.fuse(cands)
    assert len(out) == 1
    assert out.stats["duplicates_dropped"] == 1
    assert out[0].signals["lexical"] == 1.0


def test_fuse_contract_metadata():
    out = fusion.fuse([_hit("a", 1, lexical=1.0)])
    assert out.version == "ranking/v1"
    assert out.tie_break == ("score_desc", "source_id_asc",
                           "revision_desc")
    assert out.stats["candidates"] == 1
    assert out.stats["signals"]["lexical"]["min"] == 1.0


def test_fuse_rejects_malformed_candidate():
    with pytest.raises(VerbatimError) as ei:
        fusion.fuse([object()])
    assert ei.value.code is ErrorCode.VALIDATION


def test_empty_candidates():
    out = fusion.fuse([])
    assert out == [] and out.stats["candidates"] == 0


def test_deadline_param_marks_partial():
    from verbatim.retrieval.candidates import Deadline
    out = fusion.fuse(
        [_hit(f"s{i}", 1, lexical=float(i)) for i in range(50)],
        deadline=Deadline(0.0),
    )
    assert out.stats["partial"] is True
    assert out.stats["deadline_exceeded"] is True


def test_lifecycle_current_edge_over_superseded():
    """Among admitted candidates with otherwise-equal evidence, the
    ``lifecycle_current`` signal keeps the current record ahead of the
    superseded one — fusion's declared share of lifecycle discipline
    (strict superseded exclusion for current-state queries is the
    eligibility layer's job; signals only rank what was admitted)."""
    cands = [
        _hit("old", 1, lexical=2.0),                    # superseded
        _hit("cur", 1, lexical=2.0, lifecycle_current=1.0),
    ]
    out = fusion.fuse(cands)
    assert out[0].source_id == "cur"
    assert out[0].score > out[1].score
    assert out[0].score_detail["lifecycle_current"]["contribution"] > 0
