"""Rank fusion over lane hits (SPEC_V3 §29.01–29.03, §29.10).

Baseline fusion is reciprocal rank fusion
``R(c) = Σ w_lane / (60 + rank_lane(c))`` with one-based ranks and zero
contribution for absent lanes. Lane weights are the published §29.01
values: 1.0 for exact/lexical/dense/structured/procedural, 0.75 for
sparse/late/temporal, 0.5 for episode/graph/causal/freshness — versioned
per development set, never mixed onto one raw-score scale (V3-29.02).

Bounded multiplicative adjustments apply *after* fusion among equally
eligible items (V3-29.03): recency ±10 %, temporal proximity ±10 %, proof
count ±5 %. Adjustments never override eligibility tiers and never
suppress required contrary evidence. Final ties break on stable object
identifiers (V3-29.10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

RRF_K = 60

# §29.01 lane weights (versioned per development set).
LANE_WEIGHTS: dict[str, float] = {
    "exact_id": 1.0,
    "lexical": 1.0,
    "dense": 1.0,
    "structured": 1.0,
    "procedural_signature": 1.0,
    "sparse": 0.75,
    "late": 0.75,
    "temporal": 0.75,
    "episode_hierarchy": 0.5,
    "graph": 0.5,
    "causal": 0.5,
    "freshness_env": 0.5,
    "browse": 0.5,
    "working": 0.5,
    "dependency": 0.0,
}

# Bounded post-fusion adjustment magnitudes (V3-29.03).
RECENCY_RANGE = 0.10
TEMPORAL_RANGE = 0.10
PROOF_RANGE = 0.05

FUSION_PROFILE = "fusion_v3_r1"


@dataclass
class RankedEntry:
    """One fused candidate: object key plus final score and its hit."""

    key: tuple          # (object_kind, object_id)
    score: float
    hit: Any            # UnionHit


class Ranked(list):
    """``list[RankedEntry]`` carrying the union for downstream stages."""

    def __init__(self, rows: list, union=None) -> None:
        super().__init__(rows)
        self.union = union


def _recency_factor(hit: Any, max_recorded: int) -> float:
    """±10 % recency prior (V3-29.03): newest candidate earns the boost."""
    if max_recorded <= 0 or hit.recorded_from <= 0:
        return 1.0
    norm = hit.recorded_from / max_recorded  # ∈ (0, 1]
    return 1.0 + RECENCY_RANGE * (2.0 * norm - 1.0)


def _temporal_factor(hit: Any, query_point: Optional[int],
                     covering: set) -> float:
    """±10 % temporal-proximity prior for time-qualified queries."""
    if query_point is None:
        return 1.0
    if hit.key in covering:
        return 1.0 + TEMPORAL_RANGE
    return 1.0 - TEMPORAL_RANGE * 0.0  # no penalty: ± bound kept one-sided


def _proof_factor(hit: Any) -> float:
    """±5 % proof-count prior (derived objects only)."""
    if not getattr(hit, "proof_count", 0):
        return 1.0
    return 1.0 + min(PROOF_RANGE, hit.proof_count * 0.005)


def fuse(union: Any, lane_weights: Optional[dict] = None,
         covering_keys: Optional[set] = None,
         query_point_us: Optional[int] = None) -> Ranked:
    """Fuse per-lane ranks into one deterministic ordering.

    ``covering_keys`` — object keys whose valid interval provably covers
    the query's ``query_point_us`` (supplied by the caller's temporal
    probe); powers the bounded temporal-proximity prior. Pure function:
    no I/O, no clock.
    """
    w = LANE_WEIGHTS if lane_weights is None else lane_weights
    max_recorded = max(
        (h.recorded_from for h in union.values()), default=0
    )
    covering = covering_keys or set()
    scored: list = []
    for key, hit in union.items():
        base = sum(
            w.get(lane, 0.0) / (RRF_K + rank)
            for lane, rank in hit.lane_ranks.items()
            if rank > 0
        )
        score = base
        score *= _recency_factor(hit, max_recorded)
        score *= _temporal_factor(hit, query_point_us, covering)
        score *= _proof_factor(hit)
        scored.append(RankedEntry(key, score, hit))
    scored.sort(
        key=lambda e: (
            -e.score,
            -e.hit.recorded_from,
            e.key[0],
            e.key[1],
        )
    )
    return Ranked(scored, union=union)
