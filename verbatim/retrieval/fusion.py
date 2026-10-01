"""Reciprocal-rank fusion and deterministic final ordering (SPEC §30).

BM25, cosine, and slot-match scores are incomparable, so fusion uses rank,
not raw score: ``R(c) = Σ w_s / (60 + rank_s(c))`` over the sources that
retrieved the claim, with one-based ranks and zero contribution when
absent. The final order is relevance tier, fused score, recency, then a
stable claim-id tiebreak — the same input always yields the same order.
"""

from __future__ import annotations

from typing import Optional

DEFAULT_WEIGHTS = {"lexical": 1.0, "semantic": 1.0, "structured": 1.0, "graph": 0.5}
RRF_K = 60


class Ranked(list):
    """``list[(claim_id, score)]`` carrying the candidate map for packaging.

    ``hits`` lets ``package`` recover per-claim reasons and the resolved
    revision without re-running eligibility.
    """

    def __init__(self, rows: list, hits=None) -> None:
        super().__init__(rows)
        self.hits = hits


def _tier(claim_id: str, hits) -> int:
    """Exact-match tier (SPEC §30): structured-slot and verified-phrase hits
    rank above everything else; substring luck does not qualify."""
    hit = hits[claim_id]
    if "structured" in hit.source_ranks or "phrase_hit" in hit.reasons:
        return 0
    return 1


def rrf(hits, weights: Optional[dict] = None, k: int = RRF_K) -> Ranked:
    """Fuse per-source ranks into one deterministic ordering.

    ``hits`` maps claim_id → CandidateHit (with ``source_ranks``,
    ``reasons``, ``recorded_from``). Pure function: no I/O, no clock.
    """
    w = DEFAULT_WEIGHTS if weights is None else weights
    scored = []
    for claim_id, hit in hits.items():
        score = sum(
            w.get(source, 0.0) / (k + rank)
            for source, rank in hit.source_ranks.items()
            if rank > 0
        )
        scored.append((claim_id, score))
    scored.sort(
        key=lambda entry: (
            _tier(entry[0], hits),          # exact structured/phrase first
            -entry[1],                       # fused score, descending
            -hits[entry[0]].recorded_from,   # recency tie-break (§30)
            entry[0],                        # stable claim id
        )
    )
    return Ranked(scored, hits=hits)
