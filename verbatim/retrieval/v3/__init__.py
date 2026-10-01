"""Verbatim v3 retrieval and controller layer (SPEC_V3 §26–§32).

Vertical slice: deterministic routing, multi-lane candidate union,
eligibility-first filtering, bounded candidate caps, calibrated
abstention, typed context packs under shared budgets, influence handles,
and routing-decision logs suitable for replay.

Pipeline (``recall_v3``):

    authorize → analyze → plan_routes → lanes → union → fuse →
    groups → abstain → packs → influence → decision log
"""

from .abstain import AbstentionVerdict, abstain, extract_hard_identifiers
from .controller import RouteSet, discretize, log_decision, plan_routes
from .fusion_v3 import LANE_WEIGHTS, Ranked, RankedEntry, fuse
from .influence import (
    BlastRadius,
    blast_radius,
    mint_handle,
    record_delivery,
    record_feedback,
)
from .lanes import LaneContext, LaneResult, LaneRunReport, run_lanes
from .pack import (
    AssemblyResult,
    DetailTier,
    assemble_packs,
    normalize_detail_tier,
)
from .recall import classify, context_v3, expand_item, recall_v3
from .union import CandidateUnion, UnionHit, build_union

__all__ = [
    "recall_v3",
    "context_v3",
    "expand_item",
    "DetailTier",
    "normalize_detail_tier",
    "plan_routes",
    "classify",
    "discretize",
    "log_decision",
    "RouteSet",
    "run_lanes",
    "LaneContext",
    "LaneResult",
    "LaneRunReport",
    "build_union",
    "CandidateUnion",
    "UnionHit",
    "fuse",
    "Ranked",
    "RankedEntry",
    "LANE_WEIGHTS",
    "abstain",
    "AbstentionVerdict",
    "extract_hard_identifiers",
    "assemble_packs",
    "AssemblyResult",
    "mint_handle",
    "record_delivery",
    "record_feedback",
    "blast_radius",
    "BlastRadius",
]
