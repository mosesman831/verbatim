"""Observations and consolidation module (SPEC_V3 §23–§25, §17).

Derived beliefs consolidated from admitted facts across distinct evidence
families — the ``slot_aggregate_v1`` rules producer (V3-23.10) — plus the
freshness classes, working sets, environment state, and observer-scoped
social memory that surround them.

Public surface:
- :func:`consolidate` — full idempotent pass over a scope's eligible
  claim revisions (V3-23.01..23.10).
- :func:`aggregate_candidates` / :class:`SlotCandidate` — the raw
  slot-grouping computation behind consolidation (proof counts are
  family-distinct, V3-17.05).
- :func:`record_observation` — manual derived-observation write with
  derivation edges (V3-17.02/17.07).
- ``freshness`` helpers — :func:`set_freshness`, :func:`get_freshness`,
  :func:`mark_anchored_stale`, :func:`is_stale` (§24).
- ``working`` helpers — :func:`create_set`, :func:`add_item`,
  :func:`get_set`, :func:`list_sets` (V3-24.03).
- ``environment`` helpers — :func:`set_state`, :func:`get_state`,
  :func:`list_state`, :func:`clear_state` (V3-24.04).
- ``social`` helpers — :func:`set_record`, :func:`get_record`,
  :func:`list_records` (§25).
"""

from . import environment, freshness, social, working
from .aggregate import (
    DEFAULT_MIN_PROOF,
    ELIGIBLE_MODALITIES,
    ELIGIBLE_STATES,
    SLOT_AGGREGATE_V1,
    PersistResult,
    SlotCandidate,
    aggregate_candidates,
    observation_id_for,
    record_observation,
    render_text,
)
from .consolidate import (
    ConsolidationReport,
    ConsolidationWindow,
    consolidate,
    consolidate_windowed,
)
from .reflect import (
    REFLECT_PRODUCER_ID,
    ReflectBudget,
    ReflectReport,
    hypothesis_id_for,
    reflect,
)
from .environment import clear_state, get_state, list_state, set_state
from .freshness import (
    anchors_for,
    current_seq,
    get_freshness,
    is_stale,
    mark_anchored_stale,
    next_seq,
    set_freshness,
)
from .social import get_record, list_records, record_id_for, set_record
from .working import (
    PROMOTE_ENVELOPE_KIND,
    WORKING_SET_ITEM_CAP,
    add_item,
    create_set,
    get_set,
    item_count,
    list_sets,
    promote,
)

__all__ = [
    # submodules
    "environment",
    "freshness",
    "social",
    "working",
    # consolidation
    "aggregate_candidates",
    "consolidate",
    "consolidate_windowed",
    "ConsolidationWindow",
    "ConsolidationReport",
    # bounded reflection (SPEC_V4 §24, §27)
    "reflect",
    "ReflectBudget",
    "ReflectReport",
    "hypothesis_id_for",
    "REFLECT_PRODUCER_ID",
    "observation_id_for",
    "record_observation",
    "render_text",
    "PersistResult",
    "SlotCandidate",
    "SLOT_AGGREGATE_V1",
    "DEFAULT_MIN_PROOF",
    "ELIGIBLE_STATES",
    "ELIGIBLE_MODALITIES",
    # freshness
    "anchors_for",
    "current_seq",
    "get_freshness",
    "is_stale",
    "mark_anchored_stale",
    "next_seq",
    "set_freshness",
    # environment state
    "clear_state",
    "get_state",
    "list_state",
    "set_state",
    # working sets
    "PROMOTE_ENVELOPE_KIND",
    "WORKING_SET_ITEM_CAP",
    "add_item",
    "create_set",
    "get_set",
    "item_count",
    "list_sets",
    "promote",
    # social memory
    "get_record",
    "list_records",
    "record_id_for",
    "set_record",
]
