"""Deterministic query analysis + update-candidate detection (V5).

Worker contract docs/v5_contracts.md §8:

- ``analyze(query) -> QueryAnalysis`` — ``query_analysis/v1`` classes,
  extracted identifiers/entities, temporal intent resolved to a
  bitemporal hint (SPEC_V5 §31.1). Pure, versioned, no model.
- ``detect_update_candidates(conn, namespace, new_record)`` — prior live
  records sharing subject/entity/type but differing in value, polarity,
  version, or time; writes ``update_candidates`` rows ``state='open'``
  (§30.5). Advisory only — nothing changes lifecycle.
- ``adopt_candidate`` / ``dismiss_candidate`` resolve open rows;
  ``list_open_candidates`` feeds §13 completeness checks (V5-30.21).
"""

from .analyze import (
    HINT_CURRENT,
    HINT_KNOWN_AT,
    HINT_TIMELINE,
    Mention,
    QueryAnalysis,
    TemporalView,
    analyze,
)
from .updates import (
    MAX_CANDIDATES,
    PRIOR_SCAN_LIMIT,
    UPDATE_DETECTOR_VERSION,
    DetectedCandidate,
    NewRecord,
    adopt_candidate,
    detect_update_candidates,
    dismiss_candidate,
    list_open_candidates,
    possible_updates,
)
from .calibration import (
    CALIBRATED_ENCODERS,
    CALIBRATION_VERSION,
    SimilarityCalibration,
    calibration_for,
    is_calibrated,
    similarity_floor,
)
from .verdict import (
    DEFAULT_TERM_FLOOR_RATIO,
    VERDICT_VERSION,
    SupportVerdict,
    search_verdict,
)

__all__ = [
    "analyze",
    "QueryAnalysis",
    "Mention",
    "TemporalView",
    "HINT_CURRENT",
    "HINT_TIMELINE",
    "HINT_KNOWN_AT",
    "NewRecord",
    "DetectedCandidate",
    "detect_update_candidates",
    "adopt_candidate",
    "dismiss_candidate",
    "list_open_candidates",
    "possible_updates",
    "UPDATE_DETECTOR_VERSION",
    "MAX_CANDIDATES",
    "PRIOR_SCAN_LIMIT",
    "search_verdict",
    "SupportVerdict",
    "VERDICT_VERSION",
    "DEFAULT_TERM_FLOOR_RATIO",
    "CALIBRATION_VERSION",
    "CALIBRATED_ENCODERS",
    "SimilarityCalibration",
    "calibration_for",
    "is_calibrated",
    "similarity_floor",
]
