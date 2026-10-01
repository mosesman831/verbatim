"""Influence/exposure accounting for the consumer delivery path
(SPEC_V6 V6-03.14, docs/v6_contracts.md §8).

The claim lane already instruments deliveries: ``recall_v3`` mints one
``influence`` row per pack item inside its write phase (V3-32.02). The
consumer ``Memory.search`` path delivers *source-lane* items but, until
V6, minted zero exposure rows — leaving ``exposure_rows_recorded``
inconclusive. This package is the source-side twin: ``source_exposure``
journals one row per delivered source item, written by the post-delivery
hook ``emit_deliveries`` in its own short write transaction — the same
discipline as ``controller.log_decision`` (never inside the read tx).

The claim-lane machinery itself stays in
``verbatim.retrieval.v3.influence``; this package adds the consumer-path
exposure sink only.
"""

from .exposure import (
    emit_deliveries,
    exposure_count,
    record_source_deliveries,
    recent_exposures,
)

__all__ = [
    "emit_deliveries",
    "exposure_count",
    "record_source_deliveries",
    "recent_exposures",
]
