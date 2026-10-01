"""V5 source-pipeline dispatch map (docs/v5_contracts.md §5, SPEC_V5 §08.16).

Same lazy-import convention as ``ingest._V3_KIND_HANDLERS``: each kind
maps to ``(module, function)`` and the integration session merges this
into ``Ingester._execute``'s dispatch chain. An absent module fails
``CAPABILITY_UNAVAILABLE`` — loud, never a silent no-op.

Lane note (V5-08.16): ``source_project``/``source_embed`` run on the
ordinary lane for fresh writes; ``source_backfill`` runs maintenance.
The lane is chosen at enqueue (``enqueue_source_jobs`` /
``enqueue_source_backfill``), not by the handler.
"""

from __future__ import annotations

from ..core.types import JobKind

V5_KIND_HANDLERS = {
    JobKind.SOURCE_PROJECT: (
        "verbatim.jobs.source_jobs",
        "handle_source_project",
    ),
    JobKind.SOURCE_EMBED: (
        "verbatim.jobs.source_jobs",
        "handle_source_embed",
    ),
    JobKind.SOURCE_BACKFILL: (
        "verbatim.jobs.source_jobs",
        "handle_source_backfill",
    ),
}

__all__ = ["V5_KIND_HANDLERS"]
