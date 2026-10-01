"""verbatim.profiles — SPEC_V4 §21 profiles, preferences, and
perspective-aware personalization.

Derived, evidence-backed profile state:

* ``ProfileService.compile`` derives ``observed``/``inferred`` entries
  from live claim evidence — on-demand inside a transaction (a durable
  ``profile_compile`` job kind is *deferred*: widening the jobs-table
  CHECK constraint is schema-owned, so the report says so rather than
  pretending a queue lane exists).
* ``upsert_entry`` records ``explicit``/``task_local``/``sensitive``
  declarations; every entry is revisioned, attributable, and carries
  derivation parents in ``derivations`` + ``dependency_edges``
  (``child_kind='profile'``).
* Contradictions keep every alternative under a shared conflict group;
  nothing silently resolves.
* Reads are audience-closed (empty audience = subject-only) and every
  derivation parent is re-verified through ``Kernel`` —
  purged/quarantined evidence withholds the entry.
* Perspective packs narrow/rank a caller's own candidates only —
  attenuation, never widening; ``quote``-less callers get ranked items
  that are never marked liftable.
* ``refresh`` reconciles stored entries against evidence viability
  (expire task-locals, withhold invalidated, tombstone the fully
  unsupported, scrub pointers to erased objects, revive recovered).
"""

from __future__ import annotations

from typing import Any

from .service import ProfileService
from .types import (
    ConflictPolicy,
    EntryKind,
    EntryState,
    InferenceMode,
    Multiplicity,
    PerspectivePack,
    Sensitivity,
    TopicSpec,
)

__all__ = [
    "ProfileService",
    "PerspectivePack",
    "TopicSpec",
    "EntryKind",
    "EntryState",
    "Multiplicity",
    "Sensitivity",
    "ConflictPolicy",
    "InferenceMode",
    "probe_capability",
]


def probe_capability(store: Any) -> dict:
    """Runtime probe for the profiles capability lane (V4-50.01/02).

    ``healthy`` when the module's tables are materialized and the store
    is writable-shaped; ``implemented`` when the code is here but the
    tables have not been created yet (they materialize inside the first
    write transaction). The durable compile job is reported as deferred,
    never implied.
    """
    from ..storage.repos import has_table

    details: dict[str, Any] = {
        "compile_mode": "on_demand_in_transaction",
        "durable_job": "deferred",
        "durable_job_reason": (
            "no profile_compile JobKind — widening the jobs-table CHECK "
            "constraint is schema-owned"
        ),
        "closure_integration": (
            "derivations-graph discovery + read-path kernel gating; "
            "profile rows classify outside the physically-erasable set "
            "(privacy/deletion kind registry)"
        ),
    }
    try:
        with store.read() as conn:
            present = has_table(conn, "profile_entries") and has_table(
                conn, "profile_topics"
            )
            entries = (
                conn.execute("SELECT COUNT(*) FROM profile_entries").fetchone()[0]
                if present
                else 0
            )
    except Exception:
        return {
            "state": "unavailable",
            "degraded_reason": "store not readable for profile probe",
            "details": details,
        }
    details["tables_present"] = present
    details["entry_rows"] = entries
    if present:
        return {"state": "healthy", "degraded_reason": None,
                "details": details}
    return {
        "state": "implemented",
        "degraded_reason": (
            "profile tables materialize inside the first write "
            "transaction; durable compile job deferred"
        ),
        "details": details,
    }
