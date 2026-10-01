"""Evidence-access and delivery kernel (SPEC_V4 §08–§09).

Single gate between stored evidence and consumers. Contract surface:

- ``resolve_access`` — per-scope authorization + ancestry/lifecycle/
  quarantine/suppression/retention evaluation → ``EligibilityLease``
  (denials return ``denied=True`` indistinguishably; ``VALIDATION`` and
  ``STALE_EPOCH`` are the only raised, distinguishable classes).
- ``read_verified`` — digest re-authenticated byte reads under a live
  lease; ``quote`` required for canonical bytes, ``read`` yields metadata
  plus approved derived views only (V4-08.02); group-atomic (V4-08.05).
- ``derive_inputs`` — verified input bundle + intersected inherited
  restrictions for producers (V4-10.07, V4-11.05).
- ``seal_delivery`` — disclosure linearization: contributing scopes and
  dependency versions rechecked inside one tx before the permit commits
  (V4-09.06); permits age out at 1s by default (V4-09.08).
- ``open_dispatch`` — lazy delegation to ``privacy.broker`` (owned by
  another worker); ``CAPABILITY_UNAVAILABLE`` when unprovisioned.
- ``invalidate`` — epoch bumps + affected-object obligations for
  quarantine/correction/revocation/erasure events (V4-09.07).
- ``review_metadata`` — scope-local metadata flows only inside a
  lease-reviewed structure (V4-08.06).
"""

from .service import (
    DENIAL_MESSAGE,
    DerivedInputs,
    EvidenceKernel,
    INVALIDATION_KINDS,
    InvalidationReport,
    Kernel,
    ReviewedMetadata,
)

__all__ = [
    "DENIAL_MESSAGE",
    "DerivedInputs",
    "EvidenceKernel",
    "INVALIDATION_KINDS",
    "InvalidationReport",
    "Kernel",
    "ReviewedMetadata",
]
