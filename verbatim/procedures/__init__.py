"""Procedural memory v3: the ``coding_rules_v1`` reference producer
(SPEC_V3 §21–§22).

Completed coding episodes compile into evidence-backed, advisory
procedures — never executable code, never authority (V3-21.01). The
promotion ladder ``candidate → reviewed → active`` requires explicit
review in v3.0 (V3-22.04); applicability is three-valued and unknown is
never a wildcard (V3-22.15).

This package is side-effect free on import: every mutation takes the
caller's transaction ``conn`` and reads return plain ``dict`` snapshots
(docs/v3_contracts.md).
"""

from .applicability import ApplicabilityResult, check_applicability
from .compiler import (
    CompilationResult,
    ProcedureCompiler,
    compile_episode,
    refine_procedure,
)
from .exposures import exposures_for, record_exposure, reuse_stats
from .operations import classify
from .review import (
    activate,
    requires_paired_evidence,
    review,
    review_procedure,
    suspend,
)
from .signatures import (
    COMPILER_MANIFEST,
    compute_signature,
    find_by_signature,
    index_procedure,
    index_scope,
)
from .transfer import (
    TRANSFER_POLICY_ID,
    Counterexample,
    TransferDelivery,
    counterexamples,
    deliver_procedure,
    transfer_success,
)

__all__ = [
    # compilation (§22)
    "compile_episode",
    "refine_procedure",
    "CompilationResult",
    "ProcedureCompiler",
    "classify",
    # signatures (§21.04)
    "compute_signature",
    "find_by_signature",
    "index_procedure",
    "index_scope",
    "COMPILER_MANIFEST",
    # promotion ladder (§22.04)
    "review",
    "review_procedure",
    "activate",
    "suspend",
    "requires_paired_evidence",
    # applicability (§22.15)
    "check_applicability",
    "ApplicabilityResult",
    # reuse accounting (§22.06)
    "record_exposure",
    "exposures_for",
    "reuse_stats",
    # env- and counterexample-qualified transfer (v4.5 I4, V45-06)
    "deliver_procedure",
    "transfer_success",
    "counterexamples",
    "TransferDelivery",
    "Counterexample",
    "TRANSFER_POLICY_ID",
]
