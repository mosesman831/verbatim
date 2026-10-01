"""Verbatim provider + run driver for the Agent Memory Benchmark
(AMB) — SPEC_V8 §15.2 (V8-15.06–15.13; closes D8-27).

* :mod:`eval.amb.provider` — the ``MemoryProvider``-contract adapter:
  real ``Memory.add`` ingest, ``Memory.search`` with ``as_of`` +
  ``timeout_ms=500``, pack-renderer context at a row token budget,
  per-query token records for the accuracy-vs-tokens curve.
* :mod:`eval.amb.manifest` — run manifests (provider version, verbatim
  head commit, AMB pin placeholders, dataset pin, model ids marked
  ``pending_authorization``, arm config).
* :mod:`eval.amb.runner` — the authorization-gated CLI: without the
  §26 O-decision records every row emits ``blocked_on_authorization``
  and exits nonzero with the report written; model calls go through a
  pluggable backend whose default raises — never a silent stub.

CODE ONLY: no AMB run has executed — O2/O5/O6 authorizations are not
recorded, so every execution path reports ``blocked_on_authorization``
(not_run discipline).
"""

from .provider import (  # noqa: F401
    AMBDoc,
    DEFAULT_TIMEOUT_MS,
    DOC_MODES,
    PROVIDER_NAME,
    PROVIDER_VERSION,
    TOKEN_BUDGETS,
    VerbatimAMBProvider,
    context_string,
    register,
    retrieval_context,
)

__all__ = [
    "AMBDoc",
    "DEFAULT_TIMEOUT_MS",
    "DOC_MODES",
    "PROVIDER_NAME",
    "PROVIDER_VERSION",
    "TOKEN_BUDGETS",
    "VerbatimAMBProvider",
    "context_string",
    "register",
    "retrieval_context",
]
