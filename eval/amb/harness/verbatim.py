"""Verbatim memory provider — V8.5 adapter over eval.amb.provider.

The real implementation lives in the verbatim repo
(``eval/amb/provider.py::VerbatimAMBProvider``); this file is the
AMB-facing ``MemoryProvider`` shell.  Env knobs:

- ``AMB_VERBATIM_REPO`` — path to the verbatim checkout (its
  ``verbatim`` package and ``eval.amb`` provider must be importable).
- ``AMB_VERBATIM_CONCURRENCY`` — runner semaphore width; bounds the
  whole per-query pipeline (retrieve + answer + judge), not just the
  verbatim call.  Retrieve holds a write lock only around ingest, so
  raising this parallelizes the LLM-bound portion safely.
- ``VERBATIM_AMB_TOKEN_BUDGET`` / ``VERBATIM_AMB_NEIGHBOR_W`` /
  ``VERBATIM_AMB_SEARCH_LIMIT`` — delivery budget and expansion arms.
"""

import os
import sys
from pathlib import Path

from .base import MemoryProvider
from ..models import Document

_REPO = os.environ.get("AMB_VERBATIM_REPO", "")
if not _REPO or not Path(_REPO, "eval", "amb", "provider.py").exists():
    raise RuntimeError(
        "AMB_VERBATIM_REPO must point at the verbatim repo checkout "
        "(the directory containing eval/amb/provider.py); "
        f"got {_REPO!r}"
    )
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from eval.amb.provider import VerbatimAMBProvider  # noqa: E402


class VerbatimProvider(MemoryProvider):
    name = "verbatim"
    description = "Verbatim local memory engine (V8.5 pipeline)"
    kind = "local"
    provider = None
    variant = None
    link = None
    logo = None
    supports_filters = False
    concurrency = int(os.environ.get("AMB_VERBATIM_CONCURRENCY", "4"))

    def __init__(self) -> None:
        self._impl = VerbatimAMBProvider()

    def initialize(self) -> None:
        self._impl.initialize()

    def prepare(
        self,
        store_dir: Path,
        unit_ids: set[str] | None = None,
        reset: bool = True,
    ) -> None:
        self._impl.prepare(store_dir, unit_ids=unit_ids, reset=reset)

    def cleanup(self) -> None:
        self._impl.cleanup()

    def set_extraction_labels(self, labels) -> None:
        if hasattr(self._impl, "set_extraction_labels"):
            self._impl.set_extraction_labels(labels)

    def ingest(self, documents: list[Document]) -> None:
        self._impl.ingest(documents)

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
        filters: dict | None = None,
    ) -> tuple[list[Document], dict | None]:
        return self._impl.retrieve(
            query, k=k, user_id=user_id,
            query_timestamp=query_timestamp, filters=filters,
        )
