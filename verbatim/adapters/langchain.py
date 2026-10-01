"""LangChain retriever adapter (SPEC_V6 V6-01.09) — a duck-typed
retriever over :class:`verbatim.Memory`.

One verified framework seam: a LangChain-compatible retriever that calls
``Memory.search`` and returns ``Document``-shaped objects carrying
``source_id``. It is the *only* framework adapter this build ships —
no matrix is promised.

Duck-typed by contract: ``langchain``/``langchain_core`` are NOT runtime
dependencies and this module imports cleanly without them. When
``langchain_core.documents.Document`` is importable the real class is
used so returned documents interop natively with LangChain pipelines;
otherwise :class:`_Doc` — a minimal stand-in with the identical
``.page_content``/``.metadata`` surface (plus ``type``/``id`` parity
fields) — is returned, and it is documented as a stand-in, never passed
off as the framework type.

The retriever implements the method surface LangChain calls on a
retriever today:

- ``get_relevant_documents(query, **kw)`` — the legacy retriever
  protocol (``BaseRetriever.get_relevant_documents`` shape).
- ``aget_relevant_documents(query, **kw)`` — the async twin. LangChain's
  own default runs the sync method on an executor; this adapter does the
  same (``run_in_executor``) because ``Memory.search`` performs blocking
  store I/O, and the bound ``Store`` is already cross-thread safe (the
  managed worker drives it from a daemon thread).
- ``invoke(query, config=None, **kw)`` / ``ainvoke(...)`` — the
  LCEL ``Runnable`` protocol: ``input`` is the query string, ``config``
  is accepted and ignored.

``VerbatimRetriever`` deliberately does not subclass
``BaseRetriever``: subclassing would make the module unimportable
without the framework, and every consumer that matters (custom chains,
``RunnableLambda``, agent tools) calls the public method surface above.

Mapping rules:

- ``k`` → ``Memory.search(limit=k)``. ``limit`` is the retriever's own
  knob: passing ``limit`` inside ``search_kwargs`` or per-call kwargs is
  a typed error pointing at ``k``, never a silent double-mapping.
- ``search_kwargs`` (constructor) merge into every ``Memory.search``
  call; per-call kwargs merge over them (call-time wins). Every key must
  belong to the declared ``Memory.search`` vocabulary —
  ``filters``, ``after``, ``consistency``, ``ready_timeout_ms``,
  ``timeout_ms``, ``strict`` — an unknown key is a typed
  ``VALIDATION`` error, never silently dropped.
- LangChain plumbing kwargs (``callbacks``, ``tags``, ``metadata``,
  ``run_name``, ``run_manager``, ``config``, …) are accepted and ignored
  — they are framework call-scaffolding, not search options.

Honesty (the same rule the facade enforces — V6-01.03): a search whose
status is ``pending`` / ``blocked`` / ``unavailable`` / ``insufficient``
returns ``[]`` and records the real status on
``self._verbatim_status`` — never a fake success. ``ready``/``partial``
deliver their items; ``_verbatim_status`` (plus ``_verbatim_result``,
the raw ``SearchResult``) is updated on every call so a chain author can
surface "the memory answered pending/insufficient" instead of guessing
from an empty doc list. ``_verbatim_status`` is ``None`` until the
first search completes.

Each delivered ``Hit`` becomes one document: ``page_content`` is the
hit's quote; ``metadata`` carries ``source_id`` (the hit's source id —
``hit.memory_id`` for source hits, parsed from the ``MemoryRef``
otherwise), ``ref``, ``object_ref``, ``kind``, ``score``,
``support_status``, ``lifecycle``, and the remaining provenance fields.
Non-source hits (claim/view/card) have no source id — ``source_id`` is
``""`` and ``object_ref``/``kind`` carry the identity.
"""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.types import ErrorCode, VerbatimError
from ..memory.types import MemoryRef

#: Real framework Document when langchain_core is installed; otherwise
#: the _Doc stand-in below. The import is the only langchain reference
#: in the module and lives entirely inside try/except — the adapter is
#: importable with zero langchain packages present.
try:
    from langchain_core.documents import Document as _FrameworkDocument
except Exception:  # langchain_core absent — the declared baseline
    _FrameworkDocument = None


@dataclass
class _Doc:
    """Minimal stand-in for ``langchain_core.documents.Document``.

    Returned only when langchain_core is not installed. It mirrors the
    two attributes every LangChain consumer reads — ``page_content``
    (str) and ``metadata`` (dict) — plus the framework's ``type``
    discriminator and optional ``id``. It is a stand-in, documented as
    such; with langchain_core present the real ``Document`` is used
    instead and this class never ships a doc.
    """

    page_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    type: str = "Document"
    id: Optional[str] = None


#: The document class this adapter returns: the framework's when
#: importable, else the stand-in. Exposed so callers can introspect.
Document = _FrameworkDocument or _Doc

#: Whether the real langchain_core Document class was importable.
LANGCHAIN_CORE = _FrameworkDocument is not None


class VerbatimRetriever:
    """LangChain-compatible retriever over ``verbatim.Memory``.

    ``memory``: a live facade object exposing ``search(query, *,
    limit=..., **kwargs) -> SearchResult`` — duck-typed like everything
    else here, so tests may bind a stub and deployments bind ``Memory``.
    ``k``: documents returned per query → ``Memory.search(limit=k)``.
    ``search_kwargs``: extra ``Memory.search`` kwargs merged into every
    call; unknown keys fail typed at construction (never silently
    dropped); ``limit`` is reserved to ``k``.
    """

    #: Keyword vocabulary of ``Memory.search`` minus ``limit`` (owned by
    #: ``k``). Anything outside this set in ``search_kwargs`` or per-call
    #: kwargs is a typed rejection — a misspelled option must never be a
    #: silent no-op.
    SEARCH_KWARGS = frozenset(
        {
            "filters",
            "after",
            "consistency",
            "ready_timeout_ms",
            "timeout_ms",
            "strict",
        }
    )

    #: LangChain call-scaffolding accepted and ignored — these travel on
    #: every framework invocation and are not search options.
    PLUMBING_KWARGS = frozenset(
        {
            "callbacks",
            "tags",
            "metadata",
            "run_name",
            "run_manager",
            "run_id",
            "parent_run_id",
            "config",
            "recursion_limit",
            "max_concurrency",
            "kwargs",
        }
    )

    #: SearchResult statuses that must surface as an honest empty — the
    #: causal barrier is unmet, the subsystem is down, or support was
    #: examined and found wanting. Delivering hits here would fake a
    #: successful answer (V6-01.03).
    EMPTY_STATUSES = frozenset(
        {"pending", "blocked", "unavailable", "insufficient"}
    )

    def __init__(
        self,
        memory: Any,
        *,
        k: int = 8,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not callable(getattr(memory, "search", None)):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "memory must expose a callable search() — bind a live "
                "verbatim.Memory (or a stub honoring its contract)",
            )
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise VerbatimError(
                ErrorCode.VALIDATION, "k must be a positive int"
            )
        self._memory = memory
        self._k = k
        self._search_kwargs = self._validate_kwargs(
            search_kwargs, where="search_kwargs"
        )
        #: Honest status of the most recent search — see module docstring.
        self._verbatim_status: Optional[str] = None
        #: The raw SearchResult of the most recent search (warnings,
        #: coverage, causal_token), for callers that need the detail.
        self._verbatim_result: Optional[Any] = None

    # ------------------------------------------------------------------
    # LangChain retriever protocol (legacy)
    # ------------------------------------------------------------------

    def get_relevant_documents(self, query: str, **kw: Any) -> List[Any]:
        """Sync retrieval — the ``BaseRetriever.get_relevant_documents``
        shape LangChain chains and agents call."""
        kwargs = self._search_kwargs_for(kw)
        result = self._memory.search(query, limit=self._k, **kwargs)
        self._verbatim_result = result
        self._verbatim_status = str(getattr(result, "status", "") or "")
        if self._verbatim_status in self.EMPTY_STATUSES:
            return []
        return [self._hit_to_doc(hit) for hit in (result.items or ())]

    async def aget_relevant_documents(self, query: str, **kw: Any) -> List[Any]:
        """Async retrieval — same shape LangChain's
        ``aget_relevant_documents`` expects. ``Memory.search`` is blocking
        store I/O, so (like the framework's own default) the sync path
        runs on the loop's executor rather than faking async."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(self.get_relevant_documents, query, **kw),
        )

    # ------------------------------------------------------------------
    # LCEL Runnable protocol
    # ------------------------------------------------------------------

    def invoke(
        self, input: Any, config: Optional[Any] = None, **kw: Any
    ) -> List[Any]:
        """LCEL ``Runnable.invoke``: ``input`` is the query string;
        ``config`` is RunnableConfig scaffolding — accepted, ignored."""
        if not isinstance(input, str):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "retriever input must be a query string",
            )
        return self.get_relevant_documents(input, **kw)

    async def ainvoke(
        self, input: Any, config: Optional[Any] = None, **kw: Any
    ) -> List[Any]:
        """LCEL ``Runnable.ainvoke`` — async twin of :meth:`invoke`."""
        if not isinstance(input, str):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "retriever input must be a query string",
            )
        return await self.aget_relevant_documents(input, **kw)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @classmethod
    def _validate_kwargs(
        cls, value: Optional[Dict[str, Any]], *, where: str
    ) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise VerbatimError(
                ErrorCode.VALIDATION, f"{where} must be a mapping"
            )
        unknown = sorted(set(value) - cls.SEARCH_KWARGS)
        if "limit" in unknown:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "limit is set by k — remove it from " + where,
            )
        if unknown:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"unknown Memory.search kwargs {unknown} — allowed: "
                f"{sorted(cls.SEARCH_KWARGS)}",
            )
        return dict(value)

    def _search_kwargs_for(self, call_kw: Dict[str, Any]) -> Dict[str, Any]:
        """Per-call kwargs over constructor ``search_kwargs``: plumbing
        stripped, the rest validated against the declared vocabulary."""
        extra = {
            k: v for k, v in call_kw.items() if k not in self.PLUMBING_KWARGS
        }
        unknown = sorted(set(extra) - self.SEARCH_KWARGS)
        if "limit" in unknown:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "limit is set by k — do not pass it per call",
            )
        if unknown:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"unknown Memory.search kwargs {unknown} — allowed: "
                f"{sorted(self.SEARCH_KWARGS)}",
            )
        merged = dict(self._search_kwargs)
        merged.update(extra)
        return merged

    def _hit_to_doc(self, hit: Any) -> Any:
        source_id = str(getattr(hit, "memory_id", "") or "")
        if not source_id:
            source_id = self._source_id_from_ref(
                str(getattr(hit, "ref", "") or "")
            )
        metadata = {
            "source_id": source_id,
            "ref": str(getattr(hit, "ref", "") or ""),
            "object_ref": str(getattr(hit, "object_ref", "") or ""),
            "kind": str(getattr(hit, "kind", "") or ""),
            "score": float(getattr(hit, "score", 0.0) or 0.0),
            "score_family": str(getattr(hit, "score_family", "") or ""),
            "support_status": str(getattr(hit, "support_status", "") or ""),
            "lifecycle": str(getattr(hit, "lifecycle", "") or ""),
            "role": str(getattr(hit, "role", "") or ""),
            "type": str(getattr(hit, "type", "") or ""),
            "corroboration": int(getattr(hit, "corroboration", 0) or 0),
            "collapsed_duplicates": int(
                getattr(hit, "collapsed_duplicates", 0) or 0
            ),
            "valid_time": getattr(hit, "valid_time", None),
            "recorded_time": getattr(hit, "recorded_time", None),
            "warnings": list(getattr(hit, "warnings", None) or ()),
        }
        object_ref = metadata["object_ref"]
        return Document(
            page_content=str(getattr(hit, "quote", "") or ""),
            metadata=metadata,
            id=object_ref or None,
        )

    @staticmethod
    def _source_id_from_ref(ref: str) -> str:
        if not ref:
            return ""
        try:
            return MemoryRef.parse(ref).source_id
        except Exception:
            return ""


__all__ = [
    "Document",
    "LANGCHAIN_CORE",
    "VerbatimRetriever",
    "_Doc",
]
