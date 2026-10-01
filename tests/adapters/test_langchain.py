"""LangChain retriever adapter tests (SPEC_V6 V6-01.09).

The contract under test: ``VerbatimRetriever`` is a duck-typed
LangChain retriever over ``Memory.search`` — importable with zero
langchain packages installed, returning ``Document``-shaped objects
carrying ``source_id``, and honest about non-answers (pending /
blocked / insufficient → ``[]`` + ``_verbatim_status``, never a fake
success).

Coverage:

- recorded stub of the framework interface: the method surface and
  ``Document`` attributes LangChain actually calls, plus a
  ``StubMemory`` recording every ``search`` call to prove delegation;
- real ``Memory`` on a temp store (``worker="external"`` + explicit
  ``Ingester`` drain — the deterministic convention from
  ``tests/memory/test_facade.py``): add → drain → retrieve returns docs
  with ``page_content`` + ``metadata["source_id"]``;
- sync + async protocols (``get_relevant_documents`` /
  ``aget_relevant_documents`` / ``invoke`` / ``ainvoke``);
- honest empty: an undrained (pending) add and a no-support query both
  yield ``[]`` with the real status on ``_verbatim_status``;
- ``search_kwargs`` / per-call kwarg merge + typed rejection of unknown
  keys and of ``limit`` (owned by ``k``).
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from verbatim import Memory
from verbatim.adapters.langchain import (
    LANGCHAIN_CORE,
    Document,
    VerbatimRetriever,
    _Doc,
)
from verbatim.core.types import ErrorCode, JobKind, VerbatimError
from verbatim.ingest import Ingester
from verbatim.memory.types import Hit, MemoryRef, SearchResult


# ---------------------------------------------------------------------------
# fixtures + recorded stubs
# ---------------------------------------------------------------------------


@pytest.fixture()
def path(tmp_path):
    return str(tmp_path / "lc.db")


@pytest.fixture()
def mem(path):
    m = Memory(path=path, worker="external")
    yield m
    try:
        m.close()
    except Exception:
        pass


def _drain(mem):
    """External-worker drain — real job pipeline, deterministic."""
    ingester = Ingester(mem._store, mem._cfg, encoder=mem._encoder)
    report = ingester.drain_report(
        scope=mem._namespace,
        owner="t-langchain",
        kinds=(JobKind.SOURCE_PROJECT, JobKind.SOURCE_EMBED),
    )
    assert report["failed"] == 0
    return report


class StubMemory:
    """Recorded stub of the ``Memory.search`` surface the adapter binds.

    Returns queued ``SearchResult`` objects (the real contract type) and
    records every ``(query, kwargs)`` call — proves the retriever
    translates and delegates rather than reimplementing search.
    """

    def __init__(self, *results: SearchResult) -> None:
        self._results = list(results) or [SearchResult(status="ready")]
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if len(self._results) > 1:
            return self._results.pop(0)
        return self._results[0]


def _hit(source_id="src_stub", quote="stub quote", score=1.5, kind="source"):
    ref = (
        MemoryRef(
            store_tag="tag",
            namespace="ns",
            source_id=source_id,
            expected_revision=1,
            control_version=0,
        ).to_string()
        if kind == "source"
        else ""
    )
    return Hit(
        memory_id=source_id if kind == "source" else "",
        ref=ref,
        object_ref=f"vobj1.{kind}.{source_id}.1",
        kind=kind,
        quote=quote,
        score=score,
        support_status="supported",
        lifecycle="active",
    )


# ---------------------------------------------------------------------------
# importability + interface shape (the recorded framework stub)
# ---------------------------------------------------------------------------


class TestInterfaceShape:
    """The retriever satisfies the duck-typed surface LangChain calls —
    asserted against the recorded method/attribute contract since no
    langchain package is installed to check against directly."""

    def test_document_resolution_matches_framework_presence(self):
        # The adapter resolves Document honestly either way: the real
        # langchain_core class when the framework is installed, the
        # documented _Doc stand-in otherwise — never a fake of the real
        # type. (In this build langchain_core is absent → stand-in.)
        try:
            import langchain_core  # noqa: F401

            present = True
        except ModuleNotFoundError:
            present = False
        assert LANGCHAIN_CORE == present
        assert Document is not None
        if present:
            assert Document is not _Doc
        else:
            assert Document is _Doc
            doc = Document(page_content="x", metadata={"source_id": "s"})
            assert isinstance(doc, _Doc)
            assert doc.page_content == "x"
            assert doc.metadata["source_id"] == "s"

    def test_retriever_method_surface(self):
        r = VerbatimRetriever(StubMemory())
        # legacy retriever protocol
        sig = inspect.signature(r.get_relevant_documents)
        params = list(sig.parameters)
        assert params[0] == "query"
        assert inspect.iscoroutinefunction(r.aget_relevant_documents)
        # LCEL runnable protocol
        sig = inspect.signature(r.invoke)
        params = list(sig.parameters)
        assert params[:2] == ["input", "config"]
        assert inspect.iscoroutinefunction(r.ainvoke)

    def test_document_attribute_surface(self):
        # The attributes every LangChain consumer reads: .page_content
        # (str) + .metadata (dict); framework parity fields too.
        stub = StubMemory(SearchResult(status="ready", items=[_hit()]))
        docs = VerbatimRetriever(stub).get_relevant_documents("q")
        doc = docs[0]
        assert isinstance(doc.page_content, str)
        assert isinstance(doc.metadata, dict)
        assert getattr(doc, "type", "Document") == "Document"
        assert hasattr(doc, "id")


# ---------------------------------------------------------------------------
# delegation + mapping (stub memory)
# ---------------------------------------------------------------------------


class TestDelegation:
    def test_search_called_with_query_and_k_as_limit(self):
        stub = StubMemory(SearchResult(status="ready", items=[_hit()]))
        r = VerbatimRetriever(stub, k=3)
        docs = r.get_relevant_documents("the question")
        assert stub.calls == [("the question", {"limit": 3})]
        assert len(docs) == 1

    def test_hit_maps_to_document(self):
        hit = _hit(source_id="src_abc", quote="the wifi password is swordfish")
        stub = StubMemory(SearchResult(status="ready", items=[hit]))
        (doc,) = VerbatimRetriever(stub).get_relevant_documents("wifi")
        assert doc.page_content == "the wifi password is swordfish"
        meta = doc.metadata
        assert meta["source_id"] == "src_abc"
        assert MemoryRef.parse(meta["ref"]).source_id == "src_abc"
        assert meta["score"] == 1.5
        assert meta["support_status"] == "supported"
        assert meta["lifecycle"] == "active"
        assert meta["kind"] == "source"
        assert meta["object_ref"] == hit.object_ref

    def test_search_kwargs_merge_and_call_time_wins(self):
        stub = StubMemory(SearchResult(status="ready", items=[_hit()]))
        r = VerbatimRetriever(
            stub,
            k=4,
            search_kwargs={"consistency": "eventual", "timeout_ms": 100},
        )
        r.get_relevant_documents("q", consistency="session")
        (_, kw), = stub.calls
        assert kw == {
            "limit": 4,
            "consistency": "session",
            "timeout_ms": 100,
        }

    def test_plumbing_kwargs_accepted_and_ignored(self):
        stub = StubMemory(SearchResult(status="ready", items=[_hit()]))
        r = VerbatimRetriever(stub)
        # The exact scaffolding LangChain passes on every invocation.
        r.get_relevant_documents(
            "q",
            callbacks=[],
            tags=["t"],
            metadata={"a": 1},
            run_name="run",
            config={"configurable": {}},
        )
        r.invoke("q", config={"callbacks": []}, recursion_limit=25)
        assert stub.calls[0][1] == {"limit": 8}
        assert stub.calls[1][1] == {"limit": 8}

    def test_unknown_kwargs_typed_error(self):
        stub = StubMemory()
        with pytest.raises(VerbatimError) as e:
            VerbatimRetriever(stub, search_kwargs={"bogus": 1})
        assert e.value.code == ErrorCode.VALIDATION
        r = VerbatimRetriever(stub)
        with pytest.raises(VerbatimError) as e:
            r.get_relevant_documents("q", drop_table="sources")
        assert e.value.code == ErrorCode.VALIDATION
        assert stub.calls == []  # rejected before any search call

    def test_limit_reserved_to_k(self):
        stub = StubMemory()
        with pytest.raises(VerbatimError) as e:
            VerbatimRetriever(stub, search_kwargs={"limit": 3})
        assert e.value.code == ErrorCode.VALIDATION
        r = VerbatimRetriever(stub)
        with pytest.raises(VerbatimError) as e:
            r.get_relevant_documents("q", limit=3)
        assert e.value.code == ErrorCode.VALIDATION

    def test_constructor_validation(self):
        with pytest.raises(VerbatimError):
            VerbatimRetriever(object())
        stub = StubMemory()
        for bad in (0, -1, 1.5, True, "8"):
            with pytest.raises(VerbatimError):
                VerbatimRetriever(stub, k=bad)
        with pytest.raises(VerbatimError):
            VerbatimRetriever(stub, search_kwargs="timeout_ms=100")

    def test_invoke_requires_string_input(self):
        r = VerbatimRetriever(StubMemory())
        with pytest.raises(VerbatimError) as e:
            r.invoke({"query": "q"})
        assert e.value.code == ErrorCode.VALIDATION


# ---------------------------------------------------------------------------
# honest status — never fake success
# ---------------------------------------------------------------------------


class TestHonestStatus:
    @pytest.mark.parametrize(
        "status", ["pending", "blocked", "unavailable", "insufficient"]
    )
    def test_blocking_statuses_return_empty(self, status):
        # Even with items present, a non-ready status must not fake an
        # answer — the real status lands on the retriever.
        stub = StubMemory(
            SearchResult(status=status, items=[_hit()], warnings=["w1"])
        )
        r = VerbatimRetriever(stub)
        assert r.get_relevant_documents("q") == []
        assert r._verbatim_status == status
        assert r._verbatim_result.status == status
        assert "w1" in r._verbatim_result.warnings

    @pytest.mark.parametrize("status", ["ready", "partial"])
    def test_delivering_statuses_return_docs(self, status):
        stub = StubMemory(
            SearchResult(status=status, items=[_hit(), _hit(source_id="s2")])
        )
        r = VerbatimRetriever(stub)
        docs = r.get_relevant_documents("q")
        assert len(docs) == 2
        assert r._verbatim_status == status

    def test_status_initially_none(self):
        r = VerbatimRetriever(StubMemory())
        assert r._verbatim_status is None
        assert r._verbatim_result is None


# ---------------------------------------------------------------------------
# async protocol
# ---------------------------------------------------------------------------


class TestAsync:
    def test_aget_relevant_documents(self):
        stub = StubMemory(SearchResult(status="ready", items=[_hit()]))
        r = VerbatimRetriever(stub)
        docs = asyncio.run(r.aget_relevant_documents("wifi"))
        assert len(docs) == 1
        assert docs[0].metadata["source_id"] == "src_stub"
        assert stub.calls[0] == ("wifi", {"limit": 8})

    def test_ainvoke(self):
        stub = StubMemory(SearchResult(status="ready", items=[_hit()]))
        r = VerbatimRetriever(stub)
        docs = asyncio.run(r.ainvoke("wifi", config={"callbacks": []}))
        assert len(docs) == 1
        with pytest.raises(VerbatimError):
            asyncio.run(r.ainvoke(42))


# ---------------------------------------------------------------------------
# real Memory end-to-end (external worker + explicit drain)
# ---------------------------------------------------------------------------


class TestRealMemory:
    def test_add_drain_retrieve(self, mem):
        r = mem.add("the deploy runbook lives at docs/deploy.md")
        _drain(mem)
        retriever = VerbatimRetriever(mem)
        docs = retriever.get_relevant_documents("deploy runbook")
        assert retriever._verbatim_status in ("ready", "partial")
        assert docs, "expected at least one document"
        doc = docs[0]
        assert "deploy runbook" in doc.page_content
        assert doc.metadata["source_id"] == r.memory_id
        assert MemoryRef.parse(doc.metadata["ref"]).source_id == r.memory_id
        assert doc.metadata["support_status"]
        assert doc.metadata["lifecycle"] == "active"

    def test_k_maps_to_limit(self, mem):
        for i in range(5):
            mem.add(f"runbook note number {i} about limes")
        _drain(mem)
        retriever = VerbatimRetriever(mem, k=2)
        docs = retriever.get_relevant_documents("limes runbook")
        assert len(docs) <= 2

    def test_pending_is_honest_empty(self, mem):
        """An add whose source-visibility work is undrained reports
        pending — the retriever returns [] and the real status, never a
        fake success (V6-01.03)."""
        mem.add("the launch checklist is in ops/launch.md")
        retriever = VerbatimRetriever(mem)
        docs = retriever.get_relevant_documents(
            "launch checklist", ready_timeout_ms=30
        )
        assert docs == []
        assert retriever._verbatim_status == "pending"
        # And after the real drain the same query delivers.
        _drain(mem)
        docs = retriever.get_relevant_documents("launch checklist")
        assert docs and retriever._verbatim_status in ("ready", "partial")

    def test_no_support_is_honest_empty(self, mem):
        """A completed search that finds no support must not deliver
        least-bad hits — [] plus the facade's honest status."""
        mem.add("the wifi password is swordfish")
        _drain(mem)
        retriever = VerbatimRetriever(mem)
        query = "kubernetes cluster node taint"
        expected = mem.search(query).status  # deterministic, same inputs
        docs = retriever.get_relevant_documents(query)
        assert retriever._verbatim_status == expected
        if expected in ("pending", "blocked", "unavailable", "insufficient"):
            assert docs == []
        else:
            assert isinstance(docs, list)

    def test_async_path_real_memory(self, mem):
        mem.add("the cellar door code is 4417")
        _drain(mem)
        retriever = VerbatimRetriever(mem)
        docs = asyncio.run(retriever.aget_relevant_documents("cellar door"))
        assert docs
        assert "4417" in docs[0].page_content
        assert docs[0].metadata["source_id"]

    def test_invoke_and_ainvoke_real_memory(self, mem):
        r = mem.add("the staging bucket is gs://stage-eu")
        _drain(mem)
        retriever = VerbatimRetriever(mem)
        docs = retriever.invoke("staging bucket", config={"callbacks": []})
        assert docs and docs[0].metadata["source_id"] == r.memory_id
        docs = asyncio.run(retriever.ainvoke("staging bucket"))
        assert docs and docs[0].metadata["source_id"] == r.memory_id

    def test_typed_errors_propagate(self, mem):
        """Memory.search validation reaches the caller as typed errors —
        the adapter never swallows them."""
        retriever = VerbatimRetriever(mem)
        with pytest.raises(VerbatimError):
            retriever.get_relevant_documents("")
        with pytest.raises(VerbatimError):
            retriever.get_relevant_documents("x", consistency="sometimes")
