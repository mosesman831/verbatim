"""Durable tests for the AMB adapter — SPEC_V8 §15.2 (V8-15.06–15.13).

Pins the honest-execution contract:

* the provider constructs and ingests on a synthetic store offline —
  AMB ``Document``-shaped inputs map to ``Memory.add`` (spy capture),
  and a real ``verbatim.Memory`` end-to-end run works with no network;
* ``retrieve`` passes ``as_of=<query_timestamp>`` + ``timeout_ms=500``
  to ``Memory.search`` when the facade accepts them (spy assert), and
  records ``applied=false`` honestly when it does not;
* token counting records per-query ``est_context_tokens`` and the
  ``token_budget`` bounds the delivered session excerpts (V85-02.04 —
  the metered context never exceeds B);
* the runner validates authorizations FIRST — an absent/decision-free
  ``authorizations.json`` emits ``blocked_on_authorization`` rows
  naming the missing O ids and exits nonzero with the report written;
* with decisions granted but the default backend, the model call still
  blocks honestly (``AuthorizationBlocked`` — never a silent stub);
* with decisions granted + injected test backends, the full
  ingest→search→judge loop executes and reports real numbers.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from eval.amb import provider as P
from eval.amb import runner as R
from eval.amb.manifest import (
    PENDING,
    build_manifest,
    verify_manifest,
)


# ---------------------------------------------------------------------------
# spies + fakes
# ---------------------------------------------------------------------------


class SpyHit:
    """Minimal governed-hit shape (``SearchResult.items`` elements)."""

    def __init__(self, i: int, quote: str, kind: str = "source") -> None:
        self.memory_id = f"src-{i}"
        self.ref = ""
        # unit refs resolve through the units table when present; a
        # source kind keeps the spy store-free.
        self.object_ref = f"vobj1.{kind}.{self.memory_id.encode().hex()}.1"
        self.kind = kind
        self.quote = quote
        self.lifecycle = "active"
        self.support_status = "supported"
        self.recorded_time = "2024-01-01T00:00:00+00:00"
        self.score = 1.0 - i * 0.01
        self.warnings = []


class SpyResult:
    def __init__(self, items) -> None:
        self.status = "ready"
        self.items = list(items)
        self.warnings = []
        self.readiness = {}
        self.coverage = {"support": {"verdict": "supported"}}
        self.causal_token = ""


class SpyMemory:
    """``Memory``-shaped spy — ``**kwargs`` on search means the
    provider's as_of feature-detect must pass the time pin through."""

    def __init__(self) -> None:
        self.add_calls = []
        self.search_calls = []
        self._hits = [
            SpyHit(0, "Caroline adopted a rescue dog named Biscuit."),
            SpyHit(1, "The pottery studio opens at 9am on Saturdays."),
        ]

    def add(self, content, **kw):
        self.add_calls.append((content, kw))
        n = len(self.add_calls) - 1
        return SimpleNamespace(
            memory_id=f"src-{n}", ref="", receipt_id=f"r{n}",
            acceptance="accepted",
        )

    def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return SpyResult(self._hits)

    def wait_ready(self, receipt, timeout_ms=None):
        return SimpleNamespace(state="ready")

    def close(self):
        pass


class StrictSpyMemory(SpyMemory):
    """A facade without ``as_of`` — explicit params, no **kwargs."""

    def search(self, query, *, limit=8, timeout_ms=None):
        self.search_calls.append(
            (query, {"limit": limit, "timeout_ms": timeout_ms})
        )
        return SpyResult(self._hits)


def _spy_factory(spies):
    def make(path, unit_key):
        mem = SpyMemory()
        spies[unit_key] = mem
        return mem
    return make


def _docs(n=2):
    return [
        {
            "id": f"d{i}",
            "content": f"Document {i} about Biscuit the dog and pottery.",
            "user_id": None,
            "timestamp": "2024-01-0%dT10:00:00Z" % (i + 1),
            "messages": [{"speaker": "Caroline", "text": f"turn {i}"}],
            "tags": ["t%d" % i],
            "context": "bench",
            "session_id": "s1",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# provider — construction + ingest + retrieve contract
# ---------------------------------------------------------------------------


def test_provider_ingest_maps_amb_documents(tmp_path):
    spies = {}
    prov = P.VerbatimAMBProvider(
        store_dir=tmp_path, memory_factory=_spy_factory(spies))
    prov.prepare(tmp_path)
    rep = prov.ingest(_docs(3))
    assert rep["documents"] == 3
    assert rep["indexed"] == 3
    mem = spies["_shared"]
    assert len(mem.add_calls) == 3
    content, kw = mem.add_calls[0]
    assert "Biscuit" in content
    # V85-02.02: speaker/session/timestamp preserved through add args;
    # session_id is the DOCUMENT id (the full conversation identifier
    # used by later retrieval), never the AMB session_id field.
    assert kw["occurred_at"] == "2024-01-01T10:00:00Z"
    assert kw["messages"] == [{"speaker": "Caroline", "text": "turn 0"}]
    assert kw["session_id"] == "d0"
    assert kw["metadata"]["amb_doc_id"] == "d0"
    prov.cleanup()


def test_retrieve_passes_as_of_and_timeout(tmp_path):
    spies = {}
    prov = P.VerbatimAMBProvider(
        store_dir=tmp_path, memory_factory=_spy_factory(spies),
        timeout_ms=500.0)
    prov.prepare(tmp_path)
    prov.ingest(_docs(1))
    docs, raw = prov.retrieve(
        "what dog did Caroline adopt", k=5,
        query_timestamp="2024-02-01T12:00:00Z")
    mem = spies["_shared"]
    (_q, kwargs), = mem.search_calls
    # V85-02.03: as_of=<dataset question time>, timeout_ms=500, and the
    # engine-cap limit 64 — AMB's k is recorded, never obeyed.
    assert kwargs["as_of"] == "2024-02-01T12:00:00Z"
    assert kwargs["timeout_ms"] == 500.0
    assert kwargs["limit"] == 64
    # V85-02.05: raw_response is ALWAYS None — LoCoMo would serialize a
    # non-None value into the reader prompt.
    assert raw is None
    rec = prov.query_records[-1]
    assert rec["as_of"]["applied"] is True
    assert rec["k"] == 5
    assert rec["status"] == "ready"
    # V85-02.05: session-excerpt header — [conv · session n · date];
    # the doc's ``context`` field supplies the conversation label.
    assert docs and docs[0].id == "d0"
    assert docs[0].source_ids == ["d0"]
    assert docs[0].content.startswith("[bench · 2024-01-01, Monday]")
    assert "Caroline: turn 0" in docs[0].content
    prov.cleanup()


def test_retrieve_records_as_of_unsupported(tmp_path):
    spies = {}

    def make(path, unit_key):
        mem = StrictSpyMemory()
        spies[unit_key] = mem
        return mem

    prov = P.VerbatimAMBProvider(
        store_dir=tmp_path, memory_factory=make)
    prov.prepare(tmp_path)
    prov.ingest(_docs(1))
    docs, raw = prov.retrieve(
        "query", k=3, query_timestamp="2024-02-01T00:00:00Z")
    assert raw is None
    rec = prov.query_records[-1]
    # honest degrade: the pin is recorded, never silently dropped.
    assert rec["as_of"]["requested"] == "2024-02-01T00:00:00Z"
    assert rec["as_of"]["applied"] is False
    (_q, kwargs), = spies["_shared"].search_calls
    assert "as_of" not in kwargs
    prov.cleanup()


def test_token_records_and_budget(tmp_path):
    spies = {}
    prov = P.VerbatimAMBProvider(
        store_dir=tmp_path, memory_factory=_spy_factory(spies),
        token_budget=40)
    prov.prepare(tmp_path)
    prov.ingest(_docs(1))
    docs, raw = prov.retrieve("dog", k=10)
    assert raw is None
    rec = prov.query_records[-1]
    assert isinstance(rec["est_context_tokens"], int)
    assert rec["token_budget"] == 40
    # V85-02.04: the metered delivered context never exceeds B.
    assert rec["est_context_tokens"] <= 40
    assert rec["expansion"]["turns_delivered"] >= 1
    curve = prov.token_curve()
    assert curve["token_budget"] == 40
    assert curve["est_context_tokens"] == [rec["est_context_tokens"]]
    prov.cleanup()


def test_items_mode_returns_doc_ids(tmp_path):
    spies = {}
    prov = P.VerbatimAMBProvider(
        store_dir=tmp_path, memory_factory=_spy_factory(spies),
        doc_mode="items")
    prov.prepare(tmp_path)
    prov.ingest(_docs(2))
    docs, raw = prov.retrieve("dog", k=10)
    assert [d.id for d in docs] == ["d0", "d1"]
    assert all(d.source_ids for d in docs)
    prov.cleanup()


def test_retrieve_before_ingest_raises(tmp_path):
    prov = P.VerbatimAMBProvider(
        store_dir=tmp_path,
        memory_factory=_spy_factory({}))
    prov.prepare(tmp_path)
    with pytest.raises(RuntimeError):
        prov.retrieve("anything")


def test_per_unit_banks(tmp_path):
    spies = {}
    prov = P.VerbatimAMBProvider(
        store_dir=tmp_path, memory_factory=_spy_factory(spies))
    prov.prepare(tmp_path, unit_ids={"u1", "u2"})
    prov.ingest([
        {"id": "a", "content": "user one memory", "user_id": "u1"},
        {"id": "b", "content": "user two memory", "user_id": "u2"},
    ])
    assert sorted(prov.banks()) == ["u1", "u2"]
    docs, _ = prov.retrieve("memory", user_id="u1")
    assert docs
    # an unknown unit gets an honest empty, not a crash
    docs2, raw2 = prov.retrieve("memory", user_id="nobody")
    assert docs2 == []
    assert raw2 is None
    assert prov.query_records[-1]["status"] == "no_bank"
    prov.cleanup()


def test_register_installs_provider():
    registry = {}
    out = P.register(registry)
    assert out["verbatim"] is P.VerbatimAMBProvider


def test_provider_real_memory_end_to_end(tmp_path):
    """Real ``verbatim.Memory`` offline — add → drain → search → render,
    no fixture shortcuts (the V8-15.06 write/read path itself)."""
    prov = P.VerbatimAMBProvider(store_dir=tmp_path / "store")
    prov.prepare(tmp_path / "store")
    rep = prov.ingest([
        {"id": "d0",
         "content": "Caroline adopted a rescue dog named Biscuit "
                    "last March.",
         "timestamp": "2024-01-05T09:00:00Z"},
        {"id": "d1",
         "content": "The pottery studio opens at 9am on Saturdays.",
         "timestamp": "2024-01-06T09:00:00Z"},
    ])
    assert rep["indexed"] == 2
    assert rep["add_errors"] == 0
    docs, raw = prov.retrieve(
        "what kind of dog did Caroline adopt", k=5,
        query_timestamp="2024-02-01T00:00:00Z")
    assert raw is None
    rec = prov.query_records[-1]
    assert isinstance(rec["est_context_tokens"], int)
    assert rec["pack_source"] == "session_expansion/v1"
    if docs:
        assert docs[0].content.startswith("[d0")
    prov.cleanup()


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def test_manifest_placeholders_and_digest():
    m = build_manifest(
        provider_version=P.PROVIDER_VERSION,
        arm={"name": "verbatim", "timeout_ms": 500.0,
             "token_budget": 4500, "doc_mode": "pack"},
        dataset={"id": "locomo10", "split": "test"},
        authorizations={"decisions": {}, "blocked_on": ["O5"]},
        environment={"python_version": "test"},
        vcs={"system": "git", "status": "ok", "sha": "deadbeef",
             "dirty": False},
    )
    assert m["schema"] == "run_manifest/amb-v8"
    assert m["amb"]["commit"] == PENDING
    assert m["models"]["reader"]["id"] == PENDING
    assert m["models"]["judge"]["authorization"] == "O5"
    assert m["provider"]["verbatim_commit"] == "deadbeef"
    assert verify_manifest(m)
    # same inputs → identical digest (determinism)
    m2 = build_manifest(
        provider_version=P.PROVIDER_VERSION,
        arm={"name": "verbatim", "timeout_ms": 500.0,
             "token_budget": 4500, "doc_mode": "pack"},
        dataset={"id": "locomo10", "split": "test"},
        authorizations={"decisions": {}, "blocked_on": ["O5"]},
        environment={"python_version": "test"},
        vcs={"system": "git", "status": "ok", "sha": "deadbeef",
             "dirty": False},
    )
    assert m2["digest"] == m["digest"]


# ---------------------------------------------------------------------------
# runner — authorization gate + loop
# ---------------------------------------------------------------------------


def _dataset_file(tmp_path, queries=None):
    data = {
        "dataset": "locomo10",
        "split": "test",
        "task_type": "open",
        "documents": _docs(2),
        "queries": queries or [
            {"id": "q1", "query": "what dog did Caroline adopt",
             "gold_answers": ["a rescue dog named Biscuit"],
             "meta": {"query_timestamp": "2024-02-01T00:00:00Z"}},
            {"id": "q2", "query": "when does the studio open",
             "gold_answers": ["9am Saturdays"],
             "meta": {"query_timestamp": "2024-02-02T00:00:00Z"}},
        ],
    }
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(data))
    return str(path)


def _auth_file(tmp_path, grants):
    path = tmp_path / "authorizations.json"
    path.write_text(json.dumps(grants))
    return str(path)


def test_runner_blocked_without_authorizations(tmp_path, capsys):
    ds = _dataset_file(tmp_path)
    auth = _auth_file(tmp_path, {"schema": "amb_authorizations/v1",
                                 "decisions": {}})
    out = tmp_path / "report.json"
    code = R.main([
        "--dataset", "locomo10", "--mode", "rag",
        "--dataset-file", ds, "--authorizations", auth,
        "--token-budget", "4500", "--workdir", str(tmp_path / "st"),
        "--out", str(out),
    ])
    assert code == R.EXIT_BLOCKED
    rep = json.loads(out.read_text())
    assert rep["rows"], "blocked rows must be emitted"
    for row in rep["rows"]:
        assert row["status"] == "blocked_on_authorization"
        assert "O1" in row["blocked_on"]
        assert "O5" in row["blocked_on"]
    # the whole V8-15.09 board is reported — comparator needs O6.
    board = {r["row"]: r for r in rep["plan"]}
    assert board["longmemeval_s.rag"]["blocked_on"] == [
        "O2", "O5", "amb_license", "amb_pin"]
    assert "O6" in board["hindsight.comparator"]["blocked_on"]
    assert rep["manifest"]["status"] == "pinned"


def test_runner_blocked_default_backend(tmp_path):
    """O records granted but the DEFAULT backend — the model call still
    raises the authorization error (never a silent stub)."""
    ds = _dataset_file(tmp_path)
    auth = _auth_file(tmp_path, {
        "decisions": {"O1": {"granted": True}, "O5": {"granted": True}},
        "amb_license": {"verified": True},
        "amb_pin": {"commit": "abc123"},
    })
    out = tmp_path / "report.json"
    spies = {}
    code = R.main([
        "--dataset", "locomo10", "--mode", "rag",
        "--dataset-file", ds, "--authorizations", auth,
        "--token-budget", "4500", "--workdir", str(tmp_path / "st"),
        "--out", str(out),
    ], provider_factory=lambda row, **kw: R.make_provider(
            row, memory_factory=_spy_factory(spies), **kw),
        prompts=R.StaticPromptSource(
            open_prompt="Ctx: {context}\nQ: {query}"),
    )
    assert code == R.EXIT_BLOCKED
    rep = json.loads(out.read_text())
    (row,) = rep["rows"]
    assert row["status"] == "blocked_on_authorization"
    assert row["blocked_on"] == ["O5"]
    # ingest still ran for real before the model call blocked
    assert spies["_shared"].add_calls


class _FakeBackend(R.ModelBackend):
    """Test-injected backend — the pluggable seam working, not a
    default.  Returns deterministic canned output."""

    def answer(self, prompt, *, query="", context=""):
        return {"answer": "a rescue dog named Biscuit",
                "reasoning": "test"}

    def judge(self, query, answer, gold_answers, *, context=""):
        return {"correct": True, "reason": "test judge"}


def test_runner_executes_with_injected_backends(tmp_path):
    ds = _dataset_file(tmp_path)
    auth = _auth_file(tmp_path, {
        "decisions": {"O1": {"granted": True}, "O5": {"granted": True}},
        "amb_license": {"verified": True},
        "amb_pin": {"commit": "abc123"},
    })
    out = tmp_path / "report.json"
    spies = {}
    fake = _FakeBackend()
    code = R.main([
        "--dataset", "locomo10", "--mode", "rag",
        "--dataset-file", ds, "--authorizations", auth,
        "--token-budget", "4500", "--workdir", str(tmp_path / "st"),
        "--out", str(out), "--query-limit", "2",
    ], provider_factory=lambda row, **kw: R.make_provider(
            row, memory_factory=_spy_factory(spies), **kw),
        reader=fake, judge=fake,
        prompts=R.StaticPromptSource(
            open_prompt="Ctx: {context}\nQ: {query}"),
    )
    assert code == R.EXIT_OK
    rep = json.loads(out.read_text())
    (row,) = rep["rows"]
    assert row["status"] == "executed"
    s = row["summary"]
    assert s["n_answered"] == 2
    assert s["n_correct"] == 2
    assert s["accuracy"] == 1.0
    assert s["timeout_ms"] == 500.0
    assert s["est_context_tokens"]["p50"] is not None
    assert s["token_curve"]["token_budget"] == 4500
    assert rep["manifest"]["status"] == "pinned"
    # as_of passed to the spy on every query
    calls = spies["_shared"].search_calls
    assert all(c[1].get("as_of") for c in calls)


def test_runner_retrieval_mode_no_models(tmp_path):
    """PrecisionMemBench-style row — id-set scoring needs no model at
    all (V8-15.09 step 1), only license + pin."""
    data = {
        "dataset": "precisionmembench", "task_type": "retrieval",
        "documents": _docs(2),
        "queries": [
            {"id": "q1", "query": "dog",
             "assert_include": ["d0"], "assert_exclude": ["d9"]},
        ],
    }
    ds = tmp_path / "ds.json"
    ds.write_text(json.dumps(data))
    auth = _auth_file(tmp_path, {
        "decisions": {},
        "amb_license": {"verified": True},
        "amb_pin": {"commit": "abc123"},
    })
    out = tmp_path / "report.json"
    spies = {}
    code = R.main([
        "--dataset", "precisionmembench", "--mode", "retrieval",
        "--dataset-file", str(ds), "--authorizations", auth,
        "--token-budget", "4500", "--workdir", str(tmp_path / "st"),
        "--out", str(out),
    ], provider_factory=lambda row, **kw: R.make_provider(
            row, memory_factory=_spy_factory(spies), **kw))
    assert code == R.EXIT_OK
    rep = json.loads(out.read_text())
    (row,) = rep["rows"]
    assert row["status"] == "executed"
    assert row["summary"]["n_correct"] == 1
    assert row["queries"][0]["correct"] is True


def test_runner_missing_amb_pin_blocks_even_granted(tmp_path):
    """Granted O5 but no amb_pin → rag rows cannot form the unmodified
    prompt: still blocked, honestly."""
    ds = _dataset_file(tmp_path)
    auth = _auth_file(tmp_path, {
        "decisions": {"O1": {"granted": True}, "O5": {"granted": True}},
        "amb_license": {"verified": True},
        # no amb_pin
    })
    out = tmp_path / "report.json"
    code = R.main([
        "--dataset", "locomo10", "--mode", "rag",
        "--dataset-file", ds, "--authorizations", auth,
        "--token-budget", "4500", "--workdir", str(tmp_path / "st"),
        "--out", str(out),
    ])
    assert code == R.EXIT_BLOCKED
    rep = json.loads(out.read_text())
    (row,) = rep["rows"]
    assert row["status"] == "blocked_on_authorization"
    assert "amb_pin" in row["blocked_on"]
