"""SPEC_V5 §06 frozen-contract tests — the pieces that are real today.

``verbatim/memory/types.py`` is landed and frozen (docs/v5_contracts.md
§1), so the record/ref/enum contracts are asserted for real here —
these tests are not xfail and must stay green from day one.

E01's import-hygiene half is also real today: ``import verbatim`` must
not create files, threads, sockets, stores, or downloads. The lazy
``from verbatim import Memory`` export is asserted in a subprocess and
xfailed until the facade worker lands it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from verbatim.memory.types import (
    CONTRACT,
    CAP_SOURCE_LEXICAL,
    CAP_SOURCE_VECTOR,
    ENRICHMENT_VERSION,
    QUERY_ANALYSIS_VERSION,
    RANKING_VERSION,
    SOURCE_STATE_KIND,
    Acceptance,
    AddResult,
    ChangeKind,
    CloseReport,
    Consistency,
    ForgetResult,
    Hit,
    Inspection,
    Lifecycle,
    MatchClass,
    MemoryRef,
    MemoryStatus,
    MemoryType,
    Polarity,
    QueryClass,
    Readiness,
    SearchResult,
    SearchStatus,
    SupportStatus,
    TimePrecision,
    TimeStatus,
    UpdateCandidate,
    WorkerMode,
)


REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


# =====================================================================
# E01 — import creates no files/threads/sockets/stores/downloads
# (§03, §06); lazy `Memory` export
# =====================================================================


def test_e01_import_side_effect_free(tmp_path):
    """E01 (half) / V5-06.* preamble: ``import verbatim`` in a clean
    working directory creates no files, threads, sockets, stores, or
    downloads — verified in a fresh subprocess so ambient test state
    cannot mask a regression.
    """
    script = r"""
import json, os, socket, sys, threading

before_files = sorted(os.listdir("."))
before_threads = sorted(t.name for t in threading.enumerate())

class _SocketGuard:
    def __init__(self, *a, **k):
        raise AssertionError("socket opened during 'import verbatim'")

_real_socket = socket.socket
socket.socket = _SocketGuard
import verbatim  # noqa: F401
import verbatim.memory.types  # noqa: F401
socket.socket = _real_socket

after_files = sorted(os.listdir("."))
after_threads = sorted(t.name for t in threading.enumerate())

assert before_files == after_files, (
    f"import created files: {sorted(set(after_files) - set(before_files))}"
)
assert before_threads == after_threads, (
    f"import spawned threads: {sorted(set(after_threads) - set(before_threads))}"
)
assert "verbatim.memory.facade" not in sys.modules, (
    "import verbatim must not eagerly load the facade module"
)
print(json.dumps({"ok": True, "version": verbatim.__version__}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(tmp_path), env=env, capture_output=True, text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"import verbatim violated the no-side-effect contract:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert payload["version"]  # package version is exposed


def test_e01_memory_lazy_export(tmp_path):
    """E01 (half) / contracts §10: ``from verbatim import Memory`` works
    and resolves lazily — the facade module is absent from sys.modules
    until the attribute is actually touched. Runs in a subprocess so the
    assertion measures a real cold import.
    """
    script = r"""
import sys
import verbatim
assert "verbatim.memory.facade" not in sys.modules, (
    "facade must load lazily, not at 'import verbatim'"
)
from verbatim import Memory
assert callable(Memory)
assert "verbatim.memory.facade" in sys.modules, (
    "accessing Memory must resolve the facade module lazily"
)
print("ok")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(tmp_path), env=env, capture_output=True, text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"lazy Memory export missing or eager:\n{proc.stderr}"
    )
    assert "ok" in proc.stdout


# =====================================================================
# MemoryRef — version-bound, opaque, path-free identity (§06.11)
# =====================================================================


def test_contract_memoryref_roundtrip():
    """V5-06.11 / contracts §1: MemoryRef serializes to an opaque
    ``mref1.…`` token that round-trips exactly and carries store tag,
    namespace, source id, expected revision, and control version."""
    ref = MemoryRef(
        store_tag="st-1", namespace="ns", source_id="src-9",
        expected_revision=3, control_version=7,
    )
    text = ref.to_string()
    assert text.startswith("mref1.")
    parsed = MemoryRef.parse(text)
    assert parsed == ref
    assert parsed.expected_revision == 3
    assert parsed.control_version == 7

    d = ref.to_dict()
    assert d["ref"] == text
    assert d["namespace"] == "ns"
    assert d["source_id"] == "src-9"
    assert d["expected_revision"] == 3
    assert d["control_version"] == 7
    json.dumps(d)  # deterministic serialization (V5-06.01)


def test_contract_memoryref_escapes_dots_and_percents():
    """Namespace/source ids containing ``.``/``%`` must round-trip —
    the dotted wire form escapes them instead of corrupting identity."""
    ref = MemoryRef(
        store_tag="st.1", namespace="a.b%c", source_id="s.1%2",
        expected_revision=1, control_version=0,
    )
    text = ref.to_string()
    # The encoded segments must not contain raw separators that would
    # break the six-field parse.
    parts = text.split(".")
    assert len(parts) == 6
    assert MemoryRef.parse(text) == ref


def test_contract_memoryref_rejects_garbage_and_paths():
    """A malformed or hostile token fails parsing — refs never carry
    filesystem paths or key locations (V5-06.11)."""
    for bad in ("", "not-a-ref", "mref2.a.b.c.1.2", "mref1.a.b.c.x.2",
                "mref1.a.b.c.1", "/etc/passwd"):
        with pytest.raises((ValueError, Exception)):
            MemoryRef.parse(bad)


# =====================================================================
# Result records — typed, versioned, deterministic serialization
# (V5-06.01–06.04, §30.1 shapes)
# =====================================================================


def test_contract_results_serialize_deterministically():
    """Every public result record exposes ``to_dict()`` producing a
    JSON-serializable dict tagged with the frozen contract version —
    the Python and wire schemas must mean the same thing (V5-06.01)."""
    records = [
        AddResult(memory_id="s1", ref="mref1.a.b.c.1.0",
                  receipt_id="r1", acceptance="accepted"),
        SearchResult(status="ready", causal_token="ct"),
        Hit(memory_id="s1", ref="mref1.a.b.c.1.0", quote="q"),
        Inspection(ref="mref1.a.b.c.1.0", found=True),
        ForgetResult(mode="preview", mutated=False),
        Readiness(receipt_id="r1", state="pending"),
        MemoryStatus(profile="local_memory"),
        CloseReport(closed=True),
        UpdateCandidate(ref="mref1.a.b.c.1.0", relation="contradicts",
                        reason="r", score=0.5),
    ]
    for rec in records:
        d = rec.to_dict()
        assert isinstance(d, dict)
        assert d.get("schema") == CONTRACT
        blob = json.dumps(d, sort_keys=True)
        assert json.loads(blob) == d  # stable, total serialization


def test_contract_addresult_spec_fields():
    """V5-06.02/06.18: AddResult carries identity, revision, receipt,
    acceptance disposition, readiness, replay flag, inference state,
    and advisory possible_updates."""
    r = AddResult(memory_id="s1", receipt_id="rc1", source_revision=2)
    d = r.to_dict()
    for key in ("memory_id", "ref", "source_revision", "receipt_id",
                "acceptance", "replayed", "readiness", "warnings",
                "possible_updates", "inference"):
        assert key in d, key
    assert d["acceptance"] == Acceptance.ACCEPTED.value
    assert d["inference"] in (
        "not_requested", "queued", "deferred", "unavailable"
    )


def test_contract_hit_separate_dimensions():
    """V5-06.06: lifecycle, support assessment, and evidence role are
    independent fields — a hit never conflates 'superseded' with
    'unsupported'."""
    h = Hit(
        lifecycle=Lifecycle.SUPERSEDED.value,
        support_status=SupportStatus.DISPUTED.value,
        role="contrary",
    )
    d = h.to_dict()
    assert d["lifecycle"] == "superseded"
    assert d["support_status"] == "disputed"
    assert d["role"] == "contrary"
    # source hits carry identity + corroboration accounting (§30.2)
    for key in ("memory_id", "ref", "object_ref", "kind", "quote",
                "score", "score_family", "collapsed_duplicates",
                "corroboration", "score_detail"):
        assert key in d, key


def test_contract_searchresult_status_vocabulary():
    """V5-06.03/06.04: SearchResult distinguishes ready / partial /
    pending / blocked / unavailable and carries warnings, readiness,
    coverage, and the causal token — a bare list is not the API."""
    r = SearchResult(status=SearchStatus.PENDING.value)
    d = r.to_dict()
    for key in ("items", "status", "warnings", "readiness",
                "coverage", "causal_token"):
        assert key in d, key
    assert d["status"] == "pending"


def test_contract_forgetresult_preview_vs_operation():
    """V5-06.18: ForgetResult is a discriminated record — preview carries
    selection + confirmation token with ``mutated=false``; operation
    carries suppression/closure state and receipt."""
    preview = ForgetResult(mode="preview", mutated=False,
                           confirmation_token="ct1",
                           selection=["mref1.a.b.c.1.0"])
    d = preview.to_dict()
    assert d["mode"] == "preview" and d["mutated"] is False
    assert d["confirmation_token"] == "ct1"

    op = ForgetResult(mode="operation", mutated=True,
                      suppression_state="suppressed",
                      closure_state="pending", receipt_id="r9")
    d2 = op.to_dict()
    assert d2["mode"] == "operation" and d2["mutated"] is True
    assert d2["suppression_state"] == "suppressed"


# =====================================================================
# Enum vocabularies + version pins (frozen by docs/v5_contracts.md §1)
# =====================================================================


def test_contract_enum_vocabularies():
    """The enum vocabularies are the spec's words — acceptance, search
    status, change kind, consistency, worker mode, polarity, lifecycle,
    support, match class, memory type, query class."""
    # V5-06.18 core vocabulary is {accepted, held, protected}; V7's
    # bulk_add (V7-13.09/21.01) adds the per-item ``failed`` outcome —
    # the item's record rolled back, the typed reason rides ``error``.
    assert {a.value for a in Acceptance} == {
        "accepted", "held", "protected", "failed"}
    assert {s.value for s in SearchStatus} == {
        "ready", "partial", "pending", "blocked", "unavailable"}
    assert {c.value for c in ChangeKind} == {"supersede", "correct"}
    assert {c.value for c in Consistency} == {"session", "eventual"}
    assert {w.value for w in WorkerMode} == {"managed", "external"}
    assert {p.value for p in Polarity} == {
        "affirmative", "negated", "hedged", "hypothetical", "quoted"}
    assert {l.value for l in Lifecycle} >= {
        "active", "superseded", "corrected", "retracted"}
    assert {s.value for s in SupportStatus} == {
        "supported", "disputed", "insufficient", "unassessed"}
    assert {m.value for m in MatchClass} == {
        "exact", "normalized", "paraphrase"}
    assert {t.value for t in TimePrecision} == {
        "exact", "day", "month", "year", "relative", "unknown"}
    assert {t.value for t in TimeStatus} == {
        "ongoing", "completed", "planned", "unknown"}
    assert {t.value for t in MemoryType} >= {
        "fact", "preference", "decision", "plan", "state", "event",
        "relationship", "procedure_hint", "absence", "untyped"}
    # §31.1 deterministic query classes
    assert {q.value for q in QueryClass} == {
        "identifier", "entity", "temporal", "preference",
        "procedural", "factual", "no_answer_likely"}


def test_contract_version_pins():
    """The producer/contract versions are pinned strings — enrichment,
    query analysis, ranking, source-state, and the two new readiness
    capabilities (docs/v5_contracts.md §1/§4)."""
    assert QUERY_ANALYSIS_VERSION == "query_analysis/v1"
    assert RANKING_VERSION == "ranking/v1"
    assert ENRICHMENT_VERSION == "enrich/v1"
    assert SOURCE_STATE_KIND == "source_state/v1"
    assert CAP_SOURCE_LEXICAL == "source_lexical_ready"
    assert CAP_SOURCE_VECTOR == "source_vector_ready"
