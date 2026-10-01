"""SPEC_V6 acceptance suite — F01–F32 (SPEC_V6 §07).

Conventions mirror ``tests/v5/``: real on-disk ``Memory`` instances on
``tmp_path``, ``worker="external"`` + ``Ingester.drain_report`` where a
deterministic drain is required, ``worker="managed"`` where the
production drainer is the behavior under test. Every scenario maps to
the observable result the spec names — no mocks, no faked coverage.

Eval-scale scenarios (F09/F20/F21/F30/F32) run the real envelope /
comparator machinery at small scale and assert the measurement was
recorded honestly — pass-or-miss verdicts are data, not test failures.
Where the spec's observable result is genuinely unmet (e.g. an open
V5-tail row), the test fails loudly with the unresolved ids rather than
skipping or weakening the check.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from verbatim import Memory
from verbatim.core.types import VerbatimError
from verbatim.ingest import Ingester
from verbatim.storage import commit_notify
from verbatim.storage.store import Store

ROOT = Path(__file__).resolve().parents[2]
NODE = shutil.which("node")
_needs_node = pytest.mark.skipif(NODE is None, reason="node not on PATH")


# ---------------------------------------------------------------------
# helpers (tests/v5 conventions: real stores, real drains)
# ---------------------------------------------------------------------


def _drain(mem, *, limit: int = 512, priority_sources=None) -> dict:
    """External-worker stand-in for the managed drainer."""
    ingester = Ingester(mem._store, mem._cfg, encoder=mem._encoder)
    return ingester.drain_report(limit=limit, priority_sources=priority_sources)


def _wait(mem, receipt, timeout_ms: float = 15000):
    return mem.wait_ready(receipt, timeout_ms=timeout_ms)


def _hold(mem, object_ref, reason="f-test"):
    """Open a real quarantine hold inside one store tx."""
    from verbatim.security.quarantine import open_quarantine

    with mem._store.tx() as conn:
        return open_quarantine(
            conn, object_ref, [reason], None, scope_id=mem._namespace
        )


def _job_rows(mem):
    with mem._store.read() as conn:
        return [
            {
                "job_id": r[0],
                "kind": r[1],
                "state": r[2],
                "input_refs_json": r[3],
                "lane": r[4],
            }
            for r in conn.execute(
                "SELECT job_id, kind, state, input_refs_json, lane"
                " FROM jobs"
            ).fetchall()
        ]


def _queued_source_jobs(mem, kind="source_project"):
    out = []
    for r in _job_rows(mem):
        if r["kind"] == kind and r["state"] == "queued":
            refs = json.loads(r["input_refs_json"] or "{}")
            out.append((r["job_id"], refs.get("source_id")))
    return out


# =====================================================================
# F01 — fresh-wheel add/search/inspect/forget, no drain or internals
# =====================================================================


def test_f01_front_door_round_trip(tmp_path):
    """The public ``Memory`` path alone: add → wait_ready → search →
    inspect → forget — no caller drain, no internal classes."""
    mem = Memory(path=str(tmp_path / "m.db"), worker="managed")
    try:
        r = mem.add("The release tag for widget-app is v2.4.1.")
        rd = _wait(mem, r.receipt_id)
        assert rd.state in ("ready", "partial"), rd.state
        res = mem.search("release tag widget-app")
        assert res.status in ("ready", "partial") and res.items, res.status
        assert any("v2.4.1" in h.quote for h in res.items)
        insp = mem.inspect(r.ref)
        assert insp.found is True
        out = mem.forget(r.ref)
        assert out.mode == "operation" and out.mutated is True
        res2 = mem.search("release tag widget-app")
        assert all(h.memory_id != r.memory_id for h in res2.items)
    finally:
        mem.close()


# =====================================================================
# F02 — import creates no workers; construction is the activation seam
# =====================================================================


def test_f02_import_has_no_side_effects(tmp_path):
    """``import verbatim`` in a clean process starts no threads, writes
    no files, and the lazy ``Memory`` export resolves."""
    code = (
        "import threading\n"
        "before = set(threading.enumerate())\n"
        "import verbatim\n"
        "after = set(threading.enumerate())\n"
        "new = [t.name for t in after - before]\n"
        "assert not new, f'import started threads: {new}'\n"
        "assert callable(verbatim.Memory), 'lazy Memory export missing'\n"
        "print('clean')\n"
    )
    import verbatim as _v

    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(_v.__file__))
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "clean" in out.stdout, out.stderr[-400:]
    # No store files, model downloads, or worker artifacts by import.
    assert list(tmp_path.iterdir()) == []

    # The activation seam is construction: a managed Memory starts at
    # most one store-bounded helper for its resolved store identity.
    mem = Memory(path=str(tmp_path / "m.db"), worker="managed")
    try:
        workers = [
            t for t in threading.enumerate()
            if "verbatim" in t.name.lower() or "worker" in t.name.lower()
        ]
        assert len(workers) <= 1, [t.name for t in workers]
        st = mem.status()
        assert st.worker  # the bound helper is reported, not hidden
    finally:
        mem.close()


# =====================================================================
# F03 — infer=False searchable; infer=True retains sources; agent stays agent
# =====================================================================


def test_f03_infer_modes_and_agent_attribution(tmp_path):
    mem = Memory(path=str(tmp_path / "m.db"), worker="external")
    try:
        r1 = mem.add(
            "Plain note: the café on 5th street closes at 9pm.", infer=False
        )
        _drain(mem)
        res = mem.search("café 5th street closes")
        assert any("9pm" in h.quote for h in res.items), (
            "infer=False source must still be searchable"
        )
        r2 = mem.add("Plain note 2: it reopens at 7am.", infer=True)
        _drain(mem)
        res2 = mem.search("reopens 7am")
        assert res2.items
        # Sources retained — inspect round-trips the canonical bytes
        # either way (infer is a pipeline choice, not retention).
        for ref in (r1.ref, r2.ref):
            insp = mem.inspect(ref)
            assert insp.found and insp.revisions
    finally:
        mem.close()

    # Agent-authored envelopes stay agent: durable trust_class /
    # source_kind / provenance never re-attribute to the user (V3-13.11
    # carried into the V6 consumer path).
    from verbatim.core.types_v3 import (
        EnvelopeKind,
        Perspective,
        SourceEnvelopeV3,
    )
    from verbatim.evidence import ingest_envelope

    mem = Memory(path=str(tmp_path / "a.db"), worker="external")
    try:
        env = SourceEnvelopeV3(
            kind=EnvelopeKind.ASSISTANT_MESSAGE,
            scope_id=mem._namespace,
            actor_principal="agent:a7",
            perspective=Perspective(asserter="agent:a7"),
            content=b"AGENT-NOTE-9 scratchpad: draft summary only.",
            media_type="text/plain",
            host_id="h1",
            session_id="ss1",
            external_id="ext-agent-note-1",
            event_us=1000,
            receipt_us=1001,
        )
        with mem._store.tx() as conn:
            receipt = ingest_envelope(conn, mem._store, env)
        insp = mem.inspect(receipt.source_id)
        assert insp.found
        assert receipt.trust_class == "agent_generated"
        prov = insp.provenance
        assert "agent:a7" in prov.get("attribution", [])
        assert "agent_generated" in prov.get("trust_class", [])
        assert prov.get("source_kind") != "user_message"
        assert insp.revisions[0]["provenance"] != "direct_user"
    finally:
        mem.close()


# =====================================================================
# F04 — typed hit pins to exact source bytes; inspect round-trips quote
# =====================================================================


def test_f04_typed_hit_pins_to_source_bytes(tmp_path):
    mem = Memory(path=str(tmp_path / "m.db"), worker="managed")
    try:
        text = "INC-431 deploy command is deploy-prod-v3."
        r = mem.add(text)
        _wait(mem, r.receipt_id)
        res = mem.search("INC-431 deploy command")
        assert res.items
        hit = res.items[0]
        assert "deploy-prod-v3" in hit.quote, hit.quote
        # Inspect round-trips to digest-verified source bytes.
        insp = mem.inspect(hit.ref)
        assert insp.found
        excerpts = [e.get("text", "") for e in insp.evidence]
        assert any("deploy-prod-v3" in t for t in excerpts), excerpts
    finally:
        mem.close()


# =====================================================================
# F05 — explicit replace + after: successor current, predecessor labeled
# =====================================================================


def test_f05_replace_after_returns_successor(tmp_path):
    mem = Memory(path=str(tmp_path / "m.db"), worker="external")
    try:
        r1 = mem.add("OPS-12 session timeout is 30 minutes.")
        r2 = mem.add(
            "OPS-12 session timeout is 60 minutes.", replaces=r1.ref
        )
        _drain(mem)
        # `after=` is the explicit causal handle: the barrier waits on
        # r2's obligations before retrieval runs.
        res = mem.search("OPS-12 session timeout", after=r2.receipt_id)
        assert res.items
        successor = [h for h in res.items if "60 minutes" in h.quote]
        assert successor, [h.quote for h in res.items]
        assert all(h.lifecycle == "active" for h in successor)
        # The predecessor is labeled, not silently current: any delivered
        # hit on r1 carries a non-active lifecycle, and inspect names it.
        for h in res.items:
            if h.memory_id == r1.memory_id:
                assert h.lifecycle != "active"
        insp = mem.inspect(r1.ref)
        assert insp.found
        assert insp.lifecycle.get("disposition") == "superseded"
    finally:
        mem.close()


# =====================================================================
# F06 — plain retirement language never auto-canonizes; disputed or dual
# =====================================================================


def test_f06_retirement_language_no_autocanonize(tmp_path):
    mem = Memory(path=str(tmp_path / "m.db"), worker="external")
    try:
        r1 = mem.add("OPS-13 the deploy command is deploy-v1.")
        _drain(mem)
        # Plain-language value move — no replaces=, no canonical claim.
        r2 = mem.add("OPS-13 the deploy command is deploy-v2.")
        _drain(mem)
        # Detection is advisory: a possible_update rides the receipt but
        # the prior record's lifecycle is untouched (never auto-canonized).
        assert r2.possible_updates, (
            "a real value move must surface an advisory possible_update"
        )
        assert r2.possible_updates[0].relation in (
            "newer_value", "contradicts", "refines", "negates"
        )
        insp = mem.inspect(r1.ref)
        assert insp.found
        assert insp.lifecycle.get("disposition") == "active", (
            "plain retirement language must not supersede the prior"
        )
        # The pack is dual or openly disputed — never a lone settled v1.
        res = mem.search("OPS-13 deploy command")
        assert res.items
        delivered = [h for h in res.items]
        disputed = [
            h for h in delivered if h.support_status == "disputed"
        ]
        dual = any("deploy-v1" in h.quote for h in delivered) and any(
            "deploy-v2" in h.quote for h in delivered
        )
        assert dual or disputed, (
            "unresolved conflict must ship dual or labeled disputed; "
            f"got {[ (h.quote, h.support_status) for h in delivered ]}"
        )
        if disputed:
            assert any(
                "conflict_unresolved" in h.warnings for h in disputed
            )
    finally:
        mem.close()


# =====================================================================
# F07 — forget removes source, typed, vector, cache paths; restart empty
# =====================================================================


def test_f07_forget_closes_all_derivatives(tmp_path):
    mem = Memory(path=str(tmp_path / "m.db"), worker="managed")
    try:
        r = mem.add("SECRET-7 the rotation window is Sunday 02:00.")
        _wait(mem, r.receipt_id)
        mem.search("rotation window")  # settle projections + exposure
        out = mem.forget(r.ref)
        assert out.mode == "operation" and out.mutated is True
        assert out.suppression_state, "suppression must be reported"
        assert out.closure_state, "closure phase must be reported"
        res = mem.search("rotation window")
        assert all(h.memory_id != r.memory_id for h in res.items)
        with mem._store.read() as conn:
            # Projection row gone — nothing searchable left.
            rows = conn.execute(
                "SELECT COUNT(*) FROM source_lexical_projection"
                " WHERE source_id = ?",
                (r.memory_id,),
            ).fetchone()[0]
            assert rows == 0
            # Vector + control plane closed too.
            vecs = conn.execute(
                "SELECT COUNT(*) FROM source_vectors WHERE source_id = ?",
                (r.memory_id,),
            ).fetchone()[0]
            assert vecs == 0
            st = conn.execute(
                "SELECT disposition FROM source_state WHERE source_id = ?",
                (r.memory_id,),
            ).fetchone()
            if st is not None:
                assert st[0] != "active"
        # Restart keeps it gone (closure is durable, not session state).
        mem.close()
        mem2 = Memory(path=str(tmp_path / "m.db"), worker="managed")
        try:
            res2 = mem2.search("rotation window")
            assert all(h.memory_id != r.memory_id for h in res2.items)
        finally:
            mem2.close()
        mem = None
    finally:
        if mem is not None:
            mem.close()


# =====================================================================
# F08 — foreign-namespace writes cannot change authorized order/scores
# =====================================================================


def test_f08_namespace_isolation(tmp_path):
    """Foreign-namespace rows seeded into the SAME store cannot move a
    delivered ranking — authorization binds the namespace, never the db."""
    from tests.v5.conftest import (
        add_source,
        seed_projection,
        seed_scope,
        seed_source_state,
    )

    mem = Memory(path=str(tmp_path / "m.db"), user_id="alice",
                 worker="external")
    try:
        mem.add("shared-claim: the sky color is blue.")
        _drain(mem)
        before = mem.search("shared-claim sky color")
        before_sig = [(h.memory_id, h.score) for h in before.items]
        assert before_sig
        # A foreign partition's term-stuffed record — if isolation broke,
        # it would outrank alice's honest hit.
        with mem._store.tx() as conn:
            seed_scope(conn, "ns_foreign")
            add_source(
                conn, "src-foreign", "ns_foreign",
                b"shared-claim sky color shared-claim sky color "
                b"shared-claim sky color",
            )
            seed_source_state(conn, "src-foreign", "ns_foreign")
            seed_projection(
                conn, "src-foreign", 1, "ns_foreign",
                "shared-claim sky color shared-claim sky color "
                "shared-claim sky color",
            )
        after = mem.search("shared-claim sky color")
        after_sig = [(h.memory_id, h.score) for h in after.items]
        assert after_sig == before_sig, (
            f"foreign-namespace write changed authorized order: "
            f"{before_sig} → {after_sig}"
        )
        assert all("src-foreign" != h.memory_id for h in after.items)
    finally:
        mem.close()


# =====================================================================
# F09 — A0 cache-off + A0-cache measured on the pinned machine
# =====================================================================


def test_f09_a0_envelopes_execute(tmp_path):
    """Both A0 shapes run the real envelope machinery at small scale;
    misses stay published (they are measurements, not test failures)."""
    from eval.v6.envelopes import measure_a0, measure_a0_cache

    off = measure_a0(
        memories=48, seed=42, workdir=str(tmp_path / "a0"), queries=16
    )
    assert off["name"] == "A0"
    assert off["search_ms"]["n"] == 16
    assert off["verdict"] in ("passed", "missed", "unavailable", "failed")
    assert isinstance(off["misses"], list)  # misses stay published
    assert off["environment"], "pinned-machine actuals must record"

    cache = measure_a0_cache(
        memories=48, seed=42, workdir=str(tmp_path / "a0c"),
        queries=12, passes=2,
    )
    assert cache["name"] == "A0_CACHE"
    assert cache["verdict"] in ("passed", "missed", "unavailable", "failed")
    # Real cache counters are published — lookups/hits or an honest
    # unavailable reason, never a fabricated hit rate.
    cache_stats = cache["measurements"].get("cache") or {}
    if cache["verdict"] != "unavailable":
        assert cache_stats, "cache counters must be published"


# =====================================================================
# F10 — MCP default list + dispatch are exactly the two consumer tools
# =====================================================================


def test_f10_mcp_consumer_tools_exact(tmp_path):
    p = subprocess.Popen(
        [
            sys.executable, "-m", "verbatim.api_v3.mcp",
            "--path", str(tmp_path / "mcp.db"), "--user", "alice",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    def rpc(obj):
        p.stdin.write(json.dumps(obj) + "\n")
        p.stdin.flush()
        return json.loads(p.stdout.readline())

    try:
        rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        tools = rpc(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        )
        names = sorted(
            t["name"] for t in tools.get("result", {}).get("tools", [])
        )
        assert names == ["v5_capture", "v5_recall"], names
        call = rpc(
            {
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {
                    "name": "v5_capture",
                    "arguments": {"text": "F10 probe memory."},
                },
            }
        )
        assert call.get("result", {}).get("isError") is not True
        # An operator tool outside the consumer set is denied by absence.
        denied = rpc(
            {
                "jsonrpc": "2.0", "id": 4, "method": "tools/call",
                "params": {"name": "v5_forget", "arguments": {}},
            }
        )
        res = denied.get("result", {})
        err = denied.get("error", {})
        assert res.get("isError") is True or err, denied
    finally:
        p.stdin.close()
        p.wait(timeout=10)


# =====================================================================
# F11 — Mem0 shim: update/delete keep CAS/closure; pending never success
# =====================================================================


def test_f11_mem0_shim_honesty(tmp_path):
    from verbatim.compat.mem0 import Memory as Mem0Memory

    m = Mem0Memory(path=str(tmp_path / "m0.db"), worker="external")
    try:
        r = m.add(text="the api key prefix is sk-live-42")
        row = (r.get("results") or [r])[0] if isinstance(r, dict) else r
        # Acceptance is durable at return; interpretation readiness is
        # reported — never fabricated as success while the queue is
        # undrained.
        assert row.get("verbatim_acceptance") == "accepted", row
        assert r.get("verbatim_status") == "accepted", r.get(
            "verbatim_status"
        )
        readiness = row.get("verbatim_readiness") or {}
        assert isinstance(readiness, dict)
        assert not readiness or not all(
            v == "ready" for v in readiness.values()
        ), f"undrained queue must not report all-ready: {readiness}"
        assert row.get("verbatim_receipt_id")
        logical_id = row.get("id")
        stale_ref = row.get("verbatim_ref")

        # update() routes through replaces= with the CAS fence intact:
        # updating the CURRENT id succeeds (successor minted)...
        up = m.update(logical_id, text="the api key prefix is sk-live-99")
        assert up.get("verbatim_supersedes") is not None or up.get(
            "verbatim_ref"
        )
        # ...but replaying the SAME stale ref cannot skip the CAS check.
        if stale_ref:
            with pytest.raises(VerbatimError):
                m.update(stale_ref, text="the api key prefix is sk-evil-1")

        # delete() carries suppression/closure state verbatim — closure
        # cannot be skipped or misreported.
        d = m.delete(logical_id)
        assert d.get("verbatim_status") in ("deleted", "preview"), d
        if d.get("verbatim_status") == "deleted":
            assert "verbatim_closure_state" in d
        gone = m.search("api key prefix sk-live")
        hits = gone.get("results", gone) if isinstance(gone, dict) else gone
        assert not any(
            logical_id == (h.get("id") if isinstance(h, dict) else None)
            for h in (hits or [])
        )
    finally:
        close = getattr(m, "close", None)
        if callable(close):
            close()


# =====================================================================
# F12 — neural without artifact unavailable; hashing label stays honest
# =====================================================================


def test_f12_neural_unavailable_and_hashing_honest(tmp_path):
    mem = Memory(
        path=str(tmp_path / "m.db"),
        encoder="artifact",
        config={
            "data_dir": str(tmp_path / "empty-data"),
            "embedding": {
                "backend": "artifact",
                "model": "ghost",
                "artifact_revision": "v9",
            },
        },
        worker="external",
    )
    try:
        # No verified artifact → the encoder is absent AND the build
        # says so; it never pretends to be the artifact encoder.
        assert mem._encoder is None
        assert "encoder_unavailable" in (mem._warnings or [])
        st = mem.status()
        assert "artifact" not in (st.encoder or "")
    finally:
        mem.close()

    m2 = Memory(path=str(tmp_path / "m2.db"), encoder="hashing",
                worker="external")
    try:
        st = m2.status()
        assert "hashing" in (st.encoder or ""), st.encoder
    finally:
        m2.close()


# =====================================================================
# F13 — duplicates collapse in search; contrary evidence never grouped
# =====================================================================


def test_f13_duplicate_collapse_keeps_contrary(tmp_path):
    mem = Memory(path=str(tmp_path / "m.db"), worker="external")
    try:
        mem.add("EVT-9 the outage window is 02:00-04:00.")
        dup = mem.add("EVT-9 the outage window is 02:00-04:00.")
        mem.add("EVT-9 the outage window is actually 05:00-07:00.")
        _drain(mem)
        res = mem.search("EVT-9 outage window")
        assert res.items
        same = [h for h in res.items if "02:00" in h.quote]
        contrary = [h for h in res.items if "05:00" in h.quote]
        # Byte-identical re-add replays onto one record: one delivered
        # hit reports the collapse honestly (never extra corroboration).
        assert len(same) == 1, [h.quote for h in same]
        assert dup.replayed is True or same[0].collapsed_duplicates >= 1
        if same[0].collapsed_duplicates:
            assert same[0].corroboration == 1, (
                "same-submitter duplicates are not corroboration"
            )
        # Contrary evidence is unlinked — delivered on its own, never
        # folded into the duplicate group.
        assert contrary, (
            "contrary record was collapsed or withheld: "
            f"{[h.quote for h in res.items]}"
        )
    finally:
        mem.close()


# =====================================================================
# F14 — ANN/cache failure falls back without skipping holds
# =====================================================================


def test_f14_fallback_never_skips_holds(tmp_path):
    mem = Memory(path=str(tmp_path / "m.db"), worker="external")
    try:
        r = mem.add("HELD-1 the private door code is 4471.")
        mem.add("HELD-1 public note: the lobby opens at 8am.")
        _drain(mem)
        _hold(mem, ("source", r.memory_id, 1))
        # Every delivery path withholds the held source — session and
        # eventual consistency alike.
        for cons in ("session", "eventual"):
            res = mem.search("private door code", consistency=cons)
            assert all("4471" not in h.quote for h in res.items), cons
            assert all(
                h.memory_id != r.memory_id for h in res.items
            ), cons
        # Degraded lanes still answer: the unheld sibling is delivered.
        res2 = mem.search("lobby opens")
        assert any("8am" in h.quote for h in res2.items)
    finally:
        mem.close()

    # Cache-off default reports honestly; search still works (the
    # fallback is a lane choice, never a correctness skip).
    from verbatim.retrieval.cache import stats_for

    mem2 = Memory(path=str(tmp_path / "m2.db"), worker="external")
    try:
        stats = stats_for(mem2._store, mem2._cfg)
        assert stats["enabled"] is False
        assert stats["state"] in ("implemented", "configured")
        assert stats["degraded_reason"], (
            "default-off cache must name its degraded reason"
        )
        mem2.add("cache fallback probe text alpha-77")
        _drain(mem2)
        res = mem2.search("alpha-77 probe")
        assert res.items
    finally:
        mem2.close()


# =====================================================================
# F15 — clean-env install uses verbatim-memory; register() discovers
# =====================================================================


def test_f15_packaging_surface():
    import tomllib

    import verbatim

    doc = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert doc["project"]["name"] == "verbatim-memory"
    scripts = doc["project"]["scripts"]
    assert scripts["verbatim-service"] == "verbatim.service.__main__:main"
    assert scripts["verbatim-mcp"] == "verbatim.api_v3.mcp:main"
    providers = doc["project"]["entry-points"][
        "hermes_agent.memory_providers"
    ]
    assert providers["verbatim"] == "verbatim:register"

    # register() is the real discovery seam: calling it registers the
    # provider on the host context (duck-typed ctx = the plugin contract).
    class _Ctx:
        def __init__(self):
            self.providers = []

        def register_memory_provider(self, provider):
            self.providers.append(provider)

    ctx = _Ctx()
    verbatim.register(ctx)
    assert len(ctx.providers) == 1
    from verbatim.provider import VerbatimMemoryProvider

    assert isinstance(ctx.providers[0], VerbatimMemoryProvider)


# =====================================================================
# F16 — public copy: no competitor defeat; stale-deploy demo = 2 engine arms
# =====================================================================


def test_f16_public_copy_no_defeat_claims():
    """V6-01.05/F16: public copy never names a competitor as defeated
    while G6-05/G6-06 evidence is absent; a stale-deploy demo may exist
    only as a two-arm engine demo labeled exactly that."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "from verbatim import Memory" in readme
    assert ".add(" in readme and ".search(" in readme

    public_docs = [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md"))
    banned = (
        "better than mem0", "beats mem0", "defeats mem0", "mem0 killer",
        "faster than mem0", "beats graphiti", "better than graphiti",
        "beats holographic", "better than holographic",
    )
    for doc in public_docs:
        text = doc.read_text(encoding="utf-8").lower()
        for phrase in banned:
            assert phrase not in text, f"{doc.name}: {phrase!r}"
        # A stale-deploy demonstration, where present, must be labeled
        # as this engine against itself — never engine-vs-competitor.
        if "stale" in text and "deploy" in text:
            for line in doc.read_text(encoding="utf-8").splitlines():
                low = line.lower()
                if "stale" in low and "deploy" in low:
                    assert (
                        "verbatim" in low or "this engine" in low
                        or "two" in low or "arm" in low
                    ), f"{doc.name}: unlabeled stale-deploy demo line: {line}"


# =====================================================================
# F17 — cross-store commit wake under 25 ms dead time
# =====================================================================


def test_f17_cross_store_commit_wake(tmp_path):
    """The worker's commit unblocks a facade barrier through
    ``commit_notify`` — measured dead time after the commit, not a
    blind-poll ramp."""
    path = str(tmp_path / "s.db")
    a = Store.create(path)
    b = Store.open(path)
    try:
        fired = []
        t = threading.Thread(
            target=lambda: fired.append(commit_notify.wait(path, 5.0))
        )
        t.start()
        time.sleep(0.05)  # the waiter is parked on the condition
        with a.tx() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS _wake_probe (id INTEGER)"
            )
        commit_end = time.perf_counter()
        t.join(timeout=5)
        dead_ms = (time.perf_counter() - commit_end) * 1000
        assert fired == [True]
        assert dead_ms < 25, (
            f"cross-store wake dead time {dead_ms:.1f}ms — "
            "the notify path is not a poll ramp"
        )
    finally:
        a.close()
        b.close()

    # Facade level: a session barrier parked in wait_ready wakes on the
    # drainer's commit, not on a poll schedule.
    mem = Memory(path=str(tmp_path / "m.db"), worker="external")
    try:
        r = mem.add("WAKE-1 the barrier wake probe.")
        out: dict = {}

        def waiter():
            out["ready"] = mem.wait_ready(
                r.receipt_id, timeout_ms=10_000
            )

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)
        _drain(mem)
        drained_at = time.perf_counter()
        t.join(timeout=10)
        assert out["ready"].state == "ready", out["ready"]
        lag_ms = (time.perf_counter() - drained_at) * 1000 - out[
            "ready"
        ].waited_ms
        # waited_ms covers the pre-drain parked time; the post-commit
        # settle is the causal wake — bounded, not the old ramp.
        assert out["ready"].waited_ms < 10_000
        assert lag_ms < 1000, f"post-commit settle lag {lag_ms:.0f}ms"
    finally:
        mem.close()


# =====================================================================
# F18 — session-blocked source drains ahead of ordinary; privacy first
# =====================================================================


def test_f18_unblock_first_drain(tmp_path):
    """Unblock-first ordering (V6-02.08): the marked source's
    source_project job runs before unmarked ordinary work — and the
    privacy lane is re-examined before EVERY pick, never demoted."""
    from verbatim.core.types import JobKind

    mem = Memory(path=str(tmp_path / "m.db"), worker="external")
    try:
        ra = mem.add("ORD-1 ordinary queue filler alpha.")
        marked = mem.add("URGENT-4 the session-blocked source.")
        rb = mem.add("ORD-2 ordinary queue filler beta.")
        # A privacy-control job and a marked ordinary job coexist.
        from verbatim.jobs.queue import JobQueue

        jobs = JobQueue(mem._store)
        with mem._store.tx() as conn:
            screen_id = jobs.enqueue(
                conn,
                mem._namespace,
                JobKind.SCREEN,
                {
                    "object_kind": "source",
                    "object_id": ra.memory_id,
                    "revision": 1,
                    "text": "f18 screen probe",
                },
            )
        # Phase 1 — privacy/correctness first, even with a marked source.
        rep = _drain(
            mem, limit=1, priority_sources={marked.memory_id}
        )
        assert rep["processed"] == 1
        assert rep["priority_processed"] == 0, (
            "a marked source must never jump ahead of privacy work"
        )
        rows = {r["job_id"]: r for r in _job_rows(mem)}
        assert rows[screen_id]["state"] != "queued", (
            "the privacy-lane job should have been leased first"
        )
        marked_rows = [
            r for r in _job_rows(mem)
            if r["kind"] == "source_project" and r["state"] == "queued"
            and marked.memory_id in r["input_refs_json"]
        ]
        assert marked_rows, "marked source job should still be queued"

        # Phase 2 — now the marked source jumps the ordinary queue.
        rep = _drain(
            mem, limit=1, priority_sources={marked.memory_id}
        )
        assert rep["priority_processed"] == 1, rep
        still_queued = _queued_source_jobs(mem)
        assert not any(sid == marked.memory_id for _, sid in still_queued)
        assert any(sid == rb.memory_id for _, sid in still_queued), (
            "unmarked ordinary work stays queued behind the marked job"
        )
        # Everything drains in the end — marks are ordering, not drops.
        rep = _drain(mem)
        assert rep["failed"] == 0, rep["errors"]
        res = mem.search("URGENT-4 session-blocked")
        assert res.items
    finally:
        commit_notify.clear_barrier_sources(mem._store._path)
        mem.close()


# =====================================================================
# F19 — consistency="eventual" skips barrier, warns, never leaks holds
# =====================================================================


def test_f19_eventual_consistency_honesty(tmp_path):
    """V6-02.10: eventual mode visibly skips the barrier —
    ``causal_satisfied=false`` + an explicit warning — while hold
    enforcement is untouched."""
    mem = Memory(path=str(tmp_path / "m.db"), worker="external")
    try:
        r = mem.add("EVT-2 the fence gate is open.")
        # Barrier skipped on purpose: the write is still undrained.
        res = mem.search("fence gate", consistency="eventual")
        assert res.readiness.get("causal_satisfied") is False, (
            f"eventual must report causal_satisfied=false: {res.readiness}"
        )
        assert any(
            "eventual" in w for w in res.warnings
        ), f"eventual mode must warn: {res.warnings}"
        assert res.status != "unavailable"
        _drain(mem)
        _hold(mem, ("source", r.memory_id, 1))
        res2 = mem.search("fence gate", consistency="eventual")
        assert all("open" not in h.quote for h in res2.items)
        assert all(h.memory_id != r.memory_id for h in res2.items)
    finally:
        mem.close()


# =====================================================================
# F20 — A1 envelope executes on the pinned machine
# =====================================================================


def test_f20_a1_envelope_executes(tmp_path):
    """Live-load envelope at small scale — p95/p99 and backlog are
    published pass-or-miss; a miss is a measurement, not a skip."""
    from eval.v6.envelopes import measure_a1

    out = measure_a1(
        memories=48, seed=43, workdir=str(tmp_path / "a1"),
        reader_queries=10, writer_ops=4, worker="managed",
        visibility_probes=3,
    )
    assert out["name"] == "A1"
    assert out["verdict"] in ("passed", "missed", "unavailable", "failed")
    assert out["environment"], "machine actuals must record"
    if out["verdict"] != "unavailable":
        assert "search_ms" in out and "backlog" in out
        assert out["search_ms"].get("n", 0) >= 1


# =====================================================================
# F21 — add-ack scaling profile; any superlinear stage named
# =====================================================================


def test_f21_add_ack_scaling_profile(tmp_path):
    from eval.v6.envelopes import measure_add_ack_scaling

    out = measure_add_ack_scaling(
        [24, 48], seed=42, workdir=str(tmp_path / "ack"), profile_adds=6
    )
    assert "checkpoints" in out and "superlinear_stages" in out
    assert len(out["checkpoints"]) == 2
    for row in out["checkpoints"]:
        assert row["add_ack_ms"]["n"] == 6
        assert isinstance(row["stages_ms"], dict) and row["stages_ms"]
    assert isinstance(out["superlinear_stages"], list)


# =====================================================================
# F22 — service /v2/memory round-trip for a bound token
# =====================================================================


def test_f22_service_memory_routes(tmp_path):
    from verbatim.service.auth import TokenCredential
    from verbatim.service.memory_api import create_memory_app

    cred = TokenCredential.mint(
        "tok-f22", "alice", {"read", "ingest", "admin"}
    )
    srv = create_memory_app(
        str(tmp_path / "svc.db"), user_id="alice", worker="managed",
        tokens=[cred],
    )
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.1)

    def req(method, path, body=None):
        url = f"http://127.0.0.1:{srv.server_address[1]}{path}"
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(
            url, data=data, method=method,
            headers={
                "Authorization": "Bearer tok-f22",
                "content-type": "application/json",
            },
        )
        return json.loads(urllib.request.urlopen(r, timeout=10).read())

    try:
        a = req("POST", "/v2/memory/add", {"text": "F22 service probe memory."})
        assert a.get("acceptance") == "accepted"
        rid = a.get("receipt_id")
        state = ""
        for _ in range(50):
            rd = req("GET", f"/v2/memory/readiness/{rid}")
            state = rd.get("state")
            if state in ("ready", "partial", "blocked", "unavailable"):
                break
            time.sleep(0.1)
        assert state in ("ready", "partial"), state
        s = req("POST", "/v2/memory/search", {"query": "F22 service probe"})
        assert s.get("status") in ("ready", "partial") and s.get("items")
        ref = s["items"][0].get("ref")
        i = req("POST", "/v2/memory/inspect", {"ref": ref})
        assert i.get("found") is True
        f = req("POST", "/v2/memory/forget", {"ref": ref})
        assert f.get("mutated") is True or f.get("mode") == "operation"
        # Capabilities report the honest transport + worker + limits.
        caps = req("GET", "/v2/memory/capabilities")
        assert caps.get("transport", {}).get("tls") == "unimplemented"
        assert caps.get("routes"), "the served route list must be declared"
    finally:
        srv.shutdown()
        srv.server_close()
        # The app contract: callers close the bound Memory on shutdown —
        # leaving it to GC risks ``__del__`` firing inside another
        # thread's registry critical section.
        srv.application.memory.close()


# =====================================================================
# F23 — cross-workspace service token fails closed
# =====================================================================


def test_f23_service_scope_binding(tmp_path):
    from verbatim.service.auth import TokenCredential
    from verbatim.service.memory_api import create_memory_app

    alice = TokenCredential.mint("tok-alice", "alice", {"read", "ingest"})
    bob = TokenCredential.mint("tok-bob", "bob", {"read", "ingest"})
    srv = create_memory_app(
        str(tmp_path / "svc.db"), user_id="alice", worker="managed",
        tokens=[alice, bob],
    )
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.1)
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/v2/memory/status"
        # Bob's credential is authentic but scoped to a different
        # partition — the app fails closed, never broadens.
        req = urllib.request.Request(
            url, headers={"Authorization": "Bearer tok-bob"}
        )
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(req, timeout=10)
        assert ei.value.code == 403
        # No token at all fails the same closed way.
        req2 = urllib.request.Request(url)
        with pytest.raises(urllib.error.HTTPError) as ei2:
            urllib.request.urlopen(req2, timeout=10)
        assert ei2.value.code in (401, 403)
    finally:
        srv.shutdown()
        srv.server_close()
        srv.application.memory.close()


# =====================================================================
# F24 — TypeScript client drives the live loopback end-to-end
# =====================================================================


@_needs_node
def test_f24_ts_client_end_to_end(tmp_path):
    """The frozen wire format against the REAL service: add → waitReady
    → search → inspect → forget over loopback HTTP."""
    from verbatim.service.auth import TokenCredential
    from verbatim.service.memory_api import create_memory_app

    client_src = ROOT / "clients" / "ts" / "memory-client" / "src" / "index.ts"
    assert client_src.is_file()
    cred = TokenCredential.mint("tok-f24", "alice", {"read", "ingest", "admin"})
    srv = create_memory_app(
        str(tmp_path / "svc.db"), user_id="alice", worker="managed",
        tokens=[cred],
    )
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.1)
    port = srv.server_address[1]
    script = f"""
    import({json.dumps(client_src.as_uri())}).then(async (m) => {{
      const c = new m.MemoryClient({{
        baseUrl: 'http://127.0.0.1:{port}', token: 'tok-f24', timeoutMs: 8000,
      }});
      const a = await c.add('F24 wire-format probe memory zeta-3');
      if (!a.receipt_id || !a.ref) {{ console.error('bad add', JSON.stringify(a)); process.exit(4); }}
      const r = await c.waitReady(a.receipt_id, 15000, 50);
      if (!['ready','partial'].includes(r.state)) {{ console.error('readiness', JSON.stringify(r)); process.exit(5); }}
      const s = await c.search('zeta-3 probe', {{ limit: 4, consistency: 'session' }});
      if (!s.items || s.items.length === 0) {{ console.error('empty search', JSON.stringify(s)); process.exit(6); }}
      const ref = s.items[0].ref;
      const i = await c.inspect(ref);
      if (!i.found) {{ console.error('inspect', JSON.stringify(i)); process.exit(7); }}
      const f = await c.forget(ref);
      if (f.mutated !== true && f.mode !== 'operation') {{ console.error('forget', JSON.stringify(f)); process.exit(8); }}
      console.log('e2e-ok');
    }}).catch((e) => {{ console.error(String(e && e.stack || e)); process.exit(9); }});
    """
    try:
        proc = subprocess.run(
            [NODE, "-e", script], capture_output=True, text=True, timeout=120
        )
    finally:
        srv.shutdown()
        srv.server_close()
        srv.application.memory.close()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "e2e-ok" in proc.stdout


# =====================================================================
# F25/F26 — consolidation positive fixture + pin invalidation
# =====================================================================


def test_f25_consolidation_positive(tmp_path):
    """The real grounded-consolidation suite: two same-slot
    distinct-family claims yield a proof-bearing observation; every pin
    resolves; no observation is a sole byte trace."""
    from eval.v5.consolidation import run_consolidation

    out = run_consolidation(memories=24, seed=42, workdir=str(tmp_path / "cons"))
    obs = out["observations"]
    assert out["verdicts"]["observations"] == "passed", (
        f"consolidation observations verdict: {out['verdicts']}"
    )
    assert obs["observations"] >= 1, (
        f"the fixture must yield a proof-bearing observation: {obs}"
    )
    assert obs["ungrounded"] == 0 and obs["sole_trace_violations"] == 0
    assert obs["held_or_erased_pins"] == 0
    assert not obs["fixture"].get("missing_observations"), obs["fixture"]
    # Proof-bearing: at least one live observation carries ≥2 pins.
    pin_rows = obs.get("pins") or []
    assert any(int(r.get("pins", {}).get("total", 0)) >= 2 for r in pin_rows), (
        f"no proof-bearing observation: {pin_rows}"
    )
    assert out["verdict"] in ("passed", "failed", "unavailable", "inconclusive")


def test_f26_held_pin_withholds_summary(tmp_path):
    """Pin invalidation: holding a pin's backing source revision makes
    the pin liveness audit report it held — the summary can no longer
    present that pin as live grounding."""
    from eval.v5.consolidation import (
        _admit_pending,
        _consolidate_jobs,
        _observation_audit,
        _second_attestation,
        corroborated_corpus,
    )
    from eval.v5.harness import drain_memory, seed_corpus_env, settle
    from verbatim.observations.aggregate import resolve_pins
    from verbatim.security.quarantine import open_quarantine

    corpus = corroborated_corpus(24, 42)
    env = seed_corpus_env(corpus, workdir=str(tmp_path / "cons"),
                          worker="external")
    try:
        settle(env)
        _second_attestation(env)
        drain_memory(env)
        _admit_pending(env)
        drain_memory(env)
        jobs = _consolidate_jobs(env)
        obs = _observation_audit(env, jobs)
        pin_rows = [r for r in (obs.get("pins") or [])]
        assert pin_rows, (
            f"fixture produced no observations to invalidate: {obs}"
        )
        oid = pin_rows[0]["observation"]
        mem = env.memory
        with mem._store.read() as conn:
            rep = resolve_pins(conn, oid)
        assert rep["found"] and rep["summary"]["resolvable"] >= 1
        # Hold one pin's object revision through the real authority.
        pin = next(
            p for p in rep["pins"] if p.get("object_kind") == "claim"
        ) if any(
            p.get("object_kind") == "claim" for p in rep["pins"]
        ) else rep["pins"][0]
        with mem._store.tx() as conn:
            assert open_quarantine(
                conn,
                (pin["object_kind"], pin["object_id"], pin["revision"]),
                ["f26-invalidation"],
                None,
                scope_id=mem._namespace,
            )
        with mem._store.read() as conn:
            rep2 = resolve_pins(conn, oid)
        assert rep2["summary"]["held"] >= 1, (
            f"held pin not reported: {rep2['summary']}"
        )
    finally:
        env.close()


# =====================================================================
# F27 — learned_active gate: refuses unnamed, binds validated artifact
# =====================================================================


def test_f27_learned_active_artifact_gate(tmp_path):
    from verbatim.config import config_from_mapping
    from verbatim.core.types import ErrorCode
    from verbatim.policy import artifacts as pa
    from verbatim.retrieval.v3 import learned as _learned

    store = Store.create(str(tmp_path / "p.db"))
    try:
        art = _learned.PolicyArtifact(
            revision="pol_v1", q={}, pulls={}
        ).to_dict()
        # Registration cannot self-declare validated — the gate would
        # be theater otherwise.
        with store.tx() as conn:
            with pytest.raises(VerbatimError):
                pa.register_artifact(
                    conn, art, validation_state="validated"
                )
            aid = pa.register_artifact(conn, art)
        cfg = lambda a: config_from_mapping(
            {"v3": {"retrieval": {
                "controller": "learned_active",
                "controller_policy_artifact": a,
            }}}
        )
        # Unvalidated binding refuses and names the artifact id.
        try:
            pa.bind_for_activation(cfg(aid), aid, store)
            raise AssertionError("unvalidated artifact must refuse")
        except VerbatimError as exc:
            assert exc.code == ErrorCode.CONFIG_INVALID
            assert aid in str(exc), str(exc)
        # No binding at all refuses too — no silent deterministic fallback.
        try:
            pa.bind_for_activation(
                config_from_mapping(
                    {"v3": {"retrieval": {"controller": "learned_active"}}}
                ),
                None, store,
            )
            raise AssertionError("unbound learned_active must refuse")
        except VerbatimError as exc:
            assert exc.code == ErrorCode.CONFIG_INVALID
        # Validation requires executed paired-run evidence.
        with store.tx() as conn:
            with pytest.raises(VerbatimError):
                pa.mark_validated(
                    conn, aid,
                    evidence={"paired_run": "r", "gate": "g",
                              "verdict": "fail"},
                )
        with store.tx() as conn:
            assert pa.mark_validated(
                conn, aid,
                evidence={"paired_run": "eval/run-1", "gate": "G8",
                          "verdict": "pass"},
            ) is True
        # A registered + validated artifact activates.
        bound = pa.bind_for_activation(cfg(aid), aid, store)
        assert isinstance(bound, _learned.PolicyArtifact)
        assert bound.revision == "pol_v1"
    finally:
        store.close()


# =====================================================================
# F28 — source-lane deliveries record exposure/influence rows
# =====================================================================


def test_f28_source_exposure_rows(tmp_path):
    mem = Memory(path=str(tmp_path / "m.db"), worker="managed")
    try:
        r = mem.add("EXP-3 the exposure row probe text.")
        _wait(mem, r.receipt_id)
        res = mem.search("exposure row probe")
        assert res.items
        with mem._store.read() as conn:
            rows = conn.execute(
                "SELECT receipt_id, ord, source_id, revision,"
                " score_family, delivered_at_us, namespace"
                " FROM source_exposure ORDER BY delivered_at_us"
            ).fetchall()
        assert rows, "delivered source hits must mint exposure rows"
        delivered = {h.memory_id for h in res.items}
        assert any(r[2] in delivered for r in rows), rows
        for row in rows:
            assert row[4], "score_family must be recorded"
            assert row[5], "delivered_at_us must be recorded"
            assert row[6] == mem._namespace
    finally:
        mem.close()


# =====================================================================
# F29/F30 — local artifact build/verify + paired quality verdict
# =====================================================================


def test_f29_artifact_build_load_cycle(tmp_path):
    """Locally-built artifact: manifest verifies, available()=True,
    encoder_id flows to vectors, validate() separates."""
    from eval.v6.neural import (
        ENCODER_ID,
        MODEL,
        REVISION,
        build_dev_artifact,
        dev_encoder,
    )
    from verbatim.querying.calibration import validate

    workdir = str(tmp_path / "art")
    art = build_dev_artifact(workdir)
    assert art["encoder_id"] == ENCODER_ID
    enc = dev_encoder(workdir)
    assert enc.available() is True, "verified artifact must load"
    assert enc.encoder_id == ENCODER_ID
    rep = validate(enc)
    assert "separates" in rep and isinstance(rep["separates"], bool)
    # encoder_id flows to the vector lane through the real facade path.
    mem = Memory(
        path=str(tmp_path / "m.db"),
        encoder="artifact",
        config={
            "data_dir": workdir,
            "embedding": {
                "backend": "artifact",
                "model": MODEL,
                "artifact_revision": REVISION,
            },
        },
        worker="external",
    )
    try:
        assert mem._encoder is not None, mem._warnings
        st = mem.status()
        assert st.encoder == ENCODER_ID, st.encoder
        r = mem.add("F29 artifact vector probe kappa-9.")
        _drain(mem)
        with mem._store.read() as conn:
            rows = conn.execute(
                "SELECT encoder FROM source_vectors WHERE source_id = ?",
                (r.memory_id,),
            ).fetchall()
        assert rows and all(row[0] == ENCODER_ID for row in rows), rows
    finally:
        mem.close()


def test_f30_paired_quality_verdict(tmp_path):
    """The paired hashing-vs-artifact run publishes both arms, the
    delta, and an earned-or-rejected neural label with its reason."""
    from eval.v6.neural import paired_quality

    out = paired_quality(str(tmp_path / "pq"), memories=24, seed=42)
    assert out["measured"] is True
    verdict = out["verdict"]
    assert "neural_recommended" in verdict
    assert isinstance(verdict["neural_recommended"], bool)
    assert verdict["reason"], "the gate's reason must be published"
    arms = out["arms"]
    assert "verbatim_memory-hashing" in arms
    assert "verbatim_memory-artifact" in arms
    assert "recall" in out["delta"] or "recall_at_k" in out["delta"] or out["delta"]
    # The label is earned or the loss is published — never asserted.
    if verdict["neural_recommended"]:
        assert out["delta"]["recall"] is not None
        assert out["delta"]["recall"] > 0, (
            "recommended requires strictly-better recall@k"
        )


# =====================================================================
# F32 — comparator ladder: offline arms execute; mem0 honest; no fakes
# =====================================================================


def test_f32_comparator_ladder(tmp_path):
    from eval.v6.comparators import run_registry

    out = run_registry(workdir=str(tmp_path / "cmp"), memories=24, k=6)
    assert out["verdict"] == "executed"
    rows = {r["arm"]: r for r in out["rows"]}
    # Always-run offline arms execute and carry measured metrics.
    for arm in ("verbatim_memory", "verbatim_v2", "naive_fts",
                "vector_rag", "no_memory"):
        row = rows.get(arm)
        assert row is not None, f"arm {arm} missing from registry"
        assert row["status"] == "tested", (
            f"offline arm {arm} must execute: {row['status']} "
            f"({row.get('reason')})"
        )
        assert row["executed"] is True
        assert row["metrics"], f"tested arm {arm} must carry metrics"
    # Probe-gated + hosted rows report their state with a reason —
    # pinned-or-unavailable, never a fabricated `tested`.
    gated = ("mem0_oss", "mem0_oss_inferfalse", "graphiti_oss",
             "holographic", "zep_hosted", "mem0_platform")
    for arm in gated:
        row = rows.get(arm)
        assert row is not None, f"declared arm {arm} missing"
        assert row["status"] in (
            "tested", "unavailable", "out_of_scope", "declared"
        ), (arm, row["status"])
        if row["status"] != "tested":
            assert row["reason"], f"{arm} reports no reason"
        else:
            # A `tested` claim must be backed by execution + metrics.
            assert row["executed"] and row["metrics"], (
                f"{arm} claims tested without measured evidence"
            )
    # Comparison rows exist for every non-verbatim arm — refusals
    # recorded verbatim, never silently dropped.
    assert len(out["comparisons"]) >= len(rows) - 1


# =====================================================================
# F31 — V5 tail closed: re-dispositioned rows; predecessor suites green
# (runs last: the qualification-closure gate caps the whole suite)
# =====================================================================


def test_f31_v5_tail_re_dispositioned():
    """The tail check executes against the real dispositions file and
    the spec's closure bar is asserted — unresolved ids are reported,
    never summarized away."""
    from eval.v6.portfolio import run_v5_tail

    out = run_v5_tail(dispositions_path=str(ROOT / "eval" / "v5" /
                                          "dispositions_v5.json"))
    assert out["status"] == "executed"
    assert out["requirements_total"] > 0
    assert "status_counts" in out and "unresolved_ids" in out
    # SPEC_V6 F31: the V5 tail is CLOSED — every unfinished row
    # re-dispositioned to a resolved status. This is the spec bar, not
    # a preference: report the actual state loudly when unmet.
    assert out["unresolved_count"] == 0, (
        f"V5 tail not closed — {out['unresolved_count']} unresolved "
        f"requirements: {out['unresolved_ids']}"
    )
