"""V5 consumer facade tests — ``verbatim.Memory`` (SPEC_V5 §05–§09, §15).

Real on-disk ``Store`` per test (``tmp_path``), real governance, real
ingest, real readiness, real purge machinery.  Worker modes are exercised
both ways (``external`` for deterministic lifecycle tests, ``managed``
for the shared-worker path).  No kernel mocks.
"""

from __future__ import annotations

import os

import pytest

from verbatim import Memory
from verbatim.core.types import ErrorCode, JobKind, VerbatimError
from verbatim.ingest import Ingester
from verbatim.memory.types import (
    AddResult,
    ForgetResult,
    MemoryRef,
    Readiness,
    SearchResult,
)

OWNER = "local-owner"  # LocalHost's authenticated principal


@pytest.fixture()
def path(tmp_path):
    return str(tmp_path / "m.db")


@pytest.fixture()
def mem(path):
    m = Memory(path=path, worker="external")
    yield m
    try:
        m.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_create_and_close(self, path):
        m = Memory(path=path, worker="external")
        assert os.path.exists(path)
        report = m.close()
        assert report.closed is True

    def test_context_manager(self, path):
        with Memory(path=path, worker="external") as m:
            r = m.add("hello")
            assert r.memory_id
        assert m._closed

    def test_create_false_missing_store(self, tmp_path):
        missing = str(tmp_path / "absent.db")
        with pytest.raises(VerbatimError):
            Memory(path=missing, create=False, worker="external")

    def test_profile_validation(self, path):
        with pytest.raises(VerbatimError) as e:
            Memory(path=path, profile="team_service", worker="external")
        assert e.value.code == ErrorCode.CONFIG_INVALID

    def test_worker_mode_validation(self, path):
        with pytest.raises(VerbatimError) as e:
            Memory(path=path, worker="sideways")
        assert e.value.code == ErrorCode.VALIDATION

    def test_encoder_validation(self, path):
        with pytest.raises(VerbatimError):
            Memory(path=path, worker="external", encoder="quantum")
        m = Memory(path=path, worker="external", encoder="none")
        assert m.status().encoder == "none"
        m.close()

    def test_ready_timeout_validation(self, path):
        with pytest.raises(VerbatimError):
            Memory(path=path, worker="external", ready_timeout_ms=-1)
        with pytest.raises(VerbatimError):
            Memory(path=path, worker="external", ready_timeout_ms="fast")

    def test_config_mapping(self, tmp_path):
        m = Memory(
            config={"data_dir": str(tmp_path / "cfgdata"), "v3": {"profile": "local_memory"}},
            worker="external",
        )
        m.close()

    def test_bad_config_type(self, path):
        with pytest.raises(VerbatimError):
            Memory(path=path, worker="external", config="yaml-ish")

    def test_import_is_side_effect_free(self):
        import verbatim

        assert "Memory" in verbatim.__all__
        assert verbatim.Memory is Memory

    def test_reopen_convergent(self, path):
        m1 = Memory(path=path, worker="external")
        r = m1.add("persist me")
        ns = m1._namespace
        m1.close()
        m2 = Memory(path=path, worker="external")
        assert m2._namespace == ns
        s = m2.search("persist me")
        assert any(h.memory_id == r.memory_id for h in s.items)
        m2.close()

    def test_user_id_is_a_namespace_alias(self, path):
        m1 = Memory(path=path, worker="external")
        ns_default = m1._namespace
        m1.close()
        m2 = Memory(path=path, worker="external", user_id="work")
        assert m2._namespace != ns_default
        assert m2._owner == OWNER  # alias label, never authority
        m2.close()
        m3 = Memory(path=path, worker="external")
        assert m3._namespace == ns_default
        m3.close()


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_result_shape(self, mem):
        r = mem.add("the wifi password is swordfish")
        assert isinstance(r, AddResult)
        assert r.memory_id
        assert r.source_revision == 1
        assert r.receipt_id
        assert r.acceptance == "accepted"
        assert r.replayed is False
        ref = MemoryRef.parse(r.ref)
        assert ref.source_id == r.memory_id
        assert ref.namespace == mem._namespace
        assert ref.store_tag == mem._tag

    def test_add_bytes_content(self, mem):
        r = mem.add(b"bytes are fine")
        assert r.memory_id

    def test_add_rejects_non_utf8(self, mem):
        with pytest.raises(VerbatimError) as e:
            mem.add(b"\xff\xfe invalid bytes")
        assert e.value.code == ErrorCode.VALIDATION

    def test_add_rejects_empty(self, mem):
        with pytest.raises(VerbatimError):
            mem.add("")
        with pytest.raises(VerbatimError):
            mem.add(None)
        with pytest.raises(VerbatimError):
            mem.add(42)

    def test_add_content_bound(self, mem):
        with pytest.raises(VerbatimError):
            mem.add("x" * (1_048_577))

    def test_metadata_validation(self, mem):
        assert mem.add("ok", metadata={"a": {"b": [1, 2, None]}}).memory_id
        with pytest.raises(VerbatimError):
            mem.add("bad", metadata="not a mapping")
        with pytest.raises(VerbatimError):
            mem.add("bad", metadata={1: "non-str key"})
        deep = cur = {}
        for _ in range(12):
            cur["d"] = {}
            cur = cur["d"]
        with pytest.raises(VerbatimError):
            mem.add("bad", metadata=deep)

    def test_idempotent_replay(self, mem):
        r1 = mem.add("same call", idempotency_key="op-1")
        r2 = mem.add("same call", idempotency_key="op-1")
        assert r2.replayed is True
        assert r2.memory_id == r1.memory_id
        assert r2.receipt_id == r1.receipt_id

    def test_idempotency_conflict(self, mem):
        mem.add("first", idempotency_key="op-2")
        with pytest.raises(VerbatimError) as e:
            mem.add("different content", idempotency_key="op-2")
        assert e.value.code == ErrorCode.OPERATION_CONFLICT

    def test_content_dedup_replay(self, mem):
        r1 = mem.add("identical bytes land once")
        r2 = mem.add("identical bytes land once")
        assert r2.replayed is True
        assert r2.memory_id == r1.memory_id

    def test_idempotency_survives_reopen(self, path):
        m1 = Memory(path=path, worker="external")
        r1 = m1.add("durable key", idempotency_key="k-durable")
        m1.close()
        m2 = Memory(path=path, worker="external")
        r2 = m2.add("durable key", idempotency_key="k-durable")
        assert r2.replayed and r2.memory_id == r1.memory_id
        m2.close()

    def test_replaces_supersede(self, mem):
        r1 = mem.add("the launch date is friday")
        r2 = mem.add("the launch date is monday", replaces=r1.ref)
        assert r2.memory_id != r1.memory_id
        ref = MemoryRef.parse(r1.ref)
        with mem._store.read() as conn:
            row = conn.execute(
                "SELECT disposition, superseded_by FROM source_state"
                " WHERE source_id = ?",
                (ref.source_id,),
            ).fetchone()
        assert row[0] == "superseded"
        assert row[1] and r2.memory_id in row[1]

    def test_replaces_stale_ref_conflicts(self, mem):
        r1 = mem.add("v1")
        mem.add("v2", replaces=r1.ref)
        with pytest.raises(VerbatimError) as e:
            mem.add("v3", replaces=r1.ref)  # control_version now stale
        assert e.value.code in (
            ErrorCode.STALE_DEPENDENCY,
            ErrorCode.OPERATION_CONFLICT,
            ErrorCode.INVALID_TRANSITION,
        )

    def test_replaces_foreign_ref_denied(self, mem, tmp_path):
        other = Memory(
            path=str(tmp_path / "other.db"), worker="external", user_id="other"
        )
        try:
            r_other = other.add("foreign memory")
            with pytest.raises(VerbatimError) as e:
                mem.add("graft attempt", replaces=r_other.ref)
            assert e.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
        finally:
            other.close()

    def test_effective_at_requires_replaces(self, mem):
        with pytest.raises(VerbatimError):
            mem.add("x", effective_at="2026-01-01T00:00:00Z")
        r1 = mem.add("correct me")
        with pytest.raises(VerbatimError):
            mem.add("fixed", replaces=r1.ref, change="correct",
                    effective_at="2026-01-01T00:00:00Z")

    def test_infer_false_keeps_source_readiness(self, mem):
        r = mem.add("no inference please", infer=False)
        assert r.inference == "not_requested"
        # Source obligations are still declared (deferred when the source
        # jobs seam is unprovisioned) — never stranded.
        assert "source_lexical_ready" in r.readiness


# ---------------------------------------------------------------------------
# wait_ready
# ---------------------------------------------------------------------------


class TestWaitReady:
    def test_wait_ready_default_source_lexical(self, mem):
        r = mem.add("ready check")
        # The source-jobs seam is provisioned in this build: capture
        # enqueued real source_project/source_embed work, and
        # worker="external" means no in-process drain — an external
        # worker (here an inline Ingester bound to the same store) runs
        # the durable jobs through the real coordinator dispatch.
        ingester = Ingester(mem._store, mem._cfg, encoder=mem._encoder)
        report = ingester.drain_report(
            scope=mem._namespace,
            owner="t-ready",
            kinds=(JobKind.SOURCE_PROJECT, JobKind.SOURCE_EMBED),
        )
        assert report["failed"] == 0
        assert report["succeeded"] >= 1
        w = mem.wait_ready(r)
        assert isinstance(w, Readiness)
        assert w.receipt_id == r.receipt_id
        # Drained honestly: source_lexical_ready is fulfilled by the
        # committed projection, never a fabricated "ready".
        assert w.state == "ready"
        assert w.capabilities["source_lexical_ready"] == "succeeded"
        assert w.causal_satisfied is True

    def test_wait_ready_accepts_receipt_forms(self, mem):
        r = mem.add("forms")
        for form in (r, r.receipt_id, {"receipt_id": r.receipt_id}):
            w = mem.wait_ready(form, timeout_ms=50)
            assert w.receipt_id == r.receipt_id

    def test_wait_ready_rejects_bad_forms(self, mem):
        r = mem.add("bad forms")
        with pytest.raises(VerbatimError):
            mem.wait_ready(r.ref)  # MemoryRef is not a receipt
        with pytest.raises(VerbatimError):
            mem.wait_ready(12345)
        with pytest.raises(VerbatimError):
            mem.wait_ready({"no_receipt": "x"})

    def test_wait_ready_foreign_receipt_denied(self, mem, tmp_path):
        other = Memory(path=str(tmp_path / "o.db"), worker="external")
        try:
            r = other.add("other ns")
            with pytest.raises(VerbatimError) as e:
                mem.wait_ready(r.receipt_id, timeout_ms=50)
            assert e.value.code in (
                ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
                ErrorCode.NOT_FOUND_OR_FORBIDDEN,
            )
        finally:
            other.close()

    def test_wait_ready_honest_pending(self, mem):
        """A receipt with genuinely owed work reports pending, not ready."""
        with mem._store.tx() as conn:
            mem._engine.record_obligations(
                conn, "rc_fake_pending", mem._namespace, include_source=True
            )
        w = mem.wait_ready("rc_fake_pending", timeout_ms=50)
        assert w.state == "pending"
        assert w.causal_satisfied is False


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_returns_result(self, mem):
        mem.add("the cellar door code is 4417")
        s = mem.search("cellar door code")
        assert isinstance(s, SearchResult)
        assert s.status in ("ready", "partial", "pending", "blocked", "unavailable")
        assert s.items and s.items[0].quote
        hit = s.items[0]
        assert hit.kind == "source"
        assert hit.memory_id
        assert MemoryRef.parse(hit.ref).source_id == hit.memory_id
        assert hit.object_ref.startswith("vobj1.source.")

    def test_search_honors_limit(self, mem):
        for i in range(6):
            mem.add(f"note number {i} about limes")
        s = mem.search("limes", limit=3)
        assert len(s.items) <= 3

    def test_search_validation(self, mem):
        with pytest.raises(VerbatimError):
            mem.search("")
        with pytest.raises(VerbatimError):
            mem.search(None)
        with pytest.raises(VerbatimError):
            mem.search("x", limit=0)
        with pytest.raises(VerbatimError):
            mem.search("x", consistency="sometimes")
        with pytest.raises(VerbatimError):
            mem.search("x", filters={"drop_table": "sources"})

    def test_search_filters(self, mem):
        r = mem.add("filterable fact")
        s = mem.search("filterable", filters={"source_id": r.memory_id})
        assert all(h.memory_id == r.memory_id for h in s.items)
        s2 = mem.search("filterable", filters={"source_id": "nonexistent"})
        assert not s2.items
        s3 = mem.search("filterable", filters={"kind": "claim"})
        assert not s3.items

    def test_session_consistency_barrier(self, mem):
        r = mem.add("barrier test")
        s = mem.search("barrier test")
        assert s.readiness["receipts"] >= 1
        assert isinstance(s.readiness["causal_satisfied"], bool)

    def test_pending_frontier_reports_pending(self, mem):
        """Unresolved causal work surfaces as pending, not ready."""
        with mem._store.tx() as conn:
            mem._engine.record_obligations(
                conn, "rc_stuck", mem._namespace, include_source=True
            )
        mem._session_receipts.append("rc_stuck")
        s = mem.search("anything", ready_timeout_ms=30)
        assert s.status == "pending"
        assert "rc_stuck" in s.readiness["unresolved"]

    def test_strict_pending_raises(self, mem):
        with mem._store.tx() as conn:
            mem._engine.record_obligations(
                conn, "rc_stuck2", mem._namespace, include_source=True
            )
        mem._session_receipts.append("rc_stuck2")
        with pytest.raises(VerbatimError) as e:
            mem.search("anything", ready_timeout_ms=30, strict=True)
        assert e.value.code == ErrorCode.DEADLINE_EXCEEDED

    def test_eventual_skips_barrier(self, mem):
        with mem._store.tx() as conn:
            mem._engine.record_obligations(
                conn, "rc_stuck3", mem._namespace, include_source=True
            )
        mem._session_receipts.append("rc_stuck3")
        s = mem.search("anything", consistency="eventual")
        assert s.status in ("ready", "partial")

    def test_after_receipt_binding(self, mem):
        r = mem.add("causal write")
        mem2_ns = mem._namespace
        s = mem.search("causal", after=r.receipt_id)
        assert s.status in ("ready", "partial", "pending")
        # A foreign receipt is indistinguishable from unknown.
        with pytest.raises(VerbatimError) as e:
            mem.search("x", after="cr_nonexistent0000000000000000000000")
        assert e.value.code in (
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
            ErrorCode.NOT_FOUND_OR_FORBIDDEN,
        )

    def test_causal_token_reattach(self, path):
        m1 = Memory(path=path, worker="external")
        r = m1.add("cross-process promise")
        s1 = m1.search("promise")
        token = s1.causal_token
        assert token.startswith("mc1.")
        m1.close()
        m2 = Memory(path=path, worker="external")
        s2 = m2.search("promise", after=token)
        assert s2.status in ("ready", "partial", "pending")
        m2.close()

    def test_causal_token_tamper_denied(self, mem):
        with pytest.raises(VerbatimError) as e:
            mem.search("x", after="mc1.ffffffffffffffffffffffff")
        assert e.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    def test_readiness_unavailable_reports(self, mem):
        """No readiness table → honest unavailable, never fabricated."""
        mem._engine = _UnavailableEngine()
        r = mem.add("still works")
        assert r.readiness == {}
        s = mem.search("still works")
        assert s.status == "unavailable"


class _UnavailableEngine:
    available = False


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


class TestInspect:
    def test_inspect_evidence(self, mem):
        r = mem.add("inspect me please")
        i = mem.inspect(r.ref)
        assert i.found is True
        assert i.detail == "evidence"
        assert i.lifecycle["disposition"] == "active"
        assert i.revisions and i.revisions[0]["revision"] == 1
        assert any(e.get("text") for e in i.evidence)  # quote verb bound
        assert "principal_reported" in i.provenance["trust_class"]
        assert OWNER in i.provenance["attribution"]

    def test_inspect_metadata(self, mem):
        r = mem.add("meta only")
        i = mem.inspect(r.ref, detail="metadata")
        assert i.found is True
        assert all("text" not in e for e in i.evidence)

    def test_inspect_bare_source_id(self, mem):
        r = mem.add("bare id inspect")
        i = mem.inspect(r.memory_id)
        assert i.found is True

    def test_inspect_foreign_denied(self, mem, tmp_path):
        other = Memory(path=str(tmp_path / "o.db"), worker="external")
        try:
            r = other.add("not yours")
            with pytest.raises(VerbatimError) as e:
                mem.inspect(r.ref)
            assert e.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
            with pytest.raises(VerbatimError):
                mem.inspect("src_does_not_exist")
        finally:
            other.close()

    def test_inspect_claim_ref_denied_indistinguishably(self, mem):
        # A claim object ref is accepted by the ref grammar; inspecting a
        # nonexistent one denies identically to a missing source.
        import verbatim.memory.controls as C

        claim_ref = C.object_ref_to_string("claim", "abc", 1)
        with pytest.raises(VerbatimError) as e:
            mem.inspect(claim_ref)
        assert e.value.code in (
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
            ErrorCode.NOT_FOUND_OR_FORBIDDEN,
        )
        # ...and bare claim/view refs are never valid mutation targets.
        with pytest.raises(VerbatimError) as e2:
            mem.forget(ref=claim_ref)
        assert e2.value.code == ErrorCode.VALIDATION

    def test_inspect_bad_detail(self, mem):
        r = mem.add("x")
        with pytest.raises(VerbatimError):
            mem.inspect(r.ref, detail="everything")


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


class TestForget:
    def test_forget_ref(self, mem):
        r = mem.add("forget this secret")
        f = mem.forget(ref=r.ref)
        assert isinstance(f, ForgetResult)
        assert f.mode == "operation"
        assert f.mutated is True
        # suppression_state aggregates the engine-owned purge rows
        # (pending→purging→completed); small sources complete synchronously.
        assert f.suppression_state == "completed"
        assert f.closure_state.startswith("completed")
        s = mem.search("forget this secret")
        assert not any(h.memory_id == r.memory_id for h in s.items)

    def test_forget_idempotent_ref(self, mem):
        r = mem.add("forget me twice")
        f1 = mem.forget(ref=r.ref)
        assert f1.mutated
        f2 = mem.forget(ref=r.ref)
        assert f2.mutated and any("replayed" in w for w in f2.warnings)

    def test_forget_stale_ref_conflicts(self, mem):
        r1 = mem.add("replaceable")
        r2 = mem.add("replacement", replaces=r1.ref)
        with pytest.raises(VerbatimError) as e:
            mem.forget(ref=r1.ref)
        assert e.value.code in (
            ErrorCode.STALE_DEPENDENCY,
            ErrorCode.OPERATION_CONFLICT,
            ErrorCode.INVALID_TRANSITION,
        )

    def test_forget_requires_target(self, mem):
        with pytest.raises(VerbatimError):
            mem.forget()
        with pytest.raises(VerbatimError):
            mem.forget(ref="mref1.a.b.c.1.0", query="both")

    def test_query_forget_preview_then_confirm(self, mem):
        r = mem.add("query forgettable item")
        preview = mem.forget(query="forgettable")
        assert preview.mode == "preview"
        assert preview.mutated is False
        assert preview.confirmation_token.startswith("vfc1.")
        assert r.ref in preview.selection
        # Nothing mutated yet.
        assert any(
            h.memory_id == r.memory_id
            for h in mem.search("forgettable").items
        )
        done = mem.forget(confirmation=preview.confirmation_token)
        assert done.mode == "operation" and done.mutated
        assert r.ref in done.selection
        assert not any(
            h.memory_id == r.memory_id
            for h in mem.search("forgettable").items
        )

    def test_confirmation_token_tamper_denied(self, mem):
        # Well-formed envelope, forged MAC → indistinguishable denial.
        with pytest.raises(VerbatimError) as e:
            mem.forget(confirmation="vfc1.aaaa.bbbb")
        assert e.value.code in (
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
            ErrorCode.NOT_FOUND_OR_FORBIDDEN,
        )
        # Malformed envelope → validation, not a crash.
        with pytest.raises(VerbatimError) as e2:
            mem.forget(confirmation="vfc1.nothreeparts")
        assert e2.value.code == ErrorCode.VALIDATION

    def test_confirmation_token_single_use(self, mem):
        r = mem.add("single use token")
        preview = mem.forget(query="single use")
        mem.forget(confirmation=preview.confirmation_token)
        with pytest.raises(VerbatimError) as e:
            mem.forget(confirmation=preview.confirmation_token)
        assert e.value.code == ErrorCode.OPERATION_CONFLICT  # single-use

    def test_forget_idempotency_key(self, mem):
        r = mem.add("keyed forget")
        f1 = mem.forget(ref=r.ref, idempotency_key="fg-1")
        f2 = mem.forget(ref=r.ref, idempotency_key="fg-1")
        assert any("replayed" in w for w in f2.warnings)
        assert f2.receipt_id == f1.receipt_id


# ---------------------------------------------------------------------------
# status / close / lifecycle
# ---------------------------------------------------------------------------


class TestStatusClose:
    def test_status_shape(self, mem):
        st = mem.status()
        assert st.profile == "local_memory"
        assert st.namespace == mem._namespace
        assert st.caller == OWNER
        assert st.store_tag == mem._tag
        assert st.encoder.startswith("hashing:")
        assert st.worker["mode"] == "external"
        assert isinstance(st.capabilities, dict)
        assert st.capabilities["source_lane"] in ("available", "unavailable")

    def test_close_idempotent(self, mem):
        c1 = mem.close()
        c2 = mem.close()
        assert c1.closed and c2.closed

    def test_calls_after_close_fail_typed(self, mem):
        mem.close()
        for fn in (
            lambda: mem.add("x"),
            lambda: mem.search("x"),
            lambda: mem.status(),
            lambda: mem.wait_ready("rc_x"),
        ):
            with pytest.raises(VerbatimError) as e:
                fn()
            assert e.value.code == ErrorCode.INVALID_TRANSITION

    def test_external_worker_reports_external(self, mem):
        st = mem.status()
        assert st.worker["mode"] == "external"
        assert st.worker["running"] is False


class TestManagedWorker:
    def test_managed_lifecycle(self, path):
        m = Memory(path=path)  # default: managed
        try:
            st = m.status()
            assert st.worker["mode"] == "managed"
            assert st.worker["running"] is True
            r = m.add("managed pipeline note")
            assert r.inference in ("queued", "deferred")
        finally:
            c = m.close(timeout_ms=8000)
        assert c.closed is True
        assert c.worker_stopped is True

    def test_managed_workers_share_per_store(self, path):
        m1 = Memory(path=path)
        m2 = Memory(path=path)
        try:
            assert m1._worker_handle is not m2._worker_handle
            assert m1._worker_handle._worker is m2._worker_handle._worker
        finally:
            m1.close()
            m2.close()


class TestForkSafety:
    @pytest.mark.skipif(not hasattr(os, "fork"), reason="posix only")
    def test_forked_child_rejects_calls(self, path):
        m = Memory(path=path, worker="external")
        pid = os.fork()
        if pid == 0:  # child
            try:
                m.add("child write")
                os._exit(1)
            except VerbatimError as e:
                os._exit(0 if e.code == ErrorCode.INVALID_TRANSITION else 2)
            except BaseException:
                os._exit(3)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        # Parent object unaffected.
        assert m.add("parent write").memory_id
        m.close()


class TestUpgradeBackfill:
    """V5-07.13 / §26 P1: opening a store whose sources lack v5 projection
    rows (pre-v5 data, wiped projection, migration) enqueues one deduped
    ``source_backfill`` plan — queued honestly under external workers and
    drained to coverage under the managed worker."""

    def _wipe_projections(self, path):
        import sqlite3

        conn = sqlite3.connect(path)
        conn.execute("DELETE FROM source_lexical_projection")
        conn.execute("DELETE FROM source_fts")
        conn.commit()
        conn.close()

    def _jobs(self, path, kind="source_backfill"):
        import sqlite3

        conn = sqlite3.connect(path)
        rows = list(
            conn.execute("SELECT kind, state FROM jobs WHERE kind = ?", (kind,))
        )
        conn.close()
        return rows

    def test_external_open_enqueues_queued_backfill(self, path):
        with Memory(path=path) as m:
            m.add("legacy memory body")
        self._wipe_projections(path)
        with Memory(path=path, worker="external") as m2:
            jobs = self._jobs(path)
            assert len(jobs) == 1 and jobs[0][1] in ("queued", "leased")
            assert "backfill_enqueue_failed" not in m2._warnings

    def test_reopen_dedups_inflight_plan(self, path):
        with Memory(path=path) as m:
            m.add("legacy memory body")
        self._wipe_projections(path)
        Memory(path=path, worker="external").close()
        Memory(path=path, worker="external").close()
        assert len(self._jobs(path)) == 1

    def test_managed_open_drains_to_coverage(self, path):
        with Memory(path=path) as m:
            m.add("legacy memory about docker")
        self._wipe_projections(path)
        with Memory(path=path) as m2:
            import time

            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                jobs = self._jobs(path)
                if jobs and jobs[0][1] == "succeeded":
                    break
                time.sleep(0.1)
            import sqlite3

            conn = sqlite3.connect(path)
            proj = conn.execute(
                "SELECT COUNT(*) FROM source_lexical_projection"
            ).fetchone()[0]
            cur = list(conn.execute("SELECT done FROM backfill_cursor"))
            conn.close()
            assert proj == 1
            assert cur and cur[0][0] == 1
            assert m2.search("docker").status == "ready"

    def test_projected_store_enqueues_nothing(self, path):
        with Memory(path=path) as m:
            m.add("already projected")
        with Memory(path=path, worker="external"):
            pass
        assert self._jobs(path) == []
