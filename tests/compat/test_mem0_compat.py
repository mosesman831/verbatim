"""Mem0 compat-shim tests (SPEC_V5 §18.2, V5-18.06..13).

Two layers:

* ``TestShimWithoutFacade`` — pure shim behavior (validation, option
  rejection, alias/ref translation). These never bind a facade and must
  pass even while ``verbatim.memory.facade`` is mid-flight.
* ``TestWithFacade`` — real store round trips. Skipped cleanly when the
  facade ``Memory`` is not yet importable/constructible.
"""

from __future__ import annotations

import time

import pytest

from verbatim.compat import mem0
from verbatim.core.types import ErrorCode, VerbatimError

FACADE_OK = mem0.facade_available()
FACADE_REASON = (
    "verbatim.memory.facade.Memory not importable in this build "
    "(v5 facade mid-flight) — compat facade tests skipped"
)
needs_facade = pytest.mark.skipif(not FACADE_OK, reason=FACADE_REASON)

READY_STATES = {
    "ready",
    "partial",
    "pending",
    "blocked",
    "unavailable",
    # The V7 structural verdict (verdict_v2, V7-11.01b) emits
    # "insufficient" for a completed search with zero eligible
    # candidates — a real terminal state the facade surfaces verbatim,
    # distinct from "ready" with an empty list.
    "insufficient",
}


_MIDFLIGHT = (mem0.FacadeUnavailableError, NotImplementedError, ImportError)


@pytest.fixture
def mem(tmp_path):
    if not FACADE_OK:
        pytest.skip(FACADE_REASON)
    try:
        instance = mem0.Memory(path=str(tmp_path / "store.vdb"))
    except _MIDFLIGHT as exc:
        pytest.skip(f"facade present but not constructible yet: {exc}")
    # Functional probe: the facade class may resolve while write/read
    # paths are still mid-flight (missing deps, unprovisioned workers).
    # The shim always attaches metadata, so the probe exercises the real
    # metadata path too. Scoped to a probe namespace — never collides.
    try:
        instance.add("__facade_probe__", user_id="__probe__")
        instance.search("__facade_probe__", user_id="__probe__")
    except _MIDFLIGHT as exc:
        instance.close()
        pytest.skip(f"facade constructible but write/read mid-flight: {exc}")
    try:
        yield instance
    finally:
        instance.close()


def wait_search(mem, query, timeout=15.0, **kwargs):
    """Poll a compat search until the verbatim status is terminal.

    ``partial`` with results is a delivered (degraded-lane) answer —
    terminal for our purposes. ``partial``/``pending`` with no results
    may still be warming, so we keep polling.
    """
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = mem.search(query, **kwargs)
        assert last["verbatim_status"] in READY_STATES
        if last["verbatim_status"] in ("ready", "blocked", "unavailable"):
            return last
        if last["verbatim_status"] == "partial" and last["results"]:
            return last
        time.sleep(0.15)
    return last


def wait_get_all(mem, timeout=15.0, **kwargs):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = mem.get_all(**kwargs)
        assert last["verbatim_status"] in READY_STATES
        if last["verbatim_status"] in ("ready", "blocked", "unavailable"):
            return last
        if last["verbatim_status"] == "partial" and last["results"]:
            return last
        time.sleep(0.15)
    return last


# ---------------------------------------------------------------------------
# Layer 1 — no facade required: validation, option rejection, translation.
# ---------------------------------------------------------------------------


class TestShimWithoutFacade:
    def test_module_importable_and_pinned(self):
        assert mem0.MEM0_SURFACE_PIN
        assert "mem0" in mem0.MEM0_SURFACE_PIN.lower()
        assert mem0.MEM0_COMPAT_SCHEMA == "verbatim.compat.mem0/v1"
        assert issubclass(mem0.Mem0CompatError, VerbatimError)
        for cls in (
            mem0.UnsupportedOptionError,
            mem0.ConfirmationRequiredError,
            mem0.NamespaceBindingError,
            mem0.MemoryNotFoundError,
            mem0.MemoryConflictError,
            mem0.FacadeUnavailableError,
        ):
            assert issubclass(cls, mem0.Mem0CompatError)

    def test_errors_carry_typed_codes(self):
        err = mem0.UnsupportedOptionError("nope")
        assert err.code is ErrorCode.VALIDATION
        assert err.to_dict()["code"] == "VALIDATION"
        assert mem0.MemoryConflictError("x").code is ErrorCode.OPERATION_CONFLICT
        assert (
            mem0.MemoryNotFoundError("x").code
            is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
        )
        assert (
            mem0.FacadeUnavailableError("x").code
            is ErrorCode.CAPABILITY_UNAVAILABLE
        )

    def test_constructor_rejects_mem0_provider_config(self):
        with pytest.raises(mem0.UnsupportedOptionError):
            mem0.Memory(
                config={"vector_store": {"provider": "qdrant"}, "llm": {}}
            )
        with pytest.raises(mem0.UnsupportedOptionError):
            mem0.Memory(config={"embedder": {"provider": "openai"}})

    def test_constructor_accepts_verbatim_config(self):
        m = mem0.Memory(config={"path": None, "user_id": "cfg-user"})
        assert m._default_scope == ("cfg-user", None, None)
        m.close()

    def test_unknown_kwargs_raise_typed(self, tmp_path):
        m = mem0.Memory(path=str(tmp_path))
        with pytest.raises(mem0.UnsupportedOptionError, match="bogus"):
            m.search("q", bogus_option=1)
        with pytest.raises(mem0.UnsupportedOptionError, match="bogus"):
            m.add("x", bogus_option=1)
        with pytest.raises(mem0.UnsupportedOptionError):
            m.get_all(no_such=1)
        with pytest.raises(mem0.UnsupportedOptionError):
            mem0.Memory(path=str(tmp_path), bogus=1)
        m.close()

    def test_unsupported_add_options(self, tmp_path):
        m = mem0.Memory(path=str(tmp_path))
        for kw in (
            {"memory_type": "procedural_memory"},
            {"prompt": "extract facts"},
            {"llm": object()},
            {"custom_instructions": "be terse"},
            {"app_id": "app1"},
            {"expiration_date": "2030-01-01"},
            {"output_format": "v1.0"},
        ):
            with pytest.raises(mem0.UnsupportedOptionError):
                m.add("some text", **kw)
        m.close()

    def test_search_unsupported_and_versions(self, tmp_path):
        m = mem0.Memory(path=str(tmp_path))
        with pytest.raises(mem0.UnsupportedOptionError):
            m.search("q", rerank=True)
        with pytest.raises(mem0.UnsupportedOptionError):
            m.search("q", version="v1")
        with pytest.raises(mem0.UnsupportedOptionError):
            m.get_all(version="v1")
        m.close()

    def test_filter_operators(self):
        # OR/NOT combinators exceed the bounded conjunction (V5-06.13).
        with pytest.raises(mem0.UnsupportedOptionError):
            mem0._translate_filters({"OR": [{"a": 1}, {"b": 2}]}, "search")
        with pytest.raises(mem0.UnsupportedOptionError):
            mem0._translate_filters({"NOT": {"a": 1}}, "search")
        with pytest.raises(mem0.UnsupportedOptionError):
            mem0._translate_filters({"kind": {"icontains": "x"}}, "search")
        # Keys outside the facade's allowlist reject — silently dropping
        # a filter would broaden the result set.
        with pytest.raises(mem0.UnsupportedOptionError):
            mem0._translate_filters({"topic": "deploy"}, "search")
        with pytest.raises(mem0.UnsupportedOptionError):
            mem0._translate_filters({"updated_at": {"gte": "x"}}, "search")
        # Scope ids cannot be multi-valued (would broaden the namespace).
        with pytest.raises(mem0.UnsupportedOptionError):
            mem0._translate_filters({"user_id": {"in": ["a", "b"]}}, "search")
        clauses, scope, notes = mem0._translate_filters(
            {
                "AND": [
                    {"kind": {"in": ["a", "b"]}},
                    {"lifecycle": "active"},
                    {"created_at": {"gte": "2026-01-01"}},
                ],
                "user_id": "alice",
            },
            "search",
        )
        assert scope == {"user_id": "alice"}
        assert clauses["kind"] == ["a", "b"]
        assert clauses["lifecycle"] == "active"
        assert clauses["created_after"] == "2026-01-01"
        # 'gte' → strict facade bound — the approximation is surfaced.
        assert any("created_after" in n for n in notes)

    def test_scope_conflict_arg_vs_filters(self, tmp_path):
        m = mem0.Memory(path=str(tmp_path))
        with pytest.raises(mem0.Mem0CompatError, match="conflicting user_id"):
            m.search("q", user_id="a", filters={"user_id": "b"})
        m.close()

    def test_delete_all_requires_confirm(self, tmp_path):
        m = mem0.Memory(path=str(tmp_path))
        with pytest.raises(mem0.ConfirmationRequiredError):
            m.delete_all(user_id="victim")
        m.close()

    def test_delete_all_requires_scope(self, tmp_path):
        m = mem0.Memory(path=str(tmp_path))
        with pytest.raises(mem0.Mem0CompatError):
            m.delete_all(confirm=True)
        m.close()

    def test_delete_all_query_and_limit_validation(self, tmp_path):
        m = mem0.Memory(path=str(tmp_path))
        with pytest.raises(mem0.Mem0CompatError):
            m.delete_all(user_id="x", confirm=True, query=123)
        with pytest.raises(mem0.Mem0CompatError):
            m.delete_all(user_id="x", confirm=True, query="   ")
        with pytest.raises(mem0.Mem0CompatError):
            m.delete_all(user_id="x", confirm=True, limit=0)
        m.close()

    def test_client_platform_args_rejected(self, tmp_path):
        with pytest.raises(mem0.UnsupportedOptionError, match="api_key"):
            mem0.MemoryClient(path=str(tmp_path), api_key="sk-x")
        with pytest.raises(mem0.UnsupportedOptionError, match="org_id"):
            mem0.MemoryClient(path=str(tmp_path), org_id="o1")
        with pytest.raises(mem0.UnsupportedOptionError, match="host"):
            mem0.MemoryClient(path=str(tmp_path), host="https://api.mem0.ai")
        client = mem0.MemoryClient(path=str(tmp_path))
        for method in ("users", "feedback", "get_project", "update_project"):
            with pytest.raises(mem0.UnsupportedOptionError):
                getattr(client, method)()
        with pytest.raises(mem0.UnsupportedOptionError):
            client.reset()
        client.close()

    def test_message_normalization(self):
        assert mem0._normalize_messages("hello", None) == [
            {"role": "user", "content": "hello"}
        ]
        msgs = mem0._normalize_messages(
            [{"role": "user", "content": "c1"},
             {"role": "assistant", "content": "c2"}],
            None,
        )
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        parts = mem0._normalize_messages(
            [{"role": "user", "content": [{"type": "text", "text": "t"}]}], None
        )
        assert parts[0]["content"] == "t"
        with pytest.raises(mem0.UnsupportedOptionError):
            mem0._normalize_messages(
                [{"role": "user", "content": [{"type": "image_url"}]}], None
            )
        with pytest.raises(mem0.Mem0CompatError):
            mem0._normalize_messages(None, None)
        with pytest.raises(mem0.Mem0CompatError):
            mem0._normalize_messages("x", "y")

    def test_alias_composition_roundtrip(self):
        # Bare user ids pass through verbatim (facade's own label
        # convention); multi-id tuples compose one deterministic label.
        assert mem0._compose_alias("alice", None, None) == "alice"
        assert mem0._compose_alias("a@b.com", None, None) == "a@b.com"
        assert mem0._compose_alias(None, None, None) is None
        composite = mem0._compose_alias("alice", "bot", "run1")
        assert mem0._ids_from_alias(composite) == {
            "user_id": "alice",
            "agent_id": "bot",
            "run_id": "run1",
        }
        # ':' inside components is escaped — no scope confusion.
        tricky = mem0._compose_alias("al:x", "b", None)
        assert mem0._ids_from_alias(tricky) == {
            "user_id": "al:x",
            "agent_id": "b",
        }
        # Opaque namespaces / bare labels never decode to a guessed id.
        assert mem0._ids_from_alias("ns_abc123") == {}
        assert mem0._ids_from_alias("alice") == {}
        # Agent-only / run-only scopes are distinct namespaces.
        assert mem0._compose_alias(None, "bot", None) != mem0._compose_alias(
            None, None, "bot"
        )
        assert mem0._compose_alias("u", "a", None) != mem0._compose_alias(
            "u", None, None
        )

    def test_alias_label_validation(self):
        with pytest.raises(mem0.Mem0CompatError):
            mem0._validate_alias_label("bad\x00label")
        with pytest.raises(mem0.Mem0CompatError):
            mem0._validate_alias_label("x" * 300)
        with pytest.raises(mem0.Mem0CompatError):
            mem0._validate_alias_label("   ")
        assert mem0._validate_alias_label("alice") == "alice"

    def test_ref_parsing_fallback(self):
        fields = mem0._parse_ref_fields("mref1.st.ns1.src9.2.7")
        assert fields == ("st", "ns1", "src9", 2, 7)
        assert mem0._parse_ref_fields("plain-id") is None
        assert mem0._parse_ref_fields("mref1.too.short") is None
        assert mem0._ids_from_ref("mref1.st.u:al:a:bt.src.1.1") == {
            "user_id": "al",
            "agent_id": "bt",
        }

    def test_limit_and_threshold_validation(self, tmp_path):
        m = mem0.Memory(path=str(tmp_path))
        with pytest.raises(mem0.Mem0CompatError):
            m.search("q", limit=0)
        with pytest.raises(mem0.Mem0CompatError):
            m.search("q", limit=5, top_k=6)
        with pytest.raises(mem0.Mem0CompatError):
            m.search("q", threshold=1.5)
        m.close()


# ---------------------------------------------------------------------------
# Layer 2 — real facade + store. Skipped while the facade is mid-flight.
# ---------------------------------------------------------------------------


@needs_facade
class TestWithFacade:
    def test_add_search_roundtrip(self, mem):
        res = mem.add(
            "The deploy command is quixotic-zebra-7.", user_id="alice"
        )
        assert res["results"], res
        item = res["results"][0]
        assert item["event"] == "ADD"
        assert item["id"]
        assert item["verbatim_ref"]
        assert item["user_id"] == "alice"
        assert res["verbatim_status"]

        out = wait_search(mem, "quixotic-zebra-7", user_id="alice")
        assert out["verbatim_status"] in READY_STATES
        assert isinstance(out["results"], list)
        if out["verbatim_status"] == "ready":
            assert any(
                "quixotic-zebra-7" in (h.get("memory") or "")
                for h in out["results"]
            ), out
        else:
            # Honest pending/blocked — surfaced, not faked (V5-18.10).
            assert out["verbatim_status"] in ("pending", "partial", "blocked")

    def test_user_id_isolation(self, mem):
        mem.add("alice-secret-marker-41", user_id="alice")
        wait_search(mem, "alice-secret-marker-41", user_id="alice")
        other = mem.search("alice-secret-marker-41", user_id="bob")
        assert all(
            "alice-secret-marker-41" not in (h.get("memory") or "")
            for h in other["results"]
        )
        if other["verbatim_status"] == "ready":
            assert other["results"] == []

    def test_message_list_one_result_per_message(self, mem):
        res = mem.add(
            [
                {"role": "user", "content": "list-marker-a"},
                {"role": "assistant", "content": "list-marker-b"},
            ],
            user_id="carol",
            infer=False,
        )
        assert len(res["results"]) == 2
        roles = {r["metadata"].get("verbatim_role") for r in res["results"]}
        assert roles == {"user", "assistant"}
        for r in res["results"]:
            assert r["event"] == "ADD" and r["verbatim_ref"]

    def test_infer_false_source_only(self, mem):
        res = mem.add("inferless-marker-9", user_id="dan", infer=False)
        item = res["results"][0]
        assert item["verbatim_inference"] in (
            "not_requested",
            "deferred",
            "unavailable",
        )
        out = wait_search(mem, "inferless-marker-9", user_id="dan")
        if out["verbatim_status"] == "ready":
            assert any(
                "inferless-marker-9" in (h.get("memory") or "")
                for h in out["results"]
            )

    def test_update_cas_and_stale_conflict(self, mem):
        added = mem.add("deploy tool is starship-v1", user_id="ed")
        mid = added["results"][0]["id"]
        stale_ref = added["results"][0]["verbatim_ref"]

        upd = mem.update(mid, "deploy tool is starship-v2")
        assert upd["message"] == "Memory updated successfully!"
        assert upd["id"] == mid  # same logical memory, new revision
        assert upd["verbatim_ref"]

        # Replaying the pre-update ref must hit the CAS fence — never a
        # blind last-write-wins (V5-18.11).
        with pytest.raises(VerbatimError) as ei:
            mem.update(stale_ref, "deploy tool is starship-v3")
        assert ei.value.code in {
            ErrorCode.OPERATION_CONFLICT,
            ErrorCode.STALE_PROPOSAL,
            ErrorCode.STALE_EPOCH,
            ErrorCode.STALE_DEPENDENCY,
            ErrorCode.STORE_CONFLICT,
            ErrorCode.INVALID_TRANSITION,
        }

    def test_delete_then_get_none(self, mem):
        added = mem.add("delete-me-marker", user_id="fay")
        mid = added["results"][0]["id"]
        out = mem.delete(mid)
        assert "message" in out
        assert out["verbatim_status"] in ("deleted", "preview")
        got = mem.get(mid)
        assert got is None or got.get("verbatim_lifecycle") in (
            "suppressed",
            "retracted",
            "erased",
        )

    def test_delete_all_preview_confirm(self, mem):
        mem.add("victim-marker-1", user_id="victim")
        mem.add("victim-marker-2", user_id="victim")
        with pytest.raises(mem0.ConfirmationRequiredError):
            mem.delete_all(user_id="victim")

        # The query path drives the facade's real two-phase flow:
        # forget(query=…) preview → token → forget(confirmation=token).
        out = mem.delete_all(
            user_id="victim", query="victim-marker", confirm=True
        )
        assert out["message"] == "Memories deleted successfully!"
        assert out["verbatim_status"] == "deleted"
        assert isinstance(out["verbatim_deleted"], int)
        assert out["verbatim_deleted"] >= 0

        # Mem0's plain match-all call enumerates the bound namespace and
        # closes per-ref. If enumeration is pending/blocked/unavailable
        # the shim reports "not performed"; a partial/degraded
        # enumeration still executes but must report honestly — never a
        # fake "all deleted" (V5-18.10/13).
        mem.add("victim-marker-3", user_id="victim2")
        out2 = mem.delete_all(user_id="victim2", confirm=True)
        if out2["verbatim_status"] == "deleted":
            assert out2["message"] == "Memories deleted successfully!"
            assert out2["verbatim_deleted"] >= 0
        elif out2.get("verbatim_mutated") is False:
            assert "not performed" in out2["message"].lower()
        else:
            # Executed under partial enumeration coverage — the honest
            # report says "incomplete", carries the real status, and
            # names what it actually closed.
            assert out2["verbatim_status"] == "partial"
            assert "incomplete" in out2["message"].lower()
            assert out2["verbatim_enumeration_status"] in (
                "partial",
                "degraded",
            )
            assert out2["verbatim_deleted"] >= 0
            assert isinstance(out2["verbatim_audit_remnants"], list)

    def test_get_all_honest_paging(self, mem):
        for i in range(3):
            mem.add(f"page-marker-{i}", user_id="gina")
        res = wait_get_all(mem, user_id="gina", limit=2)
        # Never a bare list — status travels with results (V5-18.10).
        assert "verbatim_status" in res
        assert res["verbatim_status"] in READY_STATES
        assert isinstance(res["results"], list)
        assert len(res["results"]) <= 2
        page2 = mem.get_all(user_id="gina", page=2, page_size=2)
        assert page2["verbatim_paging"]["mode"] == "client_side_slice"
        assert isinstance(page2["results"], list)

    def test_history_revision_chain(self, mem):
        added = mem.add("history-marker v1", user_id="henry")
        mid = added["results"][0]["id"]
        mem.update(mid, "history-marker v2")
        hist = mem.history(mid)
        assert isinstance(hist, list)
        assert hist, "inspect() returned no revision chain"
        assert hist[0]["event"] == "ADD"
        assert all("verbatim_disposition" in h or "verbatim_revision" in h
                   for h in hist)
        assert mem.history("nonexistent-id") == []

    def test_chain_resolution_after_update(self, mem):
        """verbatim mints a new source per update — a Mem0 logical memory
        is the supersession chain, and bare-id get/delete must follow
        ``superseded_by`` to the live head (V5-14.10/18.11)."""
        added = mem.add("chain-marker v1", user_id="ivy")
        mid = added["results"][0]["id"]
        upd = mem.update(mid, "chain-marker v2")
        new_sid = upd.get("verbatim_current_id")
        assert upd["id"] == mid
        assert new_sid and new_sid != mid
        assert upd["verbatim_supersedes"] == mid

        # get(old id) resolves to the live head — current text, not the
        # superseded predecessor's payload.
        got = mem.get(mid)
        assert got is not None
        assert "chain-marker v2" in (got.get("memory") or "")
        assert got.get("verbatim_current_id") == new_sid

        # search/get_all surface only the live head — the superseded
        # predecessor is dropped with an honest count.
        out = wait_search(mem, "chain-marker", user_id="ivy")
        if out["verbatim_status"] in ("ready", "partial"):
            lives = [h for h in out["results"]
                     if "chain-marker" in (h.get("memory") or "")]
            assert len(lives) <= 1, out
            if lives:
                assert "v2" in lives[0]["memory"]
                assert lives[0].get("verbatim_lifecycle") in (
                    None, "active"
                )
            assert out.get("verbatim_dropped_noncurrent", 0) >= 0

        # delete(old id) closes the live head; the superseded link
        # remains as retained audit history, reported not hidden.
        out = mem.delete(mid)
        assert out["verbatim_status"] == "deleted", out
        assert out.get("verbatim_current_id") == new_sid
        assert mid in out.get("verbatim_chain_links", [])
        assert mem.get(mid) is None
        # The chain's audit trail is still inspectable via history().
        hist = mem.history(mid)
        assert any(h.get("event") == "ADD" for h in hist)
        assert hist[-1].get("event") == "DELETE"

    def test_pending_state_not_fake_empty(self, mem):
        # Whatever the facade does, the dict must carry the real state.
        res = mem.search("anything", user_id="iris")
        assert res["verbatim_status"] in READY_STATES
        assert "results" in res
        assert "verbatim_readiness" in res or "verbatim_coverage" in res
