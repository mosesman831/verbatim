"""Native-adapter conformance tests (SPEC_V3 §13 depth 2, §15, §49).

The contract under test: adapters *translate only* — host events become
canonical capture calls and every admission/authorization/screening/
storage decision stays inside the SDK. Coverage:

- ``GenericEventAdapter``: canonical dict → capture roundtrip against a
  real Store (the reference §13.02 event stream drives a full session).
- ``HermesV3Adapter``: provider-hook → SDK delegation proven with a spy
  client (no engine, no Hermes runtime); provenance translation is
  asserted on the envelopes it builds.
- ``AdkMemoryAdapter``: event→envelope translation correctness plus the
  MemoryService-style ops over a fake transport — no live ADK install.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from verbatim.adapters import (
    AdkMemoryAdapter,
    GenericEventAdapter,
    HermesV3Adapter,
    adk_event_to_envelope,
    adk_scope_id,
)
from verbatim.adapters.base import dispatch
from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v3 import EnvelopeKind, TrustClass
from verbatim.governance import create_grant
from verbatim.sdk import CaptureClient
from verbatim.storage import repos_v3
from verbatim.storage.store import Store

PRINCIPAL = "agent:w11"
ISSUER = "ops:test"
SCOPE = "scope:adapter-test"


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "adapters.db"))
    yield s
    s.close()


@pytest.fixture
def client(store):
    return CaptureClient(store, principal_id=PRINCIPAL, host_id="adapter-test")


def _grant(store, scope=SCOPE, principal=PRINCIPAL):
    with store.tx() as conn:
        return create_grant(
            conn,
            scope_id=scope,
            principal_id=principal,
            verbs=["ingest"],
            issuer_id=ISSUER,
        )


def _envelopes(store, scope):
    with store.read() as conn:
        return repos_v3.query(
            conn, "source_envelopes", {"scope_id": scope}, order="receipt_us"
        )


def _jobs(store, kind=None):
    sql = "SELECT job_id, kind, state, input_refs_json FROM jobs"
    params = ()
    if kind is not None:
        sql += " WHERE kind = ?"
        params = (kind,)
    with store.read() as conn:
        cols = [d[0] for d in conn.execute(sql, params).description]
        return [dict(zip(cols, r)) for r in conn.execute(sql, params).fetchall()]


# ---------------------------------------------------------------------
# GenericEventAdapter — canonical event stream → capture (§13.02)
# ---------------------------------------------------------------------


def test_generic_adapter_event_roundtrip(client, store):
    """A dict event stream drives begin → submit → step → outcome → end
    against the real SDK; durable rows prove the roundtrip."""
    _grant(store)
    adapter = GenericEventAdapter(client, issuer=ISSUER)
    adapter.capture_auth(SCOPE)  # host-side provisioning (§11.11)

    sid = adapter.emit(
        {
            "op": "begin_session",
            "principal_id": PRINCIPAL,
            "host_id": "generic",
            "metadata": {"scope_id": SCOPE, "task_id": "task:g1"},
        }
    )["result"]
    src = adapter.emit(
        {
            "op": "submit_source",
            "session_id": sid,
            "scope_id": SCOPE,
            "content": "the flag is --dry-run",
            "declared_type": "agent_note",
        }
    )["result"]
    step_id = adapter.emit(
        {"op": "step", "session_id": sid, "source_id": src, "step": {}}
    )["result"]
    outcome_id = adapter.emit(
        {
            "op": "outcome",
            "session_id": sid,
            "source_id": src,
            "outcome": {"outcome": "success"},
            "receipts": [
                {
                    "checker_id": "pytest",
                    "checker_version": "8.4.2",
                    "invocation_id": "inv-g",
                    "selected_tests": ["t"],
                    "completed": True,
                    "exit_code": 0,
                }
            ],
        }
    )["result"]
    summary = adapter.emit({"op": "end_session", "session_id": sid})["result"]

    assert summary["closed"] is True
    assert summary["episode_build_job_id"] is not None
    envs = _envelopes(store, SCOPE)
    kinds = {e["envelope_kind"] for e in envs}
    assert {"agent_note", "test_result"} <= kinds
    assert len(_jobs(store, "episode_build")) == 1
    with store.read() as conn:
        assert repos_v3.get(conn, "trajectory_steps", {"step_id": step_id})
        assert repos_v3.get(conn, "source_envelopes", {"envelope_id": outcome_id})


def test_generic_adapter_aliases_and_idempotent_auth(client, store):
    _grant(store)
    adapter = GenericEventAdapter(client, issuer=ISSUER)
    aid1 = adapter.capture_auth(SCOPE)
    assert adapter.capture_auth(SCOPE) == aid1  # reused, not reissued
    sid = adapter.emit(
        {
            "type": "session_begin",  # alias → begin_session
            "principal_id": PRINCIPAL,
            "metadata": {"scope_id": SCOPE},
        }
    )["result"]
    assert client.session_state(sid)["closed"] is False
    adapter.emit({"type": "session_end", "session_id": sid})
    assert client.session_state(sid)["closed"] is True


def test_generic_adapter_unknown_op_typed_error(client):
    adapter = GenericEventAdapter(client)
    with pytest.raises(VerbatimError) as exc:
        adapter.emit({"op": "delete_everything"})
    assert exc.value.code == ErrorCode.VALIDATION


def test_dispatch_requires_mapping(client):
    with pytest.raises(VerbatimError) as exc:
        dispatch(client, "not-a-dict")
    assert exc.value.code == ErrorCode.VALIDATION


# ---------------------------------------------------------------------
# HermesV3Adapter — hook → SDK delegation (spy client)
# ---------------------------------------------------------------------


class SpyClient:
    """Records every CaptureClient call the adapter makes — proves the
    adapter translates and delegates rather than reimplementing."""

    def __init__(self):
        self.calls = []
        self.store = None  # setup skipped under provision=False

    def begin_session(self, principal_id, host_id, metadata=None):
        self.calls.append(("begin_session", principal_id, host_id, metadata))
        return "sess:spy"

    def authorize(self, scope_id, granted_by, **kw):
        self.calls.append(("authorize", scope_id, granted_by, kw))
        return "auth:spy"

    def capture_envelope(self, session_id, envelope):
        self.calls.append(("capture_envelope", session_id, envelope))
        return SimpleNamespace(source_id=f"src:{len(self.calls)}", envelope_id="env:x")

    def record_step(self, session_id, source_id, step):
        self.calls.append(("record_step", session_id, source_id, step))
        return "step:spy"

    def record_outcome(self, session_id, source_id, outcome, receipts=None):
        self.calls.append(("record_outcome", session_id, source_id, outcome, receipts))
        return "env:outcome"

    def submit_source(self, *a, **kw):
        self.calls.append(("submit_source", a, kw))
        return "src:submitted"

    def end_session(self, session_id, status="complete"):
        self.calls.append(("end_session", session_id, status))
        return {"closed": True, "session_id": session_id}

    def drain_pending(self, scope_id=None, **kw):
        self.calls.append(("drain_pending", scope_id, kw))
        return 0

    def session_state(self, session_id):
        return {"session_id": session_id, "closed": True}


def _hermes_home(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "memory:\n"
        "  verbatim:\n"
        "    capture:\n"
        "      enabled: true\n"
        "      assistant_context: true\n"
        "      tool_outputs: true\n",
        encoding="utf-8",
    )
    return str(home)


def test_hermes_sync_turn_delegates_with_honest_provenance(tmp_path):
    spy = SpyClient()
    adapter = HermesV3Adapter(spy)
    adapter.initialize(
        "conv-1",
        hermes_home=_hermes_home(tmp_path),
        user_id="user:alice",
        provision=False,
    )
    adapter.sync_turn("deploy it friday", "done — flag is --dry-run")

    captures = [c for c in spy.calls if c[0] == "capture_envelope"]
    assert len(captures) == 2
    user_env, assist_env = captures[0][2], captures[1][2]
    assert user_env.kind == EnvelopeKind.USER_MESSAGE
    assert user_env.trust_class == TrustClass.PRINCIPAL_DIRECT
    assert user_env.actor_principal == "user:alice"
    assert user_env.content == b"deploy it friday"
    assert assist_env.kind == EnvelopeKind.ASSISTANT_MESSAGE
    assert assist_env.trust_class == TrustClass.AGENT_GENERATED
    assert assist_env.actor_principal != "user:alice"  # never user-attributed
    steps = [c for c in spy.calls if c[0] == "record_step"]
    assert len(steps) == 1
    adapter.shutdown()


def test_hermes_bot_author_never_principal_direct(tmp_path):
    spy = SpyClient()
    adapter = HermesV3Adapter(spy)
    adapter.initialize(
        "conv-1",
        hermes_home=_hermes_home(tmp_path),
        user_id="user:alice",
        provision=False,
    )
    adapter.sync_turn(
        "bot text", "", turn_author={"is_bot": True, "id": "bot-9"}
    )
    captures = [c for c in spy.calls if c[0] == "capture_envelope"]
    assert len(captures) == 1
    assert captures[0][2].trust_class == TrustClass.UNKNOWN
    adapter.shutdown()


def test_hermes_subagent_context_captures_nothing(tmp_path):
    spy = SpyClient()
    adapter = HermesV3Adapter(spy)
    adapter.initialize(
        "conv-1",
        hermes_home=_hermes_home(tmp_path),
        user_id="user:alice",
        agent_context="subagent",
        provision=False,
    )
    adapter.sync_turn("hi", "hello")
    assert [c for c in spy.calls if c[0] == "capture_envelope"] == []
    adapter.shutdown()


def test_hermes_tool_result_and_task_end_delegate(tmp_path):
    spy = SpyClient()
    adapter = HermesV3Adapter(spy)
    adapter.initialize(
        "conv-1",
        hermes_home=_hermes_home(tmp_path),
        user_id="user:alice",
        provision=False,
    )
    adapter.on_tool_result("pytest", "3 passed")
    tool_env = [c for c in spy.calls if c[0] == "capture_envelope"][0][2]
    assert tool_env.kind == EnvelopeKind.TOOL_RESULT
    assert tool_env.trust_class == TrustClass.HOST_OBSERVED
    adapter.on_task_end(
        {"outcome": "success"},
        checker={
            "checker_id": "pytest",
            "checker_version": "8.4.2",
            "invocation_id": "inv-h",
            "completed": True,
            "exit_code": 0,
        },
    )
    outcome_calls = [c for c in spy.calls if c[0] == "record_outcome"]
    assert len(outcome_calls) == 1
    assert outcome_calls[0][4][0]["checker_id"] == "pytest"
    summary = adapter.on_session_end()
    assert summary["closed"] is True
    assert ("end_session", "sess:spy", "complete") in spy.calls
    # §40: session end is the drain seam — queued work executes inline.
    assert summary["drained"] == 0
    assert ("drain_pending", None, {"limit": 64}) in spy.calls
    adapter.shutdown()


def test_hermes_session_switch_closes_and_rebinds(tmp_path):
    spy = SpyClient()
    adapter = HermesV3Adapter(spy)
    adapter.initialize(
        "conv-1",
        hermes_home=_hermes_home(tmp_path),
        user_id="user:alice",
        provision=False,
    )
    adapter.on_session_switch("conv-2")
    assert ("end_session", "sess:spy", "complete") in spy.calls
    adapter.sync_turn("next turn", "reply")
    begins = [c for c in spy.calls if c[0] == "begin_session"]
    assert len(begins) == 2  # a fresh session armed on next capture
    adapter.shutdown()


def test_hermes_capability_matrix_declared(tmp_path):
    adapter = HermesV3Adapter(SpyClient())
    m = adapter.capability_matrix()
    assert m["adapter"] == "hermes"
    assert "sync_turn" in m["hooks"]
    assert "user_message" in m["capture_kinds"]


# ---------------------------------------------------------------------
# AdkMemoryAdapter — translation correctness + MemoryService ops
# ---------------------------------------------------------------------


def test_adk_translation_user_turn():
    builders = adk_event_to_envelope(
        {
            "author": "user",
            "id": "e1",
            "timestamp": 1_700_000_000.5,
            "content": {"parts": [{"text": "book the flight"}]},
        },
        scope_id="scope:adk",
        session_id="s1",
        user_principal="user:bob",
    )
    assert len(builders) == 1
    env = builders[0].build()
    assert env.kind == EnvelopeKind.USER_MESSAGE
    assert env.trust_class == TrustClass.PRINCIPAL_DIRECT
    assert env.actor_principal == "user:bob"
    assert env.content == b"book the flight"
    assert env.event_us == 1_700_000_000_500_000


def test_adk_translation_agent_turn_never_human():
    builders = adk_event_to_envelope(
        {"author": "gemini", "content": {"parts": [{"text": "done"}]}},
        scope_id="scope:adk",
        session_id="s1",
    )
    env = builders[0].build()
    assert env.kind == EnvelopeKind.ASSISTANT_MESSAGE
    assert env.trust_class == TrustClass.AGENT_GENERATED
    assert env.actor_principal == "adk:agent"


def test_adk_translation_function_call_and_response():
    call = adk_event_to_envelope(
        {
            "author": "gemini",
            "content": {
                "parts": [{"function_call": {"name": "search", "args": {"q": "x"}}}]
            },
        },
        scope_id="scope:adk",
        session_id="s1",
    )[0].build()
    assert call.kind == EnvelopeKind.TOOL_CALL
    assert call.trust_class == TrustClass.AGENT_GENERATED
    assert call.metadata["tool_name"] == "search"
    resp = adk_event_to_envelope(
        {
            "author": "gemini",
            "content": {
                "parts": [
                    {"function_response": {"name": "search", "response": {"hits": 2}}}
                ]
            },
        },
        scope_id="scope:adk",
        session_id="s1",
    )[0].build()
    assert resp.kind == EnvelopeKind.TOOL_RESULT
    assert resp.trust_class == TrustClass.HOST_OBSERVED


def test_adk_translation_empty_event_yields_nothing():
    assert (
        adk_event_to_envelope(
            {"author": "gemini", "content": {"parts": []}},
            scope_id="scope:adk",
            session_id="s1",
        )
        == []
    )


class FakeTransport:
    """The injected live-host surface: resolves session ids to event
    dicts — stands in for ADK's SessionService in tests."""

    def __init__(self, sessions):
        self._sessions = sessions
        self.fetched = []

    def fetch_session(self, app_name, user_id, session_id):
        self.fetched.append(session_id)
        return self._sessions.get(session_id)


def test_adk_add_session_to_memory_via_transport(client, store):
    session = {
        "id": "sess-adk-1",
        "app_name": "travel",
        "user_id": "user:bob",
        "events": [
            {"author": "user", "id": "e1", "content": {"parts": [{"text": "hi"}]}},
            {
                "author": "gemini",
                "id": "e2",
                "content": {"parts": [{"text": "hello!"}]},
            },
            {
                "author": "gemini",
                "id": "e3",
                "content": {
                    "parts": [
                        {"function_response": {"name": "t", "response": "ok"}}
                    ]
                },
            },
        ],
    }
    transport = FakeTransport({"sess-adk-1": session})
    adapter = AdkMemoryAdapter(client, transport=transport)
    summary = adapter.add_session_to_memory("sess-adk-1")

    assert transport.fetched == ["sess-adk-1"]
    assert summary["closed"] is True
    assert summary["captured_sources"] == 3
    assert summary["episode_build_job_id"] is not None

    scope = summary["scope_id"]
    envs = _envelopes(store, scope)
    kinds = [e["envelope_kind"] for e in envs]
    assert kinds == ["user_message", "assistant_message", "tool_result"]
    trust = {e["envelope_kind"]: e["trust_class"] for e in envs}
    assert trust["user_message"] == TrustClass.PRINCIPAL_DIRECT.value
    assert trust["assistant_message"] == TrustClass.AGENT_GENERATED.value
    assert trust["tool_result"] == TrustClass.HOST_OBSERVED.value
    assert len(_jobs(store, "episode_build")) == 1


def test_adk_add_events_open_then_close(client, store):
    adapter = AdkMemoryAdapter(client)
    result = adapter.add_events_to_memory(
        [{"author": "user", "content": {"parts": [{"text": "one"}]}}],
        app_name="app",
        user_id="user:c",
        session_id="s-open",
    )
    assert result["closed"] is False
    assert _jobs(store, "episode_build") == []
    summary = adapter.end_adk_session("s-open")
    assert summary["closed"] is True
    assert len(_jobs(store, "episode_build")) == 1


def test_adk_add_memory_is_agent_note_not_testimony(client, store):
    adapter = AdkMemoryAdapter(client)
    source_id = adapter.add_memory(
        "user prefers aisle seats", app_name="app", user_id="user:c"
    )
    envs = _envelopes(store, adk_scope_id("app", "user:c", ""))
    assert len(envs) == 1
    assert envs[0]["envelope_kind"] == "agent_note"
    assert envs[0]["trust_class"] == TrustClass.AGENT_GENERATED.value


def test_adk_search_memory_requires_searcher(client):
    adapter = AdkMemoryAdapter(client)
    with pytest.raises(VerbatimError) as exc:
        adapter.search_memory(app_name="app", user_id="u", query="q")
    assert exc.value.code == ErrorCode.CAPABILITY_UNAVAILABLE
    adapter2 = AdkMemoryAdapter(client, searcher=lambda *a, **k: {"items": []})
    assert adapter2.search_memory(app_name="app", user_id="u", query="q") == {
        "items": []
    }


def test_adk_session_id_without_transport_unavailable(client):
    adapter = AdkMemoryAdapter(client)
    with pytest.raises(VerbatimError) as exc:
        adapter.add_session_to_memory("sess-missing")
    assert exc.value.code == ErrorCode.CAPABILITY_UNAVAILABLE


def test_adk_emit_canonical_event(client, store):
    adapter = AdkMemoryAdapter(client)
    out = adapter.emit(
        {
            "author": "user",
            "session_id": "s-emit",
            "app_name": "app",
            "user_id": "user:e",
            "content": {"parts": [{"text": "via emit"}]},
        }
    )
    assert out["ok"] is True
    assert out["result"], "event produced no captured sources"


# ---------------------------------------------------------------------
# Job draining (§40) — adapters expose + drive the drain seam
# ---------------------------------------------------------------------


def test_generic_adapter_drain_event_runs_pipeline(client, store):
    """The canonical ``"drain"`` op is a first-class event: a pure
    event-stream host drains queued work through dispatch."""
    _grant(store)
    adapter = GenericEventAdapter(client, issuer=ISSUER)
    adapter.capture_auth(SCOPE)
    sid = adapter.emit(
        {
            "op": "begin_session",
            "principal_id": PRINCIPAL,
            "metadata": {"scope_id": SCOPE},
        }
    )["result"]
    adapter.emit(
        {
            "op": "capture",
            "session_id": sid,
            "envelope": {
                "kind": "user_message",
                "scope_id": SCOPE,
                "actor_principal": "user:alice",
                "content": "the deploy runbook lives at docs/deploy.md",
                "trust_class": "principal_direct",
            },
        }
    )
    assert [j for j in _jobs(store, "harvest") if j["state"] == "queued"]
    out = adapter.emit({"op": "drain"})
    assert out["ok"] is True and out["result"] >= 1
    states = {j["kind"]: j["state"] for j in _jobs(store)}
    assert states.get("harvest") == "succeeded"
    assert states.get("admit") == "succeeded"


def test_native_adapter_drain_delegates(client, store):
    """``NativeAdapter.drain`` is the host-facing worker capability —
    bounded, explicit, no threads."""
    _grant(store)
    adapter = GenericEventAdapter(client, issuer=ISSUER)
    adapter.capture_auth(SCOPE)
    assert adapter.drain() == 0
    sid = adapter.emit(
        {
            "op": "begin_session",
            "principal_id": PRINCIPAL,
            "metadata": {"scope_id": SCOPE},
        }
    )["result"]
    adapter.emit(
        {
            "op": "capture",
            "session_id": sid,
            "envelope": {
                "kind": "user_message",
                "scope_id": SCOPE,
                "actor_principal": "user:alice",
                "content": "the deploy runbook lives at docs/deploy.md",
                "trust_class": "principal_direct",
            },
        }
    )
    assert adapter.drain(kinds=["harvest"]) == 1
    assert _jobs(store, "harvest")[0]["state"] == "succeeded"
    queued = [j for j in _jobs(store, "admit") if j["state"] == "queued"]
    assert queued
    assert adapter.drain() >= 1


def test_adk_session_close_drains(client, store):
    """``end_adk_session`` is a declared boundary — it drains queued work
    and reports the count (§40)."""
    adapter = AdkMemoryAdapter(client)
    result = adapter.add_events_to_memory(
        [
            {
                "author": "user",
                "content": {"parts": [{"text": "the runbook is at d.md"}]},
            }
        ],
        app_name="app",
        user_id="user:c",
        session_id="s-drain",
    )
    assert result["closed"] is False
    assert [j for j in _jobs(store, "harvest") if j["state"] == "queued"]
    summary = adapter.end_adk_session("s-drain")
    assert summary["closed"] is True
    assert summary["drained"] >= 1
    assert _jobs(store, "harvest")[0]["state"] == "succeeded"


def test_adk_add_session_to_memory_drains(client, store):
    adapter = AdkMemoryAdapter(client)
    session = {
        "id": "sess-d2",
        "app_name": "app",
        "user_id": "user:c",
        "events": [
            {
                "author": "user",
                "content": {"parts": [{"text": "the runbook is at d.md"}]},
            }
        ],
    }
    summary = adapter.add_session_to_memory(session)
    assert summary["closed"] is True
    assert summary["drained"] >= 1
    assert _jobs(store, "episode_build")[0]["state"] == "succeeded"


# ---------------------------------------------------------------------
# F4-21 (V4-05.13, V4-14.07/11, C83) — drain/hook failures are visible,
# durable, and non-fatal — never a silent ``except: pass``
# ---------------------------------------------------------------------


def test_hermes_session_end_drain_failure_is_visible(tmp_path):
    """A drain failure on the session-end seam is non-fatal but NEVER
    silent: the summary carries ``drain_error`` + the durable pending
    count, ``adapter.drain_error`` stays set until a seam succeeds, and a
    diagnostic event is journaled."""
    s = Store.create(str(tmp_path / "h.db"))
    client = CaptureClient(s, principal_id="user:alice", host_id="hermes")
    adapter = HermesV3Adapter(client)
    adapter.initialize(
        "conv-1",
        hermes_home=_hermes_home(tmp_path),
        user_id="user:alice",
    )
    adapter.sync_turn("I prefer aisle seats", "noted")
    assert [j for j in _jobs(s, "harvest") if j["state"] == "queued"]

    def boom(*a, **k):
        raise VerbatimError(ErrorCode.STORE_BUSY, "drain offline")

    real = client.drain_pending
    client.drain_pending = boom
    try:
        summary = adapter.on_session_end([])
    finally:
        client.drain_pending = real

    # Session close still committed; the drain failure is reported, and
    # durable obligations are visibly still owed.
    assert summary["closed"] is True
    assert "drain_error" in summary
    assert "STORE_BUSY" in summary["drain_error"]
    assert summary["pending"] >= 1
    assert adapter.drain_error is not None
    assert adapter.drain_error["code"] == ErrorCode.STORE_BUSY.value
    with s.read() as conn:
        rows = conn.execute(
            "SELECT kind, payload_json FROM events"
            " WHERE kind='adapter_hook_failed'"
        ).fetchall()
    assert rows, "drain failure was not journaled"
    assert "drain" in rows[0][1]

    # Recovery: the same seam drains and clears the error surface.
    summary2 = adapter.on_session_end([])
    assert summary2.get("drain_error") is None
    assert adapter.drain_error is None
    assert summary2["pending"] == 0
    s.close()


def test_hermes_end_session_failure_keeps_handle_and_reports(tmp_path):
    """A session-close failure reports ``end_session_error`` +
    ``closed: False`` and KEEPS the session handle so a later seam can
    retry — the drain still runs."""
    s = Store.create(str(tmp_path / "h.db"))
    client = CaptureClient(s, principal_id="user:alice", host_id="hermes")
    adapter = HermesV3Adapter(client)
    adapter.initialize(
        "conv-1", hermes_home=_hermes_home(tmp_path), user_id="user:alice"
    )
    sdk_session = adapter._sdk_session

    def boom(session_id, status="complete"):
        raise VerbatimError(ErrorCode.STORE_BUSY, "close failed")

    real = client.end_session
    client.end_session = boom
    try:
        summary = adapter.on_session_end([])
    finally:
        client.end_session = real

    assert summary["closed"] is False
    assert "end_session_error" in summary
    assert "drained" in summary  # the drain seam still ran
    assert adapter._sdk_session == sdk_session  # handle kept for retry
    assert adapter.last_hook_error is not None
    assert adapter.last_hook_error["phase"] == "end_session"
    s.close()


def test_hermes_capture_failure_records_and_declines(tmp_path):
    """A non-denial capture failure is recorded on ``last_hook_error``
    (and journaled) while the hook still declines to the host; policy
    denials stay silent per SPEC §43."""
    s = Store.create(str(tmp_path / "h.db"))
    client = CaptureClient(s, principal_id="user:alice", host_id="hermes")
    adapter = HermesV3Adapter(client)
    adapter.initialize(
        "conv-1", hermes_home=_hermes_home(tmp_path), user_id="user:alice"
    )

    def boom(*a, **k):
        raise VerbatimError(ErrorCode.STORE_BUSY, "write failed")

    real = client.capture_envelope
    client.capture_envelope = boom
    try:
        adapter.sync_turn("I prefer aisle seats", "noted")  # no raise
    finally:
        client.capture_envelope = real
    assert adapter.last_hook_error is not None
    assert adapter.last_hook_error["code"] == ErrorCode.STORE_BUSY.value
    assert adapter.last_hook_error["phase"] == "sync_turn"
    s.close()


def test_hermes_policy_denial_still_declines_silently(tmp_path):
    """SPEC §43 unchanged: a capture denial (e.g. capture disabled) is a
    decline, not a recorded failure."""
    spy = SpyClient()
    adapter = HermesV3Adapter(spy)

    def deny(*a, **k):
        raise VerbatimError(ErrorCode.CAPTURE_DISABLED, "capture off")

    spy.capture_envelope = deny
    adapter.initialize(
        "conv-1",
        hermes_home=_hermes_home(tmp_path),
        user_id="user:alice",
        provision=False,
    )
    adapter.sync_turn("hi", "hello")
    assert adapter.last_hook_error is None
    adapter.shutdown()
