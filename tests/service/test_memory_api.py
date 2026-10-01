"""V5 consumer HTTP surface tests (docs/v6_contracts.md §4, SPEC_V6 §05).

Real ``Memory`` on a temp store, real threaded loopback server — the
tests talk to the wire through ``create_memory_app`` and the mounted
``create_api`` path, never to stubs.
"""

from __future__ import annotations

import http.client
import json

import pytest

from verbatim.api import open_store
from verbatim.config import config_from_mapping
from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.host import LocalHost
from verbatim.service import TokenCredential, create_server
from verbatim.service.auth import scope_mismatch
from verbatim.service.httpd import HttpConfig, RunningServer
from verbatim.service.memory_api import MemoryApp, create_memory_app

TOKEN = "alice-secret-token"
OTHER = "mallory-secret-token"
READONLY = "reader-secret-token"
VERBS = ["read", "quote", "derive", "ingest", "review", "admin"]


def _creds():
    return [
        TokenCredential.mint(TOKEN, "alice", VERBS),
        TokenCredential.mint(OTHER, "mallory", VERBS),
        TokenCredential.mint(READONLY, "alice", ["read"]),
    ]


def _start(httpd):
    srv = RunningServer(httpd).start()
    return srv


@pytest.fixture()
def server(tmp_path):
    httpd = create_memory_app(
        str(tmp_path / "mem.db"), user_id="alice", tokens=_creds()
    )
    srv = _start(httpd)
    yield srv
    srv.stop()
    httpd.application.memory.close()


def _call(srv, method, path, body=None, token=TOKEN, raw=None,
          ctype="application/json"):
    conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=30)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = raw
    if body is not None:
        data = json.dumps(body)
    if data is not None:
        headers["Content-Type"] = ctype
    conn.request(method, path, body=data, headers=headers)
    resp = conn.getresponse()
    payload = resp.read()
    conn.close()
    try:
        decoded = json.loads(payload) if payload else {}
    except ValueError:
        decoded = {"_raw": payload}
    return resp.status, decoded


def _add(srv, text, **kw):
    body = {"text": text}
    body.update(kw)
    s, b = _call(srv, "POST", "/v2/memory/add", body)
    assert s == 200, b
    return b


def _wait_ready(srv, receipt_id, timeout_ms=20000):
    s, b = _call(
        srv, "GET", f"/v2/memory/readiness/{receipt_id}?timeout_ms={timeout_ms}"
    )
    assert s == 200, b
    return b


# ---------------------------------------------------------------------------
# contract: bind-time + auth
# ---------------------------------------------------------------------------


def test_import_is_side_effect_free():
    """The factory module imports without binding a socket."""
    import verbatim.service.memory_api as mod

    assert callable(mod.create_memory_app)
    assert issubclass(mod.MemoryApp, object)


def test_no_usable_credential_is_config_invalid(tmp_path):
    """Every token scoped elsewhere → CONFIG_INVALID at bind time (§4)."""
    with pytest.raises(VerbatimError) as ei:
        create_memory_app(
            str(tmp_path / "m.db"),
            user_id="alice",
            tokens=[TokenCredential.mint(OTHER, "mallory", VERBS)],
        )
    assert ei.value.code == ErrorCode.CONFIG_INVALID


def test_scope_pinned_credential_rejected():
    """A credential carrying a /v1 scope pin cannot bind this surface."""
    pinned = TokenCredential.mint(
        "pinned-tok", "alice", VERBS, scope={"workspace_id": "w1"}
    )
    assert scope_mismatch(pinned, "alice") == "scope"
    assert scope_mismatch(
        TokenCredential.mint("t", "bob", VERBS), "alice"
    ) == "principal_id"
    assert scope_mismatch(TokenCredential.mint("t2", "alice", VERBS), "alice") is None


def test_unauthenticated_401(server):
    s, b = _call(server, "GET", "/v2/memory/status", token=None)
    assert s == 401
    s, b = _call(server, "POST", "/v2/memory/add", {"text": "x"}, token=None)
    assert s == 401
    s, b = _call(server, "GET", "/v2/memory/status", token="wrong")
    assert s == 401


def test_wrong_scope_token_403(server):
    """A valid token bound to another principal → flat 403 (§4)."""
    for method, path, body in (
        ("GET", "/v2/memory/status", None),
        ("POST", "/v2/memory/add", {"text": "mallory was here"}),
        ("POST", "/v2/memory/search", {"query": "x"}),
        ("POST", "/v2/memory/forget", {"ref": "x"}),
    ):
        s, b = _call(server, method, path, body, token=OTHER)
        assert s == 403, (path, s, b)
        assert b["code"] == "FORBIDDEN" and b["error"] and "retryable" in b


def test_verb_ceiling_enforced(server):
    """The token's verb ceiling narrows the bound owner's authority —
    a read-only credential cannot mutate through this surface."""
    s, b = _call(server, "POST", "/v2/memory/add", {"text": "nope"}, token=READONLY)
    assert s == 403 and b["code"] == "FORBIDDEN"
    s, b = _call(server, "POST", "/v2/memory/forget", {"ref": "x"}, token=READONLY)
    assert s == 403
    # read-verb routes still serve the read-only credential.
    s, b = _call(server, "GET", "/v2/memory/status", token=READONLY)
    assert s == 200
    s, b = _call(server, "POST", "/v2/memory/search", {"query": "x"}, token=READONLY)
    assert s == 200


# ---------------------------------------------------------------------------
# contract: validation + honest errors
# ---------------------------------------------------------------------------


def test_invalid_bodies_are_typed_400(server):
    for body in (
        {},
        {"text": 42},
        {"text": ""},
        {"text": "ok", "infer": "yes"},
        {"text": "ok", "metadata": [1]},
    ):
        s, b = _call(server, "POST", "/v2/memory/add", body)
        assert s == 400, (body, s, b)
        assert b["code"] == "VALIDATION" and b["error"]
        assert b["retryable"] is False


def test_payload_cannot_mint_identity(server):
    """user_id/namespace/identity fields are unknown → 400, never bound."""
    for key in ("user_id", "namespace", "principal_id", "scope", "identity"):
        s, b = _call(server, "POST", "/v2/memory/add", {"text": "hi", key: "mallory"})
        assert s == 400, (key, s, b)
        assert b["code"] == "VALIDATION"
    s, b = _call(server, "POST", "/v2/memory/search", {"query": "q", "user_id": "x"})
    assert s == 400


def test_strict_json_rejected(server):
    """Malformed + duplicate-key bodies never reach the facade."""
    s, b = _call(server, "POST", "/v2/memory/add", raw="{not json")
    assert s == 400
    s, b = _call(server, "POST", "/v2/memory/add", raw='{"text": "a", "text": "b"}')
    assert s == 400 and b["code"] == "VALIDATION"


def test_missing_or_foreign_ref_denied_identically(server):
    s, b = _call(server, "POST", "/v2/memory/inspect", {"ref": "mref1.a.b.c.1.0"})
    assert s == 404
    assert b["code"] == "NOT_FOUND_OR_UNAUTHORIZED"
    s2, b2 = _call(server, "POST", "/v2/memory/forget", {"ref": "mref1.a.b.c.1.0"})
    assert s2 == 404 and b2 == b


# ---------------------------------------------------------------------------
# contract: the round-trip
# ---------------------------------------------------------------------------


def test_add_readiness_search_inspect_forget_round_trip(server):
    res = _add(server, "Alice rotates the deploy key every Friday.")
    assert res["schema"] == "v5-contracts/1"
    assert res["ref"].startswith("mref1.")
    assert res["receipt_id"] and res["acceptance"] == "accepted"
    assert res["memory_id"]

    ready = _wait_ready(server, res["receipt_id"])
    assert ready["receipt_id"] == res["receipt_id"]
    assert ready["state"] in ("ready", "partial")
    assert "source_lexical_ready" in ready["capabilities"]

    s, found = _call(server, "POST", "/v2/memory/search", {"query": "deploy key"})
    assert s == 200
    assert found["status"] in ("ready", "partial")
    assert any("deploy key" in h["quote"] for h in found["items"]), found

    s, detail = _call(server, "POST", "/v2/memory/inspect", {"ref": res["ref"]})
    assert s == 200
    assert detail["found"] is True
    assert detail["provenance"]["source_id"] == res["memory_id"]
    assert detail["lifecycle"]["disposition"] in ("recorded", "active")

    s, gone = _call(server, "POST", "/v2/memory/forget", {"ref": res["ref"]})
    assert s == 200
    assert gone["mode"] == "operation" and gone["mutated"] is True
    assert res["ref"] in gone["selection"]

    s, after = _call(server, "POST", "/v2/memory/search", {"query": "deploy key"})
    assert s == 200
    assert not any("deploy key" in h["quote"] for h in after["items"])


def test_pending_state_is_honest_not_fake_empty(tmp_path):
    """worker='external' drains nothing: session search reports pending,
    never a ready-looking empty result."""
    httpd = create_memory_app(
        str(tmp_path / "mem.db"),
        user_id="alice",
        worker="external",
        tokens=_creds(),
    )
    srv = _start(httpd)
    try:
        res = _add(srv, "the migration checklist lives in ops/docs")
        assert res["acceptance"] in ("accepted", "held", "protected")

        ready = _wait_ready(srv, res["receipt_id"], timeout_ms=0)
        assert ready["state"] == "pending"  # nothing drained — honest

        s, out = _call(srv, "POST", "/v2/memory/search", {
            "query": "migration checklist", "ready_timeout_ms": 50})
        assert s == 200
        assert out["status"] == "pending"
        assert out["readiness"]["causal_satisfied"] is False
        assert out["readiness"]["unresolved"]

        s, ev = _call(srv, "POST", "/v2/memory/search", {
            "query": "migration checklist", "consistency": "eventual"})
        assert s == 200
        assert ev["status"] != "pending"  # eventual skips the barrier
    finally:
        srv.stop()
        httpd.application.memory.close()


def test_forget_preview_then_confirm(server):
    _add(server, "the quarterly report draft is in the shared drive")
    s, preview = _call(server, "POST", "/v2/memory/forget", {
        "ref": "quarterly report", "preview": True})
    assert s == 200
    assert preview["mode"] == "preview" and preview["mutated"] is False
    assert preview["confirmation_token"] and preview["selection"]

    s, done = _call(server, "POST", "/v2/memory/forget", {
        "confirm_token": preview["confirmation_token"]})
    assert s == 200
    assert done["mode"] == "operation" and done["mutated"] is True
    assert done["selection"] == preview["selection"]

    # Single-use: replaying the consumed token is a typed conflict.
    s, replay = _call(server, "POST", "/v2/memory/forget", {
        "confirm_token": preview["confirmation_token"]})
    assert s == 409 and replay["code"] == "OPERATION_CONFLICT"


def test_status_and_capabilities_honest(server):
    s, st = _call(server, "GET", "/v2/memory/status")
    assert s == 200
    assert st["profile"] == "local_memory"
    assert st["namespace"].startswith("ns_")
    assert st["caller"] == "local-owner"
    assert st["worker"]["mode"] == "managed"

    s, caps = _call(server, "GET", "/v2/memory/capabilities")
    assert s == 200
    assert caps["transport"]["tls"] == "unimplemented"
    assert caps["transport"]["bind"] == "loopback"
    assert caps["transport"]["authn"] == "bearer-token"
    assert caps["worker"]["mode"] == "managed"
    assert caps["limits"]["max_text_bytes"] > 0
    assert caps["limits"]["max_limit"] == 64
    assert caps["capabilities"]["source_lane"] in ("available", "unavailable")


def test_readiness_unknown_receipt_404(server):
    s, b = _call(server, "GET", "/v2/memory/readiness/rcpt_missing_1")
    assert s == 404
    assert b["code"] == "NOT_FOUND_OR_UNAUTHORIZED"


# ---------------------------------------------------------------------------
# mounted under /v1 via create_api (route registration)
# ---------------------------------------------------------------------------


def _engine(tmp_path):
    return open_store(
        str(tmp_path / "eng.db"),
        config_from_mapping({"mode": "offline_rules", "capture": {"enabled": True}}),
        LocalHost(profile_id="demo", principal_id="op", conversation_id="c1"),
        create=True,
    )


def test_mounted_surface_and_default_off(tmp_path):
    """create_server(enable_v5_memory=True) mounts /v2/memory on the same
    socket; default off keeps the surface absent."""
    eng = _engine(tmp_path)
    creds = [
        TokenCredential.mint(TOKEN, "op", VERBS),
        TokenCredential.mint(OTHER, "agent", ["read", "ingest"]),
    ]
    try:
        srv = create_server(
            eng, creds, HttpConfig(),
            enable_v5_memory=True,
            memory_path=str(tmp_path / "mem.db"),
            memory_user="op",
        ).start()
        try:
            s, st = _call(srv, "GET", "/v1/status")
            assert s == 200 and st["mode"] == "offline_rules"

            s, mem = _call(srv, "GET", "/v2/memory/status")
            assert s == 200 and mem["profile"] == "local_memory"

            # The agent token is valid for /v1 but scoped elsewhere for /v2.
            s, b = _call(srv, "GET", "/v2/memory/status", token=OTHER)
            assert s == 403 and b["code"] == "FORBIDDEN"

            res = _add(srv, "mounted add path works")
            assert res["ref"].startswith("mref1.")
        finally:
            srv.stop()
            srv.httpd.application.memory.close()

        srv2 = create_server(eng, creds, HttpConfig()).start()
        try:
            s, b = _call(srv2, "GET", "/v2/memory/status")
            assert s == 404  # surface absent by default
        finally:
            srv2.stop()
    finally:
        eng.close()


def test_error_shape_never_a_traceback(server):
    """Errors are flat {error, code, retryable} — no traceback body."""
    s, b = _call(server, "POST", "/v2/memory/inspect", {"ref": 12345})
    assert s == 400
    assert set(b) == {"error", "code", "retryable"}
    assert "Traceback" not in b["error"]
