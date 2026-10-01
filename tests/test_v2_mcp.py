"""Stdio MCP adapter: framing, JSON-RPC dispatch, startup-bound caller
(SPEC_V2 §44). All tests run against a real engine over simulated pipes —
no transport is ever opened on import or by the dispatcher itself.
"""

from __future__ import annotations

import io
import json

import pytest

from verbatim.api import open_store
from verbatim.config import config_from_mapping
from verbatim.core.time import now_us
from verbatim.core.types import (
    CallerContext,
    GrantKind,
    Provenance,
    SourceEnvelope,
    SourceKind,
)
from verbatim.host import LocalHost
from verbatim.mcp import MCP_GRANTS, McpServer, read_message, serve_stdio

T0 = 1_700_000_000_000_000


def _engine(tmp_path):
    cfg = config_from_mapping(
        {"mode": "offline_rules", "capture": {"enabled": True}}
    )
    return open_store(
        str(tmp_path),
        cfg,
        LocalHost(profile_id="demo", principal_id="me", conversation_id="c1"),
        create=True,
    )


def _caller():
    return CallerContext(
        profile_id="demo",
        principal_id="me",
        agent_id="mcp",
        session_id="sess",
        conversation_id="c1",
        grants=MCP_GRANTS,
    )


def _seed(eng, scope):
    eng.ingest(
        SourceEnvelope(
            origin="mcp-test",
            source_kind=SourceKind.USER_MESSAGE,
            scope=scope,
            speaker_id="me",
            payload=b"The launch is scheduled for March.",
            event_us=T0,
            captured_us=T0,
            provenance=Provenance.DIRECT_USER,
        )
    )
    eng.run_pending(limit=64)


def _frame(msg) -> str:
    body = json.dumps(msg)
    return f"Content-Length: {len(body)}\r\n\r\n{body}"


def _serve(eng, caller, requests, *, framed=False):
    text = "".join(
        (_frame(m) if framed else json.dumps(m) + "\n") for m in requests
    )
    out = io.StringIO()
    rc = serve_stdio(eng, caller, instream=io.StringIO(text), outstream=out)
    return rc, [json.loads(line) for line in out.getvalue().splitlines()]


def test_import_has_no_io_side_effects():
    import importlib

    mod = importlib.import_module("verbatim.mcp")
    # Constructing the dispatcher is pure; serve_stdio owns the loop.
    assert hasattr(mod, "serve_stdio") and hasattr(mod, "McpServer")


def test_initialize_tools_list_tools_call(tmp_path):
    eng = _engine(tmp_path)
    try:
        _seed(eng, eng.host.default_scope())
        with eng.store.read() as conn:
            (cid,) = conn.execute(
                "SELECT claim_id FROM claim_revisions LIMIT 1"
            ).fetchone()
        rc, out = _serve(
            eng,
            _caller(),
            [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2024-11-05"}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                 "params": {"name": "verbatim_feedback",
                            "arguments": {"claim_id": cid, "kind": "helpful"}}},
            ],
        )
        assert rc == 0
        assert out[0]["result"]["serverInfo"]["name"] == "verbatim"
        names = {t["name"] for t in out[1]["result"]["tools"]}
        assert names == {
            "verbatim_recall",
            "verbatim_remember",
            "verbatim_evidence",
            "verbatim_feedback",
        }
        assert out[2]["result"]["structuredContent"] == {"recorded": True}
    finally:
        eng.close()


def test_recall_and_inspect_tools(tmp_path):
    eng = _engine(tmp_path)
    try:
        scope = eng.host.default_scope()
        _seed(eng, scope)
        # Admit the pending claim so recall can see it (trusted operator).
        with eng.store.read() as conn:
            cid, rev = conn.execute(
                "SELECT claim_id, revision FROM claim_revisions"
                " ORDER BY revision DESC LIMIT 1"
            ).fetchone()
        from verbatim.core.types import EffectProposal

        eng.apply_proposal(
            EffectProposal(
                version=1, operation_id="op-ad", effect="admit",
                targets=((cid, rev),), actor_id="me", reason="ok",
            )
        )

        rc, out = _serve(
            eng,
            _caller(),
            [
                {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                 "params": {"name": "verbatim_recall",
                            "arguments": {"query": "launch"}}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                 "params": {"name": "verbatim_evidence",
                            "arguments": {"claim_id": cid}}},
            ],
        )
        assert rc == 0
        items = out[0]["result"]["structuredContent"]["items"]
        assert len(items) == 1
        assert "launch" in items[0]["text"]
        assert out[0]["result"]["structuredContent"]["text"]
        detail = out[1]["result"]["structuredContent"]
        assert detail["claim_id"] == cid and detail["evidence"]
    finally:
        eng.close()


def test_remember_tool_returns_claim_id(tmp_path):
    eng = _engine(tmp_path)
    try:
        scope = eng.host.default_scope()
        receipt = eng.ingest(
            SourceEnvelope(
                origin="mcp-test",
                source_kind=SourceKind.USER_MESSAGE,
                scope=scope,
                speaker_id="me",
                payload=b"Wi-Fi is down since Tuesday.",
                event_us=T0,
                captured_us=T0,
                provenance=Provenance.DIRECT_USER,
            )
        )
        sid = receipt.accepted[0]
        rc, out = _serve(
            eng,
            _caller(),
            [
                {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                 "params": {"name": "verbatim_remember",
                            "arguments": {"source_id": sid, "start": 0, "end": 9}}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                 "params": {"name": "verbatim_remember",
                            "arguments": {"source_id": sid, "start": 0, "end": 9}}},
            ],
        )
        assert rc == 0
        c1 = out[0]["result"]["structuredContent"]["claim_id"]
        c2 = out[1]["result"]["structuredContent"]["claim_id"]
        assert c1 and c1 == c2  # idempotent replay
    finally:
        eng.close()


def test_framed_and_line_delimited_both_accepted(tmp_path):
    eng = _engine(tmp_path)
    try:
        rc, out = _serve(
            eng,
            _caller(),
            [
                {"jsonrpc": "2.0", "id": 1, "method": "ping"},
                {"jsonrpc": "2.0", "id": 2, "method": "ping"},
            ],
            framed=True,
        )
        assert rc == 0
        assert [m["id"] for m in out] == [1, 2]
        assert all("result" in m for m in out)
    finally:
        eng.close()


def test_error_envelopes_are_privacy_safe(tmp_path):
    eng = _engine(tmp_path)
    try:
        rc, out = _serve(
            eng,
            _caller(),
            [
                # Unknown claim: NOT_FOUND_OR_FORBIDDEN, nothing else.
                {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                 "params": {"name": "verbatim_evidence",
                            "arguments": {"claim_id": "does-not-exist"}}},
                # Unknown tool.
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                 "params": {"name": "verbatim_purge", "arguments": {}}},
                # Unknown method.
                {"jsonrpc": "2.0", "id": 3, "method": "verbatim/admin"},
                # Malformed request.
                {"id": 4, "method": "tools/list"},
            ],
        )
        assert rc == 0
        e1 = out[0]["error"]
        assert e1["code"] == -32000
        assert e1["message"] == "NOT_FOUND_OR_FORBIDDEN"
        assert "traceback" not in json.dumps(e1).lower()
        assert "does-not-exist" not in json.dumps(e1)
        assert out[1]["error"]["code"] == -32602
        assert out[2]["error"]["code"] == -32601
        assert out[3]["error"]["code"] == -32600
    finally:
        eng.close()


def test_parse_error_answered_then_stream_continues(tmp_path):
    eng = _engine(tmp_path)
    try:
        text = "not-json\n" + json.dumps(
            {"jsonrpc": "2.0", "id": 7, "method": "ping"}
        ) + "\n"
        out = io.StringIO()
        rc = serve_stdio(eng, _caller(), instream=io.StringIO(text), outstream=out)
        msgs = [json.loads(l) for l in out.getvalue().splitlines()]
        assert rc == 0
        assert msgs[0]["error"]["code"] == -32700
        assert msgs[1]["result"] == {}
    finally:
        eng.close()


def test_caller_identity_not_taken_from_request(tmp_path):
    """initialize params may never mint identity or grants (V2-09)."""
    eng = _engine(tmp_path)
    try:
        caller = _caller()
        server = McpServer(eng, caller)
        resp = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "clientInfo": {"name": "evil"},
                    "verbatim": {"principal_id": "root", "grants": ["operator"]},
                },
            }
        )
        assert "result" in resp
        assert server._caller is caller
        assert server._caller.principal_id == "me"
        assert GrantKind.OPERATOR not in server._caller.grants
    finally:
        eng.close()


def test_read_message_rejects_garbage():
    with pytest.raises(Exception):
        read_message(io.StringIO("Content-Length: abc\r\n\r\n{}"))
    assert read_message(io.StringIO("")) is None
