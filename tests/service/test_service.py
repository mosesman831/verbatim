"""HTTP service surface tests (SPEC_V4 §44/§45/§52).

Real engine on a real store, real threaded ``http.server`` on an ephemeral
loopback port — the tests talk to the wire, not to stubs.
"""

from __future__ import annotations

import http.client
import json
import socket

import pytest

from verbatim.api import open_store
from verbatim.config import config_from_mapping
from verbatim.host import LocalHost
from verbatim.service import (
    TokenCredential,
    create_server,
    live_servers,
    parse_credentials,
)
from verbatim.service.httpd import HttpConfig

TOKEN = "op-secret-token"
AGENT_TOKEN = "agent-secret-token"


def _cfg():
    return config_from_mapping(
        {
            "mode": "offline_rules",
            "capture": {"enabled": True},
            "admission": {"require_review": True},
        }
    )


def _engine(tmp_path, principal="op", conversation="c1"):
    return open_store(
        str(tmp_path),
        _cfg(),
        LocalHost(profile_id="demo", principal_id=principal, conversation_id=conversation),
        create=True,
    )


def _operator_token(principal="op"):
    return TokenCredential.mint(
        TOKEN, principal,
        ["read", "quote", "derive", "ingest", "review", "admin"],
    )


@pytest.fixture()
def server(tmp_path):
    eng = _engine(tmp_path)
    srv = create_server(
        eng,
        [_operator_token(), TokenCredential.mint(AGENT_TOKEN, "agent", ["read", "ingest"])],
        HttpConfig(),
    ).start()
    yield eng, srv
    srv.stop()
    eng.close()


def _call(srv, method, path, body=None, token=TOKEN, raw=None, ctype="application/json"):
    conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=15)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = raw
    if body is not None:
        data = json.dumps(body) if ctype == "application/json" else body
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
    return resp.status, dict(resp.getheaders()), decoded


def _capture(srv, text, **kw):
    body = {"operation_id": kw.pop("operation_id", "op-1"), "text": text,
            "kind": "user_message", "speaker_id": "me"}
    body.update(kw)
    s, h, b = _call(srv, "POST", "/v1/capture", body)
    assert s == 200, b
    return b


def _drain(srv):
    s, h, b = _call(srv, "POST", "/v1/drain", {})
    assert s == 200, b
    return b


def _admit_all(srv):
    """Approve every open review through the real review path."""
    s, h, b = _call(srv, "GET", "/v1/reviews")
    assert s == 200
    for r in b["reviews"]:
        s, h, out = _call(srv, "POST", f"/v1/reviews/{r['review_id']}/approve", {})
        assert s == 200, out


# ---------------------------------------------------------------------------
# import-time safety (V4-45.08)
# ---------------------------------------------------------------------------


def test_no_listener_on_import():
    """Importing the package must not bind a socket."""
    import importlib
    import verbatim.service

    importlib.reload(verbatim.service)
    assert live_servers() == ()
    # No listener exists yet — nothing answered on the ports a later serve
    # would pick (ephemeral anyway). The honest check: the module exposes
    # no bound server objects.
    import verbatim.service.server as srv_mod

    assert srv_mod.live_servers() == ()


def test_refuses_non_loopback_bind(tmp_path):
    from verbatim.core.types import ErrorCode, VerbatimError

    eng = _engine(tmp_path)
    try:
        with pytest.raises(VerbatimError) as ei:
            create_server(eng, [_operator_token()], HttpConfig(host="0.0.0.0"))
        assert ei.value.code == ErrorCode.CONFIG_INVALID
    finally:
        eng.close()


def test_requires_credentials(tmp_path):
    from verbatim.core.types import ErrorCode, VerbatimError

    eng = _engine(tmp_path)
    try:
        with pytest.raises(VerbatimError) as ei:
            create_server(eng, [])
        assert ei.value.code == ErrorCode.CONFIG_INVALID
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# authentication
# ---------------------------------------------------------------------------


def test_unauthenticated_is_401(server):
    eng, srv = server
    for method, path in (
        ("GET", "/v1/status"),
        ("POST", "/v1/recall"),
        ("POST", "/v1/capture"),
        ("POST", "/v1/drain"),
        ("GET", "/v1/claims/x"),
        ("POST", "/v1/forget"),
    ):
        status, headers, body = _call(srv, method, path, token=None)
        assert status == 401, (path, status, body)
        assert headers.get("WWW-Authenticate", "").startswith("Bearer")
        assert body["error"]["code"] == "UNAUTHENTICATED"


def test_bad_token_is_401(server):
    _, srv = server
    status, _, body = _call(srv, "GET", "/v1/status", token="wrong-token")
    assert status == 401
    assert body["error"]["code"] == "UNAUTHENTICATED"


def test_malformed_authorization_header(server):
    _, srv = server
    conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=15)
    conn.request("GET", "/v1/status", headers={"Authorization": "Basic abc"})
    resp = conn.getresponse()
    assert resp.status == 401
    resp.read()
    conn.close()


# ---------------------------------------------------------------------------
# authorization — single authority, indistinguishable denial
# ---------------------------------------------------------------------------


def test_agent_token_denied_operator_endpoints(server):
    """An agent-class token lacks ``admin`` — denial is the 404 shape."""
    _, srv = server
    status, _, body = _call(srv, "POST", "/v1/drain", {}, token=AGENT_TOKEN)
    assert status == 404
    assert body["error"]["code"] == "NOT_FOUND_OR_UNAUTHORIZED"

    status, _, body = _call(srv, "POST", "/v1/forget", {"targets": [
        {"object_kind": "claim", "object_id": "x"}]}, token=AGENT_TOKEN)
    assert status == 404
    assert body["error"]["code"] == "NOT_FOUND_OR_UNAUTHORIZED"


def test_denial_indistinguishable_missing_vs_private(tmp_path):
    """A claim in a partition the credential cannot read is byte-identical
    to a missing claim (§09.09/§10.05, §52)."""
    eng = _engine(tmp_path)
    try:
        # A credential pinned to a different conversation partition.
        other = TokenCredential.mint(
            "other-token", "op", ["read"], scope={"conversation_id": "other-room"}
        )
        srv = create_server(eng, [_operator_token(), other], HttpConfig()).start()
        try:
            _capture(srv, "The wifi password is on the fridge.")
            _drain(srv)
            _admit_all(srv)
            s, _, b = _call(srv, "POST", "/v1/recall", {"query": "wifi"})
            cid = b["items"][0]["claim_id"]

            s1, _, missing = _call(srv, "GET", "/v1/claims/definitely-missing")
            s2, _, private = _call(srv, "GET", f"/v1/claims/{cid}", token="other-token")
            assert s1 == s2 == 404
            # Same body shape, same code — the per-request correlation id
            # is fresh randomness, not a distinguishing detail.
            missing["error"].pop("correlation_id")
            private["error"].pop("correlation_id")
            assert missing == private
            assert missing["error"]["code"] == "NOT_FOUND_OR_UNAUTHORIZED"
        finally:
            srv.stop()
    finally:
        eng.close()


def test_revoked_grant_loses_access(server):
    """Revocation through the grant table fences the token immediately —
    the HTTP surface holds no cached authority of its own."""
    from verbatim import governance

    eng, srv = server
    s, _, _ = _call(srv, "GET", "/v1/status")
    assert s == 200
    from verbatim.core.identity import scope_key

    sid = scope_key(eng.host.default_scope())
    with eng.store.read() as conn:
        rows = conn.execute(
            "SELECT grant_id FROM grants_v3 WHERE scope_id = ? AND principal_id = 'op'",
            (sid,),
        ).fetchall()
    assert rows
    with eng.store.tx() as conn:
        for (gid,) in rows:
            governance.revoke_grant(conn, gid)
    s, _, b = _call(srv, "GET", "/v1/status")
    assert s == 404
    assert b["error"]["code"] == "NOT_FOUND_OR_UNAUTHORIZED"


# ---------------------------------------------------------------------------
# contract surface: capture → recall → inspect → correct → forget
# ---------------------------------------------------------------------------


def test_capture_recall_inspect_correct_forget(server):
    eng, srv = server
    cap = _capture(srv, "The wifi password is on the fridge.")
    assert cap["accepted"] and cap["receipts"]

    # readiness: the capture receipt is a real obligation DAG handle.
    rid = cap["receipts"][cap["accepted"][0]]
    s, _, st = _call(srv, "GET", f"/v1/readiness/{rid}")
    assert s == 200 and st["receipt_id"] == rid

    _drain(srv)
    s, _, b = _call(srv, "POST", f"/v1/readiness/{rid}/wait", {"timeout_s": 5})
    assert s == 200

    # pending claims are not recalled until admitted.
    s, _, b = _call(srv, "POST", "/v1/recall", {"query": "wifi password"})
    assert s == 200 and b["items"] == []

    _admit_all(srv)
    s, _, b = _call(srv, "POST", "/v1/recall", {"query": "wifi password"})
    assert s == 200 and len(b["items"]) == 1
    cid = b["items"][0]["claim_id"]

    s, _, detail = _call(srv, "GET", f"/v1/claims/{cid}")
    assert s == 200
    assert detail["evidence"] and detail["revisions"]

    # correction proposal → review → approve path.
    s, _, out = _call(srv, "POST", "/v1/corrections", {
        "claim_id": cid, "effect": "archive", "reason": "stale",
    })
    assert s == 200 and out["review_id"]
    s, _, out = _call(srv, "POST", f"/v1/reviews/{out['review_id']}/approve", {})
    assert s == 200 and out["state"] == "approved"

    # forget: preview → execute → gone, indistinguishable from missing.
    s, _, plan = _call(srv, "POST", "/v1/forget", {
        "targets": [{"object_kind": "claim", "object_id": cid}]})
    assert s == 200 and plan["state"] == "previewed"
    s, _, out = _call(srv, "POST", f"/v1/forget/{plan['purge_id']}/execute", {})
    assert s == 400  # confirmation required (V4-45.05)
    s, _, out = _call(srv, "POST", f"/v1/forget/{plan['purge_id']}/execute",
                      {"confirm": True})
    assert s == 200 and out["state"] == "completed"

    s, _, b = _call(srv, "POST", "/v1/recall", {"query": "wifi password"})
    assert b["items"] == []
    s, _, b = _call(srv, "GET", f"/v1/claims/{cid}")
    assert s == 404


def test_capture_idempotent_replay(server):
    _, srv = server
    r1 = _capture(srv, "same payload", operation_id="dup-1")
    r2 = _capture(srv, "same payload", operation_id="dup-1")
    assert r1["accepted"] == r2["accepted"]
    assert r2["duplicate"] is True


def test_effects_apply_proposal_idempotent(server):
    _, srv = server
    _capture(srv, "Pinned at revision one.")
    _drain(srv)
    s, _, b = _call(srv, "GET", "/v1/reviews")
    rid = b["reviews"][0]["review_id"]
    s, _, review = _call(srv, "GET", f"/v1/reviews/{rid}")
    cid = review["proposed_effect"]["claim_id"]

    body = {
        "operation_id": "eff-1",
        "effect": "admit",
        "targets": [[cid, 1]],
        "reason": "operator",
    }
    s, _, r1 = _call(srv, "POST", "/v1/effects", body)
    assert s == 200 and r1["replayed"] is False
    s, _, r2 = _call(srv, "POST", "/v1/effects", body)
    assert s == 200 and r2["replayed"] is True
    s, _, r3 = _call(srv, "POST", "/v1/effects", {
        "operation_id": "eff-1", "effect": "archive",
        "targets": [[cid, 2]], "reason": "changed input"})
    assert s == 400  # operation id reused with different input


def test_remember_and_feedback(server):
    _, srv = server
    cap = _capture(srv, "Call the dentist on Monday.")
    sid = cap["accepted"][0]
    _drain(srv)
    s, _, b = _call(srv, "POST", "/v1/remember", {
        "source_id": sid, "start_byte": 0, "end_byte": 10,
        "operation_id": "rem-1"})
    assert s == 200 and b["claim_id"]
    s, _, b = _call(srv, "POST", "/v1/feedback", {
        "claim_id": b["claim_id"], "kind": "helpful"})
    assert s == 200


def test_suppress_and_lift(server):
    _, srv = server
    _capture(srv, "temporary secret")
    _drain(srv)
    _admit_all(srv)
    s, _, b = _call(srv, "POST", "/v1/recall", {"query": "temporary secret"})
    cid = b["items"][0]["claim_id"]

    s, _, out = _call(srv, "POST", "/v1/forget/suppress", {
        "targets": [{"object_kind": "claim", "object_id": cid}]})
    assert s == 200 and out["state"] == "suppressed"
    s, _, b = _call(srv, "POST", "/v1/recall", {"query": "temporary secret"})
    assert b["items"] == []
    s, _, listing = _call(srv, "GET", "/v1/purges")
    assert any(p["state"] == "suppressed" for p in listing["purges"])

    s, _, out = _call(srv, "POST", f"/v1/forget/{out['purge_id']}/lift", {})
    assert s == 200
    s, _, b = _call(srv, "POST", "/v1/recall", {"query": "temporary secret"})
    assert len(b["items"]) == 1


# ---------------------------------------------------------------------------
# transport validation (V4-45.07/45.09)
# ---------------------------------------------------------------------------


def test_validation_errors(server):
    _, srv = server
    s, _, b = _call(srv, "POST", "/v1/recall", {"query": 42})
    assert s == 400 and b["error"]["code"] == "VALIDATION"
    s, _, b = _call(srv, "POST", "/v1/recall", {"query": "x", "bogus": 1})
    assert s == 400  # unknown fields rejected
    s, _, b = _call(srv, "POST", "/v1/capture", {"text": "x"})
    assert s == 400  # operation_id required (V4-45.01)
    s, _, b = _call(srv, "GET", "/v1/nope")
    assert s == 404
    s, _, b = _call(srv, "POST", "/v1/recall", raw="{not json")
    assert s == 400
    s, _, b = _call(srv, "POST", "/v1/recall",
                    raw=json.dumps({"query": "x"}), ctype="text/plain")
    assert s == 415


def test_body_size_bound(tmp_path):
    eng = _engine(tmp_path)
    try:
        srv = create_server(eng, [_operator_token()], HttpConfig(max_body_bytes=4096)).start()
        try:
            s, _, b = _call(srv, "POST", "/v1/capture", {
                "operation_id": "big", "text": "x" * 8000})
            assert s == 413
        finally:
            srv.stop()
    finally:
        eng.close()


def test_status_and_capabilities_honest(server):
    _, srv = server
    s, _, st = _call(srv, "GET", "/v1/status")
    assert s == 200 and st["mode"] == "offline_rules"
    assert "capabilities" in st
    s, _, caps = _call(srv, "GET", "/v1/capabilities")
    assert s == 200
    assert caps["transport"]["tls"] == "unimplemented"
    assert caps["transport"]["bind"] == "loopback"


def test_token_file_and_env_parsing(tmp_path):
    doc = {"tokens": [
        {"token": "t1", "principal_id": "p1", "verbs": ["read"]},
        {"token": "t2", "principal_id": "p2", "verbs": ["admin"]},
    ]}
    creds = parse_credentials(doc)
    assert len(creds) == 2
    assert creds[0].principal_id == "p1"
    # Raw secrets are digested — never retained on the credential.
    assert b"t1" not in creds[0].token_digest

    from verbatim.core.types import VerbatimError

    with pytest.raises(VerbatimError):
        parse_credentials({"tokens": []})
    with pytest.raises(VerbatimError):
        parse_credentials({"tokens": [{"token": "x", "principal_id": "p",
                                       "verbs": ["not_a_verb"]}]})
    with pytest.raises(VerbatimError):
        parse_credentials({"tokens": [{"token": "x", "principal_id": "p",
                                       "verbs": ["read"], "bogus": 1}]})
