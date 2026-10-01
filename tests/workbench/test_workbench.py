"""Operator workbench tests (SPEC_V4 §49) — real server, real store.

Every request goes through the threaded loopback HTTP server; mutations
ride the same Engine review/purge paths the CLI uses.
"""

from __future__ import annotations

import http.client
import json
from urllib.parse import urlencode

import pytest

from verbatim.api import open_store
from verbatim.config import config_from_mapping
from verbatim.host import LocalHost
from verbatim.service import TokenCredential
from verbatim.service.httpd import HttpConfig
from verbatim.workbench import create_server

TOKEN = "wb-operator-token"


def _cfg():
    return config_from_mapping(
        {
            "mode": "offline_rules",
            "capture": {"enabled": True},
            "admission": {"require_review": True},
        }
    )


@pytest.fixture()
def server(tmp_path):
    eng = open_store(
        str(tmp_path),
        _cfg(),
        LocalHost(profile_id="demo", principal_id="op", conversation_id="c1"),
        create=True,
    )
    creds = [
        TokenCredential.mint(
            TOKEN, "op", ["read", "quote", "derive", "ingest", "review", "admin"]
        ),
        TokenCredential.mint("agent-token", "agent", ["read", "ingest"]),
    ]
    srv = create_server(eng, creds, HttpConfig()).start()
    yield eng, srv
    srv.stop()
    eng.close()


def _call(srv, method, path, body=None, token=TOKEN, form=False, follow=False):
    conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=15)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if body is not None:
        if form:
            data = urlencode(body)
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            data = json.dumps(body)
            headers["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=headers)
    resp = conn.getresponse()
    payload = resp.read()
    hdrs = dict(resp.getheaders())
    conn.close()
    return resp.status, hdrs, payload


def _capture_direct(eng, text):
    """Seed evidence through the engine itself (the workbench serves an
    operator — capture itself is the service/SDK's job)."""
    from verbatim.core.time import now_us
    from verbatim.core.types import (
        Provenance,
        SourceEnvelope,
        SourceKind,
    )

    scope = eng.host.default_scope()
    env = SourceEnvelope(
        origin="wb-test",
        source_kind=SourceKind.USER_MESSAGE,
        scope=scope,
        speaker_id="op",
        payload=text.encode("utf-8"),
        event_us=now_us(),
        captured_us=now_us(),
        provenance=Provenance.DIRECT_USER,
    )
    receipt = eng.ingest(env)
    eng.run_pending(limit=64)
    return receipt


def _claim_ids(eng):
    with eng.store.read() as conn:
        return [
            r[0]
            for r in conn.execute("SELECT claim_id FROM claims ORDER BY claim_id")
        ]


# ---------------------------------------------------------------------------
# authentication + transport honesty
# ---------------------------------------------------------------------------


def test_no_listener_on_import():
    import verbatim.workbench as wb

    assert not hasattr(wb, "_server")
    from verbatim.service import live_servers

    assert live_servers() == ()


def test_auth_required_and_operator_class(server):
    _, srv = server
    s, h, b = _call(srv, "GET", "/", token=None)
    assert s == 401 and h.get("WWW-Authenticate", "").startswith("Bearer")
    s, _, b = _call(srv, "GET", "/", token="bad")
    assert s == 401
    # A provisioned non-operator token gets the same 401 — the workbench
    # simply does not exist for it (§49 operator surface).
    s, _, b = _call(srv, "GET", "/", token="agent-token")
    assert s == 401


def test_pages_render_with_security_headers(server):
    _, srv = server
    for path in ("/", "/status", "/objects", "/reviews", "/jobs",
                 "/readiness", "/forget", "/search?query=x"):
        s, h, body = _call(srv, "GET", path)
        assert s == 200, (path, s, body[:200])
        assert h.get("X-Content-Type-Options") == "nosniff"
        assert h.get("X-Frame-Options") == "DENY"
        assert "script-src" not in h.get("Content-Security-Policy", "")
        assert "default-src 'none'" in h.get("Content-Security-Policy", "")
        assert b"<script" not in body  # the workbench emits no scripts


# ---------------------------------------------------------------------------
# evidence views — verified reads + quote gating
# ---------------------------------------------------------------------------


def test_objects_and_claim_detail(server):
    eng, srv = server
    _capture_direct(eng, "The wifi password is on the fridge.")

    s, _, body = _call(srv, "GET", "/objects")
    assert s == 200 and b"claims" in body.lower()

    # admit the pending claim through the review queue UI
    s, _, body = _call(srv, "GET", "/reviews")
    assert s == 200
    s, _, listing = _call(srv, "GET", "/api/reviews")
    rid = json.loads(listing)["reviews"][0]["review_id"]
    s, h, _ = _call(srv, "POST", f"/reviews/{rid}/approve", form=True, body={})
    assert s == 303

    cid = _claim_ids(eng)[0]
    s, _, body = _call(srv, "GET", f"/claims/{cid}")
    assert s == 200
    # Exact evidence beside the interpretation, escaped but verbatim.
    assert b"wifi password" in body
    assert b"revisions" in body and b"evidence" in body


def test_stored_text_is_escaped(server):
    """Evidence rendering cannot become an XSS channel (V4-49.08)."""
    eng, srv = server
    payload = '<script>alert("xss")</script> & <b>bold</b>'
    _capture_direct(eng, payload)
    cid = _claim_ids(eng)[0]
    s, _, body = _call(srv, "GET", f"/claims/{cid}")
    assert s == 200
    assert b'<script>alert("xss")</script>' not in body
    assert html_escaped(payload) in body
    # search view escapes too
    s, _, body = _call(srv, "GET", "/search?query=xss")
    assert s == 200
    assert b'<script>alert("xss")</script>' not in body


def html_escaped(text: str) -> bytes:
    import html

    return html.escape(text).encode("utf-8")


def test_suppressed_items_render_withheld(server):
    """Suppressed objects stay visible as withheld — never rendered (V4-49.05)."""
    eng, srv = server
    _capture_direct(eng, "secret meeting location")
    cid = _claim_ids(eng)[0]

    s, _, _ = _call(srv, "POST", "/forget/suppress", form=True, body={
        "object_kind": "claim", "object_id": cid})
    assert s == 303

    s, _, body = _call(srv, "GET", "/objects")
    assert s == 200
    assert b"withheld" in body
    assert b"secret meeting location" not in body

    # The claim detail page renders withheld, not content, not a stack trace.
    s, _, body = _call(srv, "GET", f"/claims/{cid}")
    assert s == 200
    assert b"withheld" in body.lower()
    assert b"secret meeting location" not in body

    # JSON mirror reports the same withholding.
    s, _, listing = _call(srv, "GET", "/api/objects")
    doc = json.loads(listing)
    assert doc["claims"][0]["withheld"] is True


def test_forget_preview_execute_flow(server):
    eng, srv = server
    _capture_direct(eng, "delete me entirely")
    cid = _claim_ids(eng)[0]

    s, _, body = _call(srv, "POST", "/forget/preview", form=True, body={
        "object_kind": "claim", "object_id": cid})
    assert s == 200 and b"preview" in body.lower()
    assert b"nothing deleted yet" in body

    s, _, listing = _call(srv, "GET", "/api/objects")
    # preview alone changes nothing
    assert json.loads(listing)["claims"][0]["withheld"] is False

    from verbatim.core.identity import scope_key
    with eng.store.read() as conn:
        pid = conn.execute(
            "SELECT purge_id FROM purges WHERE scope_id = ?",
            (scope_key(eng.host.default_scope()),),
        ).fetchone()[0]

    # execute without the checkbox is refused
    s, _, body = _call(srv, "POST", "/forget/execute", form=True, body={
        "purge_id": pid})
    assert s == 200 and b"error" in body.lower()

    s, _, body = _call(srv, "POST", "/forget/execute", form=True, body={
        "purge_id": pid, "confirm": "yes"})
    assert s == 200 and b"completed" in body.lower()

    s, _, body = _call(srv, "GET", f"/claims/{cid}")
    assert b"withheld" in body.lower() or b"not found" in body.lower()
    s, _, body = _call(srv, "GET", "/forget")
    assert b"verified erased" in body


def test_correction_creates_review(server):
    eng, srv = server
    _capture_direct(eng, "old phone number")
    _capture_direct(eng, "new phone number")
    cids = _claim_ids(eng)
    assert len(cids) >= 2

    s, h, _ = _call(srv, "POST", f"/claims/{cids[0]}/correct", form=True, body={
        "effect": "dispute", "reason": "contradicted by newer evidence"})
    assert s == 303 and h["Location"].startswith("/reviews/")

    s, _, listing = _call(srv, "GET", "/api/reviews")
    reviews = json.loads(listing)["reviews"]
    assert any(r["proposed_effect"]["effect"] == "dispute" for r in reviews)


def test_drain_and_readiness_pages(server):
    eng, srv = server
    receipt = _capture_direct(eng, "readiness probe text")
    sid = receipt.accepted[0]
    s, _, body = _call(srv, "POST", "/jobs/drain", form=True, body={})
    assert s == 200 and b"drain" in body.lower()
    s, _, body = _call(srv, "GET", f"/readiness?receipt_id=rc_ingest:{sid}:1")
    assert s == 200 and b"receipt" in body.lower()
    s, _, body = _call(srv, "GET", "/readiness?receipt_id=rc_ingest:nope:1")
    assert s == 200 and b"not_found" in body.lower()  # honest error, not a crash


def test_api_mirrors(server):
    eng, srv = server
    _capture_direct(eng, "api mirror evidence")
    s, _, b = _call(srv, "GET", "/api/status")
    assert s == 200 and json.loads(b)["mode"] == "offline_rules"
    s, _, b = _call(srv, "GET", "/api/objects")
    assert s == 200 and "claims" in json.loads(b)
    s, _, b = _call(srv, "GET", "/api/reviews")
    assert s == 200
