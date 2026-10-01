"""Remote-encoder permit gate (SPEC_V4 §12, F4-04, C14/C15/C18).

Every transport-bound encoder must refuse to dispatch without a
broker-minted ``DispatchPermit`` bound to its endpoint and the exact
request body. These tests prove the check lives INSIDE the encoder — a
call site that simply fails to pass a permit cannot reach the transport.
"""

from __future__ import annotations

import json
import sqlite3
from decimal import Decimal
from typing import Optional

import pytest

from verbatim.config import EmbeddingConfig, JudgeConfig, VerbatimConfig
from verbatim.core.types import ErrorCode, Mode, VerbatimError
from verbatim.core.types_v4 import DispatchPermit
from verbatim.embeddings.cloudflare import CloudflareEncoder
from verbatim.embeddings.encoder import encoder_requires_permit, get_encoder
from verbatim.embeddings.hashing import HashingEncoder
from verbatim.embeddings.ollama import HttpResult, OllamaEncoder
from verbatim.privacy.broker import (
    EndpointDescriptor,
    TransportBroker,
    payload_digest,
)
from verbatim.storage.schema import DDL_V1, DDL_V2
from verbatim.storage.schema_v4 import DDL_V4, _split
from tests.conftest import FakeClock, TestStore, grant_consent

CF_ID = "cloudflare:api.cloudflare.com"
OL_ID = "ollama:127.0.0.1:11434"


class _FakeTransport:
    def __init__(self, result: Optional[HttpResult] = None):
        self.result = result
        self.calls: list[dict] = []

    def request(self, method, url, *, body, headers, timeout_s):
        self.calls.append({"method": method, "url": url, "body": body})
        return self.result or HttpResult(status=404, body=b"")


def _cfg() -> VerbatimConfig:
    return VerbatimConfig(
        mode=Mode.REMOTE_ASSISTED,
        judge=JudgeConfig(backend="jev", daily_budget_usd=Decimal("1.00")),
        embedding=EmbeddingConfig(backend="cloudflare", account_id="acct-1"),
    )


def _store() -> TestStore:
    store = TestStore()
    with store.read() as conn:  # autocommit — executescript must not sit in tx
        conn.executescript(DDL_V2)
        for stmt in _split(DDL_V4):
            if "dispatch_permits" in stmt:
                conn.execute(stmt)
    return store


def _broker(store, clock) -> TransportBroker:
    enc = _cf(broker=None)
    return TransportBroker(
        store,
        _cfg(),
        endpoints=[enc.endpoint_descriptor()],
        clock=clock,
    )


def _cf(*, broker, transport: Optional[_FakeTransport] = None) -> CloudflareEncoder:
    return CloudflareEncoder(
        EmbeddingConfig(backend="cloudflare", model="m", account_id="acct-1"),
        account_id="acct-1",
        secret_getter=lambda name: "tok",
        http=transport or _FakeTransport(),
        broker=broker,
    )


def _permit_for(broker, store, enc, texts, scope_id, purpose="embed_document"):
    return broker.open_dispatch(
        store,
        caller="test",
        recipient=enc.endpoint_id,
        purpose=purpose,
        payload_digest=enc.payload_digest(texts),
        scope_ids=[scope_id],
        max_spend=0.001,
    )


# ---------------------------------------------------------------------------
# CloudflareEncoder
# ---------------------------------------------------------------------------


def test_cloudflare_encode_without_broker_never_dispatches():
    """An encoder with no broker attached can never reach the transport —
    the unsafe pre-F4-04 path is closed by construction."""
    transport = _FakeTransport()
    enc = _cf(broker=None, transport=transport)
    with pytest.raises(VerbatimError) as ei:
        enc.encode(["secret text"])
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    assert transport.calls == []


def test_cloudflare_encode_without_permit_never_dispatches():
    store, clock = _store(), FakeClock()
    broker = _broker(store, clock)
    transport = _FakeTransport()
    enc = _cf(broker=broker, transport=transport)
    with pytest.raises(VerbatimError) as ei:
        enc.encode(["secret text"])
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    assert transport.calls == []


def test_cloudflare_encode_with_forged_permit_denied():
    """A caller-constructed DispatchPermit (no broker issuance, no ledger
    row) is rejected — the broker re-reads durable state, not the token."""
    store, clock = _store(), FakeClock()
    broker = _broker(store, clock)
    transport = _FakeTransport()
    enc = _cf(broker=broker, transport=transport)
    forged = DispatchPermit(
        permit_id="forged",
        recipient=enc.endpoint_id,
        purpose="embed_document",
        payload_digest=enc.payload_digest(["x"]),
        scope_ids=("s",),
        consent_refs=(),
        reservation_id="r",
        max_spend=1.0,
        issued_us=clock.t,
        expires_us=clock.t + 1_000_000,
    )
    with pytest.raises(VerbatimError) as ei:
        enc.encode(["x"], permit=forged)
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    assert transport.calls == []


def test_cloudflare_encode_with_valid_permit_dispatches_once():
    store, clock = _store(), FakeClock()
    broker = _broker(store, clock)
    grant_consent(store, "scope-1", processor=CF_ID, purpose="embed_document")
    body = json.dumps({"result": {"data": [[1.0, 0.0]]}, "success": True}).encode()
    transport = _FakeTransport(HttpResult(status=200, body=body))
    enc = _cf(broker=broker, transport=transport)

    permit = _permit_for(broker, store, enc, ["hello"], "scope-1")
    blobs = enc.encode(["hello"], permit=permit)
    assert len(blobs) == 1
    assert len(transport.calls) == 1
    assert json.loads(transport.calls[0]["body"])["input"]["text"] == ["hello"]
    assert broker.permit(permit.permit_id).state == "dispatched"

    # the same permit cannot carry a second request (C18)
    transport2 = _FakeTransport(HttpResult(status=200, body=body))
    enc2 = _cf(broker=broker, transport=transport2)
    with pytest.raises(VerbatimError) as ei:
        enc2.encode(["hello"], permit=permit)
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    assert transport2.calls == []


def test_cloudflare_permit_for_other_payload_denied():
    store, clock = _store(), FakeClock()
    broker = _broker(store, clock)
    grant_consent(store, "scope-1", processor=CF_ID, purpose="embed_document")
    transport = _FakeTransport()
    enc = _cf(broker=broker, transport=transport)
    permit = _permit_for(broker, store, enc, ["hello"], "scope-1")
    with pytest.raises(VerbatimError) as ei:
        enc.encode(["DIFFERENT TEXT"], permit=permit)
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    assert transport.calls == []
    # digest mismatch does not consume the permit
    assert broker.permit(permit.permit_id).state == "open"


def test_cloudflare_permit_for_other_recipient_denied():
    """A permit minted for a different endpoint cannot dispatch here."""
    store, clock = _store(), FakeClock()
    other = EndpointDescriptor(
        endpoint_id=OL_ID, origin="http://127.0.0.1:11434", require_tls=False
    )
    broker = TransportBroker(
        store, _cfg(), endpoints=[_cf(broker=None).endpoint_descriptor(), other],
        clock=clock,
    )
    grant_consent(store, "scope-1", processor=OL_ID, purpose="embed_document")
    transport = _FakeTransport()
    enc = _cf(broker=broker, transport=transport)
    # permit is bound to the ollama endpoint, not this encoder's
    permit = broker.open_dispatch(
        store,
        caller="test",
        recipient=OL_ID,
        purpose="embed_document",
        payload_digest=enc.payload_digest(["hello"]),
        scope_ids=["scope-1"],
        max_spend=0.001,
    )
    with pytest.raises(VerbatimError) as ei:
        enc.encode(["hello"], permit=permit)
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    assert transport.calls == []


# ---------------------------------------------------------------------------
# OllamaEncoder — loopback is still egress
# ---------------------------------------------------------------------------


def _ollama(*, broker, transport: Optional[_FakeTransport] = None) -> OllamaEncoder:
    return OllamaEncoder(
        EmbeddingConfig(backend="ollama", model="m"),
        http=transport or _FakeTransport(),
        broker=broker,
    )


def test_ollama_encode_without_permit_denied():
    store, clock = _store(), FakeClock()
    desc = _ollama(broker=None).endpoint_descriptor()
    broker = TransportBroker(store, _cfg(), endpoints=[desc], clock=clock)
    transport = _FakeTransport()
    enc = _ollama(broker=broker, transport=transport)
    with pytest.raises(VerbatimError) as ei:
        enc.encode(["loopback text"])
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    assert transport.calls == []


def test_ollama_encode_with_permit_dispatches():
    store, clock = _store(), FakeClock()
    desc = _ollama(broker=None).endpoint_descriptor()
    broker = TransportBroker(store, _cfg(), endpoints=[desc], clock=clock)
    grant_consent(store, "scope-1", processor=OL_ID, purpose="embed_document")
    body = json.dumps({"embeddings": [[1.0, 0.0]]}).encode()
    transport = _FakeTransport(HttpResult(status=200, body=body))
    enc = _ollama(broker=broker, transport=transport)
    permit = _permit_for(broker, store, enc, ["hello"], "scope-1")
    blobs = enc.encode(["hello"], permit=permit)
    assert len(blobs) == 1
    assert len(transport.calls) == 1


# ---------------------------------------------------------------------------
# local encoders + factory plumbing
# ---------------------------------------------------------------------------


def test_local_encoder_needs_no_permit():
    """HashingEncoder performs no transport — no permit, no broker."""
    enc = HashingEncoder(EmbeddingConfig(backend="hashing"))
    assert not encoder_requires_permit(enc)
    assert enc.encode(["x"])  # works permit-free


def test_remote_marker_and_factory_broker():
    cfg = _cfg()
    store, clock = _store(), FakeClock()
    broker = _broker(store, clock)
    enc = get_encoder(
        cfg, http=_FakeTransport(), secret_getter=lambda n: "tok", broker=broker
    )
    assert isinstance(enc, CloudflareEncoder)
    assert encoder_requires_permit(enc)
    assert enc._broker is broker
    # unbrokered factory output is fail-closed
    enc2 = get_encoder(cfg, http=_FakeTransport(), secret_getter=lambda n: "tok")
    with pytest.raises(VerbatimError) as ei:
        enc2.encode(["x"])
    assert ei.value.code == ErrorCode.EGRESS_DENIED


def test_offline_profile_no_endpoints_no_dispatch():
    """V4-12.09: with no endpoints configured (offline profile) nothing can
    be minted — process-level denial, not a mocked transport."""
    store, clock = _store(), FakeClock()
    off_cfg = VerbatimConfig(
        mode=Mode.OFFLINE_RULES,
        judge=JudgeConfig(backend="rules", daily_budget_usd=Decimal("0")),
    )
    broker = TransportBroker(store, off_cfg, endpoints=[], clock=clock)
    transport = _FakeTransport()
    enc = _cf(broker=broker, transport=transport)
    with pytest.raises(VerbatimError) as ei:
        enc.encode(["x"], permit="whatever")  # type: ignore[arg-type]
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    assert transport.calls == []


def test_payload_digest_stability():
    d1 = payload_digest(b"abc")
    d2 = payload_digest("abc")
    assert d1 == d2 and d1.startswith("sha256:")
    assert payload_digest(b"abd") != d1
