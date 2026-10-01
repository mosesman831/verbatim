"""TransportBroker tests — the single egress authority (SPEC_V4 §12, F4-04).

Coverage: permit issuance requires consent+scope coverage (V4-12.06), zero
budget denies even nominally-free calls (V4-12.04), expired/replayed/
wrong-recipient/digest-mismatched permits cannot dispatch (C18, V4-12.02/03),
reservations survive a store reopen (V4-12.05/10), endpoint descriptors are
allowlist-validated (V4-12.07), and the offline profile denies all dispatch.

The v4 ``dispatch_permits`` table and the v2 ``disclosures`` table are
applied on top of the conftest v1 store — the same DDL the real Store runs.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from verbatim.config import EmbeddingConfig, JudgeConfig, VerbatimConfig
from verbatim.core.types import ErrorCode, Mode, Scope, VerbatimError
from verbatim.privacy.broker import (
    EndpointDescriptor,
    TransportBroker,
    payload_digest,
)
from verbatim.storage.schema import DDL_V2
from verbatim.storage.schema_v4 import DDL_V4, _split
from tests.conftest import grant_consent, qrow

# The endpoint identities the tests grant consent under — the descriptor's
# consent_processor defaults to endpoint_id.
CF_ID = "cloudflare:api.cloudflare.com"
OL_ID = "ollama:127.0.0.1:11434"
PURPOSE = "embed_document"

CF_DESC = EndpointDescriptor(
    endpoint_id=CF_ID,
    origin="https://api.cloudflare.com",
    account="acct-1",
)
OL_DESC = EndpointDescriptor(
    endpoint_id=OL_ID,
    origin="http://127.0.0.1:11434",
    require_tls=False,
)


def _v4_store(store):
    """Apply the v4 permit table + v2 receipt tables to the conftest store.

    Runs through ``read()`` (autocommit) — ``executescript`` issues an
    implicit COMMIT and must not sit inside a BEGIN IMMEDIATE block.
    """
    with store.read() as conn:
        conn.executescript(DDL_V2)
        for stmt in _split(DDL_V4):
            if "dispatch_permits" in stmt:
                conn.execute(stmt)
    return store


def _cfg(mode=Mode.REMOTE_ASSISTED, budget="1.00"):
    return VerbatimConfig(
        mode=mode,
        judge=JudgeConfig(backend="jev", daily_budget_usd=Decimal(budget)),
        embedding=EmbeddingConfig(backend="cloudflare", account_id="acct-1"),
    )


def _broker(store, clock, *, endpoints=(CF_DESC, OL_DESC), cfg=None, **kw):
    return TransportBroker(
        store,
        cfg or _cfg(),
        endpoints=endpoints,
        clock=clock,
        **kw,
    )


@pytest.fixture()
def broker(store, clock):
    return _broker(_v4_store(store), clock)


def _digest(*texts: str) -> str:
    return payload_digest(json.dumps({"input": list(texts)}).encode())


def _open(broker, store, scope_id, **kw):
    args = dict(
        caller="ingest.embed",
        recipient=CF_ID,
        purpose=PURPOSE,
        payload_digest=_digest("hello"),
        scope_ids=[scope_id],
        max_spend=0.001,
        est_tokens=4,
    )
    args.update(kw)
    return broker.open_dispatch(store, **args)


# ---------------------------------------------------------------------------
# issuance — consent, scopes, budget (V4-12.02/04/06)
# ---------------------------------------------------------------------------


def test_open_dispatch_mints_permit_and_reservation(store, clock, broker, scope_id):
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    permit = _open(broker, store, scope_id)
    assert permit.state == "open"
    assert permit.recipient == CF_ID
    assert permit.scope_ids == (scope_id,)
    assert permit.consent_refs  # verified consent ids bound at issuance
    assert permit.expires_us == permit.issued_us + broker.DEFAULT_MAX_PERMIT_AGE_US
    with store.read() as conn:
        row = qrow(
            conn,
            "SELECT * FROM dispatch_permits WHERE permit_id = ?",
            (permit.permit_id,),
        )
        res = qrow(
            conn,
            "SELECT * FROM budget_ledger WHERE reservation_id = ?",
            (permit.reservation_id,),
        )
    assert row["state"] == "open"
    assert row["payload_digest"] == permit.payload_digest
    assert res["state"] == "reserved"
    assert res["reserved_cost_microusd"] == 1000  # ceil(0.001 * 1e6)


def test_open_dispatch_requires_consent_for_every_scope(store, clock, broker, scope_id):
    """V4-12.06: a multi-scope payload needs consent on EVERY scope — the
    consented scope cannot carry the unconsented one."""
    other = "scope-b"
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    with pytest.raises(VerbatimError) as ei:
        _open(broker, store, scope_id, scope_ids=[scope_id, other])
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    grant_consent(store, other, processor=CF_ID, purpose=PURPOSE)
    permit = _open(broker, store, scope_id, scope_ids=[scope_id, other])
    assert set(permit.scope_ids) == {scope_id, other}
    assert len(permit.consent_refs) == 2


def test_open_dispatch_wrong_purpose_or_recipient_consent(store, clock, broker, scope_id):
    # consent under a different purpose must not authorize this dispatch
    grant_consent(store, scope_id, processor=CF_ID, purpose="query_rerank")
    with pytest.raises(VerbatimError) as ei:
        _open(broker, store, scope_id)
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    # consent keyed to a different processor doesn't cover this endpoint
    grant_consent(store, scope_id, processor="someone-else", purpose=PURPOSE)
    with pytest.raises(VerbatimError) as ei:
        _open(broker, store, scope_id)
    assert ei.value.code == ErrorCode.EGRESS_DENIED


def test_zero_budget_denies_even_nominally_free(store, clock, scope_id):
    """V4-12.04: zero remote budget prohibits dispatch — a 'free' request
    is still a dispatch and still needs a positive reservation."""
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    broke = _broker(_v4_store(store), clock, cfg=_cfg(budget="0"))
    with pytest.raises(VerbatimError) as ei:
        _open(broke, store, scope_id)
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    # and a zero max_spend bound is refused even WITH budget
    ok = _broker(_v4_store(store), clock)
    with pytest.raises(VerbatimError) as ei:
        _open(ok, store, scope_id, max_spend=0.0)
    assert ei.value.code == ErrorCode.EGRESS_DENIED


def test_budget_room_enforced(store, clock, scope_id):
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    tiny = _broker(_v4_store(store), clock, cfg=_cfg(budget="0.000002"))  # 2µ$/day
    with pytest.raises(VerbatimError) as ei:
        _open(tiny, store, scope_id, max_spend=0.000003)  # 3µ$ > 2µ$ ceiling
    assert ei.value.code == ErrorCode.BUDGET_EXHAUSTED
    permit = _open(tiny, store, scope_id, max_spend=0.000002)  # fits exactly
    assert permit.state == "open"
    with pytest.raises(VerbatimError) as ei:
        _open(tiny, store, scope_id, max_spend=0.000002)  # day committed
    assert ei.value.code == ErrorCode.BUDGET_EXHAUSTED


def test_recipient_allowlist(store, clock, broker, scope_id):
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    with pytest.raises(VerbatimError) as ei:
        _open(broker, store, scope_id, recipient="evil.example.com")
    assert ei.value.code == ErrorCode.EGRESS_DENIED


def test_consent_refs_must_be_verified(store, clock, broker, scope_id):
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    with pytest.raises(VerbatimError) as ei:
        _open(broker, store, scope_id, consent_refs=["consent-fabricated"])
    assert ei.value.code == ErrorCode.EGRESS_DENIED


def test_offline_mode_denies_all_dispatch(store, clock, scope_id):
    """Offline profiles: no mode covers egress, so every dispatch is denied
    — and an empty allowlist denies even in remote-capable modes."""
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    off = _broker(
        _v4_store(store), clock, cfg=_cfg(mode=Mode.OFFLINE_RULES, budget="1.00")
    )
    with pytest.raises(VerbatimError) as ei:
        _open(off, store, scope_id)
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    # even a loopback endpoint is unreachable in offline mode
    with pytest.raises(VerbatimError) as ei:
        _open(off, store, scope_id, recipient=OL_ID)
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    # empty allowlist: nothing can be minted regardless of mode
    no_eps = _broker(_v4_store(store), clock, endpoints=())
    with pytest.raises(VerbatimError) as ei:
        _open(no_eps, store, scope_id)
    assert ei.value.code == ErrorCode.EGRESS_DENIED


def test_dispatch_permits_table_missing_fails_closed(store, clock, scope_id):
    """A store without the v4 ledger cannot mint permits — an unrecordable
    dispatch is a denied dispatch."""
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    broker = _broker(store, clock)  # no _v4_store() — table absent
    with pytest.raises(VerbatimError) as ei:
        _open(broker, store, scope_id)
    assert ei.value.code == ErrorCode.EGRESS_DENIED


# ---------------------------------------------------------------------------
# dispatch — one-use, expiry, binding (C18, V4-12.03)
# ---------------------------------------------------------------------------


def test_dispatch_consumes_permit_once(store, clock, broker, scope_id):
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    permit = _open(broker, store, scope_id)
    out = broker.dispatch(
        permit, recipient=CF_ID, payload_digest=permit.payload_digest
    )
    assert out.state == "dispatched"
    # replay — same permit, correct bindings — is denied (C18)
    with pytest.raises(VerbatimError) as ei:
        broker.dispatch(
            permit, recipient=CF_ID, payload_digest=permit.payload_digest
        )
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    # mark_dispatched agrees
    with pytest.raises(VerbatimError):
        broker.mark_dispatched(permit.permit_id)


def test_expired_permit_cannot_dispatch(store, clock, broker, scope_id):
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    permit = _open(broker, store, scope_id)
    clock.advance(2.0)  # past the 1s default permit age
    with pytest.raises(VerbatimError) as ei:
        broker.dispatch(permit)
    assert ei.value.code == ErrorCode.PERMIT_EXPIRED
    assert broker.permit(permit.permit_id).state == "expired"
    with store.read() as conn:
        res = qrow(
            conn,
            "SELECT state, actual_cost_microusd, reserved_cost_microusd"
            " FROM budget_ledger WHERE reservation_id = ?",
            (permit.reservation_id,),
        )
    # timeout accounting: full conservative charge until reconciled (V4-12.05)
    assert res["state"] == "expired"
    assert res["actual_cost_microusd"] == res["reserved_cost_microusd"]


def test_wrong_recipient_permit_denied(store, clock, broker, scope_id):
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    permit = _open(broker, store, scope_id)
    with pytest.raises(VerbatimError) as ei:
        broker.dispatch(permit, recipient=OL_ID)
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    # binding failure does NOT consume the permit
    assert broker.permit(permit.permit_id).state == "open"


def test_payload_digest_mismatch_denied(store, clock, broker, scope_id):
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    permit = _open(broker, store, scope_id)
    with pytest.raises(VerbatimError) as ei:
        broker.dispatch(permit, payload_digest=_digest("DIFFERENT BODY"))
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    assert broker.permit(permit.permit_id).state == "open"


def test_dispatch_rejects_foreign_token(store, clock, broker, scope_id):
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    with pytest.raises(VerbatimError) as ei:
        broker.dispatch("not-a-permit")  # type: ignore[arg-type]
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    with pytest.raises(VerbatimError) as ei:
        broker.dispatch(None)  # type: ignore[arg-type]
    assert ei.value.code == ErrorCode.EGRESS_DENIED


def test_consent_revoked_between_issue_and_dispatch(store, clock, broker, scope_id):
    """V4-12.03: consent is rechecked at dispatch — revocation in the gap
    denies the handoff and releases the reservation at zero."""
    cid = grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    permit = _open(broker, store, scope_id)
    with store.tx() as conn:
        conn.execute(
            "UPDATE consents SET revoked_us = 1 WHERE consent_id = ?", (cid,)
        )
    with pytest.raises(VerbatimError) as ei:
        broker.dispatch(permit)
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    assert broker.permit(permit.permit_id).state == "denied"
    with store.read() as conn:
        res = qrow(
            conn,
            "SELECT state, actual_cost_microusd FROM budget_ledger"
            " WHERE reservation_id = ?",
            (permit.reservation_id,),
        )
    assert res["state"] == "settled" and res["actual_cost_microusd"] == 0


def test_purge_suppression_blocks_dispatch_recheck(store, clock, broker, scope_id):
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    permit = _open(broker, store, scope_id)
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO purges (purge_id, selection_digest, scope_id, state,"
            " requested_us) VALUES ('p1', ?, ?, 'suppressed', 1)",
            (b"\x00" * 32, scope_id),
        )
    with pytest.raises(VerbatimError) as ei:
        broker.dispatch(permit)
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    assert broker.permit(permit.permit_id).state == "denied"


def test_deadline_caps_permit_expiry(store, clock, broker, scope_id):
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    now = clock.t
    permit = _open(broker, store, scope_id, deadline_us=now + 300_000)
    assert permit.expires_us == now + 300_000  # earlier than the 1s default
    clock.advance(0.4)
    with pytest.raises(VerbatimError) as ei:
        broker.dispatch(permit)
    assert ei.value.code == ErrorCode.PERMIT_EXPIRED


def test_caller_conn_permit_atomic_with_caller_tx(store, clock, broker, scope_id):
    """open_dispatch on a caller conn writes inside THAT transaction — the
    permit+reservation roll back with it (atomic, V4-12.05)."""
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    with pytest.raises(RuntimeError):
        with store.tx() as conn:
            permit = broker.open_dispatch(
                conn,
                caller="coordinator",
                recipient=CF_ID,
                purpose=PURPOSE,
                payload_digest=_digest("x"),
                scope_ids=[scope_id],
                max_spend=0.001,
            )
            raise RuntimeError("abort caller tx")
    # rolled back — no permit row, no reservation
    with pytest.raises(VerbatimError):
        broker.permit(permit.permit_id)
    assert broker.gate.day_spend() == 0


# ---------------------------------------------------------------------------
# reconcile — settlement + crash correlation (V4-12.05/10)
# ---------------------------------------------------------------------------


def test_reconcile_settles_and_closes_permit(store, clock, broker, scope_id):
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    permit = _open(broker, store, scope_id)
    broker.dispatch(permit)
    state = broker.reconcile(permit.reservation_id, 0.0004, "settled")
    assert state == "settled"
    assert broker.permit(permit.permit_id).state == "reconciled"
    # idempotent — never double-debits
    assert broker.reconcile(permit.reservation_id, 9.0, "settled") == "settled"
    with store.read() as conn:
        res = qrow(
            conn,
            "SELECT actual_cost_microusd FROM budget_ledger WHERE reservation_id = ?",
            (permit.reservation_id,),
        )
    assert res["actual_cost_microusd"] == 400


def test_reconcile_overrun_counts_full_actual(store, clock, broker, scope_id):
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    permit = _open(broker, store, scope_id)
    broker.dispatch(permit)
    state = broker.reconcile(permit.reservation_id, 0.005, "settled")
    # caller labels don't matter — actual > reserved is always overrun
    assert state == "overrun"
    assert broker.permit(permit.permit_id).state == "reconciled"


def test_reconcile_expired_charges_full(store, clock, broker, scope_id):
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    permit = _open(broker, store, scope_id)
    broker.dispatch(permit)
    state = broker.reconcile(permit.reservation_id, 0.0, "timeout")
    assert state == "expired"
    assert broker.permit(permit.permit_id).state == "expired"


def test_reconcile_failed_settles_at_actual(store, clock, broker, scope_id):
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    permit = _open(broker, store, scope_id)
    broker.dispatch(permit)
    state = broker.reconcile(permit.reservation_id, 0.0, "failed")
    assert state == "settled"
    with store.read() as conn:
        res = qrow(
            conn,
            "SELECT actual_cost_microusd FROM budget_ledger WHERE reservation_id = ?",
            (permit.reservation_id,),
        )
    assert res["actual_cost_microusd"] == 0


def test_reconcile_unknown_reservation(store, clock, broker):
    with pytest.raises(VerbatimError) as ei:
        broker.reconcile("nope", 0.0, "settled")
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


def test_expire_stale_sweeps_open_permits(store, clock, broker, scope_id):
    grant_consent(store, scope_id, processor=CF_ID, purpose=PURPOSE)
    p1 = _open(broker, store, scope_id)
    p2 = _open(broker, store, scope_id)
    broker.dispatch(p2)  # consumed permits are NOT swept
    clock.advance(2.0)
    assert broker.expire_stale() == 1
    assert broker.permit(p1.permit_id).state == "expired"
    assert broker.permit(p2.permit_id).state == "dispatched"


# ---------------------------------------------------------------------------
# crash persistence (V4-12.10) — permit→request correlation across reopen
# ---------------------------------------------------------------------------


def test_permit_and_reservation_survive_reopen(tmp_path, clock):
    """A real file store: mint a permit, close, reopen — the pending
    reservation still holds and the permit can still be consumed exactly
    once (crash correlation without retaining payload text)."""
    from verbatim.storage.repos import ConsentsRepo
    from verbatim.storage.store import Store

    path = str(tmp_path / "v4.db")
    store = Store.create(path)
    sid = "scope-file-1"
    with store.tx() as conn:
        ConsentsRepo(store).grant(conn, sid, CF_ID, PURPOSE, "digest-1")
    broker = _broker(store, clock)
    permit = _open(broker, store, sid)
    store.close()

    reopened = Store.open(path)
    try:
        broker2 = _broker(reopened, clock)
        persisted = broker2.permit(permit.permit_id)
        assert persisted.state == "open"
        assert persisted.reservation_id == permit.reservation_id
        # reservation still counts against the day
        assert broker2.gate.day_spend() == 1000
        out = broker2.dispatch(
            persisted,
            recipient=CF_ID,
            payload_digest=persisted.payload_digest,
        )
        assert out.state == "dispatched"
        with pytest.raises(VerbatimError):
            broker2.dispatch(persisted)  # replay across instances — denied
    finally:
        reopened.close()


# ---------------------------------------------------------------------------
# descriptors (V4-12.07)
# ---------------------------------------------------------------------------


class TestEndpointDescriptor:
    @pytest.mark.parametrize(
        "kw",
        [
            dict(origin="http://api.example.com"),          # plain http remote
            dict(origin="https://h.com/path"),              # path
            dict(origin="https://h.com?q=1"),               # query
            dict(origin="https://user:pw@h.com"),           # embedded creds
            dict(origin="ftp://h.com"),                     # scheme
            dict(origin="https://h.com", require_tls=False),# disable verify on https
        ],
    )
    def test_rejects_unsafe_origins(self, kw):
        with pytest.raises(VerbatimError) as ei:
            EndpointDescriptor(endpoint_id="ep-1", **kw)
        assert ei.value.code == ErrorCode.CONFIG_INVALID

    def test_loopback_http_requires_explicit_no_tls(self):
        with pytest.raises(VerbatimError):
            EndpointDescriptor(
                endpoint_id="ep-1", origin="http://127.0.0.1:11434"
            )  # require_tls defaults True → http rejected
        ok = EndpointDescriptor(
            endpoint_id="ep-1", origin="http://127.0.0.1:11434", require_tls=False
        )
        assert ok.is_loopback

    def test_redirect_plus_credential_forwarding_rejected(self):
        with pytest.raises(VerbatimError) as ei:
            EndpointDescriptor(
                endpoint_id="ep-1",
                origin="https://h.com",
                allow_redirects=True,
                forward_credentials=True,
            )
        assert ei.value.code == ErrorCode.CONFIG_INVALID

    def test_duplicate_endpoint_ids_rejected(self, store, clock):
        with pytest.raises(VerbatimError) as ei:
            _broker(store, clock, endpoints=[CF_DESC, CF_DESC])
        assert ei.value.code == ErrorCode.CONFIG_INVALID


# ---------------------------------------------------------------------------
# encode_permitted seam — the call-site integration helper (F4-04)
# ---------------------------------------------------------------------------


class _FakeTransport:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def request(self, method, url, *, body, headers, timeout_s):
        self.calls.append({"method": method, "url": url, "body": body})
        return self._result


def _cf_encoder(broker, transport):
    from verbatim.embeddings.cloudflare import CloudflareEncoder

    return CloudflareEncoder(
        EmbeddingConfig(backend="cloudflare", model="m", account_id="acct-1"),
        account_id="acct-1",
        secret_getter=lambda name: "tok",
        http=transport,
        broker=broker,
    )


def test_encode_permitted_full_path(store, clock, broker, scope_id):
    """The documented call-site seam: one call mints, dispatches, encodes,
    and reconciles — the transport only runs under a consumed permit."""
    from verbatim.embeddings.cloudflare import HttpResult
    from verbatim.embeddings.codec import Float32Codec

    grant_consent(store, scope_id, processor=CF_ID, purpose="embed_document")
    body = json.dumps({"result": {"data": [[1.0, 0.0]]}, "success": True}).encode()
    transport = _FakeTransport(HttpResult(status=200, body=body))
    enc = _cf_encoder(broker, transport)
    blobs = broker.encode_permitted(
        enc, ["hello"], scope_ids=[scope_id], purpose="embed_document",
        caller="test",
    )
    assert len(blobs) == 1
    assert Float32Codec.unpack(blobs[0], 2) == pytest.approx((1.0, 0.0))
    assert len(transport.calls) == 1
    # permit consumed + reconciled, reservation settled
    rows = []
    with store.read() as conn:
        rows.append(
            qrow(conn, "SELECT state FROM dispatch_permits")
        )
        res = qrow(conn, "SELECT state FROM budget_ledger")
    assert rows[0]["state"] == "reconciled"
    assert res["state"] == "settled"


def test_encode_permitted_denies_without_consent(store, clock, broker, scope_id):
    from verbatim.embeddings.cloudflare import HttpResult

    body = json.dumps({"result": {"data": [[1.0]]}, "success": True}).encode()
    transport = _FakeTransport(HttpResult(status=200, body=body))
    enc = _cf_encoder(broker, transport)
    with pytest.raises(VerbatimError) as ei:
        broker.encode_permitted(
            enc, ["hello"], scope_ids=[scope_id], purpose="embed_document",
            caller="test",
        )
    assert ei.value.code == ErrorCode.EGRESS_DENIED
    assert transport.calls == []  # C14: never reaches the transport


def test_encode_permitted_transport_failure_retains_reservation(
    store, clock, broker, scope_id
):
    """A retryable transport error expires the reservation at the full
    estimate — the provider may have done the work (V4-12.05)."""
    grant_consent(store, scope_id, processor=CF_ID, purpose="embed_document")

    class _Fail:
        def request(self, *a, **k):
            raise VerbatimError(ErrorCode.ENCODER_UNAVAILABLE, "timeout", retryable=True)

    enc = _cf_encoder(broker, _Fail())
    with pytest.raises(VerbatimError):
        broker.encode_permitted(
            enc, ["hello"], scope_ids=[scope_id], purpose="embed_document",
            caller="test",
        )
    with store.read() as conn:
        res = qrow(conn, "SELECT state, actual_cost_microusd,"
                   " reserved_cost_microusd FROM budget_ledger")
        perm = qrow(conn, "SELECT state FROM dispatch_permits")
    assert res["state"] == "expired"
    assert res["actual_cost_microusd"] == res["reserved_cost_microusd"]
    assert perm["state"] == "expired"
