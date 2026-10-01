"""X7 — session-to-durable promotion consent gate (SPEC_V4_5 §09 X7,
V45-09.07; acceptance D19; parent binding V3-11.11).

Session working memory is TTL-bound and temporary by default. It becomes
durable ONLY through ``working.promote`` under an explicit, persisted,
live ``capture_authorizations`` row — retention consent. An approved
tool call is not that record (V3-11.11): no grant, no implicit session
access, and no in-memory claim substitutes for the persisted row.

Real ``Store.create`` fixtures; ``promote`` runs inside the caller's tx
so denials leave zero durable rows.
"""

from __future__ import annotations

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v3 import CaptureAuthorization, EnvelopeKind
from verbatim.governance import (
    create_grant,
    issue_capture_authorization,
    register_principal,
    revoke_capture_authorization,
)
from verbatim.observations import working
from verbatim.storage.store import Store

SCOPE = "scopeA"
SESSION = "sess-1"
PRINCIPAL = "alice"


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "w7.db"))
    yield s
    s.close()


def _provision(store, *, consent=True, auth_kinds=None, auth_scopes=None,
               auth_principal=PRINCIPAL, expires_us=None):
    """Scope + principal + ingest grant (tool permission!) + optionally
    a persisted retention-consent row. Returns the authorization_id."""
    aid = None
    with store.tx() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO scopes"
            "(scope_id,profile_id,principal_id,visibility)"
            " VALUES(?,?,?,'owner')",
            (SCOPE, "v3", PRINCIPAL),
        )
        register_principal(conn, kind="human", principal_id=PRINCIPAL)
        create_grant(
            conn, scope_id=SCOPE, principal_id=PRINCIPAL,
            verbs={"ingest"}, issuer_id=PRINCIPAL,
        )
        if consent:
            aid = issue_capture_authorization(
                conn,
                principal_id=auth_principal,
                issuer_id=PRINCIPAL,
                allowed_kinds=auth_kinds or {EnvelopeKind.SYSTEM_EVENT},
                retention_policy="keep",
                policy_revision="pol1",
                scope_ids=(auth_scopes if auth_scopes is not None
                           else {SCOPE}),
                expires_us=expires_us,
            )
    return aid


def _auth_obj(aid, *, principal=PRINCIPAL, kinds=None, scopes=None,
              expires_us=None):
    """A CaptureAuthorization matching the persisted row."""
    return CaptureAuthorization(
        authorization_id=aid,
        issuer_id=PRINCIPAL,
        principal_id=principal,
        allowed_kinds=kinds or {EnvelopeKind.SYSTEM_EVENT},
        scope_ids=scopes if scopes is not None else {SCOPE},
        retention_policy="keep",
        policy_revision="pol1",
        issued_us=1,
        expires_us=expires_us,
    )


def _set_with_text(store, texts=("session scratch note",), *,
                   scope=SCOPE, session=SESSION):
    with store.tx() as conn:
        sid = working.create_set(conn, scope, session, 60_000_000)
        for t in texts:
            working.add_item(conn, sid, "decision", text=t)
        return sid


def _count(conn, sql, params=()):
    return conn.execute(sql, params).fetchone()[0]


# ---------------------------------------------------------------------------
# D19 — denial paths: no consent, no durable write
# ---------------------------------------------------------------------------


def test_promotion_requires_retention_consent(store):
    """D19: absent any authorization record the promotion denies with
    CONSENT_REQUIRED and writes nothing durable."""
    _provision(store, consent=False)
    sid = _set_with_text(store)
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as ei:
            working.promote(
                conn, store, SCOPE, SESSION,
                principal_id=PRINCIPAL, authorization=None,
            )
        assert ei.value.code is ErrorCode.CONSENT_REQUIRED
        # Nothing durable — no sources, no envelopes, no events.
        assert _count(conn, "SELECT COUNT(*) FROM source_envelopes") == 0
        assert _count(conn, "SELECT COUNT(*) FROM sources") == 0
        # The working set itself is untouched.
        assert working.item_count(conn, sid) == 1


def test_tool_permission_is_not_consent(store):
    """V3-11.11: an ingest grant — tool permission — never substitutes
    for a retention-consent row."""
    _provision(store, consent=False)  # grant exists, consent does not
    _set_with_text(store)
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as ei:
            working.promote(
                conn, store, SCOPE, SESSION,
                principal_id=PRINCIPAL,
                authorization=_auth_obj("cauth:forged"),
            )
        assert ei.value.code is ErrorCode.CONSENT_REQUIRED
        assert _count(conn, "SELECT COUNT(*) FROM sources") == 0


def test_expired_revoked_consent_denies(store):
    _provision(store)
    sid = _set_with_text(store)
    # Expired authorization.
    aid_exp = issue_capture_authorization  # placeholder for clarity
    with store.tx() as conn:
        expired_id = issue_capture_authorization(
            conn,
            principal_id=PRINCIPAL,
            issuer_id=PRINCIPAL,
            allowed_kinds={EnvelopeKind.SYSTEM_EVENT},
            retention_policy="keep",
            policy_revision="pol1",
            scope_ids={SCOPE},
            issued_us=100,
            expires_us=200,
        )
        with pytest.raises(VerbatimError) as ei:
            working.promote(
                conn, store, SCOPE, SESSION,
                principal_id=PRINCIPAL,
                authorization=_auth_obj(expired_id, expires_us=200),
                now=300,
            )
        assert ei.value.code is ErrorCode.CONSENT_REQUIRED
    # Revoked authorization.
    aid = None
    with store.tx() as conn:
        aid = issue_capture_authorization(
            conn,
            principal_id=PRINCIPAL,
            issuer_id=PRINCIPAL,
            allowed_kinds={EnvelopeKind.SYSTEM_EVENT},
            retention_policy="keep",
            policy_revision="pol1",
            scope_ids={SCOPE},
        )
        revoke_capture_authorization(conn, aid)
        with pytest.raises(VerbatimError) as ei:
            working.promote(
                conn, store, SCOPE, SESSION,
                principal_id=PRINCIPAL,
                authorization=_auth_obj(aid),
            )
        assert ei.value.code is ErrorCode.CONSENT_REQUIRED
        assert _count(conn, "SELECT COUNT(*) FROM sources") == 0


def test_wrong_principal_scope_kind_all_deny_identically(store):
    _provision(store)
    sid = _set_with_text(store)
    cases = []
    with store.tx() as conn:
        aid = issue_capture_authorization(
            conn,
            principal_id="bob",           # consent issued for bob…
            issuer_id=PRINCIPAL,
            allowed_kinds={EnvelopeKind.SYSTEM_EVENT},
            retention_policy="keep",
            policy_revision="pol1",
            scope_ids={SCOPE},
        )
        # …does not authorize alice's session.
        with pytest.raises(VerbatimError) as ei:
            working.promote(
                conn, store, SCOPE, SESSION,
                principal_id=PRINCIPAL,
                authorization=_auth_obj(aid, principal="bob"),
            )
        assert ei.value.code is ErrorCode.CONSENT_REQUIRED
        # Kind not covered by the consent.
        aid2 = issue_capture_authorization(
            conn,
            principal_id=PRINCIPAL,
            issuer_id=PRINCIPAL,
            allowed_kinds={EnvelopeKind.USER_MESSAGE},
            retention_policy="keep",
            policy_revision="pol1",
            scope_ids={SCOPE},
        )
        with pytest.raises(VerbatimError) as ei:
            working.promote(
                conn, store, SCOPE, SESSION,
                principal_id=PRINCIPAL,
                authorization=_auth_obj(
                    aid2, kinds={EnvelopeKind.USER_MESSAGE}),
            )
        assert ei.value.code is ErrorCode.CONSENT_REQUIRED
        # Scope not covered.
        aid3 = issue_capture_authorization(
            conn,
            principal_id=PRINCIPAL,
            issuer_id=PRINCIPAL,
            allowed_kinds={EnvelopeKind.SYSTEM_EVENT},
            retention_policy="keep",
            policy_revision="pol1",
            scope_ids={"other-scope"},
        )
        with pytest.raises(VerbatimError) as ei:
            working.promote(
                conn, store, SCOPE, SESSION,
                principal_id=PRINCIPAL,
                authorization=_auth_obj(aid3, scopes={"other-scope"}),
            )
        assert ei.value.code is ErrorCode.CONSENT_REQUIRED
        # Asserted policy mismatch.
        aid4 = issue_capture_authorization(
            conn,
            principal_id=PRINCIPAL,
            issuer_id=PRINCIPAL,
            allowed_kinds={EnvelopeKind.SYSTEM_EVENT},
            retention_policy="keep",
            policy_revision="pol1",
            scope_ids={SCOPE},
        )
        with pytest.raises(VerbatimError) as ei:
            working.promote(
                conn, store, SCOPE, SESSION,
                principal_id=PRINCIPAL,
                authorization=_auth_obj(aid4),
                retention_policy="forget-after-review",
            )
        assert ei.value.code is ErrorCode.CONSENT_REQUIRED
        # A minted object naming a real id but wider scopes is not that
        # consent — the persisted row is authoritative.
        with pytest.raises(VerbatimError) as ei:
            working.promote(
                conn, store, SCOPE, SESSION,
                principal_id=PRINCIPAL,
                authorization=_auth_obj(aid4, scopes=set()),  # widened
            )
        assert ei.value.code is ErrorCode.CONSENT_REQUIRED
        assert _count(conn, "SELECT COUNT(*) FROM sources") == 0


def test_expired_or_foreign_set_denies(store):
    _provision(store)
    aid = None
    with store.tx() as conn:
        aid = issue_capture_authorization(
            conn,
            principal_id=PRINCIPAL,
            issuer_id=PRINCIPAL,
            allowed_kinds={EnvelopeKind.SYSTEM_EVENT},
            retention_policy="keep",
            policy_revision="pol1",
            scope_ids={SCOPE},
        )
    with store.tx() as conn:
        # Set expires at t=100; promoting at t=200 refuses.
        sid = working.create_set(conn, SCOPE, SESSION, 100, now=1)
        with pytest.raises(VerbatimError) as ei:
            working.promote(
                conn, store, SCOPE, SESSION, set_id=sid,
                principal_id=PRINCIPAL,
                authorization=_auth_obj(aid), now=200,
            )
        assert ei.value.code is ErrorCode.VALIDATION
        # A set from another session is NOT_FOUND_OR_UNAUTHORIZED — a
        # caller cannot promote a peer's working memory.
        other = working.create_set(conn, SCOPE, "sess-2", 60_000_000)
        with pytest.raises(VerbatimError) as ei:
            working.promote(
                conn, store, SCOPE, SESSION, set_id=other,
                principal_id=PRINCIPAL,
                authorization=_auth_obj(aid),
            )
        assert ei.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


# ---------------------------------------------------------------------------
# D19 — success path: explicit consent, durable through the real path
# ---------------------------------------------------------------------------


def test_promotion_with_consent_writes_through_ingest(store):
    _provision(store)
    aid = None
    with store.tx() as conn:
        aid = issue_capture_authorization(
            conn,
            principal_id=PRINCIPAL,
            issuer_id=PRINCIPAL,
            allowed_kinds={EnvelopeKind.SYSTEM_EVENT},
            retention_policy="keep",
            policy_revision="pol1",
            scope_ids={SCOPE},
        )
    sid = _set_with_text(store, ("deploy plan approved", "note two"))
    with store.tx() as conn:
        out = working.promote(
            conn, store, SCOPE, SESSION,
            principal_id=PRINCIPAL, authorization=_auth_obj(aid),
        )
        assert out["retention_policy"] == "keep"
        assert out["authorization_id"] == aid
        assert out["envelope_kind"] == "system_event"
        assert len(out["promoted"]) == 2
        # Each item became a real durable envelope — sources + revisions
        # + spans + envelope row, capture_proof bound to the consent.
        envs = conn.execute(
            "SELECT envelope_kind, capture_proof, metadata_json,"
            " session_id FROM source_envelopes ORDER BY receipt_us"
        ).fetchall()
        assert len(envs) == 2
        import json as _json
        metas = []
        for kind, proof, meta, sess in envs:
            assert kind == "system_event"
            assert proof == aid
            assert sess == SESSION
            m = _json.loads(meta)
            assert m["promoted_from"] == "working_set"
            assert m["working_set_id"] == sid
            assert m["retention_policy"] == "keep"
            metas.append(m)
        item_ids = {m["working_item_id"] for m in metas}
        items = conn.execute(
            "SELECT item_id FROM working_set_items WHERE set_id=?",
            (sid,),
        ).fetchall()
        assert item_ids == {r[0] for r in items}


def test_promotion_idempotent_and_refs_skipped(store):
    _provision(store)
    with store.tx() as conn:
        aid = issue_capture_authorization(
            conn,
            principal_id=PRINCIPAL,
            issuer_id=PRINCIPAL,
            allowed_kinds={EnvelopeKind.SYSTEM_EVENT},
            retention_policy="keep",
            policy_revision="pol1",
            scope_ids={SCOPE},
        )
    with store.tx() as conn:
        sid = working.create_set(conn, SCOPE, SESSION, 60_000_000)
        working.add_item(conn, sid, "recent_evidence",
                         object_ref="claim:cl9")
        working.add_item(conn, sid, "decision", text="ship it")
        out1 = working.promote(
            conn, store, SCOPE, SESSION,
            principal_id=PRINCIPAL, authorization=_auth_obj(aid),
        )
        # object_ref items already point at durable objects — reported,
        # not re-captured.
        assert len(out1["skipped_object_refs"]) == 1
        assert len(out1["promoted"]) == 1
        n_sources = _count(conn, "SELECT COUNT(*) FROM sources")
        assert n_sources == 1
        # Replay: same dedup key → same receipt, no double-capture.
        out2 = working.promote(
            conn, store, SCOPE, SESSION,
            principal_id=PRINCIPAL, authorization=_auth_obj(aid),
        )
        assert out2["promoted"][0]["dedup_key"] == \
            out1["promoted"][0]["dedup_key"]
        assert _count(conn, "SELECT COUNT(*) FROM sources") == 1
