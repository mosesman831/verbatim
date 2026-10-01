"""V3 sensitive-value vault tests (SPEC_V3 §35, §36, §37, §09.07, §10.04,
§11, §46).

Covers: key provisioning (env/file/external), AES-256-GCM seal/open
roundtrip + tamper/integrity, per-entry data-key uniqueness, consent-
gated hydration with opaque one-use handles, one-use action tickets,
honest erasure reporting, placeholder redaction bookkeeping, wrap-key
rotation, and the privacy-lane job handlers through Ingester.run_pending.

All tests run against a real ``Store.create`` (schema v3) — the v1 shim
has no v3 tables.
"""

from __future__ import annotations

import base64
import json
import os

import pytest

from verbatim.config import VaultConfig, V3Config, VerbatimConfig
from verbatim.core.time import now_us
from verbatim.core.types import ErrorCode, JobKind, VerbatimError, new_id
from verbatim.ingest import Ingester
from verbatim.privacy import (
    KeyNotProvisioned,
    Vault,
    erase_entry,
    erasure_report,
    hydrate,
    issue_ticket,
    load_wrap_key,
    lookup_placeholder,
    redact_view,
    refs_for_view,
    resolve_handle,
    scope_slug,
    seal,
    spans_for_view,
    vault_open,
    verify_ticket,
)
from verbatim.storage import repos_v3
from verbatim.storage.store import Store

KEY_A = b"\xaa" * 32
KEY_B = b"\xbb" * 32
KEY_C = b"\xcc" * 32


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "v3.db"))
    yield s
    s.close()


@pytest.fixture
def scope_id(store):
    sid = "scope:vault"
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES (?, 'prof', 'owner')",
            (sid,),
        )
    return sid


@pytest.fixture
def provider():
    """External key provider over a mutable version map; tests add/remove
    versions to exercise rotation and retirement."""
    keys: dict[str, dict[int, bytes]] = {}

    def _p(sid: str, version):
        vers = keys.get(sid)
        if not vers:
            return None
        if version is None:
            v = max(vers)
            return vers[v], v
        k = vers.get(int(version))
        return (k, int(version)) if k is not None else None

    _p.keys = keys
    return _p


@pytest.fixture
def cfg():
    return VerbatimConfig(
        v3=V3Config(
            vault=VaultConfig(enabled=True, key_source="external")
        )
    )


@pytest.fixture
def vault(store, cfg, provider, scope_id):
    provider.keys[scope_id] = {1: KEY_A}
    return Vault(store, cfg, key_provider=provider)


@pytest.fixture
def seeded(provider, scope_id):
    """Provision wrap key v1 for the scope without building a Vault."""
    provider.keys[scope_id] = {1: KEY_A}
    return provider


def _cfg(**kw):
    base = dict(enabled=True, key_source="external")
    base.update(kw)
    return VerbatimConfig(v3=V3Config(vault=VaultConfig(**base)))


def grant_consent(
    store,
    scope_id,
    *,
    processor="local_model",
    purpose="care",
    data_classes=("s2", "s3"),
    expires_us=None,
    revoked_us=None,
):
    cid = f"consent-{new_id()[:12]}"
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO consents (consent_id, scope_id, processor, purpose,"
            " granted_us, revoked_us, policy_digest, data_classes_json,"
            " expires_us) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                cid,
                scope_id,
                processor,
                purpose,
                now_us(),
                revoked_us,
                "digest-1",
                json.dumps(list(data_classes)),
                expires_us,
            ),
        )
    return cid


def _rows(conn, sql, params=()):
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _row(conn, sql, params=()):
    r = _rows(conn, sql, params)
    return r[0] if r else None


# ---------------------------------------------------------------------------
# keys (§35.11, §37.02)
# ---------------------------------------------------------------------------


def test_scope_slug():
    assert scope_slug("scope:vault-1.x") == "SCOPE_VAULT_1_X"


def test_env_key_base64(monkeypatch, cfg, scope_id):
    cfg_env = _cfg(key_source="env")
    monkeypatch.setenv(
        f"VERBATIM_VAULT_KEY_{scope_slug(scope_id)}",
        base64.b64encode(KEY_A).decode(),
    )
    key, ver = load_wrap_key(scope_id, cfg_env)
    assert key == KEY_A and ver == 1


def test_env_key_hex_and_versioned(monkeypatch, scope_id):
    cfg_env = _cfg(key_source="env")
    slug = scope_slug(scope_id)
    monkeypatch.setenv(
        f"VERBATIM_VAULT_KEY_{slug}", "0x" + KEY_B.hex()
    )
    key, ver = load_wrap_key(scope_id, cfg_env)
    assert key == KEY_B and ver == 1
    monkeypatch.setenv(
        f"VERBATIM_VAULT_KEY_{slug}_V7", base64.b64encode(KEY_C).decode()
    )
    key, ver = load_wrap_key(scope_id, cfg_env, version=7)
    assert key == KEY_C and ver == 7
    # unprovisioned version fails closed
    with pytest.raises(KeyNotProvisioned) as exc:
        load_wrap_key(scope_id, cfg_env, version=99)
    assert exc.value.code == ErrorCode.CAPABILITY_UNAVAILABLE


def test_env_key_missing_is_capability_unavailable(scope_id):
    cfg_env = _cfg(key_source="env")
    with pytest.raises(KeyNotProvisioned) as exc:
        load_wrap_key(scope_id, cfg_env)
    assert exc.value.code == ErrorCode.CAPABILITY_UNAVAILABLE


def test_env_key_malformed_never_leaks(monkeypatch, scope_id):
    cfg_env = _cfg(key_source="env")
    secret_b64 = base64.b64encode(b"short").decode()
    monkeypatch.setenv(
        f"VERBATIM_VAULT_KEY_{scope_slug(scope_id)}", secret_b64
    )
    with pytest.raises(KeyNotProvisioned) as exc:
        load_wrap_key(scope_id, cfg_env)
    # the malformed material never appears in the error text (§37.02)
    assert secret_b64 not in str(exc.value)


def test_file_key_source(tmp_path, scope_id):
    kf = tmp_path / "vault-keys.json"
    kf.write_text(
        json.dumps({scope_id: {"versions": {"1": base64.b64encode(KEY_A).decode(),
                                           "2": base64.b64encode(KEY_B).decode()}}})
    )
    os.chmod(kf, 0o600)
    cfg_file = _cfg(key_source="file", key_file=str(kf))
    key, ver = load_wrap_key(scope_id, cfg_file)
    assert (key, ver) == (KEY_B, 2)  # newest provisioned = current
    key, ver = load_wrap_key(scope_id, cfg_file, version=1)
    assert (key, ver) == (KEY_A, 1)


def test_file_key_source_rejects_open_perms(tmp_path, scope_id):
    kf = tmp_path / "keys.json"
    kf.write_text(json.dumps({scope_id: base64.b64encode(KEY_A).decode()}))
    os.chmod(kf, 0o644)
    cfg_file = _cfg(key_source="file", key_file=str(kf))
    with pytest.raises(KeyNotProvisioned) as exc:
        load_wrap_key(scope_id, cfg_file)
    assert "permission" in str(exc.value).lower()


def test_external_provider_missing_callable(scope_id):
    cfg_ext = _cfg(key_source="external")
    with pytest.raises(KeyNotProvisioned):
        load_wrap_key(scope_id, cfg_ext)


def test_external_provider_shape(scope_id):
    cfg_ext = _cfg(key_source="external")
    key, ver = load_wrap_key(
        scope_id, cfg_ext, provider=lambda s, v: (KEY_A, 3)
    )
    assert (key, ver) == (KEY_A, 3)
    with pytest.raises(KeyNotProvisioned):
        load_wrap_key(scope_id, cfg_ext, provider=lambda s, v: None)


# ---------------------------------------------------------------------------
# seal / open (§35.02)
# ---------------------------------------------------------------------------


def test_seal_open_roundtrip_exact_bytes(vault, scope_id):
    payload = bytes(range(256)) * 8  # binary, not just text
    eid = vault.seal(
        scope_id, payload, sensitivity="s2", placeholder="[VAULT:S2:aaaa0001]"
    )
    assert vault.open(eid) == payload


def test_entry_metadata_is_non_secret(vault, store, scope_id):
    eid = vault.seal(
        scope_id, b"ssn-123-45-6789", sensitivity="s2",
        placeholder="[VAULT:S2:bbbb0002]",
    )
    meta = vault.entry(eid)
    assert meta["algorithm"] == "aes-256-gcm:v1"
    assert meta["sensitivity"] == "s2"
    assert meta["wrap_key_version"] == 1
    assert b"ssn-123-45-6789" not in bytes(meta["ciphertext"])
    assert bytes(meta["wrapped_key"]) != b"" and len(meta["nonce"]) == 12


def test_identical_plaintexts_get_distinct_data_keys(vault, store, scope_id):
    e1 = vault.seal(scope_id, b"same-secret", sensitivity="s2",
                    placeholder="[VAULT:S2:cccc0001]")
    e2 = vault.seal(scope_id, b"same-secret", sensitivity="s2",
                    placeholder="[VAULT:S2:cccc0002]")
    r1, r2 = vault.entry(e1), vault.entry(e2)
    assert e1 != e2
    assert bytes(r1["nonce"]) != bytes(r2["nonce"])
    assert bytes(r1["ciphertext"]) != bytes(r2["ciphertext"])
    assert bytes(r1["wrapped_key"]) != bytes(r2["wrapped_key"])
    assert vault.open(e1) == vault.open(e2) == b"same-secret"


def test_wrong_wrap_key_is_integrity(store, cfg, scope_id, provider):
    provider.keys[scope_id] = {1: KEY_A}
    v1 = Vault(store, cfg, key_provider=provider)
    eid = v1.seal(scope_id, b"payload", sensitivity="s2",
                  placeholder="[VAULT:S2:dddd0001]")

    def wrong_provider(sid, version):
        return KEY_C, 1

    v2 = Vault(store, cfg, key_provider=wrong_provider)
    with pytest.raises(VerbatimError) as exc:
        v2.open(eid)
    assert exc.value.code == ErrorCode.INTEGRITY


def test_tampered_ciphertext_is_integrity(vault, store, scope_id):
    eid = vault.seal(scope_id, b"payload", sensitivity="s2",
                     placeholder="[VAULT:S2:eeee0001]")
    with store.tx() as conn:
        row = repos_v3.get(conn, "vault_entries", {"entry_id": eid})
        ct = bytearray(row["ciphertext"])
        ct[0] ^= 0xFF
        repos_v3.update(conn, "vault_entries",
                        {"ciphertext": bytes(ct)}, {"entry_id": eid})
    with pytest.raises(VerbatimError) as exc:
        vault.open(eid)
    assert exc.value.code == ErrorCode.INTEGRITY


def test_tampered_metadata_is_integrity(vault, store, scope_id):
    eid = vault.seal(scope_id, b"payload", sensitivity="s2",
                     placeholder="[VAULT:S2:ffff0001]")
    with store.tx() as conn:
        repos_v3.update(
            conn, "vault_entries",
            {"placeholder": "[VAULT:S2:evil0000]"}, {"entry_id": eid},
        )
    with pytest.raises(VerbatimError) as exc:
        vault.open(eid)
    assert exc.value.code == ErrorCode.INTEGRITY


def test_tampered_nonce_is_integrity(vault, store, scope_id):
    eid = vault.seal(scope_id, b"payload", sensitivity="s2",
                     placeholder="[VAULT:S2:abab0001]")
    with store.tx() as conn:
        repos_v3.update(conn, "vault_entries",
                        {"nonce": b"\x00" * 12}, {"entry_id": eid})
    with pytest.raises(VerbatimError) as exc:
        vault.open(eid)
    assert exc.value.code == ErrorCode.INTEGRITY


def test_vault_disabled_fails_closed(store, scope_id):
    cfg_off = _cfg(enabled=False)
    v = Vault(store, cfg_off, key_provider=lambda s, ver: (KEY_A, 1))
    with pytest.raises(VerbatimError) as exc:
        v.seal(scope_id, b"x", sensitivity="s2", placeholder="[VAULT:S2:x]")
    assert exc.value.code == ErrorCode.CAPABILITY_UNAVAILABLE
    with pytest.raises(VerbatimError) as exc:
        v.open("anything")
    assert exc.value.code == ErrorCode.CAPABILITY_UNAVAILABLE


def test_open_missing_entry_is_not_found(vault, store):
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            vault_open(conn, "missing-entry", cfg=vault.cfg,
                       key_provider=lambda s, v: (KEY_A, 1))
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_seal_without_provisioned_key_fails_closed(store, scope_id):
    cfg_ext = _cfg()
    v = Vault(store, cfg_ext, key_provider=lambda s, v: None)
    with pytest.raises(VerbatimError) as exc:
        v.seal(scope_id, b"x", sensitivity="s2", placeholder="[VAULT:S2:y]")
    assert exc.value.code == ErrorCode.CAPABILITY_UNAVAILABLE


# ---------------------------------------------------------------------------
# hydration (§35.04, §11.07)
# ---------------------------------------------------------------------------


def test_hydrate_without_consent(vault, store, scope_id, cfg, provider):
    eid = vault.seal(scope_id, b"ssn", sensitivity="s2",
                     placeholder="[VAULT:S2:11110001]")
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            hydrate(
                conn, "alice", eid, purpose="care", consent_id="nope",
                downstream_processor="local_model", epoch=None,
                cfg=cfg, key_provider=provider,
            )
    assert exc.value.code == ErrorCode.CONSENT_REQUIRED


def test_hydrate_expired_consent(vault, store, scope_id, cfg, provider):
    eid = vault.seal(scope_id, b"ssn", sensitivity="s2",
                     placeholder="[VAULT:S2:11110002]")
    cid = grant_consent(store, scope_id, expires_us=1)
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            hydrate(
                conn, "alice", eid, purpose="care", consent_id=cid,
                downstream_processor="local_model", epoch=None,
                cfg=cfg, key_provider=provider,
            )
    assert exc.value.code == ErrorCode.CONSENT_REQUIRED


def test_hydrate_revoked_consent(vault, store, scope_id, cfg, provider):
    eid = vault.seal(scope_id, b"ssn", sensitivity="s2",
                     placeholder="[VAULT:S2:11110003]")
    cid = grant_consent(store, scope_id, revoked_us=now_us())
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            hydrate(
                conn, "alice", eid, purpose="care", consent_id=cid,
                downstream_processor="local_model", epoch=None,
                cfg=cfg, key_provider=provider,
            )
    assert exc.value.code == ErrorCode.CONSENT_REQUIRED


def test_hydrate_wrong_purpose_denied(vault, store, scope_id, cfg, provider):
    eid = vault.seal(scope_id, b"ssn", sensitivity="s2",
                     placeholder="[VAULT:S2:11110004]")
    cid = grant_consent(store, scope_id, purpose="billing")
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            hydrate(
                conn, "alice", eid, purpose="care", consent_id=cid,
                downstream_processor="local_model", epoch=None,
                cfg=cfg, key_provider=provider,
            )
    assert exc.value.code == ErrorCode.CONSENT_REQUIRED


def test_hydrate_uncovered_data_class_denied(
    vault, store, scope_id, cfg, provider
):
    eid = vault.seal(scope_id, b"health", sensitivity="s3",
                     placeholder="[VAULT:S3:11110005]")
    cid = grant_consent(store, scope_id, data_classes=("s2",))
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            hydrate(
                conn, "alice", eid, purpose="care", consent_id=cid,
                downstream_processor="local_model", epoch=None,
                cfg=cfg, key_provider=provider,
            )
    assert exc.value.code == ErrorCode.CONSENT_REQUIRED


def test_hydrate_wrong_processor_denied(
    vault, store, scope_id, cfg, provider
):
    eid = vault.seal(scope_id, b"ssn", sensitivity="s2",
                     placeholder="[VAULT:S2:11110006]")
    cid = grant_consent(store, scope_id, processor="task_tool_broker")
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            hydrate(
                conn, "alice", eid, purpose="care", consent_id=cid,
                downstream_processor="local_model", epoch=None,
                cfg=cfg, key_provider=provider,
            )
    assert exc.value.code == ErrorCode.CONSENT_REQUIRED


def test_hydrate_undeclared_processor_denied(
    vault, store, scope_id, cfg, provider
):
    eid = vault.seal(scope_id, b"ssn", sensitivity="s2",
                     placeholder="[VAULT:S2:11110007]")
    cid = grant_consent(store, scope_id)
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            hydrate(
                conn, "alice", eid, purpose="care", consent_id=cid,
                downstream_processor="", epoch=None,
                cfg=cfg, key_provider=provider,
            )
    assert exc.value.code == ErrorCode.CONSENT_REQUIRED


def test_hydrate_stale_epoch(vault, store, scope_id, cfg, provider):
    eid = vault.seal(scope_id, b"ssn", sensitivity="s2",
                     placeholder="[VAULT:S2:11110008]")
    cid = grant_consent(store, scope_id)
    with store.tx() as conn:
        conn.execute(
            "UPDATE scopes SET authz_revision = 5 WHERE scope_id = ?",
            (scope_id,),
        )
        with pytest.raises(VerbatimError) as exc:
            hydrate(
                conn, "alice", eid, purpose="care", consent_id=cid,
                downstream_processor="local_model", epoch=3,
                cfg=cfg, key_provider=provider,
            )
    assert exc.value.code == ErrorCode.STALE_EPOCH


def test_hydrate_plaintext_for_local_trusted(
    vault, store, scope_id, cfg, provider
):
    eid = vault.seal(scope_id, b"ssn-value", sensitivity="s2",
                     placeholder="[VAULT:S2:11110009]")
    cid = grant_consent(store, scope_id, processor="local_model")
    with store.tx() as conn:
        result = hydrate(
            conn, "alice", eid, purpose="care", consent_id=cid,
            downstream_processor="local_model", epoch=0,
            cfg=cfg, key_provider=provider,
        )
        assert result.kind == "plaintext"
        assert result.value == b"ssn-value"
        prop = _row(
            conn,
            "SELECT * FROM propagations WHERE propagation_id = ?",
            (result.propagation_id,),
        )
    assert prop["object_kind"] == "vault_entry"
    assert prop["object_id"] == eid
    assert prop["recipient_id"] == "alice"
    assert prop["purpose"] == "care"


def test_hydrate_default_path_returns_opaque_handle(
    vault, store, scope_id, cfg, provider
):
    eid = vault.seal(scope_id, b"ssn-value", sensitivity="s2",
                     placeholder="[VAULT:S2:11110010]")
    cid = grant_consent(store, scope_id, processor="task_tool_broker")
    with store.tx() as conn:
        result = hydrate(
            conn, "broker-1", eid, purpose="care", consent_id=cid,
            downstream_processor="task_tool_broker", epoch=None,
            cfg=cfg, key_provider=provider,
        )
        assert result.kind == "handle"
        assert result.value is None
        assert result.handle_id
        handle = _row(
            conn,
            "SELECT * FROM value_handles WHERE handle_id = ?",
            (result.handle_id,),
        )
    assert handle["vault_entry_id"] == eid
    assert handle["recipient_id"] == "broker-1"
    assert handle["consent_id"] == cid
    assert handle["consumed_us"] is None


def test_handle_resolves_once_then_replay_denied(
    vault, store, scope_id, cfg, provider
):
    eid = vault.seal(scope_id, b"ssn-value", sensitivity="s2",
                     placeholder="[VAULT:S2:11110011]")
    cid = grant_consent(store, scope_id, processor="task_tool_broker")
    with store.tx() as conn:
        result = hydrate(
            conn, "broker-1", eid, purpose="care", consent_id=cid,
            downstream_processor="task_tool_broker", epoch=None,
            cfg=cfg, key_provider=provider,
        )
        value = resolve_handle(
            conn, result.handle_id, recipient_id="broker-1",
            action_digest=result.action_digest,
            cfg=cfg, key_provider=provider,
        )
        assert value == b"ssn-value"
        row = _row(conn, "SELECT consumed_us FROM value_handles"
                         " WHERE handle_id = ?", (result.handle_id,))
        assert row["consumed_us"] is not None
        with pytest.raises(VerbatimError) as exc:
            resolve_handle(
                conn, result.handle_id, recipient_id="broker-1",
                action_digest=result.action_digest,
                cfg=cfg, key_provider=provider,
            )
    assert exc.value.code == ErrorCode.VALIDATION


def test_handle_wrong_recipient_denied(
    vault, store, scope_id, cfg, provider
):
    eid = vault.seal(scope_id, b"ssn", sensitivity="s2",
                     placeholder="[VAULT:S2:11110012]")
    cid = grant_consent(store, scope_id, processor="task_tool_broker")
    with store.tx() as conn:
        result = hydrate(
            conn, "broker-1", eid, purpose="care", consent_id=cid,
            downstream_processor="task_tool_broker", epoch=None,
            cfg=cfg, key_provider=provider,
        )
        with pytest.raises(VerbatimError) as exc:
            resolve_handle(
                conn, result.handle_id, recipient_id="mallory",
                action_digest=result.action_digest,
                cfg=cfg, key_provider=provider,
            )
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_handle_wrong_digest_denied(
    vault, store, scope_id, cfg, provider
):
    eid = vault.seal(scope_id, b"ssn", sensitivity="s2",
                     placeholder="[VAULT:S2:11110013]")
    cid = grant_consent(store, scope_id, processor="task_tool_broker")
    with store.tx() as conn:
        result = hydrate(
            conn, "broker-1", eid, purpose="care", consent_id=cid,
            downstream_processor="task_tool_broker", epoch=None,
            cfg=cfg, key_provider=provider,
        )
        with pytest.raises(VerbatimError) as exc:
            resolve_handle(
                conn, result.handle_id, recipient_id="broker-1",
                action_digest=b"\x00" * 32,
                cfg=cfg, key_provider=provider,
            )
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_expired_handle_denied(store, scope_id, cfg, provider):
    provider.keys[scope_id] = {1: KEY_A}
    v = Vault(store, cfg, key_provider=provider)
    eid = v.seal(scope_id, b"ssn", sensitivity="s2",
                 placeholder="[VAULT:S2:11110014]")
    cid = grant_consent(store, scope_id, processor="task_tool_broker")
    t0 = now_us()
    with store.tx() as conn:
        result = hydrate(
            conn, "broker-1", eid, purpose="care", consent_id=cid,
            downstream_processor="task_tool_broker", epoch=None,
            cfg=cfg, key_provider=provider, now=t0,
        )
        with pytest.raises(VerbatimError) as exc:
            resolve_handle(
                conn, result.handle_id, recipient_id="broker-1",
                action_digest=result.action_digest, cfg=cfg,
                key_provider=provider,
                now=t0 + (cfg.v3.vault.handle_ttl_s + 1) * 1_000_000,
            )
    assert exc.value.code == ErrorCode.VALIDATION


def test_allow_plaintext_hydration(store, scope_id, provider):
    cfg_p = _cfg(allow_plaintext_hydration=True)
    provider.keys[scope_id] = {1: KEY_A}
    v = Vault(store, cfg_p, key_provider=provider)
    eid = v.seal(scope_id, b"ssn", sensitivity="s2",
                 placeholder="[VAULT:S2:11110015]")
    cid = grant_consent(store, scope_id, processor="task_tool_broker")
    with store.tx() as conn:
        result = hydrate(
            conn, "broker-1", eid, purpose="care", consent_id=cid,
            downstream_processor="task_tool_broker", epoch=None,
            cfg=cfg_p, key_provider=provider,
        )
    assert result.kind == "plaintext"
    assert result.value == b"ssn"


# ---------------------------------------------------------------------------
# action tickets (§09.07)
# ---------------------------------------------------------------------------

_DIGEST = b"\x11" * 32


def _ticket(store, scope_id, **kw):
    args = dict(
        recipient_id="agent-1",
        action_digest=_DIGEST,
        purpose="deploy",
        epoch=None,
        objects=[("obj-1", 1), ("obj-2", 3)],
        ttl_us=None,
        gateway_id="gw-1",
    )
    args.update(kw)
    with store.tx() as conn:
        return issue_ticket(conn, scope_id, **args)


def test_ticket_verify_consumes_once(store, scope_id):
    tid = _ticket(store, scope_id)
    with store.tx() as conn:
        out = verify_ticket(conn, tid, "gw-1", _DIGEST)
        assert out["ticket_id"] == tid
        assert out["consumed_us"] is not None
        assert {o["object_id"] for o in out["objects"]} == {"obj-1", "obj-2"}
        with pytest.raises(VerbatimError) as exc:
            verify_ticket(conn, tid, "gw-1", _DIGEST)
    assert exc.value.code == ErrorCode.VALIDATION


def test_ticket_wrong_gateway_denied(store, scope_id):
    tid = _ticket(store, scope_id, gateway_id="gw-1")
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            verify_ticket(conn, tid, "gw-evil", _DIGEST)
        # the ticket is not burned — the right gateway can still verify
        out = verify_ticket(conn, tid, "gw-1", _DIGEST)
        assert out["ticket_id"] == tid
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_ticket_expired_denied(store, scope_id):
    t0 = now_us()
    with store.tx() as conn:
        tid = issue_ticket(
            conn, scope_id, "agent-1", _DIGEST, "deploy", None,
            [("o", 1)], 10_000_000, "gw-1", now=t0,
        )
        with pytest.raises(VerbatimError) as exc:
            verify_ticket(conn, tid, "gw-1", _DIGEST,
                          now=t0 + 11_000_000)
    assert exc.value.code == ErrorCode.VALIDATION


def test_ticket_wrong_action_digest_denied(store, scope_id):
    tid = _ticket(store, scope_id)
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            verify_ticket(conn, tid, "gw-1", b"\x99" * 32)
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_ticket_stale_epoch_denied(store, scope_id):
    tid = _ticket(store, scope_id)
    with store.tx() as conn:
        conn.execute(
            "UPDATE scopes SET authz_revision = 9 WHERE scope_id = ?",
            (scope_id,),
        )
        with pytest.raises(VerbatimError) as exc:
            verify_ticket(conn, tid, "gw-1", _DIGEST)
    assert exc.value.code == ErrorCode.STALE_EPOCH


def test_ticket_issue_at_stale_epoch_denied(store, scope_id):
    with store.tx() as conn:
        conn.execute(
            "UPDATE scopes SET authz_revision = 4 WHERE scope_id = ?",
            (scope_id,),
        )
        with pytest.raises(VerbatimError) as exc:
            issue_ticket(conn, scope_id, "a", _DIGEST, "p", 1,
                         [], None, "gw-1")
    assert exc.value.code == ErrorCode.STALE_EPOCH


def test_ticket_lifetime_cap(store, scope_id):
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            issue_ticket(conn, scope_id, "a", _DIGEST, "p", None,
                         [], 61_000_000, "gw-1")
    assert exc.value.code == ErrorCode.VALIDATION


def test_ticket_unknown_is_denied(store, scope_id):
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            verify_ticket(conn, "tk-nonexistent", "gw-1", _DIGEST)
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


# ---------------------------------------------------------------------------
# erasure (§35.05, §36)
# ---------------------------------------------------------------------------


def test_erase_entry_sets_tombstone_and_purges_refs(
    vault, store, scope_id, cfg, provider
):
    eid = vault.seal(scope_id, b"ssn", sensitivity="s2",
                     placeholder="[VAULT:S2:22220001]")
    with store.tx() as conn:
        repos_v3.insert(conn, "vault_refs", {
            "placeholder": "[VAULT:S2:22220001]", "entry_id": eid,
            "scope_id": scope_id, "view_id": "v1",
            "start_byte": 0, "end_byte": 18,
        })
        assert erase_entry(conn, eid, hmac_fn=store.hmac) is True
        # idempotent
        assert erase_entry(conn, eid) is False
        assert repos_v3.query(conn, "vault_refs", {"entry_id": eid}) == []
        assert _row(conn, "SELECT erased_event FROM vault_entries"
                          " WHERE entry_id = ?", (eid,))["erased_event"] is not None
        assert _row(conn, "SELECT * FROM erasure_ledger"
                          " WHERE object_kind = 'vault_entry'") is not None
    with pytest.raises(VerbatimError) as exc:
        vault.open(eid)
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_hydrate_after_erase_denied(vault, store, scope_id, cfg, provider):
    eid = vault.seal(scope_id, b"ssn", sensitivity="s2",
                     placeholder="[VAULT:S2:22220002]")
    cid = grant_consent(store, scope_id)
    with store.tx() as conn:
        erase_entry(conn, eid)
        with pytest.raises(VerbatimError) as exc:
            hydrate(
                conn, "alice", eid, purpose="care", consent_id=cid,
                downstream_processor="local_model", epoch=None,
                cfg=cfg, key_provider=provider,
            )
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_erasure_report_honest_by_default(vault, store, scope_id):
    eid = vault.seal(scope_id, b"ssn", sensitivity="s2",
                     placeholder="[VAULT:S2:22220003]")
    rep = vault.report(eid)
    assert rep["cryptographic_erasure"] == "unproven"
    assert rep["logical_erasure"] is False
    vault.erase(eid)
    rep = vault.report(eid)
    assert rep["cryptographic_erasure"] == "unproven"
    assert rep["logical_erasure"] is True
    rep = vault.report(eid, verified_key_destruction=True)
    assert rep["cryptographic_erasure"] == "proven"


def test_erasure_report_absent_entry(store, scope_id):
    with store.read() as conn:
        rep = erasure_report(conn, "missing-entry")
    assert rep["cryptographic_erasure"] == "unproven"
    assert rep["present"] is False


# ---------------------------------------------------------------------------
# redaction (§35.01, §35.07)
# ---------------------------------------------------------------------------


def test_redact_view_replaces_value_with_placeholder(
    vault, store, scope_id, cfg, provider
):
    text = "My SSN is 123-45-6789 and I like tea."
    start = text.index("123-45-6789")
    end = start + len("123-45-6789")

    def detector(t):
        return [(start, end, "s2")]

    with store.tx() as conn:
        result = redact_view(
            conn, scope_id, "view-1", text.encode(), cfg=cfg,
            detector=detector, key_provider=provider,
        )
        assert b"123-45-6789" not in result.accepted
        ph = result.redactions[0].placeholder
        assert ph.startswith("[VAULT:S2:")
        assert ph.encode() in result.accepted
        # redaction_spans: orig vs accepted offsets are DISTINCT columns
        spans = spans_for_view(conn, scope_id, "view-1")
        assert len(spans) == 1
        sp = spans[0]
        assert (sp["orig_start"], sp["orig_end"]) == (start, end)
        assert (sp["accepted_start"], sp["accepted_end"]) == (
            start, start + len(ph)
        )
        assert sp["sensitivity"] == "s2"
        assert sp["entry_id"] == result.redactions[0].entry_id
        # vault_refs make the placeholder searchable (accepted-view bytes)
        refs = refs_for_view(conn, scope_id, "view-1")
        assert len(refs) == 1
        assert refs[0]["placeholder"] == ph
        assert (refs[0]["start_byte"], refs[0]["end_byte"]) == (
            sp["accepted_start"], sp["accepted_end"]
        )
        assert lookup_placeholder(conn, scope_id, ph)["entry_id"] == sp["entry_id"]
    # sealed value roundtrips
    assert vault.open(sp["entry_id"]) == b"123-45-6789"


def test_redact_view_owner_declared_basis(
    vault, store, scope_id, cfg, provider
):
    text = "call me at 555-0100 please"
    start = text.index("555-0100")
    with store.tx() as conn:
        result = redact_view(
            conn, scope_id, "view-2", text.encode(), cfg=cfg,
            declared=[(start, start + 8, "s3")],
            key_provider=provider,
        )
        entry = repos_v3.get(
            conn, "vault_entries",
            {"entry_id": result.redactions[0].entry_id},
        )
    assert entry["detection"] == "owner_declared"
    assert entry["sensitivity"] == "s3"


def test_redact_view_s4_zero_retention(
    vault, store, scope_id, cfg, provider
):
    text = "token: sk-live-abcdef123456"
    start = text.index("sk-live")
    end = len(text)

    def detector(t):
        return [(start, end, "s4")]

    with store.tx() as conn:
        result = redact_view(
            conn, scope_id, "view-3", text.encode(), cfg=cfg,
            detector=detector, key_provider=provider,
        )
        assert b"sk-live" not in result.accepted
        sp = spans_for_view(conn, scope_id, "view-3")[0]
        assert sp["sensitivity"] == "s4"
        assert sp["entry_id"] is None
        # zero retention: no vault entry, no searchable ref, no digest
        assert refs_for_view(conn, scope_id, "view-3") == []
        assert repos_v3.query(conn, "vault_entries",
                              {"scope_id": scope_id}) == []


def test_redact_view_s1_passes_through(
    vault, store, scope_id, cfg, provider
):
    text = "I prefer dark mode"
    with store.tx() as conn:
        result = redact_view(
            conn, scope_id, "view-4", text.encode(), cfg=cfg,
            declared=[(2, 8, "s1")], key_provider=provider,
        )
        assert result.accepted == text.encode()
        assert result.redactions == ()
        assert result.skipped and result.skipped[0][2] == "s1"


def test_redact_view_overlapping_spans_rejected(
    vault, store, scope_id, cfg, provider
):
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            redact_view(
                conn, scope_id, "view-5", b"abcdef", cfg=cfg,
                declared=[(0, 4, "s2"), (2, 5, "s2")],
                key_provider=provider,
            )
    assert exc.value.code == ErrorCode.VALIDATION


def test_redact_view_nonascii_offsets(
    vault, store, scope_id, cfg, provider
):
    text = "héllo wörld secret-välüe end"
    start = text.index("secret-välüe")
    end = start + len("secret-välüe")
    with store.tx() as conn:
        result = redact_view(
            conn, scope_id, "view-6", text.encode("utf-8"), cfg=cfg,
            declared=[(start, end, "s2")], key_provider=provider,
        )
        assert "secret-välüe".encode("utf-8") not in result.accepted
        assert result.accepted.decode("utf-8").startswith("héllo wörld [VAULT:S2:")
        entry_id = result.redactions[0].entry_id
    assert vault.open(entry_id) == "secret-välüe".encode("utf-8")


# ---------------------------------------------------------------------------
# rotation (§35.12, §37.03)
# ---------------------------------------------------------------------------


def test_rotation_rewraps_to_new_version(
    vault, store, scope_id, provider
):
    e1 = vault.seal(scope_id, b"one", sensitivity="s2",
                    placeholder="[VAULT:S2:33330001]")
    e2 = vault.seal(scope_id, b"two", sensitivity="s2",
                    placeholder="[VAULT:S2:33330002]")
    ct1_before = bytes(vault.entry(e1)["ciphertext"])
    provider.keys[scope_id][2] = KEY_B  # provision v2; current becomes 2
    receipt = vault.rotate_scope(scope_id)
    assert receipt["rewrapped"] == 2
    assert receipt["retired_wrap_versions"] == [1]
    for eid in (e1, e2):
        meta = vault.entry(eid)
        assert meta["wrap_key_version"] == 2
    assert bytes(vault.entry(e1)["ciphertext"]) == ct1_before
    # entries open under the NEW wrap key even after v1 is retired
    del provider.keys[scope_id][1]
    assert vault.open(e1) == b"one"
    assert vault.open(e2) == b"two"


def test_rotate_scope_noop_when_current(vault, scope_id):
    vault.seal(scope_id, b"one", sensitivity="s2",
               placeholder="[VAULT:S2:33330003]")
    receipt = vault.rotate_scope(scope_id)
    assert receipt["rewrapped"] == 0
    assert receipt["retired_wrap_versions"] == []


# ---------------------------------------------------------------------------
# handlers (§36.02, §40) — through Ingester.run_pending
# ---------------------------------------------------------------------------


@pytest.fixture
def ingester(store, cfg, provider):
    ing = Ingester(store, cfg)
    ing.vault_key_provider = provider
    return ing


def _drain(ing, kind, refs, scope_id):
    with ing.store.tx() as conn:
        jid = ing.jobs.enqueue(conn, scope_id, kind, refs)
    n = ing.run_pending(limit=8, owner="test-worker")
    assert n >= 1
    with ing.store.read() as conn:
        job = _row(conn, "SELECT * FROM jobs WHERE job_id = ?", (jid,))
    return jid, job


def test_handle_purge_vault_idempotent(
    store, scope_id, cfg, provider, ingester, seeded
):
    vault = Vault(store, cfg, key_provider=provider)
    e1 = vault.seal(scope_id, b"a", sensitivity="s2",
                    placeholder="[VAULT:S2:44440001]")
    e2 = vault.seal(scope_id, b"b", sensitivity="s2",
                    placeholder="[VAULT:S2:44440002]")
    jid, job = _drain(
        ingester, JobKind.PURGE_VAULT, {"entry_ids": [e1, e2]}, scope_id
    )
    assert job["state"] == "succeeded"
    for eid in (e1, e2):
        with pytest.raises(VerbatimError):
            vault.open(eid)
        assert vault.entry(eid)["erased_event"] is not None
    # replay the same selection — idempotent, still succeeds
    jid2, job2 = _drain(
        ingester, JobKind.PURGE_VAULT, {"entry_ids": [e1, e2]}, scope_id
    )
    assert job2["state"] == "succeeded"


def test_handle_purge_vault_by_view(
    store, scope_id, cfg, provider, ingester, seeded
):
    vault = Vault(store, cfg, key_provider=provider)
    text = "pin 1234 ok"
    start = text.index("1234")
    with store.tx() as conn:
        redact_view(
            conn, scope_id, "view-purge", text.encode(), cfg=cfg,
            declared=[(start, start + 4, "s2")], key_provider=provider,
        )
    jid, job = _drain(
        ingester, JobKind.PURGE_VAULT, {"view_id": "view-purge"}, scope_id
    )
    assert job["state"] == "succeeded"
    with store.read() as conn:
        assert refs_for_view(conn, scope_id, "view-purge") == []
        entries = repos_v3.query(conn, "vault_entries",
                                 {"scope_id": scope_id})
        assert all(e["erased_event"] is not None for e in entries)


def test_handle_purge_vault_requires_selection(
    store, scope_id, ingester
):
    jid, job = _drain(
        ingester, JobKind.PURGE_VAULT, {}, scope_id
    )
    assert job["state"] == "failed"
    assert job["error_code"] == "VALIDATION"


def test_handle_vault_rotate(
    store, scope_id, cfg, provider, ingester, seeded
):
    vault = Vault(store, cfg, key_provider=provider)
    e1 = vault.seal(scope_id, b"a", sensitivity="s2",
                    placeholder="[VAULT:S2:55550001]")
    provider.keys[scope_id][2] = KEY_B
    jid, job = _drain(ingester, JobKind.VAULT_ROTATE, {}, scope_id)
    assert job["state"] == "succeeded"
    assert vault.entry(e1)["wrap_key_version"] == 2
    del provider.keys[scope_id][1]
    assert vault.open(e1) == b"a"


def test_handle_purge_derived(store, scope_id, cfg, ingester):
    with store.tx() as conn:
        repos_v3.insert(conn, "derivations", {
            "child_kind": "observation", "child_id": "obs-1",
            "child_revision": 1, "parent_kind": "source",
            "parent_id": "src-1", "parent_revision": 1,
            "producer_kind": "job", "producer_id": "j1",
            "seq": 1, "scope_id": scope_id,
        })
        repos_v3.insert(conn, "derivations", {
            "child_kind": "observation", "child_id": "obs-2",
            "child_revision": 1, "parent_kind": "observation",
            "parent_id": "obs-1", "parent_revision": 1,
            "producer_kind": "job", "producer_id": "j1",
            "seq": 2, "scope_id": scope_id,
        })
        for oid in ("obs-1", "obs-2"):
            repos_v3.insert(conn, "observations", {
                "observation_id": oid, "scope_id": scope_id,
                "revision": 1, "text": f"text-{oid}", "proof_count": 0,
                "freshness": "unknown", "recorded_from": 0,
            })
    jid, job = _drain(
        ingester, JobKind.PURGE_DERIVED,
        {"parents": [{"kind": "source", "id": "src-1", "revision": 1}]},
        scope_id,
    )
    assert job["state"] == "succeeded"
    with store.read() as conn:
        # transitive closure: obs-2 deleted via obs-1 (§36.02)
        assert repos_v3.query(conn, "observations",
                              {"scope_id": scope_id}) == []
        assert repos_v3.query(conn, "derivations",
                              {"scope_id": scope_id}) == []


def test_handle_purge_derived_reports_unhandled(
    store, scope_id, cfg, ingester
):
    with store.tx() as conn:
        repos_v3.insert(conn, "derivations", {
            "child_kind": "mystery_kind", "child_id": "m-1",
            "child_revision": 1, "parent_kind": "source",
            "parent_id": "src-9", "parent_revision": 1,
            "producer_kind": "job", "producer_id": "j1",
            "seq": 1, "scope_id": scope_id,
        })
    jid, job = _drain(
        ingester, JobKind.PURGE_DERIVED,
        {"parents": [{"kind": "source", "id": "src-9"}]},
        scope_id,
    )
    assert job["state"] == "succeeded"
    # edges closed even for unhandled kinds
    with store.read() as conn:
        assert repos_v3.query(conn, "derivations",
                              {"scope_id": scope_id}) == []


# ---------------------------------------------------------------------------
# honest accounting (§35.02, §37.02): no key material anywhere it mustn't be
# ---------------------------------------------------------------------------


def test_error_messages_never_carry_key_material(store, scope_id, provider):
    provider.keys[scope_id] = {1: KEY_A}
    cfg_ext = _cfg()
    v = Vault(store, cfg_ext, key_provider=lambda s, ver: None)
    with pytest.raises(VerbatimError) as exc:
        v.seal(scope_id, b"x", sensitivity="s2", placeholder="[VAULT:S2:z]")
    assert KEY_A.hex() not in str(exc.value)
    assert base64.b64encode(KEY_A).decode() not in str(exc.value)
