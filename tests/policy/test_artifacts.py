"""Policy-artifact graduation tests (SPEC_V6 V6-03.15/16, F27).

Covers the register → mark_validated → bind_for_activation flow:
unvalidated artifacts cannot activate; validated-with-evidence can;
revoked can do neither; the kill-switch refusal names the missing
binding; and a fabricated registered+validated artifact activates the
learned controller through ``recall_v3`` (the binding check enforced at
controller construction inside the recall read snapshot). Real on-disk
``Store.create`` fixtures; learned-policy artifacts in the frozen
``learned_policy_v1`` shape (same minimal fabrication ``test_learned``
uses — ``eval/v3/g8_policy.json`` is the honest-rejection exemplar).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verbatim.config import config_from_mapping
from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v3 import (
    ActionIntent,
    BudgetTier,
    QueryClass,
    RecallRequestV3,
)
from verbatim.policy import artifacts as _polart
from verbatim.retrieval.v3 import controller as _ctrl
from verbatim.retrieval.v3 import learned as _learned
from verbatim.retrieval.v3 import recall_v3
from verbatim.storage.repos import has_table
from verbatim.storage.store import Store

from tests.retrieval.v3.test_retrieval_v3 import (
    _TEST_HMAC_KEY,
    seed_auth,
    seed_claim,
    seed_scope,
)


@pytest.fixture
def store(tmp_path):
    # seed_claim signs payloads with _TEST_HMAC_KEY — pin the store key
    # so the packaging integrity re-check does not report corruption.
    (tmp_path / "v.db.key").write_bytes(_TEST_HMAC_KEY)
    s = Store.create(str(tmp_path / "v.db"))
    yield s
    s.close()


def _state() -> _ctrl._State:
    return _ctrl._State(
        query_class=QueryClass.CURRENT_STATE,
        action_intent=ActionIntent.NONE,
        size_band="small",
        procedures=False,
        stuck=False,
        session_phase="none",
        freshness_required=False,
        tier=BudgetTier.MID,
        has_identifiers=False,
        memory_kinds=(),
        prefetch=False,
    )


def _request(scope_id="sA", **kw) -> RecallRequestV3:
    return RecallRequestV3(
        query="when does the release ship", scope_id=scope_id,
        caller_id="human:alice", purpose="recall", **kw,
    )


def _seeded(store, scope_id="sA", n=3):
    gen = store.projection_generation()
    with store.tx() as conn:
        seed_scope(conn, scope_id)
        seed_auth(conn, scope_id)
        for i in range(n):
            seed_claim(
                conn, f"cl{i}", scope_id, f"src{i}", f"sp{i}",
                f"the release ships on Friday item {i}", gen,
            )


def _artifact(revision="pol_v1", *, deviating=True) -> dict:
    """Minimal valid learned_policy_v1 payload. When ``deviating``, the
    artifact carries measured pulls at the seeded recall's state_key
    preferring ``items_half`` over ``base`` — enough for ``choose`` to
    deviate from the deterministic plan (same shape as g8_policy.json)."""
    skey = _ctrl._state_key(_state())
    art = _learned.PolicyArtifact(
        revision=revision,
        q={skey: {"base": 0.1, "items_half": 0.9}} if deviating else {},
        pulls={skey: {"base": 3, "items_half": 3}} if deviating else {},
    )
    return art.to_dict()


_EVIDENCE = {
    "paired_run": "eval/v3/g8.py run-2026-09-19",
    "gate": "G8",
    "verdict": "pass",
}

_ACTIVE_CFG = lambda aid: {
    "v3": {"retrieval": {
        "controller": "learned_active",
        "controller_policy_artifact": aid,
    }},
}


def _register(store, artifact=None, **kw):
    with store.tx() as conn:
        return _polart.register_artifact(conn, artifact or _artifact(), **kw)


def _validate(store, aid, evidence=_EVIDENCE):
    with store.tx() as conn:
        return _polart.mark_validated(conn, aid, evidence=evidence)


def _revoke(store, aid):
    with store.tx() as conn:
        return _polart.mark_revoked(conn, aid, reason="test")


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def test_register_lands_unvalidated(store):
    aid = _register(store)
    assert aid.startswith("pa_")
    with store.read() as conn:
        row = conn.execute(
            "SELECT kind, validation_state, digest, declared_json"
            " FROM policy_artifacts WHERE artifact_id = ?",
            (aid,),
        ).fetchone()
    assert row is not None
    assert row[0] == "learned_policy_v1"
    assert row[1] == "unvalidated"
    assert len(row[2]) == 32


def test_register_idempotent_same_content(store):
    aid1 = _register(store)
    aid2 = _register(store)
    assert aid1 == aid2
    with store.read() as conn:
        n = conn.execute("SELECT COUNT(*) FROM policy_artifacts").fetchone()
    assert n[0] == 1


def test_register_rejects_validated_state(store):
    """Registration cannot self-declare validated — the state is earned
    through mark_validated's evidence gate only."""
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            _polart.register_artifact(
                conn, _artifact(), validation_state="validated",
            )
        assert exc.value.code == ErrorCode.VALIDATION
        assert "mark_validated" in str(exc.value)


def test_register_rejects_bad_artifacts(store):
    with store.tx() as conn:
        for bad in (
            {"kind": "other"},
            {"kind": "learned_policy_v1"},            # missing revision
            {"kind": "learned_policy_v1", "revision": "x",
             "actions": ["fly_to_moon"]},             # outside bounded set
            "not json",
            42,
        ):
            with pytest.raises(VerbatimError) as exc:
                _polart.register_artifact(conn, bad)
            assert exc.value.code == ErrorCode.VALIDATION


def test_register_accepts_g8_shaped_json(store, tmp_path):
    """A real fitted artifact file (g8_policy.json shape — honestly
    rejected by the G8 gate, but a VALID learned_policy_v1 payload)
    registers cleanly; g8_pass=false only means it must NOT validate."""
    art = _learned.PolicyArtifact(revision="g8w1_like").to_json()
    path = tmp_path / "pol.json"
    path.write_text(art)
    with store.tx() as conn:
        aid = _polart.register_artifact(conn, str(path))
    assert aid.startswith("pa_")


# ---------------------------------------------------------------------------
# mark_validated — the evidence gate
# ---------------------------------------------------------------------------


def test_mark_validated_requires_evidence(store):
    aid = _register(store)
    with store.tx() as conn:
        for bad in (
            None,
            {},
            {"paired_run": "r"},                          # missing gate+verdict
            {"paired_run": "r", "gate": "G8"},            # missing verdict
            {"paired_run": "r", "gate": "G8",
             "verdict": "fail"},                          # failed gate
            {"paired_run": "", "gate": "G8",
             "verdict": "pass"},                          # empty run name
            "paired-run-passed",                          # not a dict
        ):
            with pytest.raises(VerbatimError) as exc:
                _polart.mark_validated(conn, aid, evidence=bad)
            assert exc.value.code == ErrorCode.VALIDATION


def test_mark_validated_transitions_and_attests(store):
    aid = _register(store)
    assert _validate(store, aid) is True
    with store.read() as conn:
        row = conn.execute(
            "SELECT validation_state FROM policy_artifacts"
            " WHERE artifact_id = ?",
            (aid,),
        ).fetchone()
        assert row[0] == "validated"
        att = _polart.attestations(conn, aid)
    assert len(att) == 1
    assert att[0]["state"] == "validated"
    assert att[0]["evidence"]["paired_run"] == _EVIDENCE["paired_run"]
    assert att[0]["evidence"]["gate"] == "G8"


def test_mark_validated_idempotent(store):
    aid = _register(store)
    assert _validate(store, aid) is True
    assert _validate(store, aid) is False  # already validated — no-op


def test_mark_validated_unknown_artifact(store):
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            _polart.mark_validated(
                conn, "pa_doesnotexist", evidence=_EVIDENCE,
            )
        assert exc.value.code == ErrorCode.VALIDATION


def test_revoked_artifact_cannot_validate(store):
    aid = _register(store)
    assert _revoke(store, aid) is True
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            _polart.mark_validated(conn, aid, evidence=_EVIDENCE)
        assert exc.value.code == ErrorCode.VALIDATION
        assert "revoked" in str(exc.value).lower()
    # attestation trail records the revocation
    with store.read() as conn:
        att = _polart.attestations(conn, aid)
    assert [a["state"] for a in att] == ["revoked"]


# ---------------------------------------------------------------------------
# bind_for_activation — the store-side half of the config gate
# ---------------------------------------------------------------------------


def test_bind_unvalidated_refuses_naming_artifact(store):
    aid = _register(store)
    cfg = config_from_mapping(_ACTIVE_CFG(aid))
    with pytest.raises(VerbatimError) as exc:
        _polart.bind_for_activation(cfg, aid, store)
    assert exc.value.code == ErrorCode.CONFIG_INVALID
    assert aid in str(exc.value)
    assert "unvalidated" in str(exc.value)


def test_bind_missing_artifact_refuses_naming_it(store):
    aid = "pa_000000000000000000000000"
    cfg = config_from_mapping(_ACTIVE_CFG(aid))
    with pytest.raises(VerbatimError) as exc:
        _polart.bind_for_activation(cfg, aid, store)
    assert exc.value.code == ErrorCode.CONFIG_INVALID
    assert aid in str(exc.value)
    assert "not" in str(exc.value) and "registered" in str(exc.value)


def test_bind_revoked_refuses(store):
    aid = _register(store)
    _validate(store, aid)
    _revoke(store, aid)
    cfg = config_from_mapping(_ACTIVE_CFG(aid))
    with pytest.raises(VerbatimError) as exc:
        _polart.bind_for_activation(cfg, aid, store)
    assert exc.value.code == ErrorCode.CONFIG_INVALID
    assert "revoked" in str(exc.value)


def test_bind_validated_returns_artifact(store):
    aid = _register(store)
    _validate(store, aid)
    cfg = config_from_mapping(_ACTIVE_CFG(aid))
    art = _polart.bind_for_activation(cfg, aid, store)
    assert isinstance(art, _learned.PolicyArtifact)
    assert art.revision == "pol_v1"


def test_bind_mismatched_binding_refuses(store):
    aid = _register(store)
    _validate(store, aid)
    other = _register(store, _artifact(revision="pol_other"))
    _validate(store, other)
    cfg = config_from_mapping(_ACTIVE_CFG(other))
    with pytest.raises(VerbatimError) as exc:
        _polart.bind_for_activation(cfg, aid, store)
    assert exc.value.code == ErrorCode.CONFIG_INVALID
    assert aid in str(exc.value) and other in str(exc.value)


def test_bind_unbound_refuses(store):
    cfg = config_from_mapping({})  # no v3.retrieval section at all
    with pytest.raises(VerbatimError) as exc:
        _polart.bind_for_activation(cfg, None, store)
    assert exc.value.code == ErrorCode.CONFIG_INVALID
    assert "controller_policy_artifact" in str(exc.value)


# ---------------------------------------------------------------------------
# config gate — learned_active refuses naming the missing binding
# ---------------------------------------------------------------------------


def test_learned_active_without_binding_refuses_naming_it():
    with pytest.raises(VerbatimError) as exc:
        config_from_mapping({"v3": {"retrieval": {"controller": "learned_active"}}})
    assert exc.value.code == ErrorCode.CONFIG_INVALID
    assert "controller_policy_artifact" in str(exc.value)


def test_learned_active_with_binding_passes_config(store):
    aid = _register(store)
    cfg = config_from_mapping(_ACTIVE_CFG(aid))
    assert cfg.v3.retrieval.controller == "learned_active"
    assert cfg.v3.retrieval.controller_policy_artifact == aid


def test_memory_kill_switch_names_binding(tmp_path):
    """The consumer-level gate: Memory(config={learned_active}) refuses
    CONFIG_INVALID naming the missing artifact binding — not a bare
    'validated artifact required' (V6-03.15/F27)."""
    from verbatim import Memory
    with pytest.raises(VerbatimError) as exc:
        Memory(
            str(tmp_path / "k.db"), user_id="fb", worker="external",
            config={"v3": {"retrieval": {"controller": "learned_active"}}},
        )
    assert exc.value.code == ErrorCode.CONFIG_INVALID
    assert "controller_policy_artifact" in str(exc.value)


# ---------------------------------------------------------------------------
# activation through recall_v3 — the enforced binding point
# ---------------------------------------------------------------------------


def test_learned_active_unvalidated_refuses_at_recall(store):
    """Config accepts the declared binding; the store half refuses at the
    controller-construction point inside recall — unvalidated artifact,
    named."""
    _seeded(store)
    aid = _register(store)
    cfg = config_from_mapping(_ACTIVE_CFG(aid))
    with pytest.raises(VerbatimError) as exc:
        recall_v3(store, _request(), cfg=cfg)
    assert exc.value.code == ErrorCode.CONFIG_INVALID
    assert aid in str(exc.value)


def test_learned_active_validated_activates(store):
    """A registered+validated artifact activates: the learned plan applies
    (items_half halves the delivered pack) and the decision log records
    policy_revision=learned:<rev> — not a shadow row."""
    _seeded(store)
    aid = _register(store)
    _validate(store, aid)
    cfg = config_from_mapping(_ACTIVE_CFG(aid))
    out = recall_v3(store, _request(), cfg=cfg)
    assert not out.abstained
    assert any("learned_action:items_half" in w for w in out.warnings)
    items = [i for p in out.packs for i in p.items]
    assert len(items) <= 4  # mid 8 → 4 under items_half
    with store.read() as conn:
        row = conn.execute(
            "SELECT policy_revision FROM routing_decisions"
            " WHERE policy_revision LIKE 'learned:%'"
        ).fetchone()
        shadow = conn.execute(
            "SELECT COUNT(*) FROM routing_decisions"
            " WHERE policy_revision='learned_shadow_v1'"
        ).fetchone()[0]
    assert row is not None and row[0] == "learned:pol_v1"
    assert shadow == 0


def test_learned_active_revoked_after_validation_refuses(store):
    """Revoking a previously-validated artifact closes activation again —
    the withdrawal takes effect at the very next recall."""
    _seeded(store)
    aid = _register(store)
    _validate(store, aid)
    _revoke(store, aid)
    cfg = config_from_mapping(_ACTIVE_CFG(aid))
    with pytest.raises(VerbatimError) as exc:
        recall_v3(store, _request(), cfg=cfg)
    assert exc.value.code == ErrorCode.CONFIG_INVALID
    assert "revoked" in str(exc.value)


def test_learned_active_wrong_store_refuses(store, tmp_path):
    """Validation is per-store: an artifact validated in store A does not
    activate store B's learned controller."""
    aid = _register(store)
    _validate(store, aid)
    cfg = config_from_mapping(_ACTIVE_CFG(aid))
    (tmp_path / "b.db.key").write_bytes(_TEST_HMAC_KEY)
    other = Store.create(str(tmp_path / "b.db"))
    try:
        _seeded(other)
        with pytest.raises(VerbatimError) as exc:
            recall_v3(other, _request(), cfg=cfg)
        assert exc.value.code == ErrorCode.CONFIG_INVALID
    finally:
        other.close()


def test_deterministic_and_shadow_unaffected(store):
    """The binding path is opt-in: deterministic recalls never probe
    policy_artifacts (absent attestation table stays absent)."""
    _seeded(store)
    out = recall_v3(store, _request())
    assert not out.abstained
    with store.read() as conn:
        assert not has_table(conn, "policy_artifact_attestations")
