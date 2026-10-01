"""EgressGate tests: consent, budget reservation lifecycle, minimization."""

from __future__ import annotations

from decimal import Decimal

import pytest

from verbatim.config import JudgeConfig, VerbatimConfig
from verbatim.core.types import ErrorCode, Mode, VerbatimError
from verbatim.privacy.egress import EgressGate, build_state, minimize
from tests.conftest import grant_consent, qrow


def _cfg(mode=Mode.JEV_ASSISTED, budget="1.00", backend="jev"):
    return VerbatimConfig(
        mode=mode,
        judge=JudgeConfig(backend=backend, daily_budget_usd=Decimal(budget)),
    )


@pytest.fixture()
def gate(store):
    return EgressGate(store, _cfg())


def test_check_only_requires_mode_consent_budget(store, gate, scope_id):
    # no consent → False
    assert gate.check_only(scope_id, "typesafe", "candidate_curation") is False
    grant_consent(store, scope_id)
    assert gate.check_only(scope_id, "typesafe", "candidate_curation") is True
    # wrong purpose → False
    assert gate.check_only(scope_id, "typesafe", "query_rerank") is False
    # offline mode → False even with consent
    off = EgressGate(store, _cfg(mode=Mode.OFFLINE_RULES, backend="rules"))
    assert off.check_only(scope_id, "typesafe", "candidate_curation") is False
    # zero budget → False
    broke = EgressGate(store, _cfg(budget="0"))
    assert broke.check_only(scope_id, "typesafe", "candidate_curation") is False


def test_authorize_reserves(store, gate, scope_id):
    grant_consent(store, scope_id)
    rid = gate.authorize(scope_id, "typesafe", "candidate_curation", 1000)
    assert rid
    with store.read() as conn:
        row = qrow(conn, "SELECT * FROM budget_ledger WHERE reservation_id = ?", (rid,))
    assert row["state"] == "reserved"
    assert row["token_bound"] == 1000
    # 42 microUSD/1K tokens → 1000 tokens = 42 microUSD
    assert row["reserved_cost_microusd"] == 42
    assert gate.day_spend() == 42


def test_authorize_no_consent(store, gate, scope_id):
    with pytest.raises(VerbatimError) as ei:
        gate.authorize(scope_id, "typesafe", "candidate_curation", 100)
    assert ei.value.code == ErrorCode.EGRESS_DISABLED


def test_authorize_mode_off(store, scope_id):
    grant_consent(store, scope_id)
    off = EgressGate(store, _cfg(mode=Mode.OFFLINE_RULES, backend="rules"))
    with pytest.raises(VerbatimError) as ei:
        off.authorize(scope_id, "typesafe", "candidate_curation", 100)
    assert ei.value.code == ErrorCode.EGRESS_DISABLED


def test_authorize_budget_exhausted(store, scope_id):
    grant_consent(store, scope_id)
    tiny = EgressGate(store, _cfg(budget="0.0001"))  # 100 microUSD
    rid = tiny.authorize(scope_id, "typesafe", "candidate_curation", 2000)  # 84 micro
    assert rid
    with pytest.raises(VerbatimError) as ei:
        tiny.authorize(scope_id, "typesafe", "candidate_curation", 2000)
    assert ei.value.code == ErrorCode.BUDGET_EXHAUSTED
    assert ei.value.retryable is False


def test_settle_actual_and_overrun(store, gate, scope_id):
    grant_consent(store, scope_id)
    rid = gate.authorize(scope_id, "typesafe", "candidate_curation", 1000)  # 42 reserved
    gate.settle(rid, 500)  # actual 21 micro < reserved
    with store.read() as conn:
        row = qrow(conn, "SELECT * FROM budget_ledger WHERE reservation_id = ?", (rid,))
    assert row["state"] == "settled"
    assert row["actual_cost_microusd"] == 21
    assert gate.day_spend() == 21

    rid2 = gate.authorize(scope_id, "typesafe", "candidate_curation", 1000)
    gate.settle(rid2, 5000)  # actual 210 > reserved 42 → overrun
    with store.read() as conn:
        row2 = qrow(conn, "SELECT * FROM budget_ledger WHERE reservation_id = ?", (rid2,))
    assert row2["state"] == "overrun"
    assert row2["actual_cost_microusd"] == 210
    # day spend counts actuals (21 + 210)
    assert gate.day_spend() == 231


def test_expire_counts_full_estimate(store, gate, scope_id):
    grant_consent(store, scope_id)
    rid = gate.authorize(scope_id, "typesafe", "candidate_curation", 2000)  # 84
    gate.expire(rid)
    with store.read() as conn:
        row = qrow(conn, "SELECT * FROM budget_ledger WHERE reservation_id = ?", (rid,))
    assert row["state"] == "expired"
    assert row["actual_cost_microusd"] == row["reserved_cost_microusd"]
    assert gate.day_spend() == 84
    # already-settled rows are untouched by expire
    rid2 = gate.authorize(scope_id, "typesafe", "candidate_curation", 1000)
    gate.settle(rid2, 100)
    gate.expire(rid2)
    with store.read() as conn:
        row2 = qrow(conn, "SELECT * FROM budget_ledger WHERE reservation_id = ?", (rid2,))
    assert row2["state"] == "settled"


def test_revocation_blocks_new_dispatch(store, gate, scope_id):
    cid = grant_consent(store, scope_id)
    assert gate.check_only(scope_id, "typesafe", "candidate_curation") is True
    with store.tx() as conn:
        conn.execute(
            "UPDATE consents SET revoked_us = 1 WHERE consent_id = ?", (cid,)
        )
    assert gate.check_only(scope_id, "typesafe", "candidate_curation") is False
    with pytest.raises(VerbatimError) as ei:
        gate.authorize(scope_id, "typesafe", "candidate_curation", 10)
    assert ei.value.code == ErrorCode.EGRESS_DISABLED


def test_unknown_model_price_fails_closed(store, scope_id):
    grant_consent(store, scope_id)
    cfg = _cfg(backend="jev")
    object.__setattr__(cfg.judge, "model", "jev-99.0.0")
    g = EgressGate(store, cfg)
    with pytest.raises(VerbatimError) as ei:
        g.authorize(scope_id, "typesafe", "candidate_curation", 10)
    assert ei.value.code == ErrorCode.CONFIG_INVALID


def test_settle_unknown_reservation(gate):
    with pytest.raises(VerbatimError) as ei:
        gate.settle("nope", 1)
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


def _purge_row(store, scope_id, state, purge_id="purge-1"):
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO purges (purge_id, selection_digest, scope_id, state,"
            " requested_us) VALUES (?,?,?,?,?)",
            (purge_id, b"\x00" * 32, scope_id, state, 1),
        )


def test_purge_suppression_blocks_dispatch(store, gate, scope_id):
    """SPEC §27: every dispatch rechecks purge suppression — a live
    erasure tombstone pauses remote disclosure of scope-derived data."""
    grant_consent(store, scope_id)
    assert gate.check_only(scope_id, "typesafe", "candidate_curation") is True
    _purge_row(store, scope_id, "suppressed")
    assert gate.check_only(scope_id, "typesafe", "candidate_curation") is False
    with pytest.raises(VerbatimError) as ei:
        gate.authorize(scope_id, "typesafe", "candidate_curation", 10)
    assert ei.value.code == ErrorCode.EGRESS_DISABLED
    with store.tx() as conn:
        conn.execute("UPDATE purges SET state = 'purging' WHERE purge_id = 'purge-1'")
    assert gate.check_only(scope_id, "typesafe", "candidate_curation") is False
    # completed erasure leaves the scope clean — egress resumes for
    # surviving content (erased objects stay tombstoned elsewhere)
    with store.tx() as conn:
        conn.execute("UPDATE purges SET state = 'completed' WHERE purge_id = 'purge-1'")
    assert gate.check_only(scope_id, "typesafe", "candidate_curation") is True


def test_purge_preview_does_not_block(store, gate, scope_id):
    """A preview is not yet an erasure order."""
    grant_consent(store, scope_id)
    _purge_row(store, scope_id, "previewed")
    assert gate.check_only(scope_id, "typesafe", "candidate_curation") is True


def test_suppression_check_failure_fails_closed(store, scope_id):
    grant_consent(store, scope_id)
    gate = EgressGate(
        store, _cfg(),
        is_scope_suppressed=lambda sid: (_ for _ in ()).throw(RuntimeError("db")),
    )
    assert gate.check_only(scope_id, "typesafe", "candidate_curation") is False
    with pytest.raises(VerbatimError):
        gate.authorize(scope_id, "typesafe", "candidate_curation", 10)


def test_policy_digest_epoch_binding(store, scope_id):
    """Consent epoch: a grant under an older policy digest must not carry
    over silently (SPEC §27 rechecks consent epoch per dispatch)."""
    grant_consent(store, scope_id)  # conftest grants under "digest-1"
    bound = EgressGate(store, _cfg(), policy_digest="digest-1")
    assert bound.check_only(scope_id, "typesafe", "candidate_curation") is True
    stale = EgressGate(store, _cfg(), policy_digest="digest-2")
    assert stale.check_only(scope_id, "typesafe", "candidate_curation") is False
    with pytest.raises(VerbatimError) as ei:
        stale.authorize(scope_id, "typesafe", "candidate_curation", 10)
    assert ei.value.code == ErrorCode.EGRESS_DISABLED


# ------------------------------------------------------------ minimization


def test_minimize_role_map_and_mapping():
    text = "Alice told Bob the deploy key is api_key=abcdef123456."
    out, mapping = minimize(text, {"Alice": "SPEAKER_A", "Bob": "SPEAKER_B"})
    assert "Alice" not in out and "Bob" not in out
    assert "SPEAKER_A" in out and "SPEAKER_B" in out
    assert mapping == {"SPEAKER_A": "Alice", "SPEAKER_B": "Bob"}
    assert "abcdef123456" not in out
    assert "[REDACTED]" in out


def test_minimize_emails_phones_pem():
    text = (
        "Mail me at jane.doe@example.com or call +1 415-555-0132.\n"
        "-----BEGIN PRIVATE KEY-----\nABC\n-----END PRIVATE KEY-----"
    )
    out, _ = minimize(text)
    assert "jane.doe@example.com" not in out
    assert "415-555-0132" not in out
    assert "PRIVATE KEY" not in out
    assert out.count("[REDACTED]") >= 3


def test_minimize_no_role_map():
    out, mapping = minimize("nothing sensitive here")
    assert out == "nothing sensitive here"
    assert mapping == {}


def test_build_state_bounds_and_shape():
    st = build_state(["quote one", "quote two"], {"time_overlap": "unknown"})
    assert st == {
        "quotes": ["quote one", "quote two"],
        "qualifiers": {"time_overlap": "unknown"},
    }
    with pytest.raises(VerbatimError) as ei:
        build_state(["x" * 20000])
    assert ei.value.code == ErrorCode.EVIDENCE_TOO_LARGE
    with pytest.raises(VerbatimError):
        build_state("not-a-list")  # type: ignore[arg-type]
