"""Learned-controller policy + recall-channel tests (SPEC_V3 §26.03/04,
§43.04/05, B36).

Covers: bounded action set, artifact validation/round-trip, warm-start
fallback on unseen states, forced-action replay arms, replay_learned
applied plans, deterministic fallback on missing/corrupt artifacts, and
real learned-shadow divergences.
"""

from __future__ import annotations

import pytest

from verbatim.core.types_v3 import (
    ActionIntent,
    BudgetTier,
    QueryClass,
    RecallRequestV3,
    Route,
)
from verbatim.retrieval.v3 import recall_v3
from verbatim.retrieval.v3 import controller as _ctrl
from verbatim.retrieval.v3 import learned as _learned
from verbatim.storage.store import Store

from tests.retrieval.v3.test_retrieval_v3 import (
    _TEST_HMAC_KEY,
    seed_auth,
    seed_claim,
    seed_scope,
)


@pytest.fixture
def store(tmp_path):
    # seed_claim signs source payloads/excerpts with _TEST_HMAC_KEY — the
    # store must hold the same key or packaging's integrity re-check
    # (fail-closed, C03) reports every seeded source as corrupt.
    (tmp_path / "v3.db.key").write_bytes(_TEST_HMAC_KEY)
    s = Store.create(str(tmp_path / "v3.db"))
    yield s
    s.close()


def _state(qc=QueryClass.CURRENT_STATE, tier=BudgetTier.MID) -> _ctrl._State:
    return _ctrl._State(
        query_class=qc,
        action_intent=ActionIntent.NONE,
        size_band="small",
        procedures=False,
        stuck=False,
        session_phase="none",
        freshness_required=False,
        tier=tier,
        has_identifiers=False,
        memory_kinds=(),
        prefetch=False,
    )


def _request(scope_id="sA", tier=BudgetTier.MID, **kw) -> RecallRequestV3:
    return RecallRequestV3(
        query="when does the release ship", scope_id=scope_id,
        caller_id="human:alice", purpose="recall", budget_tier=tier, **kw,
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


# ---------------------------------------------------------------------------
# artifact validation
# ---------------------------------------------------------------------------


def test_artifact_roundtrip(tmp_path):
    art = _learned.PolicyArtifact(
        revision="test_v1",
        q={"s1": {"tier_tighter": 0.8, "base": 0.4}},
        pulls={"s1": {"tier_tighter": 3, "base": 3}},
    )
    path = tmp_path / "pol.json"
    path.write_text(art.to_json())
    loaded = _learned.load_artifact(str(path))
    assert loaded is not None
    assert loaded.revision == "test_v1"
    assert loaded.q["s1"]["tier_tighter"] == 0.8


def test_artifact_rejects_bad_payloads():
    assert _learned.load_artifact(None) is None
    assert _learned.load_artifact({"kind": "other"}) is None
    assert _learned.load_artifact({"kind": "learned_policy_v1"}) is None
    assert _learned.load_artifact("not json") is None
    assert _learned.load_artifact(
        {"kind": "learned_policy_v1", "revision": "x",
         "actions": ["fly_to_moon"]}
    ) is None


def test_apply_delta_bounded_actions():
    state = _state()
    req = _request()
    base = _ctrl.plan_routes(req, None, None, state)
    # items_half tightens
    d = _learned.apply_delta(base, state, req, "items_half")
    assert d.budgets["max_items"] == max(1, base.budgets["max_items"] // 2)
    # tier_tighter moves toward LOW tier budgets
    d = _learned.apply_delta(base, state, req, "tier_tighter")
    assert d.budgets["candidate_cap"] == _ctrl.TIER_BUDGETS[BudgetTier.LOW][
        "candidate_cap"
    ]
    # declared drop removes exactly that lane
    d = _learned.apply_delta(base, state, req, "drop_temporal")
    assert "temporal" not in d.lanes
    # declared add appends a lane the class didn't already route
    d = _learned.apply_delta(base, state, req, "add_browse")
    assert "browse" in d.lanes
    # caller ceilings never exceeded even when tier loosens
    tight_req = _request(max_items=2)
    d = _learned.apply_delta(base, state, tight_req, "tier_looser")
    assert d.budgets["max_items"] <= 2


# ---------------------------------------------------------------------------
# choose — warm start + measured deviation
# ---------------------------------------------------------------------------


def test_choose_warm_start_unseen_state():
    art = _learned.PolicyArtifact(revision="t1")
    state = _state()
    req = _request()
    base = _ctrl.plan_routes(req, None, None, state)
    chosen, action, _ = _learned.choose(state, req, base, art)
    assert action == "base"
    assert chosen is base


def test_choose_deviates_on_measured_reward():
    state = _state()
    req = _request()
    base = _ctrl.plan_routes(req, None, None, state)
    skey = _ctrl._state_key(state)
    art = _learned.PolicyArtifact(
        revision="t2",
        q={skey: {"base": 0.3, "items_half": 0.9}},
        pulls={skey: {"base": 2, "items_half": 2}},
    )
    chosen, action, scores = _learned.choose(state, req, base, art)
    assert action == "items_half"
    assert chosen.budgets["max_items"] < base.budgets["max_items"]
    assert scores["items_half"] == 0.9


def test_choose_no_deviation_without_advantage():
    state = _state()
    req = _request()
    base = _ctrl.plan_routes(req, None, None, state)
    skey = _ctrl._state_key(state)
    art = _learned.PolicyArtifact(
        revision="t3",
        q={skey: {"base": 0.9, "items_half": 0.3}},
        pulls={skey: {"base": 2, "items_half": 2}},
    )
    chosen, action, _ = _learned.choose(state, req, base, art)
    assert action == "base"
    assert chosen is base


def test_choose_priors_cannot_deviate_unseen_state():
    """V3-26.03 warm start: feature priors may order, never select, at a
    state with zero executed pulls — no counterfactual labels (§43.04)."""
    state = _state()
    req = _request()
    base = _ctrl.plan_routes(req, None, None, state)
    art = _learned.PolicyArtifact(
        revision="t4",
        priors={"qc=current_state": {"items_half": 9.9}},
        # no q/pulls entries at all → unseen state
    )
    chosen, action, _ = _learned.choose(state, req, base, art)
    assert action == "base"
    assert chosen is base


def test_choose_unmeasured_action_cannot_win():
    """At a seen state, an action with zero pulls still cannot be
    selected — deviations require an executed measurement."""
    state = _state()
    req = _request()
    base = _ctrl.plan_routes(req, None, None, state)
    skey = _ctrl._state_key(state)
    art = _learned.PolicyArtifact(
        revision="t5",
        q={skey: {"base": -0.5}},           # base measured poorly
        pulls={skey: {"base": 3}},          # items_half: never executed
        priors={"qc=current_state": {"items_half": 9.9}},
    )
    chosen, action, _ = _learned.choose(state, req, base, art)
    assert action == "base"
    assert chosen is base


def test_artifact_rejects_nonfinite_values():
    skey = "s1"
    bad = _learned.PolicyArtifact(
        revision="nan", q={skey: {"base": float("nan")}}
    ).to_dict()
    assert _learned.load_artifact(bad) is None
    bad2 = _learned.PolicyArtifact(
        revision="inf", priors={"f": {"base": float("inf")}}
    ).to_dict()
    assert _learned.load_artifact(bad2) is None
    bad3 = _learned.PolicyArtifact(
        revision="neg", pulls={skey: {"base": -1}}
    ).to_dict()
    assert _learned.load_artifact(bad3) is None


# ---------------------------------------------------------------------------
# recall_v3 integration — forced / replay_learned / fallback / shadow
# ---------------------------------------------------------------------------


def test_forced_action_arm_applies_delta(store):
    _seeded(store)
    out = recall_v3(
        store, _request(),
        policy_tables={"controller": {"kind": "forced",
                                      "action": "items_half"}},
    )
    assert not out.abstained
    assert any("forced_action:items_half" in w for w in out.warnings)
    assert len([i for p in out.packs for i in p.items]) <= 4  # mid 8 → 4


def test_forced_unknown_action_falls_back(store):
    _seeded(store)
    out = recall_v3(
        store, _request(),
        policy_tables={"controller": {"kind": "forced",
                                      "action": "fly_to_moon"}},
    )
    assert not out.abstained
    assert any(
        "forced_action_unknown:fly_to_moon" in w for w in out.warnings
    )
    # deterministic plan ran — all seeded items delivered
    assert len([i for p in out.packs for i in p.items]) == 3


def test_replay_learned_applies_artifact(store):
    _seeded(store)
    state = _state()
    skey = _ctrl._state_key(state)
    art = _learned.PolicyArtifact(
        revision="pol_x",
        q={skey: {"base": 0.1, "items_half": 0.9}},
        pulls={skey: {"base": 3, "items_half": 3}},
    )
    out = recall_v3(
        store, _request(),
        policy_tables={
            "controller": {"kind": "replay_learned",
                           "artifact": art.to_dict()}
        },
    )
    assert any("learned_action:items_half" in w for w in out.warnings)
    assert len([i for p in out.packs for i in p.items]) <= 4


def test_replay_learned_missing_artifact_falls_back(store):
    _seeded(store)
    out = recall_v3(
        store, _request(),
        policy_tables={
            "controller": {"kind": "replay_learned", "artifact": None}
        },
    )
    assert any(
        "learned_artifact_missing" in w for w in out.warnings
    )
    # deterministic plan ran — full mid-tier item budget applies
    assert len([i for p in out.packs for i in p.items]) == 3


def test_learned_shadow_logs_real_choice(store):
    _seeded(store)
    state = _state()
    skey = _ctrl._state_key(state)
    art = _learned.PolicyArtifact(
        revision="pol_y",
        q={skey: {"base": 0.1, "drop_temporal": 0.9}},
        pulls={skey: {"base": 3, "drop_temporal": 3}},
    )
    out = recall_v3(
        store, _request(),
        policy_tables={
            "controller": {"kind": "learned_shadow",
                           "artifact": art.to_dict()}
        },
    )
    with store.read() as conn:
        row = conn.execute(
            "SELECT lane_set_json, policy_revision FROM routing_decisions"
            " WHERE policy_revision='learned_shadow_v1'"
        ).fetchone()
    assert row is not None
    import json as _json
    lanes = _json.loads(row[0])
    assert "temporal" not in lanes  # the learned drop landed in shadow


def test_learned_shadow_without_artifact_warm_starts(store):
    _seeded(store)
    out = recall_v3(
        store, _request(),
        policy_tables={"controller": {"kind": "learned_shadow"}},
    )
    with store.read() as conn:
        row = conn.execute(
            "SELECT lane_set_json FROM routing_decisions"
            " WHERE policy_revision='learned_shadow_v1'"
        ).fetchone()
    import json as _json
    assert row is not None
    assert "temporal" in _json.loads(row[0])  # identical to deterministic
