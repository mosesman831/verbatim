"""Policy reducer tests (SPEC §13, §16, §17, §23, §26): review-gated default,
rules admission, pair classification producing reviews (never transitions),
judge abstention paths, and recorded decisions."""

from __future__ import annotations

import pytest

from verbatim.core.claims import propose
from verbatim.core.lifecycle import read_claim_head
from verbatim.core.policy import (
    PolicyContext,
    admit,
    propose_supersede,
    relate,
)
from verbatim.core.types import (
    DecisionRequest,
    DecisionResult,
    ErrorCode,
    Lifecycle,
    Provenance,
    SourceKind,
    VerbatimError,
    json_dumps,
    safe_json_loads,
)
from verbatim.config import (
    AdmissionConfig,
    CaptureConfig,
    VerbatimConfig,
)
from verbatim.storage.repos import ReviewsRepo

from tests.core.conftest import (
    make_envelope,
    q,
    seed_claim,
    seed_span_for,
)


def _cfg(**kw):
    capture = kw.pop("capture", CaptureConfig(enabled=True, user_messages=True))
    admission = kw.pop("admission", AdmissionConfig(require_review=True))
    return VerbatimConfig(capture=capture, admission=admission, **kw)


def _proposal_for(store, scope, text, **env_kw):
    """Harvest-envelope-proposal pipeline as the ingest path would run it."""
    env = make_envelope(scope, text, **env_kw)
    span = seed_span_for(store, env)
    p = propose(text, span, env, env.event_us)
    return env, p


def _review_rows(store):
    return q(
        store,
        "SELECT review_id, scope_id, proposed_effect_json, state,"
        " expected_versions_json FROM reviews",
    )


# ---------------------------------------------------------------------------
# admit — capture gate
# ---------------------------------------------------------------------------


def test_admit_capture_disabled(store, scope):
    ctx = PolicyContext(cfg=_cfg(capture=CaptureConfig(enabled=False)))
    env, p = _proposal_for(store, scope, "I use Neovim.")
    out = admit(store, p, env, ctx)
    assert out.claim_id is None
    assert out.reason == "capture_disabled"
    assert q(store, "SELECT COUNT(*) FROM claims")[0][0] == 0


def test_admit_kind_not_allowed(store, scope):
    ctx = PolicyContext(cfg=_cfg())
    env, p = _proposal_for(
        store,
        scope,
        "I use Neovim.",
        kind=SourceKind.TOOL_OUTPUT,
        provenance=Provenance.APPROVED_TOOL,
    )
    out = admit(store, p, env, ctx)
    assert out.claim_id is None
    assert out.reason == "capture_kind_not_allowed"


# ---------------------------------------------------------------------------
# admit — review-gated default
# ---------------------------------------------------------------------------


def test_admit_review_gated_default(store, scope):
    ctx = PolicyContext(cfg=_cfg())  # require_review=True is the default
    env, p = _proposal_for(store, scope, "I use Neovim.")
    out = admit(store, p, env, ctx)
    assert out.state == Lifecycle.PENDING
    assert out.claim_id is not None
    assert out.review_id is not None
    assert out.reason == "review_required"
    rows = _review_rows(store)
    assert len(rows) == 1
    effect = safe_json_loads(rows[0][2])
    assert effect["effect"] == "admit" and effect["claim_id"] == out.claim_id
    assert rows[0][3] == "open"
    with store.tx() as conn:
        head = read_claim_head(conn, out.claim_id)
        assert head.state == Lifecycle.PENDING


def test_admit_rules_path_whitelisted(store, scope):
    ctx = PolicyContext(cfg=_cfg(admission=AdmissionConfig(require_review=False)))
    env, p = _proposal_for(store, scope, "I use Neovim.")
    out = admit(store, p, env, ctx)
    assert out.state == Lifecycle.ACTIVE
    assert out.reason == "rule_whitelist"
    assert out.review_id is None


def test_admit_rules_path_abstains_unstructured(store, scope):
    ctx = PolicyContext(cfg=_cfg(admission=AdmissionConfig(require_review=False)))
    env, p = _proposal_for(store, scope, "The weather is nice today.")
    out = admit(store, p, env, ctx)
    assert out.state == Lifecycle.PENDING
    assert out.reason == "abstained"
    assert out.review_id is not None


def test_admit_explicit_remember_rules_path(store, scope):
    ctx = PolicyContext(cfg=_cfg(admission=AdmissionConfig(require_review=False)))
    env, p = _proposal_for(store, scope, "Remember that I like tea.")
    assert p.method == "explicit_remember"
    out = admit(store, p, env, ctx)
    assert out.state == Lifecycle.ACTIVE
    assert out.reason == "explicit_remember"


def test_admit_hypothetical_abstains(store, scope):
    ctx = PolicyContext(cfg=_cfg(admission=AdmissionConfig(require_review=False)))
    env, p = _proposal_for(store, scope, "I might switch to Helix.")
    out = admit(store, p, env, ctx)
    assert out.state == Lifecycle.PENDING


def test_admit_sensitive_hint_reviewed(store, scope):
    ctx = PolicyContext(cfg=_cfg(admission=AdmissionConfig(require_review=False)))
    env, p = _proposal_for(store, scope, "my api key is abc123def456ghi789jklmno")
    out = admit(store, p, env, ctx)
    assert out.state == Lifecycle.PENDING
    assert out.reason == "sensitive_hint"
    assert out.review_id is not None


def test_admit_sensitive_slot_reviewed(store, scope):
    # residence is a sensitive slot → review even on the rules path
    ctx = PolicyContext(cfg=_cfg(admission=AdmissionConfig(require_review=False)))
    env, p = _proposal_for(store, scope, "I live in Oslo.")
    out = admit(store, p, env, ctx)
    assert out.state == Lifecycle.PENDING
    assert out.reason == "sensitive_hint"


def test_admit_writes_event(store, scope):
    ctx = PolicyContext(cfg=_cfg())
    env, p = _proposal_for(store, scope, "I use Neovim.")
    admit(store, p, env, ctx)
    row = q(store, "SELECT kind, policy_version FROM events WHERE kind='claim_proposed'")[0]
    assert row is not None and row[1] == "policy-1"


# ---------------------------------------------------------------------------
# admit — redelivery convergence (primary-span heal)
# ---------------------------------------------------------------------------


def test_admit_retry_reuses_claim_pending(store, scope):
    """A redelivered admission for the same primary span resolves to the
    claim the first attempt committed — the span→claim link is the durable
    dedup key, so a crash after the claim commit but before the job
    receipt cannot mint a duplicate."""
    ctx = PolicyContext(cfg=_cfg())
    env, p = _proposal_for(store, scope, "I use Neovim.")
    first = admit(store, p, env, ctx)
    assert first.state == Lifecycle.PENDING
    second = admit(store, p, env, ctx)
    assert second.claim_id == first.claim_id
    assert second.state == Lifecycle.PENDING
    assert second.reason == "already_admitted"
    assert second.review_id == first.review_id  # same open review, no twin
    # exactly one claim / revision / evidence link / review row exists
    assert q(store, "SELECT COUNT(*) FROM claims")[0][0] == 1
    assert q(store, "SELECT COUNT(*) FROM claim_revisions")[0][0] == 1
    assert q(store, "SELECT COUNT(*) FROM claim_evidence")[0][0] == 1
    assert len(_review_rows(store)) == 1


def test_admit_retry_reuses_claim_active(store, scope):
    """Same convergence on the auto-activated path — a replayed admit
    returns the live claim rather than minting a shadow."""
    ctx = PolicyContext(cfg=_cfg(admission=AdmissionConfig(require_review=False)))
    env, p = _proposal_for(store, scope, "I use Neovim.")
    first = admit(store, p, env, ctx)
    assert first.state == Lifecycle.ACTIVE
    second = admit(store, p, env, ctx)
    assert second.claim_id == first.claim_id
    assert second.state == Lifecycle.ACTIVE
    assert second.reason == "already_admitted"
    assert second.review_id is None
    assert q(store, "SELECT COUNT(*) FROM claims")[0][0] == 1


def test_admit_distinct_span_still_admits(store, scope):
    """The heal keys on the *span*, not the text — a genuinely new span
    (fresh ingest of similar material) is new work, not a retry."""
    ctx = PolicyContext(cfg=_cfg(admission=AdmissionConfig(require_review=False)))
    env1, p1 = _proposal_for(store, scope, "I use Neovim.")
    env2, p2 = _proposal_for(store, scope, "I use Neovim.")
    first = admit(store, p1, env1, ctx)
    second = admit(store, p2, env2, ctx)
    assert first.claim_id != second.claim_id
    assert q(store, "SELECT COUNT(*) FROM claims")[0][0] == 2


# ---------------------------------------------------------------------------
# admit — judge paths
# ---------------------------------------------------------------------------


class FakeJudge:
    """Deterministic fixture backend — never a real inference call (§23)."""

    def __init__(self, outcome=None, abstained=False, exc=None):
        self.outcome = outcome or {"label": "durable"}
        self.abstained = abstained
        self.exc = exc
        self.requests: list[DecisionRequest] = []
        self.backend = "fake"

    def evaluate(self, req: DecisionRequest) -> DecisionResult:
        self.requests.append(req)
        if self.exc:
            raise self.exc
        return DecisionResult(
            task=req.task,
            backend="fake",
            model_revision="fake-1",
            rubric_version=req.rubric_version,
            outcome=dict(self.outcome),
            abstained=self.abstained,
            reason="fixture",
        )


def test_admit_judge_durable_allows_whitelisted(store, scope):
    judge = FakeJudge({"label": "durable"})
    ctx = PolicyContext(
        cfg=_cfg(admission=AdmissionConfig(require_review=False)), judge=judge
    )
    env, p = _proposal_for(store, scope, "I use Neovim.")
    out = admit(store, p, env, ctx)
    assert out.state == Lifecycle.ACTIVE
    assert len(out.decision_ids) == 1
    row = q(
        store,
        "SELECT task, backend FROM decisions WHERE decision_id=?",
        (out.decision_ids[0],),
    )[0]
    assert row == ("durability", "fake")
    inp = q(
        store,
        "SELECT object_kind FROM decision_inputs WHERE decision_id=?",
        (out.decision_ids[0],),
    )
    assert {r[0] for r in inp} == {"claim", "span"}


def test_admit_judge_not_durable_pending(store, scope):
    judge = FakeJudge({"label": "not_durable"})
    ctx = PolicyContext(
        cfg=_cfg(admission=AdmissionConfig(require_review=False)), judge=judge
    )
    env, p = _proposal_for(store, scope, "I use Neovim.")
    out = admit(store, p, env, ctx)
    assert out.state == Lifecycle.PENDING
    assert out.reason == "judge_not_durable"
    assert out.review_id is not None


def test_admit_judge_abstain_pending(store, scope):
    judge = FakeJudge(abstained=True)
    ctx = PolicyContext(
        cfg=_cfg(admission=AdmissionConfig(require_review=False)), judge=judge
    )
    env, p = _proposal_for(store, scope, "I use Neovim.")
    out = admit(store, p, env, ctx)
    assert out.state == Lifecycle.PENDING
    assert out.reason == "judge_abstain"


def test_admit_judge_error_is_abstention_not_crash(store, scope):
    judge = FakeJudge(exc=RuntimeError("backend down"))
    ctx = PolicyContext(
        cfg=_cfg(admission=AdmissionConfig(require_review=False)), judge=judge
    )
    env, p = _proposal_for(store, scope, "I use Neovim.")
    out = admit(store, p, env, ctx)
    assert out.state == Lifecycle.PENDING
    assert out.review_id is not None


# ---------------------------------------------------------------------------
# relate — pair classification → reviews, never transitions
# ---------------------------------------------------------------------------


def _active_claim(store, scope, predicate, obj, cond=None, subject="alice", text=None):
    return seed_claim(
        store,
        scope,
        predicate=predicate,
        object_text=obj,
        state="active",
        condition_json=cond,
        subject_id=subject,
        span_text=text or f"claim about {obj}",
    )


def test_relate_incompatible_creates_review_not_transition(store, scope):
    old, _ = _active_claim(store, scope, "editor", "neovim")
    new, _ = _active_claim(store, scope, "editor", "vscode")
    ctx = PolicyContext(cfg=_cfg())
    out = relate(store, new, ctx)
    # edge + review
    assert len(out) == 2
    rows = _review_rows(store)
    assert len(rows) == 1
    effect = safe_json_loads(rows[0][2])
    assert effect["effect"] == "supersede"
    assert effect["predecessor_id"] == old
    assert effect["successor_id"] == new
    # claims untouched — reviews propose, they never apply
    with store.tx() as conn:
        assert read_claim_head(conn, old).state == Lifecycle.ACTIVE
        assert read_claim_head(conn, new).state == Lifecycle.ACTIVE
    edge = q(
        store,
        "SELECT source_id, target_id, edge_type FROM edges"
        " WHERE edge_type='conflicts_with'",
    )[0]
    assert edge is not None
    assert {edge[0], edge[1]} == {old, new}


def test_relate_retry_dedups_open_review(store, scope):
    """A redelivered relate pass converges on the queued artifacts: the
    edge add is already idempotent, and the supersession proposal now
    resolves to the open review instead of queuing a twin."""
    _active_claim(store, scope, "editor", "neovim")
    new, _ = _active_claim(store, scope, "editor", "vscode")
    ctx = PolicyContext(cfg=_cfg())
    first = relate(store, new, ctx)
    second = relate(store, new, ctx)
    assert len(first) == 2  # edge + review
    assert second == first
    rows = _review_rows(store)
    assert len(rows) == 1
    assert rows[0][3] == "open"
    assert q(store, "SELECT COUNT(*) FROM edges")[0][0] == 1


def test_relate_resolved_review_allows_reproposal(store, scope):
    """Dedup only covers *open* reviews — once the pair is decided, a later
    conflict signal may legitimately re-propose it."""
    old, _ = _active_claim(store, scope, "editor", "neovim")
    new, _ = _active_claim(store, scope, "editor", "vscode")
    ctx = PolicyContext(cfg=_cfg())
    first = relate(store, new, ctx)
    with store.tx() as conn:
        ReviewsRepo(store).resolve(conn, first[-1], "rejected", 99)
    second = relate(store, new, ctx)
    assert len(second) == 2
    assert second[0] == first[0]  # same canonical edge
    assert second[1] != first[1]  # fresh review id
    rows = _review_rows(store)
    assert len(rows) == 2
    open_rows = [r for r in rows if r[3] == "open"]
    assert len(open_rows) == 1 and open_rows[0][0] == second[1]


def test_relate_different_scope_no_artifacts(store, scope):
    cond_work = json_dumps({"op": "eq", "key": "context", "value": "work"})
    cond_home = json_dumps({"op": "eq", "key": "context", "value": "home"})
    _active_claim(store, scope, "editor", "vscode", cond=cond_work)
    new, _ = _active_claim(store, scope, "editor", "neovim", cond=cond_home)
    out = relate(store, new, PolicyContext(cfg=_cfg()))
    assert out == []
    assert q(store, "SELECT COUNT(*) FROM edges")[0][0] == 0
    assert q(store, "SELECT COUNT(*) FROM reviews")[0][0] == 0


def test_relate_set_valued_slot_compatible(store, scope):
    _active_claim(store, scope, "tool_use", "docker")
    new, _ = _active_claim(store, scope, "tool_use", "kubernetes")
    out = relate(store, new, PolicyContext(cfg=_cfg()))
    assert out == []


def test_relate_different_subject_is_different_scope(store, scope):
    _active_claim(store, scope, "residence", "oslo", subject="alice")
    new, _ = _active_claim(store, scope, "residence", "bergen", subject="bob")
    out = relate(store, new, PolicyContext(cfg=_cfg()))
    assert out == []


def test_relate_equivalent_no_artifacts(store, scope):
    _active_claim(store, scope, "editor", "neovim")
    new, _ = _active_claim(store, scope, "editor", "neovim")
    out = relate(store, new, PolicyContext(cfg=_cfg()))
    assert out == []


def test_relate_judge_incompatible_creates_review(store, scope):
    old, _ = _active_claim(store, scope, "editor", "neovim")
    new, _ = _active_claim(store, scope, "editor", "vscode")
    judge = FakeJudge({"label": "incompatible"})
    ctx = PolicyContext(cfg=_cfg(), judge=judge)
    out = relate(store, new, ctx)
    assert len(out) == 2  # edge + review
    # decision recorded for the pair + change signal probe
    tasks = {r[0] for r in q(store, "SELECT DISTINCT task FROM decisions")}
    assert "pair_relation" in tasks
    assert "change_signal" in tasks
    with store.tx() as conn:
        assert read_claim_head(conn, old).state == Lifecycle.ACTIVE


def test_relate_judge_abstain_falls_back_to_rules(store, scope):
    _active_claim(store, scope, "tool_use", "docker")
    new, _ = _active_claim(store, scope, "tool_use", "kubernetes")
    judge = FakeJudge(abstained=True)
    ctx = PolicyContext(cfg=_cfg(), judge=judge)
    out = relate(store, new, ctx)
    assert out == []  # rules said compatible (set-valued)


def test_relate_unknown_claim(store):
    with pytest.raises(VerbatimError) as ei:
        relate(store, "ghost", PolicyContext(cfg=_cfg()))
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


def test_relate_event_written(store, scope):
    _active_claim(store, scope, "editor", "neovim")
    new, _ = _active_claim(store, scope, "editor", "vscode")
    relate(store, new, PolicyContext(cfg=_cfg()))
    row = q(store, "SELECT kind, payload_json FROM events WHERE kind='pair_comparison'")[0]
    assert row is not None
    payload = safe_json_loads(row[1])
    assert payload["new_claim_id"] == new


# ---------------------------------------------------------------------------
# propose_supersede
# ---------------------------------------------------------------------------


def test_propose_supersede_creates_review_with_versions(store, scope):
    old, old_rev = _active_claim(store, scope, "editor", "neovim")
    new, new_rev = _active_claim(store, scope, "editor", "helix")
    rid = propose_supersede(
        store, old, new, "operator", "user confirmed switch",
        ctx=PolicyContext(cfg=_cfg()),
    )
    rows = _review_rows(store)
    assert len(rows) == 1 and rows[0][0] == rid
    effect = safe_json_loads(rows[0][2])
    assert effect["effect"] == "supersede"
    versions = safe_json_loads(rows[0][4])
    assert versions == {old: old_rev, new: new_rev}


def test_propose_supersede_requires_active_predecessor(store, scope):
    old, _ = seed_claim(store, scope, state="pending")
    new, _ = _active_claim(store, scope, "editor", "helix")
    with pytest.raises(VerbatimError) as ei:
        propose_supersede(store, old, new, "op", "x", ctx=PolicyContext(cfg=_cfg()))
    assert ei.value.code == ErrorCode.INVALID_TRANSITION


def test_propose_supersede_bad_successor(store, scope):
    old, _ = _active_claim(store, scope, "editor", "neovim")
    dead, _ = seed_claim(store, scope, state="erased", object_text="x")
    with pytest.raises(VerbatimError) as ei:
        propose_supersede(store, old, dead, "op", "x", ctx=PolicyContext(cfg=_cfg()))
    assert ei.value.code == ErrorCode.INVALID_TRANSITION


def test_propose_supersede_missing_claim(store, scope):
    old, _ = _active_claim(store, scope, "editor", "neovim")
    with pytest.raises(VerbatimError) as ei:
        propose_supersede(store, old, "ghost", "op", "x", ctx=PolicyContext(cfg=_cfg()))
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN
