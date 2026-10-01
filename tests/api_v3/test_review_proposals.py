"""``ReviewMixin.propose_transition`` coverage (audit F6).

The public ``Engine.propose_transition`` method — the v1/v2 "propose a
lifecycle effect for operator review" entry point — had a real
implementation but zero callers and zero tests. These tests exercise the
full propose → approve loop the spec intends (SPEC §15/§17, SPEC_V2
§19–§20): ``propose_transition`` records an open review carrying the
proposed effect and the fenced expected versions; approval applies the
effect and resolves the review in ONE transaction through
``apply_proposal`` with ``params.review_id`` — the same convention the
CLI's ``reviews approve`` path uses (``cli._proposal_for_review``).

This test module lives under tests/api_v3/ per the change-scope
constraint of the audit-fix wave; it exercises the v2 ``Engine`` facade
(``verbatim.api``), not ``VerbatimV3``.
"""

from __future__ import annotations

import pytest

from verbatim.api import open_store
from verbatim.config import config_from_mapping
from verbatim.core.types import (
    EffectProposal,
    ErrorCode,
    Provenance,
    SourceEnvelope,
    SourceKind,
    TransitionCommand,
    VerbatimError,
)
from verbatim.host import LocalHost
from verbatim.storage.repos import ReviewsRepo

T0 = 1_700_000_000_000_000


def _cfg():
    return config_from_mapping(
        {
            "mode": "offline_rules",
            "capture": {"enabled": True},
            "admission": {"require_review": True},
        }
    )


def _open(tmp_path):
    return open_store(
        str(tmp_path),
        _cfg(),
        LocalHost(profile_id="demo", principal_id="me", conversation_id="c1"),
        create=True,
    )


def _env(text, scope):
    return SourceEnvelope(
        origin="v2test",
        source_kind=SourceKind.USER_MESSAGE,
        scope=scope,
        speaker_id="me",
        payload=text.encode("utf-8"),
        event_us=T0,
        captured_us=T0,
        provenance=Provenance.DIRECT_USER,
    )


def _ingest_claim(eng, scope, text):
    eng.ingest(_env(text, scope))
    eng.run_pending(limit=64)


def _heads(eng):
    with eng.store.read() as conn:
        rows = conn.execute(
            "SELECT cr.claim_id, cr.revision, cr.state FROM claim_revisions cr"
            " JOIN (SELECT claim_id, MAX(revision) m FROM claim_revisions"
            "       GROUP BY claim_id) h"
            "   ON h.claim_id = cr.claim_id AND h.m = cr.revision"
        ).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def _proposal(effect, targets, op, actor="me", reason="approve", **kw):
    return EffectProposal(
        version=1,
        operation_id=op,
        effect=effect,
        targets=targets,
        actor_id=actor,
        reason=reason,
        **kw,
    )


def test_propose_transition_admit_then_approve(tmp_path):
    """propose_transition records an open review; apply_proposal with the
    review id applies the effect and resolves it atomically."""
    eng = _open(tmp_path)
    try:
        scope = eng.host.default_scope()
        _ingest_claim(eng, scope, "The deploy moved to Friday.")
        cid, (rev, state) = next(iter(_heads(eng).items()))
        assert state == "pending"

        rid = eng.propose_transition(
            TransitionCommand(
                claim_id=cid,
                expected_revision=rev,
                effect="admit",
                actor_id="me",
                reason="evidence verified",
            ),
            scope,
        )
        review = ReviewsRepo(eng.store).get(rid)
        assert review["state"] == "open"
        assert review["proposed_effect"]["effect"] == "admit"
        assert review["expected_versions"] == {cid: rev}
        # The proposal alone never mutates the claim.
        assert _heads(eng)[cid][1] == "pending"

        receipt = eng.apply_proposal(
            _proposal(
                "admit",
                ((cid, rev),),
                f"review-approve:{rid}",
                params={"review_id": rid},
            )
        )
        assert receipt["review_id"] == rid
        assert receipt["targets"][0]["state"] == "active"
        assert _heads(eng)[cid][1] == "active"
        review = ReviewsRepo(eng.store).get(rid)
        assert review["state"] == "approved"
        assert review["resolved_event"] == receipt["committed_event"]
    finally:
        eng.close()


def test_propose_transition_supersede_loop(tmp_path):
    """``effect='supersede'`` delegates to propose_supersede; approval
    supersedes the predecessor and resolves the review in one tx."""
    eng = _open(tmp_path)
    try:
        scope = eng.host.default_scope()
        _ingest_claim(eng, scope, "Standup is at nine.")
        _ingest_claim(eng, scope, "Standup moved to ten.")
        c1, c2 = list(_heads(eng))
        for cid in (c1, c2):
            eng.apply_proposal(
                _proposal("admit", ((cid, 1),), f"op-ad-{cid[:8]}")
            )
        rev1, rev2 = _heads(eng)[c1][0], _heads(eng)[c2][0]

        rid = eng.propose_transition(
            TransitionCommand(
                claim_id=c1,
                expected_revision=rev1,
                effect="supersede",
                actor_id="me",
                reason="newer fact",
                successor_claim_id=c2,
            ),
            scope,
        )
        review = ReviewsRepo(eng.store).get(rid)
        assert review["state"] == "open"
        assert review["proposed_effect"]["successor_id"] == c2
        assert review["expected_versions"] == {c1: rev1, c2: rev2}

        receipt = eng.apply_proposal(
            _proposal(
                "supersede",
                ((c1, rev1),),
                f"review-approve:{rid}",
                successor_claim_id=c2,
                params={
                    "review_id": rid,
                    "successor_expected_revision": rev2,
                },
            )
        )
        assert receipt["review_id"] == rid
        assert _heads(eng)[c1][1] == "superseded"
        assert _heads(eng)[c2][1] == "active"
        assert ReviewsRepo(eng.store).get(rid)["state"] == "approved"
    finally:
        eng.close()


def test_propose_transition_denied_and_missing(tmp_path):
    """A bound caller without PROPOSE is denied; a foreign claim id
    resolves identically to a missing one (§10.05 opacity)."""
    from verbatim.core.types import CallerContext, GrantKind

    eng = _open(tmp_path)
    try:
        scope = eng.host.default_scope()
        _ingest_claim(eng, scope, "Scoped evidence.")
        cid, (rev, _state) = next(iter(_heads(eng).items()))

        reader = CallerContext(
            profile_id="demo",
            principal_id="me",
            conversation_id="c1",
            grants=frozenset({GrantKind.READ_EVIDENCE}),
        )
        with pytest.raises(VerbatimError) as ei:
            eng.propose_transition(
                TransitionCommand(
                    claim_id=cid,
                    expected_revision=rev,
                    effect="admit",
                    actor_id="agent",
                    reason="x",
                ),
                scope,
                caller=reader,
            )
        assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN

        with pytest.raises(VerbatimError) as ei2:
            eng.propose_transition(
                TransitionCommand(
                    claim_id="no-such-claim",
                    expected_revision=1,
                    effect="admit",
                    actor_id="me",
                    reason="x",
                ),
                scope,
            )
        assert ei2.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN
    finally:
        eng.close()
