"""v2 host-facing API + CLI: apply_proposal, idempotent ingest/remember,
expanded CLI surface, Hermes provider caller wiring (SPEC_V2 §09, §20, §39,
§43–§46).

Real stores on disk throughout — no mocks inside the engine.
"""

from __future__ import annotations

import json
import os
import sqlite3

import pytest

from verbatim.api import open_store
from verbatim.cli import main as cli_main
from verbatim.config import config_from_mapping
from verbatim.core.identity import scope_key
from verbatim.core.time import now_us
from verbatim.core.types import (
    CallerContext,
    EffectProposal,
    ErrorCode,
    GrantKind,
    Provenance,
    RecallMode,
    RecallRequest,
    SourceEnvelope,
    SourceKind,
    VerbatimError,
)
from verbatim.host import LocalHost

T0 = 1_700_000_000_000_000


def _cfg(**over):
    m = {
        "mode": "offline_rules",
        "capture": {"enabled": True},
        "admission": {"require_review": True},
    }
    m.update(over)
    return config_from_mapping(m)


def _open(tmp_path, profile="demo", principal="me", conv="c1", cfg=None):
    return open_store(
        str(tmp_path),
        cfg or _cfg(),
        LocalHost(profile_id=profile, principal_id=principal, conversation_id=conv),
        create=True,
    )


def _env(text, scope, **kw):
    return SourceEnvelope(
        origin="v2test",
        source_kind=SourceKind.USER_MESSAGE,
        scope=scope,
        speaker_id="me",
        payload=text.encode("utf-8"),
        event_us=T0,
        captured_us=T0,
        provenance=Provenance.DIRECT_USER,
        **kw,
    )


def _heads(eng):
    with eng.store.read() as conn:
        rows = conn.execute(
            "SELECT cr.claim_id, cr.revision, cr.state FROM claim_revisions cr"
            " JOIN (SELECT claim_id, MAX(revision) m FROM claim_revisions"
            "       GROUP BY claim_id) h"
            "   ON h.claim_id = cr.claim_id AND h.m = cr.revision"
        ).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def _proposal(effect, targets, op, actor="me", reason="test", **kw):
    return EffectProposal(
        version=1,
        operation_id=op,
        effect=effect,
        targets=targets,
        actor_id=actor,
        reason=reason,
        **kw,
    )


def _ingest_claim(eng, scope, text):
    eng.ingest(_env(text, scope))
    eng.run_pending(limit=64)


# ---------------------------------------------------------------------------
# Engine.apply_proposal
# ---------------------------------------------------------------------------


def test_apply_proposal_admit_receipt_and_replay(tmp_path):
    eng = _open(tmp_path)
    try:
        scope = eng.host.default_scope()
        _ingest_claim(eng, scope, "The deploy moved to Friday.")
        cid, (rev, state) = next(iter(_heads(eng).items()))
        assert state == "pending"

        prop = _proposal("admit", ((cid, rev),), "op-admit-1")
        receipt = eng.apply_proposal(prop)
        assert receipt["operation_id"] == "op-admit-1"
        assert receipt["effect"] == "admit"
        assert receipt["replayed"] is False
        assert receipt["targets"][0]["state"] == "active"
        assert _heads(eng)[cid][1] == "active"

        # Replay: same stored receipt, no second mutation.
        again = eng.apply_proposal(prop)
        assert again["replayed"] is True
        assert again["committed_event"] == receipt["committed_event"]
        assert _heads(eng)[cid][0] == receipt["targets"][0]["revision"]

        # Same operation key + different input = conflict, not a re-apply.
        with pytest.raises(VerbatimError) as ei:
            eng.apply_proposal(_proposal("archive", ((cid, 2),), "op-admit-1"))
        assert ei.value.code == ErrorCode.VALIDATION
    finally:
        eng.close()


def test_apply_proposal_supersede_via_review(tmp_path):
    """propose_supersede review → apply_proposal resolves both atomically."""
    from verbatim.core.policy import propose_supersede
    from verbatim.storage.repos import ReviewsRepo

    eng = _open(tmp_path)
    try:
        scope = eng.host.default_scope()
        _ingest_claim(eng, scope, "Standup is at nine.")
        _ingest_claim(eng, scope, "Standup moved to ten.")
        ids = list(_heads(eng))
        c1, c2 = ids[0], ids[1]
        for cid in ids:
            eng.apply_proposal(
                _proposal("admit", ((cid, 1),), f"op-ad-{cid[:8]}")
            )
        rev1 = _heads(eng)[c1][0]
        rev2 = _heads(eng)[c2][0]

        rid = propose_supersede(
            eng.store, c1, c2, actor_id="policy", reason="newer fact"
        )
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
        review = ReviewsRepo(eng.store).get(rid)
        assert review["state"] == "approved"
        assert review["resolved_event"] == receipt["committed_event"]

        # Approving the same review again replays the receipt.
        again = eng.apply_proposal(
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
        assert again["replayed"] is True
        assert _heads(eng)[c1][1] == "superseded"
    finally:
        eng.close()


def test_apply_proposal_stale_and_unknown(tmp_path):
    eng = _open(tmp_path)
    try:
        scope = eng.host.default_scope()
        _ingest_claim(eng, scope, "Pinned at revision one.")
        cid, (rev, _state) = next(iter(_heads(eng).items()))

        with pytest.raises(VerbatimError) as ei:
            eng.apply_proposal(_proposal("admit", ((cid, rev + 5),), "op-stale"))
        assert ei.value.code == ErrorCode.STALE_PROPOSAL

        with pytest.raises(VerbatimError) as ei:
            eng.apply_proposal(_proposal("admit", (("nope", 1),), "op-miss"))
        assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN
        # Nothing applied: still pending, no receipt recorded.
        assert _heads(eng)[cid][1] == "pending"
    finally:
        eng.close()


def test_apply_proposal_correct_reconsider_and_edges(tmp_path):
    eng = _open(tmp_path)
    try:
        scope = eng.host.default_scope()
        _ingest_claim(eng, scope, "Old phone number.")
        _ingest_claim(eng, scope, "New phone number.")
        c1, c2 = list(_heads(eng))
        for cid in (c1, c2):
            eng.apply_proposal(_proposal("admit", ((cid, 1),), f"op-ad-{cid[:6]}"))

        # correct: active → rejected, linking the replacement via `corrects`.
        eng.apply_proposal(
            _proposal(
                "correct", ((c1, 2),), "op-correct", successor_claim_id=c2
            )
        )
        assert _heads(eng)[c1][1] == "rejected"
        with eng.store.read() as conn:
            edge = conn.execute(
                "SELECT source_id, target_id FROM edges"
                " WHERE edge_type = 'corrects' AND retired_event IS NULL"
            ).fetchone()
        assert edge == (c2, c1)

        # reconsider: rejected → pending (re-review; not auto-activation).
        eng.apply_proposal(_proposal("reconsider", ((c1, 3),), "op-recon"))
        assert _heads(eng)[c1][1] == "pending"
    finally:
        eng.close()


def test_apply_proposal_erase_authority(tmp_path):
    eng = _open(tmp_path)
    try:
        scope = eng.host.default_scope()
        _ingest_claim(eng, scope, "Forget me.")
        cid = next(iter(_heads(eng)))
        eng.apply_proposal(_proposal("admit", ((cid, 1),), "op-ad"))

        # A transport caller holding only PURGE — no operator — is denied.
        agent = CallerContext(
            profile_id="demo",
            principal_id="me",
            conversation_id="c1",
            grants=frozenset({GrantKind.PURGE}),
        )
        with pytest.raises(VerbatimError) as ei:
            eng.apply_proposal(
                _proposal("erase", ((cid, 2),), "op-er-1", actor="agent"),
                caller=agent,
            )
        assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN
        assert _heads(eng)[cid][1] == "active"

        # A caller missing PURGE entirely is also denied.
        reader = CallerContext(
            profile_id="demo",
            principal_id="me",
            conversation_id="c1",
            grants=frozenset({GrantKind.READ_EVIDENCE}),
        )
        with pytest.raises(VerbatimError) as ei:
            eng.apply_proposal(
                _proposal("erase", ((cid, 2),), "op-er-2", actor="agent"),
                caller=reader,
            )
        assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN

        # Operator authority (trusted boundary) applies the erasure.
        receipt = eng.apply_proposal(
            _proposal("erase", ((cid, 2),), "op-er-3", actor="me")
        )
        assert receipt["targets"][0]["state"] == "erased"
        assert _heads(eng)[cid][1] == "erased"
    finally:
        eng.close()


def test_apply_proposal_caller_cannot_cross_scope(tmp_path):
    eng = _open(tmp_path)
    try:
        scope = eng.host.default_scope()
        _ingest_claim(eng, scope, "Scoped evidence.")
        cid = next(iter(_heads(eng)))

        foreign = CallerContext(
            profile_id="demo",
            principal_id="me",
            conversation_id="other-room",
            grants=frozenset(GrantKind),
        )
        with pytest.raises(VerbatimError) as ei:
            eng.apply_proposal(
                _proposal("admit", ((cid, 1),), "op-x"), caller=foreign
            )
        assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN

        other_profile = CallerContext(
            profile_id="other",
            principal_id="me",
            conversation_id="c1",
            grants=frozenset(GrantKind),
        )
        with pytest.raises(VerbatimError) as ei:
            eng.apply_proposal(
                _proposal("admit", ((cid, 1),), "op-y"), caller=other_profile
            )
        assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN
        assert _heads(eng)[cid][1] == "pending"
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# Idempotent ingest / remember (SPEC_V2 §39)
# ---------------------------------------------------------------------------


def test_ingest_replays_receipt_without_duplicates(tmp_path):
    eng = _open(tmp_path)
    try:
        scope = eng.host.default_scope()
        env = _env("retry-safe ingest", scope)
        r1 = eng.ingest(env)
        r2 = eng.ingest(env)
        assert r1.accepted == r2.accepted
        assert r1.job_ids == r2.job_ids
        assert r2.duplicate is True

        with eng.store.read() as conn:
            n_sources = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            n_jobs = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            n_ops = conn.execute("SELECT COUNT(*) FROM operations").fetchone()[0]
        assert n_sources == 1 and n_jobs == 1 and n_ops == 1

        # Explicit operation id pinned by the transport also replays.
        env2 = _env("other payload", scope, metadata={"operation_id": "ext-9"})
        r3 = eng.ingest(env2)
        env3 = _env("other payload", scope, metadata={"operation_id": "ext-9"})
        r4 = eng.ingest(env3)
        assert r3.accepted == r4.accepted and r4.duplicate is True

        # Same key, different payload → validation conflict.
        env4 = _env("CHANGED payload", scope, metadata={"operation_id": "ext-9"})
        with pytest.raises(VerbatimError) as ei:
            eng.ingest(env4)
        assert ei.value.code == ErrorCode.VALIDATION
    finally:
        eng.close()


def test_remember_idempotent_same_claim(tmp_path):
    eng = _open(tmp_path)
    try:
        scope = eng.host.default_scope()
        receipt = eng.ingest(_env("The wifi password is on the fridge.", scope))
        sid = receipt.accepted[0]
        eng.run_pending(limit=64)

        c1 = eng.remember(sid, 4, 13, scope)
        c2 = eng.remember(sid, 4, 13, scope)
        assert c1 == c2
        with eng.store.read() as conn:
            # The deterministic span persists exactly once and feeds exactly
            # one claim — the replay cannot duplicate the write.
            n_spans = conn.execute(
                "SELECT COUNT(*) FROM spans"
                " WHERE source_id = ? AND harvester_version = 'agent-remember-1'",
                (sid,),
            ).fetchone()[0]
            n_claims = conn.execute(
                "SELECT COUNT(DISTINCT claim_id) FROM claim_evidence ce"
                " JOIN spans s ON s.span_id = ce.span_id"
                " WHERE s.source_id = ? AND s.harvester_version = 'agent-remember-1'",
                (sid,),
            ).fetchone()[0]
        assert n_spans == 1 and n_claims == 1

        # A different byte range is a different operation → a different claim.
        c3 = eng.remember(sid, 0, 4, scope)
        assert c3 != c1
    finally:
        eng.close()


def test_status_reports_capabilities(tmp_path):
    eng = _open(tmp_path)
    try:
        st = eng.status()
        assert st["mode"] == "offline_rules"
        caps = st["capabilities"]
        assert caps["lexical_search"]["state"] == "healthy"
        assert caps["judge"]["details"]["backend"] == "rules"
        # No encoder is wired in offline_rules — reported, not hidden.
        assert caps["encoder"]["state"] != "healthy"
        assert caps["semantic_recall"]["state"] != "healthy"
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# CLI surface (real store, in-process argv)
# ---------------------------------------------------------------------------


def _cli(tmp_path, *argv):
    cfg = tmp_path / "verbatim.json"
    if not cfg.exists():
        cfg.write_text(
            json.dumps(
                {
                    "mode": "offline_rules",
                    "capture": {"enabled": True},
                    "admission": {"require_review": True},
                }
            ),
            encoding="utf-8",
        )
    return cli_main(
        ["--data-dir", str(tmp_path), "--config", str(cfg), "--json", *argv]
    )


def test_cli_status_and_doctor(tmp_path, capsys):
    assert _cli(tmp_path, "status") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "offline_rules"
    assert "capabilities" in out

    assert _cli(tmp_path, "doctor") == 0
    out = json.loads(capsys.readouterr().out)
    assert "quick_check" in out and "schema_version" in out


def test_cli_ingest_search_inspect_explain_spans(tmp_path, capsys):
    payload = tmp_path / "note.txt"
    payload.write_text("I keep my spare key under the blue pot.", encoding="utf-8")
    assert _cli(tmp_path, "ingest", str(payload)) == 0
    out = json.loads(capsys.readouterr().out)
    # cmd_ingest drains the queue itself: the harvest/admit jobs already ran.
    assert out["accepted"] and out["jobs_drained"] >= 1

    # jobs run is a supported command; the queue is already empty here.
    assert _cli(tmp_path, "jobs", "run") == 0
    drained = json.loads(capsys.readouterr().out)
    assert drained["jobs_drained"] >= 0
    assert _cli(tmp_path, "jobs", "list") == 0
    capsys.readouterr()

    # Pending claims stay out of search until admitted.
    assert _cli(tmp_path, "search", "spare key") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["items"] == []

    # Admit through the review the policy created.
    assert _cli(tmp_path, "reviews", "list") == 0
    reviews = json.loads(capsys.readouterr().out)
    assert reviews, "expected an open review"
    rid = reviews[0]["review_id"]

    assert _cli(tmp_path, "reviews", "approve", rid) == 2  # needs --yes
    capsys.readouterr()
    assert _cli(tmp_path, "reviews", "approve", rid, "--yes") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["state"] == "approved"
    assert out["receipt"]["targets"][0]["state"] == "active"

    assert _cli(tmp_path, "search", "spare key") == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["items"]) == 1
    cid = out["items"][0]["claim_id"]

    assert _cli(tmp_path, "inspect", cid) == 0
    detail = json.loads(capsys.readouterr().out)
    assert detail["evidence"], "inspect must cite evidence spans"

    assert _cli(tmp_path, "explain", cid) == 0
    explained = json.loads(capsys.readouterr().out)
    assert "revisions" in explained and "decisions" in explained

    assert _cli(tmp_path, "spans", cid) == 0
    spans = json.loads(capsys.readouterr().out)
    assert spans["evidence"][0]["text"].encode("utf-8") in payload.read_bytes()


def test_cli_remember(tmp_path, capsys):
    payload = tmp_path / "note.txt"
    payload.write_text("Call the dentist on Monday.", encoding="utf-8")
    assert _cli(tmp_path, "ingest", str(payload)) == 0
    out = json.loads(capsys.readouterr().out)
    sid = out["accepted"][0]

    assert _cli(tmp_path, "remember", sid, "0", "10") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["claim_id"]

    # Replay: identical remember converges on the same claim.
    assert _cli(tmp_path, "remember", sid, "0", "10") == 0
    assert json.loads(capsys.readouterr().out)["claim_id"] == out["claim_id"]


# ---------------------------------------------------------------------------
# Hermes provider caller wiring
# ---------------------------------------------------------------------------


def _provider(tmp_path, **init_kw):
    from verbatim.provider import VerbatimMemoryProvider

    home = tmp_path / "hermes"
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(
        "memory:\n  verbatim:\n    capture:\n      enabled: true\n",
        encoding="utf-8",
    )
    kw = {"hermes_home": str(home), "user_id": "alice"}
    kw.update(init_kw)
    prov = VerbatimMemoryProvider()
    prov.initialize("sess-1", **kw)
    return prov


def test_provider_builds_caller_from_host_metadata(tmp_path):
    prov = _provider(tmp_path)
    try:
        caller = prov._caller("sess-1", grants=prov._READ_GRANTS)
        assert caller.profile_id == prov._host.profile_id()
        assert caller.principal_id == "alice"
        assert caller.conversation_id == "sess-1"
        assert caller.session_id == "sess-1"
        assert caller.grants == frozenset({GrantKind.READ_EVIDENCE})
        assert caller.audience == ()
        assert not caller.is_operator

        # Audience arrives only via host metadata.
        c2 = prov._caller(
            "sess-1",
            grants=prov._READ_GRANTS,
            turn_author={"id": "alice", "participants": ["alice", "bob"]},
        )
        assert c2.audience == ("alice", "bob")
    finally:
        prov.shutdown()


def test_provider_recall_and_remember_roundtrip(tmp_path):
    prov = _provider(tmp_path)
    try:
        prov.sync_turn(
            "My anniversary is June twelfth.",
            "Noted.",
            session_id="sess-1",
        )
        prov._engine.run_pending(limit=64)
        out = json.loads(
            prov.handle_tool_call(
                "verbatim_recall", {"query": "anniversary"}, session_id="sess-1"
            )
        )
        assert out["ok"] is True

        # remember through the tool, then recall after admission.
        src = None
        with prov._engine.store.read() as conn:
            src = conn.execute("SELECT source_id FROM sources LIMIT 1").fetchone()[0]
        out = json.loads(
            prov.handle_tool_call(
                "verbatim_evidence",
                {"action": "remember", "source_id": src, "start_byte": 0, "end_byte": 15},
                session_id="sess-1",
            )
        )
        assert out["ok"] is True and out["data"]["claim_id"]
    finally:
        prov.shutdown()


def test_provider_caller_denied_foreign_scope(tmp_path):
    """A session-bound caller cannot reach another conversation's partition."""
    prov = _provider(tmp_path)
    try:
        prov.sync_turn("secret handshake", "ok", session_id="sess-1")
        prov._engine.run_pending(limit=64)
        caller = prov._caller("other-session", grants=prov._READ_GRANTS)
        scope_b = prov._session_scope("other-session")
        res = prov._engine.recall(
            RecallRequest(query="handshake", scope=scope_b), caller=caller
        )
        assert res.items == ()
        # Even an explicit ask for the original conversation is denied:
        # the caller's home partition is bound to other-session.
        foreign = prov._session_scope("sess-1")
        with pytest.raises(VerbatimError) as ei:
            prov._engine.recall(
                RecallRequest(query="handshake", scope=foreign), caller=caller
            )
        assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN
    finally:
        prov.shutdown()
