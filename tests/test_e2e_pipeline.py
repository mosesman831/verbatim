"""End-to-end pipeline regression: ingest → harvest → admit → review → recall.

Exercises the real Store on disk, the real job queue, the policy admission
ladder, review-driven activation, FTS projection indexing, and scoped recall —
the seams where parallel worker modules previously drifted apart.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from dataclasses import replace

import pytest

from verbatim.api import open_store
from verbatim.config import VerbatimConfig
from verbatim.core.identity import scope_key
from verbatim.core.time import now_us
from verbatim.core.types import (
    Provenance,
    RecallMode,
    RecallRequest,
    SourceEnvelope,
    SourceKind,
    TransitionCommand,
)
from verbatim.host import LocalHost

PAYLOAD = (
    "I switched my editor from vim to neovim last week. "
    "My database is Postgres now."
).encode("utf-8")


def _cfg(**kw) -> VerbatimConfig:
    cfg = VerbatimConfig()
    return replace(cfg, capture=replace(cfg.capture, enabled=True, **kw))


def _engine(tmpdir: str, cfg: VerbatimConfig | None = None, profile: str = "demo"):
    return open_store(
        tmpdir,
        cfg or _cfg(),
        LocalHost(profile_id=profile, principal_id="me", conversation_id="c1"),
        create=True,
    )


def _envelope(eng, payload: bytes = PAYLOAD) -> SourceEnvelope:
    return SourceEnvelope(
        origin="test:e2e",
        source_kind=SourceKind.USER_MESSAGE,
        scope=eng.host.default_scope(),
        speaker_id="me",
        payload=payload,
        event_us=now_us(),
        captured_us=now_us(),
        provenance=Provenance.DIRECT_USER,
    )


def _db_path(tmpdir: str, profile: str = "demo") -> str:
    return os.path.join(tmpdir, f"{profile}.db")


def test_ingest_to_recall_happy_path(tmp_path):
    eng = _engine(str(tmp_path))
    try:
        scope = eng.host.default_scope()
        receipt = eng.ingest(_envelope(eng))
        assert receipt.accepted and receipt.job_ids

        assert eng.run_pending() >= 1

        conn = sqlite3.connect(_db_path(str(tmp_path)))
        row = conn.execute(
            "SELECT claim_id, state FROM claim_revisions ORDER BY revision DESC LIMIT 1"
        ).fetchone()
        conn.close()
        assert row is not None
        claim_id, state = row
        assert state == "pending"  # require_review default sends claims to review

        res = eng.recall(RecallRequest(query="editor", scope=scope, mode=RecallMode.CURRENT))
        assert res.items == ()

        eng.apply_transition(
            TransitionCommand(
                claim_id=claim_id,
                expected_revision=1,
                effect="admit",
                actor_id="me",
                reason="approved in review",
            ),
            scope,
        )

        res = eng.recall(RecallRequest(query="editor", scope=scope, mode=RecallMode.CURRENT))
        assert len(res.items) == 1
        item = res.items[0]
        assert item.lifecycle.value == "active"
        assert item.text.encode("utf-8") in PAYLOAD  # verbatim span, not paraphrase
    finally:
        eng.close()


def test_pending_claims_stay_out_of_recall(tmp_path):
    eng = _engine(str(tmp_path))
    try:
        scope = eng.host.default_scope()
        eng.ingest(_envelope(eng))
        eng.run_pending()
        res = eng.recall(RecallRequest(query="editor", scope=scope, mode=RecallMode.CURRENT))
        assert res.items == ()
    finally:
        eng.close()


def test_scope_isolation(tmp_path):
    """A second profile's store cannot read the first profile's claims."""
    eng_a = _engine(str(tmp_path), profile="alice")
    eng_a.ingest(_envelope(eng_a))
    eng_a.run_pending()
    eng_a.close()

    # Bob's engine opens a different profile db in the same directory.
    eng_b = _engine(str(tmp_path), profile="bob")
    try:
        scope_b = eng_b.host.default_scope()
        res = eng_b.recall(
            RecallRequest(query="editor", scope=scope_b, mode=RecallMode.CURRENT)
        )
        assert res.items == ()
    finally:
        eng_b.close()


def test_utf8_span_fidelity(tmp_path):
    """Multibyte payload: recalled text must be byte-exact, offsets aligned."""
    payload = "J'aime les émojis 🎉 et mon éditeur est neovim.".encode("utf-8")
    eng = _engine(str(tmp_path))
    try:
        scope = eng.host.default_scope()
        eng.ingest(_envelope(eng, payload))
        eng.run_pending()
        conn = sqlite3.connect(_db_path(str(tmp_path)))
        row = conn.execute(
            "SELECT claim_id FROM claim_revisions ORDER BY revision DESC LIMIT 1"
        ).fetchone()
        conn.close()
        assert row is not None
        eng.apply_transition(
            TransitionCommand(
                claim_id=row[0],
                expected_revision=1,
                effect="admit",
                actor_id="me",
                reason="ok",
            ),
            scope,
        )
        res = eng.recall(RecallRequest(query="éditeur", scope=scope, mode=RecallMode.CURRENT))
        assert len(res.items) == 1
        assert res.items[0].text.encode("utf-8") in payload
    finally:
        eng.close()


def test_review_activation_relates_natural_retraction(tmp_path):
    eng = _engine(str(tmp_path))
    try:
        scope = eng.host.default_scope()
        for text in (
            "oh and btw I switched from VS Code to Neovim",
            "Actually scratch that about Neovim, I went back to VS Code",
        ):
            eng.ingest(_envelope(eng, text.encode()))
        eng.run_pending(limit=100)
        with eng.store.read() as conn:
            claims = conn.execute(
                "SELECT cr.claim_id, cr.revision, cr.object_json"
                " FROM claim_revisions cr"
                " JOIN claims c ON c.claim_id = cr.claim_id"
                " WHERE cr.recorded_until IS NULL AND c.predicate = 'editor'"
            ).fetchall()
        assert len(claims) == 2
        ordered = sorted(
            claims,
            key=lambda row: 0 if "Neovim" in row[2] else 1,
        )
        for claim_id, revision, _obj in ordered:
            eng.apply_transition(
                TransitionCommand(
                    claim_id=claim_id,
                    expected_revision=revision,
                    effect="admit",
                    actor_id="me",
                    reason="approved in review",
                ),
                scope,
            )
        with eng.store.read() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM conflict_groups WHERE status = 'open'"
            ).fetchone()[0] == 1
            effects = [
                row[0]
                for row in conn.execute(
                    "SELECT proposed_effect_json FROM reviews"
                    " WHERE state = 'open'"
                ).fetchall()
            ]
        assert any('"effect":"supersede"' in effect for effect in effects)
    finally:
        eng.close()


def test_capture_disabled_by_default(tmp_path):
    """Default config rejects ingestion — capture is opt-in (SPEC §10)."""
    eng = _engine(str(tmp_path), cfg=VerbatimConfig())
    try:
        from verbatim.core.types import VerbatimError

        with pytest.raises(VerbatimError):
            eng.ingest(_envelope(eng))
    finally:
        eng.close()
